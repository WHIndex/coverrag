#!/usr/bin/env python3
"""COVER-RAG v1: revise ALCE answers using verified atomic claims.

v0 audits generated answers without changing them. v1 uses the same atomic
claim decomposition and evidence verification machinery, then rewrites the
answer using only claims that are supported by evidence already available in
the ALCE result file.

Input:
    ALCE-style result JSON produced by run.py / cover_run.py

Output:
    ALCE-compatible result JSON whose `output` field is replaced by a revised,
    citation-constrained answer. The original answer and all intermediate
    verification metadata are preserved under `cover_v1`.

Recommended first real run:

    python scripts/cover_v1_revise.py \
      --input result/asqa-...json \
      --output result/asqa-...cover_v1.json \
      --openai-api \
      --decomposer llm \
      --decompose-model gpt-4o-mini \
      --verifier llm \
      --llm-verify-model gpt-4o-mini \
      --revision-model gpt-4o-mini \
      --evidence-scope cited_then_all \
      --cache-file cache/cover_v1_cache.json \
      --pretty
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import re
import statistics
from pathlib import Path
from typing import Any, Sequence

from cover_posthoc_audit import (
    CLAIM_DECOMPOSITION_SYSTEM,
    JsonCache,
    OpenAIChatClient,
    audit_item,
    format_doc,
    infer_answer_format,
    load_payload,
    make_decomposer,
    make_verifier,
    normalize_space,
    parse_citations,
    remove_citations,
    safe_float,
    stable_hash,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("cover_v1_revise")


REVISION_SYSTEM = """You are a careful editor for citation-grounded question answering.
You must write the final answer using only the verified claims provided by the system.
Do not add outside knowledge. Do not add new factual claims."""


REVISION_USER = """Question:
{question}

Original draft answer:
{draft_answer}

Verified claims you may use:
{verified_claims}

Claims you must NOT use because they were unsupported or contradicted:
{rejected_claims}

Instructions:
1. Write one concise paragraph.
2. Use only the verified claims. Do not introduce facts that are not in the verified-claims list.
3. Every factual sentence must end with one to three citations, such as [1] or [1][2].
4. Use the citation ids attached to each verified claim. Do not invent citation ids.
5. Merge claims naturally when possible, but keep all citations faithful.
6. If the verified claims are insufficient to answer the question, say that the provided evidence is insufficient and cite the closest supporting evidence if available.
7. Do not use bullet points, markdown, headings, or line breaks.

Final answer:"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COVER-RAG v1 answer revision for ALCE result JSON.")
    parser.add_argument("--input", required=True, type=Path, help="ALCE result JSON with draft outputs.")
    parser.add_argument("--output", required=True, type=Path, help="Where to write revised ALCE result JSON.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N examples.")
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset name. Used to infer answer formatting, especially qampari.",
    )
    parser.add_argument(
        "--answer-format",
        choices=["auto", "paragraph", "qampari"],
        default="auto",
        help=(
            "How to split and render answers. `qampari` preserves comma-separated entity lists; "
            "`paragraph` uses sentence-like answers."
        ),
    )

    # Fields and ALCE structure.
    parser.add_argument("--output-field", default="output", help="Field containing generated answer.")
    parser.add_argument("--docs-field", default="docs", help="Field containing docs.")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--title-field", default="title")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--sent-field", default="sent")

    # LLM / decomposition / verification settings. These mirror v0 so the two
    # scripts can be compared fairly.
    parser.add_argument("--openai-api", action="store_true", help="Use OpenAI-compatible chat API.")
    parser.add_argument("--decomposer", choices=["llm", "sentence", "claimify"], default="llm")
    parser.add_argument("--decompose-model", default="gpt-4o-mini")
    parser.add_argument(
        "--claimify-context-before",
        type=int,
        default=3,
        help="Number of previous answer sentences used by Claimify-style decomposition.",
    )
    parser.add_argument(
        "--claimify-context-after",
        type=int,
        default=2,
        help="Number of following answer sentences used by Claimify-style decomposition.",
    )
    parser.add_argument(
        "--claimify-max-stage-retries",
        type=int,
        default=1,
        help="Extra retries for each Claimify JSON stage after the initial attempt.",
    )
    parser.add_argument(
        "--claimify-selection-completions",
        type=int,
        default=1,
        help="Number of LLM completions for Claimify selection voting. Use 3 for a stricter paper-style run.",
    )
    parser.add_argument(
        "--claimify-selection-min-successes",
        type=int,
        default=1,
        help="Minimum selection completions that must find verifiable content.",
    )
    parser.add_argument(
        "--claimify-disambiguation-completions",
        type=int,
        default=1,
        help="Number of LLM completions for Claimify disambiguation voting. Use 3 for a stricter paper-style run.",
    )
    parser.add_argument(
        "--claimify-disambiguation-min-successes",
        type=int,
        default=1,
        help="Minimum disambiguation completions that must resolve the sentence.",
    )
    parser.add_argument(
        "--claimify-decomposition-completions",
        type=int,
        default=1,
        help="Number of LLM completions for Claimify decomposition voting.",
    )
    parser.add_argument(
        "--claimify-decomposition-min-successes",
        type=int,
        default=1,
        help="Minimum decomposition completions that must produce the same normalized claim.",
    )
    parser.add_argument(
        "--claimify-implementation",
        choices=["local", "external"],
        default="local",
        help=(
            "`local` uses COVER-RAG's built-in Claimify-compatible extractor. "
            "`external` imports a user-provided deshwalmahesh/claimify source tree."
        ),
    )
    parser.add_argument(
        "--claimify-external-path",
        type=Path,
        default=None,
        help="Path to a local external Claimify repo root, src directory, or claimify.py file.",
    )
    parser.add_argument(
        "--claimify-use-external-defaults",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "In external mode, keep the external repository's own Claimify hyperparameters. "
            "Use --no-claimify-use-external-defaults to force COVER-RAG's claimify-* args."
        ),
    )
    parser.add_argument("--verifier", choices=["nli", "llm", "lexical", "hybrid"], default="llm")
    parser.add_argument("--llm-verify-model", default="gpt-4o-mini")
    parser.add_argument("--nli-model", default="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")
    parser.add_argument("--device", default=None)
    parser.add_argument("--nli-batch-size", type=int, default=8)
    parser.add_argument("--nli-max-length", type=int, default=512)
    parser.add_argument("--entail-threshold", type=float, default=0.50)
    parser.add_argument("--contradiction-threshold", type=float, default=0.50)
    parser.add_argument("--ambiguous-margin", type=float, default=0.10)
    parser.add_argument("--nli-aggregation", choices=["joint", "max_doc", "joint_then_max"], default="joint_then_max")
    parser.add_argument(
        "--evidence-scope",
        choices=["cited", "all_docs", "cited_then_all"],
        default="cited_then_all",
        help=(
            "For v1, `cited_then_all` is recommended: first verify cited evidence; "
            "if it fails, check whether another already-retrieved doc supports the claim."
        ),
    )
    parser.add_argument("--max-all-docs", type=int, default=20)
    parser.add_argument("--max-evidence-chars", type=int, default=6000)
    parser.add_argument("--lexical-threshold", type=float, default=0.45)

    # Revision generation.
    parser.add_argument(
        "--revision-mode",
        choices=["llm", "template", "atomic", "qampari", "auto"],
        default="auto",
        help=(
            "`atomic` writes one verified claim per sentence; `qampari` writes a comma-separated "
            "entity list; `auto` uses qampari mode for QAMPARI and atomic mode otherwise. "
            "This is the recommended v1 setting because it preserves claim-citation alignment."
        ),
    )
    parser.add_argument("--revision-model", default="gpt-4o-mini")
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-top-p", type=float, default=1.0)
    parser.add_argument("--llm-max-retries", type=int, default=5)
    parser.add_argument("--revision-max-tokens", type=int, default=300)
    parser.add_argument("--max-verified-claims", type=int, default=16)
    parser.add_argument(
        "--max-qampari-items",
        type=int,
        default=80,
        help="Maximum verified QAMPARI answer items to keep. QAMPARI needs a much larger list than ASQA.",
    )
    parser.add_argument("--max-rejected-claims", type=int, default=12)
    parser.add_argument(
        "--citation-policy",
        choices=["minimal", "all", "source"],
        default="source",
        help=(
            "How to choose final citations for each claim. `minimal` picks the most claim-relevant "
            "document ids; `all` keeps verifier ids; `source` keeps the original source sentence ids."
        ),
    )
    parser.add_argument(
        "--max-citations-per-claim",
        type=int,
        default=3,
        help="Maximum citations attached to each final claim/item.",
    )
    parser.add_argument(
        "--allow-citation-repair",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use claims labeled wrong_or_missing_citation with repaired evidence ids.",
    )
    parser.add_argument(
        "--include-background-claims",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow verified background claims in the final answer.",
    )
    parser.add_argument(
        "--abstain-if-no-supported",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write an evidence-insufficient answer if no usable verified claims remain.",
    )
    parser.add_argument(
        "--final-audit",
        action="store_true",
        help="Run v0 audit again on revised outputs. More reliable but slower/costlier.",
    )
    parser.add_argument(
        "--fallback-template-on-empty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic template output if LLM revision is empty or citation-invalid.",
    )

    parser.add_argument("--cache-file", type=Path, default=None, help="JSON cache for LLM calls.")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


class RevisionGenerator:
    def revise(
        self,
        item: dict[str, Any],
        verified_claims: Sequence[dict[str, Any]],
        rejected_claims: Sequence[dict[str, Any]],
        args: argparse.Namespace,
    ) -> str:
        raise NotImplementedError


class TemplateRevisionGenerator(RevisionGenerator):
    def revise(
        self,
        item: dict[str, Any],
        verified_claims: Sequence[dict[str, Any]],
        rejected_claims: Sequence[dict[str, Any]],
        args: argparse.Namespace,
    ) -> str:
        if not verified_claims:
            return evidence_insufficient_answer(args)

        sentences = []
        for claim in verified_claims[: args.max_verified_claims]:
            citations = format_citations(claim.get("final_citation_ids", []))
            if citations:
                sentences.append(f"{claim['claim'].rstrip('.')} {citations}.")
        if not sentences:
            return evidence_insufficient_answer(args)
        return clean_final_answer(" ".join(sentences), num_docs=len(item.get(args.docs_field, []) or []))


class AtomicSentenceRevisionGenerator(RevisionGenerator):
    """Deterministic renderer: one verified atomic claim per sentence.

    This is less fluent than LLM rewriting, but it is much friendlier to
    sentence-level citation evaluators because each sentence has a minimal,
    claim-specific citation set.
    """

    def revise(
        self,
        item: dict[str, Any],
        verified_claims: Sequence[dict[str, Any]],
        rejected_claims: Sequence[dict[str, Any]],
        args: argparse.Namespace,
    ) -> str:
        if not verified_claims:
            return evidence_insufficient_answer(args)

        sentences = []
        for claim in verified_claims[: verified_claim_limit(args)]:
            citations = format_citations(claim.get("final_citation_ids", []), max_citations=args.max_citations_per_claim)
            if not citations:
                continue
            claim_text = normalize_generated_claim_sentence(str(claim.get("claim", "")))
            if claim_text:
                sentences.append(f"{claim_text} {citations}.")
        if not sentences:
            return evidence_insufficient_answer(args)
        return clean_final_answer(" ".join(sentences), num_docs=len(item.get(args.docs_field, []) or []))


class QampariListRevisionGenerator(RevisionGenerator):
    """Deterministic renderer for QAMPARI entity-list answers."""

    def revise(
        self,
        item: dict[str, Any],
        verified_claims: Sequence[dict[str, Any]],
        rejected_claims: Sequence[dict[str, Any]],
        args: argparse.Namespace,
    ) -> str:
        if not verified_claims:
            return evidence_insufficient_answer(args)

        parts = []
        seen = set()
        for claim in verified_claims[: verified_claim_limit(args)]:
            entity = extract_qampari_answer_item(claim)
            key = normalize_qampari_item_key(entity)
            if not entity or not key or key in seen:
                continue
            citations = format_citations(claim.get("final_citation_ids", []), max_citations=args.max_citations_per_claim)
            if not citations:
                continue
            seen.add(key)
            parts.append(f"{entity} {citations}")

        if not parts:
            return evidence_insufficient_answer(args)
        answer = ", ".join(parts).rstrip(", ")
        return clean_final_answer(answer + ".", num_docs=len(item.get(args.docs_field, []) or []))


class AutoRevisionGenerator(RevisionGenerator):
    def __init__(self) -> None:
        self.atomic = AtomicSentenceRevisionGenerator()
        self.qampari = QampariListRevisionGenerator()

    def revise(
        self,
        item: dict[str, Any],
        verified_claims: Sequence[dict[str, Any]],
        rejected_claims: Sequence[dict[str, Any]],
        args: argparse.Namespace,
    ) -> str:
        if infer_answer_format(args) == "qampari":
            return self.qampari.revise(item, verified_claims, rejected_claims, args)
        return self.atomic.revise(item, verified_claims, rejected_claims, args)


class LLMRevisionGenerator(RevisionGenerator):
    def __init__(self, client: OpenAIChatClient, cache: JsonCache) -> None:
        self.client = client
        self.cache = cache
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def revise(
        self,
        item: dict[str, Any],
        verified_claims: Sequence[dict[str, Any]],
        rejected_claims: Sequence[dict[str, Any]],
        args: argparse.Namespace,
    ) -> str:
        if not verified_claims and args.abstain_if_no_supported:
            return evidence_insufficient_answer(args)

        question = str(item.get(args.question_field, ""))
        draft_answer = str(item.get(args.output_field, "") or "")
        verified_block = render_verified_claims(verified_claims, item, args)
        rejected_block = render_rejected_claims(rejected_claims, args)
        key = "v1-revise:" + stable_hash(question, draft_answer, verified_block, rejected_block)
        cached = self.cache.get(key)
        if cached is not None:
            return str(cached)

        user_prompt = REVISION_USER.format(
            question=question,
            draft_answer=draft_answer,
            verified_claims=verified_block or "None.",
            rejected_claims=rejected_block or "None.",
        )
        raw, usage = self.client.chat(REVISION_SYSTEM, user_prompt, max_tokens=args.revision_max_tokens)
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        answer = clean_final_answer(raw, num_docs=len(item.get(args.docs_field, []) or []))
        self.cache.set(key, answer)
        return answer


def evidence_insufficient_answer(args: argparse.Namespace) -> str:
    # No citation is added here because there is no verified supporting claim.
    # ALCE will penalize this, which is appropriate for an abstention case.
    return "The provided evidence is insufficient to answer the question."


def format_citations(ids: Sequence[int], max_citations: int = 3) -> str:
    unique_ids = []
    for doc_id in ids:
        try:
            value = int(doc_id)
        except Exception:
            continue
        if value > 0 and value not in unique_ids:
            unique_ids.append(value)
    return "".join(f"[{doc_id}]" for doc_id in unique_ids[: max(1, max_citations)])


def normalize_generated_claim_sentence(text: str) -> str:
    text = remove_citations(text)
    text = normalize_space(text).strip()
    text = text.rstrip(".;:,")
    return text


def extract_qampari_answer_item(claim: dict[str, Any]) -> str:
    # For QAMPARI, the source answer item is the entity span. The decomposed
    # claim may be a full proposition, e.g. "Harmony Korine directed Gummo";
    # using the source item preserves the entity-list format expected by F1.
    #
    # v3 expansion claims are different: their source sentence is the evidence
    # sentence, not the answer item. Prefer the explicit answer span so QAMPARI
    # output stays a comma-separated entity list instead of leaking full
    # evidence sentences into the answer.
    candidates = []
    for key in ("v3_answer_span", "answer_span"):
        value = str(claim.get(key, "") or "").strip()
        if value:
            candidates.append(value)
    for span in claim.get("protected_answer_spans", []) or []:
        value = str(span or "").strip()
        if value:
            candidates.append(value)

    source = str(claim.get("source_sentence_without_citations") or "").strip()
    if source:
        # Original QAMPARI answer items are usually short. Expanded evidence
        # sentences are long and should not become list items.
        source_clean = normalize_space(remove_citations(source)).strip().strip('"').strip("'")
        if len(source_clean.split()) <= 10 and not re.search(r"[.!?;:]", source_clean):
            candidates.append(source_clean)

    candidates.append(str(claim.get("claim", "") or ""))

    item = ""
    for candidate in candidates:
        candidate = remove_citations(candidate)
        candidate = re.sub(r"^\s*[-*\d.)]+\s*", "", candidate)
        candidate = normalize_space(candidate).strip().strip('"').strip("'")
        candidate = candidate.rstrip(".;:,")
        if not candidate:
            continue
        # Avoid returning full explanatory clauses as QAMPARI answer items.
        if len(candidate.split()) > 12:
            continue
        item = candidate
        break
    if not item:
        item = remove_citations(str(claim.get("claim", "") or ""))
    item = remove_citations(item)
    item = re.sub(r"^\s*[-*\d.)]+\s*", "", item)
    item = normalize_space(item).strip().strip('"').strip("'")
    item = item.rstrip(".;:,")
    return item


def normalize_qampari_item_key(text: str) -> str:
    text = remove_citations(text).lower()
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return normalize_space(text)


def clean_final_answer(text: str, num_docs: int) -> str:
    text = str(text or "").replace("<|im_end|>", " ").strip()
    text = re.sub(r"^```(?:text)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    text = re.sub(r"\s*\n+\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = remove_invalid_citations(text, num_docs)
    text = normalize_citation_spacing(text)
    return text


def remove_invalid_citations(text: str, num_docs: int) -> str:
    def replace(match: re.Match[str]) -> str:
        citation_id = int(match.group(1))
        if 1 <= citation_id <= num_docs:
            return f"[{citation_id}]"
        return ""

    return re.sub(r"\[(\d+)\]", replace, text)


def normalize_citation_spacing(text: str) -> str:
    # Keep ALCE-friendly bracket citations while preserving readable spacing:
    # "claim. [1]" -> "claim [1]."; "[1] [2]" -> "[1][2]".
    text = re.sub(r"\s*(\[\d+\])", r" \1", text)
    text = re.sub(r"(\[\d+\])\s+(\[\d+\])", r"\1\2", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def render_verified_claims(
    verified_claims: Sequence[dict[str, Any]],
    item: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    lines = []
    docs = item.get(args.docs_field, []) or []
    for idx, claim in enumerate(verified_claims[: args.max_verified_claims], start=1):
        citations = format_citations(claim.get("final_citation_ids", []), max_citations=args.max_citations_per_claim)
        evidence_bits = []
        for doc_id in claim.get("final_citation_ids", [])[:3]:
            doc_idx = int(doc_id) - 1
            if 0 <= doc_idx < len(docs):
                evidence = format_doc(docs[doc_idx], args)
                evidence_bits.append(f"[{doc_id}] {truncate(evidence, 700)}")
        evidence_text = " | ".join(evidence_bits)
        lines.append(
            f"{idx}. Claim: {claim['claim']}\n"
            f"   Required citations: {citations}\n"
            f"   Importance: {claim.get('importance', 'supporting')}\n"
            f"   Evidence: {evidence_text}"
        )
    return "\n".join(lines)


def render_rejected_claims(rejected_claims: Sequence[dict[str, Any]], args: argparse.Namespace) -> str:
    lines = []
    for idx, claim in enumerate(rejected_claims[: args.max_rejected_claims], start=1):
        lines.append(f"{idx}. {claim.get('claim', '')} ({claim.get('label', 'unknown')})")
    return "\n".join(lines)


def truncate(text: str, max_chars: int) -> str:
    text = normalize_space(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def make_revision_generator(args: argparse.Namespace, cache: JsonCache) -> RevisionGenerator:
    if args.revision_mode == "template":
        return TemplateRevisionGenerator()
    if args.revision_mode == "atomic":
        return AtomicSentenceRevisionGenerator()
    if args.revision_mode == "qampari":
        return QampariListRevisionGenerator()
    if args.revision_mode == "auto":
        return AutoRevisionGenerator()
    if not args.openai_api:
        raise ValueError("`--revision-mode llm` requires `--openai-api`.")
    client = OpenAIChatClient(
        model=args.revision_model,
        temperature=args.llm_temperature,
        top_p=args.llm_top_p,
        max_retries=args.llm_max_retries,
    )
    return LLMRevisionGenerator(client, cache)


def fill_dataset_name_from_payload(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    """Infer dataset name from ALCE metadata when the user did not pass it.

    ALCE result files usually store the original run config under payload["args"].
    If we miss this and the file path does not contain "qampari", QAMPARI gets
    treated as a paragraph answer and v1 destroys list-style F1.
    """

    if getattr(args, "dataset_name", None):
        return
    payload_args = payload.get("args", {}) if isinstance(payload, dict) else {}
    if not isinstance(payload_args, dict):
        return
    for key in ("dataset_name", "dataset", "data_name"):
        value = payload_args.get(key)
        if value:
            args.dataset_name = str(value)
            logger.info("Inferred dataset_name=%s from result metadata.", args.dataset_name)
            return


def claim_is_usable(claim: dict[str, Any], args: argparse.Namespace) -> bool:
    label = claim.get("label")
    if label == "supported":
        pass
    elif label == "wrong_or_missing_citation" and args.allow_citation_repair:
        pass
    else:
        return False

    if not args.include_background_claims and claim.get("importance") == "background":
        return False
    return bool(claim.get("evidence_doc_ids"))


def prepare_claims_for_revision(
    audit: dict[str, Any],
    args: argparse.Namespace,
    item: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_claims = list(audit.get("claims", []) or [])
    usable = []
    rejected = []
    seen = set()
    docs = (item or {}).get(args.docs_field, []) or []

    for claim in raw_claims:
        normalized_key = normalize_revision_claim_key(claim, args)
        if not normalized_key or normalized_key in seen:
            continue
        seen.add(normalized_key)

        if claim_is_usable(claim, args):
            updated = dict(claim)
            updated["final_citation_ids"] = choose_final_citations(claim, args, docs)
            if updated["final_citation_ids"]:
                usable.append(updated)
            else:
                rejected.append(dict(claim))
        else:
            rejected.append(dict(claim))

    usable.sort(key=claim_priority_key)
    return usable[: verified_claim_limit(args)], rejected


def verified_claim_limit(args: argparse.Namespace) -> int:
    if infer_answer_format(args) == "qampari":
        return max(int(args.max_verified_claims), int(args.max_qampari_items))
    return int(args.max_verified_claims)


def normalize_revision_claim_key(claim: dict[str, Any], args: argparse.Namespace) -> str:
    if infer_answer_format(args) == "qampari":
        return normalize_qampari_item_key(extract_qampari_answer_item(claim))
    return normalize_claim_key(str(claim.get("claim", "")))


def normalize_claim_key(claim: str) -> str:
    claim = remove_citations(claim).lower()
    claim = re.sub(r"[^a-z0-9]+", " ", claim)
    return normalize_space(claim)


def choose_final_citations(claim: dict[str, Any], args: argparse.Namespace, docs: Sequence[dict[str, Any]]) -> list[int]:
    source_ids = unique_positive_ints(claim.get("citation_ids") or [])
    evidence_ids = unique_positive_ints(claim.get("evidence_doc_ids") or [])

    if args.citation_policy == "source":
        ids = source_ids
    elif args.citation_policy == "all":
        ids = evidence_ids or source_ids
    else:
        ids = rank_candidate_citations(claim, evidence_ids or source_ids, source_ids, docs, args)

    final = []
    for doc_id in ids:
        try:
            value = int(doc_id)
        except Exception:
            continue
        if value > 0 and value not in final:
            final.append(value)
    return final[: max(1, int(args.max_citations_per_claim))]


def unique_positive_ints(values: Sequence[Any]) -> list[int]:
    output = []
    for value in values:
        try:
            doc_id = int(value)
        except Exception:
            continue
        if doc_id > 0 and doc_id not in output:
            output.append(doc_id)
    return output


def rank_candidate_citations(
    claim: dict[str, Any],
    candidate_ids: Sequence[int],
    source_ids: Sequence[int],
    docs: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> list[int]:
    candidates = unique_positive_ints(candidate_ids)
    if not candidates:
        return []
    if len(candidates) == 1 or not docs:
        return candidates

    claim_text = str(claim.get("claim", "") or "")
    if infer_answer_format(args) == "qampari":
        claim_text = f"{extract_qampari_answer_item(claim)} {claim_text}"

    scored = []
    for doc_id in candidates:
        doc_idx = doc_id - 1
        if not (0 <= doc_idx < len(docs)):
            continue
        evidence = format_doc(docs[doc_idx], args)
        score = lexical_relevance_score(claim_text, evidence)
        if doc_id in source_ids:
            score += 0.05
        scored.append((score, doc_id))

    if not scored:
        return candidates
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [doc_id for _, doc_id in scored]


def lexical_relevance_score(query: str, document: str) -> float:
    query_tokens = content_tokens(query)
    if not query_tokens:
        return 0.0
    doc_tokens = content_tokens(document)
    if not doc_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / len(query_tokens)


def content_tokens(text: str) -> set[str]:
    stopwords = {
        "the",
        "a",
        "an",
        "of",
        "to",
        "and",
        "or",
        "in",
        "on",
        "for",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "that",
        "this",
        "it",
        "as",
        "at",
        "from",
        "what",
        "which",
        "who",
    }
    tokens = re.findall(r"[a-z0-9]+", remove_citations(text).lower())
    return {token for token in tokens if token not in stopwords and len(token) > 1}


def claim_priority_key(claim: dict[str, Any]) -> tuple[int, int, int]:
    importance_rank = {"critical": 0, "supporting": 1, "background": 2}
    label_rank = {"supported": 0, "wrong_or_missing_citation": 1}
    return (
        importance_rank.get(str(claim.get("importance", "supporting")), 1),
        label_rank.get(str(claim.get("label", "supported")), 0),
        int(claim.get("source_sentence_id", 0)),
    )


def output_has_valid_citation(answer: str, num_docs: int) -> bool:
    citation_ids = parse_citations(answer)
    return any(1 <= citation_id <= num_docs for citation_id in citation_ids)


def revise_item(
    item: dict[str, Any],
    item_id: int,
    decomposer: Any,
    verifier: Any,
    revision_generator: RevisionGenerator,
    args: argparse.Namespace,
) -> dict[str, Any]:
    updated = dict(item)
    draft_output = str(updated.get(args.output_field, "") or "")
    draft_audit = audit_item(updated, item_id, decomposer, verifier, args)
    verified_claims, rejected_claims = prepare_claims_for_revision(draft_audit, args, updated)

    revised_output = revision_generator.revise(updated, verified_claims, rejected_claims, args)
    if (
        args.fallback_template_on_empty
        and verified_claims
        and (not revised_output or not output_has_valid_citation(revised_output, len(updated.get(args.docs_field, []) or [])))
    ):
        logger.warning("Falling back to template revision for item %s because LLM output was empty or citation-invalid.", item_id)
        revised_output = AutoRevisionGenerator().revise(updated, verified_claims, rejected_claims, args)

    updated["cover_v1"] = {
        "original_output": draft_output,
        "draft_audit": draft_audit,
        "verified_claims": verified_claims,
        "rejected_claims": rejected_claims,
        "revision_metrics": compute_revision_metrics(draft_audit, verified_claims, rejected_claims, revised_output),
    }
    updated[args.output_field] = revised_output

    if args.final_audit:
        updated["cover_v1"]["final_audit"] = audit_item(updated, item_id, decomposer, verifier, args)
    return updated


def compute_revision_metrics(
    draft_audit: dict[str, Any],
    verified_claims: Sequence[dict[str, Any]],
    rejected_claims: Sequence[dict[str, Any]],
    revised_output: str,
) -> dict[str, Any]:
    draft_claims = draft_audit.get("claims", []) or []
    return {
        "draft_num_claims": len(draft_claims),
        "kept_verified_claims": len(verified_claims),
        "rejected_claims": len(rejected_claims),
        "claim_keep_rate": len(verified_claims) / len(draft_claims) if draft_claims else 0.0,
        "revised_length": len((revised_output or "").split()),
        "revised_num_citations": len(parse_citations(revised_output)),
    }


def summarize_v1(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    records = [item.get("cover_v1", {}) for item in items if item.get("cover_v1")]
    metrics = [record.get("revision_metrics", {}) for record in records]

    total_draft_claims = sum(int(m.get("draft_num_claims", 0)) for m in metrics)
    total_kept = sum(int(m.get("kept_verified_claims", 0)) for m in metrics)
    total_rejected = sum(int(m.get("rejected_claims", 0)) for m in metrics)
    lengths = [float(m.get("revised_length", 0.0)) for m in metrics]
    citations = [float(m.get("revised_num_citations", 0.0)) for m in metrics]

    label_counter: collections.Counter[str] = collections.Counter()
    for record in records:
        for claim in record.get("draft_audit", {}).get("claims", []) or []:
            label_counter[str(claim.get("label", "unknown"))] += 1

    return {
        "num_examples": len(records),
        "draft_num_claims": total_draft_claims,
        "kept_verified_claims": total_kept,
        "rejected_claims": total_rejected,
        "claim_keep_rate_micro": total_kept / total_draft_claims if total_draft_claims else 0.0,
        "rejected_claim_rate_micro": total_rejected / total_draft_claims if total_draft_claims else 0.0,
        "avg_revised_length": statistics.mean(lengths) if lengths else 0.0,
        "avg_revised_num_citations": statistics.mean(citations) if citations else 0.0,
        "draft_label_counts": dict(label_counter),
    }


def summarize_final_audit(items: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    final_audits = [item.get("cover_v1", {}).get("final_audit") for item in items]
    final_audits = [audit for audit in final_audits if audit]
    if not final_audits:
        return None
    total_claims = sum(int(audit.get("metrics", {}).get("num_claims", 0)) for audit in final_audits)
    total_supported = sum(int(audit.get("metrics", {}).get("supported_claims", 0)) for audit in final_audits)
    total_unsupported = sum(int(audit.get("metrics", {}).get("unsupported_claims", 0)) for audit in final_audits)
    return {
        "num_examples": len(final_audits),
        "num_claims": total_claims,
        "supported_claims": total_supported,
        "unsupported_claims": total_unsupported,
        "atomic_claim_support_rate_micro": total_supported / total_claims if total_claims else 0.0,
        "unsupported_claim_rate_micro": total_unsupported / total_claims if total_claims else 0.0,
    }


def main() -> None:
    args = parse_args()
    if args.decomposer in {"llm", "claimify"} and not args.openai_api:
        raise ValueError(f"`--decomposer {args.decomposer}` requires `--openai-api`.")
    if args.verifier in {"llm", "hybrid"} and not args.openai_api:
        raise ValueError(f"`--verifier {args.verifier}` requires `--openai-api`.")
    if args.revision_mode == "llm" and not args.openai_api:
        raise ValueError("`--revision-mode llm` requires `--openai-api`.")

    payload, items = load_payload(args.input)
    fill_dataset_name_from_payload(args, payload)
    process_items = items if args.limit is None else items[: args.limit]
    cache = JsonCache(args.cache_file)
    decomposer = make_decomposer(args, cache)
    verifier = make_verifier(args, cache)
    revision_generator = make_revision_generator(args, cache)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **_: x

    revised_items = []
    for item_id, item in enumerate(tqdm(process_items, desc="COVER-RAG v1 revision")):
        revised_items.append(revise_item(item, item_id, decomposer, verifier, revision_generator, args))

    if args.limit is not None and len(items) > args.limit:
        revised_items.extend(items[args.limit :])

    payload["data"] = revised_items
    payload["cover_v1_summary"] = summarize_v1(revised_items[: len(process_items)])
    final_summary = summarize_final_audit(revised_items[: len(process_items)])
    if final_summary is not None:
        payload["cover_v1_final_audit_summary"] = final_summary
    payload["cover_v1_config"] = {
        "dataset_name": args.dataset_name,
        "answer_format": args.answer_format,
        "inferred_answer_format": infer_answer_format(args),
        "decomposer": args.decomposer,
        "decompose_model": args.decompose_model if args.decomposer in {"llm", "claimify"} else None,
        "claimify_context_before": args.claimify_context_before if args.decomposer == "claimify" else None,
        "claimify_context_after": args.claimify_context_after if args.decomposer == "claimify" else None,
        "claimify_selection_completions": args.claimify_selection_completions if args.decomposer == "claimify" else None,
        "claimify_selection_min_successes": args.claimify_selection_min_successes if args.decomposer == "claimify" else None,
        "claimify_disambiguation_completions": args.claimify_disambiguation_completions if args.decomposer == "claimify" else None,
        "claimify_disambiguation_min_successes": args.claimify_disambiguation_min_successes if args.decomposer == "claimify" else None,
        "claimify_decomposition_completions": args.claimify_decomposition_completions if args.decomposer == "claimify" else None,
        "claimify_decomposition_min_successes": args.claimify_decomposition_min_successes if args.decomposer == "claimify" else None,
        "claimify_implementation": args.claimify_implementation if args.decomposer == "claimify" else None,
        "claimify_external_path": str(args.claimify_external_path) if args.decomposer == "claimify" and args.claimify_external_path else None,
        "claimify_use_external_defaults": args.claimify_use_external_defaults if args.decomposer == "claimify" else None,
        "verifier": args.verifier,
        "llm_verify_model": args.llm_verify_model if args.verifier in {"llm", "hybrid"} else None,
        "nli_model": args.nli_model if args.verifier in {"nli", "hybrid"} else None,
        "revision_mode": args.revision_mode,
        "revision_model": args.revision_model if args.revision_mode == "llm" else None,
        "max_verified_claims": args.max_verified_claims,
        "max_qampari_items": args.max_qampari_items,
        "citation_policy": args.citation_policy,
        "max_citations_per_claim": args.max_citations_per_claim,
        "evidence_scope": args.evidence_scope,
        "allow_citation_repair": args.allow_citation_repair,
        "include_background_claims": args.include_background_claims,
        "final_audit": args.final_audit,
    }

    token_usage = collect_token_usage(decomposer, verifier, revision_generator)
    if token_usage:
        payload["cover_v1_token_usage"] = token_usage

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(safe_float(payload), ensure_ascii=False, indent=2 if args.pretty else None),
        encoding="utf-8",
    )
    logger.info("Wrote COVER-RAG v1 result to %s", args.output)
    logger.info("COVER-RAG v1 summary: %s", json.dumps(payload["cover_v1_summary"], ensure_ascii=False, indent=2))


def collect_token_usage(decomposer: Any, verifier: Any, revision_generator: RevisionGenerator) -> dict[str, int]:
    usage: dict[str, int] = {}
    for prefix, obj in [
        ("decomposition", decomposer),
        ("verification", verifier),
        ("revision", revision_generator),
    ]:
        prompt_tokens = getattr(obj, "prompt_tokens", None)
        completion_tokens = getattr(obj, "completion_tokens", None)
        if prompt_tokens is not None:
            usage[f"{prefix}_prompt_tokens"] = int(prompt_tokens)
        if completion_tokens is not None:
            usage[f"{prefix}_completion_tokens"] = int(completion_tokens)
        # Hybrid verifier stores LLM usage one level deeper.
        llm = getattr(obj, "llm", None)
        if llm is not None:
            usage[f"{prefix}_prompt_tokens"] = int(getattr(llm, "prompt_tokens", 0))
            usage[f"{prefix}_completion_tokens"] = int(getattr(llm, "completion_tokens", 0))
    return usage


if __name__ == "__main__":
    main()

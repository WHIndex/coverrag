#!/usr/bin/env python3
"""COVER-RAG v2: revise answers with targeted evidence recovery.

v1 is conservative: unsupported atomic claims are removed before answer
revision. v2 adds the missing step needed for accuracy:

    draft answer
    -> atomic claim audit
    -> targeted search for unsupported answer-relevant claims
    -> verify recovered evidence
    -> revise final answer from supported + recovered claims

The script stays ALCE-compatible: it reads a normal result JSON, replaces the
`output` field with the revised answer, and preserves all intermediate metadata
under `cover_v2`.
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
    JsonCache,
    OpenAIChatClient,
    VerificationResult,
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
from cover_v1_revise import (
    AutoRevisionGenerator,
    RevisionGenerator,
    choose_final_citations,
    claim_priority_key,
    clean_final_answer,
    collect_token_usage,
    format_citations,
    make_revision_generator,
    normalize_revision_claim_key,
    output_has_valid_citation,
    unique_positive_ints,
    verified_claim_limit,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("cover_v2_revise")


ALCE_LLM_SYSTEM = """You are a careful citation-grounded question answering system.
Answer using only the provided verified evidence. Every factual sentence must end with one to three citations like [1] or [1][2]. Do not invent citation ids."""


ALCE_LLM_USER = """Question:
{question}

Original draft answer:
{draft_answer}

Verified atomic claims that may be used:
{verified_claims}

Allowed evidence documents:
{evidence_docs}

Unsupported claims that must not be used:
{rejected_claims}

Instructions:
1. Write one concise paragraph, not a list.
2. Preserve useful wording and answer entities from the original draft when the verified claims support them.
3. Use only the verified claims and allowed evidence documents. Do not add outside knowledge.
4. Every factual sentence must end with citations from the allowed evidence documents.
5. Prefer exact names, dates, titles, and aliases that appear in the verified claims or evidence.
6. If the verified evidence is insufficient, answer only the supported part and say the remaining evidence is insufficient.

Final answer:"""


STOPWORDS = {
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
    "whom",
    "whose",
    "when",
    "where",
    "why",
    "how",
    "has",
    "have",
    "had",
    "does",
    "do",
    "did",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COVER-RAG v2 targeted evidence recovery for ALCE result JSON.")
    parser.add_argument("--input", required=True, type=Path, help="ALCE result JSON with draft outputs.")
    parser.add_argument("--output", required=True, type=Path, help="Where to write revised ALCE result JSON.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N examples.")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--answer-format", choices=["auto", "paragraph", "qampari"], default="auto")

    # ALCE fields.
    parser.add_argument("--output-field", default="output")
    parser.add_argument("--docs-field", default="docs")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--title-field", default="title")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--sent-field", default="sent")

    # Candidate evidence pool.
    parser.add_argument(
        "--candidate-docs-file",
        type=Path,
        default=None,
        help=(
            "Optional ALCE eval/retrieval JSON containing a larger docs pool, e.g. top100 docs. "
            "Records are matched by sample_id, question, then index. Raw docs stay as the prefix "
            "so old citation ids remain valid."
        ),
    )
    parser.add_argument("--max-doc-pool", type=int, default=100)

    # LLM / decomposition / verification settings. Defaults are chosen for a
    # reproducible first ASQA run with local NLI verification.
    parser.add_argument("--openai-api", action="store_true")
    parser.add_argument("--decomposer", choices=["llm", "sentence", "claimify"], default="llm")
    parser.add_argument("--decompose-model", default="gpt-4o-mini")
    parser.add_argument("--llm-verify-model", default="gpt-4o-mini")
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-top-p", type=float, default=1.0)
    parser.add_argument("--llm-max-retries", type=int, default=5)

    # Claimify-compatible args are accepted so make_decomposer can be reused.
    parser.add_argument("--claimify-context-before", type=int, default=3)
    parser.add_argument("--claimify-context-after", type=int, default=2)
    parser.add_argument("--claimify-max-stage-retries", type=int, default=1)
    parser.add_argument("--claimify-selection-completions", type=int, default=1)
    parser.add_argument("--claimify-selection-min-successes", type=int, default=1)
    parser.add_argument("--claimify-disambiguation-completions", type=int, default=1)
    parser.add_argument("--claimify-disambiguation-min-successes", type=int, default=1)
    parser.add_argument("--claimify-decomposition-completions", type=int, default=1)
    parser.add_argument("--claimify-decomposition-min-successes", type=int, default=1)
    parser.add_argument("--claimify-implementation", choices=["local", "external"], default="local")
    parser.add_argument("--claimify-external-path", type=Path, default=None)
    parser.add_argument("--claimify-use-external-defaults", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--verifier", choices=["nli", "llm", "lexical", "hybrid"], default="nli")
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
        default="cited",
        help="Draft and final audits should normally verify cited evidence only; v2 recovery does targeted search separately.",
    )
    parser.add_argument("--max-all-docs", type=int, default=20)
    parser.add_argument("--max-evidence-chars", type=int, default=6000)
    parser.add_argument("--lexical-threshold", type=float, default=0.45)

    # Recovery policy.
    parser.add_argument("--recover-labels", default="not_supported,no_citation,wrong_or_missing_citation")
    parser.add_argument(
        "--recover-importance",
        default="critical,supporting",
        help="Comma-separated claim importance labels eligible for recovery. Background is excluded by default.",
    )
    parser.add_argument("--recovery-top-k", type=int, default=6, help="Lexically ranked docs to verify per claim.")
    parser.add_argument(
        "--recovery-group-size",
        type=int,
        default=3,
        help="If no single doc supports a claim, verify the top N docs jointly as a last attempt.",
    )
    parser.add_argument("--min-recovery-score", type=float, default=0.03)
    parser.add_argument("--max-recovered-citations", type=int, default=3)
    parser.add_argument(
        "--keep-wrong-missing-if-recovered",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat wrong_or_missing_citation as recoverable instead of deleting it.",
    )

    # Revision generation.
    parser.add_argument("--revision-mode", choices=["llm", "template", "atomic", "qampari", "auto"], default="auto")
    parser.add_argument(
        "--v2-render-mode",
        choices=["source_aware", "alce_llm", "v1"],
        default="source_aware",
        help=(
            "source_aware preserves original answer sentences and citations when all claims in that sentence "
            "are supported, and falls back to atomic recovered claims otherwise. This is more compatible with "
            "ALCE's sentence-level citation metrics. alce_llm regenerates with an ALCE-style citation prompt "
            "from verified claims and evidence. v1 uses the v1 revision renderer directly."
        ),
    )
    parser.add_argument("--revision-model", default="gpt-4o-mini")
    parser.add_argument("--revision-max-tokens", type=int, default=300)
    parser.add_argument("--alce-llm-max-docs", type=int, default=8)
    parser.add_argument("--alce-llm-max-claims", type=int, default=18)
    parser.add_argument("--alce-llm-max-doc-chars", type=int, default=900)
    parser.add_argument("--max-verified-claims", type=int, default=18)
    parser.add_argument("--max-qampari-items", type=int, default=80)
    parser.add_argument("--max-rejected-claims", type=int, default=12)
    parser.add_argument(
        "--citation-policy",
        choices=["source", "all", "minimal"],
        default="source",
        help="Initial supported claims keep their original source citations by default.",
    )
    parser.add_argument("--max-citations-per-claim", type=int, default=3)
    parser.add_argument("--include-background-claims", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--abstain-if-no-supported", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fallback-template-on-empty", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--final-audit", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--cache-file", type=Path, default=None)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


class CandidateDocProvider:
    def __init__(self, path: Path | None, args: argparse.Namespace) -> None:
        self.path = path
        self.args = args
        self.by_sample_id: dict[str, list[dict[str, Any]]] = {}
        self.by_question: dict[str, list[dict[str, Any]]] = {}
        self.by_index: dict[int, list[dict[str, Any]]] = {}
        if path is not None:
            self._load(path)

    def _load(self, path: Path) -> None:
        if not path.exists():
            logger.warning("Candidate docs file does not exist: %s. Falling back to result JSON docs.", path)
            return
        payload, records = load_payload(path)
        _ = payload
        for idx, record in enumerate(records):
            docs = list(record.get(self.args.docs_field, []) or [])
            if not docs:
                continue
            self.by_index[idx] = docs
            sample_id = record.get("sample_id")
            if sample_id is not None:
                self.by_sample_id[str(sample_id)] = docs
            question = normalize_space(str(record.get(self.args.question_field, "") or "")).lower()
            if question:
                self.by_question[question] = docs
        logger.info(
            "Loaded candidate docs from %s: %d sample ids, %d questions, %d indexed records.",
            path,
            len(self.by_sample_id),
            len(self.by_question),
            len(self.by_index),
        )

    def docs_for(self, item: dict[str, Any], item_id: int) -> list[dict[str, Any]]:
        sample_id = item.get("sample_id")
        if sample_id is not None and str(sample_id) in self.by_sample_id:
            return self.by_sample_id[str(sample_id)]
        question = normalize_space(str(item.get(self.args.question_field, "") or "")).lower()
        if question and question in self.by_question:
            return self.by_question[question]
        return self.by_index.get(item_id, [])


def fill_dataset_name_from_payload(args: argparse.Namespace, payload: dict[str, Any]) -> None:
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


def doc_identity(doc: dict[str, Any], args: argparse.Namespace) -> str:
    title = normalize_space(str(doc.get(args.title_field, "") or "")).lower()
    body = normalize_space(str(doc.get(args.sent_field) or doc.get(args.text_field, "") or "")).lower()
    return f"{title}\n{body[:1000]}"


def merged_doc_pool(
    raw_docs: Sequence[dict[str, Any]],
    candidate_docs: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for doc in list(raw_docs) + list(candidate_docs):
        key = doc_identity(doc, args)
        if not key or key in seen:
            continue
        seen.add(key)
        pool.append(doc)
        if len(pool) >= args.max_doc_pool:
            break
    return pool


def content_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", remove_citations(text).lower())
    return {token for token in tokens if token not in STOPWORDS and len(token) > 1}


def overlap_score(query: str, document: str) -> float:
    query_tokens = content_tokens(query)
    if not query_tokens:
        return 0.0
    doc_tokens = content_tokens(document)
    if not doc_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / len(query_tokens)


def recovery_query(question: str, claim: dict[str, Any]) -> str:
    parts = [
        question,
        str(claim.get("claim", "") or ""),
        str(claim.get("source_sentence_without_citations", "") or ""),
    ]
    return " ".join(part for part in parts if part)


def rank_recovery_docs(
    question: str,
    claim: dict[str, Any],
    docs: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    query = recovery_query(question, claim)
    claim_text = str(claim.get("claim", "") or "")
    source_text = str(claim.get("source_sentence_without_citations", "") or "")
    source_ids = set(unique_positive_ints(claim.get("source_citation_ids") or claim.get("citation_ids") or []))
    audit_evidence_ids = set(unique_positive_ints(claim.get("evidence_doc_ids") or []))

    scored = []
    for doc_id, doc in enumerate(docs, start=1):
        doc_text = format_doc(doc, args)
        score = 0.55 * overlap_score(claim_text, doc_text)
        score += 0.30 * overlap_score(query, doc_text)
        score += 0.10 * overlap_score(source_text, doc_text)
        if doc_id in audit_evidence_ids:
            score += 0.20
        if doc_id in source_ids:
            score += 0.05
        title = str(doc.get(args.title_field, "") or "")
        score += 0.05 * overlap_score(claim_text, title)
        scored.append(
            {
                "doc_id": doc_id,
                "score": score,
                "doc": doc,
            }
        )

    scored.sort(key=lambda row: (-float(row["score"]), int(row["doc_id"])))
    return [row for row in scored if float(row["score"]) >= args.min_recovery_score]


def parse_csv_set(text: str) -> set[str]:
    return {part.strip() for part in str(text or "").split(",") if part.strip()}


def claim_is_initially_usable(claim: dict[str, Any], args: argparse.Namespace) -> bool:
    if claim.get("label") != "supported":
        return False
    if not args.include_background_claims and claim.get("importance") == "background":
        return False
    return bool(claim.get("evidence_doc_ids"))


def claim_should_recover(claim: dict[str, Any], args: argparse.Namespace) -> bool:
    label = str(claim.get("label", ""))
    if label == "wrong_or_missing_citation" and not args.keep_wrong_missing_if_recovered:
        return False
    if label not in parse_csv_set(args.recover_labels):
        return False
    importance = str(claim.get("importance", "supporting") or "supporting")
    if importance not in parse_csv_set(args.recover_importance):
        return False
    if not args.include_background_claims and importance == "background":
        return False
    return True


def verify_doc(
    verifier: Any,
    question: str,
    claim_text: str,
    doc_id: int,
    doc: dict[str, Any],
    args: argparse.Namespace,
) -> VerificationResult:
    result = verifier.verify(question, claim_text, [(doc_id, doc)], args)
    result.evidence_scope = "v2_targeted_single_doc"
    return result


def recover_claim(
    claim: dict[str, Any],
    question: str,
    docs: Sequence[dict[str, Any]],
    verifier: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    claim_text = normalize_space(str(claim.get("claim", "") or ""))
    ranked = rank_recovery_docs(question, claim, docs, args)
    top_ranked = ranked[: max(1, args.recovery_top_k)]
    attempts = []
    supported_doc_ids: list[int] = []
    best_result: VerificationResult | None = None

    for row in top_ranked:
        doc_id = int(row["doc_id"])
        result = verify_doc(verifier, question, claim_text, doc_id, row["doc"], args)
        attempts.append(
            {
                "doc_id": doc_id,
                "rank_score": float(row["score"]),
                "label": result.label,
                "confidence": result.confidence,
                "entailment": result.entailment,
                "rationale": result.rationale,
            }
        )
        if best_result is None or (result.entailment or 0.0) > (best_result.entailment or 0.0):
            best_result = result
        if result.label == "supported":
            supported_doc_ids.append(doc_id)
            if len(supported_doc_ids) >= max(1, args.max_recovered_citations):
                break

    if not supported_doc_ids and args.recovery_group_size > 1 and top_ranked:
        group = top_ranked[: args.recovery_group_size]
        group_docs = [(int(row["doc_id"]), row["doc"]) for row in group]
        group_result = verifier.verify(question, claim_text, group_docs, args)
        group_result.evidence_scope = "v2_targeted_doc_group"
        attempts.append(
            {
                "doc_id": [doc_id for doc_id, _ in group_docs],
                "rank_score": [float(row["score"]) for row in group],
                "label": group_result.label,
                "confidence": group_result.confidence,
                "entailment": group_result.entailment,
                "rationale": group_result.rationale,
            }
        )
        if best_result is None or (group_result.entailment or 0.0) > (best_result.entailment or 0.0):
            best_result = group_result
        if group_result.label == "supported":
            supported_doc_ids = [doc_id for doc_id, _ in group_docs[: args.max_recovered_citations]]

    diagnostics = {
        "attempted": bool(top_ranked),
        "num_ranked_candidates": len(ranked),
        "top_candidates": [
            {"doc_id": int(row["doc_id"]), "score": float(row["score"])}
            for row in top_ranked
        ],
        "attempts": attempts,
    }

    if not supported_doc_ids:
        diagnostics["best_label"] = best_result.label if best_result else None
        diagnostics["best_entailment"] = best_result.entailment if best_result else None
        return None, diagnostics

    recovered = dict(claim)
    recovered["label"] = "supported"
    recovered["v2_status"] = "recovered"
    recovered["v2_recovered_from_label"] = claim.get("label")
    recovered["v2_recovery"] = diagnostics
    recovered["citation_ids"] = supported_doc_ids
    recovered["evidence_doc_ids"] = supported_doc_ids
    recovered["final_citation_ids"] = supported_doc_ids
    recovered["evidence_scope"] = "v2_targeted_recovery"
    if best_result is not None:
        recovered["confidence"] = best_result.confidence
        recovered["entailment"] = best_result.entailment
        recovered["neutral"] = best_result.neutral
        recovered["contradiction"] = best_result.contradiction
        recovered["rationale"] = "Recovered by targeted evidence search. " + best_result.rationale
        recovered["verifier"] = f"{best_result.verifier}+v2_recovery"
    return recovered, diagnostics


def prepare_claims_for_v2(
    audit: dict[str, Any],
    item: dict[str, Any],
    verifier: Any,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_claims = list(audit.get("claims", []) or [])
    docs = item.get(args.docs_field, []) or []
    question = str(item.get(args.question_field, "") or "")
    usable: list[dict[str, Any]] = []
    recovered_claims: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    recovery_attempts: list[dict[str, Any]] = []
    seen: set[str] = set()

    for claim in raw_claims:
        normalized_key = normalize_revision_claim_key(claim, args)
        if not normalized_key or normalized_key in seen:
            continue
        seen.add(normalized_key)

        if claim_is_initially_usable(claim, args):
            updated = dict(claim)
            updated["v2_status"] = "initial_supported"
            updated["final_citation_ids"] = choose_final_citations(updated, args, docs)
            if updated["final_citation_ids"]:
                usable.append(updated)
            else:
                rejected.append(dict(claim))
            continue

        if claim_should_recover(claim, args):
            recovered, diagnostics = recover_claim(claim, question, docs, verifier, args)
            recovery_attempts.append(
                {
                    "claim": claim.get("claim"),
                    "source_label": claim.get("label"),
                    "importance": claim.get("importance"),
                    "recovered": recovered is not None,
                    "diagnostics": diagnostics,
                }
            )
            if recovered is not None:
                recovered_claims.append(recovered)
                usable.append(recovered)
                continue

        rejected_claim = dict(claim)
        rejected_claim["v2_status"] = "rejected_after_recovery" if claim_should_recover(claim, args) else "not_recovery_eligible"
        rejected.append(rejected_claim)

    usable.sort(key=claim_priority_key)
    limited = usable[: verified_claim_limit(args)]
    limited_keys = {normalize_revision_claim_key(claim, args) for claim in limited}
    overflow = [claim for claim in usable if normalize_revision_claim_key(claim, args) not in limited_keys]
    for claim in overflow:
        updated = dict(claim)
        updated["v2_status"] = "dropped_by_answer_budget"
        rejected.append(updated)
    return limited, rejected, recovered_claims, recovery_attempts


def compute_v2_metrics(
    draft_audit: dict[str, Any],
    verified_claims: Sequence[dict[str, Any]],
    rejected_claims: Sequence[dict[str, Any]],
    recovered_claims: Sequence[dict[str, Any]],
    recovery_attempts: Sequence[dict[str, Any]],
    revised_output: str,
    doc_pool_size: int,
    raw_doc_count: int,
) -> dict[str, Any]:
    draft_claims = draft_audit.get("claims", []) or []
    attempted = sum(1 for attempt in recovery_attempts if attempt.get("diagnostics", {}).get("attempted"))
    successful = sum(1 for attempt in recovery_attempts if attempt.get("recovered"))
    initially_supported = sum(1 for claim in verified_claims if claim.get("v2_status") == "initial_supported")
    return {
        "draft_num_claims": len(draft_claims),
        "initially_supported_claims": initially_supported,
        "recovered_claims": len(recovered_claims),
        "kept_verified_claims": len(verified_claims),
        "rejected_claims": len(rejected_claims),
        "recovery_attempted_claims": attempted,
        "recovery_successful_claims": successful,
        "recovery_success_rate": successful / attempted if attempted else 0.0,
        "claim_keep_rate": len(verified_claims) / len(draft_claims) if draft_claims else 0.0,
        "rejected_claim_rate": len(rejected_claims) / len(draft_claims) if draft_claims else 0.0,
        "revised_length": len((revised_output or "").split()),
        "revised_num_citations": len(parse_citations(revised_output)),
        "raw_doc_count": raw_doc_count,
        "doc_pool_size": doc_pool_size,
    }


def normalize_atomic_sentence(text: str) -> str:
    text = remove_citations(text)
    text = normalize_space(text).strip()
    return text.rstrip(".;:,")


def claim_is_render_eligible(claim: dict[str, Any], args: argparse.Namespace) -> bool:
    if not args.include_background_claims and claim.get("importance") == "background":
        return False
    return bool(normalize_revision_claim_key(claim, args))


def render_claim_atomic(claim: dict[str, Any], args: argparse.Namespace, num_docs: int) -> str:
    claim_text = normalize_atomic_sentence(str(claim.get("claim", "") or ""))
    citations = format_citations(
        claim.get("final_citation_ids", []),
        max_citations=args.max_citations_per_claim,
    )
    if not claim_text or not citations:
        return ""
    return clean_final_answer(f"{claim_text} {citations}.", num_docs=num_docs)


def source_sentence_can_be_preserved(
    sentence_claims: Sequence[dict[str, Any]],
    kept_by_key: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> bool:
    if not sentence_claims:
        return False
    for claim in sentence_claims:
        key = normalize_revision_claim_key(claim, args)
        if key not in kept_by_key:
            return False
        kept = kept_by_key[key]
        # Preserve the original sentence only when its original citations
        # already supported all included claims. Recovered claims use atomic
        # rendering with repaired citations instead.
        if kept.get("v2_status") != "initial_supported":
            return False
        if not kept.get("source_citation_ids") and not kept.get("citation_ids"):
            return False
    return True


def render_source_aware_answer(
    item: dict[str, Any],
    draft_audit: dict[str, Any],
    verified_claims: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    """Render v2 answers in a way that is friendlier to ALCE citation metrics.

    ALCE's citation metric evaluates sentence-level support. Pure atomic
    rendering is good for claim support, but it often changes sentence shape
    and citation placement. This renderer keeps an original sentence unchanged
    when every decomposed claim in that sentence survived verification with its
    original citations. Only partially supported or recovered content is written
    as atomic claim sentences.
    """

    if not verified_claims:
        return "The provided evidence is insufficient to answer the question."

    docs = item.get(args.docs_field, []) or []
    num_docs = len(docs)
    kept_by_key = {
        normalize_revision_claim_key(claim, args): claim
        for claim in verified_claims
        if normalize_revision_claim_key(claim, args)
    }
    consumed: set[str] = set()
    sentences: list[str] = []
    rendered_claim_count = 0
    max_claims = verified_claim_limit(args)

    for sentence_record in draft_audit.get("sentences", []) or []:
        if rendered_claim_count >= max_claims:
            break
        sentence_claims = [
            claim
            for claim in sentence_record.get("claims", []) or []
            if claim_is_render_eligible(claim, args)
        ]
        sentence_keys = [
            normalize_revision_claim_key(claim, args)
            for claim in sentence_claims
            if normalize_revision_claim_key(claim, args)
        ]
        sentence_keys = list(dict.fromkeys(sentence_keys))
        kept_sentence_keys = [key for key in sentence_keys if key in kept_by_key]
        if not kept_sentence_keys:
            continue

        original_sentence = normalize_space(str(sentence_record.get("sentence") or ""))
        can_preserve = (
            source_sentence_can_be_preserved(sentence_claims, kept_by_key, args)
            and bool(parse_citations(original_sentence))
            and rendered_claim_count + len(kept_sentence_keys) <= max_claims
        )
        if can_preserve:
            cleaned = clean_final_answer(original_sentence, num_docs=num_docs)
            if cleaned and parse_citations(cleaned):
                sentences.append(cleaned)
                consumed.update(kept_sentence_keys)
                rendered_claim_count += len(kept_sentence_keys)
                continue

        for key in kept_sentence_keys:
            if rendered_claim_count >= max_claims:
                break
            if key in consumed:
                continue
            atomic = render_claim_atomic(kept_by_key[key], args, num_docs)
            if atomic:
                sentences.append(atomic)
                consumed.add(key)
                rendered_claim_count += 1

    for claim in verified_claims:
        if rendered_claim_count >= max_claims:
            break
        key = normalize_revision_claim_key(claim, args)
        if not key or key in consumed:
            continue
        atomic = render_claim_atomic(claim, args, num_docs)
        if atomic:
            sentences.append(atomic)
            consumed.add(key)
            rendered_claim_count += 1

    if not sentences:
        return "The provided evidence is insufficient to answer the question."
    return clean_final_answer(" ".join(sentences), num_docs=num_docs)


class ALCEStyleRevisionGenerator(RevisionGenerator):
    """LLM renderer using ALCE-style citation constraints.

    The goal is to make v2 comparable to vanilla ALCE RAG: both are generated
    by an LLM under strong citation instructions. The difference is that v2
    exposes only verified/recovered claims and their supporting documents.
    """

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
        if not verified_claims:
            return "The provided evidence is insufficient to answer the question."

        question = str(item.get(args.question_field, "") or "")
        draft_answer = str(item.get("cover_v2_original_output") or item.get(args.output_field, "") or "")
        verified_block = render_alce_verified_claims(verified_claims, args)
        evidence_block = render_alce_evidence_docs(item, verified_claims, args)
        rejected_block = render_alce_rejected_claims(rejected_claims, args)
        key = "v2-alce-llm:" + stable_hash(question, draft_answer, verified_block, evidence_block, rejected_block)
        cached = self.cache.get(key)
        if cached is not None:
            return str(cached)

        user_prompt = ALCE_LLM_USER.format(
            question=question,
            draft_answer=draft_answer,
            verified_claims=verified_block or "None.",
            evidence_docs=evidence_block or "None.",
            rejected_claims=rejected_block or "None.",
        )
        raw, usage = self.client.chat(ALCE_LLM_SYSTEM, user_prompt, max_tokens=args.revision_max_tokens)
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        answer = clean_final_answer(raw, num_docs=len(item.get(args.docs_field, []) or []))
        self.cache.set(key, answer)
        return answer


def make_alce_style_revision_generator(args: argparse.Namespace, cache: JsonCache) -> ALCEStyleRevisionGenerator:
    if not args.openai_api:
        raise ValueError("`--v2-render-mode alce_llm` requires `--openai-api`.")
    client = OpenAIChatClient(
        model=args.revision_model,
        temperature=args.llm_temperature,
        top_p=args.llm_top_p,
        max_retries=args.llm_max_retries,
    )
    return ALCEStyleRevisionGenerator(client, cache)


def render_alce_verified_claims(verified_claims: Sequence[dict[str, Any]], args: argparse.Namespace) -> str:
    lines = []
    for idx, claim in enumerate(verified_claims[: args.alce_llm_max_claims], start=1):
        citations = format_citations(
            claim.get("final_citation_ids", []),
            max_citations=args.max_citations_per_claim,
        )
        status = str(claim.get("v2_status", "verified"))
        source = normalize_space(str(claim.get("source_sentence_without_citations", "") or ""))
        source_note = f" Source sentence: {source}" if source else ""
        lines.append(f"{idx}. {claim.get('claim', '')} {citations} ({status}).{source_note}")
    return "\n".join(lines)


def render_alce_rejected_claims(rejected_claims: Sequence[dict[str, Any]], args: argparse.Namespace) -> str:
    lines = []
    for idx, claim in enumerate(rejected_claims[: args.max_rejected_claims], start=1):
        lines.append(f"{idx}. {claim.get('claim', '')} ({claim.get('label', 'unknown')})")
    return "\n".join(lines)


def render_alce_evidence_docs(
    item: dict[str, Any],
    verified_claims: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    docs = item.get(args.docs_field, []) or []
    doc_ids = []
    for claim in verified_claims:
        for doc_id in claim.get("final_citation_ids", []) or []:
            try:
                value = int(doc_id)
            except Exception:
                continue
            if value > 0 and value not in doc_ids:
                doc_ids.append(value)
    doc_ids = doc_ids[: max(1, args.alce_llm_max_docs)]

    rendered = []
    for doc_id in doc_ids:
        idx = doc_id - 1
        if 0 <= idx < len(docs):
            text = normalize_space(format_doc(docs[idx], args))
            if len(text) > args.alce_llm_max_doc_chars:
                text = text[: args.alce_llm_max_doc_chars - 3].rstrip() + "..."
            rendered.append(f"[{doc_id}] {text}")
    return "\n\n".join(rendered)


def summarize_v2(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    records = [item.get("cover_v2", {}) for item in items if item.get("cover_v2")]
    metrics = [record.get("revision_metrics", {}) for record in records]

    total_draft_claims = sum(int(m.get("draft_num_claims", 0)) for m in metrics)
    total_initial = sum(int(m.get("initially_supported_claims", 0)) for m in metrics)
    total_recovered = sum(int(m.get("recovered_claims", 0)) for m in metrics)
    total_kept = sum(int(m.get("kept_verified_claims", 0)) for m in metrics)
    total_rejected = sum(int(m.get("rejected_claims", 0)) for m in metrics)
    total_attempted = sum(int(m.get("recovery_attempted_claims", 0)) for m in metrics)
    total_successful = sum(int(m.get("recovery_successful_claims", 0)) for m in metrics)
    lengths = [float(m.get("revised_length", 0.0)) for m in metrics]
    citations = [float(m.get("revised_num_citations", 0.0)) for m in metrics]
    doc_pool_sizes = [float(m.get("doc_pool_size", 0.0)) for m in metrics]

    label_counter: collections.Counter[str] = collections.Counter()
    for record in records:
        for claim in record.get("draft_audit", {}).get("claims", []) or []:
            label_counter[str(claim.get("label", "unknown"))] += 1

    return {
        "num_examples": len(records),
        "draft_num_claims": total_draft_claims,
        "initially_supported_claims": total_initial,
        "recovered_claims": total_recovered,
        "kept_verified_claims": total_kept,
        "rejected_claims": total_rejected,
        "recovery_attempted_claims": total_attempted,
        "recovery_successful_claims": total_successful,
        "recovery_success_rate_micro": total_successful / total_attempted if total_attempted else 0.0,
        "claim_keep_rate_micro": total_kept / total_draft_claims if total_draft_claims else 0.0,
        "rejected_claim_rate_micro": total_rejected / total_draft_claims if total_draft_claims else 0.0,
        "avg_revised_length": statistics.mean(lengths) if lengths else 0.0,
        "avg_revised_num_citations": statistics.mean(citations) if citations else 0.0,
        "avg_doc_pool_size": statistics.mean(doc_pool_sizes) if doc_pool_sizes else 0.0,
        "draft_label_counts": dict(label_counter),
    }


def summarize_final_audit(items: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    final_audits = [item.get("cover_v2", {}).get("final_audit") for item in items]
    final_audits = [audit for audit in final_audits if audit]
    if not final_audits:
        return None
    total_claims = sum(int(audit.get("metrics", {}).get("num_claims", 0)) for audit in final_audits)
    total_supported = sum(int(audit.get("metrics", {}).get("supported_claims", 0)) for audit in final_audits)
    total_unsupported = sum(int(audit.get("metrics", {}).get("unsupported_claims", 0)) for audit in final_audits)
    total_no_citation = sum(int(audit.get("metrics", {}).get("no_citation_claims", 0)) for audit in final_audits)
    return {
        "num_examples": len(final_audits),
        "num_claims": total_claims,
        "supported_claims": total_supported,
        "unsupported_claims": total_unsupported,
        "no_citation_claims": total_no_citation,
        "atomic_claim_support_rate_micro": total_supported / total_claims if total_claims else 0.0,
        "unsupported_claim_rate_micro": total_unsupported / total_claims if total_claims else 0.0,
    }


def revise_item_v2(
    item: dict[str, Any],
    item_id: int,
    candidate_provider: CandidateDocProvider,
    decomposer: Any,
    verifier: Any,
    revision_generator: RevisionGenerator,
    args: argparse.Namespace,
) -> dict[str, Any]:
    updated = dict(item)
    draft_output = str(updated.get(args.output_field, "") or "")
    updated["cover_v2_original_output"] = draft_output
    raw_docs = list(updated.get(args.docs_field, []) or [])
    candidate_docs = candidate_provider.docs_for(updated, item_id)
    doc_pool = merged_doc_pool(raw_docs, candidate_docs, args)
    updated[args.docs_field] = doc_pool

    draft_audit = audit_item(updated, item_id, decomposer, verifier, args)
    verified_claims, rejected_claims, recovered_claims, recovery_attempts = prepare_claims_for_v2(
        draft_audit,
        updated,
        verifier,
        args,
    )

    if args.v2_render_mode == "source_aware":
        revised_output = render_source_aware_answer(updated, draft_audit, verified_claims, args)
    elif args.v2_render_mode == "alce_llm":
        revised_output = revision_generator.revise(updated, verified_claims, rejected_claims, args)
    else:
        revised_output = revision_generator.revise(updated, verified_claims, rejected_claims, args)
    if (
        args.fallback_template_on_empty
        and verified_claims
        and (not revised_output or not output_has_valid_citation(revised_output, len(updated.get(args.docs_field, []) or [])))
    ):
        logger.warning("Falling back to deterministic revision for item %s because output was citation-invalid.", item_id)
        revised_output = AutoRevisionGenerator().revise(updated, verified_claims, rejected_claims, args)
    revised_output = clean_final_answer(revised_output, num_docs=len(updated.get(args.docs_field, []) or []))
    updated.pop("cover_v2_original_output", None)

    updated["cover_v2"] = {
        "original_output": draft_output,
        "raw_doc_count": len(raw_docs),
        "candidate_doc_count": len(candidate_docs),
        "doc_pool_size": len(doc_pool),
        "draft_audit": draft_audit,
        "verified_claims": verified_claims,
        "recovered_claims": recovered_claims,
        "rejected_claims": rejected_claims,
        "recovery_attempts": recovery_attempts,
        "revision_metrics": compute_v2_metrics(
            draft_audit,
            verified_claims,
            rejected_claims,
            recovered_claims,
            recovery_attempts,
            revised_output,
            len(doc_pool),
            len(raw_docs),
        ),
    }
    updated[args.output_field] = revised_output

    if args.final_audit:
        updated["cover_v2"]["final_audit"] = audit_item(updated, item_id, decomposer, verifier, args)
    return updated


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
    candidate_provider = CandidateDocProvider(args.candidate_docs_file, args)
    decomposer = make_decomposer(args, cache)
    verifier = make_verifier(args, cache)
    if args.v2_render_mode == "alce_llm":
        revision_generator = make_alce_style_revision_generator(args, cache)
    else:
        revision_generator = make_revision_generator(args, cache)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **_: x

    revised_items = []
    for item_id, item in enumerate(tqdm(process_items, desc="COVER-RAG v2 recovery")):
        revised_items.append(revise_item_v2(item, item_id, candidate_provider, decomposer, verifier, revision_generator, args))

    if args.limit is not None and len(items) > args.limit:
        revised_items.extend(items[args.limit :])

    payload["data"] = revised_items
    payload["cover_v2_summary"] = summarize_v2(revised_items[: len(process_items)])
    final_summary = summarize_final_audit(revised_items[: len(process_items)])
    if final_summary is not None:
        payload["cover_v2_final_audit_summary"] = final_summary
    payload["cover_v2_config"] = {
        "dataset_name": args.dataset_name,
        "answer_format": args.answer_format,
        "inferred_answer_format": infer_answer_format(args),
        "candidate_docs_file": str(args.candidate_docs_file) if args.candidate_docs_file else None,
        "max_doc_pool": args.max_doc_pool,
        "decomposer": args.decomposer,
        "decompose_model": args.decompose_model if args.decomposer in {"llm", "claimify"} else None,
        "verifier": args.verifier,
        "llm_verify_model": args.llm_verify_model if args.verifier in {"llm", "hybrid"} else None,
        "nli_model": args.nli_model if args.verifier in {"nli", "hybrid"} else None,
        "revision_mode": args.revision_mode,
        "v2_render_mode": args.v2_render_mode,
        "revision_model": args.revision_model if args.revision_mode == "llm" or args.v2_render_mode == "alce_llm" else None,
        "evidence_scope": args.evidence_scope,
        "alce_llm_max_docs": args.alce_llm_max_docs if args.v2_render_mode == "alce_llm" else None,
        "alce_llm_max_claims": args.alce_llm_max_claims if args.v2_render_mode == "alce_llm" else None,
        "alce_llm_max_doc_chars": args.alce_llm_max_doc_chars if args.v2_render_mode == "alce_llm" else None,
        "recover_labels": args.recover_labels,
        "recover_importance": args.recover_importance,
        "recovery_top_k": args.recovery_top_k,
        "recovery_group_size": args.recovery_group_size,
        "min_recovery_score": args.min_recovery_score,
        "max_recovered_citations": args.max_recovered_citations,
        "max_verified_claims": args.max_verified_claims,
        "max_qampari_items": args.max_qampari_items,
        "citation_policy": args.citation_policy,
        "max_citations_per_claim": args.max_citations_per_claim,
        "include_background_claims": args.include_background_claims,
        "final_audit": args.final_audit,
    }

    token_usage = collect_token_usage(decomposer, verifier, revision_generator)
    if token_usage:
        payload["cover_v2_token_usage"] = token_usage

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(safe_float(payload), ensure_ascii=False, indent=2 if args.pretty else None),
        encoding="utf-8",
    )
    logger.info("Wrote COVER-RAG v2 result to %s", args.output)
    logger.info("COVER-RAG v2 summary: %s", json.dumps(payload["cover_v2_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

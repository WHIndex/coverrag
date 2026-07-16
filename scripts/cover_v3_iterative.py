#!/usr/bin/env python3
"""COVER-RAG v3: iterative evidence-covered answer expansion.

v2 repairs unsupported claims from the draft answer. That improves atomic
faithfulness, but it cannot recover correct answer spans that never appeared in
the draft. v3 adds one more iteration:

    draft answer
    -> atomic claim audit
    -> targeted recovery for unsupported draft claims
    -> build an evidence core from supported/recovered claims
    -> retrieve evidence sentences from the larger doc pool
    -> extract answer-expansion claims
    -> verify expansion claims
    -> render a span-preserving final answer

The key design choice is conservative expansion: new claims may enter the final
answer only after direct evidence verification. This keeps the COVER-RAG claim
faithfulness goal while giving correctness metrics a chance to improve.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import logging
import math
import re
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

from cover_posthoc_audit import (
    JsonCache,
    OpenAIChatClient,
    VerificationResult,
    audit_item,
    extract_json_object,
    format_doc,
    infer_answer_format,
    load_payload,
    make_decomposer,
    make_verifier,
    normalize_space,
    parse_citations,
    remove_citations,
    safe_float,
    sent_tokenize_safe,
    stable_hash,
)
from cover_v1_revise import (
    AutoRevisionGenerator,
    choose_final_citations,
    clean_final_answer,
    collect_token_usage,
    extract_qampari_answer_item,
    format_citations,
    normalize_claim_key,
    normalize_revision_claim_key,
    normalize_qampari_item_key,
    output_has_valid_citation,
    unique_positive_ints,
    verified_claim_limit,
)
from coverage_reranker import (
    CoverageAwareReranker,
    format_candidate_evidence,
    format_coverage_query,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("cover_v3_iterative")


NLI_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"


EXPANSION_SYSTEM = """You are an evidence extraction module for citation-grounded QA.
Your job is to find missing answer claims that are directly supported by the
provided evidence sentences. You must not use outside knowledge."""


EXPANSION_USER = """Question:
{question}

Intermediate supported answer:
{supported_answer}

Current supported answer claims:
{supported_claims}

Unsupported or rejected draft claims:
{rejected_claims}

Candidate evidence sentences:
{evidence_sentences}

Candidate answer spans copied from those evidence sentences:
{candidate_answer_spans}

Recall-oriented completion targets:
{completion_targets}

Return a JSON object with this schema:
{{
  "candidates": [
    {{
      "span_id": "s1",
      "claim": "short claim copied or minimally edited from the span sentence",
      "target_gap": "the missing facet or uncovered answer unit this claim fills"
    }}
  ]
}}

Rules:
1. Only include claims that directly help answer the question.
2. Every candidate must choose one span_id from the candidate answer span list.
3. The claim must be supported by the sentence shown for that span_id.
4. Do not copy evidence_sentence, doc_id, or answer_span into the JSON.
5. Do not include background facts that are not needed for the answer.
6. Prefer missing answer entities, dates, titles, aliases, numbers, causes, mechanisms, conditions, or effects.
7. Strongly prefer answer spans from the candidate answer span list when they answer the question.
8. If an evidence sentence contains several possible answer spans, create separate atomic claims with separate span_id values.
9. Treat the intermediate supported answer as a partial answer. Add only answer units that fill a visible gap, refine an underspecified point, or add a missing item.
10. Do not rewrite, repeat, or contradict the intermediate supported answer.
11. Prefer claims that add a new answer span or explanatory answer unit not already covered by the current supported answer.
12. If the current supported answer already covers all answer spans visible in the evidence, return {{"candidates": []}}.
13. If no useful missing claim is supported by the evidence, return {{"candidates": []}}.
14. When recall-oriented completion targets are provided, every candidate must fill one listed missing/gap target.
15. Keep each claim under 22 words.
16. The claim should usually contain the selected answer span text.
17. Return minified strict JSON only, with no markdown and no explanatory text.
"""


EXPANSION_SPAN_ID_USER = """Question:
{question}

Intermediate supported answer:
{supported_answer}

Current supported answer claims:
{supported_claims}

Unsupported or rejected draft claims:
{rejected_claims}

Candidate evidence sentences:
{evidence_sentences}

Candidate answer spans copied from those evidence sentences:
{candidate_answer_spans}

Recall-oriented completion targets:
{completion_targets}

Select evidence span ids that should be added as missing answer units.
Think like a gold-answer evaluator: choose short spans that would plausibly be
counted as a missing entity, date, title, alias, number, cause, mechanism,
condition, or effect for the question. Do not choose spans that are merely
topical background, webpage boilerplate, examples unrelated to the asked
relation, or duplicates of the intermediate answer.

Return a JSON object with exactly this schema:
{{"selected_span_ids":["s1","s2"]}}

Rules:
1. Only select span_id values that appear in the candidate answer span list.
2. Every selected span must directly fill one visible missing/gap target when targets are provided.
3. Prefer exact answer spans over whole clauses.
4. Do not select evidence snippets that look like site navigation, download ads, dates from page metadata, author bylines, category labels, or truncated text.
5. Select at most {max_selected_span_ids} span ids. Prefer the best few; never select every span.
6. If the intermediate supported answer already covers all answer units visible in the evidence, return {{"selected_span_ids":[]}}.
7. If no useful missing answer unit is supported by the evidence, return {{"selected_span_ids":[]}}.
8. Return minified strict JSON only, with no markdown and no explanatory text.
"""


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


GENERIC_QAMPARI_ANSWER_KEYS = {
    "",
    "a",
    "an",
    "and",
    "answer",
    "bounty",
    "british",
    "chapter",
    "college",
    "country",
    "draft",
    "english",
    "evidence",
    "film",
    "films",
    "german",
    "group",
    "high school",
    "jr",
    "language",
    "movie",
    "movies",
    "music",
    "provided evidence",
    "question",
    "school",
    "software",
    "song",
    "therefore",
    "the",
    "type",
    "university",
    "work",
    "works",
}


QAMPARI_BAD_SUBSTRINGS = (
    "provided evidence",
    "provided documents",
    "search results",
    "insufficient",
    "cannot answer",
    "cannot provide",
    "therefore",
)


class CandidateDocProvider:
    """Load optional top-k candidate docs without depending on v2 internals."""

    def __init__(self, path: Path | None, args: argparse.Namespace) -> None:
        self.path = path
        self.args = args
        self.by_sample_id: dict[str, list[dict[str, Any]]] = {}
        self.by_question: dict[str, list[dict[str, Any]]] = {}
        self.by_index: dict[int, list[dict[str, Any]]] = {}
        self.global_docs: list[dict[str, Any]] = []
        self._global_doc_keys: set[str] = set()
        self._dynamic_index_ready = False
        self._dynamic_index_lock = threading.Lock()
        self._dynamic_postings: dict[str, list[int]] = {}
        self._dynamic_idf: dict[str, float] = {}
        self._dynamic_doc_tokens: list[set[str]] = []
        if path is not None:
            self._load(path)

    def _add_global_docs(self, docs: Sequence[dict[str, Any]]) -> None:
        for doc in docs:
            key = doc_identity(doc, self.args)
            if not key or key in self._global_doc_keys:
                continue
            self._global_doc_keys.add(key)
            self.global_docs.append(doc)

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
            self._add_global_docs(docs)
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

    def _ensure_dynamic_index(self) -> None:
        if self._dynamic_index_ready:
            return
        with self._dynamic_index_lock:
            if self._dynamic_index_ready:
                return
            postings: dict[str, list[int]] = collections.defaultdict(list)
            doc_tokens: list[set[str]] = []
            for doc_idx, doc in enumerate(self.global_docs):
                text = dynamic_retrieval_doc_text(doc, self.args)
                tokens = content_tokens(text)
                doc_tokens.append(tokens)
                for token in tokens:
                    postings[token].append(doc_idx)
            total_docs = max(1, len(self.global_docs))
            self._dynamic_postings = dict(postings)
            self._dynamic_idf = {
                token: math.log((total_docs + 1.0) / (len(doc_ids) + 0.5)) + 1.0
                for token, doc_ids in self._dynamic_postings.items()
            }
            self._dynamic_doc_tokens = doc_tokens
            self._dynamic_index_ready = True
            if self.global_docs:
                logger.info(
                    "Built dynamic candidate-doc retrieval index: %d docs, %d tokens.",
                    len(self.global_docs),
                    len(self._dynamic_postings),
                )

    def search_dynamic(
        self,
        query: str,
        limit: int,
        args: argparse.Namespace,
        *,
        exclude_keys: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        if limit <= 0 or not query or not self.global_docs:
            return []
        if getattr(args, "dynamic_retrieval_mode", "off") != "global_candidate_docs":
            return []
        self._ensure_dynamic_index()
        query_tokens = content_tokens(query)
        if len(query_tokens) < max(1, int(getattr(args, "dynamic_retrieval_min_query_terms", 2) or 2)):
            return []
        scores: dict[int, float] = collections.defaultdict(float)
        for token in query_tokens:
            idf = self._dynamic_idf.get(token, 0.0)
            if idf <= 0:
                continue
            for doc_idx in self._dynamic_postings.get(token, []):
                scores[doc_idx] += idf
        if not scores:
            return []

        exclude_keys = exclude_keys or set()
        ranked: list[tuple[float, int]] = []
        query_len = max(1, len(query_tokens))
        min_score = float(getattr(args, "dynamic_retrieval_min_score", 0.0) or 0.0)
        for doc_idx, lexical_score in scores.items():
            doc = self.global_docs[doc_idx]
            key = doc_identity(doc, self.args)
            if key in exclude_keys:
                continue
            title = normalize_space(str(doc.get(self.args.title_field, "") or ""))
            doc_text = dynamic_retrieval_doc_text(doc, self.args)
            normalized_score = lexical_score / query_len
            normalized_score += 0.25 * overlap_score(query, title)
            normalized_score += 0.10 * overlap_score(query, doc_text[:600])
            if normalized_score < min_score:
                continue
            ranked.append((normalized_score, doc_idx))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        ranked = rerank_dynamic_docs_with_coverage_model(query, ranked, self.global_docs, args)

        output: list[dict[str, Any]] = []
        for score, doc_idx in ranked[:limit]:
            doc = dict(self.global_docs[doc_idx])
            doc["cover_dynamic_retrieval_score"] = float(score)
            doc["cover_dynamic_retrieval_query"] = truncate_text(query, 240)
            output.append(doc)
        return output

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


def coverage_reranker_enabled(args: argparse.Namespace) -> bool:
    return getattr(args, "_coverage_reranker", None) is not None


def coverage_reranker_score_pairs(
    args: argparse.Namespace,
    queries: Sequence[str],
    evidences: Sequence[str],
) -> list[float]:
    reranker = getattr(args, "_coverage_reranker", None)
    if reranker is None or not queries:
        return [0.0 for _ in queries]
    try:
        return list(reranker.score_pairs(queries, evidences))
    except Exception as exc:
        logger.warning("Coverage-aware reranker scoring failed; falling back to lexical scores. Error: %s", exc)
        return [0.0 for _ in queries]


def rerank_dynamic_docs_with_coverage_model(
    query: str,
    ranked: Sequence[tuple[float, int]],
    docs: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> list[tuple[float, int]]:
    if not ranked or not coverage_reranker_enabled(args):
        return list(ranked)
    prefilter = max(
        int(getattr(args, "coverage_reranker_doc_prefilter", 64) or 64),
        int(getattr(args, "dynamic_retrieval_top_k", 12) or 12),
    )
    weight = max(0.0, min(float(getattr(args, "coverage_reranker_doc_weight", 0.35) or 0.0), 1.0))
    if weight <= 0.0:
        return list(ranked)
    head = list(ranked[:prefilter])
    tail = list(ranked[prefilter:])
    max_lexical = max((score for score, _idx in head), default=1.0) or 1.0
    query_texts = [format_coverage_query(query, missing_unit=query) for _score, _idx in head]
    evidence_texts = []
    for _score, doc_idx in head:
        doc = docs[doc_idx]
        title = normalize_space(str(doc.get(args.title_field, "") or ""))
        body = dynamic_retrieval_doc_text(doc, args)
        evidence_texts.append(format_candidate_evidence(body, title=title))
    model_scores = coverage_reranker_score_pairs(args, query_texts, evidence_texts)

    reranked: list[tuple[float, int]] = []
    for (lexical_score, doc_idx), model_score in zip(head, model_scores):
        lexical_norm = max(0.0, min(float(lexical_score) / max_lexical, 1.0))
        blended = (1.0 - weight) * lexical_norm + weight * max(0.0, min(float(model_score), 1.0))
        reranked.append((blended, doc_idx))
    reranked.sort(key=lambda row: (-row[0], row[1]))
    if tail:
        tail_offset = min((score for score, _idx in reranked), default=0.0) - 1e-6
        reranked.extend((tail_offset * max(0.0, min(score / max_lexical, 1.0)), idx) for score, idx in tail)
    return reranked


def parse_csv_set(text: str) -> set[str]:
    return {part.strip() for part in str(text or "").split(",") if part.strip()}


def claim_is_initially_usable(claim: dict[str, Any], args: argparse.Namespace) -> bool:
    if claim.get("label") != "supported":
        return False
    if not args.include_background_claims and claim.get("importance") == "background":
        return False
    return bool(claim.get("evidence_doc_ids"))


def extract_protected_spans(text: str) -> list[str]:
    """Extract answer-like surface forms that should not be paraphrased."""

    text = str(text or "")
    spans: list[str] = []
    patterns = [
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b\d{4}\b",
        r"\b\d+(?:\.\d+)?\s*(?:million|billion|thousand|percent|%)?\b",
        r"\"[^\"]{2,80}\"",
        r"'[^']{2,80}'",
        r"\b[A-Z][A-Za-z0-9&'.-]+(?:\s+(?:of|the|and|&|de|da|van|von|[A-Z][A-Za-z0-9&'.-]+)){0,6}\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            span = normalize_space(match.group(0)).strip("\"'")
            if is_protected_span(span) and span not in spans:
                spans.append(span)
    return spans[:80]


def is_protected_span(span: str) -> bool:
    span = normalize_space(span)
    if len(span) < 2:
        return False
    lowered = span.lower()
    if lowered in {"the", "a", "an", "answer", "question", "document", "title", "however", "although", "according"}:
        return False
    if lowered.startswith(("document ", "title ")):
        return False
    return True


def is_bad_general_answer_span(
    span: str,
    question: str = "",
    evidence_sentence: str = "",
    answer_type_gate_mode: str = "strict",
) -> tuple[bool, str]:
    """Conservative answer-span filter for ASQA/ELI5 expansion.

    This is much lighter than the QAMPARI entity-list filter. It only removes
    spans that are clearly generic, copied from the question, malformed, or not
    present in the evidence sentence. The point is to let expansion improve
    correctness by adding missing answer strings without opening the door to
    supported but off-topic background facts.
    """

    raw = normalize_space(remove_citations(span)).strip().strip("\"'")
    raw = raw.rstrip(".;:,")
    key = protected_span_key(raw)
    if not key:
        return True, "empty_answer_span"
    lowered = raw.lower()
    if lowered in {
        "the",
        "this",
        "that",
        "these",
        "those",
        "it",
        "he",
        "she",
        "they",
        "we",
        "document",
        "question",
        "answer",
        "evidence",
    }:
        return True, "generic_answer_span"
    words = key.split()
    if len(words) > 14:
        return True, "answer_span_too_long"
    if len(raw) < 2:
        return True, "answer_span_too_short"
    if words and words[-1] in {"and", "or", "of", "the", "a", "an", "in", "for", "to", "with"}:
        return True, "trailing_connector"

    q_tokens = content_tokens(question)
    span_tokens = content_tokens(raw)
    if question and span_tokens and q_tokens and span_tokens <= q_tokens:
        return True, "answer_tokens_all_in_question"
    if evidence_sentence and key not in protected_span_key(evidence_sentence):
        return True, "answer_span_not_in_evidence"
    type_reason = general_answer_span_type_mismatch(raw, question, evidence_sentence)
    if answer_type_mismatch_is_hard(type_reason, answer_type_gate_mode):
        return True, type_reason
    return False, ""


WEB_BOILERPLATE_PATTERNS = (
    r"\bfree download\b",
    r"\bdownload (?:the )?(?:game|app|software|file)\b",
    r"\bposted by\b",
    r"\bposted on\b",
    r"\bfiled under\b",
    r"\btagged\b",
    r"\btags?\b",
    r"\|\s*tags?\b",
    r"\bcomments?\b",
    r"\bhome\s*/",
    r"\bread more\b",
    r"\bclick here\b",
    r"\brating\b",
    r"\breview & buying guide\b",
    r"\bprivacy policy\b",
    r"\bterms of use\b",
    r"\bcopyright\b",
    r"\bhttp[s]?://",
    r"\bwww\.",
    r"\b\d{4}-\d{2}-\d{2}\b",
)


def bad_expansion_surface_reason(
    claim: str,
    answer_span: str = "",
    evidence_sentence: str = "",
    question: str = "",
    answer_type_gate_mode: str = "strict",
) -> str:
    """Reject supported-but-not-answer-like expansion surfaces.

    The verifier can prove a webpage fragment is entailed by a source, but the
    main task scorer only rewards answer units. This filter is intentionally
    conservative and only catches obvious boilerplate, truncation, and answer
    spans that do not appear to be anchored in the final claim.
    """

    cleaned_claim = normalize_space(remove_citations(claim)).strip()
    cleaned_span = normalize_space(remove_citations(answer_span)).strip()
    cleaned_evidence = normalize_space(remove_citations(evidence_sentence)).strip()
    if not cleaned_claim:
        return "empty_claim"
    lowered = cleaned_claim.lower()
    evidence_lowered = cleaned_evidence.lower()
    for pattern in WEB_BOILERPLATE_PATTERNS:
        if (
            re.search(pattern, lowered)
            or (cleaned_span and re.search(pattern, cleaned_span.lower()))
            or (cleaned_evidence and re.search(pattern, cleaned_evidence.lower()))
        ):
            return "web_boilerplate"
    if re.search(r"\b(?:[A-Z][a-z]{2,8}\s+){1,2}\d{1,2},\s*\d{4}\b", cleaned_claim) and overlap_score(question, cleaned_claim) < 0.05:
        return "page_date_like"
    if cleaned_claim.endswith((",", ";", ":", "–", "-", "—")):
        return "truncated_claim"
    if cleaned_claim.startswith(("and ", "or ", "but ", "which ", "while ", "whereas ", "because ")):
        return "leading_connector"
    if (
        answer_type_gate_mode == "strict"
        and
        question
        and not is_explanatory_question(question)
        and cleaned_claim[:1].islower()
        and word_count(cleaned_claim) > 3
    ):
        return "leading_lowercase_fragment"
    if len(cleaned_claim) > 0 and cleaned_claim.count('"') % 2 == 1:
        return "unbalanced_quote"
    if cleaned_claim.count("(") != cleaned_claim.count(")"):
        return "unbalanced_parenthesis"
    if cleaned_span:
        span_key = protected_span_key(cleaned_span)
        claim_key = protected_span_key(cleaned_claim)
        evidence_key = protected_span_key(cleaned_evidence)
        if span_key and span_key not in claim_key and span_key not in evidence_key:
            return "span_not_anchored"
    if word_count(cleaned_claim) <= 1 and not re.search(r"\d", cleaned_claim):
        return "too_short_non_numeric"
    if cleaned_evidence and cleaned_claim.lower() == evidence_lowered and word_count(cleaned_claim) > 24:
        return "whole_sentence_claim"
    type_reason = general_answer_span_type_mismatch(cleaned_span or cleaned_claim, question, cleaned_evidence)
    if answer_type_mismatch_is_hard(type_reason, answer_type_gate_mode):
        return type_reason
    return ""


def is_explanatory_question(question: str) -> bool:
    lowered = normalize_space(question).lower()
    if re.search(
        r"\bhow (?:many|much|long|old|far|often|large|small|tall|wide|deep|high|fast)\b",
        lowered,
    ):
        return False
    if re.search(r"\b(why|explain|what happens|what causes|what makes|what is the reason)\b", lowered):
        return True
    return bool(re.search(r"\bhow (?:do|does|did|can|could|would|is|are|was|were|to)\b", lowered))


MONTH_PATTERN = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    flags=re.IGNORECASE,
)


def question_answer_type(question: str) -> str:
    lowered = normalize_space(question).lower()
    if re.search(r"\bhow (?:many|much|long|old|far|often)\b|\bnumber of\b|\btotal\b|\bcount\b", lowered):
        return "number"
    if re.search(r"\bwhen\b|\bwhat (?:date|year|time)\b", lowered):
        return "date"
    if re.search(r"\bwhere\b|\bwhat (?:city|country|state|place|location)\b", lowered):
        return "place"
    if re.search(r"\bwho\b|\bwhom\b|\bwhose\b", lowered):
        return "person"
    if is_explanatory_question(question):
        return "explanation"
    return "open"


def span_is_temporal_like(span: str) -> bool:
    text = normalize_space(span)
    if MONTH_PATTERN.search(text):
        return True
    if re.search(r"\b(?:19|20)\d{2}\b", text):
        return True
    if re.search(r"\b\d{1,2}/\d{1,2}/(?:\d{2}|\d{4})\b", text):
        return True
    return False


def span_is_numeric_like(span: str) -> bool:
    text = normalize_space(span)
    return bool(re.search(r"\d", text)) and not bool(re.search(r"[A-Za-z]", text))


def span_is_bare_year(span: str) -> bool:
    return bool(re.fullmatch(r"(?:19|20)\d{2}", normalize_space(span)))


def evidence_looks_like_web_boilerplate(evidence_sentence: str) -> bool:
    lowered = normalize_space(evidence_sentence).lower()
    if not lowered:
        return False
    return any(re.search(pattern, lowered) for pattern in WEB_BOILERPLATE_PATTERNS)


HARD_ANSWER_TYPE_MISMATCH_REASONS = {
    "web_boilerplate_context",
    "person_question_non_person_span",
    "person_question_leading_adverb",
    "date_question_non_temporal_span",
    "number_question_year_span",
    "place_question_non_place_span",
}


def answer_type_mismatch_is_hard(reason: str, mode: str = "strict") -> bool:
    if not reason:
        return False
    mode = str(mode or "strict").lower()
    if mode == "off":
        return False
    if mode == "soft":
        return reason in HARD_ANSWER_TYPE_MISMATCH_REASONS
    return True


def general_answer_span_type_mismatch(span: str, question: str = "", evidence_sentence: str = "") -> str:
    """Catch answer-span candidates that match evidence but not the question type."""

    raw = normalize_space(remove_citations(span)).strip().strip("\"'")
    raw = raw.rstrip(".;:,")
    if not raw or not question:
        return ""

    qtype = question_answer_type(question)
    key_words = protected_span_key(raw).split()
    span_tokens = content_tokens(raw)

    if evidence_looks_like_web_boilerplate(evidence_sentence):
        return "web_boilerplate_context"

    if qtype == "explanation":
        # ELI5 gains come from mechanism/definition clauses. Standalone dates,
        # single tokens, names, and web metadata are usually supported noise.
        if span_is_numeric_like(raw) or span_is_bare_year(raw) or len(span_tokens) <= 1:
            return "explanatory_atomic_span"
        if len(key_words) <= 3 and raw[:1].isupper() and overlap_score(question, raw) < 0.08:
            return "explanatory_entity_span"
        return ""

    if qtype == "person":
        if span_is_temporal_like(raw) or span_is_numeric_like(raw):
            return "person_question_non_person_span"
        if raw.lower().startswith(("notably ", "currently ", "previously ", "originally ")):
            return "person_question_leading_adverb"
        if len(key_words) > 8:
            return "person_answer_too_long"
        return ""

    if qtype == "date":
        if not span_is_temporal_like(raw):
            return "date_question_non_temporal_span"
        return ""

    if qtype == "number":
        if not re.search(r"\d", raw):
            return "number_question_non_numeric_span"
        if span_is_bare_year(raw):
            return "number_question_year_span"
        return ""

    if qtype == "place":
        if span_is_temporal_like(raw) or span_is_numeric_like(raw):
            return "place_question_non_place_span"
        return ""

    return ""


def salient_answer_phrases(sentence: str, question: str = "", limit: int = 12) -> list[str]:
    """Extract short explanatory answer units for ELI5-style questions.

    Proper-name/date spans are not enough for explanation tasks. These phrases
    stay conservative: each one must be copied from the evidence sentence and
    contain at least one content word that is not already in the question.
    """

    sentence = normalize_space(remove_citations(sentence))
    if not sentence:
        return []

    patterns = [
        r"\b(?:because|due to|as a result of|caused by|causes?|results? from|stems? from)\s+([^.;:]{3,110})",
        r"\b(?:leads? to|results? in|creates?|produces?|allows?|enables?|prevents?|helps?|requires?)\s+([^.;:]{3,110})",
        r"\b(?:is|are|was|were)\s+(?:called|known as|defined as|described as|made of|composed of)\s+([^.;:]{3,110})",
        r"\b(?:the reason is|the main reason is|this happens when|this happens because)\s+([^.;:]{3,110})",
    ]
    spans: list[str] = []
    q_tokens = content_tokens(question)

    def add(raw: str) -> None:
        span = normalize_space(raw).strip(" \"'()[]")
        span = span.rstrip(".;:,")
        if not span:
            return
        words = span.split()
        if len(words) < 2 or len(words) > 14:
            return
        key = protected_span_key(span)
        if not key or key in {protected_span_key(existing) for existing in spans}:
            return
        span_tokens = content_tokens(span)
        if not span_tokens:
            return
        if q_tokens and span_tokens <= q_tokens:
            return
        if span.lower().startswith(("the ", "a ", "an ")):
            # Keep the phrase grammatical but avoid huge noun phrases that are
            # mostly copied topic setup.
            if len(words) > 10:
                return
        spans.append(span)

    for pattern in patterns:
        for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
            fragment = re.split(r",\s+(?:and|but|which|while|whereas)\b|,\s+", match.group(1))[0]
            add(fragment)
            if len(spans) >= limit:
                return spans

    if is_explanatory_question(question):
        # Backstop: use compact clauses with strong question overlap. This gives
        # ELI5 a chance to add mechanism claims even when the sentence lacks an
        # explicit "because" marker.
        for clause in re.split(r";\s+|,\s+(?:and|but|while|whereas|which)\b|\.\s+", sentence):
            clause = normalize_space(clause)
            if overlap_score(question, clause) < 0.18:
                continue
            add(clause)
            if len(spans) >= limit:
                break
    return spans


def candidate_answer_spans_from_sentence(
    sentence: str,
    verified_claims: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    question: str = "",
) -> list[str]:
    spans: list[str] = []
    seen: set[str] = set()
    covered = covered_answer_spans(verified_claims)
    answer_targeted_paragraph = (
        getattr(args, "answer_targeted_expansion", False) and infer_answer_format(args) != "qampari"
    )
    require_phrase_span = (
        answer_targeted_paragraph
        and getattr(args, "explanatory_expansion_require_phrase_span", False)
        and is_explanatory_question(question)
    )
    if require_phrase_span:
        raw_spans = salient_answer_phrases(sentence, question, limit=8)
        raw_spans.extend(
            span
            for span in extract_protected_spans(sentence)
            if re.search(r"\d|%|percent|million|billion|thousand", span, flags=re.IGNORECASE)
        )
    else:
        raw_spans = list(extract_protected_spans(sentence))
        if answer_targeted_paragraph:
            raw_spans.extend(salient_answer_phrases(sentence, question, limit=8))

    for span in raw_spans:
        key = protected_span_key(span)
        if not key or key in seen:
            continue
        if args.skip_covered_answer_spans and key in covered:
            continue
        if infer_answer_format(args) == "qampari" and args.qampari_answer_relevance_filter:
            bad, _reason = qampari_is_bad_answer_span(span, question, sentence)
            if bad:
                continue
        elif getattr(args, "expansion_filter_general_answer_spans", False):
            bad, _reason = is_bad_general_answer_span(
                span,
                question,
                sentence,
                getattr(args, "answer_type_gate_mode", "strict"),
            )
            if bad:
                continue
        seen.add(key)
        spans.append(span)
    return spans


def span_in_text(span: str, text: str) -> bool:
    if not span or not text:
        return False
    return normalize_space(span).lower() in normalize_space(remove_citations(text)).lower()


def claim_answer_spans(claim: dict[str, Any], protected_spans: Sequence[str]) -> list[str]:
    haystack = " ".join(
        str(claim.get(key, "") or "")
        for key in ("claim", "source_sentence", "source_sentence_without_citations")
    )
    return [span for span in protected_spans if span_in_text(span, haystack)]


def v2_claim_priority_key(claim: dict[str, Any]) -> tuple[int, int, int, int, int]:
    importance_rank = {"critical": 0, "supporting": 1, "background": 2}
    status_rank = {"initial_supported": 0, "recovered": 1, "expanded": 1}
    label_rank = {"supported": 0, "wrong_or_missing_citation": 1}
    span_count = len(claim.get("protected_answer_spans", []) or [])
    return (
        -span_count,
        importance_rank.get(str(claim.get("importance", "supporting")), 1),
        status_rank.get(str(claim.get("v2_status", "")), 2),
        label_rank.get(str(claim.get("label", "supported")), 0),
        int(claim.get("source_sentence_id", 0)),
    )


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
        scored.append({"doc_id": doc_id, "score": score, "doc": doc})

    scored.sort(key=lambda row: (-float(row["score"]), int(row["doc_id"])))
    return [row for row in scored if float(row["score"]) >= args.min_recovery_score]


def verify_doc(
    verifier: Any,
    question: str,
    claim_text: str,
    doc_id: int,
    doc: dict[str, Any],
    args: argparse.Namespace,
) -> VerificationResult:
    result = verifier.verify(question, claim_text, [(doc_id, doc)], args)
    result.evidence_scope = "v3_targeted_single_doc"
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
        group_result.evidence_scope = "v3_targeted_doc_group"
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
        "top_candidates": [{"doc_id": int(row["doc_id"]), "score": float(row["score"])} for row in top_ranked],
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
    recovered["evidence_scope"] = "v3_targeted_recovery"
    if best_result is not None:
        recovered["confidence"] = best_result.confidence
        recovered["entailment"] = best_result.entailment
        recovered["neutral"] = best_result.neutral
        recovered["contradiction"] = best_result.contradiction
        recovered["rationale"] = "Recovered by targeted evidence search. " + best_result.rationale
        recovered["verifier"] = f"{best_result.verifier}+v3_recovery"
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
    protected_spans = extract_protected_spans(str(item.get(args.output_field, "") or ""))
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
            updated["protected_answer_spans"] = claim_answer_spans(updated, protected_spans)
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
                recovered["protected_answer_spans"] = claim_answer_spans(recovered, protected_spans)
                recovered_claims.append(recovered)
                usable.append(recovered)
                continue

        rejected_claim = dict(claim)
        rejected_claim["v2_status"] = "rejected_after_recovery" if claim_should_recover(claim, args) else "not_recovery_eligible"
        rejected.append(rejected_claim)

    usable.sort(key=v2_claim_priority_key)
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
    citations = format_citations(claim.get("final_citation_ids", []), max_citations=args.max_citations_per_claim)
    if not claim_text or not citations:
        return ""
    return clean_final_answer(f"{claim_text} {citations}.", num_docs=num_docs)


def source_sentence_word_count(sentence: str) -> int:
    return len(re.findall(r"\b\w+\b", normalize_space(remove_citations(sentence))))


def unique_sentence_citations(
    sentence_record: dict[str, Any],
    kept_keys: Sequence[str],
    kept_by_key: dict[str, dict[str, Any]],
    args: argparse.Namespace | None = None,
) -> list[int]:
    """Choose citations for preserved source sentences.

    The answer-preserving renderer keeps source sentence wording for task
    metrics. Claim-minimal citations are best for claim precision, but ALCE's
    sentence-level AIS evaluates the whole sentence, so original sentence
    citations can be safer for preserved long/multi-claim sentences.
    """

    source_ids: list[int] = []
    verified_ids: list[int] = []

    def append_unique(target: list[int], values: Sequence[Any]) -> None:
        for value in values:
            try:
                doc_id = int(value)
            except Exception:
                continue
            if doc_id > 0 and doc_id not in target:
                target.append(doc_id)

    append_unique(source_ids, sentence_record.get("citation_ids", []) or [])
    if not source_ids:
        append_unique(source_ids, parse_citations(str(sentence_record.get("sentence") or "")))

    for key in kept_keys:
        claim = kept_by_key.get(key) or {}
        claim_citation_ids = (
            claim.get("final_citation_ids")
            or claim.get("evidence_doc_ids")
            or claim.get("citation_ids")
            or []
        )
        append_unique(verified_ids, claim_citation_ids)

    mode = str(getattr(args, "sentence_citation_source", "source") if args is not None else "source")
    if mode == "verified":
        return verified_ids or source_ids
    if mode == "hybrid":
        citation_ids = list(source_ids)
        append_unique(citation_ids, verified_ids)
        return citation_ids
    return source_ids or verified_ids


def sentence_has_answer_like_span(sentence_record: dict[str, Any], sentence_claims: Sequence[dict[str, Any]]) -> bool:
    source = normalize_space(str(sentence_record.get("sentence_without_citations") or ""))
    if not source:
        return False
    spans = extract_protected_spans(source)
    if not spans:
        return False
    important_claims = [
        claim for claim in sentence_claims if claim.get("importance") in {"critical", "supporting"}
    ]
    if len(spans) >= 2 and important_claims:
        return True
    if re.search(r"['\"].+?['\"]", source) and important_claims:
        return True
    for claim in sentence_claims:
        if claim.get("importance") in {"critical", "supporting"} and claim_answer_spans(claim, spans):
            return True
    return False


def should_preserve_source_sentence(
    sentence_record: dict[str, Any],
    sentence_claims: Sequence[dict[str, Any]],
    sentence_keys: Sequence[str],
    kept_sentence_keys: Sequence[str],
    args: argparse.Namespace,
) -> bool:
    """Decide whether preserving the draft wording is safer than atomizing.

    ASQA/ELI5 correctness is string-sensitive: rewriting "gold and silver" as
    two separate supported claims can lose the gold answer string even when
    grounding improves. This gate preserves short, answer-like source sentences
    when enough of their claim content survived verification.
    """

    if not getattr(args, "preserve_answer_sentences", True):
        return False
    if not sentence_keys or not kept_sentence_keys:
        return False
    source = normalize_space(str(sentence_record.get("sentence_without_citations") or ""))
    if not source:
        return False
    if source_sentence_word_count(source) > int(getattr(args, "max_preserved_source_words", 48)):
        return False
    kept_fraction = len(set(kept_sentence_keys)) / max(1, len(set(sentence_keys)))
    if kept_fraction >= float(getattr(args, "min_preserved_claim_fraction", 0.5)):
        return True
    return sentence_has_answer_like_span(sentence_record, sentence_claims)


def is_fragmented_source_sentence(source: str) -> bool:
    source = normalize_space(str(source or ""))
    if not source:
        return False
    if source_sentence_word_count(source) > 4:
        return False
    if re.fullmatch(r"(?:[A-Z]\.){1,5}", source):
        return True
    if source.endswith(("!", "?")) and extract_protected_spans(source):
        return True
    return False


def stitch_fragment_with_next(
    records: Sequence[dict[str, Any]],
    index: int,
) -> tuple[dict[str, Any], bool]:
    """Undo common sentence-tokenizer splits such as "W.C." / "Handy ..."."""

    record = dict(records[index])
    source = normalize_space(str(record.get("sentence_without_citations") or ""))
    if index + 1 >= len(records) or not is_fragmented_source_sentence(source):
        return record, False

    next_record = records[index + 1]
    next_source = normalize_space(str(next_record.get("sentence_without_citations") or ""))
    if not next_source:
        return record, False

    record["sentence_without_citations"] = normalize_space(f"{source} {next_source}")
    record["sentence"] = normalize_space(
        f"{record.get('sentence') or source} {next_record.get('sentence') or next_source}"
    )
    record["citation_ids"] = unique_positive_ints(
        list(record.get("citation_ids", []) or []) + list(next_record.get("citation_ids", []) or [])
    )
    record["claims"] = list(record.get("claims", []) or []) + list(next_record.get("claims", []) or [])
    return record, True


def should_preserve_rejected_critical_sentence(
    item: dict[str, Any],
    sentence_record: dict[str, Any],
    sentence_claims: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> bool:
    """Keep likely answer sentences when the verifier is probably over-strict."""

    if not getattr(args, "preserve_rejected_critical_answer_sentences", False):
        return False
    source = normalize_space(str(sentence_record.get("sentence_without_citations") or ""))
    if not source or source_sentence_word_count(source) > int(getattr(args, "max_preserved_source_words", 48)):
        return False
    if not unique_sentence_citations(sentence_record, [], {}, args):
        return False
    if not extract_protected_spans(source):
        return False
    if not any(claim.get("importance") in {"critical", "supporting"} for claim in sentence_claims):
        return False
    question = str(item.get(args.question_field, "") or "")
    min_overlap = float(getattr(args, "min_rejected_preserve_question_overlap", 0.25))
    return overlap_score(question, source) >= min_overlap


def render_answer_preserving_answer(
    item: dict[str, Any],
    draft_audit: dict[str, Any],
    verified_claims: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    """Preserve answer strings while keeping verified claim-level citations.

    This mode is less aggressive than citation_aligned. It still renders
    unsupported-only sentences away, but it avoids breaking short source
    sentences that carry answer spans, dates, names, or quoted strings.
    """

    if not verified_claims:
        return "The provided evidence is insufficient to answer the question."

    num_docs = len(item.get(args.docs_field, []) or [])
    kept_by_key = {
        normalize_revision_claim_key(claim, args): claim
        for claim in verified_claims
        if normalize_revision_claim_key(claim, args)
    }
    consumed: set[str] = set()
    sentences: list[str] = []
    rendered_claim_count = 0
    max_claims = final_claim_limit(args)

    records = list(draft_audit.get("sentences", []) or [])
    skip_next = False
    for record_index, raw_sentence_record in enumerate(records):
        if skip_next:
            skip_next = False
            continue
        if rendered_claim_count >= max_claims:
            break
        sentence_record, stitched_next = stitch_fragment_with_next(records, record_index)
        sentence_claims = [
            claim
            for claim in sentence_record.get("claims", []) or []
            if claim_is_render_eligible(claim, args)
        ]
        keys = list(
            dict.fromkeys(
                normalize_revision_claim_key(claim, args)
                for claim in sentence_claims
                if normalize_revision_claim_key(claim, args)
            )
        )
        kept_keys = [key for key in keys if key in kept_by_key and key not in consumed]
        if not kept_keys:
            if should_preserve_rejected_critical_sentence(item, sentence_record, sentence_claims, args):
                source = normalize_space(str(sentence_record.get("sentence_without_citations") or ""))
                citation_ids = unique_sentence_citations(sentence_record, [], kept_by_key, args)
                citations = format_citations(citation_ids, max_citations=args.max_citations_per_sentence)
                if source and citations and rendered_claim_count < max_claims:
                    sentences.append(clean_final_answer(f"{source.rstrip('.')} {citations}.", num_docs=num_docs))
                    rendered_claim_count += max(1, min(len(keys), max_claims - rendered_claim_count))
                    skip_next = stitched_next
                continue
            continue

        if should_preserve_source_sentence(sentence_record, sentence_claims, keys, kept_keys, args):
            source = normalize_space(str(sentence_record.get("sentence_without_citations") or ""))
            citation_ids = unique_sentence_citations(sentence_record, kept_keys, kept_by_key, args)
            citations = format_citations(citation_ids, max_citations=args.max_citations_per_sentence)
            if source and citations:
                sentences.append(clean_final_answer(f"{source.rstrip('.')} {citations}.", num_docs=num_docs))
                consumed.update(kept_keys)
                rendered_claim_count += len(kept_keys)
                skip_next = stitched_next
                continue

        for key in kept_keys:
            if rendered_claim_count >= max_claims or key in consumed:
                break
            atomic = render_claim_atomic(kept_by_key[key], args, num_docs)
            if atomic:
                sentences.append(atomic)
                consumed.add(key)
                rendered_claim_count += 1

    tail_claims = [claim for claim in verified_claims if normalize_revision_claim_key(claim, args) not in consumed]
    tail_claims.sort(key=v2_claim_priority_key)
    for claim in tail_claims:
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


def render_span_preserving_answer(
    item: dict[str, Any],
    draft_audit: dict[str, Any],
    verified_claims: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
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
    max_claims = final_claim_limit(args)

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

        if set(sentence_keys) == set(kept_sentence_keys) and rendered_claim_count + len(kept_sentence_keys) <= max_claims:
            citation_ids: list[int] = []
            for key in kept_sentence_keys:
                claim = kept_by_key[key]
                for doc_id in claim.get("final_citation_ids", []) or []:
                    try:
                        value = int(doc_id)
                    except Exception:
                        continue
                    if value > 0 and value not in citation_ids:
                        citation_ids.append(value)
            original = normalize_space(str(sentence_record.get("sentence_without_citations") or ""))
            citations = format_citations(citation_ids, max_citations=args.max_citations_per_claim)
            if original and citations:
                sentences.append(clean_final_answer(f"{original.rstrip('.')} {citations}.", num_docs=num_docs))
                consumed.update(kept_sentence_keys)
                rendered_claim_count += len(kept_sentence_keys)
                continue

        partial_claims = [kept_by_key[key] for key in kept_sentence_keys if key not in consumed]
        partial_claims.sort(key=v2_claim_priority_key)
        for claim in partial_claims:
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

    tail_claims = [claim for claim in verified_claims if normalize_revision_claim_key(claim, args) not in consumed]
    tail_claims.sort(key=v2_claim_priority_key)
    for claim in tail_claims:
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


def render_atomic_answer(
    item: dict[str, Any],
    verified_claims: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    """Write one factual claim per sentence for citation-level evaluation."""

    if not verified_claims:
        return "The provided evidence is insufficient to answer the question."
    num_docs = len(item.get(args.docs_field, []) or [])
    sentences: list[str] = []
    seen: set[str] = set()
    for claim in verified_claims[: final_claim_limit(args)]:
        key = normalize_revision_claim_key(claim, args)
        if not key or key in seen:
            continue
        atomic = render_claim_atomic(claim, args, num_docs)
        if atomic:
            sentences.append(atomic)
            seen.add(key)
    if not sentences:
        return "The provided evidence is insufficient to answer the question."
    return clean_final_answer(" ".join(sentences), num_docs=num_docs)


def render_citation_aligned_answer(
    item: dict[str, Any],
    draft_audit: dict[str, Any],
    verified_claims: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    """Balance exact answer spans with ALCE-style citation evaluation.

    A source sentence is preserved only when it contains exactly one retained
    atomic claim. For multi-claim source sentences, claims are rendered
    separately so a citation is never asked to entail unrelated facts.
    """

    if not verified_claims:
        return "The provided evidence is insufficient to answer the question."
    num_docs = len(item.get(args.docs_field, []) or [])
    kept_by_key = {
        normalize_revision_claim_key(claim, args): claim
        for claim in verified_claims
        if normalize_revision_claim_key(claim, args)
    }
    consumed: set[str] = set()
    sentences: list[str] = []
    rendered_claim_count = 0
    max_claims = final_claim_limit(args)

    for sentence_record in draft_audit.get("sentences", []) or []:
        sentence_claims = [
            claim
            for claim in sentence_record.get("claims", []) or []
            if claim_is_render_eligible(claim, args)
        ]
        keys = list(
            dict.fromkeys(
                normalize_revision_claim_key(claim, args)
                for claim in sentence_claims
                if normalize_revision_claim_key(claim, args)
            )
        )
        kept_keys = [key for key in keys if key in kept_by_key]
        if not kept_keys:
            continue

        if len(keys) == 1 and len(kept_keys) == 1 and rendered_claim_count < max_claims:
            key = kept_keys[0]
            claim = kept_by_key[key]
            source = normalize_space(str(sentence_record.get("sentence_without_citations") or ""))
            citations = format_citations(claim.get("final_citation_ids", []), max_citations=args.max_citations_per_claim)
            if source and citations:
                sentences.append(clean_final_answer(f"{source.rstrip('.')} {citations}.", num_docs=num_docs))
                consumed.add(key)
                rendered_claim_count += 1
                continue

        for key in kept_keys:
            if rendered_claim_count >= max_claims or key in consumed:
                break
            atomic = render_claim_atomic(kept_by_key[key], args, num_docs)
            if atomic:
                sentences.append(atomic)
                consumed.add(key)
                rendered_claim_count += 1

    tail_claims = [claim for claim in verified_claims if normalize_revision_claim_key(claim, args) not in consumed]
    for claim in tail_claims:
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


def final_claim_limit(args: argparse.Namespace) -> int:
    configured = int(getattr(args, "max_final_claims", 0) or 0)
    if configured > 0:
        return configured
    return verified_claim_limit(args) + max(0, int(getattr(args, "max_expanded_claims", 0) or 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COVER-RAG v3 iterative answer expansion for ALCE result JSON.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--answer-format", choices=["auto", "paragraph", "qampari"], default="auto")

    parser.add_argument("--output-field", default="output")
    parser.add_argument("--docs-field", default="docs")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--title-field", default="title")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--sent-field", default="sent")

    parser.add_argument("--candidate-docs-file", type=Path, default=None)
    parser.add_argument("--max-doc-pool", type=int, default=100)
    parser.add_argument(
        "--dynamic-retrieval-mode",
        choices=["off", "global_candidate_docs"],
        default="off",
        help=(
            "Retrieve additional evidence for missing facets before expansion. "
            "`global_candidate_docs` builds a lightweight lexical index over all "
            "docs in --candidate-docs-file and can fetch docs outside the current "
            "example's top100 candidate pool."
        ),
    )
    parser.add_argument("--dynamic-retrieval-top-k", type=int, default=12)
    parser.add_argument("--dynamic-retrieval-per-query-docs", type=int, default=4)
    parser.add_argument("--dynamic-retrieval-max-queries", type=int, default=4)
    parser.add_argument("--dynamic-retrieval-max-rejected-claims", type=int, default=4)
    parser.add_argument("--dynamic-retrieval-max-doc-pool", type=int, default=140)
    parser.add_argument("--dynamic-retrieval-query-max-chars", type=int, default=900)
    parser.add_argument("--dynamic-retrieval-min-query-terms", type=int, default=2)
    parser.add_argument("--dynamic-retrieval-sentence-boost", type=float, default=0.12)
    parser.add_argument(
        "--dynamic-retrieval-query-strategy",
        choices=["missing_facet", "question_facet", "hybrid"],
        default="missing_facet",
        help=(
            "How to build dynamic retrieval queries. `missing_facet` uses only "
            "rejected/explicit recall targets; `question_facet` also searches "
            "with the original question when no explicit missing target remains; "
            "`hybrid` does both."
        ),
    )
    parser.add_argument(
        "--dynamic-retrieval-min-score",
        type=float,
        default=0.0,
        help="Minimum lexical score for docs returned by dynamic retrieval.",
    )
    parser.add_argument(
        "--dynamic-retrieval-require-missing-signal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only run dynamic retrieval when rejected claims or explicit recall targets exist.",
    )
    parser.add_argument(
        "--dynamic-retrieval-use-supported-answer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the current supported answer summary in missing-facet retrieval queries.",
    )
    parser.add_argument(
        "--coverage-guided-retrieval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use claim-coverage gaps as the controller for dynamic retrieval. "
            "When enabled, retrieval is triggered only by uncovered answer-unit "
            "targets rather than broad question-level queries."
        ),
    )
    parser.add_argument("--coverage-guided-targets-per-round", type=int, default=3)
    parser.add_argument("--coverage-guided-min-targets", type=int, default=1)
    parser.add_argument("--coverage-guided-min-question-overlap", type=float, default=0.08)
    parser.add_argument("--coverage-guided-query-supported-claims", type=int, default=6)
    parser.add_argument(
        "--coverage-guided-allow-evidence-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow high-scoring evidence-derived answer-unit targets, not only rejected draft claims.",
    )
    parser.add_argument(
        "--coverage-reranker-model",
        type=Path,
        default=None,
        help=(
            "Optional trained coverage-aware cross encoder. When set, dynamic "
            "retrieval candidates and expansion candidates are reranked by "
            "answer-unit usefulness rather than lexical relevance alone."
        ),
    )
    parser.add_argument("--coverage-reranker-device", default=None)
    parser.add_argument("--coverage-reranker-batch-size", type=int, default=16)
    parser.add_argument("--coverage-reranker-max-length", type=int, default=384)
    parser.add_argument(
        "--coverage-reranker-doc-prefilter",
        type=int,
        default=64,
        help="Lexical dynamic-retrieval docs scored by the coverage-aware reranker before top-k selection.",
    )
    parser.add_argument(
        "--coverage-reranker-doc-weight",
        type=float,
        default=0.35,
        help="Interpolation weight for reranker score in dynamic-retrieval doc ranking.",
    )
    parser.add_argument(
        "--coverage-reranker-candidate-weight",
        type=float,
        default=0.20,
        help="Interpolation weight for reranker score in expansion answer-unit scoring.",
    )

    parser.add_argument("--openai-api", action="store_true")
    parser.add_argument("--decomposer", choices=["llm", "sentence", "claimify"], default="llm")
    parser.add_argument("--decompose-model", default="gpt-4o-mini")
    parser.add_argument("--llm-verify-model", default="gpt-4o-mini")
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-top-p", type=float, default=1.0)
    parser.add_argument("--llm-max-retries", type=int, default=5)

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
    parser.add_argument("--nli-model", default=NLI_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--nli-batch-size", type=int, default=8)
    parser.add_argument("--nli-max-length", type=int, default=512)
    parser.add_argument("--nli-premise-mode", choices=["doc", "sentence_window", "doc_then_sentence"], default="doc")
    parser.add_argument("--nli-top-sentences", type=int, default=4)
    parser.add_argument("--nli-sentence-window-size", type=int, default=1)
    parser.add_argument("--entail-threshold", type=float, default=0.50)
    parser.add_argument("--contradiction-threshold", type=float, default=0.50)
    parser.add_argument("--ambiguous-margin", type=float, default=0.10)
    parser.add_argument("--nli-aggregation", choices=["joint", "max_doc", "joint_then_max"], default="joint_then_max")
    parser.add_argument("--evidence-scope", choices=["cited", "all_docs", "cited_then_all"], default="cited")
    parser.add_argument("--max-all-docs", type=int, default=20)
    parser.add_argument("--max-evidence-chars", type=int, default=6000)
    parser.add_argument("--lexical-threshold", type=float, default=0.45)

    parser.add_argument("--recover-labels", default="not_supported,no_citation,wrong_or_missing_citation")
    parser.add_argument("--recover-importance", default="critical,supporting")
    parser.add_argument("--recovery-top-k", type=int, default=6)
    parser.add_argument("--recovery-group-size", type=int, default=3)
    parser.add_argument("--min-recovery-score", type=float, default=0.03)
    parser.add_argument("--max-recovered-citations", type=int, default=3)
    parser.add_argument("--keep-wrong-missing-if-recovered", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--expansion-mode",
        choices=["llm", "extractive", "off"],
        default="llm",
        help="llm extracts answer-expansion claims from ranked evidence sentences; extractive uses deterministic sentence candidates.",
    )
    parser.add_argument("--expansion-model", default="gpt-4o-mini")
    parser.add_argument("--expansion-max-tokens", type=int, default=900)
    parser.add_argument(
        "--expansion-output-mode",
        choices=["claims", "span_ids"],
        default="claims",
        help=(
            "`claims` asks the LLM to write candidate claims. `span_ids` asks "
            "the LLM only to select evidence span ids; code then renders and "
            "verifies the claim. The span-id mode is more stable for answer "
            "recall and avoids JSON/boilerplate drift."
        ),
    )
    parser.add_argument("--expansion-evidence-sentences", type=int, default=12)
    parser.add_argument("--max-expansion-candidates", type=int, default=10)
    parser.add_argument("--max-expansion-verifications", type=int, default=8)
    parser.add_argument("--max-expanded-claims", type=int, default=6)
    parser.add_argument("--min-expansion-score", type=float, default=0.08)
    parser.add_argument("--min-answer-unit-score", type=float, default=0.25)
    parser.add_argument(
        "--strict-expansion-answer-unit-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop expansion candidates that are supported but look unlike missing gold answer units.",
    )
    parser.add_argument(
        "--strict-expansion-min-target-relevance",
        type=float,
        default=0.45,
        help="When strict gate is enabled, require this target relevance unless the selected span exactly matches a target.",
    )
    parser.add_argument(
        "--answer-type-gate-mode",
        choices=["off", "soft", "strict"],
        default="soft",
        help=(
            "How aggressively to use question-answer-type heuristics for ASQA/ELI5 expansion. "
            "`soft` only hard-drops obvious mismatches and otherwise applies a scoring penalty."
        ),
    )
    parser.add_argument("--max-expansion-claim-words", type=int, default=22)
    parser.add_argument(
        "--expansion-candidate-spans",
        type=int,
        default=40,
        help="Maximum answer-like spans exposed to the LLM expansion prompt.",
    )
    parser.add_argument(
        "--max-selected-span-ids",
        type=int,
        default=8,
        help=(
            "In span-id expansion mode, ask the LLM for at most this many span ids "
            "and truncate any overlong or salvaged selection to this budget."
        ),
    )
    parser.add_argument(
        "--expansion-include-extractive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic evidence-sentence candidates only when the LLM finds no candidate.",
    )
    parser.add_argument(
        "--expansion-filter-general-answer-spans",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For ASQA/ELI5 expansion, drop generic answer spans and spans copied from the question.",
    )
    parser.add_argument(
        "--expansion-min-question-overlap",
        type=float,
        default=0.0,
        help="Minimum lexical overlap between the question and either an expansion claim or its evidence sentence.",
    )
    parser.add_argument(
        "--verify-expansion-with-sentence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify expansion claims against their evidence sentence snippet to avoid long-document NLI truncation.",
    )
    parser.add_argument("--skip-covered-answer-spans", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require-expansion-answer-span",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only verify/add expansion candidates that expose an answer-like surface span.",
    )
    parser.add_argument(
        "--expansion-query-source",
        choices=["question", "question_and_evidence_core", "question_evidence_and_rejected"],
        default="question_and_evidence_core",
    )
    parser.add_argument(
        "--expansion-use-intermediate-answer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Pass a compact answer rendered from verified claims into the expansion "
            "prompt so the second pass can search for missing answer units."
        ),
    )
    parser.add_argument(
        "--answer-targeted-expansion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Bias expansion toward main-task answer units. For paragraph tasks, "
            "also mine compact causal/mechanism phrases from evidence, merge "
            "extractive candidates with LLM candidates, and still require NLI support."
        ),
    )
    parser.add_argument(
        "--answer-targeted-extractive-top-k",
        type=int,
        default=2,
        help=(
            "For paragraph answer-targeted expansion, cap deterministic "
            "evidence-sentence fallback candidates so they do not crowd out LLM candidates."
        ),
    )
    parser.add_argument(
        "--answer-targeted-min-sentence-overlap",
        type=float,
        default=0.16,
        help=(
            "For paragraph answer-targeted extractive candidates, require this "
            "minimum question/evidence sentence lexical overlap."
        ),
    )
    parser.add_argument(
        "--explanatory-expansion-require-phrase-span",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For explanatory paragraph questions, only expose compact causal/mechanism "
            "phrases or numeric spans as expansion answer spans. This avoids adding "
            "topic-related named entities that hurt ELI5-style main metrics."
        ),
    )
    parser.add_argument(
        "--explanatory-title-span-mode",
        choices=["off", "on"],
        default="off",
        help=(
            "Whether doc-title answer-span candidates are allowed for explanatory "
            "questions. Off keeps ELI5-style expansion from adding supported but "
            "non-gold topic/entity facts."
        ),
    )
    parser.add_argument(
        "--max-explanatory-expanded-claims",
        type=int,
        default=2,
        help="For explanatory questions, cap added expansion claims to this many.",
    )
    parser.add_argument(
        "--max-explanatory-expansion-verifications",
        type=int,
        default=4,
        help="For explanatory questions, cap NLI verification attempts for expansion candidates.",
    )
    parser.add_argument(
        "--min-explanatory-answer-unit-score",
        type=float,
        default=0.35,
        help="For explanatory questions, require at least this answer-unit score.",
    )
    parser.add_argument(
        "--explanatory-precision-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For explanatory questions, require added claims to look like core "
            "mechanism/reason/definition answer units rather than supported background."
        ),
    )
    parser.add_argument(
        "--explanatory-precision-min-target-relevance",
        type=float,
        default=0.58,
        help="Minimum target relevance used by --explanatory-precision-gate.",
    )
    parser.add_argument(
        "--explanatory-precision-min-question-relevance",
        type=float,
        default=0.18,
        help="Minimum question/evidence relevance used by --explanatory-precision-gate.",
    )
    parser.add_argument(
        "--explanation-integrated-refinement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For free-form explanatory answers, render the first-pass supported "
            "answer first and only integrate a tiny high-utility expansion budget. "
            "This avoids improving support by appending extra supported background "
            "that dilutes main-task correctness."
        ),
    )
    parser.add_argument(
        "--max-explanation-integrated-claims",
        type=int,
        default=1,
        help="Maximum expansion claims allowed into an explanatory final answer.",
    )
    parser.add_argument(
        "--max-explanation-length-growth",
        type=float,
        default=0.10,
        help="For explanatory refinement, cap added answer words relative to the base rendered answer.",
    )
    parser.add_argument(
        "--candidate-output-selection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Generate a few citation-grounded final-answer candidates and choose "
            "the best one with a final-audit proxy. This is answer-type adaptive "
            "rather than dataset-specific."
        ),
    )
    parser.add_argument(
        "--candidate-output-recall-claims",
        type=int,
        default=2,
        help="Maximum expansion claims in the recall-heavy candidate for explanation-form answers.",
    )
    parser.add_argument(
        "--candidate-output-recall-growth",
        type=float,
        default=0.18,
        help="Length growth budget for the recall-heavy candidate.",
    )
    parser.add_argument(
        "--candidate-output-min-support",
        type=float,
        default=0.94,
        help="Minimum final-audit claim support rate for a candidate to be eligible.",
    )
    parser.add_argument(
        "--candidate-output-support-tolerance",
        type=float,
        default=0.015,
        help="Allow candidates within this support-rate gap from the best candidate.",
    )
    parser.add_argument(
        "--candidate-output-max-unsupported-rate",
        type=float,
        default=0.06,
        help="Maximum final-audit unsupported-claim rate for a candidate to be eligible.",
    )
    parser.add_argument(
        "--candidate-output-min-answer-unit-gain",
        type=float,
        default=0.0,
        help=(
            "For integrated/recall-heavy candidates, require at least this much "
            "answer-unit utility before replacing the current output."
        ),
    )
    parser.add_argument(
        "--candidate-output-min-supported-claim-gain",
        type=int,
        default=0,
        help=(
            "For integrated/recall-heavy candidates, require this many additional "
            "supported claims over the current output before replacing it."
        ),
    )
    parser.add_argument(
        "--recall-oriented-completion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Treat the supported intermediate answer as partial coverage and "
            "only verify expansion candidates that fill an explicit missing "
            "answer target from rejected claims or ranked evidence."
        ),
    )
    parser.add_argument("--recall-completion-max-targets", type=int, default=10)
    parser.add_argument("--recall-completion-rejected-targets", type=int, default=4)
    parser.add_argument("--recall-completion-evidence-sentences", type=int, default=24)
    parser.add_argument("--recall-completion-supported-claims", type=int, default=18)
    parser.add_argument("--recall-completion-min-sentence-overlap", type=float, default=0.0)
    parser.add_argument("--recall-completion-min-target-overlap", type=float, default=0.10)
    parser.add_argument(
        "--answer-unit-relevance-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Require expansion candidates to look like direct answer units, not "
            "merely supported evidence facts. This is stricter than NLI support "
            "and is intended to protect main-task metrics during recall completion."
        ),
    )
    parser.add_argument(
        "--answer-unit-min-directness",
        type=float,
        default=0.44,
        help="Minimum direct-answer score for non-explanatory expansion candidates.",
    )
    parser.add_argument(
        "--explanatory-answer-unit-min-directness",
        type=float,
        default=0.58,
        help="Minimum direct-answer score for explanatory/ELI5-style expansion candidates.",
    )
    parser.add_argument(
        "--iterative-completion-rounds",
        type=int,
        default=0,
        help=(
            "Run additional supported-answer -> missing-target -> dynamic evidence reranking "
            "rounds after the first expansion pass. Each round reuses the current verified "
            "claims as partial coverage and searches the doc pool for still-missing answer units."
        ),
    )
    parser.add_argument(
        "--iterative-completion-max-new-claims-per-round",
        type=int,
        default=2,
        help="Maximum newly verified expansion claims accepted in each iterative completion round.",
    )
    parser.add_argument(
        "--iterative-completion-max-verifications-per-round",
        type=int,
        default=4,
        help="Maximum expansion candidates verified in each iterative completion round.",
    )
    parser.add_argument(
        "--iterative-completion-evidence-sentences",
        type=int,
        default=0,
        help="Override --expansion-evidence-sentences inside iterative rounds when > 0.",
    )
    parser.add_argument(
        "--iterative-completion-candidate-spans",
        type=int,
        default=0,
        help="Override --expansion-candidate-spans inside iterative rounds when > 0.",
    )
    parser.add_argument(
        "--iterative-completion-max-candidates",
        type=int,
        default=0,
        help="Override --max-expansion-candidates inside iterative rounds when > 0.",
    )

    parser.add_argument("--revision-mode", choices=["auto"], default="auto")
    parser.add_argument("--max-verified-claims", type=int, default=24)
    parser.add_argument("--max-final-claims", type=int, default=0)
    parser.add_argument("--preserve-first-pass-claims", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--render-mode",
        choices=["citation_aligned", "span_preserving", "answer_preserving", "atomic"],
        default="citation_aligned",
    )
    parser.add_argument("--preserve-answer-sentences", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preserve-rejected-critical-answer-sentences", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-preserved-source-words", type=int, default=48)
    parser.add_argument("--min-preserved-claim-fraction", type=float, default=0.50)
    parser.add_argument("--min-rejected-preserve-question-overlap", type=float, default=0.25)
    parser.add_argument("--max-qampari-items", type=int, default=80)
    parser.add_argument(
        "--qampari-answer-relevance-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For QAMPARI, drop answer items that are generic, copied from the question, or not entity-like.",
    )
    parser.add_argument("--citation-policy", choices=["source", "all", "minimal"], default="source")
    parser.add_argument(
        "--sentence-citation-source",
        choices=["source", "verified", "hybrid"],
        default="source",
        help=(
            "Citation source for answer-preserving source sentences. `source` "
            "keeps original sentence citations for ALCE sentence-level AIS; "
            "`verified` uses claim-level final citations; `hybrid` appends "
            "verified citations after source citations."
        ),
    )
    parser.add_argument("--max-citations-per-claim", type=int, default=3)
    parser.add_argument("--max-citations-per-sentence", type=int, default=4)
    parser.add_argument("--include-background-claims", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--abstain-if-no-supported", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fallback-template-on-empty", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--final-audit", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--cache-file", type=Path, default=None)
    parser.add_argument(
        "--cache-save-every",
        type=int,
        default=1,
        help="Flush the JSON cache every N new entries. Increase this when running many workers to reduce disk I/O.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of examples to process concurrently. Use 1 for the original sequential behavior.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help=(
            "Write <output>.checkpoint.json every N processed examples. "
            "Useful for long ASQA runs so progress is inspectable before the final write."
        ),
    )
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def truncate_text(text: str, max_chars: int) -> str:
    text = normalize_space(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def supported_claim_block(claims: Sequence[dict[str, Any]], limit: int = 12) -> str:
    lines = []
    for idx, claim in enumerate(claims[:limit], start=1):
        citations = format_citations(claim.get("final_citation_ids", []), max_citations=3)
        lines.append(f"{idx}. {claim.get('claim', '')} {citations}".strip())
    return "\n".join(lines) or "None."


def supported_answer_summary(claims: Sequence[dict[str, Any]], limit: int = 18, max_chars: int = 1800) -> str:
    """Render verified claims as a compact intermediate answer for expansion."""

    sentences: list[str] = []
    seen: set[str] = set()
    for claim in claims[:limit]:
        text = normalize_space(str(claim.get("claim", "") or ""))
        if not text:
            continue
        key = normalize_claim_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        sentences.append(text.rstrip(".") + ".")
    return truncate_text(" ".join(sentences), max_chars) or "None."


def rejected_claim_block(claims: Sequence[dict[str, Any]], limit: int = 10) -> str:
    lines = []
    for idx, claim in enumerate(claims[:limit], start=1):
        lines.append(f"{idx}. {claim.get('claim', '')} ({claim.get('label', 'unknown')})")
    return "\n".join(lines) or "None."


def protected_span_key(span: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(span or "").lower()).strip()


def expansion_claim_word_limit(args: argparse.Namespace) -> int:
    return max(8, int(getattr(args, "max_expansion_claim_words", 22) or 22))


def max_selected_span_ids(args: argparse.Namespace) -> int:
    return max(1, int(getattr(args, "max_selected_span_ids", 8) or 8))


def explanatory_max_expanded_claims(args: argparse.Namespace) -> int:
    configured = int(getattr(args, "max_explanatory_expanded_claims", 0) or 0)
    if configured > 0:
        return configured
    return int(getattr(args, "max_expanded_claims", 6) or 6)


def min_answer_unit_score_for_question(args: argparse.Namespace, question: str) -> float:
    base = float(getattr(args, "min_answer_unit_score", 0.25) or 0.25)
    if question_answer_type(question) == "explanation":
        return max(base, float(getattr(args, "min_explanatory_answer_unit_score", base) or base))
    return base


def max_expanded_claims_for_question(args: argparse.Namespace, question: str) -> int:
    base = max(0, int(getattr(args, "max_expanded_claims", 0) or 0))
    if question_answer_type(question) == "explanation":
        return min(base, max(0, explanatory_max_expanded_claims(args)))
    return base


def max_expansion_verifications_for_question(args: argparse.Namespace, question: str) -> int:
    base = max(1, int(getattr(args, "max_expansion_verifications", 1) or 1))
    if question_answer_type(question) == "explanation":
        configured = int(getattr(args, "max_explanatory_expansion_verifications", 0) or 0)
        if configured > 0:
            return min(base, configured)
    return base


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", normalize_space(remove_citations(text))))


def is_explanation_refinement_candidate(item: dict[str, Any], args: argparse.Namespace) -> bool:
    question = str(item.get(args.question_field, "") or "")
    return infer_answer_format(args) != "qampari" and question_answer_type(question) == "explanation"


def claim_is_expansion(claim: dict[str, Any]) -> bool:
    return (
        str(claim.get("v2_status", "") or "") == "expanded"
        or str(claim.get("v3_status", "") or "") == "expanded_supported"
        or str(claim.get("claim_type", "") or "") == "answer_expansion"
    )


def claim_float_feature(claim: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(claim.get(key, default) or default)
    except Exception:
        return default


def explanation_integrated_claim_budget(args: argparse.Namespace) -> int:
    return max(0, int(getattr(args, "max_explanation_integrated_claims", 1) or 0))


def explanation_length_growth_budget(
    base_output: str,
    draft_output: str,
    args: argparse.Namespace,
    growth_override: float | None = None,
) -> int:
    growth = (
        max(0.0, float(growth_override))
        if growth_override is not None
        else max(0.0, float(getattr(args, "max_explanation_length_growth", 0.10) or 0.0))
    )
    base_words = word_count(base_output)
    draft_words = max(word_count(draft_output), base_words)
    # At least one short factual sentence can fit, but long background additions cannot.
    return base_words + max(10, int(draft_words * growth))


def select_explanation_integrated_claims(
    item: dict[str, Any],
    expanded_claims: Sequence[dict[str, Any]],
    base_output: str,
    args: argparse.Namespace,
    budget_override: int | None = None,
    min_score_override: float | None = None,
) -> list[dict[str, Any]]:
    budget = max(0, int(budget_override)) if budget_override is not None else explanation_integrated_claim_budget(args)
    if budget <= 0:
        return []
    question = str(item.get(args.question_field, "") or "")
    threshold = (
        float(min_score_override)
        if min_score_override is not None
        else min_answer_unit_score_for_question(args, question)
    )
    base_key_text = normalize_claim_key(base_output)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    candidates = sorted(
        expanded_claims,
        key=lambda claim: (
            -claim_float_feature(claim, "v3_answer_unit_directness"),
            -claim_float_feature(claim, "v3_answer_unit_score"),
            -claim_float_feature(claim, "v3_question_overlap"),
            v2_claim_priority_key(claim),
        ),
    )
    for claim in candidates:
        if len(selected) >= budget:
            break
        text = normalize_space(str(claim.get("claim", "") or ""))
        key = normalize_claim_key(text)
        if not text or not key or key in seen:
            continue
        if key and base_key_text and key in base_key_text:
            continue
        if claim_float_feature(claim, "v3_answer_unit_score") < threshold:
            continue
        if getattr(args, "answer_unit_relevance_gate", False):
            min_directness = float(
                getattr(args, "explanatory_answer_unit_min_directness", threshold) or threshold
            )
            if claim_float_feature(claim, "v3_answer_unit_directness") < min_directness:
                continue
        if not claim.get("final_citation_ids"):
            continue
        selected.append(claim)
        seen.add(key)
    return selected


def expansion_answer_unit_score_sum(claims: Sequence[dict[str, Any]]) -> float:
    return sum(claim_float_feature(claim, "v3_answer_unit_score") for claim in claims)


def split_base_and_expansion_claims(
    verified_claims: Sequence[dict[str, Any]],
    expanded_claims: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expanded_keys = {
        normalize_revision_claim_key(claim, args)
        for claim in expanded_claims
        if normalize_revision_claim_key(claim, args)
    }
    base_claims: list[dict[str, Any]] = []
    explicit_expanded: list[dict[str, Any]] = []
    for claim in verified_claims:
        key = normalize_revision_claim_key(claim, args)
        if key and (key in expanded_keys or claim_is_expansion(claim)):
            explicit_expanded.append(claim)
        else:
            base_claims.append(claim)
    for claim in expanded_claims:
        key = normalize_revision_claim_key(claim, args)
        if key and all(normalize_revision_claim_key(existing, args) != key for existing in explicit_expanded):
            explicit_expanded.append(claim)
    return base_claims, explicit_expanded


def render_explanation_integrated_answer(
    item: dict[str, Any],
    draft_audit: dict[str, Any],
    verified_claims: Sequence[dict[str, Any]],
    expanded_claims: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]] | None:
    if not getattr(args, "explanation_integrated_refinement", False):
        return None
    if not is_explanation_refinement_candidate(item, args):
        return None

    base_claims, expansion_pool = split_base_and_expansion_claims(verified_claims, expanded_claims, args)
    base_claims = base_claims or list(verified_claims)
    base_output = render_answer_preserving_answer(item, draft_audit, base_claims, args)
    selected_expanded = select_explanation_integrated_claims(item, expansion_pool, base_output, args)
    diagnostics: dict[str, Any] = {
        "mode": "explanation_integrated_refinement",
        "base_claims": len(base_claims),
        "expansion_pool": len(expansion_pool),
        "selected_expansion_claims": len(selected_expanded),
        "length_limited_expansion_claims": 0,
    }
    if not selected_expanded:
        return base_output, diagnostics

    candidate_output = render_answer_preserving_answer(item, draft_audit, list(base_claims) + selected_expanded, args)
    max_words = explanation_length_growth_budget(base_output, str(item.get(args.output_field, "") or ""), args)
    if word_count(candidate_output) > max_words:
        diagnostics["length_limited_expansion_claims"] = len(selected_expanded)
        diagnostics["selected_expansion_claims"] = 0
        diagnostics["max_words"] = max_words
        diagnostics["candidate_words"] = word_count(candidate_output)
        return base_output, diagnostics

    diagnostics["max_words"] = max_words
    diagnostics["candidate_words"] = word_count(candidate_output)
    return candidate_output, diagnostics


def fit_explanation_candidate_output(
    label: str,
    item: dict[str, Any],
    draft_audit: dict[str, Any],
    base_claims: Sequence[dict[str, Any]],
    selected_expanded: Sequence[dict[str, Any]],
    base_output: str,
    growth: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    kept_expanded = list(selected_expanded)
    max_words = explanation_length_growth_budget(base_output, str(item.get(args.output_field, "") or ""), args, growth)
    while True:
        claim_list = list(base_claims) + kept_expanded
        output = render_answer_preserving_answer(item, draft_audit, claim_list, args)
        output = clean_final_answer(output, num_docs=len(item.get(args.docs_field, []) or []))
        candidate_words = word_count(output)
        if candidate_words <= max_words or not kept_expanded:
            break
        kept_expanded = kept_expanded[:-1]
    if not kept_expanded and selected_expanded:
        output = base_output
        candidate_words = word_count(output)
    return {
        "label": label,
        "output": output,
        "included_expansion_claims": kept_expanded,
        "answer_unit_score_sum": expansion_answer_unit_score_sum(kept_expanded),
        "length_limited_expansion_claims": max(0, len(selected_expanded) - len(kept_expanded)),
        "candidate_words": candidate_words,
        "max_words": max_words,
    }


def candidate_output_selection_applies(item: dict[str, Any], args: argparse.Namespace) -> bool:
    if not getattr(args, "candidate_output_selection", False):
        return False
    if not getattr(args, "final_audit", False):
        return False
    if args.render_mode != "answer_preserving":
        return False
    return is_explanation_refinement_candidate(item, args)


def output_candidate_key(output: str) -> str:
    return normalize_claim_key(remove_citations(output))


def build_output_candidates(
    item: dict[str, Any],
    draft_audit: dict[str, Any],
    verified_claims: Sequence[dict[str, Any]],
    expanded_claims: Sequence[dict[str, Any]],
    current_output: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [
        {
            "label": "current",
            "output": current_output,
            "included_expansion_claims": [],
            "answer_unit_score_sum": 0.0,
            "length_limited_expansion_claims": 0,
            "candidate_words": word_count(current_output),
            "max_words": None,
        }
    ]
    if not candidate_output_selection_applies(item, args):
        return candidates

    base_claims, expansion_pool = split_base_and_expansion_claims(verified_claims, expanded_claims, args)
    base_claims = base_claims or list(verified_claims)
    base_output = clean_final_answer(
        render_answer_preserving_answer(item, draft_audit, base_claims, args),
        num_docs=len(item.get(args.docs_field, []) or []),
    )
    candidates.append(
        {
            "label": "base_preserving",
            "output": base_output,
            "included_expansion_claims": [],
            "answer_unit_score_sum": 0.0,
            "length_limited_expansion_claims": 0,
            "candidate_words": word_count(base_output),
            "max_words": word_count(base_output),
        }
    )

    integrated_claims = select_explanation_integrated_claims(item, expansion_pool, base_output, args)
    if integrated_claims:
        candidates.append(
            fit_explanation_candidate_output(
                "integrated",
                item,
                draft_audit,
                base_claims,
                integrated_claims,
                base_output,
                float(getattr(args, "max_explanation_length_growth", 0.10) or 0.10),
                args,
            )
        )

    recall_budget = max(
        explanation_integrated_claim_budget(args),
        int(getattr(args, "candidate_output_recall_claims", 2) or 0),
    )
    if recall_budget > 0:
        recall_claims = select_explanation_integrated_claims(
            item,
            expansion_pool,
            base_output,
            args,
            budget_override=recall_budget,
        )
        if recall_claims:
            candidates.append(
                fit_explanation_candidate_output(
                    "recall_heavy",
                    item,
                    draft_audit,
                    base_claims,
                    recall_claims,
                    base_output,
                    float(getattr(args, "candidate_output_recall_growth", 0.18) or 0.18),
                    args,
                )
            )

    deduped_by_key: dict[str, dict[str, Any]] = {}
    output_order: list[str] = []
    num_docs = len(item.get(args.docs_field, []) or [])
    for candidate in candidates:
        output = clean_final_answer(str(candidate.get("output", "") or ""), num_docs=num_docs)
        if not output_has_valid_citation(output, num_docs):
            continue
        key = output_candidate_key(output)
        if not key:
            continue
        candidate = dict(candidate)
        candidate["output"] = output
        existing = deduped_by_key.get(key)
        if existing is None:
            deduped_by_key[key] = candidate
            output_order.append(key)
            continue
        existing_score = float(existing.get("answer_unit_score_sum", 0.0) or 0.0)
        candidate_score = float(candidate.get("answer_unit_score_sum", 0.0) or 0.0)
        if candidate_score > existing_score:
            deduped_by_key[key] = candidate
    deduped = [deduped_by_key[key] for key in output_order]
    return deduped or candidates[:1]


def audit_output_candidate(
    item: dict[str, Any],
    item_id: int,
    output: str,
    decomposer: Any,
    verifier: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidate_item = dict(item)
    candidate_item[args.output_field] = output
    return audit_item(candidate_item, item_id, decomposer, verifier, args)


def candidate_metric_float(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(metrics.get(key, default) or default)
    except Exception:
        return default


def score_output_candidate(
    candidate: dict[str, Any],
    audit: dict[str, Any],
    base_words: int,
    best_support: float,
    current_supported_claims: int,
    args: argparse.Namespace,
) -> tuple[float, bool, dict[str, Any]]:
    metrics = audit.get("metrics", {}) or {}
    support_rate = candidate_metric_float(metrics, "claim_support_rate")
    unsupported_rate = candidate_metric_float(metrics, "unsupported_claim_rate")
    citation_coverage = candidate_metric_float(metrics, "citation_coverage_rate")
    num_claims = int(metrics.get("num_claims", 0) or 0)
    supported_claims = int(metrics.get("supported_claims", 0) or 0)
    unsupported_claims = int(metrics.get("unsupported_claims", 0) or 0)
    no_citation_claims = int(metrics.get("no_citation_claims", 0) or 0)
    words = int(candidate.get("candidate_words", 0) or word_count(str(candidate.get("output", "") or "")))
    length_growth = max(0.0, (words - max(1, base_words)) / max(1, base_words))
    answer_unit_score = float(candidate.get("answer_unit_score_sum", 0.0) or 0.0)
    min_support = float(getattr(args, "candidate_output_min_support", 0.94) or 0.94)
    support_tolerance = float(getattr(args, "candidate_output_support_tolerance", 0.015) or 0.015)
    max_unsupported = float(getattr(args, "candidate_output_max_unsupported_rate", 0.06) or 0.06)
    label = str(candidate.get("label", "unknown") or "unknown")
    expansion_candidate = label not in {"current", "base_preserving"}
    min_answer_unit_gain = float(getattr(args, "candidate_output_min_answer_unit_gain", 0.0) or 0.0)
    min_supported_gain = int(getattr(args, "candidate_output_min_supported_claim_gain", 0) or 0)
    answer_unit_gain_ok = (not expansion_candidate) or answer_unit_score >= min_answer_unit_gain
    supported_claim_gain = supported_claims - int(current_supported_claims or 0)
    supported_gain_ok = (not expansion_candidate) or supported_claim_gain >= min_supported_gain
    eligible = (
        num_claims > 0
        and support_rate >= min_support
        and support_rate >= best_support - support_tolerance
        and unsupported_rate <= max_unsupported
        and citation_coverage >= 0.95
        and answer_unit_gain_ok
        and supported_gain_ok
    )
    score = (
        4.0 * answer_unit_score
        + 3.0 * support_rate
        + 0.35 * citation_coverage
        + 0.03 * min(supported_claims, 20)
        - 3.5 * unsupported_rate
        - 0.22 * unsupported_claims
        - 0.18 * no_citation_claims
        - 1.15 * length_growth
    )
    summary = {
        "label": label,
        "eligible": eligible,
        "score": score,
        "support_rate": support_rate,
        "unsupported_rate": unsupported_rate,
        "citation_coverage_rate": citation_coverage,
        "num_claims": num_claims,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "no_citation_claims": no_citation_claims,
        "answer_unit_score_sum": answer_unit_score,
        "answer_unit_gain_ok": answer_unit_gain_ok,
        "min_answer_unit_gain": min_answer_unit_gain,
        "supported_claim_gain": supported_claim_gain,
        "supported_gain_ok": supported_gain_ok,
        "min_supported_claim_gain": min_supported_gain,
        "candidate_words": words,
        "length_growth": length_growth,
        "length_limited_expansion_claims": int(candidate.get("length_limited_expansion_claims", 0) or 0),
    }
    return score, eligible, summary


def select_best_output_candidate(
    item: dict[str, Any],
    item_id: int,
    draft_audit: dict[str, Any],
    verified_claims: Sequence[dict[str, Any]],
    expanded_claims: Sequence[dict[str, Any]],
    current_output: str,
    decomposer: Any,
    verifier: Any,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    candidates = build_output_candidates(item, draft_audit, verified_claims, expanded_claims, current_output, args)
    diagnostics: dict[str, Any] = {
        "enabled": bool(getattr(args, "candidate_output_selection", False)),
        "num_candidates": len(candidates),
        "selected_label": "current",
        "changed_output": False,
        "audit_failures": 0,
        "candidates": [],
    }
    if len(candidates) <= 1:
        return current_output, diagnostics, None

    audited: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for candidate in candidates:
        try:
            audit = audit_output_candidate(item, item_id, str(candidate.get("output", "") or ""), decomposer, verifier, args)
        except Exception as exc:
            diagnostics["audit_failures"] = int(diagnostics.get("audit_failures", 0)) + 1
            diagnostics["candidates"].append(
                {
                    "label": candidate.get("label", "unknown"),
                    "eligible": False,
                    "audit_error": truncate_text(str(exc), 240),
                }
            )
            continue
        audited.append((candidate, audit))

    if not audited:
        return current_output, diagnostics, None

    best_support = max(candidate_metric_float(audit.get("metrics", {}) or {}, "claim_support_rate") for _, audit in audited)
    current_supported_claims = 0
    for candidate, audit in audited:
        if candidate.get("label") == "current":
            current_supported_claims = int((audit.get("metrics", {}) or {}).get("supported_claims", 0) or 0)
            break
    base_words = word_count(current_output)
    scored: list[tuple[float, bool, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for candidate, audit in audited:
        score, eligible, summary = score_output_candidate(
            candidate,
            audit,
            base_words,
            best_support,
            current_supported_claims,
            args,
        )
        diagnostics["candidates"].append(summary)
        scored.append((score, eligible, candidate, audit, summary))

    eligible_scored = [row for row in scored if row[1]]
    if eligible_scored:
        selected = max(eligible_scored, key=lambda row: row[0])
    else:
        current_rows = [row for row in scored if row[2].get("label") == "current"]
        selected = current_rows[0] if current_rows else max(scored, key=lambda row: row[0])

    _score, _eligible, selected_candidate, selected_audit, selected_summary = selected
    selected_output = str(selected_candidate.get("output", "") or current_output)
    diagnostics["selected_label"] = selected_candidate.get("label", "unknown")
    diagnostics["selected_score"] = selected_summary.get("score")
    diagnostics["selected_eligible"] = selected_summary.get("eligible")
    diagnostics["changed_output"] = output_candidate_key(selected_output) != output_candidate_key(current_output)
    return selected_output, diagnostics, selected_audit


def clean_answer_unit_fragment(text: str) -> str:
    text = normalize_space(remove_citations(text)).strip()
    text = text.strip(" \"'()[]")
    text = text.strip(" ,;:")
    text = re.sub(r"^(?:and|but|while|whereas|which|that|who|whose|because)\s+", "", text, flags=re.IGNORECASE)
    return text.rstrip(".;:,")


def answer_span_window(sentence: str, answer_span: str, max_words: int) -> str:
    sentence = normalize_space(remove_citations(sentence))
    answer_span = normalize_space(remove_citations(answer_span))
    if not sentence or not answer_span:
        return ""
    lower = sentence.lower()
    idx = lower.find(answer_span.lower())
    if idx < 0:
        return ""
    words = sentence.split()
    prefix_words = sentence[:idx].split()
    span_words = max(1, len(answer_span.split()))
    start = max(0, len(prefix_words) - 8)
    end = min(len(words), len(prefix_words) + span_words + 10)
    if end - start > max_words:
        end = start + max_words
    return clean_answer_unit_fragment(" ".join(words[start:end]))


def compact_answer_unit_claim(
    sentence: str,
    answer_span: str,
    fallback_claim: str,
    args: argparse.Namespace,
) -> str:
    """Prefer a short evidence-backed answer unit over a full background sentence."""

    max_words = expansion_claim_word_limit(args)
    fallback = clean_answer_unit_fragment(fallback_claim)
    answer_span = normalize_space(remove_citations(answer_span)).strip()
    if fallback and word_count(fallback) <= max_words:
        return fallback

    sentence = normalize_space(remove_citations(sentence))
    span_key = protected_span_key(answer_span)
    candidates: list[str] = []
    if sentence and span_key:
        pieces = re.split(
            r"(?<=[.;!?])\s+|;\s+|:\s+|,\s+(?=(?:and|but|while|whereas|which|who|that|because|although)\b)",
            sentence,
        )
        for piece in pieces:
            fragment = clean_answer_unit_fragment(piece)
            if not fragment:
                continue
            if span_key in protected_span_key(fragment) or answer_span.lower() in fragment.lower():
                candidates.append(fragment)
        window = answer_span_window(sentence, answer_span, max_words)
        if window:
            candidates.append(window)

    candidates = [candidate for candidate in candidates if candidate and word_count(candidate) <= max_words]
    if candidates:
        candidates.sort(key=lambda text: (word_count(text), len(text)))
        return candidates[0]
    if fallback and word_count(fallback) <= max_words + 6:
        return fallback
    return ""


def qampari_question_tokens(question: str) -> set[str]:
    return content_tokens(question)


def title_answer_span_candidate(title: str, question: str, evidence_sentence: str, args: argparse.Namespace) -> str:
    """Return a doc title when it looks like a plausible missing answer unit."""

    title = clean_answer_unit_fragment(title)
    if not title or infer_answer_format(args) == "qampari":
        return ""
    if (
        question_answer_type(question) == "explanation"
        and str(getattr(args, "explanatory_title_span_mode", "off") or "off").lower() == "off"
    ):
        return ""
    if word_count(title) > 10:
        return ""
    key = protected_span_key(title)
    if not key:
        return ""
    q_tokens = content_tokens(question)
    title_tokens = content_tokens(title)
    if title_tokens and q_tokens and title_tokens <= q_tokens:
        return ""
    bad, _reason = is_bad_general_answer_span(
        title,
        question,
        f"{title}. {evidence_sentence}",
        getattr(args, "answer_type_gate_mode", "soft"),
    )
    if bad:
        return ""
    return title


def qampari_is_bad_answer_span(span: str, question: str = "", evidence_sentence: str = "") -> tuple[bool, str]:
    """Filter QAMPARI answer spans that are supported but not answer-like.

    QAMPARI is an entity-list task. A span can be faithful to evidence and still
    be irrelevant, e.g. the subject entity from the question ("Ryoichi Ikegami")
    or a generic fragment ("German", "Type", "School"). This filter is
    deliberately conservative and deterministic so it can be ablated cleanly.
    """

    raw = normalize_space(remove_citations(span)).strip().strip("\"'")
    raw = raw.rstrip(".;:,")
    key = normalize_qampari_item_key(raw)
    if not key:
        return True, "empty_answer_span"
    if key in GENERIC_QAMPARI_ANSWER_KEYS:
        return True, "generic_answer_span"
    lowered = raw.lower()
    if any(bad in lowered for bad in QAMPARI_BAD_SUBSTRINGS):
        return True, "non_answer_phrase"
    words = key.split()
    if len(words) > 8:
        return True, "answer_span_too_long"
    if len(raw) < 2:
        return True, "answer_span_too_short"
    if words and words[-1] in {"and", "or", "of", "the", "a", "an", "in", "for", "to", "with"}:
        return True, "trailing_connector"

    question_key = normalize_qampari_item_key(question)
    if question_key and key in question_key:
        return True, "answer_span_copied_from_question"

    q_tokens = qampari_question_tokens(question)
    span_tokens = content_tokens(raw)
    if span_tokens and q_tokens and span_tokens <= q_tokens:
        return True, "answer_tokens_all_in_question"

    # Single-token adjectives and category words are common false positives in
    # QAMPARI, but single-token proper names/titles can be real answers.
    if len(words) == 1:
        token = words[0]
        if len(token) <= 2:
            return True, "single_token_too_short"
        if raw[:1].islower():
            return True, "lowercase_single_token"
        if token in q_tokens:
            return True, "single_token_from_question"

    if evidence_sentence:
        evidence_key = normalize_qampari_item_key(evidence_sentence)
        if key and key not in evidence_key:
            return True, "answer_span_not_in_evidence"

    return False, ""


def qampari_claim_answer_span(claim: dict[str, Any]) -> str:
    return extract_qampari_answer_item(claim)


def qampari_claim_is_answer_relevant(claim: dict[str, Any], question: str) -> tuple[bool, str]:
    span = qampari_claim_answer_span(claim)
    evidence = str(claim.get("source_sentence_without_citations") or claim.get("source_sentence") or "")
    bad, reason = qampari_is_bad_answer_span(span, question, evidence_sentence=evidence if claim.get("v2_status") == "expanded" else "")
    if bad:
        return False, reason
    return True, ""


def filter_qampari_claims_for_answer_relevance(
    claims: Sequence[dict[str, Any]],
    rejected_claims: Sequence[dict[str, Any]],
    question: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if infer_answer_format(args) != "qampari" or not args.qampari_answer_relevance_filter:
        return list(claims), list(rejected_claims)

    kept: list[dict[str, Any]] = []
    rejected = list(rejected_claims)
    seen: set[str] = set()
    for claim in claims:
        span = qampari_claim_answer_span(claim)
        key = normalize_qampari_item_key(span)
        ok, reason = qampari_claim_is_answer_relevant(claim, question)
        if not ok:
            dropped = dict(claim)
            dropped["v3_status"] = "dropped_by_answer_relevance"
            dropped["answer_relevance_reason"] = reason
            rejected.append(dropped)
            continue
        if key in seen:
            dropped = dict(claim)
            dropped["v3_status"] = "dropped_duplicate_answer_item"
            dropped["answer_relevance_reason"] = "duplicate_answer_item"
            rejected.append(dropped)
            continue
        seen.add(key)
        kept.append(claim)
    return kept, rejected


def covered_answer_spans(claims: Sequence[dict[str, Any]]) -> set[str]:
    covered: set[str] = set()
    for claim in claims:
        for span in claim.get("protected_answer_spans", []) or []:
            key = protected_span_key(span)
            if key:
                covered.add(key)
    return covered


def claim_texts_for_query(claims: Sequence[dict[str, Any]], limit: int = 12) -> str:
    return " ".join(str(claim.get("claim", "") or "") for claim in claims[:limit])


def expansion_query(
    question: str,
    verified_claims: Sequence[dict[str, Any]],
    rejected_claims: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    parts = [question]
    if getattr(args, "expansion_use_intermediate_answer", False):
        parts.append(supported_answer_summary(verified_claims))
    if args.expansion_query_source in {"question_and_evidence_core", "question_evidence_and_rejected"}:
        parts.append(claim_texts_for_query(verified_claims))
    if args.expansion_query_source == "question_evidence_and_rejected":
        parts.append(claim_texts_for_query(rejected_claims))
    return " ".join(part for part in parts if part)


def doc_body(doc: dict[str, Any], args: argparse.Namespace) -> str:
    return str(doc.get(args.sent_field) or doc.get(args.text_field, "") or "")


def dynamic_retrieval_doc_text(doc: dict[str, Any], args: argparse.Namespace) -> str:
    title = normalize_space(str(doc.get(args.title_field, "") or ""))
    body = normalize_space(doc_body(doc, args) or format_doc(doc, args))
    return normalize_space(f"{title}. {body}" if title else body)


def sentence_answer_span_bonus(sentence: str, covered_spans: set[str], question: str, args: argparse.Namespace) -> float:
    spans = extract_protected_spans(sentence)
    if getattr(args, "answer_targeted_expansion", False) and infer_answer_format(args) != "qampari":
        spans.extend(salient_answer_phrases(sentence, question, limit=6))
    if not spans:
        return 0.0
    new_spans = [span for span in spans if protected_span_key(span) not in covered_spans]
    if not new_spans:
        return 0.0
    targeted_bonus = 0.06 if getattr(args, "answer_targeted_expansion", False) else 0.0
    return min(0.36, targeted_bonus + 0.08 * len(new_spans))


def rank_evidence_sentences(
    item: dict[str, Any],
    verified_claims: Sequence[dict[str, Any]],
    rejected_claims: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    docs = item.get(args.docs_field, []) or []
    question = str(item.get(args.question_field, "") or "")
    query = expansion_query(question, verified_claims, rejected_claims, args)
    evidence_core = claim_texts_for_query(verified_claims)
    rejected_text = claim_texts_for_query(rejected_claims)
    covered_spans = covered_answer_spans(verified_claims)

    ranked: list[dict[str, Any]] = []
    for doc_id, doc in enumerate(docs, start=1):
        title = normalize_space(str(doc.get(args.title_field, "") or ""))
        body = doc_body(doc, args)
        sentences = sent_tokenize_safe(body or format_doc(doc, args))
        for sent_id, sentence in enumerate(sentences):
            sentence = normalize_space(remove_citations(sentence))
            if len(sentence) < 20:
                continue
            title_sentence = normalize_space(f"{title}. {sentence}" if title else sentence)
            score = 0.40 * overlap_score(question, sentence)
            score += 0.20 * overlap_score(question, title_sentence)
            score += 0.18 * overlap_score(query, title_sentence)
            score += 0.08 * overlap_score(evidence_core, title_sentence)
            score += 0.08 * overlap_score(rejected_text, title_sentence)
            score += 0.06 * overlap_score(question, title)
            score += sentence_answer_span_bonus(title_sentence, covered_spans, question, args)
            if title_answer_span_candidate(title, question, sentence, args):
                score += 0.10
            if doc_id <= 20:
                score += 0.03 * (1.0 - ((doc_id - 1) / 20.0))
            if doc.get("cover_dynamic_retrieval_score") is not None:
                score += float(getattr(args, "dynamic_retrieval_sentence_boost", 0.12) or 0.0)
                try:
                    score += min(0.08, 0.02 * float(doc.get("cover_dynamic_retrieval_score", 0.0) or 0.0))
                except Exception:
                    pass
            if getattr(args, "answer_targeted_expansion", False) and infer_answer_format(args) != "qampari":
                # Main-task gains come from adding answer-bearing evidence, not
                # merely more supported background. Reward compact clauses that
                # contain a new answer unit and still overlap the question.
                if candidate_answer_spans_from_sentence(title_sentence, verified_claims, args, question=question):
                    score += 0.08 * overlap_score(question, title_sentence)
            # Very short token overlap can still be useful for entity-list
            # questions, but pure zero-overlap sentences are almost always noise.
            if score < args.min_expansion_score:
                continue
            ranked.append(
                {
                    "doc_id": doc_id,
                    "sent_id": sent_id,
                    "score": score,
                    "title": title,
                    "sentence": sentence,
                    "doc": doc,
                }
            )

    ranked.sort(key=lambda row: (-float(row["score"]), int(row["doc_id"]), int(row["sent_id"])))
    return ranked


def evidence_sentence_block(ranked_sentences: Sequence[dict[str, Any]], max_sentences: int) -> str:
    lines = []
    for row in ranked_sentences[:max_sentences]:
        title = normalize_space(str(row.get("title", "") or ""))
        title_part = f" Title: {title}." if title else ""
        lines.append(
            f"[doc_id={int(row['doc_id'])}, score={float(row['score']):.3f}]{title_part} "
            f"Sentence: {row['sentence']}"
        )
    return "\n".join(lines) or "None."


def candidate_answer_span_rows(
    ranked_sentences: Sequence[dict[str, Any]],
    verified_claims: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    question: str = "",
) -> list[dict[str, Any]]:
    covered = covered_answer_spans(verified_claims)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in ranked_sentences:
        sentence = normalize_space(str(row.get("sentence", "") or ""))
        title = normalize_space(str(row.get("title", "") or ""))
        title_sentence = normalize_space(f"{title}. {sentence}" if title else sentence)
        title_span = title_answer_span_candidate(title, question, sentence, args)
        raw_spans: list[str] = []
        if title_span:
            raw_spans.append(title_span)
        raw_spans.extend(candidate_answer_spans_from_sentence(title_sentence, verified_claims, args, question=question))
        for span in raw_spans:
            key = protected_span_key(span)
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "span_id": f"s{len(rows) + 1}",
                    "span": span,
                    "doc_id": int(row["doc_id"]),
                    "sent_id": int(row.get("sent_id", 0)),
                    "score": float(row["score"]),
                    "sentence": title_sentence,
                }
            )
            if len(rows) >= max(1, args.expansion_candidate_spans):
                return rows
    return rows


def candidate_answer_span_block(span_rows: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in span_rows:
        lines.append(
            f"- span_id: {row['span_id']} | span: {row['span']} | doc_id: {int(row['doc_id'])} | "
            f"score: {float(row['score']):.3f} | sentence: {row['sentence']}"
        )
    return "\n".join(lines) or "None."


def span_lookup_from_rows(span_rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("span_id", "")): dict(row) for row in span_rows if row.get("span_id")}


def recall_completion_target_key(text: str) -> str:
    key = normalize_claim_key(text)
    if key:
        return key
    return protected_span_key(text)


def recall_completion_target_is_covered(
    text: str,
    verified_claims: Sequence[dict[str, Any]],
    supported_text: str,
) -> bool:
    key = protected_span_key(text)
    if not key:
        return True
    if key in covered_answer_spans(verified_claims):
        return True
    supported_key = protected_span_key(supported_text)
    if not supported_key:
        return False
    key_words = key.split()
    if len(key_words) >= 2 and key in supported_key:
        return True
    return recall_completion_target_key(text) in {
        recall_completion_target_key(str(claim.get("claim", "") or ""))
        for claim in verified_claims
    }


def add_recall_completion_target(
    targets: list[dict[str, Any]],
    seen: set[str],
    target: str,
    source: str,
    question: str,
    verified_claims: Sequence[dict[str, Any]],
    supported_text: str,
    *,
    doc_id: int | None = None,
    evidence_sentence: str = "",
    answer_span: str = "",
    answer_type_gate_mode: str = "strict",
) -> None:
    target = normalize_space(remove_citations(target)).strip().strip("\"'")
    target = target.rstrip(".;:,")
    if not target:
        return
    if source == "ranked_evidence":
        bad, _reason = is_bad_general_answer_span(
            answer_span or target,
            question,
            evidence_sentence,
            answer_type_gate_mode,
        )
        if bad:
            return
    key = recall_completion_target_key(target)
    if not key or key in seen:
        return
    if recall_completion_target_is_covered(target, verified_claims, supported_text):
        return
    if question and overlap_score(question, target) <= 0 and not answer_span:
        # A pure evidence phrase with zero question overlap is usually topical
        # background, not missing answer coverage.
        return
    seen.add(key)
    targets.append(
        {
            "target": target,
            "source": source,
            "doc_id": doc_id,
            "evidence_sentence": evidence_sentence,
            "answer_span": answer_span or target,
        }
    )


def build_recall_completion_targets(
    question: str,
    verified_claims: Sequence[dict[str, Any]],
    rejected_claims: Sequence[dict[str, Any]],
    ranked_sentences: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not getattr(args, "recall_oriented_completion", False):
        return []

    supported_text = supported_answer_summary(
        verified_claims,
        limit=max(8, int(getattr(args, "recall_completion_supported_claims", 18))),
        max_chars=2400,
    )
    max_targets = max(1, int(getattr(args, "recall_completion_max_targets", 10)))
    max_rejected = max(0, int(getattr(args, "recall_completion_rejected_targets", 4)))
    max_evidence = max(0, int(getattr(args, "recall_completion_evidence_sentences", 24)))
    min_sentence_overlap = float(getattr(args, "recall_completion_min_sentence_overlap", 0.0))

    targets: list[dict[str, Any]] = []
    seen: set[str] = set()

    for claim in rejected_claims[:max_rejected]:
        if len(targets) >= max_targets:
            break
        if str(claim.get("importance", "supporting")) == "background":
            continue
        text = normalize_space(str(claim.get("claim", "") or ""))
        spans = list(claim.get("protected_answer_spans", []) or [])
        spans.extend(extract_protected_spans(text))
        if is_explanatory_question(question):
            spans.extend(salient_answer_phrases(text, question, limit=4))
        source_sentence = normalize_space(
            str(claim.get("source_sentence_without_citations") or claim.get("source_sentence") or "")
        )
        for span in spans or [text]:
            add_recall_completion_target(
                targets,
                seen,
                span,
                "rejected_claim",
                question,
                verified_claims,
                supported_text,
                evidence_sentence=source_sentence,
                answer_span=span,
                answer_type_gate_mode=getattr(args, "answer_type_gate_mode", "strict"),
            )
            if len(targets) >= max_targets:
                break

    for row in ranked_sentences[:max_evidence]:
        if len(targets) >= max_targets:
            break
        sentence = normalize_space(str(row.get("sentence", "") or ""))
        if min_sentence_overlap > 0 and overlap_score(question, sentence) < min_sentence_overlap:
            continue
        spans = candidate_answer_spans_from_sentence(sentence, verified_claims, args, question=question)
        if is_explanatory_question(question):
            spans.extend(salient_answer_phrases(sentence, question, limit=6))
        for span in spans:
            add_recall_completion_target(
                targets,
                seen,
                span,
                "ranked_evidence",
                question,
                verified_claims,
                supported_text,
                doc_id=int(row.get("doc_id", 0) or 0) or None,
                evidence_sentence=sentence,
                answer_span=span,
                answer_type_gate_mode=getattr(args, "answer_type_gate_mode", "strict"),
            )
            if len(targets) >= max_targets:
                break
    return targets


def dynamic_retrieval_enabled(args: argparse.Namespace) -> bool:
    return str(getattr(args, "dynamic_retrieval_mode", "off") or "off") != "off"


def dynamic_retrieval_claim_texts(
    claims: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    *,
    limit: int,
) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for claim in claims:
        if str(claim.get("importance", "supporting")) == "background":
            continue
        text = normalize_space(str(claim.get("claim", "") or ""))
        if not text:
            continue
        key = normalize_claim_key(text) or protected_span_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(text)
        if len(output) >= limit:
            break
    return output


def coverage_guided_retrieval_enabled(args: argparse.Namespace) -> bool:
    return dynamic_retrieval_enabled(args) and bool(getattr(args, "coverage_guided_retrieval", False))


def coverage_guided_target_text(target: dict[str, Any]) -> str:
    return normalize_space(str(target.get("answer_span") or target.get("target") or ""))


def coverage_guided_target_key(target: dict[str, Any]) -> str:
    return recall_completion_target_key(coverage_guided_target_text(target))


def select_coverage_guided_retrieval_targets(
    question: str,
    verified_claims: Sequence[dict[str, Any]],
    rejected_claims: Sequence[dict[str, Any]],
    ranked_sentences: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Choose uncovered answer-unit targets that are strong enough to drive retrieval.

    This is the controller step for coverage-guided iterative RAG: dynamic
    retrieval should only happen when claim verification exposes a concrete gap,
    not merely because the original question is broad.
    """

    target_args = copy.copy(args)
    target_args.recall_oriented_completion = True
    max_targets = max(1, int(getattr(args, "coverage_guided_targets_per_round", 3) or 3))
    target_args.recall_completion_max_targets = max(
        int(getattr(args, "recall_completion_max_targets", 10) or 10),
        max_targets * 3,
    )
    target_args.recall_completion_rejected_targets = max(
        int(getattr(args, "recall_completion_rejected_targets", 4) or 4),
        int(getattr(args, "dynamic_retrieval_max_rejected_claims", 4) or 4),
    )

    raw_targets = build_recall_completion_targets(
        question,
        verified_claims,
        rejected_claims,
        ranked_sentences,
        target_args,
    )
    allow_evidence_targets = bool(getattr(args, "coverage_guided_allow_evidence_targets", True))
    min_question_overlap = float(getattr(args, "coverage_guided_min_question_overlap", 0.08) or 0.08)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    supported_text = supported_answer_summary(
        verified_claims,
        limit=max(6, int(getattr(args, "coverage_guided_query_supported_claims", 6) or 6)),
        max_chars=900,
    )
    for index, target in enumerate(raw_targets):
        source = normalize_space(str(target.get("source", "") or ""))
        if source == "ranked_evidence" and not allow_evidence_targets:
            continue

        target_text = coverage_guided_target_text(target)
        if not target_text:
            continue
        key = coverage_guided_target_key(target)
        if not key or key in seen:
            continue
        if recall_completion_target_is_covered(
            target_text,
            verified_claims,
            supported_text,
        ):
            continue

        evidence_sentence = normalize_space(str(target.get("evidence_sentence", "") or ""))
        question_overlap = max(overlap_score(question, target_text), overlap_score(question, evidence_sentence))
        if source == "ranked_evidence" and question_overlap < min_question_overlap:
            continue
        if source == "ranked_evidence" and looks_like_title_or_source_fragment(target_text):
            continue
        if source == "ranked_evidence" and recall_target_is_generic(target_text, source):
            continue

        row = dict(target)
        row["coverage_guided_target_key"] = key
        row["coverage_guided_question_overlap"] = question_overlap
        row["coverage_guided_priority"] = (
            1.0 if source == "rejected_claim" else 0.0,
            question_overlap,
            -index,
        )
        selected.append(row)
        seen.add(key)

    selected.sort(key=lambda row: row.get("coverage_guided_priority", (0.0, 0.0, 0)), reverse=True)
    return selected[:max_targets]


def build_coverage_guided_retrieval_queries(
    question: str,
    verified_claims: Sequence[dict[str, Any]],
    targets: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> list[str]:
    max_queries = max(1, int(getattr(args, "dynamic_retrieval_max_queries", 4) or 4))
    supported_text = (
        supported_answer_summary(
            verified_claims,
            limit=max(1, int(getattr(args, "coverage_guided_query_supported_claims", 6) or 6)),
            max_chars=900,
        )
        if getattr(args, "dynamic_retrieval_use_supported_answer", True)
        else ""
    )
    supported_phrases = salient_answer_phrases(supported_text, question, limit=4) if supported_text else []
    supported_brief = " ".join(supported_phrases[:4])

    queries: list[str] = []
    seen: set[str] = set()
    for target in targets:
        if len(queries) >= max_queries:
            break
        target_text = coverage_guided_target_text(target)
        if not target_text:
            continue
        evidence_sentence = normalize_space(str(target.get("evidence_sentence", "") or ""))
        parts = [question, "missing answer unit:", target_text]
        if evidence_sentence:
            parts.extend(["evidence clue:", truncate_text(evidence_sentence, 180)])
        if supported_brief:
            parts.extend(["already covered:", supported_brief])
        query = normalize_space(" ".join(part for part in parts if part))
        query = truncate_text(query, int(getattr(args, "dynamic_retrieval_query_max_chars", 900) or 900))
        if len(content_tokens(query)) < max(1, int(getattr(args, "dynamic_retrieval_min_query_terms", 2) or 2)):
            continue
        key = protected_span_key(query)
        if not key or key in seen:
            continue
        seen.add(key)
        queries.append(query)
    return queries


def build_dynamic_retrieval_queries(
    question: str,
    verified_claims: Sequence[dict[str, Any]],
    rejected_claims: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> list[str]:
    if not dynamic_retrieval_enabled(args):
        return []

    max_queries = max(1, int(getattr(args, "dynamic_retrieval_max_queries", 4) or 4))
    max_rejected = max(0, int(getattr(args, "dynamic_retrieval_max_rejected_claims", 4) or 4))
    require_missing = bool(getattr(args, "dynamic_retrieval_require_missing_signal", True))
    query_strategy = str(getattr(args, "dynamic_retrieval_query_strategy", "missing_facet") or "missing_facet")
    rejected_texts = dynamic_retrieval_claim_texts(rejected_claims, args, limit=max_rejected)
    supported_text = (
        supported_answer_summary(
            verified_claims,
            limit=max(6, int(getattr(args, "recall_completion_supported_claims", 18) or 18)),
            max_chars=1600,
        )
        if getattr(args, "dynamic_retrieval_use_supported_answer", True)
        else ""
    )
    targets = build_recall_completion_targets(question, verified_claims, rejected_claims, [], args)
    target_texts = [
        normalize_space(str(target.get("answer_span") or target.get("target") or ""))
        for target in targets[:max_queries]
    ]
    target_texts = [text for text in target_texts if text]

    has_missing_signal = bool(rejected_texts or target_texts)
    if require_missing and not has_missing_signal:
        return []

    queries: list[str] = []
    seen: set[str] = set()

    def add_query(*parts: str) -> None:
        if len(queries) >= max_queries:
            return
        query = normalize_space(" ".join(part for part in parts if part))
        query = truncate_text(query, int(getattr(args, "dynamic_retrieval_query_max_chars", 900) or 900))
        if not query:
            return
        if len(content_tokens(query)) < max(1, int(getattr(args, "dynamic_retrieval_min_query_terms", 2) or 2)):
            return
        key = protected_span_key(query)
        if not key or key in seen:
            return
        seen.add(key)
        queries.append(query)

    if query_strategy in {"question_facet", "hybrid"}:
        add_query(question)
        supported_phrases = salient_answer_phrases(supported_text, question, limit=6) if supported_text else []
        supported_brief = " ".join(supported_phrases[:4]) or truncate_text(supported_text, 420)
        if supported_brief:
            add_query(question, "current supported answer:", supported_brief)

    missing_block = " ".join(target_texts[:3] + rejected_texts[:3])
    if query_strategy in {"missing_facet", "hybrid"} or has_missing_signal:
        add_query(question, "missing answer facets:", missing_block, supported_text)
        for target in target_texts:
            add_query(question, "missing answer facet:", target, supported_text)
        for text in rejected_texts:
            add_query(question, "unsupported draft claim to verify or replace:", text, supported_text)
    return queries


def merge_dynamic_doc_pool(
    existing_docs: Sequence[dict[str, Any]],
    dynamic_docs: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    max_pool = max(
        int(getattr(args, "max_doc_pool", 100) or 100),
        int(getattr(args, "dynamic_retrieval_max_doc_pool", 140) or 140),
    )
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for doc in list(existing_docs) + list(dynamic_docs):
        key = doc_identity(doc, args)
        if not key or key in seen:
            continue
        seen.add(key)
        pool.append(doc)
        if len(pool) >= max_pool:
            break
    return pool


def apply_dynamic_retrieval_to_item(
    item: dict[str, Any],
    item_id: int,
    candidate_provider: CandidateDocProvider | None,
    verified_claims: Sequence[dict[str, Any]],
    rejected_claims: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    *,
    phase: str,
    round_index: int = 0,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "enabled": dynamic_retrieval_enabled(args),
        "phase": phase,
        "round": round_index,
        "queries": [],
        "added_docs": 0,
    }
    if not diagnostics["enabled"]:
        return diagnostics
    if candidate_provider is None:
        diagnostics["reason"] = "no_candidate_provider"
        return diagnostics
    question = str(item.get(args.question_field, "") or "")
    if coverage_guided_retrieval_enabled(args):
        ranked_sentences = rank_evidence_sentences(item, verified_claims, rejected_claims, args)
        targets = select_coverage_guided_retrieval_targets(
            question,
            verified_claims,
            rejected_claims,
            ranked_sentences,
            args,
        )
        diagnostics["coverage_guided"] = True
        diagnostics["coverage_guided_targets"] = [
            {
                "target": coverage_guided_target_text(target),
                "source": target.get("source", ""),
                "question_overlap": target.get("coverage_guided_question_overlap", 0.0),
                "evidence_sentence": truncate_text(str(target.get("evidence_sentence", "") or ""), 220),
            }
            for target in targets
        ]
        min_targets = max(0, int(getattr(args, "coverage_guided_min_targets", 1) or 1))
        if len(targets) < min_targets:
            diagnostics["queries"] = []
            diagnostics["reason"] = "coverage_guided_no_missing_targets"
            return diagnostics
        queries = build_coverage_guided_retrieval_queries(question, verified_claims, targets, args)
    else:
        diagnostics["coverage_guided"] = False
        queries = build_dynamic_retrieval_queries(question, verified_claims, rejected_claims, args)
    diagnostics["queries"] = queries
    if not queries:
        diagnostics["reason"] = "no_missing_facet_queries"
        return diagnostics

    existing_docs = list(item.get(args.docs_field, []) or [])
    seen_keys = {doc_identity(doc, args) for doc in existing_docs}
    additions: list[dict[str, Any]] = []
    per_query = max(1, int(getattr(args, "dynamic_retrieval_per_query_docs", 4) or 4))
    total_limit = max(1, int(getattr(args, "dynamic_retrieval_top_k", 12) or 12))
    query_rows: list[dict[str, Any]] = []
    for query_id, query in enumerate(queries, start=1):
        if len(additions) >= total_limit:
            break
        docs = candidate_provider.search_dynamic(
            query,
            min(per_query, total_limit - len(additions)),
            args,
            exclude_keys=seen_keys,
        )
        added_for_query = 0
        for doc in docs:
            key = doc_identity(doc, args)
            if not key or key in seen_keys:
                continue
            doc = dict(doc)
            doc["cover_dynamic_retrieval_phase"] = phase
            doc["cover_dynamic_retrieval_round"] = round_index
            doc["cover_dynamic_retrieval_query_id"] = query_id
            additions.append(doc)
            seen_keys.add(key)
            added_for_query += 1
            if len(additions) >= total_limit:
                break
        query_rows.append(
            {
                "query_id": query_id,
                "query": truncate_text(query, 220),
                "returned_docs": len(docs),
                "added_docs": added_for_query,
            }
        )
    diagnostics["query_results"] = query_rows
    diagnostics["added_docs"] = len(additions)
    if additions:
        item[args.docs_field] = merge_dynamic_doc_pool(existing_docs, additions, args)
    return diagnostics


def summarize_dynamic_retrieval_diagnostics(diagnostics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    active = [diag for diag in diagnostics if diag.get("enabled")]
    coverage_guided = [diag for diag in active if diag.get("coverage_guided")]
    return {
        "dynamic_retrieval_rounds": len(active),
        "dynamic_retrieval_queries": sum(len(diag.get("queries", []) or []) for diag in active),
        "dynamic_retrieval_added_docs": sum(int(diag.get("added_docs", 0) or 0) for diag in active),
        "dynamic_retrieval_successful_rounds": sum(1 for diag in active if int(diag.get("added_docs", 0) or 0) > 0),
        "coverage_guided_retrieval_rounds": len(coverage_guided),
        "coverage_guided_retrieval_targets": sum(
            len(diag.get("coverage_guided_targets", []) or []) for diag in coverage_guided
        ),
        "coverage_guided_retrieval_skipped_no_targets": sum(
            1 for diag in coverage_guided if diag.get("reason") == "coverage_guided_no_missing_targets"
        ),
    }


def recall_completion_target_block(targets: Sequence[dict[str, Any]]) -> str:
    if not targets:
        return "None."
    lines = []
    for idx, target in enumerate(targets, start=1):
        answer_span = normalize_space(str(target.get("answer_span", "") or target.get("target", "") or ""))
        doc_id = target.get("doc_id")
        sentence = normalize_space(str(target.get("evidence_sentence", "") or ""))
        location = f"doc_id={doc_id}" if doc_id else "doc_id=unknown"
        evidence = f" | evidence: {truncate_text(sentence, 180)}" if sentence else ""
        lines.append(
            f"{idx}. missing_target: {target.get('target', '')} | answer_span: {answer_span} | {location}{evidence}"
        )
    return "\n".join(lines)


def candidate_matches_recall_completion_target(
    candidate: dict[str, Any],
    targets: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[bool, str, dict[str, Any] | None]:
    if not getattr(args, "recall_oriented_completion", False):
        return True, "", None
    if not targets:
        return False, "recall_completion_no_targets", None

    claim = normalize_space(str(candidate.get("claim", "") or ""))
    answer_span = normalize_space(str(candidate.get("answer_span", "") or ""))
    target_gap = normalize_space(str(candidate.get("target_gap", "") or ""))
    evidence_sentence = normalize_space(str(candidate.get("evidence_sentence", "") or ""))
    candidate_text = " ".join(part for part in [claim, answer_span, target_gap, evidence_sentence] if part)
    candidate_key = protected_span_key(candidate_text)
    min_target_overlap = float(getattr(args, "recall_completion_min_target_overlap", 0.10))

    for target in targets:
        target_text = normalize_space(str(target.get("target", "") or ""))
        target_span = normalize_space(str(target.get("answer_span", "") or ""))
        target_key = protected_span_key(target_span or target_text)
        if target_key and target_key in candidate_key:
            return True, "", target
        if target_text and overlap_score(target_text, candidate_text) >= min_target_overlap:
            return True, "", target
    return False, "recall_completion_target_mismatch", None


def expansion_candidate_answer_unit_score(
    candidate: dict[str, Any],
    question: str,
    targets: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[float, dict[str, float]]:
    """Score candidates as missing answer units, not merely supported facts."""

    claim = normalize_space(str(candidate.get("claim", "") or ""))
    answer_span = normalize_space(str(candidate.get("answer_span", "") or ""))
    target_gap = normalize_space(str(candidate.get("target_gap", "") or ""))
    evidence_sentence = normalize_space(str(candidate.get("evidence_sentence", "") or ""))
    candidate_text = " ".join(part for part in [claim, answer_span, target_gap, evidence_sentence] if part)
    candidate_key = protected_span_key(candidate_text)

    question_relevance = max(overlap_score(question, claim), overlap_score(question, evidence_sentence))
    target_relevance = 0.0
    for target in targets:
        target_text = normalize_space(str(target.get("target", "") or ""))
        target_span = normalize_space(str(target.get("answer_span", "") or target_text))
        target_key = protected_span_key(target_span or target_text)
        if target_key and target_key in candidate_key:
            exact_relevance = 1.0
            if str(target.get("source", "")) == "ranked_evidence":
                # Evidence-derived targets are only hypotheses about missing
                # gold units. Do not let a span selected from evidence prove
                # itself as a perfect target match.
                exact_relevance = 0.55
            target_relevance = max(target_relevance, exact_relevance)
            if exact_relevance >= 1.0:
                break
        if target_text:
            target_relevance = max(target_relevance, overlap_score(target_text, candidate_text))

    answer_span_bonus = 0.0
    span_key = protected_span_key(answer_span)
    if span_key and span_key in candidate_key:
        answer_span_bonus = 1.0

    try:
        rank_score = max(0.0, min(float(candidate.get("rank_score", 0.0)), 1.0))
    except Exception:
        rank_score = 0.0
    compactness_penalty = 0.0
    if infer_answer_format(args) != "qampari":
        claim_words = word_count(claim)
        if claim_words > expansion_claim_word_limit(args):
            compactness_penalty += 0.25
        if claim_words > 36:
            compactness_penalty += 0.20
        if span_key and span_key not in protected_span_key(claim):
            compactness_penalty += 0.08
        type_mismatch_reason = general_answer_span_type_mismatch(answer_span or claim, question, evidence_sentence)
        answer_type_gate_mode = getattr(args, "answer_type_gate_mode", "strict")
        type_mismatch_penalty = 0.0
        if (
            type_mismatch_reason
            and answer_type_gate_mode != "off"
            and not answer_type_mismatch_is_hard(type_mismatch_reason, answer_type_gate_mode)
        ):
            type_mismatch_penalty = 0.10
            compactness_penalty += 0.10
    else:
        type_mismatch_reason = ""
        type_mismatch_penalty = 0.0

    score = (
        0.35 * target_relevance
        + 0.25 * question_relevance
        + 0.25 * answer_span_bonus
        + 0.15 * rank_score
        - compactness_penalty
    )
    coverage_reranker_score = 0.0
    coverage_reranker_weight = max(
        0.0,
        min(float(getattr(args, "coverage_reranker_candidate_weight", 0.20) or 0.0), 1.0),
    )
    if coverage_reranker_weight > 0.0 and coverage_reranker_enabled(args):
        target_texts = []
        for target in targets:
            target_text = normalize_space(str(target.get("answer_span") or target.get("target") or ""))
            if target_text:
                target_texts.append(target_text)
        missing_unit = target_gap or " | ".join(target_texts[:4])
        query_text = format_coverage_query(question, missing_unit=missing_unit)
        evidence_text = format_candidate_evidence(evidence_sentence, claim=claim)
        coverage_reranker_score = coverage_reranker_score_pairs(args, [query_text], [evidence_text])[0]
        score = (1.0 - coverage_reranker_weight) * score + coverage_reranker_weight * coverage_reranker_score
    return score, {
        "target_relevance": target_relevance,
        "question_relevance": question_relevance,
        "answer_span_bonus": answer_span_bonus,
        "rank_score": rank_score,
        "coverage_reranker_score": coverage_reranker_score,
        "coverage_reranker_weight": coverage_reranker_weight if coverage_reranker_enabled(args) else 0.0,
        "compactness_penalty": compactness_penalty,
        "type_mismatch_penalty": type_mismatch_penalty,
    }


EXPLANATORY_SIGNAL_PATTERN = re.compile(
    r"\b("
    r"because|due to|as a result|caused by|causes?|results? from|results? in|stems? from|"
    r"leads? to|allows?|enables?|prevents?|helps?|requires?|explains?|means?|refers? to|"
    r"defined as|made of|composed of|happens when|happens because|the reason"
    r")\b",
    flags=re.IGNORECASE,
)


GENERIC_RECALL_TARGETS = {
    "also",
    "another",
    "background",
    "example",
    "however",
    "included",
    "includes",
    "known",
    "main",
    "more",
    "other",
    "previous",
    "several",
    "some",
    "there",
    "these",
    "this",
    "widely",
}


def looks_like_title_or_source_fragment(text: str) -> bool:
    cleaned = normalize_space(remove_citations(text)).strip()
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if re.search(r"\s+\|\s+", cleaned):
        return True
    if re.search(r"\b(?:home|archive|blog|faq|wiki|wikipedia|download|review|buying guide)\b", lowered):
        return True
    if re.search(r"\b(?:from the|on the)\s+[A-Z][A-Za-z0-9&'.-]+(?:\s+[A-Z][A-Za-z0-9&'.-]+){0,4}\s+(?:s-1|filing|website|page)\b", cleaned):
        return True
    if re.search(r"\bYear\s+\d{1,2}\s+Year\s+\d{1,2}\b", cleaned):
        return True
    if re.search(r"\b(?:ed\.secundaria|curso|igcse|gce)\b", lowered):
        return True
    return False


def recall_target_is_generic(target_gap: str, target_source: str = "") -> bool:
    target = normalize_space(remove_citations(target_gap)).strip().strip("\"'")
    if not target:
        return False
    key_words = protected_span_key(target).split()
    if len(key_words) == 1 and not re.search(r"\d", target):
        return key_words[0] in GENERIC_RECALL_TARGETS or target_source == "ranked_evidence"
    if len(key_words) <= 2 and all(word in GENERIC_RECALL_TARGETS for word in key_words):
        return True
    return False


def answer_unit_directness_diagnostics(
    candidate: dict[str, Any],
    question: str,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    claim = normalize_space(str(candidate.get("claim", "") or ""))
    answer_span = normalize_space(str(candidate.get("answer_span", "") or ""))
    target_gap = normalize_space(str(candidate.get("target_gap", "") or ""))
    target_source = normalize_space(str(candidate.get("target_source", "") or ""))
    evidence_sentence = normalize_space(str(candidate.get("evidence_sentence", "") or ""))
    score_parts = candidate.get("answer_unit_score_parts", {}) or {}
    qtype = question_answer_type(question)

    try:
        target_relevance = float(score_parts.get("target_relevance", 0.0) or 0.0)
    except Exception:
        target_relevance = 0.0
    try:
        answer_span_bonus = float(score_parts.get("answer_span_bonus", 0.0) or 0.0)
    except Exception:
        answer_span_bonus = 0.0
    question_relevance = max(overlap_score(question, claim), overlap_score(question, evidence_sentence))
    directness = 0.42 * question_relevance + 0.34 * target_relevance + 0.24 * answer_span_bonus

    claim_words = word_count(claim)
    span_words = word_count(answer_span)
    has_explanatory_signal = bool(EXPLANATORY_SIGNAL_PATTERN.search(" ".join([claim, evidence_sentence])))
    title_like = (
        looks_like_title_or_source_fragment(claim)
        or looks_like_title_or_source_fragment(answer_span)
        or looks_like_title_or_source_fragment(evidence_sentence)
    )
    generic_target = recall_target_is_generic(target_gap, target_source)
    if title_like:
        directness -= 0.35
    if generic_target:
        directness -= 0.18
    if qtype == "explanation":
        if has_explanatory_signal:
            directness += 0.18
        if claim_words >= 7 and question_relevance >= 0.14:
            directness += 0.08
        if span_words <= 2 and answer_span[:1].isupper() and question_relevance < 0.18:
            directness -= 0.28
    elif qtype in {"number", "date", "place"}:
        if question_relevance >= 0.12:
            directness += 0.08
    elif qtype in {"person", "open"}:
        if target_source == "rejected_claim":
            directness += 0.10
        if question_relevance >= 0.10:
            directness += 0.06

    min_directness = float(getattr(args, "answer_unit_min_directness", 0.44) or 0.44)
    if qtype == "explanation":
        min_directness = float(getattr(args, "explanatory_answer_unit_min_directness", min_directness) or min_directness)

    diagnostics = {
        "answer_unit_directness": directness,
        "answer_unit_min_directness": min_directness,
        "answer_unit_question_relevance": question_relevance,
        "answer_unit_target_relevance": target_relevance,
        "answer_unit_answer_span_bonus": answer_span_bonus,
        "answer_unit_has_explanatory_signal": has_explanatory_signal,
        "answer_unit_title_like": title_like,
        "answer_unit_generic_target": generic_target,
        "answer_unit_question_type": qtype,
    }

    if title_like:
        return "answer_unit_title_or_source_fragment", diagnostics
    if generic_target and target_relevance < 0.90:
        return "answer_unit_generic_target", diagnostics
    if qtype == "explanation":
        if span_words <= 2 and answer_span[:1].isupper() and question_relevance < 0.18:
            return "answer_unit_explanatory_entity_span", diagnostics
        if claim_words < 5:
            return "answer_unit_explanatory_too_short", diagnostics
        if not has_explanatory_signal and question_relevance < 0.16:
            return "answer_unit_low_question_relevance", diagnostics
    elif qtype in {"number", "date", "place"} and question_relevance < 0.10 and target_relevance < 0.90:
        return "answer_unit_low_question_relevance", diagnostics
    elif qtype in {"person", "open"} and target_source == "ranked_evidence" and question_relevance < 0.08:
        return "answer_unit_low_question_relevance", diagnostics

    if directness < min_directness:
        return "answer_unit_low_directness", diagnostics
    return "", diagnostics


def explanatory_precision_gate_reason(
    candidate: dict[str, Any],
    question: str,
    answer_unit_score_parts: dict[str, float],
    args: argparse.Namespace,
) -> str:
    if not getattr(args, "explanatory_precision_gate", False):
        return ""
    if infer_answer_format(args) == "qampari" or question_answer_type(question) != "explanation":
        return ""

    claim = normalize_space(str(candidate.get("claim", "") or ""))
    answer_span = normalize_space(str(candidate.get("answer_span", "") or ""))
    evidence_sentence = normalize_space(str(candidate.get("evidence_sentence", "") or ""))
    if not claim:
        return "low_explanatory_precision"

    try:
        target_relevance = float(answer_unit_score_parts.get("target_relevance", 0.0) or 0.0)
    except Exception:
        target_relevance = 0.0
    try:
        question_relevance = float(answer_unit_score_parts.get("question_relevance", 0.0) or 0.0)
    except Exception:
        question_relevance = 0.0

    min_target = float(getattr(args, "explanatory_precision_min_target_relevance", 0.58) or 0.58)
    min_question = float(getattr(args, "explanatory_precision_min_question_relevance", 0.18) or 0.18)
    signal_text = " ".join(part for part in [claim, evidence_sentence] if part)
    has_explanatory_signal = bool(EXPLANATORY_SIGNAL_PATTERN.search(signal_text))
    span_words = word_count(answer_span)
    claim_words = word_count(claim)

    if target_relevance >= min_target and question_relevance >= min_question:
        return ""
    if has_explanatory_signal and question_relevance >= min_question and claim_words >= 4:
        return ""
    if target_relevance >= 0.90 and claim_words >= 5 and span_words >= 3:
        return ""
    return "low_explanatory_precision"


class ExpansionCandidateGenerator:
    prompt_tokens = 0
    completion_tokens = 0

    def generate(
        self,
        item: dict[str, Any],
        verified_claims: Sequence[dict[str, Any]],
        rejected_claims: Sequence[dict[str, Any]],
        ranked_sentences: Sequence[dict[str, Any]],
        args: argparse.Namespace,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


class ExtractiveExpansionGenerator(ExpansionCandidateGenerator):
    def generate(
        self,
        item: dict[str, Any],
        verified_claims: Sequence[dict[str, Any]],
        rejected_claims: Sequence[dict[str, Any]],
        ranked_sentences: Sequence[dict[str, Any]],
        args: argparse.Namespace,
    ) -> list[dict[str, Any]]:
        question = str(item.get(args.question_field, "") or "")
        covered = covered_answer_spans(verified_claims)
        completion_targets = build_recall_completion_targets(
            question,
            verified_claims,
            rejected_claims,
            ranked_sentences,
            args,
        )
        answer_targeted_paragraph = (
            getattr(args, "answer_targeted_expansion", False) and infer_answer_format(args) != "qampari"
        )
        candidate_limit = int(args.max_expansion_candidates)
        if answer_targeted_paragraph:
            candidate_limit = min(candidate_limit, max(0, int(getattr(args, "answer_targeted_extractive_top_k", 2))))
        min_sentence_overlap = (
            float(getattr(args, "answer_targeted_min_sentence_overlap", 0.0))
            if answer_targeted_paragraph
            else 0.0
        )
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in ranked_sentences:
            if len(candidates) >= candidate_limit:
                break
            sentence = normalize_space(str(row.get("sentence", "") or ""))
            if min_sentence_overlap > 0 and overlap_score(question, sentence) < min_sentence_overlap:
                continue
            spans = candidate_answer_spans_from_sentence(sentence, verified_claims, args, question=question)
            answer_span = ""
            for span in spans:
                key = protected_span_key(span)
                if not args.skip_covered_answer_spans or key not in covered:
                    answer_span = span
                    break
            if not answer_span and args.skip_covered_answer_spans:
                continue
            if getattr(args, "recall_oriented_completion", False):
                matched, _reason, matched_target = candidate_matches_recall_completion_target(
                    {
                        "claim": sentence.rstrip("."),
                        "answer_span": answer_span,
                        "evidence_sentence": sentence,
                    },
                    completion_targets,
                    args,
                )
                if not matched:
                    continue
                target_gap = str(matched_target.get("target", "") if matched_target else "")
            else:
                target_gap = ""
            claim = compact_answer_unit_claim(sentence, answer_span, sentence, args)
            if not claim:
                continue
            key = protected_span_key(claim)
            if not key or key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "claim": claim,
                    "answer_span": answer_span,
                    "doc_id": int(row["doc_id"]),
                    "evidence_sentence": sentence,
                    "target_gap": target_gap,
                    "source": "extractive",
                    "rank_score": float(row["score"]),
                }
            )
        return candidates


def selected_span_ids_from_payload(payload: dict[str, Any], args: argparse.Namespace) -> list[str]:
    raw_ids = payload.get("selected_span_ids", [])
    if not isinstance(raw_ids, list):
        raw_ids = payload.get("span_ids", [])
    if not isinstance(raw_ids, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for raw_id in raw_ids:
        span_id = normalize_space(str(raw_id))
        if not span_id or not re.fullmatch(r"s\d+", span_id):
            continue
        if span_id in seen:
            continue
        seen.add(span_id)
        output.append(span_id)
        if len(output) >= max_selected_span_ids(args):
            break
    return output


def salvage_selected_span_ids(raw: str, args: argparse.Namespace) -> list[str]:
    """Recover span ids from malformed/truncated span-id JSON outputs."""

    output: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"\bs\d+\b", str(raw or "")):
        span_id = match.group(0)
        if span_id in seen:
            continue
        seen.add(span_id)
        output.append(span_id)
        if len(output) >= max_selected_span_ids(args):
            break
    return output


class LLMExpansionGenerator(ExpansionCandidateGenerator):
    def __init__(self, client: OpenAIChatClient, cache: JsonCache) -> None:
        self.client = client
        self.cache = cache
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def generate(
        self,
        item: dict[str, Any],
        verified_claims: Sequence[dict[str, Any]],
        rejected_claims: Sequence[dict[str, Any]],
        ranked_sentences: Sequence[dict[str, Any]],
        args: argparse.Namespace,
    ) -> list[dict[str, Any]]:
        question = str(item.get(args.question_field, "") or "")
        supported_answer = (
            supported_answer_summary(verified_claims)
            if getattr(args, "expansion_use_intermediate_answer", False)
            else "None."
        )
        supported = supported_claim_block(verified_claims)
        rejected = rejected_claim_block(rejected_claims)
        evidence = evidence_sentence_block(ranked_sentences, args.expansion_evidence_sentences)
        span_rows = candidate_answer_span_rows(ranked_sentences, verified_claims, args, question=question)
        candidate_spans = candidate_answer_span_block(span_rows)
        completion_targets = build_recall_completion_targets(
            question,
            verified_claims,
            rejected_claims,
            ranked_sentences,
            args,
        )
        completion_target_text = recall_completion_target_block(completion_targets)
        key = "v3-expand-v3:" + stable_hash(
            question,
            supported_answer,
            supported,
            rejected,
            evidence,
            candidate_spans,
            completion_target_text,
            str(args.expansion_include_extractive),
            str(getattr(args, "answer_targeted_extractive_top_k", 2)),
            str(getattr(args, "answer_targeted_min_sentence_overlap", 0.0)),
            str(getattr(args, "recall_oriented_completion", False)),
            str(getattr(args, "expansion_output_mode", "claims")),
            str(max_selected_span_ids(args)),
            str(getattr(args, "explanatory_title_span_mode", "off")),
            str(getattr(args, "max_explanatory_expanded_claims", 0)),
            str(getattr(args, "max_explanatory_expansion_verifications", 0)),
            str(getattr(args, "min_explanatory_answer_unit_score", 0.0)),
            str(getattr(args, "explanatory_precision_gate", False)),
            str(getattr(args, "explanatory_precision_min_target_relevance", 0.0)),
            str(getattr(args, "explanatory_precision_min_question_relevance", 0.0)),
            stable_hash(json.dumps(span_rows, ensure_ascii=False, sort_keys=True)),
        )
        cached = self.cache.get(key)
        if cached is not None:
            return list(cached or [])

        prompt_template = EXPANSION_SPAN_ID_USER if getattr(args, "expansion_output_mode", "claims") == "span_ids" else EXPANSION_USER
        prompt = prompt_template.format(
            question=question,
            supported_answer=supported_answer,
            supported_claims=supported,
            rejected_claims=rejected,
            evidence_sentences=evidence,
            candidate_answer_spans=candidate_spans,
            completion_targets=completion_target_text,
            max_selected_span_ids=max_selected_span_ids(args),
        )
        candidates: list[dict[str, Any]] = []
        try:
            raw, usage = self.client.chat(EXPANSION_SYSTEM, prompt, max_tokens=args.expansion_max_tokens)
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            try:
                payload = extract_json_object(raw)
                if getattr(args, "expansion_output_mode", "claims") == "span_ids":
                    raw_candidates = [{"span_id": span_id} for span_id in selected_span_ids_from_payload(payload, args)]
                else:
                    raw_candidates = payload.get("candidates", [])
                if isinstance(raw_candidates, list):
                    candidates = [row for row in raw_candidates if isinstance(row, dict)]
            except Exception as exc:
                if getattr(args, "expansion_output_mode", "claims") == "span_ids":
                    salvaged_ids = salvage_selected_span_ids(raw, args)
                    if salvaged_ids:
                        logger.warning(
                            "Expansion JSON parse failed; salvaged %d span ids. Error: %s",
                            len(salvaged_ids),
                            exc,
                        )
                        candidates = [{"span_id": span_id} for span_id in salvaged_ids]
                    else:
                        logger.warning("Expansion JSON parse failed; falling back to no LLM candidates. Error: %s", exc)
                else:
                    logger.warning("Expansion JSON parse failed; falling back to no LLM candidates. Error: %s", exc)
        except Exception as exc:
            logger.warning(
                "LLM expansion failed for one example; continuing with extractive fallback/no expansion. Error: %s",
                exc,
            )

        normalized = normalize_expansion_candidates(candidates, ranked_sentences, args, question=question, span_rows=span_rows)
        include_extractive = args.expansion_include_extractive and (
            not normalized
            or (getattr(args, "answer_targeted_expansion", False) and infer_answer_format(args) != "qampari")
        )
        if include_extractive:
            fallback = ExtractiveExpansionGenerator().generate(
                item,
                verified_claims,
                rejected_claims,
                ranked_sentences,
                args,
            )
            fallback_limit = max(0, args.max_expansion_candidates - len(normalized))
            if getattr(args, "answer_targeted_expansion", False) and infer_answer_format(args) != "qampari":
                fallback_limit = min(
                    fallback_limit,
                    max(0, int(getattr(args, "answer_targeted_extractive_top_k", 2))),
                )
            added_fallback = 0
            seen = {protected_span_key(str(row.get("claim", "") or "")) for row in normalized}
            for row in fallback:
                if added_fallback >= fallback_limit:
                    break
                key = protected_span_key(str(row.get("claim", "") or ""))
                if not key or key in seen:
                    continue
                row = dict(row)
                row["source"] = "extractive_fallback"
                normalized.append(row)
                seen.add(key)
                added_fallback += 1
                if len(normalized) >= args.max_expansion_candidates:
                    break
        self.cache.set(key, normalized)
        return normalized


def normalize_expansion_candidates(
    candidates: Sequence[dict[str, Any]],
    ranked_sentences: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    question: str = "",
    span_rows: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    by_doc: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in ranked_sentences:
        by_doc[int(row["doc_id"])].append(row)
    span_lookup = span_lookup_from_rows(span_rows or [])

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        claim = normalize_space(str(candidate.get("claim", "") or ""))
        answer_span = normalize_space(str(candidate.get("answer_span", "") or ""))
        span_id = normalize_space(str(candidate.get("span_id", "") or ""))
        target_gap = normalize_space(str(candidate.get("target_gap", "") or ""))
        evidence_sentence = normalize_space(str(candidate.get("evidence_sentence", "") or ""))
        span_row = span_lookup.get(span_id) if span_id else None
        candidate_rank_score = 0.0
        if span_row:
            doc_id = int(span_row["doc_id"])
            answer_span = normalize_space(str(span_row.get("span", "") or answer_span))
            evidence_sentence = normalize_space(str(span_row.get("sentence", "") or evidence_sentence))
            try:
                candidate_rank_score = float(span_row.get("score", 0.0) or 0.0)
            except Exception:
                candidate_rank_score = 0.0
        else:
            try:
                doc_id = int(candidate.get("doc_id"))
            except Exception:
                continue
            try:
                candidate_rank_score = float(candidate.get("rank_score", 0.0) or 0.0)
            except Exception:
                candidate_rank_score = 0.0
        if doc_id <= 0:
            continue
        if doc_id not in by_doc:
            continue
        if not evidence_sentence:
            doc_sentences = [normalize_space(str(row.get("sentence", ""))) for row in by_doc[doc_id]]
            if answer_span:
                answer_key = protected_span_key(answer_span)
                matched_sentences = [
                    sentence
                    for sentence in doc_sentences
                    if answer_key and answer_key in protected_span_key(sentence)
                ]
                if matched_sentences:
                    evidence_sentence = matched_sentences[0]
            if not evidence_sentence:
                evidence_sentence = doc_sentences[0]
        # Keep doc_id honest: the sentence must be close to one of the ranked
        # sentences from the same doc. If the LLM copied only part of the
        # sentence, this substring check still passes.
        candidates_for_doc = [normalize_space(str(row.get("sentence", ""))) for row in by_doc[doc_id]]
        if not any(evidence_sentence in sent or sent in evidence_sentence for sent in candidates_for_doc):
            if answer_span:
                answer_key = protected_span_key(answer_span)
                span_matches = [
                    sent for sent in candidates_for_doc if answer_key and answer_key in protected_span_key(sent)
                ]
                evidence_sentence = span_matches[0] if span_matches else candidates_for_doc[0]
            else:
                evidence_sentence = candidates_for_doc[0]
        if answer_span and protected_span_key(answer_span) not in protected_span_key(evidence_sentence) and answer_span.lower() not in claim.lower():
            answer_span = ""
        if infer_answer_format(args) != "qampari":
            compact_claim = compact_answer_unit_claim(evidence_sentence, answer_span, claim, args)
            if compact_claim:
                claim = compact_claim
            elif word_count(claim) > expansion_claim_word_limit(args):
                continue
        elif not claim:
            claim = answer_span or evidence_sentence
        if not claim:
            continue
        if getattr(args, "strict_expansion_answer_unit_gate", False):
            bad_reason = bad_expansion_surface_reason(
                claim,
                answer_span,
                evidence_sentence,
                question,
                getattr(args, "answer_type_gate_mode", "strict"),
            )
            if bad_reason:
                continue
        if infer_answer_format(args) == "qampari" and args.qampari_answer_relevance_filter:
            bad, _reason = qampari_is_bad_answer_span(answer_span or claim, question, evidence_sentence)
            if bad:
                continue
        elif getattr(args, "expansion_filter_general_answer_spans", False):
            bad, _reason = is_bad_general_answer_span(
                answer_span or claim,
                question,
                evidence_sentence,
                getattr(args, "answer_type_gate_mode", "strict"),
            )
            if bad:
                continue
        if getattr(args, "expansion_min_question_overlap", 0.0) > 0:
            relevance = max(overlap_score(question, claim), overlap_score(question, evidence_sentence))
            if relevance < float(args.expansion_min_question_overlap):
                continue
        key = protected_span_key(claim)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "claim": claim.rstrip("."),
                "answer_span": answer_span,
                "span_id": span_id,
                "doc_id": doc_id,
                "evidence_sentence": evidence_sentence,
                "target_gap": target_gap,
                "rank_score": candidate_rank_score,
                "source": "llm",
            }
        )
        if len(output) >= args.max_expansion_candidates:
            break
    return output


def make_expansion_generator(args: argparse.Namespace, cache: JsonCache) -> ExpansionCandidateGenerator:
    if args.expansion_mode == "off":
        return ExtractiveExpansionGenerator()
    if args.expansion_mode == "extractive":
        return ExtractiveExpansionGenerator()
    if not args.openai_api:
        logger.warning("`--expansion-mode llm` requires --openai-api; using extractive expansion instead.")
        return ExtractiveExpansionGenerator()
    client = OpenAIChatClient(
        model=args.expansion_model,
        temperature=args.llm_temperature,
        top_p=args.llm_top_p,
        max_retries=args.llm_max_retries,
    )
    return LLMExpansionGenerator(client, cache)


def verify_expansion_candidate(
    candidate: dict[str, Any],
    question: str,
    docs: Sequence[dict[str, Any]],
    verifier: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        doc_id = int(candidate.get("doc_id"))
    except Exception:
        return None, {"attempted": False, "reason": "invalid_doc_id"}
    if doc_id <= 0 or doc_id > len(docs):
        return None, {"attempted": False, "reason": "doc_id_out_of_range", "doc_id": doc_id}

    claim = normalize_space(str(candidate.get("claim", "") or ""))
    if not claim:
        return None, {"attempted": False, "reason": "empty_claim", "doc_id": doc_id}

    source_doc = docs[doc_id - 1]
    evidence_sentence = normalize_space(str(candidate.get("evidence_sentence", "") or ""))
    if infer_answer_format(args) == "qampari" and args.qampari_answer_relevance_filter:
        answer_span = normalize_space(str(candidate.get("answer_span", "") or ""))
        bad, reason = qampari_is_bad_answer_span(answer_span or claim, question, evidence_sentence)
        if bad:
            return None, {"attempted": False, "reason": reason, "doc_id": doc_id}
    elif getattr(args, "expansion_filter_general_answer_spans", False):
        answer_span = normalize_space(str(candidate.get("answer_span", "") or ""))
        bad, reason = is_bad_general_answer_span(
            answer_span or claim,
            question,
            evidence_sentence,
            getattr(args, "answer_type_gate_mode", "strict"),
        )
        if bad:
            return None, {"attempted": False, "reason": reason, "doc_id": doc_id}
    if getattr(args, "expansion_min_question_overlap", 0.0) > 0:
        relevance = max(overlap_score(question, claim), overlap_score(question, evidence_sentence))
        if relevance < float(args.expansion_min_question_overlap):
            return None, {
                "attempted": False,
                "reason": "low_question_relevance",
                "doc_id": doc_id,
                "question_relevance": relevance,
            }
    relevance_reason = ""
    relevance_diagnostics: dict[str, Any] = {}
    if getattr(args, "answer_unit_relevance_gate", False):
        relevance_reason, relevance_diagnostics = answer_unit_directness_diagnostics(candidate, question, args)
        if relevance_reason:
            return None, {
                "attempted": False,
                "reason": relevance_reason,
                "doc_id": doc_id,
                **relevance_diagnostics,
            }
    if args.verify_expansion_with_sentence and evidence_sentence:
        verify_doc_payload = {
            args.title_field: source_doc.get(args.title_field, ""),
            args.text_field: evidence_sentence,
        }
    else:
        verify_doc_payload = source_doc

    result: VerificationResult = verifier.verify(question, claim, [(doc_id, verify_doc_payload)], args)
    result.evidence_scope = "v3_expansion_single_doc"
    diagnostics = {
        "attempted": True,
        "doc_id": doc_id,
        "label": result.label,
        "confidence": result.confidence,
        "entailment": result.entailment,
        "answer_unit_score": float(candidate.get("answer_unit_score", 0.0) or 0.0),
        "answer_unit_score_parts": candidate.get("answer_unit_score_parts", {}),
        **relevance_diagnostics,
        "rationale": result.rationale,
    }
    if result.label != "supported":
        return None, diagnostics

    answer_span = normalize_space(str(candidate.get("answer_span", "") or ""))
    citation_ids = unique_positive_ints(result.evidence_doc_ids or [doc_id]) or [doc_id]
    claim_record = {
        "claim_id": f"v3-expanded-{stable_hash(question, claim, str(doc_id))[:12]}",
        "claim": claim,
        "importance": "critical",
        "claim_type": "answer_expansion",
        "claim_extractor": f"v3_{candidate.get('source', args.expansion_mode)}",
        "decontextualized_sentence": None,
        "selection_status": None,
        "disambiguation_status": None,
        "claimify_vote_count": None,
        "claimify_num_completions": None,
        "source_sentence_id": 100000 + doc_id,
        "source_sentence": f"{evidence_sentence} {format_citations([doc_id])}".strip(),
        "source_sentence_without_citations": evidence_sentence,
        "citation_ids": [doc_id],
        "source_citation_ids": [doc_id],
        "source_citation_text": format_citations([doc_id]),
        "label": result.label,
        "confidence": result.confidence,
        "entailment": result.entailment,
        "neutral": result.neutral,
        "contradiction": result.contradiction,
        "rationale": "Expanded by iterative evidence search. " + result.rationale,
        "verifier": f"{result.verifier}+v3_expansion",
        "evidence_scope": result.evidence_scope,
        "evidence_doc_ids": citation_ids,
        "final_citation_ids": citation_ids,
        "v2_status": "expanded",
        "v3_status": "expanded_supported",
        "v3_answer_span": answer_span,
        "v3_target_gap": normalize_space(str(candidate.get("target_gap", "") or "")),
        "v3_target_source": normalize_space(str(candidate.get("target_source", "") or "")),
        "v3_answer_unit_score": float(candidate.get("answer_unit_score", 0.0) or 0.0),
        "v3_answer_unit_score_parts": candidate.get("answer_unit_score_parts", {}),
        "v3_answer_unit_directness": float(relevance_diagnostics.get("answer_unit_directness", 0.0) or 0.0),
        "v3_answer_unit_question_relevance": float(
            relevance_diagnostics.get("answer_unit_question_relevance", 0.0) or 0.0
        ),
        "v3_answer_unit_target_relevance": float(
            relevance_diagnostics.get("answer_unit_target_relevance", 0.0) or 0.0
        ),
        "v3_answer_unit_has_explanatory_signal": bool(
            relevance_diagnostics.get("answer_unit_has_explanatory_signal", False)
        ),
        "protected_answer_spans": [span for span in [answer_span] if span],
    }
    if not claim_record["protected_answer_spans"]:
        claim_record["protected_answer_spans"] = extract_protected_spans(claim + " " + evidence_sentence)[:3]
    return claim_record, diagnostics


def expand_verified_claims(
    item: dict[str, Any],
    draft_audit: dict[str, Any],
    verified_claims: Sequence[dict[str, Any]],
    rejected_claims: Sequence[dict[str, Any]],
    verifier: Any,
    expansion_generator: ExpansionCandidateGenerator,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    del draft_audit
    if args.expansion_mode == "off":
        return list(verified_claims), [], [], []

    ranked_sentences = rank_evidence_sentences(item, verified_claims, rejected_claims, args)
    candidates = expansion_generator.generate(item, verified_claims, rejected_claims, ranked_sentences, args)
    question = str(item.get(args.question_field, "") or "")
    docs = item.get(args.docs_field, []) or []
    verification_budget = max_expansion_verifications_for_question(args, question)
    expanded_budget = max_expanded_claims_for_question(args, question)
    min_answer_unit_score = min_answer_unit_score_for_question(args, question)
    completion_targets = build_recall_completion_targets(
        question,
        verified_claims,
        rejected_claims,
        ranked_sentences,
        args,
    )
    existing_keys = {normalize_revision_claim_key(claim, args) for claim in verified_claims}
    existing_spans = covered_answer_spans(verified_claims)
    expanded: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []

    for candidate in candidates:
        answer_span_key = protected_span_key(str(candidate.get("answer_span", "") or ""))
        if args.require_expansion_answer_span and not answer_span_key:
            attempts.append({"candidate": candidate, "attempted": False, "reason": "missing_answer_span"})
            continue
        if args.skip_covered_answer_spans and answer_span_key and answer_span_key in existing_spans:
            attempts.append({"candidate": candidate, "attempted": False, "reason": "answer_span_already_covered"})
            continue
        matched, reason, matched_target = candidate_matches_recall_completion_target(candidate, completion_targets, args)
        if not matched:
            attempts.append({"candidate": candidate, "attempted": False, "reason": reason})
            continue
        if matched_target is not None:
            candidate = dict(candidate)
            candidate["target_gap"] = str(matched_target.get("target", "") or "")
            candidate["target_source"] = str(matched_target.get("source", "") or "")
        answer_unit_score, answer_unit_score_parts = expansion_candidate_answer_unit_score(
            candidate,
            question,
            completion_targets,
            args,
        )
        candidate = dict(candidate)
        candidate["answer_unit_score"] = answer_unit_score
        candidate["answer_unit_score_parts"] = answer_unit_score_parts
        if getattr(args, "strict_expansion_answer_unit_gate", False):
            surface_reason = bad_expansion_surface_reason(
                str(candidate.get("claim", "") or ""),
                str(candidate.get("answer_span", "") or ""),
                str(candidate.get("evidence_sentence", "") or ""),
                question,
                getattr(args, "answer_type_gate_mode", "strict"),
            )
            if surface_reason:
                attempts.append(
                    {
                        "candidate": candidate,
                        "attempted": False,
                        "reason": surface_reason,
                        "answer_unit_score": answer_unit_score,
                        "answer_unit_score_parts": answer_unit_score_parts,
                    }
                )
                continue
            target_relevance = float(answer_unit_score_parts.get("target_relevance", 0.0) or 0.0)
            min_target_relevance = float(getattr(args, "strict_expansion_min_target_relevance", 0.45))
            if getattr(args, "recall_oriented_completion", False) and target_relevance < min_target_relevance:
                attempts.append(
                    {
                        "candidate": candidate,
                        "attempted": False,
                        "reason": "low_target_relevance",
                        "answer_unit_score": answer_unit_score,
                        "answer_unit_score_parts": answer_unit_score_parts,
                    }
                )
                continue
        if answer_unit_score < min_answer_unit_score:
            attempts.append(
                {
                    "candidate": candidate,
                    "attempted": False,
                    "reason": "low_answer_unit_score",
                    "answer_unit_score": answer_unit_score,
                    "answer_unit_score_parts": answer_unit_score_parts,
                }
            )
            continue
        explanatory_precision_reason = explanatory_precision_gate_reason(
            candidate,
            question,
            answer_unit_score_parts,
            args,
        )
        if explanatory_precision_reason:
            attempts.append(
                {
                    "candidate": candidate,
                    "attempted": False,
                    "reason": explanatory_precision_reason,
                    "answer_unit_score": answer_unit_score,
                    "answer_unit_score_parts": answer_unit_score_parts,
                }
            )
            continue
        if sum(1 for attempt in attempts if attempt.get("attempted")) >= verification_budget:
            attempts.append({"candidate": candidate, "attempted": False, "reason": "verification_budget_exhausted"})
            continue
        verified, diagnostics = verify_expansion_candidate(candidate, question, docs, verifier, args)
        attempts.append({"candidate": candidate, **diagnostics, "recovered": verified is not None})
        if verified is None:
            continue
        key = normalize_revision_claim_key(verified, args)
        if not key or key in existing_keys:
            continue
        existing_keys.add(key)
        for span in verified.get("protected_answer_spans", []) or []:
            span_key = protected_span_key(span)
            if span_key:
                existing_spans.add(span_key)
        expanded.append(verified)
        if len(expanded) >= expanded_budget:
            break

    if args.preserve_first_pass_claims:
        base_claims = list(verified_claims)
        room = max(0, final_claim_limit(args) - len(base_claims))
        selected_expanded = expanded[:room]
        limited = base_claims + selected_expanded
    else:
        combined = list(verified_claims) + expanded
        combined.sort(key=v2_claim_priority_key)
        limited = combined[: final_claim_limit(args)]
        limited_keys_for_expansion = {normalize_revision_claim_key(claim, args) for claim in limited}
        selected_expanded = [
            claim for claim in expanded if normalize_revision_claim_key(claim, args) in limited_keys_for_expansion
        ]
    limited_keys = {normalize_revision_claim_key(claim, args) for claim in limited}
    overflow = [
        claim
        for claim in list(verified_claims) + expanded
        if normalize_revision_claim_key(claim, args) not in limited_keys
    ]
    return limited, selected_expanded, attempts, overflow


def tag_expansion_round(
    claims: Sequence[dict[str, Any]],
    attempts: Sequence[dict[str, Any]],
    *,
    round_index: int,
    phase: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tagged_claims: list[dict[str, Any]] = []
    for claim in claims:
        row = dict(claim)
        row["v3_iteration_round"] = round_index
        row["v3_expansion_phase"] = phase
        tagged_claims.append(row)
    tagged_attempts: list[dict[str, Any]] = []
    for attempt in attempts:
        row = dict(attempt)
        row["expansion_round"] = round_index
        row["expansion_phase"] = phase
        tagged_attempts.append(row)
    return tagged_claims, tagged_attempts


def iterative_completion_args(args: argparse.Namespace, round_index: int, remaining_room: int) -> argparse.Namespace:
    round_args = copy.copy(args)
    round_args.recall_oriented_completion = True
    round_args.expansion_use_intermediate_answer = True
    round_args.expansion_query_source = "question_evidence_and_rejected"

    per_round_claims = max(1, int(getattr(args, "iterative_completion_max_new_claims_per_round", 2) or 2))
    round_args.max_expanded_claims = max(0, min(per_round_claims, remaining_room))
    if int(getattr(args, "max_explanatory_expanded_claims", 0) or 0) > 0:
        round_args.max_explanatory_expanded_claims = max(0, min(per_round_claims, remaining_room))

    per_round_verifications = int(getattr(args, "iterative_completion_max_verifications_per_round", 0) or 0)
    if per_round_verifications > 0:
        round_args.max_expansion_verifications = max(1, per_round_verifications)
        if int(getattr(args, "max_explanatory_expansion_verifications", 0) or 0) > 0:
            round_args.max_explanatory_expansion_verifications = max(1, per_round_verifications)

    evidence_sentences = int(getattr(args, "iterative_completion_evidence_sentences", 0) or 0)
    if evidence_sentences > 0:
        round_args.expansion_evidence_sentences = evidence_sentences
    candidate_spans = int(getattr(args, "iterative_completion_candidate_spans", 0) or 0)
    if candidate_spans > 0:
        round_args.expansion_candidate_spans = candidate_spans
    max_candidates = int(getattr(args, "iterative_completion_max_candidates", 0) or 0)
    if max_candidates > 0:
        round_args.max_expansion_candidates = max_candidates

    return round_args


def run_iterative_completion_rounds(
    item: dict[str, Any],
    item_id: int,
    draft_audit: dict[str, Any],
    verified_claims: Sequence[dict[str, Any]],
    rejected_claims: Sequence[dict[str, Any]],
    verifier: Any,
    expansion_generator: ExpansionCandidateGenerator,
    candidate_provider: CandidateDocProvider | None,
    args: argparse.Namespace,
    dynamic_retrieval_diagnostics: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rounds = max(0, int(getattr(args, "iterative_completion_rounds", 0) or 0))
    if rounds <= 0 or args.expansion_mode == "off":
        return list(verified_claims), [], [], []

    current_verified = list(verified_claims)
    extra_expanded: list[dict[str, Any]] = []
    extra_attempts: list[dict[str, Any]] = []
    extra_overflow: list[dict[str, Any]] = []
    seen_keys = {normalize_revision_claim_key(claim, args) for claim in current_verified}

    for round_index in range(1, rounds + 1):
        remaining_room = max(0, final_claim_limit(args) - len(current_verified))
        if args.preserve_first_pass_claims and remaining_room <= 0:
            extra_attempts.append(
                {
                    "attempted": False,
                    "reason": "iterative_completion_no_answer_budget",
                    "expansion_round": round_index,
                    "expansion_phase": "iterative_completion",
                }
            )
            break

        round_args = iterative_completion_args(args, round_index, remaining_room or final_claim_limit(args))
        if int(getattr(round_args, "max_expanded_claims", 0) or 0) <= 0:
            break
        if dynamic_retrieval_enabled(round_args):
            dynamic_diag = apply_dynamic_retrieval_to_item(
                item,
                item_id,
                candidate_provider,
                current_verified,
                rejected_claims,
                round_args,
                phase="iterative_completion",
                round_index=round_index,
            )
            if dynamic_retrieval_diagnostics is not None:
                dynamic_retrieval_diagnostics.append(dynamic_diag)

        before_keys = {normalize_revision_claim_key(claim, args) for claim in current_verified}
        next_verified, round_expanded, round_attempts, round_overflow = expand_verified_claims(
            item,
            draft_audit,
            current_verified,
            rejected_claims,
            verifier,
            expansion_generator,
            round_args,
        )
        round_expanded, round_attempts = tag_expansion_round(
            round_expanded,
            round_attempts,
            round_index=round_index,
            phase="iterative_completion",
        )
        round_overflow = [
            {
                **dict(claim),
                "v3_iteration_round": round_index,
                "v3_expansion_phase": "iterative_completion_overflow",
            }
            for claim in round_overflow
        ]

        accepted_round_claims: list[dict[str, Any]] = []
        for claim in round_expanded:
            key = normalize_revision_claim_key(claim, args)
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            accepted_round_claims.append(claim)

        extra_attempts.extend(round_attempts)
        extra_overflow.extend(round_overflow)
        if not accepted_round_claims:
            break

        next_by_key = {normalize_revision_claim_key(claim, args): claim for claim in next_verified}
        current_verified = [
            dict(next_by_key.get(normalize_revision_claim_key(claim, args), claim))
            for claim in current_verified
            if normalize_revision_claim_key(claim, args)
        ]
        for claim in accepted_round_claims:
            key = normalize_revision_claim_key(claim, args)
            if key and key not in before_keys:
                current_verified.append(claim)
        extra_expanded.extend(accepted_round_claims)

    return current_verified, extra_expanded, extra_attempts, extra_overflow


def compute_v3_metrics(
    draft_audit: dict[str, Any],
    verified_claims: Sequence[dict[str, Any]],
    rejected_claims: Sequence[dict[str, Any]],
    recovered_claims: Sequence[dict[str, Any]],
    recovery_attempts: Sequence[dict[str, Any]],
    expanded_claims: Sequence[dict[str, Any]],
    expansion_attempts: Sequence[dict[str, Any]],
    expansion_overflow: Sequence[dict[str, Any]],
    revised_output: str,
    doc_pool_size: int,
    raw_doc_count: int,
) -> dict[str, Any]:
    metrics = compute_v2_metrics(
        draft_audit,
        verified_claims,
        rejected_claims,
        recovered_claims,
        recovery_attempts,
        revised_output,
        doc_pool_size,
        raw_doc_count,
    )
    draft_claims = draft_audit.get("claims", []) or []
    draft_aligned_kept = max(0, len(verified_claims) - len(expanded_claims))
    attempted = sum(1 for attempt in expansion_attempts if attempt.get("attempted"))
    successful = sum(1 for attempt in expansion_attempts if attempt.get("recovered"))
    recall_target_mismatches = sum(
        1 for attempt in expansion_attempts if attempt.get("reason") == "recall_completion_target_mismatch"
    )
    recall_no_targets = sum(
        1 for attempt in expansion_attempts if attempt.get("reason") == "recall_completion_no_targets"
    )
    low_answer_unit_score = sum(
        1 for attempt in expansion_attempts if attempt.get("reason") == "low_answer_unit_score"
    )
    low_target_relevance = sum(
        1 for attempt in expansion_attempts if attempt.get("reason") == "low_target_relevance"
    )
    low_explanatory_precision = sum(
        1 for attempt in expansion_attempts if attempt.get("reason") == "low_explanatory_precision"
    )
    surface_filtered_reasons = {
        "web_boilerplate",
        "page_date_like",
        "truncated_claim",
        "leading_connector",
        "leading_lowercase_fragment",
        "unbalanced_quote",
        "unbalanced_parenthesis",
        "span_not_anchored",
        "too_short_non_numeric",
        "whole_sentence_claim",
        "web_boilerplate_context",
        "explanatory_atomic_span",
        "explanatory_entity_span",
        "person_question_non_person_span",
        "person_question_leading_adverb",
        "person_answer_too_long",
        "date_question_non_temporal_span",
        "number_question_non_numeric_span",
        "number_question_year_span",
        "place_question_non_place_span",
    }
    surface_filtered = sum(
        1 for attempt in expansion_attempts if attempt.get("reason") in surface_filtered_reasons
    )
    answer_unit_filtered = sum(
        1 for attempt in expansion_attempts if str(attempt.get("reason", "") or "").startswith("answer_unit_")
    )
    iterative_attempts = [attempt for attempt in expansion_attempts if int(attempt.get("expansion_round", 0) or 0) > 0]
    iterative_expanded = [claim for claim in expanded_claims if int(claim.get("v3_iteration_round", 0) or 0) > 0]
    iterative_successful = sum(1 for attempt in iterative_attempts if attempt.get("recovered"))
    metrics.update(
        {
            "draft_aligned_kept_claims": draft_aligned_kept,
            "verified_plus_expanded_claims": len(verified_claims),
            "claim_keep_rate": draft_aligned_kept / len(draft_claims) if draft_claims else 0.0,
            "expanded_claims": len(expanded_claims),
            "expansion_candidate_claims": len(expansion_attempts),
            "expansion_attempted_claims": attempted,
            "expansion_successful_claims": successful,
            "expansion_success_rate": successful / attempted if attempted else 0.0,
            "expansion_overflow_claims": len(expansion_overflow),
            "expanded_answer_spans": sum(len(claim.get("protected_answer_spans", []) or []) for claim in expanded_claims),
            "recall_completion_target_mismatches": recall_target_mismatches,
            "recall_completion_no_targets": recall_no_targets,
            "expansion_low_answer_unit_score": low_answer_unit_score,
            "expansion_low_target_relevance": low_target_relevance,
            "expansion_low_explanatory_precision": low_explanatory_precision,
            "expansion_surface_filtered": surface_filtered,
            "expansion_answer_unit_filtered": answer_unit_filtered,
            "iterative_completion_claims": len(iterative_expanded),
            "iterative_completion_attempts": len(iterative_attempts),
            "iterative_completion_successful_claims": iterative_successful,
            "iterative_completion_success_rate": iterative_successful / len(iterative_attempts) if iterative_attempts else 0.0,
        }
    )
    return metrics


def summarize_v3(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    records = [item.get("cover_v3", {}) for item in items if item.get("cover_v3")]
    metrics = [record.get("revision_metrics", {}) for record in records]
    if not metrics:
        return {"num_examples": 0}

    total_draft = sum(int(m.get("draft_num_claims", 0)) for m in metrics)
    total_initial = sum(int(m.get("initially_supported_claims", 0)) for m in metrics)
    total_recovered = sum(int(m.get("recovered_claims", 0)) for m in metrics)
    total_kept = sum(int(m.get("kept_verified_claims", 0)) for m in metrics)
    total_rejected = sum(int(m.get("rejected_claims", 0)) for m in metrics)
    total_recovery_attempted = sum(int(m.get("recovery_attempted_claims", 0)) for m in metrics)
    total_recovery_successful = sum(int(m.get("recovery_successful_claims", 0)) for m in metrics)
    total_expanded = sum(int(m.get("expanded_claims", 0)) for m in metrics)
    total_draft_aligned_kept = sum(int(m.get("draft_aligned_kept_claims", 0)) for m in metrics)
    total_verified_plus_expanded = sum(int(m.get("verified_plus_expanded_claims", 0)) for m in metrics)
    total_expansion_candidates = sum(int(m.get("expansion_candidate_claims", 0)) for m in metrics)
    total_expansion_attempted = sum(int(m.get("expansion_attempted_claims", 0)) for m in metrics)
    total_expansion_successful = sum(int(m.get("expansion_successful_claims", 0)) for m in metrics)
    total_expanded_spans = sum(int(m.get("expanded_answer_spans", 0)) for m in metrics)
    total_recall_target_mismatches = sum(int(m.get("recall_completion_target_mismatches", 0)) for m in metrics)
    total_recall_no_targets = sum(int(m.get("recall_completion_no_targets", 0)) for m in metrics)
    total_low_answer_unit_score = sum(int(m.get("expansion_low_answer_unit_score", 0)) for m in metrics)
    total_low_target_relevance = sum(int(m.get("expansion_low_target_relevance", 0)) for m in metrics)
    total_low_explanatory_precision = sum(int(m.get("expansion_low_explanatory_precision", 0)) for m in metrics)
    total_surface_filtered = sum(int(m.get("expansion_surface_filtered", 0)) for m in metrics)
    total_answer_unit_filtered = sum(int(m.get("expansion_answer_unit_filtered", 0)) for m in metrics)
    total_iterative_claims = sum(int(m.get("iterative_completion_claims", 0)) for m in metrics)
    total_iterative_attempts = sum(int(m.get("iterative_completion_attempts", 0)) for m in metrics)
    total_iterative_successful = sum(int(m.get("iterative_completion_successful_claims", 0)) for m in metrics)
    total_dynamic_rounds = sum(int(m.get("dynamic_retrieval_rounds", 0)) for m in metrics)
    total_dynamic_queries = sum(int(m.get("dynamic_retrieval_queries", 0)) for m in metrics)
    total_dynamic_added_docs = sum(int(m.get("dynamic_retrieval_added_docs", 0)) for m in metrics)
    total_dynamic_successful_rounds = sum(int(m.get("dynamic_retrieval_successful_rounds", 0)) for m in metrics)
    total_explanation_render_examples = sum(
        1
        for record in records
        if record.get("render_diagnostics", {}).get("mode") == "explanation_integrated_refinement"
    )
    total_explanation_render_selected = sum(
        int(record.get("render_diagnostics", {}).get("selected_expansion_claims", 0)) for record in records
    )
    total_explanation_render_length_limited = sum(
        int(record.get("render_diagnostics", {}).get("length_limited_expansion_claims", 0)) for record in records
    )
    candidate_selection_records = [
        record.get("candidate_selection_diagnostics", {})
        for record in records
        if record.get("candidate_selection_diagnostics", {})
    ]
    total_candidate_selection_examples = sum(1 for diag in candidate_selection_records if diag.get("enabled"))
    total_candidate_selection_changed = sum(1 for diag in candidate_selection_records if diag.get("changed_output"))
    total_candidate_selection_candidates = sum(int(diag.get("num_candidates", 0) or 0) for diag in candidate_selection_records)
    total_candidate_selection_audit_failures = sum(
        int(diag.get("audit_failures", 0) or 0) for diag in candidate_selection_records
    )
    selected_label_counter: collections.Counter[str] = collections.Counter(
        str(diag.get("selected_label", "unknown") or "unknown") for diag in candidate_selection_records if diag.get("enabled")
    )
    lengths = [float(m.get("revised_length", 0.0)) for m in metrics]
    citations = [float(m.get("revised_num_citations", 0.0)) for m in metrics]
    doc_pool_sizes = [float(m.get("doc_pool_size", 0.0)) for m in metrics]

    label_counter: collections.Counter[str] = collections.Counter()
    for record in records:
        for claim in record.get("draft_audit", {}).get("claims", []) or []:
            label_counter[str(claim.get("label", "unknown"))] += 1

    return {
        "num_examples": len(records),
        "draft_num_claims": total_draft,
        "initially_supported_claims": total_initial,
        "recovered_claims": total_recovered,
        "expanded_claims": total_expanded,
        "draft_aligned_kept_claims": total_draft_aligned_kept,
        "verified_plus_expanded_claims": total_verified_plus_expanded,
        "kept_verified_claims": total_kept,
        "rejected_claims": total_rejected,
        "recovery_attempted_claims": total_recovery_attempted,
        "recovery_successful_claims": total_recovery_successful,
        "recovery_success_rate_micro": total_recovery_successful / total_recovery_attempted if total_recovery_attempted else 0.0,
        "expansion_candidate_claims": total_expansion_candidates,
        "expansion_attempted_claims": total_expansion_attempted,
        "expansion_successful_claims": total_expansion_successful,
        "expansion_success_rate_micro": total_expansion_successful / total_expansion_attempted if total_expansion_attempted else 0.0,
        "expanded_answer_spans": total_expanded_spans,
        "recall_completion_target_mismatches": total_recall_target_mismatches,
        "recall_completion_no_targets": total_recall_no_targets,
        "expansion_low_answer_unit_score": total_low_answer_unit_score,
        "expansion_low_target_relevance": total_low_target_relevance,
        "expansion_low_explanatory_precision": total_low_explanatory_precision,
        "expansion_surface_filtered": total_surface_filtered,
        "expansion_answer_unit_filtered": total_answer_unit_filtered,
        "iterative_completion_claims": total_iterative_claims,
        "iterative_completion_attempts": total_iterative_attempts,
        "iterative_completion_successful_claims": total_iterative_successful,
        "iterative_completion_success_rate_micro": (
            total_iterative_successful / total_iterative_attempts if total_iterative_attempts else 0.0
        ),
        "dynamic_retrieval_rounds": total_dynamic_rounds,
        "dynamic_retrieval_queries": total_dynamic_queries,
        "dynamic_retrieval_added_docs": total_dynamic_added_docs,
        "dynamic_retrieval_successful_rounds": total_dynamic_successful_rounds,
        "dynamic_retrieval_added_docs_per_example": total_dynamic_added_docs / len(records) if records else 0.0,
        "explanation_render_examples": total_explanation_render_examples,
        "explanation_render_selected_claims": total_explanation_render_selected,
        "explanation_render_length_limited_claims": total_explanation_render_length_limited,
        "candidate_selection_examples": total_candidate_selection_examples,
        "candidate_selection_changed_outputs": total_candidate_selection_changed,
        "candidate_selection_total_candidates": total_candidate_selection_candidates,
        "candidate_selection_audit_failures": total_candidate_selection_audit_failures,
        "candidate_selection_selected_labels": dict(selected_label_counter),
        "claim_keep_rate_micro": total_draft_aligned_kept / total_draft if total_draft else 0.0,
        "rejected_claim_rate_micro": total_rejected / total_draft if total_draft else 0.0,
        "avg_revised_length": statistics.mean(lengths) if lengths else 0.0,
        "avg_revised_num_citations": statistics.mean(citations) if citations else 0.0,
        "avg_doc_pool_size": statistics.mean(doc_pool_sizes) if doc_pool_sizes else 0.0,
        "draft_label_counts": dict(label_counter),
    }


def summarize_final_audit_v3(items: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    final_audits = [item.get("cover_v3", {}).get("final_audit") for item in items]
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


class LockedVerifier:
    """Serialize shared verifier calls when the backend owns heavyweight model state."""

    def __init__(self, verifier: Any) -> None:
        self.verifier = verifier
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.verifier, name)

    def verify(self, *args: Any, **kwargs: Any) -> VerificationResult:
        with self._lock:
            return self.verifier.verify(*args, **kwargs)


class RevisionWorkerContext:
    def __init__(self, args: argparse.Namespace, cache: JsonCache) -> None:
        self.args = args
        self.cache = cache
        self._local = threading.local()
        self._states: list[tuple[Any, Any, ExpansionCandidateGenerator]] = []
        self._states_lock = threading.Lock()
        self._shared_verifier: Any | None = None
        if args.verifier in {"nli", "hybrid"}:
            verifier = make_verifier(args, cache)
            self._shared_verifier = LockedVerifier(verifier) if max(1, int(args.workers or 1)) > 1 else verifier

    def get(self) -> tuple[Any, Any, ExpansionCandidateGenerator]:
        state = getattr(self._local, "state", None)
        if state is None:
            decomposer = make_decomposer(self.args, self.cache)
            verifier = self._shared_verifier if self._shared_verifier is not None else make_verifier(self.args, self.cache)
            expansion_generator = make_expansion_generator(self.args, self.cache)
            state = (decomposer, verifier, expansion_generator)
            self._local.state = state
            with self._states_lock:
                self._states.append(state)
        return state

    def states(self) -> list[tuple[Any, Any, ExpansionCandidateGenerator]]:
        with self._states_lock:
            return list(self._states)


def merge_token_usage(total: dict[str, int], usage: dict[str, int]) -> None:
    for key, value in usage.items():
        total[key] = total.get(key, 0) + int(value)


def collect_worker_token_usage(worker_context: RevisionWorkerContext) -> dict[str, int]:
    token_usage: dict[str, int] = {}
    seen_decomposers: set[int] = set()
    seen_verifiers: set[int] = set()
    seen_expansion_generators: set[int] = set()
    placeholder = object()
    revision_generator = AutoRevisionGenerator()

    for decomposer, verifier, expansion_generator in worker_context.states():
        if id(decomposer) not in seen_decomposers:
            merge_token_usage(token_usage, collect_token_usage(decomposer, placeholder, revision_generator))
            seen_decomposers.add(id(decomposer))
        if id(verifier) not in seen_verifiers:
            merge_token_usage(token_usage, collect_token_usage(placeholder, verifier, revision_generator))
            seen_verifiers.add(id(verifier))
        if id(expansion_generator) not in seen_expansion_generators:
            prompt_tokens = getattr(expansion_generator, "prompt_tokens", None)
            completion_tokens = getattr(expansion_generator, "completion_tokens", None)
            if prompt_tokens is not None:
                token_usage["expansion_prompt_tokens"] = token_usage.get("expansion_prompt_tokens", 0) + int(prompt_tokens)
            if completion_tokens is not None:
                token_usage["expansion_completion_tokens"] = token_usage.get("expansion_completion_tokens", 0) + int(completion_tokens)
            seen_expansion_generators.add(id(expansion_generator))
    return token_usage


def revise_item_v3(
    item: dict[str, Any],
    item_id: int,
    candidate_provider: CandidateDocProvider,
    decomposer: Any,
    verifier: Any,
    expansion_generator: ExpansionCandidateGenerator,
    args: argparse.Namespace,
) -> dict[str, Any]:
    updated = dict(item)
    draft_output = str(updated.get(args.output_field, "") or "")
    updated["cover_v3_original_output"] = draft_output
    raw_docs = list(updated.get(args.docs_field, []) or [])
    candidate_docs = candidate_provider.docs_for(updated, item_id)
    doc_pool = merged_doc_pool(raw_docs, candidate_docs, args)
    updated[args.docs_field] = doc_pool

    draft_audit = audit_item(updated, item_id, decomposer, verifier, args)
    first_pass_claims, rejected_claims, recovered_claims, recovery_attempts = prepare_claims_for_v2(
        draft_audit,
        updated,
        verifier,
        args,
    )
    dynamic_retrieval_diagnostics: list[dict[str, Any]] = []
    if dynamic_retrieval_enabled(args):
        dynamic_retrieval_diagnostics.append(
            apply_dynamic_retrieval_to_item(
                updated,
                item_id,
                candidate_provider,
                first_pass_claims,
                rejected_claims,
                args,
                phase="initial_expansion",
                round_index=0,
            )
        )
    verified_claims, expanded_claims, expansion_attempts, expansion_overflow = expand_verified_claims(
        updated,
        draft_audit,
        first_pass_claims,
        rejected_claims,
        verifier,
        expansion_generator,
        args,
    )
    expanded_claims, expansion_attempts = tag_expansion_round(
        expanded_claims,
        expansion_attempts,
        round_index=0,
        phase="initial_expansion",
    )
    tagged_initial_expansions = {normalize_revision_claim_key(claim, args): claim for claim in expanded_claims}
    verified_claims = [
        tagged_initial_expansions.get(normalize_revision_claim_key(claim, args), claim) for claim in verified_claims
    ]
    if int(getattr(args, "iterative_completion_rounds", 0) or 0) > 0:
        (
            verified_claims,
            iterative_expanded_claims,
            iterative_expansion_attempts,
            iterative_expansion_overflow,
        ) = run_iterative_completion_rounds(
            updated,
            item_id,
            draft_audit,
            verified_claims,
            rejected_claims,
            verifier,
            expansion_generator,
            candidate_provider,
            args,
            dynamic_retrieval_diagnostics,
        )
        expanded_claims.extend(iterative_expanded_claims)
        expansion_attempts.extend(iterative_expansion_attempts)
        expansion_overflow.extend(iterative_expansion_overflow)
    verified_claims, rejected_claims = filter_qampari_claims_for_answer_relevance(
        verified_claims,
        rejected_claims,
        str(updated.get(args.question_field, "") or ""),
        args,
    )
    verified_keys_after_relevance = {normalize_revision_claim_key(claim, args) for claim in verified_claims}
    expanded_claims = [
        claim
        for claim in expanded_claims
        if normalize_revision_claim_key(claim, args) in verified_keys_after_relevance
    ]

    render_diagnostics: dict[str, Any] = {}
    if infer_answer_format(args) == "qampari":
        revised_output = AutoRevisionGenerator().revise(updated, verified_claims, rejected_claims, args)
    elif args.render_mode == "atomic":
        revised_output = render_atomic_answer(updated, verified_claims, args)
    elif args.render_mode == "span_preserving":
        revised_output = render_span_preserving_answer(updated, draft_audit, verified_claims, args)
    elif args.render_mode == "answer_preserving":
        integrated = render_explanation_integrated_answer(updated, draft_audit, verified_claims, expanded_claims, args)
        if integrated is not None:
            revised_output, render_diagnostics = integrated
        else:
            revised_output = render_answer_preserving_answer(updated, draft_audit, verified_claims, args)
    else:
        revised_output = render_citation_aligned_answer(updated, draft_audit, verified_claims, args)
    if (
        args.fallback_template_on_empty
        and verified_claims
        and (not revised_output or not output_has_valid_citation(revised_output, len(updated.get(args.docs_field, []) or [])))
    ):
        logger.warning("Falling back to deterministic atomic revision for item %s because v3 output was citation-invalid.", item_id)
        revised_output = AutoRevisionGenerator().revise(updated, verified_claims, rejected_claims, args)
    revised_output = clean_final_answer(revised_output, num_docs=len(updated.get(args.docs_field, []) or []))
    candidate_selection_diagnostics: dict[str, Any] = {}
    selected_final_audit: dict[str, Any] | None = None
    if candidate_output_selection_applies(updated, args):
        revised_output, candidate_selection_diagnostics, selected_final_audit = select_best_output_candidate(
            updated,
            item_id,
            draft_audit,
            verified_claims,
            expanded_claims,
            revised_output,
            decomposer,
            verifier,
            args,
        )
        revised_output = clean_final_answer(revised_output, num_docs=len(updated.get(args.docs_field, []) or []))
    updated.pop("cover_v3_original_output", None)

    overflow_rejected = []
    for claim in expansion_overflow:
        claim = dict(claim)
        claim["v3_status"] = "dropped_by_answer_budget"
        overflow_rejected.append(claim)

    final_doc_pool = list(updated.get(args.docs_field, []) or [])
    revision_metrics = compute_v3_metrics(
        draft_audit,
        verified_claims,
        list(rejected_claims) + overflow_rejected,
        recovered_claims,
        recovery_attempts,
        expanded_claims,
        expansion_attempts,
        expansion_overflow,
        revised_output,
        len(final_doc_pool),
        len(raw_docs),
    )
    revision_metrics.update(summarize_dynamic_retrieval_diagnostics(dynamic_retrieval_diagnostics))

    updated["cover_v3"] = {
        "original_output": draft_output,
        "raw_doc_count": len(raw_docs),
        "candidate_doc_count": len(candidate_docs),
        "doc_pool_size": len(final_doc_pool),
        "draft_audit": draft_audit,
        "first_pass_verified_claims": first_pass_claims,
        "verified_claims": verified_claims,
        "recovered_claims": recovered_claims,
        "expanded_claims": expanded_claims,
        "rejected_claims": list(rejected_claims) + overflow_rejected,
        "recovery_attempts": recovery_attempts,
        "expansion_attempts": expansion_attempts,
        "expansion_overflow": expansion_overflow,
        "dynamic_retrieval_diagnostics": dynamic_retrieval_diagnostics,
        "render_diagnostics": render_diagnostics,
        "candidate_selection_diagnostics": candidate_selection_diagnostics,
        "revision_metrics": revision_metrics,
    }
    updated[args.output_field] = revised_output

    if args.final_audit:
        updated["cover_v3"]["final_audit"] = selected_final_audit or audit_item(updated, item_id, decomposer, verifier, args)
    return updated


def write_checkpoint_payload(
    output_path: Path,
    payload: dict[str, Any],
    revised_items: Sequence[dict[str, Any]],
    remaining_items: Sequence[dict[str, Any]],
    processed_count: int,
    args: argparse.Namespace,
    checkpoint_items: Sequence[dict[str, Any]] | None = None,
) -> None:
    checkpoint_payload = dict(payload)
    checkpoint_data = list(checkpoint_items) if checkpoint_items is not None else list(revised_items) + list(remaining_items)
    checkpoint_payload["data"] = checkpoint_data
    checkpoint_payload["cover_v3_checkpoint"] = {
        "processed_examples": processed_count,
        "total_examples": len(checkpoint_data),
        "final": False,
    }
    checkpoint_payload["cover_v3_summary"] = summarize_v3(revised_items)
    final_summary = summarize_final_audit_v3(revised_items)
    if final_summary is not None:
        checkpoint_payload["cover_v3_final_audit_summary"] = final_summary

    checkpoint_path = Path(str(output_path) + ".checkpoint.json")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(safe_float(checkpoint_payload), ensure_ascii=False, indent=2 if args.pretty else None),
        encoding="utf-8",
    )
    logger.info("Wrote COVER-RAG v3 checkpoint after %d examples to %s", processed_count, checkpoint_path)


def revise_items_v3(
    process_items: Sequence[dict[str, Any]],
    all_items: Sequence[dict[str, Any]],
    payload: dict[str, Any],
    candidate_provider: CandidateDocProvider,
    worker_context: RevisionWorkerContext,
    cache: JsonCache,
    args: argparse.Namespace,
    tqdm: Any,
) -> list[dict[str, Any]]:
    workers = max(1, int(args.workers or 1))
    workers = min(workers, max(1, len(process_items)))

    def revise_one(item_id: int, item: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        decomposer, verifier, expansion_generator = worker_context.get()
        return item_id, revise_item_v3(
            item,
            item_id,
            candidate_provider,
            decomposer,
            verifier,
            expansion_generator,
            args,
        )

    if workers <= 1 or len(process_items) <= 1:
        revised_items = []
        for item_id, item in enumerate(tqdm(process_items, desc="COVER-RAG v3 iterative expansion")):
            _, revised = revise_one(item_id, item)
            revised_items.append(revised)
            processed_count = len(revised_items)
            if args.checkpoint_every and processed_count % max(1, args.checkpoint_every) == 0:
                cache.flush()
                remaining = process_items[processed_count:]
                if args.limit is not None and len(all_items) > args.limit:
                    remaining = list(remaining) + list(all_items[args.limit :])
                write_checkpoint_payload(args.output, payload, revised_items, remaining, processed_count, args)
        return revised_items

    logger.info("Processing %d examples with %d worker threads.", len(process_items), workers)
    revised_by_index: list[dict[str, Any] | None] = [None] * len(process_items)
    completed_count = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(revise_one, item_id, item)
            for item_id, item in enumerate(process_items)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="COVER-RAG v3 iterative expansion"):
            item_id, revised = future.result()
            revised_by_index[item_id] = revised
            completed_count += 1
            if args.checkpoint_every and completed_count % max(1, args.checkpoint_every) == 0:
                cache.flush()
                completed_items = [item for item in revised_by_index if item is not None]
                checkpoint_items = [
                    revised_by_index[index] if revised_by_index[index] is not None else process_items[index]
                    for index in range(len(process_items))
                ]
                if args.limit is not None and len(all_items) > args.limit:
                    checkpoint_items.extend(all_items[args.limit :])
                write_checkpoint_payload(
                    args.output,
                    payload,
                    completed_items,
                    [],
                    completed_count,
                    args,
                    checkpoint_items=checkpoint_items,
                )

    return [item for item in revised_by_index if item is not None]


def main() -> None:
    args = parse_args()
    if args.decomposer in {"llm", "claimify"} and not args.openai_api:
        raise ValueError(f"`--decomposer {args.decomposer}` requires `--openai-api`.")
    if args.verifier in {"llm", "hybrid"} and not args.openai_api:
        raise ValueError(f"`--verifier {args.verifier}` requires `--openai-api`.")
    if args.expansion_mode == "llm" and not args.openai_api:
        raise ValueError("`--expansion-mode llm` requires `--openai-api`.")
    args._coverage_reranker = None
    if args.coverage_reranker_model is not None:
        args._coverage_reranker = CoverageAwareReranker(
            args.coverage_reranker_model,
            device=args.coverage_reranker_device,
            batch_size=args.coverage_reranker_batch_size,
            max_length=args.coverage_reranker_max_length,
        )

    payload, items = load_payload(args.input)
    fill_dataset_name_from_payload(args, payload)
    process_items = items if args.limit is None else items[: args.limit]
    cache = JsonCache(args.cache_file, save_every=args.cache_save_every)
    candidate_provider = CandidateDocProvider(args.candidate_docs_file, args)
    worker_context = RevisionWorkerContext(args, cache)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **_: x

    revised_items = revise_items_v3(process_items, items, payload, candidate_provider, worker_context, cache, args, tqdm)
    cache.flush()

    if args.limit is not None and len(items) > args.limit:
        revised_items.extend(items[args.limit :])

    payload["data"] = revised_items
    payload["cover_v3_summary"] = summarize_v3(revised_items[: len(process_items)])
    final_summary = summarize_final_audit_v3(revised_items[: len(process_items)])
    if final_summary is not None:
        payload["cover_v3_final_audit_summary"] = final_summary
    payload["cover_v3_config"] = {
        "dataset_name": args.dataset_name,
        "answer_format": args.answer_format,
        "inferred_answer_format": infer_answer_format(args),
        "candidate_docs_file": str(args.candidate_docs_file) if args.candidate_docs_file else None,
        "max_doc_pool": args.max_doc_pool,
        "dynamic_retrieval_mode": args.dynamic_retrieval_mode,
        "dynamic_retrieval_top_k": args.dynamic_retrieval_top_k,
        "dynamic_retrieval_per_query_docs": args.dynamic_retrieval_per_query_docs,
        "dynamic_retrieval_max_queries": args.dynamic_retrieval_max_queries,
        "dynamic_retrieval_max_rejected_claims": args.dynamic_retrieval_max_rejected_claims,
        "dynamic_retrieval_max_doc_pool": args.dynamic_retrieval_max_doc_pool,
        "dynamic_retrieval_query_strategy": args.dynamic_retrieval_query_strategy,
        "dynamic_retrieval_min_score": args.dynamic_retrieval_min_score,
        "dynamic_retrieval_require_missing_signal": args.dynamic_retrieval_require_missing_signal,
        "dynamic_retrieval_use_supported_answer": args.dynamic_retrieval_use_supported_answer,
        "coverage_guided_retrieval": args.coverage_guided_retrieval,
        "coverage_guided_targets_per_round": args.coverage_guided_targets_per_round,
        "coverage_guided_min_targets": args.coverage_guided_min_targets,
        "coverage_guided_min_question_overlap": args.coverage_guided_min_question_overlap,
        "coverage_guided_query_supported_claims": args.coverage_guided_query_supported_claims,
        "coverage_guided_allow_evidence_targets": args.coverage_guided_allow_evidence_targets,
        "coverage_reranker_model": str(args.coverage_reranker_model) if args.coverage_reranker_model else None,
        "coverage_reranker_device": args.coverage_reranker_device,
        "coverage_reranker_batch_size": args.coverage_reranker_batch_size,
        "coverage_reranker_max_length": args.coverage_reranker_max_length,
        "coverage_reranker_doc_prefilter": args.coverage_reranker_doc_prefilter,
        "coverage_reranker_doc_weight": args.coverage_reranker_doc_weight,
        "coverage_reranker_candidate_weight": args.coverage_reranker_candidate_weight,
        "decomposer": args.decomposer,
        "decompose_model": args.decompose_model if args.decomposer in {"llm", "claimify"} else None,
        "verifier": args.verifier,
        "llm_verify_model": args.llm_verify_model if args.verifier in {"llm", "hybrid"} else None,
        "nli_model": args.nli_model if args.verifier in {"nli", "hybrid"} else None,
        "nli_premise_mode": args.nli_premise_mode,
        "nli_top_sentences": args.nli_top_sentences,
        "nli_sentence_window_size": args.nli_sentence_window_size,
        "evidence_scope": args.evidence_scope,
        "recover_labels": args.recover_labels,
        "recover_importance": args.recover_importance,
        "recovery_top_k": args.recovery_top_k,
        "recovery_group_size": args.recovery_group_size,
        "min_recovery_score": args.min_recovery_score,
        "max_recovered_citations": args.max_recovered_citations,
        "expansion_mode": args.expansion_mode,
        "expansion_model": args.expansion_model if args.expansion_mode == "llm" else None,
        "expansion_output_mode": args.expansion_output_mode,
        "expansion_evidence_sentences": args.expansion_evidence_sentences,
        "expansion_candidate_spans": args.expansion_candidate_spans,
        "max_selected_span_ids": max_selected_span_ids(args),
        "expansion_include_extractive": args.expansion_include_extractive,
        "verify_expansion_with_sentence": args.verify_expansion_with_sentence,
        "max_expansion_candidates": args.max_expansion_candidates,
        "max_expansion_verifications": args.max_expansion_verifications,
        "max_expanded_claims": args.max_expanded_claims,
        "min_expansion_score": args.min_expansion_score,
        "min_answer_unit_score": args.min_answer_unit_score,
        "strict_expansion_answer_unit_gate": args.strict_expansion_answer_unit_gate,
        "strict_expansion_min_target_relevance": args.strict_expansion_min_target_relevance,
        "answer_type_gate_mode": args.answer_type_gate_mode,
        "max_expansion_claim_words": args.max_expansion_claim_words,
        "skip_covered_answer_spans": args.skip_covered_answer_spans,
        "require_expansion_answer_span": args.require_expansion_answer_span,
        "expansion_query_source": args.expansion_query_source,
        "expansion_use_intermediate_answer": args.expansion_use_intermediate_answer,
        "answer_targeted_expansion": args.answer_targeted_expansion,
        "answer_targeted_extractive_top_k": args.answer_targeted_extractive_top_k,
        "answer_targeted_min_sentence_overlap": args.answer_targeted_min_sentence_overlap,
        "explanatory_expansion_require_phrase_span": args.explanatory_expansion_require_phrase_span,
        "explanatory_title_span_mode": args.explanatory_title_span_mode,
        "max_explanatory_expanded_claims": args.max_explanatory_expanded_claims,
        "max_explanatory_expansion_verifications": args.max_explanatory_expansion_verifications,
        "min_explanatory_answer_unit_score": args.min_explanatory_answer_unit_score,
        "explanatory_precision_gate": args.explanatory_precision_gate,
        "explanatory_precision_min_target_relevance": args.explanatory_precision_min_target_relevance,
        "explanatory_precision_min_question_relevance": args.explanatory_precision_min_question_relevance,
        "explanation_integrated_refinement": args.explanation_integrated_refinement,
        "max_explanation_integrated_claims": args.max_explanation_integrated_claims,
        "max_explanation_length_growth": args.max_explanation_length_growth,
        "candidate_output_selection": args.candidate_output_selection,
        "candidate_output_recall_claims": args.candidate_output_recall_claims,
        "candidate_output_recall_growth": args.candidate_output_recall_growth,
        "candidate_output_min_support": args.candidate_output_min_support,
        "candidate_output_support_tolerance": args.candidate_output_support_tolerance,
        "candidate_output_max_unsupported_rate": args.candidate_output_max_unsupported_rate,
        "candidate_output_min_answer_unit_gain": args.candidate_output_min_answer_unit_gain,
        "candidate_output_min_supported_claim_gain": args.candidate_output_min_supported_claim_gain,
        "recall_oriented_completion": args.recall_oriented_completion,
        "recall_completion_max_targets": args.recall_completion_max_targets,
        "recall_completion_rejected_targets": args.recall_completion_rejected_targets,
        "recall_completion_evidence_sentences": args.recall_completion_evidence_sentences,
        "recall_completion_supported_claims": args.recall_completion_supported_claims,
        "recall_completion_min_sentence_overlap": args.recall_completion_min_sentence_overlap,
        "recall_completion_min_target_overlap": args.recall_completion_min_target_overlap,
        "max_verified_claims": args.max_verified_claims,
        "max_final_claims": final_claim_limit(args),
        "preserve_first_pass_claims": args.preserve_first_pass_claims,
        "render_mode": args.render_mode,
        "preserve_answer_sentences": args.preserve_answer_sentences,
        "preserve_rejected_critical_answer_sentences": args.preserve_rejected_critical_answer_sentences,
        "max_preserved_source_words": args.max_preserved_source_words,
        "min_preserved_claim_fraction": args.min_preserved_claim_fraction,
        "min_rejected_preserve_question_overlap": args.min_rejected_preserve_question_overlap,
        "max_qampari_items": args.max_qampari_items,
        "qampari_answer_relevance_filter": args.qampari_answer_relevance_filter,
        "citation_policy": args.citation_policy,
        "sentence_citation_source": args.sentence_citation_source,
        "max_citations_per_claim": args.max_citations_per_claim,
        "max_citations_per_sentence": args.max_citations_per_sentence,
        "include_background_claims": args.include_background_claims,
        "final_audit": args.final_audit,
        "workers": max(1, int(args.workers or 1)),
        "cache_save_every": max(1, int(args.cache_save_every or 1)),
    }

    token_usage = collect_worker_token_usage(worker_context)
    if token_usage:
        payload["cover_v3_token_usage"] = token_usage

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(safe_float(payload), ensure_ascii=False, indent=2 if args.pretty else None),
        encoding="utf-8",
    )
    logger.info("Wrote COVER-RAG v3 result to %s", args.output)
    logger.info("COVER-RAG v3 summary: %s", json.dumps(payload["cover_v3_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

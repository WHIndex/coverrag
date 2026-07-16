#!/usr/bin/env python3
"""COVER-RAG for QAMPARI: answer-set selection instead of claim rewriting.

QAMPARI in ALCE is evaluated as a comma-separated answer set.  The main
correctness quantities are precision and Rec.-5, not paragraph-level QA F1.
This script therefore keeps the surface form of answer items whenever possible,
repairs or attaches citations, and only uses evidence verification to rank or
filter candidate answer items.

The output remains ALCE-compatible: it writes a normal result JSON whose
``output`` field is a cited comma-separated QAMPARI list.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from cover_posthoc_audit import (
    JsonCache,
    OpenAIChatClient,
    extract_json_object,
    format_doc,
    load_payload,
    normalize_space,
    parse_citations,
    remove_citations,
    safe_float,
    split_qampari_answer_items,
    stable_hash,
)
from cover_v1_revise import format_citations, normalize_qampari_item_key, unique_positive_ints


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("cover_qampari_select")


VERIFY_SYSTEM = """You are a strict evaluator for QAMPARI-style list QA.
Given a question, one candidate short answer, and evidence passages, decide
whether the candidate is a valid direct answer to the question and is supported
by the evidence. Return only valid JSON."""


VERIFY_USER = """Question:
{question}

Candidate short answer:
{answer}

Evidence passages:
{evidence}

Decision rules:
1. The candidate must answer the question's requested relation and answer type.
   For example, if the question asks for a person, a school or company is wrong.
   If it asks for a movie, a director or actor is wrong.
2. The evidence must explicitly support that candidate as an answer to the
   question. Mere co-occurrence in the passage is not enough.
3. The candidate should be a short answer item, not a whole explanatory clause.
4. Do not use outside knowledge.

Return JSON with exactly this schema:
{{
  "label": "supported|wrong_type|not_supported",
  "confidence": 0.0,
  "supporting_doc_ids": [1],
  "normalized_answer": "{answer}",
  "rationale": "brief reason"
}}"""


EXPAND_SYSTEM = """You extract candidate short answers for QAMPARI-style list QA.
Use only the provided evidence. Return only valid JSON."""


EXPAND_USER = """Question:
{question}

Existing answer items:
{existing}

Evidence passages:
{evidence}

Task:
Extract up to {max_candidates} short answer candidates that directly answer the
question. Candidate answers must be copied from the evidence passages and must
match the requested answer type/relation.

Return JSON with exactly this schema:
{{
  "candidates": [
    {{
      "answer": "short answer string copied from evidence",
      "doc_id": 1,
      "evidence_sentence": "supporting sentence copied from evidence"
    }}
  ]
}}"""


JOINT_SELECT_SYSTEM = """You are a QAMPARI answer-set selector.
QAMPARI is evaluated as a comma-separated list of short answers. Your job is to
choose up to K answer items that directly answer the question and are supported
by the provided evidence. Return only valid JSON."""


JOINT_SELECT_USER = """Question:
{question}

Original answer candidates from the draft:
{original_candidates}

Evidence passages:
{evidence}

Candidate answer spans copied from evidence:
{candidate_spans}

Task:
Select up to {max_items} short answer items.

Rules:
1. Prefer original candidates when they are correct; preserve their exact string.
2. You may add a missing answer only if it is copied exactly from the evidence.
3. First infer the expected answer type from the question, such as person, movie, book, school, organization, location, year, competition, song, software, language, or other.
4. Each selected answer must match both the question's requested relation and expected answer type.
   Example: if the question asks "Who directed a movie...", the answer must be a person/director, not a movie title.
   Example: if the question asks "Which movie...", the answer must be a movie/work title, not a person.
4. Each selected answer must have at least one supporting doc id from the evidence above.
5. Do not select generic category words, copied question entities, or whole clauses.
6. Optimize for QAMPARI precision and Rec.-5: a compact list of high-confidence answers is better than a long noisy list.
7. Avoid deleting high-confidence original answers just to add lower-confidence expanded ones.

Return JSON with exactly this schema:
{{
  "selected": [
    {{
      "answer": "short answer string",
      "doc_ids": [1],
      "source": "original|expanded",
      "expected_answer_type": "person|movie|book|school|organization|location|year|competition|song|software|language|other",
      "candidate_type": "person|movie|book|school|organization|location|year|competition|song|software|language|other",
      "confidence": 0.0,
      "rationale": "brief reason"
    }}
  ]
}}"""


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


GENERIC_KEYS = {
    "",
    "answer",
    "evidence",
    "question",
    "provided evidence",
    "provided documents",
    "film",
    "films",
    "movie",
    "movies",
    "person",
    "people",
    "school",
    "university",
    "college",
    "song",
    "work",
    "works",
    "type",
    "language",
    "country",
    "city",
    "company",
    "club",
    "team",
    "group",
}


BAD_SUBSTRINGS = (
    "provided evidence",
    "provided documents",
    "insufficient",
    "cannot answer",
    "cannot provide",
    "do not contain",
    "no relevant",
    "therefore",
)


TYPE_ALIASES = {
    "person": {"person", "people", "human", "director", "writer", "composer", "player", "golfer", "politician"},
    "movie": {"movie", "film", "motion picture", "work", "title"},
    "book": {"book", "novel", "manga", "work", "title"},
    "song": {"song", "music", "musical work", "work", "title"},
    "school": {"school", "university", "college", "institution", "organization"},
    "organization": {"organization", "company", "team", "club", "institution", "agency"},
    "location": {"location", "place", "city", "country", "region", "station"},
    "year": {"year", "date", "season"},
    "competition": {"competition", "tournament", "league", "cup"},
    "software": {"software", "program", "application"},
    "language": {"language"},
    "country": {"country", "nation", "location", "place"},
    "publication": {"publication", "magazine", "newspaper", "periodical", "book", "work", "title"},
}


@dataclass
class Candidate:
    answer: str
    key: str
    source: str
    source_rank: int
    citation_ids: list[int] = field(default_factory=list)
    support_label: str = "unverified"
    support_confidence: float = 0.0
    support_doc_ids: list[int] = field(default_factory=list)
    rationale: str = ""
    score: float = 0.0
    evidence_sentence: str = ""
    expected_answer_type: str = ""
    candidate_type: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "key": self.key,
            "source": self.source,
            "source_rank": self.source_rank,
            "citation_ids": self.citation_ids,
            "support_label": self.support_label,
            "support_confidence": self.support_confidence,
            "support_doc_ids": self.support_doc_ids,
            "rationale": self.rationale,
            "score": self.score,
            "evidence_sentence": self.evidence_sentence,
            "expected_answer_type": self.expected_answer_type,
            "candidate_type": self.candidate_type,
        }


class CandidateDocProvider:
    def __init__(self, path: Path | None, args: argparse.Namespace) -> None:
        self.args = args
        self.by_sample_id: dict[str, list[dict[str, Any]]] = {}
        self.by_question: dict[str, list[dict[str, Any]]] = {}
        self.by_index: dict[int, list[dict[str, Any]]] = {}
        if path is not None:
            self._load(path)

    def _load(self, path: Path) -> None:
        if not path.exists():
            logger.warning("Candidate docs file does not exist: %s", path)
            return
        _payload, records = load_payload(path)
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
            "Loaded candidate docs: %d sample ids, %d questions, %d indexed records.",
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


class QampariItemVerifier:
    def __init__(self, args: argparse.Namespace, cache: JsonCache) -> None:
        self.args = args
        self.cache = cache
        self.client: OpenAIChatClient | None = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        if args.item_verifier == "llm":
            self.client = OpenAIChatClient(
                model=args.item_verify_model,
                temperature=args.llm_temperature,
                top_p=args.llm_top_p,
                max_retries=args.llm_max_retries,
            )

    def verify(
        self,
        question: str,
        answer: str,
        docs: Sequence[tuple[int, dict[str, Any]]],
    ) -> dict[str, Any]:
        if self.args.item_verifier == "none":
            return {
                "label": "supported",
                "confidence": 0.5,
                "supporting_doc_ids": [doc_id for doc_id, _ in docs],
                "normalized_answer": answer,
                "rationale": "Verification disabled.",
            }
        if self.args.item_verifier == "lexical":
            return self._lexical_verify(question, answer, docs)
        return self._llm_verify(question, answer, docs)

    def _lexical_verify(
        self,
        question: str,
        answer: str,
        docs: Sequence[tuple[int, dict[str, Any]]],
    ) -> dict[str, Any]:
        del question
        answer_tokens = content_tokens(answer)
        if not answer_tokens:
            return {"label": "not_supported", "confidence": 0.0, "supporting_doc_ids": [], "normalized_answer": answer}
        useful = []
        best = 0.0
        for doc_id, doc in docs:
            doc_tokens = content_tokens(format_doc(doc, self.args))
            score = len(answer_tokens & doc_tokens) / len(answer_tokens)
            best = max(best, score)
            if score >= self.args.lexical_item_threshold:
                useful.append(doc_id)
        return {
            "label": "supported" if useful else "not_supported",
            "confidence": best,
            "supporting_doc_ids": useful,
            "normalized_answer": answer,
            "rationale": "Lexical item verification.",
        }

    def _llm_verify(
        self,
        question: str,
        answer: str,
        docs: Sequence[tuple[int, dict[str, Any]]],
    ) -> dict[str, Any]:
        assert self.client is not None
        evidence = "\n\n".join(f"[{doc_id}] {format_doc(doc, self.args)}" for doc_id, doc in docs)
        evidence = evidence[: self.args.max_evidence_chars]
        key = "qampari-item-verify:" + stable_hash(question, answer, evidence)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        raw, usage = self.client.chat(
            VERIFY_SYSTEM,
            VERIFY_USER.format(question=question, answer=answer, evidence=evidence),
            max_tokens=350,
        )
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        try:
            payload = extract_json_object(raw)
            label = str(payload.get("label", "not_supported"))
            if label not in {"supported", "wrong_type", "not_supported"}:
                label = "not_supported"
            confidence = float(payload.get("confidence", 0.0))
            doc_ids = unique_positive_ints(payload.get("supporting_doc_ids") or [])
            result = {
                "label": label,
                "confidence": max(0.0, min(confidence, 1.0)),
                "supporting_doc_ids": doc_ids,
                "normalized_answer": normalize_answer_item(str(payload.get("normalized_answer") or answer)),
                "rationale": normalize_space(str(payload.get("rationale", ""))),
            }
        except Exception as exc:
            logger.warning("QAMPARI item verifier JSON parse failed; marking not_supported. Error: %s", exc)
            result = {
                "label": "not_supported",
                "confidence": 0.0,
                "supporting_doc_ids": [],
                "normalized_answer": answer,
                "rationale": "Verifier output could not be parsed.",
            }
        self.cache.set(key, result)
        return result


class QampariExpansionGenerator:
    def __init__(self, args: argparse.Namespace, cache: JsonCache) -> None:
        self.args = args
        self.cache = cache
        self.client: OpenAIChatClient | None = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        if args.expand_from_docs == "llm":
            self.client = OpenAIChatClient(
                model=args.expansion_model,
                temperature=args.llm_temperature,
                top_p=args.llm_top_p,
                max_retries=args.llm_max_retries,
            )

    def generate(
        self,
        question: str,
        docs: Sequence[dict[str, Any]],
        existing: Sequence[Candidate],
        args: argparse.Namespace,
    ) -> list[Candidate]:
        if args.expand_from_docs == "off":
            return []
        ranked = rank_docs(question, "", docs, args)[: args.expansion_docs]
        if args.expand_from_docs == "extractive":
            return extractive_candidates(question, ranked, existing, args)
        return self._llm_generate(question, ranked, existing, args)

    def _llm_generate(
        self,
        question: str,
        ranked_docs: Sequence[tuple[int, dict[str, Any], float]],
        existing: Sequence[Candidate],
        args: argparse.Namespace,
    ) -> list[Candidate]:
        assert self.client is not None
        evidence = "\n\n".join(
            f"[{doc_id}] {format_doc(doc, args)[: args.expansion_doc_chars]}"
            for doc_id, doc, _ in ranked_docs
        )
        existing_text = ", ".join(candidate.answer for candidate in existing[: args.max_output_items]) or "None"
        key = "qampari-expand:" + stable_hash(question, existing_text, evidence, str(args.max_expansion_candidates))
        cached = self.cache.get(key)
        if cached is None:
            raw, usage = self.client.chat(
                EXPAND_SYSTEM,
                EXPAND_USER.format(
                    question=question,
                    existing=existing_text,
                    evidence=evidence,
                    max_candidates=args.max_expansion_candidates,
                ),
                max_tokens=700,
            )
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            try:
                cached = extract_json_object(raw)
            except Exception as exc:
                logger.warning("QAMPARI expansion JSON parse failed; using no LLM candidates. Error: %s", exc)
                cached = {"candidates": []}
            self.cache.set(key, cached)
        rows = cached.get("candidates", []) if isinstance(cached, dict) else []
        candidates: list[Candidate] = []
        seen = {candidate.key for candidate in existing}
        for rank, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            answer = normalize_answer_item(str(row.get("answer", "")))
            key = normalize_qampari_item_key(answer)
            if not answer or key in seen or is_bad_answer(answer, question)[0]:
                continue
            try:
                doc_id = int(row.get("doc_id"))
            except Exception:
                doc_id = 0
            if doc_id <= 0 or doc_id > len(ranked_docs) + args.max_doc_pool:
                doc_id = 0
            candidates.append(
                Candidate(
                    answer=answer,
                    key=key,
                    source="expanded_llm",
                    source_rank=rank,
                    citation_ids=[doc_id] if doc_id > 0 else [],
                    evidence_sentence=normalize_space(str(row.get("evidence_sentence", ""))),
                )
            )
            seen.add(key)
            if len(candidates) >= args.max_expansion_candidates:
                break
        if args.expansion_include_extractive and len(candidates) < args.max_expansion_candidates:
            candidates.extend(extractive_candidates(question, ranked_docs, existing + candidates, args))
        return candidates[: args.max_expansion_candidates]


class QampariJointSelector:
    """One LLM call per question for QAMPARI answer selection.

    This is much faster than itemwise verification because it asks the model to
    jointly choose the final answer set and citation ids. It also matches the
    ALCE/QAMPARI objective better: select a short answer set, not a paragraph of
    verified claims.
    """

    def __init__(self, args: argparse.Namespace, cache: JsonCache) -> None:
        self.args = args
        self.cache = cache
        self.client = OpenAIChatClient(
            model=args.joint_model,
            temperature=args.llm_temperature,
            top_p=args.llm_top_p,
            max_retries=args.llm_max_retries,
        )
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @staticmethod
    def _original_subspan_matches(raw_key: str, originals: Sequence[Candidate]) -> list[Candidate]:
        if not raw_key or raw_key in GENERIC_KEYS:
            return []
        raw_tokens = set(raw_key.split())
        if not raw_tokens:
            return []
        matches: dict[str, Candidate] = {}
        for candidate in originals:
            if not candidate.key or candidate.key == raw_key:
                continue
            candidate_tokens = set(candidate.key.split())
            if raw_key not in candidate.key and not raw_tokens.issubset(candidate_tokens):
                continue
            if len(raw_tokens) == 1 and not re.search(r"\d", candidate.key):
                continue
            matches[candidate.key] = candidate
        return list(matches.values())

    @classmethod
    def _unique_original_subspan_match(cls, raw_key: str, originals: Sequence[Candidate]) -> Candidate | None:
        matches = cls._original_subspan_matches(raw_key, originals)
        return matches[0] if len(matches) == 1 else None

    def select(
        self,
        question: str,
        original_candidates: Sequence[Candidate],
        docs: Sequence[dict[str, Any]],
        args: argparse.Namespace,
    ) -> tuple[list[Candidate], list[Candidate]]:
        usable_original = [
            candidate
            for candidate in original_candidates
            if candidate.source != "original_dropped" and candidate.answer and candidate.key
        ]
        evidence_docs = select_joint_docs(question, usable_original, docs, args)
        candidate_block = format_original_candidate_block(usable_original)
        span_block = format_candidate_span_block(question, evidence_docs, usable_original, args)
        evidence_block = "\n\n".join(
            f"[{doc_id}] {format_doc(doc, args)[: args.joint_doc_chars]}"
            for doc_id, doc in evidence_docs
        )
        key = "qampari-joint-select:" + stable_hash(
            question,
            candidate_block,
            evidence_block,
            span_block,
            str(args.max_output_items),
            str(args.joint_require_type_match),
        )
        cached = self.cache.get(key)
        if cached is None:
            raw, usage = self.client.chat(
                JOINT_SELECT_SYSTEM,
                JOINT_SELECT_USER.format(
                    question=question,
                    original_candidates=candidate_block or "None",
                    evidence=evidence_block or "None",
                    candidate_spans=span_block or "None",
                    max_items=args.max_output_items,
                ),
                max_tokens=args.joint_max_tokens,
            )
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            try:
                cached = extract_json_object(raw)
            except Exception as exc:
                logger.warning("QAMPARI joint selection JSON parse failed. Error: %s", exc)
                cached = {"selected": []}
            self.cache.set(key, cached)

        original_by_key = {candidate.key: candidate for candidate in usable_original}
        selected: list[Candidate] = []
        seen: set[str] = set()
        evidence_doc_ids = {doc_id for doc_id, _ in evidence_docs}
        for candidate in usable_original[: max(0, int(getattr(args, "preserve_original_top_k", 0)))]:
            if len(selected) >= args.max_output_items:
                break
            if candidate.key in seen:
                continue
            if args.backfill_requires_type_safe:
                type_ok, _type_reason = original_backfill_type_safe(question, candidate.answer)
                if not type_ok:
                    continue
            doc_ids = [
                doc_id
                for doc_id in candidate.citation_ids
                if 1 <= doc_id <= len(docs) and (not evidence_doc_ids or doc_id in evidence_doc_ids)
            ]
            if not doc_ids:
                ranked = rank_docs(question, candidate.answer, docs, args)
                doc_ids = [doc_id for doc_id, _doc, _score in ranked[:1]]
            if not doc_ids:
                continue
            preserved = Candidate(**candidate.to_json())
            preserved.source = "joint_preserve_original"
            preserved.support_label = args.preserved_original_label
            preserved.citation_ids = doc_ids
            preserved.support_doc_ids = doc_ids if preserved.support_label == "supported" else []
            preserved.score = 3.0 - 0.01 * preserved.source_rank
            selected.append(preserved)
            seen.add(preserved.key)
        rows = cached.get("selected", []) if isinstance(cached, dict) else []
        for rank, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            raw_answer = normalize_answer_item(str(row.get("answer", "")))
            key = normalize_qampari_item_key(raw_answer)
            if not raw_answer or not key or key in seen:
                continue
            bad, _reason = is_bad_answer(raw_answer, question)
            if bad:
                continue
            original = original_by_key.get(key)
            if original is None and args.restore_original_subspans:
                subspan_matches = self._original_subspan_matches(key, usable_original)
                if len(subspan_matches) > 1:
                    # Ambiguous shortened surfaces such as "PGA Championship"
                    # collapse several exact answers. Reject them so original
                    # backfill can preserve "1963 PGA Championship", etc.
                    continue
                original = subspan_matches[0] if subspan_matches else None
                if original is not None:
                    key = original.key
            if key in seen:
                continue
            answer = original.answer if original is not None else raw_answer
            bad, _reason = is_bad_answer(answer, question)
            if bad:
                continue
            doc_ids = [
                doc_id
                for doc_id in coerce_doc_ids(row.get("doc_ids") or row.get("doc_id"))
                if doc_id in evidence_doc_ids
            ]
            if not doc_ids and original is not None:
                doc_ids = [doc_id for doc_id in original.citation_ids if 1 <= doc_id <= len(docs)]
            if not doc_ids:
                continue
            source = str(row.get("source") or ("original" if original is not None else "expanded"))
            if source not in {"original", "expanded"}:
                source = "original" if original is not None else "expanded"
            if original is not None:
                source = "original"
            try:
                confidence = float(row.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            candidate = Candidate(
                answer=answer,
                key=key,
                source=f"joint_{source}",
                source_rank=original.source_rank if original is not None else 1000 + rank,
                citation_ids=doc_ids,
                support_label="supported",
                support_confidence=max(0.0, min(confidence, 1.0)),
                support_doc_ids=doc_ids,
                rationale=normalize_space(str(row.get("rationale", ""))),
                score=2.0 + confidence - 0.01 * rank,
                expected_answer_type=normalize_space(str(row.get("expected_answer_type", ""))),
                candidate_type=normalize_space(str(row.get("candidate_type", ""))),
            )
            if args.joint_require_type_match and candidate.expected_answer_type and candidate.candidate_type:
                if not answer_types_compatible(candidate.expected_answer_type, candidate.candidate_type):
                    continue
            if args.strict_question_type_guard:
                type_ok, type_reason = question_type_guard_allows(question, candidate)
                if not type_ok:
                    candidate.support_label = "type_guard_rejected"
                    candidate.rationale = type_reason
                    continue
            selected.append(candidate)
            seen.add(key)
            if len(selected) >= args.max_output_items:
                break

        # Backfill high-ranked original items if the joint selector is too
        # conservative. This protects Rec.-5 and makes the run less brittle.
        for candidate in usable_original:
            if len(selected) >= min(args.backfill_original_until, args.max_output_items):
                break
            if candidate.key in seen:
                continue
            if args.backfill_requires_type_safe:
                type_ok, _type_reason = original_backfill_type_safe(question, candidate.answer)
                if not type_ok:
                    continue
            doc_ids = [doc_id for doc_id in candidate.citation_ids if 1 <= doc_id <= len(docs)]
            if not doc_ids:
                ranked = rank_docs(question, candidate.answer, docs, args)
                doc_ids = [doc_id for doc_id, _doc, _score in ranked[:1]]
            if not doc_ids:
                continue
            backfilled = Candidate(**candidate.to_json())
            backfilled.source = "joint_backfill_original"
            backfilled.support_label = "backfilled"
            backfilled.citation_ids = doc_ids
            backfilled.support_doc_ids = doc_ids
            backfilled.score = 0.5 - 0.01 * backfilled.source_rank
            selected.append(backfilled)
            seen.add(backfilled.key)

        dropped = [candidate for candidate in usable_original if candidate.key not in seen]
        return selected, dropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COVER-RAG QAMPARI answer-set selector.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--candidate-docs-file", type=Path, default=None)
    parser.add_argument("--max-doc-pool", type=int, default=100)

    parser.add_argument("--output-field", default="output")
    parser.add_argument("--docs-field", default="docs")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--title-field", default="title")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--sent-field", default="sent")

    parser.add_argument("--openai-api", action="store_true")
    parser.add_argument(
        "--selection-mode",
        choices=["joint", "itemwise"],
        default="joint",
        help=(
            "joint uses one LLM call per question to select the final QAMPARI answer set; "
            "itemwise verifies each candidate separately and is much slower."
        ),
    )
    parser.add_argument("--joint-model", default="gpt-4o-mini")
    parser.add_argument("--joint-docs", type=int, default=12)
    parser.add_argument("--joint-doc-chars", type=int, default=850)
    parser.add_argument("--joint-candidate-spans", type=int, default=80)
    parser.add_argument("--joint-require-type-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--strict-question-type-guard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply deterministic guards for high-confidence QAMPARI answer-type patterns, "
            "for example rejecting movie titles for questions that ask who directed a work."
        ),
    )
    parser.add_argument("--joint-max-tokens", type=int, default=700)
    parser.add_argument("--item-verifier", choices=["llm", "lexical", "none"], default="llm")
    parser.add_argument("--item-verify-model", default="gpt-4o-mini")
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-top-p", type=float, default=1.0)
    parser.add_argument("--llm-max-retries", type=int, default=5)
    parser.add_argument("--cache-file", type=Path, default=Path("cache/cover_qampari_select_cache.json"))
    parser.add_argument("--cache-save-every", type=int, default=25)

    parser.add_argument("--max-evidence-chars", type=int, default=5000)
    parser.add_argument("--lexical-item-threshold", type=float, default=0.85)
    parser.add_argument("--repair-top-k", type=int, default=5)
    parser.add_argument("--repair-group-size", type=int, default=3)

    parser.add_argument("--expand-from-docs", choices=["off", "extractive", "llm"], default="llm")
    parser.add_argument("--expansion-model", default="gpt-4o-mini")
    parser.add_argument("--expansion-docs", type=int, default=8)
    parser.add_argument("--expansion-doc-chars", type=int, default=900)
    parser.add_argument("--max-expansion-candidates", type=int, default=8)
    parser.add_argument("--expansion-include-extractive", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--max-output-items", type=int, default=5)
    parser.add_argument("--backfill-original-until", type=int, default=5)
    parser.add_argument(
        "--preserve-original-top-k",
        type=int,
        default=0,
        help=(
            "Before joint LLM additions, reserve slots for the first K original "
            "answer items that pass deterministic guards. This protects QAMPARI "
            "recall from selector deletions."
        ),
    )
    parser.add_argument(
        "--preserved-original-label",
        choices=["supported", "backfilled"],
        default="backfilled",
        help="Support label assigned to items kept by --preserve-original-top-k.",
    )
    parser.add_argument(
        "--backfill-requires-type-safe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When backfilling unverified original answers, skip items that violate deterministic question type guards.",
    )
    parser.add_argument(
        "--empty-output-policy",
        choices=["empty", "original"],
        default="empty",
        help=(
            "What to emit when no supported QAMPARI answer item is selected. "
            "empty avoids carrying citation-less refusal text into list evaluation; original preserves old behavior."
        ),
    )
    parser.add_argument("--min-supported-confidence", type=float, default=0.55)
    parser.add_argument("--keep-unsupported-original", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--restore-original-subspans",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When joint selection emits a shortened form that uniquely matches an original answer item, "
            "restore the original exact string to protect QAMPARI recall."
        ),
    )
    parser.add_argument("--drop-obvious-bad", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-citations-per-item", type=int, default=1)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def make_json_cache(path: Path | None, save_every: int) -> JsonCache:
    """Construct JsonCache across slightly different local implementations."""

    try:
        return JsonCache(path, save_every=save_every)
    except TypeError:
        return JsonCache(path)


def flush_json_cache(cache: JsonCache) -> None:
    """Flush JsonCache without assuming whether the method is named flush/save."""

    flush = getattr(cache, "flush", None)
    if callable(flush):
        flush()
        return
    save = getattr(cache, "save", None)
    if callable(save):
        save()


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", remove_citations(text).lower())
        if token not in STOPWORDS and len(token) > 1
    }


def overlap_score(query: str, text: str) -> float:
    query_tokens = content_tokens(query)
    if not query_tokens:
        return 0.0
    text_tokens = content_tokens(text)
    if not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def normalize_answer_item(text: str) -> str:
    text = remove_citations(text)
    # Strip list markers such as "- Foo", "1. Foo", or "1) Foo" without
    # deleting numeric answer prefixes such as "70 Volt Parade" or
    # "1975 PGA Championship".
    text = re.sub(r"^\s*(?:[-*]\s+|\(?\d{1,3}[\).]\s+)", "", text)
    text = normalize_space(text).strip().strip("\"'")
    text = text.rstrip(".;:,")
    return text


def is_bad_answer(answer: str, question: str = "") -> tuple[bool, str]:
    raw = normalize_answer_item(answer)
    key = normalize_qampari_item_key(raw)
    if not key:
        return True, "empty"
    if key in GENERIC_KEYS:
        return True, "generic"
    lowered = raw.lower()
    if any(bad in lowered for bad in BAD_SUBSTRINGS):
        return True, "abstention_or_non_answer"
    words = key.split()
    if len(words) > 9:
        return True, "too_long"
    if len(raw) < 2:
        return True, "too_short"
    if words and words[-1] in {"and", "or", "of", "the", "a", "an", "in", "for", "to", "with"}:
        return True, "trailing_connector"
    question_key = normalize_qampari_item_key(question)
    if question_key and key in question_key:
        return True, "copied_from_question"
    q_tokens = content_tokens(question)
    a_tokens = content_tokens(raw)
    if a_tokens and q_tokens and a_tokens <= q_tokens:
        return True, "answer_tokens_all_in_question"
    return False, ""


def merge_doc_pool(raw_docs: Sequence[dict[str, Any]], candidate_docs: Sequence[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    seen = set()
    for doc in list(raw_docs) + list(candidate_docs):
        title = normalize_space(str(doc.get(args.title_field, "") or "")).lower()
        body = normalize_space(str(doc.get(args.sent_field) or doc.get(args.text_field, "") or "")).lower()
        key = f"{title}\n{body[:1000]}"
        if not key.strip() or key in seen:
            continue
        seen.add(key)
        pool.append(doc)
        if len(pool) >= args.max_doc_pool:
            break
    return pool


def docs_by_ids(docs: Sequence[dict[str, Any]], ids: Sequence[int]) -> list[tuple[int, dict[str, Any]]]:
    selected = []
    for doc_id in unique_positive_ints(ids):
        if 1 <= doc_id <= len(docs):
            selected.append((doc_id, docs[doc_id - 1]))
    return selected


def coerce_doc_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value] if value > 0 else []
    if isinstance(value, str):
        return unique_positive_ints(re.findall(r"\d+", value))
    if isinstance(value, list):
        return unique_positive_ints(value)
    return []


def select_joint_docs(
    question: str,
    original_candidates: Sequence[Candidate],
    docs: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> list[tuple[int, dict[str, Any]]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    seen: set[int] = set()

    def add(doc_id: int) -> None:
        if len(selected) >= args.joint_docs:
            return
        if doc_id in seen or doc_id <= 0 or doc_id > len(docs):
            return
        seen.add(doc_id)
        selected.append((doc_id, docs[doc_id - 1]))

    for candidate in original_candidates:
        for doc_id in candidate.citation_ids:
            add(doc_id)
    query = question + " " + " ".join(candidate.answer for candidate in original_candidates[:8])
    for doc_id, _doc, _score in rank_docs(query, "", docs, args):
        add(doc_id)
        if len(selected) >= args.joint_docs:
            break
    return selected


def format_original_candidate_block(candidates: Sequence[Candidate]) -> str:
    lines = []
    for idx, candidate in enumerate(candidates, start=1):
        citations = format_citations(candidate.citation_ids, max_citations=3)
        lines.append(f"{idx}. {candidate.answer} {citations}".strip())
    return "\n".join(lines)


def canonical_type_set(label: str) -> set[str]:
    """Map noisy LLM type labels such as "person|organization" to canonical types."""

    raw = str(label or "").lower().strip()
    if not raw:
        return set()
    output: set[str] = set()
    for piece in re.split(r"[|,;/]+", raw):
        key = normalize_qampari_item_key(piece)
        if not key or key == "other":
            continue
        matched = False
        for canonical, names in TYPE_ALIASES.items():
            if key == canonical or key in names:
                output.add(canonical)
                matched = True
        if not matched:
            output.add(key)
    return output


def answer_types_compatible(expected: str, candidate: str) -> bool:
    expected_raw = normalize_qampari_item_key(expected)
    candidate_raw = normalize_qampari_item_key(candidate)
    if not expected_raw or not candidate_raw:
        return True
    if expected_raw == "other" or candidate_raw == "other":
        return True
    expected_types = canonical_type_set(expected)
    candidate_types = canonical_type_set(candidate)
    if not expected_types or not candidate_types:
        return True
    return bool(expected_types & candidate_types)


def infer_question_type_guard(question: str) -> tuple[set[str], str]:
    """Infer high-confidence answer-type constraints from QAMPARI question wording.

    This guard is deliberately conservative.  It only fires on patterns where a
    wrong semantic type is a common source of QAMPARI regressions, for example
    selecting a movie title for "Who directed ...".
    """

    q = normalize_space(question).lower()
    if not q:
        return set(), ""
    person_patterns = [
        r"\bwho\s+(?:directed|was the director|were the directors)\b",
        r"\bwho\s+(?:wrote|screenwrote|composed|performed|played|won)\b",
        r"\bwho\s+spoke\b",
        r"\bwhat\s+person\b",
        r"\bname of (?:a|an) (?:person|individual)\b",
        r"\bindividual who\b",
        r"\bperson who\b",
    ]
    if any(re.search(pattern, q) for pattern in person_patterns):
        return {"person"}, "question asks for a person"
    if re.search(r"\bwho\s+(?:created|founded|invented|provides|employs)\b", q):
        return {"person", "organization"}, "question asks for an agent"
    if re.search(r"\bwho do .* work for\b", q) or re.search(r"\bwho does .* work for\b", q):
        return {"person", "organization", "school"}, "question asks for an employer or affiliation"
    if re.search(r"\b(?:which|what|for which)\s+(?:movie|film|motion picture)\b", q):
        return {"movie"}, "question asks for a movie or film"
    if re.search(r"\b(?:which|what)\s+(?:book|novel|manga)\b", q):
        return {"book"}, "question asks for a book or manga"
    if re.search(r"\b(?:which|what)\s+(?:song|music|musical work)\b", q):
        return {"song"}, "question asks for a song or musical work"
    if re.search(r"\b(?:which|what)\s+(?:software|program|application)\b", q):
        return {"software"}, "question asks for software"
    if re.search(r"\b(?:which|what)\s+(?:competition|tournament|league|cup)\b", q):
        return {"competition"}, "question asks for a competition"
    if re.search(r"\b(?:which|what)\s+(?:language|native spoken language)\b", q):
        return {"language"}, "question asks for a language"
    if re.search(r"\b(?:what|which)\s+(?:magazine|newspaper|periodical|publication)\b", q):
        return {"publication", "book"}, "question asks for a publication"
    if q.startswith("where ") or re.search(r"\bwhere did\b", q):
        return {"location", "school"}, "question asks for a place"
    if q.startswith("when ") or re.search(r"\bwhat year\b", q):
        return {"year"}, "question asks for a date or year"
    return set(), ""


def question_type_guard_allows(question: str, candidate: Candidate) -> tuple[bool, str]:
    allowed, reason = infer_question_type_guard(question)
    if not allowed:
        return True, ""
    candidate_types = canonical_type_set(candidate.candidate_type)
    if not candidate_types:
        # The LLM did not provide a meaningful type. Keep the item here; the
        # optional backfill guard handles untyped original items separately.
        return True, ""
    if candidate_types & allowed:
        return True, ""
    return (
        False,
        f"Rejected by deterministic type guard: {reason}; candidate_type={candidate.candidate_type!r}.",
    )


def original_backfill_type_safe(question: str, answer: str) -> tuple[bool, str]:
    allowed, reason = infer_question_type_guard(question)
    if not allowed:
        return True, ""
    answer_key = normalize_qampari_item_key(answer)
    question_key = normalize_qampari_item_key(question)
    if not answer_key or answer_key in GENERIC_KEYS:
        return False, f"Backfill rejected by type guard: {reason}; generic answer."
    if answer_key and answer_key in question_key:
        return False, f"Backfill rejected by type guard: {reason}; answer repeats question entity."
    lower = answer.lower()
    if allowed == {"person"} and re.search(
        r"\b(movie|film|season|episode|program|software|school|university|college|language|cup|league|tournament)\b",
        lower,
    ):
        return False, f"Backfill rejected by type guard: {reason}; answer surface looks non-person."
    return True, ""


def format_candidate_span_block(
    question: str,
    evidence_docs: Sequence[tuple[int, dict[str, Any]]],
    original_candidates: Sequence[Candidate],
    args: argparse.Namespace,
) -> str:
    seen = {candidate.key for candidate in original_candidates}
    spans: list[tuple[str, int]] = []
    pattern = re.compile(
        r"\b(?:[A-Z][A-Za-z0-9&'.-]+|\d{4}|\d+(?:\.\d+)?)(?:\s+(?:of|the|and|&|de|da|van|von|[A-Z][A-Za-z0-9&'.-]+|\d{4}|\d+(?:\.\d+)?)){0,6}\b"
    )
    for doc_id, doc in evidence_docs:
        text = format_doc(doc, args)
        for match in pattern.finditer(text):
            span = normalize_answer_item(match.group(0))
            key = normalize_qampari_item_key(span)
            if not span or key in seen:
                continue
            bad, _reason = is_bad_answer(span, question)
            if bad:
                continue
            seen.add(key)
            spans.append((span, doc_id))
            if len(spans) >= args.joint_candidate_spans:
                break
        if len(spans) >= args.joint_candidate_spans:
            break
    return "\n".join(f"- {span} [{doc_id}]" for span, doc_id in spans)


def rank_docs(question: str, answer: str, docs: Sequence[dict[str, Any]], args: argparse.Namespace) -> list[tuple[int, dict[str, Any], float]]:
    query = f"{question} {answer}".strip()
    rows = []
    for doc_id, doc in enumerate(docs, start=1):
        title = str(doc.get(args.title_field, "") or "")
        body = format_doc(doc, args)
        score = 0.75 * overlap_score(query, body) + 0.25 * overlap_score(answer, title)
        if answer and normalize_qampari_item_key(answer) in normalize_qampari_item_key(body):
            score += 0.45
        rows.append((doc_id, doc, score))
    rows.sort(key=lambda row: (-row[2], row[0]))
    return rows


def parse_original_candidates(output: str, question: str, args: argparse.Namespace) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen = set()
    for rank, unit in enumerate(split_qampari_answer_items(output), start=1):
        answer = normalize_answer_item(unit)
        key = normalize_qampari_item_key(answer)
        if not key or key in seen:
            continue
        bad, reason = is_bad_answer(answer, question)
        if args.drop_obvious_bad and bad:
            candidates.append(
                Candidate(
                    answer=answer,
                    key=key,
                    source="original_dropped",
                    source_rank=rank,
                    citation_ids=parse_citations(unit),
                    support_label="obvious_bad",
                    rationale=reason,
                    score=-1.0,
                )
            )
            seen.add(key)
            continue
        candidates.append(
            Candidate(
                answer=answer,
                key=key,
                source="original",
                source_rank=rank,
                citation_ids=parse_citations(unit),
                score=1.0 - 0.01 * rank,
            )
        )
        seen.add(key)
    return candidates


def extractive_candidates(
    question: str,
    ranked_docs: Sequence[tuple[int, dict[str, Any], float]],
    existing: Sequence[Candidate],
    args: argparse.Namespace,
) -> list[Candidate]:
    del question
    seen = {candidate.key for candidate in existing}
    output: list[Candidate] = []
    pattern = re.compile(
        r"\b[A-Z][A-Za-z0-9&'.-]+(?:\s+(?:of|the|and|&|de|da|van|von|[A-Z][A-Za-z0-9&'.-]+)){0,5}\b"
    )
    for doc_rank, (doc_id, doc, _score) in enumerate(ranked_docs, start=1):
        text = format_doc(doc, args)
        for match in pattern.finditer(text):
            answer = normalize_answer_item(match.group(0))
            key = normalize_qampari_item_key(answer)
            if not answer or key in seen or key in GENERIC_KEYS:
                continue
            seen.add(key)
            output.append(
                Candidate(
                    answer=answer,
                    key=key,
                    source="expanded_extractive",
                    source_rank=doc_rank,
                    citation_ids=[doc_id],
                    evidence_sentence=sentence_around(text, match.start()),
                )
            )
            if len(output) >= args.max_expansion_candidates:
                return output
    return output


def sentence_around(text: str, offset: int) -> str:
    left = max(text.rfind(".", 0, offset), text.rfind("?", 0, offset), text.rfind("!", 0, offset))
    right_candidates = [idx for idx in [text.find(".", offset), text.find("?", offset), text.find("!", offset)] if idx >= 0]
    right = min(right_candidates) if right_candidates else min(len(text), offset + 300)
    return normalize_space(text[left + 1 : right + 1])


def verify_candidate(candidate: Candidate, question: str, docs: Sequence[dict[str, Any]], verifier: QampariItemVerifier, args: argparse.Namespace) -> Candidate:
    if candidate.support_label == "obvious_bad":
        return candidate
    cited_docs = docs_by_ids(docs, candidate.citation_ids)
    result = None
    if cited_docs:
        result = verifier.verify(question, candidate.answer, cited_docs)
        if result.get("label") == "supported" and float(result.get("confidence", 0.0)) >= args.min_supported_confidence:
            return apply_verify_result(candidate, result, candidate.citation_ids, source_suffix="cited")

    ranked = rank_docs(question, candidate.answer, docs, args)
    repair_docs = [(doc_id, doc) for doc_id, doc, _ in ranked[: args.repair_top_k]]
    if repair_docs:
        result = verifier.verify(question, candidate.answer, repair_docs)
        if result.get("label") == "supported" and float(result.get("confidence", 0.0)) >= args.min_supported_confidence:
            support_ids = unique_positive_ints(result.get("supporting_doc_ids") or [])
            if not support_ids:
                support_ids = [doc_id for doc_id, _ in repair_docs[: args.repair_group_size]]
            return apply_verify_result(candidate, result, support_ids, source_suffix="repaired")

    updated = Candidate(**candidate.to_json())
    if result:
        updated.support_label = str(result.get("label", "not_supported"))
        updated.support_confidence = float(result.get("confidence", 0.0) or 0.0)
        updated.rationale = str(result.get("rationale", ""))
    else:
        updated.support_label = "not_supported"
    updated.score = candidate.score - 0.35
    return updated


def apply_verify_result(candidate: Candidate, result: dict[str, Any], citation_ids: Sequence[int], source_suffix: str) -> Candidate:
    updated = Candidate(**candidate.to_json())
    updated.support_label = "supported"
    updated.support_confidence = float(result.get("confidence", 0.0) or 0.0)
    support_doc_ids = unique_positive_ints(result.get("supporting_doc_ids") or citation_ids)
    updated.support_doc_ids = support_doc_ids
    updated.citation_ids = unique_positive_ints(support_doc_ids or citation_ids)
    updated.rationale = str(result.get("rationale", ""))
    updated.source = f"{updated.source}_{source_suffix}" if source_suffix not in updated.source else updated.source
    updated.score = max(updated.score, 0.0) + 0.75 + updated.support_confidence
    return updated


def select_candidates(candidates: Sequence[Candidate], args: argparse.Namespace) -> tuple[list[Candidate], list[Candidate]]:
    by_key: dict[str, Candidate] = {}
    for candidate in candidates:
        current = by_key.get(candidate.key)
        if current is None or candidate.score > current.score:
            by_key[candidate.key] = candidate
    unique = list(by_key.values())
    supported = [c for c in unique if c.support_label == "supported" and c.citation_ids]
    original = [c for c in unique if c.source.startswith("original") and c.support_label != "obvious_bad"]
    expanded = [c for c in unique if c.source.startswith("expanded") and c.support_label == "supported" and c.citation_ids]

    supported.sort(key=lambda c: (0 if c.source.startswith("original") else 1, c.source_rank, -c.score))
    expanded.sort(key=lambda c: (c.source_rank, -c.score))
    original.sort(key=lambda c: c.source_rank)

    selected: list[Candidate] = []
    seen = set()

    def add(candidate: Candidate) -> None:
        if len(selected) >= args.max_output_items or candidate.key in seen:
            return
        selected.append(candidate)
        seen.add(candidate.key)

    for candidate in supported:
        add(candidate)
    for candidate in expanded:
        add(candidate)
    if args.keep_unsupported_original:
        for candidate in original:
            if len(selected) >= min(args.backfill_original_until, args.max_output_items):
                break
            add(candidate)

    dropped = [candidate for candidate in unique if candidate.key not in seen]
    return selected, dropped


def render_output(candidates: Sequence[Candidate], args: argparse.Namespace) -> str:
    parts = []
    for candidate in candidates:
        citation_ids = candidate.citation_ids or candidate.support_doc_ids
        citations = format_citations(citation_ids, max_citations=args.max_citations_per_item)
        if not citations:
            continue
        parts.append(f"{candidate.answer} {citations}")
    return ", ".join(parts).rstrip(", ") + ("." if parts else "")


def revise_item(
    item: dict[str, Any],
    item_id: int,
    provider: CandidateDocProvider,
    verifier: QampariItemVerifier,
    expander: QampariExpansionGenerator,
    joint_selector: QampariJointSelector | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    updated = dict(item)
    question = str(updated.get(args.question_field, "") or "")
    raw_docs = list(updated.get(args.docs_field, []) or [])
    candidate_docs = provider.docs_for(updated, item_id)
    docs = merge_doc_pool(raw_docs, candidate_docs, args)
    updated[args.docs_field] = docs

    original_output = str(updated.get(args.output_field, "") or "")
    original_candidates = parse_original_candidates(original_output, question, args)

    expanded_candidates: list[Candidate] = []
    if args.selection_mode == "joint":
        if joint_selector is None:
            raise ValueError("joint selection mode requires a joint selector")
        selected, dropped = joint_selector.select(question, original_candidates, docs, args)
        candidates = list(selected) + list(dropped)
    else:
        candidates = [verify_candidate(candidate, question, docs, verifier, args) for candidate in original_candidates]
        expanded_candidates = expander.generate(question, docs, candidates, args)
        verified_expanded = [
            verify_candidate(candidate, question, docs, verifier, args)
            for candidate in expanded_candidates
            if candidate.key not in {c.key for c in candidates}
        ]
        candidates.extend(verified_expanded)
        selected, dropped = select_candidates(candidates, args)
    revised_output = render_output(selected, args)
    if not revised_output and original_output:
        if args.empty_output_policy == "original":
            # Legacy fallback.  Useful for debugging, but it can carry
            # citation-less refusal text into QAMPARI's comma-separated parser.
            revised_output = original_output
        else:
            revised_output = ""
    updated[args.output_field] = revised_output
    updated["cover_qampari"] = {
        "original_output": original_output,
        "revised_output": revised_output,
        "raw_doc_count": len(raw_docs),
        "candidate_doc_count": len(candidate_docs),
        "doc_pool_size": len(docs),
        "original_candidates": [candidate.to_json() for candidate in original_candidates],
        "verified_candidates": [candidate.to_json() for candidate in candidates],
        "selected_candidates": [candidate.to_json() for candidate in selected],
        "dropped_candidates": [candidate.to_json() for candidate in dropped],
        "metrics": {
            "num_original_candidates": len([c for c in original_candidates if c.source != "original_dropped"]),
            "num_expanded_candidates": len(expanded_candidates),
            "num_selected": len(selected),
            "num_supported_selected": sum(1 for c in selected if c.support_label == "supported"),
            "num_original_selected": sum(1 for c in selected if "original" in c.source),
            "num_expanded_selected": sum(1 for c in selected if "expanded" in c.source),
            "num_joint_selected": sum(1 for c in selected if c.source.startswith("joint")),
        },
    }
    return updated


def summarize(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    records = [item.get("cover_qampari", {}) for item in items if isinstance(item.get("cover_qampari"), dict)]
    metrics = [record.get("metrics", {}) for record in records]
    label_counts: collections.Counter[str] = collections.Counter()
    for record in records:
        for candidate in record.get("selected_candidates", []) or []:
            label_counts[str(candidate.get("support_label", "unknown"))] += 1

    def mean(key: str) -> float:
        values = [float(metric.get(key, 0.0) or 0.0) for metric in metrics]
        return statistics.mean(values) if values else 0.0

    total_selected = sum(int(metric.get("num_selected", 0) or 0) for metric in metrics)
    total_supported = sum(int(metric.get("num_supported_selected", 0) or 0) for metric in metrics)
    return {
        "num_examples": len(records),
        "avg_original_candidates": mean("num_original_candidates"),
        "avg_expanded_candidates": mean("num_expanded_candidates"),
        "avg_selected": mean("num_selected"),
        "avg_supported_selected": mean("num_supported_selected"),
        "avg_original_selected": mean("num_original_selected"),
        "avg_expanded_selected": mean("num_expanded_selected"),
        "selected_support_rate": total_supported / total_selected if total_selected else 0.0,
        "selected_label_counts": dict(label_counts),
    }


def main() -> None:
    args = parse_args()
    if args.item_verifier == "llm" and not args.openai_api:
        raise ValueError("--item-verifier llm requires --openai-api")
    if args.expand_from_docs == "llm" and not args.openai_api:
        raise ValueError("--expand-from-docs llm requires --openai-api")
    if args.selection_mode == "joint" and not args.openai_api:
        raise ValueError("--selection-mode joint requires --openai-api")

    payload, items = load_payload(args.input)
    process_items = items if args.limit is None else items[: args.limit]
    cache = make_json_cache(args.cache_file, args.cache_save_every)
    provider = CandidateDocProvider(args.candidate_docs_file, args)
    verifier = QampariItemVerifier(args, cache)
    expander = QampariExpansionGenerator(args, cache)
    joint_selector = QampariJointSelector(args, cache) if args.selection_mode == "joint" else None

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = lambda x, **_: x  # type: ignore

    revised = []
    for item_id, item in enumerate(tqdm(process_items, desc="COVER-QAMPARI select")):
        revised.append(revise_item(item, item_id, provider, verifier, expander, joint_selector, args))
        if args.cache_save_every > 0 and (item_id + 1) % args.cache_save_every == 0:
            flush_json_cache(cache)
    if args.limit is not None and len(items) > args.limit:
        revised.extend(items[args.limit :])

    payload["data"] = revised
    payload["cover_qampari_summary"] = summarize(revised[: len(process_items)])
    payload["cover_qampari_config"] = {
        "item_verifier": args.item_verifier,
        "item_verify_model": args.item_verify_model if args.item_verifier == "llm" else None,
        "selection_mode": args.selection_mode,
        "joint_model": args.joint_model if args.selection_mode == "joint" else None,
        "joint_docs": args.joint_docs,
        "joint_candidate_spans": args.joint_candidate_spans,
        "joint_require_type_match": args.joint_require_type_match,
        "strict_question_type_guard": args.strict_question_type_guard,
        "expand_from_docs": args.expand_from_docs,
        "expansion_model": args.expansion_model if args.expand_from_docs == "llm" else None,
        "max_output_items": args.max_output_items,
        "backfill_original_until": args.backfill_original_until,
        "preserve_original_top_k": args.preserve_original_top_k,
        "preserved_original_label": args.preserved_original_label,
        "backfill_requires_type_safe": args.backfill_requires_type_safe,
        "empty_output_policy": args.empty_output_policy,
        "keep_unsupported_original": args.keep_unsupported_original,
        "drop_obvious_bad": args.drop_obvious_bad,
        "max_citations_per_item": args.max_citations_per_item,
    }
    token_usage: dict[str, int] = {}
    if verifier.prompt_tokens or verifier.completion_tokens:
        token_usage["item_verify_prompt_tokens"] = verifier.prompt_tokens
        token_usage["item_verify_completion_tokens"] = verifier.completion_tokens
    if expander.prompt_tokens or expander.completion_tokens:
        token_usage["expansion_prompt_tokens"] = expander.prompt_tokens
        token_usage["expansion_completion_tokens"] = expander.completion_tokens
    if joint_selector is not None and (joint_selector.prompt_tokens or joint_selector.completion_tokens):
        token_usage["joint_prompt_tokens"] = joint_selector.prompt_tokens
        token_usage["joint_completion_tokens"] = joint_selector.completion_tokens
    if token_usage:
        payload["cover_qampari_token_usage"] = token_usage

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(safe_float(payload), ensure_ascii=False, indent=2 if args.pretty else None),
        encoding="utf-8",
    )
    flush_json_cache(cache)
    logger.info("Wrote COVER-QAMPARI result to %s", args.output)
    logger.info("Summary: %s", json.dumps(payload["cover_qampari_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

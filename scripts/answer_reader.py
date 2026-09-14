#!/usr/bin/env python3
"""Evidence-first answer realization for COVER-RAG.

The controller builds a citation-authorized claim blueprint before asking the
host reader to write an answer.  The reader may group and verbalize blueprint
claims, but code owns citation binding and falls back to verified atomic claims
when the realization violates the blueprint.  Realization therefore requires
at most one reader call and never starts a retrieve/regenerate loop.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Sequence

from claim_audit import (
    JsonCache,
    OpenAIChatClient,
    extract_json_object,
    normalize_space,
    remove_citations,
    stable_hash,
)
from answer_revision import clean_final_answer, format_citations, unique_positive_ints


logger = logging.getLogger("answer_reader")


READER_SYSTEM = """You are the answer realization component of a citation-grounded RAG system.
An upstream controller has already selected and verified every factual claim.
Use only the supplied plan. Do not introduce facts, names, dates, numbers,
causes, comparisons, qualifications, or conclusions that are absent from it.
Return strict JSON only. Do not write citation markers; the controller binds
citations after validating your plan references."""


READER_USER = """Question:
{question}

Answer format: {answer_format}

Verified answer plan:
{plan_json}

Return this JSON schema:
{{"units":[{{"plan_ids":["p1"],"text":"a concise answer unit"}}]}}

Rules:
1. Every output unit must cite one or more supplied plan_ids.
2. Use every plan_id exactly once; do not invent plan_ids.
3. A unit may combine plan_ids only when its text expresses all and only those claims.
4. Preserve names, dates, quantities, negation, comparisons, and qualifications exactly.
5. For paragraph format, write concise complete sentences that directly answer the question.
6. For qampari format, write only the answer item represented by each plan entry.
7. Do not add introductions, conclusions, source descriptions, caveats, or outside knowledge.
8. Do not include bracketed citations or markdown.
9. Keep each unit under {max_unit_words} words.
10. Return minified valid JSON only.
"""


def _plan_claim_text(claim: dict[str, Any]) -> str:
    return normalize_space(str(claim.get("claim", "") or ""))


def _plan_answer_span(claim: dict[str, Any]) -> str:
    return normalize_space(str(claim.get("answer_span", "") or ""))


def _plan_citations(claim: dict[str, Any], num_docs: int) -> list[int]:
    values = (
        claim.get("final_citation_ids")
        or claim.get("evidence_doc_ids")
        or claim.get("citation_ids")
        or claim.get("source_citation_ids")
        or []
    )
    return [citation_id for citation_id in unique_positive_ints(values) if citation_id <= num_docs]


def build_answer_blueprint(
    claims: Sequence[dict[str, Any]],
    docs: Sequence[dict[str, Any]],
    *,
    max_claims: int,
) -> list[dict[str, Any]]:
    """Convert verified claims into citation-authorized reader plan entries."""

    plan: list[dict[str, Any]] = []
    seen: set[str] = set()
    for claim in claims:
        claim_text = _plan_claim_text(claim)
        citations = _plan_citations(claim, len(docs))
        key = remove_citations(claim_text).lower()
        if not claim_text or not citations or not key or key in seen:
            continue
        seen.add(key)
        plan.append(
            {
                "plan_id": f"p{len(plan) + 1}",
                "claim": claim_text,
                "answer_span": _plan_answer_span(claim),
                "citation_ids": citations,
                "evidence": normalize_space(
                    str(claim.get("source_sentence_without_citations", "") or claim.get("source_sentence", "") or "")
                ),
                "confidence": float(claim.get("confidence", 0.0) or 0.0),
                "target_gap": normalize_space(str(claim.get("target_gap", "") or "")),
                "iteration_round": int(claim.get("iteration_round", 0) or 0),
            }
        )
        if len(plan) >= max(1, int(max_claims)):
            break
    return plan


def _safe_unit_text(text: Any) -> str:
    value = normalize_space(remove_citations(str(text or "")))
    return value.strip(" \t\r\n-,:;")


def _fallback_text(entry: dict[str, Any], answer_format: str) -> str:
    if answer_format == "qampari":
        span = normalize_space(str(entry.get("answer_span", "") or ""))
        if span:
            return span
    return normalize_space(str(entry.get("claim", "") or ""))


def _render_unit(text: str, citation_ids: Sequence[int], max_citations: int, answer_format: str) -> str:
    citations = format_citations(citation_ids, max_citations=max_citations)
    if answer_format == "qampari":
        return f"{text.rstrip('.,;:')} {citations}".strip()
    sentence = text.rstrip()
    punctuation = sentence[-1] if sentence and sentence[-1] in ".!?" else "."
    sentence = sentence.rstrip(".!?")
    return f"{sentence} {citations}{punctuation}".strip()


class BlueprintAnswerReader:
    """Realize a verified answer blueprint with one optional host-reader call."""

    def __init__(
        self,
        model: str,
        cache: JsonCache,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_retries: int = 5,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.cache = cache
        self.client = client or OpenAIChatClient(model, temperature, top_p, max_retries)
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.api_calls = 0

    def _verify_unit(
        self,
        question: str,
        text: str,
        entries: Sequence[dict[str, Any]],
        docs: Sequence[dict[str, Any]],
        verifier: Any,
        args: Any,
    ) -> tuple[bool, dict[str, Any]]:
        citation_ids = unique_positive_ints(
            [citation_id for entry in entries for citation_id in entry.get("citation_ids", []) or []]
        )
        evidence_docs = [
            (citation_id, docs[citation_id - 1])
            for citation_id in citation_ids
            if 0 < citation_id <= len(docs)
        ]
        if not evidence_docs:
            return False, {"label": "no_evidence", "citation_ids": citation_ids}
        result = verifier.verify(question, text, evidence_docs, args)
        return result.label == "supported", {
            "label": result.label,
            "confidence": result.confidence,
            "entailment": result.entailment,
            "citation_ids": citation_ids,
        }

    def realize(
        self,
        question: str,
        claims: Sequence[dict[str, Any]],
        docs: Sequence[dict[str, Any]],
        verifier: Any,
        args: Any,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        answer_format = str(getattr(args, "answer_format", "paragraph") or "paragraph")
        if answer_format == "auto":
            dataset = str(getattr(args, "dataset_name", "") or "").lower()
            answer_format = "qampari" if dataset == "qampari" else "paragraph"
        max_plan_claims = max(1, int(getattr(args, "reader_max_plan_claims", 24) or 24))
        max_unit_words = max(4, int(getattr(args, "reader_max_unit_words", 80) or 80))
        max_citations = max(1, int(getattr(args, "max_citations_per_sentence", 4) or 4))
        plan = build_answer_blueprint(claims, docs, max_claims=max_plan_claims)
        diagnostics: dict[str, Any] = {
            "mode": "blueprint_single_call",
            "model": self.model,
            "planned_claims": len(plan),
            "logical_reader_calls": 0,
            "api_calls": 0,
            "cache_hit": False,
            "structured_success": False,
            "accepted_generated_units": 0,
            "fallback_units": 0,
            "unknown_plan_ids": 0,
            "duplicate_plan_ids": 0,
            "conformance_rejections": 0,
        }
        if not plan:
            return "The provided evidence is insufficient to answer the question.", plan, diagnostics

        prompt_plan = [
            {
                "plan_id": entry["plan_id"],
                "claim": entry["claim"],
                "answer_span": entry["answer_span"],
                "evidence": entry["evidence"],
            }
            for entry in plan
        ]
        plan_json = json.dumps(prompt_plan, ensure_ascii=False, separators=(",", ":"))
        key = "evidence-first-reader:" + stable_hash(question, answer_format, self.model, plan_json)
        raw = self.cache.get(key)
        diagnostics["logical_reader_calls"] = 1
        if raw is None:
            try:
                user = READER_USER.format(
                    question=question,
                    answer_format=answer_format,
                    plan_json=plan_json,
                    max_unit_words=max_unit_words,
                )
                raw, usage = self.client.chat(
                    READER_SYSTEM,
                    user,
                    max_tokens=max(64, int(getattr(args, "reader_max_tokens", 1200) or 1200)),
                )
                self.api_calls += 1
                diagnostics["api_calls"] = 1
                self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                self.cache.set(key, raw)
            except Exception as exc:
                diagnostics["reader_error"] = str(exc)
                logger.warning("Blueprint reader failed; using verified-plan fallback: %s", exc)
                raw = ""
        else:
            diagnostics["cache_hit"] = True

        try:
            payload = extract_json_object(str(raw or ""))
            units = payload.get("units", [])
            if not isinstance(units, list):
                raise ValueError("reader response field `units` is not a list")
            diagnostics["structured_success"] = True
        except Exception as exc:
            diagnostics["parse_error"] = str(exc)
            units = []

        by_id = {entry["plan_id"]: entry for entry in plan}
        consumed: set[str] = set()
        rendered: list[str] = []
        validation_rows: list[dict[str, Any]] = []
        verify_conformance = bool(getattr(args, "reader_conformance_verify", True))

        for unit in units:
            if not isinstance(unit, dict):
                continue
            raw_ids = unit.get("plan_ids", [])
            if not isinstance(raw_ids, list):
                raw_ids = []
            selected_ids: list[str] = []
            for value in raw_ids:
                plan_id = str(value or "").strip()
                if plan_id not in by_id:
                    diagnostics["unknown_plan_ids"] += 1
                    continue
                if plan_id in consumed or plan_id in selected_ids:
                    diagnostics["duplicate_plan_ids"] += 1
                    continue
                selected_ids.append(plan_id)
            text = _safe_unit_text(unit.get("text", ""))
            if not selected_ids or not text or len(text.split()) > max_unit_words:
                continue
            entries = [by_id[plan_id] for plan_id in selected_ids]
            accepted = True
            verify_diag: dict[str, Any] = {"label": "not_checked"}
            if verify_conformance:
                accepted, verify_diag = self._verify_unit(question, text, entries, docs, verifier, args)
            validation_rows.append({"plan_ids": selected_ids, "text": text, **verify_diag, "accepted": accepted})
            if not accepted:
                diagnostics["conformance_rejections"] += 1
                continue
            citation_ids = unique_positive_ints(
                [citation_id for entry in entries for citation_id in entry.get("citation_ids", []) or []]
            )
            rendered.append(_render_unit(text, citation_ids, max_citations, answer_format))
            consumed.update(selected_ids)
            diagnostics["accepted_generated_units"] += 1

        for entry in plan:
            if entry["plan_id"] in consumed:
                continue
            text = _fallback_text(entry, answer_format)
            if not text:
                continue
            rendered.append(_render_unit(text, entry["citation_ids"], max_citations, answer_format))
            consumed.add(entry["plan_id"])
            diagnostics["fallback_units"] += 1

        diagnostics["realized_plan_claims"] = len(consumed)
        diagnostics["validation"] = validation_rows
        if not rendered:
            answer = "The provided evidence is insufficient to answer the question."
        elif answer_format == "qampari":
            answer = ", ".join(part.rstrip("., ") for part in rendered) + "."
        else:
            answer = " ".join(rendered)
        return clean_final_answer(answer, num_docs=len(docs)), plan, diagnostics

#!/usr/bin/env python3
"""Build weak-supervision data for a coverage-aware evidence reranker.

Use this on training splits, not on the ALCE eval split used for reporting.
The output is JSONL with pair-classification fields:

    text_a: question + covered answer + missing answer unit
    text_b: candidate evidence sentence/passage
    label: 1 if evidence appears to fill the missing unit, else 0
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from coverage_reranker import format_candidate_evidence, format_coverage_query
from cover_posthoc_audit import normalize_space, remove_citations, sent_tokenize_safe


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
    "when",
    "where",
    "why",
    "how",
}


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", remove_citations(str(text or "")).lower())
        if token not in STOPWORDS and len(token) > 1
    }


def overlap_score(a: str, b: str) -> float:
    a_tokens = content_tokens(a)
    if not a_tokens:
        return 0.0
    b_tokens = content_tokens(b)
    if not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens)


def unit_key(text: str) -> str:
    return " ".join(sorted(content_tokens(text)))


def flatten_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key in ("answer", "answers", "short_answer", "short_answers", "claim", "claims", "text"):
            if key in value:
                yield from flatten_strings(value[key])
    elif isinstance(value, list):
        for item in value:
            yield from flatten_strings(item)


def gold_units(item: dict[str, Any], dataset_name: str) -> list[str]:
    units: list[str] = []
    if dataset_name == "eli5" and item.get("claims"):
        units.extend(flatten_strings(item.get("claims")))
    if dataset_name == "qampari" and item.get("answers"):
        units.extend(flatten_strings(item.get("answers")))
    if dataset_name == "asqa" and item.get("qa_pairs"):
        units.extend(flatten_strings(item.get("qa_pairs")))
    if not units:
        units.extend(flatten_strings(item.get("answer")))

    output: list[str] = []
    seen: set[str] = set()
    for unit in units:
        unit = normalize_space(remove_citations(unit)).strip().strip("\"'")
        unit = unit.rstrip(".;:,")
        if len(unit) < 2:
            continue
        key = unit_key(unit) or unit.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(unit)
    return output[:80]


def doc_title(doc: dict[str, Any]) -> str:
    return normalize_space(str(doc.get("title", "") or doc.get("wikipedia_title", "") or ""))


def doc_text(doc: dict[str, Any]) -> str:
    return normalize_space(str(doc.get("sent") or doc.get("text") or doc.get("contents") or ""))


def candidate_sentences(item: dict[str, Any], max_docs: int, max_sentences_per_doc: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for doc_idx, doc in enumerate(list(item.get("docs", []) or [])[:max_docs], start=1):
        title = doc_title(doc)
        text = doc_text(doc)
        if not text:
            continue
        sentences = sent_tokenize_safe(text)
        if not sentences:
            sentences = [text]
        for sent_idx, sentence in enumerate(sentences[:max_sentences_per_doc], start=1):
            sentence = normalize_space(sentence)
            if len(sentence) < 20:
                continue
            rows.append(
                {
                    "doc_id": doc_idx,
                    "sent_id": sent_idx,
                    "title": title,
                    "sentence": sentence,
                    "evidence": format_candidate_evidence(sentence, title=title),
                }
            )
    return rows


def sentence_fills_unit(sentence: str, unit: str, min_unit_overlap: float) -> bool:
    if not unit:
        return False
    sentence_key = " " + normalize_space(sentence).lower() + " "
    unit_lower = normalize_space(unit).lower()
    if len(unit_lower) >= 4 and f" {unit_lower} " in sentence_key:
        return True
    return overlap_score(unit, sentence) >= min_unit_overlap


def build_examples_for_item(
    item: dict[str, Any],
    dataset_name: str,
    *,
    max_docs: int,
    max_sentences_per_doc: int,
    positives_per_unit: int,
    negatives_per_unit: int,
    min_unit_overlap: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    question = normalize_space(str(item.get("question", "") or item.get("question_ctx", "") or ""))
    if not question:
        return []
    units = gold_units(item, dataset_name)
    if not units:
        return []
    sentences = candidate_sentences(item, max_docs, max_sentences_per_doc)
    if not sentences:
        return []

    examples: list[dict[str, Any]] = []
    for unit_idx, missing_unit in enumerate(units):
        covered_units = [unit for idx, unit in enumerate(units) if idx != unit_idx]
        rng.shuffle(covered_units)
        covered_answer = "; ".join(covered_units[:6])
        text_a = format_coverage_query(question, covered_answer=covered_answer, missing_unit=missing_unit)

        positives = [
            row for row in sentences if sentence_fills_unit(row["sentence"], missing_unit, min_unit_overlap)
        ]
        negatives = [
            row
            for row in sentences
            if row not in positives and overlap_score(question, row["sentence"]) >= 0.04
        ]
        rng.shuffle(positives)
        rng.shuffle(negatives)

        for row in positives[:positives_per_unit]:
            examples.append(
                {
                    "text_a": text_a,
                    "text_b": row["evidence"],
                    "label": 1,
                    "dataset": dataset_name,
                    "question": question,
                    "missing_unit": missing_unit,
                    "doc_id": row["doc_id"],
                    "sent_id": row["sent_id"],
                    "weak_label_source": "gold_unit_sentence_match",
                }
            )
        for row in negatives[:negatives_per_unit]:
            examples.append(
                {
                    "text_a": text_a,
                    "text_b": row["evidence"],
                    "label": 0,
                    "dataset": dataset_name,
                    "question": question,
                    "missing_unit": missing_unit,
                    "doc_id": row["doc_id"],
                    "sent_id": row["sent_id"],
                    "weak_label_source": "question_relevant_non_filling_sentence",
                }
            )
    return examples


def infer_dataset_name(path: Path, requested: str | None) -> str:
    if requested:
        return requested
    lowered = path.name.lower()
    for name in ("asqa", "eli5", "qampari"):
        if name in lowered:
            return name
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build weak data for a coverage-aware evidence reranker.")
    parser.add_argument("--input", type=Path, required=True, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name", choices=["asqa", "eli5", "qampari", "unknown"], default=None)
    parser.add_argument("--allow-eval-input", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-docs", type=int, default=100)
    parser.add_argument("--max-sentences-per-doc", type=int, default=12)
    parser.add_argument("--positives-per-unit", type=int, default=2)
    parser.add_argument("--negatives-per-unit", type=int, default=4)
    parser.add_argument("--min-unit-overlap", type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    for path in args.input:
        if not args.allow_eval_input and "eval" in path.name.lower():
            raise ValueError(
                f"{path} looks like an evaluation file. Use official train data, "
                "or pass --allow-eval-input only for diagnostics."
            )
        dataset_name = infer_dataset_name(path, args.dataset_name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        data = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(data, list):
            raise ValueError(f"Expected list-like data in {path}")
        if args.max_items is not None:
            data = data[: args.max_items]
        for item in data:
            if isinstance(item, dict):
                rows.extend(
                    build_examples_for_item(
                        item,
                        dataset_name,
                        max_docs=args.max_docs,
                        max_sentences_per_doc=args.max_sentences_per_doc,
                        positives_per_unit=args.positives_per_unit,
                        negatives_per_unit=args.negatives_per_unit,
                        min_unit_overlap=args.min_unit_overlap,
                        rng=rng,
                    )
                )

    rng.shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    positives = sum(1 for row in rows if int(row.get("label", 0)) == 1)
    print(f"Wrote {len(rows)} reranker examples to {args.output} ({positives} positive, {len(rows)-positives} negative).")


if __name__ == "__main__":
    main()


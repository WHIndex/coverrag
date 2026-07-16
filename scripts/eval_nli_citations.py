#!/usr/bin/env python3
"""Evaluate citation support with a local NLI verifier.

This is a lightweight replacement for ALCE's AutoAIS citation evaluator when
AutoAIS cannot be loaded, or when we want attribution evaluation to use the
same NLI decision rule as COVER-RAG's claim verifier.

Important: this is not the official ALCE AutoAIS metric. By default the script
writes both:

    citation_nli_rec / citation_nli_prec

and, with --overwrite-alce-names, also fills:

    citation_rec / citation_prec

so existing summary scripts can keep working.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

from cover_posthoc_audit import (
    NLIVerifier,
    docs_by_citation_ids,
    load_payload,
    normalize_space,
    parse_citations,
    remove_citations,
    split_answer_units,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("eval_nli_citations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NLI-based citation evaluation for ALCE-style result JSON.")
    parser.add_argument("--f", "--result", dest="result", required=True, type=Path)
    parser.add_argument(
        "--score-file",
        type=Path,
        default=None,
        help="Score JSON to update. Default: <result>.score",
    )
    parser.add_argument(
        "--base-score",
        type=Path,
        default=None,
        help="Optional existing score JSON to copy first, e.g. the raw ALCE .json.score.",
    )
    parser.add_argument("--output-field", default="output")
    parser.add_argument("--docs-field", default="docs")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--title-field", default="title")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--sent-field", default="sent")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--answer-format", choices=["auto", "paragraph", "qampari"], default="auto")

    parser.add_argument("--nli-model", default="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")
    parser.add_argument("--device", default=None)
    parser.add_argument("--nli-batch-size", type=int, default=8)
    parser.add_argument("--nli-max-length", type=int, default=512)
    parser.add_argument("--nli-aggregation", choices=["joint", "max_doc", "joint_then_max"], default="joint_then_max")
    parser.add_argument("--entail-threshold", type=float, default=0.50)
    parser.add_argument("--contradiction-threshold", type=float, default=0.50)
    parser.add_argument("--ambiguous-margin", type=float, default=0.10)
    parser.add_argument("--max-evidence-chars", type=int, default=6000)
    parser.add_argument("--lexical-threshold", type=float, default=0.45)
    parser.add_argument(
        "--precision-mode",
        choices=["single_doc", "joint_without_doc"],
        default="single_doc",
        help=(
            "single_doc: a citation is precise if that individual cited document supports the sentence. "
            "joint_without_doc: a citation is precise if removing it makes the cited evidence no longer support the sentence."
        ),
    )
    parser.add_argument(
        "--overwrite-alce-names",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write citation_rec/citation_prec so existing ALCE summary scripts can read them.",
    )
    parser.add_argument("--details", type=Path, default=None, help="Optional per-sentence diagnostic JSON.")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def fill_dataset_name_from_payload(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    if args.dataset_name:
        return
    payload_args = payload.get("args", {}) if isinstance(payload, dict) else {}
    if isinstance(payload_args, dict):
        value = payload_args.get("dataset_name") or payload_args.get("dataset") or payload_args.get("data_name")
        if value:
            args.dataset_name = str(value)
            return
    name = args.result.name.lower()
    for dataset in ("asqa", "eli5", "qampari"):
        if name.startswith(dataset + "-"):
            args.dataset_name = dataset
            return


def load_score(path: Path | None) -> dict[str, Any]:
    if path and path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
        except Exception as exc:
            logger.warning("Could not read score file %s: %s", path, exc)
    return {}


def valid_cited_docs(docs: Sequence[dict[str, Any]], citation_ids: Sequence[int]) -> list[tuple[int, dict[str, Any]]]:
    return docs_by_citation_ids(docs, citation_ids)


def verify_supported(verifier: NLIVerifier, question: str, claim: str, evidence_docs: Sequence[tuple[int, dict[str, Any]]], args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    result = verifier.verify(question, claim, evidence_docs, args)
    return result.label == "supported", {
        "label": result.label,
        "confidence": result.confidence,
        "entailment": result.entailment,
        "neutral": result.neutral,
        "contradiction": result.contradiction,
        "evidence_doc_ids": result.evidence_doc_ids or [],
        "rationale": result.rationale,
    }


def citation_is_precise_single_doc(
    verifier: NLIVerifier,
    question: str,
    claim: str,
    doc_pair: tuple[int, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[bool, dict[str, Any]]:
    return verify_supported(verifier, question, claim, [doc_pair], args)


def citation_is_precise_joint_without_doc(
    verifier: NLIVerifier,
    question: str,
    claim: str,
    cited_docs: Sequence[tuple[int, dict[str, Any]]],
    target_doc_id: int,
    args: argparse.Namespace,
) -> tuple[bool, dict[str, Any]]:
    remaining = [(doc_id, doc) for doc_id, doc in cited_docs if doc_id != target_doc_id]
    if not remaining:
        return True, {"label": "necessary_single_citation", "evidence_doc_ids": [target_doc_id]}
    still_supported, diagnostics = verify_supported(verifier, question, claim, remaining, args)
    return not still_supported, diagnostics


def compute_nli_citations(payload: dict[str, Any], items: Sequence[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fill_dataset_name_from_payload(args, payload)
    verifier = NLIVerifier(args)

    sentence_total = 0
    sentence_supported = 0
    no_citation_sentences = 0
    invalid_citation_links = 0
    citation_links = 0
    citation_links_supported = 0
    details = []

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **_: x

    for item_id, item in enumerate(tqdm(items, desc="NLI citation eval")):
        question = str(item.get(args.question_field, "") or "")
        answer = str(item.get(args.output_field, "") or "")
        docs = item.get(args.docs_field, []) or []
        sentences = split_answer_units(answer, args)

        for sent_id, sentence in enumerate(sentences):
            claim = normalize_space(remove_citations(sentence))
            if not claim:
                continue
            citation_ids = parse_citations(sentence)
            cited_docs = valid_cited_docs(docs, citation_ids)
            valid_ids = [doc_id for doc_id, _ in cited_docs]
            invalid_citation_links += max(0, len(citation_ids) - len(valid_ids))

            sentence_total += 1
            record: dict[str, Any] = {
                "item_id": item_id,
                "sentence_id": sent_id,
                "sentence": sentence,
                "sentence_without_citations": claim,
                "citation_ids": citation_ids,
                "valid_citation_ids": valid_ids,
            }

            if not cited_docs:
                no_citation_sentences += 1
                record["recall_supported"] = False
                record["recall_diagnostics"] = {"label": "no_valid_citation"}
                details.append(record)
                continue

            recall_supported, recall_diagnostics = verify_supported(verifier, question, claim, cited_docs, args)
            if recall_supported:
                sentence_supported += 1
            record["recall_supported"] = recall_supported
            record["recall_diagnostics"] = recall_diagnostics

            link_records = []
            for doc_pair in cited_docs:
                doc_id, _ = doc_pair
                citation_links += 1
                if args.precision_mode == "joint_without_doc":
                    precise, diag = citation_is_precise_joint_without_doc(
                        verifier, question, claim, cited_docs, doc_id, args
                    )
                else:
                    precise, diag = citation_is_precise_single_doc(
                        verifier, question, claim, doc_pair, args
                    )
                if precise:
                    citation_links_supported += 1
                link_records.append({"doc_id": doc_id, "precise": precise, "diagnostics": diag})
            record["precision_links"] = link_records
            details.append(record)

    citation_rec = 100.0 * sentence_supported / sentence_total if sentence_total else 0.0
    citation_prec = 100.0 * citation_links_supported / citation_links if citation_links else 0.0
    metrics = {
        "citation_nli_rec": citation_rec,
        "citation_nli_prec": citation_prec,
        "citation_nli_sentence_count": sentence_total,
        "citation_nli_supported_sentences": sentence_supported,
        "citation_nli_no_citation_sentences": no_citation_sentences,
        "citation_nli_link_count": citation_links,
        "citation_nli_supported_links": citation_links_supported,
        "citation_nli_invalid_links": invalid_citation_links,
        "citation_nli_precision_mode": args.precision_mode,
        "citation_nli_model": args.nli_model,
        "citation_nli_entail_threshold": args.entail_threshold,
    }
    return metrics, details


def main() -> None:
    args = parse_args()
    payload, items = load_payload(args.result)
    score_path = args.score_file or Path(str(args.result) + ".score")

    score = load_score(args.base_score)
    score.update(load_score(score_path))
    metrics, details = compute_nli_citations(payload, items, args)

    if args.overwrite_alce_names:
        if "citation_rec" in score and "autoais_citation_rec" not in score:
            score["autoais_citation_rec"] = score["citation_rec"]
        if "citation_prec" in score and "autoais_citation_prec" not in score:
            score["autoais_citation_prec"] = score["citation_prec"]
        score["citation_rec"] = metrics["citation_nli_rec"]
        score["citation_prec"] = metrics["citation_nli_prec"]

    score.update(metrics)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(
        json.dumps(score, ensure_ascii=False, indent=2 if args.pretty else None),
        encoding="utf-8",
    )
    logger.info("Wrote NLI citation score to %s", score_path)
    logger.info("NLI citation metrics: %s", json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.details:
        args.details.parent.mkdir(parents=True, exist_ok=True)
        args.details.write_text(
            json.dumps(details, ensure_ascii=False, indent=2 if args.pretty else None),
            encoding="utf-8",
        )
        logger.info("Wrote NLI citation details to %s", args.details)


if __name__ == "__main__":
    main()

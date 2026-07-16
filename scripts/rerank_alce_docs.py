#!/usr/bin/env python3
"""Rerank ALCE candidate documents with a cross-encoder reranker.

This script is intentionally separated from ALCE's `run.py`.
It turns an existing retrieval file such as:

    data/asqa_eval_gtr_top100.json

into a reranked retrieval file such as:

    data/asqa_eval_gtr_top100_bge_reranked.json

The output preserves ALCE's original item structure. Only the order of each
item's `docs` list changes, so ALCE's original generation and evaluation
scripts can keep working unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_MODEL = "BAAI/bge-reranker-large"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerank ALCE top-k retrieved documents with a cross-encoder reranker."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="ALCE retrieval JSON file, e.g. data/asqa_eval_gtr_top100.json",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Where to write the reranked ALCE retrieval JSON.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HF reranker model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device. Default: cuda when available, else cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Pairs scored per forward pass. Lower this if GPU memory is tight.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Tokenizer max length for each (query, document) pair.",
    )
    parser.add_argument(
        "--candidate-topn",
        type=int,
        default=None,
        help="Only rerank the first N docs and keep the remainder after them. Default: rerank all docs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N items. Useful for quick debugging.",
    )
    parser.add_argument(
        "--question-field",
        default="question",
        help="Field name containing the query/question text.",
    )
    parser.add_argument(
        "--docs-field",
        default="docs",
        help="Field name containing candidate documents.",
    )
    parser.add_argument(
        "--title-field",
        default="title",
        help="Field name containing a document title.",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Field name containing a document body.",
    )
    parser.add_argument(
        "--keep-scores",
        action="store_true",
        help="Attach `_rerank_score` and `_original_rank` to each reranked document for debugging.",
    )
    parser.add_argument(
        "--report-topk",
        type=int,
        default=5,
        help="Report overlap between original and reranked top-k docs. Default: 5.",
    )
    return parser.parse_args()


def load_items(payload: Any) -> tuple[list[dict[str, Any]], bool]:
    """Return items plus whether they came from a top-level {'data': ...} object."""
    if isinstance(payload, list):
        return payload, False
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"], True
    raise ValueError(
        "Unsupported JSON structure. Expected either a list of ALCE items "
        "or a dict containing a list under the key `data`."
    )


def restore_payload(
    original_payload: Any,
    reranked_items: list[dict[str, Any]],
    wrapped_in_data: bool,
) -> Any:
    if wrapped_in_data:
        output_payload = copy.deepcopy(original_payload)
        output_payload["data"] = reranked_items
        return output_payload
    return reranked_items


def build_document_text(
    doc: dict[str, Any],
    title_field: str,
    text_field: str,
) -> str:
    title = str(doc.get(title_field, "") or "").strip()
    text = str(doc.get(text_field, "") or "").strip()
    if title and text:
        return f"{title}. {text}"
    return title or text


def chunked(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        device: str,
        max_length: int,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Reranking requires `torch` and `transformers`. "
                "Install them in the same environment that runs ALCE."
            ) from exc

        self.torch = torch
        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def score_pairs(
        self,
        query: str,
        passages: Sequence[str],
        batch_size: int,
    ) -> list[float]:
        scores: list[float] = []
        with self.torch.inference_mode():
            for batch_passages in chunked(passages, batch_size):
                batch_queries = [query] * len(batch_passages)
                encoded = self.tokenizer(
                    batch_queries,
                    list(batch_passages),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits

                if logits.ndim == 1:
                    batch_scores = logits
                elif logits.shape[-1] == 1:
                    batch_scores = logits.squeeze(-1)
                else:
                    # Most binary sequence-classification rerankers place the
                    # positive/relevance logit in the last column.
                    batch_scores = logits[:, -1]

                scores.extend(batch_scores.detach().float().cpu().tolist())
        return scores


def rerank_docs(
    item: dict[str, Any],
    reranker: CrossEncoderReranker,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], float | None]:
    if args.question_field not in item:
        raise KeyError(f"Missing question field `{args.question_field}` in item.")
    if args.docs_field not in item or not isinstance(item[args.docs_field], list):
        raise KeyError(f"Missing docs list field `{args.docs_field}` in item.")

    question = str(item[args.question_field])
    docs = copy.deepcopy(item[args.docs_field])
    if not docs:
        updated = copy.deepcopy(item)
        updated[args.docs_field] = docs
        return updated, None

    candidate_topn = len(docs) if args.candidate_topn is None else min(args.candidate_topn, len(docs))
    rerank_candidates = docs[:candidate_topn]
    untouched_tail = docs[candidate_topn:]

    passages = [
        build_document_text(doc, args.title_field, args.text_field)
        for doc in rerank_candidates
    ]
    scores = reranker.score_pairs(question, passages, args.batch_size)

    ranked = sorted(
        enumerate(zip(rerank_candidates, scores)),
        key=lambda pair: pair[1][1],
        reverse=True,
    )

    reranked_docs: list[dict[str, Any]] = []
    for new_rank, (original_rank, (doc, score)) in enumerate(ranked):
        reranked_doc = copy.deepcopy(doc)
        if args.keep_scores:
            reranked_doc["_rerank_score"] = float(score)
            reranked_doc["_original_rank"] = int(original_rank)
            reranked_doc["_new_rank"] = int(new_rank)
        reranked_docs.append(reranked_doc)
    reranked_docs.extend(untouched_tail)

    updated = copy.deepcopy(item)
    updated[args.docs_field] = reranked_docs

    overlap = compute_topk_overlap(candidate_topn, ranked, args.report_topk)
    return updated, overlap


def compute_topk_overlap(
    candidate_topn: int,
    ranked: Sequence[tuple[int, tuple[dict[str, Any], float]]],
    report_topk: int,
) -> float | None:
    if report_topk <= 0 or candidate_topn <= 0:
        return None
    k = min(report_topk, candidate_topn, len(ranked))
    original_topk = set(range(k))
    reranked_topk = {original_rank for original_rank, _ in ranked[:k]}
    return len(original_topk & reranked_topk) / k


def summarize_scores(
    overlaps: Sequence[float],
    processed_items: int,
    docs_per_item: Sequence[int],
) -> None:
    print(f"Processed items: {processed_items}")
    if docs_per_item:
        print(f"Average docs per item: {statistics.mean(docs_per_item):.2f}")
    if overlaps:
        mean_overlap = statistics.mean(overlaps)
        print(f"Average original-vs-reranked top-k overlap: {mean_overlap:.4f}")


def main() -> None:
    args = parse_args()
    try:
        import torch
        from tqdm import tqdm
    except ImportError as exc:
        raise ImportError(
            "Reranking requires `torch` and `tqdm`. "
            "Install them in the same environment that runs ALCE."
        ) from exc

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.batch_size <= 0:
        raise ValueError("`--batch-size` must be positive.")
    if args.max_length <= 0:
        raise ValueError("`--max-length` must be positive.")
    if args.candidate_topn is not None and args.candidate_topn <= 0:
        raise ValueError("`--candidate-topn` must be positive when provided.")

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    items, wrapped_in_data = load_items(payload)
    items_to_process = items if args.limit is None else items[: args.limit]

    reranker = CrossEncoderReranker(
        model_name=args.model,
        device=args.device,
        max_length=args.max_length,
    )

    reranked_items: list[dict[str, Any]] = []
    overlaps: list[float] = []
    docs_per_item: list[int] = []

    for item in tqdm(items_to_process, desc="Reranking ALCE docs"):
        reranked_item, overlap = rerank_docs(item, reranker, args)
        reranked_items.append(reranked_item)
        docs_per_item.append(len(reranked_item.get(args.docs_field, [])))
        if overlap is not None and not math.isnan(overlap):
            overlaps.append(overlap)

    if args.limit is not None and len(items) > args.limit:
        reranked_items.extend(copy.deepcopy(items[args.limit :]))

    output_payload = restore_payload(payload, reranked_items, wrapped_in_data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summarize_scores(overlaps, len(items_to_process), docs_per_item)
    print(f"Wrote reranked file to: {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a blinded human-evaluation pack for base RAG vs COVER-RAG.

The output consists of:

* annotation CSV: given to annotators, with system names hidden as A/B
* annotation JSONL: same examples with fuller metadata and cited snippets
* key CSV: private mapping from A/B back to base/cover
* README: short annotation instructions

The annotator fills the blank score columns in the annotation CSV.  Use
analyze_cover_human_eval.py afterwards to unblind and summarize the results.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any


DEFAULT_DATASETS = {
    "asqa": (
        "result/origin/asqa-gpt-4o-mini-gtr-shot2-ndoc5-42.json",
        "result/cover_v3_recall_completion_v13_integrated_explanation_full/cover_v3/asqa-gpt-4o-mini-gtr-shot2-ndoc5-42.cover_v3.json",
    ),
    "eli5": (
        "result/origin/eli5-gpt-4o-mini-bm25-shot2-ndoc5-42.json",
        "result/cover_v3_recall_completion_v13_integrated_explanation_full/cover_v3/eli5-gpt-4o-mini-bm25-shot2-ndoc5-42.cover_v3.json",
    ),
    "qampari": (
        "result/origin/qampari-gpt-4o-mini-gtr-shot2-ndoc5-42.json",
        "result/cover_v3_recall_completion_v13_integrated_explanation_full/cover_v3/qampari-gpt-4o-mini-gtr-shot2-ndoc5-42.cover_v3.json",
    ),
}

ANNOTATION_FIELDS = [
    "eval_id",
    "dataset",
    "sample_id",
    "question",
    "reference_answer",
    "system_a_answer",
    "system_a_cited_evidence",
    "system_b_answer",
    "system_b_cited_evidence",
    "correctness_a_1_to_5",
    "correctness_b_1_to_5",
    "citation_faithfulness_a_1_to_5",
    "citation_faithfulness_b_1_to_5",
    "unsupported_claims_a_count",
    "unsupported_claims_b_count",
    "fluency_a_1_to_5",
    "fluency_b_1_to_5",
    "preference_a_b_tie",
    "notes",
]

KEY_FIELDS = [
    "eval_id",
    "dataset",
    "sample_id",
    "system_a",
    "system_b",
    "base_path",
    "cover_path",
]


def root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported result JSON shape: {path}")


def item_id(item: dict[str, Any], fallback: int) -> str:
    for key in ("sample_id", "id", "question_id"):
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return str(fallback)


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def answer_text(item: dict[str, Any]) -> str:
    return normalize_space(item.get("output") or item.get("answer") or "")


def reference_answer(item: dict[str, Any]) -> str:
    answer = item.get("answer")
    if answer not in (None, ""):
        return normalize_space(answer)
    annotations = item.get("annotations")
    if isinstance(annotations, list) and annotations:
        return normalize_space(annotations[0])
    return ""


def citation_ids(answer: str) -> list[int]:
    ids: list[int] = []
    for match in re.findall(r"\[(\d+)\]", answer or ""):
        try:
            value = int(match)
        except Exception:
            continue
        if value > 0 and value not in ids:
            ids.append(value)
    return ids


def doc_text(doc: dict[str, Any]) -> str:
    title = normalize_space(doc.get("title") or doc.get("id") or "")
    text = normalize_space(doc.get("text") or doc.get("contents") or doc.get("content") or "")
    if title and text:
        return f"{title}: {text}"
    return title or text


def cited_evidence(item: dict[str, Any], answer: str, max_chars_per_doc: int) -> str:
    docs = item.get("docs") or item.get("ctxs") or item.get("documents") or []
    if not isinstance(docs, list):
        return ""
    parts: list[str] = []
    for cid in citation_ids(answer):
        index = cid - 1
        if 0 <= index < len(docs) and isinstance(docs[index], dict):
            snippet = doc_text(docs[index])[:max_chars_per_doc]
            parts.append(f"[{cid}] {snippet}")
    return "\n\n".join(parts)


def split_counts(total: int, datasets: list[str]) -> dict[str, int]:
    base = total // len(datasets)
    rem = total % len(datasets)
    return {dataset: base + (1 if i < rem else 0) for i, dataset in enumerate(datasets)}


def parse_dataset_spec(value: str) -> tuple[str, str, str]:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--dataset must use name:base_json:cover_json")
    return parts[0].strip().lower(), parts[1], parts[2]


def build_examples(args: argparse.Namespace, root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rng = random.Random(args.seed)
    dataset_specs = args.dataset or [
        f"{name}:{base}:{cover}" for name, (base, cover) in DEFAULT_DATASETS.items()
    ]
    parsed_specs = [parse_dataset_spec(spec) for spec in dataset_specs]
    counts = split_counts(args.num_examples, [name for name, _, _ in parsed_specs])

    examples: list[dict[str, Any]] = []
    keys: list[dict[str, str]] = []
    eval_index = 1
    for dataset, base_text, cover_text in parsed_specs:
        base_path = resolve(root, base_text)
        cover_path = resolve(root, cover_text)
        base_items = load_items(base_path)
        cover_items = load_items(cover_path)
        base_by_id = {item_id(item, i): item for i, item in enumerate(base_items)}
        cover_by_id = {item_id(item, i): item for i, item in enumerate(cover_items)}
        shared_ids = sorted(set(base_by_id) & set(cover_by_id))
        rng.shuffle(shared_ids)
        wanted = min(counts[dataset], len(shared_ids))

        for sid in shared_ids[:wanted]:
            base_item = base_by_id[sid]
            cover_item = cover_by_id[sid]
            systems = [
                ("base", base_item, answer_text(base_item)),
                ("cover", cover_item, answer_text(cover_item)),
            ]
            rng.shuffle(systems)
            a_name, a_item, a_answer = systems[0]
            b_name, b_item, b_answer = systems[1]
            eval_id = f"{dataset}-{eval_index:04d}"
            question = normalize_space(base_item.get("question") or cover_item.get("question") or "")
            reference = reference_answer(base_item) or reference_answer(cover_item)
            row = {
                "eval_id": eval_id,
                "dataset": dataset,
                "sample_id": sid,
                "question": question,
                "reference_answer": reference,
                "system_a_answer": a_answer,
                "system_a_cited_evidence": cited_evidence(a_item, a_answer, args.max_chars_per_doc),
                "system_b_answer": b_answer,
                "system_b_cited_evidence": cited_evidence(b_item, b_answer, args.max_chars_per_doc),
                "correctness_a_1_to_5": "",
                "correctness_b_1_to_5": "",
                "citation_faithfulness_a_1_to_5": "",
                "citation_faithfulness_b_1_to_5": "",
                "unsupported_claims_a_count": "",
                "unsupported_claims_b_count": "",
                "fluency_a_1_to_5": "",
                "fluency_b_1_to_5": "",
                "preference_a_b_tie": "",
                "notes": "",
            }
            examples.append(row)
            keys.append(
                {
                    "eval_id": eval_id,
                    "dataset": dataset,
                    "sample_id": sid,
                    "system_a": a_name,
                    "system_b": b_name,
                    "base_path": str(base_path),
                    "cover_path": str(cover_path),
                }
            )
            eval_index += 1
    return examples, keys


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_readme(path: Path, num_examples: int) -> None:
    path.write_text(
        f"""# COVER-RAG Human Evaluation Pack

This pack contains {num_examples} blinded pairwise examples comparing a base
RAG answer against the corresponding COVER-RAG answer. Annotators should use
`annotation_sheet.csv`; keep `blind_key.csv` hidden until analysis.

## Scores

- `correctness_*_1_to_5`: 1 = wrong/unhelpful, 3 = partially correct, 5 = fully correct and complete.
- `citation_faithfulness_*_1_to_5`: 1 = citations do not support most claims, 3 = mixed/partial support, 5 = cited evidence supports the factual claims.
- `unsupported_claims_*_count`: approximate number of factual claims not supported by the cited evidence.
- `fluency_*_1_to_5`: 1 = hard to read, 5 = clear and fluent.
- `preference_a_b_tie`: choose `A`, `B`, or `Tie` based on overall usefulness and trustworthiness.

## Recommended Protocol

Use at least two annotators for a subset of examples if possible. Annotators
should not open `blind_key.csv`. After annotation, run:

```bash
python scripts/analyze_cover_human_eval.py \\
  --annotations paper/human_eval/cover_v13_pairwise/annotation_sheet.csv \\
  --key paper/human_eval/cover_v13_pairwise/blind_key.csv \\
  --output-dir paper/human_eval/cover_v13_pairwise/results
```
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-chars-per-doc", type=int, default=900)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset spec: name:base_json:cover_json. Defaults to vanilla v13 ASQA/ELI5/QAMPARI.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/human_eval/cover_v13_pairwise"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = root_dir()
    output_dir = resolve(root, args.output_dir)
    examples, keys = build_examples(args, root)
    write_csv(output_dir / "annotation_sheet.csv", ANNOTATION_FIELDS, examples)
    write_jsonl(output_dir / "annotation_sheet.jsonl", examples)
    write_csv(output_dir / "blind_key.csv", KEY_FIELDS, keys)
    write_readme(output_dir / "README.md", len(examples))
    print(f"Wrote {len(examples)} examples to {output_dir}")


if __name__ == "__main__":
    main()

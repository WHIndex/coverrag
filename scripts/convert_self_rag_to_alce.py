#!/usr/bin/env python3
"""Convert Self-RAG generations to the ALCE result JSON format.

The original Self-RAG code in methods/self-rag can emit slightly different
formats depending on the script:

* run_long_form_static.py writes an ALCE-like {"data": [...]} JSON for ASQA/ELI5.
* run_short_form.py and run_baseline_lm.py write JSONL rows with an "output".
* Some local runs preserve retrieval fields as "docs", "ctxs", or "top_contexts".

This adapter keeps the gold/task metadata from a reference ALCE result file and
replaces only the generated answer and, when available, the retrieved docs. The
result can then be used as the raw v0 input for COVER-RAG.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any


CONTROL_TOKENS = [
    "[Fully supported]",
    "[Partially supported]",
    "[No support / Contradictory]",
    "[No Retrieval]",
    "[Retrieval]",
    "[Continue to Use Evidence]",
    "[Irrelevant]",
    "[Relevant]",
    "[Utility:1]",
    "[Utility:2]",
    "[Utility:3]",
    "[Utility:4]",
    "[Utility:5]",
    "<paragraph>",
    "</paragraph>",
    "<s>",
    "</s>",
    "[PAD]",
    "<unk>",
]


def load_json_or_jsonl(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    return json.loads(path.read_text(encoding="utf-8"))


def data_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    if isinstance(payload, list):
        return payload
    raise SystemExit("Input must be a JSON list, JSONL file, or JSON object with a data list.")


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, dict):
        for key in ("0", 0, "text", "output", "prediction", "final_prediction"):
            if key in text:
                return clean_text(text[key])
        for value in text.values():
            cleaned = clean_text(value)
            if cleaned:
                return cleaned
        return ""
    if isinstance(text, list):
        return " ".join(clean_text(value) for value in text if clean_text(value)).strip()
    out = str(text)
    for token in CONTROL_TOKENS:
        out = out.replace(token, "")
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return out.strip()


def output_from_item(item: dict[str, Any]) -> str:
    for key in (
        "output",
        "final_output",
        "prediction",
        "pred",
        "response",
        "generated_text",
        "final_prediction",
    ):
        value = item.get(key)
        cleaned = clean_text(value)
        if cleaned:
            return cleaned
    return ""


def normalize_doc(doc: Any, fallback_id: int) -> dict[str, Any] | None:
    if doc is None:
        return None
    if isinstance(doc, str):
        text = doc.strip()
        if not text:
            return None
        title = ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) >= 2 and len(lines[0].split()) <= 16:
            title = lines[0]
            text = " ".join(lines[1:])
        return {"id": str(fallback_id), "title": title, "text": text}
    if not isinstance(doc, dict):
        return {"id": str(fallback_id), "title": "", "text": clean_text(doc)}

    title = clean_text(
        doc.get("title")
        or doc.get("wikipage")
        or doc.get("page_title")
        or doc.get("name")
        or ""
    )
    text = clean_text(
        doc.get("text")
        or doc.get("contents")
        or doc.get("content")
        or doc.get("passage")
        or doc.get("paragraph")
        or doc.get("body")
        or ""
    )
    if not text and title:
        text = title
        title = ""
    if not text:
        return None
    out = dict(doc)
    out["id"] = str(doc.get("id", doc.get("doc_id", fallback_id)))
    out["title"] = title
    out["text"] = text
    return out


def docs_from_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    raw_docs: Any = None
    for key in ("docs", "ctxs", "top_contexts", "contexts", "retrieved_docs", "retrieval_docs"):
        value = item.get(key)
        if value:
            raw_docs = value
            break
    if isinstance(raw_docs, dict):
        if all(isinstance(key, str) and key.isdigit() for key in raw_docs):
            raw_docs = [raw_docs[key] for key in sorted(raw_docs, key=lambda value: int(value))]
        else:
            raw_docs = list(raw_docs.values())
    if raw_docs is None:
        return []
    if not isinstance(raw_docs, list):
        raw_docs = [raw_docs]
    docs = []
    for idx, doc in enumerate(raw_docs, start=1):
        normalized = normalize_doc(doc, idx)
        if normalized is not None:
            docs.append(normalized)
    return docs


def citation_ids(text: str) -> list[int]:
    ids = []
    for match in re.findall(r"\[(\d+)\]", text):
        try:
            ids.append(int(match))
        except ValueError:
            continue
    return ids


def shift_zero_based_citations(text: str, mode: str) -> str:
    if mode == "never":
        return text
    ids = citation_ids(text)
    if not ids:
        return text
    should_shift = mode == "always" or (mode == "auto" and 0 in ids)
    if not should_shift:
        return text

    def replace(match: re.Match[str]) -> str:
        return f"[{int(match.group(1)) + 1}]"

    return re.sub(r"\[(\d+)\]", replace, text)


def item_keys(item: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for key in ("sample_id", "id", "qid", "question_id"):
        value = item.get(key)
        if value not in (None, ""):
            keys.append(f"{key}:{value}")
    question = clean_text(item.get("question") or item.get("instruction") or item.get("input"))
    if question:
        keys.append(f"question:{question.lower()}")
    return keys


def build_reference_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key in item_keys(row):
            index.setdefault(key, row)
    return index


def match_reference(
    item: dict[str, Any],
    index: dict[str, dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    idx: int,
) -> dict[str, Any]:
    for key in item_keys(item):
        if key in index:
            return index[key]
    if idx < len(reference_rows):
        return reference_rows[idx]
    return {}


def convert(args: argparse.Namespace) -> dict[str, Any]:
    self_payload = load_json_or_jsonl(args.self_rag_output)
    self_rows = data_rows(self_payload)
    reference_payload: Any = {}
    reference_rows: list[dict[str, Any]] = []
    if args.reference:
        reference_payload = load_json_or_jsonl(args.reference)
        reference_rows = data_rows(reference_payload)
    reference_index = build_reference_index(reference_rows)

    output_rows: list[dict[str, Any]] = []
    missing_outputs = 0
    used_self_docs = 0
    shifted_citation_rows = 0
    for idx, self_item in enumerate(self_rows):
        ref_item = match_reference(self_item, reference_index, reference_rows, idx)
        merged = copy.deepcopy(ref_item) if ref_item else copy.deepcopy(self_item)
        output = output_from_item(self_item)
        if not output:
            missing_outputs += 1
            output = output_from_item(ref_item) if ref_item else ""
        before_ids = citation_ids(output)
        output = shift_zero_based_citations(output, args.shift_zero_based_citations)
        if citation_ids(output) != before_ids:
            shifted_citation_rows += 1
        merged["output"] = output

        self_docs = docs_from_item(self_item)
        if self_docs and not args.keep_reference_docs:
            merged["docs"] = self_docs
            used_self_docs += 1
        elif "docs" not in merged:
            merged["docs"] = self_docs

        if args.dataset:
            merged.setdefault("dataset", args.dataset)
        output_rows.append(merged)

    if isinstance(reference_payload, dict):
        out_payload = {key: value for key, value in reference_payload.items() if key != "data"}
    elif isinstance(self_payload, dict):
        out_payload = {key: value for key, value in self_payload.items() if key != "data"}
    else:
        out_payload = {}
    out_payload["data"] = output_rows
    out_payload.setdefault("args", [])
    out_payload["self_rag_adapter"] = {
        "source": str(args.self_rag_output),
        "reference": str(args.reference) if args.reference else "",
        "dataset": args.dataset,
        "num_examples": len(output_rows),
        "missing_outputs": missing_outputs,
        "rows_using_self_rag_docs": used_self_docs,
        "rows_shifted_zero_based_citations": shifted_citation_rows,
        "keep_reference_docs": args.keep_reference_docs,
    }
    return out_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Self-RAG output to ALCE result JSON.")
    parser.add_argument("--self-rag-output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None, help="Reference ALCE result JSON with gold metadata.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("asqa", "eli5", "qampari"), default="")
    parser.add_argument(
        "--shift-zero-based-citations",
        choices=("auto", "always", "never"),
        default="auto",
        help="Self-RAG long-form output uses [0]-based citations; ALCE expects [1]-based citations.",
    )
    parser.add_argument(
        "--keep-reference-docs",
        action="store_true",
        help="Keep reference ALCE docs even if Self-RAG output contains docs.",
    )
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = convert(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None),
        encoding="utf-8",
    )
    meta = payload.get("self_rag_adapter", {})
    print(
        "Wrote {path} ({n} examples, self-doc rows={docs}, shifted citation rows={shifted}, missing outputs={missing})".format(
            path=args.output,
            n=meta.get("num_examples"),
            docs=meta.get("rows_using_self_rag_docs"),
            shifted=meta.get("rows_shifted_zero_based_citations"),
            missing=meta.get("missing_outputs"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

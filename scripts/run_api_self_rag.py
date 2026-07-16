#!/usr/bin/env python3
"""Generate API-based Self-RAG-style ALCE outputs.

This is not the original selfrag/selfrag_llama2 checkpoint inference path. The
paper source uses vLLM to run a local model that emits reflection tokens. This
script implements the same high-level method shape with an OpenAI-compatible
chat model:

1. read a question and retrieved documents,
2. ask the model to internally select relevant evidence and self-check support,
3. emit an answer with ALCE bracket citations.

The output is an ALCE result JSON and can be used as a base-method result for
COVER-RAG plug-in experiments.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

from cover_posthoc_audit import (
    JsonCache,
    OpenAIChatClient,
    extract_json_object,
    load_payload,
    normalize_space,
    sent_tokenize_safe,
    stable_hash,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("api_self_rag")


SYSTEM = """You are a self-reflective retrieval-augmented QA system.
Use only the provided documents. Internally decide which documents are relevant,
check whether each answer sentence is supported, and then output a concise
citation-grounded answer. Do not reveal the internal reasoning."""


USER_TEMPLATE = """Dataset/task: {dataset}

Question:
{question}

Retrieved documents:
{docs}

Return strict JSON only:
{{
  "need_retrieval": true,
  "selected_doc_ids": [1, 2],
  "answer": "final answer with bracket citations"
}}

Rules:
1. Use only the retrieved documents. Do not use outside knowledge.
2. Cite every factual sentence or list item with one or two bracket citations like [1].
3. Citation ids must refer to the numbered documents above.
4. Prefer directly relevant answer units over background.
5. If the evidence is incomplete, give the best supported partial answer rather than inventing.
6. Do not say the documents are insufficient if at least one useful supported answer unit is present.
7. Keep the answer compact.
{dataset_rules}
"""


DATASET_RULES = {
    "asqa": (
        "ASQA rule: answer ambiguous questions by covering multiple valid interpretations "
        "when the evidence supports them. Use 1 short paragraph."
    ),
    "eli5": (
        "ELI5 rule: explain in simple words, but keep claims grounded in the documents. "
        "Use 1 compact paragraph."
    ),
    "qampari": (
        "QAMPARI rule: output only a comma-separated list of answer entities, each with "
        "a citation, e.g. Heat [2], Sanctuary [4]. Do not write a full explanatory sentence."
    ),
}


def doc_text(doc: dict[str, Any]) -> str:
    title = normalize_space(str(doc.get("title") or doc.get("wikipage") or ""))
    text = normalize_space(
        str(doc.get("text") or doc.get("contents") or doc.get("content") or doc.get("passage") or "")
    )
    if title and text:
        return f"{title}\n{text}"
    return title or text


def format_docs(docs: Sequence[dict[str, Any]], max_doc_chars: int) -> str:
    lines = []
    for idx, doc in enumerate(docs, start=1):
        text = doc_text(doc)
        if len(text) > max_doc_chars:
            text = text[:max_doc_chars].rsplit(" ", 1)[0] + " ..."
        lines.append(f"[{idx}] {text}")
    return "\n\n".join(lines)


def infer_dataset(path: Path, payload: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lowered = path.name.lower()
    for name in ("asqa", "eli5", "qampari"):
        if name in lowered:
            return name
    if rows:
        first = rows[0]
        if "qa_pairs" in first:
            return "asqa"
        if "claims" in first:
            return "eli5"
        if "answers" in first:
            return "qampari"
    args = payload.get("args", {}) if isinstance(payload, dict) else {}
    if isinstance(args, dict):
        for key in ("dataset_name", "dataset", "data_name"):
            value = str(args.get(key) or "").lower()
            for name in ("asqa", "eli5", "qampari"):
                if name in value:
                    return name
    return ""


class CandidateDocProvider:
    def __init__(self, path: Path | None) -> None:
        self.by_sample_id: dict[str, list[dict[str, Any]]] = {}
        self.by_question: dict[str, list[dict[str, Any]]] = {}
        self.by_index: dict[int, list[dict[str, Any]]] = {}
        if path is not None and path.exists():
            _payload, records = load_payload(path)
            for idx, record in enumerate(records):
                docs = list(record.get("docs", []) or [])
                if not docs:
                    continue
                self.by_index[idx] = docs
                if record.get("sample_id") is not None:
                    self.by_sample_id[str(record["sample_id"])] = docs
                if record.get("id") is not None:
                    self.by_sample_id[str(record["id"])] = docs
                question = normalize_space(str(record.get("question", "") or "")).lower()
                if question:
                    self.by_question[question] = docs
            logger.info("Loaded candidate docs from %s", path)
        elif path is not None:
            logger.warning("Candidate docs file not found: %s; using docs in input JSON.", path)

    def docs_for(self, item: dict[str, Any], idx: int) -> list[dict[str, Any]]:
        for key in ("sample_id", "id"):
            value = item.get(key)
            if value is not None and str(value) in self.by_sample_id:
                return self.by_sample_id[str(value)]
        question = normalize_space(str(item.get("question", "") or "")).lower()
        if question and question in self.by_question:
            return self.by_question[question]
        return self.by_index.get(idx, [])


def valid_citation_ids(text: str, n_docs: int) -> list[int]:
    ids: list[int] = []
    for match in re.findall(r"\[(\d+)\]", text):
        value = int(match)
        if 1 <= value <= n_docs and value not in ids:
            ids.append(value)
    return ids


def append_missing_citations(answer: str, default_ids: Sequence[int], dataset: str, n_docs: int) -> str:
    default_ids = [value for value in default_ids if 1 <= int(value) <= n_docs][:2]
    if not default_ids:
        return answer
    cite = "".join(f"[{value}]" for value in default_ids)
    if dataset == "qampari":
        pieces = [piece.strip() for piece in answer.split(",") if piece.strip()]
        fixed = []
        for piece in pieces:
            fixed.append(piece if valid_citation_ids(piece, n_docs) else f"{piece} {cite}")
        return ", ".join(fixed)

    sentences = sent_tokenize_safe(answer)
    if not sentences:
        return answer
    fixed = []
    for sentence in sentences:
        stripped = sentence.strip()
        if not stripped:
            continue
        if valid_citation_ids(stripped, n_docs):
            fixed.append(stripped)
        elif re.search(r"\b(cannot|insufficient|not enough|no information)\b", stripped, re.I):
            fixed.append(stripped)
        else:
            fixed.append(f"{stripped.rstrip('.')} {cite}.")
    return " ".join(fixed)


def parse_selected_ids(value: Any, n_docs: int) -> list[int]:
    if not isinstance(value, list):
        return []
    ids = []
    for raw in value:
        try:
            doc_id = int(raw)
        except Exception:
            continue
        if 1 <= doc_id <= n_docs and doc_id not in ids:
            ids.append(doc_id)
    return ids


def generate_one(
    idx: int,
    item: dict[str, Any],
    dataset: str,
    docs: list[dict[str, Any]],
    client: OpenAIChatClient,
    cache: JsonCache,
    args: argparse.Namespace,
) -> tuple[int, dict[str, Any], dict[str, int]]:
    docs = docs[: args.max_docs]
    question = normalize_space(str(item.get("question", "") or item.get("instruction", "") or ""))
    prompt = USER_TEMPLATE.format(
        dataset=dataset,
        question=question,
        docs=format_docs(docs, args.max_doc_chars),
        dataset_rules=DATASET_RULES.get(dataset, ""),
    )
    key = "api_self_rag:" + stable_hash(dataset, question, prompt, args.model)
    cached = cache.get(key)
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    if cached is None:
        try:
            raw, usage = client.chat(SYSTEM, prompt, max_tokens=args.max_tokens)
            try:
                parsed = extract_json_object(raw)
            except Exception:
                parsed = {"answer": raw, "selected_doc_ids": []}
            cache.set(key, parsed)
        except Exception as exc:
            logger.warning("API Self-RAG failed for example %s; falling back to existing output. Error: %s", idx, exc)
            parsed = {"answer": item.get("output", ""), "selected_doc_ids": []}
            cache.set(key, parsed)
    else:
        parsed = cached

    answer = normalize_space(str(parsed.get("answer", "") or item.get("output", "") or ""))
    selected_ids = parse_selected_ids(parsed.get("selected_doc_ids"), len(docs))
    if not selected_ids:
        selected_ids = valid_citation_ids(answer, len(docs))[:2]
    if not selected_ids and docs:
        selected_ids = [1]
    answer = append_missing_citations(answer, selected_ids, dataset, len(docs))

    out = copy.deepcopy(item)
    out["docs"] = docs
    out["output"] = answer
    out["api_self_rag"] = {
        "model": args.model,
        "selected_doc_ids": selected_ids,
        "need_retrieval": bool(parsed.get("need_retrieval", True)),
    }
    return idx, out, usage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run API-based Self-RAG-style generation on ALCE data.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-docs-file", type=Path, default=None)
    parser.add_argument("--dataset", choices=("asqa", "eli5", "qampari"), default="")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=450)
    parser.add_argument("--max-docs", type=int, default=8)
    parser.add_argument("--max-doc-chars", type=int, default=900)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache-file", type=Path, default=Path("cache/api_self_rag_cache.json"))
    parser.add_argument("--cache-save-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload, rows = load_payload(args.input)
    dataset = args.dataset or infer_dataset(args.input, payload, rows)
    if not dataset:
        raise SystemExit("Could not infer dataset; pass --dataset.")
    if args.limit is not None:
        rows = rows[: args.limit]

    provider = CandidateDocProvider(args.candidate_docs_file)
    cache = JsonCache(args.cache_file, save_every=args.cache_save_every)
    client = OpenAIChatClient(args.model, args.temperature, args.top_p, args.max_retries)

    out_rows: list[dict[str, Any] | None] = [None] * len(rows)
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    logger.info("Generating %d %s examples with model=%s workers=%d", len(rows), dataset, args.model, args.workers)

    def submit(idx: int, item: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, int]]:
        docs = provider.docs_for(item, idx) or list(item.get("docs", []) or [])
        return generate_one(idx, item, dataset, docs, client, cache, args)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(submit, idx, item) for idx, item in enumerate(rows)]
        for done_count, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="API Self-RAG"), start=1):
            idx, out, usage = future.result()
            out_rows[idx] = out
            total_usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            total_usage["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            if args.checkpoint_every and done_count % args.checkpoint_every == 0:
                write_payload(args.output.with_suffix(args.output.suffix + ".checkpoint.json"), payload, out_rows, args, dataset, total_usage)
                cache.flush()

    final_rows = [row for row in out_rows if row is not None]
    write_payload(args.output, payload, final_rows, args, dataset, total_usage)
    cache.flush()
    logger.info("Wrote %s", args.output)
    return 0


def write_payload(
    path: Path,
    input_payload: dict[str, Any],
    rows: Sequence[dict[str, Any] | None],
    args: argparse.Namespace,
    dataset: str,
    total_usage: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_rows = [row for row in rows if row is not None]
    payload = {key: value for key, value in input_payload.items() if key != "data"} if isinstance(input_payload, dict) else {}
    payload["data"] = clean_rows
    payload["args"] = {
        "method": "api_self_rag_style",
        "dataset": dataset,
        "model": args.model,
        "candidate_docs_file": str(args.candidate_docs_file or ""),
        "max_docs": args.max_docs,
        "max_doc_chars": args.max_doc_chars,
    }
    payload["total_cost"] = input_payload.get("total_cost", 0.0) if isinstance(input_payload, dict) else 0.0
    payload["api_self_rag_summary"] = {
        "num_examples": len(clean_rows),
        "prompt_tokens": total_usage.get("prompt_tokens", 0),
        "completion_tokens": total_usage.get("completion_tokens", 0),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

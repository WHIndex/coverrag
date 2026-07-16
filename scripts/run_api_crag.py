#!/usr/bin/env python3
"""Generate API-based CRAG-style ALCE outputs.

The original CRAG repository uses a trained retrieval evaluator, optional web
search, and a local generator.  For the ALCE plug-in experiments here we keep
the CRAG method shape but use an OpenAI-compatible chat model:

1. evaluate retrieved document quality for the question,
2. decide whether to use, filter, supplement, or combine retrieved evidence,
3. refine selected documents into focused evidence snippets,
4. generate a citation-grounded answer from those corrected documents.

The output is an ALCE result JSON and can be used as the raw v0 input for
COVER-RAG.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
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
logger = logging.getLogger("api_crag")


DATASET_RULES = {
    "asqa": (
        "ASQA: answer the ambiguous question by covering multiple valid interpretations "
        "when the corrected evidence supports them. Use one concise paragraph."
    ),
    "eli5": (
        "ELI5: explain in simple words, but keep every factual statement grounded in "
        "the corrected evidence. Use one compact paragraph."
    ),
    "qampari": (
        "QAMPARI: output only a comma-separated list of answer entities, each with "
        "a citation, e.g. Heat [2], Sanctuary [4]. Do not add explanations."
    ),
}


CRAG_EVALUATOR_SYSTEM = """You are CRAG's retrieval evaluator.
Assess whether the retrieved documents contain enough useful evidence to answer
the question. Return strict JSON only."""


CRAG_EVALUATOR_USER = """Question:
{question}

Retrieved documents:
{docs}

Choose one action:
- use: documents already contain enough directly relevant evidence.
- filter: some documents are useful but noisy; select only the relevant ones.
- supplement: the shown documents are weak; search the wider candidate pool.
- combine: some shown documents are useful, but the answer also needs more evidence.

Return JSON with exactly this schema:
{{
  "action": "use|filter|supplement|combine",
  "selected_doc_ids": [1, 2],
  "confidence": 0.0,
  "reason": "short reason"
}}

Rules:
1. Select only document ids that directly help answer the question.
2. Prefer precision over quantity.
3. If no shown document is useful, use "supplement" and selected_doc_ids can be empty.
4. Do not answer the question here."""


CRAG_GENERATOR_SYSTEM = """You are a corrective retrieval-augmented QA system.
Use only the corrected evidence documents. Write the final answer with ALCE
bracket citations. Do not reveal document-quality analysis."""


CRAG_GENERATOR_USER = """Dataset/task: {dataset}

Question:
{question}

Corrected evidence documents:
{docs}

Return strict JSON only:
{{
  "answer": "final answer with bracket citations"
}}

Rules:
1. Use only the corrected evidence documents.
2. Cite every factual sentence or list item with one or two bracket citations like [1].
3. Citation ids must refer to the numbered corrected evidence documents above.
4. Prefer directly relevant answer units over background.
5. If evidence is incomplete, give the best supported partial answer rather than inventing.
6. Do not say the documents are insufficient if at least one useful supported answer unit is present.
7. Keep the answer compact.
{dataset_rules}
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
    "when",
    "where",
    "how",
    "why",
    "did",
    "does",
    "do",
    "can",
    "could",
    "would",
    "should",
    "has",
    "have",
    "had",
    "into",
    "about",
    "than",
    "then",
    "their",
    "his",
    "her",
    "its",
    "they",
    "them",
}


def content_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return {token for token in tokens if len(token) > 1 and token not in STOPWORDS}


def token_overlap(query: str, text: str) -> float:
    query_tokens = content_tokens(query)
    if not query_tokens:
        return 0.0
    text_tokens = content_tokens(text)
    if not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def doc_text(doc: dict[str, Any]) -> str:
    title = normalize_space(str(doc.get("title") or doc.get("wikipage") or doc.get("page_title") or ""))
    text = normalize_space(
        str(doc.get("text") or doc.get("contents") or doc.get("content") or doc.get("passage") or "")
    )
    if title and text:
        return f"{title}\n{text}"
    return title or text


def short_doc_text(doc: dict[str, Any], max_chars: int) -> str:
    title = normalize_space(str(doc.get("title") or doc.get("wikipage") or doc.get("page_title") or ""))
    extraction = normalize_space(str(doc.get("extraction") or ""))
    summary = normalize_space(str(doc.get("summary") or ""))
    body = normalize_space(str(doc.get("text") or doc.get("contents") or doc.get("content") or doc.get("passage") or ""))
    pieces = []
    if title:
        pieces.append(f"Title: {title}")
    if extraction and extraction.lower() != "irrelevant.":
        pieces.append(f"Key: {extraction}")
    if summary and summary.lower() != "irrelevant.":
        pieces.append(f"Summary: {summary}")
    if body:
        pieces.append(body)
    text = "\n".join(pieces)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + " ..."
    return text


def format_docs(docs: Sequence[dict[str, Any]], max_chars: int) -> str:
    lines = []
    for idx, doc in enumerate(docs, start=1):
        text = short_doc_text(doc, max_chars)
        if text:
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
                for key in ("sample_id", "id", "qid", "question_id"):
                    if record.get(key) is not None:
                        self.by_sample_id[str(record[key])] = docs
                question = normalize_space(str(record.get("question", "") or "")).lower()
                if question:
                    self.by_question[question] = docs
            logger.info("Loaded candidate docs from %s", path)
        elif path is not None:
            logger.warning("Candidate docs file not found: %s; using docs in input JSON.", path)

    def docs_for(self, item: dict[str, Any], idx: int) -> list[dict[str, Any]]:
        for key in ("sample_id", "id", "qid", "question_id"):
            value = item.get(key)
            if value is not None and str(value) in self.by_sample_id:
                return self.by_sample_id[str(value)]
        question = normalize_space(str(item.get("question", "") or item.get("instruction", "") or "")).lower()
        if question and question in self.by_question:
            return self.by_question[question]
        return self.by_index.get(idx, [])


def retrieval_score(doc: dict[str, Any]) -> float:
    raw = doc.get("score")
    try:
        value = float(raw)
    except Exception:
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def rank_docs(question: str, docs: Sequence[dict[str, Any]]) -> list[tuple[float, int, dict[str, Any]]]:
    ranked = []
    for idx, doc in enumerate(docs, start=1):
        title = str(doc.get("title") or doc.get("wikipage") or "")
        extraction = str(doc.get("extraction") or "")
        summary = str(doc.get("summary") or "")
        text = doc_text(doc)
        overlap = 0.55 * token_overlap(question, title + " " + extraction)
        overlap += 0.30 * token_overlap(question, summary)
        overlap += 0.15 * token_overlap(question, text)
        score = 0.78 * overlap + 0.22 * retrieval_score(doc)
        ranked.append((score, idx, doc))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return ranked


def selected_ids_from_lexical(question: str, docs: Sequence[dict[str, Any]], top_k: int, min_score: float) -> list[int]:
    selected = [idx for score, idx, _doc in rank_docs(question, docs) if score >= min_score][:top_k]
    if not selected:
        selected = [idx for _score, idx, _doc in rank_docs(question, docs)[: max(1, min(2, top_k))]]
    return selected


def parse_selected_ids(value: Any, n_docs: int) -> list[int]:
    if not isinstance(value, list):
        return []
    ids: list[int] = []
    for raw in value:
        try:
            doc_id = int(raw)
        except Exception:
            continue
        if 1 <= doc_id <= n_docs and doc_id not in ids:
            ids.append(doc_id)
    return ids


def evaluate_docs(
    question: str,
    docs: list[dict[str, Any]],
    dataset: str,
    client: OpenAIChatClient,
    cache: JsonCache,
    args: argparse.Namespace,
) -> dict[str, Any]:
    shown = docs[: args.max_eval_docs]
    fallback_ids = selected_ids_from_lexical(question, shown, args.max_selected_docs, args.min_doc_quality_score)
    if args.evaluator == "lexical" or not shown:
        return {"action": "filter", "selected_doc_ids": fallback_ids, "confidence": 0.0, "source": "lexical"}

    prompt = CRAG_EVALUATOR_USER.format(question=question, docs=format_docs(shown, args.max_eval_doc_chars))
    key = "api_crag_eval:" + stable_hash(dataset, question, prompt, args.model)
    cached = cache.get(key)
    if cached is None:
        try:
            raw, _usage = client.chat(CRAG_EVALUATOR_SYSTEM, prompt, max_tokens=args.eval_max_tokens)
            parsed = extract_json_object(raw)
            cache.set(key, parsed)
        except Exception as exc:
            logger.warning("CRAG document evaluation failed; using lexical fallback. Error: %s", exc)
            parsed = {"action": "filter", "selected_doc_ids": fallback_ids, "confidence": 0.0, "source": "lexical_fallback"}
            cache.set(key, parsed)
    else:
        parsed = cached

    action = str(parsed.get("action") or "filter").lower()
    if action not in {"use", "filter", "supplement", "combine"}:
        action = "filter"
    ids = parse_selected_ids(parsed.get("selected_doc_ids"), len(shown))
    if not ids and action in {"use", "filter", "combine"}:
        ids = fallback_ids
    return {
        "action": action,
        "selected_doc_ids": ids,
        "confidence": parsed.get("confidence", 0.0),
        "reason": parsed.get("reason", ""),
        "source": "api",
    }


def docs_by_local_ids(docs: Sequence[dict[str, Any]], ids: Sequence[int]) -> list[dict[str, Any]]:
    out = []
    for doc_id in ids:
        idx = int(doc_id) - 1
        if 0 <= idx < len(docs):
            out.append(docs[idx])
    return out


def dedupe_docs(docs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen: set[str] = set()
    for doc in docs:
        key = str(doc.get("id") or "") or normalize_space(str(doc.get("title") or "") + " " + doc_text(doc))[:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


def corrected_docs(question: str, all_docs: list[dict[str, Any]], eval_result: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    eval_pool = all_docs[: args.max_eval_docs]
    selected = docs_by_local_ids(eval_pool, eval_result.get("selected_doc_ids", []))
    action = str(eval_result.get("action") or "filter")
    ranked_all = [doc for _score, _idx, doc in rank_docs(question, all_docs[: args.max_doc_pool])]

    if action == "use":
        docs = selected or ranked_all[: args.max_output_docs]
    elif action == "filter":
        docs = selected or ranked_all[: args.max_output_docs]
    elif action == "supplement":
        docs = ranked_all[: args.max_output_docs]
    else:
        docs = selected + [doc for doc in ranked_all if doc not in selected]
    return dedupe_docs(docs)[: args.max_output_docs]


def top_sentences(question: str, doc: dict[str, Any], max_sentences: int) -> list[str]:
    title = normalize_space(str(doc.get("title") or doc.get("wikipage") or ""))
    extraction = normalize_space(str(doc.get("extraction") or ""))
    summary = normalize_space(str(doc.get("summary") or ""))
    body = normalize_space(str(doc.get("text") or doc.get("contents") or doc.get("content") or doc.get("passage") or ""))
    candidates = []
    if extraction and extraction.lower() != "irrelevant.":
        candidates.append(extraction)
    if summary and summary.lower() != "irrelevant.":
        candidates.extend(sent_tokenize_safe(summary))
    candidates.extend(sent_tokenize_safe(body))
    if not candidates and title:
        candidates = [title]

    scored = []
    seen: set[str] = set()
    for idx, sentence in enumerate(candidates):
        sent = normalize_space(sentence)
        if not sent or sent.lower() in seen:
            continue
        seen.add(sent.lower())
        score = token_overlap(question, title + " " + sent)
        if extraction and sent == extraction:
            score += 0.15
        if summary and sent in summary:
            score += 0.05
        scored.append((score, idx, sent))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [sent for _score, _idx, sent in scored[:max_sentences]]


def refine_docs(question: str, docs: Sequence[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    refined = []
    for idx, doc in enumerate(docs, start=1):
        updated = copy.deepcopy(doc)
        title = normalize_space(str(updated.get("title") or updated.get("wikipage") or ""))
        sentences = top_sentences(question, updated, args.refined_sentences_per_doc)
        text = " ".join(sentences) if sentences else doc_text(updated)
        if len(text) > args.max_refined_doc_chars:
            text = text[: args.max_refined_doc_chars].rsplit(" ", 1)[0] + " ..."
        updated["id"] = str(updated.get("id") or updated.get("doc_id") or idx)
        updated["title"] = title
        updated["text"] = normalize_space(text)
        updated["crag_refined"] = True
        refined.append(updated)
    return refined


def valid_citation_ids(text: str, n_docs: int) -> list[int]:
    ids: list[int] = []
    for match in re.findall(r"\[(\d+)\]", text or ""):
        value = int(match)
        if 1 <= value <= n_docs and value not in ids:
            ids.append(value)
    return ids


def remove_invalid_citations(answer: str, n_docs: int) -> str:
    def repl(match: re.Match[str]) -> str:
        value = int(match.group(1))
        return match.group(0) if 1 <= value <= n_docs else ""

    return normalize_space(re.sub(r"\[(\d+)\]", repl, answer or ""))


def append_missing_citations(answer: str, default_ids: Sequence[int], dataset: str, n_docs: int) -> str:
    default_ids = [int(value) for value in default_ids if 1 <= int(value) <= n_docs][:2]
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


def generate_answer(
    question: str,
    docs: list[dict[str, Any]],
    dataset: str,
    client: OpenAIChatClient,
    cache: JsonCache,
    args: argparse.Namespace,
) -> str:
    prompt = CRAG_GENERATOR_USER.format(
        dataset=dataset,
        question=question,
        docs=format_docs(docs, args.max_generate_doc_chars),
        dataset_rules=DATASET_RULES.get(dataset, ""),
    )
    key = "api_crag_generate:" + stable_hash(dataset, question, prompt, args.model)
    cached = cache.get(key)
    if cached is None:
        raw, _usage = client.chat(CRAG_GENERATOR_SYSTEM, prompt, max_tokens=args.max_tokens)
        try:
            parsed = extract_json_object(raw)
            answer = normalize_space(str(parsed.get("answer") or ""))
        except Exception:
            answer = normalize_space(raw)
        cache.set(key, answer)
    else:
        answer = str(cached or "")

    answer = remove_invalid_citations(answer, len(docs))
    default_ids = valid_citation_ids(answer, len(docs))[:2] or ([1] if docs else [])
    return append_missing_citations(answer, default_ids, dataset, len(docs))


def generate_one(
    idx: int,
    item: dict[str, Any],
    dataset: str,
    all_docs: list[dict[str, Any]],
    client: OpenAIChatClient,
    cache: JsonCache,
    args: argparse.Namespace,
) -> tuple[int, dict[str, Any]]:
    question = normalize_space(str(item.get("question", "") or item.get("instruction", "") or ""))
    if not all_docs:
        all_docs = list(item.get("docs", []) or [])
    if not all_docs:
        out = copy.deepcopy(item)
        out["output"] = normalize_space(str(item.get("output", "") or ""))
        out["api_crag"] = {"error": "no_docs"}
        return idx, out

    eval_result = evaluate_docs(question, all_docs, dataset, client, cache, args)
    chosen = corrected_docs(question, all_docs, eval_result, args)
    refined = refine_docs(question, chosen, args)

    try:
        answer = generate_answer(question, refined, dataset, client, cache, args)
    except Exception as exc:
        logger.warning("API CRAG generation failed for example %s; falling back to source answer. Error: %s", idx, exc)
        refined = list(item.get("docs", []) or refined)
        answer = normalize_space(str(item.get("output", "") or ""))

    out = copy.deepcopy(item)
    out["docs"] = refined
    out["output"] = answer
    out["api_crag"] = {
        "model": args.model,
        "evaluator": args.evaluator,
        "action": eval_result.get("action"),
        "selected_doc_ids": eval_result.get("selected_doc_ids", []),
        "evaluator_confidence": eval_result.get("confidence", 0.0),
        "num_candidate_docs": len(all_docs),
        "num_corrected_docs": len(refined),
    }
    return idx, out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run API-based CRAG-style generation on ALCE data.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-docs-file", type=Path, default=None)
    parser.add_argument("--dataset", choices=("asqa", "eli5", "qampari"), default="")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=450)
    parser.add_argument("--eval-max-tokens", type=int, default=220)
    parser.add_argument("--evaluator", choices=("api", "lexical"), default="api")
    parser.add_argument("--max-doc-pool", type=int, default=100)
    parser.add_argument("--max-eval-docs", type=int, default=8)
    parser.add_argument("--max-output-docs", type=int, default=6)
    parser.add_argument("--max-selected-docs", type=int, default=4)
    parser.add_argument("--min-doc-quality-score", type=float, default=0.10)
    parser.add_argument("--refined-sentences-per-doc", type=int, default=4)
    parser.add_argument("--max-refined-doc-chars", type=int, default=1200)
    parser.add_argument("--max-eval-doc-chars", type=int, default=700)
    parser.add_argument("--max-generate-doc-chars", type=int, default=1100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache-file", type=Path, default=Path("cache/api_crag_cache.json"))
    parser.add_argument("--cache-save-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def write_payload(
    path: Path,
    input_payload: dict[str, Any],
    rows: Sequence[dict[str, Any] | None],
    args: argparse.Namespace,
    dataset: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_rows = [row for row in rows if row is not None]
    payload = {key: value for key, value in input_payload.items() if key != "data"} if isinstance(input_payload, dict) else {}
    payload["data"] = clean_rows
    source_args = input_payload.get("args", {}) if isinstance(input_payload, dict) and isinstance(input_payload.get("args"), dict) else {}
    payload["args"] = {
        **source_args,
        "method": "api_crag_style",
        "dataset_name": dataset,
        "dataset": dataset,
        "tag": "api_crag",
        "model": args.model,
        "shot": source_args.get("shot", 2),
        "ndoc": args.max_output_docs,
        "evaluator": args.evaluator,
        "candidate_docs_file": str(args.candidate_docs_file or ""),
        "max_doc_pool": args.max_doc_pool,
        "max_eval_docs": args.max_eval_docs,
        "max_output_docs": args.max_output_docs,
        "refined_sentences_per_doc": args.refined_sentences_per_doc,
    }
    payload["total_cost"] = input_payload.get("total_cost", 0.0) if isinstance(input_payload, dict) else 0.0
    payload["api_crag_summary"] = {
        "num_examples": len(clean_rows),
        "method_note": "API CRAG uses document-quality correction before answer generation.",
        "source_args": source_args,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None), encoding="utf-8")


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
    logger.info("Generating %d %s examples with API-CRAG model=%s workers=%d", len(rows), dataset, args.model, args.workers)

    def submit(idx: int, item: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        docs = provider.docs_for(item, idx) or list(item.get("docs", []) or [])
        docs = docs[: args.max_doc_pool]
        return generate_one(idx, item, dataset, docs, client, cache, args)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(submit, idx, item) for idx, item in enumerate(rows)]
        for done_count, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="API CRAG"), start=1):
            idx, out = future.result()
            out_rows[idx] = out
            if args.checkpoint_every and done_count % args.checkpoint_every == 0:
                write_payload(args.output.with_suffix(args.output.suffix + ".checkpoint.json"), payload, out_rows, args, dataset)
                cache.flush()

    write_payload(args.output, payload, out_rows, args, dataset)
    cache.flush()
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit atomic claims and their citations in attributed RAG outputs."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import math
import os
import re
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("claim_audit")


CLAIM_DECOMPOSITION_SYSTEM = """You are an expert fact-checking assistant.
Your job is to split an answer sentence into minimal, self-contained factual claims.
Return only valid JSON. Do not include markdown fences."""


CLAIM_DECOMPOSITION_USER = """Question:
{question}

Full answer:
{answer}

Sentence to decompose:
{sentence}

Task:
Extract atomic factual claims from the sentence.

Rules:
1. A claim is atomic if it checks one factual relation, attribute, date, quantity, event, or comparison.
2. Split conjunctions and appositives. Example: "A founded X in 2010 and led it until 2015" becomes two claims.
3. Split lists of entities into separate claims when each entity relation should be independently checked.
   Example: "X was founded by A, B, and C in 2010" becomes:
   - "X was founded in 2010."
   - "A founded X."
   - "B founded X."
   - "C founded X."
4. A claim should not contain multiple people, organizations, locations, dates, or quantities unless they are inseparable.
5. Make each claim self-contained. Resolve pronouns using the question and full answer.
6. Do not invent facts that are not stated in the sentence.
7. Ignore pure discourse phrases, opinions, and citation markers.
8. Preserve uncertainty when present, such as "may", "reportedly", or "is disputed".
9. If the sentence has no factual claim, return an empty list.

Return JSON with exactly this schema:
{{
  "claims": [
    {{
      "claim": "self-contained atomic claim",
      "importance": "critical|supporting|background",
      "claim_type": "entity|relation|date|quantity|event|comparison|definition|other"
    }}
  ]
}}"""


LLM_VERIFIER_SYSTEM = """You are a strict citation fact checker.
Given a question, a factual claim, and evidence passages, decide whether the evidence supports the claim.
Return only valid JSON. Do not include markdown fences."""


LLM_VERIFIER_USER = """Question:
{question}

Claim:
{claim}

Evidence:
{evidence}

Labels:
- supported: the evidence directly supports the claim, or the claim follows from a small, explicit combination of the evidence.
- contradicted: the evidence directly contradicts the claim.
- not_supported: the evidence is related but insufficient, or it does not mention the claim.

Be strict. Do not use outside knowledge.

Return JSON with exactly this schema:
{{
  "label": "supported|contradicted|not_supported",
  "confidence": 0.0,
  "rationale": "short explanation"
}}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-hoc atomic claim audit for ALCE result JSON.")
    parser.add_argument("--input", required=True, type=Path, help="ALCE result JSON with a top-level `data` list.")
    parser.add_argument("--output", required=True, type=Path, help="Where to write the audited result JSON.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N examples.")
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset name. Used to infer answer formatting, especially qampari.",
    )
    parser.add_argument(
        "--answer-format",
        choices=["auto", "paragraph", "qampari"],
        default="auto",
        help=(
            "How to split generated answers before claim decomposition. "
            "`paragraph` uses sentence splitting; `qampari` treats each cited list item as an answer unit."
        ),
    )

    parser.add_argument(
        "--decomposer",
        choices=["llm", "sentence"],
        default="llm",
        help="Claim decomposition method.",
    )
    parser.add_argument("--openai-api", action="store_true", help="Use OpenAI-compatible chat API for LLM modules.")
    parser.add_argument("--decompose-model", default="gpt-4o-mini", help="Chat model for LLM claim decomposition.")
    parser.add_argument("--llm-verify-model", default="gpt-4o-mini", help="Chat model for LLM verification.")
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-top-p", type=float, default=1.0)
    parser.add_argument("--llm-max-retries", type=int, default=5)
    parser.add_argument(
        "--verifier",
        choices=["nli", "llm", "lexical", "hybrid"],
        default="nli",
        help=(
            "Verification method. `nli` is local NLI, `llm` uses a chat model, "
            "`hybrid` uses NLI first and LLM for ambiguous cases."
        ),
    )
    parser.add_argument(
        "--nli-model",
        default="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        help="HF NLI model for claim-evidence verification.",
    )
    parser.add_argument("--device", default=None, help="NLI device. Default: cuda if available else cpu.")
    parser.add_argument("--nli-batch-size", type=int, default=8)
    parser.add_argument("--nli-max-length", type=int, default=512)
    parser.add_argument(
        "--nli-premise-mode",
        choices=["doc", "sentence_window", "doc_then_sentence"],
        default="doc",
        help=(
            "`doc` verifies against full cited docs; `sentence_window` first selects short relevant "
            "sentence windows from cited docs, which is usually better for long/noisy retrieved docs."
        ),
    )
    parser.add_argument("--nli-top-sentences", type=int, default=4)
    parser.add_argument("--nli-sentence-window-size", type=int, default=1)
    parser.add_argument("--entail-threshold", type=float, default=0.50)
    parser.add_argument("--contradiction-threshold", type=float, default=0.50)
    parser.add_argument(
        "--ambiguous-margin",
        type=float,
        default=0.10,
        help="Hybrid mode calls LLM if entailment is within this margin of the decision threshold.",
    )
    parser.add_argument(
        "--nli-aggregation",
        choices=["joint", "max_doc", "joint_then_max"],
        default="joint_then_max",
        help="How to score multiple cited documents.",
    )

    parser.add_argument(
        "--evidence-scope",
        choices=["cited", "all_docs", "cited_then_all"],
        default="cited",
        help=(
            "`cited` verifies against citations in the source sentence. "
            "`all_docs` ignores citations and searches all item docs. "
            "`cited_then_all` diagnoses wrong citations by checking all docs when cited evidence fails."
        ),
    )
    parser.add_argument("--max-all-docs", type=int, default=20, help="Max docs to check for all_docs diagnostics.")
    parser.add_argument("--max-evidence-chars", type=int, default=6000, help="Truncate evidence text for LLM prompts.")
    parser.add_argument("--lexical-threshold", type=float, default=0.45, help="Threshold used by the lexical verifier.")

    parser.add_argument("--output-field", default="output", help="Field containing generated answer.")
    parser.add_argument("--docs-field", default="docs", help="Field containing candidate docs.")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--title-field", default="title")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--sent-field", default="sent", help="Alternative doc text field used by some ALCE docs.")
    parser.add_argument("--cache-file", type=Path, default=None, help="Optional JSON cache for LLM calls.")
    parser.add_argument("--pretty", action="store_true", help="Write indented JSON.")
    return parser.parse_args()


def get_tqdm():
    try:
        from tqdm import tqdm

        return tqdm
    except ImportError:
        return lambda x, **_: x


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def remove_citations(text: str) -> str:
    text = re.sub(r"\s*\[\d+\]", "", text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_citations(text: str) -> list[int]:
    ids = []
    for match in re.findall(r"\[(\d+)\]", text or ""):
        try:
            citation_id = int(match)
        except ValueError:
            continue
        if citation_id > 0 and citation_id not in ids:
            ids.append(citation_id)
    return ids


def format_citation_text(citation_ids: Sequence[int]) -> str:
    return "".join(f"[{citation_id}]" for citation_id in citation_ids)


def sent_tokenize_safe(text: str) -> list[str]:
    text = normalize_space(text)
    if not text:
        return []
    try:
        from nltk import sent_tokenize

        sents = sent_tokenize(text)
    except Exception:
        # Keep bracket citations attached to their sentence. This fallback is
        # intentionally conservative; ALCE outputs are usually one paragraph.
        sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [s.strip() for s in sents if s.strip()]


def infer_answer_format(args: argparse.Namespace) -> str:
    answer_format = str(getattr(args, "answer_format", "auto") or "auto").lower()
    if answer_format != "auto":
        return answer_format

    dataset_name = str(getattr(args, "dataset_name", "") or "").lower()
    input_path = str(getattr(args, "input", "") or "").lower()
    if dataset_name == "qampari" or "qampari" in input_path:
        return "qampari"
    return "paragraph"


def split_answer_units(answer: str, args: argparse.Namespace) -> list[str]:
    if infer_answer_format(args) == "qampari":
        units = split_qampari_answer_items(answer)
        if units:
            return units
    return sent_tokenize_safe(answer)


def split_qampari_answer_items(answer: str) -> list[str]:
    """Split QAMPARI comma-list answers while keeping citation ids per item.

    QAMPARI answers are normally serialized as:
        entity [1], entity [2], entity with comma in title [3].

    A plain comma split breaks titles such as "Mai, the Psychic Girl". Instead,
    we primarily split after citation groups. If the model produced no citations,
    we fall back to conservative comma splitting because ALCE will penalize the
    missing citations anyway.
    """

    text = normalize_space(answer).strip()
    if not text:
        return []

    cited_units = []
    start = 0
    # Split only at a comma that immediately follows one or more citation ids.
    for match in re.finditer(r"((?:\[\d+\])+)\s*,\s*", text):
        end = match.end(1)
        unit = text[start:end].strip().rstrip(",;")
        if unit:
            cited_units.append(unit)
        start = match.end()

    if cited_units:
        tail = text[start:].strip().rstrip(".;")
        if tail:
            cited_units.append(tail)
        return [unit for unit in cited_units if unit]

    if parse_citations(text):
        return [text]

    # No citations: this is usually an abstention or a malformed QAMPARI answer.
    # Keep obvious abstentions intact, otherwise split list-like outputs.
    lowered = text.lower()
    if "do not contain" in lowered or "cannot provide" in lowered or "insufficient" in lowered:
        return [text]
    return [part.strip().rstrip(".;") for part in re.split(r"\s*,\s*", text) if part.strip()]


def load_payload(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload, payload["data"]
    if isinstance(payload, list):
        return {"data": payload}, payload
    raise ValueError("Expected ALCE result JSON: either a list or a dict containing `data`.")


def format_doc(doc: dict[str, Any], args: argparse.Namespace) -> str:
    title = str(doc.get(args.title_field, "") or "").strip()
    body = str(doc.get(args.sent_field) or doc.get(args.text_field, "") or "").strip()
    if title and body:
        return f"Title: {title}\n{body}"
    return title or body


NLI_SENTENCE_STOPWORDS = {
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
}


def content_tokens_for_nli(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return {token for token in tokens if token not in NLI_SENTENCE_STOPWORDS and len(token) > 1}


def token_overlap_for_nli(query: str, text: str) -> float:
    query_tokens = content_tokens_for_nli(query)
    if not query_tokens:
        return 0.0
    text_tokens = content_tokens_for_nli(text)
    if not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def nli_sentence_window_candidates(
    question: str,
    claim: str,
    doc_id: int,
    doc: dict[str, Any],
    args: argparse.Namespace,
) -> list[tuple[str, list[int], str]]:
    """Build short evidence premises so NLI can see the actual supporting sentence.

    Full retrieved documents are often too long/noisy for a 512-token NLI model.
    A correct claim can be marked unsupported simply because the supporting
    sentence is truncated away or diluted by unrelated text. We therefore score
    sentences by lexical overlap and verify the top local windows.
    """

    title = normalize_space(str(doc.get(args.title_field, "") or ""))
    body = normalize_space(str(doc.get(args.sent_field) or doc.get(args.text_field, "") or ""))
    sentences = sent_tokenize_safe(body) if body else []
    if not sentences and title:
        sentences = [title]
    if not sentences:
        return []

    ranked: list[tuple[float, int]] = []
    for idx, sentence in enumerate(sentences):
        score = 0.70 * token_overlap_for_nli(claim, sentence)
        score += 0.20 * token_overlap_for_nli(question, sentence)
        score += 0.10 * token_overlap_for_nli(claim, title)
        ranked.append((score, idx))

    ranked.sort(key=lambda row: (-row[0], row[1]))
    top_k = max(1, int(getattr(args, "nli_top_sentences", 4) or 4))
    side = max(0, int(getattr(args, "nli_sentence_window_size", 1) or 0))
    candidates: list[tuple[str, list[int], str]] = []
    seen: set[str] = set()
    for _score, idx in ranked[:top_k]:
        start = max(0, idx - side)
        end = min(len(sentences), idx + side + 1)
        window = " ".join(sentences[start:end])
        premise = f"Title: {title}\n{window}" if title else window
        premise = normalize_space(premise)
        if premise and premise not in seen:
            seen.add(premise)
            candidates.append((f"doc_{doc_id}_sent_{idx}", [doc_id], premise))
    return candidates


def docs_by_citation_ids(
    docs: Sequence[dict[str, Any]],
    citation_ids: Sequence[int],
) -> list[tuple[int, dict[str, Any]]]:
    selected = []
    for citation_id in citation_ids:
        idx = citation_id - 1
        if 0 <= idx < len(docs):
            selected.append((citation_id, docs[idx]))
    return selected


def stable_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()


class JsonCache:
    def __init__(self, path: Path | None, save_every: int = 1) -> None:
        self.path = path
        self.data: dict[str, Any] = {}
        self.save_every = max(1, int(save_every or 1))
        self._dirty_writes = 0
        self._lock = threading.RLock()
        if path and path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Could not read cache file %s; starting with an empty cache.", path)

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if self.data.get(key) == value:
                return
            self.data[key] = value
            self._dirty_writes += 1
            if self.path and self._dirty_writes >= self.save_every:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self.path or self._dirty_writes <= 0:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f"{self.path.name}.tmp")
        tmp_path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)
        self._dirty_writes = 0


def validate_openai_api_key(api_key: str | None) -> str:
    """Validate an API key before it is placed in an HTTP header."""
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    if api_key != api_key.strip() or any(char.isspace() for char in api_key):
        raise ValueError("OPENAI_API_KEY contains leading, trailing, or embedded whitespace.")
    if not api_key.isascii():
        raise ValueError(
            "OPENAI_API_KEY must contain ASCII characters only. "
            "Replace any Chinese placeholder text or smart quotes with the real API key."
        )
    placeholder = api_key.lower()
    if placeholder in {"your_api_key", "your-api-key", "openai_api_key", "api_key"} or "placeholder" in placeholder:
        raise ValueError("OPENAI_API_KEY still contains a placeholder instead of a real API key.")
    return api_key


class OpenAIChatClient:
    def __init__(self, model: str, temperature: float, top_p: float, max_retries: int) -> None:
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries
        try:
            import openai
        except ImportError as exc:
            raise ImportError("`--openai-api` requires the `openai` package.") from exc

        self.openai = openai
        self.uses_new_client = hasattr(openai, "OpenAI")

        api_key = validate_openai_api_key(os.environ.get("OPENAI_API_KEY"))
        base_url = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
        org_id = os.environ.get("OPENAI_ORG_ID")

        if self.uses_new_client:
            kwargs: dict[str, Any] = {}
            kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url
            if org_id:
                kwargs["organization"] = org_id
            self.client = openai.OpenAI(**kwargs)
        else:
            openai.api_key = api_key
            if org_id:
                openai.organization = org_id
            if base_url:
                openai.api_base = base_url
            self.client = None

    def chat(self, system: str, user: str, max_tokens: int) -> tuple[str, dict[str, int]]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.uses_new_client:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=max_tokens,
                    )
                    content = response.choices[0].message.content or ""
                    usage = {
                        "prompt_tokens": int(getattr(response.usage, "prompt_tokens", 0) or 0),
                        "completion_tokens": int(getattr(response.usage, "completion_tokens", 0) or 0),
                    }
                    return content, usage
                response = self.openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=max_tokens,
                )
                content = response["choices"][0]["message"]["content"]
                usage = {
                    "prompt_tokens": int(response.get("usage", {}).get("prompt_tokens", 0)),
                    "completion_tokens": int(response.get("usage", {}).get("completion_tokens", 0)),
                }
                return content, usage
            except Exception as exc:
                last_error = exc
                logger.warning("OpenAI-compatible API retry %s/%s after error: %s", attempt, self.max_retries, exc)
                time.sleep(min(2 * attempt, 10))
        raise RuntimeError(f"Chat API failed after {self.max_retries} retries: {last_error}")


def extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return json.loads(raw[start : end + 1])
    raise ValueError(f"Could not parse JSON from model output: {text[:500]}")


class ClaimDecomposer:
    def decompose(self, question: str, answer: str, sentence: str) -> list[dict[str, Any]]:
        raise NotImplementedError


class SentenceClaimDecomposer(ClaimDecomposer):
    def decompose(self, question: str, answer: str, sentence: str) -> list[dict[str, Any]]:
        clean = remove_citations(sentence)
        if not clean:
            return []
        return [{"claim": clean, "importance": "critical", "claim_type": "other"}]


class LLMClaimDecomposer(ClaimDecomposer):
    def __init__(self, client: OpenAIChatClient, cache: JsonCache) -> None:
        self.client = client
        self.cache = cache
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def decompose(self, question: str, answer: str, sentence: str) -> list[dict[str, Any]]:
        clean_sentence = remove_citations(sentence)
        if not clean_sentence:
            return []

        key = "decompose:" + stable_hash(question, answer, clean_sentence)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        user_prompt = CLAIM_DECOMPOSITION_USER.format(
            question=question,
            answer=answer,
            sentence=clean_sentence,
        )
        try:
            raw, usage = self.client.chat(CLAIM_DECOMPOSITION_SYSTEM, user_prompt, max_tokens=600)
        except Exception as exc:
            logger.warning(
                "Claim decomposition API failed; falling back to sentence claim. Error: %s",
                exc,
            )
            normalized = [{"claim": clean_sentence, "importance": "critical", "claim_type": "other"}]
            self.cache.set(key, normalized)
            return normalized

        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)

        try:
            payload = extract_json_object(raw)
            claims = payload.get("claims", [])
            normalized = []
            for claim in claims:
                if not isinstance(claim, dict):
                    continue
                text = normalize_space(str(claim.get("claim", "")))
                if not text:
                    continue
                normalized.append(
                    {
                        "claim": text,
                        "importance": str(claim.get("importance", "supporting")),
                        "claim_type": str(claim.get("claim_type", "other")),
                    }
                )
        except Exception as exc:
            logger.warning("Claim decomposition JSON parse failed; falling back to sentence claim. Error: %s", exc)
            normalized = [{"claim": clean_sentence, "importance": "critical", "claim_type": "other"}]

        normalized = split_compound_membership_claims(normalized)
        self.cache.set(key, normalized)
        return normalized


def split_compound_membership_claims(claims: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split simple membership-list claims that LLM decomposition often misses.

    Example:
      "Simon & Garfunkel consists of Paul Simon and Art Garfunkel."
    becomes:
      "Paul Simon is a member of Simon & Garfunkel."
      "Art Garfunkel is a member of Simon & Garfunkel."

    This conservative post-pass only handles explicit "consists of" /
    "is composed of" / "is made up of" patterns. It avoids broad semantic
    rewriting so normal relation claims are left untouched.
    """

    output: list[dict[str, Any]] = []
    for claim in claims:
        text = normalize_space(str(claim.get("claim", "") or "")).rstrip(".")
        split_claims = split_membership_claim_text(text)
        if not split_claims:
            output.append(claim)
            continue
        for split_text in split_claims:
            updated = dict(claim)
            updated["claim"] = split_text
            updated["claim_type"] = "relation"
            output.append(updated)
    return output


def split_membership_claim_text(text: str) -> list[str]:
    patterns = [
        r"^(?P<group>.+?)\s+consists?\s+of\s+(?P<members>.+)$",
        r"^(?P<group>.+?)\s+is\s+composed\s+of\s+(?P<members>.+)$",
        r"^(?P<group>.+?)\s+is\s+made\s+up\s+of\s+(?P<members>.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        group = normalize_space(match.group("group")).strip("\"'")
        members = split_entity_list(match.group("members"))
        if len(members) < 2:
            return []
        return [f"{member} is a member of {group}." for member in members]
    return []


def split_entity_list(text: str) -> list[str]:
    text = normalize_space(text).strip().rstrip(".")
    text = re.sub(r"^(?:members\s+)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(and|or)\b", ",", text)
    parts = [normalize_space(part).strip(" \"'") for part in text.split(",")]
    cleaned = []
    for part in parts:
        part = re.sub(
            r"^(?:singer-songwriter|singer|songwriter|actor|actress|musician|composer|producer)\s+",
            "",
            part,
            flags=re.IGNORECASE,
        )
        part = normalize_space(part).strip(" \"'")
        if part:
            cleaned.append(part)
    return cleaned


@dataclass
class VerificationResult:
    label: str
    confidence: float
    entailment: float | None = None
    neutral: float | None = None
    contradiction: float | None = None
    rationale: str = ""
    verifier: str = ""
    evidence_scope: str = ""
    evidence_doc_ids: list[int] | None = None


class Verifier:
    def verify(
        self,
        question: str,
        claim: str,
        evidence_docs: Sequence[tuple[int, dict[str, Any]]],
        args: argparse.Namespace,
    ) -> VerificationResult:
        raise NotImplementedError


class LexicalVerifier(Verifier):
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
    }

    def verify(
        self,
        question: str,
        claim: str,
        evidence_docs: Sequence[tuple[int, dict[str, Any]]],
        args: argparse.Namespace,
    ) -> VerificationResult:
        evidence = " ".join(format_doc(doc, args) for _, doc in evidence_docs)
        claim_tokens = self._tokens(claim)
        evidence_tokens = self._tokens(evidence)
        if not claim_tokens:
            score = 0.0
        else:
            score = len(claim_tokens & evidence_tokens) / len(claim_tokens)
        label = "supported" if score >= args.lexical_threshold else "not_supported"
        return VerificationResult(
            label=label,
            confidence=score,
            rationale="Lexical-overlap verifier; use NLI or an LLM for stronger semantic verification.",
            verifier="lexical",
            evidence_doc_ids=[doc_id for doc_id, _ in evidence_docs],
        )

    def _tokens(self, text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
        return {tok for tok in tokens if tok not in self.STOPWORDS and len(tok) > 2}


class LLMBasedVerifier(Verifier):
    def __init__(self, client: OpenAIChatClient, cache: JsonCache) -> None:
        self.client = client
        self.cache = cache
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def verify(
        self,
        question: str,
        claim: str,
        evidence_docs: Sequence[tuple[int, dict[str, Any]]],
        args: argparse.Namespace,
    ) -> VerificationResult:
        evidence = "\n\n".join(
            f"[{doc_id}] {format_doc(doc, args)}" for doc_id, doc in evidence_docs
        )
        evidence = evidence[: args.max_evidence_chars]
        key = "verify:" + stable_hash(question, claim, evidence)
        cached = self.cache.get(key)
        if cached is not None:
            return VerificationResult(**cached)

        user_prompt = LLM_VERIFIER_USER.format(question=question, claim=claim, evidence=evidence)
        raw, usage = self.client.chat(LLM_VERIFIER_SYSTEM, user_prompt, max_tokens=350)
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)

        try:
            payload = extract_json_object(raw)
            label = str(payload.get("label", "not_supported"))
            if label not in {"supported", "contradicted", "not_supported"}:
                label = "not_supported"
            confidence = float(payload.get("confidence", 0.0))
            rationale = normalize_space(str(payload.get("rationale", "")))
        except Exception as exc:
            logger.warning("LLM verifier JSON parse failed; marking not_supported. Error: %s", exc)
            label = "not_supported"
            confidence = 0.0
            rationale = "Verifier output could not be parsed."

        result = VerificationResult(
            label=label,
            confidence=max(0.0, min(confidence, 1.0)),
            rationale=rationale,
            verifier="llm",
            evidence_doc_ids=[doc_id for doc_id, _ in evidence_docs],
        )
        self.cache.set(key, result.__dict__)
        return result


class NLIVerifier(Verifier):
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError("NLI verification requires `torch` and `transformers`.") from exc

        self.torch = torch
        self.device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(args.nli_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(args.nli_model)
        self.model.to(self.device)
        self.model.eval()
        self.label_map = self._build_label_map()
        logger.info("Loaded NLI verifier %s on %s", args.nli_model, self.device)

    def _build_label_map(self) -> dict[str, int | None]:
        id2label = getattr(self.model.config, "id2label", {}) or {}
        labels = {int(i): str(label).lower() for i, label in id2label.items()}
        mapping: dict[str, int | None] = {"entailment": None, "neutral": None, "contradiction": None}
        for idx, label in labels.items():
            if "entail" in label:
                mapping["entailment"] = idx
            elif "neutral" in label:
                mapping["neutral"] = idx
            elif "contrad" in label:
                mapping["contradiction"] = idx
        # Common fallback for MNLI models: contradiction, neutral, entailment.
        if mapping["entailment"] is None and len(labels) == 3:
            mapping["contradiction"] = 0
            mapping["neutral"] = 1
            mapping["entailment"] = 2
        if mapping["entailment"] is None:
            raise ValueError(f"Cannot identify entailment label from model config: {id2label}")
        return mapping

    def verify(
        self,
        question: str,
        claim: str,
        evidence_docs: Sequence[tuple[int, dict[str, Any]]],
        args: argparse.Namespace,
    ) -> VerificationResult:
        if not evidence_docs:
            return VerificationResult(
                label="no_citation",
                confidence=0.0,
                rationale="No cited evidence was provided.",
                verifier="nli",
                evidence_doc_ids=[],
            )

        doc_ids = [doc_id for doc_id, _ in evidence_docs]
        candidates: list[tuple[str, list[int], str]] = []
        premise_mode = str(getattr(args, "nli_premise_mode", "doc") or "doc")
        use_doc_premises = premise_mode in {"doc", "doc_then_sentence"}
        use_sentence_premises = premise_mode in {"sentence_window", "doc_then_sentence"}

        if use_doc_premises and args.nli_aggregation in {"joint", "joint_then_max"}:
            joint = "\n\n".join(f"[{doc_id}] {format_doc(doc, args)}" for doc_id, doc in evidence_docs)
            candidates.append(("joint", doc_ids, joint))
        if use_doc_premises and args.nli_aggregation in {"max_doc", "joint_then_max"}:
            for doc_id, doc in evidence_docs:
                candidates.append((f"doc_{doc_id}", [doc_id], format_doc(doc, args)))
        if use_sentence_premises:
            for doc_id, doc in evidence_docs:
                candidates.extend(nli_sentence_window_candidates(question, claim, doc_id, doc, args))
        if not candidates:
            for doc_id, doc in evidence_docs:
                candidates.append((f"doc_{doc_id}", [doc_id], format_doc(doc, args)))

        scored = [self._score(premise, claim, label, ids, args) for label, ids, premise in candidates]
        best = max(scored, key=lambda result: result.entailment or 0.0)
        return best

    def _score(
        self,
        premise: str,
        claim: str,
        evidence_label: str,
        evidence_doc_ids: list[int],
        args: argparse.Namespace,
    ) -> VerificationResult:
        premise = normalize_space(premise)
        claim = normalize_space(claim)
        with self.torch.inference_mode():
            encoded = self.tokenizer(
                premise,
                claim,
                padding=True,
                truncation=True,
                max_length=args.nli_max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            logits = self.model(**encoded).logits[0]
            probs = self.torch.softmax(logits.float(), dim=-1).detach().cpu().tolist()

        entail_idx = self.label_map["entailment"]
        neutral_idx = self.label_map["neutral"]
        contra_idx = self.label_map["contradiction"]
        entail = float(probs[entail_idx]) if entail_idx is not None else 0.0
        neutral = float(probs[neutral_idx]) if neutral_idx is not None else None
        contradiction = float(probs[contra_idx]) if contra_idx is not None else None

        if contradiction is not None and contradiction >= args.contradiction_threshold and contradiction > entail:
            label = "contradicted"
            confidence = contradiction
        elif entail >= args.entail_threshold:
            label = "supported"
            confidence = entail
        else:
            label = "not_supported"
            confidence = 1.0 - entail

        return VerificationResult(
            label=label,
            confidence=confidence,
            entailment=entail,
            neutral=neutral,
            contradiction=contradiction,
            rationale=f"NLI aggregation candidate: {evidence_label}.",
            verifier="nli",
            evidence_doc_ids=evidence_doc_ids,
        )


class HybridVerifier(Verifier):
    def __init__(self, nli: NLIVerifier, llm: LLMBasedVerifier) -> None:
        self.nli = nli
        self.llm = llm

    def verify(
        self,
        question: str,
        claim: str,
        evidence_docs: Sequence[tuple[int, dict[str, Any]]],
        args: argparse.Namespace,
    ) -> VerificationResult:
        nli_result = self.nli.verify(question, claim, evidence_docs, args)
        entail = nli_result.entailment or 0.0
        ambiguous = abs(entail - args.entail_threshold) <= args.ambiguous_margin
        if nli_result.label in {"no_citation", "supported", "contradicted"} and not ambiguous:
            nli_result.verifier = "hybrid:nli"
            return nli_result
        llm_result = self.llm.verify(question, claim, evidence_docs, args)
        llm_result.verifier = "hybrid:llm"
        llm_result.entailment = nli_result.entailment
        llm_result.neutral = nli_result.neutral
        llm_result.contradiction = nli_result.contradiction
        return llm_result


def make_decomposer(args: argparse.Namespace, cache: JsonCache) -> ClaimDecomposer:
    if args.decomposer == "sentence":
        return SentenceClaimDecomposer()
    if not args.openai_api:
        raise ValueError("`--decomposer llm` requires `--openai-api`.")
    client = OpenAIChatClient(
        model=args.decompose_model,
        temperature=args.llm_temperature,
        top_p=args.llm_top_p,
        max_retries=args.llm_max_retries,
    )
    return LLMClaimDecomposer(client, cache)


def make_verifier(args: argparse.Namespace, cache: JsonCache) -> Verifier:
    if args.verifier == "lexical":
        return LexicalVerifier()
    if args.verifier == "nli":
        return NLIVerifier(args)
    if args.verifier == "llm":
        if not args.openai_api:
            raise ValueError("`--verifier llm` requires `--openai-api`.")
        client = OpenAIChatClient(args.llm_verify_model, args.llm_temperature, args.llm_top_p, args.llm_max_retries)
        return LLMBasedVerifier(client, cache)
    if args.verifier == "hybrid":
        if not args.openai_api:
            raise ValueError("`--verifier hybrid` requires `--openai-api`.")
        nli = NLIVerifier(args)
        client = OpenAIChatClient(args.llm_verify_model, args.llm_temperature, args.llm_top_p, args.llm_max_retries)
        return HybridVerifier(nli, LLMBasedVerifier(client, cache))
    raise ValueError(f"Unknown verifier: {args.verifier}")


def select_evidence_docs(
    docs: Sequence[dict[str, Any]],
    citation_ids: Sequence[int],
    args: argparse.Namespace,
) -> tuple[list[tuple[int, dict[str, Any]]], str]:
    if args.evidence_scope == "all_docs":
        return [(idx + 1, doc) for idx, doc in enumerate(docs[: args.max_all_docs])], "all_docs"
    cited = docs_by_citation_ids(docs, citation_ids)
    return cited, "cited"


def verify_with_optional_recovery(
    verifier: Verifier,
    question: str,
    claim: str,
    docs: Sequence[dict[str, Any]],
    citation_ids: Sequence[int],
    args: argparse.Namespace,
) -> VerificationResult:
    evidence_docs, scope = select_evidence_docs(docs, citation_ids, args)
    result = verifier.verify(question, claim, evidence_docs, args)
    result.evidence_scope = scope

    if args.evidence_scope == "cited_then_all" and result.label != "supported":
        all_docs = [(idx + 1, doc) for idx, doc in enumerate(docs[: args.max_all_docs])]
        recovery = verifier.verify(question, claim, all_docs, args)
        if recovery.label == "supported":
            recovery.label = "wrong_or_missing_citation"
            recovery.evidence_scope = "all_docs_recovery"
            recovery.rationale = (
                "The cited evidence did not support the claim, but another available document did. "
                + recovery.rationale
            )
            return recovery
    return result


def audit_item(
    item: dict[str, Any],
    item_id: int,
    decomposer: ClaimDecomposer,
    verifier: Verifier,
    args: argparse.Namespace,
) -> dict[str, Any]:
    question = str(item.get(args.question_field, ""))
    answer = str(item.get(args.output_field, "") or "")
    docs = item.get(args.docs_field, []) or []
    sentences = split_answer_units(answer, args)

    audited_sentences = []
    all_claims = []

    for sent_id, sentence in enumerate(sentences):
        citation_ids = parse_citations(sentence)
        clean_sentence = remove_citations(sentence)
        claims = decomposer.decompose(question, answer, sentence)
        audited_claims = []
        for claim_offset, claim_payload in enumerate(claims):
            claim_text = normalize_space(str(claim_payload.get("claim", "")))
            if not claim_text:
                continue

            verification = verify_with_optional_recovery(
                verifier=verifier,
                question=question,
                claim=claim_text,
                docs=docs,
                citation_ids=citation_ids,
                args=args,
            )
            claim_record = {
                "claim_id": f"{item_id}-{sent_id}-{claim_offset}",
                "claim": claim_text,
                "importance": claim_payload.get("importance", "supporting"),
                "claim_type": claim_payload.get("claim_type", "other"),
                "claim_extractor": claim_payload.get("claim_extractor", args.decomposer),
                "decontextualized_sentence": claim_payload.get("decontextualized_sentence"),
                "selection_status": claim_payload.get("selection_status"),
                "disambiguation_status": claim_payload.get("disambiguation_status"),
                "source_sentence_id": sent_id,
                "source_sentence": sentence,
                "source_sentence_without_citations": clean_sentence,
                "citation_ids": citation_ids,
                "source_citation_ids": citation_ids,
                "source_citation_text": format_citation_text(citation_ids),
                "label": verification.label,
                "confidence": verification.confidence,
                "entailment": verification.entailment,
                "neutral": verification.neutral,
                "contradiction": verification.contradiction,
                "rationale": verification.rationale,
                "verifier": verification.verifier,
                "evidence_scope": verification.evidence_scope,
                "evidence_doc_ids": verification.evidence_doc_ids or [],
            }
            audited_claims.append(claim_record)
            all_claims.append(claim_record)

        audited_sentences.append(
            {
                "sentence_id": sent_id,
                "sentence": sentence,
                "sentence_without_citations": clean_sentence,
                "citation_ids": citation_ids,
                "claims": audited_claims,
            }
        )

    metrics = compute_item_metrics(sentences, all_claims)
    return {"sentences": audited_sentences, "claims": all_claims, "metrics": metrics}


def compute_item_metrics(sentences: Sequence[str], claims: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels = collections.Counter(str(claim.get("label", "unknown")) for claim in claims)
    total_claims = len(claims)
    supported = labels.get("supported", 0)
    recoverable = labels.get("wrong_or_missing_citation", 0)
    unsupported_like = total_claims - supported
    citation_claims = sum(1 for claim in claims if claim.get("citation_ids"))
    critical_claims = [claim for claim in claims if claim.get("importance") == "critical"]
    critical_supported = sum(1 for claim in critical_claims if claim.get("label") == "supported")

    return {
        "num_sentences": len(sentences),
        "num_claims": total_claims,
        "claims_per_sentence": total_claims / len(sentences) if sentences else 0.0,
        "supported_claims": supported,
        "unsupported_claims": unsupported_like,
        "wrong_or_missing_citation_claims": recoverable,
        "contradicted_claims": labels.get("contradicted", 0),
        "not_supported_claims": labels.get("not_supported", 0),
        "no_citation_claims": labels.get("no_citation", 0),
        "claim_support_rate": supported / total_claims if total_claims else 0.0,
        "unsupported_claim_rate": unsupported_like / total_claims if total_claims else 0.0,
        "citation_coverage_rate": citation_claims / total_claims if total_claims else 0.0,
        "critical_claim_support_rate": critical_supported / len(critical_claims) if critical_claims else None,
        "label_counts": dict(labels),
    }


def summarize_dataset(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    audits = [item.get("cover_audit", {}) for item in items]
    metrics = [audit.get("metrics", {}) for audit in audits if audit.get("metrics")]

    total_claims = sum(int(metric.get("num_claims", 0)) for metric in metrics)
    total_supported = sum(int(metric.get("supported_claims", 0)) for metric in metrics)
    total_unsupported = sum(int(metric.get("unsupported_claims", 0)) for metric in metrics)
    total_no_citation = sum(int(metric.get("no_citation_claims", 0)) for metric in metrics)
    total_contradicted = sum(int(metric.get("contradicted_claims", 0)) for metric in metrics)
    total_recoverable = sum(int(metric.get("wrong_or_missing_citation_claims", 0)) for metric in metrics)
    macro_support_rates = [float(metric.get("claim_support_rate", 0.0)) for metric in metrics]

    summary = {
        "num_examples": len(items),
        "num_claims": total_claims,
        "supported_claims": total_supported,
        "unsupported_claims": total_unsupported,
        "no_citation_claims": total_no_citation,
        "contradicted_claims": total_contradicted,
        "wrong_or_missing_citation_claims": total_recoverable,
        "atomic_claim_support_rate_micro": total_supported / total_claims if total_claims else 0.0,
        "unsupported_claim_rate_micro": total_unsupported / total_claims if total_claims else 0.0,
        "atomic_claim_support_rate_macro": statistics.mean(macro_support_rates) if macro_support_rates else 0.0,
        "avg_claims_per_example": total_claims / len(items) if items else 0.0,
    }
    return summary


def safe_float(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: safe_float(val) for key, val in value.items()}
    if isinstance(value, list):
        return [safe_float(val) for val in value]
    return value


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


def main() -> None:
    args = parse_args()
    if args.decomposer == "llm" and not args.openai_api:
        raise ValueError(
            f"`--decomposer {args.decomposer}` needs `--openai-api`. "
            "Use `--decomposer sentence` when API-backed decomposition is not needed."
        )
    if args.verifier in {"llm", "hybrid"} and not args.openai_api:
        raise ValueError(f"`--verifier {args.verifier}` needs `--openai-api`.")

    payload, items = load_payload(args.input)
    fill_dataset_name_from_payload(args, payload)
    process_items = items if args.limit is None else items[: args.limit]
    cache = JsonCache(args.cache_file)
    decomposer = make_decomposer(args, cache)
    verifier = make_verifier(args, cache)
    tqdm = get_tqdm()

    audited_items = []
    for item_id, item in enumerate(tqdm(process_items, desc="CoverRAG claim audit")):
        updated = dict(item)
        updated["cover_audit"] = audit_item(updated, item_id, decomposer, verifier, args)
        audited_items.append(updated)

    if args.limit is not None and len(items) > args.limit:
        audited_items.extend(items[args.limit :])

    payload["data"] = audited_items
    payload["cover_audit_summary"] = summarize_dataset(audited_items[: len(process_items)])
    payload["cover_audit_config"] = {
        "dataset_name": args.dataset_name,
        "answer_format": args.answer_format,
        "inferred_answer_format": infer_answer_format(args),
        "decomposer": args.decomposer,
        "decompose_model": args.decompose_model if args.decomposer == "llm" else None,
        "verifier": args.verifier,
        "llm_verify_model": args.llm_verify_model if args.verifier in {"llm", "hybrid"} else None,
        "nli_model": args.nli_model if args.verifier in {"nli", "hybrid"} else None,
        "evidence_scope": args.evidence_scope,
        "nli_aggregation": args.nli_aggregation,
        "entail_threshold": args.entail_threshold,
        "contradiction_threshold": args.contradiction_threshold,
    }

    # Include token usage when LLM modules are used.
    token_usage = {}
    if getattr(decomposer, "prompt_tokens", None) is not None:
        token_usage["decomposition_prompt_tokens"] = int(decomposer.prompt_tokens)
        token_usage["decomposition_completion_tokens"] = int(decomposer.completion_tokens)
    if isinstance(verifier, LLMBasedVerifier):
        token_usage["verification_prompt_tokens"] = verifier.prompt_tokens
        token_usage["verification_completion_tokens"] = verifier.completion_tokens
    elif isinstance(verifier, HybridVerifier):
        token_usage["verification_prompt_tokens"] = verifier.llm.prompt_tokens
        token_usage["verification_completion_tokens"] = verifier.llm.completion_tokens
    if token_usage:
        payload["cover_audit_token_usage"] = token_usage

    args.output.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if args.pretty else None
    args.output.write_text(json.dumps(safe_float(payload), ensure_ascii=False, indent=indent), encoding="utf-8")

    logger.info("Wrote CoverRAG claim audit to %s", args.output)
    logger.info("Audit summary: %s", json.dumps(payload["cover_audit_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

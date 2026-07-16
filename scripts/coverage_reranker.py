#!/usr/bin/env python3
"""Coverage-aware evidence reranker utilities.

This module is intentionally small and optional. COVER-RAG can run without it;
when a trained model path is provided, the controller can use it to score
whether an evidence sentence/passage is useful for a missing answer unit.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Sequence


logger = logging.getLogger("coverage_reranker")


def _compact(text: str, limit: int = 900) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def format_coverage_query(question: str, covered_answer: str = "", missing_unit: str = "") -> str:
    """Create the query side for a coverage-aware cross encoder."""

    parts = [f"Question: {_compact(question, 500)}"]
    if covered_answer:
        parts.append(f"Already covered answer: {_compact(covered_answer, 500)}")
    if missing_unit:
        parts.append(f"Missing answer unit: {_compact(missing_unit, 400)}")
    parts.append("Task: decide whether the evidence directly helps fill the missing answer unit.")
    return "\n".join(parts)


def format_candidate_evidence(evidence: str, title: str = "", claim: str = "") -> str:
    """Create the evidence side for a coverage-aware cross encoder."""

    parts = []
    if title:
        parts.append(f"Title: {_compact(title, 160)}")
    if claim:
        parts.append(f"Candidate claim: {_compact(claim, 360)}")
    parts.append(f"Evidence: {_compact(evidence, 900)}")
    return "\n".join(parts)


class CoverageAwareReranker:
    """Lazy HuggingFace cross-encoder scorer.

    The model is expected to be an AutoModelForSequenceClassification. For
    two-label heads, the positive-class softmax probability is returned. For
    one-label heads, the sigmoid score is returned.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 384,
    ) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.batch_size = max(1, int(batch_size or 1))
        self.max_length = max(64, int(max_length or 384))
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            if self.device:
                device = self.device
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            self.device = device
            logger.info("Loaded coverage-aware reranker from %s on %s", self.model_path, self.device)

    def score_pairs(self, queries: Sequence[str], evidences: Sequence[str]) -> list[float]:
        if len(queries) != len(evidences):
            raise ValueError("queries and evidences must have the same length")
        if not queries:
            return []
        self._ensure_loaded()
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None

        scores: list[float] = []
        torch = self._torch
        with self._lock:
            for start in range(0, len(queries), self.batch_size):
                batch_queries = list(queries[start : start + self.batch_size])
                batch_evidences = list(evidences[start : start + self.batch_size])
                encoded = self._tokenizer(
                    batch_queries,
                    batch_evidences,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                with torch.no_grad():
                    logits = self._model(**encoded).logits
                if logits.shape[-1] == 1:
                    batch_scores = torch.sigmoid(logits.squeeze(-1))
                else:
                    batch_scores = torch.softmax(logits, dim=-1)[:, -1]
                scores.extend(float(value) for value in batch_scores.detach().cpu().tolist())
        return scores

    def score_pair(self, query: str, evidence: str) -> float:
        return self.score_pairs([query], [evidence])[0]


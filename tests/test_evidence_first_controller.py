from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from claim_audit import JsonCache, VerificationResult  # noqa: E402
from answer_reader import BlueprintAnswerReader  # noqa: E402


class FakeClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def chat(self, system: str, user: str, max_tokens: int):
        self.calls += 1
        return json.dumps(self.payload), {"prompt_tokens": 10, "completion_tokens": 5}


class FakeVerifier:
    def verify(self, question, claim, evidence_docs, args):
        supported = "invented" not in claim.lower()
        return VerificationResult(
            label="supported" if supported else "not_supported",
            confidence=0.95,
            entailment=0.95 if supported else 0.05,
            evidence_doc_ids=[doc_id for doc_id, _doc in evidence_docs],
        )


def args(**overrides):
    values = {
        "answer_format": "paragraph",
        "dataset_name": "asqa",
        "reader_max_plan_claims": 8,
        "reader_max_unit_words": 40,
        "reader_max_tokens": 300,
        "reader_conformance_verify": True,
        "max_citations_per_sentence": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class BlueprintAnswerReaderTest(unittest.TestCase):
    def test_citations_are_code_bound_and_rejected_unit_falls_back(self):
        client = FakeClient(
            {
                "units": [
                    {"plan_ids": ["p1"], "text": "Alpha is the first answer"},
                    {"plan_ids": ["p2"], "text": "An invented unsupported detail"},
                    {"plan_ids": ["p99"], "text": "Unknown plan content"},
                ]
            }
        )
        reader = BlueprintAnswerReader("fake", JsonCache(None), client=client)
        claims = [
            {
                "claim": "Alpha is the first answer",
                "final_citation_ids": [1],
                "source_sentence_without_citations": "Alpha is the first answer.",
            },
            {
                "claim": "Beta is the second answer",
                "final_citation_ids": [2],
                "source_sentence_without_citations": "Beta is the second answer.",
            },
        ]
        docs = [{"title": "A", "text": "Alpha is the first answer."}, {"title": "B", "text": "Beta is the second answer."}]

        answer, plan, diagnostics = reader.realize("What are the answers?", claims, docs, FakeVerifier(), args())

        self.assertEqual(client.calls, 1)
        self.assertEqual(len(plan), 2)
        self.assertIn("Alpha is the first answer [1].", answer)
        self.assertIn("Beta is the second answer [2].", answer)
        self.assertNotIn("invented", answer.lower())
        self.assertNotIn("[99]", answer)
        self.assertEqual(diagnostics["api_calls"], 1)
        self.assertEqual(diagnostics["conformance_rejections"], 1)
        self.assertEqual(diagnostics["fallback_units"], 1)
        self.assertEqual(diagnostics["unknown_plan_ids"], 1)

    def test_cached_reader_response_does_not_make_second_api_call(self):
        client = FakeClient({"units": [{"plan_ids": ["p1"], "text": "Alpha"}]})
        cache = JsonCache(None)
        reader = BlueprintAnswerReader("fake", cache, client=client)
        claims = [{"claim": "Alpha", "final_citation_ids": [1], "source_sentence_without_citations": "Alpha."}]
        docs = [{"title": "A", "text": "Alpha."}]

        first, _plan, first_diag = reader.realize("Name it", claims, docs, FakeVerifier(), args())
        second, _plan, second_diag = reader.realize("Name it", claims, docs, FakeVerifier(), args())

        self.assertEqual(first, second)
        self.assertEqual(client.calls, 1)
        self.assertEqual(first_diag["api_calls"], 1)
        self.assertTrue(second_diag["cache_hit"])
        self.assertEqual(second_diag["api_calls"], 0)

    def test_auto_format_uses_qampari_list_realization(self):
        client = FakeClient({"units": [{"plan_ids": ["p1"], "text": "Manga Title"}]})
        reader = BlueprintAnswerReader("fake", JsonCache(None), client=client)
        claims = [
            {
                "claim": "Manga Title was drawn by the requested artist",
                "answer_span": "Manga Title",
                "final_citation_ids": [1],
                "source_sentence_without_citations": "Manga Title was drawn by the requested artist.",
            }
        ]
        docs = [{"title": "Manga Title", "text": "Manga Title was drawn by the requested artist."}]

        answer, _plan, _diag = reader.realize(
            "Which manga?",
            claims,
            docs,
            FakeVerifier(),
            args(answer_format="auto", dataset_name="qampari"),
        )

        self.assertEqual(answer, "Manga Title [1].")


if __name__ == "__main__":
    unittest.main()

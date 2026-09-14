from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import coverrag as core  # noqa: E402


class FakeReader:
    def __init__(self) -> None:
        self.calls = 0

    def realize(self, question, claims, docs, verifier, args):
        self.calls += 1
        return (
            "The verified answer [1].",
            [{"plan_id": "p1", "claim": "The verified answer", "citation_ids": [1]}],
            {"api_calls": 1, "fallback_units": 0, "accepted_generated_units": 1},
        )


class EvidenceFirstPipelineTest(unittest.TestCase):
    def test_branch_plans_before_answer_and_calls_reader_once(self):
        argv = [
            "coverrag.py",
            "--input",
            "asqa-input.json",
            "--output",
            "out.json",
            "--control-mode",
            "evidence_seeded",
            "--decomposer",
            "sentence",
            "--verifier",
            "lexical",
            "--expansion-mode",
            "extractive",
            "--no-final-audit",
        ]
        with patch.object(sys, "argv", argv):
            args = core.parse_args()
        args.dataset_name = "asqa"

        claim = {
            "claim_id": "seed-1",
            "claim": "The verified answer",
            "label": "supported",
            "confidence": 0.99,
            "final_citation_ids": [1],
            "citation_ids": [1],
            "source_sentence_without_citations": "The verified answer.",
            "coverage_status": "expanded_supported",
            "answer_span": "verified answer",
            "iteration_round": 0,
            "protected_answer_spans": ["verified answer"],
        }
        item = {
            "question": "What is the answer?",
            "output": "An untrusted host draft [1].",
            "docs": [{"title": "Evidence", "text": "The verified answer."}],
        }
        provider = core.CandidateDocProvider(None, args)
        reader = FakeReader()

        with patch.object(
            core,
            "expand_verified_claims",
            return_value=([claim], [claim], [], []),
        ), patch.object(core, "audit_item", side_effect=AssertionError("draft audit must not run")):
            result = core.revise_item(
                item,
                0,
                provider,
                Mock(),
                Mock(),
                Mock(),
                reader,
                args,
            )

        self.assertEqual(reader.calls, 1)
        self.assertEqual(result["output"], "The verified answer [1].")
        self.assertEqual(result["coverrag"]["control_mode"], "evidence_seeded")
        self.assertEqual(result["coverrag"]["original_output"], "An untrusted host draft [1].")
        self.assertEqual(result["coverrag"]["revision_metrics"]["blueprint_claims"], 1)


if __name__ == "__main__":
    unittest.main()

# CoverRAG

CoverRAG is an inference-time controller for attributed retrieval-augmented
generation. It audits atomic claims against their citations, retrieves or
selects evidence for unresolved gaps, regenerates a cited answer, and audits
the result again. The controller works on JSON output from an existing RAG
system and does not require training or modifying the host retriever or reader.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt
```

Set credentials for an OpenAI-compatible chat endpoint:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://your-endpoint.example"
```

`OPENAI_BASE_URL` is optional when using the default OpenAI endpoint.

## Input Format

The input can be a JSON list or an object with a top-level `data` list. Each
record needs a question, a host answer with bracket citations, and retrieved
documents. Citation `[1]` refers to the first document in `docs`.

```json
{
  "data": [
    {
      "sample_id": "example-1",
      "question": "What happened and why?",
      "output": "The event happened in 2020 [1].",
      "docs": [
        {
          "title": "Source title",
          "text": "Evidence text for the cited claim."
        }
      ]
    }
  ]
}
```

Document text may also be stored in `sent`. Field names can be changed with
the corresponding command-line options.

## Run CoverRAG

The answer-seeded mode audits and repairs an existing host answer:

```bash
MODEL="your-chat-model"

python scripts/coverrag.py \
  --input result/host.json \
  --output result/host.coverrag.json \
  --control-mode answer_seeded \
  --openai-api \
  --decomposer llm \
  --decompose-model "$MODEL" \
  --verifier llm \
  --llm-verify-model "$MODEL" \
  --expansion-mode llm \
  --expansion-model "$MODEL" \
  --cache-file cache/coverrag.json \
  --pretty
```

The evidence-seeded mode constructs a verified claim-citation plan before the
final reader call:

```bash
python scripts/coverrag.py \
  --input result/host.json \
  --output result/host.coverrag.json \
  --control-mode evidence_seeded \
  --openai-api \
  --reader-model "$MODEL" \
  --decomposer llm \
  --decompose-model "$MODEL" \
  --verifier llm \
  --llm-verify-model "$MODEL" \
  --expansion-mode llm \
  --expansion-model "$MODEL" \
  --cache-file cache/coverrag.json \
  --pretty
```

To let the controller search beyond each record's initial documents, pass a
second JSON file containing a larger candidate pool:

```bash
python scripts/coverrag.py \
  --input result/host.json \
  --output result/host.coverrag.json \
  --candidate-docs-file data/candidates.json \
  --dynamic-retrieval-mode global_candidate_docs \
  --coverage-guided-retrieval \
  --openai-api \
  --decomposer llm \
  --decompose-model "$MODEL" \
  --verifier llm \
  --llm-verify-model "$MODEL" \
  --expansion-model "$MODEL"
```

Candidate records are matched by `sample_id`, normalized question, or input
position. Run `python scripts/coverrag.py --help` for retrieval, verification,
budget, concurrency, and resume options.

## Claim Audit Only

```bash
python scripts/claim_audit.py \
  --input result/host.json \
  --output result/host.audit.json \
  --openai-api \
  --decomposer llm \
  --decompose-model "$MODEL" \
  --verifier llm \
  --llm-verify-model "$MODEL" \
  --cache-file cache/audit.json \
  --pretty
```

## Batch Processing

```bash
python scripts/run_batch.py \
  --inputs result/asqa.json result/eli5.json result/qampari.json \
  --output-dir result/coverrag \
  --reader-model "$MODEL" \
  --decompose-model "$MODEL" \
  --verifier llm \
  --llm-verify-model "$MODEL" \
  --expansion-model "$MODEL" \
  --revision-workers 4 \
  --checkpoint-every 20
```

Existing outputs, score files, and caches are reused unless a matching
`--force-*` option is supplied.

## Evaluation

`eval.py` computes the task and sentence-level citation metrics used by ALCE.
Claim-level citation metrics can be added after the final audit:

```bash
python eval.py --f result/host.coverrag.json --citations --qa
python scripts/compute_claim_citation_metrics.py \
  --result result/host.coverrag.json \
  --pretty
python scripts/merge_scores.py \
  --result result/host.coverrag.json \
  --pretty
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Main Files

- `scripts/coverrag.py`: closed-loop controller and command-line entry point
- `scripts/claim_audit.py`: claim decomposition and citation verification
- `scripts/answer_revision.py`: evidence-constrained answer revision
- `scripts/answer_reader.py`: verified blueprint realization
- `scripts/coverage_reranker.py`: optional coverage-aware reranking
- `scripts/run_batch.py`: resumable multi-file runner

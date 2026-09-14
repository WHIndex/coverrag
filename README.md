# CoverRAG

CoverRAG audits claim-citation coverage in an existing RAG answer, retrieves
evidence for unresolved gaps, regenerates the cited answer, and checks it
again. It works with any host RAG system that can export questions, answers,
and retrieved documents in the JSON format below.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt
```

Set an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://your-endpoint.example"
```

## Input

```json
{
  "data": [
    {
      "question": "What happened and why?",
      "output": "The event happened in 2020 [1].",
      "docs": [
        {"title": "Source", "text": "Evidence for the cited claim."}
      ]
    }
  ]
}
```

Citations are one-indexed: `[1]` refers to the first document in `docs`.

## Run

```bash
MODEL="your-chat-model"

python scripts/coverrag.py \
  --input result/host.json \
  --output result/host.coverrag.json \
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

Use a larger candidate evidence pool when available:

```bash
python scripts/coverrag.py \
  --input result/host.json \
  --output result/host.coverrag.json \
  --candidate-docs-file data/candidates.json \
  --dynamic-retrieval-mode global_candidate_docs \
  --coverage-guided-retrieval \
  --openai-api \
  --decompose-model "$MODEL" \
  --verifier llm \
  --llm-verify-model "$MODEL" \
  --expansion-model "$MODEL"
```

## Batch Run

```bash
python scripts/run_batch.py \
  --inputs result/asqa.json result/eli5.json result/qampari.json \
  --output-dir result/coverrag \
  --decompose-model "$MODEL" \
  --verifier llm \
  --llm-verify-model "$MODEL" \
  --expansion-model "$MODEL" \
  --revision-workers 4 \
  --checkpoint-every 20
```

## Test

```bash
python -m unittest discover -s tests -v
```

Run `python scripts/coverrag.py --help` for all options.

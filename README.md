# Codebase Assistant

An AI assistant that understands your codebase. Point it at a GitHub repo, ask natural language questions, and get accurate, sourced answers with file and line citations.

## What Makes This Different

**Tree-sitter AST chunking** — code is split at function and class boundaries, not arbitrary character counts. A function stays whole. This directly improves retrieval precision (measurable via RAGAS scores).

**Async ingestion pipeline** — submitting a repo doesn't block the API. A QStash message queues the job; a worker clones, chunks, embeds, and updates the index asynchronously.

## Architecture

```
POST /repos  →  QStash Queue  →  Webhook  →  Clone → AST Chunk → Upstash Vector
POST /query  →  Vector Search  →  Groq LLM  →  Sourced Answer
                                     ↓
                               Langfuse Tracing
```

## Tech Stack

| Layer | Service | Free Tier |
|---|---|---|
| Vector DB | Upstash Vector (built-in BGE-large) | 10k req/day |
| Async Queue | Upstash QStash | 1k msgs/day |
| LLM | Groq (Llama 3.3 70B) | ~30 RPM |
| Observability | Langfuse Cloud | 50k obs/month |
| Deployment | Render | 750 hrs/month |

## Quick Start

### 1. Install dependencies

```bash
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in: UPSTASH_VECTOR_URL, UPSTASH_VECTOR_TOKEN, GROQ_API_KEY
# Optional: LANGFUSE_*, QSTASH_*, GITHUB_TOKEN, OPENAI_API_KEY
```

**Upstash Vector index setup:**
1. Go to [console.upstash.com/vector](https://console.upstash.com/vector)
2. Create a new index
3. Choose built-in model: `BAAI/bge-large-en-v1.5` (1024 dimensions)
4. Copy the URL and token to `.env`

### 3. Run locally

```bash
# Start the API server
uvicorn src.main:app --reload

# Or with Docker
docker-compose up
```

### 4. Seed a repository (local test)

```bash
# Dry run — just clone and count files
python scripts/seed_repo.py https://github.com/owner/repo --dry-run

# Full ingestion
python scripts/seed_repo.py https://github.com/owner/repo
```

### 5. Use the API

```bash
# Submit a repo
curl -X POST http://localhost:8000/repos \
  -H "Content-Type: application/json" \
  -d '{"url": "https://github.com/tiangolo/fastapi"}'

# Response: {"repo_id": "abc-123", "status": "queued", ...}

# Poll status
curl http://localhost:8000/repos/abc-123

# Ask a question (once status=ready)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "repo_id": "abc-123",
    "question": "Where is request validation handled?"
  }'
```

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## Running Tests

```bash
# All tests
pytest

# Chunker tests only (no external deps needed)
pytest tests/test_chunker.py -v

# With coverage
pytest --cov=src --cov-report=term-missing
```

## RAGAS Evaluation

Measure retrieval and generation quality:

```bash
# Evaluate with Groq
python eval/run_eval.py --repo-id <repo_id> --provider groq

# Compare Groq vs OpenAI side by side
python eval/run_eval.py --repo-id <repo_id> --compare
```

Target scores: context_precision > 0.80, faithfulness > 0.85

## Supported Languages

| Language | Extensions | AST Node Types |
|---|---|---|
| Python | `.py` | function_definition, class_definition |
| TypeScript | `.ts`, `.tsx` | function_declaration, class_declaration, arrow_function |
| JavaScript | `.js`, `.jsx` | function_declaration, class_declaration, arrow_function |
| Go | `.go` | function_declaration, method_declaration, type_declaration |
| Rust | `.rs` | function_item, impl_item, struct_item, enum_item |
| Java | `.java` | class_declaration, method_declaration |
| C/C++ | `.c`, `.cpp` | function_definition, class_specifier |
| Ruby | `.rb` | method, class, module |

## API Reference

### `POST /repos`
Submit a GitHub repo for ingestion.

```json
{ "url": "https://github.com/owner/repo", "branch": "main" }
```

### `GET /repos/{repo_id}`
Poll ingestion status. Status values: `queued` → `cloning` → `chunking` → `embedding` → `ready` / `failed`

### `POST /query`
Ask a question. Returns sourced answer with file:line citations.

```json
{
  "repo_id": "abc-123",
  "question": "Where is payment retry logic handled?",
  "top_k": 8
}
```

### `GET /health`
Service health and dependency status.

## Deployment (Render)

1. Push to GitHub
2. Create a new Web Service on [render.com](https://render.com)
3. Connect your repo, set build command: `pip install -e .`
4. Set start command: `uvicorn src.main:app --host 0.0.0.0 --port $PORT`
5. Add all environment variables from `.env.example`
6. For QStash webhooks, set `APP_PUBLIC_URL` to your Render service URL

"""
FastAPI application entrypoint.

Wires together all routes, initializes the database,
and exposes health check + OpenAPI docs.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import query, repos, webhooks
from src.config import get_settings
from src.db.database import init_db
from src.models import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle."""
    settings = get_settings()
    logger.info(f"Starting Codebase Assistant API (env={settings.app_env})")

    # Initialize database tables
    await init_db()
    logger.info("Database initialized")

    # Validate required credentials
    warnings = []
    if not settings.has_groq and not settings.has_openai:
        warnings.append("No LLM API keys configured (GROQ_API_KEY or OPENAI_API_KEY)")
    if not settings.upstash_vector_url:
        warnings.append("UPSTASH_VECTOR_URL not configured — vector operations will fail")
    if not settings.has_langfuse:
        warnings.append("Langfuse not configured — tracing disabled")

    for w in warnings:
        logger.warning(f"⚠️  {w}")

    yield

    logger.info("Codebase Assistant API shutting down")


app = FastAPI(
    title="Codebase Assistant API",
    description="""
An AI assistant that understands codebases.

## Key Features
- **AST-aware chunking** via tree-sitter (functions and classes stay whole)
- **Async ingestion** via QStash (submit a repo and get answers while it processes)
- **Semantic search** via Upstash Vector (built-in BGE embeddings)
- **Sourced answers** with file:line citations for every claim
- **LLM tracing** via Langfuse for observability and A/B model comparison

## Usage
1. `POST /repos` — Submit a GitHub repo URL
2. `GET /repos/{id}` — Poll until status=ready
3. `POST /query` — Ask questions about the codebase
""",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — open for development, restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(repos.router)
app.include_router(query.router)
app.include_router(webhooks.router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Health check — returns service status and dependency availability."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        services={
            "groq": settings.has_groq,
            "openai": settings.has_openai,
            "upstash_vector": bool(settings.upstash_vector_url),
            "qstash": settings.has_qstash,
            "langfuse": settings.has_langfuse,
            "github_token": bool(settings.github_token),
        },
    )

"""
Query API route.

POST /query — Ask a natural language question about an ingested codebase.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, status

from src.db.repos import get_repo
from src.models import QueryRequest, QueryResponse, RepoStatus
from src.observability.tracing import get_tracer
from src.retrieval.generator import get_generator
from src.retrieval.retriever import retrieve_chunks

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


@router.post(
    "",
    response_model=QueryResponse,
    summary="Ask a question about an ingested codebase",
)
async def query_codebase(body: QueryRequest) -> QueryResponse:
    """
    Ask a natural language question about a previously ingested repository.

    The system retrieves the most relevant code chunks via vector search,
    then uses an LLM to generate a sourced answer citing file paths and line numbers.

    Requires the repository to have status=ready.
    """
    start_ms = time.monotonic() * 1000

    # Validate repo exists and is ready
    record = await get_repo(body.repo_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository {body.repo_id!r} not found",
        )
    if record.status != RepoStatus.READY:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Repository is not ready (current status: {record.status}). Wait for ingestion to complete.",
        )

    # 1. Retrieve relevant chunks
    chunks = retrieve_chunks(
        question=body.question,
        repo_id=body.repo_id,
        top_k=body.top_k,
    )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No relevant code found for this question. The repository may be empty or the question may be too broad.",
        )

    # 2. Generate answer
    repo_name = record.url.split("github.com/")[-1] if "github.com" in record.url else record.url
    generator = get_generator()

    try:
        answer, model_used, usage = generator.generate(
            question=body.question,
            chunks=chunks,
            repo_name=repo_name,
            model=body.model,
        )
    except Exception as e:
        logger.error(f"LLM generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM generation failed: {str(e)[:200]}",
        )

    latency_ms = int(time.monotonic() * 1000 - start_ms)

    # 3. Trace to Langfuse
    tracer = get_tracer()
    trace_id = tracer.trace_query(
        repo_id=body.repo_id,
        question=body.question,
        retrieved_chunks=[{"content": c.content[:300]} for c in chunks],
        answer=answer,
        model=model_used,
        usage=usage,
        latency_ms=latency_ms,
        top_k=body.top_k,
    )

    return QueryResponse(
        answer=answer,
        sources=chunks,
        repo_id=body.repo_id,
        model=model_used,
        trace_id=trace_id,
        latency_ms=latency_ms,
    )

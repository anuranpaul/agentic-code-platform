"""
Langfuse observability integration.

Instruments the query pipeline and ingestion pipeline with traces,
spans, and generation records. Allows:
- End-to-end latency tracking
- Token usage monitoring
- A/B model comparison (Groq vs OpenAI)
- RAGAS score upload as evaluation runs

Falls back gracefully if Langfuse is not configured (no API keys).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

from src.config import get_settings

logger = logging.getLogger(__name__)


class _NoopTracer:
    """A no-op tracer used when Langfuse is not configured."""

    def trace(self, *args, **kwargs) -> "_NoopSpan":
        return _NoopSpan()

    def flush(self) -> None:
        pass


class _NoopSpan:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def span(self, *args, **kwargs) -> "_NoopSpan":
        return _NoopSpan()

    def generation(self, *args, **kwargs) -> "_NoopSpan":
        return _NoopSpan()

    def update(self, *args, **kwargs) -> None:
        pass

    def end(self, *args, **kwargs) -> None:
        pass

    @property
    def id(self) -> str | None:
        return None


class TracingClient:
    """Thin wrapper over the Langfuse SDK."""

    def __init__(self) -> None:
        settings = get_settings()
        self._enabled = settings.has_langfuse
        self._client = None

        if self._enabled:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
                logger.info("Langfuse tracing enabled")
            except Exception as e:
                logger.warning(f"Could not initialize Langfuse: {e}. Tracing disabled.")
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def trace_query(
        self,
        *,
        repo_id: str,
        question: str,
        retrieved_chunks: list[dict],
        answer: str,
        model: str,
        usage: dict,
        latency_ms: int,
        top_k: int,
    ) -> str | None:
        """
        Create a complete trace for a query operation.
        Returns the trace ID if Langfuse is enabled.
        """
        if not self._enabled or self._client is None:
            return None

        try:
            trace = self._client.trace(
                name="codebase-query",
                metadata={
                    "repo_id": repo_id,
                    "top_k": top_k,
                    "model": model,
                    "latency_ms": latency_ms,
                },
                tags=["query", model.split("/")[0]],
            )

            # Retrieval span
            retrieval_span = trace.span(
                name="retrieval",
                input={"question": question, "repo_id": repo_id, "top_k": top_k},
                output={"chunks_retrieved": len(retrieved_chunks)},
            )
            retrieval_span.end()

            # LLM generation record
            context_text = "\n\n".join(
                c.get("content", "") for c in retrieved_chunks[:3]
            )
            generation = trace.generation(
                name="llm-generation",
                model=model,
                input=[
                    {"role": "user", "content": f"Q: {question}\nContext: {context_text[:500]}..."}
                ],
                output=answer,
                usage={
                    "input": usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                    "total": usage.get("total_tokens", 0),
                },
                metadata={"latency_ms": latency_ms},
            )
            generation.end()

            self._client.flush()
            return trace.id

        except Exception as e:
            logger.warning(f"Langfuse trace failed: {e}")
            return None

    def trace_ingestion(
        self,
        *,
        repo_id: str,
        github_url: str,
        file_count: int,
        chunk_count: int,
        vectors_upserted: int,
        duration_seconds: float,
        status: str,
        error: str | None = None,
    ) -> str | None:
        """Create a trace for an ingestion job."""
        if not self._enabled or self._client is None:
            return None

        try:
            trace = self._client.trace(
                name="repo-ingestion",
                metadata={
                    "repo_id": repo_id,
                    "github_url": github_url,
                    "file_count": file_count,
                    "chunk_count": chunk_count,
                    "vectors_upserted": vectors_upserted,
                    "duration_seconds": round(duration_seconds, 2),
                    "status": status,
                    "error": error,
                },
                tags=["ingestion", status],
            )
            self._client.flush()
            return trace.id
        except Exception as e:
            logger.warning(f"Langfuse ingestion trace failed: {e}")
            return None

    def upload_eval_scores(
        self,
        trace_id: str,
        scores: dict[str, float],
    ) -> None:
        """Upload RAGAS evaluation scores to a trace."""
        if not self._enabled or self._client is None:
            return

        try:
            for metric_name, value in scores.items():
                self._client.score(
                    trace_id=trace_id,
                    name=metric_name,
                    value=value,
                    comment=f"RAGAS {metric_name}",
                )
            self._client.flush()
        except Exception as e:
            logger.warning(f"Failed to upload eval scores: {e}")

    def flush(self) -> None:
        if self._client:
            try:
                self._client.flush()
            except Exception:
                pass


# Singleton
_tracer: TracingClient | None = None


def get_tracer() -> TracingClient:
    global _tracer
    if _tracer is None:
        _tracer = TracingClient()
    return _tracer

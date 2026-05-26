"""
Vector retriever.

Queries Upstash Vector for chunks relevant to a user question,
re-ranks by score, and deduplicates overlapping results.
"""

from __future__ import annotations

import logging

from src.ingestion.embedder import get_embedder
from src.models import SourceChunk

logger = logging.getLogger(__name__)


def _deduplicate_chunks(results: list[dict]) -> list[dict]:
    """Remove results with overlapping file + line ranges."""
    seen: list[tuple[str, int, int]] = []  # (file_path, start, end)
    deduplicated = []

    for result in results:
        meta = result.get("metadata", {})
        file_path = meta.get("file_path", "")
        start = meta.get("start_line", 0)
        end = meta.get("end_line", 0)

        overlap = False
        for seen_path, seen_start, seen_end in seen:
            if (
                seen_path == file_path
                and not (end < seen_start or start > seen_end)  # Ranges overlap
            ):
                overlap = True
                break

        if not overlap:
            seen.append((file_path, start, end))
            deduplicated.append(result)

    return deduplicated


def retrieve_chunks(
    question: str,
    repo_id: str,
    top_k: int = 8,
) -> list[SourceChunk]:
    """
    Retrieve the most relevant code chunks for a question.

    Args:
        question: Natural language question about the codebase
        repo_id: Restrict search to this repository
        top_k: Number of chunks to return

    Returns:
        List of SourceChunk objects sorted by relevance score
    """
    embedder = get_embedder()

    # Fetch slightly more than needed to allow for deduplication
    fetch_k = min(top_k * 2, 20)
    raw_results = embedder.query(question=question, repo_id=repo_id, top_k=fetch_k)

    if not raw_results:
        logger.warning(f"No results from vector search for repo_id={repo_id!r}")
        return []

    # Deduplicate overlapping chunks, then trim to top_k
    deduplicated = _deduplicate_chunks(raw_results)[:top_k]

    source_chunks = []
    for result in deduplicated:
        meta = result.get("metadata", {})
        try:
            chunk = SourceChunk(
                file_path=meta.get("file_path", "unknown"),
                node_type=meta.get("node_type", "unknown"),
                node_name=meta.get("node_name", "unknown"),
                start_line=int(meta.get("start_line", 0)),
                end_line=int(meta.get("end_line", 0)),
                language=meta.get("language", "unknown"),
                content=meta.get("content", ""),
                score=float(result.get("score", 0.0)),
            )
            source_chunks.append(chunk)
        except Exception as e:
            logger.warning(f"Could not parse result into SourceChunk: {e}")
            continue

    logger.info(
        f"Retrieved {len(source_chunks)} chunks for repo={repo_id!r} "
        f"(raw={len(raw_results)}, after dedup={len(deduplicated)})"
    )
    return source_chunks

"""
Upstash Vector embedder.

Takes CodeChunk objects and upserts them into Upstash Vector.
Uses Upstash's built-in BGE embedding (bge-large-en-v1.5, 1024 dims)
— no external embedding API needed, saves a service dependency and
stays within free tier constraints.

Free tier limits:
- 10,000 requests/day (each upsert batch = 1 request)
- Batch up to 1,000 vectors per upsert call
"""

from __future__ import annotations

import logging
from typing import Any

from upstash_vector import Index, Vector

from src.config import get_settings
from src.ingestion.chunker import CodeChunk

logger = logging.getLogger(__name__)

# How many chunks to upsert per batch (conservative — well under the 1000 limit)
UPSERT_BATCH_SIZE = 100


def _build_vector_metadata(chunk: CodeChunk) -> dict[str, Any]:
    """Build the metadata dict stored alongside each vector."""
    return {
        "repo_id": chunk.repo_id,
        "file_path": chunk.file_path,
        "language": chunk.language,
        "node_type": chunk.node_type,
        "node_name": chunk.node_name,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "token_count": chunk.token_count,
        # Store the raw content for retrieval (returned with search results)
        "content": chunk.content[:4000],  # Upstash metadata value size limit
    }


class VectorEmbedder:
    """
    Manages upsert and query operations on Upstash Vector.

    Uses text-mode upsert so Upstash handles embedding server-side
    via the built-in BGE model configured at index creation time.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._index = Index(
            url=settings.upstash_vector_url,
            token=settings.upstash_vector_token,
        )

    def upsert_chunks(self, chunks: list[CodeChunk]) -> int:
        """
        Upsert a list of CodeChunk objects into Upstash Vector.

        Returns the total number of vectors successfully upserted.
        """
        if not chunks:
            return 0

        total_upserted = 0

        # Process in batches
        for i in range(0, len(chunks), UPSERT_BATCH_SIZE):
            batch = chunks[i : i + UPSERT_BATCH_SIZE]
            vectors = [
                Vector(
                    id=chunk.id,
                    data=chunk.embedding_text(),  # text → Upstash embeds it
                    metadata=_build_vector_metadata(chunk),
                )
                for chunk in batch
            ]

            try:
                self._index.upsert(vectors=vectors)
                total_upserted += len(batch)
                logger.debug(f"Upserted batch {i // UPSERT_BATCH_SIZE + 1}: {len(batch)} vectors")
            except Exception as e:
                logger.error(
                    f"Failed to upsert batch {i // UPSERT_BATCH_SIZE + 1} "
                    f"(chunks {i}-{i + len(batch)}): {e}",
                    exc_info=True,
                )
                # Continue with next batch rather than failing entirely
                continue

        logger.info(f"Upserted {total_upserted}/{len(chunks)} vectors")
        return total_upserted

    def query(
        self,
        question: str,
        repo_id: str,
        top_k: int = 8,
    ) -> list[dict]:
        """
        Query the vector index for chunks relevant to a question.

        Args:
            question: Natural language question to embed and search
            repo_id: Filter results to this repo namespace
            top_k: Number of results to return

        Returns:
            List of result dicts with score and metadata
        """
        try:
            results = self._index.query(
                data=question,  # text → Upstash embeds it
                top_k=top_k,
                include_metadata=True,
                include_data=False,
                filter=f'repo_id = "{repo_id}"',
            )
            return [
                {
                    "id": r.id,
                    "score": r.score,
                    "metadata": r.metadata or {},
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Vector query failed: {e}", exc_info=True)
            return []

    def delete_repo(self, repo_id: str) -> None:
        """Delete all vectors belonging to a repository (for re-ingestion)."""
        try:
            # Fetch IDs of all vectors for this repo, then delete
            # Upstash Vector supports metadata-filtered delete via query + delete by ID
            results = self._index.query(
                data="placeholder",
                top_k=1000,
                include_metadata=True,
                filter=f'repo_id = "{repo_id}"',
            )
            ids = [r.id for r in results]
            if ids:
                self._index.delete(ids=ids)
                logger.info(f"Deleted {len(ids)} vectors for repo {repo_id}")
        except Exception as e:
            logger.warning(f"Could not delete vectors for repo {repo_id}: {e}")

    def get_repo_stats(self, repo_id: str) -> dict:
        """Return basic stats about vectors for a repo."""
        try:
            info = self._index.info()
            return {
                "total_vectors": info.vector_count,
                "pending_vectors": getattr(info, "pending_vector_count", 0),
            }
        except Exception as e:
            logger.warning(f"Could not get index stats: {e}")
            return {}


# Singleton
_embedder: VectorEmbedder | None = None


def get_embedder() -> VectorEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = VectorEmbedder()
    return _embedder

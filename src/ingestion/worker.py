"""
Ingestion worker.

Orchestrates the full ingestion pipeline:
  1. Clone the GitHub repository (shallow, depth=1)
  2. Walk the file tree, filter by language/size
  3. Parse each file via tree-sitter AST chunker
  4. Upsert chunks into Upstash Vector
  5. Update repository status in SQLite

This module runs in two modes:
  - QStash mode: called as a FastAPI endpoint (POST /webhooks/qstash)
  - Standalone mode: called directly (e.g., from scripts/seed_repo.py)
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from src.ingestion.chunker import get_chunker
from src.ingestion.cloner import cleanup_repo, clone_repo, walk_repo_files
from src.ingestion.embedder import get_embedder
from src.models import RepoStatus

logger = logging.getLogger(__name__)


class IngestionResult:
    """Result of a completed ingestion job."""

    def __init__(
        self,
        repo_id: str,
        status: RepoStatus,
        file_count: int = 0,
        chunk_count: int = 0,
        vectors_upserted: int = 0,
        duration_seconds: float = 0.0,
        error: str | None = None,
    ):
        self.repo_id = repo_id
        self.status = status
        self.file_count = file_count
        self.chunk_count = chunk_count
        self.vectors_upserted = vectors_upserted
        self.duration_seconds = duration_seconds
        self.error = error

    def __repr__(self) -> str:
        return (
            f"IngestionResult(repo_id={self.repo_id!r}, status={self.status}, "
            f"files={self.file_count}, chunks={self.chunk_count}, "
            f"vectors={self.vectors_upserted}, duration={self.duration_seconds:.1f}s)"
        )


async def run_ingestion(
    repo_id: str,
    github_url: str,
    branch: str = "main",
    status_callback: object | None = None,
) -> IngestionResult:
    """
    Run the full ingestion pipeline for a repository.

    Args:
        repo_id: Unique identifier for this repository
        github_url: GitHub repository URL
        branch: Branch to ingest (default: "main")
        status_callback: Optional async callable(repo_id, status, **kwargs) for status updates

    Returns:
        IngestionResult with final counts and status
    """
    start_time = time.monotonic()
    repo_path: Path | None = None

    async def _update_status(status: RepoStatus, **kwargs) -> None:
        if status_callback:
            try:
                await status_callback(repo_id, status, **kwargs)
            except Exception as e:
                logger.warning(f"Status callback failed: {e}")

    try:
        # ----------------------------------------------------------------
        # Phase 1: Clone
        # ----------------------------------------------------------------
        logger.info(f"[{repo_id}] Starting ingestion for {github_url}")
        await _update_status(RepoStatus.CLONING)

        # Clone is synchronous (subprocess) — run in thread pool
        loop = asyncio.get_event_loop()
        repo_path, actual_branch = await loop.run_in_executor(
            None, clone_repo, github_url, branch
        )

        # ----------------------------------------------------------------
        # Phase 2: Walk files
        # ----------------------------------------------------------------
        await _update_status(RepoStatus.CHUNKING)

        file_list = await loop.run_in_executor(
            None, walk_repo_files, repo_path
        )
        file_count = len(file_list)
        logger.info(f"[{repo_id}] Found {file_count} files to process")

        if file_count == 0:
            await _update_status(
                RepoStatus.FAILED,
                error="No supported source files found in repository",
            )
            return IngestionResult(
                repo_id=repo_id,
                status=RepoStatus.FAILED,
                error="No supported source files found",
            )

        # ----------------------------------------------------------------
        # Phase 3: Chunk
        # ----------------------------------------------------------------
        chunker = get_chunker()
        all_chunks = []

        def _chunk_all() -> list:
            chunks = []
            for rel_path, content, language in file_list:
                try:
                    file_chunks = chunker.chunk_file(repo_id, rel_path, content, language)
                    chunks.extend(file_chunks)
                except Exception as e:
                    logger.warning(f"Chunking failed for {rel_path}: {e}")
            return chunks

        all_chunks = await loop.run_in_executor(None, _chunk_all)
        chunk_count = len(all_chunks)
        logger.info(f"[{repo_id}] Produced {chunk_count} chunks from {file_count} files")

        # ----------------------------------------------------------------
        # Phase 4: Embed + Upsert
        # ----------------------------------------------------------------
        await _update_status(
            RepoStatus.EMBEDDING,
            file_count=file_count,
            chunk_count=chunk_count,
        )

        embedder = get_embedder()
        vectors_upserted = await loop.run_in_executor(
            None, embedder.upsert_chunks, all_chunks
        )

        # ----------------------------------------------------------------
        # Phase 5: Done
        # ----------------------------------------------------------------
        duration = time.monotonic() - start_time
        logger.info(
            f"[{repo_id}] Ingestion complete: {file_count} files, "
            f"{chunk_count} chunks, {vectors_upserted} vectors in {duration:.1f}s"
        )

        await _update_status(
            RepoStatus.READY,
            file_count=file_count,
            chunk_count=chunk_count,
        )

        return IngestionResult(
            repo_id=repo_id,
            status=RepoStatus.READY,
            file_count=file_count,
            chunk_count=chunk_count,
            vectors_upserted=vectors_upserted,
            duration_seconds=duration,
        )

    except Exception as e:
        duration = time.monotonic() - start_time
        error_msg = str(e)
        logger.error(
            f"[{repo_id}] Ingestion failed after {duration:.1f}s: {error_msg}",
            exc_info=True,
        )
        await _update_status(RepoStatus.FAILED, error=error_msg)
        return IngestionResult(
            repo_id=repo_id,
            status=RepoStatus.FAILED,
            error=error_msg,
            duration_seconds=duration,
        )
    finally:
        # Always clean up the cloned repo from disk
        if repo_path and repo_path.exists():
            cleanup_repo(repo_path)

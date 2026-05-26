"""
Repos API routes.

POST /repos          — Submit a GitHub repo for ingestion
GET  /repos          — List all repos
GET  /repos/{id}     — Get status of a specific repo
DELETE /repos/{id}   — Delete a repo and its vectors
"""

from __future__ import annotations

import asyncio
import logging
import re

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from src.config import get_settings
from src.db.repos import create_repo, get_repo, list_repos, record_to_response, update_repo_status
from src.ingestion.worker import run_ingestion
from src.models import (
    RepoListResponse,
    RepoStatus,
    RepoStatusResponse,
    SubmitRepoRequest,
    SubmitRepoResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/repos", tags=["repositories"])

GITHUB_URL_PATTERN = re.compile(
    r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?$"
)


def _validate_github_url(url: str) -> str:
    """Strip trailing .git, validate format."""
    clean = url.strip().rstrip("/").removesuffix(".git")
    if not GITHUB_URL_PATTERN.match(clean):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid GitHub URL: {url!r}. Expected format: https://github.com/owner/repo",
        )
    return clean


async def _ingestion_status_callback(
    repo_id: str,
    new_status: RepoStatus,
    file_count: int | None = None,
    chunk_count: int | None = None,
    error: str | None = None,
) -> None:
    """Callback used by the ingestion worker to update DB status."""
    await update_repo_status(
        repo_id=repo_id,
        status=new_status,
        file_count=file_count,
        chunk_count=chunk_count,
        error=error,
    )


async def _run_ingestion_background(
    repo_id: str,
    github_url: str,
    branch: str,
) -> None:
    """Wrapper for background task — swallows exceptions (already logged in worker)."""
    try:
        result = await run_ingestion(
            repo_id=repo_id,
            github_url=github_url,
            branch=branch,
            status_callback=_ingestion_status_callback,
        )
        logger.info(f"Background ingestion complete: {result}")
    except Exception as e:
        logger.error(f"Background ingestion error for {repo_id}: {e}", exc_info=True)


@router.post(
    "",
    response_model=SubmitRepoResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a GitHub repository for ingestion",
)
async def submit_repo(
    body: SubmitRepoRequest,
    background_tasks: BackgroundTasks,
) -> SubmitRepoResponse:
    """
    Submit a GitHub repository URL for asynchronous ingestion.

    The repository will be cloned, chunked by AST structure, embedded,
    and stored in the vector index. Poll /repos/{repo_id} for status.
    """
    settings = get_settings()
    clean_url = _validate_github_url(body.url)

    # Create DB record
    record = await create_repo(url=clean_url, branch=body.branch)
    repo_id = record.id

    # Run ingestion as a FastAPI background task (non-blocking response)
    background_tasks.add_task(
        _run_ingestion_background,
        repo_id=repo_id,
        github_url=clean_url,
        branch=body.branch,
    )

    logger.info(f"Queued ingestion for {clean_url} → repo_id={repo_id}")

    return SubmitRepoResponse(
        repo_id=repo_id,
        status=RepoStatus.QUEUED,
        message=f"Repository queued for ingestion. Poll /repos/{repo_id} for status.",
    )


@router.get(
    "",
    response_model=RepoListResponse,
    summary="List all repositories",
)
async def list_all_repos() -> RepoListResponse:
    """Return all repositories with their current ingestion status."""
    records = await list_repos(limit=50)
    return RepoListResponse(
        repos=[record_to_response(r) for r in records],
        total=len(records),
    )


@router.get(
    "/{repo_id}",
    response_model=RepoStatusResponse,
    summary="Get repository ingestion status",
)
async def get_repo_status(repo_id: str) -> RepoStatusResponse:
    """Get the current ingestion status and statistics for a repository."""
    record = await get_repo(repo_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository {repo_id!r} not found",
        )
    return record_to_response(record)


@router.delete(
    "/{repo_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a repository and its vectors",
)
async def delete_repo(repo_id: str) -> None:
    """Delete a repository record and all its associated vectors from the index."""
    from sqlalchemy import delete as sql_delete
    from src.db.database import RepoRecord, get_session_factory
    from src.ingestion.embedder import get_embedder

    record = await get_repo(repo_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository {repo_id!r} not found",
        )

    # Delete vectors (best-effort)
    try:
        embedder = get_embedder()
        embedder.delete_repo(repo_id)
    except Exception as e:
        logger.warning(f"Could not delete vectors for {repo_id}: {e}")

    # Delete DB record
    factory = get_session_factory()
    async with factory() as session:
        await session.execute(
            sql_delete(RepoRecord).where(RepoRecord.id == repo_id)
        )
        await session.commit()

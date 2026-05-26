"""
Repo CRUD operations.

Thin data-access layer over the RepoRecord ORM model.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.database import RepoRecord, get_session_factory
from src.models import RepoStatus, RepoStatusResponse


async def create_repo(url: str, branch: str = "main") -> RepoRecord:
    """Create a new repo record with status=queued."""
    factory = get_session_factory()
    async with factory() as session:
        record = RepoRecord(
            id=str(uuid.uuid4()),
            url=url,
            branch=branch,
            status=RepoStatus.QUEUED,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)
        return record


async def get_repo(repo_id: str) -> RepoRecord | None:
    """Fetch a repo record by ID."""
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(RepoRecord).where(RepoRecord.id == repo_id)
        )
        return result.scalar_one_or_none()


async def list_repos(limit: int = 50) -> list[RepoRecord]:
    """List all repo records, newest first."""
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(RepoRecord)
            .order_by(RepoRecord.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


async def update_repo_status(
    repo_id: str,
    status: RepoStatus,
    file_count: int | None = None,
    chunk_count: int | None = None,
    error: str | None = None,
) -> None:
    """Update a repo's status and optional counters."""
    factory = get_session_factory()
    values: dict = {
        "status": status,
        "updated_at": datetime.now(timezone.utc),
    }
    if file_count is not None:
        values["file_count"] = file_count
    if chunk_count is not None:
        values["chunk_count"] = chunk_count
    if error is not None:
        values["error"] = error

    async with factory() as session:
        await session.execute(
            update(RepoRecord).where(RepoRecord.id == repo_id).values(**values)
        )
        await session.commit()


def record_to_response(record: RepoRecord) -> RepoStatusResponse:
    """Convert an ORM record to the API response schema."""
    return RepoStatusResponse(
        repo_id=record.id,
        url=record.url,
        branch=record.branch,
        status=RepoStatus(record.status),
        file_count=record.file_count,
        chunk_count=record.chunk_count,
        error=record.error,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )

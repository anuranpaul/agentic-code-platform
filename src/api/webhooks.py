"""
QStash webhook receiver.

QStash calls POST /webhooks/qstash when it delivers an ingestion job.
This endpoint verifies the request signature and triggers the ingestion worker.

In local development (APP_ENV=development), signature verification is skipped.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, status

from src.config import get_settings
from src.db.repos import create_repo, get_repo, update_repo_status
from src.ingestion.worker import run_ingestion
from src.models import RepoStatus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhooks", tags=["webhooks"])


async def _verify_qstash_signature(request: Request) -> dict:
    """
    Verify that the request came from QStash using the signing key.
    Returns the decoded body as a dict.
    """
    settings = get_settings()

    if settings.is_development:
        logger.debug("Skipping QStash signature verification in development mode")
        try:
            return await request.json()
        except Exception:
            return {}

    try:
        from qstash import Receiver

        receiver = Receiver(
            current_signing_key=settings.qstash_current_signing_key,
            next_signing_key=settings.qstash_next_signing_key,
        )
        body_bytes = await request.body()
        signature = request.headers.get("upstash-signature", "")

        body_str = body_bytes.decode("utf-8")
        receiver.verify(
            body=body_str,
            signature=signature,
            url=str(request.url),
        )
        import json
        return json.loads(body_str)

    except Exception as e:
        logger.warning(f"QStash signature verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid QStash signature",
        )


async def _ingestion_status_callback(
    repo_id: str,
    new_status: RepoStatus,
    file_count: int | None = None,
    chunk_count: int | None = None,
    error: str | None = None,
) -> None:
    await update_repo_status(
        repo_id=repo_id,
        status=new_status,
        file_count=file_count,
        chunk_count=chunk_count,
        error=error,
    )


@router.post(
    "/qstash",
    status_code=status.HTTP_200_OK,
    summary="QStash webhook endpoint for async ingestion",
    include_in_schema=False,  # Hide from public API docs
)
async def qstash_webhook(request: Request) -> dict:
    """
    Receives async ingestion jobs from QStash.

    Expected payload:
    {
        "repo_id": "...",
        "github_url": "https://github.com/...",
        "branch": "main"
    }

    QStash will retry on non-2xx responses (up to 3 times by default).
    """
    body = await _verify_qstash_signature(request)

    repo_id = body.get("repo_id")
    github_url = body.get("github_url")
    branch = body.get("branch", "main")

    if not repo_id or not github_url:
        logger.error(f"Invalid QStash payload: {body}")
        # Return 200 to prevent QStash from retrying malformed messages
        return {"error": "Invalid payload — missing repo_id or github_url"}

    logger.info(f"QStash webhook received for repo_id={repo_id}, url={github_url}")

    # Run ingestion synchronously within the webhook (QStash waits for 200)
    # For very large repos, this may approach Vercel's 30s timeout —
    # in that case, use BackgroundTasks and return 200 immediately
    result = await run_ingestion(
        repo_id=repo_id,
        github_url=github_url,
        branch=branch,
        status_callback=_ingestion_status_callback,
    )

    return {
        "repo_id": repo_id,
        "status": result.status,
        "file_count": result.file_count,
        "chunk_count": result.chunk_count,
        "vectors_upserted": result.vectors_upserted,
        "duration_seconds": round(result.duration_seconds, 2),
    }

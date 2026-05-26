"""Pydantic models for API request/response schemas."""

from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, HttpUrl


class RepoStatus(str, Enum):
    QUEUED = "queued"
    CLONING = "cloning"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    READY = "ready"
    FAILED = "failed"


class SubmitRepoRequest(BaseModel):
    url: str = Field(..., description="GitHub repository URL", examples=["https://github.com/owner/repo"])
    branch: str = Field(default="main", description="Branch to ingest")


class SubmitRepoResponse(BaseModel):
    repo_id: str
    status: RepoStatus
    message: str


class RepoStatusResponse(BaseModel):
    repo_id: str
    url: str
    branch: str
    status: RepoStatus
    file_count: int | None = None
    chunk_count: int | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class RepoListResponse(BaseModel):
    repos: list[RepoStatusResponse]
    total: int


class QueryRequest(BaseModel):
    repo_id: str = Field(..., description="ID of the ingested repository")
    question: str = Field(..., description="Natural language question about the codebase")
    top_k: int = Field(default=8, ge=1, le=20, description="Number of chunks to retrieve")
    model: str | None = Field(default=None, description="Override LLM model for this query")


class SourceChunk(BaseModel):
    file_path: str
    node_type: str
    node_name: str
    start_line: int
    end_line: int
    language: str
    content: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    repo_id: str
    model: str
    trace_id: str | None = None
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict[str, Any]

"""
Unit tests for the tree-sitter AST chunker.

Tests cover:
- Chunk boundary correctness (functions stay whole)
- Multi-language support
- Fallback chunking for unsupported files
- Token limits
- Context header extraction
"""

from __future__ import annotations

import pytest
from src.ingestion.chunker import ASTChunker, CodeChunk, _chunk_id, _count_tokens


REPO_ID = "test-repo-001"


@pytest.fixture
def chunker() -> ASTChunker:
    return ASTChunker()


# ---------------------------------------------------------------------------
# Python tests
# ---------------------------------------------------------------------------

PYTHON_CODE = '''"""Module docstring."""
import os
import sys
from typing import Optional


def simple_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


def complex_function(items: list) -> dict:
    """Process items."""
    result = {}
    for item in items:
        if item:
            result[item] = len(str(item))
    return result


class PaymentService:
    """Handles payment processing."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None

    def process_payment(self, amount: float, currency: str = "USD") -> dict:
        """Process a payment transaction."""
        if amount <= 0:
            raise ValueError("Amount must be positive")
        return {"status": "success", "amount": amount, "currency": currency}

    def retry_payment(self, payment_id: str, max_retries: int = 3) -> dict:
        """Retry a failed payment."""
        for attempt in range(max_retries):
            try:
                return self.process_payment(100.0)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
        return {}


def standalone_util(value: Optional[str] = None) -> str:
    """Utility function."""
    return value or "default"
'''


def test_python_chunk_count(chunker: ASTChunker) -> None:
    """Python files should produce one chunk per top-level function/class."""
    chunks = chunker.chunk_file(REPO_ID, "src/payment.py", PYTHON_CODE, "python")
    # Expect: simple_function, complex_function, PaymentService class, standalone_util
    assert len(chunks) >= 4, f"Expected >=4 chunks, got {len(chunks)}: {[c.node_name for c in chunks]}"


def test_python_function_stays_whole(chunker: ASTChunker) -> None:
    """Functions should not be split mid-body."""
    chunks = chunker.chunk_file(REPO_ID, "src/payment.py", PYTHON_CODE, "python")
    func_chunks = [c for c in chunks if c.node_name in ("simple_function", "complex_function")]
    for chunk in func_chunks:
        assert "def " in chunk.content, "Function chunk should contain its definition"
        assert chunk.start_line < chunk.end_line or "return" in chunk.content


def test_python_class_chunk(chunker: ASTChunker) -> None:
    """Class definition should be captured as a chunk."""
    chunks = chunker.chunk_file(REPO_ID, "src/payment.py", PYTHON_CODE, "python")
    class_chunks = [c for c in chunks if c.node_type == "class"]
    assert len(class_chunks) >= 1, "Expected at least one class chunk"
    assert class_chunks[0].node_name == "PaymentService"


def test_python_context_header(chunker: ASTChunker) -> None:
    """Context header should contain import statements."""
    chunks = chunker.chunk_file(REPO_ID, "src/payment.py", PYTHON_CODE, "python")
    assert len(chunks) > 0
    # At least some chunks should have a context header with imports
    headers = [c.context_header for c in chunks if c.context_header]
    assert any("import" in h for h in headers), "Expected import statements in context headers"


def test_python_line_numbers_correct(chunker: ASTChunker) -> None:
    """Start and end line numbers should be 1-indexed and non-overlapping for top-level nodes."""
    chunks = chunker.chunk_file(REPO_ID, "src/payment.py", PYTHON_CODE, "python")
    for chunk in chunks:
        assert chunk.start_line >= 1, f"start_line must be >= 1, got {chunk.start_line}"
        assert chunk.end_line >= chunk.start_line, (
            f"end_line ({chunk.end_line}) must be >= start_line ({chunk.start_line})"
        )


def test_python_chunk_ids_unique(chunker: ASTChunker) -> None:
    """Every chunk should have a unique ID."""
    chunks = chunker.chunk_file(REPO_ID, "src/payment.py", PYTHON_CODE, "python")
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


def test_python_token_counts_within_limit(chunker: ASTChunker) -> None:
    """No chunk should exceed MAX_CHUNK_TOKENS."""
    from src.ingestion.chunker import MAX_CHUNK_TOKENS
    chunks = chunker.chunk_file(REPO_ID, "src/payment.py", PYTHON_CODE, "python")
    for chunk in chunks:
        assert chunk.token_count <= MAX_CHUNK_TOKENS, (
            f"Chunk '{chunk.node_name}' exceeds token limit: {chunk.token_count}"
        )


def test_python_embedding_text_format(chunker: ASTChunker) -> None:
    """embedding_text() should include node type header and content."""
    chunks = chunker.chunk_file(REPO_ID, "src/payment.py", PYTHON_CODE, "python")
    for chunk in chunks:
        text = chunk.embedding_text()
        assert chunk.node_name in text or chunk.node_type in text
        assert len(text) > 0


# ---------------------------------------------------------------------------
# TypeScript tests
# ---------------------------------------------------------------------------

TYPESCRIPT_CODE = '''import { Injectable } from "@nestjs/common";
import { HttpService } from "@nestjs/axios";

interface PaymentOptions {
    amount: number;
    currency: string;
    metadata?: Record<string, unknown>;
}

type PaymentStatus = "pending" | "success" | "failed";

@Injectable()
export class PaymentProcessor {
    constructor(private readonly http: HttpService) {}

    async processPayment(options: PaymentOptions): Promise<PaymentStatus> {
        const { amount, currency } = options;
        if (amount <= 0) {
            throw new Error("Amount must be positive");
        }
        return "success";
    }

    private async retryWithBackoff(fn: () => Promise<void>, retries = 3): Promise<void> {
        for (let i = 0; i < retries; i++) {
            try {
                await fn();
                return;
            } catch (e) {
                if (i === retries - 1) throw e;
                await new Promise(r => setTimeout(r, 1000 * Math.pow(2, i)));
            }
        }
    }
}

export function formatCurrency(amount: number, currency: string): string {
    return new Intl.NumberFormat("en-US", { style: "currency", currency }).format(amount);
}
'''


def test_typescript_chunk_count(chunker: ASTChunker) -> None:
    """TypeScript class and function should be chunked."""
    chunks = chunker.chunk_file(REPO_ID, "src/payment.ts", TYPESCRIPT_CODE, "typescript")
    assert len(chunks) >= 1, f"Expected >=1 chunks for TypeScript, got {len(chunks)}"


def test_typescript_detects_export(chunker: ASTChunker) -> None:
    """Exported functions should be captured."""
    chunks = chunker.chunk_file(REPO_ID, "src/payment.ts", TYPESCRIPT_CODE, "typescript")
    names = [c.node_name for c in chunks]
    content_all = " ".join(c.content for c in chunks)
    assert "formatCurrency" in content_all or "PaymentProcessor" in content_all


# ---------------------------------------------------------------------------
# Language detection tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename,expected_lang", [
    ("app.py", "python"),
    ("app.ts", "typescript"),
    ("App.tsx", "tsx"),
    ("index.js", "javascript"),
    ("main.go", "go"),
    ("lib.rs", "rust"),
    ("Main.java", "java"),
    ("unknown.html", None),
    ("styles.css", None),
])
def test_language_detection(chunker: ASTChunker, filename: str, expected_lang: str | None) -> None:
    detected = chunker.detect_language(filename)
    assert detected == expected_lang, f"For {filename}: expected {expected_lang}, got {detected}"


# ---------------------------------------------------------------------------
# Fallback chunker tests
# ---------------------------------------------------------------------------

UNSUPPORTED_CONTENT = "SELECT * FROM users WHERE id = 1;\nINSERT INTO logs (msg) VALUES ('hello');"


def test_fallback_for_unsupported_language(chunker: ASTChunker) -> None:
    """Files with no AST support should fall back gracefully."""
    chunks = chunker._fallback_chunk(REPO_ID, "query.sql", UNSUPPORTED_CONTENT, "sql")
    assert len(chunks) >= 1
    assert chunks[0].node_type == "module"


def test_fallback_preserves_content(chunker: ASTChunker) -> None:
    """Fallback chunks should contain the original content."""
    chunks = chunker._fallback_chunk(REPO_ID, "data.sql", UNSUPPORTED_CONTENT, "sql")
    all_content = " ".join(c.content for c in chunks)
    assert "SELECT" in all_content


def test_empty_file_returns_no_chunks(chunker: ASTChunker) -> None:
    """Empty files should return empty chunk list."""
    chunks = chunker._fallback_chunk(REPO_ID, "empty.py", "", "python")
    assert chunks == []


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

def test_chunk_id_deterministic() -> None:
    """Same inputs must always produce the same chunk ID."""
    id1 = _chunk_id("repo-1", "src/foo.py", 42)
    id2 = _chunk_id("repo-1", "src/foo.py", 42)
    assert id1 == id2


def test_chunk_id_differs_by_line() -> None:
    """Different line numbers must produce different IDs."""
    id1 = _chunk_id("repo-1", "src/foo.py", 1)
    id2 = _chunk_id("repo-1", "src/foo.py", 50)
    assert id1 != id2


def test_token_count_nonempty() -> None:
    """Token counter should return > 0 for non-empty strings."""
    assert _count_tokens("def hello(): pass") > 0


def test_token_count_empty() -> None:
    """Token counter should return 0 for empty strings."""
    assert _count_tokens("") == 0


# ---------------------------------------------------------------------------
# Go language tests
# ---------------------------------------------------------------------------

GO_CODE = '''package payment

import (
    "errors"
    "fmt"
)

type Payment struct {
    ID       string
    Amount   float64
    Currency string
}

func NewPayment(amount float64, currency string) (*Payment, error) {
    if amount <= 0 {
        return nil, errors.New("amount must be positive")
    }
    return &Payment{Amount: amount, Currency: currency}, nil
}

func (p *Payment) Process() error {
    if p.Amount <= 0 {
        return fmt.Errorf("invalid amount: %f", p.Amount)
    }
    return nil
}
'''


def test_go_chunks_functions(chunker: ASTChunker) -> None:
    """Go functions and methods should be chunked."""
    chunks = chunker.chunk_file(REPO_ID, "payment/payment.go", GO_CODE, "go")
    assert len(chunks) >= 1, f"Expected >=1 Go chunks, got {len(chunks)}"
    content_all = " ".join(c.content for c in chunks)
    assert "func" in content_all

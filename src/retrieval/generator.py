"""
LLM response generator.

Builds a structured prompt from retrieved code chunks and generates
a sourced answer using Groq (Llama 3.3 70B) or OpenAI (GPT-4o-mini).

The system prompt enforces:
- Citation of file paths and line numbers for every claim
- Explicit "I don't know" when context is insufficient
- Precision over recall (avoid hallucination)
"""

from __future__ import annotations

import logging

from groq import Groq
from openai import OpenAI

from src.config import get_settings
from src.models import SourceChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert code analyst. You are given relevant source code excerpts from a repository and must answer questions about the codebase accurately.

Rules:
1. Answer ONLY based on the code context provided — do not invent or assume implementations not shown.
2. For every specific claim, cite the source using the format: [path/to/file.py:L{start}-L{end}]
3. If the provided context is insufficient to answer confidently, say explicitly: "The provided context doesn't contain enough information to answer this question. You may need to look at [specific area]."
4. Be precise and technical. Use correct terminology for the detected programming language.
5. When referencing function names, class names, or variables, use backticks (e.g., `process_payment`).
"""


def _format_chunks_for_prompt(chunks: list[SourceChunk]) -> str:
    """Format retrieved chunks into the prompt context block."""
    if not chunks:
        return "(No relevant code context found)"

    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"## [{i}] {chunk.file_path}:L{chunk.start_line}-L{chunk.end_line} ({chunk.language} {chunk.node_type}: {chunk.node_name})"
        parts.append(f"{header}\n```{chunk.language}\n{chunk.content}\n```")

    return "\n\n".join(parts)


def _build_user_message(repo_name: str, question: str, chunks: list[SourceChunk]) -> str:
    context = _format_chunks_for_prompt(chunks)
    return f"""Repository: {repo_name}

## Retrieved Code Context

{context}

## Question

{question}

Please answer the question using the code context above. Cite sources for all specific claims."""


class LLMGenerator:
    """Wraps Groq and OpenAI clients for answer generation."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._groq: Groq | None = None
        self._openai: OpenAI | None = None

    def _get_groq(self) -> Groq:
        if self._groq is None:
            self._groq = Groq(api_key=self._settings.groq_api_key)
        return self._groq

    def _get_openai(self) -> OpenAI:
        if self._openai is None:
            self._openai = OpenAI(api_key=self._settings.openai_api_key)
        return self._openai

    def generate(
        self,
        question: str,
        chunks: list[SourceChunk],
        repo_name: str = "this repository",
        model: str | None = None,
        provider: str | None = None,
    ) -> tuple[str, str, dict]:
        """
        Generate an answer from retrieved chunks.

        Returns:
            Tuple of (answer_text, model_used, usage_stats)
        """
        settings = self._settings
        effective_provider = provider or settings.llm_provider
        effective_model = model or settings.llm_model

        user_message = _build_user_message(repo_name, question, chunks)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        try:
            if effective_provider == "groq":
                return self._call_groq(messages, effective_model)
            else:
                return self._call_openai(messages, effective_model)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            raise

    def _call_groq(
        self,
        messages: list[dict],
        model: str,
    ) -> tuple[str, str, dict]:
        client = self._get_groq()
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.1,    # Low temp for factual code answers
            max_tokens=2048,
        )
        answer = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }
        return answer, f"groq/{model}", usage

    def _call_openai(
        self,
        messages: list[dict],
        model: str = "gpt-4o-mini",
    ) -> tuple[str, str, dict]:
        client = self._get_openai()
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.1,
            max_tokens=2048,
        )
        answer = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }
        return answer, f"openai/{model}", usage


# Singleton
_generator: LLMGenerator | None = None


def get_generator() -> LLMGenerator:
    global _generator
    if _generator is None:
        _generator = LLMGenerator()
    return _generator

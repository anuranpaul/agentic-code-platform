"""
Tree-sitter AST-aware code chunker.

Instead of splitting code every N characters, this module parses the AST
and extracts semantically complete units (functions, classes, methods).
A function always stays whole — this directly improves retrieval quality.

Supported languages: Python, TypeScript, JavaScript, Go, Rust
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import tiktoken
from tree_sitter import Language, Node, Parser
from tree_sitter_languages import get_language, get_parser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# File extension → tree-sitter language name
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".rb": "ruby",
}

# AST node types that represent top-level "chunk boundaries" per language
CHUNK_NODE_TYPES: dict[str, list[str]] = {
    "python": [
        "function_definition",
        "async_function_definition",
        "class_definition",
        "decorated_definition",
    ],
    "typescript": [
        "function_declaration",
        "function_expression",
        "arrow_function",
        "class_declaration",
        "class_expression",
        "method_definition",
        "export_statement",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
    ],
    "tsx": [
        "function_declaration",
        "function_expression",
        "arrow_function",
        "class_declaration",
        "class_expression",
        "method_definition",
        "export_statement",
        "jsx_element",
        "interface_declaration",
        "type_alias_declaration",
    ],
    "javascript": [
        "function_declaration",
        "function_expression",
        "arrow_function",
        "class_declaration",
        "class_expression",
        "method_definition",
        "export_statement",
    ],
    "go": [
        "function_declaration",
        "method_declaration",
        "type_declaration",
        "const_declaration",
        "var_declaration",
    ],
    "rust": [
        "function_item",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "mod_item",
        "use_declaration",
        "type_item",
        "const_item",
    ],
    "java": [
        "class_declaration",
        "interface_declaration",
        "method_declaration",
        "enum_declaration",
        "constructor_declaration",
    ],
    "cpp": [
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "namespace_definition",
    ],
    "c": [
        "function_definition",
        "struct_specifier",
    ],
    "ruby": [
        "method",
        "singleton_method",
        "class",
        "module",
    ],
}

# Token limits per chunk (leave room for context header + prompt)
MAX_CHUNK_TOKENS = 1_500
MIN_CHUNK_TOKENS = 50

# Tokenizer (used for counting, model-agnostic approximation)
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CodeChunk:
    """A semantically complete code unit extracted by AST parsing."""

    id: str                  # Deterministic hash: sha256(repo_id + file_path + start_line)
    repo_id: str
    file_path: str           # Relative path within the repo
    language: str
    node_type: str           # "function" | "class" | "method" | "module" | "other"
    node_name: str           # e.g., "process_payment", "PaymentService"
    start_line: int
    end_line: int
    content: str             # Full source text of the chunk
    context_header: str      # Imports + module docstring (prepended for embedding)
    token_count: int

    def embedding_text(self) -> str:
        """Formatted text sent to the embedding model."""
        parts = []
        if self.context_header:
            parts.append(self.context_header)
        parts.append(f"# {self.node_type}: {self.node_name}")
        parts.append(self.content)
        return "\n\n".join(parts)

    def display_location(self) -> str:
        return f"{self.file_path}:L{self.start_line}-L{self.end_line}"


def _chunk_id(repo_id: str, file_path: str, start_line: int) -> str:
    raw = f"{repo_id}:{file_path}:{start_line}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text, disallowed_special=()))


# ---------------------------------------------------------------------------
# Language-specific helpers
# ---------------------------------------------------------------------------


def _get_node_name(node: Node, source: bytes, language: str) -> str:
    """Extract the identifier/name of an AST node."""
    # Walk immediate children for a named identifier
    for child in node.children:
        if child.type in ("identifier", "name", "type_identifier", "field_identifier"):
            return source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    # Fallback: extract first token on first line
    first_line = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace").split("\n")[0]
    return first_line[:60].strip()


def _classify_node_type(ts_node_type: str) -> str:
    """Map tree-sitter node type string to our simplified taxonomy."""
    if "class" in ts_node_type or "struct" in ts_node_type or "impl" in ts_node_type:
        return "class"
    if "function" in ts_node_type or "method" in ts_node_type or "arrow" in ts_node_type:
        return "function"
    if "interface" in ts_node_type or "trait" in ts_node_type:
        return "interface"
    if "type" in ts_node_type or "enum" in ts_node_type:
        return "type"
    return "other"


def _extract_context_header(root: Node, source: bytes, language: str) -> str:
    """
    Extract the import/use/require statements and module-level docstring
    from the top of the file to use as context for each chunk.
    """
    import_types: dict[str, list[str]] = {
        "python": ["import_statement", "import_from_statement", "expression_statement"],
        "typescript": ["import_statement", "import_declaration"],
        "tsx": ["import_statement", "import_declaration"],
        "javascript": ["import_statement", "import_declaration", "require_call"],
        "go": ["import_declaration", "package_clause"],
        "rust": ["use_declaration", "extern_crate_declaration"],
        "java": ["import_declaration", "package_declaration"],
        "cpp": ["preproc_include", "using_declaration"],
        "c": ["preproc_include"],
        "ruby": ["require_relative", "require"],
    }

    target_types = import_types.get(language, [])
    header_lines: list[str] = []
    max_header_tokens = 300

    for child in root.children:
        if child.type in target_types:
            text = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
            header_lines.append(text)
            if _count_tokens("\n".join(header_lines)) > max_header_tokens:
                break

    return "\n".join(header_lines)


# ---------------------------------------------------------------------------
# Core chunker
# ---------------------------------------------------------------------------


class ASTChunker:
    """
    Parses source files via tree-sitter and yields CodeChunk objects,
    one per top-level function/class/method boundary.
    """

    def __init__(self) -> None:
        self._parsers: dict[str, Parser] = {}
        self._languages: dict[str, Language] = {}

    def _get_parser(self, language: str) -> Parser | None:
        if language not in self._parsers:
            try:
                self._parsers[language] = get_parser(language)
                self._languages[language] = get_language(language)
            except Exception as e:
                logger.warning(f"Could not load parser for language '{language}': {e}")
                return None
        return self._parsers[language]

    def detect_language(self, file_path: str | Path) -> str | None:
        """Return the tree-sitter language name for a file, or None if unsupported."""
        ext = Path(file_path).suffix.lower()
        return EXTENSION_TO_LANGUAGE.get(ext)

    def chunk_file(
        self,
        repo_id: str,
        file_path: str,
        content: str,
        language: str,
    ) -> list[CodeChunk]:
        """
        Parse a single file and return a list of CodeChunk objects.
        Falls back to whole-file chunking if the AST parse fails.
        """
        parser = self._get_parser(language)
        if parser is None:
            return self._fallback_chunk(repo_id, file_path, content, language)

        try:
            source = content.encode("utf-8")
            tree = parser.parse(source)
        except Exception as e:
            logger.warning(f"Parse failed for {file_path}: {e}")
            return self._fallback_chunk(repo_id, file_path, content, language)

        root = tree.root_node
        context_header = _extract_context_header(root, source, language)
        target_types = set(CHUNK_NODE_TYPES.get(language, []))

        chunks: list[CodeChunk] = []
        self._walk_node(
            node=root,
            source=source,
            repo_id=repo_id,
            file_path=file_path,
            language=language,
            target_types=target_types,
            context_header=context_header,
            chunks=chunks,
            depth=0,
        )

        # If no AST nodes matched, treat the whole file as one chunk
        if not chunks:
            return self._fallback_chunk(repo_id, file_path, content, language)

        return chunks

    def _walk_node(
        self,
        node: Node,
        source: bytes,
        repo_id: str,
        file_path: str,
        language: str,
        target_types: set[str],
        context_header: str,
        chunks: list[CodeChunk],
        depth: int,
    ) -> None:
        """Recursively walk the AST, extracting chunks at boundary nodes."""
        if depth > 10:  # Safety guard against pathological ASTs
            return

        for child in node.children:
            if child.type in target_types:
                chunk_text = source[child.start_byte:child.end_byte].decode(
                    "utf-8", errors="replace"
                )
                token_count = _count_tokens(chunk_text)

                # If chunk is too large, recurse into its children
                if token_count > MAX_CHUNK_TOKENS:
                    self._walk_node(
                        node=child,
                        source=source,
                        repo_id=repo_id,
                        file_path=file_path,
                        language=language,
                        target_types=target_types,
                        context_header=context_header,
                        chunks=chunks,
                        depth=depth + 1,
                    )
                elif token_count >= MIN_CHUNK_TOKENS:
                    start_line = child.start_point[0] + 1  # 1-indexed
                    end_line = child.end_point[0] + 1

                    chunk = CodeChunk(
                        id=_chunk_id(repo_id, file_path, start_line),
                        repo_id=repo_id,
                        file_path=file_path,
                        language=language,
                        node_type=_classify_node_type(child.type),
                        node_name=_get_node_name(child, source, language),
                        start_line=start_line,
                        end_line=end_line,
                        content=chunk_text,
                        context_header=context_header,
                        token_count=token_count,
                    )
                    chunks.append(chunk)
                else:
                    # Too small on its own — recurse to find nested chunks
                    self._walk_node(
                        node=child,
                        source=source,
                        repo_id=repo_id,
                        file_path=file_path,
                        language=language,
                        target_types=target_types,
                        context_header=context_header,
                        chunks=chunks,
                        depth=depth + 1,
                    )
            else:
                # Not a target node type — recurse to find nested boundaries
                self._walk_node(
                    node=child,
                    source=source,
                    repo_id=repo_id,
                    file_path=file_path,
                    language=language,
                    target_types=target_types,
                    context_header=context_header,
                    chunks=chunks,
                    depth=depth + 1,
                )

    def _fallback_chunk(
        self,
        repo_id: str,
        file_path: str,
        content: str,
        language: str,
    ) -> list[CodeChunk]:
        """
        Fallback: if AST parsing fails or yields nothing,
        split the file into sliding windows by line count.
        This is the 'everyone else does it this way' approach — we avoid it where possible.
        """
        lines = content.split("\n")
        if not lines:
            return []

        token_count = _count_tokens(content)

        # Small file: return as single chunk
        if token_count <= MAX_CHUNK_TOKENS:
            return [
                CodeChunk(
                    id=_chunk_id(repo_id, file_path, 1),
                    repo_id=repo_id,
                    file_path=file_path,
                    language=language,
                    node_type="module",
                    node_name=Path(file_path).name,
                    start_line=1,
                    end_line=len(lines),
                    content=content,
                    context_header="",
                    token_count=token_count,
                )
            ]

        # Large file: split into ~1000-token windows with 100-line stride
        chunks: list[CodeChunk] = []
        window = 80  # lines per chunk
        stride = 60  # lines to advance

        i = 0
        while i < len(lines):
            end = min(i + window, len(lines))
            chunk_text = "\n".join(lines[i:end])
            chunks.append(
                CodeChunk(
                    id=_chunk_id(repo_id, file_path, i + 1),
                    repo_id=repo_id,
                    file_path=file_path,
                    language=language,
                    node_type="module",
                    node_name=f"{Path(file_path).name}:L{i+1}-L{end}",
                    start_line=i + 1,
                    end_line=end,
                    content=chunk_text,
                    context_header="",
                    token_count=_count_tokens(chunk_text),
                )
            )
            i += stride
            if end >= len(lines):
                break

        return chunks

    def chunk_repository(
        self,
        repo_id: str,
        repo_path: Path,
        file_list: list[tuple[str, str, str]],  # (relative_path, content, language)
    ) -> Iterator[CodeChunk]:
        """
        Chunk all files in a repository, yielding CodeChunk objects one by one.
        Generator for memory efficiency on large repos.
        """
        for relative_path, content, language in file_list:
            try:
                file_chunks = self.chunk_file(repo_id, relative_path, content, language)
                yield from file_chunks
            except Exception as e:
                logger.error(f"Failed to chunk {relative_path}: {e}", exc_info=True)
                continue


# Singleton instance
_chunker: ASTChunker | None = None


def get_chunker() -> ASTChunker:
    global _chunker
    if _chunker is None:
        _chunker = ASTChunker()
    return _chunker

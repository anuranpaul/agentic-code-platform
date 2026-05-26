"""
Repository cloner and file walker.

Clones a GitHub repo with --depth 1 (shallow) into a temp directory,
walks the file tree, and returns files eligible for chunking.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import git

from src.config import get_settings
from src.ingestion.chunker import EXTENSION_TO_LANGUAGE

logger = logging.getLogger(__name__)

# Directories to skip entirely during file walking
SKIP_DIRS = {
    "node_modules",
    ".git",
    "__pycache__",
    ".pytest_cache",
    "vendor",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "coverage",
    ".venv",
    "venv",
    "env",
    ".mypy_cache",
    ".ruff_cache",
    "target",       # Rust build output
    "bin",
    "obj",
    ".gradle",
    ".idea",
    ".vscode",
}

# Files to skip by name
SKIP_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.lock",
    "poetry.lock",
    "go.sum",
    "*.min.js",
    "*.min.css",
    "*.map",
}


def _should_skip_dir(name: str) -> bool:
    return name in SKIP_DIRS or name.startswith(".")


def _should_skip_file(path: Path, max_bytes: int) -> bool:
    if path.name in SKIP_FILES:
        return True
    if path.suffix.lower() not in EXTENSION_TO_LANGUAGE:
        return True
    try:
        size = path.stat().st_size
        return size == 0 or size > max_bytes
    except OSError:
        return True


def clone_repo(github_url: str, branch: str = "main") -> tuple[Path, str]:
    """
    Shallow-clone a GitHub repo and return (temp_dir_path, actual_branch_used).
    Caller is responsible for cleaning up the temp directory.
    """
    settings = get_settings()
    clone_url = github_url.strip()

    # Inject auth token for private repos / higher rate limits
    if settings.github_token and "github.com" in clone_url:
        clone_url = clone_url.replace(
            "https://github.com",
            f"https://{settings.github_token}@github.com",
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="codebase_assistant_"))
    logger.info(f"Cloning {github_url} (branch={branch}) into {tmp_dir}")

    try:
        repo = git.Repo.clone_from(
            clone_url,
            str(tmp_dir),
            depth=1,
            branch=branch,
            no_single_branch=False,
        )
        actual_branch = repo.active_branch.name
        logger.info(f"Cloned successfully. Branch: {actual_branch}")
        return tmp_dir, actual_branch
    except git.exc.GitCommandError as e:
        # Try 'master' as fallback if 'main' doesn't exist
        if branch == "main" and "not found" in str(e).lower():
            logger.warning("Branch 'main' not found, retrying with 'master'")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return clone_repo(github_url, branch="master")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"Git clone failed for {github_url}: {e}") from e
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"Unexpected error cloning {github_url}: {e}") from e


def walk_repo_files(
    repo_path: Path,
    max_files: int | None = None,
    max_file_size: int | None = None,
) -> list[tuple[str, str, str]]:
    """
    Walk a cloned repository and return a list of (relative_path, content, language) tuples.

    Args:
        repo_path: Root directory of the cloned repo
        max_files: Maximum number of files to return (free tier safety)
        max_file_size: Maximum file size in bytes

    Returns:
        List of (relative_path, content, language) for all eligible files
    """
    settings = get_settings()
    max_files = max_files or settings.max_files_per_repo
    max_file_size = max_file_size or settings.max_file_size_bytes

    files: list[tuple[str, str, str]] = []
    seen_count = 0

    for path in sorted(repo_path.rglob("*")):
        if seen_count >= max_files:
            logger.info(f"Reached max_files limit ({max_files}), stopping file walk")
            break

        # Skip directories
        if path.is_dir():
            continue

        # Check if any parent directory should be skipped
        relative_parts = path.relative_to(repo_path).parts
        if any(_should_skip_dir(part) for part in relative_parts[:-1]):
            continue

        # Skip ineligible files
        if _should_skip_file(path, max_file_size):
            continue

        language = EXTENSION_TO_LANGUAGE.get(path.suffix.lower())
        if not language:
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError) as e:
            logger.debug(f"Could not read {path}: {e}")
            continue

        if not content.strip():
            continue

        relative_path = str(path.relative_to(repo_path))
        files.append((relative_path, content, language))
        seen_count += 1

    logger.info(f"Found {len(files)} eligible files in {repo_path}")
    return files


def cleanup_repo(repo_path: Path) -> None:
    """Remove the cloned repo from disk."""
    try:
        shutil.rmtree(repo_path, ignore_errors=True)
        logger.debug(f"Cleaned up {repo_path}")
    except Exception as e:
        logger.warning(f"Could not clean up {repo_path}: {e}")

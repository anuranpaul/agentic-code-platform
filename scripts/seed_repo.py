"""
seed_repo.py — Local development helper.

Ingests a GitHub repository end-to-end without needing the API server.
Useful for testing the ingestion pipeline locally before running the full stack.

Usage:
    python scripts/seed_repo.py https://github.com/owner/repo
    python scripts/seed_repo.py https://github.com/owner/repo --branch develop
    python scripts/seed_repo.py https://github.com/owner/repo --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("seed_repo")


async def seed(github_url: str, branch: str, dry_run: bool) -> None:
    from src.ingestion.chunker import get_chunker
    from src.ingestion.cloner import cleanup_repo, clone_repo, walk_repo_files
    from src.ingestion.embedder import get_embedder

    repo_id = str(uuid.uuid4())[:8]
    logger.info(f"Seeding repo: {github_url} (branch={branch}, repo_id={repo_id})")

    # 1. Clone
    logger.info("Cloning repository...")
    repo_path, actual_branch = clone_repo(github_url, branch)
    logger.info(f"Cloned to {repo_path} (branch: {actual_branch})")

    try:
        # 2. Walk files
        file_list = walk_repo_files(repo_path)
        logger.info(f"Found {len(file_list)} files")

        if dry_run:
            # Show file breakdown by language
            from collections import Counter
            lang_counts = Counter(lang for _, _, lang in file_list)
            logger.info("Language breakdown:")
            for lang, count in lang_counts.most_common():
                logger.info(f"  {lang}: {count} files")
            logger.info("DRY RUN — skipping chunking and embedding")
            return

        # 3. Chunk
        chunker = get_chunker()
        all_chunks = []
        logger.info("Chunking files...")
        for i, (rel_path, content, language) in enumerate(file_list):
            chunks = chunker.chunk_file(repo_id, rel_path, content, language)
            all_chunks.extend(chunks)
            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i+1}/{len(file_list)} files, {len(all_chunks)} chunks so far")

        logger.info(f"Total chunks: {len(all_chunks)}")

        # Print sample chunks
        logger.info("\nSample chunks:")
        for chunk in all_chunks[:5]:
            logger.info(
                f"  [{chunk.node_type}] {chunk.node_name} "
                f"@ {chunk.file_path}:L{chunk.start_line}-L{chunk.end_line} "
                f"({chunk.token_count} tokens)"
            )

        # 4. Embed
        logger.info("Upserting to Upstash Vector...")
        embedder = get_embedder()
        upserted = embedder.upsert_chunks(all_chunks)
        logger.info(f"Upserted {upserted}/{len(all_chunks)} vectors")

        # Save summary
        summary = {
            "repo_id": repo_id,
            "github_url": github_url,
            "branch": actual_branch,
            "file_count": len(file_list),
            "chunk_count": len(all_chunks),
            "vectors_upserted": upserted,
        }
        out_path = Path(f"data/seed_{repo_id}.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        logger.info(f"\nSummary saved to {out_path}")
        logger.info(f"Use this repo_id for queries: {repo_id}")

    finally:
        cleanup_repo(repo_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed a GitHub repo into the vector index")
    parser.add_argument("url", help="GitHub repository URL")
    parser.add_argument("--branch", default="main", help="Branch to ingest (default: main)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Clone and walk files only, skip chunking and embedding",
    )
    args = parser.parse_args()

    asyncio.run(seed(args.url, args.branch, args.dry_run))


if __name__ == "__main__":
    main()

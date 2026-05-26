"""
Publish a built LanceDB index to a Hugging Face Storage Bucket.

Buckets are S3-like mutable object storage on HF, Xet-backed and content-deduplicated,
so re-publishing only transfers changed chunks. No tar/extract dance, no LFS rate limit.

Usage:
    # HF_TOKEN must be in env (don't pass tokens on the command line)
    export HF_TOKEN=hf_xxx

    # Default: uploads to hf://buckets/<bucket>/lance_db
    uv run python -m medirag.index.publisher \\
        --db ./lance_db --bucket alvinhenrick/dailymed-embeddings

    # Versioned prefix — keep the previous one published while testing a new build
    uv run python -m medirag.index.publisher \\
        --db ./lance_db --bucket alvinhenrick/dailymed-embeddings --prefix lance_db/v2

Consumer side (mirror to local dir):
    from huggingface_hub import sync_bucket
    sync_bucket("hf://buckets/alvinhenrick/dailymed-embeddings/lance_db", "./lance_db")
"""

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import create_bucket, sync_bucket
from loguru import logger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", required=True, help="LanceDB directory to publish")
    parser.add_argument("--bucket", required=True, help="HF bucket id (e.g. user/dailymed-embeddings)")
    parser.add_argument(
        "--prefix",
        default="lance_db",
        help="Path inside the bucket (default: lance_db). Use e.g. lance_db/v2 to version.",
    )
    parser.add_argument("--private", action="store_true", help="Create the bucket as private if it doesn't exist")
    parser.add_argument("--delete", action="store_true", help="Remove remote files not present locally")
    args = parser.parse_args(argv)

    if not os.environ.get("HF_TOKEN"):
        logger.error("HF_TOKEN env var not set")
        return 1

    db_path = Path(args.db).resolve()
    if not db_path.is_dir():
        logger.error(f"{db_path} is not a directory")
        return 1

    logger.info(f"Ensuring bucket {args.bucket} exists")
    create_bucket(args.bucket, private=args.private, exist_ok=True)

    dest = f"hf://buckets/{args.bucket}/{args.prefix.strip('/')}"
    logger.info(f"Syncing {db_path} → {dest}")
    t0 = time.time()
    sync_bucket(str(db_path), dest, delete=args.delete)
    logger.info(f"Sync complete in {time.time() - t0:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())

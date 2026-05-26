"""
Publish a built LanceDB index to a Hugging Face dataset repo as a single tar archive.

The index directory contains thousands of small files; uploading them individually
hits HF's per-IP rate limit (1000 API calls per 5 min). Packing into one tar before
upload is reliable and produces a single LFS file. Consumers download + extract.

Usage:
    # HF_TOKEN must be in env (don't pass tokens on the command line)
    export HF_TOKEN=hf_xxx
    uv run python -m medirag.index.publisher \\
        --db ./lance_db --repo alvinhenrick/medirag-dailymed

Consumer side (download + extract):
    from huggingface_hub import hf_hub_download
    import tarfile, lancedb

    archive = hf_hub_download(
        repo_id="alvinhenrick/medirag-dailymed",
        filename="lance_db.tar",
        repo_type="dataset",
    )
    tarfile.open(archive).extractall(".")   # creates ./lance_db/
    db = lancedb.connect("./lance_db")
"""

import argparse
import os
import sys
import tarfile
import time
from pathlib import Path

from huggingface_hub import HfApi
from loguru import logger


ARCHIVE_NAME = "lance_db.tar"


def _make_tar(db_path: Path, tar_path: Path) -> None:
    """
    Pack the LanceDB directory into a plain tar (no gzip — lance files are already compressed columnar, gzip adds ~5% at
    best).
    """
    logger.info(f"Packing {db_path} → {tar_path}")
    t0 = time.time()
    with tarfile.open(tar_path, "w") as tar:
        tar.add(db_path, arcname=db_path.name)
    size_mb = tar_path.stat().st_size / 1e6
    logger.info(f"Packed {size_mb:.1f} MB in {time.time() - t0:.1f}s")


def _upload(tar_path: Path, repo_id: str, token: str) -> None:
    api = HfApi(token=token)
    logger.info(f"Ensuring dataset repo {repo_id} exists")
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)

    logger.info(f"Uploading {tar_path.name} → {repo_id}")
    t0 = time.time()
    api.upload_file(
        path_or_fileobj=str(tar_path),
        path_in_repo=ARCHIVE_NAME,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Publish LanceDB index {time.strftime('%Y-%m-%d %H:%M:%S')}",
    )
    logger.info(f"Upload complete in {time.time() - t0:.1f}s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", required=True, help="LanceDB directory to publish")
    parser.add_argument("--repo", required=True, help="HF dataset repo (e.g. user/medirag-dailymed)")
    parser.add_argument(
        "--tar-path",
        default=None,
        help="Where to write the tar (default: <db>.tar next to the db dir)",
    )
    parser.add_argument("--keep-tar", action="store_true", help="Don't delete the tar after upload")
    args = parser.parse_args(argv)

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN env var not set")
        return 1

    db_path = Path(args.db).resolve()
    if not db_path.is_dir():
        logger.error(f"{db_path} is not a directory")
        return 1

    tar_path = Path(args.tar_path).resolve() if args.tar_path else db_path.with_suffix(".tar")

    _make_tar(db_path, tar_path)
    try:
        _upload(tar_path, args.repo, token)
    finally:
        if not args.keep_tar:
            logger.info(f"Removing {tar_path}")
            tar_path.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

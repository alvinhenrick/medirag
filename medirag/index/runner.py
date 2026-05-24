"""
Build a LanceDB index from DailyMed SPL bundles.

Streams one part at a time, batches records, deletes the source zip after each
part so the peak disk stays small. Optionally publishes the finished index to a
Hugging Face dataset repo via `huggingface_hub.upload_folder`.

Usage examples:
    # Index one local zip into ./lance_db
    uv run python -m medirag.index.runner --source path/to/part1.zip --db ./lance_db

    # Index all DailyMed parts from URLs
    uv run python -m medirag.index.runner --all --db ./lance_db

    # Limit each part to N SPLs (smoke test)
    uv run python -m medirag.index.runner --source part1.zip --db ./lance_db --limit 50

    # Publish after building
    uv run python -m medirag.index.runner --all --db ./lance_db \\
        --publish-repo user/medirag-dailymed --publish-token $HF_TOKEN
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

import requests
from loguru import logger

from medirag.core.reader import ProductCard, SectionRecord, parse_spl_zip
from medirag.index.lance import LanceIndexer


DAILYMED_BASE = "https://dailymed-data.nlm.nih.gov/public-release-files"
DAILYMED_PARTS = [f"{DAILYMED_BASE}/dm_spl_release_human_rx_part{i}.zip" for i in range(1, 7)]


def _is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def _download(url: str, dest: Path) -> Path:
    logger.info(f"Downloading {url} → {dest}")
    t0 = time.time()
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        written = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MiB
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
        if total and written != total:
            logger.warning(f"download size mismatch: expected {total}, got {written}")
    logger.info(f"Downloaded {written / 1e6:.1f} MB in {time.time() - t0:.1f}s")
    return dest


def _stream_records(zip_path: Path, limit: int | None = None) -> Iterable[list[ProductCard | SectionRecord]]:
    """
    Yield record-lists for each SPL in the zip, optionally capped.
    """
    n = 0
    for records in parse_spl_zip(zip_path):
        if not records:
            continue
        yield records
        n += 1
        if limit is not None and n >= limit:
            logger.info(f"Reached --limit {limit}, stopping for this part")
            return


def _index_part(
    source: str,
    indexer: LanceIndexer,
    batch_size: int,
    limit: int | None,
    keep_zip: bool,
    work_dir: Path,
) -> tuple[int, int]:
    """
    Process one source (URL or local zip).

    Returns (spl_count, record_count).
    """
    if _is_url(source):
        zip_path = work_dir / Path(source).name
        if not zip_path.exists():
            _download(source, zip_path)
        else:
            logger.info(f"Using cached download {zip_path}")
        downloaded = True
    else:
        zip_path = Path(source)
        if not zip_path.exists():
            raise FileNotFoundError(zip_path)
        downloaded = False

    spl_count = 0
    record_count = 0
    batch: list[ProductCard | SectionRecord] = []

    logger.info(f"Parsing {zip_path}")
    for records in _stream_records(zip_path, limit=limit):
        spl_count += 1
        batch.extend(records)
        if len(batch) >= batch_size:
            indexer.add(batch)
            record_count += len(batch)
            logger.info(f"  inserted batch ({len(batch)} records) — totals: {spl_count} SPLs, {record_count} records")
            batch = []

    if batch:
        indexer.add(batch)
        record_count += len(batch)

    logger.info(f"Finished {zip_path.name}: {spl_count} SPLs, {record_count} records")

    if downloaded and not keep_zip:
        logger.info(f"Deleting downloaded zip {zip_path}")
        zip_path.unlink(missing_ok=True)

    return spl_count, record_count


def _publish(db_path: Path, repo_id: str, token: str | None) -> None:
    from huggingface_hub import HfApi, upload_folder

    api = HfApi(token=token)
    try:
        api.repo_info(repo_id, repo_type="dataset")
    except Exception:
        logger.info(f"Creating dataset repo {repo_id}")
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)

    logger.info(f"Uploading {db_path} → {repo_id}")
    upload_folder(
        folder_path=str(db_path),
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"Build {time.strftime('%Y-%m-%d %H:%M:%S')}",
    )
    logger.info("Upload complete")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", action="append", help="URL or local path to a DailyMed zip (repeatable)")
    src.add_argument("--all", action="store_true", help="Index all 6 DailyMed parts from official URLs")

    parser.add_argument("--db", required=True, help="LanceDB directory")
    parser.add_argument("--table", default="spl", help="Table name (default: spl)")
    parser.add_argument("--batch-size", type=int, default=512, help="Records per LanceDB insert batch (default: 512)")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N SPLs per part (smoke testing)")
    parser.add_argument("--keep-zip", action="store_true", help="Don't delete downloaded zips after processing")
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Where to download zips to (default: a temp dir, removed on exit)",
    )

    parser.add_argument("--publish-repo", default=None, help="HF dataset repo to upload to")
    parser.add_argument(
        "--publish-token",
        default=None,
        help="HF token (or use HF_TOKEN env var)",
    )

    args = parser.parse_args(argv)

    sources = DAILYMED_PARTS if args.all else args.source
    db_path = Path(args.db).resolve()
    db_path.mkdir(parents=True, exist_ok=True)

    if args.work_dir:
        work_dir = Path(args.work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup_work_dir = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="medirag-runner-"))
        cleanup_work_dir = True

    logger.info(f"Output DB: {db_path}")
    logger.info(f"Work dir: {work_dir}")
    logger.info(f"Sources: {sources}")

    indexer = LanceIndexer(db_path=db_path, table_name=args.table)

    total_spls = 0
    total_records = 0
    t_start = time.time()

    try:
        for source in sources:
            spls, records = _index_part(
                source=source,
                indexer=indexer,
                batch_size=args.batch_size,
                limit=args.limit,
                keep_zip=args.keep_zip,
                work_dir=work_dir,
            )
            total_spls += spls
            total_records += records

        logger.info("Creating FTS index for hybrid search…")
        indexer.create_fts_index()
    finally:
        if cleanup_work_dir:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)

    elapsed = time.time() - t_start
    logger.info(
        f"Done: {total_spls} SPLs, {total_records} records in {elapsed / 60:.1f} min "
        f"({total_records / elapsed:.0f} records/s)"
    )

    if args.publish_repo:
        token = args.publish_token or os.environ.get("HF_TOKEN")
        if not token:
            logger.error("--publish-repo set but no token provided (use --publish-token or HF_TOKEN env)")
            return 1
        _publish(db_path, args.publish_repo, token)

    return 0


if __name__ == "__main__":
    sys.exit(main())

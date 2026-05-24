"""Smoke test: run the CLI runner end-to-end against a synthetic SPL zip."""

import zipfile

from medirag.index.lance import LanceIndexer
from medirag.index.runner import main


SAMPLE_XML = "BE27854A-A805-4300-9729-ACCD1B7F226F.xml"


def test_runner_indexes_local_zip(data_dir, tmp_path):
    """
    parse_spl_zip handles nested zips (zips-of-zips).

    Build one that mirrors DailyMed's structure: outer.zip → spl.zip → spl.xml.
    """
    xml_bytes = (data_dir / SAMPLE_XML).read_bytes()

    inner_zip = tmp_path / "spl_BE27854A.zip"
    with zipfile.ZipFile(inner_zip, "w") as z:
        z.writestr(SAMPLE_XML, xml_bytes)

    outer_zip = tmp_path / "dm_spl_release_test.zip"
    with zipfile.ZipFile(outer_zip, "w") as z:
        z.write(inner_zip, arcname=inner_zip.name)

    db_path = tmp_path / "lance"

    rc = main(
        [
            "--source",
            str(outer_zip),
            "--db",
            str(db_path),
            "--batch-size",
            "4",
        ]
    )
    assert rc == 0

    # Re-open the index and verify content
    reopened = LanceIndexer(db_path=db_path)
    assert reopened.table is not None
    n = reopened.table.count_rows()
    assert n > 1, f"expected at least product card + sections, got {n}"

    hits = reopened.retrieve("urinary tract infection", top_k=2)
    assert len(hits) > 0
    assert all(h.set_id == "BE27854A-A805-4300-9729-ACCD1B7F226F" for h in hits)

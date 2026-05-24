"""Integration tests: build a tiny LanceDB index from the one sample XML, query it.

These tests download the PubMedBERT model on first run (cached afterwards).
"""

import pytest

from medirag.core.reader import parse_spl
from medirag.index.lance import LanceIndexer


SAMPLE_XML = "BE27854A-A805-4300-9729-ACCD1B7F226F.xml"


@pytest.fixture
def lance_index(data_dir, tmp_path):
    """
    Build a one-document LanceDB index in a tmpdir.
    """
    records = parse_spl(data_dir / SAMPLE_XML)
    indexer = LanceIndexer(db_path=tmp_path / "lance")
    n = indexer.add(records)
    assert n > 1, "expected at least product card + a few sections"
    indexer.create_fts_index()
    return indexer


def test_retrieve_returns_hits(lance_index):
    hits = lance_index.retrieve("urinary tract infection antibiotic", top_k=3)
    assert len(hits) > 0
    assert all(h.set_id == "BE27854A-A805-4300-9729-ACCD1B7F226F" for h in hits)


def test_allergy_query_finds_inactive_ingredients(lance_index):
    """Patient asks: 'is there anything I'm allergic to in this drug?'
    Should retrieve the product card which lists inactive ingredients.
    """
    hits = lance_index.retrieve("what inactive ingredients are in this capsule gelatin", top_k=3)
    assert any(h.kind == "product_card" for h in hits)


def test_dosing_query_finds_dosage_section(lance_index):
    hits = lance_index.retrieve("how many capsules should I take each day", top_k=3)
    loincs = {h.loinc for h in hits}
    assert "34068-7" in loincs  # Dosage & Administration


def test_contraindication_query_finds_contraindications(lance_index):
    hits = lance_index.retrieve("can I take this if I am allergic to sulfonamide", top_k=3)
    loincs = {h.loinc for h in hits}
    # contraindications (34070-3) or warnings (34071-1) — both mention sulfonamide
    assert "34070-3" in loincs or "34071-1" in loincs


def test_side_effect_query_finds_adverse_reactions(lance_index):
    hits = lance_index.retrieve("nausea diarrhea side effects", top_k=3)
    loincs = {h.loinc for h in hits}
    assert "34084-4" in loincs


def test_pill_identification_finds_product_card(lance_index):
    """
    Patient asks 'I have a capsule with Pfizer;092 imprint, what is it?'.
    """
    hits = lance_index.retrieve("yellow capsule imprint Pfizer 092", top_k=3)
    assert any(h.kind == "product_card" for h in hits)


def test_metadata_filter_patient_facing(lance_index):
    """
    Should be able to restrict to product cards via SQL filter.
    """
    hits = lance_index.retrieve(
        "what does this drug do",
        top_k=10,
        where="kind = 'product_card'",
    )
    assert all(h.kind == "product_card" for h in hits)
    assert len(hits) >= 1


def test_hybrid_search_with_exact_term(lance_index):
    """
    Hybrid search should heavily weight exact terms like brand names.
    """
    hits = lance_index.retrieve("Urobiotic", top_k=3, hybrid=True)
    assert len(hits) > 0
    assert all("urobiotic" in h.text.lower() for h in hits[:1])


def test_persistence_across_indexer_instances(data_dir, tmp_path):
    """
    Indexed data persists when a new indexer points at the same path.
    """
    records = parse_spl(data_dir / SAMPLE_XML)
    LanceIndexer(db_path=tmp_path / "lance").add(records)

    reopened = LanceIndexer(db_path=tmp_path / "lance")
    hits = reopened.retrieve("urinary tract infection", top_k=2)
    assert len(hits) > 0

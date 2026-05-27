"""
LanceDB-backed indexer for SPL records.

Uses LanceDB's native sentence-transformers embedding registry with NeuML/pubmedbert-base-embeddings (768d). Stores full
metadata for filtering and supports hybrid (vector + BM25) search.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import lancedb
import torch
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from loguru import logger
from sentence_transformers import SentenceTransformer

from medirag.core.reader import ProductCard, SectionRecord


EMBED_MODEL = "NeuML/pubmedbert-base-embeddings"
EMBED_DIM = 768
DEFAULT_TABLE = "spl"


def _build_schema(embed_model_name: str = EMBED_MODEL):
    """
    Build the LanceDB Pydantic schema with an embedding function attached.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = get_registry().get("sentence-transformers").create(name=embed_model_name, device=device)

    class SplRecord(LanceModel):
        # identity
        set_id: str
        version: str | None = None
        drug_name: str
        kind: str  # "product_card" | "section"

        # the searchable text — source field for the embedding
        text: str = embedder.SourceField()
        vector: Vector(embedder.ndims()) = embedder.VectorField()  # type: ignore[valid-type] # noqa: F821

        # section metadata (empty strings for product_card)
        loinc: str = ""
        section_title: str = ""
        is_patient_facing: bool = False

        # product metadata for cross-drug filtering (empty for section rows)
        generic_name: str = ""
        manufacturer: str = ""
        dosage_form: str = ""
        route: str = ""
        active_ingredient_uniis: list[str] = []
        active_ingredient_names: list[str] = []
        inactive_ingredient_names: list[str] = []
        ndcs: list[str] = []

    return SplRecord, embedder


@dataclass
class RetrievalHit:
    text: str
    drug_name: str
    set_id: str
    kind: str
    loinc: str
    section_title: str
    score: float
    is_patient_facing: bool


class LanceIndexer:
    """
    LanceDB indexer with PubMedBERT embeddings and optional hybrid search.
    """

    def __init__(
        self,
        db_path: str | Path,
        table_name: str = DEFAULT_TABLE,
        embed_model: str = EMBED_MODEL,
    ):
        self.db_path = str(db_path)
        self.table_name = table_name
        self.embed_model = embed_model
        self._schema, self._embedder = _build_schema(embed_model)
        self._db = lancedb.connect(self.db_path)
        self._table = None
        if table_name in _list_table_names(self._db):
            self._table = self._db.open_table(table_name)
        # CPU-pinned query encoder. Used at search time to bypass the embedder
        # baked into the table (which can call `.cuda()` on CPU-only deploys).
        # Indexing still uses the table's stored embedder via `add()`.
        self._query_encoder: SentenceTransformer | None = None

    @property
    def table(self):
        return self._table

    def _ensure_table(self):
        if self._table is None:
            self._table = self._db.create_table(
                self.table_name,
                schema=self._schema,  # type: ignore[arg-type]
                mode="create",
            )
        return self._table

    def add(self, records: Iterable[ProductCard | SectionRecord]) -> int:
        """
        Insert records.

        Lance computes embeddings automatically.
        """
        rows = [_record_to_row(r) for r in records]
        if not rows:
            return 0
        table = self._ensure_table()
        table.add(rows)
        return len(rows)

    def _encode_query(self, query: str):
        encoder = self._query_encoder
        if encoder is None:
            encoder = SentenceTransformer(self.embed_model, device="cpu")
            self._query_encoder = encoder
        # LanceDB's sentence-transformers wrapper normalizes by default; match
        # that so query and indexed vectors live in the same space.
        vec = encoder.encode([query], show_progress_bar=False, normalize_embeddings=True)
        return vec[0]

    def create_fts_index(self) -> None:
        """
        Create a full-text search index on `text` for hybrid retrieval.
        """
        if self._table is None:
            return
        self._table.create_fts_index("text", replace=True)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        with_reranker: bool = False,
        hybrid: bool = False,
        where: str | None = None,
    ) -> list[RetrievalHit]:
        """
        Vector (default) or hybrid (vector + BM25) retrieval.

        `where` is a SQL filter (e.g. "is_patient_facing = true").
        """
        if self._table is None:
            return []

        query_vec = self._encode_query(query)
        if hybrid:
            # Hybrid: set vector and text explicitly so lancedb never invokes
            # its stored embedder (which can call `.cuda()` on CPU-only deploys).
            search = self._table.search(query_type="hybrid").vector(query_vec).text(query)
        else:
            search = self._table.search(query_vec, query_type="vector")
        if where:
            search = search.where(where, prefilter=True)
        results = search.limit(top_k).to_list()

        hits = []
        for r in results:
            score = float(r.get("_relevance_score") or r.get("_distance") or 0.0)
            hits.append(
                RetrievalHit(
                    text=r["text"],
                    drug_name=r["drug_name"],
                    set_id=r["set_id"],
                    kind=r["kind"],
                    loinc=r.get("loinc", ""),
                    section_title=r.get("section_title", ""),
                    score=score,
                    is_patient_facing=bool(r.get("is_patient_facing", False)),
                )
            )

        if with_reranker:
            logger.debug("Reranker requested but not yet wired up; returning vector results")
        return hits


def _list_table_names(db) -> list[str]:
    """
    Get table names regardless of which list_tables API version is in use.
    """
    result = db.list_tables()
    if hasattr(result, "tables"):
        return list(result.tables)
    return list(result)


def _record_to_row(r: ProductCard | SectionRecord) -> dict:
    """
    Flatten a dataclass record into the LanceDB row dict.
    """
    if isinstance(r, ProductCard):
        return {
            "set_id": r.set_id,
            "version": r.version,
            "drug_name": r.drug_name,
            "kind": r.kind,
            "text": r.text,
            "loinc": "",
            "section_title": "",
            "is_patient_facing": False,
            "generic_name": r.generic_name or "",
            "manufacturer": r.manufacturer or "",
            "dosage_form": r.dosage_form or "",
            "route": r.route or "",
            "active_ingredient_uniis": r.active_ingredient_uniis,
            "active_ingredient_names": r.active_ingredient_names,
            "inactive_ingredient_names": r.inactive_ingredients,
            "ndcs": r.ndcs,
        }
    return {
        "set_id": r.set_id,
        "version": r.version,
        "drug_name": r.drug_name,
        "kind": r.kind,
        "text": r.text,
        "loinc": r.loinc,
        "section_title": r.section_title,
        "is_patient_facing": r.is_patient_facing,
        "generic_name": "",
        "manufacturer": "",
        "dosage_form": "",
        "route": "",
        "active_ingredient_uniis": [],
        "active_ingredient_names": [],
        "inactive_ingredient_names": [],
        "ndcs": [],
    }

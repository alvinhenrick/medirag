"""
SPL XML extractor.

Emits two kinds of records per drug:
  - one ProductCard summarizing structured product data (ingredients, NDC, appearance, route)
  - one SectionRecord per narrative section, keyed by LOINC

Records are plain dataclasses, so the indexer doesn't depend on LlamaIndex.
"""

import re
import zipfile
from dataclasses import dataclass, field
from html import unescape
from io import BytesIO
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup, Tag
from loguru import logger


def _find(parent: Tag | BeautifulSoup, name: str, *, recursive: bool = True) -> Tag | None:
    """
    Type-narrowing wrapper around bs4 find(): returns Tag or None.
    """
    el = parent.find(name, recursive=recursive)
    return el if isinstance(el, Tag) else None


def _find_all(parent: Tag | BeautifulSoup, name: str, *, recursive: bool = True) -> list[Tag]:
    """
    Type-narrowing wrapper around bs4 find_all(): returns only Tags.
    """
    return [el for el in parent.find_all(name, recursive=recursive) if isinstance(el, Tag)]


def _attr(tag: Tag, name: str) -> str | None:
    """
    Type-narrowing wrapper for tag.get(): always returns str or None.

    bs4 types attributes as `str | AttributeValueList | None` because of HTML
    multi-value attrs like `class`. SPL XML attributes are always single-valued.
    """
    v = tag.get(name)
    if v is None:
        return None
    if isinstance(v, list):
        return v[0] if v else None
    return v


PATIENT_LOINCS: frozenset[str] = frozenset(
    {
        "42230-3",  # Patient Package Insert
        "42231-1",  # Medication Guide
        "34076-0",  # Information for Patients
        "68498-5",  # Patient Information / Patient Counseling
    }
)

# Sections with no useful narrative for patient Q&A — skip to keep index lean.
SKIP_LOINCS: frozenset[str] = frozenset(
    {
        "48780-1",  # SPL Listing Data Elements
        "60575-8",  # SPL Establishment Section
        "51945-4",  # Package Label / Principal Display Panel (just NDC stickers)
    }
)


@dataclass
class ProductCard:
    set_id: str
    version: str | None
    drug_name: str
    generic_name: str | None
    manufacturer: str | None
    dosage_form: str | None
    route: str | None
    active_ingredients: list[dict] = field(default_factory=list)  # [{name, unii, strength, unit}]
    inactive_ingredients: list[str] = field(default_factory=list)
    ndcs: list[str] = field(default_factory=list)
    appearance: dict = field(default_factory=dict)  # color/shape/size/imprint/score/coating
    text: str = ""

    @property
    def kind(self) -> str:
        return "product_card"

    @property
    def active_ingredient_uniis(self) -> list[str]:
        return [a["unii"] for a in self.active_ingredients if a.get("unii")]

    @property
    def active_ingredient_names(self) -> list[str]:
        return [a["name"] for a in self.active_ingredients if a.get("name")]


@dataclass
class SectionRecord:
    set_id: str
    version: str | None
    drug_name: str
    loinc: str
    section_title: str
    text: str
    is_patient_facing: bool

    @property
    def kind(self) -> str:
        return "section"


def _clean(text: str) -> str:
    """
    Whitespace-normalize, preserve punctuation/numbers/units.
    """
    if not text:
        return ""
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _section_text(section: Tag) -> str:
    """Extract narrative text from a section: paragraphs, list items, table cells.

    Walks only the direct <text> child, so nested subsections aren't double-counted
    (the caller walks them separately).
    """
    text_block = _find(section, "text", recursive=False)
    if text_block is None:
        return ""

    parts: list[str] = []
    for el in text_block.descendants:
        if not isinstance(el, Tag):
            continue
        if el.name in ("paragraph", "item", "caption"):
            t = _clean(el.get_text(" ", strip=True))
            if t:
                parts.append(t)
        elif el.name == "td" or el.name == "th":
            t = _clean(el.get_text(" ", strip=True))
            if t:
                parts.append(t)
    return "\n".join(parts)


def _extract_product_metadata(doc: BeautifulSoup) -> tuple[ProductCard | None, Tag | None]:
    """
    Find the manufacturedProduct section and pull all structured data from it.

    Returns (card, product_section_tag). The section tag is returned so the caller can skip re-processing it as a
    narrative section.
    """
    set_id_tag = _find(doc, "setId")
    set_id = _attr(set_id_tag, "root") if set_id_tag else None
    if not set_id:
        logger.warning("SPL missing setId, skipping")
        return None, None

    version_tag = _find(doc, "versionNumber")
    version = _attr(version_tag, "value") if version_tag else None

    effective = _find(doc, "effectiveTime")
    if version is None and effective is not None:
        version = _attr(effective, "value")

    org_name = _find(doc, "representedOrganization")
    org_name_tag = _find(org_name, "name") if org_name else None
    manufacturer = _clean(org_name_tag.get_text()) if org_name_tag else None

    product = _find(doc, "manufacturedProduct")
    if product is None:
        return None, None
    product_section_el = product.find_parent("section")
    product_section: Tag | None = product_section_el if isinstance(product_section_el, Tag) else None

    medicine = _find(product, "manufacturedMedicine") or _find(product, "manufacturedProduct")
    name_tag = _find(medicine, "name", recursive=False) if medicine else None
    drug_name = _clean(name_tag.get_text()) if name_tag else "Unknown"

    generic_name = None
    generic = _find(product, "genericMedicine")
    generic_name_tag = _find(generic, "name") if generic else None
    if generic_name_tag:
        generic_name = _clean(generic_name_tag.get_text())

    form_code = _find(medicine, "formCode") if medicine else None
    dosage_form = _attr(form_code, "displayName") if form_code is not None else None

    route_tag = _find(product, "routeCode")
    route = _attr(route_tag, "displayName") if route_tag is not None else None

    active_ingredients: list[dict] = []
    seen_uniis: set[str] = set()
    for ai in _find_all(medicine, "activeIngredient", recursive=False) if medicine else []:
        substance = _find(ai, "activeIngredientSubstance")
        if not substance:
            continue
        sub_name = _find(substance, "name", recursive=False)
        code = _find(substance, "code", recursive=False)
        unii = _attr(code, "code") if code is not None else None
        if unii and unii in seen_uniis:
            continue
        if unii:
            seen_uniis.add(unii)
        numerator = _find(ai, "numerator")
        strength = _attr(numerator, "value") if numerator is not None else None
        unit = _attr(numerator, "unit") if numerator is not None else None
        active_ingredients.append(
            {
                "name": _clean(sub_name.get_text()) if sub_name else None,
                "unii": unii,
                "strength": strength,
                "unit": unit,
            }
        )

    inactive_ingredients: list[str] = []
    seen_inactive: set[str] = set()
    for ii in _find_all(medicine, "inactiveIngredient", recursive=False) if medicine else []:
        sub = _find(ii, "inactiveIngredientSubstance")
        sub_name = _find(sub, "name") if sub else None
        if sub_name:
            name = _clean(sub_name.get_text())
            if name and name.lower() not in seen_inactive:
                seen_inactive.add(name.lower())
                inactive_ingredients.append(name)

    ndcs: list[str] = []
    for code in _find_all(product, "code"):
        if _attr(code, "codeSystem") == "2.16.840.1.113883.6.69":  # FDA NDC code system
            ndc = _attr(code, "code")
            if ndc and ndc not in ndcs:
                ndcs.append(ndc)

    appearance: dict = {}
    for char in _find_all(product, "characteristic"):
        code = _find(char, "code")
        if code is None:
            continue
        key = (_attr(code, "code") or "").upper()
        value_tag = _find(char, "value")
        if value_tag is None:
            continue
        display_name = _attr(value_tag, "displayName")
        value_str = _attr(value_tag, "value")
        if display_name:
            appearance[key] = display_name
        elif value_str:
            unit = _attr(value_tag, "unit")
            appearance[key] = f"{value_str} {unit}" if unit else value_str
        else:
            t = _clean(value_tag.get_text())
            if t:
                appearance[key] = t

    card_text = _format_product_card_text(
        drug_name=drug_name,
        generic_name=generic_name,
        manufacturer=manufacturer,
        dosage_form=dosage_form,
        route=route,
        active_ingredients=active_ingredients,
        inactive_ingredients=inactive_ingredients,
        ndcs=ndcs,
        appearance=appearance,
    )

    card = ProductCard(
        set_id=set_id,
        version=version,
        drug_name=drug_name,
        generic_name=generic_name,
        manufacturer=manufacturer,
        dosage_form=dosage_form,
        route=route,
        active_ingredients=active_ingredients,
        inactive_ingredients=inactive_ingredients,
        ndcs=ndcs,
        appearance=appearance,
        text=card_text,
    )
    return card, product_section


def _format_product_card_text(
    *,
    drug_name: str,
    generic_name: str | None,
    manufacturer: str | None,
    dosage_form: str | None,
    route: str | None,
    active_ingredients: list[dict],
    inactive_ingredients: list[str],
    ndcs: list[str],
    appearance: dict,
) -> str:
    """
    Synthesize structured product data into searchable prose.
    """
    lines: list[str] = [f"{drug_name}"]
    if generic_name and generic_name.lower() != drug_name.lower():
        lines.append(f"Generic name: {generic_name}.")
    if manufacturer:
        lines.append(f"Manufacturer: {manufacturer}.")
    if dosage_form or route:
        parts = []
        if dosage_form:
            parts.append(f"form: {dosage_form.lower()}")
        if route:
            parts.append(f"route: {route.lower()}")
        lines.append(", ".join(parts).capitalize() + ".")

    if active_ingredients:
        ai_parts = []
        for ai in active_ingredients:
            name = ai.get("name") or "unknown"
            strength = ai.get("strength")
            unit = ai.get("unit")
            if strength and unit:
                ai_parts.append(f"{name} {strength} {unit}")
            else:
                ai_parts.append(name)
        lines.append(f"Active ingredients: {', '.join(ai_parts)}.")

    if inactive_ingredients:
        lines.append(f"Inactive ingredients: {', '.join(inactive_ingredients)}.")

    if appearance:
        app_parts = []
        for key, label in (
            ("SPLCOLOR", "color"),
            ("SPLSHAPE", "shape"),
            ("SPLSIZE", "size"),
            ("SPLIMPRINT", "imprint"),
            ("SPLSCORE", "score"),
            ("SPLCOATING", "coating"),
        ):
            if key in appearance:
                app_parts.append(f"{label} {appearance[key]}")
        if app_parts:
            lines.append(f"Appearance: {', '.join(app_parts)}.")

    if ndcs:
        lines.append(f"NDC codes: {', '.join(ndcs)}.")

    return " ".join(lines)


def _extract_sections(
    doc: BeautifulSoup, set_id: str, version: str | None, drug_name: str, skip_section: Tag | None
) -> Iterable[SectionRecord]:
    """
    Walk every <section> with a LOINC code, emit one record each.
    """
    for section in _find_all(doc, "section"):
        if section is skip_section:
            continue
        code = _find(section, "code", recursive=False)
        if code is None:
            continue
        loinc = _attr(code, "code")
        if not loinc or loinc in SKIP_LOINCS:
            continue

        title_tag = _find(section, "title", recursive=False)
        title = _clean(title_tag.get_text()) if title_tag else ""
        display_name = _attr(code, "displayName") or title or "SPL Section"

        body = _section_text(section)
        if not body:
            continue

        # Prepend drug + section context so embeddings carry it.
        prefixed = f"{drug_name} — {display_name}:\n{body}"

        yield SectionRecord(
            set_id=set_id,
            version=version,
            drug_name=drug_name,
            loinc=loinc,
            section_title=display_name,
            text=prefixed,
            is_patient_facing=loinc in PATIENT_LOINCS,
        )


def parse_spl(source: str | Path | bytes) -> list[ProductCard | SectionRecord]:
    """
    Parse one SPL XML file into a list of records (product card + sections).

    Accepts a file path or raw bytes.
    """
    if isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            data = f.read()
    else:
        data = source

    doc = BeautifulSoup(data, "lxml-xml")
    card, product_section = _extract_product_metadata(doc)
    if card is None:
        logger.warning("Could not extract product metadata, skipping document")
        return []

    records: list[ProductCard | SectionRecord] = [card]
    records.extend(
        _extract_sections(
            doc=doc,
            set_id=card.set_id,
            version=card.version,
            drug_name=card.drug_name,
            skip_section=product_section,
        )
    )
    return records


def parse_spl_zip(zip_path: str | Path) -> Iterable[list[ProductCard | SectionRecord]]:
    """
    Stream-parse every XML inside an SPL zip (handles nested zips).

    DailyMed bundles are zips-of-zips: each outer zip contains per-SPL zips, each
    containing one XML. Yields one list[ProductCard|SectionRecord] per SPL.
    """
    with zipfile.ZipFile(zip_path, "r") as outer:
        for name in outer.namelist():
            if name.endswith(".xml"):
                with outer.open(name) as f:
                    yield parse_spl(f.read())
            elif name.endswith(".zip"):
                with outer.open(name) as inner_zip_bytes:
                    inner_data = inner_zip_bytes.read()
                with zipfile.ZipFile(BytesIO(inner_data)) as inner:
                    for inner_name in inner.namelist():
                        if inner_name.endswith(".xml"):
                            with inner.open(inner_name) as f:
                                yield parse_spl(f.read())

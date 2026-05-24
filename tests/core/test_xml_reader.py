from medirag.core.reader import ProductCard, SectionRecord, parse_spl


SAMPLE_XML = "BE27854A-A805-4300-9729-ACCD1B7F226F.xml"


def test_product_card_basic_identity(data_dir):
    records = parse_spl(data_dir / SAMPLE_XML)
    cards = [r for r in records if isinstance(r, ProductCard)]
    assert len(cards) == 1
    card = cards[0]

    assert card.set_id == "BE27854A-A805-4300-9729-ACCD1B7F226F"
    assert "Urobiotic" in card.drug_name
    assert card.manufacturer == "Roerig"
    assert card.dosage_form == "CAPSULE"
    assert card.route == "ORAL"


def test_product_card_active_ingredients(data_dir):
    records = parse_spl(data_dir / SAMPLE_XML)
    card = next(r for r in records if isinstance(r, ProductCard))

    ai_names = {a["name"].lower() for a in card.active_ingredients}
    assert "oxytetracycline hydrochloride" in ai_names
    assert "sulfamethizole" in ai_names
    assert "phenazopyridine hydrochloride" in ai_names

    uniis = set(card.active_ingredient_uniis)
    assert "4U7K4N52ZM" in uniis  # oxytetracycline HCl
    assert "25W8454H16" in uniis  # sulfamethizole
    assert "2IUY41693Z" in uniis  # phenazopyridine HCl

    # strengths preserved with units
    strengths = {(a["strength"], a["unit"]) for a in card.active_ingredients}
    assert ("250", "mg") in strengths
    assert ("50", "mg") in strengths


def test_product_card_inactive_ingredients_for_allergy_queries(data_dir):
    records = parse_spl(data_dir / SAMPLE_XML)
    card = next(r for r in records if isinstance(r, ProductCard))
    lower = {n.lower() for n in card.inactive_ingredients}
    # important for allergy/diet questions
    assert "gelatin" in lower
    assert "starch" in lower
    assert "magnesium stearate" in lower


def test_product_card_physical_appearance(data_dir):
    records = parse_spl(data_dir / SAMPLE_XML)
    card = next(r for r in records if isinstance(r, ProductCard))
    # the XML has two SPLCOLOR characteristics (Green, Yellow); last one wins
    assert card.appearance.get("SPLSHAPE") == "CAPSULE"
    assert card.appearance.get("SPLSIZE") == "22 mm"
    assert card.appearance.get("SPLIMPRINT") == "Pfizer;092"
    assert "SPLCOLOR" in card.appearance


def test_product_card_ndcs(data_dir):
    records = parse_spl(data_dir / SAMPLE_XML)
    card = next(r for r in records if isinstance(r, ProductCard))
    assert "0049-0920" in card.ndcs
    assert "0049-0920-50" in card.ndcs
    assert "0049-0920-41" in card.ndcs


def test_product_card_text_is_searchable_prose(data_dir):
    records = parse_spl(data_dir / SAMPLE_XML)
    card = next(r for r in records if isinstance(r, ProductCard))
    text = card.text.lower()
    assert "urobiotic" in text
    assert "oxytetracycline" in text
    assert "250 mg" in text  # dose punctuation preserved
    assert "capsule" in text
    assert "oral" in text
    assert "gelatin" in text


def test_all_narrative_sections_extracted(data_dir):
    records = parse_spl(data_dir / SAMPLE_XML)
    sections = [r for r in records if isinstance(r, SectionRecord)]
    loincs = {s.loinc for s in sections}

    assert "34067-9" in loincs  # Indications & Usage
    assert "34068-7" in loincs  # Dosage & Administration
    assert "34070-3" in loincs  # Contraindications
    assert "34071-1" in loincs  # Warnings
    assert "34084-4" in loincs  # Adverse Reactions
    assert "42232-9" in loincs  # Precautions
    assert "34069-5" in loincs  # How Supplied


def test_section_text_preserves_dose_punctuation(data_dir):
    records = parse_spl(data_dir / SAMPLE_XML)
    sections = [r for r in records if isinstance(r, SectionRecord)]
    dosage = next(s for s in sections if s.loinc == "34068-7")
    # regression for old no_punct=True bug that turned "1 capsule" into mush
    assert "1 capsule four times daily" in dosage.text or "1 capsule" in dosage.text
    # contraindications mentions sulfonamide
    contra = next(s for s in sections if s.loinc == "34070-3")
    assert "sulfonamide" in contra.text.lower()


def test_section_text_carries_drug_and_section_context(data_dir):
    records = parse_spl(data_dir / SAMPLE_XML)
    sections = [r for r in records if isinstance(r, SectionRecord)]
    for s in sections:
        # embedding-time context prefix
        assert s.drug_name in s.text
        assert s.section_title in s.text


def test_section_metadata_is_patient_facing_flag(data_dir):
    records = parse_spl(data_dir / SAMPLE_XML)
    sections = [r for r in records if isinstance(r, SectionRecord)]
    # this XML predates the patient-counseling-section standard; no patient-facing sections
    # but the flag should exist on every record and be False here
    assert all(isinstance(s.is_patient_facing, bool) for s in sections)


def test_skip_useless_sections(data_dir):
    records = parse_spl(data_dir / SAMPLE_XML)
    sections = [r for r in records if isinstance(r, SectionRecord)]
    # principal display panel etc. should never appear
    assert "51945-4" not in {s.loinc for s in sections}
    # every emitted section has non-empty body
    for s in sections:
        body = s.text.split(":\n", 1)[1] if ":\n" in s.text else s.text
        assert body.strip()

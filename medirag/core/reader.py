import zipfile
import re
from bs4 import BeautifulSoup
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader


def normalize_text(text):
    """Normalize the text by lowercasing, removing extra spaces, and stripping unnecessary characters."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def format_output_string(drug_name, sections_data):
    output = [f"Drug and Generic Names: {drug_name}"]

    for title, paragraphs in sections_data.items():
        output.append(f"{title}:")
        for paragraph in paragraphs:
            output.append(f" - {paragraph}")
        output.append("")  # Add a newline after each section

    return "\n".join(output)


def extract_drug_and_generic_names(structured_body):
    # Extract the main drug name and any generic names
    drug_names = set()  # Use a set to avoid duplicates

    # Look for manufacturedProduct elements
    manufactured_products = structured_body.find_all("manufacturedProduct")

    for manufactured_product in manufactured_products:
        # Extract the main drug name
        name_tag = manufactured_product.find("name")
        if name_tag:
            drug_names.add(name_tag.get_text(strip=True))

        # Extract the generic names if available
        as_generic = manufactured_product.find("asEntityWithGeneric")
        if as_generic:
            generic_name_tag = as_generic.find("genericMedicine").find("name")
            if generic_name_tag:
                drug_names.add(generic_name_tag.get_text(strip=True))

    return list(drug_names)


def parse_drug_information(soup, extra_info=None):
    # Extract the setId
    set_id = None
    set_id_tag = soup.find("setId")
    if set_id_tag:
        set_id = set_id_tag.get("root", None)
    if not set_id:
        return None

    structured_body = soup.find("structuredBody")

    # Extract the drug name
    drug_names = extract_drug_and_generic_names(structured_body)

    if len(drug_names) == 0:
        return None

    drug_name = " | ".join(drug_names)

    # Iterate over components and extract sections
    components = structured_body.find_all("component")
    sections_data = {}

    for component in components:
        sections = component.find_all("section")
        for section in sections:
            title_tag = section.find("title")
            if title_tag:
                title_text = normalize_text(title_tag.get_text(strip=True))
            else:
                continue  # Skip if title is not found

            paragraphs = section.find_all("paragraph")
            paragraphs_text = []
            seen_paragraphs = set()  # Set to track unique paragraphs
            for paragraph in paragraphs:
                paragraph_text = normalize_text(paragraph.get_text(strip=True))
                if paragraph_text and paragraph_text not in seen_paragraphs:
                    paragraphs_text.append(paragraph_text)
                    seen_paragraphs.add(paragraph_text)

            # Only include sections with non-empty, non-duplicate paragraphs
            if paragraphs_text:
                if title_text in sections_data:
                    existing_paragraphs = set(sections_data[title_text])
                    # Add only unique paragraphs that aren't already in the title's list
                    unique_paragraphs = [p for p in paragraphs_text if p not in existing_paragraphs]
                    sections_data[title_text].extend(unique_paragraphs)
                else:
                    sections_data[title_text] = paragraphs_text

    drug_info_str = format_output_string(drug_name, sections_data)

    # Create a Document object with the extracted information
    document = Document(doc_id=set_id, text=drug_info_str, extra_info=extra_info or {})
    return document


class MedZipFileReader(BaseReader):
    def load_data(self, file, extra_info=None):
        documents = []
        with zipfile.ZipFile(file, "r") as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith(".xml"):
                    with zip_ref.open(file_name) as xml_file:
                        soup = BeautifulSoup(xml_file, "lxml-xml")
                        document = parse_drug_information(soup, extra_info)
                        if document:
                            documents.append(document)
        return documents

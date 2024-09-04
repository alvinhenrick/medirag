import zipfile
import re
from bs4 import BeautifulSoup
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from loguru import logger


def normalize_text(text):
    """
    Normalize the text by lowercasing, removing extra spaces, and stripping unnecessary characters.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def format_output_string(drug_name, sections_data):
    output = [f"Drug and Generic Names: {drug_name}"]
    for title, paragraphs in sections_data.items():
        output.append(f"{title}:")
        for paragraph in paragraphs:
            output.append(f" - {paragraph}")
        output.append("")  # Add a newline after each section
    return "\n".join(output)


def extract_names(manufactured_product):
    """
    Extracts both the main and generic drug names from the product.
    """
    drug_names = set()
    name_tag = manufactured_product.find("name")
    if name_tag:
        drug_names.add(name_tag.get_text(strip=True))
    as_generic = manufactured_product.find("asEntityWithGeneric")
    if as_generic:
        generic_name_tag = as_generic.find("genericMedicine").find("name")
        if generic_name_tag:
            drug_names.add(generic_name_tag.get_text(strip=True))
    return drug_names


def extract_drug_and_generic_names(structured_body):
    """
    Extracts all drug names from the structured body of the XML.
    """
    drug_names = set()
    for manufactured_product in structured_body.find_all("manufacturedProduct"):
        drug_names.update(extract_names(manufactured_product))
    return list(drug_names)


def extract_section_data(section):
    """
    Extracts title and paragraphs data from a section.
    """
    title_tag = section.find("title")
    if not title_tag:
        return None, []
    title_text = normalize_text(title_tag.get_text(strip=True))
    paragraphs = [normalize_text(p.get_text(strip=True)) for p in section.find_all("paragraph")]
    return title_text, paragraphs


def compile_sections_data(components):
    """
    Compiles data from all sections within components.
    """
    sections_data = {}
    for component in components:
        for section in component.find_all("section"):
            title_text, paragraphs_text = extract_section_data(section)
            if title_text:
                sections_data.setdefault(title_text, set()).update(paragraphs_text)
    return sections_data


def parse_drug_information(soup, extra_info=None):
    set_id_tag = soup.find("setId")
    if not set_id_tag or not set_id_tag.get("root"):
        logger.warning("Set ID not found or is missing in the XML.")
        return None

    structured_body = soup.find("structuredBody")
    drug_names = extract_drug_and_generic_names(structured_body)
    if not drug_names:
        logger.warning("No drug names found in the structured body.")
        return None

    components = structured_body.find_all("component")
    sections_data = compile_sections_data(components)
    drug_info_str = format_output_string(" | ".join(drug_names), sections_data)
    return Document(doc_id=set_id_tag.get("root"), text=drug_info_str, extra_info=extra_info or {})


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

import zipfile
from bs4 import BeautifulSoup
from html import unescape
from cleantext import clean
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from loguru import logger


def normalize_text(text):
    """
    Normalize the text using cleantext to ensure all unwanted characters and formatting are removed.
    """
    text = clean(
        text,
        no_line_breaks=True,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_currency_symbols=True,
        no_punct=True,
        no_numbers=False,
        no_digits=False,
    )
    return unescape(text)


def extract_section_data(section):
    """
    Extracts title and paragraphs data from a section.
    """
    title_tag = section.find("title")
    if not title_tag:
        logger.info("No title tag found in section.")
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
                sections_data.setdefault(title_text, []).extend(paragraphs_text)
    return sections_data


def parse_drug_information(soup):
    """
    Parses the structured body of the document to extract drug names and section texts.
    """
    structured_body = soup.find("structuredBody")
    if not structured_body:
        logger.warning("No structured body found in the document.")
        return None

    drug_names = extract_drug_names(structured_body)
    components = structured_body.find_all("component")
    sections_data = compile_sections_data(components)
    if not sections_data:
        logger.warning("No section data was compiled from the components.")
    drug_info_str = format_output_string(drug_names, sections_data)
    return Document(doc_id=soup.find("setId").get("root"), text=drug_info_str)


# Function to format the output string for drug information
def format_output_string(drug_names, sections_data):
    output = [f"Drug Names: {', '.join(drug_names)}"]
    for title, paragraphs in sections_data.items():
        output.append(f"{title}:")
        for paragraph in paragraphs:
            output.append(f" - {paragraph}")
        output.append("")  # Add a newline after each section
    return "\n".join(output)


# Function to extract drug names from the structured body
def extract_drug_names(structured_body):
    """
    Extracts drug names from the structured body, including generic names.
    """
    drug_names = set()
    for manufactured_product in structured_body.find_all("manufacturedProduct"):
        name_tag = manufactured_product.find("name")
        if name_tag:
            drug_names.add(normalize_text(name_tag.get_text()))
        as_generic = manufactured_product.find("asEntityWithGeneric")
        if as_generic:
            generic_name_tag = as_generic.find("genericMedicine").find("name")
            if generic_name_tag:
                drug_names.add(normalize_text(generic_name_tag.get_text()))
    if not drug_names:
        logger.warning("No drug names found.")
    return list(drug_names)


class MedZipFileReader(BaseReader):
    def load_data(self, file, extra_info=None):
        documents = []
        with zipfile.ZipFile(file, "r") as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith(".xml"):
                    with zip_ref.open(file_name) as xml_file:
                        soup = BeautifulSoup(xml_file, "lxml-xml")
                        document = parse_drug_information(soup)
                        if document:
                            documents.append(document)
                        else:
                            logger.error("Failed to parse drug information from file.")
        return documents

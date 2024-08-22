import zipfile
import re
from bs4 import BeautifulSoup
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader


def normalize_text(text):
    """Normalize the text by lowercasing, removing extra spaces, and stripping unnecessary characters."""
    text = text.lower()  # Lowercase the text
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()


def format_output_string(drug_name, sections_data):
    """Format the output string for document embedding."""
    output = [f"Drug Name: {drug_name}"]

    for title, paragraphs in sections_data.items():
        output.append(f"{title}:")
        for paragraph in paragraphs:
            output.append(f" - {paragraph}")
        output.append("")  # Add a newline after each section

    return "\n".join(output)


def parse_drug_information(soup, extra_info=None):
    # Extract the setId
    set_id = None
    set_id_tag = soup.find("setId")
    if set_id_tag:
        set_id = set_id_tag.get("root", None)

    if not set_id:
        return None

    # Ensure structured body exists
    structured_body = soup.find("structuredBody")
    if not structured_body:
        return None

    # Extract the drug name
    drug_name = None
    manufactured_product = structured_body.find("manufacturedProduct")
    if manufactured_product:
        inner_product = manufactured_product.find("manufacturedProduct")
        if inner_product:
            name_tag = inner_product.find("name")
            if name_tag:
                drug_name = name_tag.get_text(strip=True)

    if not drug_name:
        return None

    # Iterate over components and extract sections
    components = structured_body.find_all("component")
    sections_data = {}

    for component in components:
        sections = component.find_all("section")
        for section in sections:
            title_tag = section.find("title")
            title_text = normalize_text(title_tag.get_text(strip=True)) if title_tag else None
            if not title_text:
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
                    sections_data[title_text].extend(paragraphs_text)
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

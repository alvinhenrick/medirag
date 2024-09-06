from bs4 import BeautifulSoup

from medirag.core.reader import parse_drug_information


def test_xml_reader(data_dir):
    # Example usage:
    xml_file_path = data_dir.joinpath("BE27854A-A805-4300-9729-ACCD1B7F226F.xml")

    with open(xml_file_path) as xml_file:
        soup = BeautifulSoup(xml_file, "lxml-xml")
        document = parse_drug_information(soup)

    assert document is not None

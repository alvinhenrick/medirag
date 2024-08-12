from llama_index.core import SimpleDirectoryReader

from medirag.core.reader import MedZipFileReader


class DailyMedDocumentProcessor:
    def __init__(self, extracted_dir):
        self.extracted_dir = extracted_dir

    def load_documents(self):
        reader = SimpleDirectoryReader(
            input_dir=self.extracted_dir, recursive=True, file_extractor={".zip": MedZipFileReader()}
        )
        return reader.load_data()

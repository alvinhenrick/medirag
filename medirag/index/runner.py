from medirag.core.document_processor import DailyMedDocumentProcessor
from medirag.index.local import DailyMedIndexer

download_sources = [
    "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_rx_part1.zip",
]
# "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_rx_part2.zip",
# "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_rx_part3.zip",
# "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_rx_part4.zip",
# "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_rx_part5.zip"

# Initialize and manage data
from medirag.core.data_manager import DailyMedDataManager

data_manager = DailyMedDataManager(download_sources=download_sources)
data_manager.download_and_extract_zip()
extracted_dir = data_manager.get_extracted_dir()

# Process documents
document_processor = DailyMedDocumentProcessor(extracted_dir=extracted_dir)
documents = document_processor.load_documents()

# Index and query documents
indexer = DailyMedIndexer()
indexer.load_index(documents=documents)
indexer.save_index("./dm_spl_release_human_rx_part1")

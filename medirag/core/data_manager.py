import tempfile
import zipfile
from pathlib import Path
import requests
import shutil
from loguru import logger


class DailyMedDataManager:
    def __init__(self, download_sources):
        self.download_sources = download_sources
        # Create a temporary directory
        self.temp_dir = Path(tempfile.mkdtemp())
        self.extracted_dir = self.temp_dir / "common"
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.downloaded_files = []
        logger.info("Initialized DailyMedDataManager with temporary directories.")

    def download_zip(self, source):
        """Downloads a zip file from a URL or processes a local file path."""
        try:
            if source.startswith("http://") or source.startswith("https://"):
                logger.info(f"Downloading and processing: {source}")
                local_zip_path = self.temp_dir / Path(source).name
                with requests.get(source, stream=True, timeout=10) as r:
                    r.raise_for_status()
                    with open(local_zip_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=65536):
                            f.write(chunk)
                self.downloaded_files.append(local_zip_path)
            else:
                logger.info(f"Processing local file: {source}")
                local_zip_path = Path(source)
                self.downloaded_files.append(local_zip_path)
            return local_zip_path
        except requests.RequestException as e:
            logger.error(f"Failed to download {source}: {e}")
            return None

    def extract_zip(self, zip_path):
        """Extracts the zip file into the common subdirectory."""
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.extracted_dir)
                logger.info(f"Extracted {zip_path} successfully.")
        except zipfile.BadZipFile as e:
            logger.error(f"Failed to extract {zip_path}: {e}")

    def download_and_extract_zip(self):
        """Downloads and extracts all zip files."""
        for source in self.download_sources:
            zip_path = self.download_zip(source)
            if zip_path:
                self.extract_zip(zip_path)

    def get_extracted_dir(self):
        """Returns the directory containing extracted files."""
        return self.extracted_dir

    def cleanup(self):
        """Cleans up the temporary directory."""
        try:
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up temporary directories successfully.")
        except Exception as e:
            logger.error(f"Failed to clean up temporary directory: {e}")

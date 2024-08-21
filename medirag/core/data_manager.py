import tempfile
import zipfile
from pathlib import Path
import requests
import os


class DailyMedDataManager:
    def __init__(self, download_sources):
        self.download_sources = download_sources
        # Create a temporary directory
        self.temp_dir = Path(tempfile.mkdtemp())
        # Create a subdirectory for extracted files
        self.extracted_dir = self.temp_dir / "common"
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.downloaded_files = []

    def download_zip(self, source):
        """Downloads a zip file from a URL or processes a local file path."""
        if source.startswith("http://") or source.startswith("https://"):
            print(f"Downloading and processing: {source}")
            local_zip_path = self.temp_dir / Path(source).name
            with requests.get(source, stream=True, timeout=10) as r:
                r.raise_for_status()  # Ensure we notice bad responses
                with open(local_zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
            self.downloaded_files.append(local_zip_path)
        else:
            # Assume it's a local file path
            print(f"Processing local file: {source}")
            local_zip_path = Path(source)
            self.downloaded_files.append(local_zip_path)

        return local_zip_path

    def extract_zip(self, zip_path):
        """Extracts the zip file into the common subdirectory."""
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.extracted_dir)

    def download_and_extract_zip(self):
        """Downloads and extracts all zip files."""
        for source in self.download_sources:
            zip_path = self.download_zip(source)
            self.extract_zip(zip_path)

    def get_extracted_dir(self):
        """Returns the directory containing extracted files."""
        return self.extracted_dir

    def cleanup(self):
        """Cleans up the temporary directory."""
        for child in self.temp_dir.iterdir():
            if child.is_file():
                try:
                    child.unlink()
                except PermissionError:
                    print(f"Skipping file due to permission error: {child}")
            else:
                for sub_child in child.rglob("*"):
                    try:
                        if sub_child.is_file():
                            sub_child.unlink()
                        else:
                            sub_child.rmdir()
                    except PermissionError:
                        print(f"Skipping due to permission error: {sub_child}")
                try:
                    child.rmdir()
                except PermissionError:
                    print(f"Skipping directory due to permission error: {child}")
        try:
            os.rmdir(self.temp_dir)  # Use os.rmdir instead of rmdir from Path
        except PermissionError:
            print(f"Skipping directory due to permission error: {self.temp_dir}")

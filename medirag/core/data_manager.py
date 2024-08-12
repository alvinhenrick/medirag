import tempfile
import zipfile
from pathlib import Path
import requests


class DailyMedDataManager:
    def __init__(self, download_sources):
        self.download_sources = download_sources
        self.temp_dir = Path(tempfile.TemporaryDirectory().name)

    def download_and_extract_zip(self):
        for source in self.download_sources:
            if source.startswith("http://") or source.startswith("https://"):
                print(f"Downloading and processing: {source}")
                local_zip_path = self.temp_dir / Path(source).name
                with requests.get(source, stream=True, timeout=10) as r:
                    with open(local_zip_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            else:
                # Assume it's a local file path
                print(f"Processing local file: {source}")
                local_zip_path = Path(source)

            # Extract the zip file
            with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
                zip_ref.extractall(self.temp_dir)

    def get_extracted_dir(self):
        return self.temp_dir

    def cleanup(self):
        # Clean up the temporary directory
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
            self.temp_dir.rmdir()
        except PermissionError:
            print(f"Skipping directory due to permission error: {self.temp_dir}")

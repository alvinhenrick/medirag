import os

from dotenv import load_dotenv
import kdbai_client as kdbai

load_dotenv()

session = kdbai.Session(api_key=os.getenv("KDBAI_API_KEY"), endpoint=os.getenv("KDBAI_ENDPOINT"))

schema = [
    {"name": "document_id", "type": "bytes"},
    {"name": "text", "type": "bytes"},
    {"name": "embedding", "type": "float32s"},
]

indexFlat = {
    "name": "flat_index",
    "type": "flat",
    "column": "embedding",
    "params": {"dims": 768, "metric": "CS"},
}

KDBAI_TABLE_NAME = "daily_med_v2"
database = session.database("default")
# First ensure the table does not already exist
for table in database.tables:
    if table.name == KDBAI_TABLE_NAME:
        ## table.drop()
        print("exist")
        break
table = database.create_table(KDBAI_TABLE_NAME, schema=schema, indexes=[indexFlat])

import os

from dotenv import load_dotenv
import kdbai_client as kdbai

load_dotenv()

session = kdbai.Session(api_key=os.getenv("KDBAI_API_KEY"), endpoint=os.getenv("KDBAI_ENDPOINT"))

schema = dict(
    columns=[
        dict(name="document_id", pytype="bytes"),
        dict(name="text", pytype="bytes"),
        dict(
            name="embedding",
            vectorIndex=dict(type="flat", metric="CS", dims=768),
        ),
    ]
)

KDBAI_TABLE_NAME = "daily_med_v1"

table = session.database("default").table(KDBAI_TABLE_NAME)
print()
# # First ensure the table does not already exist
# if KDBAI_TABLE_NAME in session.list():
#     pass
#     # session.table(KDBAI_TABLE_NAME).drop()
# else:
#     # Create the table
#     table = session.database("default").create_table(KDBAI_TABLE_NAME, schema=schema)

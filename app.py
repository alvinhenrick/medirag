"""
MediRAG Gradio app — DSPy 3 + LanceDB + PubMedBERT.
"""

import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import dspy
import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from loguru import logger

from medirag.cache.local import LocalSemanticCache
from medirag.index.lance import LanceIndexer
from medirag.rag.dspy import DspyRAG
from medirag.rag.pipeline import answer_stream


load_dotenv()

LANCE_DB_PATH = os.getenv("LANCE_DB_PATH", "./lance_db")
LANCE_TABLE = os.getenv("LANCE_TABLE", "spl")
CACHE_FILE = os.getenv("CACHE_FILE", "rag_cache.json")
HF_DATASET = os.getenv("HF_DATASET")  # e.g. "alvinhenrick/medirag-dailymed"


def _bootstrap_index_from_hf(dataset_repo: str, local_path: Path) -> None:
    logger.info(f"Bootstrapping index from {dataset_repo} → {local_path}")
    archive = hf_hub_download(
        repo_id=dataset_repo,
        filename="lance_db.tar",
        repo_type="dataset",
    )
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=local_path.parent) as tmp:
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        children = list(Path(tmp).iterdir())
        if len(children) != 1 or not children[0].is_dir():
            raise RuntimeError(f"unexpected tar contents: {children}")
        shutil.move(str(children[0]), str(local_path))
    logger.info(f"Index ready at {local_path}")


if HF_DATASET and not Path(LANCE_DB_PATH).exists():
    _bootstrap_index_from_hf(HF_DATASET, Path(LANCE_DB_PATH))

MODELS = {
    "GPT-4o mini (fast, cheap)": "openai/gpt-4o-mini",
    "GPT-4o (" " quality)": "openai/gpt-4o",
}
DEFAULT_MODEL_LABEL = "GPT-4o mini (fast, cheap)"


logger.info(f"Loading LanceDB index from {LANCE_DB_PATH} (table={LANCE_TABLE})")
indexer = LanceIndexer(db_path=LANCE_DB_PATH, table_name=LANCE_TABLE)
if indexer.table is None:
    logger.warning(f"No index found at {LANCE_DB_PATH}. Build one with `uv run python -m medirag.index.runner`.")

semantic_cache = LocalSemanticCache(
    model_name="sentence-transformers/all-mpnet-base-v2",
    dimension=768,
    json_file=CACHE_FILE,
)


def clear_cache() -> None:
    semantic_cache.clear()
    gr.Info("Cache cleared", duration=1)


async def ask(query: str, model_label: str, top_k: int):
    if not query or not query.strip():
        yield "Please enter a question."
        return

    model_id = MODELS.get(model_label, MODELS[DEFAULT_MODEL_LABEL])
    lm = dspy.LM(model_id, max_tokens=1500)

    rag = DspyRAG(indexer=indexer, k=int(top_k), hybrid=True)

    # DSPy 3 requires per-task configuration; Gradio spawns a new task per request,
    # so we use `dspy.context()` instead of a global `dspy.configure()`.
    with dspy.context(lm=lm):
        accumulated = ""
        async for chunk in answer_stream(rag, semantic_cache, query):
            accumulated += chunk
            yield accumulated


css = """
h1 { text-align: center; display: block; }
"""


with gr.Blocks(css=css, title="MediRAG") as app:
    gr.Markdown("# MediRAG — Ask about your medication")
    gr.Markdown(
        "Replace the tiny-print leaflet that came with your pills. Ask anything: "
        "side effects, ingredients, dosing, interactions, what the pill looks like. "
        "Answers are grounded in FDA DailyMed drug labels."
    )

    with gr.Row():
        model_dd = gr.Dropdown(
            choices=list(MODELS.keys()),
            value=DEFAULT_MODEL_LABEL,
            label="Model",
            scale=2,
        )
        top_k_dd = gr.Dropdown(
            choices=[3, 5, 7, 10],
            value=5,
            label="Documents to retrieve",
            scale=1,
        )
        clear_bt = gr.Button("Clear cache", scale=1)

    input_text = gr.Textbox(
        lines=2,
        label="Your question",
        placeholder="e.g. What are the side effects of metformin? Can I take ibuprofen if I'm on warfarin?",
    )
    output_text = gr.Markdown(label="Answer")
    submit_bt = gr.Button("Ask", variant="primary")

    submit_bt.click(fn=ask, inputs=[input_text, model_dd, top_k_dd], outputs=output_text)
    input_text.submit(fn=ask, inputs=[input_text, model_dd, top_k_dd], outputs=output_text)
    clear_bt.click(fn=clear_cache)


if __name__ == "__main__":
    app.launch()

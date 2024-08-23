import gradio as gr
from huggingface_hub import scan_cache_dir

from medirag.cache.local import SemanticCaching
from medirag.index.local import DailyMedIndexer
from medirag.rag.qa import RAG, DailyMedRetrieve
import dspy
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Initialize the components
data_dir = Path("data")
index_path = data_dir.joinpath("dm_spl_release_human_rx_part1")
indexer = DailyMedIndexer(persist_dir=index_path)
indexer.load_index()
rm = DailyMedRetrieve(daily_med_indexer=indexer)

turbo = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=4000)
dspy.settings.configure(lm=turbo, rm=rm)

rag = RAG(k=5)
sm = SemanticCaching(model_name='sentence-transformers/all-mpnet-base-v2', dimension=768,
                     json_file='rag_test_cache.json', cosine_threshold=.90, rag=rag)
sm.load_cache()


def ask_med_question(query):
    response = sm.ask(query)
    return response


# Set up the Gradio interface
with gr.Blocks() as app:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Image("doc/images/MediRag.png", width=100, height=100, min_width=75,
                     show_label=False, show_download_button=False, show_share_button=False,
                     show_fullscreen_button=False)
        with gr.Column(scale=2):
            gr.Markdown("# DailyMed RAG Question Answering")
    with gr.Row():
        gr.Markdown("### Ask any question about medication usage and get answers based on DailyMed data.")

    input_text = gr.Textbox(lines=2, label="Question", placeholder="Enter your question about a drug...")
    button = gr.Button("Submit")
    output_text = gr.Textbox(interactive=False, label="Response", lines=10)
    button.click(fn=ask_med_question, inputs=input_text, outputs=output_text)

app.launch()

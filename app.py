import gradio as gr
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

turbo = dspy.OpenAI(model='gpt-4o')
dspy.settings.configure(lm=turbo, rm=rm)

rag = RAG(k=5)
sm = SemanticCaching(model_name='sentence-transformers/all-mpnet-base-v2', dimension=768,
                     json_file='rag_test_cache.json', cosine_threshold=.85, rag=rag)
sm.load_cache()


def ask_med_question(query):
    response = sm.ask(query)
    return response


# Set up the Gradio interface
with gr.Blocks() as app:
    gr.Row([
        gr.Markdown("# Medical RAG Question Answering")
    ])
    gr.Row([
        gr.Markdown("## Ask any question about medication usage and get answers based on DailyMed data.")
    ])
    with gr.Row():
        input_text = gr.Textbox(lines=2, placeholder="Enter your question about a drug...")
    with gr.Row():
        button = gr.Button("Submit")
    with gr.Row():
        output_text = gr.Textbox(interactive=False, label="Response", show_label=False,
                                 lines=10)

    button.click(fn=ask_med_question, inputs=input_text, outputs=output_text)

app.launch()

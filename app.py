import dspy
import gradio as gr
from dotenv import load_dotenv

from medirag.cache.local import LocalSemanticCache
from medirag.index.kdbai import KDBAIDailyMedIndexer
from medirag.rag.dspy import DspyRAG, DailyMedRetrieve
from medirag.rag.llama_index import WorkflowRAG
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

from medirag.rag.qa_rag import QuestionAnswerRunner

# Load Env
load_dotenv()

# Initialize the Retriever
indexer = KDBAIDailyMedIndexer()
indexer.load_index()
rm = DailyMedRetrieve(indexer=indexer)

# Set the LLM model for DSPy
llm = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=4000)
dspy.settings.configure(lm=llm, rm=rm)

# Set the LLM model for LlamaIndex
Settings.llm = OpenAI(model="gpt-3.5-turbo")

sm = LocalSemanticCache(
    model_name="sentence-transformers/all-mpnet-base-v2", dimension=768, json_file="rag_test_cache.json"
)

# Initialize RAGWorkflow with indexer
dspy_rag = DspyRAG(k=5)
llama_index_rag = WorkflowRAG(indexer=indexer, timeout=60, with_reranker=False, top_k=5, top_n=3)


def clear_cache():
    sm.clear()
    gr.Info("Cache is cleared", duration=1)


async def ask_med_question(query: str, enable_stream: bool):
    if enable_stream:
        qa = QuestionAnswerRunner(sm=sm, rag=llama_index_rag)
    else:
        qa = QuestionAnswerRunner(sm=sm, rag=dspy_rag)
    accumulated_response = ""

    response = qa.ask(query, enable_stream=enable_stream)

    async for chunk in response:
        accumulated_response += chunk
        yield accumulated_response


css = """
h1 {
    text-align: center;
    display:block;
}
#md {margin-top: 70px}
"""

# Set up the Gradio interface with a checkbox for enabling streaming
with gr.Blocks(css=css) as app:
    gr.Markdown("# DailyMed RAG")
    with gr.Row():
        with gr.Column(scale=1, min_width=100):
            gr.Image(
                "doc/images/MediRag.png",
                width=100,
                min_width=100,
                show_label=False,
                show_download_button=False,
                show_share_button=False,
                show_fullscreen_button=False,
            )
        with gr.Column(scale=10):
            gr.Markdown(
                "### Ask any question about medication usage and get answers based on DailyMed data.", elem_id="md"
            )
    with gr.Row():
        enable_stream_chk = gr.Checkbox(label="Enable Streaming", value=False)
        clear_cache_bt = gr.Button("Clear Cache")

    input_text = gr.Textbox(lines=2, label="Question", placeholder="Enter your question about a drug...")
    output_text = gr.Textbox(interactive=False, label="Response", lines=10)
    submit_bt = gr.Button("Submit")

    # Update the button click function to include the checkbox value
    submit_bt.click(fn=ask_med_question, inputs=[input_text, enable_stream_chk], outputs=output_text)

    # Update the button click function to include the checkbox value
    clear_cache_bt.click(fn=clear_cache)

app.launch()

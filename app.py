import dspy
import gradio as gr
from dotenv import load_dotenv

from medirag.cache.local import SemanticCaching
from medirag.index.kdbai import KDBAIDailyMedIndexer
from medirag.rag.qa import RAG, DailyMedRetrieve
from medirag.rag.wf import RAGWorkflow
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

load_dotenv()

# Initialize the components
indexer = KDBAIDailyMedIndexer()
indexer.load_index()
rm = DailyMedRetrieve(indexer=indexer)

turbo = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=4000)
dspy.settings.configure(lm=turbo, rm=rm)
# Set the LLM model
Settings.llm = OpenAI(model="gpt-3.5-turbo")

sm = SemanticCaching(
    model_name="sentence-transformers/all-mpnet-base-v2", dimension=768, json_file="rag_test_cache.json"
)

# Initialize RAGWorkflow with indexer
rag = RAG(k=5)
streaming_rag = RAGWorkflow(indexer=indexer, timeout=60, with_reranker=False, top_k=5, top_n=3)


def clear_cache():
    sm.clear()
    gr.Info("Cache is cleared", duration=1)


async def ask_med_question(query: str, enable_stream: bool):
    # Check the cache first
    response = sm.lookup(question=query, cosine_threshold=0.9)
    if response:
        # Return cached response if found
        yield response
    else:
        if enable_stream:
            # Stream response using RAGWorkflow
            result = await streaming_rag.run(query=query)

            # Handle streaming response
            if hasattr(result, "async_response_gen"):
                accumulated_response = ""

                async for chunk in result.async_response_gen():
                    accumulated_response += chunk
                    yield accumulated_response  # Accumulate and yield the updated response

                # Save the accumulated response to the cache after streaming is complete
                sm.save(query, accumulated_response)
            elif isinstance(result, str):
                # Handle non-streaming string response
                yield result
                sm.save(query, result)
            else:
                # Handle unexpected response types
                print("Unexpected response type:", result)
                yield "An unexpected error occurred."
        else:
            # Use RAG without streaming
            response = rag(query).answer
            yield response

            # Save the response in the cache
            if response:
                sm.save(query, response)


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

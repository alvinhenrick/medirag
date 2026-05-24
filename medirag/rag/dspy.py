"""
DSPy 3.x RAG module: input guard → retrieve → ChainOfThought → output guard.

The module owns the LanceIndexer directly — no global `dspy.settings.rm` dance,
no `dspy.Retrieve` wrapper. Streaming is exposed via `dspy.streamify` with a
StreamListener on the `answer` field.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Callable

import dspy

from medirag.guardrail.input import InputGuardrail
from medirag.guardrail.output import OutputGuardrail
from medirag.index.lance import LanceIndexer


class GenerateAnswer(dspy.Signature):
    """
    Answer the patient's question using only the provided context.

    Rules:
      - Use plain language a non-medical patient can understand.
      - Cite which section the answer comes from (e.g., "from the Adverse Reactions section").
      - If multiple drugs are mentioned, compare them clearly.
      - Do not give a diagnosis or personalised treatment advice; suggest consulting a pharmacist or doctor for personal medical decisions.
      - If the context does not contain the answer, say so plainly.
    """

    context: str = dspy.InputField(desc="Drug label sections relevant to the question")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Patient-friendly answer grounded in the context")


class DspyRAG(dspy.Module):
    def __init__(self, indexer: LanceIndexer, k: int = 5, hybrid: bool = True):
        super().__init__()
        self.indexer = indexer
        self.k = k
        self.hybrid = hybrid
        self.input_guard = dspy.Predict(InputGuardrail)
        self.output_guard = dspy.Predict(OutputGuardrail)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def _retrieve(self, question: str) -> str:
        hits = self.indexer.retrieve(question, top_k=self.k, hybrid=self.hybrid)
        return "\n\n---\n\n".join(h.text for h in hits)

    def forward(self, question: str) -> dspy.Prediction:
        if self.input_guard(user_input=question).should_block:
            return dspy.Prediction(context="", answer="I'm sorry, I can't respond to that.")

        context = self._retrieve(question)
        prediction = self.generate_answer(context=context, question=question)

        blocked = self.output_guard(user_input=question, bot_response=prediction.answer).should_block
        if blocked:
            return dspy.Prediction(
                context=context,
                answer="I'm sorry, I don't have relevant information to respond to that.",
            )
        return dspy.Prediction(context=context, answer=prediction.answer)


StreamedRAG = Callable[..., AsyncIterator[Any]]


def _streamify_answer(rag: DspyRAG) -> StreamedRAG:
    """
    Wrap dspy.streamify with the correct return type.

    DSPy types streamify as returning `Callable[..., Awaitable[Any]]`, but the
    underlying function is an async generator. We type-correct it here so the
    rest of the module sees the real shape: a callable returning an AsyncIterator.
    """
    listener = dspy.streaming.StreamListener(signature_field_name="answer")
    return dspy.streamify(rag, stream_listeners=[listener])  # type: ignore[return-value]


async def stream_answer(rag: DspyRAG, question: str) -> AsyncIterator[str]:
    """
    Async generator yielding answer chunks as the LM streams them.

    Falls back to the final prediction if no chunks arrive (e.g. when the input guardrail blocks before the answer LM is
    invoked).
    """
    streamed = _streamify_answer(rag)

    saw_chunk = False
    final_text = ""
    async for event in streamed(question=question):
        if isinstance(event, dspy.streaming.StreamResponse):
            saw_chunk = True
            yield event.chunk
        elif isinstance(event, dspy.Prediction):
            final_text = event.answer or ""

    if not saw_chunk and final_text:
        yield final_text

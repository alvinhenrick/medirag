from typing import Optional

import dspy
from dsp import dotdict

from medirag.guardrail.input import InputGuardrail
from medirag.guardrail.output import OutputGuardrail
from medirag.index.ptc import Indexer


class DailyMedRetrieve(dspy.Retrieve):
    def __init__(self, indexer: Indexer, k: int = 3):
        super().__init__(k=k)
        self.indexer = indexer

    def forward(
        self,
        query_or_queries: Optional[str | list[str]] = None,
        query: Optional[str] = None,
        k: Optional[int] = None,
        by_prob: bool = True,
        with_metadata: bool = False,
        with_reranker: bool = False,
        **kwargs,
    ) -> list[dspy.Prediction]:
        if query_or_queries is None:
            if query is None:
                raise ValueError("Either query_or_queries or query must be provided.")
            query_or_queries = query

        actual_k = k if k is not None else self.k
        results = self.indexer.retrieve(query=query_or_queries, top_k=actual_k, with_reranker=with_reranker)
        return [dotdict({"long_text": result.text}) for result in results]  # noqa


class GenerateAnswer(dspy.Signature):
    """
    You are an AI assistant designed to answer questions based on provided context:
      - Do not provide any form of diagnosis or treatment advice.
    """

    context = dspy.InputField(desc="Contains relevant facts about drug labels")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Answer with detailed summary")


class DspyRAG(dspy.Module):
    def __init__(self, k: int = 3, with_reranker: bool = False):
        super().__init__()
        self.input_guardrail = dspy.TypedPredictor(InputGuardrail)
        self.output_guardrail = dspy.TypedPredictor(OutputGuardrail)

        self.retrieve = dspy.Retrieve(k=k)
        self.with_reranker = with_reranker
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question, with_reranker=self.with_reranker).passages

        in_gr = self.input_guardrail(user_input=question)

        if in_gr.should_block == "Yes":
            return dspy.Prediction(context=question, answer="I'm sorry, I can't respond to that.")

        prediction = self.generate_answer(context=context, question=question)

        out_gr = self.output_guardrail(user_input=question, bot_response=prediction.answer)

        if out_gr.should_block == "Yes":
            return dspy.Prediction(
                context=context, answer="I'm sorry, I don't have relevant information to respond to that."
            )

        return dspy.Prediction(context=context, answer=prediction.answer)

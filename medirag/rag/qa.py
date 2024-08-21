from typing import Union, List, Optional

import dspy
from dsp import dotdict

from medirag.guardrail.input import InputGuardrail
from medirag.guardrail.output import OutputGuardrail
from medirag.index.local import DailyMedIndexer


class DailyMedRetrieve(dspy.Retrieve):
    def __init__(self, daily_med_indexer: DailyMedIndexer, k: int = 3):
        super().__init__(k=k)  # Correctly called at the beginning
        self.daily_med_indexer = daily_med_indexer

    def forward(
            self,
            query_or_queries: Union[str, List[str]],
            k: Optional[int] = None,
            by_prob: bool = True,
            with_metadata: bool = False,
            **kwargs,
    ) -> dspy.Prediction:
        results = self.daily_med_indexer.retrieve(query=query_or_queries, top_k=k)
        return [dotdict({"long_text": result.text}) for result in results]  # noqa


class GenerateAnswer(dspy.Signature):
    """
    You are an AI assistant designed to answer questions based on provided context:
      - Do not provide any form of diagnosis or treatment advice.
    """
    context = dspy.InputField(desc="Contains relevant facts about drug labels")
    question = dspy.InputField()
    answer = dspy.OutputField()


class RAG(dspy.Module):
    def __init__(self, k: int = 3):
        super().__init__()
        self.input_guardrail = dspy.TypedPredictor(InputGuardrail)
        self.output_guardrail = dspy.TypedPredictor(OutputGuardrail)

        self.retrieve = dspy.Retrieve(k=k)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages

        in_gr = self.input_guardrail(user_input=question)

        if in_gr.should_block == 'Yes':
            return dspy.Prediction(context=question, answer="I'm sorry, I can't respond to that.")

        prediction = self.generate_answer(context=context, question=question)

        out_gr = self.output_guardrail(user_input=question, bot_response=prediction.answer)

        if out_gr.should_block == 'Yes':
            return dspy.Prediction(context=context,
                                   answer="I'm sorry, I don't have relevant information to respond to that.")

        return dspy.Prediction(context=context, answer=prediction.answer)

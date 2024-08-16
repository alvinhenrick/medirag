from typing import Union, List, Optional

import dspy
from dsp import dotdict

from medirag.index.ingest import DailyMedIndexer


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
    context = dspy.InputField(desc="Contains relevant facts about drug labels")
    question = dspy.InputField()
    answer = dspy.OutputField(
        desc="A concise response, summarizing key drug information, side effects, or interactions.")


class RAG(dspy.Module):
    def __init__(self, k: int = 3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=k)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages

        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

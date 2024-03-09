from typing import Callable, List, Optional, Union

import dspy
from dspy import InputField, OutputField

from ragatouille.data.corpus_processor import CorpusProcessor
from ragatouille.data.preprocessors import llama_index_sentence_splitter


class Query(dspy.Signature):
    """Create a query in the language of the document"""

    document = InputField(desc="document")
    query = OutputField(desc="A short query that may be used to query the document.")


class SyntheticQueryGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_query = dspy.ChainOfThought(Query)

    def forward(self, document: List[dict]):
        doc = document["content"]
        query = self.generate_query(document=doc).query
        pairs = [query, doc]

        return pairs


class DSPyDataProcessor:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-0125-preview",
        max_tokens: int = 2000,
        document_splitter_fn: Optional[Callable] = llama_index_sentence_splitter,
        preprocessing_fn: Optional[Union[Callable, list[Callable]]] = None,
    ):
        self.api_key = api_key
        self.model = model
        self_max_tokens = max_tokens
        self.corpus_processor = CorpusProcessor(document_splitter_fn, preprocessing_fn)

    def process_corpus(
        self,
        documents: List[str],
        document_ids: Optional[list[str]] = None,
        **splitter_kwargs,
    ) -> List[List[str]]:
        documents = self.corpus_processor.process_corpus(
            documents, document_ids, **splitter_kwargs
        )

        # Add other LLM providers
        llm = dspy.OpenAI(
            model=self.model,
            max_tokens=self.max_tokens,
            model_type="chat",
            api_key=self.api_key,
        )

        dspy.settings.configure(lm=llm)
        query_generator = SyntheticQueryGenerator()
        pairs = [query_generator.forward(doc) for doc in documents]

        return pairs

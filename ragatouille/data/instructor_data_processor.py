from typing import Callable, List, Optional, Union

from ragatouille.data.corpus_processor import CorpusProcessor
from ragatouille.data.preprocessors import llama_index_sentence_splitter

import instructor
import random

from openai import OpenAI
from pydantic import BaseModel, Field


class QueryForPassage(BaseModel):
    hypothetical_questions: List[str] = Field(
        default_factory=list,
        description="A wide variety of hypothetical questions that this document could answer.",
    )
    hypothetical_queries: List[str] = Field(
        default_factory=list,
        description="A wide variety of hypothetical queries that this document would be relevant to, in the context of a search engine or a retrieval pipeline.",
    )


class InstructorDataProcessor:
    def __init__(
        self,
        documents: list[str],
        api_key: str,
        model: str = "gpt-4-1106-preview",
        document_splitter_fn: Optional[Callable] = llama_index_sentence_splitter,
        preprocessing_fn: Optional[Union[Callable, list[Callable]]] = None,
    ):
        self.documents = documents
        self.api_key = api_key
        self.model = model
        self.corpus_processor = CorpusProcessor(document_splitter_fn, preprocessing_fn)

    def process_corpus(
        self,
        documents: list[str],
        document_ids: Optional[list[str]] = None,
        num_questions: int = 1,
        num_queries: int = 1,
        **splitter_kwargs,
    ) -> List[List[str]]:

        documents = self.corpus_processor.process_corpus(
            documents, document_ids, **splitter_kwargs
        )

        client = instructor.patch(OpenAI(api_key=self.api_key))

        candidate_queries = []
        for doc in documents:
            candidate = client.chat.completions.create(
                model=self.model,
                response_model=QueryForPassage,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert AI assisting us in creating a high quality, diverse synthetic dataset to train Information Retrieval models. 
                        Your role is to analyse the document chunk given to you and provide us with high quality potential queries.""",
                    },
                    {"role": "user", "content": doc},
                ],
            )
            candidate_queries.append(candidate)

        pairs = []
        random.seed(42)

        for candidates, doc in zip(candidate_queries, documents):
            candidates = candidates.model_dump()
            queries = random.sample(candidates["hypothetical_questions"], num_questions)
            queries += random.sample(candidates["hypothetical_queries"], num_queries)
            for q in queries:
                pairs.append([q, doc])
        return pairs

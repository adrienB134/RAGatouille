from typing import Callable, List, Optional, Union

import dspy
from colbert import Trainer
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert.infra.run import Run
from dspy.teleprompt import BootstrapFinetune
from tqdm import tqdm

from ragatouille.data import CorpusProcessor, SyntheticQueryGenerator
from ragatouille.data.preprocessors import llama_index_sentence_splitter
from ragatouille.RAGPretrainedModel import RAGPretrainedModel
from ragatouille.RAGTrainer import RAGTrainer
from ragatouille.udapdr import reranker_training


class RAGInDomainTrainer:
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
        self.max_tokens = max_tokens
        self.corpus_processor = CorpusProcessor(document_splitter_fn, preprocessing_fn)

    def generate_data(
        self,
        first_stage_corpus,
        second_stage_corpus,
        second_stage_model: str = "google/flan-t5-small",
        **splitter_kwargs,
    ) -> str:
        print("#> Processing first stage documents\n")
        first_stage_documents = self.corpus_processor.process_corpus(
            first_stage_corpus,
            **splitter_kwargs,
        )
        second_stage_trainset = [
            dspy.Example(document=document).with_inputs("document")
            for document in tqdm(first_stage_documents)
        ]

        print(f"\n#> Finetuning {second_stage_model}\n")
        # TODO Put this in func args
        finetuning_config = dict(
            target=second_stage_model,
            epochs=1,
            bf16=False,
            bsize=4,
            accumsteps=2,
            lr=5e-5,
        )

        # TODO Add other LLM providers
        teacher_llm = dspy.OpenAI(
            model=self.model,
            max_tokens=self.max_tokens,
            model_type="chat",
            api_key=self.api_key,
        )

        dspy.settings.configure(lm=teacher_llm)
        tp2 = BootstrapFinetune(metric=None)
        t5_program = tp2.compile(
            SyntheticQueryGenerator(),
            teacher=SyntheticQueryGenerator(),
            trainset=second_stage_trainset,
            **finetuning_config,
        )

        # Deactivate chain of thought prompting. Use the model directly. (Faster and similar quality.)
        for p in t5_program.predictors():
            p.activated = False

        print(
            f"\n#> Generating synthetic queries with finetuned {second_stage_model}\n"
        )
        second_stage_documents = self.corpus_processor.process_corpus(
            second_stage_corpus,
            **splitter_kwargs,
        )
        third_stage_trainset = [
            t5_program.forward(doc) for doc in tqdm(second_stage_documents)
        ]

        print("\n#> Preparing reranker data\n")  # TODO To delete or rewrite later
        third_stage_trainset = RAGTrainer(
            "udapdr", "colbert-ir/ColBERTv2.0"
        ).prepare_training_data(
            raw_data=third_stage_trainset,
            all_documents=second_stage_documents,
        )

        return third_stage_trainset

    def train_rerankers(self):
        reranker_training()

    def train(
        self,
        corpus: List[str],
        model_name: str,
        pretrained_model_name: str,
        use_reranker: bool = False,
        training_params: Optional[dict] = dict(),
        language_code: str = "en",
        n_usable_gpus: int = -1,
        first_stage_split: int = 2,
        second_stage_split: int = 4,
        second_stage_model: str = "google/flan-t5-small",
        **splitter_kwargs,
    ) -> str:
        data = self.generate_data(
            first_stage_corpus=corpus[:first_stage_split],
            second_stage_corpus=corpus[:second_stage_split],
            second_stage_model=second_stage_model,
            **splitter_kwargs,
        )

        if use_reranker:
            self.train_rerankers()

        # trained_model = RAGTrainer(
        #     model_name=model_name,
        #     pretrained_model_name=pretrained_model_name,
        #     language_code=language_code,
        #     n_usable_gpus=n_usable_gpus,
        # ).train(data_directory=data, **training_params)

        with Run().context(RunConfig(nranks=1)):
            triples = "./data/distillation.json"
            queries = "./data/queries.train.colbert.tsv"
            collection = "./data/corpus.train.colbert.tsv"

            config = ColBERTConfig(
                bsize=1,
                lr=1e-05,
                warmup=20_000,
                doc_maxlen=180,
                dim=128,
                attend_to_mask_tokens=False,
                nway=2,
                accumsteps=1,
                similarity="cosine",
                use_ib_negatives=True,
            )
            trainer = Trainer(
                triples=triples, queries=queries, collection=collection, config=config
            )

            trained_model = trainer.train(checkpoint=pretrained_model_name)
        return trained_model

    def index(
        self,
        corpus: List[str],
        model_name: str,
        pretrained_model_name: str,
        training_params: Optional[dict] = dict(),
        language_code: str = "en",
        n_usable_gpus: int = -1,
        first_stage_split: int = 2,
        second_stage_split: int = 4,
        second_stage_model: str = "google/flan-t5-small",
        **splitter_kwargs,
    ):
        trained_model = self.train(
            self,
            corpus=corpus,
            model_name=model_name,
            pretrained_model_name=pretrained_model_name,
            training_params=training_params,
            language_code=language_code,
            n_usable_gpus=n_usable_gpus,
            first_stage_split=first_stage_split,
            second_stage_split=second_stage_split,
            second_stage_model=second_stage_model,
            **splitter_kwargs,
        )

        model = RAGPretrainedModel().from_pretrained(trained_model)
        # TODO expose args
        model.index(corpus)

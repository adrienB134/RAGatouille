import json
import os
import random

from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from torch.utils.data import DataLoader
from tqdm import tqdm


def reranker_training() -> str:
    print("\n#> Training reranker\n")

    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    train_batch_size = 2
    num_epochs = 1
    warmup_steps = 5000
    model = CrossEncoder(model_name, num_labels=1, max_length=512)

    data_folder = "./data/"
    os.makedirs(data_folder, exist_ok=True)

    corpus = {}
    collection_filepath = os.path.join(data_folder, "corpus.train.colbert.tsv")
    with open(collection_filepath, "r", encoding="utf8") as collection:
        for line in collection:
            pid, passage = line.strip().split("\t")
            corpus[int(pid)] = passage

    queries = {}
    queries_filepath = os.path.join(data_folder, "queries.train.colbert.tsv")
    with open(queries_filepath, "r", encoding="utf8") as synthetic_queries:
        for line in synthetic_queries:
            qid, query = line.strip().split("\t")
            queries[int(qid)] = query

    train_samples = []
    dev_samples = {}
    dev_sample_size = 50

    train_dev_filepath = os.path.join(data_folder, "triples.train.colbert.jsonl")

    with open(train_dev_filepath, "rt") as triples:
        for line in triples:
            qid, pos_id, neg_id = json.loads(line)

            if qid not in dev_samples:
                dev_samples[qid] = {
                    "query": queries[qid],
                    "positive": set(),
                    "negative": set(),
                }

            if qid in dev_samples:
                dev_samples[qid]["positive"].add(corpus[pos_id])
                dev_samples[qid]["negative"].add(corpus[neg_id])

        dev_samples = dict(random.sample(sorted(dev_samples.items()), dev_sample_size))

    pos_neg_ratio = 4  # ! to remove
    with open(train_dev_filepath, "rt") as triples:
        cnt = 0
        for line in tqdm(triples):
            qid, pos_id, neg_id = json.loads(line)

            if qid in dev_samples:
                continue

            query = queries[qid]
            if (cnt % (pos_neg_ratio + 1)) == 0:
                passage = corpus[pos_id]
                label = 1
            else:
                passage = corpus[neg_id]
                label = 0

            train_samples.append(InputExample(texts=[query, passage], label=label))
            cnt += 1

    train_dataloader = DataLoader(
        train_samples, shuffle=True, batch_size=train_batch_size
    )
    evaluator = CERerankingEvaluator(dev_samples, name="dev")

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=10000,
        warmup_steps=warmup_steps,
        use_amp=True,
    )

    print("\n#> Generating distillation triples\n")
    distillation = {}
    with open(train_dev_filepath, "rt") as triples:
        for line in tqdm(triples):
            qid, pos_id, neg_id = json.loads(line)

            if qid not in distillation:
                distillation[qid] = {
                    "passage": list(),
                }
                distillation[qid]["passage"].append(qid)
                distillation[qid]["passage"].append(
                    [
                        pos_id,
                        float(model.rank(queries[qid], {corpus[pos_id]})[0]["score"]),
                    ]
                )

            if qid in distillation:
                distillation[qid]["passage"].append(
                    [
                        neg_id,
                        float(model.rank(queries[qid], {corpus[neg_id]})[0]["score"]),
                    ]
                )

    output_filepath = "./data/distillation.json"
    output_file = open(output_filepath, "w")
    print("\n#> Generating output file\n")
    for qid in distillation.values():
        json.dump(qid["passage"], output_file)
        output_file.write("\n")

    print(f"\n\nDistillation triples saved: {output_filepath}\n")

    return output_filepath

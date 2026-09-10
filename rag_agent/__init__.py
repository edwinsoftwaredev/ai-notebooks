import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import dask
import dask.dataframe as ddf
import huggingface_hub
import numpy as np
import pandas as pd
import torch
import wandb
from kaggle_secrets import UserSecretsClient  # pyright: ignore
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from rag_agent.enums import (
    DATASET_PATH,
    INDEX_FINAL_PATH,
    INDEX_PART1_PATH,
    INDEX_PATH,
    LOCAL_W_PATH,
    TRAINED_INDEX_PATH,
)

torch.backends.cudnn.benchmark = True

try:
    user_secrets = UserSecretsClient()  # pyright: ignore
    secret_value_0 = user_secrets.get_secret("WANDB_API_KEY")
    wandb.login(key=secret_value_0)
except Exception as exc:  # noqa: BLE001
    print(exc)


def _download_model():
    from kaggle_secrets import UserSecretsClient  # pyright: ignore

    user_secrets = UserSecretsClient()
    hf_gemma_token = user_secrets.get_secret("hf_gemma_3_4b_token")
    huggingface_hub.login(hf_gemma_token, skip_if_logged_in=True)
    model_id = "google/gemma-3-4b-it"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        model_kwargs={"quantization_config": quantization_config},
        device_map="auto",
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
                },
                {"type": "text", "text": "What animal is on the candy?"},
            ],
        },
    ]

    return pipe(text=messages)  # pyright: ignore


def id_document_df(question=False):
    def map_partition(p):
        if question:
            p = p[
                p["document"].apply(lambda x: isinstance(x, dict))
                & p["question"].apply(lambda x: isinstance(x, dict))
            ]

        else:
            p = p[p["document"].apply(lambda x: isinstance(x, dict))]

        if p.empty:
            return p

        def map_tokens(doc: dict):
            doc = doc["tokens"]
            return [
                doc["token"][i]
                for i in range(len(doc["is_html"]))
                if not doc["is_html"][i]
            ]

        p["document"] = p["document"].apply(map_tokens)
        p["id"] = p["id"].apply(int)

        if question:
            p["question"] = p["question"].apply(lambda d: d["tokens"].tolist())

        return p

    df = ddf.read_parquet(f"{DATASET_PATH}/nq-dataset/nq_dataset/*")
    cols = ["id", "document"]

    if question:
        cols.append("question")

    df = df[cols]
    df = df.map_partitions(map_partition)

    return df


class PartitionDataset(Dataset):
    def __init__(self, pdf, col):
        self.pdf = pdf
        self.col = col
        self.pdf = self.pdf[self.pdf[col].apply(lambda x: isinstance(x, list))]
        # remove URLS
        self.pdf[col] = self.pdf[col].map(
            lambda doc: (
                [t for t in doc if not t.startswith(("https://", "http://"))]
                if isinstance(doc, list)
                else doc
            )
        )

    def __len__(self):
        return len(self.pdf)

    def __getitem__(self, idx):
        doc = self.pdf.iloc[idx]
        return doc["id"], doc[self.col]


def collate_fn(batch, tokenizer):
    max_len = 512
    stride = int(max_len * 0.35)
    doc_ids, doc_batch = zip(*batch)
    doc_ids, doc_batch = list(doc_ids), list(doc_batch)
    inputs = tokenizer(
        doc_batch,
        is_split_into_words=True,
        max_length=max_len,
        stride=stride,
        padding="max_length",
        truncation=True,
        return_overflowing_tokens=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    return doc_ids, inputs


def partition_to_embeddings(p, col, tokenizer, model, training=False):
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mean pooling
    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def to_embeddings(p, training=False):
        partition_embeddings = []
        partition_doc_ids = []

        ds = PartitionDataset(p, col)
        coll_fn = lambda batch: collate_fn(batch, tokenizer)
        dl = DataLoader(
            ds,
            32,
            num_workers=4,
            collate_fn=coll_fn,
            pin_memory=True,
            persistent_workers=True,
        )

        t_p = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):  # pyright: ignore
            for i, (doc_ids, inputs) in enumerate(dl):
                id_mapping = inputs.pop("overflow_to_sample_mapping")
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

                outputs = model(**inputs)
                embeddings = mean_pooling(outputs[0], inputs["attention_mask"])

                del outputs

                partition_embeddings.append(embeddings.float().cpu())

                if not training:
                    partition_doc_ids.extend([doc_ids[i] for i in id_mapping])

        partition_embeddings = torch.cat(partition_embeddings).numpy()

        print(f"p delta: {time.perf_counter() - t_p}")

        if training:
            return partition_embeddings

        assert len(partition_embeddings) == len(partition_doc_ids)

        return partition_embeddings, partition_doc_ids

    return to_embeddings(p, training)


def _build_index(training, first_part=True):
    import faiss

    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained("facebook/contriever")

    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)
    model.eval()

    D = 512
    M = D // 4

    idx_factory_string = f"OPQ{M}_{D},IVF65536_HNSW32,PQ{M}x8"

    if training:
        try:
            os.remove(f"{LOCAL_W_PATH}/nq_indexes/trained.index")

        except FileNotFoundError:
            pass

        index = faiss.index_factory(768, idx_factory_string, faiss.METRIC_INNER_PRODUCT)
        index_ivf = faiss.extract_index_ivf(index)
        clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
        index_ivf.clustering_index = clustering_index

    else:
        if first_part:
            index = faiss.read_index(
                f"{DATASET_PATH}/{TRAINED_INDEX_PATH}/nq_indexes/trained.index"
            )
        else:
            index = faiss.read_index(f"{DATASET_PATH}/{INDEX_PART1_PATH}/{INDEX_PATH}")

    df = id_document_df()

    if training:
        wandb.init(
            project="rag_agent",
            group="training_1",
            config={
                "n_partitions": 110,
                "index": idx_factory_string,
            },
        )

        parts = df.partitions[:110].to_delayed()

        parts = [
            dask.delayed(partition_to_embeddings)(  # pyright: ignore
                p,
                "document",
                tokenizer,
                model,
                training,
            )
            for p in parts
        ]
        parts = dask.compute(*parts, scheduler="single-threaded")  # pyright: ignore
        parts = np.concat(parts)
        faiss.normalize_L2(parts)
        index.train(parts)
        os.makedirs(f"{LOCAL_W_PATH}/nq_indexes", exist_ok=True)
        faiss.write_index(index, f"{LOCAL_W_PATH}/nq_indexes/trained.index")

        print("Done")

    else:
        metadata = None
        offset = 0
        if first_part:
            try:
                os.remove(f"{LOCAL_W_PATH}/nq_indexes/metadata.parquet")

            except FileNotFoundError:
                pass

            metadata = pd.DataFrame(
                {
                    "id": pd.Series(dtype=np.int64),
                    "doc_id": pd.Series(dtype=np.int64),
                }
            )

        else:
            metadata = pd.read_parquet(
                f"{DATASET_PATH}/{INDEX_PART1_PATH}/nq_indexes/metadata.parquet"
            )
            offset = len(metadata)

        start, end = 0 if first_part else 75, 75 if first_part else None
        embeddings = df.partitions[start:end].to_delayed()
        embeddings = [
            dask.delayed(partition_to_embeddings)(  # pyright: ignore
                p,
                "document",
                tokenizer,
                model,
                training,
            )
            for p in embeddings
        ]
        embeddings = dask.compute(*embeddings, scheduler="single-threaded")  # pyright: ignore
        embeddings, doc_ids = [x[0] for x in embeddings], [x[1] for x in embeddings]
        embeddings = np.concat(embeddings)
        doc_ids = np.concat(doc_ids)

        ids = np.arange(offset, offset + len(embeddings), dtype=np.int64)

        faiss.normalize_L2(embeddings)
        index.add_with_ids(embeddings, ids)
        os.makedirs(f"{LOCAL_W_PATH}/nq_indexes", exist_ok=True)

        faiss.write_index(index, f"{LOCAL_W_PATH}/{INDEX_PATH}")

        metadata = pd.concat(
            [metadata, pd.DataFrame({"id": ids, "doc_id": doc_ids})], ignore_index=True
        )
        metadata.to_parquet(f"{LOCAL_W_PATH}/nq_indexes/metadata.parquet", index=False)

        print("Done")


def recall(k=5):
    import time

    import faiss

    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained("facebook/contriever")

    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)
    model.eval()

    df = id_document_df(question=True)
    embeddings = df.partitions[:75].to_delayed()
    embeddings = [
        dask.delayed(partition_to_embeddings)(p, "document", tokenizer, model, False)  # pyright: ignore
        for p in embeddings
    ]
    embeddings = dask.compute(*embeddings, scheduler="single-threaded")  # pyright: ignore

    embeddings, doc_ids = [x[0] for x in embeddings], [x[1] for x in embeddings]
    embeddings = np.concat(embeddings)
    doc_ids = np.concat(doc_ids)

    offset = 0
    flat_index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
    ids = np.arange(offset, offset + len(embeddings), dtype=np.int64)

    faiss.normalize_L2(embeddings)
    flat_index.add_with_ids(embeddings, ids)  # pyright: ignore

    # Queries
    queries = df[["id", "question"]].partitions[:75].to_delayed()
    queries = [
        dask.delayed(partition_to_embeddings)(p, "question", tokenizer, model, False)  # pyright: ignore
        for p in queries
    ]
    queries = dask.compute(*queries, scheduler="single-threaded")  # pyright: ignore

    queries = [x[0] for x in queries]
    queries = np.concat(queries)
    faiss.normalize_L2(queries)

    # TODO: MAKE FLAT INDEX DATASET

    pq_index = faiss.read_index(f"{DATASET_PATH}/{INDEX_PART1_PATH}/{INDEX_PATH}")

    recalls = []
    index_ivf = faiss.extract_index_ivf(pq_index)
    for nprobe in [1, 4, 16, 32, 64, 128, 256, 512, 1024, 4096, index_ivf.nlist]:
        index_ivf.nprobe = nprobe

        _flat_index_D, flat_index_I = flat_index.search(queries, k)  # pyright: ignore

        t_q = time.perf_counter()
        _pq_index_D, pq_index_I = pq_index.search(queries, k)
        t_q = f"pq_index q delta: {time.perf_counter() - t_q}"

        # truth table -> recall per query -> avg recall across all queries
        # Note that this is chunk level recall
        recalls.append(
            (
                nprobe,
                (pq_index_I[:, :, None] == flat_index_I[:, None, :])
                .any(axis=2)
                .mean(axis=1)
                .mean(),
                t_q,
            )
        )

    return recalls

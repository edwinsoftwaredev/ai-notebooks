import os

os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import dask
import dask.dataframe as ddf
import faiss
import huggingface_hub
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, pipeline


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


def _download_dataset():
    import shutil

    from rag_agent import nq_schema

    def update_document(d):
        if "html" in d:
            del d["html"]

        if "tokens" in d:
            if "end_byte" in d["tokens"]:
                del d["tokens"]["end_byte"]

            if "start_byte" in d["tokens"]:
                del d["tokens"]["start_byte"]

        return d

    def update_annotations(d):
        if "id" in d:
            del d["id"]

        if "long_answer" in d:
            for dd in d["long_answer"]:
                if "candidate_index" in dd:
                    del dd["candidate_index"]

                if "start_byte" in dd:
                    del dd["start_byte"]

                if "end_byte" in dd:
                    del dd["end_byte"]

        if "short_answers" in d:
            del d["short_answers"]

        if "yes_no_answer" in d:
            del d["yes_no_answer"]

        return d

    def update_question(d):
        if "text" in d:
            del d["text"]

        return d

    def update_partitions(df):
        df = df.drop(columns=["long_answer_candidates"])
        df["question"] = df["question"].map(update_question)
        df["document"] = df["document"].map(update_document)
        df["annotations"] = df["annotations"].map(update_annotations)
        return df

    n_parquets = int(287 * 0.5)
    repo_id = "google-research-datasets/natural_questions"
    for i in range(0, n_parquets, 25):
        files = [
            f"default/train-{(i + j):05d}-of-00287.parquet"
            for j in range(25)
            if (i + j) <= n_parquets
        ]

        parquets = snapshot_download(
            repo_id=repo_id,
            allow_patterns=files,
            repo_type="dataset",
        )

        df = ddf.read_parquet(f"{parquets}/**/*.parquet")
        df = df.map_partitions(update_partitions)
        df.to_parquet(
            "/kaggle/working/nq_dataset",
            engine="pyarrow",
            schema=nq_schema.schema,
            append=True,
            write_index=False,
        )

        shutil.rmtree(
            "/kaggle/working/hf_cache/hub/datasets--google-research-datasets--natural_questions",
            ignore_errors=True,
        )

        # SAVE NOTEBOOK AS DATASET AND USE AS NOTEBOOK INPUT LATER


def id_document_df(question=False):
    def map_partition(p):
        p = p[
            p["document"].apply(lambda x: isinstance(x, dict))
            & p["question"].apply(lambda x: isinstance(x, dict))
        ]

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

    df = ddf.read_parquet(
        "/kaggle/input/datasets/edwinsoftwaredev/rag-agent-trained-index/nq_dataset/*"
    )
    cols = ["id", "document"]

    if question:
        cols.append("question")

    df = df[cols]
    df = df.map_partitions(map_partition)

    return df


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

        def batch_to_embeddings(doc_batch):
            max_len = 512
            stride = int(max_len * 0.35)

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

            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            embeddings = mean_pooling(outputs[0], inputs["attention_mask"])

            id_mapping = inputs["overflow_to_sample_mapping"]

            del outputs
            del inputs

            return embeddings, id_mapping

        def col_selector(doc):
            if col == "question":
                return doc.question

            return doc.document

        # remove URLS
        p[col] = p[col].map(
            lambda doc: (
                [t for t in doc if not t.startswith(("https://", "http://"))]
                if isinstance(doc, list)
                else doc
            )
        )

        t_p = time.perf_counter()
        with torch.inference_mode():
            doc_ids = []
            doc_batch = []
            for doc in p.itertuples():
                if not isinstance(col_selector(doc), list):
                    continue

                doc_id = doc.id
                doc = col_selector(doc)

                doc_batch.append(doc)
                doc_ids.append(doc_id)

                if len(doc_batch) == 16:
                    embeddings, id_map = batch_to_embeddings(doc_batch)
                    partition_embeddings.append(embeddings.cpu())

                    if not training:
                        partition_doc_ids.extend([doc_ids[i] for i in id_map])

                    doc_ids = []
                    doc_batch = []

            if len(doc_batch):
                embeddings, id_map = batch_to_embeddings(doc_batch)
                partition_embeddings.append(embeddings.cpu())

                if not training:
                    partition_doc_ids.extend([doc_ids[i] for i in id_map])

        partition_embeddings = torch.cat(partition_embeddings).numpy()

        print(f"p delta: {time.perf_counter() - t_p}")

        if training:
            return partition_embeddings

        assert len(partition_embeddings) == len(partition_doc_ids)

        return partition_embeddings, partition_doc_ids

    return to_embeddings(p, training)


def _build_index(training, p_start):
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained("facebook/contriever")

    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)
    model.eval()

    D = 256
    M = D // 4

    if training:
        try:
            os.remove("/kaggle/working/nq_indexes/trained.index")

        except FileNotFoundError:
            pass

        index = faiss.index_factory(
            768, f"OPQ{M}_{D},IVF65536_HNSW32,PQ{M}", faiss.METRIC_INNER_PRODUCT
        )
        index_ivf = faiss.extract_index_ivf(index)
        clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
        index_ivf.clustering_index = clustering_index

    else:
        index = faiss.read_index(
            "/kaggle/input/datasets/edwinsoftwaredev/rag-agent-trained-index/nq_indexes/trained.index"
        )

    df = id_document_df()

    if training:
        parts = df.partitions[:70].to_delayed()

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
        os.makedirs("/kaggle/working/nq_indexes", exist_ok=True)
        faiss.write_index(index, "/kaggle/working/nq_indexes/trained.index")

        print("Done")

    else:
        metadata = None
        offset = 0
        if p_start == 0:
            try:
                os.remove("/kaggle/working/nq_indexes/metadata.parquet")

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
                "/kaggle/input/datasets/edwinsoftwaredev/rag-agent-trained-index/nq_indexes/metadata.parquet"
            )
            offset = len(metadata)

        embeddings = df.partitions[p_start : p_start + 5].to_delayed()
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
        faiss.write_index(index, "/kaggle/working/nq_indexes/trained.index")

        metadata = pd.concat(
            [metadata, pd.DataFrame({"id": ids, "doc_id": doc_ids})], ignore_index=True
        )
        metadata.to_parquet("/kaggle/working/nq_indexes/metadata.parquet", index=False)

        print("Done")


def recall(k=5):
    import time

    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained("facebook/contriever")

    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)
    model.eval()

    df = id_document_df(question=True)
    embeddings = df.partitions[:5].to_delayed()
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
    queries = df[["id", "question"]].partitions[:5].to_delayed()
    queries = [
        dask.delayed(partition_to_embeddings)(p, "question", tokenizer, model, False)  # pyright: ignore
        for p in queries
    ]
    queries = dask.compute(*queries, scheduler="single-threaded")  # pyright: ignore

    queries = [x[0] for x in queries]
    queries = np.concat(queries)
    faiss.normalize_L2(queries)

    # TODO: Remove this
    pq_index = faiss.read_index("/kaggle/working/nq_indexes/trained.index")

    # pq_index = faiss.read_index(
    #     "/kaggle/input/datasets/edwinsoftwaredev/rag-agent-trained-index/nq_indexes/trained.index"
    # )
    #

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

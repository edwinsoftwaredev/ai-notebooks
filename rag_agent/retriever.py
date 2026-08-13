import time

import faiss
import numpy as np

from rag_agent.embedder import queries_to_embeddings


def retrieve(queries, tokenizer, embedder, index, metadata, df, k=40):
    index_ivf = faiss.extract_index_ivf(index)
    nprobe = int(index_ivf.nlist * 0.35)
    index_ivf.nprobe = nprobe

    queries = queries_to_embeddings(queries, tokenizer, embedder)
    result = {}

    faiss.normalize_L2(queries)
    _pq_index_D, pq_index_I = index.search(queries, k)

    result["queries"] = queries
    doc_ids = np.unique(pq_index_I)
    doc_ids = (
        metadata[metadata["id"].isin(doc_ids)]["doc_id"].astype(str).unique().tolist()  # pyright: ignore
    )

    docs = df.loc[doc_ids].compute()  # NOTE: random access on a randomly sorted dataset

    docs = [
        [
            doc["tokens"]["token"][i]
            for i in range(len(doc["tokens"]["is_html"]))
            if not (
                doc["tokens"]["is_html"][i]
                or doc["tokens"]["token"][i].startswith(("https://", "http://"))
            )
        ]
        for doc in docs["document"].tolist()
    ]

    result["docs"] = docs

    return result

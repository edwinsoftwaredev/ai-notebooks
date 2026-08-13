import faiss

from rag_agent.embedder import docs_to_embeddings


def from_flat_index(input, tokenizer, embedder, k=10):
    queries = input["queries"]
    docs = input["docs"]
    docs, ids = docs_to_embeddings(docs, tokenizer, embedder)

    flat_index = faiss.IndexFlatIP(768)
    faiss.normalize_L2(docs)
    flat_index.add(docs)  # pyright: ignore

    _, index_ids = flat_index.search(queries, k)  # pyright: ignore
    index_ids = index_ids.ravel()

    ids = [ids[id] for id in index_ids]

    passages = tokenizer.batch_decode(ids, skip_special_tokens=True)

    del docs
    del input
    del ids
    del queries
    del index_ids
    flat_index.reset()
    del flat_index

    return passages

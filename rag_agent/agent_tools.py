from rag_agent import reranker, retriever


def wikipedia_articles_retriever_func(tokenizer, embedder, index, index_metadata, df):
    def wikipedia_articles_retriever(queries: list[str]):
        """
        Retrieve passages from Wikipedia articles relevant to the provided queries.

        Phrase queries as search terms or statements rather than questions.
        For better retrieval results provide multiple queries.


        Args:
            queries: a list of queries

        Returns:
            Retrieved passages enclosed in <documents> tags. Each passage is
            enclosed in <doc> tags with a numeric id attribute.

        """

        input_docs = retriever.retrieve(
            queries, tokenizer, embedder, index, index_metadata, df, 5
        )

        passages = reranker.from_flat_index(input_docs, tokenizer, embedder, 10)

        passages = "\n\n".join(
            [f"<doc id='{i + 1}'>\n{p}\n</doc>" for i, p in enumerate(passages)]
        )

        # user_content = f"""Context Documents:
        #     <documents>
        #     {passages}
        #     </documents>

        # User Query: {user_query}"""

        passages = f"""<documents>
            {passages}
        </documents>"""

        return passages

    return wikipedia_articles_retriever

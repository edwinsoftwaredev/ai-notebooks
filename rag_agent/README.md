# Agentic RAG

This project implements an agentic RAG system where the language model controls the retrieval process. The agent decides when and how to invoke tools, whether additional information needs to be retrieved, determine when sufficient information has been gathered, and request clarification from the user when necessary.

## Components

### Index

* **Embedding Model:** [Contriever](https://huggingface.co/facebook/contriever)
* **Similarity Search Index:** [FAISS](https://faiss.ai/)

### LLM

* **Model:** [Gemma 4 12B Instruction-Tuned](https://www.kaggle.com/models/google/gemma-4/Transformers/gemma-4-12b-it/2)

## Code

[GitHub Repository](https://github.com/edwinsoftwaredev/ai-notebooks/tree/main/rag_agent)

## References

* Jurafsky, D., & Martin, J. H. *Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition with Language Models* (3rd Edition Draft).
  https://web.stanford.edu/~jurafsky/slp3/

## Online Resources

* [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki/)
* [FAISS: The Missing Manual](https://www.pinecone.io/learn/series/faiss/)
* [Retrieval-Augmented Generation (RAG) on Anyscale](https://docs.anyscale.com/rag)

## Datasets

This project uses the following publicly available dataset:

### Natural Questions (NQ)

* **Source:** [Natural Questions](https://huggingface.co/datasets/google-research-datasets/natural_questions)
* **Description:** Natural Questions (NQ) is a question-answering dataset consisting of questions from real users. Answering these questions requires systems to read and understand entire Wikipedia articles, which may or may not contain the answer.
* **License:** CC BY-SA 3.0

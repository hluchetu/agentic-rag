from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class VectorStoreRetriever:
    def __init__(self, vector_store: VectorStore, top_k: int = 5) -> None:
        self._retriever = vector_store.as_retriever(
            search_kwargs={"k": top_k},
        )

    def invoke(self, query: str) -> list[Document]:
        return self._retriever.invoke(query)

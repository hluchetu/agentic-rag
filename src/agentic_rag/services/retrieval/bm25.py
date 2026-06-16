from __future__ import annotations

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


class BM25DocumentRetriever:
    def __init__(self, documents: list[Document], top_k: int = 5) -> None:
        self._retriever = BM25Retriever.from_documents(documents)
        self._retriever.k = top_k

    def invoke(self, query: str, top_k: int) -> list[Document]:
        self._retriever.k = top_k
        return self._retriever.invoke(query)

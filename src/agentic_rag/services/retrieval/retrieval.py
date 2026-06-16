from __future__ import annotations

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


class LangChainRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self._retriever = BM25Retriever.from_documents(documents)

    def invoke(self, query: str, top_k: int) -> list[Document]:
        self._retriever.k = top_k
        return self._retriever.invoke(query)

from __future__ import annotations

from langchain_core.documents import Document

from agentic_rag.services.retrieval.interface import Retriever
from agentic_rag.state import AgentState


class RetrieveManyDocumentsNode:
    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    def __call__(self, state: AgentState) -> dict[str, list[Document]]:
        documents_by_key: dict[str, Document] = {}

        for query in state["retrieval_queries"]:
            documents = self._retriever.invoke(query)

            for document in documents:
                key = self._document_key(document)
                documents_by_key[key] = document

        return {
            "documents": list(documents_by_key.values()),
        }

    def _document_key(self, document: Document) -> str:
        source = document.metadata.get("source")
        document_id = document.metadata.get("id")

        if source is not None and document_id is not None:
            return f"{source}:{document_id}"

        if source is not None:
            return f"{source}:{document.page_content}"

        return document.page_content

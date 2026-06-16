from __future__ import annotations

from langchain_core.documents import Document


from agentic_rag.services.retrieval.interface import Retriever
from agentic_rag.state import AgentState


class RetriveDocumentNode:
    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    def __call__(self, state: AgentState) -> dict[str, list[Document]]:
        documents = self._retriever.invoke(state["active_query"])
        return {"documents": documents}

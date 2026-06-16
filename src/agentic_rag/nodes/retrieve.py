from __future__ import annotations


from agentic_rag.services.retrieval.interface import Retriever
from agentic_rag.state import AgentState


class RetriveDocumentNode:
    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    def __call__(self, state: AgentState) -> AgentState:
        documents = self._retriever.invoke(state["question"])
        return {**state, "documents": documents}

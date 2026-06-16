from __future__ import annotations
from typing import Protocol


from langchain_core.documents import Document
from agentic_rag.domain.models import RetrievalDecision


class RetrievalGrader(Protocol):
    def grade(self, question: str, documents: list[Document]) -> RetrievalDecision: ...

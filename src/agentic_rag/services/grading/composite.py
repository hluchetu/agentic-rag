from __future__ import annotations

from langchain_core.documents import Document

from agentic_rag.domain.models import RetrievalDecision
from agentic_rag.services.grading.interface import RetrievalGrader


class CompositeRetrievalGrader:
    def __init__(self, graders: list[RetrievalGrader]) -> None:
        self._graders = graders

    def grade(self, question: str, documents: list[Document]) -> RetrievalDecision:
        for grader in self._graders:
            decision = grader.grade(question, documents)
            if decision.grade == "bad":
                return decision
        return RetrievalDecision(
            grade="good",
            reason="All graders passed, documents are relevant and non-empty",
        )

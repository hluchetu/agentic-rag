from __future__ import annotations

from agentic_rag.domain.models import RetrievalDecision
from agentic_rag.services.grading.interface import RetrievalGrader
from agentic_rag.state import AgentState


class GradeRetrievalNode:
    def __init__(self, grader: RetrievalGrader) -> None:
        self._grader = grader

    def __call__(self, state: AgentState) -> dict[str, RetrievalDecision]:
        decision = self._grader.grade(
            question=state["question"],
            documents=state["documents"],
        )

        return {"retrieval_decision": decision}

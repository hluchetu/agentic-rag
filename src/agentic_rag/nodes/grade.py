from __future__ import annotations

from agentic_rag.services.grading.interface import RetrievalGrader
from agentic_rag.state import AgentState


class GradeRetrievalNode:
    def __init__(self, grader: RetrievalGrader) -> None:
        self._grader = grader

    def __call__(self, state: AgentState) -> AgentState:
        decision = self._grader.grade(
            state["question"],
            state["documents"],
        )
        return {"retrieval_decision": decision}

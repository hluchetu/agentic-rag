from __future__ import annotations

from typing import Literal

from agentic_rag.state import AgentState


RouteAfterGrading = Literal["generate", "rewrite"]


def route_after_grading(state: AgentState) -> RouteAfterGrading:
    decision = state["retrieval_decision"]

    if decision is None:
        raise ValueError("Cannot route because retrieval_decision is missing.")

    if decision.grade == "good":
        return "generate"

    return "rewrite"

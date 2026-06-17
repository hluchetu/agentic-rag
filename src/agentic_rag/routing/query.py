from __future__ import annotations

from typing import Literal

from agentic_rag.state import AgentState
from agentic_rag.errors import MissingStateError


RouteAfterQueryClassification = Literal["standard", "multi_query"]


def route_after_query_classification(
    state: AgentState,
) -> RouteAfterQueryClassification:
    classification = state["query_classification"]

    if classification is None:
        raise MissingStateError("Cannot route because query_classification is missing.")

    return classification.route

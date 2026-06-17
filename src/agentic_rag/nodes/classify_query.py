from __future__ import annotations

from agentic_rag.domain.models import QueryClassification
from agentic_rag.services.classifier.query.interface import ClassifyQuery
from agentic_rag.state import AgentState


class ClassifyQueryNode:
    def __init__(self, classifier: ClassifyQuery) -> None:
        self._classifier = classifier

    def __call__(self, state: AgentState) -> dict[str, QueryClassification]:
        query_classification = self._classifier.classify(state["question"])

        return {
            "query_classification": query_classification,
        }

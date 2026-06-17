from __future__ import annotations

from typing import Protocol

from agentic_rag.domain.models import QueryClassification


class ClassifyQuery(Protocol):
    def classify(self, question: str) -> QueryClassification: ...

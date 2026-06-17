from __future__ import annotations

from agentic_rag.domain.models import GeneratedAnswer
from agentic_rag.services.generation.interface import AnswerGenerator
from agentic_rag.state import AgentState


class GenerateAnswerNode:
    def __init__(self, generator: AnswerGenerator) -> None:
        self._generator = generator

    def __call__(self, state: AgentState) -> dict[str, GeneratedAnswer]:
        answer = self._generator.generate(
            question=state["question"],
            documents=state["documents"],
        )

        return {
            "answer": answer,
        }

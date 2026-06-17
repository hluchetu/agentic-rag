from __future__ import annotations

from typing import Protocol

from langchain_core.documents import Document

from agentic_rag.domain.models import GeneratedAnswer


class AnswerGenerator(Protocol):
    def generate(
        self,
        question: str,
        documents: list[Document],
    ) -> GeneratedAnswer:
        ...

from __future__ import annotations

from langchain_core.documents import Document

from agentic_rag.domain.models import RetrievalDecision


class RuleBasedRetrievalGrader:
    def grade(self, question: str, documents: list[Document]) -> RetrievalDecision:
        if not question.strip():
            return RetrievalDecision(grade="bad", reason="Empty question")
        if not documents:
            return RetrievalDecision(grade="bad", reason="No documents retrieved")

        has_non_empty_document = any(
            document.page_content.strip() for document in documents
        )
        if not has_non_empty_document:
            return RetrievalDecision(grade="bad", reason="Documents are empty")

        return RetrievalDecision(
            grade="good", reason="Documents are relevant and non-empty"
        )

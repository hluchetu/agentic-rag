from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from agentic_rag.domain.models import RetrievalDecision


class LLMBasedRetrievalGrader:
    def __init__(self, model: BaseChatModel) -> None:
        self._model = model

    def grade(self, question: str, documents: list[Document]) -> RetrievalDecision:
        context = "\n\n".join(document.page_content for document in documents)
        result = self._model.invoke(
            f"""
 You are grading whether retrieved documents are relevant to a user question.

Question:
{question}

Retrieved documents:
{context}

Return:
- grade="good" if the documents contain enough relevant information to answer the question.
- grade="bad" if the documents are irrelevant, empty, or insufficient.
"""
        )

        return result

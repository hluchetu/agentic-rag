from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document

from agentic_rag.domain.models import GeneratedAnswer
from agentic_rag.domain.models import QueryRewrite
from agentic_rag.domain.models import RetrievalDecision


class AgentState(TypedDict):
    question: str
    active_query: str
    documents: list[Document]
    retrieval_decision: RetrievalDecision | None
    query_rewrite: QueryRewrite | None
    answer: GeneratedAnswer | None

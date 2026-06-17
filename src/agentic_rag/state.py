from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document

from agentic_rag.domain.models import (
    GeneratedAnswer,
    MultiQueryRewrite,
    QueryClassification,
    QueryRewrite,
    RetrievalDecision,
)


class AgentState(TypedDict):
    question: str
    active_query: str
    retrieval_queries: list[str]
    documents: list[Document]
    query_classification: QueryClassification | None
    retrieval_decision: RetrievalDecision | None
    query_rewrite: QueryRewrite | None
    multi_query_rewrite: MultiQueryRewrite | None
    answer: GeneratedAnswer | None

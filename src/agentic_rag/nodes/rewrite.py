from __future__ import annotations

from agentic_rag.domain.models import QueryRewrite
from agentic_rag.services.rewriting.interface import QueryRewriter
from agentic_rag.state import AgentState


class RewriteQueryNode:
    def __init__(self, rewriter: QueryRewriter) -> None:
        self._rewriter = rewriter

    def __call__(self, state: AgentState) -> dict[str, QueryRewrite]:
        query_rewrite = self._rewriter.rewrite(state["question"])

        return {
            "query_rewrite": query_rewrite,
            "active_query": query_rewrite.rewritten_query,
        }

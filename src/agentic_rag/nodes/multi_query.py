from __future__ import annotations

from typing import Any

from agentic_rag.services.transformation.query.interface import MultiQueryRewriter
from agentic_rag.state import AgentState


class MultiQueryNode:
    def __init__(self, rewriter: MultiQueryRewriter) -> None:
        self._rewriter = rewriter

    def __call__(self, state: AgentState) -> dict[str, Any]:
        multi_query_rewrite = self._rewriter.rewrite(state["question"])

        return {
            "multi_query_rewrite": multi_query_rewrite,
            "retrieval_queries": multi_query_rewrite.queries,
        }

from __future__ import annotations

from typing import Protocol

from agentic_rag.domain.models import MultiQueryRewrite
from agentic_rag.domain.models import QueryRewrite


class QueryRewriter(Protocol):
    def rewrite(self, question: str) -> QueryRewrite: ...


class MultiQueryRewriter(Protocol):
    def rewrite(self, question: str) -> MultiQueryRewrite: ...

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


RetrievalGrade = Literal["good", "bad"]
QueryRoute = Literal["standard", "multi_query"]


class RetrievalDecision(BaseModel):
    grade: RetrievalGrade
    reason: str


class GeneratedAnswer(BaseModel):
    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)


class QueryClassification(BaseModel):
    route: QueryRoute
    reason: str


class QueryRewrite(BaseModel):
    rewritten_query: str
    reason: str


class MultiQueryRewrite(BaseModel):
    queries: list[str]
    reason: str

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


RetrievalGrade = Literal["good", "bad"]


class RetrievalDecision(BaseModel):
    grade: RetrievalGrade
    reason: str


class QueryRewrite(BaseModel):
    rewritten_query: str
    reason: str


class GeneratedAnswer(BaseModel):
    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)

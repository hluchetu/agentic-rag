from __future__ import annotations

from typing import Protocol
from langchain_core.documents import Document


class Retriever(Protocol):
    def invoke(self, query: str, top_k: int) -> list[Document]: ...

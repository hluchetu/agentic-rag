from __future__ import annotations

from typing import Literal

from langchain_core.documents import Document

from agentic_rag.ingestion.chunking import chunk_documents
from agentic_rag.ingestion.loaders import load_pdf
from agentic_rag.ingestion.loaders import load_web_page


SourceType = Literal["pdf", "web"]


def load_and_chunk_source(
    source: str,
    source_type: SourceType,
    chunk_size: int = 1_000,
    chunk_overlap: int = 200,
) -> list[Document]:
    if source_type == "pdf":
        documents = load_pdf(source)
    elif source_type == "web":
        documents = load_web_page(source)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

    return chunk_documents(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

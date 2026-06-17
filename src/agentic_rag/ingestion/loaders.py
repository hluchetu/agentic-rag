from __future__ import annotations

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document


def load_pdf(path: str) -> list[Document]:
    loader = PyPDFLoader(path)

    return loader.load()


def load_web_page(url: str) -> list[Document]:
    loader = WebBaseLoader(url)

    return loader.load()

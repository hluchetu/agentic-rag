# Recruiter Notes

This project demonstrates practical AI engineering work around RAG, LangGraph, and production-style code organization.

## What This Shows

- Graph-based agentic RAG orchestration
- Typed graph state
- Conditional routing
- Dependency injection
- Clean separation between nodes and services
- LangChain document loaders and retrievers
- Pydantic domain models and settings
- Structured LLM outputs
- Query classification and multi-query retrieval
- Corrective retrieval through grading and rewriting
- DeepSeek integration through an OpenAI-compatible LangChain model

## Engineering Emphasis

The graph is intentionally composed from interfaces:

```text
Retriever
Query classifier
Query rewriter
Multi-query rewriter
Retrieval grader
Answer generator
```

That means the workflow can replace BM25 with vector search, DeepSeek with another chat model, or rule-based grading with a reranker without rewriting graph orchestration.

## Current Workflow

```text
load source
chunk documents
build retriever
classify query
retrieve documents
grade retrieval
rewrite if weak
generate grounded answer
```

The project is scoped to a working agentic RAG baseline, with clear extension points for vector retrieval, reranking, web fallback, guardrails, and offline quality measurement.

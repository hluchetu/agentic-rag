# Retrieval Strategy Notes

The current retrieval flow supports standard retrieval and multi-query retrieval.

## Standard Retrieval

Standard retrieval uses one active query:

```text
question -> active_query -> retriever -> documents
```

This is the default path for clear, specific questions.

## Multi-Query Retrieval

Multi-query retrieval expands one question into several retrieval queries:

```text
question
  -> query classifier chooses multi_query
  -> LLM generates multiple retrieval queries
  -> retriever runs once per query
  -> documents are deduplicated
```

This helps when user wording may not match the document wording.

Example:

```text
Question:
How does LangGraph save state?

Generated retrieval queries:
- How does LangGraph persist graph state?
- What is checkpointing in LangGraph?
- How does LangGraph resume execution from saved state?
```

## Corrective Retrieval

After retrieval, the graph grades the documents.

If retrieval is weak:

```text
bad retrieval -> rewrite query -> retrieve again
```

This is the corrective RAG behavior.

## Why BM25 First

BM25 is used first because it is lightweight, transparent, and does not require embeddings or a vector database. It is a strong baseline for exact terms, document names, error codes, and technical vocabulary.

Future retrieval upgrades can add vector search, hybrid search, reranking, HyDE, question decomposition, or web fallback behind the same graph boundaries.

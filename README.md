# Agentic RAG

Production-oriented Retrieval-Augmented Generation workflow built with LangGraph, LangChain, typed state, query routing, corrective retrieval, and DeepSeek-backed generation.

The project models RAG as a graph instead of a fixed one-pass pipeline. It can classify a query, choose standard or multi-query retrieval, grade retrieved evidence, rewrite weak queries, retrieve again, and generate a grounded answer from the final context.

## What It Does

- Loads PDF or web sources with LangChain loaders
- Chunks documents with a recursive text splitter
- Retrieves evidence with BM25
- Classifies queries into standard or multi-query retrieval
- Generates multiple retrieval queries when useful
- Grades retrieved documents before generation
- Rewrites weak queries and retries retrieval
- Generates structured answers with citations
- Uses Pydantic Settings for environment configuration
- Keeps graph orchestration separate from retrieval, grading, transformation, and generation services

## Architecture

```text
source
  -> load documents
  -> chunk documents
  -> build retriever
  -> classify query
  -> standard retrieval OR multi-query retrieval
  -> grade retrieval
  -> generate answer OR rewrite query and retrieve again
```

Core package layout:

```text
src/agentic_rag/
├── cli.py
├── graph.py
├── settings.py
├── state.py
├── ingestion/
├── nodes/
├── prompts/
├── routing/
└── services/
```

More detail:

- [Architecture Notes](docs/architecture.md)
- [Retrieval Strategy Notes](docs/retrieval-strategies.md)
- [Project Overview](docs/project-overview.md)

## Setup

```bash
uv pip install -r requirements.txt
```

Create a `.env` file:

```text
DEEPSEEK_API_KEY=your-key
DEEPSEEK_MODEL=deepseek-v4-flash
```

`.env` is ignored by git.

## Run

PDF source:

```bash
PYTHONPATH=src python -m agentic_rag.cli \
  --source /path/to/file.pdf \
  --source-type pdf \
  --question "What is this document about?"
```

Web source:

```bash
PYTHONPATH=src python -m agentic_rag.cli \
  --source "https://example.com/article" \
  --source-type web \
  --question "What are the key ideas?"
```

## Current Scope

Implemented:

- LangGraph workflow
- DeepSeek model configuration through Pydantic Settings
- PDF and web ingestion
- Recursive document chunking
- BM25 retrieval
- Query classification
- Multi-query retrieval path
- Retrieval grading
- Corrective query rewriting
- Grounded answer generation

Not in scope yet:

- Vector database persistence
- Backend-native hybrid search
- Web search fallback
- HyDE
- Question decomposition
- Guardrails
- Offline evaluation suite

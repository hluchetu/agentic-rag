# Agentic RAG

Production-oriented agentic Retrieval-Augmented Generation workflows built with LangGraph.

The system focuses on retrieval control, correction, routing, and fallback behavior for cases where a fixed one-pass RAG pipeline is not enough.

## Objectives

- Implement Corrective RAG (CRAG)
- Implement Adaptive RAG query routing
- Implement multi-hop retrieval and synthesis
- Add web search fallback for failed local retrieval
- Add guardrails for prompt injection, PII handling, and access control
- Keep graph orchestration separate from retrieval, grading, rewriting, generation, and search services

## Core Architecture

Agentic RAG is modeled as a stateful graph:

```text
state -> node -> state update -> router -> next node
```

LangGraph provides the graph runtime:

- State carries the question, retrieved chunks, decisions, rewritten queries, and final answer
- Nodes perform bounded units of work
- Conditional edges route between generation, retry, rewrite, clarification, and fallback paths
- Services encapsulate retrievers, graders, rewriters, generators, and web search clients

## Package Structure

```text
src/agentic_rag/
├── __init__.py
├── state.py
├── errors.py
├── graph.py
├── domain/
│   ├── __init__.py
│   └── models.py
├── nodes/
│   ├── classify_query.py
│   ├── grade.py
│   ├── retrieve.py
│   └── rewrite.py
├── routing/
│   ├── query.py
│   └── retrieval.py
├── prompts/
│   ├── query_classification.yaml
│   ├── multi_query.yaml
│   └── rewriting.yaml
├── services/
│   ├── retrieval/
│   │   ├── interface.py
│   │   └── bm25.py
│   ├── grading/
│   │   └── interface.py
│   ├── classifier/
│   │   └── query/
│   │       ├── interface.py
│   │       └── llm_based.py
│   ├── transformation/
│   │   └── query/
│   │       ├── interface.py
│   │       ├── rewrite.py
│   │       └── multi_query.py
│   └── generation/
│       └── interface.py
└── guardrails/
    ├── injection.py
    ├── pii.py
    └── access_control.py
```

## Corrective RAG

CRAG adds retrieval quality control before generation.

```text
question
-> retrieve local context
-> grade retrieved context
-> if context is sufficient: generate
-> if context is weak: rewrite query
-> retrieve again
-> if still weak: use web search fallback
-> generate grounded answer
```

## Adaptive RAG

Adaptive RAG routes the query before retrieval.

```text
question
-> classify query
-> simple: answer directly
-> retrieval-needed: run CRAG
-> ambiguous: ask for clarification
-> external/current: use web search
```

## Multi-Hop RAG

Multi-hop RAG handles questions that require chained evidence.

```text
question
-> decompose into sub-questions
-> retrieve evidence for each sub-question
-> produce partial answers
-> synthesize final answer
```

## Engineering Principles

- Graph state is typed and explicit
- Nodes are small functions with clear state updates
- Services own external behavior and implementation details
- Graphs depend on interfaces, not concrete providers
- Retrieval quality is graded before answer generation
- Fallback paths are explicit and observable
- Local retrieval, web search, and generation remain independently testable

## Installation

```bash
uv pip install -r requirements.txt
```

## Run

```bash
python -m agentic_rag
```

## Roadmap

1. Define typed domain models and graph state
2. Implement CRAG graph
3. Add retrieval grading and query rewriting services
4. Add web search fallback
5. Add adaptive routing
6. Add multi-hop retrieval
7. Add guardrails
8. Add evaluation and tracing

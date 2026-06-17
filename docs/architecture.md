# Architecture Notes

This project separates orchestration from capabilities.

LangGraph owns the workflow:

```text
state -> node -> partial state update -> routing -> next node
```

Services own capabilities:

```text
retrieval
grading
query transformation
generation
ingestion
```

## State

`AgentState` is the graph's shared runtime data.

It stores:

- original question
- active retrieval query
- multi-query retrieval queries
- retrieved documents
- query classification decision
- retrieval grading decision
- query rewrite result
- multi-query rewrite result
- generated answer

The original question stays unchanged. `active_query` can change when corrective rewriting happens.

## Nodes

Nodes are thin LangGraph adapters around services.

Examples:

- `ClassifyQueryNode` calls a query classifier
- `RetrieveDocumentsNode` calls a retriever with `active_query`
- `RetrieveManyDocumentsNode` retrieves for multiple generated queries
- `GradeRetrievalNode` calls retrieval graders
- `RewriteQueryNode` updates `active_query`
- `GenerateAnswerNode` writes the final answer

Nodes return partial state updates. LangGraph merges those updates into the current state.

## Routing

Routing functions decide which node runs next.

Current routing:

```text
query classification:
  standard -> retrieve
  multi_query -> multi_query -> retrieve_many

retrieval grading:
  good -> generate_answer
  bad -> rewrite_query -> retrieve
```

The graph depends on interfaces, not concrete implementations. This keeps model providers, retrievers, graders, and query transformation strategies replaceable.

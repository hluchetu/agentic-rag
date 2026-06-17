from __future__ import annotations

from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph

from agentic_rag.nodes.classify_query import ClassifyQueryNode
from agentic_rag.nodes.generate import GenerateAnswerNode
from agentic_rag.nodes.grade import GradeRetrievalNode
from agentic_rag.nodes.multi_query import MultiQueryNode
from agentic_rag.nodes.retrieve import RetrieveDocumentsNode
from agentic_rag.nodes.retrieve_many import RetrieveManyDocumentsNode
from agentic_rag.nodes.rewrite import RewriteQueryNode
from agentic_rag.routing.query import route_after_query_classification
from agentic_rag.routing.retrieval import route_after_grading
from agentic_rag.services.classifier.query.interface import ClassifyQuery
from agentic_rag.services.generation.interface import AnswerGenerator
from agentic_rag.services.grading.interface import RetrievalGrader
from agentic_rag.services.retrieval.interface import Retriever
from agentic_rag.services.transformation.query.interface import MultiQueryRewriter
from agentic_rag.services.transformation.query.interface import QueryRewriter
from agentic_rag.state import AgentState


def build_graph(
    *,
    retriever: Retriever,
    query_classifier: ClassifyQuery,
    multi_query_rewriter: MultiQueryRewriter,
    retrieval_grader: RetrievalGrader,
    query_rewriter: QueryRewriter,
    answer_generator: AnswerGenerator,
):
    graph = StateGraph(AgentState)

    graph.add_node("classify_query", ClassifyQueryNode(query_classifier))
    graph.add_node("retrieve", RetrieveDocumentsNode(retriever))
    graph.add_node("multi_query", MultiQueryNode(multi_query_rewriter))
    graph.add_node("retrieve_many", RetrieveManyDocumentsNode(retriever))
    graph.add_node("grade_retrieval", GradeRetrievalNode(retrieval_grader))
    graph.add_node("rewrite_query", RewriteQueryNode(query_rewriter))
    graph.add_node("generate_answer", GenerateAnswerNode(answer_generator))

    graph.add_edge(START, "classify_query")
    graph.add_conditional_edges(
        "classify_query",
        route_after_query_classification,
        {
            "standard": "retrieve",
            "multi_query": "multi_query",
        },
    )
    graph.add_edge("multi_query", "retrieve_many")
    graph.add_edge("retrieve", "grade_retrieval")
    graph.add_edge("retrieve_many", "grade_retrieval")
    graph.add_conditional_edges(
        "grade_retrieval",
        route_after_grading,
        {
            "generate": "generate_answer",
            "rewrite": "rewrite_query",
        },
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate_answer", END)

    return graph.compile()

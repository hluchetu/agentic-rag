from __future__ import annotations

import argparse

from langchain_openai import ChatOpenAI

from agentic_rag.graph import build_graph
from agentic_rag.ingestion.pipeline import SourceType
from agentic_rag.ingestion.pipeline import load_and_chunk_source
from agentic_rag.settings import Settings
from agentic_rag.settings import get_settings
from agentic_rag.services.classifier.query.llm_based import LLMQueryClassifier
from agentic_rag.services.generation.llm_based import LLMAnswerGenerator
from agentic_rag.services.grading.composite import CompositeRetrievalGrader
from agentic_rag.services.grading.llm_based import LLMBasedRetrievalGrader
from agentic_rag.services.grading.rule_based import RuleBasedRetrievalGrader
from agentic_rag.services.retrieval.bm25 import BM25DocumentRetriever
from agentic_rag.services.transformation.query.multi_query import LLMMultiQueryRewriter
from agentic_rag.services.transformation.query.rewrite import LLMQueryRewriter


def main() -> None:
    settings = get_settings()
    args = parse_args()
    model = create_deepseek_model(
        model_name=args.model or settings.deepseek_model,
        settings=settings,
    )
    documents = load_and_chunk_source(
        source=args.source,
        source_type=args.source_type,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    retriever = BM25DocumentRetriever(
        documents=documents,
        top_k=args.top_k,
    )
    graph = build_graph(
        retriever=retriever,
        query_classifier=LLMQueryClassifier(model=model),
        multi_query_rewriter=LLMMultiQueryRewriter(model=model),
        retrieval_grader=CompositeRetrievalGrader(
            graders=[
                RuleBasedRetrievalGrader(),
                LLMBasedRetrievalGrader(model=model),
            ]
        ),
        query_rewriter=LLMQueryRewriter(model=model),
        answer_generator=LLMAnswerGenerator(model=model),
    )

    result = graph.invoke(
        {
            "question": args.question,
            "active_query": args.question,
            "retrieval_queries": [],
            "documents": [],
            "query_classification": None,
            "retrieval_decision": None,
            "query_rewrite": None,
            "multi_query_rewrite": None,
            "answer": None,
        }
    )

    print(result["answer"].answer)


def create_deepseek_model(model_name: str, settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=model_name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the agentic RAG workflow against a PDF or web source.",
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-type", choices=["pdf", "web"], required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--model")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=1_000)
    parser.add_argument("--chunk-overlap", type=int, default=200)

    return parser.parse_args()


if __name__ == "__main__":
    main()

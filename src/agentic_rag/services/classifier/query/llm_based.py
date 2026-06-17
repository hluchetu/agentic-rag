from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

from agentic_rag.domain.models import QueryClassification
from agentic_rag.prompts.loader import load_prompt


class LLMQueryClassifier:
    def __init__(self, model: BaseChatModel) -> None:
        self._model = model.with_structured_output(QueryClassification)
        self._prompt = load_prompt(
            file_name="query_classification.yaml",
            prompt_name="query_classification",
        )

    def classify(self, question: str) -> QueryClassification:
        messages = [
            SystemMessage(content=self._prompt["system"]),
            HumanMessage(
                content=self._prompt["user"].format(
                    question=question,
                )
            ),
        ]

        result = self._model.invoke(messages)

        return result

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

from agentic_rag.domain.models import GeneratedAnswer
from agentic_rag.prompts.loader import load_prompt


class LLMAnswerGenerator:
    def __init__(self, model: BaseChatModel) -> None:
        self._model = model.with_structured_output(GeneratedAnswer)
        self._prompt = load_prompt(
            file_name="generation.yaml",
            prompt_name="answer_generation",
        )

    def generate(
        self,
        question: str,
        documents: list[Document],
    ) -> GeneratedAnswer:
        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        messages = [
            SystemMessage(content=self._prompt["system"]),
            HumanMessage(
                content=self._prompt["user"].format(
                    question=question,
                    context=context,
                )
            ),
        ]

        result = self._model.invoke(messages)

        return result

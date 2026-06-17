class AgenticRAGError(Exception):
    """Base error for agentic RAG failures."""


class MissingStateError(AgenticRAGError):
    """Raised when required graph state is missing."""

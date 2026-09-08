"""Diagnostic follow-up question generation for failed RAG turns.

This package implements *only* the follow-up question-generation stage of the
RAG failure-attribution study. It never classifies or predicts the six RAG
failure categories; a separate downstream component consumes the generated
follow-up (and its subsequent RAG answer) to do that.

Research question
-----------------
Can answer-conditioned *dynamic* follow-up question generation provide better
evidence for RAG failure attribution than *static* follow-up generation?

Two generators share the same failed turns:

* :class:`~.static_question_adapter.StaticFollowUpGenerator` (Xie-style baseline)
  sees the conversation history and the failed question, but **never** the
  failed RAG answer.
* :class:`~.dynamic_question_generator.DynamicFollowUpGenerator` (proposed
  method) additionally sees the failed RAG answer, so its question can react to
  the specific answer that was produced.

Neither generator is ever given the gold answer, ``truth_history``, judge
reasoning, retrieved/final-context documents, or any failure label. See
:mod:`.conversation_loader` and :mod:`.schemas` for how that boundary is
enforced in code.
"""

from .schemas import (
    DynamicFollowUpInput,
    DynamicGenerationRecord,
    GeneratedQuestion,
    GenerationAttempt,
    GenerationInputProvenance,
    StaticFollowUpInput,
    StaticGenerationRecord,
    ValidationResult,
)
from .question_strategies import QuestionStrategy

__all__ = [
    "DynamicFollowUpInput",
    "StaticFollowUpInput",
    "GeneratedQuestion",
    "ValidationResult",
    "GenerationAttempt",
    "GenerationInputProvenance",
    "DynamicGenerationRecord",
    "StaticGenerationRecord",
    "QuestionStrategy",
]

"""Typed schemas for diagnostic follow-up generation.

The most important type here is :class:`DynamicFollowUpInput`: it is the *only*
object handed to the dynamic generator, and by construction it can hold **only**
permitted information. There is deliberately no field for the gold answer,
``truth_history``, judge reasoning, retrieved/final-context documents, or any
failure label — so those cannot leak into the generator even by accident.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .question_strategies import QuestionStrategy

# A chat message is a plain ``{"role": ..., "content": ...}`` dict, matching the
# on-disk ``input.conversation`` format so no lossy translation is needed.
Message = dict[str, str]

# Canonical label for the RePAIR-inspired proposed method. Kept as a constant so
# the two experimental conditions ("static" baseline vs this) are named in one
# place and the label is easy to keep stable across runs.
PROPOSED_METHOD_NAME = "proposed_dynamic_diagnostic"


class TargetSource(str, Enum):
    """What the follow-up reacts to in the failed turn.

    * ``failed_answer_span`` — a specific span/claim/topic in the failed answer
      (e.g. a named person the answer asserted).
    * ``response_behavior`` — an observable behavior of the answer rather than a
      span (e.g. it was empty, echoed the user, or omitted a requested part).
    """

    FAILED_ANSWER_SPAN = "failed_answer_span"
    RESPONSE_BEHAVIOR = "response_behavior"

    @classmethod
    def from_text(cls, value: str) -> "TargetSource":
        normalized = (value or "").strip().lower()
        for source in cls:
            if source.value == normalized:
                return source
        return cls.FAILED_ANSWER_SPAN


class InformationBoundaryError(ValueError):
    """Raised when a permitted-input envelope is constructed inconsistently."""


@dataclass(frozen=True)
class DynamicFollowUpInput:
    """Permitted inputs for *dynamic* (answer-conditioned) generation.

    Allowed (and all that is present):
        * ``conversation_id`` / ``failed_turn_id`` — identifiers only.
        * ``rag_history`` — the RAG conversation up to **and including** the
          failed assistant response.
        * ``failed_question`` — the user question of the failed turn.
        * ``failed_rag_answer`` — the actual answer the RAG produced.

    Forbidden (and therefore absent by design): gold answer, ``truth_history``,
    judge reasoning, correctness explanation, retrieved documents, final-context
    documents, human/predicted failure labels, oracle attribution.

    The incorrect *label* itself is not stored here — it is used only upstream to
    decide that this turn needs a follow-up.
    """

    conversation_id: str
    failed_turn_id: str
    rag_history: list[Message]
    failed_question: str
    failed_rag_answer: str

    def __post_init__(self) -> None:
        # The last two messages of the history must be exactly the failed
        # question/answer pair; this keeps ``rag_history`` and the explicit
        # fields consistent and lets prior_history() be derived safely.
        if len(self.rag_history) < 2:
            raise InformationBoundaryError(
                f"[conv={self.conversation_id} turn={self.failed_turn_id}] "
                "rag_history must contain at least the failed question and answer."
            )
        last_user, last_assistant = self.rag_history[-2], self.rag_history[-1]
        if last_user.get("role") != "user" or last_assistant.get("role") != "assistant":
            raise InformationBoundaryError(
                f"[conv={self.conversation_id} turn={self.failed_turn_id}] "
                "rag_history must end with a user question then an assistant answer."
            )
        if last_user.get("content") != self.failed_question:
            raise InformationBoundaryError(
                f"[conv={self.conversation_id} turn={self.failed_turn_id}] "
                "failed_question must match the last user message in rag_history."
            )
        if last_assistant.get("content") != self.failed_rag_answer:
            raise InformationBoundaryError(
                f"[conv={self.conversation_id} turn={self.failed_turn_id}] "
                "failed_rag_answer must match the last assistant message in rag_history."
            )

    def prior_history(self) -> list[Message]:
        """Conversation turns *before* the failed turn (safe reference context)."""
        return list(self.rag_history[:-2])


@dataclass(frozen=True)
class StaticFollowUpInput:
    """Permitted inputs for the *static* (Xie-style) baseline generator.

    Identical to the dynamic envelope **except** that the failed RAG answer is
    never present: static generation must not react to the answer. ``rag_history``
    here ends with the failed *question* (the failed assistant turn is dropped).
    """

    conversation_id: str
    failed_turn_id: str
    rag_history: list[Message]
    failed_question: str

    def __post_init__(self) -> None:
        if not self.rag_history or self.rag_history[-1].get("role") != "user":
            raise InformationBoundaryError(
                f"[conv={self.conversation_id} turn={self.failed_turn_id}] "
                "static rag_history must end with the failed user question."
            )
        if self.rag_history[-1].get("content") != self.failed_question:
            raise InformationBoundaryError(
                f"[conv={self.conversation_id} turn={self.failed_turn_id}] "
                "failed_question must match the last user message in rag_history."
            )
        # A static input that somehow carried the failed answer would violate the
        # experiment; guard against it explicitly.
        for message in self.rag_history:
            if message.get("role") == "assistant" and message is self.rag_history[-1]:
                raise InformationBoundaryError(  # pragma: no cover - defensive
                    "static rag_history must not end with an assistant answer."
                )


@dataclass
class GeneratedQuestion:
    """Structured output of one generation call (mirrors the model JSON)."""

    strategy: QuestionStrategy
    target_from_failed_answer: str
    answer_conditioning_explanation: str
    follow_up_question: str
    target_source: TargetSource = TargetSource.FAILED_ANSWER_SPAN
    introduced_new_information: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "target_source": self.target_source.value,
            "target_from_failed_answer": self.target_from_failed_answer,
            "answer_conditioning_explanation": self.answer_conditioning_explanation,
            "follow_up_question": self.follow_up_question,
            "introduced_new_information": self.introduced_new_information,
        }


@dataclass
class ValidationResult:
    """Outcome of validating one generated question.

    ``is_valid`` is the conjunction of every required rule; ``reasons`` lists the
    machine-readable codes of the rules that failed (empty when valid).
    """

    is_single_question: bool
    same_information_need: bool
    is_answer_conditioned: bool
    contains_forbidden_information: bool
    introduces_unrelated_topic: bool
    provides_correct_answer: bool
    mentions_failure_category: bool
    is_grammatical: bool
    allows_abstention: bool
    is_valid: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationAttempt:
    """A single generate→validate attempt, retained for provenance/auditing."""

    attempt_number: int
    raw_response: str
    generated: GeneratedQuestion | None
    validation: ValidationResult | None
    retry_feedback_given: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "raw_response": self.raw_response,
            "generated": self.generated.to_dict() if self.generated else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "retry_feedback_given": self.retry_feedback_given,
            "error": self.error,
        }


@dataclass
class GenerationInputProvenance:
    """Explicit record of what information the generator was and was not given.

    The forbidden flags default to ``False`` and are never set to ``True`` by
    this package — they exist so downstream consumers can verify the boundary.
    """

    used_rag_history: bool = True
    used_failed_rag_answer: bool = True
    used_gold_answer: bool = False
    used_truth_history: bool = False
    used_judge_reasoning: bool = False
    used_failure_category: bool = False
    used_retrieved_documents: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DynamicGenerationRecord:
    """One output row for a dynamically generated diagnostic follow-up."""

    conversation_id: str
    failed_turn_id: str
    original_question: str
    original_rag_answer: str
    question_strategy: QuestionStrategy | None
    target_source: TargetSource | None
    target_from_failed_answer: str
    generated_follow_up: str
    answer_conditioning_explanation: str
    generation_input_provenance: GenerationInputProvenance
    validation: ValidationResult | None
    generation_attempts: int
    is_valid: bool
    attempts: list[GenerationAttempt] = field(default_factory=list)
    method: str = PROPOSED_METHOD_NAME
    dataset: str | None = None
    rag: str | None = None
    source_conversation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "failed_turn_id": self.failed_turn_id,
            "method": self.method,
            "dataset": self.dataset,
            "rag": self.rag,
            "source_conversation_id": self.source_conversation_id,
            "original_question": self.original_question,
            "original_rag_answer": self.original_rag_answer,
            "question_strategy": (
                self.question_strategy.value if self.question_strategy else None
            ),
            "target_source": self.target_source.value if self.target_source else None,
            "target_from_failed_answer": self.target_from_failed_answer,
            "generated_follow_up": self.generated_follow_up,
            "answer_conditioning_explanation": self.answer_conditioning_explanation,
            "generation_input_provenance": self.generation_input_provenance.to_dict(),
            "validation": self.validation.to_dict() if self.validation else None,
            "is_valid": self.is_valid,
            "generation_attempts": self.generation_attempts,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


@dataclass
class StaticGenerationRecord:
    """One output row for the static (Xie-style) baseline follow-up.

    Note the provenance defaults: the static generator does **not** use the
    failed RAG answer.
    """

    conversation_id: str
    failed_turn_id: str
    original_question: str
    generated_follow_up: str
    generation_input_provenance: GenerationInputProvenance
    generation_attempts: int
    method: str = "static"
    dataset: str | None = None
    rag: str | None = None
    source_conversation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "failed_turn_id": self.failed_turn_id,
            "method": self.method,
            "dataset": self.dataset,
            "rag": self.rag,
            "source_conversation_id": self.source_conversation_id,
            "original_question": self.original_question,
            "generated_follow_up": self.generated_follow_up,
            "generation_input_provenance": self.generation_input_provenance.to_dict(),
            "generation_attempts": self.generation_attempts,
        }


def static_provenance() -> GenerationInputProvenance:
    """Provenance for the static baseline: history only, no failed answer."""
    return GenerationInputProvenance(
        used_rag_history=True,
        used_failed_rag_answer=False,
    )

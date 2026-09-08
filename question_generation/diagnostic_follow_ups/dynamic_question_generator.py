"""Dynamic (answer-conditioned) diagnostic follow-up generator.

Given one failed turn's permitted inputs, this produces a single follow-up
question that reacts to the actual failed RAG answer, validates it, and retries
with targeted feedback up to a small fixed maximum. It never sees or emits gold
answers or failure categories.
"""

from __future__ import annotations

import json
import re

from .conversation_loader import dynamic_provenance
from .model_client import FollowUpModelClient
from .prompt_templates import build_dynamic_question_prompt
from .question_strategies import QuestionStrategy, suggest_strategy
from .question_validator import QuestionValidator, build_retry_feedback
from .schemas import (
    PROPOSED_METHOD_NAME,
    DynamicFollowUpInput,
    DynamicGenerationRecord,
    GeneratedQuestion,
    GenerationAttempt,
    TargetSource,
    ValidationResult,
)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class GeneratedQuestionParseError(ValueError):
    """Raised when a model response cannot be parsed into a GeneratedQuestion."""


def parse_generated_question(raw_response: str) -> GeneratedQuestion:
    """Parse a model response (one JSON object) into a :class:`GeneratedQuestion`.

    Tolerant of surrounding prose/code fences: the first ``{...}`` span is used.
    Missing optional keys default sensibly; a missing question is an error.
    """
    match = _JSON_OBJECT_RE.search(raw_response or "")
    if not match:
        raise GeneratedQuestionParseError("no JSON object found in response")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise GeneratedQuestionParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise GeneratedQuestionParseError("response JSON is not an object")

    follow_up = str(payload.get("follow_up_question", "")).strip()
    if not follow_up:
        raise GeneratedQuestionParseError("response is missing 'follow_up_question'")

    return GeneratedQuestion(
        strategy=QuestionStrategy.from_text(str(payload.get("strategy", ""))),
        target_source=TargetSource.from_text(str(payload.get("target_source", ""))),
        target_from_failed_answer=str(payload.get("target_from_failed_answer", "")),
        answer_conditioning_explanation=str(
            payload.get("answer_conditioning_explanation", "")
        ),
        follow_up_question=follow_up,
        introduced_new_information=bool(payload.get("introduced_new_information", False)),
    )


class DynamicFollowUpGenerator:
    """Generate, validate, and retry one dynamic follow-up per failed turn."""

    def __init__(
        self,
        client: FollowUpModelClient,
        *,
        validator: QuestionValidator | None = None,
        max_attempts: int = 3,
        offer_strategy_hint: bool = True,
        method_name: str = PROPOSED_METHOD_NAME,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.client = client
        self.validator = validator or QuestionValidator()
        self.max_attempts = max_attempts
        self.offer_strategy_hint = offer_strategy_hint
        self.method_name = method_name

    def generate_follow_up_for_failed_turn(
        self, failed_input: DynamicFollowUpInput
    ) -> DynamicGenerationRecord:
        """Run the generate→validate→retry loop and return one output record."""
        strategy_hint = (
            suggest_strategy(failed_input) if self.offer_strategy_hint else None
        )

        attempts: list[GenerationAttempt] = []
        retry_feedback: str | None = None
        last_generated: GeneratedQuestion | None = None
        last_validation: ValidationResult | None = None

        for attempt_number in range(1, self.max_attempts + 1):
            prompt = build_dynamic_question_prompt(
                failed_input,
                strategy_hint=strategy_hint,
                retry_feedback=retry_feedback,
            )
            raw_response = self.client.generate(prompt)

            try:
                generated = parse_generated_question(raw_response)
            except GeneratedQuestionParseError as exc:
                attempts.append(
                    GenerationAttempt(
                        attempt_number=attempt_number,
                        raw_response=raw_response,
                        generated=None,
                        validation=None,
                        retry_feedback_given=retry_feedback,
                        error=str(exc),
                    )
                )
                retry_feedback = (
                    "Respond with exactly one JSON object containing the required "
                    "keys, including a non-empty 'follow_up_question'."
                )
                continue

            validation = self.validator.validate_generated_question(
                failed_input, generated
            )
            attempts.append(
                GenerationAttempt(
                    attempt_number=attempt_number,
                    raw_response=raw_response,
                    generated=generated,
                    validation=validation,
                    retry_feedback_given=retry_feedback,
                )
            )
            last_generated, last_validation = generated, validation

            if validation.is_valid:
                return self._build_record(failed_input, generated, validation, attempts)

            # Targeted, gold-free feedback for the next attempt.
            retry_feedback = build_retry_feedback(
                validation, generated.follow_up_question
            )

        # Every attempt failed validation (or failed to parse). Never silently
        # accept: mark the record invalid, preserving the best attempt we have.
        return self._build_record(
            failed_input, last_generated, last_validation, attempts, forced_invalid=True
        )

    def _build_record(
        self,
        failed_input: DynamicFollowUpInput,
        generated: GeneratedQuestion | None,
        validation: ValidationResult | None,
        attempts: list[GenerationAttempt],
        *,
        forced_invalid: bool = False,
    ) -> DynamicGenerationRecord:
        is_valid = bool(validation and validation.is_valid) and not forced_invalid
        return DynamicGenerationRecord(
            conversation_id=failed_input.conversation_id,
            failed_turn_id=failed_input.failed_turn_id,
            original_question=failed_input.failed_question,
            original_rag_answer=failed_input.failed_rag_answer,
            question_strategy=generated.strategy if generated else None,
            target_source=generated.target_source if generated else None,
            target_from_failed_answer=(
                generated.target_from_failed_answer if generated else ""
            ),
            generated_follow_up=generated.follow_up_question if generated else "",
            answer_conditioning_explanation=(
                generated.answer_conditioning_explanation if generated else ""
            ),
            generation_input_provenance=dynamic_provenance(),
            validation=validation,
            generation_attempts=len(attempts),
            is_valid=is_valid,
            attempts=attempts,
            method=self.method_name,
        )

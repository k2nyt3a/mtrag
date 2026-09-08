"""Tests for deterministic validation and targeted retry feedback."""

from __future__ import annotations

import json
import unittest

from question_generation.diagnostic_follow_ups.question_strategies import QuestionStrategy
from question_generation.diagnostic_follow_ups.question_validator import (
    QuestionValidator,
    build_retry_feedback,
)
from question_generation.diagnostic_follow_ups.schemas import GeneratedQuestion

from tests.diagnostic_follow_ups import fixtures


def _question(text: str, strategy: QuestionStrategy) -> GeneratedQuestion:
    return GeneratedQuestion(
        strategy=strategy,
        target_from_failed_answer="Stafford Cripps",
        answer_conditioning_explanation="reacts to the failed claim",
        follow_up_question=text,
    )


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = QuestionValidator()
        self.evidence_input = fixtures.dynamic_input_for_turn(fixtures.EVIDENCE_TURN)

    def test_good_question_is_valid(self) -> None:
        generated = _question(
            "You said Stafford Cripps led the Union army. Can you verify that "
            "using the available sources?",
            QuestionStrategy.EVIDENCE_VERIFICATION,
        )
        result = self.validator.validate_generated_question(
            self.evidence_input, generated
        )
        self.assertTrue(result.is_valid, result.reasons)
        self.assertEqual(result.reasons, [])

    def test_failure_category_mention_is_rejected(self) -> None:
        generated = _question(
            "You mentioned the Union army; was this a retrieval failure according "
            "to the sources?",
            QuestionStrategy.EVIDENCE_VERIFICATION,
        )
        result = self.validator.validate_generated_question(
            self.evidence_input, generated
        )
        self.assertFalse(result.is_valid)
        self.assertTrue(result.mentions_failure_category)
        self.assertTrue(result.contains_forbidden_information)
        self.assertIn("mentions_failure_category", result.reasons)

    def test_new_number_is_treated_as_forbidden_information(self) -> None:
        generated = _question(
            "You said the Union army was led by someone in 1849; can you verify "
            "that claim?",
            QuestionStrategy.EVIDENCE_VERIFICATION,
        )
        result = self.validator.validate_generated_question(
            self.evidence_input, generated
        )
        self.assertTrue(result.provides_correct_answer)
        self.assertFalse(result.is_valid)
        self.assertIn("introduces_new_facts_or_answer", result.reasons)

    def test_unrelated_topic_shift_is_rejected(self) -> None:
        generated = _question(
            "Can you explain quantum chromodynamics and gluon confinement in "
            "particle physics experiments instead?",
            QuestionStrategy.GENERAL_VERIFICATION,
        )
        result = self.validator.validate_generated_question(
            self.evidence_input, generated
        )
        self.assertTrue(result.introduces_unrelated_topic)
        self.assertFalse(result.is_valid)
        self.assertIn("unrelated_topic_shift", result.reasons)

    def test_multiple_questions_are_rejected(self) -> None:
        generated = _question(
            "You said Stafford Cripps led the army. Is that right? Where is it "
            "stated?",
            QuestionStrategy.EVIDENCE_VERIFICATION,
        )
        result = self.validator.validate_generated_question(
            self.evidence_input, generated
        )
        self.assertFalse(result.is_single_question)
        self.assertFalse(result.is_valid)

    def test_validation_result_is_serializable(self) -> None:
        generated = _question(
            "You said Stafford Cripps led the Union army. Can you verify that "
            "using the available sources?",
            QuestionStrategy.EVIDENCE_VERIFICATION,
        )
        result = self.validator.validate_generated_question(
            self.evidence_input, generated
        )
        payload = json.dumps(result.to_dict())
        self.assertIn("is_valid", payload)


class RetryFeedbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = QuestionValidator()
        self.evidence_input = fixtures.dynamic_input_for_turn(fixtures.EVIDENCE_TURN)

    def test_feedback_targets_the_failed_rule(self) -> None:
        # On-topic but not reacting to the failed answer -> not answer-conditioned.
        generated = _question(
            "Who led the Union army according to the sources?",
            QuestionStrategy.EVIDENCE_VERIFICATION,
        )
        result = self.validator.validate_generated_question(
            self.evidence_input, generated
        )
        self.assertFalse(result.is_answer_conditioned)
        self.assertIn("not_answer_conditioned", result.reasons)

        feedback = build_retry_feedback(result, generated.follow_up_question)
        self.assertIn("React to a specific span", feedback)
        self.assertIn(generated.follow_up_question, feedback)

    def test_feedback_never_contains_forbidden_markers(self) -> None:
        generated = _question(
            "Tell me something.",  # will fail several rules
            QuestionStrategy.GENERAL_VERIFICATION,
        )
        result = self.validator.validate_generated_question(
            self.evidence_input, generated
        )
        feedback = build_retry_feedback(result, generated.follow_up_question)
        for marker in fixtures.FORBIDDEN_MARKERS:
            self.assertNotIn(marker, feedback)


if __name__ == "__main__":
    unittest.main()

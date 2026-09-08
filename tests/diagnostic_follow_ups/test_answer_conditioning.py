"""Tests that strategy selection and generation react to the failed answer."""

from __future__ import annotations

import unittest

from question_generation.diagnostic_follow_ups.model_client import build_offline_question
from question_generation.diagnostic_follow_ups.question_strategies import (
    REPAIR_OPERATION_MAPPING,
    QuestionStrategy,
    suggest_strategy,
)
from question_generation.diagnostic_follow_ups.question_validator import QuestionValidator
from question_generation.diagnostic_follow_ups.schemas import TargetSource

from tests.diagnostic_follow_ups import fixtures

_GUINEA_PIG_PRIOR = [
    {"role": "user", "content": "Tell me about guinea pigs."},
    {"role": "assistant", "content": "Guinea pigs are small rodents kept as pets."},
]


class StrategySelectionTests(unittest.TestCase):
    def test_entity_mistake_yields_contrastive_reformulation(self) -> None:
        dynamic_input = fixtures.dynamic_input_for_turn(fixtures.CONTRASTIVE_TURN)
        self.assertEqual(
            suggest_strategy(dynamic_input),
            QuestionStrategy.CONTRASTIVE_REFORMULATION,
        )

    def test_specific_claim_yields_evidence_verification(self) -> None:
        dynamic_input = fixtures.dynamic_input_for_turn(fixtures.EVIDENCE_TURN)
        self.assertEqual(
            suggest_strategy(dynamic_input),
            QuestionStrategy.EVIDENCE_VERIFICATION,
        )

    def test_empty_answer_yields_answerability_probe(self) -> None:
        dynamic_input = fixtures.dynamic_input_for_turn(fixtures.ANSWERABILITY_TURN)
        self.assertEqual(
            suggest_strategy(dynamic_input),
            QuestionStrategy.ANSWERABILITY_PROBE,
        )

    def test_echoed_answer_yields_answerability_probe(self) -> None:
        question = "Where do guinea pigs sleep in the wild?"
        echoed = fixtures.make_dynamic_input(
            _GUINEA_PIG_PRIOR, question, question  # answer just repeats the question
        )
        self.assertEqual(
            suggest_strategy(echoed), QuestionStrategy.ANSWERABILITY_PROBE
        )

    def test_multipart_omission_yields_completion_probe(self) -> None:
        dynamic_input = fixtures.dynamic_input_for_turn(fixtures.COMPLETION_TURN)
        self.assertEqual(
            suggest_strategy(dynamic_input),
            QuestionStrategy.COMPLETION_PROBE,
        )

    def test_complex_request_yields_atomic_decomposition(self) -> None:
        dynamic_input = fixtures.dynamic_input_for_turn(fixtures.ATOMIC_TURN)
        self.assertEqual(
            suggest_strategy(dynamic_input),
            QuestionStrategy.ATOMIC_DECOMPOSITION,
        )

    def test_short_contextual_query_yields_retrieval_oriented(self) -> None:
        # A short, elliptical query whose answer drifted to another topic.
        short_query = fixtures.make_dynamic_input(
            _GUINEA_PIG_PRIOR,
            "Vitamin C research",
            "Guinea pigs are used in scientific research as human subjects.",
        )
        self.assertEqual(
            suggest_strategy(short_query),
            QuestionStrategy.RETRIEVAL_ORIENTED_REFORMULATION,
        )


class RepairMappingTests(unittest.TestCase):
    def test_seven_strategies_including_retrieval_oriented(self) -> None:
        values = {s.value for s in QuestionStrategy}
        self.assertEqual(len(values), 7)
        self.assertIn("retrieval_oriented_reformulation", values)

    def test_repair_mapping_excludes_refinedoc(self) -> None:
        self.assertIsNone(REPAIR_OPERATION_MAPPING["REFINEDOC"])
        self.assertEqual(
            REPAIR_OPERATION_MAPPING["RETRIEVAL"],
            QuestionStrategy.RETRIEVAL_ORIENTED_REFORMULATION,
        )
        self.assertEqual(
            REPAIR_OPERATION_MAPPING["DECOMPOSE"],
            QuestionStrategy.ATOMIC_DECOMPOSITION,
        )


class AnswerConditioningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = QuestionValidator()

    def test_offline_questions_are_answer_conditioned_and_valid(self) -> None:
        for turn in (
            fixtures.CONTRASTIVE_TURN,
            fixtures.EVIDENCE_TURN,
            fixtures.ANSWERABILITY_TURN,
            fixtures.COMPLETION_TURN,
            fixtures.ATOMIC_TURN,
        ):
            dynamic_input = fixtures.dynamic_input_for_turn(turn)
            strategy = suggest_strategy(dynamic_input)
            generated = build_offline_question(dynamic_input, strategy)
            result = self.validator.validate_generated_question(dynamic_input, generated)
            self.assertTrue(
                result.is_answer_conditioned,
                f"turn {turn} question was not answer-conditioned: {generated.follow_up_question!r}",
            )
            self.assertTrue(
                result.is_valid,
                f"turn {turn} question invalid ({result.reasons}): {generated.follow_up_question!r}",
            )
            # target_source is always populated with a valid enum value.
            self.assertIn(generated.target_source, set(TargetSource))

    def test_empty_answer_targets_response_behavior(self) -> None:
        dynamic_input = fixtures.dynamic_input_for_turn(fixtures.ANSWERABILITY_TURN)
        generated = build_offline_question(
            dynamic_input, suggest_strategy(dynamic_input)
        )
        self.assertEqual(generated.target_source, TargetSource.RESPONSE_BEHAVIOR)

    def test_two_specific_claims_produce_different_targets(self) -> None:
        # Spec example: same question, two different (specific) failed answers.
        prior = [
            {"role": "user", "content": "Tell me about the Quit India Movement."},
            {"role": "assistant", "content": "It began in 1942."},
        ]
        question = "Who was the Viceroy during the Quit India Movement?"
        input_a = fixtures.make_dynamic_input(prior, question, "Stafford Cripps.")
        input_b = fixtures.make_dynamic_input(prior, question, "Mahatma Gandhi.")

        gen_a = build_offline_question(input_a, suggest_strategy(input_a))
        gen_b = build_offline_question(input_b, suggest_strategy(input_b))

        self.assertIn("Stafford Cripps", gen_a.target_from_failed_answer)
        self.assertIn("Mahatma Gandhi", gen_b.target_from_failed_answer)
        self.assertNotEqual(gen_a.follow_up_question, gen_b.follow_up_question)
        self.assertNotIn("Mahatma", gen_a.follow_up_question)
        self.assertNotIn("Cripps", gen_b.follow_up_question)

    def test_different_failed_answers_change_the_generated_question(self) -> None:
        prior = [
            {"role": "user", "content": "Tell me about the Quit India Movement."},
            {"role": "assistant", "content": "It began in 1942."},
        ]
        question = "Who was the Viceroy during the Quit India Movement?"
        input_claim = fixtures.make_dynamic_input(prior, question, "Stafford Cripps.")
        input_empty = fixtures.make_dynamic_input(prior, question, "")

        gen_claim = build_offline_question(
            input_claim, suggest_strategy(input_claim)
        )
        gen_empty = build_offline_question(
            input_empty, suggest_strategy(input_empty)
        )

        # The claim triggers evidence verification and quotes the claim; the empty
        # answer triggers an answerability probe. The questions must differ.
        self.assertEqual(gen_claim.strategy, QuestionStrategy.EVIDENCE_VERIFICATION)
        self.assertIn("Stafford Cripps", gen_claim.follow_up_question)
        self.assertEqual(gen_empty.strategy, QuestionStrategy.ANSWERABILITY_PROBE)
        self.assertNotEqual(
            gen_claim.follow_up_question, gen_empty.follow_up_question
        )


if __name__ == "__main__":
    unittest.main()

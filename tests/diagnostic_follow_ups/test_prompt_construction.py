"""Tests that prompts carry exactly the permitted information."""

from __future__ import annotations

import unittest

from question_generation.diagnostic_follow_ups.conversation_loader import (
    build_static_input,
)
from question_generation.diagnostic_follow_ups.prompt_templates import (
    build_dynamic_question_prompt,
    build_static_question_prompt,
)

from tests.diagnostic_follow_ups import fixtures


class DynamicPromptTests(unittest.TestCase):
    def test_prompt_includes_rag_history(self) -> None:
        dynamic_input = fixtures.dynamic_input_for_turn(fixtures.CONTRASTIVE_TURN)
        prompt = build_dynamic_question_prompt(dynamic_input)
        # A message from before the failed turn must be present.
        self.assertIn("U.S. Civil War", prompt.user)

    def test_prompt_includes_failed_question_and_answer(self) -> None:
        dynamic_input = fixtures.dynamic_input_for_turn(fixtures.EVIDENCE_TURN)
        prompt = build_dynamic_question_prompt(dynamic_input)
        self.assertIn("Who led the Union army?", prompt.user)
        self.assertIn("Stafford Cripps", prompt.user)

    def test_prompt_never_contains_forbidden_markers(self) -> None:
        for turn in (
            fixtures.CONTRASTIVE_TURN,
            fixtures.EVIDENCE_TURN,
            fixtures.ANSWERABILITY_TURN,
            fixtures.COMPLETION_TURN,
            fixtures.ATOMIC_TURN,
        ):
            dynamic_input = fixtures.dynamic_input_for_turn(turn)
            prompt = build_dynamic_question_prompt(dynamic_input)
            text = prompt.as_single_string()
            for marker in fixtures.FORBIDDEN_MARKERS:
                self.assertNotIn(marker, text)

    def test_prompt_never_contains_retrieved_documents(self) -> None:
        # The fixture's retrieved / final-context documents carry a distinctive
        # marker; it must never appear in a generator-facing prompt.
        for turn in (fixtures.EVIDENCE_TURN, fixtures.ANSWERABILITY_TURN):
            dynamic_input = fixtures.dynamic_input_for_turn(turn)
            text = build_dynamic_question_prompt(dynamic_input).as_single_string()
            self.assertNotIn("RETRIEVEDSECRET", text)

    def test_prompt_does_not_mention_failure_categories(self) -> None:
        dynamic_input = fixtures.dynamic_input_for_turn(fixtures.EVIDENCE_TURN)
        prompt = build_dynamic_question_prompt(dynamic_input)
        # The instructions explicitly forbid categories; ensure none is named as a
        # target in the rendered user content.
        lowered = prompt.user.lower()
        for category in ("knowledge boundary", "context selection", "response coverage"):
            self.assertNotIn(category, lowered)

    def test_same_question_different_answers_produce_different_prompts(self) -> None:
        prior = [
            {"role": "user", "content": "Tell me about the Quit India Movement."},
            {"role": "assistant", "content": "It began in 1942."},
        ]
        question = "Who was the Viceroy during the Quit India Movement?"
        input_a = fixtures.make_dynamic_input(prior, question, "Stafford Cripps.")
        input_b = fixtures.make_dynamic_input(prior, question, "Lord Linlithgow was Viceroy.")

        prompt_a = build_dynamic_question_prompt(input_a)
        prompt_b = build_dynamic_question_prompt(input_b)

        # Each prompt reflects its own failed answer, so they must differ.
        self.assertIn("Stafford Cripps.", prompt_a.user)
        self.assertIn("Lord Linlithgow was Viceroy.", prompt_b.user)
        self.assertNotEqual(prompt_a.user, prompt_b.user)
        self.assertNotIn("Linlithgow", prompt_a.user)
        self.assertNotIn("Stafford Cripps", prompt_b.user)


class StaticPromptTests(unittest.TestCase):
    def test_static_prompt_excludes_the_failed_answer(self) -> None:
        conversation = fixtures.sample_conversation()
        failed_turn = next(conversation.iter_failed_turns())  # turn 2
        static_input = build_static_input(conversation, failed_turn)
        prompt = build_static_question_prompt(static_input)
        text = prompt.as_single_string()
        self.assertNotIn("World War I was terrible", text)
        # But the failed question is still present.
        self.assertIn("What made that war terrible?", text)

    def test_static_prompt_never_contains_forbidden_markers(self) -> None:
        conversation = fixtures.sample_conversation()
        for failed_turn in conversation.iter_failed_turns():
            static_input = build_static_input(conversation, failed_turn)
            text = build_static_question_prompt(static_input).as_single_string()
            for marker in fixtures.FORBIDDEN_MARKERS:
                self.assertNotIn(marker, text)


if __name__ == "__main__":
    unittest.main()

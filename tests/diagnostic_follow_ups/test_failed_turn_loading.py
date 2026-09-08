"""Tests for failed-turn selection and independent per-turn branching."""

from __future__ import annotations

import unittest

from question_generation.diagnostic_follow_ups.conversation_loader import (
    build_dynamic_input,
)
from question_generation.diagnostic_follow_ups.dynamic_question_generator import (
    DynamicFollowUpGenerator,
)
from question_generation.diagnostic_follow_ups.model_client import (
    OfflineHeuristicModelClient,
)

from tests.diagnostic_follow_ups import fixtures


class FailedTurnSelectionTests(unittest.TestCase):
    def test_incorrect_turns_selected_correct_skipped(self) -> None:
        conversation = fixtures.sample_conversation()
        selected = list(conversation.iter_failed_turns())
        self.assertEqual([ft.turn_id for ft in selected], ["2", "3", "4", "5", "6"])
        self.assertNotIn("1", [ft.turn_id for ft in selected])


class IndependentBranchTests(unittest.TestCase):
    def test_each_failed_turn_builds_history_through_its_own_answer(self) -> None:
        conversation = fixtures.sample_conversation()
        for failed_turn in conversation.iter_failed_turns():
            dynamic_input = build_dynamic_input(conversation, failed_turn)
            expected_len = 2 * (failed_turn.turn_index + 1)
            self.assertEqual(len(dynamic_input.rag_history), expected_len)
            # History ends exactly at this turn's failed answer.
            self.assertEqual(
                dynamic_input.rag_history[-1]["content"],
                dynamic_input.failed_rag_answer,
            )

    def test_branches_do_not_contaminate_each_other(self) -> None:
        conversation = fixtures.sample_conversation()
        original_messages = [dict(m) for m in conversation.messages]

        # Generating for one failed turn must not mutate the conversation or
        # leak a generated follow-up into another turn's input.
        generator = DynamicFollowUpGenerator(OfflineHeuristicModelClient())
        failed_turns = list(conversation.iter_failed_turns())

        first_input = build_dynamic_input(conversation, failed_turns[0])
        first_record = generator.generate_follow_up_for_failed_turn(first_input)

        later_input = build_dynamic_input(conversation, failed_turns[-1])
        history_text = " ".join(m["content"] for m in later_input.rag_history)

        # The follow-up produced for the first branch is absent from a later branch.
        self.assertNotIn(first_record.generated_follow_up, history_text)
        # The underlying conversation is unchanged.
        self.assertEqual(conversation.messages, original_messages)

    def test_reasoning_is_never_carried_onto_a_failed_turn(self) -> None:
        # Selection uses only the label; judge reasoning is not attached to the
        # FailedTurn object.
        conversation = fixtures.sample_conversation()
        failed_turn = next(conversation.iter_failed_turns())
        self.assertFalse(hasattr(failed_turn, "reasoning"))
        self.assertEqual(failed_turn.label, "incorrect")


if __name__ == "__main__":
    unittest.main()

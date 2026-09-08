"""Tests for turn selection and the generator information boundary."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from question_generation.diagnostic_follow_ups.conversation_loader import (
    build_dynamic_input,
    build_static_input,
    load_conversation_file,
)
from question_generation.diagnostic_follow_ups.schemas import (
    DynamicFollowUpInput,
    StaticFollowUpInput,
)

from tests.diagnostic_follow_ups import fixtures


class TurnSelectionTests(unittest.TestCase):
    def test_only_incorrect_turns_are_selected(self) -> None:
        conversation = fixtures.sample_conversation()
        selected = list(conversation.iter_failed_turns())
        # Turns 2-6 are incorrect (0-based indices 1-5); turn 1 is correct.
        self.assertEqual([ft.turn_index for ft in selected], [1, 2, 3, 4, 5])
        self.assertTrue(all(ft.label == "incorrect" for ft in selected))

    def test_correct_turn_is_skipped(self) -> None:
        conversation = fixtures.sample_conversation()
        selected_indices = {ft.turn_index for ft in conversation.iter_failed_turns()}
        self.assertNotIn(0, selected_indices)  # the correct first turn

    def test_turn_ids_are_carried_through(self) -> None:
        conversation = fixtures.sample_conversation()
        first_failed = next(conversation.iter_failed_turns())
        self.assertEqual(first_failed.turn_id, "2")
        self.assertEqual(first_failed.conversation_id, "001")
        self.assertEqual(first_failed.dataset, "testds")
        self.assertEqual(first_failed.rag, "testrag")


class InformationBoundaryTests(unittest.TestCase):
    def test_annotated_conversation_has_no_forbidden_fields(self) -> None:
        conversation = fixtures.sample_conversation()
        for forbidden_attr in (
            "ground_truth_answers",
            "truth_history",
            "reasoning",
            "judge_reasoning",
            "retrieved_documents",
            "final_context_documents",
        ):
            self.assertFalse(
                hasattr(conversation, forbidden_attr),
                f"AnnotatedConversation unexpectedly exposes {forbidden_attr!r}",
            )

    def test_dynamic_input_schema_cannot_hold_forbidden_information(self) -> None:
        field_names = set(DynamicFollowUpInput.__dataclass_fields__)
        self.assertEqual(
            field_names,
            {
                "conversation_id",
                "failed_turn_id",
                "rag_history",
                "failed_question",
                "failed_rag_answer",
            },
        )

    def test_dynamic_input_contains_history_and_failed_answer(self) -> None:
        dynamic_input = fixtures.dynamic_input_for_turn(fixtures.EVIDENCE_TURN)
        self.assertEqual(
            dynamic_input.failed_question, "Who led the Union army?"
        )
        self.assertIn("Stafford Cripps", dynamic_input.failed_rag_answer)
        # rag_history ends with the failed question then the failed answer.
        self.assertEqual(dynamic_input.rag_history[-2]["role"], "user")
        self.assertEqual(dynamic_input.rag_history[-1]["role"], "assistant")
        self.assertEqual(
            dynamic_input.rag_history[-1]["content"], dynamic_input.failed_rag_answer
        )
        # prior_history excludes the failed turn's own pair.
        self.assertNotIn(
            dynamic_input.failed_rag_answer,
            [m["content"] for m in dynamic_input.prior_history()],
        )

    def test_no_forbidden_markers_reach_dynamic_input(self) -> None:
        for turn in (
            fixtures.CONTRASTIVE_TURN,
            fixtures.EVIDENCE_TURN,
            fixtures.ANSWERABILITY_TURN,
            fixtures.COMPLETION_TURN,
            fixtures.ATOMIC_TURN,
        ):
            dynamic_input = fixtures.dynamic_input_for_turn(turn)
            serialized = json.dumps(
                {
                    "history": dynamic_input.rag_history,
                    "question": dynamic_input.failed_question,
                    "answer": dynamic_input.failed_rag_answer,
                }
            )
            for marker in fixtures.FORBIDDEN_MARKERS:
                self.assertNotIn(marker, serialized)

    def test_static_input_excludes_the_failed_answer(self) -> None:
        conversation = fixtures.sample_conversation()
        failed_turn = next(conversation.iter_failed_turns())  # turn 2
        static_input = build_static_input(conversation, failed_turn)
        self.assertIsInstance(static_input, StaticFollowUpInput)
        # No field for the failed answer at all.
        self.assertNotIn("failed_rag_answer", StaticFollowUpInput.__dataclass_fields__)
        # And the answer text is absent from the static history.
        history_text = " ".join(m["content"] for m in static_input.rag_history)
        self.assertNotIn("World War I was terrible", history_text)
        self.assertEqual(static_input.rag_history[-1]["role"], "user")

    def test_loader_reads_from_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "conversation.json"
            path.write_text(json.dumps([fixtures.raw_conversation()]), encoding="utf-8")
            conversations = load_conversation_file(path)
        self.assertEqual(len(conversations), 1)
        self.assertEqual(conversations[0].num_turns, 6)
        # Dynamic input can be built without any forbidden data being available.
        dynamic_input = build_dynamic_input(
            conversations[0], next(conversations[0].iter_failed_turns())
        )
        self.assertEqual(dynamic_input.conversation_id, "001")


if __name__ == "__main__":
    unittest.main()

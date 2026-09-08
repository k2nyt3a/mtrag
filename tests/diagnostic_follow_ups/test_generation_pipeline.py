"""End-to-end tests for the generator, retry loop, and pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from question_generation.diagnostic_follow_ups.conversation_loader import (
    build_static_input,
)
from question_generation.diagnostic_follow_ups.dynamic_question_generator import (
    DynamicFollowUpGenerator,
)
from question_generation.diagnostic_follow_ups.generation_pipeline import (
    DiagnosticFollowUpPipeline,
    discover_conversation_files,
    save_generation_results,
)
from question_generation.diagnostic_follow_ups.model_client import (
    OfflineHeuristicModelClient,
    ScriptedModelClient,
)
from question_generation.diagnostic_follow_ups.static_question_adapter import (
    StaticFollowUpGenerator,
)

from tests.diagnostic_follow_ups import fixtures

_VALID_EVIDENCE_JSON = json.dumps(
    {
        "strategy": "evidence_verification",
        "target_from_failed_answer": "Stafford Cripps",
        "answer_conditioning_explanation": "verifies the named person",
        "follow_up_question": (
            "You said Stafford Cripps led the Union army. Can you verify that "
            "using the available sources?"
        ),
        "introduced_new_information": False,
    }
)

_INVALID_CATEGORY_JSON = json.dumps(
    {
        "strategy": "evidence_verification",
        "target_from_failed_answer": "Stafford Cripps",
        "answer_conditioning_explanation": "x",
        "follow_up_question": (
            "You said Stafford Cripps led the Union army; was this a retrieval "
            "failure?"
        ),
        "introduced_new_information": False,
    }
)


class RetryLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence_input = fixtures.dynamic_input_for_turn(fixtures.EVIDENCE_TURN)

    def test_invalid_then_valid_retries_with_feedback(self) -> None:
        client = ScriptedModelClient([_INVALID_CATEGORY_JSON, _VALID_EVIDENCE_JSON])
        generator = DynamicFollowUpGenerator(client, max_attempts=3)
        record = generator.generate_follow_up_for_failed_turn(self.evidence_input)

        self.assertTrue(record.is_valid)
        self.assertEqual(record.generation_attempts, 2)
        # The second call carried targeted retry feedback.
        self.assertIsNotNone(record.attempts[1].retry_feedback_given)
        self.assertIn("previous attempt was rejected", client.calls[1].user)

    def test_all_attempts_invalid_is_not_silently_accepted(self) -> None:
        client = ScriptedModelClient(
            [_INVALID_CATEGORY_JSON, _INVALID_CATEGORY_JSON, _INVALID_CATEGORY_JSON]
        )
        generator = DynamicFollowUpGenerator(client, max_attempts=3)
        record = generator.generate_follow_up_for_failed_turn(self.evidence_input)

        self.assertFalse(record.is_valid)
        self.assertEqual(record.generation_attempts, 3)

    def test_response_with_code_fence_is_parsed(self) -> None:
        wrapped = "Here you go:\n```json\n" + _VALID_EVIDENCE_JSON + "\n```"
        client = ScriptedModelClient([wrapped])
        generator = DynamicFollowUpGenerator(client, max_attempts=1)
        record = generator.generate_follow_up_for_failed_turn(self.evidence_input)
        self.assertTrue(record.is_valid)
        self.assertIn("Stafford Cripps", record.generated_follow_up)


class StaticBaselineTests(unittest.TestCase):
    def test_static_record_shape_and_provenance(self) -> None:
        conversation = fixtures.sample_conversation()
        failed_turn = next(conversation.iter_failed_turns())  # turn 2
        static_input = build_static_input(conversation, failed_turn)
        generator = StaticFollowUpGenerator(OfflineHeuristicModelClient())
        record = generator.generate_follow_up_for_failed_turn(static_input)

        self.assertEqual(record.method, "static")
        self.assertFalse(record.generation_input_provenance.used_failed_rag_answer)
        # The baseline follow-up must not quote the (unseen) failed answer.
        self.assertNotIn("World War I was terrible", record.generated_follow_up)


class PipelineTests(unittest.TestCase):
    def _write_fixture(self, directory: Path) -> Path:
        path = directory / "sub" / "conversation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([fixtures.raw_conversation()]), encoding="utf-8")
        return path

    def test_offline_pipeline_generates_valid_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            files = discover_conversation_files([root])
            self.assertEqual(len(files), 1)

            pipeline = DiagnosticFollowUpPipeline(OfflineHeuristicModelClient())
            dynamic_records, static_records, summary = pipeline.run(
                files, method="both"
            )

        # Five incorrect turns, none of them the correct first turn.
        self.assertEqual(summary.failed_turns_selected, 5)
        self.assertEqual(summary.dynamic_generated, 5)
        self.assertEqual(summary.dynamic_valid, 5)
        self.assertEqual(summary.static_generated, 5)
        self.assertNotIn("1", {r["failed_turn_id"] for r in dynamic_records})

        for record in dynamic_records:
            self.assertTrue(record["is_valid"], record)
            # RePAIR-inspired proposed method label + structured target_source.
            self.assertEqual(record["method"], "proposed_dynamic_diagnostic")
            self.assertIn(
                record["target_source"],
                {"failed_answer_span", "response_behavior"},
            )
            provenance = record["generation_input_provenance"]
            self.assertTrue(provenance["used_failed_rag_answer"])
            self.assertFalse(provenance["used_gold_answer"])
            self.assertFalse(provenance["used_truth_history"])
            self.assertFalse(provenance["used_judge_reasoning"])
            self.assertFalse(provenance["used_failure_category"])
            self.assertFalse(provenance["used_retrieved_documents"])
            # No forbidden marker anywhere in the serialized record.
            blob = json.dumps(record)
            for marker in fixtures.FORBIDDEN_MARKERS:
                self.assertNotIn(marker, blob)

    def test_records_are_saved_and_reloadable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            files = discover_conversation_files([root])
            pipeline = DiagnosticFollowUpPipeline(OfflineHeuristicModelClient())
            dynamic_records, _, _ = pipeline.run(files, method="dynamic")

            out_path = root / "out" / "dynamic_follow_ups.jsonl"
            save_generation_results(dynamic_records, out_path)
            self.assertTrue(out_path.exists())
            lines = out_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 5)
            reloaded = [json.loads(line) for line in lines]
            self.assertEqual(reloaded, dynamic_records)

    def test_dataset_and_rag_metadata_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            files = discover_conversation_files([root])
            pipeline = DiagnosticFollowUpPipeline(OfflineHeuristicModelClient())
            dynamic_records, _, _ = pipeline.run(files, method="dynamic")
        self.assertTrue(all(r["dataset"] == "testds" for r in dynamic_records))
        self.assertTrue(all(r["rag"] == "testrag" for r in dynamic_records))
        self.assertTrue(
            all(r["method"] == "proposed_dynamic_diagnostic" for r in dynamic_records)
        )


if __name__ == "__main__":
    unittest.main()

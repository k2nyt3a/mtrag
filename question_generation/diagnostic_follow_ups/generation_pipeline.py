"""Orchestration and CLI for diagnostic follow-up generation.

Reads annotated ``conversation.json`` files, selects incorrect turns, generates
dynamic and/or static follow-ups, and writes new output files. It never modifies
the input conversations, retrieval, or the RAG system.

Run a no-API dry run::

    python -m question_generation.diagnostic_follow_ups.generation_pipeline \
        --input . --method both --limit 20 --output-dir generated_follow_ups

Run a live generation (requires ANTHROPIC_API_KEY)::

    python -m question_generation.diagnostic_follow_ups.generation_pipeline \
        --input . --method proposed --live --output-dir generated_follow_ups
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .conversation_loader import (
    AnnotatedConversation,
    FailedTurn,
    build_dynamic_input,
    build_static_input,
    load_conversation_file,
)
from .dynamic_question_generator import DynamicFollowUpGenerator
from .model_client import (
    AnthropicModelClient,
    FollowUpModelClient,
    OfflineHeuristicModelClient,
)
from .static_question_adapter import StaticFollowUpGenerator

CONVERSATION_FILENAME = "conversation.json"


@dataclass
class PipelineSummary:
    """Counts describing one generation run."""

    conversations_seen: int = 0
    failed_turns_selected: int = 0
    dynamic_generated: int = 0
    dynamic_valid: int = 0
    static_generated: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "conversations_seen": self.conversations_seen,
            "failed_turns_selected": self.failed_turns_selected,
            "dynamic_generated": self.dynamic_generated,
            "dynamic_valid": self.dynamic_valid,
            "static_generated": self.static_generated,
        }


def discover_conversation_files(inputs: Iterable[str | Path]) -> list[Path]:
    """Expand input paths into concrete ``conversation.json`` files.

    Accepts individual files, directories (searched recursively), or glob
    strings. De-duplicates while preserving a stable sorted order.
    """
    found: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(path.rglob(CONVERSATION_FILENAME)))
        else:
            # Treat as a glob relative to the current working directory.
            found.extend(sorted(Path().glob(str(raw))))
    unique = sorted({p.resolve() for p in found})
    return unique


class DiagnosticFollowUpPipeline:
    """Generate dynamic and/or static follow-ups over selected failed turns."""

    def __init__(
        self,
        client: FollowUpModelClient,
        *,
        max_attempts: int = 3,
        dataset_filter: str | None = None,
        rag_filter: str | None = None,
    ) -> None:
        self.dynamic_generator = DynamicFollowUpGenerator(
            client, max_attempts=max_attempts
        )
        self.static_generator = StaticFollowUpGenerator(client)
        self.dataset_filter = dataset_filter
        self.rag_filter = rag_filter

    def _keep(self, conversation: AnnotatedConversation) -> bool:
        if self.dataset_filter and conversation.dataset != self.dataset_filter:
            return False
        if self.rag_filter and conversation.rag != self.rag_filter:
            return False
        return True

    def iter_selected_failed_turns(
        self, files: Iterable[Path]
    ) -> Iterable[tuple[AnnotatedConversation, FailedTurn]]:
        """Yield ``(conversation, failed_turn)`` for every selected incorrect turn."""
        for file_path in files:
            for conversation in load_conversation_file(file_path):
                if not self._keep(conversation):
                    continue
                for failed_turn in conversation.iter_failed_turns():
                    yield conversation, failed_turn

    def run(
        self,
        files: Iterable[Path],
        *,
        method: str = "dynamic",
        limit: int | None = None,
    ) -> tuple[list[dict], list[dict], PipelineSummary]:
        """Generate follow-ups and return ``(dynamic_records, static_records, summary)``.

        Both methods operate on exactly the same selected failed turns, so the
        underlying failure cases are identical across the two conditions.
        """
        summary = PipelineSummary()
        dynamic_records: list[dict] = []
        static_records: list[dict] = []
        seen_conversation_ids: set[str] = set()

        for conversation, failed_turn in self.iter_selected_failed_turns(files):
            if limit is not None and summary.failed_turns_selected >= limit:
                break
            seen_conversation_ids.add(conversation.source_conversation_id)
            summary.failed_turns_selected += 1

            if method in ("dynamic", "both"):
                dynamic_input = build_dynamic_input(conversation, failed_turn)
                record = self.dynamic_generator.generate_follow_up_for_failed_turn(
                    dynamic_input
                )
                record.dataset = failed_turn.dataset
                record.rag = failed_turn.rag
                record.source_conversation_id = failed_turn.source_conversation_id
                dynamic_records.append(record.to_dict())
                summary.dynamic_generated += 1
                if record.is_valid:
                    summary.dynamic_valid += 1

            if method in ("static", "both"):
                static_input = build_static_input(conversation, failed_turn)
                static_record = self.static_generator.generate_follow_up_for_failed_turn(
                    static_input
                )
                static_record.dataset = failed_turn.dataset
                static_record.rag = failed_turn.rag
                static_record.source_conversation_id = failed_turn.source_conversation_id
                static_records.append(static_record.to_dict())
                summary.static_generated += 1

        summary.conversations_seen = len(seen_conversation_ids)
        return dynamic_records, static_records, summary


def save_generation_results(records: list[dict], output_path: str | Path) -> Path:
    """Write records to ``output_path`` (``.jsonl`` per-line, else pretty JSON).

    Creates parent directories as needed; never overwrites input conversations
    because the caller chooses a distinct output location.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            json.dump(records, handle, ensure_ascii=False, indent=2)
    return path


def _build_client(use_live: bool) -> FollowUpModelClient:
    if use_live:
        return AnthropicModelClient()
    return OfflineHeuristicModelClient()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate diagnostic follow-up questions for incorrect RAG turns. "
            "Offline by default; pass --live to call the Anthropic API."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=["."],
        help="Files, directories, or globs to search for conversation.json.",
    )
    parser.add_argument(
        "--method",
        choices=["proposed", "dynamic", "static", "both"],
        default="proposed",
        help=(
            "Which follow-up method(s) to generate. 'proposed' (default) and its "
            "alias 'dynamic' both run the RePAIR-inspired proposed generator "
            "(records are labelled 'proposed_dynamic_diagnostic'); 'static' runs "
            "the Xie baseline; 'both' runs proposed + static."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="generated_follow_ups",
        help="Directory for output JSONL files (created if missing).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of failed turns to process (for pilots).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum generate/validate attempts per turn (default: 3).",
    )
    parser.add_argument("--dataset", default=None, help="Only process this dataset.")
    parser.add_argument("--rag", default=None, help="Only process this RAG method.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use the Anthropic API instead of the offline deterministic client.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicitly request the offline client (the default).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.live and args.dry_run:
        print("error: --live and --dry-run are mutually exclusive.", file=sys.stderr)
        return 2

    files = discover_conversation_files(args.input)
    if not files:
        print("error: no conversation.json files found for the given --input.", file=sys.stderr)
        return 1

    mode = "LIVE (Anthropic API)" if args.live else "OFFLINE (deterministic, no API calls)"
    print(f"Mode: {mode}")
    print(f"Found {len(files)} conversation file(s).")

    # "proposed" is the explicit name; "dynamic" is kept as a backward-compatible
    # alias. Both drive the same proposed generator (labelled internally as the
    # "dynamic" control-flow branch).
    method = "dynamic" if args.method == "proposed" else args.method

    pipeline = DiagnosticFollowUpPipeline(
        _build_client(args.live),
        max_attempts=args.max_attempts,
        dataset_filter=args.dataset,
        rag_filter=args.rag,
    )
    dynamic_records, static_records, summary = pipeline.run(
        files, method=method, limit=args.limit
    )

    output_dir = Path(args.output_dir)
    written: list[Path] = []
    if method in ("dynamic", "both"):
        written.append(
            save_generation_results(
                dynamic_records, output_dir / "proposed_dynamic_follow_ups.jsonl"
            )
        )
    if method in ("static", "both"):
        written.append(
            save_generation_results(
                static_records, output_dir / "static_follow_ups.jsonl"
            )
        )

    print("Summary:", json.dumps(summary.as_dict(), indent=2))
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

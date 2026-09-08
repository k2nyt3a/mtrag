"""Load annotated RAG conversations and build permitted generation inputs.

This module is the single choke point where on-disk conversations become
generator inputs. It is deliberately strict about the information boundary:

* :class:`AnnotatedConversation` copies **only** the permitted fields out of the
  raw JSON (messages, correctness labels, turn ids, dataset/rag identifiers). It
  has no attribute for the gold answer, ``truth_history``, judge reasoning, or
  retrieved documents, so downstream code cannot read what was never loaded.
* :func:`build_dynamic_input` / :func:`build_static_input` construct the narrow
  :class:`~.schemas.DynamicFollowUpInput` / :class:`~.schemas.StaticFollowUpInput`
  envelopes, which enforce their own invariants.

Verified data layout (see repository README): ``input.conversation`` is a flat
list of alternating ``user``/``assistant`` messages, so turn ``i`` (0-based) is
user message ``2*i`` and assistant message ``2*i + 1``; ``turn_evaluations[i]``
and ``turn_ids[i]`` describe that same turn, with ``turn_evaluations[i].turn ==
i + 1``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .schemas import (
    DynamicFollowUpInput,
    GenerationInputProvenance,
    Message,
    StaticFollowUpInput,
)

INCORRECT_LABEL = "incorrect"
CORRECT_LABEL = "correct"


@dataclass(frozen=True)
class FailedTurn:
    """A single incorrect turn selected for follow-up generation.

    ``turn_index`` is 0-based; ``turn_id`` is the dataset's own turn identifier
    string (e.g. ``"4"``). Only the correctness *label* is carried, and only so
    callers can confirm the turn is incorrect — no judge reasoning travels with it.
    """

    conversation_id: str
    source_conversation_id: str
    dataset: str
    rag: str
    turn_index: int
    turn_id: str
    label: str


@dataclass
class AnnotatedConversation:
    """A permitted-only view of one annotated conversation.

    Intentionally omits gold answers, judge reasoning, and retrieved/final-context
    documents: those fields are never copied out of the raw record, so nothing
    downstream can accidentally include them in a prompt.
    """

    conversation_id: str
    source_conversation_id: str
    dataset: str
    rag: str
    messages: list[Message]
    turn_labels: list[str]
    turn_ids: list[str]
    num_turns: int

    @classmethod
    def from_raw(cls, raw: dict) -> "AnnotatedConversation":
        """Build from one raw ``conversation.json`` record, copying only safe fields."""
        payload = raw.get("input", {})
        metadata = payload.get("metadata", {})
        evaluation = raw.get("evaluation", {})

        conversation_number = str(raw.get("conversation_number", raw.get("id", "?")))
        messages: list[Message] = [
            {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
            for m in payload.get("conversation", [])
        ]
        turn_evaluations = evaluation.get("turn_evaluations", [])
        # Copy ONLY the label from each turn evaluation. The judge reasoning that
        # sits alongside it is forbidden information and is deliberately dropped.
        turn_labels = [str(te.get("label", "")) for te in turn_evaluations]
        turn_ids = [str(t) for t in metadata.get("turn_ids", [])]
        num_turns = int(metadata.get("num_turns", len(turn_labels)))

        return cls(
            conversation_id=conversation_number,
            source_conversation_id=str(raw.get("id", conversation_number)),
            dataset=str(metadata.get("dataset", "")),
            rag=str(metadata.get("rag", "")),
            messages=messages,
            turn_labels=turn_labels,
            turn_ids=turn_ids,
            num_turns=num_turns,
        )

    # -- turn accessors -------------------------------------------------------
    def user_message(self, turn_index: int) -> Message:
        return self.messages[2 * turn_index]

    def assistant_message(self, turn_index: int) -> Message:
        return self.messages[2 * turn_index + 1]

    def turn_id(self, turn_index: int) -> str:
        if turn_index < len(self.turn_ids):
            return self.turn_ids[turn_index]
        return str(turn_index + 1)

    def iter_failed_turns(self) -> Iterator[FailedTurn]:
        """Yield every turn whose correctness label is ``incorrect``.

        Correct (or unlabeled) turns are skipped: the *incorrect* label is used
        solely to decide which turns require a diagnostic follow-up.
        """
        for turn_index, label in enumerate(self.turn_labels):
            if label != INCORRECT_LABEL:
                continue
            # Guard against malformed records where messages and labels disagree.
            if 2 * turn_index + 1 >= len(self.messages):
                raise ValueError(
                    f"[conv={self.conversation_id} turn={self.turn_id(turn_index)}] "
                    "label present but conversation has no matching assistant turn."
                )
            yield FailedTurn(
                conversation_id=self.conversation_id,
                source_conversation_id=self.source_conversation_id,
                dataset=self.dataset,
                rag=self.rag,
                turn_index=turn_index,
                turn_id=self.turn_id(turn_index),
                label=label,
            )


def load_conversation_file(path: str | Path) -> list[AnnotatedConversation]:
    """Load one ``conversation.json`` file into permitted-only conversations."""
    file_path = Path(path)
    with file_path.open(encoding="utf-8") as handle:
        raw_records = json.load(handle)
    return [AnnotatedConversation.from_raw(record) for record in raw_records]


def load_failed_conversation_turns(
    path: str | Path,
) -> list[tuple[AnnotatedConversation, FailedTurn]]:
    """Load a file and return every ``(conversation, failed_turn)`` pair.

    Convenience for callers that want the flat list of failed turns without
    re-implementing the correct/incorrect selection.
    """
    pairs: list[tuple[AnnotatedConversation, FailedTurn]] = []
    for conversation in load_conversation_file(path):
        for failed_turn in conversation.iter_failed_turns():
            pairs.append((conversation, failed_turn))
    return pairs


def build_dynamic_input(
    conversation: AnnotatedConversation, failed_turn: FailedTurn
) -> DynamicFollowUpInput:
    """Build the answer-conditioned envelope for one failed turn.

    ``rag_history`` runs up to and including the failed assistant answer; the
    failed question and answer are also surfaced explicitly. No gold answer,
    reasoning, or document ever enters this object.
    """
    turn_index = failed_turn.turn_index
    history = [dict(m) for m in conversation.messages[: 2 * turn_index + 2]]
    failed_question = conversation.user_message(turn_index)["content"]
    failed_answer = conversation.assistant_message(turn_index)["content"]
    return DynamicFollowUpInput(
        conversation_id=conversation.conversation_id,
        failed_turn_id=failed_turn.turn_id,
        rag_history=history,
        failed_question=failed_question,
        failed_rag_answer=failed_answer,
    )


def build_static_input(
    conversation: AnnotatedConversation, failed_turn: FailedTurn
) -> StaticFollowUpInput:
    """Build the Xie-style baseline envelope, excluding the failed RAG answer.

    ``rag_history`` runs up to and including the failed *question* only; the
    failed assistant answer is dropped so static generation cannot react to it.
    """
    turn_index = failed_turn.turn_index
    history = [dict(m) for m in conversation.messages[: 2 * turn_index + 1]]
    failed_question = conversation.user_message(turn_index)["content"]
    return StaticFollowUpInput(
        conversation_id=conversation.conversation_id,
        failed_turn_id=failed_turn.turn_id,
        rag_history=history,
        failed_question=failed_question,
    )


def dynamic_provenance() -> GenerationInputProvenance:
    """Provenance for a dynamic input: history + failed answer, nothing forbidden."""
    return GenerationInputProvenance(
        used_rag_history=True,
        used_failed_rag_answer=True,
    )

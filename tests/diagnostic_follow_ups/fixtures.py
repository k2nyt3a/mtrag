"""Deterministic fixtures for diagnostic-follow-up tests.

The synthetic conversation mirrors the real ``conversation.json`` schema and
deliberately embeds distinctive *forbidden* markers (gold answers, judge
reasoning, retrieved documents) so tests can prove those never reach a prompt:

* ``GOLDSECRET_*``      -> ground_truth_answers (gold answers)
* ``REASONINGSECRET_*`` -> turn_evaluations[i].reasoning (judge reasoning)
* ``RETRIEVEDSECRET_*`` -> retrieved / final-context documents

Turn labels are engineered so that turn 1 is correct (must be skipped) and turns
2-6 are incorrect, each shaped to trigger a different question strategy.
"""

from __future__ import annotations

from question_generation.diagnostic_follow_ups.conversation_loader import (
    AnnotatedConversation,
    build_dynamic_input,
)
from question_generation.diagnostic_follow_ups.schemas import DynamicFollowUpInput

# (user question, RAG answer, correctness label)
_TURNS: list[tuple[str, str, str]] = [
    (
        "Tell me about the U.S. Civil War.",
        "The U.S. Civil War was fought from 1861 to 1865.",
        "correct",
    ),
    (
        "What made that war terrible?",
        "World War I was terrible due to trench warfare and chemical weapons.",
        "incorrect",
    ),
    (
        "Who led the Union army?",
        "The Union army was led by Stafford Cripps.",
        "incorrect",
    ),
    (
        "What about casualties?",
        "",
        "incorrect",
    ),
    (
        "Compare the North and South economies and populations.",
        "The North had a larger economy.",
        "incorrect",
    ),
    (
        "Compare the causes and consequences and long-term outcomes of the war.",
        (
            "The war had many causes including economic differences regional "
            "tensions political disagreements social factors and various "
            "circumstances that developed gradually over several decades of "
            "national conflict."
        ),
        "incorrect",
    ),
]


def raw_conversation() -> dict:
    """A single raw ``conversation.json`` record with all forbidden fields present."""
    conversation: list[dict[str, str]] = []
    ground_truth_answers: list[str] = []
    turn_evaluations: list[dict] = []
    retrieved_chunks_by_turn: list[dict] = []
    turn_ids: list[str] = []

    for index, (question, answer, label) in enumerate(_TURNS):
        turn_number = index + 1
        conversation.append({"role": "user", "content": question})
        conversation.append({"role": "assistant", "content": answer})
        ground_truth_answers.append(
            f"GOLDSECRET_{turn_number}: the real reference answer for turn {turn_number}."
        )
        turn_evaluations.append(
            {
                "turn": turn_number,
                "label": label,
                "reasoning": f"REASONINGSECRET_{turn_number}: why the judge decided this.",
            }
        )
        document = {
            "doc_id": f"doc-{turn_number}",
            "rank": 1,
            "score": None,
            "score_type": "similarity",
            "text": f"RETRIEVEDSECRET_{turn_number}: retrieved passage text.",
            "title": "Source",
        }
        retrieved_chunks_by_turn.append(
            {
                "query": question,
                "k": 5,
                "retrieved_documents": [document],
                "final_context_documents": [document],
                "rewritten_query": None,
            }
        )
        turn_ids.append(str(turn_number))

    return {
        "conversation_number": "001",
        "id": "hash001",
        "input": {
            "conversation": conversation,
            "ground_truth_answers": ground_truth_answers,
            "metadata": {
                "dataset": "testds",
                "rag": "testrag",
                "num_turns": len(_TURNS),
                "turn_ids": turn_ids,
                "retrieved_chunks_by_turn": retrieved_chunks_by_turn,
            },
        },
        "evaluation": {
            "conversation_label": "incorrect",
            "turn_evaluations": turn_evaluations,
        },
        "judge": {"model": "test-judge"},
    }


# Distinctive markers that must never appear in any generator-facing prompt.
FORBIDDEN_MARKERS: tuple[str, ...] = (
    "GOLDSECRET",
    "REASONINGSECRET",
    "RETRIEVEDSECRET",
)

# 1-based turn numbers and the strategy each incorrect turn is engineered to hint.
CONTRASTIVE_TURN = 2
EVIDENCE_TURN = 3
ANSWERABILITY_TURN = 4
COMPLETION_TURN = 5
ATOMIC_TURN = 6


def sample_conversation() -> AnnotatedConversation:
    """The fixture as a permitted-only :class:`AnnotatedConversation`."""
    return AnnotatedConversation.from_raw(raw_conversation())


def dynamic_input_for_turn(turn_number: int) -> DynamicFollowUpInput:
    """Build the dynamic input for one 1-based turn number of the fixture."""
    conversation = sample_conversation()
    turn_index = turn_number - 1
    failed_turn = next(
        ft for ft in conversation.iter_failed_turns() if ft.turn_index == turn_index
    )
    return build_dynamic_input(conversation, failed_turn)


def make_dynamic_input(
    prior_history: list[dict[str, str]],
    question: str,
    answer: str,
    *,
    conversation_id: str = "001",
    failed_turn_id: str = "1",
) -> DynamicFollowUpInput:
    """Directly construct a dynamic input from parts (for prompt-content tests)."""
    history = [dict(m) for m in prior_history]
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return DynamicFollowUpInput(
        conversation_id=conversation_id,
        failed_turn_id=failed_turn_id,
        rag_history=history,
        failed_question=question,
        failed_rag_answer=answer,
    )

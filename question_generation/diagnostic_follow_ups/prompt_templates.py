"""Centralized prompt templates for diagnostic follow-up generation.

All prompt strings live here so there is a single source of truth and no
duplicated instructions. Both the dynamic and static builders render the
permitted conversation history the same way; they differ only in whether the
failed RAG answer is shown (dynamic: yes, static: never).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .question_strategies import STRATEGY_GUIDANCE, QuestionStrategy
from .schemas import DynamicFollowUpInput, Message, StaticFollowUpInput


@dataclass(frozen=True)
class Prompt:
    """A system/user prompt pair for a chat-style model client.

    ``context`` carries the structured, permitted-only input envelope
    (:class:`~.schemas.DynamicFollowUpInput` or
    :class:`~.schemas.StaticFollowUpInput`). Real LLM clients ignore it and read
    only ``system``/``user``; deterministic offline and scripted test clients may
    read it to produce a response without a network call. Because the envelope
    contains no forbidden information, exposing it here is boundary-safe.
    """

    system: str
    user: str
    context: Any = field(default=None, compare=False)

    def as_single_string(self) -> str:
        """Flatten to one string for clients without a separate system slot."""
        return f"{self.system}\n\n{self.user}"


# The strategy names, listed for the model. These are QUESTION strategies, never
# failure categories.
_STRATEGY_MENU = "\n".join(
    f"- {strategy.value}: {STRATEGY_GUIDANCE[strategy]}" for strategy in QuestionStrategy
)

# Core generation instructions (the eleven research requirements). Kept verbatim
# in one place; referenced by both the first attempt and every retry.
DYNAMIC_SYSTEM_INSTRUCTIONS = f"""\
You generate exactly one diagnostic follow-up question after an incorrect RAG
response. The follow-up will be re-asked to the same RAG system so a separate
component can gather evidence about why the turn failed.

You may use ONLY:
- the permitted RAG conversation history shown below;
- the failed user question;
- the failed RAG answer.

You must NOT use or assume any correct answer, reference answer, judge
explanation, retrieved documents, or failure category. You do not know the
correct answer and must not guess or reveal it.

Choose one question strategy from this menu and generate the follow-up:
{_STRATEGY_MENU}

Requirements:
1. Preserve the original information need.
2. React to a specific part or behavior of the failed RAG answer.
3. Resolve references (pronouns, "that war", "they") using ONLY the permitted
   conversation history.
4. Do not provide or suggest the correct answer.
5. Do not use gold-answer information.
6. Do not introduce unrelated facts, entities, or topics.
7. Do not mention or predict a RAG failure category.
8. Do not assume that the requested evidence exists.
9. Allow the RAG to state that the information is unavailable.
10. Change only one main aspect of the original question.
11. Generate exactly one natural follow-up question.

If the intended entity or correction cannot be established from the permitted
inputs, do not invent it; fall back to a verification question.

Respond with a single JSON object and nothing else, using exactly these keys:
{{
  "strategy": "<one of the strategy names above>",
  "target_source": "<'failed_answer_span' if you target specific text in the answer, or 'response_behavior' if you target how it responded (empty, echo, refusal, omission)>",
  "target_from_failed_answer": "<the specific span/claim you are reacting to, or a short behavior tag like 'empty_or_echo_response'>",
  "answer_conditioning_explanation": "<why this follow-up reacts to the failed answer; do NOT name a failure category>",
  "follow_up_question": "<one natural follow-up question ending in '?'>",
  "introduced_new_information": <true or false: did you add any fact not present in the permitted inputs?>
}}"""

STATIC_SYSTEM_INSTRUCTIONS = """\
You generate exactly one static follow-up question after a RAG turn, using ONLY
the conversation history and the user's most recent question. This is a baseline
that must NOT see or react to the RAG's answer.

You may use ONLY:
- the conversation history shown below;
- the user's most recent question.

You must NOT use any RAG answer, correct/reference answer, judge explanation,
retrieved documents, or failure category.

Requirements:
1. Preserve the original information need.
2. Resolve references using ONLY the conversation history.
3. Do not provide or suggest any answer.
4. Do not introduce unrelated facts, entities, or topics.
5. Generate exactly one natural follow-up question ending in '?'.

Respond with a single JSON object and nothing else, using exactly these keys:
{
  "follow_up_question": "<one natural follow-up question ending in '?'>"
}"""


def _render_history(messages: list[Message]) -> str:
    """Render conversation messages as a readable, role-labelled transcript."""
    if not messages:
        return "(no prior conversation)"
    lines = []
    for message in messages:
        role = message.get("role", "?").upper()
        content = message.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_dynamic_question_prompt(
    failed_input: DynamicFollowUpInput,
    *,
    strategy_hint: QuestionStrategy | None = None,
    retry_feedback: str | None = None,
) -> Prompt:
    """Assemble the dynamic (answer-conditioned) prompt for one failed turn.

    The failed RAG answer is shown explicitly so the model can react to it.
    ``strategy_hint`` (if given) is offered as a non-binding suggestion; the model
    still selects the final strategy. ``retry_feedback`` is appended verbatim when
    a previous attempt failed validation.
    """
    prior = _render_history(failed_input.prior_history())
    user_sections = [
        f"Conversation id: {failed_input.conversation_id}",
        f"Failed turn id: {failed_input.failed_turn_id}",
        "",
        "Permitted conversation history (before the failed turn):",
        prior,
        "",
        "Failed user question:",
        failed_input.failed_question,
        "",
        "Failed RAG answer (react to something specific in THIS answer):",
        failed_input.failed_rag_answer,
    ]
    if strategy_hint is not None:
        user_sections += [
            "",
            f"Suggested strategy (non-binding hint): {strategy_hint.value}",
        ]
    if retry_feedback:
        user_sections += [
            "",
            "Your previous attempt was rejected. Fix these issues and try again:",
            retry_feedback,
        ]
    return Prompt(
        system=DYNAMIC_SYSTEM_INSTRUCTIONS,
        user="\n".join(user_sections),
        context=failed_input,
    )


def build_static_question_prompt(failed_input: StaticFollowUpInput) -> Prompt:
    """Assemble the static (Xie-style) baseline prompt.

    Only the history through the failed question is shown; the failed RAG answer
    is never included.
    """
    history = _render_history(failed_input.rag_history)
    user = "\n".join(
        [
            f"Conversation id: {failed_input.conversation_id}",
            f"Failed turn id: {failed_input.failed_turn_id}",
            "",
            "Conversation history (through the user's most recent question):",
            history,
            "",
            "Most recent user question:",
            failed_input.failed_question,
        ]
    )
    return Prompt(system=STATIC_SYSTEM_INSTRUCTIONS, user=user, context=failed_input)

"""Question-generation strategies for diagnostic follow-ups.

IMPORTANT research constraint
-----------------------------
These are *question strategies* (what shape of follow-up to ask), **not**
failure-category predictions. Nothing in this module infers, names, or ranks
the six RAG failure categories (Knowledge Boundary, Chunking, Retrieval,
Context Selection, Grounding, Response Coverage). Selecting a strategy only
decides how to phrase the follow-up; the downstream attribution component is
what later interprets the follow-up's RAG answer.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime import cycle with schemas
    from .schemas import DynamicFollowUpInput


class QuestionStrategy(str, Enum):
    """The allowed diagnostic follow-up question strategies.

    ``str`` mix-in so values serialize directly to JSON (e.g. ``"evidence_verification"``).
    """

    CONTRASTIVE_REFORMULATION = "contrastive_reformulation"
    EVIDENCE_VERIFICATION = "evidence_verification"
    RETRIEVAL_ORIENTED_REFORMULATION = "retrieval_oriented_reformulation"
    COMPLETION_PROBE = "completion_probe"
    ANSWERABILITY_PROBE = "answerability_probe"
    ATOMIC_DECOMPOSITION = "atomic_decomposition"
    GENERAL_VERIFICATION = "general_verification"

    @classmethod
    def from_text(cls, value: str) -> "QuestionStrategy":
        """Parse a strategy name, tolerating surrounding whitespace/case.

        Falls back to :attr:`GENERAL_VERIFICATION` for anything unrecognised so
        a slightly malformed model response never crashes the pipeline.
        """
        normalized = (value or "").strip().lower()
        for strategy in cls:
            if strategy.value == normalized:
                return strategy
        return cls.GENERAL_VERIFICATION


# Human-readable guidance injected into the generation prompt. Keeping the text
# here (rather than inline in the prompt) keeps every strategy description in one
# place and avoids duplicated prompt strings.
STRATEGY_GUIDANCE: dict[QuestionStrategy, str] = {
    QuestionStrategy.CONTRASTIVE_REFORMULATION: (
        "Use when the failed answer resolves an entity, topic, event, or "
        "pronoun incorrectly. Restate the intended reference using ONLY the "
        "conversation history, contrast it with the mistaken reference from the "
        "failed answer, and re-ask the original information need."
    ),
    QuestionStrategy.EVIDENCE_VERIFICATION: (
        "Use when the failed answer makes a specific claim (person, date, "
        "number, place, definition, cause, relationship, or list). Ask the RAG "
        "to verify that exact claim against the available sources. You may quote "
        "the claim but must not supply the correct value."
    ),
    QuestionStrategy.RETRIEVAL_ORIENTED_REFORMULATION: (
        "Use when the original question is short, elliptical, or context-"
        "dependent and the answer appears to be about the wrong topic. Re-ask the "
        "same information need as a self-contained question that makes the "
        "intended topic (from the history) explicit, so the unchanged RAG can "
        "retrieve it. Do NOT call the retriever and do NOT add keywords taken "
        "from the gold answer or documents."
    ),
    QuestionStrategy.COMPLETION_PROBE: (
        "Use only when an omitted requirement is explicitly visible in the "
        "original question (e.g. it asked about price AND performance but the "
        "answer covered only price). Ask for the missing, explicitly requested "
        "part."
    ),
    QuestionStrategy.ANSWERABILITY_PROBE: (
        "Use when the answer is empty, repeats the user, asks the question back, "
        "refuses irrelevantly, or is otherwise unusable. Make the entity and "
        "request explicit from the history and ask whether it can be answered "
        "from the sources, allowing the RAG to abstain."
    ),
    QuestionStrategy.ATOMIC_DECOMPOSITION: (
        "Use for an explicitly complex, comparative, multi-part, or multi-hop "
        "request. Ask the RAG to answer in parts while preserving the original "
        "information need; introduce no new topics or difficulty."
    ),
    QuestionStrategy.GENERAL_VERIFICATION: (
        "Use when no more specific strategy applies safely. Ask the RAG to "
        "reconsider and verify its previous answer using only the available "
        "sources, allowing it to say the information is unavailable."
    ),
}


# RePAIR-inspired adaptation: RePAIR maps a flawed RAG state directly to a
# corrective *action*; here we map the flawed question/answer directly to a
# diagnostic *question strategy* (no failure category is predicted first). This
# adapts only RePAIR's response-to-action principle, not its executor or its
# training pipeline. ``REFINEDOC`` is intentionally unmapped: this generator can
# neither inspect nor modify retrieved documents.
REPAIR_OPERATION_MAPPING: dict[str, QuestionStrategy | None] = {
    "REWRITE": QuestionStrategy.CONTRASTIVE_REFORMULATION,
    "DECOMPOSE": QuestionStrategy.ATOMIC_DECOMPOSITION,
    "RETRIEVAL": QuestionStrategy.RETRIEVAL_ORIENTED_REFORMULATION,
    # GENERATEANSWER covers both verifying a produced claim and probing an
    # explicitly omitted part of the request.
    "GENERATEANSWER": QuestionStrategy.EVIDENCE_VERIFICATION,
    "REFINEDOC": None,
}


# --- Deterministic strategy hint --------------------------------------------
# These deictic tokens signal that the failed question depends on an entity
# established earlier in the conversation (a common trigger for contrastive
# reformulation when the answer drifts to the wrong entity).
_REFERENCE_TOKENS = frozenset(
    {"it", "its", "they", "them", "their", "that", "this", "those",
     "these", "there", "he", "she", "him", "her", "his"}
)

# Surface cues that the original request explicitly asked for more than one thing.
_MULTIPART_CUES = (
    " and ", " both ", " compare", "comparison", " versus ", " vs ", " vs. ",
    "difference between", " as well as ", " first ", " then ", " finally ",
)

# Surface cues that the answer did not really answer the question.
_UNUSABLE_ANSWER_CUES = (
    "i cannot", "i can't", "i'm not able", "i am not able", "cannot answer",
    "no information", "not enough information", "i don't know", "i do not know",
    "as an ai", "could you clarify", "what do you mean",
)


def _content_words(text: str) -> set[str]:
    """Lowercased alphanumeric tokens of length > 2 (a cheap stopword filter)."""
    return {tok for tok in _tokenize(text) if len(tok) > 2}


def _tokenize(text: str) -> list[str]:
    token: list[str] = []
    tokens: list[str] = []
    for char in (text or "").lower():
        if char.isalnum():
            token.append(char)
        elif token:
            tokens.append("".join(token))
            token = []
    if token:
        tokens.append("".join(token))
    return tokens


def _answer_looks_unusable(failed_question: str, failed_answer: str) -> bool:
    answer = (failed_answer or "").strip()
    if not answer:
        return True
    lowered = answer.lower()
    if any(cue in lowered for cue in _UNUSABLE_ANSWER_CUES):
        return True
    # Echo: the answer is essentially the user's own question repeated back.
    question = (failed_question or "").strip()
    if question and (answer == question or answer.lower() == question.lower()):
        return True
    if question:
        q_words = _content_words(question)
        a_words = _content_words(answer)
        if q_words and a_words:
            overlap = len(q_words & a_words) / len(a_words)
            if overlap >= 0.8 and len(a_words) <= len(q_words) + 2:
                return True
    # Answer that only asks a question back and adds nothing.
    if answer.endswith("?") and len(_content_words(answer)) <= 4:
        return True
    return False


def _mentions_specific_claim(failed_answer: str) -> bool:
    """True if the answer contains a proper noun, number, or date-like token."""
    for raw in (failed_answer or "").split():
        stripped = raw.strip(".,;:()[]\"'")
        if not stripped:
            continue
        if any(ch.isdigit() for ch in stripped):
            return True
        # Capitalised token that is not the leading word "The"/"They" etc.
        if stripped[0].isupper() and stripped.lower() not in {
            "the", "they", "there", "this", "that", "these", "those", "it", "yes", "no",
        }:
            return True
    return False


def suggest_strategy(failed_input: "DynamicFollowUpInput") -> QuestionStrategy:
    """Return a deterministic *hint* for which question strategy fits best.

    This is a lightweight, side-effect-free heuristic used two ways:

    * as an optional soft hint inside the dynamic prompt, and
    * as the strategy chosen by the offline (no-API) model client and as a
      fallback when a model response omits a usable strategy.

    It reasons only about the *shape* of the failed question and answer — never
    about correctness, gold content, or a failure category — so it respects the
    same information boundary as everything else in this package.
    """
    question = failed_input.failed_question or ""
    answer = failed_input.failed_rag_answer or ""
    q_lower = f" {question.lower()} "

    # 1. Unusable / empty / echoed answer -> make the request explicit and ask
    #    whether it is answerable at all.
    if _answer_looks_unusable(question, answer):
        return QuestionStrategy.ANSWERABILITY_PROBE

    q_tokens = set(_tokenize(question))
    has_reference = bool(q_tokens & _REFERENCE_TOKENS)
    is_multipart = any(cue in q_lower for cue in _MULTIPART_CUES)

    # 2. Multi-part original question. If the answer clearly covered only one
    #    part (short relative to a two-part ask) prefer a completion probe;
    #    otherwise decompose the complex request.
    if is_multipart:
        if len(_content_words(answer)) <= 12:
            return QuestionStrategy.COMPLETION_PROBE
        return QuestionStrategy.ATOMIC_DECOMPOSITION

    # 3. Deictic/pronoun question whose answer drifted to a different entity.
    if has_reference and _answer_drifts_from_history(failed_input):
        return QuestionStrategy.CONTRASTIVE_REFORMULATION

    # 4. Short / elliptical / context-dependent question -> re-ask it as a
    #    self-contained question so the unchanged RAG can retrieve the right topic
    #    (RePAIR's RETRIEVAL action, adapted to a user-visible reformulation).
    if _is_short_contextual_query(failed_input):
        return QuestionStrategy.RETRIEVAL_ORIENTED_REFORMULATION

    # 5. A concrete factual claim to verify.
    if _mentions_specific_claim(answer):
        return QuestionStrategy.EVIDENCE_VERIFICATION

    # 6. Fall back to a safe general verification.
    return QuestionStrategy.GENERAL_VERIFICATION


# Interrogative openers that mark a question as self-contained rather than a bare
# elliptical fragment.
_QUESTION_WORDS = frozenset(
    {"who", "what", "when", "where", "why", "how", "which", "whom", "whose",
     "does", "do", "did", "is", "are", "was", "were", "can", "could", "would",
     "should", "will", "has", "have", "had"}
)


def _is_short_contextual_query(failed_input: "DynamicFollowUpInput") -> bool:
    """Heuristic: the failed question leans on prior turns to be understood.

    True when there is prior history AND the question is either very short
    (<= 3 content words) or a bare phrase with no interrogative opener (an
    elliptical fragment such as "Vitamin C research"). Such queries are the ones
    a retrieval-oriented reformulation helps most.
    """
    if not failed_input.prior_history():
        return False
    question = (failed_input.failed_question or "").strip()
    if not question:
        return False
    tokens = _tokenize(question)
    starts_with_interrogative = bool(tokens) and tokens[0] in _QUESTION_WORDS
    is_bare_phrase = not question.endswith("?") and not starts_with_interrogative
    is_short = len(_content_words(question)) <= 3
    return is_short or is_bare_phrase


def _answer_drifts_from_history(failed_input: "DynamicFollowUpInput") -> bool:
    """Heuristic: the answer introduces salient nouns absent from the history.

    Used only to distinguish "the answer talks about the wrong entity" from a
    normal on-topic answer. Reads only permitted inputs (history + failed
    answer).
    """
    history_words: set[str] = set()
    # Everything before the failed turn is the safe reference for the intended
    # entity; the failed question tokens count as history context too.
    for message in failed_input.prior_history():
        history_words |= _content_words(message.get("content", ""))
    history_words |= _content_words(failed_input.failed_question)

    answer_words = _content_words(failed_input.failed_rag_answer)
    if not answer_words:
        return False
    novel = answer_words - history_words
    # Many brand-new content words in a short pronoun-driven answer suggests the
    # RAG resolved the reference to a different subject.
    return len(novel) >= max(2, len(answer_words) // 2)

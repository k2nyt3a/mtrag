"""Deterministic validation of generated diagnostic follow-up questions.

Validation is intentionally rule-based and API-free: it protects the experiment
without a paid model call and is fully reproducible in tests. An optional
LLM-based validator could be layered on later, but the deterministic rules below
are the authoritative gate.

The validator only ever sees permitted inputs plus the generated question, so it
cannot itself leak gold information into retry feedback.
"""

from __future__ import annotations

from .question_strategies import _content_words, _tokenize
from .schemas import DynamicFollowUpInput, GeneratedQuestion, ValidationResult

# The six RAG failure categories (and generic diagnostic wording). A valid
# follow-up must never name these — that would leak the downstream task into the
# question itself.
FAILURE_CATEGORY_TERMS: tuple[str, ...] = (
    "knowledge boundary",
    "chunking",
    "context selection",
    "response coverage",
    "grounding failure",
    "retrieval failure",
    "failure category",
    "failure mode",
    "failure label",
)

# Words that are part of the follow-up *scaffolding* rather than new topical
# content; excluded when measuring how much novel material a question introduces.
# This includes the structural verbs/adverbs a follow-up naturally uses ("also",
# "address", "remaining", ...) so that only genuinely new subject matter counts
# toward an unrelated-topic shift.
GENERIC_FOLLOWUP_WORDS: frozenset[str] = frozenset(
    {
        "you", "your", "please", "could", "would", "can", "the", "and", "that",
        "this", "these", "those", "with", "from", "about", "into", "using",
        "based", "available", "sources", "source", "information", "unavailable",
        "previous", "answer", "answers", "answered", "answering", "response",
        "responses", "question", "questions", "verify", "verifying", "confirm",
        "confirmed", "reconsider", "claim", "claims", "identify", "identified",
        "stated", "state", "says", "said", "mentioned", "named", "listed",
        "compare", "compared", "comparison", "part", "parts", "specifically",
        "mean", "meant", "support", "supported", "supports", "unsupported",
        "asked", "asking", "was", "were", "are", "is", "not", "but", "for",
        "what", "which", "who", "whom", "where", "when", "why", "how", "does",
        "did", "do", "them", "they", "their", "there", "it", "its", "he", "she",
        "his", "her", "discussed", "discussing", "discuss", "previously",
        "earlier", "correct", "reference", "also", "address", "addressed",
        "covered", "cover", "covering", "only", "remaining", "remainder", "rest",
        "provide", "provided", "explain", "explained", "clarify", "clarified",
        "detail", "details", "more", "further", "additional", "aspect",
        "aspects", "following", "focus", "focused", "resolve", "resolved",
        "resolving", "fully", "separate", "separately", "give", "given",
        "according", "whether", "still", "again", "missing", "omitted", "other",
        "another", "regarding", "instead",
    }
)

# Explicit "I am reacting to your answer" phrasings. Their presence is treated as
# (weak) answer-conditioning even when the question shares few answer tokens.
_REACTIVE_PHRASES: tuple[str, ...] = (
    "you identified", "you said", "you stated", "you named", "you listed",
    "you mentioned", "you compared", "you answered", "you told", "your previous",
    "your answer", "your response", "you claimed", "you gave", "you provided",
    "previous answer", "you resolved", "you referred", "you described",
)

# Phrasings that let the RAG abstain when evidence may not exist.
_ABSTENTION_PHRASES: tuple[str, ...] = (
    "unavailable", "if it cannot", "if it can", "if unsupported", "or say",
    "state that", "if there is no", "if the information", "cannot be answered",
    "if it is not", "if that is not",
)


def _numbers(text: str) -> set[str]:
    """Digit-bearing tokens found in ``text`` (e.g. dates, counts, amounts).

    A follow-up that introduces a *new* number not present in the permitted
    inputs is likely asserting a fabricated fact or a correct answer, so this is
    used as a robust, low-false-positive leakage check. (Proper-noun leakage of a
    single new entity is not reliably detectable without the gold answer; see the
    README's limitations note.)

    Uses the same alphanumeric tokenization as the rest of the package so that,
    e.g., "1040NR-EZ" and "1040NR" compare consistently and a number already in
    the inputs is never mistaken for a new one.
    """
    return {token for token in _tokenize(text) if any(ch.isdigit() for ch in token)}


class QuestionValidator:
    """Validate a generated question against the fourteen research rules.

    ``min_words``/``max_words`` bound grammaticality; ``novelty_threshold`` is the
    share of non-scaffolding content words that may be new before the question is
    judged to have shifted topic.
    """

    def __init__(
        self,
        *,
        min_words: int = 3,
        max_words: int = 60,
        novelty_threshold: float = 0.5,
    ) -> None:
        self.min_words = min_words
        self.max_words = max_words
        self.novelty_threshold = novelty_threshold

    def validate_generated_question(
        self, failed_input: DynamicFollowUpInput, generated: GeneratedQuestion
    ) -> ValidationResult:
        """Return a full :class:`ValidationResult` for one generated question."""
        question = (generated.follow_up_question or "").strip()

        permitted_words = self._permitted_words(failed_input)
        permitted_numbers = self._permitted_numbers(failed_input)
        answer_words = _content_words(failed_input.failed_rag_answer)
        question_words = _content_words(question)
        question_lower = question.lower()

        is_single_question = question.count("?") == 1
        same_information_need = self._shares_information_need(
            failed_input, question_words
        )
        is_answer_conditioned = self._is_answer_conditioned(
            question_words, answer_words, failed_input, question_lower
        )
        mentions_failure_category = any(
            term in question_lower for term in FAILURE_CATEGORY_TERMS
        )
        novel_numbers = _numbers(question) - permitted_numbers
        provides_correct_answer = bool(novel_numbers)
        introduces_unrelated_topic = self._introduces_unrelated_topic(
            question_words, permitted_words
        )
        contains_forbidden_information = (
            mentions_failure_category or provides_correct_answer
        )
        is_grammatical = self._is_grammatical(question)
        allows_abstention = any(
            phrase in question_lower for phrase in _ABSTENTION_PHRASES
        )

        reasons: list[str] = []
        if not is_single_question:
            reasons.append("not_single_question")
        if not same_information_need:
            reasons.append("information_need_changed")
        if not is_answer_conditioned:
            reasons.append("not_answer_conditioned")
        if mentions_failure_category:
            reasons.append("mentions_failure_category")
        if provides_correct_answer:
            reasons.append("introduces_new_facts_or_answer")
        if introduces_unrelated_topic:
            reasons.append("unrelated_topic_shift")
        if not is_grammatical:
            reasons.append("not_grammatical")

        # allows_abstention is recorded for auditing but is a soft signal: a
        # re-asked information need (e.g. contrastive reformulation) is valid
        # without an explicit abstention clause, so it is not part of is_valid.
        is_valid = not reasons

        return ValidationResult(
            is_single_question=is_single_question,
            same_information_need=same_information_need,
            is_answer_conditioned=is_answer_conditioned,
            contains_forbidden_information=contains_forbidden_information,
            introduces_unrelated_topic=introduces_unrelated_topic,
            provides_correct_answer=provides_correct_answer,
            mentions_failure_category=mentions_failure_category,
            is_grammatical=is_grammatical,
            allows_abstention=allows_abstention,
            is_valid=is_valid,
            reasons=reasons,
        )

    # -- individual rules -----------------------------------------------------
    def _permitted_words(self, failed_input: DynamicFollowUpInput) -> set[str]:
        words: set[str] = set()
        for message in failed_input.rag_history:
            words |= _content_words(message.get("content", ""))
        words |= _content_words(failed_input.failed_question)
        words |= _content_words(failed_input.failed_rag_answer)
        return words

    def _permitted_numbers(self, failed_input: DynamicFollowUpInput) -> set[str]:
        numbers: set[str] = set()
        for message in failed_input.rag_history:
            numbers |= _numbers(message.get("content", ""))
        numbers |= _numbers(failed_input.failed_question)
        numbers |= _numbers(failed_input.failed_rag_answer)
        return numbers

    def _shares_information_need(
        self, failed_input: DynamicFollowUpInput, question_words: set[str]
    ) -> bool:
        """The follow-up must stay tied to the original need's vocabulary.

        Satisfied by sharing a content word with the failed question, or (for
        reference-resolving strategies) with the prior conversation history.
        """
        if question_words & _content_words(failed_input.failed_question):
            return True
        history_words: set[str] = set()
        for message in failed_input.prior_history():
            history_words |= _content_words(message.get("content", ""))
        return len(question_words & history_words) >= 2

    def _is_answer_conditioned(
        self,
        question_words: set[str],
        answer_words: set[str],
        failed_input: DynamicFollowUpInput,
        question_lower: str,
    ) -> bool:
        """Meaningfully react to the failed answer.

        Strong signal: the question echoes a salient token from the failed answer
        that was not already in the failed question. Weak signal: an explicit
        "reacting to your previous answer" phrase. Either suffices.
        """
        question_only_answer_terms = (
            question_words & answer_words
        ) - _content_words(failed_input.failed_question)
        strong = bool(question_only_answer_terms)
        weak = any(phrase in question_lower for phrase in _REACTIVE_PHRASES)
        return strong or weak

    def _introduces_unrelated_topic(
        self, question_words: set[str], permitted_words: set[str]
    ) -> bool:
        topical = {
            word for word in question_words if word not in GENERIC_FOLLOWUP_WORDS
        }
        if not topical:
            return False
        novel = {word for word in topical if word not in permitted_words}
        if len(novel) < 3:
            return False
        return (len(novel) / len(topical)) > self.novelty_threshold

    def _is_grammatical(self, question: str) -> bool:
        if not question or not question.endswith("?"):
            return False
        word_count = len(_tokenize(question))
        if word_count < self.min_words or word_count > self.max_words:
            return False
        return question[0].isupper() or question[0].isdigit()


# -- retry feedback ----------------------------------------------------------
# Human-readable guidance per failed rule. Deliberately contains no gold-answer
# information — only instructions about how to fix the question's form.
_FEEDBACK_BY_REASON: dict[str, str] = {
    "not_single_question": "Ask exactly one question (use a single '?').",
    "information_need_changed": (
        "Keep the original information need: reuse the question's key terms "
        "instead of drifting to a different topic."
    ),
    "not_answer_conditioned": (
        "React to a specific span or behavior in the failed RAG answer (quote or "
        "name the exact claim you are questioning)."
    ),
    "mentions_failure_category": (
        "Do not mention or predict any failure category or diagnostic label."
    ),
    "introduces_new_facts_or_answer": (
        "Do not introduce any new name, number, or fact that is not already in "
        "the conversation or the failed answer, and never supply a correct answer."
    ),
    "unrelated_topic_shift": (
        "Stay on the original topic; remove unrelated entities or subjects."
    ),
    "not_grammatical": (
        "Write one clear, grammatical question that ends with '?'."
    ),
}


def build_retry_feedback(
    validation: ValidationResult, previous_question: str
) -> str:
    """Turn failed validation rules into targeted, gold-free retry guidance."""
    items = [
        _FEEDBACK_BY_REASON[reason]
        for reason in validation.reasons
        if reason in _FEEDBACK_BY_REASON
    ]
    lines = [f"- {item}" for item in items]
    trimmed = previous_question.strip()
    if trimmed:
        lines.append(f'- Your previous question was: "{trimmed}". Revise it.')
    return "\n".join(lines)

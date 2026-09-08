"""Model-client abstraction for follow-up generation.

Three implementations are provided:

* :class:`OfflineHeuristicModelClient` — deterministic, no network, no API key.
  Produces genuinely answer-conditioned questions from the permitted inputs and
  is used for dry-run pilots and as an offline default.
* :class:`ScriptedModelClient` — returns canned responses for unit tests.
* :class:`AnthropicModelClient` — calls the Anthropic API for real runs; the SDK
  is imported lazily so this module has no hard dependency and tests never touch
  the network.

The follow-up *generation model is separate from the frozen RAG system* that
already produced ``conversation.json``; nothing here changes the RAG's model,
retriever, seeds, or top-k.
"""

from __future__ import annotations

import json
import os
from typing import Protocol, runtime_checkable

from .prompt_templates import Prompt
from .question_strategies import (
    QuestionStrategy,
    _content_words,
    _tokenize,
    suggest_strategy,
)
from .schemas import (
    DynamicFollowUpInput,
    GeneratedQuestion,
    StaticFollowUpInput,
    TargetSource,
)


@runtime_checkable
class FollowUpModelClient(Protocol):
    """Minimal interface every follow-up model client implements."""

    def generate(self, prompt: Prompt) -> str:
        """Return the model's raw response text (expected to be one JSON object)."""
        ...


# --- deterministic offline generation ---------------------------------------
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "are", "was", "were", "you", "your", "they",
        "them", "their", "that", "this", "these", "those", "with", "from",
        "have", "has", "had", "what", "which", "who", "does", "did", "can",
        "could", "would", "about", "into", "there", "here", "its", "his",
        "her", "she", "him", "not", "but", "any", "all", "how", "why", "when",
        "where", "some", "such", "than", "then", "also", "yes",
    }
)


def _sanitize(text: str, *, max_words: int = 12) -> str:
    """Trim to ``max_words`` words and strip characters that break the template.

    In particular '?' is removed so an embedded snippet cannot introduce a second
    question mark (which would break the single-question rule).
    """
    words = (text or "").split()[:max_words]
    snippet = " ".join(words).replace("?", "").replace("\n", " ").strip()
    return snippet.strip(".,;:!\"' ")


def _focus_term(text: str, fallback: str = "the topic") -> str:
    """Longest non-stopword content token in ``text`` (a crude topical keyword)."""
    candidates = [
        tok for tok in _tokenize(text) if len(tok) > 3 and tok not in _STOPWORDS
    ]
    if not candidates:
        return fallback
    return max(candidates, key=len)


def _salient_entity(text: str) -> str | None:
    """A salient proper-noun-like token in ``text``.

    Picks the longest capitalized, non-stopword token that is not sentence-initial
    (so it prefers real names over words merely capitalized at the start of a
    sentence). Falls back to the longest content word.
    """
    candidates: list[str] = []
    previous_ended_sentence = True  # the first token is sentence-initial
    for raw_word in text.split():
        stripped = raw_word.strip(".,;:()[]{}\"'?!")
        # Only count a token as ending a sentence when the character before its
        # terminal punctuation is a lowercase letter or digit ("War.", "1865.")
        # so abbreviations like "U.S." do not falsely split the sentence.
        core = raw_word.rstrip(".?!")
        ends_sentence = (
            raw_word != core and bool(core) and (core[-1].islower() or core[-1].isdigit())
        )
        if (
            len(stripped) > 2
            and stripped[0].isupper()
            and stripped.lower() not in _STOPWORDS
            and not previous_ended_sentence
        ):
            candidates.append(stripped)
        previous_ended_sentence = ends_sentence
    if candidates:
        return max(candidates, key=len)
    focus = _focus_term(text, fallback="")
    return focus or None


def build_offline_question(
    failed_input: DynamicFollowUpInput, strategy: QuestionStrategy
) -> GeneratedQuestion:
    """Construct a deterministic, answer-conditioned follow-up for one strategy.

    This is the offline stand-in for an LLM. Real experiments use
    :class:`AnthropicModelClient`; this exists so the full generate→validate→
    retry loop can run in a pilot without any paid call. Every template reacts to
    the failed answer (directly or via an explicit reference) and reuses the
    original question's wording to preserve the information need.
    """
    q_snippet = _sanitize(failed_input.failed_question)
    answer_snippet = _sanitize(failed_input.failed_rag_answer)
    focus = _focus_term(failed_input.failed_question)

    prior_text = " ".join(
        m.get("content", "") for m in failed_input.prior_history()
    )
    intended = _salient_entity(prior_text) or focus
    # A "mistaken" reference is something the answer names that the prior history
    # did not — i.e. where the RAG may have drifted.
    history_words = _content_words(prior_text) | _content_words(
        failed_input.failed_question
    )
    mistaken_entity = None
    for raw_word in failed_input.failed_rag_answer.split():
        stripped = raw_word.strip(".,;:()[]{}\"'?!")
        if (
            len(stripped) > 2
            and stripped[0].isupper()
            and stripped.lower() not in _STOPWORDS
            and stripped.lower() not in history_words
        ):
            mistaken_entity = stripped
            break
    # Sanitize so an internal '?'/newline in the extracted reference can never
    # add a second question mark to the generated follow-up.
    mistaken = _sanitize(mistaken_entity or answer_snippet, max_words=8)

    if strategy is QuestionStrategy.EVIDENCE_VERIFICATION:
        question = (
            f'You answered "{answer_snippet}". Can you verify that claim about '
            f"{focus} using the available sources, or state that it is unavailable?"
        )
        target = answer_snippet
        explanation = (
            "The follow-up quotes and asks the RAG to verify the exact claim its "
            "failed answer produced."
        )
    elif strategy is QuestionStrategy.CONTRASTIVE_REFORMULATION:
        question = (
            f"Earlier we were discussing {intended}, but your answer focused on "
            f'{mistaken}. Regarding {intended}, can you answer "{q_snippet}" '
            "using the available sources?"
        )
        target = f"the answer's focus on {mistaken}"
        explanation = (
            "The follow-up contrasts the intended subject from the history with "
            "the different subject the failed answer described."
        )
    elif strategy is QuestionStrategy.COMPLETION_PROBE:
        question = (
            f'Your previous answer covered only part of "{q_snippet}"; can you '
            "also address the remaining part using the available sources?"
        )
        target = "the part of the request the answer omitted"
        explanation = (
            "The follow-up reacts to the failed answer covering only part of the "
            "original request and asks for the omitted part."
        )
    elif strategy is QuestionStrategy.ANSWERABILITY_PROBE:
        question = (
            f'Your previous answer did not address "{q_snippet}". Can this be '
            "answered from the available sources, or is the information unavailable?"
        )
        target = "the failed answer not providing a usable response"
        explanation = (
            "The follow-up reacts to the failed answer being unusable and asks "
            "whether the question is answerable at all."
        )
    elif strategy is QuestionStrategy.RETRIEVAL_ORIENTED_REFORMULATION:
        question = (
            f"Given that your previous answer focused on {mistaken}, what do the "
            f'available sources specifically say about "{q_snippet}"?'
        )
        target = f"the answer's focus on {mistaken}"
        explanation = (
            "The follow-up re-asks the same need as a self-contained question so "
            "the RAG retrieves the intended topic instead of the one the failed "
            "answer drifted to."
        )
    elif strategy is QuestionStrategy.ATOMIC_DECOMPOSITION:
        question = (
            f'Your previous answer did not fully resolve "{q_snippet}". Could you '
            "answer it in separate parts using only the available sources?"
        )
        target = "the failed answer not resolving the multi-part request"
        explanation = (
            "The follow-up reacts to the failed answer leaving a complex request "
            "unresolved and asks for a part-by-part answer."
        )
    else:  # GENERAL_VERIFICATION
        question = (
            f"Could you reconsider your previous answer about {focus} and verify "
            "it using only the available sources, or state that the information "
            "is unavailable?"
        )
        target = "the failed answer as a whole"
        explanation = (
            "The follow-up asks the RAG to reconsider and verify the specific "
            "previous answer it produced."
        )

    return GeneratedQuestion(
        strategy=strategy,
        target_source=_target_source_for(strategy),
        target_from_failed_answer=target,
        answer_conditioning_explanation=explanation,
        follow_up_question=question,
        introduced_new_information=False,
    )


# Strategies that react to a concrete span of the answer vs. its behavior.
_SPAN_STRATEGIES = frozenset(
    {
        QuestionStrategy.EVIDENCE_VERIFICATION,
        QuestionStrategy.CONTRASTIVE_REFORMULATION,
        QuestionStrategy.RETRIEVAL_ORIENTED_REFORMULATION,
    }
)


def _target_source_for(strategy: QuestionStrategy) -> TargetSource:
    """Map a strategy to whether it targets an answer span or a response behavior."""
    if strategy in _SPAN_STRATEGIES:
        return TargetSource.FAILED_ANSWER_SPAN
    return TargetSource.RESPONSE_BEHAVIOR


def build_offline_static_question(static_input: StaticFollowUpInput) -> str:
    """Deterministic static baseline follow-up (no reference to any RAG answer)."""
    q_snippet = _sanitize(static_input.failed_question)
    focus = _focus_term(static_input.failed_question)
    return (
        f'Regarding {focus}, could you provide more detail on "{q_snippet}" '
        "using the available sources?"
    )


class OfflineHeuristicModelClient:
    """A no-API client that emits deterministic, boundary-safe follow-ups."""

    def generate(self, prompt: Prompt) -> str:
        context = prompt.context
        if isinstance(context, DynamicFollowUpInput):
            strategy = suggest_strategy(context)
            generated = build_offline_question(context, strategy)
            return json.dumps(generated.to_dict())
        if isinstance(context, StaticFollowUpInput):
            return json.dumps(
                {"follow_up_question": build_offline_static_question(context)}
            )
        raise ValueError(
            "OfflineHeuristicModelClient requires a Prompt carrying a structured "
            "context; got context of type "
            f"{type(context).__name__}."
        )


class ScriptedModelClient:
    """Return predetermined responses for tests (FIFO queue or matcher).

    ``responses`` is consumed in order. ``matcher`` (optional) takes the
    :class:`Prompt` and returns a response string; when it returns ``None`` the
    next queued response is used.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        matcher=None,
    ) -> None:
        self._responses = list(responses or [])
        self._matcher = matcher
        self.calls: list[Prompt] = []

    def generate(self, prompt: Prompt) -> str:
        self.calls.append(prompt)
        if self._matcher is not None:
            matched = self._matcher(prompt)
            if matched is not None:
                return matched
        if not self._responses:
            raise AssertionError("ScriptedModelClient ran out of responses.")
        return self._responses.pop(0)


class AnthropicModelClient:
    """Call the Anthropic API for real generation runs.

    The ``anthropic`` SDK is imported lazily so importing this module never
    requires the dependency, and unit tests never construct this client. The API
    key is read from the environment and never logged.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        api_key: str | None = None,
    ) -> None:
        self.model = model or os.environ.get(
            "DIAGNOSTIC_FOLLOWUP_MODEL", "claude-sonnet-5"
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None  # created on first use

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set; cannot run live generation. Use "
                "--dry-run for an offline pilot."
            )
        try:
            import anthropic  # noqa: PLC0415 - lazy optional dependency
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "The 'anthropic' package is required for live generation. Install "
                "it or use --dry-run."
            ) from exc
        self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def generate(self, prompt: Prompt) -> str:
        client = self._ensure_client()
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=prompt.system,
            messages=[{"role": "user", "content": prompt.user}],
        )
        return "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )

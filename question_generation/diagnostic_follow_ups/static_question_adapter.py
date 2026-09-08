"""Static (Xie-style) baseline follow-up generator.

This baseline exists so the static and dynamic methods run over the *same* failed
turns. Its defining property — the experimental control — is that it never sees
the failed RAG answer: it is built from :class:`~.schemas.StaticFollowUpInput`,
whose envelope contains only the conversation history through the failed question.

No prior static generator existed in this repository; this establishes the
baseline behavior and, once written, should not be changed so the control stays
fixed across runs.
"""

from __future__ import annotations

import json
import re

from .model_client import FollowUpModelClient
from .prompt_templates import build_static_question_prompt
from .schemas import StaticFollowUpInput, StaticGenerationRecord, static_provenance

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_static_question(raw_response: str) -> str:
    """Extract ``follow_up_question`` from a static-generation response."""
    match = _JSON_OBJECT_RE.search(raw_response or "")
    if match:
        try:
            payload = json.loads(match.group(0))
            if isinstance(payload, dict):
                question = str(payload.get("follow_up_question", "")).strip()
                if question:
                    return question
        except json.JSONDecodeError:
            pass
    # Fall back to the raw text so a non-JSON baseline response is still captured.
    return (raw_response or "").strip()


class StaticFollowUpGenerator:
    """Generate one static follow-up per failed turn, without the failed answer."""

    def __init__(self, client: FollowUpModelClient) -> None:
        self.client = client

    def generate_follow_up_for_failed_turn(
        self, static_input: StaticFollowUpInput
    ) -> StaticGenerationRecord:
        """Produce the baseline follow-up for one failed turn."""
        prompt = build_static_question_prompt(static_input)
        raw_response = self.client.generate(prompt)
        follow_up = _parse_static_question(raw_response)
        return StaticGenerationRecord(
            conversation_id=static_input.conversation_id,
            failed_turn_id=static_input.failed_turn_id,
            original_question=static_input.failed_question,
            generated_follow_up=follow_up,
            generation_input_provenance=static_provenance(),
            generation_attempts=1,
        )

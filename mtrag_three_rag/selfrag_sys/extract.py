#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Conservative answer extraction for official Self-RAG (``run_short_form.py``) output.

The raw ``run_short_form`` decode can carry non-answer artifacts around the actual
answer:
  * a leading role tag ("Assistant:" / "AI:"),
  * an echoed copy of the (history-folded) question,
  * hallucinated dialogue continuations ("\nUser: ...", "\nAssistant: ..."),
  * Self-RAG reflection control tokens ("[Retrieval]", "[Relevant]",
    "[Fully supported]", "[Utility:5]", "<paragraph>", ...).

``extract_selfrag_answer`` removes ONLY these mechanical artifacts. It never
rewrites, improves, fact-corrects, or otherwise changes the semantic content of the
answer -- a genuine RAG mistake must survive extraction verbatim, because the whole
experiment depends on observing real failures.

Everything here is pure string handling and unit-testable with no GPU / model.
"""
from __future__ import annotations

import re
from typing import Optional

# Self-RAG reflection / structural control tokens (NOT answer content).
_REFLECTION_TOKENS = [
    "[Retrieval]", "[No Retrieval]", "[Continue to Use Evidence]",
    "[Relevant]", "[Irrelevant]",
    "[Fully supported]", "[Partially supported]",
    "[No support / Contradictory]", "[No support/Contradictory]",
    "<paragraph>", "</paragraph>",
]
# [Utility:1] .. [Utility:5]
_UTILITY_RE = re.compile(r"\[Utility:\s*\d\]")
# a line that starts a (new) dialogue turn -> everything from here on is continuation
_ROLE_LINE_RE = re.compile(r"^\s*(user|assistant|ai|human|system)\s*:", re.IGNORECASE)
# a leading role tag on the answer's own first line ("Assistant: <answer>")
_LEADING_ROLE_RE = re.compile(r"^\s*(assistant|ai)\s*:\s*", re.IGNORECASE)
# a leading "User:"/"Human:" line -> the output opens with an echoed question
_USER_LINE_RE = re.compile(r"^\s*(user|human)\s*:", re.IGNORECASE)


def _strip_reflection_tokens(text: str) -> str:
    for tok in _REFLECTION_TOKENS:
        text = text.replace(tok, " ")
    text = _UTILITY_RE.sub(" ", text)
    return text


def extract_selfrag_answer(raw: str, question: Optional[str] = None) -> str:
    """Return only the actual answer from a raw Self-RAG short-form decode.

    ``question`` (optional) is the exact generation input the model was given; when
    the decode echoes it verbatim at the start, the echo is dropped. Semantic
    content is never altered.
    """
    if raw is None:
        return ""
    text = _strip_reflection_tokens(str(raw)).strip()

    # Drop a verbatim echo of the (possibly history-folded) question at the very start.
    if question:
        q = question.strip()
        if q and text.startswith(q):
            text = text[len(q):].lstrip(" \n:->")

    lines = text.split("\n")
    # skip leading blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ""

    # If the output is framed as dialogue that opens with a "User:"/"Human:" line
    # (an echoed question, sometimes followed by an empty "Assistant:"), the answer
    # is the content AFTER the first "Assistant:"/"AI:" marker. If no such marker
    # exists, there is no actual answer -> empty (a genuine no-answer failure, kept
    # as-is, never patched with gold).
    if _USER_LINE_RE.match(lines[0]):
        start = None
        for i, ln in enumerate(lines):
            m = _LEADING_ROLE_RE.match(ln)   # matches "Assistant:"/"AI:"
            if m:
                lines[i] = _LEADING_ROLE_RE.sub("", ln)
                start = i
                break
        if start is None:
            return ""
        lines = lines[start:]
    else:
        # strip a leading role tag on the first content line ("Assistant: ...")
        lines[0] = _LEADING_ROLE_RE.sub("", lines[0])

    # keep lines until a NEW dialogue turn begins (hallucinated continuation)
    kept = []
    for i, ln in enumerate(lines):
        if i > 0 and _ROLE_LINE_RE.match(ln):
            break
        kept.append(ln)

    return "\n".join(kept).strip()

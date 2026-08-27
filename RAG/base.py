#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG connector contract + shared generation + gold-leakage safeguards.

Every tested RAG exposes the same tiny interface so the runner can drive them
identically while each keeps its OWN native retrieval/generation:

    class SomeRAG(BaseRAG):
        name = "vector"
        available = True / False
        unavailable_reason = "<why>"        # when available is False
        def info(self) -> dict: ...          # impl/version/model/index config (for metadata)
        def retrieve(self, query: str, k: int) -> list[dict]: ...
        def generate(self, history, question, docs) -> dict: ...

``retrieve`` returns rich per-doc records (rank/doc_id/score/text + system-specific
provenance). ``generate`` returns {"answer", "final_context_documents", "model",
"messages", "parameters", ...} so retrieval candidates and the context actually
used for generation are kept separate (a RAG may rerank/filter/select).

Gold safeguard: the MTRAG reference answer and gold evidence text must never appear
in the input handed to a RAG. ``assert_no_gold_leak`` enforces this per turn.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional


# ---- shared MTRAG-baseline generator ---------------------------------------
# Verbatim official MTRAG generator system instruction (the neutral baseline used
# across the framework). Reproduced here so the one locally-runnable system does
# not depend on importing research code; retrieval is what differs between systems.
MTRAG_SYSTEM_INSTRUCTION = (
    "You are an AI Assistant, tasked with providing responses that are "
    "well-grounded in the provided documents. Given one or more documents and a "
    "user query, generate a response to the query. If no answer can be found in "
    "the documents, say, \"I do not have specific information\"."
)


def build_messages(history: List[Dict], question: str, passages: List[str]) -> List[Dict]:
    """Chat messages: system(instruction + docs) + folded history + current question.

    ``history`` is the tested RAG's OWN conversation so far ([{role, content}, ...]).
    """
    doc_block = "\n\n".join(f"[DOCUMENT {i + 1}]\n{p}\n[END]" for i, p in enumerate(passages))
    system = MTRAG_SYSTEM_INSTRUCTION
    if doc_block:
        system = system + "\n\nDocuments:\n" + doc_block
    messages = [{"role": "system", "content": system}]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": question})
    return messages


def openai_generate(client, history: List[Dict], question: str, passages: List[str],
                    model: str = "gpt-4o-mini", temperature: float = 0.0,
                    max_tokens: int = 512) -> Dict:
    """Run the shared generator and return a full generation record."""
    messages = build_messages(history, question, passages)
    answer = ""
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                max_tokens=max_tokens)
            answer = (resp.choices[0].message.content or "").strip()
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    return {
        "answer": answer,
        "model": model,
        "messages": messages,
        "parameters": {"temperature": temperature, "max_tokens": max_tokens},
    }


# ---- gold-leakage safeguard -------------------------------------------------
def _norm(s: str) -> str:
    return " ".join((s or "").split()).lower()


def assert_no_gold_leak(rag_input_texts: List[str], reference_answer: str,
                        reference_evidence_texts: Optional[List[str]] = None,
                        retrieved_texts: Optional[List[str]] = None) -> None:
    """Raise if the MTRAG reference answer (or gold evidence the RAG did NOT itself
    retrieve) appears in the text handed to the RAG.

    ``rag_input_texts`` = every string given to the RAG this turn (history contents,
    the question, the retrieved passages, the final context). Gold evidence is only
    a leak if it was injected WITHOUT being independently retrieved, so evidence text
    that also appears in ``retrieved_texts`` is allowed.
    """
    blob = "\n".join(_norm(t) for t in rag_input_texts if t)
    retrieved_blob = "\n".join(_norm(t) for t in (retrieved_texts or []) if t)
    ref = _norm(reference_answer)
    # Reference answers can be short/generic ("I'm sorry, ..."); only flag a
    # substantive verbatim reference answer (>= 6 tokens) appearing in the input.
    # A reference answer that also appears in an independently-retrieved passage is
    # NOT a leak: the RAG legitimately pulled that public-corpus passage itself
    # (some MTRAG reference answers are verbatim corpus passages). Same exemption
    # as the gold-evidence check below.
    if (ref and len(ref.split()) >= 6 and ref in blob
            and ref not in retrieved_blob):
        raise AssertionError(
            "gold leak: MTRAG reference answer appears verbatim in RAG input")
    for ev in (reference_evidence_texts or []):
        evn = _norm(ev)
        if evn and len(evn.split()) >= 12 and evn in blob and evn not in retrieved_blob:
            raise AssertionError(
                "gold leak: gold evidence injected into RAG input without being "
                "independently retrieved")


# ---- base class -------------------------------------------------------------
class BaseRAG:
    """Minimal contract. Subclasses set ``name`` and implement retrieve/generate."""

    name: str = "base"
    available: bool = False
    unavailable_reason: str = "not implemented"

    def info(self) -> Dict:
        """Reproducibility metadata (impl/version, models, index config)."""
        return {"name": self.name}

    def retrieve(self, query: str, k: int) -> List[Dict]:
        raise NotImplementedError

    def generate(self, history: List[Dict], question: str, docs: List[Dict]) -> Dict:
        """Given retrieved ``docs``, produce the answer + the final context used.

        Default: use the top docs as the final context and the shared generator.
        Systems whose generation is inseparable from the method override this.
        """
        raise NotImplementedError

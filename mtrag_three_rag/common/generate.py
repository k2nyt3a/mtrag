#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared, controlled RAG generator for the retrieval-method comparison.

RAPTOR and HippoRAG each keep their OWN native retrieval/indexing mechanism (the
method under test). Their retrieved passages are then handed to THIS single shared
generator so that the only thing that differs between them is retrieval. Self-RAG
is the exception: its trained 7B model is inseparable from the method, so it uses
its own generator (handled in selfrag_sys/).

Generator config (identical to the neutral MTRAG baseline in conv-annotate/rag):
  * model = gpt-4o-mini, temperature 0, max 512 tokens
  * system instruction = MTRAG's OWN generator prompt (verbatim)
  * history passed as chat turns; retrieved passages placed in the system message
"""
from __future__ import annotations
import time
from typing import Dict, List

MTRAG_SYSTEM_INSTRUCTION = (
    "You are an AI Assistant, tasked with providing responses that are "
    "well-grounded in the provided documents. Given one or more documents and a "
    "user query, generate a response to the query. If no answer can be found in "
    "the documents, say, \"I do not have specific information\"."
)


def build_messages(history: List[Dict], question: str, passages: List[str]) -> List[Dict]:
    doc_block = "\n\n".join(f"[DOCUMENT {i+1}]\n{p}\n[END]" for i, p in enumerate(passages))
    system = MTRAG_SYSTEM_INSTRUCTION
    if doc_block:
        system = system + "\n\nDocuments:\n" + doc_block
    messages = [{"role": "system", "content": system}]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": question})
    return messages


def generate(client, history: List[Dict], question: str, passages: List[str],
             model: str = "gpt-4o-mini", max_tokens: int = 512) -> str:
    messages = build_messages(history, question, passages)
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=0.0, max_tokens=max_tokens)
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Self-RAG connector — delegates to the existing pipeline in
``jaist/mtrag_three_rag/selfrag_sys/`` (official ``selfrag/selfrag_llama2_7b``
trained reflection model via vLLM + official Contriever retriever). Its trained 7B
generator is inseparable from the method, so — unlike Vector/HippoRAG/RAPTOR — it
does NOT use the shared generator.

By design Self-RAG runs on a 24 GB GPU box (vLLM). It is therefore split into:
  1. prepare inputs locally (history-folded question + last-turn retrieval query,
     reference answer kept for EVAL ONLY — no gold leaks to the model),
  2. run retrieval + reflection generation on the GPU box (remote), and
  3. ingest the remote outputs back into this pipeline's schema.

This connector never fabricates a local stand-in. When the GPU stack (``vllm`` +
trained model) is not present, it reports itself unavailable with the exact reason.
Once remote outputs exist they can be ingested via ``ingest_remote_outputs``.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from ..base import BaseRAG
from .. import _jaist

GEN_MODEL = "selfrag/selfrag_llama2_7b"


def _probe():
    if not _jaist.ensure_on_path():
        return False, f"research repo not found (set JAIST_ROOT); expected {_jaist.THREE_RAG}"
    try:
        import vllm  # noqa: F401
    except Exception as e:
        return False, (f"Self-RAG needs the trained 7B model served via vLLM on a 24 GB "
                       f"GPU ('vllm' not importable: {type(e).__name__}). This is a remote "
                       f"job - see mtrag_three_rag/selfrag_sys/REMOTE_README.md. Prepare "
                       f"inputs locally, run on the GPU box, then ingest_remote_outputs().")
    return True, ""


class SelfRAGConnector(BaseRAG):
    name = "selfrag"

    def __init__(self, domain: str, k: int = 5):
        self.domain = domain
        self.k = k
        self.available, self.unavailable_reason = _probe()

    def info(self) -> Dict:
        return {
            "name": self.name,
            "implementation": "official selfrag_llama2_7b (vLLM) + Contriever, upstream",
            "adapter": "jaist/mtrag_three_rag/selfrag_sys/ (remote GPU)",
            "generation_model": GEN_MODEL,
            "retriever": "facebook/contriever-msmarco",
            "runbook": "mtrag_three_rag/selfrag_sys/REMOTE_README.md",
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
        }

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict]:
        raise RuntimeError(f"selfrag unavailable in-process: {self.unavailable_reason}")

    def generate(self, history: List[Dict], question: str, docs: List[Dict]) -> Dict:
        raise RuntimeError(f"selfrag unavailable in-process: {self.unavailable_reason}")


def build(domain: str, k: int = 5, **kw) -> SelfRAGConnector:
    return SelfRAGConnector(domain, k=k)

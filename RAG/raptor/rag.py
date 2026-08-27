#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAPTOR connector — delegates to the existing upstream-backed adapter in
``jaist/mtrag_three_rag/raptor_sys/raptor_adapter.py`` (official RAPTOR: recursive
GMM/UMAP clustering + LLM cluster summarisation + collapsed-tree retrieval).
Nothing about RAPTOR is reimplemented or faked here.

Availability requires BOTH:
  * the ``raptor`` package importable (WSL raptor_env + repos/raptor .pth), and
  * a prebuilt tree for the domain at
    ``mtrag_three_rag/indices/raptor/<domain>_full.pkl`` (building one embeds +
    clusters + LLM-summarises the whole corpus — never triggered implicitly).

Retrieval preserves RAPTOR's hierarchical provenance (node id / level / type and,
for summary nodes, the source leaf document ids).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from ..base import BaseRAG, openai_generate
from .. import _jaist, mtrag_data

GEN_MODEL = "gpt-4o-mini"


def _tree_path(domain: str) -> str:
    return os.path.join(_jaist.THREE_RAG, "indices", "raptor", f"{domain}_full.pkl")


def _probe(domain: str):
    if not _jaist.ensure_on_path():
        return False, f"research repo not found (set JAIST_ROOT); expected {_jaist.THREE_RAG}"
    try:
        import raptor  # noqa: F401
    except Exception as e:
        return False, (f"'raptor' package not importable in this interpreter "
                       f"({type(e).__name__}: {str(e)[:80]}). Run inside the WSL "
                       f"raptor_env (see mtrag_three_rag/setup_uni.sh).")
    tp = _tree_path(domain)
    if not os.path.isfile(tp):
        return False, (f"no prebuilt RAPTOR tree for '{domain}' at {tp}. Build it with "
                       f"mtrag_three_rag/run_full.py raptor {domain} (expensive: "
                       f"embeds + clusters + LLM-summarises the full corpus).")
    return True, ""


class RaptorConnector(BaseRAG):
    name = "raptor"

    def __init__(self, domain: str, k: int = 5):
        self.domain = domain
        self.k = k
        self.available, self.unavailable_reason = _probe(domain)
        self._impl = None
        self._client = None
        if not self.available:
            return
        mtrag_data.load_openai_key()
        from openai import OpenAI
        self._client = OpenAI()
        import pickle
        from raptor_sys.raptor_adapter import RaptorMTRAG  # type: ignore
        d = pickle.load(open(_tree_path(domain), "rb"))
        self._impl = RaptorMTRAG(d["tree"], d["leaf_id_map"], d["n_leaves"],
                                 self._client, k=k)

    def info(self) -> Dict:
        return {
            "name": self.name,
            "implementation": "official RAPTOR (collapsed-tree retrieval), upstream package",
            "adapter": "jaist/mtrag_three_rag/raptor_sys/raptor_adapter.py",
            "tree": _tree_path(self.domain),
            "embedding_model": "text-embedding-3-small",
            "summariser_model": GEN_MODEL,
            "generation_model": GEN_MODEL,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
        }

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict]:
        if not self.available:
            raise RuntimeError(f"raptor unavailable: {self.unavailable_reason}")
        self._impl.k = k or self.k
        docs = self._impl.retrieve(query)
        for d in docs:
            d.setdefault("passage_id", d.get("doc_id"))
        return docs

    def generate(self, history: List[Dict], question: str, docs: List[Dict]) -> Dict:
        final = docs
        gen = openai_generate(self._client, history, question,
                              [d["text"] for d in final], model=GEN_MODEL)
        gen["final_context_documents"] = [
            {"doc_id": d["doc_id"], "rank": d["rank"], "score": d.get("score"),
             "raptor_node_id": d.get("raptor_node_id"),
             "raptor_node_level": d.get("raptor_node_level"),
             "raptor_node_type": d.get("raptor_node_type"),
             "source_document_ids": d.get("source_document_ids"),
             "text": d["text"]} for d in final]
        gen["intermediate"] = {"note": "RAPTOR collapsed-tree nodes passed to shared generator"}
        return gen


def build(domain: str, k: int = 5, **kw) -> RaptorConnector:
    return RaptorConnector(domain, k=k)

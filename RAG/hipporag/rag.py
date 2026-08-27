#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HippoRAG connector — delegates to the existing upstream-backed adapter in
``jaist/mtrag_three_rag/hipporag_sys/hipporag_adapter.py`` (official HippoRAG v2:
LLM OpenIE knowledge graph + Personalized-PageRank retrieval fused with dense
retrieval). Nothing about HippoRAG is reimplemented or faked here.

Availability requires BOTH:
  * the ``hipporag`` package importable in the running interpreter, and
  * a prebuilt OpenIE graph index for the domain under
    ``mtrag_three_rag/indices/hipporag/<domain>_full`` (building one runs LLM OpenIE
    over the whole corpus — hours + API cost — so it is never triggered implicitly).

If either is missing the connector reports itself unavailable with the exact reason
and the runner refuses to substitute another system.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from ..base import BaseRAG, openai_generate
from .. import _jaist, mtrag_data

GEN_MODEL = "gpt-4o-mini"


def _index_dir(domain: str) -> str:
    return os.path.join(_jaist.THREE_RAG, "indices", "hipporag", f"{domain}_full")


def _probe(domain: str):
    """Return (available: bool, reason: str)."""
    if not _jaist.ensure_on_path():
        return False, f"research repo not found (set JAIST_ROOT); expected {_jaist.THREE_RAG}"
    try:
        import hipporag  # noqa: F401
    except Exception as e:
        return False, (f"'hipporag' package not importable in this interpreter "
                       f"({type(e).__name__}: {str(e)[:80]}). Run inside the WSL "
                       f"hipporag_env (see mtrag_three_rag/setup_uni.sh).")
    idx = _index_dir(domain)
    if not os.path.isdir(idx):
        return False, (f"no prebuilt OpenIE index for '{domain}' at {idx}. Build it "
                       f"with mtrag_three_rag/run_full.py hipporag {domain} "
                       f"(expensive: LLM OpenIE over the full corpus).")
    return True, ""


class HippoRAGConnector(BaseRAG):
    name = "hipporag"

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
        # Load the existing index via the upstream-backed adapter (no rebuild).
        from hipporag_sys.hipporag_adapter import HippoMTRAG  # type: ignore
        from hipporag import HippoRAG  # type: ignore
        import pickle
        save_dir = _index_dir(domain)
        id_map = pickle.load(open(os.path.join(save_dir, "text_to_corpus_id.pkl"), "rb"))
        from hipporag.utils.config_utils import BaseConfig  # type: ignore
        cfg = BaseConfig()
        cfg.save_dir = save_dir
        cfg.llm_name = GEN_MODEL
        cfg.embedding_model_name = "text-embedding-3-small"
        self._impl = HippoMTRAG(HippoRAG(global_config=cfg), id_map, k=k)

    def info(self) -> Dict:
        return {
            "name": self.name,
            "implementation": "official HippoRAG v2 (OpenIE KG + PPR + dense), upstream package",
            "adapter": "jaist/mtrag_three_rag/hipporag_sys/hipporag_adapter.py",
            "index": _index_dir(self.domain),
            "generation_model": GEN_MODEL,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
        }

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict]:
        if not self.available:
            raise RuntimeError(f"hipporag unavailable: {self.unavailable_reason}")
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
             "score_type": d.get("score_type"), "text": d["text"]} for d in final]
        gen["intermediate"] = {"note": "HippoRAG graph-ranked passages passed to shared generator"}
        return gen


def build(domain: str, k: int = 5, **kw) -> HippoRAGConnector:
    return HippoRAGConnector(domain, k=k)

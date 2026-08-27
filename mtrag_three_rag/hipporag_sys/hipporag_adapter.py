#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Official HippoRAG v2 (Gutiérrez et al.) integration for MTRAG.

Uses the UPSTREAM `hipporag` package unchanged: it runs LLM OpenIE over every
passage to build a knowledge graph offline, then does graph search (Personalized
PageRank over facts/entities) + dense passage retrieval online. Nothing about the
HippoRAG algorithm is reimplemented.

  * llm (OpenIE + reader) = gpt-4o-mini via the OpenAI API
  * embeddings            = text-embedding-3-small via the OpenAI API
  * graph                 = built ONLY from the MTRAG domain corpus (no answers /
                            qrels / future questions).

The index (OpenIE results + graph + passage embeddings) is stored under save_dir
on D:. Retrieved passages come back as the passage STRINGS we indexed, so we map
each back to its MTRAG corpus id via a text->id table captured at index time
(provenance for retrieval eval).

We use HippoRAG's native retrieval, then hand the retrieved passages to the shared
controlled generator (common/generate.py) — identical generation to RAPTOR, so only
the retrieval method differs.
"""
from __future__ import annotations
import json
import glob
import os
import pickle
from typing import Dict, List, Optional

from hipporag import HippoRAG


def _texts_to_id_map(texts: List[str], ids: List[str]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for t, i in zip(texts, ids):
        m.setdefault(t, i)          # first id wins on the (rare) duplicate passage text
    return m


def build_index(texts: List[str], ids: List[str], save_dir: str,
                llm_model_name: str = "gpt-4o-mini",
                embedding_model_name: str = "text-embedding-3-small",
                llm_base_url: str = "https://api.openai.com/v1",
                openie_max_workers: int = 8,
                embedding_batch_size: int = 16,
                tolerant_openie: bool = True) -> "HippoMTRAG":
    """Build (or load) the HippoRAG OpenIE graph for a domain and return a ready
    retriever. Idempotent: HippoRAG caches its OpenIE/graph under save_dir.

    openie_max_workers controls OpenIE (NER+triple) concurrency. It is read by the
    OpenIE module at HippoRAG-construction time, so it MUST be set on the BaseConfig
    passed as global_config (not mutated afterwards)."""
    os.makedirs(save_dir, exist_ok=True)
    hipmap_path = os.path.join(save_dir, "text_to_corpus_id.pkl")
    from hipporag.utils.config_utils import BaseConfig
    cfg = BaseConfig()
    cfg.save_dir = save_dir
    cfg.llm_name = llm_model_name
    cfg.embedding_model_name = embedding_model_name
    cfg.llm_base_url = llm_base_url
    cfg.openie_mode = "online"                 # OpenAI-backed OpenIE
    cfg.openie_max_workers = openie_max_workers
    cfg.embedding_batch_size = embedding_batch_size   # default 16 is very slow at 1M+ entities;
                                               # raise (e.g. 256) on a bigger-RAM box
    cfg.save_openie = True                     # persist OpenIE results (resumable)
    cfg.max_retry_attempts = 10                # ride out 429 rate limits w/ backoff
                                               # (default 2 is too few at high concurrency)
    # Resume guard: if a prior run left chunk embeddings but no graph (e.g. it aborted
    # during OpenIE), HippoRAG refuses to continue unless force_index_from_scratch=True.
    # Enable it ONLY in that partial state — it rebuilds the graph while OpenIE calls hit
    # the sqlite cache (NER cache-hits => no re-bill; only new triples are billed). Once a
    # graph exists this stays False so a completed index is reused, not rebuilt.
    _subs = glob.glob(os.path.join(save_dir, "*text-embedding-3-small*"))
    _has_chunks = any(glob.glob(os.path.join(s, "chunk_embeddings", "*.parquet")) for s in _subs)
    _has_graph = any(glob.glob(os.path.join(s, "graph.pickle")) for s in _subs)
    if _has_chunks and not _has_graph:
        cfg.force_index_from_scratch = True
        print("[resume] partial index (embeddings, no graph) -> force_index_from_scratch=True "
              "(OpenIE served from cache; graph will be rebuilt)", flush=True)
    hippo = HippoRAG(global_config=cfg)

    if tolerant_openie:
        # Graceful degradation: official HippoRAG aborts the ENTIRE build if any single
        # chunk's NER/triple extraction errors (openie_openai.py raises on any error).
        # On the full clapnq corpus ~0.01% of chunks are pathological — OpenAI content-filter
        # blocks, or deterministic non-JSON output on LaTeX / dense term-lists / fragments —
        # which no config can fix. We wrap the OpenIE methods (leaving the official package
        # file untouched) so a failed chunk is treated as zero-entity/zero-triple and the
        # build continues. Such chunks remain in the corpus and dense-retrievable; they just
        # contribute no graph nodes/edges. NER token cap is left at the official default (512)
        # to preserve the LLM cache (the cache key includes max_tokens, so raising it would
        # re-bill every cached call). Each skip is logged so a rate-limit storm is visible.
        import functools
        _skips = {"NER": 0, "TRIPLE": 0}

        def _wrap(bound_method, kind):
            @functools.wraps(bound_method)
            def inner(chunk_key, *a, **kw):
                res = bound_method(chunk_key, *a, **kw)
                md = getattr(res, "metadata", None)
                if isinstance(md, dict) and md.get("error"):
                    _skips[kind] += 1
                    print(f"[tolerant-openie] {kind} skipped chunk {chunk_key}: "
                          f"{str(md.get('error'))[:80]}", flush=True)
                    md.pop("error", None)
                    md["skipped_openie_error"] = True
                return res
            return inner

        hippo.openie.ner = _wrap(hippo.openie.ner, "NER")
        hippo.openie.triple_extraction = _wrap(hippo.openie.triple_extraction, "TRIPLE")
        hippo._mtrag_openie_skips = _skips

    # index() reuses the OpenIE/LLM sqlite cache (resumable); passing full corpus is safe
    hippo.index(docs=list(texts))
    if tolerant_openie:
        print(f"[tolerant-openie] total skipped: NER={hippo._mtrag_openie_skips['NER']} "
              f"TRIPLE={hippo._mtrag_openie_skips['TRIPLE']}", flush=True)
    id_map = _texts_to_id_map(texts, ids)
    with open(hipmap_path, "wb") as f:
        pickle.dump(id_map, f)
    return HippoMTRAG(hippo, id_map)


class HippoMTRAG:
    name = "hipporag"

    def __init__(self, hippo: "HippoRAG", id_map: Dict[str, str], k: int = 5):
        self.hippo = hippo
        self.id_map = id_map
        self.k = k

    def retrieve(self, query: str) -> List[Dict]:
        sols = self.hippo.retrieve(queries=[query], num_to_retrieve=self.k)
        sol = sols[0] if isinstance(sols, list) else sols
        docs = sol.docs or []
        scores = sol.doc_scores.tolist() if getattr(sol, "doc_scores", None) is not None else [None] * len(docs)
        out = []
        for rank, (txt, sc) in enumerate(zip(docs, scores)):
            out.append({
                "rank": rank + 1,
                "doc_id": self.id_map.get(txt, f"hipporag_unmapped_{rank}"),
                "score": round(float(sc), 6) if sc is not None else None,
                "score_type": "hipporag_graph",   # PPR-over-OpenIE-graph + dense fusion
                "text": txt,
            })
        return out

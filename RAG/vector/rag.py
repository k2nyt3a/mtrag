#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vector RAG over the official MTRAG domain corpus.

Fully self-contained and runnable locally: dense retrieval using the corpus
embedding matrix that is precomputed ONCE per domain (``<domain>.embindex.npy``,
text-embedding-3-small, L2-normalised, fingerprint-verified row-aligned to the
corpus). The query is embedded with the SAME model and L2-normalised, so ranking
is cosine similarity. Retrieved passages are handed to the shared MTRAG-baseline
generator — so this is genuine dense-retrieval RAG, not a stand-in for any other
system.

The index is built ONLY from the domain corpus — no reference answers, qrels, or
future questions ever enter it.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from ..base import BaseRAG, openai_generate
from .. import mtrag_data

EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-4o-mini"


class VectorRAG(BaseRAG):
    name = "vector"
    available = True
    unavailable_reason = ""

    def __init__(self, domain: str, k: int = 5, client=None):
        self.domain = domain
        self.k = k
        mtrag_data.load_openai_key()
        from openai import OpenAI
        self.client = client or OpenAI()

        # corpus (row-aligned to the precomputed embedding matrix)
        self.texts, self.ids, self.titles = mtrag_data.load_corpus(domain)
        base = mtrag_data.embindex_path(domain)
        meta = json.load(open(base + ".json", encoding="utf-8"))
        self.matrix = np.load(base + ".npy", mmap_mode="r")
        if self.matrix.shape[0] != len(self.texts):
            raise RuntimeError(
                f"embindex/corpus mismatch for {domain}: "
                f"{self.matrix.shape[0]} vs {len(self.texts)} passages")
        self.embed_model = meta.get("model", EMBED_MODEL)
        self.dim = int(meta.get("dim") or self.matrix.shape[1])
        self._built = True

    # ---- reproducibility metadata ----
    def info(self) -> Dict:
        return {
            "name": self.name,
            "implementation": "dense retrieval (cosine) over official MTRAG corpus",
            "embedding_model": self.embed_model,
            "generation_model": GEN_MODEL,
            "corpus": os.path.basename(mtrag_data.corpus_path(self.domain)),
            "corpus_passages": len(self.texts),
            "embedding_index": os.path.basename(mtrag_data.embindex_path(self.domain)) + ".npy",
            "embedding_dim": self.dim,
            "chunking": "official MTRAG passage-level corpus (1 passage = 1 chunk)",
            "index_scope": "domain corpus only (no answers/qrels/future questions)",
        }

    # ---- native retrieval ----
    def _embed_query(self, query: str) -> np.ndarray:
        for attempt in range(6):
            try:
                v = self.client.embeddings.create(
                    input=[query.replace("\n", " ")], model=self.embed_model).data[0].embedding
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        q = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(q) or 1.0
        return q / n

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict]:
        k = k or self.k
        q = self._embed_query(query)
        # corpus matrix is already L2-normalised -> dot product == cosine
        sims = np.asarray(self.matrix @ q, dtype=np.float32)
        topk = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        topk = topk[np.argsort(-sims[topk])]
        out: List[Dict] = []
        for rank, i in enumerate(topk):
            out.append({
                "rank": rank + 1,
                "doc_id": self.ids[i],
                "passage_id": self.ids[i],
                "chunk_id": int(i),
                "title": self.titles[i],
                "score": round(float(sims[i]), 6),
                "score_type": "cosine",
                "text": self.texts[i],
                "metadata": {"corpus_row": int(i)},
            })
        return out

    # ---- generation over the retrieved passages ----
    def generate(self, history: List[Dict], question: str, docs: List[Dict]) -> Dict:
        final = docs  # Vector RAG uses its top-k retrieved passages as-is (no rerank)
        gen = openai_generate(self.client, history, question,
                              [d["text"] for d in final], model=GEN_MODEL)
        gen["final_context_documents"] = [
            {"doc_id": d["doc_id"], "passage_id": d.get("passage_id"),
             "chunk_id": d.get("chunk_id"), "rank": d["rank"], "score": d["score"],
             "title": d.get("title", ""), "text": d["text"]}
            for d in final]
        gen["intermediate"] = {"note": "no rerank/filter; final context = top-k retrieved"}
        return gen


def build(domain: str, k: int = 5, **kw) -> VectorRAG:
    return VectorRAG(domain, k=k)

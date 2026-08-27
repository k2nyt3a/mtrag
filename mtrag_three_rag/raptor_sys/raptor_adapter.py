#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Official RAPTOR (Sarthi et al., ICLR 2024) integration for MTRAG.

Uses the UPSTREAM raptor package (repos/raptor) unchanged — its recursive
GMM/UMAP clustering, LLM cluster summarisation, and collapsed-tree retrieval —
with custom OpenAI model classes plugged in via RAPTOR's documented base classes
(BaseEmbeddingModel / BaseSummarizationModel / BaseQAModel). Nothing about the
RAPTOR algorithm is reimplemented.

  * embeddings   = text-embedding-3-small  (leaf embeddings batch-precomputed;
                   summary-node embeddings computed on demand during the build)
  * summariser   = gpt-4o-mini   (builds the tree's summary nodes)
  * tree         = built ONLY from the MTRAG domain corpus (no answers/qrels/future
                   questions) — 1 MTRAG passage == 1 leaf, so leaf index i maps to
                   corpus passage id[i] (provenance anchor).

Retrieval = RAPTOR collapsed tree: leaves + all summary nodes pooled, top-k by
cosine. We record, per retrieved node: node id, tree level, node type
(leaf|summary), the cosine score, and — for summaries — the corpus document ids of
all leaf descendants (provenance back to MTRAG doc ids for retrieval eval).
"""
from __future__ import annotations
import copy
import os
import pickle
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI

# raptor package (importable via the raptor_env .pth)
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.EmbeddingModels import BaseEmbeddingModel
from raptor.SummarizationModels import BaseSummarizationModel
from raptor.QAModels import BaseQAModel
from raptor.tree_structures import Tree
from raptor.tree_retriever import TreeRetriever


# ---------------- custom OpenAI model classes (RAPTOR base classes) ----------
class OpenAIEmbed(BaseEmbeddingModel):
    """text-embedding-3-small with an optional precomputed leaf cache (by text)."""

    def __init__(self, model="text-embedding-3-small", client=None, cache: Optional[Dict[str, list]] = None):
        self.client = client or OpenAI()
        self.model = model
        self.cache = cache or {}

    def create_embedding(self, text):
        v = self.cache.get(text)
        if v is not None:
            return v
        t = text.replace("\n", " ")
        for attempt in range(6):
            try:
                return self.client.embeddings.create(input=[t], model=self.model).data[0].embedding
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)


class OpenAISummarize(BaseSummarizationModel):
    def __init__(self, model="gpt-4o-mini", client=None):
        self.client = client or OpenAI()
        self.model = model

    def summarize(self, context, max_tokens=200, stop_sequence=None):
        for attempt in range(6):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, max_tokens=max_tokens, temperature=0.0,
                    messages=[{"role": "system", "content":
                               "Write a concise, information-dense summary of the passages "
                               "below that preserves key facts, names and numbers."},
                              {"role": "user", "content": context}])
                return (r.choices[0].message.content or "").strip()
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)


class OpenAIQA(BaseQAModel):
    """Required by RAPTOR config; the MTRAG runs use the shared generator instead,
    but a functional QA model is provided for completeness / standalone use."""

    def __init__(self, model="gpt-4o-mini", client=None):
        self.client = client or OpenAI()
        self.model = model

    def answer_question(self, context, question, max_tokens=512, stop_sequence=None):
        r = self.client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, temperature=0.0,
            messages=[{"role": "system", "content": "Answer the question from the context."},
                      {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}])
        return (r.choices[0].message.content or "").strip()


def make_config(client, embed_cache=None, tr_top_k=5, tb_num_layers=5) -> RetrievalAugmentationConfig:
    emb = OpenAIEmbed(client=client, cache=embed_cache)
    return RetrievalAugmentationConfig(
        embedding_model=emb,                       # sets tb+tr embedding to "EMB"
        summarization_model=OpenAISummarize(client=client),
        qa_model=OpenAIQA(client=client),
        tr_top_k=tr_top_k,
        tb_num_layers=tb_num_layers,
        tb_max_tokens=100000,                      # do NOT re-split MTRAG passages
    )


# ---------------- tree build (1 passage == 1 leaf) ---------------------------
def _batch_embed_leaves(client, texts: List[str], model="text-embedding-3-small",
                        batch=1024) -> Dict[str, list]:
    """Batch-precompute leaf embeddings so the tree build does not make one API call
    per passage. Returns {text: vector}."""
    cache: Dict[str, list] = {}
    uniq = list(dict.fromkeys(texts))
    for i in range(0, len(uniq), batch):
        chunk = [t.replace("\n", " ") for t in uniq[i:i + batch]]
        for attempt in range(6):
            try:
                resp = client.embeddings.create(input=chunk, model=model)
                for t, d in zip(uniq[i:i + batch], resp.data):
                    cache[t] = d.embedding
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        print(f"  embedded leaves {min(i+batch,len(uniq))}/{len(uniq)}", flush=True)
    return cache


def build_tree(texts: List[str], ids: List[str], cache_path: str, client,
               tb_num_layers=5, precompute=True) -> Tuple[Tree, List[str], int]:
    """Build the RAPTOR tree from MTRAG passages and cache it to `cache_path`.

    Returns (tree, leaf_id_map, n_leaves). leaf_id_map[i] = corpus passage id of leaf i.
    """
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            d = pickle.load(f)
        return d["tree"], d["leaf_id_map"], d["n_leaves"]

    embed_cache = _batch_embed_leaves(client, texts) if precompute else None
    config = make_config(client, embed_cache=embed_cache, tb_num_layers=tb_num_layers)
    RA = RetrievalAugmentation(config=config)         # builds tree_builder (no tree yet)
    tb = RA.tree_builder

    # Replicate build_from_text WITHOUT split_text: our chunks ARE the leaves.
    leaf_nodes = tb.multithreaded_create_leaf_nodes(texts)     # index i == passage i
    layer_to_nodes = {0: list(leaf_nodes.values())}
    all_nodes = copy.deepcopy(leaf_nodes)
    root_nodes = tb.construct_tree(all_nodes, all_nodes, layer_to_nodes)
    tree = Tree(all_nodes, root_nodes, leaf_nodes, tb.num_layers, layer_to_nodes)

    n_leaves = len(texts)
    leaf_id_map = list(ids)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"tree": tree, "leaf_id_map": leaf_id_map, "n_leaves": n_leaves,
                     "num_layers": tree.num_layers}, f)
    return tree, leaf_id_map, n_leaves


# ---------------- retrieval + provenance -------------------------------------
class RaptorMTRAG:
    name = "raptor"

    def __init__(self, tree: Tree, leaf_id_map: List[str], n_leaves: int, client,
                 k: int = 5):
        self.tree = tree
        self.leaf_id_map = leaf_id_map
        self.n_leaves = n_leaves
        self.client = client
        self.k = k
        self.emb = OpenAIEmbed(client=client)
        self._layer = self._reverse_layers(tree)

    @staticmethod
    def _reverse_layers(tree: Tree) -> Dict[int, int]:
        m = {}
        for layer, nodes in tree.layer_to_nodes.items():
            for n in nodes:
                m[n.index] = layer
        return m

    def _leaf_descendants(self, node_index: int) -> List[int]:
        """All leaf node indices reachable from a node (itself if it's a leaf)."""
        if node_index < self.n_leaves:
            return [node_index]
        seen, stack, leaves = set(), [node_index], []
        while stack:
            i = stack.pop()
            node = self.tree.all_nodes[i]
            for c in node.children:
                if c in seen:
                    continue
                seen.add(c)
                if c < self.n_leaves:
                    leaves.append(c)
                else:
                    stack.append(c)
        return leaves

    def retrieve(self, query: str) -> List[Dict]:
        config = RetrievalAugmentationConfig(
            embedding_model=self.emb, summarization_model=OpenAISummarize(client=self.client),
            qa_model=OpenAIQA(client=self.client), tr_top_k=self.k, tb_max_tokens=100000)
        retriever = TreeRetriever(config.tree_retriever_config, self.tree)
        context, layer_info = retriever.retrieve(
            query, top_k=self.k, max_tokens=10**7, collapse_tree=True,
            return_layer_information=True)
        qv = np.asarray(self.emb.create_embedding(query), dtype=np.float32)
        qn = qv / (np.linalg.norm(qv) or 1.0)
        out = []
        for rank, li in enumerate(layer_info):
            ni = li["node_index"]
            node = self.tree.all_nodes[ni]
            level = li["layer_number"]
            is_leaf = ni < self.n_leaves
            nv = np.asarray(node.embeddings["EMB"], dtype=np.float32)
            score = float(nv @ qn / (np.linalg.norm(nv) or 1.0))
            leaves = self._leaf_descendants(ni)
            source_doc_ids = [self.leaf_id_map[l] for l in leaves]
            out.append({
                "rank": rank + 1,
                "doc_id": self.leaf_id_map[ni] if is_leaf else f"raptor_summary_{ni}",
                "score": round(score, 6),
                "score_type": "cosine",
                "text": node.text,
                # RAPTOR provenance (per the logging spec):
                "raptor_node_id": int(ni),
                "raptor_node_level": int(level),
                "raptor_node_type": "leaf" if is_leaf else "summary",
                "source_document_ids": source_doc_ids,
            })
        return out

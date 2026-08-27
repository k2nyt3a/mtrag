#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Official-style MTRAG retrieval metrics (BEIR conventions): Recall@k and nDCG@k
over the official qrels. Binary relevance (dev.tsv scores are 1). Averaged over the
queries that have qrels (the official Last-Turn evaluation set)."""
from __future__ import annotations
import json
import math
import os
from typing import Dict, List

from .mtrag_io import RETRIEVAL_TASKS_DIR, load_qrels

KS = [1, 3, 5, 10]


def load_lastturn_queries(domain: str) -> Dict[str, str]:
    """Official Last-Turn retrieval queries: {task_id: query_text}.
    query text is the MTRAG last-turn form (e.g. '|user|: ...')."""
    p = os.path.join(RETRIEVAL_TASKS_DIR, domain, f"{domain}_lastturn.jsonl")
    out: Dict[str, str] = {}
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                out[str(r["_id"])] = r["text"]
    return out


def _dcg(rels: List[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def compute_metrics(ranked: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]],
                    ks: List[int] = KS) -> Dict[str, float]:
    """ranked: {query_id: [doc_id ranked]}. qrels: {query_id: {doc_id: rel}}.
    Only query_ids present in BOTH ranked and qrels (with >=1 gold) are scored."""
    agg = {f"Recall@{k}": [] for k in ks}
    agg.update({f"nDCG@{k}": [] for k in ks})
    scored = 0
    for qid, gold in qrels.items():
        gold_ids = {d for d, r in gold.items() if r > 0}
        if not gold_ids or qid not in ranked:
            continue
        scored += 1
        run = ranked[qid]
        for k in ks:
            topk = run[:k]
            hits = sum(1 for d in topk if d in gold_ids)
            agg[f"Recall@{k}"].append(hits / len(gold_ids))
            rels = [1 if d in gold_ids else 0 for d in topk]
            idcg = _dcg([1] * min(k, len(gold_ids)))
            agg[f"nDCG@{k}"].append(_dcg(rels) / idcg if idcg > 0 else 0.0)
    result = {m: (sum(v) / len(v) if v else 0.0) for m, v in agg.items()}
    result["num_scored_queries"] = scored
    return result


def metrics_from_log(log_path: str, domain: str) -> Dict[str, float]:
    """Read a per-turn log jsonl (retrieved_document_ids per turn) and score it."""
    ranked: Dict[str, List[str]] = {}
    for line in open(log_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        ranked[r["task_id"]] = [d for d in r.get("retrieved_document_ids", []) if d]
    return compute_metrics(ranked, load_qrels(domain))

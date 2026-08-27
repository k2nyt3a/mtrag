#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared per-turn runner for RAPTOR / HippoRAG over a full MTRAG domain index.

Retrieval query = official Last-Turn query text (keyed by task_id) -> retrieve
top-k_retrieve (for R@k/nDCG@k). Generation = shared generator over top-k_gen
passages + folded history. Writes the per-turn log jsonl and the MTRAG eval jsonl.
Resumable: skips task_ids already present in the log."""
from __future__ import annotations
import json
import os
import time
from typing import Dict, List

from .mtrag_io import load_turns, load_qrels, load_corpus, make_log_record, to_mtrag_eval_record
from .eval_retrieval import load_lastturn_queries
from .generate import generate


def _done_task_ids(path: str):
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["task_id"])
                except Exception:
                    pass
    return done


def run_domain(system, domain: str, client, out_log: str, out_eval: str,
               k_retrieve: int = 10, k_gen: int = 5, verbose: bool = True):
    system.k = k_retrieve
    lastturn = load_lastturn_queries(domain)
    qrels = load_qrels(domain)
    corpus_texts, corpus_ids = load_corpus(domain)
    id2text = dict(zip(corpus_ids, corpus_texts))
    turns = load_turns(domains=[domain])

    os.makedirs(os.path.dirname(out_log) or ".", exist_ok=True)
    done = _done_task_ids(out_log)
    flog = open(out_log, "a", encoding="utf-8")
    feval = open(out_eval, "a", encoding="utf-8")
    n = 0
    for t in turns:
        if t.task_id in done:
            continue
        q_retr = lastturn.get(t.task_id, t.question)      # official last-turn query text
        t0 = time.time()
        docs = system.retrieve(q_retr)                     # top-k_retrieve
        gen_docs = docs[:k_gen]
        ans = generate(client, t.history, t.question, [d["text"] for d in gen_docs])
        lat = round(time.time() - t0, 2)
        gold = list(qrels.get(t.task_id, {}).keys())
        rec = make_log_record(t, system.name, docs, ans, gold_doc_ids=gold,
                              gold_passages=[id2text.get(g, "") for g in gold],
                              latency_s=lat, extra={"retrieval_query": q_retr})
        flog.write(json.dumps(rec, ensure_ascii=False) + "\n"); flog.flush()
        feval.write(json.dumps(to_mtrag_eval_record(t, gen_docs, ans), ensure_ascii=False) + "\n"); feval.flush()
        n += 1
        if verbose and n % 10 == 0:
            print(f"  {system.name}/{domain}: {n} turns done", flush=True)
    flog.close(); feval.close()
    return n

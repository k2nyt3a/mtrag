#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full-corpus build + run + eval for one system on one MTRAG domain.

  python run_full.py <hipporag|raptor> <domain> [openie_workers]

Builds the OFFICIAL index over the FULL domain corpus (the correct BEIR retrieval
universe — verified: e.g. clapnq = all 183,408 passages), runs every domain turn
(retrieval query = official Last-Turn text; top-10 for metrics, top-5 for
generation), writes the per-turn log + MTRAG eval jsonl, and prints official-style
retrieval metrics (Recall@/nDCG@ 1,3,5,10). Resumable at both stages (index caches
+ log skip-set).
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.mtrag_io import load_env_openai_key, load_corpus          # noqa: E402
from common.runner import run_domain                                   # noqa: E402
from common.eval_retrieval import metrics_from_log                     # noqa: E402

system = sys.argv[1]
domain = sys.argv[2]
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 32

load_env_openai_key()
from openai import OpenAI                                              # noqa: E402
client = OpenAI()

print(f"[{system}/{domain}] loading FULL corpus ...", flush=True)
texts, ids = load_corpus(domain)                                       # full corpus, no limit
print(f"  {len(texts)} passages (retrieval universe)", flush=True)

t0 = time.time()
if system == "hipporag":
    from hipporag_sys.hipporag_adapter import build_index
    save_dir = os.path.join(HERE, "indices", "hipporag", f"{domain}_full")
    emb_batch = int(os.environ.get("HIPPO_EMB_BATCH", "256"))   # 16 default is slow; raise on big-RAM box
    print(f"[build] HippoRAG OpenIE graph, {workers} workers, emb_batch={emb_batch} -> {save_dir}", flush=True)
    rag = build_index(texts, ids, save_dir, openie_max_workers=workers,
                      embedding_batch_size=emb_batch)
elif system == "raptor":
    from raptor_sys.raptor_adapter import build_tree, RaptorMTRAG
    cache = os.path.join(HERE, "indices", "raptor", f"{domain}_full.pkl")
    print(f"[build] RAPTOR tree -> {cache}", flush=True)
    tree, leaf_id_map, n_leaves = build_tree(texts, ids, cache, client)
    rag = RaptorMTRAG(tree, leaf_id_map, n_leaves, client, k=10)
else:
    raise SystemExit(f"unknown system: {system}")
print(f"[build] done in {time.time()-t0:.0f}s", flush=True)

out_log = os.path.join(HERE, "outputs", system, f"{domain}.log.jsonl")
out_eval = os.path.join(HERE, "outputs", system, f"{domain}.mtrag_eval.jsonl")
print(f"[run] turns -> {out_log}", flush=True)
n = run_domain(rag, domain, client, out_log, out_eval, k_retrieve=10, k_gen=5)
print(f"[run] {n} new turns processed", flush=True)

m = metrics_from_log(out_log, domain)
print(f"\n===== {system} / {domain} retrieval metrics (official qrels, Last-Turn) =====")
print(f"scored queries: {m['num_scored_queries']}")
for k in [1, 3, 5, 10]:
    print(f"  Recall@{k}: {m[f'Recall@{k}']:.4f}   nDCG@{k}: {m[f'nDCG@{k}']:.4f}")

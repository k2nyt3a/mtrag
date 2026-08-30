#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full-corpus build + run + eval for one system on one MTRAG domain.

  python run_full.py <hipporag|raptor> <domain> [openie_workers]
  # pilot (3 selected conversations, separate output dir, no overwrite):
  python run_full.py hipporag fiqa --conversation-ids <cid1> <cid2> <cid3> \
      --out-dir outputs_pilot/hipporag

Builds the OFFICIAL index over the FULL domain corpus (the correct BEIR retrieval
universe — verified: e.g. clapnq = all 183,408 passages), runs every domain turn
(retrieval query = official Last-Turn text; top-10 for metrics, top-5 for
generation), writes the per-turn log + MTRAG eval jsonl, and prints official-style
retrieval metrics (Recall@/nDCG@ 1,3,5,10). Resumable at both stages (index caches
+ log skip-set). When --conversation-ids / --num-conversations are given, only those
whole conversations run (boundaries preserved); index building is unchanged and, if
the cache already exists, loads without rebuilding (no OpenIE/embedding cost).
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.mtrag_io import load_env_openai_key, load_corpus          # noqa: E402
from common.runner import run_domain                                   # noqa: E402
from common.eval_retrieval import metrics_from_log                     # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("system", choices=["hipporag", "raptor"])
ap.add_argument("domain")
ap.add_argument("workers", nargs="?", type=int, default=32,
                help="OpenIE workers (HippoRAG build); positional, backward-compatible")
ap.add_argument("--conversation-ids", nargs="*", default=None,
                help="run only these conversation_ids (pilot); whole conversations only")
ap.add_argument("--num-conversations", type=int, default=None,
                help="run only the first N conversations in load order (pilot)")
ap.add_argument("--out-dir", default=None,
                help="override flat-log output dir (default outputs/<system>); use a "
                     "separate dir for pilots so full runs are never overwritten")
ap.add_argument("--conv-out-dir", default=None,
                help="also emit corrected conversation_NNN.json schema here "
                     "(conversations.json + per-conversation files)")
args = ap.parse_args()

system = args.system
domain = args.domain
workers = args.workers

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

out_base = args.out_dir or os.path.join(HERE, "outputs", system)
out_log = os.path.join(out_base, f"{domain}.log.jsonl")
out_eval = os.path.join(out_base, f"{domain}.mtrag_eval.jsonl")
print(f"[run] turns -> {out_log}", flush=True)
n = run_domain(rag, domain, client, out_log, out_eval, k_retrieve=10, k_gen=5,
               conversation_ids=args.conversation_ids,
               num_conversations=args.num_conversations,
               out_conv_dir=args.conv_out_dir)
print(f"[run] {n} new turns processed", flush=True)

m = metrics_from_log(out_log, domain)
print(f"\n===== {system} / {domain} retrieval metrics (official qrels, Last-Turn) =====")
print(f"scored queries: {m['num_scored_queries']}")
for k in [1, 3, 5, 10]:
    print(f"  Recall@{k}: {m[f'Recall@{k}']:.4f}   nDCG@{k}: {m[f'nDCG@{k}']:.4f}")

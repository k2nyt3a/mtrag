#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAPTOR plumbing smoke test (NOT a valid retrieval result).

Builds a RAPTOR tree over a SMALL clapnq corpus subset and runs the first clapnq
conversation through retrieve + the shared generator, to prove the official RAPTOR
pipeline works end-to-end (tree build, collapsed-tree retrieval, node provenance,
MTRAG eval record). Real numbers require the full-corpus tree.

Run inside raptor_env:
  ~/mtrag3/envs/raptor_env/bin/python smoke_raptor.py [SUBSET] [N_TURNS]
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common.mtrag_io import (load_env_openai_key, load_corpus, load_turns,
                             list_conversation_ids, make_log_record, to_mtrag_eval_record)
from common.generate import generate
from raptor_sys.raptor_adapter import build_tree, RaptorMTRAG

SUBSET = int(sys.argv[1]) if len(sys.argv) > 1 else 500
N_TURNS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

load_env_openai_key()
from openai import OpenAI
client = OpenAI()

print(f"[1] loading clapnq corpus subset (first {SUBSET} passages) ...", flush=True)
texts, ids = load_corpus("clapnq", limit=SUBSET)
print(f"    {len(texts)} leaf passages", flush=True)

cache = os.path.join(HERE, "..", "indices", "raptor", f"clapnq_smoke_{SUBSET}.pkl")
print("[2] building RAPTOR tree (clusters + LLM summaries) ...", flush=True)
t0 = time.time()
tree, leaf_id_map, n_leaves = build_tree(texts, ids, os.path.abspath(cache), client,
                                         tb_num_layers=3)
print(f"    built in {time.time()-t0:.1f}s: {n_leaves} leaves, {len(tree.all_nodes)} total nodes, "
      f"{tree.num_layers} layers", flush=True)

rag = RaptorMTRAG(tree, leaf_id_map, n_leaves, client, k=5)

cid = list_conversation_ids("clapnq")[0]
turns = [t for t in load_turns(conv_ids=[cid]) if int(t.turn) <= N_TURNS]
print(f"[3] running conversation {cid[:8]} — {len(turns)} turns\n", flush=True)

records = []
for t in turns:
    t1 = time.time()
    docs = rag.retrieve(t.question)
    ans = generate(client, t.history, t.question, [d["text"] for d in docs])
    lat = time.time() - t1
    print("=" * 90)
    print(f"RAPTOR | clapnq | conv {t.conversation_id[:8]} | turn {t.turn} "
          f"| type={t.question_type} | ans={t.answerability}")
    print(f"Q: {t.question}")
    if t.history:
        print(f"history: {len(t.history)} prior turns")
    print("Retrieved (collapsed tree — leaf/summary):")
    for d in docs:
        prov = f" src_docs={len(d['source_document_ids'])}" if d["raptor_node_type"] == "summary" else ""
        print(f"  {d['rank']}. [{d['raptor_node_type']} L{d['raptor_node_level']} "
              f"id={d['doc_id']}] score={d['score']:.3f}{prov}  {d['text'][:90]!r}")
    print(f"RAG answer : {ans[:300]}")
    print(f"MTRAG ref  : {t.reference_answer[:300]}")
    print(f"latency: {lat:.1f}s")
    records.append(make_log_record(t, "raptor", docs, ans, gold_doc_ids=[],
                                   gold_passages=[], latency_s=round(lat, 2)))

outp = os.path.join(HERE, "..", "outputs", f"raptor_smoke_clapnq_{cid[:8]}.jsonl")
with open(os.path.abspath(outp), "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\n[4] wrote {len(records)} log records -> {outp}")
print("NOTE: PLUMBING TEST on a corpus subset — retrieval is not representative.")

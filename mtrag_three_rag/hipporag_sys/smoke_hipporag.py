#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HippoRAG plumbing smoke test (small subset — NOT the full clapnq build).

Verifies the WHOLE official-HippoRAG chain end-to-end on a small corpus:
  MTRAG corpus subset -> HippoRAG official OpenIE index/graph
  -> MTRAG human question + correct history -> HippoRAG retrieval
  -> retrieved evidence carrying the ORIGINAL MTRAG document ids
  -> shared generator answer -> MTRAG eval record.

The subset = the first clapnq conversation's GOLD passages (from official qrels)
+ distractor passages from the corpus head, so retrieval can actually surface the
right evidence and we can confirm the id-mapping is correct. This is a plumbing
check; it is NOT a representative retrieval result.

Run inside hipporag_env (with env.sh sourced so caches go to D:):
  source /mnt/d/1\\ DynaicQA/jaist/mtrag_three_rag/env.sh
  WSLENV=OPENAI_API_KEY  "$HIPPORAG_ENV" smoke_hipporag.py [N_DISTRACTORS] [N_TURNS]
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common.mtrag_io import (load_env_openai_key, load_corpus, load_turns,
                             list_conversation_ids, load_qrels,
                             make_log_record, to_mtrag_eval_record)
from common.generate import generate
from hipporag_sys.hipporag_adapter import build_index

N_DISTRACTORS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
N_TURNS = int(sys.argv[2]) if len(sys.argv) > 2 else 4

load_env_openai_key()
from openai import OpenAI
client = OpenAI()

DOMAIN = "clapnq"
cid = list_conversation_ids(DOMAIN)[0]
turns = [t for t in load_turns(conv_ids=[cid]) if int(t.turn) <= N_TURNS]
qrels = load_qrels(DOMAIN)

# gold passage ids for this conversation's turns (from official qrels)
gold_ids = set()
for t in turns:
    gold_ids.update(qrels.get(t.task_id, {}).keys())
print(f"conversation {cid[:8]}: {len(turns)} turns, {len(gold_ids)} gold passage ids", flush=True)

# subset = gold passages + distractors from the corpus head (dedup, preserve corpus text)
all_texts, all_ids = load_corpus(DOMAIN)                    # full corpus (ids only cheap to scan)
id2text = dict(zip(all_ids, all_texts))
sub_ids, sub_texts, seen = [], [], set()
for gid in gold_ids:                                        # gold first
    if gid in id2text and gid not in seen:
        sub_ids.append(gid); sub_texts.append(id2text[gid]); seen.add(gid)
for i in range(len(all_ids)):                               # distractors
    if len(sub_ids) >= len(gold_ids) + N_DISTRACTORS:
        break
    if all_ids[i] not in seen:
        sub_ids.append(all_ids[i]); sub_texts.append(all_texts[i]); seen.add(all_ids[i])
print(f"smoke subset: {len(sub_ids)} passages "
      f"({len(gold_ids)} gold + {len(sub_ids)-len(gold_ids)} distractors)", flush=True)

save_dir = os.path.join(HERE, "..", "indices", "hipporag", f"clapnq_smoke_{len(sub_ids)}")
print("building official HippoRAG index (OpenIE graph) over the subset ...", flush=True)
t0 = time.time()
rag = build_index(sub_texts, sub_ids, os.path.abspath(save_dir))
rag.k = 5
print(f"index built in {time.time()-t0:.1f}s", flush=True)

records = []
for t in turns:
    t1 = time.time()
    docs = rag.retrieve(t.question)
    ans = generate(client, t.history, t.question, [d["text"] for d in docs])
    lat = time.time() - t1
    this_gold = set(qrels.get(t.task_id, {}).keys())
    hit = [d["doc_id"] for d in docs if d["doc_id"] in this_gold]
    print("=" * 90)
    print(f"HippoRAG | clapnq | conv {t.conversation_id[:8]} | turn {t.turn} "
          f"| type={t.question_type} | ans={t.answerability}")
    print(f"Q: {t.question}")
    if t.history:
        print(f"history: {len(t.history)} prior turns")
    print(f"Gold supporting doc ids ({len(this_gold)}): {sorted(this_gold)}")
    print("Retrieved (HippoRAG graph retrieval — original MTRAG doc ids):")
    for d in docs:
        star = " <== GOLD" if d["doc_id"] in this_gold else ""
        sc = f"{d['score']:.4f}" if d["score"] is not None else "None"
        print(f"  {d['rank']}. id={d['doc_id']} score={sc}{star}  {d['text'][:80]!r}")
    print(f"retrieval hit: {len(hit)}/{len(this_gold)} gold ids retrieved")
    print(f"RAG answer : {ans[:300]}")
    print(f"MTRAG ref  : {t.reference_answer[:300]}")
    print(f"latency: {lat:.1f}s")
    records.append(make_log_record(t, "hipporag", docs, ans,
                                   gold_doc_ids=sorted(this_gold),
                                   gold_passages=[id2text.get(g, "") for g in sorted(this_gold)],
                                   latency_s=round(lat, 2)))

outp = os.path.join(HERE, "..", "outputs", f"hipporag_smoke_clapnq_{cid[:8]}.jsonl")
os.makedirs(os.path.dirname(os.path.abspath(outp)), exist_ok=True)
with open(os.path.abspath(outp), "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
# also emit MTRAG eval-format records
evalp = os.path.join(HERE, "..", "outputs", f"hipporag_smoke_clapnq_{cid[:8]}.mtrag_eval.jsonl")
with open(os.path.abspath(evalp), "w", encoding="utf-8") as f:
    for t in turns:
        docs = None  # recomputed above per turn; re-derive from records
    for r, t in zip(records, turns):
        f.write(json.dumps(to_mtrag_eval_record(t, r["retrieved_docs"], r["rag_generated_answer"]),
                           ensure_ascii=False) + "\n")
print(f"\nwrote {len(records)} log records -> {outp}")
print(f"wrote MTRAG eval records -> {evalp}")
print("NOTE: PLUMBING smoke on a subset — not a representative retrieval result.")

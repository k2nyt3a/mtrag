#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Merge Contriever `ctxs` into the Self-RAG generation input.

Each output row: {id: task_id, question: history-folded question, ctxs: [{id,title,text,score}]}.
`question` is the generation input (history folded, per the uniform protocol);
`ctxs` are what Self-RAG's Contriever retrieved (its own evidence, NOT gold)."""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--turns", required=True)
ap.add_argument("--retrieved", required=True, help="passage_retrieval.py output (jsonl or json list) with ctxs")
ap.add_argument("--out", required=True)
ap.add_argument("--ndocs", type=int, default=5)
a = ap.parse_args()

folded = {}
for line in open(a.turns, encoding="utf-8"):
    r = json.loads(line)
    folded[r["task_id"]] = r["question"]

# passage_retrieval writes a json list (or jsonl) of {id/question, ctxs:[...]}
raw = open(a.retrieved, encoding="utf-8").read().strip()
try:
    retr = json.loads(raw)
    if isinstance(retr, dict):
        retr = [retr]
except json.JSONDecodeError:
    retr = [json.loads(l) for l in raw.splitlines() if l.strip()]

n = 0
with open(a.out, "w", encoding="utf-8") as w:
    for item in retr:
        tid = item.get("id") or item.get("_id")
        ctxs = []
        for c in (item.get("ctxs") or [])[:a.ndocs]:
            ctxs.append({"id": c.get("id"), "title": c.get("title", ""),
                          "text": c.get("text", ""), "score": c.get("score")})
        w.write(json.dumps({"id": tid, "question": folded.get(tid, item.get("question", "")),
                            "ctxs": ctxs}, ensure_ascii=False) + "\n")
        n += 1
print(f"wrote {n} self-rag input rows -> {a.out}")

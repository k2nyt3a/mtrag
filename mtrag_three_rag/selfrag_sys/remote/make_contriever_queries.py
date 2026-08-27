#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""<domain>_turns.jsonl -> Contriever query file: {id: task_id, question: last-turn}.
Retrieval query = the current user turn only (uniform Last-Turn setting)."""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--turns", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

rows = []
for line in open(a.turns, encoding="utf-8"):
    r = json.loads(line)
    rows.append({"id": r["task_id"], "question": r["retrieval_query"]})
json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
print(f"wrote {len(rows)} queries -> {a.out}")

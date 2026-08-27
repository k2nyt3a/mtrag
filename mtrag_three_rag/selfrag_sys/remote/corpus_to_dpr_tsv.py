#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MTRAG passage corpus (jsonl: _id/title/text) -> Self-RAG/DPR passages TSV.
Header row `id\ttext\ttitle`; tabs/newlines in text are spaced out. The `id` column
keeps the real MTRAG corpus _id so retrieved ids map straight to gold qrels."""
import argparse, json, csv, sys

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

n = 0
with open(a.corpus, encoding="utf-8") as f, open(a.out, "w", encoding="utf-8", newline="") as w:
    tw = csv.writer(w, delimiter="\t")
    tw.writerow(["id", "text", "title"])
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        pid = str(r.get("_id") or r.get("id") or "")
        title = (r.get("title") or "").replace("\t", " ").replace("\n", " ")
        text = (r.get("text") or "").replace("\t", " ").replace("\n", " ")
        tw.writerow([pid, text, title])
        n += 1
print(f"wrote {n} passages -> {a.out}", file=sys.stderr)

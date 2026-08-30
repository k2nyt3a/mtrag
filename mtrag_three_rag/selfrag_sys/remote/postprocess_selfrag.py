#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert official Self-RAG output into the SHARED schema used by RAPTOR/HippoRAG:
  * out_log  : per-turn log record (question, history, gold ids/passages, retrieved
               ids/passages/scores, Self-RAG answer, ...).
  * out_eval : official MTRAG eval JSONL (predictions=Self-RAG answer,
               targets=MTRAG reference, contexts=Contriever-retrieved passages).

Alignment: run_short_form.py writes a single JSON {"preds": [...]} in the SAME order
as the input rows (selfrag_in). We map input order -> task_id -> MTRAG Turn."""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))          # -> mtrag_three_rag
from common.mtrag_io import load_turns, load_qrels, load_corpus, make_log_record, to_mtrag_eval_record  # noqa
from selfrag_sys.extract import extract_selfrag_answer  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--turns", required=True)
ap.add_argument("--selfrag_in", required=True)
ap.add_argument("--selfrag_out", required=True)
ap.add_argument("--domain", required=True)
ap.add_argument("--out_log", required=True)
ap.add_argument("--out_eval", required=True)
a = ap.parse_args()

# ordered self-rag input rows (id + ctxs)
sin = [json.loads(l) for l in open(a.selfrag_in, encoding="utf-8") if l.strip()]
# self-rag output: single JSON with preds[] aligned to sin order
out = json.load(open(a.selfrag_out, encoding="utf-8"))
preds = out["preds"] if isinstance(out, dict) else out
assert len(preds) == len(sin), f"preds({len(preds)}) != inputs({len(sin)})"

turns = {t.task_id: t for t in load_turns(domains=[a.domain])}
qrels = load_qrels(a.domain)
corpus_texts, corpus_ids = load_corpus(a.domain)
id2text = dict(zip(corpus_ids, corpus_texts))

os.makedirs(os.path.dirname(a.out_log) or ".", exist_ok=True)
os.makedirs(os.path.dirname(a.out_eval) or ".", exist_ok=True)

flog = open(a.out_log, "w", encoding="utf-8")
feval = open(a.out_eval, "w", encoding="utf-8")
n = 0
for row, raw_ans in zip(sin, preds):
    tid = row["id"]
    turn = turns.get(tid)
    if turn is None:
        continue
    # Conservative extraction: strip role tags / echoed question / dialogue
    # continuations / Self-RAG reflection tokens. Semantic content (incl. genuine
    # mistakes) is preserved verbatim.
    ans = extract_selfrag_answer(raw_ans, question=row.get("question"))
    retrieved = [{"rank": i + 1, "doc_id": c.get("id"), "score": c.get("score"),
                  "score_type": "contriever", "text": c.get("text", ""),
                  "title": c.get("title", "")} for i, c in enumerate(row.get("ctxs", []))]
    gold_ids = list(qrels.get(tid, {}).keys())
    gold_passages = [id2text.get(g, "") for g in gold_ids]
    flog.write(json.dumps(make_log_record(turn, "selfrag", retrieved, ans,
                                          gold_ids, gold_passages,
                                          extra={"rag_answer_raw": raw_ans}), ensure_ascii=False) + "\n")
    feval.write(json.dumps(to_mtrag_eval_record(turn, retrieved, ans), ensure_ascii=False) + "\n")
    n += 1
flog.close(); feval.close()
print(f"wrote {n} records -> {a.out_log} + {a.out_eval}")

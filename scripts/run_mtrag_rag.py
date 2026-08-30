#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collect raw RAG conversation data over MTRAG-Human.

    python scripts/run_mtrag_rag.py --dataset fiqa --rag vector
        -> RAG/conversation/fiqa/vector/conversations.json
           RAG/conversation/fiqa/vector/run_metadata.json

Replays each MTRAG-Human conversation, turn by turn, through ONE tested RAG on that
dataset's OWN corpus. Preserves the exact original human questions, conversation
boundaries and turn order. Each RAG uses ITS OWN previous answers as history (not
the MTRAG reference answers). The MTRAG reference answer + gold evidence are stored
for evaluation only and are never given to the RAG (enforced by a safeguard).

This stage does NO failure analysis and touches no question-generation code.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from RAG import mtrag_data                                    # noqa: E402
from RAG.base import assert_no_gold_leak                      # noqa: E402
from RAG.registry import get_rag, RAG_NAMES                   # noqa: E402

DATASETS = list(mtrag_data.DATASETS)


def _select_conversations(convs, conv_ids, num, seed):
    if conv_ids:
        wanted = set(conv_ids)
        return [c for c in convs if c.conversation_id in wanted]
    if num is not None and num < len(convs):
        rng = random.Random(seed)
        # prefer multi-turn conversations for pilots, deterministic by seed
        multi = [c for c in convs if len(c.turns) > 1]
        pool = multi if len(multi) >= num else convs
        idx = sorted(rng.sample(range(len(pool)), num))
        return [pool[i] for i in idx]
    return convs


def _load_existing(out_path):
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _write_split(out_dir, results):
    """Write one conversation_NNN.json per conversation (legacy per-file layout).

    conversation_%03d.json[i] == conversations.json[i] (1-based, order preserved),
    so the array file and the split files stay byte-for-byte the same content.
    """
    for i, conv in enumerate(results, start=1):
        _write_json(os.path.join(out_dir, f"conversation_{i:03d}.json"), conv)


def list_availability(dataset, k):
    print(f"RAG availability for dataset={dataset}:")
    for name in RAG_NAMES:
        try:
            rag = get_rag(name, dataset, k=k)
            status = "working" if getattr(rag, "available", False) else "unavailable"
            reason = "" if rag.available else f"  ({rag.unavailable_reason})"
        except Exception as e:
            status = "unavailable"
            reason = f"  (build error: {type(e).__name__}: {str(e)[:100]})"
        print(f"  {name:10s}: {status}{reason}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--rag", choices=RAG_NAMES)
    ap.add_argument("--conversation-ids", nargs="*", default=None,
                    help="explicit MTRAG conversation_ids to run")
    ap.add_argument("--num-conversations", type=int, default=None,
                    help="sample this many conversations (multi-turn preferred), seeded")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="cap turns per conversation (debug); default = all turns")
    ap.add_argument("--retrieval-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="skip conversations already present in the output file")
    ap.add_argument("--output", default=None,
                    help="override output dir (default RAG/conversation/<dataset>/<rag>)")
    ap.add_argument("--list-availability", action="store_true",
                    help="probe every RAG for the dataset and exit")
    args = ap.parse_args()

    if args.list_availability:
        if not args.dataset:
            ap.error("--list-availability requires --dataset")
        list_availability(args.dataset, args.retrieval_k)
        return
    if not args.dataset or not args.rag:
        ap.error("--dataset and --rag are required")

    dataset, rag_name, k = args.dataset, args.rag, args.retrieval_k
    out_dir = args.output or os.path.join(REPO, "RAG", "conversation", dataset, rag_name)
    out_conv = os.path.join(out_dir, "conversations.json")
    out_meta = os.path.join(out_dir, "run_metadata.json")

    # --- build the tested RAG; refuse to substitute if unavailable ---
    print(f"[build] {rag_name} on {dataset} (k={k}) ...", flush=True)
    rag = get_rag(rag_name, dataset, k=k)
    if not getattr(rag, "available", False):
        print(f"\n!! RAG '{rag_name}' is UNAVAILABLE for dataset '{dataset}':\n   "
              f"{rag.unavailable_reason}\n   No substitute is used. Nothing written.",
              file=sys.stderr)
        sys.exit(2)

    # --- MTRAG data (correct corpus for this dataset) ---
    convs = mtrag_data.load_conversations(dataset)
    convs = _select_conversations(convs, args.conversation_ids, args.num_conversations, args.seed)
    # id -> gold evidence text (for storage + leakage guard); uses the SAME corpus
    ctexts, cids, _ = mtrag_data.load_corpus(dataset)
    id2text = dict(zip(cids, ctexts))

    existing = _load_existing(out_conv) if args.resume else []
    done_ids = {c["conversation_id"] for c in existing} if args.resume else set()
    results = list(existing)

    t_start = time.time()
    n_turns = 0
    for conv in convs:
        if conv.conversation_id in done_ids:
            continue
        turns = conv.turns[: args.max_turns] if args.max_turns else conv.turns
        rag_history = []          # THIS RAG's own conversation so far
        turn_records = []
        for t in turns:
            question = t.question                     # exact original human question
            history_before = [dict(m) for m in rag_history]
            retrieval_query = question                # MTRAG last-turn protocol (raw question)

            docs = rag.retrieve(retrieval_query, k)
            gen = rag.generate(rag_history, question, docs)
            answer = gen["answer"]

            # ---- gold-leakage safeguard ----
            ev_texts = [id2text.get(eid, "") for eid in t.reference_evidence_ids]
            input_texts = ([m["content"] for m in history_before] + [question]
                           + [d.get("text", "") for d in docs]
                           + [d.get("text", "") for d in gen.get("final_context_documents", [])])
            assert_no_gold_leak(input_texts, t.reference_answer, ev_texts,
                                retrieved_texts=[d.get("text", "") for d in docs])

            turn_records.append({
                "turn_id": t.turn,
                "task_id": t.task_id,
                "question": question,
                "rag_history_before_turn": history_before,
                "retrieval": {
                    "query": retrieval_query,
                    "rewritten_query": None,
                    "mtrag_official_lastturn_query": t.lastturn_query,
                    "k": k,
                    "retrieved_documents": docs,
                    "intermediate": gen.get("intermediate", {}),
                    "final_context_documents": gen.get("final_context_documents", []),
                },
                "generation": {
                    "model": gen.get("model"),
                    "messages": gen.get("messages"),
                    "parameters": gen.get("parameters", {}),
                },
                "rag_answer": answer,
                # top-level alias of the generation context (legacy schema);
                # kept identical to retrieval.final_context_documents.
                "documents_used": gen.get("final_context_documents", []),
                "reference": {
                    "answer": t.reference_answer,
                    "evidence_ids": t.reference_evidence_ids,
                    "evidence": [{"doc_id": eid, "text": id2text.get(eid, "")}
                                 for eid in t.reference_evidence_ids],
                },
                "mtrag_metadata": {
                    "answerability": t.answerability,
                    "question_type": t.question_type,
                    "multi_turn": t.multi_turn,
                },
            })
            # this RAG experiences ITS OWN answer as the next turn's history
            rag_history.append({"role": "user", "content": question})
            rag_history.append({"role": "assistant", "content": answer})
            n_turns += 1

        results.append({
            "conversation_id": conv.conversation_id,
            "dataset": dataset,
            "collection": conv.collection,
            "rag": rag_name,
            "num_turns": len(turn_records),
            "turns": turn_records,
        })
        _write_json(out_conv, results)           # incremental / resumable checkpoint
        _write_split(out_dir, results)           # per-conversation files (legacy layout)
        print(f"  {dataset}/{rag_name}: conv {conv.conversation_id} "
              f"({len(turn_records)} turns) done", flush=True)

    meta = {
        "dataset": dataset,
        "rag": rag_name,
        "rag_info": rag.info(),
        "num_conversations": len(results),
        "num_turns_this_run": n_turns,
        "retrieval_k": k,
        "seed": args.seed,
        "selection": {
            "conversation_ids": args.conversation_ids,
            "num_conversations": args.num_conversations,
            "max_turns": args.max_turns,
        },
        "retrieval_query_policy": "raw current human question (MTRAG last-turn protocol; no rewrite)",
        "history_policy": "tested RAG's own previous answers (no reference leakage)",
        "corpus": mtrag_data.corpus_path(dataset),
        "mtrag_tasks": mtrag_data.TASKS_PATH,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(time.time() - t_start, 1),
    }
    _write_json(out_meta, meta)
    print(f"\n[done] {len(results)} conversations, {n_turns} new turns -> {out_conv}")


if __name__ == "__main__":
    main()

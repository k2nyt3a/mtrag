#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared per-turn runner for RAPTOR / HippoRAG over a full MTRAG domain index.

Retrieval query = official Last-Turn query text (keyed by task_id) -> retrieve
top-k_retrieve (for R@k/nDCG@k). Generation = shared generator over top-k_gen
passages + SELF-THREADED history.

SELF-THREADED HISTORY (the experiment's required protocol): each conversation is
executed sequentially, turn by turn. Turn N's generation history is
``prev user questions + this RAG system's OWN actual previous answers`` (built as
we go), NEVER the MTRAG canonical/gold assistant answers (``turn.history``). If the
RAG makes a mistake at an earlier turn, that actual erroneous answer stays in the
history the RAG sees on later turns -- we do NOT silently correct it with gold.

``turn.history`` (MTRAG gold assistant answers) is still read from the tasks and is
still recorded in the log as ``conversation_history`` for gold/reference/judging,
but it is NEVER fed into generation.

Resumable at CONVERSATION granularity: a conversation is skipped only if ALL of its
task_ids are already present in the log. A conversation can never be resumed
mid-way, because later turns depend on this run's own earlier answers; a partially
logged conversation is re-executed in full."""
from __future__ import annotations
import json
import os
import time
from itertools import groupby
from typing import Dict, List

from .mtrag_io import load_turns, load_qrels, load_corpus, make_log_record, to_mtrag_eval_record
from .eval_retrieval import load_lastturn_queries
from .generate import generate, build_messages
from . import conv_schema


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


def _select_conversations(turns, conversation_ids, num_conversations):
    """Restrict a flat, conversation-ordered turn list to selected conversations.

    Whole conversations only (boundaries preserved). Default (both None) = no filter,
    so existing full-domain behavior is unchanged. Deterministic: no RNG/seed. When
    ``conversation_ids`` is given it wins; otherwise ``num_conversations`` keeps the
    first N conversations in load order.
    """
    if conversation_ids:
        wanted = set(conversation_ids)
        return [t for t in turns if t.conversation_id in wanted]
    if num_conversations is not None:
        order = []
        for t in turns:
            if t.conversation_id not in order:
                order.append(t.conversation_id)
        keep = set(order[:num_conversations])
        return [t for t in turns if t.conversation_id in keep]
    return turns


def run_domain(system, domain: str, client, out_log: str, out_eval: str,
               k_retrieve: int = 10, k_gen: int = 5, verbose: bool = True,
               conversation_ids=None, num_conversations=None, out_conv_dir=None):
    """Run a domain self-threaded. Always writes the flat log + MTRAG eval jsonl.

    When ``out_conv_dir`` is set (used for pilots), ALSO emits the corrected
    ``conversation_NNN.json`` schema there via the shared ``conv_schema`` emitter, so
    HippoRAG/RAPTOR pilots land in the same final schema as Vector/Self-RAG.
    """
    system.k = k_retrieve
    lastturn = load_lastturn_queries(domain)
    qrels = load_qrels(domain)
    corpus_texts, corpus_ids = load_corpus(domain)
    id2text = dict(zip(corpus_ids, corpus_texts))
    turns = load_turns(domains=[domain])       # already sorted by (conversation_id, turn)
    turns = _select_conversations(turns, conversation_ids, num_conversations)  # pilot subset

    os.makedirs(os.path.dirname(out_log) or ".", exist_ok=True)
    done = _done_task_ids(out_log)
    flog = open(out_log, "a", encoding="utf-8")
    feval = open(out_eval, "a", encoding="utf-8")
    conversations: List[Dict] = []             # corrected-schema accumulator (out_conv_dir)
    n = 0
    # Group turns into conversations and execute each conversation sequentially so
    # that history is self-threaded from this run's OWN answers.
    for conv_id, conv_iter in groupby(turns, key=lambda x: x.conversation_id):
        conv_turns = list(conv_iter)
        # conversation-granular resume: skip only if every turn is already logged
        if all(t.task_id in done for t in conv_turns):
            continue
        rag_history: List[Dict] = []           # THIS RAG's own conversation so far (reset per conv)
        turn_records: List[Dict] = []
        for t in conv_turns:
            q_retr = lastturn.get(t.task_id, t.question)      # official last-turn query text
            history_before = [dict(m) for m in rag_history]
            t0 = time.time()
            docs = system.retrieve(q_retr)                     # top-k_retrieve
            gen_docs = docs[:k_gen]
            # generation history = self-threaded (own prior answers), NOT t.history (gold)
            ans = generate(client, rag_history, t.question, [d["text"] for d in gen_docs])
            lat = round(time.time() - t0, 2)
            gold = list(qrels.get(t.task_id, {}).keys())
            rec = make_log_record(t, system.name, docs, ans, gold_doc_ids=gold,
                                  gold_passages=[id2text.get(g, "") for g in gold],
                                  latency_s=lat,
                                  extra={"retrieval_query": q_retr,
                                         # the self-threaded history actually seen by the RAG
                                         "rag_history_before_turn": history_before})
            flog.write(json.dumps(rec, ensure_ascii=False) + "\n"); flog.flush()
            feval.write(json.dumps(to_mtrag_eval_record(t, gen_docs, ans), ensure_ascii=False) + "\n"); feval.flush()
            if out_conv_dir:
                retrieval = {
                    "query": q_retr, "rewritten_query": None,
                    "mtrag_official_lastturn_query": q_retr, "k": k_retrieve,
                    "retrieved_documents": docs, "intermediate": {},
                    "final_context_documents": gen_docs,
                }
                generation = {
                    "model": "gpt-4o-mini",
                    "messages": build_messages(history_before, t.question,
                                               [d["text"] for d in gen_docs]),
                    "parameters": {"temperature": 0.0, "max_tokens": 512},
                }
                reference = {
                    "answer": t.reference_answer,
                    "evidence_ids": gold,
                    "evidence": [{"doc_id": g, "text": id2text.get(g, "")} for g in gold],
                }
                mtrag_metadata = {"answerability": t.answerability,
                                  "question_type": t.question_type,
                                  "multi_turn": t.multi_turn}
                turn_records.append(conv_schema.build_turn_record(
                    t.turn, t.task_id, t.question, history_before, retrieval,
                    generation, ans, reference, mtrag_metadata))
            # the RAG experiences ITS OWN actual answer as the next turn's history
            rag_history.append({"role": "user", "content": t.question})
            rag_history.append({"role": "assistant", "content": ans})
            n += 1
            if verbose and n % 10 == 0:
                print(f"  {system.name}/{domain}: {n} turns done", flush=True)
        if out_conv_dir:
            conversations.append(conv_schema.assemble_conversation(
                conv_id, domain, conv_turns[0].collection, system.name, turn_records))
            conv_schema.write_conversations(out_conv_dir, conversations)
    flog.close(); feval.close()
    return n

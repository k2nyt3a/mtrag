#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Self-RAG SEQUENTIAL, self-threaded per-conversation driver (the smallest change that
makes Self-RAG obey the experiment's history protocol).

WHY THIS EXISTS
---------------
The official Self-RAG remote pipeline is batch: every MTRAG turn is generated
independently, and its generation input was folded from the MTRAG *gold* assistant
answers. The experiment requires each turn to instead see this system's OWN actual
previous answers:

    turn 1: history = []                                  -> rag_answer_1
    turn 2: history = [q1, rag_answer_1]                  -> rag_answer_2
    turn 3: history = [q1, rag_answer_1, q2, rag_answer_2]-> rag_answer_3
    ...     (reset history = [] at each conversation boundary)

WHAT STAYS UNCHANGED (reused, not redesigned)
---------------------------------------------
* RETRIEVAL is history-INDEPENDENT (query = current user turn only), so the existing
  batched Contriever retrieval (make_contriever_queries.py -> passage_retrieval.py)
  is reused verbatim. This driver consumes its per-task_id `ctxs`; it introduces NO
  history-dependent retrieval and NO query rewriting.
* GENERATION INPUT folding reuses ``build_selfrag_generation_input``.
* ANSWER cleanup reuses ``extract_selfrag_answer``.
* OUTPUT uses the corrected ``conversation_NNN.json`` schema (same turn keys as
  scripts/run_mtrag_rag.py); the array + per-conversation split are written with that
  script's ``_write_json`` / ``_write_split`` emitter.

Only the GENERATION step becomes sequential (it must, because turn N's folded input
contains turns 1..N-1's actual model answers, which only exist once generated).

MODEL BOUNDARY
--------------
The driver is model-agnostic: it takes a ``generate_fn(folded_question, ctxs) -> raw``
callable. On the GPU box this wraps the official ``run_short_form`` decode (the trained
selfrag_llama2_7b via vLLM, loaded ONCE) with the exact paper params. In tests it is a
stub. Nothing about Self-RAG's method is reimplemented here.
"""
from __future__ import annotations

import os
import sys
from itertools import groupby
from typing import Callable, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORK = os.path.abspath(os.path.join(_HERE, ".."))        # -> mtrag_three_rag
if _WORK not in sys.path:
    sys.path.insert(0, _WORK)

from common.mtrag_io import load_turns, load_qrels, load_corpus   # noqa: E402
from common import conv_schema                                    # noqa: E402
from selfrag_sys.prepare_selfrag_inputs import build_selfrag_generation_input  # noqa: E402
from selfrag_sys.extract import extract_selfrag_answer            # noqa: E402

GEN_MODEL = "selfrag/selfrag_llama2_7b"
# paper short-form adaptive-retrieval params (recorded for provenance; the real decode
# lives in run_short_form on the box).
DEFAULT_PARAMS = {
    "max_new_tokens": 300, "ndocs": 5, "mode": "adaptive_retrieval",
    "threshold": 0.2, "use_groundness": True, "use_utility": True,
    "use_seqscore": True, "w_rel": 1.0, "w_sup": 1.0, "w_use": 0.5, "dtype": "half",
}

GenerateFn = Callable[[str, List[Dict]], str]     # (folded_question, ctxs) -> raw answer


def _ctx_to_retrieved(ctxs: List[Dict], k: int) -> List[Dict]:
    out = []
    for i, c in enumerate(ctxs[:k]):
        out.append({"rank": i + 1, "doc_id": c.get("id") or c.get("doc_id"),
                    "score": c.get("score"), "score_type": "contriever",
                    "title": c.get("title", ""), "text": c.get("text", "")})
    return out


def run_conversation(conv_turns: List, ctxs_by_task_id: Dict[str, List[Dict]],
                     generate_fn: GenerateFn, k_gen: int = 5,
                     qrels: Optional[Dict] = None, id2text: Optional[Dict] = None,
                     params: Optional[Dict] = None) -> List[Dict]:
    """Run ONE conversation sequentially; return corrected-schema turn records.

    ``conv_turns`` are mtrag_io.Turn objects for a single conversation, in turn order.
    """
    qrels = qrels or {}
    id2text = id2text or {}
    params = params or DEFAULT_PARAMS
    rag_history: List[Dict] = []              # this system's OWN answers (reset per conv)
    records: List[Dict] = []
    for t in conv_turns:
        question = t.question                 # exact current human question
        history_before = [dict(m) for m in rag_history]
        # generation input = SELF-THREADED history + current question (NOT gold history)
        folded = build_selfrag_generation_input(rag_history, question)
        # retrieval is history-independent (current turn only) -> reuse batched ctxs
        ctxs = ctxs_by_task_id.get(t.task_id, [])
        retrieved = _ctx_to_retrieved(ctxs, k_gen)

        raw = generate_fn(folded, ctxs[:k_gen])            # actual Self-RAG decode
        answer = extract_selfrag_answer(raw, question=folded)

        gold_ids = list(qrels.get(t.task_id, {}).keys())
        retrieval = {
            "query": question,                          # current turn only
            "rewritten_query": None,
            "mtrag_official_lastturn_query": None,
            "k": k_gen,
            "retrieved_documents": retrieved,
            "intermediate": {"note": "official Contriever ctxs (batched, history-independent)"},
            "final_context_documents": retrieved,
        }
        generation = {
            "model": GEN_MODEL,
            "messages": [{"role": "user", "content": folded}],
            "parameters": params,
            "raw_answer": raw,
        }
        reference = {
            "answer": t.reference_answer,
            "evidence_ids": gold_ids,
            "evidence": [{"doc_id": g, "text": id2text.get(g, "")} for g in gold_ids],
        }
        mtrag_metadata = {"answerability": t.answerability,
                          "question_type": t.question_type,
                          "multi_turn": t.multi_turn}
        records.append(conv_schema.build_turn_record(
            t.turn, t.task_id, question, history_before, retrieval, generation,
            answer, reference, mtrag_metadata))
        # the RAG experiences ITS OWN actual answer as the next turn's history
        rag_history.append({"role": "user", "content": question})
        rag_history.append({"role": "assistant", "content": answer})
    return records


def run_domain(domain: str, ctxs_by_task_id: Dict[str, List[Dict]],
               generate_fn: GenerateFn, k_gen: int = 5,
               conversation_ids: Optional[List[str]] = None,
               params: Optional[Dict] = None) -> List[Dict]:
    """Run every (or selected) conversation in a domain; return the conversations list
    in the corrected ``conversation_NNN.json`` schema."""
    turns = load_turns(domains=[domain])          # sorted by (conversation_id, turn)
    if conversation_ids:
        wanted = set(conversation_ids)
        turns = [t for t in turns if t.conversation_id in wanted]
    qrels = load_qrels(domain)
    corpus_texts, corpus_ids = load_corpus(domain)
    id2text = dict(zip(corpus_ids, corpus_texts))

    conversations: List[Dict] = []
    for conv_id, it in groupby(turns, key=lambda x: x.conversation_id):
        conv_turns = list(it)
        recs = run_conversation(conv_turns, ctxs_by_task_id, generate_fn, k_gen=k_gen,
                                qrels=qrels, id2text=id2text, params=params)
        conversations.append({
            "conversation_id": conv_id,
            "dataset": domain,
            "collection": conv_turns[0].collection,
            "rag": "selfrag",
            "num_turns": len(recs),
            "turns": recs,
        })
    return conversations


# ---- box entry point (loads real Contriever ctxs + real Self-RAG generate_fn) ------
def _load_ctxs(selfrag_in_path: str) -> Dict[str, List[Dict]]:
    """Reuse the existing batched retrieval output (selfrag_in/<D>.jsonl rows:
    {id, question(folded-old), ctxs[...]}). We take only id -> ctxs; the folded
    question there is IGNORED (we re-fold with self-threaded history)."""
    import json
    out: Dict[str, List[Dict]] = {}
    for line in open(selfrag_in_path, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r["id"]] = r.get("ctxs", [])
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--selfrag_in", required=True,
                    help="batched retrieval output (selfrag_in/<D>.jsonl) -> id/ctxs")
    ap.add_argument("--out_dir", required=True,
                    help="output dir; writes conversations.json + conversation_NNN.json")
    ap.add_argument("--conversation-ids", nargs="*", default=None)
    ap.add_argument("--k-gen", type=int, default=5)
    args = ap.parse_args()

    # The real Self-RAG model call (trained 7B via vLLM, loaded ONCE) is provided here
    # on the GPU box. It MUST reuse run_short_form's formatting + DEFAULT_PARAMS decode.
    from selfrag_sys.selfrag_model import load_generate_fn      # box-only; see report
    generate_fn = load_generate_fn(GEN_MODEL, DEFAULT_PARAMS)

    conversations = run_domain(args.domain, _load_ctxs(args.selfrag_in), generate_fn,
                               k_gen=args.k_gen, conversation_ids=args.conversation_ids)

    # reuse the shared corrected emitter (array + per-conversation split, documents_used alias)
    conv_schema.write_conversations(args.out_dir, conversations)
    print(f"[selfrag seq] {len(conversations)} conversations -> {args.out_dir}")


if __name__ == "__main__":
    main()

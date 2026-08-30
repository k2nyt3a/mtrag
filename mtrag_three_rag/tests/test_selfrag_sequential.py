#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression test for the Self-RAG SEQUENTIAL self-threaded driver
(selfrag_sys/run_selfrag_sequential.py).

Uses a STUB model whose answer is deliberately different from the MTRAG gold answer,
and proves that turn N+1's Self-RAG generation input contains the previous ACTUAL
model answer -- NOT the reference/gold answer. Also checks the conversation-boundary
reset and that retrieval stays the current-question-only ctxs (history-independent).

Run:  python mtrag_three_rag/tests/test_selfrag_sequential.py   (or pytest)
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.abspath(os.path.join(HERE, ".."))          # -> mtrag_three_rag
if WORK not in sys.path:
    sys.path.insert(0, WORK)

from common.mtrag_io import Turn                            # noqa: E402
from selfrag_sys import run_selfrag_sequential as seq       # noqa: E402

GOLD = {"S::1": "GOLD_1", "S::2": "GOLD_2", "S::3": "GOLD_3", "T::1": "GOLD_T1"}
STUB = {"S::1": "STUB_ANSWER_1", "S::2": "STUB_ANSWER_2", "S::3": "STUB_ANSWER_3",
        "T::1": "STUB_ANSWER_T1"}
Q = {"S::1": "QS1", "S::2": "QS2", "S::3": "QS3", "T::1": "QT1"}


def _turn(conv, tid, turn):
    return Turn(conversation_id=conv, task_id=tid, turn=str(turn), domain="d",
                collection="c", question=Q[tid], history=[],  # gold history intentionally unused
                reference_answer=GOLD[tid])


def _run():
    turns = [_turn("S", "S::1", 1), _turn("S", "S::2", 2), _turn("S", "S::3", 3),
             _turn("T", "T::1", 1)]
    # ctxs are the SAME regardless of history (retrieval = current turn only)
    ctxs_by_tid = {tid: [{"id": f"doc_{tid}", "text": f"passage for {tid}", "score": 1.0}]
                   for tid in Q}

    seen_inputs = []   # (task order) folded generation inputs the stub received

    def stub_generate(folded_question, ctxs):
        seen_inputs.append(folded_question)
        # map the current question in the folded input back to its task_id
        tid = next(t for t in Q if Q[t] in folded_question and folded_question.rstrip().endswith(Q[t]))
        # add role artifacts to also exercise extraction
        return f"Assistant: {STUB[tid]}"

    # patch the three IO deps so no corpus/qrels/tasks files are needed
    orig = {}
    for name, val in {"load_turns": lambda domains=None: turns,
                      "load_qrels": lambda domain: {},
                      "load_corpus": lambda domain: ([], [])}.items():
        orig[name] = getattr(seq, name)
        setattr(seq, name, val)
    try:
        convs = seq.run_domain("d", ctxs_by_tid, stub_generate, k_gen=5)
    finally:
        for name, val in orig.items():
            setattr(seq, name, val)
    return convs, seen_inputs


def test_selfrag_sequential_self_threading():
    convs, seen = _run()
    by_id = {c["conversation_id"]: c for c in convs}
    S = by_id["S"]["turns"]

    # gold really differs from the stub answers
    for tid in GOLD:
        assert STUB[tid] != GOLD[tid]

    # extracted answers are the stub answers (role tag stripped, not gold)
    assert [t["rag_answer"] for t in S] == ["STUB_ANSWER_1", "STUB_ANSWER_2", "STUB_ANSWER_3"]

    # history invariant on the emitted records
    assert S[0]["rag_history_before_turn"] == []
    assert S[1]["rag_history_before_turn"] == [
        {"role": "user", "content": "QS1"},
        {"role": "assistant", "content": "STUB_ANSWER_1"}]
    assert S[2]["rag_history_before_turn"] == [
        {"role": "user", "content": "QS1"},
        {"role": "assistant", "content": "STUB_ANSWER_1"},
        {"role": "user", "content": "QS2"},
        {"role": "assistant", "content": "STUB_ANSWER_2"}]

    # THE KEY PROOF: turn N+1's generation input contains the actual prior answer,
    # never the gold answer.
    #   seen[0] = turn S1 input, seen[1] = S2, seen[2] = S3, seen[3] = T1
    assert "STUB_ANSWER_1" in seen[1] and "GOLD_1" not in seen[1]
    assert "STUB_ANSWER_1" in seen[2] and "STUB_ANSWER_2" in seen[2]
    assert "GOLD_1" not in seen[2] and "GOLD_2" not in seen[2]

    # gold answers never appear in ANY emitted history
    for c in convs:
        for t in c["turns"]:
            blob = str(t["rag_history_before_turn"])
            for g in GOLD.values():
                assert g not in blob

    # conversation boundary: T never sees S's history
    assert by_id["T"]["turns"][0]["rag_history_before_turn"] == []
    assert "STUB_ANSWER_1" not in seen[3] and "QS1" not in seen[3]

    # retrieval stayed current-question-only ctxs (history-independent)
    assert S[1]["retrieval"]["retrieved_documents"][0]["doc_id"] == "doc_S::2"
    assert S[1]["retrieval"]["query"] == "QS2"
    print("[ok] selfrag_sequential_self_threading")


TESTS = [test_selfrag_sequential_self_threading]

if __name__ == "__main__":
    fails = 0
    for t in TESTS:
        try:
            t()
        except AssertionError as e:
            fails += 1
            print(f"[FAIL] {t.__name__}: {e}")
    print(f"\n{len(TESTS) - fails}/{len(TESTS)} test functions passed")
    sys.exit(1 if fails else 0)

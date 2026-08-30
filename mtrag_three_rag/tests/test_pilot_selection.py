#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for pilot conversation-selection in the HippoRAG/RAPTOR shared
runner (common/runner.run_domain) and that retrieval stays current-question-based.

Run:  python mtrag_three_rag/tests/test_pilot_selection.py   (or pytest)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.abspath(os.path.join(HERE, ".."))
if WORK not in sys.path:
    sys.path.insert(0, WORK)

from common import runner                                    # noqa: E402
from common.mtrag_io import Turn                             # noqa: E402


def _turns():
    """3 conversations: A(2 turns), B(2 turns), C(1 turn)."""
    def mk(conv, n, q):
        return Turn(conversation_id=conv, task_id=f"{conv}::{n}", turn=str(n), domain="d",
                    collection="c", question=q, history=[], reference_answer=f"GOLD_{conv}{n}",
                    raw={"input": [{"speaker": "user", "text": q}],
                         "targets": [{"text": f"GOLD_{conv}{n}"}]})
    return [mk("A", 1, "qa1"), mk("A", 2, "qa2"),
            mk("B", 1, "qb1"), mk("B", 2, "qb2"),
            mk("C", 1, "qc1")]


class _Stub:
    name = "stub"; k = 10
    def __init__(self): self.retr_queries = []
    def retrieve(self, query):
        self.retr_queries.append(query)
        return [{"rank": 1, "doc_id": "d1", "score": 1.0, "text": "p"}]


def _run(conversation_ids=None, num_conversations=None):
    turns = _turns()
    stub = _Stub()
    orig = {}
    for name, val in {"load_turns": lambda domains=None: turns,
                      "load_qrels": lambda domain: {},
                      "load_corpus": lambda domain: ([], []),
                      "load_lastturn_queries": lambda domain: {},   # -> query falls back to question
                      "generate": lambda client, history, question, passages: f"ANS_{question}"}.items():
        orig[name] = getattr(runner, name)
        setattr(runner, name, val)
    try:
        with tempfile.TemporaryDirectory() as td:
            log = os.path.join(td, "l.jsonl"); ev = os.path.join(td, "e.jsonl")
            runner.run_domain(stub, "d", client=None, out_log=log, out_eval=ev,
                              verbose=False, conversation_ids=conversation_ids,
                              num_conversations=num_conversations)
            recs = [json.loads(l) for l in open(log, encoding="utf-8") if l.strip()]
    finally:
        for name, val in orig.items():
            setattr(runner, name, val)
    return recs, stub


def test_select_conversation_ids_runs_only_those():
    recs, stub = _run(conversation_ids=["A", "C"])
    convs = {r["conversation_id"] for r in recs}
    assert convs == {"A", "C"}, convs
    task_ids = {r["task_id"] for r in recs}
    assert task_ids == {"A::1", "A::2", "C::1"}, task_ids
    # retrieval query = current question only (history-independent)
    assert stub.retr_queries == ["qa1", "qa2", "qc1"]
    print("[ok] select_conversation_ids_runs_only_those")


def test_num_conversations_takes_first_n():
    recs, _ = _run(num_conversations=2)
    assert {r["conversation_id"] for r in recs} == {"A", "B"}
    print("[ok] num_conversations_takes_first_n")


def test_no_flag_preserves_full_run():
    recs, _ = _run()
    assert {r["conversation_id"] for r in recs} == {"A", "B", "C"}
    assert len(recs) == 5
    print("[ok] no_flag_preserves_full_run")


def test_selfrag_model_wrapper_lazy_and_importable():
    """Wrapper imports on CPU; asks clearly for the upstream repo when absent."""
    from selfrag_sys import selfrag_model
    assert callable(selfrag_model.load_generate_fn)
    os.environ.pop("SELFRAG_REPO", None)
    try:
        selfrag_model.load_generate_fn("selfrag/selfrag_llama2_7b", {})
        raised = False
    except RuntimeError as e:
        raised = "self-rag" in str(e).lower() or "selfrag_repo" in str(e).lower()
    assert raised, "expected a clear RuntimeError pointing at the upstream repo"
    print("[ok] selfrag_model_wrapper_lazy_and_importable")


TESTS = [test_select_conversation_ids_runs_only_those, test_num_conversations_takes_first_n,
         test_no_flag_preserves_full_run, test_selfrag_model_wrapper_lazy_and_importable]

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

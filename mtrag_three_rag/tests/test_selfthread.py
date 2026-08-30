#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for the SELF-THREADED conversation-history protocol and the
Self-RAG answer extraction fix.

These prove, with NO GPU / model / corpus, that:
  * HippoRAG/RAPTOR shared runner (common/runner.run_domain) threads each
    conversation on its OWN actual previous answers, resets at conversation
    boundaries, and NEVER substitutes the MTRAG gold answers;
  * a genuinely wrong RAG answer stays in the history the RAG sees later;
  * Self-RAG output extraction strips mechanical artifacts without altering
    semantic content (a wrong answer stays wrong).

Run directly:  python mtrag_three_rag/tests/test_selfthread.py
or with pytest.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.abspath(os.path.join(HERE, ".."))          # -> mtrag_three_rag
if WORK not in sys.path:
    sys.path.insert(0, WORK)

from common import runner                                  # noqa: E402
from common.mtrag_io import Turn                           # noqa: E402
from selfrag_sys.extract import extract_selfrag_answer     # noqa: E402

# ---- synthetic fixtures ----------------------------------------------------
GOLD = {  # task_id -> MTRAG gold/reference answer
    "A::1": "GOLD_ANSWER_1", "A::2": "GOLD_ANSWER_2", "A::3": "GOLD_ANSWER_3",
    "B::1": "GOLD_ANSWER_B1", "B::2": "GOLD_ANSWER_B2",
}
STUB = {  # task_id -> what the stub RAG actually answers (deliberately != gold)
    "A::1": "STUB_RAG_ANSWER_1", "A::2": "STUB_RAG_ANSWER_2", "A::3": "STUB_RAG_ANSWER_3",
    "B::1": "STUB_RAG_ANSWER_B1", "B::2": "STUB_RAG_ANSWER_B2",
}
QUESTION = {
    "A::1": "QA1", "A::2": "QA2", "A::3": "QA3",
    "B::1": "QB1", "B::2": "QB2",
}


def _mk_turn(conv, task_id, turn, gold_history):
    q = QUESTION[task_id]
    raw = {"input": [{"speaker": "user", "text": q}],
           "targets": [{"text": GOLD[task_id]}]}
    return Turn(conversation_id=conv, task_id=task_id, turn=str(turn), domain="testdom",
                collection="c", question=q, history=gold_history,
                reference_answer=GOLD[task_id], raw=raw)


def _synthetic_turns():
    # conversation A: 3 turns; each turn's .history is the GOLD history (old wrong protocol)
    a1 = _mk_turn("A", "A::1", 1, [])
    a2 = _mk_turn("A", "A::2", 2, [{"role": "user", "content": "QA1"},
                                   {"role": "assistant", "content": GOLD["A::1"]}])
    a3 = _mk_turn("A", "A::3", 3, [{"role": "user", "content": "QA1"},
                                   {"role": "assistant", "content": GOLD["A::1"]},
                                   {"role": "user", "content": "QA2"},
                                   {"role": "assistant", "content": GOLD["A::2"]}])
    b1 = _mk_turn("B", "B::1", 1, [])
    b2 = _mk_turn("B", "B::2", 2, [{"role": "user", "content": "QB1"},
                                   {"role": "assistant", "content": GOLD["B::1"]}])
    return [a1, a2, a3, b1, b2]


class _StubSystem:
    name = "stub"
    k = 10
    def retrieve(self, query):
        return [{"rank": 1, "doc_id": "d1", "score": 1.0, "text": "passage"}]


def _run(tmpdir, monkeypatch=None):
    """Drive the REAL run_domain with all IO monkeypatched to synthetic data."""
    turns = _synthetic_turns()
    calls = []

    def fake_generate(client, history, question, passages):
        calls.append((question, [dict(m) for m in history]))
        tid = next(t.task_id for t in turns if t.question == question)
        return STUB[tid]

    # patch module-level dependencies used inside run_domain
    orig = {}
    for name, val in {
        "load_turns": lambda domains=None: turns,
        "load_qrels": lambda domain: {},
        "load_corpus": lambda domain: ([], []),
        "load_lastturn_queries": lambda domain: {},
        "generate": fake_generate,
    }.items():
        orig[name] = getattr(runner, name)
        setattr(runner, name, val)

    out_log = os.path.join(tmpdir, "log.jsonl")
    out_eval = os.path.join(tmpdir, "eval.jsonl")
    try:
        runner.run_domain(_StubSystem(), "testdom", client=None,
                          out_log=out_log, out_eval=out_eval, k_retrieve=10, k_gen=5,
                          verbose=False)
    finally:
        for name, val in orig.items():
            setattr(runner, name, val)

    recs = {json.loads(l)["task_id"]: json.loads(l)
            for l in open(out_log, encoding="utf-8") if l.strip()}
    return recs, calls


def test_selfthreaded_history_and_no_gold_substitution():
    with tempfile.TemporaryDirectory() as td:
        recs, calls = _run(td)

    # gold really differs from stub answers (so the test can prove substitution absence)
    for tid in GOLD:
        assert STUB[tid] != GOLD[tid]

    # --- history invariant (from the logged rag_history_before_turn) ---
    assert recs["A::1"]["rag_history_before_turn"] == []
    assert recs["A::2"]["rag_history_before_turn"] == [
        {"role": "user", "content": "QA1"},
        {"role": "assistant", "content": "STUB_RAG_ANSWER_1"}]
    assert recs["A::3"]["rag_history_before_turn"] == [
        {"role": "user", "content": "QA1"},
        {"role": "assistant", "content": "STUB_RAG_ANSWER_1"},
        {"role": "user", "content": "QA2"},
        {"role": "assistant", "content": "STUB_RAG_ANSWER_2"}]

    # the answer actually stored is the stub RAG's answer (not gold)
    assert recs["A::2"]["rag_generated_answer"] == "STUB_RAG_ANSWER_2"

    # gold answers must NEVER appear in any turn's self-threaded history
    for rec in recs.values():
        blob = json.dumps(rec["rag_history_before_turn"])
        for g in GOLD.values():
            assert g not in blob, f"gold answer {g} leaked into rag history"

    # --- the model actually RECEIVED the self-threaded history (not gold) ---
    hist_by_q = {q: h for q, h in calls}
    assert hist_by_q["QA3"] == [
        {"role": "user", "content": "QA1"},
        {"role": "assistant", "content": "STUB_RAG_ANSWER_1"},
        {"role": "user", "content": "QA2"},
        {"role": "assistant", "content": "STUB_RAG_ANSWER_2"}]

    # --- conversation boundary: B never receives A's history ---
    assert recs["B::1"]["rag_history_before_turn"] == []
    assert recs["B::2"]["rag_history_before_turn"] == [
        {"role": "user", "content": "QB1"},
        {"role": "assistant", "content": "STUB_RAG_ANSWER_B1"}]
    assert hist_by_q["QB1"] == []
    for q, h in calls:
        if q in ("QB1", "QB2"):
            assert all("QA" not in m["content"] and "ANSWER_1" not in m["content"]
                       and "ANSWER_2" not in m["content"] and "ANSWER_3" not in m["content"]
                       for m in h), "conversation A history leaked into B"
    print("[ok] selfthreaded_history_and_no_gold_substitution")


def test_wrong_earlier_answer_persists_in_history():
    """A genuine earlier RAG error must remain visible to later turns (not corrected)."""
    with tempfile.TemporaryDirectory() as td:
        recs, _ = _run(td)
    # STUB_RAG_ANSWER_1 is the (possibly wrong) actual answer; it must be what turn 3 saw,
    # and gold must NOT have been substituted in.
    hist3 = recs["A::3"]["rag_history_before_turn"]
    assert {"role": "assistant", "content": "STUB_RAG_ANSWER_1"} in hist3
    assert {"role": "assistant", "content": "GOLD_ANSWER_1"} not in hist3
    print("[ok] wrong_earlier_answer_persists_in_history")


def test_selfrag_extraction():
    # leading role tag
    assert extract_selfrag_answer("Assistant: Paris is the capital.") == "Paris is the capital."
    # hallucinated dialogue continuation is cut
    assert extract_selfrag_answer("Paris.\nUser: And Germany?\nAssistant: Berlin.") == "Paris."
    # reflection control tokens stripped, content kept
    assert extract_selfrag_answer("[Retrieval]Paris[Fully supported][Utility:5]") == "Paris"
    # echoed (folded) question at the start is dropped
    q = "Conversation so far:\nUser: hi\n\nCurrent question: capital of France?"
    assert extract_selfrag_answer(q + "\nParis.", question=q) == "Paris."
    # CONSERVATIVE: a genuinely wrong answer is preserved verbatim (no fact-correction)
    assert extract_selfrag_answer("Assistant: London is the capital of France.") == \
        "London is the capital of France."
    # dialogue-framed: answer is content after the Assistant: marker
    assert extract_selfrag_answer("User: What is FICA?\nAssistant: A payroll tax law.") == \
        "A payroll tax law."
    # degenerate real artifact: echoed question + empty Assistant: -> no answer (kept empty)
    assert extract_selfrag_answer("User: What are tax deductions?\nAssistant:") == ""
    # empty / None safe
    assert extract_selfrag_answer("") == ""
    assert extract_selfrag_answer(None) == ""
    print("[ok] selfrag_extraction")


TESTS = [test_selfthreaded_history_and_no_gold_substitution,
         test_wrong_earlier_answer_persists_in_history,
         test_selfrag_extraction]

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

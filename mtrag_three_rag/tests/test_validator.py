#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for validate_pilot on REAL generated pilot output (built here with the stub
Self-RAG driver, whose answers differ from gold) plus a negative gold-threaded case.

Run:  python mtrag_three_rag/tests/test_validator.py   (or pytest)
"""
from __future__ import annotations

import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.abspath(os.path.join(HERE, ".."))
if WORK not in sys.path:
    sys.path.insert(0, WORK)

from common.mtrag_io import Turn                             # noqa: E402
from selfrag_sys import run_selfrag_sequential as seq        # noqa: E402
import validate_pilot                                        # noqa: E402

GOLD = {"S::1": "GOLD ONE two three four five", "S::2": "GOLD TWO two three four five",
        "S::3": "GOLD THREE two three four five"}
STUB = {"S::1": "STUB ANSWER one two three four", "S::2": "STUB ANSWER two two three four",
        "S::3": "STUB ANSWER three two three four"}
Q = {"S::1": "QS1", "S::2": "QS2", "S::3": "QS3"}


def _make_pilot():
    turns = [Turn("S", tid, i, "d", "c", Q[tid], [], GOLD[tid])
             for i, tid in enumerate(["S::1", "S::2", "S::3"], start=1)]
    ctxs = {tid: [{"id": f"doc_{tid}", "text": "p", "score": 1.0}] for tid in Q}

    def stub_gen(folded, ctxs_):
        tid = next(t for t in Q if folded.rstrip().endswith(Q[t]))
        return f"Assistant: {STUB[tid]}"

    orig = {}
    for name, val in {"load_turns": lambda domains=None: turns,
                      "load_qrels": lambda domain: {},
                      "load_corpus": lambda domain: ([], [])}.items():
        orig[name] = getattr(seq, name); setattr(seq, name, val)
    try:
        convs = seq.run_domain("d", ctxs, stub_gen, k_gen=5)
    finally:
        for name, val in orig.items():
            setattr(seq, name, val)
    return convs


def test_validator_passes_on_corrected_and_reports_divergence():
    convs = _make_pilot()
    res = validate_pilot.validate_paths([])   # smoke: empty
    r = validate_pilot.validate_conversation(convs[0])
    assert r["errors"] == [], r["errors"]
    # stub answers differ from gold, so later turns must be flagged as divergences
    assert r["divergences"], "expected divergence turns (rag_answer != reference)"
    # documents_used alias holds
    for t in convs[0]["turns"]:
        assert t["documents_used"] == t["retrieval"]["final_context_documents"]
    print("[ok] validator_passes_on_corrected_and_reports_divergence")


def test_validator_catches_gold_threaded_history():
    """A conversation whose history was gold-threaded must FAIL the validator."""
    convs = _make_pilot()
    bad = copy.deepcopy(convs[0])
    # corrupt turn 2's history to carry the GOLD answer instead of the actual one
    bad["turns"][1]["rag_history_before_turn"] = [
        {"role": "user", "content": "QS1"},
        {"role": "assistant", "content": GOLD["S::1"]}]
    r = validate_pilot.validate_conversation(bad)
    assert any("rag_history_before_turn" in e for e in r["errors"]), r["errors"]
    print("[ok] validator_catches_gold_threaded_history")


def test_validator_catches_gold_in_generation_input():
    convs = _make_pilot()
    bad = copy.deepcopy(convs[0])
    # inject a prior gold answer into turn 3's generation messages (history-pipeline leak)
    bad["turns"][2]["generation"]["messages"] = [
        {"role": "user", "content": "Conversation so far:\nAssistant: " + GOLD["S::1"]
         + "\n\nCurrent question: QS3"}]
    r = validate_pilot.validate_conversation(bad)
    assert any("leaked into generation input" in e for e in r["errors"]), r["errors"]
    print("[ok] validator_catches_gold_in_generation_input")


TESTS = [test_validator_passes_on_corrected_and_reports_divergence,
         test_validator_catches_gold_threaded_history,
         test_validator_catches_gold_in_generation_input]

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

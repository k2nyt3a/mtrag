#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lightweight invariant tests for the MTRAG RAG collection pipeline.

Runs against whatever pilot outputs exist under RAG/conversation/. Pure asserts —
run either with pytest or directly:  ``python RAG/tests/test_pipeline.py``

Invariants (per spec): dataset->corpus separation, conversation integrity (ids /
turn order / original human questions), own-answer history, no gold leakage, output
routing, and cross-RAG question consistency.
"""
from __future__ import annotations

import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from RAG import mtrag_data          # noqa: E402
from RAG.base import _norm          # noqa: E402

CONV_ROOT = os.path.join(REPO, "RAG", "conversation")


def _load_outputs():
    """{(dataset, rag): conversations_list} for every conversations.json present."""
    out = {}
    for p in glob.glob(os.path.join(CONV_ROOT, "*", "*", "conversations.json")):
        rag = os.path.basename(os.path.dirname(p))
        dataset = os.path.basename(os.path.dirname(os.path.dirname(p)))
        out[(dataset, rag)] = (p, json.load(open(p, encoding="utf-8")))
    return out


def _source_questions(dataset):
    """{conversation_id: {turn_id: exact human question}} from MTRAG-Human."""
    src = {}
    for conv in mtrag_data.load_conversations(dataset):
        src[conv.conversation_id] = {t.turn: t.question for t in conv.turns}
    return src


def test_output_routing():
    """Each file's records carry the dataset/rag of its folder path."""
    outs = _load_outputs()
    assert outs, "no pilot outputs found under RAG/conversation/"
    for (dataset, rag), (path, convs) in outs.items():
        for c in convs:
            assert c["dataset"] == dataset, f"{path}: dataset {c['dataset']} != {dataset}"
            assert c["rag"] == rag, f"{path}: rag {c['rag']} != {rag}"
    print(f"[ok] output_routing ({len(outs)} files)")


def test_conversation_integrity():
    """conversation_ids preserved, turns strictly increasing, questions byte-identical."""
    for (dataset, rag), (path, convs) in _load_outputs().items():
        src = _source_questions(dataset)
        for c in convs:
            assert c["conversation_id"] in src, f"unknown conv {c['conversation_id']}"
            turn_ids = [t["turn_id"] for t in c["turns"]]
            assert turn_ids == sorted(turn_ids), f"{path}: turn order not preserved"
            for t in c["turns"]:
                exp = src[c["conversation_id"]][t["turn_id"]]
                assert t["question"] == exp, (
                    f"{path}: question altered at {c['conversation_id']} turn {t['turn_id']}")
    print("[ok] conversation_integrity (ids/order/exact questions)")


def test_history_uses_own_answers():
    """Turn N's history = prior (human Q, THIS RAG's A). Never the reference answer."""
    for (dataset, rag), (path, convs) in _load_outputs().items():
        for c in convs:
            expected = []
            for t in c["turns"]:
                assert t["rag_history_before_turn"] == expected, (
                    f"{path}: history mismatch at {c['conversation_id']} turn {t['turn_id']}")
                ref = t["reference"]["answer"]
                for m in t["rag_history_before_turn"]:
                    assert m["content"] != ref or ref == "", (
                        f"{path}: reference answer leaked into history")
                expected = expected + [
                    {"role": "user", "content": t["question"]},
                    {"role": "assistant", "content": t["rag_answer"]},
                ]
    print("[ok] history_uses_own_answers")


def test_no_gold_leak():
    """Reference answer never verbatim in RAG input; gold evidence only if retrieved."""
    for (dataset, rag), (path, convs) in _load_outputs().items():
        for c in convs:
            for t in c["turns"]:
                inputs = ([m["content"] for m in t["rag_history_before_turn"]]
                          + [t["question"]]
                          + [d.get("text", "") for d in t["retrieval"]["retrieved_documents"]]
                          + [d.get("text", "") for d in t["retrieval"]["final_context_documents"]])
                blob = "\n".join(_norm(x) for x in inputs)
                retrieved = "\n".join(_norm(d.get("text", ""))
                                      for d in t["retrieval"]["retrieved_documents"])
                ref = _norm(t["reference"]["answer"])
                # A reference answer that also appears in an independently-retrieved
                # passage is not a leak (some MTRAG reference answers are verbatim
                # corpus passages the RAG legitimately retrieves). Mirrors the guard
                # in RAG.base.assert_no_gold_leak.
                if ref and len(ref.split()) >= 6 and ref not in retrieved:
                    assert ref not in blob, f"{path}: reference answer leaked ({c['conversation_id']})"
                for ev in t["reference"].get("evidence", []):
                    evn = _norm(ev.get("text", ""))
                    if evn and len(evn.split()) >= 12 and evn in blob:
                        assert evn in retrieved, (
                            f"{path}: gold evidence in input without being retrieved")
    print("[ok] no_gold_leak")


def test_dataset_corpus_separation():
    """Every retrieved doc id and gold evidence id belongs to that dataset's corpus."""
    for (dataset, rag), (path, convs) in _load_outputs().items():
        try:
            _, ids, _ = mtrag_data.load_corpus(dataset)
        except FileNotFoundError:
            print(f"[skip] dataset_corpus_separation ({dataset}): corpus not present")
            continue
        corpus = set(ids)
        for c in convs:
            for t in c["turns"]:
                for d in t["retrieval"]["retrieved_documents"]:
                    did = d["doc_id"]
                    # RAPTOR summary nodes are synthetic ids; their source ids must be real
                    if isinstance(did, str) and did.startswith("raptor_summary_"):
                        for s in d.get("source_document_ids", []):
                            assert s in corpus, f"{path}: raptor source {s} not in {dataset}"
                        continue
                    assert did in corpus, f"{path}: retrieved {did} not in {dataset} corpus"
                for eid in t["reference"]["evidence_ids"]:
                    assert eid in corpus, f"{path}: evidence {eid} not in {dataset} corpus"
    print("[ok] dataset_corpus_separation")


def test_cross_rag_consistency():
    """For a dataset, the same conversation has identical human questions across RAGs."""
    outs = _load_outputs()
    by_dataset = {}
    for (dataset, rag), (_, convs) in outs.items():
        by_dataset.setdefault(dataset, {})[rag] = {
            c["conversation_id"]: [t["question"] for t in c["turns"]] for c in convs}
    checked = 0
    for dataset, rags in by_dataset.items():
        if len(rags) < 2:
            continue
        names = list(rags)
        base = rags[names[0]]
        for other in names[1:]:
            for cid, qs in rags[other].items():
                if cid in base:
                    assert qs == base[cid], (
                        f"{dataset}: questions differ between {names[0]} and {other} for {cid}")
                    checked += 1
    print(f"[ok] cross_rag_consistency ({checked} shared conversations compared)"
          if checked else "[skip] cross_rag_consistency (need >=2 RAGs on a dataset)")


TESTS = [test_output_routing, test_conversation_integrity, test_history_uses_own_answers,
         test_no_gold_leak, test_dataset_corpus_separation, test_cross_rag_consistency]


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

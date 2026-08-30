#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validator for generated pilot conversations (the corrected self-threaded schema).

Proves, for a produced conversation file/dir, that:
  (1) for every turn N>1:  rag_history_before_turn == previous (user question,
      previous ACTUAL rag_answer) pairs  -- NOT gold;
  (2) reports every turn where a previous rag_answer != previous reference.answer,
      showing the next turn used the RAG answer and not gold (the whole point);
  (3) documents_used == retrieval.final_context_documents (per turn);
  (4) no reference/gold answer leaks into the generation input (e.g. Self-RAG's
      folded history) -- a prior reference answer must not appear in a later turn's
      generation messages unless the RAG itself actually produced that text.

Usage:
  python mtrag_three_rag/validate_pilot.py <conversations.json | conversation_dir> [...]
Exit code 0 = all checks pass; 1 = at least one violation.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, List


def _norm(s: str) -> str:
    return " ".join((s or "").split()).lower()


def _load(path: str) -> List[Dict]:
    """Load conversations from a conversations.json, a single conversation_*.json,
    or a directory containing them."""
    if os.path.isdir(path):
        arr = os.path.join(path, "conversations.json")
        if os.path.isfile(arr):
            return json.load(open(arr, encoding="utf-8"))
        convs = []
        for f in sorted(glob.glob(os.path.join(path, "conversation_*.json"))):
            convs.append(json.load(open(f, encoding="utf-8")))
        return convs
    obj = json.load(open(path, encoding="utf-8"))
    return obj if isinstance(obj, list) else [obj]


def _gen_input_text(turn: Dict) -> str:
    """All text the model saw as generation input this turn (messages contents)."""
    msgs = (turn.get("generation") or {}).get("messages") or []
    parts = []
    for m in msgs:
        if isinstance(m, dict):
            parts.append(m.get("content", ""))
    return "\n".join(parts)


def validate_conversation(conv: Dict) -> Dict:
    """Return {'errors': [...], 'divergences': [...]} for one conversation."""
    errors: List[str] = []
    divergences: List[str] = []
    cid = conv.get("conversation_id")
    turns = conv.get("turns", [])
    expected: List[Dict] = []          # self-threaded history built from ACTUAL answers
    prior = []                          # list of (question, rag_answer, reference_answer)
    for i, t in enumerate(turns):
        q = t.get("question", "")
        rag_ans = t.get("rag_answer", "")
        ref_ans = (t.get("reference") or {}).get("answer", "")

        # (1) history invariant
        if t.get("rag_history_before_turn") != expected:
            errors.append(f"[{cid} turn {t.get('turn_id')}] rag_history_before_turn "
                          f"!= previous (question, actual rag_answer) pairs")

        # (3) documents_used == retrieval.final_context_documents
        fcd = (t.get("retrieval") or {}).get("final_context_documents")
        if t.get("documents_used") != fcd:
            errors.append(f"[{cid} turn {t.get('turn_id')}] documents_used != "
                          f"retrieval.final_context_documents")

        # (4) no PRIOR gold answer leaked into this turn's generation input
        gen_blob = _norm(_gen_input_text(t))
        if gen_blob:
            for (pq, pra, pref) in prior:
                pref_n = _norm(pref)
                if pref_n and len(pref_n.split()) >= 6 and pref_n in gen_blob \
                        and pref_n != _norm(pra):
                    errors.append(f"[{cid} turn {t.get('turn_id')}] prior GOLD answer "
                                  f"leaked into generation input (history pipeline)")
                    break

        # (2) divergence report: prior rag_answer != prior reference.answer, and the
        #     history carried the RAG answer (checked in (1)).
        for j, (pq, pra, pref) in enumerate(prior, start=1):
            if _norm(pra) != _norm(pref):
                divergences.append(
                    f"[{cid} turn {t.get('turn_id')}] uses turn {j}'s ACTUAL rag_answer "
                    f"(differs from its reference) in history")
                break

        expected = expected + [{"role": "user", "content": q},
                               {"role": "assistant", "content": rag_ans}]
        prior.append((q, rag_ans, ref_ans))
    return {"conversation_id": cid, "errors": errors, "divergences": divergences}


def validate_paths(paths: List[str]) -> Dict:
    all_errors, all_div, n_conv, n_turns = [], [], 0, 0
    for p in paths:
        for conv in _load(p):
            n_conv += 1
            n_turns += len(conv.get("turns", []))
            r = validate_conversation(conv)
            all_errors += r["errors"]
            all_div += r["divergences"]
    return {"num_conversations": n_conv, "num_turns": n_turns,
            "errors": all_errors, "divergences": all_div}


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    res = validate_paths(sys.argv[1:])
    print(f"validated {res['num_conversations']} conversations, {res['num_turns']} turns")
    if res["divergences"]:
        print(f"\n[divergence] {len(res['divergences'])} turn(s) carried an actual "
              f"rag_answer that differs from its reference (RAG answer used, not gold):")
        for d in res["divergences"][:50]:
            print("  " + d)
    if res["errors"]:
        print(f"\n[FAIL] {len(res['errors'])} violation(s):")
        for e in res["errors"][:50]:
            print("  " + e)
        sys.exit(1)
    print("\n[ok] all invariants hold (history self-threaded; documents_used alias; "
          "no gold leak into generation input)")


if __name__ == "__main__":
    main()

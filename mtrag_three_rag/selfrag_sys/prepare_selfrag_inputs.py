#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STAGE A (local, no GPU): stage the MTRAG-side inputs for official Self-RAG.

Produces, per domain, one jsonl row per turn with everything the remote GPU box
needs to run official Self-RAG faithfully — WITHOUT leaking any gold:

  {
    task_id, conversation_id, turn, domain,
    retrieval_query,   # current user turn only -> Contriever query (Last-Turn setting)
    question,          # history-folded question -> the Self-RAG generation input
    history,           # [{role, content}] prior turns (for reference/formatting)
    reference_answer   # MTRAG gold answer -> kept for EVAL ONLY, never fed to the model
  }

The remote pipeline (see REMOTE_README.md):
  1. Contriever retrieval over the MTRAG domain corpus using `retrieval_query`
     -> attaches `ctxs` (top-k passages) to each row.
  2. run_short_form.py (selfrag_llama2_7b, vLLM) consumes `question` + `ctxs`.
  3. postprocess -> same per-turn log + MTRAG eval jsonl as RAPTOR / HippoRAG.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common.mtrag_io import load_turns, list_conversation_ids   # noqa: E402


def fold_history(history, question):
    """Fold conversation history into a single Self-RAG generation input string.
    Self-RAG short-form is single-turn; MTRAG is conversational, so prior turns are
    provided as context (same 'generation sees history' rule as the other systems)."""
    if not history:
        return question
    lines = []
    for m in history:
        who = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content']}")
    return ("Conversation so far:\n" + "\n".join(lines) +
            f"\n\nCurrent question: {question}")


def main():
    domains = sys.argv[1:] or ["clapnq", "govt", "fiqa", "cloud"]
    out_dir = os.path.join(HERE, "inputs")
    os.makedirs(out_dir, exist_ok=True)
    for dom in domains:
        turns = load_turns(domains=[dom])
        path = os.path.join(out_dir, f"{dom}_turns.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for t in turns:
                f.write(json.dumps({
                    "task_id": t.task_id,
                    "conversation_id": t.conversation_id,
                    "turn": t.turn,
                    "domain": t.domain,
                    "retrieval_query": t.question,
                    "question": fold_history(t.history, t.question),
                    "history": t.history,
                    "reference_answer": t.reference_answer,
                    "answerability": t.answerability,
                    "question_type": t.question_type,
                }, ensure_ascii=False) + "\n")
        print(f"{dom}: wrote {len(turns)} turns -> {path}  "
              f"({len(list_conversation_ids(dom))} conversations)")


if __name__ == "__main__":
    main()

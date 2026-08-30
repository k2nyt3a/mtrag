#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single source of truth for the corrected ``conversation_NNN.json`` schema + writer,
shared by every engine path (HippoRAG/RAPTOR runner and the Self-RAG sequential
driver) so no backend invents its own converter.

* ``build_turn_record`` produces one turn dict with the canonical key set/order
  (identical to scripts/run_mtrag_rag.py and the legacy files) and sets
  ``documents_used`` == ``retrieval["final_context_documents"]``.
* ``assemble_conversation`` wraps a conversation's turn records.
* ``write_conversations`` writes ``conversations.json`` + ``conversation_NNN.json``
  by REUSING scripts/run_mtrag_rag.py's ``_write_json`` / ``_write_split`` emitter
  (not a reimplementation).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List


def build_turn_record(turn_id, task_id: str, question: str,
                      history_before: List[Dict], retrieval: Dict, generation: Dict,
                      rag_answer: str, reference: Dict, mtrag_metadata: Dict) -> Dict:
    """Canonical turn record. ``retrieval`` must carry ``final_context_documents``;
    ``documents_used`` is aliased to it (never a separate object)."""
    return {
        "turn_id": turn_id,
        "task_id": task_id,
        "question": question,
        "rag_history_before_turn": history_before,
        "retrieval": retrieval,
        "generation": generation,
        "rag_answer": rag_answer,
        "documents_used": retrieval.get("final_context_documents", []),
        "reference": reference,
        "mtrag_metadata": mtrag_metadata,
    }


def assemble_conversation(conversation_id: str, dataset: str, collection: str,
                          rag: str, turn_records: List[Dict]) -> Dict:
    return {
        "conversation_id": conversation_id,
        "dataset": dataset,
        "collection": collection,
        "rag": rag,
        "num_turns": len(turn_records),
        "turns": turn_records,
    }


def _emitter():
    """Import the existing emitter functions from scripts/run_mtrag_rag.py."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    scripts = os.path.join(repo, "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from run_mtrag_rag import _write_json, _write_split       # noqa: E402
    return _write_json, _write_split


def write_conversations(out_dir: str, conversations: List[Dict]) -> None:
    """Write conversations.json + conversation_NNN.json, reusing the shared emitter."""
    _write_json, _write_split = _emitter()
    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, "conversations.json"), conversations)
    _write_split(out_dir, conversations)

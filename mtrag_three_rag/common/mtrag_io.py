#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared MTRAG-Human I/O + protocol for the three-system comparison
(Self-RAG vs HippoRAG vs RAPTOR).

This module is the single source of truth for how the benchmark inputs are read
so that ALL THREE systems see byte-identical questions, conversation history,
corpus, and gold data. Nothing here is system-specific.

Definitions (kept rigorously separate — see the user's spec):

  * MTRAG human question      = task["input"][-1]["text"]   (benchmark INPUT)
  * conversation history      = task["input"][:-1]          (all prior turns)
  * MTRAG reference answer     = task["targets"][0]["text"]  (GOLD answer; mixtral)
  * MTRAG gold supporting docs = official qrels (retrieval_tasks/<domain>/qrels)
                                 -> corpus passage ids judged relevant (GOLD only)
  * retrieved passages/answer  = produced by each tested RAG system (NOT gold)

UNIFORM PROTOCOL (identical across all three systems, documented so no system is
advantaged):

  RETRIEVAL QUERY  = the current user turn text only  (MTRAG "Last Turn" setting).
                     The same query string is handed to RAPTOR, HippoRAG, and
                     Self-RAG's retriever. This is the cleanest apples-to-apples
                     retrieval comparison and matches MTRAG's own lastturn task.
  GENERATION INPUT = full conversation history (mapped to chat roles) + the
                     current question + the passages THAT system retrieved.
                     Each system uses its own native reader; where the reader's
                     model is configurable (RAPTOR, HippoRAG) it is pinned to the
                     same gpt-4o-mini so only the RETRIEVAL method differs. Self-RAG
                     uses its trained 7B generator (inseparable from the method).

The MTRAG `reference answer` and `gold passages` are NEVER given to a tested RAG
system as input — only used afterwards for evaluation.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---- repo-relative paths (resolved from this file's location) ---------------
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))            # D:\1 DynaicQA\jaist
WORK = os.path.join(REPO, "mtrag_three_rag")

# Data locations are env-overridable so the workspace is portable to another box
# (e.g. a GPU server). Defaults match this repo's layout.
TASKS_REFERENCE = os.environ.get("MTRAG_TASKS") or os.path.join(
    REPO, "conv-annotate", "rag", "mtrag_data", "reference.jsonl")
CORPORA_DIR = os.environ.get("MTRAG_CORPORA_DIR") or os.path.join(
    REPO, "mtrag_validation", "corpora", "passage_level")
RETRIEVAL_TASKS_DIR = os.environ.get("MTRAG_RETRIEVAL_TASKS") or os.path.join(
    WORK, "data", "retrieval_tasks")   # official qrels/query files staged here

# MTRAG conversation.Collection  ->  corpus file stem (domain).
COLLECTION_TO_DOMAIN = {
    "mt-rag-clapnq-elser-512-100-20240503": "clapnq",
    "mt-rag-govt-elser-512-100-20240611": "govt",
    "mt-rag-fiqa-beir-elser-512-100-20240501": "fiqa",
    "mt-rag-ibmcloud-elser-512-100-20240502": "cloud",
}


def load_env_openai_key() -> str:
    """Read OPENAI_API_KEY from the repo .env (never printed)."""
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    envp = os.path.join(REPO, ".env")
    if os.path.exists(envp):
        for line in open(envp, encoding="utf-8"):
            line = line.strip()
            if line.startswith("OPENAI_API_KEY="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                os.environ["OPENAI_API_KEY"] = val
                return val
    raise RuntimeError("OPENAI_API_KEY not found in environment or repo .env")


# ---- MTRAG generation tasks -------------------------------------------------
@dataclass
class Turn:
    conversation_id: str
    task_id: str
    turn: str
    domain: str                       # clapnq | govt | fiqa | cloud
    collection: str
    question: str                     # current user turn (the retrieval query)
    history: List[Dict]               # [{role: user|assistant, content: str}, ...] prior turns
    reference_answer: str             # MTRAG gold answer (targets[0].text)
    answerability: Optional[str] = None
    question_type: Optional[List[str]] = None
    multi_turn: Optional[List[str]] = None
    raw: Dict = field(default_factory=dict)


def _speaker_to_role(sp: str) -> str:
    return "user" if sp == "user" else "assistant"


def load_turns(conv_ids: Optional[List[str]] = None,
               domains: Optional[List[str]] = None,
               tasks_path: str = TASKS_REFERENCE) -> List[Turn]:
    """Load MTRAG generation tasks as Turn objects.

    `contexts` in reference.jsonl (empty) and RAG.jsonl (ELSER top-5) are IGNORED:
    every tested system retrieves its own evidence. Only input/targets/metadata used.
    """
    conv_ids = set(conv_ids) if conv_ids else None
    domains = set(domains) if domains else None
    out: List[Turn] = []
    with open(tasks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            dom = COLLECTION_TO_DOMAIN.get(t.get("Collection", ""), "?")
            if conv_ids and t["conversation_id"] not in conv_ids:
                continue
            if domains and dom not in domains:
                continue
            inp = t["input"]
            history = [{"role": _speaker_to_role(m["speaker"]), "content": m["text"]}
                       for m in inp[:-1]]
            out.append(Turn(
                conversation_id=t["conversation_id"],
                task_id=t["task_id"],
                turn=str(t["turn"]),
                domain=dom,
                collection=t.get("Collection", ""),
                question=inp[-1]["text"],
                history=history,
                reference_answer=(t.get("targets") or [{}])[0].get("text", ""),
                answerability=(t.get("Answerability") or [None])[0] if t.get("Answerability") else None,
                question_type=t.get("Question Type"),
                multi_turn=t.get("Multi-Turn"),
                raw=t,
            ))
    out.sort(key=lambda x: (x.conversation_id, int(x.turn)))
    return out


def list_conversation_ids(domain: str, tasks_path: str = TASKS_REFERENCE) -> List[str]:
    """Ordered unique conversation_ids for a domain (order of first appearance)."""
    seen: List[str] = []
    s = set()
    for line in open(tasks_path, encoding="utf-8"):
        t = json.loads(line)
        if COLLECTION_TO_DOMAIN.get(t.get("Collection", "")) == domain:
            cid = t["conversation_id"]
            if cid not in s:
                s.add(cid); seen.append(cid)
    return seen


# ---- corpus -----------------------------------------------------------------
def load_corpus(domain: str, limit: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """Return (texts, ids) row-aligned for a domain's official passage corpus.

    ids[i] is the corpus passage _id (traceable to MTRAG gold + qrels).
    `limit` (plumbing only) truncates to the first N passages — never for real runs.
    """
    path = os.path.join(CORPORA_DIR, f"{domain}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"corpus not found: {path}")
    texts, ids = [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = str(r.get("_id") or r.get("id") or "")
            title = r.get("title") or ""
            text = r.get("text") or ""
            # MTRAG passages: keep title as a prefix line when present (matches ELSER text form)
            full = (title + "\n" + text).strip() if title else text
            texts.append(full)
            ids.append(pid)
    return texts, ids


# ---- gold relevance (official qrels) ---------------------------------------
def load_qrels(domain: str, variant: str = "lastturn") -> Dict[str, Dict[str, int]]:
    """Load official BEIR qrels: {query_id: {corpus_id: rel}}.

    Files staged under data/retrieval_tasks/<domain>/qrels/*.tsv (downloaded from
    the official repo). Returns {} if not present yet."""
    qdir = os.path.join(RETRIEVAL_TASKS_DIR, domain, "qrels")
    qrels: Dict[str, Dict[str, int]] = {}
    if not os.path.isdir(qdir):
        return qrels
    import csv
    for fn in os.listdir(qdir):
        if not fn.endswith(".tsv"):
            continue
        with open(os.path.join(qdir, fn), encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if len(row) < 3:
                    continue
                qid, cid, rel = row[0], row[1], int(row[2])
                qrels.setdefault(qid, {})[cid] = rel
    return qrels


# ---- unified per-turn log record (the logging spec) ------------------------
def make_log_record(turn: Turn, rag_system: str, retrieved: List[Dict],
                    generated_answer: str, gold_doc_ids: List[str],
                    gold_passages: List[str], latency_s: Optional[float] = None,
                    extra: Optional[Dict] = None) -> Dict:
    """Assemble the full per-turn log record required by the spec.

    `retrieved` is a list of {rank, doc_id (corpus _id or node id), score,
    score_type, text, ...system-specific provenance...}.
    """
    rec = {
        "conversation_id": turn.conversation_id,
        "turn_id": turn.turn,
        "task_id": turn.task_id,
        "dataset": turn.domain,
        "collection": turn.collection,
        "rag_system": rag_system,
        "question": turn.question,
        "conversation_history": turn.history,
        "answerability": turn.answerability,
        "question_type": turn.question_type,
        "multi_turn": turn.multi_turn,
        "mtrag_reference_answer": turn.reference_answer,
        "mtrag_gold_supporting_document_ids": gold_doc_ids,
        "mtrag_gold_supporting_passages": gold_passages,
        "retrieved_document_ids": [d.get("doc_id") for d in retrieved],
        "retrieved_passages": [d.get("text") for d in retrieved],
        "retrieval_scores": [d.get("score") for d in retrieved],
        "retrieved_docs": retrieved,          # full records incl. provenance
        "rag_generated_answer": generated_answer,
        "latency_s": latency_s,
    }
    if extra:
        rec.update(extra)
    return rec


def to_mtrag_eval_record(turn: Turn, retrieved: List[Dict], generated_answer: str) -> Dict:
    """Emit a record in the OFFICIAL MTRAG evaluation JSONL shape so the unmodified
    eval scripts (run_algorithmic.py / run_generation_eval.py) can score it.

    predictions = tested-system answer; targets = MTRAG reference; contexts =
    tested-system retrieved passages (NOT gold)."""
    return {
        "conversation_id": turn.conversation_id,
        "task_id": turn.task_id,
        "task_type": "rag",
        "turn": turn.turn,
        "Collection": turn.collection,
        "dataset": turn.raw.get("dataset", "MT-RAG Authors (Internal)"),
        "input": turn.raw["input"],
        "contexts": [{"document_id": d.get("doc_id"), "text": d.get("text", ""),
                       "score": d.get("score"), "title": d.get("title", "")}
                      for d in retrieved],
        "targets": turn.raw["targets"],
        "predictions": [{"text": generated_answer}],
        **{k: turn.raw[k] for k in ("Answerability", "Question Type", "Multi-Turn",
                                     "No. References") if k in turn.raw},
    }

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MTRAG-Human data access for the RAG conversation-collection pipeline.

Single source of truth for reading the benchmark inputs so every tested RAG sees
byte-identical questions, history, corpus, and gold data. Reads THIS repository's
``mtrag-human/`` files directly (schema verified against the actual files, not
assumed). Corpora + precomputed embeddings are read from the sibling research repo
(``jaist/mtrag_validation``) where the uncompressed corpora live; the path is
env-overridable.

Rigorous separation (per spec):
  * current human question      = input[-1]["text"]           (sent to the RAG)
  * conversation history (human)= input[:-1]                   (prior turns)
  * MTRAG reference answer       = targets[0]["text"]           (GOLD; eval only)
  * MTRAG reference evidence ids = official qrels (dev.tsv)      (GOLD; eval only)

The reference answer and gold evidence are NEVER handed to a tested RAG as input.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---- repo-relative paths ----------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
MTRAG_REPO = os.path.abspath(os.path.join(_HERE, ".."))          # mt-rag-benchmark
MTRAG_HUMAN = os.path.join(MTRAG_REPO, "mtrag-human")

TASKS_PATH = os.environ.get("MTRAG_TASKS") or os.path.join(
    MTRAG_HUMAN, "generation_tasks", "reference.jsonl")
RETRIEVAL_TASKS_DIR = os.environ.get("MTRAG_RETRIEVAL_TASKS") or os.path.join(
    MTRAG_HUMAN, "retrieval_tasks")

# Uncompressed corpora + precomputed <domain>.embindex.{npy,json} live in the
# sibling research repo by default. Override with MTRAG_CORPORA_DIR to relocate.
_DEFAULT_CORPORA = os.path.abspath(
    os.path.join(MTRAG_REPO, "..", "jaist", "mtrag_validation", "corpora", "passage_level"))
CORPORA_DIR = os.environ.get("MTRAG_CORPORA_DIR") or _DEFAULT_CORPORA

DATASETS = ("clapnq", "cloud", "fiqa", "govt")

# MTRAG conversation.Collection -> domain (verified from generation_tasks).
COLLECTION_TO_DOMAIN = {
    "mt-rag-clapnq-elser-512-100-20240503": "clapnq",
    "mt-rag-govt-elser-512-100-20240611": "govt",
    "mt-rag-fiqa-beir-elser-512-100-20240501": "fiqa",
    "mt-rag-ibmcloud-elser-512-100-20240502": "cloud",
}


def load_openai_key() -> str:
    """Read OPENAI_API_KEY from the environment or the sibling repo .env (never printed)."""
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    for envp in (os.path.join(MTRAG_REPO, ".env"),
                 os.path.abspath(os.path.join(MTRAG_REPO, "..", "jaist", ".env"))):
        if os.path.exists(envp):
            for line in open(envp, encoding="utf-8"):
                line = line.strip()
                if line.startswith("OPENAI_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        os.environ["OPENAI_API_KEY"] = val
                        return val
    raise RuntimeError("OPENAI_API_KEY not found in environment or repo .env files")


# ---- per-turn record --------------------------------------------------------
@dataclass
class Turn:
    conversation_id: str
    task_id: str                      # "<conv_id><::><turn>" — preserved MTRAG id
    turn: int
    domain: str                       # clapnq | cloud | fiqa | govt
    collection: str
    question: str                     # current human question (input[-1].text) — EXACT
    human_history: List[Dict]         # [{role, content}] prior HUMAN turns (context only)
    reference_answer: str             # MTRAG gold answer (eval only)
    reference_evidence_ids: List[str] # official qrels corpus ids (eval only)
    answerability: Optional[str] = None
    question_type: Optional[List[str]] = None
    multi_turn: Optional[List[str]] = None
    lastturn_query: Optional[str] = None   # MTRAG official last-turn retrieval text
    raw: Dict = field(default_factory=dict)


@dataclass
class Conversation:
    conversation_id: str
    domain: str
    collection: str
    turns: List[Turn]


def _speaker_to_role(sp: str) -> str:
    return "user" if sp == "user" else "assistant"


def _load_lastturn(domain: str) -> Dict[str, str]:
    """task_id -> official last-turn retrieval query text (strips the '|user|:' prefix)."""
    path = os.path.join(RETRIEVAL_TASKS_DIR, domain, f"{domain}_lastturn.jsonl")
    out: Dict[str, str] = {}
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        txt = r.get("text", "")
        if txt.startswith("|user|:"):
            txt = txt[len("|user|:"):].strip()
        out[str(r["_id"])] = txt
    return out


def load_qrels(domain: str) -> Dict[str, List[str]]:
    """task_id -> list of gold corpus passage ids (official qrels dev.tsv)."""
    qdir = os.path.join(RETRIEVAL_TASKS_DIR, domain, "qrels")
    out: Dict[str, List[str]] = {}
    if not os.path.isdir(qdir):
        return out
    for fn in os.listdir(qdir):
        if not fn.endswith(".tsv"):
            continue
        with open(os.path.join(qdir, fn), encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)  # header: query-id, corpus-id, score
            for row in reader:
                if len(row) < 3:
                    continue
                qid, cid, rel = row[0], row[1], int(row[2])
                if rel > 0:
                    out.setdefault(qid, []).append(cid)
    return out


def load_conversations(domain: str) -> List[Conversation]:
    """Load all MTRAG-human conversations for one domain, turn-ordered.

    Conversation boundaries and turn ordering are preserved exactly. Gold answer +
    gold evidence ids are attached for STORAGE ONLY (never given to a RAG).
    """
    if domain not in DATASETS:
        raise ValueError(f"unknown dataset {domain!r}; expected one of {DATASETS}")
    lastturn = _load_lastturn(domain)
    qrels = load_qrels(domain)

    by_conv: Dict[str, List[Turn]] = {}
    conv_meta: Dict[str, Tuple[str, str]] = {}
    with open(TASKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if COLLECTION_TO_DOMAIN.get(t.get("Collection", "")) != domain:
                continue
            inp = t["input"]
            hist = [{"role": _speaker_to_role(m["speaker"]), "content": m["text"]}
                    for m in inp[:-1]]
            tid = t["task_id"]
            turn = Turn(
                conversation_id=t["conversation_id"],
                task_id=tid,
                turn=int(t["turn"]),
                domain=domain,
                collection=t.get("Collection", ""),
                question=inp[-1]["text"],
                human_history=hist,
                reference_answer=(t.get("targets") or [{}])[0].get("text", ""),
                reference_evidence_ids=qrels.get(tid, []),
                answerability=(t.get("Answerability") or [None])[0] if t.get("Answerability") else None,
                question_type=t.get("Question Type"),
                multi_turn=t.get("Multi-Turn"),
                lastturn_query=lastturn.get(tid),
                raw=t,
            )
            by_conv.setdefault(t["conversation_id"], []).append(turn)
            conv_meta[t["conversation_id"]] = (domain, t.get("Collection", ""))

    convs: List[Conversation] = []
    for cid, turns in by_conv.items():
        turns.sort(key=lambda x: x.turn)
        dom, col = conv_meta[cid]
        convs.append(Conversation(conversation_id=cid, domain=dom, collection=col, turns=turns))
    # deterministic order = order of first appearance in the tasks file
    order = {}
    for i, t in enumerate(open(TASKS_PATH, encoding="utf-8")):
        try:
            cid = json.loads(t)["conversation_id"]
        except Exception:
            continue
        order.setdefault(cid, i)
    convs.sort(key=lambda c: order.get(c.conversation_id, 1 << 30))
    return convs


# ---- corpus -----------------------------------------------------------------
def corpus_path(domain: str) -> str:
    return os.path.join(CORPORA_DIR, f"{domain}.jsonl")


def load_corpus(domain: str) -> Tuple[List[str], List[str], List[str]]:
    """Return (texts, ids, titles) row-aligned for a domain's official passage corpus.

    Text formatting matches the precomputed embedding index exactly: ``text.strip()``
    only, skipping empty-text rows — so the returned rows line up 1:1 with the
    ``<domain>.embindex.npy`` matrix (fingerprint-verified) and no re-embedding occurs.
    """
    path = corpus_path(domain)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"corpus not found: {path}\nSet MTRAG_CORPORA_DIR to the folder holding "
            f"the uncompressed <domain>.jsonl corpora.")
    texts, ids, titles = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            text = (r.get("text") or "").strip()
            if not text:
                continue
            texts.append(text)
            ids.append(str(r.get("_id") or r.get("id") or ""))
            titles.append(r.get("title") or "")
    return texts, ids, titles


def has_embindex(domain: str) -> bool:
    base = os.path.join(CORPORA_DIR, f"{domain}.embindex")
    return os.path.exists(base + ".npy") and os.path.exists(base + ".json")


def embindex_path(domain: str) -> str:
    return os.path.join(CORPORA_DIR, f"{domain}.embindex")

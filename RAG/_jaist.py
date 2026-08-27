#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Locate ``mtrag_three_rag`` — the engine holding the Self-RAG / HippoRAG / RAPTOR
implementations that the connectors under ``RAG/<system>/rag.py`` import at call time.

Resolution order (first that exists wins):
  1. ``$JAIST_ROOT/mtrag_three_rag``            — explicit override
  2. ``<repo>/mtrag_three_rag``                 — bundled inside THIS repo (default)
  3. ``<repo>/../jaist/mtrag_three_rag``        — sibling research repo (legacy layout)

The engine CODE is bundled in this repo; its heavy prebuilt indices are NOT (they
exceed GitHub's file-size limits). Build them on the box with
``mtrag_three_rag/run_full.py <system> <domain>`` or fetch a prebuilt bundle.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))

_CANDIDATES = []
if os.environ.get("JAIST_ROOT"):
    _CANDIDATES.append(os.path.join(os.environ["JAIST_ROOT"], "mtrag_three_rag"))
_CANDIDATES.append(os.path.join(_REPO, "mtrag_three_rag"))                 # bundled
_CANDIDATES.append(os.path.abspath(
    os.path.join(_REPO, "..", "jaist", "mtrag_three_rag")))                # sibling

THREE_RAG = next((p for p in _CANDIDATES if os.path.isdir(p)), _CANDIDATES[-2])


def ensure_on_path() -> bool:
    """Put mtrag_three_rag on sys.path so its adapter packages import. Returns
    False if the engine directory is not present."""
    if not os.path.isdir(THREE_RAG):
        return False
    if THREE_RAG not in sys.path:
        sys.path.insert(0, THREE_RAG)
    return True

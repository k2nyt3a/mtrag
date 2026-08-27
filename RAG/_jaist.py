#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Locate the sibling research repo that holds the existing RAG implementations.

The Self-RAG / HippoRAG / RAPTOR implementations are NOT copied into this repo.
The connectors under ``RAG/<system>/rag.py`` import the upstream-backed adapters
from ``jaist/mtrag_three_rag`` at call time. Override the location with the env var
``JAIST_ROOT`` if the research repo lives elsewhere.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
JAIST_ROOT = os.environ.get("JAIST_ROOT") or os.path.abspath(
    os.path.join(_HERE, "..", "..", "jaist"))
THREE_RAG = os.path.join(JAIST_ROOT, "mtrag_three_rag")


def ensure_on_path() -> bool:
    """Put mtrag_three_rag on sys.path so its adapter packages import. Returns
    False if the research repo is not present."""
    if not os.path.isdir(THREE_RAG):
        return False
    if THREE_RAG not in sys.path:
        sys.path.insert(0, THREE_RAG)
    return True

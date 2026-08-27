#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG registry: name -> builder. The runner asks the registry for a system by name;
the builder returns an adapter instance whose ``.available`` / ``.unavailable_reason``
say whether it can actually run in this environment. The runner NEVER substitutes a
different system for an unavailable one.
"""
from __future__ import annotations

from typing import Callable, Dict, List

RAG_NAMES: List[str] = ["selfrag", "hipporag", "raptor", "vector"]


def _vector_build(domain: str, k: int = 5, **kw):
    from .vector.rag import build
    return build(domain, k=k, **kw)


def _hipporag_build(domain: str, k: int = 5, **kw):
    from .hipporag.rag import build
    return build(domain, k=k, **kw)


def _raptor_build(domain: str, k: int = 5, **kw):
    from .raptor.rag import build
    return build(domain, k=k, **kw)


def _selfrag_build(domain: str, k: int = 5, **kw):
    from .selfrag.rag import build
    return build(domain, k=k, **kw)


_BUILDERS: Dict[str, Callable] = {
    "vector": _vector_build,
    "hipporag": _hipporag_build,
    "raptor": _raptor_build,
    "selfrag": _selfrag_build,
}


def get_rag(name: str, domain: str, k: int = 5, **kw):
    if name not in _BUILDERS:
        raise ValueError(f"unknown rag {name!r}; expected one of {RAG_NAMES}")
    return _BUILDERS[name](domain, k=k, **kw)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Version-controlled wrapper around the OFFICIAL Self-RAG short-form generation path.

``run_selfrag_sequential.py`` needs a model callable

    generate_fn(folded_question: str, ctxs: list[dict]) -> raw_answer: str

so it can drive Self-RAG turn-by-turn with self-threaded history. This module builds
that callable by REUSING the official ``run_short_form`` adaptive-retrieval decode
(the trained ``selfrag/selfrag_llama2_7b`` via vLLM + the paper's reflection-token
reranking). Nothing about Self-RAG's method or decoding parameters is reimplemented
or changed here — we import and call the upstream functions.

Reproducibility: this wrapper is in the repo (not a box-only script). The heavy
pieces (vLLM, the trained model, the upstream ``run_short_form``/``utils`` modules)
are imported LAZILY inside ``load_generate_fn`` so this file imports fine on a
CPU/laptop for unit tests. Point ``--selfrag-repo`` / ``$SELFRAG_REPO`` at a clone of
``AkariAsai/self-rag/retrieval_lm`` on the GPU box.

The decode params are exactly the paper's short-form adaptive-retrieval config,
sourced from ``run_selfrag_sequential.DEFAULT_PARAMS`` — do not change them here.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

GenerateFn = Callable[[str, List[Dict]], str]


def _resolve_selfrag_repo(selfrag_repo: Optional[str]) -> str:
    repo = selfrag_repo or os.environ.get("SELFRAG_REPO")
    if not repo:
        raise RuntimeError(
            "Self-RAG upstream repo not found. Clone AkariAsai/self-rag and set "
            "--selfrag-repo /path/to/self-rag/retrieval_lm (or $SELFRAG_REPO). This "
            "wrapper reuses that repo's run_short_form decode; it is not vendored.")
    if not os.path.isdir(repo):
        raise RuntimeError(f"SELFRAG_REPO does not exist: {repo}")
    return repo


def load_generate_fn(model_name: str, params: Dict,
                     selfrag_repo: Optional[str] = None,
                     download_dir: Optional[str] = None,
                     dtype: Optional[str] = None) -> GenerateFn:
    """Load the trained Self-RAG model ONCE and return a per-turn ``generate_fn``.

    Heavy imports (vllm + upstream run_short_form/utils) happen here, lazily, so the
    module stays importable without a GPU. ``params`` is the paper's short-form
    adaptive-retrieval config (passed through unchanged).
    """
    repo = _resolve_selfrag_repo(selfrag_repo)
    import sys
    if repo not in sys.path:
        sys.path.insert(0, repo)

    # --- lazy heavy imports (GPU box only) ---
    from vllm import LLM, SamplingParams                       # noqa: E402
    from transformers import AutoTokenizer                     # noqa: E402
    # upstream Self-RAG helpers (the SAME ones run_short_form.py uses)
    from run_short_form import call_model_rerank_w_scores_batch  # noqa: E402
    from utils import load_special_tokens, PROMPT_DICT           # noqa: E402

    _dtype = dtype or params.get("dtype", "half")
    model = LLM(model=model_name, dtype=_dtype, download_dir=download_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sampling = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=params.get("max_new_tokens", 300),
        logprobs=32000, skip_special_tokens=False)
    # reflection-token id maps, gated by the same flags as the paper config
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=params.get("use_groundness", True),
        use_utility=params.get("use_utility", True))

    def _format_prompt(instruction: str) -> str:
        # upstream no-input instruction template
        return PROMPT_DICT["prompt_no_input"].format_map({"instruction": instruction})

    def generate_fn(folded_question: str, ctxs: List[Dict]) -> str:
        evidences = [{"title": c.get("title", ""), "text": c.get("text", "")}
                     for c in ctxs]
        prompt = _format_prompt(folded_question)
        # Official adaptive-retrieval short-form decode + reflection reranking.
        pred, _results, _do_retrieve = call_model_rerank_w_scores_batch(
            prompt, evidences=evidences, model=model, max_new_tokens=params.get("max_new_tokens", 300),
            ret_tokens=ret_tokens, rel_tokens=rel_tokens, grd_tokens=grd_tokens,
            ut_tokens=ut_tokens, sampling_params=sampling,
            threshold=params.get("threshold", 0.2), use_seqscore=params.get("use_seqscore", True),
            w_rel=params.get("w_rel", 1.0), w_sup=params.get("w_sup", 1.0),
            w_use=params.get("w_use", 0.5), mode=params.get("mode", "adaptive_retrieval"))
        return pred if isinstance(pred, str) else str(pred)

    return generate_fn

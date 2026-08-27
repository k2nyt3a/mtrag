#!/usr/bin/env bash
# Shared environment for the MTRAG three-system comparison.
# Envs live in WSL (~/mtrag3/envs); ALL large caches/downloads/outputs go to D:.
# Source this before installing or running: `source /mnt/d/1\ DynaicQA/jaist/mtrag_three_rag/env.sh`
# OPENAI_API_KEY is forwarded from Windows via WSLENV=OPENAI_API_KEY (never written to disk).

export MTRAG3_ROOT="/mnt/d/1 DynaicQA/jaist/mtrag_three_rag"
export MTRAG3_DATA_D="$MTRAG3_ROOT/_caches"      # D:-resident caches (keep off C:)
mkdir -p "$MTRAG3_DATA_D"

# keep every package/model/tmp cache OFF the (full) C: drive
export UV_CACHE_DIR="$MTRAG3_DATA_D/uv"
export PIP_CACHE_DIR="$MTRAG3_DATA_D/pip"
export HF_HOME="$MTRAG3_DATA_D/hf"
export HUGGINGFACE_HUB_CACHE="$MTRAG3_DATA_D/hf/hub"
export TRANSFORMERS_CACHE="$MTRAG3_DATA_D/hf/transformers"
export TORCH_HOME="$MTRAG3_DATA_D/torch"
export XDG_CACHE_HOME="$MTRAG3_DATA_D/xdg"
export TMPDIR="$MTRAG3_DATA_D/tmp"
export NUMBA_CACHE_DIR="$MTRAG3_DATA_D/numba"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$NUMBA_CACHE_DIR"

export RAPTOR_ENV="$HOME/mtrag3/envs/raptor_env/bin/python"
export HIPPORAG_ENV="$HOME/mtrag3/envs/hipporag_env/bin/python"
export PATH="$HOME/.local/bin:$PATH"

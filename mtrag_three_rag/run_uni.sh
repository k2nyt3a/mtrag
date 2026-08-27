#!/usr/bin/env bash
# Run a full-corpus build+eval on the uni box.  Usage:
#   export OPENAI_API_KEY=sk-...
#   bash run_uni.sh            # -> hipporag clapnq   (default)
#   bash run_uni.sh raptor     # -> raptor   clapnq
#   SYS=hipporag DOMAIN=govt bash run_uni.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"

SYS="${1:-${SYS:-hipporag}}"
DOMAIN="${DOMAIN:-clapnq}"

# self-contained data paths (bundled)
export MTRAG_CORPORA_DIR="$ROOT/_data/corpora/passage_level"
export MTRAG_TASKS="$ROOT/_data/reference.jsonl"
export MTRAG_RETRIEVAL_TASKS="$ROOT/data/retrieval_tasks"

# keep caches on local disk
export HF_HOME="$ROOT/_caches/hf"; export UV_CACHE_DIR="$ROOT/_caches/uv"
export TMPDIR="$ROOT/_caches/tmp"; export NUMBA_CACHE_DIR="$ROOT/_caches/numba"
mkdir -p "$HF_HOME" "$TMPDIR" "$NUMBA_CACHE_DIR"

: "${OPENAI_API_KEY:?Set it first:  export OPENAI_API_KEY=sk-...}"

# tuning for a big-RAM box (the batch size is the real speedup, not the GPU):
export HIPPO_EMB_BATCH="${HIPPO_EMB_BATCH:-256}"   # vs default 16 -> ~15x fewer embedding calls
WORKERS="${WORKERS:-24}"                            # OpenIE concurrency (24 avoids OpenAI 429s)

if [ "$SYS" = "hipporag" ]; then
  PY="envs/hipporag_env/bin/python"
else
  PY="envs/raptor_env/bin/python"
fi
echo ">>> $SYS / $DOMAIN | workers=$WORKERS emb_batch=$HIPPO_EMB_BATCH"
echo ">>> NER+triple LLM calls are served from the bundled cache (no re-billing);"
echo ">>> only entity/fact embeddings (~\$0.5) + generation are newly billed."
exec "$PY" run_full.py "$SYS" "$DOMAIN" "$WORKERS"

#!/usr/bin/env bash
# One-time setup on a fresh Linux box (GPU box works; GPU only needed for Self-RAG).
# Run from INSIDE the extracted mtrag_three_rag/ folder:  bash setup_uni.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "[1/4] installing uv + Python 3.11"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11

echo "[2/4] cloning official repos (HippoRAG + RAPTOR)"
mkdir -p repos
[ -d repos/HippoRAG ] || git clone --depth 1 https://github.com/OSU-NLP-Group/HippoRAG.git repos/HippoRAG
[ -d repos/raptor ]   || git clone --depth 1 https://github.com/parthsarthi03/raptor.git repos/raptor

echo "[3/4] hipporag_env  (official HippoRAG from source; vllm is an extra, so it is skipped)"
uv venv --python 3.11 envs/hipporag_env
uv pip install --python envs/hipporag_env/bin/python "./repos/HippoRAG"

echo "[4/4] raptor_env  (modern openai + RAPTOR deps; RAPTOR pkg via .pth)"
uv venv --python 3.11 envs/raptor_env
uv pip install --python envs/raptor_env/bin/python \
  openai umap-learn scikit-learn sentence-transformers faiss-cpu tiktoken tenacity numpy scipy torch
SP="$(envs/raptor_env/bin/python -c 'import site;print(site.getsitepackages()[0])')"
echo "$ROOT/repos/raptor" > "$SP/raptor_src.pth"

echo
echo "Setup done. Next:"
echo "  export OPENAI_API_KEY=sk-...        # your key (never commit it)"
echo "  bash run_uni.sh                     # HippoRAG full clapnq (resumes from bundled cache)"
echo "  bash run_uni.sh raptor              # RAPTOR full clapnq"

# Clone-and-run: HippoRAG / RAPTOR / Self-RAG

The engine **code** for the three upstream systems is bundled in this repo (this
folder). A single `git clone` of `k2nyt3a/mtrag` gives you everything you need to
**run**, with two unavoidable exceptions noted below.

```
git clone https://github.com/k2nyt3a/mtrag.git
cd mtrag
export TR="$PWD/mtrag_three_rag"
```

## 1. Corpora (one-time) — already in the clone, just unzip

The passage-level corpora ship in this repo as zips (`corpora/passage_level/*.jsonl.zip`).
Extract them where the runner expects them:

```bash
mkdir -p "$TR/_data/corpora/passage_level"
for z in corpora/passage_level/*.jsonl.zip; do
  unzip -o "$z" -d "$TR/_data/corpora/passage_level/"
done
```

## 2. HippoRAG & RAPTOR — CPU, no GPU needed

```bash
cd "$TR"
bash setup_uni.sh                 # one-time: builds envs/hipporag_env + envs/raptor_env (uv)
export OPENAI_API_KEY=sk-...
for D in clapnq fiqa cloud govt; do SYS=raptor   DOMAIN=$D bash run_uni.sh; done
for D in clapnq fiqa cloud govt; do SYS=hipporag DOMAIN=$D bash run_uni.sh; done
```

Outputs: `$TR/outputs/<system>/<domain>.log.jsonl` (+ `.mtrag_eval.jsonl`).

## 3. Self-RAG — needs a 24 GB GPU (run in the GPU container)

Its MTRAG inputs are bundled (`selfrag_sys/inputs/*_turns.jsonl`). Follow
[`selfrag_sys/REMOTE_README.md`](selfrag_sys/REMOTE_README.md) end-to-end (Contriever
index build → retrieval → vLLM reflection generation → postprocess).

## The two things a clone canNOT give you

1. **HippoRAG's prebuilt OpenIE index (2.1 GB, incl. a 1.2 GB file).** Too large for
   GitHub (100 MB/file cap). On first run `run_uni.sh hipporag <domain>` **rebuilds**
   it (LLM OpenIE over the full corpus — hours + API cost). To skip the rebuild, fetch
   a prebuilt index bundle (e.g. a GitHub **Release** asset) and drop it under
   `$TR/indices/hipporag/<domain>_full/` before running.
2. **The Self-RAG 7B model + GPU.** Downloaded/served on the GPU box; never in the repo.

RAPTOR's tree and Self-RAG's Contriever index are (re)built on the box on first run, so
those two are fully clone-and-run.

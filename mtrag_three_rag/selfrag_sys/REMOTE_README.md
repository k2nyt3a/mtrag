# Official Self-RAG on MTRAG — remote 24 GB GPU runbook

Runs the **official** `selfrag/selfrag_llama2_7b` (trained reflection model, via vLLM)
with the **official Contriever** retriever over each MTRAG domain corpus. No
behavioral reimplementation, no GPT-4o-mini substitute. The generation/reflection is
the actual Self-RAG method.

> **Why remote:** the trained 7B model needs a GPU. Faithful config = **fp16 on a
> single 24 GB GPU** (RTX 3090/4090/A10G/L4). Job is short (~<1 h on a 3090). See the
> VRAM table in the main chat. Avoid 4-bit if you want the reflection-token thresholds
> exact.

The three systems share the SAME MTRAG questions, conversation-history protocol,
corpus, evaluation turns, and metrics. Self-RAG differs only where the method is
inseparable (its trained generator + Contriever retriever).

## 0. What you carry to the box

From this repo:
- `mtrag_three_rag/selfrag_sys/inputs/<domain>_turns.jsonl`  — 842 MTRAG turns
  (retrieval_query = last user turn; question = history-folded; reference_answer kept
  for EVAL ONLY). **Already generated locally — no gold leaks to the model.**
- `mtrag_three_rag/selfrag_sys/remote/*.py`                   — the helper scripts below.
- The 4 official corpora `mtrag_validation/corpora/passage_level/<domain>.jsonl`
  (or re-download from IBM/mt-rag-benchmark — they are byte-identical).

## 1. Box setup

```bash
git clone https://github.com/AkariAsai/self-rag.git && cd self-rag/retrieval_lm
conda create -n selfrag python=3.10 -y && conda activate selfrag
pip install -r requirements.txt          # includes vllm; torch matched to the GPU CUDA
pip install vllm                          # if not pulled by requirements
# Contriever retriever weights are downloaded automatically (facebook/contriever-msmarco)
```

## 2. Build the Contriever index over the MTRAG corpus (ONE per domain)

Convert the corpus to Self-RAG's DPR TSV, then encode + it stays on the box:

```bash
D=clapnq   # repeat for govt fiqa cloud
python /path/to/mtrag_three_rag/selfrag_sys/remote/corpus_to_dpr_tsv.py \
    --corpus /path/to/mtrag_validation/corpora/passage_level/$D.jsonl \
    --out corpora/$D.tsv

python generate_passage_embeddings.py \
    --model_name_or_path facebook/contriever-msmarco \
    --passages corpora/$D.tsv \
    --output_dir embeddings/$D --shard_id 0 --num_shards 1 \
    --per_gpu_batch_size 512
```

> Index is built ONLY from the corpus — no reference answers, qrels, or future
> questions. clapnq is the largest (183 408 passages); encoding on a 24 GB GPU is minutes.

## 3. Contriever retrieval → ctxs (Last-Turn query)

```bash
python /path/to/mtrag_three_rag/selfrag_sys/remote/make_contriever_queries.py \
    --turns /path/to/mtrag_three_rag/selfrag_sys/inputs/$D_turns.jsonl \
    --out queries/$D.jsonl                 # {id, question=retrieval_query}

python passage_retrieval.py \
    --model_name_or_path facebook/contriever-msmarco \
    --passages corpora/$D.tsv \
    --passages_embeddings "embeddings/$D/*.pkl" \
    --data queries/$D.jsonl \
    --n_docs 5 \
    --output_dir retrieved/$D            # writes queries with "ctxs" added
```

## 4. Assemble the Self-RAG input (history-folded question + ctxs)

```bash
python /path/to/mtrag_three_rag/selfrag_sys/remote/merge_ctxs_to_selfrag.py \
    --turns  /path/to/.../inputs/$D_turns.jsonl \
    --retrieved retrieved/$D/queries.jsonl \
    --out selfrag_in/$D.jsonl            # each row: {id, question(folded), ctxs[5]}
```

## 5. Run official Self-RAG generation (vLLM, adaptive retrieval + reflection)

```bash
python run_short_form.py \
    --model_name selfrag/selfrag_llama2_7b \
    --input_file selfrag_in/$D.jsonl \
    --output_file selfrag_out/$D.jsonl \
    --max_new_tokens 300 --ndocs 5 --world_size 1 --dtype half \
    --mode adaptive_retrieval \
    --threshold 0.2 --use_groundness --use_utility --use_seqscore \
    --w_rel 1.0 --w_sup 1.0 --w_use 0.5
```

This is the paper's adaptive-retrieval short-form config (reflection tokens
`[Retrieval]/[Relevant]/[Fully supported]/[Utility]` drive the decoding). Record the
final answer AND, if you want, the per-passage reflection scores it logs.

## 6. Convert to the shared schema (do on the box or after pulling results back)

```bash
python /path/to/mtrag_three_rag/selfrag_sys/remote/postprocess_selfrag.py \
    --turns /path/to/.../inputs/$D_turns.jsonl \
    --selfrag_in selfrag_in/$D.jsonl \
    --selfrag_out selfrag_out/$D.jsonl \
    --domain $D \
    --out_log  /path/to/.../outputs/selfrag/$D.log.jsonl \
    --out_eval /path/to/.../outputs/selfrag/$D.mtrag_eval.jsonl
```

`out_eval` is in the **official MTRAG eval format** (predictions=Self-RAG answer,
targets=MTRAG reference, contexts=Contriever-retrieved passages) so it scores with the
same unmodified `conv-annotate/rag/eval/` scripts as RAPTOR and HippoRAG. `out_log` is
the same per-turn log record (question, history, gold ids/passages, retrieved
ids/passages/scores, answer, latency) as the other two systems.

## Notes / fidelity
- Retrieval query = last user turn (uniform across all three systems).
- Generation input folds the conversation history (Self-RAG is single-turn; MTRAG is
  conversational) — same "generation sees history" rule as RAPTOR/HippoRAG.
- Contriever ids map straight to MTRAG corpus ids (the TSV keeps `_id`), so retrieved
  vs. gold retrieval eval works identically to the other systems.
- **These remote scripts are prepared but not GPU-tested from this machine** (no local
  GPU) — sanity-check the first few rows on the box before the full 842-turn run.

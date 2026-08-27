# RAG conversation collection over MTRAG-Human

Raw RAG conversation data ONLY. No failure analysis, no question generation
(Xie / Proposed / static / dynamic), no method comparison. Each MTRAG-Human
conversation is replayed, turn by turn, through one tested RAG on that dataset's
own corpus.

```
MTRAG human conversation  (mtrag-human/generation_tasks/reference.jsonl)
        ↓  choose dataset: clapnq | cloud | fiqa | govt
        ↓  correct dataset corpus (never mixed)
        ↓  tested RAG: selfrag | hipporag | raptor | vector
        ↓  retrieval  → final context → answer
        ↓
RAG/conversation/<dataset>/<rag>/conversations.json  (+ run_metadata.json)
```

## Design guarantees

- **Original human questions** are sent verbatim (`input[-1].text`); never rewritten,
  paraphrased, decomposed, or generated. Retrieval query = the raw current question
  (which is exactly the MTRAG "last-turn" retrieval protocol — verified, not a rewrite).
- **Multi-turn** structure preserved: conversation boundaries + turn order intact.
- **Own-history**: each turn's history is the tested RAG's OWN prior (question, answer)
  pairs — not the MTRAG reference answers. Each RAG develops its own trajectory.
- **No gold leakage**: the MTRAG reference answer and gold evidence are stored for
  evaluation only and never enter RAG input (`RAG.base.assert_no_gold_leak` enforces
  this every turn).
- **Dataset separation**: clapnq→clapnq corpus, cloud→cloud, fiqa→fiqa, govt→govt.
- **MTRAG ids preserved**: `conversation_id`, `task_id`, `turn`, corpus `doc_id`s,
  gold evidence ids — so retrieved vs reference evidence can be compared later.

## Layout

```
RAG/
├── base.py         RAG contract + MTRAG-baseline generator + leakage safeguard
├── registry.py     name → adapter builder (+ availability)
├── mtrag_data.py   MTRAG-Human loaders (turns / corpus / lastturn / qrels)
├── _jaist.py       locates the sibling research repo for the delegating connectors
├── vector/rag.py   Vector RAG (self-contained; runs locally)
├── hipporag/rag.py connector → jaist/mtrag_three_rag/hipporag_sys  (upstream HippoRAG v2)
├── raptor/rag.py   connector → jaist/mtrag_three_rag/raptor_sys    (upstream RAPTOR)
├── selfrag/rag.py  connector → jaist/mtrag_three_rag/selfrag_sys   (remote GPU / vLLM)
├── tests/          invariant tests
└── conversation/   outputs, per <dataset>/<rag>
```

The Self-RAG / HippoRAG / RAPTOR implementations are **not copied** here; the
connectors delegate to the existing upstream-backed adapters in the sibling repo
`jaist/mtrag_three_rag`. Set `JAIST_ROOT` to relocate it. Set `MTRAG_CORPORA_DIR`
to point at the uncompressed `<domain>.jsonl` corpora (+ precomputed `.embindex`).

## Run

```bash
# see which systems can run in this environment (and why not)
python scripts/run_mtrag_rag.py --dataset fiqa --list-availability

# collect one dataset × RAG
python scripts/run_mtrag_rag.py --dataset fiqa --rag vector \
    --num-conversations 2 --seed 0 --retrieval-k 5

# resume after a crash (skips completed conversations, no duplicates)
python scripts/run_mtrag_rag.py --dataset fiqa --rag vector --num-conversations 2 --resume
```

Options: `--dataset {clapnq,cloud,fiqa,govt}` `--rag {selfrag,hipporag,raptor,vector}`
`--conversation-ids …` `--num-conversations N` `--max-turns N` `--retrieval-k K`
`--seed S` `--resume` `--output DIR`.

## Availability

`vector` runs locally (precomputed `text-embedding-3-small` corpus embeddings for all
four domains + `gpt-4o-mini` generation). `hipporag` / `raptor` need their upstream
packages (WSL `hipporag_env` / `raptor_env`) plus a prebuilt per-domain index/tree.
`selfrag` is a remote 24 GB GPU job (vLLM + trained 7B model). The runner reports the
exact reason when a system is unavailable and never substitutes another system.

## Tests

```bash
python RAG/tests/test_pipeline.py
```

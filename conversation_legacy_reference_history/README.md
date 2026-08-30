# Legacy affected conversations — reference/gold-history protocol

These are verbatim backups of the RAG conversations that were generated with the
**wrong** conversation-history protocol.

```
history_protocol = reference/gold-history
```

For the affected systems (`selfrag`, `hipporag`, `raptor`), each turn's
`rag_history_before_turn` was built from the MTRAG **gold/reference** answers of
prior turns instead of the RAG system's **actual** prior `rag_answer` values.
Verified: turn-2 `rag_history_before_turn[assistant]` == turn-1 `reference.answer`
(gold), not turn-1 `rag_answer` (actual).

Self-RAG answers here also contain raw `run_short_form` artifacts
(`Assistant:` / `User:` leakage, echoed questions, dialogue continuations).

**Do not modify, judge, or reuse these as results for the corrected runs.** They are
kept only as experimental provenance / backup. The corrected, self-threaded runs
(`history_protocol = self-threaded-rag-history`) will occupy the original
`RAG/conversation/<dataset>/<rag>/` paths after the pilot is approved.

`vector` is NOT here: it was already correctly self-threaded and is unchanged.

## Backed-up directories (source -> here), verified byte-identical

| dataset/rag      | files |
|------------------|-------|
| clapnq/selfrag   | 30    |
| cloud/hipporag   | 27    |
| cloud/selfrag    | 27    |
| fiqa/hipporag    | 28    |
| fiqa/raptor      | 28    |
| fiqa/selfrag     | 28    |
| govt/hipporag    | 29    |
| govt/selfrag     | 29    |

(counts include `conversations.json` + the `conversation_NNN.json` split files.)
Copied verbatim from `RAG/conversation/<dataset>/<rag>/`; verified with `cmp` (0
byte-mismatches).

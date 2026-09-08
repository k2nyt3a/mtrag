# Diagnostic follow-up question generation

This package generates **one diagnostic follow-up question per failed RAG turn**.
It implements *only* the follow-up generation stage of the RAG failure-attribution
study. It never classifies, predicts, or names the six RAG failure categories —
a separate downstream component consumes each generated follow-up (and the RAG
answer it elicits) to do attribution.

The proposed method is **RePAIR-inspired** (see §1a): it maps a flawed RAG
question/answer directly to a diagnostic *question strategy*, without first
predicting a failure category.

## 1. Purpose

Research question:

> Can **answer-conditioned dynamic** follow-up question generation provide better
> evidence for RAG failure attribution than **static** follow-up generation?

For every turn a RAG conversation got wrong, we generate a follow-up that can be
re-asked to the same RAG system. Two methods run over the *same* failed turns:

| Method | Label | Sees history | Sees failed answer | Sees gold |
|--------|-------|:------------:|:------------------:|:---------:|
| static (Xie-style baseline) | `static` | yes | **no** | no |
| proposed (RePAIR-inspired dynamic) | `proposed_dynamic_diagnostic` | yes | **yes** | no |

The proposed method's defining property is that its question *reacts to the actual
answer the RAG produced*, so if the RAG had produced a different wrong answer the
follow-up would probably change.

## 1a. RePAIR-inspired adaptation

RePAIR maps a flawed RAG state directly to a corrective *action* (and a revised
answer) without needing a detailed failure category first. We adapt **only that
response-to-action principle**:

```
RePAIR:        failed RAG state          → corrective action      → revised answer
This project:  failed question + answer  → question strategy      → follow-up question
```

RePAIR operation → question strategy mapping (`question_strategies.REPAIR_OPERATION_MAPPING`):

| RePAIR operation | Question-generation adaptation |
|------------------|--------------------------------|
| `REWRITE` | `contrastive_reformulation` |
| `DECOMPOSE` | `atomic_decomposition` |
| `RETRIEVAL` | `retrieval_oriented_reformulation` (re-ask, does **not** call the retriever) |
| `GENERATEANSWER` | `evidence_verification` / `completion_probe` |
| `REFINEDOC` | *not implemented* — the generator cannot inspect or modify documents |

**Why this is not a full RePAIR implementation.** We intentionally exclude
RePAIR's executor, its document refinement, its answer regeneration, its internal
query rewriting / retrieval, and all of its training (DPO, off-/on-policy,
token-level-F1 reward). No corrective *action* is executed and no revised answer
is produced — the only output is a follow-up **question**. The generator also
never predicts one of the six failure categories before choosing a strategy.

## 2. Information boundaries (static vs dynamic)

The dynamic generator may receive **only**: conversation id, failed turn id, the
`rag_history` up to and including the failed response, the failed question, and
the failed RAG answer.

It must **never** receive: the gold answer, `truth_history` / ground-truth
answers, judge reasoning, correctness explanations, retrieved documents,
final-context documents, or any human/predicted failure label. The *incorrect
label* is used only to select which turns get a follow-up.

How the boundary is enforced in code:

- `conversation_loader.AnnotatedConversation` copies **only** permitted fields out
  of `conversation.json`. It has no attribute for gold answers, judge reasoning,
  or documents, so nothing downstream can read what was never loaded.
- `schemas.DynamicFollowUpInput` is a frozen dataclass whose only fields are the
  permitted ones — there is literally nowhere to put forbidden data.
- `schemas.StaticFollowUpInput` has no `failed_rag_answer` field at all, and its
  history ends at the failed *question*.
- Tests in `tests/diagnostic_follow_ups/` assert that distinctive gold / reasoning
  / retrieved markers never appear in any prompt or output record.

## 3. Question strategies

Strategies describe *how to phrase the follow-up*; they are **not** failure
categories. The model selects one per turn (`question_strategies.QuestionStrategy`):

- `contrastive_reformulation` — the answer resolved an entity/pronoun to the
  wrong thing; restate the intended reference (from history) and re-ask.
- `evidence_verification` — the answer made a specific claim; ask the RAG to
  verify that exact claim against its sources.
- `retrieval_oriented_reformulation` — a short/elliptical/context-dependent
  question whose answer drifted; re-ask it as a self-contained question so the
  unchanged RAG can retrieve the intended topic. **Does not call the retriever**
  and adds no keywords from the gold answer or documents.
- `completion_probe` — the original question explicitly asked for more than one
  thing and the answer covered only part; ask for the omitted part.
- `answerability_probe` — the answer was empty/echoed/unusable; ask whether the
  question is answerable at all, allowing abstention.
- `atomic_decomposition` — an explicitly complex/multi-hop request; ask for a
  part-by-part answer preserving the original need.
- `general_verification` — safe fallback; ask the RAG to reconsider and verify
  its previous answer.

Each generated question also records a `target_source`
(`schemas.TargetSource`): `failed_answer_span` when it targets specific text in
the answer, or `response_behavior` when it targets how the answer behaved (empty,
echo, refusal, omission).

`question_strategies.suggest_strategy` is a deterministic *hint* (used as a soft
suggestion in the prompt and by the offline client); the LLM still chooses.

## 4. How answer conditioning works

The dynamic prompt (`prompt_templates.build_dynamic_question_prompt`) shows the
failed RAG answer explicitly and instructs the model to react to a specific span
or behavior of it. The validator's `is_answer_conditioned` check requires either:

- a *strong* signal: the follow-up echoes a salient token from the failed answer
  that was not already in the failed question, or
- a *weak* signal: an explicit "reacting to your previous answer" phrasing.

Because the generator never receives the gold answer, an answer-conditioned
question targets *what the RAG said*, never *what was correct*.

## 5. Validation and retry

Validation (`question_validator.QuestionValidator`) is deterministic and API-free.
A question is valid only if it is a single question, preserves the information
need, is answer-conditioned, stays on topic, introduces no new numbers/facts,
names no failure category, and is grammatical. (An abstention clause is recorded
but not required.) Each rule is reported individually in the output.

On failure, `build_retry_feedback` produces targeted, **gold-free** guidance about
the specific rules that failed, and the generator retries up to `max_attempts`
(default 3). Every attempt is recorded. If all attempts fail, the record is marked
`is_valid: false` — invalid questions are never silently accepted.

## 6. Input and output schemas

Input envelope (dynamic): `schemas.DynamicFollowUpInput`
(`conversation_id`, `failed_turn_id`, `rag_history`, `failed_question`,
`failed_rag_answer`).

Output record (one JSON object per failed turn, written as JSONL):

```json
{
  "conversation_id": "001",
  "failed_turn_id": "4",
  "method": "proposed_dynamic_diagnostic",
  "dataset": "clapnq",
  "rag": "selfrag",
  "original_question": "Do they have predators?",
  "original_rag_answer": "They are preyed upon by hawks, owls, and snakes.",
  "question_strategy": "evidence_verification",
  "target_source": "failed_answer_span",
  "target_from_failed_answer": "hawks, owls, and snakes",
  "generated_follow_up": "You answered \"...hawks, owls, and snakes\". Can you verify that claim ...?",
  "answer_conditioning_explanation": "The question verifies the exact predator list generated by the RAG.",
  "generation_input_provenance": {
    "used_rag_history": true,
    "used_failed_rag_answer": true,
    "used_gold_answer": false,
    "used_truth_history": false,
    "used_judge_reasoning": false,
    "used_failure_category": false,
    "used_retrieved_documents": false
  },
  "validation": { "is_answer_conditioned": true, "same_information_need": true, "is_valid": true, "reasons": [] },
  "is_valid": true,
  "generation_attempts": 1
}
```

No predicted failure category is ever included. The original `conversation.json`
files are never modified; output goes to a separate directory.

**Independent per-turn branching.** When a conversation has several failed turns,
each is processed as an independent branch whose `rag_history` runs only through
*its own* failed answer. A generated follow-up is never inserted back into the
conversation, so one branch's question cannot contaminate another's input.

## 7. Running a dry run (no API, no cost)

The pipeline is **offline by default** — it uses a deterministic stand-in client
(`OfflineHeuristicModelClient`) that makes no network calls:

```bash
python -m question_generation.diagnostic_follow_ups.generation_pipeline \
    --input . --method both --limit 20 --output-dir generated_follow_ups
```

- `--input` accepts files, directories (searched recursively for
  `conversation.json`), or globs.
- `--method` is `proposed` (default; alias `dynamic`), `static`, or `both`.
- `--limit` caps the number of failed turns (useful for pilots).

Outputs are written to `proposed_dynamic_follow_ups.jsonl` and/or
`static_follow_ups.jsonl` in the output directory.

## 8. Running live generation (requires an API key)

```bash
export ANTHROPIC_API_KEY=...        # never hard-code the key
python -m question_generation.diagnostic_follow_ups.generation_pipeline \
    --input . --method proposed --live --output-dir generated_follow_ups
```

The generation model (configurable via `DIAGNOSTIC_FOLLOWUP_MODEL`) is **separate
from the frozen RAG system** that produced `conversation.json`; running this
changes nothing about retrieval, the RAG's model, seeds, or top-k.

## 9. Running the tests

Stdlib `unittest`, no third-party dependencies, no paid calls:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## 10. Short example

Failed question: `"Do they have predators?"`
Failed RAG answer: `"They are preyed upon by hawks, owls, and snakes."`

Dynamic follow-up (`evidence_verification`):
`"You answered '...hawks, owls, and snakes'. Can you verify that claim about
predators using the available sources, or state that it is unavailable?"`

The follow-up targets the exact predator list the RAG produced and never reveals
what the correct predators are.

## Limitations

- The offline stand-in client is a deterministic placeholder for exercising the
  pipeline without paid calls; its contrastive-reformulation phrasing can be
  clumsy when a good intended entity cannot be extracted heuristically. Live runs
  with the LLM produce more fluent questions.
- Deterministic validation cannot detect a *plausibly worded* correct answer that
  happens to match an unseen gold value (it has no access to gold). The primary
  guarantee against gold leakage is the input boundary, not the validator; the
  validator adds structural safety (single question, on-topic, no new numbers, no
  failure-category mention).

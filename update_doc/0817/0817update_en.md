# 2026-08-17 Update: Reader Prompt Fix (Comparison-Question Answer Misalignment)

## 1. Finding the Problem: Reading Through Concrete Cases, Not Just Aggregate Metrics

After the gate three-signal ablation (0810 update) was completed, instead of continuing to
tune aggregate metrics, a case study was run on arm D (feature-level fusion, MuSiQue dev
full set). Wrong-answer cases were first split by whether retrieval was fully correct:

| | Count |
|---|---|
| Wrong answer (EM=0) | 1,376 |
| — of which: every hop retrieved the gold passage, but the final answer was still wrong | 485 |

These 485 cases, where retrieval and the gate both did their job correctly and the answer
was still wrong, have nothing to do with retrieval or the gate — the problem is in the last
step, where the reader synthesizes all intermediate results into a final answer. Comparing
each of these 485 final answers against the per-hop short answers (`prior`) yields three
categories:

| Type | Count | Share |
|---|---|---|
| Final answer equals the last hop's short answer, but still graded EM=0 (pure wording mismatch against gold, e.g. "McKinley" vs. "President McKinley") | 372 | 77% |
| Final answer equals an **earlier** hop's short answer (the reader genuinely picked the wrong hop's conclusion) | 39 | 8% |
| Matches neither (the reader synthesized a new, wrong answer on its own) | 74 | 15% |

The first category is an artifact of EM's strictness, not a real error. The second and third
are the real problems, together about 23%.

Repeating the same analysis on 2WikiMultihopQA and HotpotQA showed the "reader picked the
wrong hop" pattern is **not MuSiQue-specific — its share is even higher on the other two
datasets**:

| Dataset | Retrieval fully correct, answer wrong | — of which "reader picked wrong hop" | Share |
|---|---|---|---|
| MuSiQue | 485 | 39 | 8.0% |
| 2WikiMultihopQA | 4,894 | 639 | 13.1% |
| HotpotQA | 1,522 | 232 | 15.2% |

Digging further into the 2WikiMultihopQA and HotpotQA cases showed "picked the wrong hop"
concentrates heavily on **comparison-style questions** (2WikiMultihopQA has a dedicated
`comparison` question type) — and it isn't simply "picked the wrong hop," the reader isn't
performing the comparison at all, and is directly outputting a raw fact from some hop
(usually a date) as the answer:

```
Q: Which film was released more recently, Royal Treasure or When Love Begins?
gold: Royal Treasure   |   final: January 15, 2016
per-hop short answers (prior): ['January 15, 2016', '2008']

Q: Did the board game San Marco or About Time come out first?
gold: San Marco   |   final: 2001
per-hop short answers (prior): ['2001', '2007']
```

The two hops correctly looked up each film's/game's release date, but the answer that
should have been output is "which one" (the title/name) — the reader just echoed one of the
raw dates verbatim instead.

## 2. Root Cause and Fix: The Reader Prompt Was Missing an Explicit Comparison-Question Instruction

The final-answer generation prompt is built by
`retrieval/run_retrieval_exp_wavefront.py::build_final_reader_cot_prompt`. Its only sentence
constraining answer format is:

```
Give a short span or phrase only (entity, date, number, or yes/no).
```

This explicitly lists "date" and "number" as **valid answer types**, treated identically
across all question types. For most bridge questions, the assumption "the last hop's short
answer IS the final answer" holds; but for comparison questions, the dates/numbers looked up
at each hop are only **intermediate values used to make the comparison** — the actual
answer should be the name of whichever entity wins the comparison. The prompt never
distinguished these two cases, so the reader frequently just output a hop's raw date as the
final answer.

The fix adds a new function to `run_retrieval_exp_wavefront.py`,
`build_final_reader_cot_prompt_comparison_hint` (**the original function is left untouched**,
the new one sits alongside it), appending exactly one sentence to the base prompt:

```
If the original question asks you to COMPARE two or more entities (for example "which was
released more recently", "who was born first", "which is longer"), the per-step answers
above are the values used to make that comparison, not the final answer themselves -- you
must perform the comparison yourself and answer with the NAME of the entity that satisfies
it, not a date, number, or other raw value.
```

`retrieval/run_retrieval_exp_wavefront_gate_v3_rawprefix.py` gained a new
`--final-reader-prompt {default,comparison_hint}` flag to switch between the two versions,
with everything else (retrieval, gate scoring, per-hop candidate selection) left completely
unchanged — guaranteeing a clean single-variable comparison.

## 3. Results: 2WikiMultihopQA Full-Set Validation

MuSiQue dev, λ=0.50, rawprefix, with only `--final-reader-prompt` switched:

| | recall@1 | chain@1 | EM | F1 |
|---|---|---|---|---|
| default prompt | 0.9289 | 0.8613 | 0.4990 | 0.5759 |
| **comparison_hint** | 0.9289 | 0.8613 | **0.5982** | **0.6770** |
| Δ | flat | flat | **+9.92** | **+10.11** |

recall@1 and chain@1 are completely unchanged — retrieval and the gate weren't touched at
all, proving the gain comes cleanly from this one prompt change. EM/F1 rose by roughly 10
percentage points, the single largest gain from any change in this project so far, larger
than λ tuning, rawprefix text alignment, and the BGE retrieval prefix combined.

This also confirmed the root-cause diagnosis: the "reader picked wrong hop" bug's
reproduction rate dropped from 13.1% (639/4,894) to **2.6%** (98/3,833), a 5x reduction.

## 4. MuSiQue and HotpotQA Validation — Completed

All three datasets are now validated. recall@1/chain@1 stay exactly flat against the default
prompt on every dataset (retrieval and the gate are untouched, confirming the change is
clean again), and EM/F1 are positive across the board:

| Dataset | recall@1 | chain@1 | EM (default→comparison_hint) | F1 (default→comparison_hint) |
|---|---|---|---|---|
| MuSiQue | 0.7018 (flat) | 0.5143 (flat) | 0.4307→0.4427 (+1.20) | 0.5270→0.5386 (+1.16) |
| 2WikiMultihopQA | 0.9289 (flat) | 0.8613 (flat) | 0.4990→0.5982 (**+9.92**) | 0.5759→0.6770 (**+10.11**) |
| HotpotQA | 0.6596→0.6595 (flat) | 0.5568→0.5567 (flat) | 0.5217→0.5517 (+3.00) | 0.6517→0.6870 (+3.53) |

The size of the gain ranks 2wiki > hotpot > musique, which doesn't exactly match the ranking
of the "picked wrong hop" bug's **share within affected cases** (hotpot's 15.2% is higher
than 2wiki's 13.1%) — the reason is that hotpot's retrieval baseline is lower to begin with
(recall@1 only ~66%), so the "retrieval fully correct" precondition covers fewer examples;
what actually drives the overall EM gain is the number of affected cases **as a share of the
full dataset**, and there 2wiki (639/12,576 = 5.1%) is higher than hotpot (232/7,405 = 3.1%).

All three datasets moved in the same, positive direction with no regressions anywhere, so
`comparison_hint` was adopted as the new default prompt.

## 5. Further Exploration — Tried, Not Adopted; `comparison_hint` Is Final

Continuing the case study on top of `comparison_hint` (the residual "retrieval fully
correct, answer still wrong" cases on 2WikiMultihopQA) surfaced two categories of problems
`comparison_hint` still didn't cover, and a new prompt fix was tried for each. Both were
validated on all three datasets using `replay_final_reader.py` (reconstructs each hop's
input from the `prior`/`para_ids` already stored in an existing case file and only reruns
the final-answer step — retrieval, the gate, and the per-hop short answers are never rerun,
which is much cheaper than a full end-to-end run). Both were net-negative and were not
adopted:

- **`prompt_short_answer_with_context_type_match`** (per-hop short answer, constrains the
  answer's type to match the interrogative word): helped on MuSiQue (EM +1.94/F1 +2.32, on
  top of comparison_hint), but was roughly flat on 2wiki with a small drop on the retrieval
  side, and roughly flat on HotpotQA — inconsistent across the three datasets, not adopted.
- **`build_final_reader_cot_prompt_comparison_reasoning`** (require an explicit reasoning
  line before the final answer): helped on 2wiki (EM +2.65/F1 +2.31), but was net-negative
  on both MuSiQue (EM -1.32/F1 -1.29) and HotpotQA (EM -2.35/F1 -2.92). The case study found
  the root cause: the model was treating comparison_hint's "answer with the entity's name,
  not a date/number" instruction as a rule to apply unconditionally, invoking it whenever a
  name and a date both happened to appear in its own reasoning — even for questions that
  weren't comparison questions at all (a striking example: the model correctly derived "June
  1982" as the answer, then overrode it to the person's name "Diego Maradona" purely because
  of this rule).
- **`build_final_reader_cot_prompt_comparison_positive`** (rewrote the same rule in positive
  form and dropped the illustrative examples): net-negative on all three datasets, with the
  largest drop on 2wiki (EM -3.05/F1 -3.12). The case study found the opposite root cause
  this time: without the illustrative examples, the model failed to recognize many
  comparison questions as comparison questions at all, so the trigger rate collapsed and the
  reader reverted to the original comparison_hint-era problem (echoing a raw date from one
  hop instead of comparing).

Conclusion: `comparison_hint`'s existing wording (examples included, "not a date/number"
clause included) already sits at a reasonably good balance between **triggering on real
comparison questions** and **not over-triggering on non-comparison ones** — each of the two
follow-up attempts changed only one half of that balance and made things worse. Further
local patching of this prompt has diminishing returns; `comparison_hint` is the final
version, and this line of exploration is closed.

Next steps: backfill every number in `gate_v3_main_method_report.md` that references the
"current complete method," and consider rerunning arm B (gate v2 rawprefix) and arm C
(h-only rawprefix) with this new prompt as well.

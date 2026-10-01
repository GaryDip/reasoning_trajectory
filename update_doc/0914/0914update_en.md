# 2026-09-14 Update: Two Parallel Experiments (Decomposer Training-Data Ablation / Retrieval Precision Comparison)

This period ran two experiments at the same time. They look at **two different stages** of the pipeline, are
independent of each other, and are recorded in two parts:

| | Pipeline stage | Variable | Reference |
|---|---|---|---|
| **Part 1** | **Question decomposition** — the decomposer's training data | Drop the self-annotated 2Wiki data and train the decomposer on MuSiQue only | Production decomposer (MuSiQue + 2Wiki), i.e. the D-group in the 0831 report |
| **Part 2** | **Retrieval** — how much of what is retrieved is noise | Add precision metrics on top of recall | D-group vs ChainRAG vs GRITHopper (the three-way comparison in the 0831 report) |

Both parts use the D-group end-to-end configuration (gate v3, λ=0.60, distractor setting, comparison_hint
reader). Part 1 only swaps the decomposition file; Part 2 does not change the D-group configuration at all and
only adds a new evaluation metric.

---

# Part 1: Decomposer Training-Data Ablation — How Much Does the Self-Annotated 2Wiki Data Contribute?

## 1. Question

The BART decomposer used in end-to-end evaluation is trained on a mix of two datasets: MuSiQue's official
question-decomposition annotations (19,938 examples), plus decompositions we annotated ourselves with GPT on
the 2Wiki training set (10,444 examples). MuSiQue ships with decomposition annotations and 2Wiki does not, so
the latter is data we constructed. This experiment asks: **if the decomposer is trained only on MuSiQue's own
annotations, how much does end-to-end performance drop** — in other words, how much does the self-annotated
2Wiki data actually contribute?

The 2Wiki annotations were **sampled from the 2Wiki training set with fixed per-type quotas**: 10,500 examples
in total (bridge_comparison 5000 / comparison 2500 / compositional 1500 / inference 1500), annotated by
gpt-5-mini in MuSiQue's decomposition format. 10,444 succeeded (56 failures were dropped), and all of them went
into the training set, making up 34% of the 30,382-example mixed training set. A further 1,043 2Wiki dev
annotations are used only for checkpoint selection (together with MuSiQue dev) and are not trained on.

| Question type | Original 2Wiki train | Share | Annotated, in training set | Share | Sampling rate |
|---|---|---|---|---|---|
| bridge_comparison (4 hops) | 34,631 | 20.7% | 4,998 | **47.9%** | 14.4% |
| comparison | 51,963 | 31.0% | 2,453 | 23.5% | 4.7% |
| compositional | 76,481 | 45.7% | 1,500 | 14.4% | 2.0% |
| inference | 4,379 | 2.6% | 1,493 | 14.3% | 34.1% |
| Total | 167,454 | | 10,444 | | 6.2% |

By hop count: 5,446 two-hop and 4,998 four-hop examples (all four-hop examples are bridge_comparison). The
sampling is **deliberately skewed toward comparison questions**: comparison-type questions (bridge_comparison +
comparison) make up 52% of the original training set but 71% of the annotated data, and the four-hop
bridge_comparison type goes from 20.7% to 47.9%. Compositional bridge questions, whose structure is close to
MuSiQue's, were sampled at only 2%.

## 2. Setup

- **The only difference between the two decomposers is the training data**: both are BART-large with the same
  hyperparameters (lr 3e-5, batch 16, 3 epochs, seed 100). The MuSiQue-only model is trained on the 19,938
  examples; the mixed model on 19,938 + 10,444. Both use the epoch-3 checkpoint.
- **Everything downstream is unchanged**: both decompositions go through the same processing (BART output →
  converted to the format with `[Answer N]` placeholders → rewritten into natural-language sub-questions by
  Llama) and are then fed into the same D-group pipeline, with identical retrieval, gate, reader, prompts
  and λ.
- **The reference is the D-group row in Section 1.5 of the 0831 report**, which was not rerun. To know how large
  a gap counts as a real difference, the D-group was rerun in full with the same configuration: metrics moved by
  at most 0.2pt on MuSiQue, at most 0.06pt on 2Wiki, and were identical to every digit on HotpotQA. The gaps
  below are far larger than this noise; only differences of a few tenths of a point on MuSiQue need to be read
  with care.
- **Metrics match the D-group's**: recall@1 / recall@3 / chain@1 are position-aligned (the passage retrieved at
  hop j must match the annotated gold for hop j); EM / F1 follow the official HotpotQA definition.

## 3. The Decomposers on Their Own

**MuSiQue dev** (has official decomposition annotations, so it can be scored directly):

| Decomposer | BLEU | Hop-count accuracy | 2-hop | 3-hop | 4-hop |
|---|---|---|---|---|---|
| MuSiQue + 2Wiki (production) | 76.81 | 0.8366 | 0.982 | 0.782 | **0.489** |
| MuSiQue only | 75.94 | 0.8155 | 0.988 | 0.738 | 0.427 |

**2Wiki / HotpotQA dev** (no decomposition annotations; the only check is the predicted hop count against the
number of supporting facts):

| Dataset | Decomposer | Hop-count accuracy | Predicted hop distribution | Gold hop distribution |
|---|---|---|---|---|
| 2Wiki | MuSiQue + 2Wiki | 0.9651 | 2 hops 9689 / 3 hops 140 / **4 hops 2739** / 5 hops 8 | 2 hops 9595 / 3 hops 88 / 4 hops 2806 / 5+ hops 87 |
| | MuSiQue only | 0.7616 | 1 hop 10 / 2 hops 12428 / 3 hops 138 / **4 hops 0** | same as above |
| HotpotQA | MuSiQue + 2Wiki | 0.8763 | 1 hop 14 / 2 hops 6489 / 3 hops 853 / 4+ hops 49 | all 2 hops |
| | MuSiQue only | 0.9369 | 1 hop 40 / 2 hops 6938 / 3 hops 422 / 4+ hops 5 | same as above |

Two observations:

1. **The MuSiQue-only model never produces a four-hop decomposition** — not one out of 12.6k 2Wiki questions,
   while 2Wiki has 2806 four-hop questions. On MuSiQue it also tends to under-split: more than half of the
   four-hop questions get decomposed into three hops.
2. HotpotQA is all two-hop. The MuSiQue-only model splits more "conservatively" and actually gets a higher
   hop-count accuracy (0.937 vs 0.876).

## 4. End-to-End Results

Each cell is the MuSiQue-only result, with the change relative to the production decomposer (D-group) in
parentheses, in pt:

| Dataset | recall@1 | recall@3 | chain@1 | EM | F1 |
|---|---|---|---|---|---|
| MuSiQue | 0.7069 (+0.8) | 0.8474 (+0.4) | 0.5275 (+1.6) | 0.4419 (+0.3) | 0.5363 (+0.2) |
| 2WikiMultihopQA | 0.6907 (**−24.1**) | 0.8689 (−11.5) | 0.5651 (**−30.3**) | 0.5189 (**−8.3**) | 0.5976 (**−8.4**) |
| HotpotQA | 0.5825 (−8.0) | 0.8290 (−4.1) | 0.4704 (−9.1) | 0.5423 (−1.0) | 0.6792 (−0.9) |

Raw D-group numbers for reference: MuSiQue 0.6994 / 0.8435 / 0.5118 / 0.4386 / 0.5341, 2Wiki 0.9316 / 0.9842 /
0.8685 / 0.6023 / 0.6819, HotpotQA 0.6620 / 0.8701 / 0.5610 / 0.5519 / 0.6883.

## 5. Analysis: Where the Drop Comes From

### 5.1 2Wiki: Almost the Entire Drop Comes from Comparison Questions

Broken down by 2Wiki's official question types:

| Question type | # questions | EM change | F1 change | Hops produced by the MuSiQue-only model |
|---|---|---|---|---|
| bridge_comparison (4 hops) | 2751 | **−20.1pt** | −19.7pt | 2 hops 2621 / 3 hops 130 |
| comparison | 3040 | **−12.6pt** | −12.2pt | almost all 2 hops |
| inference | 1549 | −4.8pt | −4.8pt | 2 hops |
| compositional | 5236 | −0.8pt | −1.4pt | 2 hops |

- **Compositional (bridge) questions are almost unaffected**: their "find A, then use A to find B" structure is
  the same as MuSiQue's, and MuSiQue-only training already covers it.
- **Comparison questions drop the most**: for a question like "Who was born earlier, A or B?", the right
  decomposition is to look up A and B separately and then compare. MuSiQue has almost no questions with this
  structure, so the MuSiQue-only model does not know how to decompose them. bridge_comparison ("Whose director
  is older, A's or B's?") needs four hops; this model never produces four hops and collapses all of them into
  two, and EM drops by 20pt.

**Conclusion: the self-annotated 2Wiki data mainly teaches the decomposer how to decompose comparison
questions**, including composite comparisons that need four hops. These make up 46% of 2Wiki (5791 / 12576),
so removing this data costs 8pt overall on 2Wiki. This also matches the composition of the annotated data (see
the table in Section 1): comparison questions are 71% of the annotated data, and they are exactly the two types
that drop the most once it is removed; compositional questions are only 14% of the annotated data and are
structurally close to MuSiQue, and removing the data barely affects them.

### 5.2 MuSiQue: Slightly Up Overall, but Four-Hop Questions Drop

Broken down by annotated hop count:

| Gold hops | # questions | EM change | F1 change |
|---|---|---|---|
| 2 hops | 1252 | +1.8pt | +2.2pt |
| 3 hops | 760 | −0.1pt | −1.3pt |
| 4 hops | 405 | **−3.5pt** | −3.1pt |

The small overall gain is driven by two-hop questions (more than half of the set), and the longer the chain,
the larger the drop — consistent with Section 3's observation that the MuSiQue-only model tends to under-split.
**Adding the 2Wiki data also helps MuSiQue's own long-chain questions**, which matches the decomposer's own
metric (four-hop hop-count accuracy 0.427 → 0.489): the composite comparison questions in 2Wiki, which have four
pieces of evidence, teach the model to decompose long chains.

In addition, because the MuSiQue-only model produces fewer hops, only 5964 hops are scored under position
alignment (6037 for the reference). Fewer of the harder later hops get scored, so part of the small retrieval
gain comes from this rather than from retrieval actually getting better.

### 5.3 HotpotQA: Again, Comparison Questions Drop

HotpotQA is all two-hop. The MuSiQue-only model gets the hop count right more often, yet the position-aligned
retrieval metrics drop by 8–9pt while EM/F1 drop by only about 1pt. Broken down by HotpotQA's official
question types:

| Question type | # questions | EM change | F1 change | Selected passage is gold (position-aligned) | Selected passage is gold (any gold counts) |
|---|---|---|---|---|---|
| bridge | 5918 | −0.2pt | +0.0pt | 0.652 → 0.627 | 0.895 → 0.893 |
| comparison | 1487 | **−4.1pt** | **−4.6pt** | 0.699 → **0.396** | 0.896 → 0.821 |

- **At the answer level, the entire loss comes from comparison questions**, consistent with 2Wiki; bridge
  questions are unaffected.
- **At the retrieval level, most of the position-aligned drop is an ordering issue**: for a comparison question
  "which of A and B …", looking up A first or B first are both valid decompositions, but the annotation records
  only one order, so position alignment counts the reversed order as an error. Counting only "is the selected
  passage any gold passage of this question", bridge questions barely change (−0.2pt) and comparison questions
  drop by 7.5pt — that part is the real retrieval loss (a wrong decomposition leading to the wrong passage);
  the remaining ~23pt is ordering.

## 6. Summary

- **The self-annotated 2Wiki data is necessary**: without it, 2Wiki end-to-end EM drops 8.3pt and chain@1
  drops 30pt, and the loss comes almost entirely from comparison questions (bridge_comparison EM −20pt,
  comparison −12.6pt).
- **The effect on MuSiQue itself is structural**: flat overall, but four-hop EM drops 3.5pt, offset by a small
  gain on two-hop questions.
- **The effect on HotpotQA is small, but the mechanism is the same**: the overall answer drop is only about
  1pt, all of it from comparison questions (comparison EM −4.1pt, bridge −0.2pt); comparison questions are only
  20% of HotpotQA, so the overall impact is much smaller than on 2Wiki.
- **All three datasets point to the same conclusion**: the value of the self-annotated 2Wiki data is that it
  teaches the decomposer to decompose **comparison questions** and **long-chain questions**, two structures
  that are absent or rare in MuSiQue's official annotations.

---

# Part 2: Retrieval Precision Comparison — D-group vs ChainRAG vs GRITHopper

## 1. Motivation

The three-way comparison in the 0831 report only looked at recall-type metrics (per-hop recall@1/@3 and
per-question gold coverage), and did not answer "how much of what is retrieved is noise". This part adds
precision. All data comes from the three methods' existing per-question outputs; only the D-group was rerun,
with exactly the same configuration, in order to keep each hop's candidate ranking (the rerun results differ
from the reported numbers by no more than the noise described in Part 1, Section 2).

As in 0831, the three-way comparison can only use an **order-invariant** definition (GRITHopper's and
ChainRAG's hops have no positional correspondence to the gold passages), and can only match by **passage
title** (these two methods only store titles in their outputs).

## 2. Two Kinds of Precision

**1. Evidence-set precision**: the fraction of gold passages in the evidence set each question finally uses to
generate its answer.

$$\text{precision} = \frac{|\text{retrieved set} \cap \text{gold}|}{|\text{retrieved set}|}$$

The "retrieved set" is the same set used for 0831's gold coverage (for the D-group and GRITHopper, the passage
finally selected at each hop; for ChainRAG, the passages behind every sentence placed into the context after
graph expansion). Coverage is the same quantity with the gold count as the denominator.

For the D-group and GRITHopper, this metric is **numerically almost identical to recall@1**: both take exactly
one passage per hop, namely the top-ranked one, so "how much of the final evidence is gold" and "is each hop's
top-1 gold" count the same things (the difference is under 0.5pt). The new information it adds is therefore
only about ChainRAG.

**2. Per-hop precision@3**: the number of gold passages among each hop's top 3, divided by 3.

How it differs from recall@3: the "recall@3" in the 0831 report (strictly speaking hit@3) only checks
**whether** at least one gold passage is in the top 3, so it is 0 or 1; precision@3 checks **how many**. When
the top 3 is [gold A, noise, noise] or [gold A, gold B, noise], recall@3 is 1 in both cases while precision@3 is
1/3 and 2/3 respectively. All three methods' recall@3 values are squeezed into 0.96–0.99 and cannot be told
apart; precision@3 can separate them.

Note that precision@3 has a ceiling: when a question has only 2 gold passages, the maximum is 2/3. Every
HotpotQA question has 2, so its ceiling is 0.667. **Absolute precision@3 values therefore cannot be compared
across datasets**; methods can only be compared within the same dataset.

## 3. Results

### 3.1 Evidence-Set Precision

| Dataset | Method | Precision | Coverage | Set F1 | Avg. # retrieved | Avg. # gold |
|---|---|---|---|---|---|---|
| MuSiQue | D-group | 0.8837 | 0.7958 | 0.8275 | 2.28 | 2.60 |
| | ChainRAG | 0.2068 | **0.9587** | 0.3235 | 14.54 | |
| | GRITHopper | **0.9091** | 0.8207 | **0.8517** | 2.31 | |
| 2WikiMultihopQA | D-group | **0.9673** | 0.9408 | **0.9496** | 2.37 | 2.44 |
| | ChainRAG | 0.3559 | **0.9870** | 0.4998 | 7.61 | |
| | GRITHopper | 0.9540 | 0.8922 | 0.9139 | 2.24 | |
| HotpotQA | D-group | 0.8955 | 0.8532 | 0.8650 | 1.92 | 2.00 |
| | ChainRAG | 0.2752 | **0.9864** | 0.4215 | 7.86 | |
| | GRITHopper | **0.9658** | **0.9464** | **0.9528** | 1.95 | |

(Precision, coverage and set F1 are computed per question and then averaged.)

### 3.2 Per-Hop Precision@3

| Dataset | Method | # hops | recall@1 | recall@3 | **precision@3** | Avg. # gold in top 3 |
|---|---|---|---|---|---|---|
| MuSiQue | D-group | 6037 | 0.8836 | 0.9742 | 0.5126 | 1.54 |
| | ChainRAG | 5162 | 0.6716 | 0.8628 | 0.3949 | 1.18 |
| | GRITHopper | 6404 | **0.9058** | **0.9863** | **0.6087** | 1.83 |
| 2WikiMultihopQA | D-group | 30621 | **0.9724** | **0.9972** | 0.6158 | 1.85 |
| | ChainRAG | 22737 | 0.9199 | 0.9785 | 0.5859 | 1.76 |
| | GRITHopper | 30654 | 0.9500 | 0.9964 | **0.6786** | 2.04 |
| HotpotQA | D-group | 14796 | 0.8988 | 0.9805 | 0.5035 | 1.51 |
| | ChainRAG | 13658 | 0.8032 | 0.9496 | 0.4679 | 1.40 |
| | GRITHopper | 14810 | **0.9658** | **0.9959** | **0.5994** | 1.80 |

GRITHopper's and ChainRAG's recall@1/@3 match the 0831 report to every digit. The D-group's MuSiQue recall@1 is
0.8836 here versus 0.8552 in the report. The difference comes from the matching method: the report matches by
paragraph index, while here only title matching is possible, and MuSiQue has paragraphs that share a title (in
about 6.7% of questions the D-group selected a paragraph with the same title as a gold paragraph that is not
itself gold, which title matching counts as a hit). On 2Wiki and HotpotQA the two matching methods give the same
result. GRITHopper and ChainRAG are probably inflated by a similar amount on MuSiQue, but this cannot be
quantified.

## 4. Interpretation

**ChainRAG's high coverage is bought with volume.** On MuSiQue it sends the reader 14.5 passages per question
on average against only 2.6 gold ones, so about 80% of its context is noise. This quantifies the explanation in
Section 2.5.1 of the 0831 report (graph expansion "casting a wide net" dilutes the correct signal): the highest
coverage of all three methods, yet the lowest EM/F1. Once precision and coverage are combined into set F1, the
ranking becomes GRITHopper ≈ D-group ≫ ChainRAG.

**Per-hop precision@3 ranks the methods the same way on all three datasets: GRITHopper > D-group > ChainRAG.**
The gap between the D-group and GRITHopper, however, has to be read in light of how each method queries:

- GRITHopper uses the **full original question** as the query at every step, so all of the question's gold
  passages rank high at every step — on MuSiQue its top 3 contains 1.83 gold passages on average.
- The D-group first decomposes the question into sub-questions, and each hop's query targets **only the one
  passage that hop needs**. That other hops' gold passages do not rank high is simply a sign that the query is
  focused — its top 3 contains 1.54 gold passages on average.

So precision@3 partly measures "how focused the query is" here, and decomposition-based methods (D-group,
ChainRAG) are naturally at a disadvantage. **The D-group scoring below GRITHopper mainly reflects the difference
between the two querying styles, not worse retrieval by the D-group** — the D-group's per-hop top-1 hit rate is
actually higher than GRITHopper's on 2Wiki (0.9724 vs 0.9500), and top-1 is the evidence the D-group actually
commits.

**The D-group and ChainRAG are both decomposition-based, so their querying styles are comparable.** The
D-group's precision@3 is higher than ChainRAG's on all three datasets (MuSiQue +11.8pt, 2Wiki +3.0pt,
HotpotQA +3.6pt), and it leads on recall@1 across the board: more gold and less noise in the top 3. However,
the two differ in both decomposition (BART decomposer vs ChainRAG's own LLM decomposition) and ranking (BGE +
gate reranking vs embedding + cross-encoder), so this gap cannot be attributed to the gate alone.

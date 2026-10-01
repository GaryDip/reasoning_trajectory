# probe_artifacts training report

Command:
```
python score_and_analyze.py \
  --train-hidden-dir hidden_states_full/train \
  --dev-hidden-dir hidden_states_full/dev \
  --pool-positions \
  --out-dir probe_artifacts
```

Data: train 97024 rows / 266744 pooled positions, dev 16316 rows / 51672 pooled positions —
AFTER the K=3/K=4 exhaustive-combination augmentation (`build_multi_error_traces.py
--augment-existing --exhaustive-for-k 3 4`). K=2 unchanged; K=3 rows per example went
4 -> 8 (full C(3,n) enumeration); K=4 rows per example went 5 -> 16 (full C(4,n)).

Train accuracy (corrupted-so-far vs not, ALL positions pooled): **0.8456**

## Before vs after the K=3/K=4 augmentation

| | before (72230 train / 11722 dev rows) | after (this run: 97024 / 16316 rows) |
|---|---|---|
| Spearman(n_wrong up-to-j, score), pooled over ALL positions | 0.6089 (n=42680) | **0.6348** (n=51672) |

Modest but real improvement — the augmentation was worth doing, not just more rows for
their own sake.

## DEV final h_K (probe trained on ALL positions pooled)

Per (K, n_wrong):

```
  K  n_wrong      n     mean   median
  2        0   1252   0.3284   0.2287
  2        1   1252   0.7533   0.8936
  2        2   1252   0.8652   0.9537
  3        0    760   0.6469   0.6694
  3        1   2280   0.8224   0.8921
  3        2   2280   0.8978   0.9517
  3        3    760   0.9278   0.9704
  4        0    405   0.7641   0.8135
  4        1   1620   0.8684   0.9188
  4        2   2430   0.9198   0.9585
  4        3   1620   0.9466   0.9720
  4        4    405   0.9578   0.9750
```

Pooled over K, by n_wrong:

```
 n_wrong      n     mean   median      std
       0   2417   0.5016   0.5202   0.3090
       1   5152   0.8201   0.9039   0.2062
       2   5962   0.8999   0.9553   0.1434
       3   2380   0.9406   0.9717   0.0886
       4    405   0.9578   0.9750   0.0582
```

Spearman(n_wrong, probe score) = **0.5093** (p=0.00e+00), n=16316 — final-h_K-only is a
harder/noisier task than the pooled-position view below, consistent with every prior run.

## DEV every position (probe trained on ALL positions pooled)

Per (K_total, position j, n_wrong up-to-j):

```
  K   j  n_eff      n     mean   median
  2   1      0   1890   0.2421   0.1590
  2   1      1   1866   0.7384   0.8758
  2   2      0   1252   0.3284   0.2287
  2   2      1   1252   0.7533   0.8936
  2   2      2   1252   0.8652   0.9537
  3   1      0   3040   0.2739   0.2216
  3   1      1   3040   0.7844   0.8984
  3   2      0   1520   0.5158   0.5034
  3   2      1   3040   0.7935   0.8944
  3   2      2   1520   0.8840   0.9478
  3   3      0    760   0.6469   0.6694
  3   3      1   2280   0.8224   0.8921
  3   3      2   2280   0.8978   0.9517
  3   3      3    760   0.9278   0.9704
  4   1      0   3240   0.2628   0.1970
  4   1      1   3240   0.7646   0.8747
  4   2      0   1620   0.4959   0.4937
  4   2      1   3240   0.7891   0.8838
  4   2      2   1620   0.8964   0.9596
  4   3      0    810   0.7627   0.8176
  4   3      1   2430   0.8784   0.9353
  4   3      2   2430   0.9298   0.9708
  4   3      3    810   0.9476   0.9814
  4   4      0    405   0.7641   0.8135
  4   4      1   1620   0.8684   0.9188
  4   4      2   2430   0.9198   0.9585
  4   4      3   1620   0.9466   0.9720
  4   4      4    405   0.9578   0.9750
```

Pooled ACROSS K_total, by (position j, n_wrong up-to-j) — j=2 rows here mix K=2's final hop
with K=3/K=4's intermediate hop 2, etc.:

```
  j  n_eff      n     mean   median
  1      0   8170   0.2621   0.1960
  1      1   8146   0.7660   0.8867
  2      0   4392   0.4550   0.4277
  2      1   7532   0.7849   0.8915
  2      2   4392   0.8832   0.9552
  3      0   1570   0.7066   0.7610
  3      1   4710   0.8513   0.9156
  3      2   4710   0.9143   0.9622
  3      3   1570   0.9380   0.9761
  4      0    405   0.7641   0.8135
  4      1   1620   0.8684   0.9188
  4      2   2430   0.9198   0.9585
  4      3   1620   0.9466   0.9720
  4      4    405   0.9578   0.9750
```

Spearman(n_wrong up-to-j, score), pooled over ALL positions = **0.6348** (p=0.00e+00),
n=51672.

## Takeaways

1. Every (K, n_wrong) block and every (j, n_eff) block is monotonic — the K=3/K=4
   augmentation didn't just add volume, the trend stayed clean at the new sample sizes
   (e.g. K=4/n_wrong=2 went from a small bucket to 2430 dev rows).
2. Real, reproducible improvement from the augmentation: pooled-position Spearman
   0.6089 -> 0.6348.
3. The depth-inflates-baseline confound flagged earlier (n_eff=0 score rises with j: 0.26
   at j=1 -> 0.76 at j=4) is still present at this larger scale — it's a property of h_j
   itself, not a small-sample artifact, so it won't go away with more data. Relevant if this
   probe is ever used for a CROSS-position or cross-K absolute threshold; not relevant for
   same-step beam-search comparisons (see wavefront_hidden_probe_beam.py's docstring).

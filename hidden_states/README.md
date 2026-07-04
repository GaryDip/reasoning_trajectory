# hidden_states — 从 trace 提取的 activation

## ⚠️ `.npz` 未提交到 git

`train/activations/` 和 `dev/activations/` 下的 `.npz` 文件（共 92,748 个，约 2.4GB）因体积过大未纳入 git 仓库，已在根目录 `.gitignore` 中通过 `**/*.npz` 排除。`manifest.jsonl` 和 `run_meta.json` 已提交。

需要这些 activation 时，用下面的脚本重新提取（依赖已提交的 `traces/merged/` trace 数据，以及本地的 Llama-3.1-8B-Instruct 模型）：

```bash
cd hidden_states
./run_extract_train.sh                 # 或
python extract_hidden_states.py --split train --trace-file ../traces/merged/musique/train.jsonl --device cuda:0 --dtype bfloat16 --resume
```

## 目录布局

```
hidden_states/
├── train/
│   ├── activations/
│   │   ├── pos/*.npz
│   │   └── neg/*.npz
│   └── manifest.jsonl
└── dev/
    ├── activations/
    │   ├── pos/*.npz
    │   └── neg/*.npz
    └── manifest.jsonl
```

## 每个 .npz 内容

- `h_0 .. h_{K+1}`：各 transition prefix 的 last-token hidden（layer 31）
- `wrong_hops`, `first_wrong_hop`, `K`, `id`, `trace_type`

## Δ 特征（gate 训练用）

对 transition j：`Δ_j = h_j - h_{j-1}`（在 `gate/fit_lr_gate_pooled.py` 中计算）

## pos / neg 划分

| trace_type | 目录 | should_intervene 正类 |
|------------|------|----------------------|
| correct (gold) | `pos/` | 无（全 0） |
| error (counterfactual) | `neg/` | hop ∈ wrong_hops 的 transition |

## 脚本

`extract_hidden_states.py` — 读 `traces/merged/` jsonl，forward Llama，写 npz + manifest

**Forward：** 每个 cumulative prefix 独立 full forward；先按 **token 长度分桶**（`--length-bucket-tokens`，默认 256），桶内 left-pad batch（`--batch-size`）。

```bash
cd multihop_trace/hidden_states
python extract_hidden_states.py \
  --split train \
  --trace-file ../traces/merged/musique/train.jsonl \
  --device cuda:0 --dtype bfloat16 --resume
```

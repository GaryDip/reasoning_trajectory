# multihop_trace

Multi-hop retrieval gate pipeline：从原始数据集 → 构造 trace → 提取 hidden state → 训练 LR gate → 端到端召回实验。

## 目录结构

```
multihop_trace/
├── config/                 # 路径与超参配置
├── data/
│   ├── raw/                # 数据集原始文件
│   └── decompose/          # 分解 jsonl（gt / bart，供下游读）
├── decompose/              # 问题分解
│   ├── gt/                 # MuSiQue GT NL + exact_gt_traces
│   └── bart/               # BART 训练/预测（train.py, run_train.sh）
├── traces/                 # 构造的 reasoning trace
├── hidden_states/
├── gate/
└── retrieval/
```

## 流水线（5 步）

| 步骤 | 目录 | 脚本 | 输入 → 输出 |
|------|------|------|-------------|
| 0 | `data/raw` | — | MuSiQue / 2Wiki / Hotpot 原始 json/jsonl |
| 1a | `decompose/gt` | `gt_decompose_to_nl.py` | MuSiQue GT 分解 → NL jsonl |
| 1b | `decompose/bart` | `decompose_to_nl.py` | 三数据集 → BART NL 分解 |
| 2 | `traces/` | `construct_*_traces.py` | **GT** decompose + gold → trace jsonl |
| 3 | `hidden_states/` | `extract_hidden_states.py` | trace → activations |
| 4 | `gate/` | `fit_lr_gate_pooled.py` | hidden_states → joblib |
| 5 | `retrieval/` | `run_wavefront_all.sh` | **BART** decompose + gate → metrics |

## 快速开始

```bash
cd /home/ertai/reasoning_trajectory/multihop_trace

# 1. 检查数据
ls data/raw/musique/

# 2. 构造 trace（示例，MuSiQue dev）
python traces/construct_gold_traces.py --dataset musique --split dev
python traces/construct_counterfactual_traces.py --dataset musique --split train

# 3. 提取 hidden state
python hidden_states/extract_hidden_states.py --split train --trace-dir traces/gold

# 4. 训练 pooled LR gate
python gate/fit_lr_gate_pooled.py --hs-root hidden_states --out-dir gate/artifacts_pooled

# 5. 端到端召回
./retrieval/run_wavefront_all.sh
```

详细说明见各子目录下的 `README.md`。

## 数据集

| 数据集 | raw 路径 | 设定 |
|--------|----------|------|
| MuSiQue | `data/raw/musique/` | in-pool (~20 段) |
| 2WikiMultihopQA | `data/raw/2wikimultihopqa/` | in-pool |
| HotpotQA | `data/raw/hotpotqa/` | distractor (10 段) |

## Trace 构造策略

- **gold/**：每 hop 用 gold evidence → 全部 transition label = 不干预
- **counterfactual/**：对每个 hop h 注入 wrong evidence → `wrong_hops=[h]`，仅 transition j=h-1 为正类

Label 协议与 `gate/hop_labels.py` 一致。

# decompose — 问题分解（GT 标注 + BART 模型）

MuSiQue GT 自然语言分解、exact-GT trace，以及 BART decomposer 的训练与推理。

## 子目录

```
decompose/
├── gt/                              # MuSiQue GT 分解
│   ├── musique_gt_nl_train.jsonl
│   ├── musique_gt_nl_dev.jsonl
│   ├── train_exact_gt_traces.jsonl
│   ├── dev_exact_gt_traces.jsonl
│   └── scripts/
│       ├── gt_decompose_to_nl.py
│       └── build_exact_gt_trace.py
│
└── bart/                            # BART decomposer（训练 + 推理）
    ├── train.py, predict.py, data_utils.py, metrics.py
    ├── run_train.sh                 # ★ 训练入口
    ├── run_predict_musique_2wiki_hotpot.sh
    ├── data/                        # 训练用官方格式 jsonl
    └── outputs/bart_decomposer_musique_2wiki_repro/
```

## 两套分解

| | GT (`gt/`) | BART (`bart/`) |
|--|-----------|----------------|
| 来源 | MuSiQue `question_decomposition` + GPT 转 NL | 自训 BART |
| 数据集 | **仅 MuSiQue** | MuSiQue / 2Wiki / Hotpot dev |
| 用途 | 构造训练 trace、exact-GT hidden state | 端到端 retrieval eval |
| 下游路径 | `data/decompose/musique/gt/` | `data/decompose/{ds}/bart/` |

## GT 标注流程

1. MuSiQue 原始数据含 `question_decomposition` 与 gold paragraph idx
2. `gt_decompose_to_nl.py` 将结构化分解改写为 NL sub-questions
3. `build_exact_gt_trace.py` 拼接 NL + gold evidence → reasoning trace

## 常用命令

```bash
# 重新生成 GT NL（需 OpenAI API）
cd decompose/gt/scripts
python gt_decompose_to_nl.py --split train --max 0

# 构建 exact-GT trace
python build_exact_gt_trace.py \
  --nl-decomp ../musique_gt_nl_train.jsonl \
  --musique ../../../data/raw/musique/musique_ans_v1.0_train.jsonl \
  --output ../train_exact_gt_traces.jsonl

# BART 推理（三数据集 dev）
cd ../../bart
./run_predict_musique_2wiki_hotpot.sh
./run_convert_predictions_to_v1.sh
./run_decompose_to_nl_all.sh
```

## 与 `data/decompose/` 的关系

`data/decompose/` 是 `traces/` 和 `retrieval/` 读取的统一入口；更新本目录产出后，同步到 `data/decompose/` 即可。

# BART Decomposer — 训练与推理

MuSiQue 官方格式的 BART-large 多跳问题分解器：训练、预测、转 NL。

## 代码

| 文件 | 作用 |
|------|------|
| `train.py` | HuggingFace Seq2Seq 训练 BART |
| `predict.py` | 对 dev 集做 beam search 分解 |
| `data_utils.py` | MuSiQue `[[CQS]]/[[CQE]]` 格式读写 |
| `metrics.py` | 按 gold hop 数评估 |
| `convert_predictions_to_v1.py` | raw 预测 → v1 hop 文本 |
| `decompose_to_nl.py` | v1 → 自然语言 sub-questions（Llama） |
| `annotate_2wiki_gpt_decompose.py` | GPT 标注 2Wiki 混合训练集 |

## 训练数据

```
data/
├── musique_raw/
│   ├── musique_ans_gold_context_version_train.jsonl   # MuSiQue 官方 BART 格式
│   └── musique_ans_gold_context_version_dev.jsonl
├── 2wiki_gpt_mixed_train.jsonl                        # 2Wiki GPT 标注（混合采样）
└── 2wiki_gpt_mixed_dev.jsonl
```

MuSiQue 使用 `component_question_texts`（`[[CQS]] 0 [[CQE]] ...` 格式）；2Wiki 由 `annotate_2wiki_gpt_decompose.py` 生成同格式记录。

## 一键脚本

```bash
cd decompose/bart

# 1. （可选）重新标注 2Wiki 训练数据 — 需 OpenAI API
./run_annotate_2wiki.sh

# 2. 训练 BART（MuSiQue + 2Wiki 混合）
./run_train.sh

# 仅 MuSiQue
TRAIN_MUSIQUE_ONLY=1 ./run_train.sh

# 3. 三数据集 dev 预测
./run_predict_musique_2wiki_hotpot.sh

# 4. raw → v1 → NL
./run_convert_predictions_to_v1.sh
./run_decompose_to_nl_all.sh
```

训练产出在 `outputs/bart_decomposer_musique_2wiki_repro/`；NL 预测同步到 `data/decompose/{dataset}/bart/dev_nl.jsonl` 供 retrieval 使用。

## 已有 checkpoint

`outputs/bart_decomposer_musique_2wiki_repro/checkpoint-1425/` 含完整 `model.safetensors`，可直接用于 `predict.py`。

默认超参（MuSiQue + 2Wiki 混合训练，与 `checkpoint-1425` 的 `train_meta.json` 一致）见 `TRAINING_NOTES.md`：

- `learning_rate=3e-5`
- `batch_size=16`
- `max_target_length=100`
- `--bf16`

## 输出流水线

```
train.py → checkpoint-*/
predict.py → raw_predictions/*.jsonl
convert_predictions_to_v1.py → v1_predictions/*.jsonl
decompose_to_nl.py → nl_predictions/*.jsonl → data/decompose/*/bart/dev_nl.jsonl
```

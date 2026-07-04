# BART Decomposer Training Notes

## 环境

稳定训练环境：`transformers==4.57.3`，CUDA 与 PyTorch 版本需匹配（见 checkpoint `config.json`）。

## 训练数据

| 文件 | 说明 |
|------|------|
| `data/musique_raw/musique_ans_gold_context_version_train.jsonl` | MuSiQue 官方 BART 格式 train |
| `data/musique_raw/musique_ans_gold_context_version_dev.jsonl` | MuSiQue dev |
| `data/2wiki_gpt_mixed_train.jsonl` | 2Wiki GPT 混合采样 train |
| `data/2wiki_gpt_mixed_dev.jsonl` | 2Wiki GPT 混合采样 dev |

## MuSiQue + 2Wiki 混合训练

```bash
cd decompose/bart
./run_train.sh
```

等价命令：

```bash
python train.py \
  --train-file \
    data/musique_raw/musique_ans_gold_context_version_train.jsonl \
    data/2wiki_gpt_mixed_train.jsonl \
  --dev-file data/musique_raw/musique_ans_gold_context_version_dev.jsonl \
  --output-dir outputs/bart_decomposer_musique_2wiki_repro \
  --learning-rate 3e-5 \
  --batch-size 16 \
  --max-target-length 100 \
  --bf16
```

2Wiki 混合集由 `annotate_2wiki_gpt_decompose.py` 生成，采样 preset 见脚本内 `TYPE_SAMPLE_PRESETS`。

Checkpoint 步数随训练样本量变化；当前最佳 checkpoint 为 `checkpoint-1425`。

## 仅 MuSiQue

```bash
TRAIN_MUSIQUE_ONLY=1 ./run_train.sh
```

## 推理与 NL 转换

见 `README.md` 中的 `run_predict_*` / `run_convert_*` / `run_decompose_to_nl_all.sh`。

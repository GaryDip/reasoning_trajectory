# 模型权重未提交

本目录（以及各 `checkpoint-*/`）下的 `model.safetensors`、`optimizer.pt`、`scheduler.pt`、`scaler.pt` 因体积过大（单个 checkpoint 约 4.8GB）未纳入 git 仓库，已在 `.gitignore` 中排除。

## 如何得到这个模型

在 `decompose/bart/` 下运行：

```bash
./run_train.sh
```

超参（与本目录 `train_meta.json` 一致）：`learning_rate=3e-5`、`batch_size=16`、`max_target_length=100`、`epochs=3`、`--bf16`，训练数据为 MuSiQue + 2Wiki GPT 混合集（见同目录 `README.md`）。

推荐使用的 checkpoint 是 `checkpoint-1425`（训练效果最好，见 `eval_by_hop_latest.json`）。

其余文件（`config.json`、tokenizer 相关文件、`train_meta.json`、`trainer_state.json`、`eval_by_hop_*.json`、`raw_predictions/`、`v1_predictions/`、`nl_predictions/`）均已提交，可用于核对复现结果。

# 模型权重未提交

本目录（以及各 `checkpoint-*/`）下的 `model.safetensors`、`optimizer.pt`、`scheduler.pt`、`scaler.pt` 因体积过大未纳入 git 仓库，已在 `.gitignore` 中排除。

## 这个 checkpoint 是什么

早期的 MuSiQue-only 训练结果（迁移前用 `/home/ertai/reasoning_trajectory/multihop_trajectory/decompose-musique/...` 下的数据训练，见 `train_meta.json` 中记录的路径），保留作参考，不是当前推荐使用的模型。当前使用的模型见 `../bart_decomposer_musique_2wiki_repro/MODEL_NOTE.md`。

## 如何重新得到这个模型

```bash
cd decompose/bart
TRAIN_MUSIQUE_ONLY=1 ./run_train.sh
```

超参（与本目录 `train_meta.json` 一致）：`learning_rate=3e-5`、`batch_size=16`、`max_target_length=100`、`epochs=3`。

其余文件（`config.json`、tokenizer 相关文件、`train_meta.json`、`trainer_state.json`、`eval_by_hop_*.json`）均已提交。

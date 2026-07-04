# gate — LR gate 训练

在 hidden-state Δ 向量上训练 **PCA + Logistic Regression**，判断当前 transition 是否应触发检索干预。

## 目录布局

```
gate/
├── hop_labels.py           # label 协议（共享）
├── lr_artifacts.py         # 加载 / 打分工具
├── fit_lr_gate.py          # per-(K,j) 版本
├── fit_lr_gate_pooled.py   # ★ 推荐：按语义 transition j 池化
├── apply_lr_gate_pooled.py # 离线 dev 评估
├── artifacts/              # non-pooled joblib
├── artifacts_pooled/       # pooled 默认
├── results_pooled/         # dev 分数与 metrics
└── viz/                    # transition + LR 决策面可视化
    ├── plot_lr_planes.py
    ├── transition_viz_utils.py
    └── results/figures/
```

## Pooled 协议

| j | transition | 用于 K |
|---|------------|--------|
| 0 | Q→E1 | 2,3,4 |
| 1 | E1→E2 | 2,3,4 |
| 2 | E2→E3 | 3,4 |
| 3 | E3→E4 | 4 |

不含 final transition（E_K→Final），因无对应 evidence selection hop。

## Label

```python
should_intervene(j) = (j + 1) in wrong_hops  # 仅 error trace
```

## 训练

```bash
python gate/fit_lr_gate_pooled.py \
  --hs-root ../hidden_states \
  --artifacts-dir gate/artifacts_pooled \
  --target-fpr 0.15
```

产出：`artifacts_pooled/j{0..3}.joblib` + `meta.json`（pooled, target FPR=0.15）

## 下游

`retrieval/` 通过 `load_artifacts(artifacts_pooled, mode="pooled")` 在线打分。

## 可视化（LR 决策平面）

```bash
# dev 上四个 pooled transition 的决策面图
./gate/viz/run_plot_lr_planes.sh

# 只看 j=0 (Q→E1)
python gate/viz/plot_lr_planes.py --js 0 --split dev

# 3D：transition Δ → 3D t-SNE，红=应触发 / 蓝=不应触发，LR 决策面
python gate/viz/plot_transition_3d.py --js 0 --save-html
./gate/viz/run_plot_transition_3d.sh
```

产出在 `gate/viz/results/figures/`：
- **LR-normal plane**：沿 `w/‖w‖` 投影，决策边界是一条竖线（64-D 超平面的 2-D 截面）
- **PC1×PC2**：固定 `z₃…=0` 的截面 + score 等高线 + 线性边界
- **Score 直方图** + **t-SNE**（PCA 空间内）

**3D 版**（`plot_transition_3d.py` → `results/figures_3d/`）：
- 左：PC1–PC3 散点 + LR **精确平面**（64-D 超平面在 3D 上的截面）
- 右：PCA-64 → **3D t-SNE**，红/蓝=应/不应触发；LR 超平面采样点与数据 **联合 t-SNE**，呈现为绿色曲面（非线性嵌入里平面变曲面，属正常）
- 加 `--save-html` 可输出可旋转的 plotly 页面

旧版 per-(K,j) t-SNE 参考：`llama_gt_decompose_trace/transition_viz/`。

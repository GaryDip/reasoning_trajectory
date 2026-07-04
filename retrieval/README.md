# retrieval — 端到端召回实验

BART 分解 → BGE cosine retrieve → LR gate rerank → vLLM select/answer → final reader。

## 目录布局

```
retrieval/
├── run_retrieval_exp.py        # 主 pipeline（多 method）
├── run_retrieval_exp_wavefront.py  # batch vLLM 版（推荐）
├── run_wavefront_all.sh        # 三数据集一键跑
├── merge_retrieval_results.py
├── compute_answer_metrics_from_cases.py
├── results/                    # 按 session 时间戳归档
│   └── YYYYMMDD_HHMMSS_*/
└── logs/
```

## Methods

| method | 说明 |
|--------|------|
| `baseline` | BGE top-k，无 gate |
| `lr_rerank` | cosine + λ·LR score 混合排序 |
| `gated_rule_a` | top-1 abnormal 超阈值则 expand rerank |
| `oracle` | 上界（gold passage 在 pool 内） |

## 示例

```bash
# 单数据集 smoke test
GPUS=0 LIMIT=20 DATASETS=musique ./run_wavefront_all.sh

# 全量三数据集
GPUS=0,1,2 LAMBDA_LR=0.25 ./run_wavefront_all.sh
```

## 输出

每个 dataset 目录：

- `retrieval_exp_dev_{dataset}_*.json` — 汇总 metrics
- `retrieval_cases_dev_{dataset}_*.jsonl` — per-case 明细

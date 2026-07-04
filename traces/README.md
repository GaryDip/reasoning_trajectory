# traces — 构造 reasoning trace（LR gate 训练用）

确定性构造 trace 与 label，不依赖 LLM 自然犯错。

## 构造协议

| 类型 | sub_questions | evidence | label |
|------|---------------|----------|-------|
| **正例** | GT 自然语言分解 | 每 hop gold passage | `wrong_hops=[]` |
| **负例** | GT 自然语言分解 | hop 1..h-1 gold；**hop h = cosine top-k 非 gold**；h+1..K gold | `wrong_hops=[h]` |

负例 query：第 h hop 的 sub_question，其中 `[Answer N]` 用 gold hop answer 展开（与线上一致）。

## 目录

| 目录 | 内容 |
|------|------|
| `gold/` | 全 gold 正例 |
| `counterfactual/` | 全部负例（未平衡） |
| `merged/` | 正例 + 按 (K, hop) 平衡后的负例 |

## 脚本

| 脚本 | 作用 |
|------|------|
| **`construct_balanced_traces.py`** | **推荐**：一次跑完，同时写 `gold/` + `counterfactual/` + `merged/` |
| `construct_gold_traces.py` | 仅正例（单独调试） |
| `construct_counterfactual_traces.py` | 仅负例（单独调试） |
| `merge_traces.py` | 把已有 gold+cf jsonl 简单拼接 |
| `run_construct_musique_train.sh` | 调用 `construct_balanced_traces.py` |

## 数据量与平衡

每个 K-hop 样本：
- 1 条正例
- 最多 `K × max_wrong_per_hop` 条负例（每 hop 取 cosine top-k 里前 N 个非 gold）

`construct_balanced_traces.py` 对每个 `(K, hop h)` 桶：
- 目标负例数 = `pos_count[K] × balance_ratio`
- 不足则 **upsample**（4-hop 稀有桶重点受益）

## 示例

```bash
cd multihop_trace/traces

# 一条命令出 gold / counterfactual / merged 三个文件
python construct_balanced_traces.py --dataset musique --split train

# 或
./run_construct_musique_train.sh
```

提取 hidden state 用 **`merged/musique/train.jsonl`**。

## JSONL 字段

```json
{
  "id": "2hop__xxx__wrong_h2__cos0",
  "trace_type": "error",
  "wrong_hops": [2],
  "K": 2,
  "wrong_hop": 2,
  "wrong_mode": "cosine_topk_non_gold",
  "wrong_paragraph_idx": 7,
  "cosine_score": 0.82,
  "reasoning_trace": "Question: ..."
}
```

## Hidden state 训练

`trace_type=correct` → `hidden_states/.../pos/`  
`trace_type=error` → `hidden_states/.../neg/`  
transition j = h-1 上 `should_intervene=True` 当且仅当 `h ∈ wrong_hops`

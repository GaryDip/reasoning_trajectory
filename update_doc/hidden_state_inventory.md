# Hidden state 提取 / 打分现状清单

纯盘点,不改动任何文件。目的是把现在散落在 `hidden_states/`、`gate/`、`error_propagation_probe/`、
`rerank_listwise_contrastive/`、`rerank_pairwise_mlp/`、`adaptive_lambda_rerank/`、
`combined_gate_rerank/`、`retrieval/results/` 里的 hidden-state 提取、打分器 artifact、候选池级
缓存、端到端结果这四类东西理清楚:谁是谁的数据源、谁复用谁、别混着用。

## 全局结构:三层

```
第一层  底层 hidden state 提取(过一次 Llama,把隐藏状态存到磁盘)
   ↓
第二层  打分器 artifact(PCA/LR/MLP,在第一层数据上训练出来的 joblib/pt)
   ↓
第三层  应用:
        (a) 候选池级缓存 —— 把第二层的打分器/或原始向量应用到"真实检索出来的候选池"上，存成 npz，供离线实验反复用
        (b) 端到端 wavefront 结果 —— 真跑一次检索+生成，出 retrieval_exp_*.json / retrieval_cases_*.jsonl
```

## 第一层:三份互相独立的底层 hidden state 提取

**不要混用** —— 三份用的 trace 来源、层、粒度都不一样,拿错了会导致训练/打分对不上。

| 目录 | 来自哪个 trace | 层 | 粒度 | 用途 |
|---|---|---|---|---|
| `hidden_states/{train,dev}/` | `traces/merged/musique/{split}.jsonl` | 单层(31) | 每条 trace 只存**最终** h_K | 最老的版本,给 `gate/fit_lr_gate.py`(非 pooled)用,`rerank_pairwise_mlp` 复用它构造 pair(不重新提取) |
| `hidden_states/pilot_multilayer/` | 同上 `traces/merged/musique/{split}.jsonl`,但按 (K, hop) 分桶重采样(每桶上限 1500) | 多层(7/15/23/31) | 每个 (K, hop) 一条正例一条负例 | **gate_v2/gate_v3(pooled)的训练数据源**,`gate/artifacts_pooled_v2`、`gate/gate_v3/artifacts_pooled_v3` 都从这里读 |
| `error_propagation_probe/hidden_states_full/` | `error_propagation_probe/data/musique/{split}_multi_error.jsonl`(自己独立构造的 trace,允许多跳同时注错,跟 `traces/merged` 不是同一批) | 单层(31) | 每条 trace **每个中间 prefix 位置**都存(h_1..h_K,不只是 h_K) | `error_propagation_probe/probe_artifacts` 的训练数据源 |

磁盘占用:`pilot_multilayer` 8.6G,`error_propagation_probe/hidden_states_full` 3.8G,
`hidden_states/train` 2.3G,`hidden_states/dev` 255M。

## 第二层:各个打分器 artifact

| Artifact 目录 | 训练脚本 | 输入特征 | 结构 | 用的层 |
|---|---|---|---|---|
| `gate/artifacts_pooled/` | `gate/fit_lr_gate_pooled.py`(早期版本,已被 v2 取代) | Δ_j | PCA(64)+LR,per 池化 j | 单层(31,旧 hs 源) |
| `gate/artifacts_pooled_v2/` | 同上,数据源换成 `pilot_multilayer` | Δ_j | PCA(64)+LR,per 池化 j | j=0/1→15, j=2/3→23(已验证的最优选层) |
| `gate/gate_v3/artifacts_pooled_v3/` | `gate/gate_v3/fit_lr_gate_pooled_v3.py`(这个 session 新建) | **concat(PCA(h_after,64), PCA(Δ_j,64))** —— h 和 Δ **各自独立** PCA | 拼接后单层 LR,per 池化 j | 同 v2,j=0/1→15, j=2/3→23 |
| `error_propagation_probe/probe_artifacts/` | `error_propagation_probe/score_and_analyze.py --pool-positions` | h_after(累计前缀状态,不分 j,所有中间位置池化一起训) | PCA(64)+LR | 固定 31 层 |
| `rerank_pairwise_mlp/artifacts/` | `rerank_pairwise_mlp/train_pairwise_mlp.py` | [emb_score, PCA(Δ)] | PCA + 小 MLP,pairwise ranking loss | 沿用 `hidden_states/{train,dev}` 的单层(31) |
| `rerank_listwise_contrastive/`(无独立 artifacts 目录,权重存在训练脚本自己的输出里) | `train_listwise_ranker.py` | [emb_score, PCA(Δ)] | 同结构,listwise softmax loss(真实 top-k 候选池,不是人工采样 pair) | 31 |

**环境坑记录**:v3 第一次训练用了 conda `base` 环境(sklearn 1.8.0),而打分/wavefront 用的是
`mlsys`(sklearn 1.7.2)——1.8.0 之后 `LogisticRegression` 不再存 `multi_class` 属性,1.7.2 的
`predict_proba` 还依赖它,直接 `AttributeError`。**所有 joblib 训练都必须在 `mlsys` 环境下跑**,
已经在 v3 上踩过一次,以后新增打分器训练要记得先 `conda activate mlsys`。

## 第三层 (a):候选池级缓存 —— 应用到真实检索候选池,供离线实验反复复用

| 目录/文件 | 候选池来源 | 已有的列 | 说明 |
|---|---|---|---|
| `rerank_listwise_contrastive/data/musique_{split}_pools.jsonl` | `build_topk_pools.py`,真实 BGE top-k 检索(retrieve-k=20) | 原始候选池(段落文本 + emb_score + is_gold) | **这是 `adaptive_lambda_rerank` 的候选池数据源**(`--pools` 默认指到这里) |
| `adaptive_lambda_rerank/data/musique_{split}_lambda_data.npz` | 同上 pools.jsonl(train 额外拼了 `musique_train_pools_enhance.jsonl`,K=3/4 数据增强) | `cand_emb_score`(cosine)、`cand_abnormal_score`(probe 打的 h 分)、`cand_gate_v2_score`(gate_v2 打的 Δ 分)、**`cand_h_pca`(这次新加,probe 层的原始 h_after PCA 向量,64 维,每候选一份)** | 目前离线组合实验(置信度加权/网格搜索/三方凸组合/context-gate 原型)全部基于这份数据,零 GPU 复用 |
| `rerank_pairwise_mlp/data/musique_{split}_pairwise.npz` | `hidden_states/{train,dev}` 里已有的 gold+counterfactual pair(不是真实检索池,是人工构造的一对一比较) | deltas, emb_scores, labels | Scheme A,跟 listwise/adaptive_lambda 的"真实候选池"性质不同,不要混着比 |

## 第三层 (b):端到端 wavefront 结果目录(检索+生成指标,不是特征缓存)

| 目录 | 对应方法 |
|---|---|
| `retrieval/results/` | production 方法(baseline/gated_rule_a/lr_rerank)+ 这个 session 新加的 `run_retrieval_exp_wavefront_gate_v3.py` 结果 |
| `combined_gate_rerank/results/` | gate_v2 + probe 联合打分(sum/veto/rrf 三种投票) |
| `error_propagation_probe/results/` | probe 单独 beam search(probe_only / weighted) |
| `rerank_pairwise_mlp/results/`、`rerank_listwise_contrastive/`(训练脚本内自带 eval,没有单独 results 目录) | Scheme A/B 两种独立重排方案 |
| `reader_ceiling_probe/results/` | oracle evidence 上限探测(reader 侧,不涉及 gate/rerank) |

这些都是 git-ignored 的运行产物(时间戳命名),不是需要"整理"的对象,只是列出来方便知道去哪找对比数据。

## 复用关系图(谁 import 谁 / 谁读谁的数据)

```
retrieval/run_retrieval_exp.py, run_retrieval_exp_wavefront.py   ← 生产代码,零改动,被下面所有脚本 import
  ├─ gate/fit_lr_gate_pooled.py            读 hidden_states/pilot_multilayer  → gate/artifacts_pooled_v2
  ├─ gate/gate_v3/fit_lr_gate_pooled_v3.py 读 hidden_states/pilot_multilayer  → gate/gate_v3/artifacts_pooled_v3
  ├─ error_propagation_probe/score_and_analyze.py 读 hidden_states_full → probe_artifacts
  ├─ rerank_pairwise_mlp/build_training_data.py    读 hidden_states/{train,dev}(不重新提取)
  ├─ rerank_listwise_contrastive/build_topk_pools.py  真实 BGE 检索 → musique_{split}_pools.jsonl
  │     └─ adaptive_lambda_rerank/build_training_data.py 读这份 pools.jsonl,过 Llama → lambda_data.npz
  │           ├─ add_gate_v2_scores.py   补 cand_gate_v2_score 列(复用 gate/artifacts_pooled_v2)
  │           └─ add_probe_h_vector.py   补 cand_h_pca 列(复用 probe_artifacts 的 PCA,这次新加)
  ├─ combined_gate_rerank/wavefront_combined_gate.py   同时加载 gate_v2 + probe,端到端跑
  └─ retrieval/run_retrieval_exp_wavefront_gate_v3.py  加载 gate_v3,端到端跑
```

## 看下来比较明显的重复/可以合并的地方(先记录,不动手)

1. **`hidden_states/{train,dev}` vs `hidden_states/pilot_multilayer`**:前者是单层旧版,只有
   `rerank_pairwise_mlp` 还在用;`gate/artifacts_pooled`(非 v2)现在应该也没人在用了。如果确认
   `fit_lr_gate.py`(非 pooled)和 `gate/artifacts_pooled`(非 v2)已经彻底废弃,这两份加起来
   ~2.5G 的旧数据可以考虑清掉,但需要先确认没有脚本还依赖它们(`rerank_pairwise_mlp` 依赖
   `hidden_states/{train,dev}` 本身,不能删)。
2. **候选池来源目前只有一份**(`rerank_listwise_contrastive/data/musique_{split}_pools.jsonl`),
   `adaptive_lambda_rerank`、`rerank_listwise_contrastive` 自己的 `train_listwise_ranker.py`
   共用同一份池子,这个已经是"单一数据源、多处复用"的好状态,不用动。
3. **`error_propagation_probe/hidden_states_full` 和 `hidden_states/pilot_multilayer` 的 trace
   来源不是同一批**(前者是自己独立构造的 `_multi_error.jsonl`,后者是 `traces/merged`)——
   v3 想要"h_after 和 Δ 用同一份数据联合训练"之所以选择基于 `pilot_multilayer`(而不是复用
   probe 自己的 `hidden_states_full`),就是因为 `pilot_multilayer` 本来就有 Δ 需要的相邻两跳,
   `hidden_states_full` 目前没有对应的 gate_v2 训练标签体系,两者暂时没法直接合并成一份。

## 目前还缺的一块

`combined_gate_rerank/wavefront_combined_gate.py` 和 `run_retrieval_exp_wavefront_gate_v3.py`
的 hidden state 全部是**跑端到端时现算现扔**,没有落盘缓存——如果以后还要反复对比这两条线的候选级
分数(不只是最终的 retrieval_exp 指标),会需要专门再加一次落盘,目前没有。

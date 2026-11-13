# 多跳检索 gate：工作汇总（毕业论文素材）

这份文档把 `multihop_trace` 仓库里的工作整理成三部分，供写论文时查阅：

- **第一部分 仓库与数据资产地图**：每个目录是什么、产出什么、数据在哪、哪些在 git 里、哪些要重算。
- **第二部分 方法演进时间线**：方法怎么一步步从 gate v1 走到现在，每一步的动机、做法、结果、决策；
  以及所有被证伪或未采纳的旁支探索（每个一小节）。
- **第三部分 结果汇总**：所有实验的完整指标表，每张表标注 pipeline 配置和对应的 `results/` 目录名。

**阅读时最重要的一条**：**跨周期的数字不能直接比**。这条 pipeline 在 7-9 月间有多次口径变更
（selection 步骤移除、rawprefix、BGE 检索前缀、reader prompt、λ 默认值、真实 recall@3 的计算方式），
每一次都会整体平移所有数字。可比性规则集中写在 [2.20 节](#220-口径变更史与可比性规则)。

本文档不涉及 `new_data_recipe/`、`gate_mtl_shared_backbone/` 两个目录，它们属于另一条工作线。

---

# 第一部分 仓库与数据资产地图

## 1.1 五阶段 pipeline

| # | 阶段 | 目录 | 关键脚本 | 输入 → 输出 |
|---|---|---|---|---|
| 0 | 原始数据 | `data/raw/` | — | MuSiQue / 2WikiMultihopQA / HotpotQA 官方 json/jsonl |
| 1 | 问题分解 | `decompose/gt/`、`decompose/bart/` | `gt_decompose_to_nl.py`、`train.py`、`predict.py`、`decompose_to_nl.py` | 主问题 → 自然语言子问题序列 |
| 2 | trace 构造 | `traces/` | `construct_balanced_traces.py` | GT 分解 + gold/反事实证据 → trace jsonl（带标签） |
| 3 | 隐状态提取 | `hidden_states/` | `extract_hidden_states.py`、`extract_hidden_states_multilayer_pilot.py` | trace → 逐跳累计前缀的 Llama 隐状态 `.npz` |
| 4 | gate 训练 | `gate/` | `fit_lr_gate_pooled.py`、`gate_v3/fit_lr_gate_pooled_v3.py` | 隐状态 → PCA + LogisticRegression artifact |
| 5 | 端到端检索评测 | `retrieval/` | `run_retrieval_exp_wavefront_gate_v3_rawprefix.py` | BART 分解 + gate 打分 → 检索/答案指标 |

**两套分解器不能混用**（`CLAUDE.md` 也强调了这一点）：

- **GT**（`decompose/gt/`）：把 MuSiQue 自带的 `question_decomposition` 字段用 GPT 改写成自然语言。
  仅 MuSiQue 有。用于**构造训练 trace**（阶段 2）。
- **BART**（`decompose/bart/`）：自己训练的 BART-large 分解器，三个数据集都能跑。用于**阶段 5 的端到端
  评测**，因为推理时没有 ground-truth 分解可用。

两者都汇入 `data/decompose/{dataset}/{gt,bart}/*.jsonl`，这是下游唯一读取的路径。

## 1.2 目录职责

### 主线目录

| 目录 | 作用 | 关键产出 | 相关周报 |
|---|---|---|---|
| `decompose/bart/` | BART-large 分解器训练与推理 | `outputs/bart_decomposer_musique_2wiki_repro/`（线上模型，checkpoint-1425）、`outputs/bart_decomposer_old/`（只用 MuSiQue 训练的对照模型，checkpoint-936） | 0914 |
| `decompose/gt/` | MuSiQue 自带分解 → 自然语言 | `data/decompose/musique/gt/*.jsonl` | — |
| `traces/` | 确定性构造 gold / 反事实 trace 与标签 | `traces/merged/musique/{train,dev}.jsonl`（83,927 / 8,821 条） | — |
| `hidden_states/` | 逐跳累计前缀的隐状态提取 | `train/`、`dev/`（单层 31）、`pilot_multilayer/`（7/15/23/31 四层，92,748 个 npz） | 0705 |
| `gate/` | gate 各版本的训练与离线评测 | 见 [1.4 节](#14-gate-artifact-版本清单) | 0705、0719、0720、0727、0810 |
| `retrieval/` | 端到端评测主战场 | `results/` 下 147 个 run 目录 | 几乎每期 |

`retrieval/` 里脚本的演进关系（后者都大量 import 复用前者，生产脚本本身没有被改坏）：

```
run_retrieval_exp.py                              数据加载 / 检索 / artifact 加载等公用函数
  └─ run_retrieval_exp_wavefront.py               批量 vLLM 版主流程（gate v1/v2 时代，含 selection 步骤）
       ├─ run_retrieval_exp_wavefront_gate_v3.py          gate v3 双 PCA 打分，去掉 selection
       │    └─ ..._gate_v3_rawprefix.py                   当前主力：gate 打分用未展开子问题的前缀
       │         （后续又加了 --retriever colbert、--retrieval-scope global、--final-reader-prompt 等开关）
       ├─ ..._gate_v3_decoupled.py                        探索：检索也不展开占位符（负面结果）
       ├─ ..._gate_v2_rawprefix.py / ..._gate_h_only_rawprefix.py   对照组 B / C
       └─ replay_final_reader.py                          只重跑最终答案这一步，用于 prompt A/B
```

`retrieval/` 下几个工具脚本：`build_global_corpus.py`（global 语料库构建）、
`compute_evidence_precision.py`（三方 precision 对比）、`analyze_global_gold_rank.py`（gold 在 BGE
排序中的位置分析）、`compare_decomposer_ablation.py`（分解器消融对比）、`run_d_rankedpool_all.sh`。

### 旁支探索目录

| 目录 | 想解决什么 | 结论 | 小节 |
|---|---|---|---|
| `error_propagation_probe/` | 错误能不能在后续跳的隐状态里留下痕迹；beam search 前置验证 | 痕迹确实存在（Spearman 0.63）；beam search 与贪心持平 | [2.B.1](#2b1-beam-search累计异常分剪枝)、[2.B.5](#2b5-h_j-probe-单独扛排序) |
| `reader_ceiling_probe/` | 答案指标不涨是不是 reader 的天花板 | 否证：换 gold 证据 EM 涨 6–8 点 | [2.4](#24-否证reader-是天花板这个假设0713-第五节) |
| `rerank_pairwise_mlp/` | 用学出来的模型替代固定加权公式（pairwise） | 离线 92% 配对准确率，端到端 recall@1 倒退 5 点 | [2.B.3](#2b3-pairwise-mlp-重排方案-a) |
| `rerank_listwise_contrastive/` | 同上，改成真实 top-k 候选池 + listwise | 候选池基础设施被后续大量复用，端到端对比未跑 | [2.B.4](#2b4-listwise-对比学习重排方案-b) |
| `adaptive_lambda_rerank/` | 让 λ 随 hop 自适应 | 三次失败，方向被证伪；但产出的候选池缓存成为后续离线实验的基础 | [2.B.6](#2b6-自适应-λ三次失败) |
| `combined_gate_rerank/` | 事后组合 Δ 与 h 两个分数 | `sum` 最好但不如特征级融合 | [2.B.7](#2b7-后融合的各种投票与权重方案) |
| `context_gate_rerank/` | 逐候选学习组合权重 | 离线双赢，端到端与 v3 打平，因结构复杂未采纳 | [2.B.8](#2b8-context-gate逐候选学习权重) |
| `online_listwise_gate/` | DAgger 式在线增量训练 + listwise | 公平对比下全指标落后 v3 | [2.B.11](#2b11-在线增量-listwise-gate) |
| `steering/` | 用 gate 的权重方向干预生成 | 只完成调研，未实现 | [2.B.12](#2b12-steering仅调研) |
| `chainrag/`、`grithopper/` | 复现两个同类工作做对比 | 三方对比完成 | [2.13](#213-chainraggrithopper-复现对比0831-第二部分) |

### 其他

- `config/default.yaml`：集中记录路径与超参（模型名、层、PCA 维度、target FPR、λ 等）。注意里面
  `gate.artifacts_dir` 和 `hidden_states.layer: 31` 仍是 v1 时代的值，实际主力是
  `gate/gate_v3/artifacts_pooled_v3` 与 per-j 分层（15/15/23/23）。
- `update_doc/`：14 期周报 + 4 份专题文档（见 [1.6 节](#16-文档资产)）。
- `CLAUDE.md`：给 AI 助手的仓库说明。其中「The gate operates on Δ features」是 v2 时代的描述，
  v3 的特征是 `concat(PCA(h_after), PCA(Δ))`，需要更正。

## 1.3 数据资产清单

| 资产 | 位置 | 规模 | 在 git | 怎么重算 |
|---|---|---|---|---|
| 三数据集原始数据 | `data/raw/` | 1.6 GB | 否 | 官方下载 |
| GT / BART 分解结果 | `data/decompose/` | 17 MB | 是 | `decompose/` 下重跑 |
| MuSiQue GT 自然语言分解 | `decompose/gt/musique_gt_nl_{train,dev}.jsonl`、`*_exact_gt_traces.jsonl` | — | 是 | `gt_decompose_to_nl.py`（需 OpenAI API）+ `build_exact_gt_trace.py` |
| 2Wiki GPT 分解标注 | `decompose/bart/data/2wiki_gpt_mixed_{train,dev}.jsonl` | train 10,444 / dev 1,043 条 | 是 | `run_annotate_2wiki.sh`（需 OpenAI API） |
| K=3/4 改写增强 | `data/decompose/musique/gt/train_nl_enhance.jsonl` | 5,562 条候选，4,132 条通过校验 | 是 | `traces/enhance_decompose_nl.py`（两遍 vLLM） |
| trace（gold + 反事实 + merged） | `traces/` | 643 MB | 部分 | `construct_balanced_traces.py` |
| 隐状态 `.npz` | `hidden_states/` | 12 GB，92,748 + 92,748 个文件 | 否（`.gitignore` 排除 `**/*.npz`，但 `manifest.jsonl`、`run_meta.json` 在） | `extract_hidden_states*.py`，需要 GPU + 本地 Llama checkpoint |
| gate artifact | `gate/*/artifacts_*` | 每个几 MB | 是 | `fit_lr_gate_pooled*.py`，纯 CPU，几分钟 |
| BART 分解器 checkpoint | `decompose/bart/outputs/*/checkpoint-*/` | 每个 ~1.6 GB | 否 | `run_train.sh`，3 epoch |
| 端到端 run 结果 | `retrieval/results/`（147 个目录） | — | 是（每个目录含 `retrieval_exp_*.json` 指标 + `retrieval_cases_*.jsonl` 逐题记录） | 重跑对应命令 |
| global 共享语料库 | `retrieval/global_corpus/{musique,2wiki,hotpot}_dev/` | MuSiQue 21,100 / 2Wiki 56,687 / HotpotQA 66,635 篇段落 + BGE embedding | 否 | `build_global_corpus.py`，约 1–3 分钟 |
| 多错 trace 与隐状态 | `error_propagation_probe/data/`、`hidden_states_full*/` | dev 16,316 条 trace / 51,672 跳 | 部分 | `build_multi_error_traces.py` + `extract_full_hidden_states_multilayer.py` |

**训练 / 评测数据规模**：trace 训练集 83,927 条（MuSiQue train），dev 8,821 条。
端到端评测固定用 dev 全量：MuSiQue 2,417 / 2Wiki 12,576 / HotpotQA 7,405 题。

## 1.4 gate artifact 版本清单

全部是「每个池化转移位置 j 一个独立模型」（j=0..3，对应 Q→E1 … E3→E4，跨 K 共享），
阈值按训练集 target FPR=0.15 校准。离线指标是 MuSiQue dev（8,821 条 trace）上的二分类指标。

**当前使用的三个变体**（统一第 15 层，见 [2.2 节](#22-层选择中间层优于最后一层0705)的层选择）：

| 变体 | 目录 | 特征 | dev AUC | TPR | FPR | Precision | F1 | 用途 |
|---|---|---|---|---|---|---|---|---|
| **v3（主方法）** | `gate/layer15_uniform/v3_artifacts/` | concat(PCA(h_after), PCA(Δ_j))，两路独立 PCA | **0.9357** | 0.8840 | 0.1821 | 0.7154 | 0.7908 | D 组 |
| 仅 Δ | `gate/layer15_uniform/v2_artifacts/` | Δ_j | 0.9203 | 0.8423 | 0.1712 | 0.7181 | 0.7753 | B 组对照 |
| 仅 h | `gate/layer15_uniform/h_only_artifacts/` | PCA(h_after) | 0.9345 | 0.8768 | 0.1747 | 0.7222 | 0.7920 | C 组对照 |

三者的层策略、PCA 维度（64）、C（1.0）、阈值校准方式（train 上 target FPR=0.15）完全一致，唯一变量是
输入特征——这是三信号对照成立的前提。**仅用 h 已经非常接近融合**（AUC 0.9345 vs 0.9357，F1 几乎相同），
**仅用 Δ 明显更弱**（0.9203）。

v3 按 j 的 dev 表现：j=0 TPR 0.9338 / FPR 0.1727、j=1 0.8730 / 0.1778、j=2 0.8034 / 0.2076、
j=3 0.8840 / 0.2741。

**为什么 dev 上的 FPR 普遍高于校准目标 0.15**（整体 0.1821）：阈值是在 **train 上**按 target FPR=0.15
取的（每个 j 一个阈值，见 `meta.json` 的 `threshold` 字段），dev 从未参与阈值选择，所以 dev 上的实测
FPR 是「阈值迁移到未见数据上」的结果，偏高属于预期。**j=3 偏离最多（0.2741）**，因为它在 dev 上只有
405 个正例 / 405 个负例，train 上也只有 4,046 条样本，是四个位置里最少的，阈值估计的方差最大。
论文第 4 章报 dev 指标时应当说明这一点，不要写成「校准失败」。

v3 的四个 j 的训练规模与训练集 AUC：j=0 n=83,927（正例 28.7%，AUC 0.9707）、j=1 n=59,857（40.2%，
0.9472）、j=2 n=21,411（45.3%，0.9171）、j=3 n=4,046（50.0%，0.9633）。**j=3 只有 4,046 条样本**，
是后段跳偏弱的直接原因之一。

**早期与旁支的 artifact**（保留以备查，不参与当前对照）：

| 版本 | 目录 | 特征 | 层 | dev AUC | TPR | FPR | F1 | 端到端评过 |
|---|---|---|---|---|---|---|---|---|
| v1（最初生产） | `gate/artifacts_pooled/` | Δ_j | 全部 31 | 0.9074 | 0.8235 | 0.1756 | 0.7616 | 是 |
| v2（per-j 混层） | `gate/artifacts_pooled_v2/` | Δ_j | 15/15/23/23 | 0.9221 | 0.8456 | 0.1693 | 0.7785 | 是 |
| h-only（per-j 混层） | `gate/gate_h_only/artifacts_pooled_h_only/` | h_after | 15/15/23/23 | 0.9339 | 0.8760 | 0.1714 | 0.7939 | 是 |
| v3（per-j 混层） | `gate/gate_v3/artifacts_pooled_v3/` | h ⊕ Δ | 15/15/23/23 | 0.9357 | 0.8838 | 0.1788 | 0.7930 | 是 |
| v4（双线性） | `gate/gate_v4/artifacts_pooled_v4/` | `(W_pre h_{j-1})·(W_post h_j) + w_Δ^T Δ_j` 低秩双线性 | 15/15/23/23 | 0.9457 | 0.9521 | 0.2847 | 0.7611 | **否** |

v4 离线 AUC 最高但 FPR 跑到 0.2847，从未接端到端。per-j 混层与统一第 15 层的 v3 在 dev 上
**AUC 逐位相同（0.9357）**，四项指标差异都在小数第三位。

## 1.5 关键配置与环境

- **LLM**：`meta-llama/Llama-3.1-8B-Instruct`，bfloat16。同一个模型承担三件事：gate 打分的隐状态提取
  （HF transformers）、逐跳短答案生成、最终答案生成（后两者走 vLLM）。
- **检索器**：`BAAI/bge-base-en-v1.5`，查询侧加 BGE 官方指令前缀
  `"Represent this sentence for searching relevant passages: "`，段落侧不加。对照实验用过
  `colbert-ir/colbertv2.0`。
- **conda 环境**：vLLM 端到端评测在 `mlsys`（vLLM 0.19.0、scikit-learn 1.7.2，与 gate artifact 的
  pickle 版本一致）。`base` 没装 vLLM，且 sklearn 1.8.0 会在加载 artifact 时报版本警告。
  ChainRAG / GRITHopper 各有自己的环境。
- **当前主力命令形态**：

```bash
conda activate mlsys
cd retrieval
python run_retrieval_exp_wavefront_gate_v3_rawprefix.py \
  --dataset musique --decompose-mode bart_decompose \
  --decompose-file ../data/decompose/musique/bart/dev_nl.jsonl \
  --artifacts-dir ../gate/gate_v3/artifacts_pooled_v3 \
  --lambda-gate 0.60 --beam-width 1 --retrieve-k 10 --retriever cosine \
  --final-reader-prompt comparison_hint --short-answer-prompt default \
  --limit 0 --gpu-memory-utilization 0.3 --max-model-len 8192 --gate-device cuda:0 \
  --run-tag <tag>
```

（`--limit` 默认是 20，跑全量必须显式写 `--limit 0`。）

## 1.6 文档资产

| 文档 | 内容 |
|---|---|
| `update_doc/0706/0705update.md` … `0928/0928update.md` | 14 期周报，按时间顺序，是所有数字的一手来源 |
| `update_doc/gate_v3_main_method_report.md`（583 行） | 方法报告：背景、训练数据构造、隐状态提取、特征定义、gate 模型、实验。**写论文方法章时的主要基础** |
| `update_doc/experiment_results_summary.md` | 早期（7 月）的结果汇总表，口径停留在 selection 时代 |
| `update_doc/hidden_state_inventory.md` | 隐状态提取/打分的三层结构清单，含复用关系图 |
| `update_doc/steering_research_report.md` | steering 方案调研报告 |

部分周报有英文版（`*_en.md`）和 PDF。

---

# 第二部分 方法演进时间线

## 2.A 主线

### 2.0 上游：分解器、trace 构造、隐状态提取

gate 之前有三层上游：把问题拆成子问题（分解器）、把子问题和证据拼成带标签的 trace（训练数据）、
从 trace 抽隐状态（特征）。三层的设计决定了 gate 能学到什么，也决定了它的上限。

#### 2.0.1 两套分解器与格式流水线

- **GT 分解**（`decompose/gt/`）：MuSiQue 原始数据自带 `question_decomposition` 和 gold 段落编号，
  用 GPT 把结构化分解改写成自然语言子问题（`gt_decompose_to_nl.py`），再拼上 gold 证据生成
  exact-GT trace（`build_exact_gt_trace.py`）。**仅 MuSiQue 有**，用途是构造 gate 的训练 trace。
- **BART 分解**（`decompose/bart/`）：自训 BART-large，三个数据集都能跑，用于端到端评测
  （推理时没有 ground-truth 分解）。

BART 的输出要经过三步才能给下游用，对应三个脚本：

```
predict.py                  → raw_predictions/    MuSiQue 官方 [[CQS]]/[[CQE]] hop 格式，num_beams=10
convert_predictions_to_v1.py → v1_predictions/    转成带 [Answer N] 占位符的 hop 文本，并把 id 从
                                                  double__/triple_ii__ 翻译成 2hop__/3hop1__ 等官方写法
decompose_to_nl.py           → nl_predictions/    用 Llama 改写成自然语言子问题，带一遍校验（--no-verify 可关），
                                                  有 checkpoint 断点续跑；产物同步到 data/decompose/{ds}/bart/dev_nl.jsonl
```

#### 2.0.2 训练配置与数据构成

超参（与 `checkpoint-1425` 的 `train_meta.json` 一致）：BART-large、lr 3e-5、batch 16、3 epoch、bf16、
`max_source_length/max_target_length=100`、label smoothing 0.1、eval beams 10、早停 patience 10、
按 BLEU 选最优 checkpoint。训练环境锁 `transformers==4.57.3`。

训练数据 30,382 条 = MuSiQue 官方分解 19,938 条 + **2Wiki GPT 标注 10,444 条（占 34%）**。
2Wiki 标注是**按题型定额采样**的（`TYPE_SAMPLE_PRESETS`），用 gpt-5-mini 按 MuSiQue 的分解格式标注、
`hop_source=support_titles`（一跳对应取回一篇 gold 段落，而不是一条 KB 三元组）：从 167,454 条里选
10,500 条，成功 10,444 条。**采样刻意偏向比较类题**：比较类在原始训练集占 52%、在标注数据里占 71%，
其中需要 4 跳的 bridge_comparison 从 20.7% 提到 47.9%，而与 MuSiQue 结构相近的 compositional 只采了
原始条数的 2%。另有 1,043 条 2Wiki dev 标注，只和 MuSiQue dev 一起用于挑 checkpoint。

**HotpotQA 的标注准备好了但从未使用**：`annotate_hotpot_gpt_decompose.py`、`run_annotate_hotpot.sh`
都在，`run_train.sh` 也有 `INCLUDE_HOTPOT=1` 开关，但 `data/hotpot_gpt_mixed_train.jsonl` 不存在——
这条路径从未产出数据。HotpotQA 全程是**零训练数据的跨数据集泛化测试**。

#### 2.0.3 K=3/4 改写增强（query rewrite）

MuSiQue 的 K 分布极不均衡（train GT 分解里 2-hop 占 72%：14,376 / 4,387 / 1,175），长链样本稀缺直接
造成 gate 的 j=3 只有 4,046 条训练样本。`traces/enhance_decompose_nl.py` 用两遍 vLLM 做增强：
先逐跳改写（每次一句，保留 `[Answer N]` 占位符），再整链校验（通顺度 / 链条逻辑 / 可检索性）。
产出 5,562 条候选，**校验通过 4,132 条**，并入 trace 构造后 K 分布变成 14,376 / 7,671 / 2,023，
3/4 跳样本量接近翻倍。

这批数据后来被反复复用：`adaptive_lambda_rerank` 用它重新检索造出额外的真实候选池（把候选池 K 分布
从 61.7/28.2/10.1 改善到 48.0/38.5/13.5），`online_listwise_gate` 的全量训练也混入了它。
**注意它是「换一个措辞重新检索一遍」，不是复制数据**，候选池天然不同。

#### 2.0.4 分解器质量：长链是瓶颈

完整指标见 [3.12 节](#312-分解器质量指标)。关键事实：

- MuSiQue dev 上 BLEU 76.81、整体跳数准确率 0.8366，但**按 gold 跳数拆开是 2 跳 0.982 / 3 跳 0.782 /
  4 跳 0.489**——4 跳题有一半被压短。早期报告就据此判断「后续提升不仅要改 gate，也要继续改长链分解」。
- 跳数匹配率（`k_match`，BART 预测跳数与 gold 一致的比例）按数据集差别很大：2Wiki 98.66% >
  HotpotQA 87.63% > MuSiQue 83.82%。这是 2Wiki 端到端指标整体最高的原因之一。
- **GT 与 BART 的差距是一个明确的上界参考**：同一套方法换成 GT 分解，MuSiQue 上 recall@1 能到
  0.78–0.84、EM 到 0.46–0.48（0808 的 λ 扫描用的是 GT 分解的 600 条 train 子集；
  `online_listwise_gate` 的冻结模型评测也测到 GT 0.7823 / BART 0.6709）。**跨 GT 和 BART 的数字
  绝对不能直接比**——`online_listwise_gate` 那次就差点因此误判为「新方法大幅领先」，见
  [2.B.11](#2b11-在线增量-listwise-gate)。

#### 2.0.5 2Wiki 标注数据到底贡献了多少

这是 0914 做的消融，见 [2.15 节](#215-分解器训练数据消融0914-第一部分)与
[3.8 节](#38-分解器训练数据消融)。一句话结论：**它教会的是「拆比较类问题」和「拆长链」**——去掉之后
只用 MuSiQue 训练的模型在 2Wiki 上一个 4 跳分解都产生不出来，2Wiki EM 掉 8.3 点、chain@1 掉 30.3 点。

#### 2.0.6 trace 构造协议：不依赖 LLM 自然犯错

**核心设计决策**：gate 的训练标签不靠「让 LLM 自己犯错再标注」，而是**确定性地注入错误**——自然错误
不可控、标签不稳定，而注入错误可以精确知道哪一跳、哪一个 transition 应该触发干预。
`traces/construct_balanced_traces.py` 产出两类 trace：

| 类型 | 证据设置 | 标签含义 |
|---|---|---|
| **gold**（`trace_type=correct`） | 每一跳都用 gold 证据 | 所有 transition 都不需要干预 |
| **counterfactual**（`trace_type=error`） | 指定一跳换成 cosine top-k 里的非 gold 段落，其余跳保持 gold | 该 transition 标为需要干预 |

几个细节决定了数据质量：

- 负例的错误段落取自 **BGE cosine top-k 的非 gold 段落**，不是随机采样——这样负例是「看起来很像但
  不对」的困难样本，与真实检索失败的分布接近。
- 检索 query 里的 `[Answer N]` 用**前面跳的 gold 答案**展开，与线上使用方式一致。
- 标签定义：`should_intervene(j) = (j + 1) in wrong_hops`，且**只在 `trace_type=error` 上生效**，
  gold trace 按构造全部是负类。
- **单跳注入**：`wrong_hops` 长度 100% 为 1。这个限制直到 0928 才被专门构造的多错数据突破，
  见 [2.19 节](#219-gate-能力剖析它到底在判断什么0928)。
- trace 文本的拼装统一走 `traces/trace_format.py::assemble_trace()`
  （`Question: … Step i: … Evidence: "…" Final Answer: …`），不在别处重复实现。

**规模与平衡**（MuSiQue）：

| split | 正例 | 未平衡负例 | 平衡后负例 | merged 总数 |
|---|---|---|---|---|
| train | 24,070 | 179,571 | 59,857 | **83,927** |
| dev | 2,417 | 19,212 | 6,404 | **8,821** |

负例按 `(K, hop)` 分桶、按 `pos_count[K] × balance_ratio` 采样，保证每个 hop 位置都有对应的错误
transition 可学。train 正例的 K 分布是 14,376 / 7,671 / 2,023（已含 [2.0.3](#203-k34-改写增强query-rewrite)
的增强数据）。**阶段 3 的隐状态提取只读 `merged/`**，不读 `gold/` 或 `counterfactual/`。

#### 2.0.7 隐状态提取协议

对每条 trace，用 Llama-3.1-8B-Instruct 对**每一个累计前缀**做一次前向，取最后一个 token 的隐状态：

```
h_0: Question: …
h_1: Question: … Step 1: … Evidence: "…"
h_2: Question: … Step 1: … Evidence: "…" Step 2: … Evidence: "…"
…
```

转移 j 的特征由相邻两个前缀构成：`Δ_j = h_{j+1} − h_j`（代码里 `hidden[j+1] - hidden[j]`），
v3 还额外用 `h_{j+1}` 本身。**Δ 不存在 `.npz` 里，是训练时现算的**。

两个影响后续工作的设计细节：

1. **`n_take = wrong_hop + 1`**：counterfactual trace 的隐状态在注入错误的那一跳之后就不再抽取。
   这是出于节省存储的考虑，但直接导致 9.2 万个 npz 里**没有任何「错误发生之后」的隐状态**——
   0928 的能力剖析必须另建数据才能做，见 [2.19 节](#219-gate-能力剖析它到底在判断什么0928)。
2. **多层提取**：`extract_hidden_states_multilayer_pilot.py` 一次前向读出 7/15/23/31 四层
   （`output_hidden_states=True` 本来就会算出全部层，原来只读了第 31 层），这是 [2.2 节](#22-层选择中间层优于最后一层0705)
   层选择实验的前提。per-j 用哪一层是**artifact 自己携带的元数据**（每个 joblib 存了自己的 `layer`
   字段），检索脚本里没有写死「第几跳用第几层」。

工程上值得记一笔：多层提取脚本的内存控制有两处非平凡设计——`pad_to_multiple_of` 把每批的 padding
长度对齐到固定网格（否则每批形状都不同，PyTorch 缓存分配器会不断新开内存块，保留显存只增不减），
以及按 token 预算自适应 batch size + 写盘后立刻驱逐已完成 trace 的隐状态。

### 2.1 起点：gate v1 与它的两个问题

最初的生产 gate：Llama 第 31 层（最后一层）、特征 `Δ_j = h_j − h_{j−1}`、PCA(64) + LogisticRegression，
四个池化转移位置共享同一层。离线 AUC 0.9074。

两个已知问题：最后一层未必是最优选择；K=4（深层转移）判别力明显弱于 K=2。

### 2.2 层选择：中间层优于最后一层（0705）

**动机**：最初的生产 gate 固定用 Llama 的最后一层（31），四个转移位置共用。两个问题待验证：最后一层
是不是最优；能不能缓解 K=4（深层转移）判别力偏弱。

**做法**：新写多层提取脚本（一次 forward 同时读出 7/15/23/31 四层，`output_hidden_states=True` 本来
就会算出全部层，原来只读了第 31 层），然后比较四个候选层在每个转移位置上的判别力。

**选择协议**：候选层的比较在**训练集内部的留出集**上做——按 `source_id`（题目）把 MuSiQue train 的
83,927 条 trace 切成拟合集与选择集（同一道题的 gold trace 和它的全部反事实变体整体分到同一侧，避免
泄漏），在拟合集上训 PCA(64)+LR，在留出的 20% 题目上比 AUC。**dev 不参与选层**，它只用于最后汇报
gate 的离线指标。特征口径与生产一致（`concat(PCA(h_after), PCA(Δ_j))`），脚本
`gate/select_layers_train_holdout.py`，结果 `gate/pilot_layer_compare/layer_selection_train_holdout.json`。

**结果**（全量，留出集 AUC）：

| j | 转移 | 行数 | layer 7 | layer 15 | layer 23 | layer 31 | 最优 |
|---|---|---|---|---|---|---|---|
| 0 | Q→E1 | 83,927 | 0.8938 | **0.9686** | 0.9646 | 0.9581 | 15 |
| 1 | E1→E2 | 59,857 | 0.8441 | **0.9399** | 0.9356 | 0.9264 | 15 |
| 2 | E2→E3 | 21,411 | 0.7905 | **0.9180** | 0.9146 | 0.9072 | 15 |
| 3 | E3→E4 | 4,046 | 0.8404 | **0.9448** | 0.9280 | 0.9313 | 15 |

**发现**：
1. **中间层明显优于最后一层**：15 比 31 高 **1.1–1.3** 个 AUC 点（1.05 / 1.35 / 1.08 / 1.35），四个位置一致。
2. **第 7 层（浅层）全面垫底**（0.79–0.89），比第 15 层低 **7.5–12.8** 个点——说明这不是「层越靠前越好」，而是存在一个
   中段的甜蜜点。
3. **统一第 15 层在四个位置上都是最优**。23 层在后段位置（j=2/j=3）与 15 层接近（差 0.0034 / 0.0168），
   但没有反超；j=3 的差距偏大与它只有 4,046 行、是最小的桶有关。
4. 「维度多了就好」被排除：`concat(15+15)` 与单层 15 几乎相同（0.9152 vs 0.9151）。
5. 把 31 混进 concat 反而拖累（`concat(15+23+31)` 掉回 0.901）。推测最后一层被「预测下一个 token」
   的训练目标塑形过度，压缩掉了部分「这段是否自洽」的抽象语义。
6. 浅层 MLP 的 AUC 比 LR 更高，**但阈值校准跑偏**：target FPR=0.15 下实测 FPR=0.2348，所以保持 LR。

**决策**：统一使用第 15 层 + 保持 LR，不引入 MLP。层的选择写进 artifact 的元数据（每个 joblib 自带
`layer` 字段），检索脚本按元数据取层，没有写死「第几跳用第几层」。

**第三部分的所有端到端结果都已换成统一第 15 层的 artifact**（`gate/layer15_uniform/`，2026-10-08~10
重跑的 36 个 run）。per-j 混层的旧 artifact 只在 [1.4 节](#14-gate-artifact-版本清单)「早期与旁支」
里保留备查，两套数字不能混用（见 [2.20 节](#220-口径变更史与可比性规则)）。

### 2.3 gate v2 端到端：检索涨了，答案没涨（0713 第一节）

三个数据集 dev 全量，`gated_rule_a`（二元触发）模式下，recall@1 / chain@1 一致提升（MuSiQue
0.6513→0.6565，2Wiki 0.8484→0.8569，HotpotQA 0.6186→0.6257），但**答案 EM/F1 三个数据集都在
±0.004 内浮动**。

这个结果引出一个怀疑：是不是只要 reader 不变，答案层面的天花板就已经被 reader 摁住了？

### 2.4 否证「reader 是天花板」这个假设（0713 第五节）

**做法**：`reader_ceiling_probe/oracle_evidence_reader.py`，跳过检索与 gate，每一跳强制使用 gold
证据，reader 完全不变。设计上注意了一个坑：BART 分解的跳序与 gold 分解不保证一一对应，所以 BART
模式下不是按位置硬塞，而是把该题全部 gold 证据放进一个池子，让模型每跳自己选一个、选后移除。

**结果**（MuSiQue dev 全量，gold 覆盖率 99.33%）：

| 方法 | EM | F1 |
|---|---|---|
| gate v2 | 0.4042 | 0.4983 |
| oracle（gold 证据） | **0.4708** | **0.5754** |

**结论**：假设不成立，reader 有能力答对，瓶颈在证据选得准不准。这个否证**翻转了后续的优先级**，
证明继续投入检索侧是值得的。

### 2.5 两个信号：Δ（局部）与 h_after（全局）（0719、0720）

同一次前向传播可以提取两种性质不同的信号：

- **Δ_j = h_j − h_{j−1}**：这一步证据让模型内部状态变了多少（局部）。
- **h_after**：截至当前跳，整条推理前缀的状态（全局/累积）。

`error_propagation_probe` 在 h 上训的 probe 有一个关键性质：标签只是二分类「到这里为止有没有错」，
但 held-out 分数会随真实错误数单调上升，Spearman(n_wrong, score) 池化全部位置可达 0.63。

**互补性验证**（MuSiQue dev，按 id 对齐 2,417 题）：两项指标上都有 13–14% 的样例「只有一方做对」，
理想完美择优的 EM 上限 47.3%（单独使用各约 39–41%）。结论：两个信号不是同一件事的两种说法，
值得组合。

### 2.6 gate v3：特征层面融合（0720 第七、八节；0727）

前面所有组合尝试都是**后融合**（两个模型各自训好，推理时组合输出分数，见
[2.B.7](#2b7-后融合的各种投票与权重方案)）。v3 换成**特征层面融合**：`h_after` 与 `Δ_j` 各自独立做
PCA（协方差结构不同，不共用），拼接后只训**一个** LogisticRegression：

```
p_j = σ(w_h^T PCA(h_after) + w_Δ^T PCA(Δ_j) + b)
```

层选择直接复用 v2 的结论，训练数据复用已提取的隐状态，**不需要新的 GPU 提取**，纯 CPU 训练。

**离线**：AUC 0.9203→0.9357，TPR 0.8423→0.8840；j=2（v2 最弱的一跳）F1 涨幅最大（0.7417→0.7713）。

**端到端**（MuSiQue dev 全量，λ=0.25，同机制对比）：

| 方法 | recall@1 | chain@1 | EM | F1 |
|---|---|---|---|---|
| gate v2（Δ only） | 0.6627 | 0.4357 | 0.4100 | 0.5045 |
| gate v3 | **0.6950** | **0.5010** | **0.4224** | **0.5146** |

四项全赢，chain@1 涨幅最大（+6.5 点），与离线「最弱的 j=2 改善最多」一致：链路越长，单跳弱点被
放大得越厉害，修好最弱一跳对整条链的杠杆最大。

**这是第一次「离线赢」干净兑现成「端到端赢」**。因为生产打分入口按「单 PCA + 单 LR 作用在 Δ 上」
硬编码，v3 另起了 `run_retrieval_exp_wavefront_gate_v3.py`。

### 2.7 移除 selection 步骤，并发现一个训练/推理不一致（0727 第五节）

**selection 步骤**：v1/v2 时代每跳在排序后还要再调一次 LLM，从 top-3 里选一个。对比两个口径：

| | recall@1（排序阶段） | selection_acc（LLM 选完） |
|---|---|---|
| 整体 | 0.6565 | 0.6185 |
| K=2 / K=3 / K=4 | 0.7188 / 0.6212 / 0.5526 | 0.6736 / 0.5906 / 0.5179 |

**每个 K 桶，LLM 再选一次都比直接相信排序更差**。据此 v3 起去掉这一步，排序完直接取 rank-1。

**训练/推理不一致**：MuSiQue 子问题里 49.0% 的跳带 `[Answer N]` 占位符。gate 训练数据
（`traces/merged` 的 `reasoning_trace`）里占位符**从未展开**，而推理时子问题会先用前面跳的短答案
展开再喂给 gate 打分——约一半的跳存在分布不一致。修复见 [2.8 节](#28-rawprefix-与-bge-检索前缀0730)，
一次失败的修法见 [2.B.9](#2b9-decoupled检索也不展开占位符)。

### 2.8 rawprefix 与 BGE 检索前缀（0730）

**rawprefix**：每条候选路径维护两条并行记录——`hop_steps`（展开版，用于检索查询、短答案生成、最终
reader，三处都不变）与 `gate_hop_steps`（原始未展开版，**只**用来拼喂给 gate 打分的累计前缀）。
这样 gate 从第一跳到当前跳看到的整条前缀都与训练分布一致，而检索完全不受影响。

**BGE 检索前缀**：发现查询从一开始就没加 BGE 官方推荐的非对称检索指令前缀。新增 `--query-instruction`
参数（默认即官方那句），只作用于 cosine 分支。

**效果**（MuSiQue dev 全量，λ=0.50）：

| | recall@1 | chain@1 | EM | F1 |
|---|---|---|---|---|
| v3（λ=0.50） | 0.6984 | 0.5114 | 0.4249 | 0.5187 |
| + rawprefix | 0.7008 | 0.5143 | 0.4274 | 0.5216 |
| + BGE 前缀 | **0.7018** | 0.5143 | **0.4299** | **0.5256** |

两个改动方向一致、可叠加，但量级（累计 +0.3~0.7 点）比 λ 调参小一个数量级。

### 2.9 三信号对照：A/B/C/D 四组（0810）

统一配置（rawprefix、λ=0.50、BGE 前缀、真实 recall@3）下的单变量对照，三个数据集全量。完整表见
[3.1 节](#31-主结果三数据集三信号对照abcd)。核心结论：

1. **D（特征融合）在 recall@1、chain@1、EM、F1 四项上三个数据集全部领先**，recall@3 偶尔被 B/C
   反超 ≤0.002（噪声内）。
2. **C（只用 h）几乎全面超过 B（只用 Δ）**——与离线指标一致：h 单独已是比 Δ 更强的信号，
   融合相对 h-only 的边际提升小于 h-only 相对 Δ-only 的提升。这个发现在 0928 的拆解里得到了
   机制层面的解释（见 [2.19](#219-gate-能力剖析它到底在判断什么0928)）。

### 2.10 reader prompt 修复：comparison_hint（0817）

**发现方式是翻案例，不是看聚合指标**。MuSiQue 上 1,376 个答错案例中，485 个「每跳都选中 gold 但
答案仍错」。把这 485 个的最终答案与逐跳短答案逐一比对：77% 只是措辞与 gold 不完全一致（EM 严格性
导致），8% 是 reader 选错了该用哪一跳的结论，15% 是 reader 自己综合出一个新的错误答案。

在 2Wiki / HotpotQA 上重复分析，「reader 选错跳」的比例更高（13.1% / 15.2%），且集中在**比较类
问题**：两跳分别查出两个日期，最终该输出「哪一个」（实体名），reader 却直接把某个日期原样输出。

**根因**：最终 reader prompt 里唯一约束答案格式的那句把 date / number 列为合法答案类型，对所有
问题一视同仁。对桥接题「最后一跳短答案就是最终答案」成立，对比较题则不成立。

**修复**：新增 `build_final_reader_cot_prompt_comparison_hint`（保留原函数），只加一句话，明确告诉
reader 比较类问题的逐跳答案是用于比较的中间量，必须自己完成比较并回答实体名字。

**结果**（三个数据集，只切 prompt，检索与 gate 完全不动）。**注意这组数字的配置是 λ=0.5 + per-j
混层 artifact**（0817 当时的生产配置）：两臂之间 prompt 是唯一变量，所以 delta 有效，但**绝对值
不能与第三部分的表同列**。当前口径（λ=0 / λ=0.60 + 统一第 15 层）下的同一项测量见
[3.4 节](#34-reader-prompt-效果)，prompt 的贡献复现为 +0.99 / +9.13 / +2.84，与下表在 0.8 点以内一致。


| 数据集 | recall@1 | chain@1 | EM | F1 |
|---|---|---|---|---|
| MuSiQue | 持平 | 持平 | 0.4307→0.4427（+1.20） | 0.5270→0.5386（+1.16） |
| 2Wiki | 持平 | 持平 | 0.4990→**0.5982**（+9.92） | 0.5759→**0.6770**（+10.11） |
| HotpotQA | 持平 | 持平 | 0.5217→0.5517（+3.00） | 0.6517→0.6870（+3.53） |

recall@1 / chain@1 完全持平证明改动干净。**这是项目单次改动里最大的一次提升**，超过 λ 调参、
rawprefix、BGE 前缀三项之和。「reader 选错跳」的复现率从 13.1% 降到 2.6%。

涨幅排序（2Wiki > HotpotQA > MuSiQue）不等于 bug 占比排序，决定因素是**受影响样本占全量的比例**：
2Wiki 639/12576=5.1%，HotpotQA 232/7405=3.1%。

后续两版 prompt 变体均为净负，见 [2.B.10](#2b10-两版后续-reader-prompt-变体)。

### 2.11 ColBERT：验证 gate 的可迁移性（0824）

**动机**：gate 一直接在 BGE 后面调，收益是不是绑定在这一个检索器上？

**关键工程问题**：ColBERT 的 MaxSim 分数是逐 query token 取最大再加总，量级在个位数到二三十，且
query 越长分数天然越大，没有统一上下界。直接代入 `emb_score − λ·gate_score`（λ=0.5，gate 分最大
1.0），gate 会被完全架空。**修复**：`minmax_normalize_candidates()`，只在 colbert 分支对每跳自己的
候选池内部做 min-max 归一化。

**结果**（完整表见 [3.5 节](#35-colbert-检索器对照)）：ColBERT 加 gate 后 recall@1 涨 3–4 点、
chain@1 涨 6–8 点、EM 涨 4–5 点，**与 BGE 那边加 gate 的提升量级相当**；但 ColBERT 本身三个数据集
全面弱于 BGE（recall@1 差 4–6 点）。

**结论**：最有价值的发现是**gate 的增量价值不依赖具体检索器**，是一个可以接在任意打分范围检索器
后面的通用插件，只要做好分数归一化。不把 ColBERT 换成默认检索器。

### 2.12 λ 在训练集上重新调优（0831 第一部分）

**动机**：此前 λ 一直在 dev 上扫（0727 扫出 0.50），存在「调参与最终评测用同一份数据」的隐患。

**做法**：改到 MuSiQue **train** 的 3,000 条子集上扫 λ∈[0.10, 0.80]，步长 0.05，用完整当前方法
（rawprefix + comparison_hint）。特意确认过 BART 在这批 train 子集上的输出仍带真实生成噪声，
不是被记忆成接近 GT 的干净文本。

**结果**：EM/F1 在 0.60–0.65 见顶，0.70 之后四项一起回落，是真实的内部最优区间（完整曲线见
[3.3 节](#33-λ-扫描)）。

**dev 全量验证 0.50 → 0.60**：MuSiQue 四项小幅**下降**（EM −0.41），2Wiki、HotpotQA 全部小幅上升。
方向很关键——0.50 当初主要是针对 MuSiQue 调出来的，而 0.60 在两个完全没参与调参的数据集上都有提升，
说明它是跨数据集更稳健的取值。**采纳 λ=0.60 为新默认值。**

### 2.13 ChainRAG / GRITHopper 复现对比（0831 第二部分）

克隆并本地化两个思路接近的开源工作，统一口径重新跑：

- **ChainRAG**（ACL 2025）：零训练，拆子问题 + 句子图扩展兜底。
- **GRITHopper**（EACL 2026）：无需拆解的迭代稠密检索，GritLM-7B 底座 + 专门微调。

**统一口径的关键**：三方都用**顺序无关**的 recall@k（ChainRAG / GRITHopper 的跳与 gold 之间没有
位置对应关系），且三方最终答案生成都改成调同一个本地 Llama-3.1-8B-Instruct。

**结果**（完整表见 [3.6 节](#36-三方对比-vs-chainrag--grithopper)）：

- 检索命中率排名：GRITHopper > D 组 > ChainRAG。
- gold 覆盖率排名：ChainRAG > GRITHopper ≈ D 组（ChainRAG 靠广撒网，口径不对等）。
- **EM/F1 排名：D 组明显领先**，GRITHopper 与 ChainRAG 互有胜负、都明显落后。

最反直觉的两个模式：ChainRAG 覆盖率全场最高、EM/F1 全场最低；GRITHopper 召回全场最高、EM/F1 仍
明显落后 D 组。原因不同——ChainRAG 是图扩展把正确信号稀释了，GRITHopper 是没有拆解脚手架、答案
合成负担全部后移。D 组用远小于 GRITHopper 的训练投入（一个 PCA+LR 的小 gate vs 微调 7B）取得了
最好的答案效果。

### 2.14 global setting：把候选池换成全数据集共享语料库（0907 第二部分）

**动机**：此前所有端到端评测都是 distractor setting——每题只在自带的约 20 篇段落里检索，gold 大概率
就在里面，检索难度被人为降低。

**做法**：按 `(title, 正文)` 一起去重合并该数据集 dev split 全部题目的段落（不只按 title 去重，因为
同一标题在不同题目的 context 里摘出的文本可能不同），用 BGE 统一编码存成矩阵。评测时只编码 query
再做一次矩阵乘法。gold 编号按 title+正文重映射到全局 id，三个数据集全部映射成功、0 丢失。
检索之后的流程（重排、打分、生成）完全不变，代码改动只是新增
`--retrieval-scope {distractor,global}` 与 `--corpus-dir` 两个参数。

**结果**：三个数据集全线下降，chain@1 掉得最多（2Wiki −27.7 点）。**但 gate 在开放语料下仍然有效**
（recall@1 +4.2~+7.3、EM +2.8~+8.4，约为 distractor 下的一半），落差主要来自第一阶段召回而非重排。
完整的 {λ=0, λ=0.60} × {k=10, k=50} 2×2 见 [3.7 节](#37-global-setting)。

### 2.15 分解器训练数据消融（0914 第一部分）

**问题**：线上分解器用 MuSiQue 官方分解标注（19,938 条）+ 自己用 GPT 给 2Wiki 标注的分解（10,444 条）
混合训练。去掉后者会掉多少？

2Wiki 标注是**按题型定额采样**的，刻意偏向比较类题：比较类（bridge_comparison + comparison）在原始
训练集占 52%，在标注数据里占 71%，其中需要 4 跳的 bridge_comparison 从 20.7% 提到 47.9%。

**结果**：2Wiki EM −8.3 点、chain@1 −30.3 点；HotpotQA EM −1.0；MuSiQue 反而略升。按题型拆开后
三个数据集结论一致：**2Wiki 自标注数据的价值在于教会分解器拆比较类问题和长链问题**
（2Wiki bridge_comparison EM −20.1，comparison −12.6，compositional 仅 −0.8；HotpotQA comparison
−4.1，bridge −0.2；MuSiQue 4 跳题 −3.5、2 跳题 +1.8）。只用 MuSiQue 训练的模型**一个 4 跳分解都
产生不出来**（2Wiki 1.26 万题中 0 个）。完整表见 [3.8 节](#38-分解器训练数据消融)。

### 2.16 precision 口径（0914 第二部分）

0831 的三方对比只有 recall 类指标，没回答「检索到的东西里有多少是噪声」。补了两个口径：

- **证据集合 precision**：最终用于生成答案的证据集合里 gold 的比例。对 D 组和 GRITHopper 数值上
  几乎等于 recall@1（都是每跳只取 1 篇），真正新增的信息只在 ChainRAG 上。
- **逐跳 precision@3**：每跳前 3 名里有几篇 gold 再除以 3。与 recall@3（严格说是 hit@3，只看有没有
  至少一篇）的区别是「有几篇」。三方 recall@3 都挤在 0.96–0.99 分不开，precision@3 能分开。

**结果**：ChainRAG 的 context 里约 80% 是噪声（MuSiQue 平均每题 14.5 篇、gold 2.6 篇）。precision@3
排名三个数据集一致：GRITHopper > D 组 > ChainRAG。但这个指标部分衡量的是「查询有多聚焦」——
GRITHopper 每步用完整原问题查询，所有 gold 都会排得靠前；D 组每跳只针对这一跳要找的那一篇，
别的 gold 排不上来反而是查询精准的表现。完整表见 [3.9 节](#39-precision-对比)。

### 2.17 global 下扩大候选池：top10 → top50（0921）

**结果**：三个数据集五项指标全部提升（EM +0.6 到 +2.3），说明 λ=0.60 在 50 篇候选下仍然有效、
没有被更大的候选池带偏；代价是 2.4–4.4 倍耗时。

补上 λ=0 对照臂之后可以说得更强：**无 gate 时扩大候选池的收益精确为零**（MuSiQue / HotpotQA 的
recall@1 逐位相同——纯余弦排序下第 1 名不会因为池子变大而改变），**有 gate 时才涨 +1.0~+2.0 点**。
也就是说 gate 不只是受益于更大的候选池，它是**利用**更大候选池的必要条件。完整表见
[3.7 节](#37-global-setting)。

### 2.18 候选池分析：瓶颈在 BGE 召回（0921 第五节）

用每跳实际发出的查询，在整个语料库上算 gold 的 BGE 名次（纯 CPU/GPU 编码，不调 LLM）。
自检：由记录的候选池重算的 recall@1/@3 与报告逐位一致，「BGE 名次 ≤ k」与「gold 在记录的 top-k
候选池里」一致率 99.99%–100%。

| 数据集 | gold 在 BGE 前 10 | 11–50 | 51–100 | 100 名以外 |
|---|---|---|---|---|
| MuSiQue | 67.9% | 9.5% | 3.5% | 19.1% |
| 2Wiki | 85.3% | 1.3% | 0.5% | 12.9% |
| HotpotQA | 64.2% | 6.8% | 2.8% | 26.3% |

**结论**：2Wiki 的 gold 分布是两极的（85% 在前 10，13% 在 100 名以外，中间几乎没有），这解释了它
扩大候选池几乎无效。13%–29% 的跳 gold 在前 50 名以外，重排再怎么做也救不回来——**global 与
distractor 之间剩下的 8–12 点差距主要来自召回这一步，不是重排范围**。

### 2.19 gate 能力剖析：它到底在判断什么（0928）

**为什么需要新数据**：生产 trace 的构造协议是单跳注入（`wrong_hops` 长度 100% 为 1），且隐状态提取
在出错那一跳就截断（`n_take = wrong_hop + 1`），所以现有 9.2 万个 npz 里**没有任何「错误发生之后」
的隐状态**。为此在 MuSiQue dev 上构造 `n_wrong = 0..K` 的全套变体（16,316 条 trace / 51,672 跳，
K=3/4 枚举全部错误位置组合），并补抽第 15/23 层隐状态。

**三个核心发现**（完整表见 [3.10 节](#310-gate-能力剖析0928)）：

1. **误报率与设计值吻合**：全对前缀下的正确跳 17% 超阈值（训练时 target FPR=0.15），迁移到未见过的
   dev 数据上仍然成立。错误跳 78%–87% 超阈值。
2. **它是「前缀被污染了没有」的检测器，不是「哪一跳错了」的定位器**。错误之前的跳完全不受影响
   （0.125–0.164，与全对基线 0.169 持平），错误之后本身选对的跳被抬高到 2.2–2.6 倍
   （0.376–0.448），且基本不衰减。固定跳的位置做对照后效应仍然成立（K=4 第 4 跳：前缀全对 0.211
   vs 前缀有错 0.385）。
3. **fusion 两半里，判别力几乎全部来自 h**。logit 可精确拆为
   `截距 + w_h·PCA(h) + w_Δ·PCA(Δ)`，按 j 分开算对错分离度（Cohen's d）：h 为 1.51–2.20，
   Δ 为 0.01–0.65。这与 2.9 节「C 组几乎全面超过 B 组」在端到端层面的观察互相印证。准确表述是
   「在已有 h 的情况下 Δ 没有提供额外判别力」，不是 Δ 本身没信息——两半特征并不独立
   （`Δ_j = h_j − h_{j−1}` 与 `h_j` 共用一项）。

**由此统一解释了三个原本看起来独立的现象**：错误之后正确跳被抬高（h 必然的行为）、错误越多检出
强度不升反降（它不在定位哪一跳错）、单跳注入的训练数据也能学好（它学的是更容易的「前缀正常 vs
不正常」）。

**代价**：一旦某跳选错，后面本身正确的候选有 38%–45% 被判成异常（全对基线 17%）。生产是贪心逐跳
推进，这意味着错误会被放大，而且由于不能定位错误位置，也无法支撑「回退到出错那一跳重选」这类策略。
**这个副作用与它的主要能力是同一件事的两面。**

### 2.20 口径变更史与可比性规则

下面每一项都会整体平移所有数字，**跨越任一项的两个数字不能直接比**：

| 时间 | 变更 | 影响 |
|---|---|---|
| 0727 | 移除每跳的 LLM selection 步骤 | 评测主口径从 `selection_acc` 变成排序 rank-1 |
| 0727 | λ 默认 0.25 → 0.50 | 四项指标整体上移（当时测得 recall@1 +4.2、chain@1 +11.6） |
| 0730 | rawprefix（gate 打分文本改为未展开子问题） | +0.2~0.3 点 |
| 0730 | BGE 检索指令前缀 | +0.1~0.4 点 |
| 0730 | 新增 `full_pool_gold_rank_*`：在剪枝前的完整候选池上算真正的 recall@1/@3 | 此前 `--beam-width 1` 时 recall@1/@3/mrr 会退化成同一个数 |
| 0817 | reader prompt → `comparison_hint` | **只影响 EM/F1**：MuSiQue +1.2、2Wiki +9.9、HotpotQA +3.0；recall/chain 完全不动 |
| 0831 | λ 0.50 → 0.60，且改在 train 上调 | 小幅 |
| 0907 起 | 新增 global setting | 与 distractor 的数字属于不同实验条件 |
| 1008 | gate 层策略 per-j 混层（15/15/23/23）→ **统一第 15 层**，层在训练集内部留出集上选 | 离线 AUC 逐位相同（0.9357），端到端在重跑噪声内（见 [3.3.3 节](#333-λ060-的-dev-全量验证最终主结果)）；但**两套 artifact 的数字不可混用** |
| 1008 | B/C 两组脚本补上 `--final-reader-prompt`（此前硬编码 default） | 旧 A/B/C/D 表的 **EM/F1 列跨组不可比**，检索指标不受影响（见 [3.1 节](#31-主结果三数据集三信号对照abcd)） |

**两条具体的坑**：

1. **0713 与 0719 对同一个数字的标注互相矛盾**：MuSiQue `recall@1=0.6513` 在 0713 里标为
   「baseline `gated_rule_a`（31 层 gate）」，在 0719 里标为「baseline（不使用 gate，纯余弦排序）」。
   两者不可能都对。**引用「无 gate 基线」时应该用 0810 的 A 组**
   （`20260806_090938_baseline_only_musique_hotpot`、`20260806_091034_baseline_only_2wiki`，
   MuSiQue recall@1=0.6003），那是在统一 rawprefix + BGE 前缀口径下、真正 `--lambda-gate 0`/
   `baseline` 方法跑出来的。
2. **A 组用的是旧 reader prompt（default），D 组最终版用的是 comparison_hint**。所以 0831 第 1.5 节
   「加 gate vs 不加 gate」的 EM/F1 差值是「gate + reader prompt 修复」的合计。想要 gate 单独的
   贡献，应该用同为 default prompt 的 A 组与 D 组（λ=0.50）对比，见 [3.2 节](#32-gate-单独贡献同-prompt-口径)。

## 2.B 旁支探索（被证伪或未采纳）

### 2.B.1 beam search（累计异常分剪枝）

**想法**：每跳保留 B 条候选路径，用隐状态给每条路径打分，分数差的剪掉——即使某跳的错误当时看不出来，
只要它在后续跳留下痕迹，还有机会被淘汰。路径打分 = 累计异常分之和，不掺相似度；相似度只用于圈定
候选池。beam 剪枝本身就是选择机制，比贪心反而少一次 LLM 调用。

**结果**（MuSiQue dev 全量，bw=3，同一份分解与 gate）：EM 0.4055 vs 贪心 0.4042，F1 0.4973 vs
0.4983——**在噪声范围内持平**。

**为什么**：`gated_rule_a` 本身已有局部重排机制；「异常分低的路径」未必是「最后能答对的路径」，
这个代理指标本身不够准；bw=3 可能不够宽。代码：`error_propagation_probe/beam_search_retrieval.py`。

### 2.B.2 把分解也纳入同一个 beam（设计完成，未实现）

**想法**：BART 推理时把 `num_return_sequences` 从 1 改成 M，beam 的起点从 1 条空路径变成 M 条
种子路径，每条带着自己的子问题序列，让现有 gate 间接评判「哪个分解更好」。跳数不一致的路径改比
「每跳平均异常分」，并加一条「单跳异常分超阈值直接淘汰」的保底规则。

**前置验证要求**（写代码前就定好的）：BART `num_beams=10` 解码时，第 2、3 名候选是否与第 1 名有
实质差异，以及扩大到 top-3/top-5 后正确分解是否真的会出现。`check_decompose_diversity.py` 已写好，
**未跑**。方案本身未实现。

### 2.B.3 pairwise MLP 重排（方案 A）

**想法**：把固定公式 `emb_score − λ·abnormal` 换成学出来的模型，输入 `[emb_score, PCA(Δ)]`，
用 RankNet 式 pairwise 损失训练（gold vs 一个反事实构造的错误候选）。

**离线非常好看**：dev pairwise accuracy 0.7238（只用 emb_score）→ **0.9208**。

**端到端倒退**：MuSiQue recall@1 0.6513 → **0.5955**，chain@1 0.4233 → 0.3496。

**诊断**：92% 测的是「gold 对一个特定错误候选能不能赢」，真实检索要**同时赢过 top-10 里其余 9 个**。
就算独立同分布，单场 92% 打赢 9 个也只有 0.92⁹≈45%，现实中候选间的错误还不独立。追加「hop 深度
特征」（one-hot(j) + K，因为原版把所有 K、所有 j 混在一起训了一个模型）把 EM/F1 缺口缩小一半，
但 recall@1 仍差 6 点——**方向对、量级不够，没触及分布不匹配这个根本问题**。

### 2.B.4 listwise 对比学习重排（方案 B）

**想法**：打分模型结构与方案 A 相同，但训练数据与目标改为「gold 要在**真实检索出来的整个 top-k
候选池**里胜出」，listwise softmax 交叉熵。

**状态**：候选池构建脚本 `build_topk_pools.py` 完成并被后续大量复用（`adaptive_lambda_rerank`、
`combined_gate_rerank`、`context_gate_rerank` 的离线实验全部建立在这批候选池缓存上）。端到端对比
一直没跑，优先级被 h_j probe 那条线挤掉了。

### 2.B.5 h_j probe 单独扛排序

**第一次**：排序**只用** probe 分数，cosine 只用来圈定候选池。结果全面倒退（MuSiQue recall@1
0.6513 → 0.5675，EM 0.4079 → 0.3645）。

**诊断**：复盘发现生产从来没让 gate/probe 单独扛过排序——`lr_rerank` 是小权重修正项，
`gated_rule_a` 只是触发开关。这是一次没有先例的新用法，不是重复 pairwise MLP 那个老问题。

**混回 cosine 之后**（`emb_score − 0.25·probe_score`）：recall@1 0.6735、chain 0.4659，**检索侧
反而略高于 baseline**，但 EM/F1 仍低一截（0.3930/0.4816 vs 0.4079/0.5012）。

**留下一个值得记住的反直觉现象**：检索排序更好 ≠ 最终答案更准。

### 2.B.6 自适应 λ（三次失败）

**想法**：不用固定 λ，训一个小模型，输入这一跳检索前的上下文（主问题 embedding、`h_prev`、
当前子问题 embedding），输出这一跳专属的 λ。

- **第一次**：`softplus` 无上界，λ 跑飞到 mean=9.99，recall@1 0.701 < 纯 cosine 的 0.8067。
- **第二次**：加 `[0,2]` 硬边界后训练卡死（loss 1.86 横跳）。查真实数据发现根因：`h_prev` 的 L2
  范数均值 ≈149，而 BGE 向量是 1.0，**尺度差 150 倍**——无界激活会跑飞、有界激活会饱和，
  两次失败是同一个根因的两种症状。归一化后 loss 恢复下降，但**塌缩成常数**（所有跳都输出上界 2.0）。
- **第三次**：换成 [0,1] 凸组合 + 池内 z-score 校准，训练终于健康（gate mean=0.4817、std=0.3788、
  覆盖 0~1），但 recall@1 0.8042 **仍不如固定 λ=0.25 的 0.8601**。

**结论**：这条路线的根本问题是输入全部是**检索发生之前**就定好的信息，模型从头到尾没看过这一跳
实际召回出来的候选长什么样，靠预判猜权重信息量不够。**方向本身被证伪，不是训练没调好。**

副产品：这条线建的候选池打分缓存（`cand_emb_score` / `cand_gate_v2_score` / `cand_abnormal_score`
/ `cand_h_pca`）成了后续所有离线重排实验的基础设施，省掉了反复跑 GPU 端到端。

### 2.B.7 后融合的各种投票与权重方案

两个信号各自独立训练，只在推理阶段组合输出分数。在同一次批量前向里一起打分（gate v2 需要的
{15,23} 与 probe 需要的 {31} 共享一次 Llama forward）。

| 方案 | 做法 | 结果 |
|---|---|---|
| `sum` | `emb − 0.25·gate_v2 − 0.25·probe` | 端到端 recall@1 0.6869 / EM 0.4141，三种投票里唯一全面超过两个单独方法 |
| `veto` | 任一模型判异常（p>0.5）就否决 | 没有效果 |
| `rrf` | 两边各自重排后融合**名次**（k=60） | recall@1 0.6705，EM 0.3922，答案侧甚至不如 baseline |
| 置信度加权 | 按池内打分标准差动态分权 | 离线 recall@1 0.8646 < sum 的 0.8660 |
| 粗筛精排 | probe 筛 top-k 再用 gate_v2 重排 | 0.8643（k=3）/ 0.8624（k=5） |
| 网格搜索 | 独立扫 (λ_gate, λ_probe) | 最优 (0.25,0.10) → 0.8690，但协同修复率从 5.5% 掉到 1.5% |
| 三方凸组合 | cosine + probe + gate_v2 权重和为 1 | 最优 (0.6,0.2,0.2) → 0.8679；均等三分 0.8463 但协同修复率 16.0% |

**协同作用是真的**：`sum` 在两个单独方法都错的 1,186 个样例里修好了 4.9%（约 58 个），不是
「学会偏向更强的一方」。

**但反复出现同一个权衡**：越偏向整体准确率最大化的权重配置，互补修复率越低；权重越均衡，互补修复
率越高但整体数字下降。所有后融合方法都停在这条权衡曲线上。**也反复验证了另一条经验：越简单直接的
固定权重，比越「聪明」的方案更稳**（`rrf` 丢掉了置信度幅度，`veto` 丢掉了排序信息）。

### 2.B.8 context gate（逐候选学习权重）

**与自适应 λ 的本质区别**：权重精细到**每一个候选**，由该候选自己选中后的 `h_after` 决定
（`α_i = σ(w^T PCA(h_after)_i + b)`，65 个参数），同一跳不同候选可以算出不同 α——训练时第一次有了
真实的「因候选而异」的梯度信号。

**离线第一次「整体 + 协同」双赢**：recall@1 0.8776（sum 0.8660），两者皆错时的修复率 16.4%
（sum 5.5%，约 3 倍）。α 分布健康（mean 0.5593、std 0.4636、覆盖 0~1），没有塌缩。

**端到端与 v3 打平**：recall@1 0.7018 vs 0.6950、chain 0.5184 vs 0.5010（检索侧更好），
EM 0.4208 vs 0.4224（略低，噪声内）。

**决策**：效果打平时选活动部件更少的方案。v3 是一个 LR + 复用已验证的层选择；context gate 需要
维护 gate v2 + probe + 一个 PyTorch 门控网络三个组件。**v3 定为默认，context gate 作为已验证的
备选保留。**

### 2.B.9 decoupled：检索也不展开占位符

**想法**：为了消除 2.7 节发现的训练/推理不一致，让子问题**始终用原始未展开文本**，检索和 gate
打分两处都不展开，同时去掉逐跳短答案生成（顺带消除「某跳输出 NA 污染后续所有跳」这个影响约 9.2%
样例的级联失败）。

**写代码前就标记出的唯一风险**：原始子问题语义残缺，可能拖累 BGE 检索。

**结果**：四项全面下降（recall@1 −5.93、chain −9.80、EM −6.16、F1 −5.04）。按跳位置拆开精确定位
到原因：**第一跳两版分毫不差**（第一跳不需要占位符，也证明唯一变量就是占位符处理），从第二跳开始
全面更差，K 越大掉得越狠（K=4 第 2 跳 0.2896 vs 0.5068，直接腰斩）。

**结论**：事先标记的风险成立且影响大于预期。正确的修法是把两件事分开——检索继续展开（保住检索
质量），只把 gate 打分的文本换成未展开版，这就是 [2.8 节](#28-rawprefix-与-bge-检索前缀0730)的 rawprefix。

### 2.B.10 两版后续 reader prompt 变体

在 `comparison_hint` 基础上继续翻案例，发现两类未覆盖的问题，各做一版修复，用
`replay_final_reader.py`（只重跑最终答案这一步）在三个数据集上验证，**两版都是净负**：

- **`comparison_reasoning`**（最终答案前先显式写一步推理）：2Wiki +2.65 EM，但 MuSiQue −1.32、
  HotpotQA −2.35。根因：模型把「答案要是实体名、不能是日期/数字」当成无差别规则，连非比较类问题
  也套用（典型案例：正确推出「June 1982」，又因为这条规则强行改成人名）。
- **`comparison_positive`**（把规则改成正向表述、去掉举例）：三个数据集全部净负，2Wiki 降幅最大
  （−3.05 EM）。根因相反：去掉举例后模型识别不出「这是比较类问题」，触发率暴跌，退回老问题。
- 另有 **`short_answer_prompt=type_match`**（约束逐跳短答案的类型匹配疑问词）：MuSiQue +1.94 EM，
  但 2Wiki 基本打平且检索侧小幅下降，HotpotQA 持平，三数据集不一致，未采纳。

**结论**：`comparison_hint` 现有措辞在「触发识别」与「过度触发」之间已是一个不错的平衡点，
两次尝试各改动其中一半，都打破了平衡。这条 prompt 上的局部修补收益递减。

### 2.B.11 在线增量 listwise gate

**想法**：DAgger 式在线增量训练——每跳用**当前模型自己的真实选择**去展开下一跳的 `[Answer N]`
（而不是 gold 答案），让训练上下文分布贴近推理；listwise softmax 代替独立二分类；特征上把 BGE
query 向量、BGE passage 向量、`h_after`、`Δh` 四路各自独立 PCA 后拼接，并因为「线性层学不出向量
比较」而显式加了双线性交叉项 `q^T W d`；只训一个模型（不按 (K,j) 池化）；经验池只增不减、增量重训。

**训练过程指标**（全量 MuSiQue train 含增强数据 24,070 条，单卡 4.43 小时）：逐跳 pick accuracy
hop1 0.9025 → hop2 0.8168 → hop3 0.7376 → hop4 0.7879。

**一个容易踩的坑**：冻结模型评测时，GT decompose 那组（recall@1 0.7823 / EM 0.4791）看起来比 D 组
高 3.6~9.3 点，但**这个对比无效**——D 组的基准本身是 BART decompose 跑的。换成同样用 BART 的公平
对比后，**新方法五项全部落后**（recall@1 −3.09、recall@3 −2.15、chain@1 −2.24、EM −0.91、F1 −1.08）。

**猜测原因**（未验证）：训练只见过 GT 的干净 query，评测换成 BART 的嘈杂 query 有分布偏移；四路
PCA 只在 hop1 的 GT 数据上拟合过一次；超参完全没调过。方法复杂度明显增加，换来的效果更差。

### 2.B.12 steering（仅调研）

**调研结论**（`update_doc/steering_research_report.md`）：

- **方向向量不需要重新训练**：把现有 gate 的 `LogisticRegression.coef_` 反投影回 4096 维隐藏空间
  即可得到一个可用的 steering 方向。
- **核心工程约束**：gate 打分用 HF transformers（可挂 hook 但不生成文本），真正生成推理链文本的是
  vLLM（v1 引擎的 CUDA Graph / PagedAttention 不支持标准 forward hook）。这个约束决定了落地路径。
- **建议的第一个检查点**（几小时、不需要 GPU 生成）：把方向向量加到已知「应判为异常」的负例隐状态
  上，看 gate 自己的判分是否系统性往「正常」方向移动。这一步不成立，后续生成侧的投入就没有意义。

**状态**：只完成调研，未实现。

### 2.B.13 MLP 与双线性 gate（离线更好，未接端到端）

- **浅层 MLP**（0705）：pilot 上 AUC 比 LR 高（0.9266 vs 0.9147），但 target FPR=0.15 下实测
  dev FPR=0.2348（LR 只有 0.198），阈值校准在 train→dev 之间迁移不准。全量验证特意选了纯 LR。
- **gate v4 双线性**（`gate/gate_v4/`，未写进任何周报）：出发点是「v3 的 LR 只能分别给 h_after 和
  Δ_j 打分再相加，无法表达『同样的 Δ_j 在不同起点 h_{j−1} 上意义不同』」。模型用低秩双线性
  `(W_pre h_{j−1})·(W_post h_j) + w_Δ^T Δ_j + b` 显式建模这种交互。脚本注释记录它是**替代一次更早
  的 MLP 尝试**（在 concat 特征上的 MLP 宽度 32 严重过拟合，缩到宽度 8 加重 L2 后仍不如 v3）。
  离线 AUC 0.9457（高于 v3 的 0.9357）、TPR 0.9521，但 **FPR 0.2847**（v3 是 0.1821）、F1 0.7611
  反而低于 v3——又是一次「阈值校准跑偏」。**没有接端到端评测**，`retrieval/results/` 里没有任何 run
  使用 `artifacts_pooled_v4`。

**这三次尝试指向同一条经验**：在这个任务上把 gate 做复杂，离线 AUC 能涨，但阈值校准会跟着变差，
而生产用法（无论是触发阈值还是混进排序公式）对校准是敏感的。

## 2.C 评测指标的精确定义

所有指标的实现都在 `retrieval/run_retrieval_exp.py` 的几个累加器类里，这里写清楚每个指标的分子分母，
因为周报里多处只写了名字。

### 逐跳检索指标（`MetricAccum`，分母是**跳数**）

在**剪枝前的完整候选池**（`--retrieve-k`，默认 10）上，按最终排序（`emb_score − λ·gate_score`）找 gold
的名次，输出 `recall@1`、`recall@3`、`mrr`、`avg_rank`。gold 不在候选池里时名次记为 `None`，计入分母但
不计入分子。

**gold 的匹配方式有两种，这是三方对比时最容易混的地方**：

| 函数 | 口径 | 用在哪 | summary 字段 |
|---|---|---|---|
| `find_gold_rank(ranked, gold_idx)` | **按位置对齐**：第 j 跳只认标注里第 j 跳那一篇 gold | 本项目内部所有对比（A/B/C/D、λ、prompt、global） | `full_pool_gold_rank_overall` |
| `find_gold_rank_set(ranked, gold_idx_set)` | **顺序无关**：命中这道题的任意一篇 gold 即算 | 与 ChainRAG / GRITHopper 对比（它们的跳与 gold 没有位置对应关系） | `full_pool_gold_rank_orderinvariant_overall` |

同一个 run 两个口径都会算，所以 MuSiQue 的 recall@1 有三个值，**引用时必须说明是哪一个**：
**0.7002**（按位置对齐、按段落编号）、**0.8561**（顺序无关、按段落编号）、**0.8827**（顺序无关、
按标题匹配，[3.9 节](#39-precision-对比)三方对比用的口径）。后两者的差距来自同名段落，见 3.9 的口径注。

#### 逐跳指标的分母：只统计「对齐上的跳」

`gold_idx = ex.gold_idxs[hop_j - 1] if hop_j - 1 < len(ex.gold_idxs) else None` —— 循环跑在**预测
跳**上，而 gold 必须在同一下标存在，所以**分母是 min(预测跳数, 标注跳数)**。分解器把题压短时，
后面的 gold 跳根本不进分母。实测漏掉的比例：

| 数据集 | gold 跳总数 | 进入分母 | 漏掉 | 预测偏少的题 |
|---|---|---|---|---|
| MuSiQue | 6,404 | 6,037 | **367（5.73%）** | 350 |
| 2Wiki | 30,654 | 30,621 | 33（0.11%） | 33 |
| HotpotQA | 14,810 | 14,796 | 14（0.09%） | 14 |

MuSiQue 按标注跳数拆开：K=2 一题没压短，**K=3 有 19.3%（147/760）、K=4 有 50.1%（203/405）被压短**。
所以 **MuSiQue 的逐跳 recall@1 偏乐观**，论文里定义指标时必须写明分母是「对齐上的跳」。
2Wiki / HotpotQA 不受影响（漏掉 0.1%）。

#### chain@1 是宽松版：用 `zip` 比对

`hop_matches = [pid == gid for pid, gid in zip(best.para_ids, ex.gold_idxs)]` —— `zip` 截断到较短的
一边，`all(hop_matches)` 只在**重合的那几跳**上判。预测跳数多于标注时，多出来的跳是白送的。
严格版（额外要求 `预测跳数 == 标注跳数`）实测：

| 数据集 | 宽松 chain@1（现用） | 严格 chain@1 | 差 |
|---|---|---|---|
| MuSiQue | 0.5130 | 0.4977 | −1.53 |
| 2Wiki | 0.8627 | 0.8507 | −1.19 |
| HotpotQA | 0.5610 | **0.4974** | **−6.36** |

**HotpotQA 受影响最大**：它的标注跳数恒为 2，而分解器对 902 题预测了 3 跳以上，这些多出来的跳在
`zip` 下完全不计。论文里要么改报严格版，要么在定义处写明是宽松版。

#### EM / F1 用了 answer aliases

判分函数是 `multihop_trajectory/llama_infer_reasoning/run_musique_pipeline.py::judge_answer_official`
（在本仓库之外、同机器上），实现与 MuSiQue 官方 `evaluate_v1.0` 的 answerable 分支一致：
`gts = [answer] + answer_aliases`，EM 与 F1 都取对所有 gold 的最大值。
**MuSiQue dev 有 680/2,417 题（28.1%）带非空 aliases**；2Wiki 与 HotpotQA 的记录里没有 aliases 字段，
等价于只比单个 gold answer。

#### 2Wiki / HotpotQA 的「标注跳数」是怎么定的

这两个数据集都没有标注分解。`normalize_open_qa_record()` 把 **`supporting_facts` 去重后的标题序列**
（截断到 `BART_MAX_HOPS`）当作 gold 跳序列，每个标题映射到候选池里对应的段落下标。所以：

- **HotpotQA 的标注跳数恒为 2**（全部 7,405 题），`k_match` 87.63% = 6,489/7,405 题预测出恰好 2 跳；
- **2Wiki 的标注跳数只有 2 和 4 两个值**，而且由题型唯一决定（见下表）。

**2Wiki dev 的跳数分布**（按 `supporting_facts` 去重后的标题数，这就是代码里的 `K_gold`）：

| 题型 | 标注跳数 | 题数 | 占比 | BART 预测对跳数的比例 |
|---|---|---|---|---|
| compositional | 2 | 5,236 | 41.6% | 99.92% |
| comparison | 2 | 3,040 | 24.2% | 95.66% |
| inference | 2 | 1,549 | 12.3% | 100.00% |
| bridge_comparison | **4** | 2,751 | 21.9% | 98.51% |
| **合计** | | **12,576** | | **98.59%** |

几点说明：

- `supporting_facts` 的**条目数**是 2/3/4/5/6（9,805 / 20 / 2,739 / 11 / 1 题），但**去重标题数**
  只有 2 和 4 —— 多出来的条目是同一篇文章的不同句子。
- `BART_MAX_HOPS=4` 的截断**从未生效**（最大去重标题数就是 4），所有标题也都能在候选池里找到对应
  段落，所以「去重标题数 = 截断后 = 实际 gold 跳数」三者相同。
- 预测侧的跳数分布是 2:9,689 / 3:140 / 4:2,739 / 5:8。错配主要来自 comparison 题（95.66%，
  132 题被预测成 3 或 4 跳）。
- 我复现出的总 `k_match` 是 **98.59%**（12,399/12,576），与 2.0.4 节记录的 **98.66%** 差 8 题，
  应是端到端 run 跳过了几题（分解为空等）。引用时以 run 的 summary 为准。

注意和 [3.8 节](#38-分解器训练数据消融)里「2Wiki dev 跳数准确率 0.9651」**不是同一个指标**：那个是
分解器自身在 **GPT 标注的 1,043 题 2Wiki dev** 上的评测，gold 来自 GPT 生成的分解；这里的 `k_match`
是端到端 run 在**全部 12,576 题**上、以 supporting-fact 标题数为 gold 算的。两个数字不可互相替换。


### 链级与题级指标（分母是**题数**）

- **`chain@1` / `chain@3`**（`ChainAccum`）：一道题的**所有跳**的 gold 名次都 ≤1 / ≤3 才算成功。
  任一跳 gold 不在候选池（`None`）即失败。这是本方法最有说服力的指标。
- **`final_beam_hop_match_rate` / `final_beam_chain_match_rate`**：拿最终真正提交的证据
  （`para_ids`）和 `gold_idxs` 逐位比较，前者按跳平均、后者要求全对。与 `chain@1` 的区别是它衡量
  「最后交出去的那一篇」而不是「排序里的名次」，`beam_width=1` 时两者数值接近。
- **`case_recall@k`**（`CaseRecallAccum`）：与 chain 同义，但分母只统计有 gold 标注的题。
- **`selection_acc`**（`SelectionAccum`）：LLM 从 top-3 里选中 gold 的比例。**v3 起这一步已移除**
  （见 [2.7 节](#27-移除-selection-步骤并发现一个训练推理不一致0727-第五节)），只在 v1/v2 时代的结果里有。
- **`k_match`**：BART 预测跳数与 gold 跳数一致的题数占比。只取决于分解器，与 gate 无关。

### 答案指标

`AnswerAccum`，HotpotQA 官方口径：`normalize_answer`（去冠词/标点/大小写）后算 EM 与 token 重叠 F1，
按题平均。MuSiQue 的 `answer_aliases` 会参与匹配（取最大值）。

### gate 自身指标

- 离线分类：AUC、TPR、FPR、Precision、F1，阈值按训练集 `target_fpr=0.15` 校准。
- 运行时触发率（`GateAccum`）：`hop_trigger_rate`（超阈值的跳占比）、`example_trigger_rate`
  （至少触发一次的题占比）。

### 三方对比专用指标

- **gold 覆盖率**：每题最终用到的证据集合与 gold 集合的交集占 gold 数的比例；**全覆盖率**：覆盖率
  =1.0 的题占比。D 组 / GRITHopper 是「每跳 1 篇」，ChainRAG 是「图扩展后进入 context 的全部段落」，
  所以 ChainRAG 的覆盖率天然占优，**这个指标不能跨方法直接相减**。
- **证据集合 precision**：同一个集合，分母换成检索集合大小。
- **逐跳 precision@3**：每跳前 3 名里的 gold 数 ÷ 3。**有上限**：一题只有 2 篇 gold 时最高 2/3
  （HotpotQA 全部如此），所以跨数据集不能比绝对值。

### 一个曾经踩过的指标坑

`beam_width=1` 时，`oracle_gold_rank_among_survivors` 里的 recall@1/@3/mrr 会**退化成同一个数**——
剪枝后只剩 1 个候选，没有第二三名可比。0730 之后新增 `full_pool_gold_rank_*` 才有了真正的 recall@3，
**0730 之前的周报里的 recall@3 不可用**。

## 2.D 生成侧：prompt 栈与答案解析

gate 只管排序，答案由两级生成产生，这一层对 EM/F1 的影响不比 gate 小（见
[2.10 节](#210-reader-prompt-修复comparison_hint0817)：一句 prompt 带来 2Wiki +9.9 EM）。

### 累计推理上下文

`build_reasoning_context()` 把已走过的跳拼成：

```
Original question: <主问题>

Previous reasoning steps:
Step i:
Subquestion: <展开后的子问题>
Selected evidence: <证据正文>
Answer: <这一跳的短答案>
```

### 第一级：逐跳短答案

`prompt_short_answer_with_context()` = 累计上下文 + 当前子问题 + 选中证据的标题与正文 + 三条约束：
只输出短 span 或短语、数字就输出数字、**段落里没有答案就输出 `NA`**。

输出经 `normalize_short_answer()` 清洗：只取第一行、去掉尾句点；若长度 >40 且含
`no information` / `not mentioned` / `not specified` / `cannot be determined` / `unknown` 等标记，
判为 `NA`。

**这个短答案有两个用途**：展开下一跳子问题里的 `[Answer N]` 占位符，以及进入最终 reader 的上下文。
所以 `NA` 会沿着链条传播——这就是 [2.F.1](#2f1-na-级联失败) 那个失败模式的来源。

未采纳的变体 `prompt_short_answer_with_context_type_match()`：加一句「答案类型要匹配疑问词」
（where→地点、when→日期、who→人或机构、how many→数字）。动机来自案例分析——证据里同时出现日期和地点
时（"died July 11, 1937 in Hollywood"），原 prompt 那句无条件的「数字就输出数字」会让模型在问
"where" 时答出日期。MuSiQue 上 EM +1.94，但 2Wiki/HotpotQA 不一致，未采纳。

### 第二级：最终 reader

system prompt 固定要求「只用给定的推理轨迹，重读原问题后在单独一行输出 `FINAL ANSWER: <短答案>`，
之后不要再写任何东西」。user prompt 由四个变体之一构造：

| 变体 | 相对 default 的差别 | 状态 |
|---|---|---|
| `default` | — | 0817 之前的默认 |
| **`comparison_hint`** | 加一句：比较类问题的逐跳答案只是用于比较的中间量，必须自己完成比较并回答**实体名** | **当前默认** |
| `comparison_reasoning` | 要求先显式写一步推理再给答案 | 净负，未采纳 |
| `comparison_positive` | 把规则改成正向表述、去掉举例 | 净负，未采纳 |

`parse_final_answer_from_cot()`：取**最后一个** `FINAL ANSWER:` 匹配，剥掉 markdown 包裹、引号、
尾句点；没有匹配到就回退到 fallback（**= 最后一跳的短答案**），再不行输出 `NA`。
这个 fallback 很关键——它意味着「CoT 没按格式输出」不会直接判零分，而是退化成「用最后一跳的答案」。

## 2.E 三个数据集的特性与差异

同一套方法在三个数据集上的绝对数字差很多，主要原因在数据集本身，不在方法：

| | MuSiQue | 2WikiMultihopQA | HotpotQA |
|---|---|---|---|
| dev 题数 | 2,417 | 12,576 | 7,405 |
| 候选池大小（distractor） | 17–20（均值 **20.0**） | 恒 **10** | 2–10（均值 **10.0**） |
| gold 段落数 | = 跳数：2 跳 1,252 / 3 跳 760 / 4 跳 405 | 2 篇 9,825 / 4 篇 2,751 | 恒 2 篇 |
| 题型 | 按跳数分 2/3/4 hop | compositional 5,236 / comparison 3,040 / bridge_comparison 2,751 / inference 1,549 | bridge 5,918 / comparison 1,487（全部 hard） |
| gold 标注粒度 | **段落级**（`paragraph_support_idx`） | 句子级 `supporting_facts` + KB 三元组 `evidences` | 句子级 `supporting_facts`（均值 2.4 句） |
| 自带问题分解 | **有**（`question_decomposition`） | 无（有 KB 三元组可间接推） | 无 |
| 分解器跳数匹配率 | 83.82% | 98.66% | 87.63% |

**由此解释几个一直出现的现象**：

- **2Wiki 所有指标最高**（recall@1 0.93、chain@1 0.87）：候选池只有 10 篇、题型结构最规整
  （2 跳或 4 跳两种模式）、分解器跳数匹配率 98.7%。
- **MuSiQue chain@1 最低**（0.51）：候选池最大（20 篇）、2/3/4 跳混合、4 跳题的分解跳数准确率只有
  0.49，链条越长越容易在某一跳断掉。
- **HotpotQA 全是 2 跳却 recall@1 只有 0.66**：说明它的难点不在链长，而在候选池里的干扰段落与
  gold 语义更接近（它是为「干扰段落强」专门构造的），并且问题措辞更口语化。
- **只有 MuSiQue 能训 gate**：它是唯一提供逐跳 gold 证据对应关系（段落级 `paragraph_support_idx`）
  的数据集。2Wiki / HotpotQA 的 `supporting_facts` 只说「哪些句子是支撑」，没有「第 j 跳用哪一篇」，
  所以本项目的 gate 全部用 MuSiQue 训练，另外两个数据集是纯跨数据集泛化测试
  （HotpotQA 更彻底——分解器也没有用过它的任何标注）。

## 2.F 失败模式汇总

论文的 error analysis 章可以直接用这一节。每一条都有量化数据和来源。

### 2.F.1 NA 级联失败

某一跳的短答案生成失败输出 `NA`，这个 `NA` 会被写进下一跳子问题的 `[Answer N]` 占位符，污染后续
所有跳的检索查询。MuSiQue dev 全量里约 **9.2%** 的样例最终卡在这个模式上（0727 第 5.3 节）。
0727 曾想靠「完全不展开占位符」一次性消除它，但那样检索质量掉得更多（见
[2.B.9](#2b9-decoupled检索也不展开占位符)），最终没有专门修复这一条。

### 2.F.2 reader 选错跳

检索全对但答案错的案例里，最终答案等于**更早一跳**的短答案（而不是该综合出的答案）：
MuSiQue 8.0%（39/485）、2Wiki 13.1%（639/4,894）、HotpotQA 15.2%（232/1,522）。集中在比较类问题上，
且往往不是「选错了跳」而是**根本没执行比较**，直接把某一跳查到的日期原样输出。
`comparison_hint` 把 2Wiki 上的复现率从 13.1% 降到 2.6%（0817）。

### 2.F.3 EM 的严格性造成大量「假错」

MuSiQue 上「检索全对但 EM=0」的 485 个案例里，**77%（372 个）只是措辞与 gold 不完全一致**
（如 "McKinley" vs "President McKinley"），不是真实错误。所以 **EM 系统性低估了方法的实际表现**，
论文里报 F1 更公平；真正的问题只占这类案例的约 23%。

### 2.F.4 错误放大

一旦某跳选错证据，gate 会把后续**本身正确**的候选也判成异常：43%–50% 超阈值，而全对前缀下的基线
只有 17%（0928）。生产是贪心逐跳推进，所以错误会沿链放大，而且 gate 不能定位错误位置，无法支撑
「回退重选」。见 [2.19 节](#219-gate-能力剖析它到底在判断什么0928)。

### 2.F.5 分解器把长链压短

4 跳题有 51% 的跳数预测不对，绝大多数是被压成 3 跳（MuSiQue dev，`hop count acc`=0.489）。
跳数少了，有些 gold 段落根本不会被任何一跳的查询指向。

### 2.F.6 gold 不在候选池

distractor setting 下这一项不严重（候选池只有 10–20 篇，gold 基本都在）。但 global setting 下
**13%–29% 的跳 gold 排在 BGE 前 50 名之外**，重排完全无能为力——这是 global 与 distractor 之间
8–12 点差距的主要来源（见 [2.18 节](#218-候选池分析瓶颈在-bge-召回0921-第五节)）。

### 2.F.7 检索提升不传导到答案

至少出现过三次（gate v2 端到端、probe + cosine、context gate）：检索侧指标明显变好，EM/F1 却持平
甚至略降。0713 曾怀疑是 reader 的天花板，但 oracle 实验否证了这个解释
（见 [2.4 节](#24-否证reader-是天花板这个假设0713-第五节)）。更可能的解释是 EM/F1 的严格性
（2.F.3）加上答案合成环节本身的损耗。

## 2.G 与相关工作的定位

### 早期整理的相关方向

| 方向 | 代表工作 | 与本项目的关系 |
|---|---|---|
| 多跳 QA 数据集 | MuSiQue、2WikiMultihopQA、HotpotQA | 提供多跳问题与证据监督；本项目用 MuSiQue 构造 trace，在另两个上做泛化验证 |
| RAG 的 query 分解 | TableRAG、Transforming Questions and Documents for Semantically Aligned RAG | 同样依赖分解，但本项目进一步学习**每一跳的错误检测** |
| 自适应 / 不确定性感知检索 | FLARE、Self-RAG、Adaptive-RAG、CRAG | 这些方法用置信度、反思 token 或检索质量评估决定是否检索/修正；本项目用隐状态 transition 训一个轻量 LR，**不需要每步做复杂反思，也不改生成模型** |
| 图 / 路径式 RAG | S-Path-RAG | 改进知识图谱路径检索；本项目不改索引结构，只在推理过程中控制每跳的检索决策 |

（文献链接见 `update_doc/汇报报告.md` 第 2 节。）

### 复现对比的两个工作（0831）

| | ChainRAG（ACL 2025） | GRITHopper（EACL 2026） | 本方法 |
|---|---|---|---|
| 是否拆解 | 拆 | **不拆**（decomposition-free 迭代检索） | 拆 |
| 训练投入 | 零训练 | 微调 7B（GritLM 底座，训练数据含这三个数据集） | 零训练底座 + 一个 PCA+LR 小 gate |
| 兜底机制 | 句子图扩展，广撒网 | 迭代检索 + 停止概率 | gate 重排 |
| 检索命中率 | 最低 | **最高** | 中间 |
| 最终 EM/F1 | **最低** | 落后本方法 | **最高**（HotpotQA 与 GRITHopper 基本持平） |

### 本方法的定位

1. **轻量**：gate 是 PCA(64)+LR，训练只要 CPU 几分钟；检索器、生成模型、索引结构全都不动。
2. **可迁移**：同一个 gate 接在 BGE（双塔 cosine）和 ColBERT（late interaction）后面，带来同量级
   提升（见 [2.11 节](#211-colbert验证-gate-的可迁移性0824)），只要做好分数归一化。
3. **收益集中在链级**：chain@1 提升 +18 到 +26 点，远大于单跳 recall@1 的 +8 到 +12 点。
4. **用的是生成模型自己的内部状态**，不额外调用 LLM 做反思/评判——与 Self-RAG / CRAG 这类
   「再问一次模型」的路线相比，推理成本只多一次前向（而且可以和已有的生成批处理共享）。
5. **机制已被刻画清楚**：它本质是「前缀污染检测器」（见
   [2.19 节](#219-gate-能力剖析它到底在判断什么0928)），这既解释了收益为什么集中在链级，
   也划出了它的能力边界。

---

# 第三部分 结果汇总

每张表标注 pipeline 配置和 `retrieval/results/` 下的目录名。所有端到端评测都是 dev 全量
（MuSiQue 2,417 / 2Wiki 12,576 / HotpotQA 7,405），`--beam-width 1`、`--retrieve-k 10`、
BART decompose，除注明外检索器是 BGE。

## 3.1 主结果：三数据集、三信号对照（A/B/C/D）

**配置**：λ=0.60、统一第 15 层 artifact、rawprefix、BGE 检索指令前缀、`comparison_hint` reader
prompt、k=10、distractor、真实 recall@1/@3（剪枝前完整候选池）。**四组唯一变量是用哪个 gate 信号**：
A 不用 gate（λ=0），B 只用转移 Δ_j，C 只用累积状态 h_j，D 用 `concat(PCA(h), PCA(Δ))`。

| 数据集 | 组别 | recall@1 | recall@3 | chain@1 | EM | F1 | run 目录 |
|---|---|---|---|---|---|---|---|
| MuSiQue | A 无 gate | 0.6094 | 0.7679 | 0.3612 | 0.3289 | 0.4138 | `20261003_070503_nogate_musique_lambda0_dev` |
| | B 仅 Δ | 0.6869 | 0.8453 | 0.4812 | 0.4328 | 0.5291 | `20261008_204721_l15_B_deltaonly_musique_dev` |
| | C 仅 h | 0.6982 | **0.8458** | 0.5097 | 0.4386 | 0.5359 | `20261008_213411_l15_C_honly_musique_dev` |
| | **D 融合** | **0.7002** | 0.8438 | **0.5130** | **0.4410** | **0.5359** | `20261008_071635_l15_musique_lambda0p60_dev` |
| 2Wiki | A 无 gate | 0.8334 | 0.9148 | 0.6694 | 0.5212 | 0.5855 | `20261003_070301_nogate_2wiki_lambda0_dev` |
| | B 仅 Δ | 0.9142 | 0.9821 | 0.8309 | 0.5882 | 0.6651 | `20261009_002550_l15_B_deltaonly_2wiki_dev` |
| | C 仅 h | 0.9285 | **0.9849** | 0.8601 | 0.5994 | 0.6784 | `20261009_021558_l15_C_honly_2wiki_dev` |
| | **D 融合** | **0.9302** | 0.9848 | **0.8627** | **0.6014** | **0.6809** | `20261008_085006_l15_2wiki_lambda0p60_dev` |
| HotpotQA | A 无 gate | 0.5745 | 0.7739 | 0.3897 | 0.4604 | 0.5825 | `20261003_070253_nogate_hotpot_lambda0_dev` |
| | B 仅 Δ | 0.6592 | **0.8798** | 0.5409 | 0.5467 | 0.6812 | `20261008_221555_l15_B_deltaonly_hotpot_dev` |
| | C 仅 h | 0.6605 | 0.8683 | 0.5508 | 0.5487 | 0.6848 | `20261008_233432_l15_C_honly_hotpot_dev` |
| | **D 融合** | **0.6620** | 0.8701 | **0.5610** | **0.5515** | **0.6878** | `20261008_074208_l15_hotpot_lambda0p60_dev` |

**相对 A 组的增益**（百分点）：

| 数据集 | 组别 | recall@1 | chain@1 宽松 | **chain@1 严格** | EM | F1 |
|---|---|---|---|---|---|---|
| MuSiQue | B 仅 Δ | +7.75 | +12.00 | +11.91 | +10.39 | +11.53 |
| | C 仅 h | +8.88 | +14.85 | +14.97 | +10.97 | +12.21 |
| | **D 融合** | **+9.08** | **+15.18** | **+15.26** | **+11.21** | **+12.21** |
| 2Wiki | B 仅 Δ | +8.08 | +16.15 | +16.11 | +6.70 | +7.96 |
| | C 仅 h | +9.51 | +19.07 | +19.02 | +7.82 | +9.29 |
| | **D 融合** | **+9.68** | **+19.33** | **+19.29** | **+8.02** | **+9.54** |
| HotpotQA | B 仅 Δ | +8.47 | +15.12 | +13.82 | +8.63 | +9.87 |
| | C 仅 h | +8.60 | +16.11 | +14.90 | +8.83 | +10.23 |
| | **D 融合** | **+8.75** | **+17.13** | **+15.84** | **+9.11** | **+10.53** |

四组共用同一个分解器，所以宽松与严格的差值接近常数（MuSiQue/2Wiki 几乎不变，HotpotQA 统一低约
1.3 点）。**B < C < D 的单调性在严格口径下同样成立**，结论不变。各臂的严格 chain@1 绝对值：
A 组 0.3451 / 0.6578 / 0.3390，B 0.4642 / 0.8189 / 0.4772，C 0.4948 / 0.8480 / 0.4880，
D 0.4977 / 0.8507 / 0.4974。

**三条结论，三个数据集完全一致**：

1. **B < C < D 单调**，recall@1 / chain@1 / EM / F1 四项无一例外。
2. **仅用 h 已经非常接近融合**：C→D 只差 +0.15~+0.20 recall@1、+0.03~+0.26 chain@1、+0.20~+0.28 EM。
   与离线指标吻合（h-only AUC 0.9345 vs v3 0.9357，见 [1.4 节](#14-gate-artifact-版本清单)）。
3. **仅用 Δ 明显更弱**：B→C 差 +0.13~+1.43 recall@1、+0.99~+2.92 chain@1，同样与离线一致
   （Δ-only AUC 0.9203）。

所以「融合两种信号」的增量很小但方向稳定；**真正承担判别的是累积状态 h，而不是转移 Δ**。这与
[3.10 节](#310-gate-能力剖析0928)的 fusion logit 拆解一致（h 的 Cohen's d 1.51~2.20，Δ 只有
0.01~0.65）。recall@3 上 B 组偶尔最高（MuSiQue 之外），但那一项三组差异都在 0.2 点内，不改变结论。

**口径说明（重要）**：此前 0810 的这张表里，**B/C 两组用的是 default reader prompt、D 组用的是
`comparison_hint`**，因为 `run_retrieval_exp_wavefront_gate_v2_rawprefix.py` 与
`..._gate_h_only_rawprefix.py` 当时没有 `--final-reader-prompt` 参数、硬编码调用
`build_final_reader_cot_prompt`。那张表的 EM/F1 列因此**跨组不可比**（检索指标不受影响）。本次重跑
前给两个脚本补上了该参数（默认 `comparison_hint`，与 D 组一致），四组现已同口径。

## 3.2 gate 单独贡献（同脚本、同 prompt、唯一变量是 λ）

对照臂是 `--lambda-gate 0`：同一个脚本、同一套 BGE 前缀与 `comparison_hint` prompt、同一批 batch
顺序，只是 `rank_key = emb_score − λ·gate_score` 里的 λ 取 0,gate 分数不进排序。**这是目前唯一
干净的 gate 贡献口径**（λ=0 时 artifact 不参与任何决策，所以这三个 run 不需要跟着层策略重跑）。

| 数据集 | 指标 | λ=0 | λ=0.60 | Δ（百分点） |
|---|---|---|---|---|
| MuSiQue | recall@1 | 0.6094 | 0.7002 | **+9.08** |
| | recall@3 | 0.7679 | 0.8438 | +7.59 |
| | chain@1 | 0.3612 | 0.5130 | **+15.18** |
| | EM | 0.3289 | 0.4410 | **+11.21** |
| | F1 | 0.4138 | 0.5359 | +12.21 |
| 2Wiki | recall@1 | 0.8334 | 0.9302 | **+9.68** |
| | recall@3 | 0.9148 | 0.9848 | +7.00 |
| | chain@1 | 0.6694 | 0.8627 | **+19.33** |
| | EM | 0.5212 | 0.6014 | **+8.02** |
| | F1 | 0.5855 | 0.6809 | +9.54 |
| HotpotQA | recall@1 | 0.5745 | 0.6620 | **+8.75** |
| | recall@3 | 0.7739 | 0.8701 | +9.62 |
| | chain@1 | 0.3897 | 0.5610 | **+17.13** |
| | EM | 0.4604 | 0.5515 | **+9.11** |
| | F1 | 0.5825 | 0.6878 | +10.53 |

对照臂 run：`20261003_070503/070301/070253_nogate_{musique,2wiki,hotpot}_lambda0_dev`；
实验臂见 [3.3.3 节](#333-λ060-的-dev-全量验证最终主结果)。

**换成不受跳数偏向影响的口径后，gate 的贡献基本不变**（百分点）：

| 数据集 | chain@1 宽松 | **chain@1 严格** | recall@1 对齐分母 | **recall@1 全分母** |
|---|---|---|---|---|
| MuSiQue | +15.18 | **+15.26** | +9.08 | **+8.56** |
| 2Wiki | +19.33 | **+19.29** | +9.68 | **+9.67** |
| HotpotQA | +17.13 | **+15.84** | +8.75 | **+8.75** |

两臂共用同一个分解器，预测跳数分布完全相同，所以口径的偏向对两边一样、相减后基本抵消。最大的变化
是 HotpotQA 的 chain@1（+17.13 → +15.84，低 1.29 点）。**结论不依赖口径选择。**
λ=0 对照臂的严格 chain@1 绝对值为 0.3451 / 0.6578 / 0.3390。

**逐题配对统计**（per-example EM，精确二项 McNemar + 2,000 次配对 bootstrap，seed 0）：

| 数据集 | ΔEM | 95% CI | gate 独赢 | 基线独赢 | McNemar p |
|---|---|---|---|---|---|
| MuSiQue | +11.21 | [+9.64, +12.74] | 355 | 84 | 1.1e-40 |
| 2Wiki | +8.02 | [+7.47, +8.60] | 1,170 | 162 | 1.1e-188 |
| HotpotQA | +9.11 | [+8.33, +9.94] | 847 | 172 | 1.2e-107 |

**chain@1 的提升量级（+15 到 +19）远大于单跳 recall@1（+9 左右）**，因为整条链里任一跳被纠回来都
可能让链从错变对,逐跳的改善在链级累积。

**代价也是真实的**：基线独赢的题数是 84 / 162 / 172,占全量的 3.5% / 1.3% / 2.3%——gate 在全对
前缀下仍有约 17% 的误报率（见 [3.10 节](#310-gate-能力剖析0928)），这些误报会压低本来排第一的
正确候选。净收益远大于代价,但论文里应当把这部分损失一起报出来。

## 3.3 λ 扫描

### 3.3.1 dev 全量扫描（0727，v3 pre-rawprefix 口径）

| λ | recall@1 | chain@1 | EM | F1 |
|---|---|---|---|---|
| 0.10 | 0.6727 | 0.4626 | 0.3906 | 0.4796 |
| 0.20 | 0.6934 | 0.4973 | 0.4216 | 0.5129 |
| 0.25（旧默认） | 0.6950 | 0.5010 | 0.4224 | 0.5146 |
| 0.30 | 0.6975 | 0.5081 | 0.4270 | 0.5180 |
| 0.40 | 0.6980 | 0.5118 | 0.4270 | 0.5202 |
| 0.45 | 0.6990 | **0.5143** | 0.4261 | 0.5195 |
| **0.50** | **0.6992** | 0.5126 | **0.4274** | **0.5207** |
| 0.55 | 0.6979 | 0.5093 | 0.4253 | 0.5193 |
| 0.60 | 0.6960 | 0.5081 | 0.4245 | 0.5183 |

这批 run 目录未保留，数字只在 0727 周报里。λ=0.50 被确认能跨数据集泛化（三个数据集四项全部提升，
2Wiki chain@1 +4.48）。

### 3.3.2 train 子集扫描（1008，当前完整方法口径）

**配置**：MuSiQue **train** 3,000 条子集、BART decompose、rawprefix、comparison_hint、k=10、
统一第 15 层的 v3 artifact。**dev 完全不参与 λ 的选择**。

| λ | recall@1 | recall@3 | chain@1 | EM | F1 | run 目录 |
|---|---|---|---|---|---|---|
| 0.30 | 0.7578 | 0.9032 | 0.5247 | 0.4637 | 0.5486 | `20261008_020714_l15_trainsubset_lambda0p30` |
| 0.40 | **0.7626** | **0.9039** | 0.5333 | 0.4670 | 0.5524 | `20261008_031113_..._lambda0p40` |
| 0.50 | **0.7626** | 0.9011 | 0.5363 | 0.4683 | 0.5539 | `20261008_041518_..._lambda0p50` |
| **0.60** | 0.7622 | 0.9015 | **0.5370** | **0.4707** | **0.5545** | `20261008_051914_..._lambda0p60` |
| 0.70 | 0.7609 | 0.9012 | 0.5353 | 0.4703 | 0.5536 | `20261008_060406_..._lambda0p70` |
| 0.80 | 0.7586 | 0.9003 | 0.5307 | 0.4700 | 0.5524 | `20261008_064018_..._lambda0p80` |

（目录前缀统一为 `l15_trainsubset_`。）

**曲线形状**：0.40–0.70 是一个平台。检索指标（recall@1）在 0.40/0.50 到顶，答案指标（EM / F1）和
chain@1 在 0.60 到顶，两者峰值不在同一个 λ 上，但差距都在小数点后第三位。0.30 明显偏低，0.80 开始
回落。

**逐题配对检验**（per-example EM，精确二项 McNemar，基准 λ=0.60）：

| 对比 | 0.60 独赢 | 对手独赢 | p |
|---|---|---|---|
| vs 0.30 | 75 | 54 | 0.078 |
| vs 0.40 | 33 | 22 | 0.177 |
| vs 0.50 | 34 | 27 | 0.443 |
| vs 0.70 | 11 | 10 | 1.000 |
| vs 0.80 | 18 | 16 | 0.864 |

**平台内部两两差异都不显著**（0.60 与 0.70 之间只有 21 道题结果不同，11 胜 10 负）。只有 0.30 与 0.60
的差距接近显著。所以结论是「**0.40–0.70 为性能平台，取 λ=0.60**」，而不是「0.60 显著最优」——论文里
不应把平台内的 0.2–0.4 点差异当成可解释的效应。

**一个稳健性观察**：此前在 per-j 混层 artifact 上扫过 0.10–0.80 共 9 个值（0831，目录前缀
`gate_v3_rawprefix_musique_trainsubset_big_bart_`），峰值也在 0.60。**层策略换掉之后最优 λ 没有变化**，
说明 λ 的选择对层策略不敏感。

另有 0808 在 600 条 train 子集上扫过 0.10–0.60（目录前缀
`gate_v3_rawprefix_musique_trainsubset_lambda*`），**但那一批用的是 GT 分解**，所以 recall@1 在
0.84 量级、EM 在 0.47–0.49 量级，与上表（BART 分解）不可比；它的峰值落在 λ≈0.25–0.45。

### 3.3.3 λ=0.60 的 dev 全量验证（最终主结果）

**配置**：rawprefix、comparison_hint、λ=0.60。**这是当前方法的最终版本，也是 0914/0921/0928 一切
对比的基准（「D 组」）**。

| 数据集 | n | recall@1 | recall@3 | MRR | chain@1 宽松 | **chain@1 严格** | EM | F1 | 耗时 | run 目录 |
|---|---|---|---|---|---|---|---|---|---|---|
| MuSiQue | 2,417 | 0.7002 | 0.8438 | 0.7786 | 0.5130 | **0.4977** | 0.4410 | 0.5359 | 0.42h | `20261008_071635_l15_musique_lambda0p60_dev` |
| 2Wiki | 12,576 | 0.9302 | 0.9848 | 0.9583 | 0.8627 | **0.8507** | 0.6014 | 0.6809 | 2.42h | `20261008_085006_l15_2wiki_lambda0p60_dev` |
| HotpotQA | 7,405 | 0.6620 | 0.8701 | 0.7790 | 0.5610 | **0.4974** | 0.5515 | 0.6878 | 1.13h | `20261008_074208_l15_hotpot_lambda0p60_dev` |

**严格 chain@1 额外要求「预测跳数 == 标注跳数」**（宽松版用 `zip` 比对，多预测的跳不计、少预测的
跳不查，见[指标口径](#chain1-是宽松版用-zip-比对)）。HotpotQA 两者差 **6.36 点**最大，因为它标注
跳数恒为 2 而分解器对 902 题预测了 3 跳以上。**论文主表建议两列都报。**
同一行的 recall@1 用的是对齐分母；全分母版本（未打分的 gold 跳计为未命中）分别是
**0.6601 / 0.9292 / 0.6614**，MuSiQue 低 4.01 点，另两个几乎不变。

**与此前 per-j 混层 artifact 的同配置结果对比**（验证层策略改动是否有损）：

| 数据集 | recall@1 | chain@1 | EM | F1 |
|---|---|---|---|---|
| MuSiQue | 0.6994 → 0.7002 (+0.08) | 0.5118 → 0.5130 (+0.12) | 0.4386 → 0.4410 (+0.24) | 0.5341 → 0.5359 (+0.18) |
| 2Wiki | 0.9316 → 0.9302 (−0.14) | 0.8685 → 0.8627 (−0.58) | 0.6023 → 0.6014 (−0.09) | 0.6819 → 0.6809 (−0.10) |
| HotpotQA | 0.6620 → 0.6620 (0.00) | 0.5610 → 0.5610 (0.00) | 0.5519 → 0.5515 (−0.04) | 0.6883 → 0.6878 (−0.05) |

HotpotQA 的检索指标逐位相同,MuSiQue 全项微涨,2Wiki 全项微降（chain@1 −0.58 是三个数据集里最大的
变化）。全部落在下面记录的重跑噪声量级内。**结论:把层策略从 per-j 混层（15/15/23/23）换成统一
第 15 层,端到端性能不变**——这与两者 dev 离线 AUC 逐位相同（0.9357）一致。换层的目的是让超参选择
不再依赖 dev,不是为了涨点。旧 run 目录:`20260828_121002/123331/142420_gate_v3_rawprefix_*_lambda060_dev`。

**同配置重跑的噪声水平**（用于判断多大的差异算真实）：MuSiQue 各项 ≤0.2 点，2Wiki ≤0.06 点，
HotpotQA 逐位一致。重跑目录：`20260910_152228/163521/154534_..._rankedpool`（这三个还额外记录了
每跳重排后的完整候选池 `hop_ranked_para_ids`）。

## 3.4 reader prompt 效果

**当前口径**（统一第 15 层 artifact、k=10、distractor）。`default` 臂由
`replay_final_reader.py` 重放 λ=0 run 的最终一步得到——reader prompt 只影响最后一步生成，检索、
gate、逐跳短答案都不变，所以不需要重跑端到端。**重放保真度自检**：用 `comparison_hint` 重放
λ=0 的 MuSiQue，EM 复现到 0.3289（逐位相同，2,417 题里仅 2 题答案变化、一正一负），F1 差 0.04 点，
即 vLLM 批处理噪声量级。自检 run：`20261008_163024_replaycheck_lambda0_musique_hint`。

### 三级消融阶梯（每级只动一个变量）

| 数据集 | 级 | 配置 | EM | F1 | ΔEM | ΔF1 |
|---|---|---|---|---|---|---|
| MuSiQue | 1 纯余弦 | λ=0 + `default` | 0.3190 | 0.4028 | — | — |
| | 2 + prompt | λ=0 + `comparison_hint` | 0.3289 | 0.4138 | +0.99 | +1.10 |
| | 3 + gate | λ=0.60 + `comparison_hint` | **0.4410** | **0.5359** | **+11.21** | +12.21 |
| | | **1→3 总提升** | | | **+12.20** | +13.31 |
| 2Wiki | 1 纯余弦 | λ=0 + `default` | 0.4299 | 0.4927 | — | — |
| | 2 + prompt | λ=0 + `comparison_hint` | 0.5212 | 0.5855 | **+9.13** | +9.28 |
| | 3 + gate | λ=0.60 + `comparison_hint` | **0.6014** | **0.6809** | +8.02 | +9.54 |
| | | **1→3 总提升** | | | **+17.15** | +18.82 |
| HotpotQA | 1 纯余弦 | λ=0 + `default` | 0.4320 | 0.5479 | — | — |
| | 2 + prompt | λ=0 + `comparison_hint` | 0.4604 | 0.5825 | +2.84 | +3.46 |
| | 3 + gate | λ=0.60 + `comparison_hint` | **0.5515** | **0.6878** | **+9.11** | +10.53 |
| | | **1→3 总提升** | | | **+11.95** | +13.99 |

`default` 臂 run：`20261008_163331/163607/164603_l15_lambda0_{musique,2wiki,hotpot}_defaultprompt`。

### prompt 与 gate 的贡献可加（2×2 交互项）

把 λ=0.60 + `default` 这一格也补上（同样用重放），得到完整的 2×2：

| 数据集 | λ=0 + default | λ=0 + hint | λ=0.60 + default | λ=0.60 + hint |
|---|---|---|---|---|
| MuSiQue | 0.3190 | 0.3289 | 0.4286 | **0.4410** |
| 2Wiki | 0.4299 | 0.5212 | 0.5014 | **0.6014** |
| HotpotQA | 0.4320 | 0.4604 | 0.5228 | **0.5515** |

| 数据集 | prompt @λ=0 | prompt @λ=0.60 | gate @default | gate @hint | **交互项** |
|---|---|---|---|---|---|
| MuSiQue | +0.99 | +1.24 | +10.96 | +11.21 | **+0.25** |
| 2Wiki | +9.13 | +10.00 | +7.15 | +8.02 | **+0.87** |
| HotpotQA | +2.84 | +2.87 | +9.08 | +9.11 | **+0.03** |

**三个数据集的交互项都不超过 +0.87 点**（HotpotQA 只有 +0.03）。所以上面三级阶梯里
「1→3 总提升 = prompt 贡献 + gate 贡献」不是近似，而是实测成立的；论文里可以写
*the two contributions are additive to within 0.9 EM points on all three datasets*。

交互项全为正，说明两者有轻微协同——gate 把更多正确证据送到 reader 面前，那一句比较类提示能起作用
的样本就更多——但量级小到不改变任何结论。

λ=0.60 + `default` 臂 run：`20261010_074742/075104_080204_l15_lambda0p60_{musique,2wiki,hotpot}_defaultprompt`。

**最重要的一点**：在 2Wiki 上 **prompt 的贡献（+9.13）大于 gate 的贡献（+8.02）**。写论文时必须把
这两项分开报，否则读者会把那 9 点算到 gate 头上（见 [附二](#附二写论文时需要注意的几处)）。
MuSiQue 与 HotpotQA 则相反，gate 的贡献是 prompt 的 3–11 倍。

**与 0817 旧口径的一致性**：当时在 λ=0.5 + 旧混层 artifact 下测得 prompt 效应为
+1.20 / +9.92 / +3.00，现在在 λ=0 + 统一 15 层下是 +0.99 / +9.13 / +2.84，**三个数据集都在 0.8 点
以内复现**。说明 prompt 的收益基本不依赖 gate 配置，可以当作独立的一项改进来报。

### 这个 prompt 修了什么（0817 的发现过程）

详见 [2.10 节](#210-reader-prompt-修复comparison_hint0817)。一句话版本：原 prompt 里约束答案格式的
那句把 date / number 列为合法答案类型，对所有问题一视同仁；对桥接题「最后一跳短答案就是最终答案」
成立，对**比较类题**不成立（两跳查出两个日期，最终该答「哪一个」而不是日期本身）。修复是加一句话
明确这个区分。受影响样本占比决定了涨幅排序：2Wiki 5.1% > HotpotQA 3.1% > MuSiQue 更低。

未采纳的两版后续变体见 [2.B.10](#2b10-两版后续-reader-prompt-变体)。

## 3.5 ColBERT 检索器对照

**配置**：gate v3（统一第 15 层）、rawprefix、comparison_hint、k=10。「加 gate」用 λ=0.60，
「无 gate」用 λ=0.0。ColBERT 分支额外做候选池内 min-max 归一化（MaxSim 分数没有统一上下界，
不归一化会让 gate 被架空，见 [2.11 节](#211-colbert验证-gate-的可迁移性0824)）。
**无 gate 那三组沿用 0818 的 run**：λ=0 时 gate 分数不进排序，与 artifact 无关；已核对它们的
`final_reader_prompt` 同样是 `comparison_hint`，所以 EM/F1 跨行可比。

| 数据集 | 方案 | recall@1 | recall@3 | chain@1 | EM | F1 | run 目录 |
|---|---|---|---|---|---|---|---|
| MuSiQue | ColBERT 无 gate | 0.6175 | 0.7577 | 0.3732 | 0.3322 | 0.4163 | `20260818_154358_colbert_musique_nogate_full` |
| | ColBERT + gate | 0.6634 | 0.8064 | 0.4427 | 0.3873 | 0.4778 | `20261008_112624_l15_colbert_musique_gate` |
| | *增益* | *+4.59* | *+4.87* | *+6.95* | *+5.51* | *+6.15* | |
| | BGE + gate | **0.7002** | **0.8438** | **0.5130** | **0.4410** | **0.5359** | （3.3.3 节） |
| 2Wiki | ColBERT 无 gate | 0.8413 | 0.9167 | 0.6802 | 0.5223 | 0.5881 | `20260818_170355_colbert_2wiki_nogate_full` |
| | ColBERT + gate | 0.8784 | 0.9641 | 0.7538 | 0.5527 | 0.6230 | `20261008_124446_l15_colbert_2wiki_gate` |
| | *增益* | *+3.71* | *+4.74* | *+7.36* | *+3.04* | *+3.49* | |
| | BGE + gate | **0.9302** | **0.9848** | **0.8627** | **0.6014** | **0.6809** | — |
| HotpotQA | ColBERT 无 gate | 0.5606 | 0.7468 | 0.3685 | 0.4623 | 0.5817 | `20260819_003311_colbert_hotpot_nogate_full` |
| | ColBERT + gate | 0.6083 | 0.8186 | 0.4619 | 0.5218 | 0.6499 | `20261008_114748_l15_colbert_hotpot_gate` |
| | *增益* | *+4.77* | *+7.18* | *+9.34* | *+5.95* | *+6.82* | |
| | BGE + gate | **0.6620** | **0.8701** | **0.5610** | **0.5515** | **0.6878** | — |

**严格 chain@1**（同一分解器，仅作口径补齐，不改结论）：ColBERT + gate 为
**0.4282 / 0.7421 / 0.4059**（宽松 0.4427 / 0.7538 / 0.4619），BGE + gate 为
0.4977 / 0.8507 / 0.4974。两种口径下「ColBERT 加 gate 有效、但整体弱于 BGE」的结论一致。

**两个结论**：

1. **gate 的增量价值不绑定在某一个检索器上**。接在 ColBERT 后面同样涨 recall@1 +3.7~+4.8、
   chain@1 +7.0~+9.3、EM +3.0~+6.0，方向与结构和 BGE 那边一致，只是量级小一些（BGE 上是
   recall@1 +8.8~+9.7、chain@1 +15.2~+19.3，见 3.2 节）。前提是把检索分数归一化到可比范围。
2. **ColBERT 本身三个数据集全面弱于 BGE**（recall@1 差 3.7~5.4 点），所以不把它换成默认检索器。

## 3.6 三方对比 vs ChainRAG / GRITHopper

**口径**：recall@1/@3 为**顺序无关**（三方唯一可共用的定义）；覆盖率 = 每题最终用到的证据集合与
gold 集合的交集比例；EM/F1 为 HotpotQA 官方口径，三方最终答案都由同一个本地
Llama-3.1-8B-Instruct 生成。来源：0831 周报第 2.4 节。

| 数据集 | 方法 | recall@1 | recall@3 | gold 覆盖率 | 全覆盖率 | EM | F1 |
|---|---|---|---|---|---|---|---|
| MuSiQue | D 组（λ=0.60） | 0.8827 | 0.9738 | 0.7970 | 0.5652 | **0.4410** | **0.5359** |
| | ChainRAG | 0.6716 | 0.8628 | **0.9587** | **0.9048** | 0.2764 | 0.3847 |
| | GRITHopper | **0.9058** | **0.9863** | 0.8207 | 0.6090 | 0.2793 | 0.3734 |
| 2Wiki | D 组 | **0.9726** | **0.9974** | 0.9392 | 0.8733 | **0.6014** | **0.6809** |
| | ChainRAG | 0.9199 | 0.9785 | **0.9870** | **0.9696** | 0.4426 | 0.5220 |
| | GRITHopper | 0.9500 | 0.9964 | 0.8922 | 0.7751 | 0.5390 | 0.6267 |
| HotpotQA | D 组 | 0.8988 | 0.9805 | 0.8530 | 0.7313 | 0.5515 | 0.6878 |
| | ChainRAG | 0.8032 | 0.9496 | **0.9864** | **0.9741** | 0.4598 | 0.6027 |
| | GRITHopper | **0.9658** | **0.9959** | **0.9464** | **0.9068** | **0.5546** | **0.6926** |

D 组三行来自新的 `main` run（`20261008_071635/074208/085006_l15_*_lambda0p60_dev`，统一第 15 层）；
ChainRAG / GRITHopper 两行是它们各自系统的输出，不受本项目 gate 影响，未重算。

三方输出数据：`chainrag/processed_data/{musique,2wikimqa,hotpotqa}/results.jsonl`、
`grithopper/results/{musique,2wiki,hotpot}/{retrieval,answers}.jsonl`。

**三方检索投入对比**（解释 recall 排名）：ChainRAG 零训练（现成 embedding + cross-encoder）；
D 组零训练底座（BGE）+ 一个小 gate；GRITHopper 微调了 7B 级模型，训练数据直接含这三个数据集。
recall 排名基本与投入成正比，但**答案效果排名与之相反**。

## 3.7 global setting

**配置**：gate v3（统一第 15 层）、rawprefix、`comparison_hint`、统一第 15 层 artifact。语料库为该
数据集 dev split 全部段落按 `(title, 正文)` 去重合并（MuSiQue 21,100 / 2Wiki 56,687 /
HotpotQA 66,635 篇）。gold 编号重映射全部成功、0 丢失。

本节是 **{λ=0, λ=0.60} × {k=10, k=50}** 的 2×2，再加上 distractor 的参照行。λ=0 两臂是这次新补的
——此前所有 global run 都是 λ=0.60，无法说明 gate 在开放语料下是否仍然有效。

| 数据集 | 条件 | k | λ | recall@1 | recall@3 | chain@1 | EM | F1 | 耗时 | run 目录 |
|---|---|---|---|---|---|---|---|---|---|---|
| MuSiQue | distractor | 10 | 0.60 | **0.7002** | **0.8438** | **0.5130** | **0.4410** | **0.5359** | 0.42h | （3.3.3 节） |
| | global | 10 | 0 | 0.4840 | 0.5904 | 0.2288 | 0.2321 | 0.3169 | 1.22h | `20261008_161412_l15_nogate_global_musique_k10` |
| | global | 10 | 0.60 | 0.5367 | 0.6333 | 0.2975 | 0.2975 | 0.3837 | 0.33h | `20261009_061318_l15_global_musique_k10` |
| | global | 50 | 0 | 0.4840 | 0.5904 | 0.2288 | 0.2321 | 0.3166 | 4.50h | `20261009_035050_l15_nogate_global_musique_k50` |
| | global | 50 | 0.60 | 0.5471 | 0.6627 | 0.3132 | 0.3124 | 0.4042 | 1.48h | `20261008_161903_l15_global_musique_k50` |
| 2Wiki | distractor | 10 | 0.60 | **0.9302** | **0.9848** | **0.8627** | **0.6014** | **0.6809** | 2.42h | （3.3.3 节） |
| | global | 10 | 0 | 0.7433 | 0.8281 | 0.5173 | 0.4843 | 0.5426 | 7.58h | `20261008_201547_l15_nogate_global_2wiki_k10` |
| | global | 10 | 0.60 | 0.7850 | 0.8484 | 0.5815 | 0.5123 | 0.5758 | 2.11h | `20261009_071742_l15_global_2wiki_k10` |
| | global | 50 | 0 | 0.7432 | 0.8281 | 0.5173 | 0.4844 | 0.5427 | 10.43h | `20261009_134820_l15_nogate_global_2wiki_k50` |
| | global | 50 | 0.60 | 0.7867 | 0.8548 | 0.5859 | 0.5183 | 0.5831 | 9.01h | `20261008_211248_l15_global_2wiki_k50` |
| HotpotQA | distractor | 10 | 0.60 | **0.6620** | **0.8701** | **0.5610** | **0.5515** | **0.6878** | 1.13h | （3.3.3 节） |
| | global | 10 | 0 | 0.4790 | 0.5700 | 0.2890 | 0.3753 | 0.4792 | 2.81h | `20261008_172723_l15_nogate_global_hotpot_k10` |
| | global | 10 | 0.60 | 0.5326 | 0.6141 | 0.3726 | 0.4361 | 0.5515 | 0.74h | `20261009_063306_l15_global_hotpot_k10` |
| | global | 50 | 0 | 0.4790 | 0.5702 | 0.2891 | 0.3749 | 0.4789 | 5.46h | `20261009_082043_l15_nogate_global_hotpot_k50` |
| | global | 50 | 0.60 | 0.5523 | 0.6517 | 0.4050 | 0.4591 | 0.5780 | 3.41h | `20261008_174752_l15_global_hotpot_k50` |

### gate 在开放语料下的净贡献（λ=0 → λ=0.60，百分点）

| 数据集 | k | recall@1 | recall@3 | chain@1 | EM | F1 |
|---|---|---|---|---|---|---|
| MuSiQue | 10 | +5.27 | +4.29 | +6.87 | +6.54 | +6.68 |
| | 50 | +6.31 | +7.23 | +8.44 | +8.03 | +8.76 |
| 2Wiki | 10 | +4.17 | +2.03 | +6.42 | +2.80 | +3.32 |
| | 50 | +4.35 | +2.67 | +6.86 | +3.39 | +4.04 |
| HotpotQA | 10 | +5.36 | +4.41 | +8.36 | +6.08 | +7.23 |
| | 50 | +7.33 | +8.15 | +11.59 | +8.42 | +9.91 |

**严格 chain@1**（同一分解器，口径补齐）：k=10 的 λ=0 / λ=0.60 分别是
0.2185→0.2888（MuSiQue）、0.5063→0.5703（2Wiki）、0.2473→0.3215（HotpotQA）；
k=50 是 0.2185→0.3045、0.5064→0.5747、0.2473→0.3512。**gate 的净贡献在严格口径下为
+7.03 / +6.40 / +7.42（k=10）与 +8.60 / +6.83 / +10.39（k=50）**，与宽松口径同量级，结论不变。

**gate 在开放语料下仍然有效**，三个数据集一致：recall@1 +4.2~+7.3、chain@1 +6.4~+11.6、EM
+2.8~+8.4。量级约为 distractor 下的一半（那边是 recall@1 +8.8~+9.7、chain@1 +15.2~+19.3、
EM +8.0~+11.2，见 [3.2 节](#32-gate-单独贡献同脚本同-prompt唯一变量是-λ)），因为开放语料下第一阶段
召回本身就漏掉了大量 gold，重排无从下手（见下）。

### 候选池从 10 扩到 50：只有加了 gate 才有收益

| 数据集 | λ=0 的 k=10→50 | λ=0.60 的 k=10→50 |
|---|---|---|
| MuSiQue | recall@1 0.4840 → 0.4840（**±0.00**） | 0.5367 → 0.5471（**+1.04**） |
| 2Wiki | 0.7433 → 0.7432（**−0.01**） | 0.7850 → 0.7867（**+0.17**） |
| HotpotQA | 0.4790 → 0.4790（**±0.00**） | 0.5326 → 0.5523（**+1.97**） |

**无 gate 时扩大候选池没有任何收益**——MuSiQue 与 HotpotQA 的 recall@1 逐位相同、2Wiki 变动 −0.01
点（一跳之差），因为纯余弦排序下第 1 名不会因为候选池从 10 篇变成 50 篇而改变（多出来的 11–50 名
分数一定更低；唯一能动的是并列分数的排序抖动）。**有 gate 时才能把 11–50 名里的 gold 捞到第 1**：
MuSiQue +1.04、HotpotQA +1.97，而 **2Wiki 只有 +0.17**，因为它只有 1.36% 的跳落在 11–50 档
（见下面的名次分布表）。这比 0921 的「增减分解表」更直接：**gate 是利用更大候选池的必要条件，
而不只是受益者**。代价是 2.4–4.4 倍耗时。

### 候选池里 gold 到底在哪（按 BGE 全局名次分档）

**gold 在 BGE 全局排名的累计分布**（k=50 run 的全部跳，统一第 15 层）：

| 数据集 | ≤1 | ≤3 | ≤10 | ≤20 | ≤50 | **>50（重排救不回来）** |
|---|---|---|---|---|---|---|
| MuSiQue | 0.488 | 0.594 | 0.679 | 0.724 | 0.774 | **0.226** |
| 2Wiki | 0.782 | 0.835 | 0.853 | 0.859 | 0.866 | **0.134** |
| HotpotQA | 0.487 | 0.580 | 0.642 | 0.672 | 0.710 | **0.290** |

**13%–29% 的跳，gold 根本不在前 50 名里**——这是开放语料下 recall@1 的硬上限，与重排质量无关。

**gate 在各档位上做到了什么**（k=50，`gate_top1` = 该档位内被 gate 排到第 1 的比例）：

| 数据集 | BGE 档位 | 占全部跳 | n | gate 排第 1 | gate 进前 3 | 对 recall@1 的贡献 |
|---|---|---|---|---|---|---|
| MuSiQue | 1–10 | 0.6816 | 4,115 | 0.7699 | 0.9091 | 0.5248 |
| | **11–50** | 0.0936 | 565 | **0.2389** | 0.4602 | **+0.0224** |
| | >50 | 0.2248 | 1,357 | 0.0000 | 0.0000 | 0 |
| 2Wiki | 1–10 | 0.8524 | 26,102 | 0.9155 | 0.9907 | 0.7804 |
| | **11–50** | 0.0136 | 417 | **0.4628** | 0.7626 | **+0.0063** |
| | >50 | 0.1340 | 4,102 | 0.0000 | 0.0000 | 0 |
| HotpotQA | 1–10 | 0.6440 | 9,529 | 0.8184 | 0.9410 | 0.5271 |
| | **11–50** | 0.0677 | 1,001 | **0.3726** | 0.6753 | **+0.0252** |
| | >50 | 0.2883 | 4,266 | 0.0000 | 0.0000 | 0 |

三档的贡献相加精确等于该 run 的 recall@1（0.5471 / 0.7867 / 0.5523），可作为自洽性检查。

**「11–50 档」就是把 k 从 10 扩到 50 多出来的那部分候选**：gate 在其中把 23.9% / 46.3% / 37.3% 的
gold 排到了第 1，折算成整体 recall@1 的 **+2.24 / +0.63 / +2.52 点**——与上面「λ=0.60 下扩池 +1.04 /
+0.17 / +1.97」量级一致（差额来自扩池同时也会把前 10 档里原本排第 1 的挤掉一部分）。2Wiki 的收益
最小，因为它只有 1.36% 的跳落在 11–50 档（它的 gold 要么在前 10，要么在 50 名开外）。

脚本与结果：`retrieval/analyze_global_gold_rank.py`、
`retrieval/results/global_gold_rank_analysis.json`（脚本的 `latest_run()` 已改为优先匹配
`*_l15_global_*` 的新 run）。

### global 相对 distractor 的落差

| 数据集 | recall@1 | chain@1 | EM |
|---|---|---|---|
| MuSiQue | 0.7002 → 0.5471（**−15.31**） | 0.5130 → 0.3132（−19.98） | 0.4410 → 0.3124（−12.86） |
| 2Wiki | 0.9302 → 0.7867（**−14.35**） | 0.8627 → 0.5859（−27.68） | 0.6014 → 0.5183（−8.31） |
| HotpotQA | 0.6620 → 0.5523（**−10.97**） | 0.5610 → 0.4050（−15.60） | 0.5515 → 0.4591（−9.24） |

（左为 distractor k=10、右为 global k=50，两者都是 λ=0.60。）落差 11~15 点 recall@1，而 gate 在
global 下只能补回 4~7 点——**开放语料的瓶颈在第一阶段召回,不在重排**。与 [2.18 节](#218-候选池分析瓶颈在-bge-召回0921-第五节)
记录的「13%–29% 的跳 gold 排在 BGE 前 50 名之外」一致：那部分 gold 根本不在候选池里，重排再好也救不回来。

## 3.8 分解器训练数据消融

**配置**：D 组配置（λ=0.60、comparison_hint），唯一变量是分解器。基线为线上分解器
（MuSiQue + 2Wiki，`checkpoint-1425`），实验组为只用 MuSiQue 训练（`bart_decomposer_old/checkpoint-936`，
超参完全相同：lr 3e-5、batch 16、3 epoch、seed 100）。

### 分解器自身

| decomposer | MuSiQue dev BLEU | 跳数准确率 | 2 跳 | 3 跳 | 4 跳 |
|---|---|---|---|---|---|
| MuSiQue + 2Wiki | 76.81 | 0.8366 | 0.982 | 0.782 | **0.489** |
| 只用 MuSiQue | 75.94 | 0.8155 | 0.988 | 0.738 | 0.427 |

2Wiki dev 跳数准确率 0.9651 → 0.7616，**只用 MuSiQue 的模型产生的 4 跳分解数为 0**（线上模型 2,739 个）。

### 端到端

| 数据集 | recall@1 | recall@3 | chain@1 | EM | F1 | run 目录 |
|---|---|---|---|---|---|---|
| MuSiQue | 0.7072（+0.70） | 0.8494（+0.56） | 0.5283（+1.53） | 0.4477（+0.67） | 0.5437（+0.78） | `20261008_154506_l15_musique_musiqueonly_decomposer` |
| 2Wiki | 0.6906（**−23.96**） | 0.8689（−11.59） | 0.5651（**−29.76**） | 0.5188（**−8.26**） | 0.5975（**−8.34**） | `20261008_175915_l15_2wiki_musiqueonly_decomposer` |
| HotpotQA | 0.5826（−7.94） | 0.8293（−4.08） | 0.4706（−9.04） | 0.5433（−0.82） | 0.6801（−0.77） | `20261008_162924_l15_hotpot_musiqueonly_decomposer` |

括号内是相对基线（新 `main` run）的变化，单位百分点。**两臂用同一套统一第 15 层 artifact**，唯一
变量是分解器的训练数据。

#### ⚠️ 本节必须以严格口径为准

上表用的是默认指标，而它们对「预测跳数少」有系统性偏向：`chain@1` 用 `zip` 比对、`recall@1` 的分母是
**对齐上的跳**（见[指标口径](#逐跳指标的分母只统计对齐上的跳)）。只用 MuSiQue 训练的分解器**更少
预测长链**（它在 2Wiki 上产生的 4 跳分解数为 0，逐跳总数 30,621 → 25,272），所以「少比了几跳」本身
就会让它的分数变好看。补算两个不受这个偏向影响的口径：

| 数据集 | 指标 | 基线（MuSiQue+2Wiki） | 只用 MuSiQue | Δ |
|---|---|---|---|---|
| MuSiQue | chain@1 宽松 | 0.5130 | 0.5283 | +1.53 |
| | **chain@1 严格** | 0.4977 | 0.5118 | **+1.41** |
| | recall@1 对齐分母 | 0.7002 | 0.7072 | +0.70 |
| | **recall@1 全分母** | 0.6601 | 0.6587 | **−0.14** |
| 2Wiki | chain@1 宽松 | 0.8627 | 0.5651 | −29.76 |
| | **chain@1 严格** | 0.8507 | 0.5503 | **−30.04** |
| | recall@1 对齐分母 | 0.9302 | 0.6906 | −23.96 |
| | **recall@1 全分母** | 0.9292 | 0.5694 | **−35.98** |
| HotpotQA | chain@1 宽松 | 0.5610 | 0.4706 | −9.04 |
| | **chain@1 严格** | 0.4974 | 0.4469 | **−5.05** |
| | recall@1 对齐分母 | 0.6620 | 0.5826 | −7.94 |
| | **recall@1 全分母** | 0.6614 | 0.5810 | **−8.04** |

（严格 chain@1 额外要求预测跳数 == 标注跳数；全分母 recall@1 的分母是**全部标注 gold 跳**，没被
打分的跳计为未命中。脚本 `retrieval/compute_strict_chain.py`，结果
`retrieval/results/strict_chain_at1.json`。）

**三条结论因此要改写**：

1. **MuSiQue 上「只用 MuSiQue 的分解器反而更好」在检索指标上不成立**：recall@1 的 +0.70 换成全分母
   后变成 **−0.14**，优势完全来自少比了几跳。chain@1 的优势缩小但仍在（+1.53 → +1.41），所以准确
   的说法是「链级略好、逐跳持平」，不是「全面略好」。
2. **2Wiki 的损失被低估了**：全分母下 recall@1 是 **−35.98** 而不是 −23.96。它在 2Wiki 上把大量
   4 跳题压成 2 跳，那些从未被打分的跳在对齐分母下直接消失了。
3. **HotpotQA 的链级损失被高估了**：严格口径下是 **−5.05** 而不是 −9.04。因为基线自己的宽松
   chain@1 就虚高——HotpotQA 标注跳数恒为 2，而基线分解器对 902 题预测了 3 跳以上，多出来的跳在
   `zip` 下不计。

**论文里这一节应直接用严格口径的表**，宽松表放附录或删掉。其他对照（B/C、ColBERT、global、
λ=0 vs λ=0.60）两臂共用同一个分解器、预测跳数分布相同，宽松与严格的差值接近常数，结论不受影响。

### 按题型/跳数拆解

| 数据集 | 分组 | 题数 | EM 变化 |
|---|---|---|---|
| 2Wiki | bridge_comparison（4 跳） | 2,751 | **−20.1** |
| | comparison | 3,040 | **−12.6** |
| | inference | 1,549 | −4.8 |
| | compositional | 5,236 | −0.8 |
| MuSiQue | gold 2 跳 | 1,252 | +1.8 |
| | gold 3 跳 | 760 | −0.1 |
| | gold 4 跳 | 405 | **−3.5** |
| HotpotQA | comparison | 1,487 | **−4.1** |
| | bridge | 5,918 | −0.2 |

HotpotQA 的检索指标掉 8–9 点但答案只掉 1 点：比较类题按位置对齐的命中率从 0.699 掉到 0.396，而
「选中任意一篇 gold」只从 0.896 掉到 0.821——**约 23 点是顺序问题**（比较类题先查 A 还是先查 B 都
对，标注只有一种顺序），7.5 点是真实检索损失。

对比脚本：`retrieval/compare_decomposer_ablation.py`、
`retrieval/results/decomposer_ablation_musiqueonly_decomposer.json`。

## 3.9 precision 对比

**口径**：标题匹配、顺序无关（GRITHopper / ChainRAG 的输出只存了段落标题）。D 组用的是 3.3.3 节
那三个新 `main` run（它们自带每跳重排后的候选池 `hop_ranked_para_ids`，precision@3 可直接算）。

### 证据集合 precision

| 数据集 | 方法 | precision | 覆盖率 | 集合 F1 | 平均检索篇数 | 平均 gold 篇数 |
|---|---|---|---|---|---|---|
| MuSiQue | D 组 | 0.8823 | 0.7970 | 0.8279 | 2.29 | 2.60 |
| | ChainRAG | 0.2068 | **0.9587** | 0.3235 | 14.54 | |
| | GRITHopper | **0.9091** | 0.8207 | **0.8517** | 2.31 | |
| 2Wiki | D 组 | **0.9680** | 0.9392 | **0.9491** | 2.37 | 2.44 |
| | ChainRAG | 0.3559 | **0.9870** | 0.4998 | 7.61 | |
| | GRITHopper | 0.9540 | 0.8922 | 0.9139 | 2.24 | |
| HotpotQA | D 组 | 0.8962 | 0.8530 | 0.8653 | 1.91 | 2.00 |
| | ChainRAG | 0.2752 | **0.9864** | 0.4215 | 7.86 | |
| | GRITHopper | **0.9658** | **0.9464** | **0.9528** | 1.95 | |

### 逐跳 precision@3

| 数据集 | 方法 | 跳数 | recall@1 | recall@3 | **precision@3** | 前 3 名平均 gold 篇数 |
|---|---|---|---|---|---|---|
| MuSiQue | D 组 | 6,037 | 0.8827 | 0.9738 | 0.5111 | 1.53 |
| | ChainRAG | 5,162 | 0.6716 | 0.8628 | 0.3949 | 1.18 |
| | GRITHopper | 6,404 | **0.9058** | **0.9863** | **0.6087** | 1.83 |
| 2Wiki | D 组 | 30,621 | **0.9726** | **0.9974** | 0.6238 | 1.87 |
| | ChainRAG | 22,737 | 0.9199 | 0.9785 | 0.5859 | 1.76 |
| | GRITHopper | 30,654 | 0.9500 | 0.9964 | **0.6786** | 2.04 |
| HotpotQA | D 组 | 14,796 | 0.8988 | 0.9805 | 0.5035 | 1.51 |
| | ChainRAG | 13,658 | 0.8032 | 0.9496 | 0.4679 | 1.40 |
| | GRITHopper | 14,810 | **0.9658** | **0.9959** | **0.5994** | 1.80 |

**三个口径注意点**：

1. **precision@3 有上限**——一题只有 2 篇 gold 时最高 2/3，HotpotQA 全是 2 篇，所以跨数据集不能比
   绝对值。
2. **MuSiQue 上按标题匹配会虚高**：D 组 recall@1 按标题是 0.8827、按段落编号是 0.8561，差 2.66 点，
   因为有题选中了与 gold 同名但不是 gold 的那一篇。2Wiki / HotpotQA 两种匹配方式结果一致。
3. **这个虚高对三方都成立，而且对 ChainRAG / GRITHopper 无法测量**——它们的输出只存了标题，没有段落
   编号，所以拿不到「按段落编号」的对照。虚高的上限是候选池里 gold 标题不唯一的比例：**MuSiQue dev
   有 27.7% 的题、14.0% 的跳**存在与某个 gold 同标题的其他段落。D 组实测虚高 2.66 点远低于这个上限。
   **方向上三方同受其利**，所以对比没有系统性偏向某一方，但三方的绝对值都应视为上界。

脚本与结果：`retrieval/compute_evidence_precision.py`、
`retrieval/results/evidence_precision_3way.json`。

## 3.10 gate 能力剖析（0928）

**数据**：MuSiQue dev 多错 trace，2,417 题 / 16,316 条 trace / 51,672 跳，`n_wrong = 0..K`。
**打分**：gate v3，与生产逐字一致。**自检**：51,672 跳全部打分成功、无缺层。

### 分数 vs 前缀里的错误数

| 前缀中错误数 | 当前跳是错的 | 当前跳是对的 |
|---|---|---|
| 0 | — | 0.170（14,537）/ 超阈值 0.17 |
| 1 | 0.767（13,899）/ 0.86 | 0.398（8,109）/ 0.42 |
| 2 | 0.710（8,747）/ 0.80 | 0.442（2,785）/ 0.48 |
| 3 | 0.697（2,785）/ 0.78 | 0.537（405）/ 0.62 |
| 4 | 0.762（405）/ 0.87 | — |

### 相对第一个错误跳的位置

| 位置 | 全部 | 其中本身选对的 |
|---|---|---|
| 错误前第 3 / 2 / 1 跳 | 0.125 / 0.135 / 0.168 | 同左（都是选对的） |
| 错误发生的那一跳 | **0.767** | —（构造上为空） |
| 错误后第 1 / 2 / 3 跳 | 0.584 / 0.534 / 0.571 | **0.448 / 0.376 / 0.384** |
| 全 gold trace（对照） | 0.186 | — |

### 控制跳的位置

| K | 跳 | 本跳错 | 本跳对、前缀全对 | 本跳对、前缀有错 |
|---|---|---|---|---|
| 2 | 2 | 0.759 | 0.171 | **0.506** |
| 3 | 3 | 0.665 | 0.188 | **0.309** |
| 4 | 3 | 0.687 | 0.352 | **0.428** |
| 4 | 4 | 0.746 | 0.211 | **0.385** |

### trace 级别

| K | n_wrong=0 均值 / 最大值 | n_wrong=K 均值 / 最大值 |
|---|---|---|
| 2 | 0.171 / 0.275 | 0.770 / 0.910 |
| 3 | 0.182 / 0.369 | 0.736 / 0.943 |
| 4 | 0.216 / 0.522 | 0.732 / 0.964 |

Spearman(n_wrong, 分数均值) = 0.614，Spearman(n_wrong, 分数最大值) = 0.512。
**全 gold trace 的最大分随 K 上升**（0.275 → 0.369 → 0.522），所以用「路径最高分」做判别时，
跳数越多区分度越窄。

### fusion 两半的贡献

| 类别 | 跳数 | 分数 | logit(h) | logit(Δ) |
|---|---|---|---|---|
| 正确跳，前缀全对 | 14,537 | 0.170 | **−1.61** | −0.32 |
| 正确跳，前缀有 1 错 | 8,109 | 0.398 | **−0.02** | −0.62 |
| 正确跳，前缀有 2+ 错 | 3,190 | 0.454 | **+0.20** | −0.82 |
| 错误跳，前缀无其他错 | 13,899 | 0.767 | **+3.18** | −0.01 |
| 错误跳，前缀有其他错 | 11,937 | 0.709 | **+1.91** | −0.31 |

按 j 分开的对错分离度（Cohen's d）：j=0 h 2.20 / Δ 0.18；j=1 h 1.70 / Δ 0.47；
j=2 h 1.51 / Δ 0.01；j=3 h 1.93 / Δ 0.65。

脚本与结果：`error_propagation_probe/score_gate_v3_vs_nwrong.py`、
`error_propagation_probe/results/gate_l15_vs_nwrong_dev.json`。

## 3.12 分解器质量指标

### BART 分解器在 MuSiQue dev 上的内在指标

| 评估口径 | n | BLEU | 跳数准确率 |
|---|---|---|---|
| 训练时 eval（MuSiQue dev + 2Wiki dev 混合） | 3,460 | 80.58 | 0.8795 |
| `predict.py` 在 MuSiQue dev 上 | 2,417 | 76.81 | 0.8366 |

按 gold 跳数拆开（MuSiQue dev，线上 checkpoint-1425）：

| gold 跳数 | n | 跳数准确率 | 主要错误模式 |
|---|---|---|---|
| 2 | 1,252 | 0.9824 | 少量预测成 3 跳 |
| 3 | 760 | 0.7816 | 部分被压成 2 跳 |
| 4 | 405 | **0.4889** | 大量被压成 3 跳 |

`exact_match` 恒为 0——子问题是自由文本，与 GT 完全逐字一致几乎不可能，所以这个字段没有参考价值，
实际看的是 BLEU 和跳数准确率。

### 跳数匹配率（k_match，端到端口径）

| 数据集 | k_match | 说明 |
|---|---|---|
| 2Wiki | 98.66% | 绝大多数是 2 跳或 4 跳的规整结构 |
| HotpotQA | 87.63% | 全部是 2 跳 |
| MuSiQue | 83.82% | 2/3/4 跳混合，4 跳最难 |

**已在新 main run 上复算，三个数字逐位相同**（`ch5/scripts/ch5_failure_analysis.py`，
2026-10-11）：MuSiQue 0.8382、2Wiki 0.9866、HotpotQA 0.8763。原始来源是
`retrieval/results/20260628_120339_wavefront_pooled_lam0p25`（gate v1 时代），复算确认了
「k_match 只取决于分解器、与 gate 版本无关」这个判断。论文引用新 run 的数字即可。

### 三个数据集上的跳数分布（线上分解器 vs 只用 MuSiQue 训练）

| 数据集 | 线上（MuSiQue+2Wiki）预测跳数分布 | 只用 MuSiQue 预测跳数分布 | gold 跳数分布 |
|---|---|---|---|
| 2Wiki | 2 跳 9,689 / 3 跳 140 / **4 跳 2,739** / 5 跳 8 | 1 跳 10 / 2 跳 12,428 / 3 跳 138 / **4 跳 0** | 2 跳 **9,825** / 4 跳 **2,751**（见下注） |
| HotpotQA | 1 跳 14 / 2 跳 6,489 / 3 跳 853 / 4+ 跳 49 | 1 跳 40 / 2 跳 6,938 / 3 跳 422 / 4+ 跳 5 | 全部 2 跳 |

**2Wiki「gold 跳数」的口径注**：这一列此前写的是 `2 跳 9,595 / 3 跳 88 / 4 跳 2,806 / 5+ 跳 87`，
那是按 **`evidences` 三元组数**算的；而**检索代码用的是另一种定义**——
`normalize_open_qa_record()` 取 `supporting_facts` **去重后的标题数**（截断到 `BART_MAX_HOPS=4`），
得到的是**只有 2 和 4 两个值**的分布（9,825 / 2,751）。三种可能的定义实测如下：

| 定义 | 2Wiki dev 分布 | 谁在用 |
|---|---|---|
| `supporting_facts` 去重标题数 | **2: 9,825 / 4: 2,751** | 检索代码、`k_match`、论文表 4.1 |
| `evidences` 三元组数 | 2: 9,595 / 3: 88 / 4: 2,806 / 5: 81 / 6: 5 / 7: 1 | 本表原来的写法 |
| `supporting_facts` 条目数（未去重） | 2: 9,805 / 3: 20 / 4: 2,739 / 5: 11 / 6: 1 | 无 |

同一张表里「预测跳数分布」是端到端口径，所以**必须用第一种定义**才能和它比。已改。

## 3.13 完整系统的失败模式（第 5 章用）

**口径**：新 `main` run 的逐题日志；答案规范化直接 import MuSiQue 官方
`metrics/answer.py`（`judge_answer_official` 用的同一个模块），所以「等于」就是 EM 意义下的等于。
脚本 `ch5/scripts/ch5_failure_analysis.py`，结果 `ch5/results/ch5_failure_analysis.json`。
第 5 章的脚本、数据、图都收在 `ch5/` 下，见 [`ch5/README.md`](ch5/README.md)。

**gold 段落的顺序口径**：HotpotQA 只给两个支持段落、标注固定了其中一种顺序，2Wiki
`comparison` / `bridge_comparison` 的两条分支同理——**先查哪个都对**，但按位置比较会把换序判成
检索错。所以 1a / 1b / 1e 全部**同时**报两种判定：

- **按位置**（positional）：`para_ids[j] == gold_idxs[j]`；
- **顺序无关**（order-invariant，v2）：`para_ids[j] ∈ set(gold_idxs)` **且 `para_ids[j]` 没被
  更早的跳取过**。适用范围：**HotpotQA 全部题**、**2Wiki `comparison` + `bridge_comparison`**
  （5,791 题 / 46.1%）。MuSiQue 与 2Wiki 的 compositional / inference 保持按位置，所以它们在两种
  判定下**数字逐位相同**——这也是脚本的自检。

**「没被更早的跳取过」这一条是 1011 补的,很要紧**：第一版顺序无关实现只判 `∈ set(gold_idxs)`，
于是**两跳取了同一篇 gold 也算全对**，而这恰恰是比较类题最常见的真实错误之一。按新规则统计，
main run 里有 **1,425 道 HotpotQA 题（19.2%）和 474 道 2Wiki 题（8.2%）**重复取了同一篇 gold；
λ=0 对照里更多（2,168 / 29.3% 与 908 / 15.7%）。详见 [3.14 节](#314-顺序无关-gold-判定v21011)。

口径实现集中在 `retrieval/goldmatch.py`（`order_invariant_applies` / `hop_hits` /
`chain_at1` / `selected_a_gold_twice`），第 4、5 章共用同一个模块。

### 3.13.1 四类互斥的结局

按顺序判定，命中即停：A 答对 → B 跳数错 → C 检索有错 → D 检索全对但答错。

| 数据集 | 判定 | A 答对 | B 跳数错 | C 检索有错 | D 检索全对但答错 |
|---|---|---|---|---|---|
| MuSiQue | 两种相同 | 1,066（44.1%） | 268（11.1%） | 607（25.1%） | 476（19.7%） |
| 2Wiki | 按位置 | 7,563（60.1%） | 33（0.3%） | 1,136（9.0%） | 3,844（30.6%） |
| | **顺序无关** | 7,563（60.1%） | 33（0.3%） | 1,113（8.9%） | **3,867（30.8%）** |
| HotpotQA | 按位置 | 4,084（55.2%） | 410（5.5%） | 1,496（20.2%） | 1,415（19.1%） |
| | **顺序无关** | 4,084（55.2%） | 410（5.5%） | **1,115（15.1%）** | **1,796（24.3%）** |

**HotpotQA 上这个口径差别很大**：按位置算的 C 类（20.2%）里有 **381 题（占全体 5.1 个点）
其实是把两个支持段落换序取了**，顺序无关下它们移到 D 类。所以 HotpotQA 的「检索有错」应报
**15.1%**，「检索全对却答错」应报 **24.3%**。2Wiki 只动了 23 题（比较类题本来 k_match 就高）。

**2Wiki 的主要失败模式不是检索而是答案生成**：检索全对却答错的占 30.8%，是检索出错（8.9%）的
三倍多。**HotpotQA 在订正后也是生成侧占多数**（24.3% vs 15.1%），这一点按位置口径是看不出来的。
MuSiQue 相反，B+C（跳数错 + 检索错）合计 36.2%。

按标注跳数拆开（MuSiQue）：

| gold 跳数 | A 答对 | B 跳数错 | C 检索有错 | D 检索全对但答错 |
|---|---|---|---|---|
| 2 | 50.8% | 0.9% | 23.2% | 25.2% |
| 3 | 41.3% | 14.5% | 26.8% | 17.4% |
| 4 | **28.6%** | **36.3%** | 27.9% | 7.2% |

**4 跳题的失败有一半以上来自跳数预测**（36.3% 落在 B 类），而不是检索或生成。

### 3.13.2 D 类（检索全对却答错）的三档拆解

四档**互斥且按顺序判定**（1024 订正：原来三档，把弃答混进了①）：⓪ 弃答（规范化后等于 `NA`）；
① 只是措辞不同；② 最终答案重复了**某个非末跳**的短答案；③ 其余。D 类本身按顺序无关口径取。

**①「措辞不同」只能当上界，所以报成区间**：宽松规则（规范化后一方是另一方的子串，或 token
F1 ≥ 0.5）会把**答错了另一个问题**的预测也算进来——例如 MuSiQue `2hop__62369_84616` 预测
"Ondine"、gold 是 "for Best Performance by a Leading Actress in a Play in Ondine"，剧名被 gold
包含，但问的是奖项，这是答错；HotpotQA 的 "Audi Quattro" vs "Audi" 则两可。严格规则
（双向 token F1 ≥ 0.8）不会命中这类，是下界。

| 数据集 | D 类题数 | ⓪ 弃答 NA | ① 措辞不同（下界–上界） | ② 重复中间跳答案 | ③ 其他 |
|---|---|---|---|---|---|
| MuSiQue | 476 | 6.7% | 8.6% – 41.0% | **6.5%** | 45.8% |
| 2Wiki | 3,867 | 3.1% | 7.5% – 39.2% | **1.5%** | 56.3% |
| HotpotQA | 1,796 | 0.4% | 16.3% – 57.4% | **8.2%** | 33.9% |

**正文写法**：没做人工核对之前，只能写「**最多** 41% / 39% / 57% 只是措辞问题」，或写区间。
`ch5/results/bucketD_wording_audit_sample.jsonl` 已经从每个数据集的①里随机抽了 50 条
（含 question / prediction / gold / 规则命中的档位 / F1，留了 `verdict_fill_in` 字段），
人工过一遍就能给出规则准确率 p，正文改写成「约 41%×p」。**这一步还没做。**

② 的定义只算**非末跳**：桥接题的最终答案本来就应该等于最后一跳的答案，那不是失败模式；
`comparison_hint` 针对的是「把用于比较的中间值当成答案」。弃答 `NA` 也不算②（原来当某个
中间跳也是 NA 时会误判成「重复了中间跳答案」），这让 MuSiQue 的②从 8.0% 降到 6.5%。

**这一档直接验证了 prompt 的作用**。把同一批链用 `default` prompt 重放后再算②：

| 数据集 | `comparison_hint` | `default` | 0817 旧口径 |
|---|---|---|---|
| MuSiQue | 6.5% | 6.9% | — |
| 2Wiki | **1.5%** | **13.3%** | 13.1% → 2.6% |
| HotpotQA | 8.2% | 12.5% | — |

**2Wiki 上 13.3% → 1.5%，和 0817 手工统计的 13.1% → 2.6% 几乎一致**（这次是规则化、可复现的口径）。
MuSiQue 两边都是 6~7%，因为它本来就没有比较类题——这正是 prompt 在 MuSiQue 上只值 +0.99 EM 的原因。

按题型（D 类内的 ② 比例）：2Wiki inference 3.4% > comparison 1.8% > compositional 1.3% >
bridge_comparison 0.5%；HotpotQA comparison 9.5% > bridge 8.0%。

### 3.13.3 NA 级联

**定义**：某一跳 j < K 的短答案规范化后为 `NA`，**且**后面至少有一个子问题用 `[Answer j]` 引用了它。

**NA 级联的题可以落在任何一类**（A/B/C/D 都有），所以两个方向的比例都列出来，不要只写「占 C 类」
（本节原来那一列 23.4% / 12.3% / 14.6% 是**转抄错误**，json 从来没有这三个数，现已按下表订正）：

| 数据集 | 出现 NA 级联 | 其中落在 C 类 | 占 C 类的比例 | 落在 A 类 | 落在 B 类 | 落在 D 类 |
|---|---|---|---|---|---|---|
| MuSiQue | 261（10.8%） | 34.5% | **14.8%** | 33.0% | 15.3% | 17.2% |
| 2Wiki | 337（2.7%） | 15.7% | **4.8%** | 42.4% | 0.6% | 41.3% |
| HotpotQA | 454（6.1%） | 19.4% | **7.9%** | 53.3% | 6.0% | 21.4% |

读法：「其中落在 C 类」分母是**该数据集的 NA 级联题数**，「占 C 类的比例」分母是**C 类题数**。
HotpotQA 一半以上的 NA 级联题最后还是答对了（落在 A 类），所以 NA 本身不等于失败。
（表里 C 类按顺序无关口径取；按位置口径下 2Wiki 4.8%、HotpotQA 6.8%。）

跳命中率（NA 之后的跳 vs 完全没有 NA 的题里的跳）：

| 数据集 | 判定 | NA 之后 | 无 NA 题 | 差 |
|---|---|---|---|---|
| MuSiQue | 两种相同 | **0.4978** | 0.7222 | −22.4 点 |
| 2Wiki | 按位置 | 0.8190 | 0.9318 | −11.3 点 |
| | 顺序无关 | 0.8234 | 0.9455 | −12.2 点 |
| HotpotQA | 按位置 | 0.6327 | 0.6628 | −3.0 点 |
| | **顺序无关** | **0.8013** | **0.8461** | **−4.5 点** |

**NA 确实把后续检索带偏**：MuSiQue 上 NA 之后的跳命中率比无 NA 的题低 **22.4 点**，2Wiki 低 12.2 点。
HotpotQA 只低 3~4.5 点——它几乎全是 2 跳，NA 之后通常只剩一跳。注意 HotpotQA 的**绝对**命中率在两种
判定下差了 17~18 个点（0.63 vs 0.80），再一次说明那边必须用顺序无关口径。旧口径记的
「MuSiQue 约 9.2%」与现在的 10.8% 同量级。

### 3.13.4 gate 改变了什么（解释第 4 章的 84 / 162 / 172）

把 main 与 control 逐题配对，在**第一个两者选择不同的跳**上判断发生了什么：

HotpotQA 和 2Wiki 比较类题按**顺序无关**口径（粗体行）；MuSiQue 两种判定相同。

| 数据集 | 判定 | 组 | n | ② gate 纠正了错误 | ① gate 压下了正确的第一名 | ③ 两者都不是 gold | ④ 选择完全相同 |
|---|---|---|---|---|---|---|---|
| MuSiQue | 两种相同 | gate 独赢 | 355 | **262（73.8%）** | 35 | 56 | 2 |
| | | control 独赢 | 84 | 15 | **43（51.2%）** | 22 | 4 |
| 2Wiki | 按位置 | gate 独赢 | 1,170 | 1,097（93.8%） | 14 | 59 | 0 |
| | | control 独赢 | 162 | 62 | 60（37.0%） | 39 | 1 |
| **2Wiki** | **顺序无关** | gate 独赢 | 1,170 | **1,123（96.0%）** | 9 | 38 | 0 |
| | | control 独赢 | 162 | 74 | **51（31.5%）** | 36 | 1 |
| HotpotQA | 按位置 | gate 独赢 | 847 | 528（62.3%） | 129 | 190 | 0 |
| | | control 独赢 | 172 | 68 | 55（32.0%） | 49 | 0 |
| **HotpotQA** | **顺序无关** | gate 独赢 | 847 | **713（84.2%）** | **14（1.7%）** | 120 | 0 |
| | | control 独赢 | 172 | 48 | **50（29.1%）** | 74 | 0 |

**HotpotQA 的「gate 压下了正确的第一名」从 129 掉到 14**：按位置口径下那 129 题最后全都答对了，
很反常——实际上 gate 多半只是先取了另一个支持段落，与标注顺序不一致而已。这正是上面那个口径
问题的直接证据，**正文必须用顺序无关的 14（1.7%）**。

**两个读法**：

1. **gate 的收益主要来自真正纠正检索错误**（74%–96% 属于②），不是生成侧的随机性——④「选择完全相同、
   只是答案生成不同」在六组里最多只有 4 题。顺序无关口径下②反而**更大**（HotpotQA 528 → 713），
   因为 λ=0 对照重复取同一篇 gold 的情况比 main 多得多（29.3% vs 19.2%），按位置口径把这类错误
   也判成「两者都不是 gold」，掩盖了 gate 的功劳。
2. **gate 的损失有三成到一半来自把正确的第一名压下去**（①：51% / 32% / 29%），这与
   [3.10 节](#310-gate-能力剖析0928)测得的约 17% 误报率、以及错误之后对正确候选的抬高是同一件事的
   两面。但另有相当一部分损失属于③（两者都没选中 gold，差异来自别处），**不能全部算在 gate 头上**。
   注意**损失侧的①在两种判定下都在 29%~51%**，没有被顺序口径救回去——gate 的误报是真的。

### 3.13.5 自然错误下的前缀污染（按错误模式分组）

[prelim_analysis](prelim_analysis/) 的实验 4 按「包含第 h 跳错误」分组，组间重叠严重（一条两跳都错
的链同时进两组），结论不可读。改为按**完整的错误跳元组**分组，每条链只属于一组；对照组是同一个 K
下**一跳都没错**的链；同时报两种分数：Δ 对「正确转移参考分布」的 Mahalanobis z，以及**生产 gate v3
自己的打分**。脚本 `ch5/scripts/exp4b_natural_by_pattern.py`（前置的隐状态由
`prelim_analysis/` 的实验 4 第 2 步产出）。

| 组 | K | 错误模式 | 转移 | n | Δz（95% CI） | **Δgate（95% CI）** |
|---|---|---|---|---|---|---|
| 错误之后的正确跳 | 2 | (1,) | 2 | 91 | +0.75 [+0.52,+0.97] | **+0.239 [+0.174,+0.305]** |
| 错误之后的正确跳 | 3 | (2,) | 3 | 94 | +0.91 [+0.67,+1.16] | **+0.208 [+0.154,+0.264]** |
| 错误发生的那一跳 | 2 | (2,) | 2 | 389 | +1.02 [+0.88,+1.15] | +0.538 [+0.503,+0.574] |
| 错误发生的那一跳 | 3 | (3,) | 3 | 110 | +0.45 [+0.26,+0.63] | +0.508 [+0.446,+0.568] |
| **错误之前的跳** | 2 | (2,) | 1 | 389 | −0.06 [−0.18,+0.07] | **+0.056 [+0.028,+0.085]** |
| **错误之前的跳** | 3 | (3,) | 1 | 110 | −0.17 [−0.33,+0.00] | +0.023 [−0.004,+0.054] |
| **错误之前的跳** | 3 | (3,) | 2 | 110 | +0.28 [+0.09,+0.46] | **+0.096 [+0.047,+0.145]** |

（K=3、错误模式 (1,) 只有 10 条链，已标注不解读。）

**结论一：自然错误下也有前缀污染。** 错误之后**本身选对**的跳，gate 分数比无错链高
**+0.21~+0.24**（CI 不含 0），方向与 [3.10 节](#310-gate-能力剖析0928)在构造错误上的发现一致，
论文里可以写 "consistent with"。

**结论二：存在一个较小但真实的选择效应，结论要相应收紧。** 错误**之前**的跳也略高
（+0.02~+0.10，其中两格 CI 不含 0）——也就是说「会出错的题」本身就和不出错的题有点不同。
不过这个量级只有错误之后效应的 **1/2 到 1/10**，所以前缀污染不能被它解释掉，但论文里应当把它写出来。

---

## 3.14 顺序无关 gold 判定（v2，1011）

**决定**（`ORDER_INVARIANT_TASK.md`）：第 4 章的 recall / chain@1 在「标注顺序本来就随意」的题上
改用顺序无关判定。本节是全量重算的结果。**旧数字一律保留**，`retrieval/compute_strict_chain.py`
没有改动，新口径单独走 `retrieval/compute_order_invariant_v2.py` →
`retrieval/results/order_invariant_v2.json`。

### 3.14.1 协议

口径实现集中在 `retrieval/goldmatch.py`，第 4、5 章共用。

| 项 | 定义 |
|---|---|
| 适用范围 S | HotpotQA 全部题；2Wiki `comparison` + `bridge_comparison`（5,791 / 46.1%）。MuSiQue 全部、2Wiki `compositional` / `inference` 保持按位置 |
| 逐跳命中 | `para_ids[j] ∈ set(gold_idxs)` **且** `para_ids[j] ∉ para_ids[:j]` |
| recall@3 | 第 j 跳重排后前 3 名里存在一篇 `∈ set(gold_idxs)` 且不在 `para_ids[:j]` 里 |
| chain@1 宽松 | 前 `min(Kp, Kg)` 跳两两不同且全部命中 |
| chain@1 严格 | `Kp == Kg` 且 `para_ids` 是 `gold_idxs` 的一个排列 |
| 分母 | 对齐分母 `min(Kp, Kg)` 与全分母 `Kg` 都报（下表用全分母，对齐分母在 json 里） |

**自检**：MuSiQue 和 2Wiki 的 compositional / inference 不启用顺序无关，所以这些 run 的两列
**逐位相同**（下表 MuSiQue 六行可见）。

**「不重复」这一条是关键修正。** 第一版实现只判 `∈ set(gold_idxs)`，于是两跳取同一篇 gold 也算
全对——这正是比较类题最常见的真实错误之一，等于把它放过了。旧规则把 chain@1 严格**高估了
最多 22.2 个百分点**（HotpotQA ColBERT 无 gate），见 3.14.3。

### 3.14.2 全量结果

**表 4.6 主结果 / λ=0 对照**

| run | EM | chain@1 严格 位置 | chain@1 严格 顺序无关 | recall@1 全分母 位置 | 顺序无关 | recall@3 全分母 位置 | 顺序无关 | 重复取 gold 的题 |
|---|---|---|---|---|---|---|---|---|
| `musique/gate_lambda0p60` | 0.4410 | 0.4977 | **0.4977** | 0.6601 | **0.6601** | 0.7954 | **0.7954** | — |
| `musique/control_lambda0` | 0.3289 | 0.3451 | **0.3451** | 0.5745 | **0.5745** | 0.7239 | **0.7239** | — |
| `2wiki/gate_lambda0p60` | 0.6014 | 0.8507 | **0.8623** | 0.9292 | **0.9426** | 0.9837 | **0.9901** | 474（8.2%） |
| `2wiki/control_lambda0` | 0.5212 | 0.6578 | **0.6640** | 0.8325 | **0.8414** | 0.9138 | **0.9250** | 908（15.7%） |
| `hotpot/gate_lambda0p60` | 0.5515 | 0.4974 | **0.6262** | 0.6614 | **0.8427** | 0.8693 | **0.9629** | 1425（19.2%） |
| `hotpot/control_lambda0` | 0.4604 | 0.3390 | **0.4189** | 0.5739 | **0.7072** | 0.7732 | **0.8859** | 2168（29.3%） |

**表 4.9 特征对照（B 仅 Δ / C 仅 h）**

| run | EM | chain@1 严格 位置 | chain@1 严格 顺序无关 | recall@1 全分母 位置 | 顺序无关 | recall@3 全分母 位置 | 顺序无关 | 重复取 gold 的题 |
|---|---|---|---|---|---|---|---|---|
| `musique/B_delta_only` | 0.4328 | 0.4642 | **0.4642** | 0.6476 | **0.6476** | — | **—** | — |
| `musique/C_h_only` | 0.4386 | 0.4948 | **0.4948** | 0.6582 | **0.6582** | — | **—** | — |
| `hotpot/B_delta_only` | 0.5467 | 0.4772 | **0.5969** | 0.6586 | **0.8262** | — | **—** | 1624（21.9%） |
| `hotpot/C_h_only` | 0.5487 | 0.4880 | **0.6130** | 0.6599 | **0.8350** | — | **—** | 1543（20.8%） |
| `2wiki/B_delta_only` | 0.5882 | 0.8189 | **0.8288** | 0.9133 | **0.9251** | — | **—** | 623（10.8%） |
| `2wiki/C_h_only` | 0.5994 | 0.8480 | **0.8610** | 0.9275 | **0.9423** | — | **—** | 429（7.4%） |

**表 4.10 分解器消融（MuSiQue-only 分解器）**

| run | EM | chain@1 严格 位置 | chain@1 严格 顺序无关 | recall@1 全分母 位置 | 顺序无关 | recall@3 全分母 位置 | 顺序无关 | 重复取 gold 的题 |
|---|---|---|---|---|---|---|---|---|
| `musique/musique_only_decomposer` | 0.4477 | 0.5118 | **0.5118** | 0.6587 | **0.6587** | 0.7911 | **0.7911** | — |
| `hotpot/musique_only_decomposer` | 0.5433 | 0.4469 | **0.6307** | 0.5810 | **0.8178** | 0.8271 | **0.9523** | 1112（15.0%） |
| `2wiki/musique_only_decomposer` | 0.5188 | 0.5503 | **0.6304** | 0.5694 | **0.7462** | 0.7163 | **0.8128** | 672（11.6%） |

**表 4.11 ColBERT 检索器**

| run | EM | chain@1 严格 位置 | chain@1 严格 顺序无关 | recall@1 全分母 位置 | 顺序无关 | recall@3 全分母 位置 | 顺序无关 | 重复取 gold 的题 |
|---|---|---|---|---|---|---|---|---|
| `musique/colbert_nogate` | 0.3322 | 0.3600 | **0.3600** | 0.5821 | **0.5821** | — | **—** | — |
| `musique/colbert_gate` | 0.3873 | 0.4282 | **0.4282** | 0.6254 | **0.6254** | 0.7601 | **0.7601** | — |
| `2wiki/colbert_nogate` | 0.5223 | 0.6688 | **0.6747** | 0.8404 | **0.8489** | — | **—** | 981（16.9%） |
| `2wiki/colbert_gate` | 0.5527 | 0.7421 | **0.7504** | 0.8775 | **0.8883** | 0.9631 | **0.9709** | 795（13.7%） |
| `hotpot/colbert_nogate` | 0.4623 | 0.3206 | **0.4158** | 0.5600 | **0.7080** | — | **—** | 2292（31.0%） |
| `hotpot/colbert_gate` | 0.5218 | 0.4059 | **0.5315** | 0.6077 | **0.7846** | 0.8178 | **0.9349** | 2039（27.5%） |

**表 4.12 开放语料（global）**

| run | EM | chain@1 严格 位置 | chain@1 严格 顺序无关 | recall@1 全分母 位置 | 顺序无关 | recall@3 全分母 位置 | 顺序无关 | 重复取 gold 的题 |
|---|---|---|---|---|---|---|---|---|
| `musique/global_k10_lambda0` | 0.2321 | 0.2185 | **0.2185** | 0.4563 | **0.4563** | 0.5565 | **0.5565** | — |
| `musique/global_k10_lambda0p60` | 0.2975 | 0.2888 | **0.2888** | 0.5059 | **0.5059** | 0.5970 | **0.5970** | — |
| `musique/global_k50_lambda0` | 0.2321 | 0.2185 | **0.2185** | 0.4563 | **0.4563** | 0.5565 | **0.5565** | — |
| `musique/global_k50_lambda0p60` | 0.3124 | 0.3045 | **0.3045** | 0.5158 | **0.5158** | 0.6248 | **0.6248** | — |
| `2wiki/global_k10_lambda0` | 0.4843 | 0.5063 | **0.5114** | 0.7425 | **0.7499** | 0.8272 | **0.8356** | 336（5.8%） |
| `2wiki/global_k10_lambda0p60` | 0.5123 | 0.5703 | **0.5757** | 0.7842 | **0.7920** | 0.8475 | **0.8557** | 223（3.9%） |
| `2wiki/global_k50_lambda0` | 0.4844 | 0.5064 | **0.5115** | 0.7424 | **0.7499** | 0.8272 | **0.8356** | 336（5.8%） |
| `2wiki/global_k50_lambda0p60` | 0.5183 | 0.5747 | **0.5800** | 0.7859 | **0.7937** | 0.8539 | **0.8621** | 204（3.5%） |
| `hotpot/global_k10_lambda0` | 0.3753 | 0.2473 | **0.2983** | 0.4785 | **0.5927** | 0.5694 | **0.6953** | 1164（15.7%） |
| `hotpot/global_k10_lambda0p60` | 0.4361 | 0.3215 | **0.4049** | 0.5321 | **0.6837** | 0.6135 | **0.7473** | 962（13.0%） |
| `hotpot/global_k50_lambda0` | 0.3749 | 0.2473 | **0.2980** | 0.4786 | **0.5926** | 0.5697 | **0.6953** | 1164（15.7%） |
| `hotpot/global_k50_lambda0p60` | 0.4591 | 0.3512 | **0.4419** | 0.5518 | **0.7105** | 0.6511 | **0.7885** | 995（13.4%） |

> recall@3 在三组 run 上是 `—`：B/C 特征对照（`gate_v2` / `h_only` 脚本）和 2026-08 的 ColBERT
> 无 gate run 的逐题日志里没有 `hop_ranked_para_ids`，候选列表没落盘，补不回来。如果表 4.9 /
> 4.11 要报 recall@3，这三组得重跑。

### 3.14.3 旧（无去重）规则高估了多少

单位：chain@1 严格，百分点。只列适用顺序无关的 run（MuSiQue 全部为 0）。

| run | 位置 | 新顺序无关 | 旧（无去重） | 旧规则高估 |
|---|---|---|---|---|
| `2wiki/gate_lambda0p60` | 0.8507 | 0.8623 | 0.8889 | **+2.66** |
| `2wiki/control_lambda0` | 0.6578 | 0.6640 | 0.7265 | **+6.24** |
| `hotpot/gate_lambda0p60` | 0.4974 | 0.6262 | 0.7229 | **+9.67** |
| `hotpot/control_lambda0` | 0.3390 | 0.4189 | 0.6285 | **+20.96** |
| `hotpot/B_delta_only` | 0.4772 | 0.5969 | 0.7290 | **+13.21** |
| `hotpot/C_h_only` | 0.4880 | 0.6130 | 0.7244 | **+11.14** |
| `2wiki/B_delta_only` | 0.8189 | 0.8288 | 0.8681 | **+3.93** |
| `2wiki/C_h_only` | 0.8480 | 0.8610 | 0.8843 | **+2.33** |
| `hotpot/musique_only_decomposer` | 0.4469 | 0.6307 | 0.7359 | **+10.52** |
| `2wiki/musique_only_decomposer` | 0.5503 | 0.6304 | 0.6677 | **+3.73** |
| `2wiki/colbert_nogate` | 0.6688 | 0.6747 | 0.7436 | **+6.89** |
| `2wiki/colbert_gate` | 0.7421 | 0.7504 | 0.8038 | **+5.34** |
| `hotpot/colbert_nogate` | 0.3206 | 0.4158 | 0.6381 | **+22.23** |
| `hotpot/colbert_gate` | 0.4059 | 0.5315 | 0.7080 | **+17.65** |
| `2wiki/global_k10_lambda0` | 0.5063 | 0.5114 | 0.5339 | **+2.25** |
| `2wiki/global_k10_lambda0p60` | 0.5703 | 0.5757 | 0.5877 | **+1.20** |
| `2wiki/global_k50_lambda0` | 0.5064 | 0.5115 | 0.5340 | **+2.25** |
| `2wiki/global_k50_lambda0p60` | 0.5747 | 0.5800 | 0.5901 | **+1.01** |
| `hotpot/global_k10_lambda0` | 0.2473 | 0.2983 | 0.4157 | **+11.74** |
| `hotpot/global_k10_lambda0p60` | 0.3215 | 0.4049 | 0.4848 | **+7.99** |
| `hotpot/global_k50_lambda0` | 0.2473 | 0.2980 | 0.4154 | **+11.74** |
| `hotpot/global_k50_lambda0p60` | 0.3512 | 0.4419 | 0.5184 | **+7.66** |

**读法**：`旧（无去重）` 一列就是 1024 那版顺序无关实现。它在 HotpotQA 上把 chain@1 严格抬高了
8~22 个百分点，在 2Wiki 上 1~7 个百分点。**正文一律用「新顺序无关」那一列。**

### 3.14.4 bridge_comparison 的分支结构（已核实）

`ORDER_INVARIANT_TASK.md` 假设去重后的 `supporting_facts` 标题序是「分支 1 的两篇 + 分支 2 的两篇」。
**核实结果：不是。** 顺序是**交错**的：

```
Which film has the director who was born later, El Extraño Viaje or Love In Pawn?
gold = (El extraño viaje, Love in Pawn, Fernando Fernán Gómez, Charles Saunders (director))
         分支1 第1跳      分支2 第1跳    分支1 第2跳            分支2 第2跳
```

即 `g1→g3`、`g2→g4`。2,751 道 dev bridge_comparison 题**全部**满足「去重标题序 = `evidences`
三元组首次提到各主语的顺序」；用「三元组的宾语 = 下一个三元组的主语」验证链结构，2,601/2,751 成立，
余下 150 条差别只在消歧后缀（`Charles Saunders` vs `Charles Saunders (director)`），是字符串问题
不是结构问题。

所以按任务书自己给的退路，**正式口径用「排列且不重复」**。另外我把真正的约束（每部电影必须排在
自己导演之前）也算了一遍当诊断：

| run | n | chain@1 严格（排列） | chain@1 严格（保持分支内先后） |
|---|---|---|---|
| `2wiki/gate_lambda0p60` | 2718 | 0.8915 | 0.8915 |
| `2wiki/control_lambda0` | 2718 | 0.7502 | 0.7502 |
| `2wiki/B_delta_only` | 2718 | 0.8790 | 0.8790 |
| `2wiki/C_h_only` | 2718 | 0.9205 | 0.9202 |
| `2wiki/colbert_nogate` | 2718 | 0.7509 | 0.7509 |
| `2wiki/colbert_gate` | 2718 | 0.8197 | 0.8194 |
| `2wiki/global_k10_lambda0` | 2718 | 0.5574 | 0.5574 |
| `2wiki/global_k10_lambda0p60` | 2718 | 0.5743 | 0.5743 |
| `2wiki/global_k50_lambda0` | 2718 | 0.5574 | 0.5574 |
| `2wiki/global_k50_lambda0p60` | 2718 | 0.5548 | 0.5548 |

**两列几乎完全相同**（最大差 0.0003，即 1 题）——说明退回「排列」几乎没有放宽任何东西，
这个退路是安全的，不需要为 bridge_comparison 单独写规则。

### 3.14.5 对结论的影响

1. **方向不变，gate 的优势反而变大。** 1024 我按旧（无去重）规则估的是「幅度缩水」，**那个判断是
   错的**，原因就是旧规则把对照组重复取 gold 的错误放过了：

   | 数据集 | Δchain@1 严格 位置 | Δ 顺序无关 | Δrecall@1 全分母 位置 | Δ 顺序无关 |
   |---|---|---|---|---|
   | MuSiQue | +15.26 | +15.26 | +8.56 | +8.56 |
   | 2Wiki | +19.29 | **+19.83** | +9.67 | **+10.12** |
   | HotpotQA | +15.84 | **+20.73** | +8.75 | **+13.55** |

2. **「重复取同一篇 gold」本身是个值得写的发现**：main run 里 HotpotQA 19.2% / 2Wiki 8.2%，
   λ=0 对照里 29.3% / 15.7%。**gate 把这个比例压低了约 10 个和 7 个百分点**，这是 gate 作用的
   一个独立证据（它让后续跳避开已经取过的段落）。
3. **EM / F1 完全不受影响**，它们根本不读 `gold_idxs`。表 4.7 不用动。
4. 第 5 章已同步（1a 的分桶本来就用多重集比较，不受「不重复」影响，数字没变；1b 的跳命中率和
   1e 的归因变了，见 3.13）。

---

## 3.11 旁支探索结论一览

| 探索 | 离线结果 | 端到端结果 | 结论 | 代码 |
|---|---|---|---|---|
| beam search（bw=3） | — | EM 0.4055 vs 贪心 0.4042 | 持平，未采纳 | `error_propagation_probe/beam_search_retrieval.py` |
| 分解纳入 beam | — | 未实现 | 前置验证未跑 | — |
| pairwise MLP | 配对准确率 0.9208（基线 0.7238） | recall@1 0.5955 vs 0.6513 | 倒退，分布不匹配 | `rerank_pairwise_mlp/` |
| listwise 对比 | — | 未跑 | 候选池基础设施被复用 | `rerank_listwise_contrastive/` |
| probe 单独排序 | — | recall@1 0.5675 | 全面倒退 | `error_propagation_probe/wavefront_hidden_probe_beam.py` |
| probe + cosine（λ=0.25） | — | recall@1 0.6735 / EM 0.3930 | 检索侧超基线，答案侧不及 | 同上 `--rerank-mode weighted` |
| 自适应 λ（3 次） | 离线 recall@1 最好 0.8042 vs 固定 0.8601 | — | 方向被证伪 | `adaptive_lambda_rerank/` |
| 后融合 sum | 离线 0.8660，协同修复 5.5% | recall@1 0.6869 / EM 0.4141 | 三种投票里最好，仍不如 v3 | `combined_gate_rerank/` |
| 后融合 rrf / veto | 0.8463–0.8690 | rrf EM 0.3922 | 均不如 sum | 同上 |
| context gate | 离线 0.8776，协同修复 16.4% | recall@1 0.7018 / EM 0.4208 | 与 v3 打平，因复杂度未采纳 | `context_gate_rerank/` |
| decoupled（都不展开） | — | recall@1 −5.93 / chain −9.80 | 负面，被 rawprefix 取代 | `..._gate_v3_decoupled.py` |
| reader prompt 变体 ×3 | — | 均净负或三数据集不一致 | comparison_hint 为最终版 | `replay_final_reader.py` |
| 在线增量 listwise | 训练 pick acc hop1 0.90 | 五项全部落后 v3 | 未采纳 | `online_listwise_gate/` |
| steering | — | — | 仅调研 | `steering/`、`steering_research_report.md` |
| MLP gate | pilot AUC 0.9266 | 未跑 | FPR 校准跑偏（0.2348） | `gate/compare_layers_pilot.py` |
| 双线性 gate v4 | AUC 0.9457 / TPR 0.9521 | **未跑** | FPR 0.2847，未接端到端 | `gate/gate_v4/` |

---

## 附一：从零复现的命令序列

按依赖顺序排。耗时标注里，带「实测」的是本项目真实观测到的，其余是量级估计。
环境：vLLM 相关的步骤用 `conda activate mlsys`；BART 训练锁 `transformers==4.57.3`。

### 第 1 步 问题分解（上游，CPU + OpenAI API + 1 卡）

```bash
# 1a. MuSiQue GT 分解 → 自然语言（需 OpenAI API）
cd decompose/gt/scripts
python gt_decompose_to_nl.py --split train --max 0
python gt_decompose_to_nl.py --split dev   --max 0
python build_exact_gt_trace.py --nl-decomp ../musique_gt_nl_train.jsonl \
  --musique ../../../data/raw/musique/musique_ans_v1.0_train.jsonl \
  --output ../train_exact_gt_traces.jsonl

# 1b. K=3/4 子问题改写增强（两遍 vLLM，1 卡）
cd ../../../traces
NUM_VARIANTS=2 ./run_enhance_decompose_train.sh      # → train_nl_enhance.jsonl（5,562 条，4,132 条通过校验）

# 1c. 2Wiki GPT 分解标注（需 OpenAI API，gpt-5-mini，64 并发）
cd ../decompose/bart
./run_annotate_2wiki.sh                   # train：10,500 选中 → 10,444 成功
PRESET=mixed_dev ./run_annotate_2wiki.sh  # dev：1,043 条

# 1d. 训练 BART 分解器（1 卡，3 epoch）
./run_train.sh                            # → outputs/bart_decomposer_musique_2wiki_repro/checkpoint-1425
# 只用 MuSiQue 的对照模型：TRAIN_MUSIQUE_ONLY=1 OUT_DIR=outputs/bart_decomposer_musique_only ./run_train.sh

# 1e. 三数据集 dev 推理 → v1 格式 → 自然语言子问题
./run_predict_musique_2wiki_hotpot.sh     # BART beam search，几分钟
./run_convert_predictions_to_v1.sh        # CPU，秒级
./run_decompose_to_nl_all.sh              # Llama 改写：实测 MuSiQue ~36min / 2Wiki ~3.2h / HotpotQA ~1.6h（单卡）
# 产物同步到 data/decompose/{dataset}/bart/dev_nl.jsonl
```

**注意 1e 的 MuSiQue 输入**：`predict.py` 的 MuSiQue 分支读的是 BART 专用格式
（`decompose/bart/data/musique_raw/musique_ans_gold_context_version_dev.jsonl`，含
`composed_question_text` 字段），**不是** `data/raw/musique/musique_ans_v1.0_dev.jsonl`。
仓库里的 `run_predict_musique_2wiki_hotpot.sh` 写的是后者，是过期路径。

### 第 2 步 trace 构造（1 卡，BGE 检索负例）

```bash
cd traces
./run_construct_musique_train.sh
# 等价：python construct_balanced_traces.py --split train \
#   --decompose-file ../data/decompose/musique/gt/train_nl.jsonl \
#   --enhance-decompose-file ../data/decompose/musique/gt/train_nl_enhance.jsonl \
#   --cos-topk 10 --max-wrong-per-hop 3 --balance-ratio 1.0
python construct_balanced_traces.py --split dev   # dev 同样跑一遍
# → traces/{gold,counterfactual,merged}/musique/{train,dev}.jsonl（merged：83,927 / 8,821）
```

### 第 3 步 隐状态提取（1 卡，最耗时）

```bash
cd hidden_states
# 多层版（gate v2/v3 需要 15/23 层；一次 forward 读出多层）
python extract_hidden_states_multilayer_pilot.py --split train --layers 7,15,23,31
python extract_hidden_states_multilayer_pilot.py --split dev   --layers 7,15,23,31
# → hidden_states/pilot_multilayer/（92,748 个 npz，约 12GB）
```

实测参考：0928 的多错数据（16,296 条 trace、约 5.5 万个前缀、两层）在一张被共享的卡上约 **40 分钟**
（6.4 trace/s）；全量 train 的四层提取按比例是数小时量级。峰值显存实测 **31.9 GiB 已分配 /
37.4 GiB 保留**（含 15 GiB 权重，`--max-tokens-per-batch 32768`）。

### 第 4 步 gate 训练（纯 CPU，几分钟）

```bash
cd gate/gate_v3
python fit_lr_gate_pooled_v3.py \
  --hs-root ../../hidden_states/pilot_multilayer \
  --per-j-layers 0:15,1:15,2:23,3:23 \
  --pca-dim-h 64 --pca-dim-delta 64 --C 1.0 --target-fpr 0.15 --max-j 3
# → gate/gate_v3/artifacts_pooled_v3/{j0..j3}.joblib + meta.json，results_pooled_v3/ 下是 dev 指标
```

对照变体：`gate/fit_lr_gate_pooled.py`（v1/v2，Δ only）、
`gate/gate_h_only/fit_lr_gate_pooled_h_only.py`（只用 h）。

### 第 5 步 端到端评测（1 卡，vLLM）

```bash
conda activate mlsys && cd retrieval
# distractor setting（主结果），三个数据集分别跑
python run_retrieval_exp_wavefront_gate_v3_rawprefix.py \
  --dataset musique --decompose-mode bart_decompose \
  --decompose-file ../data/decompose/musique/bart/dev_nl.jsonl \
  --artifacts-dir ../gate/gate_v3/artifacts_pooled_v3 \
  --lambda-gate 0.60 --beam-width 1 --retrieve-k 10 --retriever cosine \
  --final-reader-prompt comparison_hint --short-answer-prompt default \
  --limit 0 --gpu-memory-utilization 0.3 --max-model-len 8192 --gate-device cuda:0 \
  --run-tag <tag>
```

实测耗时（空闲卡）：MuSiQue **0.39h** / HotpotQA **1.4h** / 2Wiki **3.6h**。
`--limit` 默认 20，跑全量必须显式写 `--limit 0`。

### 第 6 步 global setting（可选）

```bash
cd retrieval
python build_global_corpus.py --dataset musique --split dev     # 实测 52s（2Wiki 151s / HotpotQA 185s）
# 然后在第 5 步命令上加：
#   --retrieval-scope global --corpus-dir global_corpus/musique_dev
# top50 版本再把 --retrieve-k 改成 50（耗时变 2.4–4.4 倍）
```

### 第 7 步 分析脚本（CPU 为主）

```bash
cd retrieval
python compare_decomposer_ablation.py --tag-suffix <实验组 run tag 后缀>   # 分解器消融对比表
python compute_evidence_precision.py --d-runs musique=<run> 2wiki=<run> hotpot=<run>  # 三方 precision
python analyze_global_gold_rank.py                                        # gold 在 BGE 排序中的位置（需 1 卡做 query 编码）

cd ../error_propagation_probe
python build_multi_error_traces.py --split dev --exhaustive-for-k 3 4     # n_wrong=0..K 全套变体
python extract_full_hidden_states_multilayer.py --split dev \
  --trace-file data/musique/dev_multi_error.jsonl \
  --out-dir hidden_states_multilayer/dev --layers 15,23 \
  --reserve-gb 40 --empty-cache-every 0 --resume                          # 实测约 40min
python score_gate_v3_vs_nwrong.py --hidden-dir hidden_states_multilayer/dev  # gate 能力剖析四张表
```

### 第 8 步 两个对比工作（各自独立环境）

```bash
# ChainRAG：先起本地 vLLM server（OpenAI 兼容），再跑检索
cd chainrag && conda activate mlsys
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"   # 否则 libstdc++ 版本冲突
GPU_MEMORY_UTILIZATION=0.2 ./run_local_vllm_server.sh         # 终端 A
conda activate chainrag && python retrieval.py --dataset musique   # 终端 B
python merge_shards.py && python evaluate.py

# GRITHopper
cd ../grithopper && conda activate grithopper
python run_retrieval.py --dataset musique --num-shards 4 --shard-idx 0   # 四分片并行
python run_answer_gen.py --dataset musique
python merge_and_score.py --dataset musique --num-shards 4
```

## 附二：写论文时需要注意的几处

1. **「无 gate 基线」只认 λ=0 对照臂**（`20261003_070503/070301/070253_nogate_*_lambda0_dev`）：同一个
   脚本、同一套 prompt 与 BGE 前缀，唯一变量是 `λ`。**不要再用 0810 的 A 组**——它少了 BGE 前缀、多了
   一步 LLM selection、reader prompt 也不同，三个变量混在一起，会系统性低估 gate 的贡献
   （它给出的 EM 提升 +5.50/+4.46/+2.99，真实值是 +11.21/+8.02/+9.11）。0713/0719 两份周报对
   `0.6513` 的标注互相矛盾，一律不引用。
2. **gate 的贡献与 reader prompt 的贡献必须分开报**。[3.4 节](#34-reader-prompt-效果)的三级阶梯给出了
   同口径的拆分：prompt 贡献 +0.99 / +9.13 / +2.84 EM，gate 贡献 +11.21 / +8.02 / +9.11 EM。
   **2Wiki 上 prompt 的贡献比 gate 还大**，不拆开报会让读者把那 9 点算到 gate 头上。
3. **chain@1 是这个方法最有说服力的指标**（distractor 下提升 +15.2 到 +19.3 点，远大于单跳 recall@1
   的 +8.8 到 +9.7），因为 gate 的价值正是「把整条链里某一跳的错纠回来」，逐跳的改善在链级累积。
4. **0928 的能力剖析给了方法一个机制层面的解释**：gate 实际是前缀污染检测器，这同时解释了它为什么
   有效（链级杠杆）和它的副作用（错误放大、无法定位）。h-only 几乎等于 v3 这个端到端观察
   （[3.1 节](#31-主结果三数据集三信号对照abcd) B/C/D 对照）与 fusion 拆解
   （[3.10 节](#310-gate-能力剖析0928)）互为印证，可以合并成一条论证线。
5. **检索指标提升不保证答案指标提升**，这个现象在本项目出现过至少三次（gate v2 端到端、
   probe + cosine、context gate）。
6. **离线指标好 ≠ 端到端好**，出现过至少四次（pairwise MLP、MLP gate、gate v4、context gate）。
   反过来，gate v3 是唯一一次「离线赢干净兑现成端到端赢」。
7. **GT 分解与 BART 分解的数字绝对不能混在一张表里**。GT 分解下 MuSiQue recall@1 能到 0.78–0.84，
   BART 只有 0.70——这个差距是分解器质量造成的，与 gate 无关。论文里凡是写「端到端」的表都应标明
   用的是哪一套分解。
8. **`汇报报告.md` 写 BART 学习率是 `1e-5`，但 `train_meta.json` 和 `TRAINING_NOTES.md` 都是 `3e-5`**。
   以 `train_meta.json` 为准（它是训练脚本自动写出的）。
9. **HotpotQA 从头到尾没有任何训练数据参与**（分解器没有它的标注、gate 只用 MuSiQue trace 训练），
   所以它上面的结果是最干净的跨数据集泛化证据，论文里值得专门点出来。
10. **三方对比的 EM/F1 列不是同 prompt 口径**。GRITHopper 没有答案生成步骤，其 reader 由本项目实现并
   刻意写入与 D 组等价的比较类提示（偏保守的选择）；ChainRAG 自带 reader，按其原始实现评测，**不含
   CoT 也不含比较类提示**。用 3.4 节测得的 prompt 单独贡献做最保守校正后，D 组仍领先 ChainRAG
   +6.75 / +15.47 / +6.33 EM，所以结论不依赖这个差异——但论文里要把它写明。检索侧的列不受影响。
11. **「扩大候选池」的收益必须和 gate 绑在一起说**。[3.7 节](#37-global-setting)显示 λ=0 时 k 从 10 扩到
   50 的 recall@1 收益为零（MuSiQue / HotpotQA 逐位相同，2Wiki −0.01），λ=0.60 时才有 +0.17~+1.97。所以 gate 是
   利用更大候选池的**必要条件**，不要写成「更大的候选池带来提升」。
12. **逐跳指标的分母是「对齐上的跳」，chain@1 是宽松版**。MuSiQue 有 5.73% 的 gold 跳（4 跳题的
    50.1%）因为分解器压短而不进 recall 的分母；chain@1 用 `zip` 比对，HotpotQA 的严格版比宽松版低
    **6.36 点**（0.4974 vs 0.5610）。第 4 章定义指标时必须写清楚。
13. **GRITHopper 的 reader 是本项目实现的**，脚本 `grithopper/run_answer_gen.py` 的
    `READER_SYSTEM_PROMPT` 明确含比较类提示（"If the question is a comparison … answer with the
    entity/item being asked about — not a raw date or number"）。这是刻意给对手的优待，要写明，
    否则读者会以为三方 reader 完全一致。
14. **λ=0 的 run 比 λ=0.60 慢 3–4 倍是 GPU 争抢，不是方法开销**（如 MuSiQue global k=10：1.22h vs
    0.33h）。按代码两者计算量相同——λ=0 时 gate 照样前向打分，只是分数乘 0。**论文里谈开销只用
    「每跳的前向次数」，不要引用任何 wall-clock 数字。**

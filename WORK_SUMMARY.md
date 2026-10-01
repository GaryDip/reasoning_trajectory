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

| 版本 | 目录 | 特征 | 层 | dev AUC | TPR | FPR | F1 | 端到端评过 | 结论 |
|---|---|---|---|---|---|---|---|---|---|
| v1（最初生产） | `gate/artifacts_pooled/` | Δ_j | 全部 31 | 0.9074 | 0.8235 | 0.1756 | 0.7616 | 是 | 起点 |
| v2 | `gate/artifacts_pooled_v2/` | Δ_j | 15/15/23/23 | 0.9221 | 0.8456 | 0.1693 | 0.7785 | 是 | 层选择有效 |
| h-only | `gate/gate_h_only/artifacts_pooled_h_only/` | h_after | 15/15/23/23 | 0.9339 | 0.8760 | 0.1714 | 0.7939 | 是（C 组） | h 比 Δ 更强 |
| **v3（当前主力）** | `gate/gate_v3/artifacts_pooled_v3/` | concat(PCA(h_after), PCA(Δ_j))，两路独立 PCA | 15/15/23/23 | **0.9357** | 0.8838 | 0.1788 | 0.7930 | 是（D 组） | 默认方案 |
| v4（双线性） | `gate/gate_v4/artifacts_pooled_v4/` | `(W_pre h_{j-1})·(W_post h_j) + w_Δ^T Δ_j` 低秩双线性 | 15/15/23/23 | 0.9457 | 0.9521 | 0.2847 | 0.7611 | **否** | 离线 AUC 最高但 FPR 跑到 0.28，未接端到端 |

v3 的四个 j 的训练规模与训练集 AUC：j=0 n=83,927（正例 28.7%，AUC 0.9707）、j=1 n=59,857（40.2%，
0.9472）、j=2 n=21,411（45.3%，0.9143）、j=3 n=4,046（50.0%，0.9535）。**j=3 只有 4,046 条样本**，
是后段跳偏弱的直接原因之一。

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

**做法**：新写多层提取脚本（一次 forward 读出 7/15/23/31 四层），在 pilot 子采样上扫层、扫分类器。

**发现**：
1. 中间层（15、23）整体 AUC 比 31 高 1–1.5 点。
2. 不同层擅长不同位置：15 在 j=0/1 更强，23 在 j=2/3 更强。
3. 「维度多了就好」被排除：`concat(15+15)` 与单层 15 几乎相同（0.9152 vs 0.9151）。
4. 把 31 混进 concat 反而拖累（`concat(15+23+31)` 掉回 0.901）。推测最后一层被「预测下一个 token」
   的训练目标塑形过度，压缩掉了部分「这段是否自洽」的抽象语义。
5. 浅层 MLP 的 AUC 比 LR 更高，**但阈值校准跑偏**：target FPR=0.15 下实测 dev FPR=0.2348。

**决策**：采用 per-j 混层（15/15/23/23）+ 保持 LR，不引入 MLP。全量验证全面提升且 FPR 更低
（0.1693 vs 0.1756），产出 `gate/artifacts_pooled_v2/`。

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

**离线**：AUC 0.9221→0.9357，TPR 0.8456→0.8838；j=2（v2 最弱的一跳）F1 涨幅最大（0.7535→0.7919）。

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

**结果**（三个数据集，只切 prompt，检索与 gate 完全不动）：

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

**结果**：三个数据集全线下降，chain@1 掉得最多（2Wiki −28 点）。完整表见 [3.7 节](#37-global-setting)。

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
没有被更大的候选池带偏；代价是 2.4–4.4 倍耗时。完整表见 [3.7 节](#37-global-setting)。

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
   dev 数据上仍然成立。错误跳 79%–89% 超阈值。
2. **它是「前缀被污染了没有」的检测器，不是「哪一跳错了」的定位器**。错误之前的跳完全不受影响
   （0.125–0.164，与全对基线 0.169 持平），错误之后本身选对的跳被抬高到 2.3–2.7 倍
   （0.389–0.457），且基本不衰减。固定跳的位置做对照后效应仍然成立（K=4 第 4 跳：前缀全对 0.209
   vs 前缀有错 0.405）。
3. **fusion 两半里，判别力几乎全部来自 h**。logit 可精确拆为
   `截距 + w_h·PCA(h) + w_Δ·PCA(Δ)`，按 j 分开算对错分离度（Cohen's d）：h 为 1.43–2.20，
   Δ 为 0.07–0.47。这与 2.9 节「C 组几乎全面超过 B 组」在端到端层面的观察互相印证。准确表述是
   「在已有 h 的情况下 Δ 没有提供额外判别力」，不是 Δ 本身没信息——两半特征并不独立
   （`Δ_j = h_j − h_{j−1}` 与 `h_j` 共用一项）。

**由此统一解释了三个原本看起来独立的现象**：错误之后正确跳被抬高（h 必然的行为）、错误越多检出
强度不升反降（它不在定位哪一跳错）、单跳注入的训练数据也能学好（它学的是更容易的「前缀正常 vs
不正常」）。

**代价**：一旦某跳选错，后面本身正确的候选有 43%–50% 被判成异常（全对基线 17%）。生产是贪心逐跳
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
  离线 AUC 0.9457（高于 v3 的 0.9357）、TPR 0.9521，但 **FPR 0.2847**（v3 是 0.1788）、F1 0.7611
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

同一个 run 两个口径都会算，所以 MuSiQue 的 recall@1 有 0.6994（按位置）和 0.8552（顺序无关）两个值，
**引用时必须说明是哪一个**。

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

**配置**：rawprefix、λ=0.50、BGE 检索前缀、真实 recall@1/@3（剪枝前完整候选池）、**reader prompt =
default**。四组唯一变量是用哪个 gate 信号。来源：0810 周报。

| 数据集 | 组别 | recall@1 | recall@3 | chain@1 | EM | F1 | run 目录 |
|---|---|---|---|---|---|---|---|
| MuSiQue | A 无 gate | 0.6003 | 0.7611 | 0.3335 | 0.3757 | 0.4652 | `20260806_090938_baseline_only_musique_hotpot/musique` |
| | B 仅 Δ（v2） | 0.6846 | 0.8435 | 0.4775 | 0.4150 | 0.5105 | `20260806_145434_gate_v2_rawprefix_musique_bw1` |
| | C 仅 h | 0.6977 | 0.8433 | 0.5077 | 0.4245 | 0.5216 | `20260807_060129_gate_h_only_rawprefix_musique_bw1` |
| | **D 融合（v3）** | **0.7023** | **0.8446** | **0.5143** | **0.4307** | **0.5270** | `20260806_132739_gate_v3_rawprefix_musique_bw1_fullpool` |
| 2Wiki | A 无 gate | 0.8123 | 0.9021 | 0.6047 | 0.4544 | 0.5230 | `20260806_091034_baseline_only_2wiki/2wiki` |
| | B 仅 Δ | 0.9119 | 0.9813 | 0.8255 | 0.4912 | 0.5661 | `20260806_152220_gate_v2_rawprefix_2wiki_bw1` |
| | C 仅 h | 0.9244 | **0.9839** | 0.8515 | 0.4967 | 0.5733 | `20260807_062755_gate_h_only_rawprefix_2wiki_bw1` |
| | **D 融合** | **0.9289** | 0.9835 | **0.8613** | **0.4990** | **0.5759** | `20260806_135739_gate_v3_rawprefix_2wiki_bw1` |
| HotpotQA | A 无 gate | 0.5746 | 0.7760 | 0.3361 | 0.4918 | 0.6163 | `20260806_090938_baseline_only_musique_hotpot/hotpot` |
| | B 仅 Δ | 0.6557 | **0.8791** | 0.5350 | 0.5175 | 0.6462 | `20260806_172829_gate_v2_rawprefix_hotpot_bw1` |
| | C 仅 h | 0.6573 | 0.8671 | 0.5449 | 0.5183 | 0.6478 | `20260807_082919_gate_h_only_rawprefix_hotpot_bw1` |
| | **D 融合** | **0.6596** | 0.8685 | **0.5568** | **0.5217** | **0.6517** | `20260806_160848_gate_v3_rawprefix_hotpot_bw1` |

## 3.2 gate 单独贡献（同 prompt 口径）

A 组与 D 组的 reader prompt 相同（default），是 gate 单独贡献的干净对比：

| 数据集 | recall@1 | recall@3 | chain@1 | EM | F1 |
|---|---|---|---|---|---|
| MuSiQue | +10.20 | +8.35 | +18.08 | +5.50 | +6.18 |
| 2Wiki | +11.66 | +8.14 | +25.66 | +4.46 | +5.29 |
| HotpotQA | +8.50 | +9.25 | +22.07 | +2.99 | +3.54 |

（单位：百分点。）**chain@1 的提升量级（+18 到 +26）远大于单跳 recall@1**，因为整条链里任一跳被
纠回来都可能让链从错变对，逐跳的改善在链级累积。

加上 reader prompt 修复之后的最终版本（λ=0.60 + comparison_hint）相对 A 组：
MuSiQue EM +6.29、2Wiki EM +14.79、HotpotQA EM +6.01——**其中 2Wiki 有约 10 点来自 prompt，
不是 gate**。

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

### 3.3.2 train 子集扫描（0831，当前完整方法口径）

**配置**：MuSiQue **train** 3,000 条子集、BART decompose、rawprefix、comparison_hint。

| λ | recall@1 | chain@1 | EM | F1 | run 目录 |
|---|---|---|---|---|---|
| 0.10 | 0.7252 | 0.4710 | 0.4193 | 0.5005 | `20260827_142454_..._lambda0p10` |
| 0.20 | 0.7515 | 0.5133 | 0.4413 | 0.5271 | `20260827_153321_..._lambda0p20` |
| 0.30 | 0.7555 | 0.5183 | 0.4527 | 0.5414 | `20260828_042244_..._lambda0p30` |
| 0.40 | 0.7561 | 0.5217 | 0.4543 | 0.5426 | `20260828_053120_..._lambda0p40` |
| 0.50 | 0.7558 | 0.5223 | 0.4540 | 0.5425 | `20260828_063956_..._lambda0p50` |
| **0.60** | 0.7556 | 0.5250 | **0.4593** | 0.5460 | `20260828_074835_..._lambda0p60` |
| **0.65** | 0.7540 | 0.5220 | 0.4587 | **0.5461** | `20260828_094100_..._lambda0p65` |
| 0.70 | 0.7523 | 0.5210 | 0.4580 | 0.5453 | `20260828_101546_..._lambda0p70` |
| 0.80 | 0.7500 | 0.5163 | 0.4567 | 0.5437 | `20260828_112529_..._lambda0p80` |

（目录前缀统一为 `gate_v3_rawprefix_musique_trainsubset_big_bart_`。）

另有 0808 在 600 条 train 子集上扫过 0.10–0.60（目录前缀
`gate_v3_rawprefix_musique_trainsubset_lambda*`），**但那一批用的是 GT 分解**，所以 recall@1 在
0.84 量级、EM 在 0.47–0.49 量级，与上表（BART 分解）不可比；它的峰值落在 λ≈0.25–0.45。

### 3.3.3 λ=0.60 的 dev 全量验证（最终主结果）

**配置**：rawprefix、comparison_hint、λ=0.60。**这是当前方法的最终版本，也是 0914/0921/0928 一切
对比的基准（「D 组」）**。

| 数据集 | recall@1 | recall@3 | chain@1 | EM | F1 | run 目录 |
|---|---|---|---|---|---|---|
| MuSiQue | 0.6994 | 0.8435 | 0.5118 | 0.4386 | 0.5341 | `20260828_121002_gate_v3_rawprefix_musique_lambda060_dev` |
| 2Wiki | 0.9316 | 0.9842 | 0.8685 | 0.6023 | 0.6819 | `20260828_123331_gate_v3_rawprefix_2wiki_lambda060_dev` |
| HotpotQA | 0.6620 | 0.8701 | 0.5610 | 0.5519 | 0.6883 | `20260828_142420_gate_v3_rawprefix_hotpot_lambda060_dev` |

**同配置重跑的噪声水平**（用于判断多大的差异算真实）：MuSiQue 各项 ≤0.2 点，2Wiki ≤0.06 点，
HotpotQA 逐位一致。重跑目录：`20260910_152228/163521/154534_..._rankedpool`（这三个还额外记录了
每跳重排后的完整候选池 `hop_ranked_para_ids`）。

## 3.4 reader prompt 效果

**配置**：λ=0.50、rawprefix，只切 `--final-reader-prompt`。来源：0817 周报。

| 数据集 | recall@1 | chain@1 | EM（default → hint） | F1（default → hint） | hint 的 run 目录 |
|---|---|---|---|---|---|
| MuSiQue | 持平 | 持平 | 0.4307 → 0.4427（+1.20） | 0.5270 → 0.5386（+1.16） | `20260813_161210_gate_v3_rawprefix_musique_comparisonhint_full` |
| 2Wiki | 持平 | 持平 | 0.4990 → **0.5982**（+9.92） | 0.5759 → **0.6770**（+10.11） | `20260813_072509_gate_v3_rawprefix_2wiki_comparisonhint_full` |
| HotpotQA | 持平 | 持平 | 0.5217 → 0.5517（+3.00） | 0.6517 → 0.6870（+3.53） | `20260813_163744_gate_v3_rawprefix_hotpot_comparisonhint_full` |

未采纳的变体（`bothprompts` = comparison_hint + short_answer type_match）：
`20260814_083308/091013/122306_..._bothprompts_full`。

## 3.5 ColBERT 检索器对照

**配置**：gate v3、rawprefix、comparison_hint；「加 gate」用 λ=0.50（沿用 BGE 的值，未单独扫），
「无 gate」用 λ=0.0。ColBERT 分支额外做候选池内 min-max 归一化。

| 数据集 | 方案 | recall@1 | recall@3 | chain@1 | EM | F1 | run 目录 |
|---|---|---|---|---|---|---|---|
| MuSiQue | ColBERT 无 gate | 0.6175 | 0.7577 | 0.3732 | 0.3322 | 0.4163 | `20260818_154358_colbert_musique_nogate_full` |
| | ColBERT + gate | 0.6581 | 0.7987 | 0.4348 | 0.3839 | 0.4734 | `20260818_162450_colbert_musique_gate_full` |
| | BGE + gate | **0.7018** | **0.8441** | **0.5143** | **0.4427** | **0.5386** | （3.4 节 hint 那行） |
| 2Wiki | ColBERT 无 gate | 0.8413 | 0.9167 | 0.6802 | 0.5223 | 0.5881 | `20260818_170355_colbert_2wiki_nogate_full` |
| | ColBERT + gate | 0.8742 | 0.9600 | 0.7454 | 0.5496 | 0.6192 | `20260818_210614_colbert_2wiki_gate_full` |
| | BGE + gate | **0.9289** | **0.9839** | **0.8613** | **0.5982** | **0.6770** | — |
| HotpotQA | ColBERT 无 gate | 0.5606 | 0.7468 | 0.3685 | 0.4623 | 0.5817 | `20260819_003311_colbert_hotpot_nogate_full` |
| | ColBERT + gate | 0.6027 | 0.8125 | 0.4498 | 0.5136 | 0.6408 | `20260819_012023_colbert_hotpot_gate_full` |
| | BGE + gate | **0.6595** | **0.8683** | **0.5567** | **0.5517** | **0.6870** | — |

## 3.6 三方对比 vs ChainRAG / GRITHopper

**口径**：recall@1/@3 为**顺序无关**（三方唯一可共用的定义）；覆盖率 = 每题最终用到的证据集合与
gold 集合的交集比例；EM/F1 为 HotpotQA 官方口径，三方最终答案都由同一个本地
Llama-3.1-8B-Instruct 生成。来源：0831 周报第 2.4 节。

| 数据集 | 方法 | recall@1 | recall@3 | gold 覆盖率 | 全覆盖率 | EM | F1 |
|---|---|---|---|---|---|---|---|
| MuSiQue | D 组（λ=0.60） | 0.8552 | 0.9622 | 0.7733 | 0.5283 | **0.4377** | **0.5324** |
| | ChainRAG | 0.6716 | 0.8628 | **0.9587** | **0.9048** | 0.2764 | 0.3847 |
| | GRITHopper | **0.9058** | **0.9863** | 0.8207 | 0.6090 | 0.2793 | 0.3734 |
| 2Wiki | D 组 | 0.9723 | 0.9970 | 0.9407 | 0.8799 | **0.6023** | **0.6819** |
| | ChainRAG | 0.9199 | 0.9785 | **0.9870** | **0.9696** | 0.4426 | 0.5220 |
| | GRITHopper | **0.9500** | **0.9964** | 0.8922 | 0.7751 | 0.5390 | 0.6267 |
| HotpotQA | D 组 | 0.8988 | 0.9805 | 0.8532 | 0.7317 | **0.5519** | **0.6883** |
| | ChainRAG | 0.8032 | 0.9496 | **0.9864** | **0.9741** | 0.4598 | 0.6027 |
| | GRITHopper | **0.9658** | **0.9959** | 0.9464 | 0.9068 | 0.5546 | 0.6926 |

三方输出数据：`chainrag/processed_data/{musique,2wikimqa,hotpotqa}/results.jsonl`、
`grithopper/results/{musique,2wiki,hotpot}/{retrieval,answers}.jsonl`。

**三方检索投入对比**（解释 recall 排名）：ChainRAG 零训练（现成 embedding + cross-encoder）；
D 组零训练底座（BGE）+ 一个小 gate；GRITHopper 微调了 7B 级模型，训练数据直接含这三个数据集。
recall 排名基本与投入成正比，但**答案效果排名与之相反**。

## 3.7 global setting

**配置**：gate v3、rawprefix、λ=0.60。语料库为该数据集 dev split 全部段落按 `(title, 正文)` 去重
合并（MuSiQue 21,100 / 2Wiki 56,687 / HotpotQA 66,635 篇）。

| 数据集 | 条件 | reader prompt | k | recall@1 | recall@3 | chain@1 | EM | F1 | 耗时 | run 目录 |
|---|---|---|---|---|---|---|---|---|---|---|
| MuSiQue | distractor | hint | 10 | 0.6994 | 0.8435 | 0.5118 | 0.4386 | 0.5341 | — | （3.3.3 节） |
| | global | default | 10 | 0.5357 | 0.6329 | 0.2991 | 0.2884 | 0.3746 | 0.36h | `20260904_163127_global_musique_full` |
| | global | hint | 10 | 0.5357 | 0.6329 | 0.2991 | 0.2987 | 0.3858 | 0.60h | `20260916_155625_global_musique_k10_comparisonhint` |
| | global | hint | **50** | 0.5438 | 0.6616 | 0.3136 | 0.3136 | 0.4022 | 2.6h | `20260916_155806_global_musique_k50_comparisonhint` |
| 2Wiki | distractor | hint | 10 | 0.9316 | 0.9842 | 0.8685 | 0.6023 | 0.6819 | — | （3.3.3 节） |
| | global | default | 10 | 0.7879 | 0.8487 | 0.5873 | 0.4222 | 0.4841 | 3.7h | `20260905_051009_global_2wiki_full` |
| | global | hint | 10 | 0.7881 | 0.8487 | 0.5879 | 0.5130 | 0.5765 | 3.6h | `20260916_175557_global_2wiki_k10_comparisonhint` |
| | global | hint | **50** | 0.7905 | 0.8561 | 0.5939 | 0.5193 | 0.5842 | 8.8h | `20260916_225214_global_2wiki_k50_comparisonhint` |
| HotpotQA | distractor | hint | 10 | 0.6620 | 0.8701 | 0.5610 | 0.5519 | 0.6883 | — | （3.3.3 节） |
| | global | default | 10 | 0.5326 | 0.6141 | 0.3726 | 0.4105 | 0.5193 | 1.35h | `20260905_051556_global_hotpot_full` |
| | global | hint | 10 | 0.5325 | 0.6142 | 0.3723 | 0.4362 | 0.5515 | 1.4h | `20260916_163223_global_hotpot_k10_comparisonhint` |
| | global | hint | **50** | 0.5523 | 0.6517 | 0.4050 | 0.4593 | 0.5781 | 4.3h | `20260916_183527_global_hotpot_k50_comparisonhint` |

**注意**：0907 周报里的 global 行用的是 default prompt，与 distractor 行（hint）的 EM/F1 口径不一致。
统一到 hint 之后，global 与 distractor 的真实 EM 差距是 MuSiQue −14.0、2Wiki −8.9、HotpotQA −11.6
（k=10）或 −12.5 / −8.3 / −9.3（k=50）。检索指标不受 prompt 影响。

**top10 → top50 的增减分解**（0921 第 5.2 节，按 BGE 名次分档）：

| 数据集 | gold 在 BGE 11–50 档被 gate 排到第 1（占全部跳） | BGE 前 10 档中被新候选挤掉（原排第 1 的比例变化） | recall@1 净变化 |
|---|---|---|---|
| MuSiQue | +2.2% | 78.9% → 76.7% | +0.8 |
| 2Wiki | +0.6% | 92.4% → 92.0% | +0.2 |
| HotpotQA | +2.5% | 82.9% → 81.8% | +2.0 |

分析脚本与结果：`retrieval/analyze_global_gold_rank.py`、
`retrieval/results/global_gold_rank_analysis.json`。

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
| MuSiQue | 0.7069（+0.8） | 0.8474（+0.4） | 0.5275（+1.6） | 0.4419（+0.3） | 0.5363（+0.2） | `20260910_205233_..._musiqueonly_decomposer` |
| 2Wiki | 0.6907（**−24.1**） | 0.8689（−11.5） | 0.5651（**−30.3**） | 0.5189（**−8.3**） | 0.5976（**−8.4**） | `20260910_211502_..._musiqueonly_decomposer` |
| HotpotQA | 0.5825（−8.0） | 0.8290（−4.1） | 0.4704（−9.1） | 0.5423（−1.0） | 0.6792（−0.9） | `20260910_223856_..._musiqueonly_decomposer` |

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
那三个 `rankedpool` 重跑。

### 证据集合 precision

| 数据集 | 方法 | precision | 覆盖率 | 集合 F1 | 平均检索篇数 | 平均 gold 篇数 |
|---|---|---|---|---|---|---|
| MuSiQue | D 组 | 0.8837 | 0.7958 | 0.8275 | 2.28 | 2.60 |
| | ChainRAG | 0.2068 | **0.9587** | 0.3235 | 14.54 | |
| | GRITHopper | **0.9091** | 0.8207 | **0.8517** | 2.31 | |
| 2Wiki | D 组 | **0.9673** | 0.9408 | **0.9496** | 2.37 | 2.44 |
| | ChainRAG | 0.3559 | **0.9870** | 0.4998 | 7.61 | |
| | GRITHopper | 0.9540 | 0.8922 | 0.9139 | 2.24 | |
| HotpotQA | D 组 | 0.8955 | 0.8532 | 0.8650 | 1.92 | 2.00 |
| | ChainRAG | 0.2752 | **0.9864** | 0.4215 | 7.86 | |
| | GRITHopper | **0.9658** | **0.9464** | **0.9528** | 1.95 | |

### 逐跳 precision@3

| 数据集 | 方法 | 跳数 | recall@1 | recall@3 | **precision@3** | 前 3 名平均 gold 篇数 |
|---|---|---|---|---|---|---|
| MuSiQue | D 组 | 6,037 | 0.8836 | 0.9742 | 0.5126 | 1.54 |
| | ChainRAG | 5,162 | 0.6716 | 0.8628 | 0.3949 | 1.18 |
| | GRITHopper | 6,404 | **0.9058** | **0.9863** | **0.6087** | 1.83 |
| 2Wiki | D 组 | 30,621 | **0.9724** | **0.9972** | 0.6158 | 1.85 |
| | ChainRAG | 22,737 | 0.9199 | 0.9785 | 0.5859 | 1.76 |
| | GRITHopper | 30,654 | 0.9500 | 0.9964 | **0.6786** | 2.04 |
| HotpotQA | D 组 | 14,796 | 0.8988 | 0.9805 | 0.5035 | 1.51 |
| | ChainRAG | 13,658 | 0.8032 | 0.9496 | 0.4679 | 1.40 |
| | GRITHopper | 14,810 | **0.9658** | **0.9959** | **0.5994** | 1.80 |

**两个口径注意点**：(1) precision@3 有上限——一题只有 2 篇 gold 时最高 2/3，HotpotQA 全是 2 篇，
所以跨数据集不能比绝对值。(2) MuSiQue 上按标题匹配会让 D 组虚高（recall@1 0.8836 vs 按段落编号
的 0.8552），因为约 6.7% 的题 D 组选中了与 gold 同名但不是 gold 的段落；2Wiki / HotpotQA 两种匹配
方式结果一致。

脚本与结果：`retrieval/compute_evidence_precision.py`、
`retrieval/results/evidence_precision_3way_rankedpool.json`。

## 3.10 gate 能力剖析（0928）

**数据**：MuSiQue dev 多错 trace，2,417 题 / 16,316 条 trace / 51,672 跳，`n_wrong = 0..K`。
**打分**：gate v3，与生产逐字一致。**自检**：51,672 跳全部打分成功、无缺层。

### 分数 vs 前缀里的错误数

| 前缀中错误数 | 当前跳是错的 | 当前跳是对的 |
|---|---|---|
| 0 | — | 0.169±0.226（14,537）/ 超阈值 0.17 |
| 1 | 0.766±0.302（13,899）/ 0.85 | 0.407±0.334（8,109）/ 0.43 |
| 2 | 0.708±0.304（8,747）/ 0.79 | 0.461±0.334（2,785）/ 0.50 |
| 3 | 0.703±0.299（2,785）/ 0.80 | 0.556±0.351（405）/ 0.66 |
| 4 | 0.766±0.274（405）/ 0.89 | — |

### 相对第一个错误跳的位置

| 位置 | 全部 | 其中本身选对的 |
|---|---|---|
| 错误前第 3 / 2 / 1 跳 | 0.125 / 0.135 / 0.164 | 同左（都是选对的） |
| 错误发生的那一跳 | **0.766** | —（构造上为空） |
| 错误后第 1 / 2 / 3 跳 | 0.591 / 0.542 / 0.569 | **0.457 / 0.395 / 0.389** |
| 全 gold trace（对照） | 0.186 | — |

### 控制跳的位置

| K | 跳 | 本跳错 | 本跳对、前缀全对 | 本跳对、前缀有错 |
|---|---|---|---|---|
| 2 | 2 | 0.759 | 0.171 | **0.506** |
| 3 | 3 | 0.678 | 0.217 | **0.356** |
| 4 | 3 | 0.671 | 0.303 | **0.419** |
| 4 | 4 | 0.744 | 0.209 | **0.405** |

### trace 级别

| K | n_wrong=0 均值 / 最大值 | n_wrong=K 均值 / 最大值 |
|---|---|---|
| 2 | 0.171 / 0.275 | 0.770 / 0.910 |
| 3 | 0.191 / 0.379 | 0.739 / 0.940 |
| 4 | 0.203 / 0.495 | 0.733 / 0.959 |

Spearman(n_wrong, 分数均值) = 0.606，Spearman(n_wrong, 分数最大值) = 0.508。
**全 gold trace 的最大分随 K 上升**（0.275 → 0.379 → 0.495），所以用「路径最高分」做判别时，
跳数越多区分度越窄。

### fusion 两半的贡献

| 类别 | 跳数 | 分数 | logit(h) | logit(Δ) |
|---|---|---|---|---|
| 正确跳，前缀全对 | 14,537 | 0.169 | **−1.62** | −0.31 |
| 正确跳，前缀有 1 错 | 8,109 | 0.407 | **−0.08** | −0.45 |
| 正确跳，前缀有 2+ 错 | 3,190 | 0.474 | **−0.14** | −0.31 |
| 错误跳，前缀无其他错 | 13,899 | 0.766 | **+3.20** | −0.04 |
| 错误跳，前缀有其他错 | 11,937 | 0.709 | **+1.88** | −0.26 |

按 j 分开的对错分离度（Cohen's d）：j=0 h 2.20 / Δ 0.18；j=1 h 1.70 / Δ 0.47；
j=2 h 1.43 / Δ 0.07；j=3 h 1.83 / Δ 0.39。

脚本与结果：`error_propagation_probe/score_gate_v3_vs_nwrong.py`、
`error_propagation_probe/results/gate_v3_vs_nwrong_dev.json`、
`error_propagation_probe/logs/score_gate_v3_vs_nwrong_result.log`。

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

来源：`retrieval/results/20260628_120339_wavefront_pooled_lam0p25`（gate v1 时代的 session，
k_match 只取决于分解器，与 gate 版本无关）。

### 三个数据集上的跳数分布（线上分解器 vs 只用 MuSiQue 训练）

| 数据集 | 线上（MuSiQue+2Wiki）预测跳数分布 | 只用 MuSiQue 预测跳数分布 | gold 跳数分布 |
|---|---|---|---|
| 2Wiki | 2 跳 9,689 / 3 跳 140 / **4 跳 2,739** / 5 跳 8 | 1 跳 10 / 2 跳 12,428 / 3 跳 138 / **4 跳 0** | 2 跳 9,595 / 3 跳 88 / 4 跳 2,806 / 5+ 跳 87 |
| HotpotQA | 1 跳 14 / 2 跳 6,489 / 3 跳 853 / 4+ 跳 49 | 1 跳 40 / 2 跳 6,938 / 3 跳 422 / 4+ 跳 5 | 全部 2 跳 |

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

1. **「无 gate 基线」只认 0810 的 A 组**（`20260806_09*_baseline_only_*`）。0713/0719 两份周报对
   `0.6513` 这个数字的标注互相矛盾（一处说是带 31 层 gate 的 `gated_rule_a`，一处说是纯余弦无 gate）。
2. **gate 的贡献与 reader prompt 的贡献必须分开报**。见 [3.2 节](#32-gate-单独贡献同-prompt-口径)：
   2Wiki 最终 EM 提升里约 10 点来自 prompt。
3. **chain@1 是这个方法最有说服力的指标**（提升 +18 到 +26 点，远大于单跳 recall@1），因为 gate
   的价值正是「把整条链里某一跳的错纠回来」。
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

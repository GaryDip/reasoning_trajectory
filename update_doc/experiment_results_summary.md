# 实验与结果汇总(持续更新)

跟 [hidden_state_inventory.md](hidden_state_inventory.md)(数据/artifact 怎么产出、谁复用谁)是
两份不同的清单——这份只关心**做过哪些实验、结果是多少、现在谁最好**。

**三种评测口径不能直接混着比**,下面分三节:

1. 端到端 wavefront(真跑检索+生成,musique dev 全量 2417 条,四个指标口径一致,互相可比)
2. 候选池级 gold-rank(同一份 `adaptive_lambda_rerank` 的真实检索候选池数据,离线算,互相可比,但**跟第一节不是一回事**——这里比的是"gold 在候选池里排不排第一",不涉及生成)
3. gate 自己的分类指标(AUC/F1,回答"这跳选错了没有"这个二分类问题,数据源是固定的 gold/counterfactual trace,不是真实候选池,**不能跟前两节比**)

## 一、端到端 wavefront 结果(可比,当前最好:**gate v3**)

全部 musique dev 全量、`--beam-width 1`、`bart_decompose`,四项指标:hop 级 recall@1、chain
(整条链路全对)、答案 EM、答案 F1。

| 方法 | 目录 | recall@1 | chain | EM | F1 | 备注 |
|---|---|---|---|---|---|---|
| baseline(`gated_rule_a`,31层) | `retrieval/` | 0.6513 | 0.4233 | 0.4079 | 0.5012 | 生产默认 |
| gate_v2 `lr_rerank`(Δ only,分层) | `retrieval/` | 0.6627 | 0.4357 | 0.4100 | 0.5045 | 第一次四项全赢 baseline |
| h_j-probe `weighted`(λ=0.25) | `error_propagation_probe/` | — | 0.4659 | 0.393 | 0.4816 | oracle recall(非hop_match口径)=0.6735,答案侧不如 baseline |
| h_j-probe `probe_only` beam bw=3 | `error_propagation_probe/` | 0.4991 | 0.3488 | 0.3645 | 0.4511 | 单独用 probe 分数重排,倒退 |
| pairwise MLP(方案A) | `rerank_pairwise_mlp/` | 0.5844 | 0.3678 | 0.4009 | 0.4913 | 两轮迭代里较好的一次 |
| combined_gate `sum`(固定0.25/0.25) | `combined_gate_rerank/` | 0.6869 | 0.4894 | 0.4141 | 0.5089 | gate_v2+probe 固定权重求和,当时最好 |
| combined_gate `rrf` | `combined_gate_rerank/` | 0.6705 | 0.4564 | 0.3922 | 0.4822 | 不如 sum |
| combined_gate `veto` | `combined_gate_rerank/` | — | — | — | — | 用户反馈"没啥用" |
| gate v3(h_after ⊕ Δ 特征融合) | `retrieval/run_retrieval_exp_wavefront_gate_v3.py` | 0.6950 | 0.5010 | **0.4224** | 0.5146 | 四项全赢之前所有方法 |
| **context gate(学出来的逐候选权重)** | `context_gate_rerank/` | **0.7018** | **0.5184** | 0.4208 | **0.5147** | 检索侧比 v3 更好,答案侧基本打平 |

**当前结论:检索侧(recall@1/chain)目前最好的是 context gate,答案侧(EM/F1)v3 和 context gate
基本打平(EM 差 0.16 个点在噪声范围内,F1 几乎完全一样)。** context gate 相对 v3 的 chain 涨幅
达到 +1.74 个点,是目前所有方案里最高的,但这部分新修对的检索链路里有一部分没能进一步转化成更高的
答案 EM——大概率是 reader 本身的瓶颈(EM 是严格字符串匹配),不代表 context gate 检索侧的提升是
虚的,但也不是 v3 那种"四项指标同步全面碾压"的干净胜出,两个方案目前是"各有侧重"而不是一个完全
取代另一个。

## 二、候选池级 gold-rank(离线,同一份 `adaptive_lambda_rerank` 数据,互相可比)

数据源:`adaptive_lambda_rerank/data/musique_dev_lambda_data.npz`(6404 个真实 BGE 检索候选池)。
指标是 gold 候选在池子里的排名(recall@1 = 排第一的比例)。

| 方法 | recall@1 | 两个单独都错时,修好的比例(协同) |
|---|---|---|
| emb_score only | 0.8067 | — |
| probe only(0.25) | 0.8601 | — |
| gate_v2 only(0.25) | 0.8620 | — |
| sum(固定 0.25/0.25) | 0.8660 | 5.5% |
| confidence-weighted(启发式,按池内标准差分权重) | 0.8646 | 未测 |
| coarse(probe)→fine(gate_v2) k=3/5 | 0.8643 / 0.8624 | 未测 |
| 网格搜索最优(λ_gate=0.25, λ_probe=0.10) | 0.8690 | 1.5%(牺牲了协同换整体) |
| 三方凸组合最优(0.6/0.2/0.2) | 0.8679 | 未测 |
| 三方凸组合均等(0.33/0.33/0.33) | 0.8463(整体更差) | 16.0% |
| **learned context gate**(逐候选学出来的 α) | **0.8776** | **16.4%** |

**这一节目前最好的是 learned context gate**——不仅整体 recall@1 最高,协同修复率也最高,是这个
session 里第一次"整体准确率"和"协同"同时双赢(之前每次都要二选一)。但**这是离线候选池排序指标,
还没有做端到端 wavefront 验证**,不能因为这个数字比 v3 的端到端数字"看起来大"就认为它更好——两者
任务口径完全不同,唯一诚实的做法是把 context gate 也跑一遍端到端(`context_gate_rerank/
wavefront_context_gate.py` 已经写好并通过 import/编译验证,还没实际跑)。

## 三、gate 自己的分类指标(AUC/F1,判断"这跳选错了没有",不涉及候选池排序)

数据源:`hidden_states/pilot_multilayer`(固定 gold/counterfactual trace,不是真实检索候选池)。

| 版本 | 特征 | 层 | dev AUC | dev overall F1 |
|---|---|---|---|---|
| gate_v2(Δ only) | Δ_j | 按 j 分层(15/23) | 0.9221 | 0.7785 |
| **gate v3**(h_after ⊕ Δ) | concat(PCA(h_after), PCA(Δ_j)) | 同 v2 分层 | **0.9357** | **0.793** |

v3 在这个口径下也是最好的,而且第一节已经验证了这个离线的赢**真正兑现成了端到端的赢**(session 里
第一次)——这是目前把"gate 自己判断准不准"和"端到端检索/答案指标"都能对上号的唯一一条结果。

## 现在的整体结论

- **context gate 的端到端结果已经跑出来了**(见第一节):检索侧(recall@1/chain)比 v3 更好,答案侧
  (EM/F1)基本打平,不是全面碾压,是"各有侧重"。
- **效果基本打平的情况下,选 v3 作为默认方案**:v3 结构上就是一个 LR(复用 v2 已验证的层选择,
  sklearn 一次训练完事);context gate 是 gate_v2 + probe + 额外一个 PyTorch 门控网络,三个组件
  拼起来,链路更长、要维护的 artifact 更多(训练时就因为 sklearn 版本不一致炸过一次)。效果差不多
  时,更少活动部件的方案更值得作为默认——这也是这个 session 里反复验证到的规律(简单固定权重的
  `sum` 在很多轮"更聪明"的组合尝试里也一直很难被打败)。
- context gate 目前当一个已经验证过、检索侧略有优势的备用方案留着即可,不需要再继续投入(比如
  拆案例查为什么答案侧没跟上),除非之后有场景明确更看重检索准确率而不是最终答案准确率。

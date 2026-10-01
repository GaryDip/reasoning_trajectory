# 0713 update — gate v2 端到端检索结果 + beam search 新思路

承接 [0705update.md](../0706/0705update.md)：那份文档记录了 per-j 混层 gate(`gate/artifacts_pooled_v2`)
在 gate 自身指标(TPR/FPR/F1)上的全量验证结论，但当时明确留了一条"下一步"没做——端到端接回检索流程，
看这个提升能不能传导到真实检索/答案指标。这次把这个补上了，同时记录两个还在探索阶段的新思路
（beam search、以及把拆解也纳入 beam search），跟"继续优化 gate/decompose 单个模块"是不同性质的方向；
最后专门验证了一个中途冒出来的假设——"答案指标不动是不是因为 reader 本身摁住了天花板"——结果这个
假设**不成立**，瓶颈还是在证据选得准不准（第五节）。

## 一、gate v2 端到端检索结果（已完成的验证）

用 `run_retrieval_exp_wavefront.py --methods gated_rule_a`，`--artifacts-dir` 分别指向生产
`gate/artifacts_pooled`（31 层，基线）和 `gate/artifacts_pooled_v2`（per-j 混层），在 musique / 2wiki /
hotpot 三个数据集的 dev 全量上跑了一遍，逐字段对比：

| 数据集 | 指标 | baseline(31层) | gate_v2(混层) | 差值 |
|---|---|---|---|---|
| **MuSiQue** | recall@1 | 0.6513 | 0.6565 | +0.0052 |
| | chain_recall@1 | 0.4233 | 0.4290 | +0.0057 |
| | answer EM / F1 | 0.4079 / 0.5012 | 0.4042 / 0.4983 | **-0.0037 / -0.0029** |
| **2Wiki** | recall@1 | 0.8484 | 0.8569 | +0.0085 |
| | chain_recall@1 | 0.6684 | 0.6819 | +0.0135 |
| | answer EM / F1 | 0.4753 / 0.5493 | 0.4758 / 0.5499 | +0.0005 / +0.0006 |
| **HotpotQA** | recall@1 | 0.6186 | 0.6257 | +0.0071 |
| | chain_recall@1 | 0.4093 | 0.4112 | +0.0019 |
| | answer EM / F1 | 0.5130 / 0.6396 | 0.5129 / 0.6397 | -0.0001 / +0.0001 |

**结论**：

1. **检索质量指标（recall@1/@3、MRR、chain_recall）在三个数据集上一致提升**——这是比 0705update.md
   里"gate 自身 TPR/FPR/F1 变好"更有说服力的证据：三个数据集构造方式、领域都不一样，跨数据集一致
   变好，不太像是训练集分布内的巧合。
2. **但最终答案 EM/F1 几乎没有变化**（三个数据集都在 ±0.001~0.004 内浮动，musique 上甚至略微下降）。
   检索变好了，没有传导成下游任务的可衡量收益，说明当前流水线里 reader 这一环可能才是瓶颈，
   而不是检索/gate 这一段。
3. gate 触发率的变化方向在数据集间不一致：musique 上更少触发（0.3251→0.2958，更谨慎），
   2wiki/hotpot 上更多触发（0.1748→0.2167，0.2515→0.2789）——不是一个统一的"更保守"或"更激进"。

**要不要转正**：目前看是"检索层面确实更好，但对最终任务没有可衡量的收益"这个结论，转不转正需要
权衡"检索指标好看"和"没有下游收益"这两件事，属于人为决策，这份文档只提供数据支撑。

产出物：`retrieval/results/20260628_120339_wavefront_pooled_lam0p25`（baseline）、
`retrieval/results/20260706_153235_gate_v2_single_gpu`（gate v2）。

## 二、为什么要找新思路，而不是继续优化 gate / decompose

上面这个结果暗示：单独在 gate 或 decompose 层面做完善式的改动，边际收益可能已经不大了——gate 这次
已经是"per-j 混层"这种级别的改动，检索指标涨了，答案指标却没跟着涨；继续在这个模块里调参，大概率
也就是修修补补，很难指望有质变。所以这次往下想的两个方向，**都不是"改进某个模块的效果"，而是
"这个模块产出的隐状态信号能不能用出新的方式"**——跟一直以来"用 hidden state 判断推理链是否正常"
这个核心想法一致，只是用法不一样。

## 三、新思路 1：Beam Search（已实现，已跑出真实结果）

### 动机

现在的检索流程每一跳贪心地选一个候选证据就往下走，只在选择那一刻用 gate 判断"这一步像不像错的"。
如果某一跳选错了，后面的跳只能将错就错。想法是：每一跳保留多条候选路径（beam），用隐状态给每条
路径打分，分数差的路径随时剪掉——这样即使某一跳的错误在当时看不出来，只要它在后续跳的隐状态里
留下了痕迹，还有机会在后面被淘汰。

### 设计

- **路径打分 = 累计异常分之和**，不掺相似度。相似度只用在"每条路径自己检索候选"这一步（先筛出
  值得打分的候选，不然要给整个候选池都打分，太浪费）；跨路径比较、决定留哪几条，只看 gate 异常分。
- **没有"选 1 of 3"的 LLM 调用**——beam 剪枝本身就是选择机制，比贪心版本反而少一步。
- **每一跳流程**：beam 里 B 条路径各自展开子问题（可能因为前面选的证据不同，中间答案不同，展开出
  来的子问题也不同）→ 各自检索 top-k 候选 → 打包成 B×k 个"路径+候选"的延续，一次性批量过 gate
  打分 → 按新的累计分排序，全局剪枝回 B 条 → 只对剪枝后活下来的 B 条生成中间答案（不是 B×k 条，
  省了不少 vLLM 调用）。
- **最终答案**：不把多条 beam 的证据链都塞进一个 prompt 给模型判断（怕上下文太长、模型读串），
  直接用 beam 内部已经算出来的累计异常分最低的那一条，走现有单链最终答案生成流程，改动最小。

### 实现状态

新脚本 `error_propagation_probe/beam_search_retrieval.py`，大量 import 复用
`retrieval/run_retrieval_exp.py` 和 `run_retrieval_exp_wavefront.py` 里现成的数据加载 / 检索 /
批量 gate 打分（`score_gate_requests`）/ vLLM 生成封装，两个生产脚本一行没改。先用**假的 gate 打分 +
假的 vLLM 生成、但真实的数据集加载和检索**跑通了完整流程结构（beam 从 1 条自然长到 3 条、多跳剪枝、
只对幸存路径生成答案、最终选分数最低的 beam 生成答案、写出 JSONL），随后在真实 GPU（Llama + vLLM +
`gate/artifacts_pooled_v2`）上跑了 musique dev 全量（2417 题，`--decompose-mode bart_decompose`，
`beam_width=3`）。

真实跑法：
```bash
cd error_propagation_probe
python beam_search_retrieval.py --limit 0 --beam-width 3
```

### 结果：跟现有贪心 `gated_rule_a` 基本持平

拿同一份拆解来源（bart_decompose）、同一份 gate（`artifacts_pooled_v2`）做受控对比——唯一变量是
"贪心单路径 vs beam search 多路径"：

| | 贪心 `gated_rule_a`（第一节 gate v2 结果） | beam search（bw=3） | 差值 |
|---|---|---|---|
| answer EM | 0.4042 | 0.4055 | +0.0013 |
| answer F1 | 0.4983 | 0.4973 | -0.0010 |

**结论：这版 beam search（累计异常分求和剪枝，beam_width=3）在 EM/F1 上跟贪心版本没有可衡量的差异**，
在噪声范围内，既不是决定性提升也不是明显变差。这是一个真实、干净（拆解来源和 gate 都对齐了）的
中性结果，可能的原因：

- `gated_rule_a` 本身已经带了"异常分超阈值才 expand 候选池重排"的机制，某种程度上已经在做局部
  最优选择，beam search 多保留几条路径未必能挖出额外信息；
- beam 打分只看"累计异常分低不低"，但"异常分低的路径"不一定是"最后能答对的路径"，这个代理指标
  本身可能不够准；
- `beam_width=3` 可能还不够宽，没有真正探索到贪心会错过的路径。

**放在一起看，这次的结果当时让人怀疑一个更大的模式**：无论是换 gate 层（第一节：检索指标涨、答案
指标不涨），还是换搜索策略（这里：beam search 跟贪心持平），会不会是"只要下游 reader（最终答案
生成那一步）不变，答案层面的天花板已经被 reader 本身摁住了"？**这个假设第五节里专门验证过，结论是
不成立**——给同一个 reader 换上真实 gold 证据，答案指标明显往上跳了 6~8 个点，说明 reader 没有
被摁住，瓶颈还是在证据选得准不准。

## 四、新思路 2：把拆解（decompose）也纳入同一个 beam（设计已讨论，未实现）

### 动机

decompose 现在是独立预处理步骤，BART 只输出 top-1 拆解，好不好完全由 BART 自己训练时的目标
（BLEU/exact-match 对 GT 子问题）决定，跟"这个拆解到不到最后有没有帮助检索/推理"没有直接关系。
既然核心想法是"用隐状态判断推理链正不正常"，那这把尺子也应该能用来评判"拆解选得好不好"，而不是
让拆解继续用一套完全独立的、跟下游脱节的指标去优化。

### 设计（两种方案，取舍不同）

**方案 A（成本几乎为零，优先尝试）**：不训练任何新东西，把"选拆解"这件事折进现有的 evidence beam
search 里，让它被现有 gate **间接**评判。

**动手写这部分代码之前，必须先验证一个前提，否则整个方案 A 可能没有意义**：BART 现在
`num_beams=10` 解码时，**排名第 2、3...名的候选，到底是不是跟第 1 名有实质差异的、多样化的拆解**，
还是几乎一样、只差一两个词的近似重复？如果候选之间高度雷同，那"多留几个候选给 beam 去选"根本没有
东西可选；另外还要看，**扩大到 top-3/top-5 范围之后，正确（或更接近 GT）的拆解到底会不会出现**——
如果正确答案基本只会出现在第 1 名，扩大候选范围也没有意义。这两点都要先用真实数据、真实训好的
checkpoint（`decompose/bart/outputs/bart_decomposer_musique_2wiki_repro/`）跑一次验证，具体看：
top-N 候选之间的两两 BLEU（低=多样性高）、以及"best-of-N"随 N 变化的趋势（N=1/3/5/10 时，N 个
候选里最接近 GT 的那个分数/命中率有没有随 N 增大而明显变好）。验证通过（候选确实多样、且扩大范围
后确实更容易覆盖到接近正确的拆解）才值得往下做方案 A 剩下的部分；如果验证不通过，方案 A 大概率
要换成先改 BART 的解码策略（比如换成 diverse beam search）才有意义。**这一步还没跑，是接下来最先
要做的事。**

- BART 推理时把 `num_return_sequences` 从 1 改成 M（比如 3），拿到 M 个候选拆解——这一步几乎不
  增加算力，因为 BART 默认 `num_beams=10`，beam search 内部本来就在探索 10 条候选，现在只是把
  第 2、3 名也捞出来，不是重新生成。
- beam 的起点从"1 条空路径"变成"M 条种子路径"，每条路径除了记录已选证据/答案/累计分，还额外
  带着"自己是用哪个候选拆解生成的"（自己的 `sub_questions` 列表）。
- 第 1 跳照常：M 条路径各自展开子问题（不同拆解此刻文本可能不同）→ 各自检索 → 一起打包批量过
  gate 打分 → 全局剪枝到 B 条。这一步没有专门比较"哪个拆解更好"，纯粹按打分排序——拆解写得烂，
  检索出来的证据接上去大概率也会显得不连贯，异常分偏高，自然被剪掉，不需要额外逻辑。
- 剪枝后幸存的路径继续沿着**自己所属的那份拆解**往后走（第 2 跳用这份拆解自己的第 2 个子问题），
  不同路径可能来自不同拆解，互不影响。
- **拆解跳数不一致的问题**：如果拆解 A 是 2 跳、拆解 B 是 3 跳，路径分数不能直接比"累计异常分之
  和"（跳数少的天然分低，不公平），改成比"每跳平均异常分"。但平均值本身还有一个更细的坑：跳数少
  的路径打分次数少，平均值方差大，容易"运气好"出现虚低的平均分，统计上天然偏向跳数少的路径——
  这个问题目前打算先不处理，跑真实数据看偏差有多大再说，同时加一条"单跳异常分超过阈值直接淘汰"
  的硬性保底规则，防止"一步错得很明显，但被其余跳的低分平均掉"这种情况。
- 最终答案：跟方案一的 beam search 一样，把不同拆解、不同长度、走到底的所有路径放在一起，按
  "每跳平均异常分"（过滤掉被保底规则淘汰的）选最优，生成最终答案。

**方案 B（更直接，但成本高很多，暂缓）**：专门为"子问题本身好不好"训一个新的、独立于 evidence
gate 的信号——在还没检索证据之前，就用"prefix + 候选子问题"的隐状态判断这个子问题接不接得上前面
的推理。这个信号更直接、更早，但没有现成标签，需要仿照现有 gold/counterfactual 协议，人为构造一批
"坏子问题"（比如从别的题或别的跳错位插入一个子问题）训练数据，重新抽隐状态、重新训一个 PCA+LR，
相当于给"拆解质量"单独造一个平行于现在 gate 的新模块，工作量明显大于方案 A。

**顺序**：先做方案 A（零训练成本，直接在方案一 beam search 基础上小改），如果发现"间接评判"效果
不够（比如异常分主要被证据牵着走，子问题本身的问题被稀释掉了、beam 剪枝救不回来明显烂的拆解），
再考虑方案 B。

### 实现状态

设计已经讨论清楚，**代码还没写**——落地时会在 `error_propagation_probe/beam_search_retrieval.py`
基础上改（`BeamPath` 加一个自带的 `sub_questions` 字段，beam 初始化从 1 条改成 M 条，加一个 BART
`num_return_sequences` 的小脚本产出候选拆解），不需要另起一个独立系统。

## 五、验证"reader 是天花板"这个假设——结果：假设不成立

### 实验设计

`reader_ceiling_probe/oracle_evidence_reader.py`（新建的独立目录，跟 `error_propagation_probe` 平行，
不混在一起）：跳过检索/gate 整个环节，每一跳强制使用 **gold 证据**，reader（中间答案生成 + 最终
答案生成）本身完全不变，看答案 EM/F1 的天花板在哪。不加载 gate 模型（没有任何东西需要打分），比
beam search 更轻量。

设计上有一个一开始没注意到、后来讨论中发现的坑：如果按位置直接取 `gold_decomp[hop_j-1]` 的证据
硬塞给 bart_decompose 拆出来的第 j 个子问题，**两者顺序不一定对得上**（BART 拆解的跳序不保证和
gold 拆解一一对应）——硬塞的话可能塞进一段答非所问的证据，让整个"oracle"实验失去意义。最终方案：

- `--decompose-mode gt`：子问题本来就是 gold 拆解转换来的，`sub_questions[hop_j-1]` 直接就是
  gold 第 j 跳，按位置直接取证据，没有对不上的风险。
- `--decompose-mode bart_decompose`：把这道题**全部**的 gold 证据放进一个池子，每一跳用现成的
  `prompt_select_passage_with_context`（生产贪心流程本来就有的"选 1 of 3"函数，这里候选池换成
  纯 gold 证据）让模型自己选一个跟当前子问题最匹配的，选中后从池子里移除，保证每条 gold 证据只用
  一次；池子用完（bart 预测跳数多于 gold 覆盖的跳数）就退回 top-1 检索兜底。

### 结果（musique dev 全量，`--decompose-mode bart_decompose`，可直接跟前面几节比）

Gold 覆盖率 99.33%（6037/6078 跳用了真实 gold 证据，only 41 跳 fallback 到检索，覆盖率很高，
结果不是靠少数简单题撑起来的）：

| 方法 | EM | F1 |
|---|---|---|
| baseline `gated_rule_a`（31 层） | 0.4079 | 0.5012 |
| gate_v2 `gated_rule_a`（混层） | 0.4042 | 0.4983 |
| beam search（bw=3） | 0.4055 | 0.4973 |
| **oracle reader（gold 证据）** | **0.4708** | **0.5754** |

分 K：K=2 EM=0.5050/F1=0.6094，K=3 EM=0.4601/F1=0.5590，K=4 EM=0.2941/F1=0.4204（跳数越多天花板
本身越低，符合直觉，但每个 K 上都明显高于现有方法）。

**结论：给同一个 reader 换上 gold 证据，EM/F1 涨了 6~8 个点，远超之前几次改动（±0.001~0.004）的
噪声范围。"reader 是天花板"这个假设不成立——reader 本身有能力答对，瓶颈确实还是在证据选得准不准。**
现在 baseline/gate_v2/beam search 的 recall@1 都只有 60% 出头，离拿到真正的 gold 证据还差得远，
gate 层混合、beam search（累计异常分求和）这两种具体做法都没能把"选中 gold"这件事真正提上去。

真实跑法：
```bash
cd reader_ceiling_probe
python oracle_evidence_reader.py --decompose-mode bart_decompose --limit 0
```

## 六、新思路 3：重排结构换成学出来的模型（方案 A：pairwise MLP）——离线好看，实战倒退

### 设计回顾

不再用 `emb_score - λ·abnormal_score` 这个固定加权公式，改成一个真正训出来的小模型，输入
`[emb_score, PCA(Δ)]`，直接输出一个排序分数。方案 A（`rerank_pairwise_mlp/`）用 **pairwise 排序
损失** 训练：同一跳里 gold 证据 vs 一个 counterfactual 构造出来的具体错误证据，让模型学
`score(gold) > score(wrong)`（RankNet 式 `-log(sigmoid(score(gold) - score(wrong)))`）。跟方案一
（beam search）、方案二（拆解纳入 beam）不是同一条线——这次动的是"打分公式本身"，不是"搜索策略"。

### 离线验证：非常好看

在全量 train/dev 数据上（train 35476 对，dev 4472 对）训完：

| | dev pairwise accuracy |
|---|---|
| baseline（只用 emb_score 排序） | 0.7238 |
| **训出来的 pairwise MLP** | **0.9208** |

错误率从 27.62% 降到 7.92%，看起来是很扎实的提升——说明 Δ 确实带有余弦相似度覆盖不到的信息。

### 真实接回检索流水线：recall@1 反而退步了

用 `wavefront_pairwise_mlp.py`（跟生产 `run_retrieval_exp_wavefront.py` 同一套评估方法论，真实
vLLM 选段/生成）在 musique dev 全量上跑（`bart_decompose`，可直接跟前几节比）：

| | recall@1 | chain_recall@1 | EM | F1 |
|---|---|---|---|---|
| baseline `gated_rule_a`（31层） | 0.6513 | 0.4233 | 0.4079 | 0.5012 |
| gate_v2（混层） | 0.6565 | 0.4290 | 0.4042 | 0.4983 |
| **pairwise MLP（新）** | **0.5955** | **0.3496** | **0.393** | **0.4835** |

**recall@1 从 65% 掉到 60%，是倒退，不是持平。**

### 诊断：离线指标和真实场景的"及格线"不是一回事

92% 的 pairwise accuracy 测的是"gold 对一个特定的错误候选，能不能赢"；但真实检索时，gold 要**同时
赢过 top-10 里其余 9 个候选**，不是单挑一个。训练数据(`build_training_data.py`)里的"wrong"来自
counterfactual trace 构造时 cosine-top-k 挑出来的**一个具体候选**，模型只学过"在这一个对手面前赢"，
从没在训练时见过要同时打赢一整个真实候选池——离线数字好看，是因为及格线本来就比真实场景低很多
（粗略估计，就算独立同分布，单场胜率 92% 打赢 9 个对手的联合胜率也只有 $0.92^9≈45\%$，现实中显然
更低，且候选之间的错误还不独立）。**这正是当初设计方案 B（listwise，用真实 top-k 候选池训练）时
担心的问题，这次是真实数据实锤了这个担心是对的，不是假设。**

### 追加实验：给特征加上 hop 深度信息（一个真实但不够的修正）

讨论中发现一个之前漏掉的问题：生产 gate 是 **4 个独立模型**，每个 pooled j 各训一个；但
`build_training_data.py` 把所有 K、所有 j 的样本混在一起训了**一个**模型，模型完全不知道"现在
是第几跳"。补了一版特征：`[emb_score, PCA(Δ), one-hot(j, 4类), K]`，让同一个模型至少能条件化地
区分"跳数位置"。重新跑 `wavefront_pairwise_mlp.py`：

| | recall@1 | chain_recall@1 | EM | F1 |
|---|---|---|---|---|
| baseline `gated_rule_a`（31层） | 0.6513 | 0.4233 | 0.4079 | 0.5012 |
| pairwise MLP（无 hop 特征） | 0.5955 | 0.3496 | 0.3930 | 0.4835 |
| **pairwise MLP（加 hop 特征）** | **0.5915** | **0.3500** | **0.4009** | **0.4913** |

跟"无 hop 特征"版本比：EM/F1 各涨了约 0.008，recall@1/chain_recall@1 基本没动（recall@1 甚至
微降）。跟 baseline 比：EM/F1 差距从 -0.015/-0.018 缩小到 -0.007/-0.010，缺口小了一半左右，但
recall@1 还是差 6 个点，没解决。**结论：hop 特征方向是对的、确实带来了一点真实提升，但量级不够，
没有触及"训练时只跟一个具体对手比、真实检索要同时赢过整个候选池"这个更根本的分布不匹配问题**——
这进一步说明问题的主要根源还是训练数据的真实性(pairwise vs listwise)，不是特征这一层的小修小补。

### 下一步（已经在做）

跑 `rerank_listwise_contrastive/wavefront_listwise_contrastive.py` 做同样的端到端对比——如果
listwise 版本（训练时就要求 gold 赢过真实检索出来的整个候选池）recall@1 明显好于这次的 0.5955、
甚至好于 baseline 的 0.6513，就能确认"问题出在训练时的比较范围不够真实"，而不是"Δ+emb_score 这
个组合思路本身不行"；反过来如果 listwise 也不行，就要重新考虑这条"学一个排序模型"的路线本身。
listwise 版本也应该顺手加上同样的 hop 特征，看两个改动能不能叠加。

## 七、下一步

1. **最优先：跑 `wavefront_listwise_contrastive.py` 做端到端对比**（见第六节）——判断 pairwise MLP
   recall@1 倒退是"训练时比较范围不够真实"这个可以修的问题，还是"学一个排序模型"这条路线本身有
   更深的毛病。这个结果出来之前，不建议再往 pairwise MLP 或拆解 beam 这些方向继续投入。
2. **第五节的结论翻转了第三/四节的优先级判断**——之前因为 beam search 跟贪心持平，怀疑"继续折腾
   检索侧"意义不大，现在证明检索侧确实还有 6~8 个点的空间，**"新思路 2（拆解也纳入 beam）"重新
   变得值得做**，只是要想清楚 beam search 本身（累计异常分求和剪枝）这个机制为什么没能兑现这部分
   空间——不能假设"多加一层拆解搜索"就能自动解决"当前打分机制选不中 gold"这个更根本的问题。
3. **验证 BART beam 候选的多样性和"扩大范围能不能覆盖到正确拆解"**（见第四节方案 A 里的前置验证
   要求）。`error_propagation_probe/check_decompose_diversity.py` 已经写好，**还没有实际跑出
   结果**，是接下来要做的第一件事——尤其现在知道检索侧有真实空间，这一步的价值比之前更高了。
4. **更根本的问题，是 beam search 现在用的"累计异常分求和"这把尺子，到底跟"选没选中 gold"相关性
   有多强**——可以拿 `reader_ceiling_probe` 产出的数据反过来验证：现有方法选中的证据 vs gold
   证据，两者的异常分分布差多少？如果差得不多，说明这把尺子本身就不够灵敏，不管是套在贪心还是
   beam search 上都难有起色，需要先回头改进"怎么判断证据选得对不对"这件事本身，而不是急着在
   现有打分机制上叠加更多搜索维度（拆解 beam）。
5. ~~`error_propagation_probe/` 里之前那三个"错误传导"分析脚本还没跑过真实数据~~ ——**已用真实
   数据跑通，而且顺着这条线继续往下做了 beam search 接入 + 加权 rerank，结果见第八节**。这条线
   实际上后来居上，比 listwise 那条路线（第 1 条）投入更多、优先级也更高——第 1 条列的
   `wavefront_listwise_contrastive.py` 对比至今没跑，暂时搁置。
6. `artifacts_pooled_v2` 转不转正，还是那个悬而未决的决策——第一节的数据已经给全了，等一个人为决定。

## 八、新思路4：error_propagation_probe 的 h_j probe 接入检索——真实结果：单独用不如混合用

### 背景：h_j probe 本身已经验证过是有效信号

`error_propagation_probe/score_and_analyze.py` 在 h_K（以及池化所有中间位置 h_j）上训了一个
PCA+LR probe，标签只是二分类"到这里为止有没有错"，但held-out 上分数会随真实错误数量单调上升，
Spearman(n_wrong, score) 池化全部位置能到 0.6~0.65（K=3/4 数据增量之后从 0.6089 涨到 0.6348），
是一个真实、可复现的发现（细节见 `error_propagation_probe/probe_artifacts/training_report.md`）。
这一节是把这个 probe 接进真实端到端检索之后的结果——**结论跟 probe 本身的有效性无关，是"怎么用"
这一步出了问题**。

### 第一次真实结果：单独用 probe 打分，全面比 baseline 差

`wavefront_hidden_probe_beam.py`：hop 1 对 top-10 候选逐个打分，留分数最低的 top-3，hop 2 开始
每个 beam 的候选混在一起全局取 top-3，最后取分数最低的 beam 生成答案——排序**只用 probe 分数**，
检索时的 cosine 相似度只用来圈定候选池，圈完之后就完全不参与排序。

| | oracle/recall@1 | chain 相关 | EM | F1 |
|---|---|---|---|---|
| baseline `gated_rule_a`（31层） | 0.6513 | chain_recall@1=0.4233 | 0.4079 | 0.5012 |
| h_j-probe beam（beam_width=3，纯 probe 打分） | 0.5675 | chain_match=0.3488 | 0.3645 | 0.4511 |
| h_j-probe 贪心（beam_width=1，纯 probe 打分） | 0.5726 | chain_match=0.3839 | 0.3579 | 0.4419 |

（这里的"oracle/recall@1"和"chain_match"跟生产 recall@1/chain_recall@1 不是完全同一个指标口径，
beam_width=1 时二者近似重合；具体差异见脚本 docstring。）

**全面倒退，不是持平。** 而且 beam_width=3→1 之后，检索侧指标反而变好（chain_match 0.35→0.38），
答案指标反而变差（EM 0.365→0.358）——说明这次 beam search 本身没有额外收益，问题出在打分方式上。

### 诊断：生产的 gate 从来没有单独扛过排序，这次是一次没有先例的新用法

复盘生产怎么用 gate 分数：`lr_rerank` 是 `emb_score - λ·gate_score`，排序主力还是 cosine，gate
只是小权重修正项；`gated_rule_a` 更干脆，gate 根本不参与排序，只是"top-1 候选 gate 分数超阈值就
扩池"的二元触发开关。**production 从未让 gate/probe 分数单独扛起"把真实候选池排出先后"这件事**。
这次 beam search 完全抛开 cosine、纯用 probe 分数排序，是一个之前系统里不存在、没被验证过的新
用法，不是重复了之前 pairwise MLP 那次"训练/部署分布不匹配"的老问题。

### 加个开关，混回 cosine：`--rerank-mode weighted`

照抄生产 `rerank_by_final_score(emb_score, abnormal_score, lam)` 的公式，`final_score =
emb_score - λ·probe_score`（λ 默认 0.25，跟生产 `--lambda-lr` 默认一致），beam_width=1：

| | oracle/recall@1 | chain 相关 | EM | F1 |
|---|---|---|---|---|
| baseline `gated_rule_a`（31层） | 0.6513 | chain_recall@1=0.4233 | 0.4079 | 0.5012 |
| h_j-probe 贪心（纯 probe 打分） | 0.5726 | chain_match=0.3839 | 0.3579 | 0.4419 |
| **h_j-probe 贪心（`weighted`，λ=0.25）** | **0.6735** | **chain_match=0.4659** | **0.3930** | **0.4816** |

混回 cosine 之后提升非常明显——oracle/recall@1 从 0.57 涨到 0.67（反而略高于 baseline 的
0.6513），chain_match 从 0.38 涨到 0.47（也略高于 baseline 的 0.4233）。**但 EM/F1 依然比
baseline 低一截**（0.393 vs 0.4079，0.4816 vs 0.5012）——检索侧的排序质量已经追平甚至略超
baseline，答案质量却还没追上，这是一个值得记住的反直觉现象：**检索排出来的证据顺序更好，不等于
最终答案一定更准**，中间的 select/answer 生成环节可能还有别的损耗，值得后续单独排查。

**结论：`weighted` 止住了"全面倒退"，但还没有真正超过 baseline，只是从"全面差"变成"检索侧打平、
答案侧仍有缺口"。** 单独用这个 probe 排序是不成立的用法，必须跟 cosine 混合；固定 λ=0.25 是照抄
生产默认值，没有针对这个新场景专门调过。

### 下一步（正在做）：让 λ 自己学，而不是拍一个固定值

`adaptive_lambda_rerank/` 是接下来的方向——不再用固定 λ，而是训一个很小的模型，输入这一跳的上下文
（主问题 BGE embedding、h_prev、当前子问题 BGE embedding），输出这一跳专属的 λ，同一个 λ 用在这
一跳全部候选上（不针对具体候选，只决定"这一跳该多信 cosine 还是多信 probe"）。训练复用
`rerank_listwise_contrastive/build_topk_pools.py` 的真实候选池 + 现有 probe 打分，loss 是 listwise
softmax（gold 排第一），K=3/4 数据不够的问题用 `data/decompose/musique/gt/train_nl_enhance.jsonl`
（LLM 改写过的 K=3/4 子问题，真实检索一遍造出额外的真实候选池，不是复制数据）做了补充。目前代码
已经写完、小规模冒烟测试过，还没跑全量真实结果——跑出来之后回来更新这一节。

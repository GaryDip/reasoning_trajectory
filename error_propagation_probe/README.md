# error_propagation_probe — beam search 前置探索

这个目录是一次独立的探索性实验，**不改动、不依赖**任何 production 代码/数据(`traces/`、
`hidden_states/{train,dev}`、`gate/` 只以只读方式被 import/加载)。目的是回答一个在设计
"beam search 式检索"之前必须先搞清楚的问题。

## 动机

现在的检索流程(`retrieval/`)每一跳贪心地选一个候选证据就往下走，只在选择的那一刻用
gate 判断"这一步像不像错的"。想尝试 beam search 的思路：每一跳保留多条候选路径(prefix)，
用隐状态给每条路径打一个"这条路径现在看起来有多不正常"的分数，分数差的路径剪掉。

这个思路成立的前提是：**某一跳选错了证据，这个错误得能在后面几跳的隐状态里留下痕迹**——
否则贪心地只在犯错的那一刻检查一次，和保留多条路径、每一步都检查一次，效果不会有实质区别，
beam search 就没有意义。这就是本目录要验证的核心问题。

## 出发点：现有数据回答不了这个问题(两个先于写代码就查到的事实)

1. **`hidden_states/extract_hidden_states.py` 里 `n_take = wrong_hop + 1`**：counterfactual
   trace 在抽取隐状态时，一旦过了"注入错误的那一跳"就不再往后抽了。也就是说，现在全部
   92,748 个 `.npz` 里，**没有任何一条包含"错误发生之后"的隐状态**——不是分析方法选错了，
   是这份数据从一开始就没被存下来。
2. **`wrong_hops` 长度全部是 1**：查了 train(59,857 条 error trace)+ dev(6,404 条)的
   manifest，`wrong_hops` 长度分布 100% 是 `{1: N}`，没有一条同时错两跳以上的 trace——
   `traces/construct_balanced_traces.py` 的构造协议就是"单跳注入"，从设计上没有多错场景。

结论：这两件事都得靠**新建数据**才能验证，不能在现有 `.npz` 上做分析。

## 已实现的内容

### 1. `build_multi_error_traces.py`
对每道题(K 跳)构造 `n_wrong = 0, 1, ..., K` 一整套变体：`n_wrong=0` 是纯 gold；
`n_wrong=n` 时随机挑 n 跳、每跳换成 cosine top-1 的非 gold 干扰段落(跟生产
`construct_balanced_traces.py` 用的是同一套 `CosinePoolRanker` 逻辑，import 复用，不重写)。
一道题产出 K+1 行，`wrong_hops` 记录具体是哪几跳被换了。

已用真实 BGE 模型 + 真实 musique dev 数据(15 道题)跑通：产出 45 行(15×3，因为样本恰好全是
K=2)，`n_wrong` 分布、`wrong_hops` 内容、生成的 trace 文本都验证正确，且能被
`hidden_states/trace_parse.py::parse_trace_structure` 正常解析。

### 2. `extract_full_hidden_states.py`
复用 `hidden_states/extract_hidden_states.py` 里的 `HiddenExtractor`/批处理逻辑(原样
import，没有复制)，只改了一处：自己写的 `full_trace_job_from_row` 让 `n_take` **永远等于
`len(prefixes)`**，不再套用生产那条"错误跳+1就停"的截断规则——这正是缺失数据的根因，现在
每条 trace(不管 n_wrong 是多少)都会一路抽到 `h_K`。已验证 `n_take` 逻辑本身正确(对
n_wrong=0/1/2 的样本都拿到了完整的 K+1 个 prefix)。**这一步需要真实 GPU + Llama
checkpoint 才能实际跑，还没有真实数据产出。**

### 3. `score_and_analyze.py`
读取上一步的隐状态，对每条 trace 取最后一个 prefix 的隐状态 `h_K`，训一个**全新的**
PCA(64)+LogisticRegression 概率探针，标签是"这条 prefix 有没有被污染(`n_wrong>0`)"，
按 `source_id` 分组切 train/test(同一道题的 n_wrong=0..K 几个变体必须整体分到同一边，
否则会泄露)，在**没参与训练**的那部分数据上看预测概率是否随 `n_wrong` 单调上升
(报告按 (K, n_wrong) 和 pooled 两种粒度的 mean/median，外加 Spearman 相关系数)。
已用人工注入信号的合成数据跑通整条链路(分组切分、训练、held-out 评估全部工作正常，
合成信号下 Spearman≈0.87)，真实结论要等第 2 步产出真实隐状态后才能跑出来。

## 一个关键的设计决策：为什么不能直接套用现有的 `gate/artifacts_pooled_v2`

一开始想偷懒直接拿生产 gate 打分，但这是不对的：生产 gate 训练的特征是
**`Δ_j = h_j - h_{j-1}`**，回答的是"这一步跟上一步比，是不是刚被动了手脚"——一个局部的、
逐步比较的信号，天然适合"揪出发生在这一刻的错误"。但 beam search 要做的是**比较不同候选
路径整条 prefix 的好坏**，需要的是"这个 prefix 本身现在处于什么状态"，不是"相对上一步变化
了多少"。所以 `score_and_analyze.py` 训练的探针用的是 `h_K` 本身(绝对状态)，不是 Δ。

## 后续要联动的地方(还没做，先记下来)

**`h_j - h_{j-1}` 这个 Δ，跟现在生产检索(`retrieval/`)里 gate 打分用的特征，其实是同一个
东西**——现在只是换了个角度在问同一份隐状态：production 问"这一步的 Δ 正不正常"，这里问
"整条路径现在正不正常"。这两者不是互不相干的两套东西，后面值得想一想能不能联动，比如：

- beam search 给一条路径打分，是不是可以不用重新训一个"整条路径"的探针，而是直接把这条
  路径上**每一步已经算过的 Δ 异常分**汇总起来(比如求和、取最大值、或者看有没有某一步显著
  超过阈值)？如果这样就够用，等于说"路径级别的判断"可以完全建立在现有 production gate
  的逐步打分之上，不需要新训一个东西。
- 反过来，如果 `h_K` 探针(这里训的这个)明显比"汇总每步 Δ 分数"更准，说明整条路径的隐状态
  里确实携带了比"每一步局部变化"更多的信息，那 production 现在只看 Δ、不看绝对状态，可能
  是漏了一部分信号。

这个问题要等第 2、3 步真实数据的结果出来后才能回答，先记在这里，不要忘了。

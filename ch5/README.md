# 第 5 章 Discussion：数据与图

本目录是第 5 章所有定量结果与图的**唯一落点**。对应 `CH5_PLAN.md` 的任务 1–4；叙述与解读写在
`WORK_SUMMARY.md` 的 **3.13 节**，这里只放可机读的结果、图，以及复现方式。

**自成一体**：第 5 章专用的脚本都已经收到 `ch5/scripts/` 下，数据在 `ch5/results/`，图在
`ch5/figures/`，复现只需要在仓库根执行 `python ch5/scripts/<脚本>`。只有两样东西留在外面，
因为它们是和别的章共用的：`retrieval/goldmatch.py`（gold 判定协议，第 4 章也用）和
`error_propagation_probe/score_gate_v3_vs_nwrong.py`（0928 的能力剖析探针）。

```
ch5/
├── README.md
├── results/
│   ├── ch5_failure_analysis.json        任务 1：完整系统的失败模式（1a–1f）+ k_match 复算
│   ├── bucketD_wording_audit_sample.jsonl  每数据集 50 条「只是措辞不同」样本，待人工核对
│   ├── order_invariant_metrics.json     gold 顺序口径：chain@1 / recall@1 两种判定的对照
│   ├── exp4b_natural_by_pattern.json    任务 2：自然错误按错误模式分组的前缀污染
│   ├── gate_l15_hops_dev.jsonl          51,672 条逐跳 gate 分数（图 5.1 误差条的原始数据，12.9 MB）
│   └── gate_l15_vs_nwrong_dev.json      0928 能力剖析的聚合结果（统一第 15 层重新打分，图 5.2 的数据源）
├── scripts/                             第 5 章专用脚本（全部纯 CPU）
│   ├── ch5_failure_analysis.py          任务 1：1a–1f + k_match 复算（约 1 分钟）
│   ├── exp4b_natural_by_pattern.py      任务 2：自然错误按错误模式分组（约 2 分钟）
│   ├── make_ch5_figures.py              任务 3：两张图
│   └── order_invariant_metrics.py       gold 顺序口径对照（十几秒）
└── figures/
    ├── disc_score_vs_error_position.pdf   图 5.1   9.00 × 5.98 cm
    └── disc_logit_decomposition.pdf       图 5.2  16.00 × 5.49 cm
```

---

## 怎么复现（全部 CPU，不需要 GPU）

```bash
cd /home/ertai/reasoning_trajectory/multihop_trace

# 任务 1：失败模式 + k_match（约 1 分钟）
python ch5/scripts/ch5_failure_analysis.py

# 逐跳分数 dump（约 3 分钟，图 5.1 的误差条需要它；已生成则可跳过）
cd error_propagation_probe && python score_gate_v3_vs_nwrong.py \
    --hidden-dir hidden_states_multilayer/dev \
    --artifacts-dir ../gate/layer15_uniform/v3_artifacts \
    --out results/gate_l15_vs_nwrong_dev.json \
    --dump-hops ../ch5/results/gate_l15_hops_dev.jsonl && cd ..

# 任务 2：自然错误按错误模式分组（约 2 分钟）
python ch5/scripts/exp4b_natural_by_pattern.py

# 任务 3：两张图
python ch5/scripts/make_ch5_figures.py          # 或 …/make_ch5_figures.py 5.1

# gold 顺序口径：chain@1 / recall@1 的两种判定对照（十几秒）
python ch5/scripts/order_invariant_metrics.py

# 第 4 章全量 32 个 run 的同口径重算（约 2 分钟，输出在 retrieval/results/）
python retrieval/compute_order_invariant_v2.py
```

| 脚本 | 产出 |
|---|---|
| `ch5/scripts/ch5_failure_analysis.py` | `results/ch5_failure_analysis.json`、`results/bucketD_wording_audit_sample.jsonl` |
| `ch5/scripts/exp4b_natural_by_pattern.py` | `results/exp4b_natural_by_pattern.json` |
| `ch5/scripts/make_ch5_figures.py` | `figures/*.pdf` |
| `ch5/scripts/order_invariant_metrics.py` | `results/order_invariant_metrics.json` |
| `error_propagation_probe/score_gate_v3_vs_nwrong.py --dump-hops`（共用） | `results/gate_l15_hops_dev.jsonl`、`results/gate_l15_vs_nwrong_dev.json` |

**依赖的外部模块**（脚本自己插 `sys.path`，不用设环境变量）：`retrieval/goldmatch.py`（gold 判定）、
`retrieval/run_retrieval_exp_wavefront_gate_v3.py` 的 `score_gate_v3` 与 `gate/lr_artifacts.py`
（任务 2 要给 trace 重新打分）、`prelim_analysis/common.py`（任务 2 的参考分布与画图样式）、
`../multihop_trajectory/.../musique/metrics/answer.py`（官方答案规范化）。
搬目录时这四处是唯一需要跟着改的地方。

**口径**：所有输入都是统一第 15 层的新 run（main = λ=0.60 + `comparison_hint`，control = λ=0），
与第 4 章同口径。答案规范化直接 import MuSiQue 官方 `metrics/answer.py`，与 EM 完全一致。

**gold 段落顺序（v2 协议，1011）**：HotpotQA 只有两个支持段落、标注固定了一种顺序，2Wiki
`comparison` / `bridge_comparison` 的两条分支同理，**先查哪个都对**；按位置比较会把换序判成检索错。
所以任务 1 的 1a / 1b / 1e 全部**同时**报两种判定。协议实现在 `retrieval/goldmatch.py`，
与第 4 章共用（决定见 `ORDER_INVARIANT_TASK.md`，第 4 章全量重算见
`retrieval/results/order_invariant_v2.json` 与 WS 3.14）。

| 项 | 定义 |
|---|---|
| 逐跳命中（顺序无关） | `para_ids[j] ∈ set(gold_idxs)` **且 `para_ids[j]` 没被更早的跳取过** |
| 适用范围 | HotpotQA 全部题；2Wiki `comparison` + `bridge_comparison` |
| 不适用 | MuSiQue 全部、2Wiki `compositional` / `inference` —— 两种判定**逐位相同**（自检） |

| 数据集 | 顺序无关题数 | 其中重复取同一篇 gold |
|---|---|---|
| MuSiQue | 0 | — |
| 2Wiki | 5,791（46.1%） | 474（8.2%） |
| HotpotQA | 7,405（100%） | 1,425（19.2%） |

**「没被更早的跳取过」是 1011 补的关键修正**：第一版只判 `∈ set(gold_idxs)`，于是两跳取同一篇
gold 也算全对，而那是比较类题最常见的真实错误之一。λ=0 对照里这类题更多（HotpotQA 2,168 /
29.3%，2Wiki 908 / 15.7%），所以旧规则系统性地偏向对照组。

json 里带 `_order_invariant` 后缀的键是新口径，**HotpotQA / 2Wiki 正文应以它为准**。

---

## 结果速查

### 任务 1a：四类互斥结局（main run）

| 数据集 | 判定 | A 答对 | B 跳数错 | C 检索有错 | D 检索全对却答错 |
|---|---|---|---|---|---|
| MuSiQue | 两种相同 | 1,066（44.1%） | 268（11.1%） | 607（25.1%） | 476（19.7%） |
| 2Wiki | 按位置 | 7,563（60.1%） | 33（0.3%） | 1,136（9.0%） | 3,844（30.6%） |
| **2Wiki** | **顺序无关** | 7,563（60.1%） | 33（0.3%） | 1,113（8.9%） | **3,867（30.8%）** |
| HotpotQA | 按位置 | 4,084（55.2%） | 410（5.5%） | 1,496（20.2%） | 1,415（19.1%） |
| **HotpotQA** | **顺序无关** | 4,084（55.2%） | 410（5.5%） | **1,115（15.1%）** | **1,796（24.3%）** |

**HotpotQA 上 381 题（5.1 个点）从 C 移到 D**：它们只是把两个支持段落换序取了。订正后 HotpotQA
也变成「生成侧失败多于检索侧」（24.3% vs 15.1%），与 2Wiki 同向。2Wiki 只动了 23 题。

MuSiQue 按标注跳数：4 跳题的 **36.3%** 落在 B（跳数错），比检索错（27.9%）还多。

### 任务 1c/1d：D 类的四档互斥拆解

四档互斥、按顺序判定：⓪ 弃答 `NA` → ① 只是措辞不同 → ② 重复某个**非末跳**的答案 → ③ 其他。
D 类按顺序无关口径取。

**①「措辞不同」报成区间，因为宽松规则只是上界。** 宽松规则 = 规范化后一方是另一方的子串，或
token F1 ≥ 0.5；它会把「答错了另一个问题」也算进来：

| 例子 | prediction | gold | 真实情况 |
|---|---|---|---|
| `2hop__62369_84616` | `Ondine` | `for Best Performance by a Leading Actress in a Play in Ondine` | **答错**（问奖项，答剧名）|
| HotpotQA | `Audi Quattro` | `Audi` | 两可 |
| `2hop__669837_31270` | `45 miles` | `45` | 确实只是措辞 |

严格规则 = 双向 token F1 ≥ 0.8，不会命中上面前两类，作为下界。

| 数据集 | D 类 n | ⓪ 弃答 NA | ① 措辞不同（下界–上界） | ② 重复中间跳答案 | ③ 其他 |
|---|---|---|---|---|---|
| MuSiQue | 476 | 6.7% | 8.6% – 41.0% | **6.5%** | 45.8% |
| 2Wiki | 3,867 | 3.1% | 7.5% – 39.2% | **1.5%** | 56.3% |
| HotpotQA | 1,796 | 0.4% | 16.3% – 57.4% | **8.2%** | 33.9% |

**没做人工核对前，正文只能写「最多 41% / 39% / 57%」或写区间。**
`results/bucketD_wording_audit_sample.jsonl` 已按 `random.Random(0)` 从每个数据集的①里抽了 50 条，
字段 `question / prediction / gold / rule / f1 / verdict_fill_in`；人工填完 `verdict_fill_in`
（`ok` 或 `wrong`）即可得到规则准确率 p，正文改成「约 41%×p」。**这一步还没做。**

同一批链换 `default` prompt 重放后的「重复中间跳答案」：2Wiki **13.3%**（vs hint 的 1.5%），
HotpotQA 12.5%（vs 8.2%），MuSiQue 6.9%（几乎不变——它没有比较类题）。
与 0817 手工统计的 13.1% → 2.6% 吻合。

> 弃答 `NA` 以前被算进①、并且在某个中间跳也是 `NA` 时还会误判成②，1024 已拆成独立一档。
> 这让 MuSiQue 的②从 8.0% 降到 6.5%、2Wiki 从 1.9% 降到 1.5%。

### 任务 1b：NA 级联

NA 级联的题**可以落在任何一类**，所以两个方向的分母都列出来。
（本节原来那一列「占 C 类 23.4% / 12.3% / 14.6%」是转抄错误，json 里从来没有这三个数。）

| 数据集 | 出现 NA 级联 | 落在 A | 落在 B | **落在 C** | 落在 D | **占 C 类的比例** |
|---|---|---|---|---|---|---|
| MuSiQue | 261（10.8%） | 33.0% | 15.3% | 34.5% | 17.2% | **14.8%** |
| 2Wiki | 337（2.7%） | 42.4% | 0.6% | 15.7% | 41.3% | **4.8%** |
| HotpotQA | 454（6.1%） | 53.3% | 6.0% | 19.4% | 21.4% | **7.9%** |

A–D 四列的分母是**该数据集的 NA 级联题数**（四列相加 = 100%）；最后一列的分母是**C 类题数**。
HotpotQA 超过一半的 NA 级联题最后仍然答对，所以 NA 不等于失败。C 类按顺序无关口径取；
按位置口径下最后一列是 2Wiki 4.8%、HotpotQA 6.8%。

跳命中率：

| 数据集 | 判定 | NA 之后 | 无 NA 题 | 差 |
|---|---|---|---|---|
| MuSiQue | 两种相同 | **0.4978** | 0.7222 | −22.4 点 |
| 2Wiki | 按位置 | 0.8190 | 0.9318 | −11.3 点 |
| | 顺序无关 | 0.8234 | 0.9455 | −12.2 点 |
| HotpotQA | 按位置 | 0.6327 | 0.6628 | −3.0 点 |
| **HotpotQA** | **顺序无关** | **0.8013** | **0.8461** | **−4.5 点** |

HotpotQA 的绝对命中率两种判定差 17~18 个点，再一次说明那边必须用顺序无关口径。

### 任务 1e：gate 改变了什么

在第一个两者选择不同的跳上判定：

| 数据集 | 判定 | 组 | n | gate 纠正了错误 | gate 压下了正确第一名 | 两者都不是 gold | 选择相同 |
|---|---|---|---|---|---|---|---|
| MuSiQue | 两种相同 | gate 独赢 | 355 | **262** | 35 | 56 | 2 |
| | | control 独赢 | 84 | 15 | **43** | 22 | 4 |
| 2Wiki | 按位置 | gate 独赢 | 1,170 | 1,097 | 14 | 59 | 0 |
| | | control 独赢 | 162 | 62 | 60 | 39 | 1 |
| **2Wiki** | **顺序无关** | gate 独赢 | 1,170 | **1,123** | 9 | 38 | 0 |
| | | control 独赢 | 162 | 74 | **51** | 36 | 1 |
| HotpotQA | 按位置 | gate 独赢 | 847 | 528 | 129 | 190 | 0 |
| | | control 独赢 | 172 | 68 | 55 | 49 | 0 |
| **HotpotQA** | **顺序无关** | gate 独赢 | 847 | **713** | **14** | 120 | 0 |
| | | control 独赢 | 172 | 48 | **50** | 74 | 0 |

**HotpotQA「gate 压下了正确第一名」129 → 14**：按位置口径下那 129 题最后全答对了，本身就反常
——gate 多半只是先取了另一个支持段落。正文用顺序无关的 14。新口径下②「gate 纠正了错误」反而
**变大**（528 → 713），因为 λ=0 对照重复取同一篇 gold 的比例（29.3%）远高于 main（19.2%），
按位置口径把这类错误也归到③里，掩盖了 gate 的功劳。**损失侧**的「压下正确第一名」两种判定都在
29%–51%，没有被口径救回去——gate 的误报是真的。

### 任务 2：自然错误下的前缀污染

| 组 | K | 错误模式 | 转移 | n | Δz（95% CI） | Δgate（95% CI） |
|---|---|---|---|---|---|---|
| 错误之后的正确跳 | 2 | (1,) | 2 | 91 | +0.75 [+0.52,+0.97] | **+0.239 [+0.174,+0.305]** |
| 错误之后的正确跳 | 3 | (2,) | 3 | 94 | +0.91 [+0.67,+1.16] | **+0.208 [+0.154,+0.264]** |
| 错误发生的那一跳 | 2 | (2,) | 2 | 389 | +1.02 [+0.88,+1.15] | +0.538 [+0.503,+0.574] |
| 错误发生的那一跳 | 3 | (3,) | 3 | 110 | +0.45 [+0.26,+0.63] | +0.508 [+0.446,+0.568] |
| 错误之前的跳 | 2 | (2,) | 1 | 389 | −0.06 [−0.18,+0.07] | **+0.056 [+0.028,+0.085]** |
| 错误之前的跳 | 3 | (3,) | 1 | 110 | −0.17 [−0.33,+0.00] | +0.023 [−0.004,+0.054] |
| 错误之前的跳 | 3 | (3,) | 2 | 110 | +0.28 [+0.09,+0.46] | **+0.096 [+0.047,+0.145]** |

K=3 + 错误模式 (1,) 只有 10 条链，json 里标了 `do_not_interpret`。

> 任务 2 的输入是 MuSiQue GT 分解构造的 trace，**只有 MuSiQue**，不涉及上面的 gold 顺序问题；
> 图 5.1 / 5.2 同理（它们的输入是逐跳 gate 分数，与 gold 段落的标注顺序无关），1024 的订正
> 不影响这两张图，无需重画。

**两条结论**：(a) 自然错误下也有前缀污染，错误之后本身选对的跳分数高 +0.21~+0.24，与构造错误的
发现一致，论文写 "consistent with"；(b) **存在较小但真实的选择效应**——错误之前的跳也略高
（+0.02~+0.10，两格 CI 不含 0），量级是污染效应的 1/2~1/10，论文里要写出来并相应收紧措辞。

### 任务 4：口径订正（已改进 WORK_SUMMARY 3.12）

- **2Wiki 的 gold 跳数**有三种可能定义，检索代码用的是 `supporting_facts` **去重标题数**
  （2: 9,825 / 4: 2,751）；3.12 原来写的 `2: 9,595 / 3: 88 / 4: 2,806 / 5+: 87` 是
  **`evidences` 三元组数**，与同表的「预测跳数分布」不可比。已统一。
- **k_match 在新 main run 上复算**：0.8382 / 0.9866 / 0.8763，与 6 月旧 run 逐位相同，
  确认「只取决于分解器」。

---

## 图

| 文件 | 尺寸 | 内容 |
|---|---|---|
| `disc_score_vs_error_position.pdf` | 9.00 × 5.98 cm | gate 分数 vs 相对第一个错误的位置；两条线（全部跳 / 仅错误之后本身正确的跳），误差条为 2,000 次 bootstrap 的 95% CI，两条参照线为全 gold 链均值 0.19 与按位置加权的判定阈值 0.36 |
| `disc_logit_decomposition.pdf` | 16.00 × 5.49 cm | logit 拆成 `w_h·PCA(h)` 与 `w_Δ·PCA(Δ)` 两半，五个类别分组条形图 |

规格与第 4 章一致：8 pt 字号、PDF、图内无总标题（说明交给图注）、宽度精确且四边留 1 mm。

---

## 附：gold 顺序口径（v2 协议，1011）

第 4 章已经决定一起换到顺序无关判定（`ORDER_INVARIANT_TASK.md`），全量 32 个 run 的重算在
`retrieval/results/order_invariant_v2.json` 和 WS 3.14。这里只留第 5 章用得到的那一小块
（`results/order_invariant_metrics.json`，main = λ=0.60，control = λ=0）：

| 数据集 / 组 | EM | chain@1 按位置 | chain@1 顺序无关 | recall@1 按位置 | recall@1 顺序无关 | 重复取 gold |
|---|---|---|---|---|---|---|
| MuSiQue gate | 0.4410 | 0.4977 | 0.4977 | 0.6601 | 0.6601 | 0 |
| MuSiQue control | 0.3289 | 0.3451 | 0.3451 | 0.5745 | 0.5745 | 0 |
| **Δ（点）** | +11.21 | +15.26 | +15.26 | +8.56 | +8.56 | |
| 2Wiki gate | 0.6014 | 0.8507 | **0.8623** | 0.9292 | **0.9426** | 474 |
| 2Wiki control | 0.5212 | 0.6578 | **0.6640** | 0.8325 | **0.8414** | 908 |
| **Δ（点）** | +8.02 | +19.29 | **+19.83** | +9.67 | **+10.12** | |
| HotpotQA gate | 0.5515 | 0.4974 | **0.6262** | 0.6614 | **0.8427** | 1,425 |
| HotpotQA control | 0.4604 | 0.3390 | **0.4189** | 0.5739 | **0.7072** | 2,168 |
| **Δ（点）** | +9.11 | +15.84 | **+20.73** | +8.75 | **+13.55** | |

四条结论：

1. **MuSiQue 两列逐位相同**（顺序无关不对它启用）——这同时就是脚本的自检。
2. **HotpotQA 的绝对值差 +12.9 点**（chain@1 0.4974 → 0.6262），recall@1 差 +18.1 点。
   2Wiki 差 +1.2 / +1.3 点。
3. **方向不变，而且 gate 的优势变大**：HotpotQA 的 chain@1 增益从 +15.84 涨到 **+20.73**，
   recall@1 从 +8.75 涨到 **+13.55**。原因见下一条。
4. **「重复取同一篇 gold」是个独立发现**：λ=0 对照里 HotpotQA 29.3% / 2Wiki 15.7% 的题会在两跳
   取同一篇 gold，main run 只有 19.2% / 8.2%——**gate 压低了约 10 和 7 个百分点**。第一版顺序无关
   实现（只判 `∈ set(gold_idxs)`）恰好把这类错误放过了，所以它系统性地偏向对照组，把 gate 的增益
   压成了 +9.44。那个数是错的，不要再用。

EM / F1 在任何口径下都不变，它们根本不读 `gold_idxs`。旧的按位置脚本
`retrieval/compute_strict_chain.py` 按任务书要求**没有改动**，旧数字保留以便对照。

## 注意

`results/gate_l15_hops_dev.jsonl` 有 **12.9 MB**，是中间产物（可由上面的命令重新生成），
已加进根 `.gitignore`，不进版本库。其余 json 都是 KB 级，正常提交。

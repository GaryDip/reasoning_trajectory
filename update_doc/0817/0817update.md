# 2026-08-17 更新:reader prompt 修复(比较类问题答案错位)

## 一、问题发现:翻具体案例,而不是只看聚合指标

在 gate 三信号对照实验(0810 更新)完成之后,对 D 组(特征级融合,MuSiQue dev 全量)做了一次
案例分析,而不是继续在聚合指标上调参。先把"答错的案例"按"检索是否全对"拆开:

| | 数量 |
|---|---|
| 答错(EM=0) | 1376 |
| 其中:每一跳都选中了 gold 段落,但最终答案依然错 | 485 |

这 485 个"检索/gate 全部做对、答案还是错"的案例,跟检索和 gate 完全无关,问题出在最后一步
"综合所有中间结果生成答案"的 reader 身上。把这 485 个案例的最终答案,跟每一跳自己生成的短
答案(`prior`)逐个比对,分成三类:

| 类型 | 数量 | 占比 |
|---|---|---|
| 最终答案 = 最后一跳短答案,但 EM 判定仍为错(纯粹是措辞跟 gold 不完全一致,如 "McKinley" vs "President McKinley") | 372 | 77% |
| 最终答案 = **更早一跳**的短答案(reader 真的选错了该用哪一跳的结论) | 39 | 8% |
| 两者都不匹配(reader 自己重新综合出一个新的错误答案) | 74 | 15% |

第一类是 EM 指标本身的严格性导致的,不是真实错误。第二、三类才是真问题,合计约 23%。

在 2WikiMultihopQA、HotpotQA 上重复同样的分析,发现"reader 选错跳"这个模式**不是 MuSiQue
特有的,而且在另外两个数据集上比例更高**:

| 数据集 | 检索全对但答错 | 其中"reader 选错跳" | 占比 |
|---|---|---|---|
| MuSiQue | 485 | 39 | 8.0% |
| 2WikiMultihopQA | 4894 | 639 | 13.1% |
| HotpotQA | 1522 | 232 | 15.2% |

进一步看 2WikiMultihopQA、HotpotQA 里的具体案例,发现"选错跳"经常集中在**比较类问题**上
(2WikiMultihopQA 数据集里专门有一个 `comparison` 问题类型),而且不是简单的"选了错误的那一
跳",是 reader 根本没有执行"比较"这个动作,直接把某一跳查到的原始事实(通常是日期)当成
答案输出:

```
Q: Which film was released more recently, Royal Treasure or When Love Begins?
gold: Royal Treasure   |   final: January 15, 2016
每跳短答案(prior): ['January 15, 2016', '2008']

Q: Did the board game San Marco or About Time come out first?
gold: San Marco   |   final: 2001
每跳短答案(prior): ['2001', '2007']
```

两跳分别查出了两部电影/两款桌游各自的日期,但最终该输出的是"哪一个"(片名/游戏名),
reader 却直接把其中一个日期原样吐出来了。

## 二、根因与修复:reader prompt 缺一句"比较类问题"的显式指令

最终答案生成用的 prompt 由 `retrieval/run_retrieval_exp_wavefront.py::build_final_reader_cot_prompt`
构造,其中唯一约束答案格式的一句是:

```
Give a short span or phrase only (entity, date, number, or yes/no).
```

这句话把"date"(日期)"number"(数字)明确列为**合法的答案类型**,对所有问题一视同仁。
对大多数桥接类问题(bridge)来说,"最后一跳的短答案就是最终答案"这个假设是对的;但对比较类
问题来说,每一跳查到的日期/数字只是**用来做比较判断的中间量**,真正该输出的是比较之后胜出
的那个实体名字——prompt 里完全没有区分这两种情况,reader 于是经常直接把某一跳的日期当答案。

修复是在 `run_retrieval_exp_wavefront.py` 里新增一个函数
`build_final_reader_cot_prompt_comparison_hint`(**保留原函数不动**,新函数追加在旁边),
在原 prompt 基础上只加一句话:

```
If the original question asks you to COMPARE two or more entities (for example "which was
released more recently", "who was born first", "which is longer"), the per-step answers
above are the values used to make that comparison, not the final answer themselves -- you
must perform the comparison yourself and answer with the NAME of the entity that satisfies
it, not a date, number, or other raw value.
```

`retrieval/run_retrieval_exp_wavefront_gate_v3_rawprefix.py` 新增 `--final-reader-prompt
{default,comparison_hint}` 参数,可以在两个版本之间切换,其余流程(检索、gate 打分、逐跳
候选选择)完全不变——保证这是一次干净的单变量对照。

## 三、结果:2WikiMultihopQA 全量验证

MuSiQue dev、λ=0.50、rawprefix,只切 `--final-reader-prompt`:

| | recall@1 | chain@1 | EM | F1 |
|---|---|---|---|---|
| default prompt | 0.9289 | 0.8613 | 0.4990 | 0.5759 |
| **comparison_hint** | 0.9289 | 0.8613 | **0.5982** | **0.6770** |
| 差值 | 持平 | 持平 | **+9.92** | **+10.11** |

recall@1、chain@1 完全没变——检索和 gate 全都没动,证明提升干干净净来自这一句 prompt 改动。
EM/F1 涨了约 10 个百分点,是这个项目目前为止单次改动里最大的一次提升,超过 λ 调参、rawprefix
文本对齐、BGE 检索前缀这几项加起来的总和。

同时验证了根因诊断是对的:"reader 选错跳"这个 bug 的复现率从 13.1%(639/4894)降到
**2.6%**(98/3833),降了 5 倍。

## 四、MuSiQue、HotpotQA 验证——已完成

三个数据集全部验证完,recall@1/chain@1 在每个数据集上都跟 default prompt 完全持平(检索、
gate 全都没动,再次确认改动干净),EM/F1 全部正向:

| 数据集 | recall@1 | chain@1 | EM(default→comparison_hint) | F1(default→comparison_hint) |
|---|---|---|---|---|
| MuSiQue | 0.7018(持平) | 0.5143(持平) | 0.4307→0.4427(+1.20) | 0.5270→0.5386(+1.16) |
| 2WikiMultihopQA | 0.9289(持平) | 0.8613(持平) | 0.4990→0.5982(**+9.92**) | 0.5759→0.6770(**+10.11**) |
| HotpotQA | 0.6596→0.6595(持平) | 0.5568→0.5567(持平) | 0.5217→0.5517(+3.00) | 0.6517→0.6870(+3.53) |

涨幅排序是 2wiki > hotpot > musique,不完全等于"选错跳"bug 在受影响样本里的**占比**排序
(hotpot 15.2% 比 2wiki 13.1% 更高)——原因是 hotpot 检索基线本身更低(recall@1 只有
66%),"检索全对"这个前提覆盖的样本更少,受影响的**绝对样本数占全量的比例**才是决定整体
EM 涨多少的关键:2wiki 是 639/12576=5.1% 的全量样本受影响,hotpot 只有 232/7405=3.1%。

三个数据集方向一致、全部正向,没有一个数据集下降,`comparison_hint` 定为新的默认 prompt。

## 五、后续探索——已尝试，未采纳，`comparison_hint` 为最终版本

在 `comparison_hint` 基础上继续案例分析（2WikiMultihopQA 上"检索全对但答错"的残留案例），发现
两类 `comparison_hint` 没能覆盖的问题，各自尝试了一版新 prompt 修复，用 `replay_final_reader.py`
（复用已有 case 文件里的 `prior`/`para_ids` 重建每跳输入，只重跑最终答案这一步，不用重跑检索/
gate/逐跳短答案，比端到端快很多）在三个数据集上验证，两版都是净负，未采纳：

- **`prompt_short_answer_with_context_type_match`**（逐跳短答案，约束答案类型匹配疑问词）：
  MuSiQue 上有提升（EM+1.94/F1+2.32，叠加 comparison_hint 之后），但 2wiki 基本打平、检索侧
  还小幅下降，HotpotQA 基本持平，三数据集不一致，未采纳。
- **`build_final_reader_cot_prompt_comparison_reasoning`**（最终答案前先显式写一步推理）：
  2wiki 上有提升（EM+2.65/F1+2.31），但 MuSiQue（EM-1.32/F1-1.29）、HotpotQA（EM-2.35/F1-2.92）
  都是净负。案例分析发现根因：模型把 `comparison_hint` 里"答案要是实体名字，不能是日期/数字"
  这句话当成了无差别适用的规则，只要推理里同时出现名字和日期就套用，即使原问题根本不是比较类
  问题（典型案例：模型正确推出答案是"June 1982"，又因为这条规则强行改答成了人名"Diego
  Maradona"）。
- **`build_final_reader_cot_prompt_comparison_positive`**（把上面那条规则改成正向表述、去掉
  举例）：三个数据集全部净负，且 2wiki 降幅最大（EM-3.05/F1-3.12）。案例分析发现根因相反：
  去掉举例之后模型识别不出"这是比较类问题"，触发率暴跌，直接退回到 `comparison_hint` 之前的
  老问题（原样输出某一跳的日期，不做比较）。

结论：`comparison_hint` 现有的措辞（含举例、含"不是日期/数字"这句）在**触发识别**和**过度触发**
之间已经是一个还不错的平衡点，两次尝试分别只改动其中一半，都打破了这个平衡、变得更差。继续在
这一条 prompt 上做局部修补收益递减，`comparison_hint` 定为最终版本，不再继续探索这个方向。

下一步是回填 `gate_v3_main_method_report.md` 里所有引用"当前完整方法"的数字,并考虑给 B 组
(gate v2 rawprefix)、C 组(h-only rawprefix)也接上这个新 prompt 重新跑一遍。

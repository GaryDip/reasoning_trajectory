#!/usr/bin/env python3
"""Per-question and one-time compute for the three systems, from measured token counts.

Inputs come from cost/results/measured_tokens.json (written by measure_tokens.py: real tokenizer,
real prompt builders, real candidate lists, and ChainRAG's own per-question `llm_calls` field).
Only the handful of quantities listed under "assumed" there are estimates, and each says why.

Accounting: OUR system as the code actually runs it (the layer-16 truncation is implemented; the
prefix-KV reuse is not), the baselines as their released code runs.

FLOPs use 2 * N * T, with N the parameters actually multiplied -- embedding lookups are not
matmuls, and the gate has no lm_head after truncation, so N is given per role rather than as the
checkpoint's headline size:

  Llama-3.1-8B: 8.03 B total = 0.525 (embed) + 0.525 (lm_head) + 32 x 0.218 (blocks)
    generation prefill   6.98 B   (blocks; vLLM only projects logits for the last position)
    generation decode    7.51 B   (blocks + lm_head, per generated token)
    gate, truncated      3.49 B   (16 blocks, lm_head replaced by Identity)
    gate, before ②       7.51 B   (32 blocks + lm_head, which ForCausalLM computed and discarded)
  GritLM-7B (Mistral-7B): 7.24 B total; 6.98 B of blocks, used as an encoder (no lm_head)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

N_GEN_PREFILL, N_GEN_DECODE = 6.98e9, 7.51e9
N_GATE_TRUNC, N_GATE_FULL = 3.49e9, 7.51e9
N_GRIT_ENC = 6.98e9
N_BGE, N_RERANK, N_BART = 0.109e9, 0.56e9, 0.406e9

MEM_GB = {"gate_trunc": 8.0, "gate_full": 16.1, "vllm_llama8b": 16.1,
          "grit7b": 14.5, "bge": 0.22, "rerank": 1.1, "bart": 0.8}

KDIST = {2: 1252, 3: 760, 4: 405}
K_MEAN = sum(k * n for k, n in KDIST.items()) / sum(KDIST.values())
N_PASSAGES, POOL = 20, 10
# dev sizes and EM per method, for the total-cost and cost-per-correct-answer accounting.
# EM comes from WORK_SUMMARY.md section 3.6 (the three-way comparison; all three systems answer
# with the same local Llama-3.1-8B-Instruct, HotpotQA-official scoring).
N_DEV = {"musique": 2_417, "2wiki": 12_576, "hotpot": 7_405}
N_DEV_ALL = sum(N_DEV.values())
EM = {"我们": {"musique": 0.4410, "2wiki": 0.6014, "hotpot": 0.5515},
      "GRITHopper": {"musique": 0.2793, "2wiki": 0.5390, "hotpot": 0.5546},
      "ChainRAG": {"musique": 0.2764, "2wiki": 0.4426, "hotpot": 0.4598}}

# phase A, from the repo (see REPORT.md section 7). Two scopes:
#   "reproduce" -- what building the final method needs: train split only. The dev extractions and
#                  the dev GT decompositions are evaluation/analysis artefacts, and the 4-layer
#                  pilot was layer-selection exploration; none is needed to rebuild the method.
#   "all"       -- everything that was actually spent, including those.
N_TRACE_TRAIN, N_TRACE_DEV, PILOT_TRACES = 65_754, 8_821, 1_500 * 9
N_API_REPRODUCE = (19_938 + 5_562) + 10_444      # train GT + enhance + 2Wiki train annotation
N_API_ALL = N_API_REPRODUCE + 2_417 + 1_043      # + dev GT + 2Wiki dev annotation
BART_TOTAL_FLOS = 7.019624277344256e15          # trainer_state.json, real value
N_GLOBAL_PARAS = 21_094
# GRITHopper's paper (arXiv 2503.07519v2, App. I / Table 11): 8 x A100-80GB, 181 h = 1448 GPU h
A100_PEAK_BF16, GRIT_GPU_HOURS, GRIT_MFU = 312e12, 1448, (0.30, 0.50)
A_GRIT_RANGE = tuple(GRIT_GPU_HOURS * 3600 * A100_PEAK_BF16 * m for m in GRIT_MFU)
OURS_MFU = 0.40


def fwd(params: float, tokens: float) -> float:
    return 2 * params * tokens


def load_measured() -> dict:
    f = ROOT / "cost/results/measured_tokens.json"
    if not f.exists():
        raise SystemExit(f"missing {f} -- run cost/scripts/measure_tokens.py first")
    return json.loads(f.read_text())


def ours(m: dict, *, truncated: bool = True, prefix_kv_reuse: bool = False) -> dict:
    a = {k: v[0] for k, v in m["assumed"].items()}
    gate_tok = m["gate_tokens"]
    if prefix_kv_reuse:
        # each hop would pay the shared prefix once instead of once per candidate; the prefix is
        # what grows, so this is close to dividing the candidate part by the pool size
        gate_tok = gate_tok / POOL + m["gate_tokens"] * (1 - 1 / POOL) * 0.33
    gen_prefill = K_MEAN * m["hop_answer_prompt_tokens"] + m["final_reader_prompt_tokens"]
    gen_decode = K_MEAN * a["hop_answer_decode_tokens"] + a["final_reader_decode_tokens"]
    bge_tok = N_PASSAGES * m["passage_tokens"] + K_MEAN * (m["question_tokens"]
                                                           + m["sub_question_tokens"])
    return {"gate prefill": fwd(N_GATE_TRUNC if truncated else N_GATE_FULL, gate_tok),
            "生成 8B": fwd(N_GEN_PREFILL, gen_prefill) + fwd(N_GEN_DECODE, gen_decode),
            "检索 BGE": fwd(N_BGE, bge_tok),
            "分解 BART": fwd(N_BART, m["question_tokens"] + K_MEAN * m["sub_question_tokens"]),
            "_calls": K_MEAN + 1, "_fwd": m["gate_forwards"],
            "_serial": 1 + 3 * K_MEAN + 1,
            "_mem": (MEM_GB["gate_trunc"] if truncated else MEM_GB["gate_full"])
                    + MEM_GB["vllm_llama8b"] + MEM_GB["bge"] + MEM_GB["bart"]}


def grithopper(m: dict) -> dict:
    a = {k: v[0] for k, v in m["assumed"].items()}
    psg = N_PASSAGES * m["passage_tokens"]                       # encoded once per question
    qry = sum((m["question_tokens"] + (j - 1) * m["passage_tokens"]) * n / sum(KDIST.values())
              for K, n in KDIST.items() for j in range(1, K + 1))
    gen_prefill = K_MEAN * m["passage_tokens"] + m["question_tokens"] \
        + m["final_reader_system_tokens"]
    return {"检索 7B 编码": fwd(N_GRIT_ENC, psg + qry),
            "生成 8B": fwd(N_GEN_PREFILL, gen_prefill)
                       + fwd(N_GEN_DECODE, a["grithopper_answer_decode_tokens"]),
            "_calls": 1.0, "_fwd": 0.0, "_serial": K_MEAN + 1,
            "_mem": MEM_GB["grit7b"] + MEM_GB["vllm_llama8b"]}


def chainrag(m: dict, ds: str = "musique") -> dict:
    a = {k: v[0] for k, v in m["assumed"].items()}
    c = m["chainrag"][ds]
    calls, k_sub = c["llm_calls_mean"], c["sub_questions_mean"]
    pairs = k_sub * a["chainrag_rerank_pairs_per_hop"]
    return {"LLM 调用 8B": fwd(N_GEN_PREFILL, calls * a["chainrag_prompt_tokens"])
                           + fwd(N_GEN_DECODE, calls * 30),
            "重排 0.56B": fwd(N_RERANK, pairs * (m["sub_question_tokens"]
                                                 + a["chainrag_sentence_tokens"])),
            "句子嵌入 BGE": fwd(N_BGE, c["context_sentences_mean"]
                                 * a["chainrag_sentence_tokens"]),
            "_calls": calls, "_fwd": 0.0, "_serial": calls + 1,
            "_mem": MEM_GB["vllm_llama8b"] + MEM_GB["rerank"] + MEM_GB["bge"]}


def phase_a(m: dict, scope: str = "reproduce") -> dict:
    """One-time. A5 re-runs a full 8B forward per prefix (run_meta: full_forward_per_prefix)."""
    hop_step = m["sub_question_tokens"] + m["passage_tokens"]
    tot = sum(KDIST.values())
    tok_per_trace = sum(sum(m["question_tokens"] + j * hop_step for j in range(K + 2)) * n / tot
                        for K, n in KDIST.items())
    n_trace = N_TRACE_TRAIN if scope == "reproduce" else (N_TRACE_TRAIN + N_TRACE_DEV
                                                          + PILOT_TRACES)
    return {"A5 隐状态抽取": fwd(N_GATE_FULL, n_trace * tok_per_trace),
            "A3 BART 训练": BART_TOTAL_FLOS, "A6 gate 拟合": 0.0}


def chainrag_sensitivity(m: dict, ours_tflops: float) -> None:
    """ChainRAG's prompt length is the one input that is neither logged nor measurable offline,
    and 94.8% of its compute hangs on it. Report the sensitivity and the break-even length."""
    base = m["assumed"]["chainrag_prompt_tokens"][0]
    c = m["chainrag"]["musique"]
    fixed = total(chainrag(m)) - fwd(N_GEN_PREFILL, c["llm_calls_mean"] * base)
    print("\n—— ChainRAG 的 prompt 长度敏感性（它 94.8% 的算力取决于这一项）——")
    print("| 每次调用 prompt token | ChainRAG TFLOPs/题 | 相对我们 |")
    print("|---|---|---|")
    for t in (250, 510, 700, 1020, 1500):
        v = fixed + fwd(N_GEN_PREFILL, c["llm_calls_mean"] * t)
        mark = " ← 与我们持平" if abs(v / 1e12 - ours_tflops) < 2 else ""
        print(f"| {t:,} | {v/1e12:.0f} | {ours_tflops/(v/1e12):.2f}×{mark} |")
    be = (ours_tflops * 1e12 - fixed) / (2 * N_GEN_PREFILL * c["llm_calls_mean"])
    print(f"    盈亏点：prompt 平均短于 **{be:.0f} token** 时，ChainRAG 每题就比我们便宜")
    print(f"    结构下界：承载上下文的调用约 {c['context_sentences_mean']/c['sub_questions_mean']:.0f} "
          f"句 × 29.2 token/句 + 48 token 模板 ≈ "
          f"{c['context_sentences_mean']/c['sub_questions_mean']*29.2+48:.0f} token/次")


def total(d: dict) -> float:
    return sum(v for k, v in d.items() if not k.startswith("_"))


def show(name: str, d: dict):
    t = total(d)
    print(f"\n{name}  合计 {t/1e12:7.1f} TFLOPs | 生成类调用 {d['_calls']:5.2f} | "
          f"gate 前向 {d['_fwd']:5.2f} | 串行步 {d['_serial']:5.2f} | 常驻 {d['_mem']:.0f} GB")
    for k, v in sorted(((k, v) for k, v in d.items() if not k.startswith("_")),
                       key=lambda x: -x[1]):
        print(f"    {k:16s} {v/1e12:7.2f} TFLOPs  {v/t:5.1%}")


def appendix_kv_reuse(m: dict) -> None:
    """The counterfactual table for the thesis appendix: what the method costs if the shared
    prefix KV were reused across a hop's candidates. Everything else is identical -- the forward
    COUNT does not change (still 1 + sum|C_j|), only the tokens each forward processes."""
    o78, o43 = ours(m), ours(m, prefix_kv_reuse=True)
    g, c = grithopper(m), chainrag(m)
    a_ours = total(phase_a(m)) + fwd(N_BGE, N_GLOBAL_PARAS * m["passage_tokens"])
    lo = A_GRIT_RANGE[0]
    corr = {k: sum(N_DEV[d] * EM[k][d] for d in N_DEV) for k in EM}
    print("### 附录：采用 prefix KV 复用（①）后的对比\n")
    print("| 维度 | 我们（①后） | 我们（主表，①前） | GRITHopper | ChainRAG |")
    print("|---|---|---|---|---|")
    rowsets = [("推理算力 TFLOPs/题", lambda d: f"{total(d)/1e12:.0f}"),
               ("⤷ gate / 检索器占比", None),
               ("生成类 LLM 调用/题", lambda d: f"{d['_calls']:.2f}"),
               ("gate 前向（仅 prefill）/题", lambda d: f"{d['_fwd']:.1f}"),
               ("串行步/题", lambda d: f"{d['_serial']:.1f}"),
               ("常驻权重 GB", lambda d: f"{d['_mem']:.0f}")]
    main_part = {id(o43): "gate prefill", id(o78): "gate prefill",
                 id(g): "检索 7B 编码", id(c): "LLM 调用 8B"}
    for label, fn in rowsets:
        if fn is None:
            cells = [f"{d[main_part[id(d)]]/1e12:.1f}（{d[main_part[id(d)]]/total(d):.0%}）"
                     for d in (o43, o78, g, c)]
        else:
            cells = [fn(d) for d in (o43, o78, g, c)]
        print(f"| {label} | " + " | ".join(cells) + " |")
    tots, pers = {}, {}
    for name, d, a in (("o43", o43, a_ours), ("o78", o78, a_ours),
                       ("g", g, lo), ("c", c, 0.0)):
        k = "我们" if name.startswith("o") else ("GRITHopper" if name == "g" else "ChainRAG")
        tots[name] = a + total(d) * N_DEV_ALL
        pers[name] = tots[name] / corr[k]
    print(f"| 总算力（{N_DEV_ALL:,} 题） | " +
          " | ".join(f"{tots[n]/1e15:,.0f} PF" for n in ("o43", "o78", "g", "c")) + " |")
    print("| 每个正确答案 | " +
          " | ".join(f"{pers[n]/1e15:.3f} PF" for n in ("o43", "o78", "g", "c")) + " |")
    for n, nm in (("g", "GRITHopper"), ("c", "ChainRAG")):
        print(f"| ⤷ 我们/{nm}（总算力 ÷ 每正确答案） | "
              f"{tots['o43']/tots[n]:.3f}× / {pers['o43']/pers[n]:.3f}× | "
              f"{tots['o78']/tots[n]:.3f}× / {pers['o78']/pers[n]:.3f}× | — | — |")
    print("\n每个正确答案的推理算力（TFLOPs，不含训练）\n")
    print("| 数据集 | 我们（①后） | 我们（①前） | GRITHopper | ChainRAG |")
    print("|---|---|---|---|---|")
    for d in N_DEV:
        print(f"| {d} | **{total(o43)/EM['我们'][d]/1e12:.0f}** | "
              f"{total(o78)/EM['我们'][d]/1e12:.0f} | "
              f"{total(g)/EM['GRITHopper'][d]/1e12:.0f} | "
              f"{total(c)/EM['ChainRAG'][d]/1e12:.0f} |")
    for name, c_other, a_other in (("ChainRAG", total(c), 0.0),
                                   ("GRITHopper（MFU 30%）", total(g), lo),
                                   ("GRITHopper（MFU 50%）", total(g), A_GRIT_RANGE[1])):
        n43 = (a_ours - a_other) / (c_other - total(o43))
        n78 = (a_ours - a_other) / (c_other - total(o78))
        print(f"\nN* vs {name}：①后 {n43:,.0f} 题；①前 {n78:,.0f} 题")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--appendix-kv-reuse", action="store_true",
                    help="只输出附录用的「采用 ① 之后」对比表（markdown）")
    args = ap.parse_args()
    m = load_measured()
    if args.appendix_kv_reuse:
        appendix_kv_reuse(m)
        return
    o, g, c = ours(m), grithopper(m), chainrag(m)

    print(f"按 MuSiQue dev 跳数分布加权（K=2:1252 / 3:760 / 4:405，均值 {K_MEAN:.2f}），每题")
    print("口径：我们 = 代码实际运行（②层截断已实现，①prefix KV 复用未实现）；baseline = 其发布实现\n")
    print("| 方法 | TFLOPs/题 | 生成类调用 | 串行步 | 常驻权重 GB |")
    print("|---|---|---|---|---|")
    for name, d in (("我们", o), ("GRITHopper", g), ("ChainRAG", c)):
        print(f"| {name} | **{total(d)/1e12:.0f}** | {d['_calls']:.2f} | {d['_serial']:.1f} | "
              f"{d['_mem']:.0f} |")
    for name, d in (("我们", o), ("GRITHopper", g), ("ChainRAG", c)):
        show(name, d)
    show("我们（若回退 ②，即不截断）", ours(m, truncated=False))
    show("我们（若再做 ①，未实现，粗估）", ours(m, prefix_kv_reuse=True))

    a = phase_a(m)
    ta = total(a)
    ta_all = total(phase_a(m, scope="all"))
    per_q = total(o)
    print("\n—— 阶段 A：一次性训练（口径：复现最终方法所需）——")
    print(f"我们  合计 {ta/1e15:8.1f} PFLOPs | 外部 API 调用 {N_API_REPRODUCE:,}（闭源，无 FLOPs）")
    for k, v in sorted(a.items(), key=lambda x: -x[1]):
        print(f"    {k:16s} {v/1e15:8.2f} PFLOPs  {v/ta:5.1%}")
    print(f"  折算 = {ta/per_q:,.0f} 道题的推理 = MuSiQue dev（2,417 题）的 {ta/per_q/2417:.1f} 倍")
    lo, hi = A_GRIT_RANGE
    print(f"  GRITHopper：1,448 A100·小时（论文表 11），MFU {GRIT_MFU[0]:.0%}~{GRIT_MFU[1]:.0%} "
          f"→ {lo/1e15:,.0f}~{hi/1e15:,.0f} PFLOPs = 我们的 {lo/ta:,.0f}~{hi/ta:,.0f} 倍")
    print("  ChainRAG：0")
    print(f"  注：若按「实际付出」（含 dev 抽取 {N_TRACE_DEV:,} 条 + 层选择 pilot "
          f"{PILOT_TRACES:,} 条 + dev 标注），我们是 {ta_all/1e15:,.0f} PFLOPs、"
          f"API {N_API_ALL:,} 次；主表用「复现所需」。")

    b = fwd(N_BGE, N_GLOBAL_PARAS * m["passage_tokens"])
    print(f"\n—— 阶段 B ——\n    global 语料 BGE 嵌入（{N_GLOBAL_PARAS:,} 段）  {b/1e15:.2f} PFLOPs")

    print("\n—— 阶段 D：摊销（总成本 = A + B + C·N）——")
    print("    N* 解 A_ours + C_ours·N = A_other + C_other·N，即 "
          "N* = (A_ours − A_other) / (C_other − C_ours)")
    a_ours = ta + b
    for name, c_other, a_other in (("ChainRAG", total(c), 0.0),
                                   ("GRITHopper（MFU 30%）", total(g), lo),
                                   ("GRITHopper（MFU 50%）", total(g), hi)):
        if c_other == per_q:
            print(f"    vs {name}：每题开销相同，差距恒为训练开销之差")
            continue
        nstar = (a_ours - a_other) / (c_other - per_q)
        if nstar < 0:
            cheaper = "我们" if (a_ours < a_other and per_q < c_other) else "对方"
            print(f"    vs {name}：无正解 —— {cheaper}在任何 N 下总成本都更低")
            continue
        if per_q < c_other:        # 我们每题更省，但训练更贵：要量才能摊平
            msg = (f"超过 N* 我们总成本更低；三数据集 dev 合计 {N_DEV_ALL:,} 题，"
                   f"{'已超过' if N_DEV_ALL > nstar else '**未超过**'}")
        else:                      # 我们每题更贵，但训练远便宜：量大之后对方才赢
            msg = (f"**N* 以内我们总成本更低**；三数据集 dev 合计 {N_DEV_ALL:,} 题，"
                   f"{'在范围内' if N_DEV_ALL < nstar else '已超出'}")
        print(f"    vs {name}：N* = {nstar:,.0f} 题 —— {msg}")

    report_totals({"我们": per_q, "GRITHopper": total(g), "ChainRAG": total(c)},
                  {"我们": a_ours, "GRITHopper": lo, "ChainRAG": 0.0})
    chainrag_sensitivity(m, per_q / 1e12)


def report_totals(per_q: dict[str, float], a_phase: dict[str, float]) -> None:
    """Total cost at the evaluation scale, and cost per CORRECT answer.

    Per-question compute on its own rewards a system for answering cheaply and wrongly; dividing
    by the number of questions it actually gets right is the standard cost-effectiveness view and
    is the only one in which the three systems are compared at equal output.
    """
    print(f"\n—— 总账：评测规模 N = {N_DEV_ALL:,} 题（三数据集 dev 合计）——")
    print("| 方法 | 阶段A 训练 | 推理 C·N | 总计 | 答对题数 | 每个正确答案 |")
    print("|---|---|---|---|---|---|")
    out: dict[str, tuple[float, float]] = {}
    for k, c in per_q.items():
        inf = c * N_DEV_ALL
        tot = a_phase[k] + inf
        corr = sum(N_DEV[d] * EM[k][d] for d in N_DEV)
        out[k] = (tot, corr)
        print(f"| {k} | {a_phase[k]/1e15:,.0f} PF | {inf/1e15:,.0f} PF | "
              f"**{tot/1e15:,.0f} PF** | {corr:,.0f} | **{tot/corr/1e15:.3f} PF** |")
    t_o, c_o = out["我们"]
    for k in ("GRITHopper", "ChainRAG"):
        t_x, c_x = out[k]
        print(f"    vs {k:11s} 总算力 {t_o/t_x:7.3f}×   每正确答案 {(t_o/c_o)/(t_x/c_x):.3f}×")

    print("\n—— 只看推理（不含训练）：每个正确答案的算力 TFLOPs ——")
    print("| 数据集 | " + " | ".join(per_q) + " |")
    print("|---" * (len(per_q) + 1) + "|")
    for d in N_DEV:
        print(f"| {d} | " + " | ".join(f"{per_q[k]/EM[k][d]/1e12:.0f}" for k in per_q) + " |")


if __name__ == "__main__":
    main()

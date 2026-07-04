"""Parse reasoning traces into cumulative prefix strings for hidden-state extraction."""

from __future__ import annotations

import re


def consume_evidence_quote(s: str, after_evidence_mark: int) -> tuple[str, int]:
    i = after_evidence_mark
    out: list[str] = []
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            out.append(s[i + 1])
            i += 2
        elif s[i] == '"':
            return "".join(out), i
        else:
            out.append(s[i])
            i += 1
    raise ValueError("unterminated Evidence string")


def parse_trace_structure(trace: str) -> tuple[str, list[str], str]:
    if " Final Answer: " not in trace:
        raise ValueError("missing ' Final Answer: '")
    head, ans = trace.rsplit(" Final Answer: ", 1)
    ans = ans.strip()
    if not head.startswith("Question:"):
        raise ValueError("trace must start with Question:")
    s1 = head.find("Step 1:")
    if s1 < 0:
        raise ValueError("missing Step 1")
    q_part = head[len("Question:") : s1].strip()
    rest = head[s1:].strip()

    blocks: list[str] = []
    i = 0
    while i < len(rest):
        m = re.match(r"Step (\d+):\s*", rest[i:])
        if not m:
            if rest[i:].strip():
                raise ValueError(f"unexpected trailing text at {rest[i : i + 80]!r}")
            break
        hop = int(m.group(1))
        if hop != len(blocks) + 1:
            raise ValueError(f"expected Step {len(blocks) + 1}, got Step {hop}")
        abs_block_start = i + m.start()
        after_label = i + m.end()
        ev = rest.find('Evidence: "', after_label)
        if ev < 0:
            raise ValueError(f"No Evidence for step {hop}")
        q_inner_start = ev + len('Evidence: "')
        _, q_close_idx = consume_evidence_quote(rest, q_inner_start)
        block_end = q_close_idx + 1
        blocks.append(rest[abs_block_start:block_end].strip())
        i = block_end
        while i < len(rest) and rest[i].isspace():
            i += 1
    if not blocks:
        raise ValueError("no step blocks parsed")
    return q_part, blocks, ans


def cumulative_prefix_strings(question: str, step_blocks: list[str]) -> list[str]:
    prefixes: list[str] = []
    acc = f"Question: {question.strip()}"
    prefixes.append(acc)
    for blk in step_blocks:
        acc = f"{acc} {blk.strip()}"
        prefixes.append(acc)
    return prefixes


def first_wrong_hop_from_row(row: dict) -> int | None:
    wh = row.get("first_wrong_hop")
    if wh is not None:
        v = int(wh)
        if v >= 1:
            return v
    hops = row.get("wrong_hops") or []
    if hops:
        return int(hops[0])
    legacy = row.get("wrong_evidence_at_hop")
    if legacy is not None:
        v = int(legacy)
        if v >= 1:
            return v
    return None

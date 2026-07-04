"""Reasoning trace text assembly (shared by gold / counterfactual constructors)."""

from __future__ import annotations


def escape_double_quotes(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def assemble_trace(
    question: str,
    step_reasoning: list[str],
    evidence_texts: list[str],
    final_answer: str,
) -> str:
    """Build canonical reasoning trace string."""
    parts = [f"Question: {question.strip()}"]
    for i, (step, ev) in enumerate(zip(step_reasoning, evidence_texts), start=1):
        parts.append(f"Step {i}: {step.strip()} Evidence: \"{escape_double_quotes(ev.strip())}\"")
    parts.append(f"Final Answer: {final_answer.strip()}")
    return " ".join(parts)

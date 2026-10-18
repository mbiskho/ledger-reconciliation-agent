"""Accuracy evaluation — measures agent performance against known ground truth."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import structlog

from ..agent.loop import ReconciliationResult
from ..ledger.generator import Break, BreakType

logger = structlog.get_logger()


@dataclass
class EvalMetrics:
    total_breaks: int
    resolved: int
    escalated: int
    false_resolutions: int
    resolution_rate: float
    false_resolution_rate: float
    escalation_rate: float
    mean_steps: float
    mean_cost: float
    total_cost: float
    by_break_type: dict[str, dict[str, Any]]
    audit_trail_complete: bool
    no_unlogged_state_changes: bool


@dataclass
class EvalConfig:
    max_false_resolution_rate: float = 0.05
    min_resolution_rate: float = 0.80
    max_escalation_rate: float = 0.30


def evaluate_results(
    results: list[ReconciliationResult],
    ground_truth: list[dict[str, Any]],
    config: EvalConfig | None = None,
) -> EvalMetrics:
    if config is None:
        config = EvalConfig()

    gt_by_id = {gt["id"]: gt for gt in ground_truth}

    resolved = sum(1 for r in results if r.resolved)
    escalated = sum(1 for r in results if r.escalated)
    total = len(results)

    false_resolutions = 0
    for r in results:
        if r.resolved:
            gt = gt_by_id.get(r.break_id, {})
            if gt.get("expected_resolution") == "flag_for_investigation":
                false_resolutions += 1

    by_type: dict[str, dict[str, Any]] = {}
    for r in results:
        gt = gt_by_id.get(r.break_id, {})
        btype = gt.get("break_type", "unknown")
        if btype not in by_type:
            by_type[btype] = {"total": 0, "resolved": 0, "escalated": 0, "false_resolutions": 0}
        by_type[btype]["total"] += 1
        if r.resolved:
            by_type[btype]["resolved"] += 1
        if r.escalated:
            by_type[btype]["escalated"] += 1

    steps = [r.steps_taken for r in results]
    costs = [r.cost_estimate for r in results]

    audit_complete = all(len(r.audit_trail) == r.steps_taken for r in results)
    no_unlogged = all(r.steps_taken > 0 for r in results if r.resolved or r.escalated)

    return EvalMetrics(
        total_breaks=total,
        resolved=resolved,
        escalated=escalated,
        false_resolutions=false_resolutions,
        resolution_rate=resolved / total if total > 0 else 0.0,
        false_resolution_rate=false_resolutions / resolved if resolved > 0 else 0.0,
        escalation_rate=escalated / total if total > 0 else 0.0,
        mean_steps=float(sum(steps)) / len(steps) if steps else 0.0,
        mean_cost=float(sum(costs)) / len(costs) if costs else 0.0,
        total_cost=float(sum(costs)),
        by_break_type=by_type,
        audit_trail_complete=audit_complete,
        no_unlogged_state_changes=no_unlogged,
    )


def check_thresholds(metrics: EvalMetrics, config: EvalConfig | None = None) -> tuple[bool, list[str]]:
    if config is None:
        config = EvalConfig()

    failures = []

    if metrics.false_resolution_rate > config.max_false_resolution_rate:
        failures.append(
            f"False resolution rate {metrics.false_resolution_rate:.1%} exceeds max {config.max_false_resolution_rate:.1%}"
        )

    if metrics.resolution_rate < config.min_resolution_rate:
        failures.append(
            f"Resolution rate {metrics.resolution_rate:.1%} below min {config.min_resolution_rate:.1%}"
        )

    if not metrics.audit_trail_complete:
        failures.append("Audit trail is incomplete — some steps are missing from the log")

    if not metrics.no_unlogged_state_changes:
        failures.append("Unlogged state changes detected")

    return len(failures) == 0, failures


def create_eval_charts(metrics: EvalMetrics, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Resolved", "Escalated", "False Resolutions"],
        y=[metrics.resolved, metrics.escalated, metrics.false_resolutions],
        marker_color=["#4CAF50", "#FF9800", "#F44336"],
        text=[metrics.resolved, metrics.escalated, metrics.false_resolutions],
        textposition="auto",
    ))
    fig.update_layout(
        title=f"Reconciliation Results ({metrics.total_breaks} breaks)",
        yaxis_title="Count",
        template="plotly_white",
    )
    fig.write_html(str(output_dir / "results_summary.html"))

    if metrics.by_break_type:
        types = list(metrics.by_break_type.keys())
        resolved_by_type = [metrics.by_break_type[t]["resolved"] for t in types]
        escalated_by_type = [metrics.by_break_type[t]["escalated"] for t in types]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Resolved", x=types, y=resolved_by_type, marker_color="#4CAF50"))
        fig2.add_trace(go.Bar(name="Escalated", x=types, y=escalated_by_type, marker_color="#FF9800"))
        fig2.update_layout(
            title="Resolution by Break Type",
            barmode="stack",
            xaxis_title="Break Type",
            yaxis_title="Count",
            template="plotly_white",
        )
        fig2.write_html(str(output_dir / "by_break_type.html"))


def generate_eval_report(metrics: EvalMetrics, output_dir: Path) -> str:
    passed, failures = check_thresholds(metrics)
    create_eval_charts(metrics, output_dir)

    lines = [
        "=" * 60,
        "  LEDGER RECONCILIATION AGENT — EVALUATION REPORT",
        "=" * 60,
        "",
        f"  Total breaks:          {metrics.total_breaks}",
        f"  Resolved:              {metrics.resolved} ({metrics.resolution_rate:.1%})",
        f"  Escalated:             {metrics.escalated} ({metrics.escalation_rate:.1%})",
        f"  False resolutions:     {metrics.false_resolutions} ({metrics.false_resolution_rate:.1%})",
        "",
        f"  Mean steps per break:  {metrics.mean_steps:.1f}",
        f"  Mean cost per break:   ${metrics.mean_cost:.4f}",
        f"  Total cost:            ${metrics.total_cost:.4f}",
        "",
        f"  Audit trail complete:  {'Yes' if metrics.audit_trail_complete else 'NO — FAILURE'}",
        f"  No unlogged changes:   {'Yes' if metrics.no_unlogged_state_changes else 'NO — FAILURE'}",
        "",
        "  Thresholds:",
        f"    {'PASS' if passed else 'FAIL'} — {'All checks passed' if passed else 'Failures detected'}",
    ]

    for failure in failures:
        lines.append(f"      FAIL: {failure}")

    lines += [
        "",
        "  Results by break type:",
    ]

    for btype, info in metrics.by_break_type.items():
        rate = info["resolved"] / info["total"] if info["total"] > 0 else 0
        lines.append(f"    {btype:<25}: {info['resolved']}/{info['total']} resolved ({rate:.0%})")

    lines.append("=" * 60)

    report = "\n".join(lines)

    with open(output_dir / "eval_report.txt", "w") as f:
        f.write(report)

    with open(output_dir / "eval_metrics.json", "w") as f:
        json.dump({
            "total_breaks": metrics.total_breaks,
            "resolved": metrics.resolved,
            "escalated": metrics.escalated,
            "false_resolutions": metrics.false_resolutions,
            "resolution_rate": metrics.resolution_rate,
            "false_resolution_rate": metrics.false_resolution_rate,
            "escalation_rate": metrics.escalation_rate,
            "mean_steps": metrics.mean_steps,
            "mean_cost": metrics.mean_cost,
            "total_cost": metrics.total_cost,
            "by_break_type": metrics.by_break_type,
            "passed": passed,
            "failures": failures,
        }, f, indent=2)

    return report

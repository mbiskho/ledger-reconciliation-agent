"""CLI — ledger-rec command-line interface."""

import json
from pathlib import Path

import click
import pandas as pd
import structlog

from .agent.loop import AgentConfig, ReconciliationResult, run_reconciliation_agent
from .audit.log import AuditLog
from .escalation.queue import EscalationItem, EscalationPriority, EscalationQueue
from .eval.metrics import EvalConfig, EvalMetrics, evaluate_results, generate_eval_report
from .guardrails.engine import DeterministicGuardrail
from .ledger.generator import BreakType, LedgerConfig, generate_ledgers, save_ledgers
from .tools.suite import (
    DateNormalizer,
    FXNormalizer,
    FuzzyMatchTool,
    LedgerQueryTool,
    ProposalTool,
)

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@click.group()
@click.version_option()
def main():
    """Ledger Reconciliation Agent — finds payment breaks and explains what it did."""


@main.command()
@click.option("--transactions", default=5000, help="Number of transactions")
@click.option("--break-rate", default=0.15, help="Fraction of transactions with breaks")
@click.option("--seed", default=42, help="Random seed")
@click.option("--output", default="data/synthetic", help="Output directory")
def generate(transactions, break_rate, seed, output):
    """Generate synthetic ledgers with seeded breaks."""
    config = LedgerConfig(num_transactions=transactions, break_rate=break_rate, seed=seed)
    df_a, df_b, breaks = generate_ledgers(config)
    save_ledgers(df_a, df_b, breaks, output)

    click.echo(f"Generated {len(df_a)} transactions with {len(breaks)} breaks in {output}/")
    click.echo(f"Break types: {[b.break_type.value for b in breaks[:10]]}...")


@main.command()
@click.option("--data-dir", default="data/synthetic", help="Directory with ledger parquet files")
@click.option("--model", default="gpt-4o", help="LLM model")
@click.option("--max-steps", default=15, help="Max agent steps per break")
@click.option("--max-cost", default=0.50, help="Max cost per break in USD")
@click.option("--output", default="results", help="Results output directory")
@click.option("--dry-run", is_flag=True, help="Dry run without LLM calls")
def reconcile(data_dir, model, max_steps, max_cost, output, dry_run):
    """Run the reconciliation agent against all breaks."""
    import os

    df_a = pd.read_parquet(f"{data_dir}/system_a.parquet")
    df_b = pd.read_parquet(f"{data_dir}/system_b.parquet")

    with open(f"{data_dir}/ground_truth.json") as f:
        ground_truth = json.load(f)

    config = AgentConfig(model=model, max_steps=max_steps, max_cost_usd=max_cost)
    query_tool = LedgerQueryTool(df_a, df_b)
    fuzzy_tool = FuzzyMatchTool(df_a, df_b)
    date_norm = DateNormalizer()
    fx_norm = FXNormalizer()
    proposal_tool = ProposalTool()
    guardrail = DeterministicGuardrail()
    audit_log = AuditLog(db_path=f"{output}/audit.db")
    escalation_q = EscalationQueue()

    def guardrail_fn(action, args):
        result = guardrail.check(action, args)
        return result.allowed, result.reason

    def audit_fn(event_type, details):
        audit_log.log(event_type, details.get("break_id", ""), "agent", details.get("action", ""), details)

    results: list[ReconciliationResult] = []

    for i, break_case in enumerate(ground_truth):
        click.echo(f"[{i+1}/{len(ground_truth)}] Reconciling {break_case['id']} ({break_case['break_type']})")

        if dry_run:
            result = ReconciliationResult(
                break_id=break_case["id"], resolved=True, resolution_type="dry_run",
                resolution_detail="Dry run", escalated=False, confidence=0.9,
                steps_taken=3, cost_estimate=0.0,
            )
        else:
            result = run_reconciliation_agent(
                break_case, query_tool, fuzzy_tool, date_norm, fx_norm, proposal_tool,
                config, guardrail_fn=guardrail_fn, audit_log_fn=audit_fn,
            )

        results.append(result)

        if result.escalated:
            escalation_q.add(
                break_id=break_case["id"],
                reason="Agent escalated — requires human review",
                agent_reasoning=result.resolution_detail,
                evidence={"steps": result.steps_taken, "cost": result.cost_estimate},
                priority=EscalationPriority.MEDIUM,
            )

        click.echo(f"  -> {'Resolved' if result.resolved else 'Escalated'} in {result.steps_taken} steps, ${result.cost_estimate:.4f}")

    os.makedirs(output, exist_ok=True)
    with open(f"{output}/results.json", "w") as f:
        json.dump([{
            "break_id": r.break_id, "resolved": r.resolved,
            "resolution_type": r.resolution_type, "resolution_detail": r.resolution_detail,
            "escalated": r.escalated, "confidence": r.confidence,
            "steps_taken": r.steps_taken, "cost_estimate": r.cost_estimate,
        } for r in results], f, indent=2)

    escalation_q.export(Path(f"{output}/escalations.json"))
    audit_log.export_to_json(Path(f"{output}/audit_log.json"))

    integrity_ok, issues = audit_log.verify_integrity()
    click.echo(f"\nAudit log integrity: {'OK' if integrity_ok else 'FAILED'}")

    if dry_run:
        from .eval.metrics import evaluate_results, generate_eval_report

        metrics = evaluate_results(results, ground_truth)
        report = generate_eval_report(metrics, Path(output))
        click.echo(report)


@main.command()
@click.option("--results", "results_file", default="results/results.json", help="Results JSON")
@click.option("--ground-truth", "gt_file", default="data/synthetic/ground_truth.json", help="Ground truth JSON")
@click.option("--output", default="results", help="Output directory")
def evaluate(results_file, gt_file, output):
    """Evaluate agent performance against ground truth."""
    with open(results_file) as f:
        results_data = json.load(f)
    with open(gt_file) as f:
        ground_truth = json.load(f)

    results = [ReconciliationResult(**r) for r in results_data]
    metrics = evaluate_results(results, ground_truth)
    report = generate_eval_report(metrics, Path(output))

    click.echo(report)

    passed, failures = metrics.check_thresholds(metrics) if hasattr(metrics, 'check_thresholds') else (True, [])
    if not passed:
        click.echo("\nEVALUATION FAILED:")
        for f in failures:
            click.echo(f"  FAIL: {f}")
        raise SystemExit(1)


@main.command()
@click.option("--port", default=8000, help="Server port")
def serve(port):
    """Start the FastAPI server."""
    import uvicorn
    uvicorn.run("ledger_rec.api.server:app", host="0.0.0.0", port=port, reload=True)


@main.command()
@click.option("--data-dir", default="data/synthetic", help="Ledger data directory")
@click.option("--reference", help="Search by reference")
@click.option("--counterparty", help="Search by counterparty")
@click.option("--min-amount", type=float, help="Minimum amount")
@click.option("--max-amount", type=float, help="Maximum amount")
def query(data_dir, reference, counterparty, min_amount, max_amount):
    """Query the ledgers directly."""
    df_a = pd.read_parquet(f"{data_dir}/system_a.parquet")
    df_b = pd.read_parquet(f"{data_dir}/system_b.parquet")
    query_tool = LedgerQueryTool(df_a, df_b)

    if reference:
        results = query_tool.query_by_reference(reference)
    elif counterparty:
        results = query_tool.query_by_counterparty(counterparty)
    elif min_amount is not None and max_amount is not None:
        results = query_tool.query_by_amount_range(min_amount, max_amount)
    else:
        results = query_tool.get_summary_stats()

    click.echo(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()

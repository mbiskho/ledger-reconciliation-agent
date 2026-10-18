"""FastAPI server — exposes the reconciliation agent as an HTTP API."""

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from ..agent.loop import AgentConfig, ReconciliationResult, run_reconciliation_agent
from ..audit.log import AuditLog
from ..escalation.queue import EscalationItem, EscalationPriority, EscalationQueue
from ..guardrails.engine import DeterministicGuardrail
from ..ledger.generator import Break, generate_ledgers, save_ledgers
from ..tools.suite import (
    DateNormalizer,
    FXNormalizer,
    FuzzyMatchTool,
    LedgerQueryTool,
    ProposalTool,
)

app = FastAPI(title="Ledger Reconciliation Agent", version="0.1.0")

guardrail = DeterministicGuardrail()
audit_log = AuditLog(db_path="data/audit.db")
escalation_queue = EscalationQueue()

_df_a: pd.DataFrame | None = None
_df_b: pd.DataFrame | None = None
_ground_truth: list[dict[str, Any]] = []


class GenerateRequest(BaseModel):
    num_transactions: int = 500
    break_rate: float = 0.15
    seed: int = 42


class ReconcileRequest(BaseModel):
    break_id: str
    model: str = "gpt-4o"


class EscalationResolve(BaseModel):
    item_id: str
    notes: str
    action: str = "resolve"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate_ledgers_endpoint(req: GenerateRequest):
    global _df_a, _df_b, _ground_truth

    from ..ledger.generator import LedgerConfig

    config = LedgerConfig(
        num_transactions=req.num_transactions,
        break_rate=req.break_rate,
        seed=req.seed,
    )
    df_a, df_b, breaks = generate_ledgers(config)
    _df_a = df_a
    _df_b = df_b
    _ground_truth = [{
        "id": b.id, "break_type": b.break_type.value, "side_a_ref": b.side_a_ref,
        "side_b_ref": b.side_b_ref, "description": b.description,
        "expected_resolution": b.expected_resolution, "amount_delta": b.amount_delta,
    } for b in breaks]

    save_ledgers(df_a, df_b, breaks, "data/synthetic")
    return {
        "status": "generated",
        "total_transactions": len(df_a),
        "breaks_seeded": len(breaks),
        "break_rate": req.break_rate,
    }


@app.get("/stats")
def get_stats():
    if _df_a is None or _df_b is None:
        raise HTTPException(404, "No ledgers loaded. Call /generate first.")

    query_tool = LedgerQueryTool(_df_a, _df_b)
    return query_tool.get_summary_stats()


@app.post("/reconcile")
def reconcile_break(req: ReconcileRequest):
    if _df_a is None or _df_b is None:
        raise HTTPException(404, "No ledgers loaded. Call /generate first.")

    break_case = next((b for b in _ground_truth if b["id"] == req.break_id), None)
    if break_case is None:
        raise HTTPException(404, f"Break {req.break_id} not found")

    config = AgentConfig(model=req.model)
    query_tool = LedgerQueryTool(_df_a, _df_b)
    fuzzy_tool = FuzzyMatchTool(_df_a, _df_b)
    date_norm = DateNormalizer()
    fx_norm = FXNormalizer()
    proposal_tool = ProposalTool()

    def guardrail_fn(action: str, args: dict) -> tuple[bool, str]:
        result = guardrail.check(action, args, req.break_id)
        return result.allowed, result.reason

    def audit_fn(event_type: str, details: dict) -> None:
        audit_log.log(event_type, req.break_id, "agent", details.get("action", ""), details)

    result = run_reconciliation_agent(
        break_case, query_tool, fuzzy_tool, date_norm, fx_norm, proposal_tool,
        config, guardrail_fn=guardrail_fn, audit_log_fn=audit_fn,
    )

    if result.escalated:
        escalation_queue.add(
            break_id=req.break_id,
            reason="Agent escalated",
            agent_reasoning=result.resolution_detail,
            evidence={"steps": result.steps_taken, "cost": result.cost_estimate},
        )

    return {
        "break_id": result.break_id,
        "resolved": result.resolved,
        "escalated": result.escalated,
        "resolution_type": result.resolution_type,
        "resolution_detail": result.resolution_detail,
        "confidence": result.confidence,
        "steps_taken": result.steps_taken,
        "cost_estimate": result.cost_estimate,
        "audit_trail": result.audit_trail,
    }


@app.get("/escalations")
def get_escalations(status: str = "pending"):
    if status == "pending":
        items = escalation_queue.get_pending()
    else:
        items = escalation_queue.items
    return {
        "count": len(items),
        "items": [{
            "id": i.id, "break_id": i.break_id, "priority": i.priority.value,
            "status": i.status.value, "reason": i.reason,
            "created_at": i.created_at,
        } for i in items],
    }


@app.post("/escalations/resolve")
def resolve_escalation(req: EscalationResolve):
    if req.action == "resolve":
        ok = escalation_queue.resolve(req.item_id, req.notes)
    elif req.action == "dismiss":
        ok = escalation_queue.dismiss(req.item_id, req.notes)
    else:
        raise HTTPException(400, f"Unknown action: {req.action}")

    if not ok:
        raise HTTPException(404, f"Escalation item {req.item_id} not found")

    audit_log.log("escalation_resolved", req.item_id, "human", req.action, {"notes": req.notes})
    return {"status": "ok", "action": req.action}


@app.get("/audit")
def get_audit_log(limit: int = 100):
    entries = audit_log.get_all_entries(limit)
    return {
        "count": len(entries),
        "entries": [{
            "id": e.id, "timestamp": e.timestamp, "event_type": e.event_type,
            "break_id": e.break_id, "actor": e.actor, "action": e.action,
            "result": e.result,
        } for e in entries],
    }


@app.get("/audit/integrity")
def verify_audit_integrity():
    valid, issues = audit_log.verify_integrity()
    return {"valid": valid, "issues": issues}

"""Reconciliation agent — LLM-powered agent loop with step limits and budget caps."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

import pandas as pd
import structlog

from ..tools.suite import (
    DateNormalizer,
    FXNormalizer,
    FuzzyMatchTool,
    LedgerQueryTool,
    ProposalTool,
)

logger = structlog.get_logger()


class AgentState(str, Enum):
    IDLE = "idle"
    INVESTIGATING = "investigating"
    PROPOSING = "proposing"
    WAITING_APPROVAL = "waiting_approval"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentStep:
    step_number: int
    thought: str
    action: str
    tool_calls: list[dict[str, Any]]
    observation: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ReconciliationResult:
    break_id: str
    resolved: bool
    resolution_type: str
    resolution_detail: str
    escalated: bool
    confidence: float
    steps_taken: int
    cost_estimate: float
    audit_trail: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentConfig:
    model: str = "gpt-4o"
    max_steps: int = 15
    max_cost_usd: float = 0.50
    temperature: float = 0.0
    system_prompt: str = ""

    def __post_init__(self):
        if not self.system_prompt:
            self.system_prompt = """You are a payment ledger reconciliation agent. Your job is to find and resolve discrepancies between two payment systems (System A and System B).

You have access to these tools:
- query_by_reference(reference, system) — search for transactions by reference ID
- query_by_amount_range(min, max, system) — search by amount range
- query_by_date_range(start, end, system) — search by date range
- query_by_counterparty(name, system) — search by counterparty name
- get_unmatched(system) — get transactions without a match in the other system
- get_summary_stats() — get ledger-level statistics
- fuzzy_match(reference) — find similar reference IDs
- cross_match_unmatched() — fuzzy match unmatched transactions across systems
- normalize_date(date) — normalize to next business day
- normalize_currency(amount, currency) — convert to USD equivalent
- are_equivalent(amount_a, currency_a, amount_b, currency_b) — check if amounts are FX-equivalent
- propose_match(side_a_id, side_b_id, reason, confidence) — propose linking two transactions
- propose_adjustment(transaction_id, field, old_value, new_value, reason) — propose a field correction
- propose_escalation(break_id, reason, evidence) — escalate for human review
- propose_flag(transaction_id, flag_type, reason) — flag a transaction for attention

Rules:
1. You can investigate and propose, but you CANNOT execute any writes or adjustments.
2. Every proposal goes through a deterministic guardrail before it lands.
3. If you're unsure, escalate. A false resolution is worse than an escalation.
4. Always explain your reasoning in the thought field.
5. Be thorough: check both systems, cross-reference by amount and date.
6. Consider FX conversions when amounts are in different currencies.
7. Check for date shifts — settlement can take 2-3 business days.

Output your reasoning and next action as JSON:
{
  "thought": "string explaining your analysis",
  "action": "tool_name",
  "tool_args": {"arg1": "value1"},
  "confidence": 0.0-1.0
}"""


def run_reconciliation_agent(
    break_case: dict[str, Any],
    query_tool: LedgerQueryTool,
    fuzzy_tool: FuzzyMatchTool,
    date_norm: DateNormalizer,
    fx_norm: FXNormalizer,
    proposal_tool: ProposalTool,
    config: AgentConfig | None = None,
    llm_call: Optional[Callable] = None,
    guardrail_fn: Optional[Callable] = None,
    audit_log_fn: Optional[Callable] = None,
) -> ReconciliationResult:
    if config is None:
        config = AgentConfig()

    steps: list[AgentStep] = []
    total_cost = 0.0
    messages: list[dict[str, str]] = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": f"Investigate this potential break:\n{json.dumps(break_case, indent=2, default=str)}"},
    ]

    resolved = False
    escalated = False
    resolution_type = "unknown"
    resolution_detail = ""

    for step_num in range(1, config.max_steps + 1):
        if total_cost >= config.max_cost_usd:
            logger.warning("budget cap reached", steps=step_num, cost=total_cost)
            escalated = True
            resolution_type = "budget_exceeded"
            resolution_detail = f"Budget cap of ${config.max_cost_usd} reached after {step_num} steps"
            break

        response = _call_llm(messages, config, llm_call)
        total_cost += _estimate_cost(response, config.model)

        try:
            parsed = _parse_agent_response(response)
        except (json.JSONDecodeError, KeyError):
            logger.error("failed to parse agent response", response=response[:200])
            continue

        tool_result = _execute_tool(
            parsed["action"], parsed.get("tool_args", {}),
            break_case, query_tool, fuzzy_tool, date_norm, fx_norm, proposal_tool,
        )

        step = AgentStep(
            step_number=step_num,
            thought=parsed.get("thought", ""),
            action=parsed["action"],
            tool_calls=[{"action": parsed["action"], "args": parsed.get("tool_args", {}), "result": str(tool_result)[:500]}],
            observation=str(tool_result)[:500],
        )
        steps.append(step)

        if audit_log_fn:
            audit_log_fn("agent_step", {
                "break_id": break_case.get("id", ""), "step": step_num,
                "thought": step.thought, "action": step.action,
                "observation": step.observation,
            })

        if guardrail_fn:
            allowed, reason = guardrail_fn(parsed["action"], parsed.get("tool_args", {}))
            if not allowed:
                logger.warning("guardrail blocked action", action=parsed["action"], reason=reason)
                messages.append({"role": "assistant", "content": json.dumps(parsed)})
                messages.append({"role": "user", "content": f"Action blocked by guardrail: {reason}. Try a different approach."})
                continue

        messages.append({"role": "assistant", "content": json.dumps(parsed)})
        messages.append({"role": "user", "content": f"Tool result: {json.dumps(tool_result, default=str)[:1000]}"})

        if parsed["action"] in ("propose_escalation", "final_answer"):
            escalated = parsed["action"] == "propose_escalation"
            resolved = not escalated
            resolution_type = "escalated" if escalated else "resolved"
            resolution_detail = parsed.get("thought", "")
            break

        if step_num == config.max_steps:
            escalated = True
            resolution_type = "max_steps_reached"
            resolution_detail = f"Reached maximum steps ({config.max_steps}) without resolution"

    return ReconciliationResult(
        break_id=break_case.get("id", "unknown"),
        resolved=resolved,
        resolution_type=resolution_type,
        resolution_detail=resolution_detail,
        escalated=escalated,
        confidence=float(parsed.get("confidence", 0.5)) if "parsed" in dir() else 0.0,
        steps_taken=len(steps),
        cost_estimate=round(total_cost, 4),
        audit_trail=[{
            "step": s.step_number, "thought": s.thought,
            "action": s.action, "observation": s.observation,
        } for s in steps],
    )


def _call_llm(messages: list[dict], config: AgentConfig, llm_call: Optional[Callable] = None) -> str:
    if llm_call:
        return llm_call(messages, config)

    import openai
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


def _estimate_cost(response: str, model: str) -> float:
    token_estimate = len(response) // 4
    if "gpt-4o" in model:
        return token_estimate * 0.000015
    elif "gpt-3.5" in model:
        return token_estimate * 0.000002
    return token_estimate * 0.00001


def _parse_agent_response(response: str) -> dict[str, Any]:
    response = response.strip()
    if response.startswith("```"):
        response = response.split("\n", 1)[1]
        if response.endswith("```"):
            response = response[:-3]

    start = response.find("{")
    end = response.rfind("}") + 1
    if start >= 0 and end > start:
        response = response[start:end]

    return json.loads(response)


def _execute_tool(
    action: str, args: dict[str, Any], break_case: dict[str, Any],
    query_tool: LedgerQueryTool, fuzzy_tool: FuzzyMatchTool,
    date_norm: DateNormalizer, fx_norm: FXNormalizer, proposal_tool: ProposalTool,
) -> Any:
    try:
        if action == "query_by_reference":
            return query_tool.query_by_reference(args.get("reference", ""), args.get("system", "both"))
        elif action == "query_by_amount_range":
            return query_tool.query_by_amount_range(args.get("min", 0), args.get("max", 999999), args.get("system", "both"))
        elif action == "query_by_date_range":
            return query_tool.query_by_date_range(args.get("start", ""), args.get("end", ""), args.get("system", "both"))
        elif action == "query_by_counterparty":
            return query_tool.query_by_counterparty(args.get("name", ""), args.get("system", "both"))
        elif action == "get_unmatched":
            return query_tool.get_unmatched(args.get("system", "both"))
        elif action == "get_summary_stats":
            return query_tool.get_summary_stats()
        elif action == "fuzzy_match":
            return fuzzy_tool.find_fuzzy_matches(args.get("reference", ""))
        elif action == "cross_match_unmatched":
            return fuzzy_tool.cross_match_unmatched(
                query_tool.df_a[~query_tool.df_a["reference"].isin(query_tool.df_b["reference"])],
                query_tool.df_b[~query_tool.df_b["reference"].isin(query_tool.df_a["reference"])],
            )
        elif action == "normalize_date":
            return date_norm.normalize(args.get("date", ""))
        elif action == "normalize_currency":
            return fx_norm.normalize_to_usd(args.get("amount", 0), args.get("currency", "USD"))
        elif action == "are_equivalent":
            return fx_norm.are_equivalent(
                args.get("amount_a", 0), args.get("currency_a", "USD"),
                args.get("amount_b", 0), args.get("currency_b", "USD"),
            )
        elif action == "propose_match":
            return proposal_tool.propose_match(
                args.get("side_a_id", ""), args.get("side_b_id", ""),
                args.get("reason", ""), args.get("confidence", 0.5),
            )
        elif action == "propose_adjustment":
            return proposal_tool.propose_adjustment(
                args.get("transaction_id", ""), args.get("field", ""),
                args.get("old_value", ""), args.get("new_value", ""),
                args.get("reason", ""),
            )
        elif action == "propose_escalation":
            return proposal_tool.propose_escalation(
                args.get("break_id", ""), args.get("reason", ""),
                args.get("evidence", {}),
            )
        elif action == "propose_flag":
            return proposal_tool.propose_flag(
                args.get("transaction_id", ""), args.get("flag_type", ""),
                args.get("reason", ""),
            )
        elif action == "final_answer":
            return {"status": "complete", "summary": args.get("summary", "")}
        else:
            return {"error": f"Unknown action: {action}"}
    except Exception as e:
        return {"error": str(e), "action": action, "args": args}

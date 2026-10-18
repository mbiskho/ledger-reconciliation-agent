"""Deterministic guardrails — every state change must pass before it lands."""

from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger()

ALLOWED_ACTIONS = {
    "query_by_reference", "query_by_amount_range", "query_by_date_range",
    "query_by_counterparty", "get_unmatched", "get_summary_stats",
    "fuzzy_match", "cross_match_unmatched",
    "normalize_date", "normalize_currency", "are_equivalent",
    "propose_match", "propose_adjustment", "propose_escalation",
    "propose_flag", "final_answer",
}

READ_ONLY_ACTIONS = {
    "query_by_reference", "query_by_amount_range", "query_by_date_range",
    "query_by_counterparty", "get_unmatched", "get_summary_stats",
    "fuzzy_match", "cross_match_unmatched",
    "normalize_date", "normalize_currency", "are_equivalent",
}

WRITE_PROPOSAL_ACTIONS = {
    "propose_match", "propose_adjustment", "propose_escalation", "propose_flag",
}

FORBIDDEN_PATTERNS = [
    "DROP", "DELETE", "TRUNCATE", "INSERT", "UPDATE", "ALTER",
    "EXEC", "EXECUTE", "SYSTEM", "sudo", "rm -rf",
    "/etc/", "/var/", "/root/",
]

MAX_PROPOSALS_PER_BREAK = 10
MAX_ADJUSTMENT_AMOUNT = 1_000_000.0


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str
    action: str
    proposal_count: int = 0


class DeterministicGuardrail:
    def __init__(self):
        self.proposal_counts: dict[str, int] = {}

    def check(self, action: str, args: dict[str, Any], break_id: str = "") -> GuardrailResult:
        if action not in ALLOWED_ACTIONS:
            return GuardrailResult(
                allowed=False,
                reason=f"Action '{action}' is not in the allowed set",
                action=action,
            )

        for key, value in args.items():
            value_str = str(value).upper()
            for pattern in FORBIDDEN_PATTERNS:
                if pattern in value_str:
                    return GuardrailResult(
                        allowed=False,
                        reason=f"Argument contains forbidden pattern: '{pattern}'",
                        action=action,
                    )

        if action in WRITE_PROPOSAL_ACTIONS and break_id:
            self.proposal_counts[break_id] = self.proposal_counts.get(break_id, 0) + 1
            count = self.proposal_counts[break_id]
            if count > MAX_PROPOSALS_PER_BREAK:
                return GuardrailResult(
                    allowed=False,
                    reason=f"Exceeded maximum proposals per break ({MAX_PROPOSALS_PER_BREAK})",
                    action=action,
                    proposal_count=count,
                )

        if action == "propose_adjustment":
            new_value = args.get("new_value", 0)
            try:
                if isinstance(new_value, str):
                    new_value = float(new_value.replace(",", "").replace("$", ""))
                if abs(float(new_value)) > MAX_ADJUSTMENT_AMOUNT:
                    return GuardrailResult(
                        allowed=False,
                        reason=f"Adjustment amount exceeds maximum ({MAX_ADJUSTMENT_AMOUNT})",
                        action=action,
                    )
            except (ValueError, TypeError):
                pass

        if action == "propose_match":
            confidence = args.get("confidence", 0)
            if confidence is not None and confidence < 0.5:
                return GuardrailResult(
                    allowed=False,
                    reason=f"Match confidence ({confidence}) below minimum threshold (0.5)",
                    action=action,
                )

        return GuardrailResult(allowed=True, reason="", action=action)

    def get_proposal_count(self, break_id: str) -> int:
        return self.proposal_counts.get(break_id, 0)

    def reset(self) -> None:
        self.proposal_counts.clear()

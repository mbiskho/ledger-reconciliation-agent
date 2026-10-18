"""Human escalation queue — displays unresolved breaks with agent reasoning."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


class EscalationPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EscalationStatus(str, Enum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


@dataclass
class EscalationItem:
    id: str
    break_id: str
    priority: EscalationPriority
    status: EscalationStatus
    reason: str
    agent_reasoning: str
    evidence: dict[str, Any]
    created_at: str
    assigned_to: str = ""
    resolution_notes: str = ""
    resolved_at: str = ""


@dataclass
class EscalationQueue:
    items: list[EscalationItem] = field(default_factory=list)

    def add(
        self, break_id: str, reason: str, agent_reasoning: str,
        evidence: dict[str, Any], priority: EscalationPriority = EscalationPriority.MEDIUM,
    ) -> EscalationItem:
        import uuid

        item = EscalationItem(
            id=str(uuid.uuid4()),
            break_id=break_id,
            priority=priority,
            status=EscalationStatus.PENDING,
            reason=reason,
            agent_reasoning=agent_reasoning,
            evidence=evidence,
            created_at=datetime.utcnow().isoformat(),
        )
        self.items.append(item)
        logger.info("break escalated", break_id=break_id, priority=priority.value)
        return item

    def get_pending(self) -> list[EscalationItem]:
        return [i for i in self.items if i.status == EscalationStatus.PENDING]

    def get_by_priority(self, priority: EscalationPriority) -> list[EscalationItem]:
        return [i for i in self.items if i.priority == priority and i.status != EscalationStatus.RESOLVED]

    def assign(self, item_id: str, assignee: str) -> bool:
        for item in self.items:
            if item.id == item_id:
                item.assigned_to = assignee
                item.status = EscalationStatus.IN_REVIEW
                return True
        return False

    def resolve(self, item_id: str, notes: str) -> bool:
        for item in self.items:
            if item.id == item_id:
                item.status = EscalationStatus.RESOLVED
                item.resolution_notes = notes
                item.resolved_at = datetime.utcnow().isoformat()
                return True
        return False

    def dismiss(self, item_id: str, notes: str) -> bool:
        for item in self.items:
            if item.id == item_id:
                item.status = EscalationStatus.DISMISSED
                item.resolution_notes = notes
                return True
        return False

    def stats(self) -> dict[str, int]:
        return {
            "total": len(self.items),
            "pending": len(self.get_pending()),
            "in_review": len([i for i in self.items if i.status == EscalationStatus.IN_REVIEW]),
            "resolved": len([i for i in self.items if i.status == EscalationStatus.RESOLVED]),
            "dismissed": len([i for i in self.items if i.status == EscalationStatus.DISMISSED]),
            "by_priority": {
                p.value: len(self.get_by_priority(p))
                for p in EscalationPriority
            },
        }

    def format_for_display(self, item: EscalationItem) -> str:
        lines = [
            "=" * 70,
            f"  ESCALATION: {item.break_id}",
            f"  Priority:  {item.priority.value.upper()}",
            f"  Status:    {item.status.value}",
            f"  Created:   {item.created_at}",
            f"  Assignee:  {item.assigned_to or 'Unassigned'}",
            "",
            f"  Reason: {item.reason}",
            "",
            "  Agent Reasoning:",
        ]
        for line in item.agent_reasoning.split("\n"):
            lines.append(f"    {line}")

        lines += [
            "",
            "  Evidence:",
        ]
        for k, v in item.evidence.items():
            lines.append(f"    {k}: {v}")

        lines += [
            "",
            f"  Resolution: {item.resolution_notes or 'Pending'}",
            "=" * 70,
        ]
        return "\n".join(lines)

    def export(self, path: Path) -> None:
        data = [{
            "id": i.id, "break_id": i.break_id, "priority": i.priority.value,
            "status": i.status.value, "reason": i.reason,
            "agent_reasoning": i.agent_reasoning, "evidence": i.evidence,
            "created_at": i.created_at, "assigned_to": i.assigned_to,
            "resolution_notes": i.resolution_notes, "resolved_at": i.resolved_at,
        } for i in self.items]

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

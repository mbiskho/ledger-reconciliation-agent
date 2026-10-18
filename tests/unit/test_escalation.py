import json

import pytest

from ledger_rec.escalation.queue import (
    EscalationItem,
    EscalationPriority,
    EscalationQueue,
    EscalationStatus,
)


class TestEscalationQueue:
    @pytest.fixture
    def queue(self):
        return EscalationQueue()

    def test_add_escalation(self, queue):
        item = queue.add(
            break_id="BRK-001",
            reason="Cannot resolve",
            agent_reasoning="Ambiguous amount mismatch",
            evidence={"amount_delta": 50.0},
            priority=EscalationPriority.HIGH,
        )
        assert item.break_id == "BRK-001"
        assert item.priority == EscalationPriority.HIGH
        assert item.status == EscalationStatus.PENDING

    def test_get_pending(self, queue):
        queue.add("BRK-001", "Reason 1", "Reasoning", {})
        queue.add("BRK-002", "Reason 2", "Reasoning", {})

        pending = queue.get_pending()
        assert len(pending) == 2

    def test_assign(self, queue):
        item = queue.add("BRK-001", "Reason", "Reasoning", {})
        ok = queue.assign(item.id, "analyst@example.com")
        assert ok
        assert item.status == EscalationStatus.IN_REVIEW
        assert item.assigned_to == "analyst@example.com"

    def test_resolve(self, queue):
        item = queue.add("BRK-001", "Reason", "Reasoning", {})
        ok = queue.resolve(item.id, "Manually matched")
        assert ok
        assert item.status == EscalationStatus.RESOLVED

    def test_dismiss(self, queue):
        item = queue.add("BRK-001", "Reason", "Reasoning", {})
        ok = queue.dismiss(item.id, "Not a real break")
        assert ok
        assert item.status == EscalationStatus.DISMISSED

    def test_stats(self, queue):
        queue.add("BRK-001", "R1", "A1", {}, EscalationPriority.HIGH)
        queue.add("BRK-002", "R2", "A2", {}, EscalationPriority.LOW)
        queue.add("BRK-003", "R3", "A3", {}, EscalationPriority.MEDIUM)

        stats = queue.stats()
        assert stats["total"] == 3
        assert stats["pending"] == 3
        assert stats["resolved"] == 0

    def test_format_display(self, queue):
        item = queue.add("BRK-001", "Cannot determine", "The amounts differ by 50.0", {"delta": 50.0})
        output = queue.format_for_display(item)
        assert "BRK-001" in output
        assert "Cannot determine" in output
        assert "50.0" in output

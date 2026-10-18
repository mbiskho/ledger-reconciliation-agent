import tempfile
from pathlib import Path

import pytest

from ledger_rec.audit.log import AuditLog


class TestAuditLog:
    @pytest.fixture
    def audit_log(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name

        log = AuditLog(db_path=path)
        yield log
        Path(path).unlink(missing_ok=True)

    def test_log_entry(self, audit_log):
        entry = audit_log.log("agent_step", "BRK-001", "agent", "query", {"reference": "TXN-001"})
        assert entry.id
        assert entry.break_id == "BRK-001"
        assert entry.event_type == "agent_step"

    def test_get_entries_for_break(self, audit_log):
        audit_log.log("agent_step", "BRK-001", "agent", "query", {"ref": "TXN-001"})
        audit_log.log("proposal", "BRK-001", "agent", "propose_match", {"match": "A-001"})
        audit_log.log("agent_step", "BRK-002", "agent", "query", {"ref": "TXN-002"})

        entries = audit_log.get_entries_for_break("BRK-001")
        assert len(entries) == 2

        entries_2 = audit_log.get_entries_for_break("BRK-002")
        assert len(entries_2) == 1

    def test_get_all_entries(self, audit_log):
        for i in range(5):
            audit_log.log("test", f"BRK-{i:03d}", "test", "action", {"i": i})

        entries = audit_log.get_all_entries(limit=3)
        assert len(entries) == 3

    def test_verify_integrity(self, audit_log):
        audit_log.log("test", "BRK-001", "agent", "query", {"ref": "TXN-001"})
        audit_log.log("test", "BRK-001", "agent", "propose", {"match": "A-001"})

        valid, issues = audit_log.verify_integrity()
        assert valid
        assert len(issues) == 0

    def test_export_json(self, audit_log):
        audit_log.log("test", "BRK-001", "agent", "query", {"ref": "TXN-001"})

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.json"
            audit_log.export_to_json(path)
            assert path.exists()

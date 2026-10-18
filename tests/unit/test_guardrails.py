import pytest

from ledger_rec.guardrails.engine import DeterministicGuardrail


class TestGuardrail:
    @pytest.fixture
    def guardrail(self):
        return DeterministicGuardrail()

    def test_allows_read_action(self, guardrail):
        result = guardrail.check("query_by_reference", {"reference": "TXN-001"})
        assert result.allowed

    def test_allows_propose_match(self, guardrail):
        result = guardrail.check("propose_match", {"side_a_id": "A-001", "side_b_id": "B-001", "confidence": 0.95})
        assert result.allowed

    def test_blocks_unknown_action(self, guardrail):
        result = guardrail.check("delete_all_records", {})
        assert not result.allowed

    def test_blocks_forbidden_pattern(self, guardrail):
        result = guardrail.check("query_by_reference", {"reference": "'; DROP TABLE transactions; --"})
        assert not result.allowed

    def test_blocks_low_confidence_match(self, guardrail):
        result = guardrail.check("propose_match", {"side_a_id": "A-001", "side_b_id": "B-001", "confidence": 0.3})
        assert not result.allowed

    def test_blocks_large_adjustment(self, guardrail):
        result = guardrail.check("propose_adjustment", {
            "transaction_id": "A-001", "field": "amount",
            "old_value": 100, "new_value": 2000000, "reason": "test",
        })
        assert not result.allowed

    def test_proposal_count_limit(self, guardrail):
        for i in range(10):
            result = guardrail.check("propose_match", {"side_a_id": f"A-{i}", "side_b_id": f"B-{i}", "confidence": 0.95}, "break-1")
            assert result.allowed

        result = guardrail.check("propose_match", {"side_a_id": "A-99", "side_b_id": "B-99", "confidence": 0.95}, "break-1")
        assert not result.allowed

    def test_separate_break_counts(self, guardrail):
        for _ in range(5):
            guardrail.check("propose_match", {"side_a_id": "A-1", "side_b_id": "B-1", "confidence": 0.95}, "break-1")
            guardrail.check("propose_match", {"side_a_id": "A-2", "side_b_id": "B-2", "confidence": 0.95}, "break-2")

        assert guardrail.get_proposal_count("break-1") == 5
        assert guardrail.get_proposal_count("break-2") == 5

    def test_reset(self, guardrail):
        guardrail.check("propose_match", {"side_a_id": "A-1", "side_b_id": "B-1", "confidence": 0.95}, "break-1")
        guardrail.reset()
        assert guardrail.get_proposal_count("break-1") == 0

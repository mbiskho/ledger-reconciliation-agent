import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from ledger_rec.tools.suite import (
    DateNormalizer,
    FXNormalizer,
    FuzzyMatchTool,
    LedgerQueryTool,
    MatchResult,
    ProposalTool,
)


@pytest.fixture
def sample_data():
    df_a = pd.DataFrame([
        {"id": "A-001", "reference": "TXN-001", "system": "SystemA", "amount": 100.0, "currency": "USD", "counterparty": "ACME Corp", "timestamp": "2025-06-15", "status": "settled"},
        {"id": "A-002", "reference": "TXN-002", "system": "SystemA", "amount": 250.0, "currency": "EUR", "counterparty": "Globex", "timestamp": "2025-06-16", "status": "settled"},
        {"id": "A-003", "reference": "", "system": "SystemA", "amount": 0.0, "currency": "", "counterparty": "", "timestamp": "2025-06-16", "status": "pending"},
    ])
    df_b = pd.DataFrame([
        {"id": "B-001", "reference": "TXN-001", "system": "SystemB", "amount": 100.0, "currency": "USD", "counterparty": "ACME Corp", "timestamp": "2025-06-15", "status": "settled"},
        {"id": "B-002", "reference": "TXN-002", "system": "SystemB", "amount": 230.0, "currency": "EUR", "counterparty": "Globex", "timestamp": "2025-06-17", "status": "settled"},
        {"id": "B-003", "reference": "TXN-003", "system": "SystemB", "amount": 500.0, "currency": "USD", "counterparty": "Initech", "timestamp": "2025-06-15", "status": "settled"},
    ])
    return df_a, df_b


class TestLedgerQueryTool:
    def test_query_by_reference(self, sample_data):
        df_a, df_b = sample_data
        tool = LedgerQueryTool(df_a, df_b)

        results = tool.query_by_reference("TXN-001")
        assert len(results["system_a"]) == 1
        assert len(results["system_b"]) == 1
        assert results["system_a"][0]["amount"] == 100.0

    def test_query_by_counterparty(self, sample_data):
        df_a, df_b = sample_data
        tool = LedgerQueryTool(df_a, df_b)

        results = tool.query_by_counterparty("ACME")
        assert len(results["system_a"]) == 1
        assert len(results["system_b"]) == 1

    def test_get_unmatched(self, sample_data):
        df_a, df_b = sample_data
        tool = LedgerQueryTool(df_a, df_b)

        results = tool.get_unmatched()
        assert len(results["system_b"]) == 1

    def test_get_summary_stats(self, sample_data):
        df_a, df_b = sample_data
        tool = LedgerQueryTool(df_a, df_b)

        stats = tool.get_summary_stats()
        assert stats["system_a"]["total_transactions"] == 3
        assert stats["system_b"]["total_transactions"] == 3

    def test_query_by_amount_range(self, sample_data):
        df_a, df_b = sample_data
        tool = LedgerQueryTool(df_a, df_b)

        results = tool.query_by_amount_range(90, 110)
        assert len(results["system_a"]) == 1
        assert results["system_a"][0]["reference"] == "TXN-001"


class TestFuzzyMatchTool:
    def test_fuzzy_match_exact(self, sample_data):
        df_a, df_b = sample_data
        tool = FuzzyMatchTool(df_a, df_b)

        matches = tool.find_fuzzy_matches("TXN-001")
        assert any(m["reference"] == "TXN-001" and m["similarity"] == 100 for m in matches)

    def test_fuzzy_match_similar(self, sample_data):
        df_a, df_b = sample_data
        tool = FuzzyMatchTool(df_a, df_b)

        matches = tool.find_fuzzy_matches("TXN-01")
        assert any(m["reference"] == "TXN-001" for m in matches)

    def test_cross_match(self, sample_data):
        df_a, df_b = sample_data
        tool = FuzzyMatchTool(df_a, df_b)

        unmatched_a = df_a[df_a["reference"] == ""]
        unmatched_b = df_b[~df_b["reference"].isin(df_a["reference"])]
        matches = tool.cross_match_unmatched(unmatched_a, unmatched_b)
        assert isinstance(matches, list)


class TestDateNormalizer:
    def test_normalize_weekend(self):
        normalizer = DateNormalizer()
        result = normalizer.normalize("2025-06-14")
        assert result == "2025-06-16"

    def test_settlement_delay(self):
        normalizer = DateNormalizer()
        assert normalizer.is_settlement_delay_expected("2025-06-15", "2025-06-17")
        assert not normalizer.is_settlement_delay_expected("2025-06-15", "2025-06-25")


class TestFXNormalizer:
    def test_normalize_to_usd(self):
        normalizer = FXNormalizer()
        usd = normalizer.normalize_to_usd(92.0, "EUR")
        assert abs(usd - 100.0) < 5.0

    def test_are_equivalent_same_currency(self):
        normalizer = FXNormalizer()
        assert normalizer.are_equivalent(100.0, "USD", 100.50, "USD")
        assert not normalizer.are_equivalent(100.0, "USD", 120.0, "USD")

    def test_are_equivalent_fx(self):
        normalizer = FXNormalizer()
        usd = normalizer.normalize_to_usd(92.0, "EUR")
        assert normalizer.are_equivalent(usd, "USD", 92.0, "EUR")


class TestProposalTool:
    def test_propose_match(self):
        tool = ProposalTool()
        result = tool.propose_match("A-001", "B-001", "Same reference", 0.95)
        assert result["action"] == "match"
        assert not result["requires_approval"]

    def test_propose_match_low_confidence(self):
        tool = ProposalTool()
        result = tool.propose_match("A-001", "B-001", "Fuzzy match", 0.75)
        assert result["requires_approval"]

    def test_propose_adjustment(self):
        tool = ProposalTool()
        result = tool.propose_adjustment("A-001", "amount", 100.0, 105.0, "Correction")
        assert result["action"] == "adjust"
        assert result["requires_approval"]

    def test_propose_escalation(self):
        tool = ProposalTool()
        result = tool.propose_escalation("BRK-001", "Cannot determine", {"reason": "Ambiguous"})
        assert result["action"] == "escalate"

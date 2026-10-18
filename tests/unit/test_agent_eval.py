import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from ledger_rec.agent.loop import (
    AgentConfig,
    ReconciliationResult,
    _estimate_cost,
    _parse_agent_response,
    run_reconciliation_agent,
)
from ledger_rec.eval.metrics import (
    EvalConfig,
    EvalMetrics,
    check_thresholds,
    evaluate_results,
    generate_eval_report,
)
from ledger_rec.tools.suite import (
    DateNormalizer,
    FXNormalizer,
    FuzzyMatchTool,
    LedgerQueryTool,
    ProposalTool,
)


class TestAgentLoop:
    def test_agent_config_defaults(self):
        config = AgentConfig()
        assert config.model == "gpt-4o"
        assert config.max_steps == 15
        assert config.max_cost_usd == 0.50

    def test_estimate_cost(self):
        cost = _estimate_cost("Hello world" * 100, "gpt-4o")
        assert cost > 0

    def test_parse_valid_json(self):
        response = '{"thought": "test", "action": "query_by_reference", "tool_args": {"reference": "TXN-001"}, "confidence": 0.9}'
        parsed = _parse_agent_response(response)
        assert parsed["action"] == "query_by_reference"
        assert parsed["confidence"] == 0.9

    def test_parse_json_with_markdown(self):
        response = '```json\n{"thought": "test", "action": "final_answer", "tool_args": {}, "confidence": 1.0}\n```'
        parsed = _parse_agent_response(response)
        assert parsed["action"] == "final_answer"

    def test_reconciliation_result(self):
        result = ReconciliationResult(
            break_id="BRK-001", resolved=True, resolution_type="matched",
            resolution_detail="Matched by reference", escalated=False,
            confidence=0.95, steps_taken=3, cost_estimate=0.005,
        )
        assert result.resolved
        assert result.steps_taken == 3


class TestEvaluation:
    def test_evaluate_all_resolved(self):
        results = [
            ReconciliationResult(break_id="BRK-001", resolved=True, resolution_type="matched", resolution_detail="ok", escalated=False, confidence=0.9, steps_taken=2, cost_estimate=0.01),
            ReconciliationResult(break_id="BRK-002", resolved=True, resolution_type="matched", resolution_detail="ok", escalated=False, confidence=0.9, steps_taken=3, cost_estimate=0.01),
        ]
        gt = [
            {"id": "BRK-001", "break_type": "amount_mismatch", "expected_resolution": "use_smaller_amount"},
            {"id": "BRK-002", "break_type": "date_shift", "expected_resolution": "use_earlier_date"},
        ]

        metrics = evaluate_results(results, gt)
        assert metrics.resolution_rate == 1.0
        assert metrics.escalation_rate == 0.0
        assert metrics.false_resolution_rate == 0.0

    def test_evaluate_mixed(self):
        results = [
            ReconciliationResult(break_id="BRK-001", resolved=True, resolution_type="matched", resolution_detail="ok", escalated=False, confidence=0.9, steps_taken=2, cost_estimate=0.01),
            ReconciliationResult(break_id="BRK-002", resolved=False, resolution_type="escalated", resolution_detail="", escalated=True, confidence=0.3, steps_taken=5, cost_estimate=0.02),
        ]
        gt = [
            {"id": "BRK-001", "break_type": "amount_mismatch", "expected_resolution": "use_smaller_amount"},
            {"id": "BRK-002", "break_type": "missing_side_b", "expected_resolution": "flag_for_investigation"},
        ]

        metrics = evaluate_results(results, gt)
        assert metrics.resolution_rate == 0.5
        assert metrics.escalation_rate == 0.5

    def test_false_resolution_detection(self):
        results = [
            ReconciliationResult(break_id="BRK-001", resolved=True, resolution_type="matched", resolution_detail="ok", escalated=False, confidence=0.9, steps_taken=2, cost_estimate=0.01),
        ]
        gt = [
            {"id": "BRK-001", "break_type": "missing_side_b", "expected_resolution": "flag_for_investigation"},
        ]

        metrics = evaluate_results(results, gt)
        assert metrics.false_resolutions == 1
        assert metrics.false_resolution_rate == 1.0

    def test_check_thresholds_pass(self):
        metrics = EvalMetrics(
            total_breaks=10, resolved=9, escalated=1, false_resolutions=0,
            resolution_rate=0.9, false_resolution_rate=0.0, escalation_rate=0.1,
            mean_steps=3.0, mean_cost=0.01, total_cost=0.1,
            by_break_type={}, audit_trail_complete=True, no_unlogged_state_changes=True,
        )
        passed, failures = check_thresholds(metrics)
        assert passed

    def test_check_thresholds_fail(self):
        metrics = EvalMetrics(
            total_breaks=10, resolved=8, escalated=2, false_resolutions=3,
            resolution_rate=0.8, false_resolution_rate=0.375, escalation_rate=0.2,
            mean_steps=3.0, mean_cost=0.01, total_cost=0.1,
            by_break_type={}, audit_trail_complete=False, no_unlogged_state_changes=False,
        )
        passed, failures = check_thresholds(metrics)
        assert not passed
        assert len(failures) > 0

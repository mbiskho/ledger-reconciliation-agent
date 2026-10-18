"""Tool suite — query, fuzzy match, date/FX normalization, and propose-adjustment tools for the reconciliation agent."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
import structlog
from thefuzz import fuzz

logger = structlog.get_logger()


@dataclass
class MatchResult:
    side_a_id: str | None
    side_b_id: str | None
    reference: str
    match_type: str
    confidence: float
    amount_a: float = 0.0
    amount_b: float = 0.0
    amount_delta: float = 0.0
    notes: str = ""


class LedgerQueryTool:
    def __init__(self, df_a: pd.DataFrame, df_b: pd.DataFrame):
        self.df_a = df_a
        self.df_b = df_b

    def query_by_reference(self, reference: str, system: str = "both") -> dict[str, Any]:
        results: dict[str, Any] = {}
        if system in ("both", "A"):
            matches = self.df_a[self.df_a["reference"] == reference]
            results["system_a"] = matches.to_dict("records") if not matches.empty else []
        if system in ("both", "B"):
            matches = self.df_b[self.df_b["reference"] == reference]
            results["system_b"] = matches.to_dict("records") if not matches.empty else []
        return results

    def query_by_amount_range(self, min_amount: float, max_amount: float, system: str = "both") -> dict[str, Any]:
        results: dict[str, Any] = {}
        if system in ("both", "A"):
            matches = self.df_a[(self.df_a["amount"] >= min_amount) & (self.df_a["amount"] <= max_amount)]
            results["system_a"] = matches.to_dict("records")[:100]
        if system in ("both", "B"):
            matches = self.df_b[(self.df_b["amount"] >= min_amount) & (self.df_b["amount"] <= max_amount)]
            results["system_b"] = matches.to_dict("records")[:100]
        return results

    def query_by_date_range(self, start: str, end: str, system: str = "both") -> dict[str, Any]:
        results: dict[str, Any] = {}
        if system in ("both", "A"):
            matches = self.df_a[(self.df_a["timestamp"] >= start) & (self.df_a["timestamp"] <= end)]
            results["system_a"] = matches.to_dict("records")[:100]
        if system in ("both", "B"):
            matches = self.df_b[(self.df_b["timestamp"] >= start) & (self.df_b["timestamp"] <= end)]
            results["system_b"] = matches.to_dict("records")[:100]
        return results

    def query_by_counterparty(self, counterparty: str, system: str = "both") -> dict[str, Any]:
        results: dict[str, Any] = {}
        if system in ("both", "A"):
            matches = self.df_a[self.df_a["counterparty"].str.contains(counterparty, case=False, na=False)]
            results["system_a"] = matches.to_dict("records")[:100]
        if system in ("both", "B"):
            matches = self.df_b[self.df_b["counterparty"].str.contains(counterparty, case=False, na=False)]
            results["system_b"] = matches.to_dict("records")[:100]
        return results

    def get_unmatched(self, system: str = "both") -> dict[str, Any]:
        results: dict[str, Any] = {}
        if system in ("both", "A"):
            refs_b = set(self.df_b["reference"].unique())
            unmatched_a = self.df_a[~self.df_a["reference"].isin(refs_b) | (self.df_a["reference"] == "")]
            results["system_a"] = unmatched_a.to_dict("records")[:100]
        if system in ("both", "B"):
            refs_a = set(self.df_a["reference"].unique())
            unmatched_b = self.df_b[~self.df_b["reference"].isin(refs_a) | (self.df_b["reference"] == "")]
            results["system_b"] = unmatched_b.to_dict("records")[:100]
        return results

    def get_summary_stats(self) -> dict[str, Any]:
        return {
            "system_a": {
                "total_transactions": len(self.df_a),
                "total_amount": float(self.df_a["amount"].sum()),
                "unique_references": int(self.df_a["reference"].nunique()),
                "currencies": self.df_a["currency"].unique().tolist(),
                "date_range": [str(self.df_a["timestamp"].min()), str(self.df_a["timestamp"].max())],
            },
            "system_b": {
                "total_transactions": len(self.df_b),
                "total_amount": float(self.df_b["amount"].sum()),
                "unique_references": int(self.df_b["reference"].nunique()),
                "currencies": self.df_b["currency"].unique().tolist(),
                "date_range": [str(self.df_b["timestamp"].min()), str(self.df_b["timestamp"].max())],
            },
        }


class FuzzyMatchTool:
    def __init__(self, df_a: pd.DataFrame, df_b: pd.DataFrame, threshold: int = 85):
        self.df_a = df_a
        self.df_b = df_b
        self.threshold = threshold

    def find_fuzzy_matches(self, reference: str, max_results: int = 10) -> list[dict[str, Any]]:
        candidates_a = self._fuzzy_score_list(reference, self.df_a["reference"].unique())
        candidates_b = self._fuzzy_score_list(reference, self.df_b["reference"].unique())

        results = []
        for ref, score in sorted(candidates_a + candidates_b, key=lambda x: x[1], reverse=True):
            if score >= self.threshold and len(results) < max_results:
                results.append({"reference": ref, "similarity": score})
        return results

    def _fuzzy_score_list(self, target: str, candidates: list[str]) -> list[tuple[str, int]]:
        results = []
        for c in candidates:
            if c and target:
                score = fuzz.ratio(target, c)
                results.append((c, score))
        return results

    def cross_match_unmatched(
        self, unmatched_a: pd.DataFrame, unmatched_b: pd.DataFrame, top_n: int = 5
    ) -> list[MatchResult]:
        matches = []

        for _, row_a in unmatched_a.iterrows():
            best_score = 0
            best_row: Any = None

            for _, row_b in unmatched_b.iterrows():
                ref_score = fuzz.ratio(str(row_a["reference"]), str(row_b["reference"]))
                cp_score = fuzz.ratio(str(row_a["counterparty"]), str(row_b["counterparty"]))

                amount_close = abs(row_a["amount"] - row_b["amount"]) < max(1.0, abs(row_a["amount"]) * 0.02)
                combined = ref_score * 0.5 + cp_score * 0.2 + (30 if amount_close else 0)

                if combined > best_score:
                    best_score = combined
                    best_row = row_b

            if best_score >= 60 and best_row is not None:
                matches.append(MatchResult(
                    side_a_id=row_a["id"], side_b_id=best_row["id"],
                    reference=str(row_a["reference"]),
                    match_type="fuzzy", confidence=best_score / 100,
                    amount_a=float(row_a["amount"]), amount_b=float(best_row["amount"]),
                    amount_delta=abs(float(row_a["amount"]) - float(best_row["amount"])),
                    notes=f"Fuzzy match: ref={best_score}, amt_close={amount_close}",
                ))

        return sorted(matches, key=lambda m: m.confidence, reverse=True)[:top_n]


class DateNormalizer:
    EXCHANGE_HOLIDAYS = {
        "2025-01-01", "2025-12-25", "2026-01-01",
    }

    def normalize(self, date_str: str) -> str:
        dt = pd.to_datetime(date_str)
        while dt.strftime("%Y-%m-%d") in self.EXCHANGE_HOLIDAYS or dt.weekday() >= 5:
            dt = dt + timedelta(days=1)
        return dt.strftime("%Y-%m-%d")

    def is_settlement_delay_expected(self, date_a: str, date_b: str, max_days: int = 3) -> bool:
        dt_a = pd.to_datetime(date_a)
        dt_b = pd.to_datetime(date_b)
        return abs((dt_b - dt_a).days) <= max_days


class FXNormalizer:
    FX_RATES = {
        "USD": 1.0, "EUR": 0.92, "GBP": 0.79,
        "JPY": 149.0, "CHF": 0.88, "CAD": 1.35,
        "AUD": 1.52, "CNY": 7.24,
    }

    def normalize_to_usd(self, amount: float, currency: str) -> float:
        rate = self.FX_RATES.get(currency, 1.0)
        return round(amount / rate, 2)

    def are_equivalent(self, amount_a: float, currency_a: str, amount_b: float, currency_b: str, tolerance: float = 0.02) -> bool:
        if currency_a == currency_b:
            return abs(amount_a - amount_b) / max(abs(amount_a), 1.0) <= tolerance
        usd_a = self.normalize_to_usd(amount_a, currency_a)
        usd_b = self.normalize_to_usd(amount_b, currency_b)
        return abs(usd_a - usd_b) / max(abs(usd_a), 1.0) <= tolerance

    def get_rate(self, from_currency: str, to_currency: str) -> float:
        from_rate = self.FX_RATES.get(from_currency, 1.0)
        to_rate = self.FX_RATES.get(to_currency, 1.0)
        return round(to_rate / from_rate, 6)


class ProposalTool:
    def propose_match(self, side_a_id: str, side_b_id: str, reason: str, confidence: float) -> dict[str, Any]:
        return {
            "action": "match",
            "side_a_id": side_a_id,
            "side_b_id": side_b_id,
            "reason": reason,
            "confidence": confidence,
            "requires_approval": confidence < 0.90,
        }

    def propose_adjustment(self, transaction_id: str, field: str, old_value: Any, new_value: Any, reason: str) -> dict[str, Any]:
        return {
            "action": "adjust",
            "transaction_id": transaction_id,
            "field": field,
            "old_value": str(old_value),
            "new_value": str(new_value),
            "reason": reason,
            "requires_approval": True,
        }

    def propose_escalation(self, break_id: str, reason: str, evidence: dict[str, Any]) -> dict[str, Any]:
        return {
            "action": "escalate",
            "break_id": break_id,
            "reason": reason,
            "evidence": evidence,
            "requires_approval": True,
        }

    def propose_flag(self, transaction_id: str, flag_type: str, reason: str) -> dict[str, Any]:
        return {
            "action": "flag",
            "transaction_id": transaction_id,
            "flag_type": flag_type,
            "reason": reason,
            "requires_approval": False,
        }

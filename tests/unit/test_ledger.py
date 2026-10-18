import json
import tempfile
from pathlib import Path

import pytest

from ledger_rec.ledger.generator import (
    Break,
    BreakType,
    LedgerConfig,
    Transaction,
    TransactionStatus,
    generate_ledgers,
    save_ledgers,
)


class TestLedgerGenerator:
    def test_generates_transactions(self):
        config = LedgerConfig(num_transactions=100, break_rate=0.1, seed=42)
        df_a, df_b, breaks = generate_ledgers(config)

        assert len(df_a) == 100
        assert len(df_b) == 100
        assert len(breaks) >= 5

    def test_break_rate_approximate(self):
        config = LedgerConfig(num_transactions=1000, break_rate=0.2, seed=42)
        _, _, breaks = generate_ledgers(config)
        rate = len(breaks) / 1000
        assert 0.1 < rate < 0.3

    def test_reproducible_with_seed(self):
        config = LedgerConfig(num_transactions=100, break_rate=0.1, seed=42)
        _, _, breaks1 = generate_ledgers(config)
        _, _, breaks2 = generate_ledgers(config)

        assert len(breaks1) == len(breaks2)
        assert breaks1[0].id == breaks2[0].id
        assert breaks1[0].break_type == breaks2[0].break_type

    def test_different_seeds_different_results(self):
        config1 = LedgerConfig(num_transactions=100, break_rate=0.1, seed=1)
        config2 = LedgerConfig(num_transactions=100, break_rate=0.1, seed=999)
        _, _, breaks1 = generate_ledgers(config1)
        _, _, breaks2 = generate_ledgers(config2)

        assert breaks1[0].id != breaks2[0].id

    def test_all_break_types_represented(self):
        config = LedgerConfig(num_transactions=5000, break_rate=0.3, seed=42)
        _, _, breaks = generate_ledgers(config)

        types = {b.break_type for b in breaks}
        assert len(types) >= 5

    def test_save_and_load_ledgers(self):
        config = LedgerConfig(num_transactions=100, break_rate=0.1, seed=42)
        df_a, df_b, breaks = generate_ledgers(config)

        with tempfile.TemporaryDirectory() as tmp:
            save_ledgers(df_a, df_b, breaks, tmp)
            assert Path(f"{tmp}/system_a.parquet").exists()
            assert Path(f"{tmp}/system_b.parquet").exists()
            assert Path(f"{tmp}/ground_truth.json").exists()

            with open(f"{tmp}/ground_truth.json") as f:
                gt = json.load(f)
            assert len(gt) == len(breaks)

    def test_transaction_dataclass(self):
        txn = Transaction(
            id="A-001", reference="TXN-001", system="SystemA",
            amount=100.0, currency="USD", counterparty="ACME Corp",
            timestamp="2025-01-01", status=TransactionStatus.SETTLED,
        )
        assert txn.amount == 100.0
        assert txn.system == "SystemA"

    def test_break_dataclass(self):
        brk = Break(
            id="BRK-001", break_type=BreakType.AMOUNT_MISMATCH,
            side_a_ref="A-001", side_b_ref="B-001",
            description="Amount mismatch", expected_resolution="use_smaller_amount",
            amount_delta=5.0,
        )
        assert brk.break_type == BreakType.AMOUNT_MISMATCH
        assert brk.amount_delta == 5.0

    def test_clean_transactions_match(self):
        config = LedgerConfig(num_transactions=100, break_rate=0.0, seed=42)
        df_a, df_b, breaks = generate_ledgers(config)

        assert len(breaks) == 0

        refs_a = set(df_a["reference"])
        refs_b = set(df_b["reference"])
        assert refs_a == refs_b

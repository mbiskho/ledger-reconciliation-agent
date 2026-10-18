"""Synthetic ledger generator — creates realistic payment ledgers with seeded breaks for reconciliation testing."""

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import structlog
from faker import Faker

logger = structlog.get_logger()

fake = Faker()
fake.seed_instance(42)


class BreakType(str, Enum):
    AMOUNT_MISMATCH = "amount_mismatch"
    MISSING_SIDE_A = "missing_side_a"
    MISSING_SIDE_B = "missing_side_b"
    DUPLICATE = "duplicate"
    DATE_SHIFT = "date_shift"
    CURRENCY_MISMATCH = "currency_mismatch"
    REFERENCE_TYPO = "reference_typo"
    SWAPPED_COUNTERPARTY = "swapped_counterparty"
    REVERSED_SIGN = "reversed_sign"
    FEE_DISCREPANCY = "fee_discrepancy"


class TransactionStatus(str, Enum):
    PENDING = "pending"
    SETTLED = "settled"
    FAILED = "failed"
    REVERSED = "reversed"


@dataclass
class Transaction:
    id: str
    reference: str
    system: str
    amount: float
    currency: str
    counterparty: str
    timestamp: datetime
    status: TransactionStatus
    description: str = ""
    fees: float = 0.0
    exchange_rate: float = 1.0
    settlement_date: Optional[datetime] = None


@dataclass
class Break:
    id: str
    break_type: BreakType
    side_a_ref: str
    side_b_ref: str
    description: str
    expected_resolution: str
    amount_delta: float = 0.0
    severity: str = "medium"


@dataclass
class LedgerConfig:
    num_transactions: int = 5000
    break_rate: float = 0.15
    date_range_days: int = 90
    currencies: list[str] = field(default_factory=lambda: ["USD", "EUR", "GBP", "JPY", "CHF"])
    seed: int = 42


def generate_ledgers(config: LedgerConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[Break]]:
    if config is None:
        config = LedgerConfig()

    random.seed(config.seed)
    np.random.seed(config.seed)
    Faker.seed(config.seed)

    num_clean = int(config.num_transactions * (1 - config.break_rate))
    num_breaks = config.num_transactions - num_clean

    end_date = datetime(2026, 1, 15)
    start_date = end_date - timedelta(days=config.date_range_days)

    side_a: list[Transaction] = []
    side_b: list[Transaction] = []
    breaks: list[Break] = []

    counter = 0
    counterparties = [fake.company() for _ in range(50)]

    for i in range(num_clean):
        ref = f"TXN-{counter:08d}"
        amount = round(random.uniform(10, 50000), 2)
        currency = random.choice(config.currencies)
        cp = random.choice(counterparties)
        ts = fake.date_time_between(start_date=start_date, end_date=end_date)
        status = random.choices([TransactionStatus.SETTLED, TransactionStatus.PENDING], weights=[0.9, 0.1])[0]
        desc = f"Payment to {cp}"

        side_a.append(Transaction(
            id=f"A-{counter:08d}", reference=ref, system="SystemA",
            amount=amount, currency=currency, counterparty=cp,
            timestamp=ts, status=status, description=desc,
        ))
        side_b.append(Transaction(
            id=f"B-{counter:08d}", reference=ref, system="SystemB",
            amount=amount, currency=currency, counterparty=cp,
            timestamp=ts, status=status, description=desc,
        ))
        counter += 1

    break_generators = [
        _amount_mismatch, _missing_side_a, _missing_side_b,
        _duplicate, _date_shift, _currency_mismatch,
        _reference_typo, _swapped_counterparty, _reversed_sign, _fee_discrepancy,
    ]

    for _ in range(num_breaks):
        gen = random.choice(break_generators)
        a_txn, b_txn, brk = gen(counter, counterparties, config, start_date, end_date)
        side_a.append(a_txn)
        side_b.append(b_txn)
        breaks.append(brk)
        counter += 1

    df_a = _to_dataframe(side_a)
    df_b = _to_dataframe(side_b)

    logger.info("ledgers generated", clean=num_clean, breaks=len(breaks), total=len(side_a))
    return df_a, df_b, breaks


def _to_dataframe(transactions: list[Transaction]) -> pd.DataFrame:
    return pd.DataFrame([{
        "id": t.id, "reference": t.reference, "system": t.system,
        "amount": t.amount, "currency": t.currency, "counterparty": t.counterparty,
        "timestamp": t.timestamp, "status": t.status.value,
        "description": t.description, "fees": t.fees,
        "exchange_rate": t.exchange_rate,
        "settlement_date": t.settlement_date,
    } for t in transactions])


def _amount_mismatch(counter: int, cps: list[str], cfg: LedgerConfig, start: datetime, end: datetime) -> tuple[Transaction, Transaction, Break]:
    ref = f"TXN-{counter:08d}"
    amount = round(random.uniform(100, 10000), 2)
    delta = round(random.uniform(0.01, amount * 0.1), 2)
    cp = random.choice(cps)
    ts = fake.date_time_between(start_date=start, end_date=end)

    a = Transaction(id=f"A-{counter:08d}", reference=ref, system="SystemA", amount=amount, currency="USD", counterparty=cp, timestamp=ts, status=TransactionStatus.SETTLED)
    b = Transaction(id=f"B-{counter:08d}", reference=ref, system="SystemB", amount=amount + delta, currency="USD", counterparty=cp, timestamp=ts, status=TransactionStatus.SETTLED)

    brk = Break(id=f"BRK-{counter:08d}", break_type=BreakType.AMOUNT_MISMATCH, side_a_ref=a.id, side_b_ref=b.id, description=f"Amount mismatch: A={amount:.2f}, B={amount+delta:.2f}", expected_resolution="use_smaller_amount", amount_delta=delta)
    return a, b, brk


def _missing_side_a(counter: int, cps: list[str], cfg: LedgerConfig, start: datetime, end: datetime) -> tuple[Transaction, Transaction, Break]:
    ref = f"TXN-{counter:08d}"
    amount = round(random.uniform(100, 50000), 2)
    cp = random.choice(cps)
    ts = fake.date_time_between(start_date=start, end_date=end)
    b = Transaction(id=f"B-{counter:08d}", reference=ref, system="SystemB", amount=amount, currency="USD", counterparty=cp, timestamp=ts, status=TransactionStatus.SETTLED)
    brk = Break(id=f"BRK-{counter:08d}", break_type=BreakType.MISSING_SIDE_A, side_a_ref="", side_b_ref=b.id, description=f"Transaction exists only in System B: {ref}", expected_resolution="flag_for_investigation")
    return Transaction(id=f"A-{counter:08d}", reference="", system="SystemA", amount=0, currency="", counterparty="", timestamp=ts, status=TransactionStatus.PENDING, description="PLACEHOLDER"), b, brk


def _missing_side_b(counter: int, cps: list[str], cfg: LedgerConfig, start: datetime, end: datetime) -> tuple[Transaction, Transaction, Break]:
    ref = f"TXN-{counter:08d}"
    amount = round(random.uniform(100, 50000), 2)
    cp = random.choice(cps)
    ts = fake.date_time_between(start_date=start, end_date=end)
    a = Transaction(id=f"A-{counter:08d}", reference=ref, system="SystemA", amount=amount, currency="USD", counterparty=cp, timestamp=ts, status=TransactionStatus.SETTLED)
    brk = Break(id=f"BRK-{counter:08d}", break_type=BreakType.MISSING_SIDE_B, side_a_ref=a.id, side_b_ref="", description=f"Transaction exists only in System A: {ref}", expected_resolution="flag_for_investigation")
    return a, Transaction(id=f"B-{counter:08d}", reference="", system="SystemB", amount=0, currency="", counterparty="", timestamp=ts, status=TransactionStatus.PENDING, description="PLACEHOLDER"), brk


def _duplicate(counter: int, cps: list[str], cfg: LedgerConfig, start: datetime, end: datetime) -> tuple[Transaction, Transaction, Break]:
    ref = f"TXN-{counter:08d}"
    amount = round(random.uniform(500, 20000), 2)
    cp = random.choice(cps)
    ts = fake.date_time_between(start_date=start, end_date=end)
    a = Transaction(id=f"A-{counter:08d}", reference=ref, system="SystemA", amount=amount, currency="USD", counterparty=cp, timestamp=ts, status=TransactionStatus.SETTLED)
    b = Transaction(id=f"B-{counter:08d}", reference=ref, system="SystemB", amount=amount, currency="USD", counterparty=cp, timestamp=ts, status=TransactionStatus.SETTLED)
    brk = Break(id=f"BRK-{counter:08d}", break_type=BreakType.DUPLICATE, side_a_ref=a.id, side_b_ref=b.id, description=f"Duplicate entry for {ref} in System B", expected_resolution="deduplicate_keep_one")
    return a, b, brk


def _date_shift(counter: int, cps: list[str], cfg: LedgerConfig, start: datetime, end: datetime) -> tuple[Transaction, Transaction, Break]:
    ref = f"TXN-{counter:08d}"
    amount = round(random.uniform(100, 30000), 2)
    cp = random.choice(cps)
    ts_a = fake.date_time_between(start_date=start, end_date=end)
    ts_b = ts_a + timedelta(days=random.randint(1, 5))
    a = Transaction(id=f"A-{counter:08d}", reference=ref, system="SystemA", amount=amount, currency="USD", counterparty=cp, timestamp=ts_a, status=TransactionStatus.SETTLED)
    b = Transaction(id=f"B-{counter:08d}", reference=ref, system="SystemB", amount=amount, currency="USD", counterparty=cp, timestamp=ts_b, status=TransactionStatus.SETTLED)
    brk = Break(id=f"BRK-{counter:08d}", break_type=BreakType.DATE_SHIFT, side_a_ref=a.id, side_b_ref=b.id, description=f"Date mismatch: A={ts_a.date()}, B={ts_b.date()}", expected_resolution="use_earlier_date")
    return a, b, brk


def _currency_mismatch(counter: int, cps: list[str], cfg: LedgerConfig, start: datetime, end: datetime) -> tuple[Transaction, Transaction, Break]:
    ref = f"TXN-{counter:08d}"
    amount = round(random.uniform(1000, 50000), 2)
    cp = random.choice(cps)
    ts = fake.date_time_between(start_date=start, end_date=end)
    fx = round(random.uniform(0.8, 1.3), 4)
    cur_a, cur_b = random.sample(cfg.currencies, 2)

    a = Transaction(id=f"A-{counter:08d}", reference=ref, system="SystemA", amount=amount, currency=cur_a, counterparty=cp, timestamp=ts, status=TransactionStatus.SETTLED, exchange_rate=1.0)
    b = Transaction(id=f"B-{counter:08d}", reference=ref, system="SystemB", amount=round(amount * fx, 2), currency=cur_b, counterparty=cp, timestamp=ts, status=TransactionStatus.SETTLED, exchange_rate=fx)
    brk = Break(id=f"BRK-{counter:08d}", break_type=BreakType.CURRENCY_MISMATCH, side_a_ref=a.id, side_b_ref=b.id, description=f"Currency mismatch: A={cur_a} {amount:.2f}, B={cur_b} {amount*fx:.2f} (rate={fx})", expected_resolution="normalize_to_base_currency", amount_delta=amount * abs(1 - fx))
    return a, b, brk


def _reference_typo(counter: int, cps: list[str], cfg: LedgerConfig, start: datetime, end: datetime) -> tuple[Transaction, Transaction, Break]:
    ref_a = f"TXN-{counter:08d}"
    ref_b = _typo_variant(ref_a)
    amount = round(random.uniform(100, 25000), 2)
    cp = random.choice(cps)
    ts = fake.date_time_between(start_date=start, end_date=end)
    a = Transaction(id=f"A-{counter:08d}", reference=ref_a, system="SystemA", amount=amount, currency="USD", counterparty=cp, timestamp=ts, status=TransactionStatus.SETTLED)
    b = Transaction(id=f"B-{counter:08d}", reference=ref_b, system="SystemB", amount=amount, currency="USD", counterparty=cp, timestamp=ts, status=TransactionStatus.SETTLED)
    brk = Break(id=f"BRK-{counter:08d}", break_type=BreakType.REFERENCE_TYPO, side_a_ref=a.id, side_b_ref=b.id, description=f"Reference typo: A='{ref_a}', B='{ref_b}'", expected_resolution="fuzzy_match_reference")
    return a, b, brk


def _typo_variant(ref: str) -> str:
    chars = list(ref)
    i = random.randint(1, len(chars) - 2)
    if random.random() < 0.5:
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    else:
        chars[i] = random.choice("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-")
    return "".join(chars)


def _swapped_counterparty(counter: int, cps: list[str], cfg: LedgerConfig, start: datetime, end: datetime) -> tuple[Transaction, Transaction, Break]:
    ref = f"TXN-{counter:08d}"
    amount = round(random.uniform(500, 30000), 2)
    cp_a, cp_b = random.sample(cps, 2)
    ts = fake.date_time_between(start_date=start, end_date=end)
    a = Transaction(id=f"A-{counter:08d}", reference=ref, system="SystemA", amount=amount, currency="USD", counterparty=cp_a, timestamp=ts, status=TransactionStatus.SETTLED)
    b = Transaction(id=f"B-{counter:08d}", reference=ref, system="SystemB", amount=amount, currency="USD", counterparty=cp_b, timestamp=ts, status=TransactionStatus.SETTLED)
    brk = Break(id=f"BRK-{counter:08d}", break_type=BreakType.SWAPPED_COUNTERPARTY, side_a_ref=a.id, side_b_ref=b.id, description=f"Counterparty mismatch: A='{cp_a}', B='{cp_b}'", expected_resolution="use_system_a_counterparty")
    return a, b, brk


def _reversed_sign(counter: int, cps: list[str], cfg: LedgerConfig, start: datetime, end: datetime) -> tuple[Transaction, Transaction, Break]:
    ref = f"TXN-{counter:08d}"
    amount = round(random.uniform(100, 20000), 2)
    cp = random.choice(cps)
    ts = fake.date_time_between(start_date=start, end_date=end)
    a = Transaction(id=f"A-{counter:08d}", reference=ref, system="SystemA", amount=amount, currency="USD", counterparty=cp, timestamp=ts, status=TransactionStatus.SETTLED)
    b = Transaction(id=f"B-{counter:08d}", reference=ref, system="SystemB", amount=-amount, currency="USD", counterparty=cp, timestamp=ts, status=TransactionStatus.SETTLED)
    brk = Break(id=f"BRK-{counter:08d}", break_type=BreakType.REVERSED_SIGN, side_a_ref=a.id, side_b_ref=b.id, description=f"Sign mismatch: A={amount:.2f}, B={-amount:.2f}", expected_resolution="use_absolute_value")
    return a, b, brk


def _fee_discrepancy(counter: int, cps: list[str], cfg: LedgerConfig, start: datetime, end: datetime) -> tuple[Transaction, Transaction, Break]:
    ref = f"TXN-{counter:08d}"
    amount = round(random.uniform(1000, 50000), 2)
    fee_a = round(random.uniform(0, amount * 0.02), 2)
    fee_b = round(fee_a * random.uniform(0.5, 2.0), 2)
    cp = random.choice(cps)
    ts = fake.date_time_between(start_date=start, end_date=end)
    a = Transaction(id=f"A-{counter:08d}", reference=ref, system="SystemA", amount=amount, currency="USD", counterparty=cp, timestamp=ts, status=TransactionStatus.SETTLED, fees=fee_a)
    b = Transaction(id=f"B-{counter:08d}", reference=ref, system="SystemB", amount=amount, currency="USD", counterparty=cp, timestamp=ts, status=TransactionStatus.SETTLED, fees=fee_b)
    brk = Break(id=f"BRK-{counter:08d}", break_type=BreakType.FEE_DISCREPANCY, side_a_ref=a.id, side_b_ref=b.id, description=f"Fee mismatch: A={fee_a:.2f}, B={fee_b:.2f}", expected_resolution="flag_for_review", amount_delta=abs(fee_a - fee_b))
    return a, b, brk


def save_ledgers(df_a: pd.DataFrame, df_b: pd.DataFrame, breaks: list[Break], output_dir: str) -> None:
    import json
    import os

    os.makedirs(output_dir, exist_ok=True)
    df_a.to_parquet(f"{output_dir}/system_a.parquet")
    df_b.to_parquet(f"{output_dir}/system_b.parquet")

    breaks_data = [{
        "id": b.id, "break_type": b.break_type.value, "side_a_ref": b.side_a_ref,
        "side_b_ref": b.side_b_ref, "description": b.description,
        "expected_resolution": b.expected_resolution, "amount_delta": b.amount_delta,
        "severity": b.severity,
    } for b in breaks]

    with open(f"{output_dir}/ground_truth.json", "w") as f:
        json.dump(breaks_data, f, indent=2)

    logger.info("ledgers saved", path=output_dir)

"""Регрессии временных признаков, влияющих на роли и приоритет."""
import math

import pandas as pd
import pytest

from moneygraph.features import _fast_forward, _time_consistent, temporal


def test_fifo_old_balance_is_spent_before_recent_receipt():
    share, lag = _fast_forward([(1, 100_000, 1), (10, 10_000, 2)], [(10, 10_000, 3)])
    assert share == 0
    assert lag == 9


def test_fifo_only_recent_part_of_partial_payment_is_fast():
    share, lag = _fast_forward([(1, 100_000, 1), (10, 10_000, 2)], [(10, 105_000, 3)])
    assert share == pytest.approx(5_000 / 110_000)
    assert lag == 9


def test_fifo_never_spends_future_money_or_reuses_lots():
    share, lag = _fast_forward([(5, 20_000, 1)], [(2, 90_000, 2), (5, 10_000, 2), (6, 50_000, 3)])
    assert share == 1
    assert lag == 0


def test_fifo_with_no_matching_receipts():
    share, lag = _fast_forward([(4, 10_000, 1)], [(1, 10_000, 2)])
    assert share == 0
    assert math.isnan(lag)


def test_temporal_patterns_cross_month_boundary():
    transactions = pd.DataFrame({
        "src": [1, 2], "dst": [2, 3], "sum_kzt": [10_000.0, 10_000.0],
        "date": pd.to_datetime(["2026-07-31", "2026-08-01"]),
    })
    result = temporal(transactions, pd.DataFrame(index=[1, 2, 3]))
    assert result.at[2, "fast_in_share"] == 1
    assert result.at[2, "lag_median_days"] == 1
    assert result.at[2, "sync_date"] == "2026-07-31"


def test_return_cycle_requires_monotone_dates():
    assert _time_consistent([(1, 2), (2, 1)], {(1, 2): [10, 20], (2, 1): [15]})
    assert not _time_consistent([(1, 2), (2, 1)], {(1, 2): [20], (2, 1): [15]})

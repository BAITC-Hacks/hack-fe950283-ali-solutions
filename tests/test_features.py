"""Регрессии временных признаков, влияющих на роли и приоритет."""
import math

import pandas as pd
import pytest

from moneygraph import config as C
from moneygraph.features import _busiest_window, _fast_forward, _time_consistent, temporal


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
    result, _ = temporal(transactions, pd.DataFrame(index=[1, 2, 3]))
    assert result.at[2, "fast_in_share"] == 1
    assert result.at[2, "lag_median_days"] == 1
    assert result.at[2, "sync_date"] == "2026-07-31"


def test_return_cycle_requires_monotone_dates():
    assert _time_consistent([(1, 2), (2, 1)], {(1, 2): [10, 20], (2, 1): [15]})
    assert not _time_consistent([(1, 2), (2, 1)], {(1, 2): [20], (2, 1): [15]})


def test_busiest_window_counts_transfers_within_window():
    assert _busiest_window([1, 5, 6, 7, 7, 20]) == (4, 5)
    assert _busiest_window([3]) == (1, 3)


def test_burst_needs_enough_transfers_and_majority_share():
    days = [10] * C.BURST_MIN_TX + [25]
    transactions = pd.DataFrame({
        "src": [1] * len(days), "dst": list(range(2, 2 + len(days))), "sum_kzt": [10_000.0] * len(days),
        "date": pd.Timestamp("2026-07-01") + pd.to_timedelta([d - 1 for d in days], unit="D"),
    })
    result, _ = temporal(transactions, pd.DataFrame(index=range(1, 2 + len(days))))
    assert result.at[1, "burst"] and result.at[1, "burst_tx"] == C.BURST_MIN_TX and result.at[1, "burst_date"] == "2026-07-10"
    assert not result.at[2, "burst"]  # один перевод — не всплеск


def test_stable_route_listing_matches_counts():
    transactions = pd.DataFrame({
        "src": [1, 2, 1, 2, 1, 2], "dst": [2, 3, 2, 3, 2, 4], "sum_kzt": [10_000.0] * 6,
        "date": pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-10", "2026-07-11", "2026-07-20", "2026-07-25"]),
    })
    result, routes = temporal(transactions, pd.DataFrame(index=[1, 2, 3, 4]))
    assert routes == [{"a": 1, "b": 2, "c": 3, "days": [1, 10], "dates": ["2026-07-01", "2026-07-10"]}]
    assert result.at[2, "repeated_routes"] == 1

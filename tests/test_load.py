"""Ошибки данных должны выявляться до расчёта ролей."""
import numpy as np
import pandas as pd
import pytest

from moneygraph.load import sanity_check


@pytest.fixture
def tables():
    nodes = pd.DataFrame({"gid": [101, 102], "depth": [0, 1], "is_seed": [True, False]})
    edges = pd.DataFrame({"src": [101], "dst": [102], "sum_kzt": [10_000.0], "n_tx": [1], "depth": [1]})
    transactions = pd.DataFrame({"src": [101], "dst": [102], "date": pd.to_datetime(["2026-07-01"]), "sum_kzt": [10_000.0]})
    return edges, nodes, transactions


def test_valid_data(tables):
    assert sanity_check(*tables)["n_nodes"] == 2


@pytest.mark.parametrize("problem, expected", [
    ("duplicate_node", "дублирующиеся gid"), ("duplicate_edge", "дублирующиеся пары"),
    ("floating_gid", "теряет точность"), ("unknown_gid", "вне nodes"),
    ("nonfinite_amount", "конечные суммы"), ("below_threshold", "5 000"),
    ("aggregate_mismatch", "суммы edges"), ("count_mismatch", "n_tx в edges"),
    ("wrong_depth", "колену отправителя"), ("wrong_seed", "is_seed должен"),
    ("missing_column", "отсутствуют колонки"), ("missing_value", "пропущены"),
])
def test_invalid_data_is_rejected(tables, problem, expected):
    edges, nodes, transactions = tables
    if problem == "duplicate_node":
        nodes = pd.concat([nodes, nodes.iloc[:1]])
    elif problem == "duplicate_edge":
        edges = pd.concat([edges, edges])
    elif problem == "floating_gid":
        nodes["gid"] = nodes.gid.astype(float)
    elif problem == "unknown_gid":
        transactions.loc[0, "dst"] = 999
    elif problem == "nonfinite_amount":
        transactions.loc[0, "sum_kzt"] = np.inf
    elif problem == "below_threshold":
        transactions.loc[0, "sum_kzt"] = 4999
    elif problem == "aggregate_mismatch":
        edges.loc[0, "sum_kzt"] += 0.5
    elif problem == "count_mismatch":
        edges.loc[0, "n_tx"] = 2
    elif problem == "wrong_depth":
        edges.loc[0, "depth"] = 2
    elif problem == "wrong_seed":
        nodes.loc[1, "is_seed"] = True
    elif problem == "missing_column":
        nodes = nodes.drop(columns="depth")
    elif problem == "missing_value":
        transactions.loc[0, "date"] = pd.NaT
    with pytest.raises(ValueError, match=expected):
        sanity_check(edges, nodes, transactions)

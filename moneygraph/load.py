"""Загрузка parquet и проверка консистентности выгрузки."""
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from . import config as C

SCHEMAS = {
    "nodes": ["gid", "depth", "is_seed"],
    "edges": ["src", "dst", "sum_kzt", "n_tx", "depth"],
    "transactions": ["src", "dst", "date", "sum_kzt"],
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _columns(name, table):
    missing = set(SCHEMAS[name]) - set(table.columns)
    _require(not missing, f"{name}: отсутствуют колонки {', '.join(sorted(missing))}")


def load(data_dir: Path):
    tables = {}
    for name in SCHEMAS:
        path = data_dir / f"{name}.parquet"
        _require(path.is_file(), f"Не найден входной файл: {path}")
        tables[name] = pd.read_parquet(path)
        _columns(name, tables[name])
    edges, nodes, tx = tables["edges"], tables["nodes"], tables["transactions"]
    tx["date"] = pd.to_datetime(tx["date"], errors="raise")
    edges = edges.sort_values(["src", "dst"]).reset_index(drop=True)
    nodes = nodes.sort_values("gid").reset_index(drop=True)
    tx = tx.sort_values(["date", "src", "dst", "sum_kzt"]).reset_index(drop=True)
    return edges, nodes, tx


def sanity_check(edges, nodes, tx) -> dict:
    """Проверки до построения модели. Падает, если выгрузка несогласована."""
    for name, table in (("nodes", nodes), ("edges", edges), ("transactions", tx)):
        _columns(name, table)
        _require(not table.empty, f"{name}: пустая таблица")
        _require(table[SCHEMAS[name]].notna().all().all(), f"{name}: пропущены обязательные значения")
        for column in (["gid"] if name == "nodes" else ["src", "dst"]):
            identifiers = table[column]
            _require(pd.api.types.is_integer_dtype(identifiers), f"{name}.{column}: нужен целочисленный int64, float теряет точность gid")
            _require(identifiers.between(0, np.iinfo(np.int64).max).all(), f"{name}.{column}: gid вне диапазона int64")
    _require(nodes.gid.is_unique, "nodes: дублирующиеся gid")
    _require(not edges.duplicated(["src", "dst"]).any(), "edges: дублирующиеся пары src/dst")
    _require(pd.api.types.is_bool_dtype(nodes.is_seed), "nodes.is_seed: нужен bool")
    for name, values in (("nodes.depth", nodes.depth), ("edges.depth", edges.depth), ("edges.n_tx", edges.n_tx)):
        _require(pd.api.types.is_integer_dtype(values), f"{name}: нужен целочисленный тип")
    _require(nodes.depth.between(0, C.MAX_DEPTH).all(), "nodes.depth: допустимы колена 0–4")
    _require((nodes.is_seed == (nodes.depth == 0)).all(), "nodes: is_seed должен соответствовать depth=0")
    _require(nodes.is_seed.any(), "nodes: нет ни одного seed")
    _require(edges.depth.between(1, C.MAX_DEPTH).all(), "edges.depth: допустимы колена 1–4")
    _require((edges.n_tx > 0).all(), "edges.n_tx: число переводов должно быть положительным")
    for name, values in (("edges.sum_kzt", edges.sum_kzt), ("transactions.sum_kzt", tx.sum_kzt)):
        _require(pd.api.types.is_numeric_dtype(values), f"{name}: нужен числовой тип")
        _require(np.isfinite(values).all() and (values >= C.MIN_TX_KZT).all(), f"{name}: нужны конечные суммы ≥5 000 KZT")
    _require(pd.api.types.is_datetime64_any_dtype(tx.date), "transactions.date: некорректные даты")
    known = set(nodes.gid)
    _require(set(edges.src) | set(edges.dst) | set(tx.src) | set(tx.dst) <= known, "в переводах есть gid вне nodes")
    depths = nodes.set_index("gid").depth
    _require((edges.src.map(depths) + 1 == edges.depth).all(), "edges.depth не соответствует колену отправителя + 1")
    _require((edges.dst.map(depths) <= edges.depth).all(), "nodes.depth получателя противоречит глубине обхода")
    agg = tx.groupby(["src", "dst"]).agg(s=("sum_kzt", "sum"), c=("sum_kzt", "size")).reset_index()
    m = edges.merge(agg, on=["src", "dst"], how="outer", indicator=True)
    _require((m._merge == "both").all(), "edges и transactions не сходятся по парам")
    _require(np.allclose(m.s, m.sum_kzt, rtol=0, atol=0.01), "суммы edges != суммы transactions")
    _require((m.c == m.n_tx).all(), "n_tx в edges != числу транзакций")
    in_edges = set(edges.src) | set(edges.dst)
    seeds = set(nodes.loc[nodes.is_seed, "gid"])
    return {
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_tx": len(tx),
        "n_seed": len(seeds),
        "turnover_kzt": float(edges.sum_kzt.sum()),
        "date_min": str(tx.date.min().date()),
        "date_max": str(tx.date.max().date()),
        "isolated_seeds": len(seeds - in_edges),
        "seed_without_outgoing": len(seeds - set(edges.src)),
        "truncated_nodes": int((nodes.depth == C.MAX_DEPTH).sum()),
    }


def build_graph(edges: pd.DataFrame, nodes: pd.DataFrame) -> nx.DiGraph:
    """Направленный граф: вес ребра sum_kzt, n_tx — число переводов."""
    G = nx.DiGraph()
    G.add_nodes_from(nodes.gid)
    for r in edges.itertuples(index=False):
        G.add_edge(r.src, r.dst, sum_kzt=float(r.sum_kzt), n_tx=int(r.n_tx))
    return G

"""Загрузка parquet и проверка консистентности выгрузки."""
from pathlib import Path

import networkx as nx
import pandas as pd


def load(data_dir: Path):
    edges = pd.read_parquet(data_dir / "edges.parquet")
    nodes = pd.read_parquet(data_dir / "nodes.parquet")
    tx = pd.read_parquet(data_dir / "transactions.parquet")
    tx["date"] = pd.to_datetime(tx["date"])
    return edges, nodes, tx


def sanity_check(edges, nodes, tx) -> dict:
    """Проверки до построения модели. Падает, если выгрузка несогласована."""
    agg = tx.groupby(["src", "dst"]).agg(s=("sum_kzt", "sum"), c=("sum_kzt", "size")).reset_index()
    m = edges.merge(agg, on=["src", "dst"], how="outer", indicator=True)
    assert (m._merge == "both").all(), "edges и transactions не сходятся по парам"
    assert (m.s - m.sum_kzt).abs().max() < 1, "суммы edges != суммы transactions"
    assert (m.c == m.n_tx).all(), "n_tx в edges != числу транзакций"
    known = set(nodes.gid)
    assert set(edges.src) <= known and set(edges.dst) <= known, "в рёбрах есть gid вне nodes"
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
    }


def build_graph(edges: pd.DataFrame, nodes: pd.DataFrame) -> nx.DiGraph:
    """Направленный граф: вес ребра sum_kzt, n_tx — число переводов."""
    G = nx.DiGraph()
    G.add_nodes_from(nodes.gid)
    for r in edges.itertuples(index=False):
        G.add_edge(r.src, r.dst, sum_kzt=float(r.sum_kzt), n_tx=int(r.n_tx))
    return G

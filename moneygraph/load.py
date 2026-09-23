"""Load and validate the observed network without silently repairing input."""
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd

SCHEMAS = {"nodes": ("gid", "depth", "is_seed"),
           "edges": ("src", "dst", "sum_kzt", "n_tx", "depth"),
           "transactions": ("src", "dst", "date", "sum_kzt")}
SUM_ATOL, SUM_RTOL = 0.01, 1e-10


def load(data_dir: Path):
    tables = {}
    for name, columns in SCHEMAS.items():
        path = data_dir / f"{name}.parquet"
        if not path.is_file():
            raise ValueError(f"Отсутствует файл: {path}")
        table = pd.read_parquet(path)
        missing = set(columns) - set(table.columns)
        if missing:
            raise ValueError(f"{name}: отсутствуют колонки {sorted(missing)}")
        tables[name] = table
    edges, nodes, tx = tables["edges"], tables["nodes"], tables["transactions"]
    try:
        tx["date"] = pd.to_datetime(tx["date"], errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("transactions: неверная календарная дата") from exc
    sanity_check(edges, nodes, tx)
    return (edges.sort_values(["src", "dst"]).reset_index(drop=True),
            nodes.sort_values("gid").reset_index(drop=True),
            tx.sort_values(["date", "src", "dst", "sum_kzt"]).reset_index(drop=True))


def sanity_check(edges, nodes, tx) -> dict:
    for name, table in (("nodes", nodes), ("edges", edges), ("transactions", tx)):
        if not set(SCHEMAS[name]) <= set(table.columns):
            raise ValueError(f"{name}: неверная схема")
        if table[list(SCHEMAS[name])].isna().any().any():
            raise ValueError(f"{name}: пустые обязательные значения")
        for col in (("gid",) if name == "nodes" else ("src", "dst")):
            if not pd.api.types.is_integer_dtype(table[col].dtype):
                raise ValueError(f"{name}.{col}: нужен int64, преобразование из float запрещено")
            if len(table) and not table[col].between(0, np.iinfo(np.int64).max).all():
                raise ValueError(f"{name}.{col}: вне диапазона int64")
    if nodes.empty or not nodes.gid.is_unique:
        raise ValueError("nodes: нужен непустой набор уникальных gid")
    if not pd.api.types.is_bool_dtype(nodes.is_seed.dtype):
        raise ValueError("nodes.is_seed: требуется bool")
    for name, table, low, high in (("nodes", nodes, 0, 4), ("edges", edges, 1, 4)):
        if not pd.api.types.is_integer_dtype(table.depth.dtype) or not table.depth.between(low, high).all():
            raise ValueError(f"{name}.depth: ожидаются целые {low}..{high}")
    if edges.duplicated(["src", "dst"]).any():
        raise ValueError("edges: повтор пары src,dst")
    known = set(nodes.gid)
    for name, table in (("edges", edges), ("transactions", tx)):
        if not set(table.src) <= known or not set(table.dst) <= known:
            raise ValueError(f"{name}: endpoint вне nodes")
        if (not pd.api.types.is_numeric_dtype(table.sum_kzt.dtype)
                or not np.isfinite(table.sum_kzt).all() or not (table.sum_kzt > 0).all()):
            raise ValueError(f"{name}: суммы должны быть конечными и положительными")
    if not pd.api.types.is_integer_dtype(edges.n_tx.dtype) or not (edges.n_tx > 0).all():
        raise ValueError("edges.n_tx: требуется положительное целое")
    if not pd.api.types.is_datetime64_any_dtype(tx.date.dtype) or tx.date.isna().any():
        raise ValueError("transactions.date: неверные даты")
    if tx.date.dt.tz is not None or not tx.date.eq(tx.date.dt.normalize()).all():
        raise ValueError("transactions.date: нужны календарные даты без времени и timezone")
    agg = tx.groupby(["src", "dst"]).agg(s=("sum_kzt", "sum"), c=("sum_kzt", "size")).reset_index()
    m = edges.merge(agg, on=["src", "dst"], how="outer", indicator=True)
    if not (m._merge == "both").all():
        raise ValueError("edges и transactions не сходятся по парам")
    if not np.isclose(m.s, m.sum_kzt, atol=SUM_ATOL, rtol=SUM_RTOL).all():
        raise ValueError("суммы edges != суммы transactions")
    if not (m.c == m.n_tx).all():
        raise ValueError("n_tx в edges != числу транзакций")
    in_edges = set(edges.src) | set(edges.dst)
    seeds = set(nodes.loc[nodes.is_seed, "gid"])
    graph = build_graph(edges, nodes)
    components = list(nx.weakly_connected_components(graph))
    warnings = ["Наблюдаемые потоки не являются полным балансом; неизвестен начальный остаток."]
    if (tx.sum_kzt < 5000).any():
        warnings.append("Есть операции ниже порога 5000 KZT, заявленного для исходного набора.")
    return {"n_nodes": len(nodes), "n_edges": len(edges), "n_tx": len(tx), "n_seed": len(seeds),
            "turnover_kzt": float(edges.sum_kzt.sum()),
            "date_min": str(tx.date.min().date()) if len(tx) else None,
            "date_max": str(tx.date.max().date()) if len(tx) else None,
            "isolated_seeds": len(seeds - in_edges), "isolates": len(set(nodes.gid) - in_edges),
            "components_all": len(components),
            "components_without_isolates": sum(any(graph.degree(g) for g in c) for c in components),
            "self_loops": int((edges.src == edges.dst).sum()),
            "duplicate_transaction_rows": int(tx.duplicated().sum()),
            "sum_atol": SUM_ATOL, "sum_rtol": SUM_RTOL, "warnings": warnings}


def build_graph(edges, nodes) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(nodes.gid))
    for row in edges.sort_values(["src", "dst"]).itertuples(index=False):
        graph.add_edge(row.src, row.dst, sum_kzt=float(row.sum_kzt), n_tx=int(row.n_tx))
    return graph

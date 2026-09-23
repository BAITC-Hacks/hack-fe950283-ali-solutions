"""Устойчивость сети: что станет со структурой, если исключить из копии графа N узлов.

Сравниваем стратегии с одинаковым N: наш топ по priority_score, топ по обороту,
seed с наибольшими исходящими (типичная практика — блокировать известных курьеров)
и случайный выбор (среднее по 30 прогонам).
"""
import networkx as nx
import numpy as np
import pandas as pd

from . import config as C


def _metrics(G: nx.DiGraph, removed, total_kzt: float) -> dict:
    H = G.copy()
    H.remove_nodes_from(removed)
    comps = sorted(nx.weakly_connected_components(H), key=len, reverse=True)
    giant = comps[0] if comps else set()
    giant_kzt = sum(d["sum_kzt"] for u, v, d in H.edges(data=True) if u in giant)
    return {
        "giant_nodes": len(giant),
        "fragments_3plus": sum(len(c) >= 3 for c in comps),
        "giant_turnover_share": giant_kzt / total_kzt if total_kzt else 0.0,
        "edges_left_share": H.number_of_edges() / G.number_of_edges() if G.number_of_edges() else 0.0,
    }


def simulate(G: nx.DiGraph, df: pd.DataFrame, ns=(5, 10, 20, 50)) -> pd.DataFrame:
    total = sum(d["sum_kzt"] for _, _, d in G.edges(data=True))
    rng = np.random.default_rng(C.RANDOM_SEED)
    turnover = df[["in_kzt", "out_kzt"]].max(axis=1)
    strategies = {
        "наш топ (priority_score)": df.sort_values("priority_score", ascending=False).index,
        "топ по обороту": turnover.sort_values(ascending=False).index,
        "seed с макс. исходящими": df[df.is_seed].out_kzt.sort_values(ascending=False).index,
    }
    rows = [{"strategy": "исходная сеть", "n_removed": 0, **_metrics(G, [], total)}]
    for n in sorted({min(n, len(df)) for n in ns}):
        for name, order in strategies.items():
            rows.append({"strategy": name, "n_removed": min(n, len(order)), **_metrics(G, list(order[:n]), total)})
        rand = [_metrics(G, list(rng.choice(df.index, n, replace=False)), total) for _ in range(30)]
        rows.append({"strategy": "случайные", "n_removed": n,
                     **{k: float(np.mean([r[k] for r in rand])) for k in rand[0]}})
    out = pd.DataFrame(rows)
    for c in ("giant_turnover_share", "edges_left_share"):
        out[c] = out[c].round(3)
    out["giant_nodes"] = out.giant_nodes.round(0).astype(int)
    out["fragments_3plus"] = out.fragments_3plus.round(1)
    return out

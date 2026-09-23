"""Три выгрузки фиксированной схемы ТЗ + механическая проверка схемы."""
from pathlib import Path

import pandas as pd

from . import config as C

NODE_COLS = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]
NODE_EXTRA = [
    "rank", "role_secondary", "flags", "depth", "is_seed", "in_deg", "out_deg", "in_kzt", "out_kzt",
    "in_tx", "out_tx", "pass_through", "truncated", "p_forward", "seed_payers", "seed_upstream",
    "seed_kzt_in", "seed_share", "key_links", "pagerank", "betweenness", "fast_in_share",
    "lag_median_days", "sync_payers_max", "n_return_cycles", "repeated_routes", "split_days",
    "fast_status", "fast_matched_kzt", "fast_eligible_kzt", "fast_coverage", "same_day_kzt",
    "priority_raw", "priority_seed_factor", "priority_normalizer", "anomaly", "c_role", "c_seed", "c_volume", "c_network", "c_patterns",
]
CLUSTER_COLS = ["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"]
TOP_COLS = ["rank", "gid", "role", "priority_score", "why"]


def write(out: Path, df: pd.DataFrame, clusters: pd.DataFrame, extra: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    nodes = df.reset_index().rename(columns={"index": "gid"}).sort_values("rank")
    nodes = nodes[NODE_COLS + NODE_EXTRA]
    nodes.to_csv(out / "nodes_roles.csv", index=False)
    clusters.to_csv(out / "clusters.csv", index=False)
    top = nodes.head(C.TOP_N).merge(df[["why"]], left_on="gid", right_index=True)
    top[TOP_COLS + ["role_score", "cluster_id", "is_seed", "depth", "evidence"]].to_csv(out / "top_nodes.csv", index=False)
    for name, table in extra.items():
        table.to_csv(out / name, index=False)


def validate(out: Path, n_nodes: int, gids: set) -> list:
    """Проверки из ТЗ (must have 2, 4, 5). Возвращает список (ok, текст)."""
    nr = pd.read_csv(out / "nodes_roles.csv")
    cl = pd.read_csv(out / "clusters.csv")
    tp = pd.read_csv(out / "top_nodes.csv")
    checks = [
        (len(nr) == n_nodes, f"nodes_roles.csv: {len(nr)} строк (нужно {n_nodes})"),
        (set(nr.gid) == gids and nr.gid.is_unique, "nodes_roles.csv: каждый gid ровно один раз"),
        (nr[NODE_COLS].notna().all().all() and (nr.evidence.str.len() > 0).all(), "обязательные колонки заполнены"),
        (nr.role.isin(C.ROLES).all(), "все роли из словаря ТЗ"),
        (nr.role_score.between(0, 1).all() and nr.priority_score.between(0, 1).all(), "role_score и priority_score в [0, 1]"),
        ((nr.evidence.str.len() <= 200).all() and nr.evidence.str.contains(r"\d").all(), "evidence ≤200 символов и содержит числа"),
        (set(nr.cluster_id) == set(cl.cluster_id) and cl.cluster_id.is_unique, "cluster_id каждого узла есть в clusters.csv"),
        (cl[CLUSTER_COLS].notna().all().all() and cl.n_nodes.sum() == n_nodes, "clusters.csv заполнен, размеры сходятся"),
        (len(tp) >= min(20, n_nodes) and tp.priority_score.is_monotonic_decreasing and list(tp["rank"]) == list(range(1, len(tp) + 1)),
         f"top_nodes.csv: {len(tp)} строк, отсортирован по приоритету"),
        (tp.why.notna().all() and (tp.why.str.len() > 0).all(), "у каждой позиции топа есть обоснование"),
    ]
    expected = nr.set_index("gid").reindex(tp.gid)
    checks.extend([
        (tp.gid.is_unique and set(tp.gid) <= gids, "top: уникальные известные gid"),
        (list(tp.role) == list(expected.role) and list(tp.priority_score) == list(expected.priority_score), "top совпадает с nodes_roles"),
        (int(cl.n_seed.sum()) == int(nr.is_seed.sum()), "число seed в кластерах совпадает"),
        (list(tp.gid) == list(tp.sort_values(["priority_score", "gid"], ascending=[False, True]).gid), "равные приоритеты отсортированы по gid"),
    ])
    return checks

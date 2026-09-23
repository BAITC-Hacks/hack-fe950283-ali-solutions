"""Три выгрузки фиксированной схемы ТЗ + механическая проверка схемы."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

NODE_COLS = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]
NODE_EXTRA = [
    "rank", "role_secondary", "flags", "depth", "is_seed", "in_deg", "out_deg", "in_kzt", "out_kzt",
    "in_tx", "out_tx", "pass_through", "truncated", "p_forward", "seed_payers", "seed_upstream",
    "seed_kzt_in", "seed_share", "key_links", "pagerank", "betweenness", "fast_in_share",
    "lag_median_days", "sync_payers_max", "n_return_cycles", "repeated_routes", "split_days",
    "anomaly", "c_role", "c_seed", "c_volume", "c_network", "c_patterns",
]
CLUSTER_COLS = ["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"]
TOP_COLS = ["rank", "gid", "role", "priority_score", "why"]


def write(out: Path, df: pd.DataFrame, clusters: pd.DataFrame, extra: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    nodes = df.reset_index().rename(columns={"index": "gid"}).sort_values("rank")
    nodes = nodes[NODE_COLS + NODE_EXTRA].round(6)
    nodes[NODE_COLS].to_csv(out / "nodes_roles.csv", index=False)
    nodes.to_csv(out / "nodes_features.csv", index=False)
    clusters[CLUSTER_COLS].to_csv(out / "clusters.csv", index=False)
    clusters.to_csv(out / "clusters_details.csv", index=False)
    top = nodes.head(C.TOP_N).merge(df[["why"]], left_on="gid", right_index=True)
    top[TOP_COLS].to_csv(out / "top_nodes.csv", index=False)
    for name, table in extra.items():
        table.to_csv(out / name, index=False)


def validate(out: Path, n_nodes: int, gids: set, seed_gids=None, edges=None) -> list:
    """Проверки из ТЗ (must have 2, 4, 5). Возвращает список (ok, текст)."""
    tables = {}
    checks = []
    for filename, columns in (("nodes_roles.csv", NODE_COLS), ("clusters.csv", CLUSTER_COLS), ("top_nodes.csv", TOP_COLS)):
        try:
            table = pd.read_csv(out / filename)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            checks.append((False, f"{filename}: не удалось прочитать ({error})"))
            continue
        matches = list(table.columns) == columns
        checks.append((matches, f"{filename}: точная схема ТЗ"))
        if matches:
            tables[filename] = table
    if len(tables) != 3:
        return checks
    nr, cl, tp = (tables[name] for name in ("nodes_roles.csv", "clusters.csv", "top_nodes.csv"))
    for name, table, columns in (("nodes_roles.csv", nr, ["gid", "cluster_id"]),
                                  ("clusters.csv", cl, ["cluster_id", "n_nodes", "n_seed"]),
                                  ("top_nodes.csv", tp, ["rank", "gid"])):
        checks.append((all(pd.api.types.is_integer_dtype(table[column]) for column in columns), f"{name}: целочисленные идентификаторы и счётчики"))
    if not all(passed for passed, _ in checks):
        return checks
    for name, table, columns in (("nodes_roles.csv", nr, ["evidence"]), ("clusters.csv", cl, ["hypothesis"]), ("top_nodes.csv", tp, ["why"])):
        for column in columns:
            valid_text = table[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all()
            checks.append((valid_text, f"{name}: {column} содержит непустой текст"))
            table[column] = table[column].astype("string")
    for table, columns in ((nr, ["role_score", "priority_score"]), (cl, ["sum_kzt_internal"]), (tp, ["priority_score"])):
        for column in columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")
    checks += [
        (len(nr) == n_nodes, f"nodes_roles.csv: {len(nr)} строк (нужно {n_nodes})"),
        (set(nr.gid) == gids and nr.gid.is_unique, "nodes_roles.csv: каждый gid ровно один раз"),
        (nr[NODE_COLS].notna().all().all() and (nr.evidence.str.len() > 0).all(), "обязательные колонки заполнены"),
        (nr.role.isin(C.ROLES).all(), "все роли из словаря ТЗ"),
        (nr.role_score.between(0, 1).all() and nr.priority_score.between(0, 1).all(), "role_score и priority_score в [0, 1]"),
        ((nr.evidence.str.len() <= 200).all() and nr.evidence.str.contains(r"\d").all(), "evidence ≤200 символов и содержит числа"),
        (set(nr.cluster_id) == set(cl.cluster_id) and cl.cluster_id.is_unique, "cluster_id каждого узла есть в clusters.csv"),
        (cl[CLUSTER_COLS].notna().all().all() and cl.n_nodes.sum() == n_nodes, "clusters.csv заполнен, размеры сходятся"),
        (len(tp) >= 20 and tp.priority_score.is_monotonic_decreasing and list(tp["rank"]) == list(range(1, len(tp) + 1)),
         f"top_nodes.csv: {len(tp)} строк, отсортирован по приоритету"),
        (tp.why.notna().all() and (tp.why.str.len() > 0).all(), "у каждой позиции топа есть обоснование"),
        (tp.gid.is_unique and set(tp.gid) <= gids, "top_nodes.csv: уникальные известные gid"),
        (np.isfinite(cl.sum_kzt_internal).all() and (cl.sum_kzt_internal >= 0).all(), "clusters.csv: конечный неотрицательный оборот"),
        ((cl.n_nodes > 0).all() and cl.n_seed.between(0, cl.n_nodes).all(), "clusters.csv: допустимое число узлов и seed"),
    ]
    if nr.gid.is_unique and cl.cluster_id.is_unique:
        membership = nr.set_index("gid").cluster_id
        sizes = nr.groupby("cluster_id").size().reindex(cl.cluster_id, fill_value=0).to_numpy()
        checks.append((np.array_equal(sizes, cl.n_nodes.to_numpy()), "размер каждого кластера совпадает с назначениями узлов"))
        joined = tp.merge(nr, on="gid", suffixes=("_top", "_node"), how="left")
        checks.append(((joined.role_top == joined.role_node).all() and np.allclose(joined.priority_score_top, joined.priority_score_node), "топ: роли и приоритеты совпадают с nodes_roles.csv"))
        expected = nr.sort_values("priority_score", ascending=False, kind="stable").head(len(tp))
        checks.append((expected.gid.tolist() == tp.gid.tolist(), "топ соответствует первым узлам общего ранжирования"))
        valid_members = all(
            bool(str(cluster.top_gids).strip()) and all(
                identifier.isdigit() and int(identifier) in gids
                and int(identifier) in membership.index
                and membership.at[int(identifier)] == cluster.cluster_id
                for identifier in str(cluster.top_gids).split(";")
            ) for cluster in cl.itertuples()
        )
        checks.append((valid_members, "top_gids принадлежат соответствующим кластерам"))
        if seed_gids is not None:
            counts = nr[nr.gid.isin(seed_gids)].groupby("cluster_id").size()
            actual = counts.reindex(cl.cluster_id, fill_value=0).to_numpy()
            checks.append((np.array_equal(actual, cl.n_seed.to_numpy()), "число seed каждого кластера совпадает с входными данными"))
        if edges is not None:
            source_cluster = edges.src.map(membership)
            target_cluster = edges.dst.map(membership)
            internal = edges.loc[source_cluster == target_cluster, "sum_kzt"].groupby(source_cluster).sum()
            actual = internal.reindex(cl.cluster_id, fill_value=0).to_numpy()
            checks.append((np.allclose(actual, cl.sum_kzt_internal, rtol=0, atol=.01), "внутренний оборот каждого кластера совпадает с рёбрами"))
    return checks

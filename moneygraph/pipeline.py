"""Полный пересчёт: parquet → метрики → роли → приоритет → кластеры → выгрузки + экран."""
import time
from pathlib import Path

import networkx as nx

from . import clusters, export, features, load, priority, report, resilience, roles, truncation, validation, viewer


def compute(data_dir: Path, log=print) -> dict:
    t0 = time.time()
    edges, nodes, tx = load.load(data_dir)
    stats = load.sanity_check(edges, nodes, tx)
    log(f"  данные: {stats['n_nodes']} узлов, {stats['n_edges']} рёбер, {stats['n_tx']} транзакций — консистентны")
    G = load.build_graph(edges, nodes)
    stats["weak_components"] = nx.number_weakly_connected_components(G)
    df, cyc = features.compute_all(G, nodes, tx)
    log(f"  метрики: степени, суммы, seed-атрибуция, PageRank, betweenness, время, {len(cyc)} циклов")
    df, trunc = truncation.fit_predict(G, df)
    log(f"  модель обрыва 4-го колена: ROC-AUC {trunc['cv_auc']:.2f}, "
        f"≈{trunc['expected_true_sinks']:.0f} из {trunc['truncated_nodes']} — настоящие стоки")
    df = roles.assign(G, df)
    df = priority.score(df)
    df, stab, modularity = clusters.detect(G, df)
    cl = clusters.summarize(G, df, stab)
    log(f"  роли: {df.role.value_counts().to_dict()}")
    log(f"  кластеров: {len(cl)} (модулярность {modularity:.3f})")
    res = resilience.simulate(G, df)
    req = report.data_requests(df)
    ctx = {"G": G, "df": df, "clusters": cl, "stats": stats, "trunc": trunc, "cycles": cyc, "tx": tx,
           "modularity": modularity, "resilience": res, "requests": req, "nodes": nodes, "edges": edges}
    log(f"  расчёт: {time.time() - t0:.1f} с")
    return ctx


def run(data_dir: Path, out_dir: Path, log=print) -> dict:
    started = time.perf_counter()
    ctx = compute(data_dir, log)
    ctx["input_sha256"] = validation.input_fingerprint(data_dir)
    export.write(out_dir, ctx["df"], ctx["clusters"], {
        "resilience.csv": ctx["resilience"],
        "data_requests.csv": ctx["requests"],
    })
    ctx["checks"] = export.validate(out_dir, ctx["stats"]["n_nodes"], set(ctx["nodes"].gid),
                                    set(ctx["nodes"].loc[ctx["nodes"].is_seed, "gid"]), ctx["edges"])
    if not all(passed for passed, _ in ctx["checks"]):
        raise ValueError("Проверка выгрузок не пройдена: " + "; ".join(message for passed, message in ctx["checks"] if not passed))
    report.write_report(out_dir / "report.md", ctx)
    viewer.build(out_dir, ctx["G"], ctx)
    ctx["elapsed_seconds"] = time.perf_counter() - started
    validation.write(out_dir, ctx, ctx["elapsed_seconds"])
    ctx["checks"].append((ctx["elapsed_seconds"] <= 300, "полный пересчёт не более 300 секунд"))
    return ctx

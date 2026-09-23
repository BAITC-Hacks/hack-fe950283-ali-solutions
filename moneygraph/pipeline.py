"""Полный пересчёт: parquet → метрики → роли → приоритет → кластеры → выгрузки + экран."""
import time
from pathlib import Path

from . import clusters, export, features, load, priority, report, resilience, roles, truncation, viewer


def compute(data_dir: Path, log=print) -> dict:
    t0 = time.time()
    edges, nodes, tx = load.load(data_dir)
    stats = load.sanity_check(edges, nodes, tx)
    log(f"  данные: {stats['n_nodes']} узлов, {stats['n_edges']} рёбер, {stats['n_tx']} транзакций — консистентны")
    G = load.build_graph(edges, nodes)
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
    ctx = compute(data_dir, log)
    export.write(out_dir, ctx["df"], ctx["clusters"], {
        "resilience.csv": ctx["resilience"],
        "data_requests.csv": ctx["requests"],
    })
    report.write_report(out_dir / "report.md", ctx)
    viewer.build(out_dir, ctx["G"], ctx)
    ctx["checks"] = export.validate(out_dir, ctx["stats"]["n_nodes"], set(ctx["nodes"].gid))
    return ctx

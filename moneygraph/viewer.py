"""Экран просмотра: один самодостаточный HTML (Cytoscape.js вшит, работает офлайн).

Раскладка считается здесь, детерминированно: каждый кластер раскладывается
отдельно (spring layout), затем кластеры расставляются без перекрытий —
так на схеме видны и связи, и группы.
"""
import json
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from . import config as C
from .fmt import short

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "viewer" / "template.html"
CYTOSCAPE = ROOT / "viewer" / "vendor" / "cytoscape.min.js"
REVIEW = ROOT / "viewer" / "review.js"

ROLE_COLORS = {  # тёмная тема экрана: пастельные цвета на тёмно-синем фоне
    "coordinator": "#f25f73",
    "consolidator": "#e45dbf",
    "distributor": "#3fa3ec",
    "transit": "#34d3a1",
    "terminal": "#f4c95d",
    "peripheral": "#aebdd3",
}


def layout(G: nx.DiGraph, df: pd.DataFrame) -> dict:
    cid = df.cluster_id.to_dict()
    local, radius = {}, {}
    for k, members in df.groupby("cluster_id").groups.items():
        members = list(members)
        n = len(members)
        if k == 0 or n == 1:
            side = int(np.ceil(np.sqrt(n)))
            p = {m: np.array([i % side, i // side], float) for i, m in enumerate(members)}
        else:
            sub = G.subgraph(members).to_undirected()
            p = nx.spring_layout(sub, seed=C.RANDOM_SEED, iterations=200 if n < 400 else 120, k=1.5 / np.sqrt(n))
        arr = np.array([p[m] for m in members], float)
        arr -= arr.mean(axis=0)
        maxr = np.linalg.norm(arr, axis=1).max() or 1.0
        R = 22 * np.sqrt(n) + 12
        local[k] = {m: arr[i] / maxr * R for i, m in enumerate(members)}
        radius[k] = R

    keys = sorted(k for k in local if k != 0)  # кластер 0 (узлы без переводов) ставим отдельно, в угол
    CG = nx.Graph()
    CG.add_nodes_from(keys)
    for u, v in G.edges():
        a, b = cid[u], cid[v]
        if a != b and a and b:
            CG.add_edge(a, b, w=CG.get_edge_data(a, b, {"w": 0})["w"] + 1)
    init = nx.spring_layout(CG, seed=C.RANDOM_SEED, weight="w", iterations=300)
    scale = sum(radius.values()) / 2.5
    P = np.array([init[k] for k in keys]) * scale
    R = np.array([radius[k] for k in keys])
    for it in range(600):
        diff = P[:, None, :] - P[None, :, :]
        dist = np.linalg.norm(diff, axis=2) + np.eye(len(keys))
        need = R[:, None] + R[None, :] + 30 - dist
        np.fill_diagonal(need, 0)
        push = np.where(need > 0, need / 2, 0)[:, :, None] * diff / dist[:, :, None]
        P += push.sum(axis=1)
        if it < 450:
            P *= 0.985  # притяжение к центру: компактная картинка, перекрытия снимает push
    pos = {}
    for i, k in enumerate(keys):
        for m, xy in local[k].items():
            pos[m] = (P[i] + xy).round(1).tolist()
    if 0 in local:
        lo = (P - R[:, None]).min(axis=0)
        hi = (P + R[:, None]).max(axis=0)
        corner = np.array([lo[0] + radius[0], hi[1] + radius[0] + 80])
        for m, xy in local[0].items():
            pos[m] = (corner + xy).round(1).tolist()
    return pos


def build(out: Path, G: nx.DiGraph, ctx: dict) -> Path:
    df, cl = ctx["df"], ctx["clusters"]
    pos = layout(G, df)
    req = defaultdict(list)
    for r in ctx["requests"].itertuples():
        req[r.gid].append(f"{r.request}: {r.reason}")
    routes = defaultdict(list)  # устойчивые маршруты через узел: [A, C, даты поступления в узел]
    for r in ctx["routes"]:
        routes[r["b"]].append([str(r["a"]), str(r["c"]), r["dates"]])

    nodes = []
    for g, r in df.iterrows():
        nodes.append({
            "id": str(g), "s": short(g), "r": r.role, "rs": round(r.role_score, 3), "r2": r.role_secondary,
            "p": round(r.priority_score, 4), "rk": int(r["rank"]), "c": int(r.cluster_id), "sd": int(r.is_seed),
            "d": int(r.depth), "x": pos[g][0], "y": pos[g][1], "ev": r.evidence,
            "why": r.why, "fl": r["flags"],
            "ind": int(r.in_deg), "outd": int(r.out_deg), "ink": round(r.in_kzt), "outk": round(r.out_kzt),
            "intx": int(r.in_tx), "outtx": int(r.out_tx),
            "pt": None if pd.isna(r.pass_through) else round(r.pass_through, 3),
            "tr": int(r.truncated), "pf": round(r.p_forward, 3), "su": int(r.seed_upstream),
            "sk": round(r.seed_kzt_in), "ss": round(r.seed_share, 3), "kl": int(r.key_links),
            "bt": round(r.betweenness, 5), "fs": round(r.fast_in_share, 3),
            "lag": None if pd.isna(r.lag_median_days) else r.lag_median_days,
            "sync": int(r.sync_payers_max), "syncd": int(r.sync_day), "syncdate": r.sync_date, "rc": int(r.n_return_cycles),
            "rr": int(r.repeated_routes), "sp": int(r.split_days), "an": int(r.anomaly),
            "bu": int(r.burst_tx) if r.burst else 0, "bdate": r.burst_date, "rt": routes.get(g, [])[:5],
            "cp": [round(r[f"c_{k}"], 3) for k in C.PRIORITY_WEIGHTS], "req": req.get(g, []),
        })
    txs = defaultdict(list)
    start = ctx["tx"].date.min().normalize()
    for t in ctx["tx"].itertuples(index=False):
        day = (t.date.normalize() - start).days + 1
        txs[(t.src, t.dst)].append([day, round(float(t.sum_kzt), 2)])
    edges = [{"s": str(u), "t": str(v), "w": round(d["sum_kzt"]), "n": d["n_tx"], "tx": sorted(txs[(u, v)])}
             for u, v, d in G.edges(data=True)]
    data = {
        "meta": {
            "stats": ctx["stats"], "modularity": round(ctx["modularity"], 3), "trunc": ctx["trunc"],
            "roles": C.ROLES, "role_ru": C.ROLE_RU, "colors": ROLE_COLORS,
            "weights": C.PRIORITY_WEIGHTS, "role_weight": C.ROLE_WEIGHT,
            "n_cycles": len(ctx["cycles"]), "n_return_cycles": sum(c["returned"] for c in ctx["cycles"]),
            "n_routes": len(ctx["routes"]), "n_burst": int(df.burst.sum()), "fast_days": C.FAST_DAYS,
            "burst": {"days": C.BURST_WINDOW_DAYS, "min_tx": C.BURST_MIN_TX, "min_share": C.BURST_MIN_SHARE},
            "rules": rules_table(),
            "checks": [{"passed": bool(passed), "description": text} for passed, text in ctx["checks"]],
            "dataset_id": "-".join(ctx["input_sha256"].values()),
        },
        "nodes": nodes, "edges": edges,
        "clusters": json.loads(cl.to_json(orient="records", force_ascii=False)),
        "resilience": json.loads(ctx["resilience"].to_json(orient="records", force_ascii=False)),
        "exports": {name: (out / name).read_text(encoding="utf-8") for name in ("nodes_roles.csv", "clusters.csv", "top_nodes.csv", "data_requests.csv", "report.md")},
    }
    for c in data["clusters"]:
        c["top_gids"] = c["top_gids"].split(";")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("/*__CYTOSCAPE__*/", CYTOSCAPE.read_text(encoding="utf-8"))
    html = html.replace("/*__REVIEW__*/", REVIEW.read_text(encoding="utf-8"))
    html = html.replace("/*__DATA__*/", f"const DATA = {payload};")
    path = out / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def rules_table() -> list:
    """Правила ролей для вкладки «Методика» — из тех же констант, что использует пайплайн."""
    return [
        ["coordinator", f"≥{C.HUB_MIN_PAYERS} плательщиков И ≥{C.HUB_MIN_RECIPIENTS} получателей (хаб) И "
                        f"≥{C.COORD_MIN_KEY_LINKS} прямых связей с ключевыми узлами"],
        ["consolidator", f"≥{C.CONS_MIN_PAYERS} плательщиков (или ≥{C.CONS_ALT_PAYERS}, из них ≥{C.CONS_ALT_SEED_PAYERS} seed) "
                         f"и плательщиков ≥ получателей; либо хаб, у которого получателей меньше "
                         f"{C.HUB_DISTR_RATIO}× плательщиков (сбор доминирует)"],
        ["distributor", f"≥{C.DISTR_MIN_RECIPIENTS} получателей и получателей ≥{C.DISTR_FANOUT_RATIO}× плательщиков; "
                        f"либо хаб, у которого получателей ≥{C.HUB_DISTR_RATIO}× плательщиков (раздача доминирует)"],
        ["transit", f"не seed, исходящие наблюдаемы; отдал {C.TRANSIT_PT[0]:.0%}–{C.TRANSIT_PT[1]:.0%} полученного, "
                    f"или ≥{C.TRANSIT_FAST_SHARE:.0%} ушло ≤{C.FAST_DAYS} дн. при пропуске {C.TRANSIT_FAST_PT[0]}–{C.TRANSIT_FAST_PT[1]}; "
                    f"через узел ≥{C.TRANSIT_MIN_KZT // 1000} тыс ₸. Seed: переправил ≥{C.SEED_TRANSIT_MIN_KZT // 1000} тыс ₸ "
                    f"≤{C.SEED_TRANSIT_MAX_RCPT} получателям"],
        ["terminal", f"исходящие наблюдаемы и ушло ≤{C.TERM_MAX_PT:.0%} полученного; получил ≥{C.TERM_MIN_KZT // 1000} тыс ₸, "
                     f"или от ≥{C.TERM_MIN_PAYERS} плательщиков, или ≥{C.TERM_MIN_TX} поступлений. 4-е колено — только при "
                     f"P(сток) ≥ {C.TRUNC_TERMINAL_P} по модели обрыва"],
        ["peripheral", "ни одно правило не выполнено; уверенность тем выше, чем дальше узел от порогов"],
    ]

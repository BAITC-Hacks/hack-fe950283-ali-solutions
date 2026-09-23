"""Экран просмотра: один самодостаточный HTML (Cytoscape.js вшит, работает офлайн).

Раскладка считается здесь, детерминированно: каждый кластер раскладывается
отдельно (spring layout), затем кластеры расставляются без перекрытий —
так на схеме видны и связи, и группы.
"""
import base64
import json
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from . import config as C
from .fmt import short
from .contracts import clean

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "viewer" / "template.html"
CYTOSCAPE = ROOT / "viewer" / "vendor" / "cytoscape.min.js"

ROLE_COLORS = {
    "coordinator": "#f47bb4",
    "consolidator": "#ffbb7b",
    "distributor": "#b6a0ff",
    "transit": "#35d1ef",
    "terminal": "#72d6ac",
    "peripheral": "#72abc7",
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
    if not keys:
        return {g: xy.round(1).tolist() for members in local.values() for g,xy in members.items()}
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

    nodes = []
    for g, r in df.iterrows():
        nodes.append({
            "id": str(g), "s": short(g), "r": r.role, "rs": r.role_score, "r2": r.role_secondary,
            "p": round(r.priority_score, 4), "rk": int(r["rank"]), "c": int(r.cluster_id), "sd": int(r.is_seed),
            "d": int(r.depth), "x": pos[g][0], "y": pos[g][1], "ev": r.evidence,
            "why": r.why, "fl": r["flags"],
            "ind": int(r.in_deg), "outd": int(r.out_deg), "ink": r.in_kzt, "outk": r.out_kzt,
            "intx": int(r.in_tx), "outtx": int(r.out_tx),
            "pt": None if pd.isna(r.pass_through) else r.pass_through,
            "tr": int(r.truncated), "pf": round(r.p_forward, 3), "su": int(r.seed_upstream),
            "sk": r.seed_kzt_in, "ss": round(r.seed_share, 3), "kl": int(r.key_links),
            "bt": round(r.betweenness, 5), "fs": r.fast_in_share,
            "lag": None if pd.isna(r.lag_median_days) else r.lag_median_days,
            "sync": int(r.sync_payers_max), "syncd": r.sync_day, "rc": int(r.n_return_cycles),
            "rr": int(r.repeated_routes), "sp": int(r.split_days), "an": int(r.anomaly),
            "cycles": int(r.n_cycles), "fast_routes": int(r.fast_routes),
            "an_feature": r.anomaly_feature, "an_pct": r.anomaly_pct,
            "burst": {k: r[k] for k in ("burst_in_max_tx", "burst_in_day", "burst_in_mean_daily_tx",
                       "burst_in_ratio", "burst_observation_days", "burst_in_flag")},
            "cp": [float(r[f"c_{k}"]) for k in C.PRIORITY_WEIGHTS],
            "raw": r.priority_raw, "sf": r.priority_seed_factor, "norm": r.priority_normalizer,
            "fast_status": r.fast_status, "matched": r.fast_matched_kzt, "eligible": r.fast_eligible_kzt,
            "coverage": r.fast_coverage, "same_day": r.same_day_kzt, "req": req.get(g, []),
        })
    txs = defaultdict(list)
    for t in ctx["tx"].itertuples(index=False):
        txs[(t.src, t.dst)].append([t.date.date().isoformat(), float(t.sum_kzt)])
    edges = [{"s": str(u), "t": str(v), "w": float(d["sum_kzt"]), "n": d["n_tx"], "tx": sorted(txs[(u, v)])}
             for u, v, d in G.edges(data=True)]
    data = {
        "meta": {
            "manifest": ctx["manifest"], "stats": ctx["stats"], "modularity": round(ctx["modularity"], 3), "trunc": ctx["trunc"],
            "roles": C.ROLES, "role_ru": C.ROLE_RU, "colors": ROLE_COLORS,
            "weights": C.PRIORITY_WEIGHTS, "role_weight": C.ROLE_WEIGHT,
            "n_cycles": len(ctx["cycles"]), "n_return_cycles": sum(c["returned"] for c in ctx["cycles"]),
            "rules": rules_table(),
        },
        "nodes": nodes, "edges": edges,
        "csv": {name: (out / name).read_text(encoding="utf-8") for name in ("nodes_roles.csv", "clusters.csv", "top_nodes.csv")},
        "clusters": json.loads(cl.to_json(orient="records", force_ascii=False)),
        "resilience": json.loads(ctx["resilience"].to_json(orient="records", force_ascii=False)),
    }
    for c in data["clusters"]:
        c["top_gids"] = c["top_gids"].split(";")
    payload = json.dumps(clean(data), ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("/*__CYTOSCAPE__*/", CYTOSCAPE.read_text(encoding="utf-8"))
    html = html.replace("/*__DATA__*/", f"const DATA = {payload};")
    html = html.replace("/*__ADDITIONS__*/", (ROOT / "viewer" / "additions.js").read_text(encoding="utf-8"))
    html = html.replace("/*__THEME__*/", (ROOT / "viewer" / "theme.css").read_text(encoding="utf-8"))
    html = html.replace("/*__WORKSPACE__*/", (ROOT / "viewer" / "workspace.js").read_text(encoding="utf-8"))
    html = html.replace("/*__ICONS__*/", (ROOT / "viewer" / "icons.js").read_text(encoding="utf-8"))
    font = ROOT / "viewer" / "vendor" / "inter" / "InterVariable.woff2"
    html = html.replace("__INTER_WOFF2__", base64.b64encode(font.read_bytes()).decode("ascii"))
    path = out / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def rules_table() -> list:
    """Правила ролей для вкладки «Методика» — из тех же констант, что использует пайплайн."""
    return [
        ["coordinator", f"≥{C.HUB_MIN_PAYERS} плательщиков И ≥{C.HUB_MIN_RECIPIENTS} получателей (хаб) И "
                        f"≥{C.COORD_MIN_KEY_LINKS} прямых связей с ключевыми узлами"],
        ["consolidator", f"≥{C.CONS_MIN_PAYERS} плательщиков (или ≥{C.CONS_ALT_PAYERS}, из них ≥{C.CONS_ALT_SEED_PAYERS} seed) "
                         f"и плательщиков ≥ получателей; либо хаб, где сбор доминирует"],
        ["distributor", f"≥{C.DISTR_MIN_RECIPIENTS} получателей и получателей ≥{C.DISTR_FANOUT_RATIO}× плательщиков; "
                        f"либо хаб, где раздача доминирует (получателей ≥2× плательщиков)"],
        ["transit", f"не seed, исходящие наблюдаемы; out/in {C.TRANSIT_PT}; либо совместимость ≥{C.TRANSIT_FAST_SHARE:.0%} за 1–2 дня при out/in {C.TRANSIT_FAST_PT}. Оборот ≥{C.TRANSIT_MIN_KZT} ₸"],
        ["terminal", f"не seed, depth<4, наблюдаемый выход ≤{C.TERM_MAX_PT:.0%} входа; существенный вход по правилу. Кандидат в пределах выгрузки; модель роли не назначает"],
        ["peripheral", "остальные; у изолята соответствие роли=0; неполнота наблюдения показана отдельно"],

    ]

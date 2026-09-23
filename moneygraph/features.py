"""Метрики узлов: структура, деньги, связь с фигурантами, время, циклы."""
from collections import defaultdict, deque

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from . import config as C


# ---------------------------------------------------------------- структура и суммы

def basic(G: nx.DiGraph, nodes: pd.DataFrame) -> pd.DataFrame:
    df = nodes[["gid", "depth", "is_seed"]].copy().set_index("gid")
    df["in_deg"] = pd.Series(dict(G.in_degree()))
    df["out_deg"] = pd.Series(dict(G.out_degree()))
    df["in_kzt"] = pd.Series(dict(G.in_degree(weight="sum_kzt")))
    df["out_kzt"] = pd.Series(dict(G.out_degree(weight="sum_kzt")))
    df["in_tx"] = pd.Series(dict(G.in_degree(weight="n_tx"))).astype(int)
    df["out_tx"] = pd.Series(dict(G.out_degree(weight="n_tx"))).astype(int)
    # исходящие наблюдаемы только у узлов 0–3 колена: обход дальше 4-го не шёл
    df["out_observed"] = df.depth < C.MAX_DEPTH
    df["truncated"] = (df.depth == C.MAX_DEPTH) & (df.out_deg == 0)
    df["pass_through"] = np.where(df.in_kzt > 0, df.out_kzt / df.in_kzt.where(df.in_kzt > 0), np.nan)
    df["n_edges"] = df.in_deg + df.out_deg

    seeds = set(df.index[df.is_seed])
    df["seed_payers"] = [sum(p in seeds for p in G.predecessors(g)) for g in df.index]
    df["seed_recipients"] = [sum(s in seeds for s in G.successors(g)) for g in df.index]

    # доля крупнейшего контрагента: 1.0 — все деньги одному/от одного
    top_out = {g: max((d["sum_kzt"] for _, _, d in G.out_edges(g, data=True)), default=0.0) for g in df.index}
    top_in = {g: max((d["sum_kzt"] for _, _, d in G.in_edges(g, data=True)), default=0.0) for g in df.index}
    df["top_out_share"] = (pd.Series(top_out) / df.out_kzt.replace(0, np.nan)).fillna(0.0)
    df["top_in_share"] = (pd.Series(top_in) / df.in_kzt.replace(0, np.nan)).fillna(0.0)
    return df


def seed_linkage(G: nx.DiGraph, df: pd.DataFrame) -> pd.DataFrame:
    """Сколько разных seed доходят до узла за ≤4 перевода и сколько «их» денег пришло.

    Атрибуция пропорциональная: деньги на счёте обезличены, поэтому каждый исходящий
    перевод узла несёт ту же долю денег фигурантов, что и весь доступный узлу объём.
    Доступный объём = max(вход, выход): если отдал больше, чем получил в выборке,
    разница пришла извне и «разбавляет» долю.
    """
    seeds = list(df.index[df.is_seed])
    reach = defaultdict(set)
    for s in seeds:
        for v, dist in nx.single_source_shortest_path_length(G, s, cutoff=C.MAX_DEPTH).items():
            if v != s:
                reach[v].add(s)
    df["seed_upstream"] = [len(reach.get(g, ())) for g in df.index]

    order = list(df.index)
    pos = {g: i for i, g in enumerate(order)}
    e = [(pos[u], pos[v], d["sum_kzt"]) for u, v, d in G.edges(data=True)]
    rows, cols, vals = zip(*e)
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(len(order), len(order)))
    is_seed = df.is_seed.to_numpy()
    avail = np.maximum(df.in_kzt.to_numpy(), df.out_kzt.to_numpy())
    c = is_seed.astype(float)
    for _ in range(500):
        T = W.T @ c
        c_new = np.where(is_seed, 1.0, np.minimum(1.0, T / np.where(avail > 0, avail, 1.0)))
        if np.abs(c_new - c).max() < 1e-10:
            break
        c = c_new
    df["seed_kzt_in"] = W.T @ c
    df["seed_share"] = c
    return df


def centrality(G: nx.DiGraph, df: pd.DataFrame) -> pd.DataFrame:
    df["pagerank"] = pd.Series(nx.pagerank(G, weight="sum_kzt"))
    # направленное посредничество: доля кратчайших путей по направлению денег через узел
    df["betweenness"] = pd.Series(nx.betweenness_centrality(G, normalized=True))
    df["wcc_size"] = 0
    for comp in nx.weakly_connected_components(G):
        df.loc[list(comp), "wcc_size"] = len(comp)
    return df


# ---------------------------------------------------------------- время

def temporal(tx: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    tx = tx.copy()
    tx["day"] = tx.date.dt.day.astype(int)
    tx = tx.sort_values(["day", "src", "dst"])
    # кортежи (день, сумма, контрагент): gid держим как int — в float64 18 знаков не влезают
    ins, outs = defaultdict(list), defaultdict(list)
    for r in tx.itertuples(index=False):
        ins[r.dst].append((int(r.day), float(r.sum_kzt), int(r.src)))
        outs[r.src].append((int(r.day), float(r.sum_kzt), int(r.dst)))

    rows = {}
    for g in df.index:
        i_l, o_l = ins.get(g, []), outs.get(g, [])
        r = {
            "first_in_day": i_l[0][0] if i_l else 0,
            "last_in_day": i_l[-1][0] if i_l else 0,
            "max_in_tx": max((a for _, a, _ in i_l), default=0.0),
            "in_days": len({d for d, _, _ in i_l}),
            "out_days": len({d for d, _, _ in o_l}),
            "fast_in_share": 0.0,
            "lag_median_days": np.nan,
            "sync_payers_max": 0,
            "sync_day": 0,
        }
        if i_l and o_l:
            r["fast_in_share"], r["lag_median_days"] = _fast_forward(i_l, o_l)
        if i_l:
            by_day = defaultdict(set)
            for day, _, src in i_l:
                by_day[day].add(src)
            day, payers = max(by_day.items(), key=lambda kv: (len(kv[1]), -kv[0]))
            r["sync_payers_max"], r["sync_day"] = len(payers), day
        rows[g] = r
    t = pd.DataFrame.from_dict(rows, orient="index")

    # дробление: несколько переводов одному получателю в один день
    split = tx.groupby(["src", "dst", "day"]).size()
    split = split[split >= 2].groupby(level=0).size()
    t["split_days"] = split.reindex(t.index).fillna(0).astype(int)

    # устойчивые маршруты A→B→C: B переслал дальше в пределах FAST_DAYS; повтор — в ≥2 разные даты
    fast_routes, repeated = {}, {}
    for g in df.index:
        if g not in ins or g not in outs:
            continue
        seen = defaultdict(set)
        for d_in, _, a in ins[g]:
            for d_out, _, c in outs[g]:
                if 0 <= d_out - d_in <= C.FAST_DAYS and a != c:
                    seen[(a, c)].add(d_in)
        if seen:
            fast_routes[g] = len(seen)
            repeated[g] = sum(len(v) >= 2 for v in seen.values())
    t["fast_routes"] = pd.Series(fast_routes).reindex(t.index).fillna(0).astype(int)
    t["repeated_routes"] = pd.Series(repeated).reindex(t.index).fillna(0).astype(int)
    return df.join(t)


def _fast_forward(i_l, o_l):
    """FIFO: какая доля полученного ушла дальше не позже FAST_DAYS после поступления."""
    lots = deque()  # [day, остаток]
    total_in = sum(a for _, a, _ in i_l)
    matched, k = 0.0, 0
    for d_out, amt, _ in o_l:
        while k < len(i_l) and i_l[k][0] <= d_out:
            lots.append([i_l[k][0], i_l[k][1]])
            k += 1
        while lots and lots[0][0] < d_out - C.FAST_DAYS:
            lots.popleft()
        need = amt
        while need > 0 and lots:
            take = min(need, lots[0][1])
            matched += take
            need -= take
            lots[0][1] -= take
            if lots[0][1] <= 1e-9:
                lots.popleft()
    out_days = np.array(sorted(d for d, _, _ in o_l))
    lags = []
    for d_in, _, _ in i_l:
        j = np.searchsorted(out_days, d_in)
        if j < len(out_days):
            lags.append(out_days[j] - d_in)
    return (matched / total_in if total_in else 0.0), (float(np.median(lags)) if lags else np.nan)


# ---------------------------------------------------------------- циклы (возвратные потоки)

def cycles(G: nx.DiGraph, tx: pd.DataFrame, df: pd.DataFrame, max_len: int = 6):
    """Простые циклы длиной ≤6. «Возвратный» — если по датам деньги могли пройти круг."""
    days = defaultdict(list)
    for r in tx.itertuples(index=False):
        days[(r.src, r.dst)].append(r.date.day)
    for k in days:
        days[k].sort()

    n_cyc, n_ret = defaultdict(int), defaultdict(int)
    listing = []
    for cyc in nx.simple_cycles(G, length_bound=max_len):
        pairs = list(zip(cyc, cyc[1:] + cyc[:1]))
        bottleneck = min(G[u][v]["sum_kzt"] for u, v in pairs)
        returned = any(_time_consistent(pairs[i:] + pairs[:i], days) for i in range(len(pairs)))
        for g in cyc:
            n_cyc[g] += 1
            n_ret[g] += returned
        listing.append({"nodes": cyc, "len": len(cyc), "bottleneck_kzt": bottleneck, "returned": returned})
    df["n_cycles"] = pd.Series(n_cyc).reindex(df.index).fillna(0).astype(int)
    df["n_return_cycles"] = pd.Series(n_ret).reindex(df.index).fillna(0).astype(int)
    return df, listing


def _time_consistent(pairs, days) -> bool:
    t = -1
    for p in pairs:
        nxt = next((d for d in days[p] if d >= t), None)
        if nxt is None:
            return False
        t = nxt
    return True


# ---------------------------------------------------------------- аномалии относительно колена

ANOMALY_COLS = {"in_deg": "плательщиков", "out_deg": "получателей", "in_kzt": "вход ₸", "out_kzt": "выход ₸"}


def anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Процентиль признака внутри своего колена: сравниваем узел с «соседями по глубине»."""
    pct = pd.DataFrame(index=df.index)
    for col in ANOMALY_COLS:
        pct[col] = df.groupby("depth")[col].rank(pct=True, method="min")
        pct.loc[df[col] <= 0, col] = 0.0
    df["anomaly_pct"] = pct.max(axis=1)
    df["anomaly_feature"] = pct.idxmax(axis=1)
    material = (df.in_deg >= 3) | (df.out_deg >= 5) | (df[["in_kzt", "out_kzt"]].max(axis=1) >= 500_000)
    df["anomaly"] = (df.anomaly_pct >= 0.99) & material
    return df


def compute_all(G, nodes, tx):
    df = basic(G, nodes)
    df = seed_linkage(G, df)
    df = centrality(G, df)
    df = temporal(tx, df)
    df, cyc = cycles(G, tx, df)
    df = anomalies(df)
    return df, cyc

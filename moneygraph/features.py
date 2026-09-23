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
    df["in_deg"] = [len(set(G.predecessors(g)) - {g}) for g in df.index]
    df["out_deg"] = [len(set(G.successors(g)) - {g}) for g in df.index]
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
    разница моделируется как неатрибутированные средства (внешний вход или начальный остаток).
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
    if not e:
        df["seed_kzt_in"] = 0.0
        df["seed_share"] = 0.0
        return df
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

def temporal(tx: pd.DataFrame, df: pd.DataFrame, period_end=None) -> pd.DataFrame:
    tx = tx[tx.src != tx.dst].copy()  # self-transfers are not independent counterparties
    tx["day"] = tx.date.dt.normalize()
    tx = tx.sort_values(["day", "src", "dst", "sum_kzt"])
    end = pd.Timestamp(period_end).normalize() if period_end is not None and pd.notna(period_end) else tx.date.max()
    ins, outs = defaultdict(list), defaultdict(list)
    for r in tx.itertuples(index=False):
        ins[r.dst].append((r.day, float(r.sum_kzt), int(r.src)))
        outs[r.src].append((r.day, float(r.sum_kzt), int(r.dst)))
    rows = {}
    for g in df.index:
        incoming, outgoing = ins.get(g, []), outs.get(g, [])
        metrics = _fast_forward(incoming, outgoing, end, bool(df.at[g, "out_observed"]))
        by_day = defaultdict(set)
        for day, _, src in incoming:
            by_day[day].add(src)
        day, payers = max(by_day.items(), key=lambda kv: (len(kv[1]), -kv[0].value)) if by_day else (None, set())
        rows[g] = {
            "first_in_day": incoming[0][0].toordinal() if incoming else 0,
            "last_in_day": incoming[-1][0].toordinal() if incoming else 0,
            "max_in_tx": max((a for _, a, _ in incoming), default=0.0),
            "in_days": len(by_day), "out_days": len({d for d, _, _ in outgoing}),
            "sync_payers_max": len(payers), "sync_day": day.date().isoformat() if day is not None else "",
            **metrics,
        }
    t = pd.DataFrame.from_dict(rows, orient="index")
    split = tx.groupby(["src", "dst", "day"]).size()
    split = split[split >= 2].groupby(level=0).size()
    t["split_days"] = split.reindex(t.index).fillna(0).astype(int)
    routes, repeated = {}, {}
    for g in df.index:
        seen = defaultdict(set)
        for d_in, _, a in ins.get(g, []):
            for d_out, _, c in outs.get(g, []):
                if 1 <= (d_out-d_in).days <= C.FAST_DAYS and a != c:
                    seen[(a, c)].add(d_in)
        routes[g] = len(seen)
        repeated[g] = sum(len(v) >= 2 for v in seen.values())
    t["fast_routes"] = pd.Series(routes)
    t["repeated_routes"] = pd.Series(repeated)
    return df.join(t)


def _fast_forward(i_l, o_l, period_end=None, observed=True):
    """FIFO calendar lag 1..2; only inputs with a complete window enter the ratio.
    Floating residual tolerance: 1e-9 KZT; amounts are never rounded here.
    Same-day activity is descriptive and consumes no FIFO capacity.
    """
    incoming, outgoing = sorted(i_l), sorted(o_l)
    end = pd.Timestamp(period_end) if period_end is not None else max(
        [d for d, _, _ in incoming + outgoing], default=pd.NaT)
    total = sum(a for _, a, _ in incoming)
    eligible = sum(a for d, a, _ in incoming if d + pd.Timedelta(days=C.FAST_DAYS) <= end)
    in_daily, out_daily = defaultdict(float), defaultdict(float)
    for d,a,_ in incoming: in_daily[d] += a
    for d,a,_ in outgoing: out_daily[d] += a
    same_day = sum(min(a, out_daily[d]) for d,a in in_daily.items())
    result = {"fast_in_share": np.nan, "lag_median_days": np.nan,
              "fast_matched_kzt": np.nan, "fast_eligible_kzt": eligible,
              "fast_coverage": eligible / total if total else np.nan,
              "fast_status": "no_full_window" if not eligible else "available",
              "same_day_kzt": same_day, "fast_eligible_events": sum(
                  d + pd.Timedelta(days=C.FAST_DAYS) <= end for d,_,_ in incoming)}
    if not observed:
        result["fast_status"] = "outgoing_unobserved"
        return result
    if not eligible:
        return result
    lots, k, matched, lags = deque(), 0, 0.0, []
    for d_out, amount, _ in outgoing:
        while k < len(incoming) and incoming[k][0] < d_out:
            lots.append([incoming[k][0], incoming[k][1]])
            k += 1
        while lots and (d_out-lots[0][0]).days > C.FAST_DAYS:
            lots.popleft()
        while amount > 1e-9 and lots:
            take = min(amount, lots[0][1])
            if lots[0][0] + pd.Timedelta(days=C.FAST_DAYS) <= end:
                matched += take
                lags.append((d_out-lots[0][0]).days)
            amount -= take
            lots[0][1] -= take
            if lots[0][1] <= 1e-9:
                lots.popleft()
    result.update(fast_in_share=min(1.0, matched/eligible), fast_matched_kzt=matched,
                  lag_median_days=float(np.median(lags)) if lags else np.nan)
    return result


# ---------------------------------------------------------------- структурные циклы и совместимость дат

def cycles(G: nx.DiGraph, tx: pd.DataFrame, df: pd.DataFrame, max_len: int = 6):
    """Простые циклы длиной ≤6 и строгий порядок дат; без атрибуции средств."""
    days = defaultdict(list)
    for r in tx.itertuples(index=False):
        days[(r.src, r.dst)].append(r.date)
    for k in days:
        days[k].sort()

    n_cyc, n_ret = defaultdict(int), defaultdict(int)
    listing = []
    G.graph["cycles_truncated"] = False
    for cyc in nx.simple_cycles(G, length_bound=max_len):
        if len(listing) >= C.MAX_CYCLES:
            G.graph["cycles_truncated"] = True
            break
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
    t = pd.Timestamp.min
    for p in pairs:
        nxt = next((d for d in days[p] if d > t), None)
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


def compute_all(G, nodes, tx, period_end=None):
    df = basic(G, nodes)
    df = seed_linkage(G, df)
    df = centrality(G, df)
    df = temporal(tx, df, period_end)
    df, cyc = cycles(G, tx, df)
    df = anomalies(df)
    return df, cyc

"""Кластеры: Louvain на НЕориентированной проекции (оговорка: направление здесь
сознательно игнорируется — кластер отвечает на вопрос «кто с кем связан деньгами»,
а направление потоков учитывают роли). Вес ребра = log(1 + сумма/5000): крупные
переводы важнее мелких, но не подавляют структуру.
"""
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd

from . import config as C
from .fmt import kzt, short


def _undirected(G: nx.DiGraph) -> nx.Graph:
    UG = nx.Graph()
    UG.add_nodes_from(G)
    for u, v, d in G.edges(data=True):
        if u == v:
            continue
        w = np.log1p(d["sum_kzt"] / C.MIN_TX_KZT)
        if UG.has_edge(u, v):
            UG[u][v]["w"] += w
        else:
            UG.add_edge(u, v, w=w)
    return UG


def detect(G: nx.DiGraph, df: pd.DataFrame):
    UG = _undirected(G)
    if not UG.number_of_edges():
        df["cluster_id"] = 0
        return df, {0: 1.0}, 0.0
    comms = nx.community.louvain_communities(UG, weight="w", seed=C.RANDOM_SEED)
    modularity = nx.community.modularity(UG, comms, weight="w")

    # устойчивость: насколько кластер воспроизводится при других random seed
    runs = [nx.community.louvain_communities(UG, weight="w", seed=s) for s in range(1, C.LOUVAIN_RUNS + 1)]
    runs_map = [{g: i for i, c in enumerate(r) for g in c} for r in runs]

    isolated = {g for g in UG if UG.degree(g) == 0}
    real = [c for c in comms if not c <= isolated]
    # нумерация: 1 — самый приоритетный кластер; 0 — узлы без единого перевода
    real.sort(key=lambda c: (-df.loc[sorted(c), "priority_score"].sum(), min(c)))
    cid = {g: 0 for g in isolated}
    stab = {}
    for i, c in enumerate(real, start=1):
        for g in c:
            cid[g] = i
        stab[i] = float(np.mean([_best_jaccard(c, rm, r) for rm, r in zip(runs_map, runs)]))
    df["cluster_id"] = pd.Series(cid).reindex(df.index).astype(int)
    return df, stab, modularity


def _best_jaccard(c, run_map, run):
    counts = Counter(run_map[g] for g in c)
    best = 0.0
    for k, inter in counts.items():
        best = max(best, inter / len(c | run[k]))
    return best


def summarize(G: nx.DiGraph, df: pd.DataFrame, stab: dict) -> pd.DataFrame:
    rows = []
    cid = df.cluster_id.to_dict()
    for k, sub in df.groupby("cluster_id"):
        members = set(sub.index)
        internal = sum(d["sum_kzt"] for u, v, d in G.edges(data=True) if cid[u] == k and cid[v] == k)
        inflow = sum(d["sum_kzt"] for u, v, d in G.in_edges(members, data=True) if cid[u] != k)
        outflow = sum(d["sum_kzt"] for u, v, d in G.out_edges(members, data=True) if cid[v] != k)
        top = sub.sort_values("rank")
        roles = sub.role.value_counts()
        rows.append({
            "cluster_id": int(k),
            "n_nodes": len(sub),
            "n_seed": int(sub.is_seed.sum()),
            "sum_kzt_internal": round(float(internal), 2),
            "top_gids": ";".join(str(g) for g in top.index[:5]),
            "hypothesis": _hypothesis(k, sub, top, internal, G, df.role),
            "archetype": _archetype(k, sub),
            "sum_kzt_in_from_other": round(float(inflow), 2),
            "sum_kzt_out_to_other": round(float(outflow), 2),
            "roles": ";".join(f"{r}:{c}" for r, c in roles.items()),
            "n_return_cycles_nodes": int((sub.n_return_cycles > 0).sum()),
            "truncated_share": round(float(sub.truncated.mean()), 3),
            "stability": round(stab.get(k, 1.0), 3),
            "priority_sum": round(float(sub.priority_score.sum()), 3),
        })
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def _archetype(k, sub) -> str:
    if k == 0:
        return "нет данных"
    roles = sub.role.value_counts()
    if roles.get("coordinator", 0):
        return "ядро"
    cons = sub[sub.role == "consolidator"]
    if len(cons) and (cons.seed_payers >= 2).any():
        return "контур сбора"
    if roles.get("distributor", 0):
        return "веерные выплаты"
    if len(cons):
        return "контур сбора"
    if roles.get("transit", 0) / len(sub) >= 0.2:
        return "транзитная цепочка"
    if sub.truncated.mean() >= 0.5:
        return "граница выборки"
    return "периферия"


def _hypothesis(k, sub, top, internal, G, role) -> str:
    n, ns = len(sub), int(sub.is_seed.sum())
    if k == 0:
        return (f"{n} узлов без внешних связей в выгрузке: "
                f"отсутствие данных. Нужен запрос полных выписок.")
    arche = _archetype(k, sub)
    lead = top.iloc[0]
    lead_s = f"{short(top.index[0])} ({lead.role}, {lead.in_deg}→{lead.out_deg})"
    base = f"{n} узлов, {ns} seed, внутр. оборот {kzt(internal)}. "
    if arche == "ядро":
        coords = sub[sub.role == "coordinator"]
        cyc = int((sub.n_return_cycles > 0).sum())
        txt = (f"Признаки ядра сети: {len(coords)} координатор(а) ({', '.join(short(g) for g in coords.index[:3])}) "
               f"одновременно собирают и раздают средства, связаны с другими ключевыми узлами"
               + (f"; {cyc} узлов в циклах с возрастающими датами (возврат тех же средств не доказан)" if cyc else "")
               + ". Гипотеза: узлы связывают несколько направлений переводов; нужна проверка.")
    elif arche == "контур сбора":
        cs = sub[sub.role == "consolidator"].sort_values(["seed_payers", "in_deg"], ascending=False)
        c, cg = cs.iloc[0], cs.index[0]
        txt = (f"Признаки консолидации: {short(cg)} получает от {c.in_deg} плательщиков"
               f" (seed: {c.seed_payers}), дальше уходит {min(c.out_kzt / c.in_kzt, 9.99):.0%}. "
               f"Гипотеза: точка сходящихся наблюдаемых потоков.")
    elif arche == "веерные выплаты":
        dd = sub[sub.role == "distributor"].sort_values("out_deg", ascending=False)
        g, r = dd.index[0], dd.iloc[0]
        rec = [v for v in G.successors(g)]
        end_share = np.mean([role[v] in ("terminal", "peripheral") for v in rec]) if rec else 0
        txt = (f"Признаки веерного распределения: {short(g)} рассылает {kzt(r.out_kzt)} на {r.out_deg} получателей, "
               f"{end_share:.0%} из них — конечные/периферия. Гипотеза: веер переводов; назначение операций неизвестно.")
    elif arche == "транзитная цепочка":
        txt = (f"Признаки транзита: {int((sub.role == 'transit').sum())} узлов пропускают деньги дальше "
               f"почти без остатка. Гипотеза: структурная цепочка с совместимыми потоками.")
    elif arche == "граница выборки":
        txt = (f"{sub.truncated.mean():.0%} узлов обрезаны 4-м коленом — структура не видна. "
               f"Гипотеза не формируется без исходящих 5-го колена.")
    else:
        txt = "Пороги выбранных структурных правил не достигнуты; назначение переводов неизвестно."
    return f"{base}{txt} Ключевой узел: {lead_s}."

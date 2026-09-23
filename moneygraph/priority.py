"""Приоритет для аналитика: взвешенная сумма пяти интерпретируемых компонент 0..1."""
import numpy as np
import pandas as pd

from . import config as C
from .fmt import kzt, clip_text
from .features import ANOMALY_COLS
from .roles import sat

COMPONENT_RU = {
    "role": "роль",
    "seed": "связь с фигурантами",
    "volume": "оборот",
    "network": "положение в сети",
    "patterns": "паттерны",
}

ACTION = {
    "coordinator": "источник средств, начальный остаток и полную выписку, связи с другими хабами",
    "consolidator": "полную выписку и возможные снятие наличных, межбанк, покупка активов",
    "distributor": "источник средств, начальный остаток и получателей веера",
    "transit": "цепочку до и после узла: возможен транзитный счёт",
    "terminal": "полную выписку: наблюдаемый выход не превышает 10% входа",
    "peripheral": "достаточность наблюдений перед дальнейшими выводами",
}


def score(df: pd.DataFrame) -> pd.DataFrame:
    d = df
    comp = pd.DataFrame(index=d.index)
    comp["role"] = d.role.map(C.ROLE_WEIGHT) * d.role_score
    comp["seed"] = 0.5 * sat(d.seed_upstream.clip(lower=1), 1, 12) + 0.5 * sat(d.seed_kzt_in.clip(lower=1), 1e4, 2e6)
    comp["volume"] = sat(d[["in_kzt", "out_kzt"]].max(axis=1).clip(lower=1), 5e4, 1e7)
    comp["network"] = 0.5 * d.pagerank.rank(pct=True, method="min") + \
        0.5 * d.betweenness.where(d.betweenness > 0, 0).rank(pct=True, method="min").where(d.betweenness > 0, 0)
    pat = pd.DataFrame({
        "fast": (d.fast_in_share >= C.TRANSIT_FAST_SHARE) & (d.in_kzt >= C.TRANSIT_MIN_KZT),
        "sync": d.sync_payers_max >= 3,
        "cycles": d.n_return_cycles > 0,
        "anomaly": d.anomaly,
        "routes": d.repeated_routes > 0,
        "split": d.split_days >= 3,
    })
    comp["patterns"] = np.minimum(pat.sum(axis=1) / 4, 1.0)
    comp.loc[d.n_edges == 0, :] = 0.0
    for k in comp:
        d[f"c_{k}"] = comp[k]

    d["priority_raw"] = sum(C.PRIORITY_WEIGHTS[k] * comp[k] for k in C.PRIORITY_WEIGHTS)
    d["priority_seed_factor"] = np.where(d.is_seed, C.SEED_NOVELTY_FACTOR, 1.0)
    adjusted = d.priority_raw * d.priority_seed_factor
    adjusted = adjusted.where(d.n_edges > 0, 0.0)
    d["priority_normalizer"] = float(adjusted.max()) or 1.0
    d["priority_score"] = (adjusted / d.priority_normalizer).round(4)
    ordered = d.reset_index().sort_values(["priority_score", "gid"], ascending=[False, True])
    ranks = pd.Series(range(1, len(d)+1), index=ordered.gid)
    d["rank"] = ranks.reindex(d.index).astype(int)
    d["why"] = [_why(r) for r in d.itertuples()]
    return d


def _why(r) -> str:
    facts = []
    if r.seed_upstream >= 2:
        facts.append(f"достижим из {r.seed_upstream} seed (≤4 рёбер); модель смешивания ≈{kzt(r.seed_kzt_in)}")
    if r.n_return_cycles:
        facts.append(f"{r.n_return_cycles} циклов с возрастающими датами (не трассировка средств)")
    if r.sync_payers_max >= 3:
        facts.append(f"{r.sync_payers_max} плательщиков в один день ({r.sync_day})")
    if r.fast_in_share >= C.TRANSIT_FAST_SHARE and r.in_kzt >= C.TRANSIT_MIN_KZT:
        facts.append(f"{r.fast_in_share:.0%} пригодного входа совместимо с лагом 1–{C.FAST_DAYS} дн.")
    if r.repeated_routes:
        facts.append(f"{r.repeated_routes} повторяющихся маршрутов A→узел→B")
    if r.anomaly:
        facts.append(f"{ANOMALY_COLS[r.anomaly_feature]} — топ-1% своего колена")
    if r.betweenness > 0.001:
        facts.append(f"посредник: betweenness {r.betweenness:.4f}")
    action = "исходящие следующего колена и полную выписку" if r.truncated else ("полную выписку: наблюдений для роли недостаточно" if r.n_edges == 0 else ACTION[r.role])
    comps = {k: getattr(r, f"c_{k}") * C.PRIORITY_WEIGHTS[k] for k in C.PRIORITY_WEIGHTS}
    top = sorted(comps.items(), key=lambda kv: -kv[1])[:3]
    drivers = ", ".join(COMPONENT_RU[k] for k, value in top if value > 0) or "нет наблюдаемых сигналов"
    seed = " Уже в деле (seed)." if r.is_seed else ""
    text = (f"{C.ROLE_RU[r.role].capitalize()} (соответствие правилу {r.role_score:.2f}): {r.evidence}. "
            f"Главные факторы: {drivers}. " + ("Признаки: " + "; ".join(facts) + ". " if facts else "")
            + f"Проверить: {action}.{seed}")
    return text

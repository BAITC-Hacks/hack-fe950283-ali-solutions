"""Приоритет для аналитика: взвешенная сумма пяти интерпретируемых компонент 0..1."""
import numpy as np
import pandas as pd

from . import config as C
from .fmt import CYCLES, PAYERS, ROUTES, TRANSFERS, kzt, plural
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
    "coordinator": "источник средств и полную выписку (вход извне выборки), связи с другими хабами",
    "consolidator": "куда уходят накопления: снятие наличных, межбанк, покупка активов",
    "distributor": "источник средств (вход в выборке занижен) и получателей веера",
    "transit": "цепочку до и после узла: возможен транзитный счёт",
    "terminal": "полную выписку, снятие наличных и межбанковские переводы: видимый выход невелик",
    "peripheral": "наличие дополнительных сведений: выбранные пороги ролей не достигнуты",
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
    for k in comp:
        d[f"c_{k}"] = comp[k].round(4)

    raw = sum(C.PRIORITY_WEIGHTS[k] * comp[k] for k in C.PRIORITY_WEIGHTS)
    raw = raw * np.where(d.is_seed, C.SEED_NOVELTY_FACTOR, 1.0)
    d["priority_score"] = (raw / raw.max()).round(4)
    d["rank"] = d.priority_score.rank(ascending=False, method="first").astype(int)
    # для текста: в какой верхний процент узлов входит посредничество (betweenness)
    d["betweenness_top_pct"] = np.ceil(100 * (1 - d.betweenness.rank(pct=True))).clip(lower=1).astype(int)
    d["why"] = [_why(r) for r in d.itertuples()]
    return d


def _why(r) -> str:
    facts = []
    if r.seed_upstream >= 2:
        facts.append(f"к узлу ведут цепочки переводов от {r.seed_upstream} seed "
                     f"(их средств по пропорциональной атрибуции ≈{kzt(r.seed_kzt_in)})")
    if r.n_return_cycles:
        facts.append(f"{plural(r.n_return_cycles, CYCLES)} (по датам деньги могли вернуться к отправителю)")
    if r.sync_payers_max >= 3:
        facts.append(f"{plural(r.sync_payers_max, PAYERS)} в один день ({r.sync_date})")
    if r.fast_in_share >= C.TRANSIT_FAST_SHARE and r.in_kzt >= C.TRANSIT_MIN_KZT:
        facts.append(f"{r.fast_in_share:.0%} полученного ушло ≤{C.FAST_DAYS} дн.")
    if r.burst:
        facts.append(f"всплеск активности: {plural(r.burst_tx, TRANSFERS)} за {C.BURST_WINDOW_DAYS} дн. с {r.burst_date}")
    if r.repeated_routes:
        facts.append(f"{plural(r.repeated_routes, ROUTES)} A→узел→B")
    if r.anomaly:
        facts.append(f"{ANOMALY_COLS[r.anomaly_feature]} — в топ-1% своего колена")
    if r.betweenness > 0.001:
        facts.append(f"посредник между частями сети (топ-{r.betweenness_top_pct}% узлов по числу проходящих через него цепочек)")
    comps = {k: getattr(r, f"c_{k}") * C.PRIORITY_WEIGHTS[k] for k in C.PRIORITY_WEIGHTS}
    top = sorted(comps.items(), key=lambda kv: -kv[1])[:3]
    drivers = ", ".join(COMPONENT_RU[k] for k, _ in top)
    role = C.ROLE_RU[r.role].capitalize() + (", seed — уже в деле" if r.is_seed else "")
    action = "исходящие 5-го колена: отсутствие выгрузки не означает удержание средств" if r.truncated else ACTION[r.role]
    return (f"{role} (уверенность {r.role_score:.2f}): {r.evidence}. "
            f"Главные факторы: {drivers}. " + ("Признаки: " + "; ".join(facts) + ". " if facts else "")
            + f"Проверить: {action}.")

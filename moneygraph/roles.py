"""Роли: формальные правила (gate) → сила признака (strength) → role_score и evidence.

Роль назначается ТОЛЬКО при выполнении правила-порога из config.py. Правила
проверяются в порядке config.ROLES (координатор → консолидатор → распределитель →
транзит → конечный получатель): структурная роль важнее потоковой, поэтому узел,
который собирает от 6 плательщиков и ничего не отдаёт, — консолидатор, а
«конечный получатель» уходит в role_secondary. role_score = 0.5 + 0.5·strength:
прохождение порога даёт уверенность ≥0.5, сила признака добавляет остальное.
"""
import numpy as np
import pandas as pd

from . import config as C
from .fmt import kzt, pct, clip_text


def sat(x, lo, hi):
    """Насыщение по лог-шкале: lo → 0, hi → 1."""
    x = np.maximum(np.asarray(x, dtype=float), 1e-9)
    return np.clip((np.log(x) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0.0, 1.0)


def assign(G, df: pd.DataFrame) -> pd.DataFrame:
    d = df
    pt = d.pass_through.fillna(0.0)
    nonseed_obs = (~d.is_seed) & d.out_observed

    # ---------- правила
    hub = (d.in_deg >= C.HUB_MIN_PAYERS) & (d.out_deg >= C.HUB_MIN_RECIPIENTS)
    out_side = d.out_deg >= 2 * d.in_deg  # у хаба сторона «раздаёт» доминирует
    distr = ((d.out_deg >= C.DISTR_MIN_RECIPIENTS) & (d.out_deg >= C.DISTR_FANOUT_RATIO * d.in_deg)) \
        | (hub & out_side)
    cons = (((d.in_deg >= C.CONS_MIN_PAYERS)
             | ((d.in_deg >= C.CONS_ALT_PAYERS) & (d.seed_payers >= C.CONS_ALT_SEED_PAYERS)))
            & (d.in_deg >= d.out_deg)) | (hub & ~out_side)
    key = hub | distr | cons
    key_set = set(d.index[key])
    d["key_links"] = [len((set(G.predecessors(g)) | set(G.successors(g))) & key_set - {g}) for g in d.index]
    coord = hub & (d.key_links >= C.COORD_MIN_KEY_LINKS)

    pt_ok = pt.between(*C.TRANSIT_PT)
    fast_ok = (d.fast_in_share >= C.TRANSIT_FAST_SHARE) & pt.between(*C.TRANSIT_FAST_PT)
    transit = nonseed_obs & (d.out_deg > 0) & (pt_ok | fast_ok) & (np.minimum(d.in_kzt, d.out_kzt) >= C.TRANSIT_MIN_KZT)
    seed_transit = d.is_seed & (d.out_kzt >= C.SEED_TRANSIT_MIN_KZT) & d.out_deg.between(1, C.SEED_TRANSIT_MAX_RCPT)

    material = (d.in_kzt >= C.TERM_MIN_KZT) | (d.in_deg >= C.TERM_MIN_PAYERS) | (d.in_tx >= C.TERM_MIN_TX)
    p_sink = 1.0 - d.p_forward
    term_obs = d.out_observed & (d.in_deg > 0) & (pt <= C.TERM_MAX_PT) & material
    term_trunc = d.truncated & (p_sink >= C.TRUNC_TERMINAL_P) & material
    terminal = term_obs | term_trunc

    # ---------- сила признака 0..1
    S = pd.DataFrame(0.0, index=d.index, columns=C.ROLES)
    S["coordinator"] = np.where(coord, (sat(d.in_deg, 5, 25) + sat(d.out_deg, 10, 100) + sat(d.key_links, 2, 10)) / 3, 0)
    fanin = d.in_deg / (d.in_deg + d.out_deg).replace(0, 1)
    S["consolidator"] = np.where(cons, 0.5 * sat(d.in_deg, 3, 20) + 0.25 * fanin + 0.25 * np.minimum(d.seed_payers / 3, 1), 0)
    fanout = 1 - d.in_deg / d.out_deg.replace(0, np.nan)
    S["distributor"] = np.where(distr, 0.6 * sat(d.out_deg, 10, 100) + 0.4 * fanout.fillna(0), 0)
    pt_close = np.clip(1 - np.abs(np.log(pt.clip(lower=1e-9))) / np.log(2), 0, 1)
    S["transit"] = np.where(transit, 0.5 * pt_close + 0.5 * d.fast_in_share, 0)
    S.loc[seed_transit, "transit"] = (0.8 * (0.5 * d.top_out_share + 0.5 * sat(d.out_kzt, 1e5, 2e6)))[seed_transit]
    retention = 1 - np.minimum(pt, 1)
    term_s = 0.4 * retention + 0.3 * sat(d.in_kzt, 1e5, 2e6) + 0.3 * sat(d.in_deg, 1, 6)
    S["terminal"] = np.where(terminal, term_s, 0)

    gated = pd.DataFrame({
        "coordinator": coord, "consolidator": cons, "distributor": distr,
        "transit": transit | seed_transit, "terminal": terminal,
    })
    has = gated.any(axis=1)
    d["role"] = np.where(has, gated.idxmax(axis=1), "peripheral")  # первая True по порядку ROLES
    best = pd.Series([S.at[g, r] if r != "peripheral" else 0.0 for g, r in zip(d.index, d.role)], index=d.index)
    d["role_score"] = np.where(has, 0.5 + 0.5 * best, 0.0)
    d.loc[(d.role == "terminal") & d.truncated, "role_score"] *= p_sink
    d["role_secondary"] = [
        ";".join(r for r in gated.columns if gated.at[g, r] and r != d.at[g, "role"]) for g in d.index
    ]

    # периферия: уверенность тем выше, чем дальше узел от любого порога
    near = np.maximum.reduce([
        sat(d.in_deg, 1, C.CONS_MIN_PAYERS), sat(d.out_deg, 1, C.DISTR_MIN_RECIPIENTS),
        sat(d[["in_kzt", "out_kzt"]].max(axis=1), 1e4, 1e6),
    ])
    periph = 0.9 - 0.4 * near
    periph = np.where(d.truncated, periph * (1 - 0.5 * d.p_forward), periph)
    isolated = d.n_edges == 0
    periph = np.where(isolated, 0.3, periph)
    d["role_score"] = np.where(d.role == "peripheral", periph, d.role_score).round(3)

    d["flags"] = _flags(d)
    d["evidence"] = [clip_text(_evidence(r), 200) for r in d.itertuples()]
    return d


def _flags(d):
    f = pd.DataFrame(index=d.index)
    f["быстрый_транзит"] = (d.fast_in_share >= C.TRANSIT_FAST_SHARE) & (d.in_kzt >= C.TRANSIT_MIN_KZT)
    f["синхронный_сбор"] = d.sync_payers_max >= 3
    f["возвратные_циклы"] = d.n_return_cycles > 0
    f["аномалия_колена"] = d.anomaly
    f["дробление"] = d.split_days >= 3
    f["обрыв_4_колена"] = d.truncated
    f["seed"] = d.is_seed
    return [";".join(c for c in f.columns if f.at[g, c]) for g in d.index]


def _evidence(r) -> str:
    seedp = f" (seed: {r.seed_payers})" if r.seed_payers else ""
    up = f"; деньги {r.seed_upstream} seed доходят до узла" if r.seed_upstream >= 3 else ""
    fwd = r.out_kzt / r.in_kzt if r.in_kzt else 0
    if r.role == "coordinator":
        return (f"Хаб: {r.in_deg} плательщиков{seedp} → {r.out_deg} получателей; вход {kzt(r.in_kzt)}, "
                f"выход {kzt(r.out_kzt)}; связан с {r.key_links} ключевыми узлами"
                + (f"; возвратных циклов: {r.n_return_cycles}" if r.n_return_cycles else ""))
    if r.role == "consolidator":
        if not r.out_observed:
            tail = "исходящие не выгружены (4-е колено)"
        elif not r.out_deg:
            tail = "дальше не ушло ничего"
        elif fwd > 1.2:
            tail = f"отдал {kzt(r.out_kzt)} {r.out_deg} получ. — больше, чем получил в выборке"
        else:
            tail = f"дальше ушло {pct(fwd)} ({r.out_deg} получ.)"
        return f"Сбор от {r.in_deg} плательщиков{seedp}: {kzt(r.in_kzt)}; {tail}{up}"
    if r.role == "distributor":
        med = r.out_kzt / max(r.out_tx, 1)
        return (f"Веер: {r.out_deg} получателей при {r.in_deg} плательщиках; разослал {kzt(r.out_kzt)}, "
                f"в среднем {kzt(med)} за перевод"
                + ("; вход занижен (seed)" if r.is_seed else ""))
    if r.role == "transit":
        if r.is_seed:
            return (f"Признаки передачи средств seed: вход неполон; переправил {kzt(r.out_kzt)} {r.out_deg} получ., "
                    f"{pct(r.top_out_share)} — одному")
        fast = f"; {pct(r.fast_in_share)} ушло ≤{C.FAST_DAYS} дн. после поступления" if r.fast_in_share >= 0.3 else ""
        return f"Пропуск {pct(fwd)}: получил {kzt(r.in_kzt)} от {r.in_deg}, отдал {kzt(r.out_kzt)} {r.out_deg} получ.{fast}"
    if r.role == "terminal":
        if r.truncated:
            return (f"4-е колено, исходящие не выгружены; модель: P(сток)={1 - r.p_forward:.2f}; "
                    f"получил {kzt(r.in_kzt)} от {r.in_deg}{up}")
        kept = "исходящих нет" if r.out_deg == 0 else f"дальше ушло лишь {pct(fwd)}"
        return f"Получил {kzt(r.in_kzt)} от {r.in_deg} плательщиков{seedp}, {kept} (исходящие наблюдаемы){up}"
    # peripheral
    if r.n_edges == 0:
        return "Узел без переводов ≥5 тыс ₸ внутри банка за период: данных для роли нет; нужен запрос входящих и межбанка"
    if r.truncated:
        return (f"4-е колено (обход оборван): {r.in_tx} поступл. на {kzt(r.in_kzt)} от {r.in_deg}; "
                f"P(передаёт дальше)={r.p_forward:.2f} — порогов ролей не достигает")
    parts = []
    if r.in_deg:
        parts.append(f"получил {kzt(r.in_kzt)} от {r.in_deg}")
    if r.out_deg:
        parts.append(f"отдал {kzt(r.out_kzt)} {r.out_deg} получ.")
    if r.in_kzt and r.out_observed and not r.is_seed and r.out_deg:
        parts.append(f"пропуск {pct(fwd)}")
    base = ", ".join(parts)
    return f"{base[:1].upper()}{base[1:]}; пороги ролей не достигнуты" + ("; seed: вход занижен" if r.is_seed else "")

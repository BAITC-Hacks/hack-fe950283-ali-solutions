"""Оценка полноты (какие данные запросить) и сводный отчёт report.md."""
import pandas as pd

from . import config as C
from .fmt import kzt, short


def data_requests(df: pd.DataFrame) -> pd.DataFrame:
    """Белые пятна выгрузки → конкретный запрос по конкретному gid."""
    rows = []
    d = df
    t = d[d.truncated & (d.p_forward >= 0.5) & (d.in_kzt >= 100_000)]
    for g, r in t.iterrows():
        rows.append((g, "исходящие 5-го колена",
                     f"обход оборван; модель: P(передаёт дальше)={r.p_forward:.2f}; получил {kzt(r.in_kzt)}",
                     r.in_kzt * r.p_forward))
    ext = d[(~d.is_seed) & (d.out_kzt > 1.2 * d.in_kzt + 100_000)]
    for g, r in ext.iterrows():
        rows.append((g, "входящие извне выборки",
                     f"отдал {kzt(r.out_kzt)}, получил в выборке {kzt(r.in_kzt)}: разница {kzt(r.out_kzt - r.in_kzt)}; проверить начальный остаток и невидимый вход",
                     r.out_kzt - r.in_kzt))
    s = d[d.is_seed & (d.out_kzt >= 100_000)]
    for g, r in s.iterrows():
        rows.append((g, "входящие seed (источник выручки)",
                     f"seed переправил {kzt(r.out_kzt)}; входящие seed в выгрузке неполны", r.out_kzt))
    iso = d[d.n_edges == 0]
    for g, r in iso.iterrows():
        rows.append((g, "межбанк / наличные / переводы <5 тыс ₸",
                     "узел без внутрибанковских переводов ≥5 тыс ₸ за период", 0.0))
    sink = d[d.out_observed & (d.out_deg == 0) & (d.in_kzt >= 500_000)]
    for g, r in sink.iterrows():
        rows.append((g, "снятие наличных / межбанк / покупки",
                     f"получил {kzt(r.in_kzt)} от {r.in_deg}, внутрибанковских исходящих нет", r.in_kzt))
    out = pd.DataFrame(rows, columns=["gid", "request", "reason", "weight_kzt"])
    out = out.merge(d[["priority_score"]], left_on="gid", right_index=True)
    return out.sort_values(["priority_score", "weight_kzt"], ascending=False).reset_index(drop=True)


def _table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("-" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("|", "/").replace("\n", " ") for c in cols) + " |")
    return "\n".join(lines)


def write_report(path, ctx) -> None:
    df, cl, stats, trunc, res, cyc, req = (ctx[k] for k in ("df", "clusters", "stats", "trunc", "resilience", "cycles", "requests"))
    L = []
    L.append("# Граф денег — аналитическая справка\n")
    L.append("> Автоматически сгенерировано `python run.py`. Все выводы — **гипотезы для проверки**, "
             "а не утверждения о виновности. Данные обезличены (gid).\n")
    L.append("## Сеть\n")
    L.append(f"- узлов **{stats['n_nodes']}**, рёбер **{stats['n_edges']}**, транзакций **{stats['n_tx']}**, "
             f"seed **{stats['n_seed']}** (без переводов: {stats['isolated_seeds']})")
    L.append(f"- оборот **{kzt(stats['turnover_kzt'])}**, период {stats['date_min']} — {stats['date_max']}")
    L.append(f"- кластеров **{cl.shape[0]}** (Louvain, модулярность {ctx['modularity']:.3f})")
    L.append(f"- слабосвязных компонент с учётом изолятов: **{stats['weak_components']}**; seed без исходящих: **{stats['seed_without_outgoing']}**")
    L.append(f"- возвратных циклов (≤6 шагов, согласованы по датам): **{sum(c['returned'] for c in cyc)}** из {len(cyc)}\n")

    L.append("## Роли\n")
    rc = df.role.value_counts().reindex(C.ROLES).fillna(0).astype(int)
    L.append(_table(pd.DataFrame({"роль": rc.index, "смысл": [C.ROLE_RU[r] for r in rc.index],
                                  "узлов": rc.values,
                                  "из них seed": [int(df[(df.role == r) & df.is_seed].shape[0]) for r in rc.index]})))
    L.append("")

    L.append("## Топ-20: кого смотреть первым\n")
    top = df.sort_values("rank").head(20)
    L.append(_table(pd.DataFrame({
        "#": top["rank"].values, "gid": top.index, "роль": top.role.values,
        "priority": top.priority_score.values, "seed": top.is_seed.map({True: "да", False: ""}).values,
        "обоснование": top.evidence.values})))
    L.append("")

    L.append("## Кластеры (первые 12 по приоритету)\n")
    c = cl[cl.cluster_id > 0].head(12)
    L.append(_table(c[["cluster_id", "archetype", "n_nodes", "n_seed", "sum_kzt_internal", "stability", "hypothesis"]]))
    L.append("")

    L.append("## Артефакт обрыва 4-го колена\n")
    L.append(f"Логистическая регрессия «передаёт ли узел деньги дальше» обучена на {trunc['train_nodes']} узлах "
             f"1–3 колена (их исходящие наблюдаемы; доля стоков {trunc['train_sink_rate']:.0%}) только по признакам "
             f"входящей стороны. Кросс-валидация: **ROC-AUC {trunc['cv_auc']:.2f}**.\n")
    L.append(f"- обрезанных узлов: {trunc['truncated_nodes']} — наивно все они были бы «стоками»")
    L.append(f"- Brier по отложенным предсказаниям: **{trunc['cv_brier']:.3f}**, постоянный прогноз: **{trunc['baseline_brier']:.3f}** (меньше лучше)")
    L.append(f"- неопределённая зона 0.4 < P(передаёт) < 0.6: **{trunc['uncertain_nodes']}** узлов")
    L.append(f"- ожидаемо настоящих стоков: **≈{trunc['expected_true_sinks']:.0f}**; "
             f"вероятно передают дальше (P≥0.6): {trunc['likely_forwarders']}; вероятные стоки (P(сток)≥0.6): {trunc['likely_sinks']}")
    L.append("- коэффициенты (стандартизованные признаки): " +
             ", ".join(f"{k}: {v:+.2f}" for k, v in trunc["coefficients"].items()) + "\n")
    L.append("Оценки модели не доказывают удержание средств: проверка проводится на коленах 1–3, а у 4-го возможен сдвиг распределения. role_score — сила соответствия правилу, не вероятность виновности.\n")
    L.append("FIFO расходует сначала старые поступления и только затем новые. В быстрый транзит входят лишь сопоставления с лагом ≤2 дня. Порядок внутри дня неизвестен, поэтому это возможный транзит, а не доказанная трассировка конкретных денег. Превышение выхода над входом может объясняться начальным остатком.\n")

    L.append("## Устойчивость: что если заблокировать N узлов\n")
    L.append(_table(res[res.n_removed.isin([0, 10, 20])]))
    L.append("")

    L.append("## Оценка полноты: какие данные запросить дальше\n")
    cnt = req.request.value_counts()
    L.append(_table(pd.DataFrame({"запрос": cnt.index, "узлов": cnt.values})))
    L.append("\nПервые 15 запросов по приоритету узла:\n")
    L.append(_table(req.head(15)[["gid", "request", "reason"]]))
    L.append("")

    L.append("## Топ возвратных циклов (по «узкому месту» суммы)\n")
    rc_ = sorted([c for c in cyc if c["returned"]], key=lambda c: -c["bottleneck_kzt"])[:10]
    L.append(_table(pd.DataFrame({
        "длина": [c["len"] for c in rc_],
        "маршрут": [" → ".join(short(g) for g in c["nodes"] + c["nodes"][:1]) for c in rc_],
        "мин. сумма на шаге": [kzt(c["bottleneck_kzt"]) for c in rc_]})))
    path.write_text("\n".join(L), encoding="utf-8")

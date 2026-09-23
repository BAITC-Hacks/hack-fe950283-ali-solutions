"""Data requests and deterministic, qualified summary."""
import pandas as pd
from .fmt import kzt

def data_requests(df):
    rows = []
    for g,r in df.iterrows():
        if r.truncated:
            model = f"; диагностическая модель={r.p_forward:.2f}" if pd.notna(r.p_forward) else "; модель недоступна"
            rows.append((g, "исходящие 5-го колена",
                         f"Исходящие не наблюдаются; вход {kzt(r.in_kzt)}{model}", r.in_kzt))
        if not r.is_seed and r.out_kzt > 1.2*r.in_kzt+100000:
            rows.append((g, "полная выписка и начальный остаток",
                         f"Выход {kzt(r.out_kzt)} больше входа {kzt(r.in_kzt)}; возможны внешний вход или начальный остаток",
                         r.out_kzt-r.in_kzt))
        if r.is_seed and r.out_kzt > 0:
            rows.append((g, "входящие seed и начальный остаток", "Вход seed неполон; назначение средств неизвестно", r.out_kzt))
        if r.n_edges == 0:
            rows.append((g, "полная выписка", "0 внешних контрагентов в наблюдаемой выгрузке", 0.0))
        if r.out_observed and r.out_deg == 0 and r.in_kzt >= 500000:
            rows.append((g, "операции вне наблюдаемой выгрузки", f"Вход {kzt(r.in_kzt)}, наблюдаемых исходящих нет", r.in_kzt))
    table = pd.DataFrame(rows, columns=["gid","request","reason","weight_kzt"])
    table["gid"] = table.gid.astype("int64")
    table = table.merge(df[["priority_score"]], left_on="gid", right_index=True)
    return table.sort_values(["priority_score","weight_kzt","gid"], ascending=[False,False,True]).reset_index(drop=True)

def _table(df):
    cols=list(df.columns)
    lines=["| "+" | ".join(cols)+" |", "|"+"|".join("-" for _ in cols)+"|"]
    for row in df.itertuples(index=False, name=None):
        lines.append("| "+" | ".join(str(v).replace("|","/").replace("\n"," ") for v in row)+" |")
    return "\n".join(lines)

def write_report(path, ctx):
    d,s,t=ctx["df"],ctx["stats"],ctx["trunc"]
    text=["# MoneyGraph — аналитическая справка",
          "> Роли — эвристики наблюдаемой сети. Связи и суммы не доказывают назначение операций.",
          f"Run: {ctx['manifest']['run_id']}",
          f"Период операций: {s['date_min']} — {s['date_max']}; конец наблюдения: {s['period_end']}.",
          f"Узлов: {s['n_nodes']}; рёбер: {s['n_edges']}; транзакций: {s['n_tx']}; seed: {s['n_seed']}.",
          f"Компонент: {s['components_all']} со всеми узлами, {s['components_without_isolates']} без изолятов.",
          f"Наблюдаемый оборот: {kzt(s['turnover_kzt'])}.",
          "## Роли", _table(d.role.value_counts().rename_axis("role").reset_index(name="nodes")),
          "## Топ-20", _table(d.sort_values("rank").head(20).reset_index()[["gid","role","priority_score","evidence"]]),
          "## Ограничения",
          "FIFO: совместимость объёмов через 1–2 календарных дня; последний двухдневный интервал исключён из знаменателя. Активность одного дня показана отдельно.",
          "Достижимость от seed ≤4 рёбер — структурная связь. Пропорциональная атрибуция — модель смешивания, не трассировка конкретных денег.",
          "Выход больше входа может объясняться внешними поступлениями или начальным остатком.",
          "Граница depth=4 и seed с неполным входом не получают terminal. Прогноз модели не назначает роли.",
          "## Диагностическая модель",
          f"Статус: {t['status']}; причина недоступности: {t['reason']}.",
          f"CV AUC: {t['cv_auc']}; целевая переменная — наличие исходящих на depth=1–3, не правильность финансовой роли. Перенос на depth=4 не валидирован.",
          "## Кластеры", _table(ctx["clusters"][["cluster_id","n_nodes","n_seed","hypothesis"]]),
          "## Циклы",
          f"Структурных циклов: {len(ctx['cycles'])}; строго возрастающие даты: {sum(c['returned'] for c in ctx['cycles'])}; лимит достигнут: {ctx['G'].graph.get('cycles_truncated', False)}.",
          "Даже возрастающие даты не доказывают возврат тех же средств.",
          "## Исключение узлов из копии графа", _table(ctx["resilience"]),
          "Симуляция описывает наблюдаемую структуру без адаптации участников; не оценка предотвращённого ущерба.",
          "## Запросы данных", _table(ctx["requests"].head(30))]
    path.write_text("\n\n".join(text)+"\n", encoding="utf-8")

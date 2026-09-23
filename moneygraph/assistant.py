"""AI-ассистент аналитика: вопрос на естественном языке → ответ по графу со ссылками на gid.

Два режима с одними и теми же инструментами над графом:
  * LLM-агент (OpenAI Chat Completions + function calling), если задан OPENAI_API_KEY.
    OPENAI_BASE_URL позволяет подключить любой OpenAI-совместимый сервер, в т.ч.
    локальную модель (Ollama, vLLM) — данные не покидают контур банка.
  * офлайн: разбор типовых вопросов по шаблонам — демо работает без интернета.
LLM не видит сырых данных целиком: только то, что вернули инструменты.
"""
import json
import os
import re
import urllib.error
import urllib.request
from collections import defaultdict

import networkx as nx

from . import config as C
from .fmt import kzt, short

ROLE_WORDS = {
    "координатор": "coordinator", "организатор": "coordinator", "консолидатор": "consolidator",
    "сборщик": "consolidator", "распределител": "distributor", "веер": "distributor",
    "транзит": "transit", "конечн": "terminal", "терминал": "terminal", "периферия": "peripheral",
}

SYSTEM_PROMPT = f"""Ты — ассистент AML-аналитика банка. Отвечаешь на вопросы о графе внутрибанковских переводов
(июль 2026, 2 248 клиентов, 81 seed — клиенты из списка правоохранительных органов). Роли узлов: {", ".join(C.ROLES)}.
Правила:
- Все факты бери ТОЛЬКО из инструментов. Не выдумывай gid, суммы и связи.
- Называй узлы полным 18-значным gid, суммы — в тенге.
- Формулируй выводы как признаки и гипотезы для проверки («признаки консолидации»), а не как утверждения о виновности.
- Учитывай ограничения выгрузки: только исходящие переводы от seed на 4 колена, у 4-го колена исходящие не выгружены,
  переводы < 5 000 ₸ не видны, вход seed занижен.
- Отвечай по-русски, кратко: 3–8 пунктов, в конце — что проверить дальше."""


class GraphTools:
    def __init__(self, ctx: dict):
        self.G: nx.DiGraph = ctx["G"]
        self.df = ctx["df"]
        self.clusters = ctx["clusters"].set_index("cluster_id")
        self.requests = ctx["requests"]
        self.trunc = ctx["trunc"]
        self.stats = ctx["stats"]
        self.by_short = {short(g): g for g in self.df.index}

    # ---------------------------------------------------------------- утилиты
    def resolve(self, x):
        s = str(x).strip()
        if s.isdigit() and int(s) in self.df.index:
            return int(s)
        return self.by_short.get(s)

    def brief(self, g) -> dict:
        r = self.df.loc[g]
        return {"gid": str(g), "role": r.role, "priority_rank": int(r["rank"]), "priority": round(float(r.priority_score), 3),
                "is_seed": bool(r.is_seed), "cluster_id": int(r.cluster_id), "evidence": r.evidence}

    # ---------------------------------------------------------------- инструменты
    def node_card(self, gid):
        g = self.resolve(gid)
        if g is None:
            return {"error": f"gid {gid} не найден"}
        r = self.df.loc[g]
        payers = sorted(self.G.in_edges(g, data=True), key=lambda e: -e[2]["sum_kzt"])[:8]
        rcpts = sorted(self.G.out_edges(g, data=True), key=lambda e: -e[2]["sum_kzt"])[:8]
        return {
            **self.brief(g), "role_score": round(float(r.role_score), 2), "also_matches": r.role_secondary,
            "depth": int(r.depth), "in_kzt": round(r.in_kzt), "out_kzt": round(r.out_kzt), "payers": int(r.in_deg),
            "recipients": int(r.out_deg), "seed_upstream": int(r.seed_upstream), "seed_money_in_kzt": round(r.seed_kzt_in),
            "outgoing_observed": bool(r.out_observed), "p_forward_if_truncated": round(float(r.p_forward), 2) if r.truncated else None,
            "flags": r["flags"], "why": r.why,
            "top_payers": [{"gid": str(u), "kzt": round(d["sum_kzt"]), "n_tx": d["n_tx"], "role": self.df.at[u, "role"]} for u, _, d in payers],
            "top_recipients": [{"gid": str(v), "kzt": round(d["sum_kzt"]), "n_tx": d["n_tx"], "role": self.df.at[v, "role"]} for _, v, d in rcpts],
            "data_gaps": self.requests[self.requests.gid == g][["request", "reason"]].to_dict("records"),
        }

    def common_downstream(self, gids, max_hops=3):
        """Кто собирает деньги с нескольких узлов: узлы, достижимые по направлению денег из ≥2 заданных."""
        src = [self.resolve(x) for x in gids]
        src = [g for g in src if g is not None]
        if len(src) < 2:
            return {"error": "нужно минимум два известных gid"}
        reach = defaultdict(dict)
        for s in src:
            for v, dist in nx.single_source_shortest_path_length(self.G, s, cutoff=int(max_hops)).items():
                if v != s:
                    reach[v][s] = dist
        rows = []
        for v, d in reach.items():
            if len(d) >= 2:
                rows.append({**self.brief(v), "reached_from": len(d), "of": len(src),
                             "hops": {str(k): h for k, h in d.items()}})
        rows.sort(key=lambda x: (-x["reached_from"], -x["priority"]))
        direct = {}
        for s in src:
            for _, v, dd in self.G.out_edges(s, data=True):
                direct.setdefault(v, []).append((s, dd["sum_kzt"]))
        shared_direct = [{"gid": str(v), "role": self.df.at[v, "role"], "from": [str(s) for s, _ in l],
                          "kzt": round(sum(a for _, a in l))} for v, l in direct.items() if len(l) >= 2]
        return {"sources": [str(s) for s in src], "direct_common_recipients": shared_direct[:10], "reachable_common": rows[:10]}

    def trace(self, gid, direction="up", max_hops=2, limit=15):
        """Цепочка денег: up — откуда пришли, down — куда ушли. Рёбра с суммами, по убыванию."""
        g = self.resolve(gid)
        if g is None:
            return {"error": f"gid {gid} не найден"}
        H = self.G.reverse(copy=False) if direction == "up" else self.G
        dist = nx.single_source_shortest_path_length(H, g, cutoff=int(max_hops))
        edges = []
        for u, v, d in H.edges(dist.keys(), data=True):
            if v in dist and dist[v] == dist[u] + 1:
                a, b = (v, u) if direction == "up" else (u, v)
                edges.append({"from": str(a), "to": str(b), "kzt": round(d["sum_kzt"]), "n_tx": d["n_tx"], "hop": dist[v]})
        edges.sort(key=lambda e: (e["hop"], -e["kzt"]))
        nodes = sorted((v for v in dist if v != g), key=lambda v: self.df.at[v, "rank"])[:limit]
        return {"gid": str(g), "direction": direction, "n_nodes": len(dist) - 1, "edges": edges[:limit * 2],
                "key_nodes": [self.brief(v) for v in nodes]}

    def path_between(self, a, b):
        ga, gb = self.resolve(a), self.resolve(b)
        if ga is None or gb is None:
            return {"error": "gid не найден"}
        out = {"a": str(ga), "b": str(gb)}
        for name, (s, t) in {"a_to_b": (ga, gb), "b_to_a": (gb, ga)}.items():
            try:
                p = nx.shortest_path(self.G, s, t)
                out[name] = [{"from": str(u), "to": str(v), "kzt": round(self.G[u][v]["sum_kzt"])} for u, v in zip(p, p[1:])]
            except nx.NetworkXNoPath:
                out[name] = None
        if not out["a_to_b"] and not out["b_to_a"]:
            try:
                p = nx.shortest_path(self.G.to_undirected(as_view=True), ga, gb)
                out["undirected_path"] = [str(x) for x in p]
            except nx.NetworkXNoPath:
                out["undirected_path"] = None
        out["same_cluster"] = int(self.df.at[ga, "cluster_id"]) == int(self.df.at[gb, "cluster_id"])
        return out

    def top_nodes(self, role=None, cluster_id=None, limit=10):
        d = self.df
        if role:
            d = d[d.role == role]
        if cluster_id is not None:
            d = d[d.cluster_id == int(cluster_id)]
        return [self.brief(g) for g in d.sort_values("rank").index[: int(limit)]]

    def cluster_info(self, cluster_id):
        k = int(cluster_id)
        if k not in self.clusters.index:
            return {"error": f"кластера {k} нет"}
        c = self.clusters.loc[k]
        return {"cluster_id": k, "n_nodes": int(c.n_nodes), "n_seed": int(c.n_seed), "internal_kzt": round(c.sum_kzt_internal),
                "archetype": c.archetype, "stability": c.stability, "roles": c.roles, "hypothesis": c.hypothesis,
                "top": [self.brief(int(g)) for g in str(c.top_gids).split(";")]}

    def data_gaps(self, limit=10):
        r = self.requests.head(int(limit))
        return {"summary": self.requests.request.value_counts().to_dict(),
                "top": [{"gid": str(x.gid), "request": x.request, "reason": x.reason} for x in r.itertuples()]}

    def network_summary(self):
        return {**self.stats, "roles": self.df.role.value_counts().to_dict(), "n_clusters": len(self.clusters),
                "truncation_model": {k: self.trunc[k] for k in ("cv_auc", "expected_true_sinks", "truncated_nodes")},
                "top5": self.top_nodes(limit=5)}


TOOL_SPECS = [
    ("node_card", "Карточка узла: роль, обоснование, суммы, главные плательщики и получатели, пробелы в данных.",
     {"gid": {"type": "string", "description": "полный 18-значный gid или короткий 8-значный"}}, ["gid"]),
    ("common_downstream", "Кто собирает деньги с нескольких узлов: общие получатели (прямые и через ≤max_hops переводов).",
     {"gids": {"type": "array", "items": {"type": "string"}}, "max_hops": {"type": "integer", "default": 3}}, ["gids"]),
    ("trace", "Проследить деньги от узла: direction=up — откуда пришли, down — куда ушли.",
     {"gid": {"type": "string"}, "direction": {"type": "string", "enum": ["up", "down"]},
      "max_hops": {"type": "integer", "default": 2}}, ["gid", "direction"]),
    ("path_between", "Кратчайшая денежная цепочка между двумя узлами в обе стороны.",
     {"a": {"type": "string"}, "b": {"type": "string"}}, ["a", "b"]),
    ("top_nodes", "Топ узлов по приоритету, опционально с фильтром по роли и кластеру.",
     {"role": {"type": "string", "enum": C.ROLES}, "cluster_id": {"type": "integer"}, "limit": {"type": "integer", "default": 10}}, []),
    ("cluster_info", "Описание кластера: размер, seed, оборот, роли, гипотеза.",
     {"cluster_id": {"type": "integer"}}, ["cluster_id"]),
    ("data_gaps", "Какие данные запросить дальше, чтобы закрыть белые пятна выгрузки.",
     {"limit": {"type": "integer", "default": 10}}, []),
    ("network_summary", "Сводка по сети целиком.", {}, []),
]


def _tool_schema():
    return [{"type": "function", "function": {"name": n, "description": d,
                                              "parameters": {"type": "object", "properties": p, "required": r}}}
            for n, d, p, r in TOOL_SPECS]


class Assistant:
    def __init__(self, ctx: dict):
        self.tools = GraphTools(ctx)
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-5.4")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    @property
    def llm(self) -> bool:
        return bool(self.api_key)

    def _call(self, name, args):
        fn = getattr(self.tools, name, None)
        if fn is None or name.startswith("_") or name not in {t[0] for t in TOOL_SPECS}:
            return {"error": f"нет инструмента {name}"}
        try:
            return fn(**args)
        except Exception as e:  # ответ модели мог содержать неверные аргументы
            return {"error": f"{type(e).__name__}: {e}"}

    def ask(self, question: str, history=None) -> dict:
        if self.llm:
            try:
                return self._ask_llm(question, history or [])
            except (urllib.error.URLError, KeyError, ValueError, TimeoutError) as e:
                res = self._ask_offline(question)
                res["answer"] = f"(LLM недоступен: {e}; ответ офлайн-режима)\n\n" + res["answer"]
                return res
        return self._ask_offline(question)

    # ---------------------------------------------------------------- LLM-агент
    def _post(self, payload):
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read())

    def _ask_llm(self, question, history):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-6:] + [{"role": "user", "content": question}]
        used = []
        for _ in range(8):
            resp = self._post({"model": self.model, "messages": msgs, "tools": _tool_schema(), "tool_choice": "auto"})
            m = resp["choices"][0]["message"]
            calls = m.get("tool_calls") or []
            if not calls:
                answer = m.get("content") or ""
                return {"answer": answer, "mode": f"LLM {self.model}", "tools": used, "focus": _first_gid(answer)}
            msgs.append({"role": "assistant", "content": m.get("content"), "tool_calls": calls})
            for c in calls:
                name = c["function"]["name"]
                args = json.loads(c["function"].get("arguments") or "{}")
                used.append(name)
                result = self._call(name, args)
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result, ensure_ascii=False, default=str)[:12000]})
        return {"answer": "Не удалось завершить рассуждение за 8 шагов — уточните вопрос.", "mode": f"LLM {self.model}", "tools": used}

    # ---------------------------------------------------------------- офлайн
    def _gids(self, q):
        found = []
        for m in re.findall(r"\d{18}|\d{8}", q):
            g = self.tools.resolve(m)
            if g is not None and g not in found:
                found.append(g)
        return found

    def _ask_offline(self, q):
        T, ql, gids = self.tools, q.lower(), self._gids(q)
        role = next((r for w, r in ROLE_WORDS.items() if w in ql), None)
        num = re.search(r"топ[- ]?(\d+)", ql)
        limit = int(num.group(1)) if num else 10
        L, used, focus = [], [], None

        if len(gids) >= 2 and re.search(r"собира|общ|сход|консолид|кому|куда", ql):
            used.append("common_downstream")
            r = T.common_downstream(gids)
            L.append(f"Деньги от {len(r['sources'])} указанных узлов (по направлению переводов, ≤3 шага):")
            for x in r["direct_common_recipients"]:
                L.append(f"• {x['gid']} ({x['role']}) напрямую получает от {len(x['from'])} из них, всего {kzt(x['kzt'])}")
            for x in r["reachable_common"][:6]:
                L.append(f"• {x['gid']} ({x['role']}, приоритет #{x['priority_rank']}) — доходят деньги {x['reached_from']} из {x['of']}: {x['evidence']}")
            if len(L) == 1:
                L.append("• общих получателей в пределах 3 шагов не найдено")
            focus = (r["direct_common_recipients"] or r["reachable_common"] or [{}])[0].get("gid")
            L.append("Это признаки точки сбора, а не доказательство: проверьте выписки этих получателей.")
        elif len(gids) == 2 and re.search(r"связ|пут|цепоч|между", ql):
            used.append("path_between")
            r = T.path_between(*gids)
            for key, title in (("a_to_b", f"{r['a']} → {r['b']}"), ("b_to_a", f"{r['b']} → {r['a']}")):
                if r[key]:
                    L.append(f"Цепочка {title}: " + " → ".join([r[key][0]["from"]] + [f"[{kzt(e['kzt'])}] {e['to']}" for e in r[key]]))
                else:
                    L.append(f"Направленной цепочки {title} нет.")
            if r.get("undirected_path"):
                L.append("Без учёта направления связаны через: " + " — ".join(r["undirected_path"]))
            L.append("Один кластер." if r["same_cluster"] else "Разные кластеры.")
            focus = r["a"]
        elif gids and re.search(r"откуда|кто плат|кто перев|источник|вверх|кто отправ", ql):
            used.append("trace")
            r = T.trace(gids[0], "up")
            L.append(f"Откуда деньги у {r['gid']} (≤2 шага, узлов: {r['n_nodes']}):")
            L += [f"• {e['from']} → {e['to']}: {kzt(e['kzt'])} ({e['n_tx']} пер.), шаг {e['hop']}" for e in r["edges"][:10]]
            L.append("Ключевые узлы выше по потоку: " + ", ".join(f"{x['gid']} ({x['role']})" for x in r["key_nodes"][:5]))
            focus = r["gid"]
        elif gids and re.search(r"куда|кому|получател|вниз|ушл", ql):
            used.append("trace")
            r = T.trace(gids[0], "down")
            L.append(f"Куда ушли деньги {r['gid']} (≤2 шага, узлов: {r['n_nodes']}):")
            L += [f"• {e['from']} → {e['to']}: {kzt(e['kzt'])} ({e['n_tx']} пер.), шаг {e['hop']}" for e in r["edges"][:10]]
            focus = r["gid"]
        elif gids:
            for g in gids[:3]:
                used.append("node_card")
                c = T.node_card(g)
                L.append(f"{c['gid']} — {C.ROLE_RU[c['role']]} (уверенность {c['role_score']}), приоритет #{c['priority_rank']}"
                         + (", seed" if c["is_seed"] else "") + f", кластер {c['cluster_id']}.")
                L.append(f"Обоснование: {c['evidence']}")
                if c["top_payers"]:
                    L.append("Главные плательщики: " + ", ".join(f"{p['gid']} ({kzt(p['kzt'])})" for p in c["top_payers"][:3]))
                if c["top_recipients"]:
                    L.append("Главные получатели: " + ", ".join(f"{p['gid']} ({kzt(p['kzt'])})" for p in c["top_recipients"][:3]))
                for gap in c["data_gaps"]:
                    L.append(f"Не хватает данных: {gap['request']} — {gap['reason']}")
                L.append("")
            focus = str(gids[0])
        elif re.search(r"кластер\D*(\d+)", ql):
            used.append("cluster_info")
            c = T.cluster_info(re.search(r"кластер\D*(\d+)", ql).group(1))
            L.append(c.get("error") or f"Кластер {c['cluster_id']} ({c['archetype']}): {c['hypothesis']}\nКлючевые: "
                     + ", ".join(f"{x['gid']} ({x['role']})" for x in c["top"]))
        elif re.search(r"данн|запрос|полнот|не хватает|пятн", ql):
            used.append("data_gaps")
            r = T.data_gaps()
            L.append("Какие данные запросить (по числу узлов): " + "; ".join(f"{k} — {v}" for k, v in r["summary"].items()))
            L += [f"• {x['gid']}: {x['request']} — {x['reason']}" for x in r["top"]]
        elif role or re.search(r"топ|приоритет|первым|главн|кого", ql):
            used.append("top_nodes")
            rows = T.top_nodes(role=role, limit=min(limit, 30))
            L.append(f"Топ-{len(rows)}" + (f" ({C.ROLE_RU[role]})" if role else "") + " по приоритету:")
            L += [f"{x['priority_rank']}. {x['gid']} — {x['role']}{', seed' if x['is_seed'] else ''}: {x['evidence']}" for x in rows]
            focus = rows[0]["gid"] if rows else None
        elif re.search(r"обрыв|4.?(е|го)? колен|усеч", ql):
            t = T.trunc
            L.append(f"Модель обрыва: обучена на {t['train_nodes']} узлах 1–3 колена, ROC-AUC {t['cv_auc']:.2f}. "
                     f"Из {t['truncated_nodes']} узлов 4-го колена ожидаемо настоящих стоков ≈{t['expected_true_sinks']:.0f}.")
        else:
            s = T.network_summary()
            L.append(f"Сеть: {s['n_nodes']} узлов, {s['n_edges']} рёбер, {s['n_seed']} seed, {s['n_clusters']} кластеров.")
            L.append("Офлайн-режим понимает вопросы вида:\n• «Кто собирает деньги с <gid>, <gid>, <gid>?»\n• «Расскажи про <gid>»\n"
                     "• «Откуда деньги у <gid>?» / «Куда ушли деньги <gid>?»\n• «Как связаны <gid> и <gid>?»\n"
                     "• «Топ-5 консолидаторов» • «Кластер 3» • «Какие данные запросить?»\n"
                     "С OPENAI_API_KEY вопросы можно задавать в свободной форме.")
        return {"answer": "\n".join(L).strip(), "mode": "офлайн (шаблоны)", "tools": used, "focus": focus}


def _first_gid(text):
    m = re.search(r"\b\d{18}\b", text or "")
    return m.group(0) if m else None

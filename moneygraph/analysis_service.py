"""Read-only facts shared by UI and AI. All public identifiers are decimal strings."""
import hashlib
import json
import networkx as nx
from .contracts import clean, SCHEMA_VERSION
from . import config as C

PUBLIC_QUERIES = ("get_node_report", "find_common_recipients", "trace_paths_from_seeds", "get_cluster_report")
LIMITATIONS = [
    "Только наблюдаемые переводы; исходный остаток, внешние операции и подпороговые переводы неизвестны.",
    "Роли и scores — эвристики, не вероятности и не выводы о виновности.",
    "Направленный путь и совместимость дат не доказывают движение одних и тех же средств.",
]

class AnalysisService:
    def __init__(self,ctx):
        self.ctx=ctx
        self.G=ctx["G"]
        self.df=ctx["df"]
        self.manifest=ctx["manifest"]

    def _gid(self,gid):
        if not isinstance(gid,str) or not gid.isascii() or not gid.isdecimal() or str(int(gid))!=gid:
            raise ValueError("gid должен быть полной десятичной строкой")
        value=int(gid)
        if value not in self.df.index: raise ValueError(f"gid {gid} не найден")
        return value

    @staticmethod
    def _bound(value,low,high,name):
        if type(value) is not int or not low<=value<=high:
            raise ValueError(f"{name}: требуется целое {low}..{high}")
        return value

    def _envelope(self,kind,result,*,truncated=False,limits=None,highlight=None,limitations=None):
        result=clean(result)
        digest=hashlib.sha256(json.dumps(result,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()[:16]
        fid=f"{self.manifest['run_id']}:{kind}:{digest}"
        return {"schema_version":SCHEMA_VERSION,"run_id":self.manifest["run_id"],
                "data_hash":self.manifest["data_hash"],"facts":[{"id":fid,"kind":kind,"value":result}],
                "result":result,"evidence_refs":[fid],"limitations":LIMITATIONS+(limitations or []),
                "truncated":bool(truncated),"limits":limits or {},"highlight_gids":highlight or []}

    def query(self,action,args):
        if action not in PUBLIC_QUERIES: raise ValueError("Неизвестный аналитический запрос")
        if not isinstance(args,dict): raise ValueError("args должен быть объектом")
        return getattr(self,action)(**args)

    def get_node_report(self,gid: str) -> dict:
        g=self._gid(gid); r=self.df.loc[g]
        def edges(direction):
            values=self.G.in_edges(g,data=True) if direction=="in" else self.G.out_edges(g,data=True)
            return [{"src":str(u),"dst":str(v),"sum_kzt":d["sum_kzt"],"n_tx":d["n_tx"]}
                    for u,v,d in sorted(values,key=lambda e:(-e[2]["sum_kzt"],e[0],e[1]))[:20]]
        components=[{"metric":k,"normalized_value":float(r["c_"+k]),"weight":w,
                     "contribution":float(r["c_"+k])*w} for k,w in C.PRIORITY_WEIGHTS.items()]
        result={"gid":gid,"role":r.role,"role_score":r.role_score,"priority_score":r.priority_score,
                "rank":int(r["rank"]),"cluster_id":int(r.cluster_id),"is_seed":bool(r.is_seed),
                "depth":int(r.depth),"evidence":r.evidence,"why":r.why,
                "metrics":{k:r[k] for k in ("in_deg","out_deg","in_kzt","out_kzt","in_tx","out_tx","in_days","out_days",
                                          "pass_through","seed_upstream","seed_kzt_in","pagerank","betweenness")},
                "roles":[{"role":role,"eligible":bool(r["eligible_"+role]),"score":r["score_"+role]}
                         for role in C.ROLES if role!="peripheral"],
                "priority":{"components":components,"raw":r.priority_raw,"seed_factor":r.priority_seed_factor,
                            "normalizer":r.priority_normalizer,"isolated":bool(r.n_edges==0)},
                "temporal":{k:r[k] for k in ("fast_in_share","fast_status","fast_matched_kzt","fast_eligible_kzt",
                                           "fast_coverage","same_day_kzt")},
                "truncated_by_depth":bool(r.truncated),"model_forward_hint":r.p_forward if r.truncated else None,
                "incoming":edges("in"),"outgoing":edges("out"),
                "data_requests":self.ctx["requests"].loc[self.ctx["requests"].gid==g,["request","reason"]].to_dict("records")}
        result=clean(result)
        result["markdown"]=node_markdown(result,self.manifest["run_id"])
        return self._envelope("node",result,truncated=r.in_deg>20 or r.out_deg>20,
                              limits={"counterparties_per_direction":20,"seed_cutoff":4},highlight=[gid],
                              limitations=["Исходящие depth=4 не наблюдаются; модель — только подсказка."] if r.truncated else [])

    def find_common_recipients(self,gids: list[str],mode: str="direct",max_hops: int=1) -> dict:
        if not isinstance(gids,list) or not 2<=len(gids)<=20:
            raise ValueError("Выберите 2..20 gid")
        sources=list(dict.fromkeys(self._gid(g) for g in gids))
        if len(sources)<2: raise ValueError("Нужны минимум два разных gid")
        if mode not in ("direct","reachable"): raise ValueError("mode: direct или reachable")
        hops=self._bound(max_hops,1,4,"max_hops")
        if mode=="direct" and hops!=1: raise ValueError("Для direct max_hops=1")
        reach={}
        for s in sources:
            reach[s]={v:h for v,h in nx.single_source_shortest_path_length(self.G,s,cutoff=hops).items() if v!=s}
        common=set.intersection(*(set(v) for v in reach.values()))
        ordered=sorted(common,key=lambda g:(self.df.at[g,"rank"],g))
        rows=[]
        for g in ordered[:50]:
            rows.append({"gid":str(g),"role":self.df.at[g,"role"],"matched_sources":len(sources),
                         "total_sources":len(sources),"hops":{str(s):reach[s][g] for s in sources},
                         "direct_edges":[{"src":str(s),"dst":str(g),**self.G[s][g]} for s in sources if self.G.has_edge(s,g)]})
        return self._envelope("common_recipients",{"sources":[str(s) for s in sources],"mode":mode,
                    "match":"all_sources","total_results":len(ordered),"recipients":rows},
                    truncated=len(ordered)>50,limits={"max_hops":hops,"limit":50},
                    highlight=[r["gid"] for r in rows])

    def trace_paths_from_seeds(self,gid: str,max_hops: int=4,limit: int=10) -> dict:
        target=self._gid(gid)
        hops=self._bound(max_hops,1,4,"max_hops")
        limit=self._bound(limit,1,50,"limit")
        found=[]
        for seed in sorted(self.df.index[self.df.is_seed]):
            if seed==target: continue
            paths=nx.single_source_shortest_path(self.G,seed,cutoff=hops)
            if target not in paths: continue
            path=paths[target]
            found.append({"seed":str(seed),"gids":[str(g) for g in path],
                          "edges":[{"src":str(u),"dst":str(v),**self.G[u][v]} for u,v in zip(path,path[1:])]})
        found.sort(key=lambda p:(len(p["gids"]),int(p["seed"])))
        result=found[:limit]
        return self._envelope("seed_paths",{"gid":gid,"paths":result,"reachable_seed_count":len(found)},
                   truncated=len(found)>limit,limits={"max_hops":hops,"limit":limit,"one_shortest_path_per_seed":True},
                   highlight=list(dict.fromkeys(g for p in result for g in p["gids"])),
                   limitations=["Отсутствие пути в пределах cutoff не исключает более длинного пути. Суммы рёбер не складываются в доставленную сумму."])

    def get_cluster_report(self,cluster_id: int) -> dict:
        if type(cluster_id) is not int: raise ValueError("cluster_id должен быть целым")
        table=self.ctx["clusters"].set_index("cluster_id")
        if cluster_id not in table.index: raise ValueError("Кластер не найден")
        row=table.loc[cluster_id].to_dict()
        row["cluster_id"]=cluster_id
        row["top_gids"]=str(row["top_gids"]).split(";") if row["top_gids"] else []
        return self._envelope("cluster",row,highlight=row["top_gids"])

def node_markdown(r,run_id):
    lines=[f"# Узел {r['gid']}",f"Run: {run_id}",f"Роль: {r['role']} (соответствие правилу {r['role_score']})",
           f"Приоритет: {r['priority_score']}; место {r['rank']}",r["evidence"],r["why"],
           "## Наблюдаемые показатели"]
    lines += [f"- {k}: {v}" for k,v in r["metrics"].items()]
    lines += ["## Расчёт приоритета"]+[f"- {c['metric']}: {c['normalized_value']} × {c['weight']} = {c['contribution']}" for c in r["priority"]["components"]]
    p=r["priority"]
    lines += [f"Сумма {p['raw']} × seed-поправка {p['seed_factor']} / {p['normalizer']}; изолят: {p['isolated']}.",
              "## Время"]+[f"- {k}: {v}" for k,v in r["temporal"].items()]
    lines += ["## Ограничения"]+LIMITATIONS
    if r["truncated_by_depth"]: lines.append("Depth=4: исходящие неизвестны, модель не назначает роль.")
    return "\n\n".join(lines)+"\n"

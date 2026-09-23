"""Behavioral regression tests for observation limits, provenance and query semantics."""
import json
import math
from pathlib import Path
import pandas as pd
import pytest
from moneygraph import pipeline, load, features, truncation
from moneygraph.analysis_service import AnalysisService
from moneygraph.assistant import Assistant, _bounded_result

GIDS=[str(100000000000000001+i) for i in range(8)]

def tables():
    a,b,c,d,e,f,g,h=map(int,GIDS)
    nodes=pd.DataFrame({"gid":[a,b,c,d,e,f,g,h],"depth":[0,0,1,2,4,0,1,1],
                        "is_seed":[True,True,False,False,False,True,False,False]})
    tx=pd.DataFrame({"src":[a,b,c,d,a,b],"dst":[c,c,d,e,g,h],
                     "date":pd.to_datetime(["2026-01-30","2026-01-30","2026-01-31","2026-02-02","2026-02-01","2026-02-02"]),
                     "sum_kzt":[100000.,100000.,200000.,200000.,10000.,10000.]})
    edges=tx.groupby(["src","dst"],as_index=False).agg(sum_kzt=("sum_kzt","sum"),n_tx=("sum_kzt","size"))
    edges["depth"]=[int(nodes.set_index("gid").at[s,"depth"])+1 for s in edges.src]
    return edges,nodes,tx

def save_dataset(folder,edges,nodes,tx):
    folder.mkdir()
    for name,t in (("edges",edges),("nodes",nodes),("transactions",tx)):
        t.to_parquet(folder/f"{name}.parquet",index=False)

@pytest.fixture(scope="module")
def small(tmp_path_factory):
    base=tmp_path_factory.mktemp("small")
    source=base/"data"
    save_dataset(source,*tables())
    return pipeline.run(source,base/"out",log=lambda *a:None,period_end="2026-02-04"),base

def event(day,amount=100):
    return (pd.Timestamp(day),amount,1)

@pytest.mark.parametrize("outday,ratio",[
    ("2026-01-29",0),("2026-01-30",0),("2026-01-31",1),("2026-02-01",1),("2026-02-02",0)])
def test_fifo_calendar_order(outday,ratio):
    r=features._fast_forward([event("2026-01-30")],[event(outday)],pd.Timestamp("2026-02-05"))
    assert r["fast_in_share"]==ratio

def test_censoring_and_capacity():
    r=features._fast_forward([event("2026-01-01"),event("2026-01-31")],
                             [event("2026-01-02")],pd.Timestamp("2026-01-31"))
    assert r["fast_in_share"]==1 and r["fast_coverage"]==.5
    r=features._fast_forward([event("2026-01-01",70),event("2026-01-01",80)],
                             [event("2026-01-02",100)],pd.Timestamp("2026-01-04"))
    assert r["fast_matched_kzt"]==100 and r["fast_in_share"]==pytest.approx(2/3)

def test_unavailable_and_same_day():
    r=features._fast_forward([event("2026-01-31")],[event("2026-01-31")],pd.Timestamp("2026-01-31"))
    assert math.isnan(r["fast_in_share"]) and r["same_day_kzt"]==100
    r=features._fast_forward([event("2026-01-01")],[],pd.Timestamp("2026-01-31"),False)
    assert r["fast_status"]=="outgoing_unobserved" and math.isnan(r["fast_matched_kzt"])

def test_cycle_strict_dates():
    pairs=[("a","b"),("b","a")]
    assert not features._time_consistent(pairs,{pairs[0]:[pd.Timestamp("2026-01-01")],pairs[1]:[pd.Timestamp("2026-01-01")]})
    assert features._time_consistent(pairs,{pairs[0]:[pd.Timestamp("2026-01-31")],pairs[1]:[pd.Timestamp("2026-02-01")]})

def test_incoming_burst_calendar_baseline():
    incoming=[event("2026-01-31")]*3+[event("2026-02-01")]
    result=features._incoming_burst(incoming,pd.Timestamp("2026-01-30"),pd.Timestamp("2026-02-02"))
    assert result["burst_observation_days"]==4
    assert result["burst_in_max_tx"]==3 and result["burst_in_day"]=="2026-01-31"
    assert result["burst_in_mean_daily_tx"]==1 and result["burst_in_ratio"]==3
    assert result["burst_in_flag"] is True
    assert not features._incoming_burst(incoming,pd.Timestamp("2026-01-31"),pd.Timestamp("2026-02-01"))["burst_in_flag"]
    assert not features._incoming_burst(incoming[:2],pd.Timestamp("2026-01-01"),pd.Timestamp("2026-02-01"))["burst_in_flag"]
    quiet=features._incoming_burst([],pd.NaT,pd.NaT)
    assert quiet["burst_observation_days"]==0 and quiet["burst_in_day"]==""
    assert not quiet["burst_in_flag"] and math.isnan(quiet["burst_in_ratio"])

def test_incoming_burst_tie_and_uniform():
    incoming=[event("2026-02-02")]*3+[event("2026-02-01")]*3
    assert features._incoming_burst(incoming,pd.Timestamp("2026-02-01"),pd.Timestamp("2026-02-06"))["burst_in_day"]=="2026-02-01"
    uniform=[event(f"2026-02-0{d}") for d in range(1,5)]
    assert not features._incoming_burst(uniform,pd.Timestamp("2026-02-01"),pd.Timestamp("2026-02-04"))["burst_in_flag"]

def test_repeated_routes_require_distinct_input_days():
    a,b,c=1,2,3
    def metrics(dates):
        tx=pd.DataFrame({"src":[a,a,b,b],"dst":[b,b,c,c],"date":pd.to_datetime(dates),"sum_kzt":[10.,20.,10.,20.]})
        return features.temporal(tx,pd.DataFrame({"out_observed":[True]},index=[b]),"2026-02-05").loc[b]
    assert metrics(["2026-01-30","2026-02-01","2026-01-31","2026-02-02"]).repeated_routes==1
    assert metrics(["2026-01-30","2026-01-30","2026-01-31","2026-02-02"]).repeated_routes==0

@pytest.mark.parametrize("case",["duplicate_gid","duplicate_edge","endpoint","sum","count","nan","date","float_gid"])
def test_input_rejections(case):
    edges,nodes,tx=tables()
    if case=="duplicate_gid": nodes=pd.concat([nodes,nodes.iloc[:1]],ignore_index=True)
    if case=="duplicate_edge": edges=pd.concat([edges,edges.iloc[:1]],ignore_index=True)
    if case=="endpoint": edges.loc[0,"dst"]=9
    if case=="sum": edges.loc[0,"sum_kzt"]+=100
    if case=="count": edges.loc[0,"n_tx"]+=1
    if case=="nan": tx.loc[0,"sum_kzt"]=float("nan")
    if case=="date": tx.loc[0,"date"]=pd.NaT
    if case=="float_gid": nodes["gid"]=nodes.gid.astype(float)
    with pytest.raises(ValueError): load.sanity_check(edges,nodes,tx)

def test_reports_and_rules(small):
    ctx,_=small
    d=ctx["df"]
    assert ctx["trunc"]["status"]=="unavailable"
    assert not (d[d.truncated].role=="terminal").any()
    assert not d[d.is_seed].role.isin(["terminal","transit"]).any()
    iso=d[d.n_edges==0]
    assert (iso.priority_score==0).all() and (iso.role_score==0).all()
    service=AnalysisService(ctx)
    r=service.get_node_report(GIDS[2])
    assert r["run_id"]==ctx["manifest"]["run_id"]
    assert r["result"]["gid"]==GIDS[2]
    p=r["result"]["priority"]
    raw=sum(c["contribution"] for c in p["components"])
    assert raw==pytest.approx(p["raw"])
    assert round(raw*p["seed_factor"]/p["normalizer"],4)==r["result"]["priority_score"]
    assert r["result"]["patterns"]["sync_payers_max"]==int(d.loc[int(GIDS[2]),"sync_payers_max"])
    assert r["result"]["temporal"]["burst_in_max_tx"]==int(d.loc[int(GIDS[2]),"burst_in_max_tx"])
    assert "burst_in_ratio" in r["result"]["markdown"]
    json.dumps(r,allow_nan=False)

def test_common_and_paths(small):
    ctx,_=small;service=AnalysisService(ctx)
    r=service.find_common_recipients(GIDS[:2])
    assert [n["gid"] for n in r["result"]["recipients"]]==[GIDS[2]]
    r=service.find_common_recipients(GIDS[:2],mode="reachable",max_hops=3)
    assert GIDS[4] in [n["gid"] for n in r["result"]["recipients"]]
    r=service.trace_paths_from_seeds(GIDS[4],max_hops=3,limit=1)
    assert r["truncated"] and r["result"]["reachable_seed_count"]==2
    for path in r["result"]["paths"]:
        assert path["gids"][-1]==GIDS[4]
        assert all(ctx["G"].has_edge(int(e["src"]),int(e["dst"])) for e in path["edges"])
    assert not service.trace_paths_from_seeds(GIDS[0])["result"]["paths"]
    assert not service.trace_paths_from_seeds(GIDS[4],max_hops=1)["result"]["paths"]

@pytest.mark.parametrize("action,args",[
    ("get_node_report",{"gid":100000000000000001}),
    ("get_node_report",{"gid":"9"}),
    ("find_common_recipients",{"gids":[GIDS[0]]*2}),
    ("find_common_recipients",{"gids":GIDS*3}),
    ("find_common_recipients",{"gids":GIDS[:2],"mode":"direct","max_hops":4}),
    ("trace_paths_from_seeds",{"gid":GIDS[2],"max_hops":5}),
    ("trace_paths_from_seeds",{"gid":GIDS[2],"limit":0}),
    ("get_cluster_report",{"cluster_id":999999}),
    ("__init__",{}),
])
def test_service_invalid(small,action,args):
    with pytest.raises(ValueError): AnalysisService(small[0]).query(action,args)

def test_edgeless_and_self_loop(tmp_path):
    for self_loop in (False,True):
        edges,nodes,tx=tables()
        nodes=nodes.iloc[:1].copy()
        if self_loop:
            tx=tx.iloc[:1].copy();tx["src"]=nodes.gid.iloc[0];tx["dst"]=nodes.gid.iloc[0]
            edges=tx.groupby(["src","dst"],as_index=False).agg(sum_kzt=("sum_kzt","sum"),n_tx=("sum_kzt","size"));edges["depth"]=1
        else:
            edges=edges.iloc[:0];tx=tx.iloc[:0]
        path=tmp_path/str(self_loop)
        save_dataset(path,edges,nodes,tx)
        ctx=pipeline.run(path,tmp_path/("out"+str(self_loop)),log=lambda *a:None)
        assert len(ctx["df"])==1 and all(ok for ok,_ in ctx["checks"])

def test_failed_publication_keeps_previous(small,monkeypatch):
    ctx,base=small
    previous=(base/"out"/"nodes_roles.csv").read_bytes()
    monkeypatch.setattr(pipeline,"compute",lambda *a,**k:ctx)
    def fail(*a,**k): raise RuntimeError("simulated write failure")
    monkeypatch.setattr(pipeline.viewer,"build",fail)
    with pytest.raises(RuntimeError): pipeline.run(base/"data",base/"out")
    assert (base/"out"/"nodes_roles.csv").read_bytes()==previous
    assert not (base/".out.lock").exists()

def test_repeatability(small,tmp_path):
    ctx,base=small
    other=pipeline.run(base/"data",tmp_path/"out",log=lambda *a:None,period_end="2026-02-04")
    assert ctx["manifest"]["run_id"]==other["manifest"]["run_id"]
    for name in ("nodes_roles.csv","top_nodes.csv","clusters.csv"):
        assert (base/"out"/name).read_bytes()==(tmp_path/"out"/name).read_bytes()

def test_ai_disabled_and_allowlist(small,monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    bot=Assistant(small[0])
    monkeypatch.setattr(bot,"_post",lambda *a:pytest.fail("Network invoked in offline mode"))
    assert bot.ask("Топ-5")["mode"].startswith("офлайн")
    assert "error" in bot._call("__dict__",{})
    assert "error" in bot._call("trace",{"gid":GIDS[0],"max_hops":999})
    assert "error" in bot._call("get_node_report",{"gid":GIDS[0],"shell":"pwd"})

def test_offline_common_all_sources_by_default(small,monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    bot=Assistant(small[0])
    question="Кто собирает деньги с "+", ".join(GIDS[:2])+"?"
    result=bot.ask(question)
    assert result["tools"]==["find_common_recipients"]
    fact=result["facts"][0]
    assert fact["kind"]=="common_recipients" and fact["value"]["mode"]=="direct"
    assert fact["value"]["match"]=="all_sources"
    assert [n["gid"] for n in fact["value"]["recipients"]]==[GIDS[2]]
    assert result["fact_ids"]==result["audit"][0]["evidence_refs"]
    # A and B share C, but isolated F has no outgoing edges: all three must match.
    result=bot.ask("Кто собирает деньги с "+", ".join([*GIDS[:2],GIDS[5]]))
    assert result["facts"][0]["value"]["recipients"]==[]
    assert result["focus"] is None

def test_offline_common_opt_in_modes(small,monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    bot=Assistant(small[0])
    result=bot.ask("Кто достижим через 3 шага от "+", ".join(GIDS[:2]))
    assert result["facts"][0]["value"]["mode"]=="reachable"
    assert GIDS[4] in [n["gid"] for n in result["facts"][0]["value"]["recipients"]]
    result=bot.ask("Кто собирает хотя бы от двух из "+", ".join([*GIDS[:2],GIDS[5]]))
    assert result["tools"]==["common_downstream"]
    assert result["facts"][0]["value"]["match"]=="at_least_two_sources"
    assert "2 из 3" in result["answer"]

def test_offline_five_sources_excludes_partial_match(small,monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    ctx=dict(small[0]);ctx["G"]=ctx["G"].copy()
    sources=[GIDS[i] for i in (0,1,3,5,6)]
    for source in sources:
        ctx["G"].add_edge(int(source),int(GIDS[2]),sum_kzt=10000.,n_tx=1)
    for source in sources[:4]:
        ctx["G"].add_edge(int(source),int(GIDS[7]),sum_kzt=10000.,n_tx=1)
    result=Assistant(ctx).ask("Кто собирает деньги с этих пяти: "+", ".join(sources))
    recipients=result["facts"][0]["value"]["recipients"]
    assert [row["gid"] for row in recipients]==[GIDS[2]]
    assert recipients[0]["matched_sources"]==5
    assert "5 из 5" in result["answer"]

@pytest.mark.parametrize("gids,suffix,error",[
    ([GIDS[0],"999999999999999999"],"","Неизвестный gid"),
    ([GIDS[0]]*2,"","Нужны минимум два разных gid"),
    (GIDS*3,"","не более 20"),
    (GIDS[:2]," через 5 шагов","от 1 до 4"),
])
def test_offline_common_invalid_sources(small,monkeypatch,gids,suffix,error):
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    result=Assistant(small[0]).ask("Кто собирает деньги с "+", ".join(gids)+suffix)
    assert error in result["answer"]
    assert result["facts"][0]["kind"]=="input_validation"
    assert not result["tools"]

@pytest.mark.parametrize("failure",["unknown_fact","unknown_gid","malformed","timeout","disconnect","budget","injection","response_shape","tool_shape"])
def test_ai_failure_fallback(small,monkeypatch,failure):
    import http.client
    monkeypatch.setenv("OPENAI_API_KEY","fake");monkeypatch.setenv("AI_ENABLED","true")
    bot=Assistant(small[0]);calls=[]
    def fake(payload):
        calls.append(payload)
        if failure=="timeout": raise TimeoutError()
        if failure=="disconnect": raise http.client.RemoteDisconnected()
        if failure=="response_shape": return {"choices":[{"message":[]}]}
        if failure=="tool_shape": return {"choices":[{"message":{"tool_calls":[None]}}]}
        if len(calls)==1 or failure=="budget":
            tc={"id":"c"+str(len(calls)),"type":"function","function":{"name":"get_node_report","arguments":json.dumps({"gid":GIDS[2]})}}
            if failure=="injection": tc["function"]={"name":"__dict__","arguments":"{}"}
            return {"choices":[{"message":{"tool_calls":[tc]* (7 if failure=="budget" else 1),"content":None}}]}
        fact=[m for m in payload["messages"] if m["role"]=="tool"][-1]
        refs=json.loads(fact["content"])["evidence_refs"]
        text=json.dumps({"answer":"Проверка","fact_ids":["invented"] if failure in ("unknown_fact","injection") else refs,
                         "highlight_gids":["999"] if failure=="unknown_gid" else []})
        return {"choices":[{"message":{"content":"bad json" if failure=="malformed" else text}}]}
    monkeypatch.setattr(bot,"_post",fake)
    result=bot.ask("Расскажи про "+GIDS[2])
    assert result["mode"].startswith("офлайн")
    assert len(calls)<=3
    assert result["run_id"]==small[0]["manifest"]["run_id"]
    assert result["audit"] and result["facts"]
    assert all("result" in entry for entry in result["audit"])

def test_bounded_payload_valid_json():
    result={"facts":[{"id":"f","value":list(range(100))}],"evidence_refs":["f"],"truncated":False}
    bounded=_bounded_result(result)
    assert bounded["truncated"]
    assert len(json.loads(json.dumps(bounded))["facts"][0]["value"])==10


def test_http_contract(small,monkeypatch):
    import threading,urllib.request,urllib.error
    from http.server import ThreadingHTTPServer
    from serve import make_handler
    ctx,base=small
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    srv=ThreadingHTTPServer(("127.0.0.1",0),make_handler(ctx,base/"out"))
    worker=threading.Thread(target=srv.serve_forever,daemon=True);worker.start()
    url=f"http://127.0.0.1:{srv.server_port}"
    def post(body,path="/api/query"):
        req=urllib.request.Request(url+path,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=3) as response:
            return json.load(response)
    try:
        with urllib.request.urlopen(url+"/api/health",timeout=3) as response:
            assert json.load(response)["llm"] is False
        result=post({"action":"get_node_report","args":{"gid":GIDS[2]},"run_id":ctx["manifest"]["run_id"]})
        assert result["result"]["gid"]==GIDS[2]
        with urllib.request.urlopen(url+"/download/nodes_roles.csv",timeout=3) as response:
            assert response.headers["Content-Disposition"].startswith("attachment")
            assert response.read()==(base/"out"/"nodes_roles.csv").read_bytes()
        with urllib.request.urlopen(url+"/api/node-report/"+GIDS[2]+".md",timeout=3) as response:
            assert "attachment" in response.headers["Content-Disposition"]
            assert GIDS[2] in response.read().decode()
        for payload,status in [
            ({"action":"get_node_report","args":{"gid":"unknown"}},400),
            ({"action":"get_node_report","args":{"gid":GIDS[2]},"run_id":"old"},409),
            ([],400)]:
            with pytest.raises(urllib.error.HTTPError) as error: post(payload)
            assert error.value.code==status
        assert post({"question":"Топ-5"},"/api/ask")["mode"].startswith("офлайн")
        with urllib.request.urlopen(url+"/nodes_roles.csv",timeout=3) as response:
            assert response.read()==(base/"out"/"nodes_roles.csv").read_bytes()
    finally:
        srv.shutdown();srv.server_close();worker.join(timeout=3)

def test_no_optional_model_dependency(small,monkeypatch):
    ctx,_=small
    def unavailable(*a,**k): raise ImportError("optional sklearn absent")
    monkeypatch.setattr(truncation,"_fit_predict",unavailable)
    df,status=truncation.fit_predict(ctx["G"],ctx["df"].copy())
    assert status["status"]=="unavailable"
    assert df.loc[df.truncated,"p_forward"].isna().all()

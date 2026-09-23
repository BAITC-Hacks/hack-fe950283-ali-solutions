"""Compute a reproducible run, validate it, then publish its complete directory."""
import hashlib
import importlib.metadata
import json
import os
import platform
import math
import shutil
import tempfile
import time
from pathlib import Path
import pandas as pd
from . import clusters, config as C, export, features, load, priority, report, resilience, roles, truncation, viewer
from .contracts import clean

def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def compute(data_dir: Path, log=print, period_end=None) -> dict:
    started = time.perf_counter()
    if any(not math.isfinite(w) or w < 0 for w in C.PRIORITY_WEIGHTS.values()) or not math.isclose(sum(C.PRIORITY_WEIGHTS.values()), 1):
        raise ValueError("Веса приоритета должны быть конечными, неотрицательными и давать сумму 1")
    edges,nodes,tx = load.load(data_dir)
    stats = load.sanity_check(edges,nodes,tx)
    end = pd.Timestamp(period_end).normalize() if period_end is not None else tx.date.max()
    if period_end is not None and (end.tzinfo is not None or pd.isna(end) or (len(tx) and end < tx.date.max())):
        raise ValueError("Конец наблюдения должен быть календарной датой не раньше последней операции")
    stats["period_end"] = None if pd.isna(end) else end.date().isoformat()
    stats["period_end_source"] = "explicit" if period_end is not None else "last_observed_transaction"
    if period_end is None:
        stats["warnings"].append("Конец наблюдения принят равным последней операции; можно задать --period-end.")
    G = load.build_graph(edges,nodes)
    df,cyc = features.compute_all(G,nodes,tx,end)
    df,trunc = truncation.fit_predict(G,df)
    df=priority.score(roles.assign(G,df))
    df,stab,modularity=clusters.detect(G,df)
    cl=clusters.summarize(G,df,stab)
    res=resilience.simulate(G,df)
    requests=report.data_requests(df)
    hashes={name:_hash(data_dir/name) for name in ("nodes.parquet","edges.parquet","transactions.parquet")}
    config={k:v for k,v in vars(C).items() if k.isupper()}
    config["period_end"]=stats["period_end"]
    data_hash=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    versions={}
    for name in ("pandas","numpy","pyarrow","networkx","scipy","scikit-learn"):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    code_hash=hashlib.sha256(b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
                           + b"".join(p.read_bytes() for p in sorted((viewer.ROOT/"viewer").rglob("*")) if p.suffix in (".html",".js",".css",".woff2"))).hexdigest()
    run_id=hashlib.sha256((data_hash+config_hash+code_hash+json.dumps(versions,sort_keys=True)).encode()).hexdigest()[:20]
    manifest={"schema_version":C.SCHEMA_VERSION,"run_id":run_id,"data_hash":data_hash,"input_hashes":hashes,
              "config_hash":config_hash,"code_hash":code_hash,"config":config,"versions":versions,
              "python":platform.python_version(),"platform":platform.platform(),"cpu":platform.processor(),
              "stats":stats,"algorithms":{"pagerank":"directed sum_kzt","betweenness":"directed exact unweighted",
              "community":"Louvain; reciprocal log1p(amount/5000) weights summed","seed_cutoff":4,
              "temporal":"FIFO 1–2 calendar days, full-window incoming only"},
              "optional":{"truncation":trunc,"cycles_truncated":G.graph.get("cycles_truncated",False)},
              "timings":{"compute_seconds":time.perf_counter()-started}}
    ctx={"G":G,"df":df,"clusters":cl,"stats":stats,"trunc":trunc,"cycles":cyc,"tx":tx,
         "modularity":modularity,"resilience":res,"requests":requests,"nodes":nodes,"edges":edges,
         "manifest":clean(manifest)}
    log(f"  {len(df)} узлов; {len(cl)} кластеров; модель: {trunc['status']}")
    return ctx

def _write(out,ctx):
    export.write(out,ctx["df"],ctx["clusters"],{"resilience.csv":ctx["resilience"],"data_requests.csv":ctx["requests"]})
    report.write_report(out/"report.md",ctx)
    viewer.build(out,ctx["G"],ctx)
    checks=export.validate(out,ctx["stats"]["n_nodes"],set(ctx["nodes"].gid))
    ctx["checks"]=checks
    failed=[text for ok,text in checks if not ok]
    if failed:
        raise ValueError("Выгрузки не прошли проверку: "+"; ".join(failed))

def run(data_dir: Path,out_dir: Path,log=print,period_end=None):
    started=time.perf_counter()
    out=out_dir.resolve()
    data=data_dir.resolve()
    if out==data or out==Path(__file__).resolve().parent.parent or out in data.parents:
        raise ValueError("Папка результатов не должна заменять исходные данные или проект")
    out.parent.mkdir(parents=True,exist_ok=True)
    lock=out.parent/f".{out.name}.lock"
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Для {out} уже выполняется запись (lock: {lock})") from exc
    os.close(fd)
    try:
        ctx=compute(data,log,period_end)
        with tempfile.TemporaryDirectory(prefix=f".{out.name}.stage-",dir=out.parent) as tmp:
            stage=Path(tmp)
            _write(stage,ctx)
            ctx["manifest"]["timings"]["pipeline_seconds"]=time.perf_counter()-started
            (stage/"run_manifest.json").write_text(json.dumps(clean(ctx["manifest"]),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
            (stage/"data_quality.json").write_text(json.dumps(ctx["stats"],ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
            # A failed publication restores the previous complete run. No files are replaced piecemeal.
            backup=Path(tempfile.mkdtemp(prefix=f".{out.name}.previous-",dir=out.parent))
            backup.rmdir()
            had_old=out.exists()
            try:
                if had_old: out.rename(backup)
                stage.rename(out)
            except Exception:
                if had_old and backup.exists() and not out.exists(): backup.rename(out)
                raise
            finally:
                if backup.exists() and out.exists():
                    if backup.parent != out.parent or not backup.name.startswith(f".{out.name}.previous-"):
                        raise RuntimeError("Unexpected backup path")
                    shutil.rmtree(backup)
        return ctx
    finally:
        lock.unlink(missing_ok=True)

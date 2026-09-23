"""Проверки must-have из ТЗ и формальности правил ролей: python -m pytest -q"""
import re
import time
from pathlib import Path

import pandas as pd
import pytest

from moneygraph import config as C
from moneygraph import pipeline

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    t0 = time.time()
    ctx = pipeline.run(ROOT / "data", out, log=lambda *a: None)
    ctx["elapsed"] = time.time() - t0
    ctx["out"] = out
    return ctx


def test_must_have_checks_pass(run):
    failed = [text for ok, text in run["checks"] if not ok]
    assert not failed, failed


def test_runtime_under_5_minutes(run):
    assert run["elapsed"] < 300


def test_all_outputs_written(run):
    for f in ("nodes_roles.csv", "clusters.csv", "top_nodes.csv", "index.html", "report.md"):
        assert (run["out"] / f).stat().st_size > 0


def test_roles_follow_documented_rules(run):
    d = pd.read_csv(run["out"] / "nodes_roles.csv")
    co = d[d.role == "coordinator"]
    assert ((co.in_deg >= C.HUB_MIN_PAYERS) & (co.out_deg >= C.HUB_MIN_RECIPIENTS) & (co.key_links >= C.COORD_MIN_KEY_LINKS)).all()
    di = d[d.role == "distributor"]
    assert (di.out_deg >= C.DISTR_MIN_RECIPIENTS).all()
    tr = d[(d.role == "transit") & ~d.is_seed]
    assert tr.pass_through.between(*C.TRANSIT_FAST_PT).all() and (tr.depth < C.MAX_DEPTH).all()
    te = d[d.role == "terminal"]
    observed = te[~te.truncated]
    assert (observed.pass_through <= C.TERM_MAX_PT).all()
    assert not te.truncated.any() and not te.is_seed.any()


def test_truncation_artifact_not_naive(run):
    """Наивное «out_deg=0 ⇒ сток» дало бы 444 терминала на 4-м колене — у нас их заметно меньше."""
    d = pd.read_csv(run["out"] / "nodes_roles.csv")
    assert d.truncated.sum() == 444
    assert (d[d.truncated].role == "terminal").sum() == 0
    assert run["trunc"]["status"] in ("available", "unavailable")
    if run["trunc"]["status"] == "available":
        assert 0 <= run["trunc"]["cv_auc"] <= 1


def test_no_hardcoded_gids():
    for p in (ROOT / "moneygraph").glob("*.py"):
        assert not re.search(r"\b1000000\d{11}\b", p.read_text(encoding="utf-8")), f"gid в коде: {p.name}"


def test_evidence_explains_with_numbers(run):
    top = pd.read_csv(run["out"] / "top_nodes.csv")
    assert top.why.str.contains(r"\d").all()
    assert len(top) >= 20


def test_viewer_bundles_assets_and_exact_facts(run):
    import json
    html = (run["out"] / "index.html").read_text(encoding="utf-8")
    data, _ = json.JSONDecoder().raw_decode(html.split("const DATA = ", 1)[1])
    for n in data["nodes"]:
        row = run["df"].loc[int(n["id"])]
        assert n["ink"] == row.in_kzt and n["outk"] == row.out_kzt
        assert n["rs"] == row.role_score and n["p"] == row.priority_score
    assert all(marker not in html for marker in ("/*__THEME__*/", "/*__WORKSPACE__*/", "/*__ADDITIONS__*/", "/*__ICONS__*/", "__INTER_WOFF2__"))
    assert 'id="closePanel"' in html and 'id="mobileFilters"' in html
    assert "AbortController" in html and "aria-label" in html
    assert not re.search(r'<script[^>]+src=["\']https?://', html)
    assert 'src:url(data:font/woff2;base64,' in html
    assert 'id="assistantDock" role="dialog"' in html
    assert 'id="aiLauncher"' in html and 'id="economyMode"' in html
    assert 'prefers-reduced-motion' in html
    assert 'egoCy.destroy()' not in html
    assert 'pixelRatio:1' in html
    assert all('burst' in n and 'cycles' in n for n in data['nodes'])

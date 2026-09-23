"""Проверки must-have из ТЗ и формальности правил ролей: python -m pytest -q"""
import re
import json
import shutil
import time
from pathlib import Path

import pandas as pd
import pytest

from moneygraph import config as C
from moneygraph import pipeline
from moneygraph import export, validation

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
    d = pd.read_csv(run["out"] / "nodes_features.csv")
    co = d[d.role == "coordinator"]
    assert ((co.in_deg >= C.HUB_MIN_PAYERS) & (co.out_deg >= C.HUB_MIN_RECIPIENTS) & (co.key_links >= C.COORD_MIN_KEY_LINKS)).all()
    di = d[d.role == "distributor"]
    assert (di.out_deg >= C.DISTR_MIN_RECIPIENTS).all()
    tr = d[(d.role == "transit") & ~d.is_seed]
    assert tr.pass_through.between(*C.TRANSIT_FAST_PT).all() and (tr.depth < C.MAX_DEPTH).all()
    te = d[d.role == "terminal"]
    observed = te[~te.truncated]
    assert (observed.pass_through <= C.TERM_MAX_PT).all()
    assert (1 - te[te.truncated].p_forward >= C.TRUNC_TERMINAL_P).all()


def test_truncation_artifact_not_naive(run):
    """Наивное «out_deg=0 ⇒ сток» дало бы 444 терминала на 4-м колене — у нас их заметно меньше."""
    d = pd.read_csv(run["out"] / "nodes_features.csv")
    assert d.truncated.sum() == 444
    assert (d[d.truncated].role == "terminal").sum() < 444 * 0.5
    assert run["trunc"]["cv_auc"] > 0.6


def test_no_hardcoded_gids():
    for p in (ROOT / "moneygraph").glob("*.py"):
        assert not re.search(r"\b1000000\d{11}\b", p.read_text(encoding="utf-8")), f"gid в коде: {p.name}"


def test_evidence_explains_with_numbers(run):
    top = pd.read_csv(run["out"] / "top_nodes.csv")
    assert top.why.str.contains(r"\d").all()
    assert len(top) >= 20


def test_exact_csv_contracts(run):
    for name, columns in (("nodes_roles.csv", export.NODE_COLS), ("clusters.csv", export.CLUSTER_COLS), ("top_nodes.csv", export.TOP_COLS)):
        table = pd.read_csv(run["out"] / name)
        assert list(table.columns) == columns


def test_truncated_terminal_confidence_is_discounted(run):
    nodes = run["df"]
    terminals = nodes[(nodes.role == "terminal") & nodes.truncated]
    assert not terminals.empty
    assert (terminals.role_score <= 1 - terminals.p_forward + .0005).all()


def test_validation_manifest_matches_actual_files(run):
    manifest = json.loads((run["out"] / "validation.json").read_text())
    assert manifest["passed"] and manifest["elapsed_seconds"] < 300
    assert manifest["stats"]["n_nodes"] == 2248
    for name, digest in manifest["input_sha256"].items():
        assert validation.sha256(ROOT / "data" / name) == digest
    for name, digest in manifest["output_sha256"].items():
        assert validation.sha256(run["out"] / name) == digest


@pytest.mark.parametrize("corruption", ["missing_column", "wrong_role", "wrong_cluster_size", "duplicate_top"])
def test_export_validation_rejects_inconsistent_outputs(run, tmp_path, corruption):
    for name in ("nodes_roles.csv", "clusters.csv", "top_nodes.csv"):
        shutil.copy(run["out"] / name, tmp_path / name)
    if corruption == "missing_column":
        filename = "nodes_roles.csv"
        table = pd.read_csv(tmp_path / filename).drop(columns="role")
    elif corruption == "wrong_cluster_size":
        filename = "clusters.csv"
        table = pd.read_csv(tmp_path / filename)
        table.loc[0, "n_nodes"] += 1
    else:
        filename = "top_nodes.csv"
        table = pd.read_csv(tmp_path / filename)
        if corruption == "wrong_role":
            table.loc[0, "role"] = "peripheral"
        else:
            table.loc[1, "gid"] = table.loc[0, "gid"]
    table.to_csv(tmp_path / filename, index=False)
    checks = export.validate(tmp_path, len(run["df"]), set(run["df"].index))
    assert any(not passed for passed, _ in checks)


def test_csv_results_ignore_input_row_order(run, tmp_path):
    data_dir = tmp_path / "shuffled"
    data_dir.mkdir()
    for name in ("nodes", "edges", "transactions"):
        original = pd.read_parquet(ROOT / "data" / f"{name}.parquet")
        original.sample(frac=1, random_state=17).to_parquet(data_dir / f"{name}.parquet", index=False)
    other = pipeline.run(data_dir, tmp_path / "out", log=lambda *args: None)
    for filename in ("nodes_roles.csv", "clusters.csv", "top_nodes.csv", "nodes_features.csv", "resilience.csv"):
        assert (run["out"] / filename).read_bytes() == (tmp_path / "out" / filename).read_bytes()
    assert all(passed for passed, _ in other["checks"])

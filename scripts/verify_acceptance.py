"""Bounded acceptance run: tests, two fresh processes, deterministic artifacts."""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_FILES = ("nodes_roles.csv", "clusters.csv", "top_nodes.csv", "data_requests.csv", "resilience.csv")


def command(args, timeout):
    started = time.perf_counter()
    try:
        result = subprocess.run([sys.executable, "-X", "utf8", *args], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8", timeout=timeout)
        return {"status": "passed" if result.returncode == 0 else "failed",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "returncode": result.returncode, "output_tail": (result.stdout + result.stderr)[-4000:]}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "elapsed_seconds": round(time.perf_counter() - started, 3),
                "reason": f"Stopped after {timeout}s; not counted as passed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/acceptance.json")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 300:
        parser.error("--timeout must be 1..300 seconds")
    report = {"tests": command(["-m", "pytest", "-q"], min(120, args.timeout)), "runs": []}
    print("Tests:", report["tests"]["status"], flush=True)
    with tempfile.TemporaryDirectory(prefix="moneygraph-acceptance-") as tmp:
        paths = [Path(tmp) / name for name in ("a", "b")]
        for path in paths:
            outcome = command(["run.py", "--out", str(path)], args.timeout)
            if outcome["status"] == "passed":
                manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
                outcome.update(run_id=manifest["run_id"], timings=manifest["timings"])
            report["runs"].append(outcome)
            print("Run", path.name, outcome["status"], outcome["elapsed_seconds"], flush=True)
        if all(r["status"] == "passed" for r in report["runs"]):
            report["csv"] = {}
            for name in CSV_FILES:
                digests = [hashlib.sha256((p / name).read_bytes()).hexdigest() for p in paths]
                report["csv"][name] = {"identical": digests[0] == digests[1], "sha256": digests[0]}
            report["same_run_id"] = report["runs"][0]["run_id"] == report["runs"][1]["run_id"]
            report["under_300_seconds"] = all(r["elapsed_seconds"] < 300 for r in report["runs"])
    report["passed"] = (report["tests"]["status"] == "passed" and report.get("same_run_id", False)
                        and report.get("under_300_seconds", False)
                        and all(c["identical"] for c in report.get("csv", {}).values()))
    target = ROOT / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Acceptance:", report["passed"], target, flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

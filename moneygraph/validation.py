"""Машиночитаемый паспорт воспроизводимости пересчёта."""
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from . import config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_fingerprint(data_dir: Path) -> dict:
    return {path.name: sha256(path) for path in sorted(data_dir.glob("*.parquet"))}


def write(out_dir: Path, ctx: dict, elapsed: float) -> None:
    checks = [{"passed": bool(passed), "description": description} for passed, description in ctx["checks"]]
    checks.append({"passed": elapsed <= 300, "description": "полный пересчёт не более 300 секунд"})
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": all(check["passed"] for check in checks),
        "elapsed_seconds": round(elapsed, 3),
        "stats": ctx["stats"],
        "input_sha256": ctx["input_sha256"],
        "output_sha256": {path.name: sha256(path) for path in sorted(out_dir.iterdir()) if path.is_file() and path.suffix in (".csv", ".html", ".md")},
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("pandas", "pyarrow", "networkx", "numpy", "scipy", "scikit-learn")},
        "parameters": {name: value for name, value in vars(config).items() if name.isupper()},
        "checks": checks,
    }
    (out_dir / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

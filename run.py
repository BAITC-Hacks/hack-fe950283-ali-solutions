#!/usr/bin/env python3
"""Граф денег — один запуск от сырых parquet до всех выгрузок и экрана просмотра.

    python run.py                      # data/ → out/
    python run.py --data data --out out --open
"""
import argparse
import sys
import time
import webbrowser
from pathlib import Path

from moneygraph import pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data", help="папка с edges/nodes/transactions.parquet")
    ap.add_argument("--out", default="out", help="куда писать выгрузки")
    ap.add_argument("--open", action="store_true", help="открыть экран просмотра в браузере")
    a = ap.parse_args()

    t0 = time.time()
    print("Граф денег: полный пересчёт")
    try:
        ctx = pipeline.run(Path(a.data), Path(a.out))
    except (ValueError, OSError) as error:
        print(f"Ошибка входных данных или выгрузки: {error}", file=sys.stderr)
        return 1
    print("\nПроверка выгрузок по ТЗ:")
    ok = True
    for passed, text in ctx["checks"]:
        print(f"  {'OK ' if passed else 'ERR'} {text}")
        ok &= bool(passed)
    out = Path(a.out)
    print(f"\nГотово за {time.time() - t0:.1f} с. Файлы в {out}/:")
    for f in ("nodes_roles.csv", "clusters.csv", "top_nodes.csv", "index.html", "report.md",
              "data_requests.csv", "resilience.csv", "nodes_features.csv", "clusters_details.csv", "validation.json"):
        print(f"  {f}")
    print(f"\nЭкран просмотра: откройте {out / 'index.html'} "
          f"(или `python serve.py` — с AI-ассистентом)")
    if a.open:
        webbrowser.open((out / "index.html").resolve().as_uri())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

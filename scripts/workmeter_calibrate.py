"""0.332: calibrate tests/workmeter.RATES — calls per quiet-box CPU-second for each guarded block (and RATE_FLOOR, the lowest).

Usage (quiet box, nothing else running):  python3 scripts/workmeter_calibrate.py tests/test_core_099.py tests/…  [--min-cpu 0.05]

Each file runs twice in a fresh process (tests/run_tests.py --one): WM_CALIBRATE=cpu logs each guard's unmetered CPU
seconds, WM_CALIBRATE=calls its deterministic call count; rate = calls / cpu per guard label.  Prints every label and
the minimum over labels with CPU ≥ --min-cpu (shorter blocks are timer noise); paste rates (rounded down to 2
significant figures) into tests/workmeter.RATES and the minimum into RATE_FLOOR."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(f: str, mode: str) -> dict:
    with tempfile.NamedTemporaryFile("r", suffix=".log", delete=False) as t:
        log = t.name
    env = {**os.environ, "WM_CALIBRATE": mode, "WM_LOG": log, "PYTHONPATH": str(ROOT)}
    p = subprocess.run([sys.executable, str(ROOT / "tests" / "run_tests.py"), "--one", f], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    out = {}
    for line in Path(log).read_text().splitlines():
        lab, _, v = line.split("\t")
        out[lab] = out.get(lab, 0.0) + float(v)
    os.unlink(log)
    if "@@RESULT" not in p.stdout:
        print(p.stdout[-2000:], p.stderr[-2000:], file=sys.stderr)
    return out


def main(argv: list[str]) -> None:
    min_cpu = 0.05
    if "--min-cpu" in argv:
        i = argv.index("--min-cpu")
        min_cpu = float(argv[i + 1])
        del argv[i:i + 2]
    rows = []
    for f in argv:
        cpu, calls = _run(f, "cpu"), _run(f, "calls")
        for lab in sorted(cpu):
            c, n = cpu[lab], calls.get(lab)
            rows.append((lab, c, n, n / c if n and c > 0 else None))
    for lab, c, n, r in rows:
        print(f"{lab:48s} cpu {c:9.3f} s  calls {n or 0:>13,.0f}  rate {r or 0:>12,.0f}/s")
    ok = [r for _, c, n, r in rows if r and c >= min_cpu]
    print(f"RATE_FLOOR: min rate over {len(ok)} guards with cpu ≥ {min_cpu} s: {min(ok):,.0f} calls/s" if ok else "no guard")


if __name__ == "__main__":
    main(sys.argv[1:])

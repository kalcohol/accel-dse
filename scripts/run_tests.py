#!/usr/bin/env python3
"""0.158/0.159: stdlib parallel test runner (no pytest) — one tests/run_tests.py --one process per test file, longest-first, per-file timeout.

    python3 scripts/run_tests.py [-j N] [--timeout S] [files…]

Same tests as ``python3 tests/run_tests.py`` (every tests/test_*.py, nothing skipped or filtered); only the scheduling differs.
Per-file wall times are kept in scripts/test_durations.json to order the next run longest-first (shortest makespan).
Exit code 0 iff every file passed; a file over --timeout is killed and reported as TIMEOUT (a hang, not a pass)."""
from __future__ import annotations

import argparse, json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "tests" / "run_tests.py"
DUR = ROOT / "scripts" / "test_durations.json"


def run(f: Path, timeout: float) -> tuple[str, str, float, str, int, int]:
    """One test file in its own interpreter through the project's stdlib harness (tests/run_tests.py --one)."""
    t0 = time.monotonic()
    try:
        p = subprocess.run([sys.executable, str(HARNESS), "--one", str(f)], cwd=ROOT, capture_output=True, text=True,
                           timeout=timeout)
        out = p.stdout + p.stderr
        m = re.search(r"^@@RESULT (\d+) (\d+)$", p.stdout, re.M)
        ok, bad = (int(m[1]), int(m[2])) if m else (0, 1)          # no result line = crash → counted as a failure
        st = "ok" if (p.returncode == 0 and m and bad == 0) else "FAIL"
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        st, ok, bad = "TIMEOUT", 0, 1
    return f.name, st, time.monotonic() - t0, out, ok, bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("files", nargs="*")
    a = ap.parse_args(argv)
    files = [Path(x).resolve() for x in a.files] or sorted((ROOT / "tests").glob("test_*.py"))
    dur = json.loads(DUR.read_text()) if DUR.exists() else {}
    files.sort(key=lambda f: -dur.get(f.name, 1e9))          # unknown files first (assume slow)
    t0, passed, bad = time.monotonic(), 0, []
    with ThreadPoolExecutor(a.j) as ex:
        for fut in as_completed([ex.submit(run, f, a.timeout) for f in files]):
            name, st, dt, out, ok, _ = fut.result()
            dur[name] = round(dt, 2)
            passed += ok
            print(f"{st:7s} {dt:7.1f}s {name}", flush=True)
            if st != "ok":
                bad.append((name, st, out[-3000:]))
    DUR.write_text(json.dumps(dict(sorted(dur.items())), indent=0) + "\n")
    for name, st, out in bad:
        print(f"\n===== {st} {name}\n{out}")
    print(f"\n{passed} passed, {len(bad)} file(s) failed/timed out, {len(files)} files, wall {time.monotonic() - t0:.0f}s")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

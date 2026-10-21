#!/usr/bin/env python3
"""Zero-dep test runner (pytest optional).

Usage: python3 tests/run_tests.py [-j N] [substring ...]   — runs every tests/test_*.py
(optionally only test files / functions whose name contains a substring).

-j N runs N test modules at once, each in its own interpreter (0.72; default: min(8, CPU count); -j 1 = the old
in-process sequential run).  Every test still runs exactly once and alone in its module's process; only the wall
time changes.  Each OK line carries the test's duration.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).parent
# known slow modules first, so they do not end up as the tail of a parallel run
SLOW_FIRST = ("test_core_scenario_search_topk1", "test_core_scenario_search_topk2", "test_core_057", "test_core_057b",
              "test_core_scenario_search", "test_core_053", "test_core_061_3", "test_core_052", "test_core_061_4",
              "test_core_060", "test_core_058", "test_core_062", "test_core_054", "test_core_055", "test_core_056")


DURATIONS = HERE.parent / "scripts" / "test_durations.json"


def _durations() -> dict:
    """Per-file wall seconds recorded by scripts/run_tests.py (tracked; local updates stay out of commits)."""
    try:
        return json.loads(DURATIONS.read_text())
    except (OSError, ValueError):
        return {}


def order_longest_first(mods: list[Path], dur: dict) -> list[Path]:
    """0.292: longest-first (LPT) by the recorded per-file times, so a long module never starts last and becomes the
    tail of a parallel run (the 0.291 suite ended on test_core_domain, 125 s, which SLOW_FIRST did not list).  Files
    without a record: SLOW_FIRST ones first, the rest after the known ones, in name order.  Only the start order
    changes — every module still runs once, in its own process."""
    def key(m: Path):
        d = dur.get(m.name)
        if not isinstance(d, (int, float)):
            d = 1e9 if m.stem in SLOW_FIRST else -1.0
        return (-d, SLOW_FIRST.index(m.stem) if m.stem in SLOW_FIRST else len(SLOW_FIRST), m.name)
    return sorted(mods, key=key)


def _run_module(mod_path: Path, argv: list[str]) -> tuple[int, int]:
    failed = passed = 0
    spec = importlib.util.spec_from_file_location(mod_path.stem, mod_path)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for name in sorted(n for n in dir(m) if n.startswith("test_")):
        if argv and not _wants(mod_path.stem, name, argv):
            continue
        t1 = time.perf_counter()
        try:
            getattr(m, name)()
            passed += 1
            print(f"OK   {mod_path.stem}::{name}  ({time.perf_counter() - t1:.1f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {mod_path.stem}::{name}: {exc}")
            traceback.print_exc()
    return passed, failed


def _norm(a: str) -> str:
    return Path(a.split("::")[0]).stem if a.endswith(".py") or "/" in a.split("::")[0] else a.split("::")[0]


def _wants(stem: str, name: str, argv: list[str]) -> bool:
    """0.74: exact selection — ``test_x`` / ``tests/test_x.py`` = whole module, ``test_x::test_y`` = one test,
    any other word = a test function of that exact name; ``-k WORD`` = substring match (the pre-0.74 behaviour)."""
    for a in argv:
        if a.startswith("-k="):
            if a[3:] in stem or a[3:] in name:
                return True
        elif "::" in a:
            if _norm(a) == stem and a.split("::", 1)[1] == name:
                return True
        elif _norm(a) == stem or a == name:
            return True
    return False


def _selected(mod_path: Path, argv: list[str]) -> bool:
    if not argv:
        return True
    src = mod_path.read_text(encoding="utf-8")
    names = re.findall(r"^def (test_\w+)", src, re.M)
    return any(_wants(mod_path.stem, n, argv) for n in names) or any(_norm(a) == mod_path.stem for a in argv)


def _kflags(argv: list[str]) -> list[str]:
    out, it = [], iter(argv)
    for a in it:
        out.append("-k=" + next(it, "") if a == "-k" else a)
    return out


def main(argv: list[str]) -> int:
    jobs = min(8, os.cpu_count() or 1)
    if argv and argv[0] == "--one":                       # child: one module, sequential
        p, f = _run_module(Path(argv[1]), argv[2:])
        print(f"@@RESULT {p} {f}")
        return 1 if f else 0
    if argv and argv[0].startswith("-j"):
        if argv[0] == "-j":
            jobs, argv = int(argv[1]), argv[2:]
        else:
            jobs, argv = int(argv[0][2:]), argv[1:]
    argv = _kflags(argv)
    failed = passed = 0
    t0 = time.perf_counter()
    mods = sorted(HERE.glob("test_*.py"))
    if argv and not any(_selected(m, argv) for m in mods):
        print(f"no test module / test matches {argv} (exact names; use -k WORD for substring)")
        return 2
    if jobs <= 1:
        for mod_path in mods:
            p, f = _run_module(mod_path, argv)
            passed += p
            failed += f
    else:
        mods = [m for m in mods if _selected(m, argv)]
        mods = order_longest_first(mods, _durations())

        def child(m: Path):
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--one", str(m), *argv],
                               capture_output=True, text=True, env=os.environ.copy())
            return m, r

        with ThreadPoolExecutor(max_workers=jobs) as ex:
            for fut in as_completed([ex.submit(child, m) for m in mods]):
                m, r = fut.result()
                out = r.stdout
                mres = re.search(r"^@@RESULT (\d+) (\d+)$", out, re.M)
                sys.stdout.write(re.sub(r"^@@RESULT .*\n?", "", out, flags=re.M))
                if r.stderr:
                    sys.stdout.write(r.stderr)
                if mres:
                    passed += int(mres[1])
                    failed += int(mres[2])
                else:                                     # import error / crash: count the module as failed
                    failed += 1
                    print(f"FAIL {m.stem}: module crashed (exit {r.returncode})")
                sys.stdout.flush()
    print(f"{'PASSED' if failed == 0 else 'FAILED'}: {passed} passed, {failed} failed in {time.perf_counter() - t0:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

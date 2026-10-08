#!/usr/bin/env python3
"""Zero-dep test runner (pytest optional).

Usage: python3 tests/run_tests.py [substring ...]   — runs every tests/test_*.py
(optionally only test files / functions whose name contains a substring).
"""
from __future__ import annotations

import importlib.util
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv: list[str]) -> int:
    failed = passed = 0
    t0 = time.time()
    for mod_path in sorted(Path(__file__).parent.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(mod_path.stem, mod_path)
        assert spec and spec.loader
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        for name in sorted(n for n in dir(m) if n.startswith("test_")):
            if argv and not any(a in mod_path.stem or a in name for a in argv):
                continue
            try:
                getattr(m, name)()
                passed += 1
                print(f"OK   {mod_path.stem}::{name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {mod_path.stem}::{name}: {exc}")
                traceback.print_exc()
    print(f"{'PASSED' if failed == 0 else 'FAILED'}: {passed} passed, {failed} failed in {time.time() - t0:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

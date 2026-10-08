#!/usr/bin/env python3
"""Zero-dep test runner (pytest optional)."""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    mod_path = Path(__file__).with_name("test_conservation.py")
    spec = importlib.util.spec_from_file_location("test_conservation", mod_path)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    failed = 0
    for name in sorted(n for n in dir(m) if n.startswith("test_")):
        try:
            getattr(m, name)()
            print(f"OK  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
    print(f"{'PASSED' if failed == 0 else 'FAILED'} ({failed} failures)")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())

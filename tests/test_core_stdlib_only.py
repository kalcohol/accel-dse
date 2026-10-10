"""Guard: the suite is stdlib-only (tests/run_tests.py is the runner; msa1 has no pytest)."""
import re
from pathlib import Path


def test_no_pytest_in_tests():
    bad = []
    for p in sorted(Path(__file__).parent.glob("*.py")):
        if p.name == Path(__file__).name:
            continue
        s = p.read_text(encoding="utf-8")
        if re.search(r"^\s*(import pytest|from pytest\b)", s, re.M) or "pytest." in s:
            bad.append(p.name)
        # the runner calls test functions with no arguments: no fixtures / parametrize
        bad += [f"{p.name}:{m}" for m in re.findall(r"^def (test_\w+)\((?!\)|\w+=)", s, re.M)]
    assert not bad, bad

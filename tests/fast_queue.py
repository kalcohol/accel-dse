"""0.302 suite time: skip the SLO-goodput rate searches in tests that never read them.

``pdqueue.queue_report`` (and the PD split table of ``disagg``) bisect the SLO-goodput and stability rates with
``_slo_rate`` — on the PD-queue scenarios that is ≥ 90 % of the report's time (cProfile, test_core_057: 190 of 199 s).
Many queueing / DES tests read only the operating-point outputs (TTFT / TPOT / waits / KV-capacity / energy at λ),
which ``_slo_rate`` does not touch (it only reads ``ctx`` and the per-λ memo).  For those tests the rate search is
replaced by the constant ``SKIPPED`` (−1 req/s: impossible as a rate, and equal to itself so whole-report identity checks
still compare); every field it produces is in ``SLO_KEYS`` and nothing else changes.

Equivalence check (not part of the suite): ``ACCEL_FASTQ_VERIFY=1 python3 tests/run_tests.py --one tests/test_x.py``
runs every wrapped test with the real search and, at each ``queue_report`` call, also a stubbed run on a deep copy of
the context, and fails on any difference outside ``SLO_KEYS`` (NOTES_0302 records the run).  The SLO rate search
itself stays covered by the tests that read it (test_core_060 / 061 / domain / scenario_search …, not wrapped)."""
import contextlib
import copy
import functools
import math
import os

from accel_dse.core import disagg, pdqueue

SLO_KEYS = frozenset(("slo_rate_rps", "slo_goodput_per_card", "stable_rate_rps", "pd_slo_splits", "pd_slo_best_split",
                      "pd_slo_splits_sampled", "best_slo"))
VERIFY = os.environ.get("ACCEL_FASTQ_VERIFY") == "1"


SKIPPED = -1.0


def _skipped_rate(*a, **k):
    return SKIPPED


def _diff(a, b, path="") -> list:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in set(a) | set(b):
            if k in SLO_KEYS:
                continue
            if k not in a or k not in b:
                out.append(f"{path}.{k} missing")
            else:
                out += _diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path} len"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return []
    if callable(a) and callable(b):
        return []
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


@contextlib.contextmanager
def no_slo_search():
    if VERIFY:
        real = pdqueue.queue_report

        def checked(ctx, *a, **k):
            c2 = copy.deepcopy(ctx)
            with _patched():
                fast = real(c2, *a, **k)
            full = real(ctx, *a, **k)
            d = _diff(full, fast)
            assert not d, d[:5]
            return full
        pdqueue.queue_report, old = checked, real
        dq = disagg.queue_report
        disagg.queue_report = checked
        try:
            yield
        finally:
            pdqueue.queue_report, disagg.queue_report = old, dq
        return
    with _patched():
        yield


@contextlib.contextmanager
def _patched():
    a, b = pdqueue._slo_rate, disagg._slo_rate
    pdqueue._slo_rate = disagg._slo_rate = _skipped_rate
    try:
        yield
    finally:
        pdqueue._slo_rate, disagg._slo_rate = a, b


def fast(f):
    @functools.wraps(f)
    def g(*a, **k):
        with no_slo_search():
            return f(*a, **k)
    return g


def fast_module(ns: dict, only=None, skip=()):
    for k, v in list(ns.items()):
        if k.startswith("test_") and callable(v) and (only is None or k in only) and k not in skip:
            ns[k] = fast(v)

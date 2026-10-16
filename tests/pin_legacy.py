"""0.173: pin a test to the ≤ 0.172 mapping (no wide GEMM orientation / instance plan, no split-K).

Used by the queueing / DES operating-point tests and by documented worked examples (MODEL §19 etc.): they check
mechanisms or reproduce published worked numbers at fixed inputs; the 0.173 faster mapping moves those inputs (ρ, step),
which is a load change, not a model error.  Old → new of the default path is recorded in CHANGELOG 0.173.0."""
import functools

from accel_dse.core import evaluate as E


def legacy_0172(f):
    @functools.wraps(f)
    def g(*a, **k):
        with E.model_flags(**E.LEGACY_0172, **E.LEGACY_0192, FUSE_QKV=False):   # 0.198: + unfused QKV   # 0.193: + the ≤ 0.192 reference GPUs
            return f(*a, **k)
    return g


def pin_module(ns: dict, only=None):
    for k, v in list(ns.items()):
        if k.startswith("test_") and callable(v) and (only is None or k in only):
            ns[k] = legacy_0172(v)


def legacy_0192(f):
    """0.193: also pin the ≤ 0.192 reference GPUs (wide schedule with N padding, guide split-K set).  Combined with
    the ≤ 0.172 mapping when the test was already pinned."""
    @functools.wraps(f)
    def g(*a, **k):
        with E.model_flags(**E.LEGACY_0192, FUSE_QKV=False):   # 0.198: + unfused QKV
            return f(*a, **k)
    return g


def pin_module_0192(ns: dict, only=None):
    for k, v in list(ns.items()):
        if k.startswith("test_") and callable(v) and (only is None or k in only):
            ns[k] = legacy_0192(v)


def unfused_qkv(f):
    """0.198: run on the ≤ 0.197 IR (separate q / k / v GEMMs) — per-projection shape / kernel-count checks."""
    @functools.wraps(f)
    def g(*a, **k):
        with E.model_flags(FUSE_QKV=False):
            return f(*a, **k)
    return g


def pin_module_unfused(ns: dict, only=None):
    for k, v in list(ns.items()):
        if k.startswith("test_") and callable(v) and (only is None or k in only):
            ns[k] = unfused_qkv(v)

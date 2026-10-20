"""0.291: per-run cache of heavy, pure report calls shared across test files (stdlib only).

``cached(fn)`` wraps a pure report function (``disagg_report``, ``pdsim.capture_ctx``).  The key is the SHA-256 of
  * the code version: every file under ``accel_dse/`` (sources + data), hashed once per process;
  * the live model state: every loaded ``accel_dse`` module's UPPER-CASE data globals (``model_flags`` / pin_legacy
    overrides, ``REF_HW``, ``SPLIT_K`` …), the code location of every module-level function and class method (a
    monkeypatched one changes it) and the value of every module-level ContextVar (e.g. ``disagg.CTX_SINK``);
  * the function and its full arguments (canonical text: every dataclass field, sorted sets / dicts).
An argument that is not plain data, or a result that cannot be pickled, is simply not cached (the call runs as before).

Storage: always an in-process dict; additionally, when ``tests/run_tests.py`` sets ``ACCEL_DSE_TEST_RUNCACHE`` to a
fresh per-run directory, an on-disk entry (atomic rename) shared by the parallel test processes of that run.  A process
that finds another one computing the same key (O_EXCL lock file) waits for its result instead of recomputing; the
directory is removed at the end of the run, so nothing survives across runs.  A hit returns an unpickled copy of
the stored result (bit-identical floats, independent objects), so a test that mutates its result cannot affect
another; the computing call returns its own object as before.  Assertions are untouched.
"""
from __future__ import annotations

import contextvars
import dataclasses
import enum
import functools
import hashlib
import os
import pathlib
import pickle
import re
import sys
import time
import types

ENV = "ACCEL_DSE_TEST_RUNCACHE"
_ROOT = pathlib.Path(__file__).resolve().parent.parent / "accel_dse"
_MEM: dict = {}
_CODE: list = []
STATS = {"hit_mem": 0, "hit_disk": 0, "wait": 0, "miss": 0, "uncached": 0, "saved_s": 0.0}
_COST: dict = {}                      # key → seconds the computing call took (reported as saved on a hit)
_ATEXIT: list = []
WAIT_S = 900.0


def _tree_hash(root: pathlib.Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(root)).encode()); h.update(b"\0"); h.update(p.read_bytes())
    return h.hexdigest()


def code_version() -> str:
    """SHA-256 over every file under accel_dse/ (sources + release / chip data), once per process."""
    if not _CODE:
        _CODE.append(_tree_hash(_ROOT))
    return _CODE[0]


def canon(v, depth=0) -> str:
    """Process-independent text of a data value: sets / dict items sorted (no PYTHONHASHSEED order), every dataclass
    field included (also ``repr=False`` ones), floats as exact ``repr``.  TypeError for anything else."""
    if depth > 40:
        raise TypeError("too deep")
    if v is None or isinstance(v, (bool, int, float, str, bytes)):
        return f"{type(v).__name__}:{v!r}"
    if isinstance(v, (tuple, list)):
        return f"{type(v).__name__}[" + ",".join(canon(x, depth + 1) for x in v) + "]"
    if isinstance(v, (set, frozenset)):
        return f"{type(v).__name__}{{" + ",".join(sorted(canon(x, depth + 1) for x in v)) + "}"
    if isinstance(v, dict):
        return "dict{" + ",".join(sorted(f"{canon(k, depth + 1)}:{canon(x, depth + 1)}" for k, x in v.items())) + "}"
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        t = type(v)
        return (f"{t.__module__}.{t.__qualname__}(" +
                ",".join(f"{f.name}={canon(getattr(v, f.name), depth + 1)}" for f in dataclasses.fields(v)) + ")")
    if isinstance(v, enum.Enum):
        return f"{type(v).__module__}.{type(v).__qualname__}.{v.name}"
    raise TypeError(type(v).__name__)


def _canon_or_none(v):
    try:
        return canon(v)
    except (TypeError, RecursionError):
        return None


# exact memo tables (performance only; the model_flags context clears memo._TABLES): their fill level depends on what ran
# before in the process, not on the model
_PERF_MEMO = re.compile(r"^accel_dse\.[\w.]+\._\w*(MEMO|CACHE)$|^accel_dse\.core\.memo\._TABLES$"
                        r"|^accel_dse\.core\.scenario\._HINTS$")      # _HINTS: lazily filled type-hint table
_ALL: list = []


def _import_all():
    """Fingerprint the same module set in every process (whatever a test file happened to import)."""
    if not _ALL:
        import importlib
        import pkgutil
        import accel_dse
        for m in pkgutil.walk_packages(accel_dse.__path__, "accel_dse."):
            if not m.name.endswith("__main__"):
                importlib.import_module(m.name)
        _ALL.append(True)


def state() -> str:
    """Fingerprint of the live model state (see module docstring)."""
    h = hashlib.sha256()
    for item in _state_items():
        h.update(item.encode()); h.update(b"\0")
    return h.hexdigest()


def _state_items():
    _import_all()
    for name in sorted(n for n in sys.modules if n == "accel_dse" or n.startswith("accel_dse.")):
        mod = sys.modules[name]
        if mod is None:
            continue
        yield name
        for k in sorted(vars(mod)):
            v = vars(mod)[k]
            if isinstance(v, contextvars.ContextVar):
                try:
                    cv = repr(v.get())                                # the effective value (default included)
                except LookupError:
                    cv = "<unset>"
                yield f"{name}.{k}=cv:{cv}"
            elif isinstance(v, types.FunctionType):
                c = v.__code__
                yield f"{name}.{k}=fn:{v.__module__}.{v.__qualname__}@{c.co_filename}:{c.co_firstlineno}"
            elif isinstance(v, functools._lru_cache_wrapper) or hasattr(v, "__wrapped__"):
                w = getattr(v, "__wrapped__", None)
                c = getattr(w, "__code__", None)
                yield (f"{name}.{k}=wr:{getattr(w, '__qualname__', '?')}@{getattr(c, 'co_filename', '?')}:"
                       f"{getattr(c, 'co_firstlineno', '?')}")
            elif isinstance(v, type) and v.__module__ == name:       # patched methods of the module's classes
                for a, f in sorted(vars(v).items()):
                    c = getattr(f, "__code__", None)
                    if c is not None:
                        yield f"{name}.{k}.{a}@{c.co_filename}:{c.co_firstlineno}"
            elif k.isupper() and not _PERF_MEMO.match(f"{name}.{k}"):
                c = _canon_or_none(v)
                if c is not None:
                    yield f"{name}.{k}={c}"


def _key(fn, a, k):
    args = _canon_or_none((a, sorted(k.items())))
    if args is None:
        return None
    h = hashlib.sha256()
    for part in (code_version(), state(), f"{fn.__module__}.{fn.__qualname__}"):
        h.update(part.encode()); h.update(b"\0")
    h.update(args.encode())
    return h.hexdigest()


def _disk(key):
    d = os.environ.get(ENV)
    if not d or not os.path.isdir(d):
        return None
    return pathlib.Path(d) / key


def _store(path, data):
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _stats_at_exit(d):
    """Leave this process's counters in the run directory (tests/run_tests.py prints the run total)."""
    if not _ATEXIT:
        import atexit
        import json
        _ATEXIT.append(True)
        atexit.register(lambda: (d.is_dir() and (d / f"stats.{os.getpid()}.json").write_text(json.dumps(STATS))))


def cached(fn):
    @functools.wraps(fn)
    def g(*a, **k):
        key = _key(fn, a, k)
        if key is None:
            STATS["uncached"] += 1
            return fn(*a, **k)
        if key in _MEM:
            STATS["hit_mem"] += 1
            STATS["saved_s"] += _COST.get(key, 0.0)
            return pickle.loads(_MEM[key])
        path = _disk(key)
        lock = None
        if path is not None:
            t0 = time.monotonic()
            while True:
                if path.exists():
                    _MEM[key] = path.read_bytes()
                    try:
                        _COST[key] = float(path.with_suffix(".s").read_text())
                    except (OSError, ValueError):
                        _COST[key] = 0.0
                    STATS["hit_disk"] += 1
                    STATS["saved_s"] += _COST[key]
                    _stats_at_exit(path.parent)
                    return pickle.loads(_MEM[key])
                try:
                    fd = os.open(f"{path}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd); lock = f"{path}.lock"
                    break
                except FileExistsError:
                    if time.monotonic() - t0 > WAIT_S:          # a stuck / killed holder: compute ourselves
                        break
                    STATS["wait"] += 1
                    time.sleep(0.2)
        try:
            t0 = time.process_time()
            r = fn(*a, **k)
            _COST[key] = time.process_time() - t0
            try:
                data = pickle.dumps(r, protocol=4)
            except Exception:
                STATS["uncached"] += 1
                return r
            STATS["miss"] += 1
            _MEM[key] = data
            if path is not None:
                path.with_suffix(".s").write_text(f"{_COST[key]:.3f}")
                _store(path, data)
                _stats_at_exit(path.parent)
            return r
        finally:
            if lock:
                try:
                    os.unlink(lock)
                except OSError:
                    pass
    g.uncached = fn
    return g

#!/usr/bin/env python3
"""0.158/0.159: stdlib parallel test runner (no pytest) — one tests/run_tests.py --one process per test file, longest-first, per-file timeout.

    python3 scripts/run_tests.py [-j N] [--timeout S] [files…]

Same tests as ``python3 tests/run_tests.py`` (every tests/test_*.py, nothing skipped or filtered); only the scheduling differs.
Per-file wall times are kept in scripts/test_durations.json to order the next run longest-first (shortest makespan).
Exit code 0 iff every file passed; a file over --timeout is killed and reported as TIMEOUT (a hang, not a pass)."""
from __future__ import annotations

import argparse, json, os, re, signal, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "tests" / "run_tests.py"
DUR = ROOT / "scripts" / "test_durations.json"


STATUS = ROOT / "scripts" / ".run_status.json"     # 0.160: progress / liveness record (git-ignored)
_LIVE: set = set()                                  # running child processes (killed on a catchable signal)
_LOCK = threading.Lock()


def run(f: Path, timeout: float) -> tuple[str, str, float, str, int, int]:
    """One test file in its own interpreter through the project's stdlib harness (tests/run_tests.py --one)."""
    t0 = time.monotonic()
    p = subprocess.Popen([sys.executable, str(HARNESS), "--one", str(f)], cwd=ROOT, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    with _LOCK:
        _LIVE.add(p)
    try:
        so, se = p.communicate(timeout=timeout)
        out = so + se
        m = re.search(r"^@@RESULT (\d+) (\d+)$", so, re.M)
        ok, bad = (int(m[1]), int(m[2])) if m else (0, 1)          # no result line = crash → counted as a failure
        st = "ok" if (p.returncode == 0 and m and bad == 0) else "FAIL"
        if not m:
            out += f"\n[runner] no @@RESULT line: child exit {p.returncode} (crash / killed)"
    except subprocess.TimeoutExpired:
        p.kill()
        so, se = p.communicate()
        out, st, ok, bad = (so or "") + (se or ""), "TIMEOUT", 0, 1
    finally:
        with _LOCK:
            _LIVE.discard(p)
    return f.name, st, time.monotonic() - t0, out, ok, bad


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _write_status(d: dict) -> None:
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=0) + "\n")
    tmp.replace(STATUS)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--resume", action="store_true", help="skip files that passed in the previous (dead) run")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args(argv)
    files = [Path(x).resolve() for x in a.files] or sorted((ROOT / "tests").glob("test_*.py"))
    prev = json.loads(STATUS.read_text()) if STATUS.exists() else None
    done_prev: dict = {}
    if prev and prev.get("state") == "running" and not _alive(prev.get("pid", -1)):
        # 0.160: SIGKILL / VM teardown cannot be caught — the next run reports the dead one instead of staying silent
        print(f"[runner] WARNING: previous run (pid {prev['pid']}, started {prev.get('started')}) died without a "
              f"summary after {len(prev.get('done', {}))}/{prev.get('total')} files; pending then: "
              f"{sorted(set(prev.get('files', [])) - set(prev.get('done', {})))[:20]}", flush=True)
        if a.resume:
            done_prev = {k: v for k, v in prev.get("done", {}).items() if v["st"] == "ok"}
    elif a.resume:
        print("[runner] --resume: no dead previous run found; running everything", flush=True)
    dur = json.loads(DUR.read_text()) if DUR.exists() else {}
    todo = [f for f in files if f.name not in done_prev]
    os.environ['ACCEL_TEST_JOBS'] = str(min(a.j, len(todo)))    # 0.277: CPU-second bounds know files run in parallel
    todo.sort(key=lambda f: -dur.get(f.name, 1e9))          # unknown files first (assume slow)
    own_cache = "ACCEL_DSE_TEST_RUNCACHE" not in os.environ   # 0.291: per-run dir for tests/runcache.py (removed at exit)
    if own_cache:
        import shutil, tempfile
        os.environ["ACCEL_DSE_TEST_RUNCACHE"] = tempfile.mkdtemp(prefix="accel-dse-runcache-")
        import atexit
        atexit.register(shutil.rmtree, os.environ["ACCEL_DSE_TEST_RUNCACHE"], True)
    status = {"state": "running", "pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S %z"),
              "total": len(files), "files": [f.name for f in files], "done": dict(done_prev)}
    _write_status(status)
    t0, bad = time.monotonic(), []
    passed = sum(v["passed"] for v in done_prev.values())
    if done_prev:
        print(f"[runner] resumed: {len(done_prev)} file(s) / {passed} tests already passed in the dead run", flush=True)

    def on_signal(sig, _frame):
        with _LOCK:
            for p in list(_LIVE):
                p.kill()
        status["state"] = f"interrupted:{signal.Signals(sig).name}"
        _write_status(status)
        pend = sorted(set(status["files"]) - set(status["done"]))
        print(f"\n[runner] INTERRUPTED by {signal.Signals(sig).name}: {len(status['done'])}/{len(files)} files done, "
              f"{passed} passed so far, {len(bad)} bad; pending {pend[:20]}{' …' if len(pend) > 20 else ''}",
              flush=True)
        os._exit(128 + sig)
    old = {sg: signal.signal(sg, on_signal) for sg in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    try:
        with ThreadPoolExecutor(a.j) as ex:
            for fut in as_completed([ex.submit(run, f, a.timeout) for f in todo]):
                name, st, dt, out, ok, _ = fut.result()
                dur[name] = round(dt, 2)
                passed += ok
                status["done"][name] = {"st": st, "passed": ok, "s": round(dt, 2)}
                _write_status(status)
                print(f"{st:7s} {dt:7.1f}s {name}   [{len(status['done'])}/{len(files)}]", flush=True)
                if st != "ok":
                    bad.append((name, st, out[-3000:]))
    except BaseException as e:                       # never exit without saying why
        status["state"] = f"crashed:{type(e).__name__}"
        _write_status(status)
        print(f"\n[runner] CRASHED: {type(e).__name__}: {e}", flush=True)
        raise
    finally:
        for sg, h in old.items():
            signal.signal(sg, h)
    DUR.write_text(json.dumps(dict(sorted(dur.items())), indent=0) + "\n")
    for name, st, out in bad:
        print(f"\n===== {st} {name}\n{out}")
    status["state"] = "done"
    _write_status(status)
    print(f"\n{passed} passed, {len(bad)} file(s) failed/timed out, {len(files)} files, wall {time.monotonic() - t0:.0f}s")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

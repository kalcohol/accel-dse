"""0.184: fuzz every sizing / headroom entry point (sizing_report per-resource + joint, upgrade_path, insight_report)
over random small chips (1 core, 1 package, tiny SRAM / lanes / arrays) with a per-call wall-clock timeout.
A hang, an exception other than a clean ValueError, or a broken report invariant is a finding.
Usage: python3 scripts/fuzz_sizing.py [n_cases] [seed] [timeout_s]"""
from __future__ import annotations

import math
import random
import signal
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accel_dse.core.budget import Budget, apply_preset          # noqa: E402
from accel_dse.core.insight import SIZING, insight_report, sizing_report, upgrade_path   # noqa: E402
from accel_dse.core.scenario import Scenario                    # noqa: E402

MODEL_SETS = [["qwen3-8b"], ["gpt-oss-120b"], ["qwen3-8b", "gpt-oss-120b"], ["esm2-650m"], ["wan2.1-1.3b"]]
MEMS = ["lpddr5x_1x64_8533_16g", "lpddr5x_2x64_8533_16g", "lpddr5x_4x64_8533_16g", "hbm3e_1s_8h24g_9600",
        "hbm3e_2s_12h24g_9600"]


class Hang(Exception):
    pass


def _alarm(sig, frm):
    raise Hang()


def random_case(rng: random.Random):
    s = Scenario().replace("mem_id", rng.choice(MEMS))
    ch = s.chip
    ch = replace(ch, engines=rng.choice([1, 1, 2, 3, 16]), rows=rng.choice([1, 2, 8, 56]), cols=rng.choice([1, 4, 56]),
                 vector_lanes=rng.choice([None, 1, 2, 17, 256]), sram_mib=rng.choice([0.25, 1.0, 3.0, 64.0]),
                 formats=replace(ch.formats, dequant_elems_per_cycle=rng.choice([None, 1.0, 2.0, 64.0])))
    s = s.replace("chip", ch)
    budget = rng.choice([None, Budget(), Budget(mm2_per_dq_lpddr=0.03, mm2_per_dq_hbm=14 / 1024),
                         Budget(**{k: v for k, v in apply_preset({"density_preset": "n7_tpuv4i"}).items()})])
    models = list(rng.choice(MODEL_SETS))
    return s, budget, models


def check_sizing(rep):
    for r in rep["resources"]:
        assert r["key"] in SIZING
        assert r["min_ok"] <= r["current"] * (1 + 1e-9), r
        assert r["min_ok"] > 0 and math.isfinite(r["min_ok"]), r
        if r["integer"]:
            assert r["min_ok"] == int(r["min_ok"]) and r["min_ok"] >= 1, r
        if r["fail_at"] is not None:
            assert r["fail_at"] < r["min_ok"], r
        if r.get("mm2_saved") is not None:
            assert r["mm2_saved"] >= -1e-9, r
    j = rep.get("joint")
    if j:
        for e in j["resources"]:
            assert e["key"] != "dram_units"
            assert e["solo_min"] <= e["joint"] * (1 + 1e-9) <= e["current"] * (1 + 1e-6) if "current" in e else True, e


def run(n=30, seed=184, timeout=120):
    rng = random.Random(seed)
    signal.signal(signal.SIGALRM, _alarm)
    findings, rejected = [], []
    for i in range(n):
        s, b, models = random_case(rng)
        calls = [("sizing", lambda: check_sizing(sizing_report(s, models, b))),
                 ("sizing_joint", lambda: check_sizing(sizing_report(s, models, b, joint=True,
                                                                      unpriced=rng.choice(["last", "free"])))),
                 ("path", lambda: upgrade_path(s, models, None, b, steps=rng.choice([1, 2]), beam=rng.choice([1, 2]))),
                 ("insight", lambda: insight_report(s, models, b))]
        for name, fn in calls:
            t0 = time.time()
            signal.alarm(timeout)
            try:
                fn()
            except Hang:
                findings.append((i, name, "HANG", s.chip.engines, s.mem_id, models))
            except ValueError as e:                           # clean rejection of an invalid input is allowed, but counted
                rejected.append((i, name, str(e)[:120]))
            except Exception as e:                              # noqa: BLE001
                findings.append((i, name, repr(e)[:300], s.chip.engines, s.mem_id, models, traceback.format_exc()[-800:]))
            finally:
                signal.alarm(0)
            print(f"case {i} {name} {time.time() - t0:.1f}s", flush=True)
    print(f"{len(rejected)} clean rejection(s):", *sorted({r[2] for r in rejected}), sep="\n  ")
    return findings


if __name__ == "__main__":
    a = sys.argv[1:]
    f = run(int(a[0]) if a else 30, int(a[1]) if len(a) > 1 else 184, int(a[2]) if len(a) > 2 else 120)
    for x in f:
        print("FINDING", *x)
    print(f"{len(f)} finding(s)")
    sys.exit(1 if f else 0)

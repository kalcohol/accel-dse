"""0.323: cross-module consistency — for the same design every path computes the same numbers: eval vs energy-DSE
row vs design report vs compare vs the wire (server ≡ static Pages build: both serve.respond) vs CLI.  Property tests
over a seeded random set of designs (stdlib random).  Plus the capped-DSE speedups (mapping memo keyed without the
clock, memoised HBM id parses) — results bit-identical."""
import copy
import json
import math
import os
import random
import subprocess
import sys
import tempfile

from accel_dse import api, serve
from accel_dse.core import energy_dse, ir, mapping, powercap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rel(a, b):
    if a is None or b is None:
        return 0.0 if a is b else math.inf
    if isinstance(a, (str, bool)) or isinstance(b, (str, bool)):
        return 0.0 if a == b else math.inf
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


def _design(rng):
    b = {"chip_preset": rng.choice(["100T", "1P", "H100-like", "1P-lp4x64"]),
         "scenario": {"model": rng.choice(["qwen3-8b", "qwen3-30b-a3b", "qwen3-1.7b"]),
                      "serving": {"phase": rng.choice(["decode", "decode", "prefill"]), "batch": rng.choice([1, 8, 64]),
                                  "prompt": rng.choice([512, 2048]), "ctx": 4096}},
         "energy": {"energy_preset": rng.choice(["tpuv4i_7nm", "horowitz_45nm"])},
         "budget": {"density_preset": "n5_derived"}}
    cool = rng.choice([None, "air_oam", "air_hgx", "liquid"])
    if cool:
        b["cooling"] = cool
    cap = rng.choice([None, None, 250, 600])
    if cap:
        b["budget"]["power_W_card"] = cap
    return b, rng.choice(["n5_cowos_s", "n5_cowos_l"]), rng.choice(["accel_2026", "server_2026"]), \
        rng.choice([None, 0.05, 0.1])


def test_eval_equals_dse_base_row():
    rng, n = random.Random(323), 0
    for _ in range(10):
        b, cost, tco, d0 = _design(rng)
        dd = {"d0_per_cm2": d0} if d0 is not None else {}
        e = api.api_eval({**b, "tco": {"preset": tco, "cost": cost, **dd},
                          **({"power_cap": True} if b["budget"].get("power_W_card") else {})})
        r = api.api_insight_dse({**b, "mac_scales": [1], "sram_scales": [1], "cost": cost, "tco": tco, **dd})
        row = r["rows"][r["base"]]
        et, en, th, sh = e["tco"], e["energy"], e.get("thermal") or {}, e.get("shoreline") or {}
        pc = en.get("power_cap") or {}
        pairs = {
            "J": (pc["dvfs"]["dfs"]["J_per_unit"] if pc.get("capped") else en["J_per_unit"], row["J_per_unit"]),
            "W_card": (th.get("card_W"), row["W_card"]) if th else (None, None),
            "thr": (et["thr"], row["thr"]),
            "pkg_W": (th.get("pkg_W"), (row.get("thermal") or {}).get("pkg_W")),
            "th_status": (th.get("status"), (row.get("thermal") or {}).get("status")),
            "buildable": (sh.get("feasible") is not False and th.get("feasible") is not False, row["buildable"]),
            "shoreline": (sh.get("status"), row.get("shoreline")),
            "mem_edges": (en["noc"]["dram"]["mem_edges"], row["mem_edges"]),
            "shore_edges": (sh.get("mem", {}).get("edges", row["mem_edges"]), row["mem_edges"]),
            "usd_good_pkg": (et["pkg_usd"], row["usd_good_pkg"]),
            "usd_per_M": (et["usd_per_Munit"], row["tco"]["usd_per_Munit"]),
            "capped": (bool(pc.get("capped")), bool(row.get("capped"))),
        }
        bad = {k: v for k, v in pairs.items() if _rel(*v) > 1e-9}
        assert not bad, (b, cost, tco, d0, bad)
        if pc:
            assert pc["cap_src"] == "budget.power_W_card"
        if d0 is None:
            assert any("未设 D0" in x for x in et.get("notes", [])) and "未设 D0" in et["pkg_src"]
        n += 1
    assert n == 10


def _items(rep):
    return {it["key"]: it for s in rep["sections"] for it in s["items"]}


def test_report_and_compare_equal_row():
    rng = random.Random(3231)
    flat = ("thr", "J_per_unit", "W_card", "static_W", "bound", "shoreline", "buildable", "usd_good_pkg", "capped")
    sub = {"pkg_W": ("thermal", "pkg_W"), "th_status": ("thermal", "status"), "usd_per_Munit": ("tco", "usd_per_Munit")}
    checked = 0
    for _ in range(3):
        b = {"chip_preset": rng.choice(["1P", "H100-like"]),
             "scenario": {"model": rng.choice(["qwen3-8b", "qwen3-1.7b"]),
                          "serving": {"phase": rng.choice(["decode", "prefill"]), "batch": 8, "prompt": 2048, "ctx": 4096}},
             "energy": {"energy_preset": "tpuv4i_7nm"}, "budget": {"density_preset": "n5_derived",
                                                                   "power_W_card": rng.choice([200, 400])},
             "mac_scales": [0.5, 1], "sram_scales": [1], "cost": "n5_cowos_s", "tco": "accel_2026",
             "cooling": rng.choice(["air_hgx", "liquid"]), "f_scales": [1, 0.7]}
        r = api.api_insight_dse(b)
        picks = rng.sample([k for k, x in enumerate(r["rows"]) if x.get("feasible")], 2)
        reps = {}
        for k in picks:
            it, row = _items(api.api_report({**b, "report_row": k, "report_format": "json"})["report"]), r["rows"][k]
            reps[k] = it
            for key in flat:
                if key in it:
                    assert _rel(it[key]["value"], row.get(key)) <= 1e-12, (key, it[key]["value"], row.get(key))
            for key, (a, f) in sub.items():
                if key in it:
                    assert _rel(it[key]["value"], (row.get(a) or {}).get(f)) <= 1e-12, key
            checked += 1
        cmp_ = api.api_report({**b, "report_rows": picks, "report_format": "json"})["compare"]
        for s in cmp_["sections"]:
            for it in s["items"]:
                for j, k in enumerate(picks):
                    ri, v = reps[k].get(it["key"]), it["cells"][j]
                    if ri is None:
                        continue
                    cv = v.get("value") if isinstance(v, dict) else v
                    assert cv == ri["value"] or _rel(cv, ri["value"]) <= 1e-12, (it["key"], cv, ri["value"])
                    if isinstance(v, dict):
                        assert v.get("tag") == ri["tag"] and v.get("unit", ri["unit"]) == ri["unit"], it["key"]
    assert checked == 6


_CLI_BODY = {"chip_preset": "H100-like", "scenario": {"model": "qwen3-30b-a3b",
             "serving": {"phase": "decode", "batch": 1, "prompt": 2048, "ctx": 4096}},
             "energy": {"energy_preset": "tpuv4i_7nm"}, "budget": {"density_preset": "n5_derived", "power_W_card": 150},
             "cooling": "liquid", "power_cap": True, "tco": {"preset": "accel_2026", "cost": "n5_cowos_s"}}


def test_wire_static_and_cli_equal_api():
    # server ≡ static Pages build by construction (both call serve.respond); the bytes are api.clean(api_x(body))
    for path, fn, body in (("/api/eval", api.api_eval, _CLI_BODY),
                           ("/api/insight_dse", api.api_insight_dse,
                            {**{k: v for k, v in _CLI_BODY.items() if k not in ("power_cap", "tco")},
                             "mac_scales": [0.5, 1], "sram_scales": [1], "cost": "n5_cowos_s", "tco": "accel_2026"})):
        st, raw = serve.respond("POST", path, json.dumps(body).encode())
        assert st == 200
        assert json.loads(raw) == json.loads(json.dumps(api.clean(fn(copy.deepcopy(body))), allow_nan=False))
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(_CLI_BODY, f)
    try:
        o = subprocess.run([sys.executable, "-m", "accel_dse.cli", "eval", "--body", f.name, "--json"], cwd=ROOT,
                           capture_output=True, text=True)
    finally:
        os.unlink(f.name)
    assert o.returncode == 0, o.stderr
    st, raw = serve.respond("POST", "/api/eval", json.dumps(_CLI_BODY).encode())
    a, w = json.loads(o.stdout), json.loads(raw)
    for k in ("energy", "thermal", "tco", "shoreline"):
        assert a[k] == w[k], k


def test_cap_labels_iters_and_edges_on_budget_die():
    assert powercap.CAP_ITERS == 16 and powercap.dvfs_cap.__defaults__[1] == 16     # 0.323: DSE 12 / eval 22 → 16
    e = api.api_eval({**_CLI_BODY, "budget": {"density_preset": "n5_derived"}, "energy": {"energy_preset": "tpuv4i_7nm",
                                                                                          "tdp_W": 150}})
    pc = e["energy"]["power_cap"]
    assert pc["cap_src"] == "energy.tdp_W" and e["thermal"]["operating_point"] == "capped"
    assert abs(e["thermal"]["pkg_W"] - pc["dvfs"]["dfs"]["pkg_W"]) < 1e-9 and e["thermal"]["pkg_W"] <= 150 * (1 + 1e-9)
    assert abs(e["tco"]["thr"] / pc["dvfs"]["dfs"]["throughput_factor"]
               - api.api_eval({k: v for k, v in _CLI_BODY.items() if k != "power_cap"})["tco"]["thr"]) < 1e-6 * e["tco"]["thr"]
    # H100-like with auto edges: eval sizes the memory edges on the budget die (as the DSE row), not the preset 814 mm²
    b = {"chip_preset": "H100-like", "scenario": {"model": "qwen3-8b", "serving": {"phase": "decode", "batch": 64,
         "prompt": 2048, "ctx": 4096}}, "energy": {"energy_preset": "tpuv4i_7nm"}, "budget": {"density_preset": "n5_derived"},
         "cooling": "air_oam", "sensitivity": False}
    e = api.api_eval(b)
    r = api.api_insight_dse({**{k: v for k, v in b.items() if k != "sensitivity"}, "mac_scales": [1], "sram_scales": [1]})
    row = r["rows"][r["base"]]
    assert e["energy"]["noc"]["dram"]["mem_edges"] == row["mem_edges"] == e["shoreline"]["mem"]["edges"] == 3   # ≤0.322: 2
    assert e["energy"]["J_per_unit"] == row["J_per_unit"]
    pc = api.api_eval({**b, "power_cap": True})["energy"]["power_cap"]          # no cap anywhere: named, no default
    assert pc["error"].startswith("power_cap 需要功耗上限：budget.power_W_card 或 energy.tdp_W") and pc["cap_src"] is None


def _clear():
    import gc
    for o in gc.get_objects():
        if callable(getattr(o, "cache_clear", None)) and getattr(o, "__module__", "").startswith("accel_dse"):
            o.cache_clear()


def test_speedups_bit_identical():
    b = {"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "serving": {"phase": "decode", "batch": 64,
         "prompt": 2048, "ctx": 4096}}, "energy": {"energy_preset": "tpuv4i_7nm"},
         "budget": {"density_preset": "n5_derived", "power_W_card": 150}, "mac_scales": [0.5, 1, 2], "sram_scales": [1],
         "topologies": ["mesh", "ring"], "f_scales": [1, 0.7], "cost": "n5_cowos_s", "tco": "accel_2026",
         "cooling": "air_hgx", "d0_per_cm2": 0.1}
    out = {}
    try:
        for flag in (False, True):                 # ≤ 0.322 keying / no IR memo vs 0.323
            mapping.FREQ_FREE_KEY = energy_dse.RANK_MEMO = flag
            _clear()
            out[flag] = json.dumps(api.clean(api.api_insight_dse(copy.deepcopy(b))), sort_keys=True)
    finally:
        mapping.FREQ_FREE_KEY = energy_dse.RANK_MEMO = True
    assert out[False] == out[True]
    assert sum(1 for x in json.loads(out[True])["rows"] if x.get("capped")) >= 3      # the bisection ran


def test_gemm_cost_independent_of_clock():
    from dataclasses import replace
    from accel_dse.core.hardware import CHIPS as CHIP_PRESETS
    rng = random.Random(3232)
    for _ in range(40):
        ch = CHIP_PRESETS[rng.choice(sorted(CHIP_PRESETS))]
        m, k, n = rng.choice([1, 7, 64, 512]), rng.choice([128, 4096]), rng.choice([96, 4096, 12288])
        org = rng.choice(mapping.ORGS)
        a = mapping._gemm_cost.__wrapped__(ch, org, m, k, n, 1, "bf16", "bf16", None)
        b = mapping._gemm_cost.__wrapped__(replace(ch, freq_ghz=ch.freq_ghz * rng.choice([0.15, 0.37, 0.7])), org,
                                           m, k, n, 1, "bf16", "bf16", None)
        assert a == b, (ch.name, org, m, k, n)


def test_rank_memo_is_run_scoped():
    assert ir._RANK_MEMO is None
    from accel_dse.core.catalog import get_model
    from accel_dse.core.ir import Phase
    m, ph = get_model("qwen3-8b"), Phase("decode", 8, 1, 2048)
    with ir.rank_memo():
        a = ir.build_rank_ops(m, 0, ph)
        a.append("x")                                       # callers get their own list
        b = ir.build_rank_ops(m, 0, ph)
        assert "x" not in b and len(ir._RANK_MEMO) == 1
        with ir.rank_memo():                                # nested: the outer memo is kept
            assert len(ir._RANK_MEMO) == 1
    assert ir._RANK_MEMO is None and b == ir.build_rank_ops(m, 0, ph)
    api.api_insight_dse({"chip_preset": "1P", "scenario": {"model": "qwen3-1.7b"}, "energy": {"energy_preset":
                         "tpuv4i_7nm"}, "mac_scales": [1], "sram_scales": [1]})
    assert ir._RANK_MEMO is None                            # off again after a search (flags toggled later stay live)

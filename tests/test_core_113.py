"""0.113: dtype downcast label; hardware catalog / CLI–API / URL audits (standing tests)."""
from accel_dse.core.dtypes import FormatSupport, gemm_exec


def test_fp32_on_no_fp32_hw_is_downcast():
    sup = FormatSupport((("bf16", 1.0), ("fp16", 1.0), ("fp8", 2.0), ("int8", 2.0)))
    assert gemm_exec("fp32", "fp32", sup) == ("bf16", 1.0, "downcast", 3)
    assert gemm_exec("int8", "fp32", sup)[2] == "downcast"
    assert gemm_exec("fp8", "fp8", FormatSupport((("bf16", 1.0),)))[2] == "upcast_both"
    assert gemm_exec("fp32", "fp32", FormatSupport((("bf16", 1.0), ("fp32", 0.5))))[2] == "none"


def test_fingerprints_unchanged():
    from tests.test_core_0100 import _fp
    assert _fp({"exec_overlap": "stage"}) == "6dca732e67c83a75"   # 0.198 re-baseline (FUSE_QKV; was 5e72913c351849ac) ← 0.193 re-baseline (WIDE_WAVES, cta refhw; 0.192 8410808fd4f1e578) ← 0.192 re-baseline (state pinned first; 0.173 84df3be517a53ef4) (was 0a954915c5fb003f)
    assert _fp({}) == "bc06528a82e19e2a"   # 0.198 re-baseline (FUSE_QKV; was eb634f2e8b513ffb) ← 0.193 re-baseline (WIDE_WAVES, cta refhw; 0.192 b8ca44d92e7589cd) ← 0.192 re-baseline (state pinned first; 0.173 1b755ac357ec6d2e) (was 53ee9e08165795c6)


def test_cli_scenario_file_matches_api(tmp=None):
    import json, os, subprocess, sys, tempfile
    from accel_dse import api as A
    sc = {"model": "qwen3-30b-a3b", "layout": {"tp": 4, "ep": 4}, "chip": {"freq_ghz": 1.3, "attn_fmt": "kv"},
          "serving": {"batch": 8}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"scenario": sc}, f)
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run([sys.executable, "-m", "accel_dse", "eval", "--scenario", f.name, "--json"], cwd=root,
                             capture_output=True, text=True, check=True).stdout
    finally:
        os.unlink(f.name)
    c, a = json.loads(out), A.api_eval({"scenario": sc})
    assert c["hash"] == a["hash"] and c["summary"] == a["summary"]


def test_cli_flags_reach_scenario():
    import argparse
    from accel_dse import api as A, cli as C
    p = argparse.ArgumentParser(); C._scenario_args(p)
    h0 = A.scenario_from_body(C._body(p.parse_args([]))).hash()
    for argv in (["--attn-fmt", "kv"], ["--slc-policy", "lru"], ["--link-topology", "ring"], ["--d2d-std", "bow-256"],
                 ["--pp-split", "layers"], ["--moe-skew", "2"]):
        assert A.scenario_from_body(C._body(p.parse_args(argv))).hash() != h0, argv
    p0 = A.scenario_from_body(C._body(p.parse_args(["--pd"]))).hash()
    for argv in (["--pd-ttft-resid", "busy"], ["--pd-kv-policy", "swap"], ["--pd-load", "0.3"]):
        assert A.scenario_from_body(C._body(p.parse_args(["--pd"] + argv))).hash() != p0, argv


def test_api_scenario_echo_round_trip():
    import json
    from accel_dse import api as A
    for sc in ({"model": "qwen3-8b", "chip": {"freq_ghz": 0.9}, "link": {"topology": "ring"}},
               {"model": "carbon-500m", "pd": {"enabled": True, "load": 0.4, "ttft_resid": "busy"}},
               {"model": "qwen3-8b", "d2d_enabled": True, "package_cards": 2, "fabric": {"enabled": True}}):
        r1 = A.api_eval({"scenario": sc, "sensitivity": False})
        r2 = A.api_eval({"scenario": json.loads(json.dumps(r1["scenario"])), "sensitivity": False})
        assert r1["hash"] == r2["hash"] and r1["summary"] == r2["summary"]


def test_hw_catalog_consistency():
    import itertools
    from accel_dse import mem_catalog as M
    from accel_dse.core.refhw import REF_HW
    from accel_dse.core.d2d_catalog import D2D_STANDARDS
    for t, c in M.HBM_CATALOG.items():
        for r, h, d in itertools.product(c["rates"], c["heights"], c["densities"]):
            sp = M.make_spec(t, count=8, height=h, die_Gb=d, rate=r)
            assert sp.raw_GBps == 8 * c["stack_width"] * r / 8000 and sp.capacity_GB == 8 * h * d / 8
            assert M.parse_mem_id(sp.id).id == sp.id
    for k, h in REF_HW.items():
        ch = h.chip
        assert abs(2 * ch.rows * ch.cols * ch.engines * ch.freq_ghz / 1e3 / h.peak_bf16_tflops - 1) < 1e-9, k
        sp = M.parse_mem_id(h.mem_id)
        assert abs(sp.raw_GBps / h.hbm_GBps - 1) < 0.01 and abs(sp.capacity_GB / h.hbm_GB - 1) < 0.08, k
        assert h.link.GBps == h.link_GBps_bidir / 2, k
    for s, (l, r) in {"ucie-a-48": (64, 48), "ucie-a-64": (64, 64), "ucie-s-32": (16, 32), "bow-256": (16, 16),
                      "bow-512": (16, 32)}.items():
        assert D2D_STANDARDS[s]["per_unit_GBps"] == l * r / 8

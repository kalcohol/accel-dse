"""0.297: public MoE kernel timings vs the GPU reference path (core/extkern.py) — grouped-GEMM efficiency at small
per-expert row counts is already priced; it does not explain the 0.296 shared signature."""
import json
from pathlib import Path

from accel_dse.core import extkern as XK

ROOT = Path(__file__).resolve().parents[1]


def test_sources_recorded():
    d = XK.data()
    g, m = d["grouped_h800"], d["module_b200"]
    assert "7c6910172fd8b87405f2f8f089f97982bd5107c9" in g["source"] and "masked" in g["kernel"]
    assert len(g["rows"]) == 122 and sum(r["repeats"] for r in g["rows"]) == 340
    assert {r["m"] for r in g["rows"]} == {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}
    assert "0372799" in m["source"] and m["experts"] == 256 and m["top_k"] == 8 and m["intermediate"] == 2048 // 8
    assert [r["tokens"] for r in m["rows"]][:3] == [1, 8, 16] and m["rows"][-1] == {"tokens": 384, "time_us": 186.92}


def test_grouped_row_bytes_match_script():
    # the script's bandwidth column = groups·(m·k + k·n + 2·m·n) / t; DeepSeek_Simulator row (18, 4, 4096, 7168): 2875 GB/s
    r = next(x for x in XK.grouped_table() if (x["groups"], x["m"], x["n"]) == (18, 4, 4096))
    assert abs(r["meas_GBps"] - 18 * (4 * 7168 + 7168 * 4096 + 2 * 4 * 4096) / (r["meas_us"] * 1e-6) / 1e9) < 1


def test_small_rows_per_expert_already_priced():
    s = XK.summary()
    # 4 … 128 rows per group: model within a few % (bandwidth-bound; measured 1.5 – 2.9 TB/s on H800)
    assert 0.95 < s["grouped_small_m_ratio"] < 1.1
    assert all(0.95 < x["ratio"] < 1.15 for x in s["grouped_by_m"] if x["m"] <= 128)
    # 256+ rows: the masked grouped kernel runs below the model's FP8 ceiling (open, not the decode regime)
    assert s["grouped_large_m_ratio"] < 0.8


def test_module_b200_and_slope_bound():
    s = XK.summary()
    assert all(0.8 < r < 1.05 for r in s["module_ratio_floor"])
    sl = s["module_slope"]
    assert sl["meas_us_per_token"] > 10 * sl["pred_us_per_token"]          # flatter than measured after saturation …
    # … but transferred to LFM2.5 (per pair × MACs × top-4 × 40 layers) it bounds ≤ 4 µs/token of the ≈ 19 missing
    assert s["lfm_slope_bound_us_per_token"] < 4.0


def test_npu_small_m_is_array_padding_only():
    for chip in ("1P", "100T", "H100-like"):
        c = XK.npu_rows_curve(chip)
        assert c[0]["array_util"] < 0.2 and c[-1]["array_util"] > 0.95       # M granularity: padded rows idle
        assert all(x["bound"] == "dram" for x in c if x["m"] <= 128)          # … but MoE decode is weight-streaming


def test_ext_validation_kernels_fresh():
    d = json.loads((ROOT / "accel_dse/data/ext_validation.json").read_text())
    assert d["kernels"]["default"]["summary"] == XK.summary("default"), "re-run scripts/ext_validate.py"

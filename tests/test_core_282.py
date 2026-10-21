"""0.282: Drama 3 on one H200 (SGLang-Omni cookbook Tables 1–3) as an external speech diagnostic (core/extspeech.py),
and DeepSeek-V4 / GLM-5.3 hyper-connection scales counted as parameters."""
import importlib.util
import json
from pathlib import Path

from accel_dse.core import extspeech as XS
from accel_dse.core.catalog import get_model

ROOT = Path(__file__).resolve().parents[1]


def _sr():
    spec = importlib.util.spec_from_file_location("summarize_release", ROOT / "scripts/summarize_release.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_measurement_rows_sourced():
    rs = XS.rows()
    t1 = [r for r in rs if r["metric"] == "frames_per_s"]
    assert [r["concurrency"] for r in t1] == [1, 2, 4, 8, 16, 32, 64] and t1[0]["value"] == 166.7
    assert all(r["url"].startswith("https://sgl-project.github.io/sglang-omni/cookbook/") for r in rs)
    t3 = next(r for r in rs if r["metric"] == "latency_mean_s")
    assert abs(XS.OUT_S - t3["value"] / t3["rtf_mean"]) < 1e-12 and abs(XS.OUT_S - 4.395) < 1e-3


def test_sglang_windows():
    # strides 10, 10, 12, 32, then 64; overlap min(32, emitted)
    assert XS.sglang_windows(95) == [(10, 0, 0), (10, 10, 10), (12, 20, 20), (32, 32, 32), (31, 32, 64)]
    # window lengths = stream_window_lengths(10, (10, 12, 32), 64, 32) = {10, 20, 32, 64, 96} 「来源」
    assert [n + c for n, c, _ in XS.sglang_windows(200)] == [10, 20, 32, 64, 96, 96, 40]


def test_gap_decomposition():
    cat, dflt = XS.table("catalog"), XS.table("default")
    # roofline without GPU per-kernel costs: ≥ 3× optimistic at c = 1 (the 0.276 "≈ 2.2×" gap was at b = 1 frame steps)
    assert cat[0]["ratio"] > 3.0
    # trace-sourced per-kernel floor (ext default variant) + closed-loop prefill + SGLang vocoder windows
    assert 1.0 < dflt[0]["ratio"] < 1.35
    # ≤ 0.295 serial prefill pricing (kept per row as pred_serial): within 1.5× everywhere, second GPU within 10 %
    assert all(x["pred_serial"] / x["meas"] < 1.5 for x in dflt)
    t2 = {(x["c"], x["vocoder_gpu"]): x for x in XS.table("default", t2=True)}
    assert all(0.9 <= t2[(c, "second")]["pred_serial"] / t2[(c, "second")]["meas"] <= 1.05 for c in (128, 256))
    # 0.296 batched prefill (SGLang's one-forward prefill batch): c = 1 unchanged, never slower, and it removes the
    # serial pricing's compensation — the second-GPU rows (no vocoder) now over-predict by > 1.3× (diagnostic, open)
    assert dflt[0]["pred"] == dflt[0]["pred_serial"] and all(x["pred"] >= x["pred_serial"] for x in dflt)
    assert all(t2[(c, "second")]["ratio"] > 1.3 for c in (128, 256))
    assert 0.1 < dflt[-1]["shares"]["prefill"] < 0.25              # batched: prefill ≈ a sixth of the cycle at c = 64
    assert t2[(128, "same")]["ratio"] > t2[(128, "second")]["ratio"] + 0.3


def test_ext_validation_speech_fresh():
    d = json.loads((ROOT / "accel_dse/data/ext_validation.json").read_text())
    cur = {x["id"]: x["ratio"] for x in XS.table("default")}
    assert {x["id"]: x["ratio"] for x in d["speech"]["default"]["rows"]} == cur, "re-run scripts/ext_validate.py"


def test_hc_scales_are_parameters():
    sr = _sr()
    for n in ("layers.3.hc_attn_scale", "hc_head_scale", "model.language_model.layers.7.hc_ffn_scale", "mtp.0.hc_head_scale"):
        assert sr.NOT_SCALE_RE.search(n), n
    assert not sr.NOT_SCALE_RE.search("layers.3.attn.k_scale") and not sr.NOT_SCALE_RE.search("x.weight_scale")
    for mid in ("deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v4.1-flash", "glm-5.3-flash"):
        pc = get_model(mid).param_check()
        assert pc["rel_err"] == 0 and pc["mtp_ours"] == pc["mtp_release"], (mid, pc)

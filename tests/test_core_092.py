"""0.92 standing tests: GPU kernel model for reference hardware (instance_sched "cta": thread-block tile / wave
quantization, split-K, epilogue; Chip.kernel_us per-kernel cost) — opt-in, NOT adopted as refhw default (MODEL §38);
in-flight (chunked-prefill) memory check for max-load rows — adopted; NPU catalog defaults bit-identical."""
import math
from dataclasses import replace

from accel_dse.core import extval as X
from accel_dse.core import mapping as MP
from accel_dse.core.evaluate import evaluate, stage_ops
from accel_dse.core.hardware import CHIPS
from accel_dse.core.refhw import REF_HW
from accel_dse.core.scenario import Scenario


def test_defaults_unchanged():
    assert all(c.instance_sched != "cta" and c.kernel_us == 0.0 for c in CHIPS.values())
    assert all(h.chip.instance_sched == "wide" and h.chip.kernel_us == 0.0 for h in REF_HW.values())
    s = Scenario.from_dict({"model": "qwen3-8b"})
    assert "kernel_us" not in s.to_dict()["chip"]


def test_cta_wave_quantization_matches_guide_example():
    """NVIDIA Matrix Multiplication Background §3.2: A100, 256×128 tiles, M = 2304, K = 4096: N = 1536 is one full
    108-tile wave, N = 1544 adds a 9-tile tail wave → duration roughly doubles."""
    c = replace(REF_HW["A100-SXM-80GB"].chip, instance_sched="cta")
    saved = MP.CTA_TILES
    try:
        MP.CTA_TILES = ((256, 128),); MP._gemm_cost.cache_clear()
        a, b = (MP.gemm_cost(c, "reconf", 2304, 4096, n).cycles for n in (1536, 1544))
        assert 1.9 < b / a < 2.1
    finally:
        MP.CTA_TILES = saved; MP._gemm_cost.cache_clear()
    # library choice over the tile set softens the cliff; large GEMMs fill the GPU
    a, b = (MP.gemm_cost(c, "reconf", 2304, 4096, n).cycles for n in (1536, 1544))
    assert b / a < 1.2
    for hw in ("H100-SXM", "B200", "MI300X"):
        ch = replace(REF_HW[hw].chip, instance_sched="cta")
        m, k, n = 131072, 4096, 2048
        assert m * k * n / ch.macs / MP.gemm_cost(ch, "reconf", m, k, n).cycles > 0.95
        w = REF_HW[hw].chip
        assert m * k * n / w.macs / MP.gemm_cost(w, "reconf", m, k, n).cycles < 0.3       # wide (default)


def test_cta_split_k_and_grouped_waves():
    c = replace(REF_HW["H100-SXM"].chip, instance_sched="cta")
    one = MP.gemm_cost(c, "reconf", 128, 16384, 128).cycles           # 1 tile: split-K spreads K over SMs
    assert one < 128 * 128 * 16384 / (c.rows * c.cols) / 4
    g1 = MP.gemm_cost(c, "reconf", 256, 2048, 128, count=132).cycles  # 132 tiles = one wave
    g2 = MP.gemm_cost(c, "reconf", 256, 2048, 128, count=133).cycles  # tail wave
    assert g1 < g2 <= 2.0 * g1          # tail wave (softened by the library's choice of a smaller tile)


def test_kernel_us_is_serial_per_kernel():
    from accel_dse.core.evaluate import _sum_ops, _memos
    base = Scenario.from_dict({"model": "llama-3.1-8b", "exec_overlap": "serial"})
    ch = REF_HW["H100-SXM"].chip
    a = evaluate(replace(base, chip=ch))
    b = evaluate(replace(base, chip=replace(ch, kernel_us=1.0)))
    m = a.model
    per_layer = sum(1 for o in __import__("accel_dse.core.ir", fromlist=["x"]).build_rank_ops(m, 0, __import__(
        "accel_dse.core.ir", fromlist=["x"]).Phase("decode", 1, 1, base.serving.ctx))
        if o.kind == "gemm" or (o.kind == "attn" and o.name.startswith("qk")))
    d = (b.step - a.step) / 1e-6
    assert per_layer * len(m.layers) <= d <= per_layer * len(m.layers) + 4     # + head / embed kernels


def test_inflight_fit_relaxes_prefill_only_for_max_rows():
    r = next(x for x in X.rows(X.DATA_R4) if x["id"] == "trt120-b200-gpt-oss-120b-mxfp4-dep2-8192-1024")
    new = X.predict(r, "catalog_kernel")
    old_v = X.gpu_variant("catalog_kernel", 0.0, "wide", False)
    old = X.predict(r, old_v)
    assert new["batch"] > old["batch"] and new["pred"] > old["pred"]
    _, f_old, _ = X._static(r, new["batch"], "catalog_kernel")              # whole-batch prefill: does not fit
    assert not f_old
    s = next(x for x in X.rows() if x["metric"] == "static_tok_s_gpu")     # static rows unchanged
    assert X.predict(s, "catalog_kernel")["pred"] == X.predict(s, old_v)["pred"]


def test_adoption_bands():
    """Headline variant catalog_kernel, refhw default (wide + in-flight fit): MODEL §38 table."""
    def err(rows):
        lr = [math.log(X.predict(r, "catalog_kernel")["pred"] / r["value"]) for r in rows]
        return math.exp(sum(lr) / len(lr)), 100 * sum(map(abs, lr)) / len(lr)
    g, e = err([r for r in X.rows(X.DATA_MOE) if r["metric"] == "max_tok_s_total"])
    assert e < 17.5
    g, e = err(X.rows(X.DATA_R4))
    assert 0.95 < g < 1.08 and e < 20      # 0.172 re-baseline: 0.892 / 26.6 % → 1.016 / 16.8 % (attention orientation)


from tests.pin_legacy import pin_module  # noqa: E402  (0.173: study bands at the 0.172 mapping; current numbers in data/ext_validation.json)
pin_module(globals())

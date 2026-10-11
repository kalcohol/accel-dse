"""0.94: H200 DeepSeek study — published usable capacity, runtime KV fraction, Hopper kernel ceilings (DeepGEMM /
FlashMLA public numbers); all opt-in, bundled into candidate_variant (not default)."""
import math
from dataclasses import replace
from accel_dse.core import extval as X
from accel_dse.core.evaluate import evaluate
from accel_dse.core.refhw import REF_HW
from accel_dse.core.scenario import Scenario


def test_defaults_unchanged():
    assert X.MEM_RT is False
    s = Scenario()
    assert s.mem_cap_GB == 0 and s.kv_mem_frac == 1.0 and s.chip.fmt_ceiling == ()
    d = s.to_dict(); assert "mem_cap_GB" not in d and "kv_mem_frac" not in d and "fmt_ceiling" not in str(d)


def test_mem_cap_and_kv_frac():
    r = next(x for x in X.rows(X.DATA_R4) if x["id"] == "trt120-h200-deepseek-r1-fp8-dep8-1024-1024")
    a = X.scenario(r, "catalog_kernel", phase="decode", batch=2048, ctx=2048, out_len=1024)
    m0 = evaluate(a).stages[0].mem
    assert abs(m0.dram_cap / 2**30 - 144) < 0.1                   # catalog stacks
    b = replace(a, **X._mem_runtime(r, REF_HW["H200-SXM"]))
    assert b.mem_cap_GB == 141.0 and b.kv_mem_frac == 0.9
    m1 = evaluate(b).stages[0].mem
    assert m1.dram_cap == 141e9 and m1.kv_total == m0.kv_total
    # fit honours the 0.9 KV share: find a batch that fits on stacks but not under the runtime view
    fits = lambda s, bb: evaluate(replace(s, serving=replace(s.serving, batch=bb))).fits
    def bmax(sc):
        lo, hi = 8, 1 << 16
        while hi - lo > 8:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if fits(sc, mid) else (lo, mid)
        return lo
    assert bmax(b) < 0.8 * bmax(a)
    for bad in (dict(mem_cap_GB=-1), dict(kv_mem_frac=0), dict(kv_mem_frac=1.5)):
        try:
            replace(a, **bad); assert False
        except ValueError:
            pass


def test_hopper_ceilings():
    assert abs(X.HOPPER_FP8_CEIL - 0.7832) < 1e-3 and abs(X.HOPPER_ATTN_CEIL - 0.667) < 1e-3
    r = next(x for x in X.rows(X.DATA_R4) if x["id"] == "trt120-h200-deepseek-r1-fp8-dep8-1024-1024")
    v = X.candidate_variant()
    s = X.scenario(r, v, phase="prefill", batch=8, prompt=1024, out_len=1)
    assert dict(s.chip.fmt_ceiling)["fp8"] == X.HOPPER_FP8_CEIL
    s0 = replace(s, chip=replace(s.chip, fmt_ceiling=()))
    assert evaluate(s).stages[0].time.t_array > evaluate(s0).stages[0].time.t_array
    rb = next(x for x in X.rows(X.DATA_R4) if x["hw"] == "B200")
    assert X.scenario(rb, v, phase="decode", batch=8, ctx=1024, out_len=8).chip.fmt_ceiling == ()   # Hopper only


def test_candidate_v094_bands():
    v = X.candidate_variant()
    e = lambda rows, var: 100 * sum(abs(math.log(X.predict(r, var)["pred"] / r["value"])) for r in rows) / len(rows)
    dec = [r for r in X.rows(X.DATA_MOE) if r["metric"] == "decode_tok_s_node"]
    assert e(dec, v) < 8                                            # 33.4 → 5.4 %
    one = [r for r in X.rows(X.DATA_MOE) if r["metric"] == "static_tok_s_total"]
    assert e(one, v) > 50                                           # still blocks adoption (MODEL §40)

"""0.173: roofline-ratio invariant across op kinds and chips (the 0.172 bug class, swept).

For every GEMM / attention op of a representative model set (dense, GQA, MLA, MoE, hyper-connection, linear attention,
Hyena, DiT, protein encoder) in prefill / full / decode, on every reference GPU and the abstract presets:

    array time ≤ max(3 × g × FLOP time, operand bytes / DRAM bandwidth) + 2 µs

g is the *geometric* underfill of the better orientation — tile quantisation of the short side over the array rows and
of the long side over the columns (a "wide" schedule gangs E·cols columns, which instances may share).  Anything above that is a mapping artifact
(an operand forced into one orientation, instances serialised on a ganged array, K reduced serially on one tile);
genuine small-M underfill of an output-stationary array (decode GEMV on "os") stays allowed — it is the design effect
the os / os_vec / reconf choice exists to show."""
import math
import sys
sys.path.insert(0, ".")
from accel_dse.core import evaluate as E
from accel_dse.core.catalog import get_model
from accel_dse.core.hardware import CHIPS, System
from accel_dse.core.ir import Phase, Shard, build_rank_ops
from accel_dse.core.refhw import REF_HW

MODELS = ("llama-3.1-8b", "qwen3-8b", "deepseek-v4-pro", "kimi-k3", "qwen3.5-397b-a17b", "qwen3-next-80b-a3b",
          "evo2-7b", "esmfold", "hunyuanvideo", "flux.2-klein-4b", "gpt-oss-120b", "mixtral-8x22b", "step-5-preview")
PHASES = (Phase("prefill", 1, 8192, 0), Phase("full", 1, 4096, 0), Phase("decode", 64, 1, 8192), Phase("decode", 1, 1, 8192))


def _q(x, t):
    return math.ceil(x / t) * t / x


def _g(ch, m, n, count):
    lo, hi = min(m, n), max(m, n)
    if ch.instance_sched == "cta":      # 0.193 (reference GPUs): smallest CTA tile of the guide's set is 64 × 64
        return _q(lo, 64) * _q(hi, 64)
    if ch.instance_sched == "wide":     # independent instances can fill the ganged columns side by side
        return _q(lo, ch.rows) * max(_q(hi, ch.cols), ch.cols * ch.engines / (hi * count))
    return _q(lo, ch.rows) * _q(hi, ch.cols)


def _chips():
    out = [(n, h.chip, h.mem_id) for n, h in REF_HW.items()]
    return out + [(n, CHIPS[n], REF_HW["H100-SXM"].mem_id) for n in ("100T", "1P")]


def _ops(m, ph):
    seen, ops = set(), []
    for li in range(len(m.layers)):
        try:
            lo = build_rank_ops(m, li, ph, Shard())
        except Exception:      # noqa: BLE001 — phase not applicable to this model kind
            return []
        for o in lo:
            key = (o.name, o.kind, o.m, o.k, o.n, o.count, o.w_fmt, o.a_fmt)
            if o.kind in ("gemm", "attn") and key not in seen:
                seen.add(key)
                ops.append(o)
    return ops


def violations():
    bad = []
    for mid in MODELS:
        m = get_model(mid)
        for ph in PHASES:
            ops = _ops(m, ph)
            for cn, ch, mem in _chips():
                sysm = System(ch, mem)
                for o in ops:
                    r = E._op_seconds(o, sysm, "os", m)
                    t, ideal = r[0], r[5]
                    mem = (o.w_params * o.w_bits / 8 + o.act_bytes + o.kv_read) / (sysm.mem.effective_GBps * 1e9)
                    if ideal > 0 and t > max(3 * _g(ch, o.m, o.n, o.count) * ideal, mem) + 2e-6:
                        bad.append((mid, ph.kind, ph.batch, cn, o.name, o.m, o.k, o.n, o.count, round(t / ideal, 1)))
    return bad


def test_roofline_ratio_all_kinds_and_chips():
    bad = violations()
    assert not bad, bad[:10]


def test_pre_173_artifacts_were_caught():
    """The invariant fails with the 0.173 fixes off (wide orientation, wide instance split, split-K)."""
    # 0.193: the reference GPUs now run cta; the artefacts lived on the ≤ 0.192 wide-padded chips
    global REF_HW
    from dataclasses import replace
    keep = REF_HW
    REF_HW = {k: replace(h, chip=replace(h.chip, instance_sched="wide", tile_io="", gemm_ceil_mul=False, wide_pad=True))
              for k, h in keep.items()}
    try:
        with E.model_flags(WIDE_GEMM_ORIENT=False, WIDE_GEMM_SPLIT=False, SPLIT_K=False, WIDE_WAVES=False):
            bad = violations()
    finally:
        REF_HW = keep
    names = {b[4] for b in bad}
    assert {"attn.k", "attn.v"} <= names                      # small-N projections on wide (8–18×)
    assert any("hc_fn" in x for x in names)                   # hyper-connection mixers: serial K / N = 24
    assert any(x.startswith("lin_") for x in names)           # chunked linear attention: instances serialised

"""0.196: GEMM roofline invariant for every mapping / variant (fuzzed shapes); occupancy fill + SMEM-bound small tiles.

array time ≥ max(FLOP time at peak, unique operand + output bytes / DRAM bandwidth) for every tile-I/O variant (the
array term carries the kernel's DRAM traffic there), and ≥ the FLOP time for the legacy (stage-level DRAM) variants."""
import random
import sys
from dataclasses import replace
sys.path.insert(0, ".")
from accel_dse.core import evaluate as E, extval as X, mapping as M
from accel_dse.core.catalog import get_model
from accel_dse.core.hardware import System
from accel_dse.core.ir import Op

VARS = ("default", "default_occ", "default_tioauto", "default_0194", "default_0192")
HW = ("H100-SXM", "B200", "MI300X", "A100-SXM-80GB")
FMTS = (("bf16", "bf16", 16.0), ("fp8", "fp8", 8.0), ("bf16", "fp8", 8.0))


def _sys(row, v):
    sc = X.scenario(row, v, phase="decode", batch=1, ctx=1, out_len=1)
    s = System(sc.chip, sc.mem_id, sc.mem_eff)
    if sc.chip.tile_io and not sc.chip.tile_io_Bpc:
        s = replace(s, chip=replace(sc.chip, tile_io_Bpc=s.dram_GBps / sc.chip.freq_ghz))
    return s


def violations(n=120, seed=196):
    md = get_model("qwen3-8b")
    rows = {r["hw"]: r for r in X.rows() + X.rows(X.DATA_R3)}
    rnd = random.Random(seed)
    shapes = [(rnd.choice((1, 4, 64, 128, 512, 2048, 8192)), rnd.choice((128, 512, 1536, 4096, 7168, 18432)),
               rnd.choice((24, 64, 128, 512, 4096, 16384)), rnd.choice((1, 8, 16, 64, 128)), rnd.choice(FMTS))
              for _ in range(n)]
    bad = []
    for v in VARS:
        for hn in HW:
            s = _sys(rows[hn], v)
            for m, k, nn, c, (af, wf, wbits) in shapes:
                o = Op("g", "gemm", 0, m, k, nn, count=c, w_fmt=wf, a_fmt=af, w_bits=wbits)
                r = E._op_seconds(o, s, "reconf", md)
                t, ideal = r[0], r[5]
                ab = M._fmt(af).bytes
                fl = (m * k * ab + k * nn * wbits / 8 + m * nn * max(2.0, ab)) * c / (s.dram_GBps * 1e9) \
                    if s.chip.tile_io_Bpc else 0.0
                if t < max(ideal, fl) * (1 - 1e-9):
                    bad.append((v, hn, m, k, nn, c, af, round(t / max(ideal, fl), 3)))
    return bad


def test_roofline_every_variant():
    bad = violations()
    assert not bad, bad[:10]


def test_pre_196_bypass_caught():
    """≤ 0.195: tile-I/O variants ran below the DRAM roofline (occupancy worst, ~0.1×)."""
    with E.model_flags(CTA_DRAM_FLOOR=False, CTA_OCC_FILL=False, CTA_SMEM_BW=False):
        bad = violations(n=40)
    vs = {b[0] for b in bad}
    assert {"default", "default_occ"} <= vs
    assert min(b[-1] for b in bad if b[0] == "default_occ") < 0.3


def test_mla_absorbed_bmm_h100():
    from accel_dse.core.ir import Phase, Shard, build_rank_ops
    m = get_model("deepseek-v3")
    o = next(o for li in range(len(m.layers)) for o in build_rank_ops(m, li, Phase("decode", 64, 1, 8192), Shard())
             if o.name == "attn.kv_b.uk")
    rows = {r["hw"]: r for r in X.rows()}
    s = _sys(rows["H100-SXM"], "default_occ")
    t = E._op_seconds(o, s, "reconf", m)[0]
    ab, wb = M._fmt(o.a_fmt).bytes, o.w_bits / 8
    fl = (o.m * o.k * ab + o.k * o.n * wb + o.m * o.n * max(2.0, ab)) * o.count / (s.dram_GBps * 1e9)
    assert t >= fl * (1 - 1e-9) and t < 1.05 * fl


def test_smem_bound_small_tiles():
    from accel_dse.core.refhw import REF_HW
    b = REF_HW["B200"].chip
    assert b.smem_Bpc_sm == 128.0
    # B200 bf16, 1 SM: 64×64 needs 256 B per 4096 MACs → SMEM-bound at 128 B/clk (50 %); 128×256 is not
    macs = b.rows * b.cols
    assert (64 * 2 + 64 * 2) / b.smem_Bpc_sm > 64 * 64 / macs
    assert (128 * 2 + 256 * 2) / b.smem_Bpc_sm <= 128 * 256 / macs
    assert X.VARIANTS["default"]["tile_io"] == "dram" and X.VARIANTS["default_occ"]["tile_io"] == "occupancy"
    assert X.VARIANTS["default_occ"]["trtllm_fp4"]

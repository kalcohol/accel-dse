"""0.202 standing leak audit: GPU-specific mechanisms (0.176–0.201) must not move the NPU design path, or carry a
documented reason why they apply to an NPU.  Add every new GPU-only mechanism here."""
import sys
from contextlib import contextmanager
from dataclasses import replace, fields
sys.path.insert(0, ".")
from accel_dse.core import evaluate as E, extval as X, schedule as S
from accel_dse.core.hardware import Chip, CHIPS, Link
from accel_dse.core.scenario import Scenario

NPU = ("100T", "1P")

# Chip fields that carry a GPU mechanism; NPU presets must leave them at their defaults (gate = the field itself)
GPU_CHIP_FIELDS = {
    "kernel_us": "per-kernel device floor + launch gap (CUDA graph / eager, extval TRACE_GAP_US)",
    "kernel_vec": "stand-alone vector ops as kernels",
    "gpu_glue": "GPU-runtime glue kernels (ir.glue_ops)",
    "attn_sched": "FMHA / FlashMLA CTA grid",
    "tile_io": "CTA tile prologue / occupancy",
    "tile_io_Bpc": "CTA DRAM floor / L2 activations",
    "occ_smem_kib": "CUDA occupancy", "occ_reg_kib": "CUDA occupancy", "occ_threads": "CUDA occupancy",
    "smem_Bpc_sm": "SMEM bandwidth per SM",
    "fmt_ceiling": "achieved-peak kernel ceilings (TRT-LLM FP4 rates, netting)",
    "gemm_ceil_mul": "achieved-peak multiplicative netting",
    "wide_pad": "≤ 0.192 wide padding",
    "gpu": "GPU reference marker (gates WIDE_WAVES refill)",
}

# module flags of GPU-only mechanisms: flipping them must leave NPU results bit-identical
GPU_FLAGS = [dict(WIDE_WAVES=False), dict(CTA_DRAM_FLOOR=False), dict(CTA_SMEM_BW=False), dict(CTA_OCC_FILL=False),
             dict(CTA_L2_ACT=False), dict(CUSTOM_AR=False), dict(NCCL_AR_LAT=False), dict(ROCM_AR=False),
             dict(CTA_SPLIT_K=(1,))]
EXTVAL_FLAGS = dict(FMT_CEIL_REPLACE=True, NET_FULL=False)

# reach the NPU path on purpose — reason documented (MODEL §149)
NPU_APPLICABLE = {
    "SPLIT_K": "0.173 split-K over core groups for output-stationary arrays (non-cta); an NPU dataflow choice 「推导」",
    "FUSE_QKV": "IR transform (concatenated weights); work-conserving, compiler choice",
    "W_BITS_ZERO_FIX": "bug fix (unset w_bits ≠ 0-byte weights)",
    "idle_frac_tdp": "energy preset only with tdp_W; output labels it a GPU whole-card value (idle_source_note)",
    "pJ_bit_phy_* / mm2_per_dq_*": "DRAM interface PHY per pin — memory-interface physics, not GPU-specific",
}


def _scns():
    for n in NPU:
        for ph in ("decode", "prefill"):
            for tp in (1, 4):
                s = Scenario(chip=CHIPS[n])
                yield replace(s, serving=replace(s.serving, phase=ph), layout=replace(s.layout, tp=tp))


def _key(r):
    return (r.tick, r.step, r.ttft, r.tpot, r.throughput, [st.flops_u for st in r.stages])


@contextmanager
def _xv(**kw):
    old = {k: getattr(X, k) for k in kw}
    for k, v in kw.items():
        setattr(X, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(X, k, v)


def test_npu_presets_leave_gpu_fields_default():
    dflt = {f.name: f.default for f in fields(Chip)}
    for n in NPU:
        for k in GPU_CHIP_FIELDS:
            assert getattr(CHIPS[n], k) == dflt[k], (n, k)
        assert CHIPS[n].instance_sched != "cta"
        assert CHIPS[n].kernel_us == 0.0      # NPU dispatch: statically scheduled graph, no per-op launch 「假设」


def test_gpu_chip_fields_are_live_gates():
    s = Scenario(chip=CHIPS["100T"])
    base = _key(E.evaluate(s))
    assert _key(E.evaluate(replace(s, chip=replace(s.chip, kernel_us=2.0)))) != base   # the gate is real, not dead


def test_gpu_flags_inert_on_npu():
    for s in _scns():
        base = _key(E.evaluate(s))
        for kw in GPU_FLAGS:
            with E.model_flags(**kw):
                assert _key(E.evaluate(s)) == base, (s.chip.name, s.serving.phase, s.layout.tp, kw)
        with _xv(**EXTVAL_FLAGS):
            assert _key(E.evaluate(s)) == base


def test_npu_link_alpha_and_rocm_floor():
    assert S.collective_seconds("allreduce", 2048, 8, Scenario().link)[1] == Scenario().link.alpha_us * 1e-6
    r = Link(448.0, 3.0, ar_lat="rocm")
    assert abs(S.collective_seconds("allreduce", 2048, 2, r)[1] - 7.26e-6) < 1e-12
    assert abs(S.collective_seconds("allreduce", 2048, 8, r)[1] - 12.29e-6) < 1e-12
    with E.model_flags(ROCM_AR=False):
        assert abs(S.collective_seconds("allreduce", 2048, 8, r)[1] - 6.8e-6) < 1e-12


def test_documented_npu_applicable_list():
    assert set(NPU_APPLICABLE) >= {"SPLIT_K", "FUSE_QKV", "idle_frac_tdp"}
    from accel_dse.core.energy import ENERGY_PRESETS
    assert "H100" in ENERGY_PRESETS["tpuv4i_7nm"]["fields"]["idle_frac_tdp"][2]


def test_w_bits_legacy_branch_handles_unset():
    from accel_dse.core.mapping import gemm_cost
    with E.model_flags(W_BITS_ZERO_FIX=False):
        assert gemm_cost(CHIPS["100T"], "os", 64, 1024, 1024).cycles > 0


def test_wide_waves_leak_fixed():
    # ≤ 0.201 the GPU wave refill reached NPU "auto"→"wide" plans (100T prefill −1.9 %)
    s = Scenario(chip=CHIPS["100T"])
    s = replace(s, serving=replace(s.serving, phase="prefill"))
    base = _key(E.evaluate(s))
    with E.model_flags(GPU_WAVES_ONLY=False):
        old = _key(E.evaluate(s))
    assert old != base and old[0] < base[0]
    from accel_dse.core.refhw import REF_HW
    assert all(h.chip.gpu for h in REF_HW.values())

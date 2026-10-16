"""Evaluate one Scenario: per-stage op graphs → mapping → memory plan → schedule.

Everything goes through the per-rank IR; a single card is Layout().
"""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, field, replace
import contextlib
from functools import lru_cache
from types import SimpleNamespace

from .catalog import get_model
from .dtypes import fmt as _fmt
from .hardware import System
from .domain import ResolvedWorkload, resolve_workload
from .ir import (Op, Phase, Shard, _cdiv, build_rank_ops, embed_ops, full_io_ops, head_ops, layer_repeat,
                 mtp_ops)
from .mapping import gemm_cost, vector_seconds
from .memo import layer_groups, model_cache
from .memplan import RUNTIME_RESERVE, MemPlan, act_stream, plan, stage_storage, step_dram_bytes, touched
from .model import ModelSpec, with_formats
from .parallel import Layout, plan_stages, plan_stages_balanced
from .pipeline import TILING, pipeline_for, stored_bytes, te_layers, text_ops, vae_ops
from .scenario import Scenario
from .vision import audio_ops, expand_images, image_grid, video_grid_of, vision_act_peak, vision_ops
from .ir import dspark_ops, red_bytes, skew_from_load
from . import fabric
from . import overlap as _ovl
from . import mapping as _mapping
from .schedule import (StageTime, collective_seconds, collective_wire, fabric_collective, p2p_tier,
                       spec_expected_tokens, wire_single)


@dataclass
class StageResult:
    index: int
    layers: tuple[int, int]
    time: StageTime
    mem: MemPlan
    dram: dict
    convert_elems: float
    ops: list[Op] = field(default_factory=list, repr=False)
    flops_u: float = 0.0    # useful FLOPs of one rank (ops replicated on r ranks count 1/r) — request-FLOP KPI
    sram_bytes: float = 0.0  # bytes through the SRAM ↔ datapath port of one rank per tick (0.47.1 action counts)
    ideal_w: float = 0.0     # 0.63: array s at 100 % of the mean rank's useful work (Op.share) — energy MAC count
    vec_w: float = 0.0       # 0.63: vector s of the mean rank's token work (excl. weight conversion)
    conv_w: float = 0.0      # 0.63: vector s of weight conversion (once per pass on every busy rank)
    exp_acts: dict = field(default_factory=dict)  # 0.65: routed-expert part {sram, conv_w, link_frac} (idle EP ranks)
    pipe_ctx: tuple | None = field(default=None, repr=False, compare=False)   # 0.102: inputs of the pipe replay
    exp_route: tuple = field(default=(), repr=False, compare=False)  # 0.103: (E, top-k, tokens, EP, skew) of MoE layers


@dataclass
class Result:
    scenario: Scenario
    model: ModelSpec
    stages: list[StageResult]
    tick: float               # slowest stage time per micro-batch
    step: float               # one decode step (all micro-batches) or prefill latency
    tokens_per_step: float    # accepted tokens per sequence per step
    tpot: float               # s / output token / sequence (decode)
    ttft: float               # prefill latency (prefill phase)
    throughput: float         # tokens / s per replica (decode: output; prefill: prompt tokens)
    per_card: float
    fits: bool
    bound: str
    heaviest_stage: int
    warnings: list[str]
    microbatches: int
    latency: float = 0.0      # s per request: decode TPOT, prefill TTFT; video: one clip; protein: one batch
    workload: ResolvedWorkload | None = None
    pipeline: dict | None = None   # video (0.44): text-encoder / VAE-decode time, FLOP, storage (None = DiT only)
    moe_skew: float = 1.0          # effective EP load skew used (0.49)
    vision: dict | None = None     # VLM (0.62): images, image tokens, encoder time / FLOP / storage (None = text-only)
    host: dict | None = None       # 0.164 (opt-in): host / runtime step time {s, device_s, exposed_s, host_bound, …}
    step_cache: dict | None = None  # gen (0.167): TeaCache-style skipped steps
    prefix_kv: dict | None = None  # gen (0.116): text-prefix K/V reuse — extract / cached step times and FLOP, cache size

    @property
    def domain(self) -> str:
        return self.model.domain

    @property
    def slo_ok(self) -> bool:
        """Latency SLO of the scenario: LLM decode TPOT (prefill: none), video clip latency, protein batch latency."""
        sv, w = self.scenario.serving, self.scenario.workload
        if self.model.domain == "gen":
            return self.latency <= w.clip_slo_s
        if self.model.domain in ("protein", "genomics"):
            return self.latency * 1e3 <= self._seq_slo_ms
        return self.tpot * 1e3 <= sv.tpot_slo_ms

    @property
    def _seq_slo_ms(self) -> float:
        w = self.scenario.workload
        return w.fold_slo_s * 1e3 if self.model.is_pair else w.seq_slo_ms

    def _tflop_per_request(self) -> float:
        """Useful FLOPs of one request over the whole replica: per-rank useful FLOPs × the TP·SP ranks of a stage,
        per sequence of the rank's micro-batch, × forward sequences per request × steps (0.45: was per rank)."""
        w, lay = self.workload, self.scenario.layout
        # 0.63: flops_u is the mean rank's useful work (Op.share) → × all TP·SP·DP ranks / sequences of the micro-batch
        b_mb = _cdiv(self.scenario.serving.batch * w.seqs_per_request, self.microbatches)
        if self.prefix_kv:      # 0.116: one extract step + (steps − 1) cached steps
            pk = self.prefix_kv
            return (pk["extract_tflop_seq"] + (w.steps - 1) * pk["cached_tflop_seq"]) * w.seqs_per_request
        per_seq = sum(st.flops_u for st in self.stages) * lay.tp * lay.sp * lay.dp * lay.cp / b_mb
        if self.step_cache:     # 0.167: skipped steps run embed + head + residual add only
            sc = self.step_cache
            return (per_seq / 1e12 * (w.steps - sc["skip"]) + sc["skip"] * sc["tflop_seq_skip"]) * w.seqs_per_request
        return per_seq * w.seqs_per_request * w.steps / 1e12

    def domain_summary(self) -> dict:
        """Domain KPIs of a non-autoregressive model (empty for LLMs)."""
        w = self.workload
        if w is None:
            return {}
        cards = self.scenario.layout.cards
        b = self.scenario.serving.batch
        heavy = self.stages[self.heaviest_stage]
        period = (self.pipeline or {}).get("period_s", self.latency)   # 0.47 overlap: steady-state request period
        d = {"unit": w.unit, "latency_s": self.latency, "requests_per_s": b / period, "period_s": period,
             "units_per_s": self.throughput, "units_per_s_card": self.per_card, "workload": w.info,
             "forward_ms": self.tick * max(self.microbatches, self.scenario.layout.pp) * 1e3,
             "seqs_per_forward": b * w.seqs_per_request, "act_GiB": heavy.mem.act_total / 2**30,
             "tflop_per_request": self._tflop_per_request()}
        if w.kind == "gen":
            pl = self.pipeline
            denoise = pl["denoise_s"] if pl else self.latency
            if pl:
                d["tflop_per_request"] += pl["tflop"] / b
                d["dit_tflop_per_request"] = d["tflop_per_request"] - pl["tflop"] / b
            d.update(pipeline=pl, denoise_s=denoise)
            if self.prefix_kv:          # 0.116 (key only present when the reuse is active)
                d["prefix_kv"] = self.prefix_kv
            if self.step_cache:         # 0.167 (key only present when step skipping is active)
                d["step_cache"] = self.step_cache
            d.update(clip_s=self.latency, s_per_frame=self.latency / w.units, step_ms=denoise / w.steps * 1e3,
                     frames_per_s_card=self.per_card, clips_per_hour_card=3600 * b / period / cards,
                     realtime_x=(w.info["video_s"] * b / self.latency) if w.info.get("video_s") else None,
                     slo_s=self.scenario.workload.clip_slo_s)
        elif w.kind == "genomics":     # 0.106: windows of a long record (overlap re-read, new bases counted)
            d.update(batch_ms=self.latency * 1e3, windows_per_s_card=self.per_card, seq_per_s_card=self.per_card,
                     tokens_per_s_card=self.per_card * w.info["seq_len"],
                     bp_per_s_card=self.per_card * w.info["new_bp_per_window"], slo_ms=self._seq_slo_ms)
        else:
            d.update(batch_ms=self.latency * 1e3, seq_per_s_card=self.per_card,
                     residues_per_s_card=self.per_card * w.info["seq_len"], slo_ms=self._seq_slo_ms)
        return d

    def summary(self) -> dict:
        s = self.scenario
        heavy = self.stages[self.heaviest_stage]
        return {
            "model": self.model.id, "layout": s.layout.label, "cards": s.layout.cards, "mapping": s.mapping,
            "phase": s.serving.phase, "batch": s.serving.batch,
            "tpot_ms": self.tpot * 1e3, "ttft_ms": self.ttft * 1e3, "step_ms": self.step * 1e3,
            "tok_s": self.throughput, "tok_s_card": self.per_card, "fits": self.fits, "bound": self.bound,
            "heaviest_stage": self.heaviest_stage, "dram_need_GiB": heavy.mem.dram_need / 2**30,
            "residency": heavy.mem.residency, "microbatches": self.microbatches, "warnings": self.warnings,
            "array_util": heavy.time.array_util, "domain": self.model.domain, "slo_ok": self.slo_ok,
            "latency_ms": self.latency * 1e3, **({"gen": self.domain_summary()} if self.workload else {}),
            **({"moe_skew": self.moe_skew} if self.moe_skew != 1.0 else {}),
            **({"vision": {k: v for k, v in self.vision.items() if k != "acts"}} if self.vision else {}),
            **({"host": dict(self.host)} if self.host else {}),
            **({"decision": self.decision()} if s.serving.labels else {}),
        }

    def balance(self) -> dict:
        """0.205: machine balance vs workload arithmetic intensity.  Chip: bf16 peak FLOP/s ÷ DRAM bytes/s (raw, and
        at the scenario's effective bandwidth).  Workload: FLOPs ÷ DRAM bytes of the heaviest stage's step (what the
        roofline sees).  Warn when they differ by more than BALANCE_WARN× and say which side idles."""
        from .hardware import System
        sysm = System(self.scenario.chip, self.scenario.mem_id, self.scenario.mem_eff)
        peak = self.scenario.chip.peak_tflops_bf16 * 1e12
        raw, eff = sysm.mem.raw_GBps * 1e9, sysm.dram_GBps * 1e9
        t = self.stages[self.heaviest_stage].time
        ai = t.flops / t.dram_bytes if t.dram_bytes > 0 else float("inf")
        bal = peak / eff if eff > 0 else float("inf")
        r = bal / ai if ai > 0 else float("inf")
        warn = None
        if r > BALANCE_WARN:
            warn = (f"算力闲置 compute idle：芯片 {bal:.0f} FLOP/B（有效带宽）比负载算术强度 {ai:.1f} FLOP/B 高 {r:.0f}×，"
                    f"步受 DRAM 限制，多余算力浪费 · chip balance {r:.0f}× the workload's intensity — compute is wasted"
                    + ("（小 batch decode 的常态 · expected for small-batch decode）" if self.scenario.serving.phase == "decode" else ""))
        elif r < 1 / BALANCE_WARN:
            warn = (f"带宽闲置 bandwidth idle：负载算术强度 {ai:.0f} FLOP/B 比芯片 {bal:.0f} FLOP/B 高 {1 / r:.0f}×，"
                    f"步受算力限制，DRAM 带宽浪费 · workload intensity {1 / r:.0f}× the chip balance — bandwidth is wasted")
        return {"chip_flop_per_byte_raw": peak / raw if raw > 0 else None, "chip_flop_per_byte": bal,
                "workload_flop_per_byte": ai, "ratio": r, "threshold": BALANCE_WARN,
                "idle": "compute" if r > BALANCE_WARN else "bandwidth" if r < 1 / BALANCE_WARN else None, "warning": warn,
                "peak_tflops_bf16": peak / 1e12, "dram_GBps_raw": raw / 1e9, "dram_GBps_eff": eff / 1e9}

    def walls(self) -> dict:
        """0.203: which wall one step hits — the heaviest stage's time split into compute / DRAM / collective
        (bandwidth, latency) / dispatch / fixed overhead.  0.204: "fixed_overhead" (was "static") = the fixed
        per-layer overhead time (layer_overhead_us) — a time term, not static power.  The overlapped window (stage
        total − t_sync) is charged to the resource that sets it (the others are hidden behind it, shown as raw); the
        serial t_sync is split into collective latency, dispatch (Chip.dispatch, or GPU kernel_us) and fixed overhead.
        Exact for exec_overlap "stage"; for the other overlap modes the window attribution is the dominant-resource
        reading."""
        t = self.stages[self.heaviest_stage].time
        tot = t.total
        disp = t.t_disp + max(0.0, t.t_kern - t.t_fixed)
        fixed = t.t_fixed
        lat = max(0.0, t.t_sync - disp - fixed)
        raw = {"compute": t.t_compute, "dram": max(t.t_dram, t.t_slc), "collective_bw": t.t_link}
        win = max(0.0, tot - t.t_sync)
        wall = max(raw, key=raw.get) if win > 0 else "none"
        exp = {k: (win if k == wall else 0.0) for k in raw}
        exp.update(collective_lat=lat, dispatch=disp, fixed_overhead=fixed)
        top = max(exp, key=exp.get) if tot > 0 else "none"
        return {"stage": self.heaviest_stage, "stage_ms": tot * 1e3, "step_ms": self.step * 1e3,
                "raw_ms": {**{k: v * 1e3 for k, v in raw.items()}, "collective_lat": lat * 1e3, "dispatch": disp * 1e3,
                           "fixed_overhead": fixed * 1e3},
                "exposed_ms": {k: v * 1e3 for k, v in exp.items()},
                "share": {k: (v / tot if tot > 0 else 0.0) for k, v in exp.items()},
                "wall": top, "overlap": t.overlap, "exact": t.overlap == "stage",
                "dispatch_mode": self.scenario.chip.dispatch or ("gpu_kernel" if self.scenario.chip.kernel_us else "ideal")}

    def decision(self) -> dict:
        """0.191 decision readout: one forward per question (prefill of prompt + options), no decode loop."""
        sv = self.scenario.serving
        think_s, think_note = 0.0, ""
        if sv.think:
            # 0.192: the thought is a decode run after the prompt prefill — T steps at ctx growing prompt → prompt + T,
            # integrated at 4 segment midpoints; the closing readout is one more step on the label rows (≈ a decode
            # step, inside the thought's last segment within the 4-point error)
            T, n = sv.think_tokens, 4
            for i in range(n):
                c = sv.prompt + round((i + 0.5) * T / n)
                d = self.scenario.replace("serving.think", False).replace("serving.labels", 0) \
                    .replace("serving.phase", "decode").replace("serving.ctx", c)
                think_s += evaluate(d).step * T / n
            think_note = (f"思考 {T} token 解码（{'默认 = StartLux-Decision v1.1 AutoThink 卡片均值 596，gate 0.9 「二手」' if T == 596 else '用户填写'}）"
                          "+ 读出；同批 batch 行并行")
        total = self.step + think_s
        dps = sv.batch / total if total > 0 else 0.0
        return {"labels": sv.labels, "latency_ms": total * 1e3, "readout_ms": self.step * 1e3,
                "think_tokens": sv.think_tokens if sv.think else 0, "think_ms": think_s * 1e3, "think_note": think_note,
                "decisions_per_s": dps,
                "decisions_per_s_card": dps / self.scenario.layout.cards, "prompt": sv.prompt,
                "basis": "每个决策 = 一次前向（prompt + 选项）+ 末位置对 ≤ labels 个选项 token 的 logits（输出头按行取），"
                         "无解码循环；批内各行即各问题（StartLux-Decision 推理包同一前向内多问题）"}


@lru_cache(maxsize=64)
def _attn_chip(ch):
    return replace(ch, instance_sched=ch.attn_sched, attn_sched="")


def union_routing(sd, sp, f: float):
    """0.103: (decode routing, routing of decode + chunk tokens) of the co-located iteration — the chunk's tokens
    touch more distinct experts (uniform-routing expectation, ir.moe_routing). None for dense models."""
    if not sd.exp_route or not sp.exp_route:
        return None
    from .ir import moe_routing
    E, k, td, ep, sk = sd.exp_route
    tu = td + max(1, round(f * sp.exp_route[2]))
    return moe_routing(E, k, td, ep, sk), moe_routing(E, k, tu, ep, sk)


def extra_expert_bytes(sd, sp, f: float) -> float:
    """0.103: DRAM bytes of the experts the chunk's tokens touch beyond the decode batch's (per stage)."""
    ru = union_routing(sd, sp, f)
    if ru is None or not ru[0].hit:
        return 0.0
    return sd.dram.get("exp_dram", 0.0) * (ru[1].hit / ru[0].hit - 1.0)


def _exp_route(m, ph, sh) -> tuple:
    """0.103: routing inputs of the stage's MoE layers (for the co-located chunk's extra experts)."""
    if not m.is_moe:
        return ()
    f = next(L.ffn for L in m.layers if L.ffn.kind == "moe")
    return (f.n_experts, f.top_k, ph.tokens, sh.ep, ph.skew)


def _vec_seconds(ch, op) -> float:
    """Vector seconds of an op. 0.103 (opt-in Chip.sfu_per_cycle): transcendentals issue on a separate SFU pipe
    alongside the lanes — the op takes the slower of the two."""
    t = vector_seconds(ch, op.vec)
    if ch.sfu_per_cycle and op.sfu:
        t = max(t, op.sfu / (ch.sfu_per_cycle * ch.freq_ghz * 1e9))
    return t


def _convert_seconds(ch, elem_ops: float) -> float:
    """0.95: format conversion runs on the dequant path — formats.dequant_elems_per_cycle (elements / cycle / chip)
    when set, else the vector lanes (the pre-0.95 behaviour; the field was validated but never read)."""
    d = ch.formats.dequant_elems_per_cycle
    return elem_ops / (d * ch.freq_ghz * 1e9) if d else vector_seconds(ch, elem_ops)


def _shape_cap(fc: tuple, op: Op, fmt: str) -> float | None:
    """0.143: published per-shape kernel throughput (Chip.fmt_ceiling keys "gemm:<fmt>@<N>x<K>" dense and
    "gemm:<fmt>@g<N>x<K>" grouped): the entry nearest the op's (N, K) in log space (grouped ops prefer grouped
    entries).  None when the chip carries no shape entries for this kind / format."""
    if op.kind != "gemm" or not fc:
        return None
    pre = f"gemm:{fmt}@"
    pts = [(k[len(pre):], v) for k, v in fc if k.startswith(pre)]
    if not pts:
        return None
    grp = [(k[1:], v) for k, v in pts if k.startswith("g")]
    den = [(k, v) for k, v in pts if not k.startswith("g")]
    use = grp if (op.count > 1 and grp) or not den else den
    ln, lk = math.log(max(1, op.n)), math.log(max(1, op.k))

    def dist(e):
        n_, k_ = (float(x) for x in e[0].split("x"))
        return (math.log(n_) - ln) ** 2 + (math.log(k_) - lk) ** 2, e[0]
    return min(use, key=dist)[1]


ATTN_ORIENT = True     # 0.172 default (False = ≤ 0.171 behaviour, kept for the re-baseline record / tests)


@contextlib.contextmanager
def model_flags(**kw):
    """0.172 / 0.173: run with module-level mapping flags overridden (ATTN_ORIENT, WIDE_GEMM_ORIENT, WIDE_GEMM_SPLIT,
    SPLIT_K, WIDE_GEMM_AUTO); historical guards pin pre-fix numbers with it.  Clears the stage memos."""
    from . import memo, mapping as _mp, memplan as _mpl
    from . import refhw as _rh
    kw = dict(kw)
    refhw_wide = kw.pop("REFHW_WIDE", None)      # 0.193: reference GPUs back on the ≤ 0.192 wide schedule
    cta_split = kw.pop("CTA_SPLIT_K", None)      # 0.193: mapping.SPLIT_K (cta split-K set)
    from . import ir as _ir, schedule as _sch
    mods = {"ROCM_AR": _sch, "GPU_AR_ONLY": _sch, "CUSTOM_AR": _sch, "NCCL_AR_LAT": _sch, "FUSE_QKV": _ir, "WIDE_WAVES": _mp, "GPU_WAVES_ONLY": _mp, "STATE_FIRST": _mpl, "CTA_DRAM_FLOOR": _mp, "CTA_SMEM_BW": _mp, "CTA_OCC_FILL": _mp, "CTA_L2_ACT": _mp, "W_BITS_ZERO_FIX": _mp}   # 0.192: flags living in other modules
    tgt = {k: (vars(mods[k]) if k in mods else globals()) for k in kw}
    old = {k: tgt[k][k] for k in kw}
    for k, v in kw.items():
        tgt[k][k] = v
    old_rh, old_sk = dict(_rh.REF_HW), _mp.SPLIT_K
    if refhw_wide:
        for k, h in old_rh.items():
            _rh.REF_HW[k] = replace(h, chip=replace(h.chip, instance_sched="wide", tile_io="", gemm_ceil_mul=False))
    if cta_split is not None:
        _mp.SPLIT_K = cta_split
    memo._TABLES.clear(); _mp._gemm_cost.cache_clear()
    try:
        yield
    finally:
        for k, v in old.items():
            tgt[k][k] = v
        _rh.REF_HW.update(old_rh); _mp.SPLIT_K = old_sk
        memo._TABLES.clear(); _mp._gemm_cost.cache_clear()


LEGACY_0171 = dict(ATTN_ORIENT=False, WIDE_GEMM_ORIENT=False, WIDE_GEMM_SPLIT=False, SPLIT_K=False)
LEGACY_0191 = dict(WIDE_WAVES=False, STATE_FIRST=False)   # 0.192
# 0.193: ≤ 0.192 reference GPUs (wide, padded) and cta split-K set
LEGACY_0192 = dict(WIDE_WAVES=False, REFHW_WIDE=True, CTA_SPLIT_K=(1, 2, 4, 8, 16))
LEGACY_0172 = dict(WIDE_GEMM_ORIENT=False, WIDE_GEMM_SPLIT=False, SPLIT_K=False)


def attn_orient(on: bool):
    """0.172: ``on`` False = the ≤ 0.171 mapping (0.173: also without the 0.173 wide / split-K alternatives)."""
    return model_flags(ATTN_ORIENT=True) if on else model_flags(**LEGACY_0171)


WIDE_GEMM_ORIENT = True   # 0.173 default
WIDE_GEMM_SPLIT = True    # 0.173 default
WIDE_GEMM_AUTO = False    # 0.173 diagnostic: also the core-group (auto) plan for count = 1


def _wide_alt(ch, org, op, c):
    """0.173: the "wide" schedule gangs every engine into one rows × (E·cols) array (GPU references, MODEL §26).  Two
    shapes it forced into an unphysical mapping (the 0.172 bug class, sweep in MODEL §119):
    * small N (k / v projections N = 1024, MLA / hyper-connection mixers N = 4–24): N filled N of E·cols columns.  A
      GEMM library computes Cᵀ = Wᵀ·Xᵀ just as well (cuBLAS row-major is exactly that), so take the cheaper
      orientation;
    * many small instances (grouped MoE experts with few tokens, chunked linear-attention blocks): each instance
      occupied the whole ganged array in turn.  Grouped / batched GEMM kernels spread instances over SMs (CUTLASS
      grouped GEMM tiles every expert's output; the FlashAttention grid is batch × head × q-block) — take the core-group
      (auto) plan, which spreads instances and splits each over the remaining groups, when cheaper."""
    if WIDE_GEMM_ORIENT and op.n < op.m:
        # swapped roles: the weight is now the streamed operand.  gemm_exec's conversion flags follow the operands, so
        # converted element counts are unchanged (w side k·n' = k·m activations, a side m'·k = weight elements); the
        # per-pass weight-dequant tag (convert_w) is kept from the native orientation; packed-format scale bits
        # (w_bits above the format's) are dropped on the swapped side (≤ 10 %, feed term only)
        t = gemm_cost(ch, org, op.n, op.k, op.m, count=op.count, w_fmt=op.a_fmt, a_fmt=op.w_fmt)
        if t.cycles < c.cycles:
            c = replace(t, convert_w=c.convert_w)
    if WIDE_GEMM_AUTO:
        t = gemm_cost(replace(ch, instance_sched="auto"), org, op.m, op.k, op.n, count=op.count, w_fmt=op.w_fmt,
                      a_fmt=op.a_fmt, w_bits=op.w_bits)
        if t.cycles < c.cycles:
            c = t
    if WIDE_GEMM_SPLIT and op.count > 1:
        t = gemm_cost(replace(ch, instance_sched="auto"), org, op.m, op.k, op.n, count=op.count, w_fmt=op.w_fmt,
                      a_fmt=op.a_fmt, w_bits=op.w_bits)
        if t.cycles < c.cycles:
            c = t
    return c


SPLIT_K = True            # 0.173 default


def _splitk_alt(ch, org, m, k, n, count, w_fmt, a_fmt, w_bits, c):
    """0.173: split-K over core groups.  Output-stationary arrays (all non-"cta" schedules) reduce K serially in one
    tile, so a tiny output with a long K (hyper-connection mixers 1 × 28672 × 24, shared-expert gates N = 1, routers
    at decode) took K cycles on one tile while every other core idled (deepseek-v4-pro decode: 3 such ops / layer ×
    61 layers ≈ 5 ms / step on 100T).  GPU libraries split K over CTAs (cuBLAS / CUTLASS split-K, the 0.92 cta model
    already does); here s groups each take K / s on E / s engines (port / accumulators shared out as the auto plan
    does), then the s partial m × n tiles are summed on the vector lanes 「推导」."""
    if ch.engines < 2 or m * n * count >= ch.rows * ch.cols * ch.engines or k < 256:
        return c
    best = c
    s = 2
    while s <= ch.engines and k // s >= 64:
        sub = replace(ch, engines=ch.engines // s, sram_port_Bpc=ch.port_Bpc / s, sram_port_Bpc_per_engine=None,
                      acc_kib=ch.acc_total_kib / s, acc_kib_per_engine=None)
        one = gemm_cost(sub, org, m, -(-k // s), n, count=count, w_fmt=w_fmt, a_fmt=a_fmt, w_bits=w_bits)
        red = m * n * count * (s - 1) / ch.lanes
        if one.cycles + red < best.cycles:
            # conversion work is the operands' (every K slice converted once, by its own group): unchanged
            best = replace(one, cycles=one.cycles + red, mac_cycles=one.mac_cycles + red,
                           convert_elems=c.convert_elems, convert_w=c.convert_w)
        s *= 2
    return best


def _op_seconds(op: Op, sys: System, org: str, model: ModelSpec) -> tuple:
    """(array s, mac-bound share s, feed-bound share s, vector s, convert elems, ideal s, SRAM-port bytes,
    vector s of per-pass weight conversion)."""
    ch = sys.chip
    f = ch.freq_ghz * 1e9 * ch.mac_eff
    if op.kind == "attn" and ch.attn_sched:      # 0.93 (default off)
        ch = _attn_chip(ch)
    if op.kind == "attn" and ch.tile_io_Bpc:     # 0.136: flash attention keeps S / P on chip and its K / V stream is
        ch = replace(ch, tile_io_Bpc=0.0)        # already DRAM traffic — tile I/O is a GEMM-kernel term only
    if op.kind == "gemm":
        c = gemm_cost(ch, org, op.m, op.k, op.n, count=op.count, w_fmt=op.w_fmt, a_fmt=op.a_fmt, w_bits=op.w_bits)
        if ch.instance_sched == "wide" and not ch.split_instances:
            c = _wide_alt(ch, org, op, c)
    elif op.kind == "attn" and op.orient:
        # full-sequence (DiT / encoder) attention: both operands are activations in the activation dtype, and a
        # flash kernel may compute O = P·V or Oᵀ = Vᵀ·Pᵀ — take the cheaper orientation (PV has N = head_dim,
        # which leaves most of a wide array idle).  0.172: LLM prefill / decode attention takes the cheaper one too (below).
        af = model.act_fmt
        c = min((gemm_cost(ch, org, a, op.k, b, count=op.count, w_fmt=af, a_fmt=af)
                 for a, b in ((op.m, op.n), (op.n, op.m))), key=lambda x: x.cycles)
    elif op.kind == "attn":
        kf = op.w_fmt or model.kv_fmt
        if ch.attn_fmt == "kv" and kf != op.a_fmt and ch.formats.native(kf):
            # 0.96 native KV-format attention: Q quantised on the vector path (QK); P is cast inside the softmax epilogue
            # (FlashAttention-3 FP8), already charged with softmax
            c = gemm_cost(ch, org, op.m, op.k, op.n, count=op.count, w_fmt=kf, a_fmt=kf)
            if op.name.startswith("qk"):
                c = replace(c, convert_elems=c.convert_elems + op.m * op.k * op.count)
        else:
            c = gemm_cost(ch, org, op.m, op.k, op.n, count=op.count, w_fmt=kf, a_fmt=op.a_fmt)
            # 0.172: an attention kernel may also compute the transposed product (Sᵀ = K·Qᵀ, Oᵀ = Vᵀ·Pᵀ) — the same
            # freedom the full-sequence path (orient) has had since 0.40.  Without it, PV of a long prefill (N =
            # head_dim) used head_dim of the C·E columns of a ganged ("wide") array: 128 / 8448 on the GPU references,
            # 66× its FLOP time (llama-3.1-8b 131k: attention 152 s vs 6.4 s ideal)
            if ATTN_ORIENT:
                t = gemm_cost(ch, org, op.n, op.k, op.m, count=op.count, w_fmt=op.a_fmt, a_fmt=kf)
                if t.cycles < c.cycles:
                    c = t
    else:
        return 0.0, 0.0, 0.0, _vec_seconds(ch, op), 0.0, 0.0, 0.0, 0.0
    if (op.kind == "attn" and WIDE_GEMM_SPLIT and op.count > 1 and ch.instance_sched == "wide"
            and not ch.split_instances):
        # 0.173: (batch × head × chunk) attention instances spread over SMs (see _wide_alt) — same formats as costed
        kf = op.w_fmt or model.kv_fmt
        wf, af = (model.act_fmt, model.act_fmt) if op.orient else (c.exec_fmt, c.exec_fmt) if c.exec_fmt == kf else (kf, op.a_fmt)
        sp = replace(ch, instance_sched="auto")   # instances over SMs, rows of each over the rest (FA grid)
        t = min((gemm_cost(sp, org, a, op.k, b, count=op.count, w_fmt=wf, a_fmt=af)
                 for a, b in ((op.m, op.n), (op.n, op.m))), key=lambda x: x.cycles)
        if t.cycles < c.cycles:
            c = replace(t, convert_elems=c.convert_elems)
    if SPLIT_K and ch.instance_sched != "cta":
        if op.kind == "gemm":
            c = _splitk_alt(ch, org, op.m, op.k, op.n, op.count, op.w_fmt, op.a_fmt, op.w_bits, c)
        elif op.kind == "attn":
            # Flash-Decoding (Dao et al., 2023): decode attention splits the KV length over thread blocks and
            # rescales / sums the partial outputs — split-K of PV (K = ctx); QK's K is head_dim (no split)
            kf = model.act_fmt if op.orient else (op.w_fmt or model.kv_fmt)
            c = _splitk_alt(ch, org, op.m, op.k, op.n, op.count, kf, model.act_fmt if op.orient else op.a_fmt, None, c)
    t = c.cycles * op.causal / f
    vec = _vec_seconds(ch, op) + _convert_seconds(ch, c.convert_elems * 2.0)   # 2 element-ops / element 「假设」
    mac_share = t if c.bound == "mac" else 0.0
    rate = ch.formats.rate(c.exec_fmt) or 1.0
    ideal = op.flops / 2.0 / (ch.macs * rate * ch.freq_ghz * 1e9)   # 100 % array utilisation
    if ch.fmt_ceiling:                                   # 0.94 (default off)
        fc = dict(ch.fmt_ceiling)
        cap = _shape_cap(ch.fmt_ceiling, op, c.exec_fmt) or fc.get(f"{op.kind}:{c.exec_fmt}", fc.get(c.exec_fmt))
        if cap and op.kind == "gemm" and ch.gemm_ceil_mul and ch.instance_sched == "cta" and not ch.tile_io_Bpc:
            # (under tile I/O, mapping._cta already runs the main loop at the ceiling)
            d = t / cap - t                              # 0.193: main loop at the achieved fraction of peak
            t += d
            mac_share += d
        elif cap and ideal / cap > t:
            d = ideal / cap - t
            t += d
            mac_share += d
    vw = _convert_seconds(ch, c.convert_w * 2.0) if op.kind == "gemm" else 0.0   # weight dequant: per pass, not per token
    return t, mac_share, t - mac_share, vec, c.convert_elems, ideal, c.feed_cycles * op.causal * ch.port_Bpc, vw


def stage_ops(model: ModelSpec, first: int, last: int, has_embed: bool, has_head: bool, ph: Phase, sh: Shard,
              spec_k: int) -> list[Op]:
    """Full op list of one rank of a stage (debug / tests / UI drill-down)."""
    ops: list[Op] = []
    if has_embed:
        ops += embed_ops(model, ph, sh)
    for li in range(first, last):
        ops += build_rank_ops(model, li, ph, sh)
    ops += _tail_ops(model, has_head, ph, sh, spec_k)
    return ops


def _tail_ops(model, has_head, ph, sh, spec_k, glue: bool = False):
    ops: list[Op] = []
    if has_head:
        ops += head_ops(model, ph, sh, glue)
        if spec_k and model.draft == "dspark":
            ops += dspark_ops(model, ph, sh)       # one block pass per step (prefill: seed the draft windows)
        elif spec_k and model.mtp_layers and ph.kind == "decode":
            dph = Phase("decode", ph.batch, 1, ph.ctx, skew=ph.skew)
            for d in range(spec_k):
                ops += mtp_ops(model, dph, sh, depth=d)
    return ops


BALANCE_WARN = 10.0    # 0.205: machine balance vs arithmetic intensity warning ratio 「假设」

_SUM_KEYS = ("t_arr", "t_mac", "t_feed", "t_vec", "conv", "t_ideal", "flops", "link_bw", "sync", "link_bytes",
             "hot", "exp", "kv_read", "kv_write", "state", "lookup", "max_act", "act", "w_extra", "max_act_tot",
             "flops_u", "sram", "d2d_bytes", "net_bytes", "busy_d2d", "busy_link", "busy_net", "bw_max",
             "t_ideal_w", "t_vec_w", "t_conv_w", "sram_e", "conv_w_e", "link_bytes_e", "t_arr_attn", "t_vec_attn",
             "wire_d2d", "wire_link", "wire_net", "kern", "disp")
_MAX_KEYS = ("max_act", "max_act_tot", "bw_max")
_EP_COMM = ("moe_dispatch", "moe_combine", "expert_allreduce")
_TIERS = ("d2d", "link", "net")


_COLL_MEMO: dict = {}       # 0.61: System → {(kind, bytes, group, stride): collective_full(...)} (pure function)
_MEMO_CAP = 200_000


def _memos(sys: System, org: str, model: ModelSpec) -> tuple:
    """0.61: per-op cost memo — the same per-rank op shapes recur across layouts / batches of a search (e.g. B/dp
    equal at the doubling points): ~29× hits on _op_seconds, ~9× on the fabric collectives at 1024 cards.  Pure
    functions of (op, chip, mapping, model formats) / (collective, System); the collective memo is off while
    fabric_report logs collectives.  Looked up once per evaluate."""
    try:
        om = model_cache(model).setdefault("op_secs", {})
    except TypeError:           # a shim namespace (no weak reference): no memo across calls
        return {}, None
    if len(om) > 64:
        om.clear()
    om = om.setdefault((sys.chip, org, ATTN_ORIENT, WIDE_GEMM_ORIENT, WIDE_GEMM_SPLIT, WIDE_GEMM_AUTO, SPLIT_K, _mapping.WIDE_WAVES, _mapping.GPU_WAVES_ONLY), {})   # 0.172: the flag changes attention op costs
    if len(om) > _MEMO_CAP:
        om.clear()
    cm = None
    if sys.fabric.enabled and fabric.log() is None:
        if len(_COLL_MEMO) > 256:
            _COLL_MEMO.clear()
        cm = _COLL_MEMO.setdefault(sys, {})
        if len(cm) > _MEMO_CAP:
            cm.clear()
    return om, cm


def _exp_acts(agg: dict) -> dict:
    """0.65: the routed-expert part of a stage step — SRAM-port bytes and weight-dequant vector seconds of the
    expert GEMMs, and the EP dispatch / combine / expert all-reduce share of the stage's collective bytes."""
    lb = agg["link_bytes"]
    return {"sram": agg["sram_e"], "conv_w": agg["conv_w_e"], "link_frac": agg["link_bytes_e"] / lb if lb > 0 else 0.0}


def _sum_ops(ops: list[Op], sys: System, org: str, model: ModelSpec, memos: tuple | None = None) -> dict:
    d = dict.fromkeys(_SUM_KEYS, 0.0)
    om, cm = memos if memos is not None else _memos(sys, org, model)
    _po = sys.chip.dispatch_s()[1]
    for o in ops:
        if _po and (o.kind in ("gemm", "comm") or (o.kind == "attn" and o.name.startswith("qk"))
                    or (o.kind == "vector" and not o.name.endswith("softmax"))):
            d["sync"] += _po * max(1, o.launch)     # 0.203: one dispatched op (fused attention = one)
            d["disp"] += _po * max(1, o.launch)
        if o.kind == "comm":
            if sys.fabric.enabled:      # 0.60: per-tier busy seconds (overlap = ports) + candidates (auto_overlap)
                if cm is not None:
                    ck = (o.comm_kind, o.comm_bytes, o.comm_group, o.comm_stride)
                    cr = cm.get(ck)
                    if cr is None:
                        cr = cm[ck] = fabric.collective_full(o.comm_kind, o.comm_bytes, o.comm_group, sys,
                                                             o.comm_stride)
                    bw, a, fd, fn, busy, full, wr = cr
                else:
                    bw, a, fd, fn, busy, full, wr = fabric.collective_full(o.comm_kind, o.comm_bytes, o.comm_group, sys,
                                                                       o.comm_stride)
                for t, x in busy.items():
                    d["busy_" + t] += x
                d["bw_max"] = max(d["bw_max"], bw)
                if full is not None:
                    c = d.setdefault("_cands", {})
                    c[full[0]] = [c[full[0]][0] + 1 if full[0] in c else 1, full]
            else:
                bw, a, fd, fn = _comm(sys, o.comm_kind, o.comm_bytes, o.comm_group, o.comm_stride, net_scale=o.comm_net)
                wr = _comm_wire(sys, o.comm_kind, o.comm_bytes, o.comm_group, o.comm_stride, net_scale=o.comm_net)
            if sys.chip.gpu_glue and sys.chip.kernel_us:   # 0.177: collectives are GPU kernels too (EP a2a: send + recv)
                nk = 2 if o.name in ("moe_dispatch", "moe_combine") else 1
                d["sync"] += sys.chip.kernel_us * 1e-6 * nk
                d["kern"] += sys.chip.kernel_us * 1e-6 * nk
            for t, x in wr.items():                # 0.78: physical bytes on each tier's wire (energy)
                d["wire_" + t] += x
            d["link_bw"] += bw; d["sync"] += a; d["link_bytes"] += o.comm_bytes; d["d2d_bytes"] += o.comm_bytes * fd
            if o.name in _EP_COMM:            # 0.65: what a DP rank without sequences still sends / receives under EP
                d["link_bytes_e"] += o.comm_bytes
            d["net_bytes"] += o.comm_bytes * fn
            continue
        r_ = om.get(o)
        if r_ is None:
            r_ = om[o] = _op_seconds(o, sys, org, model)
        a_, ma, fe, v, ce, idl, sb, vw = r_
        if o.kind == "attn":
            d["t_arr_attn"] += a_
        if o.kind == "attn" or o.name.endswith("softmax"):
            d["t_vec_attn"] += v                    # 0.104: attention-core vector work (Chip.attn_vec_overlap)
        if sys.chip.kernel_us and o.launch:
            d["sync"] += sys.chip.kernel_us * 1e-6 * o.launch    # 0.120: explicit launches (structure-model ops, chunks)
            d["kern"] += sys.chip.kernel_us * 1e-6 * o.launch
        elif sys.chip.kernel_us and (o.kind == "gemm" or (o.kind == "attn" and o.name.startswith("qk"))):
            d["sync"] += sys.chip.kernel_us * 1e-6    # 0.92: one kernel per GEMM / fused attention (QK·softmax·PV)
            d["kern"] += sys.chip.kernel_us * 1e-6
        elif sys.chip.kernel_us and sys.chip.kernel_vec and o.kind == "vector" and not o.name.endswith("softmax"):
            d["sync"] += sys.chip.kernel_us * 1e-6    # 0.136 opt-in: stand-alone vector kernels
            d["kern"] += sys.chip.kernel_us * 1e-6
        d["t_arr"] += a_; d["t_mac"] += ma; d["t_feed"] += fe; d["t_vec"] += v; d["conv"] += ce
        d["t_ideal"] += idl; d["sram"] += sb
        # 0.110 (audit): the SFU pipe (Chip.sfu_per_cycle) only changes the op's *time* (max of lanes and SFU); the
        # vector work counted for energy / KPIs stays the lane work (SFU-bound ops inflated the vec count)
        sfu_x = _vec_seconds(sys.chip, o) - vector_seconds(sys.chip, o.vec) if (sys.chip.sfu_per_cycle and o.sfu) else 0.0
        d["t_ideal_w"] += idl * o.share; d["t_vec_w"] += (v - vw - sfu_x) * o.share   # 0.63: mean-rank work (energy, KPIs)
        d["t_conv_w"] += vw * o.wshare        # 0.64: mean-rank weight elements (uneven TP / EP splits)
        if o.role == "expert":                # 0.65: routed-expert share (idle EP ranks, energy action counts)
            d["sram_e"] += sb
            d["conv_w_e"] += vw * o.wshare
        d["flops"] += o.flops
        d["flops_u"] += o.flops * o.share / max(1, o.replicated)
        d["max_act"] = max(d["max_act"], o.act_bytes / max(o.count, 1))
        if o.stream:
            a, wx = act_stream(o, _fmt(model.act_fmt).bytes, sys.chip.sram_bytes, sys.chip.engines)
            d["act"] += a; d["w_extra"] += wx
            ab = _fmt(model.act_fmt).bytes
            tot = (((o.in_elems or o.m * o.k * o.count) + o.m * o.n * o.count) * ab if o.kind == "gemm"
                   else o.act_bytes if (o.bmm or o.kind == "state")
                   else 2.0 * o.count * (o.m + o.n) * o.k * ab)
            d["max_act_tot"] = max(d["max_act_tot"], tot)
    for k, v in touched(ops).items():
        d[k] += v
    return d


def _moe_skew(m: ModelSpec, sv, lay: Layout, warnings: list[str]) -> float:
    """Effective EP load skew of the scenario (0.49): ``serving.moe_skew`` or derived from ``moe_expert_load``."""
    if sv.moe_skew == 1.0 and not sv.moe_expert_load:
        return 1.0
    if not m.is_moe or lay.ep <= 1:
        warnings.append("MoE 负载倾斜已忽略：" + ("非 MoE 模型" if not m.is_moe else "EP = 1（专家全在每个 rank 上，无跨 rank 不均衡）"))
        return 1.0
    if sv.moe_expert_load:
        n = next(L.ffn.n_experts for L in m.layers if L.ffn.kind == "moe")
        if len(sv.moe_expert_load) != n:
            raise ValueError(f"serving.moe_expert_load needs {n} entries (one per routed expert), got "
                             f"{len(sv.moe_expert_load)}")
        return skew_from_load(sv.moe_expert_load, lay.ep)
    # 0.74: physical cap — a token sends its top-k pairs to distinct experts, at most min(k, E/EP) of them to one
    # rank, so the busiest rank holds ≤ min(k, E/EP)/k of all pairs: skew ≤ EP · min(k, E/EP) / k (≤ EP)
    f = next(L.ffn for L in m.layers if L.ffn.kind == "moe")
    k = max(1, f.top_k)
    cap = lay.ep * min(k, -(-f.n_experts // lay.ep)) / k
    if sv.moe_skew > cap * (1 + 1e-12):
        warnings.append(f"serving.moe_skew = {sv.moe_skew:g} 超过物理上限 {cap:g}（EP {lay.ep}、top-{k}、"
                        f"{f.n_experts} 专家：最忙 rank 至多承担全部 token-专家对的 min(k, E/EP)/k），按上限计")
        return float(cap)
    return float(sv.moe_skew)


def _a2a_sm_share(stt, ph, sv):
    """0.134 opt-in (serving.moe_a2a_sm): node-mode all-to-all kernels (DeepEP normal) run on a share of the compute
    units while they overlap the other micro-batch's compute (deepseek-ai/profile-data: prefill all-to-all occupies SMs,
    decode's low-latency kernels do not), so prefill array / vector time grows by 1 / (1 − share)."""
    f = sv.moe_a2a_sm
    if f <= 0 or ph.ep_nodes <= 1 or ph.kind == "decode":
        return stt
    k = 1.0 / (1.0 - f)
    return replace(stt, t_array=stt.t_array * k, t_mac=stt.t_mac * k, t_feed=stt.t_feed * k, t_vector=stt.t_vector * k,
                   t_arr_attn=stt.t_arr_attn * k)


def _comm(sys: System, kind: str, payload: float, group: int, stride: int = 1,
          members: tuple[int | None, int | None] | None = None, net_scale: float = 1.0) -> tuple[float, float, float, float]:
    """One collective on the system's three-tier fabric (0.50; 0.48 two-tier when node_cards = 0):
    (bandwidth s, α s, D2D share, network share of the bytes).  ``members``: (ranks inside one package, inside one
    node) when the group is not a uniform stride (None entries = derive from the stride)."""
    kp, kn = members if members is not None else (None, None)
    if sys.fabric.enabled:      # 0.59 topology-aware collectives 「假设」
        return fabric.collective(kind, payload, group, sys, stride, k_pkg=kp, k_node=kn)
    if sys.package_cards <= 1 and sys.node_cards <= 0:
        return (*collective_seconds(kind, payload, group, sys.link), 0.0, 0.0)
    return fabric_collective(kind, payload, group, sys.link, sys.d2d if sys.package_cards > 1 else None,
                             sys.package_cards, sys.net if sys.node_cards > 0 else None, sys.node_cards, stride,
                             k_pkg=kp, k_node=kn, net_scale=net_scale)


def _comm_wire(sys: System, kind: str, payload: float, group: int, stride: int = 1,
               members: tuple[int | None, int | None] | None = None, net_scale: float = 1.0) -> dict:
    """0.78: {tier: bytes one rank sends on the wire} of the collective ``_comm`` times (per algorithm / level)."""
    kp, kn = members if members is not None else (None, None)
    if sys.fabric.enabled:
        return fabric.collective_full(kind, payload, group, sys, stride, k_pkg=kp, k_node=kn)[6]
    if sys.package_cards <= 1 and sys.node_cards <= 0:
        x = wire_single(kind, payload, group)
        return {"link": x} if x else {}
    return collective_wire(kind, payload, group, sys.package_cards > 1, sys.package_cards, sys.node_cards, stride,
                           k_pkg=kp, k_node=kn, net_scale=net_scale)


def _add_wire(w: dict, d: dict, k: float = 1.0) -> None:
    for t, x in d.items():
        w[t] = w.get(t, 0.0) + k * x


def _p2p(sys: System, payload: float, stage: int, stage_cards: int) -> tuple[float, float, float, float]:
    return _p2p_full(sys, payload, stage, stage_cards)[:4]


def _p2p_full(sys: System, payload: float, stage: int, stage_cards: int) -> tuple[float, float, float, float, str]:
    """PP hand-off of ``stage`` → next: (bw s, α s, D2D share, network share of the bytes, tier of the slowest pair).
    0.63 (external review): every rank j of the stage sends its slice to rank j of the next stage at once (cards
    numbered TP, SP, DP, PP) — each pair on its own tier (same package → D2D, same node → scale-up link, else the
    network, oversubscribed when the two nodes sit under different leaves / pods); the hand-off ends with the slowest
    pair.  Was one representative pair (the stage's last card → the next stage's first): PP2·TP64 on 8-card nodes,
    4:1 leaves — 1 ms instead of 4 ms (rank 63 → 64 share a leaf, rank 0 → 64 do not)."""
    pk, nd = sys.package_cards, sys.node_cards
    keys: dict = {}
    for j in range(max(1, stage_cards)):
        a = stage * stage_cards + j
        b = a + stage_cards
        if pk > 1 and a // pk == b // pk:
            t = "d2d"
        elif nd <= 0 or a // nd == b // nd:
            t = "link"
        else:
            t = "net"
        # pairs with the same tier and (node, node) positions cost the same; keep one representative per class
        k = (t, a // nd if (t == "net" and nd > 0) else 0, b // nd if (t == "net" and nd > 0) else 0)
        if k not in keys:
            keys[k] = [a, b, 0]
        keys[k][2] += 1
    n_pairs = sum(v[2] for v in keys.values())
    fd = sum(v[2] for k, v in keys.items() if k[0] == "d2d") / n_pairs
    fn = sum(v[2] for k, v in keys.items() if k[0] == "net") / n_pairs
    best = None
    for (t, _, _), (a, b, _) in keys.items():
        ln = {"d2d": sys.d2d, "link": sys.link, "net": sys.net}[t]
        if sys.fabric.enabled and payload > 0:      # 0.59: oversubscribed leaf uplinks + per-step hop latency
            bw, al, pr = fabric.p2p_alpha(fabric.p2p(payload, t, ln, sys, a, b, stage_cards), t, ln, sys, a, b,
                                          with_proto=True)
        else:
            bw, al = collective_seconds("p2p", payload, 2, ln)
            pr = None
        if best is None or bw + al > best[0] + best[1]:
            best = (bw, al, t, pr)
    bw, al, t, pr = best
    if sys.fabric.enabled and payload > 0 and fabric.log() is not None:
        nm = "p2p" if pr is None else f"p2p/{pr}"
        e = fabric.log().setdefault(("p2p", stage, round(payload)), {
            "kind": "p2p", "group": 2, "bytes": payload, "count": 0, "levels": [[t, 2]], "algo": nm,
            "top": "", "cands": {nm: [bw, al]}, "stage": stage})
        e["count"] += 1
    return bw, al, fd, fn, t


def _slc_time(sys: System, dram: dict) -> float:
    s = dram.get("slc", 0.0)
    return s / (sys.chip.slc_GBps * 1e9) if s > 0 else 0.0


def _links(scn: Scenario) -> tuple:
    if scn.fabric.enabled:      # 0.59: topology factors depend on the replica's cards (scale-up domain on one node)
        return (scn.link, scn.d2d_link if scn.package_eff > 1 else None, scn.package_eff,
                scn.net if scn.node_cards > 0 else None, scn.node_cards, scn.fabric, scn.layout.cards)
    if scn.package_eff <= 1 and scn.node_cards <= 0:
        return scn.link
    return (scn.link, scn.d2d_link if scn.package_eff > 1 else None, scn.package_eff,
            scn.net if scn.node_cards > 0 else None, scn.node_cards)


def _dram_t(sys: System, dram: dict) -> float:
    """DRAM seconds of a stage's traffic; 0.108: KV-cache bytes at System.kv_eff when set (else = total / BW)."""
    s = sys.kv_time_scale
    b = dram["total"] if s == 1.0 else dram["total"] + (dram["kv_read"] + dram["kv_write"]) * (s - 1.0)
    return b / (sys.dram_GBps * 1e9)


def _pipe_time(sys: System, org: str, m: ModelSpec, mm, segments, mp: MemPlan, store, dram: dict,
               tail_comm: float = 0.0, hidden_link: float = 0.0) -> float:
    """0.98: exec_overlap "pipe" — replay the stage's op sequence on DMA + compute (core/overlap.py).
    segments = [(op list, repeat)] in program order."""
    om = mm[0] if mm else {}
    ab = _fmt(m.act_fmt).bytes

    def op_secs(o):
        r_ = om.get(o)
        if r_ is None:
            r_ = om[o] = _op_seconds(o, sys, org, m)
        return r_

    def comm_secs(o):
        if sys.fabric.enabled:
            return fabric.collective_full(o.comm_kind, o.comm_bytes, o.comm_group, sys, o.comm_stride)[0]
        return _comm(sys, o.comm_kind, o.comm_bytes, o.comm_group, o.comm_stride, net_scale=o.comm_net)[0]

    def act_b(o):
        a, wx = act_stream(o, ab, sys.chip.sram_bytes, sys.chip.engines)
        return a + wx
    segs = [(_ovl.items_of(ops, op_secs, comm_secs, act_b), n) for ops, n in segments]
    kvst = mp.kv_total + mp.state_total
    f = {"hot": 0.0 if store.hot_w <= 0 else 1.0 - mp.pinned_hot / store.hot_w,
         "exp": 0.0 if store.expert_w <= 0 else 1.0 - mp.pinned_expert / store.expert_w,
         "kv": 0.0 if kvst <= 0 else 1.0 - mp.kv_sram / kvst, "lookup": 1.0, "act": 1.0}
    nb = sum(n * sum(x * f.get(k, 1.0) for _, b, _ in its for k, x in b.items()) for its, n in segs)
    t_mem = max(_dram_t(sys, dram), _slc_time(sys, dram))
    if nb <= 0 and t_mem > 0:          # traffic the ops do not carry (e.g. FSDP copies): one leading DMA step
        segs.insert(0, ([(0.0, {"x": 1.0}, 0.0)], 1)); nb = 1.0
    spb = t_mem / nb if nb > 0 else 0.0
    ch = sys.chip            # the DMA issue limit (Chip.dma_*) is already in sys.dram_GBps (every overlap mode)
    buf = max(mp.staging, ch.sram_bytes - mp.pinned_hot - mp.pinned_expert - mp.kv_sram)
    return _ovl.stage_pipe_time(segs, f, spb, buf, ch.pipe_cross_op, 1.0, tail_comm, hidden_link)


def pipe_fused_time(sd, sp, f: float, rr: float, t_mem: float) -> tuple | None:
    """0.102: pipe makespan of one co-located iteration (decode stage sd ⊕ an f share of prefill stage sp), replayed
    op by op (overlap.merge_chunk) under the decode's memory plan; t_mem = the iteration's DRAM / SLC seconds.
    Returns (makespan without α, array s, vector s) of the merged ops; None when the stages carry no pipe replay
    inputs or their layer groups do not line up."""
    if sd.pipe_ctx is None or sp.pipe_ctx is None:
        return None
    key = (id(sp.pipe_ctx), f, rr, t_mem)
    memo = sd.pipe_ctx[-1] if len(sd.pipe_ctx) > 8 else None
    if memo is not None and key in memo:
        return memo[key]
    r = _pipe_fused(sd, sp, f, rr, t_mem)
    if memo is not None:
        if len(memo) > 4096:
            memo.clear()
        memo[key] = r
    return r


def _pipe_fused(sd, sp, f, rr, t_mem):
    picks = sd.pipe_ctx[8] if len(sd.pipe_ctx) > 9 else None   # 0.142: the decode stage's pipe re-pick
    tok = fabric.set_pick(picks)
    try:
        return _pipe_fused_(sd, sp, f, rr, t_mem)
    finally:
        fabric.reset_pick(tok)


def _pipe_fused_(sd, sp, f, rr, t_mem):
    sys, org, m, mm, segs_d, mp, store, tail = sd.pipe_ctx[:8]
    segs_p = sp.pipe_ctx[4]
    ru = union_routing(sd, sp, f)
    if len(segs_d) != len(segs_p):
        return None
    merged = []
    for (od, n), (op_, n2) in zip(segs_d, segs_p):
        if n != n2:
            return None
        mo = _ovl.merge_chunk(od, op_, f, rr, ru)
        if mo is None:
            return None
        merged.append((mo, n))
    om = mm[0] if mm else {}
    ab = _fmt(m.act_fmt).bytes

    def op_secs(o):
        r_ = om.get(o)
        if r_ is None:
            r_ = om[o] = _op_seconds(o, sys, org, m)
        return r_

    def comm_secs(o):
        if sys.fabric.enabled:
            return fabric.collective_full(o.comm_kind, o.comm_bytes, o.comm_group, sys, o.comm_stride)[0]
        return _comm(sys, o.comm_kind, o.comm_bytes, o.comm_group, o.comm_stride, net_scale=o.comm_net)[0]

    def act_b(o):
        a, wx = act_stream(o, ab, sys.chip.sram_bytes, sys.chip.engines)
        return a + wx
    segs = [(_ovl.items_of(ops, op_secs, comm_secs, act_b), n) for ops, n in merged]
    t_arr = t_vec = 0.0                 # the merged iteration's own array / vector seconds (bracket bounds)
    for ops, n in merged:
        for e in ops:
            o, sc = (e[0], e[1]) if isinstance(e, tuple) else (e, 1.0)
            if o.kind != "comm":
                r_ = op_secs(o)
                t_arr += n * sc * r_[0]; t_vec += n * sc * r_[3]
    kvst = mp.kv_total + mp.state_total
    fr = {"hot": 0.0 if store.hot_w <= 0 else 1.0 - mp.pinned_hot / store.hot_w,
          "exp": 0.0 if store.expert_w <= 0 else 1.0 - mp.pinned_expert / store.expert_w,
          "kv": 0.0 if kvst <= 0 else 1.0 - mp.kv_sram / kvst, "lookup": 1.0, "act": 1.0}
    nb = sum(n * sum(x * fr.get(k, 1.0) for _, b, _ in its for k, x in b.items()) for its, n in segs)
    if nb <= 0 and t_mem > 0:
        segs.insert(0, ([(0.0, {"x": 1.0}, 0.0)], 1)); nb = 1.0
    spb = t_mem / nb if nb > 0 else 0.0
    ch = sys.chip
    buf = max(mp.staging, ch.sram_bytes - mp.pinned_hot - mp.pinned_expert - mp.kv_sram)
    return _ovl.stage_pipe_time(segs, fr, spb, buf, ch.pipe_cross_op, 1.0, tail, 0.0), t_arr, t_vec


def _acc(a: dict, b: dict, n: int = 1) -> None:
    for k in _SUM_KEYS:
        if k in _MAX_KEYS:
            a[k] = max(a[k], b[k])
        else:
            a[k] += b[k] * n
    if "_cands" in b:           # 0.60 auto_overlap: collective classes with their counts (new lists: b may be memoised)
        c = a.setdefault("_cands", {})
        for key, (cnt, full) in b["_cands"].items():
            c[key] = [c[key][0] + cnt * n if key in c else cnt * n, full]


def _p2p_tier_of(fd: float, fn: float) -> str:
    return "net" if fn >= 1.0 else "d2d" if fd >= 1.0 else "link"


def _stage_link(sys: System, agg: dict, link_bw: float, sync: float, d2d_b: float, net_b: float, extra: list,
                window: float, stage: int = 0, wire: dict | None = None, sel: tuple | None = None,
                out: dict | None = None) -> tuple[float, float, float, float]:
    """0.60 (fabric on): stage link seconds under ``fabric.overlap`` and the ``auto_overlap`` re-pick.
    extra = [(bw, tier)] of the stage's PP / FSDP transfers already in link_bw.  Returns (link, sync, d2d, net)."""
    fab = sys.fabric
    busy = {t: agg["busy_" + t] for t in _TIERS}
    bw_max = agg["bw_max"]
    for bw, t in extra:
        busy[t] += bw
        bw_max = max(bw_max, bw)
    cl = list(agg.get("_cands", {}).values())
    ports = fab.overlap == "ports"

    def link_of(lb: float, bz: dict, bm: float) -> float:
        return max(max(bz.values()), bm) if ports else lb

    choice = None
    if fab.algo == "auto_overlap" and cl:
        base_lb, base_sync, base_bz = link_bw, sync, dict(busy)
        for cnt, (key, pay, dflt, cands) in cl:     # strip the default picks
            bw, a, bz, _, _, _ = cands[dflt]
            base_lb -= cnt * bw
            base_sync -= cnt * a
            for t, x in bz.items():
                base_bz[t] -= cnt * x
        names = [sorted(f[3]) for _, f in cl]

        def cost(sel):
            lb, sy, bz, bm = base_lb, base_sync, dict(base_bz), bw_max
            for (cnt, (key, pay, dflt, cands)), nm in zip(cl, sel):
                bw, a, b_, _, _, _ = cands[nm]
                lb += cnt * bw
                sy += cnt * a
                for t, x in b_.items():
                    bz[t] += cnt * x
                bm = max(bm, bw)
            return max(window, link_of(lb, bz, bm)) + sy, lb, sy, bz, bm

        dsel = [f[2] for _, f in cl]
        n_comb = math.prod(len(x) for x in names)
        if out is not None:     # 0.142: the search space for the pipe re-pick (_pipe_repick)
            out["names"], out["dsel"], out["keys"] = names, tuple(dsel), [f[0] for _, f in cl]
        if sel is not None:
            best = tuple(sel)
        elif n_comb <= 729:
            best = min(itertools.product(*names), key=lambda sel: (cost(sel)[0], sel != tuple(dsel)))
        else:                   # coordinate descent from the default picks
            best = list(dsel)
            for _ in range(3):
                for i in range(len(best)):
                    best[i] = min(names[i], key=lambda nm: (cost(best[:i] + [nm] + best[i + 1:])[0], nm != best[i]))
            best = tuple(best)
        if out is not None:
            out["best"] = best
        _, link_bw, sync, busy, bw_max = cost(best)
        for (cnt, (key, pay, dflt, cands)), nm in zip(cl, best):
            if nm != dflt:
                _, _, _, fd0, fn0, w0 = cands[dflt]
                _, _, _, fd1, fn1, w1 = cands[nm]
                d2d_b += cnt * pay * (fd1 - fd0)
                net_b += cnt * pay * (fn1 - fn0)
                if wire is not None:            # 0.78: the re-picked algorithm's wire bytes
                    _add_wire(wire, w0, -cnt)
                    _add_wire(wire, w1, cnt)
        choice = {key: nm for (cnt, (key, *_)), nm in zip(cl, best)}
    link = link_of(link_bw, busy, bw_max)
    if fabric.log() is not None:
        if choice:
            for key, nm in choice.items():
                if key in fabric.log():
                    fabric.log()[key]["algo"] = nm
        fabric.log().setdefault("_stages", []).append(
            {"stage": stage, "link_sum_us": link_bw * 1e6, "link_ports_us": max(max(busy.values()), bw_max) * 1e6,
             "busy_us": {t: v * 1e6 for t, v in busy.items()}, "window_us": window * 1e6, "sync_us": sync * 1e6,
             "link_us": link * 1e6})
    return link, sync, d2d_b, net_b



def _pipe_repick(sys, agg, pre: tuple, extra: list, window: float, idx: int, so: dict, sync_add: float,
                 tpipe, make_stt, stt0):
    """0.142 (exec_overlap = pipe, fabric.algo = auto_overlap): ≤ 0.141 the stage re-pick set the link / sync terms
    while the flow-shop replay still ran every collective at its default (per-collective) pick — an inconsistent mix
    that could undercut both (fuzz seed 41: 2× D2D units +0.009 % because the 4-unit point mixed hier α with ring
    bandwidth).  Now the replay uses the same picks, and the combination is chosen on the pipe total itself (which
    also clamps to the stage / serial bracket): the stage-optimal and the default picks, then
    combination on the pipe total itself: the stage-optimal and the default picks, then coordinate descent (≤ 2 rounds)
    from the better one.  Returns (StageTime, picks {collective key: algorithm} | None)."""
    names, keys = so.get("names"), so.get("keys")
    if not names or math.prod(len(x) for x in names) <= 1:
        return stt0, None
    best0 = tuple(so["best"])
    lg = fabric.log()
    fabric.set_log(None)
    memo = {}       # stt0 replayed the default picks in the flow shop while its link / sync used best0 (≤ 0.141)

    def run(sel):
        sel = tuple(sel)
        if sel not in memo:
            w = dict(pre[4])
            lb, sy, dd, nn = _stage_link(sys, agg, pre[0], pre[1], pre[2], pre[3], extra, window, idx, w, sel=sel)
            tok = fabric.set_pick(dict(zip(keys, sel)))
            try:
                tp = tpipe()
            finally:
                fabric.reset_pick(tok)
            memo[sel] = make_stt(lb, sy + sync_add, dd, nn, w, tp)
        return memo[sel].total

    try:
        cur = list(min((best0, tuple(so["dsel"])), key=lambda x: (run(x), x != best0)))
        for _ in range(2):
            moved = False
            for i in range(len(cur)):
                nm = min(names[i], key=lambda n: (run(cur[:i] + [n] + cur[i + 1:]), n != cur[i]))
                moved |= nm != cur[i]
                cur[i] = nm
            if not moved:
                break
    finally:
        fabric.set_log(lg)
    cur = tuple(cur)
    if cur == best0 and best0 == tuple(so["dsel"]):
        return stt0, dict(zip(keys, cur))
    if lg is not None and cur != best0:          # the log follows the chosen combination
        if lg.get("_stages"):
            lg["_stages"].pop()
        _stage_link(sys, agg, pre[0], pre[1], pre[2], pre[3], extra, window, idx, dict(pre[4]), sel=cur)
    return memo[cur], dict(zip(keys, cur))


_DRAM_KEYS = ("hot", "exp", "kv_read", "kv_write", "state", "lookup", "act", "w_extra")


def _plan_by_cost(scn, m, sys, pp, groups, ph, sh, mm, ops_memo, store_memo, ctx, n_mtp, b_rank, first_ops, last_ops,
                  p2p_payload, repeat: bool = False):
    """0.62 PP split (``scn.pp_split``): "cost" (default) balances the per-stage time of the stage rule over layers,
    embedding / io-pre (stage 0), head / io-post / MTP (last stage) and the PP hand-off (core/parallel.py
    ``plan_stages_balanced``); "layers" = equal layer counts (≤ 0.61).  Returns (stages, info) -- info None when the
    equal-count split is used outright."""
    lc = plan_stages(m.n_layers, pp)
    if pp == 1 or scn.pp_split == "layers":
        return lc, None
    log = fabric.log(); fabric.set_log(None)         # the proxy pass must not count collectives for fabric_report
    try:
        bw = sys.dram_GBps * 1e9

        def vec(a, k=1.0):
            return (a["t_arr"] * k, a["t_vec"] * k, sum(a[x] for x in _DRAM_KEYS) * k / bw, a["link_bw"] * k,
                    a["sync"] * k)

        per_g: dict = {}
        costs = []
        for li in range(m.n_layers):
            g = groups[li]
            if g not in per_g:
                okey = (g, ph, sh, scn.mapping, scn.chip, _links(scn))
                hit = ops_memo.get(okey)
                if hit is None:
                    hit = ops_memo[okey] = _sum_ops(build_rank_ops(m, li, ph, sh, glue=scn.chip.gpu_glue), sys, scn.mapping, m, mm)
                per_g[g] = vec(hit, layer_repeat(m.layers[li], ph) if repeat else 1)
            costs.append(per_g[g])
        first = vec(_sum_ops(first_ops, sys, scn.mapping, m, mm)) if first_ops else (0.0,) * 5
        last = vec(_sum_ops(last_ops, sys, scn.mapping, m, mm)) if last_ops else (0.0,) * 5
        t_p, a_p, _, _ = _p2p(sys, p2p_payload, 0, scn.layout.cards // pp)
    finally:
        fabric.set_log(log)

    def static(st_):
        return st_.weights + (st_.kv_per_seq + st_.idx_per_seq + st_.state_per_seq) * b_rank

    def stor(first_, last_, e, h, nm):
        key = (first_, last_, e, h, sh, ctx, nm)
        x = store_memo.get(key)
        if x is None:
            x = store_memo[key] = stage_storage(m, first_, last_, e, h, sh, ctx, nm)
        return x

    L = m.n_layers
    mem_g: dict = {}
    mem = []
    for li in range(L):
        g = groups[li]
        if g not in mem_g:
            mem_g[g] = static(stor(li, li + 1, False, False, 0))
        mem.append(mem_g[g])
    mem_first = static(stor(0, 1, True, False, 0)) - static(stor(0, 1, False, False, 0))
    mem_last = static(stor(L - 1, L, False, True, n_mtp)) - static(stor(L - 1, L, False, False, 0))

    def peak(plan_):
        return max(sum(mem[st.first:st.last]) + (mem_first if st.has_embed else 0.0) + (mem_last if st.has_head else 0.0)
                   for st in plan_)

    lc_peak = peak(lc)
    cands = plan_stages_balanced(costs, pp, first, last, (0.0, 0.0, 0.0, t_p, a_p), mem, mem_first, mem_last,
                                 max(lc_peak, sys.dram_bytes), alternatives=True)
    if not isinstance(cands, tuple):
        cands = ()                       # 0.140: the proxy sees no gain, but the full model still refines locally
    return cands, {"lc_peak": lc_peak}


def _pick_split(run, cands: tuple, lc) -> list:
    """The proxy proposed splits other than equal counts: evaluate them and the equal-count split with the full model
    and keep the fastest that fits (equal counts win ties; the proxy ignores SRAM / SLC residency, activations and
    fabric effects).  fabric_report's collective log keeps the chosen run's entries only."""
    log = fabric.log()
    base = copy.deepcopy(log) if log is not None else None
    runs = []
    for plan_ in (lc, *cands):
        if log is not None:
            log.clear(); log.update(copy.deepcopy(base))
        st = run(plan_)
        runs.append((st, copy.deepcopy(log) if log is not None else None))

    def key(i):
        st = runs[i][0]
        return (not all(s.mem.fits for s in st), max(s.time.total for s in st), i)
    best = min(range(len(runs)), key=key)
    # 0.140: local refinement with the full model — move one stage boundary by ±1 layer while that is strictly
    # faster (fits first).  The proxy ignores residency / overlap, so its split can sit next to a better one; without
    # this, more DRAM bandwidth could change the proxy's answer to a split slower than the old one.
    plans = [lc, *cands]
    seen = {tuple(st.first for st in p) for p in plans}
    for _ in range(PP_REFINE_STEPS):
        cur = plans[best]
        bd = [st.first for st in cur] + [cur[-1].last]
        moved = False
        for j in range(1, len(bd) - 1):
            for d in (-1, 1):
                nb = list(bd); nb[j] += d
                if nb[j] <= nb[j - 1] or nb[j] >= nb[j + 1] or tuple(nb[:-1]) in seen:
                    continue
                seen.add(tuple(nb[:-1]))
                pl = [type(cur[0])(i, nb[i], nb[i + 1], i == 0, i == len(cur) - 1) for i in range(len(cur))]
                if log is not None:
                    log.clear(); log.update(copy.deepcopy(base))
                st = run(pl)
                runs.append((st, copy.deepcopy(log) if log is not None else None)); plans.append(pl)
                if key(len(runs) - 1)[:2] < key(best)[:2]:
                    best, moved = len(runs) - 1, True
        if not moved:
            break
    if log is not None:
        log.clear(); log.update(runs[best][1])
    return runs[best][0]


PP_REFINE_STEPS = 8


def _dequant_warn(stages, sys, warnings: list) -> None:
    """0.102: weight conversion on the vector / dequant path exceeds the stage's DRAM time — the dequant rate, a
    「假设」 constant (lanes or formats.dequant_elems_per_cycle), decides the step.  Public chips keep ≥ 1.7×
    int4 headroom (core/vecsanity.py)."""
    for st in stages:
        if st.conv_w > 0 and st.conv_w > st.time.t_dram:
            from .vecsanity import check
            h = check(sys.chip, sys.dram_GBps)["dequant_headroom"][0]
            warnings.append(f"权重格式转换（反量化）{st.conv_w * 1e3:.3g} ms 超过 DRAM 时间 {st.time.t_dram * 1e3:.3g} ms，"
                            f"结果取决于反量化吞吐假设（int4 余量 {h:.2f}×，公开芯片 ≥ 1.7×；见 vecsanity）")
            return


def _pp_imbalance_warn(stages, tick: float, warnings: list, split: str = "cost") -> None:
    """0.61.4: flag a lopsided pipeline (0.62: by default the split is already cost-balanced, so this mostly fires for
    pp_split = "layers" or when whole layers are too coarse to balance)."""
    if len(stages) < 2 or tick <= 0:
        return
    lo = min(s.time.total for s in stages)
    if tick > 1.5 * lo:
        if split == "layers":
            warnings.append(f"流水级按层数均分（pp_split = layers，不按代价平衡）：最慢级 / 最快级 = {tick / max(lo, 1e-30):.1f}×，"
                            "节拍取最慢级——异构层栈（蛋白质 trunk / 扩散、首层稠密等）的 PP 结果偏悲观；默认 cost 按代价平衡")
        else:
            warnings.append(f"流水级已按代价平衡，仍不均：最慢级 / 最快级 = {tick / max(lo, 1e-30):.1f}×（整层粒度或容量上限所限），"
                            "节拍取最慢级")

def _cp_check(m: ModelSpec, sv) -> None:
    """0.88: context parallel is modelled for LLM prefill over softmax attention (GQA / MLA, incl. plain DSA top-k)."""
    if not m.kv_cache or m.domain != "llm":
        raise ValueError("layout.cp（上下文并行）只用于 LLM；视频 / 蛋白模型请用 sp")
    if sv.phase != "prefill":
        raise ValueError("layout.cp（上下文并行）只用于 prefill；decode 的 KV 分片未建模（PD 时把 CP 放在 prefill_layout）")
    if sv.prefix_cached:
        raise ValueError("layout.cp 与 serving.prefix_cached 不能同时使用（分块 + 前缀缓存的 zigzag 划分未建模）")
    if sv.images or sv.videos or sv.audio_s:
        raise ValueError("layout.cp 与 serving.images / videos / audio_s 不能同时使用")
    for L in m.layers:
        c = L.core
        if c is None:
            continue
        if c.lin == "hyena":       # 0.171 savanna use_cp_hyena: all-to-all to channel-split full sequences
            continue
        if c.kind not in ("gqa", "mla") or c.window or c.compress > 1 or c.win_bytes or c.dual or c.qsa or c.idx_kpool:
            raise ValueError("layout.cp 只支持全局 softmax 注意力（GQA / MLA，含逐 token 的 DSA top-k）；该模型有线性 / "
                             "滑窗 / 压缩 KV 层，环形注意力的分块等价不成立")


def _cp_ring_ops(m: ModelSpec, first: int, last: int, ring) -> list:
    """Ring-attention KV traffic of one stage: per attention layer an all-gather (cp − 1 blocks received per rank) of
    this rank's KV block — GQA: ⌈kv heads / TP⌉ · (qk + v) dims; MLA: the latent (every TP rank holds it); + DSA
    index keys."""
    cp, tp, tok = ring
    kvb = _fmt(m.kv_fmt).bytes
    ops = []
    for li in range(first, last):
        c = m.layers[li].core
        if c is None:
            continue
        per = _cdiv(c.n_kv, tp) * (c.qk_dim + c.v_dim) * kvb if c.kind == "gqa" else (c.kv_lora + c.rope_dim) * kvb
        if c.idx_heads:
            per += c.idx_bytes_per_token(kvb)
        ops.append(Op("cp_ring_kv", "comm", li, comm_kind="allgather", comm_group=cp, comm_bytes=tok * per,
                      comm_stride=tp))
    return ops


def _streams(m: ModelSpec) -> int:
    """Residual streams passed between pipeline stages (mHC: hc_mult streams of width hidden)."""
    return max((max(l.hc, l.gres_n) for l in m.layers), default=0) or 1


def evaluate(scn: Scenario, model: ModelSpec | None = None) -> Result:
    m = model or get_model(scn.model)
    if scn.formats_override:
        wi = model_cache(m).setdefault("what_if", {})
        key = tuple(scn.formats_override)
        if key not in wi:
            wi[key] = with_formats(m, dict(scn.formats_override))
        m = wi[key]
    warnings: list[str] = []
    if not m.is_pair and m.domain in ("llm", "vlm") and (scn.workload.tri_attn != "fused" or scn.workload.tri_chunk):
        warnings.append("workload.tri_attn / tri_chunk 只用于带三角注意力的结构模型，已忽略")   # 0.119
    if scn.indexer != "reference":
        from .model import has_indexer, with_indexer
        if not has_indexer(m):
            warnings.append(f"indexer = {scn.indexer} 已忽略：该模型没有稀疏注意力 indexer")
        else:
            ix = model_cache(m).setdefault("indexer", {})
            if scn.indexer not in ix:
                ix[scn.indexer] = with_indexer(m, scn.indexer)
            m = ix[scn.indexer]
    if scn.attn_mode != "reference":
        from .model import with_attn_mode
        if not m.kv_cache:
            warnings.append(f"attn_mode = {scn.attn_mode} 已忽略：该模型本身就是整序列前向（编码器 / DiT），注意力按发布")
        elif scn.serving.phase != "prefill":
            raise ValueError("attn_mode = bidir（双向、无 KV cache）只用于 prefill / 整序列前向；decode 需要因果 KV cache")
        else:
            am = model_cache(m).setdefault("attn_mode", {})
            if scn.attn_mode not in am:
                am[scn.attn_mode] = with_attn_mode(m, scn.attn_mode)
            m = am[scn.attn_mode]
            warnings.append("attn_mode = bidir（what-if）：自注意力不加因果掩码（整 S² 打分），prefill 不写 KV cache")
    if scn.hyena_prefill != "reference":
        from .model import with_hyena_prefill
        if not any(L.core is not None and L.core.lin == "hyena" for L in m.layers):
            warnings.append(f"hyena_prefill = {scn.hyena_prefill} 已忽略：该模型没有 Hyena 层")
        else:
            hp = model_cache(m).setdefault("hyena_prefill", {})
            if scn.hyena_prefill not in hp:
                hp[scn.hyena_prefill] = with_hyena_prefill(m, scn.hyena_prefill)
            m = hp[scn.hyena_prefill]
            if scn.hyena_prefill == "fused":
                warnings.append("hyena_prefill = fused（what-if）：MR / LR 的 FFT 卷积按单个片上内核计，频谱放不下时按四步 FFT 多遍")
    if scn.serving.audio_s and not scn.serving.image_tokens and getattr(m.vision, "audio", None) is None:
        warnings.append("serving.audio_s 已忽略：该模型没有音频编码器")
    if (scn.serving.images or scn.serving.videos or scn.serving.audio_s) and not scn.serving.image_tokens:
        if m.vision is None:
            if scn.serving.images or scn.serving.videos:
                warnings.append(("serving.images 已忽略" if scn.serving.images else "serving.videos 已忽略")
                            + "：该模型没有视觉编码器（只有 VLM 计图像 / 视频）")
        else:
            scn = expand_images(scn, m)     # 0.62: image positions join prompt / ctx (KV, prefill length)
    sv, lay = scn.serving, scn.layout
    if lay.cp > 1 and not m.kv_cache:     # 0.90: say why, not just "layout invalid"
        raise ValueError(f"layout.cp = {lay.cp}（上下文并行）只用于 LLM prefill；视频 / 蛋白模型的序列切分请用 sp"
                         + ("（结构模型为 DAP）" if m.is_pair else "（Ulysses）"))
    if not lay.valid_for(m.is_moe, full=not m.kv_cache, pair=m.is_pair):
        raise ValueError(f"layout {lay.label} invalid for {'MoE' if m.is_moe else 'dense'} model "
                         + ("(structure models: tp = 1 — TP of the pair stack not modelled; use DAP (sp) / DP / PP)"
                            if m.is_pair else
                            "(video / protein models: ep = etp = 1)" if not m.kv_cache else
                            "(MoE needs ep·etp = tp·dp; dense needs dp=ep=etp=1; sp only for video / protein models)"))
    if lay.pp > m.n_layers:
        raise ValueError("pp exceeds layer count")
    if lay.cp > 1:
        _cp_check(m, sv)
    sys = System(scn.chip, scn.mem_id, scn.mem_eff, scn.link, scn.d2d_link, scn.package_eff, scn.net, scn.node_cards,
                 scn.fabric, lay.cards, scn.mem_cap_GB, scn.kv_mem_frac, scn.kv_mem_eff)
    if scn.chip.tile_io and not scn.chip.tile_io_Bpc:   # 0.135: DRAM bytes / cycle for the CTA tile I/O
        sys = replace(sys, chip=replace(scn.chip, tile_io_Bpc=sys.dram_GBps / scn.chip.freq_ghz))
    if scn.chip.slc_bytes and scn.chip.slc_bytes >= sys.dram_bytes:
        warnings.append("SLC 容量不小于每卡 DRAM 容量——不现实的设计点（SLC 不增加容量，按包含式缓存计）")
    if sys.package_cards > 1 and lay.cards > sys.package_cards and lay.cards % sys.package_cards:
        warnings.append(f"package_cards = {sys.package_cards} 不整除总卡数 {lay.cards}：末尾封装未满，"
                        "按每个通信组的整除部分近似分层")
    if scn.package_cards > 1 and not scn.d2d_enabled:
        warnings.append(f"package_cards = {scn.package_cards} 未生效：D2D 关闭（单片大 die，每卡一个封装）——"
                        "需 d2d_enabled = true 才按芯粒封装分层")
    if scn.d2d_enabled and scn.package_cards == 1:
        warnings.append("d2d_enabled = true 但 package_cards = 1：每个封装只有一个 die，D2D 层无流量")
    if sys.node_cards > 0 and lay.cards > sys.node_cards and lay.cards % sys.node_cards:
        warnings.append(f"node_cards = {sys.node_cards} 不整除总卡数 {lay.cards}：末尾节点未满，"
                        "按每个通信组的整除部分近似分层")
    # 0.90 (0.91: pin_bw; 0.108 audit: lru too — its hits are also served at slc_GBps, previously silent)
    if scn.chip.slc_bytes and scn.chip.slc_policy in ("pin", "lru") and scn.chip.slc_GBps < sys.dram_GBps:
        warnings.append(f"SLC 带宽 {scn.chip.slc_GBps:.0f} GB/s 低于 DRAM {sys.dram_GBps:.0f} GB/s：{scn.chip.slc_policy} 策略把热数据放在更慢的 "
                        "SLC 上，读得反而更慢（与计算重叠时可能被掩盖；serial / kernel 下可见）——提高 slc_GBps，或改用 pin_bw（按带宽均衡分流）" + (" / lru" if scn.chip.slc_policy == "pin" else ""))
    if not m.kv_cache:
        if scn.pd.enabled:              # 0.90: was silently ignored
            warnings.append("PD 分离（pd.enabled）对视频 / 蛋白模型无效：没有 KV 缓存，也没有 decode 阶段——按单一部署评估，PD 设置被忽略")
        return _evaluate_full(scn, m, sys, warnings)
    if scn.workload.step_skip:
        warnings.append("workload.step_skip 已忽略：只用于视频 / 图像生成（去噪步）")
    spec_k = sv.spec_k if (sv.phase == "decode" and m.mtp_layers) else 0
    if m.draft == "dspark" and sv.spec_k:
        if sv.phase == "decode" and spec_k > m.draft_block:
            warnings.append(f"spec_k {spec_k} 超过 DSpark 草稿块 {m.draft_block}：按 {m.draft_block} 计")
            spec_k = m.draft_block
        if sv.phase == "prefill":
            spec_k = sv.spec_k          # prefill seeds the DSpark draft windows (no draft tokens)
    need_ctx = sv.ctx + 1 + spec_k if sv.phase == "decode" else sv.prompt
    if m.max_ctx and need_ctx > m.max_ctx:    # 0.61.4: evaluated as asked, but say the release does not cover it
        warnings.append(f"上下文 {need_ctx} 超过发布 config 的 max_position_embeddings {m.max_ctx}"
                        "（需要 RoPE 外推 / YaRN 等，发布未必支持；KV 与注意力仍按所给长度计）")
    if sv.spec_k and not m.mtp_layers and sv.phase == "decode":
        warnings.append("spec_k 已忽略：该模型没有 MTP 模块（独立草稿模型的投机解码未建模）")
    pp = lay.pp
    skew = _moe_skew(m, sv, lay, warnings)
    rskew = skew                     # 0.132: what the phases route with (SkewSpec carries the busiest rank's shape)
    if skew != 1.0:
        from .ir import SkewSpec, hot_rank_probs
        f_ = next(L.ffn for L in m.layers if L.ffn.kind == "moe")
        if sv.moe_expert_load:
            rskew = SkewSpec(skew, hot_rank_probs(tuple(sv.moe_expert_load), lay.ep, f_.top_k))
        elif sv.moe_skew_spread != "popular":
            rskew = SkewSpec(skew, (), sv.moe_skew_spread)
    if sv.moe_rank_max != "off" and lay.ep > 1 and any(L.ffn.kind == "moe" for L in m.layers):
        from .ir import SkewSpec, hot_rank_probs
        f_ = next(L.ffn for L in m.layers if L.ffn.kind == "moe")
        hp = hot_rank_probs(tuple(sv.moe_expert_load), lay.ep, f_.top_k) if sv.moe_expert_load else ()
        sp = rskew.spread if isinstance(rskew, SkewSpec) else "popular"
        rskew = SkewSpec(skew, hp, sp, sv.moe_rank_max)
    # 0.134: node-deduplicating EP all-to-all (DeepEP normal kernels) — serving.moe_a2a; "auto" = prefill only (decode
    # runs the low-latency kernels, which send every token-expert pair over RDMA)
    ep_nodes = 1
    a2a = sv.moe_a2a if sv.moe_a2a != "auto" else ("node" if sv.phase != "decode" else "pair")
    if a2a == "node" and lay.ep > 1 and sys.node_cards > 0 and not sys.fabric.enabled:
        from .schedule import _members
        import math as _m
        ep_nodes = lay.ep // _m.gcd(max(1, _members(sys.node_cards, lay.ep, lay.etp)), lay.ep)
    if sv.phase == "decode":
        mb = sv.microbatches or min(pp, sv.batch)
        mb = max(1, min(mb, sv.batch))
        q = 1 + spec_k
        ph = Phase("decode", _cdiv(sv.batch, mb), q, sv.ctx, skew=rskew, ep_nodes=ep_nodes)
        ctx_cap = sv.ctx + q
    else:
        mb = sv.microbatches or min(pp, sv.batch)
        mb = max(1, min(mb, sv.batch))
        ph = Phase("prefill", _cdiv(sv.batch, mb), sv.prompt - sv.prefix_cached, sv.prefix_cached, skew=rskew,
                   ep_nodes=ep_nodes, labels=sv.labels)
        ctx_cap = sv.prompt
    cp_ring = None
    if lay.cp > 1:
        # 0.88 context parallel (zigzag ring attention): rank i holds chunks i and 2cp−1−i of every sequence, so every
        # rank has q = ⌈S/cp⌉ query tokens and exactly 1/cp of the causal work S²/2.  A chunked-prefill phase of q new
        # tokens behind a prefix c does q·(c + q/2) — c = S²/(2·cp·q) − q/2 makes it equal; every token-linear op sees
        # q tokens.  So a CP rank = an attention-DP replica prefilling one chunk of each sequence (virtual batch ×cp);
        # resident KV = its own shard (the two in-flight ring blocks of one layer are not counted); the KV blocks
        # circulate as one all-gather of the rank's block per attention layer over the CP group (stride TP).
        cp, S = lay.cp, sv.prompt
        q_loc = _cdiv(S, cp)
        c_eq = max(0, round(S * S / (2 * cp * q_loc) - q_loc / 2))
        if any(L.core is not None and L.core.kind in ("gqa", "mla") and not L.core.causal for L in m.layers):
            # 0.107: attn_mode = bidir — no mask, every rank's q queries see all S keys (was the causal S²/(2cp):
            # CP silently removed up to half of the bidirectional attention work)
            c_eq = S - q_loc
        ph = Phase("prefill", _cdiv(sv.batch, mb) * cp, q_loc, c_eq, skew=rskew, cp=cp, cp_len=S, ep_nodes=ep_nodes,
                   labels=sv.labels)
        ctx_cap = q_loc
        cp_ring = (cp, lay.tp, _cdiv(sv.batch, mb) * q_loc / lay.dp)   # tokens of this rank's block per step
        lay = Layout(lay.pp, lay.tp, lay.dp * cp, lay.ep, lay.etp, 1)   # internal: CP ranks as attention replicas
    if lay.dp > 1 and ph.batch < lay.dp and m.kv_cache:     # 0.87
        warnings.append(f"注意力 DP 副本空闲：每个微批只有 {ph.batch} 条序列，DP = {lay.dp}，{lay.dp - ph.batch} 个注意力副本"
                        "无事可做（长上下文单请求 prefill 请用 TP / indexer = key_split，或增大 batch）")
    sh = lay.shard
    stages = []
    n_mtp = (len(m.mtp_layers) if m.draft == "dspark" else min(spec_k, len(m.mtp_layers))) if spec_k else 0
    b_rank = _cdiv(sv.batch, scn.layout.dp)   # sequences whose KV lives on one rank (all micro-batches; CP: a shard each)
    cache: dict = {}
    groups = layer_groups(m)
    store_memo = model_cache(m).setdefault("stage_storage", {})
    ops_memo = model_cache(m).setdefault("layer_sums", {})
    if len(ops_memo) > 200_000:
        ops_memo.clear()
    if fabric.log() is not None:     # fabric_report: run every collective (no layer memo)
        ops_memo = {}
    mm = _memos(sys, scn.mapping, m)
    def _run(plan_):
        stages = []
        for st in plan_:
            agg = dict.fromkeys(_SUM_KEYS, 0.0)
            if st.has_embed:
                _acc(agg, _sum_ops(embed_ops(m, ph, sh), sys, scn.mapping, m, mm))
            counts: dict = {}
            for li in range(st.first, st.last):
                g = groups[li]
                first_li, n = counts.get(g, (li, 0))
                counts[g] = (first_li, n + 1)
            if fabric.log() is not None:     # count every stage's collectives (ops_memo is a private dict here)
                cache.clear()
                ops_memo.clear()
            for g, (li, n) in counts.items():
                if g not in cache:
                    okey = (g, ph, sh, scn.mapping, scn.chip, _links(scn))
                    cache[g] = ops_memo.get(okey)
                    if cache[g] is None:
                        fabric.set_mult(n)
                        cache[g] = ops_memo[okey] = _sum_ops(build_rank_ops(m, li, ph, sh, glue=scn.chip.gpu_glue), sys, scn.mapping, m, mm)
                        fabric.set_mult(1)
                _acc(agg, cache[g], n)
            if cp_ring is not None:
                _acc(agg, _sum_ops(_cp_ring_ops(m, st.first, st.last, cp_ring), sys, scn.mapping, m, mm))
            tail = _tail_ops(m, st.has_head, ph, sh, spec_k, scn.chip.gpu_glue)
            if tail:                        # 0.61: non-head stages have no tail ops (summing zeros is a no-op)
                _acc(agg, _sum_ops(tail, sys, scn.mapping, m, mm))
            skey = (st.first, st.last, st.has_embed, st.has_head, sh, ctx_cap, n_mtp)
            store = store_memo.get(skey)
            if store is None:
                store = store_memo[skey] = stage_storage(m, st.first, st.last, st.has_embed, st.has_head, sh, ctx_cap,
                                                         n_mtp)
            # 0.63: activations streamed (stream=True: staging ≤ the SRAM's activation half, spills in agg["act"]);
            # a working set beyond the runtime reserve needs its excess in DRAM 「假设」
            mp = plan(store, b_rank, sys.chip.sram_bytes, sys.dram_bytes, agg["max_act"],
                      act_need=max(0.0, agg["max_act_tot"] - RUNTIME_RESERVE), stream=True,
                      slc_bytes=sys.chip.slc_bytes, slc_policy=sys.chip.slc_policy, kv_frac=sys.kv_frac)
            dram = step_dram_bytes(mp, store, agg, sys.chip.slc_GBps / sys.dram_GBps)
            link_bw, sync, link_bytes, d2d_b = agg["link_bw"], agg["sync"], agg["link_bytes"], agg["d2d_bytes"]
            net_b = agg["net_bytes"]
            wire = {t: agg["wire_" + t] for t in _TIERS}
            extra = []
            if pp > 1 and not st.has_head:
                act = ph.batch * ph.q * m.hidden * red_bytes(m) / lay.dp * _streams(m)   # 0.61.1: ≥ bf16; × mHC streams
                bw, a, fd, fn, tier = _p2p_full(sys, act, st.index, lay.cards // pp)
                link_bw += bw; sync += a; link_bytes += act; d2d_b += act * fd; net_b += act * fn
                _add_wire(wire, {"d2d": act * fd, "net": act * fn, "link": act * (1 - fd - fn)})
                extra.append((bw, tier))
            t_dram = _dram_t(sys, dram)
            so, pre = {}, (link_bw, sync, d2d_b, net_b, dict(wire))
            win = max(agg["t_arr"], agg["t_vec"], t_dram, _slc_time(sys, dram))
            if sys.fabric.enabled:
                link_bw, sync, d2d_b, net_b = _stage_link(sys, agg, link_bw, sync, d2d_b, net_b, extra, win, st.index,
                                                          wire, out=so)
            if scn.exec_overlap == "tbo":   # 0.70: two micro-batches each stream the stage's weights once
                t_dram += dram["weights"] / (sys.dram_GBps * 1e9)
            s_add = 0.0
            if scn.layer_overhead_us:      # 0.69 (off by default: bit-identical)
                s_add = (st.last - st.first) * scn.layer_overhead_us * 1e-6     # [first, last) layers
                sync += s_add
                agg["kern"] += s_add
            agg["fixed"] = s_add
            _ps = sys.chip.dispatch_s()[0]
            if _ps:                        # 0.203: static-graph launch + token sync, once per step per stage
                s_add += _ps
                sync += _ps
                agg["disp"] += _ps
            t_pipe, pctx, tpipe = 0.0, None, None
            if scn.exec_overlap == "pipe":  # 0.98 (opt-in)
                segs = ([(embed_ops(m, ph, sh), 1)] if st.has_embed else [])
                segs += [(build_rank_ops(m, li, ph, sh, glue=scn.chip.gpu_glue), n) for g, (li, n) in counts.items()]
                if cp_ring is not None:
                    segs.append((_cp_ring_ops(m, st.first, st.last, cp_ring), 1))
                if tail:
                    segs.append((tail, 1))

                def tpipe(segs=segs, mp=mp, store=store, dram=dram, extra=extra):
                    return _pipe_time(sys, scn.mapping, m, mm, segs, mp, store, dram, sum(b for b, _ in extra))
                t_pipe = tpipe()

            def mk(link_bw, sync, d2d_b, net_b, wire, t_pipe, agg=agg, dram=dram, t_dram=t_dram, link_bytes=link_bytes):
                stt = StageTime(agg["t_arr"], agg["t_mac"], agg["t_feed"], agg["t_vec"], t_dram, link_bw, sync,
                                dram["total"], link_bytes, agg["flops"], agg["t_ideal"], _slc_time(sys, dram),
                                dram.get("slc", 0.0), d2d_b, net_b, agg["t_arr_attn"],
                                ((dram["kv_read"] + dram["kv_write"]) * sys.kv_time_scale + dram["state"]) / (sys.dram_GBps * 1e9),
                                scn.exec_overlap, max(0.0, wire["d2d"]), max(0.0, wire["link"]), max(0.0, wire["net"]),
                                t_pipe)
                stt = _a2a_sm_share(replace(stt, t_kern=min(agg["kern"], stt.t_sync), t_disp=agg["disp"],
                                            t_fixed=agg.get("fixed", 0.0)), ph, sv)
                if sys.chip.attn_vec_overlap:     # 0.104 (opt-in)
                    stt = replace(stt, t_vec_attn=agg["t_vec_attn"], attn_ovl=True)
                return stt
            stt = mk(link_bw, sync, d2d_b, net_b, wire, t_pipe)
            picks = None
            if tpipe is not None and so:
                stt, picks = _pipe_repick(sys, agg, pre, extra, win, st.index, so, s_add, tpipe, mk, stt)
            if tpipe is not None:
                pctx = (sys, scn.mapping, m, mm, segs, mp, store, sum(b for b, _ in extra), picks, {})
            stages.append(StageResult(st.index, (st.first, st.last), stt, mp, dram, agg["conv"], flops_u=agg["flops_u"],
                                      sram_bytes=agg["sram"], ideal_w=agg["t_ideal_w"], vec_w=agg["t_vec_w"],
                                      conv_w=agg["t_conv_w"], exp_acts=_exp_acts(agg), pipe_ctx=pctx,
                                      exp_route=_exp_route(m, ph, sh)))
        return stages

    plan_, pinfo = _plan_by_cost(scn, m, sys, pp, groups, ph, sh, mm, ops_memo, store_memo, ctx_cap, n_mtp, b_rank,
                                 embed_ops(m, ph, sh), _tail_ops(m, True, ph, sh, spec_k, scn.chip.gpu_glue),
                                 ph.batch * ph.q * m.hidden * red_bytes(m) / lay.dp * _streams(m))
    stages = _pick_split(_run, plan_, plan_stages(m.n_layers, pp)) if pinfo else _run(plan_)
    vis = _vision_cost(scn, m, sys, stages) if (m.vision is not None and (sv.images or sv.videos or sv.audio_s)
                                                 and sv.image_tokens) else None
    heavy = max(range(len(stages)), key=lambda i: stages[i].time.total)
    tick = stages[heavy].time.total
    _pp_imbalance_warn(stages, tick, warnings, scn.pp_split)
    _dequant_warn(stages, sys, warnings)
    cap_heavy = max(range(len(stages)), key=lambda i: stages[i].mem.dram_need)
    fits = all(s.mem.fits for s in stages)
    if not fits:
        warnings.append(f"容量不足：流水级 {cap_heavy} 每卡需要 {stages[cap_heavy].mem.dram_need / 2**30:.1f} GiB，"
                        f"超过每卡 {sys.dram_bytes / 2**30:.0f} GiB"
                        + (f"（含视觉编码器权重 {stages[cap_heavy].mem.pipe_w / 2**30:.2f} GiB）"
                           if vis and stages[cap_heavy].mem.pipe_w else ""))
    if m.coverage == "proxy":
        warnings.append("「架构代理」模型：结果为近似值（见建模说明）")
    host = None
    if scn.host_step_us or scn.host_seq_us:     # 0.164 (opt-in): host / runtime time per engine step
        seqs = _cdiv(sv.batch, lay.dp)
        h = (scn.host_step_us + scn.host_seq_us * seqs) * 1e-6
        if sv.phase == "decode":
            h /= scn.host_steps                  # multi-step scheduling: one host pass per host_steps device steps
        host = {"s": h, "overlap": scn.host_overlap, "steps": scn.host_steps if sv.phase == "decode" else 1,
                "seqs_per_rank": seqs}
    if sv.phase == "decode":
        step = max(mb, pp) * tick
        if host:
            dev = step
            step = dev + host["s"] if scn.host_overlap == "serial" else max(dev, host["s"])
            host.update(device_s=dev, exposed_s=step - dev, host_bound=host["s"] > dev)
        e_tok = spec_expected_tokens(spec_k, sv.spec_accept) if spec_k else 1.0
        tpot = step / e_tok
        thr = sv.batch * e_tok / step
        ttft = 0.0
    else:
        # 0.62: the vision encoder runs before the LLM prefill of the step (serial 「假设」)
        step = (mb + pp - 1) * tick + (vis["s"] if vis else 0.0)
        if host:
            dev = step
            step = dev + host["s"] if scn.host_overlap == "serial" else max(dev, host["s"])
            host.update(device_s=dev, exposed_s=step - dev, host_bound=host["s"] > dev)
        e_tok = 1.0
        tpot = 0.0
        ttft = step
        thr = sv.batch * sv.prompt / step
    return Result(scn, m, stages, tick, step, e_tok, tpot, ttft, thr, thr / lay.cards, fits,
                  stages[heavy].time.bound, heavy, warnings, mb, moe_skew=skew, vision=vis, host=host)



def _vision_cost(scn: Scenario, m: ModelSpec, sys: System, stages: list) -> dict:
    """VLM vision encoder (0.62; ``core/vision.py``).  Placement 「假设」: data-parallel encoder on the first pipeline
    stage (vLLM ``mm_encoder_tp_mode = data``): every card of stage 0 holds the encoder weights (bf16) and encodes its
    share of the step's images (a DP rank's requests on its TP cards); the encode runs before the LLM prefill, serially.
    Decode steps carry only the weights.  Adds the weights (and, at prefill, the encoder's largest activation beyond the
    stage's own) to stage 0's DRAM need."""
    sv, lay = scn.serving, scn.layout
    v = m.vision
    g = image_grid(v, sv.image_w, sv.image_h)
    w = v.params * 2.0
    out = {"images": sv.images, "px": [sv.image_w, sv.image_h], "resized_px": list(g.px), "patches": [g.ph, g.pw],
           "merged": g.merged, "tokens_per_image": g.llm_tokens, "image_tokens": sv.image_tokens,
           "text_prompt": sv.prompt - sv.image_tokens, "label": v.label, "params_B": v.params / 1e9,
           "weights_GB": w / 1e9, "s": 0.0, "tflop": 0.0, "per_card": 0, "cards": 0}
    vg = video_grid_of(m, sv)
    if vg is not None:
        out["video"] = {"videos": sv.videos, "frames": vg.frames, "px": [sv.video_w, sv.video_h],
                        "resized_px": list(vg.slice.px), "slices": vg.t, "patches": [vg.slice.ph, vg.slice.pw],
                        "tokens_per_slice": vg.slice.llm_tokens, "tokens_per_video": vg.llm_tokens, "cap": sv.video_cap and vg.slice.tframes == 1,
                        "frames_per_slice": vg.slice.tframes if vg.slice.tframes > 1 else m.vision.tpatch}
    act = 0.0
    if sv.phase == "prefill":
        # 0.165: encoder passes = images (one grid each) + video temporal slices (the tower's cu_seqlens split a clip
        # per t: attention within a slice, Conv3d patch-embed over temporal_patch frames — the same pass as a still)
        units = [(g, sv.images)] if sv.images else []
        if vg is not None:
            units.append((vg.slice, sv.videos * (vg.t - (1 if vg.tail is not None else 0))))
            if vg.tail is not None:                               # 0.166 Kimi: shorter last chunk
                units.append((vg.tail, sv.videos))
        tot = None
        for gi, cnt in units:
            per_rank = _cdiv(sv.batch, lay.dp) * cnt             # passes of one DP rank's requests in the step
            c0 = max(1, lay.tp * lay.sp)                          # that rank's cards in the first stage
            per_card = _cdiv(per_rank, c0)
            busy = _cdiv(per_rank, per_card)
            dp_busy = min(lay.dp, _cdiv(sv.batch, _cdiv(sv.batch, lay.dp)))
            key = ("vision", gi, per_card, scn.mapping, scn.chip, scn.mem_id, scn.mem_eff)
            memo = model_cache(m).setdefault("vision", {})
            c = memo.get(key)
            if c is None:
                if len(memo) > 4096:
                    memo.clear()
                c = memo[key] = _component_time([(vision_ops(v, gi, per_card), 1)], sys, scn.mapping, "bf16")
            act = max(act, vision_act_peak(v, gi, per_card))
            n = busy * dp_busy
            # 0.63 (external review): a card's time is set by its ⌈images / card⌉, but only real images are work — MAC /
            # vector counts and FLOPs × images / per_card (3 images on TP 2 counted 4); byte counts per busy card
            k_w = sv.batch * cnt / per_card
            acts = {k: x * (k_w if k in ("mac", "vec") else n) for k, x in c["acts"].items()}
            if tot is None:
                tot = dict(s=c["s"], tflop=c["tflop"] * k_w)
                if gi is g:
                    tot["tflop_per_image"] = c["tflop"] / per_card
                tot.update(per_card=per_card, cards=n, bound=c["bound"], dram_GB=c["dram_GB"] * n, acts=acts)
            else:
                if c["s"] > tot["s"]:
                    tot["bound"] = c["bound"]
                tot.update(s=tot["s"] + c["s"], tflop=tot["tflop"] + c["tflop"] * k_w, dram_GB=tot["dram_GB"] + c["dram_GB"] * n,
                           per_card=max(tot["per_card"], per_card), cards=max(tot["cards"], n),
                           acts={k: tot["acts"].get(k, 0.0) + acts.get(k, 0.0) for k in set(tot["acts"]) | set(acts)})
            if vg is not None and (gi is vg.slice or gi is vg.tail):
                ov = out["video"]
                ov.update(s=ov.get("s", 0.0) + c["s"], tflop=ov.get("tflop", 0.0) + c["tflop"] * k_w,
                          slices_per_card=max(ov.get("slices_per_card", 0), per_card))
        au = v.audio if sv.audio_s else None
        if au is not None:
            # 0.175 audio encoder: same placement as the vision tower (each rank's requests split over its stage-0 cards)
            per_rank = _cdiv(sv.batch, lay.dp)
            per_card = _cdiv(per_rank, max(1, lay.tp * lay.sp))
            c = _component_time([(audio_ops(au, sv.audio_s, per_card), 1)], sys, scn.mapping, "bf16")
            k_w = sv.batch / per_card
            out["audio"] = {"seconds": sv.audio_s, "tokens": au.tokens(sv.audio_s), "params_M": au.params / 1e6,
                            "s": c["s"], "tflop": c["tflop"] * k_w, "clips_per_card": per_card}
            if tot is None:
                tot = dict(s=c["s"], tflop=c["tflop"] * k_w, per_card=0, cards=per_card and _cdiv(per_rank, per_card),
                           bound=c["bound"], dram_GB=c["dram_GB"], acts=dict(c["acts"]))
            else:
                tot.update(s=tot["s"] + c["s"], tflop=tot["tflop"] + c["tflop"] * k_w)
        if tot is not None:
            out.update(tot)
    mp = stages[0].mem
    need = mp.dram_need + w + max(0.0, act - mp.act_total)
    stages[0].mem = replace(mp, dram_need=need, fits=need <= mp.dram_cap, pipe_w=mp.pipe_w + w)
    return out

def _cfg_chain(seqs: int, mb: int, per_req: int) -> int:
    """Largest set of microbatches linked by a shared request (sequences request-major, microbatches of ⌈S/mb⌉
    consecutive sequences): the microbatches of one connected group depend on each other at every denoise step."""
    if per_req <= 1 or mb <= 1:
        return 1
    msz = _cdiv(seqs, mb)
    nmb = _cdiv(seqs, msz)
    parent = list(range(nmb))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for r in range(_cdiv(seqs, per_req)):
        a, b = (r * per_req) // msz, (min(seqs, (r + 1) * per_req) - 1) // msz
        for j in range(a + 1, b + 1):
            parent[find(j)] = find(a)
    sizes: dict = {}
    for i in range(nmb):
        sizes[find(i)] = sizes.get(find(i), 0) + 1
    return max(sizes.values())


def _evaluate_full(scn: Scenario, m: ModelSpec, sys: System, warnings: list[str]) -> Result:
    """Non-autoregressive request: video clip (steps × CFG forwards over every latent token) or protein batch
    (one encoder forward).  Same op-graph → mapping → memplan → schedule path as the LLM phases.

      forward sequences  S = batch · cfg (video) or batch (protein); microbatches mb = min(pp, S) unless set
      video clip latency = steps · max(mb, k + pp − 1) · t_stage  (k = microbatches holding the CFG forwards of one
                           request, which must all finish a step before the next; k = 1 → max(mb, pp): the pipeline
                           is kept full across steps by independent micro-batches)
      protein latency    = (mb + pp − 1) · t_stage              (one pass, like an LLM prefill)
      throughput         = batch · units / latency               (frames/s or sequences/s per replica)
    """
    sv, lay = scn.serving, scn.layout
    wl = resolve_workload(m, scn.workload)
    warnings.extend(wl.warnings)
    if sv.spec_k:
        warnings.append("投机解码对视频 / 蛋白质模型无意义，已忽略")
    seqs = sv.batch * wl.seqs_per_request
    pp = lay.pp
    mb = max(1, min(sv.microbatches or min(pp, seqs), seqs))
    pd = wl.pair
    if pd is not None and lay.sp > 1 and scn.workload.sample_split and pd.samples > 1:
        pd = replace(pd, split=True)
    ph = Phase("full", _cdiv(seqs, mb), wl.tokens, wl.ctx, frames=wl.frames, aux=wl.aux, pair=pd)
    sh = lay.shard
    ab = _fmt(m.act_fmt).bytes
    groups = layer_groups(m)
    ops_memo = model_cache(m).setdefault("layer_sums", {})
    store_memo = model_cache(m).setdefault("stage_storage", {})
    if len(ops_memo) > 200_000:
        ops_memo.clear()
    if fabric.log() is not None:     # fabric_report: run every collective (no layer memo)
        ops_memo = {}
    stages = []
    resid = _cdiv(seqs, lay.dp) * _cdiv(wl.tokens, lay.sp) * m.hidden * ab   # residual streams of in-flight sequences
    pair_act = 0.0
    if wl.pair is not None:      # structure models: the pair representation N²·c_z rides along with the single track
        pair_act = _cdiv(wl.pair.n, lay.sp) * wl.pair.n * m.workload.pair_dim * ab   # DAP: residue-axis slice
        resid += _cdiv(seqs, lay.dp) * pair_act
        # 0.118 audit: the activation plan assumes fused (flash-style) triangle attention — no H·N³ logits tensor.
        # Unchunked reference kernels (HF / esm ESMFold with chunk_size None, plain PyTorch OpenFold / Boltz without
        # chunking) materialise one per block; flag when that alone exceeds the card's DRAM
        th = max((c.heads for L in m.layers for c in L.pair_cores if c.kind == "tri_att"), default=0)
        lg = _cdiv(wl.pair.n, lay.sp) * wl.pair.n ** 2 * th * ab
        if th and lg > sys.dram_bytes and not wl.pair.tri_chunk:
            warnings.append(f"三角注意力 logits（{th} 头 × N³，N = {wl.pair.n}）未分块时 {lg / 2**30:.0f} GiB/卡，超过 DRAM "
                            f"{sys.dram_bytes / 2**30:.0f} GiB：结果按融合（flash 式）三角注意力计；参考实现需设 chunk_size"
                            "（只降峰值显存，不改计算量；workload.tri_attn = chunked / reference 可评估）")
    fsdp = lay.sp * lay.dp if scn.workload.dit_fsdp and wl.kind == "gen" and lay.sp * lay.dp > 1 else 0
    if scn.workload.dit_fsdp and not fsdp:
        warnings.append("workload.dit_fsdp 已忽略：" + ("只用于视频 DiT" if wl.kind != "gen" else
                                                       "需要 SP · DP > 1（FSDP 在同一流水级的数据 / 序列并行卡之间分片）"))
    def _run(plan_, ph=ph, resid=resid):
        stages = []
        for st in plan_:
            agg = dict.fromkeys(_SUM_KEYS, 0.0)
            if fabric.log() is not None:
                ops_memo.clear()
            if st.has_embed:
                _acc(agg, _sum_ops(full_io_ops(m, ph, sh, "pre"), sys, scn.mapping, m))
            counts: dict = {}
            for li in range(st.first, st.last):
                g = groups[li]
                first_li, n = counts.get(g, (li, 0))
                counts[g] = (first_li, n + 1)
            for g, (li, n) in counts.items():
                okey = (g, ph, sh, scn.mapping, scn.chip, _links(scn))
                hit = ops_memo.get(okey)
                if hit is None:
                    fabric.set_mult(n * layer_repeat(m.layers[li], ph))
                    hit = ops_memo[okey] = _sum_ops(build_rank_ops(m, li, ph, sh, glue=scn.chip.gpu_glue), sys, scn.mapping, m)
                    fabric.set_mult(1)
                _acc(agg, hit, n * layer_repeat(m.layers[li], ph))
            if st.has_head:
                _acc(agg, _sum_ops(full_io_ops(m, ph, sh, "post"), sys, scn.mapping, m))
            skey = (st.first, st.last, st.has_embed, st.has_head, sh, 0, 0)
            store = store_memo.get(skey)
            if store is None:
                store = store_memo[skey] = stage_storage(m, st.first, st.last, st.has_embed, st.has_head, sh, 0, 0)
            mp = plan(store, 0, sys.chip.sram_bytes, sys.dram_bytes, agg["max_act"],
                      act_need=agg["max_act_tot"] + resid, stream=True,
                      slc_bytes=sys.chip.slc_bytes, slc_policy=sys.chip.slc_policy, kv_frac=sys.kv_frac)
            dram = step_dram_bytes(mp, store, agg, sys.chip.slc_GBps / sys.dram_GBps)
            link_bw, sync, link_bytes, d2d_b = agg["link_bw"], agg["sync"], agg["link_bytes"], agg["d2d_bytes"]
            net_b = agg["net_bytes"]
            wire = {t: agg["wire_" + t] for t in _TIERS}
            extra_f: dict = {}
            if fsdp:
                # DiT weights FSDP-sharded over the stage's g = SP·DP ranks (Wan --dit_fsdp): each card keeps w / g plus
                # two gathered layers (prefetch); every stage forward all-gathers the active weights layer by layer
                # (overlapping compute through the stage max rule, α per layer) and writes the gathered copy to DRAM
                # before the usual weight streaming reads it.  An idle expert (Wan2.2) is sharded but not gathered.
                w, nl = mp.stored_w, max(1, st.last - st.first)
                sb = _standby_bytes(m, lay, st.first, st.last)
                act_w = w - sb
                keep = w / fsdp + min(act_w * (fsdp - 1) / fsdp, 2 * act_w / nl)
                need = mp.dram_need - w + keep
                mp = replace(mp, stored_w=w / fsdp, dram_need=need, fits=need <= mp.dram_cap)
                bw, a_, fd, fn = _comm(sys, "allgather", act_w / fsdp, fsdp, stride=lay.tp)  # SP·DP ranks, stride TP
                link_bw += bw; sync += a_ * nl; link_bytes += act_w / fsdp; d2d_b += act_w / fsdp * fd
                net_b += act_w / fsdp * fn
                _add_wire(wire, _comm_wire(sys, "allgather", act_w / fsdp, fsdp, stride=lay.tp))
                if sys.fabric.enabled:
                    extra_f = fabric.collective_full("allgather", act_w / fsdp, fsdp, sys, lay.tp)[4]
                gw = act_w * (fsdp - 1) / fsdp
                dram = {**dram, "weights": dram["weights"] + gw, "total": dram["total"] + gw, "fsdp_gather": gw}
            extra = [(x, t) for t, x in extra_f.items()]
            if pp > 1 and not st.has_head:
                act = ph.batch * (_cdiv(ph.q, lay.sp) * m.hidden * ab + pair_act) / lay.dp
                bw, a, fd, fn, tier = _p2p_full(sys, act, st.index, lay.cards // pp)
                link_bw += bw; sync += a; link_bytes += act; d2d_b += act * fd; net_b += act * fn
                _add_wire(wire, {"d2d": act * fd, "net": act * fn, "link": act * (1 - fd - fn)})
                extra.append((bw, tier))
            t_dram = _dram_t(sys, dram)
            so, pre = {}, (link_bw, sync, d2d_b, net_b, dict(wire))
            win = max(agg["t_arr"], agg["t_vec"], t_dram, _slc_time(sys, dram))
            if sys.fabric.enabled:
                link_bw, sync, d2d_b, net_b = _stage_link(sys, agg, link_bw, sync, d2d_b, net_b, extra, win, st.index,
                                                          wire, out=so)
            if scn.exec_overlap == "tbo":   # 0.70: two micro-batches each stream the stage's weights once
                t_dram += dram["weights"] / (sys.dram_GBps * 1e9)
            s_add = 0.0
            if scn.layer_overhead_us:      # 0.69 (off by default: bit-identical)
                s_add = (st.last - st.first) * scn.layer_overhead_us * 1e-6     # [first, last) layers
                sync += s_add
                agg["kern"] += s_add
            agg["fixed"] = s_add
            _ps = sys.chip.dispatch_s()[0]
            if _ps:                        # 0.203: static-graph launch + token sync, once per step per stage
                s_add += _ps
                sync += _ps
                agg["disp"] += _ps
            t_pipe, tpipe = 0.0, None
            if scn.exec_overlap == "pipe":  # 0.98 (opt-in); FSDP gathers are prefetches (hidden behind the chain)
                segs = [(full_io_ops(m, ph, sh, "pre"), 1)] if st.has_embed else []
                segs += [(build_rank_ops(m, li, ph, sh, glue=scn.chip.gpu_glue), n * layer_repeat(m.layers[li], ph)) for g, (li, n) in counts.items()]
                if st.has_head:
                    segs.append((full_io_ops(m, ph, sh, "post"), 1))
                hid = sum(extra_f.values()) if extra_f else 0.0
                pp_b = sum(b for b, _ in extra) - hid

                def tpipe(segs=segs, mp=mp, store=store, dram=dram, pp_b=pp_b, hid=hid):
                    return _pipe_time(sys, scn.mapping, m, None, segs, mp, store, dram, max(0.0, pp_b), hid)
                t_pipe = tpipe()

            def mk(link_bw, sync, d2d_b, net_b, wire, t_pipe, agg=agg, dram=dram, t_dram=t_dram, link_bytes=link_bytes):
                stt = StageTime(agg["t_arr"], agg["t_mac"], agg["t_feed"], agg["t_vec"], t_dram, link_bw, sync,
                                dram["total"], link_bytes, agg["flops"], agg["t_ideal"], _slc_time(sys, dram),
                                dram.get("slc", 0.0), d2d_b, net_b, agg["t_arr_attn"],
                                ((dram["kv_read"] + dram["kv_write"]) * sys.kv_time_scale + dram["state"]) / (sys.dram_GBps * 1e9),
                                scn.exec_overlap, max(0.0, wire["d2d"]), max(0.0, wire["link"]), max(0.0, wire["net"]),
                                t_pipe)
                stt = _a2a_sm_share(replace(stt, t_kern=min(agg["kern"], stt.t_sync), t_disp=agg["disp"],
                                            t_fixed=agg.get("fixed", 0.0)), ph, sv)
                if sys.chip.attn_vec_overlap:     # 0.104 (opt-in)
                    stt = replace(stt, t_vec_attn=agg["t_vec_attn"], attn_ovl=True)
                return stt
            stt = mk(link_bw, sync, d2d_b, net_b, wire, t_pipe)
            if tpipe is not None and so:
                stt, _ = _pipe_repick(sys, agg, pre, extra, win, st.index, so, s_add, tpipe, mk, stt)
            stages.append(StageResult(st.index, (st.first, st.last), stt, mp, dram, agg["conv"], flops_u=agg["flops_u"],
                                      sram_bytes=agg["sram"], ideal_w=agg["t_ideal_w"], vec_w=agg["t_vec_w"],
                                      conv_w=agg["t_conv_w"], exp_acts=_exp_acts(agg)))
        return stages

    p2p_pay = ph.batch * (_cdiv(ph.q, lay.sp) * m.hidden * ab + pair_act) / lay.dp
    plan_, pinfo = _plan_by_cost(scn, m, sys, pp, groups, ph, sh, _memos(sys, scn.mapping, m), ops_memo, store_memo, 0, 0,
                                 0, full_io_ops(m, ph, sh, "pre"), full_io_ops(m, ph, sh, "post"), p2p_pay, repeat=True)
    # 0.116: text-prefix K/V reuse (model-declared, workload.prefix_kv) — step 1 extracts the prefix K/V, steps 2…n run
    # the image tokens only against the cached prefix (diffusers QwenImage21 kv_cache "extract" / "cached")
    pkv = _prefix_kv(scn, m, wl, warnings)
    stages_x = None
    if pkv:
        T = m.workload.prefix_tokens
        ph_x = replace(ph, kvx=True)
        ph_c = Phase("full", ph.batch, wl.tokens - T, wl.ctx, frames=wl.frames, aux=wl.aux, pair=pd, kvpre=T)
        cache = _prefix_cache_bytes(m, T, lay, ab) * _cdiv(seqs, lay.dp)
        pkv = {"tokens": T, "cache_GiB_card": cache / 2**30}
        chosen: dict = {}

        def _run_x(pl):
            r_ = _run(pl, ph_x, resid + cache)
            chosen[id(r_)] = pl
            return r_
        stages_x = _pick_split(_run_x, plan_, plan_stages(m.n_layers, pp)) if pinfo else _run_x(plan_)
        stages = _run(chosen[id(stages_x)], ph_c, _cdiv(seqs, lay.dp) * _cdiv(wl.tokens - T, lay.sp) * m.hidden * ab
                      + cache)
    else:
        stages = _pick_split(_run, plan_, plan_stages(m.n_layers, pp)) if pinfo else _run(plan_)
    pipe = None
    if wl.kind == "gen":
        pipe = _pipeline_cost(scn, m, wl, sys, warnings)
        if pipe:
            pipe = _place_components(scn, m, pipe, stages, sys, warnings)
    heavy = max(range(len(stages)), key=lambda i: stages[i].time.total)
    tick = stages[heavy].time.total
    _pp_imbalance_warn(stages, tick, warnings, scn.pp_split)
    cap_heavy = max(range(len(stages)), key=lambda i: stages[i].mem.dram_need)
    fits = all(s.mem.fits for s in stages)
    if not fits:
        warnings.append(f"容量不足：流水级 {cap_heavy} 每卡需要 {stages[cap_heavy].mem.dram_need / 2**30:.1f} GiB，"
                        f"超过每卡 {sys.dram_bytes / 2**30:.0f} GiB"
                        + (f"（含文本编码器 / VAE 权重 {stages[cap_heavy].mem.pipe_w / 2**30:.1f} GiB）"
                           if stages[cap_heavy].mem.pipe_w else ""))
    if wl.kind == "gen":
        # 0.63 (external review): the CFG forwards of one request (cond / uncond, + image guidance) must all finish a
        # denoise step before its latent update starts the next one.  Microbatches that share a request form a
        # dependent chain of k microbatches: one step of the chain takes ≥ k + pp − 1 stage ticks, and the stages
        # serve mb microbatches per step → per step max(mb, k_max + pp − 1) ticks (= max(mb, pp) when every request
        # sits in one microbatch).  Was max(mb, pp): cond / uncond in different microbatches ran as independent.
        k_chain = _cfg_chain(seqs, mb, wl.seqs_per_request)
        per_step = max(mb, k_chain + pp - 1) if k_chain > 1 else max(mb, pp)
        if pkv:
            tick_x = max(st.time.total for st in stages_x)
            denoise = per_step * (tick_x + (wl.steps - 1) * tick)
            b_mb = _cdiv(seqs, mb)
            n_r = lay.tp * lay.sp * lay.dp * lay.cp
            pkv.update(extract_ms=tick_x * 1e3, cached_ms=tick * 1e3,
                       extract_tflop_seq=sum(st.flops_u for st in stages_x) * n_r / b_mb / 1e12,
                       cached_tflop_seq=sum(st.flops_u for st in stages) * n_r / b_mb / 1e12,
                       fits=all(st.mem.fits for st in stages_x))
            fits = fits and pkv["fits"]
        else:
            denoise = wl.steps * per_step * tick
        sc_ = _step_cache(scn, m, wl, sys, ph, sh, stages, seqs, mb, pp, pkv, warnings)
        if sc_:
            # skipped steps: every microbatch runs embed + head + residual add instead of the blocks (pp = 1 only)
            denoise -= sc_["skip"] * per_step * (tick - sc_["t_skip"])
            sc_["denoise_saved_s"] = sc_["skip"] * per_step * (tick - sc_["t_skip"])
            sc_["t_skip_ms"] = sc_.pop("t_skip") * 1e3
            fits = fits and sc_["fits"]
        latency = denoise + (pipe["te_s"] + pipe["decode_s"] + pipe["load_s"] if pipe else 0.0)
        if pipe:
            # 0.61.4: tflop_replica is one DP rank's share ⌈B/dp⌉ of the requests → × B / share (was × dp, which
            # counted idle ranks: batch 1 on DP 2 showed the text encoder + VAE twice per request)
            b_ = scn.serving.batch
            pipe = {**pipe, "denoise_s": denoise, "tflop": pipe["tflop_replica"] * b_ / _cdiv(b_, lay.dp)}
    else:
        latency = (mb + pp - 1) * tick
        if scn.workload.step_skip:
            warnings.append("workload.step_skip 已忽略：只用于视频 / 图像生成（去噪步）")
    period = latency
    if wl.kind == "gen" and scn.workload.overlap:
        # cross-request overlap (0.47): with the text encoder on the host CPU (te_cpu) the host encodes request
        # n+1 while the cards denoise / decode request n — two independent resources, a saturated queue →
        # steady-state period = max(card busy, host busy); clip latency unchanged.  Anything that shares the cards
        # (encoder / VAE on a card, offload reloads) stays serial: nothing else is overlapped.
        if pipe and pipe.get("te_cpu"):
            period = max(latency - pipe["te_s"], pipe["te_s"])
            pipe = {**pipe, "period_s": period, "overlap": True}
        else:
            warnings.append("workload.overlap：文本编码器与 DiT 同在卡上（未开 te_cpu），没有独立资源可跨请求重叠——按串行计")
    thr = sv.batch * wl.units / period
    r = Result(scn, m, stages, tick, latency, wl.units, 0.0, 0.0, thr, thr / lay.cards, fits,
               stages[heavy].time.bound, heavy, warnings, mb, latency, wl, pipe)
    if pkv:
        r.prefix_kv = pkv
    if wl.kind == "gen" and sc_:
        r.step_cache = sc_
    return r


# 「来源」 TeaCache4Wan2.1 teacache_generate.py: ret_steps = 1·2, cutoff = steps·2 − 2 (first and last step computed);
# use_ret_steps: ret_steps = 5·2, cutoff = steps·2 (first 5 computed).  The skip test uses the timestep embedding only
# (modulated_inp = e / e0), identical for the cond and uncond forwards → both CFG branches skip the same steps.
STEP_CACHE_FORCED = {"teacache": (1, 1), "teacache_ret": (5, 0)}


def _step_cache(scn, m, wl, sys, ph, sh, stages, seqs, mb, pp, pkv, warnings):
    """0.167: TeaCache-style step skipping for DiT generation.  A skipped forward still runs the patch / timestep /
    text embedders and the head (they precede / follow the block loop in the reference forward) plus one residual
    add over the latent tokens; it keeps one cached residual (tokens × hidden) per CFG branch per sequence."""
    k = scn.workload.step_skip
    if not k:
        return None
    if pp > 1 or pkv:
        warnings.append("workload.step_skip 已忽略：" + ("流水并行（PP > 1）下跳步的级间行为未建模" if pp > 1 else
                                                       "与 prefix_kv 同用未建模"))
        return None
    first, last = STEP_CACHE_FORCED[scn.workload.step_cache]
    cap = max(0, wl.steps - min(wl.steps, first + last))
    if k > cap:
        raise ValueError(f"workload.step_skip = {k} 超过可跳步数 {cap}（{wl.steps} 步，{scn.workload.step_cache} 固定计算"
                         f"前 {first} 步{'与最后 1 步' if last else ''}）")
    lay = scn.layout
    ab = _fmt(m.act_fmt).bytes
    tok = _cdiv(wl.tokens, lay.sp)
    add = Op("step_cache.residual_add", "vector", 0, vec=ph.batch * tok * m.hidden,
             act_bytes=3 * ph.batch * tok * m.hidden * ab)
    ops = full_io_ops(m, ph, sh, "pre") + full_io_ops(m, ph, sh, "post") + [add]
    c = _component_time([(ops, 1)], sys, scn.mapping, m.act_fmt)
    # cached residual: one per CFG branch → per sequence (seqs already counts cond / uncond) on this DP rank
    buf = _cdiv(seqs, lay.dp) * tok * m.hidden * ab
    st0 = stages[0]
    mp = st0.mem
    need = mp.dram_need + buf
    stages[0] = replace(st0, mem=replace(mp, dram_need=need, fits=need <= mp.dram_cap))
    b_mb = ph.batch
    tf_skip = c["tflop"] * lay.tp * lay.sp * lay.dp * lay.cp / b_mb
    return {"skip": k, "mode": scn.workload.step_cache, "computed": wl.steps - k, "steps": wl.steps,
            "t_skip": c["s"], "tflop_seq_skip": tf_skip, "buffer_GiB_card": buf / 2**30, "fits": need <= mp.dram_cap,
            "label": "跳步数取决于阈值与内容系数，由用户给出；固定步规则「来源」TeaCache4Wan2.1"}


def _prefix_kv(scn: Scenario, m: ModelSpec, wl, warnings: list[str]) -> bool:
    """0.116: is the text-prefix K/V reuse active?  auto = the model's declaration (reference default); on needs a
    block-causal condition mask (the prefix must not see the image tokens or the timestep); off disables it."""
    mode = scn.workload.prefix_kv
    nw = m.workload
    if wl.kind != "gen" or nw is None or mode == "off":
        return False
    ok = nw.cond_causal and nw.prefix_tokens > 0
    if mode == "on" and not ok:
        warnings.append("workload.prefix_kv = on 已忽略：该模型的文本前缀会看到图像 token / 随时间步调制（无 causal_condition），"
                        "其 K/V 每步不同，不能跨步复用")
        return False
    if not (nw.prefix_kv or mode == "on") or not ok:
        return False
    if wl.steps < 2:
        return False
    return True


def _prefix_cache_bytes(m: ModelSpec, T: int, lay, ab: float) -> float:
    """Per-card prefix K/V cache of one sequence: T tokens × the card's heads (TP, then SP head split) × (dk + dv) for
    every self-attention layer of the busiest stage (layers / PP, ceil)."""
    per = 0.0
    for L in m.layers:
        c = L.core
        if c.kind != "none":
            per += T * _cdiv(_cdiv(c.n_kv, lay.tp), lay.sp) * (c.qk_dim + c.v_dim) * ab
    return per * _cdiv(m.n_layers, lay.pp) / m.n_layers


def _component_time(groups: list[tuple[list[Op], int]], sys: System, org: str, af: str) -> dict:
    """One pipeline component on one card: same op → mapping → streaming path as a stage; its weights are larger
    than SRAM, so every weight is read from DRAM once per op execution (tiles / chunks re-read).
    ``groups``: [(ops, executions)] (a tiled VAE decode runs one op list per tile)."""
    d = dict.fromkeys(_SUM_KEYS, 0.0)
    shim = SimpleNamespace(act_fmt=af, kv_fmt=af)
    for ops, n in groups:
        _acc(d, _sum_ops(ops, sys, org, shim), n)
    dram = d["hot"] + d["exp"] + d["w_extra"] + d["act"]
    stt = StageTime(d["t_arr"], d["t_mac"], d["t_feed"], d["t_vec"], dram / (sys.dram_GBps * 1e9), 0.0, 0.0, dram,
                    0.0, d["flops"], d["t_ideal"])
    ch = sys.chip
    return {"s": stt.total, "tflop": d["flops"] / 1e12, "dram_GB": dram / 1e9, "bound": stt.bound,
            # action counts (0.47.1, core/energy.py): bf16-equivalent MAC slots, vector element-ops, SRAM-port bytes
            "acts": {"mac": d["t_ideal"] * ch.macs * ch.freq_ghz * 1e9, "vec": d["t_vec"] * ch.lanes * ch.freq_ghz * 1e9,
                     "sram": d["sram"], "dram": dram, "link": 0.0, "d2d": 0.0, "net": 0.0}}


def _pipeline_cost(scn: Scenario, m: ModelSpec, wl: ResolvedWorkload, sys: System, warnings: list[str]) -> dict | None:
    """Text encoder(s) + VAE decode of one request batch (0.44; ``core/pipeline.py``).  Runs on one card of each data-
    parallel replica (its share of the batch), serially with the denoise loop 「假设」."""
    pl = pipeline_for(m.id)
    if pl is None:
        return None
    if not scn.workload.pipeline:
        warnings.append("workload.pipeline = false：只评估 DiT 去噪主干——文本编码器与 VAE 解码不计时间与存储")
        return None
    b = _cdiv(scn.serving.batch, scn.layout.dp)
    info = wl.info
    tiling = scn.workload.vae_tiling
    if tiling and pl.vae.family not in (*TILING, "hunyuan", "h3"):
        warnings.append(f"workload.vae_tiling：{pl.vae.label} 的参考实现没有空间分块解码（按参考默认不分块计）")
    lay = scn.layout
    par = lay.pp * lay.tp * lay.sp if scn.workload.vae_parallel else 1
    if scn.workload.vae_parallel and par == 1:
        warnings.append("workload.vae_parallel：每副本只有 1 张卡，VAE 解码不拆分")
    key = (b, tuple(info["latent"]), info["frames"], info["cfg"], wl.aux, scn.mapping, scn.chip, scn.mem_id,
           scn.mem_eff, tiling, par, _links(scn), lay.dp if par > 1 else 1)
    memo = model_cache(m).setdefault("pipeline", {})
    if key in memo:
        return memo[key]
    prompts = b * (info["cfg"] if pl.neg_prompt else 1)
    parts = []
    te_s = te_w = te_act = te_lw = 0.0
    te_nl = 0
    for te in pl.text:
        nl, lw = te_layers(te)
        te_nl += nl; te_lw = max(te_lw, lw)
        ops, peak = text_ops(te, prompts, m.act_fmt)
        c = _component_time([(ops, 1)], sys, scn.mapping, m.act_fmt)
        w = stored_bytes(te.key)
        te_s += c["s"]; te_w += w; te_act = max(te_act, peak)
        parts.append({"role": "text_encoder", "label": te.label, "key": te.key, "prompts": prompts,
                      "tokens": te.tokens, "stored_GB": w / 1e9, "note": te.note, **c})
    dec_s = vae_w = vae_act = 0.0
    vae_par = 1
    lat = tuple(info["latent"])
    for v, rows in ((pl.vae, 0), (pl.audio, wl.aux)):
        if v is None:
            continue
        ops, peak, vi = vae_ops(v, lat, info["frames"], b, m.act_fmt, audio_rows=rows, tiling=tiling)
        c = _component_time(vi.pop("groups", None) or [(ops, 1)], sys, scn.mapping, vi["act"])
        extra = {}
        if par > 1 and v is pl.vae:
            if "rounds" in vi:
                cp = {**_parallel_decode(vi, par, b, lat, info, sys, scn.mapping, _replica_members(sys, lay)),
                      "tflop": c["tflop"]}
                gb, gd, gn = cp.pop("gather_B"), cp.pop("gather_d2d"), cp.pop("gather_net")
                cp["acts"] = {**c["acts"], "link": gb - gd - gn, "d2d": gd, "net": gn}   # replica totals
                extra = {"par": par, "single_s": c["s"], "gather_s": cp.pop("gather_s"), "rank_tiles": cp.pop("tiles")}
                c = cp
                vae_par = par
            else:
                warnings.append(f"workload.vae_parallel：{pl.vae.label} 未分块解码，不拆分（需 vae_tiling，"
                                "HunyuanVideo / MiniMax-H3 总是分块）")
        w = stored_bytes(v.key)
        dec_s += c["s"]; vae_w += w; vae_act = max(vae_act, peak)
        parts.append({"role": "audio_vae" if v is pl.audio else "vae", "label": v.label, "key": v.key,
                      "stored_GB": w / 1e9, "note": v.note, "act": vi["act"],
                      **{k: vi[k] for k in ("tiles", "overlap", "chunks", "tiling") if k in vi}, **extra, **c})
    out = {"te_s": te_s, "decode_s": dec_s, "tflop_replica": sum(p["tflop"] for p in parts),
           "te_w": te_w, "vae_w": vae_w, "te_act": te_act, "vae_act": vae_act, "parts": parts,
           "te_layers": te_nl, "te_layer_w": te_lw, "vae_par": vae_par,
           "batch_per_replica": b, "note": pl.note}
    if len(memo) > 4096:
        memo.clear()
    memo[key] = out
    return out


def _replica_members(sys: System, lay: Layout) -> tuple[int | None, int | None] | None:
    """Cards of one data-parallel replica (PP·TP·SP) inside one package / one node (cards numbered TP, SP, DP, PP:
    with DP > 1 and PP > 1 the replica is not contiguous — only its TP·SP block is)."""
    if sys.package_cards <= 1 and sys.node_cards <= 0:
        return None
    par = lay.pp * lay.tp * lay.sp
    blk = par if (lay.dp == 1 or lay.pp == 1) else lay.tp * lay.sp

    def inside(cap: int) -> int:
        return math.gcd(min(cap, blk), par)
    return (inside(sys.package_cards) if sys.package_cards > 1 else 1,
            inside(sys.node_cards) if sys.node_cards > 0 else None)


def _parallel_decode(vi: dict, par: int, b: int, lat: tuple, info: dict, sys: System, org: str,
                     members: int | None = None) -> dict:
    """Tile-parallel VAE decode over the ``par`` = PP·TP·SP cards of a replica (0.47, ``workload.vae_parallel``).

    The scheme of the MiniMax-H3 release (``FL2VA/video_vae`` ``vae_parallel_tiling = 1``, ``klvae.tiled_decode``):
    within each independent tiled call (a decode "round": an H3 temporal clip, a HunyuanVideo temporal tile, the
    single spatial pass of a diffusers ``enable_tiling()`` decode) rank r decodes tiles r, r + par, … then the decoded
    pixel tiles are all-gathered so every rank can blend.  Tiles are independent → arithmetic unchanged; time = the
    slowest rank's tiles + one all-gather per round (payload = its largest share of decoded pixels in the decode
    dtype, α each).  VAE weights are replicated on every card of the replica (``_place_components``).  For decoders
    other than H3 this is a design option (xDiT's DistVAE patch-parallel decode is image-VAE-only; its CogVideoX
    example asserts parallel VAE off)."""
    ops = vi["tile_ops"]
    ab = _fmt(vi["act"]).bytes
    px = info["frames"] * info["height"] * info["width"] * 3 / math.prod(lat)   # output elements per latent voxel
    loads: dict = {}
    gather = sent = sent_d = sent_n = 0.0
    for rd in vi["rounds"]:
        share = [rd[r::par] for r in range(par)]
        for r, tl in enumerate(share):
            for t in tl:
                loads.setdefault(r, {}).setdefault(t, 0)
                loads[r][t] += 1
        top = max(sum(math.prod(t) for t in tl) for tl in share) * px * ab * b
        bw, a_, fd, fn = _comm(sys, "allgather", top, par, members=members)
        gather += bw + a_
        sent += par * (par - 1) * top            # every rank's share to every other rank (upper bound)
        sent_d += par * (par - 1) * top * fd
        sent_n += par * (par - 1) * top * fn
    memo: dict = {}
    best = None
    for r, cnt in loads.items():
        k = tuple(sorted(cnt.items()))
        if k not in memo:
            memo[k] = _component_time([(ops[t], n * b) for t, n in k], sys, org, vi["act"])
        if best is None or memo[k]["s"] > best[0]["s"]:
            best = (memo[k], sum(cnt.values()))
    c = dict(best[0])                # time / DRAM of the slowest rank (the caller restores the replica FLOPs)
    c["s"] += gather
    c.update(gather_s=gather, tiles=best[1], gather_B=sent, gather_d2d=sent_d, gather_net=sent_n)
    return c


_PLACE_LABEL = {"resident": "常驻", "shard": "文本编码器分片（FSDP）", "offload": "顺序卸载（CPU offload）",
                "shard+offload": "分片 + 卸载"}


def _standby_bytes(m: ModelSpec, lay: Layout, first: int, last: int) -> float:
    """Idle-expert weights of a stage per card (Wan2.2 A14B: the second 14B expert, resident but not run)."""
    return m.standby_params * (last - first) / m.n_layers * m.fmt("attn").bits / 8 / lay.tp


def _place_components(scn: Scenario, m: ModelSpec, pipe: dict, stages: list, sys: System,
                      warnings: list[str]) -> dict:
    """Where the pipeline components live on the cards of one data-parallel replica (0.45; docs/MODEL.md §11.4).

      resident       text encoder on the first stage's card, VAE (+ audio VAE) on the last stage's card, both
                     resident next to the DiT (0.44 behaviour)
      shard          text-encoder weights sharded over all c = PP·TP·SP cards of the replica (Wan ``--t5_fsdp``):
                     te_w / c per card + 2 gathered layers (prefetch); each card all-gathers the layers while it
                     encodes — time max(compute, te_w·(c−1)/c / link) + α per layer
      offload        components and DiT time-share the card (diffusers ``enable_model_cpu_offload`` / Wan
                     ``--offload_model``): capacity = max(DiT, text encoder, VAE) instead of the sum; every request
                     re-loads TE, DiT stage and VAE weights host → card at ``workload.host_GBps`` (H2D only, the host
                     keeps its copy 「假设」); a multi-expert DiT (Wan2.2 A14B) also parks its idle expert on the host
                     (Wan2.2 ``--offload_model`` swaps the experts at the noise boundary) → standby not resident,
                     both experts still loaded once per request
      shard+offload  both
      auto           the first of resident → shard → offload → shard+offload that fits (else the smallest need)
    ``workload.te_cpu`` (0.46, Wan ``--t5_cpu``): the text encoder runs on the host CPU — no encoder weights or
    activations on any card, encode time = encoder FLOPs / ``workload.host_TFLOPS`` (「假设」, set from a measurement),
    the embeddings' copy to the card is negligible; then shard is moot and auto tries resident → offload.
    ``workload.dit_fsdp`` (0.46): the DiT stage weights are already sharded over SP·DP in the stage plan (see
    ``_evaluate_full``); an offloaded card then re-loads only its shard."""
    lay, w = scn.layout, scn.workload
    c = lay.pp * lay.tp * lay.sp
    last = len(stages) - 1
    base = [s.mem for s in stages]
    alpha = sys.link.alpha_us * 1e-6
    g = lay.sp * lay.dp if w.dit_fsdp and lay.sp * lay.dp > 1 else 1
    standby = [_standby_bytes(m, lay, *s.layers) / g for s in stages]

    cpu = w.te_cpu
    te_tflop = sum(p["tflop"] for p in pipe["parts"] if p["role"] == "text_encoder")
    te_act = 0.0 if cpu else pipe["te_act"]

    def option(place: str) -> tuple[list[MemPlan], dict]:
        shard = "shard" in place and c > 1 and not cpu
        off = "offload" in place
        te_card = (pipe["te_w"] / c + 2 * pipe["te_layer_w"]) if shard else pipe["te_w"]
        te_card = 0.0 if cpu else min(te_card, pipe["te_w"])
        mems = []
        for i, mp in enumerate(base):
            te_here = shard or i == 0
            vae_here = i == last or pipe.get("vae_par", 1) > 1
            w_te = te_card if te_here else 0.0
            w_vae = pipe["vae_w"] if vae_here else 0.0
            if off:
                need = max(mp.dram_need - standby[i],
                           RUNTIME_RESERVE + w_te + te_act if te_here and not cpu else 0.0,
                           RUNTIME_RESERVE + w_vae + pipe["vae_act"] if vae_here else 0.0)
            else:
                act = max(te_act if te_here else 0.0, pipe["vae_act"] if vae_here else 0.0)
                need = mp.dram_need + w_te + w_vae + max(0.0, act - mp.act_total)
            mems.append(replace(mp, dram_need=need, fits=need <= mp.dram_cap, pipe_w=w_te + w_vae))
        if not shard:
            gather = 0.0
        elif sys.package_cards <= 1 and sys.node_cards <= 0 and not sys.fabric.enabled:
            gather = pipe["te_w"] * (c - 1) / c / (sys.link.GBps * 1e9) + alpha * pipe["te_layers"]
        else:       # 0.48 / 0.50: the same all-gather volume on the tiered fabric, α per encoder layer
            bw_, a_, _, _ = _comm(sys, "allgather", pipe["te_w"] / c, c, members=_replica_members(sys, lay))
            gather = bw_ + a_ * pipe["te_layers"]
        te_s = (te_tflop / w.host_TFLOPS if cpu else max(pipe["te_s"], gather) if shard else pipe["te_s"])
        host = w.host_GBps * 1e9
        load = ((te_card + max(mp.stored_w for mp in base) + pipe["vae_w"]) / host) if off else 0.0
        # 0.62: host-link bytes of one reload over the whole replica (every card re-reads its own DiT stage, encoder
        # share and VAE copy; load_s above is the slowest card's) -- the energy report charges them at pJ_bit_host
        cps = lay.cards // len(base)
        load_b = sum(cps * (mp.stored_w + (te_card if (shard or i == 0) else 0.0)
                            + (pipe["vae_w"] if (i == last or pipe.get("vae_par", 1) > 1) else 0.0))
                     for i, mp in enumerate(base)) if off else 0.0
        return mems, {"place": "shard" if place == "shard+offload" and not shard else
                      ("resident" if place == "shard" and not shard else place),
                      "te_card_w": te_card, "gather_s": gather, "te_s": te_s, "te_compute_s": pipe["te_s"],
                      "load_s": load, "load_bytes": load_b, "te_cards": c if shard else 1, "te_cpu": cpu,
                      "host_TFLOPS": w.host_TFLOPS if cpu else None}

    if w.placement == "auto":
        multi = c > 1 and not cpu
        cands = ["resident"] + (["shard"] if multi else []) + ["offload"] + (["shard+offload"] if multi else [])
        opts = [(p, *option(p)) for p in cands]
        pick = next((o for o in opts if all(mp.fits for mp in o[1])), None)
        if pick is None:
            pick = min(opts, key=lambda o: max(mp.dram_need for mp in o[1]))
    else:
        pick = (w.placement, *option(w.placement))
    req, mems, info = pick
    for s_, mp in zip(stages, mems):
        s_.mem = mp
    if w.placement == "auto" and info["place"] != "resident":
        warnings.append(f"组件放置 auto → {_PLACE_LABEL[info['place']]}：常驻放不下"
                        + (f"；每请求从主机重载权重 {info['load_s']:.2f} s（{w.host_GBps:g} GB/s「假设」）" if info["load_s"] else ""))
    parts = pipe["parts"]
    if cpu:     # encoder parts report their host time (the card-side numbers stay in te_compute_s)
        parts = [{**p, "s": p["tflop"] / w.host_TFLOPS, "bound": "host CPU", "dram_GB": 0.0, "on_host": True}
                 if p["role"] == "text_encoder" else p for p in parts]
    return {**pipe, **info, "parts": parts, "placement": w.placement,
            "place_label": _PLACE_LABEL[info["place"]] + ("，文本编码器在主机 CPU" if cpu else ""),
            "host_GBps": w.host_GBps}


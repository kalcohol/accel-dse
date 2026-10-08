"""Evaluate one Scenario: per-stage op graphs → mapping → memory plan → schedule.

Everything goes through the per-rank IR; a single card is Layout().
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .catalog import get_model
from .dtypes import fmt as _fmt
from .hardware import System
from .domain import ResolvedWorkload, resolve_workload
from .ir import Op, Phase, Shard, _cdiv, build_rank_ops, embed_ops, full_io_ops, head_ops, mtp_ops
from .mapping import gemm_cost, vector_seconds
from .memo import layer_groups, model_cache
from .memplan import MemPlan, act_stream, plan, stage_storage, step_dram_bytes, touched
from .model import ModelSpec, with_formats
from .parallel import Layout, plan_stages
from .scenario import Scenario
from .schedule import StageTime, collective_seconds, spec_expected_tokens


@dataclass
class StageResult:
    index: int
    layers: tuple[int, int]
    time: StageTime
    mem: MemPlan
    dram: dict
    convert_elems: float
    ops: list[Op] = field(default_factory=list, repr=False)


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

    @property
    def domain(self) -> str:
        return self.model.domain

    @property
    def slo_ok(self) -> bool:
        """Latency SLO of the scenario: LLM decode TPOT (prefill: none), video clip latency, protein batch latency."""
        sv, w = self.scenario.serving, self.scenario.workload
        if self.model.domain == "gen":
            return self.latency <= w.clip_slo_s
        if self.model.domain == "protein":
            return self.latency * 1e3 <= w.seq_slo_ms
        return self.tpot * 1e3 <= sv.tpot_slo_ms

    def domain_summary(self) -> dict:
        """Domain KPIs of a non-autoregressive model (empty for LLMs)."""
        w = self.workload
        if w is None:
            return {}
        cards = self.scenario.layout.cards
        b = self.scenario.serving.batch
        heavy = self.stages[self.heaviest_stage]
        d = {"unit": w.unit, "latency_s": self.latency, "requests_per_s": b / self.latency,
             "units_per_s": self.throughput, "units_per_s_card": self.per_card, "workload": w.info,
             "forward_ms": self.tick * max(self.microbatches, self.scenario.layout.pp) * 1e3,
             "seqs_per_forward": b * w.seqs_per_request, "act_GiB": heavy.mem.act_total / 2**30,
             "tflop_per_request": sum(st.time.flops for st in self.stages) * self.microbatches * w.steps / b / 1e12}
        if w.kind == "gen":
            d.update(clip_s=self.latency, s_per_frame=self.latency / w.units, step_ms=self.latency / w.steps * 1e3,
                     frames_per_s_card=self.per_card, clips_per_hour_card=3600 * b / self.latency / cards,
                     realtime_x=(w.info["video_s"] * b / self.latency) if w.info.get("video_s") else None,
                     slo_s=self.scenario.workload.clip_slo_s)
        else:
            d.update(batch_ms=self.latency * 1e3, seq_per_s_card=self.per_card,
                     residues_per_s_card=self.per_card * w.info["seq_len"], slo_ms=self.scenario.workload.seq_slo_ms)
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
        }


def _op_seconds(op: Op, sys: System, org: str, model: ModelSpec) -> tuple[float, float, float, float, float]:
    """(array s, mac-bound share s, feed-bound share s, vector s, convert elems)."""
    ch = sys.chip
    f = ch.freq_ghz * 1e9 * ch.mac_eff
    if op.kind == "gemm":
        c = gemm_cost(ch, org, op.m, op.k, op.n, count=op.count, w_fmt=op.w_fmt, a_fmt=op.a_fmt, w_bits=op.w_bits)
    elif op.kind == "attn" and op.orient:
        # full-sequence (DiT / encoder) attention: both operands are activations in the activation dtype, and a
        # flash kernel may compute O = P·V or Oᵀ = Vᵀ·Pᵀ — take the cheaper orientation (PV has N = head_dim,
        # which leaves most of a wide array idle).  LLM prefill / decode attention keeps the 0.40 orientation.
        af = model.act_fmt
        c = min((gemm_cost(ch, org, a, op.k, b, count=op.count, w_fmt=af, a_fmt=af)
                 for a, b in ((op.m, op.n), (op.n, op.m))), key=lambda x: x.cycles)
    elif op.kind == "attn":
        c = gemm_cost(ch, org, op.m, op.k, op.n, count=op.count, w_fmt=model.kv_fmt, a_fmt="bf16")
    else:
        return 0.0, 0.0, 0.0, vector_seconds(ch, op.vec), 0.0, 0.0
    t = c.cycles * op.causal / f
    vec = vector_seconds(ch, op.vec + c.convert_elems * 2.0)   # 2 element-ops per converted element 「假设」
    mac_share = t if c.bound == "mac" else 0.0
    rate = ch.formats.rate(c.exec_fmt) or 1.0
    ideal = op.flops / 2.0 / (ch.macs * rate * ch.freq_ghz * 1e9)   # 100 % array utilisation
    return t, mac_share, t - mac_share, vec, c.convert_elems, ideal


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


def _tail_ops(model, has_head, ph, sh, spec_k):
    ops: list[Op] = []
    if has_head:
        ops += head_ops(model, ph, sh)
        if spec_k and model.mtp_layers and ph.kind == "decode":
            dph = Phase("decode", ph.batch, 1, ph.ctx)
            for d in range(spec_k):
                ops += mtp_ops(model, dph, sh, depth=d)
    return ops


_SUM_KEYS = ("t_arr", "t_mac", "t_feed", "t_vec", "conv", "t_ideal", "flops", "link_bw", "sync", "link_bytes",
             "hot", "exp", "kv_read", "kv_write", "state", "lookup", "max_act", "act", "w_extra", "max_act_tot")
_MAX_KEYS = ("max_act", "max_act_tot")


def _sum_ops(ops: list[Op], sys: System, org: str, model: ModelSpec) -> dict:
    d = dict.fromkeys(_SUM_KEYS, 0.0)
    for o in ops:
        if o.kind == "comm":
            bw, a = collective_seconds(o.comm_kind, o.comm_bytes, o.comm_group, sys.link)
            d["link_bw"] += bw; d["sync"] += a; d["link_bytes"] += o.comm_bytes
            continue
        a_, ma, fe, v, ce, idl = _op_seconds(o, sys, org, model)
        d["t_arr"] += a_; d["t_mac"] += ma; d["t_feed"] += fe; d["t_vec"] += v; d["conv"] += ce
        d["t_ideal"] += idl
        d["flops"] += o.flops
        d["max_act"] = max(d["max_act"], o.act_bytes / max(o.count, 1))
        if o.stream:
            a, wx = act_stream(o, _fmt(model.act_fmt).bytes, sys.chip.sram_bytes)
            d["act"] += a; d["w_extra"] += wx
            ab = _fmt(model.act_fmt).bytes
            tot = (o.m * o.k + o.m * o.n) * o.count * ab if o.kind == "gemm" else 2.0 * o.count * (o.m + o.n) * o.k * ab
            d["max_act_tot"] = max(d["max_act_tot"], tot)
    for k, v in touched(ops).items():
        d[k] += v
    return d


def _acc(a: dict, b: dict, n: int = 1) -> None:
    for k in _SUM_KEYS:
        if k in _MAX_KEYS:
            a[k] = max(a[k], b[k])
        else:
            a[k] += b[k] * n


def evaluate(scn: Scenario, model: ModelSpec | None = None) -> Result:
    m = model or get_model(scn.model)
    if scn.formats_override:
        wi = model_cache(m).setdefault("what_if", {})
        key = tuple(scn.formats_override)
        if key not in wi:
            wi[key] = with_formats(m, dict(scn.formats_override))
        m = wi[key]
    sv, lay = scn.serving, scn.layout
    warnings: list[str] = []
    if not lay.valid_for(m.is_moe, full=not m.kv_cache):
        raise ValueError(f"layout {lay.label} invalid for {'MoE' if m.is_moe else 'dense'} model "
                         + ("(video / protein models: ep = etp = 1)" if not m.kv_cache else
                            "(MoE needs ep·etp = tp·dp; dense needs dp=ep=etp=1; sp only for video / protein models)"))
    if lay.pp > m.n_layers:
        raise ValueError("pp exceeds layer count")
    sys = System(scn.chip, scn.mem_id, scn.mem_eff, scn.link)
    if not m.kv_cache:
        return _evaluate_full(scn, m, sys, warnings)
    spec_k = sv.spec_k if (sv.phase == "decode" and m.mtp_layers) else 0
    if sv.spec_k and not m.mtp_layers and sv.phase == "decode":
        warnings.append("spec_k 已忽略：该模型没有 MTP 模块（独立草稿模型的投机解码未建模）")
    pp = lay.pp
    if sv.phase == "decode":
        mb = sv.microbatches or min(pp, sv.batch)
        mb = max(1, min(mb, sv.batch))
        q = 1 + spec_k
        ph = Phase("decode", _cdiv(sv.batch, mb), q, sv.ctx)
        ctx_cap = sv.ctx + q
    else:
        mb = sv.microbatches or min(pp, sv.batch)
        mb = max(1, min(mb, sv.batch))
        ph = Phase("prefill", _cdiv(sv.batch, mb), sv.prompt, 0)
        ctx_cap = sv.prompt
    sh = lay.shard
    stages = []
    n_mtp = min(spec_k, len(m.mtp_layers)) if spec_k else 0
    b_rank = _cdiv(sv.batch, lay.dp)          # sequences whose KV lives on one rank (all micro-batches)
    cache: dict = {}
    groups = layer_groups(m)
    store_memo = model_cache(m).setdefault("stage_storage", {})
    ops_memo = model_cache(m).setdefault("layer_sums", {})
    if len(ops_memo) > 200_000:
        ops_memo.clear()
    for st in plan_stages(m.n_layers, pp):
        agg = dict.fromkeys(_SUM_KEYS, 0.0)
        if st.has_embed:
            _acc(agg, _sum_ops(embed_ops(m, ph, sh), sys, scn.mapping, m))
        counts: dict = {}
        for li in range(st.first, st.last):
            g = groups[li]
            first_li, n = counts.get(g, (li, 0))
            counts[g] = (first_li, n + 1)
        for g, (li, n) in counts.items():
            if g not in cache:
                okey = (g, ph, sh, scn.mapping, scn.chip, scn.link)
                cache[g] = ops_memo.get(okey)
                if cache[g] is None:
                    cache[g] = ops_memo[okey] = _sum_ops(build_rank_ops(m, li, ph, sh), sys, scn.mapping, m)
            _acc(agg, cache[g], n)
        _acc(agg, _sum_ops(_tail_ops(m, st.has_head, ph, sh, spec_k), sys, scn.mapping, m))
        skey = (st.first, st.last, st.has_embed, st.has_head, sh, ctx_cap, n_mtp)
        store = store_memo.get(skey)
        if store is None:
            store = store_memo[skey] = stage_storage(m, st.first, st.last, st.has_embed, st.has_head, sh, ctx_cap,
                                                     n_mtp)
        mp = plan(store, b_rank, sys.chip.sram_bytes, sys.dram_bytes, agg["max_act"])
        dram = step_dram_bytes(mp, store, agg)
        link_bw, sync, link_bytes = agg["link_bw"], agg["sync"], agg["link_bytes"]
        if pp > 1 and not st.has_head:
            act = ph.batch * ph.q * m.hidden * _fmt(m.act_fmt).bytes / lay.dp
            bw, a = collective_seconds("p2p", act, 2, sys.link)
            link_bw += bw; sync += a; link_bytes += act
        t_dram = dram["total"] / (sys.dram_GBps * 1e9)
        stt = StageTime(agg["t_arr"], agg["t_mac"], agg["t_feed"], agg["t_vec"], t_dram, link_bw, sync,
                        dram["total"], link_bytes, agg["flops"], agg["t_ideal"])
        stages.append(StageResult(st.index, (st.first, st.last), stt, mp, dram, agg["conv"]))
    heavy = max(range(len(stages)), key=lambda i: stages[i].time.total)
    tick = stages[heavy].time.total
    cap_heavy = max(range(len(stages)), key=lambda i: stages[i].mem.dram_need)
    fits = all(s.mem.fits for s in stages)
    if not fits:
        warnings.append(f"容量不足：流水级 {cap_heavy} 每卡需要 {stages[cap_heavy].mem.dram_need / 2**30:.1f} GiB，"
                        f"超过每卡 {sys.dram_bytes / 2**30:.0f} GiB")
    if m.coverage == "proxy":
        warnings.append("「架构代理」模型：结果为近似值（见建模说明）")
    if sv.phase == "decode":
        step = max(mb, pp) * tick
        e_tok = spec_expected_tokens(spec_k, sv.spec_accept) if spec_k else 1.0
        tpot = step / e_tok
        thr = sv.batch * e_tok / step
        ttft = 0.0
    else:
        step = (mb + pp - 1) * tick
        e_tok = 1.0
        tpot = 0.0
        ttft = step
        thr = sv.batch * sv.prompt / step
    return Result(scn, m, stages, tick, step, e_tok, tpot, ttft, thr, thr / lay.cards, fits,
                  stages[heavy].time.bound, heavy, warnings, mb)


def _evaluate_full(scn: Scenario, m: ModelSpec, sys: System, warnings: list[str]) -> Result:
    """Non-autoregressive request: video clip (steps × CFG forwards over every latent token) or protein batch
    (one encoder forward).  Same op-graph → mapping → memplan → schedule path as the LLM phases.

      forward sequences  S = batch · cfg (video) or batch (protein); microbatches mb = min(pp, S) unless set
      video clip latency = steps · max(mb, pp) · t_stage        (each denoise step like a decode step: the
                           pipeline is kept full across steps by independent micro-batches)
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
    ph = Phase("full", _cdiv(seqs, mb), wl.tokens, wl.ctx, frames=wl.frames, aux=wl.aux)
    sh = lay.shard
    ab = _fmt(m.act_fmt).bytes
    groups = layer_groups(m)
    ops_memo = model_cache(m).setdefault("layer_sums", {})
    store_memo = model_cache(m).setdefault("stage_storage", {})
    if len(ops_memo) > 200_000:
        ops_memo.clear()
    stages = []
    resid = _cdiv(seqs, lay.dp) * _cdiv(wl.tokens, lay.sp) * m.hidden * ab   # residual streams of in-flight sequences
    for st in plan_stages(m.n_layers, pp):
        agg = dict.fromkeys(_SUM_KEYS, 0.0)
        if st.has_embed:
            _acc(agg, _sum_ops(full_io_ops(m, ph, sh, "pre"), sys, scn.mapping, m))
        counts: dict = {}
        for li in range(st.first, st.last):
            g = groups[li]
            first_li, n = counts.get(g, (li, 0))
            counts[g] = (first_li, n + 1)
        for g, (li, n) in counts.items():
            okey = (g, ph, sh, scn.mapping, scn.chip, scn.link)
            hit = ops_memo.get(okey)
            if hit is None:
                hit = ops_memo[okey] = _sum_ops(build_rank_ops(m, li, ph, sh), sys, scn.mapping, m)
            _acc(agg, hit, n)
        if st.has_head:
            _acc(agg, _sum_ops(full_io_ops(m, ph, sh, "post"), sys, scn.mapping, m))
        skey = (st.first, st.last, st.has_embed, st.has_head, sh, 0, 0)
        store = store_memo.get(skey)
        if store is None:
            store = store_memo[skey] = stage_storage(m, st.first, st.last, st.has_embed, st.has_head, sh, 0, 0)
        mp = plan(store, 0, sys.chip.sram_bytes, sys.dram_bytes, agg["max_act"],
                  act_need=agg["max_act_tot"] + resid, stream=True)
        dram = step_dram_bytes(mp, store, agg)
        link_bw, sync, link_bytes = agg["link_bw"], agg["sync"], agg["link_bytes"]
        if pp > 1 and not st.has_head:
            act = ph.batch * _cdiv(ph.q, lay.sp) * m.hidden * ab / lay.dp
            bw, a = collective_seconds("p2p", act, 2, sys.link)
            link_bw += bw; sync += a; link_bytes += act
        t_dram = dram["total"] / (sys.dram_GBps * 1e9)
        stt = StageTime(agg["t_arr"], agg["t_mac"], agg["t_feed"], agg["t_vec"], t_dram, link_bw, sync,
                        dram["total"], link_bytes, agg["flops"], agg["t_ideal"])
        stages.append(StageResult(st.index, (st.first, st.last), stt, mp, dram, agg["conv"]))
    heavy = max(range(len(stages)), key=lambda i: stages[i].time.total)
    tick = stages[heavy].time.total
    cap_heavy = max(range(len(stages)), key=lambda i: stages[i].mem.dram_need)
    fits = all(s.mem.fits for s in stages)
    if not fits:
        warnings.append(f"容量不足：流水级 {cap_heavy} 每卡需要 {stages[cap_heavy].mem.dram_need / 2**30:.1f} GiB，"
                        f"超过每卡 {sys.dram_bytes / 2**30:.0f} GiB")
    if wl.kind == "gen":
        latency = wl.steps * max(mb, pp) * tick
    else:
        latency = (mb + pp - 1) * tick
    thr = sv.batch * wl.units / latency
    return Result(scn, m, stages, tick, latency, wl.units, 0.0, 0.0, thr, thr / lay.cards, fits,
                  stages[heavy].time.bound, heavy, warnings, mb, latency, wl)

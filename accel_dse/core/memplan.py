"""L4 — memory planning per rank of a pipeline stage.

Storage (per rank, as released):
  stored weights  = all local params of the stage (every local expert), at the
                    release's stored bits incl. scales
  KV cache        = Σ layers ctx·kv_elems_per_token (per-rank share; MLA latent
                    replicated on every attention-TP rank)
  recurrent state = linear-attention state per sequence (fp32 「假设」)
  standby weights = idle expert of a multi-expert denoiser (Wan2.2 A14B), split over stages by layer share
DRAM capacity check uses the *heaviest* stage.

SRAM policy (「假设」, design variable via chip.sram_mib):
  1. staging     = 2 × (largest per-op activation footprint), ≥ 2 MiB
  2. weights     pinned hottest-first: always-touched weights (attention, dense,
                 shared experts, router, head) then routed experts; cold bytes (looked-up embedding table, standby
                 expert, unused io tables) are never pinned (0.61.3: they used to ride with the experts)
  3. leftover    holds KV / recurrent state (partial hit: the cached fraction)
Per-step DRAM traffic = unpinned touched weights + uncached KV/state reads +
KV writes to DRAM-resident cache.

Full-sequence forward (video DiT denoise step / protein encoder; ``Op.stream``): token counts of
10⁴–10⁵ make activations far larger than SRAM, so they are streamed (``act_stream``):
  budget       = SRAM / 2 for activation tiles, SRAM / 2 for weight tiles 「假设」
  GEMM         activations in+out ≤ budget → stay on chip (no traffic); otherwise the cheaper of
               (a) activation-chunked: A_in + A_out + W·ceil((A_in+A_out)/budget)
               (b) weight-chunked:     W + A_in·ceil(W/budget) + A_out
               minus the one weight read already counted as touched.
  attention    flash-style: Q tile (bf16) + O accumulator (fp32) of Br = budget / (d·(a+4)) rows;
               K, V re-read ceil(Nq / Br) times; Q, O once.  Nothing if Q, K, V, O all fit.
Activation working set in DRAM = largest op (in+out) + the residual stream of every in-flight sequence.

System-level cache (0.48, chip.slc_mib > 0; every number and rule 「假设」): a cache between the chip and DRAM that
serves hits at chip.slc_GBps instead of the DRAM bandwidth.  It adds no capacity (inclusive: everything still has a
DRAM home) and is per card.  Policies:
  pin   software-managed, like the SRAM plan: what SRAM did not pin goes to the SLC in the same order (hot weights,
        routed experts, KV / recurrent state).  Reads of a pinned byte hit; streamed activations, FSDP-gathered
        weights, embedding-row lookups and KV writes go to DRAM (write-through / bypass).
  lru   hardware-managed.  Accesses of a decode / denoise step repeat cyclically, so LRU either holds the whole
        working set or thrashes: if everything that is not in SRAM (weights + KV / state + the activation working
        set) fits, every read hits (KV writes still go to DRAM); otherwise nothing hits (the cyclic-thrash bound —
        real replacement sits between this and pin).
Video pipeline components (text encoder, VAE) do not use the SLC.
"""

from __future__ import annotations

import math
from functools import lru_cache
from dataclasses import dataclass, replace

from .dtypes import fmt as _fmt
from .ir import Op, Phase, Shard, _cdiv, _lin_local
from .model import Layer, ModelSpec

RUNTIME_RESERVE = 1.0 * 2**30   # 「假设」 runtime / activations reserve in DRAM


@dataclass(frozen=True)
class StageStorage:
    hot_w: float          # bytes of always-touched weights (per rank)
    expert_w: float       # bytes of routed-expert weights stored (per rank)
    kv_per_seq: float     # KV bytes per sequence at the planned context (per rank)
    idx_per_seq: float    # indexer-key bytes per sequence
    state_per_seq: float  # recurrent state bytes per sequence (per rank)
    cold_w: float = 0.0   # stored, never streamed (looked-up embedding rows, standby expert, unused io tables) —
    #                       0.61.3: kept apart so the SRAM / SLC plans do not pin bytes no step reads in bulk

    @property
    def weights(self) -> float:
        return self.hot_w + self.expert_w + self.cold_w


def _layer_storage(model: ModelSpec, L: Layer, sh: Shard) -> tuple[float, float]:
    hot = 0.0
    a = model.fmt("attn").bits / 8
    for l in L.attn_linears + L.cross_linears:
        k, n = _lin_local(l, sh.tp, L.core if l in L.attn_linears else None)
        hot += k * n * l.groups * a
    f = L.ffn
    for l in L.ffn_linears:
        k, n = _lin_local(l, sh.tp)
        role = "router" if l.name.startswith(("router", "shared_expert_gate")) else (
            "shared_expert" if l.name.startswith("shared") else "mlp")
        hot += k * n * model.fmt(role).bits / 8
    # norms / gates / conv / hc vectors (bf16 「假设」; non-LLM releases: as released)
    for l in L.pair_linears:        # structure-model blocks (single rank)
        hot += l.params * model.fmt(l.role or "pair").bits / 8
    mb = model.fmt(L.misc_role).bits / 8 if L.misc_role else a
    hot += L.misc_params * (2.0 if model.domain == "llm" else mb)
    exp = 0.0
    if f.kind == "moe":
        n_loc = _cdiv(f.n_experts, sh.ep)
        exp = n_loc * f.mats * f.expert_in(model.hidden) * _cdiv(f.d_expert, sh.etp) * model.fmt("expert").bits / 8
    return hot, exp


def cache_new_bytes(model: ModelSpec, S: int, p: int) -> float:
    """0.62: whole-model KV + indexer + recurrent-state bytes a receiver still lacks after an ``S``-token prefill
    when it already holds the cache of the first ``p`` tokens (PD hand-off with the prefix cached on the decode side).
    Token-proportional entries (full / compressed attention, indexer keys): those of tokens p … S; a sliding window:
    the entries outside the held prefix's window, min(W, S − p) (the held window ends at p); recurrent state: all of
    it (the state after S tokens is not the prefix's).  ≤ 0.61 scaled the whole request's bytes by (S − p) / S, which
    under-counted window and state bytes (hybrid linear-attention / SWA models)."""
    p = max(0, min(p, S))
    kvb = _fmt(model.kv_fmt).bytes
    stb = _fmt(model.state_fmt).bytes
    kv = idx = st = 0.0
    new = S - p
    for L in model.layers:
        c = L.core
        if c.kind == "gqa":
            n = min(c.window, new) if (c.window and c.compress == 1) else new
            kv += n * c.n_kv * (c.qk_dim + c.v_dim) / c.compress * kvb
        elif c.kind == "mla":
            if c.compress > 1:
                n = math.ceil(S / c.compress) - math.ceil(p / c.compress) + (min(c.window, new) if c.window else 0)
            else:
                n = min(c.window, new) if c.window else new
            kv += n * (c.kv_lora + c.rope_dim) * kvb
        elif c.kind == "linear":
            st += (c.n_state_heads * c.state_dk * c.state_dv + c.conv_channels * max(0, c.conv_kernel - 1)) * stb
        if c.idx_heads:
            idx += (math.ceil(S / c.compress) - math.ceil(p / c.compress)) * c.idx_dim * kvb
    return kv + idx + st


def stage_storage(model: ModelSpec, first: int, last: int, has_embed: bool, has_head: bool, sh: Shard,
                  ctx: int, n_mtp: int = 0) -> StageStorage:
    hot = exp = kv = idx = st = 0.0
    kvb = _fmt(model.kv_fmt).bytes
    stb = _fmt(model.state_fmt).bytes
    layers = list(model.layers[first:last])
    if has_head:
        layers += list(model.mtp_layers[:n_mtp])
    for L in layers:
        h_, e_ = _layer_storage(model, L, sh)
        hot += h_; exp += e_
        c = L.core
        ce = min(ctx, c.window) if (c.window and c.compress == 1) else ctx
        if not model.kv_cache:
            continue
        if c.kind == "gqa":
            kv += ce * _cdiv(c.n_kv, sh.tp) * (c.qk_dim + c.v_dim) / c.compress * kvb
        elif c.kind == "mla":
            # 0.61.1: a sliding-window-only latent layer (compress 1) keeps min(ctx, window) entries (was ctx)
            kv += ((math.ceil(ctx / c.compress) + (c.window or 0)) if c.compress > 1 else ce) * (c.kv_lora + c.rope_dim) * kvb
        elif c.kind == "linear":
            st += (_cdiv(c.n_state_heads, sh.tp) * c.state_dk * c.state_dv + _cdiv(c.conv_channels, sh.tp) * max(0, c.conv_kernel - 1)) * stb
        if c.idx_heads:
            idx += math.ceil(ctx / c.compress) * c.idx_dim * kvb
    store_extra = 0.0
    if model.standby_params:        # idle expert of a multi-expert denoiser (Wan2.2 A14B): stored, not read this step
        store_extra += model.standby_params * (last - first) / model.n_layers / sh.tp * model.fmt("attn").bits / 8
    if model.domain != "llm":       # io GEMM weights are read every forward; io biases / unused tables only stored
        io_b = model.fmt("io").bits / 8
        lb = lambda l: _lin_local(l, sh.tp)[0] * _lin_local(l, sh.tp)[1] * model.fmt(l.role or "io").bits / 8
        if has_embed:
            hot += sum(lb(l) for l in model.io_pre)
            store_extra += model.io_misc_params * io_b
        if has_head:
            hot += sum(lb(l) for l in model.io_post)
            hot += (model.final_norm_params or 0) * io_b
    emb_b = model.fmt("embed").bits / 8
    table = _cdiv(model.vocab, sh.tp) * model.hidden * emb_b
    if has_embed:
        # the embedding table is looked up (a few rows / token) → cold storage,
        # unless it is tied to the LM head on this same stage (then streamed every step)
        if model.tie_embeddings and has_head:
            hot += table
        else:
            store_extra += table
        store_extra += model.lookup_params * emb_b / sh.tp
    if has_head:
        if not model.tie_embeddings:
            hot += _cdiv(model.vocab, sh.tp) * model.hidden * model.fmt("lm_head").bits / 8
        elif not has_embed:
            hot += table          # tied head on a different stage keeps its own copy
        if n_mtp:
            hot += n_mtp * model.mtp_extra_params * model.fmt("mtp").bits / 8 / sh.tp
    return StageStorage(hot, exp, kv, idx, st, store_extra)


@dataclass(frozen=True)
class MemPlan:
    stored_w: float
    kv_total: float
    state_total: float
    dram_need: float
    dram_cap: float
    fits: bool
    sram: float
    staging: float
    pinned_hot: float
    pinned_expert: float
    kv_sram: float          # bytes of KV+state kept on chip
    residency: float        # pinned / streamed (hot + expert) weights — cold storage excluded (0.61.3)
    act_total: float = 0.0  # activation working set kept in DRAM (full-sequence forward)
    pipe_w: float = 0.0     # video pipeline components stored on this card (text encoder / VAE weights, 0.44)
    slc: float = 0.0        # SLC capacity per card (0.48; 0 = none)
    slc_policy: str = ""    # pin | lru ("" without an SLC)
    slc_hot: float = 0.0    # pin: hot-weight bytes held in the SLC
    slc_expert: float = 0.0
    slc_kv: float = 0.0
    slc_all: bool = False   # lru: the whole off-SRAM working set fits → every read hits
    slc_residency: float = 0.0  # weights on chip incl. SLC / streamed weights (= residency without an SLC)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def plan(store: StageStorage, batch_local: int, sram_bytes: float, dram_cap: float, max_act: float,
         act_need: float = 0.0, stream: bool = False, slc_bytes: float = 0.0, slc_policy: str = "pin") -> MemPlan:
    kv_t = (store.kv_per_seq + store.idx_per_seq) * batch_local
    st_t = store.state_per_seq * batch_local
    need = store.weights + kv_t + st_t + RUNTIME_RESERVE + act_need
    staging = max(2 * 2**20, 2 * max_act)
    if stream:      # streamed activations use at most the activation half of SRAM (act_stream)
        staging = min(staging, max(2 * 2**20, sram_bytes / 2))
    free = max(0.0, sram_bytes - staging)
    p_hot = min(store.hot_w, free); free -= p_hot
    p_exp = min(store.expert_w, free); free -= p_exp
    kv_s = min(kv_t + st_t, free)
    strm = store.hot_w + store.expert_w            # 0.61.3: residency of the weights a step reads (cold bytes excluded)
    res = (p_hot + p_exp) / strm if strm else 1.0
    mp = MemPlan(store.weights, kv_t, st_t, need, dram_cap, need <= dram_cap, sram_bytes, staging,
                 p_hot, p_exp, kv_s, res, act_need, slc_residency=res)
    if slc_bytes <= 0:
        return mp
    r_hot, r_exp, r_kv = store.hot_w - p_hot, store.expert_w - p_exp, kv_t + st_t - kv_s
    if slc_policy == "lru":
        fits = r_hot + r_exp + r_kv + act_need <= slc_bytes
        return replace(mp, slc=slc_bytes, slc_policy="lru", slc_all=fits, slc_residency=1.0 if fits else res)
    free = slc_bytes
    s_hot = min(r_hot, free); free -= s_hot
    s_exp = min(r_exp, free); free -= s_exp
    s_kv = min(r_kv, free)
    sres = (p_hot + p_exp + s_hot + s_exp) / strm if strm else 1.0
    return replace(mp, slc=slc_bytes, slc_policy="pin", slc_hot=s_hot, slc_expert=s_exp, slc_kv=s_kv,
                   slc_residency=sres)


ACC_BYTES = 4.0     # 0.64: output-tile partial sums held on chip in fp32 「假设」


@lru_cache(maxsize=1 << 16)
def gemm_blocking(m: int, k: int, n: int, a_bytes: float, w_bytes: float, budget: float) -> tuple[int, int]:
    """0.64 — 2-D blocking of C[m,n] = A[m,k]·W[k,n] under an on-chip budget (bytes): (times A is read, times W is
    read) of the cheapest of three loop nests (DRAM bytes = A·r_A + W·r_W + C, C written once).  The budget holds
    the *resident* block; the streamed operand (a row / column / K slice) is double-buffered outside it, as in 0.63
    「假设」:
      weight-stationary      W in ⌈W/budget⌉ resident chunks, A streamed past each → r_A = ⌈W/budget⌉, r_W = 1;
      activation-stationary  A in ⌈A/budget⌉ resident chunks, W streamed past each → r_A = 1, r_W = ⌈A/budget⌉
                             (chunk counts from bytes, not whole columns / rows: a chunk edge cutting a column /
                             row along k keeps that partial sum on chip — so an exact fit, W = budget, is one pass);
      output-stationary      bm×bn fp32 partial sums resident (bm·bn·4 ≤ budget), A / W streamed in K slices
                             → r_A = ⌈n/bn⌉, r_W = ⌈m/bm⌉, bm over powers of two and m (the classic I/O-lower-bound
                             tiling, traffic ~ 2·m·n·k·e/√(budget/4)).
    a, w = bytes per element of A / W (A from ``a_bytes`` so implicit-GEMM inputs count their real size).  0.63 chose
    between the two one-sided extremes with whole-budget chunks (activation chunks with full weight re-reads,
    ⌈(A + C)/budget⌉, or weight chunks with full activation re-reads, ⌈W/budget⌉); both are (weakly) dominated by the
    first two nests, so 0.64 traffic ≤ 0.63's (exactly, not only off the exact-fit boundaries).  Every feasible set grows with the budget → never increases with SRAM.
    The output-stationary tile count is whole tiles (bm, bn integers).  budget ≤ 0 → r = (n, m)."""
    ea = a_bytes / max(1, m * k)          # A bytes per element
    ew = w_bytes / max(1, k * n)
    cost = lambda ra, rw: a_bytes * ra + w_bytes * rw
    best = None

    def take(ra: int, rw: int):
        nonlocal best
        if best is None or cost(ra, rw) < cost(*best) - 1e-9:
            best = (ra, rw)
    if budget > 0:      # one-side-resident nests: chunk counts from bytes (a chunk edge may cut a column / row along
        take(max(1, math.ceil(w_bytes / budget - 1e-12)), 1)          # k — its partial sums stay on chip)
        take(1, max(1, math.ceil(a_bytes / budget - 1e-12)))
    for bm in sorted({min(m, 1 << i) for i in range(0, max(1, m).bit_length() + 1)} | {m}):
        bn = int(budget // (bm * ACC_BYTES))
        if bn < 1:
            break
        take(math.ceil(n / min(bn, n)), math.ceil(m / bm))
    return best if best is not None else (n, m)


def act_stream(op: Op, ab: float, sram_bytes: float) -> tuple[float, float]:
    """(activation DRAM bytes, extra weight re-read bytes) of one streamed op of a full-sequence forward."""
    budget = sram_bytes / 2.0
    if op.kind == "gemm":
        a_in = (op.in_elems or op.m * op.k * op.count) * ab
        a_out = op.m * op.n * op.count * ab
        if a_in + a_out <= budget:
            return 0.0, 0.0
        # 0.63: each of the ``count`` instances (experts, heads, groups) has its own weight and activation slice,
        # so the blocking is decided per instance.  0.64: 2-D blocking (gemm_blocking) instead of chunking one side.
        c = max(1, op.count)
        ai, ao, w = a_in / c, a_out / c, op.w_bytes / c
        if ai + ao <= budget:
            return 0.0, 0.0
        r_a, r_w = gemm_blocking(op.m, op.k, op.n, ai, w, budget)
        return c * (ai * r_a + ao), c * w * (r_w - 1)
    if op.bmm:                  # operands and result are whole activations; each channel slice fits the budget,
        tot = (op.m * op.k + op.k * op.n + op.m * op.n) * op.count * ab     # so A, B are read and C written once
        return (tot, 0.0) if tot > budget else (0.0, 0.0)
    if op.kind == "attn":       # qk op of a flash-style attention (v_dim == qk_dim for these models)
        qo = 2.0 * op.count * op.m * op.k * ab
        kv = 2.0 * op.count * op.n * op.k * ab
        if qo + kv <= budget:
            return 0.0, 0.0
        br = max(1, int(budget // (op.k * (ab + 4))))
        return qo + kv * (math.ceil(op.m / br) - op.kv_pre), 0.0
    return 0.0, 0.0


def touched(ops: list[Op]) -> dict:
    """Per-step byte sums of an op list (independent of the plan)."""
    d = {"hot": 0.0, "exp": 0.0, "kv_read": 0.0, "kv_write": 0.0, "state": 0.0, "lookup": 0.0}
    for o in ops:
        if o.kind == "gemm":
            d["exp" if o.role == "expert" else "hot"] += o.w_bytes
        if o.kind == "lookup":
            d["lookup"] += o.kv_read
        else:
            d["kv_read"] += o.kv_read
        d["kv_write"] += o.kv_write
        d["state"] += o.state_rw
    return d


def step_dram_bytes(mp: MemPlan, store: StageStorage, t: dict) -> dict:
    """DRAM bytes moved in one stage step given the plan and touched-byte sums.  With an SLC the entries are the
    DRAM (miss) bytes and ``slc`` holds the bytes served by the SLC (split in ``slc_parts``)."""
    miss_hot = 0.0 if store.hot_w <= 0 else (1.0 - mp.pinned_hot / store.hot_w)
    miss_exp = 0.0 if store.expert_w <= 0 else (1.0 - mp.pinned_expert / store.expert_w)
    w = t["hot"] * miss_hot + t["exp"] * miss_exp
    kvst_total = mp.kv_total + mp.state_total
    kv_miss = 0.0 if kvst_total <= 0 else 1.0 - mp.kv_sram / kvst_total
    kv_r, kv_w, st = t["kv_read"] * kv_miss, t["kv_write"] * kv_miss, t["state"] * kv_miss
    act, wx = t.get("act", 0.0), t.get("w_extra", 0.0)
    out = {"weights": w + wx, "kv_read": kv_r, "kv_write": kv_w, "state": st, "lookup": t["lookup"], "act": act,
           "total": w + wx + kv_r + kv_w + st + t["lookup"] + act, "w_touched": t["hot"] + t["exp"]}
    # 0.64: routed-expert weight reads alone (DRAM after any SLC hits / SLC hits) — what a DP rank without sequences
    # still moves under EP (energy action counts)
    w_e = t["exp"] * miss_exp
    out["exp_dram"], out["exp_slc"] = w_e, 0.0
    if mp.slc <= 0:
        return out
    if mp.slc_policy == "lru":
        hit = {"weights": w + wx, "kv_read": kv_r, "state": st, "act": act} if mp.slc_all else {}
    else:
        f_hot = 0.0 if store.hot_w <= 0 else mp.slc_hot / store.hot_w
        f_exp = 0.0 if store.expert_w <= 0 else mp.slc_expert / store.expert_w
        r_kv = kvst_total - mp.kv_sram
        f_kv = 0.0 if r_kv <= 0 else min(1.0, mp.slc_kv / r_kv)
        w_hit = t["hot"] * f_hot + t["exp"] * f_exp
        wx_hit = wx * (w_hit / w) if w > 0 else 0.0     # streamed weight re-reads follow the weights' SLC share
        hit = {"weights": w_hit + wx_hit, "kv_read": kv_r * f_kv, "state": st * f_kv}
    if mp.slc_policy == "lru":
        e_hit = w_e if mp.slc_all else 0.0
    else:
        e_hit = t["exp"] * f_exp
    out["exp_dram"], out["exp_slc"] = w_e - e_hit, e_hit
    for k, v in hit.items():
        out[k] -= v
    s = sum(hit.values())
    out["total"] -= s
    out["slc"] = s
    out["slc_parts"] = hit
    return out

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
  dip   (0.258) hardware-managed with thrash-resistant insertion — DIP set dueling between LRU and BIP (Qureshi,
        Jaleel, Patt, Steely, Emer, ISCA 2007, §4.1 Table 3, 「来源」): on a cyclic sweep of T lines through a K-line
        cache (K < T) BIP keeps K − 1 lines resident and inserts ε·(T − K) lines per sweep at MRU → hit rate
        (K − 1 − ε·(T − K)) / T, ε = 1/32 (the paper's setting); when the working set fits, set dueling picks LRU
        and every read hits, as lru.  The retained lines are an arbitrary subset of the sweep → each read class
        (weights, KV / state, activations) hits by the same share 「推导」; the one-line term is dropped (K ≫ a line),
        the cache is taken as fully associative (set-associative with addresses spread over the sets 「近似」).  KV
        writes go to DRAM as for lru.
        0.260 per-class reuse: routed-expert weights are not all read every step.  The every-step classes (hot
        weights, KV / state, activation working set: C bytes) are re-referenced each step, an off-SRAM expert byte
        with probability e / E (e = expert bytes read per step, E = expert bytes off SRAM).  BIP promotes a line on
        a hit, so the retained set fills by reference frequency — the A0 rule, optimal under the independent
        reference model (Aho, Denning, Ullman, J. ACM 1971) 「近似」: every-step bytes first, then experts.  With
        K' = K − ε·max(0, C + e − K) (BIP churn over the step's distinct reads): every-step share min(1, K'/C), expert
        share clip((K' − C)/E, 0, 1).  E = 0 reduces to the 0.258 formula; everything fitting (C + E ≤ K) hits.
Video pipeline components (text encoder, VAE) do not use the SLC.
"""

from __future__ import annotations

import math
from functools import lru_cache
from dataclasses import dataclass, replace

from .dtypes import fmt as _fmt
from .ir import Op, Phase, Shard, _cdiv, _lin_local, groups_local
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
    # 0.269: hot weights read ``rr_reads`` times per step (TTS flow DiT: once per solver step), part of hot_w.  The
    # 0.192 pin rule (DRAM bytes saved per step per byte) puts them first; their reads miss at their own share
    rr_w: float = 0.0
    rr_reads: float = 1.0

    @property
    def weights(self) -> float:
        return self.hot_w + self.expert_w + self.cold_w


def _layer_storage(model: ModelSpec, L: Layer, sh: Shard) -> tuple[float, float]:
    if L.loop:          # 0.253 looped LM: later loops reuse the stored recurrent weights
        return 0.0, 0.0
    hot = 0.0
    a = model.fmt("attn").bits / 8
    for l in L.attn_linears + L.cross_linears:
        k, n = _lin_local(l, sh.tp, L.core if l in L.attn_linears else None)
        hot += k * n * (groups_local(l, sh.tp)[0] if l.groups > 1 else 1) * a
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
        if L.kv_shared:
            continue
        if c.kind == "gqa":
            n = min(c.window, new) if (c.window and c.compress == 1) else new
            kv += n * c.n_kv * (c.qk_dim + c.v_dim) / c.compress * kvb
        elif c.kind == "mla" and c.win_bytes:
            kv += min(c.window, new) * c.win_bytes + ((S // c.compress - p // c.compress) * c.cmp_bytes
                                                      if c.kv_owner else 0.0)
        elif c.kind == "mla":
            if c.compress > 1:
                n = math.ceil(S / c.compress) - math.ceil(p / c.compress) + (min(c.window, new) if c.window else 0)
            else:
                n = min(c.window, new) if c.window else new
            kv += n * (c.kv_lora + c.rope_dim) * kvb
        if L.ple_rows:
            st += (L.ple_conv - 1) * L.ple_dil * L.gres_n * model.hidden * 2
        if L.cross is not None and model.enc_ctx:   # 0.270: the whole cross K/V (made at the first prefill pass)
            st += model.enc_ctx * L.cross.n_kv * (L.cross.qk_dim + L.cross.v_dim) * kvb
        if c.kind == "linear":
            st += c.state_bytes(1, stb)
        if c.idx_heads and c.dual:
            idx += (S // c.compress - p // c.compress) * c.idx_kbytes if c.idx_owner else 0.0
        elif c.idx_heads and c.idx_sum and c.qsa:     # 0.145: Step-5 summaries of completed regions only
            idx += (S // c.qsa - p // c.qsa) * c.idx_dim * 5.0
        elif c.idx_heads:
            idx += (math.ceil(S / c.compress) - math.ceil(p / c.compress)) * c.idx_bytes_per_token(kvb) * c.compress
    return kv + idx + st


def stage_storage(model: ModelSpec, first: int, last: int, has_embed: bool, has_head: bool, sh: Shard,
                  ctx: int, n_mtp: int = 0, spec_q: int = 0, spec_ring: int = 0) -> StageStorage:
    """``spec_q`` (0.222): tokens of a speculative verify pass (1 + k, decode) — hybrid / linear layers keep one
    intermediate recurrent state per draft token (ir.SPEC_STATE_SNAPSHOT; SGLang intermediate_ssm_state_cache)."""
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
        if not model.kv_cache or L.kv_shared:
            continue
        if c.kind == "gqa":
            kv += ce * _cdiv(c.n_kv, sh.tp) * (c.qk_dim + c.v_dim) / c.compress * kvb
        elif c.kind == "mla" and c.win_bytes:     # V4.1: per-layer window ring + compressed cache of kv sources
            kv += min(ctx, c.window) * c.win_bytes + ((ctx // c.compress) * c.cmp_bytes if c.kv_owner else 0.0)
        elif c.kind == "mla":
            # 0.61.1: a sliding-window-only latent layer (compress 1) keeps min(ctx, window) entries (was ctx)
            kv += ((math.ceil(ctx / c.compress) + (c.window or 0)) if c.compress > 1 else ce) * (c.kv_lora + c.rope_dim) * kvb
        if L.cross is not None and model.enc_ctx:
            # 0.270 encoder-decoder: cross K/V of the encoder positions, fixed per sequence (EncoderDecoderCache)
            cc = L.cross
            st += model.enc_ctx * _cdiv(cc.n_kv, sh.tp) * (cc.qk_dim + cc.v_dim) * kvb
        if L.ple_rows:      # Qwen4Exp PLE dilated conv state, bf16 (kernel − 1)·dilation columns of N·h
            st += (L.ple_conv - 1) * L.ple_dil * L.gres_n * model.hidden * 2
        if c.kind == "linear":
            st += c.state_bytes(sh.tp, stb)
            if spec_ring and c.lin in ("gdn", "kda"):
                # 0.223 / 0.224 ReplaySSM (ir.REPLAY_RING_DEFAULT): compact ring of spec_ring entries per head instead
                # of the q intermediate states (GDN d, k + low parts + g; KDA + raw window, g per K, beta)
                ent = ((c.state_dk + c.state_dv) * 4.0 + 4.0) if c.lin == "gdn" else (4.0 * c.state_dv + 8.0 * c.state_dk + 4.0)
                st += spec_ring * -(-c.n_state_heads // sh.tp) * ent
            elif spec_q > 1 and c.lin not in ("hyena", "rwkv7"):
                st += spec_q * c.state_bytes(sh.tp, stb)
        if c.idx_heads and c.dual:
            idx += (ctx // c.compress) * c.idx_kbytes if c.idx_owner else 0.0
        elif c.idx_heads and c.idx_sum and c.qsa:     # 0.145: Step-5 fp32 + fp8 summary per completed region
            idx += (ctx // c.qsa) * c.idx_dim * 5.0
        elif c.idx_heads and c.idx_kshard:     # 0.87 key_split: this rank holds ⌈n_keys / TP⌉ index keys
            idx += _cdiv(math.ceil(ctx / c.compress), sh.tp) * c.idx_bytes_per_token(kvb) * c.compress
        elif c.idx_heads:
            idx += math.ceil(ctx / c.compress) * c.idx_bytes_per_token(kvb) * c.compress
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
        store_extra += model.lookup_params * ((model.lookup_bits / 8) if model.lookup_bits else emb_b) / sh.tp
    if has_head:
        ts = getattr(model, "tts", None)
        if ts is not None:
            # 0.269 TTS head (replicated on the head stage's ranks, bf16): flow DiT / aggregator / stop / AudioVAE
            # decoder read every step; AudioVAE encoder + spk_head stored only; the LM head is stored, never read
            wb = getattr(ts, "w_bytes", 2.0)       # 0.275: FireRedTTS3 head / RedAE stored fp32
            hot += ts.run_params * wb
            store_extra += ts.stored_params * wb
            st += ts.dec_kv_bytes        # per-sequence AudioVAE decoder sliding-window K/V ring (fixed size)
            if not model.tie_embeddings:
                store_extra += _cdiv(model.vocab, sh.tp) * model.hidden * model.fmt("lm_head").bits / 8
        elif not model.tie_embeddings:
            hot += _cdiv(model.vocab, sh.tp) * model.hidden * model.fmt("lm_head").bits / 8
        elif not has_embed:
            hot += table          # tied head on a different stage keeps its own copy
        if n_mtp:
            hot += n_mtp * model.mtp_extra_params * model.fmt("mtp").bits / 8 / sh.tp
            # 0.252 DSpine: codebook (only anchor rows read) + the serial tables built at load — resident, not streamed
            store_extra += model.draft_cold_params * model.fmt("mtp").bits / 8 + model.draft_table_bytes
    rr = 0.0
    ts = getattr(model, "tts", None) if has_head else None
    if ts is not None and ts.flow_steps > 1:
        rr = ts.flow_params * getattr(ts, "w_bytes", 2.0)
    return StageStorage(hot, exp, kv, idx, st, store_extra, rr, float(ts.flow_steps) if rr else 1.0)


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
    slc_policy: str = ""    # pin | lru | pin_bw ("" without an SLC)
    slc_hot: float = 0.0    # pin: hot-weight bytes held in the SLC
    slc_expert: float = 0.0
    slc_kv: float = 0.0
    slc_all: bool = False   # lru: the whole off-SRAM working set fits → every read hits
    slc_frac: float = 0.0   # 0.258 dip: share of the step's off-SRAM reads served by the SLC (0.260: every-step classes)
    slc_frac_exp: float = 0.0  # 0.260 dip: share of the routed-expert reads served by the SLC
    slc_frac_exp_bytes: float = 0.0  # 0.261 dip: share of the off-SRAM expert bytes retained (= slc_frac_exp if uniform)
    slc_residency: float = 0.0  # weights on chip incl. SLC / streamed weights (= residency without an SLC)
    pinned_rr: float = 0.0  # 0.269: re-read hot bytes resident in SRAM (part of pinned_hot)
    # 0.286: share of the routed-expert *reads* served by the SRAM-pinned experts (-1 = the byte share
    # pinned_expert / expert_w; with a measured expert load the pinned experts are the most-read ones)
    sram_exp_hit: float = -1.0
    slc_rr: float = 0.0     # 0.269 pin: re-read hot bytes held in the SLC (part of slc_hot)
    # 0.209: of kv_sram, the recurrent-state bytes (the rest is KV / indexer); -1 = not split (legacy plans)
    state_sram: float = -1.0
    state_buf: float = 0.0  # dedicated state-buffer capacity (Chip.state_sram_mib)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


STATE_FIRST = True        # 0.192 default


STATE_SPLIT = True        # 0.209: separate KV / state on-chip shares (≤ 0.208: one pooled miss fraction for both)


def plan(store: StageStorage, batch_local: int, sram_bytes: float, dram_cap: float, max_act: float,
         act_need: float = 0.0, stream: bool = False, slc_bytes: float = 0.0, slc_policy: str = "pin",
         kv_frac: float = 1.0, state_buf: float = 0.0, exp_read: float | None = None,
         exp_p: tuple | None = None) -> MemPlan:
    kv_t = (store.kv_per_seq + store.idx_per_seq) * batch_local
    st_t = store.state_per_seq * batch_local
    need = store.weights + kv_t + st_t + RUNTIME_RESERVE + act_need
    staging = max(2 * 2**20, 2 * max_act)
    if stream:      # streamed activations use at most the activation half of SRAM (act_stream)
        staging = min(staging, max(2 * 2**20, sram_bytes / 2))
    free = max(0.0, sram_bytes - staging)
    # 0.192: pin by DRAM bytes saved per step per byte of SRAM.  Recurrent state is read and written every step
    # (2 B / B), a weight byte is read once per pass (1 B / B), a KV byte once (+ the new row) — so state goes
    # first (STATE_FIRST; ≤ 0.191 weights → KV + state)
    # 0.209: a dedicated state buffer (Chip.state_sram_mib) holds state first; the rest competes for the shared SRAM.
    # Residency is per byte (partial: the resident share of the batch's state skips DRAM, the rest streams)
    b_st = min(st_t, state_buf)
    rr_w = min(store.rr_w, store.hot_w)
    p_rr = 0.0
    if rr_w > 0 and store.rr_reads > 2.0:           # 0.269: R B/B saved beats state's 2 B/B
        p_rr = min(rr_w, free); free -= p_rr
    p_st = b_st + (min(st_t - b_st, free) if STATE_FIRST else 0.0)
    free -= p_st - b_st
    if rr_w > 0 and store.rr_reads <= 2.0:
        p_rr = min(rr_w, free); free -= p_rr
    p_hot = p_rr + min(store.hot_w - rr_w, free); free -= p_hot - p_rr
    p_exp = min(store.expert_w, free); free -= p_exp
    p_kv = min(kv_t, free)                       # KV / indexer in what is left
    free -= p_kv
    p_st += min(st_t - p_st, free)               # STATE_FIRST off: state after the weights and KV
    kv_s = p_st + p_kv
    strm = store.hot_w + store.expert_w            # 0.61.3: residency of the weights a step reads (cold bytes excluded)
    res = (p_hot + p_exp) / strm if strm else 1.0
    # 0.94: a runtime granting the cache only kv_frac of the post-weight free memory (default 1 = unchanged)
    fit = need <= dram_cap if kv_frac >= 1 else need - kv_t - st_t + (kv_t + st_t) / kv_frac <= dram_cap
    # 0.286: expert residency in popularity order across the on-chip tiers.  The 0.192 pin rule (DRAM bytes saved per
    # step per byte of SRAM) ranks an expert byte by its read probability, so with a measured load (exp_p) the SRAM
    # holds the most-read experts and the SLC the next ones 「推导」; H(f) = share of expert reads served when the
    # most-read byte fraction f is resident (dip_exp_hit, independent reference model 「近似」; uniform load: H(f) = f,
    # i.e. unchanged).  ≤ 0.285: SRAM by byte share, and the SLC hit share was taken relative to the off-SRAM experts
    # but applied to every expert read — over-crediting the SLC (negative DRAM bytes once the SRAM pinned experts).
    E_w = store.expert_w
    rho = 1.0 if (exp_read is None or E_w <= 0) else min(1.0, exp_read / E_w)
    aware = bool(exp_p) and E_w > 0 and rho > 0

    def H(f):
        f = min(1.0, max(0.0, f))
        return dip_exp_hit(exp_p, rho, f) if aware else f
    h1 = H(p_exp / E_w) if E_w > 0 else 0.0
    mp = MemPlan(store.weights, kv_t, st_t, need, dram_cap, fit, sram_bytes, staging,
                 p_hot, p_exp, kv_s, res, act_need, slc_residency=res, state_sram=p_st, state_buf=state_buf,
                 pinned_rr=p_rr, sram_exp_hit=h1 if aware else -1.0)
    if slc_bytes <= 0:
        return mp
    r_hot, r_exp, r_kv = store.hot_w - p_hot, store.expert_w - p_exp, kv_t + st_t - kv_s
    if slc_policy == "lru":
        fits = r_hot + r_exp + r_kv + act_need <= slc_bytes
        return replace(mp, slc=slc_bytes, slc_policy="lru", slc_all=fits, slc_residency=1.0 if fits else res)
    if slc_policy == "dip":
        # 0.260: expert bytes read per step off SRAM (exp_read = touched expert bytes of the stage step; None → all)
        e = r_exp if exp_read is None else min(r_exp, exp_read * (r_exp / store.expert_w if store.expert_w else 0.0))
        fc, fe = dip_split(slc_bytes, r_hot + r_kv + act_need, r_exp, e)
        sres = (p_hot + p_exp + fc * r_hot + fe * r_exp) / strm if strm else 1.0
        # 0.261: with a measured expert load the retained experts are the most-read ones (A0 order), so the share of
        # expert *reads* served exceeds the retained byte share fe (uniform load: equal).  0.286: the off-SRAM experts
        # are the less-read remainder after the SRAM pin (was: a load-neutral subset) — hit share of the off-SRAM reads
        f1 = p_exp / E_w if E_w > 0 else 0.0
        he = ((H(f1 + fe * (1.0 - f1)) - h1) / (1.0 - h1) if h1 < 1.0 else 0.0) if (aware and r_exp > 0 and e > 0) else fe
        return replace(mp, slc=slc_bytes, slc_policy="dip", slc_all=fc >= 1.0 and fe >= 1.0, slc_frac=fc,
                       slc_frac_exp=he, slc_frac_exp_bytes=fe, slc_residency=sres)
    free = slc_bytes
    s_rr = min(rr_w - p_rr, free); free -= s_rr      # 0.269: re-read hot bytes first
    s_hot = s_rr + min(r_hot - s_rr, free); free -= s_hot - s_rr
    s_exp = min(r_exp, free); free -= s_exp
    s_kv = min(r_kv, free)
    sres = (p_hot + p_exp + s_hot + s_exp) / strm if strm else 1.0
    # 0.263: load-aware pin — with a measured expert load, the pinned expert bytes are the most-read ones (A0),
    # so the share of expert *reads* served exceeds the byte share (uniform: equal). Same helper as dip.
    # 0.286: both shares are of *all* expert bytes / reads (step_dram_bytes applies them to every expert read): bytes
    # s_exp / E, reads H((p + s) / E) − H(p / E) — the SLC holds the experts ranked after the SRAM's.  ≤ 0.285 used
    # s_exp / (E − p) for both, which exceeds the DRAM-side expert reads once the SRAM pins experts.
    fe = (s_exp / E_w) if E_w > 0 else 1.0
    he = (H((p_exp + s_exp) / E_w) - h1) if E_w > 0 else fe
    return replace(mp, slc=slc_bytes, slc_policy="pin_bw" if slc_policy == "pin_bw" else "pin",
                   slc_hot=s_hot, slc_expert=s_exp, slc_kv=s_kv, slc_residency=sres, slc_rr=s_rr,
                   slc_frac_exp=he, slc_frac_exp_bytes=fe)


DIP_EPS = 1 / 32     # 0.258: BIP bimodal throttle ε (Qureshi et al. ISCA 2007, "ε = 1/32 for all experiments")


def dip_hit(K: float, T: float, eps: float = DIP_EPS) -> float:
    """0.258: steady-state hit share of a cyclic sweep of T bytes through a K-byte BIP / DIP cache (K < T):
    (K − ε·(T − K)) / T, clipped to [0, 1] (Qureshi et al. 2007, Table 3, line term dropped)."""
    if T <= 0:
        return 1.0
    if T <= K:
        return 1.0
    return max(0.0, min(1.0, (K - eps * (T - K)) / T))


def dip_split(K: float, C: float, E: float, e: float, eps: float = DIP_EPS) -> tuple[float, float]:
    """0.260: (every-step share, expert share) of a dip SLC of K bytes — C every-step bytes, E off-SRAM expert bytes of
    which e are read per step (see the module doc).  E = 0 → (dip_hit(K, C), 1)."""
    if C + E <= K:
        return 1.0, 1.0
    Kp = max(0.0, K - eps * max(0.0, C + e - K))
    fc = min(1.0, Kp / C) if C > 0 else 1.0
    fe = min(1.0, max(0.0, (Kp - C) / E)) if E > 0 else 1.0
    return fc, fe


def dip_exp_hit(p: tuple, rho: float, f: float) -> float:
    """0.261: share of routed-expert reads served when the SLC retains the most-read fraction f of the off-SRAM experts
    (A0 frequency order under the independent reference model, Aho–Denning–Ullman 1971 「近似」).  p = per-expert
    inclusion probability per token (k·load_e/Σload, ir.hot_rank_probs, measured load); per step an expert is read with
    q_e = 1 − (1 − p_e)^T, T solved so that mean q = rho (the step's read share e / E, so the read bytes stay those of
    the op graph).  Off-SRAM experts are taken as a load-neutral subset 「近似」.  Uniform p → exactly f."""
    if f <= 0.0:
        return 0.0
    if f >= 1.0:
        return 1.0
    ps = [min(1.0, max(0.0, x)) for x in p]
    if not ps or max(ps) <= 0.0:
        return f

    def mean_q(T):
        return sum(1.0 - (1.0 - x) ** T for x in ps) / len(ps)
    lo, hi = 0.0, 1.0
    top = mean_q(1e12)
    rho = min(rho, top)
    while mean_q(hi) < rho and hi < 1e12:
        hi *= 2.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if mean_q(mid) < rho:
            lo = mid
        else:
            hi = mid
    q = sorted((1.0 - (1.0 - x) ** hi for x in ps), reverse=True)
    tot = sum(q)
    if tot <= 0:
        return f
    x = f * len(q)
    n = int(x)
    got = sum(q[:n]) + (x - n) * (q[n] if n < len(q) else 0.0)
    return max(f, min(1.0, got / tot))


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
                             → r_A = ⌈n/bn⌉, r_W = ⌈m/bm⌉, every integer bm (0.65; 0.64: powers of two) (the classic I/O-lower-bound
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
    # 0.65: every integer row tile, not only powers of two — for each distinct row-pass count r_W = ⌈m/bm⌉ the
    # smallest bm reaching it (bm = ⌈m/r_W⌉) leaves the widest bn = ⌊budget/(4·bm)⌋, so walking the O(√m) distinct
    # values of ⌈m/r⌉ is the exact integer optimum of the output-stationary nest (0.64: bm ∈ powers of two ∪ {m})
    m1 = max(1, m)
    r = 1
    while r <= m1:
        bm = -(-m1 // r)
        bn = int(budget // (bm * ACC_BYTES))
        if bn >= 1:
            take(math.ceil(n / min(bn, n)), r)
        if bm == 1:
            break
        r = -(-m1 // (bm - 1))            # next r with a smaller ⌈m/r⌉
    return best if best is not None else (n, m)


def fft_spill(n: int, sram_bytes: float, engines: int) -> float:
    """0.171 fused hyena what-if: DRAM bytes per transform row of an FFT convolution of size n whose rfft spectrum
    (n/2 complex64) stays on chip when it fits one engine's activation share (SRAM / 2 / engines「假设」); otherwise a
    four-step (Bailey) out-of-core FFT of p = ⌈log n / log n_fit⌉ passes, each extra pass one spectrum read + write,
    forward and inverse「推导」."""
    fit = max(2.0, sram_bytes / 2 / max(1, engines) / 8)
    p = math.ceil(math.log(n) / math.log(fit) - 1e-12) if n > fit else 1
    return (p - 1) * 2 * 2 * 4.0 * n


def act_stream(op: Op, ab: float, sram_bytes: float, engines: int = 1) -> tuple[float, float]:
    """(activation DRAM bytes, extra weight re-read bytes) of one streamed op of a full-sequence forward."""
    budget = sram_bytes / 2.0
    if op.kind == "state" and op.fft_n:
        return op.act_x + op.fft_rows * fft_spill(op.fft_n, sram_bytes, engines), 0.0
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
        # 0.119: an unfused triangle attention decides per chunk (its logits stay on chip when one chunk fits)
        if tot / max(1, op.chunks) > budget:
            return tot + op.act_x, 0.0
        # 0.120: one chunk fits, so its logits stay on chip — but q, k, v and the output still stream through DRAM
        # once when the module's tensors exceed the budget (as the fused kernel's single pass does); 0.119 charged 0,
        # which made small chunks look faster than the fused kernel
        io = tot - op.lg_bytes
        return (io, 0.0) if tot > budget else (0.0, 0.0)
    if op.kind == "state" and op.act_x:     # 0.170 hyena FFT kernels: each torch.fft / elementwise call round-trips
        return op.act_x, 0.0                # its fp32 / complex64 operands through DRAM (ir._hyena_ops)
    if op.kind == "attn":       # qk op of a flash-style attention (v_dim == qk_dim for these models)
        qo = 2.0 * op.count * op.m * op.k * ab
        kv = 2.0 * op.count * op.n * op.k * ab
        if qo + kv <= budget:
            return 0.0, 0.0
        br = max(1, int(budget // (op.k * (ab + 4))))
        return qo + kv * (math.ceil(op.m / br) - op.kv_pre), 0.0
    return 0.0, 0.0


RR_PREFIX = ("tts.flow",)       # 0.269: ops whose weights are the re-read hot class (StageStorage.rr_w)


def is_rr(o) -> bool:
    return o.name.startswith(RR_PREFIX)


def hot_miss(mp: "MemPlan", store: StageStorage) -> tuple[float, float]:
    """0.269: (miss share of ordinary hot weight reads, miss share of re-read hot reads) after SRAM pinning.  Without a
    re-read class: (1 − pinned_hot / hot_w, same) — the ≤ 0.268 uniform rule."""
    if store.hot_w <= 0:
        return 0.0, 0.0
    rr = min(store.rr_w, store.hot_w)
    if rr <= 0:
        m = 1.0 - mp.pinned_hot / store.hot_w
        return m, m
    rest = store.hot_w - rr
    m_rest = 0.0 if rest <= 0 else 1.0 - (mp.pinned_hot - mp.pinned_rr) / rest
    return m_rest, 1.0 - mp.pinned_rr / rr


def touched(ops: list[Op]) -> dict:
    """Per-step byte sums of an op list (independent of the plan)."""
    d = {"hot": 0.0, "exp": 0.0, "kv_read": 0.0, "kv_write": 0.0, "state": 0.0, "lookup": 0.0, "hot_rr": 0.0}
    for o in ops:
        if o.kind == "gemm":
            d["exp" if o.role == "expert" else "hot"] += o.w_bytes
            if is_rr(o):
                d["hot_rr"] += o.w_bytes
        if o.kind == "lookup":
            d["lookup"] += o.kv_read
        else:
            d["kv_read"] += o.kv_read
        d["kv_write"] += o.kv_write
        d["state"] += o.state_rw
    return d


def exp_miss(mp: MemPlan, store: StageStorage) -> float:
    """0.286: share of the routed-expert reads that miss the SRAM — 1 − the pinned experts' read share (a measured load
    makes it smaller than the byte share; uniform load / no load: 1 − pinned_expert / expert_w, as ≤ 0.285)."""
    if store.expert_w <= 0:
        return 0.0
    return 1.0 - (mp.sram_exp_hit if mp.sram_exp_hit >= 0 else mp.pinned_expert / store.expert_w)


def step_dram_bytes(mp: MemPlan, store: StageStorage, t: dict, slc_bw_ratio: float = 0.0) -> dict:
    """DRAM bytes moved in one stage step given the plan and touched-byte sums.  With an SLC the entries are the
    DRAM (miss) bytes and ``slc`` holds the bytes served by the SLC (split in ``slc_parts``)."""
    miss_hot, miss_rr = hot_miss(mp, store)
    miss_exp = exp_miss(mp, store)
    t_rr = t.get("hot_rr", 0.0) if store.rr_w > 0 else 0.0
    w = (t["hot"] - t_rr) * miss_hot + t_rr * miss_rr + t["exp"] * miss_exp
    kvst_total = mp.kv_total + mp.state_total
    kv_miss = 0.0 if kvst_total <= 0 else 1.0 - mp.kv_sram / kvst_total
    st_miss = kv_miss
    if STATE_SPLIT and mp.state_sram >= 0:       # 0.209: each stream misses by its own resident share
        st_miss = 0.0 if mp.state_total <= 0 else 1.0 - mp.state_sram / mp.state_total
        kv_miss = 0.0 if mp.kv_total <= 0 else 1.0 - (mp.kv_sram - mp.state_sram) / mp.kv_total
    kv_r, kv_w, st = t["kv_read"] * kv_miss, t["kv_write"] * kv_miss, t["state"] * st_miss
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
    elif mp.slc_policy == "dip":
        f, fe = mp.slc_frac, mp.slc_frac_exp
        w_d = w - t["exp"] * miss_exp                     # non-expert weight reads (+ streamed re-reads at that share)
        hit = {"weights": (w_d + wx) * f + t["exp"] * miss_exp * fe, "kv_read": kv_r * f, "state": st * f,
               "act": act * f} if f > 0 or fe > 0 else {}
    else:
        f_hot = 0.0 if store.hot_w <= 0 else mp.slc_hot / store.hot_w
        f_rr = f_hot
        if t_rr > 0:        # 0.269: the re-read class hits at its own SLC share
            rr = min(store.rr_w, store.hot_w)
            f_rr = mp.slc_rr / rr if rr > 0 else 0.0
            f_hot = (mp.slc_hot - mp.slc_rr) / (store.hot_w - rr) if store.hot_w > rr else 0.0
        f_exp_bytes = 0.0 if store.expert_w <= 0 else mp.slc_expert / store.expert_w
        # 0.263: load-aware pin — read-hit share may exceed the byte share (slc_frac_exp)
        f_exp = mp.slc_frac_exp if (mp.slc_frac_exp > 0 or mp.slc_frac_exp_bytes > 0) else f_exp_bytes
        if mp.slc_frac_exp_bytes > 0:
            f_exp = mp.slc_frac_exp
        else:
            f_exp = f_exp_bytes
        r_kv = kvst_total - mp.kv_sram
        f_kv = 0.0 if r_kv <= 0 else min(1.0, mp.slc_kv / r_kv)
        w_hit = (t["hot"] - t_rr) * f_hot + t_rr * f_rr + t["exp"] * f_exp
        wx_hit = wx * (w_hit / w) if w > 0 else 0.0     # streamed weight re-reads follow the weights' SLC share
        hit = {"weights": w_hit + wx_hit, "kv_read": kv_r * f_kv, "state": st * f_kv}
    if mp.slc_policy == "lru":
        e_hit = w_e if mp.slc_all else 0.0
    elif mp.slc_policy == "dip":
        e_hit = w_e * mp.slc_frac_exp
    else:
        e_hit = t["exp"] * f_exp
    if mp.slc_policy == "pin_bw" and slc_bw_ratio > 0:
        # 0.91: inclusive SLC (resident lines also live in DRAM) as a second read path.  Serve x ≤ H of the resident
        # bytes from the SLC so that both paths finish together: x/B_slc = (T − x)/B_dram → x = T·r/(1 + r), r =
        # B_slc/B_dram, T = every DRAM byte of the step (writes / lookups are DRAM-only but share its time).  Like CUDA
        # accessPolicyWindow.hitRatio chosen optimally; memory time = T/(B_dram + B_slc) when H suffices, never worse
        # than DRAM alone or than pin
        H = sum(hit.values())
        if H > 0:
            g = min(1.0, out["total"] * slc_bw_ratio / (1.0 + slc_bw_ratio) / H)
            hit = {k: v * g for k, v in hit.items()}
            e_hit *= g
    out["exp_dram"], out["exp_slc"] = w_e - e_hit, e_hit
    for k, v in hit.items():
        out[k] -= v
    s = sum(hit.values())
    out["total"] -= s
    out["slc"] = s
    out["slc_parts"] = hit
    return out

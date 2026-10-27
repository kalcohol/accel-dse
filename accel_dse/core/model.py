"""L1a — model specification parsed from official Hugging Face configs.

A ``ModelSpec`` is the *verified* architectural description of one released
checkpoint: every linear (GEMM weight) with its TP split rule, the attention
core semantics (GQA / MLA / sparse / compressed / linear-recurrent), the FFN
(dense / MoE incl. shared + latent experts) and the **as-released storage
format per tensor role** taken from the safetensors headers
(``accel_dse/data/releases/*.json``).

Coverage axis (one of the three model labels):
  * ``full``     every parameter-bearing op and the attention/KV semantics are
                 modelled; params within ±2 % of the release.
  * ``partial``  all GEMMs modelled; some attention mechanism (sparse indexer,
                 sliding window variants, attention sinks …) is approximated —
                 listed in ``notes``.
  * ``proxy``    「架构代理」: the release contains mechanisms we only
                 approximate (hyper-connections, n-gram/engram tables,
                 compressed-KV attention …). Shown with a badge; never mixed
                 silently with full-coverage results.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
RELEASES = DATA / "releases"


# ---------------------------------------------------------------- dataclasses
@dataclass(frozen=True)
class Linear:
    """y[M,N] = x[M,K] @ W[K,N].  ``split`` = TP rule:
    col  – N split across attention/FFN TP ranks
    row  – K split (partial sums → all-reduce / reduce-scatter)
    rep  – replicated on every TP rank (small low-rank projections, routers)
    head – N split in units of ``unit`` (KV heads: ceil(n_kv/tp) per rank)
    ``rows`` = which rows feed the GEMM in a non-autoregressive (``full``) forward:
    tok  – every token of the sequence (default; the only kind LLM layers use)
    img  – the video tokens only (sequence minus the text prefix of joint-attention DiTs)
    ctx  – the conditioning tokens (text-encoder output feeding cross-attention K/V)
    seq  – one row per sequence (timestep embedding / AdaLN modulation)
    txt  – the joint text rows of the sequence (dual-stream MMDiT text weights, text refiners)
    aud  – the joint audio rows of the sequence (MiniMax-H3 audio io)
    structure models (0.43; Phase.pair gives the grid sizes):
    res  – one row per residue / token (single representation)
    pair – N² rows (pair representation)        msa / xmsa – S·N rows (MSA / extra-MSA stack)
    tmpl – T·N² rows (template pair stack)      atom – A rows (atoms);  apair – A·window rows (local atom pairs)
    """
    name: str
    k: int
    n: int
    split: str = "col"
    unit: int = 0          # for split == "head": columns per head
    groups: int = 1        # block-diagonal (grouped) linear: params = k*n (k,n are totals)
    rows: str = "tok"
    role: str = ""         # weight-format role override (non-LLM io GEMMs stored in another dtype); "" = default

    @property
    def params(self) -> int:
        return self.k * self.n // self.groups


@dataclass(frozen=True)
class AttnCore:
    """Per-token attention semantics of one layer.

    kind: gqa | mla | linear | none
      gqa    – n_q heads of qk_dim/v_dim, n_kv KV heads (GQA packing g = n_q/n_kv)
      mla    – latent cache (kv_lora + rope); decode uses absorbed form
      linear – recurrent state per sequence: n_state_heads × dk × dv (fp32 「假设」)
    window      – sliding-window length (None = full context)
    topk        – sparse attention: each query attends to ≤ topk selected tokens
    compress    – KV compression ratio (DeepSeek-V4 CSA): cache holds ctx/compress
    idx_heads/idx_dim – lightning indexer (DSA) scoring cost per (query, key)
    """
    kind: str
    n_q: int = 0
    n_kv: int = 0
    qk_dim: int = 0
    v_dim: int = 0
    kv_lora: int = 0
    rope_dim: int = 0
    window: int | None = None
    topk: int | None = None
    compress: int = 1
    idx_heads: int = 0
    idx_dim: int = 0
    idx_rope: int = 0          # indexer RoPE dims (q per head and k)
    idx_fp8: bool = False      # DeepSeek-V3.2 reference indexer: Hadamard-rotated q / k quantised to fp8 e4m3 with one
                               # fp32 scale per 128 elements; key cache fp8 + scales (inference/model.py, Indexer)
    # DeepSeek-V4.1 (official inference/model.py): every attending query reads its window AND top-k compressed entries
    # (keys = min(p, w) + min(top-k, ⌊p/r⌋), exact); entry bytes per cache; compressed KV / index keys shared by layers
    dual: bool = False
    win_bytes: float = 0.0     # bytes per window-cache entry (fp8 + e8m0 scale per 32)
    cmp_bytes: float = 0.0     # bytes per compressed entry (fp4 + e4m3 scale per 16)
    kv_owner: bool = False     # this layer runs the Compressor and stores the compressed cache (kv_source_layers)
    idx_owner: bool = False    # this layer derives + stores the index keys (wk on the latent; fp4)
    idx_kbytes: float = 0.0    # bytes per index-key entry
    cmp_noape: bool = False    # Compressor without the learned position bias (V4.1)
    cand: str = ""             # two-level top-k: src (select_candidate_blocks) | use (mask to the source's blocks)
    cand_block: int = 0
    cand_topk: int = 0
    idx_split: bool = False    # DeepSeek-V4 reference: indexer heads column-parallel over TP + all-reduce of the scores
    idx_kshard: bool = False   # 0.87 key_split: cached index keys sharded over TP, local top-k + gather (opt-in)
    cmp_coff: int = 0          # DeepSeek-V4 Compressor: 1, or 2 for the overlapping ratio-4 windows (0 = no compressor)
    idx_cmp: bool = False      # DeepSeek-V4 indexer keys come from its own (Hadamard-rotated) compressor
    rope_n: int = 0            # rotary dims actually rotated per head when it differs from the MLA latent split
    qsa: int = 0               # Qwen4Exp QSA: key-block size; query sees every token until > top-k/qsa complete blocks,
                               # then top-k tokens of the best blocks + the incomplete tail block
    idx_sum: bool = False      # 0.144 Step-5 CSA region summaries: each completed region of ``qsa`` keys is compressed
                               # once into a cached fp32 + fp8 summary (kernel.py csa_compress_regions); the
                               # indexer reads the fp8 summaries, no per-forward re-pooling
    idx_kpool: int = 0         # GLM-5.3-Flash: indexer scores gated k-pools of this many keys (top-k/kpool pools)
    idx_score_type: str = ""   # 0.355 MiniMax M3.1-Flash block-level MSA: max (max-scored blocks) | mean (mean pooling); "" = token-level
    lin: str = ""              # linear-attention recurrence: gdn | kda (delta rule, scalar / per-channel gate) | lightning
    lin_chunk: int = 0         # prefill chunk (block) length of the reference chunked kernel
    sink: bool = False         # learned attention sink per head (gpt-oss): one extra softmax logit per query row
    n_state_heads: int = 0
    state_dk: int = 0
    state_dv: int = 0
    state_groups: int = 0      # 0.278 Mamba-2 SSD: B / C (the key / query side, dk wide) shared by heads in this many
                               # groups (n_groups) — C·Bᵀ is formed per group, not per head; 0 = per head
    conv_channels: int = 0
    conv_kernel: int = 0
    causal: bool = True        # False: bidirectional (DiT / protein encoders)
    # 0.292 MoVA (Jais-3-MoVA, modeling_k2_horizon K2HorizonMoVAAttention 「来源」): the value projection is v_experts
    # routed linears h → n_kv·v_dim (no bias), each token's v = Σ_top-v_topk w_e · silu(W_e x) from a sigmoid v_router
    # (+ selection bias); attn_linears carry v_router [h, E] and v_experts [h, E·n_kv·v_dim] (params = all experts)
    v_experts: int = 0
    v_topk: int = 0
    out_gate_fn: str = ""      # 0.292 attention output gate o · f(gate_proj x) applied per element: softplus (k2_horizon)
    cross: bool = False        # cross-attention: keys are the conditioning tokens (``full`` phase ``ctx``)
    # 0.170 StripedHyena 2 (Evo 2, vortex engine.py): lin = "hyena"; conv_channels / conv_kernel = the featurizer short
    # FIR over the 3h projections; the inner filter of the h-wide operator is
    #   se  short explicit FIR (hcs, hy_len taps, direct depthwise conv)
    #   mr  medium explicit FIR (hcm, hy_len taps; the reference prefills it with an FFT conv at fft size 2L)
    #   lr  long implicit filter (hcl, modal form: hy_modes real poles / residues per channel; FFT conv at 2L; decode
    #       = diagonal recurrence on an fp32 state of hy_modes per channel)
    hy: str = ""
    hy_len: int = 0
    hy_groups: int = 0          # filter groups (channels sharing one explicit filter)
    hy_modes: int = 0
    hy_prefill: str = ""        # lr prefill for generation: fft (modal-FFT state extraction, the release config) |
                                # recurrence (direct per-token recurrence)
    span: str = "full"         # full-phase self-attention extent: full (3D) | spatial (per latent frame) |
                               # temporal (per spatial position) — factorized ST-DiT (Open-Sora STDiT3)
    # 0.361 RWKV-8 DeepEmbedAttention (BlinkDL rwkv_v8_rc00_hybrid_demo.py forward_seq).  The cache stores the
    # low-rank kk / vv rows (dea_k, dea_v), fp16; each step rebuilds K (dea_kq) and V (dea_vd) over the whole
    # cache, multiplies the per-token ennk / emm rows (dea_k_emb, dea_v_emb), then one causal head.  0 = none.
    dea_q: int = 0
    dea_k: int = 0
    dea_kq: int = 0
    dea_v: int = 0
    dea_vd: int = 0
    dea_k_emb: int = 0
    dea_v_emb: int = 0
    # 0.371 EXAONE-4 full layers: Exaone4Attention applies RoPE only when sliding_window is
    # None or the layer is sliding. rope_dim 0 still means "half head" in the GQA rope op,
    # so this flag is what actually drops that vector.
    no_rope: bool = False
    # 0.437: the lumped qk_norm_rope op charges a QK-norm (3 element-ops, no rsqrt) on every
    # full-attention model. ESM-2 has rotary and no QK-norm, so that half is not billed.
    # q_scale is the published query multiply by head_dim**-0.5 (one multiply, no upcast).
    qk_norm: bool = True
    q_scale: bool = False

    def kv_elems_per_token(self) -> float:
        """KV-cache elements stored per token (one layer, whole model width)."""
        if self.kind == "gqa":
            return self.n_kv * (self.qk_dim + self.v_dim) / self.compress
        if self.kind == "mla":
            return (self.kv_lora + self.rope_dim) / self.compress
        return 0.0

    def idx_elems_per_token(self) -> float:
        return float(self.idx_dim) / self.compress if self.idx_heads else 0.0

    def idx_bytes_per_token(self, kvb: float) -> float:
        """Indexer key-cache bytes per token: fp8 key + fp32 scale per 128 (DeepSeek-V3.2 reference), else the
        KV-cache dtype (transformers GlmMoeDsaIndexer caches the bf16 key)."""
        if not self.idx_heads:
            return 0.0
        if self.idx_sum and self.qsa:   # Step-5: fp32 + fp8 summary per completed region (+ ≤ qsa pending raw keys)
            return self.idx_dim * 5 / self.qsa
        if self.idx_kpool:      # transformers Glm5NextTextIndexer packs [k, gate scores, valid] in the hidden dtype
            return (2 * self.idx_dim + 1) * kvb / self.compress
        per = self.idx_dim + 4 * math.ceil(self.idx_dim / 128) if self.idx_fp8 else self.idx_dim * kvb
        return per / self.compress

    def state_bytes(self, tp: int, stb: float) -> float:
        """Recurrent state bytes per sequence on one TP rank (linear attention / conv / hyena)."""
        cd = lambda a: -(-a // tp)
        if self.hy:
            # vortex: FIR caches hold the bf16 inputs (u[..., -(len−1):]); the modal IIR state is fp32
            fir = cd(self.conv_channels) * max(0, self.conv_kernel - 1) + cd(self.conv_channels // 3) * max(0, self.hy_len - 1)
            return fir * 2.0 + cd(self.n_state_heads) * self.state_dk * self.state_dv * 4.0
        return (cd(self.n_state_heads) * self.state_dk * self.state_dv + cd(self.conv_channels) * max(0, self.conv_kernel - 1)) * stb

    def state_elems(self) -> int:
        """Recurrent state per sequence (linear attention) incl. conv state."""
        if self.kind != "linear":
            return 0
        return self.n_state_heads * self.state_dk * self.state_dv + self.conv_channels * max(0, self.conv_kernel - 1)

    def ctx_eff(self, ctx: int) -> int:
        """Keys actually attended per query at context length ``ctx``."""
        if self.qsa:
            return ctx if ctx // self.qsa <= self.topk // self.qsa else self.topk + ctx % self.qsa
        if self.dual:
            return min(ctx, self.window or 0) + min(self.topk if self.topk is not None else ctx, ctx // self.compress)
        c = ctx
        if self.window is not None:
            c = min(c, self.window)
        if self.compress > 1:
            c = math.ceil(ctx / self.compress) + (self.window or 0)
        if self.topk is not None:
            c = min(c, self.topk + (self.window or 0) if self.compress > 1 else self.topk)
        return min(c, ctx)      # 0.126: compressed + window never more keys than tokens (decode = keys_sum's min(p, ·))

    def keys_sum(self, n: int) -> int:
        """0.61.1: Σ_{p=1..n} keys attended by the query at position p, min(p, ctx_eff(p)) — exact closed form.
        Prefill of a windowed / top-k / compressed layer charges the mean over its positions (the first queries see
        fewer than the cap).  compress > 1: min(p, ⌈p/c⌉ + w, top-k + w)."""
        if n <= 0:
            return 0
        c, w = self.compress, self.window or 0
        if self.qsa:                            # Σ keys(p): p up to p0 − 1, then top-k + p mod b
            b, p0 = self.qsa, (self.topk // self.qsa + 1) * self.qsa
            if n < p0:
                return n * (n + 1) // 2
            m = n - p0 + 1                      # positions p0..n, p0 ≡ 0 (mod b)
            a, r = divmod(m, b)
            return (p0 - 1) * p0 // 2 + m * self.topk + a * (b * (b - 1) // 2) + r * (r - 1) // 2
        if self.dual:                           # Σ min(p, w) + Σ min(top-k, ⌊p/c⌋)
            ws = n * (n + 1) // 2 if n <= w else w * (w + 1) // 2 + (n - w) * w

            def floor_sum(m: int) -> int:       # Σ_{p=1..m} ⌊p/c⌋
                a, r = divmod(m, c)
                return c * a * (a - 1) // 2 + a * (r + 1)
            k = self.topk
            if k is None or n < (k + 1) * c:
                return ws + floor_sum(n)
            pk = k * c                           # ⌊p/c⌋ ≥ k from p = k·c on
            return ws + floor_sum(pk - 1) + (n - pk + 1) * k
        if c <= 1:
            cap = min(x for x in (self.window, self.topk, n) if x is not None)
            return n * (n + 1) // 2 if n <= cap else cap * (cap + 1) // 2 + (n - cap) * cap
        top = self.topk + w if self.topk is not None else None

        def ceil_sum(m: int) -> int:           # Σ_{p=1..m} ⌈p/c⌉
            a, r = divmod(m, c)
            return c * a * (a + 1) // 2 + r * (a + 1)

        def last(ok) -> int:                   # largest p in [0, n] with ok(p) (ok monotone: true then false)
            lo, hi = 0, n
            while lo < hi:
                mid = (lo + hi + 1) // 2
                lo, hi = (mid, hi) if ok(mid) else (lo, mid - 1)
            return lo

        g = lambda p: -(-p // c) + w
        p1 = last(lambda p: p <= g(p))         # below p1 every key exists in the compressed cache + window
        hsum = lambda m: m * (m + 1) // 2 if m <= p1 else p1 * (p1 + 1) // 2 + (ceil_sum(m) - ceil_sum(p1)) + w * (m - p1)
        if top is None:
            return hsum(n)
        p3 = last(lambda p: min(p, g(p)) <= top)
        return hsum(p3) + (n - p3) * top

    def keys_mean(self, ctx: int, q: int) -> float:
        """Mean keys per query over the q positions after a cached prefix of ``ctx`` (0.61.1)."""
        return (self.keys_sum(ctx + q) - self.keys_sum(ctx)) / q


@dataclass(frozen=True)
class PairCore:
    """Activation × activation work of one structure-model module (0.43; AlphaFold-family trunks).  The weight GEMMs
    of the module are ordinary ``Linear``s in ``Layer.pair_linears``; this records only the work without weights.

    kind
      trimul    – triangle multiplicative update: ``dim`` channels of an [N×N]·[N×N] product (outgoing / incoming)
      tri_att   – triangle attention (starting / ending node): N groups of N keys, ``heads`` × ``dim``, pair bias
      row_att   – attention along the residue axis of each row of ``over`` (msa / xmsa: S groups; tmpl: T·N groups)
      col_att   – MSA column attention: N groups of S keys (``glob``: one averaged query per column, AF2 extra MSA)
      opm       – outer-product mean: [N·dim × S]·[S × N·dim2] (then a Linear from dim·dim2 to the pair width)
      pwa       – pair-weighted averaging (AF3 MSA module): ``heads`` × [N×N]·[N × S·dim]
      seq_att   – attention over the N tokens with a pair bias (single track, diffusion transformer, IPA)
      local_att – atom attention in windows of ``win[0]`` queries × ``win[1]`` keys
    ``v_dim`` – value width when it differs from ``dim`` (IPA: scalar + point + pair values).
    ``width`` – channel width of the representation the module reads (c_z / c_m / c_t, from the input dim of its
                query / projection weight): the payload of a DAP transpose (0.45)."""
    kind: str
    heads: int = 1
    dim: int = 0
    dim2: int = 0
    over: str = "pair"
    v_dim: int = 0
    glob: bool = False
    win: tuple[int, int] = (0, 0)
    width: int = 0


@dataclass(frozen=True)
class Ffn:
    kind: str                    # dense | moe | none
    d_ff: int = 0                # dense intermediate; on a moe block, a parallel dense SwiGLU of this width (0 = none)
    gated: bool = True           # SwiGLU (3 mats) vs 2 mats
    n_experts: int = 0
    top_k: int = 0
    d_expert: int = 0
    n_shared: int = 0
    d_shared: int = 0
    latent: int = 0              # latent MoE: experts operate in this width (Kimi-K3)
    # 0.255 group-limited routing (DeepSeek-V3 MoEGate / vLLM grouped_topk / Beam / Bailing): (n_group, topk_group) —
    # a token's top-k experts come from topk_group of n_group contiguous expert groups; () = unrestricted
    groups: tuple[int, ...] = ()
    # Identity experts scored by the router and applied as a copy of the token. They have no parameters.
    zero_experts: int = 0
    # 0.384 second routed pool (ERNIE-4.5-VL). Same top-k, different expert width.
    # A text token reads pool A only. A multimodal token reads pool B only.
    # Both are zero on every earlier MoE.
    n_experts_b: int = 0
    d_expert_b: int = 0
    # 0.399: routed-expert weight role when a block is not the model's expert format.
    # Empty keeps the expert role (every release before LongCat-Flash FP8).
    expert_role: str = ""

    @property
    def mats(self) -> int:
        return 3 if self.gated else 2

    def expert_in(self, hidden: int) -> int:
        return self.latent or hidden


@dataclass(frozen=True)
class Layer:
    attn_linears: tuple[Linear, ...]
    core: AttnCore
    ffn: Ffn
    ffn_linears: tuple[Linear, ...] = ()     # dense / shared / router / latent projections
    misc_params: int = 0                     # norms, gates, conv, hc — not GEMMs
    hc: int = 0                              # mHC streams (GLM-5.3-Flash): attn + ffn hyper-connection compute per layer
    hc_iters: int = 0                        # Sinkhorn-Knopp iterations of the comb weight
    hc_rows: int = 0                         # 0.320: selector GEMM rows when not (2+N)·N (Raon_mHC: 2N + (N−1)²)
    gres_n: int = 0                          # Qwen4Exp gated residual: streams (attn + mlp GatedResidual per layer)
    gres_head: bool = False                  # last layer: final hyper_connection_mixer (no combine)
    ple_rows: int = 0                        # Qwen4Exp PLE layer: n-gram rows per token (heads × (ngram − 1))
    ple_row_dim: int = 0
    ple_conv: int = 0                        # dilated depthwise conv kernel (dilation = ngram_size)
    ple_dil: int = 0
    engram_cols: int = 0                     # V4.1 Engram: hash rows fetched per token (n-gram sizes × heads)
    engram_dim: int = 0                      # row width (fp8 + e8m0 scale per 32)
    hc_head: str = ""                        # last layer only: final stream collapse — mean (GLM-5.3-Flash) | mix (V4)
    cross_linears: tuple[Linear, ...] = ()   # cross-attention projections (video DiT: q from tokens, k/v from text)
    cross: AttnCore | None = None
    post_norm: bool = False                  # 0.144 sandwich RMSNorms on attention / FFN outputs (LingBot-Video)
    fused_out: bool = False                  # parallel attention + MLP block with one fused output GEMM
                                             # (HunyuanVideo single-stream): one TP all-reduce per block
    # structure models (0.43): a block of the AlphaFold-family trunk / diffusion module
    pair_linears: tuple[Linear, ...] = ()    # every weight GEMM of the block (rows tag = the grid it runs on)
    pair_cores: tuple[PairCore, ...] = ()
    repeat: str = ""                         # "" once | recycle (× trunk passes) | diff (× diffusion steps)
    iters: int = 1                           # fixed repetitions per pass with shared weights (structure module: 8)
    samples: str = ""                        # rows × diffusion samples: "" no | tok (token / atom grids; diffusion
                                             # module, pair conditioning shared) | all (every grid; confidence head)
    misc_role: str = ""                      # format role of misc_params ("" = attn; structure trunks: pair)
    stack: str = ""                          # display label of the stack (esm / trunk / msa / pairformer / diffusion …)
    # 0.253 looped LM (Dory): loop > 0 = a later application of a shared recurrent layer — same weights (stored once,
    # counted once, but streamed again by every execution); kv_shared = its KV cache is the loop-0 layer's (last-loop
    # caching: each loop overwrites the step's rows) → no own KV storage, same KV reads / writes
    loop: int = 0
    kv_shared: bool = False
    # 0.369 Gemma-4: this layer reads the donor layer's KV tensors (no own cache, kv_write = 0).
    # Distinct from kv_shared, which still writes (Dory last-loop overwrites the donor slot).
    kv_alias: bool = False
    deepemb: int = 0                         # 0.361 RWKV-8 DeepEmbed width (ffn.enn row); 0 = none.  Current-token lookup.
    ple_in: int = 0                          # 0.369 Gemma-4 per-layer input width (gate h→ple, proj ple→h). 0 = none.
    # 0.437 ESM-2 encoder: published pre-LN + erf-GELU. "" keeps the shared full-forward assumption.
    vec_kind: str = ""

    def expert_params(self, hidden: int) -> int:
        f = self.ffn
        if f.kind != "moe":
            return 0
        p = f.n_experts * f.mats * f.expert_in(hidden) * f.d_expert
        if f.n_experts_b and f.d_expert_b:
            p += f.n_experts_b * f.mats * f.expert_in(hidden) * f.d_expert_b
        return p

    def params(self, hidden: int) -> int:
        if self.loop:
            return 0
        return (sum(l.params for l in self.attn_linears) + sum(l.params for l in self.ffn_linears)
                + sum(l.params for l in self.cross_linears) + sum(l.params for l in self.pair_linears)
                + self.expert_params(hidden) + self.misc_params)


@dataclass(frozen=True)
class RoleFormat:
    fmt: str            # format name in core.dtypes.FORMATS (or what-if override)
    bits: float         # effective stored bits / param incl. scales (from release bytes)


@dataclass(frozen=True)
class NativeWorkload:
    """Native workload of a non-autoregressive release (video DiT clip / protein sequence).  Defaults come from the
    official config / README (cited in the catalog); the scenario's ``workload`` block overrides them."""
    kind: str = ""                 # gen | protein | genomics
    frames: int = 0                # output video frames
    height: int = 0                # output pixels
    width: int = 0
    fps: int = 0
    vae_t: int = 4                 # VAE temporal / spatial compression
    vae_s: int = 8
    patch: tuple[int, int, int] = (1, 2, 2)   # (t, h, w) latent patch
    text_tokens: int = 0           # conditioning tokens (padded text length)
    prefix_tokens: int = 0         # text tokens concatenated into the attention sequence (joint attention)
    steps: int = 50                # denoise steps (reference sampler default)
    cfg: int = 2                   # forward passes per step (classifier-free guidance cond + uncond)
    vae_frames: str = "causal"     # latent-frame rule: causal (F−1)/t+1 | wan (F snapped to F//t·t+1, 0.117) | chunk17 (Open-Sora 1.2: 17 → 5 per chunk) |
                                   # h3 (MiniMax-H3: F snapped to 17n+5 → 5n+2)
    audio_per_s: int = 0           # joint audio rows per second of video (MiniMax-H3: 40 latents/s × channels)
    audio_channels: int = 0
    seq_len: int = 0               # protein residues
    msa: int = 0                   # structure models (0.43): MSA rows / extra-MSA rows / templates
    xmsa: int = 0
    templates: int = 0
    tmpl_msa: bool = False         # AF2 / OpenFold (0.61.3): the T template torsion-angle rows are concatenated to the
                                   # MSA representation and run through every Evoformer block (MSA grid = msa + T)
    atoms_per_res: float = 0.0     # all-atom models: heavy atoms per residue
    recycles: int = 0              # trunk passes (incl. the first)
    diff_steps: int = 0            # diffusion sampler steps
    samples: int = 0               # diffusion samples per request (batched)
    pair_dim: int = 0              # pair-representation width c_z (residual-stream / p2p accounting)
    max_seq: int = 0               # longest trained sequence (residues)
    special_tokens: int = 0        # protein: <cls> + <eos>
    kmer: int = 0                  # genomics (0.106): bases per token (Carbon: non-overlapping 6-mers)
    overlap: int = 0               # genomics: tokens shared by consecutive windows of a long record
    cond_causal: bool = False      # 0.116 image DiT: block-causal condition mask — text attends causally to text only,
                                   # image tokens to [text ‖ image] (Qwen-Image-2.1 causal_condition)
    max_refs: int = 0              # 0.252 image edit: reference images the release accepts (FLUX 3: MAX_REFERENCES 10);
                                   # 0 = no reference stream (workload.ref_images refused)
    prefix_kv: bool = False        # 0.116: the reference caches the text-prefix K/V at step 1 and runs later steps on the
                                   # image tokens only (needs cond_causal: the prefix is timestep-independent)
    uncond: tuple = ()             # 0.258: the CFG negative branch's own shape when the reference runs it differently:
                                   # (text tokens, skipped block indices) — e.g. MOSS-Image: empty negative prompt, block 22
                                   # skipped (uncond_skip_layers default); () = same shape as the conditional branch
    cfg_interval: tuple = ()       # 0.331: guidance interval (lo, hi) on the progress grid k / steps — the negative branch
                                   # runs only on steps with lo < k / steps < hi (PixText samplers: flow.py / meanflow.py
                                   # ``low < time_value < high``; Kynkäänniemi et al. 2024 limited-interval guidance);
                                   # () = every step.  Needs ``uncond`` (two serial forwards); see domain.guided_steps
    source: str = ""


@dataclass(frozen=True)
class ModelSpec:
    id: str
    hf_id: str
    arch: str
    hidden: int
    vocab: int
    layers: tuple[Layer, ...]
    tie_embeddings: bool = False
    mtp_layers: tuple[Layer, ...] = ()
    mtp_extra_params: int = 0                # eh_proj etc. per MTP module
    # 0.399: eh_proj params inside mtp_extra when that GEMM is not fmt("mtp").
    # 0 keeps the whole remainder on the mtp role.
    mtp_eh_params: int = 0
    mtp_eh_role: str = ""
    lookup_params: int = 0                   # n-gram / engram tables (lookup only)
    lookup_bits: float = 0.0                 # stored bits per lookup param incl. scales (0 = embed format)
    # 0.400 LongCat-Flash-Lite: NgramEmbedding order, splits, and per-table row width.
    # 0 means this checkpoint has no n-gram tables. Projections live in io_pre.
    ngram_order: int = 0
    ngram_splits: int = 0
    ngram_dim: int = 0
    # 0.402 LongCat-Flash-Omni: (input codebooks, residual codebooks, vocab).
    # () means this checkpoint has no audio codebook tables.
    audio_codebooks: tuple[int, int, int] = ()
    ple_packed: int = 0                      # 0.369 Gemma-4 PLE row width (layers * ple). 0 = none. One row per token.
    sidecar_params: int = 0                  # 0.347 same-card extras outside params_llm (MOSS audio heads, Moshi depth); storage only, not in params()
    sidecar_bits: float = 0.0                # 0 = attn format bits (used only when sidecar_bytes == 0)
    sidecar_bytes: float = 0.0               # 0.348 exact resident bytes (mixed dtypes: bf16 depth + fp32 Mimi); preferred over params*bits
    engram_ngram: int = 0                    # V4.1 engram_max_ngram_size (hash multiplies per token)
    draft: str = ""                          # dspark: V4.1 block draft head (one pass of draft_block positions)
    draft_block: int = 0
    draft_targets: int = 0                   # target layers whose stream means feed main_proj
    draft_markov: int = 0                    # markov head rank
    draft_kv: int = 0                        # 0.175: context K/V projection width per draft layer (0 = head_dim, V4.1)
    draft_reader: tuple[int, ...] = ()       # 0.221 ReaDy recurrent-state reader: (rank, state layers, heads, key dim,
                                             # value dim, bottleneck, draft injection layer); () = none
    # 0.252 DSpine (tencent, DFlash anchor-row block draft + selected-prefix code bridge): anchor-row semantics (block B =
    # anchor + B − 1 drafts, causal block attention, no Markov / confidence head); bridge = (code_size, rank, head_topk, *active draft layers 0-based); cold = codebook params (only anchor rows read); table_bytes = load-time serial tables
    draft_anchor: bool = False
    draft_bridge: tuple[int, ...] = ()
    draft_cold_params: int = 0
    draft_table_bytes: float = 0.0
    # 0.254 DFlash2 (vLLM qwen3_dflash2 / models/beam dflash2): anchor-row block draft, NON-causal block attention,
    # two grouped dynamic convolutions per draft layer + candidate selector; (taps, conv group, selector rank,
    # selector top-k, trained block); the served block is 1 + spec_k (draft_block = the card's serving default)
    draft_dflash2: tuple[int, ...] = ()
    # 0.311 UBTree (inclusionAI Ling-3.0-tiny-UBTree; config ubtree_config + safetensors header 「来源」, serving code
    # unreleased): DFlash anchor-row block backbone + unigram top-K / bigram selector → a pruned verification tree;
    # (candidate K, selector rank, factorizer width, depths, tree budget, published-tree nodes); the target verifies
    # 1 + nodes tree tokens per step
    draft_ubtree: tuple[int, ...] = ()
    # 0.253 looped LM (Dory): (input layers, recurrent layers, loops, kv mode) and the physical layer list the loops
    # expand (``with_loops``); layers = input + recurrent × loops + output
    loop_spec: tuple = ()
    loop_phys: tuple = ()
    spec_published: tuple[float, int] = ()   # 0.221: published acceptance length τ and draft block k (「来源」), () = none
    spec_published_t1: float = 0.0          # 0.253: published τ at temperature 1 (sampled rounds), same block; 0 = none
    formats: tuple[tuple[str, RoleFormat], ...] = ()
    act_fmt: str = "bf16"
    kv_fmt: str = "bf16"
    state_fmt: str = "fp32"
    coverage: str = "full"
    notes: tuple[str, ...] = ()
    release_params: int | None = None        # logical LLM params in the safetensors (excl. MTP, vision)
    release_mtp_params: int | None = None
    release_bytes: int | None = None
    quantized_release: bool = False
    what_if: bool = False                    # formats overridden by the user
    coverage_reasons: tuple[str, ...] = ()   # what is approximated (empty ⇔ coverage == full)
    vision_params: int = 0                   # VLM vision encoder + projector params (bf16 as released; 0.62 modelled)
    vision: object = None                    # core.vision.VisionSpec (0.62; None = no vision tower / text-only model)
    max_ctx: int = 0                         # config max_position_embeddings (0 = unknown) — warning only (0.61.4)
    # ---- non-autoregressive domains (video generation DiT / protein encoders); defaults = LLM
    domain: str = "llm"                      # llm | gen | protein | genomics
    kv_cache: bool = True                    # autoregressive KV cache (False: one full-sequence forward)
    io_pre: tuple[Linear, ...] = ()          # input-side GEMMs on the first stage (patch / text / timestep embedders)
    io_post: tuple[Linear, ...] = ()         # output-side GEMMs on the last stage (un-patchify head, LM-head dense)
    io_misc_params: int = 0                  # io biases / unused stored tables (counted in storage, no GEMM)
    logit_scale: bool = False                # 0.321 Dory: learned per-vocab logit scale (+vocab params, 1 multiply / logit)
    # 0.258: self-attention cores of io blocks (text refiners): ((name, rows, heads, head_dim, blocks), ...); rows "txt"
    # = the joint text rows of the forward — the refiner GEMMs are io_pre Linears, this adds their attention core
    io_attn: tuple = ()
    final_norm_params: int | None = None     # None → hidden (LLM RMSNorm)
    adaln: bool = False                      # AdaLN modulation (shift/scale/gate) around each norm
    # 0.437: "esm2" bills the published EsmEncoder final LayerNorm and EsmLMHead
    # (erf-GELU, LayerNorm, decoder bias). "" keeps the shared full-forward formulas.
    vec_kind: str = ""
    workload: NativeWorkload | None = None
    vocab_head: bool = True                  # encoders: MLM decoder over the vocab on every token (0.106: False =
                                             # token-classification heads only, in io_post — Carbon-A)
    # 0.291 decision checkpoint (pplx-decider readout.safetensors [rows, hidden]): the head is this readout, not a vocab
    # lm_head — logits of ≤ rows answer codes at each question marker; 0 = an ordinary LM head
    readout_rows: int = 0
    standby_params: int = 0                  # stored but idle per forward (Wan2.2 A14B: the other noise-level expert)
    cached_params: int = 0                   # in the release but not loaded for inference (MiniMax-H3 AdaLN branches:
                                             # modulation outputs precomputed per timestep, README)
    cosy: object = None                      # 0.271 core.cosyvoice.CosySpec: CosyVoice2 / 3 flow + HiFT (speech output)
    echo: object = None                      # 0.279 core.echo.EchoSpec: Index-Echo S2ST bridge (mapper → CosyVoice3)
    moshi: object = None                     # 0.274 core.moshi.MoshiSpec: Moshi depformer + Mimi codec (per 80 ms frame)
    kimi: object = None                      # 0.273 core.kimi_audio.KimiSpec: Kimi-Audio detokenizer DiT + BigVGAN
    omni: object = None                      # 0.271 core.omni.OmniSpec: Qwen2.5-Omni Talker + token2wav (speech output)
    fish: object = None                      # 0.276 core.drama3.FishSpec: Drama 3 codebook-MTP tower (in layers) + frame timing
    longcat_codec: object = None             # 0.403 core.longcat_codec.LongcatCodecSpec: Omni 24k waveform decoder (fp32, not in params())
    tts: object = None                       # 0.269 core.tts.TtsSpec: speech-synthesis decode head (replaces the LM head)
    qwen3_tts: object = None                 # 0.338 core.qwen3_tts.Qwen3TtsSpec: code predictor + tokenizer decode
    qwen3_omni: object = None                # 0.345 core.qwen3_omni.Qwen3OmniSpec: Talker MoE + code predictor + code2wav
    moss_tts: object = None                   # 0.346 core.moss_tts.MossTtsSpec: Delay-pattern Qwen3 + n_vq audio heads
    # 0.270 encoder-decoder (Whisper): encoder positions each decoder sequence cross-attends to (1500 per 30 s window);
    # the cross K/V of every layer is computed at prefill and kept per sequence (fixed size, like recurrent state)
    enc_ctx: int = 0
    # 0.393 Granite Speech 5 CTC: the catalog model is the encoder. params() is vision_params
    # only (no embedding table). kv_cache stays false; evaluate costs one encoder forward.
    ctc_only: bool = False
    # 0.394 d1-omni: one bidirectional forward. trunk_params is the encoder plus the
    # decision head (params_llm). d1_buf is FastConformer BatchNorm state, not a parameter.
    d1: bool = False
    trunk_params: int = 0
    d1_buf: int = 0

    # ----- derived
    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def act_fmt_for(self, role: str) -> str:
        """0.93: GEMM input-activation format of ``role`` — a per-role "act_<role>" entry (what-if only, e.g. MXFP8
        activations into MXFP4 experts on Blackwell) or the model's act_fmt."""
        for k, v in self.formats:
            if k == "act_" + role:
                return v.fmt
        return self.act_fmt

    def fmt(self, role: str) -> RoleFormat:
        d = dict(self.formats)
        if role in d:
            return d[role]
        for fb in {"shared_expert": "mlp", "expert": "mlp", "mlp": "attn", "router": "attn",
                   "lm_head": "embed", "mtp": "expert"}.get(role, "attn"), "attn":
            if fb in d:
                return d[fb]
        return RoleFormat("bf16", 16.0)

    @property
    def embed_params(self) -> int:
        return self.vocab * self.hidden

    @property
    def head_params(self) -> int:
        if self.readout_rows:
            return self.readout_rows * self.hidden     # 0.291: the decision readout replaces the lm_head
        return 0 if self.tie_embeddings else self.vocab * self.hidden

    @property
    def is_pair(self) -> bool:
        """Structure model (AlphaFold-family / ESMFold trunk): has pair-representation blocks."""
        return any(L.pair_linears or L.pair_cores for L in self.layers)

    def params(self, include_mtp: bool = False) -> int:
        # 0.393 CTC-only encoder: the release is the audio tower. There is no token embedding,
        # final norm, or untied LM head to add on top of vision_params.
        if self.ctc_only:
            return self.vision_params
        # 0.394 d1-omni: params() is the bidirectional trunk plus the decision head.
        # The SigLIP2 tower and the FastConformer stay in vision_params.
        if self.d1:
            return self.trunk_params
        p = sum(l.params(self.hidden) for l in self.layers) + self.embed_params + self.head_params + self.lookup_params
        p += self.hidden if self.final_norm_params is None else self.final_norm_params
        p += sum(l.params for l in self.io_pre + self.io_post) + self.io_misc_params
        p += self.standby_params + self.cached_params
        if include_mtp:
            p += self.mtp_params()
        # 0.358: encoder-decoder (Whisper) includes the encoder — vision_params is part of the model, not an optional tower
        if self.enc_ctx and self.vision_params:
            p += self.vision_params
        return p

    def mtp_params(self) -> int:
        return (sum(l.params(self.hidden) for l in self.mtp_layers) + self.mtp_extra_params * len(self.mtp_layers)
                + self.draft_cold_params)

    def active_params(self) -> int:
        """Params touched per token (routed experts: top_k of n; lookup tables excluded — a few rows/token)."""
        p = self.params() - self.lookup_params - self.standby_params - self.cached_params
        for l in self.layers:
            if l.ffn.kind == "moe":
                f = l.ffn
                pool = f.n_experts + f.zero_experts
                if f.zero_experts and pool and (f.top_k * f.n_experts) % pool == 0:
                    k_hit = (f.top_k * f.n_experts) // pool
                else:
                    k_hit = f.top_k
                p -= (f.n_experts - k_hit) * f.mats * f.expert_in(self.hidden) * f.d_expert
                if f.n_experts_b and f.d_expert_b:
                    # Text token: pool B is stored and not read.
                    p -= f.n_experts_b * f.mats * f.expert_in(self.hidden) * f.d_expert_b
            if l.core.v_experts:          # 0.292 MoVA: top v_topk of the routed value experts per token
                p -= (l.core.v_experts - l.core.v_topk) * self.hidden * l.core.n_kv * l.core.v_dim
        return p

    @property
    def is_moe(self) -> bool:
        return any(l.ffn.kind == "moe" for l in self.layers)

    def param_check(self) -> dict:
        ours = self.params()
        ref = self.release_params
        rel = None if not ref else (ours - ref) / ref
        return {"ours": ours, "release": ref, "rel_err": rel,
                "mtp_ours": self.mtp_params(), "mtp_release": self.release_mtp_params}


# ---------------------------------------------------------------- helpers
# 0.270 Qwen2-Audio: the released text_config lists only the fields that differ from transformers Qwen2Config, whose
# defaults (configuration_qwen2.py 「来源」) fill the rest — Qwen-7B shape, MHA
_QWEN2_DEFAULTS = {"vocab_size": 151936, "hidden_size": 4096, "intermediate_size": 22016, "num_hidden_layers": 32,
                   "num_attention_heads": 32, "num_key_value_heads": 32, "max_position_embeddings": 32768,
                   "tie_word_embeddings": False}


def _is_kimi(cfg: dict) -> bool:
    return "MoonshotKimiaForCausalLM" in (cfg.get("architectures") or [])


def _tc(cfg: dict) -> dict:
    tc = cfg.get("text_config")
    mt = cfg.get("model_type")
    if mt == "qwen3_tts" and isinstance(cfg.get("talker_config"), dict):
        # 0.338 Qwen3-TTS: talker is the AR codec LM (Qwen3 + codec_head); text_embedding is looked up /
        # projected (from_release lookup); code_predictor + speech_tokenizer → core/qwen3_tts.py
        tc = {**cfg["talker_config"], "model_type": "qwen3",
              "tie_word_embeddings": False, "torch_dtype": cfg.get("torch_dtype") or "bfloat16"}
    if mt == "lingv3_tts" and isinstance(cfg.get("llm_config"), dict):
        tc = cfg["llm_config"]      # 0.269 Ming-3.0-tts-tiny: the BailingMoeV3 decoder config sits in llm_config
    if mt == "dense" and isinstance(cfg.get("ditar_config"), dict) and isinstance(cfg.get("llm_config"), dict):
        # 0.340 Ming-3.0-tts-0.5B: Qwen2.5-0.5B backbone in llm_config; DiTAR + AudioVAE → core/tts.py
        tc = cfg["llm_config"]
    if mt == "qwen3_asr" and isinstance((cfg.get("thinker_config") or {}).get("text_config"), dict):
        tc = cfg["thinker_config"]["text_config"]      # 0.270 Qwen3-ASR: thinker_config.text_config (Qwen3)
    if mt == "moss_tts_delay" and isinstance(cfg.get("language_config"), dict):
        # 0.346 MOSS-TTS Delay: Qwen3 language_config is the catalog LLM; emb_ext + audio lm_heads → core/moss_tts.py
        tc = dict(cfg["language_config"])
        tc.setdefault("model_type", tc.get("model_type") or "qwen3")
        return tc
    if mt == "qwen3_omni_moe" and isinstance((cfg.get("thinker_config") or {}).get("text_config"), dict):
        # 0.345 Qwen3-Omni: Thinker text_config (qwen3_omni_moe_text MoE); Talker MoE + code2wav → core/qwen3_omni.py
        tc = {**cfg["thinker_config"]["text_config"]}
        tc.setdefault("model_type", "qwen3_omni_moe_text")
    if mt == "qwen2_5_omni" and isinstance((cfg.get("thinker_config") or {}).get("text_config"), dict):
        # 0.271 Qwen2.5-Omni: the Thinker's text_config (Qwen2.5-7B shape; Qwen2 attention: q / k / v with bias —
        # transformers Qwen2_5OmniAttention 「来源」); talker / token2wav → core/omni.py
        tc = {**cfg["thinker_config"]["text_config"], "attention_bias": True}
    if mt in ("cosyvoice2", "cosyvoice3") and isinstance(cfg.get("llm_config"), dict):
        # 0.271 CosyVoice2 / Fun-CosyVoice3: the Qwen2.5-0.5B backbone (CosyVoice-BlankEN/config.json) decodes speech
        # tokens — vocab = the speech-token table / llm_decoder rows (header: 6564 / 6761); the text embedding and the
        # stored text lm_head become lookup tables (from_release)
        cv = cfg.get("cosyvoice") or {}
        tc = {**cfg["llm_config"], "vocab_size": cv.get("speech_vocab") or cfg["llm_config"]["vocab_size"],
              "tie_word_embeddings": False}
    if mt == "step_audio_2" and isinstance(tc, dict):
        # 0.272 Step-Audio-2-mini: transformers Qwen2Model(text_config) — Qwen2 attention has q / k / v bias; untied
        # lm_head (modeling_step_audio_2.py 「来源」)
        tc = {**tc, "attention_bias": True, "tie_word_embeddings": False}
    if mt == "fireredtts3" and isinstance(cfg.get("backbone"), dict):
        # 0.275 FireRedTTS3-Base: the backbone is transformers Qwen3Model(Qwen3_1_7B_ConfigDict) — a dict in
        # fireredtts3_base.py (the HF config.json carries only the DiT / patch-encoder fields) 「来源」; no lm_head (the
        # step output drives stop_head + the DiT flow head, core/fireredtts3.py); weights stored fp32
        tc = {**cfg["backbone"], "torch_dtype": "float32"}
    if mt == "index_echo_s2st" and isinstance(tc, dict):
        tc = {**tc, "model_type": tc.get("model_type") or "qwen3_5_text"}   # 0.279 Index-Echo: the ST decoder (stlm_llm)
    if mt == "nemotron_h_omni" and isinstance(cfg.get("llm_config"), dict):
        tc = cfg["llm_config"]      # 0.278 Nemotron-3.5-Super-VL: NemotronHForCausalLM(llm_config) 「来源」 modeling_nemotron_h_omni.py
    if mt == "fish_omni" and isinstance(tc, dict):
        # 0.276 Fish Audio Drama 3: text_config is the Qwen3.5-MoE backbone (fish_omni_text = qwen3_5_moe_text shape);
        # the decode head is the 4096 candidate lm_head rows (core/drama3.py) — vocab = candidates, the 252 288-row
        # embedding is looked up (from_release lookup_params)
        from .drama3 import n_candidates
        tc = {**tc, "model_type": "qwen3_5_moe_text", "vocab_size": n_candidates(cfg), "tie_word_embeddings": False}
    if mt == "moshi" and isinstance(cfg.get("lm"), dict):
        # 0.274 Kyutai Moshi (moshi.models.lm.LMModel; loaders.py _lm_kwargs 「来源」 — the HF repo has no config.json):
        # the temporal transformer is a 32-layer MHA (in_proj q / k / v, no bias), SiLU-gated MLP of
        # 2/3 · hidden_scale · dim = 11264, RMSNorm, RoPE, causal with context 3000 frames (a KV ring buffer = sliding
        # window); text head text_linear (untied, 32000); the inputs (text_emb 32001 rows + 16 audio emb tables) are
        # looked up and summed per 12.5 Hz frame (from_release lookup_params).  Depformer + Mimi → core/moshi.py
        lm = cfg["lm"]
        d, L = lm["dim"], lm["num_layers"]
        return {"model_type": "moshi", "hidden_size": d, "num_hidden_layers": L, "num_attention_heads": lm["num_heads"],
                "num_key_value_heads": lm["num_heads"], "intermediate_size": int(2 * lm["hidden_scale"] * d / 3),
                "vocab_size": lm["text_card"], "tie_word_embeddings": False, "max_position_embeddings": 10 ** 6,
                "layer_types": ["sliding_attention"] * L, "sliding_window": lm["context"], "torch_dtype": "bfloat16"}
    if _is_kimi(cfg):
        # 0.273 Kimi-Audio-7B-Instruct (modeling_moonshot_kimia.py 「来源」; KimiAudioConfig(Qwen2Config) — no model_type
        # of its own): 28 Qwen2 layers (q / k / v bias) + kimia_mimo_layers 6 more of the same shape that branch off
        # the hidden state after layer kimia_mimo_transformer_from_layer_index (21) and feed the audio head.  Every
        # step (prefill and decode) runs all 28 + 6 layers and both heads, each with its own KV cache → costed as one
        # 34-layer stack 「推导」 (same FLOPs, weights and KV bytes; the two branches' overlap is not modelled); untied
        return {**cfg, "num_hidden_layers": cfg["num_hidden_layers"] + cfg.get("kimia_mimo_layers", 0),
                "attention_bias": True, "tie_word_embeddings": False}
    if mt == "qwen2_audio" and isinstance(tc, dict) and not tc.get("hidden_size"):
        tc = {**_QWEN2_DEFAULTS, **tc}
    if mt == "whisper":
        # 0.270 Whisper: the autoregressive part is the text decoder (the encoder is the audio tower, core/vision.py)
        return {**cfg, "hidden_size": cfg["d_model"], "num_hidden_layers": cfg["decoder_layers"],
                "num_attention_heads": cfg["decoder_attention_heads"], "intermediate_size": cfg["decoder_ffn_dim"],
                "max_position_embeddings": cfg["max_target_positions"], "tie_word_embeddings": True}
    if isinstance(tc, dict) and tc.get("hidden_size"):
        return {**{k: v for k, v in cfg.items() if k != "text_config"}, **tc}
    return cfg


def _g(c: dict, *keys, default=None):
    for k in keys:
        v = c.get(k)
        if v is not None:
            return v
    return default


def _ffn_dense(h: int, d: int, gated: bool = True) -> tuple[Ffn, tuple[Linear, ...]]:
    if gated:
        lin = (Linear("gate_up", h, 2 * d, "col"), Linear("down", d, h, "row"))
    else:
        lin = (Linear("up", h, d, "col"), Linear("down", d, h, "row"))
    return Ffn("dense", d_ff=d, gated=gated), lin


def _ffn_moe(h: int, E: int, k: int, de: int, n_sh: int = 0, d_sh: int = 0, *, gated: bool = True,
             latent: int = 0, shared_gate: bool = False, n_experts_b: int = 0,
             d_expert_b: int = 0) -> tuple[Ffn, tuple[Linear, ...]]:
    lin = [Linear("router", h, E, "rep")]
    if n_experts_b and d_expert_b:
        lin.append(Linear("router_b", h, n_experts_b, "rep"))
    if n_sh and d_sh:
        lin += [Linear("shared_gate_up", h, 2 * n_sh * d_sh if gated else n_sh * d_sh, "col"),
                Linear("shared_down", n_sh * d_sh, h, "row")]
    if shared_gate:
        lin.append(Linear("shared_expert_gate", h, 1, "rep"))
    if latent:
        lin += [Linear("latent_up", h, latent, "rep"), Linear("latent_down", latent, h, "rep")]
    return Ffn("moe", gated=gated, n_experts=E, top_k=k, d_expert=de, n_shared=n_sh, d_shared=d_sh,
               latent=latent, n_experts_b=n_experts_b, d_expert_b=d_expert_b), tuple(lin)


def _gqa(h: int, H: int, KV: int, hd: int, *, vd: int | None = None, out_gate: bool = False,
         window: int | None = None) -> tuple[tuple[Linear, ...], AttnCore]:
    vd = vd or hd
    lin = (Linear("q", h, H * hd * (2 if out_gate else 1), "col"),
           Linear("k", h, KV * hd, "head", unit=hd),
           Linear("v", h, KV * vd, "head", unit=vd),
           Linear("o", H * vd, h, "row"))
    return lin, AttnCore("gqa", n_q=H, n_kv=KV, qk_dim=hd, v_dim=vd, window=window)


_LING = ("bailing_hybrid", "bailing_moe_v3_vl", "ming_v3_asr", "bailing_moe_v3")    # 0.145: Ling-3.1 (text) / Ling-3.0-VL (text_config flattened); 0.268: Ming-3.0-asr-tiny (BailingMoeV3 decoder)


def _mla(c: dict, h: int) -> tuple[tuple[Linear, ...], AttnCore]:
    H = c["num_attention_heads"]
    ql = c.get("q_lora_rank")
    kvl = c["kv_lora_rank"]
    nope = c.get("qk_nope_head_dim") or 0
    rope = c.get("qk_rope_head_dim") or 0
    vd = c.get("v_head_dim") or nope
    qd = nope + rope
    lin = []
    if ql:
        lin += [Linear("q_a", h, ql, "rep"), Linear("q_b", ql, H * qd, "col")]
    else:
        lin += [Linear("q", h, H * qd, "col")]
    lin += [Linear("kv_a", h, kvl + rope, "rep"), Linear("kv_b", kvl, H * (nope + vd), "col"),
            Linear("o", H * vd, h, "row")]
    misc = (ql or 0) + kvl
    core = AttnCore("mla", n_q=H, n_kv=1, qk_dim=qd, v_dim=vd, kv_lora=kvl, rope_dim=rope)
    return tuple(lin), core, misc


def _indexer(c: dict, h: int, ql: int | None) -> tuple[tuple[Linear, ...], int, int, int]:
    ih, idim, topk = c.get("index_n_heads") or 0, c.get("index_head_dim") or 0, c.get("index_topk")
    if not ih:
        return (), 0, 0, None
    lin = (Linear("idx_q", ql or h, ih * idim, "rep"), Linear("idx_k", h, idim, "rep"),
           Linear("idx_w", h, ih, "rep"))
    return lin, ih, idim, topk


def _list(v):
    if isinstance(v, str):
        try:
            return json.loads(v.replace("'", '"'))
        except Exception:
            return None
    return v


# ---------------------------------------------------------------- family builders
def _build_std(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """Llama-like GQA decoders, optionally MoE (Qwen2/3, Llama, Mistral, Mixtral,
    Phi-3/4, Yi, InternLM2/3, Seed-OSS, Qwen3-MoE, GLM-4.5, gpt-oss)."""
    h = c["hidden_size"]
    H = c["num_attention_heads"]
    KV = c.get("num_key_value_heads") or H
    hd = c.get("head_dim") or h // H
    L = c["num_hidden_layers"]
    notes, cov = [], "full"
    mt = c.get("model_type", "")
    E = _g(c, "num_local_experts", "num_experts", "n_routed_experts", default=0) or 0
    k = _g(c, "num_experts_per_tok", "experts_per_token", "num_experts_per_token", default=0)
    de = _g(c, "moe_intermediate_size", "intermediate_size")
    n_sh = c.get("n_shared_experts") or (1 if c.get("shared_expert_intermediate_size") else 0)
    d_sh = c.get("shared_expert_intermediate_size") or (de if c.get("n_shared_experts") else 0)
    hy3 = mt == "hy_v3"
    if hy3:
        # 0.251 Tencent Hy3 (HYV3ForCausalLM, Hy-Memory-30B-A3B): the routed and shared experts are expert_hidden_dim wide
        # (safetensors experts.*.gate_proj [768, 2048]; moe_intermediate_size 1536 = gate+up), num_shared_experts shared
        # experts of the same width, sigmoid router + fp32 expert_bias, q/k RMSNorm, dense layers from mlp_layer_types
        de = c.get("expert_hidden_dim") or de
        n_sh = c.get("num_shared_experts") or 0
        d_sh = de if n_sh else 0
    k2h = mt == "k2_horizon"
    if k2h:
        # 0.291 IFM Jais 3 (modeling_k2_horizon.py 「来源」): K2HorizonSparseMoeBlock = sigmoid router (+ bias when
        # moe_gate_bias) over num_experts, top-k renormalised × router_scaling_factor, plus num_shared_experts
        # shared experts of moe_intermediate_size (safetensors mlp.shared_experts.* [1792, 6144]); dense layers =
        # mlp_only_layers; GQA with partial RoPE (rope_head_dim) and group RMSNorm (layernorm_num_groups: same params)
        n_sh = c.get("num_shared_experts") or 0
        d_sh = de if n_sh else 0
    if mt == "bailing_moe":
        # 0.373 inclusionAI Ling-flash-2.0 (modeling_bailing_moe_v2.py BailingMoeV2SparseMoeBlock): one shared MLP
        # whose intermediate size is moe_intermediate_size * num_shared_experts. The config also stores
        # moe_shared_expert_intermediate_size, and the module does not read it or add a shared_expert_gate.
        n_cfg = int(c.get("num_shared_experts") or 0)
        n_sh = 1 if n_cfg else 0
        d_sh = int(de) * n_cfg if n_cfg else 0
    first_dense = c.get("first_k_dense_replace") or 0
    mlp_only = set(c.get("mlp_only_layers") or [])
    types = _list(c.get("layer_types")) or []
    window = c.get("sliding_window")
    layers = []
    for i in range(L):
        w = None
        if types and i < len(types) and types[i] == "sliding_attention":
            w = window
        elif mt in ("mistral", "mixtral") and window and window < (c.get("max_position_embeddings") or 0):
            w = window
        sparse = bool(E and i >= first_dense and i not in mlp_only)
        gate_fn = (c.get("attention_gate_func") or "") if k2h else ""
        alin, core = _gqa(h, H, KV, hd, window=w, out_gate=bool(gate_fn))      # gate_proj [H·hd, h] = q width × 2
        # 0.371 Exaone4Attention: RoPE when sliding_window is None (1.2B, every layer) or this
        # layer is sliding. A hybrid full layer (32B LLLG) is global NoPE. KV width is unchanged.
        if mt == "exaone4" and c.get("sliding_window") is not None and w is None:
            core = replace(core, no_rope=True)
        if gate_fn:
            core = replace(core, out_gate_fn=gate_fn)
        mova = k2h and sparse and (c.get("mova_num_experts") or 0) > 0
        if mova:
            # 0.292 K2HorizonDecoderLayer: sparse layers (MoE FFN) use K2HorizonMoVAAttention — v_proj replaced by
            # mova_num_experts routed value experts (top mova_num_experts_per_tok, silu, router weights renormalised ×
            # router_scaling_factor); safetensors self_attn.v_router [E, h] (+ bias [E]), self_attn.v_experts.N [KV·hd, h]
            Ev, kv_k = c["mova_num_experts"], c["mova_num_experts_per_tok"]
            alin = tuple(l for l in alin if l.name != "v")
            alin = alin[:2] + (Linear("v_router", h, Ev, "rep"), Linear("v_experts", h, Ev * KV * hd, "head", unit=hd)) \
                + alin[2:]
            core = replace(core, v_experts=Ev, v_topk=kv_k)
        if mt == "gpt_oss" or (mt == "flint" and c.get("softmax_type") == "learnable"):
            core = replace(core, sink=True)        # 0.315 Flint: self_attn.softmax_offset [H] = learned sink logits
        hy_dense = hy3 and i < len(c.get("mlp_layer_types") or ()) and c["mlp_layer_types"][i] == "dense"
        if E and i >= first_dense and i not in mlp_only and not hy_dense:
            # 0.315 Flint: the shared expert has no shared_expert_gate tensor (safetensors header)
            ffn, flin = _ffn_moe(h, E, k, de, n_sh, d_sh,
                                 shared_gate=bool(c.get("shared_expert_intermediate_size")) and mt != "flint")
        else:
            ffn, flin = _ffn_dense(h, c["intermediate_size"])
        # 0.291 Mellum (transformers MellumAttention = Qwen3MoeAttention: q_norm / k_norm RMSNorm over head_dim).
        # 0.371 EXAONE-4 uses the same shared head_dim weights. Olmo-3 norms the full projection
        # before the head view: q is H*hd, k is KV*hd (7B MHA → 2*hidden; 32B GQA → hidden + KV*hd).
        if mt == "olmo3":
            qk = H * hd + KV * hd
        elif (mt.startswith("qwen3") or mt in ("mellum", "flint", "exaone4") or (hy3 and c.get("qk_norm"))
              or (k2h and c.get("query_key_norm")) or (mt == "bailing_moe" and c.get("use_qk_norm"))):
            qk = 2 * hd
        else:
            qk = 0
        misc = 2 * h + qk
        if k2h and ffn.kind == "moe" and c.get("moe_gate_bias"):
            misc += E                                      # router bias (mlp.gate.bias [E])
        if mova and c.get("moe_gate_bias"):
            misc += core.v_experts                         # 0.292 v_router.bias [E_v] (selection only)
        if hy3 and ffn.kind == "moe" and c.get("moe_router_enable_expert_bias"):
            misc += E                                      # fp32 expert_bias (sigmoid router selection bias)
        if mt == "bailing_moe" and ffn.kind == "moe" and c.get("moe_router_enable_expert_bias"):
            misc += E                                      # fp32 expert_bias buffer (selection only)
        if c.get("attention_bias") or mt == "qwen2" or mt == "gpt_oss":
            misc += H * hd + 2 * KV * hd + (h if mt == "gpt_oss" else 0)
        if mt == "flint" and core.sink:
            misc += H  # 0.315 softmax_offset [H]
        if mt == "gpt_oss":
            misc += H  # attention sinks
            misc += E * (2 * de + h)  # expert biases
        layers.append(Layer(alin, core, ffn, flin, misc))
    if mt == "gpt_oss":
        notes.append("attention sink 按参考实现建模（每个 query 行的 softmax 多一个可学习 logit，随后丢弃）；"
                     "128-token 交替滑窗已建模")
    if mt == "flint":
        notes.append(f"Flint（RecursiveSI）：35 层滑窗 {window}（含当前 token）+ 14 层全注意力（全注意力层无 RoPE，不影响参数与计算量）；"
                     "每头可学习 attention sink（softmax_offset）与 q / k RMSNorm 已建模；sigmoid 路由 top-k 归一化（无路由偏置）")
    if any(l.core.window for l in layers) and mt not in ("gpt_oss", "flint"):
        notes.append(f"滑窗 {window} 已建模")
    if mt == "olmo3":
        notes.append("Olmo-3：q_norm / k_norm 是投影后的 RMSNorm（Q 为 H·head_dim，K 为 KV·head_dim），"
                     "在拆成多头之前做；post-attention / post-feedforward 两个 RMSNorm 与现有 2·hidden 同一笔。"
                     "YaRN 不改变 GEMM 形状，也不改变 KV 宽度")
    if mt == "exaone4":
        n_nope = sum(1 for L in layers if L.core.no_rope)
        if n_nope:
            notes.append(f"EXAONE-4：q / k RMSNorm(head_dim)；{n_nope} 层全注意力不做 RoPE"
                         "（sliding_window 非空且该层不是 sliding），滑窗层仍做 RoPE")
        else:
            notes.append("EXAONE-4：q / k RMSNorm(head_dim)；sliding_window 为空，每一层都做 RoPE")
    if mt == "bailing_moe":
        # partial_rotary_factor 0.5 rotates half the head. rope_dim 0 already bills that width, and the cache
        # stores the full head (apply_rotary concatenates the pass-through half). Fused QKV has the same params.
        notes.append("bailing_moe：融合 QKV 与分开的 q/k/v 参数量相同；q / k 为 RMSNorm(head_dim)。"
                     "partial_rotary_factor 0.5 旋转半个 head，缓存的 K 仍是整段 head，"
                     "周期沿用 GQA 的查询头半 head 向量。"
                     f"前 {first_dense} 层稠密 SwiGLU，其余为 sigmoid MoE；"
                     "共享专家是一个 MLP，宽 moe_intermediate_size·num_shared_experts，没有 shared_expert_gate。"
                     "expert_bias 是 fp32 buffer，只参与选专家。"
                     + ("" if (c.get("num_nextn_predict_layers") or 0)
                        else "config 没有 num_nextn_predict_layers，本发布也没有 MTP 权重。"))
    mtp = []
    n_mtp = c.get("num_nextn_predict_layers") or 0
    for _ in range(n_mtp):
        alin, core = _gqa(h, H, KV, hd)
        ffn, flin = _ffn_moe(h, E, k, de, n_sh, d_sh) if E else _ffn_dense(h, c["intermediate_size"])
        mtp.append(Layer(alin, core, ffn, flin, 2 * h))
    return layers, mtp, notes, cov


def _build_mla(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """DeepSeek-V3/R1/V3.1/V3.2, Kimi-K2.x, GLM-5 (MLA + optional DSA indexer)."""
    h = c["hidden_size"]
    L = c["num_hidden_layers"]
    E = c.get("n_routed_experts") or 0
    first_dense = c.get("first_k_dense_replace") or 0
    notes, cov = [], "full"
    ilin, ih, idim, topk = _indexer(c, h, c.get("q_lora_rank"))
    itypes = c.get("indexer_types") or []
    fp8 = c.get("model_type") == "deepseek_v32"
    if ih:
        n_sh = sum(1 for t in itypes[:L] if t == "shared")
        notes.append(f"DSA lightning indexer（{ih}×{idim}）按参考实现逐项建模：q·k 打分 GEMM"
                     + ("（fp8，Hadamard 旋转 + 每 128 元素 fp32 scale）" if fp8 else "")
                     + f"、ReLU、按头加权求和、top-{topk} 选择；稀疏注意力只读 top-{topk} 个 token"
                     + (f"；{n_sh} 层为 shared（复用前一 full 层的 top-k，无索引器权重 / 缓存）" if n_sh else ""))

    def layer(i: int) -> Layer:
        alin, core, misc = _mla(c, h)
        if ih and i < len(itypes) and itypes[i] == "shared":
            core = replace(core, topk=topk)       # transformers GlmMoeDsa: reuse the previous full layer's top-k
        elif ih:
            alin = alin + ilin
            core = replace(core, idx_heads=ih, idx_dim=idim, topk=topk, idx_fp8=fp8,
                           idx_rope=c.get("qk_rope_head_dim") or 0)
            misc += 2 * idim
        if E and i >= first_dense:
            ffn, flin = _ffn_moe(h, E, c["num_experts_per_tok"], c["moe_intermediate_size"],
                                 c.get("n_shared_experts") or 0, c["moe_intermediate_size"])
            misc += E  # e_score_correction_bias
        else:
            ffn, flin = _ffn_dense(h, c["intermediate_size"])
        return Layer(alin, core, ffn, flin, misc + 2 * h)

    layers = [layer(i) for i in range(L)]
    mtp = [replace(layer(L), misc_params=layer(L).misc_params + 3 * h) for _ in range(c.get("num_nextn_predict_layers") or 0)]
    return layers, mtp, notes, cov


def _gdn(c: dict, h: int) -> tuple[tuple[Linear, ...], AttnCore, int]:
    """Gated DeltaNet (Qwen3-Next / Qwen3.5)."""
    nk, nv = c["linear_num_key_heads"], c["linear_num_value_heads"]
    dk, dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
    kern = c.get("linear_conv_kernel_dim") or 4
    qkv = 2 * nk * dk + nv * dv
    lin = (Linear("lin_qkv", h, qkv, "col"), Linear("lin_z", h, nv * dv, "col"),
           Linear("lin_ba", h, 2 * nv, "col"), Linear("lin_out", nv * dv, h, "row"))
    core = AttnCore("linear", n_state_heads=nv, state_dk=dk, state_dv=dv, conv_channels=qkv, conv_kernel=kern,
                    lin="gdn", lin_chunk=64)   # transformers torch_chunk_gated_delta_rule(chunk_size=64) / FLA
    misc = qkv * kern + 2 * nv + dv
    return lin, core, misc


def _kda(c: dict, h: int) -> tuple[tuple[Linear, ...], AttnCore, int]:
    """Kimi Delta Attention (Kimi-Linear / Kimi-K3 / GLM-5.x-Next)."""
    raon = c.get("model_type") == "raon1_mini"
    lc = _list(c.get("linear_attn_config")) or (_list(c.get("rda_config")) if raon else None) or {}
    ling = c.get("model_type") in _LING       # 0.114: Ling-3.x keeps these at the top level
    H, hd = lc.get("num_heads", c["num_attention_heads"]), lc.get("head_dim", (c.get("head_dim") if ling else None) or 128)
    kern = lc.get("short_conv_kernel_size", lc.get("conv_kernel_size", (c.get("short_conv_kernel_size") if ling else None) or 4))
    d = H * hd
    # output gate: full g_proj (Kimi-K3 release) or low-rank g_a / g_b (transformers Glm5NextTextLinearAttention)
    gate = ((Linear("kda_g_a", h, hd, "rep"), Linear("kda_g_b", hd, d, "col")) if c.get("model_type", "").startswith("glm5_next")
            else (Linear("kda_g", h, d, "col"),))
    # 0.114: Ling-3.x (bailing_hybrid, no_kda_lora): full-rank f_proj h → d instead of the low-rank f_a / f_b
    fgate = ((Linear("kda_f", h, d, "col"),) if c.get("no_kda_lora") else
             (Linear("kda_f_a", h, hd, "rep"), Linear("kda_f_b", hd, d, "col")))
    # 0.320 Raon Delta Attention (raon1_mini, modeling_raon1_mini.Raon1MiniRDA): the KDA layout (per-channel low-rank
    # decay f_a / f_b of rank head_dim, full-rank g_proj, b_proj, A_log, dt_bias, per-head o_norm, q / k / v short convs)
    # plus rda_w_proj h → heads (write gate sigmoid(β + δ) over erase gate sigmoid(β), a per-head value scale)
    wgate = (Linear("rda_w", h, H, "col"),) if raon else ()
    lin = (Linear("kda_q", h, d, "col"), Linear("kda_k", h, d, "col"), Linear("kda_v", h, d, "col")) + gate + (
        Linear("kda_b", h, H, "col"),) + wgate + fgate + (Linear("kda_o", d, h, "row"),)
    core = AttnCore("linear", n_state_heads=H, state_dk=hd, state_dv=hd, conv_channels=3 * d, conv_kernel=kern,
                    lin="kda", lin_chunk=64)   # transformers chunk_kimi_delta_attention(chunk_size=64) / FLA chunk_kda
    misc = 3 * d * kern + d + H + hd
    return lin, core, misc


def _build_hybrid(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """Hybrid linear/full attention: Qwen3-Next, Qwen3.5/3.8, Kimi-K3, GLM-5.x-Next, MiniMax."""
    h = c["hidden_size"]
    L = c["num_hidden_layers"]
    mt = c.get("model_type", "")
    notes, cov = [], "full"
    types = _list(c.get("layer_types"))
    lc = _list(c.get("linear_attn_config")) or {}
    if not types:
        if lc.get("full_attn_layers"):
            full = {x - 1 for x in lc["full_attn_layers"]}   # Kimi: 1-based
            types = ["full_attention" if i in full else "linear_attention" for i in range(L)]
        elif c.get("full_attention_interval"):
            n = c["full_attention_interval"]
            types = ["full_attention" if (i + 1) % n == 0 else "linear_attention" for i in range(L)]
        elif c.get("attn_type_list"):
            types = ["full_attention" if t == 1 else "linear_attention" for t in c["attn_type_list"]]
        elif mt == "raon1_mini" and (c.get("rda_config") or {}).get("mla_layers"):
            # 0.320: rda_config.mla_layers is 1-based (release header: model.layers.3 / .24 carry the MLA tensors)
            full = {x - 1 for x in c["rda_config"]["mla_layers"]}
            types = ["full_attention" if i in full else "linear_attention" for i in range(L)]
        elif mt in _LING and c.get("layer_group_size"):
            # 0.114 Ling-3.x: groups of layer_group_size, the last of each group MLA, the rest KDA
            n = c["layer_group_size"]
            types = ["full_attention" if (i + 1) % n == 0 else "linear_attention" for i in range(L)]
    E = _g(c, "num_experts", "n_routed_experts", "num_local_experts", default=0)
    k = _g(c, "num_experts_per_tok", "num_experts_per_token")
    de = _g(c, "moe_intermediate_size", "intermediate_size")
    mlp_types = _list(c.get("mlp_layer_types"))
    first_dense = c.get("first_k_dense_replace") or 0
    kda = mt in ("kimi_linear", "raon1_mini") + _LING or mt.startswith("glm5_next")
    is_minimax = mt.startswith("minimax")

    def ffn_for(i: int):
        dense = (mlp_types and mlp_types[i] == "dense") or (i < first_dense) or not E
        if dense:
            return _ffn_dense(h, c["intermediate_size"])
        if mt == "kimi_linear":
            return _ffn_moe(h, E, k, de, c.get("num_shared_experts") or 0, de,
                            latent=c.get("routed_expert_hidden_size") or 0)
        n_sh = c.get("n_shared_experts") or c.get("num_shared_experts") or (1 if c.get("shared_expert_intermediate_size") else 0)
        if mt == "bailing_moe_v3_vl" and not n_sh:
            n_sh = 1        # 0.145: BailingMoeV3VLConfig num_shared_experts default 1 (absent from text_config)
        d_sh = c.get("shared_expert_intermediate_size") or (de if n_sh else 0)
        return _ffn_moe(h, E, k, de, n_sh, d_sh, shared_gate=bool(c.get("shared_expert_intermediate_size")))

    def full_attn(i: int):
        if c.get("kv_lora_rank"):
            alin, core, misc = _mla(c, h)
            ilin, ih, idim, topk = _indexer(c, h, c.get("q_lora_rank"))
            if ih and (not types or types[i] != "full_attention" or mt.startswith("glm5")):
                alin = alin + ilin
                core = replace(core, idx_heads=ih, idx_dim=idim, topk=topk, idx_rope=c.get("qk_rope_head_dim") or 0)
                kp = c.get("index_kpool") or 0
                if kp and c.get("index_kpool_compress"):
                    # transformers Glm5NextTextIndexer: gate GEMM h → idx_dim, learned per-slot APE, k LayerNorm (w + b);
                    # top-(topk / kpool) pools, expanded back to tokens, + the incomplete tail pool (≤ kpool − 1 tokens)
                    alin = alin + (Linear("idx_kpool_gate", h, idim, "rep"),)
                    misc += kp * idim + 2 * idim
                    core = replace(core, idx_kpool=kp,
                                   topk=topk + (kp - 1 if c.get("index_kpool_always_select_tail") else 0))
            if c.get("mla_use_output_gate"):
                alin = alin + (Linear("o_gate", h, core.n_q * core.v_dim, "col"),)
            elif mt in _LING and c.get("gated_attention_proj_granularity_type") == "head_wise":
                alin = alin + (Linear("o_gate", h, core.n_q, "col"),)    # Ling-3.x: one sigmoid gate per head
            return alin, core, misc
        H = c["num_attention_heads"]
        KV = c.get("num_key_value_heads") or H
        hd = c.get("head_dim") or h // H
        # 0.76: sigmoid output gate on the softmax-attention q_proj (×2) — Qwen3-Next hard-codes it (modeling_qwen3_next
        # Qwen3NextAttention, no config key); MiniMax-Text-01 / M1 softmax layers have none (MiniMaxText01Attention:
        # plain q/k/v/o — the output_gate belongs to the lightning layers, see lin_attn)
        gate = c.get("attn_output_gate")
        alin, core = _gqa(h, H, KV, hd, out_gate=bool(mt == "qwen3_next" if gate is None else gate))
        # MiniMaxM2Attention: when use_qk_norm, q_norm is RMSNorm(num_attention_heads * head_dim)
        # and k_norm is RMSNorm(num_key_value_heads * head_dim). The class does not read
        # qk_norm_type. A shared head_dim vector is the Qwen3-style bill, not this class.
        # With the flag off the module has no q/k norm. rotary_dim only slices the head.
        if mt == "minimax_m2":
            qk = (H * hd + KV * hd) if c.get("use_qk_norm") else 0
            return alin, core, qk
        return alin, core, 2 * hd

    def lin_attn(i: int):
        if kda:
            return _kda(c, h)
        if is_minimax:
            H = c["num_attention_heads"]
            hd = c.get("head_dim") or h // H
            lin = (Linear("lin_qkv", h, 3 * H * hd, "col"), Linear("lin_gate", h, H * hd, "col"),
                   Linear("lin_out", H * hd, h, "row"))
            # MiniMax remote code modeling_minimax_text_01.py: BLOCK = 256, kv state fp32
            return lin, AttnCore("linear", n_state_heads=H, state_dk=hd, state_dv=hd, lin="lightning",
                                 lin_chunk=256), H * hd
        return _gdn(c, h)

    layers = []
    for i in range(L):
        t = types[i] if types and i < len(types) else "full_attention"
        alin, core, misc = full_attn(i) if t != "linear_attention" else lin_attn(i)
        ffn, flin = ffn_for(i)
        if ffn.kind == "moe" and mt in ("kimi_linear", "raon1_mini") + _LING or (ffn.kind == "moe" and c.get("n_routed_experts")):
            misc += E
        # MiniMaxM2SparseMoeBlock always registers e_score_correction_bias [num_local_experts].
        # The published class does not read use_routing_bias. The gate GEMM is the router linear.
        if mt == "minimax_m2" and ffn.kind == "moe" and E:
            misc += E
        hc = c.get("hc_mult") or 0
        if mt == "raon1_mini" and c.get("raon_mhc_streams"):
            # 0.320 Raon_mHC ×2 per layer (attention + MLP site): selector [2N + (N−1)², N·h] (phi_pre / phi_post / phi_z),
            # b_pre / b_post (N), b_z ((N−1)²), alpha_pre / alpha_post / alpha_z, eta; closed-form residual chart
            # (Helmert transform of an (N−1)² matrix, no Sinkhorn) costed as one normalisation pass 「近似」
            n = c["raon_mhc_streams"]
            rows = 2 * n + (n - 1) ** 2
            misc += 2 * (rows * n * h + rows + 4)
            layers.append(Layer(alin, core, ffn, flin, misc + 2 * h, hc=n, hc_iters=1, hc_rows=rows))
        elif hc and mt.startswith("glm5_next"):
            # transformers Glm5NextTextHyperConnection ×2 per layer (attn_hc, ffn_hc): fn [(2+N)·N, N·h], base, scale
            misc += 2 * ((2 + hc) * hc * (hc * h + 1) + 3)
            layers.append(Layer(alin, core, ffn, flin, misc + 2 * h, hc=hc, hc_iters=c.get("hc_sinkhorn_iters") or 1))
        else:
            layers.append(Layer(alin, core, ffn, flin, misc + 2 * h))
    n_lin = sum(1 for l in layers if l.core.kind == "linear")
    notes.append(f"混合注意力：{n_lin}/{L} 层线性注意力（chunk 形式按参考实现上阵列，递归状态 fp32 按参考），"
                 f"{L - n_lin} 层全注意力")
    if mt == "minimax_m2":
        notes.append("MiniMax-M2：use_qk_norm 为 true 时 q / k RMSNorm 按整段投影宽度计（头数乘 head_dim），"
                     "不是每个 head 一条。e_score_correction_bias 每层都有，类不读 use_routing_bias。"
                     "rotary_dim 只旋转 head 的前一段，KV 仍是整段 head，不另计周期。"
                     + ("没有 mtp_num_hidden_layers / num_nextn_predict_layers，不另建 MTP 层。"
                        if not (c.get("mtp_num_hidden_layers") or c.get("num_nextn_predict_layers")) else ""))
    n_mtp = c.get("mtp_num_hidden_layers") or c.get("num_nextn_predict_layers") or 0
    mtp = []
    for _ in range(n_mtp):
        alin, core, misc = full_attn(L - 1)
        ffn, flin = ffn_for(L - 1)
        mtp.append(Layer(alin, core, ffn, flin, misc + 4 * h))
    return layers, mtp, notes, cov


def _build_minimax_m3(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """0.355 MiniMax M3.1-Flash (minimax_m3_vl): native multimodal MoE with block-level max-scored sparse attention (MSA).
    text_config: 60 layers, hidden 6144, 64 heads (128 dim) / 4 KV, MoE 128 local experts / 4 per tok / 1 shared (first 3
    layers dense), intermediate 3072 (routed) / 12288 (dense) / 3072 (shared), swigluoai (alpha 1.702 / limit 7.0), partial
    RoPE (64/128), max_pos 1048576.  sparse_attention_config: block_size 128, topk_blocks 16, index_heads 4, index_dim 128,
    score_type max (per-block max of indexer scores over block keys).  quantization_config: mxfp8 / NVFP4 MoE."""
    h = c["hidden_size"]
    L = c["num_hidden_layers"]
    H = c["num_attention_heads"]
    KV = c.get("num_key_value_heads") or H
    hd = c.get("head_dim") or h // H
    rope = c.get("rotary_dim") or 0
    E = c.get("num_local_experts") or 0
    k = c.get("num_experts_per_tok") or 0
    de = c.get("intermediate_size") or 0
    d_dense = c.get("dense_intermediate_size") or 0
    d_sh = c.get("shared_intermediate_size") or 0
    n_sh = c.get("n_shared_experts") or 0
    moe_freq = _list(c.get("moe_layer_freq")) or [0] * L
    sp_cfg = c.get("sparse_attention_config") or {}
    sp_freq = _list(sp_cfg.get("sparse_attention_freq")) or [0] * L
    sp_block = sp_cfg.get("sparse_block_size") or 0
    sp_topk_blocks = sp_cfg.get("sparse_topk_blocks") or 0
    ih = sp_cfg.get("sparse_num_index_heads") or 0
    idim = sp_cfg.get("sparse_index_dim") or 0
    score_type = sp_cfg.get("sparse_score_type") or ""
    notes = [f"MiniMax M3.1-Flash：MoE {E}×{de}（4 选 1 shared）+ 块稀疏注意力（块大小 {sp_block}，top-{sp_topk_blocks} 块，"
             f"{score_type} 打分，{ih} 索引头）"]

    def layer(i: int) -> Layer:
        alin, core = _gqa(h, H, KV, hd, out_gate=False)
        if sp_freq[i] if i < len(sp_freq) else 0:
            alin = alin + (Linear("idx_q", h, ih * idim, "rep"), Linear("idx_k", h, idim, "rep"))
            core = replace(core, qsa=sp_block, topk=sp_topk_blocks * sp_block, idx_heads=ih, idx_dim=idim,
                           idx_score_type=score_type, rope_n=rope)
        is_moe = moe_freq[i] if i < len(moe_freq) else 0
        if is_moe:
            ffn, flin = _ffn_moe(h, E, k, de, n_sh, d_sh)
            misc = 2 * hd + E
        else:
            ffn, flin = _ffn_dense(h, d_dense)
            misc = 2 * hd
        return Layer(alin, core, ffn, flin, misc + 2 * h)

    layers = [layer(i) for i in range(L)]
    return layers, [], notes, "full"


def _build_dsv4_ref(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """DeepSeek-V4-Flash / -Pro per the official inference/model.py (identical in both repos): MQA head_dim 512 (one
    shared K = V entry, attention sink, per-head q RMSNorm, inverse RoPE on o), grouped low-rank output, sliding
    window 128 + per-layer Compressor (ratio 4 overlapping / 128), ratio-4 layers with a compressed-key indexer
    (heads column-parallel, top-k), hash routing (gate scores still computed; experts from tid2eid), mHC ×hc_mult
    with a learned final collapse."""
    h = c["hidden_size"]
    L = c["num_hidden_layers"]
    H, hd = c["num_attention_heads"], c["head_dim"]
    ql, ol, og = c["q_lora_rank"], c["o_lora_rank"], c["o_groups"]
    rope = c.get("qk_rope_head_dim") or 64
    win = c.get("sliding_window") or 128
    ratios = _list(c.get("compress_ratios")) or [0] * (L + 1)
    ih, idim, topk = c.get("index_n_heads") or 0, c.get("index_head_dim") or 0, c.get("index_topk")
    hc, iters = c.get("hc_mult") or 1, c.get("hc_sinkhorn_iters") or 1
    E, k, de = c["n_routed_experts"], c["num_experts_per_tok"], c["moe_intermediate_size"]
    n_hash = c.get("num_hash_layers") or 0
    hc_p = 2 * ((2 + hc) * hc * (hc * h + 1) + 3)          # hc_attn / hc_ffn: fn, base, scale
    head_p = hc * hc * h + hc + 1                            # hc_head_fn / base / scale
    notes = [f"DeepSeek-V4 按官方 inference/model.py 逐项建模：滑窗 {win} + 压缩器（比 4 重叠 / 128）、比 4 层压缩 key 索引器"
             f"（{ih} 头 TP 切分 + 分数 all-reduce，top-{topk}）、attention sink、mHC ×{hc}（Sinkhorn {iters} 次）、"
             f"前 {n_hash} 层哈希路由（gate 分数照算，专家由 tid2eid 给定）"]

    def layer(i: int, last: bool = False) -> Layer:
        r = ratios[i] if i < len(ratios) else 0
        r = r if r and r > 1 else 1
        alin = [Linear("q_a", h, ql, "rep"), Linear("q_b", ql, H * hd, "col"), Linear("kv", h, hd, "rep"),
                Linear("o_a", H * hd, og * ol, "col", groups=og), Linear("o_b", og * ol, h, "row")]
        core = AttnCore("mla", n_q=H, n_kv=1, qk_dim=hd, v_dim=hd, kv_lora=hd, rope_dim=0, rope_n=rope,
                        window=win, compress=r, sink=True)
        misc = ql + hd + H                                   # q_norm, kv_norm, attn_sink
        if r > 1:
            cw = 2 if r == 4 else 1
            alin += [Linear("cmp_kv", h, cw * hd, "rep"), Linear("cmp_gate", h, cw * hd, "rep")]
            misc += cw * r * hd + hd                         # ape, norm
            core = replace(core, cmp_coff=cw)
            if r == 4 and ih:
                alin += [Linear("idx_q", ql, ih * idim, "col"), Linear("idx_cmp_kv", h, cw * idim, "rep"),
                         Linear("idx_cmp_gate", h, cw * idim, "rep"), Linear("idx_w", h, ih, "col")]
                misc += cw * r * idim + idim
                core = replace(core, idx_heads=ih, idx_dim=idim, topk=topk, idx_split=True, idx_cmp=True,
                               idx_rope=rope)
        ffn, flin = _ffn_moe(h, E, k, de, c.get("n_shared_experts") or 0, de)
        misc += 2 * h + hc_p + (0 if i < n_hash else E)      # norms, hc, gate bias (hash layers: tid2eid instead)
        if last:
            misc += head_p
        return Layer(tuple(alin), core, ffn, flin, misc, hc=hc, hc_iters=iters, hc_head="mix" if last else "")

    layers = [layer(i, i == L - 1) for i in range(L)]
    mtp = []
    for j in range(c.get("num_nextn_predict_layers") or 0):
        m_ = layer(L + j)
        mtp.append(replace(m_, misc_params=m_.misc_params + 3 * h + head_p, hc_head="mix"))
    return layers, mtp, notes, "full"


def _build_qwen4exp(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """Qwen3.8-Flash-Next (qwen4_exp) per transformers modeling_qwen4_exp: Gated DeltaNet / gated-q GQA with a QSA
    indexer (pooled key blocks, top budget/ratio blocks + tail); hc_count residual streams with two GatedResidual
    mixers per layer (low-rank input mix + block injection; no per-layer RMSNorms) and a final mixer; PLE n-gram
    layer(s) (hashed table rows → key / value projections, stream-gated, dilated depthwise conv); sparse MoE with
    a sigmoid-gated shared expert."""
    h = c["hidden_size"]
    L = c["num_hidden_layers"]
    types = c.get("layer_types") or []
    N, r = c["hc_count"], c["hc_lowrank"]
    Nh = N * h
    H, KV, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    rope = int(hd * (c.get("partial_rotary_factor") or 1))
    ih, ikv, idim = c["indexer_n_heads"], c["indexer_kv_heads"], c["indexer_head_dim"]
    budget, ratio = c["indexer_budget"], c["indexer_compress_ratio"]
    ple_ids = set(_list(c.get("ple_layer_ids")) or [])
    ng, hpn, ped = c.get("ngram_size") or 0, c.get("heads_per_ngram") or 0, c.get("ple_embed_dim") or 0
    prow = (ng - 1) * hpn
    pk = c.get("ple_conv_kernel_size") or 0
    E, k, de = c["num_experts"], c["num_experts_per_tok"], c["moe_intermediate_size"]
    notes = [f"Qwen4Exp 按 transformers modeling_qwen4_exp 逐项建模：门控残差 ×{N}（低秩 {r}）、QSA 索引注意力"
             f"（{ih} 头 × {idim}，块 {ratio}，预算 {budget} + 尾块）、PLE n-gram 层 {sorted(i - 1 for i in ple_ids)}"
             f"（每 token {prow} 行 × {ped // max(prow, 1)}，膨胀 conv {pk}）"]

    def gres(tag: str, combine: bool = True):
        # 0.280: role "other" = the release's gated-residual / PLE projection dtype (bf16 in both releases; the FP4-Local
        # quantization leaves them out while attention goes MXFP8)
        lin = (Linear(f"gr_{tag}_down", Nh, r, "rep", role="other"), Linear(f"gr_{tag}_up", r, Nh, "rep", role="other"))
        if combine:
            lin += (Linear(f"gr_{tag}_inj", Nh, N, "rep", role="other"),)
        return lin, Nh                                # + hc_norm weight

    def full(i: int):
        alin, core = _gqa(h, H, KV, hd, out_gate=True)
        alin = alin + (Linear("idx_qk", h, (ih + ikv) * idim, "rep"),)
        core = replace(core, qsa=ratio, topk=budget, idx_heads=ih, idx_dim=idim, idx_rope=rope, rope_n=rope)
        return alin, core, 2 * hd + 2 * idim

    def layer(i: int, typ: str, last: bool = False) -> Layer:
        alin, core, misc = full(i) if typ == "full_attention" else _gdn(c, h)
        ga, ma = gres("a")
        gm, mm = gres("m")
        alin = tuple(alin) + ga + gm
        misc += ma + mm
        ffn, flin = _ffn_moe(h, E, k, de, 1, c["shared_expert_intermediate_size"], shared_gate=True)
        kw = {}
        if i + 1 in ple_ids:
            alin += (Linear("ple_key", ped, Nh, "rep", role="other"), Linear("ple_value", ped, h, "rep", role="other"))
            misc += 3 * Nh + Nh * pk
            kw = dict(ple_rows=prow, ple_row_dim=ped // prow, ple_conv=pk, ple_dil=ng)
        if last:
            gf, mf = gres("f", combine=False)
            alin += gf
            misc += mf
        return Layer(alin, core, ffn, flin, misc, gres_n=N, gres_head=last, **kw)

    layers = [layer(i, types[i] if i < len(types) else "linear_attention", i == L - 1) for i in range(L)]
    mc = c.get("mtp") or {}
    mtp = []
    for j in range(c.get("mtp_num_hidden_layers") or 0):
        typ = (mc.get("layer_types") or ["full_attention"])[j]
        m_ = layer(L + j, typ)
        gf, mf = gres("f", combine=False)
        # MTP (SGLang qwen4_exp_mtp.py _fuse_residual_linear_shared): fc_embedding(norm(embed)) + fc_hidden applied
        # to each of the N normed streams (fc weights = mtp_extra 2·h·h), pre-fc norms (h + N·h), one QSA full-attention
        # layer without PLE, its own hyper_connection_mixer
        mtp.append(replace(m_, attn_linears=m_.attn_linears + gf, misc_params=m_.misc_params + mf + h + Nh,
                           gres_head=True))
    return layers, mtp, notes, "full"


def _build_dsv41_ref(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """DeepSeek-V4.1-Flash per the official inference/model.py + engram.py: every layer MQA-512 over its sliding
    window (+ attention sink); compress_ratios > 0 layers also attend to top-k entries of a compressed KV that only
    kv_source_layers compute (Compressor: ratio 1 plain projection, ratio > 1 fp32 softmax pooling, no APE) and
    store; index_source_layers run an indexer (index keys from wk on the source latent), the layers in between reuse
    its top-k; candidate_source_layer adds block pre-selection (select_candidate_blocks) used by later indexers;
    Engram n-gram tables (fp8 rows + e8m0 scales) at engram_layer_ids; mHC ×hc_mult with the final collapse by the
    last pre_mix (no head projection); DSpark block draft head (mtp.*)."""
    h = c["hidden_size"]
    L = c["num_hidden_layers"]
    H, hd = c["num_attention_heads"], c["head_dim"]
    ql, ol, og = c["q_lora_rank"], c["o_lora_rank"], c["o_groups"]
    rope = c.get("qk_rope_head_dim") or 64
    win = c.get("sliding_window") or 128
    ratios = _list(c.get("compress_ratios")) or [0] * L
    ih, idim, topk = c["index_n_heads"], c["index_head_dim"], c["index_topk"]
    kv_src = set(_list(c.get("kv_source_layer_ids")) or [])
    idx_src = set(_list(c.get("index_source_layer_ids")) or [])
    cand_src = c.get("candidate_source_layer_id", -1)
    hc, iters = c.get("hc_mult") or 1, c.get("hc_sinkhorn_iters") or 1
    E, k, de = c["n_routed_experts"], c["num_experts_per_tok"], c["moe_intermediate_size"]
    eng_ids = set(_list(c.get("engram_layer_ids")) or [])
    ecols = (c.get("engram_max_ngram_size", 1) - 1) * (c.get("engram_n_heads") or 0)
    edim = c.get("engram_head_dim") or 0
    hc_p = 2 * ((2 + hc) * hc * (hc * h + 1) + 3)
    kvb = 2.0       # reference caches are default-dtype (bf16) buffers; fp8 / fp4 act_quant is in-place simulation
    notes = [f"DeepSeek-V4.1 按官方 inference/model.py + engram.py 逐项建模：每层滑窗 {win} + attention sink；"
             f"压缩层另读 top-{topk} 压缩条目（keys = min(p, w) + min(top-k, ⌊p/r⌋)），压缩 KV 只在 {sorted(kv_src)} 层计算 / 存储，"
             f"索引器只在 {sorted(idx_src)} 层运行（其余层复用 top-k），第 {cand_src} 层候选块预选；Engram 查表层 "
             f"{sorted(eng_ids)}（每 token {ecols} 行 × {edim} fp8 + e8m0 scale）；mHC ×{hc}；DSpark 块草稿头"]

    def attn(i: int, r: int, backbone: bool = True):
        alin = [Linear("q_a", h, ql, "rep"), Linear("q_b", ql, H * hd, "col"), Linear("kv", h, hd, "rep"),
                Linear("o_a", H * hd, og * ol, "col", groups=og), Linear("o_b", og * ol, h, "row")]
        core = AttnCore("mla", n_q=H, n_kv=1, qk_dim=hd, v_dim=hd, kv_lora=hd, rope_dim=0, rope_n=rope,
                        window=win, sink=True, win_bytes=hd * kvb)
        misc = ql + hd + H
        if r and backbone:
            own = i in kv_src
            core = replace(core, dual=True, compress=r, topk=topk, cmp_bytes=hd * kvb, kv_owner=own,
                           cmp_coff=1 if own else 0, cmp_noape=True)
            if own:
                alin += [Linear("cmp_kv", h, hd, "rep")] + ([Linear("cmp_gate", h, hd, "rep")] if r > 1 else [])
                misc += hd
            if i in idx_src:
                alin += [Linear("idx_q", ql, ih * idim, "col"), Linear("idx_w", h, ih, "col")]
                core = replace(core, idx_heads=ih, idx_dim=idim, idx_split=True, idx_rope=rope, idx_owner=own,
                               idx_kbytes=idim * kvb,
                               cand="src" if i == cand_src else ("use" if 0 <= cand_src < i else ""),
                               cand_block=c.get("candidate_block_size") or 0,
                               cand_topk=c.get("candidate_topk_blocks") or 0)
                if own:
                    misc += hd * idim + idim         # wk (on compressed latents) + k_norm
        return alin, core, misc

    layers = []
    for i in range(L):
        r = ratios[i] if i < len(ratios) else 0
        alin, core, misc = attn(i, r)
        ec = 0
        if i in eng_ids:
            alin.append(Linear("engram_wkv", ecols * edim, h * (hc + 1), "rep"))
            misc += 2 * hc * h                       # q_weight, k_weight
            ec = ecols
        ffn, flin = _ffn_moe(h, E, k, de, c.get("n_shared_experts") or 0, de)
        misc += 2 * h + hc_p + 2 * E                 # norms, hc, gate bias + bias_vl
        layers.append(Layer(tuple(alin), core, ffn, flin, misc, hc=hc, hc_iters=iters, engram_cols=ec,
                            engram_dim=edim, hc_head="pre" if i == L - 1 else ""))
    mtp = []
    n_mtp = c.get("num_nextn_predict_layers") or 0
    Ed, kd = c.get("dspark_n_routed_experts") or E, c.get("dspark_num_experts_per_tok") or k
    for j in range(n_mtp):
        alin, core, misc = attn(L + j, 0, backbone=False)
        ffn, flin = _ffn_moe(h, Ed, kd, de, c.get("n_shared_experts") or 0, de)
        misc += 2 * h + hc_p + 2 * Ed
        mtp.append(Layer(tuple(alin), core, ffn, flin, misc, hc=hc, hc_iters=iters))
    return layers, mtp, notes, "full"


def _build_dsv4(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """DeepSeek-V4 family: MQA head_dim 512, grouped low-rank output, compressed
    sparse attention (CSA) with per-layer compress ratio, sliding window, hash
    routing on the first layers, hyper-connections.  「架构代理」."""
    h = c["hidden_size"]
    L = c["num_hidden_layers"]
    H, hd = c["num_attention_heads"], c["head_dim"]
    ql, ol, og = c["q_lora_rank"], c["o_lora_rank"], c["o_groups"]
    rope = c.get("qk_rope_head_dim") or 64
    win = c.get("sliding_window") or 128
    ratios = _list(c.get("compress_ratios")) or [1] * L
    ih, idim, topk = c.get("index_n_heads") or 0, c.get("index_head_dim") or 0, c.get("index_topk")
    hc = c.get("hc_mult") or 1
    E, k, de = c["n_routed_experts"], c["num_experts_per_tok"], c["moe_intermediate_size"]
    notes = ["「架构代理」DeepSeek-V4：压缩稀疏注意力按有效上下文 ctx/ratio 近似（+ 窗口，索引层 ≤ top-k）；"
             "超连接（×%d 残差流）只计参数、不计混合计算；哈希路由层按 top-k MoE 处理" % hc]
    layers = []
    for i in range(L):
        r = ratios[i] if i < len(ratios) else 1
        r = r if r and r > 1 else 1
        alin = [Linear("q_a", h, ql, "rep"), Linear("q_b", ql, H * hd, "col"), Linear("kv", h, hd, "rep"),
                Linear("o_a", H * hd, og * ol, "col", groups=og), Linear("o_b", og * ol, h, "row")]
        core = AttnCore("mla", n_q=H, n_kv=1, qk_dim=hd, v_dim=hd, kv_lora=hd - rope, rope_dim=rope,
                        window=win, compress=r)
        misc = ql + hd + H + 2 * (2 * hc + hc * hc) * hc * h // max(1, hc) // 1
        if r > 1:
            cw = 2 if r == 4 else 1      # overlapping compression for small ratios
            alin += [Linear("cmp_kv", h, cw * hd, "rep"), Linear("cmp_gate", h, cw * hd, "rep")]
            misc += cw * r * hd + hd
            if r == 4 and ih:
                alin += [Linear("idx_q", ql, ih * idim, "rep"), Linear("idx_cmp_kv", h, cw * idim, "rep"),
                         Linear("idx_cmp_gate", h, cw * idim, "rep"), Linear("idx_w", h, ih, "rep")]
                core = replace(core, idx_heads=ih, idx_dim=idim, topk=topk)
        ffn, flin = _ffn_moe(h, E, k, de, c.get("n_shared_experts") or 0, de)
        layers.append(Layer(tuple(alin), core, ffn, flin, misc + 2 * h + E))
    mtp = [replace(layers[-1], misc_params=layers[-1].misc_params + 2 * h * h + 3 * h)
           for _ in range(c.get("num_nextn_predict_layers") or 0)]
    return layers, mtp, notes, "proxy"


def _lfm2_dense_ff(c: dict) -> int:
    """Dense SwiGLU width from ``Lfm2MLP`` (transformers 4.57 ``modeling_lfm2.py``).

    ``block_auto_adjust_ff_dim`` is false on LFM2.5-2.6B, so ``intermediate_size`` is the
    GEMM width. When the flag is true the class uses ``int(2 * intermediate_size / 3)``,
    then ``block_ffn_dim_multiplier``, then rounds up to ``block_multiple_of``. MoE expert
    width stays ``moe_intermediate_size``.
    """
    inter = int(c["intermediate_size"])
    if not c.get("block_auto_adjust_ff_dim"):
        return inter
    inter = int(2 * inter / 3)
    mult = c.get("block_ffn_dim_multiplier")
    if mult is not None:
        inter = int(float(mult) * inter)
        multiple = int(c.get("block_multiple_of") or 0)
        if multiple <= 0:
            raise ValueError("lfm2 block_multiple_of must be positive when block_ffn_dim_multiplier is set")
        inter = multiple * ((inter + multiple - 1) // multiple)
    if inter <= 0:
        raise ValueError("lfm2 adjusted intermediate_size must be positive")
    return inter


def _build_lfm2(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """0.114: Liquid LFM2 / LFM2-MoE (transformers Lfm2Moe): ``layer_types`` conv | full_attention.
    conv (Lfm2ShortConv): in_proj h → 3h (B, C, x), B·x, depthwise causal conv (kernel conv_L_cache) over h channels,
    C·(conv), out_proj h → h.  full_attention: GQA with per-head q / k RMSNorm.  FFN: SwiGLU (w1 / w3 / w2); the first
    ``num_dense_layers`` dense (intermediate_size, or the Lfm2MLP auto-adjust), the rest MoE (num_experts, top-k,
    moe_intermediate_size, sigmoid router + per-expert bias, no shared expert).  Norms: operator_norm + ffn_norm per
    layer; final embedding_norm.  0.389: ``tie_embedding`` is the Lfm2Config alias of ``tie_word_embeddings``."""
    h = c["hidden_size"]
    H = c["num_attention_heads"]
    KV = c.get("num_key_value_heads") or H
    hd = c.get("head_dim") or h // H
    L = c["num_hidden_layers"]
    kern = c.get("conv_L_cache") or 3
    types = _list(c.get("layer_types")) or ["full_attention"] * L
    if len(types) != L or any(t not in ("conv", "full_attention") for t in types):
        raise ValueError("lfm2 requires layer_types of length num_hidden_layers (conv | full_attention)")
    E = c.get("num_experts") or 0
    n_dense = c.get("num_dense_layers") or 0
    dense_ff = _lfm2_dense_ff(c)
    layers = []
    for i in range(L):
        if types[i] == "conv":
            alin = (Linear("conv_in", h, 3 * h, "col"), Linear("conv_out", h, h, "row"))
            core = AttnCore("linear", n_state_heads=0, state_dk=0, state_dv=0, conv_channels=h, conv_kernel=kern,
                            lin="conv")
            misc = h * kern + (h if c.get("conv_bias") else 0)
        else:
            alin, core = _gqa(h, H, KV, hd)
            misc = 2 * hd
        if E and i >= n_dense:
            ffn, flin = _ffn_moe(h, E, c["num_experts_per_tok"], c["moe_intermediate_size"])
            misc += E if c.get("use_expert_bias") else 0
        else:
            ffn, flin = _ffn_dense(h, dense_ff)
        layers.append(Layer(alin, core, ffn, flin, misc + 2 * h))
    n_conv = sum(1 for t in types[:L] if t == "conv")
    notes = [f"LFM2 混合：{n_conv}/{L} 层门控短卷积（in_proj 3h、深度卷积 kernel {kern}、两次逐元素门控），"
             f"{L - n_conv} 层 GQA 全注意力（q/k RMSNorm）"]
    if c.get("block_auto_adjust_ff_dim"):
        notes.append(f"block_auto_adjust_ff_dim：稠密 SwiGLU 宽 {dense_ff}"
                     f"（int(2×{c['intermediate_size']}/3) 后再按 block_multiple_of 取整），不是 config 的 intermediate_size")
    return layers, [], notes, "full"


def _build_mimo_v2(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """0.144 Xiaomi MiMo-V2.x (modeling_mimo_v2.py): ``hybrid_layer_pattern`` 0 = full / 1 = sliding-window GQA with
    per-type geometry (full: num_key_value_heads, head_dim / v_head_dim; SWA: swa_num_key_value_heads, swa_head_dim /
    swa_v_head_dim, window sliding_window) and a learned attention-sink logit per head on the SWA layers
    (add_swa_attention_sink_bias); fused qkv_proj (q / k / v parameter split identical); ``moe_layer_freq`` 0 = dense
    SwiGLU (intermediate_size), 1 = sigmoid noaux_tc MoE (n_routed_experts, top-k, no shared expert, score-correction
    bias); MTP layers = SWA attention + dense FFN + enorm / hnorm / eh_proj."""
    h = c["hidden_size"]
    L = c["num_hidden_layers"]
    H = c["num_attention_heads"]
    pat = _list(c.get("hybrid_layer_pattern")) or [0] * L
    moe = _list(c.get("moe_layer_freq")) or [1] * L
    win = c.get("sliding_window") or c.get("sliding_window_size")
    E, k, de = c["n_routed_experts"], c["num_experts_per_tok"], c["moe_intermediate_size"]
    swa_sink = bool(c.get("add_swa_attention_sink_bias"))
    full_sink = bool(c.get("add_full_attention_sink_bias"))

    def attn(swa: bool):
        if swa:
            Hs = c.get("swa_num_attention_heads") or H
            alin, core = _gqa(h, Hs, c["swa_num_key_value_heads"], c["swa_head_dim"], vd=c["swa_v_head_dim"],
                              window=win)
            return alin, (replace(core, sink=True), Hs) if swa_sink else (core, 0)
        alin, core = _gqa(h, H, c["num_key_value_heads"], c["head_dim"], vd=c.get("v_head_dim") or c["head_dim"])
        return alin, (replace(core, sink=True), H) if full_sink else (core, 0)

    layers = []
    for i in range(L):
        alin, (core, sinks) = attn(bool(pat[i]))
        if moe[i]:
            ffn, flin = _ffn_moe(h, E, k, de)
            misc = E                                   # e_score_correction_bias
        else:
            ffn, flin = _ffn_dense(h, c["intermediate_size"])
            misc = 0
        layers.append(Layer(alin, core, ffn, flin, misc + sinks + 2 * h))
    mtp = []
    n_mtp = c.get("num_nextn_predict_layers")
    if n_mtp is None:
        # 0.145 MiMo-V2.6-Pro: config omits num_nextn_predict_layers but ships model.mtp.layers.{0,1,2} → offer up to
        # 8 MTP templates; from_release trims them to the released MTP parameter count
        n_mtp = 8
    for _ in range(n_mtp):
        alin, (core, sinks) = attn(True)
        ffn, flin = _ffn_dense(h, c["intermediate_size"])
        mtp.append(Layer(alin, core, ffn, flin, sinks + 2 * h + 3 * h))   # + enorm / hnorm / final_layernorm
    n_swa = sum(1 for x in pat[:L] if x)
    notes = [f"MiMo-V2 混合：{n_swa}/{L} 层滑窗 {win}（KV {c['swa_num_key_value_heads']} 头，attention sink），"
             f"{L - n_swa} 层全注意力（KV {c['num_key_value_heads']} 头）；qk {c['head_dim']} / v {c.get('v_head_dim')}；"
             f"MoE {E} 专家 top-{k}（无共享专家），{sum(1 for x in moe[:L] if not x)} 层稠密"]
    return layers, mtp, notes, "full"


def _build_mimo_v2_flash(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """0.372 Xiaomi MiMo-V2-Flash (modeling_mimo_v2_flash.py).

    The layer math is the MiMo-V2 hybrid stack: ``hybrid_layer_pattern`` 0 = full / 1 = SWA,
    separate q/k/v (same parameter count as a fused qkv), SWA attention sinks, sigmoid
    noaux_tc MoE with no shared expert. ``partial_rotary_factor`` rotates a prefix of q/k and
    the cache still stores the full head, so the GQA rope vector stays the historical
    query-head half-head bill (MiMo-V2.6 ships the same factor). ``attention_value_scale`` is a
    scalar on V. ``attention_chunk_size`` is not read by ``MiMoV2Attention``.
    """
    layers, mtp, notes, cov = _build_mimo_v2(c)
    hd = int(c["head_dim"])
    factor = float(c.get("partial_rotary_factor") or 1.0)
    rot = int(hd * factor)
    notes = list(notes) + [
        f"MiMo-V2-Flash：RoPE 作用在 q/k 的前 {rot}/{hd} 维（partial_rotary_factor {factor:g}），"
        "缓存的 K 仍是整段 head；周期沿用 GQA 的查询头半 head 向量。"
        "V 乘 attention_value_scale。"
        "注意力按滑窗或全上下文计算（MiMoV2Attention 不读取 attention_chunk_size）。"
        "路由权重与 e_score_correction_bias 都是 fp32。",
    ]
    return layers, mtp, notes, cov


def _step3p5_moe_ids(raw) -> set[int]:
    """``moe_layers_enum`` is a comma-separated string. ``_list`` JSON-parses strings and would drop every MoE layer."""
    if raw is None:
        return set()
    if isinstance(raw, str):
        return {int(x) for x in raw.split(",") if x.strip()}
    return {int(x) for x in raw}


def _build_step3p5(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """0.374 StepFun Step-3.5-Flash (modeling_step3p5.py Step3p5DecoderLayer + safetensors headers).

    Not Step-5 DSA. Sliding layers take Q heads / KV groups / head_dim from ``attention_other_setting``;
    full layers use the top-level counts. ``num_attention_groups`` is the KV head count. Each layer has a
    bias-free head-wise gate ``g_proj`` (h → n_heads) and q/k RMSNorm(head_dim) — the class always builds
    those norms. ``partial_rotary_factors[i]`` sets ``rope_dim = int(head_dim * factor)`` (full 0.5, sliding
    1.0). Do not set ``rope_n``: that is the Qwen4Exp path (extra q/k norm FLOPs and a rope on both q and k).
    The KV cache still stores the full head. ``use_rope_layers`` empty leaves RoPE on.

    MoE layers are the comma list ``moe_layers_enum``: sigmoid router, fp32 ``router_bias`` (a Parameter, not
    ``.router.`` so the header role is ``other``), no group limit, one ungated shared MLP of ``share_expert_dim``.
    ``gate.weight`` is bf16 as released; ``need_fp32_gate`` only casts the matmul. Other layers are dense SwiGLU
    of ``intermediate_size``. ``zero_centered`` adds 1 to the RMSNorm weight in the forward (same parameter).
    ``swiglu_limits`` / yarn / ``moe_router_scaling_factor`` do not change GEMM or KV shapes.

    MTP (``num_nextn_predict_layers``) continues ``layer_types``: sliding attention, dense FFN, enorm / hnorm /
    eh_proj, and a private ``shared_head`` (norm + vocab projection). The published class does not construct
    those modules; the checkpoint stores them and the length-48 schedules cover indices L .. L+n_mtp-1.
    """
    h = c["hidden_size"]
    L = c["num_hidden_layers"]
    H0 = c["num_attention_heads"]
    KV0 = c["num_attention_groups"]
    hd0 = c["head_dim"]
    types = _list(c.get("layer_types")) or []
    rope_f = _list(c.get("partial_rotary_factors")) or []
    other = c.get("attention_other_setting") or {}
    win = c.get("sliding_window")
    moe_ids = _step3p5_moe_ids(c.get("moe_layers_enum"))
    E = c["moe_num_experts"]
    k = c["moe_top_k"]
    de = c["moe_intermediate_size"]
    d_sh = c.get("share_expert_dim") or 0
    n_mtp = c.get("num_nextn_predict_layers") or 0
    gate = bool(c.get("use_head_wise_attn_gate"))
    bias = bool(c.get("use_moe_router_bias"))

    def one(i: int, mtp: bool) -> Layer:
        swa = i < len(types) and types[i] == "sliding_attention"
        if swa:
            H = int(other.get("num_attention_heads") or H0)
            KV = int(other.get("num_attention_groups") or KV0)
            hd = int(other.get("head_dim") or hd0)
        else:
            H, KV, hd = H0, KV0, hd0
        alin, core = _gqa(h, H, KV, hd, window=win if swa else None)
        if gate:
            alin = alin + (Linear("g_head", h, H, "col"),)
        factor = float(rope_f[i]) if i < len(rope_f) else 1.0
        core = replace(core, rope_dim=int(hd * factor))
        misc = 2 * hd + 2 * h                      # q_norm, k_norm, input + post RMSNorm
        if (not mtp) and i in moe_ids:
            ffn, flin = _ffn_moe(h, E, k, de, 1 if d_sh else 0, d_sh, shared_gate=False)
            if bias:
                misc += E                          # fp32 router_bias, stored in layer misc
        else:
            ffn, flin = _ffn_dense(h, c["intermediate_size"])
        if mtp:
            misc += 3 * h                          # enorm, hnorm, shared_head.norm
        return Layer(alin, core, ffn, flin, misc)

    layers = [one(i, False) for i in range(L)]
    mtp = [one(L + j, True) for j in range(n_mtp)]
    n_sw = sum(1 for ly in layers if ly.core.window is not None)
    n_moe = sum(1 for ly in layers if ly.ffn.kind == "moe")
    notes = [
        f"Step-3.5-Flash 按 modeling_step3p5.py 建模：{L - n_sw} 层全注意力（{H0}/{KV0}，head {hd0}，"
        f"partial RoPE {int(hd0 * (rope_f[0] if rope_f else 0.5))}）与 {n_sw} 层滑窗 {win}"
        f"（{int(other.get('num_attention_heads') or H0)}/"
        f"{int(other.get('num_attention_groups') or KV0)}，旋转整段 head）。"
        "逐头 sigmoid 门 g_proj。q/k RMSNorm(head_dim)。",
        f"MoE {n_moe} 层：sigmoid 路由 {E} 专家 top-{k}，专家宽 {de}，无分组；"
        f"1 个不带门的共享专家（宽 {d_sh}）。"
        "router_bias 是 fp32，落在层 misc，记忆体计划仍按每参数 2 字节。"
        "gate.weight 按发布是 bf16；need_fp32_gate 只把那一次 matmul 转成 fp32，不另开计算格式。",
        "zero_centered 的 weight+1、swiglu clamp、YaRN 与 routed_scaling_factor 不改变 GEMM 形状，也不改变 KV 宽度，不计额外周期。",
        f"MTP ×{n_mtp}：发布类不实例化这些层，权重里有。每层是滑窗注意力 + 稠密 FFN + eh_proj，"
        "另有私有 shared_head（norm + 词表投影）。",
    ]
    return layers, mtp, notes, "full"


def _build_step5(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """0.144 StepFun Step-5 (official inference/model.py + kernel.py).  ``layer_types`` sliding_attention (window
    sliding_window) | full_attention; every layer GQA (num_attention_groups KV heads, head_dim) with q / k RMSNorm and
    a head-wise sigmoid output gate (g_proj h → heads).  Full layers are DSA layers (sparse_config): an indexer
    (sparse_indexer_num_heads × proxy_dim q, single k and z projections, fp32 head weights w) scores the cached
    summary of every completed region of region_block_size tokens (z-gated compression, fp8), each KV provider group
    keeps its top-k regions + the current region, and attention reads those tokens of the full K/V.  MoE layers
    (moe_layer_list): sigmoid router with bias, moe_num_experts top-k, one ungated shared expert (share_expert_dim);
    other layers dense SwiGLU.  MTP layers (num_nextn_predict_layers) follow the main stack (layer_types continue)
    with dense FFN, enorm / hnorm / eh_proj and their own shared_head (norm + vocab projection)."""
    h = c["hidden_size"]
    L = c["num_hidden_layers"]
    H, KV, hd = c["num_attention_heads"], c["num_attention_groups"], c["head_dim"]
    types = _list(c.get("layer_types")) or []
    win = c.get("sliding_window")
    sp = c.get("sparse_config") or {}
    sparse_on = bool(sp.get("enabled")) and "full_attention" in (sp.get("apply_to_layer_types") or [])
    ih, pd, rb, topk = (sp.get("sparse_indexer_num_heads") or 0, sp.get("proxy_dim") or 0,
                        sp.get("region_block_size") or 1, sp.get("topk") or 0)
    moe_ids = set(_list(c.get("moe_layer_list")) or [])
    E, k, de = c["moe_num_experts"], c.get("moe_top_k") or c["num_experts_per_tok"], c["moe_intermediate_size"]
    d_sh = c.get("share_expert_dim") or 0
    rope_f = _list(c.get("partial_rotary_factors")) or []

    def layer(i: int, mtp: bool = False) -> Layer:
        typ = types[i] if i < len(types) else "full_attention"
        swa = typ == "sliding_attention"
        alin, core = _gqa(h, H, KV, hd, window=win if swa else None)
        alin = alin + (Linear("g_head", h, H, "col"),)
        misc = 2 * hd + 2 * h
        if not swa and sparse_on:
            alin = alin + (Linear("idx_q", h, ih * pd, "rep"), Linear("idx_k", h, pd, "rep"),
                           Linear("idx_z", h, pd, "rep"), Linear("idx_w", h, ih, "rep"))
            rope = sp.get("sparse_indexer_rope_dim") or 0
            core = replace(core, qsa=rb, topk=topk * rb, idx_heads=ih, idx_dim=pd, idx_rope=rope, idx_sum=True,
                           rope_n=int(hd * (rope_f[i] if i < len(rope_f) else 1)))
            misc += pd + 2 * pd + H                    # q_norm, k_norm (LayerNorm w + b), ssmax_s
        if not mtp and i in moe_ids:
            ffn, flin = _ffn_moe(h, E, k, de, 1 if d_sh else 0, d_sh)
            misc += E                                  # router_bias
        else:
            ffn, flin = _ffn_dense(h, c["intermediate_size"])
        return Layer(alin, core, ffn, flin, misc + (3 * h if mtp else 0))

    layers = [layer(i) for i in range(L)]
    mtp = [layer(L + j, True) for j in range(c.get("num_nextn_predict_layers") or 0)]
    n_sp = sum(1 for l in layers if l.core.idx_heads)
    notes = [f"Step-5 按官方 inference/model.py + kernel.py 建模：{L - n_sp}/{L} 层滑窗 {win}，{n_sp} 层 DSA 稀疏注意力"
             f"（索引器 {ih}×{pd}，每 {rb} token 区域压缩为缓存 fp8 摘要，每 KV 组 top-{topk} 区域 + 当前区域 = "
             f"≤{topk * rb + rb} 个 token）；逐头 sigmoid 输出门；MoE {E} 专家 top-{k} + 1 共享专家（{len(moe_ids)} 层）"]
    return layers, mtp, notes, "full"


def _build_hyena(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """0.170 StripedHyena 2 / Evo 2 (ArcInstitute/evo2 configs + Zymrael/vortex model.py, engine.py).
    Every block: pre-norm mixer + post-norm gated MLP (l1, l2 → l3, inner_mlp_size).  Mixer by layer index:
      hcs / hcm / hcl  hyena operator: projections h → 3h, short FIR (short_filter_length) over the 3h channels,
                       split x2, x1, v; inner filter on x1·v (se: hcs_filter_length taps, mr: hcm_filter_length taps,
                       lr: modal long filter of state_size poles), gate by x2 (+ D skip on mr / lr), out_filter_dense
      attn             MHA (num_attention_heads, Wqkv 3h, out_proj with bias), RoPE; the only layers with a KV cache."""
    h = c["hidden_size"]
    H = c["num_attention_heads"]
    L = c["num_hidden_layers"]
    sf = c.get("short_filter_length") or 3
    kind = {}
    for k, t in (("hcs_layer_idxs", "se"), ("hcm_layer_idxs", "mr"), ("hcl_layer_idxs", "lr"), ("attn_layer_idxs", "attn")):
        for i in c.get(k) or []:
            kind[i] = t
    layers = []
    for i in range(L):
        t = kind.get(i, "attn")
        if t == "attn":
            alin, core = _gqa(h, H, H, h // H)
            misc = h                                   # out_proj bias
        else:
            alin = (Linear("hyena_in", h, 3 * h, "col"), Linear("hyena_out", h, h, "row"))
            ln = {"se": c.get("hcs_filter_length") or 7, "mr": c.get("hcm_filter_length") or 128}.get(t, 0)
            grp = {"se": c.get("hcs_filter_groups"), "mr": c.get("hcm_filter_groups"), "lr": c.get("hcl_filter_groups")}[t] or h
            modes = (c.get("state_size") or 16) if t == "lr" else 0
            core = AttnCore("linear", n_state_heads=h if t == "lr" else 0, state_dk=1 if t == "lr" else 0,
                            state_dv=modes, conv_channels=3 * h, conv_kernel=sf, lin="hyena", hy=t, hy_len=ln,
                            hy_groups=grp, hy_modes=modes, hy_prefill=str(c.get("prefill_style") or "fft") if t == "lr" else "")
            # short FIR taps, out_filter_dense bias; se: filter.h [groups, 1, taps]; mr: + D; lr: D, log_poles, residues
            misc = 3 * h * sf + h + {"se": grp * ln, "mr": grp * ln + h, "lr": h + 2 * h * modes}[t]
        ffn, flin = _ffn_dense(h, c["intermediate_size"])
        layers.append(Layer(alin, core, ffn, flin, misc + 2 * h))
    n = {t: sum(1 for i in range(L) if kind.get(i, "attn") == t) for t in ("se", "mr", "lr", "attn")}
    notes = [f"StripedHyena 2：{n['se']} 层 Hyena-SE（{c.get('hcs_filter_length')} 抽头 FIR）、{n['mr']} 层 Hyena-MR"
             f"（{c.get('hcm_filter_length')} 抽头，预填充走 FFT 卷积）、{n['lr']} 层 Hyena-LR（{c.get('state_size')} 模态"
             f"长滤波，FFT 卷积 / 解码递推）、{n['attn']} 层 MHA（唯一有 KV cache 的层）"]
    return layers, [], notes, "full"


def _build_muse_spark(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """0.174 Meta Muse Spark Open (vLLM PR #61879 vllm/models/muse_spark/nvidia/model.py @ 6cc692b5 + checkpoint
    config.json).  Every layer: GQA (num_attention_heads × head_dim, num_key_value_heads) with a fused per-head
    sigmoid output gate (MuseSpark1xQKVOGateParallelLinear: q / k / v / o_gate rows, 2·(H + KV)·head_dim),
    weightless q / k RMSNorm; ``layer_types`` sliding_attention (window sliding_window) | full_attention — cycle of 4
    anchored to the end (full at i % 4 == 1 for 62 layers), full layers NoPE (layer_rope_theta 0).  Every layer MoE:
    fp32 router hidden → num_local_experts (sigmoid, expert_bias), pooler hidden → moe_hidden_size, top-k SwiGLU experts
    at moe_hidden_size × moe_intermediate_size, unpooler back (blog: "latent MoE"); no shared expert
    (num_shared_experts 0).  Sandwich norm: x + post_norm(F(norm(x))) per block; gated residual (scalar gates);
    final logit soft-cap (tanh, 20) — vector ops."""
    h, H, KV, hd, L = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["num_hidden_layers"]
    E, k = c["num_local_experts"], c["num_experts_per_tok"]
    lat, de = c["moe_hidden_size"], c["moe_intermediate_size"]
    types = _list(c.get("layer_types")) or ["full_attention"] * L
    win = c.get("sliding_window")
    layers = []
    for i in range(L):
        w = win if types[i] == "sliding_attention" else None
        alin, core = _gqa(h, H, KV, hd, out_gate=True, window=w)
        lin = [Linear("router", h, E, "rep"), Linear("latent_up", h, lat, "rep"), Linear("latent_down", lat, h, "rep")]
        ffn = Ffn("moe", gated=True, n_experts=E, top_k=k, d_expert=de, latent=lat)
        layers.append(Layer(alin, core, ffn, tuple(lin), 4 * h + E))   # 4 sandwich norms + expert_bias
    nf = sum(1 for t in types[:L] if t == "full_attention")
    notes = [f"Muse Spark Open：{L} 层全 MoE（{E} 专家 top-{k}，latent {h}→{lat}，专家 {lat}×{de}，无共享专家）；"
             f"注意力 GQA {H}/{KV}×{hd} + 输出门；{L - nf} 层 SWA（窗口 {win}）+ {nf} 层全注意力（NoPE）；"
             "三明治 norm、门控残差、logit soft-cap 计入向量开销",
             "视觉塔（NaViT，32×32 块窗口 + 13 全注意力层）与音频编码器（24 kHz，1920 样本 / token）已计前向；"
             "图像 / 视频缩放按 transformers MuseGlimmerImageProcessor / MuseGlimmerVideoProcessor smart_resize"
             "（0.255，「来源」）"]
    return layers, [], notes, "full"


def _attach_muse_dspark(spec: ModelSpec, d: dict) -> ModelSpec:
    """0.175 Inferact/Muse-Spark-Open-DSpark (vLLM PR #61879 dspark.py MuseSparkDSparkModel, DFlash-style): fc over the
    concatenated hidden states of target_layer_ids → hidden, num_hidden_layers dense decoder layers (GQA + o-gate,
    sliding window, SwiGLU intermediate_size), own lm_head, Markov head (rank markov_rank: w1 embed + w2 head), confidence
    head on [hidden, rank]; block_size draft positions per pass; token embedding shared with the target."""
    c = d["config"]
    h, H, KV, hd = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    nl, r, V = c["num_hidden_layers"], c["markov_rank"], c.get("draft_vocab_size") or c["vocab_size"]
    win = c.get("sliding_window") if c.get("use_sliding_window") else None
    mtp = []
    for _ in range(nl):
        alin, core = _gqa(h, H, KV, hd, out_gate=bool(c.get("use_attn_o_gate")), window=win)
        ffn, flin = _ffn_dense(h, c["intermediate_size"])
        mtp.append(Layer(alin, core, ffn, flin, 5 * h))          # pre_qkv / pre_ffn / post_ffn norms + 2 residual gates
    nt = len(c["target_layer_ids"])
    extra = nt * h * h + V * h + 2 * V * r + (h + r + 1) + 2 * h   # fc, lm_head, markov w1 / w2, confidence, norms
    return replace(spec, mtp_layers=tuple(mtp), mtp_extra_params=extra // nl, draft="dspark",
                   draft_block=c["block_size"], draft_targets=nt, draft_markov=r, draft_kv=2 * KV * hd,
                   notes=spec.notes + (f"DSpark 草稿模型（{d['repo']}，bf16 {d['params'] / 1e9:.2f}B，独立仓库）：{nl} 层稠密，"
                                       f"块 {c['block_size']}；目标层 hidden 拼接（模板按均值 + fc 计，同 GEMM 形状）",))


def _attach_ready_dspark(spec: ModelSpec, d: dict) -> ModelSpec:
    """0.221 StartLux ReaDy-Qwen3.8-27B: a DSpark draft (SGLang native DSparkDraftModel; config.json + safetensors header)
    for a separate Qwen3.8-27B target.  fc over the concatenated hidden states of the 5 target_layer_ids → hidden_norm,
    num_hidden_layers dense GQA layers (q/k norm, full attention — no sliding window), final norm, Markov head (rank
    markov_rank: w1 + w2 over the vocab), confidence head on [hidden + rank]; NO own lm_head / embedding (the release has
    neither tensor → shared with the target).  Plus the recurrent-state reader (reader_config.json, MODEL_SPEC.txt): per
    proposal it reads the committed fp32 state [layers, heads, 128, 128] of target layers 30 / 62, RMS-normalises it,
    maps the value axis 128 → rank, reads it with a per-position rank query lifted to the key axis, appends per-head log
    state magnitude (heads × (rank + 1) = 864 features), then down (→ bottleneck) / up (→ hidden) SiLU residual at draft
    layer 4."""
    c, rc = d["config"], d.get("reader") or {}
    h, H, KV, hd = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    nl, r, V = c["num_hidden_layers"], c["markov_rank"], c.get("draft_vocab_size") or c["vocab_size"]
    win = c.get("sliding_window") if c.get("use_sliding_window") else None
    mtp = []
    for _ in range(nl):
        alin, core = _gqa(h, H, KV, hd, out_gate=bool(c.get("use_attn_o_gate")), window=win)
        ffn, flin = _ffn_dense(h, c["intermediate_size"])
        mtp.append(Layer(alin, core, ffn, flin, 2 * h + 2 * hd))     # input / post-attention norms + q / k norms
    nt = len(c.get("target_layer_ids") or c["dflash_config"]["target_layer_ids"])   # 0.294 LFM2.5 DSpark: dflash_config only
    extra = (nt * h * h + 2 * h + (V * h if d.get("own_lm_head") else 0) + 2 * V * r + (h + r + 1)
             + int(d.get("params_reader") or 0))                    # fc, hidden_norm + norm, markov w1/w2, confidence, reader
    rd = ((rc["rank"], len(rc["state_layers"]), rc["heads"], rc["key_dim"], rc["value_dim"], rc["bottleneck"],
           rc["draft_layer"]) if rc else ())
    acc = d.get("acceptance") or {}
    return replace(spec, mtp_layers=tuple(mtp), mtp_extra_params=extra // nl, draft="dspark",
                   draft_block=c["block_size"], draft_targets=nt, draft_markov=r, draft_kv=2 * KV * hd, draft_reader=rd,
                   spec_published=(float(acc["tau"]), int(acc["block"])) if acc else (),
                   notes=spec.notes + (f"DSpark 草稿模型（{d['repo']}，bf16 {d['params'] / 1e9:.2f}B，独立仓库，共享目标 "
                                       f"embedding / lm_head）：{nl} 层稠密全注意力，块 {c['block_size']}；目标层 hidden 拼接 + fc"
                                       + (f"；ReaDy 状态读取器（目标层 {rc['state_layers']} 的 fp32 循环状态，秩 {rc['rank']}，"
                                          f"注入草稿层 {rc['draft_layer']}）" if rc else "")
                                       + (f"；公开接受长度 τ {acc['tau']}（块 {acc['block']}，温度 0）→ i.i.d. 接受率 "
                                          f"{spec_accept_from_tau(acc['tau'], acc['block']):.3f}「推导」，spec_accept 默认仍为「假设」杠杆"
                                          if acc else ""),))


def _attach_ubtree(spec: ModelSpec, d: dict) -> ModelSpec:
    """0.311 UBTree draft (inclusionAI/Ling-3.0-tiny-UBTree: config.json ubtree_config / dflash_config + the safetensors
    header 「来源」; model card + arXiv 2609.39972 for the method; the SGLang serving code is not released → the selector
    and tree-build kernels are 「假设」 from the config).  Backbone = a DFlash anchor-row block draft: fc over the
    concatenated hidden states of the 5 target_layer_ids → hidden_norm, num_hidden_layers Qwen3 layers (GQA + q / k
    norm, SwiGLU, full attention) over the block (anchor + block − 1 mask rows; DFlash block attention non-causal
    「假设」), norm; no embedding / lm_head (the target's).  Tail: target lm_head on the draft rows → top-K candidates
    (candidate_k) per row; selector: hidden_projection h → rank per row (per-depth affine scales); tied token
    factorizer = predecessor / successor MLPs (h → width → rank) on the target-embedding rows of the K candidates
    (computed on the gathered rows 「假设」, no load-time table); bigram edges K × K × rank between adjacent rows +
    unary logits; pruned tree of ``tree_budget`` nodes (default; the card's benchmark used 49) → the target verifies
    1 + nodes tokens."""
    c = d["config"]
    dc = c.get("dflash_config") or {}
    ub = c["ubtree_config"]
    h, H, KV, hd = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    nl = c["num_hidden_layers"]
    K, R, F = int(ub["candidate_k"]), int(ub["rank"]), int(ub["factorizer_hidden_width"])
    D, budget = int(ub["depth_condition_count"]), int(ub["tree_budget"])
    mtp = []
    for _ in range(nl):
        alin, core = _gqa(h, H, KV, hd)
        ffn, flin = _ffn_dense(h, c["intermediate_size"])
        mtp.append(Layer(alin, core, ffn, flin, 2 * h + 2 * hd))     # input / post-attention norms + q / k norms
    nt = len(dc.get("target_layer_ids") or c["target_layer_ids"])
    te = int(ub.get("target_embedding_dim") or h)
    # fc, hidden_norm + norm, hidden_projection, predecessor + successor MLPs, unary / correction depth scales
    extra = nt * h * h + 2 * h + R * h + 2 * (F * te + R * F) + 2 * D
    acc = d.get("acceptance") or {}
    left = extra - (extra // nl) * nl
    B = int(dc.get("block_size") or c.get("block_size") or 16)
    nodes = int(acc.get("tree_nodes") or budget)
    return replace(spec, mtp_layers=tuple(mtp), mtp_extra_params=extra // nl, draft="dspark", draft_anchor=True,
                   draft_block=B, draft_targets=nt, draft_markov=0, draft_kv=2 * KV * hd,
                   draft_ubtree=(K, R, F, D, budget, nodes), draft_cold_params=left,
                   spec_published=(float(acc["tau"]), int(acc["block"])) if acc else (),
                   notes=spec.notes + (f"UBTree 草稿模型（{d['repo']}，bf16 {d['params'] / 1e6:.0f}M，独立仓库，共享目标 "
                                       f"embedding / lm_head）：DFlash 块草稿 {nl} 层（锚行 + {B - 1} 个草稿，块内非因果「假设」），"
                                       f"目标层 {dc.get('target_layer_ids')} hidden 拼接 + fc；尾部每行 top-{K} 候选 + 二元选择器"
                                       f"（秩 {R}，目标 embedding 上的 {F} 宽 MLP 因子化，相邻行 {K}×{K} 边分）→ 剪枝树"
                                       f"（预算 {budget} 节点；公开基准用 {nodes} 节点）→ 目标验证 1 + 节点数个 token；"
                                       "SGLang 实现未发布，选择器 / 建树算子按 config「假设」"
                                       + (f"；公开接受长度 τ {acc['tau']}（{acc['block']} 个草稿深度、{nodes} 节点树，温度 0）"
                                          "→ 等效链接受率「推导」；serving.spec_accept_src = published 按发布 τ 取接受率，"
                                          "默认仍为「假设」杠杆" if acc else ""),))


def _build_beam(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """0.254 Reflection Beam-1.0 (BeamForCausalLM; transformers main models/beam/modular_beam.py + vLLM main
    vllm/models/beam/nvidia/model.py, 「来源」).  Every layer: GQA (num_attention_heads / num_key_value_heads × head_dim)
    with q / k RMSNorm and a per-head sigmoid output gate (gate_proj h → H·head_dim, fused here into q as out_gate, +
    gate_bias [H]); ``layer_types`` sliding_attention (sliding_window, RoPE) | full_attention (every 6th + the last;
    NoPE with a position-dependent query scale (1 + pos/2048)^0.1 — a vector op).  ``mlp_layer_types`` dense (SwiGLU,
    intermediate_size) | sparse (num_experts SwiGLU experts × moe_intermediate_size, top-k, fp32 sigmoid router with
    e_score_correction_bias and n_group / topk_group group-limited selection, routed_scaling_factor after the combine;
    no shared expert).  Sandwich norms (input, post_attention, pre_feedforward, post_feedforward), depth-scaled residual
    adds (scalars), final logit soft-cap 30 (tanh, vector)."""
    h, H, KV, hd, L = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["num_hidden_layers"]
    E, k, de = c["num_experts"], c["num_experts_per_tok"], c["moe_intermediate_size"]
    types = _list(c.get("layer_types")) or ["full_attention"] * L
    mtypes = _list(c.get("mlp_layer_types")) or (["dense"] * 3 + ["sparse"] * (L - 3))
    win = c.get("sliding_window")
    layers = []
    for i in range(L):
        alin, core = _gqa(h, H, KV, hd, out_gate=True, window=win if types[i] == "sliding_attention" else None)
        misc = 4 * h + 2 * hd + H                                  # 4 sandwich norms + q / k norms + gate_bias
        if mtypes[i] == "sparse":
            ffn, flin = _ffn_moe(h, E, k, de, 0, 0)
            misc += E                                              # e_score_correction_bias (fp32)
        else:
            ffn, flin = _ffn_dense(h, c["intermediate_size"])
        layers.append(Layer(alin, core, ffn, flin, misc))
    nf = sum(1 for t in types[:L] if t == "full_attention")
    nd = sum(1 for t in mtypes[:L] if t == "dense")
    notes = [f"Beam：{L} 层（{nd} 层稠密 SwiGLU + {L - nd} 层 MoE，{E} 专家 top-{k}（分组 {c.get('n_group')} 选 "
             f"{c.get('topk_group')}，sigmoid + 校正偏置，fp32 router），无共享专家）；注意力 GQA {H}/{KV}×{hd} + 逐头 sigmoid "
             f"输出门；{L - nf} 层 SWA（窗口 {win}，RoPE）+ {nf} 层全注意力（NoPE，查询按位置缩放）；三明治 norm、深度缩放残差、"
             f"logit soft-cap {c.get('final_logit_softcapping')} 计入向量开销"]
    return layers, [], notes, "full"


def _attach_dspine(spec: ModelSpec, d: dict) -> ModelSpec:
    """0.252 tencent DSpine (Qwen3-8B-DSpine-b16; github tencent/dspine sglang_dspine, 「来源」): an SGLang DFlash draft —
    fc over the concatenated hidden states of the 5 target_layer_ids → hidden_norm, num_hidden_layers Qwen3 decoder
    layers (GQA + q/k norm, SwiGLU, full attention, CAUSAL inside the block: draft_block_causal), final norm, NO embedding
    / lm_head (input_embeds = the target embedding; the target lm_head scores the drafts).  Block = anchor row (last
    committed token) + block_size − 1 mask rows → block_size − 1 drafts (DFlash anchor-row semantics).  After every
    active layer (selected_prefix_feature_layers) a code-bridge write: read GEMM h → code_size (early_read_weights[l] /
    read_weight), L2-normalised codes (anchor row: codebook[anchor]), packed GEMM code_size → 3·rank, gated predecessor
    message, write GEMM rank → h (feature_write_weights[l]); bridge out.weight is zero (terminal disabled) → stored only.
    Tail: lm_head on the drafts → top-head_topk → serial_table (vocab tables TG [V, rank] + TSV [V, 2·rank], bf16,
    built at load) → greedy serial chain."""
    c = d["config"]
    dc = c.get("dflash_config") or {}
    br = dc["selected_prefix_code_bridge"]
    h, H, KV, hd = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    nl, V = c["num_hidden_layers"], c.get("draft_vocab_size") or c["vocab_size"]
    C, R = br["code_size"], br["rank"]
    act = tuple(int(i) - 1 for i in (dc.get("selected_prefix_feature_layers") or range(1, nl + 1)))   # 1-based → 0
    K = (dc.get("selected_prefix_last_write_refine") or {}).get("head_topk", 16)
    mtp = []
    for _ in range(nl):
        alin, core = _gqa(h, H, KV, hd)
        ffn, flin = _ffn_dense(h, c["intermediate_size"])
        mtp.append(Layer(alin, core, ffn, flin, 2 * h + 2 * hd))     # input / post-attention norms + q / k norms
    nt = len(dc["target_layer_ids"])
    # fc, hidden_norm + norm, bridge reads (nl − 1 early + last) / writes / gate·pred·prev / gate bias / zero out
    bridge = nl * C * h + nl * h * R + 3 * R * C + R + C * R
    extra = nt * h * h + 2 * h + bridge
    scal = sum(1 for v in (d.get("shapes") or {}).values() if not v)  # I64 version scalars (stored, 1 param each)
    cold = V * C + scal                                                # codebook: anchor rows + load-time tables only
    # TG [V, R] + TSV [V, 2R] bf16; 0.253: + the fp32 copies DSpineDraftSampler.__init__ makes for the sampling table
    # (predecessor_gate, prev [R, C] each, gate bias [R], last write [H, R], norm [H]) 「来源」 worker.py
    tables = V * 3 * R * 2 + (2 * R * C + R + h * R + h) * 4
    acc = d.get("acceptance") or {}
    left = extra - (extra // nl) * nl                                  # integer remainder kept with the cold params
    return replace(spec, mtp_layers=tuple(mtp), mtp_extra_params=extra // nl, draft="dspark", draft_anchor=True,
                   draft_block=c["block_size"], draft_targets=nt, draft_markov=0, draft_kv=2 * KV * hd,
                   draft_bridge=(C, R, K) + act, draft_cold_params=cold + left, draft_table_bytes=float(tables),
                   spec_published=(float(acc["tau"]), int(acc["block"])) if acc else (),
                   spec_published_t1=float(acc.get("tau_t1") or 0.0),
                   notes=spec.notes + (f"DSpine 草稿模型（{d['repo']}，bf16 {d['params'] / 1e9:.2f}B，独立仓库，共享目标 "
                                       f"embedding / lm_head）：DFlash 块草稿 {nl} 层（块内因果），块 {c['block_size']} = 锚行 + "
                                       f"{c['block_size'] - 1} 个草稿；目标层 {dc['target_layer_ids']} hidden 拼接 + fc；每层后 code-bridge 写入"
                                       f"（code {C}，秩 {R}）；尾部 top-{K} + 串行表（加载时建表 {tables / 1e6:.0f} MB 常驻「推导」）；"
                                       "参考实现只支持 TP = 1、块大小固定"
                                       + (f"；公开接受长度 τ {acc['tau']}（T = 0）/ {acc.get('tau_t1')}（T = 1），"
                                          f"{acc['block']} 个草稿 → i.i.d. 接受率 {spec_accept_from_tau(acc['tau'], acc['block']):.3f}"
                                          "「推导」；serving.spec_accept_src = published 按发布 τ（spec_sampling 选 T = 0 / 1）取接受率，默认仍为「假设」杠杆" if acc else ""),))


def _attach_dflash2(spec: ModelSpec, d: dict) -> ModelSpec:
    """0.254 DFlash2 draft (reflection/Beam-1.0-501B-A23B-DFlash2; vLLM main vllm/models/beam/nvidia/dflash2.py
    DFlash2BeamForCausalLM + model_executor/models/qwen3_dflash2.py DFlashGroupedConv / CandidateSelector +
    v1/worker/gpu/spec_decode/dflash2/speculator.py, 「来源」).  fc over the concatenated hidden states of the
    target_layer_ids → hidden_norm; context K/V of every draft layer from ONE stacked GEMM (h → layers·2·KV·hd, k_norm +
    RoPE) into a sliding-window (sliding_window) draft KV cache; num_hidden_layers decoder layers (GQA + q / k norm,
    SwiGLU) over the query block = anchor (bonus) row + mask rows, NON-causal (is_causal false) over window context +
    block; around attention and around the MLP a grouped dynamic convolution (kernel_projection GEMM h → 2·taps·groups
    coefficients; base_kernel [2, taps, h] + per-group delta, taps-tap causal conv inside the block, before (side 0) and
    after (side 1, fused into the residual norm) the sub-block).  No embedding / lm_head (the target's).  Tail: target
    lm_head on the block − 1 draft rows → top-k (selector_top_k) → selector: hidden_projection h → rank, edge scores
    unary + ⟨predecessor[q] ⊙ h_p, successor[c]⟩ for K × K (predecessor, candidate) pairs per row (codebooks [V, rank],
    gathered rows only) → sequential walk (gumbel-noised argmax per row; greedy = no noise)."""
    c = d["config"]
    dc = c.get("dflash_config") or {}
    h, H, KV, hd = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    nl, V = c["num_hidden_layers"], c.get("draft_vocab_size") or c["vocab_size"]
    taps, grp = int(dc["conv_kernel_size"]), int(dc["conv_group_size"])
    R, K = int(dc["selector_rank"]), int(dc["selector_top_k"])
    types = _list(c.get("layer_types")) or ["sliding_attention" if dc.get("use_swa") else "full_attention"] * nl
    win = dc.get("swa_window_size") or c.get("sliding_window")         # qwen3_dflash._resolve_layer_attention
    mtp = []
    for i in range(nl):
        alin, core = _gqa(h, H, KV, hd, window=win if types[i] == "sliding_attention" else None)
        ffn, flin = _ffn_dense(h, c["intermediate_size"])
        mtp.append(Layer(alin, core, ffn, flin, 2 * h + 2 * hd))     # input / post-attention norms + q / k norms
    nt = len(dc["target_layer_ids"])
    G = h // grp
    conv = 2 * (2 * taps * G * h + 2 * taps * h)                       # attention_conv + mlp_conv per layer
    extra = nt * h * h + 2 * h + nl * conv + R * h                     # fc, hidden_norm + norm, convs, hidden_projection
    cold = 2 * V * R                                                   # predecessor / successor codebooks (rows gathered)
    acc = d.get("acceptance") or {}
    left = extra - (extra // nl) * nl
    tb = int(dc.get("block_size") or c.get("block_size") or 16)
    serve = int(acc.get("block") or (tb - 1)) + 1                      # the card's serving block (anchor + drafts)
    return replace(spec, mtp_layers=tuple(mtp), mtp_extra_params=extra // nl, draft="dspark", draft_anchor=True,
                   draft_block=serve, draft_targets=nt, draft_markov=0, draft_kv=2 * KV * hd,
                   draft_dflash2=(taps, grp, R, K, tb), draft_cold_params=cold + left,
                   spec_published=(float(acc["tau"]), int(acc["block"])) if acc else (),
                   spec_published_t1=float(acc.get("tau_t1") or 0.0),
                   notes=spec.notes + (f"DFlash2 草稿模型（{d['repo']}，bf16 {d['params'] / 1e9:.2f}B，独立仓库，共享目标 "
                                       f"embedding / lm_head）：{nl} 层块草稿（块内非因果，上下文 SWA {win}），目标层 "
                                       f"{dc['target_layer_ids']} hidden 拼接 + fc；每层注意力 / MLP 前后各一个 {taps} 抽头分组动态卷积"
                                       f"（组宽 {grp}）；尾部 top-{K} 候选 + 低秩（{R}）选择器逐行走路径；块 = 1 + spec_k（训练 {tb}，"
                                       f"卡片服务块 {serve}）"
                                       + (f"；公开接受长度 τ {acc['tau']}（贪心）/ {acc.get('tau_t1')}（采样 T = {acc.get('t_sampled', 1.0):g}{'' if acc.get('t_sampled', 1.0) == 1.0 else '，非 T = 1「二手」'}），"
                                          f"{acc['block']} 个草稿 → i.i.d. 接受率 {spec_accept_from_tau(acc['tau'], acc['block']):.3f}"
                                          "「推导」" if acc else ""),))


def spec_accept_from_tau(tau: float, k: int) -> float:
    """0.221: the i.i.d. per-draft acceptance a with E[tokens / verify] = (1 − a^(k+1)) / (1 − a) = τ (bisection).
    τ counts committed tokens per verification including the target's own token (the schedule.py convention)."""
    if not 1.0 <= tau <= k + 1:
        raise ValueError(f"tau must be in [1, {k + 1}]")
    lo, hi = 0.0, 1.0
    for _ in range(80):
        a = (lo + hi) / 2
        e = (k + 1) if a >= 1 else (1 - a ** (k + 1)) / (1 - a)
        lo, hi = (a, hi) if e < tau else (lo, a)
    return (lo + hi) / 2


def _build_whisper(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """0.270 Whisper text decoder (transformers WhisperDecoderLayer 「来源」): pre-LN blocks of self-attention (MHA,
    q / v / o with bias, k without), cross-attention ``encoder_attn`` over the encoder output (same projections; k / v
    applied once per request to the encoder positions and cached), GELU fc1 / fc2 with bias; learned positions
    [max_target_positions, d]; final LayerNorm; proj_out tied to embed_tokens."""
    h, H, L, ff = c["hidden_size"], c["num_attention_heads"], c["num_hidden_layers"], c["intermediate_size"]
    hd = h // H
    layers = []
    for _ in range(L):
        alin, core = _gqa(h, H, H, hd)
        clin, ccore = _gqa(h, H, H, hd)
        ffn, flin = _ffn_dense(h, ff, gated=False)
        misc = 3 * h + 3 * h + (ff + h) + 3 * 2 * h      # self / cross q,v,o biases; fc1 / fc2 biases; 3 LayerNorms
        layers.append(Layer(alin, core, ffn, flin, misc, cross_linears=clin,
                            cross=replace(ccore, causal=False, cross=True)))
    notes = [f"编码器-解码器（Whisper）：解码器每层自注意力 + 交叉注意力（{L} 层），交叉 K/V 每请求在 prefill 时由编码器"
             f"输出算一次并缓存（每 30 s 窗 {c.get('max_source_positions', 1500)} 个位置），解码每步只读不写「来源」",
             "无 RoPE（学习的绝对位置表，查表）；GELU 激活按 SiLU 的逐元素开销计「近似」"]
    return layers, [], notes, "full"


def _build_rwkv7(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """0.191 RWKV-7 "Goose" (arXiv 2503.14456; release modeling_rwkv7.py).  No attention, no KV cache.
    Time mix: token shift → 6 lerps (r, w, k, v, a, g); receptance / key / value / output C×C; four low-rank chains
    w (tanh), a (sigmoid), g (sigmoid between), v (value residual, not run on layer 0 although stored); per head
    (H = C/64) the generalised delta rule on an fp32 64×64 state S ← S·diag(w) + S·(−κ̂)(κ̂·a)ᵀ + v·kᵀ, out = S·r;
    GroupNorm ln_x, bonus (r·k·r_k)·v, × g.  Channel mix: token shift, squared-ReLU C → 4C → C (2 mats).
    Per-sequence state per layer: H·64·64 wkv (fp32, config wkv_state_dtype) + two C-wide shift vectors (modelled as
    conv_channels = 2C, kernel 2 — one stored column each)."""
    h, L = c["hidden_size"], c["num_hidden_layers"]
    N = c.get("head_dim") or 64
    H = c.get("num_heads") or h // N
    dw, da, dv, dg = c["decay_low_rank_dim"], c["a_low_rank_dim"], c["v_low_rank_dim"], c["gate_low_rank_dim"]
    layers = []
    for i in range(L):
        alin = (Linear("rwkv_r", h, h, "col"), Linear("rwkv_k", h, h, "col"), Linear("rwkv_v", h, h, "col"),
                Linear("rwkv_w1", h, dw, "rep"), Linear("rwkv_w2", dw, h, "col"),
                Linear("rwkv_a1", h, da, "rep"), Linear("rwkv_a2", da, h, "col"),
                Linear("rwkv_g1", h, dg, "rep"), Linear("rwkv_g2", dg, h, "col")) + (
                () if i == 0 else (Linear("rwkv_v1", h, dv, "rep"), Linear("rwkv_v2", dv, h, "col"))) + (
                Linear("rwkv_o", h, h, "row"),)
        core = AttnCore("linear", n_state_heads=H, state_dk=N, state_dv=N, conv_channels=2 * h, conv_kernel=2,
                        lin="rwkv7", lin_chunk=64)     # release rwkv7_chunked(chunk_size=64)
        # x_r..x_g (6C), w0 a0 v0 (3C), k_k k_a (2C), r_k (H·N), ln_x w+b (2C), ffn x_k (C), ln1 + ln2 (4C);
        # layer 0: ln0 (2C) and the stored-but-unused v LoRA (2·C·dv + C)
        misc = 6 * h + 3 * h + 2 * h + H * N + 2 * h + h + 4 * h + ((2 * h + 2 * h * dv) if i == 0 else 0)
        ffn, flin = _ffn_dense(h, c["intermediate_size"], gated=False)
        layers.append(Layer(alin, core, ffn, flin, misc))
    notes = ["RWKV-7：无 KV 缓存；每层每序列状态 = H×64×64 fp32 WKV + 2 个 C 宽 token-shift 向量（按 fp32 计，"
             "发布实现为激活 dtype，偏保守 2C×2 B/层）",
             "解码代价与上下文长度无关（状态大小固定）；prefill 按发布 rwkv7_chunked（chunk 64）的矩阵乘逐项计"]
    return layers, [], notes, "full"


def _build_rwkv8(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """0.361 RWKV-8 "Heron" pile rc00 (BlinkDL/rwkv-8-pile; rwkv_v8_rc00_demo.py / rwkv_v8_rc00_hybrid_demo.py).

    Time-mix is the RWKV-7 generalised delta rule (same WKV-7 op, fp32 H x 64 x 64 state).  The published seq path
    calls that op over the whole sequence; prefill keeps the RWKV-7 chunk-64 schedule (rwkv7_chunked) so the two
    series share one recurrence.  Channel mix is squared ReLU then a per-token DeepEmbed row (ffn.enn[idx]); the
    elementwise cost stays the shared activation op (same accounting as RWKV-7).  The hybrid checkpoint adds
    DeepEmbedAttention beside the time-mix: fp16 kk / vv cache (32 + 32), full-cache up-projections, per-token
    ennk / emm rows, soft-capped single-head causal attention.  No ROSA (not in these checkpoints)."""
    h, L = c["hidden_size"], c["num_hidden_layers"]
    N = c.get("head_dim") or 64
    H = c.get("num_heads") or h // N
    dw, da, dv, dg = c["decay_low_rank_dim"], c["a_low_rank_dim"], c["v_low_rank_dim"], c["gate_low_rank_dim"]
    ln0 = bool(c.get("ln0", False))
    de = int(c.get("deepemb") or 0)
    dea = c.get("dea") or {}
    layers = []
    for i in range(L):
        alin = (Linear("rwkv_r", h, h, "col"), Linear("rwkv_k", h, h, "col"), Linear("rwkv_v", h, h, "col"),
                Linear("rwkv_w1", h, dw, "rep"), Linear("rwkv_w2", dw, h, "col"),
                Linear("rwkv_a1", h, da, "rep"), Linear("rwkv_a2", da, h, "col"),
                Linear("rwkv_g1", h, dg, "rep"), Linear("rwkv_g2", dg, h, "col")) + (
                () if i == 0 else (Linear("rwkv_v1", h, dv, "rep"), Linear("rwkv_v2", dv, h, "col"))) + (
                Linear("rwkv_o", h, h, "row"),)
        core = AttnCore("linear", n_state_heads=H, state_dk=N, state_dv=N, conv_channels=2 * h, conv_kernel=2,
                        lin="rwkv7", lin_chunk=64)
        # Same misc as _build_rwkv7.  Layer 0 stores the unused v LoRA; ln0 only when the checkpoint has it.
        misc = 6 * h + 3 * h + 2 * h + H * N + 2 * h + h + 4 * h + ((2 * h * dv) if i == 0 else 0)
        if i == 0 and ln0:
            misc += 2 * h
        if dea:
            q, k, kq = int(dea["q"]), int(dea["k"]), int(dea["kq"])
            v, vd = int(dea["v"]), int(dea["vd"])
            alin = (Linear("dea_qq", h, q, "rep"), Linear("dea_kk", h, k, "rep"), Linear("dea_vv", h, v, "rep"),
                    Linear("dea_kkup", k, kq, "rep"), Linear("dea_vvup", v, vd, "rep")) + alin
            misc += 2 * (q + kq + vd)       # lnq, lnk, lnv (weight + bias)
            core = replace(core, dea_q=q, dea_k=k, dea_kq=kq, dea_v=v, dea_vd=vd,
                           dea_k_emb=int(dea["k_emb"]), dea_v_emb=int(dea["v_emb"]))
        ffn, flin = _ffn_dense(h, c["intermediate_size"], gated=False)
        layers.append(Layer(alin, core, ffn, flin, misc, deepemb=de))
    notes = ["RWKV-8 Heron：时间混合与 RWKV-7 相同（WKV-7，fp32 状态 H×64×64；prefill 按 rwkv7_chunked chunk 64 计，"
             "与已发布的 RWKV-7 同一套递推）。通道混合是 squared ReLU，逐元素开销沿用现有激活算子（与 RWKV-7 相同）「近似」",
             "DeepEmbed：每层 ffn.enn 只按当前 token 取一行，乘到 FFN 输出（不随上下文变长）"]
    if dea:
        notes.append("DeepEmbedAttention（hybrid demo forward_seq）：每层缓存 fp16 的 kk/vv（各 32 维），"
                     "不是展开后的 K/V；每步对全部已缓存 token 做 kkup/vvup，再乘 ennk/emm 行，"
                     "soft-cap（32·tanh(qk/512)）后单头因果注意力。参考实现是单卡：这些投影按复制计，缓存不按 TP 切分")
    else:
        notes.append("无 DeepEmbedAttention，无 KV 缓存；解码代价与上下文长度无关（状态大小固定）")
    return layers, [], notes, "full"


def _build_dory(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """0.253 NVIDIA Nemotron-Labs Dory (DoryForCausalLM; vLLM PR #60149 models/dory.py 「来源」): normalized-GPT looped
    LM.  ``num_hidden_layers`` physical layers, each ONE sublayer — attention (layer_types sliding / full) or SwiGLU
    MLP — with an nGPT slerp residual (justnorm, per-channel alpha); GQA with nGPT qk-norm after RoPE, a sigmoid output
    gate (g_proj h → H·hd, fused here into q as out_gate) and justnorm before o_proj; MLP with suv scales and justnorm
    before down_proj.  Returns the PHYSICAL layers; from_release expands input + recurrent × loops + output."""
    h, H, KV = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"]
    hd = c.get("head_dim") or h // H
    types = _list(c.get("layer_types"))
    layers = []
    for i in range(c["num_hidden_layers"]):
        t = types[i]
        if t == "mlp":
            ffn, flin = _ffn_dense(h, c["intermediate_size"])
            layers.append(Layer((), AttnCore("none"), ffn, flin, 2 * c["intermediate_size"] + h))   # suv_gate/up, mlp_alpha
        else:
            alin, core = _gqa(h, H, KV, hd, out_gate=True, window=c.get("sliding_window") if t == "sliding_attention" else None)
            layers.append(Layer(alin, core, Ffn("none"), (), 2 * H * hd + h))                   # sqk, s_gate, attn_alpha
    notes = [f"Dory 循环 LM：输入 {c['n_input_layers']} / 循环 {c['n_recurrent_layers']} / 输出 {c['n_output_layers']} 个子层"
             "（每个物理层只有注意力或 MLP），循环块权重共享、每次循环重读（vLLM PR #60149「来源」）",
             f"滑窗 {c.get('sliding_window')} 已建模；nGPT justnorm / slerp 残差按向量运算计「近似」"]
    return layers, [], notes, "full"


def _mamba2(c: dict, h: int) -> tuple[tuple[Linear, ...], AttnCore, int]:
    """0.278 Mamba-2 mixer (transformers NemotronHMamba2Mixer 「来源」): in_proj h → [z (d_in) | xBC (d_in + 2·G·N) | dt (H)],
    causal depthwise conv1d (kernel 4, bias) over xBC, SSD scan (H heads of P = mamba_head_dim, state N, B / C shared per
    group of H / G heads, scalar decay exp(dt·A) per head, D skip), gated RMSNorm (group size d_in / G) with z, out_proj
    d_in → h.  Prefill = the chunked SSD of the reference kernel (mamba_chunk_scan_combined, chunk_size = 128)."""
    H, P, N, G = c["mamba_num_heads"], c["mamba_head_dim"], c["ssm_state_size"], c["n_groups"]
    d_in = H * P
    conv = d_in + 2 * G * N
    kern = c.get("conv_kernel") or 4
    lin = (Linear("mamba_in", h, d_in + conv + H, "col"), Linear("mamba_out", d_in, h, "row"))
    core = AttnCore("linear", n_state_heads=H, state_dk=N, state_dv=P, conv_channels=conv, conv_kernel=kern,
                    lin="ssd", lin_chunk=c.get("chunk_size") or 128, state_groups=G)
    misc = conv * kern + (conv if c.get("use_conv_bias", True) else 0) + 3 * H + d_in    # conv w (+ b), A_log / D / dt_bias, norm
    return lin, core, misc


# 0.363: Nemotron-H releases that predate layers_block_type publish the same schedule as
# hybrid_override_pattern. The alphabet is the one Nemotron-3.5 expands into layers_block_type
# (Nano-30B-A3B pattern == Lightning's list): M mamba, E moe, * attention, - mlp.
# Unknown characters are rejected.
_NEMOTRON_PAT = {"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}


def _nemotron_block_types(c: dict) -> tuple[list, list[str]]:
    bcs = list(c.get("block_configs") or ())
    if bcs:
        return bcs, [b["block_type"] for b in bcs]
    if c.get("layers_block_type"):
        return [], list(c["layers_block_type"])
    pat = c.get("hybrid_override_pattern") or ""
    bad = sorted({ch for ch in pat if ch not in _NEMOTRON_PAT})
    if not pat or bad:
        raise ValueError(f"nemotron_h: no layers_block_type; hybrid_override_pattern {pat!r} has unknown chars {bad}")
    return [], [_NEMOTRON_PAT[ch] for ch in pat]


def _build_nemotron_h(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """0.278 Nemotron-H (transformers NemotronHForCausalLM 「来源」): ``layers_block_type`` gives one mixer per layer —
    mamba (Mamba-2, _mamba2) | attention (GQA, no RoPE in the reference attention path's cost) | moe (NemotronHMoE: sigmoid
    top-k router with correction bias, relu² non-gated routed experts in the moe_latent_size space between
    fc1_latent_proj / fc2_latent_proj, one relu² shared expert at full width) | mlp (relu² up / down); one pre-norm
    each.  MTP (num_nextn_predict_layers, mtp_layers_block_type): eh_proj 2h → h + enorm / hnorm + one block per type.
    0.363: a release that only ships hybrid_override_pattern uses that string (see _NEMOTRON_PAT)."""
    h = c["hidden_size"]
    # 0.289 Nemotron-H Puzzle (Nemotron-3.5-Bolt, configuration_nemotron_h_puzzle.py 「来源」): ``block_configs[i]`` updates
    # the global config for layer i (get_nemotron_h_config_for_layer) — per-layer routed experts, MoE width and top-k
    bcs, types = _nemotron_block_types(c)
    lat = c.get("moe_latent_size") or 0

    def block(t: str, i: int = -1, mtp_cfg: dict | None = None):
        cc = {**c, **(mtp_cfg if mtp_cfg is not None else bcs[i] if 0 <= i < len(bcs) else {})}
        E, k = cc.get("n_routed_experts") or 0, cc.get("num_experts_per_tok") or 0
        if t == "mamba":
            alin, core, misc = _mamba2(c, h)
            return Layer(alin, core, Ffn("none"), (), misc + h)
        if t == "attention":
            H, KV = c["num_attention_heads"], c["num_key_value_heads"]
            alin, core = _gqa(h, H, KV, c.get("head_dim") or h // H)
            return Layer(alin, core, Ffn("none"), (), h)
        if t == "moe":
            ffn, flin = _ffn_moe(h, E, k, cc["moe_intermediate_size"], c.get("n_shared_experts") or 0,
                                 c.get("moe_shared_expert_intermediate_size") or 0, gated=False, latent=lat)
            return Layer((), AttnCore("none"), ffn, flin, h + E)          # norm + e_score_correction_bias
        if t == "mlp":
            ffn, flin = _ffn_dense(h, c["intermediate_size"], gated=False)
            return Layer((), AttnCore("none"), ffn, flin, h)
        raise ValueError(f"nemotron_h: unknown block type {t}")
    layers = [block(t, i) for i, t in enumerate(types)]
    mtp = []
    if bcs and c.get("mtp_block_configs"):
        raise ValueError("nemotron_h_puzzle: per-block MTP configs are not modelled (no release has them)")
    for _ in range(0 if bcs else c.get("num_nextn_predict_layers") or 0):
        # one MTP module = its blocks in sequence ([attention, moe]) → one decoder layer (attention core + MoE FFN);
        # eh_proj / enorm / hnorm / final norm → mtp_extra_params (from_release: release params_mtp − the layer)
        parts = [block(t) for t in (c.get("mtp_layers_block_type") or [])]
        at = next((p for p in parts if p.core.kind != "none"), None)
        ff = next((p for p in parts if p.ffn.kind != "none"), None)
        if at is not None and ff is not None and len(parts) == 2:
            mtp.append(Layer(at.attn_linears, at.core, ff.ffn, ff.ffn_linears, at.misc_params + ff.misc_params))
    n = {t: types.count(t) for t in dict.fromkeys(types)}
    mf = [L.ffn for L in layers if L.ffn.kind == "moe"]
    rng = lambda xs: f"{min(xs)}" if min(xs) == max(xs) else f"{min(xs)}–{max(xs)}"
    es = (f"{rng([f.n_experts for f in mf])} 专家 top-{rng([f.top_k for f in mf])}"
          + (f"，宽 {rng([f.d_expert for f in mf])}（逐层异构，Puzzle block_configs）" if bcs else "")) if mf else "无"
    mlp_n = f" + {n['mlp']} 层 relu² MLP" if n.get("mlp") else ""
    notes = [f"Nemotron-H 混合：{n.get('mamba', 0)} 层 Mamba-2（SSD，{c['mamba_num_heads']} 头 × {c['mamba_head_dim']}，"
             f"状态 {c['ssm_state_size']}，B/C 按 {c['n_groups']} 组共享，chunk {c.get('chunk_size') or 128}，状态 fp32）+ "
             f"{n.get('attention', 0)} 层 GQA + {n.get('moe', 0)} 层 latent MoE（{es}，relu² 非门控，"
             f"latent {lat}，共享专家 {c.get('moe_shared_expert_intermediate_size')}）{mlp_n}；每层只有一个 mixer"]
    return layers, mtp, notes, "full"


def with_loops(spec: ModelSpec, loops: int, mode: str) -> ModelSpec:
    """0.253 Dory: expand the physical layers for ``loops`` applications of the recurrent block (n_recurrent_loops)
    and the KV mode (recurrent_kv_cache_mode: per_loop = one cache per loop (reference default) | last_loop = one cache
    per recurrent attention layer, every loop overwriting the step's rows)."""
    n_in, n_rec = spec.loop_spec[0], spec.loop_spec[1]
    ph = spec.loop_phys
    rec = ph[n_in:n_in + n_rec]
    ls = list(ph[:n_in])
    for i in range(loops):
        ls += [L if i == 0 else replace(L, loop=i, kv_shared=(mode == "last_loop" and L.core.kind != "none"))
               for L in rec]
    ls += list(ph[n_in + n_rec:])
    return replace(spec, layers=tuple(ls), loop_spec=(n_in, n_rec, loops, mode))


def _gemma4_unloaded_kv(c: dict) -> int:
    """k_proj, v_proj and k_norm that the checkpoint stores on KV-shared layers.

    Gemma4TextAttention does not construct those modules past the sharing cut
    (_keys_to_ignore_on_load_unexpected). v_norm has with_scale=False (0 params).
    The tensors stay in the safetensors header. They are not loaded and not executed.
    """
    n_share = int(c.get("num_kv_shared_layers") or 0)
    L = int(c.get("num_hidden_layers") or 0)
    types = _list(c.get("layer_types"))
    if n_share <= 0 or L <= 0 or n_share >= L or len(types) != L:
        return 0
    h = c["hidden_size"]
    H = c["num_attention_heads"]
    kv_s = c.get("num_key_value_heads") or H
    hd_s = c.get("head_dim") or h // H
    hd_f = c.get("global_head_dim") or hd_s
    k_eq = bool(c.get("attention_k_eq_v"))
    kv_f = (c.get("num_global_key_value_heads") or kv_s) if k_eq else kv_s
    total = 0
    for i in range(L - n_share, L):
        full = types[i] == "full_attention"
        hd = hd_f if full else hd_s
        kv = kv_f if full else kv_s
        total += h * kv * hd
        if not (full and k_eq):
            total += h * kv * hd
        total += hd
    return total


def _build_gemma4(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """Gemma 4 text (gemma4_text / gemma4_unified_text).

    Sliding layers use head_dim and num_key_value_heads, with a real v_proj and
    window = sliding_window. Full layers use global_head_dim. num_global_key_value_heads
    and the omitted v_proj apply only when attention_k_eq_v is set. V is then
    RMSNorm(K) with with_scale=False (no parameter) and is still written to the
    KV cache, so the cache stores both K and V.

    The last num_kv_shared_layers read the donor layer's cache tensors
    (shared_kv_states holds the Cache.update result; sliding attention is still
    masked to sliding_window). kv_alias: no own cache and kv_write = 0. There is
    no second full-length buffer. Checkpoint k/v on those layers is cached_params.

    Per-layer embeddings: one packed row (layers * ple) per token, one projection
    GEMM on layer 0, then a gate and a projection on every layer.
    use_double_wide_mlp widens only the shared layers' MLP.
    """
    h = c["hidden_size"]
    H = c["num_attention_heads"]
    kv_s = c.get("num_key_value_heads") or H
    hd_s = c.get("head_dim") or h // H
    hd_f = c.get("global_head_dim") or hd_s
    k_eq = bool(c.get("attention_k_eq_v"))
    kv_f = (c.get("num_global_key_value_heads") or kv_s) if k_eq else kv_s
    L = c["num_hidden_layers"]
    types = _list(c.get("layer_types"))
    if not types or len(types) != L or any(t not in ("sliding_attention", "full_attention") for t in types):
        raise ValueError("gemma4 requires layer_types of length num_hidden_layers (sliding_attention | full_attention)")
    window = c.get("sliding_window")
    if any(t == "sliding_attention" for t in types) and not window:
        raise ValueError("gemma4 sliding_attention requires sliding_window")
    n_share = int(c.get("num_kv_shared_layers") or 0)
    if n_share < 0 or n_share >= L:
        raise ValueError("gemma4 num_kv_shared_layers must leave at least one donor layer")
    first_share = L - n_share
    if n_share and not set(types[first_share:]) <= set(types[:first_share]):
        raise ValueError("gemma4 KV-shared layers need a non-shared donor of the same layer_type")
    ple = int(c.get("hidden_size_per_layer_input") or 0)
    if ple < 0:
        raise ValueError("gemma4 hidden_size_per_layer_input must be >= 0")
    wide = bool(c.get("use_double_wide_mlp"))
    inter = c["intermediate_size"]
    moe = bool(c.get("enable_moe_block"))
    E = k = de = 0
    if moe:
        E, k, de = c["num_experts"], c["top_k_experts"], c["moe_intermediate_size"]
    layers = []
    for i in range(L):
        full = types[i] == "full_attention"
        shared = n_share > 0 and i >= first_share
        hd = hd_f if full else hd_s
        kv = kv_f if full else kv_s
        # Shared layers have q / o only. Full + attention_k_eq_v: no v_proj.
        if shared:
            lin = (Linear("q", h, H * hd, "col"),
                   Linear("o", H * hd, h, "row"))
        elif full and k_eq:
            lin = (Linear("q", h, H * hd, "col"),
                   Linear("k", h, kv * hd, "head", unit=hd),
                   Linear("o", H * hd, h, "row"))
        else:
            lin, _core = _gqa(h, H, kv, hd, window=None if full else window)
        if ple and i == 0:
            lin = lin + (Linear("ple_proj", h, L * ple, "col", role="other"),)
        core = AttnCore("gqa", n_q=H, n_kv=kv, qk_dim=hd, v_dim=hd, window=None if full else window)
        # Four decoder RMSNorms + layer_scalar [1] + q_norm. k_norm only when this
        # layer projects K. v_norm is with_scale=False. Final RMSNorm is ModelSpec's.
        misc = 4 * h + 1 + (hd if shared else 2 * hd)
        if ple:
            misc += h
        d_ff = inter * (2 if wide and shared else 1)
        if moe:
            # Dense MLP on every token plus routed experts. Router RMSNorm has no
            # scale. router.scale [h] and per_expert_scale [E] are parameters.
            ffn, flin = _ffn_moe(h, E, k, de, 1, d_ff)
            misc += 3 * h + h + E
        else:
            ffn, flin = _ffn_dense(h, d_ff)
        if ple:
            flin = flin + (Linear("ple_gate", h, ple, "col", role="other"),
                           Linear("ple_up", ple, h, "row", role="other"))
        layers.append(Layer(lin, core, ffn, flin, misc, kv_alias=shared, ple_in=ple))
    n_s = types.count("sliding_attention")
    n_f = types.count("full_attention")
    notes = [f"滑窗 {window}（{n_s} 层，head {hd_s}，KV {kv_s}）与全注意力（{n_f} 层，head {hd_f}，KV {kv_f}）分开记账"]
    if k_eq and n_f:
        notes.append("全注意力 attention_k_eq_v：无 v_proj；V 为无 scale 的 RMSNorm(K)，KV 仍存储 K 与 V")
    if n_share:
        notes.append(f"后 {n_share} 层读捐赠层 KV（滑窗仍按 window，不另存全长副本），本层不写 KV")
    if ple:
        notes.append(f"逐层嵌入：每 token 查一行（{L}×{ple}），第 0 层投影；每层 gate / proj")
    if wide and n_share:
        notes.append(f"共享层 MLP 双倍宽（intermediate {inter}×2）")
    if moe:
        notes.append(f"每层稠密 MLP（intermediate {inter}）与 {E} 专家 top-{k}（moe intermediate {de}）同时计算")
    notes.append("final_logit_softcapping 与 partial RoPE 不改变 GEMM / KV 宽度，不计额外周期")
    return layers, [], notes, "full"


def _longcat_attn(h: int, nh: int, q_lora: int, kv_lora: int, nope: int, rope: int, vdim: int) -> tuple:
    """LongCat-Flash MLA projections. kv_b expands to per-head K/V before the cache, so the core is GQA."""
    qk = nope + rope
    lin = (
        Linear("q_a", h, q_lora, "rep"),
        Linear("q", q_lora, nh * qk, "col"),
        Linear("kv_a", h, kv_lora + rope, "rep"),
        Linear("kv_up", kv_lora, nh * (nope + vdim), "col"),
        Linear("o", nh * vdim, h, "row"),
    )
    core = AttnCore("gqa", n_q=nh, n_kv=nh, qk_dim=qk, v_dim=vdim, rope_dim=rope)
    return lin, core


def _build_longcat_flash(c: dict, *, draft: bool = True, variant: str = "") -> tuple[list[Layer], list[Layer], list[str], str]:
    """LongCat-Flash (meituan-longcat/LongCat-Flash-Chat, modeling_longcat_flash.py).

    Each of ``num_layers`` blocks runs two sub-layers. Both have MLA attention and a dense SwiGLU
    (``ffn_hidden_size``). Sublayer 0 also runs a routed MoE on the same post-attention stream; that
    MoE output is added after sublayer 1 (``LongcatFlashDecoderLayer.forward``), not inside sublayer 0's
    residual. The GEMMs are the same as attention, dense MLP, and MoE on those widths.

    Attention is not latent-cache MLA. ``kv_b_proj`` expands the compressed KV to per-head K (nope) and V,
    RoPE is applied to the key, and ``past_key_values.update`` stores
    ``[batch, num_heads, seq, qk_nope + qk_rope]`` and ``[batch, num_heads, seq, v_head_dim]``.
    The single key RoPE vector is expanded across heads before that write.

    The router scores ``n_routed_experts + zero_expert_num`` classes (``LongcatFlashTopkRouter``).
    Identity zero-experts have no parameters; a hit copies the token. ``e_score_correction_bias`` is an
    fp32 buffer of that width, stored in the MoE sublayer's misc. ``classifier.weight`` is fp32 as released.
    ``mla_scale_*`` and ``routed_scaling_factor`` are Python floats. They do not change GEMM or KV shape,
    so they add no cycles. The published HF class still lists ``model.mtp.*`` in
    ``_keys_to_ignore_on_load_unexpected``. Public vLLM ``LongCatFlashMTP`` loads and runs that module.
    ``speculative.py`` rewrites ``model_type`` from ``longcat_flash`` to ``longcat_flash_mtp``
    (architecture ``LongCatFlashMTPModel``, ``n_predict = num_nextn_predict_layers or 1``).
    ``is_deepseek_mla()`` is true for ``longcat_flash`` and false for ``longcat_flash_mtp``, so the draft
    block is ``DeepseekV2Attention``, not latent MLA. That forward expands ``kv_b``, then pads V from
    ``v_head_dim`` to ``qk_head_dim`` before ``Attention``. ``Attention`` defaults ``head_size_v`` to
    ``head_size``, and the call passes ``qk_head_dim`` with no ``head_size_v``, so the KV cache stores
    the pad. The pad is sliced off before ``o_proj``. Released ``o`` and ``kv_b`` stay at ``v_head_dim``.
    The draft clears ``n_routed_experts``, so the MLP is the dense SwiGLU. The checkpoint has
    ``model.mtp.layers.0`` only. ``eh_proj`` (2h → h), the private embedding and ``mtp.norm`` stay in the
    ``from_release`` remainder (no separate lookup, no second ``lm_head``). ``enorm`` / ``hnorm`` are the
    extra RMSNorms on the layer. ``mla_scale_*`` is a load-time norm scale. The main stack follows the HF
    class and keeps unpadded V.

    ``draft=False`` is the n-gram subclass (``LongcatFlashNgramForCausalLM``). That class and public vLLM
    both drop ``model.mtp.*``, so this builder emits no MTP layer and does not pad V.

    ``variant="omni"`` is ``LongcatFlashOmniForCausalLM``. The text stack is the same Flash
    decoder. The checkpoint has no ``model.mtp.*``, so the builder emits no MTP layer and
    does not pad V. Audio and vision towers are attached later from their own headers.
    """
    h = int(c["hidden_size"])
    n_blocks = int(c.get("num_layers") or c["num_hidden_layers"])
    nh = int(c["num_attention_heads"])
    nope = int(c["qk_nope_head_dim"])
    rope = int(c["qk_rope_head_dim"])
    vdim = int(c["v_head_dim"])
    q_lora = int(c["q_lora_rank"])
    kv_lora = int(c["kv_lora_rank"])
    routed = int(c["n_routed_experts"])
    zero = int(c.get("zero_expert_num") or 0)
    if c.get("zero_expert_type") not in (None, "identity"):
        raise ValueError("longcat_flash zero_expert_type must be identity")
    topk = int(c["moe_topk"])
    d_exp = int(c["expert_ffn_hidden_size"])
    d_ff = int(c["ffn_hidden_size"])
    pool = routed + zero
    attn_misc = 2 * h + q_lora + kv_lora
    layers: list[Layer] = []
    for _ in range(n_blocks):
        lin, core = _longcat_attn(h, nh, q_lora, kv_lora, nope, rope, vdim)
        dense_lin = (Linear("gate_up", h, 2 * d_ff, "col"), Linear("down", d_ff, h, "row"))
        moe_lin = dense_lin + (Linear("router", h, pool, "rep"),)
        layers.append(Layer(
            lin, core,
            Ffn("moe", d_ff=d_ff, gated=True, n_experts=routed, top_k=topk, d_expert=d_exp, zero_experts=zero),
            moe_lin, attn_misc + pool))
        lin1, core1 = _longcat_attn(h, nh, q_lora, kv_lora, nope, rope, vdim)
        ffn1, flin1 = _ffn_dense(h, d_ff)
        layers.append(Layer(lin1, core1, ffn1, flin1, attn_misc))
    qk = nope + rope
    mtp: list[Layer] = []
    if variant == "omni":
        if draft:
            raise ValueError("longcat omni does not run MTP")
        mtp_note = (
            "类是 LongcatFlashOmniForCausalLM。文本栈是 LongcatFlashForCausalLM。"
            "checkpoint 没有 model.mtp.*。不建 draft，也不把 V 补到 qk_head_dim。"
            "主栈仍按 HF 类把未填充的 V 写入 cache。"
        )
    elif draft:
        if qk < vdim:
            raise ValueError("longcat_flash MTP pads V up to qk_head_dim")
        mtp_lin, mtp_core = _longcat_attn(h, nh, q_lora, kv_lora, nope, rope, vdim)
        mtp_core = replace(mtp_core, v_dim=qk)
        mtp_ffn, mtp_flin = _ffn_dense(h, d_ff)
        mtp = [Layer(mtp_lin, mtp_core, mtp_ffn, mtp_flin, attn_misc + 2 * h)]
        mtp_note = (
            "发布的 HF 类仍把 model.mtp.* 列在 _keys_to_ignore_on_load_unexpected。"
            "公开 vLLM 的 LongCatFlashMTP 会加载并运行这一层：speculative 把 model_type 改成 longcat_flash_mtp，"
            "use_mla 为假，注意力是 DeepseekV2Attention。kv_b 先展开，再把 V 从 v_head_dim 补到 qk_head_dim 写入 KV cache，"
            "o_proj 之前再切回 v_head_dim。o 与 kv_b 的形状仍是已发布的 v_head_dim。"
            "draft 清空 n_routed_experts，所以 MTP 的 MLP 是稠密 SwiGLU，没有 router。"
            "只有一层（checkpoint 是 model.mtp.layers.0，config 没有 num_nextn_predict_layers）。"
            "eh_proj 是 2h→h。enorm / hnorm 在层内 misc。私有 embedding 与 mtp.norm 留在余量里，不另做查表，也不另写 lm_head。"
            "mla_scale_* 在加载时折进 norm 权重。主栈仍按 HF 类把未填充的 V 写入 cache。"
        )
    else:
        mtp_note = (
            "这个类是 LongcatFlashNgramForCausalLM。HF 把 model.mtp.* 列在 "
            "_keys_to_ignore_on_load_unexpected，公开 vLLM 的 hf_to_vllm_mapper 把 model.mtp. 映射成空。"
            "speculative 只改写 model_type longcat_flash，不改写 longcat_flash_ngram。"
            "不建 MTP 层，也不把 V 补到 qk_head_dim。主栈仍按 HF 类把未填充的 V 写入 cache。"
        )
    notes = [
        f"LongCat-Flash：{n_blocks} 个 block，每个 block 两套注意力 + 两套稠密 SwiGLU（intermediate {d_ff}）。"
        f"第 0 子层的 MoE 与该子层稠密 MLP 读同一条 post-attention 流，MoE 输出加在第 1 子层残差之后。",
        f"注意力在写入 cache 之前用 kv_b 展开成多头 K/V（{nh} 头，K {qk}，V {vdim}），"
        f"不是 latent cache（kv_lora {kv_lora} + RoPE {rope}）。q/k 的低秩 RMSNorm 在 q_lora / kv_lora 上。",
        f"MoE：{routed} 个有权重的专家（intermediate {d_exp}）top-{topk}，另有 {zero} 个恒等 zero-expert（无参数）。"
        f"router 对 {pool} 类打分，classifier.weight 与 e_score_correction_bias 按发布是 fp32。",
        "mla_scale_q_lora / mla_scale_kv_lora 与 routed_scaling_factor 是 Python 标量，不改变 GEMM 形状，也不改变 KV 宽度，不计额外周期。",
        mtp_note,
    ]
    return layers, mtp, notes, "full"


_LONGCAT_CAUSAL_EXTRA = (
    "index_n_heads", "index_head_dim", "index_topk", "indexer_rope_interleave",
    "mtp_num_layers", "oe_vocab_size_ratio", "oe_neighbor_num", "oe_split_num",
    "ngram_vocab_size_ratio", "emb_neighbor_num", "emb_split_num", "dsa_mtp_cli",
    "cli_factor",
)


def _is_longcat_omni(c: dict) -> bool:
    """``LongcatFlashOmniForCausalLM``. The text class is Flash. Towers live beside it."""
    archs = c.get("architectures") or []
    mt = c.get("model_type") or ""
    return "LongcatFlashOmniForCausalLM" in archs or mt == "longcat_flash_omni"


def _build_longcat_omni(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """LongCat-Flash-Omni text stack. Same dual-sublayer decoder as Flash, no MTP."""
    if c.get("num_layers") is None:
        raise ValueError("longcat omni block count is num_layers")
    return _build_longcat_flash(c, draft=False, variant="omni")


def _is_longcat_causal(c: dict) -> bool:
    """Published ``LongcatForCausalLM`` (config architectures often say ``LongcatCausalLM``).

    Not ``LongcatFlashForCausalLM`` and not the n-gram subclass. ``model_type`` on the class
    is ``longcat``; the released JSON often omits it.
    """
    archs = c.get("architectures") or []
    mt = c.get("model_type") or ""
    return "LongcatCausalLM" in archs or "LongcatForCausalLM" in archs or mt == "longcat"


def _build_longcat_causal(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """LongCat-Flash-Thinking-2601 / Prover (modeling_longcat.py).

    ``LongcatDecoderLayer`` and ``LongcatMLA`` are the Flash main stack: two attention
    sub-layers per block, MoE shortcut after sublayer 1, ``kv_b`` expanded before the
    cache write. The cache stores unpadded V. ``flash_attention_2`` pads V only after
    that write, for the kernel, then slices it off before ``o_proj``.

    ``LongcatRotaryEmbedding`` calls ``ROPE_INIT_FUNCTIONS['deepseek_yarn']`` and
    multiplies cos/sin by ``attention_scaling``. ``inv_freq`` is not persistent.
    That does not change GEMM or KV width.

    The HF class has no MTP module (``_keys_to_ignore_on_load_unexpected`` is None).
    The official deployment guide runs the checkpoint MTP with vLLM method
    ``longcat_flash_mtp``. ``speculative.py`` rewrites only ``model_type == longcat_flash``,
    so this checkpoint must pass the method explicitly. That draft is the Flash MTP
    already billed (V padded to ``qk_head_dim``). The layer is emitted here and
    ``from_release`` drops it when the release has no ``model.mtp.*`` (Prover).

    LSA, the indexer, over-encoding, and n-gram fields are rejected. They are not this stack.
    """
    if c.get("attention_method") != "MLA":
        raise ValueError("longcat attention_method is not MLA")
    for key in _LONGCAT_CAUSAL_EXTRA:
        if c.get(key) not in (None, False, 0):
            raise ValueError(f"longcat field {key} is not the published MLA stack")
    rope = c.get("rope_scaling")
    if not isinstance(rope, dict) or rope.get("rope_type") != "deepseek_yarn":
        raise ValueError("longcat rope_scaling is not deepseek_yarn")
    layers, mtp, _notes, cov = _build_longcat_flash(c, draft=True)
    n_blocks = len(layers) // 2
    notes = [
        "类是 LongcatForCausalLM（config 的 architectures 写的是 LongcatCausalLM）。"
        f"modeling_longcat.py 的 LongcatDecoderLayer 与 LongcatMLA 和 Flash 是同一条主栈："
        f"{n_blocks} 个 block，每个 block 两套注意力加两套稠密 SwiGLU。"
        "第 0 子层的 MoE 与该子层稠密 MLP 读同一条 post-attention 流，MoE 输出加在第 1 子层残差之后。",
        "注意力在写入 cache 之前用 kv_b 展开成多头 K/V，不是 latent cache。"
        "flash_attention_2 只在 cache 更新之后把 V 补到 qk_head_dim 交给 kernel，再在 o_proj 前切回。"
        "eager 与 sdpa 不补。cache 里的 V 仍是未填充的 v_head_dim。",
        "LongcatRotaryEmbedding 走 ROPE_INIT_FUNCTIONS 的 deepseek_yarn。"
        f"factor {rope.get('factor')}，original_max_position_embeddings "
        f"{rope.get('original_max_position_embeddings')}，rope_theta {c.get('rope_theta')}，"
        f"max_position_embeddings {c.get('max_position_embeddings')}。"
        "forward 把 cos/sin 乘上 attention_scaling。inv_freq 不是 persistent。"
        "不改变 GEMM 形状，也不改变 KV 宽度，不计额外周期。",
        "mla_scale_q_lora / mla_scale_kv_lora 与 routed_scaling_factor 是 Python 标量，不计额外周期。"
        "hidden_act 缺省是 silu。zero-expert 是 identity，没有权重。",
        "HF 类没有 MTP 模块，_keys_to_ignore_on_load_unexpected 是 None，forward 不跑 model.mtp.*。"
        "官方部署文档用 vLLM speculative method longcat_flash_mtp 加载这份权重。"
        "speculative.py 只在 model_type 为 longcat_flash 时自动改写，这里要显式传入 method。"
        "传入之后的 draft 与 Flash 相同：kv_b 展开后把 V 补到 qk_head_dim 写入 cache，o_proj 前再切回。"
        "只有 checkpoint 里有 model.mtp.layers.0 才建这一层。没有这些张量就不建，也不补 V。",
    ]
    return layers, mtp, notes, cov


def _build_longcat_ngram(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """LongCat-Flash-Lite. Same dual-sublayer decoder as Flash. Block count is ``num_layers``.

    vLLM ``FlashNgramModel`` overwrites ``num_hidden_layers`` with ``num_layers`` when the config
    class default (attention-module count) disagrees. Tables and projections are attached from the
    safetensors header in ``_attach_longcat_ngram``, not invented here.
    """
    if c.get("num_layers") is None:
        raise ValueError("longcat_flash_ngram block count is num_layers")
    return _build_longcat_flash(c, draft=False)


def _seq0(v, default: int = 0) -> int:
    """First entry of a per-layer list, or the int itself. Hunyuan's forward indexes ``[0]`` for every layer."""
    if isinstance(v, (list, tuple)):
        return int(v[0]) if v else default
    if v is None:
        return default
    return int(v)


def _build_hunyuan_v1_moe(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """Hunyuan-A13B text MoE (tencent/Hunyuan-A13B-Instruct, hunyuan.py + modeling_hunyuan.py).

    ``HunYuanMoEV1ForCausalLM`` does not build a vision tower. ``vit_type`` / ``vit_path`` on this release are null.
    ``use_cla`` and ``use_mla`` are off: every layer is GQA with its own K and V. CLA cross-layer KV and MLA are
    not implemented here.

    ``HunYuanAttention.head_dim`` is ``hidden // num_attention_heads`` (this release also stores ``head_dim`` and
    ``attention_head_dim``, both equal to that). ``use_qk_norm`` adds one RMSNorm(head_dim) on Q and one on K.
    Those weights are parameters only; the norm is not an extra cycle, same as Qwen3.

    Every layer is ``HunYuanMoE`` when ``num_experts > 1``. The decoder does not read ``moe_layer_num_skipped``.
    Each expert is ``HunYuanMLP(is_shared_mlp=False)``, whose width is ``intermediate_size``, not
    ``moe_intermediate_size``. ``use_mixed_mlp_moe`` adds one shared MLP of width
    ``intermediate_size * num_shared_expert[0]`` (the module indexes ``[0]`` for every layer) and adds it to the
    routed sum. There is no ``shared_expert_gate``. ``HunYuanTopKGate.wg`` is an fp32 Linear with no bias.
    ``topkgating`` reads ``moe_topk[0]`` on every layer. The selected-gate renormalization, dynamic NTK RoPE,
    eval-time dropout and ``routed_scaling_factor`` do not change GEMM or KV shape.
    """
    if c.get("use_cla"):
        raise ValueError("hunyuan_v1_moe use_cla is not modeled")
    if c.get("use_mla"):
        raise ValueError("hunyuan_v1_moe use_mla is not modeled")
    h = int(c["hidden_size"])
    H = int(c["num_attention_heads"])
    KV = int(c.get("num_key_value_heads") or H)
    hd = h // H
    L = int(c["num_hidden_layers"])
    E = int(c.get("num_experts") or 0)
    k = _seq0(c.get("moe_topk"), 1)
    de = int(c["intermediate_size"])
    moe_w = c.get("moe_intermediate_size")
    if isinstance(moe_w, (list, tuple)):
        if any(int(x) != de for x in moe_w):
            raise ValueError("hunyuan_v1_moe expert GEMM uses intermediate_size; moe_intermediate_size differs")
    elif moe_w is not None and int(moe_w) != de:
        raise ValueError("hunyuan_v1_moe expert GEMM uses intermediate_size; moe_intermediate_size differs")
    factor = _seq0(c.get("num_shared_expert"), 0)
    if c.get("use_mixed_mlp_moe") and factor:
        n_sh, d_sh = 1, de * factor
    else:
        n_sh, d_sh = 0, 0
    qk = 2 * hd if c.get("use_qk_norm") else 0
    layers: list[Layer] = []
    for _ in range(L):
        alin, core = _gqa(h, H, KV, hd)
        if E > 1:
            ffn, flin = _ffn_moe(h, E, k, de, n_sh, d_sh, shared_gate=False)
        else:
            ffn, flin = _ffn_dense(h, de)
        layers.append(Layer(alin, core, ffn, flin, 2 * h + qk))
    qk_note = ("q / k 为 RMSNorm(head_dim)，只进参数，不另计周期。" if qk
               else "use_qk_norm 关闭，没有 q/k norm。")
    if n_sh:
        mix = (f"use_mixed_mlp_moe：另有一份共享 SwiGLU，宽 intermediate_size·num_shared_expert[0] = {d_sh}，"
               "没有 shared_expert_gate。")
    else:
        mix = "use_mixed_mlp_moe 关闭，没有共享 MLP。"
    notes = [
        f"Hunyuan-A13B：{L} 层 GQA（{H}/{KV}，head {hd}）。{qk_note}",
        f"每层 {E} 个专家 top-{k}，专家宽 intermediate_size {de}"
        "（HunYuanMLP 不读 moe_intermediate_size）。" + mix,
        "router 是 gate.wg，按发布是 fp32，没有 bias。topkgating 对选中专家做一次归一化。"
        "use_cla 与 use_mla 关闭，每一层都写自己的 K/V。"
        "动态 NTK、eval dropout 和 routed_scaling_factor 不改变 GEMM 形状，也不改变 KV 宽度，不计额外周期。",
        "num_experts > 1 时每一层都是 MoE，decoder 不读 moe_layer_num_skipped。没有 MTP 张量。"
        "vit_type 与 vit_path 为空，因果类不建视觉塔。",
    ]
    return layers, [], notes, "full"


def _build_ernie4_5(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """Dense ERNIE-4.5 (ernie4_5).

    The 0.3B repos ship no modeling file. Geometry is the public transformers class
    ``Ernie4_5ForCausalLM`` (modeling_ernie4_5.py): GQA and SwiGLU linears take
    ``use_bias`` (false on the shipped checkpoints), two RMSNorms per block, a final
    RMSNorm, and a tied ``lm_head`` (no second tensor in the checkpoint). ``head_dim``
    is the config field. On 0.3B it is 128 while ``hidden_size // num_attention_heads``
    is 64, so ``q_proj`` is ``[2048, 1024]`` and ``k_proj`` / ``v_proj`` are
    ``[256, 1024]``. The class has no MTP module. Interleaved RoPE does not change
    GEMM shapes or the KV width.
    """
    if c.get("use_bias"):
        raise ValueError("ernie4_5 use_bias is not modeled")
    if c.get("num_nextn_predict_layers"):
        raise ValueError("ernie4_5 has no MTP module")
    layers, mtp, notes, cov = _build_std(c)
    h = int(c["hidden_size"])
    heads = int(c["num_attention_heads"])
    hd = int(c.get("head_dim") or (h // heads))
    width = f"（hidden/heads = {h // heads}）" if heads and hd != h // heads else ""
    notes = list(notes) + [
        f"ERNIE-4.5 稠密：GQA 投影宽度按 head_dim {hd}{width}。"
        "use_bias 为 false，没有投影 bias。公开类没有 MTP。没有视觉塔。"
        "交错 RoPE 不改变 GEMM 形状，也不改变 KV 宽度，不计额外周期。",
    ]
    return layers, mtp, notes, cov


def _build_ernie4_5_moe(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """ERNIE-4.5 MoE (ernie4_5_moe).

    The checkpoint repos have no modeling file. The geometry is the public transformers class
    ``Ernie4_5_MoeForCausalLM`` (modeling_ernie4_5_moe.py, transformers v5.12.0) checked against
    the safetensors headers.

    Decoder layers whose index satisfies the published interval test are ``Ernie4_5_MoeSparseMoeBlock``:
    fp32 ``gate.weight`` [E, hidden], fp32 ``moe_statics.e_score_correction_bias`` [1, E] added after softmax
    (selection only; gathered expert weights are the pre-bias softmax scores), routed SwiGLU experts of width
    ``moe_intermediate_size``, and, when ``moe_num_shared_experts > 0``, one shared SwiGLU of width
    ``moe_intermediate_size * moe_num_shared_experts``. A zero count builds no shared MLP
    (ERNIE-4.5-300B-A47B). There is no ``shared_expert_gate``. Other layers are a dense SwiGLU of
    ``intermediate_size``. The published interval test leaves layers before ``moe_layer_start_index`` dense
    (300B: layers 0-2, MoE from layer 3). Attention is bias-free GQA. ``use_bias`` is false on the shipped
    releases; a true flag is refused rather than guessed.

    ``_keys_to_ignore_on_load_unexpected = ["mtp"]`` and the class does not build an MTP module. When
    ``num_nextn_predict_layers`` is set, each stored block is a dense decoder layer (not a clone of the last
    MoE layer) plus ``mtp_linear_proj`` [hidden, 2·hidden], ``mtp_emb_norm`` and ``mtp_hidden_norm``. Those
    tensors use the existing MTP parameter path (eh_proj is 2h→h). A config that omits the field does not
    invent MTP layers.

    PaddlePaddle/ERNIE ``modeling_moe.py`` maps the config string ``moe_gate`` through
    ``gate_class = {top2_fused: TopKGate, topk: TopKGate}``. Both names are that fused top-k gate.
    ``MOELayer`` selects ``k = config.moe_k`` (6 on the 21B cards, not 2). ``moe_capacity`` is the
    dispatch buffer (``capacity = int(cap * num_tokens // num_experts)``); the class then dispatches
    with ``k`` and does not add a GEMM. A missing key is the transformers class, which has no
    ``moe_gate`` field and uses the same router. Any other string is refused.
    """
    if c.get("use_bias"):
        raise ValueError("ernie4_5_moe use_bias is not modeled")
    h = int(c["hidden_size"])
    H = int(c["num_attention_heads"])
    KV = int(c.get("num_key_value_heads") or H)
    if h % H:
        raise ValueError("ernie4_5_moe hidden_size is not divisible by num_attention_heads")
    hd = h // H
    L = int(c["num_hidden_layers"])
    E = int(c.get("moe_num_experts") or c.get("num_experts") or 0)
    k = int(c.get("moe_k") or c.get("num_experts_per_tok") or 0)
    de = int(c["moe_intermediate_size"])
    d_dense = int(c["intermediate_size"])
    n_cfg = int(c.get("moe_num_shared_experts") or 0)
    n_sh, d_sh = (1, de * n_cfg) if n_cfg else (0, 0)
    interval = int(c.get("moe_layer_interval") or 1)
    if interval <= 0:
        raise ValueError("ernie4_5_moe moe_layer_interval must be positive")
    start = int(c.get("moe_layer_start_index") or 0)
    end_raw = c.get("moe_layer_end_index")
    end = L - 1 if end_raw is None or int(end_raw) < 0 else int(end_raw)
    gate = c.get("moe_gate")
    if gate is not None and gate not in ("topk", "top2_fused"):
        raise ValueError("ernie4_5_moe moe_gate is not the published TopKGate")

    def moe_at(i: int) -> bool:
        return ((i + 1) % interval == 0) and start <= i <= end

    if any(moe_at(i) for i in range(L)) and E < 1:
        raise ValueError("ernie4_5_moe MoE layer has no experts")
    layers: list[Layer] = []
    n_moe = 0
    for i in range(L):
        alin, core = _gqa(h, H, KV, hd)
        if moe_at(i):
            ffn, flin = _ffn_moe(h, E, k, de, n_sh, d_sh, shared_gate=False)
            misc = 2 * h + E
            n_moe += 1
        else:
            ffn, flin = _ffn_dense(h, d_dense)
            misc = 2 * h
        layers.append(Layer(alin, core, ffn, flin, misc))
    # Stored MTP block: dense GEMMs, plus the two extra norms. eh_proj stays in mtp_extra (2·h·h).
    n_mtp = int(c.get("num_nextn_predict_layers") or 0)
    mtp: list[Layer] = []
    for _ in range(n_mtp):
        alin, core = _gqa(h, H, KV, hd)
        ffn, flin = _ffn_dense(h, d_dense)
        mtp.append(Layer(alin, core, ffn, flin, 4 * h))
    mix = (f"共享 SwiGLU 宽 moe_intermediate_size·moe_num_shared_experts = {d_sh}，没有 shared_expert_gate。"
           if n_sh else "没有共享专家。")
    notes = [
        f"ERNIE-4.5 MoE：{L} 层 GQA（{H}/{KV}，head {hd}）。"
        f"层号满足 (i+1) 整除 interval {interval} 且落在 [{start}, {end}] 的 {n_moe} 层是 MoE，其余是稠密 SwiGLU"
        f"（intermediate {d_dense}）。",
        f"MoE：{E} 个专家 top-{k}，专家宽 {de}。{mix}"
        "gate.weight 与 moe_statics.e_score_correction_bias 按发布是 fp32。"
        "bias 加在 softmax 之后，只参与选专家。",
        "选中专家的归一化、moe_norm_min 和 router 的 fp32 matmul 不改变 GEMM 形状，也不改变 KV 宽度，不计额外周期。",
    ]
    if n_mtp:
        notes.append(
            "发布类把 mtp 列在 _keys_to_ignore_on_load_unexpected，forward 不跑这份权重。"
            "header 里是一层与稠密层相同的 GQA + SwiGLU、eh_proj [hidden, 2·hidden]、mtp_emb_norm 与 mtp_hidden_norm。"
            "参数按这条形状记，不克隆最后一层 MoE，也不另写递推。没有视觉塔。")
    else:
        notes.append("num_nextn_predict_layers 未给出或为 0，不建模 MTP。没有视觉塔。")
    if gate is not None:
        notes.append(
            "moe_gate 是 PaddlePaddle/ERNIE modeling_moe.py 的 gate_class 字符串："
            "top2_fused 与 topk 都是 TopKGate。选中专家数仍是 moe_k，不是 2。"
            "moe_capacity 是 dispatch 缓冲，不另计 GEMM，也不计周期。")
    return layers, mtp, notes, "full"


def _pair_index(v, n: int, name: str) -> list[int]:
    if isinstance(v, (list, tuple)):
        if len(v) != n:
            raise ValueError(f"ernie4_5_moe_vl {name} must have {n} entries")
        return [int(x) for x in v]
    return [int(v)] * n


def _build_ernie4_5_moe_vl(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """ERNIE-4.5-VL MoE (ernie4_5_moe_vl).

    Public class ``Ernie4_5_VLMoeForConditionalGeneration`` (modeling_ernie4_5_vl.py on the
    28B-A3B-PT repo). Two routed SwiGLU pools share one layer. ``moe_intermediate_size[0]``
    is the text-expert width and ``[1]`` is the vision-expert width. ``moe_num_experts`` is
    a pair of equal group sizes (MoEStatics asserts that). A layer is MoE when
    ``(i + 1) % moe_layer_interval == 0`` and ``i`` is inside ``[min(start), max(end)]``.
    Other layers are a dense SwiGLU of ``intermediate_size``. A modality whose own
    ``[start, end]`` misses that layer is ``nn.Identity`` in the class; that shape is refused.

    The multimodal kernel (``MOEAllGatherLayerV2``, ``use_padding`` false) concatenates
    ``gate.weight`` and ``gate.weight_1`` and runs both on every token when any token type
    is non-zero, then keeps the text choice only where the type is 0. A text-only step is
    the class's ``mlp_text`` path: the first router and the first pool. ``Ernie4_5_VLMoEConfig``
    defaults ``moe_use_hard_gate`` to true and then ``Ernie4_5_MoEConfig`` overwrites it to
    false when the JSON key is absent, which leaves ``lm_gate`` as None. The weights and the
    multimodal forward still describe the split above. This builder bills that split. It does
    not bill the capacity-padded zeros of the text-only ``MOELayer`` kernel.

    Shared SwiGLU exists only when ``moe_num_shared_experts > 0``. Its width is the text
    expert width times that count. ``e_score_correction_bias`` is fp32 ``[2, E]`` when
    ``moe_use_aux_free`` is true (selection only). Attention is bias-free GQA with
    ``head_dim = hidden // heads``. The class builds no MTP module.     RoPE frequency split
    and 3-D positions do not change GEMM shapes or the KV width. Video frames are
    paired in the resampler and use a separate grid. Still images stay on the image grid.

    ``modeling_ernie4_5_vl.py`` maps ``moe_gate`` through ``gate_class = {top2: Top2Gate, topk: TopKGate}``.
    Shipped configs set ``topk`` (TopKGate, ``k = moe_k``). ``top2`` is the GShard top-2 gate and
    ``top2_fused`` is not a key on this class. Both are refused. A missing key keeps the current bill.
    """
    if c.get("use_bias"):
        raise ValueError("ernie4_5_moe_vl use_bias is not modeled")
    if c.get("num_nextn_predict_layers"):
        raise ValueError("ernie4_5_moe_vl has no MTP module")
    h = int(c["hidden_size"])
    H = int(c["num_attention_heads"])
    KV = int(c.get("num_key_value_heads") or H)
    if H < 1 or h % H:
        raise ValueError("ernie4_5_moe_vl hidden_size is not divisible by num_attention_heads")
    hd = h // H
    if c.get("head_dim") not in (None, hd):
        raise ValueError("ernie4_5_moe_vl head_dim is hidden // num_attention_heads")
    L = int(c["num_hidden_layers"])
    experts = _pair_index(c.get("moe_num_experts"), 2, "moe_num_experts")
    widths = _pair_index(c.get("moe_intermediate_size"), 2, "moe_intermediate_size")
    if experts[0] != experts[1] or experts[0] < 1 or widths[0] < 1 or widths[1] < 1:
        raise ValueError("ernie4_5_moe_vl expert groups must be equal and non-empty")
    k = int(c.get("moe_k") or 0)
    if k < 1:
        raise ValueError("ernie4_5_moe_vl moe_k must be positive")
    d_dense = int(c["intermediate_size"])
    n_cfg = int(c.get("moe_num_shared_experts") or 0)
    n_sh, d_sh = (1, widths[0] * n_cfg) if n_cfg else (0, 0)
    interval = int(c.get("moe_layer_interval") or 1)
    if interval <= 0:
        raise ValueError("ernie4_5_moe_vl moe_layer_interval must be positive")
    starts = _pair_index(c.get("moe_layer_start_index"), 2, "moe_layer_start_index")
    ends = _pair_index(c.get("moe_layer_end_index"), 2, "moe_layer_end_index")
    moe_lo, moe_hi = min(starts), max(ends)
    E, Eb = experts[0], experts[1]
    de, deb = widths[0], widths[1]
    bias_n = (2 * E) if c.get("moe_use_aux_free") else 0
    gate = c.get("moe_gate")
    if gate is not None and gate != "topk":
        raise ValueError("ernie4_5_moe_vl moe_gate is not the published TopKGate")

    def moe_at(i: int) -> bool:
        return ((i + 1) % interval == 0) and moe_lo <= i <= moe_hi

    layers: list[Layer] = []
    n_moe = 0
    for i in range(L):
        alin, core = _gqa(h, H, KV, hd)
        if moe_at(i):
            for pool, (st, en) in enumerate(zip(starts, ends)):
                if not (st <= i <= en):
                    raise ValueError("ernie4_5_moe_vl identity expert pool is not modeled")
            ffn, flin = _ffn_moe(h, E, k, de, n_sh, d_sh, shared_gate=False,
                                 n_experts_b=Eb, d_expert_b=deb)
            misc = 2 * h + bias_n
            n_moe += 1
        else:
            ffn, flin = _ffn_dense(h, d_dense)
            misc = 2 * h
        layers.append(Layer(alin, core, ffn, flin, misc))
    if n_moe < 1:
        raise ValueError("ernie4_5_moe_vl has no MoE layer")
    mix = (f"共享 SwiGLU 宽 {d_sh}（文本专家宽 × moe_num_shared_experts）。"
           if n_sh else "没有共享专家。")
    notes = [
        f"ERNIE-4.5-VL：{L} 层 GQA（{H}/{KV}，head {hd}）。"
        f"层号满足 (i+1) 整除 interval {interval} 且落在 [{moe_lo}, {moe_hi}] 的 {n_moe} 层是双池 MoE，"
        f"其余是稠密 SwiGLU（intermediate {d_dense}）。",
        f"文本池 {E} 个专家宽 {de}，视觉池 {Eb} 个专家宽 {deb}，都是 top-{k}。"
        f"{mix}两个 router 都是 hidden×{E}。"
        "有非文本 token 时两个 router 都对这一步的每个 token 计；文本 token 只走文本池，其余走视觉池。"
        "纯文本步只计文本 router 和文本池。",
        "gate.weight、gate.weight_1 与 e_score_correction_bias 按发布是 fp32。"
        "bias 加在 softmax 之后，只参与选专家。校正 bias 落在层 misc，记忆体计划仍按每参数 2 字节。",
        "公开 VL config 把 moe_use_hard_gate 默认写成 true，父类在 JSON 没有该键时又写成 false，"
        "文本路径的 lm_gate 因此是空的。这里按权重和多模态 forward 的分流计，不另造第三种路由。",
        "文本参考内核按 capacity 补零行，多模态内核 use_padding 为 false，按真实命中行算。"
        "这里两种步都按 top-k 的期望行数计，不把补零行算进 GEMM，也不计 capacity 周期。",
        "交错 RoPE 与 3D 位置不改变 GEMM 形状，也不改变 KV 宽度，不计额外周期。"
        "这个类没有 MTP。图像 token 含首尾标记，走视觉专家池。"
        "视频帧在 resampler 里两两配对，时间 GEMM 的行数减半，另走视频网格，不写进静止图像网格。",
    ]
    if gate is not None:
        notes.append(
            "moe_gate 是 topk。公开 VL modeling_ernie4_5_vl.py 的 gate_class 里 topk 是 TopKGate，"
            "top2 是 Top2Gate，没有 top2_fused。选中专家数仍是 moe_k。不把 top2 或 top2_fused 记成这条账单。")
    return layers, [], notes, "full"


def _builder(c: dict):
    mt = (c.get("model_type") or "").lower()
    if _is_longcat_omni(c):
        return _build_longcat_omni, "LongCat-Flash-Omni（主栈展开 KV；无 MTP；视听塔、音频码本和 24k 波形解码器另计）"
    if "LongcatFlashNgramForCausalLM" in (c.get("architectures") or []) or mt == "longcat_flash_ngram":
        return _build_longcat_ngram, "LongCat-Flash-Lite（n-gram 嵌入；不跑 MTP）"
    if _is_longcat_causal(c):
        return _build_longcat_causal, "LongCat（MLA 展开 KV；deepseek_yarn；MTP 按 vLLM longcat_flash_mtp）"
    if "LongcatFlashForCausalLM" in (c.get("architectures") or []) or mt == "longcat_flash":
        return _build_longcat_flash, "LongCat-Flash（主栈展开 KV；MTP 按 vLLM 把 V 补到 qk_head_dim）"
    if mt == "hunyuan_v1_moe":
        return _build_hunyuan_v1_moe, "GQA（q/k RMSNorm）+ 共享 MLP 的 MoE"
    if mt == "ernie4_5":
        return _build_ernie4_5, "GQA 稠密"
    if mt == "ernie4_5_moe_vl":
        return _build_ernie4_5_moe_vl, "GQA 双池 MoE（文本 / 视觉专家）"
    if mt == "ernie4_5_moe":
        # Shared MLP exists only when moe_num_shared_experts > 0 (21B). 300B ships 0.
        if int(c.get("moe_num_shared_experts") or 0) > 0:
            return _build_ernie4_5_moe, "GQA + 共享 MLP 的 MoE"
        return _build_ernie4_5_moe, "GQA MoE（无共享专家）"
    if mt in ("gemma4_text", "gemma4_unified_text"):
        moe = " MoE" if c.get("enable_moe_block") else ""
        return _build_gemma4, "Gemma-4 混合注意力（滑窗 + 全注意力）" + moe
    if mt == "beam":
        return _build_beam, "GQA（SWA + full NoPE，输出门）MoE"
    if mt == "nemotron_h":
        return _build_nemotron_h, "Nemotron-H 混合（Mamba-2 + GQA）latent MoE"
    if mt == "nemotron_h_puzzle":       # 0.289 Nemotron-3.5-Bolt: per-layer heterogeneous MoE (block_configs)
        return _build_nemotron_h, "Nemotron-H Puzzle 混合（Mamba-2 + GQA）异构 latent MoE"
    if mt == "dory":
        return _build_dory, "nGPT 循环 LM（SWA + full GQA，输出门）"
    if mt in ("muse_spark", "muse_spark_text"):
        return _build_muse_spark, "GQA（SWA + full，输出门）latent MoE"
    if mt == "rwkv7":
        return _build_rwkv7, "RWKV-7（广义 delta 规则线性循环，无 KV）"
    if mt == "rwkv8":
        return _build_rwkv8, "RWKV-8 Heron（RWKV-7 时间混合 + DeepEmbed" + (" + DeepEmbedAttention" if c.get("dea") else "") + "）"
    if mt == "whisper":
        return _build_whisper, "编码器-解码器（MHA 自注意力 + 交叉注意力）稠密"
    if mt == "stripedhyena2":
        return _build_hyena, "StripedHyena 2（Hyena SE / MR / LR + MHA）"
    if mt == "mimo_v2_flash":
        return _build_mimo_v2_flash, "混合注意力（SWA + full）MoE"
    if mt == "mimo_v2":
        return _build_mimo_v2, "混合注意力（SWA + full）MoE"
    if mt == "step3p5":
        return _build_step3p5, "混合注意力（滑窗头数不同 + 逐头门）MoE"
    if mt == "step5":
        return _build_step5, "混合注意力（SWA + DSA 区域稀疏）MoE"
    if mt in ("lfm2", "lfm2_moe"):
        return _build_lfm2, "LFM2 混合（短卷积 + GQA）" + (" MoE" if c.get("num_experts") else "")
    if mt.startswith("qwen4_exp"):
        return _build_qwen4exp, "混合注意力（GDN + QSA）+ 门控残差 / PLE"
    if mt.startswith("deepseek_v41"):
        return _build_dsv41_ref, "DeepSeek-V4.1 (CSA/MQA-512 + Engram)"
    if mt == "deepseek_v4":
        return _build_dsv4_ref, "DeepSeek-V4 (CSA/MQA-512)"
    if mt.startswith("deepseek_v4"):
        return _build_dsv4, "DeepSeek-V4 (CSA/MQA-512)"
    if mt == "raon1_mini":     # 0.320 KRAFTON Raon-1-mini: RDA (gated delta) + NoPE gated MLA, Raon_mHC, MoE
        return _build_hybrid, "混合注意力（RDA delta 规则 + NoPE 门控 MLA）mHC MoE"
    if mt == "minimax_m3_vl":
        return _build_minimax_m3, "MiniMax M3.1-Flash（GQA + 块级 MSA）MoE"
    if mt in ("kimi_linear", "bailing_hybrid", "bailing_moe_v3_vl", "ming_v3_asr", "bailing_moe_v3", "qwen3_next", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text",
              "qwen4_exp", "qwen4_exp_text", "minimax_text_01", "minimax_m1", "minimax_m2", "minimax") or mt.startswith("glm5_next"):
        return _build_hybrid, "混合注意力（linear + full）"
    if c.get("kv_lora_rank"):
        return _build_mla, "MLA" + (" + DSA" if c.get("index_n_heads") else "") + (" MoE" if c.get("n_routed_experts") else "")
    return _build_std, "GQA" + (" MoE" if _g(c, "num_local_experts", "num_experts", "n_routed_experts") else " 稠密")


# ---------------------------------------------------------------- release formats
_ROLE_KEYS = ("attn", "mlp", "expert", "shared_expert", "router", "embed", "lm_head", "mtp", "ngram", "other")


def _release_formats(rel: dict) -> dict[str, RoleFormat]:
    out = {}
    for role in _ROLE_KEYS:
        v = rel.get("roles", {}).get(role)
        if not v:
            continue
        fmt, e = max(v.items(), key=lambda kv: kv[1]["params"])
        p = sum(x["params"] for x in v.values())
        b = sum(x["bytes"] for x in v.values())
        if p:
            name = {"uint8": "int8", "int32": "int4"}.get(fmt, fmt)
            out[role] = RoleFormat(name if name in ("fp32", "bf16", "fp16", "fp8", "mxfp4", "nvfp4", "int8", "int4",
                                                    "iq2_xxs", "q8_0") else "bf16",     # 0.289 ggml block formats
                                   8.0 * b / p)
    return out


def _kv_fmt(rel: dict) -> str:
    qc = rel.get("quantization_config") or {}
    if isinstance(qc, dict) and qc.get("kv_cache_scheme"):
        return "fp8"
    return "bf16"


def _quant_release(rel: dict) -> bool:
    return bool(rel.get("quantization_config"))


def _act_fmt(rel: dict) -> str:
    """Activation format: W8A8 block-FP8 releases with dynamic activation scaling
    run activations in fp8 (DeepSeek reference inference); 0.251: static per-tensor activation scales
    (Hy-Memory-30B-A3B-FP8: input_scale per linear) quantise activations to fp8 the same way; all else bf16."""
    qc = rel.get("quantization_config") or {}
    if isinstance(qc, dict) and qc.get("quant_method") == "fp8" and qc.get("activation_scheme") in ("dynamic", "static"):
        return "fp8"
    # 0.254: compressed-tensors W8A8 float (FP8_DYNAMIC per-token / FP8_BLOCK per-group-128 input_activations, num_bits 8,
    # type float) — vLLM CompressedTensorsW8A8Fp8 quantises each GEMM input to fp8 「来源」 (was bf16 ≤ 0.253)
    if isinstance(qc, dict) and qc.get("quant_method") == "compressed-tensors" and any(
            (g.get("input_activations") or {}).get("num_bits") == 8 and (g.get("input_activations") or {}).get("type") == "float"
            for g in (qc.get("config_groups") or {}).values() if isinstance(g, dict)):
        return "fp8"
    return "bf16"


_QUARK_IN = {"fp8_e4m3": ("fp8", "fp8", 8.0), "fp4": ("mxfp4", "mxfp4", 4.25)}


def _quark_act_roles(qc: dict, fm: dict) -> dict:
    """0.281 AMD Quark exports (quant_method "quark"): global_quant_config.input_tensors is the GEMM input scheme of every
    quantized layer 「来源」 vLLM quark.py → QuarkW8A8Fp8 (input fp8_e4m3, is_dynamic, qscheme per_channel = per-token FP8,
    Fp8LinearOp act_quant_group_shape PER_TOKEN) and the OCP MX scheme (input fp4 group 32 e8m0 dynamic = W4A4 MXFP4; native
    on MI355 / gfx950, emulated by dequantising to bf16 elsewhere — gemm_exec picks the path from the chip's formats).
    Applies to the roles stored in the weight format only (exclude list: attention / dense / lm_head stay bf16 with bf16
    inputs); per-layer / per-type overrides (layer_quant_config, layer_type_quant_config) are not modelled — none of the
    catalog's Quark releases set them.  ≤ 0.280 every Quark role kept bf16 inputs."""
    g = qc.get("global_quant_config") or {}
    inp, w = g.get("input_tensors") or {}, g.get("weight") or {}
    if qc.get("layer_quant_config") or qc.get("layer_type_quant_config") or not inp.get("dtype") in _QUARK_IN:
        return {}
    a_fmt, w_fmt, bits = _QUARK_IN[inp["dtype"]]
    if _QUARK_IN.get(w.get("dtype"), (None, None))[1] != w_fmt:
        return {}                             # mixed W/A families (e.g. W4A8 MXFP4×FP8) not in the catalog
    return {"act_" + r: RoleFormat(a_fmt, bits) for r, f in fm.items()
            if f.fmt == w_fmt and r in ("attn", "mlp", "expert", "shared_expert", "lm_head")}


def _vision_formats(rel: dict, vs) -> dict:
    """0.285 quantized vision tower (release ``vision_fmt``, scripts/summarize_release.py): per tower-linear kind the
    stored weight format and bits (dominant format by params), and the GEMM input format from ModelOpt's per-role
    algorithms 「来源」 (as in ``_w4a4_roles``): FP8 / MXFP8 = W8A8 → fp8 inputs (MXFP8 e8m0 / 32 scales ≈ fp8 bytes
    「近似」); NVFP4 = W4A4 → nvfp4 inputs unless the role also lists W4A16_NVFP4 (weight-only) → bf16.  Stored bytes per
    tower parameter from the release's vision role.  Up to 0.284 the tower ran bf16 weights and inputs and was stored
    at 2 bytes / param."""
    qc = rel.get("quantization_config") or {}
    algos = set(((qc.get("algo_by_role") or {}) if isinstance(qc, dict) else {}).get("vision") or ())
    out = []
    for k, fs in sorted(rel["vision_fmt"].items()):
        f, x = max(fs.items(), key=lambda kv: kv[1]["params"])
        bits = 8.0 * sum(y["bytes"] for y in fs.values()) / sum(y["params"] for y in fs.values())
        if f == "fp8" and algos & {"FP8", "MXFP8"}:
            a = "fp8"
        elif f == "nvfp4" and "NVFP4" in algos and "W4A16_NVFP4" not in algos:
            a = "nvfp4"
        else:
            a = "bf16"
        out.append((k, f, bits, a))
    vr = (rel.get("roles") or {}).get("vision") or {}
    p, b = sum(x["params"] for x in vr.values()), sum(x["bytes"] for x in vr.values())
    return {"wfmt": tuple(out), "w_Bpp": b / p if p else 2.0}


def _compressed_tensors_w4a4(qc: dict, fm: dict) -> dict:
    """0.391: compressed-tensors MXFP4 / NVFP4 pack (Granite-4.2).

    One group, targets Linear, ignore lm_head, no KV scheme. MX is float 4-bit,
    group 32, uint8 e8m0 scales, dynamic inputs. NVFP4 is float 4-bit, group 16,
    e4m3 scales, strategy tensor_group, dynamic "local". Those are the existing
    mxfp4 (4.25 bit) and nvfp4 (4.5 bit) formats. gemm_exec already selects the
    native rate or the bf16 upcast. output_activations is null, so attention
    stays on the model activation format. Any other scheme keeps bf16 inputs.
    """
    if qc.get("quant_method") != "compressed-tensors":
        return {}
    if qc.get("ignore") != ["lm_head"] or qc.get("kv_cache_scheme") not in (None, {}):
        return {}
    groups = [g for g in (qc.get("config_groups") or {}).values() if isinstance(g, dict)]
    if len(groups) != 1:
        return {}
    g = groups[0]
    fmt = qc.get("format")
    if g.get("format") != fmt or g.get("targets") != ["Linear"] or g.get("output_activations") is not None:
        return {}
    w = g.get("weights") if isinstance(g.get("weights"), dict) else {}
    a = g.get("input_activations") if isinstance(g.get("input_activations"), dict) else {}

    def side(block: dict, group: int, scale: str, strategy: str, dynamic: object) -> bool:
        return (block.get("num_bits") == 4 and block.get("type") == "float"
                and block.get("group_size") == group and block.get("scale_dtype") == scale
                and block.get("strategy") == strategy and block.get("symmetric") is True
                and block.get("dynamic") == dynamic and block.get("zp_dtype") is None)

    if (fmt == "mxfp4-pack-quantized" and side(w, 32, "torch.uint8", "group", False)
            and side(a, 32, "torch.uint8", "group", True)):
        name, bits = "mxfp4", 4.25
    elif (fmt == "nvfp4-pack-quantized" and side(w, 16, "torch.float8_e4m3fn", "tensor_group", False)
            and side(a, 16, "torch.float8_e4m3fn", "tensor_group", "local")):
        name, bits = "nvfp4", 4.5
    else:
        return {}
    return {"act_" + r: RoleFormat(name, bits) for r, f in fm.items()
            if f.fmt == name and r in ("attn", "mlp", "expert", "shared_expert")}


def _w4a4_roles(rel: dict, fm: dict) -> dict:
    """0.254: ModelOpt NVFP4 (hf_quant_config quant_algo "NVFP4"; config_groups input_activations num_bits 4) is W4A4 —
    vLLM ModelOptNvFp4 / SGLang modelopt_fp4 quantise each NVFP4 GEMM's input to NVFP4 (global input_scale, e4m3 per 16)
    「来源」; "W4A16_NVFP4" (Dory) stays weight-only.  The activation format applies only to the roles stored in NVFP4
    (excluded modules — attention, dense layers, router, lm_head — keep bf16 inputs).  Was bf16 for all roles ≤ 0.253."""
    qc = rel.get("quantization_config") or {}
    ct = _compressed_tensors_w4a4(qc, fm) if isinstance(qc, dict) else {}
    if ct:
        return ct
    abr = qc.get("algo_by_role") if isinstance(qc, dict) else None
    if isinstance(abr, dict) and qc.get("quant_algo") == "MIXED_PRECISION":
        # 0.280 ModelOpt MIXED_PRECISION (modelopt / modelopt_mixed): the activation format follows each role's per-layer
        # algorithm 「来源」 ModelOpt quant_algo names: "NVFP4" = W4A4 (input_scale per linear in the release),
        # "W4A16_NVFP4" = weight-only, "FP8" / "MXFP8" = W8A8 (MXFP8 activation e8m0 per-32 scales ≈ fp8 bytes 「近似」).
        # Applies to roles stored in that format only; ≤ 0.279 every mixed role kept bf16 inputs.
        out = {}
        for r, al in abr.items():
            f = fm.get(r)
            if f is None or r not in ("attn", "mlp", "expert", "shared_expert", "lm_head"):
                continue
            if set(al) == {"NVFP4"} and f.fmt == "nvfp4":
                out["act_" + r] = RoleFormat("nvfp4", 4.5)
            elif set(al) <= {"FP8", "MXFP8"} and f.fmt == "fp8":
                out["act_" + r] = RoleFormat("fp8", 8.0)
        return out
    if isinstance(qc, dict) and qc.get("quant_method") == "quark":
        return _quark_act_roles(qc, fm)
    if isinstance(qc, dict) and qc.get("quant_method") == "bsgemm-moe" and qc.get("activation_scheme") in ("dynamic", "static"):
        # 0.281 Drama 3 "bsgemm-moe": block-FP8 (128×128) routed experts with dynamic activation scaling, served by vLLM
        # block-FP8 MoE kernels (vllm-omni recipes/fishaudio/Drama-3.md: "block-FP8 routed experts") 「来源」 → the fp8
        # expert GEMMs take fp8 inputs (per-token-group dynamic, same as quant_method "fp8" + activation_scheme dynamic);
        # every other linear is bf16 and keeps bf16 inputs.  Was bf16 for all roles ≤ 0.280.
        return {"act_" + r: RoleFormat("fp8", 8.0) for r, f in fm.items()
                if f.fmt == "fp8" and r in ("attn", "mlp", "expert", "shared_expert", "lm_head")}
    if not isinstance(qc, dict) or qc.get("quant_method") != "modelopt":
        return {}
    a4 = qc.get("quant_algo") == "NVFP4" or any(
        (g.get("input_activations") or {}).get("num_bits") == 4 for g in (qc.get("config_groups") or {}).values()
        if isinstance(g, dict))
    if not a4:
        return {}
    return {"act_" + r: RoleFormat("nvfp4", 4.5) for r, f in fm.items()
            if f.fmt == "nvfp4" and r in ("attn", "mlp", "expert", "shared_expert")}


# ---------------------------------------------------------------- public API
def release_path(hf_id: str) -> Path:
    return RELEASES / (hf_id.replace(":", "_").replace("/", "__") + ".json")


@lru_cache(maxsize=None)
def load_release(hf_id: str) -> dict | None:
    if "+" in hf_id:
        # 0.175: "<target release>+<draft repo>" = the target as released, served with a separate draft model
        # (accel_dse/data/drafts/*.json: public config + safetensors header of the draft repo)
        import copy
        base, draft = hf_id.split("+", 1)
        rel, dp = load_release(base), RELEASES.parent / "drafts" / (draft.replace("/", "__") + ".json")
        if rel is None or not dp.exists():
            return None
        rel, d = copy.deepcopy(rel), json.loads(dp.read_text())
        rel.update(repo=hf_id, params_mtp=d["params"], bytes_mtp=d["bytes"], dspark=d)
        rel["roles"]["mtp"] = {"bf16": {"params": d["params"], "bytes": d["bytes"]}}
        return rel
    p = release_path(hf_id)
    return json.loads(p.read_text()) if p.exists() else None


def _coverage_reasons(c: dict, spec: ModelSpec, gap: float) -> tuple[str, ...]:
    """Concrete per-model list of approximated mechanisms (shown with the coverage label)."""
    mt = c.get("model_type", "")
    L = len(spec.layers)
    r = []
    cmp = sum(1 for l in spec.layers if l.core.compress > 1 and not (l.core.cmp_coff or l.core.dual))
    idx = [l.core for l in spec.layers if l.core.idx_heads]
    if cmp:
        r.append(f"压缩稀疏注意力 CSA（{cmp}/{L} 层）：按有效上下文 ctx/压缩比 + 滑窗近似"
                 + (f"，索引层只读 top-{idx[0].topk} 个 token" if idx else ""))
    if c.get("num_hash_layers") and mt != "deepseek_v4":
        r.append(f"哈希路由（前 {c['num_hash_layers']} 层）：按 top-k MoE 计")
    hc = next((c.get(k) for k in ("hc_mult", "hc_count", "mhc") if c.get(k)), None)
    if hc and not any(l.hc or l.gres_n for l in spec.layers):
        r.append(f"超连接（多流残差 ×{hc}）：参数计入，多流混合计算未计")
    if (spec.lookup_params and not any(l.engram_cols or l.ple_rows or l.deepemb for l in spec.layers)
            and mt not in ("whisper", "moshi", "moss_tts_delay", "rwkv8") and not c.get("cosyvoice") and spec.fish is None
            and getattr(spec, "qwen3_tts", None) is None
            and getattr(spec, "moss_tts", None) is None
            and not spec.ple_packed and not spec.ngram_dim and not spec.audio_codebooks):  # 0.274 Moshi / 0.338 Qwen3-TTS / 0.346 MOSS / 0.361 RWKV-8 / 0.369 Gemma-4 PLE / 0.400 LongCat n-gram / 0.402 LongCat Omni codebooks
        r.append(f"n-gram / engram 查表（{spec.lookup_params / 1e9:.1f}B 参数）：计存储，每 token 只读少量行")
    if getattr(spec, "moss_tts", None) is not None:
        r.append("外置 audio tokenizer / vocoder 不在本权重：波形解码未计「来源」")
    if spec.fish is not None:
        if not spec.fish.codec:
            r.append("Drama 3：Stage 1 的 Fish DAC 解码器（codec.pth，torch pickle）未读、未计时")     # 0.276
    gg = sorted({f.fmt for _, f in spec.formats if f.fmt in ("iq2_xxs", "q8_0")})
    if gg:     # 0.289 ggml block formats (Nemotron-3.5-Bolt-24GB): weight-only; the served kernels (vLLM b12x linear / MoE backends) are not modelled per format
        # 0.290: the dequant op count is the ggml reference dequant's (dtypes.DEQUANT_OPS); the served fused kernels
        # (vLLM b12x / llama.cpp MMQ: dot products on the packed values) are not modelled per format
        r.append(f"ggml 块格式（{' / '.join(gg)}）：按仅权重量化计，反量化到 bf16 GEMM（IQ2_XXS 每元素 3.66 次向量操作、"
                 f"Q8_0 2 次，按 ggml 参考反量化「推导」）；服务端融合内核未逐格式建模「近似」")
    # 0.358: Whisper encoder-decoder gap is intentional (params() includes encoder after vision_params is set)
    if abs(gap) > 0.02 and mt != "whisper":
        r.append(f"参数与发布相差 {gap:+.1%}：模板未复现的部分按发布计存储")
    return tuple(r)


_MTP_LO = {"fp8", "nvfp4", "mxfp4", "int8", "int4", "uint8", "iq2_xxs", "q8_0"}
_MTP_HI = {"bf16", "fp16", "fp32"}


def _mtp_extra_format(rel: dict, spec: "ModelSpec") -> RoleFormat | None:
    """0.364: bill ``mtp.eh_proj`` / ``mtp_extra`` at the unquantized bucket when that bucket is exactly the
    remainder of the mtp role.

    ``_release_formats`` stores one dominant format and one average bit-width for the whole role. On a block-FP8
    release the MTP decoder GEMMs are fp8 and ``mtp.fc`` (eh_proj) plus its norms stay bf16
    (``modules_to_not_convert``). The average (~9 bit, named fp8) was applied to eh_proj and to ``mtp_extra``
    storage, about half the true bf16 weight traffic. The decoder GEMMs are already costed with the main stack's
    attn / mlp / expert formats, so they must not be inside this role's bit-width.

    Replace the role only when the low-precision params equal the MTP layers' GEMM params and the high-precision
    params equal ``mtp_extra`` + layer misc, parameter for parameter. A mixed decoder (DeepSeek-V3 / V4: bf16
    weights inside the MTP block) keeps the historical average. Activation format is untouched: ``quant_method``
    fp8 stays a global fp8 input, same as ``lm_head``.
    """
    role = (rel.get("roles") or {}).get("mtp") or {}
    if not spec.mtp_layers or not role:
        return None
    lo_p = hi_p = hi_b = 0
    hi_fmt, hi_best = None, -1
    for fmt, v in role.items():
        p, b = int(v.get("params") or 0), int(v.get("bytes") or 0)
        if not p:
            continue
        if fmt in _MTP_LO:
            lo_p += p
        elif fmt in _MTP_HI:
            hi_p += p
            hi_b += b
            if p > hi_best:
                hi_best, hi_fmt = p, fmt
        else:
            return None
    if not lo_p or not hi_p or hi_fmt is None:
        return None
    h = spec.hidden
    gemm = sum(L.params(h) - L.misc_params for L in spec.mtp_layers)
    misc = sum(L.misc_params for L in spec.mtp_layers)
    extra = spec.mtp_extra_params * len(spec.mtp_layers)
    if lo_p != gemm or hi_p != extra + misc:
        return None
    name = {"uint8": "int8", "int32": "int4"}.get(hi_fmt, hi_fmt)
    if name not in ("fp32", "bf16", "fp16"):
        return None
    return RoleFormat(name, 8.0 * hi_b / hi_p)


def _gdn_bf16_attn_split(rel: dict, spec: "ModelSpec") -> tuple[RoleFormat, RoleFormat] | None:
    """0.365: bill GDN ``in_proj_ba`` at bf16 when that tensor is exactly the GEMM part of the attn role's bf16 bucket.

    ``_release_formats`` stores one dominant format and one average bit-width for the whole attn role. On
    Qwen3-Coder-Next-FP8 and Qwen3.8-27B-FP8 the block-FP8 GEMMs share that role with the unquantized bf16 pieces
    (``modules_to_not_convert``: ``linear_attn.conv1d`` and ``linear_attn.in_proj_ba``, plus ``A_log`` / ``dt_bias``).
    The average (~8.03 bit, still named fp8) was applied to every attention GEMM, so ``lin_ba`` (bf16) was billed
    near 8 bit. conv / ``A_log`` / ``dt_bias`` are already in ``misc_params`` at 2 bytes, and their bytes also sat
    in that average. Net overcount on the main stack is about 1.2 MB and 2.0 MB.

    Replace the role only when the published buckets match the builder parameter for parameter:
    the low-precision params equal every main-stack attention GEMM except ``lin_ba``, and the bf16 params equal
    ``lin_ba`` + depthwise conv (``conv_channels * conv_kernel``) + ``A_log`` + ``dt_bias`` (``2 * n_state_heads``,
    the non-norm part of ``_gdn``'s misc). The bf16 bucket must be exactly 2 bytes per param (no scale smear).
    One low format and one bf16 format only. An MTP layer that still has ``lin_ba`` keeps the historical average,
    because those weights sit in the mtp role and this split does not describe them. Activation format is
    untouched: ``quant_method`` fp8 stays a global fp8 input, same as ``lm_head``.
    """
    role = (rel.get("roles") or {}).get("attn") or {}
    if "bf16" not in role or len(role) != 2:
        return None
    lo_fmt = next(k for k in role if k != "bf16")
    if lo_fmt not in _MTP_LO:
        return None
    if any(l.name == "lin_ba" for L in spec.mtp_layers for l in L.attn_linears):
        return None
    lin_ba = conv = alog = gemm_lo = 0
    for L in spec.layers:
        bas = [l for l in L.attn_linears if l.name == "lin_ba"]
        if L.core.lin == "gdn":
            if len(bas) != 1:
                return None
            lin_ba += bas[0].params
            conv += L.core.conv_channels * L.core.conv_kernel
            alog += 2 * L.core.n_state_heads
        elif bas:
            return None
        gemm_lo += sum(l.params for l in L.attn_linears if l.name != "lin_ba")
    if not lin_ba:
        return None
    lo, hi = role[lo_fmt], role["bf16"]
    lo_p, hi_p = int(lo.get("params") or 0), int(hi.get("params") or 0)
    lo_b, hi_b = int(lo.get("bytes") or 0), int(hi.get("bytes") or 0)
    if lo_p != gemm_lo or hi_p != lin_ba + conv + alog or hi_b != 2 * hi_p or not lo_p:
        return None
    name = {"uint8": "int8", "int32": "int4"}.get(lo_fmt, lo_fmt)
    if name not in ("fp32", "bf16", "fp16", "fp8", "mxfp4", "nvfp4", "int8", "int4", "iq2_xxs", "q8_0"):
        return None
    return RoleFormat(name, 8.0 * lo_b / lo_p), RoleFormat("bf16", 16.0)


def _o_proj_bf16_attn_split(rel: dict, spec: "ModelSpec") -> tuple[RoleFormat, RoleFormat] | None:
    """0.373: bill attention ``o`` at bf16 when that GEMM plus the learned sinks are the attn role's bf16 bucket.

    ``_release_formats`` stores one dominant format and one average bit-width for the whole attn role. On
    MiMo-V2-Flash and MiMo-V2.6 the block-fp8 stack leaves every ``o_proj`` in bf16 (``ignored_layers``) and the
    SWA sinks are bf16 vectors of width ``num_attention_heads``. The average (still named fp8) was applied to
    ``o``, so that GEMM ran native fp8×fp8. q/k/v stay on the low-precision bucket, block scales included.
    The bf16 bucket must be exactly 2 bytes per param. One ``o`` per main-stack layer, one low format, and bf16.
    MTP ``o`` lives in the mtp role and is not part of this identity.
    """
    role = (rel.get("roles") or {}).get("attn") or {}
    if "bf16" not in role or len(role) != 2 or not spec.layers:
        return None
    lo_fmt = next(k for k in role if k != "bf16")
    if lo_fmt not in _MTP_LO:
        return None
    o_p = sinks = gemm_lo = 0
    for L in spec.layers:
        named = [l for l in L.attn_linears if l.name == "o"]
        if len(named) != 1:
            return None
        o_p += named[0].params
        if L.core.sink:
            sinks += L.core.n_q
        gemm_lo += sum(l.params for l in L.attn_linears if l.name != "o")
    if not o_p:
        return None
    lo, hi = role[lo_fmt], role["bf16"]
    lo_p, hi_p = int(lo.get("params") or 0), int(hi.get("params") or 0)
    lo_b, hi_b = int(lo.get("bytes") or 0), int(hi.get("bytes") or 0)
    if lo_p != gemm_lo or hi_p != o_p + sinks or hi_b != 2 * hi_p or not lo_p:
        return None
    name = {"uint8": "int8", "int32": "int4"}.get(lo_fmt, lo_fmt)
    if name not in ("fp32", "bf16", "fp16", "fp8", "mxfp4", "nvfp4", "int8", "int4", "iq2_xxs", "q8_0"):
        return None
    return RoleFormat(name, 8.0 * lo_b / lo_p), RoleFormat("bf16", 16.0)


def _mtp_oproj_bf16_format(rel: dict, spec: "ModelSpec") -> RoleFormat | None:
    """0.373: bill ``mtp.eh_proj`` at bf16 when the mtp bf16 bucket is exactly ``o`` + eh_proj + layer misc.

    0.364 replaces the mtp role only when the low-precision params equal every MTP GEMM. MiMo also leaves MTP
    ``o_proj`` in bf16, so that matcher does not fire and the role average (~9.6 bit, named fp8) was applied to
    eh_proj. Decoder GEMMs other than ``o`` stay on the main stack's attn / mlp formats. This function does not
    retag ``o``. The caller retags MTP ``o`` whenever this identity holds, and retags the main stack only
    together with ``_o_proj_bf16_attn_split``.
    """
    role = (rel.get("roles") or {}).get("mtp") or {}
    if not spec.mtp_layers or "bf16" not in role or len(role) != 2:
        return None
    lo_fmt = next(k for k in role if k != "bf16")
    if lo_fmt not in _MTP_LO:
        return None
    h = spec.hidden
    o_p = sum(l.params for L in spec.mtp_layers for l in L.attn_linears if l.name == "o")
    if not o_p:
        return None
    gemm = sum(L.params(h) - L.misc_params for L in spec.mtp_layers)
    misc = sum(L.misc_params for L in spec.mtp_layers)
    extra = spec.mtp_extra_params * len(spec.mtp_layers)
    lo, hi = role[lo_fmt], role["bf16"]
    lo_p, hi_p = int(lo.get("params") or 0), int(hi.get("params") or 0)
    hi_b = int(hi.get("bytes") or 0)
    if lo_p != gemm - o_p or hi_p != extra + misc + o_p or hi_b != 2 * hi_p:
        return None
    return RoleFormat("bf16", 16.0)


def _retag_o_proj(layers: tuple) -> tuple:
    """Point each attention ``o`` at the bf16 ``o_proj`` role. Other linears keep their role."""
    return tuple(
        replace(L, attn_linears=tuple(
            replace(l, role="o_proj") if l.name == "o" else l for l in L.attn_linears))
        if any(l.name == "o" for l in L.attn_linears) else L
        for L in layers)


def _fp8_block_bytes(k: int, n: int, block: int) -> int:
    """Stored bytes of one block-fp8 weight: the payload plus one fp32 scale per tile."""
    return k * n + 4 * math.ceil(k / block) * math.ceil(n / block)


def _role_cells(rel: dict, role: str) -> dict:
    return dict((rel.get("roles") or {}).get(role) or {})


def _cell_pb(cells: dict, fmt: str) -> tuple[int, int]:
    cell = cells.get(fmt) or {}
    return int(cell.get("params") or 0), int(cell.get("bytes") or 0)


def _is_longcat_ngram(c: dict) -> bool:
    archs = c.get("architectures") or []
    mt = c.get("model_type") or ""
    return "LongcatFlashNgramForCausalLM" in archs or mt == "longcat_flash_ngram"


def _longcat_ngram_layout(c: dict, hidden: int, vocab: int) -> tuple[int, int, int, int, int, int]:
    """HF ``NgramEmbedding``: ``num = k*(n-1)`` tables, row ``int(m + 2*i + 1)``, width ``hidden // num``.

    Returns ``(order, splits, dim, num, table_params, proj_params)``.
    """
    ratio = c.get("ngram_vocab_size_ratio")
    k = c.get("emb_split_num")
    n = c.get("emb_neighbor_num")
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or ratio <= 0:
        raise ValueError("longcat ngram vocab ratio is missing")
    if isinstance(k, bool) or not isinstance(k, int) or isinstance(n, bool) or not isinstance(n, int):
        raise ValueError("longcat ngram emb_split_num / emb_neighbor_num are not ints")
    if k < 1 or n < 2:
        raise ValueError("longcat ngram needs at least one embedder")
    num = k * (n - 1)
    if hidden % num:
        raise ValueError("longcat ngram hidden size is not divisible by the embedder count")
    dim = hidden // num
    m = ratio * vocab
    rows = [int(m + i * 2 + 1) for i in range(num)]
    if any(r <= 0 for r in rows):
        raise ValueError("longcat ngram table row count is not positive")
    table = sum(r * dim for r in rows)
    proj = num * dim * hidden
    return n, k, dim, num, table, proj


def _require_pure(rel: dict, role: str, fmt: str, params: int, byte_per: int) -> None:
    got = _role_cells(rel, role)
    if set(got) != {fmt}:
        raise ValueError(f"longcat ngram {role} is not pure {fmt}")
    p, b = _cell_pb(got, fmt)
    if p != params or b != byte_per * p:
        raise ValueError(f"longcat ngram {role} bytes do not match {byte_per} per parameter")


def _retag_kv_a(layers: tuple) -> tuple:
    return tuple(
        replace(L, attn_linears=tuple(
            replace(lin, role="kv_a") if lin.name == "kv_a" else lin for lin in L.attn_linears))
        for L in layers)


def _attach_longcat_ngram(spec: "ModelSpec", rel: dict, c: dict) -> "ModelSpec":
    """0.400: bill LongCat-Flash-Lite n-gram tables as lookups and the projections as bf16 GEMMs.

    HF ``NgramEmbedding.forward`` adds the word embedding and each projected n-gram row, then
    divides by ``1 + k*(n-1)``. vLLM concatenates the tables, runs one ``bmm``, and takes the
    mean of the word row plus those projections. The arithmetic matches. Both drop ``model.mtp.*``,
    so a checkpoint MTP module stays in ``params_mtp`` and is not a layer.

    A fp8 config must be e4m3, dynamic, block ``[128, 128]``, with ``disable_quant_module`` empty
    (attention GEMMs are quantized, unlike Chat-FP8). Ignored names must not include a main-stack
    expert, dense, or attention GEMM. ``kv_a`` is its own role when its scale tiles are not the
    uniform 128 ratio. N-gram tables and projections stay bf16 because they are ignored and the
    projection input is the bf16 embedding row. A byte mismatch raises.
    """
    if not spec.layers or len(spec.layers) % 2 or spec.mtp_layers:
        raise ValueError("longcat ngram builder did not emit paired blocks without MTP")
    n_blocks = len(spec.layers) // 2
    h = spec.hidden
    order, splits, dim, num, table, proj = _longcat_ngram_layout(c, h, spec.vocab)
    _require_pure(rel, "embed", "bf16", spec.embed_params + table, 2)
    _require_pure(rel, "ngram", "bf16", proj, 2)
    _require_pure(rel, "lm_head", "bf16", spec.vocab * h, 2)
    moe0 = spec.layers[0]
    if moe0.ffn.kind != "moe" or not moe0.ffn.gated:
        raise ValueError("longcat ngram block 0 is not a SwiGLU MoE")
    n_exp = moe0.ffn.n_experts
    d_exp = moe0.ffn.d_expert
    d_ff = moe0.ffn.d_ff
    zero = moe0.ffn.zero_experts
    q_lora = next(lin.n for lin in moe0.attn_linears if lin.name == "q_a")
    kv_n = next(lin.n for lin in moe0.attn_linears if lin.name == "kv_a")
    rope = moe0.core.rope_dim
    attn_misc = 2 * h + q_lora + (kv_n - rope)
    norm_p = 2 * n_blocks * attn_misc + h
    _require_pure(rel, "norm", "bf16", norm_p, 2)
    pool = n_exp + zero
    _require_pure(rel, "router", "fp32", n_blocks * (h * pool + pool), 4)
    io = tuple(Linear(f"ngram_{i}", dim, h, "rep", role="ngram") for i in range(num))
    fm = dict(spec.formats)
    fm["ngram"] = RoleFormat("bf16", 16.0)
    fm["act_ngram"] = RoleFormat("bf16", 16.0)
    note = (
        f"N-gram 嵌入：order {order}，splits {splits}，{num} 张表，行宽 {dim}。"
        f"第 i 张是 Embedding(int(ratio*vocab + 2*i + 1), {dim})，共 {table} 参数，bf16，只查行。"
        f"每张表一个 Linear({dim}→{h}, bias=False)，共 {proj} 参数。",
        "HF forward 把词嵌入与各投影相加，再除以 1+k*(n-1)。"
        "公开 vLLM 把表拼成一张 VocabParallelEmbedding，投影做成一次 bmm，再对词嵌入和投影求均值。"
        "两边的算术相同。投影在每个 TP rank 上是完整参数，表按词表切分。",
        "哈希按 HF _get_ngram_ids：每个 order 一次右移，k 路多项式乘加再取模。"
        "词表模数在加载时算好。不把 n-1 个历史 token 记成常驻状态。",
        "HF 与 vLLM 都丢掉 model.mtp.*。checkpoint 里的 MTP 留在 params_mtp，不建层，也不跑 draft。",
    )
    spec = replace(
        spec, lookup_params=table, lookup_bits=16.0, ngram_order=order, ngram_splits=splits,
        ngram_dim=dim, io_pre=io, formats=tuple(sorted(fm.items())), notes=spec.notes + note)
    qc = c.get("quantization_config")
    if not isinstance(qc, dict):
        qc = rel.get("quantization_config")
    if not isinstance(qc, dict) or not qc:
        one_exp = n_exp * (2 * h * d_exp + d_exp * h)
        _require_pure(rel, "expert", "bf16", n_blocks * one_exp, 2)
        one_m = 2 * (h * 2 * d_ff + d_ff * h)
        _require_pure(rel, "mlp", "bf16", n_blocks * one_m, 2)
        attn_p = sum(lin.params for L in spec.layers for lin in L.attn_linears)
        _require_pure(rel, "attn", "bf16", attn_p, 2)
        return spec
    return _apply_longcat_ngram_fp8(spec, rel, c, qc)


_NGRAM_GEMM_RE = re.compile(
    r"^model\.layers\.\d+\.(?:mlp\.experts\.\d+|mlps\.[01]|self_attn\.[01])\."
    r"(?:gate_proj|up_proj|down_proj|q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)$")


def _apply_longcat_ngram_fp8(spec: "ModelSpec", rel: dict, c: dict, qc: dict) -> "ModelSpec":
    """Bill Lite-FP8 from the block formula. Attention is quantized. kv_a splits when its tiles differ."""
    if (qc.get("quant_method") != "fp8" or qc.get("fmt") != "e4m3"
            or qc.get("activation_scheme") != "dynamic"):
        raise ValueError("longcat ngram fp8 scheme is not e4m3 with dynamic activations")
    if list(qc.get("weight_block_size") or []) != [128, 128]:
        raise ValueError("longcat ngram fp8 weight_block_size is not [128, 128]")
    if c.get("disable_quant_module") != []:
        raise ValueError("longcat ngram fp8 disable_quant_module is not empty")
    ignored = qc.get("ignored_layers")
    if not isinstance(ignored, list) or not ignored:
        raise ValueError("longcat ngram fp8 ignored_layers is missing")
    if any(isinstance(name, str) and _NGRAM_GEMM_RE.match(name) for name in ignored):
        raise ValueError("longcat ngram fp8 ignored_layers names a main-stack GEMM")
    num = spec.ngram_splits * (spec.ngram_order - 1)
    need = {f"model.ngram_embeddings.embedders.{i}" for i in range(num)}
    need |= {f"model.ngram_embeddings.post_projs.{i}" for i in range(num)}
    need |= {"lm_head", "model.embed_tokens", "model.norm"}
    got_names = {name for name in ignored if isinstance(name, str)}
    if not need <= got_names:
        raise ValueError("longcat ngram fp8 ignored_layers misses embed, ngram, or lm_head")
    h = spec.hidden
    n_blocks = len(spec.layers) // 2
    moe0 = spec.layers[0]
    n_exp, d_exp, d_ff = moe0.ffn.n_experts, moe0.ffn.d_expert, moe0.ffn.d_ff
    block = 128
    gate_p, down_p = h * d_exp, d_exp * h
    gate_b, down_b = _fp8_block_bytes(h, d_exp, block), _fp8_block_bytes(d_exp, h, block)
    if gate_b * down_p != down_b * gate_p:
        raise ValueError("longcat ngram expert gate and down fp8 scales do not share one ratio")
    one_p, one_b = 2 * gate_p + down_p, 2 * gate_b + down_b
    exp_p, exp_b = n_blocks * n_exp * one_p, n_blocks * n_exp * one_b
    got_e = _role_cells(rel, "expert")
    if set(got_e) != {"fp8"} or _cell_pb(got_e, "fp8") != (exp_p, exp_b):
        raise ValueError("longcat ngram expert bytes do not match block fp8")
    up_b = _fp8_block_bytes(h, 2 * d_ff, block)
    if up_b != 2 * _fp8_block_bytes(h, d_ff, block):
        raise ValueError("longcat ngram fused gate_up scale bytes do not equal separate gate and up")
    dn_b = _fp8_block_bytes(d_ff, h, block)
    up_p, dn_p = h * 2 * d_ff, d_ff * h
    if up_b * dn_p != dn_b * up_p:
        raise ValueError("longcat ngram dense gate_up and down fp8 scales do not share one ratio")
    mlp_p = n_blocks * 2 * (up_p + dn_p)
    mlp_b = n_blocks * 2 * (up_b + dn_b)
    got_m = _role_cells(rel, "mlp")
    if set(got_m) != {"fp8"} or _cell_pb(got_m, "fp8") != (mlp_p, mlp_b):
        raise ValueError("longcat ngram dense MLP bytes do not match block fp8")
    attn_rows: list[tuple[int, int]] = []
    kv_rows: list[tuple[int, int]] = []
    for L in spec.layers:
        for lin in L.attn_linears:
            row = (lin.params, _fp8_block_bytes(lin.k, lin.n, block))
            if lin.name == "kv_a":
                kv_rows.append(row)
            else:
                attn_rows.append(row)
    if not attn_rows or not kv_rows:
        raise ValueError("longcat ngram attention is missing kv_a or another projection")

    def _ratio(p: int, b: int) -> tuple[int, int]:
        g = math.gcd(b, p)
        return b // g, p // g

    if len({_ratio(p, b) for p, b in attn_rows}) != 1:
        raise ValueError("longcat ngram attention fp8 scales do not share one ratio")
    if len({_ratio(p, b) for p, b in kv_rows}) != 1:
        raise ValueError("longcat ngram kv_a fp8 scales do not share one ratio")
    split = _ratio(*kv_rows[0]) != _ratio(*attn_rows[0])
    if split:
        attn_p = sum(p for p, _b in attn_rows)
        attn_b = sum(b for _p, b in attn_rows)
        kv_p = sum(p for p, _b in kv_rows)
        kv_b = sum(b for _p, b in kv_rows)
    else:
        attn_p = sum(p for p, _b in attn_rows) + sum(p for p, _b in kv_rows)
        attn_b = sum(b for _p, b in attn_rows) + sum(b for _p, b in kv_rows)
        kv_p = kv_b = 0
    got_a = _role_cells(rel, "attn")
    if set(got_a) != {"fp8"} or _cell_pb(got_a, "fp8") != (attn_p + kv_p, attn_b + kv_b):
        raise ValueError("longcat ngram attention bytes do not match block fp8")
    fm = dict(spec.formats)
    fm["expert"] = RoleFormat("fp8", 8.0 * exp_b / exp_p)
    fm["mlp"] = RoleFormat("fp8", 8.0 * mlp_b / mlp_p)
    layers = spec.layers
    if split:
        fm["attn"] = RoleFormat("fp8", 8.0 * attn_b / attn_p)
        fm["kv_a"] = RoleFormat("fp8", 8.0 * kv_b / kv_p)
        layers = _retag_kv_a(spec.layers)
        kv_note = "kv_a 的 scale 按 ceil 单独记 kv_a，不并进 attn。"
    else:
        fm["attn"] = RoleFormat("fp8", 8.0 * attn_b / attn_p)
        kv_note = "kv_a 与其余注意力投影共用同一个 128 比例，留在 attn。"
    note = (
        "FP8（e4m3，动态激活，128×128 的 fp32 scale）。disable_quant_module 是空的，"
        "所以注意力 GEMM 也量化。专家和两套稠密 MLP 都不在 ignored_layers 里，整份按块 fp8 计。",
        "n-gram 表、投影、embedding、lm_head、norm 和 router 在忽略名单里。"
        "表和投影仍是 bf16。投影的激活是 bf16 嵌入行，不是全局 fp8。",
        "router 权重是 fp32。它的激活仍是全局 fp8 输入，不另记一种激活。",
        kv_note + " checkpoint 里的 MTP 仍不建层。",
    )
    return replace(spec, layers=layers, formats=tuple(sorted(fm.items())), notes=spec.notes + note)


def _apply_longcat_fp8(spec: "ModelSpec", rel: dict, c: dict) -> "ModelSpec":
    """0.399: bill LongCat-Flash FP8 from ignored_layers, not as one role-average width.

    Experts and the dense SwiGLU stay bf16 only on the blocks named in ignored_layers.
    Every other block is e4m3 with a fp32 scale per 128x128 tile. Attention is the
    self_attn module that quantization leaves in bf16. MTP eh_proj and the draft
    GEMMs are fp8; the private embedding and mtp.norm stay bf16. kv_a is its own
    role when its scale tiles are not the uniform 128 ratio. A LongCat-Flash config
    with no quantization_config is unchanged. A fp8 config whose headers do not
    match this partition raises instead of smearing the two buckets.
    """
    if _is_longcat_ngram(c):
        return spec
    archs = c.get("architectures") or []
    mt = c.get("model_type") or ""
    # 0.401: LongcatForCausalLM FP8 uses the same ignored_layers partition. Zero skipped
    # blocks (Thinking-2601-FP8) is the all-fp8 expert and dense case, not a new rule.
    if ("LongcatFlashForCausalLM" not in archs and mt != "longcat_flash"
            and not _is_longcat_causal(c) and not _is_longcat_omni(c)):
        return spec
    qc = c.get("quantization_config")
    if not isinstance(qc, dict):
        qc = rel.get("quantization_config")
    if not isinstance(qc, dict) or not qc:
        return spec
    if (qc.get("quant_method") != "fp8" or qc.get("fmt") != "e4m3"
            or qc.get("activation_scheme") != "dynamic"):
        raise ValueError("longcat_flash fp8 scheme is not e4m3 with dynamic activations")
    if list(qc.get("weight_block_size") or []) != [128, 128]:
        raise ValueError("longcat_flash fp8 weight_block_size is not [128, 128]")
    if c.get("disable_quant_module") != ["self_attn"]:
        raise ValueError("longcat_flash fp8 disable_quant_module is not ['self_attn']")
    ignored = qc.get("ignored_layers")
    if not isinstance(ignored, list) or not ignored:
        raise ValueError("longcat_flash fp8 ignored_layers is missing")
    omni = _is_longcat_omni(c)
    if not spec.layers or len(spec.layers) % 2:
        raise ValueError("longcat_flash fp8 builder did not emit paired blocks")
    if omni:
        if spec.mtp_layers or (rel.get("params_mtp") or 0) or _role_cells(rel, "mtp"):
            raise ValueError("longcat omni fp8 has no MTP module")
    elif len(spec.mtp_layers) != 1:
        raise ValueError("longcat_flash fp8 builder did not emit one MTP layer")
    n_blocks = len(spec.layers) // 2
    moe0 = spec.layers[0]
    dense1 = spec.layers[1]
    if moe0.ffn.kind != "moe" or not moe0.ffn.gated:
        raise ValueError("longcat_flash block 0 is not a SwiGLU MoE")
    if dense1.ffn.kind != "dense" or not dense1.ffn.gated or dense1.ffn.d_ff != moe0.ffn.d_ff:
        raise ValueError("longcat_flash sublayer 1 is not the dense SwiGLU")
    h = spec.hidden
    n_exp = moe0.ffn.n_experts
    d_exp = moe0.ffn.d_expert
    d_ff = moe0.ffn.d_ff
    zero = moe0.ffn.zero_experts
    exp_re = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(?:gate_proj|up_proj|down_proj)$")
    den_re = re.compile(r"^model\.layers\.(\d+)\.mlps\.[01]\.(?:gate_proj|up_proj|down_proj)$")
    exp_n = [0] * n_blocks
    den_n = [0] * n_blocks
    for name in ignored:
        if not isinstance(name, str):
            raise ValueError("longcat_flash ignored_layers entry is not a string")
        em = exp_re.match(name)
        if em:
            b, e = int(em.group(1)), int(em.group(2))
            if b >= n_blocks or e >= n_exp:
                raise ValueError(f"longcat_flash ignored expert {name} is outside the builder")
            exp_n[b] += 1
            continue
        dm = den_re.match(name)
        if dm:
            b = int(dm.group(1))
            if b >= n_blocks:
                raise ValueError(f"longcat_flash ignored dense MLP {name} is outside the builder")
            den_n[b] += 1
    full_e = n_exp * 3
    expert_skip: set[int] = set()
    dense_skip: set[int] = set()
    for b in range(n_blocks):
        if exp_n[b] == full_e:
            expert_skip.add(b)
        elif exp_n[b] != 0:
            raise ValueError(
                f"longcat_flash block {b} ignores {exp_n[b]} expert projections, not 0 or {full_e}")
        if den_n[b] == 6:
            dense_skip.add(b)
        elif den_n[b] != 0:
            raise ValueError(
                f"longcat_flash block {b} ignores {den_n[b]} dense projections, not 0 or 6")
    block = 128
    gate_p = h * d_exp
    down_p = d_exp * h
    gate_b = _fp8_block_bytes(h, d_exp, block)
    down_b = _fp8_block_bytes(d_exp, h, block)
    if gate_b * down_p != down_b * gate_p:
        raise ValueError("longcat_flash expert gate and down fp8 scales do not share one ratio")
    one_exp_p = 2 * gate_p + down_p
    one_exp_b = 2 * gate_b + down_b
    n_bf = len(expert_skip)
    n_fp = n_blocks - n_bf
    exp_bf_p, exp_bf_b = n_bf * n_exp * one_exp_p, 2 * n_bf * n_exp * one_exp_p
    exp_fp_p, exp_fp_b = n_fp * n_exp * one_exp_p, n_fp * n_exp * one_exp_b
    got_e = _role_cells(rel, "expert")
    if set(got_e) - {"bf16", "fp8"}:
        raise ValueError("longcat_flash expert role has an unexpected format")
    if (_cell_pb(got_e, "bf16"), _cell_pb(got_e, "fp8")) != ((exp_bf_p, exp_bf_b), (exp_fp_p, exp_fp_b)):
        raise ValueError("longcat_flash expert bytes do not match the ignored blocks")
    up_b = _fp8_block_bytes(h, 2 * d_ff, block)
    if up_b != 2 * _fp8_block_bytes(h, d_ff, block):
        raise ValueError("longcat_flash fused gate_up scale bytes do not equal separate gate and up")
    dn_b = _fp8_block_bytes(d_ff, h, block)
    up_p = h * 2 * d_ff
    dn_p = d_ff * h
    if up_b * dn_p != dn_b * up_p:
        raise ValueError("longcat_flash dense gate_up and down fp8 scales do not share one ratio")
    one_m_p, one_m_b = up_p + dn_p, up_b + dn_b
    n_m_bf, n_m_fp = len(dense_skip), n_blocks - len(dense_skip)
    mlp_bf_p, mlp_bf_b = n_m_bf * 2 * one_m_p, 2 * n_m_bf * 2 * one_m_p
    mlp_fp_p, mlp_fp_b = n_m_fp * 2 * one_m_p, n_m_fp * 2 * one_m_b
    got_m = _role_cells(rel, "mlp")
    if set(got_m) - {"bf16", "fp8"}:
        raise ValueError("longcat_flash mlp role has an unexpected format")
    if (_cell_pb(got_m, "bf16"), _cell_pb(got_m, "fp8")) != ((mlp_bf_p, mlp_bf_b), (mlp_fp_p, mlp_fp_b)):
        raise ValueError("longcat_flash dense MLP bytes do not match the ignored blocks")
    names = tuple(l.name for l in spec.layers[0].attn_linears)
    if names != ("q_a", "q", "kv_a", "kv_up", "o"):
        raise ValueError("longcat_flash attention projections are not the released MLA set")
    attn_one = sum(l.params for l in spec.layers[0].attn_linears)
    attn_p = 2 * n_blocks * attn_one
    got_a = _role_cells(rel, "attn")
    if set(got_a) != {"bf16"}:
        raise ValueError("longcat_flash attention is not pure bf16")
    ap, ab = _cell_pb(got_a, "bf16")
    if ap != attn_p or ab != 2 * ap:
        raise ValueError("longcat_flash attention bytes are not 2 per parameter")
    pool = n_exp + zero
    rt_p = n_blocks * (h * pool + pool)
    got_r = _role_cells(rel, "router")
    if set(got_r) != {"fp32"}:
        raise ValueError("longcat_flash router is not pure fp32")
    rp, rb = _cell_pb(got_r, "fp32")
    if rp != rt_p or rb != 4 * rp:
        raise ValueError("longcat_flash router bytes are not fp32")
    for role in ("embed", "lm_head"):
        got = _role_cells(rel, role)
        if set(got) != {"bf16"}:
            raise ValueError(f"longcat_flash {role} is not pure bf16")
        p, b = _cell_pb(got, "bf16")
        if p != spec.vocab * h or b != 2 * p:
            raise ValueError(f"longcat_flash {role} bytes are not 2 per parameter")
    q_lora = next(l.n for l in spec.layers[0].attn_linears if l.name == "q_a")
    kv_n = next(l.n for l in spec.layers[0].attn_linears if l.name == "kv_a")
    rope = spec.layers[0].core.rope_dim
    attn_misc = 2 * h + q_lora + (kv_n - rope)
    norm_p = 2 * n_blocks * attn_misc + h
    got_n = _role_cells(rel, "norm")
    if set(got_n) != {"bf16"}:
        raise ValueError("longcat_flash norm is not pure bf16")
    np_, nb = _cell_pb(got_n, "bf16")
    if np_ != norm_p or nb != 2 * np_:
        raise ValueError("longcat_flash norm bytes are not the main-stack norms")
    extra_fm: dict[str, RoleFormat] = {}
    role_of: dict[str, str] = {}
    tagged_mtp: tuple = ()
    if not omni:
        mt_layer = spec.mtp_layers[0]
        if mt_layer.ffn.kind != "dense" or mt_layer.ffn.n_experts:
            raise ValueError("longcat_flash MTP MLP is not the dense SwiGLU")
        mtp_bf_p = spec.vocab * h + h + mt_layer.misc_params
        got_mtp = _role_cells(rel, "mtp")
        if set(got_mtp) - {"bf16", "fp8"}:
            raise ValueError("longcat_flash MTP role has an unexpected format")
        if _cell_pb(got_mtp, "bf16") != (mtp_bf_p, 2 * mtp_bf_p):
            raise ValueError("longcat_flash MTP bf16 bucket is not the embedding, mtp.norm, and layer norms")
        named = {l.name: l for l in mt_layer.attn_linears}
        ff = {l.name: l for l in mt_layer.ffn_linears}
        if set(named) != {"q_a", "q", "kv_a", "kv_up", "o"} or set(ff) != {"gate_up", "down"}:
            raise ValueError("longcat_flash MTP GEMMs are not the dense draft block")
        if (ff["gate_up"].k, ff["gate_up"].n) != (h, 2 * d_ff):
            raise ValueError("longcat_flash MTP gate_up is not the fused dense SwiGLU")
        gemms = (
            ("eh", 2 * h, h),
            ("q_a", named["q_a"].k, named["q_a"].n),
            ("q", named["q"].k, named["q"].n),
            ("kv_a", named["kv_a"].k, named["kv_a"].n),
            ("kv_up", named["kv_up"].k, named["kv_up"].n),
            ("o", named["o"].k, named["o"].n),
            ("gate_up", ff["gate_up"].k, ff["gate_up"].n),
            ("down", ff["down"].k, ff["down"].n),
        )
        rows = tuple((name, k * n, _fp8_block_bytes(k, n, block)) for name, k, n in gemms)
        fp_p = sum(p for _, p, _b in rows)
        fp_b = sum(b for _, _p, b in rows)
        if _cell_pb(got_mtp, "fp8") != (fp_p, fp_b):
            raise ValueError("longcat_flash MTP fp8 bytes do not match per-GEMM 128x128 scales")
        if spec.mtp_extra_params < 2 * h * h:
            raise ValueError("longcat_flash mtp remainder is smaller than eh_proj")
        groups: dict[tuple[int, int], list[tuple[str, int, int]]] = {}
        for name, p, b in rows:
            g = math.gcd(b, p)
            groups.setdefault((b // g, p // g), []).append((name, p, b))
        ranked = sorted(groups.items(), key=lambda kv: (-sum(p for _, p, _b in kv[1]), kv[0]))
        top = sum(p for _, p, _b in ranked[0][1])
        if sum(1 for _k, items in ranked if sum(p for _, p, _b in items) == top) != 1:
            raise ValueError("longcat_flash MTP fp8 ratio clusters tie for most parameters")
        for i, (_key, items) in enumerate(ranked):
            gnames = [nm for nm, _p, _b in items]
            if i == 0:
                role = "mtp_fp8"
            elif len(gnames) == 1:
                role = "mtp_" + gnames[0]
            else:
                role = "mtp_" + "_".join(sorted(gnames))
            pb = sum(p for _, p, _b in items)
            bb = sum(b for _, _p, b in items)
            extra_fm[role] = RoleFormat("fp8", 8.0 * bb / pb)
            for nm in gnames:
                role_of[nm] = role

        def _tag(lins: tuple) -> tuple:
            out = []
            for lin in lins:
                if lin.name not in role_of:
                    raise ValueError(f"longcat_flash MTP linear {lin.name} has no fp8 role")
                out.append(replace(lin, role=role_of[lin.name]))
            return tuple(out)

        tagged_mtp = (_tag_layer(mt_layer, _tag),)

    new_layers = []
    for i, L in enumerate(spec.layers):
        blk = i // 2
        ffn = L.ffn
        fl = L.ffn_linears
        if i % 2 == 0 and blk in expert_skip:
            ffn = replace(ffn, expert_role="expert_bf16")
        if blk in dense_skip:
            fl = tuple(replace(lin, role="mlp_bf16") if lin.name in ("gate_up", "down", "up") else lin
                       for lin in fl)
        new_layers.append(replace(L, ffn=ffn, ffn_linears=fl))
    fm = dict(spec.formats)
    if exp_fp_p:
        fm["expert"] = RoleFormat("fp8", 8.0 * exp_fp_b / exp_fp_p)
    elif exp_bf_p:
        fm["expert"] = RoleFormat("bf16", 16.0)
    if exp_bf_p:
        fm["expert_bf16"] = RoleFormat("bf16", 16.0)
        fm["act_expert_bf16"] = RoleFormat("bf16", 16.0)
    if mlp_fp_p:
        fm["mlp"] = RoleFormat("fp8", 8.0 * mlp_fp_b / mlp_fp_p)
    elif mlp_bf_p:
        fm["mlp"] = RoleFormat("bf16", 16.0)
    if mlp_bf_p and mlp_fp_p:
        fm["mlp_bf16"] = RoleFormat("bf16", 16.0)
        fm["act_mlp_bf16"] = RoleFormat("bf16", 16.0)
    elif mlp_bf_p:
        fm["mlp_bf16"] = RoleFormat("bf16", 16.0)
        fm["act_mlp_bf16"] = RoleFormat("bf16", 16.0)
    fm["attn"] = RoleFormat("bf16", 16.0)
    fm["act_attn"] = RoleFormat("bf16", 16.0)
    if not omni:
        fm["mtp"] = RoleFormat("bf16", 16.0)
    fm.update(extra_fm)

    def _lst(xs: set[int]) -> str:
        return "、".join(str(i) for i in sorted(xs)) or "（无）"

    if omni:
        note = (
            f"FP8（e4m3，动态激活，128×128 的 fp32 scale）：专家只在 block {_lst(expert_skip)} 保持 bf16，"
            f"稠密 MLP 只在 block {_lst(dense_skip)} 保持 bf16。其余专家和稠密 MLP 按各自的 fp8 桶计，"
            "不把 bf16 和 fp8 平均成一个角色。",
            "注意力在 disable_quant_module 的 self_attn 里，权重和激活都是 bf16。",
            "量化过的专家和稠密 MLP 的激活仍是动态 fp8。未量化的专家和稠密 MLP 激活是 bf16。",
            "router 权重是 fp32。它的激活仍是全局 fp8 输入；forward 里把两边 cast 到 fp32 不另记一种激活。",
            "embedding、lm_head 和 norm 是 bf16。没有 MTP 模块，也不把 V 补到 qk_head_dim。",
            "视觉塔和音频塔的 .pt 与 bf16 仓库相同，不在这份 fp8 名单里，权重和激活仍是 bf16。",
        )
        return replace(spec, layers=tuple(new_layers), formats=tuple(sorted(fm.items())),
                       notes=spec.notes + note)
    eh_role = role_of["eh"]
    kv_role = role_of["kv_a"]
    note = (
        f"FP8（e4m3，动态激活，128×128 的 fp32 scale）：专家只在 block {_lst(expert_skip)} 保持 bf16，"
        f"稠密 MLP 只在 block {_lst(dense_skip)} 保持 bf16。其余专家和稠密 MLP 按各自的 fp8 桶计，"
        "不把 bf16 和 fp8 平均成一个角色。",
        "注意力在 disable_quant_module 的 self_attn 里，权重和激活都是 bf16。",
        "量化过的专家、稠密 MLP 和 MTP GEMM 的激活仍是动态 fp8。未量化的专家和稠密 MLP 激活是 bf16。",
        "router 权重是 fp32。它的激活仍是全局 fp8 输入；forward 里把两边 cast 到 fp32 不另记一种激活。",
        "embedding、lm_head、norm，以及 MTP 的私有 embedding 和 mtp.norm，是 bf16。",
        f"MTP 的 eh_proj 记 {eh_role}。kv_a 的 scale 按 ceil 单独记 {kv_role}。"
        "主栈 V 仍不填充。draft 的 V 仍补到 qk_head_dim。",
    )
    return replace(
        spec, layers=tuple(new_layers), mtp_layers=tagged_mtp,
        mtp_eh_params=2 * h * h, mtp_eh_role=eh_role,
        formats=tuple(sorted(fm.items())), notes=spec.notes + note)


def _attach_longcat_omni(spec: "ModelSpec", rel: dict, c: dict) -> "ModelSpec":
    """Bill Omni audio codebook tables and output heads. Towers stay on the vision spec.

    Input embeddings are summed into the next decode token (``AudioEmbedding.fused_lookup``).
    Residual embeddings are read between audio heads inside ``sample``. The four output
    linears have no bias and no extra projection (``has_proj`` is false). Prefill does not
    read these tables. ``language_model_embedding.pt`` duplicates ``embed_tokens`` and is
    not added again.
    """
    if not _is_longcat_omni(c):
        return spec
    meta = rel.get("longcat_omni")
    if not isinstance(meta, dict):
        raise ValueError("longcat omni tower header is missing")
    from .vision import longcat_omni_counts
    counts = longcat_omni_counts(meta)
    if counts["text_hidden"] != spec.hidden:
        raise ValueError("longcat omni text hidden does not match the language stack")
    if spec.mtp_layers or (rel.get("params_mtp") or 0):
        raise ValueError("longcat omni language stack has no MTP")
    n_in = counts["codebooks"]
    n_res = counts["residual"]
    vocab = counts["code_vocab"]
    h = spec.hidden
    heads = tuple(
        Linear(f"audio_head.{i}", h, vocab, "col", rows="seq", role="audio_head")
        for i in range(n_in))
    lookup = (n_in + n_res) * vocab * h
    linear = sum(lin.params for lin in heads)
    if lookup != counts["audio_emb"] + counts["audio_residual"]:
        raise ValueError("longcat omni codebook lookup params do not match the header")
    if linear != counts["audio_heads"]:
        raise ValueError("longcat omni audio heads do not match the header")
    io = int(rel.get("params_audio_io") or 0)
    if io != lookup + linear:
        raise ValueError("longcat omni params_audio_io does not match the codebook files")
    from .longcat_codec import codec_spec
    codec = codec_spec(rel.get("longcat_codec"))
    note = (
        f"音频输入码本 {n_in} 张 Embedding({vocab}, {h}) 在 decode 的下一拍查表并加到词嵌入。"
        "prefill 不读这几张表。",
        f"音频输出是 {n_in} 个无 bias 的 Linear({h}, {vocab})，在 sample 的行上各跑一次。"
        f"head_idx > 0 时先查 {n_res} 张残差 Embedding({vocab}, {h}) 再加到隐状态。"
        "has_proj 为 false，没有额外投影。重复惩罚不改变 GEMM 形状，不计额外周期。",
        "language_model_embedding.pt 的 word_embeddings 与 safetensors 的 embed_tokens 同形，不重复计入。",
        "波形解码器是 LongCat-Audio-Codec 的 LongCatAudioCodecDecoder（24k，causal）。"
        f"audio_codec/*.pt 有 {codec.n_tensors} 个 fp32 张量，decode 权重 {codec.params_decode}，"
        f"in_proj {codec.params_in_proj} 在文件里但 from_codes 不读。"
        "这部分不进 params()。编码器 .pt 不在这份 HF 根目录，不计入。"
        "shift_codec_back 只改下标，不增加 GEMM。"
        "24k MemDecoder 的 Sequential 以 nn.Tanh 结束，每个输出采样一次，没有参数。",
        "视觉是 Conv3d（时间 2、空间 14）加 32 层双向注意力。SwiGLU 内宽是 int(5184*2/3)。"
        "每一层在 gate 乘积和 fc2 之间做一次内层 RMSNorm：forward 升到 fp32，"
        "按 patch token（不是投影后的宽度对）计，vec 是 4 乘内宽，每行一次 rsqrt。"
        "同一条 RMSNorm.forward 还跑在 norm1、norm2，以及把 head 展平之后的 q_norm / k_norm 上。"
        "q / k 的宽度是 embed_dim，不是 head_dim，每个 token 一次 rsqrt，不是每个头一次。"
        "权重已在塔参数里，不再记 w_params。"
        "LayerScale 是分支乘 ls1 / ls2 再加回残差，宽度是 embed_dim，不升 fp32，没有 rsqrt。"
        "投影器先把宽度相邻两列拼成 2h，再做 LayerNorm(2h)，行数是 ph*pw/2，然后三层 Linear。"
        "SwiGLU 的 SiLU 只作用在 fc1 上，宽度是内宽，不是拼起来的 2 倍内宽。"
        "类没有升精度：vec 是 4 乘内宽，sfu 是 2 乘内宽（exp 与 reciprocal），a_fmt 是 bf16。"
        "SiLU 与 fc3 的乘是一次逐元素乘，宽度仍是内宽，sfu 是 0。"
        "p0 在前两层 Linear 之后各有一次 nn.GELU()，默认是 erf，不是 tanh 近似。"
        "行数是拼宽后的 ph*pw/2，宽度是 adaptor。vec 是 5 乘元素，sfu 是元素数（一次 erf），不升 fp32。"
        "最后一层 proj 没有 GELU。"
        "apply_rotary_pos_emb_vision 把 q 和 k 升到 fp32，对 freqs 做 cos 和 sin。"
        "freqs 的最后一维是 head_dim/2，不是 head 数，也不是 embed_dim。q 和 k 各算一次，不复用表。"
        "旋转乘是每个元素 3 次，作用在完整 head 上。"
        "cos 和 sin 再 repeat(1, 1, 2)，把最后一维从 head_dim/2 复制到 head_dim。"
        "q 和 k 各复制一次，不共用。vec 是 4 乘 token 乘 head_dim，a_fmt 是 fp32，sfu 是 0。"
        "softmax 仍是双向分数，vec 是 heads 乘 tokens 的平方乘 5，sfu 仍是 0。"
        "inv_freq 不是 persistent。VisionRotaryEmbedding2D 的 dim 是 head_dim/2，"
        "inv_freq 长度是 arange(0, dim, 2)。"
        "forward_ 的 torch.outer 每个编码器前向一次，seqlen 是 max(t, h, w)。"
        "静图 t 是 1，边长是 max(ph, pw)。不乘 token 数，不乘层数，也不乘图像张数。"
        "vec 是边长乘 inv_freq 长度，a_fmt 是 fp32，sfu 是 0。不记成 GEMM。"
        "pos_ids 的 gather 不单列。"
        "单张图的 post-squeeze 在 token 数不超过 max_tokens_video 时是恒等。视频多帧压缩不进默认图像网格。"
        "音频 FSMN 每一层有两次 LayerNorm(proj)：memory_ln 在深度卷积前，ffn 的 LayerNorm 在升维前。"
        "按帧计，fp32，每帧一次 rsqrt。宽度是 proj，不是 FFN 的 hidden。"
        "发布的 activation 是 relu6：min(max(x, 0), 6)，没有超越函数，不升精度。"
        "pre、每一层 FFN、以及每个 DNN Linear 之后各一次，宽度是 hidden，不是 proj。"
        "mask 乘在每一层之前和 DNN 之前各一次，宽度是 proj。bf16 乘 int32，结果是 fp32，sfu 是 0。"
        "音频投影器是 Linear、GELU、Linear。GELU 用同一套 erf 计数，宽度是文本 hidden。"
        "发布的 dropout 是 0。emb_mult 固定是 1，不单列。",
    )
    fm = dict(spec.formats)
    fm["audio_head"] = RoleFormat("bf16", 16.0)
    fm["act_audio_head"] = RoleFormat("bf16", 16.0)
    return replace(
        spec, lookup_params=lookup, lookup_bits=16.0, audio_codebooks=(n_in, n_res, vocab),
        io_post=spec.io_post + heads, formats=tuple(sorted(fm.items())),
        release_params=(spec.release_params or 0) + lookup + linear,
        longcat_codec=codec, notes=spec.notes + note)


def _tag_layer(layer: Layer, tag) -> Layer:
    return replace(layer, attn_linears=tag(layer.attn_linears), ffn_linears=tag(layer.ffn_linears))


def _moe_groups(c: dict) -> tuple[int, ...]:
    """0.255: (n_group, topk_group) when the router is group-limited — transformers DeepseekV3TopkRouter /
    BeamTopKRouter, vLLM grouped_topk (bailing_moe use_grouped_topk = both set) 「来源」: group scores = sum of the top-2
    (bias-corrected) expert scores per group, keep topk_group groups, then top-k over their experts.  n_group ≤ 1 or
    topk_group ≥ n_group is unrestricted → ()."""
    G, g = c.get("n_group"), c.get("topk_group")
    E = _g(c, "n_routed_experts", "num_experts", "num_local_experts", default=0) or 0
    if not G or not g or G <= 1 or g >= G or not E or E % G:
        return ()
    return (int(G), int(g))


def _from_omniasr_ctc(model_id: str, hf_id: str, cfg: dict, rel: dict) -> ModelSpec:
    """Encoder-only OmniASRForCTC. The head is a biased linear, not a Shaw conformer."""
    from .vision import audio_spec, vision_spec
    au = audio_spec(cfg)
    vs = vision_spec(hf_id, cfg)
    if au is None or vs is None or au.kind != "w2v" or not au.ctc:
        raise ValueError(f"{hf_id}: omniasr_ctc does not match OmniASRForCTC")
    vr = (rel.get("roles") or {}).get("vision") or {}
    p = sum(int(x["params"]) for x in vr.values())
    b = sum(int(x["bytes"]) for x in vr.values())
    if p != au.params or (rel.get("params_llm") or 0) or (rel.get("params_vision") or 0) != p:
        raise ValueError(f"{hf_id}: CTC params {au.params} != release vision {p}")
    if rel.get("params_mtp") or set(vr) != {"fp32"}:
        raise ValueError(f"{hf_id}: omniasr_ctc weights are fp32 and have no MTP")
    fp = vr["fp32"]
    if int(fp["bytes"]) != 4 * int(fp["params"]) or b != 4 * p:
        raise ValueError(f"{hf_id}: omniasr_ctc byte count is not fp32")
    vs = replace(vs, w_Bpp=4.0)
    note = ("OmniASR CTC：只有 Wav2Vec2 编码器，没有 Llama 解码器，也没有 Shaw。"
            "ctc_head 是 Linear(hidden, vocab)，带 bias，每个编码器帧读一次。"
            "注意力是双向的整段方阵，不写 KV。权重和激活都是 fp32。"
            "ACT2FN['gelu'] 是 erf 形式。特征卷积每一层、位置卷积、以及每一层 FFN 的 intermediate "
            "各计一次向量运算，不是 GEMM。类不升精度。"
            "eval 的 Dropout 和 LayerDrop 是恒等，不单列。最终 argmax 不单列。"
            "特征卷积每一层在 GELU 之前做 LayerNorm，宽度是 conv 通道，不是 hidden。"
            "投影 Linear 之前再做一次 LayerNorm，宽度仍是最后一层通道。"
            "编码器每一层在注意力前和 FFN 前各做一次 LayerNorm(hidden)，块之后再做一次。"
            "LayerNorm 在 fp32 归约，按行计一次 rsqrt，不是 GEMM。位置卷积没有 LayerNorm。"
            "不是 0.5 的残差加，以及 padding 的 masked_fill，不单列。"
            "投影器和 CTC 头没有 GELU。")
    return ModelSpec(
        id=model_id, hf_id=hf_id, arch="OmniASRForCTC", hidden=au.dims[0], vocab=au.out,
        layers=(), tie_embeddings=True, coverage="full", notes=(note,),
        formats=(("attn", RoleFormat("fp32", 32.0)),), act_fmt="fp32", kv_fmt="fp32",
        release_params=p, release_bytes=b or None, vision=vs, vision_params=au.params,
        domain="asr", kv_cache=False, ctc_only=True, final_norm_params=0)


def _from_granite_speech5_ctc(model_id: str, hf_id: str, cfg: dict, rel: dict) -> ModelSpec:
    """Encoder-only GraniteSpeech5ForCTC. No granite stack is invented."""
    from .vision import audio_spec, vision_spec
    au = audio_spec(cfg)
    vs = vision_spec(hf_id, cfg)
    if au is None or vs is None or au.kind != "gsctc":
        raise ValueError(f"{hf_id}: granite_speech5_ctc encoder does not match the released class")
    vr = (rel.get("roles") or {}).get("vision") or {}
    p = sum(int(x["params"]) for x in vr.values())
    b = sum(int(x["bytes"]) for x in vr.values())
    if p != au.params or (rel.get("params_llm") or 0) or (rel.get("params_vision") or 0) != p:
        raise ValueError(f"{hf_id}: CTC encoder params {au.params} != release vision {p}")
    if b and p:
        vs = replace(vs, w_Bpp=b / p)
    fp32 = (vr.get("fp32") or {}).get("params") or 0
    note = (f"Granite Speech 5 CTC：只有编码器，没有 granite 语言模型，也没有 Q-Former。"
            f"Shaw 分块相对位置按 context {au.chunk} 补齐。subsample_layers {list(au.gs_sub)} "
            f"每层把保留的时间长度减半。中间 CTC 的 out 与 out_mid 各跑一次，最终 ctc_head "
            f"与 out 绑在同一份权重上，再读一次。BatchNorm 仿射是 fp32（{fp32} 参数），"
            "running stats 是 buffer，rsqrt 每个通道一次，不随 clip 数变。"
            "SiLU、GLU、0.5 残差、LayerNorm 和时间池化按发布的前向计成向量运算，不是 GEMM。"
            "卷积上的 SiLU 读 fp32 的 BatchNorm 输出，不另做 cast。FFN 的 SiLU 和 GLU 仍是 bf16。"
            "activation_dropout 与 attention_dropout 都是 0，不单列。")
    return ModelSpec(
        id=model_id, hf_id=hf_id, arch="GraniteSpeech5ForCTC", hidden=au.dims[0], vocab=au.gs_mid,
        layers=(), tie_embeddings=True, coverage="full", notes=(note,),
        formats=(("attn", RoleFormat("bf16", 16.0)),), act_fmt="bf16", kv_fmt="bf16",
        release_params=p, release_bytes=b or None, vision=vs, vision_params=au.params,
        domain="asr", kv_cache=False, ctc_only=True, final_norm_params=0)


def _from_d1_omni(model_id: str, hf_id: str, cfg: dict, rel: dict) -> ModelSpec:
    """Bidirectional LFM2.5 encoder plus a decision head. Not an lfm2 decode stack."""
    from .vision import vision_spec
    vs = vision_spec(hf_id, cfg)
    if vs is None or vs.family != "d1omni" or vs.audio is None:
        raise ValueError(f"{hf_id}: d1_omni does not match the released class")
    pr = vs.rz
    trunk = int(pr["trunk_params"]) + int(pr["head_params"])
    llm = int(rel.get("params_llm") or 0)
    vis = int(rel.get("params_vision") or 0)
    if trunk != llm or vs.params != vis:
        raise ValueError(f"{hf_id}: d1 trunk {trunk} / vision {vs.params} != release {llm} / {vis}")
    buf = int(pr["buf"])
    got = int(((rel.get("roles") or {}).get("buffer") or {}).get("int64", {}).get("bytes") or 0)
    if got != buf:
        raise ValueError(f"{hf_id}: d1 buffer bytes {buf} != release {got}")
    if rel.get("vision_fmt"):
        raise ValueError(f"{hf_id}: d1 vision weights are fp32; vision_fmt must stay unset")
    note = ("d1-omni：双向 LFM2.5 编码器（居中 3-tap，稠密加性 mask，没有因果 KV，没有 lm_head）。"
            "DecisionHead 只在文本位置上跑，打分只看标记位。FastConformer 的相对位置是 2T-1，"
            "深度卷积是向量运算。SigLIP2 把每个 crop 补到 1024，投影只看 pixel-unshuffle 之后的有效行。"
            "权重和激活都是 fp32。BatchNorm running stats 是 buffer。"
            "ConformerLayer 每一层是 LayerNorm、Linear、F.silu、Linear，然后 x + ff * 0.5；"
            "卷积是 LayerNorm、pointwise 到 2*d、F.glu（切分是 view，保留 d）、深度卷积、"
            "BatchNorm1d、F.silu、pointwise。"
            "SiLU、GLU、0.5 残差、LayerNorm 和 BatchNorm 按发布的前向计成向量运算，不是 GEMM。"
            "GLU 和卷积 SiLU 的宽度是 d，不是 2*d。0.5 残差的宽度是 d。"
            "BatchNorm 的 rsqrt 每个通道一次，不随 clip 数变。类不升精度，这些向量运算的 a_fmt 是 fp32。"
            "adapter 是 LayerNorm、Linear、F.gelu、Linear。"
            "residual 是 LayerNorm、Linear、F.gelu、Linear，再加回输入。"
            "F.gelu 默认是 erf 形式，不是 gelu_pytorch_tanh。通用 GEMM 尾项不减。"
            "subsampling 的三次 nn.ReLU 是 max(x, 0)，一次截断，不是 relu6 的两次，按元素计成向量运算，不是 GEMM。"
            "时间 mask 是激活乘同 dtype 的 0/1，一次乘，比较不另计。"
            "深度卷积前的 masked_fill 是 where，不是乘法，也不走 ReLU，不单列。"
            "pad、整份残差加和位置正弦表不单列。"
            "视觉塔的 hidden_act 是 gelu_pytorch_tanh，MLP 不单列。"
            "Siglip2EncoderLayer 在注意力前和 MLP 前各做一次 LayerNorm，post_layernorm 在补齐后的序列上。"
            "vision_use_head 是 false，pooling head 不跑。"
            "Projector.forward 在 2×2 unshuffle 之后是 linear_2(F.gelu(linear_1(x)))。"
            "F.gelu 默认是 erf。行数是 unshuffle 之后的有效位置，宽度是 projector_hidden_size，不是塔的 hidden。"
            "类不升精度，这次 GELU 的 a_fmt 是 fp32。"
            "trunk 的 operator_norm、ffn_norm、embedding_norm 以及 q/k 的 RMSNorm 都升到 fp32："
            "pow(2).mean，一次 rsqrt，再乘 weight。"
            "q/k 的归约宽度是 head_dim，不是把多头展成 hidden。"
            "MLP 是 w2(F.silu(w1(x)) * w3(x))。SiLU 和乘积的宽度是 FFN，不是 hidden，也不是 2 倍 FFN。"
            "F.silu 不升精度。"
            "DecisionHead.scorer 是 LayerNorm、Linear、nn.GELU()、Linear，只看标记位。"
            "nn.GELU() 是 erf。"
            "前面的 TransformerEncoderLayer 没有传入 activation，PyTorch 2.6 的默认是 F.relu。"
            "norm_first 为真：注意力前和 FFN 前各一次 LayerNorm，宽度是 hidden，行数是文本位置。"
            "F.relu 是 max(x, 0)，一次截断，不是 relu6。它在 linear1 之后，宽度是 4 * hidden，不是 hidden。"
            "dropout 是 0.0，是恒等。整份残差加不单列。padding mask 留在原来的 softmax 里。"
            "TransformerEncoder 的 norm 是 None，没有第三份 LayerNorm。"
            "RoPE 的 cos/sin、整份残差加和位置插值不单列。通用 GEMM 尾项不减。")
    return ModelSpec(
        id=model_id, hf_id=hf_id, arch="D1OmniModel", hidden=int(pr["trunk"]), vocab=int(pr["vocab"]),
        layers=(), tie_embeddings=True, coverage="full", notes=(note,),
        formats=(("attn", RoleFormat("fp32", 32.0)),), act_fmt="fp32", kv_fmt="fp32",
        release_params=llm, release_bytes=int(rel.get("bytes_llm") or 0) or None,
        vision=vs, vision_params=vis, domain="vlm", kv_cache=False, final_norm_params=0,
        d1=True, trunk_params=trunk, d1_buf=buf)


def _whisper_vision_fmt(rel: dict, label: str = "whisper") -> str:
    """Activation dtype of a tower, from the one float vision role.

    ``config.torch_dtype`` is not the source. Distil-Whisper large-v2 says float32 and
    stores fp16. GEMM ``w_fmt`` stays the historical bf16 billing; this value is only
    the vector ``a_fmt`` for GELU, SiLU and ReLU. LayerNorm stays fp32. AuT uses the
    same rule.     Pixtral uses it for SiLU, the SwiGLU product, and the projector
    erf-GELU. Pixtral RMSNorm stays fp32. Qwen3-VL merger ``nn.GELU()`` and
    LFM2.5-VL projector ``ACT2FN["gelu"]`` use it the same way. GLM-5.3-Flash
    uses it for gate SiLU, the SwiGLU product, and merger ``nn.GELU()``.
    MiMo-V2.6 uses it for gate SiLU, the SwiGLU product, and merger ``nn.GELU()``.
    MiMo RMSNorm and LayerNorm stay fp32.
    DeepSeek-V4.1 vision ``F.silu``, the product, and ``F.gelu`` use it.
    DeepSeek RMSNorm stays fp32.
    Kimi projector ``nn.GELU()`` uses it. Kimi LayerNorm and RMSNorm stay fp32.
    Muse Spark NaViT ``ACT2FN["gelu"]`` uses it the same way. Those LayerNorms
    stay fp32. Muse Spark audio ``nn.GELU()`` uses the same float. Those
    RMSNorms stay fp32.
    Qwen3.5 / Qwen3.5-MoE / Qwen4-Exp merger ``nn.GELU()`` uses it the same way.
    Those LayerNorms stay fp32. A vision role with more than one dtype is not
    given this value.
    Ling-3.0-flash-VL ``linear_proj`` ``nn.GELU()`` uses it the same way.
    Those LayerNorms stay fp32.
    MiniCPM-V-4.7 merger ``nn.GELU()`` uses it the same way. Those LayerNorms
    stay fp32. The tower MLP and the vit_merger stay ``gelu_pytorch_tanh``
    and are not this dtype.
    A Qwen3-VL
    tower stored entirely as fp32 keeps the model activation format instead
    (WeVisOne). LayerNorm and GLM RMSNorm stay fp32.
    """
    vr = (rel.get("roles") or {}).get("vision") or {}
    keys = [k for k, v in vr.items() if isinstance(v, dict) and v.get("params")]
    if len(keys) != 1:
        raise ValueError(f"{label} tower vision role is not one dtype")
    key = keys[0]
    block = vr[key]
    bits = 8.0 * block["bytes"] / block["params"]
    expect = {"fp16": 16.0, "bf16": 16.0, "fp32": 32.0}
    if key not in expect or abs(bits - expect[key]) > 1e-3:
        raise ValueError(f"{label} tower vision role {key} is {bits} bits")
    return key


def _radio_vision_fmt(rel: dict) -> str:
    """Activation dtype of a C-RADIO tower.

    The linears are one bf16 vision role. The fp32 bucket is the six
    input-conditioner buffers (``norm_mean`` / ``norm_std``), not a second
    weight dtype. GELU uses the bf16 role. LayerNorm and RMSNorm stay fp32.
    """
    vr = (rel.get("roles") or {}).get("vision") or {}
    keys = [k for k, v in vr.items() if isinstance(v, dict) and v.get("params")]
    if keys != ["bf16", "fp32"] and keys != ["fp32", "bf16"]:
        raise ValueError("radio tower vision role is not bf16 plus the conditioner")
    fp = vr["fp32"]
    if fp.get("params") != 6 or fp.get("bytes") != 24:
        raise ValueError("radio conditioner is not 6 fp32 values")
    block = vr["bf16"]
    bits = 8.0 * block["bytes"] / block["params"]
    if abs(bits - 16.0) > 1e-3:
        raise ValueError(f"radio tower vision role bf16 is {bits} bits")
    return "bf16"


def _ernie_vision_fmt(rel: dict, cfg: dict, vision_params: int) -> str:
    """Activation dtype of the ERNIE-4.5-VL resampler ``nn.GELU()``.

    The usual vision role is two buckets. fp16 is the DFNRope tower. bf16 is
    ``VariableResolutionResamplerModel``. Thinking stores the whole tower as
    one bf16 bucket. Both layouts are 16 bits, so GEMM ``w_fmt`` stays bf16.
    ``nn.GELU()`` follows that bf16 storage. LayerNorm and RMSNorm stay fp32.
    The tower MLP's ``quick_gelu`` is not this dtype.
    """
    from .vision import ernie_resampler_params
    vr = (rel.get("roles") or {}).get("vision") or {}
    keys = [k for k, v in vr.items() if isinstance(v, dict) and v.get("params")]
    if set(keys) == {"bf16"}:
        block = vr["bf16"]
        bits = 8.0 * block["bytes"] / block["params"]
        if abs(bits - 16.0) > 1e-3:
            raise ValueError(f"ernie vision role bf16 is {bits} bits")
        if block["params"] != vision_params:
            raise ValueError("ernie bf16 vision role is not the tower")
        return "bf16"
    if set(keys) != {"fp16", "bf16"}:
        raise ValueError("ernie vision role is not fp16 tower plus bf16 resampler")
    for key in ("fp16", "bf16"):
        block = vr[key]
        bits = 8.0 * block["bytes"] / block["params"]
        if abs(bits - 16.0) > 1e-3:
            raise ValueError(f"ernie vision role {key} is {bits} bits")
    vc = cfg["vision_config"]
    res = ernie_resampler_params(
        int(vc["embed_dim"]), int(cfg["spatial_conv_size"]),
        int(cfg["temporal_conv_size"]), int(cfg["hidden_size"]))
    if vr["bf16"]["params"] != res:
        raise ValueError("ernie bf16 vision role is not the resampler")
    if vr["fp16"]["params"] + res != vision_params:
        raise ValueError("ernie vision role does not match the tower")
    return "bf16"


def _minimax_vision_fmt(rel: dict, patch_params: int, vision_params: int) -> tuple[str, float]:
    """Activation dtype of MiniMax-M3 ``ACT2FN["gelu"]``, and bytes per vision parameter.

    The bf16 vision role is the tower except the conv, plus the two projector
    linears. The fp32 role is only ``patch_embedding`` (Conv3d, no bias).
    GELU follows the bf16 weights. LayerNorm stays fp32. The conv GEMM's
    ``w_fmt`` is fp32. ``patch_merge_mlp`` is the language ``other`` role.
    """
    vr = (rel.get("roles") or {}).get("vision") or {}
    keys = [k for k, v in vr.items() if isinstance(v, dict) and v.get("params")]
    if sorted(keys) != ["bf16", "fp32"]:
        raise ValueError("minimax vision role is not bf16 plus the conv")
    fp = vr["fp32"]
    bf = vr["bf16"]
    if fp.get("params") != patch_params or fp.get("bytes") != patch_params * 4:
        raise ValueError("minimax vision fp32 role is not the conv")
    bits = 8.0 * bf["bytes"] / bf["params"]
    if abs(bits - 16.0) > 1e-3:
        raise ValueError(f"minimax vision bf16 role is {bits} bits")
    if bf["params"] + fp["params"] != vision_params:
        raise ValueError("minimax vision role params do not match the tower")
    return "bf16", (bf["bytes"] + fp["bytes"]) / vision_params


def from_release(model_id: str, hf_id: str, rel: dict | None = None, cfg: dict | None = None) -> ModelSpec:
    rel = rel if rel is not None else load_release(hf_id)
    if cfg is None:
        if rel is None:
            raise ValueError(f"no release metadata for {hf_id}; run scripts/fetch_hf_release.py")
        cfg = rel["config"]
    if (cfg.get("model_type") or "") == "omniasr_ctc":
        if rel is None:
            raise ValueError(f"no release metadata for {hf_id}; run scripts/fetch_hf_release.py")
        return _from_omniasr_ctc(model_id, hf_id, cfg, rel)
    if (cfg.get("model_type") or "") == "granite_speech5_ctc":
        if rel is None:
            raise ValueError(f"no release metadata for {hf_id}; run scripts/fetch_hf_release.py")
        return _from_granite_speech5_ctc(model_id, hf_id, cfg, rel)
    if (cfg.get("model_type") or "") == "d1_omni":
        if rel is None:
            raise ValueError(f"no release metadata for {hf_id}; run scripts/fetch_hf_release.py")
        return _from_d1_omni(model_id, hf_id, cfg, rel)
    c = _tc(cfg)
    build, arch = _builder(c)
    layers, mtp, notes, cov = build(c)
    if rel and rel.get("storage_alias") == {"U16": "bf16"} and rel.get("fused_linears"):
        notes = list(notes) + [
            "Paddle safetensors 的 dtype U16 按 PaddlePaddle/Paddle "
            "flex_checkpoint/dcp/utils.py 的 _SAFETENSORS_DTYPE_MAPPING 作为 bfloat16 运输"
            "（每个元素 2 字节；numpy 没有 bfloat16）。"
            "qkv_proj 的最后一维是 (num_attention_heads + 2·num_key_value_heads)·head_dim，"
            "拆开就是现有的 q / k / v。"
            "up_gate_proj 的最后一维是 2·intermediate（专家是 2·moe_intermediate），"
            "就是现有的 gate_up。down_proj 与 o_proj 已经是现有 GEMM。"
            "router 与 e_score_correction_bias 仍是 fp32。"
            "不另写 U16 GEMM，也不另写融合 QKV 算子。",
        ]
    gl = _moe_groups(c)
    if gl:
        layers = [replace(L, ffn=replace(L.ffn, groups=gl)) if L.ffn.kind == "moe" else L for L in layers]
        mtp = [replace(L, ffn=replace(L.ffn, groups=gl)) if L.ffn.kind == "moe" else L for L in mtp]
        notes = list(notes) + [f"分组限制路由：每个 token 只从 {gl[0]} 组中的 {gl[1]} 组选专家（n_group / topk_group）——"
                               "EP 各 rank 负载方差与节点去重 all-to-all 的扇出按此计「推导」"]
    from .drama3 import fish_spec
    fish = fish_spec(cfg, rel)
    if fish is not None:
        # 0.276 Drama 3: the codebook-MTP tower = 13 more decoder layers on every backbone row (core/drama3.py)
        from .drama3 import tower_layers
        layers = list(layers) + tower_layers(fish, Layer, Linear, _gqa, _ffn_dense)
    h = c["hidden_size"]
    tie = bool(c.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False)))
    # Mistral3Config.tie_word_embeddings defaults to True. The released JSON omits the key.
    # v4.52.4 maps language_model.lm_head onto lm_head and ties it to the text embedding.
    if ((cfg.get("model_type") or "") == "mistral3" and "tie_word_embeddings" not in c
            and "tie_word_embeddings" not in cfg):
        tie = True
    # Lfm2Config.__post_init__ pops tie_embedding into tie_word_embeddings (1.2B ships the alias only).
    mt_tie = (c.get("model_type") or cfg.get("model_type") or "")
    if mt_tie in ("lfm2", "lfm2_moe") and "tie_word_embeddings" not in c and "tie_word_embeddings" not in cfg:
        if "tie_embedding" in c or "tie_embedding" in cfg:
            tie = bool(c.get("tie_embedding", cfg.get("tie_embedding")))
    # Lfm2Vl text_config is an Lfm2Config. tie_word_embeddings defaults to True.
    # LFM2.5-VL-1.6B and VL-450M omit the key and omit tie_embedding.
    if ((cfg.get("model_type") or "") == "lfm2_vl" and mt_tie == "lfm2"
            and "tie_word_embeddings" not in c and "tie_embedding" not in c
            and "tie_word_embeddings" not in cfg):
        tie = True
    spec = ModelSpec(
        id=model_id, hf_id=hf_id, arch=arch, hidden=h, vocab=c["vocab_size"], layers=tuple(layers),
        tie_embeddings=tie, mtp_layers=tuple(mtp), mtp_extra_params=2 * h * h if mtp else 0,
        coverage=cov, notes=tuple(notes))
    if (c.get("model_type") or "") == "dory":
        # release default: config n_recurrent_loops, per-loop KV (vLLM DoryConfig default / README serve command)
        spec = with_loops(replace(spec, loop_spec=(c["n_input_layers"], c["n_recurrent_layers"]), loop_phys=spec.layers),
                          c.get("n_recurrent_loops") or 1, "per_loop")
    if (c.get("model_type") or "") in ("gemma4_text", "gemma4_unified_text"):
        ple_w = int(c.get("hidden_size_per_layer_input") or 0)
        spec = replace(spec, cached_params=_gemma4_unloaded_kv(c),
                       **({"final_norm_params": h + ple_w} if ple_w else {}))
    if rel is not None:
        fm = _release_formats(rel)
        if not any(l.role == "other" for L in spec.layers + spec.mtp_layers for l in L.attn_linears):
            fm.pop("other", None)             # 0.280: only models with role-"other" Linears (Qwen4Exp) carry it
        fm.update(_w4a4_roles(rel, fm))
        # 0.358: encoder-decoder (Whisper) release_params includes encoder + decoder
        whisper_enc_dec = (c.get("model_type") or "") == "whisper"
        rp = rel.get("params_llm") or 0
        if whisper_enc_dec:
            rp += rel.get("params_vision") or 0
        spec = replace(spec, formats=tuple(sorted(fm.items())), kv_fmt=_kv_fmt(rel), act_fmt=_act_fmt(rel),
                       release_params=rp, release_mtp_params=rel.get("params_mtp"),
                       release_bytes=rel.get("bytes_llm"), quantized_release=_quant_release(rel))
        # lookup-only tables (n-gram / engram embeddings): stored, read a few rows per token
        emb_rel = sum(v["params"] for v in rel.get("roles", {}).get("embed", {}).values())
        cosy = (cfg.get("model_type") or "") in ("cosyvoice2", "cosyvoice3")
        if _is_longcat_ngram(c):
            # 0.400: tables are the embed surplus; projections are io_pre. Do not dump both into lookup.
            spec = _attach_longcat_ngram(spec, rel, c)
        elif emb_rel > 1.05 * spec.embed_params and cosy:
            # 0.271 CosyVoice: text embedding (prompt text, a few rows / token) + the stored text lm_head (never read by
            # the speech-token loop) — stored in fp32 as released
            spec = replace(spec, lookup_params=emb_rel - spec.embed_params, lookup_bits=32.0,
                           notes=spec.notes + (f"文本 embedding + 文本 lm_head（{(emb_rel - spec.embed_params) / 1e9:.3f}B，"
                                               "fp32）：只存储 / 按行查表；解码头为 llm_decoder（语音 token 表）",))
        elif emb_rel > 1.05 * spec.embed_params and fish is not None:
            # 0.276 Drama 3: the 252 288-row text embedding + the frame-input codebook embeddings (10 × 4096) + the
            # tower's code tables — a few rows per frame each
            spec = replace(spec, lookup_params=emb_rel - spec.embed_params, lookup_bits=spec.fmt("embed").bits,
                           notes=spec.notes + (f"查表 {(emb_rel - spec.embed_params) / 1e9:.2f}B：文本 embedding（252 288 行）"
                                               "+ 帧输入码本 embedding（10 × 4096）+ 塔层码表；每帧只读少量行",))
        elif emb_rel > 1.05 * spec.embed_params and (cfg.get("model_type") or "") == "qwen3_tts":
            # 0.338: text_embedding (prompt rows) + codec_embedding (per-frame); surplus = text table
            spec = replace(spec, lookup_params=emb_rel - spec.embed_params, lookup_bits=16.0,
                           notes=spec.notes + (f"文本 embedding（{(emb_rel - spec.embed_params) / 1e9:.2f}B，bf16）："
                                               "提示查表；每帧用 codec_embedding",))
        elif emb_rel > 1.05 * spec.embed_params and (cfg.get("model_type") or "") == "moss_tts_delay":
            # 0.346: language_model.embed_tokens + emb_ext[n_vq] audio codebooks 「来源」; surplus = emb_ext
            spec = replace(spec, lookup_params=emb_rel - spec.embed_params, lookup_bits=16.0,
                           notes=spec.notes + (f"音频码本 embedding emb_ext（{(emb_rel - spec.embed_params) / 1e9:.2f}B，bf16）："
                                               "每帧按行查表并加到文本嵌入",))
        elif emb_rel > 1.05 * spec.embed_params and (c.get("model_type") or "") in (
                "gemma4_text", "gemma4_unified_text") and (c.get("hidden_size_per_layer_input") or 0):
            # 0.369: embed_tokens_per_layer is Embedding(vocab_ple, layers * ple), one row per token.
            # The projection GEMM and the per-layer gate / proj are real Linears (not this table).
            ple = int(c["hidden_size_per_layer_input"])
            packed = c["num_hidden_layers"] * ple
            vocab_ple = c.get("vocab_size_per_layer_input") or c["vocab_size"]
            expect = int(vocab_ple) * packed
            surplus = emb_rel - spec.embed_params
            if surplus != expect:
                raise ValueError(f"gemma4 PLE table {surplus} != vocab_ple * layers * ple ({expect})")
            spec = replace(spec, lookup_params=surplus, lookup_bits=spec.fmt("embed").bits, ple_packed=packed,
                           notes=spec.notes + (
                               f"逐层嵌入表（{surplus / 1e9:.3f}B，{spec.fmt('embed').fmt}）：每 token 读一行，宽 {packed}",))
        elif emb_rel > 1.05 * spec.embed_params and (cfg.get("model_type") or "") == "rwkv8":
            # 0.361: ffn.enn (+ hybrid qkv.ennk / qkv.emm) live in the embed role.  Bits are that role's
            # stored width (bf16 pile, fp16 hybrid).  Rows are read, not a hash table.
            spec = replace(spec, lookup_params=emb_rel - spec.embed_params, lookup_bits=spec.fmt("embed").bits,
                           notes=spec.notes + (f"DeepEmbed 查表（{(emb_rel - spec.embed_params) / 1e9:.3f}B，"
                                               f"{spec.fmt('embed').fmt}）：通道混合读当前 token；"
                                               "DeepEmbedAttention 读缓存中每个 token 的 ennk / emm 行",))
        elif emb_rel > 1.05 * spec.embed_params:
            spec = replace(spec, lookup_params=emb_rel - spec.embed_params,
                           notes=spec.notes + (f"查表 {(emb_rel - spec.embed_params) / 1e9:.1f}B 参数（n-gram / engram），"
                                               "按发布 safetensors 头计入存储；每 token 只读少量行",))
        # MTP modules: stored size as released (incl. embed/head copies some releases store)
        rmtp = rel.get("params_mtp") or 0
        if rmtp and not spec.mtp_layers and not spec.ngram_dim and not _is_longcat_causal(c):
            # weights ship an MTP module the config does not declare (e.g. Qwen3-Next) → infer one
            # 0.400 LongCat-Flash-Lite keeps the checkpoint MTP in params_mtp and does not run it.
            # 0.401 LongcatForCausalLM emits the vLLM draft in the builder, or drops it below.
            last = spec.layers[-1]
            spec = replace(spec, mtp_layers=(replace(last, misc_params=last.misc_params + 2 * h),),
                           notes=spec.notes + ("MTP 模块由发布权重推断（config 中没有 MTP 字段）",))
        if spec.mtp_layers and not rmtp and rel.get("params_mtp") == 0:
            # 0.144: the config declares MTP (mtp_num_hidden_layers) but the release ships no MTP weights
            # (MiniCPM-V-4.7 keeps the Qwen3.5 text config) → no draft module.
            # 0.401 Prover: same LongcatForCausalLM geometry, headers have no model.mtp.*.
            if _is_longcat_causal(c):
                dropped = "checkpoint 没有 model.mtp.*。不建 draft，也不把 V 补到 qk_head_dim。"
            else:
                dropped = "config 声明 MTP，但发布权重中没有 MTP 模块 → 不建模 MTP"
            spec = replace(spec, mtp_layers=(), mtp_extra_params=0,
                           notes=spec.notes + (dropped,))
        head_copy = 0
        if tie and "lm_head" in rel.get("roles", {}) and not rel.get("decision"):   # 0.291: a readout is no tied copy
            head_copy = sum(v["params"] for v in rel["roles"]["lm_head"].values())
        if spec.mtp_layers and rmtp:
            per = spec.mtp_layers[0].params(h)
            n_eff = max(1, min(len(spec.mtp_layers), round(rmtp / max(per, 1))))
            extra = max(0, (rmtp - n_eff * per) // n_eff)
            spec = replace(spec, mtp_layers=spec.mtp_layers[:n_eff], mtp_extra_params=extra)
        mf = _mtp_extra_format(rel, spec)
        if mf is not None:
            fm = dict(spec.formats)
            fm["mtp"] = mf
            spec = replace(spec, formats=tuple(sorted(fm.items())))
        split = _gdn_bf16_attn_split(rel, spec)
        if split is not None:
            lo, hi = split
            fm = dict(spec.formats)
            fm["attn"] = lo
            fm["lin_ba"] = hi
            layers = tuple(
                replace(L, attn_linears=tuple(
                    replace(l, role="lin_ba") if l.name == "lin_ba" else l for l in L.attn_linears))
                if any(l.name == "lin_ba" for l in L.attn_linears) else L
                for L in spec.layers)
            spec = replace(spec, formats=tuple(sorted(fm.items())), layers=layers)
        o_split = _o_proj_bf16_attn_split(rel, spec)
        mtp_o = _mtp_oproj_bf16_format(rel, spec)
        if o_split is not None:
            lo_o, hi_o = o_split
            fm = dict(spec.formats)
            fm["attn"] = lo_o
            fm["o_proj"] = hi_o
            mtp_layers = spec.mtp_layers
            notes = spec.notes + (
                "o_proj 与 attention sink 是 bf16（恰好 2 字节/参数），从 attn 角色拆出；"
                "q/k/v 按该角色的低精度桶计（含 block scale）。",)
            if mtp_o is not None:
                mtp_layers = _retag_o_proj(spec.mtp_layers)
                fm["mtp"] = mtp_o
                notes = notes + ("MTP 的 eh_proj 与 o_proj 按 bf16 计，不再使用 mtp 角色的平均位宽。",)
            spec = replace(spec, formats=tuple(sorted(fm.items())), layers=_retag_o_proj(spec.layers),
                           mtp_layers=mtp_layers, notes=notes)
        elif mtp_o is not None:
            # The bf16 bucket contains MTP o, but the main-stack attn split did not match, so those o
            # projections stay on the attn role. Retag only the MTP copies.
            fm = dict(spec.formats)
            fm["mtp"] = mtp_o
            fm["o_proj"] = RoleFormat("bf16", 16.0)
            spec = replace(spec, formats=tuple(sorted(fm.items())),
                           mtp_layers=_retag_o_proj(spec.mtp_layers),
                           notes=spec.notes + (
                               "MTP 的 bf16 桶恰好是 o_proj + eh_proj + 层内 misc，"
                               "eh_proj 与 MTP 的 o_proj 按 bf16 计。",))
        spec = _apply_longcat_fp8(spec, rel, c)
        spec = _attach_longcat_omni(spec, rel, c)
        if (c.get("model_type") or "").startswith("qwen4_exp") or (c.get("model_type") or "") == "dory":
            spec = replace(spec, final_norm_params=0)       # 0.253 Dory: nGPT, final state justnorm (no weight)
        if (c.get("model_type") or "") == "dory":
            # 0.321: lm_head.logit_scale [vocab] is an nn.Parameter; logits.float() × (logit_scale / initializer_range)
            # (vLLM PR #60149 DoryForCausalLM.compute_logits 「来源」) — ≤ 0.320 dropped as a quantisation scale
            spec = replace(spec, io_misc_params=spec.io_misc_params + c["vocab_size"], logit_scale=True)
        if (c.get("model_type") or "") in ("rwkv7", "rwkv8"):
            spec = replace(spec, final_norm_params=2 * h)      # ln_out LayerNorm (weight + bias)
        if (c.get("model_type") or "") == "whisper":
            # 0.270: decoder learned positions (looked up, one row per token) + final LayerNorm (weight + bias); the
            # encoder output length per 30 s window = max_source_positions (conv2 stride 2 over 3000 mel frames)
            spec = replace(spec, final_norm_params=2 * h, lookup_params=c["max_target_positions"] * h,
                           lookup_bits=spec.fmt("embed").bits, enc_ctx=c["max_source_positions"])
        if (cfg.get("model_type") or "") == "moshi" and isinstance(cfg.get("lm"), dict):
            # 0.274 Moshi: extra embedding rows looked up per frame — text_emb's padding row (text_card + 1) and the
            # n_q audio tables of card + 1 rows (summed into the frame input) 「来源」
            lm = cfg["lm"]
            spec = replace(spec, lookup_params=h + lm["n_q"] * (lm["card"] + 1) * h, lookup_bits=spec.fmt("embed").bits)
        if _is_kimi(cfg):
            # 0.273 Kimi-Audio: the audio head mimo_output [vocab, hidden] after mimo_norm — one row per sequence and
            # step like the lm_head (prefill: last position only 「推导」); final norm = norm + mimo_norm
            spec = replace(spec, io_post=spec.io_post + (Linear("mimo_output", h, c["vocab_size"], "col", rows="seq",
                                                                 role="lm_head"),), final_norm_params=2 * h)
        if cfg.get("jev_prompt_attention", c.get("jev_prompt_attention")) == "bidirectional":
            # 0.191 StartLux-Decision v1.1 Instant (model card): in the full-attention layers prompt tokens also attend to
            # later prompt tokens; linear-attention layers stay causal → the release's full-attention cores non-causal
            spec = replace(spec, layers=tuple(replace(L, core=replace(L.core, causal=False))
                                              if L.core is not None and L.core.kind in ("gqa", "mla") else L
                                              for L in spec.layers),
                           notes=spec.notes + ("jev_prompt_attention = bidirectional：全注意力层（1/4）prompt 内双向"
                                               "（发布 config / 模型卡），线性注意力层保持因果",))    # the final hyper_connection_mixer replaces the last norm
        dec = (rel or {}).get("decision")
        if dec:
            # 0.291 pplx-decider (model card + decision_config.json / joint_config.json 「来源」): Qwen3_5Model backbone, no
            # lm_head; readout.safetensors BF16 [255, hidden] applied at each question's <|fim_suffix|> marker (v1.2
            # joint-question-inline-readout-v1) or at the last token (v1 / v1.1, one question per sequence).
            # attention_mode noncausal_full_attention (v1.1, v1.2): the full-attention layers see the whole request,
            # the linear-attention layers stay causal; v1 has no attention_mode → causal
            nc = dec.get("attention_mode") == "noncausal_full_attention"
            if nc:
                spec = replace(spec, layers=tuple(replace(L, core=replace(L.core, causal=False))
                                                  if L.core is not None and L.core.kind in ("gqa", "mla") else L
                                                  for L in spec.layers))
            j = dec.get("joint") or {}
            spec = replace(spec, readout_rows=int(dec["readout"][0]), notes=spec.notes + (
                f"决策读出头 readout [{dec['readout'][0]}, {dec['readout'][1]}]（bf16，替代 lm_head；每题按答案代码行取 logits）"
                + ("；全注意力层非因果（attention_mode = noncausal_full_attention），线性注意力层保持因果" if nc else "；因果注意力")
                + ("；联合请求格式：一个序列含全部问题，每题一个 <|fim_suffix|> 读出位置（serving.questions）"
                   if j.get("template") == "joint-question-inline-readout-v1" else "；每序列一题，读末位置"),))
        if (c.get("model_type") or "").startswith("deepseek_v41"):
            # Engram tables stay fp8 with an e8m0 scale per 32 (ParallelEngramEmbedding) → 8.25 bits / param stored
            spec = replace(spec, lookup_bits=8 + 8 / 32, engram_ngram=c.get("engram_max_ngram_size") or 0,
                           draft="dspark" if c.get("dspark_block_size") else "",
                           draft_block=c.get("dspark_block_size") or 0,
                           draft_targets=len(_list(c.get("dspark_target_layer_ids")) or []),
                           draft_markov=c.get("dspark_markov_rank") or 0)
        if rel.get("dspark") and rel["dspark"].get("kind") == "ubtree":
            spec = _attach_ubtree(spec, rel["dspark"])            # 0.311: DFlash block + bigram tree selector
        elif rel.get("dspark") and rel["dspark"].get("kind") == "dflash2":
            spec = _attach_dflash2(spec, rel["dspark"])           # 0.254: DFlash2 block draft (conv + selector)
        elif rel.get("dspark") and (c.get("model_type") or "").startswith("muse_spark"):
            spec = _attach_muse_dspark(spec, rel["dspark"])
        elif rel.get("dspark") and rel["dspark"].get("kind") == "dspine":
            spec = _attach_dspine(spec, rel["dspark"])            # 0.252: DFlash anchor-row draft + code bridge
        elif rel.get("dspark"):
            spec = _attach_ready_dspark(spec, rel["dspark"])      # 0.221: DSpark-format draft for any other target
        exotic = [k for k in ("hc_mult", "hc_count", "mhc") if c.get(k)] \
            if not any(l.hc or l.gres_n for l in spec.layers) else []
        lk_open = spec.lookup_params and not spec.lookup_bits and not any(l.ple_rows for l in spec.layers)
        if exotic or lk_open:
            why = []
            if exotic:
                why.append("超连接（多流残差）：参数计入，混合计算未计")
            if lk_open:
                why.append("n-gram / engram 查表：只计存储")
            spec = replace(spec, coverage="proxy", notes=spec.notes + ("「架构代理」：" + "；".join(why),))
        if head_copy:
            spec = replace(spec, release_params=spec.release_params - head_copy,
                           notes=spec.notes + (f"发布另存了一份 tied lm_head（{head_copy / 1e9:.3f}B），"
                                               "已去重",))
        if fish is not None:
            # 0.276 Drama 3: 9 codebook heads (one row per sequence and step, like the lm_head); vllm-omni loads only the
            # 4096 candidate lm_head rows and drops eh_proj's embedding half in tower layers ≥ 1 (folded into the code
            # tables at load) 「来源」 → those release params are not resident
            d = fish.dim
            drop = (c.get("_full_vocab") or cfg["text_config"]["vocab_size"]) * h - spec.vocab * h \
                + (fish.tower_layers - 1) * d * d
            spec = replace(spec, io_post=spec.io_post + (Linear("codebook_heads", d, (fish.n_codebooks - 1) * fish.codebook_size,
                                                                 "col", rows="seq", role="lm_head"),),
                           release_params=spec.release_params - drop, fish=fish, notes=spec.notes + (
                               f"语音合成（TTS）：每步 = 1 帧（{fish.n_codebooks} 码本，{fish.frame_s * 1e3:.1f} ms 音频，44.1 kHz / 2048）"
                               f"= Qwen3.5-MoE 主干 1 token + 码本 MTP 塔 {fish.tower_layers} 层（{fish.shared_layers} 共享 + "
                               f"{fish.n_codebooks - 1} 逐码本，每层对整段序列做因果注意力、自带 KV）+ {fish.n_codebooks - 1} 个码本头；"
                               f"lm_head 只装 {fish.candidates} 个候选行；未装入的发布参数 {drop / 1e9:.3f}B 已扣除；"
                               + (f"DAC 解码器（codec.pth 解码路径 {sum(math.prod(t) for _, t in fish.codec) / 1e6:.0f} M 参数，fp32）"
                                  "按流式窗口在语音卡上计时（core/drama3.dac_ops）" if fish.codec else "DAC 解码器（codec.pth）未计"),))
        # 0.358: Whisper encoder-decoder gap check happens before vision_params is set, so skip it here
        if spec.release_params and (c.get("model_type") or "") != "whisper":
            gap = spec.release_params - spec.params()
            if abs(gap) / spec.release_params > 0.02:
                # parameters present in the release that the template does not reproduce
                spec = replace(spec, coverage="proxy" if cov != "full" or abs(gap) / spec.release_params > 0.05 else cov,
                               notes=spec.notes + (f"参数与发布相差 {gap / 1e9:+.2f}B "
                                                   f"（{gap / spec.release_params:+.1%}），见 docs/MODEL.md",))
        from .tts import tts_spec
        ts = tts_spec(cfg)
        if ts is not None:
            # 0.269: the release summary files flow DiT / stop / spk heads under params_llm (no "audio" in the name)
            # and the AudioVAE + aggregator under params_vision; the TTS spec carries all of them (exact, header)
            # 0.340 Ming-0.5B: summarize puts flowloss/stop/spk in role ``tts`` (outside params_llm) — do not subtract again
            side = 0 if (rel.get("roles") or {}).get("tts") else ts.llm_side_params
            spec = replace(spec, tts=ts, release_params=spec.release_params - side,
                           release_bytes=((spec.release_bytes or 0) - 2 * side) or None,
                           notes=spec.notes + (
                               f"语音合成（TTS）：每步 = LLM 1 token + 流匹配 DiT 头 {ts.flow_steps} 步 Euler × CFG {ts.cfg_mult} "
                               f"+ 聚合器 + 流式 AudioVAE 解码；每步 {ts.s_per_step * 1e3:.0f} ms 音频（44.1 kHz）；"
                               "lm_head 只存储（TTS 循环不用）",))
        from .fireredtts3 import fire_spec
        fs = fire_spec(cfg, rel)
        if fs is not None:
            # 0.275 FireRedTTS3: the speech head (release role tts) + RedAE (role codec) sit outside params_llm; the
            # spec replaces the (absent) LM head with the per-step DiT flow head + patch encoder (core/fireredtts3.py)
            ts = fs
            spec = replace(spec, tts=fs, notes=spec.notes + (
                f"语音合成（TTS）：每步 = Qwen3-1.7B 主干 1 步（bf16 autocast，权重 fp32 存储）+ DiT 流匹配头 {fs.flow_steps} 步 Euler × CFG {fs.cfg_mult}"
                f"（{fs.flow_params / 1e6:.0f}M，fp32，12 位置）+ patch 编码器（{fs.agg_params / 1e6:.0f}M）；每步 {fs.s_per_step * 1e3:.0f} ms 音频（24 kHz）；"
                f"整段生成完后 RedAE 一次解码（{fs.redae_dec_params / 1e6:.0f}M，fp32，非流式）；无 lm_head",))
        from .echo import echo_spec
        es = echo_spec(cfg, rel)
        if es is not None:
            # 0.279 Index-Echo S2ST: the ST LM is the catalog model; the Hidden2CV mapper (release role bridge, fp32) and
            # the CosyVoice3 stage (the fun-cosyvoice3 catalog model, byte-identical files) are costed by core/speech.py
            spec = replace(spec, echo=es, notes=spec.notes + (
                f"语音翻译（S2ST，非流式，一张卡串行「来源」）：AuT 编码 → ST LM 贪心解码（原文 + 译文）→ 再整段前向一次取末层隐状态"
                f" → Hidden2CV 映射器（{es.mapper_params / 1e6:.1f}M，fp32，{es.n_layer} 层 d{es.d_e}）→ CosyVoice3 LM（fp32）"
                f"采样语音 token（2m … min(20m, {es.speech_max})）→ flow + HiFT 一次合成",))
        from .cosyvoice import cosy_spec
        cs = cosy_spec(cfg, rel)
        if cs is not None:
            # 0.271 CosyVoice2 / 3: flow + HiFT (release roles flow / vocoder, outside params_llm) costed per output-audio
            # second by core/speech.py on a speech card 「假设」; the LM decodes 25 speech tokens per audio second
            spec = replace(spec, cosy=cs, notes=spec.notes + (
                f"语音输出：LM 每音频秒 {cs.token_rate} 个语音 token → flow（{cs.flow_params / 1e6:.1f}M，fp32，"
                f"{'Conformer 编码器 + DiT（K/V 缓存）' if cs.stream == 'cache' else 'Conformer 编码器 + U-Net' if cs.version == 2 else 'DiT'} 估计器，Euler {cs.steps} 步 × CFG {cs.cfg_mult}，"
                f"{'每块只算新 token，注意力走缓存（Step-Audio2 token2wav.stream）' if cs.stream == 'cache' else '每块从头重算全部已生成 token'}）+ HiFT（{cs.hift_params / 1e6:.1f}M，fp32，24 kHz）；放在每个副本一张额外卡上「假设」",))
        from .omni import omni_spec
        om = omni_spec(cfg)
        if om is not None:
            # 0.271 Qwen2.5-Omni: Talker (bf16) + token2wav DiT / BigVGAN (fp32) — release roles talker / token2wav
            # (outside params_llm); costed per output-audio second by core/speech.py on a speech card 「假设」
            spec = replace(spec, omni=om, notes=spec.notes + (
                f"语音输出：Talker（{om.talker_params / 1e9:.3f}B，bf16，每音频秒 {om.codec_per_s:.0f} 个 codec token）+ "
                f"token2wav DiT（{om.dit_params / 1e6:.1f}M，fp32，RK4 {om.evals} 次 × CFG {om.cfg_mult}，"
                f"{om.chunk} 帧分块流式）+ BigVGAN（{om.bigvgan_params / 1e6:.1f}M，fp32，24 kHz）；"
                "按每输出音频秒计（core/speech.py），放在每个副本一张额外卡上「假设」",))
        from .qwen3_tts import qwen3_tts_spec
        qs = qwen3_tts_spec(cfg, rel)
        if qs is not None:
            # 0.338 Qwen3-TTS: talker = catalog LLM; code predictor (depth) + speech tokenizer decode (codec)
            # + speaker encoder (spk, Base) costed per output-audio second by core/speech.py.
            # text_projection (release role other) is prompt-only — drop from params_llm like Ming llm_side.
            spec = replace(spec, qwen3_tts=qs,
                           release_params=(spec.release_params or 0) - qs.text_proj_params,
                           release_bytes=((spec.release_bytes or 0) - 2 * qs.text_proj_params) or None,
                           notes=spec.notes + (
                f"语音合成（TTS，12.5 Hz 多码本 LM）：Talker 每帧 1 步（bf16，首码本）+ code predictor "
                f"{qs.cp_steps} 步 × {qs.cp_layers} 层 d{qs.cp_h}（{qs.code_predictor_params / 1e6:.0f}M，bf16）"
                f"+ speech tokenizer 解码（{qs.tokenizer_dec_params / 1e6:.0f}M，fp32，{qs.sample_rate // 1000} kHz）；"
                f"每帧 {qs.frame_s * 1e3:.0f} ms 音频；放在每个副本一张额外卡上「假设」",))
        from .qwen3_omni import qwen3_omni_spec
        qo = qwen3_omni_spec(cfg, rel)
        if qo is not None:
            # 0.345 Qwen3-Omni: Talker MoE + code predictor + code2wav (bf16 as released) — roles talker / depth /
            # code2wav outside params_llm; costed per output-audio second by core/speech.py on a speech card 「假设」
            spec = replace(spec, qwen3_omni=qo, notes=spec.notes + (
                f"语音输出（Qwen3-Omni，12.5 Hz）：Talker MoE（{qo.talker_params / 1e9:.3f}B，bf16，"
                f"{qo.t_experts} 专家 top-{qo.t_topk}，每帧 1 步）+ code predictor {qo.cp_steps} 步 × "
                f"{qo.cp_layers} 层 d{qo.cp_h}（{qo.code_predictor_params / 1e6:.0f}M，bf16）+ code2wav "
                f"（{qo.code2wav_params / 1e6:.0f}M，bf16，upsample {qo.total_upsample} → {qo.sample_rate // 1000} kHz）；"
                f"每帧 {qo.frame_s * 1e3:.0f} ms；放在每个副本一张额外卡上「假设」",))
        from .moss_tts import moss_tts_spec
        ms = moss_tts_spec(cfg, rel)
        if ms is not None:
            # 0.346 MOSS-TTS Delay: Qwen3 backbone (catalog) + n_vq audio emb/heads on the same card 「来源」;
            # external audio tokenizer / vocoder not in checkpoint — waveform decode unmodeled.
            # 0.347/0.348: audio lm_heads (params_depth) on the replica card → sidecar_bytes (bf16) for evaluate.fits
            spec = replace(spec, moss_tts=ms,
                           sidecar_params=ms.audio_head_params, sidecar_bits=16.0,
                           sidecar_bytes=ms.audio_head_params * 2.0,
                           notes=spec.notes + (
                f"语音输出（MOSS-TTS Delay，{ms.frame_hz} Hz）：Qwen3 主干 + {ms.n_vq} 路码本 delay"
                f"（emb_ext + audio lm_heads {ms.audio_head_params / 1e6:.0f}M，bf16）；"
                f"首音频约 {ms.delay_frames} 帧 delay；外置 audio tokenizer 未计",))
        from .moshi import moshi_spec
        ms = moshi_spec(cfg, rel)
        if ms is not None:
            # 0.347/0.348: depth (bf16) + Mimi (fp32) on the replica card → sidecar_bytes (= MoshiSpec.stored_bytes)
            spec = replace(spec, moshi=ms,
                           sidecar_params=ms.depth_params, sidecar_bits=16.0,
                           sidecar_bytes=ms.stored_bytes,
                           notes=spec.notes + (
                f"全双工语音：每 80 ms 帧 = Mimi 编码（用户音频，fp32）+ 时间 Transformer 1 步（本模型，KV 环形缓存 {ms.context} 帧）"
                f"+ 深度 Transformer {ms.dep_q} 步 × {ms.dep_layers} 层（{ms.depth_params / 1e6:.0f}M，bf16，每步独立权重切片）"
                f"+ Mimi 解码（{ms.mimi_params / 1e6:.0f}M，fp32）；全部串行在同一张卡上「假设」",))
        from .kimi_audio import kimi_spec
        ks = kimi_spec(cfg, rel)
        if ks is not None:
            # 0.273 Kimi-Audio: detokenizer DiT + BigVGAN (release roles flow / vocoder, outside params_llm), costed per
            # output-audio second by core/speech.py on a speech card 「假设」
            spec = replace(spec, kimi=ks, notes=spec.notes + (
                f"语音输出：每步同时出 1 个文本 token（lm_head）+ 1 个音频 token（mimo_output，{ks.token_rate} Hz，前 {ks.delay} 步为空）"
                f"→ 去分词器 DiT（{ks.dit_params / 1e9:.3f}B，bf16，{ks.depth} 层 d{ks.d}，每帧 adaLN；{ks.chunk} token 一块 + {ks.look_ahead} token 前瞻，"
                f"Euler {ks.evals} 次 + 1 次 K/V 提交，K/V 缓存 ≤ {ks.max_kv} 帧）+ BigVGAN（{ks.voc_params / 1e6:.1f}M，bf16，24 kHz）；"
                "放在每个副本一张额外卡上「假设」",))
        rel_gap = (spec.release_params - spec.params()) / spec.release_params if spec.release_params else 0.0
        mp = c.get("max_position_embeddings")
        from .vision import vision_spec
        if cfg.get("model_type") == "index_echo_s2st":      # 0.279: connector shapes (stlm_ckpt) for the audio spec
            cfg = {**cfg, "_echo_connector": {n[len("connector."):]: v for n, v in (rel.get("echo_tensors") or {}).items()
                                              if n.startswith("connector.")}}
        vs_cfg = ({**cfg, "_kimi_whisper": rel["kimi_audio"]["whisper"]}
                  if _is_kimi(cfg) and rel.get("kimi_audio") else cfg)     # 0.273: Kimi-Audio whisper config
        if _is_longcat_omni(c):
            meta = rel.get("longcat_omni")
            if not isinstance(meta, dict):
                raise ValueError("longcat omni tower header is missing")
            vs_cfg = {**vs_cfg, "longcat_omni": meta}
        vs = vision_spec(hf_id, vs_cfg)
        rv = (rel.get("params_vision") or 0) if ts is None else 0
        if vs is not None and vs.family == "longcatvit" and rv != vs.params:
            raise ValueError(f"longcat omni towers {vs.params} != release vision {rv}")
        if vs is not None and rv > vs.params:
            # 0.144: multimodal parts stored with the release but not run for images (MiMo audio encoder + speech
            # embeddings) → storage only
            vs = replace(vs, extra=rv - vs.params, params=rv)
        if vs is not None and rel.get("vision_fmt"):
            vs = replace(vs, **_vision_formats(rel, vs))
        # 0.388: a Qwen3-VL tower stored entirely as fp32 (config dtype may still say bfloat16).
        # Language roles already take the safetensors dtype. The tower GEMMs follow the same file dtype;
        # activations stay the model's act format (no activation quantizer in Qwen3VLVisionModel).
        if vs is not None and vs.family == "qwen" and not vs.wfmt:
            vr = (rel.get("roles") or {}).get("vision") or {}
            block = vr.get("fp32") or {}
            if set(vr) == {"fp32"} and block.get("params") == vs.params and block.get("params"):
                bits = 8.0 * block["bytes"] / block["params"]
                if abs(bits - 32.0) > 1e-6:
                    raise ValueError(f"qwen vision fp32 role is {bits} bits, not 32")
                act = spec.act_fmt
                vs = replace(vs, w_Bpp=bits / 8.0, wfmt=tuple(
                    (k, "fp32", bits, act) for k in ("patch", "attn", "mlp", "merger")))
                spec = replace(spec, notes=spec.notes + (
                    "safetensors 是 fp32（config dtype 为 bfloat16）。语言 GEMM 与视觉塔权重按 fp32 计，激活仍是 bf16。",))
        if vs is not None and getattr(vs, "audio", None) is not None and vs.audio.kind == "whisper":
            # 0.415: vector activations follow the safetensors vision role. GEMM w_fmt stays bf16.
            if rel is None:
                raise ValueError("whisper tower vision role is missing")
            vs = replace(vs, audio=replace(vs.audio, vec_fmt=_whisper_vision_fmt(rel)))
        if vs is not None and getattr(vs, "audio", None) is not None and vs.audio.kind == "aut":
            # 0.416: same vision-role dtype for AuT GELU. GEMM w_fmt stays bf16.
            if rel is None:
                raise ValueError("aut tower vision role is missing")
            vs = replace(vs, audio=replace(vs.audio, vec_fmt=_whisper_vision_fmt(rel, "aut")))
        if vs is not None and vs.family == "pixtral":
            # 0.418: SiLU, the product, and projector GELU follow the vision role.
            # GEMM w_fmt stays bf16. RMSNorm stays fp32.
            if rel is None:
                raise ValueError("pixtral tower vision role is missing")
            vs = replace(vs, vec_fmt=_whisper_vision_fmt(rel, "pixtral"))
        if vs is not None and vs.ln_mode == "qwen3vl":
            # 0.419: merger nn.GELU() does not upcast. A fp32 vision role is the
            # WeVisOne storage dtype; those activations stay the model format.
            if rel is None:
                raise ValueError("qwen3vl tower vision role is missing")
            vr = (rel.get("roles") or {}).get("vision") or {}
            if set(vr) == {"fp32"}:
                act = spec.act_fmt
                if act not in ("fp16", "bf16", "fp32"):
                    raise ValueError("qwen3vl activation dtype is not a vision float")
                vs = replace(vs, vec_fmt=act)
            else:
                vs = replace(vs, vec_fmt=_whisper_vision_fmt(rel, "qwen3vl"))
        if vs is not None and vs.family == "lfm2vl":
            # 0.419: projector ACT2FN["gelu"] follows the vision role. LayerNorm stays fp32.
            if rel is None:
                raise ValueError("lfm2_vl tower vision role is missing")
            vs = replace(vs, vec_fmt=_whisper_vision_fmt(rel, "lfm2_vl"))
        if vs is not None and vs.family == "glm":
            # 0.422: gate SiLU, the product, and merger nn.GELU() follow the vision role.
            # GEMM w_fmt stays bf16. RMSNorm and LayerNorm stay fp32.
            if rel is None:
                raise ValueError("glm5 vision role is missing")
            vs = replace(vs, vec_fmt=_whisper_vision_fmt(rel, "glm5"))
        if vs is not None and vs.ln_mode == "mimo":
            # 0.423: gate SiLU, the product, and merger nn.GELU() follow the vision role.
            # GEMM w_fmt stays bf16. RMSNorm and LayerNorm stay fp32.
            if rel is None:
                raise ValueError("mimo vision role is missing")
            vs = replace(vs, vec_fmt=_whisper_vision_fmt(rel, "mimo"))
        if vs is not None and vs.ln_mode == "qwen35":
            # 0.424: merger nn.GELU() follows the one vision-role float. LayerNorm
            # stays fp32. A tower stored as more than one vision dtype (the FP4-Local
            # mix) is not assigned a vector dtype and is not billed.
            if rel is None:
                raise ValueError("qwen35 tower vision role is missing")
            vr = (rel.get("roles") or {}).get("vision") or {}
            keys = [k for k, v in vr.items() if isinstance(v, dict) and v.get("params")]
            if len(keys) > 1:
                vs = replace(vs, ln_mode="")
            else:
                vs = replace(vs, vec_fmt=_whisper_vision_fmt(rel, "qwen35"))
        if vs is not None and vs.ln_mode == "ling":
            # 0.431: linear_proj nn.GELU() follows the one vision-role float.
            # LayerNorm stays fp32. GEMM w_fmt stays bf16.
            if rel is None:
                raise ValueError("ling tower vision role is missing")
            vs = replace(vs, vec_fmt=_whisper_vision_fmt(rel, "ling"))
        if vs is not None and vs.family == "radio":
            # 0.425: MLP GELU follows the bf16 tower role. The six fp32 values are
            # the conditioner buffers. LayerNorm and RMSNorm stay fp32.
            if rel is None:
                raise ValueError("radio tower vision role is missing")
            vs = replace(vs, vec_fmt=_radio_vision_fmt(rel))
        if vs is not None and vs.family == "ds":
            # 0.426: F.silu, the product, and F.gelu follow the one vision-role float.
            # GEMM w_fmt stays bf16. RMSNorm stays fp32.
            if rel is None:
                raise ValueError("deepseek vision role is missing")
            vs = replace(vs, vec_fmt=_whisper_vision_fmt(rel, "deepseek"))
        if vs is not None and vs.family == "kimi" and vs.ln_mode in ("k25", "k3"):
            # 0.427: projector nn.GELU() follows the one vision-role float.
            # GEMM w_fmt stays bf16. LayerNorm and RMSNorm stay fp32.
            if rel is None:
                raise ValueError("kimi vision role is missing")
            vs = replace(vs, vec_fmt=_whisper_vision_fmt(rel, "kimi"))
        if vs is not None and vs.ln_mode == "muse":
            # 0.428: MLP and adapter ACT2FN["gelu"] follow the one vision-role float.
            # 0.432: the waveform AutoEncoder's nn.GELU() uses the same float.
            # GEMM w_fmt stays bf16. LayerNorm and audio RMSNorm stay fp32.
            # A tower stored as more than one vision dtype is not assigned a
            # vector dtype and is not billed.
            if rel is None:
                raise ValueError("muse vision role is missing")
            vr = (rel.get("roles") or {}).get("vision") or {}
            keys = [k for k, v in vr.items() if isinstance(v, dict) and v.get("params")]
            if len(keys) > 1:
                vs = replace(vs, ln_mode="")
            else:
                fmt = _whisper_vision_fmt(rel, "muse")
                audio = vs.audio
                if audio is not None:
                    audio = replace(audio, vec_fmt=fmt)
                vs = replace(vs, vec_fmt=fmt, audio=audio)
        if vs is not None and vs.family == "ernievl":
            # 0.433: resampler nn.GELU() follows the bf16 bucket. The fp16
            # bucket is the tower. GEMM w_fmt stays bf16. Norms stay fp32.
            if rel is None:
                raise ValueError("ernie vision role is missing")
            vs = replace(vs, vec_fmt=_ernie_vision_fmt(rel, cfg, vs.params))
        if vs is not None and vs.family == "minimax":
            # 0.434: ACT2FN["gelu"] follows the bf16 tower weights. The fp32
            # bucket is the conv. GEMM labels stay bf16 except that conv.
            if rel is None:
                raise ValueError("minimax vision role is missing")
            patch = 3 * vs.tpatch * vs.patch * vs.patch * vs.hidden
            fmt, bpp = _minimax_vision_fmt(rel, patch, vs.params)
            vs = replace(vs, vec_fmt=fmt, w_Bpp=bpp)
        if vs is not None and vs.family == "minicpm":
            # 0.435: merger nn.GELU() follows the one vision-role float.
            # GEMM w_fmt stays bf16. LayerNorm stays fp32.
            if rel is None:
                raise ValueError("minicpm vision role is missing")
            vs = replace(vs, vec_fmt=_whisper_vision_fmt(rel, "minicpm"))
        from .echo import ECHO_REASON
        spec = replace(spec, coverage_reasons=_coverage_reasons(c, spec, rel_gap) + ((vs.approx,) if vs is not None and vs.approx else ())
                       + ((ECHO_REASON,) if spec.echo is not None else ()),
                       vision=vs, vision_params=vs.params if vs is not None else rv,
                       max_ctx=int(mp) if isinstance(mp, (int, float)) and not isinstance(mp, bool) and mp > 0 else 0)
        if spec.coverage == "full" and spec.coverage_reasons:
            spec = replace(spec, coverage="partial")
        if ((cfg.get("model_type") or "") in ("granite_speech", "granite_speech_plus")
                and vs is not None and getattr(vs, "audio", None) is not None):
            au = vs.audio
            note = (f"Granite Speech：Conformer 用 Shaw 分块相对位置（注意力按 context {au.chunk} 补齐），"
                    f"Q-Former 每窗 {au.gs_q} 个查询，交叉注意力 K 宽 {au.gs_cross}。"
                    "中间层 CTC 的 out 与 out_mid 各跑一次。"
                    "每一层的 SiLU、GLU、BatchNorm、0.5 残差和 LayerNorm 按发布的前向计成向量运算，不是 GEMM。"
                    "Q-Former 的 GELU 是 erf 形式，行数是查询，宽度是 intermediate。"
                    "eval 下的 Dropout 是恒等，不单列。"
                    "granite 的 attention / embedding / residual / logits 标量不另计 GEMM，也不改变 KV 宽度。没有 MTP。")
            if cfg.get("model_type") == "granite_speech_plus":
                note += (" plus 把 cat_hidden_layers 的隐状态拼在最后一层上，只加宽交叉注意力的 K。"
                         "文本 config dtype 是 float32，safetensors 是 bf16，按 bf16 计。")
            spec = replace(spec, notes=spec.notes + (note,))
        if (vs is not None and getattr(vs, "audio", None) is not None and vs.audio.kind == "w2v"
                and not vs.audio.ctc):
            spec = replace(spec, notes=spec.notes + (
                "OmniASR 音频塔的 GELU 是 ACT2FN['gelu']，erf 形式，不是 tanh 近似。"
                "特征卷积每一层、位置卷积、以及每一层 FFN 的 intermediate 各计一次，宽度分别是 conv 通道、"
                "hidden 和 intermediate。不升精度，dtype 跟随发布的权重格式。"
                "multi_modal_projector 没有 GELU。eval 的 Dropout 和 LayerDrop 是恒等，不单列。"
                "特征卷积的 LayerNorm 在 GELU 之前，宽度是 conv 通道，不是 hidden。"
                "投影 Linear 之前还有一次 LayerNorm，宽度仍是最后一层通道。"
                "每一层注意力前和 FFN 前各有一次 LayerNorm(hidden)，编码器末尾再有一次。"
                "LayerNorm 在 fp32 归约，不是 GEMM。位置卷积没有 LayerNorm。"
                "不是 0.5 的残差加不单列。",))
        if (vs is not None and getattr(vs, "audio", None) is not None and vs.audio.kind == "g4conf"):
            spec = replace(spec, notes=spec.notes + (
                "gemma4_audio 的 hidden_act 是 silu，residual_weight 是 0.5。"
                "下采样两层各有一次 mask 乘、一次 bias=False 的 LayerNorm 和一次 nn.ReLU。"
                "第一次 mask 读特征提取器的 fp32，在卷积 cast 到权重 dtype 之前。"
                "第二次 mask 读 bf16 的 ReLU 输出。"
                "每一层 FFN 的 SiLU 宽度是 4*hidden。LightConv 的 GLU 和卷积 SiLU 宽度是 hidden，不是 2*hidden。"
                "0.5 残差在 FFN 的 post RMSNorm 之后，宽度是 hidden。"
                "带 weight 的 RMSNorm 升到 fp32，pow(2).mean 再一次 pow(-0.5)，乘 weight，每行一次。"
                "embed_audio 的 RMSNorm 没有 scale，不单列。"
                "softcap、softplus、clamp、整份残差加和正弦表不单列。"
                "视觉 hidden_activation 仍是 gelu_pytorch_tanh，不把 GeGLU 记成 SiLU 或 erf-GELU。"
                "通用 GEMM 尾项不减。",))
        if vs is not None and vs.family == "gemma4":
            spec = replace(spec, notes=spec.notes + (
                "Gemma-4 视觉塔：每一层的 input_layernorm、post_attention_layernorm、"
                "pre_feedforward_layernorm、post_feedforward_layernorm 是 Gemma4RMSNorm(hidden)，带 weight。"
                "q_norm 和 k_norm 是 Gemma4RMSNorm(head_dim)，按头做一次 rsqrt，不是把多头展成 hidden 再做一次。"
                "v_norm 的 with_scale 是 False，不单列。"
                "MLP 是 down(ACT2FN[hidden_activation](gate) * up)，发布值是 gelu_pytorch_tanh，"
                "不把 GeGLU 记成 SiLU 或 erf-GELU。"
                "RMSNorm 升到 fp32，pow(2).mean 再一次 pow(-0.5)，乘 weight。"
                "整份残差加、RoPE、clamp、像素的 2*(x-0.5) 不单列。通用 GEMM 尾项不减。",))
        if vs is not None and vs.family == "gemma4u":
            spec = replace(spec, notes=spec.notes + (
                "Gemma-4 unified 视觉：patch_ln1 是 nn.LayerNorm(patch_dim)，在 dense 之前。"
                "patch_ln2 是 nn.LayerNorm(mm)，在 dense 之后、位置加之前。"
                "pos_norm 是 nn.LayerNorm(mm)，在位置加之后。"
                "投影前的 Gemma4UnifiedRMSNorm 是 with_scale=False，不单列。"
                "LayerNorm 在 fp32 归约，仿射已经在参数里。"
                "位置嵌入的 gather、padding 乘和两轴相加不单列。通用 GEMM 尾项不减。",))
        if (vs is not None and getattr(vs, "audio", None) is not None and vs.audio.kind == "sanm"):
            au = vs.audio
            spec = replace(spec, notes=spec.notes + (
                "Fun-ASR-Nano：SAN-M 每一层是输入 LayerNorm、注意力、FSMN 的 mask 乘、post LayerNorm、"
                "ACT2FN['relu']。输出 LayerNorm 只在第 "
                f"{au.sanm_final[0]} 层和第 {au.sanm_final[1]} 层，其余是 Identity。"
                "第 0 层的输入 LayerNorm 宽度是堆叠后的 mel，不是 hidden。"
                "nn.ReLU 是 max(x, 0)，一次截断，不是 relu6。"
                "投影器在两层 Linear 之间再跑一次 ReLU，宽度是 projector_hidden_size。"
                "adaptor 没有 FSMN，也没有输出 LayerNorm，仍有两次 LayerNorm 和一次 ReLU，"
                "ReLU 的宽度是 adaptor 的 intermediate。"
                "LayerNorm 在 fp32 归约。ReLU 和 mask 乘不升精度，发布权重是 bf16。"
                "eval 的 Dropout 是恒等。配置里的 activation_dropout 类没有读。"
                "sqrt(d) 和正弦位置表不单列。通用 GEMM 尾项不减。",))
        if (vs is not None and getattr(vs, "audio", None) is not None and vs.audio.kind == "whisper"):
            au = vs.audio
            spec = replace(spec, notes=spec.notes + (
                "共用 Whisper 编码器：conv1 和 conv2 的 nn.functional.gelu、每一层 MLP 的 "
                "ACT2FN['gelu'] 或 nn.GELU() 都是 erf 形式（0.5 * x * (1 + erf(x / sqrt(2)))），"
                "不是 gelu_pytorch_tanh，也不是 quick_gelu。"
                "每一层先计注意力前的 LayerNorm，再计 MLP 前的 LayerNorm。"
                "编码器 LayerNorm：Whisper 在整段编码器位置上；"
                "Qwen2-Audio 和 Qwen2.5-Omni 在 stride-2 池化之后；"
                "Step-Audio 的 after_norm 也在池化之后，adaptor 再计 F.gelu 和 nn.ReLU；"
                "Kimi 的编码器 LayerNorm 在 4 路堆叠之前，VQAdaptor 再计 nn.SiLU 和 LayerNorm；"
                "Ming 的 ln_post 在投影器之前，投影器再计 nn.GELU()。"
                f"GELU、SiLU 和 ReLU 不升精度，a_fmt 是 vision 角色的 {au.vec_fmt}，"
                "不读 config 的 torch_dtype，也不跟请求里的 GEMM 激活格式走。"
                "LayerNorm 在 fp32 归约。GEMM 的 w_fmt 仍是历史的 bf16，通用 GEMM 尾项不减。"
                "解码器仍把 GELU 按 SiLU 的逐元素开销计，这次不改。"
                "eval 的 Dropout 是恒等，不单列。"
                "fp16 clamp、Omni 乘 long 的 mask、Ming 的 F.normalize 不单列。",))
        if (vs is not None and getattr(vs, "audio", None) is not None and vs.audio.kind == "aut"):
            au = vs.audio
            spec = replace(spec, notes=spec.notes + (
                "AuT 编码器：三层 Conv2d 各跑一次 F.gelu，每一层是注意力前的 LayerNorm、"
                "MLP 前的 LayerNorm、ACT2FN['gelu']。"
                "编码器 LayerNorm 在块之后。投影器是 Linear、同一次激活、Linear。"
                "F.gelu 和 ACT2FN['gelu'] 都是 erf 形式（0.5 * x * (1 + erf(x / sqrt(2)))），"
                "不是 gelu_pytorch_tanh，也不是 quick_gelu。"
                "卷积 GELU 读补齐后的 chunk 网格。LayerNorm、FFN GELU 和投影器读有效编码器位置。"
                f"GELU 不升精度，a_fmt 是 vision 角色的 {au.vec_fmt}，"
                "不读 config 的 torch_dtype，也不跟请求里的 GEMM 激活格式走。"
                "LayerNorm 在 fp32 归约。GEMM 的 w_fmt 仍是历史的 bf16，通用 GEMM 尾项不减。"
                "Index-Echo 2B 的 connector 在 w1 之后再计一次 F.gelu。9B 的 connector 是一次 Linear，没有激活。"
                "eval 的 Dropout 是恒等，不单列。"
                "正弦位置加、fp16 clamp、connector 的 exp(log_alpha) 和 beta 标量、整份残差加不单列。",))
        if vs is not None and vs.family == "pixtral":
            spec = replace(spec, notes=spec.notes + (
                "Pixtral 视觉塔：ln_pre、每一层注意力前和 SwiGLU 前的 RMSNorm、"
                "投影器在合并之前的 RMSNorm，都带 weight，升到 fp32，pow(2).mean 再一次 rsqrt。"
                "MLP 是 ACT2FN['silu']（x * sigmoid(x)）再乘 up。宽度是 intermediate，不是 2*intermediate。"
                "这不是 GLU。投影器在 linear_1 之后是 ACT2FN['gelu']，erf 形式，"
                "行数是合并后的位置，宽度是文本 hidden。"
                f"SiLU、乘积和 GELU 不升精度，a_fmt 是 vision 角色的 {vs.vec_fmt}，"
                "不读 config 的 torch_dtype，也不跟请求里的 GEMM 激活格式走。"
                "GEMM 的 w_fmt 仍是 bf16，通用 GEMM 尾项不减。"
                "RoPE 的 sin/cos、整份残差加、块状注意力 mask 不单列。",))
        if vs is not None and vs.ln_mode == "qwen3vl":
            spec = replace(spec, notes=spec.notes + (
                "Qwen3-VL 视觉塔：每一层注意力前和 MLP 前各一次 nn.LayerNorm(hidden)。"
                "最终 merger 在 view 之前对 patch 序列做 LayerNorm(hidden)。"
                "每个 deepstack merger 先 view 到 4*hidden，再做 LayerNorm(4*hidden)。"
                "merger 的激活是 nn.GELU()，默认 approximate='none'，erf 形式，"
                "行数是合并后的位置，宽度是 4*hidden，不是文本 hidden。"
                "块内 MLP 的 hidden_act 仍是 gelu_pytorch_tanh，不单列。"
                f"GELU 不升精度，a_fmt 是 {vs.vec_fmt}。"
                "权重是单独一份 fp32 时，这个 a_fmt 跟模型激活格式，不跟权重 dtype。"
                "其余塔跟 vision 角色。都不跟请求里的 GEMM 激活格式走。"
                "LayerNorm 在 fp32 归约。通用 GEMM 尾项不减。"
                "RoPE、位置插值、整份残差加不单列。",))
        if vs is not None and vs.family == "lfm2vl":
            spec = replace(spec, notes=spec.notes + (
                "LFM2.5-VL 的 SigLIP2：每一层注意力前和 MLP 前各一次 nn.LayerNorm(hidden)，"
                "块之后还有 post_layernorm。行数是补齐后的 patch，不是 unshuffle 之后的有效行。"
                "vision_use_head 是 false，pooling head 的 LayerNorm 不跑。"
                "投影器在 linear_1 和 linear_2 之间是 ACT2FN['gelu']，erf 形式，"
                "行数是 unshuffle 之后的有效位置，宽度是 projector_hidden_size。"
                "塔内 MLP 的 hidden_act 仍是 gelu_pytorch_tanh，不单列。"
                f"GELU 不升精度，a_fmt 是 vision 角色的 {vs.vec_fmt}，"
                "不跟请求里的 GEMM 激活格式走。"
                "LayerNorm 在 fp32 归约。projector_use_layernorm 为 false 时不另计投影器 LayerNorm。"
                "通用 GEMM 尾项不减。位置插值和整份残差加不单列。",))
        if vs is not None and vs.family == "glm":
            spec = replace(spec, notes=spec.notes + (
                "GLM-5.3-Flash 视觉塔：每一层 norm1 和 norm2 是带 weight 的 RMSNorm(hidden)。"
                "q_norm 和 k_norm 做在 (token, head, head_dim) 的最后一维，"
                "每头一次 rsqrt，不是把多头展成 hidden 再做一次。q 和 k 的头数相同。"
                "MLP 是 ACT2FN['silu']（x * sigmoid(x)）再乘 up。宽度是 intermediate，不是 2*intermediate。"
                "块之后 post_layernorm 仍在 patch 行上，宽度是 hidden，不是 out_hidden_size。"
                "merger 在 proj 之后是 nn.LayerNorm(out_hidden_size) 和 nn.GELU()，"
                "默认 approximate='none'，erf 形式。行数是合并后的位置。"
                "gate 上再跑一次 ACT2FN['silu']，宽度是 projection_intermediate_size，不是 2 倍。"
                f"SiLU、乘积和 GELU 不升精度，a_fmt 是 vision 角色的 {vs.vec_fmt}，"
                "不读 config 的 torch_dtype，也不跟请求里的 GEMM 激活格式走。"
                "RMSNorm 和 LayerNorm 在 fp32 归约。仿射已经在参数里。"
                "swiglu_limit 的 clamp、轴向 RoPE 的 cos/sin、整份残差加、"
                "eval 的 attention dropout 不单列。通用 GEMM 尾项不减。",))
        if vs is not None and vs.ln_mode == "mimo":
            spec = replace(spec, notes=spec.notes + (
                "MiMo-V2.6 视觉塔：每一层 norm1 和 norm2 是 nn.RMSNorm(hidden)，"
                "升到 fp32，pow(2).mean 再一次 rsqrt。宽度是 hidden，每行一次，不是按头。"
                "q 和 k 没有单独的 RMSNorm。"
                "MLP 是 ACT2FN['silu']（x * sigmoid(x)）再乘 up。宽度是 intermediate，不是 2*intermediate。"
                "merger 的 ln_q 是 nn.LayerNorm(hidden)，在 view 之前。行数是 patch，不是合并后的行。"
                "nn.GELU() 默认 approximate='none'，erf 形式。"
                "行数是合并后的位置，宽度是 4*hidden，不是 out_hidden_size。"
                f"SiLU、乘积和 GELU 不升精度，a_fmt 是 vision 角色的 {vs.vec_fmt}，"
                "不读 config 的 torch_dtype，也不跟请求里的 GEMM 激活格式走。"
                "RMSNorm 和 LayerNorm 在 fp32 归约。仿射已经在参数里。"
                "发布头里 merger 的 Linear 和 ln_q 没有 bias，类默认的 bias 不单列。"
                "RoPE 的 cos/sin、torch.outer、整份残差加、repeat_interleave、窗口 mask 不单列。"
                "通用 GEMM 尾项不减。",))
        if vs is not None and vs.ln_mode == "qwen35":
            spec = replace(spec, notes=spec.notes + (
                "Qwen3.5 / Qwen3.5-MoE / Qwen4-Exp 视觉塔：每一层注意力前和 MLP 前各一次 "
                "nn.LayerNorm(hidden)。merger 的 use_postshuffle_norm 是 false："
                "先对 patch 序列做 LayerNorm(hidden)，再 view 到 4*hidden，然后 Linear、"
                "nn.GELU()、Linear。nn.GELU() 默认 approximate='none'，erf 形式。"
                "行数是合并后的位置，宽度是 4*hidden，不是 out_hidden_size。"
                "块内 MLP 的 hidden_act 仍是 gelu_pytorch_tanh，不单列。没有 deepstack。"
                f"GELU 不升精度，a_fmt 是 vision 角色的 {vs.vec_fmt}，"
                "不读 config 的 torch_dtype，也不跟请求里的 GEMM 激活格式走。"
                "LayerNorm 在 fp32 归约。仿射已经在参数里。"
                "位置插值、RoPE 的 cos/sin、整份残差加、eval 的 attention dropout 不单列。"
                "通用 GEMM 尾项不减。",))
        if vs is not None and vs.ln_mode == "ling":
            spec = replace(spec, notes=spec.notes + (
                "Ling-3.0-flash-VL 视觉塔：每一层注意力前和 MLP 前各一次 nn.LayerNorm(hidden)。"
                "merger 只有 norm，没有 fc。use_postshuffle_norm 是 false："
                "先对 patch 序列做 LayerNorm(hidden)，再 view 到 4*hidden。"
                "linear_proj 是 Linear、nn.GELU()、Linear。nn.GELU() 默认 approximate='none'，"
                "erf 形式。行数是合并后的位置，宽度是文本 hidden，不是 4*hidden，也不是 out_hidden_size。"
                "config.json 省略 mlp_depth。BailingMMConfig 的默认是 1，那只会建一级 Linear。"
                "检查点是 linear_proj.0 和 linear_proj.2，对上 mlp_depth 2 的 Sequential。"
                "块内 MLP 的 hidden_act 仍是 gelu_pytorch_tanh，不单列。没有 deepstack。"
                "norm_query_embeds 是 false，F.normalize 不跑。"
                f"GELU 不升精度，a_fmt 是 vision 角色的 {vs.vec_fmt}，"
                "不读 config 的 torch_dtype，也不跟请求里的 GEMM 激活格式走。"
                "LayerNorm 在 fp32 归约。仿射已经在参数里。"
                "位置插值、RoPE 的 cos/sin、整份残差加不单列。"
                "llama.cpp 的 FFN_GELU 是 tanh 近似，不是这次记的 erf-GELU。"
                "通用 GEMM 尾项不减。",))
        if vs is not None and vs.family == "radio":
            spec = replace(spec, notes=spec.notes + (
                "C-RADIO 视觉塔：每一层注意力前和 MLP 前各一次 nn.LayerNorm(hidden)。"
                "MLP 是 ACT2FN['gelu']，也就是 nn.functional.gelu，默认 approximate='none'，"
                "erf 形式。宽度是 intermediate，不是 hidden。"
                "发布的 qkv 仍是一份矩阵，GEMM 不拆成三份。"
                "投影器在像素重排之后是 NemotronH_Omni_RMSNorm，宽度是 4*hidden，"
                "行数是合并后的位置。升到 fp32，pow(2).mean 再一次 rsqrt，乘 weight。"
                "Super-VL 的语言侧有 MTP，所以在去掉 cls / register 之后、像素重排之前"
                "还有一次 LayerNorm(hidden)。Bolt 的 vision_final_layernorm 是 false，没有这一次。"
                f"GELU 不升精度，a_fmt 是 vision 角色里塔权重的 {vs.vec_fmt}。"
                "另外 6 个 fp32 是输入 conditioner 的 buffer，不是第二种权重 dtype。"
                "不跟请求里的 GEMM 激活格式走。"
                "LayerNorm 和 RMSNorm 在 fp32 归约。仿射已经在参数里。"
                "ACT2FN['relu2']、layerscale_value 为 1 且检查点里没有的 LayerScale、"
                "dropout 0、输入 conditioner、位置插值、整份残差加不单列。"
                "10 个 cls / register token 仍不计入塔的行数。通用 GEMM 尾项不减。",))
        if vs is not None and vs.family == "ds":
            spec = replace(spec, notes=spec.notes + (
                "DeepSeek-V4.1 视觉塔：每一层 norm1 和 norm2 是带 weight 的 RMSNorm(hidden)。"
                "升到 fp32，square().mean 再一次 rsqrt，乘 weight，然后回到输入 dtype。"
                "宽度是 hidden，每行一次，不是按头。q 和 k 没有单独的 RMSNorm。"
                "MLP 是 F.silu（x * sigmoid(x)）再乘 up。宽度是 intermediate，不是 2*intermediate。"
                "块之后的最终 RMSNorm 仍在 patch 行上，宽度是 hidden，不是文本 hidden。"
                "aligner 在 w1 之后是 F.gelu，默认 approximate='none'，erf 形式。"
                "行数是合并后的位置（每一边按 downsample 向上取整，含 pad），宽度是文本 hidden。"
                f"SiLU、乘积和 GELU 不升精度，a_fmt 是 vision 角色的 {vs.vec_fmt}，"
                "不读 config 的 torch_dtype，也不跟请求里的 GEMM 激活格式走。"
                "RMSNorm 在 fp32 归约。仿射已经在参数里。eps 不另计。"
                "文本侧的 swiglu_limit 和 rms_norm_eps 视觉塔没有读。"
                "RoPE 的 cos/sin、pad、unfold、整份残差加、"
                "image_start / image_newline / image_end 的下标写入不单列。"
                "通用 GEMM 尾项不减。",))
        if vs is not None and vs.ln_mode == "k25":
            spec = replace(spec, notes=spec.notes + (
                "Kimi MoonViT：每一层注意力前和 MLP 前各一次 nn.LayerNorm(hidden)。"
                "块之后的 final_layernorm 仍在每一帧的 patch 行上，宽度是 hidden，在时间均值之前。"
                "PatchMergerMLP 的 pre_norm 是 LayerNorm(hidden)，在时间均值之后、view 到 4*hidden 之前。"
                "行数是合并位置乘 merge 的平方，静图上等于 patch 数，宽度不是 4*hidden。"
                "nn.GELU() 在第一级 Linear 之后，默认 approximate='none'，erf 形式。"
                "行数是合并后的位置，宽度是 4*hidden，不是文本 hidden。"
                "塔内 MLP 是 PytorchGELUTanh，不单列。"
                f"GELU 不升精度，a_fmt 是 vision 角色的 {vs.vec_fmt}，"
                "不读 config 的 torch_dtype，也不跟请求里的 GEMM 激活格式走。"
                "LayerNorm 在 fp32 归约。仿射已经在参数里。eps 不另计。"
                "RoPE、位置插值、整份残差加、单帧时间均值不单列。"
                "通用 GEMM 尾项不减。",))
        if vs is not None and vs.ln_mode == "k3":
            spec = replace(spec, notes=spec.notes + (
                "Kimi-K3 MoonViT：每一层注意力前和 MLP 前各一次 nn.RMSNorm(hidden)。"
                "升到 fp32，每行一次 rsqrt，不是按头，也不是按 qkv_hidden_size。"
                "块之后的最终 RMSNorm 仍在 patch 行上，宽度是 hidden。"
                "PatchMergerMLPV2 先 view，再 Linear、nn.GELU()、Linear，然后 post RMSNorm。"
                "nn.GELU() 默认 approximate='none'，erf 形式。"
                "行数是合并后的位置，宽度是 4*hidden，不是文本 hidden。"
                "post RMSNorm 的宽度是文本 hidden，行数是合并后的位置。"
                "塔内 MLP 是 PytorchGELUTanh，不单列。"
                f"GELU 不升精度，a_fmt 是 vision 角色的 {vs.vec_fmt}，"
                "不读 config 的 torch_dtype，也不跟请求里的 GEMM 激活格式走。"
                "RMSNorm 在 fp32 归约。仿射已经在参数里。eps 不另计。"
                "RoPE、位置插值、整份残差加不单列。"
                "通用 GEMM 尾项不减。",))
        if vs is not None and vs.ln_mode == "muse":
            spec = replace(spec, notes=spec.notes + (
                "Muse Spark NaViT：ln_pre 在窗口置换之前，对 patch 行做 nn.LayerNorm(hidden)。"
                "每一层注意力前和 MLP 前各一次 nn.LayerNorm(hidden)。"
                "块之后的 ln_post 仍在 patch 行上，在像素重排之前，宽度是 hidden，不是 4*hidden。"
                "MLP 是 ACT2FN['gelu']，也就是 nn.functional.gelu，默认 approximate='none'，"
                "erf 形式。宽度是 intermediate，不是 hidden。"
                "adapter 是 act(fc2(act(fc1(x))))，两级 Linear 之后各一次同样的 erf-GELU。"
                "行数是合并后的位置，宽度是 projector_hidden_size，不是文本 hidden。"
                "发布的 q/k/v 仍是一份融合 GEMM，不拆成三份。"
                f"GELU 不升精度，a_fmt 是 vision 角色的 {vs.vec_fmt}，"
                "不读 config 的 torch_dtype，也不跟请求里的 GEMM 激活格式走。"
                "LayerNorm 在 fp32 归约。仿射已经在参数里。eps 不另计。"
                "GEMM 的 w_fmt 仍是 bf16，通用 GEMM 尾项不减。"
                "RoPE、位置插值、像素重排、整份残差加、"
                "没有 scale 的 perception RMSNorm、以及 vision_projection 不单列。"
                "音频 AutoEncoder：每个 Patchifier 的 Linear 之后一次 RMSNorm，"
                "每个 depthwise conv 之后一次 RMSNorm。"
                "块内两级 Linear 之间是 nn.GELU()，默认 approximate='none'，erf 形式，"
                "宽度是 hidden * mlp_ratio。"
                "级联之后、o_proj 之前一次 RMSNorm，o_proj 之后的 dual_norm 再一次。"
                "音频 GELU 不升精度，a_fmt 跟视觉塔同一份 vision 角色浮点。"
                "音频 RMSNorm 在 fp32 归约。仿射已经在参数里。eps 不另计。"
                "整份残差加、pad 和 patchify view 不单列。"
                "语言侧 tok_embeddings_norm 不是编码器运算。"
                "音频 GEMM 的 w_fmt 仍是 bf16，通用 GEMM 尾项不减。",))
        if vs is not None and vs.family == "fixed" and vs.ln_mode == "step5":
            spec = replace(spec, notes=spec.notes + (
                "Step-5 Perception Encoder：每一层 ln_1 和 ln_2 是 nn.LayerNorm(hidden)。"
                "ln_pre 在块之前。发布的 vision_config 省略 use_ln_pre，Step5VisionConfig 的默认是 True。"
                "use_ln_post 的发布值是 false，ln_post 不建。"
                "hidden_act 是 quick_gelu，不单列，也不新写算子。"
                "LayerScale（ls_init_value 0.1）是 hidden 乘 gamma，不单列。"
                "RoPE、绝对位置加、整份残差加不单列。"
                "两级 stride-2 下采样卷积和投影 Linear 没有激活。"
                "LayerNorm 在 fp32 归约，每行一次 rsqrt，仿射已经在参数里。"
                "通用 GEMM 尾项不减。两份 vision 角色都是单独一份 bf16。",))
        if vs is not None and vs.family == "ernievl":
            spec = replace(spec, notes=spec.notes + (
                "ERNIE-4.5-VL：每一层 norm1 和 norm2 是 nn.LayerNorm(embed_dim)。"
                "块之后的 ln 仍在 patch 行上，宽度是 embed_dim，在 resampler 的 view 之前。"
                "VisionMlp 读 ACT2FN[hidden_act]，发布值是 quick_gelu，不单列。"
                "spatial_linear 和 temporal_linear 都是 Linear、nn.GELU()、Linear、nn.LayerNorm。"
                "nn.GELU() 不传 approximate，默认 approximate='none'，erf 形式。"
                "静图把一帧复制后拼到特征轴上，时间 GEMM 的行数不变、K 加倍。"
                "视频里塔和 spatial 看到每一帧，temporal 和投影看到成对后的一半行。"
                "投影 Linear 之后是 RMSNorm：升到 fp32，pow(2).mean，一次 rsqrt，乘 weight。"
                "宽度是文本 hidden，不是 spatial_dim。"
                f"GELU 不升精度，a_fmt 是 {vs.vec_fmt}。"
                "分桶发布里 bf16 是 resampler，fp16 是塔；Thinking 把整份视觉存成一份 bf16。"
                "不读 config 的 torch_dtype，也不跟请求里的 GEMM 激活格式走。"
                "LayerNorm 和 RMSNorm 在 fp32 归约。仿射已经在参数里。eps 不另计。"
                "拼帧的 index_select、RoPE 的 cos/sin、整份残差加不单列。"
                "塔的 GEMM 标签仍是 bf16。通用 GEMM 尾项不减。",))
        if vs is not None and vs.family == "minimax":
            spec = replace(spec, notes=spec.notes + (
                "MiniMax-M3 视觉塔：pre_layrnorm 和每一层的 layer_norm1、layer_norm2 是 nn.LayerNorm(hidden)。"
                "CLIPMLP 读 ACT2FN[hidden_act]，发布值是 gelu，erf 形式，宽度是 intermediate。"
                "投影器写死 ACT2FN['gelu']：linear_1 之后一次，merge_linear_1 之后再一次。"
                "linear_1 看到 patch 行，K 是视觉 hidden，N 是 projector_hidden_size。"
                "linear_2 仍是 patch 行，K 和 N 是 projector_hidden_size 与文本 hidden。"
                "merge 先把 spatial_merge_size**2 个文本宽度的 patch 拼到特征轴上，"
                "所以 merge_linear_1 的行数是合并后的位置，K 是文本 hidden 乘 merge 的平方。"
                "发布的 q/k/v 仍是一份融合 GEMM，不拆成三份。"
                "patch_embedding 是 Conv3d，无 bias，safetensors 是 fp32，GEMM 的 w_fmt 是 fp32。"
                "其余视觉 GEMM 的 w_fmt 仍是 bf16。"
                f"GELU 不升精度，a_fmt 是 vision 角色里塔权重的 {vs.vec_fmt}，"
                "不读 config 的 torch_dtype，也不跟请求里的 GEMM 激活格式走。"
                "LayerNorm 在 fp32 归约。仿射已经在参数里。eps 不另计。"
                "patch_merge_mlp 的参数在语言侧 other 角色里，不重复计入 vision.params。"
                "文本激活是 swigluoai，不单列。"
                "3D RoPE、整份残差加、图像的 mean/std 归一化不单列。"
                "通用 GEMM 尾项不减。",))
        if vs is not None and vs.family == "minicpm":
            spec = replace(spec, notes=spec.notes + (
                "MiniCPM-V-4.7 视觉塔：每一层注意力前和 MLP 前各一次 nn.LayerNorm(hidden)。"
                "insert_layer_id 这一层仍按完整 patch 行计算，跑完这一层之后才做 vit_merger。"
                "vit_merger 的 layer_norm1 在完整 patch 行上，宽度是 hidden。"
                "四次 GEMM 的顺序不变，全分辨率注意力仍排在它们后面。"
                "pre_norm 在 2×2 展平之后、linear_1 之前，行数是 N/4，宽度是 4*hidden。"
                "塔内 MLP 和 vit_merger 的激活是 gelu_pytorch_tanh，不单列。"
                "块之后的 post_layernorm 在已经降到 N/4 的行上，宽度是 hidden。"
                "最终 merger 的 pre_norm 在 N/16 行上，宽度是 4*hidden。"
                "linear_1 之后是 nn.GELU()，不传 approximate，默认 approximate='none'，erf 形式。"
                "宽度是 4*hidden，不是文本 hidden。linear_2 没有激活。"
                "发布的 q/k/v 仍是一份融合 GEMM，不拆成三份。"
                f"GELU 不升精度，a_fmt 是 vision 角色的 {vs.vec_fmt}，"
                "不读 config 的 torch_dtype，也不跟请求里的 GEMM 激活格式走。"
                "LayerNorm 在 fp32 归约。仿射已经在参数里。eps 不另计。"
                "窗口 gather、均值残差、整份残差加不单列。"
                "GEMM 的 w_fmt 仍是 bf16。通用 GEMM 尾项不减。",))
    return spec


def has_indexer(spec: ModelSpec) -> bool:
    return any(l.core is not None and l.core.idx_heads for l in (*spec.layers, *spec.mtp_layers))


def with_indexer(spec: ModelSpec, mode: str) -> ModelSpec:
    """0.87: sparse-attention indexer deployment over attention TP (Scenario.indexer); "reference" = as released.
    key_split needs a plain token-level indexer (DSA type: not QSA blocks, k-pools, V4.1 dual or V4 compressed keys)."""
    if mode == "reference" or not has_indexer(spec):
        return spec

    def tr(L):
        c = L.core
        if c is None or not c.idx_heads:
            return L
        if mode == "key_split" and (c.qsa or c.idx_kpool or c.dual or c.idx_cmp or c.compress > 1):
            raise ValueError("indexer = key_split 只支持逐 token 的 DSA 型 indexer（DeepSeek-V3.2 / GLM-5 类）；"
                             "该模型的 indexer 打分压缩 / 分块键，请用 replicated 或 tp_split")
        return replace(L, core=replace(c, idx_split=(mode == "tp_split"), idx_kshard=(mode == "key_split")))
    return replace(spec, layers=tuple(tr(l) for l in spec.layers), mtp_layers=tuple(tr(l) for l in spec.mtp_layers))


def with_attn_mode(spec: ModelSpec, mode: str) -> ModelSpec:
    """0.106: ``bidir`` makes every self-attention core non-causal (full S² scores; the LLM prefill path then writes
    no KV cache).  "reference" = as released.  Cross-attention / linear / no-attention layers are left as they are."""
    if mode == "reference":
        return spec

    def tr(L):
        c = L.core
        if c is None or c.kind not in ("gqa", "mla") or not c.causal:
            return L
        return replace(L, core=replace(c, causal=False))
    return replace(spec, layers=tuple(tr(l) for l in spec.layers), mtp_layers=tuple(tr(l) for l in spec.mtp_layers),
                   what_if=True)


def with_hyena_prefill(spec: ModelSpec, mode: str) -> ModelSpec:
    """0.171: StripedHyena 2 prefill implementation (scenario.HYENA_PREFILL).  recurrence: LR layers only (MR / SE
    keep the reference FIR / FFT conv); fused: MR and LR FFT convolutions as one on-chip kernel (what-if)."""
    if mode == "reference":
        return spec
    hy = "recurrence" if mode == "recurrence" else "fused"

    def tr(L):
        c = L.core
        if c is None or c.lin != "hyena" or (c.hy != "lr" and not (hy == "fused" and c.hy == "mr")):
            return L
        return replace(L, core=replace(c, hy_prefill=hy))
    return replace(spec, layers=tuple(tr(l) for l in spec.layers), what_if=hy == "fused")


def with_formats(spec: ModelSpec, overrides: dict[str, str]) -> ModelSpec:
    """What-if: replace role formats (labelled ``what_if``)."""
    from .dtypes import fmt as _fmt
    d = dict(spec.formats)
    for role, name in overrides.items():
        if role in ("act", "kv", "state"):
            continue
        d[role] = RoleFormat(name, _fmt(name).bits)
    return replace(spec, formats=tuple(sorted(d.items())), what_if=True,
                   act_fmt=overrides.get("act", spec.act_fmt), kv_fmt=overrides.get("kv", spec.kv_fmt),
                   state_fmt=overrides.get("state", spec.state_fmt))   # 0.208: recurrent-state format what-if

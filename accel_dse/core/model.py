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
    d_ff: int = 0                # dense intermediate
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

    def expert_params(self, hidden: int) -> int:
        f = self.ffn
        if f.kind != "moe":
            return 0
        return f.n_experts * f.mats * f.expert_in(hidden) * f.d_expert

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
    lookup_params: int = 0                   # n-gram / engram tables (lookup only)
    lookup_bits: float = 0.0                 # stored bits per lookup param incl. scales (0 = embed format)
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
    # 0.258: self-attention cores of io blocks (text refiners): ((name, rows, heads, head_dim, blocks), ...); rows "txt"
    # = the joint text rows of the forward — the refiner GEMMs are io_pre Linears, this adds their attention core
    io_attn: tuple = ()
    final_norm_params: int | None = None     # None → hidden (LLM RMSNorm)
    adaln: bool = False                      # AdaLN modulation (shift/scale/gate) around each norm
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
    tts: object = None                       # 0.269 core.tts.TtsSpec: speech-synthesis decode head (replaces the LM head)
    # 0.270 encoder-decoder (Whisper): encoder positions each decoder sequence cross-attends to (1500 per 30 s window);
    # the cross K/V of every layer is computed at prefill and kept per sequence (fixed size, like recurrent state)
    enc_ctx: int = 0

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
        p = sum(l.params(self.hidden) for l in self.layers) + self.embed_params + self.head_params + self.lookup_params
        p += self.hidden if self.final_norm_params is None else self.final_norm_params
        p += sum(l.params for l in self.io_pre + self.io_post) + self.io_misc_params
        p += self.standby_params + self.cached_params
        if include_mtp:
            p += self.mtp_params()
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
                p -= (f.n_experts - f.top_k) * f.mats * f.expert_in(self.hidden) * f.d_expert
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
    if mt == "lingv3_tts" and isinstance(cfg.get("llm_config"), dict):
        tc = cfg["llm_config"]      # 0.269 Ming-3.0-tts-tiny: the BailingMoeV3 decoder config sits in llm_config
    if mt == "qwen3_asr" and isinstance((cfg.get("thinker_config") or {}).get("text_config"), dict):
        tc = cfg["thinker_config"]["text_config"]      # 0.270 Qwen3-ASR: thinker_config.text_config (Qwen3)
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
             latent: int = 0, shared_gate: bool = False) -> tuple[Ffn, tuple[Linear, ...]]:
    lin = [Linear("router", h, E, "rep")]
    if n_sh and d_sh:
        lin += [Linear("shared_gate_up", h, 2 * n_sh * d_sh if gated else n_sh * d_sh, "col"),
                Linear("shared_down", n_sh * d_sh, h, "row")]
    if shared_gate:
        lin.append(Linear("shared_expert_gate", h, 1, "rep"))
    if latent:
        lin += [Linear("latent_up", h, latent, "rep"), Linear("latent_down", latent, h, "rep")]
    return Ffn("moe", gated=gated, n_experts=E, top_k=k, d_expert=de, n_shared=n_sh, d_shared=d_sh,
               latent=latent), tuple(lin)


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
        if mt == "gpt_oss":
            core = replace(core, sink=True)
        hy_dense = hy3 and i < len(c.get("mlp_layer_types") or ()) and c["mlp_layer_types"][i] == "dense"
        if E and i >= first_dense and i not in mlp_only and not hy_dense:
            ffn, flin = _ffn_moe(h, E, k, de, n_sh, d_sh, shared_gate=bool(c.get("shared_expert_intermediate_size")))
        else:
            ffn, flin = _ffn_dense(h, c["intermediate_size"])
        # 0.291 Mellum (transformers MellumAttention = Qwen3MoeAttention: q_norm / k_norm RMSNorm over head_dim 「来源」)
        misc = 2 * h + (2 * hd if mt.startswith("qwen3") or mt == "mellum" or (hy3 and c.get("qk_norm"))
                        or (k2h and c.get("query_key_norm")) else 0)
        if k2h and ffn.kind == "moe" and c.get("moe_gate_bias"):
            misc += E                                      # router bias (mlp.gate.bias [E])
        if mova and c.get("moe_gate_bias"):
            misc += core.v_experts                         # 0.292 v_router.bias [E_v] (selection only)
        if hy3 and ffn.kind == "moe" and c.get("moe_router_enable_expert_bias"):
            misc += E                                      # fp32 expert_bias (sigmoid router selection bias)
        if c.get("attention_bias") or mt == "qwen2" or mt == "gpt_oss":
            misc += H * hd + 2 * KV * hd + (h if mt == "gpt_oss" else 0)
        if mt == "gpt_oss":
            misc += H  # attention sinks
            misc += E * (2 * de + h)  # expert biases
        layers.append(Layer(alin, core, ffn, flin, misc))
    if mt == "gpt_oss":
        notes.append("attention sink 按参考实现建模（每个 query 行的 softmax 多一个可学习 logit，随后丢弃）；"
                     "128-token 交替滑窗已建模")
    if any(l.core.window for l in layers) and mt != "gpt_oss":
        notes.append(f"滑窗 {window} 已建模")
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
    lc = _list(c.get("linear_attn_config")) or {}
    ling = c.get("model_type") in _LING       # 0.114: Ling-3.x keeps these at the top level
    H, hd = lc.get("num_heads", c["num_attention_heads"]), lc.get("head_dim", (c.get("head_dim") if ling else None) or 128)
    kern = lc.get("short_conv_kernel_size", (c.get("short_conv_kernel_size") if ling else None) or 4)
    d = H * hd
    # output gate: full g_proj (Kimi-K3 release) or low-rank g_a / g_b (transformers Glm5NextTextLinearAttention)
    gate = ((Linear("kda_g_a", h, hd, "rep"), Linear("kda_g_b", hd, d, "col")) if c.get("model_type", "").startswith("glm5_next")
            else (Linear("kda_g", h, d, "col"),))
    # 0.114: Ling-3.x (bailing_hybrid, no_kda_lora): full-rank f_proj h → d instead of the low-rank f_a / f_b
    fgate = ((Linear("kda_f", h, d, "col"),) if c.get("no_kda_lora") else
             (Linear("kda_f_a", h, hd, "rep"), Linear("kda_f_b", hd, d, "col")))
    lin = (Linear("kda_q", h, d, "col"), Linear("kda_k", h, d, "col"), Linear("kda_v", h, d, "col")) + gate + (
        Linear("kda_b", h, H, "col"),) + fgate + (Linear("kda_o", d, h, "row"),)
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
        elif mt in _LING and c.get("layer_group_size"):
            # 0.114 Ling-3.x: groups of layer_group_size, the last of each group MLA, the rest KDA
            n = c["layer_group_size"]
            types = ["full_attention" if (i + 1) % n == 0 else "linear_attention" for i in range(L)]
    E = _g(c, "num_experts", "n_routed_experts", "num_local_experts", default=0)
    k = _g(c, "num_experts_per_tok", "num_experts_per_token")
    de = _g(c, "moe_intermediate_size", "intermediate_size")
    mlp_types = _list(c.get("mlp_layer_types"))
    first_dense = c.get("first_k_dense_replace") or 0
    kda = mt in ("kimi_linear",) + _LING or mt.startswith("glm5_next")
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
        if ffn.kind == "moe" and mt in ("kimi_linear",) + _LING or (ffn.kind == "moe" and c.get("n_routed_experts")):
            misc += E
        hc = c.get("hc_mult") or 0
        if hc and mt.startswith("glm5_next"):
            # transformers Glm5NextTextHyperConnection ×2 per layer (attn_hc, ffn_hc): fn [(2+N)·N, N·h], base, scale
            misc += 2 * ((2 + hc) * hc * (hc * h + 1) + 3)
            layers.append(Layer(alin, core, ffn, flin, misc + 2 * h, hc=hc, hc_iters=c.get("hc_sinkhorn_iters") or 1))
        else:
            layers.append(Layer(alin, core, ffn, flin, misc + 2 * h))
    n_lin = sum(1 for l in layers if l.core.kind == "linear")
    notes.append(f"混合注意力：{n_lin}/{L} 层线性注意力（chunk 形式按参考实现上阵列，递归状态 fp32 按参考），"
                 f"{L - n_lin} 层全注意力")
    n_mtp = c.get("mtp_num_hidden_layers") or c.get("num_nextn_predict_layers") or 0
    mtp = []
    for _ in range(n_mtp):
        alin, core, misc = full_attn(L - 1)
        ffn, flin = ffn_for(L - 1)
        mtp.append(Layer(alin, core, ffn, flin, misc + 4 * h))
    return layers, mtp, notes, cov


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


def _build_lfm2(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """0.114: Liquid LFM2 / LFM2-MoE (transformers Lfm2Moe): ``layer_types`` conv | full_attention.
    conv (Lfm2ShortConv): in_proj h → 3h (B, C, x), B·x, depthwise causal conv (kernel conv_L_cache) over h channels,
    C·(conv), out_proj h → h.  full_attention: GQA with per-head q / k RMSNorm.  FFN: SwiGLU (w1 / w3 / w2); the first
    ``num_dense_layers`` dense (intermediate_size), the rest MoE (num_experts, top-k, moe_intermediate_size, sigmoid
    router + per-expert bias, no shared expert).  Norms: operator_norm + ffn_norm per layer; final embedding_norm."""
    h = c["hidden_size"]
    H = c["num_attention_heads"]
    KV = c.get("num_key_value_heads") or H
    hd = c.get("head_dim") or h // H
    L = c["num_hidden_layers"]
    kern = c.get("conv_L_cache") or 3
    types = _list(c.get("layer_types")) or ["full_attention"] * L
    E = c.get("num_experts") or 0
    n_dense = c.get("num_dense_layers") or 0
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
            ffn, flin = _ffn_dense(h, c["intermediate_size"])
        layers.append(Layer(alin, core, ffn, flin, misc + 2 * h))
    n_conv = sum(1 for t in types[:L] if t == "conv")
    notes = [f"LFM2 混合：{n_conv}/{L} 层门控短卷积（in_proj 3h、深度卷积 kernel {kern}、两次逐元素门控），"
             f"{L - n_conv} 层 GQA 全注意力（q/k RMSNorm）"]
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
    nt = len(c["target_layer_ids"])
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


def _build_nemotron_h(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """0.278 Nemotron-H (transformers NemotronHForCausalLM 「来源」): ``layers_block_type`` gives one mixer per layer —
    mamba (Mamba-2, _mamba2) | attention (GQA, no RoPE in the reference attention path's cost) | moe (NemotronHMoE: sigmoid
    top-k router with correction bias, relu² non-gated routed experts in the moe_latent_size space between
    fc1_latent_proj / fc2_latent_proj, one relu² shared expert at full width) | mlp (relu² up / down); one pre-norm
    each.  MTP (num_nextn_predict_layers, mtp_layers_block_type): eh_proj 2h → h + enorm / hnorm + one block per type."""
    h = c["hidden_size"]
    # 0.289 Nemotron-H Puzzle (Nemotron-3.5-Bolt, configuration_nemotron_h_puzzle.py 「来源」): ``block_configs[i]`` updates
    # the global config for layer i (get_nemotron_h_config_for_layer) — per-layer routed experts, MoE width and top-k
    bcs = list(c.get("block_configs") or ())
    types = [b["block_type"] for b in bcs] if bcs else list(c["layers_block_type"])
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
    notes = [f"Nemotron-H 混合：{n.get('mamba', 0)} 层 Mamba-2（SSD，{c['mamba_num_heads']} 头 × {c['mamba_head_dim']}，"
             f"状态 {c['ssm_state_size']}，B/C 按 {c['n_groups']} 组共享，chunk {c.get('chunk_size') or 128}，状态 fp32）+ "
             f"{n.get('attention', 0)} 层 GQA + {n.get('moe', 0)} 层 latent MoE（{es}，relu² 非门控，"
             f"latent {lat}，共享专家 {c.get('moe_shared_expert_intermediate_size')}）；每层只有一个 mixer"]
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


def _builder(c: dict):
    mt = (c.get("model_type") or "").lower()
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
    if mt == "whisper":
        return _build_whisper, "编码器-解码器（MHA 自注意力 + 交叉注意力）稠密"
    if mt == "stripedhyena2":
        return _build_hyena, "StripedHyena 2（Hyena SE / MR / LR + MHA）"
    if mt == "mimo_v2":
        return _build_mimo_v2, "混合注意力（SWA + full）MoE"
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
    if mt in ("kimi_linear", "bailing_hybrid", "bailing_moe_v3_vl", "ming_v3_asr", "bailing_moe_v3", "qwen3_next", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text",
              "qwen4_exp", "qwen4_exp_text", "minimax_text_01", "minimax_m1", "minimax") or mt.startswith("glm5_next"):
        return _build_hybrid, "混合注意力（linear + full）"
    if c.get("kv_lora_rank"):
        return _build_mla, "MLA" + (" + DSA" if c.get("index_n_heads") else "") + (" MoE" if c.get("n_routed_experts") else "")
    return _build_std, "GQA" + (" MoE" if _g(c, "num_local_experts", "num_experts", "n_routed_experts") else " 稠密")


# ---------------------------------------------------------------- release formats
_ROLE_KEYS = ("attn", "mlp", "expert", "shared_expert", "router", "embed", "lm_head", "mtp", "other")


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


def _w4a4_roles(rel: dict, fm: dict) -> dict:
    """0.254: ModelOpt NVFP4 (hf_quant_config quant_algo "NVFP4"; config_groups input_activations num_bits 4) is W4A4 —
    vLLM ModelOptNvFp4 / SGLang modelopt_fp4 quantise each NVFP4 GEMM's input to NVFP4 (global input_scale, e4m3 per 16)
    「来源」; "W4A16_NVFP4" (Dory) stays weight-only.  The activation format applies only to the roles stored in NVFP4
    (excluded modules — attention, dense layers, router, lm_head — keep bf16 inputs).  Was bf16 for all roles ≤ 0.253."""
    qc = rel.get("quantization_config") or {}
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
    if spec.lookup_params and not any(l.engram_cols or l.ple_rows for l in spec.layers) and mt not in ("whisper", "moshi") \
            and not c.get("cosyvoice") and spec.fish is None:     # 0.274 Moshi: per-frame input tables, a few rows each (exact)
        r.append(f"n-gram / engram 查表（{spec.lookup_params / 1e9:.1f}B 参数）：计存储，每 token 只读少量行")
    if spec.fish is not None:
        if not spec.fish.codec:
            r.append("Drama 3：Stage 1 的 Fish DAC 解码器（codec.pth，torch pickle）未读、未计时")     # 0.276
    gg = sorted({f.fmt for _, f in spec.formats if f.fmt in ("iq2_xxs", "q8_0")})
    if gg:     # 0.289 ggml block formats (Nemotron-3.5-Bolt-24GB): weight-only; the served kernels (vLLM b12x linear / MoE backends) are not modelled per format
        # 0.290: the dequant op count is the ggml reference dequant's (dtypes.DEQUANT_OPS); the served fused kernels
        # (vLLM b12x / llama.cpp MMQ: dot products on the packed values) are not modelled per format
        r.append(f"ggml 块格式（{' / '.join(gg)}）：按仅权重量化计，反量化到 bf16 GEMM（IQ2_XXS 每元素 3.66 次向量操作、"
                 f"Q8_0 2 次，按 ggml 参考反量化「推导」）；服务端融合内核未逐格式建模「近似」")
    if abs(gap) > 0.02:
        r.append(f"参数与发布相差 {gap:+.1%}：模板未复现的部分按发布计存储")
    return tuple(r)


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


def from_release(model_id: str, hf_id: str, rel: dict | None = None, cfg: dict | None = None) -> ModelSpec:
    rel = rel if rel is not None else load_release(hf_id)
    if cfg is None:
        if rel is None:
            raise ValueError(f"no release metadata for {hf_id}; run scripts/fetch_hf_release.py")
        cfg = rel["config"]
    c = _tc(cfg)
    build, arch = _builder(c)
    layers, mtp, notes, cov = build(c)
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
    spec = ModelSpec(
        id=model_id, hf_id=hf_id, arch=arch, hidden=h, vocab=c["vocab_size"], layers=tuple(layers),
        tie_embeddings=tie, mtp_layers=tuple(mtp), mtp_extra_params=2 * h * h if mtp else 0,
        coverage=cov, notes=tuple(notes))
    if (c.get("model_type") or "") == "dory":
        # release default: config n_recurrent_loops, per-loop KV (vLLM DoryConfig default / README serve command)
        spec = with_loops(replace(spec, loop_spec=(c["n_input_layers"], c["n_recurrent_layers"]), loop_phys=spec.layers),
                          c.get("n_recurrent_loops") or 1, "per_loop")
    if rel is not None:
        fm = _release_formats(rel)
        if not any(l.role == "other" for L in spec.layers + spec.mtp_layers for l in L.attn_linears):
            fm.pop("other", None)             # 0.280: only models with role-"other" Linears (Qwen4Exp) carry it
        fm.update(_w4a4_roles(rel, fm))
        spec = replace(spec, formats=tuple(sorted(fm.items())), kv_fmt=_kv_fmt(rel), act_fmt=_act_fmt(rel),
                       release_params=rel.get("params_llm"), release_mtp_params=rel.get("params_mtp"),
                       release_bytes=rel.get("bytes_llm"), quantized_release=_quant_release(rel))
        # lookup-only tables (n-gram / engram embeddings): stored, read a few rows per token
        emb_rel = sum(v["params"] for v in rel.get("roles", {}).get("embed", {}).values())
        cosy = (cfg.get("model_type") or "") in ("cosyvoice2", "cosyvoice3")
        if emb_rel > 1.05 * spec.embed_params and cosy:
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
        elif emb_rel > 1.05 * spec.embed_params:
            spec = replace(spec, lookup_params=emb_rel - spec.embed_params,
                           notes=spec.notes + (f"查表 {(emb_rel - spec.embed_params) / 1e9:.1f}B 参数（n-gram / engram），"
                                               "按发布 safetensors 头计入存储；每 token 只读少量行",))
        # MTP modules: stored size as released (incl. embed/head copies some releases store)
        rmtp = rel.get("params_mtp") or 0
        if rmtp and not spec.mtp_layers:
            # weights ship an MTP module the config does not declare (e.g. Qwen3-Next) → infer one
            last = spec.layers[-1]
            spec = replace(spec, mtp_layers=(replace(last, misc_params=last.misc_params + 2 * h),),
                           notes=spec.notes + ("MTP 模块由发布权重推断（config 中没有 MTP 字段）",))
        if spec.mtp_layers and not rmtp and rel.get("params_mtp") == 0:
            # 0.144: the config declares MTP (mtp_num_hidden_layers) but the release ships no MTP weights
            # (MiniCPM-V-4.7 keeps the Qwen3.5 text config) → no draft module
            spec = replace(spec, mtp_layers=(), mtp_extra_params=0,
                           notes=spec.notes + ("config 声明 MTP，但发布权重中没有 MTP 模块 → 不建模 MTP",))
        head_copy = 0
        if tie and "lm_head" in rel.get("roles", {}) and not rel.get("decision"):   # 0.291: a readout is no tied copy
            head_copy = sum(v["params"] for v in rel["roles"]["lm_head"].values())
        if spec.mtp_layers and rmtp:
            per = spec.mtp_layers[0].params(h)
            n_eff = max(1, min(len(spec.mtp_layers), round(rmtp / max(per, 1))))
            extra = max(0, (rmtp - n_eff * per) // n_eff)
            spec = replace(spec, mtp_layers=spec.mtp_layers[:n_eff], mtp_extra_params=extra)
        if (c.get("model_type") or "").startswith("qwen4_exp") or (c.get("model_type") or "") == "dory":
            spec = replace(spec, final_norm_params=0)       # 0.253 Dory: nGPT, final state justnorm (no weight)
        if (c.get("model_type") or "") == "rwkv7":
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
        if rel.get("dspark") and rel["dspark"].get("kind") == "dflash2":
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
        if spec.release_params:
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
            spec = replace(spec, tts=ts, release_params=spec.release_params - ts.llm_side_params,
                           release_bytes=(spec.release_bytes or 0) - 2 * ts.llm_side_params or None,
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
        from .moshi import moshi_spec
        ms = moshi_spec(cfg, rel)
        if ms is not None:
            spec = replace(spec, moshi=ms, notes=spec.notes + (
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
        vs = vision_spec(hf_id, {**cfg, "_kimi_whisper": rel["kimi_audio"]["whisper"]}
                         if _is_kimi(cfg) and rel.get("kimi_audio") else cfg)     # 0.273: Kimi-Audio whisper config
        rv = (rel.get("params_vision") or 0) if ts is None else 0
        if vs is not None and rv > vs.params:
            # 0.144: multimodal parts stored with the release but not run for images (MiMo audio encoder + speech
            # embeddings) → storage only
            vs = replace(vs, extra=rv - vs.params, params=rv)
        if vs is not None and rel.get("vision_fmt"):
            vs = replace(vs, **_vision_formats(rel, vs))
        from .echo import ECHO_REASON
        spec = replace(spec, coverage_reasons=_coverage_reasons(c, spec, rel_gap) + ((vs.approx,) if vs is not None and vs.approx else ())
                       + ((ECHO_REASON,) if spec.echo is not None else ()),
                       vision=vs, vision_params=vs.params if vs is not None else rv,
                       max_ctx=int(mp) if isinstance(mp, (int, float)) and not isinstance(mp, bool) and mp > 0 else 0)
        if spec.coverage == "full" and spec.coverage_reasons:
            spec = replace(spec, coverage="partial")
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

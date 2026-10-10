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
    lin: str = ""              # linear-attention recurrence: gdn | kda (delta rule, scalar / per-channel gate) | lightning
    lin_chunk: int = 0         # prefill chunk (block) length of the reference chunked kernel
    sink: bool = False         # learned attention sink per head (gpt-oss): one extra softmax logit per query row
    n_state_heads: int = 0
    state_dk: int = 0
    state_dv: int = 0
    conv_channels: int = 0
    conv_kernel: int = 0
    causal: bool = True        # False: bidirectional (DiT / protein encoders)
    cross: bool = False        # cross-attention: keys are the conditioning tokens (``full`` phase ``ctx``)
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
        per = self.idx_dim + 4 * math.ceil(self.idx_dim / 128) if self.idx_fp8 else self.idx_dim * kvb
        return per / self.compress

    def state_elems(self) -> int:
        """Recurrent state per sequence (linear attention) incl. conv state."""
        if self.kind != "linear":
            return 0
        return self.n_state_heads * self.state_dk * self.state_dv + self.conv_channels * max(0, self.conv_kernel - 1)

    def ctx_eff(self, ctx: int) -> int:
        """Keys actually attended per query at context length ``ctx``."""
        c = ctx
        if self.window is not None:
            c = min(c, self.window)
        if self.compress > 1:
            c = math.ceil(ctx / self.compress) + (self.window or 0)
        if self.topk is not None:
            c = min(c, self.topk + (self.window or 0) if self.compress > 1 else self.topk)
        return c

    def keys_sum(self, n: int) -> int:
        """0.61.1: Σ_{p=1..n} keys attended by the query at position p, min(p, ctx_eff(p)) — exact closed form.
        Prefill of a windowed / top-k / compressed layer charges the mean over its positions (the first queries see
        fewer than the cap).  compress > 1: min(p, ⌈p/c⌉ + w, top-k + w)."""
        if n <= 0:
            return 0
        c, w = self.compress, self.window or 0
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
    cross_linears: tuple[Linear, ...] = ()   # cross-attention projections (video DiT: q from tokens, k/v from text)
    cross: AttnCore | None = None
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

    def expert_params(self, hidden: int) -> int:
        f = self.ffn
        if f.kind != "moe":
            return 0
        return f.n_experts * f.mats * f.expert_in(hidden) * f.d_expert

    def params(self, hidden: int) -> int:
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
    kind: str = ""                 # gen | protein
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
    vae_frames: str = "causal"     # latent-frame rule: causal (F−1)/t+1 | chunk17 (Open-Sora 1.2: 17 → 5 per chunk) |
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
    domain: str = "llm"                      # llm | gen | protein
    kv_cache: bool = True                    # autoregressive KV cache (False: one full-sequence forward)
    io_pre: tuple[Linear, ...] = ()          # input-side GEMMs on the first stage (patch / text / timestep embedders)
    io_post: tuple[Linear, ...] = ()         # output-side GEMMs on the last stage (un-patchify head, LM-head dense)
    io_misc_params: int = 0                  # io biases / unused stored tables (counted in storage, no GEMM)
    final_norm_params: int | None = None     # None → hidden (LLM RMSNorm)
    adaln: bool = False                      # AdaLN modulation (shift/scale/gate) around each norm
    workload: NativeWorkload | None = None
    standby_params: int = 0                  # stored but idle per forward (Wan2.2 A14B: the other noise-level expert)
    cached_params: int = 0                   # in the release but not loaded for inference (MiniMax-H3 AdaLN branches:
                                             # modulation outputs precomputed per timestep, README)

    # ----- derived
    @property
    def n_layers(self) -> int:
        return len(self.layers)

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
        return sum(l.params(self.hidden) for l in self.mtp_layers) + self.mtp_extra_params * len(self.mtp_layers)

    def active_params(self) -> int:
        """Params touched per token (routed experts: top_k of n; lookup tables excluded — a few rows/token)."""
        p = self.params() - self.lookup_params - self.standby_params - self.cached_params
        for l in self.layers:
            if l.ffn.kind == "moe":
                f = l.ffn
                p -= (f.n_experts - f.top_k) * f.mats * f.expert_in(self.hidden) * f.d_expert
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
def _tc(cfg: dict) -> dict:
    tc = cfg.get("text_config")
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
        alin, core = _gqa(h, H, KV, hd, window=w)
        if mt == "gpt_oss":
            core = replace(core, sink=True)
        if E and i >= first_dense and i not in mlp_only:
            ffn, flin = _ffn_moe(h, E, k, de, n_sh, d_sh, shared_gate=bool(c.get("shared_expert_intermediate_size")))
        else:
            ffn, flin = _ffn_dense(h, c["intermediate_size"])
        misc = 2 * h + (2 * hd if mt.startswith("qwen3") else 0)
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
    H, hd = lc.get("num_heads", c["num_attention_heads"]), lc.get("head_dim", 128)
    kern = lc.get("short_conv_kernel_size", 4)
    d = H * hd
    lin = (Linear("kda_q", h, d, "col"), Linear("kda_k", h, d, "col"), Linear("kda_v", h, d, "col"),
           Linear("kda_g", h, d, "col"), Linear("kda_b", h, H, "col"), Linear("kda_f_a", h, hd, "rep"),
           Linear("kda_f_b", hd, d, "col"), Linear("kda_o", d, h, "row"))
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
    E = _g(c, "num_experts", "n_routed_experts", "num_local_experts", default=0)
    k = _g(c, "num_experts_per_tok", "num_experts_per_token")
    de = _g(c, "moe_intermediate_size", "intermediate_size")
    mlp_types = _list(c.get("mlp_layer_types"))
    first_dense = c.get("first_k_dense_replace") or 0
    kda = mt in ("kimi_linear",) or mt.startswith("glm5_next")
    is_minimax = mt.startswith("minimax")

    def ffn_for(i: int):
        dense = (mlp_types and mlp_types[i] == "dense") or (i < first_dense) or not E
        if dense:
            return _ffn_dense(h, c["intermediate_size"])
        if mt == "kimi_linear":
            return _ffn_moe(h, E, k, de, c.get("num_shared_experts") or 0, de,
                            latent=c.get("routed_expert_hidden_size") or 0)
        n_sh = c.get("n_shared_experts") or c.get("num_shared_experts") or (1 if c.get("shared_expert_intermediate_size") else 0)
        d_sh = c.get("shared_expert_intermediate_size") or (de if n_sh else 0)
        return _ffn_moe(h, E, k, de, n_sh, d_sh, shared_gate=bool(c.get("shared_expert_intermediate_size")))

    def full_attn(i: int):
        if c.get("kv_lora_rank"):
            alin, core, misc = _mla(c, h)
            ilin, ih, idim, topk = _indexer(c, h, c.get("q_lora_rank"))
            if ih and (not types or types[i] != "full_attention" or mt.startswith("glm5")):
                alin = alin + ilin
                core = replace(core, idx_heads=ih, idx_dim=idim, topk=topk)
            if c.get("mla_use_output_gate"):
                alin = alin + (Linear("o_gate", h, core.n_q * core.v_dim, "col"),)
            return alin, core, misc
        H = c["num_attention_heads"]
        KV = c.get("num_key_value_heads") or H
        hd = c.get("head_dim") or h // H
        alin, core = _gqa(h, H, KV, hd, out_gate=bool(c.get("attn_output_gate")) or is_minimax)
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
        if ffn.kind == "moe" and mt in ("kimi_linear",) or (ffn.kind == "moe" and c.get("n_routed_experts")):
            misc += E
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


def _builder(c: dict):
    mt = (c.get("model_type") or "").lower()
    if mt.startswith("deepseek_v4"):
        return _build_dsv4, "DeepSeek-V4 (CSA/MQA-512)"
    if mt in ("kimi_linear", "qwen3_next", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text",
              "qwen4_exp", "qwen4_exp_text", "minimax_text_01", "minimax_m1", "minimax") or mt.startswith("glm5_next"):
        return _build_hybrid, "混合注意力（linear + full）"
    if c.get("kv_lora_rank"):
        return _build_mla, "MLA" + (" + DSA" if c.get("index_n_heads") else "") + (" MoE" if c.get("n_routed_experts") else "")
    return _build_std, "GQA" + (" MoE" if _g(c, "num_local_experts", "num_experts", "n_routed_experts") else " 稠密")


# ---------------------------------------------------------------- release formats
_ROLE_KEYS = ("attn", "mlp", "expert", "shared_expert", "router", "embed", "lm_head", "mtp")


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
            out[role] = RoleFormat(name if name in ("fp32", "bf16", "fp16", "fp8", "mxfp4", "int8", "int4") else "bf16",
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
    run activations in fp8 (DeepSeek reference inference); all else bf16."""
    qc = rel.get("quantization_config") or {}
    if isinstance(qc, dict) and qc.get("quant_method") == "fp8" and qc.get("activation_scheme") == "dynamic":
        return "fp8"
    return "bf16"


# ---------------------------------------------------------------- public API
def release_path(hf_id: str) -> Path:
    return RELEASES / (hf_id.replace(":", "_").replace("/", "__") + ".json")


@lru_cache(maxsize=None)
def load_release(hf_id: str) -> dict | None:
    p = release_path(hf_id)
    return json.loads(p.read_text()) if p.exists() else None


def _coverage_reasons(c: dict, spec: ModelSpec, gap: float) -> tuple[str, ...]:
    """Concrete per-model list of approximated mechanisms (shown with the coverage label)."""
    mt = c.get("model_type", "")
    L = len(spec.layers)
    r = []
    cmp = sum(1 for l in spec.layers if l.core.compress > 1)
    idx = [l.core for l in spec.layers if l.core.idx_heads]
    if cmp:
        r.append(f"压缩稀疏注意力 CSA（{cmp}/{L} 层）：按有效上下文 ctx/压缩比 + 滑窗近似"
                 + (f"，索引层只读 top-{idx[0].topk} 个 token" if idx else ""))
    elif idx and c.get("index_kpool"):
        r.append(f"DSA 稀疏注意力（{len(idx)}/{L} 层）：lightning indexer 打分按 DeepSeek-V3.2 / GlmMoeDsa 参考逐项建模，"
                 f"但 key 池化（index_kpool={c['index_kpool']}：门控压缩 key 后取 top-{idx[0].topk}/{c['index_kpool']} 个池再展开）"
                 "尚未逐项建模，按未池化的 key 打分（参考 transformers Glm5NextTextIndexer 已公开，可补）")
    if c.get("num_hash_layers"):
        r.append(f"哈希路由（前 {c['num_hash_layers']} 层）：按 top-k MoE 计")
    hc = next((c.get(k) for k in ("hc_mult", "hc_count", "mhc") if c.get(k)), None)
    if hc:
        r.append(f"超连接（多流残差 ×{hc}）：参数计入，多流混合计算未计")
    if spec.lookup_params:
        r.append(f"n-gram / engram 查表（{spec.lookup_params / 1e9:.1f}B 参数）：计存储，每 token 只读少量行")
    if abs(gap) > 0.02:
        r.append(f"参数与发布相差 {gap:+.1%}：模板未复现的部分按发布计存储")
    return tuple(r)


def from_release(model_id: str, hf_id: str, rel: dict | None = None, cfg: dict | None = None) -> ModelSpec:
    rel = rel if rel is not None else load_release(hf_id)
    if cfg is None:
        if rel is None:
            raise ValueError(f"no release metadata for {hf_id}; run scripts/fetch_hf_release.py")
        cfg = rel["config"]
    c = _tc(cfg)
    build, arch = _builder(c)
    layers, mtp, notes, cov = build(c)
    h = c["hidden_size"]
    tie = bool(c.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False)))
    spec = ModelSpec(
        id=model_id, hf_id=hf_id, arch=arch, hidden=h, vocab=c["vocab_size"], layers=tuple(layers),
        tie_embeddings=tie, mtp_layers=tuple(mtp), mtp_extra_params=2 * h * h if mtp else 0,
        coverage=cov, notes=tuple(notes))
    if rel is not None:
        fm = _release_formats(rel)
        spec = replace(spec, formats=tuple(sorted(fm.items())), kv_fmt=_kv_fmt(rel), act_fmt=_act_fmt(rel),
                       release_params=rel.get("params_llm"), release_mtp_params=rel.get("params_mtp"),
                       release_bytes=rel.get("bytes_llm"), quantized_release=_quant_release(rel))
        # lookup-only tables (n-gram / engram embeddings): stored, read a few rows per token
        emb_rel = sum(v["params"] for v in rel.get("roles", {}).get("embed", {}).values())
        if emb_rel > 1.05 * spec.embed_params:
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
        head_copy = 0
        if tie and "lm_head" in rel.get("roles", {}):
            head_copy = sum(v["params"] for v in rel["roles"]["lm_head"].values())
        if spec.mtp_layers and rmtp:
            per = spec.mtp_layers[0].params(h)
            n_eff = max(1, min(len(spec.mtp_layers), round(rmtp / max(per, 1))))
            extra = max(0, (rmtp - n_eff * per) // n_eff)
            spec = replace(spec, mtp_layers=spec.mtp_layers[:n_eff], mtp_extra_params=extra)
        exotic = [k for k in ("hc_mult", "hc_count", "mhc") if c.get(k)]
        if exotic or spec.lookup_params:
            why = []
            if exotic:
                why.append("超连接（多流残差）：参数计入，混合计算未计")
            if spec.lookup_params:
                why.append("n-gram / engram 查表：只计存储")
            spec = replace(spec, coverage="proxy", notes=spec.notes + ("「架构代理」：" + "；".join(why),))
        if head_copy:
            spec = replace(spec, release_params=spec.release_params - head_copy,
                           notes=spec.notes + (f"发布另存了一份 tied lm_head（{head_copy / 1e9:.3f}B），"
                                               "已去重",))
        if spec.release_params:
            gap = spec.release_params - spec.params()
            if abs(gap) / spec.release_params > 0.02:
                # parameters present in the release that the template does not reproduce
                spec = replace(spec, coverage="proxy" if cov != "full" or abs(gap) / spec.release_params > 0.05 else cov,
                               notes=spec.notes + (f"参数与发布相差 {gap / 1e9:+.2f}B "
                                                   f"（{gap / spec.release_params:+.1%}），见 docs/MODEL.md",))
        rel_gap = (spec.release_params - spec.params()) / spec.release_params if spec.release_params else 0.0
        mp = c.get("max_position_embeddings")
        from .vision import vision_spec
        vs = vision_spec(hf_id, cfg)
        spec = replace(spec, coverage_reasons=_coverage_reasons(c, spec, rel_gap),
                       vision=vs, vision_params=vs.params if vs is not None else (rel.get("params_vision") or 0),
                       max_ctx=int(mp) if isinstance(mp, (int, float)) and not isinstance(mp, bool) and mp > 0 else 0)
        if spec.coverage == "full" and spec.coverage_reasons:
            spec = replace(spec, coverage="partial")
    return spec


def with_formats(spec: ModelSpec, overrides: dict[str, str]) -> ModelSpec:
    """What-if: replace role formats (labelled ``what_if``)."""
    from .dtypes import fmt as _fmt
    d = dict(spec.formats)
    for role, name in overrides.items():
        if role in ("act", "kv"):
            continue
        d[role] = RoleFormat(name, _fmt(name).bits)
    return replace(spec, formats=tuple(sorted(d.items())), what_if=True,
                   act_fmt=overrides.get("act", spec.act_fmt), kv_fmt=overrides.get("kv", spec.kv_fmt))

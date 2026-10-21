"""L0 — Scenario: one immutable, fully-specified evaluation point.

Every sweep / search is a sequence of *controlled replacements*
``scn.replace("chip.sram_mib", 128)`` of a base scenario, so identity tests
can assert that replacing a field with its own value changes nothing.
Strict (de)serialisation: unknown keys and non-finite numbers are rejected.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import typing
from dataclasses import dataclass, field, fields, is_dataclass

from .dtypes import FormatSupport
from .d2d_catalog import D2D_DEFAULT_STD, D2D_DEFAULT_UNITS, D2D_STANDARDS, d2d_GBps
from .hardware import CHIPS, D2D_DEFAULT, NET_DEFAULT, Chip, Fabric, Link
from .mapping import ORGS
HYENA_PREFILL = ("reference", "recurrence", "fused")
from .parallel import ATTN_MODES, EXEC_OVERLAPS, INDEXER_MODES, PP_SPLITS, Layout


SERVING_MAX = 1 << 24          # 0.61.1: upper bound of serving.batch / prompt / ctx / out_len / microbatches (16.7 M)


DECISION_MAX_LABELS = 1024
DECISION_MAX_QUESTIONS = 4096   # 0.291: readouts per joint sequence (the card sets no cap; the length limit decides)
THINK_MAX = 4096          # 0.192: StartLux-Decision v1.1 AutoThink card: thought ≤ 4096 tokens 「二手」     # 0.191: option tokens read at the last position (StartLux: ≤ 26 letters per row)


@dataclass(frozen=True)
class Serving:
    phase: str = "decode"          # decode | prefill
    batch: int = 1                 # concurrent sequences per replica
    ctx: int = 4096                # decode: context length per sequence
    prompt: int = 4096             # prefill: prompt length
    out_len: int = 1024            # generated tokens per request (goodput / prefill amortisation)
    microbatches: int = 0          # 0 → auto (= pp, capped by batch)
    spec_k: int = 0                # MTP draft tokens per step (0 = off)
    spec_accept: float = 0.7       # per-draft acceptance (「假设」)
    # 0.222: an independent draft model (catalog id) for spec_k — its own weights + KV on the same cards, k sequential
    # draft decode passes, then the target verifies 1 + k tokens.  "" = MTP / DSpark of the target itself.
    draft_model: str = field(default="", metadata={"omit_default": True})
    # 0.223: spec verify on hybrid / linear layers — "snapshot" (SGLang default intermediate_ssm_state_cache: q states
    # per sequence) | "replay" (SGLang --enable-linear-replayssm-spec: compact ring, replay + fold compute)
    spec_rollback: str = field(default="snapshot", metadata={"omit_default": True})
    # 0.224: radix prefix-cache tracking of hybrid-layer states every N tokens (SGLang --mamba-track-interval, default
    # 256 when tracking runs); 0 = off (whether a deployment tracks is 「假设」)
    mamba_track_interval: int = field(default=0, metadata={"omit_default": True})
    # 0.224: tree drafting (EAGLE-style; SGLang --speculative-eagle-topk / --speculative-num-draft-tokens): spec_k =
    # draft steps (tree depth), spec_topk = branches kept per step (1 = chain), spec_draft_tokens = verified tree
    # tokens incl. the root (0 = 1 + spec_k · spec_topk 「假设」)
    spec_topk: int = field(default=1, metadata={"omit_default": True})
    spec_draft_tokens: int = field(default=0, metadata={"omit_default": True})
    # 0.225: tree per-depth acceptance model — "indep" (0.224 「假设」 i.i.d. siblings, upper bound; +49…+63 % vs the
    # published EAGLE-1 tree τ) | "chain" (lower bound: the tree's top-1 path is the chain; −17…−19 %)
    # 0.226: default chain (was indep); "rank" = Sequoia positional-acceptance vector on an SGLang-style tree (+1…+4 %)
    spec_tree_accept_model: str = field(default="chain", metadata={"omit_default": True})
    # 0.253: draft rounds' sampling — greedy (temperature 0 / top_k 1, default) | sampled (temperature > 0: DSpine adds
    # the eager fp32 serial_rs_table + serial_sample_chain, sglang_dspine worker.py DSpineDraftSampler.sample 「来源」);
    # spec_accept_src = manual (spec_accept, 「假设」 lever, default) | published (the draft's published acceptance length
    # at that temperature → i.i.d. per-draft acceptance 「推导」; drafts without a published τ keep spec_accept)
    spec_sampling: str = field(default="greedy", metadata={"omit_default": True})
    spec_accept_src: str = field(default="manual", metadata={"omit_default": True})
    tpot_slo_ms: float = 50.0
    ttft_slo_ms: float = 2000.0
    moe_skew: float = 1.0          # MoE: busiest EP rank's token-expert pairs / mean (0.49; 1 = uniform) 「假设」
    moe_expert_load: tuple[float, ...] = ()   # MoE: relative tokens per expert (measured) → skew per EP layout
    # 0.293 MoVA (k2_horizon routed value projections): measured relative tokens per *value* expert (the v_router,
    # mova_num_experts entries) → touched value experts per step and their SRAM / SLC read share.  A separate router
    # from the FFN one, so moe_expert_load never applies to it.  () = uniform routing 「假设」
    mova_expert_load: tuple[float, ...] = field(default=(), metadata={"omit_default": True})
    # 0.293: per-layer measured loads — one vector per MoE layer (moe_*, in layer order, each of that layer's expert
    # count: heterogeneous MoE such as Nemotron-H Puzzle) / per MoVA layer (mova_*).  Alternatives to the one-vector
    # forms above; each layer routes, skews and ranks its experts for residency by its own load
    moe_expert_load_layers: tuple[tuple[float, ...], ...] = field(default=(), metadata={"omit_default": True})
    mova_expert_load_layers: tuple[tuple[float, ...], ...] = field(default=(), metadata={"omit_default": True})
    # 0.132: where a scalar moe_skew's extra pairs land on the busiest rank — "popular" (already-read experts, the
    # 0.49 assumption: distinct experts = the uniform expectation) | "uniform" (spread over all the rank's experts:
    # P(token → local expert) = k·skew/E).  Opt-in; ignored with moe_expert_load (exact occupancy from the load).
    moe_skew_spread: str = field(default="popular", metadata={"omit_default": True})
    # 0.133: "expected" = route with the expected per-step MAX over EP ranks of distinct experts read and token-expert
    # pairs under random routing (normal order statistic; MODEL §79).  Opt-in; default "off" (mean-rank routing).
    # 0.141 default "expected" (MODEL §87: validation overall better; ≤ 0.140 "off")
    moe_rank_max: str = field(default="expected", metadata={"omit_default": True})
    # 0.134: EP all-to-all across nodes — "pair" (every token-expert pair crosses the network) | "node" (a token crosses
    # it once per destination node, NVLink forwards: DeepEP normal kernels) | "auto" (node in prefill, pair in decode)
    moe_a2a: str = field(default="pair", metadata={"omit_default": True})
    # 0.134: share of compute units the node-mode all-to-all kernels hold while overlapping prefill compute (DeepEP
    # normal kernels: Buffer.num_sms, 20 of 132 H800 SMs by default → 0.15).  Opt-in; 0 = none.
    moe_a2a_sm: float = field(default=0.0, metadata={"omit_default": True})
    prefix_cached: int = 0         # prefill: prompt tokens already in the KV cache (prefix-cache hit, 0.52) 「假设」
    # 0.62 — VLM images per request (vision encoder at prefill; image tokens join the prompt / KV).  Serialised only
    # when not default (scenario hashes of earlier versions unchanged).  0 = text-only request.
    images: int = field(default=0, metadata={"omit_default": True})
    image_w: int = field(default=1024, metadata={"omit_default": True})     # 「假设」 typical image, px
    image_h: int = field(default=1024, metadata={"omit_default": True})
    # 0.165 — video clips per request (Qwen3VLVideoProcessor models): sampled frames (processor fps 2 → 4…768),
    # source resolution 「假设」 1280×720; video_cap = qwen-vl-utils per-frame pixel cap (transformers opt-in)
    videos: int = field(default=0, metadata={"omit_default": True})
    video_frames: int = field(default=32, metadata={"omit_default": True})   # 「假设」 16 s at the processor's 2 fps
    video_w: int = field(default=1280, metadata={"omit_default": True})
    video_h: int = field(default=720, metadata={"omit_default": True})
    video_cap: bool = field(default=False, metadata={"omit_default": True})
    # 0.175 — seconds of audio per request (models with an audio encoder: Muse Spark, 1920 samples @ 24 kHz per token)
    audio_s: float = field(default=0.0, metadata={"omit_default": True})
    # 0.191 decision models ("System One"): one forward over prompt + options, logits over ≤ labels option tokens at
    # the last position; no decode loop.  0 = off.  Needs phase = prefill.  Metric: decisions / s, latency / decision.
    labels: int = field(default=0, metadata={"omit_default": True})
    # 0.192 AutoThink: a thought decoded before the readout (opt-in).  Default length = StartLux-Decision v1.1
    # AutoThink model card mean at gate 0.9 (596 tokens / question) 「二手」; card maximum 4096.  Needs labels.
    think: bool = field(default=False, metadata={"omit_default": True})
    # 0.291 joint decision requests (pplx-decider v1.2 joint-question-inline-readout-v1, model card): one sequence holds
    # the state and every question; each question has its own readout at a marker → questions readouts per sequence.
    # prompt = the whole joint sequence.  1 = one question per sequence (StartLux / v1 / v1.1).  Needs labels.
    questions: int = field(default=1, metadata={"omit_default": True})
    think_tokens: int = field(default=596, metadata={"omit_default": True})
    # internal: image (and 0.165 video) tokens already added to prompt / ctx by vision.expand_images (never serialised or accepted)
    image_tokens: int = field(default=0, metadata={"internal": True})

    def __post_init__(self):
        if self.phase not in ("decode", "prefill"):
            raise ValueError("serving.phase must be decode|prefill")
        # 0.61.1: booleans rejected (True passed as 1) and an upper bound (2^62 was accepted and evaluated)
        for k in ("batch", "prompt", "out_len"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= SERVING_MAX:
                raise ValueError(f"serving.{k} must be an integer in [1, {SERVING_MAX}]")
        for k in ("ctx", "microbatches", "spec_k"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= SERVING_MAX:
                raise ValueError(f"serving.{k} must be an integer in [0, {SERVING_MAX}]")
        if self.spec_k > 16:     # 0.252: was ≤ 8; DFlash / DSpine block 16 = anchor + 15 drafts
            raise ValueError("serving.spec_k must be ≤ 16")
        if isinstance(self.prefix_cached, bool) or not isinstance(self.prefix_cached, int) \
                or not 0 <= self.prefix_cached < self.prompt:
            raise ValueError("serving.prefix_cached must be an integer in [0, prompt)")
        if not (0.0 <= self.spec_accept <= 1.0):
            raise ValueError("serving.spec_accept must be in [0, 1]")
        if not isinstance(self.draft_model, str) or len(self.draft_model) > 200:
            raise ValueError("serving.draft_model must be a model id string")
        if self.spec_rollback not in ("snapshot", "replay"):
            raise ValueError("serving.spec_rollback must be snapshot|replay")
        if isinstance(self.mamba_track_interval, bool) or not isinstance(self.mamba_track_interval, int) \
                or not 0 <= self.mamba_track_interval <= 1 << 20:
            raise ValueError("serving.mamba_track_interval must be an integer in [0, 1048576]")
        for nm, lo, hi in (("spec_topk", 1, 10), ("spec_draft_tokens", 0, 64)):
            v = getattr(self, nm)
            if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
                raise ValueError(f"serving.{nm} must be an integer in [{lo}, {hi}]")
        if self.spec_sampling not in ("greedy", "sampled"):
            raise ValueError("serving.spec_sampling must be greedy | sampled")
        if self.spec_accept_src not in ("manual", "published"):
            raise ValueError("serving.spec_accept_src must be manual | published")
        if self.spec_tree_accept_model not in ("chain", "rank", "indep"):
            raise ValueError("serving.spec_tree_accept_model must be chain|rank|indep")
        if self.spec_draft_tokens == 1:
            raise ValueError("serving.spec_draft_tokens must be 0 (auto) or ≥ 2")
        for k, lo, hi in (("images", 0, 64), ("image_w", 1, 16384), ("image_h", 1, 16384), ("image_tokens", 0, SERVING_MAX)):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
                raise ValueError(f"serving.{k} must be an integer in [{lo}, {hi}]")
        for k, lo, hi in (("videos", 0, 16), ("video_frames", 2, 768), ("video_w", 1, 16384), ("video_h", 1, 16384)):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
                raise ValueError(f"serving.{k} must be an integer in [{lo}, {hi}]")
        if isinstance(self.audio_s, bool) or not isinstance(self.audio_s, (int, float)) \
                or not (0.0 <= self.audio_s <= 36000.0):
            raise ValueError("serving.audio_s must be in [0, 36000]")
        if isinstance(self.labels, bool) or not isinstance(self.labels, int) or not 0 <= self.labels <= DECISION_MAX_LABELS:
            raise ValueError(f"serving.labels must be an integer in [0, {DECISION_MAX_LABELS}]")
        if self.labels and self.phase != "prefill":
            raise ValueError("serving.labels (decision readout) needs serving.phase = prefill: one forward, no decode")
        if isinstance(self.questions, bool) or not isinstance(self.questions, int) or not 1 <= self.questions <= DECISION_MAX_QUESTIONS:
            raise ValueError(f"serving.questions must be an integer in [1, {DECISION_MAX_QUESTIONS}]")
        if self.questions > 1 and not self.labels:
            raise ValueError("serving.questions (joint decision request: readouts per sequence) needs serving.labels > 0")
        if not isinstance(self.think, bool):
            raise ValueError("serving.think must be a boolean")
        if isinstance(self.think_tokens, bool) or not isinstance(self.think_tokens, int) or not 1 <= self.think_tokens <= THINK_MAX:
            raise ValueError(f"serving.think_tokens must be an integer in [1, {THINK_MAX}]")
        if self.think and not self.labels:
            raise ValueError("serving.think (AutoThink thought before the readout) needs serving.labels > 0")
        if not isinstance(self.video_cap, bool):
            raise ValueError("serving.video_cap must be a boolean")
        if max(self.video_w, self.video_h) > 200 * min(self.video_w, self.video_h):
            raise ValueError("serving.video_w / video_h: aspect ratio must be ≤ 200 (smart_resize limit)")
        if max(self.image_w, self.image_h) > 200 * min(self.image_w, self.image_h):
            raise ValueError("serving.image_w / image_h: aspect ratio must be ≤ 200 (Qwen2-VL smart_resize limit)")
        for k in ("tpot_slo_ms", "ttft_slo_ms"):
            v = getattr(self, k)
            if not (v > 0 and math.isfinite(v)):
                raise ValueError(f"serving.{k} must be finite > 0")
        if isinstance(self.moe_skew, bool) or not isinstance(self.moe_skew, (int, float)) \
                or not (1.0 <= self.moe_skew <= 64.0):
            raise ValueError("serving.moe_skew must be in [1, 64]")
        ld = self.moe_expert_load
        if ld:
            if len(ld) > 4096 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                                     or x < 0 for x in ld) or sum(ld) <= 0:
                raise ValueError("serving.moe_expert_load: ≤ 4096 finite numbers ≥ 0 with a positive sum")
            if self.moe_skew != 1.0:
                raise ValueError("serving.moe_skew and serving.moe_expert_load are alternatives — set one")
        ld = self.mova_expert_load
        if ld and not _load_ok(ld):
            raise ValueError("serving.mova_expert_load: ≤ 4096 finite numbers ≥ 0 with a positive sum")
        for name, flat in (("moe_expert_load_layers", "moe_expert_load"), ("mova_expert_load_layers", "mova_expert_load")):
            lls = getattr(self, name)
            if not lls:
                continue
            if len(lls) > 1024 or not all(_load_ok(x) for x in lls):
                raise ValueError(f"serving.{name}: ≤ 1024 layers, each ≤ 4096 finite numbers ≥ 0 with a positive sum")
            if getattr(self, flat):
                raise ValueError(f"serving.{flat} and serving.{name} are alternatives — set one")
        if self.moe_expert_load_layers and self.moe_skew != 1.0:
            raise ValueError("serving.moe_skew and serving.moe_expert_load_layers are alternatives — set one")
        if self.moe_skew_spread not in ("popular", "uniform"):
            raise ValueError("serving.moe_skew_spread must be popular|uniform")
        if self.moe_rank_max not in ("off", "expected"):
            raise ValueError("serving.moe_rank_max must be off|expected")
        if self.moe_a2a not in ("pair", "node", "auto"):
            raise ValueError("serving.moe_a2a must be pair|node|auto")
        if isinstance(self.moe_a2a_sm, bool) or not isinstance(self.moe_a2a_sm, (int, float)) \
                or not (0.0 <= self.moe_a2a_sm < 0.9):
            raise ValueError("serving.moe_a2a_sm must be in [0, 0.9)")


PLACEMENTS = ("auto", "resident", "shard", "offload", "shard+offload")


@dataclass(frozen=True)
class Workload:
    """Non-autoregressive workloads (video-generation DiT, protein encoders); ignored for LLMs.
    0 = the release's native default (resolution / frames / sequence length from the official config / README)."""
    frames: int = 0                # output video frames
    height: int = 0                # output pixels
    width: int = 0
    steps: int = 0                 # denoise steps (0 → reference sampler default)
    cfg: int = 0                   # forward passes per denoise step: 2 = classifier-free guidance, 1 = off
    seq_len: int = 0               # protein residues
    msa: int = 0                   # structure models: MSA rows (0 → release default cap)
    recycles: int = 0              # structure models: trunk passes incl. the first (0 → release default)
    samples: int = 0               # structure models: diffusion samples per request (0 → release default)
    clip_slo_s: float = 1800.0     # video: per-clip latency SLO (「假设」)
    seq_slo_ms: float = 1000.0     # protein encoders: per-batch latency SLO (「假设」)
    fold_slo_s: float = 120.0      # protein structure models: per-batch latency SLO (「假设」)
    pipeline: bool = True          # video: also evaluate the text encoder(s) + VAE decode (time and storage); False = DiT only
    placement: str = "auto"        # video components (0.45): auto | resident | shard | offload | shard+offload
    host_GBps: float = 50.0        # video offload: host → card bandwidth per card (「假设」 PCIe 5.0 x16 effective)
    vae_tiling: bool = False       # video: diffusers enable_tiling() for the VAEs that offer it (CogVideoX / Mochi / Wan / LTX)
    sample_split: bool = True      # structure models, DAP > 1: diffusion samples split over the DAP ranks (0.46)
    dit_fsdp: bool = False         # video: DiT weights FSDP-sharded over each stage's SP·DP ranks (Wan --dit_fsdp, 0.46)
    te_cpu: bool = False           # video: text encoder runs on the host CPU (Wan --t5_cpu, 0.46)
    host_TFLOPS: float = 2.0       # te_cpu: effective host CPU throughput for the encoder (「假设」 — set from a measurement)
    vae_parallel: bool = False     # video: tiled VAE decode split over the replica's cards (H3 parallel_tiling, 0.47)
    overlap: bool = False          # video: host-CPU encode of the next request overlaps this one's denoise (0.47)
    # 0.116: text-prefix K/V reuse across denoise steps (Qwen-Image-2.1 causal_condition + QwenImage21KVCache):
    # auto = as the model declares (reference use_kv_cache default), on / off = force
    prefix_kv: str = field(default="auto", metadata={"omit_default": True})
    # 0.119: triangle-attention kernel of the structure models — fused (flash-style, default) | materialized (H·N³
    # logits in DRAM) | chunked (tri_chunk rows of logits at a time; 0 → the reference default) | reference (what the
    # release code does at this N)
    tri_attn: str = field(default="fused", metadata={"omit_default": True})
    # 0.255: FLUX 3 static mode-stream reuse (SGLang multimodal_gen flux3_image.py StaticStreams: steps 2…n skip the
    # text / reference mode blocks and modulations, reusing step 1's outputs) — off = reference (BFL / diffusers), on
    mode_cache: str = field(default="off", metadata={"omit_default": True})
    # 0.258: CFG branch shapes — ref = as the release runs them (a model-declared negative branch with its own text
    # length / skipped blocks: two serial transformer calls per step, MOSS-Image), same = both branches at the
    # conditional shape in one packed forward (pre-0.258)
    cfg_branches: str = field(default="ref", metadata={"omit_default": True})
    tri_chunk: int = field(default=0, metadata={"omit_default": True})
    # 0.167 — TeaCache-style step caching (ali-vilab/TeaCache, TeaCache4Wan2.1 teacache_generate.py): on a skipped
    # denoise step the transformer blocks are replaced by "x += previous residual"; embeddings and head still run.
    # step_skip = skipped steps per request (threshold-dependent → user input); step_cache = teacache (first and last
    # step always computed) | teacache_ret (use_ret_steps: first 5 steps computed, no forced last step)
    step_skip: int = field(default=0, metadata={"omit_default": True})
    step_cache: str = field(default="teacache", metadata={"omit_default": True})
    # 0.252 — reference-image editing (FLUX 3 Image Klein image_cond stream): reference images per request (0 = text-to-
    # image) and their size in px (0 = the output size; the release pipeline then fits one reference to ≤ 4 MP, each
    # of ≥ 2 references to ≤ 1 MP — flux3/pipeline.py reference_canvas / prepare_multi_reference)
    ref_images: int = field(default=0, metadata={"omit_default": True})
    ref_height: int = field(default=0, metadata={"omit_default": True})
    ref_width: int = field(default=0, metadata={"omit_default": True})
    # 0.253 — looped LMs (NVIDIA Nemotron-Labs Dory): applications of the shared recurrent block per token
    # (n_recurrent_loops; 0 = the release config, 2) and its KV cache (recurrent_kv_cache_mode: per_loop = one cache per
    # loop, the reference default | last_loop = one cache per recurrent attention layer, each loop overwriting the step's
    # rows).  LLM-only knobs; ignored with a warning elsewhere
    loops: int = field(default=0, metadata={"omit_default": True})
    loop_kv: str = field(default="per_loop", metadata={"omit_default": True})

    def __post_init__(self):
        if not isinstance(self.loops, int) or isinstance(self.loops, bool) or not 0 <= self.loops <= 16:
            raise ValueError("workload.loops must be an integer in [0, 16]")
        if self.loop_kv not in ("per_loop", "last_loop"):
            raise ValueError("workload.loop_kv must be per_loop | last_loop")
        for k, hi in (("ref_images", 64), ("ref_height", 8192), ("ref_width", 8192)):
            v = getattr(self, k)
            if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= hi:
                raise ValueError(f"workload.{k} must be an integer in [0, {hi}]")
        if not isinstance(self.step_skip, int) or isinstance(self.step_skip, bool) or not 0 <= self.step_skip <= 1000:
            raise ValueError("workload.step_skip must be an integer in [0, 1000]")
        if self.step_cache not in ("teacache", "teacache_ret"):
            raise ValueError("workload.step_cache must be teacache | teacache_ret")
        for k in ("frames", "height", "width", "steps", "cfg", "seq_len", "msa", "recycles", "samples"):
            v = getattr(self, k)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(f"workload.{k} must be an integer ≥ 0")
        if self.cfg > 2:
            raise ValueError("workload.cfg must be 0 (default), 1 or 2")
        if self.frames > 1024 or self.height > 4096 or self.width > 4096 or self.steps > 1000 or self.seq_len > 65536:
            raise ValueError("workload: frames ≤ 1024, height/width ≤ 4096, steps ≤ 1000, seq_len ≤ 65536")
        if self.msa > 65536 or self.recycles > 64 or self.samples > 64:
            raise ValueError("workload: msa ≤ 65536, recycles ≤ 64, samples ≤ 64")
        if not isinstance(self.pipeline, bool):
            raise ValueError("workload.pipeline must be true or false")
        if self.tri_attn not in ("fused", "materialized", "chunked", "reference"):
            raise ValueError("workload.tri_attn must be fused | materialized | chunked | reference")
        if not isinstance(self.tri_chunk, int) or isinstance(self.tri_chunk, bool) or not 0 <= self.tri_chunk <= 65536:
            raise ValueError("workload.tri_chunk must be an integer in [0, 65536]")
        if self.prefix_kv not in ("auto", "on", "off"):
            raise ValueError("workload.prefix_kv must be auto | on | off")
        if self.mode_cache not in ("on", "off"):
            raise ValueError("workload.mode_cache must be on | off")
        if self.cfg_branches not in ("ref", "same"):
            raise ValueError("workload.cfg_branches must be ref | same")
        for k in ("vae_tiling", "sample_split", "dit_fsdp", "te_cpu", "vae_parallel", "overlap"):
            if not isinstance(getattr(self, k), bool):
                raise ValueError(f"workload.{k} must be true or false")
        if not (isinstance(self.host_TFLOPS, (int, float)) and not isinstance(self.host_TFLOPS, bool)
                and 0 < self.host_TFLOPS < 1e4):
            raise ValueError("workload.host_TFLOPS must be in (0, 1e4)")
        if self.placement not in PLACEMENTS:
            raise ValueError(f"workload.placement must be one of {', '.join(PLACEMENTS)}")
        if not (isinstance(self.host_GBps, (int, float)) and not isinstance(self.host_GBps, bool)
                and 0 < self.host_GBps < 1e5):
            raise ValueError("workload.host_GBps must be in (0, 1e5)")
        for k in ("clip_slo_s", "seq_slo_ms", "fold_slo_s"):
            v = getattr(self, k)
            if not (v > 0 and math.isfinite(v)):
                raise ValueError(f"workload.{k} must be finite > 0")


def _resolve_chip(v, where: str):
    """0.81: a chip preset name → the preset (the Python constructors accepted a str for ``chip`` and failed later with
    AttributeError, and rejected one for ``pd.prefill_chip`` with a message saying names are allowed); unknown names
    and non-chips → ValueError.  Chip objects pass unchanged (hashes unaffected)."""
    if isinstance(v, str):
        if v not in CHIPS:
            raise ValueError(f"{where}: unknown chip preset {v!r} (one of {', '.join(CHIPS)})")
        return CHIPS[v]
    return v


CHUNK_PROFILE_IDS = ("vllm", "vllm_v0", "sglang", "sglang_mixed", "legacy_512", "fixed")      # 0.231: pdqueue.CHUNK_PROFILES


@dataclass(frozen=True)
class PDConfig:
    """Prefill / decode disaggregation (0.50; core/disagg.py).  Off by default = colocated serving (unchanged).
    The decode pool uses the scenario's ``layout``; the prefill pool its own ``prefill_layout``.  Pool sizes are card
    counts (multiples of each layout's cards; 0 = one replica)."""
    enabled: bool = False
    prefill_layout: Layout = Layout()
    prefill_cards: int = 0
    decode_cards: int = 0
    kv_GBps: float | None = None    # per-card KV transfer bandwidth between the pools; None = network link 「假设」
    kv_layerwise: bool = False      # stream KV layer by layer during prefill (only the last layer's chunk exposed)
    load: float = 0.8               # offered load for the queueing estimate, × the PD fluid capacity (0.51) 「假设」
    rate_rps: float | None = None   # absolute offered load, requests/s (overrides load)
    chunk_tokens: int = 512         # colocated chunked-prefill token budget per iteration (comparison only) 「假设」
    # 0.231: engine profile of the colocated chunked mode (pdqueue.CHUNK_PROFILES).  "vllm" (default, 来源): the
    # iteration token budget max_num_batched_tokens includes the running decodes, filled decode-first → prompt
    # tokens per iteration = budget − running batch; budget by card memory (vLLM arg_utils, API server: < 70 GiB
    # 2048, ≥ 70 GiB 8192, ≥ 160 GiB 16384) unless chunk_budget is set.  "vllm_v0" = the V0 docs default
    # budget 2048 (来源).  "legacy_512" = ≤ 0.230 (512 prompt tokens
    # per iteration on top of the decodes, 「假设」).  "fixed" = chunk_tokens prompt tokens per iteration.
    # 0.232: "sglang" (来源) = SGLang default (no mixed chunk): prefill-first, a long prompt runs as consecutive
    # prefill-only chunks of chunked_prefill_size (by card memory: < 20 GiB 2048, < 60 4096, < 160 8192, else 16384;
    # chunk_budget overrides) while the decodes pause; "sglang_mixed" = --enable-mixed-chunk (the vllm fused rule with
    # budget = chunked_prefill_size).
    chunk_profile: str = field(default="vllm", metadata={"omit_default": True})
    chunk_budget: int | None = field(default=None, metadata={"omit_default": True})
    # 0.52 — request-length spread and prefix caching (queueing + capacity; all 「假设」; defaults = 0.51 behaviour)
    prompt_cv: float = 0.0          # prompt-length coefficient of variation (lognormal, mean = serving.prompt)
    out_cv: float = 0.0             # output-length coefficient of variation (lognormal, mean = serving.out_len)
    length_mix: tuple[tuple[float, int, int], ...] = ()   # discrete (weight, prompt, out_len) mix; overrides the above
    prefix_hit: float = 0.0         # fraction of each prompt already in the prefix cache (skips its prefill)
    prefix_on_decode: bool = True   # the decode pool holds the same prefix → only the uncached KV is transferred
    search_layouts: bool = False    # also search the pools' layouts (not only the card split)
    # 0.53 — heterogeneous pools: the prefill pool may use another chip profile and/or memory (None = same as decode)
    prefill_chip: Chip | None = None
    prefill_mem_id: str | None = None
    # 0.53 — prefix-cache capacity + LRU eviction (Che approximation; all 「假设」).  prefix_len = 0 → off.
    # An explicit prefix_hit > 0 overrides the emergent hit rate.
    prefix_len: int = 0             # shared-prefix length, tokens (system prompt / few-shot / multi-turn history)
    prefix_count: int = 1000        # distinct prefixes in the working set (N)
    prefix_zipf: float = 1.0        # Zipf popularity exponent (0 = uniform)
    prefix_cache_GB: float | None = None   # cache capacity per replica; None = DRAM left after weights + active KV
    prefix_affinity: bool = False   # prefix-aware routing: replicas partition the prefixes (aggregate capacity)
    # 0.58 — radix / partial prefix matching (all 「假设」; () = off): levels root → leaf, each (tokens, branching,
    # Zipf α) — e.g. ((1024, 8, 1.0), (4096, 100, 1.0), (1024, 50, 0.0)) = system prompt → document → conversation.
    # Cache capacity as for prefix_len (pd.prefix_cache_GB or the free DRAM), shared by all levels, in tokens.
    prefix_tree: tuple[tuple[int, int, float], ...] = ()
    search_decode_batch: bool = False   # layout search (0.53): also pick each decode layout's batch (B/2 … 4B, TPOT SLO)
    simulate: bool = False          # 0.54: also run the request-level DES (core/pdsim) and attach per-mode tails 「假设」
    # 0.55 — decode KV capacity (all 「假设」; default off = KV never binds, the 0.54 behaviour).  Admission into the
    # running decode batch: "wait" reserves prompt + output tokens and waits FCFS when they don't fit; "recompute" admits
    # when the current footprint fits and, when growth overflows, preempts the youngest sequence (vLLM recompute: it
    # re-prefills its prompt + generated tokens on that replica later).  Applies to the PD decode pool and to both
    # colocated modes; inactive when the capacity holds the full batch anyway.
    kv_policy: str = "off"
    kv_capacity_GB: float | None = None    # KV capacity per decode replica; None = DRAM left after weights + runtime
    kv_admit: str = "before_prefill"       # 0.56: before_prefill (vLLM: KV allocated, then prefill / KV pull — the
    #                                        admission wait is part of TTFT) | after_prefill (0.55 order, wait apart)
    # 0.111 (opt-in): colocated prefill-first TTFT residual of the in-flight decode iteration — "full" (default:
    # every arrival waits step/2 at the median, a full step at p90 / p99) | "busy" (only when the prefill server is
    # idle and decode holds a request, p = (1 − ρ)(1 − π₀), R = Bern(p)·U(0, step) convolved with the TTFT law)
    ttft_resid: str = field(default="full", metadata={"omit_default": True})
    # 0.122 — multi-turn conversations on the radix tree (all 「假设」; prefix_turns = 1 → off): each leaf path is a
    # conversation of prefix_turns turns, turn t's prompt = turn t−1's prompt + its output (serving.out_len) + a new user
    # message (prefix_turn_user tokens); prefix_writeback inserts the finished sequence (prompt + output) into the tree;
    # prefix_sessions = conversations open at once (sequential turns, temporal locality; 0 = IRM over turns)
    prefix_turns: int = field(default=1, metadata={"omit_default": True})
    prefix_turn_user: int = field(default=0, metadata={"omit_default": True})
    prefix_writeback: bool = field(default=False, metadata={"omit_default": True})
    prefix_sessions: int = field(default=0, metadata={"omit_default": True})
    # 0.123: serving.prompt := the turns' mean prompt (tree + (T − 1)/2 · (output + user)) for the PD report
    prefix_turn_prompt: bool = field(default=False, metadata={"omit_default": True})
    swap_GBps: float | None = None         # 0.56 kv_policy=swap: host link per card; None = workload.host_GBps 「假设」
    # 0.168 — host-memory (L2) tier of the whole-prefix cache (SGLang HiCache: GPU L1 + host L2; write_through backs
    # every device entry up to host = inclusive, write_back demotes on eviction = exclusive).  prefix_host_GB = host
    # cache per replica (0 = off); host hits are loaded back over the host link (swap_GBps / workload.host_GBps per card)
    # layer by layer, overlapped with the prefill compute
    prefix_host_GB: float = field(default=0.0, metadata={"omit_default": True})
    prefix_host_policy: str = field(default="write_through", metadata={"omit_default": True})

    def __post_init__(self):
        for k in ("prefill_cards", "decode_cards"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 65536:
                raise ValueError(f"pd.{k} must be an integer in [0, 65536]")
        if self.prefill_cards and self.prefill_cards % self.prefill_layout.cards:
            raise ValueError(f"pd.prefill_cards must be a multiple of the prefill layout's {self.prefill_layout.cards} cards")
        if self.kv_GBps is not None and (isinstance(self.kv_GBps, bool) or not isinstance(self.kv_GBps, (int, float))
                                         or not 0 < self.kv_GBps < 1e7):
            raise ValueError("pd.kv_GBps must be > 0 or null")
        if isinstance(self.load, bool) or not isinstance(self.load, (int, float)) or not 0 < self.load < 1:
            raise ValueError("pd.load must be in (0, 1)")
        if self.rate_rps is not None and (isinstance(self.rate_rps, bool) or not isinstance(self.rate_rps, (int, float))
                                          or not 0 < self.rate_rps < 1e7):
            raise ValueError("pd.rate_rps must be > 0 or null")
        if isinstance(self.chunk_tokens, bool) or not isinstance(self.chunk_tokens, int) \
                or not 16 <= self.chunk_tokens <= 1 << 20:
            raise ValueError("pd.chunk_tokens must be an integer in [16, 1048576]")
        if self.chunk_profile not in CHUNK_PROFILE_IDS:
            raise ValueError("pd.chunk_profile must be one of " + " | ".join(CHUNK_PROFILE_IDS))
        if self.chunk_budget is not None and (isinstance(self.chunk_budget, bool) or not isinstance(self.chunk_budget, int)
                                              or not 16 <= self.chunk_budget <= 1 << 20):
            raise ValueError("pd.chunk_budget must be an integer in [16, 1048576] or null")
        if not isinstance(self.enabled, bool) or not isinstance(self.kv_layerwise, bool):
            raise ValueError("pd.enabled / pd.kv_layerwise must be booleans")
        for k in ("prompt_cv", "out_cv"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 4:
                raise ValueError(f"pd.{k} must be in [0, 4]")
        if isinstance(self.prefix_hit, bool) or not isinstance(self.prefix_hit, (int, float)) \
                or not 0 <= self.prefix_hit <= 0.99:
            raise ValueError("pd.prefix_hit must be in [0, 0.99]")
        if not isinstance(self.prefix_on_decode, bool) or not isinstance(self.search_layouts, bool):
            raise ValueError("pd.prefix_on_decode / pd.search_layouts must be booleans")
        mix = self.length_mix
        if not isinstance(mix, tuple) or len(mix) > 16:
            raise ValueError("pd.length_mix: at most 16 [weight, prompt, out_len] rows")
        for row in mix:
            if not isinstance(row, tuple) or len(row) != 3:
                raise ValueError("pd.length_mix rows must be [weight, prompt, out_len]")
            w, sp, so = row
            if isinstance(w, bool) or not isinstance(w, (int, float)) or not 0 < w < 1e9 or not math.isfinite(w):
                raise ValueError("pd.length_mix: weight must be a finite number > 0")
            for v in (sp, so):
                if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 1 << 21:
                    raise ValueError("pd.length_mix: prompt / out_len must be integers in [1, 2097152]")
        if mix and (self.prompt_cv or self.out_cv):
            raise ValueError("pd.length_mix and pd.prompt_cv / pd.out_cv are alternatives — set one")
        object.__setattr__(self, "prefill_chip", _resolve_chip(self.prefill_chip, "pd.prefill_chip"))
        if self.prefill_chip is not None and not isinstance(self.prefill_chip, Chip):
            raise ValueError("pd.prefill_chip must be a chip preset name, a chip object or null")
        if self.prefill_mem_id is not None and (not isinstance(self.prefill_mem_id, str) or not self.prefill_mem_id
                                                or len(self.prefill_mem_id) > 64):
            raise ValueError("pd.prefill_mem_id must be a memory id or null")
        if isinstance(self.prefix_len, bool) or not isinstance(self.prefix_len, int) or not 0 <= self.prefix_len <= 1 << 21:
            raise ValueError("pd.prefix_len must be an integer in [0, 2097152]")
        if isinstance(self.prefix_host_GB, bool) or not isinstance(self.prefix_host_GB, (int, float)) \
                or not 0 <= self.prefix_host_GB < 1e6:
            raise ValueError("pd.prefix_host_GB must be in [0, 1e6)")
        if self.prefix_host_policy not in ("write_through", "write_back"):
            raise ValueError("pd.prefix_host_policy must be write_through | write_back")
        if isinstance(self.prefix_count, bool) or not isinstance(self.prefix_count, int) \
                or not 1 <= self.prefix_count <= 10 ** 9:
            raise ValueError("pd.prefix_count must be an integer in [1, 1e9]")
        if isinstance(self.prefix_zipf, bool) or not isinstance(self.prefix_zipf, (int, float)) \
                or not 0 <= self.prefix_zipf <= 3:
            raise ValueError("pd.prefix_zipf must be in [0, 3]")
        if self.prefix_cache_GB is not None and (isinstance(self.prefix_cache_GB, bool)
                                                 or not isinstance(self.prefix_cache_GB, (int, float))
                                                 or not 0 <= self.prefix_cache_GB < 1e7):
            raise ValueError("pd.prefix_cache_GB must be ≥ 0 or null")
        if not isinstance(self.prefix_affinity, bool) or not isinstance(self.search_decode_batch, bool) \
                or not isinstance(self.simulate, bool):
            raise ValueError("pd.prefix_affinity / pd.search_decode_batch / pd.simulate must be booleans")
        tree = self.prefix_tree
        if tree:
            if not isinstance(tree, tuple) or len(tree) > 6:
                raise ValueError("pd.prefix_tree: at most 6 [tokens, branching, zipf] levels")
            nodes = 1.0
            for row in tree:
                if not isinstance(row, tuple) or len(row) != 3:
                    raise ValueError("pd.prefix_tree rows must be [tokens, branching, zipf]")
                L, n, a = row
                if isinstance(L, bool) or not isinstance(L, int) or not 1 <= L <= 1 << 21:
                    raise ValueError("pd.prefix_tree: tokens must be an integer in [1, 2097152]")
                if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 10 ** 6:
                    raise ValueError("pd.prefix_tree: branching must be an integer in [1, 1e6]")
                if isinstance(a, bool) or not isinstance(a, (int, float)) or not 0 <= a <= 3:
                    raise ValueError("pd.prefix_tree: zipf must be in [0, 3]")
                nodes *= n
            if nodes > 1e9:
                raise ValueError("pd.prefix_tree: at most 1e9 leaf paths")
            if self.prefix_len or self.prefix_hit:
                raise ValueError("pd.prefix_tree and pd.prefix_len / pd.prefix_hit are alternatives — set one")
        if self.kv_policy not in ("off", "wait", "recompute", "swap"):
            raise ValueError("pd.kv_policy must be off | wait | recompute | swap")
        for k, lo, hi in (("prefix_turns", 1, 64), ("prefix_turn_user", 0, 1 << 20), ("prefix_sessions", 0, 10 ** 7)):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
                raise ValueError(f"pd.{k} must be an integer in [{lo}, {hi}]")
        if not isinstance(self.prefix_writeback, bool) or not isinstance(self.prefix_turn_prompt, bool):
            raise ValueError("pd.prefix_writeback / pd.prefix_turn_prompt must be booleans")
        if self.prefix_turn_prompt and self.prefix_turns < 2:
            raise ValueError("pd.prefix_turn_prompt needs pd.prefix_turns ≥ 2")
        if (self.prefix_turns > 1 or self.prefix_writeback or self.prefix_sessions or self.prefix_turn_user) \
                and not self.prefix_tree:
            raise ValueError("pd.prefix_turns / prefix_turn_user / prefix_writeback / prefix_sessions need pd.prefix_tree")
        if self.ttft_resid not in ("full", "busy"):
            raise ValueError("pd.ttft_resid must be full | busy")
        if self.kv_admit not in ("before_prefill", "after_prefill"):
            raise ValueError("pd.kv_admit must be before_prefill | after_prefill")
        if self.swap_GBps is not None and (isinstance(self.swap_GBps, bool)
                                           or not isinstance(self.swap_GBps, (int, float))
                                           or not 0 < self.swap_GBps < 1e5):
            raise ValueError("pd.swap_GBps must be in (0, 1e5) or null")
        if self.kv_capacity_GB is not None and (isinstance(self.kv_capacity_GB, bool)
                                                or not isinstance(self.kv_capacity_GB, (int, float))
                                                or not 0 < self.kv_capacity_GB < 1e7):
            raise ValueError("pd.kv_capacity_GB must be > 0 or null")


CLUSTER_UNLIMITED_GBps = 1e9      # 0.309: on-die cluster tier with an unlimited NoC (no link width) — not a bound


@dataclass(frozen=True)
class Scenario:
    model: str = "qwen3-8b"
    chip: Chip = CHIPS["100T"]
    mem_id: str | None = None      # 0.205: None = the chip preset's default memory (hardware.PRESET_MEM, else lpddr5x_4x64_8533_16g)
    mem_eff: float | None = None
    # 0.108 (opt-in): DRAM efficiency of the KV-cache read / write streams (long sequential page reads of the decode
    # attention kernels); None = mem_eff for every byte (bit-identical)
    kv_mem_eff: float | None = field(default=None, metadata={"omit_default": True})
    # 0.94 (optional): usable DRAM capacity per card in GB (1e9; 0 = the memory config's stack capacity) and the share
    # of post-weight free memory the runtime grants the KV / state cache (TRT-LLM free_gpu_memory_fraction)
    mem_cap_GB: float = field(default=0.0, metadata={"omit_default": True})
    kv_mem_frac: float = field(default=1.0, metadata={"omit_default": True})
    link: Link = Link()                                  # in-node scale-up between packages (0.50 three tiers)
    mapping: str = "os"
    layout: Layout = Layout()
    serving: Serving = Serving()
    formats_override: tuple[tuple[str, str], ...] = ()   # what-if (labelled)
    workload: Workload = Workload()                      # video / protein models only
    d2d: Link = D2D_DEFAULT                              # die-to-die tier inside a package (0.48) 「假设」
    package_cards: int = 1                               # dies per package (0.48); inert while d2d_enabled is off
    d2d_enabled: bool = False                            # chiplet stacking with a D2D tier; off = monolithic die (0.50)
    d2d_std: str = D2D_DEFAULT_STD                       # D2D grade (core/d2d_catalog.py) or "custom" (= d2d.GBps)
    d2d_units: int = D2D_DEFAULT_UNITS                   # D2D units (UCIe modules / BoW slices / links) per die 「假设」
    net: Link = NET_DEFAULT                              # cross-node scale-out (IB / RoCEv2 class) 「假设」
    node_cards: int = 0                                  # cards per node; 0 = one node (cross-node tier unused)
    pd: PDConfig = PDConfig()                            # prefill / decode disaggregation (0.50; off = colocated)
    fabric: Fabric = Fabric()                            # topology-aware collectives (0.59; off = 0.50 α-β tiers) 「假设」
    pp_split: str = field(default="cost", metadata={"omit_default": True})   # 0.62: PP stages balanced by cost | "layers"
    # 0.66: compute / memory / link overlap inside a stage step — "stage" (one roofline over the whole stage, the
    # pre-0.66 model) | "class" (GEMM ‖ weight traffic, attention ‖ KV traffic, the two classes and the collectives
    # serial: kernel-by-kernel execution, e.g. GPUs) | "kernel" (class + vector kernels serial) | "serial" (no
    # overlap, upper bound) — see schedule.StageTime.total 「假设」
    exec_overlap: str = field(default="pipe", metadata={"omit_default": True})   # 0.100: pipe (was stage)
    # 0.69: fixed per-layer time per step (kernel launch / sync / scheduler), µs; added to the stage's exposed sync
    # time in every overlap mode.  NPU- and runtime-dependent 「假设」; default 0 (off), not fitted.
    layer_overhead_us: float = field(default=0.0, metadata={"omit_default": True})
    # 0.164 (opt-in, default off = bit-identical): host / serving-runtime time per engine step — scheduling, input
    # preparation, output processing (vLLM v0.6 blog: 29 % of time on Llama-3-8B/H100 in v0.5.3; WukLab 2024: grows
    # with the requests in the batch).  host = host_step_us + host_seq_us × sequences per DP rank; decode amortises it
    # over ``host_steps`` (multi-step scheduling); "serial" adds it to the device step, "async" overlaps it with the
    # previous step (step = max(device, host)).  No default value: runtime-specific, user input only.
    host_step_us: float = field(default=0.0, metadata={"omit_default": True})
    host_seq_us: float = field(default=0.0, metadata={"omit_default": True})
    host_overlap: str = field(default="serial", metadata={"omit_default": True})
    host_steps: int = field(default=1, metadata={"omit_default": True})
    # 0.87: sparse-attention indexer deployment over attention TP (parallel.INDEXER_MODES); default = as released
    indexer: str = field(default="reference", metadata={"omit_default": True})
    # 0.106: attention mask what-if (parallel.ATTN_MODES); default = as released
    attn_mode: str = field(default="reference", metadata={"omit_default": True})
    # 0.171: StripedHyena 2 long-filter prefill (HYENA_PREFILL): reference = as released (vortex prefill_style fft:
    # FFT conv + modal-FFT state) | recurrence (vortex prefill_via_direct_recurrence) | fused (what-if: one on-chip
    # FFT-conv kernel per operator, state by direct sum)
    hyena_prefill: str = field(default="reference", metadata={"omit_default": True})

    def __post_init__(self):
        object.__setattr__(self, "chip", _resolve_chip(self.chip, "chip"))
        if not isinstance(self.chip, Chip):
            raise ValueError("chip must be a chip preset name or a chip object")
        if self.mem_id is None:
            from .hardware import default_mem
            object.__setattr__(self, "mem_id", default_mem(self.chip))
        if self.mapping not in ORGS:
            raise ValueError(f"mapping must be one of {ORGS}")
        if self.pp_split not in PP_SPLITS:
            raise ValueError(f"pp_split must be one of {PP_SPLITS}")
        if isinstance(self.layer_overhead_us, bool) or not isinstance(self.layer_overhead_us, (int, float)) \
                or not 0.0 <= self.layer_overhead_us <= 1e4:
            raise ValueError("layer_overhead_us must be in [0, 10000]")
        for nm in ("host_step_us", "host_seq_us"):
            v = getattr(self, nm)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= v <= 1e6:
                raise ValueError(f"{nm} must be in [0, 1e6] µs")
        if self.host_overlap not in ("serial", "async"):
            raise ValueError("host_overlap must be 'serial' or 'async'")
        if isinstance(self.host_steps, bool) or not isinstance(self.host_steps, int) or not 1 <= self.host_steps <= 64:
            raise ValueError("host_steps must be an integer in [1, 64]")
        if self.exec_overlap not in EXEC_OVERLAPS:
            raise ValueError(f"exec_overlap must be one of {EXEC_OVERLAPS}")
        if self.indexer not in INDEXER_MODES:
            raise ValueError(f"indexer must be one of {INDEXER_MODES}")
        if self.attn_mode not in ATTN_MODES:
            raise ValueError(f"attn_mode must be one of {ATTN_MODES}")
        if self.hyena_prefill not in HYENA_PREFILL:
            raise ValueError(f"hyena_prefill must be one of {HYENA_PREFILL}")
        if isinstance(self.package_cards, bool) or not isinstance(self.package_cards, int) \
                or not 1 <= self.package_cards <= 1024:
            raise ValueError("package_cards must be an integer in [1, 1024]")
        if not isinstance(self.d2d_enabled, bool):
            raise ValueError("d2d_enabled must be a boolean")
        if self.d2d_std != "custom" and self.d2d_std not in D2D_STANDARDS:
            raise ValueError(f"d2d_std must be 'custom' or one of {sorted(D2D_STANDARDS)}")
        for k, lo, hi in (("d2d_units", 1, 64), ("node_cards", 0, 4096)):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
                raise ValueError(f"{k} must be an integer in [{lo}, {hi}]")
        kc = self.chip.noc_clusters
        if kc > 1 and not (self.d2d_enabled and self.package_cards == kc):
            raise ValueError(f"chip.noc_clusters = {kc}（片上 NoC 簇）需要 d2d_enabled 且 package_cards = {kc}："
                             "簇以封装内的卡建模，簇间层即 D2D 层的片上版本（0.309）")
        if self.node_cards and self.d2d_enabled and self.node_cards % self.package_cards:
            raise ValueError("node_cards must be a multiple of package_cards")
        if self.pd.decode_cards and self.pd.decode_cards % self.layout.cards:
            raise ValueError(f"pd.decode_cards must be a multiple of the (decode) layout's {self.layout.cards} cards")
        if self.mem_eff is not None and not (0 < self.mem_eff <= 1):
            raise ValueError("mem_eff must be in (0,1]")
        if self.kv_mem_eff is not None and (isinstance(self.kv_mem_eff, bool) or not (0 < self.kv_mem_eff <= 1)):
            raise ValueError("kv_mem_eff must be in (0,1]")
        if not (self.mem_cap_GB == 0 or (math.isfinite(self.mem_cap_GB) and self.mem_cap_GB > 0)):
            raise ValueError("mem_cap_GB must be 0 (stack capacity) or finite > 0")
        if not (0 < self.kv_mem_frac <= 1):
            raise ValueError("kv_mem_frac must be in (0,1]")
        # 0.310: auto memory-attach edges (Chip.noc_mem_edges = 0) — resolved here so every path (single eval,
        # energy DSE, insight panels, PD pools) uses the shoreline occupancy of this chip + memory
        ch = self.chip
        if ch.noc_mem_edges == 0 and ch.noc_mem_attach != "none":
            from .budget import mem_edges_auto
            m = mem_edges_auto(self)
            if m != ch.noc_mem_edges_auto:
                object.__setattr__(self, "chip", dataclasses.replace(ch, noc_mem_edges_auto=m))

    @property
    def d2d_link(self) -> Link:
        """Effective D2D link: the catalog grade × units (raw, per direction), or ``d2d`` when ``d2d_std`` = custom."""
        if self.chip.noc_clusters > 1:
            # 0.309: on-die cluster tier — the NoC's own boundary links (no PHY): attach links × width × clock on the
            # cluster's shared edge(s), unbounded while the NoC is (no link width); α = the D2D tier's (collective
            # sync dominates; PHY latency < 2 ns either way) 「假设」
            from .d2d_catalog import d2d_edges
            bw = self.chip.noc_d2d_GBps(d2d_edges(self.package_cards))
            return Link(CLUSTER_UNLIMITED_GBps if math.isinf(bw) else bw, self.d2d.alpha_us, self.d2d.topology)
        lk = self.d2d if self.d2d_std == "custom" else Link(d2d_GBps(self.d2d_std, self.d2d_units), self.d2d.alpha_us,
                                                             self.d2d.topology)
        # 0.308: D2D bytes enter / leave the die's NoC at the D2D edge(s): ≤ attach links × link width × clock
        # (Chip.noc_d2d_GBps; inf unless a NoC link width is set with edge attach) — the 0.305 DRAM-ingress rule
        if self.d2d_enabled and self.package_cards > 1:
            from .d2d_catalog import d2d_edges
            cap = self.chip.noc_d2d_GBps(d2d_edges(self.package_cards))
            if cap < lk.GBps:
                lk = dataclasses.replace(lk, GBps=cap)
        return lk

    @property
    def package_eff(self) -> int:
        return self.package_cards if self.d2d_enabled else 1

    # ---- controlled replacement
    def replace(self, path: str, value) -> "Scenario":
        parts = path.split(".")
        return _replace_path(self, parts, value)

    def colocated(self) -> "Scenario":
        """The scenario without its PD block (0.56): colocated searches (layouts, stability, fit) swap the layout,
        and pd.decode_cards is validated against the layout — irrelevant to a colocated evaluation."""
        return self if self.pd == PDConfig() else dataclasses.replace(self, pd=PDConfig())

    def to_dict(self, full: bool = False) -> dict:
        """Plain dict (hash input).  ``full``: also the fields left at an omit-when-default value (0.62)."""
        return _to_plain(self, full)

    @staticmethod
    def from_dict(d: dict) -> "Scenario":
        return _from_plain(Scenario, upgrade_legacy(d))

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]


def upgrade_legacy(d: dict) -> dict:
    """Pre-0.50 scenario JSON → 0.50 (the user's partial dict, before merging with defaults):
    ``package_cards`` > 1 without ``d2d_enabled`` meant chiplets with a D2D tier → d2d_enabled = true; a ``d2d.GBps``
    without ``d2d_std`` is a custom D2D figure → d2d_std = "custom" (so 0.48 / 0.49 scenarios evaluate as before)."""
    if not isinstance(d, dict):
        return d
    out = dict(d)
    pc = out.get("package_cards")
    if "d2d_enabled" not in out and isinstance(pc, int) and not isinstance(pc, bool) and pc > 1:
        out["d2d_enabled"] = True
    if "d2d_std" not in out and isinstance(out.get("d2d"), dict) and "GBps" in out["d2d"]:
        out["d2d_std"] = "custom"
    for key in ("chip", ):
        if isinstance(out.get(key), dict):
            out[key] = upgrade_chip_0141(out[key])
    pd = out.get("pd")
    # 0.231: an explicit non-default chunk_tokens without a chunk_profile meant a fixed per-iteration chunk (≤ 0.230)
    if isinstance(pd, dict) and "chunk_profile" not in pd and pd.get("chunk_tokens") not in (None, 512):
        pd = out["pd"] = {**pd, "chunk_profile": "fixed"}
    if isinstance(pd, dict) and isinstance(pd.get("prefill_chip"), dict):
        out["pd"] = {**pd, "prefill_chip": upgrade_chip_0141(pd["prefill_chip"])}
    return out


def upgrade_chip_0141(c: dict) -> dict:
    """0.141: a ≤ 0.140 saved chip (its full dict carries "acc_kib" and "sram_port_Bpc", both always written then)
    used chip-total accumulator / port: migrate both to per-engine values that give the same totals at the saved
    engine count (the old port default 4·(R + C·E)·2 included), so saved scenarios evaluate as before and scale with
    engines from now on."""
    # ≤ 0.140 dicts: numeric acc_kib + an sram_port_Bpc key, never acc_kib_per_engine (0.141 full dicts carry it;
    # 0.141 partial dicts omit sram_port_Bpc unless set)
    if c.get("acc_kib") is None or "sram_port_Bpc" not in c or "acc_kib_per_engine" in c:
        return c
    out = dict(c)
    try:
        E = int(out.get("engines", 16)); R = int(out.get("rows", 56)); C = int(out.get("cols", 56))
        acc = out.pop("acc_kib")
        if acc is not None:
            out["acc_kib_per_engine"] = float(acc) / E
        port = out.get("sram_port_Bpc")
        if not out.get("sram_port_Bpc_per_engine"):
            out.pop("sram_port_Bpc", None)
            out["sram_port_Bpc_per_engine"] = (float(port) if port else 4.0 * (R + C * E) * 2.0) / E
    except (TypeError, ValueError, ZeroDivisionError):
        return c                          # malformed: let validation report it
    return out


def _replace_path(obj, parts, value):
    if not parts:
        return value
    name = parts[0]
    if not is_dataclass(obj) or name not in {f.name for f in fields(obj)}:
        raise KeyError(f"unknown scenario field {name!r}")
    cur = getattr(obj, name)
    new = _replace_path(cur, parts[1:], value) if len(parts) > 1 else value
    return dataclasses.replace(obj, **{name: new})


def _num_canon(f, v):
    """int given for a float-typed field → float, so equal scenarios serialise / hash identically."""
    if isinstance(v, int) and not isinstance(v, bool) and "float" in str(f.type):
        return float(v)
    return v


def _serialised(o, full: bool = False) -> list:
    """Fields written by to_dict: internal ones never; ``omit_default`` ones only when not at their default (0.62)
    unless ``full`` (the API's default template lists every accepted key)."""
    out = []
    for f in fields(o):
        if f.metadata.get("internal"):
            continue
        if not full and f.metadata.get("omit_default") and getattr(o, f.name) == f.default:
            continue
        out.append(f)
    return out


def _to_plain(o, full: bool = False):
    if is_dataclass(o):
        return {f.name: _to_plain(_num_canon(f, getattr(o, f.name)), full) for f in _serialised(o, full)}
    if isinstance(o, tuple):
        return [_to_plain(x, full) for x in o]
    if isinstance(o, float) and not math.isfinite(o):
        raise ValueError("non-finite value in scenario")
    return o


def _num_canon_w(i, v):
    """length_mix rows: the weight is a float (canonical), prompt / out_len stay integers."""
    return float(v) if i == 0 and isinstance(v, int) and not isinstance(v, bool) else v


def _check_num(v, where):
    if isinstance(v, bool):
        return v
    if isinstance(v, float) and not math.isfinite(v):
        raise ValueError(f"{where}: non-finite number")
    return v


_HINTS: dict = {}
_INT_MAX = 2 ** 53


def _hints(cls) -> dict:
    if cls not in _HINTS:
        _HINTS[cls] = typing.get_type_hints(cls)
    return _HINTS[cls]


def _tname(t) -> str:
    args = typing.get_args(t)
    if args and type(None) in args:
        return " | ".join(_tname(a) for a in args if a is not type(None)) + " | null"
    if t is float:
        return "number"
    if t is int:
        return "integer"
    if t is bool:
        return "boolean"
    if t is str:
        return "string"
    if is_dataclass(t):
        return "object"
    if typing.get_origin(t) is tuple:
        return "list"
    return str(t)


def _jtype(v) -> str:
    return {bool: "boolean", int: "integer", float: "number", str: "string", list: "list", dict: "object",
            type(None): "null"}.get(type(v), type(v).__name__)


def _load_ok(ld) -> bool:
    """0.293: a measured expert-load vector: ≤ 4096 finite numbers ≥ 0 with a positive sum."""
    return (0 < len(ld) <= 4096 and all(not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)
                                        and x >= 0 for x in ld) and sum(ld) > 0)


def _coerce(v, t, where):
    """Strict JSON → field type check (0.56): wrong types raise ValueError (API → HTTP 400) instead of failing deep
    inside the evaluator.  int fields accept integral floats (2.0 → 2); float fields accept integers."""
    args = typing.get_args(t)
    if args and type(None) in args and typing.get_origin(t) is not tuple:
        if v is None:
            return None
        inner = [a for a in args if a is not type(None)]
        return _coerce(v, inner[0], where) if len(inner) == 1 else v
    if typing.get_origin(t) is tuple:
        if not isinstance(v, (list, tuple)):
            raise ValueError(f"{where}: expected a list, got {_jtype(v)}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(x, args[0], f"{where}[{n}]") for n, x in enumerate(v))
        if len(v) != len(args):
            raise ValueError(f"{where}: expected a list of {len(args)} items, got {len(v)}")
        return tuple(_coerce(x, a, f"{where}[{n}]") for n, (x, a) in enumerate(zip(v, args)))
    if t is bool:
        if not isinstance(v, bool):
            raise ValueError(f"{where}: expected boolean, got {_jtype(v)}")
        return v
    if t is int:
        if isinstance(v, float) and math.isfinite(v) and v == int(v):
            v = int(v)
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(f"{where}: expected integer, got {_jtype(v)}" + (f" {v!r}" if isinstance(v, float) else ""))
        if abs(v) > _INT_MAX:
            raise ValueError(f"{where}: integer out of range (|x| ≤ 2^53)")
        return v
    if t is float:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(f"{where}: expected number, got {_jtype(v)}")
        return _check_num(v, where)
    if t is str:
        if not isinstance(v, str):
            raise ValueError(f"{where}: expected string, got {_jtype(v)}")
        return v
    return v


def _from_plain(cls, d):
    if not isinstance(d, dict):
        raise ValueError(f"{cls.__name__}: expected object, got {_jtype(d)}")
    known = {f.name: f for f in fields(cls) if not f.metadata.get("internal")}
    unknown = set(d) - set(known)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    kw = {}
    hints = _hints(cls)
    for k, v in d.items():
        where = f"{cls.__name__}.{k}"
        t = hints[k]
        targs = [a for a in (typing.get_args(t) or (t,)) if a is not type(None)]
        dc = next((a for a in targs if is_dataclass(a)), None)
        if k in ("chip", "prefill_chip") and isinstance(v, str):
            # 0.56: a chip preset name is accepted wherever a chip object is (was: HTTP 500 for scenario.chip)
            if v not in CHIPS:
                raise ValueError(f"{where}: unknown chip preset {v!r} (one of {', '.join(CHIPS)})")
            kw[k] = CHIPS[v]
        elif dc is not None:
            if v is None and type(None) in typing.get_args(t):
                kw[k] = None
            elif isinstance(v, dict):
                kw[k] = _from_plain(dc, v)
            else:
                raise ValueError(f"{where}: expected {_tname(t)}, got {_jtype(v)}")
        elif k == "prefix_tree":
            if not isinstance(v, list) or not all(isinstance(x, list) and len(x) == 3 for x in v):
                raise ValueError("PDConfig.prefix_tree: expected a list of [tokens, branching, zipf]")
            def _row(n, x):
                out = []
                for i, (y, ty) in enumerate(zip(x, (int, int, float))):
                    c = _coerce(y, ty, f"PDConfig.prefix_tree[{n}][{i}]")
                    out.append(float(c) if i == 2 and isinstance(c, int) and not isinstance(c, bool) else c)
                return tuple(out)
            kw[k] = tuple(_row(n, x) for n, x in enumerate(v))
        elif k == "length_mix":
            if not isinstance(v, list) or not all(isinstance(x, list) and len(x) == 3 for x in v):
                raise ValueError("PDConfig.length_mix: expected a list of [weight, prompt, out_len]")
            kw[k] = tuple(tuple(_num_canon_w(i, _coerce(y, ty, f"PDConfig.length_mix[{n}][{i}]"))
                                for i, (y, ty) in enumerate(zip(x, (float, int, int)))) for n, x in enumerate(v))
        else:
            if isinstance(v, dict):
                raise ValueError(f"{where}: unexpected structure (expected {_tname(t)}, got object)")
            kw[k] = _num_canon(known[k], _coerce(v, t, where))
    return cls(**kw)

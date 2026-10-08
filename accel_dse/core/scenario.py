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
from dataclasses import dataclass, field, fields, is_dataclass

from .dtypes import FormatSupport
from .d2d_catalog import D2D_DEFAULT_STD, D2D_DEFAULT_UNITS, D2D_STANDARDS, d2d_GBps
from .hardware import CHIPS, D2D_DEFAULT, NET_DEFAULT, Chip, Link
from .mapping import ORGS
from .parallel import Layout


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
    tpot_slo_ms: float = 50.0
    ttft_slo_ms: float = 2000.0
    moe_skew: float = 1.0          # MoE: busiest EP rank's token-expert pairs / mean (0.49; 1 = uniform) 「假设」
    moe_expert_load: tuple[float, ...] = ()   # MoE: relative tokens per expert (measured) → skew per EP layout
    prefix_cached: int = 0         # prefill: prompt tokens already in the KV cache (prefix-cache hit, 0.52) 「假设」

    def __post_init__(self):
        if self.phase not in ("decode", "prefill"):
            raise ValueError("serving.phase must be decode|prefill")
        for k in ("batch", "prompt", "out_len"):
            if not isinstance(getattr(self, k), int) or getattr(self, k) < 1:
                raise ValueError(f"serving.{k} must be an integer ≥ 1")
        for k in ("ctx", "microbatches", "spec_k"):
            if not isinstance(getattr(self, k), int) or getattr(self, k) < 0:
                raise ValueError(f"serving.{k} must be an integer ≥ 0")
        if self.spec_k > 8:
            raise ValueError("serving.spec_k must be ≤ 8")
        if isinstance(self.prefix_cached, bool) or not isinstance(self.prefix_cached, int) \
                or not 0 <= self.prefix_cached < self.prompt:
            raise ValueError("serving.prefix_cached must be an integer in [0, prompt)")
        if not (0.0 <= self.spec_accept <= 1.0):
            raise ValueError("serving.spec_accept must be in [0, 1]")
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

    def __post_init__(self):
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
    search_decode_batch: bool = False   # layout search (0.53): also pick each decode layout's batch (B/2 … 4B, TPOT SLO)
    simulate: bool = False          # 0.54: also run the request-level DES (core/pdsim) and attach per-mode tails 「假设」

    def __post_init__(self):
        for k in ("prefill_cards", "decode_cards"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 4096:
                raise ValueError(f"pd.{k} must be an integer in [0, 4096]")
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
        if self.prefill_chip is not None and not isinstance(self.prefill_chip, Chip):
            raise ValueError("pd.prefill_chip must be a chip preset name, a chip object or null")
        if self.prefill_mem_id is not None and (not isinstance(self.prefill_mem_id, str) or not self.prefill_mem_id
                                                or len(self.prefill_mem_id) > 64):
            raise ValueError("pd.prefill_mem_id must be a memory id or null")
        if isinstance(self.prefix_len, bool) or not isinstance(self.prefix_len, int) or not 0 <= self.prefix_len <= 1 << 21:
            raise ValueError("pd.prefix_len must be an integer in [0, 2097152]")
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


@dataclass(frozen=True)
class Scenario:
    model: str = "qwen3-8b"
    chip: Chip = CHIPS["100T"]
    mem_id: str = "lpddr5x_4x64_8533_16g"
    mem_eff: float | None = None
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

    def __post_init__(self):
        if self.mapping not in ORGS:
            raise ValueError(f"mapping must be one of {ORGS}")
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
        if self.node_cards and self.d2d_enabled and self.node_cards % self.package_cards:
            raise ValueError("node_cards must be a multiple of package_cards")
        if self.pd.decode_cards and self.pd.decode_cards % self.layout.cards:
            raise ValueError(f"pd.decode_cards must be a multiple of the (decode) layout's {self.layout.cards} cards")
        if self.mem_eff is not None and not (0 < self.mem_eff <= 1):
            raise ValueError("mem_eff must be in (0,1]")

    @property
    def d2d_link(self) -> Link:
        """Effective D2D link: the catalog grade × units (raw, per direction), or ``d2d`` when ``d2d_std`` = custom."""
        if self.d2d_std == "custom":
            return self.d2d
        return Link(d2d_GBps(self.d2d_std, self.d2d_units), self.d2d.alpha_us, self.d2d.topology)

    @property
    def package_eff(self) -> int:
        return self.package_cards if self.d2d_enabled else 1

    # ---- controlled replacement
    def replace(self, path: str, value) -> "Scenario":
        parts = path.split(".")
        return _replace_path(self, parts, value)

    def to_dict(self) -> dict:
        return _to_plain(self)

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


def _to_plain(o):
    if is_dataclass(o):
        return {f.name: _to_plain(_num_canon(f, getattr(o, f.name))) for f in fields(o)}
    if isinstance(o, tuple):
        return [_to_plain(x) for x in o]
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


def _from_plain(cls, d):
    if not isinstance(d, dict):
        raise ValueError(f"{cls.__name__}: expected object")
    known = {f.name: f for f in fields(cls)}
    unknown = set(d) - set(known)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    kw = {}
    hints = {"chip": Chip, "prefill_chip": Chip, "link": Link, "d2d": Link, "net": Link, "pd": PDConfig, "prefill_layout": Layout, "layout": Layout, "serving": Serving, "formats": FormatSupport,
             "workload": Workload}
    for k, v in d.items():
        if k == "prefill_chip" and isinstance(v, str):
            if v not in CHIPS:
                raise ValueError(f"PDConfig.prefill_chip: unknown chip preset {v!r} (one of {', '.join(CHIPS)})")
            kw[k] = CHIPS[v]
        elif k in hints and isinstance(v, dict):
            kw[k] = _from_plain(hints[k], v)
        elif k in ("formats_override", "rates") and isinstance(v, list):
            kw[k] = tuple(tuple(x) for x in v)
        elif k == "length_mix" and isinstance(v, list):
            if not all(isinstance(x, list) for x in v):
                raise ValueError("PDConfig.length_mix: expected a list of [weight, prompt, out_len]")
            kw[k] = tuple(tuple(_num_canon_w(i, _check_num(y, "PDConfig.length_mix")) for i, y in enumerate(x)) for x in v)
        elif k == "moe_expert_load" and isinstance(v, list):
            kw[k] = tuple(_check_num(x, "Serving.moe_expert_load") for x in v)
        else:
            if isinstance(v, (list, dict)):
                raise ValueError(f"{cls.__name__}.{k}: unexpected structure")
            kw[k] = _num_canon(known[k], _check_num(v, f"{cls.__name__}.{k}"))
    return cls(**kw)

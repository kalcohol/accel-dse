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
from .hardware import CHIPS, Chip, Link
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
        if not (0.0 <= self.spec_accept <= 1.0):
            raise ValueError("serving.spec_accept must be in [0, 1]")
        for k in ("tpot_slo_ms", "ttft_slo_ms"):
            v = getattr(self, k)
            if not (v > 0 and math.isfinite(v)):
                raise ValueError(f"serving.{k} must be finite > 0")


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
        for k in ("clip_slo_s", "seq_slo_ms", "fold_slo_s"):
            v = getattr(self, k)
            if not (v > 0 and math.isfinite(v)):
                raise ValueError(f"workload.{k} must be finite > 0")


@dataclass(frozen=True)
class Scenario:
    model: str = "qwen3-8b"
    chip: Chip = CHIPS["100T"]
    mem_id: str = "lpddr5x_4x64_8533_16g"
    mem_eff: float | None = None
    link: Link = Link()
    mapping: str = "os"
    layout: Layout = Layout()
    serving: Serving = Serving()
    formats_override: tuple[tuple[str, str], ...] = ()   # what-if (labelled)
    workload: Workload = Workload()                      # video / protein models only

    def __post_init__(self):
        if self.mapping not in ORGS:
            raise ValueError(f"mapping must be one of {ORGS}")
        if self.mem_eff is not None and not (0 < self.mem_eff <= 1):
            raise ValueError("mem_eff must be in (0,1]")

    # ---- controlled replacement
    def replace(self, path: str, value) -> "Scenario":
        parts = path.split(".")
        return _replace_path(self, parts, value)

    def to_dict(self) -> dict:
        return _to_plain(self)

    @staticmethod
    def from_dict(d: dict) -> "Scenario":
        return _from_plain(Scenario, d)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]


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
    hints = {"chip": Chip, "link": Link, "layout": Layout, "serving": Serving, "formats": FormatSupport,
             "workload": Workload}
    for k, v in d.items():
        if k in hints and isinstance(v, dict):
            kw[k] = _from_plain(hints[k], v)
        elif k in ("formats_override", "rates") and isinstance(v, list):
            kw[k] = tuple(tuple(x) for x in v)
        else:
            if isinstance(v, (list, dict)):
                raise ValueError(f"{cls.__name__}.{k}: unexpected structure")
            kw[k] = _num_canon(known[k], _check_num(v, f"{cls.__name__}.{k}"))
    return cls(**kw)

"""HF / model-card series catalog loader.

Loads ``accel_dse/data/series_catalog.json`` (baked from public HF configs /
model cards / cited schema — see ``scripts/bake_series_catalog.py``) and
constructs ``ModelShape`` / ``VideoShape`` / ``ProteinShape`` instances.

Honesty:
  - Architecture dims are from public sources (not invented).
  - FLOPs / BW / frequency remain uncalibrated.
  - Hybrid linear-attn / Engram / DSA extras live in metadata only; core DSE
    still uses ModelShape GQA/MoE/MLA accounting.
  - Video clip geometry (frames/latent/n_denoise) and protein seq_len use
    documented workload defaults when not in HF config.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

from .model_shape import ModelShape
from .workloads import ProteinShape, VideoShape

_CATALOG_NAME = "series_catalog.json"


def _catalog_path() -> Path:
    """Resolve packaged data file (editable install or source tree)."""
    try:
        ref = resources.files("accel_dse").joinpath("data", _CATALOG_NAME)
        with resources.as_file(ref) as p:
            if p.is_file():
                return Path(p)
    except (TypeError, FileNotFoundError, ModuleNotFoundError):
        pass
    here = Path(__file__).resolve().parent / "data" / _CATALOG_NAME
    if here.is_file():
        return here
    raise FileNotFoundError(
        f"series catalog not found (expected accel_dse/data/{_CATALOG_NAME})"
    )


@lru_cache(maxsize=1)
def load_catalog_raw() -> dict[str, Any]:
    path = _catalog_path()
    return json.loads(path.read_text(encoding="utf-8"))


def shape_from_dict(d: dict[str, Any]) -> ModelShape | VideoShape | ProteinShape:
    """Build a frozen shape dataclass from a catalog shape dict."""
    kind = d.get("kind")
    if kind == "llm":
        return ModelShape(
            name=str(d["name"]),
            n_layers=int(d["n_layers"]),
            hidden=int(d["hidden"]),
            n_heads=int(d["n_heads"]),
            n_kv_heads=int(d["n_kv_heads"]),
            head_dim=int(d["head_dim"]),
            intermediate=int(d["intermediate"]),
            vocab=int(d["vocab"]),
            weight_bits=int(d.get("weight_bits", 16)),
            act_bits=int(d.get("act_bits", 16)),
            kv_bits=int(d.get("kv_bits", 16)),
            n_experts=int(d.get("n_experts", 1)),
            top_k=int(d.get("top_k", 1)),
            n_shared_experts=int(d.get("n_shared_experts", 0)),
            kv_lora_rank=int(d.get("kv_lora_rank", 0)),
            # v0.30 MLA / low-rank projections + dense prefix + untied LM head
            q_lora_rank=int(d.get("q_lora_rank", 0)),
            qk_nope_head_dim=int(d.get("qk_nope_head_dim", 0)),
            qk_rope_head_dim=int(d.get("qk_rope_head_dim", 0)),
            v_head_dim=int(d.get("v_head_dim", 0)),
            o_lora_rank=int(d.get("o_lora_rank", 0)),
            o_groups=int(d.get("o_groups", 0)),
            n_dense_layers=int(d.get("n_dense_layers", 0)),
            dense_intermediate=int(d.get("dense_intermediate", 0)),
            tie_embeddings=bool(d.get("tie_embeddings", True)),
            n_mtp_layers=int(d.get("n_mtp_layers", 0)),
        )
    if kind == "video":
        return VideoShape(
            name=str(d["name"]),
            n_frames=int(d["n_frames"]),
            latent_h=int(d["latent_h"]),
            latent_w=int(d["latent_w"]),
            patch_size=int(d["patch_size"]),
            hidden=int(d["hidden"]),
            n_layers=int(d["n_layers"]),
            n_heads=int(d["n_heads"]),
            head_dim=int(d["head_dim"]),
            intermediate=int(d["intermediate"]),
            n_denoise=int(d["n_denoise"]),
            batch=int(d.get("batch", 1)),
            weight_bits=int(d.get("weight_bits", 16)),
            act_bits=int(d.get("act_bits", 16)),
        )
    if kind == "protein":
        return ProteinShape(
            name=str(d["name"]),
            seq_len=int(d["seq_len"]),
            hidden=int(d["hidden"]),
            n_layers=int(d["n_layers"]),
            n_heads=int(d["n_heads"]),
            head_dim=int(d["head_dim"]),
            intermediate=int(d["intermediate"]),
            pair_dim=int(d.get("pair_dim", 0)),
            msa_depth=int(d.get("msa_depth", 1)),
            batch=int(d.get("batch", 1)),
            weight_bits=int(d.get("weight_bits", 16)),
            act_bits=int(d.get("act_bits", 16)),
        )
    raise ValueError(f"unknown catalog shape kind {kind!r}")


def iter_catalog_entries() -> list[dict[str, Any]]:
    """Raw entry dicts from the baked catalog."""
    return list(load_catalog_raw().get("entries") or [])


def catalog_skipped() -> list[dict[str, Any]]:
    return list(load_catalog_raw().get("skipped") or [])


def catalog_counts() -> dict[str, int]:
    entries = iter_catalog_entries()
    return {
        "llm": sum(1 for e in entries if e.get("domain") == "llm"),
        "video": sum(1 for e in entries if e.get("domain") == "video"),
        "protein": sum(1 for e in entries if e.get("domain") == "protein"),
        "total": len(entries),
        "skipped": len(catalog_skipped()),
    }

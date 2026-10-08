"""Model series registry — illustrative placeholders + real public HF packs.

Illustrative / toy entries remain for regression and handcheck.

Real packs are loaded from ``npu_dse/data/series_catalog.json`` (public HF
config.json / model cards / cited schema). Metadata cites ``hf:<id>`` + source.
FLOPs/BW remain uncalibrated; hybrid linear-attn / Engram / DSA extras are
metadata-only (core DSE still uses ModelShape GQA/MoE/MLA accounting).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .catalog import iter_catalog_entries, shape_from_dict
from .model_shape import (
    ILLUSTRATIVE_27B,
    ILLUSTRATIVE_MLA,
    ILLUSTRATIVE_MOE,
    TOY_SHAPE,
    ModelShape,
)
from .workloads import (
    ILLUSTRATIVE_DIT_VIDEO,
    ILLUSTRATIVE_LARGE_DIT,
    ILLUSTRATIVE_PROTEIN_PAIR,
    TOY_PROTEIN,
    TOY_VIDEO,
    ProteinShape,
    VideoShape,
)

Family = Literal["dense", "moe", "mla", "dit_video", "protein", "toy"]
Domain = Literal["llm", "video", "protein"]


@dataclass(frozen=True)
class SeriesEntry:
    """One named series / shape alias in the product registry."""

    id: str
    family: Family
    domain: Domain
    shape: ModelShape | VideoShape | ProteinShape
    aliases: tuple[str, ...] = ()
    # Real packs cite hf:<id> + source; illustrative packs say illustrative/placeholder.
    metadata: str = "illustrative placeholder (not a vendor checkpoint)"

    def summary(self) -> str:
        shape_name = getattr(self.shape, "name", "?")
        return (
            f"{self.id}: family={self.family} domain={self.domain} "
            f"→ {shape_name} [{self.metadata}]"
        )

    @property
    def is_illustrative(self) -> bool:
        meta = self.metadata.lower()
        return "illustrative" in meta or "placeholder" in meta or self.family == "toy"

    @property
    def is_hf_backed(self) -> bool:
        return self.metadata.lower().startswith("hf:") or "hf:" in self.metadata.lower()


def _entry(
    id_: str,
    family: Family,
    domain: Domain,
    shape: ModelShape | VideoShape | ProteinShape,
    *aliases: str,
    metadata: str = "illustrative placeholder (not a vendor checkpoint)",
) -> SeriesEntry:
    return SeriesEntry(
        id=id_,
        family=family,
        domain=domain,
        shape=shape,
        aliases=tuple(aliases),
        metadata=metadata,
    )


SERIES_REGISTRY: dict[str, SeriesEntry] = {}


def _register(e: SeriesEntry) -> None:
    keys = (e.id, *e.aliases)
    for k in keys:
        key = k.strip()
        if not key:
            continue
        if key in SERIES_REGISTRY and SERIES_REGISTRY[key] is not e:
            existing = SERIES_REGISTRY[key]
            if existing.id != e.id and existing.shape is not e.shape:
                raise ValueError(
                    f"duplicate series key {key!r}: {existing.id} vs {e.id}"
                )
        SERIES_REGISTRY[key] = e


# ---------------------------------------------------------------------------
# Illustrative / toy (regression + handcheck) — keep clearly labeled
# ---------------------------------------------------------------------------

_register(
    _entry(
        "illustrative_27B",
        "dense",
        "llm",
        ILLUSTRATIVE_27B,
        "27b",
        metadata="illustrative/placeholder dense ~27B-class GQA (NOT a vendor checkpoint)",
    )
)
_register(
    _entry(
        "illustrative_moe",
        "moe",
        "llm",
        ILLUSTRATIVE_MOE,
        "moe",
        metadata=(
            "illustrative/placeholder MoE (~45B total / ~13B active); "
            "NOT Mixtral/DeepSeek claim"
        ),
    )
)
_register(
    _entry(
        "illustrative_mla",
        "mla",
        "llm",
        ILLUSTRATIVE_MLA,
        "mla",
        metadata=(
            "illustrative/placeholder MLA-style compressed KV on 27B-class body; "
            "NOT a DeepSeek checkpoint claim"
        ),
    )
)
_register(
    _entry(
        "illustrative_dit_video",
        "dit_video",
        "video",
        ILLUSTRATIVE_DIT_VIDEO,
        "dit_video",
        metadata="illustrative/placeholder DiT-video (NOT Sora/CogVideo claim)",
    )
)
_register(
    _entry(
        "illustrative_large_dit",
        "dit_video",
        "video",
        ILLUSTRATIVE_LARGE_DIT,
        "large_dit",
        metadata=(
            "illustrative/placeholder large DiT (minute-scale TTFC narrative); "
            "NOT MiniMax-H3 claim — use minimax-h3 for real HF dims"
        ),
    )
)
_register(
    _entry(
        "illustrative_protein_pair",
        "protein",
        "protein",
        ILLUSTRATIVE_PROTEIN_PAIR,
        "protein",
        "protein_pair",
        metadata="illustrative/placeholder protein+pair (NOT AF2/ESM claim)",
    )
)
_register(
    _entry(
        "toy",
        "toy",
        "llm",
        TOY_SHAPE,
        "toy_llm",
        metadata="toy hand-check shape (illustrative placeholder)",
    )
)
_register(
    _entry(
        "toy_video",
        "dit_video",
        "video",
        TOY_VIDEO,
        metadata="toy video shape (illustrative placeholder)",
    )
)
_register(
    _entry(
        "toy_protein",
        "protein",
        "protein",
        TOY_PROTEIN,
        metadata="toy protein shape (illustrative placeholder)",
    )
)

# Illustrative product-facing series packs (handcheck / regression aliases)
_register(
    _entry(
        "series/dense-27b",
        "dense",
        "llm",
        ILLUSTRATIVE_27B,
        "series/dense_27b",
        metadata=(
            "product series alias → illustrative_27B "
            "(illustrative/placeholder; NOT a vendor checkpoint)"
        ),
    )
)
_register(
    _entry(
        "series/moe-active13b",
        "moe",
        "llm",
        ILLUSTRATIVE_MOE,
        "series/moe_active13b",
        metadata=(
            "product series alias → illustrative_moe "
            "(illustrative/placeholder active~13B; NOT Mixtral/DeepSeek)"
        ),
    )
)
_register(
    _entry(
        "series/mla-27b",
        "mla",
        "llm",
        ILLUSTRATIVE_MLA,
        "series/mla_27b",
        metadata=(
            "product series alias → illustrative_mla "
            "(illustrative/placeholder; NOT a DeepSeek checkpoint)"
        ),
    )
)
_register(
    _entry(
        "series/dit-video",
        "dit_video",
        "video",
        ILLUSTRATIVE_DIT_VIDEO,
        "series/dit_video",
        "series/video",
        "series/video-dit",
        metadata=(
            "product series alias → illustrative_dit_video "
            "(illustrative/placeholder; NOT Sora/CogVideo)"
        ),
    )
)
_register(
    _entry(
        "series/dit-large",
        "dit_video",
        "video",
        ILLUSTRATIVE_LARGE_DIT,
        "series/dit_large",
        "series/large-dit",
        "series/large_dit",
        metadata=(
            "product series alias → illustrative_large_dit "
            "(illustrative/placeholder; NOT MiniMax-H3 — use minimax-h3)"
        ),
    )
)
_register(
    _entry(
        "series/protein-pair",
        "protein",
        "protein",
        ILLUSTRATIVE_PROTEIN_PAIR,
        "series/protein_pair",
        "series/protein",
        metadata=(
            "product series alias → illustrative_protein_pair "
            "(illustrative/placeholder; NOT AF2/ESM)"
        ),
    )
)

ILLUSTRATIVE_SERIES_IDS: tuple[str, ...] = (
    "series/dense-27b",
    "series/moe-active13b",
    "series/mla-27b",
    "series/dit-video",
    "series/dit-large",
    "series/protein-pair",
)

# ---------------------------------------------------------------------------
# Real public HF / schema packs from baked catalog
# ---------------------------------------------------------------------------

_HF_PRODUCT_IDS: list[str] = []


def _register_catalog() -> None:
    for raw in iter_catalog_entries():
        sid = str(raw["id"])
        family = raw["family"]
        domain = raw["domain"]
        shape = shape_from_dict(raw["shape"])
        aliases = tuple(str(a) for a in (raw.get("aliases") or []) if a)
        metadata = str(raw.get("metadata") or f"hf:{raw.get('hf_id')}")
        e = _entry(sid, family, domain, shape, *aliases, metadata=metadata)
        _register(e)
        _HF_PRODUCT_IDS.append(sid)


_register_catalog()

# Product-facing: real HF-backed packs (list-series --product)
PRODUCT_SERIES_IDS: tuple[str, ...] = tuple(_HF_PRODUCT_IDS)

# Extra product aliases for illustrative packs (resolve via registry)
PRODUCT_SERIES_ALIASES: tuple[str, ...] = (
    "series/video",
    "series/video-dit",
    "series/large-dit",
    "series/protein",
)

FAMILIES: tuple[Family, ...] = ("dense", "moe", "mla", "dit_video", "protein", "toy")


def get_series(name: str) -> SeriesEntry:
    """Resolve a series / shape id."""
    key = name.strip()
    if key in SERIES_REGISTRY:
        return SERIES_REGISTRY[key]
    low = key.lower()
    for k, e in SERIES_REGISTRY.items():
        if k.lower() == low:
            return e
    for cand in (low.replace("-", "_"), low.replace("_", "-")):
        for k, e in SERIES_REGISTRY.items():
            if k.lower() == cand:
                return e
    known = sorted({e.id for e in SERIES_REGISTRY.values()})
    raise KeyError(f"unknown series/model {name!r}; known ids include {known[:40]}…")


def resolve_llm_shape(name: str) -> ModelShape:
    """Resolve an LLM ModelShape; raises if the series is video/protein."""
    e = get_series(name)
    if not isinstance(e.shape, ModelShape):
        raise TypeError(
            f"{name!r} resolves to domain={e.domain}; expected an LLM ModelShape"
        )
    return e.shape


def resolve_video_shape(name: str) -> VideoShape:
    """Resolve a VideoShape; raises if the series is not video."""
    e = get_series(name)
    if not isinstance(e.shape, VideoShape):
        raise TypeError(
            f"{name!r} resolves to domain={e.domain}; expected a VideoShape"
        )
    return e.shape


def resolve_protein_shape(name: str) -> ProteinShape:
    """Resolve a ProteinShape; raises if the series is not protein."""
    e = get_series(name)
    if not isinstance(e.shape, ProteinShape):
        raise TypeError(
            f"{name!r} resolves to domain={e.domain}; expected a ProteinShape"
        )
    return e.shape


def list_series(*, product_only: bool = False) -> list[SeriesEntry]:
    """Unique series entries (by id).

    product_only=True → real HF-backed packs (PRODUCT_SERIES_IDS).
    product_only=False → all entries including illustrative/toy.
    """
    seen: set[str] = set()
    out: list[SeriesEntry] = []
    if product_only:
        for sid in PRODUCT_SERIES_IDS:
            e = SERIES_REGISTRY[sid]
            if e.id not in seen:
                seen.add(e.id)
                out.append(e)
        return out
    for e in SERIES_REGISTRY.values():
        if e.id in seen:
            continue
        seen.add(e.id)
        out.append(e)
    out.sort(key=lambda x: (0 if x.is_hf_backed else 1, x.domain, x.family, x.id))
    return out


def list_series_rows() -> list[tuple[str, str, str, str, str]]:
    """Rows: (id, family, domain, shape_name, metadata)."""
    rows = []
    for e in list_series():
        shape_name = getattr(e.shape, "name", "?")
        rows.append((e.id, e.family, e.domain, shape_name, e.metadata))
    return rows

"""Per-model memo tables (keyed by object identity; models are immutable).

Deep-hashing a ModelSpec / Layer on every lookup dominated evaluation time, so
derived per-model data (layer groups, stage storage, what-if variants) is cached
against the model object itself, guarded by a weak reference against id reuse.
"""

from __future__ import annotations

import weakref

_TABLES: dict[int, tuple] = {}
_MAX = 1024


def model_cache(m) -> dict:
    e = _TABLES.get(id(m))
    if e is None or e[0]() is not m:
        if len(_TABLES) >= _MAX:
            _TABLES.clear()
        e = (weakref.ref(m), {})
        _TABLES[id(m)] = e
    return e[1]


def layer_groups(m) -> list[int]:
    """Group id per layer: identical Layer specs share an id (op sums are computed once per group)."""
    mc = model_cache(m)
    g = mc.get("groups")
    if g is None:
        ids: dict = {}
        g = [ids.setdefault(L, len(ids)) for L in m.layers]
        mc["groups"] = g
    return g

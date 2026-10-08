"""L2 — parallel layout and pipeline-stage plan.

A ``Layout`` places one model replica on ``pp·tp·dp`` cards:
  pp  pipeline stages (integer, contiguous layer ranges)
  tp  attention tensor parallel; dp attention data parallel inside a stage
  ep·etp = tp·dp  expert parallel × expert tensor parallel for MoE layers
  sp  sequence parallel (Ulysses) — non-autoregressive models only (video DiT, protein encoders);
      their dp splits the forward batch (incl. the CFG cond/uncond pair) inside the replica
Cards = pp·tp·dp·sp.  Single card = Layout() (all 1).  There is no second code path.

Stage plan: layers split contiguously, remainder to the *first* stages
(61 / 8 → [8,8,8,8,8,7,7,7]); embedding on stage 0; final norm + LM head +
MTP modules on the last stage.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ir import Shard


@dataclass(frozen=True)
class Layout:
    pp: int = 1
    tp: int = 1
    dp: int = 1
    ep: int = 1
    etp: int = 1
    sp: int = 1

    def __post_init__(self):
        for k in ("pp", "tp", "dp", "ep", "etp", "sp"):
            v = getattr(self, k)
            if not isinstance(v, int) or v < 1:
                raise ValueError(f"layout.{k} must be an integer ≥ 1 (got {v!r})")

    @property
    def cards(self) -> int:
        return self.pp * self.tp * self.dp * self.sp

    @property
    def shard(self) -> Shard:
        return Shard(self.tp, self.dp, self.ep, self.etp, self.sp)

    def valid_for(self, moe: bool, full: bool = False) -> bool:
        """MoE LLM: ep·etp = tp·dp; dense LLM: dp = ep = etp = 1; both sp = 1.
        Non-autoregressive (``full``, dense): ep = etp = 1, any dp / sp."""
        if full:
            return self.ep == 1 and self.etp == 1
        if self.sp != 1:
            return False
        if moe:
            return self.ep * self.etp == self.tp * self.dp
        return self.ep == 1 and self.etp == 1 and self.dp == 1

    @property
    def label(self) -> str:
        s = f"PP{self.pp}·TP{self.tp}"
        if self.dp > 1:
            s += f"·DP{self.dp}"
        if self.ep > 1:
            s += f"·EP{self.ep}"
        if self.etp > 1:
            s += f"·ETP{self.etp}"
        if self.sp > 1:
            s += f"·SP{self.sp}"
        return s


@dataclass(frozen=True)
class Stage:
    index: int
    first: int          # first layer (inclusive)
    last: int           # last layer (exclusive)
    has_embed: bool
    has_head: bool

    @property
    def n_layers(self) -> int:
        return self.last - self.first


def stage_layer_counts(n_layers: int, pp: int) -> list[int]:
    if pp < 1 or pp > n_layers:
        raise ValueError(f"pp must be in [1, {n_layers}] (got {pp})")
    base, rem = divmod(n_layers, pp)
    return [base + (1 if i < rem else 0) for i in range(pp)]


def plan_stages(n_layers: int, pp: int) -> list[Stage]:
    out, start = [], 0
    counts = stage_layer_counts(n_layers, pp)
    for i, c in enumerate(counts):
        out.append(Stage(i, start, start + c, i == 0, i == pp - 1))
        start += c
    return out


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def enumerate_layouts(cards: int, n_layers: int, moe: bool, *, max_tp: int | None = None,
                      full: bool = False) -> list[Layout]:
    """All layouts that use exactly ``cards`` cards for one replica (``full``: PP·TP·DP·SP of a dense
    non-autoregressive model)."""
    out = []
    if full:
        for pp in _divisors(cards):
            if pp > n_layers:
                continue
            for tp in _divisors(cards // pp):
                if max_tp and tp > max_tp:
                    continue
                for dp in _divisors(cards // pp // tp):
                    out.append(Layout(pp, tp, dp, 1, 1, cards // pp // tp // dp))
        return out
    for pp in _divisors(cards):
        if pp > n_layers:
            continue
        rest = cards // pp
        for tp in _divisors(rest):
            if max_tp and tp > max_tp:
                continue
            dp = rest // tp
            if not moe:
                if dp == 1:
                    out.append(Layout(pp, tp, 1, 1, 1))
                continue
            for ep in _divisors(rest):
                out.append(Layout(pp, tp, dp, ep, rest // ep))
    return out

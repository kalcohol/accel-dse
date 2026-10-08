"""Energy from action counts × a user-supplied energy table (0.47.1) — Accelergy-style ERT, nothing built in.

The engine already counts what every component does; this module only multiplies.  **No energy number ships with the
tool**: every entry of :class:`EnergyTable` is ``None`` until the user supplies one (a measurement, a vendor figure, a
paper they trust), and a missing entry contributes nothing and is listed as missing.  Without any entry the report
still gives the action counts per output unit — model outputs that need no assumption.

Actions (whole system: every rank of every stage and replica)
  mac    bf16-equivalent MAC slots = useful FLOPs / 2 / the format's rate multiplier (an fp8 MAC at 2× rate counts ½)
         — array cycles × MACs at 100 % utilisation; padding / idle slots of a partly filled tile are not charged
  vec    vector element-ops (norm, softmax, activation, dequant / convert)
  sram   bytes through the SRAM ↔ datapath port (the feed traffic of every GEMM / attention tile)
  dram   bytes read + written in external memory (weights, KV, activation streaming, FSDP gathers); with a
         system-level cache only the misses
  slc    bytes served by the system-level cache (0.48, chip.slc_mib > 0)
  link   bytes a rank sends on the scale-up / network tier (collectives' payload per rank, PP hand-offs, VAE tile
         gathers)
  d2d    the share of those bytes that stays on the die-to-die tier of a package (0.48, package_cards > 1; split
         by the ratio of bytes sent per tier of each collective, see core/schedule.py)
  idle   static / idle power × cards × the time window (``idle_W`` per card)

Window and units
  LLM decode     one decode step → tokens = throughput × step (all micro-batches, all replicas)
  LLM prefill    one prefill (TTFT) → prompt tokens
  video          one request batch: steps × the DiT stages + text encoder (on card) + VAE decode; window = clip latency
                 (the overlapped period with ``workload.overlap``) → frames; a host-CPU encoder's energy is not counted
  protein        one forward → sequences
Each stage executes ``microbatches`` ticks per pass; a tick's counts are per rank, × TP·SP·DP ranks.

MoE load skew (0.49, ``serving.moe_skew`` / ``moe_expert_load``): the stage time follows the busiest EP rank, but
skew only moves work between ranks — the work counts (MAC, vector, SRAM, DRAM, SLC, link, D2D) are taken per output
unit from the uniform-routing evaluation of the same scenario, while the window (and so card·s) is the skewed one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

ACTIONS = ("mac", "vec", "sram", "slc", "dram", "d2d", "link", "idle")
_UNIT_PJ = {"mac": "pJ_mac", "vec": "pJ_vec", "sram": "pJ_bit_sram", "slc": "pJ_bit_slc", "dram": "pJ_bit_dram",
            "d2d": "pJ_bit_d2d", "link": "pJ_bit_link"}
_BITS = ("sram", "slc", "dram", "d2d", "link")


@dataclass(frozen=True)
class EnergyTable:
    """Energy per action (all user-supplied; ``None`` = not provided)."""
    pJ_mac: float | None = None        # per bf16-equivalent MAC
    pJ_vec: float | None = None        # per vector element-op
    pJ_bit_sram: float | None = None   # per bit through the SRAM port
    pJ_bit_dram: float | None = None   # per bit of DRAM traffic
    pJ_bit_link: float | None = None   # per bit sent on the scale-up / network tier
    idle_W: float | None = None        # static / idle power per card (W)
    pJ_bit_slc: float | None = None    # per bit served by the system-level cache (0.48)
    pJ_bit_d2d: float | None = None    # per bit on the die-to-die tier (0.48)

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v < 1e6:
                raise ValueError(f"energy.{f.name} must be a finite number in [0, 1e6) or null")

    @property
    def provided(self) -> list[str]:
        return [f.name for f in fields(self) if getattr(self, f.name) is not None]

    @staticmethod
    def from_dict(d: dict | None) -> "EnergyTable":
        if d is None:
            return EnergyTable()
        if not isinstance(d, dict):
            raise ValueError("energy: expected an object")
        known = {f.name for f in fields(EnergyTable)}
        bad = set(d) - known
        if bad:
            raise ValueError(f"energy: unknown keys {sorted(bad)}")
        return EnergyTable(**d)


def action_counts(r) -> dict:
    """(counts per window, window s, output units per window, unit label) for one evaluated result."""
    sv = r.scenario.serving
    if sv.moe_skew != 1.0 or sv.moe_expert_load:
        from dataclasses import replace
        from .evaluate import evaluate
        bal = action_counts(evaluate(replace(r.scenario, serving=replace(sv, moe_skew=1.0, moe_expert_load=()))))
        out = _counts(r)
        k = out["units"] / bal["units"] if bal["units"] else 0.0
        out["counts"] = {a: bal["counts"][a] * k for a in ACTIONS}
        return out
    return _counts(r)


def _counts(r) -> dict:
    lay, ch = r.scenario.layout, r.scenario.chip
    f = ch.freq_ghz * 1e9
    ranks = lay.tp * lay.sp * lay.dp
    w = r.workload
    passes = r.microbatches * (w.steps if w is not None and w.kind == "gen" else 1)
    c = dict.fromkeys(ACTIONS, 0.0)
    for st in r.stages:
        t = st.time
        n = passes * ranks
        c["mac"] += t.t_ideal * ch.macs * f * n
        c["vec"] += t.t_vector * ch.lanes * f * n
        c["sram"] += st.sram_bytes * n
        c["dram"] += t.dram_bytes * n
        c["slc"] += t.slc_bytes * n
        c["d2d"] += t.d2d_bytes * n
        c["link"] += (t.link_bytes - t.d2d_bytes) * n
    if w is None:
        window = r.step
        unit = "token" if r.scenario.serving.phase == "decode" else "prompt token"
    elif w.kind == "gen":
        pl = r.pipeline or {}
        window = pl.get("period_s", r.latency)
        unit = w.unit
        for p in pl.get("parts", []):
            if p.get("on_host") or "acts" not in p:
                continue
            for k, v in p["acts"].items():
                c[k] += v * lay.dp                       # components run once per replica's batch share
    else:
        window = r.latency
        unit = w.unit
    units = r.throughput * window
    return {"counts": c, "window_s": window, "units": units, "unit": unit}


def energy_report(r, table: EnergyTable | None = None) -> dict:
    """Action counts per output unit; energy (J / unit, by action) only for the entries the user supplied."""
    table = table or EnergyTable()
    a = action_counts(r)
    u = a["units"] or 1.0
    per = {k: v / u for k, v in a["counts"].items() if k != "idle"}
    per["idle_card_s"] = r.scenario.layout.cards * a["window_s"] / u
    out = {"unit": a["unit"], "window_s": a["window_s"], "units_per_window": a["units"],
           "counts_per_unit": per, "provided": table.provided,
           "missing": [f.name for f in fields(EnergyTable) if getattr(table, f.name) is None],
           "basis": "动作计数 × 用户提供的每动作能耗（Accelergy 式 ERT）；工具不内置任何能耗数值"}
    if not table.provided:
        return out
    j = {}
    for k, attr in _UNIT_PJ.items():
        e = getattr(table, attr)
        if e is not None:
            j[k] = per[k] * e * 1e-12 * (8 if k in _BITS else 1)
    if table.idle_W is not None:
        j["idle"] = table.idle_W * per["idle_card_s"]
    tot = sum(j.values())
    out.update(J_per_unit=tot, J_by_action=j,
               avg_W_per_card=tot * u / a["window_s"] / r.scenario.layout.cards if a["window_s"] > 0 else 0.0)
    return out

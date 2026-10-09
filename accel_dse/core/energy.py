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
  link   bytes a rank sends on the in-node scale-up tier (collectives' payload per rank, PP hand-offs, VAE tile
         gathers — minus the shares below)
  d2d    the share of the sent bytes that stays on the die-to-die tier of a package (0.48; D2D on, package_cards > 1;
         split by the ratio of bytes sent per tier of each collective, see core/schedule.py)
  net    the share that crosses nodes on the scale-out network (0.50; node_cards > 0)
  idle   static / idle power × cards × the time window (``idle_W`` per card; the PD report charges the prefill pool
         at ``idle_W_prefill`` when given, 0.56 — its chip may differ; ignored by the single-chip report)

Window and units
  LLM decode     one decode step → tokens = throughput × step (all micro-batches, all replicas)
  LLM prefill    one prefill (TTFT) → prompt tokens
  video          one request batch: steps × the DiT stages + text encoder (on card) + VAE decode; window = clip latency
                 (the overlapped period with ``workload.overlap``) → frames; a host-CPU encoder's energy is not counted
  protein        one forward → sequences
Each stage executes ``microbatches`` ticks per pass; a tick's counts are per rank, × TP·SP·DP ranks.

MoE load skew (0.49, ``serving.moe_skew`` / ``moe_expert_load``): the stage time follows the busiest EP rank, but
skew only moves work between ranks — the work counts (MAC, vector, SRAM, DRAM, SLC, link, D2D, net) are taken per output
unit from the uniform-routing evaluation of the same scenario, while the window (and so card·s) is the skewed one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

ACTIONS = ("mac", "vec", "sram", "slc", "dram", "d2d", "link", "net", "idle")
_UNIT_PJ = {"mac": "pJ_mac", "vec": "pJ_vec", "sram": "pJ_bit_sram", "slc": "pJ_bit_slc", "dram": "pJ_bit_dram",
            "d2d": "pJ_bit_d2d", "link": "pJ_bit_link", "net": "pJ_bit_net"}
_BITS = ("sram", "slc", "dram", "d2d", "link", "net")


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
    pJ_bit_net: float | None = None    # per bit on the cross-node network tier (0.50)
    idle_W_prefill: float | None = None  # PD report: static power per card of the prefill pool's chip (0.56; null = idle_W)
    pJ_bit_host: float | None = None   # per bit on the host link (PCIe class): PD kv_policy=swap transfers (0.61.4),
    #                                    video offload weight reloads (0.62)

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


def scaleup_bytes(t) -> float:
    """Bytes on the in-node scale-up tier = sent − D2D − network share (float residue of a fully tiered group → 0)."""
    x = t.link_bytes - t.d2d_bytes - t.net_bytes
    return 0.0 if x <= 1e-9 * t.link_bytes else x


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
    w = r.workload
    # 0.61.4: DP ranks that hold no sequence (batch < dp, or a short last share) do no work — byte counts are taken
    # over the busy ranks (weight reads dominate).  0.63 (external review): MAC / vector work is the mean rank's
    # useful work (Op.share: ceil splits of batch / heads / columns / tokens / expert rows only cost time) × every rank
    # of the stage, and the last micro-batch's padding is not work: × S / (mb · ⌈S / mb⌉) (ESM-2 650M batch 3 on PP 2
    # counted 4 sequences, +33 % MAC per sequence)
    seqs = r.scenario.serving.batch * (w.seqs_per_request if w is not None else 1)
    b_mb = -(-seqs // max(1, r.microbatches))
    b_rank = -(-b_mb // lay.dp)
    dp_act = min(lay.dp, -(-b_mb // b_rank)) if b_rank else lay.dp
    # 0.63: with expert parallelism a DP rank without sequences still runs its experts.  0.64: it moves only its
    # routed-expert weights (no attention / dense / KV / activation bytes) — DRAM / SLC bytes of the idle ranks are
    # the stage's expert reads (StageResult.dram["exp_dram"]), not the busy rank's whole step
    idle = (lay.dp - dp_act) * lay.tp * lay.sp if lay.ep > 1 else 0
    if lay.ep > 1:
        dp_act = lay.dp
    ranks = lay.tp * lay.sp * dp_act
    mb_fill = seqs / (max(1, r.microbatches) * b_mb) if b_mb else 1.0
    passes = r.microbatches * (w.steps if w is not None and w.kind == "gen" else 1)
    c = dict.fromkeys(ACTIONS, 0.0)
    for st in r.stages:
        t = st.time
        n = passes * ranks
        nw = passes * lay.tp * lay.sp * lay.dp * mb_fill
        c["mac"] += st.ideal_w * ch.macs * f * nw
        c["vec"] += st.vec_w * ch.lanes * f * nw + st.conv_w * ch.lanes * f * n   # weight dequant: per pass
        c["sram"] += st.sram_bytes * n
        if idle:
            nb = passes * (ranks - idle)
            c["dram"] += t.dram_bytes * nb + st.dram.get("exp_dram", t.dram_bytes) * passes * idle
            c["slc"] += t.slc_bytes * nb + st.dram.get("exp_slc", t.slc_bytes) * passes * idle
        else:
            c["dram"] += t.dram_bytes * n
            c["slc"] += t.slc_bytes * n
        c["d2d"] += t.d2d_bytes * n
        c["net"] += t.net_bytes * n
        c["link"] += scaleup_bytes(t) * n
    if w is None:
        window = r.step
        unit = "token" if r.scenario.serving.phase == "decode" else "prompt token"
        vis = getattr(r, "vision", None) or {}
        for k, v in vis.get("acts", {}).items():      # 0.62: VLM encoder at prefill (replica totals)
            c[k] += v
    elif w.kind == "gen":
        pl = r.pipeline or {}
        window = pl.get("period_s", r.latency)
        unit = w.unit
        B = r.scenario.serving.batch                  # 0.61.4: a part's acts are for one DP rank's share ⌈B/dp⌉ of the
        comp_k = B / -(-B // lay.dp)                  # requests → × B / share (was × dp: idle ranks counted)
        for p in pl.get("parts", []):
            if p.get("on_host") or "acts" not in p:
                continue
            for k, v in p["acts"].items():
                c[k] += v * comp_k                       # components run once per replica's batch share
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
    host_b = (getattr(r, "pipeline", None) or {}).get("load_bytes", 0.0)
    if host_b:              # 0.62: video sequential offload -- weights re-read over the host link every window
        per["host"] = host_b / u
    out = {"unit": a["unit"], "window_s": a["window_s"], "units_per_window": a["units"],
           "counts_per_unit": per, "provided": table.provided,
           "missing": [f.name for f in fields(EnergyTable) if getattr(table, f.name) is None
                       and f.name not in ("idle_W_prefill", "pJ_bit_host")],
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
    if per.get("host"):
        if table.pJ_bit_host is not None:
            j["host"] = per["host"] * table.pJ_bit_host * 1e-12 * 8
        else:
            out["host_note"] = "卸载放置每次从主机重载权重的字节未计能耗：未填 pJ_bit_host"
    tot = sum(j.values())
    out.update(J_per_unit=tot, J_by_action=j,
               avg_W_per_card=tot * u / a["window_s"] / r.scenario.layout.cards if a["window_s"] > 0 else 0.0)
    return out

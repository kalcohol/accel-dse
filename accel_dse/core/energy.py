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


# 0.261: SRAM leakage vs die temperature.  Aoyagi et al. (TSMC), 'A 3-nm FinFET 563-kbit 35.5-Mbit/mm² Dual-Rail
# SRAM…', VLSI 2025 / JSSC 2025, Fig. 7 text: mean standby leakage of the 0.563-Mbit HD macro (low-leak mode) 15.5 µW at
# 25 °C and 73.5 µW at 85 °C 「来源」 → ×4.742 over 60 °C.  Between / beyond the two points: exponential in T (the
# sub-threshold form, doubling every 60·ln2/ln4.742 = 26.7 °C) 「推导」 inside 25–85 °C, 「近似」 outside (to 125 °C).
SRAM_LEAK_T = {"t_ref_C": 25.0, "t_hot_C": 85.0, "ratio_hot": 73.5 / 15.5, "t_max_C": 125.0,
               "src": "Aoyagi et al. (TSMC) VLSI 2025 / JSSC 2025: 3 nm HD 6T macro standby leakage 15.5 µW @25 °C, "
                      "73.5 µW @85 °C (0.563 Mbit, low-leak mode)"}


def leak_temp_factor(t_C: float | None) -> float:
    """Leakage multiplier vs the 25 °C figure at die temperature t_C (None → 1)."""
    if t_C is None:
        return 1.0
    L = SRAM_LEAK_T
    return L["ratio_hot"] ** ((t_C - L["t_ref_C"]) / (L["t_hot_C"] - L["t_ref_C"]))


def sram_leak_uW_per_mib(table) -> float | None:
    """Charged SRAM leakage per MiB: the 25 °C field × the temperature factor (0.261)."""
    if table.uW_per_mib_sram_leak is None:
        return None
    return table.uW_per_mib_sram_leak * leak_temp_factor(getattr(table, "sram_temp_C", None))


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
    # 0.186: DRAM energy by memory kind — the scenario's mem_id picks one (HBM stack vs any other package/module);
    # unset → pJ_bit_dram.  Lets an LPDDR ↔ HBM choice move J / token by mechanism (interface + array per bit).
    pJ_bit_dram_hbm: float | None = None
    pJ_bit_dram_lpddr: float | None = None
    # 0.188: chip-side DRAM PHY (controller-side interface) per DRAM bit, by memory kind; a separate line "dram_phy"
    # so DRAM energy = memory side + SoC side.  Unset → no PHY line (memory-side only, as ≤ 0.187).
    pJ_bit_phy_hbm: float | None = None
    pJ_bit_phy_lpddr: float | None = None
    # 0.188: static power from a measured idle fraction of TDP: idle_W := idle_frac_tdp × tdp_W when idle_W is unset.
    idle_frac_tdp: float | None = None
    tdp_W: float | None = None
    # 0.212: SRAM leakage per MiB of on-die SRAM (µW / MiB).  Charged for the dedicated state buffer always (it is not
    # in any measured base-card idle figure) and for the shared SRAM only when no idle figure (idle_W / idle_frac_tdp)
    # is set — a measured card idle already contains its SRAM leakage, so no double counting.
    uW_per_mib_sram_leak: float | None = None
    # 0.216: HBM background (standby) + refresh power, static, per DRAM die and per Gb.  Charged per stack × height
    # only when no idle figure is set (a measured card idle already contains its DRAM background) — like SRAM leakage.
    mW_hbm_die_standby: float | None = None
    mW_hbm_Gb_refresh: float | None = None
    # 0.228: per-cycle dynamic energy per MAC slot (pJ per MAC-slot per core cycle) — clock tree / distribution,
    # always-clocked pipeline registers and control: paid every core cycle the card is clocked, busy or stalled, so
    # P_clk = macs × f × pJ (scales with f and, under DVFS, with V²).  The only dynamic power that falls with the clock
    # when the op rate is set by memory.  Distinct from idle_W (a measured *idle-clock* floor) — not double counted.
    pJ_clk_mac_cycle: float | None = None
    # 0.261: die (junction) temperature for SRAM / SLC leakage, °C.  uW_per_mib_sram_leak is a 25 °C figure; the charged
    # value is × leak_temp_factor(T) (same macro measured at 25 and 85 °C, see SRAM_LEAK_T).  None = 25 °C (lower bound).
    sram_temp_C: float | None = None
    # 0.266: capacity-dependent SLC access energy.  None = flat pJ_bit_slc (legacy 「假设」= SRAM).  When set to a
    # reference capacity C_ref (MiB), effective pJ/bit = pJ_bit_slc · max(1, √(slc_mib / C_ref)).  Horowitz ISSCC 2014
    # Fig. 1.1.9: 8 KB / 32 KB / 1 MB SRAM accesses 10 / 20 / 100 pJ (64-bit) ⇒ E ∝ √C 「推导」; H-tree dominates
    # large-cache energy (CACTI / ISQED 2023).  C_ref = 1.0 matches the 1 MB Horowitz point as the wire-scale anchor.
    slc_access_ref_mib: float | None = None

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v < 1e6:
                raise ValueError(f"energy.{f.name} must be a finite number in [0, 1e6) or null")
        if self.sram_temp_C is not None and not 0 <= self.sram_temp_C <= SRAM_LEAK_T["t_max_C"]:
            raise ValueError(f"energy.sram_temp_C must be in [0, {SRAM_LEAK_T['t_max_C']:g}] °C or null")
        if self.slc_access_ref_mib is not None and not (0 < self.slc_access_ref_mib < 1e6):
            raise ValueError("energy.slc_access_ref_mib must be in (0, 1e6) MiB or null")

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



# ---------------------------------------------------------------------------------------------------------------- 0.152
# Cited energy presets (Accelergy-style action costs).  Same status labels as budget.DENSITY_PRESETS.
# Tools still ship no process library: presets fill only unset fields; user values (incl. 0) win; idle_W is never
# preset (static power is product-specific).  Insight ΔJ / rank_per_J light up once a preset or a user table is set.
_HOROWITZ = ("Horowitz, 'Computing's Energy Problem (and What We Can Do About It)', ISSCC 2014, Fig. 1.1.9 "
             "(45 nm, 0.9 V); table values as widely cited, e.g. National Academies / MIT 6.S983 lecture notes")
_DRAMSIM3_HBM2 = "DRAMsim3 (umd-memsys/DRAMsim3) configs/HBM2_8Gb_x128.ini"
_TPUV4I = ("Jouppi et al., 'Ten Lessons From Three Generations Shaped Google's TPUv4i', ISCA 2021, Table 2 / §1: "
           "45 nm → 7 nm average energy gain 2.6×, SRAM access 1.3–2.4×, DRAM access 6.3× (HBM packaging); "
           "'SRAM is 20× more energy efficient than DRAM'")
_UCIE = ("UCIe Consortium Hot Chips 2023 tutorial / UCIe 2.0 overview: Advanced-package target 0.25 pJ/bit, "
         "Standard-package target 0.5 pJ/bit (already in core/d2d_catalog.py)")
_NVL = ("NVIDIA Grace Hopper Superchip Architecture whitepaper: NVLink-C2C 1.3 pJ/bit "
        "(already in core/d2d_catalog.py as the NVLink-C2C reference)")

_FGDRAM = ("O'Connor, Chatterjee et al. (NVIDIA), 'Fine-Grained DRAM: Energy-Efficient DRAM for Extreme Bandwidth "
           "Systems', MICRO 2017, §1 / Fig. 1b")
_SKH_IEDM = ("SK hynix, 'Advanced Packaging Technologies in Memory Applications for Future Generative AI Era', "
             "IEDM 2023 paper 15.6 (handout)")
_GRACE = ("NVIDIA Grace CPU Superchip product page / Grace Performance Tuning Guide: 'LPDDR5X memory subsystem provides "
          "up to 500 GB/s of bandwidth in only about 16W'")
_PIMAI = "Ortega et al., 'PIM-AI: A Novel Architecture for High-Efficiency LLM Inference', arXiv 2411.17309 (2024), Table 1"

ENERGY_PRESETS: dict[str, dict] = {
    "horowitz_45nm": {
        "label": "45 nm（Horowitz ISSCC 2014）",
        "fields": {
            # 16b FP multiply ≈ bf16 MAC; 16b FP add as a stand-in for mixed vector ops
            "pJ_mac": (1.1, "来源", _HOROWITZ + ": 16-bit FP multiply 1.1 pJ"),
            "pJ_vec": (0.4, "来源", _HOROWITZ + ": 16-bit FP add 0.4 pJ（向量操作代理）"),
            "pJ_bit_sram": (5.0 / 32, "来源", _HOROWITZ + ": 32-bit SRAM cache read 5 pJ → 5/32 pJ/bit"),
            "pJ_bit_slc": (5.0 / 32, "假设", "= SRAM array（同宏类）on " + _HOROWITZ + "; wire vs capacity: set slc_access_ref_mib"),
            "pJ_bit_dram": (640.0 / 32, "来源", _HOROWITZ + ": 32-bit DRAM read 640 pJ → 20 pJ/bit（DDR 类接口时代；含 I/O）"),
            "pJ_bit_d2d": (0.25, "来源", _UCIE + "（Advanced）"),
            "pJ_bit_link": (1.3, "来源", _NVL + "（节点内 scale-up 代理）"),
            "pJ_bit_net": (5.0, "假设", "跨节点网卡 / 光模块量级，无单一公开数字 — 约数倍于 C2C"),
            "pJ_bit_host": (10.0, "假设", "PCIe 类主机链路；Horowitz 文中 DRAM I/O ~20 pJ/bit 量级的一半作代理"),
        },
    },
    "tpuv4i_7nm": {
        "label": "7 nm（TPUv4i Table 2 对 Horowitz 的缩放）",
        "fields": {
            # average logic 2.6×; SRAM mid of 1.3–2.4 = √(1.3·2.4)≈1.77; DRAM 6.3× (HBM)
            "uW_per_mib_sram_leak": (27.5 * 8 * 2**20 / 1e6, "来源",
                                     "3 nm proxy, lower bound — Aoyagi et al., 'A 3-nm FinFET 563-kbit 35.5-Mbit/mm² Dual-Rail SRAM…', IEEE JSSC 2025 "
                                     "(doi 10.1109/JSSC.2025.3639747; VLSI 2025): HD 6T macro standby leakage 27.5 µW/Mbit "
                                     "at 25 °C → 230.7 µW/MiB.  25 °C figure; the same macro measures 73.5 µW at 85 °C vs 15.5 µW at 25 °C "
                                     "(×4.74) — energy.sram_temp_C scales it (0.261)"),
            # 0.216: HBM2 8Gb per-channel IDD values (DRAMsim3 configs/HBM2_8Gb_x128.ini, VDD 1.2 V, 8 ch × 1 GiB stack):
            # standby IDD2N 40 mA × 1.2 V = 48 mW per channel-die; refresh (IDD5AB − IDD3N)·VDD·tRFC/tREFI =
            # 195 mA·1.2 V·260/3900 = 15.6 mW per 8 Gb → 1.95 mW/Gb.  HBM2-era, 1.2 V (HBM3 core 1.1 V): an upper-side proxy
            "mW_hbm_die_standby": (40 * 1.2, "二手", _DRAMSIM3_HBM2 + ": IDD2N 40 mA × VDD 1.2 V per die（HBM2 代理，非 HBM3E 数据手册）"),
            "mW_hbm_Gb_refresh": (195 * 1.2 * 260 / 3900 / 8, "推导", _DRAMSIM3_HBM2 + ": (IDD5AB 250 − IDD3N 55) mA × 1.2 V × tRFC 260 / tREFI 3900 ÷ 8 Gb；按容量线性外推到 24/32 Gb 晶粒「假设」"),
            "pJ_mac": (1.1 / 2.6, "推导", _TPUV4I + "; base 16b FP mult 1.1 pJ / 2.6"),
            "pJ_vec": (0.4 / 2.6, "推导", _TPUV4I + "; base 16b FP add 0.4 pJ / 2.6"),
            "pJ_bit_sram": ((5.0 / 32) / (1.3 * 2.4) ** 0.5, "推导", _TPUV4I + "; SRAM 1.3–2.4× (geom. mean)"),
            "pJ_bit_slc": ((5.0 / 32) / (1.3 * 2.4) ** 0.5, "假设", "= SRAM array after 7 nm scale; wire vs capacity: set slc_access_ref_mib"),
            "pJ_bit_dram": ((640.0 / 32) / 6.3, "推导", _TPUV4I + "; DRAM 6.3× via HBM packaging "
                            "→ ≈ 3.2 pJ/bit（I/O+阵列量级；HBM3 接口 alone 0.29 pJ/bit, Chae et al. JSSC 2023, 仅 PHY）"),
            # 0.187: per-kind DRAM from primary vendor figures (0.186 used PIM-AI's derived profile, HBM 7 / LPDDR 10,
            # which counts both interfaces ×2 and is ≈ 2× these primary numbers; ratio 1.43 vs 1.16 here).  Both are
            # memory-side at peak bandwidth; the SoC-side PHY is not included (HBM3 host PHY ≈ 0.29 pJ/bit, Chae JSSC 2023).
            "pJ_bit_dram_hbm": (4.3 * 0.8, "推导", _SKH_IEDM + ": HBM2E ≈ 4.3 pJ/bit, HBM3E 20 % lower → 3.44 pJ/bit "
                                "(memory side); cross-check " + _FGDRAM + " HBM2 ≈ 3.97"),
            "pJ_bit_dram_lpddr": (16.0 / (500e9 * 8) * 1e12, "推导", _GRACE + ": LPDDR5X subsystem up to 500 GB/s in "
                                  "about 16 W → 4.0 pJ/bit at peak (memory side); 0.186 value 10 (PIM-AI, 二手) dropped"),
            # 0.188: chip-side PHY and static power
            "pJ_bit_phy_hbm": (0.29, "来源", "Chae et al. (Samsung), 'A 4-nm 1.15 TB/s HBM3 Interface…', ISSCC 2023 28.4 / "
                               "JSSC 2023: 0.29 pJ/bit (controller-side interface); cross-check Mehta et al. VLSID 2025 "
                               "HBM3 host PHY 0.5 pJ/bit"),
            "pJ_bit_phy_lpddr": (0.65 * 2, "推导/二手", "LPDDR4X controller PHY 0.65 pJ/b and 'LPDDR5X … halves the energy "
                                 "efficiency compared to LPDDR4X' (SOCC 2025, 'A 4.266 Gbps/pin LPDDR4X PHY … 12nm', "
                                 "citing its refs [5][6]) → ≈ 1.3 pJ/bit for LPDDR5X"),
            "idle_frac_tdp": (71.8 / 700.0, "二手", "measured H100 SXM idle 71.8 W without a CUDA context (121.7 W with "
                              "one) vs 700 W TDP (lo0 tech blog 'Measuring the power of a GPU', citing controlled runs) "
                              "→ 0.103; static floor only — needs the card TDP (tdp_W, a product choice)"),
            "pJ_bit_d2d": (0.25, "来源", _UCIE),
            # 0.228: clocked-but-idle power of the same H100 measurement: 121.7 W (CUDA context, P0 clocks, no work)
            # − 71.8 W (no context, idle clocks) = 49.9 W, spread over 132 SM × 2048 bf16 MAC/clk = 270,336 MAC slots at
            # 1.830 GHz (989.4 TFLOPS dense / (2 × 270,336)) → 0.101 pJ per MAC-slot-cycle.  Upper-side proxy: the Δ also
            # holds the HBM P-state step and a GPU's non-tensor logic per slot.  Cross-checks (not fitted): at full MAC
            # load it is 0.101 / (0.101 + 0.423) ≈ 19 % of MAC+clock power, between a gated 65 nm DNN accelerator's
            # clocking 5.5 % of active power (elastic-clock chain, NSF PAR 10300339) and CPU clock 28 % + latches 46 %
            # ungated (Bose/Martonosi/Brooks, HC17 tutorial) / ≈ ½ of an Intel CPU (Brooks et al., IEEE Micro 2000)
            "pJ_clk_mac_cycle": (49.9 / (989.4e12 / 2) * 1e12, "推导/二手",
                                 "H100 SXM 有 / 无 CUDA 上下文空闲功耗差 121.7 − 71.8 = 49.9 W（lo0 tech blog，同 idle_frac_tdp 来源）"
                                 "÷ 270,336 MAC 槽 ÷ 1.83 GHz → 0.101 pJ/槽·周期；偏上限（含 HBM 档位切换与 GPU 非张量逻辑）"),
            "pJ_bit_link": (1.3, "来源", _NVL),
            "pJ_bit_net": (5.0, "假设", "跨节点网卡 / 光模块量级，无单一公开数字 — 约数倍于 C2C（与 horowitz_45nm 预设相同）"),
            "pJ_bit_host": (10.0, "假设", "PCIe 类主机链路代理（与 horowitz_45nm 预设相同；Horowitz 文中 DRAM I/O ~20 pJ/bit 量级的一半）"),
        },
    },
}


def apply_energy_preset(d: dict) -> dict:
    """Body ``energy`` with ``energy_preset``: fill every energy field the user left unset from the preset.
    ``idle_W`` / ``idle_W_prefill`` are never filled (static power is a product choice, not a literature constant)."""
    d = dict(d)
    pid = d.pop("energy_preset", None)
    if pid is None:
        return d
    if pid not in ENERGY_PRESETS:
        raise ValueError(f"energy.energy_preset must be one of {sorted(ENERGY_PRESETS)}")
    for k, (v, _, _) in ENERGY_PRESETS[pid]["fields"].items():
        if d.get(k) is None:
            d[k] = v
    return d

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
    mb_fill = seqs / (max(1, r.microbatches) * b_mb) if b_mb else 1.0
    if getattr(lay, "cp", 1) > 1:      # 0.88: CP ranks are attention replicas, each with a chunk of every sequence
        from .parallel import Layout as _L
        b_mb *= lay.cp
        lay = _L(lay.pp, lay.tp, lay.dp * lay.cp, lay.ep, lay.etp, 1)
    b_rank = -(-b_mb // lay.dp)
    dp_act = min(lay.dp, -(-b_mb // b_rank)) if b_rank else lay.dp
    # 0.63: with expert parallelism a DP rank without sequences still runs its experts.  0.64: it moves only its
    # routed-expert weights (no attention / dense / KV / activation bytes) — DRAM / SLC bytes of the idle ranks are
    # the stage's expert reads (StageResult.dram["exp_dram"]), not the busy rank's whole step
    idle = (lay.dp - dp_act) * lay.tp * lay.sp if lay.ep > 1 else 0
    if lay.ep > 1:
        dp_act = lay.dp
    ranks = lay.tp * lay.sp * dp_act
    passes = r.microbatches * (w.steps if w is not None and w.kind == "gen" else 1)
    c = dict.fromkeys(ACTIONS, 0.0)
    # 0.258: CFG negative branch at its own shape — the conditional and the negative-branch forwards each run once
    # per step (micro-batches = the 2 serial calls, B sequences each)
    sets = ([(st, w.steps) for st in r.stages] + [(st, w.steps) for st in r.stages_u]
            if getattr(r, "cfg_asym", None) else [(st, passes) for st in r.stages])
    for st, passes in sets:
        t = st.time
        n = passes * ranks
        nw = passes * lay.tp * lay.sp * lay.dp * mb_fill
        c["mac"] += st.ideal_w * ch.macs * f * nw
        c["vec"] += st.vec_w * ch.lanes * f * nw
        if idle:
            # 0.65: an idle EP rank's SRAM-port bytes and weight dequant are its expert GEMMs', its collective bytes
            # the EP dispatch / combine (+ expert all-reduce) share of the stage's (same tier split 「近似」); 0.64
            # charged it the busy rank's whole step for these
            nb, ni = passes * (ranks - idle), passes * idle
            ea = st.exp_acts or {}
            fl = ea.get("link_frac", 1.0)
            c["vec"] += st.conv_w * ch.lanes * f * nb + ea.get("conv_w", st.conv_w) * ch.lanes * f * ni
            c["sram"] += st.sram_bytes * nb + ea.get("sram", st.sram_bytes) * ni
            c["dram"] += t.dram_bytes * nb + st.dram.get("exp_dram", t.dram_bytes) * ni
            c["slc"] += t.slc_bytes * nb + st.dram.get("exp_slc", t.slc_bytes) * ni
            c["d2d"] += t.wire_d2d * (nb + fl * ni)       # 0.78: bytes on each tier's wire (per algorithm)
            c["net"] += t.wire_net * (nb + fl * ni)
            c["link"] += t.wire_link * (nb + fl * ni)
        else:
            c["vec"] += st.conv_w * ch.lanes * f * n   # weight dequant: per pass
            c["sram"] += st.sram_bytes * n
            c["dram"] += t.dram_bytes * n
            c["slc"] += t.slc_bytes * n
            c["d2d"] += t.wire_d2d * n
            c["net"] += t.wire_net * n
            c["link"] += t.wire_link * n
    dr = getattr(r, "draft", None) or {}
    for k, v in dr.get("acts", {}).items():           # 0.223: independent draft model's k passes of the step
        c[k] = c.get(k, 0.0) + v
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


def _is_hbm(r) -> bool | None:
    from .. import mem_catalog
    try:
        return "HBM" in mem_catalog.parse_mem_id(r.scenario.mem_id).mem_type.upper()
    except Exception:                       # noqa: BLE001
        return None


def phy_pJ(r, table: EnergyTable):
    """0.188: (field, value) of the chip-side DRAM PHY energy for this result's memory kind, or (None, None)."""
    h = _is_hbm(r)
    if h is None:
        return None, None
    f = "pJ_bit_phy_hbm" if h else "pJ_bit_phy_lpddr"
    v = getattr(table, f)
    return (f, v) if v is not None else (None, None)


def idle_W_of(table: EnergyTable):
    """0.188: explicit idle_W wins; else idle_frac_tdp × tdp_W when both are set; else None."""
    if table.idle_W is not None:
        return table.idle_W
    if table.idle_frac_tdp is not None and table.tdp_W is not None:
        return table.idle_frac_tdp * table.tdp_W
    return None


def dram_pJ_field(r, table: EnergyTable) -> str:
    """0.186: which field prices this result's DRAM bits: the per-kind field when set, else pJ_bit_dram."""
    from .. import mem_catalog
    try:
        hbm = "HBM" in mem_catalog.parse_mem_id(r.scenario.mem_id).mem_type.upper()
    except Exception:                       # noqa: BLE001 — unknown id: generic field
        return "pJ_bit_dram"
    f = "pJ_bit_dram_hbm" if hbm else "pJ_bit_dram_lpddr"
    return f if getattr(table, f) is not None else "pJ_bit_dram"



def clock_util(chip, mac: float, vec: float, sram_B: float, card_s: float) -> dict:
    """0.265: utilisation(s) for the gated share of P_clk.

    u_mac = useful MAC slot-cycles / (macs·f·card_s); u_vec = vec element-ops / (lanes·f·card_s);
    u_sram = SRAM-port bytes / (port_Bpc·f·card_s).  Default (clk_datapath_gate False): u = u_mac (0.229).
    With clk_datapath_gate: u = mean(u_mac, u_vec, u_sram) 「假设」 equal share of the *gated* clock-tree
    portion across the three datapath domains (closes §176 limit).
    """
    f = chip.freq_ghz * 1e9
    c_mac = chip.macs * f * card_s
    c_vec = chip.lanes * f * card_s
    c_sram = chip.port_Bpc * f * card_s
    u_mac = min(1.0, mac / c_mac) if c_mac > 0 else 0.0
    u_vec = min(1.0, vec / c_vec) if c_vec > 0 else 0.0
    u_sram = min(1.0, sram_B / c_sram) if c_sram > 0 else 0.0
    datapath = bool(getattr(chip, "clk_datapath_gate", False))
    u = (u_mac + u_vec + u_sram) / 3.0 if datapath else u_mac
    return {"mac_util": u_mac, "vec_util": u_vec, "sram_port_util": u_sram, "util": u,
            "datapath_gate": datapath, "basis": ("mean(mac,vec,sram_port)" if datapath else "mac")}



def slc_pJ_bit(table: EnergyTable, slc_mib: float) -> float | None:
    """0.266: effective pJ/bit for SLC traffic.  Flat ``pJ_bit_slc`` unless ``slc_access_ref_mib`` is set, then
    × max(1, √(slc_mib / ref)) — Horowitz ISSCC 2014 √C fit; H-tree dominated large caches (CACTI)."""
    e = table.pJ_bit_slc
    if e is None:
        return None
    ref = table.slc_access_ref_mib
    if ref is None or not (slc_mib > 0):
        return e
    return e * max(1.0, math.sqrt(float(slc_mib) / float(ref)))


def energy_report(r, table: EnergyTable | None = None, power_cap: bool = False) -> dict:
    """0.189 power_cap (opt-in): when average card power (dynamic + static) exceeds table.tdp_W, sustained throughput
    is cut to fit.  Energy per action held constant (no DVFS V² saving, 「假设」, conservative), so the window stretches
    to t' = E_dyn / (TDP − P_static), i.e. ×(P_dyn / (TDP − P_static)); static J grows by the same factor."""
    """Action counts per output unit; energy (J / unit, by action) only for the entries the user supplied."""
    table = table or EnergyTable()
    a = action_counts(r)
    u = a["units"] or 1.0
    per = {k: v / u for k, v in a["counts"].items() if k != "idle"}
    per["idle_card_s"] = r.scenario.layout.cards * a["window_s"] / u
    # 0.228: MAC-slot · core cycles per unit (the per-cycle term's count): every card clocked for the whole window
    per["clock"] = per["idle_card_s"] * r.scenario.chip.macs * r.scenario.chip.freq_ghz * 1e9
    host_b = (getattr(r, "pipeline", None) or {}).get("load_bytes", 0.0)
    if host_b:              # 0.62: video sequential offload -- weights re-read over the host link every window
        per["host"] = host_b / u
    dfield = dram_pJ_field(r, table)
    out = {"unit": a["unit"], "window_s": a["window_s"], "units_per_window": a["units"],
           "counts_per_unit": per, "provided": table.provided,
           "missing": [f.name for f in fields(EnergyTable) if getattr(table, f.name) is None
                       and f.name not in ("idle_W_prefill", "pJ_bit_host", "pJ_bit_dram_hbm", "pJ_bit_dram_lpddr",
                                          "pJ_bit_phy_hbm", "pJ_bit_phy_lpddr", "idle_frac_tdp", "tdp_W",
                                          "uW_per_mib_sram_leak", "mW_hbm_die_standby", "mW_hbm_Gb_refresh",
                                          "pJ_clk_mac_cycle", "sram_temp_C", "slc_access_ref_mib")
                       and not (f.name == "pJ_bit_dram" and dfield != "pJ_bit_dram")
                       and not (f.name == "idle_W" and (idle_W_of(table) is not None or table.idle_frac_tdp is not None))]
                      + (["tdp_W"] if table.idle_W is None and table.idle_frac_tdp is not None and table.tdp_W is None
                         else []),
           "dram_pJ_field": dfield,
           "basis": "动作计数 × 用户提供的每动作能耗（Accelergy 式 ERT）；工具不内置任何能耗数值"}
    if not table.provided:
        return out
    j = {}
    slc_mib = float(getattr(r.scenario.chip, "slc_mib", 0.0) or 0.0)
    for k, attr in _UNIT_PJ.items():
        if k == "slc":
            e = slc_pJ_bit(table, slc_mib)
            if e is not None and table.slc_access_ref_mib is not None and slc_mib > 0:
                out["slc_access"] = {"pJ_bit": e, "pJ_bit_base": table.pJ_bit_slc,
                                     "ref_mib": table.slc_access_ref_mib, "slc_mib": slc_mib,
                                     "scale": e / table.pJ_bit_slc if table.pJ_bit_slc else None,
                                     "basis": "pJ_bit_slc × max(1, √(slc_mib/ref)) (Horowitz √C, 0.266)"}
        else:
            e = getattr(table, dfield if k == "dram" else attr)
        if e is not None:
            j[k] = per[k] * e * 1e-12 * (8 if k in _BITS else 1)
    pf, pv = phy_pJ(r, table)
    if pv is not None:
        j["dram_phy"] = per["dram"] * pv * 1e-12 * 8
        out["phy_pJ_field"] = pf
    if table.pJ_clk_mac_cycle is not None:      # 0.228: per-cycle dynamic (clock tree / always-clocked registers)
        # 0.229/0.265: ungated share g always; gated share tracks MAC util, or mean(mac,vec,sram_port) when
        # chip.clk_datapath_gate (closes §176 "vector/SRAM ports not gated separately")
        ch = r.scenario.chip
        g = getattr(ch, "clk_ungated_share", 1.0)
        cu = clock_util(ch, per["mac"], per.get("vec", 0.0), per.get("sram", 0.0), per["idle_card_s"])
        util = cu["util"]
        k = g + (1.0 - g) * util
        j["clock"] = per["clock"] * table.pJ_clk_mac_cycle * 1e-12 * k
        out["clock_W_card"] = ch.macs * ch.freq_ghz * 1e9 * table.pJ_clk_mac_cycle * 1e-12 * k
        out["clock_gating"] = {"ungated_share": g, "mac_util": cu["mac_util"], "vec_util": cu["vec_util"],
                               "sram_port_util": cu["sram_port_util"], "util": util, "factor": k,
                               "datapath_gate": cu["datapath_gate"], "basis": cu["basis"]}
    iw = idle_W_of(table)
    if iw is not None:
        j["idle"] = iw * per["idle_card_s"]
        out["idle_W_used"] = iw
        out["idle_source"] = "idle_W" if table.idle_W is not None else "idle_frac_tdp × tdp_W"
        # 0.190: preset match within 0.5 % (the UI rounds the filled value to 4 digits)
        if table.idle_W is None:      # 0.189: say where the fraction comes from — a GPU number applied to an NPU
            pre = ENERGY_PRESETS["tpuv4i_7nm"]["fields"]["idle_frac_tdp"]
            out["idle_source_note"] = (f"idle_frac_tdp {table.idle_frac_tdp:g} × TDP {table.tdp_W:g} W；"
                                       + (f"「{pre[1]}」{pre[2]} — GPU 整卡空闲值套用于本 NPU，不是本芯片的泄漏测量"
                                          if abs(table.idle_frac_tdp - pre[0]) <= 5e-3 * pre[0] else "用户填写"))
    if table.uW_per_mib_sram_leak is not None:      # 0.212: SRAM leakage (static), see the field note
        ch = r.scenario.chip
        slc = getattr(ch, "slc_mib", 0.0) or 0.0     # 0.260: on-die SRAM (same macro class as pJ_bit_slc 「假设」)
        # 0.263: SLC leakage is always charged when the field is set — a measured idle is of a card without this
        # (speculative) SLC, or treats the SLC as a separate macro 「假设」. Shared sram still suppressed by idle.
        mib = (getattr(ch, "state_sram_mib", 0.0) + slc
               + (ch.sram_mib if iw is None else 0.0))
        if mib > 0:
            leak_W = mib * sram_leak_uW_per_mib(table) * 1e-6
            j["sram_leak"] = leak_W * per["idle_card_s"]
            out["sram_leak_W_card"] = leak_W
            # 0.261: temperature used and the hot-die band (85 °C, same macro) next to the charged value
            out["sram_leak_temp_C"] = table.sram_temp_C if table.sram_temp_C is not None else SRAM_LEAK_T["t_ref_C"]
            out["sram_leak_W_card_85C"] = mib * table.uW_per_mib_sram_leak * SRAM_LEAK_T["ratio_hot"] * 1e-6
            if iw is None:
                out["sram_leak_scope"] = "state_sram + sram" + (" + slc" if slc > 0 else "")
            else:
                out["sram_leak_scope"] = ("state_sram + slc（共享 SRAM 泄漏已含于 idle；SLC 按增量计「假设」)"
                                          if slc > 0 else "state_sram（共享 SRAM 泄漏已含于 idle 数值）")
    if iw is None and (table.mW_hbm_die_standby is not None or table.mW_hbm_Gb_refresh is not None):
        hs = hbm_static_W(r.scenario.mem_id, table)      # 0.216: HBM standby + refresh (static)
        if hs:
            j["hbm_static"] = hs["W"] * per["idle_card_s"]
            out["hbm_static_W_card"] = hs["W"]
            out["hbm_static"] = hs
    if per.get("host"):
        if table.pJ_bit_host is not None:
            j["host"] = per["host"] * table.pJ_bit_host * 1e-12 * 8
        else:
            out["host_note"] = "卸载放置每次从主机重载权重的字节未计能耗：未填 pJ_bit_host"
    if not j:      # 0.131: only entries this report does not use (idle_W_prefill, pJ_bit_host without host traffic)
        out["note"] = "所填能耗项不适用于本报告（idle_W_prefill 只用于 PD prefill 池；pJ_bit_host 只用于主机链路流量），未计能耗"
        return out      # was J_per_unit = 0 and avg_W_per_card = 0 — read as "zero energy"
    tot = sum(j.values())
    out.update(J_per_unit=tot, J_by_action=j,
               avg_W_per_card=tot * u / a["window_s"] / r.scenario.layout.cards if a["window_s"] > 0 else 0.0)
    stat = j.get("idle", 0.0) + j.get("sram_leak", 0.0) + j.get("hbm_static", 0.0)    # 0.212/0.216: static too
    if ("idle" in j or "sram_leak" in j or "hbm_static" in j) and tot > 0:       # 0.189: static share + dominant-term warning
        out["static_share"] = stat / tot
        out["J_dynamic_per_unit"] = tot - stat
        if out["static_share"] > STATIC_WARN:
            out["static_warning"] = (f"静态能耗占 J/{a['unit']} 的 {out['static_share']:.1%}（> {STATIC_WARN:.0%}）："
                                     "结论主要由空闲功耗假设决定，而非动态动作")
    if power_cap:
        out["power_cap"] = _power_cap(out, j, iw, table, r.scenario.layout.cards, u, a["window_s"])
        if not out["power_cap"].get("error") and out["power_cap"].get("capped"):
            # 0.227: the clock / voltage the cap implies (core/powercap.py) — dfs (headline, ≡ the duty-cycle throughput)
            # and a measured V–f curve (voltage benefit, 「二手」)
            from .powercap import CAP_CURVES, dvfs_cap
            out["power_cap"]["dvfs"] = {c: dvfs_cap(r, table, table.tdp_W, c) for c in CAP_CURVES}
    return out


STATIC_WARN = 0.5


def _power_cap(out, j, iw, table, cards, u, win) -> dict:
    """0.189: TDP-capped sustained throughput (opt-in).  Needs tdp_W.  See EnergyTable.power_cap."""
    if table.tdp_W is None or win <= 0:
        return {"error": "power_cap 需要 tdp_W（每卡 TDP，产品决策，无默认值）"}
    # 0.227 fix: SRAM leakage (0.212) and HBM standby / refresh (0.216) are static too — they were counted as
    # dynamic energy that shrinks with the slowdown (under-slowed the capped point)
    st_keys = ("idle", "sram_leak", "hbm_static")
    p_stat = (iw or 0.0) + (out.get("sram_leak_W_card") or 0.0) + (out.get("hbm_static_W_card") or 0.0)
    e_dyn_card = sum(v for k, v in j.items() if k not in st_keys) * u / cards        # J per card per window
    p_dyn = e_dyn_card / win
    need = p_dyn + p_stat
    if p_stat >= table.tdp_W:
        return {"P_needed_W": need, "tdp_W": table.tdp_W, "feasible": False,
                "error": "静态功耗 ≥ TDP：任何吞吐都超功耗"}
    slow = max(1.0, p_dyn / (table.tdp_W - p_stat))
    J = out["J_per_unit"] + (sum(j.get(k, 0.0) for k in st_keys) * (slow - 1.0))
    return {"P_needed_W": need, "P_dyn_W": p_dyn, "P_static_W": p_stat, "tdp_W": table.tdp_W, "feasible": True,
            "capped": slow > 1.0, "slowdown": slow, "throughput_factor": 1.0 / slow, "J_per_unit_capped": J,
            "basis": "每动作能耗不随频率变（不计 DVFS 的 V² 收益，「假设」，偏保守）；超 TDP 时窗口拉长到 "
                     "E_dyn / (TDP − P_static)"}


def hbm_static_W(mem_id: str | None, table: "EnergyTable") -> dict | None:
    """0.216: HBM background + refresh power of the card's stacks: stacks × height × standby/die + total Gb × refresh/Gb."""
    if not mem_id:
        return None
    from .. import mem_catalog
    try:
        sp = mem_catalog.parse_mem_id(mem_id)
    except Exception:
        return None
    if getattr(sp, "kind", None) != "HBM" or not getattr(sp, "hbm_height", None):
        return None
    dies = sp.n_units * sp.hbm_height
    stb = dies * (table.mW_hbm_die_standby or 0.0) * 1e-3
    ref = dies * sp.hbm_die_Gb * (table.mW_hbm_Gb_refresh or 0.0) * 1e-3
    return {"W": stb + ref, "standby_W": stb, "refresh_W": ref, "stacks": sp.n_units, "dies": dies}

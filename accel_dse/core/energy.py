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

# 0.299: + noc (NoC bytes × router traversals), noc_wire (NoC bytes × link length in engine pitches)
ACTIONS = ("mac", "vec", "sram", "slc", "dram", "d2d", "link", "net", "idle", "noc", "noc_wire")
_UNIT_PJ = {"mac": "pJ_mac", "vec": "pJ_vec", "sram": "pJ_bit_sram", "slc": "pJ_bit_slc", "dram": "pJ_bit_dram",
            "d2d": "pJ_bit_d2d", "link": "pJ_bit_link", "net": "pJ_bit_net",
            "noc": "pJ_bit_hop_noc", "noc_wire": "pJ_bit_mm_onchip"}     # 0.299 (noc_wire: × engine pitch, see unit_pJ)
_BITS = ("sram", "slc", "dram", "d2d", "link", "net", "noc", "noc_wire")


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
    # 0.302: per-generation DRAM pJ/bit (unset → the per-kind field above): HBM4 / HBM4E stacks, LPDDR6 packages.
    pJ_bit_dram_hbm4: float | None = None
    pJ_bit_dram_lpddr6: float | None = None
    # 0.302: LPDDR background (precharge standby IDD2N, all rails) per x16 channel of 8 Gb, and refresh per Gb.  Charged
    # like the HBM terms (only when no idle figure is set); counted on every 8 Gb channel slice of the card's capacity.
    mW_lpddr_ch_standby: float | None = None
    mW_lpddr_Gb_refresh: float | None = None
    # 0.303: LPDDR low-power residency (opt-in; preset-filled).  Per 8 Gb channel slice: power-down (IDD2P, all rails)
    # and self-refresh (power-down self refresh, all rails, refresh included); the minimum idle gap that pays for each
    # state (exit latency + minimum residency, ns).  With them the background is the residency-weighted mix of
    # standby / power-down / self-refresh at the modelled DRAM utilisation instead of standby everywhere (§248).
    mW_lpddr_ch_pd: float | None = None
    mW_lpddr_ch_sr: float | None = None
    ns_lpddr_pd_gap: float | None = None
    ns_lpddr_sr_gap: float | None = None
    # 0.228: per-cycle dynamic energy per MAC slot (pJ per MAC-slot per core cycle) — clock tree / distribution,
    # always-clocked pipeline registers and control: paid every core cycle the card is clocked, busy or stalled, so
    # P_clk = macs × f × pJ (scales with f and, under DVFS, with V²).  The only dynamic power that falls with the clock
    # when the op rate is set by memory.  Distinct from idle_W (a measured *idle-clock* floor) — not double counted.
    pJ_clk_mac_cycle: float | None = None
    # 0.261: die (junction) temperature for SRAM / SLC leakage, °C.  uW_per_mib_sram_leak is a 25 °C figure; the charged
    # value is × leak_temp_factor(T) (same macro measured at 25 and 85 °C, see SRAM_LEAK_T).  None = 25 °C (lower bound).
    sram_temp_C: float | None = None
    # 0.305: LPDDR case temperature for self-refresh / refresh power (°C, 0–95); None → sram_temp_C (DRAM case ≈ die
    # junction 「假设」, conservative for a discrete package) → None (≤ 0.304: the 85 °C maximum, 1× refresh)
    dram_temp_C: float | None = None
    # 0.266 (DEPRECATED 0.268 — use pJ_bit_mm_onchip): capacity-dependent SLC access energy.  None = flat pJ_bit_slc.  When set to a
    # reference capacity C_ref (MiB), effective pJ/bit = pJ_bit_slc · max(1, √(slc_mib / C_ref)).  Horowitz ISSCC 2014
    # Fig. 1.1.9: 8 KB / 32 KB / 1 MB SRAM accesses 10 / 20 / 100 pJ (64-bit) ⇒ E ∝ √C 「推导」; H-tree dominates
    # large-cache energy (CACTI / ISQED 2023).  C_ref = 1.0 matches the 1 MB Horowitz point as the wire-scale anchor.
    slc_access_ref_mib: float | None = None
    # 0.267: distance-based SLC wire energy (mechanism; replaces the √C heuristic when set).  Effective pJ/bit =
    # pJ_bit_slc + pJ_bit_mm_onchip · d, d = mean Manhattan distance from the die-centre port to a uniform point of
    # the square SLC array = ½·√(slc_mib · slc_mm2_per_mib).  Keckler et al. IEEE Micro 2011 Table: 256 bits over 10 mm
    # = 310 pJ at 40 nm (0.121 pJ/bit/mm), 200 pJ projected at 10 nm / 0.75 V (0.078 pJ/bit/mm) 「来源/推导」.
    pJ_bit_mm_onchip: float | None = None
    # SLC array density for the distance (mm² / MiB).  None = SLC_MM2_PER_MIB_DEFAULT (Aoyagi 3 nm macro, array only).
    slc_mm2_per_mib: float | None = None
    # 0.299: NoC router traversal energy per bit per hop (opt-in: None = no NoC term, ≤ 0.298).  Charged on the
    # dataflow's NoC traffic (mapping.noc_multicast: operand multicast across engines, cross-group operand copies,
    # split-K partial-sum reduction), plus the link wire pJ_bit_mm_onchip × engine pitch per traversal.
    # Reference: NOC_HOP_PJ_BIT_45NM (Park et al. DAC 2012, 45 nm) / 2.6 at 7 nm (NOC_HOP_PJ_BIT_7NM) 「推导」.
    pJ_bit_hop_noc: float | None = None
    # 0.299: MAC-array area density for the engine pitch (mm² per 1024 MACs).  None = NOC_MM2_PER_KMAC_DEFAULT.
    noc_mm2_per_kmac: float | None = None

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v < 1e6:
                raise ValueError(f"energy.{f.name} must be a finite number in [0, 1e6) or null")
        if self.sram_temp_C is not None and not 0 <= self.sram_temp_C <= SRAM_LEAK_T["t_max_C"]:
            raise ValueError(f"energy.sram_temp_C must be in [0, {SRAM_LEAK_T['t_max_C']:g}] °C or null")
        if self.dram_temp_C is not None and not 0 <= self.dram_temp_C <= LPDDR_T_MAX_C:
            raise ValueError(f"energy.dram_temp_C must be in [0, {LPDDR_T_MAX_C:g}] °C or null")
        if self.slc_access_ref_mib is not None and not (0 < self.slc_access_ref_mib < 1e6):
            raise ValueError("energy.slc_access_ref_mib must be in (0, 1e6) MiB or null")
        if self.noc_mm2_per_kmac is not None and not self.noc_mm2_per_kmac > 0:
            raise ValueError("energy.noc_mm2_per_kmac must be > 0 or null")
        if self.slc_mm2_per_mib is not None and not self.slc_mm2_per_mib > 0:
            raise ValueError("energy.slc_mm2_per_mib must be > 0 or null")
        if self.pJ_bit_mm_onchip is not None and self.slc_access_ref_mib is not None:
            raise ValueError("energy: set either pJ_bit_mm_onchip (distance model) or slc_access_ref_mib (√C), not both")

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

# 0.298: accumulation precision of bf16 matrix MACs (why pJ_mac = 16b mult + 32b add)
_FP32_ACC = ("Google Cloud TPU docs 'TPU architecture': 'All multiplies take bfloat16 inputs, but all accumulations are "
             "performed in FP32 number format'; Google Cloud blog 'BFloat16: the secret to high performance on Cloud TPUs' "
             "(2019): 'bfloat16 for the multiplication and 32-bit IEEE floating point for accumulation'")
# 0.298: ≤ 0.297 preset pJ_mac (multiply only) — kept for reproducibility / tests
PJ_MAC_MULT_ONLY = {"horowitz_45nm": 1.1, "tpuv4i_7nm": 1.1 / 2.6}

ENERGY_PRESETS: dict[str, dict] = {
    "horowitz_45nm": {
        "label": "45 nm（Horowitz ISSCC 2014）",
        "fields": {
            # 0.298: a bf16 MAC = 16b FP multiply + 32b FP accumulate (MXU / tensor-core MACs accumulate in FP32,
            # _FP32_ACC); ≤ 0.297 priced the multiply only (1.1).  16b FP add as a stand-in for mixed vector ops
            "pJ_mac": (1.1 + 0.9, "推导", _HOROWITZ + ": 16-bit FP multiply 1.1 + 32-bit FP add 0.9 pJ（FP32 累加，"
                       + _FP32_ACC + "）"),
            "pJ_vec": (0.4, "来源", _HOROWITZ + ": 16-bit FP add 0.4 pJ（向量操作代理）"),
            "pJ_bit_sram": (5.0 / 32, "来源", _HOROWITZ + ": 32-bit SRAM cache read 5 pJ → 5/32 pJ/bit"),
            "pJ_bit_slc": (5.0 / 32, "假设", "= SRAM array（同宏类）on " + _HOROWITZ + "；容量相关的导线项由 pJ_bit_mm_onchip（预设已填，0.300）加上"),
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
            "pJ_mac": ((1.1 + 0.9) / 2.6, "推导", _TPUV4I + "; base 16b FP mult 1.1 + 32b FP add 0.9 pJ（FP32 累加，"
                       + _FP32_ACC + "）/ 2.6"),
            "pJ_vec": (0.4 / 2.6, "推导", _TPUV4I + "; base 16b FP add 0.4 pJ / 2.6"),
            "pJ_bit_sram": ((5.0 / 32) / (1.3 * 2.4) ** 0.5, "推导", _TPUV4I + "; SRAM 1.3–2.4× (geom. mean)"),
            "pJ_bit_slc": ((5.0 / 32) / (1.3 * 2.4) ** 0.5, "假设", "= SRAM array after 7 nm scale；容量相关的导线项由 pJ_bit_mm_onchip（预设已填，0.300）加上"),
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
        if k == "pJ_bit_mm_onchip" and d.get("slc_access_ref_mib") is not None:
            continue                  # 0.300: the user's √C SLC model excludes the distance model (EnergyTable check)
        if d.get(k) is None:
            d[k] = v
    return d

def action_counts(r) -> dict:
    """(counts per window, window s, output units per window, unit label) for one evaluated result."""
    sv = r.scenario.serving
    if sv.moe_skew != 1.0 or sv.moe_expert_load or getattr(sv, "moe_expert_load_layers", ()):
        from dataclasses import replace
        from .evaluate import evaluate
        # 0.293: per-layer FFN loads are the same EP skew input (value-expert loads are not: no EP, kept)
        bal = action_counts(evaluate(replace(r.scenario, serving=replace(sv, moe_skew=1.0, moe_expert_load=(),
                                                                          moe_expert_load_layers=()))))
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
            c["noc"] += st.noc_hops * nb + ea.get("noc", st.noc_hops) * ni          # 0.299
            c["noc_wire"] += st.noc_pitch * nb + ea.get("noc_wire", st.noc_pitch) * ni
            c["dram"] += t.dram_bytes * nb + st.dram.get("exp_dram", t.dram_bytes) * ni
            c["slc"] += t.slc_bytes * nb + st.dram.get("exp_slc", t.slc_bytes) * ni
            c["d2d"] += t.wire_d2d * (nb + fl * ni)       # 0.78: bytes on each tier's wire (per algorithm)
            c["net"] += t.wire_net * (nb + fl * ni)
            c["link"] += t.wire_link * (nb + fl * ni)
        else:
            c["vec"] += st.conv_w * ch.lanes * f * n   # weight dequant: per pass
            c["sram"] += st.sram_bytes * n
            c["noc"] += st.noc_hops * n            # 0.299: NoC byte·hops / byte·pitches (energy only)
            c["noc_wire"] += st.noc_pitch * n
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
    # 0.306: DRAM bytes cross the NoC from the memory columns to the engines (noc_mem_attach "edge"): router
    # traversals / link pitches per byte from Chip.noc_mem_hops, booked on the NoC actions (energy only)
    nd = None
    if c["dram"] > 0:
        h, p = ch.noc_mem_hops()
        if h or p:
            nd = {"byte_hops": c["dram"] * h, "byte_pitches": c["dram"] * p, "hops_per_byte": h, "pitches_per_byte": p}
            c["noc"] += nd["byte_hops"]
            c["noc_wire"] += nd["byte_pitches"]
    units = r.throughput * window
    out = {"counts": c, "window_s": window, "units": units, "unit": unit}
    if nd:
        out["noc_dram"] = nd
    return out


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
        t = mem_catalog.parse_mem_id(r.scenario.mem_id).mem_type.upper()
    except Exception:                       # noqa: BLE001 — unknown id: generic field
        return "pJ_bit_dram"
    hbm = "HBM" in t
    gen = "pJ_bit_dram_hbm4" if t in ("HBM4", "HBM4E") else ("pJ_bit_dram_lpddr6" if t == "LPDDR6" else None)
    if gen is not None and getattr(table, gen) is not None:      # 0.302: per-generation figure
        return gen
    f = "pJ_bit_dram_hbm" if hbm else "pJ_bit_dram_lpddr"
    return f if getattr(table, f) is not None else "pJ_bit_dram"


def dram_util(r, per: dict) -> float | None:
    """0.303: DRAM busy fraction of a card = modelled DRAM bytes per card-second ÷ the memory's effective bandwidth."""
    from .. import mem_catalog
    try:
        sp = mem_catalog.parse_mem_id(r.scenario.mem_id)
    except Exception:                       # noqa: BLE001
        return None
    bw = getattr(sp, "effective_GBps", 0.0) * 1e9
    cs = per.get("idle_card_s") or 0.0
    if bw <= 0 or cs <= 0:
        return None
    return min(1.0, per.get("dram", 0.0) / cs / bw)


def dram_static_set(table) -> bool:
    """0.302: any DRAM background / refresh field set (HBM 0.216, LPDDR 0.302)."""
    return any(getattr(table, f) is not None for f in ("mW_hbm_die_standby", "mW_hbm_Gb_refresh",
                                                      "mW_lpddr_ch_standby", "mW_lpddr_Gb_refresh"))



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



# 0.267: Aoyagi et al. (TSMC) VLSI 2025 3 nm HD macro 35.5 Mbit/mm² → 8.388608 / 35.5 mm² per MiB (array only, so the
# distance is a lower bound; periphery / banking overhead makes a real SLC larger) 「来源」
SLC_MM2_PER_MIB_DEFAULT = 8.388608 / 35.5
RETICLE_MM2 = 858.0
# 0.268: the 0.266 √C option is kept for reproducibility but deprecated (MODEL §213.1)
SLC_SQRTC_DEPRECATION = ("slc_access_ref_mib（√C）已弃用：把 8 KB–1 MB 缓存数据外推到 GiB 级 SLC，过于悲观（MODEL §212.2）；"
                         "请改用 pJ_bit_mm_onchip（距离模型，Keckler 2011）")


def slc_wire_mm(table: EnergyTable, slc_mib: float) -> float:
    """0.267: mean Manhattan distance (mm) from a centre port to a uniform point of a square SLC array: L/2."""
    dens = table.slc_mm2_per_mib if table.slc_mm2_per_mib is not None else SLC_MM2_PER_MIB_DEFAULT
    return 0.5 * math.sqrt(max(0.0, float(slc_mib)) * dens)


def slc_pJ_bit(table: EnergyTable, slc_mib: float) -> float | None:
    """0.266: effective pJ/bit for SLC traffic.  Flat ``pJ_bit_slc`` unless ``slc_access_ref_mib`` is set, then
    × max(1, √(slc_mib / ref)) — Horowitz ISSCC 2014 √C fit; H-tree dominated large caches (CACTI).
    0.267: with ``pJ_bit_mm_onchip`` → pJ_bit_slc + pJ_bit_mm_onchip · slc_wire_mm (distance model)."""
    e = table.pJ_bit_slc
    if e is None:
        return None
    if table.pJ_bit_mm_onchip is not None:
        return e + table.pJ_bit_mm_onchip * slc_wire_mm(table, slc_mib) if slc_mib > 0 else e
    ref = table.slc_access_ref_mib
    if ref is None or not (slc_mib > 0):
        return e
    return e * max(1.0, math.sqrt(float(slc_mib) / float(ref)))


# 0.298: the on-chip SRAM is banked per engine (each engine streams its own operand partition from its local slice);
# the wire from a uniform point of that slice to the engine's array edge is the same ½·√(area) mean Manhattan distance
# as the 0.267 SLC model, area = (sram_mib / engines) × the same 3 nm HD macro density 「推导/假设」.  Cross-engine
# (NoC) traffic is not in this term.  ≤ 0.297 the SRAM pJ/bit was flat, so SRAM capacity carried no access-energy cost.
SRAM_WIRE_BASIS = ("pJ_bit_sram + pJ_bit_mm_onchip × ½√((sram_mib/engines)·mm²/MiB)（每引擎本地分片，Keckler 2011 线能耗；"
                   "Aoyagi 2025 3 nm 宏密度，阵列面积下限 → 距离下限）")


def sram_slice_mib(chip) -> float:
    """0.298: MiB of the engine-local SRAM slice (chip.sram_mib split evenly over the engines)."""
    mib = float(getattr(chip, "sram_mib", 0.0) or 0.0)
    return mib / max(1, int(getattr(chip, "engines", 1) or 1))


def sram_wire_mm(table: EnergyTable, chip) -> float:
    """0.298: mean distance (mm) from a uniform point of the engine-local SRAM slice to its port (same model as SLC)."""
    return slc_wire_mm(table, sram_slice_mib(chip))


def sram_pJ_bit(table: EnergyTable, chip) -> float | None:
    """0.298: effective pJ/bit of local SRAM port traffic.  Flat ``pJ_bit_sram`` unless ``pJ_bit_mm_onchip`` is set,
    then + pJ_bit_mm_onchip · sram_wire_mm (distance model shared with the SLC)."""
    e = table.pJ_bit_sram
    if e is None or table.pJ_bit_mm_onchip is None or chip is None:
        return e
    return e + table.pJ_bit_mm_onchip * sram_wire_mm(table, chip)


# 0.299: NoC hop energy.  Park et al. (MIT), "Approaching the theoretical limits of a mesh NoC with a 16-node chip
# prototype in 45nm SOI", DAC 2012: 4×4 mesh, 1 mm links, 1.1 V / 1 GHz, 427.3 mW (Table 2) delivering 653 Gb/s of
# broadcast traffic, of which 76.7 mW leakage 「来源」.  A broadcast flit costs k² crossbar + (k² − 1) link traversals
# for k² − 1 deliveries (their Table 1), so a delivered bit ≈ one router + one 1 mm link: (427.3 − 76.7) / 653 =
# 0.537 pJ/bit/hop 「推导」.  Router only = − Keckler 2011 40 nm full-swing wire 0.121 pJ/bit/mm × 1 mm = 0.416 (their
# link is low-swing, cheaper than full swing, so this is a lower bound on the router) 「推导/近似」; 7 nm: / 2.6
# (TPUv4i logic scaling, as pJ_mac) 「推导」.
NOC_HOP_PJ_BIT_45NM = (427.3 - 76.7) / 653.0 - 0.121
NOC_HOP_PJ_BIT_7NM = NOC_HOP_PJ_BIT_45NM / 2.6
NOC_SRC = ("Park et al., 'Approaching the theoretical limits of a mesh NoC with a 16-node chip prototype in 45nm SOI', "
           "DAC 2012: (427.3 mW − 76.7 mW leakage) / 653 Gb/s broadcast = 0.537 pJ/bit per router + 1 mm link; − Keckler "
           "2011 wire 0.121 pJ/bit/mm → router 0.416 pJ/bit (45 nm, lower bound); / 2.6 at 7 nm (TPUv4i) → 0.160")
# MAC-array density for the engine pitch: TPUv4i (Jouppi et al., ISCA 2021): 4 × 128×128 bf16 MXUs = 11 % of a
# < 400 mm² die → 0.6875 mm² per 1024 MACs (upper bound; same value as budget DENSITY_PRESETS n7_tpuv4i) 「来源」
NOC_MM2_PER_KMAC_DEFAULT = 400 * 0.11 / 64
# 0.300: the presets carry the on-chip wire (Keckler 2011) and the NoC router (Park 2012), so a preset's J/token
# includes the SRAM / SLC distance term (0.298), the NoC traffic and its wires (0.299) by default.  ≤ 0.299 both were
# unset in every preset (opt-in): the preset J/token omitted every on-chip-movement cost, which biases design
# comparisons toward many small engines and a big flat SRAM (the cost the design change adds is the one left out).
# A user value (incl. 0) still wins; a user √C (slc_access_ref_mib) keeps the preset from filling the distance model.
_KECKLER = ("Keckler et al., 'GPUs and the Future of Parallel Computing', IEEE Micro 2011, Fig. 3: moving 256 bits "
            "10 mm costs 310 pJ at 40 nm, 200 pJ projected at 10 nm / 0.75 V")
PRESET_ONCHIP = {
    "horowitz_45nm": {
        "pJ_bit_mm_onchip": (310.0 / 256 / 10, "来源", _KECKLER + " → 0.121 pJ/bit/mm（40 nm，作 45 nm 代理）"),
        "pJ_bit_hop_noc": (NOC_HOP_PJ_BIT_45NM, "推导", NOC_SRC + "（45 nm 值 0.416）"),
    },
    "tpuv4i_7nm": {
        "pJ_bit_mm_onchip": (200.0 / 256 / 10, "来源/假设", _KECKLER + " → 0.078 pJ/bit/mm（10 nm 投影作 7 nm 代理，偏低侧）"),
        "pJ_bit_hop_noc": (NOC_HOP_PJ_BIT_7NM, "推导", NOC_SRC),
    },
}
for _pid, _f in PRESET_ONCHIP.items():
    ENERGY_PRESETS[_pid]["fields"].update(_f)

# 0.302: per-generation DRAM energy and LPDDR background power (tpuv4i_7nm, the preset that carries per-kind DRAM).
# HBM4: Samsung (HBM4 product page / newsroom 2026-02, 1c DRAM + 4 nm base die) and SK hynix (newsroom 2025-09) both
#   state ≈ 40 % better power efficiency than HBM3E → 3.44 / 1.4 = 2.46 pJ/bit 「推导」 (vendor claim; HBM4E uses it).
# LPDDR6: Kim et al. (Samsung), 'A 16Gb 12.8Gb/s LPDDR6 SDRAM …', ISSCC 2026 pp. 268–270 (doi 10.1109/ISSCC49663.
#   2026.11409018): '21.0 % better energy efficiency than LPDDR5x' → 4.0 / 1.21 = 3.31 pJ/bit 「推导」.
# LPDDR background: Micron Y52Q LPDDR5X data sheet (315b/441b/561b, Rev. G 05/2025), Table 16 per x16 channel of a
#   2-channel 16 Gb die (8 Gb / channel), max values: IDD2N 1.3 mA VDD1 + 28 mA VDD2H + 0.2 mA VDD2L + 0.6 mA VDDQ at
#   1.8 / 1.05 / 0.9 / 0.5 V typ = 32.2 mW 「来源」; refresh (IDD5 − IDD3N) = (14 − 2.7) mA × 1.8 V + (124 − 35) mA ×
#   1.05 V = 113.8 mW for tRFCab 210 ns (Table 5) every tREFI 3.9 µs (JEDEC LPDDR5 at ≤ 85 °C 「二手」) → 6.12 mW
#   per 8 Gb = 0.765 mW/Gb 「推导」.  IDD2N on every slice is the non-power-down upper side (idle ranks in power-down
#   draw IDD2P 2.3 mA VDD2H) 「假设」; LPDDR6 uses the LPDDR5X figure 「假设」.
LPDDR_CH_GB = 8.0
_MICRON_Y52Q = "Micron Y52Q LPDDR5X data sheet (Rev. G 05/2025) Table 16 / Table 5"
# 0.303: LPDDR low-power states, same data sheet (max values, per x16 channel of 8 Gb):
#   power-down IDD2P (Table 16): 1.4 mA·1.8 V + 2.3 mA·1.05 V + 0.2 mA·0.9 V + 0.6 mA·0.5 V = 5.42 mW 「来源」;
#   power-down self refresh (Table 18, 85 °C maximum; refresh internal): 1.2·1.8 + 7.0·1.05 + 0.2·0.9 + 0.6·0.5 = 9.99 mW
#   「来源」 (typical 25 °C 0.77 mW — the max at 85 °C is the conservative side, like IDD2N).
#   Minimum gaps (JEDEC LPDDR5 general spec, referenced by the data sheet 「二手」): power-down exit tXP = 7.5 ns (+ entry
#   ≈ tCSPD) → 15 ns; self-refresh exit tXSR = tRFCab + 7.5 ns = 217.5 ns + minimum residency tSR 15 ns → 232.5 ns.
ENERGY_PRESETS["tpuv4i_7nm"]["fields"].update({
    "mW_lpddr_ch_pd": (1.4 * 1.8 + 2.3 * 1.05 + 0.2 * 0.9 + 0.6 * 0.5, "来源", _MICRON_Y52Q.replace("Table 16 / Table 5", "Table 16")
                       + "：IDD2P 每 x16 通道（8 Gb）1.4 mA·1.8 V + 2.3 mA·1.05 V + 0.2 mA·0.9 V + 0.6 mA·0.5 V = 5.42 mW（最大值）"),
    "mW_lpddr_ch_sr": (1.2 * 1.8 + 7.0 * 1.05 + 0.2 * 0.9 + 0.6 * 0.5, "来源", "Micron Y52Q LPDDR5X data sheet（Rev. G 05/2025）"
                       "Table 18 掉电自刷新 85 °C 最大值：1.2 mA·1.8 V + 7.0 mA·1.05 V + 0.2 mA·0.9 V + 0.6 mA·0.5 V = 9.99 mW（含内部刷新）"),
    "ns_lpddr_pd_gap": (15.0, "二手/假设", "JEDEC LPDDR5 tXP 7.5 ns + 进入 ≈ 7.5 ns（数据手册引用 JEDEC 通用规范）"),
    "ns_lpddr_sr_gap": (232.5, "二手", "JEDEC LPDDR5 tXSR = tRFCab 210 ns + 7.5 ns，加最短自刷新驻留 tSR 15 ns"),
    "pJ_bit_dram_hbm4": (4.3 * 0.8 / 1.4, "推导", "HBM3E 3.44 pJ/bit ÷ 1.4 — Samsung HBM4（官网 / 2026-02 新闻稿）与 SK hynix"
                         "（2025-09 新闻稿）均称能效比 HBM3E 高约 40 %（厂商口径）；HBM4E 沿用"),
    "pJ_bit_dram_lpddr6": (4.0 / 1.21, "推导", "LPDDR5X 4.0 pJ/bit ÷ 1.21 — Kim et al.（Samsung）ISSCC 2026 LPDDR6 12.8 Gb/s："
                           "'21.0 % better energy efficiency than LPDDR5x'（doi 10.1109/ISSCC49663.2026.11409018）"),
    "mW_lpddr_ch_standby": (1.3 * 1.8 + 28 * 1.05 + 0.2 * 0.9 + 0.6 * 0.5, "来源", _MICRON_Y52Q + "：IDD2N 每 x16 通道"
                            "（8 Gb）1.3 mA·1.8 V + 28 mA·1.05 V + 0.2 mA·0.9 V + 0.6 mA·0.5 V = 32.2 mW（最大值；非掉电待机，偏高侧）"),
    "mW_lpddr_Gb_refresh": (((14 - 2.7) * 1.8 + (124 - 35) * 1.05) * 210 / 3904 / 8, "推导/二手", _MICRON_Y52Q +
                            "：(IDD5 − IDD3N) 功耗 113.8 mW × tRFCab 210 ns / tREFI 3.9 µs（JEDEC ≤ 85 °C）÷ 8 Gb = 0.765 mW/Gb"),
})

NOC_BASIS = ("NoC：每字节 × 路由跳数 × pJ_bit_hop_noc + 每字节 × 链路长度（引擎间距）× pJ_bit_mm_onchip；流量来自数据流"
             "（A 操作数多播到持有输出列的引擎、跨组拷贝、split-K 部分和归约树），拓扑 mesh / ring / crossbar（Park DAC 2012；Keckler 2011）")


def noc_pitch_mm(table: EnergyTable, chip) -> float:
    """0.299: engine pitch (mm) = √(engine tile area): its SRAM slice (the 0.298 density) + its MAC array
    (noc_mm2_per_kmac).  Uncore / vector units are left out, so the pitch is a lower bound 「推导/假设」."""
    dens = table.slc_mm2_per_mib if table.slc_mm2_per_mib is not None else SLC_MM2_PER_MIB_DEFAULT
    km = table.noc_mm2_per_kmac if table.noc_mm2_per_kmac is not None else NOC_MM2_PER_KMAC_DEFAULT
    e = max(1, int(getattr(chip, "engines", 1) or 1))
    return math.sqrt(sram_slice_mib(chip) * dens + float(chip.macs) / e / 1024.0 * km)


def unit_pJ(table: EnergyTable, k: str, chip=None, dram_field: str | None = None) -> float | None:
    """0.299: pJ per count of action ``k`` (per bit for _BITS) on ``chip`` — the one place for the per-chip terms
    (SLC / SRAM distance, NoC wire per pitch); the NoC terms only when pJ_bit_hop_noc is set (opt-in)."""
    if k == "slc":
        return slc_pJ_bit(table, float(getattr(chip, "slc_mib", 0.0) or 0.0)) if chip is not None else table.pJ_bit_slc
    if k == "sram":
        return sram_pJ_bit(table, chip) if chip is not None else table.pJ_bit_sram
    if k == "noc":
        return table.pJ_bit_hop_noc
    if k == "noc_wire":
        if table.pJ_bit_hop_noc is None or table.pJ_bit_mm_onchip is None or chip is None:
            return None
        return table.pJ_bit_mm_onchip * noc_pitch_mm(table, chip)
    if k == "dram" and dram_field:
        return getattr(table, dram_field)
    return getattr(table, _UNIT_PJ[k])


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
                                          "pJ_bit_dram_hbm4", "pJ_bit_dram_lpddr6", "mW_lpddr_ch_standby", "mW_lpddr_Gb_refresh",
                                          "mW_lpddr_ch_pd", "mW_lpddr_ch_sr", "ns_lpddr_pd_gap", "ns_lpddr_sr_gap",
                                          "pJ_clk_mac_cycle", "sram_temp_C", "dram_temp_C", "slc_access_ref_mib", "pJ_bit_mm_onchip", "slc_mm2_per_mib",
                                          "pJ_bit_hop_noc", "noc_mm2_per_kmac")
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
                                     "basis": "pJ_bit_slc × max(1, √(slc_mib/ref)) (Horowitz √C, 0.266)",
                                     # 0.268: deprecated — extrapolates 8 KB–1 MB cache points to GiB (MODEL §212.2)
                                     "deprecated": True,
                                     "note": SLC_SQRTC_DEPRECATION}
            elif e is not None and table.pJ_bit_mm_onchip is not None and slc_mib > 0:
                d = slc_wire_mm(table, slc_mib)
                dens = table.slc_mm2_per_mib if table.slc_mm2_per_mib is not None else SLC_MM2_PER_MIB_DEFAULT
                out["slc_access"] = {"pJ_bit": e, "pJ_bit_base": table.pJ_bit_slc, "slc_mib": slc_mib,
                                     "wire_mm": d, "pJ_bit_mm": table.pJ_bit_mm_onchip, "mm2_per_mib": dens,
                                     "array_mm2": slc_mib * dens, "over_reticle": slc_mib * dens > RETICLE_MM2,
                                     "scale": e / table.pJ_bit_slc if table.pJ_bit_slc else None,
                                     "basis": "pJ_bit_slc + pJ_bit_mm_onchip × ½√(slc_mib·mm²/MiB) (Keckler 2011, 0.267)"}
        elif k == "sram":
            ch = r.scenario.chip
            e = sram_pJ_bit(table, ch)
            if e is not None and table.pJ_bit_mm_onchip is not None:
                dens = table.slc_mm2_per_mib if table.slc_mm2_per_mib is not None else SLC_MM2_PER_MIB_DEFAULT
                out["sram_access"] = {"pJ_bit": e, "pJ_bit_base": table.pJ_bit_sram, "slice_mib": sram_slice_mib(ch),
                                      "engines": ch.engines, "wire_mm": sram_wire_mm(table, ch),
                                      "pJ_bit_mm": table.pJ_bit_mm_onchip, "mm2_per_mib": dens,
                                      "scale": e / table.pJ_bit_sram if table.pJ_bit_sram else None,
                                      "basis": SRAM_WIRE_BASIS}
        elif k in ("noc", "noc_wire"):
            e = unit_pJ(table, k, r.scenario.chip)
            if k == "noc" and e is not None:
                ch = r.scenario.chip
                out["noc"] = {"pJ_bit_hop": e, "pitch_mm": noc_pitch_mm(table, ch),
                              "pJ_bit_pitch": unit_pJ(table, "noc_wire", ch), "topology": ch.noc_topology,
                              "engines": ch.engines,
                              "byte_hops": per.get("noc", 0.0), "byte_pitches": per.get("noc_wire", 0.0),
                              "basis": NOC_BASIS}
                if a.get("noc_dram"):        # 0.306: the DRAM bytes' share (memory columns → engines)
                    nd = a["noc_dram"]
                    out["noc"]["dram"] = {"byte_hops": nd["byte_hops"] / u, "byte_pitches": nd["byte_pitches"] / u,
                                          "hops_per_byte": nd["hops_per_byte"],
                                          "pitches_per_byte": nd["pitches_per_byte"],
                                          "J": (nd["byte_hops"] * e + nd["byte_pitches"] * (unit_pJ(table, "noc_wire", ch) or 0.0))
                                          * 8e-12 / u}
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
    if iw is None and dram_static_set(table):
        ut = dram_util(r, per)                            # 0.303: LPDDR residency at the modelled utilisation
        hs = hbm_static_W(r.scenario.mem_id, table, ut, lpddr_gap_s(r, ut) if ut is not None else None)
        # 0.216: HBM standby + refresh (static); 0.302: + LPDDR
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
    out["breakdown"] = energy_breakdown(out)      # 0.299: component groups (UI / insight)
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

# 0.299: component groups of J_by_action — the on-chip wire (SRAM / SLC distance terms and NoC links) is split out of
# its action so the design trade-offs (SRAM size, engine count, topology) read directly
BREAKDOWN_GROUPS = ("mac", "vec", "sram", "slc", "wire", "noc", "dram", "io", "clock", "static")
BREAKDOWN_LABEL = {"mac": "MAC", "vec": "向量", "sram": "SRAM 阵列", "slc": "SLC 阵列", "wire": "片上导线",
                   "noc": "NoC 路由", "dram": "DRAM（含 PHY）", "io": "片间 / 主机链路", "clock": "时钟",
                   "static": "静态（空闲 / 漏电 / DRAM 后台）"}


def energy_breakdown(rep: dict) -> dict:
    """0.299: {"J": {group: J/unit}, "share": {group: fraction}} from an energy report.  Σ groups = J_per_unit.
    SRAM / SLC J are split into array (pJ_bit_*) and wire (distance term) parts by their access scale."""
    j = rep.get("J_by_action") or {}
    g = dict.fromkeys(BREAKDOWN_GROUPS, 0.0)

    def split(key: str, acc: dict | None) -> None:
        v = j.get(key, 0.0)
        sc = (acc or {}).get("scale") or 1.0
        wire = v * (1.0 - 1.0 / sc) if sc > 1.0 else 0.0
        g[key] += v - wire
        g["wire"] += wire
    split("sram", rep.get("sram_access"))
    split("slc", rep.get("slc_access"))
    g["mac"] += j.get("mac", 0.0)
    g["vec"] += j.get("vec", 0.0)
    g["wire"] += j.get("noc_wire", 0.0)
    g["noc"] += j.get("noc", 0.0)
    g["dram"] += j.get("dram", 0.0) + j.get("dram_phy", 0.0)
    g["io"] += j.get("d2d", 0.0) + j.get("link", 0.0) + j.get("net", 0.0) + j.get("host", 0.0)
    g["clock"] += j.get("clock", 0.0)
    g["static"] += j.get("idle", 0.0) + j.get("sram_leak", 0.0) + j.get("hbm_static", 0.0)
    tot = sum(g.values())
    return {"J": g, "share": {k: (v / tot if tot > 0 else 0.0) for k, v in g.items()}, "total": tot,
            "labels": BREAKDOWN_LABEL}


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


# 0.305: Micron Y52Q Table 18 / Table 21 (full-array power-down self refresh, one x16 channel of 8 Gb, all rails at
# nominal 1.8 / 1.05 / 0.9 / 0.5 V): 25 / 45 / 65 °C typical (reference-only), 85 °C maximum 「来源」 — mW
LPDDR_SR_MW_T = ((25.0, 0.17 * 1.8 + 0.42 * 1.05 + 0.01 * 0.9 + 0.03 * 0.5),
                 (45.0, 0.23 * 1.8 + 0.75 * 1.05 + 0.01 * 0.9 + 0.03 * 0.5),
                 (65.0, 0.40 * 1.8 + 2.00 * 1.05 + 0.01 * 0.9 + 0.03 * 0.5),
                 (85.0, 1.20 * 1.8 + 7.00 * 1.05 + 0.20 * 0.9 + 0.60 * 0.5))
LPDDR_T_MAX_C = 95.0        # Y52Q IT grade TC ≤ 95 °C 「来源」


def dram_temp_of(table) -> float | None:
    """0.305: the LPDDR temperature in effect: dram_temp_C, else sram_temp_C, else None (85 °C maximum)."""
    t = getattr(table, "dram_temp_C", None)
    return t if t is not None else getattr(table, "sram_temp_C", None)


def lpddr_sr_temp_factor(t_C: float | None) -> float:
    """0.305: self-refresh power at t_C relative to the 85 °C maximum the preset's mW_lpddr_ch_sr holds: log-linear
    between the Y52Q points (self-refresh current is dominated by the temperature-compensated internal refresh rate and
    leakage, both ~exponential in T 「推导」); clamped to the 25 °C value below 25 °C; 1 at ≥ 85 °C (the extra refresh
    above 85 °C is lpddr_refresh_factor's).  The 25–65 °C points are typical, 85 °C a guaranteed maximum, so the curve
    steps from typical to worst-case toward 85 °C (labelled)."""
    if t_C is None or t_C >= 85.0:
        return 1.0
    pts = LPDDR_SR_MW_T
    if t_C <= pts[0][0]:
        return pts[0][1] / pts[-1][1]
    for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
        if t0 <= t_C <= t1:
            return math.exp(math.log(p0) + (math.log(p1) - math.log(p0)) * (t_C - t0) / (t1 - t0)) / pts[-1][1]
    return 1.0


def lpddr_refresh_factor(t_C: float | None) -> float:
    """0.305: controller-issued refresh rate relative to tREFI = 3.9 µs: 2× above 85 °C (MR4 refresh multiplier 0.5 ×
    tREFI, JEDEC LPDDR5 「二手」), 1× at ≤ 85 °C (the slower low-temperature MR4 rates are vendor thresholds not in the
    data sheet: not credited, the high side)."""
    return 2.0 if t_C is not None and t_C > 85.0 else 1.0


def lpddr_residency(sp, table: "EnergyTable", util: float, gap_s: float | None) -> dict:
    """0.303: per-slice time fractions of an LPDDR card's 8 Gb channel slices in standby / power-down / self-refresh.

    Mechanism 「推导」 (controller policy 「假设」: a statically scheduled accelerator knows its idle gaps, so it enters the
    deepest state whose exit latency + minimum residency fits the gap and exits ahead of the next access):
      u      DRAM busy fraction = modelled bytes/s ÷ the memory's effective bandwidth (capped at 1);
      ranks  slices per x16 channel (n_r = slices / channels): while a channel is busy only one rank is accessed, so a
             slice is in standby u / n_r of the time and in power-down for the other (n_r − 1)·u / n_r (rank bursts ≫ tXP);
      gaps   the channel-idle fraction 1 − u arrives as gaps of gap_s (caller: (1 − u)·window / layers — one weight /
             KV stream per layer); a gap ≥ ns_lpddr_sr_gap is self-refresh for gap − that (the rest standby), a gap ≥
             ns_lpddr_pd_gap power-down likewise, else standby.  gap_s None (unknown) → power-down only.
    Refresh is charged on every non-self-refresh fraction (self-refresh current already contains it)."""
    ch = max(1.0, sp.bus_bits / 16.0)
    sl = sp.nominal_capacity_GB * 8.0 / LPDDR_CH_GB
    nr = max(1.0, sl / ch)
    u = min(1.0, max(0.0, util))
    f_sb, f_pd, f_sr = u / nr, u * (1.0 - 1.0 / nr), 0.0
    idle = 1.0 - u
    pd_g = (table.ns_lpddr_pd_gap or 0.0) * 1e-9
    sr_g = (table.ns_lpddr_sr_gap or 0.0) * 1e-9
    if idle > 0:
        if gap_s is not None and table.mW_lpddr_ch_sr is not None and table.ns_lpddr_sr_gap is not None and gap_s >= sr_g:
            k = 1.0 - sr_g / gap_s
            f_sr += idle * k
            f_sb += idle * (1.0 - k)
        elif gap_s is None or gap_s >= pd_g:
            k = 1.0 if gap_s is None or gap_s <= 0 else 1.0 - pd_g / gap_s
            f_pd += idle * k
            f_sb += idle * (1.0 - k)
        else:
            f_sb += idle
    return {"util": u, "ranks_per_channel": nr, "channels": ch, "gap_s": gap_s,
            "f_standby": f_sb, "f_powerdown": f_pd, "f_selfrefresh": f_sr}


def lpddr_gap_s(r, util: float) -> float | None:
    """0.303: idle-gap length for lpddr_residency: (1 − u) × the evaluation window ÷ the card's layers (one DRAM stream
    per layer per window 「假设」)."""
    try:
        a = action_counts(r)
        nl = len(r.model.layers) / max(1, r.scenario.layout.pp)
    except Exception:                       # noqa: BLE001 — no layer structure (non-LLM pipelines): unknown gap
        return None
    if not a.get("window_s") or nl <= 0:
        return None
    return max(0.0, 1.0 - util) * a["window_s"] / nl


def lpddr_window(r) -> dict | None:
    """0.304: one evaluated iteration's DRAM window for the queue ledger: busy fraction during the iteration
    (DRAM bytes ÷ (cards × window) ÷ effective bandwidth), the window and the card's layers (gap = (1 − u)·window ÷
    layers, as lpddr_gap_s).  None without a memory spec / layer structure."""
    from .. import mem_catalog
    try:
        a = action_counts(r)
        sp = mem_catalog.parse_mem_id(r.scenario.mem_id)
        nl = len(r.model.layers) / max(1, r.scenario.layout.pp)
    except Exception:                       # noqa: BLE001
        return None
    w, bw = a.get("window_s") or 0.0, getattr(sp, "effective_GBps", 0.0) * 1e9
    cs = r.scenario.layout.cards * w
    if w <= 0 or bw <= 0 or nl <= 0 or cs <= 0:
        return None
    return {"util": min(1.0, a["counts"].get("dram", 0.0) / cs / bw), "window_s": w, "layers": nl}


def hbm_static_W_window(mem_id: str | None, table: "EnergyTable", util: float | None, win: dict | None) -> dict | None:
    """0.304: DRAM background of a queue pool whose cards alternate between iterations and empty-queue idle.

    The pool's average DRAM busy fraction ``util`` (its bytes per card-second ÷ effective bandwidth) and one
    representative iteration ``win`` (lpddr_window) give the fraction of time the cards are iterating,
    ρ = util ÷ u_iter (≤ 1).  While iterating the residency is the single-evaluation one (lpddr_residency at u_iter,
    gaps (1 − u_iter)·window ÷ layers — the same mechanism as energy_report); an empty-queue period lasts at least
    one iteration window (the next request starts an iteration 「假设」) and is spent at u = 0 with that gap.
    W = ρ·W_iter + (1 − ρ)·W_idle (residency fractions are linear in time 「推导」).  ``win`` None → the 0.303
    power-down-only bound; HBM → hbm_static_W unchanged."""
    if util is None or win is None:
        return hbm_static_W(mem_id, table, util, None)
    ui = win["util"]
    if util >= ui or ui <= 0:
        rho, ub = 1.0, max(util, ui)
    else:
        rho, ub = util / ui, ui
    b = hbm_static_W(mem_id, table, ub, max(0.0, 1.0 - ub) * win["window_s"] / win["layers"])
    if not b or "residency" not in b:
        return b
    i = hbm_static_W(mem_id, table, 0.0, win["window_s"])
    mix = lambda x, y: rho * x + (1.0 - rho) * y      # noqa: E731
    out = dict(b)
    for k in ("W", "standby_W", "powerdown_W", "selfrefresh_W", "refresh_W"):
        out[k] = mix(b[k], i[k])
    out["residency"] = {**{k: mix(b["residency"][k], i["residency"][k])
                           for k in ("f_standby", "f_powerdown", "f_selfrefresh")},
                        "util": util, "util_iter": ui, "busy_frac": rho, "window_s": win["window_s"],
                        "gap_s": b["residency"]["gap_s"], "ranks_per_channel": b["residency"]["ranks_per_channel"],
                        "channels": b["residency"]["channels"]}
    out["basis"] = b["basis"] + "；排队：迭代时段按单次评估驻留，空队时段（≥ 一个迭代窗口）自刷新（MODEL §249）"
    return out


def hbm_static_W(mem_id: str | None, table: "EnergyTable", util: float | None = None,
                 gap_s: float | None = None) -> dict | None:
    """0.216: HBM background + refresh power of the card's stacks: stacks × height × standby/die + total Gb × refresh/Gb."""
    if not mem_id:
        return None
    from .. import mem_catalog
    try:
        sp = mem_catalog.parse_mem_id(mem_id)
    except Exception:
        return None
    if getattr(sp, "kind", None) != "HBM":
        # 0.302: LPDDR — IDD2N standby on every x16 channel slice of 8 Gb (Micron die organisation) + refresh per Gb
        if table.mW_lpddr_ch_standby is None and table.mW_lpddr_Gb_refresh is None:
            return None
        gb = sp.nominal_capacity_GB * 8.0
        sl = gb / LPDDR_CH_GB
        stb = sl * (table.mW_lpddr_ch_standby or 0.0) * 1e-3
        tC = dram_temp_of(table)                                       # 0.305: temperature
        rf = lpddr_refresh_factor(tC)
        ref = gb * (table.mW_lpddr_Gb_refresh or 0.0) * 1e-3 * rf
        if util is not None and table.mW_lpddr_ch_pd is not None:      # 0.303: low-power residency
            res = lpddr_residency(sp, table, util, gap_s)
            pd = sl * res["f_powerdown"] * table.mW_lpddr_ch_pd * 1e-3
            # 0.305: self-refresh at temperature (+ the doubled refresh above 85 °C, a slice's refresh share)
            sr_mW = (table.mW_lpddr_ch_sr or 0.0) * lpddr_sr_temp_factor(tC) \
                + (rf - 1.0) * LPDDR_CH_GB * (table.mW_lpddr_Gb_refresh or 0.0)
            sr = sl * res["f_selfrefresh"] * sr_mW * 1e-3
            res = {**res, "temp_C": tC, "sr_mW_per_slice": sr_mW, "refresh_factor": rf}
            stb_r = stb * res["f_standby"]
            ref_r = ref * (1.0 - res["f_selfrefresh"])
            return {"W": stb_r + pd + sr + ref_r, "standby_W": stb_r, "powerdown_W": pd, "selfrefresh_W": sr,
                    "refresh_W": ref_r, "W_no_lowpower": stb + ref, "kind": "LPDDR", "channel_slices": sl,
                    "residency": res,
                    "basis": "LPDDR：每 8 Gb x16 通道片按驻留加权 待机 IDD2N / 掉电 IDD2P / 自刷新（Micron Y52Q）+ 非自刷新时段的刷新；"
                             "驻留来自建模的 DRAM 利用率、每通道 rank 数与层间空闲间隙（MODEL §248）"}
        return {"W": stb + ref, "standby_W": stb, "refresh_W": ref, "kind": "LPDDR", "channel_slices": sl,
                "temp_C": tC, "refresh_factor": rf,
                "basis": "LPDDR：每 8 Gb x16 通道片 IDD2N 待机（全部电源轨）+ 每 Gb 刷新（Micron Y52Q）"}
    if not getattr(sp, "hbm_height", None):
        return None
    dies = sp.n_units * sp.hbm_height
    stb = dies * (table.mW_hbm_die_standby or 0.0) * 1e-3
    ref = dies * sp.hbm_die_Gb * (table.mW_hbm_Gb_refresh or 0.0) * 1e-3
    return {"W": stb + ref, "standby_W": stb, "refresh_W": ref, "stacks": sp.n_units, "dies": dies, "kind": "HBM"}

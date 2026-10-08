"""Scenario presets (v0.24; real memory configs v0.29): package + compute + chips bundles.

Presets only bundle catalog ids that already exist in ``package_ranges``
(assumed DSE ranges) plus a chip count / parallel mapping. They carry **no**
power or price numbers — energy / cost knobs stay user-supplied (see
``econ.py`` and ``examples/energy_cost.example.json``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .package_ranges import get_compute, get_package


@dataclass(frozen=True)
class ScenarioPreset:
    id: str
    label: str
    package_id: str
    compute_id: str
    chip_count: int = 1
    tp: int | None = None
    pp: int = 1
    ep: int = 1
    note: str = ""

    def workbench_kwargs(self) -> dict[str, Any]:
        """Knobs for WorkbenchConfig (package geometry + compute + chips)."""
        from .workbench import ParallelOverride

        kw: dict[str, Any] = {}
        kw.update(get_package(self.package_id).workbench_kwargs())
        kw.update(get_compute(self.compute_id).workbench_kwargs())
        kw["chip_count"] = int(self.chip_count)
        if self.tp is not None:
            kw["parallel"] = ParallelOverride(tp=self.tp, pp=self.pp, ep=self.ep)
        else:
            kw["parallel"] = None
        return kw

    def body(self) -> dict[str, Any]:
        """API / UI body fragment (ids, not expanded geometry)."""
        b: dict[str, Any] = {
            "package_id": self.package_id,
            "compute_id": self.compute_id,
            "chip_count": self.chip_count,
        }
        if self.tp is not None:
            b.update({"tp": self.tp, "pp": self.pp, "ep": self.ep})
        return b

    def to_dict(self) -> dict[str, Any]:
        pkg = get_package(self.package_id)
        comp = get_compute(self.compute_id)
        return {
            "id": self.id,
            "label": self.label,
            "package_id": self.package_id,
            "compute_id": self.compute_id,
            "chip_count": self.chip_count,
            "tp": self.tp,
            "pp": self.pp,
            "ep": self.ep,
            "mem_kind": pkg.kind,
            "peak_tops_per_card": comp.peak_tops,
            "eff_GBps_per_card": pkg.effective_bandwidth_GBps(),
            "capacity_GB_per_card": pkg.capacity_GB,
            "mem_label": pkg.mem_spec.short_label(),
            "mem": {
                k: v
                for k, v in pkg.mem_spec.workbench_kwargs().items()
                if k.startswith(("mem_", "hbm_")) and k != "mem_kind" and v is not None
            },
            "mem_tag": pkg.mem_spec.tag,
            "mem_tag_zh": pkg.mem_spec.tag_zh,
            "note": self.note,
        }

    def summary(self) -> str:
        pkg = get_package(self.package_id)
        comp = get_compute(self.compute_id)
        par = (
            f"tp/pp/ep={self.tp}/{self.pp}/{self.ep}"
            if self.tp is not None
            else f"tp={self.chip_count} (default)"
        )
        return (
            f"{self.id}: {self.label} — {pkg.id} ({pkg.kind}, eff "
            f"{pkg.effective_bandwidth_GBps():.0f} GB/s, {pkg.capacity_GB:.0f} GB) + "
            f"{comp.id} ({comp.peak_tops:g} T) × chips={self.chip_count} {par} [assumed]"
        )


# v0.29: presets use real catalog configs (research/memory_specs_2026-10 §6.3).
# Ids of the three ≤0.28 presets are kept (deep links / tests); their memory
# moved from invented geometry to shipping configurations.
SCENARIO_PRESETS: dict[str, ScenarioPreset] = {
    p.id: p
    for p in (
        ScenarioPreset(
            id="edge-lpddr-4x64",
            label="边缘 SoC · LPDDR5X 4×x64 @8533 (256-bit, 64 GB) + 64T · 1 芯",
            package_id="lpddr5x_4x64_8533_16g",
            compute_id="cluster_64t",
            chip_count=1,
            note="edge-class: 4 × x64 LPDDR5X packages (4×16-bit ch each) @ 8533 MT/s, "
            "16 GB/pkg (厂商量产); 273 GB/s raw; BW-walled for big LLMs",
        ),
        ScenarioPreset(
            id="edge-lpddr6-4x96",
            label="边缘 SoC · LPDDR6 4×x96 @10667 (384-bit, 64 GB) + 128T · 1 芯",
            package_id="lpddr6_4x96_10667_16g",
            compute_id="cluster_128t",
            chip_count=1,
            note="LPDDR6 x96 package = 4×24-bit ch (8×12-bit sub-ch); 512 GB/s raw, "
            "455 GB/s usable after 8/9 payload; 16 GB/pkg (CXMT 量产)",
        ),
        ScenarioPreset(
            id="card-hbm-4stack",
            label="加速卡 · HBM3E 4×12H×24Gb @9200 (144 GB) + 256T · 1 芯",
            package_id="hbm3e_4s_12h24g_9200",
            compute_id="cluster_256t",
            chip_count=1,
            note="single PCIe/OAM-style card; 4 HBM3E 12-high 36 GB stacks (厂商量产)",
        ),
        ScenarioPreset(
            id="card-hbm3e-8x12h",
            label="加速卡 · HBM3E 8×12H×24Gb @9200 (288 GB) + 256T · 1 芯",
            package_id="hbm3e_8s_12h24g_9200",
            compute_id="cluster_256t",
            chip_count=1,
            note="B300 / MI355X-class geometry: 8 × 36 GB stacks, 9.42 TB/s raw",
        ),
        ScenarioPreset(
            id="server-socamm2-8",
            label="服务器 · SOCAMM2 8×128b @9600 (1.5 TB) + 256T · 1 芯",
            package_id="lpddr5x_socamm2_8x128_9600_192g",
            compute_id="cluster_256t",
            chip_count=1,
            note="Vera-class LPDDR5X modules (JESD328 SOCAMM2), 192 GB each; "
            "capacity-rich, BW 1.2 TB/s raw",
        ),
        ScenarioPreset(
            id="scaleup-8chip",
            label="Scale-up · 8 × (HBM3E 4×12H + 256T) · tp=8",
            package_id="hbm3e_4s_12h24g_9200",
            compute_id="cluster_256t",
            chip_count=8,
            note="8 cards, default tp=8 over assumed C2C (α=3 µs/collective exposed); "
            "adjust tp/pp/ep after apply",
        ),
    )
}

PRESET_ALIASES = {
    "edge": "edge-lpddr-4x64",
    "card": "card-hbm-4stack",
    "scaleup": "scaleup-8chip",
    "scaleup8": "scaleup-8chip",
    "edge-lpddr6": "edge-lpddr6-4x96",
    "card-hbm3e": "card-hbm3e-8x12h",
    "socamm2": "server-socamm2-8",
}


def get_preset(preset_id: str) -> ScenarioPreset:
    key = str(preset_id).strip().lower().replace("_", "-")
    key = PRESET_ALIASES.get(key, key)
    if key not in SCENARIO_PRESETS:
        raise KeyError(
            f"unknown scenario preset {preset_id!r}; known: {sorted(SCENARIO_PRESETS)}"
        )
    return SCENARIO_PRESETS[key]


def list_presets() -> list[ScenarioPreset]:
    return list(SCENARIO_PRESETS.values())


def format_preset_table(presets: list[ScenarioPreset] | None = None) -> str:
    rows = presets if presets is not None else list_presets()
    hdr = (
        f"{'preset':18s}  {'package':32s}  {'compute':14s}  {'chips':>5}  "
        f"{'T/card':>7}  {'GB/s/card':>9}  {'GB/card':>7}"
    )
    lines = [hdr, "-" * len(hdr)]
    for p in rows:
        d = p.to_dict()
        lines.append(
            f"{p.id:18s}  {p.package_id:32s}  {p.compute_id:14s}  {p.chip_count:5d}  "
            f"{d['peak_tops_per_card']:7.0f}  {d['eff_GBps_per_card']:9.0f}  "
            f"{d['capacity_GB_per_card']:7.0f}"
        )
    return "\n".join(lines)


def evaluate_presets(
    model_id: str,
    *,
    preset_ids: list[str] | None = None,
    **overrides: Any,
) -> list[tuple[ScenarioPreset, Any]]:
    """Evaluate each preset for ``model_id`` → [(preset, MetricsCard)].

    ``overrides`` (workload / econ knobs) are applied on top of the preset;
    preset package/compute/chips win over geometry-only overrides.
    """
    from .workbench import WorkbenchConfig, evaluate_workbench

    ids = preset_ids or [p.id for p in list_presets()]
    out: list[tuple[ScenarioPreset, Any]] = []
    for pid in ids:
        p = get_preset(pid)
        kw = {k: v for k, v in overrides.items() if v is not None}
        kw.update(p.workbench_kwargs())
        kw["model_id"] = model_id
        card = evaluate_workbench(WorkbenchConfig(**kw))
        card.notes = list(card.notes) + [f"preset={p.id} [assumed]"]
        out.append((p, card))
    return out


def format_preset_eval_table(rows: list[tuple[ScenarioPreset, Any]]) -> str:
    hdr = (
        f"{'preset':18s}  {'chips':>5}  {'primary':>10}  {'ms':>11}  {'wall':>8}  "
        f"{'oom':>3}  {'est_W':>9}  {'E/unit_J':>10}  {'sys_$':>10}  {'$/MTok':>9}"
    )
    lines = [hdr, "-" * len(hdr)]
    for p, c in rows:
        if c.domain == "video":
            key, val, e = "TTFC", c.TTFC_ms, c.est_energy_per_frame_J
        elif c.domain == "protein":
            key, val, e = "t/seq", c.time_per_seq_ms, c.est_energy_per_seq_J
        else:
            key, val, e = "TPOT", c.TPOT_ms, c.est_energy_per_token_J
        lines.append(
            f"{p.id:18s}  {c.chips:5d}  {key:>10}  {val:11.4f}  {c.wall:>8}  "
            f"{str(c.oom)[0]:>3}  {c.est_power_W:9.1f}  {e:10.4g}  "
            f"{c.est_system_cost_usd:10.0f}  {c.est_usd_per_Mtok:9.4g}"
        )
    lines.append(
        "(est_* = ASSUMED econ stub from user knobs; 0 = knob not set. Not silicon power.)"
    )
    return "\n".join(lines)

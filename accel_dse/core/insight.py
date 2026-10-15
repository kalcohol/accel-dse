"""NPU design insight (0.146): which hardware knob moves TTFT / TPOT / decode throughput most, per area / power proxy.

For a chip configuration (the scenario's chip, memory, layout, serving) and a set of catalog models, every knob is a
controlled one-variable probe of the *existing* engine (``evaluate``) — the same machinery as the eval sensitivity
and the sweeps; nothing here models hardware by itself.  Probes (each relative to the base point):

  sram_x2       SRAM MiB × 2                         (on-die; area term mm2_per_mib_sram)
  slc_add       + system-level cache = SRAM MiB       (only when the chip has none; area term mm2_per_mib_slc)
  macs_x2       MAC rows × 2 (same cols / engines)   (on-die; area term mm2_per_kmac)
  engines_x2    cores × 2 (same per-core array)      (on-die; area term mm2_per_kmac)
  shape_tall    rows × 2, cols ÷ 2 at equal MACs     (array aspect only; no area change in the proxy)
  dram_rate     next catalog speed grade, same parts (DRAM bandwidth; PHY / package — not in the die-area proxy)
  dram_count    next catalog package / stack count   (bandwidth + capacity; PHY / package)
  native_fp8    toggle native fp8 / int8 MAC (×2 bf16, the FormatSupport default 「假设」): the value of having it
  native_mxfp4  add native mxfp4 MAC at 2 × the fp8 rate 「假设」(NVIDIA Blackwell dense FP4 = 2 × FP8); used only
                by GEMMs whose weights *and* activations are mxfp4 (dtypes.gemm_exec) — as-released MXFP4-expert
                models run W4A8 / W4A16, so this probe shows what a W4A4 datapath alone buys them (usually nothing)
  mixed_w4a8 / mixed_w4a16 / mixed_w8a16   (0.147) native mixed-precision MAC pairs, dtypes.MIXED_PRESETS: the W4 / W8
                weight dequant vanishes and the pair runs at its rate (W4A8 = FP8 rate, PTX ISA kind::f8f6f4; W4A16 /
                W8A16 = bf16 rate 「假设」)
  dequant_x4    weight-dequant / convert throughput × 4 (the W4 / W8 → MAC-format path on the vector unit)
  d2d_x2        die-to-die bandwidth × 2              (only with D2D on and package_cards > 1)
  c2c_x2        card-to-card link bandwidth × 2       (only with > 1 card)

Probe sizes (× 2, next grade) are probes, not claims about cost.  Metrics per model: TTFT (prefill latency of the
scenario's prompt), TPOT and decode throughput per card at the scenario's batch.  A knob's score per metric is the
geometric mean over the models that evaluate (speed-up > 1 = better); models whose point does not fit stay in the
table but are flagged.  Area: Δ die-area proxy from the user's budget densities (budget.area_estimate) — no density,
no mm² (the tool ships no process library).  Power proxy: Δ energy per output token from the user's energy table
(energy.energy_report); without one, the Δ action counts per token (DRAM bytes, MAC slots, SRAM bytes) — model
outputs that need no assumption.
"""
from __future__ import annotations

import math
from dataclasses import replace

from .. import mem_catalog
from .budget import Budget, area_estimate
from .catalog import get_model
from .energy import EnergyTable, energy_report
from .evaluate import evaluate
from .scenario import Scenario

DEFAULT_MODELS = ("qwen3-8b", "gpt-oss-120b", "mimo-v2.6-flash", "step-5-preview", "deepseek-v3.2", "ling-3.0-flash-vl")
MAX_MODELS = 8
MXFP4_VS_FP8 = 2.0      # 「假设」 native mxfp4 MAC rate = 2 × fp8 (NVIDIA Blackwell dense FP4 vs FP8)
METRICS = ("ttft", "tpot", "thr")
TIE = 0.01              # 「假设」 within 1 % = no move (same as api.SENS_TIE)


def _features(mid: str) -> list[str]:
    m = get_model(mid)
    f = []
    cores = [l.core for l in m.layers]
    if any(c.idx_heads for c in cores):
        f.append("稀疏/索引注意力")
    if any(c.window for c in cores):
        f.append("滑窗")
    if any(c.kind == "linear" for c in cores):
        f.append("线性注意力")
    fm = dict(m.formats)
    if "expert" in fm and fm["expert"].fmt == "mxfp4":
        f.append("MXFP4 专家")
    elif any(v.fmt.startswith("fp8") for v in fm.values()):
        f.append("FP8 权重")
    if m.is_moe:
        f.append("MoE")
    return f


def model_scenario(base: Scenario, mid: str) -> Scenario:
    """The base scenario on model ``mid`` with a valid layout of the same card count (MoE: experts over the TP·DP
    ranks, else the base layout; dense: no EP)."""
    s = base.replace("model", mid)
    lay = base.layout
    moe = get_model(mid).is_moe
    tries = ([dict(ep=lay.tp * lay.dp, etp=1), dict(ep=1, etp=lay.tp * lay.dp)] if moe else [dict(dp=lay.dp * lay.ep * lay.etp // max(1, lay.ep * lay.etp), ep=1, etp=1, sp=1)])
    last = None
    for kv in tries:
        t = s
        try:
            for k, v in kv.items():
                t = t.replace("layout." + k, v)
            evaluate(t.replace("serving.phase", "decode"))
            return t
        except ValueError as e:
            last = e
    raise ValueError(f"{mid}: {last}")


def _next(vals, cur):
    up = [v for v in vals if v > cur]
    return min(up) if up else None


def knobs(base: Scenario) -> list[tuple[str, str, object, str]]:
    """(key, label, transform(Scenario) → Scenario | None, cost kind)."""
    ch = base.chip
    spec = mem_catalog.parse_mem_id(base.mem_id)
    out = []

    def chip(**kw):
        return lambda s: s.replace("chip", replace(s.chip, **kw))
    out.append(("sram_x2", f"SRAM ×2（{ch.sram_mib:g} → {ch.sram_mib * 2:g} MiB）", chip(sram_mib=ch.sram_mib * 2), "die"))
    if not ch.slc_mib:
        out.append(("slc_add", f"加 SLC {ch.sram_mib:g} MiB", chip(slc_mib=ch.sram_mib), "die"))
    out.append(("macs_x2", f"MAC 行 ×2（{ch.rows}×{ch.cols} → {ch.rows * 2}×{ch.cols}）", chip(rows=ch.rows * 2), "die"))
    out.append(("engines_x2", f"核数 ×2（{ch.engines} → {ch.engines * 2}）", chip(engines=ch.engines * 2), "die"))
    if ch.cols % 2 == 0:
        out.append(("shape_tall", f"阵列形状 {ch.rows}×{ch.cols} → {ch.rows * 2}×{ch.cols // 2}（MAC 数不变）",
                    chip(rows=ch.rows * 2, cols=ch.cols // 2), "none"))
    r = _next(mem_catalog.rates_for(spec.mem_type, spec.form), spec.rate_MTps)
    if r:
        ns = replace(spec, rate_MTps=r, legacy_id="", legacy_note="")
        out.append(("dram_rate", f"DRAM 速率 {spec.rate_MTps} → {r} MT/s（{spec.effective_GBps:.0f} → {ns.effective_GBps:.0f} GB/s）",
                    lambda s, i=ns.id: s.replace("mem_id", i), "offdie"))
    n = _next(mem_catalog.counts_for(spec.mem_type, spec.form), spec.n_units)
    if n:
        ns = replace(spec, n_units=n, legacy_id="", legacy_note="")
        out.append(("dram_count", f"DRAM {spec.unit_kind} {spec.n_units} → {n}（{spec.effective_GBps:.0f} → {ns.effective_GBps:.0f} GB/s，"
                                   f"{spec.capacity_GB:.0f} → {ns.capacity_GB:.0f} GB）", lambda s, i=ns.id: s.replace("mem_id", i), "offdie"))
    rates = dict(ch.formats.rates)
    if "fp8" in rates:
        nr = tuple((k, v) for k, v in ch.formats.rates if k not in ("fp8", "int8"))
        out.append(("native_fp8", "原生 FP8/INT8 MAC 的价值（与去掉相比）",
                    ("invert", chip(formats=replace(ch.formats, rates=nr))), "none"))
    else:
        nr = ch.formats.rates + (("fp8", 2.0), ("int8", 2.0))
        out.append(("native_fp8", "加原生 FP8/INT8 MAC（×2 bf16「假设」）", chip(formats=replace(ch.formats, rates=nr)), "none"))
    if "mxfp4" not in rates:
        fr = rates.get("fp8", 2.0) * MXFP4_VS_FP8
        out.append(("native_mxfp4", f"加原生 MXFP4 MAC（×{fr:g} bf16「假设」，仅 W4A4 GEMM 可用）",
                    chip(formats=replace(ch.formats, rates=ch.formats.rates + (("mxfp4", fr),))), "none"))
    # 0.147: native mixed-precision MAC pairs (dtypes.MIXED_PRESETS) — the dequant of W4 / W8 weights vanishes and the
    # pair runs at its own rate; already present → the value of having them (inverted)
    from .dtypes import MIXED_PRESETS
    for name, (pairs, src) in MIXED_PRESETS.items():
        have = all(ch.formats.pair_rate(w, a) is not None for w, a, _ in pairs)
        if have:
            rest = tuple(p for p in ch.formats.mixed if (p[0], p[1]) not in {(w, a) for w, a, _ in pairs})
            out.append((f"mixed_{name}", f"原生 {name.upper()} 混合精度 MAC 的价值（{src}）",
                        ("invert", chip(formats=replace(ch.formats, mixed=rest))), "none"))
        else:
            out.append((f"mixed_{name}", f"加原生 {name.upper()} 混合精度 MAC（{src}）",
                        chip(formats=replace(ch.formats, mixed=ch.formats.mixed + pairs)), "none"))
    dq = ch.formats.dequant_elems_per_cycle or float(ch.lanes)
    out.append(("dequant_x4", f"反量化 / 转换吞吐 ×4（{dq:g} → {dq * 4:g} 元素/周期）",
                chip(formats=replace(ch.formats, dequant_elems_per_cycle=dq * 4)), "none"))
    if base.d2d_enabled and base.package_cards > 1:
        out.append(("d2d_x2", f"D2D 带宽 ×2（{base.d2d.GBps:g} → {base.d2d.GBps * 2:g} GB/s）",
                    lambda s: s.replace("d2d", replace(s.d2d, GBps=s.d2d.GBps * 2)), "offdie"))
    if base.layout.cards > 1:
        out.append(("c2c_x2", f"卡间链路 ×2（{base.link.GBps:g} → {base.link.GBps * 2:g} GB/s）",
                    lambda s: s.replace("link", replace(s.link, GBps=s.link.GBps * 2)), "offdie"))
    return out


def _point(s: Scenario, table: EnergyTable) -> dict:
    d = evaluate(s.replace("serving.phase", "decode"))
    p = evaluate(s.replace("serving.phase", "prefill"))
    e = energy_report(d, table)
    per = e["counts_per_unit"]
    return {"ttft": p.ttft, "tpot": d.tpot, "thr": d.per_card, "fits": bool(d.fits and p.fits),
            "bound_d": d.bound, "bound_p": p.bound, "J_tok": e.get("J_per_unit"), "J_partial": bool(e.get("J_per_unit") and e["missing"]),
            "dram_B_tok": per.get("dram"), "mac_tok": per.get("mac"), "sram_B_tok": per.get("sram")}


def _gain(m: str, base: dict, alt: dict) -> float | None:
    a, b = base[m], alt[m]
    if not a or not b:
        return None
    return a / b if m in ("ttft", "tpot") else b / a          # speed-up, > 1 = better


def insight_report(base: Scenario, models=None, budget: Budget | None = None, table: EnergyTable | None = None) -> dict:
    table = table or EnergyTable()
    models = list(models or DEFAULT_MODELS)
    if not 1 <= len(models) <= MAX_MODELS:
        raise ValueError(f"models: 1..{MAX_MODELS} ids")
    ks = knobs(base)
    area0 = area_estimate(base.chip, budget) if budget is not None else None
    mrows, scns = [], {}
    for mid in models:
        try:
            s = model_scenario(base, mid)
            b0 = _point(s, table)
            scns[mid] = s
            mrows.append({"model": mid, "layout": s.layout.label, "features": _features(mid), **b0})
        except (ValueError, KeyError) as e:
            mrows.append({"model": mid, "error": str(e)})
    base_by = {r["model"]: r for r in mrows if "error" not in r}
    krows = []
    for key, label, tf, cost in ks:
        inv = isinstance(tf, tuple)
        f = tf[1] if inv else tf
        per_model, g = [], {m: [] for m in METRICS}
        for mid, s in scns.items():
            try:
                alt = _point(f(s), table)
            except ValueError as e:
                per_model.append({"model": mid, "error": str(e)})
                continue
            b0 = base_by[mid]
            lo, hi = (alt, b0) if inv else (b0, alt)        # invert: the value of the feature the base already has
            row = {"model": mid, "fits": hi["fits"], "bound_d": hi["bound_d"], "bound_p": hi["bound_p"],
                   "bound_flip": (hi["bound_d"], hi["bound_p"]) != (lo["bound_d"], lo["bound_p"])}
            for m in METRICS:
                x = _gain(m, lo, hi)
                row[m] = x
                if x is not None:
                    g[m].append(x)
            for c in ("J_tok", "dram_B_tok", "mac_tok", "sram_B_tok"):
                row["d_" + c] = (hi[c] / lo[c] - 1) if hi.get(c) and lo.get(c) else None
            per_model.append(row)
        score = {m: (math.exp(sum(math.log(x) for x in v) / len(v)) if v else None) for m, v in g.items()}
        d_mm2 = None
        if area0 is not None and area0["mm2"] is not None and cost == "die":
            a1 = area_estimate(f(base).chip, budget)
            d_mm2 = (a1["mm2"] - area0["mm2"]) * (-1 if inv else 1) if a1["mm2"] is not None else None
        per_mm2 = {m: ((score[m] - 1) / d_mm2 if score[m] is not None and d_mm2 else None) for m in METRICS}
        krows.append({"key": key, "label": label, "cost": cost, "inverted": inv, "score": score,
                      "d_mm2": d_mm2, "gain_per_mm2": per_mm2, "models": per_model})
    # ties within TIE (the sensitivity convention, api.SENS_TIE 「假设」) do not "move" the metric
    rank = {m: [k["key"] for k in sorted(krows, key=lambda k: -(k["score"][m] or 0))
                if k["score"][m] is not None and k["score"][m] - 1 > TIE] for m in METRICS}
    hurts = {m: [k["key"] for k in krows if k["score"][m] is not None and 1 - k["score"][m] > TIE] for m in METRICS}
    rank_mm2 = {m: [k["key"] for k in sorted((k for k in krows if k["gain_per_mm2"][m] is not None),
                                             key=lambda k: -k["gain_per_mm2"][m]) if k["score"][m] - 1 > TIE] for m in METRICS}
    return {"chip": base.chip.name, "mem_id": base.mem_id, "layout": base.layout.label,
            "serving": {"batch": base.serving.batch, "prompt": base.serving.prompt, "ctx": base.serving.ctx},
            "models": mrows, "knobs": krows, "rank": rank, "rank_per_mm2": rank_mm2, "hurts": hurts, "tie": TIE, "metrics": {"ttft": "TTFT 加速", "tpot": "TPOT 加速",
                                                                         "thr": "decode 吞吐/卡 倍数"},
            "area": area0, "energy_provided": table.provided,
            "basis": "单变量探针 × 现有引擎（evaluate）；面积只用用户密度（budget.area_estimate），功耗只用用户能耗表或动作计数；"
                     "探针幅度（×2、下一档）不是成本结论。「假设」：原生 MXFP4 速率 = 2× FP8；原生 FP8 = 2× bf16（FormatSupport 默认）。"}

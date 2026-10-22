"""NPU design insight (0.146): which hardware knob moves TTFT / TPOT / decode throughput most, per area / power proxy.

For a chip configuration (the scenario's chip, memory, layout, serving) and a set of catalog models, every knob is a
controlled one-variable probe of the *existing* engine (``evaluate``) — the same machinery as the eval sensitivity
and the sweeps; nothing here models hardware by itself.  Probes (each relative to the base point):

  sram_x2       SRAM MiB × 2                         (on-die; area term mm2_per_mib_sram)
  slc_add       + system-level cache = SRAM MiB       (only when the chip has none; area term mm2_per_mib_slc)
  (0.262)       slc_energy_report: J/token-optimal SLC size under energy.sram_temp_C (closes the 0.261 hot-die question)
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

Probe sizes (× 2, next grade) are probes, not claims about cost.  Metrics (0.151: one domain per report):
  llm       TTFT, TPOT, decode tok/s/card
  gen       clip/image latency, frames/s/card (image DiT = 1 frame → images/s)
  protein   batch latency, sequences/s/card
  genomics  batch latency, windows/s/card
A knob's score per metric is the geometric mean over the models that evaluate (speed-up > 1 = better); models whose
point does not fit stay in the table but are flagged.  Area: Δ die-area proxy from the user's budget densities
(budget.area_estimate / full_area) — no density, no mm².  Power proxy: Δ energy / action counts from the user's
energy table when given.
"""
from __future__ import annotations

import math
from dataclasses import replace

from .. import mem_catalog
from .budget import Budget, area_estimate, full_area
from .catalog import get_model
from .energy import EnergyTable, energy_report

_J_PER_KWH = 3.6e6  # 0.153: Joules per kWh


def _j_eff(pt: dict, domain: str, tdp_W: float | None) -> float | None:
    """J per output unit: energy-table J_tok wins; else constant-TDP × time 「假设」."""
    if pt.get("J_tok") is not None:
        return pt["J_tok"]
    if tdp_W is None:
        return None
    if domain == "llm":
        t = pt.get("tpot")
        return (tdp_W * t) if t else None          # decode: W · s/tok
    thr = pt.get("thr")
    return (tdp_W / thr) if thr else None          # W / (units/s/card)


def _usd_unit(j: float | None, rate: float | None) -> float | None:
    if j is None or rate is None:
        return None
    return j * rate / _J_PER_KWH


_YEAR_S = 365.25 * 86400


def _tco_unit(pt: dict, domain: str, b) -> dict | None:
    """0.154: $ per output unit = CapEx (card_usd / (amort_years · thr)) + OpEx (J · PUE · $/kWh).
    Needs card_usd + amort_years + thr; OpEx only when $/kWh and a J source exist (PUE missing → IT-only, noted)."""
    if b is None or b.card_usd is None or b.amort_years is None or not pt.get("thr"):
        return None
    capex = b.card_usd / (b.amort_years * _YEAR_S * pt["thr"])
    j = _j_eff(pt, domain, b.tdp_W_card)
    opex = (j * (b.pue or 1.0) * b.usd_per_kWh / _J_PER_KWH) if (j is not None and b.usd_per_kWh is not None) else None
    return {"capex": capex, "opex": opex, "tco": capex + (opex or 0.0)}


def _gain_per(score, d, metrics, tie=None):
    """Map Δcost → gain/cost; win-win (faster and Δ≤0) → 1e30; missing → None."""
    if tie is None:
        tie = TIE
    out = {}
    for m in metrics:
        if score[m] is None or score[m] - 1 <= tie or d is None:
            out[m] = None
        elif d > 0:
            out[m] = (score[m] - 1) / d
        else:
            out[m] = 1e30
    return out

from .evaluate import evaluate
from .scenario import Scenario

DEFAULT_MODELS = ("qwen3-8b", "gpt-oss-120b", "mimo-v2.6-flash", "step-5-preview", "deepseek-v3.2", "ling-3.0-flash-vl")
DEFAULT_MODELS_BY_DOMAIN = {
    "llm": DEFAULT_MODELS,
    "gen": ("wan2.1-1.3b", "flux.2-klein-4b", "qwen-image", "lingbot-video-moe-dmd"),
    "protein": ("esm2-650m", "esmfold", "boltz-1"),
    "genomics": ("carbon-a-1.2b",),
}
MAX_MODELS = 8
MXFP4_VS_FP8 = 2.0      # 「假设」 native mxfp4 MAC rate = 2 × fp8 (NVIDIA Blackwell dense FP4 vs FP8)
# 0.151: per-domain metric keys.  LLM keeps ttft/tpot/thr; other domains use lat (lower better) + thr (higher better).
DOMAIN_METRICS = {
    "llm": ("ttft", "tpot", "thr"),
    "gen": ("lat", "thr"),
    "protein": ("lat", "thr"),
    "genomics": ("lat", "thr"),
}
DEFAULT_METRIC = {"llm": "tpot", "gen": "thr", "protein": "thr", "genomics": "thr"}
THR_UNIT = {"llm": "tok/s/卡", "gen": "帧/s/卡", "protein": "序列/s/卡", "genomics": "窗口/s/卡"}
LAT_NAME = {"gen": "单段/图延迟", "protein": "批延迟", "genomics": "批延迟"}
METRICS = ("ttft", "tpot", "thr", "lat")          # union; a report only scores the domain's subset
TIE = 0.01              # 「假设」 within 1 % = no move (same as api.SENS_TIE)


def insight_domain(models) -> str:
    """One catalog domain for the model set, or ValueError if empty / mixed / unknown."""
    if not models:
        raise ValueError("models: need at least one id")
    doms = []
    for mid in models:
        d = get_model(mid).domain
        if d not in DOMAIN_METRICS:
            raise ValueError(f"{mid}: domain {d!r} is not supported by insight")
        doms.append(d)
    if len(set(doms)) > 1:
        raise ValueError(f"models must share one domain; got {sorted(set(doms))}")
    return doms[0]


def metric_labels(domain: str) -> dict:
    thr = f"吞吐/卡加速（{THR_UNIT[domain]}）"
    if domain == "llm":
        return {"ttft": "TTFT 加速", "tpot": "TPOT 加速", "thr": thr}
    return {"lat": f"{LAT_NAME[domain]} 加速", "thr": thr}


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
    if m.domain == "gen":
        f.append("图像/视频 DiT")
    elif m.domain == "protein":
        f.append("蛋白质")
    elif m.domain == "genomics":
        f.append("基因组")
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
            if get_model(mid).domain == "llm":
                evaluate(t.replace("serving.phase", "decode"))
            else:
                evaluate(t)
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
        if base.d2d_std == "custom":
            out.append(("d2d_x2", f"D2D 带宽 ×2（{base.d2d.GBps:g} → {base.d2d.GBps * 2:g} GB/s）",
                        lambda s: s.replace("d2d", replace(s.d2d, GBps=s.d2d.GBps * 2)), "offdie"))
        elif base.d2d_units * 2 <= 64:      # 0.149: a catalog grade's bandwidth = units × per-unit (d2d.GBps is ignored)
            out.append(("d2d_x2", f"D2D 单元 ×2（{base.d2d_units} → {base.d2d_units * 2}，"
                                  f"{base.d2d_link.GBps:g} → {base.d2d_link.GBps * 2:g} GB/s）",
                        lambda s: s.replace("d2d_units", s.d2d_units * 2), "offdie"))
    if base.layout.cards > (base.package_cards if base.d2d_enabled else 1):   # 0.150: link used only across packages
        out.append(("c2c_x2", f"卡间链路 ×2（{base.link.GBps:g} → {base.link.GBps * 2:g} GB/s）",
                    lambda s: s.replace("link", replace(s.link, GBps=s.link.GBps * 2)), "offdie"))
    return out


def _point(s: Scenario, table: EnergyTable, domain: str = "llm") -> dict:
    """One evaluated point.  LLM: decode + prefill.  Other domains: a single evaluate (phase is ignored)."""
    if domain == "llm":
        d = evaluate(s.replace("serving.phase", "decode"))
        p = evaluate(s.replace("serving.phase", "prefill"))
        e = energy_report(d, table)
        ep = energy_report(p, table)          # 0.188: prefill energy per prompt token, not decode only
        per = e["counts_per_unit"]
        return {"ttft": p.ttft, "tpot": d.tpot, "thr": d.per_card, "fits": bool(d.fits and p.fits),
                "bound_d": d.bound, "bound_p": p.bound, "J_tok": e.get("J_per_unit"),
                "J_prefill": ep.get("J_per_unit"),
                "J_partial": bool(e.get("J_per_unit") and (e["missing"] or ep["missing"])),
                "dram_B_tok": per.get("dram"), "mac_tok": per.get("mac"), "sram_B_tok": per.get("sram")}
    r = evaluate(s)
    e = energy_report(r, table)
    per = e["counts_per_unit"]
    return {"lat": r.latency, "thr": r.per_card, "fits": bool(r.fits),
            "bound_d": r.bound, "bound_p": r.bound, "J_tok": e.get("J_per_unit"),
            "J_partial": bool(e.get("J_per_unit") and e["missing"]),
            "dram_B_tok": per.get("dram"), "mac_tok": per.get("mac"), "sram_B_tok": per.get("sram"),
            "unit": r.workload.unit if r.workload else None}


def _gain(m: str, base: dict, alt: dict) -> float | None:
    a, b = base[m], alt[m]
    if not a or not b:
        return None
    return a / b if m in ("ttft", "tpot", "lat", "J_tok", "J_prefill") else b / a   # speed-up, > 1 = better (J: less = better)


def insight_report(base: Scenario, models=None, budget: Budget | None = None, table: EnergyTable | None = None) -> dict:
    table = table or EnergyTable()
    usd_kwh = budget.usd_per_kWh if budget is not None else None
    tdp_W = budget.tdp_W_card if budget is not None else None
    models = list(models) if models is not None else list(DEFAULT_MODELS)
    if not 1 <= len(models) <= MAX_MODELS:
        raise ValueError(f"models: 1..{MAX_MODELS} ids")
    domain = insight_domain(models)
    metrics = DOMAIN_METRICS[domain]
    ks = knobs(base)
    area0 = full_area(base, budget) if budget is not None else None
    mrows, scns = [], {}
    for mid in models:
        try:
            s = model_scenario(base, mid)
            b0 = _point(s, table, domain)
            scns[mid] = s
            mrows.append({"model": mid, "layout": s.layout.label, "features": _features(mid), "domain": domain, **b0})
        except (ValueError, KeyError) as e:
            mrows.append({"model": mid, "error": str(e)})
    base_by = {r["model"]: r for r in mrows if "error" not in r}
    krows = []
    for key, label, tf, cost in ks:
        inv = isinstance(tf, tuple)
        f = tf[1] if inv else tf
        per_model, g, dJ_acc, tco_acc = [], {m: [] for m in metrics}, [], []
        for mid, s in scns.items():
            try:
                alt = _point(f(s), table, domain)
            except ValueError as e:
                per_model.append({"model": mid, "error": str(e)})
                continue
            b0 = base_by[mid]
            lo, hi = (alt, b0) if inv else (b0, alt)        # invert: the value of the feature the base already has
            row = {"model": mid, "fits": hi["fits"], "bound_d": hi["bound_d"], "bound_p": hi["bound_p"],
                   "bound_flip": (hi["bound_d"], hi["bound_p"]) != (lo["bound_d"], lo["bound_p"])}
            for m in metrics:
                x = _gain(m, lo, hi)
                row[m] = x
                if x is not None:
                    g[m].append(x)
            for c in ("J_tok", "J_prefill", "dram_B_tok", "mac_tok", "sram_B_tok"):
                row["d_" + c] = (hi[c] / lo[c] - 1) if hi.get(c) and lo.get(c) else None
            t_lo, t_hi = _tco_unit(lo, domain, budget), _tco_unit(hi, domain, budget)
            if t_lo and t_hi and t_hi["tco"] > 0:
                row["tco_gain"] = t_lo["tco"] / t_hi["tco"]          # > 1 = cheaper per unit
                tco_acc.append(row["tco_gain"])
            j_lo, j_hi = _j_eff(lo, domain, tdp_W), _j_eff(hi, domain, tdp_W)
            if j_lo is not None and j_hi is not None:
                dJ_acc.append(j_hi - j_lo)               # 0.152/0.153: absolute ΔJ / unit (table or TDP×t)
            per_model.append(row)
        score = {m: (math.exp(sum(math.log(x) for x in v) / len(v)) if v else None) for m, v in g.items()}
        d_mm2 = None
        if area0 is not None and area0["mm2"] is not None and cost in ("die", "offdie"):
            a1 = full_area(f(base), budget)      # 0.149: incl. interface PHYs (DRAM pins, D2D units, C2C) when set
            d_mm2 = (a1["mm2"] - area0["mm2"]) * (-1 if inv else 1) if a1["mm2"] is not None else None
            d_mm2 = d_mm2 if d_mm2 else None
        per_mm2 = {m: ((score[m] - 1) / d_mm2 if score[m] is not None and d_mm2 else None) for m in metrics}
        d_J = (sum(dJ_acc) / len(dJ_acc)) if dJ_acc else None
        d_J = d_J if d_J is not None and abs(d_J) > 1e-12 else (0.0 if d_J is not None else None)
        # win-win (faster, ΔJ ≤ 0) sorts first as 1e30; JSON-safe stand-in for +∞
        per_J = _gain_per(score, d_J, metrics)
        d_usd = (d_J * usd_kwh / _J_PER_KWH) if (d_J is not None and usd_kwh is not None) else None
        d_usd = d_usd if d_usd is not None and abs(d_usd) > 1e-18 else (0.0 if d_usd is not None else None)
        per_usd = _gain_per(score, d_usd, metrics)
        krows.append({"key": key, "label": label, "cost": cost, "inverted": inv, "score": score,
                      "d_mm2": d_mm2, "gain_per_mm2": per_mm2, "d_J": d_J, "gain_per_J": per_J,
                      "d_usd": d_usd, "gain_per_usd": per_usd,
                      "tco_gain": (math.exp(sum(math.log(x) for x in tco_acc) / len(tco_acc)) if tco_acc else None),
                      "models": per_model})
    # ties within TIE (the sensitivity convention, api.SENS_TIE 「假设」) do not "move" the metric
    rank = {m: [k["key"] for k in sorted(krows, key=lambda k: -(k["score"][m] or 0))
                if k["score"][m] is not None and k["score"][m] - 1 > TIE] for m in metrics}
    hurts = {m: [k["key"] for k in krows if k["score"][m] is not None and 1 - k["score"][m] > TIE] for m in metrics}
    rank_mm2 = {m: [k["key"] for k in sorted((k for k in krows if k["gain_per_mm2"][m] is not None),
                                             key=lambda k: -k["gain_per_mm2"][m]) if k["score"][m] - 1 > TIE] for m in metrics}
    rank_J = {m: [k["key"] for k in sorted((k for k in krows if k["gain_per_J"][m] is not None),
                                           key=lambda k: -k["gain_per_J"][m]) if k["score"][m] - 1 > TIE] for m in metrics}
    rank_usd = {m: [k["key"] for k in sorted((k for k in krows if k["gain_per_usd"][m] is not None),
                                             key=lambda k: -k["gain_per_usd"][m]) if k["score"][m] - 1 > TIE] for m in metrics}
    rank_tco = [k["key"] for k in sorted((k for k in krows if k["tco_gain"] is not None), key=lambda k: -k["tco_gain"])
                if k["tco_gain"] - 1 > TIE]
    tco0 = {r["model"]: _tco_unit(r, domain, budget) for r in mrows if "error" not in r}
    tco0 = {k: v for k, v in tco0.items() if v}
    notes = (["pd.enabled：洞察按合并部署（prefill + decode 同卡）评估，不含 PD 分离"] if base.pd.enabled else [])
    if tdp_W is not None and not table.provided:
        notes.append("tdp_W_card：无能耗表时按恒定 TDP × 时间估 J（「假设」）；有表时表优先")
    if tco0:
        notes.append("TCO：旋钮不改卡价「假设」（硬件增量成本请看 Δmm² / 升级路径）；CapEx = card_usd ÷ (amort_years × 吞吐/卡)")
        if budget.pue is None and budget.usd_per_kWh is not None:
            notes.append("未填 PUE：OpEx 只计 IT 电费")
        if budget.usd_per_kWh is None:
            notes.append("未填电价：TCO 只含 CapEx")
    j_src = ("能耗表" if table.provided else ("TDP×时间" if tdp_W is not None else None))
    # 0.264: package 25/85 °C SLC energy into the default insight path when leakage is modellable
    slc_temp = None
    from .energy import sram_leak_uW_per_mib as _leak_uW
    if _leak_uW(table) is not None:
        pol = base.chip.slc_policy or "dip"
        try:
            cmp = slc_energy_report(base, table, [r["model"] for r in mrows if "error" not in r] or models,
                                    policy=pol, gibs=INSIGHT_SLC_GIB, temps=(25, 85))
            # compact: drop by_temp curves (UI / API consumers use summary + recommend)
            by_model = {}
            for row in cmp["summary"]:
                by_model.setdefault(row["model"], {})[row["temp_C"]] = row
            recommend = []
            for mid, ts in by_model.items():
                c25, c85 = ts.get(25.0), ts.get(85.0)
                if not c25 or not c85 or c25.get("error") or c85.get("error"):
                    continue
                recommend.append({
                    "model": mid,
                    "opt_J_gib_25C": c25.get("opt_J_gib"), "crossover_gib_25C": c25.get("crossover_gib"),
                    "opt_J_gib_85C": c85.get("opt_J_gib"), "crossover_gib_85C": c85.get("crossover_gib"),
                    "opt_time_gib_25C": c25.get("opt_time_gib"), "opt_time_gib_85C": c85.get("opt_time_gib"),
                    "hot_cuts_J_opt": (c25.get("opt_J_gib") or 0) > (c85.get("opt_J_gib") or 0),
                })
            slc_temp = {"policy": pol, "temps": [25.0, 85.0], "gibs": list(INSIGHT_SLC_GIB),
                        "leak_charged": True, "summary": cmp["summary"], "recommend": recommend,
                        "notes": cmp.get("notes") or [],
                        "basis": "insight_report embeds slc_energy_report(temps=(25,85)) (0.264)"}
            if any(x.get("hot_cuts_J_opt") for x in recommend):
                notes.append("SLC：85 °C 下 J 最优容量低于 25 °C（漏电交叉；见 slc_temp）")
            else:
                notes.append("SLC：已附 25/85 °C J 最优对比（slc_temp）")
        except ValueError as e:
            notes.append(f"SLC 25/85 °C 对比跳过：{e}")
    return {"chip": base.chip.name, "mem_id": base.mem_id, "layout": base.layout.label, "notes": notes,
            "serving": {"batch": base.serving.batch, "prompt": base.serving.prompt, "ctx": base.serving.ctx},
            "models": mrows, "knobs": krows, "rank": rank, "rank_per_mm2": rank_mm2, "rank_per_J": rank_J,
            "rank_per_usd": rank_usd, "rank_tco": rank_tco, "tco": tco0, "hurts": hurts, "tie": TIE, "domain": domain, "metrics": metric_labels(domain),
            "default_metric": DEFAULT_METRIC[domain],
            "area": area0, "energy_provided": table.provided,
            "usd_per_kWh": usd_kwh, "tdp_W_card": tdp_W, "j_source": j_src, "slc_temp": slc_temp,
            "basis": "单变量探针 × 现有引擎（evaluate）；面积只用用户密度（budget.area_estimate / full_area），功耗只用用户能耗表或动作计数；"
                     "有漏电场时附带 slc_temp（25/85 °C J 最优，0.264）；"
                     "探针幅度（×2、下一档）不是成本结论。指标随领域：LLM = TTFT/TPOT/tok/s；视频/图像 = 延迟/帧/s；"
                     "蛋白质 = 延迟/序列/s；基因组 = 延迟/窗口/s。能耗预设（0.152）或用户表点亮 ΔJ / 每焦耳排序；"
                     "idle_W 不进预设。0.153：$/kWh（电价预设或手填）+ 能耗 ΔJ（或 tdp_W_card × 时间）→ Δ$ / 每美元排序；"
                     "不内置默认电价。「假设」：原生 MXFP4 = 2× FP8；原生 FP8 = 2× bf16。"}


# ---------------------------------------------------------------------------------------------------------------- 0.148
# Multi-step upgrade path: greedy by gain per cost, re-evaluated after every step (knob interactions), never worse.
REL_COST = {  # 「假设」 relative-cost proxy units, used only when the user gave no area densities (or for non-die knobs)
    "sram_x2": 4.0, "slc_add": 3.0, "macs_x2": 4.0, "engines_x2": 6.0, "shape_tall": 0.5, "dram_rate": 1.0,
    "dram_count": 3.0, "native_fp8": 1.0, "native_mxfp4": 1.5, "mixed_w4a8": 1.0, "mixed_w4a16": 1.0,
    "mixed_w8a16": 0.5, "dequant_x4": 0.5, "d2d_x2": 2.0, "c2c_x2": 2.0}
SCALE_WITH = {"sram_x2": "sram_mib", "slc_add": "sram_mib", "macs_x2": "macs", "engines_x2": "macs"}  # ×2 of more costs more
UNIT_MM2_FRAC = 0.01    # 「假设」 area mode: 1 proxy unit of a non-die knob ≈ 1 % of the start die-area proxy
MAX_STEPS = 10


def _geo(v):
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else None


MAX_BEAM = 5


def upgrade_path(base: Scenario, models=None, metric: str | None = None, budget: Budget | None = None, steps: int = 6,
                 max_cost: float | None = None, table: EnergyTable | None = None, beam: int = 1) -> dict:
    """Multi-step upgrade path for one metric over a model set.  Each step re-runs every additive knob of
    `knobs(current)` on every model and keeps candidates that (a) raise the model-set geomean by > TIE, (b) leave every
    model's metric no worse than at the START (cumulative, exact ≥ 1), (c) keep every model that fit still fitting,
    (d) stay inside the budget (die_mm2 in area mode; max_cost units otherwise).  beam = 1: greedy, largest
    ln(gain)/cost.  beam = w > 1 (0.149): keep the w best partial paths by cumulative ln(gain)/cost each step; the
    greedy path is always computed too and the beam result is returned only if its final gain is ≥ greedy's (so
    beam is never worse than greedy).  Either way the path is never worse than the start."""
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= MAX_STEPS:
        raise ValueError(f"steps must be an integer 1..{MAX_STEPS}")
    if not isinstance(beam, int) or isinstance(beam, bool) or not 1 <= beam <= MAX_BEAM:
        raise ValueError(f"beam must be an integer 1..{MAX_BEAM}")
    if max_cost is not None and (isinstance(max_cost, bool) or not isinstance(max_cost, (int, float))
                                 or not math.isfinite(max_cost) or max_cost <= 0):
        raise ValueError("max_cost must be a finite number > 0")
    table = table or EnergyTable()
    models = list(models) if models is not None else list(DEFAULT_MODELS)
    if not 1 <= len(models) <= MAX_MODELS:
        raise ValueError(f"models: 1..{MAX_MODELS} ids")
    domain = insight_domain(models)
    metrics = DOMAIN_METRICS[domain]
    if metric is None:
        metric = DEFAULT_METRIC[domain]
    if metric not in metrics:
        raise ValueError(f"metric must be one of {metrics} for domain {domain}")
    a0 = full_area(base, budget) if budget is not None else None
    area_mode = bool(a0 and a0["mm2"] is not None)
    cap_mm2 = budget.die_mm2 if (area_mode and budget.die_mm2) else None
    unit = a0["mm2"] * UNIT_MM2_FRAC if area_mode else 1.0

    scns, start, skipped = {}, {}, []
    for mid in models:
        try:
            s = model_scenario(base, mid)
            start[mid] = _point(s, table, domain)
            scns[mid] = s
        except (ValueError, KeyError) as e:
            skipped.append({"model": mid, "error": str(e)})
    if not scns:
        raise ValueError("no model could be evaluated on the start configuration")
    memo = {}

    def chip_qty(chip, attr):
        return chip.macs if attr == "macs" else getattr(chip, attr)

    def expand(cur_base, cur, cum_cost):
        """All admissible single-knob steps from (cur_base, cur): list of candidate tuples, best ratio first."""
        cands = []
        cur_a = full_area(cur_base, budget)["mm2"] if area_mode else None
        for key, label, tf, kind in knobs(cur_base):
            if isinstance(tf, tuple) or key not in REL_COST:
                continue                                  # inverted = removing an existing feature; never an upgrade
            nb = tf(cur_base)
            mm2, area_cost = None, False
            if area_mode:
                a1 = full_area(nb, budget)["mm2"]
                if a1 is None:                            # density for this term missing (e.g. SLC) → not costable
                    continue
                if cap_mm2 is not None and a1 > cap_mm2 * (1 + 1e-12) and a1 > cur_a:
                    continue
                if a1 - cur_a > 1e-9:                     # a real area term (die, or interface PHY when set)
                    mm2, area_cost = a1 - cur_a, True
            if area_cost:
                cost = mm2
            else:                                         # no area term → 「假设」 relative cost (× 1 % area in area mode)
                c = REL_COST[key]
                if key in SCALE_WITH and not area_mode:
                    c *= chip_qty(cur_base.chip, SCALE_WITH[key]) / chip_qty(base.chip, SCALE_WITH[key])
                cost = c * unit
            if max_cost is not None and cum_cost + cost > max_cost * (1 + 1e-12):   # 0.150: both modes (cost units)
                continue
            pts, ok = {}, True
            for mid in scns:
                mk = (nb.hash(), mid)
                if mk not in memo:
                    try:
                        memo[mk] = _point(model_scenario(nb, mid), table, domain)
                    except ValueError:
                        memo[mk] = None
                p = memo[mk]
                if p is None or (start[mid]["fits"] and not p["fits"]):
                    ok = False
                    break
                gs = _gain(metric, start[mid], p)
                if gs is None or gs < 1.0:               # (b) no model worse than at the start
                    ok = False
                    break
                pts[mid] = p
            if not ok:
                continue
            step_g = _geo([_gain(metric, cur[m], pts[m]) for m in pts])
            if step_g is None or step_g - 1 <= TIE:
                continue
            cands.append({"ratio": math.log(step_g) / cost, "key": key, "label": label, "kind": kind, "cost": cost,
                          "d_mm2": mm2, "area_cost": area_cost, "gain": step_g, "base": nb, "pts": pts})
        cands.sort(key=lambda c: (-c["ratio"], c["key"]))
        return cands

    def step_row(n, c, cum_cost, cur, others):
        cum = _geo([_gain(metric, start[m], cur[m]) for m in cur])
        return {"step": n, "key": c["key"], "label": c["label"], "kind": c["kind"], "cost": c["cost"], "d_mm2": c["d_mm2"],
                "cost_basis": "mm²" if c["area_cost"] else "假设", "cum_cost": cum_cost, "gain": c["gain"], "cum_gain": cum,
                "gain_per_cost": c["ratio"], "alternatives": [{"key": o["key"], "gain": o["gain"], "cost": o["cost"]} for o in others[:3]],
                "models": [{"model": m, "cum": _gain(metric, start[m], cur[m]), "bound_d": cur[m]["bound_d"],
                            "bound_p": cur[m]["bound_p"], metric: cur[m][metric]} for m in cur]}

    def run(width):
        states = [(base, dict(start), 0.0, [])]          # (config, points, cum cost, path rows)
        finished = []
        for n in range(1, steps + 1):
            children = []
            for cb, cur, cc, path in states:
                cands = expand(cb, cur, cc)
                if not cands:
                    finished.append((cb, cur, cc, path))
                    continue
                pick = cands[:max(1, width - 1)]
                if width > 1:                              # + the biggest single step (may be pricier)
                    big = max(cands, key=lambda c: (c["gain"], -c["cost"]))
                    pick = pick + ([big] if all(big is not x for x in pick) else cands[width - 1:width])
                for c in pick:
                    ncc = cc + c["cost"]
                    row = step_row(n, c, ncc, c["pts"], [o for o in cands if o is not c])
                    children.append((c["base"], c["pts"], ncc, path + [row]))
            if not children:
                states = []
                break
            uniq = {}
            for ch in children:                          # same config reached twice → keep the cheaper
                k = ch[0].hash()
                if k not in uniq or ch[2] < uniq[k][2]:
                    uniq[k] = ch
            ranked = sorted(uniq.values(), key=lambda t: (-math.log(t[3][-1]["cum_gain"]) / t[2], t[2]))
            states = ranked[:max(1, width - 1)]           # most efficient partial paths …
            if width > 1:                                  # … plus the highest-gain one (diversity: big steps)
                rest = [t for t in ranked if all(t is not x for x in states)]
                if rest:
                    states.append(max(rest, key=lambda t: (t[3][-1]["cum_gain"], -t[2])))
        finished += states
        return max(finished, key=lambda t: ((t[3][-1]["cum_gain"] if t[3] else 1.0), -t[2]))

    g = run(1)
    best, search = g, "greedy"
    g_gain = g[3][-1]["cum_gain"] if g[3] else 1.0
    if beam > 1:
        b = run(beam)
        b_gain = b[3][-1]["cum_gain"] if b[3] else 1.0
        if b_gain > g_gain * (1 + 1e-9) or (abs(b_gain / g_gain - 1) <= 1e-9 and b[2] < g[2] * (1 - 1e-9)):
            best, search = b, f"beam{beam}"
        else:
            search = f"beam{beam}（= 贪心）"
    cur_base, cur, cum_cost, path = best
    final = {m: _gain(metric, start[m], cur[m]) for m in cur}
    assert all(x >= 1.0 for x in final.values())          # the guarantee; candidates violating it were never taken
    notes = []
    if base.pd.enabled:
        notes.append("pd.enabled：升级路径按合并部署（prefill + decode 同卡）评估，不含 PD 分离")
    return {"metric": metric, "domain": domain, "metrics": metric_labels(domain), "cost_mode": "area" if area_mode else "proxy", "search": search, "beam": beam, "notes": notes,
            "greedy_cum_gain": g_gain, "greedy_cum_cost": g[2],
            "cost_unit": "mm²" if area_mode else "相对成本单位「假设」",
            "unit_mm2": unit if area_mode else None, "cap_mm2": cap_mm2, "max_cost": max_cost,
            "start_area_mm2": a0["mm2"] if area_mode else None,
            "start": {m: start[m][metric] for m in scns}, "final": {m: cur[m][metric] for m in cur},
            "path": path, "skipped": skipped, "rel_cost": REL_COST,
            "final_chip": {"sram_mib": cur_base.chip.sram_mib, "slc_mib": cur_base.chip.slc_mib, "rows": cur_base.chip.rows,
                           "cols": cur_base.chip.cols, "engines": cur_base.chip.engines, "mem_id": cur_base.mem_id,
                           "formats": [list(x) for x in cur_base.chip.formats.rates],
                           "mixed": [list(x) for x in cur_base.chip.formats.mixed],
                           "dequant_elems_per_cycle": cur_base.chip.formats.dequant_elems_per_cycle},
            "basis": ("贪心" if beam == 1 else f"束搜索（宽 {beam}，结果不劣于贪心）") +
                     "：每步对当前配置重跑全部加法旋钮（含交互），按 ln(几何平均增益)/成本排序；只接受几何平均提升 > "
                     f"{TIE:g} 且每个模型相对起点不变差、原本放得下的仍放得下、且不超预算的步骤，因此路径不会差于起点。"
                     + ("成本 = 用户密度 / 预设下的 Δ 面积 mm²（含接口 PHY：DRAM 引脚、D2D 单元、C2C，若已设）；无面积项的旋钮按 "
                        "「假设」 1 单位 = 起点面积 1 % 折算。" if area_mode else
                        "成本 = 「假设」相对成本单位（REL_COST；SRAM / MAC 类随当前规模放大）；给出面积密度（或密度预设）后改用 mm²。")}


# ---------------------------------------------------------------------------------------------------------------- 0.157
# Datapath sizing headroom: how far can a sized resource shrink before any model's native metric degrades > TIE?
# The single-knob probes above say what doubling buys; this answers the architect's opposite question (area to cut).
def _memspec(s):
    return mem_catalog.parse_mem_id(s.mem_id, efficiency=s.mem_eff)


def _dram_delivered(s) -> float:
    """0.164: delivered DRAM GB/s = min(memory effective, opt-in DMA issue limit) — as System.dram_GBps."""
    m = float(_memspec(s).effective_GBps)
    c = s.chip
    if c.dma_engines:
        return min(m, c.dma_engines * c.dma_outstanding_KiB * 1024 / (c.dram_latency_ns * 1e-9) / 1e9, c.noc_mem_GBps())
    return min(m, c.noc_mem_GBps())          # 0.305: NoC DRAM ingress


def _chip_set(f):
    return lambda s, v: s.replace("chip", f(s.chip, v))


def _lanes_set(ch, v):
    # 0.163: the dequant path defaults to the vector width (formats.dequant_elems_per_cycle None → lanes); ≤ 0.162 a
    # narrower vector unit silently also cut dequant throughput.  Pin it so only the vector width moves.
    f = ch.formats if ch.formats.dequant_elems_per_cycle else replace(ch.formats, dequant_elems_per_cycle=float(ch.lanes))
    return replace(ch, vector_lanes=max(1, int(round(v))), formats=f)


def _cores_set(ch, v):
    # 0.158: fewer engines (cores) at fixed vector width / per-engine accumulator & port — the MAC array alone shrinks
    return replace(ch, engines=max(1, int(round(v))), vector_lanes=ch.lanes)


def _dram_units_set(s, v):
    # 0.183: drop whole DRAM packages / stacks (same parts, rate, height, mem_eff): bandwidth AND capacity shrink
    # together and the interface PHY shrinks with the bus (budget.phy_area: bus_bits × mm2_per_dq_*)
    spec = _memspec(s)
    n = max(1, int(round(v)))
    return s.replace("mem_id", replace(spec, n_units=n, legacy_id="", legacy_note="").id)


SIZING = {
    # key: (label, unit, get(Scenario) → value, set(Scenario, value) → Scenario, integer, area density field | None)
    "dequant": ("反量化 / 转换吞吐", "元素/周期",
                lambda s: float(s.chip.formats.dequant_elems_per_cycle or s.chip.lanes),
                _chip_set(lambda ch, v: replace(ch, formats=replace(ch.formats, dequant_elems_per_cycle=v))), False, "mm2_per_kdq"),
    "lanes": ("向量单元宽度", "元素/周期", lambda s: float(s.chip.lanes),
              _chip_set(_lanes_set), True, "mm2_per_klane"),
    "sram": ("SRAM 容量", "MiB", lambda s: float(s.chip.sram_mib), _chip_set(lambda ch, v: replace(ch, sram_mib=v)), False,
             "mm2_per_mib_sram"),
    # 0.158: MAC array via core (engine) count, R×C per engine fixed; area = ΔkMAC × mm2_per_kmac (density preset / user)
    "cores": ("MAC 阵列（核数）", "核", lambda s: float(s.chip.engines), _chip_set(_cores_set), True, "mm2_per_kmac"),
    # 0.158: effective DRAM bandwidth at fixed stack / capacity (scaled through mem_eff ≤ current); no PHY area (discrete)
    "dram_bw": ("DRAM 有效带宽", "GB/s", lambda s: _dram_delivered(s),
                lambda s, v: s.replace("mem_eff", v / _memspec(s).payload_GBps), False, None),
    # 0.183: discrete DRAM interface width (package / stack count) priced by the DRAM PHY density (mm² per data pin)
    "dram_units": ("DRAM 封装 / 堆栈数", "个", lambda s: float(_memspec(s).n_units), _dram_units_set, True, "mm2_per_dq"),
}
SIZING_MIN_FRAC = 1 / 1024        # search floor (10 halvings)
SIZING_REFINE = 3                 # geometric bisection steps after the first failing halving


_JOINT_ORDER = ("cores", "lanes", "dequant", "sram", "dram_bw")   # cores pins vector_lanes, so lanes is applied after it
# 0.183: dram_units is per-resource only — it and dram_bw act on the same DRAM wire (dram_bw is an absolute GB/s
# through mem_eff relative to the current bus), so joint back-off over both is not separable; joint skips it.
JOINT_SKIP = ("dram_units",)


def sizing_report(base: Scenario, models=None, budget: Budget | None = None, resources=None, joint: bool = False,
                  unpriced: str = "last", table: EnergyTable | None = None, energy_tie: bool = False) -> dict:
    """Per resource: the smallest value (≥ current × SIZING_MIN_FRAC) at which every model keeps every domain metric
    within TIE of the current chip and still fits.  Halving search, then geometric bisection; the reported minimum is
    always a probed point that passed.  ``limited_by`` = the first model / metric that fails just below it."""
    models = list(models) if models is not None else list(DEFAULT_MODELS)
    if not 1 <= len(models) <= MAX_MODELS:
        raise ValueError(f"models: 1..{MAX_MODELS} ids")
    domain = insight_domain(models)
    metrics = DOMAIN_METRICS[domain]
    res = list(resources) if resources is not None else list(SIZING)
    bad_res = [r for r in res if r not in SIZING]
    if bad_res:
        raise ValueError(f"resources must be among {sorted(SIZING)}")
    scns = {mid: model_scenario(base, mid) for mid in models}
    # 0.187: energy exposure — with an energy table each row reports J / unit at min_ok vs current (by action), and
    # energy_tie=True makes "J / unit no worse than current × (1 + TIE)" part of the feasibility contract.
    table = table or EnergyTable()
    if energy_tie and not table.provided:
        raise ValueError("energy_tie needs an energy table (energy / energy_preset)")
    ref = {mid: _point(s, table, domain) for mid, s in scns.items()}
    if energy_tie:
        metrics = tuple(metrics) + (("J_tok", "J_prefill") if domain == "llm" else ("J_tok",))   # 0.188: + prefill

    def check(key, v):
        """None if every model holds; else (model, metric | 'fits', gain)."""
        setter = SIZING[key][3]
        for mid, s in scns.items():
            try:
                p = _point(setter(s, v), table, domain)
            except ValueError:
                return (mid, "invalid", None)
            if ref[mid]["fits"] and not p["fits"]:
                return (mid, "fits", None)
            for m in metrics:
                g = _gain(m, ref[mid], p)
                if g is not None and g < 1 - TIE:
                    return (mid, m, g)
        return None

    def apply_all(s, vals):
        for k in _JOINT_ORDER:
            if k in vals:
                s = SIZING[k][3](s, vals[k])
        return s

    def worst(vals):
        """0.160: (min gain over models × metrics, or -inf on a fit loss / invalid point, first failing (model, metric))."""
        w, why = math.inf, None
        for mid, s in scns.items():
            try:
                p = _point(apply_all(s, vals), table, domain)
            except ValueError:
                return -math.inf, (mid, "invalid")
            if ref[mid]["fits"] and not p["fits"]:
                return -math.inf, (mid, "fits")
            for m in metrics:
                g = _gain(m, ref[mid], p)
                if g is not None and g < w:
                    w, why = g, (mid, m)
        return w, why

    out = []
    for key in res:
        label, unit, get, _, integer, dens = SIZING[key]
        cur = get(base)
        ok, fail, why = cur, None, None
        v = cur
        while v / 2 >= cur * SIZING_MIN_FRAC:
            prev, v = v, v / 2
            if integer:
                v = float(max(1, int(round(v))))
                if v >= prev:          # 0.183: integer floor (1) reached — ≤ 0.182 looped forever at a current of 1
                    break
            w = check(key, v)
            if w is None:
                ok = v
            else:
                fail, why = v, w
                break
        if fail is not None:
            for _ in range(SIZING_REFINE):
                mid_v = math.sqrt(ok * fail)
                if integer:
                    mid_v = float(max(1, int(round(mid_v))))
                if mid_v in (ok, fail):
                    break
                w = check(key, mid_v)
                if w is None:
                    ok = mid_v
                else:
                    fail, why = mid_v, w
        row = {"key": key, "label": label, "unit": unit, "current": cur, "min_ok": ok, "frac": ok / cur if cur else None,
               "floor_hit": fail is None, "fail_at": fail,
               # 0.184: an integer resource already at 1 has no search below it — not "headroom to the search floor"
               "integer": bool(integer),
               "unit_floor": bool(integer and fail is None and ok <= 1),
               "limited_by": None if why is None else {"model": why[0], "metric": why[1], "gain": why[2]}}
        row["traffic"] = _traffic_delta(scns, ref, SIZING[key][3], ok, table, domain)
        d = _density(budget, dens, base)
        if d is not None:
            row["mm2_saved"] = (cur - ok) * _unit_mm2(key, base, d)
        out.append(row)
    rep = {"domain": domain, "models": models, "metrics": {**metric_labels(domain), **({"J_tok": "能耗 / 单位（越低越好）"} if energy_tie else {}),
                                                         **({"J_prefill": "prefill 能耗 / prompt token（越低越好）"} if energy_tie and domain == "llm" else {})},
           "energy_tie": bool(energy_tie), "tie": TIE, "resources": out,
           "chip": base.chip.name, "mem_id": base.mem_id, "layout": base.layout.label,
            "basis": "逐项缩小（先减半再几何二分）× 现有引擎；每个模型的全部领域指标都在 ±1% 内且仍放得下才算可行；"
                     "min_ok 是实测通过的点（不是外推）。只回答「还能省多少」，面积仅在已有密度（SRAM mm²/MiB、MAC mm²/kMAC，来自密度预设或用户）时给 mm²；DRAM 带宽经 mem_eff 连续缩放（容量不变），不给 PHY 面积；DRAM 封装/堆栈数为离散项（带宽+容量同减），面积 = 位宽 × DRAM PHY 密度（若已设），不参与联合缩减。"}
    if joint:
        if unpriced not in ("last", "free"):
            raise ValueError("unpriced must be 'last' or 'free'")
        rep["joint"] = _joint_shrink(base, [r for r in out if r["key"] not in JOINT_SKIP], budget, worst, unpriced)
    rep["binding"] = _binding(out, rep.get("joint"))
    return rep


def _traffic_delta(scns, ref, setter, v, table, domain) -> dict:
    """0.187: at the reported min_ok, per model: DRAM / SRAM bytes per unit and (with a table) J per unit and its
    DRAM / SRAM parts, as ratios to the current chip.  Exposes shrinks that hold speed but move traffic (and energy)
    from SRAM to DRAM."""
    out = {}
    for mid, s in scns.items():
        try:
            p = _point(setter(s, v), table, domain)
        except ValueError:
            continue
        r0 = ref[mid]
        rat = lambda a, b: (b / a) if a else None          # noqa: E731
        e = {"dram_B": rat(r0.get("dram_B_tok"), p.get("dram_B_tok")),
             "sram_B": rat(r0.get("sram_B_tok"), p.get("sram_B_tok"))}
        if r0.get("J_tok") and p.get("J_tok"):
            e["J"] = p["J_tok"] / r0["J_tok"]
            e["J_partial"] = bool(r0.get("J_partial"))
        if r0.get("J_prefill") and p.get("J_prefill"):
            e["J_prefill"] = p["J_prefill"] / r0["J_prefill"]     # 0.188
        out[mid] = e
    return out


def _binding(rows: list, joint) -> dict:
    """0.161: which resources bind this model set — no single-resource headroom (min_ok = current) → binding; the
    rest sorted by headroom (smallest fraction first = most over-provisioned); joint back-offs = resources that only
    bind in combination."""
    tight = [r["key"] for r in rows if r["frac"] is not None and r["frac"] >= 1.0 - 1e-12]
    slack = sorted((r for r in rows if r["key"] not in tight), key=lambda r: r["frac"])
    out = {"binding": [{"key": r["key"], "label": r["label"],
                        "limited_by": r["limited_by"]} for r in rows if r["key"] in tight],
           "most_slack": [{"key": r["key"], "label": r["label"], "frac": r["frac"]} for r in slack]}
    if joint:
        out["joint_only"] = sorted({b["backoff"] for b in joint["backoffs"]} - set(tight))
    return out


def _density(budget, dens, base):
    """Density field value; 'mm2_per_dq' resolves to the HBM or non-HBM DRAM PHY density of the base memory."""
    if budget is None or not dens:
        return None
    if dens == "mm2_per_dq":
        mt = _memspec(base).mem_type
        if "HBM" in mt.upper():                          # 0.329: HBM3 / 3E have their own PHY density
            from .budget import hbm_dq_field
            dens = hbm_dq_field(budget, mt)
        else:
            dens = "mm2_per_dq_lpddr"
    return getattr(budget, dens, None)


def _unit_mm2(key: str, base: Scenario, d: float) -> float:
    """mm² per unit of a sizing resource from its density.  Densities are per 1024 (kMAC as in budget.area_estimate,
    klane, kdq); 0.162 fixes the ≤ 0.161 cores conversion that divided R·C by 1000 (+2.4 % on cores mm²)."""
    if key == "cores":
        return base.chip.rows * base.chip.cols / 1024.0 * d
    if key in ("lanes", "dequant"):
        return d / 1024.0
    if key == "dram_units":
        return _memspec(base).unit_width_bits * d
    return d


JOINT_BEAM = 3          # 0.161: beam width of the joint back-off search (the greedy path is always kept too)
JOINT_MAX_STEPS = 24


def _joint_shrink(base: Scenario, rows: list, budget, worst, unpriced: str = "last") -> dict:
    """0.160/0.161: joint sizing.  The per-resource minima interact (shared elementwise / memory time), so all minima
    together can break the ±TIE contract.  From every min_ok at once, resources are backed off one geometric step
    toward their current value (√(v·cur), snapping to cur within 3 %; integers ≥ 1) until every model holds.

    0.161 area-aware: when a density prices a resource, a step is ranked by worst-gain improvement per mm² given back
    (an unpriced step costs 0 mm² and ranks first among improving steps — its area is unknown, not free; reported as
    such).  Without densities the rank is the worst gain itself (0.160 greedy).  A beam of JOINT_BEAM states is kept
    per depth, and the 0.160 greedy path is always run, so the result is never worse than greedy by the objective:
    most priced mm² saved (else the deepest total shrink, Σ log(cur/v)), among probed passing points only.
    Terminates: all resources at current is the reference chip (gain 1)."""
    cur = {r["key"]: r["current"] for r in rows}
    start = {r["key"]: r["min_ok"] for r in rows}
    integer = {k: SIZING[k][4] for k in start}
    per_unit = {}
    for k in start:
        d = _density(budget, SIZING[k][5], base)
        if d is not None:
            per_unit[k] = _unit_mm2(k, base, d)
    priced = bool(per_unit)
    memo: dict = {}

    def W(vals):
        key = tuple(sorted(vals.items()))
        if key not in memo:
            memo[key] = worst(vals)
        return memo[key]

    def saved(vals):
        return sum((cur[k] - vals[k]) * u for k, u in per_unit.items())

    def shrink(vals):
        return sum(math.log(cur[k] / vals[k]) for k in vals if vals[k] > 0 and cur[k] > 0)

    def objective(vals):
        return (saved(vals), shrink(vals)) if priced else (shrink(vals),)

    def steps(vals):
        out = []
        for k, v in vals.items():
            if v >= cur[k]:
                continue
            nv = math.sqrt(v * cur[k])
            nv = cur[k] if nv >= 0.97 * cur[k] else nv
            if integer[k]:
                nv = float(min(cur[k], max(v + 1, int(round(nv)))))
            out.append((k, nv, {**vals, k: nv}))
        return out

    def rank(vals, k, nv, cand):
        w0, w1 = W(vals)[0], W(cand)[0]
        if not priced:
            return (w1,)
        gain = w1 - w0 if math.isfinite(w0) else (math.inf if math.isfinite(w1) else 0.0)
        if k not in per_unit:
            # 0.162: unknown area is not free.  "last" (default): unpriced steps rank after every improving priced step
            # (the area the search gives back is then priced first); "free" = ≤ 0.161 behaviour (cost 0 → first)
            if unpriced == "free":
                return (2, gain) if gain > 0 else (-1, gain)
            return (0, gain) if gain > 0 else (-1, gain)
        cost = (nv - vals[k]) * per_unit[k]
        return (1, gain / cost) if gain > 0 else (-1, gain / max(cost, 1e-12))

    ok = lambda v: W(v)[0] >= 1 - TIE
    first_w, first_why = W(start)
    found = []                                   # (vals, path) of probed passing points
    # greedy (0.160 path, area-aware rank when priced)
    vals, path = dict(start), []
    for _ in range(JOINT_MAX_STEPS * len(start)):
        if ok(vals):
            found.append((vals, path, "greedy"))
            break
        c = steps(vals)
        if not c:
            break
        k, nv, cand = max(c, key=lambda t: rank(vals, *t))
        path = path + [{"backoff": k, "to": nv, "worst_gain": W(cand)[0]}]
        vals = cand
    # beam
    beam = [(start, [])] if not ok(start) else []
    if ok(start):
        found.append((start, [], "beam"))
    for _ in range(JOINT_MAX_STEPS):
        nxt = []
        for vals, path in beam:
            for k, nv, cand in steps(vals):
                pth = path + [{"backoff": k, "to": nv, "worst_gain": W(cand)[0]}]
                if ok(cand):
                    found.append((cand, pth, "beam"))
                else:
                    nxt.append((rank(vals, k, nv, cand), objective(cand), cand, pth))
        if not nxt:
            break
        nxt.sort(key=lambda t: (t[0], t[1]), reverse=True)
        seen, beam = set(), []
        for _, _, cand, pth in nxt:
            kk = tuple(sorted(cand.items()))
            if kk not in seen:
                seen.add(kk)
                beam.append((cand, pth))
            if len(beam) >= JOINT_BEAM:
                break
    if not found:                                # cannot happen (all-current passes); keep the contract explicit
        found.append((dict(cur), [], "current"))
    greedy = next((f for f in found if f[2] == "greedy"), None)
    vals, trail, how = max(found, key=lambda f: objective(f[0]))
    w, why = W(vals)
    assert w >= 1 - TIE or how == "current"     # never report an unverified point
    res = []
    for r in rows:
        k = r["key"]
        e = {"key": k, "solo_min": r["min_ok"], "joint": vals[k], "frac": vals[k] / cur[k] if cur[k] else None,
             "priced": k in per_unit}
        if k in per_unit:
            e["mm2_saved"] = (cur[k] - vals[k]) * per_unit[k]
        res.append(e)
    return {"all_minima_hold": first_w >= 1 - TIE, "all_minima_worst_gain": first_w, "all_minima_limited_by": first_why,
            "holds": w >= 1 - TIE, "worst_gain": w, "resources": res, "backoffs": trail,
            "mm2_saved_total": saved(vals) if priced else None, "search": how, "beam": JOINT_BEAM,
            "rank": (("Δworst-gain per mm²；未定价资源排在已定价之后" if unpriced == "last" else
                      "Δworst-gain per mm²；未定价资源视为 0 mm²（优先回退）") if priced else "worst gain"),
            "unpriced_mode": unpriced, "unpriced": sorted(k for k in start if k not in per_unit),
            "greedy_mm2_saved": (saved(greedy[0]) if priced else None) if greedy else None,
            "greedy_shrink": shrink(greedy[0]) if greedy else None, "shrink": shrink(vals), "evals": len(memo)}


# ------------------------------------------------------------------ 0.262: J/token-optimal SLC size under temperature
SLC_GIB_GRID = (0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64)     # capacity probe points (GiB); 0 = no SLC
# 0.264: coarser grid embedded in insight_report (default design path) so 25/85 °C lands without a second click
INSIGHT_SLC_GIB = (0, 1, 2, 4, 8, 16, 32)


def slc_energy_report(base: Scenario, table: EnergyTable | None = None, models=None, *,
                      policy: str = "dip", gibs=None, temps=None) -> dict:
    """Sweep SLC capacity and pick the size that minimises energy per unit at the table's die temperature.

    Closes the 0.261 hot-die finding: at a high ``energy.sram_temp_C`` a GiB-scale SRAM SLC can cost more J/token than
    it saves in DRAM traffic, while still buying TPOT / latency.  This report surfaces that crossover and the
    J-optimal capacity (and the time-optimal one), so the SLC decision is an energy-vs-time trade under a temperature.

    Mechanism: for each GiB in ``gibs`` (default ``SLC_GIB_GRID``), set ``chip.slc_mib`` / ``slc_policy`` and evaluate
    via ``_point``.  Leakage uses §205.4 / §206.1 / §208.1: with ``uW_per_mib_sram_leak``, SLC MiB is always charged
    (even when idle is set — idle is assumed without this speculative SLC); shared SRAM still suppressed by idle.
    ``temps`` (e.g. (25, 85)) runs the sweep at each die temperature and returns a side-by-side compare."""
    from .energy import idle_W_of, sram_leak_uW_per_mib
    from .hardware import SLC_POLICIES
    table = table or EnergyTable()
    if policy not in SLC_POLICIES:
        raise ValueError(f"slc_policy must be one of {SLC_POLICIES}")
    models = list(models) if models is not None else list(DEFAULT_MODELS[:1])
    if not 1 <= len(models) <= MAX_MODELS:
        raise ValueError(f"models: 1..{MAX_MODELS} ids")
    domain = insight_domain(models)
    grid = tuple(gibs) if gibs is not None else SLC_GIB_GRID
    if not grid or any(not isinstance(g, (int, float)) or g < 0 or g > 256 for g in grid):
        raise ValueError("gibs: 1..N capacities in [0, 256] GiB")
    if 0.0 not in grid and 0 not in grid:
        grid = (0,) + tuple(grid)
    leak_uW = sram_leak_uW_per_mib(table)
    idle = idle_W_of(table)
    notes = []
    if leak_uW is None:
        notes.append("无 uW_per_mib_sram_leak：曲线只有 DRAM 流量变化，漏电不计入（填能耗预设 tpuv4i_7nm 可打开）")
    elif idle is not None:
        notes.append("已填 idle_W / idle_frac_tdp：共享 SRAM 漏电视为已含于 idle；SLC 仍按增量计漏电（0.263，idle 不含该 SLC）")
    if table.slc_access_ref_mib is not None:     # 0.268: deprecated √C option
        from .energy import SLC_SQRTC_DEPRECATION
        notes.append(SLC_SQRTC_DEPRECATION)
    t_C = table.sram_temp_C if table.sram_temp_C is not None else 25.0
    if leak_uW is not None and t_C <= 25.0 + 1e-9 and not temps:
        notes.append("sram_temp_C 未填或 = 25 °C（下界）；热模对比请点 25/85 °C 或设 sram_temp_C")
    if temps is not None:
        if not isinstance(temps, (list, tuple)) or not 2 <= len(temps) <= 4:
            raise ValueError("temps: 2..4 die temperatures in C")
        for t in temps:
            if not isinstance(t, (int, float)) or not 0 <= float(t) <= 125:
                raise ValueError("temps: each entry in [0, 125] C")
        by, rows = {}, []
        for t in temps:
            sub = slc_energy_report(base, replace(table, sram_temp_C=float(t)), models,
                                    policy=policy, gibs=gibs, temps=None)
            by[str(float(t))] = sub
            for m in sub["models"]:
                if m.get("error"):
                    rows.append({"temp_C": float(t), "model": m["model"], "error": m["error"]})
                else:
                    rows.append({"temp_C": float(t), "model": m["model"], "opt_J_gib": m["opt_J_gib"],
                                 "opt_J": m["opt_J"], "opt_time_gib": m["opt_time_gib"], "opt_time": m["opt_time"],
                                 "crossover_gib": m["crossover_gib"], "J0": m["J0"]})
        return {"compare": True, "temps": [float(t) for t in temps], "by_temp": by, "summary": rows,
                "policy": policy, "gibs": list(grid), "chip": base.chip.name, "mem_id": base.mem_id,
                "layout": base.layout.label, "leak_charged": leak_uW is not None,
                "notes": notes + ["multi-T compare: one curve per T, see summary"],
                "domain": domain, "basis": "slc_energy_report x each T in temps (0.263)"}
    per_model = []
    for mid in models:
        try:
            s = model_scenario(base, mid)
        except (ValueError, KeyError) as e:
            per_model.append({"model": mid, "error": str(e)})
            continue
        curve = []
        for g in grid:
            ch = replace(s.chip, slc_mib=float(g) * 1024.0, slc_policy=policy) if g > 0 else replace(s.chip, slc_mib=0.0)
            sg = s.replace("chip", ch)
            try:
                p = _point(sg, table, domain)
                r = evaluate(sg.replace("serving.phase", "decode") if domain == "llm" else sg)
                e = energy_report(r, table)
                m0 = r.stages[0].mem if r.stages else None
                curve.append({
                    "gib": float(g), "J_tok": p.get("J_tok"),
                    "tpot_ms": (p["tpot"] * 1e3) if p.get("tpot") else None,
                    "lat_s": p.get("lat"), "ttft_ms": (p["ttft"] * 1e3) if p.get("ttft") else None,
                    "fits": p["fits"], "bound": p.get("bound_d"),
                    "leak_W": e.get("sram_leak_W_card"), "leak_W_85C": e.get("sram_leak_W_card_85C"),
                    "slc_frac": (m0.slc_frac if m0 and g > 0 else 0.0),
                    "slc_frac_exp": (getattr(m0, "slc_frac_exp", 0.0) if m0 and g > 0 else 0.0),
                    # 0.267: effective SLC pJ/bit at this capacity (flat / √C / distance model)
                    "slc_pJ_bit": (e.get("slc_access") or {}).get("pJ_bit", table.pJ_bit_slc) if g > 0 else None,
                    "slc_wire_mm": (e.get("slc_access") or {}).get("wire_mm") if g > 0 else None,
                })
            except ValueError as err:
                curve.append({"gib": float(g), "error": str(err)})
        ok = [c for c in curve if c.get("J_tok") is not None and c.get("fits")]
        j0 = next((c["J_tok"] for c in ok if c["gib"] == 0), None)
        best_J = min(ok, key=lambda c: c["J_tok"]) if ok else None
        time_key = "tpot_ms" if domain == "llm" else "lat_s"
        timed = [c for c in ok if c.get(time_key) is not None]
        best_t = min(timed, key=lambda c: c[time_key]) if timed else None
        cross = None
        if j0 is not None:
            for c in sorted((c for c in ok if c["gib"] > 0), key=lambda c: c["gib"]):
                if c["J_tok"] > j0 * (1 + 1e-12):
                    cross = c["gib"]
                    break
        per_model.append({"model": mid, "curve": curve, "J0": j0,
                          "opt_J_gib": best_J["gib"] if best_J else None,
                          "opt_J": best_J["J_tok"] if best_J else None,
                          "opt_time_gib": best_t["gib"] if best_t else None,
                          "opt_time": best_t.get(time_key) if best_t else None,
                          "crossover_gib": cross, "time_metric": time_key})
    return {"domain": domain, "models": per_model, "policy": policy, "gibs": list(grid),
            "sram_temp_C": t_C, "leak_charged": leak_uW is not None,
            "chip": base.chip.name, "mem_id": base.mem_id, "layout": base.layout.label, "notes": notes,
            "basis": "逐 GiB 探针 × evaluate + energy_report；J 最优 = argmin J/单位；时间最优 = argmin TPOT/延迟；"
                     "交叉点 = 最小正容量使 J > 无 SLC。SLC 漏电在有 uW_per_mib_sram_leak 时始终计入（0.263；idle 不含该 SLC）。"}

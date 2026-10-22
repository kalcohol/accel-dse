"""0.320: design verdict report — one design of an energy DSE, everything the tool knows about it in one place.

Every number is an item {key, label, value, unit, tag, src}: tag ∈ TAGS (the five evidence labels, plus 用户输入 for
values the user typed); src names the source or the model term that produced it.  A value the tool does not have is
``None`` with a ``missing`` reason — never 0, never a borrowed default.  Model outputs are 「推导」 (computed by this
tool from the labelled inputs listed next to them).

Near-tie warnings are mechanism checks, not thresholds:
  * mono / split $/M verdict of this design's group: inputs that flip it inside their sourced ranges (0.318) and the
    verdict at the joint corners (0.319), straight from the DSE;
  * this design vs its $/M rival (the cheapest other priced row) and vs the best alternative memory of the same
    design: the same one-at-a-time scan + joint corners (core/tco_sens) on the pair;
  * margins (SLO, thermal flux / package W, shoreline, power cap) within the DSE's own tie (default 1 %).
"""
from __future__ import annotations

import math
import re

TAGS = ("来源", "二手", "推导", "假设", "近似", "用户输入")
MODEL_SRC = "accel_dse 代价模型（时间 / 放得下）"
_TAG_RE = re.compile("「(来源|二手|推导|假设|近似)」")


def tags_in(src: str | None) -> str | None:
    """The evidence labels a source string carries, joined (「来源」…「推导」 → 来源/推导); None when it names none."""
    seen = []
    for t in _TAG_RE.findall(src or ""):
        if t not in seen:
            seen.append(t)
    return "/".join(seen) or None


def item(key, label, value, unit="", tag=None, src="", missing=None) -> dict:
    if value is None:
        return {"key": key, "label": label, "value": None, "unit": unit, "tag": None, "src": src,
                "missing": missing or "工具无此值"}
    if isinstance(value, float) and not math.isfinite(value):
        return {"key": key, "label": label, "value": None, "unit": unit, "tag": None, "src": src,
                "missing": "非有限值（不可行或未定价）"}
    return {"key": key, "label": label, "value": value, "unit": unit, "tag": tag or "推导", "src": src}


def _unit(k: str) -> str:
    return ("pJ/bit/mm" if k.endswith("_mm_onchip") else "pJ/bit" if k.startswith("pJ_bit") else
            "pJ" if k.startswith("pJ_") else "")


def _sec(sid, title, items, status=None, notes=()):
    return {"id": sid, "title": title, "status": status, "items": items, "notes": [n for n in notes if n]}


def _preset_items(presets: dict, pid: str | None, keys, user: dict, what: str) -> list[dict]:
    out = []
    f = (presets.get(pid) or {}).get("fields", {}) if pid else {}
    for k in keys:
        if user.get(k) is not None:
            out.append(item(k, k, user[k], _unit(k), tag="用户输入", src=f"{what}：用户覆盖"))
        elif k in f:
            v, t, s = f[k]
            out.append(item(k, k, v, _unit(k), tag=t, src=f"{what} {pid}：{s}"))
    return out


def _row_index(r: dict, row: int | None) -> int:
    rows = r["rows"]
    if row is not None:
        if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < len(rows):
            raise ValueError(f"report_row must be an integer in [0, {len(rows)})")
        return row
    if r.get("tco") and r["tco"].get("best") is not None:
        return r["tco"]["best"]
    b = (r.get("best") or {}).get("perf_W")
    for i, x in enumerate(rows):
        if x is b:
            return i
    if r.get("front"):
        return r["front"][0]
    raise ValueError("no feasible design to report (report_row needed)")


def _base(r: dict, body: dict):
    """The sensitivity base the DSE used (tco params, cost terms, D0, user prices) — rebuilt from the result and body,
    checked against the row's own $/M before use (tco_sens.row_usd raises / mismatch → None)."""
    from .tco import tco_params
    t = r.get("tco")
    if t is None or r.get("cost") is None:
        return None
    from .energy_dse import _cost_params
    cost = body.get("cost")
    return {"tco": tco_params(body.get("tco")), "cost": _cost_params(cost), "d0": body.get("d0_per_cm2") or 0.0,
            "prices": body.get("mem_usd_per_GB")}


def _pair_check(a: dict, b: dict, base: dict, tie: float) -> dict:
    """Does "a is cheaper than b on $/M" hold over every sourced range (one at a time) and at the joint corners?"""
    from .tco_sens import INPUTS, applicable, corner, input_range, row_usd, scan_flip
    ua, ub = row_usd(a, base), row_usd(b, base)

    def who(ov):
        return "this" if row_usd(a, base, ov) <= row_usd(b, base, ov) else "rival"
    flips, skipped = [], []
    for k in INPUTS:
        if not applicable(k, base):
            continue
        if k == "mem_usd_per_GB" and a["mem_type"] != b["mem_type"]:
            skipped.append("内存 $/GB（两者内存类型不同，单输入扫描无共同区间；见联合角点）")
            continue
        if k == "d0" and base.get("d0", 0) == 0:
            continue
        rg = input_range(k, a["mem_type"], base)
        if rg is None:
            continue
        bv = None
        from .tco_sens import base_values
        bv = base_values(base, a["mem_type"]).get(k)
        s = scan_flip(who, k, rg, bv, n=32, iters=40)
        if s["flips"]:
            flips.append({"input": k, "label": s["label"], "thresholds": s["thresholds"]})
    cor = {sd: who(corner(base, sd)) for sd in ("adverse", "favourable")}
    return {"d": ub / ua - 1, "this_usd": ua, "rival_usd": ub, "base": who({}), "flips": flips, "corners": cor,
            "skipped": skipped, "within_tie": abs(ub / ua - 1) <= tie}


def _desc(x: dict) -> str:
    return (f"{x['engines']}×{x['array']} · SRAM {x['sram_mib']:g} MiB · {x['topology']} · {x['mem_id']}"
            + (f" · SLC {x['slc_mib']:g} MiB" if x.get("slc_mib") else "")
            + (f" · {x['dies']} 晶粒 {x.get('d2d_std')}" if (x.get("dies") or 1) > 1 else "")
            + (f" · f×{x['f_op']:g}" if x.get("f_op") not in (None, 1, 1.0) else ""))


def design_report(r: dict, body: dict, row: int | None = None) -> dict:
    """The report of row ``row`` (default: cheapest $/M, else best perf/W) of insight_dse result ``r``."""
    from .. import __version__
    from .budget import DENSITY_PRESETS
    from .energy import ENERGY_PRESETS
    from .energy_dse import ASM_SRC, _tco_ok, asm_note
    from .thermal import COOLING
    i = _row_index(r, row)
    rows = r["rows"]
    x = rows[i]
    tie = r.get("tie") or 0.01
    unit = x.get("unit") or "单位"
    secs, warn, margins = [], [], []
    sv = ((body.get("scenario") or {}).get("serving") or {})

    # 1 design ------------------------------------------------------------------------------------------------
    dens = (body.get("budget") or {}).get("density_preset")
    it = [item("chip", "芯片基线", r["chip"], tag="用户输入", src="chip_preset"),
          item("model", "模型", r["model"], tag="用户输入", src="scenario.model"),
          item("phase", "阶段", r["phase"], tag="用户输入", src="serving.phase"),
          item("batch", "batch", sv.get("batch"), tag="用户输入", src="serving.batch", missing="未给 batch（用默认）"),
          item("ctx", "上下文", sv.get("ctx"), "token", tag="用户输入", src="serving.ctx", missing="未给（用默认）"),
          item("array", "引擎 × 阵列", f"{x['engines']} × {x['array']}", tag="用户输入", src="DSE 网格点"),
          item("macs", "MAC 总数", x.get("macs"), tag="用户输入", src=f"DSE 网格点（MAC × {x.get('mac_x')}）"),
          item("sram_mib", "SRAM", x.get("sram_mib"), "MiB", tag="用户输入", src="DSE 网格点"),
          item("slc_mib", "SLC", x.get("slc_mib"), "MiB", tag="用户输入", src="DSE 网格点"),
          item("topology", "NoC 拓扑", x.get("topology"), tag="用户输入", src="DSE 网格点"),
          item("mem_id", "外存", x.get("mem_id"), tag="用户输入", src="DSE 网格点"),
          item("mem_GB", "外存容量", x.get("mem_GB"), "GB", tag="来源", src="内存目录（器件规格）"),
          item("mem_GBps", "外存带宽（峰值）", x.get("mem_GBps"), "GB/s", tag="来源", src="内存目录（器件规格）"),
          item("dies", "晶粒数", x.get("dies") or 1, tag="用户输入", src="DSE 晶粒轴"),
          item("d2d_std", "D2D 接口", x.get("d2d_std") if (x.get("dies") or 1) > 1 else "—", tag="用户输入", src="DSE D2D 轴"),
          item("f_ghz", "运行频率", x.get("f_ghz"), "GHz", src="工作点 × DVFS 封顶（若封顶）"),
          item("mm2", "逻辑晶粒面积（Σ）", x.get("mm2") if r.get("area") else None, "mm²", tag=None,
               src=f"面积密度预设 {dens}：" + "；".join(f"{k} {v[1]}" for k, v in
                                                     (DENSITY_PRESETS.get(dens) or {}).get("fields", {}).items()),
               missing="未给面积密度（budget.density_preset）")]
    secs.append(_sec("design", "设计", it, notes=[_desc(x)]))

    # 2 performance at SLO ---------------------------------------------------------------------------------------
    slo = r.get("slo") or {}
    it = [item("thr", f"吞吐（{unit}/s，每封装）", x.get("thr"), f"{unit}/s", src=MODEL_SRC),
          item("tpot_ms", "TPOT / 步时间", x.get("tpot_ms"), "ms", src=MODEL_SRC),
          item("bound", "瓶颈", x.get("bound"), src=MODEL_SRC),
          item("fits", "放得下（权重 + KV）", x.get("fits"), src=MODEL_SRC),
          item("slo_tpot_ms", "SLO：TPOT 上限", slo.get("tpot_ms"), "ms", tag="用户输入", src="slo_tpot_ms",
               missing="未设 SLO（slo_tpot_ms）：只报吞吐，不判达标")]
    st = None
    if slo.get("tpot_ms") and x.get("tpot_ms"):
        m = slo["tpot_ms"] / x["tpot_ms"] - 1
        it.append(item("slo_margin", "SLO 余量（上限 / TPOT − 1）", m, "", src="SLO ÷ 模型 TPOT"))
        margins.append(("SLO TPOT", m))
        st = "ok" if m >= 0 else "fail"
    if slo.get("ttft_ms"):
        it.append(item("slo_ttft_ms", "SLO：TTFT 上限", slo["ttft_ms"], "ms", tag="用户输入", src="slo_ttft_ms"))
        it.append(item("ttft_ms", "TTFT", x.get("ttft_ms"), "ms", src=MODEL_SRC,
                       missing="DSE 行不含 TTFT（DSE 以 TTFT SLO 剔除不达标点；本行在表中即达标）"))
    if x.get("capped"):
        it.append(item("thr_nominal", "未封顶吞吐", x.get("thr_nominal"), f"{unit}/s", src=MODEL_SRC))
    secs.append(_sec("perf", "SLO 下的性能", it, st))

    # 3 energy ----------------------------------------------------------------------------------------------------
    en = body.get("energy") or {}
    epid = en.get("energy_preset")
    share = sorted((x.get("share") or {}).items(), key=lambda kv: -kv[1])
    it = [item("J_per_unit", f"J/{unit}", x.get("J_per_unit"), "J", src="动作计数 × 每动作能耗（下列输入）",
               missing="未给能耗表"),
          item("perf_W", f"{unit}/J（perf/W）", x.get("perf_W"), f"{unit}/J", src="1 / (J/单位)"),
          item("W_card", "卡功耗（能耗口径）", x.get("W_card"), "W", src="J/单位 × 吞吐 + 静态"),
          item("static_W", "静态功耗", x.get("static_W"), "W", src="SRAM 漏电 + HBM 待机 / 刷新（能耗预设）")]
    it += [item(f"share_{k}", f"能耗占比：{k}", v, "", src="能耗分解") for k, v in share[:4] if v > 0]
    it += _preset_items(ENERGY_PRESETS, epid, ("pJ_mac", "pJ_bit_sram", "pJ_bit_dram_hbm", "pJ_bit_phy_hbm",
                                               "pJ_bit_dram_lpddr", "pJ_bit_mm_onchip", "pJ_bit_hop_noc", "pJ_bit_d2d"),
                        en, "能耗预设")
    secs.append(_sec("energy", "能耗", it, notes=[None if epid else "未用能耗预设：每动作能耗为用户输入"]))

    # 4 shoreline -------------------------------------------------------------------------------------------------
    sh = x.get("shore") or {}
    it = [item("shoreline", "岸线判定", x.get("shoreline"), src="岸线模型（0.312）：外存 / D2D / I/O 边长 vs 晶粒周长",
               missing="未算岸线（缺面积密度）"),
          item("buildable", "可建", x.get("buildable"), src="岸线 + 热（0.312 / 0.313）"),
          item("shore_mem", "外存边", sh.get("mem"), src="岸线模型"), item("shore_io", "I/O 边", sh.get("io"), src="岸线模型"),
          item("shore_d2d", "D2D 边", sh.get("d2d") if (x.get("dies") or 1) > 1 else "—", src="岸线模型"),
          item("used_mm", "已用岸线", sh.get("used_mm"), "mm", src="岸线模型"),
          item("perimeter_mm", "可用周长", sh.get("perimeter_mm"), "mm", src="晶粒形状（光罩 26 × 33 mm「来源」内）")]
    if sh.get("used_mm") is not None and sh.get("perimeter_mm"):
        m = 1 - sh["used_mm"] / sh["perimeter_mm"]
        it.append(item("shore_margin", "岸线余量（1 − 已用 / 周长）", m, src="岸线模型"))
        margins.append(("岸线", m))
    secs.append(_sec("shoreline", "岸线可建性", it, x.get("shoreline")))

    # 5 thermal / power -------------------------------------------------------------------------------------------
    th = x.get("thermal") or {}
    cool = r.get("cooling")
    cs = (COOLING.get(cool) or {}).get("src") if cool else None
    it = [item("th_status", "热判定", th.get("status"), src="thermal（0.313）"),
          item("pkg_W", "封装功耗（能耗 + 静态底 + 稳压器损耗）", th.get("pkg_W"), "W", src="thermal.package_power"),
          item("die_W", "晶粒发热", th.get("die_W"), "W", src="thermal"),
          item("flux_W_mm2", "封装热流", th.get("flux_W_mm2"), "W/mm²", src="封装功耗 ÷ 逻辑晶粒面积"),
          item("cooling", "冷却方案", cool, tag="用户输入", src="cooling", missing="未选冷却方案：热只报告、不判"),
          item("cool_pkg_W", "冷却上限：封装功耗", (COOLING.get(cool) or {}).get("pkg_W") if cool else None, "W",
               tag=tags_in(cs), src=cs or "", missing="未选冷却方案"),
          item("cool_flux", "冷却上限：热流", (COOLING.get(cool) or {}).get("flux_W_mm2") if cool else None, "W/mm²",
               tag=tags_in(cs), src=cs or "", missing="未选冷却方案"),
          item("flux_ratio", "热流 / 上限", th.get("flux_ratio") if cool else None, src="thermal", missing="未选冷却方案"),
          item("power_cap_W", "功耗预算", r.get("power_cap_W"), "W", tag="用户输入", src=r.get("power_cap_src") or "",
               missing="未填功耗预算（budget.power_W_card / energy.tdp_W）：不做功耗约束"),
          item("capped", "DVFS 封顶", x.get("capped"), src="powercap（0.302 / 0.316，封装口径）")]
    if cool and th.get("flux_ratio") is not None:
        margins.append(("热流", 1 - th["flux_ratio"]))
        lim = (COOLING.get(cool) or {}).get("pkg_W")
        if lim and th.get("pkg_W"):
            margins.append(("封装功耗", 1 - th["pkg_W"] / lim))
    secs.append(_sec("thermal", "热 / 功耗", it, th.get("status")))

    # 6 yield and cost per good package -----------------------------------------------------------------------------
    cp = r.get("cost")
    if cp is None:
        secs.append(_sec("cost", "良率与每好封装成本", [item("usd_good_pkg", "每好封装成本", None, "$",
                                                       missing="未选成本预设（cost）")], "missing"))
    else:
        d0 = body.get("d0_per_cm2")
        csrc = cp.get("src", "")
        it = [item("d0", "缺陷密度 D0", d0, "/cm²", tag="用户输入", src="d0_per_cm2",
                   missing="未给 D0：晶粒良率按 1 计（缺陷良率未建模）"),
              item("die_yield", "晶粒良率（Poisson e^(−A·D0)）", x.get("die_yield") if d0 else None, "",
                   src="KGD 面积 × D0", missing="缺 D0"),
              item("wafer_usd", "晶圆价", cp.get("wafer_usd"), "$", tag=tags_in(csrc.split("；")[0]), src=csrc.split("；")[0]),
              item("dpw", "每晶圆毛晶粒", x.get("dpw"), "", src="πR²/A − π·2R/√(2A)「来源」，R = 150 mm"),
              item("good_die_usd", "每好晶粒", x.get("good_die_usd"), "$", src="晶圆价 / DPW / 良率"),
              item("si_usd", "硅（好晶粒 × 晶粒数）", x.get("si_usd"), "$", src="同上"),
              item("pkg_usd", "封装（$/mm² × 封装面积）", x.get("pkg_usd"), "$", tag=tags_in(csrc), src=csrc),
              item("test_usd", "测试 / 组装", x.get("test_usd"), "$", tag=tags_in(csrc), src=csrc),
              item("asm_Y", "封装 / 键合良率", x.get("asm_Y"), "", src=asm_note(cp)),
              item("kgd_scrap_usd", "失败封装报废的 KGD", x.get("kgd_scrap_usd"), "$", src="chip-first 报废（0.319）"),
              item("mem_scrap_k", "HBM 报废系数（进 TCO 内存项）", x.get("mem_scrap_k"), "", src=ASM_SRC["scrap"]),
              item("usd_good_pkg", "每好封装成本（不含内存器件）", x.get("usd_good_pkg"), "$",
                   tag="推导" if d0 else "假设", src="硅 + 报废 + 封装 + 测试")]
        secs.append(_sec("cost", "良率与每好封装成本", it, notes=[None if d0 else "缺 D0：成本按无缺陷晶粒计「假设」"]))

    # 7 TCO with sensitivity and joint corners ------------------------------------------------------------------------
    tp = r.get("tco")
    t = x.get("tco") or {}
    base = None
    if tp is None:
        secs.append(_sec("tco", "TCO", [item("usd_per_Munit", f"$/M {unit}", None, "$",
                                             missing="未选 TCO 预设（tco，需 cost）")], "missing"))
    else:
        ps = tp.get("src") or {}
        tm = ("TCO 缺项：" + "；".join(t["missing"])) if t.get("missing") else \
            ("该点不可行 / 不可建，未定价" if not t else None)
        it = [item("usd_per_Munit", f"$/M {unit}", t.get("usd_per_Munit"), "$", src="(年化资本 + 年电费) ÷ 年服务量",
                   missing=tm),
              item("capex_usd", "资本（封装 + 内存 + 系统）", t.get("capex_usd"), "$", src="§262", missing=tm),
              item("mem_usd", "内存（GB × $/GB × 报废系数）", t.get("mem_usd"), "$", src="§262 / §264", missing=tm),
              item("mem_usd_per_GB", "内存 $/GB", t.get("mem_usd_per_GB"), "$/GB", tag=t.get("mem_price_src"),
                   src="tco.MEM_USD_PER_GB / 用户输入", missing=f"{x.get('mem_type')} 无公开价（tco.mem_usd_per_GB 可填）"),
              item("energy_usd_per_yr", "年电费", t.get("energy_usd_per_yr"), "$/年", src="封装功耗 × PUE × $/kWh",
                   missing=tm),
              item("kWh_per_yr", "年用电", t.get("kWh_per_yr"), "kWh", src="同上", missing=tm)]
        for k, lab, u in (("life_years", "寿命", "年"), ("pue", "PUE", ""), ("usd_per_kWh", "电价", "$/kWh"),
                          ("capital_rate", "资金成本", ""), ("system_usd_per_card", "系统分摊", "$/卡")):
            s = ps.get(k) if isinstance(ps, dict) else None
            it.append(item(k, lab, tp.get(k), u, tag=tags_in(s) or "假设", src=s or f"TCO 预设 {tp.get('preset')}"))
        sens, notes = None, []
        if _tco_ok(x):
            try:
                base = _base(r, body)
                from .tco_sens import tornado
                sens = tornado(x, base) if base else None
            except (ValueError, KeyError) as e:
                notes.append(f"敏感性未算：{e}")
        sec = _sec("tco", "TCO", it, notes=notes)
        if sens:
            sec["sens"] = {"bars": [{k: b[k] for k in ("input", "label", "lo", "hi", "tag_lo", "tag_hi", "d_lo", "d_hi", "src")}
                                    for b in sens["bars"]], "corners": sens["corners"], "skipped": sens["skipped"]}
        secs.append(sec)

    # 8 near ties ------------------------------------------------------------------------------------------------------
    from .tco_sens import compare_key
    ck = compare_key(x)
    for e in ((r.get("chiplet") or {}).get("compare") or []):
        if compare_key(e) != ck or not e.get("verdict_tco"):
            continue
        for f in e.get("tco_flips") or []:
            for th_ in f["thresholds"]:
                warn.append({"kind": "mono_split", "text": f"单片 / 拆分 $/M 结论（{e['verdict_tco']}）在 {f['label']} = "
                             f"{th_['at']:.4g} 处 {th_['from']} → {th_['to']}（有出处区间内，一次只动一个）"})
        for sd, v in (e.get("tco_corners") or {}).items():
            if v != e["verdict_tco"]:
                warn.append({"kind": "mono_split_corner", "text": f"单片 / 拆分 $/M 结论在联合{'不利' if sd == 'adverse' else '有利'}"
                             f"角点变为 {v}（本点 {e['verdict_tco']}）"})
    rivals = {}
    if base is not None and _tco_ok(x):
        pr = [j for j, y in enumerate(rows) if j != i and _tco_ok(y)]
        if pr:
            rivals["cheapest_other"] = min(pr, key=lambda j: rows[j]["tco"]["usd_per_Munit"])
        same = [j for j in pr if rows[j]["mem_id"] != x["mem_id"] and all(
            rows[j].get(k) == x.get(k) for k in ("mac_x", "sram_x", "side", "topology", "slc_mib", "f_op",
                                                 "noc_link_Bpc", "dies", "d2d_std", "clusters"))]
        if same:
            rivals["other_memory"] = min(same, key=lambda j: rows[j]["tco"]["usd_per_Munit"])
    checks = []
    for kind, j in rivals.items():
        c = _pair_check(x, rows[j], base, tie)
        c.update({"kind": kind, "rival_row": j, "rival": _desc(rows[j])})
        checks.append(c)
        lab = "最便宜的其他设计" if kind == "cheapest_other" else "同设计换外存的最优方案"
        rel = "更便宜" if c["base"] == "this" else "更贵"
        if c["within_tie"]:
            warn.append({"kind": kind, "text": f"与{lab}（第 {j} 行 {rows[j]['mem_id']}）$/M 相差 {c['d']:+.1%}，在持平阈 {tie:.0%} 内"})
        for f in c["flips"]:
            for th_ in f["thresholds"]:
                warn.append({"kind": kind, "text": f"本设计比{lab}（第 {j} 行）{rel}；{f['label']} = {th_['at']:.4g} 时翻转"
                             "（有出处区间内，一次只动一个）"})
        for sd, v in c["corners"].items():
            if v != c["base"]:
                warn.append({"kind": kind, "text": f"本设计比{lab}（第 {j} 行）{rel}；联合{'不利' if sd == 'adverse' else '有利'}角点下翻转"})
    for lab, m in margins:
        if m < 0:
            warn.append({"kind": "margin", "text": f"{lab}：超限 {m:+.1%}"})
        elif m <= tie:
            warn.append({"kind": "margin", "text": f"{lab}余量 {m:.1%} ≤ 持平阈 {tie:.0%}"})

    missing = [{"section": s["id"], "key": it_["key"], "label": it_["label"], "reason": it_["missing"]}
               for s in secs for it_ in s["items"] if it_["value"] is None]
    verdict = {
        "feasible": bool(x.get("feasible")), "buildable": x.get("buildable"), "shoreline": x.get("shoreline"),
        "thermal": th.get("status"), "slo": st, "fronts": {"thr_J": bool(x.get("front3")), "cost": bool(x.get("front_cost")),
                                                           "tco": bool(x.get("front_tco"))},
        "best_tco": bool(x.get("best_tco")), "usd_per_Munit": t.get("usd_per_Munit"), "n_near_ties": len(warn)}
    return {"report": "design_verdict", "version": __version__, "hash": r.get("hash"), "row": i, "unit": unit,
            "design": _desc(x), "verdict": verdict, "sections": secs, "near_ties": warn, "pair_checks": checks,
            "margins": [{"what": a, "margin": b} for a, b in margins], "missing": missing, "tie": tie,
            "basis": r.get("basis"), "notes": r.get("notes") or [],
            "rules": "每个数带标签（来源 / 二手 / 推导 / 假设 / 近似 / 用户输入）与出处；缺的输入列为缺项，不补 0、不借用默认"}


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        return f"{v:,.0f}" if abs(v) >= 1e4 else f"{v:.4g}"
    return str(v)


def _md_cell(s) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def to_markdown(rep: dict) -> str:
    v = rep["verdict"]
    L = [f"# 设计结论报告 — {rep['design']}", "",
         f"accel_dse {rep['version']} · 第 {rep['row']} 行 · hash {rep['hash']} · 持平阈 {rep['tie']:.0%}", "",
         "## 结论摘要", "",
         f"- 可行：{_fmt(v['feasible'])}；可建：{_fmt(v['buildable'])}（岸线 {_fmt(v['shoreline'])}，热 {_fmt(v['thermal'])}）",
         f"- SLO：{ {'ok': '达标', 'fail': '不达标', None: '未设 SLO'}[v['slo']] }",
         f"- 前沿：吞吐 / J {_fmt(v['fronts']['thr_J'])}；成本 {_fmt(v['fronts']['cost'])}；TCO {_fmt(v['fronts']['tco'])}"
         + ("；$/M 最低" if v["best_tco"] else ""),
         f"- $/M {rep['unit']}：{_fmt(v['usd_per_Munit'])}",
         f"- 近似持平 / 翻转警告：{v['n_near_ties']} 条；缺项：{len(rep['missing'])} 项", ""]
    for s in rep["sections"]:
        L += [f"## {s['title']}" + (f"（{s['status']}）" if s.get("status") else ""), "",
              "| 项 | 值 | 单位 | 标签 | 出处 |", "|---|---|---|---|---|"]
        for it in s["items"]:
            val = _fmt(it["value"]) if it["value"] is not None else f"缺：{it['missing']}"
            L.append(f"| {_md_cell(it['label'])} | {_md_cell(val)} | {_md_cell(it['unit'])} | {_md_cell(it['tag'] or '—')} | "
                     f"{_md_cell(it['src'])} |")
        if s.get("sens"):
            sn = s["sens"]
            L += ["", "敏感性（一次只动一个输入，有出处区间两端；Δ$/M 相对基值）：", "",
                  "| 输入 | 低端 | 标签 | Δ | 高端 | 标签 | Δ |", "|---|---|---|---|---|---|---|"]
            for b in sn["bars"]:
                L.append(f"| {b['label']} | {_fmt(b['lo'])} | {b['tag_lo']} | {b['d_lo']:+.1%} | {_fmt(b['hi'])} | "
                         f"{b['tag_hi']} | {b['d_hi']:+.1%} |")
            c = sn["corners"]
            L += ["", f"联合角点（所有适用输入同时取不利 / 有利端）：不利 {c['adverse']['d']:+.1%}，有利 {c['favourable']['d']:+.1%}"]
            if sn["skipped"]:
                L.append("未变动：" + "；".join(sn["skipped"]))
        for n in s.get("notes") or []:
            L += ["", f"> {n}"]
        L.append("")
    L += ["## 近似持平 / 翻转警告", ""]
    L += [f"- {w['text']}" for w in rep["near_ties"]] or ["- 无（有出处区间内、联合角点下各结论不翻转；余量均大于持平阈）"]
    if rep["pair_checks"]:
        L += ["", "| 对比 | 对手 | Δ$/M（对手 / 本 − 1） | 翻转输入 | 不利角点 | 有利角点 |", "|---|---|---|---|---|---|"]
        for c in rep["pair_checks"]:
            L.append(f"| {'最便宜的其他设计' if c['kind'] == 'cheapest_other' else '换外存'} | 第 {c['rival_row']} 行 "
                     f"{_md_cell(c['rival'])} | {c['d']:+.2%} | {', '.join(f['label'] for f in c['flips']) or '无'} | "
                     f"{'本' if c['corners']['adverse'] == 'this' else '对手'}更便宜 | "
                     f"{'本' if c['corners']['favourable'] == 'this' else '对手'}更便宜 |")
    L += ["", "## 余量", ""] + ([f"- {m['what']}：{m['margin']:+.1%}" for m in rep["margins"]] or ["- 无可算余量"])
    L += ["", "## 缺项", ""] + ([f"- {m['label']}：{m['reason']}" for m in rep["missing"]] or ["- 无"])
    L += ["", "## 口径", "", rep["rules"], "", rep.get("basis") or ""] + [f"- {n}" for n in rep["notes"]]
    return "\n".join(L) + "\n"

"""0.318: TCO sensitivity — which sourced input moves $/M unit most, and does a $/M verdict flip inside the sourced
ranges.

Mechanism: throughput, package power and the idle floor of a row are fixed (they come from the cost / energy model at
the operating point; none of the inputs below changes them).  Only cost terms move:

    package $ = package_cost(wafer $, D0 → Poisson yield, CoWoS $/mm² × scale, test)   (0.310, same function)
    memory $  = GB × $/GB (row's memory type)
    $/M unit  = tco_point(life, PUE, $/kWh, capital rate, host $ / W)                   (0.317, same function)

``tornado`` varies one input at a time across its sourced range (all others at the design's base values) and sorts by
the larger |Δ $/M|; 0.319: ``corner`` puts every input at its adverse (or favourable) end together.  ``scan_flip`` scans one input across its range for any verdict function and bisects the value
where the verdict first changes (one-at-a-time; joint corners are not searched).  Every range end carries its tag."""
from __future__ import annotations

import math

# input → (lo, hi, tag_lo, tag_hi, source).  host = 0 → server_2026 per-card share ($7,500 + 575 W) as one input.
RANGES = {
    "wafer_usd": (16000.0, 21300.0, "二手", "二手",
                  "N5/N4 晶圆 $16,000–21,300（Silicon Analysts 区间，引 CSET 2020 / Morgan Stanley 2025「二手」；"
                  "Morgan Stanley 经 Tom's Hardware：4 nm 约 $18k → 2025 约 $20k「二手」）；默认 $16,988（CSET）"),
    "d0": (0.05, 0.20, "假设", "二手",
           "D0 上端 0.20 /cm²：TSMC 2020 技术论坛 N5 量产前约 0.2「二手」（Moore Morris 转述）；N5 爬坡约 0.10「二手」"
           "（AnandTech）；下端 0.05 为成熟工艺「假设」（无公开 N5 成熟值）"),
    "pkg_scale": (1000.0 / 1100.0, 1200.0 / 1100.0, "二手", "二手",
                  "CoWoS 每 mm² 价 × 0.909–1.091：B200 CoWoS-L $1,000–1,200（Epoch AI 2025-12「二手」）÷ 默认 $1,100"
                  "「推导」；CoWoS-S 无公开区间，借同一相对区间「假设」；封装 / 键合良率另见载板 / 贴装良率（0.319）"),
    "mem_usd_per_GB": (None, None, None, None, "按该行内存类型取 MEM_RANGES（见 tco.MEM_USD_PER_GB 来源）"),
    "usd_per_kWh": (0.0546, 0.1385, "二手", "来源",
                    "$/kWh：下端 5.46 ¢ = EIA 州级工业电价 12 个月最低（新墨西哥，经 DC Hub 汇总「二手」）；"
                    "上端 13.85 ¢ = EIA 全国商业电价滚动 12 个月「来源」；默认 8.91 ¢（EIA 全国工业）"),
    "pue": (1.09, 1.58, "来源", "来源",
            "PUE：下端 1.09 = Google 2025 全舰队 TTM「来源」；上端 1.58 = Uptime 年度调查 2021–2025 最高（2023）「来源」；"
            "默认 1.54（Uptime 2025）"),
    "life_years": (4.0, 6.0, "二手", "来源",
                   "寿命：下端 4 年（SemiAnalysis / Meta 集群分析「二手」）；上端 6 年（Alphabet 10-K「来源」）"),
    "host": (0.0, 1.0, "假设", "推导",
             "主机 / 网络分摊：0（只算加速器）→ $7,500 + 575 W / 卡（server_2026，SemiAnalysis H100 系统价与 DGX H100 "
             "铭牌功耗「推导」）"),
    "capital_rate": (0.0, 0.0933, "假设", "二手", "资金成本：0（直线摊销）→ 9.33 %（Meta 集群分析 WACC「二手」）"),
    "carrier_yield": (None, None, None, None, "按成本预设的载板取 CARRIER_RANGES"),
    "attach_yield": (0.995, 1.0, "推导", "假设",
                     "每堆栈 / 晶粒贴装良率：下端 0.995 = Epoch CoWoS-S 下端 0.95 ÷ 0.985 开 7 次方「推导」；上端 1.0（无贴装损失）"
                     "「假设」；Chiplet Actuary 硅中介层 0.95 / 芯粒「二手」与 Epoch CoWoS-S 95–99 % 不相容（0.95^7 = 0.70），不取"),
}
# 0.319: carrier (CoWoS) assembly yield range per carrier
CARRIER_RANGES = {
    "cowos_s": (0.963, 0.99, "推导", "二手", "CoWoS-S 载板良率：Epoch 封装良率下端 0.95 ÷ 0.998^7「推导」→ TSMC 98–99 % 上端「二手」"),
    "cowos_l": (0.663, 0.985, "推导", "二手", "CoWoS-L 载板良率：Epoch 下端 0.65 ÷ 0.998^10「推导」→ TSMC 2026 称 98–99 %（5.5× 光罩 CoWoS，"
                "可能只指中介层）取 0.985「二手」"),
}
LABELS = {"wafer_usd": "晶圆 $", "d0": "D0", "pkg_scale": "CoWoS $/mm²", "mem_usd_per_GB": "内存 $/GB",
          "usd_per_kWh": "$/kWh", "pue": "PUE", "life_years": "寿命", "host": "主机分摊", "capital_rate": "资金成本",
          "carrier_yield": "载板良率", "attach_yield": "贴装良率"}
INPUTS = tuple(RANGES)
COST_INPUTS = ("wafer_usd", "d0", "pkg_scale", "carrier_yield", "attach_yield")   # need a cost model (not a user $)
ASM_INPUTS = ("carrier_yield", "attach_yield")                 # 0.319: only with asm_model "component"
LOW_IS_ADVERSE = ("life_years", "carrier_yield", "attach_yield")   # all other inputs raise $/M at their high end
HOST_USD, HOST_W = 7500.0, 575.0


def mem_range(mem_type: str):
    from .tco import MEM_RANGES
    return MEM_RANGES.get(mem_type)


def base_values(base: dict, mem_type: str) -> dict:
    """The design's own value of every input (what the tornado varies away from)."""
    from .tco import mem_price
    p = base["tco"]
    cp = base.get("cost")
    h = p["system_usd_per_card"] / HOST_USD
    v = {"usd_per_kWh": p["usd_per_kWh"], "pue": p["pue"], "life_years": p["life_years"],
         "capital_rate": p["capital_rate"],
         "host": h if math.isclose(p["system_W_per_card"], HOST_W * h, abs_tol=1e-9) else None}
    if cp is not None:
        v.update(wafer_usd=cp["wafer_usd"], d0=base["d0"], pkg_scale=1.0)
        if cp.get("asm_model") == "component":
            v.update(carrier_yield=cp["carrier_yield"], attach_yield=cp["attach_yield"])
    v["mem_usd_per_GB"] = mem_price(p, mem_type, base.get("prices"))[0]
    return v


def row_usd(row: dict, base: dict, ov: dict | None = None) -> float:
    """$/M unit of ``row`` with inputs ``ov`` overriding the base.  ``row`` needs mm2, dies, si_mm2, mem_id, mem_type,
    mem_GB, thr and tco (pkg_W, pkg_W_idle, pkg_usd, mem_usd_per_GB, mem_scrap_k); ``base`` = {tco: tco_params, cost:
    cost params or None (package $ fixed at row tco.pkg_usd), d0, prices}.  ``ov["mem_end"]`` = "lo" / "hi" puts the
    memory price at that end of the row's own type range (joint corners over mixed memory types)."""
    from .energy_dse import _hbm_footprint, hbm_stacks, package_cost
    from .tco import tco_point
    ov = ov or {}
    t = row["tco"]
    cp = base.get("cost")
    km = t.get("mem_scrap_k", 1.0)
    if cp is not None and any(k in ov for k in COST_INPUTS):
        c = dict(cp)
        c["wafer_usd"] = ov.get("wafer_usd", cp["wafer_usd"])
        c["pkg_usd_per_mm2"] = cp["pkg_usd_per_mm2"] * ov.get("pkg_scale", 1.0)
        for k in ASM_INPUTS:
            if k in ov:
                c[k] = ov[k]
        d0 = ov.get("d0", base["d0"])
        y = math.exp(-row["mm2"] * d0 / 100.0)
        cc = package_cost(c, row["mm2"], row.get("dies", 1), y, row["si_mm2"] + _hbm_footprint(row["mem_id"]),
                          hbm_stacks(row["mem_id"]))
        pkg, km = cc["usd_good_pkg"], cc["asm"]["k_mem"]
    else:
        pkg = t["pkg_usd"]
    p = dict(base["tco"])
    for k in ("usd_per_kWh", "pue", "life_years", "capital_rate"):
        if k in ov:
            p[k] = ov[k]
    if "host" in ov:
        p["system_usd_per_card"], p["system_W_per_card"] = HOST_USD * ov["host"], HOST_W * ov["host"]
    pr = ov.get("mem_usd_per_GB", t["mem_usd_per_GB"])
    if "mem_end" in ov and (mr := mem_range(row["mem_type"])) is not None:
        pr = mr[0] if ov["mem_end"] == "lo" else mr[1]
    return tco_point(p, pkg_usd=pkg, mem_usd=row["mem_GB"] * pr * km, thr=row["thr"], pkg_W=t["pkg_W"],
                     pkg_W_idle=t["pkg_W_idle"], unit=row.get("unit") or "token")["usd_per_Munit"]


def applicable(k: str, base: dict) -> bool:
    cp = base.get("cost")
    if k in COST_INPUTS and cp is None:
        return False
    if k in ASM_INPUTS and cp.get("asm_model") != "component":
        return False
    return True


def input_range(k: str, mem_type: str | None = None, base: dict | None = None):
    """(lo, hi, tag_lo, tag_hi, src) of one input, or None (no sourced range for this memory type / carrier)."""
    if k == "mem_usd_per_GB":
        return mem_range(mem_type) if mem_type else None
    if k == "carrier_yield":
        cp = (base or {}).get("cost") or {}
        return CARRIER_RANGES.get(cp.get("carrier"))
    return RANGES[k]


def corner(base: dict, side: str, skip=()) -> dict:
    """0.319: every applicable input at its adverse ($/M-raising) or favourable end together.  Memory goes to the end
    of each row's own type range (``mem_end``)."""
    ov = {}
    for k in INPUTS:
        if k in skip or not applicable(k, base):
            continue
        if k == "mem_usd_per_GB":
            ov["mem_end"] = "hi" if side == "adverse" else "lo"
            continue
        rg = input_range(k, None, base)
        if rg is None:
            continue
        lo_end = (k in LOW_IS_ADVERSE) == (side == "adverse")
        ov[k] = rg[0] if lo_end else rg[1]
    return ov


def tornado(row: dict, base: dict) -> dict:
    """One-at-a-time sensitivity of one priced row; bars sorted by the larger |Δ $/M| (fraction of base)."""
    t0 = row["tco"]["usd_per_Munit"]
    chk = row_usd(row, base)
    if not math.isclose(chk, t0, rel_tol=1e-9):
        raise ValueError(f"sensitivity base mismatch: {chk} vs {t0}")
    bv = base_values(base, row["mem_type"])
    bars, skipped = [], []
    for k in INPUTS:
        if not applicable(k, base):
            skipped.append(f"{LABELS[k]}（{'封装价为用户输入' if base.get('cost') is None else '封装良率按 0.318 口径'}）")
            continue
        rg = input_range(k, row["mem_type"], base)
        if rg is None:
            skipped.append(f"{LABELS[k]}（{row['mem_type']} 无公开价区间）")
            continue
        lo, hi, tl, th, src = rg
        ul, uh = row_usd(row, base, {k: lo}), row_usd(row, base, {k: hi})
        bars.append({"input": k, "label": LABELS[k], "lo": lo, "hi": hi, "tag_lo": tl, "tag_hi": th, "base": bv.get(k),
                     "usd_lo": ul, "usd_hi": uh, "d_lo": ul / t0 - 1, "d_hi": uh / t0 - 1,
                     "span": abs(uh - ul) / t0, "src": src})
    bars.sort(key=lambda b: -max(abs(b["d_lo"]), abs(b["d_hi"])))
    cor = {sd: row_usd(row, base, corner(base, sd)) for sd in ("adverse", "favourable")}
    return {"usd_per_Munit": t0, "unit": row.get("unit") or "token", "bars": bars, "skipped": skipped,
            "corners": {sd: {"usd": u, "d": u / t0 - 1} for sd, u in cor.items()},
            "note": "一次只动一个输入（其余取该设计的基值）；吞吐 / 封装功耗不随这些输入变化；条 = 区间两端的 $/M 相对基值变化"}


def scan_flip(verdict, k: str, rg, base_val, n: int = 64, iters: int = 48) -> dict:
    """Scan input ``k`` over [lo, hi] (``verdict(ov) → label``).  Returns the base label, whether any value inside the
    range gives another label, and the threshold nearest the base value on each side (bisected)."""
    lo, hi = rg[0], rg[1]
    v0 = verdict({})
    xs = [lo + (hi - lo) * i / n for i in range(n + 1)]
    labs = [verdict({k: x}) for x in xs]
    out = {"input": k, "label": LABELS[k], "base": v0, "lo": lo, "hi": hi, "flips": any(l_ != v0 for l_ in labs),
           "thresholds": []}
    if not out["flips"]:
        return out
    # transitions between adjacent grid points, each bisected
    for i in range(n):
        if labs[i] != labs[i + 1]:
            a, b, la = xs[i], xs[i + 1], labs[i]
            for _ in range(iters):
                m = (a + b) / 2
                if verdict({k: m}) == la:
                    a = m
                else:
                    b = m
            out["thresholds"].append({"at": (a + b) / 2, "from": labs[i], "to": labs[i + 1]})
    # 0.319: a base value outside [lo, hi] (e.g. a DSE run at D0 0.3) whose label differs from the whole nearer end —
    # the change lies between that end and the base; bisect there and mark it as outside the sourced range
    if base_val is not None and not lo <= base_val <= hi:
        e = hi if base_val > hi else lo
        le = labs[-1] if e == hi else labs[0]
        if le != v0:
            a, b = e, base_val
            for _ in range(iters):
                m = (a + b) / 2
                if verdict({k: m}) == le:
                    a = m
                else:
                    b = m
            out["thresholds"].append({"at": (a + b) / 2, "from": le, "to": v0, "outside": True})
    if base_val is not None and out["thresholds"]:
        out["nearest"] = min(out["thresholds"], key=lambda t: abs(t["at"] - base_val))
    return out


def mono_split_verdict(rows, monos, splits, base, tie=0.01):
    """$/M mono vs split verdict (as energy_dse._tco_verdict) under overrides."""
    def f(ov):
        bm = min(row_usd(rows[i], base, ov) for i in monos)
        bs = min(row_usd(rows[i], base, ov) for i in splits)
        return "split" if bs / bm - 1 < -tie else "mono"
    return f


def compare_key(x: dict) -> tuple:
    """The design group of energy_dse._split_compare (same package totals)."""
    return (x["mac_x"], x["sram_x"], x["side"], x["topology"], x["mem_id"], x["slc_mib"], x["f_op"], x["noc_link_Bpc"])


def compare_flips(rows, compare, base, tie=0.01) -> list[dict]:
    """Per mono/split compare entry with a priced verdict on both sides: which inputs flip it, and where."""
    from .energy_dse import _tco_ok
    out = []
    for e in compare or []:
        if e.get("verdict_tco") not in ("split", "mono"):
            continue
        key = compare_key(e)
        grp = [i for i, x in enumerate(rows) if compare_key(x) == key and _tco_ok(x)]
        monos = [i for i in grp if rows[i].get("dies", 1) == 1]
        splits = [i for i in grp if rows[i].get("dies", 1) > 1]
        if not monos or not splits:
            continue
        f = mono_split_verdict(rows, monos, splits, base, tie)
        mt = rows[monos[0]]["mem_type"]
        bv = base_values(base, mt)
        scans = []
        for k in INPUTS:
            if not applicable(k, base):
                continue
            rg = input_range(k, mt, base)
            if rg is None:
                continue
            scans.append(scan_flip(f, k, rg, bv.get(k), n=32, iters=40))
        out.append({"key": key, "mac_x": e["mac_x"], "sram_x": e["sram_x"], "mem_id": e["mem_id"], "verdict_tco": e["verdict_tco"],
                    "flips": [s for s in scans if s["flips"]], "n_inputs": len(scans),
                    "corners": {sd: f(corner(base, sd)) for sd in ("adverse", "favourable")}})
    return out

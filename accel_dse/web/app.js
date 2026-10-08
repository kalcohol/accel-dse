/* accel_dse workbench UI — zero CDN, debounce eval on knob change */
(function () {
  "use strict";

  const DEBOUNCE_MS = 280;
  let debounceTimer = null;
  let evalSeq = 0;
  let seriesCache = [];
  let computeCache = [];
  let domain = "llm";
  let memKind = "HBM"; // derived from memSel.mem_type (manual geometry uses it too)
  // v0.29 structured memory selection (mem_catalog via GET /api/memory)
  let memCatalog = null;
  let memSel = null;
  let attnParallel = "tp";
  let specDraft = "mtp"; // v0.31 投机解码草稿来源
  let moeShard = "tp_ep"; // v0.31 MoE 专家切分
  const DEFAULT_MEM_TYPE = "HBM3E";
  const DEFAULT_C2C_LAT_US = 3.0;
  const DEFAULT_SYNC_OVERLAP = 0.0;
  let syncingParallel = false;
  let sweepAxis = "chips";
  let sweepParallelChips = 4;
  let toastTimer = null;
  let lastSweepResult = null;
  let lastCard = null;
  let baselineCard = null;
  let baselineMs = null;
  let dualCardA = null;
  let dualCardB = null;
  let restoringUrl = false;
  let urlPushTimer = null;
  // v0.28 layout state
  let activeTab = "overview";
  let applyingPreset = false;
  let assumedWired = false;
  let sweepAutoRan = false;
  let sweepStale = false;
  const TABS = ["overview", "sweep", "pareto", "ab", "assump"];

  const DEFAULT_MEM_EFF = 0.70;
  const DEFAULT_MAC_EFF = 0.0;
  const DEFAULT_FREQ_GHZ = 1.0;
  // v0.24 assumed energy / cost stub (engine default OFF; util/duty default 1.0)
  const ECON_NUM_FIELDS = [
    "tdp_w", "watts_per_tops", "cost_per_card_usd", "mem_addon_usd",
    "usd_per_kwh", "amortize_years",
  ];
  const DEFAULT_POWER_UTIL = 1.0;
  const DEFAULT_DUTY_CYCLE = 1.0;
  let presetsCache = [];
  let econExample = {};

  const $ = (id) => document.getElementById(id);

  // --- v0.27 UI 中文化：仅显示层翻译；API 字段 / JSON key / CSV 头保持英文 ---
  const ASSUMED_TAG = "「假设」";
  const DOMAIN_ZH = { llm: "LLM", video: "视频", protein: "蛋白质" };
  const AXIS_ZH = { chips: "芯片数", package: "存储", compute: "算力", parallel: "TP/PP 并行", series: "系列" };
  const LEVEL_ZH = { core: "单核", cluster: "集群" };
  const WALL_ZH = {
    compute: "算力", memory: "访存", dram: "访存 DRAM", c2c: "C2C 通信",
    fabric: "Fabric 互连", bubble: "PP 气泡", capacity: "容量", balanced: "均衡",
    sync: "通信同步", pp_act: "PP 激活", a2a: "EP all-to-all",
  };
  function domainZh(d) { return DOMAIN_ZH[d] || d; }
  function wallZh(w) {
    if (!w) return "—";
    const k = String(w).toLowerCase();
    return WALL_ZH[k] ? WALL_ZH[k] : String(w);
  }
  function yesNo(b) { return b ? "是" : "否"; }
  // 引擎（serve.py / catalog）返回的英文说明：已知短语逐条替换，未知部分保留英文原文。
  const BANNER_EN_PREFIX = "All absolute bandwidth, efficiency, and frequency numbers";
  const BANNER_ZH =
    "所有带宽、效率与频率的绝对数值均为假设 / 未标定的 DSE 标签 —— 并非硅片实测。" +
    "MAC/TOPS 峰值由 核数 × tops_per_core @ 假设频率 推导。" +
    "MetricsCard 覆盖 LLM（TTFT/TPOT）、视频（TTFC / 帧每秒）与蛋白质（每序列时间 + pair 内存）。" +
    "视频 / 蛋白质多卡：TP/PP/EP 需满足 TP×PP×EP = 芯片数；TP 激活集合通信（ring|tree）、" +
    "PP 气泡（去噪步骤难以流水化）、无 MoE 专家时 EP 不生效。" +
    "能耗 / 成本（est_*）字段为基于用户参数（tdp_w / watts_per_tops / cost_per_card_usd）的假设估算" +
    "（ASSUMED stub）—— 非硅片功耗。";
  function bannerZh(en) {
    if (!en) return "";
    return String(en).startsWith(BANNER_EN_PREFIX) ? BANNER_ZH : zhText(en);
  }
  const PRESET_ZH = {
    "edge-lpddr-4x64": {
      label: "边缘 SoC · LPDDR5X 4×x64 @8533（256-bit，64 GB）+ 64T 集群 · 1 芯片",
      note: "边缘级：4 颗 x64 LPDDR5X 封装（每颗 4×16-bit 通道）@ 8533 MT/s，16 GB/颗（厂商量产）；大 LLM 受带宽限制",
    },
    "edge-lpddr6-4x96": {
      label: "边缘 SoC · LPDDR6 4×x96 @10667（384-bit，64 GB）+ 128T 集群 · 1 芯片",
      note: "LPDDR6 x96 封装 = 4×24-bit 通道（8×12-bit 子通道）；原始 512 GB/s，扣 8/9 后可用 455 GB/s",
    },
    "card-hbm-4stack": {
      label: "加速卡 · HBM3E 4×12H×24Gb @9200（144 GB）+ 256T 集群 · 1 芯片",
      note: "单张 PCIe/OAM 形态卡；4 个 12-high 36 GB HBM3E 堆（厂商量产）",
    },
    "card-hbm3e-8x12h": {
      label: "加速卡 · HBM3E 8×12H×24Gb @9200（288 GB）+ 256T 集群 · 1 芯片",
      note: "B300 / MI355X 级几何：8 × 36 GB 堆，原始 9.42 TB/s",
    },
    "server-socamm2-8": {
      label: "服务器 · SOCAMM2 8×128b @9600（1.5 TB）+ 256T · 1 芯片",
      note: "Vera 级 LPDDR5X 模组（JESD328 SOCAMM2）192 GB/条；容量大，原始带宽 1.2 TB/s",
    },
    "scaleup-8chip": {
      label: "Scale-up · 8 × (HBM3E 4×12H + 256T) · TP=8",
      note: "8 卡，默认 TP=8，走假设的 C2C（每次集合通信 α = 3 µs 暴露）；应用后可再调整 TP/PP/EP",
    },
  };
  const ZH_RULES = [
    [/\[rates\/caps per catalog tag; efficiency assumed\]/g, "「速率 / 容量按目录标签；效率为假设」"],
    [/exposed sync: α=([\d.]+) µs per collective × \(TP 2 all-reduce\/layer, EP 2 all-to-all\/layer, PP 1 send\/stage\) × \(1−overlap ([\d.]+)\) added outside max\(compute, mem, c2c\) \[ASSUMED α; v0\.29\]/g,
      "暴露的通信同步：每次集合通信 α = $1 µs ×（TP 每层 2 次 all-reduce、EP 每层 2 次 all-to-all、PP 每级 1 次收发）×（1 − 重叠 $2），叠加在 max(计算, 访存, C2C) 之外「假设 α；v0.29」"],
    [/exposed sync: (\d+) collectives × α=([\d.]+) µs × \(1−overlap ([\d.]+)\) = ([\d.]+) µs per forward \[assumed α\]/g,
      "暴露的通信同步：$1 次集合通信 × α = $2 µs ×（1 − 重叠 $3）= 每次前向 $4 µs「假设 α」"],
    [/^attn_parallel=dp: .*$/g, "注意力 DP：KV 在 TP 组内按 batch 切分；注意力权重每卡复制（容量 + decode 权重流）；注意力计算按 batch 拆分 ≈ 每卡 FLOPs 不变；DP↔TP 重分片集合通信近似为每层 2 次「假设」"],
    [/mem warning: /g, "存储提示："],
    [/\[assumed econ stub\]/g, "「假设能耗/成本估算」"],
    [/\[ASSUMED coarse Softmax\/RoPE\/LN\/misc; not cycle-accurate; t_compute\*=\(1\+oh\); default 0=off\]/g,
      "「假设：粗粒度 Softmax/RoPE/LN/其他；非周期精确；t_compute×(1+oh)；默认 0 = 关闭」"],
    [/\[ASSUMED coarse Softmax\/RoPE\/LN; not cycle-accurate; t_compute\*=\(1\+oh\)\]/g,
      "「假设：粗粒度 Softmax/RoPE/LN；非周期精确；t_compute×(1+oh)」"],
    [/\(finer /g, "（细分 "],
    [/\[ASSUMED user table — not silicon; peak_TOPS\*=factor, t_compute\/=factor; default all 1\.0=bytes-only\]/g,
      "「假设：用户系数表，非硅片实测；peak_TOPS×系数，t_compute÷系数；默认全 1.0 = 仅按字节计」"],
    [/\[assumed\/override\]/g, "「假设/覆盖」"],
    [/\[assumed BW\]/g, "「假设带宽」"],
    [/\(bytes-only\)/g, "（仅影响字节数）"],
    [/explicit parallel tp\/pp\/ep=/g, "显式并行 TP/PP/EP="],
    [/\(product==chip_count=(\d+)\)/g, "（乘积 = 芯片数 $1）"],
    [/DRAM preset (\w+) \(geometry defaults\)/g, "DRAM 预设 $1（默认几何参数）"],
    [/NPU from named SKU (\S+) \(overrides cores\/pe\)/g, "NPU 取自命名 SKU $1（覆盖核数 / PE）"],
    [/NPU from explicit PE/g, "NPU 取自显式 PE"],
    [/NPU default cores=16 × 6\.25 T\/core \(~100 TOPS class, assumed\)/g, "NPU 默认 16 核 × 6.25 T/核（约 100 TOPS 级，假设）"],
    [/dtype\/quant ignored for (\w+) \(use weight_bits\/act_bits\)/g, "$1 领域忽略 dtype/quant（请改用 weight_bits/act_bits）"],
    [/seq-only protein \(pair_dim=0\)/g, "仅序列的蛋白质模型（pair_dim=0）"],
    [/\b(video|protein) knobs:/g, "$1 参数："],
    [/pair act ≈/g, "pair 激活 ≈"],
    [/\$\/MTok = decode-only tokens \(no prefill, host, network, cooling, margin\)/g,
      "$/MTok 仅按 decode token 计（不含 prefill、主机、网络、散热、利润）"],
    [/\$\/MTok is LLM-only; usd_per_kwh \/ amortize_years ignored for (\w+)/g,
      "$/MTok 仅适用于 LLM；$1 领域忽略 usd_per_kwh / amortize_years"],
    [/OOM config — energy\/cost numbers are not meaningful/g, "OOM 配置 —— 能耗 / 成本数值无参考意义"],
    [/tdp_w set → watts_per_tops ignored/g, "已设置 tdp_w → 忽略 watts_per_tops"],
    [/ W\/card/g, " W/卡"],
    [/\bcost: chips=/g, "成本：芯片数="],
    [/\(card ([-\d.e+]+) \+ mem add-on ([-\d.e+]+)\) USD/g, "（每卡 $1 + 存储附加 $2）USD"],
    [/× chips=/g, "× 芯片数="],
    [/\(chips=(\d+)\)/g, "（芯片数=$1）"],
    [/× peak /g, "× 峰值 "],
    [/\[assumed, uncalibrated\]/g, "「假设，未标定」"],
    [/\[assumed mapping\]/g, "「假设映射」"],
    [/\[assumed\]/g, ASSUMED_TAG],
    [/\[derived\]/g, "「推导」"],
    [/FLOPs\/BW uncalibrated \(DSE accounting only\)/g, "FLOPs/BW 未标定（仅 DSE 估算）"],
    [/\(default mapping\)/g, "（默认映射）"],
    [/DRAM from geometry knobs on (\w+) preset defaults/g, "DRAM 取自 $1 预设的默认几何参数"],
    [/DRAM from manual geometry knobs/g, "DRAM 取自手动几何参数"],
    [/NPU cores×tops_per_core → n_engines=n_cores, near-square PE\/core/g,
      "NPU 核数×tops_per_core → n_engines=n_cores，每核近方形 PE"],
    [/scale_efficiency=\(t_single\/t_multi\)\/chips on domain primary \(([^)]+)\); ideal 1\.0; ignores host\/NIC non-ideal; C2C\/PP-bubble\/EP already in t_multi/g,
      "scale_efficiency=(t_single/t_multi)/芯片数，基于领域主指标（$1）；理想值 1.0；忽略主机/NIC 非理想因素；C2C / PP 气泡 / EP 已计入 t_multi"],
    [/scale vs chips=1:/g, "相对单卡："],
    [/(\w+): ep=1 \(EP unused — typical for DiT\/protein; no experts\)/g, "$1：EP=1（无专家，EP 不生效 —— DiT / 蛋白质的典型情形）"],
    [/(\w+) chip_count=1 → single-card roofline \(no TP\/PP collectives\)/g, "$1 芯片数=1 → 单卡 roofline（无 TP/PP 集合通信）"],
    [/^single-card$/g, "单卡"],
    [/user knobs, NOT silicon \/ PDK \/ JEDEC power; cards only \(no host\/cooling\/PUE\)/g,
      "用户参数，非硅片 / PDK / JEDEC 功耗；仅计算卡（不含主机 / 散热 / PUE）"],
    [/capacity stress; sharded \/ \(tp\*pp\) when multi-card — crude/g, "容量压力；多卡时按 (TP×PP) 分片 —— 粗略估计"],
    [/product series alias →/g, "产品系列别名 →"],
    [/illustrative\/placeholder/g, "示意/占位"],
    [/\(illustrative placeholder\)/g, "（示意占位）"],
    [/toy hand-check shape/g, "toy 手算校验 shape"],
    [/NOT a vendor checkpoint/g, "非厂商权重"],
    [/status:found/g, "状态:已找到"],
    [/status:nearest/g, "状态:最近似"],
    [/\bclaimed:/g, "标称:"],
    [/\barch:/g, "架构:"],
    [/\bsource:/g, "来源:"],
    [/\bextras:/g, "附加:"],
    [/single-core peak TOPS option/g, "单核峰值 TOPS 选项"],
    [/cluster peak; primary/g, "集群峰值；主配置"],
    [/\balts=/g, "备选="],
    [/min for 128-bit SoC bus:/g, "128-bit SoC 总线最小配置："],
    [/max practical on package:/g, "封装上实际最大："],
    [/(\d+) × 64-bit packages \(bus=(\d+)-bit\)/g, "$1 × 64-bit 封装（总线 $2-bit）"],
    [/(\d+) × 64-bit packages/g, "$1 × 64-bit 封装"],
    [/JEDEC\/marketing LPDDR5X band/g, "JEDEC / 市场 LPDDR5X 速率档"],
    [/granule=64-bit package/g, "粒度=64-bit 封装"],
    [/dies \(2×12-bit sub-ch each\)/g, "die（每颗 2×12-bit 子通道）"],
    [/SoC bus≈/g, "SoC 总线≈"],
    [/granule=x24 die NOT 48-bit/g, "粒度=x24 die，非 48-bit"],
    [/(\d+) stacks × (HBM\w*)/g, "$1 stack × $2"],
    [/assumed JEDEC (HBM\w*)-class/g, "假设 JEDEC $1 级"],
    [/assumed JEDEC (HBM\w*) base/g, "假设 JEDEC $1 基础规格"],
    [/assumed (HBM\w*) public peak label/g, "假设 $1 公开峰值标称"],
    [/assumed (HBM\w*) \*design-target\*/g, "假设 $1 设计目标"],
    [/not JEDEC-final; marketing bands vary; uncalibrated/g, "非 JEDEC 定稿；市场速率档不一；未标定"],
    [/\(uncalibrated\)/g, "（未标定）"],
    [/\bpeak\)/g, "峰值）"],
    [/\brate=/g, "速率="],
    [/Gbps band\)/g, "Gbps 速率档）"],
    [/^video\b/, "视频"],
    [/^protein\b/, "蛋白质"],
  ];
  function zhText(s) {
    if (s == null) return "";
    let out = String(s);
    for (const [re, rep] of ZH_RULES) out = out.replace(re, rep);
    return out;
  }
  const statusEl = $("status");
  const errorBox = $("error-box");

  function setStatus(text, cls) {
    statusEl.textContent = text;
    statusEl.className = "status" + (cls ? " " + cls : "");
  }

  function showError(msg) {
    errorBox.textContent = msg || "";
    errorBox.classList.toggle("show", !!msg);
  }

  function showToast(msg, kind) {
    const el = $("toast");
    if (!el) return;
    el.hidden = !msg;
    el.textContent = msg || "";
    el.className = "toast" + (kind === "ok" ? " ok" : "");
    clearTimeout(toastTimer);
    if (msg) {
      toastTimer = setTimeout(() => {
        el.hidden = true;
      }, 4200);
    }
  }

  async function apiGet(path) {
    const r = await fetch(path);
    if (!r.ok) {
      const t = await r.text();
      throw new Error(t || r.statusText);
    }
    return r.json();
  }

  async function apiPost(path, body) {
    const r = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      throw new Error(data.error || data.detail || r.statusText);
    }
    return data;
  }

  function fmt(n, digits) {
    if (n === null || n === undefined || Number.isNaN(n)) return "—";
    return Number(n).toFixed(digits);
  }

  function fmtInt(n) {
    if (n === null || n === undefined) return "—";
    return String(n);
  }

  // v0.28.1：仅显示层的人性化单位（API / CSV / JSON 导出仍为原始 ms / 字节）
  function fmtGB(v) {
    const n = Number(v);
    if (v == null || !Number.isFinite(n)) return "—";
    return trimNum(n.toPrecision(4));
  }
  function trimNum(str) {
    if (/e/i.test(str)) return String(Number(str));
    return str.indexOf(".") >= 0 ? str.replace(/\.?0+$/, "") : str;
  }
  function fmtTime(ms, sig) {
    const v = Number(ms);
    if (ms == null || ms === "" || !Number.isFinite(v)) return "—";
    if (v === 0) return "0";
    const p = sig || 4;
    const a = Math.abs(v);
    if (a >= 1000) return trimNum((v / 1000).toPrecision(p)) + " s";
    if (a < 1) return trimNum((v * 1000).toPrecision(p)) + " µs";
    return trimNum(v.toPrecision(p)) + " ms";
  }
  function fmtTimeSigned(ms, sig) {
    const v = Number(ms);
    if (!Number.isFinite(v)) return "—";
    return (v > 0 ? "+" : v < 0 ? "−" : "") + fmtTime(Math.abs(v), sig);
  }
  function rawMs(ms) {
    const v = Number(ms);
    return Number.isFinite(v) ? `${v} ms` : "";
  }

  // v0.28.1：下拉框宽度贴合「当前选中项」文字 —— 收起状态不截断
  let measureCtx = null;
  function fitSelect(sel) {
    if (!sel) return;
    if (window.matchMedia && window.matchMedia("(max-width: 768px)").matches) {
      sel.style.width = "";
      return;
    }
    const opt = sel.selectedOptions && sel.selectedOptions[0];
    const text = opt ? opt.textContent : "";
    if (!measureCtx) measureCtx = document.createElement("canvas").getContext("2d");
    const cs = getComputedStyle(sel);
    measureCtx.font = cs.font || `${cs.fontSize} ${cs.fontFamily}`;
    const w = measureCtx.measureText(text).width;
    const pad = (parseFloat(cs.paddingLeft) || 0) + (parseFloat(cs.paddingRight) || 0) +
      (parseFloat(cs.borderLeftWidth) || 0) + (parseFloat(cs.borderRightWidth) || 0);
    sel.style.width = Math.ceil(w + pad + 24) + "px"; // + 下拉箭头
  }
  function fitCoreSelects() {
    ["series", "mem-type", "compute", "preset-select", "dtype", "quant"].forEach((id) => fitSelect($(id)));
  }

  const FAMILY_ZH = { dense: "dense", moe: "MoE", mla: "MLA", dit_video: "DiT", protein: "", toy: "toy" };
  function seriesSize(s) {
    const m = /claimed:([^|]+)/.exec(s.metadata || "");
    if (!m) return "";
    const t = /~?\d[\d.]*(?:[–-]\d[\d.]*)?[KMBT](?:-A\d[\d.]*[KMBT])?/.exec(m[1]);
    return t ? t[0].replace(/^~/, "≈") : "";
  }
  function seriesShortLabel(s) {
    const fam = FAMILY_ZH[s.family] != null ? FAMILY_ZH[s.family] : s.family;
    const tail = [fam, seriesSize(s)].filter(Boolean).join(" ");
    return s.id + (tail ? " · " + tail : "") + (s.is_illustrative ? " · 示意" : "");
  }
  function seriesFullLabel(s) {
    const hf = s.hf_id && s.hf_id !== "—" ? ` · ${s.hf_id}` : "";
    return `${s.id} [${s.is_hf_backed ? "HF" : "示意"}/${s.family}]${hf} —— ${zhText(s.metadata || s.shape_name || "")}`;
  }
  function fmtBw(gbps) {
    const v = Number(gbps);
    if (!Number.isFinite(v)) return "—";
    return v >= 1000 ? trimNum((v / 1000).toPrecision(2)) + " TB/s" : Math.round(v) + " GB/s";
  }
  function computeShortLabel(c) {
    return c.level === "core" ? `${c.peak_tops}T · 单核` : `${c.peak_tops}T · ${c.n_cores} 核`;
  }
  function computeFullLabel(c) {
    return `${c.id} · ${LEVEL_ZH[c.level] || c.level} 峰值 ${c.peak_tops} TOPS = ${c.n_cores} × ${c.tops_per_core} T/核 @ ` +
      `${(c.frequency_hz / 1e9).toFixed(2)} GHz${ASSUMED_TAG} —— ${zhText(c.note || "")}`;
  }
  function appendGrouped(sel, groups) {
    for (const [label, items] of groups) {
      if (!items.length) continue;
      const g = document.createElement("optgroup");
      g.label = label;
      items.forEach((o) => g.appendChild(o));
      sel.appendChild(g);
    }
  }

  function populateSeries() {
    const sel = $("series");
    const productOnly = $("product-only").checked;
    let rows = seriesCache.filter((s) => s.domain === domain);
    if (productOnly) rows = rows.filter((s) => s.is_hf_backed);
    const prev = sel.value;
    sel.innerHTML = "";
    if (!rows.length) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "（该领域暂无系列）";
      sel.appendChild(opt);
    } else {
      const groups = new Map();
      for (const s of rows) {
        const opt = document.createElement("option");
        opt.value = s.id;
        opt.textContent = seriesShortLabel(s);
        opt.title = seriesFullLabel(s);
        const fam = FAMILY_ZH[s.family] != null ? FAMILY_ZH[s.family] : s.family;
        const gk = s.is_illustrative ? "示意 / 合成" : (fam ? fam + "（HF / 公开配置）" : "HF / 公开配置");
        if (!groups.has(gk)) groups.set(gk, []);
        groups.get(gk).push(opt);
      }
      const keys = [...groups.keys()].sort((a, b) => (a.startsWith("示意") ? 1 : 0) - (b.startsWith("示意") ? 1 : 0));
      appendGrouped(sel, keys.map((k) => [k, groups.get(k)]));
      if (prev && [...sel.options].some((o) => o.value === prev)) {
        sel.value = prev;
      } else {
        const prefer =
          domain === "video"
            ? ["minimax-h3", "illustrative_dit_video", "series/dit-video"]
            : domain === "protein"
              ? ["esmfold", "illustrative_protein_pair", "series/protein-pair"]
              : ["qwen3-32b", "glm-5.3", "illustrative_27B", "qwen3.8-27b"];
        for (const p of prefer) {
          if ([...sel.options].some((o) => o.value === p)) {
            sel.value = p;
            break;
          }
        }
      }
    }
    updateSeriesMeta();
    updateDomainNote();
  }

  function updateSeriesMeta() {
    const id = $("series").value;
    const s = seriesCache.find((x) => x.id === id);
    $("series-meta").textContent = s
      ? zhText(`${s.metadata || s.shape_name}`)
      : "—";
    $("series").title = s ? seriesFullLabel(s) : "";
    fitSelect($("series"));
  }

  function updateDomainNote() {
    const note = $("domain-note");
    const llm = $("wl-llm");
    const vid = $("wl-video");
    const pro = $("wl-protein");
    if (llm) llm.style.display = domain === "llm" ? "" : "none";
    if (vid) vid.style.display = domain === "video" ? "" : "none";
    if (pro) pro.style.display = domain === "protein" ? "" : "none";

    // Prefer defaults when switching domain series
    if (domain === "video") {
      note.classList.add("show");
      note.textContent =
        "视频 MetricsCard：TTFC = 去噪步数 × 单次前向；帧/秒；TP/PP/EP（乘积 = 芯片数）—— " +
        "TP 激活集合通信（ring）、PP 气泡（去噪步骤难以流水化）、无 MoE 时 EP 不生效。" +
        "带宽 / 效率均为假设值。";
    } else if (domain === "protein") {
      note.classList.add("show");
      note.textContent =
        "蛋白质 MetricsCard：每序列时间 + pair L²；TP/PP/EP（乘积 = 芯片数）—— " +
        "TP 集合通信、PP 微批气泡（微批数取 decode_mb）、无 MoE 时 EP 不生效；" +
        "pair 分片为粗略估计。带宽 / 效率均为假设值。";
    } else {
      note.classList.add("show");
      note.textContent =
        "LLM MetricsCard：TTFT = prefill；TPOT = 每 token decode 时间（基线 Δ 以 TPOT 计）；" +
        "TP/PP/EP 乘积 = 芯片数。带宽 / 效率均为假设值。";
    }
    const seg = $("domain-seg");
    if (seg) seg.title = note.textContent;
    updateMetricLabels();
    updateSceneSummary();
  }

  function updateMetricLabels() {
    const pk = $("m-primary-k");
    const sk = $("m-secondary-k");
    const extraCard = $("m-extra-card");
    const capCard = $("m-cap-card");
    const pairRow = $("i-pair-row");
    const wlLabel = $("i-wl-label");
    const brTitle = $("breakdown-title");
    const domLabel = $("metrics-domain-label");
    if (domLabel) domLabel.textContent = "（" + domainZh(domain) + "）";
    if (domain === "video") {
      if (pk) pk.textContent = "TTFC（首个 clip 时延）";
      if (sk) sk.textContent = "帧 / 秒";
      if (extraCard) {
        extraCard.style.display = "";
        $("m-extra-k").textContent = "去噪步数";
      }
      if (capCard) capCard.style.display = "";
      if (pairRow) pairRow.style.display = "none";
      if (wlLabel) wlLabel.textContent = "帧数 / 去噪步数 / 批大小";
      if (brTitle) brTitle.textContent = "单步前向时间分解";
    } else if (domain === "protein") {
      if (pk) pk.textContent = "每序列时间";
      if (sk) sk.textContent = "pair 字节";
      if (extraCard) {
        extraCard.style.display = "";
        $("m-extra-k").textContent = "序列长度";
      }
      if (capCard) capCard.style.display = "";
      if (pairRow) pairRow.style.display = "";
      if (wlLabel) wlLabel.textContent = "序列长度 / 批大小";
      if (brTitle) brTitle.textContent = "前向时间分解";
    } else {
      if (pk) pk.textContent = "TTFT（首 token 时延）";
      if (sk) sk.textContent = "TPOT（每 token 时延）";
      if (extraCard) extraCard.style.display = "none";
      if (capCard) capCard.style.display = "none";
      if (pairRow) pairRow.style.display = "none";
      if (wlLabel) wlLabel.textContent = "提示 / 上下文 / 批大小";
      if (brTitle) brTitle.textContent = "Decode 阶段时间分解";
    }
  }

  // ---------- v0.29 结构化存储选择器（GET /api/memory；派生量与引擎同式）----------
  const TAG_ORDER = ["jedec", "jedec_likely", "vendor_shipping", "vendor_sampling", "vendor_announced", "speculative"];
  const TAG_ZH = {
    jedec: "JEDEC", jedec_likely: "疑似 JEDEC", vendor_shipping: "厂商量产",
    vendor_sampling: "送样", vendor_announced: "已发布", speculative: "推测",
  };
  const COMP_ZH = { rate: "速率", width: "位宽", capacity: "容量", count: "数量" };
  const UNIT_ZH = { package: "颗", module: "条", stack: "堆" };
  function memKindOf(t) { return String(t || "").toUpperCase().startsWith("HBM") ? "HBM" : "LPDDR"; }
  function weakestTag(tags) {
    let w = 0;
    for (const t of tags) {
      const i = TAG_ORDER.indexOf(t);
      w = Math.max(w, i < 0 ? TAG_ORDER.length - 1 : i);
    }
    return TAG_ORDER[w];
  }
  function tagBadge(t, extra) {
    return `<span class="ptag t-${t}"${extra ? ` title="${escapeHtml(extra)}"` : ""}>${TAG_ZH[t] || t}</span>`;
  }
  function memTypeCat(t) {
    if (!memCatalog) return null;
    return memCatalog.types.find((x) => x.id === String(t || "").toUpperCase()) || null;
  }
  function memFormCat(tc, form) {
    if (!tc) return null;
    return tc.forms.find((f) => f.id === form) || tc.forms.find((f) => f.id === tc.default_form) || tc.forms[0];
  }
  function capsFor(fc, width) {
    if (!fc || !fc.caps) return [];
    return fc.caps[String(width)] || [];
  }
  function memDefaults(t, form) {
    const tc = memTypeCat(t);
    const fc = memFormCat(tc, form);
    const sel = {
      mem_type: tc ? tc.id : DEFAULT_MEM_TYPE, mem_form: fc ? fc.id : null,
      mem_width_bits: fc ? fc.default_width : null, mem_rate_MTps: fc ? fc.default_rate : null,
      mem_count: fc ? fc.default_count : null, mem_cap_GB: null, hbm_height: null, hbm_die_Gb: null,
    };
    if (tc && tc.kind === "HBM") {
      sel.hbm_height = fc.default_height;
      sel.hbm_die_Gb = fc.default_density;
    } else if (fc) {
      sel.mem_cap_GB = (fc.default_cap || {})[String(sel.mem_width_bits)] || null;
    }
    return sel;
  }
  // 将任意（含旧链接 / 预设）字段吸附到目录合法值；未知组合保留但会被标为「推测」
  function normalizeMemSel(raw) {
    const t = (raw && raw.mem_type) ? String(raw.mem_type).toUpperCase() : DEFAULT_MEM_TYPE;
    const tc = memTypeCat(t) || memTypeCat(DEFAULT_MEM_TYPE);
    const fc = memFormCat(tc, raw && raw.mem_form);
    const d = memDefaults(tc.id, fc.id);
    const pick = (v, list, dv) => (v != null && list.includes(Number(v)) ? Number(v) : dv);
    const out = Object.assign({}, d);
    out.mem_width_bits = pick(raw && raw.mem_width_bits, fc.widths.map((w) => w.bits), d.mem_width_bits);
    const rates = fc.rates.map((r) => r.MTps);
    let r = raw && raw.mem_rate_MTps != null ? Number(raw.mem_rate_MTps) : null;
    if (r != null && r < 100) r *= 1000;
    out.mem_rate_MTps = r != null ? rates.reduce((a, b) => (Math.abs(b - r) < Math.abs(a - r) ? b : a), rates[0]) : d.mem_rate_MTps;
    out.mem_count = pick(raw && raw.mem_count, fc.counts.map((c) => c.n), d.mem_count);
    if (tc.kind === "HBM") {
      out.hbm_height = pick(raw && raw.hbm_height, fc.heights, d.hbm_height);
      out.hbm_die_Gb = pick(raw && raw.hbm_die_Gb, fc.densities, d.hbm_die_Gb);
      out.mem_cap_GB = null;
    } else {
      const caps = capsFor(fc, out.mem_width_bits).map((c) => c.GB);
      const dc = (fc.default_cap || {})[String(out.mem_width_bits)];
      out.mem_cap_GB = pick(raw && raw.mem_cap_GB, caps, dc != null ? dc : caps[0]);
    }
    return out;
  }
  function clocksText(t, rate, clocks) {
    if (memKindOf(t) === "HBM") return `${trimNum((rate / 1000).toFixed(2))} Gb/s/pin`;
    const c = clocks || {};
    const wck = c.wck_MHz != null ? c.wck_MHz : rate / 2;
    const ck = c.ck_MHz != null ? c.ck_MHz : rate / 8;
    return `WCK ${Math.round(wck)} / CK ${Math.round(ck)} MHz`;
  }
  function memDerive(sel, eff) {
    const tc = memTypeCat(sel.mem_type);
    const fc = memFormCat(tc, sel.mem_form);
    const hbm = tc.kind === "HBM";
    const width = hbm ? fc.default_width : sel.mem_width_bits;
    const bus = sel.mem_count * width;
    const raw = (bus * sel.mem_rate_MTps) / 8000;
    const payload = raw * (tc.payload_factor || 1);
    const effBw = payload * (eff != null ? eff : DEFAULT_MEM_EFF);
    let cap, capTag;
    if (hbm) {
      const per = (sel.hbm_height * sel.hbm_die_Gb) / 8;
      cap = sel.mem_count * per;
      const ct = (fc.cap_tags || []).find((c) => c.height === sel.hbm_height && c.die_Gb === sel.hbm_die_Gb);
      capTag = ct ? ct.tag : "speculative";
    } else {
      cap = sel.mem_count * sel.mem_cap_GB;
      const c = capsFor(fc, width).find((x) => x.GB === sel.mem_cap_GB);
      capTag = c ? c.tag : "speculative";
    }
    const rt = fc.rates.find((x) => x.MTps === sel.mem_rate_MTps);
    const wt = fc.widths.find((x) => x.bits === width);
    const nt = fc.counts.find((x) => x.n === sel.mem_count);
    const comp = {
      rate: rt ? rt.tag : "speculative", width: wt ? wt.tag : "speculative",
      capacity: capTag, count: nt ? nt.tag : "speculative",
    };
    const tag = weakestTag(Object.values(comp));
    const warn = [];
    const beach = (memCatalog && memCatalog.beachfront_warn_bits) || 576;
    if (!hbm && fc.id === "discrete" && bus > beach)
      warn.push(`板载 LPDDR 总线 ${bus}-bit 超出常见 SoC 边长（≈≤512–576 bit；Grace≈480-bit，Vera 用 SOCAMM2 模组到 1024-bit）`);
    if (hbm && sel.mem_count > 12) warn.push(`${sel.mem_count} 堆 HBM 超出已知产品（MI455X = 12 堆）— 推测`);
    for (const [k, v] of Object.entries(comp)) if (v === "speculative") warn.push(`${COMP_ZH[k]}为推测值`);
    return { tc, fc, hbm, width, bus, raw, payload, eff: effBw, cap, comp, tag, warn, unit: fc.unit };
  }
  function memGeoText(sel, dv) {
    if (dv.hbm) return `${sel.mem_count}×${sel.hbm_height}H×${sel.hbm_die_Gb}Gb`;
    return (dv.fc.id === "discrete" ? "" : dv.fc.id + " ") + `${sel.mem_count}×${dv.width}b`;
  }
  // 场景栏紧凑摘要：如「LPDDR6 4×96b @10667 · 455 GB/s 可用 · 64 GB · 厂商量产」
  function memShortLabel(sel, dv) {
    return `${sel.mem_type} ${memGeoText(sel, dv)} @${sel.mem_rate_MTps} · ${fmtBw(dv.payload)} 可用 · ${trimNum(dv.cap.toPrecision(4))} GB · ${TAG_ZH[dv.tag]}`;
  }
  function memChipText(sel, dv) {
    return `${memGeoText(sel, dv)} @${sel.mem_rate_MTps} · ${fmtBw(dv.payload)} · ${trimNum(dv.cap.toPrecision(4))} GB`;
  }
  function segFill(id, items, cur, attr, onPick) {
    const el = $(id);
    if (!el) return;
    el.innerHTML = items.map((it) =>
      `<button type="button" data-${attr}="${escapeHtml(String(it.v))}" class="${String(it.v) === String(cur) ? "active" : ""}" title="${escapeHtml(it.title || "")}">${escapeHtml(it.label)}${it.tag ? `<i class="dot t-${it.tag}"></i>` : ""}</button>`
    ).join("");
    el.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => onPick(b.dataset[attr])));
  }
  function memChanged() {
    renderMemUi();
    updateSceneSummary();
    markPresetCustom();
    scheduleEval();
  }
  function setMemField(k, v) {
    const next = Object.assign({}, memSel, { [k]: v });
    if (k === "mem_form") Object.assign(next, memDefaults(memSel.mem_type, v));
    if (k === "mem_width_bits") next.mem_cap_GB = null; // 重新取该位宽的默认容量
    memSel = normalizeMemSel(next);
    memChanged();
  }
  function setMemType(t) {
    memSel = normalizeMemSel(memDefaults(t));
    memKind = memKindOf(t);
    memChanged();
  }
  function renderMemUi() {
    if (!memCatalog || !memSel) return;
    memKind = memKindOf(memSel.mem_type);
    const manual = $("manual-mem") && $("manual-mem").checked;
    const tsel = $("mem-type");
    if (tsel && tsel.value !== memSel.mem_type) tsel.value = memSel.mem_type;
    const dv = memDerive(memSel, parseFloat($("efficiency").value));
    const chipSum = $("mem-sum"), chipTag = $("mem-tag"), chip = $("mem-chip");
    if (manual) {
      const n = $("n_channels").value, w = $("width_bits").value, r = $("data_rate_GTs").value;
      chipSum.textContent = `手动几何 ${n}×${w}b @${r} GT/s`;
      chipTag.className = "ptag t-speculative";
      chipTag.textContent = "推测";
      chip.title = "手动几何（⚙ 高级参数 › 存储几何）—— 标为推测";
    } else {
      chipSum.textContent = memChipText(memSel, dv);
      chipTag.className = "ptag t-" + dv.tag;
      chipTag.textContent = TAG_ZH[dv.tag];
      chip.title = memShortLabel(memSel, dv) + " —— 点击展开详细选择";
    }
    chip.classList.toggle("warned", !manual && dv.warn.length > 0);
    // popover
    const tc = dv.tc, fc = dv.fc;
    $("mp-title").textContent = `${tc.id} · ${fc.label}`;
    $("mp-doc").textContent = tc.jedec_doc || "";
    $("mp-form-row").style.display = dv.hbm ? "none" : "";
    $("mp-width-row").style.display = dv.hbm ? "none" : "";
    $("mp-height-row").style.display = dv.hbm ? "" : "none";
    $("mp-die-row").style.display = dv.hbm ? "" : "none";
    $("mp-cap-row").style.display = dv.hbm ? "none" : "";
    if (!dv.hbm) {
      segFill("mp-form", tc.forms.map((f) => ({
        v: f.id, label: f.label,
        tag: weakestTag(f.widths.map((w) => w.tag)),
        title: `${f.label}（${f.unit === "module" ? "128-bit 模组" : "板载封装"}）`,
      })), memSel.mem_form, "v", (v) => setMemField("mem_form", v));
      segFill("mp-width", fc.widths.map((w) => ({
        v: w.bits, label: (fc.unit === "module" ? "" : "x") + w.bits,
        tag: w.tag, title: `${w.bits}-bit · ${TAG_ZH[w.tag]}` + (tc.id === "LPDDR6" && w.bits === 96 ? " · 4×24-bit 通道（8×12-bit 子通道）" : (w.bits === 64 && fc.unit === "package" ? " · 4×16-bit 通道" : "")),
      })), dv.width, "v", (v) => setMemField("mem_width_bits", Number(v)));
      const cs = $("mp-cap");
      cs.innerHTML = capsFor(fc, dv.width).map((c) =>
        `<option value="${c.GB}"${c.GB === memSel.mem_cap_GB ? " selected" : ""}>${c.GB} GB / ${UNIT_ZH[fc.unit] || "颗"} · ${TAG_ZH[c.tag]}</option>`).join("");
      $("mp-cap-label").textContent = fc.unit === "module" ? "单条容量" : "单颗容量";
    } else {
      segFill("mp-height", fc.heights.map((h) => {
        const ct = (fc.cap_tags || []).find((c) => c.height === h && c.die_Gb === memSel.hbm_die_Gb);
        return { v: h, label: h + "H", tag: ct ? ct.tag : "speculative", title: `${h}-high · ${ct ? TAG_ZH[ct.tag] : "推测"}` };
      }), memSel.hbm_height, "v", (v) => setMemField("hbm_height", Number(v)));
      segFill("mp-die", fc.densities.map((d) => {
        const ct = (fc.cap_tags || []).find((c) => c.height === memSel.hbm_height && c.die_Gb === d);
        return { v: d, label: d + " Gb", tag: ct ? ct.tag : "speculative", title: `${d} Gb die · 单堆 ${memSel.hbm_height * d / 8} GB` };
      }), memSel.hbm_die_Gb, "v", (v) => setMemField("hbm_die_Gb", Number(v)));
    }
    $("mp-rate").innerHTML = fc.rates.map((r) =>
      `<option value="${r.MTps}"${r.MTps === memSel.mem_rate_MTps ? " selected" : ""}>${r.MTps} MT/s · ${clocksText(tc.id, r.MTps, r.clocks)} · ${TAG_ZH[r.tag]}</option>`).join("");
    $("mp-count-label").textContent = dv.hbm ? "堆数" : fc.unit === "module" ? "模组数" : "颗数";
    $("mp-count").innerHTML = fc.counts.map((c) =>
      `<option value="${c.n}"${c.n === memSel.mem_count ? " selected" : ""}>${c.n} ${UNIT_ZH[fc.unit] || ""}${c.tag === "speculative" ? " · 推测" : ""}</option>`).join("");
    const pf = tc.payload_factor || 1;
    const effv = parseFloat($("efficiency").value) || DEFAULT_MEM_EFF;
    $("mp-derived").innerHTML =
      `<div><span class="k">总线</span> ${dv.bus} bit <span class="muted">= ${memSel.mem_count} × ${dv.width}b</span></div>` +
      `<div><span class="k">原始带宽</span> ${fmtBwFull(dv.raw)} <span class="muted">= ${dv.bus} × ${memSel.mem_rate_MTps} / 8000</span></div>` +
      `<div><span class="k">可用带宽</span> ${fmtBwFull(dv.payload)}${pf < 1 ? ` <span class="muted">= 原始 × 8/9（BL24 元数据，与效率分开计）</span>` : ' <span class="muted">（无 payload 折损）</span>'}</div>` +
      `<div><span class="k">有效带宽</span> ${fmtBwFull(dv.eff)} <span class="muted">= 可用 × 效率 ${effv.toFixed(2)}「假设」</span></div>` +
      `<div><span class="k">容量</span> ${trimNum(dv.cap.toPrecision(5))} GB <span class="muted">${dv.hbm ? `= ${memSel.mem_count} 堆 × ${memSel.hbm_height}H × ${memSel.hbm_die_Gb} Gb / 8` : `= ${memSel.mem_count} × ${memSel.mem_cap_GB} GB`}（GB = 2³⁰ B）</span></div>`;
    $("mp-tags").innerHTML = Object.entries(dv.comp).map(([k, v]) => `<span class="ctag">${COMP_ZH[k]} ${tagBadge(v)}</span>`).join("") +
      `<span class="ctag strong">综合（最弱项）${tagBadge(dv.tag)}</span>`;
    $("mp-warn").innerHTML = dv.warn.map((w) => `<li>${escapeHtml(w)}</li>`).join("");
    $("mp-warn").hidden = !dv.warn.length;
    updatePackageMeta(dv);
    fitSelect(tsel);
  }
  // v0.30：API capacity_GB / capacity_needed_GB 已为厂商标称 GB（2³⁰ B），与存储摘要同单位；
  // 十进制值见 *_decimal。保留函数以兼容旧调用点。
  function capDisp(card, v) {
    return v;
  }
  function fmtBwFull(gbps) {
    return gbps >= 1000 ? `${trimNum((gbps / 1000).toFixed(3))} TB/s（${gbps.toFixed(1)} GB/s）` : `${gbps.toFixed(1)} GB/s`;
  }
  function openMemPop(open) {
    const pop = $("mem-pop"), chip = $("mem-chip");
    if (!pop) return;
    const on = open != null ? !!open : pop.hidden;
    if (on && $("manual-mem").checked) {
      openDrawer("acc-mem");
      return;
    }
    pop.hidden = !on;
    chip.setAttribute("aria-expanded", on ? "true" : "false");
    document.body.classList.toggle("mem-pop-open", on);
    const bd = $("mem-pop-backdrop");
    if (bd) bd.hidden = !on;
    if (on) positionMemPop();
  }
  function positionMemPop() {
    const pop = $("mem-pop"), chip = $("mem-chip");
    if (!pop || pop.hidden) return;
    const r = chip.getBoundingClientRect();
    const vw = window.innerWidth;
    // v0.30: 窄屏 → 底部抽屉（bottom sheet），不再覆盖存储芯片 / 顶部控件
    pop.classList.toggle("sheet", vw <= 768);
    if (vw <= 768) {
      pop.style.left = pop.style.right = pop.style.top = pop.style.width = "";
      return;
    }
    const w = Math.min(460, vw - 16);
    pop.style.width = w + "px";
    pop.style.right = "auto";
    pop.style.left = Math.max(8, Math.min(r.left, vw - w - 8)) + "px";
    pop.style.top = r.bottom + 6 + "px";
  }
  // 旧链接 / 旧预设中的 package_id → 结构化（GET /api/memory?resolve=）
  async function resolveLegacyPackage(pid) {
    try {
      const r = await apiGet("/api/memory?resolve=" + encodeURIComponent(pid));
      const kw = r.workbench_kwargs || {};
      memSel = normalizeMemSel(kw);
      const note = (r.resolved && r.resolved.legacy_note) || "";
      showToast(note ? `旧存储 id ${pid} → ${memSel.mem_type}（${note}）` : `存储 id ${pid} → ${memSel.mem_type}`, "ok");
      return true;
    } catch (e) {
      showToast(`无法解析旧存储 id ${pid} —— 使用默认 ${DEFAULT_MEM_TYPE}`);
      memSel = normalizeMemSel(memDefaults(DEFAULT_MEM_TYPE));
      return false;
    }
  }

  function updatePackageMeta(dv) {
    const el = $("package-meta");
    if (!el) return;
    if ($("manual-mem").checked) {
      el.textContent = `手动几何：${$("n_channels").value} × ${$("width_bits").value}b @ ${$("data_rate_GTs").value} GT/s · ${$("capacity_GB").value} GB（推测）`;
      return;
    }
    if (!memSel || !dv) return;
    el.textContent = `${memShortLabel(memSel, dv)} —— 总线 ${dv.bus} bit · 原始 ${dv.raw.toFixed(1)} GB/s · 可用 ${dv.payload.toFixed(1)} GB/s · 有效 ${dv.eff.toFixed(1)} GB/s「效率假设」`;
  }

  function populateCompute() {
    const sel = $("compute");
    const prev = sel.value;
    sel.innerHTML = "";
    // Prefer primaries: cores then clusters
    const order = [...computeCache].sort((a, b) => {
      if (a.level !== b.level) return a.level === "core" ? -1 : 1;
      return a.peak_tops - b.peak_tops;
    });
    const cores = [], clusters = [];
    for (const c of order) {
      const opt = document.createElement("option");
      opt.value = c.id;
      opt.textContent = computeShortLabel(c);
      opt.title = computeFullLabel(c);
      (c.level === "core" ? cores : clusters).push(opt);
    }
    appendGrouped(sel, [["单核", cores], ["集群", clusters]]);
    // Default ~100T class: 16 × 6.25 → not in catalog exactly; pick cluster_64t or nearest
    if (prev && [...sel.options].some((o) => o.value === prev)) {
      sel.value = prev;
    } else {
      const prefer = ["cluster_256t", "cluster_64t", "core_16t", "cluster_128t"];
      for (const p of prefer) {
        if ([...sel.options].some((o) => o.value === p)) {
          sel.value = p;
          break;
        }
      }
    }
    // Also add a synthetic "custom_100t" feel via manual default
    updateComputeMeta();
  }

  function updateComputeMeta() {
    const id = $("compute").value;
    const c = computeCache.find((x) => x.id === id);
    $("compute-meta").textContent = c
      ? `${LEVEL_ZH[c.level] || c.level} 峰值 ${c.peak_tops} TOPS @ ${(c.frequency_hz / 1e9).toFixed(2)} GHz${ASSUMED_TAG} —— ${zhText(c.note)}`
      : "—";
    $("compute").title = c ? computeFullLabel(c) : "";
    fitSelect($("compute"));
    if (c && !$("manual-compute").checked) {
      $("n_cores").value = c.n_cores;
      $("tops_per_core").value = c.tops_per_core;
    }
  }

  function syncTpFromChips() {
    if ($("lock-parallel").checked) return;
    if (syncingParallel) return;
    syncingParallel = true;
    const chips = Math.max(1, parseInt($("chip_count").value, 10) || 1);
    $("tp").value = chips;
    $("pp").value = 1;
    $("ep").value = 1;
    syncingParallel = false;
    updateParallelHint();
  }

  function updateParallelHint() {
    const chips = Math.max(1, parseInt($("chip_count").value, 10) || 1);
    const tp = Math.max(1, parseInt($("tp").value, 10) || 1);
    const pp = Math.max(1, parseInt($("pp").value, 10) || 1);
    const ep = Math.max(1, parseInt($("ep").value, 10) || 1);
    const prod = tp * pp * ep;
    const el = $("parallel-hint");
    const ps = $("par-summary");
    if (ps) {
      ps.textContent = `TP ${tp} · PP ${pp} · EP ${ep}` + (prod !== chips ? " ⚠" : "") + ($("lock-parallel").checked ? " 🔒" : "");
      ps.classList.toggle("warn", prod !== chips);
    }
    updateAdvancedStatus();
    if (prod !== chips) {
      el.textContent = `⚠ TP×PP×EP = ${prod} ≠ 芯片数 ${chips} —— 修正前无法评估`;
      el.style.color = "var(--danger)";
      $("btn-eval").disabled = true;
      showToast(`TP×PP×EP = ${prod} ≠ 芯片数 ${chips}`);
    } else {
      el.textContent = $("lock-parallel").checked
        ? `显式并行有效：${tp}×${pp}×${ep} = ${chips}`
        : `默认映射：TP = 芯片数（${chips}），PP=1，EP=1`;
      el.style.color = "";
      $("btn-eval").disabled = false;
    }
  }

  function collectBody() {
    const chips = Math.max(1, parseInt($("chip_count").value, 10) || 1);
    const body = {
      model_id: $("series").value,
      chip_count: chips,
      batch: Math.max(1, parseInt($("batch").value, 10) || 1),
      efficiency: parseFloat($("efficiency").value),
      mac_efficiency: parseFloat($("mac_eff").value),
      freq_ghz: parseFloat($("freq_ghz").value),
    };
    if (domain === "llm") {
      body.prompt = Math.max(1, parseInt($("prompt").value, 10) || 512);
      body.ctx = Math.max(1, parseInt($("ctx").value, 10) || 512);
    } else if (domain === "video") {
      body.n_denoise = Math.max(1, parseInt($("n_denoise").value, 10) || 50);
      body.n_frames = Math.max(1, parseInt($("n_frames").value, 10) || 16);
    } else if (domain === "protein") {
      body.seq_len = Math.max(1, parseInt($("seq_len").value, 10) || 512);
    }
    if ($("dtype").value) body.dtype = $("dtype").value;
    if ($("quant").value) body.quant = $("quant").value;

    if ($("lock-parallel").checked) {
      body.tp = Math.max(1, parseInt($("tp").value, 10) || 1);
      body.pp = Math.max(1, parseInt($("pp").value, 10) || 1);
      body.ep = Math.max(1, parseInt($("ep").value, 10) || 1);
    }

    if ($("manual-mem").checked) {
      body.mem_kind = memKind;
      body.n_channels = parseInt($("n_channels").value, 10);
      body.width_bits = parseInt($("width_bits").value, 10);
      body.data_rate_GTs = parseFloat($("data_rate_GTs").value);
      const cg = parseFloat($("capacity_GB").value);
      if (Number.isFinite(cg) && cg > 0) body.capacity_GB = cg;
      body.efficiency = parseFloat($("efficiency").value);
    } else if (memSel) {
      // v0.29：结构化存储字段（/api/memory 目录）；效率覆盖叠加在其上
      body.mem_type = memSel.mem_type;
      body.mem_form = memSel.mem_form;
      body.mem_rate_MTps = memSel.mem_rate_MTps;
      body.mem_count = memSel.mem_count;
      if (memKindOf(memSel.mem_type) === "HBM") {
        body.hbm_height = memSel.hbm_height;
        body.hbm_die_Gb = memSel.hbm_die_Gb;
      } else {
        body.mem_width_bits = memSel.mem_width_bits;
        body.mem_cap_GB = memSel.mem_cap_GB;
      }
      body.efficiency = parseFloat($("efficiency").value);
    }
    // v0.29 通信同步 / KV 切分（假设）
    body.c2c_latency_us = parseFloat($("c2c_latency_us").value);
    body.sync_overlap = parseFloat($("sync_overlap").value);
    body.attn_parallel = attnParallel;
    // v0.31 投机解码 / MoE 切分 / PP 微批（假设）
    body.spec_k = Math.max(0, Math.min(8, parseInt($("spec_k").value, 10) || 0));
    body.spec_accept = parseFloat($("spec_accept").value);
    body.spec_draft = specDraft;
    body.spec_draft_frac = parseFloat($("spec_draft_frac").value) || 0.1;
    body.moe_shard = moeShard;
    body.decode_mb = Math.max(0, parseInt($("decode_mb").value, 10) || 0);

    if ($("manual-compute").checked) {
      body.n_cores = parseInt($("n_cores").value, 10);
      body.tops_per_core = parseFloat($("tops_per_core").value);
    } else {
      body.compute_id = $("compute").value;
    }

    // v0.26 assumed compute extras (default off)
    const oh = parseFloat($("non_gemm_overhead").value);
    if (Number.isFinite(oh) && oh > 0) body.non_gemm_overhead = oh;
    if ($("use_finer_non_gemm") && $("use_finer_non_gemm").checked) {
      const sf = parseFloat($("softmax_frac").value) || 0;
      const rf = parseFloat($("rope_frac").value) || 0;
      const nf = parseFloat($("norm_frac").value) || 0;
      if (sf > 0) body.softmax_frac = sf;
      if (rf > 0) body.rope_frac = rf;
      if (nf > 0) body.norm_frac = nf;
    }
    const dmfRaw = ($("dtype_mac_factors") && $("dtype_mac_factors").value || "").trim();
    if (dmfRaw) {
      try {
        body.dtype_mac_factors = JSON.parse(dmfRaw);
      } catch (e) {
        showToast("dtype MAC 系数 JSON 格式无效");
      }
    }
    Object.assign(body, collectEcon());
    return body;
  }


  // --- v0.26 assumed compute extras ---
  const EXAMPLE_DTYPE_MAC = {"fp16": 1.0, "fp8": 2.0, "int8": 2.0, "int4": 4.0};
  function updateAssumedComputeBadge() {
    const badge = $("assumed-compute-badge");
    if (!badge) return;
    const oh = parseFloat($("non_gemm_overhead") && $("non_gemm_overhead").value) || 0;
    const finer = $("use_finer_non_gemm") && $("use_finer_non_gemm").checked;
    const dmf = ($("dtype_mac_factors") && $("dtype_mac_factors").value || "").trim();
    const active = oh > 0 || finer || dmf.length > 0;
    badge.textContent = active ? "已覆盖" : "可选启用";
    badge.classList.toggle("assumed", !active);
    updateAdvancedStatus();
  }
  function wireAssumedCompute() {
    if (assumedWired) return; // v0.28: wire exactly once (was only wired on ?c= restore)
    assumedWired = true;
    const bindSlider = (id) => {
      const el = $(id);
      if (!el) return;
      const val = $(id + "-val");
      el.addEventListener("input", () => {
        if (val) val.textContent = Number(el.value).toFixed(2);
        updateAssumedComputeBadge();
        scheduleEval && scheduleEval();
      });
    };
    ["non_gemm_overhead", "softmax_frac", "rope_frac", "norm_frac"].forEach(bindSlider);
    if ($("use_finer_non_gemm")) {
      $("use_finer_non_gemm").addEventListener("change", () => {
        updateAssumedComputeBadge();
        scheduleEval && scheduleEval();
      });
    }
    if ($("dtype_mac_factors")) {
      $("dtype_mac_factors").addEventListener("change", () => {
        updateAssumedComputeBadge();
        scheduleEval && scheduleEval();
      });
    }
    if ($("btn-mac-example")) {
      $("btn-mac-example").addEventListener("click", () => {
        $("dtype_mac_factors").value = JSON.stringify(EXAMPLE_DTYPE_MAC);
        updateAssumedComputeBadge();
        showToast("已载入 EXAMPLE dtype MAC 系数（假设值，非硅片实测）");
        scheduleEval && scheduleEval();
      });
    }
    if ($("btn-mac-clear")) {
      $("btn-mac-clear").addEventListener("click", () => {
        $("dtype_mac_factors").value = "";
        if ($("non_gemm_overhead")) {
          $("non_gemm_overhead").value = 0;
          $("non_gemm_overhead-val").textContent = "0.00";
        }
        if ($("use_finer_non_gemm")) $("use_finer_non_gemm").checked = false;
        ["softmax_frac", "rope_frac", "norm_frac"].forEach((id) => {
          if ($(id)) { $(id).value = 0; if ($(id+"-val")) $(id+"-val").textContent = "0.00"; }
        });
        updateAssumedComputeBadge();
        scheduleEval && scheduleEval();
      });
    }
  }

  // --- v0.24 energy / cost stub (ASSUMED) ---
  function collectEcon() {
    const out = {};
    let any = false;
    for (const id of ECON_NUM_FIELDS) {
      const el = $(id);
      if (!el) continue;
      const raw = String(el.value || "").trim();
      if (raw === "") continue;
      const v = parseFloat(raw);
      if (Number.isFinite(v)) {
        out[id] = v;
        any = true;
      }
    }
    if (any) {
      const pu = parseFloat($("power_util").value);
      const dc = parseFloat($("duty_cycle").value);
      if (Number.isFinite(pu)) out.power_util = pu;
      if (Number.isFinite(dc)) out.duty_cycle = dc;
    }
    return out;
  }

  function setEconUi(vals) {
    for (const id of ECON_NUM_FIELDS) {
      const el = $(id);
      if (!el) continue;
      const v = vals ? vals[id] : null;
      el.value = v == null || v === "" ? "" : v;
    }
    const pu = vals && vals.power_util != null ? Number(vals.power_util) : DEFAULT_POWER_UTIL;
    const dc = vals && vals.duty_cycle != null ? Number(vals.duty_cycle) : DEFAULT_DUTY_CYCLE;
    $("power_util").value = pu;
    $("power_util-val").textContent = pu.toFixed(2);
    $("duty_cycle").value = dc;
    $("duty_cycle-val").textContent = dc.toFixed(2);
  }

  function loadEconExample() {
    if (!econExample || econExample.tdp_w == null && econExample.watts_per_tops == null) {
      showToast("/api/presets 未提供 EXAMPLE 能耗/成本占位值");
      return;
    }
    setEconUi(econExample);
    showToast("已载入 EXAMPLE 占位值（虚构数字 —— 请替换为你的数据）", "ok");
    scheduleEval();
  }

  function renderEcon(card) {
    const basis = $("econ-basis");
    const on = !!(card && card.econ_configured);
    const d = (card && card.domain) || domain || "llm";
    const eK = $("m-est-energy-k"), cK = $("m-est-cost-k");
    const unit = d === "video" ? "帧" : d === "protein" ? "序列" : "token";
    if (eK) eK.textContent = "能效 J / " + unit;
    if (cK) cK.textContent = d === "llm" ? "成本 $ / MTok" : "系统成本 $";
    ["kpi-energy", "kpi-cost"].forEach((id) => { const el = $(id); if (el) el.classList.toggle("off", !on); });
    if (!on) {
      $("m-est-energy").textContent = "未启用";
      $("m-est-power").innerHTML = '<button type="button" class="link" data-open-acc="econ-section">⚙ 设置能耗参数</button>';
      $("m-est-mtok").textContent = "未启用";
      $("m-est-cost").innerHTML = '<button type="button" class="link" data-open-acc="econ-section">⚙ 设置成本参数</button>';
      if (basis) basis.textContent = "";
      return;
    }
    let ev = card.est_energy_per_token_J;
    if (d === "video") ev = card.est_energy_per_frame_J;
    else if (d === "protein") ev = card.est_energy_per_seq_J;
    $("m-est-energy").textContent = Number(ev) > 0 ? Number(ev).toPrecision(4) + " J" : "—";
    $("m-est-power").textContent = "功耗 " + fmt(card.est_power_W, 1) + " W";
    const sys = Number(card.est_system_cost_usd) > 0
      ? "$" + Math.round(card.est_system_cost_usd).toLocaleString()
      : "—";
    if (d === "llm") {
      $("m-est-mtok").textContent = Number(card.est_usd_per_Mtok) > 0
        ? "$" + Number(card.est_usd_per_Mtok).toPrecision(4)
        : "—";
      $("m-est-cost").textContent = Number(card.est_usd_per_Mtok) > 0
        ? "系统 " + sys
        : "系统 " + sys + " · 需 $/kWh 或摊销年限";
    } else {
      $("m-est-mtok").textContent = sys;
      $("m-est-cost").textContent = "$/MTok 仅适用 LLM";
    }
    if (basis) basis.textContent = "能耗 / 成本估算依据：" + zhText(card.econ_basis || "") + " " + ASSUMED_TAG;
  }

  // --- v0.24 scenario presets ---
  function applyPreset(pid) {
    const pre = presetsCache.find((x) => x.id === pid);
    if (!pre) {
      showToast("未知预设 " + pid + "（目录未加载？）");
      if ($("preset-select")) $("preset-select").value = "";
      return;
    }
    applyingPreset = true;
    $("manual-mem").checked = false;
    $("manual-mem-fields").style.display = "none";
    $("manual-compute").checked = false;
    $("manual-compute-fields").style.display = "none";
    $("compute").disabled = false;
    if (pre.mem) memSel = normalizeMemSel(pre.mem);
    renderMemUi();
    if ([...$("compute").options].some((o) => o.value === pre.compute_id)) {
      $("compute").value = pre.compute_id;
    }
    updateComputeMeta();
    $("chip_count").value = pre.chip_count;
    if (pre.tp != null) {
      $("lock-parallel").checked = true;
      $("tp").value = pre.tp;
      $("pp").value = pre.pp || 1;
      $("ep").value = pre.ep || 1;
      updateParallelHint();
    } else {
      $("lock-parallel").checked = false;
      syncTpFromChips();
    }
    updateOverrideBadge();
    if ($("preset-select")) $("preset-select").value = pid;
    fitSelect($("preset-select"));
    applyingPreset = false;
    const hint = $("preset-hint");
    const pz = PRESET_ZH[pre.id] || {};
    if (hint) hint.textContent = `已应用 ${pre.id}：${pz.label || pre.label} —— ${pz.note || pre.note || ""} ${ASSUMED_TAG}`;
    showToast(`已应用预设 ${pre.id}`, "ok");
    updateSceneSummary();
    scheduleEval();
  }

  // v0.28：用户手动改动核心参数后，预设下拉回到「自定义」
  function markPresetCustom() {
    if (applyingPreset || restoringUrl) return;
    const sel = $("preset-select");
    if (sel && sel.value) {
      sel.value = "";
      fitSelect(sel);
      const hint = $("preset-hint");
      if (hint) hint.textContent = "自定义（已偏离预设）—— 预设仅组合目录 id，不含功耗 / 价格。";
    }
  }


  function updateOverrideBadge() {
    const eff = parseFloat($("efficiency").value);
    const mac = parseFloat($("mac_eff").value);
    const freq = parseFloat($("freq_ghz").value);
    const active =
      Math.abs(eff - DEFAULT_MEM_EFF) > 1e-6 ||
      Math.abs(mac - DEFAULT_MAC_EFF) > 1e-6 ||
      Math.abs(freq - DEFAULT_FREQ_GHZ) > 1e-6;
    const badge = $("override-badge");
    const sec = $("override-section");
    if (badge) badge.hidden = !active;
    if (sec) sec.classList.toggle("override-active", active);
    updateAdvancedStatus();
    return active;
  }

  // [API 字段, 显示标签]：字段名不变，仅标签中文化
  const DUAL_FIELDS = [
    ["TTFT_ms", "TTFT"],
    ["TPOT_ms", "TPOT"],
    ["TTFC_ms", "TTFC"],
    ["frames_per_s", "帧 / 秒"],
    ["time_per_seq_ms", "每序列时间"],
    ["pair_bytes", "pair 字节"],
    ["t_compute_ms", "计算时间"],
    ["t_dram_ms", "DRAM 时间"],
    ["t_c2c_ms", "C2C 时间"],
    ["t_fabric_ms", "Fabric 时间"],
    ["t_bubble_ms", "PP 遍历 / 气泡"],
    ["t_draft_ms", "投机草稿时间"],
    ["TPOT_step_ms", "TPOT 步时延"],
    ["spec_tokens_per_step", "每步期望 token"],
    ["bytes_W", "权重字节 W"],
    ["bytes_KV", "KV 字节"],
    ["bytes_coll", "集合通信字节"],
    ["util", "利用率"],
    ["peak_tops", "峰值 TOPS"],
    ["mem_eff_GBps", "有效带宽 (GB/s)"],
    ["chips", "芯片数"],
    ["tp", "TP"],
    ["pp", "PP"],
    ["ep", "EP"],
    ["scale_efficiency", "扩展效率"],
    ["speedup", "加速比"],
    ["t_single_primary_ms", "单卡主指标"],
    ["est_power_W", "估算功耗 W" + ASSUMED_TAG],
    ["est_energy_per_token_J", "估算 J / token" + ASSUMED_TAG],
    ["est_energy_per_frame_J", "估算 J / 帧" + ASSUMED_TAG],
    ["est_energy_per_seq_J", "估算 J / 序列" + ASSUMED_TAG],
    ["est_system_cost_usd", "估算系统成本 $" + ASSUMED_TAG],
    ["est_usd_per_Mtok", "估算 $ / MTok" + ASSUMED_TAG],
  ];

  function dualSummaryHtml(card) {
    if (!card) return "<em>空</em>";
    const d = card.domain || "llm";
    let primary = "—", secondary = "—";
    if (d === "video") {
      primary = `TTFC ${fmtTime(card.TTFC_ms != null ? card.TTFC_ms : card.TTFT_ms)}`;
      secondary = `${fmt(card.frames_per_s, 4)} 帧/秒`;
    } else if (d === "protein") {
      primary = `每序列 ${fmtTime(card.time_per_seq_ms != null ? card.time_per_seq_ms : card.TTFT_ms)}`;
      secondary = `pair ${fmtBytes(card.pair_bytes)}`;
    } else {
      primary = `TTFT ${fmtTime(card.TTFT_ms)}`;
      secondary = `TPOT ${fmtTime(card.TPOT_ms)}`;
    }
    const tags = [
      domainZh(d),
      `${fmtInt(card.chips)} 芯片 · TP/PP/EP ${card.tp}/${card.pp}/${card.ep}`,
      card.mem_summary ? `${card.mem_summary}（有效 ${fmt(card.mem_eff_GBps, 0)} GB/s）` : `${card.mem_kind || "?"} ${fmt(card.mem_eff_GBps, 0)} GB/s`,
      `${fmt(card.peak_tops, 1)} T`,
      `瓶颈 ${wallZh(card.wall)}`,
    ];
    if (card.oom) tags.push("OOM");
    return (
      `<div class="dual-main">${escapeHtml(primary)} · ${escapeHtml(secondary)}</div>` +
      `<div class="dual-tags">${tags.map((t) => `<span class="${t === "OOM" ? "tag-oom" : ""}">${escapeHtml(t)}</span>`).join("")}</div>`
    );
  }

  function renderDualCards() {
    const aBody = $("dual-a-body");
    const bBody = $("dual-b-body");
    const aLab = $("dual-a-label");
    const bLab = $("dual-b-label");
    if (aBody) aBody.innerHTML = dualCardA ? dualSummaryHtml(dualCardA) : "<em>从当前 MetricsCard 固定 A</em>";
    if (bBody) bBody.innerHTML = dualCardB ? dualSummaryHtml(dualCardB) : "<em>从当前 MetricsCard 固定 B</em>";
    if (aLab) aLab.textContent = dualCardA ? (dualCardA.model_id || "A") : "—";
    if (bLab) bLab.textContent = dualCardB ? (dualCardB.model_id || "B") : "—";
    renderDualDelta();
    const meta = $("dual-meta");
    if (meta) {
      meta.textContent = dualCardA && dualCardB
        ? "下方为逐字段 Δ（B 相对 A）"
        : "固定两次评估（或最近一次 + 基线）· 逐字段 Δ";
    }
    const st = $("tab-ab-state");
    if (st) st.textContent = dualCardA && dualCardB ? "A·B" : dualCardA ? "A" : dualCardB ? "B" : "";
  }

  function renderDualDelta() {
    const wrap = $("dual-delta-wrap");
    const tbody = $("dual-delta-tbody");
    if (!wrap || !tbody) return;
    if (!dualCardA || !dualCardB) {
      wrap.hidden = true;
      tbody.innerHTML = "";
      return;
    }
    wrap.hidden = false;
    const rows = [];
    // wall as string
    rows.push({
      field: "瓶颈",
      a: wallZh(dualCardA.wall),
      b: wallZh(dualCardB.wall),
      delta: dualCardA.wall === dualCardB.wall ? "相同" : `${wallZh(dualCardA.wall)}→${wallZh(dualCardB.wall)}`,
      pct: null,
      cls: dualCardA.wall === dualCardB.wall ? "same" : "",
    });
    for (const [key, label] of DUAL_FIELDS) {
      const av = dualCardA[key];
      const bv = dualCardB[key];
      if (av == null && bv == null) continue;
      const aN = Number(av);
      const bN = Number(bv);
      if (!Number.isFinite(aN) && !Number.isFinite(bN)) continue;
      // skip zeros for domain-irrelevant metrics
      if ((aN === 0 || !Number.isFinite(aN)) && (bN === 0 || !Number.isFinite(bN))) continue;
      const dlt = (Number.isFinite(aN) && Number.isFinite(bN)) ? (bN - aN) : null;
      const pct = deltaPct(bN, aN);
      let cls = "same";
      if (dlt != null && Math.abs(dlt) > 1e-12) {
        // lower is better for time-like fields
        const lowerBetter = /_ms$|bytes|pair_bytes|^est_/.test(key);
        const higherBetter = /frames_per_s|util|peak_tops|mem_eff|scale_efficiency|^speedup$/.test(key);
        if (lowerBetter) cls = dlt < 0 ? "better" : "worse";
        else if (higherBetter) cls = dlt > 0 ? "better" : "worse";
        else cls = "";
      }
      rows.push({
        key: key,
        field: label,
        a: Number.isFinite(aN) ? aN : av,
        b: Number.isFinite(bN) ? bN : bv,
        delta: dlt,
        pct: pct,
        cls: cls,
      });
    }
    const showAll = !!($("dual-show-all") && $("dual-show-all").checked);
    const shown = showAll ? rows : rows.filter((r) => r.cls !== "same");
    if (!shown.length) {
      tbody.innerHTML = `<tr><td colspan="5" class="muted">A 与 B 的所有字段相同（勾选「显示全部字段」查看）</td></tr>`;
      return;
    }
    tbody.innerHTML = shown
      .map((r) => {
        const isT = /_ms$/.test(r.key || "");
        const isB = /^bytes_|pair_bytes/.test(r.key || "");
        const nf = (v) => isT ? fmtTime(v) : isB ? fmtBytes(v)
          : (Number.isInteger(v) ? String(v) : trimNum(Number(v).toPrecision(6)));
        const aStr = typeof r.a === "number" ? nf(r.a) : String(r.a);
        const bStr = typeof r.b === "number" ? nf(r.b) : String(r.b);
        const dStr =
          r.delta == null
            ? "—"
            : typeof r.delta === "number"
              ? (isT ? fmtTimeSigned(r.delta) : isB ? fmtBytesSigned(r.delta)
                : (r.delta >= 0 ? "+" : "") + nf(r.delta))
              : String(r.delta);
        const pStr =
          r.pct == null ? "—" : (r.pct >= 0 ? "+" : "") + fmt(r.pct, 2) + "%";
        const raw = (v) => (typeof v === "number" && (isT || isB) ? ` title="${escapeHtml(String(v) + (isT ? " ms" : " B"))}"` : "");
        return (
          `<tr><td class="field">${escapeHtml(r.field)}</td>` +
          `<td class="num"${raw(r.a)}>${escapeHtml(aStr)}</td>` +
          `<td class="num"${raw(r.b)}>${escapeHtml(bStr)}</td>` +
          `<td class="num delta ${r.cls}">${escapeHtml(dStr)}</td>` +
          `<td class="num delta ${r.cls}">${escapeHtml(pStr)}</td></tr>`
        );
      })
      .join("");
  }

  function pinDual(slot, card) {
    if (!card) {
      showToast("请先评估，再固定 A|B 卡");
      return;
    }
    if (slot === "A") dualCardA = card;
    else dualCardB = card;
    renderDualCards();
    pushConfigToUrl();
    showToast(`已固定 ${slot} · ${card.model_id || "?"}` + (dualCardA && dualCardB ? " —— 见「A|B 对比」标签" : ""), "ok");
  }

  function clearDual() {
    dualCardA = null;
    dualCardB = null;
    renderDualCards();
    pushConfigToUrl();
  }

  function fmtBytes(n) {
    const v = Number(n);
    if (n == null || !Number.isFinite(v)) return "—";
    const u = ["B", "KB", "MB", "GB", "TB"];
    let x = v, i = 0;
    while (Math.abs(x) >= 1000 && i < u.length - 1) { x /= 1000; i++; }
    return (i === 0 ? String(Math.round(x)) : trimNum(x.toPrecision(4))) + " " + u[i];
  }
  function fmtBytesSigned(n) {
    const v = Number(n);
    if (!Number.isFinite(v)) return "—";
    return (v > 0 ? "+" : v < 0 ? "−" : "") + fmtBytes(Math.abs(v));
  }

  function renderCard(card) {
    lastCard = card;
    const d = card.domain || domain || "llm";
    const sub1 = $("kpi-sub-1"), sub2 = $("kpi-sub-2");
    if (d === "video") {
      $("m-ttft").textContent = fmtTime(card.TTFC_ms != null ? card.TTFC_ms : card.TTFT_ms);
      $("m-ttft").title = rawMs(card.TTFC_ms != null ? card.TTFC_ms : card.TTFT_ms);
      $("m-tpot").textContent = fmt(card.frames_per_s, 4);
      $("m-tpot").title = `frames_per_s = ${card.frames_per_s}`;
      if (sub1) sub1.textContent = `${fmtInt(card.n_frames)} 帧 · ${fmtInt(card.n_denoise)} 步去噪`;
      if (sub2) sub2.textContent = `越高越好 · 批 ${fmtInt(card.batch)}`;
      if ($("m-extra")) $("m-extra").textContent = fmtInt(card.n_denoise);
      if ($("m-cap"))
        $("m-cap").textContent =
          fmtGB(capDisp(card, card.capacity_needed_GB)) + " / " + fmtGB(capDisp(card, card.capacity_GB)) + " GB";
      if ($("m-cap")) $("m-cap").title = `capacity_needed_GB = ${card.capacity_needed_GB} · capacity_GB = ${card.capacity_GB}（GB = 2³⁰ B；十进制 ${card.capacity_GB_decimal} GB）`;
      $("i-wl").textContent = `${card.n_frames} / ${card.n_denoise} / ${card.batch}`;
    } else if (d === "protein") {
      $("m-ttft").textContent = fmtTime(card.time_per_seq_ms != null ? card.time_per_seq_ms : card.TTFT_ms);
      $("m-ttft").title = rawMs(card.time_per_seq_ms != null ? card.time_per_seq_ms : card.TTFT_ms);
      $("m-tpot").textContent = fmtBytes(card.pair_bytes);
      if (sub1) sub1.textContent = `L = ${fmtInt(card.seq_len)} · 批 ${fmtInt(card.batch)}`;
      if (sub2) sub2.textContent = "pair L² 激活张量";
      $("m-tpot").title = `${fmtInt(card.pair_bytes)} B`;
      if ($("m-extra")) $("m-extra").textContent = fmtInt(card.seq_len);
      if ($("m-cap"))
        $("m-cap").textContent =
          fmtGB(capDisp(card, card.capacity_needed_GB)) + " / " + fmtGB(capDisp(card, card.capacity_GB)) + " GB" +
          (card.oom ? " · OOM" : "");
      if ($("m-cap")) $("m-cap").title = `capacity_needed_GB = ${card.capacity_needed_GB} · capacity_GB = ${card.capacity_GB}（GB = 2³⁰ B；十进制 ${card.capacity_GB_decimal} GB）`;
      $("i-wl").textContent = `${card.seq_len} / ${card.batch}`;
      if ($("i-pair")) {
        $("i-pair").textContent = fmtBytes(card.pair_bytes);
        $("i-pair").title = `${fmtInt(card.pair_bytes)} B`;
      }
    } else {
      $("m-ttft").textContent = fmtTime(card.TTFT_ms);
      $("m-tpot").textContent = fmtTime(card.TPOT_ms);
      $("m-ttft").title = rawMs(card.TTFT_ms);
      $("m-tpot").title = rawMs(card.TPOT_ms);
      if (sub1) sub1.textContent = `prefill · 提示 ${fmtInt(card.prompt_len)}`;
      const ctxTxt = `decode · 上下文 ${fmtInt(card.decode_seq_len)}`;
      const sub2t = [];
      if (card.spec_k > 0) sub2t.push(`k=${card.spec_k} · 步 ${fmtTime(card.TPOT_step_ms)} ÷ ${fmt(card.spec_tokens_per_step, 2)}`);
      if (card.pp > 1 && card.decode_mb_eff > 1) sub2t.push(`PP 微批 ${card.decode_mb_eff}×${card.decode_microbatch}`);
      if (sub2) {
        // v0.31: compact sub when spec / PP micro-batches are active; full text in the tooltip
        sub2.textContent = sub2t.length ? sub2t.join(" · ") : ctxTxt;
        sub2.title = [ctxTxt].concat(sub2t).join(" · ");
      }
      if (card.spec_k > 0) $("m-tpot").title = rawMs(card.TPOT_ms) + ` = 步时延 ${rawMs(card.TPOT_step_ms)} / 期望 ${card.spec_tokens_per_step} token（a=${card.spec_accept}，假设）`;
      $("i-wl").textContent = `${card.prompt_len} / ${card.decode_seq_len} / ${card.batch}`;
    }
    $("m-wall").textContent = wallZh(card.wall);
    $("m-wall").title = card.wall ? "wall = " + card.wall : "";
    const wallCard = $("m-wall-card");
    wallCard.className = "metric kpi wall-" + String(card.wall || "").toLowerCase();
    const wsub = $("m-wall-sub");
    if (wsub) {
      const tc = fmtTime(card.t_compute_ms, 3), td = fmtTime(card.t_dram_ms, 3);
      const uc = tc.split(" ")[1], ud = td.split(" ")[1];
      wsub.textContent = uc && uc === ud
        ? `计算 ${tc.split(" ")[0]} · 访存 ${td}`
        : `计算 ${tc} · 访存 ${td}`;
      wsub.title = `t_compute = ${rawMs(card.t_compute_ms)} · t_dram = ${rawMs(card.t_dram_ms)}`;
    }
    $("m-oom").textContent = card.oom ? "OOM" : "可容纳";
    $("m-oom-card").className = "metric kpi" + (card.oom ? " oom" : " fits");
    const oomBadge = $("m-oom-badge");
    if (oomBadge) oomBadge.hidden = !card.oom;
    const capSub = $("m-cap-sub");
    if (capSub) {
      const g = (v) => (Math.abs(Number(v)) >= 1000 ? String(Math.round(Number(v))) : trimNum(Number(v).toPrecision(3)));
      if (Number(card.capacity_needed_GB) > 0 && card.capacity_GB != null)
        capSub.textContent = `需 ${g(capDisp(card, card.capacity_needed_GB))} / 可用 ${g(capDisp(card, card.capacity_GB))} GB`;
      else if (card.capacity_GB != null)
        capSub.textContent = `封装容量 ${g(capDisp(card, card.capacity_GB))} GB / 卡`;
      else capSub.textContent = "";
      capSub.title = card.capacity_GB != null
        ? `capacity_needed_GB = ${card.capacity_needed_GB} · capacity_GB = ${card.capacity_GB}（GB = 2³⁰ B；十进制 ${card.capacity_GB_decimal} GB）` : "";
    }
    const se = card.scale_efficiency != null ? Number(card.scale_efficiency) : null;
    const sp = card.speedup != null ? Number(card.speedup) : null;
    if ($("m-scale-eff")) {
      $("m-scale-eff").textContent = se != null && Number.isFinite(se) ? fmt(se, 3) : "—";
      const sc = $("m-scale-card");
      if (sc) {
        let cls = "metric kpi";
        if (se != null && Number.isFinite(se)) {
          if (se < 0.85) cls += " scale-low";
          else cls += " scale-ok";
        }
        sc.className = cls;
        sc.title = card.scale_metric
          ? `scale_efficiency=(t_single/t_multi)/芯片数，基于 ${card.scale_metric}；理想值 1.0`
          : "扩展效率（相对单卡）";
      }
      const ssub = $("m-scale-sub");
      if (ssub) ssub.textContent = card.scale_metric
        ? `相对单卡 · ${metricLabel(card.scale_metric === "time_per_seq" ? "time_per_seq_ms" : card.scale_metric + "_ms")}`
        : "相对单卡（理想 1.0）";
    }
    if ($("m-speedup")) {
      $("m-speedup").textContent = sp != null && Number.isFinite(sp) ? fmt(sp, 2) + "×" : "—";
      const spc = $("m-speedup-card");
      if (spc) {
        spc.title = card.t_single_primary_ms != null
          ? `加速比 · 单卡 t_single = ${fmtTime(card.t_single_primary_ms)}（${rawMs(card.t_single_primary_ms)}）· 芯片数 = ${card.chips}`
          : "加速比 = t_single / t_multi";
      }
    }
    $("m-peak").textContent = fmt(card.peak_tops, 2) + " T";
    $("m-bw").textContent = fmt(card.mem_eff_GBps, 1) + " GB/s";
    $("m-util").textContent = fmt(card.util, 4);
    $("m-chips").textContent = fmtInt(card.chips);

    [["b-compute", card.t_compute_ms], ["b-dram", card.t_dram_ms], ["b-c2c", card.t_c2c_ms],
      ["b-sync", card.t_sync_ms], ["b-fabric", card.t_fabric_ms], ["b-bubble", card.t_bubble_ms],
      ["b-draft", card.t_draft_ms]].forEach(([id, v]) => {
      $(id).textContent = fmtTime(v);
      $(id).title = rawMs(v);
    });
    renderBreakdown(card);

    $("i-model").textContent = card.model_id || "—";
    $("i-parallel").textContent = `${card.tp} / ${card.pp} / ${card.ep}`;
    if (card.mem_summary) {
      const tg = card.mem_tag || "speculative";
      $("i-mem").innerHTML = `${escapeHtml(card.mem_summary.replace(/ · [^·]+$/, ""))} ${tagBadge(tg)}<br><span class="muted">原始 ${fmt(card.mem_raw_GBps, 1)} · 可用 ${fmt(card.mem_payload_GBps, 1)} · 有效 ${fmt(card.mem_eff_GBps, 1)} GB/s</span>`;
      $("i-mem").title = card.mem_id || "";
    } else {
      $("i-mem").textContent = `${card.mem_kind} · ${fmt(card.mem_eff_GBps, 1)} GB/s（手动几何）`;
      $("i-mem").title = "";
    }
    const iKv = $("i-kv");
    if (iKv) {
      const rep = Number(card.kv_replication) || 1;
      const mode = card.attn_parallel === "dp" ? "注意力 DP（按 batch）" : "TP（按 KV 头）";
      const dp = card.attn_parallel === "dp";
      iKv.innerHTML = `${mode} · 每卡 KV ${fmtBytes(card.bytes_KV)}` +
        (rep > 1.0001
          ? (dp
            ? ` <span class="ptag t-vendor_announced" title="注意力 DP：batch ${card.batch} 不能被 tp=${card.tp} 整除 —— 每卡按 ceil(B/tp) 条序列计 KV（总计相当于 ×${fmt(rep, 2)}）">batch 不均衡 ×${fmt(rep, 2)}</span>`
            : ` <span class="ptag t-speculative" title="KV 在 TP 组内被复制 ${fmt(rep, 2)} 份（tp > n_kv 头数，或 MLA latent 不可按 TP 切分）">复制 ×${fmt(rep, 2)}</span>`)
          : "");
    }
    const iSync = $("i-sync");
    if (iSync) {
      iSync.innerHTML = Number(card.n_sync_per_token) > 0
        ? `${fmtInt(card.n_sync_per_token)} 次 × α ${fmt(card.c2c_latency_us, 1)} µs × (1 − ${fmt(card.sync_overlap, 2)}) = ${fmtTime(card.t_sync_ms)} <span class="badge assumed">假设</span>`
        : `单卡 · 无集合通信`;
    }

    $("i-cores").textContent = `${card.n_cores} × ${fmt(card.tops_per_core, 3)}`;
    $("i-pe").textContent = `${card.pe_rows}×${card.pe_cols}×${card.n_engines}`;
    $("i-bytes").textContent = `${fmtBytes(card.bytes_W)} / ${fmtBytes(card.bytes_KV)} / ${fmtBytes(card.bytes_coll)}`;
    $("i-bytes").title = `${card.bytes_W} / ${card.bytes_KV} / ${card.bytes_coll} B`;

    // 完整假设（假设与说明 标签页）—— honesty banner 已在该页顶部，不再重复
    const assumptions = card.assumptions || [];
    const notes = card.notes || [];
    let html = "";
    if (assumptions.length) {
      html += "<strong>假设</strong><ul>";
      for (const a of assumptions) html += `<li title="${escapeHtml(a)}">${escapeHtml(zhText(a))}</li>`;
      html += "</ul>";
    }
    if (notes.length) {
      html += "<strong>备注</strong><ul>";
      for (const n of notes) html += `<li title="${escapeHtml(n)}">${escapeHtml(zhText(n))}</li>`;
      html += "</ul>";
    }
    $("assumptions").innerHTML = html || "<em>无</em>";
    const cnt = $("tab-assump-count");
    if (cnt) cnt.textContent = assumptions.length + notes.length ? String(assumptions.length + notes.length) : "";
    // 概览：前 4 条
    const top = $("assumptions-top");
    if (top) {
      const all = assumptions.concat(notes);
      top.innerHTML = all.length
        ? all.slice(0, 4).map((a) => `<li title="${escapeHtml(a)}">${escapeHtml(zhText(a))}</li>`).join("")
        : '<li class="muted">无</li>';
      const meta = $("assumptions-top-meta");
      if (meta) meta.textContent = all.length > 4 ? `（共 ${all.length} 条，显示前 4 条）` : "";
    }
    renderEcon(card);
    renderKpiDelta();
  }

  // v0.28：可视化时间分解（瓶颈高亮）
  function renderBreakdown(card) {
    const comps = [
      ["compute", "计算", card.t_compute_ms],
      ["dram", "访存", card.t_dram_ms],
      ["c2c", "C2C", card.t_c2c_ms],
      ["sync", "通信同步（暴露）", card.t_sync_ms],
      ["fabric", "Fabric", card.t_fabric_ms],
      ["bubble", "PP 遍历 / 气泡", card.t_bubble_ms],
      ["draft", "投机草稿", card.t_draft_ms],
    ].map(([k, l, v]) => [k, l, Math.max(0, Number(v) || 0)]);
    const max = Math.max(...comps.map((c) => c[2]), 1e-12);
    const total = comps.reduce((a, c) => a + c[2], 0) || 1e-12;
    const w = String(card.wall || "").toLowerCase();
    const hi = new Set();
    if (w === "compute") hi.add("compute");
    else if (w === "dram" || w === "memory") hi.add("dram");
    else if (w === "c2c" || w === "fabric" || w === "bubble" || w === "sync") hi.add(w);
    else if (w === "balanced") { hi.add("compute"); hi.add("dram"); }
    if (!hi.size) {
      const m = comps.reduce((a, c) => (c[2] > a[2] ? c : a), comps[0]);
      hi.add(m[0]);
    }
    for (const [k, , v] of comps) {
      const fill = $("bf-" + k);
      if (fill) {
        fill.style.width = (v > 0 ? Math.max(1, (v / max) * 100) : 0).toFixed(2) + "%";
        fill.classList.toggle("hi", hi.has(k));
      }
      const lab = document.querySelector(`#bd-bars .lab[data-c="${k}"]`);
      if (lab) lab.classList.toggle("hi", hi.has(k));
      const num = $("b-" + k);
      if (num) num.classList.toggle("hi", hi.has(k));
    }
    const stack = $("bd-stack");
    if (stack) {
      stack.innerHTML = comps
        .filter((c) => c[2] > 0)
        .map(([k, l, v]) => `<span class="seg-${k}${hi.has(k) ? " hi" : ""}" style="flex:${(v / total).toFixed(5)}" title="${escapeHtml(l)} ${fmtTime(v)}（${((v / total) * 100).toFixed(1)}%）"></span>`)
        .join("");
    }
    const d = card.domain || domain;
    const hint = $("breakdown-hint");
    if (hint) {
      hint.textContent = d === "video" ? `TTFC ${fmtTime(card.TTFC_ms != null ? card.TTFC_ms : card.TTFT_ms)} = ${fmtInt(card.n_denoise)} 步 × 单步前向`
        : d === "protein" ? `每序列 ${fmtTime(card.time_per_seq_ms != null ? card.time_per_seq_ms : card.TTFT_ms)}`
        : `TPOT ${fmtTime(card.TPOT_ms)}`;
    }
    const lg = $("bd-legend");
    if (lg) lg.textContent = `高亮 = 瓶颈（${wallZh(card.wall)}）；单位自动缩放（µs / ms / s），悬停可见原始 ms。主指标由引擎合成，不一定等于各分量之和。`;
  }

  // v0.28：KPI 上的基线 Δ（LLM 在 TPOT，视频 TTFC，蛋白质 每序列时间）
  function renderKpiDelta() {
    const els = [$("kpi-delta-1"), $("kpi-delta-2")];
    els.forEach((el) => { if (el) { el.textContent = ""; el.className = "delta"; el.title = ""; } });
    if (!lastCard || baselineMs == null) return;
    const d = lastCard.domain || domain;
    if (baselineCard && baselineCard.domain && baselineCard.domain !== d) return;
    const pct = deltaPct(primaryMsFromCard(lastCard), baselineMs);
    if (pct == null) return;
    const el = d === "llm" ? els[1] : els[0];
    if (!el) return;
    el.textContent = "Δ " + fmtDelta(pct) + " vs 基线";
    el.classList.add(pct < -0.5 ? "better" : pct > 0.5 ? "worse" : "same");
    el.title = `基线 ${metricKeyForDomain(d)} = ${rawMs(baselineMs)}`;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }


  function primaryMsFromCard(card) {
    if (!card) return null;
    const d = card.domain || domain || "llm";
    if (d === "video") {
      const v = card.TTFC_ms != null ? card.TTFC_ms : card.TTFT_ms;
      return v == null ? null : Number(v);
    }
    if (d === "protein") {
      const v = card.time_per_seq_ms != null ? card.time_per_seq_ms : card.TTFT_ms;
      return v == null ? null : Number(v);
    }
    return card.TPOT_ms == null ? null : Number(card.TPOT_ms);
  }

  function metricKeyForDomain(d) {
    if (d === "video") return "TTFC_ms";
    if (d === "protein") return "time_per_seq_ms";
    return "TPOT_ms";
  }

  function deltaPct(value, baseline) {
    if (baseline == null || !Number.isFinite(baseline) || baseline === 0) return null;
    if (value == null || !Number.isFinite(value)) return null;
    return ((value - baseline) / baseline) * 100.0;
  }

  function fmtDelta(d) {
    if (d == null || !Number.isFinite(d)) return "—";
    const sign = d > 0 ? "+" : "";
    return sign + d.toFixed(1) + "%";
  }

  function updateBaselineHint() {
    const el = $("baseline-hint");
    if (!el) return;
    if (!baselineCard || baselineMs == null) {
      el.textContent = "未固定基线";
      el.title = "固定基线后，KPI 与扫描表显示 Δ%";
      el.classList.remove("pinned");
      renderKpiDelta();
      return;
    }
    const key = metricKeyForDomain(baselineCard.domain || domain);
    el.textContent = `基线：${baselineCard.model_id || "?"} · ${metricLabel(key)} ${fmtTime(baselineMs)}`;
    el.title =
      `已固定基线：${baselineCard.model_id || "?"} · ${key}=${rawMs(baselineMs)}` +
      ` · TP/PP/EP=${baselineCard.tp}/${baselineCard.pp}/${baselineCard.ep}` +
      ` · 芯片数=${baselineCard.chips}`;
    el.classList.add("pinned");
    renderKpiDelta();
  }

  function pinBaselineFromCard(card) {
    if (!card || card.ok === false) {
      showToast("请先评估，再固定基线");
      return;
    }
    baselineCard = card;
    baselineMs = primaryMsFromCard(card);
    updateBaselineHint();
    showToast(
      `已固定基线 · ${metricLabel(metricKeyForDomain(card.domain || domain))} ${fmtTime(baselineMs)}`,
      "ok"
    );
    if (lastSweepResult) renderSweep(lastSweepResult);
    pushConfigToUrl();
  }

  function csvEscape(v) {
    if (v == null) return "";
    const s = String(v);
    if (/[",\n\r]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
    return s;
  }

  function sweepToCsvClient(result) {
    const rows = result.rows || [];
    const metricKey = result.metric_key || "primary_ms";
    const domainHint = result.domain || domain;
    const base = baselineMs;
    const headers = [
      "axis", "label", "axis_value", "domain", "metric_key", "primary_ms", "delta_pct",
      "TTFT_ms", "TPOT_ms", "TTFC_ms", "time_per_seq_ms", "frames_per_s", "pair_bytes",
      "wall", "util", "peak_tops", "mem_eff_GBps", "tp", "pp", "ep", "chips", "oom",
      "scale_efficiency", "speedup", "t_single_primary_ms", "scale_metric", "model_id",
      "est_power_W", "est_energy_per_token_J", "est_energy_per_frame_J", "est_energy_per_seq_J",
      "est_system_cost_usd", "est_usd_per_Mtok", "raw_id",
    ];
    const lines = [headers.join(",")];
    for (const r of rows) {
      const c = r.card || {};
      const primary = r.primary_ms != null ? Number(r.primary_ms) : primaryMsFromCard(c);
      const dlt = deltaPct(primary, base);
      const vals = [
        result.axis, r.label, r.axis_value, c.domain || domainHint, metricKey, primary,
        dlt == null ? "" : Number(dlt.toFixed(4)),
        c.TTFT_ms, c.TPOT_ms, c.TTFC_ms, c.time_per_seq_ms, c.frames_per_s, c.pair_bytes,
        c.wall, c.util, c.peak_tops, c.mem_eff_GBps, c.tp, c.pp, c.ep, c.chips, c.oom,
        c.scale_efficiency, c.speedup, c.t_single_primary_ms, c.scale_metric,
        c.model_id || r.model_id,
        c.est_power_W, c.est_energy_per_token_J, c.est_energy_per_frame_J, c.est_energy_per_seq_J,
        c.est_system_cost_usd, c.est_usd_per_Mtok, r.raw_id || r.axis_value,
      ];
      lines.push(vals.map(csvEscape).join(","));
    }
    return lines.join("\n") + "\n";
  }

  function downloadBlob(filename, text, mime) {
    const blob = new Blob([text], { type: mime || "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1500);
  }

  function exportSweepCsv() {
    if (!lastSweepResult || !(lastSweepResult.rows || []).length) {
      showToast("请先运行扫描");
      return;
    }
    const axis = lastSweepResult.axis || "sweep";
    downloadBlob(
      `accel_dse_sweep_${axis}.csv`,
      sweepToCsvClient(lastSweepResult),
      "text/csv;charset=utf-8"
    );
    showToast("CSV 已下载", "ok");
  }

  function exportSweepJson() {
    if (!lastSweepResult || !(lastSweepResult.rows || []).length) {
      showToast("请先运行扫描");
      return;
    }
    const payload = Object.assign({}, lastSweepResult);
    if (baselineMs != null) {
      payload.baseline_ms = baselineMs;
      payload.baseline_card = baselineCard;
    }
    const axis = lastSweepResult.axis || "sweep";
    downloadBlob(
      `accel_dse_sweep_${axis}.json`,
      JSON.stringify(payload, null, 2),
      "application/json;charset=utf-8"
    );
    showToast("JSON 已下载", "ok");
  }

  function setExportEnabled(on) {
    ["btn-export-csv", "btn-export-json"].forEach((id) => {
      const el = $(id);
      if (el) el.disabled = !on;
    });
  }

  // --- Config deep-link (?c= base64url JSON) ---
  function b64urlEncode(str) {
    const bytes = new TextEncoder().encode(str);
    let bin = "";
    bytes.forEach((b) => { bin += String.fromCharCode(b); });
    return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  }

  function b64urlDecode(s) {
    let b64 = String(s).replace(/-/g, "+").replace(/_/g, "/");
    while (b64.length % 4) b64 += "=";
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new TextDecoder().decode(bytes);
  }

  function collectUiState() {
    const st = {
      domain: domain,
      memKind: memKind,
      product_only: $("product-only").checked,
      model_id: $("series").value,
      chip_count: parseInt($("chip_count").value, 10) || 1,
      tp: parseInt($("tp").value, 10) || 1,
      pp: parseInt($("pp").value, 10) || 1,
      ep: parseInt($("ep").value, 10) || 1,
      lock_parallel: $("lock-parallel").checked,
      mem: memSel ? Object.assign({}, memSel) : null,
      compute_id: $("compute").value,
      manual_mem: $("manual-mem").checked,
      manual_compute: $("manual-compute").checked,
      n_channels: parseInt($("n_channels").value, 10),
      width_bits: parseInt($("width_bits").value, 10),
      data_rate_GTs: parseFloat($("data_rate_GTs").value),
      capacity_GB: parseFloat($("capacity_GB").value),
      efficiency: parseFloat($("efficiency").value),
      c2c_latency_us: parseFloat($("c2c_latency_us").value),
      sync_overlap: parseFloat($("sync_overlap").value),
      attn_parallel: attnParallel,
      spec_k: parseInt($("spec_k").value, 10) || 0,
      spec_accept: parseFloat($("spec_accept").value),
      spec_draft: specDraft,
      spec_draft_frac: parseFloat($("spec_draft_frac").value) || 0.1,
      moe_shard: moeShard,
      decode_mb: parseInt($("decode_mb").value, 10) || 0,
      n_cores: parseInt($("n_cores").value, 10),
      tops_per_core: parseFloat($("tops_per_core").value),
      batch: parseInt($("batch").value, 10) || 1,
      prompt: parseInt($("prompt").value, 10) || 512,
      ctx: parseInt($("ctx").value, 10) || 512,
      n_denoise: parseInt($("n_denoise").value, 10) || 50,
      n_frames: parseInt($("n_frames").value, 10) || 16,
      seq_len: parseInt($("seq_len").value, 10) || 512,
      dtype: $("dtype").value || "",
      non_gemm_overhead: parseFloat($("non_gemm_overhead") && $("non_gemm_overhead").value) || 0,
      use_finer_non_gemm: !!($("use_finer_non_gemm") && $("use_finer_non_gemm").checked),
      softmax_frac: parseFloat($("softmax_frac") && $("softmax_frac").value) || 0,
      rope_frac: parseFloat($("rope_frac") && $("rope_frac").value) || 0,
      norm_frac: parseFloat($("norm_frac") && $("norm_frac").value) || 0,
      dtype_mac_factors: ($("dtype_mac_factors") && $("dtype_mac_factors").value) || "",

      quant: $("quant").value || "",
      mac_efficiency: parseFloat($("mac_eff").value) || 0,
      freq_ghz: parseFloat($("freq_ghz").value) || 1.0,
      sweep_axis: sweepAxis,
      sweep_parallel_chips: sweepParallelChips,
      tab: activeTab,
      slo_ttft_ms: parseFloat($("slo-ttft") && $("slo-ttft").value) || undefined,
      slo_tpot_ms: parseFloat($("slo-tpot") && $("slo-tpot").value) || undefined,
      slo_latency_ms: parseFloat($("slo-lat") && $("slo-lat").value) || undefined,
      pareto_layouts: ($("pareto-layouts") && $("pareto-layouts").value) || undefined,
      goodput_mode: ($("goodput-mode") && $("goodput-mode").value) || undefined,
      prefill_mode: ($("prefill-mode") && $("prefill-mode").value) || undefined,
      out_len: parseInt($("out-len") && $("out-len").value, 10) || undefined,
    };
    const econ = collectEcon();
    if (Object.keys(econ).length) st.econ = econ;
    if (baselineCard && baselineMs != null) {
      st.baseline_ms = baselineMs;
      st.baseline_model_id = baselineCard.model_id;
      st.baseline_domain = baselineCard.domain;
      st.baseline_primary = baselineMs;
    }
    if (dualCardA) st.dual_a = dualCardA;
    if (dualCardB) st.dual_b = dualCardB;
    return st;
  }

  function pushConfigToUrl() {
    if (restoringUrl) return;
    clearTimeout(urlPushTimer);
    urlPushTimer = setTimeout(() => {
      try {
        const enc = b64urlEncode(JSON.stringify(collectUiState()));
        const url = new URL(window.location.href);
        url.searchParams.set("c", enc);
        history.replaceState(null, "", url.pathname + "?" + url.searchParams.toString() + url.hash);
      } catch (e) {
        /* ignore URL length / btoa issues */
      }
    }, 200);
  }

  function applyUiState(st) {
    if (!st || typeof st !== "object") return;
    restoringUrl = true;
    try {
      if (st.domain) {
        domain = st.domain;
        document.querySelectorAll("#domain-seg button").forEach((b) => {
          b.classList.toggle("active", b.dataset.domain === domain);
        });
      }
      if (st.memKind) memKind = st.memKind;
      if (st.product_only != null) $("product-only").checked = !!st.product_only;
      populateSeries();
      populateCompute();
      if (st.model_id) {
        const sel = $("series");
        if ([...sel.options].some((o) => o.value === st.model_id)) sel.value = st.model_id;
      }
      updateSeriesMeta();
      updateDomainNote();
      if (st.chip_count != null) $("chip_count").value = st.chip_count;
      if (st.lock_parallel != null) $("lock-parallel").checked = !!st.lock_parallel;
      if (st.tp != null) $("tp").value = st.tp;
      if (st.pp != null) $("pp").value = st.pp;
      if (st.ep != null) $("ep").value = st.ep;
      // v0.29：结构化存储；≤0.28 链接只有 package_id → restoreConfigFromUrl 异步解析到最近的新配置
      if (st.mem && typeof st.mem === "object") memSel = normalizeMemSel(st.mem);
      else if (!st.package_id && st.memKind && !st.manual_mem) memSel = normalizeMemSel(memDefaults(st.memKind === "LPDDR" ? "LPDDR5X" : DEFAULT_MEM_TYPE));
      if (st.compute_id) {
        const sel = $("compute");
        if ([...sel.options].some((o) => o.value === st.compute_id)) sel.value = st.compute_id;
        updateComputeMeta();
      }
      if (st.manual_mem != null) {
        $("manual-mem").checked = !!st.manual_mem;
        $("manual-mem-fields").style.display = st.manual_mem ? "block" : "none";
      }
      if (st.manual_compute != null) {
        $("manual-compute").checked = !!st.manual_compute;
        $("manual-compute-fields").style.display = st.manual_compute ? "block" : "none";
        $("compute").disabled = !!st.manual_compute;
      }
      if (st.n_channels != null) $("n_channels").value = st.n_channels;
      if (st.width_bits != null) $("width_bits").value = st.width_bits;
      if (st.data_rate_GTs != null) $("data_rate_GTs").value = st.data_rate_GTs;
      if (st.capacity_GB != null && Number.isFinite(Number(st.capacity_GB))) $("capacity_GB").value = st.capacity_GB;
      setSyncUi(st.c2c_latency_us, st.sync_overlap, st.attn_parallel);
      setSpecUi(st);
      if (st.efficiency != null) {
        $("efficiency").value = st.efficiency;
        $("efficiency-val").textContent = Number(st.efficiency).toFixed(2);
      }
      if (st.n_cores != null) $("n_cores").value = st.n_cores;
      if (st.tops_per_core != null) $("tops_per_core").value = st.tops_per_core;
      if (st.batch != null) $("batch").value = st.batch;
      if (st.prompt != null) $("prompt").value = st.prompt;
      if (st.ctx != null) $("ctx").value = st.ctx;
      if (st.n_denoise != null) $("n_denoise").value = st.n_denoise;
      if (st.n_frames != null) $("n_frames").value = st.n_frames;
      if (st.seq_len != null) $("seq_len").value = st.seq_len;
      if (st.dtype != null) $("dtype").value = st.dtype;
      if (st.non_gemm_overhead != null && $("non_gemm_overhead")) {
        $("non_gemm_overhead").value = st.non_gemm_overhead;
        if ($("non_gemm_overhead-val")) $("non_gemm_overhead-val").textContent = Number(st.non_gemm_overhead).toFixed(2);
      }
      if (st.use_finer_non_gemm != null && $("use_finer_non_gemm")) $("use_finer_non_gemm").checked = !!st.use_finer_non_gemm;
      ["softmax_frac","rope_frac","norm_frac"].forEach((id) => {
        if (st[id] != null && $(id)) {
          $(id).value = st[id];
          if ($(id+"-val")) $(id+"-val").textContent = Number(st[id]).toFixed(2);
        }
      });
      if (st.dtype_mac_factors != null && $("dtype_mac_factors")) $("dtype_mac_factors").value = st.dtype_mac_factors;
      updateAssumedComputeBadge();

      if (st.quant != null) $("quant").value = st.quant;
      if (st.mac_efficiency != null) {
        $("mac_eff").value = st.mac_efficiency;
        $("mac_eff-val").textContent = Number(st.mac_efficiency).toFixed(2);
      }
      if (st.freq_ghz != null && $("freq_ghz")) {
        $("freq_ghz").value = st.freq_ghz;
        $("freq_ghz-val").textContent = Number(st.freq_ghz).toFixed(2);
      }
      if (st.econ) setEconUi(st.econ);
      if (st.dual_a) dualCardA = st.dual_a;
      if (st.dual_b) dualCardB = st.dual_b;
      if (st.sweep_axis) {
        sweepAxis = st.sweep_axis;
        document.querySelectorAll("#sweep-axis-seg button").forEach((b) => {
          b.classList.toggle("active", b.dataset.axis === sweepAxis);
        });
        updateSweepOpts();
      }
      if (st.sweep_parallel_chips) {
        sweepParallelChips = st.sweep_parallel_chips;
        document.querySelectorAll("#sweep-par-chips-seg button").forEach((b) => {
          b.classList.toggle("active", String(b.dataset.chips) === String(sweepParallelChips));
        });
      }
      if (st.baseline_ms != null || st.baseline_primary != null) {
        baselineMs = Number(st.baseline_ms != null ? st.baseline_ms : st.baseline_primary);
        baselineCard = {
          model_id: st.baseline_model_id || st.model_id,
          domain: st.baseline_domain || st.domain || domain,
          tp: st.tp,
          pp: st.pp,
          ep: st.ep,
          chips: st.chip_count,
          ok: true,
        };
        updateBaselineHint();
      }
      if (st.slo_ttft_ms > 0) $("slo-ttft").value = st.slo_ttft_ms;
      if (st.slo_tpot_ms > 0) $("slo-tpot").value = st.slo_tpot_ms;
      if (st.slo_latency_ms > 0) { $("slo-lat").value = st.slo_latency_ms; sloDomain = st.domain || domain; }
      if (st.pareto_layouts === "current" || st.pareto_layouts === "all") $("pareto-layouts").value = st.pareto_layouts;
      if (st.goodput_mode === "amortized" || st.goodput_mode === "upper") $("goodput-mode").value = st.goodput_mode;
      if (st.prefill_mode === "chunked" || st.prefill_mode === "exclusive") $("prefill-mode").value = st.prefill_mode;
      if (st.out_len > 0) $("out-len").value = st.out_len;
      setSpecUi(st);
      updateAmortUi();
      if (st.tab && TABS.includes(st.tab)) setTab(st.tab, false);
      updateParallelHint();
      updateOverrideBadge();
      wireAssumedCompute();
      updateAssumedComputeBadge();
      renderMemUi();
      renderDualCards();
      updateSceneSummary();
      fitCoreSelects();
    } finally {
      restoringUrl = false;
    }
  }

  // v0.29：async —— 旧链接（≤0.28，仅含 package_id）经 GET /api/memory?resolve= 映射到最近的结构化配置
  async function restoreConfigFromUrl() {
    let st;
    try {
      const url = new URL(window.location.href);
      const c = url.searchParams.get("c");
      if (!c) return false;
      st = JSON.parse(b64urlDecode(c));
      applyUiState(st);
    } catch (e) {
      showToast("?c= 深链无效 —— 使用默认配置");
      return false;
    }
    if (st && !st.mem && st.package_id && !st.manual_mem) {
      await resolveLegacyPackage(st.package_id);
      renderMemUi();
      updateSceneSummary();
    }
    return true;
  }

  function setSyncUi(alpha, overlap, attn) {
    if (alpha != null && Number.isFinite(Number(alpha))) {
      $("c2c_latency_us").value = alpha;
      $("c2c_latency_us-val").textContent = Number(alpha).toFixed(1);
    }
    if (overlap != null && Number.isFinite(Number(overlap))) {
      $("sync_overlap").value = overlap;
      $("sync_overlap-val").textContent = Number(overlap).toFixed(2);
    }
    if (attn === "tp" || attn === "dp") {
      attnParallel = attn;
      document.querySelectorAll("#attn-parallel-seg button").forEach((b) => {
        b.classList.toggle("active", b.dataset.attn === attnParallel);
      });
    }
  }



  async function doEval() {
    const body = collectBody();
    if (!body.model_id) {
      showError("请选择模型系列");
      return;
    }
    {
      const tp = body.tp != null ? body.tp : body.chip_count;
      const pp = body.pp != null ? body.pp : 1;
      const ep = body.ep != null ? body.ep : 1;
      if (tp * pp * ep !== body.chip_count) {
        showError(`并行 ${tp}×${pp}×${ep} ≠ 芯片数 ${body.chip_count} —— 请在 ⚙ 高级参数 › 并行 中修正（或取消锁定）`);
        setStatus("并行配置无效", "error");
        $("metrics-grid").classList.add("stale");
        return;
      }
    }

    const seq = ++evalSeq;
    setStatus("评估中…", "loading");
    showError("");
    try {
      const card = await apiPost("/api/eval", body);
      if (seq !== evalSeq) return; // stale
      if (!card.ok && card.error) throw new Error(card.error);
      renderCard(card);
      $("metrics-grid").classList.remove("stale");
      updateSloInputs();
      scheduleGoodputKpi();
      markParetoStale();
      maybeAutoPareto();
      const now = new Date();
      const hh = (n) => String(n).padStart(2, "0");
      setStatus(`已自动更新 · ${hh(now.getHours())}:${hh(now.getMinutes())}:${hh(now.getSeconds())}`, "ok");
      pushConfigToUrl();
    } catch (err) {
      if (seq !== evalSeq) return;
      const msg = String(err.message || err);
      const extra = /(tp|pp|ep) must be one of/.test(msg)
        ? " —— TP / PP / EP 仅支持 1 / 2 / 4 / 8；未锁定时 TP = 芯片数，请改用 1 / 2 / 4 / 8 片，或在 ⚙ 高级参数 › 并行 中锁定 TP×PP×EP"
        : "";
      showError("错误：" + msg + extra);
      setStatus("出错", "error");
      $("metrics-grid").classList.add("stale");
    }
  }

  function scheduleEval() {
    if (lastSweepResult && !sweepStale && !restoringUrl) {
      sweepStale = true;
      renderSweep(lastSweepResult);
    }
    updateAdvancedStatus();
    updateSceneSummary();
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(doEval, DEBOUNCE_MS);
    pushConfigToUrl();
  }


  function seedDomainKnobsFromSeries() {
    const id = $("series").value;
    const s = seriesCache.find((x) => x.id === id);
    if (!s || !s.defaults) return;
    if (domain === "video") {
      if (s.defaults.n_denoise != null) $("n_denoise").value = s.defaults.n_denoise;
      if (s.defaults.n_frames != null) $("n_frames").value = s.defaults.n_frames;
    } else if (domain === "protein") {
      if (s.defaults.seq_len != null) $("seq_len").value = s.defaults.seq_len;
    }
  }


  function updateSweepOpts() {
    const pkg = $("opt-package-kind");
    const comp = $("opt-compute-level");
    const par = $("opt-parallel-chips");
    if (pkg) pkg.style.display = sweepAxis === "package" ? "" : "none";
    if (comp) comp.style.display = sweepAxis === "compute" ? "" : "none";
    if (par) par.style.display = sweepAxis === "parallel" ? "" : "none";
    const hint = $("sweep-hint");
    if (hint) {
      const tips = {
        chips: "芯片数取 {1, 2, 4, 8}；TP 默认等于芯片数。",
        package: "围绕当前存储配置只改一个维度：数量 / 速率档 / 类型（7 种默认配置）；或扫描目录精选。每行带来源标签。",
        compute: "扫描单核（4–16T）和/或集群（64–256T）算力预设。",
        parallel: "芯片数为 4 或 8 时的 TP×PP×EP 组合矩阵（dense 模型 EP=1）。",
        series: "对比当前领域内的模型系列（上限 32 行）。",
      };
      hint.textContent =
        (tips[sweepAxis] || "") +
        "以当前参数为固定基准，仅改变所选维度。";
    }
  }

  function showSweepError(msg) {
    const el = $("sweep-error");
    if (!el) return;
    el.textContent = msg || "";
    el.classList.toggle("show", !!msg);
  }

  function collectSweepBody() {
    const body = collectBody();
    body.axis = sweepAxis;
    body.max_rows = 32;
    if (sweepAxis === "package") {
      const k = $("sweep-pkg-kind") && $("sweep-pkg-kind").value;
      if (k && k.startsWith("axis:")) body.package_axis = k.slice(5);
      else if (k) body.package_kind = k;
    }
    if (sweepAxis === "compute") {
      body.compute_level =
        ($("sweep-compute-level") && $("sweep-compute-level").value) || "all";
    }
    if (sweepAxis === "parallel") {
      body.parallel_chips = sweepParallelChips;
      body.chip_count = sweepParallelChips;
      body.tp = undefined;
      body.pp = undefined;
      body.ep = undefined;
      delete body.tp;
      delete body.pp;
      delete body.ep;
    }
    if (sweepAxis === "series") {
      body.domain = domain;
      body.product_only = $("product-only").checked;
    }
    return body;
  }

  function metricLabel(key) {
    if (key === "TTFC_ms") return "TTFC";
    if (key === "time_per_seq_ms") return "每序列时间";
    if (key === "TPOT_ms") return "TPOT";
    if (key === "TTFT_ms") return "TTFT";
    return key;
  }

  function renderSweep(result) {
    const rows = result.rows || [];
    const wrapT = $("compare-table-wrap");
    const wrapC = $("compare-chart-wrap");
    const meta = $("sweep-meta");
    if (meta) {
      meta.textContent =
        `${AXIS_ZH[result.axis] || result.axis} · ${result.count} 行` +
        (result.capped ? `（由 ${result.requested} 行截断）` : "") +
        ` · ${metricLabel(result.metric_key)}` +
        (baselineMs != null ? ` · Δ% 相对基线 ${fmtTime(baselineMs)}` : "") +
        (sweepStale ? " · ⚠ 参数已变更，点击「运行扫描」刷新" : "");
    }
    if ($("sweep-empty")) $("sweep-empty").hidden = rows.length > 0;
    if (!rows.length) {
      if (wrapT) wrapT.hidden = true;
      if (wrapC) wrapC.hidden = true;
      return;
    }

    // Find best (lowest primary_ms)
    let bestIdx = 0;
    let bestVal = Infinity;
    rows.forEach((r, i) => {
      const v = Number(r.primary_ms);
      if (Number.isFinite(v) && v < bestVal) {
        bestVal = v;
        bestIdx = i;
      }
    });
    const maxVal = Math.max(
      ...rows.map((r) => Number(r.primary_ms) || 0),
      1e-9
    );

    // Bar chart (CSS columns — no CDN / no SVG dependency beyond CSS)
    const chart = $("bar-chart");
    const title = $("chart-title");
    if (title) title.textContent = metricLabel(result.metric_key) + "（越低越好）";
    if (chart) {
      chart.innerHTML = "";
      rows.forEach((r, i) => {
        const v = Number(r.primary_ms) || 0;
        const h = Math.max(2, Math.round((v / maxVal) * 100));
        const col = document.createElement("div");
        col.className = "bar-col";
        col.title = `${r.label}${r.raw_id ? `（${r.raw_id}）` : ""}：${fmtTime(v)}（${rawMs(v)}）`;
        col.innerHTML =
          `<div class="bar-val">${escapeHtml(fmtTime(v, 3))}</div>` +
          `<div class="bar${i === bestIdx ? " best" : ""}" style="height:${h}px"></div>` +
          `<div class="bar-lab">${escapeHtml(String(r.label))}</div>`;
        chart.appendChild(col);
      });
    }
    if (wrapC) wrapC.hidden = false;

    // Table
    const thead = $("compare-thead");
    const tbody = $("compare-tbody");
    const d = result.domain || domain;
    let cols;
    if (d === "video") {
      cols = [
        ["config", "label"],
        ["TTFC_ms", "num"],
        ["frames/s", "num"],
        ["Δ%", "delta"],
        ["wall", "wall"],
        ["util", "num"],
        ["peak_T", "num"],
        ["eff_GBps", "num"],
        ["tp/pp/ep", "txt"],
        ["scale_eff", "num"],
        ["oom", "txt"],
      ];
    } else if (d === "protein") {
      cols = [
        ["config", "label"],
        ["t/seq_ms", "num"],
        ["pair_B", "num"],
        ["Δ%", "delta"],
        ["wall", "wall"],
        ["util", "num"],
        ["peak_T", "num"],
        ["eff_GBps", "num"],
        ["tp/pp/ep", "txt"],
        ["scale_eff", "num"],
        ["oom", "txt"],
      ];
    } else {
      cols = [
        ["config", "label"],
        ["TTFT_ms", "num"],
        ["TPOT_ms", "num"],
        ["Δ%", "delta"],
        ["wall", "wall"],
        ["util", "num"],
        ["peak_T", "num"],
        ["eff_GBps", "num"],
        ["tp/pp/ep", "txt"],
        ["scale_eff", "num"],
        ["oom", "txt"],
      ];
    }
    // 表头显示文本（内部列 key 不变）
    const COL_ZH = {
      "config": "配置", "TTFT_ms": "TTFT", "TPOT_ms": "TPOT", "TTFC_ms": "TTFC",
      "frames/s": "帧/秒", "t/seq_ms": "每序列时间", "pair_B": "pair 字节", "Δ%": "Δ%",
      "wall": "瓶颈", "util": "利用率", "peak_T": "峰值 T", "eff_GBps": "有效带宽 GB/s",
      "tp/pp/ep": "TP/PP/EP", "scale_eff": "扩展效率", "oom": "OOM",
      "est_J/unit": "估算 J/单位", "est_$/MTok": "估算 $/MTok",
    };
    const econOn = rows.some((r) => r.card && r.card.econ_configured);
    if (econOn) {
      cols.push(["est_J/unit", "num"]);
      if (d === "llm") cols.push(["est_$/MTok", "num"]);
    }
    if (thead) {
      thead.innerHTML =
        "<tr>" + cols.map((c) =>
          `<th class="${c[1] === "label" || c[1] === "txt" || c[1] === "wall" ? "" : "num"}" title="${escapeHtml(c[0])}">${escapeHtml(COL_ZH[c[0]] || c[0])}</th>`
        ).join("") + "</tr>";
    }
    if (tbody) {
      tbody.innerHTML = "";
      rows.forEach((r, i) => {
        const c = r.card || {};
        const tr = document.createElement("tr");
        if (i === bestIdx) tr.className = "best-row";
        const wall = String(c.wall || "—");
        const cells = [];
        for (const [name, kind] of cols) {
          let td = document.createElement("td");
          if (kind === "num") td.className = "num";
          if (name === "config") {
            td.className = "label";
            td.textContent = r.label;
            td.title = r.raw_id || r.axis_value || r.label;
          } else if (name === "TTFT_ms") {
            td.textContent = fmtTime(c.TTFT_ms);
            td.title = rawMs(c.TTFT_ms);
          } else if (name === "TPOT_ms") {
            td.textContent = fmtTime(c.TPOT_ms);
            td.title = rawMs(c.TPOT_ms);
          } else if (name === "TTFC_ms") {
            td.textContent = fmtTime(c.TTFC_ms != null ? c.TTFC_ms : c.TTFT_ms);
            td.title = rawMs(c.TTFC_ms != null ? c.TTFC_ms : c.TTFT_ms);
          } else if (name === "frames/s") {
            td.textContent = fmt(c.frames_per_s, 4);
          } else if (name === "t/seq_ms") {
            td.textContent = fmtTime(c.time_per_seq_ms != null ? c.time_per_seq_ms : c.TTFT_ms);
            td.title = rawMs(c.time_per_seq_ms != null ? c.time_per_seq_ms : c.TTFT_ms);
          } else if (name === "pair_B") {
            td.textContent = fmtBytes(c.pair_bytes);
            td.title = `${fmtInt(c.pair_bytes)} B`;
          } else if (name === "wall") {
            td.innerHTML = `<span class="wall-pill ${escapeHtml(wall.toLowerCase())}" title="${escapeHtml(wall)}">${escapeHtml(wall === "—" ? wall : wallZh(wall))}</span>`;
          } else if (name === "util") {
            td.textContent = fmt(c.util, 4);
          } else if (name === "peak_T") {
            td.textContent = fmt(c.peak_tops, 2);
          } else if (name === "eff_GBps") {
            td.textContent = fmt(c.mem_eff_GBps, 1);
          } else if (name === "tp/pp/ep") {
            td.textContent = `${c.tp}/${c.pp}/${c.ep}`;
          } else if (name === "Δ%" || kind === "delta") {
            const primary = r.primary_ms != null ? Number(r.primary_ms) : primaryMsFromCard(c);
            const dlt = deltaPct(primary, baselineMs);
            td.className = "num delta";
            td.textContent = fmtDelta(dlt);
            if (dlt != null) {
              if (dlt < -0.5) td.classList.add("better");
              else if (dlt > 0.5) td.classList.add("worse");
            }
            if (baselineMs == null) td.title = "固定基线后显示 Δ%";
            else td.title = `相对基线 ${fmtTime(baselineMs)}（${rawMs(baselineMs)}）`;
          } else if (name === "est_J/unit") {
            const ev = d === "video" ? c.est_energy_per_frame_J
              : d === "protein" ? c.est_energy_per_seq_J : c.est_energy_per_token_J;
            td.textContent = Number(ev) > 0 ? Number(ev).toPrecision(4) : "—";
            td.title = "假设能耗估算（用户参数，非硅片实测）";
          } else if (name === "est_$/MTok") {
            td.textContent = Number(c.est_usd_per_Mtok) > 0 ? Number(c.est_usd_per_Mtok).toPrecision(4) : "—";
            td.title = "假设 $/MTok 估算";
          } else if (name === "scale_eff") {
            td.textContent = c.scale_efficiency != null ? fmt(c.scale_efficiency, 4) : "—";
            td.title = c.scale_metric
              ? `${c.scale_metric} · 加速比 ${c.speedup != null ? Number(c.speedup).toFixed(3) : "—"}×`
              : "扩展效率";
            if (c.scale_efficiency != null && Number(c.scale_efficiency) < 0.85) td.style.color = "var(--warn)";
          } else if (name === "oom") {
            td.textContent = yesNo(c.oom);
            if (c.oom) {
              td.style.color = "var(--danger)";
              td.style.fontWeight = "800";
              td.textContent = "OOM";
            }
          } else {
            td.textContent = "—";
          }
          cells.push(td);
        }
        cells.forEach((td) => tr.appendChild(td));
        tbody.appendChild(tr);
      });
    }
    if (wrapT) wrapT.hidden = false;
  }

  function clearSweep() {
    lastSweepResult = null;
    sweepStale = false;
    setExportEnabled(false);
    const wrapT = $("compare-table-wrap");
    const wrapC = $("compare-chart-wrap");
    if (wrapT) wrapT.hidden = true;
    if (wrapC) wrapC.hidden = true;
    const chart = $("bar-chart");
    if (chart) chart.innerHTML = "";
    const tbody = $("compare-tbody");
    if (tbody) tbody.innerHTML = "";
    const thead = $("compare-thead");
    if (thead) thead.innerHTML = "";
    const meta = $("sweep-meta");
    if (meta) meta.textContent = "选择扫描维度 · ≤32 行";
    if ($("sweep-empty")) $("sweep-empty").hidden = false;
    showSweepError("");
  }

  // v0.28：对比/扫描 面板首次可见时自动运行一次（当前维度，≤32 行），避免空白面板
  function maybeAutoSweep() {
    if (sweepAutoRan || lastSweepResult) return;
    const wide = window.matchMedia && window.matchMedia("(min-width: 1680px)").matches;
    const visible = activeTab === "sweep" || (wide && activeTab === "overview");
    if (!visible || !seriesCache.length) return;
    sweepAutoRan = true;
    setTimeout(() => doSweep(null, { auto: true }), 0);
  }

  async function doSweep(override, opts) {
    const auto = !!(opts && opts.auto);
    const body = collectSweepBody();
    if (override) Object.assign(body, override);
    if (!body.model_id && body.axis !== "series") {
      showSweepError("请先选择模型系列");
      return;
    }
    if (!auto) setStatus("扫描中…", "loading");
    showSweepError("");
    try {
      const result = await apiPost("/api/sweep", body);
      if (!result.ok && result.error) throw new Error(result.error);
      lastSweepResult = result;
      sweepStale = false;
      setExportEnabled(true);
      renderSweep(result);
      if (!auto) {
        setStatus(
          `扫描完成 · ${AXIS_ZH[result.axis] || result.axis} × ${result.count}` +
            (result.capped ? "（已截断）" : ""),
          "ok"
        );
        showToast(`扫描 ${AXIS_ZH[result.axis] || result.axis}：${result.count} 行`, "ok");
      }
      pushConfigToUrl();
    } catch (err) {
      showSweepError("扫描错误：" + String(err.message || err));
      if (!auto) {
        setStatus("扫描出错", "error");
        showToast(String(err.message || err));
      }
    }
  }


    function wireSeg(containerId, attr, onPick) {
    const root = $(containerId);
    root.addEventListener("click", (ev) => {
      const btn = ev.target.closest("button");
      if (!btn) return;
      root.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      onPick(btn.getAttribute(attr));
    });
  }

  function resetDefaults() {
    domain = "llm";
    memKind = "HBM";
    document.querySelectorAll("#domain-seg button").forEach((b) => {
      b.classList.toggle("active", b.dataset.domain === "llm");
    });
    memSel = normalizeMemSel(memDefaults(DEFAULT_MEM_TYPE));
    setSyncUi(DEFAULT_C2C_LAT_US, DEFAULT_SYNC_OVERLAP, "tp");
    $("product-only").checked = false;
    $("chip_count").value = 2;
    $("lock-parallel").checked = false;
    $("manual-mem").checked = false;
    $("manual-compute").checked = false;
    $("manual-mem-fields").style.display = "none";
    $("manual-compute-fields").style.display = "none";
    $("compute").disabled = false;
    $("batch").value = 1;
    $("prompt").value = 512;
    $("ctx").value = 512;
    $("dtype").value = "";
    if ($("non_gemm_overhead")) {
      $("non_gemm_overhead").value = 0;
      if ($("non_gemm_overhead-val")) $("non_gemm_overhead-val").textContent = "0.00";
    }
    if ($("use_finer_non_gemm")) $("use_finer_non_gemm").checked = false;
    ["softmax_frac","rope_frac","norm_frac"].forEach((id) => {
      if ($(id)) { $(id).value = 0; if ($(id+"-val")) $(id+"-val").textContent = "0.00"; }
    });
    if ($("dtype_mac_factors")) $("dtype_mac_factors").value = "";
    updateAssumedComputeBadge();

    $("quant").value = "";
    $("efficiency").value = DEFAULT_MEM_EFF;
    $("efficiency-val").textContent = DEFAULT_MEM_EFF.toFixed(2);
    $("mac_eff").value = DEFAULT_MAC_EFF;
    $("mac_eff-val").textContent = DEFAULT_MAC_EFF.toFixed(2);
    if ($("freq_ghz")) {
      $("freq_ghz").value = DEFAULT_FREQ_GHZ;
      $("freq_ghz-val").textContent = DEFAULT_FREQ_GHZ.toFixed(2);
    }
    setEconUi(null);
    if ($("preset-select")) $("preset-select").value = "";
    if ($("preset-hint")) $("preset-hint").textContent = "预设仅组合目录 id（假设的 DSE 取值范围）—— 不含功耗 / 价格。";
    updateOverrideBadge();
    populateSeries();
    renderMemUi();
    populateCompute();
    syncTpFromChips();
    scheduleEval();
  }

  function renderBanner(en) {
    const el = $("banner-full");
    if (!el) return;
    if (!en) { el.textContent = ""; return; }
    const zh = bannerZh(en);
    el.innerHTML =
      `<p class="banner-zh">${escapeHtml(zh)}</p>` +
      (zh !== en
        ? `<details class="banner-en"><summary>英文原文</summary>${escapeHtml(en)}</details>`
        : "");
  }

  // ---------- v0.30 吞吐 / 交互：帕累托 + SLO goodput（手写 SVG，零依赖） ----------
  let paretoResult = null, paretoSeq = 0, paretoSig = "", paretoTimer = null;
  let goodputSeq = 0, goodputTimer = null, paretoHoverId = -1, sloDomain = "";
  const PARETO_COLORS = ["#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa", "#f87171", "#22d3ee", "#fb923c",
    "#a3e635", "#e879f9", "#94a3b8", "#2dd4bf", "#facc15", "#c084fc", "#4ade80", "#f97316",
    "#93c5fd", "#f9a8d4", "#6ee7b7", "#fcd34d", "#c4b5fd", "#fca5a5", "#67e8f9", "#fdba74",
    "#bef264", "#f0abfc", "#cbd5e1", "#5eead4"];
  const SLO_DEFAULTS = { llm: { ttft: 2000, tpot: 50 }, video: { lat: 30000 }, protein: { lat: 10000 } };
  const BIND_SHORT = { TTFT: "TTFT", TPOT: "TPOT", capacity: "容量", latency: "时延", batch_limit: "批量上限", "TTFT+TPOT": "TTFT+TPOT", none: "—" };
  const PARETO_Y_UNIT = { llm: "tok/s/芯片", video: "帧/s/芯片", protein: "序列/s/芯片" };

  function num4(v) {
    const x = Number(v);
    if (!Number.isFinite(x)) return "—";
    if (Math.abs(x) >= 1000) return String(Math.round(x));
    return trimNum(x.toPrecision(4));
  }
  function updateSloInputs() {
    const d = domain || "llm";
    document.querySelectorAll("#tab-pareto .slo-llm").forEach((e) => { e.hidden = d !== "llm"; });
    document.querySelectorAll("#tab-pareto .slo-aux").forEach((e) => { e.hidden = d === "llm"; });
    const lab = $("slo-lat-label");
    if (lab) lab.textContent = d === "video" ? "TTFC ≤（ms）" : "单批时间 ≤（ms）";
    if (d !== "llm" && sloDomain !== d && $("slo-lat")) $("slo-lat").value = SLO_DEFAULTS[d].lat;
    sloDomain = d;
    const note = $("pareto-domain-note");
    if (note) {
      note.textContent = d === "video"
        ? "视频：等效曲线 —— 帧/s/芯片 vs TTFC（批量 B 个请求成批生成）。视频多为算力受限，增大批量几乎不提升每芯片吞吐，曲线主要区分并行布局。"
        : d === "protein"
          ? "蛋白：等效曲线 —— 序列/s/芯片 vs 单批时间（B 条序列成批）。多为算力受限，曲线主要区分并行布局。"
          : "";
    }
    const k = $("m-goodput-k");
    if (k) k.textContent = "SLO 吞吐 / 芯片";
  }
  function paretoBody(layouts) {
    const body = collectBody();
    const d = domain || "llm";
    if (d === "llm") {
      const t1 = parseFloat($("slo-ttft").value), t2 = parseFloat($("slo-tpot").value);
      if (t1 > 0) body.slo_ttft_ms = t1;
      if (t2 > 0) body.slo_tpot_ms = t2;
      // v0.31 吞吐口径：摊销 prefill（默认）| 上界
      body.goodput_mode = $("goodput-mode").value || "amortized";
      body.prefill_mode = $("prefill-mode").value || "chunked";
      const n = parseInt($("out-len").value, 10);
      if (n > 0) body.out_len = n;
    } else {
      const l = parseFloat($("slo-lat").value);
      if (l > 0) body.slo_latency_ms = l;
    }
    body.layouts = layouts || ($("pareto-layouts").value || "all");
    return body;
  }
  function showParetoError(msg) {
    const el = $("pareto-error");
    if (!el) return;
    el.textContent = msg || "";
    el.classList.toggle("show", !!msg);
  }
  function paretoVisible() {
    return activeTab === "pareto";
  }
  function maybeAutoPareto() {
    if (!paretoVisible() || !seriesCache.length) return;
    const body = paretoBody();
    if (!body.model_id) return;
    if (paretoResult && JSON.stringify(body) === paretoSig) return;
    clearTimeout(paretoTimer);
    paretoTimer = setTimeout(runPareto, 250);
  }
  async function runPareto() {
    const body = paretoBody();
    if (!body.model_id) {
      showParetoError("请先选择模型系列");
      return;
    }
    const seq = ++paretoSeq;
    const sig = JSON.stringify(body);
    const panel = $("pareto-panel");
    panel.classList.add("busy");
    $("pareto-meta").textContent = "计算中…（批量 × 并行布局，约 0.3–5 s）";
    showParetoError("");
    try {
      const r = await apiPost("/api/pareto", body);
      if (seq !== paretoSeq) return;
      paretoResult = r;
      paretoSig = sig;
      panel.classList.remove("stale");
      renderPareto(r);
    } catch (err) {
      if (seq !== paretoSeq) return;
      showParetoError("错误：" + String(err.message || err));
      $("pareto-meta").textContent = "计算失败";
    } finally {
      if (seq === paretoSeq) panel.classList.remove("busy");
    }
  }
  function markParetoStale() {
    if (!paretoResult) return;
    const body = paretoBody();
    if (JSON.stringify(body) !== paretoSig) $("pareto-panel").classList.add("stale");
  }
  function layoutColorMap(r) {
    const m = {};
    let k = 0;  // 只给可放下（有点）的布局分配颜色；放不下的为灰色
    (r.layouts || []).forEach((l) => { m[l.layout] = l.fits ? PARETO_COLORS[k++ % PARETO_COLORS.length] : "#4b5563"; });
    return m;
  }
  function bindingText(g) {
    return BIND_SHORT[g.binding] || g.binding || "—";
  }
  function servingTxt(r) {
    const sv = r.serving || {};
    if (!sv.goodput_mode) return "";
    return sv.goodput_mode === "upper" ? "上界（仅 decode）"
      : `摊销 prefill · ${sv.prefill_mode_zh || ""} · P=${sv.prompt_len} / N=${sv.out_len}`;
  }
  function renderPareto(r) {
    const d = r.domain || "llm";
    const g = r.goodput || {};
    const ax = r.axes || {};
    $("pareto-empty").hidden = true;
    $("btn-pareto-csv").disabled = false;
    $("btn-pareto-json").disabled = false;
    const fits = (r.layouts || []).filter((l) => l.fits).length;
    $("pareto-meta").textContent =
      `${r.model_id} · ${r.chips} 芯片 · ${r.points.length} 点 · 前沿 ${r.frontier.length} · ` +
      `布局 ${fits}/${(r.layouts || []).length} 可放下 · ${r.n_evals} 次评估 · ${Math.round(r.elapsed_ms)} ms`;
    // KPI-ish goodput summary
    const grid = $("pareto-goodput");
    grid.classList.toggle("none", !g.ok);
    const yu = PARETO_Y_UNIT[d];
    $("gp-y-k").textContent = "SLO 最优吞吐";
    const sloTxt = d === "llm"
      ? `TTFT ≤ ${num4(r.slo.ttft_ms)} ms · TPOT ≤ ${num4(r.slo.tpot_ms)} ms`
      : `${d === "video" ? "TTFC" : "单批时间"} ≤ ${num4(r.slo.latency_ms)} ms`;
    const bindBox = $("gp-binding").parentElement;
    bindBox.className = "gp bind-" + String(g.binding || "none").replace("+", "_");
    if (g.ok) {
      $("gp-y").innerHTML = `${num4(g.y)} <small>${escapeHtml(yu)}</small>`;
      $("gp-y-sub").textContent = d === "llm"
        ? `每用户 ${num4(g.tok_s_user)} tok/s · ${servingTxt(r)} · ${sloTxt}`
        : `时延 ${fmtTime(g.latency_ms)} · ${sloTxt}`;
      $("gp-config").textContent = g.config;
      $("gp-config-sub").textContent = d === "llm"
        ? `TTFT ${fmtTime(g.TTFT_ms)} · TPOT ${fmtTime(g.TPOT_ms)}` +
          (g.prefill_share > 0.0005 ? ` · prefill 占 ${Math.round(g.prefill_share * 100)}%` : "")
        : `B=${g.batch}`;
      $("gp-users").textContent = String(g.max_users);
      $("gp-users-sub").textContent = d === "llm" ? "同时 decode 的序列数（整系统）" : "单批请求数";
      $("gp-binding").textContent = bindingText(g);
      $("gp-binding-sub").textContent = g.binding_zh || "";
    } else {
      $("gp-y").textContent = "无";
      $("gp-y-sub").textContent = `没有配置满足 ${sloTxt}`;
      $("gp-config").textContent = "—";
      $("gp-config-sub").textContent = "";
      $("gp-users").textContent = "0";
      $("gp-users-sub").textContent = "";
      $("gp-binding").textContent = bindingText(g);
      $("gp-binding-sub").textContent = g.binding_zh || "";
    }
    drawParetoChart(r);
    // legend
    const cmap = layoutColorMap(r);
    const used = new Set(r.points.map((p) => p.layout));
    $("pareto-legend").innerHTML =
      `<span><span class="sym">●</span>帕累托前沿</span><span><span class="sym" style="opacity:.35">●</span>被支配（变暗）</span>` +
      `<span><span class="sym">○</span>违反 SLO</span><span><span class="sym" style="color:#fde047">◎</span>SLO 最优</span>` +
      `<span><span class="sym">⬚</span>当前场景</span><span class="muted">｜颜色 = 并行布局：</span>` +
      (r.layouts || []).filter((l) => used.has(l.layout)).map((l) =>
        `<span class="lay${g.ok && g.layout === l.layout ? " best" : ""}"><i class="sw" style="background:${cmap[l.layout]}"></i>${escapeHtml(l.layout)}</span>`).join("");
    // layout table
    const tbl = $("pareto-layout-table");
    tbl.tHead.innerHTML = `<tr><th>布局</th><th class="num">容量上限 B</th>${d === "llm" ? '<th class="num">单条 prefill</th>' : ""}<th class="num">SLO 最大并发</th><th class="num">SLO 吞吐（${yu}）</th><th>约束</th></tr>`;
    tbl.tBodies[0].innerHTML = (r.layouts || []).map((l) => {
      const best = g.ok && g.layout === l.layout;
      const bnd = l.fits ? (BIND_SHORT[l.binding] || l.binding || "—") : "容量（权重放不下）";
      const why = !l.fits && l.why_not ? ` title="${escapeHtml(l.why_not)}"` : "";
      return `<tr class="${best ? "best-row" : ""}"><td class="label"><i class="sw" style="display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;background:${cmap[l.layout]}"></i>${escapeHtml(l.layout)}</td>` +
        `<td class="num">${l.fits ? l.max_batch + (l.max_batch >= r.batch_limit ? "+" : "") : "0"}</td>` +
        (d === "llm" ? `<td class="num">${l.fits ? fmtTime(l.TTFT_prefill_ms) : "—"}</td>` : "") +
        `<td class="num">${l.fits ? (l.max_users_slo || 0) : "—"}</td>` +
        `<td class="num">${l.goodput != null ? num4(l.goodput) : "—"}</td>` +
        `<td class="${!l.fits || l.binding === "capacity" ? "bind-capacity" : ""}"${why}>${escapeHtml(bnd)}</td></tr>`;
    }).join("");
    $("pareto-assump").innerHTML = (r.assumptions || []).map((a) => `<li>${escapeHtml(a)}</li>`).join("") +
      `<li class="muted">容量上限 B 搜索至 ${r.batch_limit}（“+” = 达到搜索上限）；前沿只含容量可放下的点。</li>`;
  }

  function logTicks(lo, hi) {
    const e0 = Math.floor(Math.log10(lo)), e1 = Math.ceil(Math.log10(hi));
    const span = Math.log10(hi) - Math.log10(lo);
    const sets = span <= 0.6 ? [[1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8]] : span <= 2.2 ? [[1, 2, 5], [1, 1.5, 2, 3, 5, 7]] : span <= 4.5 ? [[1, 3]] : [[1]];
    let out = [];
    for (const mults of sets) {
      out = [];
      for (let e = e0; e <= e1; e++) {
        for (const m of mults) {
          const v = m * Math.pow(10, e);
          if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v);
        }
      }
      if (out.length >= 3) break;
    }
    return out;
  }
  function msTick(v) {
    if (v >= 3600e3) return trimNum((v / 3600e3).toPrecision(2)) + " h";
    if (v >= 60e3) return trimNum((v / 60e3).toPrecision(2)) + " min";
    if (v >= 1e3) return trimNum((v / 1e3).toPrecision(2)) + " s";
    return trimNum(Number(v.toPrecision(2)).toString()) + " ms";
  }
  function tickLabel(v) {
    if (v >= 1e6) return trimNum((v / 1e6).toPrecision(3)) + "M";
    if (v >= 1e4) return trimNum((v / 1e3).toPrecision(3)) + "k";
    return trimNum(Number(v.toPrecision(3)).toString());
  }
  function drawParetoChart(r) {
    const svg = $("pareto-svg");
    const narrow = (svg.clientWidth || 760) < 560;
    const W = narrow ? 420 : 760, H = narrow ? 420 : 400;
    const m = narrow ? { l: 50, r: 12, t: 16, b: 50 } : { l: 64, r: 18, t: 16, b: 50 };
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const d = r.domain || "llm";
    const ax = r.axes || {};
    const lowerBetter = ax.x_better === "lower";
    const pts = r.points.filter((p) => !p.oom && p.x > 0 && p.y > 0);
    if (!pts.length) {
      svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" class="atitle">没有可放下的配置（容量不足）</text>`;
      return;
    }
    let x0 = Math.min(...pts.map((p) => p.x)), x1 = Math.max(...pts.map((p) => p.x));
    let y0 = Math.min(...pts.map((p) => p.y)), y1 = Math.max(...pts.map((p) => p.y));
    let sloX = null;
    if (d === "llm" && r.slo.tpot_ms > 0) sloX = 1000 / r.slo.tpot_ms;
    else if (d !== "llm" && r.slo.latency_ms > 0) sloX = r.slo.latency_ms;
    if (sloX != null && sloX > x0 / 4 && sloX < x1 * 4) { x0 = Math.min(x0, sloX); x1 = Math.max(x1, sloX); }
    const padL = (a, b) => { const la = Math.log10(a), lb = Math.log10(b), sp = Math.max(lb - la, 0.3); return [Math.pow(10, la - sp * 0.06), Math.pow(10, lb + sp * 0.06)]; };
    [x0, x1] = padL(x0, x1);
    [y0, y1] = padL(y0, y1);
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const fx = (v) => {
      const t = (Math.log10(v) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0));
      return m.l + (lowerBetter ? 1 - t : t) * pw;
    };
    const fy = (v) => m.t + (1 - (Math.log10(v) - Math.log10(y0)) / (Math.log10(y1) - Math.log10(y0))) * ph;
    const parts = [];
    // SLO infeasible zone (always left side: TPOT too slow / latency too long)
    if (sloX != null && sloX > x0 && sloX < x1) {
      const sx = fx(sloX);
      parts.push(`<rect class="slo-zone" x="${m.l}" y="${m.t}" width="${Math.max(0, sx - m.l)}" height="${ph}"/>`);
      parts.push(`<line class="slo-line" x1="${sx}" x2="${sx}" y1="${m.t}" y2="${m.t + ph}"/>`);
      const lab = d === "llm" ? `TPOT ≤ ${num4(r.slo.tpot_ms)} ms` : `${d === "video" ? "TTFC" : "时间"} ≤ ${num4(r.slo.latency_ms)} ms`;
      parts.push(`<text class="slo-lab" x="${sx - 5}" y="${m.t + ph - 8}" text-anchor="end">← 不满足 ${escapeHtml(lab)}</text>`);
    }
    // grid + ticks
    for (const v of logTicks(x0, x1)) {
      const x = fx(v);
      parts.push(`<line class="gridl" x1="${x}" x2="${x}" y1="${m.t}" y2="${m.t + ph}"/>`);
      parts.push(`<text class="tick" x="${x}" y="${m.t + ph + 15}" text-anchor="middle">${ax.x_unit === "ms" ? msTick(v) : tickLabel(v)}</text>`);
    }
    for (const v of logTicks(y0, y1)) {
      const y = fy(v);
      parts.push(`<line class="gridl" x1="${m.l}" x2="${m.l + pw}" y1="${y}" y2="${y}"/>`);
      parts.push(`<text class="tick" x="${m.l - 6}" y="${y + 3.5}" text-anchor="end">${tickLabel(v)}</text>`);
    }
    parts.push(`<rect class="axis" fill="none" x="${m.l}" y="${m.t}" width="${pw}" height="${ph}"/>`);
    const xt = narrow ? String(ax.x_label || "").replace(/（.*）/, "") : `${ax.x_label || ""}${lowerBetter ? "（→ 更快）" : "（→ 更好）"} · 对数轴`;
    const yt = narrow ? String(ax.y_label || "").replace(/（.*）/, "") : `${ax.y_label || ""} · 对数轴`;
    parts.push(`<text class="atitle" x="${m.l + pw / 2}" y="${H - 12}" text-anchor="middle">${escapeHtml(xt)}${narrow ? " →" : ""}</text>`);
    parts.push(`<text class="atitle" transform="translate(13 ${m.t + ph / 2}) rotate(-90)" text-anchor="middle">${escapeHtml(yt)}${narrow ? " →" : ""}</text>`);
    // frontier line
    const front = r.frontier.map((i) => r.points[i]).filter((p) => p.x > 0 && p.y > 0).sort((a, b) => fx(a.x) - fx(b.x));
    if (front.length > 1) parts.push(`<polyline class="front-line" points="${front.map((p) => `${fx(p.x).toFixed(1)},${fy(p.y).toFixed(1)}`).join(" ")}"/>`);
    // points: dominated first, frontier on top
    const cmap = layoutColorMap(r);
    const ordered = pts.slice().sort((a, b) => (a.frontier === b.frontier ? 0 : a.frontier ? 1 : -1));
    for (const p of ordered) {
      const cls = ["pt", p.frontier ? "front" : "dom", p.meets_slo ? "" : "viol"].join(" ");
      const c = cmap[p.layout] || "#94a3b8";
      parts.push(`<circle class="${cls}" data-id="${p.id}" cx="${fx(p.x).toFixed(1)}" cy="${fy(p.y).toFixed(1)}" r="${p.frontier ? 4.6 : 3.6}" fill="${c}" stroke="${c}"/>`);
    }
    const cur = pts.find((p) => p.current);
    if (cur) parts.push(`<rect class="cur-mark" x="${fx(cur.x) - 8}" y="${fy(cur.y) - 8}" width="16" height="16"/>`);
    const g = r.goodput || {};
    if (g.ok && g.point_id != null && r.points[g.point_id]) {
      const p = r.points[g.point_id];
      const gx = fx(p.x), gy = fy(p.y);
      parts.push(`<circle class="gp-ring" cx="${gx}" cy="${gy}" r="9"/>`);
      const right = gx < m.l + pw * 0.7;
      parts.push(`<text class="gp-lab" x="${gx + (right ? 13 : -13)}" y="${gy - 10}" text-anchor="${right ? "start" : "end"}">SLO 最优 ${num4(p.y)}</text>`);
    }
    svg.innerHTML = parts.join("");
    svg._paretoMap = { pts, fx, fy };
  }
  function paretoTipHtml(p, r) {
    const d = r.domain || "llm";
    const viol = (p.violates || []).map((v) => BIND_SHORT[v] || v).join(" + ");
    const lines = [`<b>${escapeHtml(p.layout)}</b> · B=${p.batch}${p.current ? "（当前场景）" : ""}`];
    if (d === "llm") {
      lines.push(`<span class="mono">${num4(p.y)}</span> tok/s/芯片 · <span class="mono">${num4(p.x)}</span> tok/s/用户`);
      const amort = r.serving && r.serving.goodput_mode !== "upper";
      lines.push(`TPOT ${fmtTime(p.TPOT_ms)} · TTFT ${fmtTime(p.TTFT_ms)}` +
        (amort ? `<span class="muted">（decode ${fmtTime(p.TPOT_decode_ms)}；prefill 占 ${Math.round((p.prefill_share || 0) * 100)}%）</span>`
          : `<span class="muted">（prefill ${fmtTime(p.TTFT_prefill_ms)} + 1 步）</span>`));
      if (p.decode_mb > 1 || p.spec_tokens_per_step > 1.0001) {
        lines.push(`<span class="muted">${p.decode_mb > 1 ? `PP 微批 ${p.decode_mb}` : ""}${p.decode_mb > 1 && p.spec_tokens_per_step > 1.0001 ? " · " : ""}${p.spec_tokens_per_step > 1.0001 ? `投机 ${fmt(p.spec_tokens_per_step, 2)} tok/步` : ""}</span>`);
      }
    } else {
      lines.push(`<span class="mono">${num4(p.y)}</span> ${PARETO_Y_UNIT[d]} · 时延 ${fmtTime(p.latency_ms)}`);
    }
    lines.push(`容量 ${num4(p.capacity_needed_GB)} / ${num4(p.capacity_GB)} GB · 上限 B=${p.max_batch} · 瓶颈 ${escapeHtml(String(p.wall || "—"))}`);
    lines.push(`${p.frontier ? '<span class="good">帕累托前沿</span>' : '<span class="muted">被支配</span>'} · ` +
      (p.meets_slo ? '<span class="good">满足 SLO</span>' : `<span class="bad">违反 ${escapeHtml(viol)}</span>`));
    return lines.join("<br>");
  }
  function onParetoHover(ev) {
    const svg = $("pareto-svg"), tip = $("pareto-tip");
    const mp = svg._paretoMap;
    if (!mp || !paretoResult) return;
    const pt = svg.createSVGPoint();
    const src = ev.touches && ev.touches[0] ? ev.touches[0] : ev;
    pt.x = src.clientX; pt.y = src.clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const q = pt.matrixTransform(ctm.inverse());
    let best = null, bd = 1e9;
    for (const p of mp.pts) {
      const dx = mp.fx(p.x) - q.x, dy = mp.fy(p.y) - q.y;
      const dd = dx * dx + dy * dy;
      if (dd < bd) { bd = dd; best = p; }
    }
    const lim = 16 / Math.max(ctm.a, 0.2);
    if (!best || bd > lim * lim) { hideParetoTip(); return; }
    if (paretoHoverId !== best.id) {
      svg.querySelectorAll("circle.pt.hover").forEach((c) => c.classList.remove("hover"));
      const c = svg.querySelector(`circle.pt[data-id="${best.id}"]`);
      if (c) { c.classList.add("hover"); c.parentNode.appendChild(c); }
      paretoHoverId = best.id;
      tip.innerHTML = paretoTipHtml(best, paretoResult);
    }
    tip.hidden = false;
    const wrap = $("pareto-chart-wrap").getBoundingClientRect();
    let lx = src.clientX - wrap.left + 14, ly = src.clientY - wrap.top + 14;
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    if (lx + tw > wrap.width - 4) lx = Math.max(4, src.clientX - wrap.left - tw - 14);
    if (ly + th > wrap.height - 4) ly = Math.max(4, src.clientY - wrap.top - th - 14);
    tip.style.left = lx + "px";
    tip.style.top = ly + "px";
  }
  function hideParetoTip() {
    const tip = $("pareto-tip");
    if (tip) tip.hidden = true;
    const svg = $("pareto-svg");
    if (svg) svg.querySelectorAll("circle.pt.hover").forEach((c) => c.classList.remove("hover"));
    paretoHoverId = -1;
  }
  function paretoCsv(r) {
    const cols = ["domain", "model_id", "chips", "layout", "tp", "pp", "ep", "attn", "batch", "max_batch",
      "x_key", "x", "y_key", "y", "TPOT_ms", "TTFT_ms", "TTFT_prefill_ms", "latency_ms",
      "capacity_needed_GB", "capacity_GB", "wall", "frontier", "meets_slo", "violates", "current",
      "moe_shard", "TPOT_step_ms", "TPOT_decode_ms", "prefill_share", "decode_mb", "spec_tokens_per_step",
      "goodput_mode", "prefill_mode", "out_len"];
    const lines = [cols.join(",")];
    const sv = r.serving || {};
    for (const p of r.points) {
      const row = Object.assign({}, p, {
        domain: r.domain, model_id: r.model_id, x_key: r.axes.x_key, y_key: r.axes.y_key,
        violates: (p.violates || []).join("+"),
        goodput_mode: sv.goodput_mode || "", prefill_mode: sv.prefill_mode || "", out_len: sv.out_len || "",
      });
      lines.push(cols.map((c) => csvEscape(row[c] == null ? "" : row[c])).join(","));
    }
    return lines.join("\n") + "\n";
  }
  // KPI：当前布局下的 SLO 吞吐（轻量：仅当前布局）
  function scheduleGoodputKpi() {
    clearTimeout(goodputTimer);
    goodputTimer = setTimeout(updateGoodputKpi, 200);
  }
  async function updateGoodputKpi() {
    const body = paretoBody("current");
    if (!body.model_id) return;
    const seq = ++goodputSeq;
    try {
      const r = await apiPost("/api/pareto", body);
      if (seq !== goodputSeq) return;
      const g = r.goodput || {};
      const d = r.domain || "llm";
      const unit = { llm: "tok/s", video: "帧/s", protein: "seq/s" }[d];
      $("m-goodput").textContent = g.ok ? `${num4(g.y)} ${unit}` : "不满足 SLO";
      const sv = r.serving;
      const gm = !sv ? "" : (sv.goodput_mode === "upper" ? "口径：上界（仅 decode）。"
        : `口径：摊销 prefill（${sv.prefill_mode_zh || sv.prefill_mode}，P=${sv.prompt_len} / N=${sv.out_len}）。`);
      $("m-goodput-sub").textContent = g.ok ? `B=${g.max_users} · ${bindingText(g)} 约束` : `${bindingText(g)} 约束`;
      $("kpi-goodput").title = gm + `当前布局 ${g.layout || ""} 下满足 SLO 的最优吞吐（${d === "llm" ? `TTFT ≤ ${r.slo.ttft_ms} ms，TPOT ≤ ${r.slo.tpot_ms} ms` : `时延 ≤ ${r.slo.latency_ms} ms`}）「假设」。点击查看全部布局的帕累托。`;
      $("kpi-goodput").classList.toggle("warn", !g.ok);
    } catch (err) {
      if (seq !== goodputSeq) return;
      $("m-goodput").textContent = "—";
      $("m-goodput-sub").textContent = "计算失败";
    }
  }

  // ---------- v0.28 布局：标签页 / 抽屉 / 场景摘要 / 分享 ----------
  function setTab(t, push) {
    if (!TABS.includes(t)) t = "overview";
    activeTab = t;
    document.body.dataset.tab = t;
    document.querySelectorAll("#tabs button[data-tab]").forEach((b) => {
      b.classList.toggle("active", b.dataset.tab === t);
      b.setAttribute("aria-selected", b.dataset.tab === t ? "true" : "false");
    });
    if (push !== false) pushConfigToUrl();
    maybeAutoSweep();
    maybeAutoPareto();
  }

  function openDrawer(accId) {
    document.body.classList.add("drawer-open");
    const dr = $("advanced-drawer");
    if (dr) dr.setAttribute("aria-hidden", "false");
    $("btn-advanced").classList.add("active");
    if (accId) {
      const acc = $(accId);
      if (acc) {
        acc.open = true;
        setTimeout(() => acc.scrollIntoView({ block: "start", behavior: "smooth" }), 30);
      }
    }
  }

  function closeDrawer() {
    document.body.classList.remove("drawer-open");
    const dr = $("advanced-drawer");
    if (dr) dr.setAttribute("aria-hidden", "true");
    $("btn-advanced").classList.remove("active");
  }

  function setSt(id, on, onText, offText) {
    const el = $(id);
    if (!el) return;
    el.textContent = on ? onText : offText;
    el.classList.toggle("mod", !!on);
  }

  // v0.31 投机解码 / MoE 切分 / PP 微批 UI
  function setSpecUi(st) {
    if (!st || !$("spec_k")) return;
    if (st.spec_k != null && Number.isFinite(Number(st.spec_k))) $("spec_k").value = Math.max(0, Math.min(8, parseInt(st.spec_k, 10) || 0));
    if (st.spec_accept != null && Number.isFinite(Number(st.spec_accept))) {
      $("spec_accept").value = st.spec_accept;
      $("spec_accept-val").textContent = Number(st.spec_accept).toFixed(2);
    }
    if (st.spec_draft_frac != null && Number.isFinite(Number(st.spec_draft_frac))) $("spec_draft_frac").value = st.spec_draft_frac;
    if (st.decode_mb != null && Number.isFinite(Number(st.decode_mb))) $("decode_mb").value = Math.max(0, parseInt(st.decode_mb, 10) || 0);
    if (st.spec_draft === "mtp" || st.spec_draft === "model") specDraft = st.spec_draft;
    if (st.moe_shard === "tp_ep" || st.moe_shard === "ep_all") moeShard = st.moe_shard;
    document.querySelectorAll("#spec-draft-seg button").forEach((b) => b.classList.toggle("active", b.dataset.draft === specDraft));
    document.querySelectorAll("#moe-shard-seg button").forEach((b) => b.classList.toggle("active", b.dataset.moe === moeShard));
    $("spec-frac-row").hidden = specDraft !== "model";
  }
  function updateAmortUi() {
    const amort = ($("goodput-mode") && $("goodput-mode").value) !== "upper";
    document.querySelectorAll("#tab-pareto .amort-only").forEach((e) => { e.classList.toggle("off", !amort); });
  }

  // 高级参数各组状态 + 「N 项已修改」徽标（不触发求值）
  function updateAdvancedStatus() {
    if (!$("lock-parallel")) return;
    const v = (id) => parseFloat($(id) && $(id).value);
    const lock = $("lock-parallel").checked;
    setSt("st-parallel", lock, `已锁定 ${$("tp").value}/${$("pp").value}/${$("ep").value}`, "默认映射");
    const mm = $("manual-mem").checked, mc = $("manual-compute").checked;
    setSt("st-mem", mm, "手动几何", "使用目录");
    const sy = Math.abs(v("c2c_latency_us") - DEFAULT_C2C_LAT_US) > 1e-6 ||
      Math.abs(v("sync_overlap") - DEFAULT_SYNC_OVERLAP) > 1e-6 || attnParallel !== "tp";
    setSt("st-sync", sy,
      `α ${v("c2c_latency_us").toFixed(1)} µs · 重叠 ${v("sync_overlap").toFixed(2)} · 注意力 ${attnParallel.toUpperCase()}`,
      `α ${DEFAULT_C2C_LAT_US.toFixed(0)} µs · 暴露`);
    setSt("st-compute", mc, "手动 核数 × T/核", "使用预设");
    const sk = parseInt($("spec_k").value, 10) || 0, dmb = parseInt($("decode_mb").value, 10) || 0;
    const sp = sk > 0 || moeShard !== "tp_ep" || dmb > 0;
    setSt("st-spec", sp,
      [sk > 0 ? `k=${sk} · a=${v("spec_accept").toFixed(2)} · ${specDraft === "mtp" ? "MTP" : "草稿模型"}` : "",
        moeShard !== "tp_ep" ? "EP 全卡" : "", dmb > 0 ? `微批 ${dmb}` : ""].filter(Boolean).join(" · "),
      "关闭");
    const ov =
      Math.abs(v("efficiency") - DEFAULT_MEM_EFF) > 1e-6 ||
      Math.abs(v("mac_eff") - DEFAULT_MAC_EFF) > 1e-6 ||
      Math.abs(v("freq_ghz") - DEFAULT_FREQ_GHZ) > 1e-6;
    setSt("st-override", ov, "已覆盖", "默认（假设）");
    const ac = (v("non_gemm_overhead") || 0) > 0 ||
      !!($("use_finer_non_gemm") && $("use_finer_non_gemm").checked) ||
      (($("dtype_mac_factors") && $("dtype_mac_factors").value) || "").trim().length > 0;
    setSt("st-assumed", ac, "已启用", "关闭");
    const ec = Object.keys(collectEcon()).length > 0;
    setSt("st-econ", ec, "已启用", "关闭");
    const n = [lock, mm, sy, mc, ov, ac, ec, sp].filter(Boolean).length;
    const badge = $("adv-count");
    if (badge) {
      badge.hidden = n === 0;
      badge.textContent = `${n} 项已修改`;
    }
  }

  function updateSceneSummary() {
    const el = $("scene-summary");
    if (!el || !$("series")) return;
    const chips = $("chip_count").value;
    let pkg = "—";
    if ($("manual-mem").checked) pkg = "手动存储";
    else if (memSel && memCatalog) pkg = memShortLabel(memSel, memDerive(memSel, parseFloat($("efficiency").value)));
    const comp = $("manual-compute").checked ? "手动算力" : $("compute").value || "—";
    el.textContent = `${domainZh(domain)} · ${$("series").value || "—"} · ${pkg} · ${comp} · ×${chips}`;
  }

  function buildShareUrl() {
    const url = new URL(window.location.href);
    url.searchParams.set("c", b64urlEncode(JSON.stringify(collectUiState())));
    return url.toString();
  }

  async function copyShareLink() {
    let u;
    try {
      u = buildShareUrl();
      history.replaceState(null, "", u);
    } catch (e) {
      showToast("生成链接失败：" + e);
      return;
    }
    let ok = false;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(u);
        ok = true;
      }
    } catch (e) { ok = false; }
    if (!ok) {
      try {
        const ta = document.createElement("textarea");
        ta.value = u;
        ta.setAttribute("readonly", "");
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        ok = document.execCommand("copy");
        ta.remove();
      } catch (e) { ok = false; }
    }
    showToast(ok ? "已复制链接（含当前配置 ?c=）" : "无法写入剪贴板 —— 请直接复制地址栏链接", ok ? "ok" : "");
  }

  async function init() {
    setStatus("加载目录中…", "loading");
    try {
      const [health, series, memory, compute] = await Promise.all([
        apiGet("/api/health"),
        apiGet("/api/series"),
        apiGet("/api/memory"),
        apiGet("/api/compute"),
      ]);
      $("ver").textContent = "v" + (health.version || "?");
      renderBanner(health.banner || series.banner || "");
      seriesCache = series.series || [];
      memCatalog = memory.catalog || null;
      memSel = normalizeMemSel(memDefaults(DEFAULT_MEM_TYPE));
      computeCache = compute.compute || [];
      try {
        const pr = await apiGet("/api/presets");
        presetsCache = pr.presets || [];
        econExample = pr.econ_example || {};
      } catch (e) {
        presetsCache = [];
      }
      populateSeries();
      renderMemUi();
      populateCompute();
      const restored = await restoreConfigFromUrl();
      if (!restored) syncTpFromChips();
      else updateParallelHint();
      updateBaselineHint();
      updateOverrideBadge();
      renderDualCards();
      setStatus(restored ? "就绪 · 已从 ?c= 恢复" : "就绪", "ok");
      scheduleEval();
      pushConfigToUrl();
    } catch (err) {
      showError("目录加载失败：" + err);
      setStatus("目录加载失败", "error");
    }

    wireSeg("domain-seg", "data-domain", (d) => {
      if (d === domain) return;
      domain = d;
      populateSeries();
      updateParallelHint();
      // 扫描结果属于旧领域 —— 清空；若扫描面板可见则按新领域自动重跑一次
      if (lastSweepResult || sweepAutoRan) {
        clearSweep();
        sweepAutoRan = false;
      }
      scheduleEval();
      maybeAutoSweep();
    });
    // v0.29 结构化存储选择器
    $("mem-type").addEventListener("change", () => setMemType($("mem-type").value));
    $("mem-chip").addEventListener("click", (e) => {
      e.stopPropagation();
      openMemPop();
    });
    $("mp-close").addEventListener("click", () => openMemPop(false));
    $("mp-rate").addEventListener("change", () => setMemField("mem_rate_MTps", Number($("mp-rate").value)));
    $("mp-count").addEventListener("change", () => setMemField("mem_count", Number($("mp-count").value)));
    $("mp-cap").addEventListener("change", () => setMemField("mem_cap_GB", Number($("mp-cap").value)));
    $("mp-manual").addEventListener("click", () => {
      openMemPop(false);
      openDrawer("acc-mem");
    });
    document.addEventListener("mousedown", (e) => {
      const pop = $("mem-pop");
      if (!pop || pop.hidden) return;
      if (pop.contains(e.target) || $("mem-chip").contains(e.target)) return;
      openMemPop(false);
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !$("mem-pop").hidden) openMemPop(false);
    });
    window.addEventListener("resize", positionMemPop);
    window.addEventListener("scroll", positionMemPop, { passive: true });
    // v0.29 通信同步 / 注意力并行（假设）
    ["c2c_latency_us", "sync_overlap"].forEach((id) => {
      $(id).addEventListener("input", () => {
        $(id + "-val").textContent = Number($(id).value).toFixed(id === "sync_overlap" ? 2 : 1);
        updateAdvancedStatus();
        scheduleEval();
      });
    });
    wireSeg("attn-parallel-seg", "data-attn", (a) => {
      attnParallel = a === "dp" ? "dp" : "tp";
      updateAdvancedStatus();
      scheduleEval();
    });
    // v0.31 投机解码 / MoE 切分 / PP 微批
    wireSeg("spec-draft-seg", "data-draft", (a) => {
      specDraft = a === "model" ? "model" : "mtp";
      $("spec-frac-row").hidden = specDraft !== "model";
      updateAdvancedStatus();
      scheduleEval();
    });
    wireSeg("moe-shard-seg", "data-moe", (a) => {
      moeShard = a === "ep_all" ? "ep_all" : "tp_ep";
      updateAdvancedStatus();
      scheduleEval();
    });
    $("spec_accept").addEventListener("input", () => {
      $("spec_accept-val").textContent = Number($("spec_accept").value).toFixed(2);
      updateAdvancedStatus();
      scheduleEval();
    });
    ["spec_k", "spec_draft_frac", "decode_mb"].forEach((id) => {
      $(id).addEventListener("change", () => {
        updateAdvancedStatus();
        scheduleEval();
      });
    });

    $("series").addEventListener("change", () => {
      updateSeriesMeta();
      seedDomainKnobsFromSeries();
      scheduleEval();
    });
    $("product-only").addEventListener("change", () => {
      populateSeries();
      scheduleEval();
    });
    $("compute").addEventListener("change", () => {
      updateComputeMeta();
      scheduleEval();
    });

    $("chip_count").addEventListener("input", () => {
      syncTpFromChips();
      scheduleEval();
    });
    ["tp", "pp", "ep"].forEach((id) => {
      $(id).addEventListener("input", () => {
        if (!$("lock-parallel").checked) {
          $("lock-parallel").checked = true;
        }
        updateParallelHint();
        scheduleEval();
      });
    });
    $("lock-parallel").addEventListener("change", () => {
      if (!$("lock-parallel").checked) syncTpFromChips();
      updateParallelHint();
      scheduleEval();
    });

    $("manual-mem").addEventListener("change", () => {
      const on = $("manual-mem").checked;
      $("manual-mem-fields").style.display = on ? "block" : "none";
      renderMemUi();
      updateSceneSummary();
      scheduleEval();
    });
    $("manual-compute").addEventListener("change", () => {
      const on = $("manual-compute").checked;
      $("manual-compute-fields").style.display = on ? "block" : "none";
      $("compute").disabled = on;
      scheduleEval();
    });

    ["n_channels", "width_bits", "data_rate_GTs", "capacity_GB"].forEach((id) => {
      $(id).addEventListener("input", () => { if ($("manual-mem").checked) renderMemUi(); });
    });
    [
      "n_channels",
      "width_bits",
      "data_rate_GTs",
      "capacity_GB",
      "n_cores",
      "tops_per_core",
      "batch",
      "prompt",
      "ctx",
      "n_denoise",
      "n_frames",
      "seq_len",
      "dtype",
      "quant",
    ].forEach((id) => {
      $(id).addEventListener("input", scheduleEval);
      $(id).addEventListener("change", scheduleEval);
    });

    $("efficiency").addEventListener("input", () => {
      $("efficiency-val").textContent = Number($("efficiency").value).toFixed(2);
      updateOverrideBadge();
      renderMemUi();
      scheduleEval();
    });
    $("mac_eff").addEventListener("input", () => {
      $("mac_eff-val").textContent = Number($("mac_eff").value).toFixed(2);
      updateOverrideBadge();
      scheduleEval();
    });
    if ($("freq_ghz")) {
      $("freq_ghz").addEventListener("input", () => {
        $("freq_ghz-val").textContent = Number($("freq_ghz").value).toFixed(2);
        updateOverrideBadge();
        scheduleEval();
      });
    }

    // v0.28：场景预设为下拉（<option data-preset>）
    const presetSel = $("preset-select");
    if (presetSel) {
      [...presetSel.options].forEach((o) => {
        const pz = PRESET_ZH[o.dataset.preset];
        const pre = presetsCache.find((x) => x.id === o.dataset.preset);
        if (pz || pre) o.title = `${o.dataset.preset}：${(pz && pz.label) || (pre && pre.label) || ""} —— ${(pz && pz.note) || (pre && pre.note) || ""}`;
      });
      presetSel.addEventListener("change", () => {
        const opt = presetSel.selectedOptions[0];
        if (opt && opt.dataset.preset) applyPreset(opt.dataset.preset);
      });
    }
    ["mem-type", "compute", "chip_count", "manual-mem", "manual-compute", "lock-parallel", "tp", "pp", "ep"].forEach((id) => {
      const el = $(id);
      if (el) { el.addEventListener("change", markPresetCustom); el.addEventListener("input", markPresetCustom); }
    });
    ECON_NUM_FIELDS.forEach((id) => {
      const el = $(id);
      if (el) {
        el.addEventListener("input", scheduleEval);
        el.addEventListener("change", scheduleEval);
      }
    });
    ["power_util", "duty_cycle"].forEach((id) => {
      $(id).addEventListener("input", () => {
        $(id + "-val").textContent = Number($(id).value).toFixed(2);
        scheduleEval();
      });
    });
    if ($("btn-econ-example")) $("btn-econ-example").addEventListener("click", loadEconExample);
    if ($("btn-econ-clear"))
      $("btn-econ-clear").addEventListener("click", () => {
        setEconUi(null);
        scheduleEval();
      });

    $("btn-eval").addEventListener("click", doEval);
    $("btn-reset").addEventListener("click", () => {
      baselineCard = null;
      baselineMs = null;
      dualCardA = null;
      dualCardB = null;
      updateBaselineHint();
      renderDualCards();
      resetDefaults();
      pushConfigToUrl();
    });
    if ($("btn-pin-baseline")) {
      $("btn-pin-baseline").addEventListener("click", () => {
        if (lastCard) pinBaselineFromCard(lastCard);
        else showToast("请先评估，再固定基线");
      });
    }
    if ($("btn-pin-a")) {
      $("btn-pin-a").addEventListener("click", () => pinDual("A", lastCard));
    }
    if ($("btn-pin-b")) {
      $("btn-pin-b").addEventListener("click", () => pinDual("B", lastCard));
    }
    if ($("btn-dual-clear")) {
      $("btn-dual-clear").addEventListener("click", clearDual);
    }
    if ($("btn-dual-swap")) {
      $("btn-dual-swap").addEventListener("click", () => {
        const tmp = dualCardA;
        dualCardA = dualCardB;
        dualCardB = tmp;
        renderDualCards();
        pushConfigToUrl();
      });
    }
    if ($("btn-dual-from-baseline")) {
      $("btn-dual-from-baseline").addEventListener("click", () => {
        if (!lastCard) {
          showToast("请先评估");
          return;
        }
        if (baselineCard && baselineCard.ok !== false && (baselineCard.TTFT_ms != null || baselineCard.TPOT_ms != null || baselineCard.TTFC_ms != null || baselineCard.time_per_seq_ms != null)) {
          dualCardA = baselineCard;
        } else if (baselineCard) {
          // shallow baseline from URL may lack full fields — still use last as B
          dualCardA = baselineCard;
        } else {
          showToast("请先固定扫描基线（或手动固定 A）");
          return;
        }
        dualCardB = lastCard;
        renderDualCards();
        pushConfigToUrl();
        showToast("A|B：基线 → A，最近一次评估 → B");
      });
    }

    // ---------- v0.28 布局交互 ----------
    wireAssumedCompute();
    // 引擎 TP/PP/EP ∈ {1,2,4,8}；未锁定时 TP = 芯片数 → 步进器按 2 的幂移动（未锁定上限 8）
    const step = (dlt) => {
      const el = $("chip_count");
      const cur = parseInt(el.value, 10) || 1;
      const max = $("lock-parallel").checked ? 64 : 8;
      const seq = [1, 2, 4, 8, 16, 32, 64].filter((x) => x <= max);
      const v = dlt > 0 ? seq.find((x) => x > cur) : [...seq].reverse().find((x) => x < cur);
      if (v == null) {
        if (dlt > 0) showToast(`未锁定并行时芯片数上限 ${max}（TP ∈ {1,2,4,8}）；更大规模请在 ⚙ 高级参数 › 并行 中锁定 TP/PP/EP`);
        return;
      }
      el.value = v;
      el.dispatchEvent(new Event("input", { bubbles: true }));
    };
    $("chip-dec").addEventListener("click", () => step(-1));
    $("chip-inc").addEventListener("click", () => step(+1));
    $("btn-share").addEventListener("click", copyShareLink);
    $("btn-advanced").addEventListener("click", () => {
      if (document.body.classList.contains("drawer-open")) closeDrawer();
      else openDrawer();
    });
    $("btn-drawer-close").addEventListener("click", closeDrawer);
    if ($("btn-advanced-m")) $("btn-advanced-m").addEventListener("click", () => openDrawer());
    $("drawer-mask").addEventListener("click", closeDrawer);
    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape" && document.body.classList.contains("drawer-open")) closeDrawer();
    });
    $("par-summary").addEventListener("click", () => openDrawer("acc-parallel"));
    document.addEventListener("click", (ev) => {
      const b = ev.target.closest("[data-open-acc]");
      if (b) openDrawer(b.getAttribute("data-open-acc"));
    });
    ["btn-honesty", "btn-banner-more", "btn-assump-all"].forEach((id) => {
      const el = $(id);
      if (el) el.addEventListener("click", () => setTab("assump"));
    });
    // v0.30 吞吐 / 交互
    $("btn-pareto").addEventListener("click", runPareto);
    $("goodput-mode").addEventListener("change", updateAmortUi);
    updateAmortUi();
    ["slo-ttft", "slo-tpot", "slo-lat", "pareto-layouts", "goodput-mode", "prefill-mode", "out-len"].forEach((id) => {
      $(id).addEventListener("change", () => {
        markParetoStale();
        scheduleGoodputKpi();
        pushConfigToUrl();
        maybeAutoPareto();
      });
    });
    $("btn-pareto-csv").addEventListener("click", () => {
      if (!paretoResult) return;
      downloadBlob(`pareto_${paretoResult.model_id}_${paretoResult.chips}chips.csv`, paretoCsv(paretoResult), "text/csv;charset=utf-8");
    });
    $("btn-pareto-json").addEventListener("click", () => {
      if (!paretoResult) return;
      downloadBlob(`pareto_${paretoResult.model_id}_${paretoResult.chips}chips.json`, JSON.stringify(paretoResult, null, 1), "application/json");
    });
    {
      const svg = $("pareto-svg");
      svg.addEventListener("mousemove", onParetoHover);
      svg.addEventListener("mouseleave", hideParetoTip);
      svg.addEventListener("touchstart", onParetoHover, { passive: true });
      svg.addEventListener("click", onParetoHover);
    }
    $("kpi-goodput").addEventListener("click", () => setTab("pareto"));
    {
      let rz = null, lastNarrow = null;
      window.addEventListener("resize", () => {
        clearTimeout(rz);
        rz = setTimeout(() => {
          const n = ($("pareto-svg").clientWidth || 760) < 560;
          if (paretoResult && n !== lastNarrow) drawParetoChart(paretoResult);
          lastNarrow = n;
        }, 150);
      });
    }
    updateSloInputs();
    $("tabs").addEventListener("click", (ev) => {
      const b = ev.target.closest("button[data-tab]");
      if (b) setTab(b.dataset.tab);
    });
    $("btn-scene-toggle").addEventListener("click", () => {
      const open = document.body.classList.toggle("scene-open");
      $("btn-scene-toggle").setAttribute("aria-expanded", open ? "true" : "false");
      $("btn-scene-toggle").textContent = open ? "收起 ▴" : "编辑场景 ▾";
    });
    if ($("dual-show-all")) $("dual-show-all").addEventListener("change", renderDualDelta);
    updateAdvancedStatus();
    updateSceneSummary();
    // v0.28.1：下拉框贴合选中文字；抽屉停靠在吸顶区下方（场景栏保持全宽 2 行）
    ["preset-select", "dtype", "quant"].forEach((id) => {
      const el = $(id);
      if (el) el.addEventListener("change", () => fitSelect(el));
    });
    fitCoreSelects();
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(fitCoreSelects);
    let fitTimer = null;
    window.addEventListener("resize", () => {
      clearTimeout(fitTimer);
      fitTimer = setTimeout(fitCoreSelects, 120);
    });
    const head = $("head");
    const setHeadH = () => document.documentElement.style.setProperty("--head-h", head.getBoundingClientRect().height + "px");
    setHeadH();
    if (window.ResizeObserver) new ResizeObserver(setHeadH).observe(head);
    if (window.matchMedia) {
      const mq = window.matchMedia("(min-width: 1680px)");
      if (mq.addEventListener) mq.addEventListener("change", maybeAutoSweep);
    }
    maybeAutoSweep();

    // Compare / Sweep panel
    updateSweepOpts();
    setExportEnabled(false);
    wireSeg("sweep-axis-seg", "data-axis", (a) => {
      sweepAxis = a;
      updateSweepOpts();
    });
    const parSeg = $("sweep-par-chips-seg");
    if (parSeg) {
      parSeg.addEventListener("click", (ev) => {
        const btn = ev.target.closest("button");
        if (!btn) return;
        parSeg.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        sweepParallelChips = parseInt(btn.getAttribute("data-chips"), 10) || 4;
      });
    }
    if ($("btn-sweep")) $("btn-sweep").addEventListener("click", () => doSweep());
    if ($("btn-par4"))
      $("btn-par4").addEventListener("click", () => {
        sweepAxis = "parallel";
        sweepParallelChips = 4;
        document.querySelectorAll("#sweep-axis-seg button").forEach((b) => {
          b.classList.toggle("active", b.dataset.axis === "parallel");
        });
        document.querySelectorAll("#sweep-par-chips-seg button").forEach((b) => {
          b.classList.toggle("active", b.dataset.chips === "4");
        });
        updateSweepOpts();
        doSweep({ axis: "parallel", parallel_chips: 4, chip_count: 4 });
      });
    if ($("btn-par8"))
      $("btn-par8").addEventListener("click", () => {
        sweepAxis = "parallel";
        sweepParallelChips = 8;
        document.querySelectorAll("#sweep-axis-seg button").forEach((b) => {
          b.classList.toggle("active", b.dataset.axis === "parallel");
        });
        document.querySelectorAll("#sweep-par-chips-seg button").forEach((b) => {
          b.classList.toggle("active", b.dataset.chips === "8");
        });
        updateSweepOpts();
        doSweep({ axis: "parallel", parallel_chips: 8, chip_count: 8 });
      });
    if ($("btn-sweep-clear")) $("btn-sweep-clear").addEventListener("click", clearSweep);
    if ($("btn-export-csv")) $("btn-export-csv").addEventListener("click", exportSweepCsv);
    if ($("btn-export-json")) $("btn-export-json").addEventListener("click", exportSweepJson);
    ["sweep-pkg-kind", "sweep-compute-level"].forEach((id) => {
      const el = $(id);
      if (el) el.addEventListener("change", () => {});
    });
  }

  document.addEventListener("DOMContentLoaded", init);
})();

# REQUIREMENTS_GAP.md — 原始需求对照（v0.26）

对照来源：对 Web「芯片模拟器」的残酷评审（§5.3 最小可手算微架构 + P0）以及本仓库章程（推理-only NPU + SRAM + HBM/LPDDR DSE）。  
状态：**已齐** / **半齐** / **缺**。CLI 指向当前入口。

> 绝对硬件数均为 **assumed/uncalibrated**，除非由形状 + tiling + 容量 **derived**。  
> **「标定」本版定义**：用户要的是 **可选取的 package / compute 设计空间档位（range DSE）**，不是硅后实测数。假设档已入库；若将来有实测 efficiency / freq，再替换 knobs。

---

## 1. 用户原始 bullets → 状态

| # | 原始需求（意涵） | 状态 | 对应 CLI / 模块 |
|---|------------------|------|-----------------|
| 1 | PE：固定阵列 / MAC/cycle + 明确 dataflow + **小 M 利用率公式** | **已齐** | `util-curve`；`npu.py` OS：`util(M=1)=1/R`（N%effC==0） |
| 2 | SRAM：分区容量；**由 tiling 推导外部字节**（禁止默认手填 hit-rate） | **已齐** | `sweep-sram`；`--sram-policy`；`derive_sram_partitions`；R≥1 才减 W DRAM |
| 3 | HBM/LPDDR：channels × width × rate × efficiency；KV 与 weight **争用同端口** | **已齐** | `eval --mem`；`workbench --mem` + geometry；`list-packages`；争用 `serialize`/`share_fair`/`lower_bound` |
| 4 | 输出 TTFT/TPOT，分解 compute vs mem；对 SRAM / 通道 / PE 扫拐点 | **已齐** | `scan` / `scan-sku` / `sweep-*` / `workbench-scan` / `report` |
| 5 | 负载先 1–2 个可核对结构（非十家族） | **已齐** | `toy` + `illustrative_*` 手算；另有数十个 **公开 HF** 真实维数 packs |
| 6 | Decode M=1 利用率显式曲线 | **已齐** | `util-curve`；handcheck + `test_decode_m1_util_*` |
| 7 | SRAM 容量必须改变外部字节（staging≠resident 诚实） | **已齐** | `sweep-sram`；FINDINGS §3；R=0 不减字节 |
| 8 | Weight+KV 争用带宽并在结果分解 | **已齐** | MetricsCard / PhaseResult breakdown；`report` wall 说明 |
| 9 | 手算可核对（toy 全数字） | **已齐** | `handcheck`；`examples/handcheck.md`；守恒测试 |
| 10 | 不做伪 PDK 功耗/面积 | **已齐**（刻意不做） | `process_nm` 仅标签；章程写死。v0.24 energy/cost 仅为 **用户 knob 的 assumed stub**（见 #19/#20），**不是** PDK 推导 |
| 11 | Scale-up：TP/PP/EP + C2C + KV fabric（assumed） | **已齐** | `sweep-tp` / `sweep-parallel` / `sweep-moe --ep` / `sweep-kv-fabric` / workbench |
| 12 | MoE 激活流 + MLA 压缩 KV | **已齐** | `sweep-moe` / `sweep-mla`；series packs |
| 13 | 多域：video DiT / protein pair 示意 | **已齐（示意）** | `sweep-video` / `sweep-protein` / `compare-domains`；workbench `/api/eval` MetricsCard；video/protein **tp/pp/ep** + ring/tree collectives + PP bubble |
| 14 | 产品层 Workbench：chips → parallel、geometry、MetricsCard | **已齐** | `workbench` / `workbench-scan` / `workbench-parallel`；`--json`/`--md` |
| 15 | 离线可读一页报告（Metrics/wall） | **已齐** | `python3 -m accel_dse report --out out/report.html` |
| 16 | **真实**系列维数（公开 HF / model-card dims） | **已齐（公开 HF）** | `list-series --product`；`accel_dse/data/series_catalog.json`；gated 仍缺（Llama-4/Gemma/ESM-3） |
| 17 | **Package / compute 设计空间档位（原「标定」）** | **已齐（假设档）** | `list-packages` / `list-compute`；`--sweep-package` / `--sweep-compute`；`package_ranges.py`。硅后实测 efficiency **可选**，非阻塞 |
| 18 | UI 产品化（浏览器工作台 / 图表） | **已齐（MVP+）** | `python3 -m accel_dse serve`；knobs + MetricsCard + **Compare/Sweep** + **dual card A\|B** + **Assumed/override**（mem/mac/freq；override badge；`?c=`）+ **calib inject**（CLI `--calib` / API `calib`）；CSV/JSON export；baseline Δ%；Parallel×4/×8；无 CDN；**非完整 BI** |
| 19 | **能效 energy efficiency**（早期用户指标：J/token · J/frame · J/seq） | **半齐（assumed stub）** | `tdp_w` **或** `watts_per_tops × peak_tops` × `power_util`(默认 1.0=TDP 上界) × chips → `est_power_W`；`est_energy_per_token_J`(llm) / `est_energy_per_frame_J`(video) / `est_energy_per_seq_J`(protein)。CLI `--tdp-w` / `--watts-per-tops` / `--econ`；API flat 或 `econ:{}`；UI Energy/cost 区。**引擎默认 OFF**（不设 knob 不出数）；示例占位 400 W/card 仅在 `examples/energy_cost.example.json`。缺：真实功耗模型（动态/静态、DRAM pJ/bit、SRAM、NoC、host/PUE）— 刻意不伪造 |
| 20 | **成本 cost**（系统 $ / $ per MTok） | **半齐（assumed stub）** | `est_system_cost_usd = chips × (cost_per_card_usd + mem_addon_usd)`；LLM `est_usd_per_Mtok` = 电费(`usd_per_kwh`) + capex 摊销(`amortize_years × duty_cycle`)，decode-only tokens。CLI `--cost-per-card` / `--usd-per-kwh` / `--amortize-years`。缺：BOM 拆分、良率、网络/机柜/冷却、售价模型；0.5¢/token 仅为示例 JSON 中的占位比较价 |
| 21 | 场景预设（edge / card / scale-up 一键） | **已齐（assumed 档）** | `list-presets` / `eval-presets` / `workbench --preset`；API `preset:` + `GET /api/presets`；Web 一键按钮：`edge-lpddr-4x64` · `card-hbm-4stack` · `scaleup-8chip`（只捆 package+compute+chips，不含功耗/价格） |

---

## 2. 他们没写但已提供（额外 knobs / metrics）

| 项 | CLI / 字段 | 说明 |
|----|------------|------|
| SKU 峰值模板 100T / 1P | `scan-sku` / `list-sku` / `--sku` | PE×engines×freq×2flop/MAC **derived** |
| 独立 W/KV quant | `sweep-quant`；`w4k16` 等 | 长 ctx KV 反超叙事 |
| dtype 只缩字节（默认） | `sweep-dtype` / `dtype_mac_factors` | 默认 MAC rate 不变 = 对低精度保守；v0.26 可选 assumed factor 表（默认 1.0） |
| Batch 扫参（util↑） | `sweep-batch` | 墙可能翻转 |
| SRAM 膝点 R≥1 / L/2 / L | `sweep-sram` | GiB 级 resident 才省权重流 |
| weight_hide_factor | `--weight-hide` | 仅 dbl-buf staging；折时间不折字节 |
| chip_count→tp 默认映射 | `workbench --chips` | 显式 tp/pp/ep 须乘积==chips（llm + video + protein） |
| n_cores × tops_per_core | `--cores` / `--tops-per-core` | 产品算力语言 → 近方形 PE |
| DRAM geometry knobs | `--n-channels` / `--data-rate-gts` / … | `workbench-scan --sweep-mem` 看 FLIP |
| **Package catalog** | `list-packages`；`--sweep-package`；`--package-axis type\|count\|rate`；`GET /api/memory` | v0.29 结构化：LPDDR5 / 5X **x64 封装（4×16）**（x32；x96 已发布）、SOCAMM2 / LPCAMM2 128-bit 模组；LPDDR6 **x96 封装（4×24 = 8×12 子通道，payload 8/9）**；HBM3 / 3E / 4 / 4E 堆 × 层数 × die 密度；速率依附代际；每项带来源标签（最弱项） |
| **Compute catalog** | `list-compute`；`--sweep-compute` | core 4–16T；cluster 64–256T |
| parallel 矩阵排序 | `workbench-parallel --sort-by TPOT\|TTFT` | dense ep=1；MoE EP |
| MetricsCard JSON/MD | `workbench --json/--md` | 稳定字段落盘 |
| Series packs | `list-series --product` | `series/dense-27b` 等 |
| Cross-domain 一页 | `compare-domains` | LLM TPOT / video TTFC / protein t/seq |
| CSV bundle | `export-csv` | 多 sweep 落盘 |
| HTML 报告 | `report` | walls / series / **packages** / **compute** / chips / parallel / mem / compare |
| **Web workbench MVP** | `serve` | knobs → live MetricsCard（llm/video/protein）；`/api/eval`；assumed/uncalibrated banner |
| **Compare / Sweep UI** | `POST /api/sweep` | 方案对比表 + 轻量 bar（TPOT/TTFC/t/seq）；parallel 矩阵 chips=4/8；grid ≤32；**CSV/JSON export**；**baseline Δ%** |
| **Config deep-link** | UI `?c=` | base64url JSON of knobs; refresh restores; shareable URL |
| **CalibrationOverrides** | `--calib` / API `calib` | mem_efficiency / mac_efficiency(or weight_hide) / frequency_hz；`examples/calibration.example.json`；可选非阻塞 |
| **Dual card A\|B** | UI Pin A/B | 两张 MetricsCard 字段级 Δ；可 last vs baseline |
| video/protein PP/EP + collectives | `scale_domain_parallel` | ring/tree TP acts；PP bubble（denoise poorly）；EP no-op w/o MoE |
| **Energy / cost stub（v0.24，assumed）** | `est_*` MetricsCard 字段 | `econ_basis` 文本 + `[assumed econ stub]` assumptions；sweep CSV / dual A\|B / report `#econ` 同步 |
| **Scenario presets（v0.24）** | `--preset` / `preset:` | edge-lpddr-4x64 · card-hbm-4stack · scaleup-8chip |
| **Scaling efficiency（v0.25）** | MetricsCard `scale_efficiency` / `speedup` | vs chips=1 primary；Web / A\|B / CSV / report `#scale` |

---

## 3. 真正剩余缺口（需用户拍板 / 锁定）

仅列 **需要用户锁**、且会改变「敢不敢报绝对数 / 产品形态」的项：

1. **真实系列维数（公开 HF）** — **已齐**：公开 config.json / model-card / cited schema 已入库；illustrative 仍保留手算路径。Gated（Llama-4 / Gemma / ESM-3）与缺失 dims（Chai-1 / RFdiffusion）仍跳过。Hybrid linear-attn / Engram / DSA 仅 metadata，核心仍走 ModelShape 会计。  
2. **Package / compute 设计空间档位** — **已齐（结构化目录 + 来源标签；效率为假设）**：v0.29 按 JEDEC / 厂商资料重建 —— LPDDR5 / 5X = **x64 封装（4×16-bit 通道）**；LPDDR6 = **x96 封装（4×24-bit 通道，每通道 2×12 子通道；payload 8/9 单独计）**；SOCAMM2 / LPCAMM2 模组；HBM 容量 = 层数 × die 密度；组合标签取最弱项（JEDEC … 推测）。效率仍为可调假设 0.70；若将来提供 **实测 sustained efficiency / 硅后 freq**，再替换默认 0.70 / 1 GHz（可选，不阻塞 DSE）。  
3. **UI 产品化** — **已齐（MVP+）**：本地 `serve` knobs + MetricsCard + Compare/Sweep + **dual A|B** + **Assumed/override calib**（mem/mac/freq；badge；`--calib` / `calib:{}`；`?c=`）已可用；**非完整 BI**。静态 `report.html` 仍可离线审计。

4. **能效 / 成本** — **半齐（assumed stub）**：J/token·J/frame·J/seq、系统 $、LLM $/MTok 已有字段与 knobs，但功耗与价格均为 **用户输入的假设值**（引擎默认关闭；示例 JSON 为明显假数）。若要「敢报」绝对能效，需要用户提供 实测/目标 TDP 或 W/TOPS 与卡成本；真实 PDK/DRAM pJ/bit 功耗模型仍刻意不做。

其余（**原生**低精度 MAC 硅后标定、伪 PDK、cycle-accurate NoC/Softmax、MoE 不均路由等）见 `DECISIONS.md`；v0.26 已提供 **assumed opt-in** `non_gemm_overhead` + `dtype_mac_factors`（默认 off/1.0），**不**阻塞继续扫参。

---

### v0.26 assumed compute extras

| 项 | 状态 | CLI / API / UI |
|----|------|----------------|
| Softmax/RoPE/LN 粗开销 | **半齐（assumed opt-in）** | `--non-gemm-overhead` / finer fracs；默认 0；非 cycle-accurate |
| 低精度 MAC peak factor | **半齐（assumed opt-in）** | `--dtype-mac-factor` JSON；默认全 1.0（bytes-only）；用户/assumed 非硅 |

*版本：0.31.1 — 与 CHANGELOG / README 同步。*

### v0.29 存储目录 + 乐观假设修正

| 项 | 状态 | 说明 |
|----|------|------|
| 存储结构化选择（类型 / 形态 / 位宽 / 速率 / 数量 / 容量；HBM 层数 / die 密度 / 堆数） | **已齐** | `mem_catalog.py`；API 结构化字段；UI 存储弹层；CLI `--mem-type` 等 |
| 来源标签（最弱项） | **已齐** | 资料截至 2026-10（`research/memory_specs_2026-10.*`）；新产品发布后需人工更新标签 |
| LPDDR6 payload 8/9 | **已齐** | 原始 / 可用分开显示 |
| 暴露的通信同步 α | **半齐（假设 α）** | 默认 3 µs / 次、完全暴露；未标定；未区分拓扑（ring / tree / switch）与消息大小 |
| KV 按 TP 切分 / MLA 复制 / 注意力 DP | **半齐** | GQA / MLA 已齐；DP 的注意力计算与 DP↔TP 重分片只做近似；MLA 注意力权重 v0.30 已按投影建模（见下） |

### v0.30 MLA 权重 / 容量单位 / 吞吐–交互帕累托 + SLO goodput

| 项 | 状态 | 说明 |
|----|------|------|
| MLA 注意力权重（q_lora / kv_lora / nope / rope / v 投影） | **已齐** | DeepSeek-V3 670.9B（公开 ≈ 671B，测试 ±1%）；前置稠密层 + 非共享 LM head；MTP 层不计 |
| DeepSeek-V4（无 kv_lora） | **半齐（假设）** | 低秩 Q + K/V + 分组低秩 O 按字段名推断；compress_ratios / 稀疏注意力 KV 未建模 |
| 容量单位 | **已齐** | `capacity_GB` = 2³⁰ B（厂商标称），`*_decimal` = 10⁹ B |
| 吞吐–交互性帕累托 | **已齐（解析稳态）** | batch 1 → KV 容量上限 × 全部布局；CLI / API / Web / CSV |
| SLO goodput | **半齐（假设）** | v0.31 默认按输出长度 N 摊销 prefill（分块混合 / 独占），上界口径可选；仍无排队 / 到达过程 / PD 分离 |
| 视频 / 蛋白等效曲线 | **已齐（成批时延）** | 帧/s/芯片 vs TTFC；序列/s/芯片 vs 单批时间；算力受限时曲线主要区分布局 |

### v0.31 引擎：TP×EP / PP 微批 / prefill 摊销 / 投机解码 / LM head

| 项 | 状态 | 说明 |
|----|------|------|
| TP×EP 专家切分（tp_ep / ep_all） | **已齐** | 每卡容量手算核对；MoE ep>1 时注意力在 ep 组间按 DP 运行（KV 按 batch 切分）；deepseek-v3 ×8 的 28 个布局全部放得下 |
| MoE 路由 | **半齐（假设）** | 期望专家覆盖按均匀独立路由；无热点 / 负载不均 / 容量因子 / token drop |
| PP decode 微批 | **已齐（解析）** | max(mb, pp) · t_stage(微批)；stage 负载均衡假设；无微批调度开销 |
| prefill 摊销 goodput | **半齐（假设）** | 输出长度 N 固定；分块混合把整条 prompt 折入一个 tick（TTFT 偏乐观）；无排队 / PD 分离 |
| 投机解码 / MTP | **半齐（假设接受率）** | 逐位置 i.i.d. 接受；草稿 = 1 层 + head（MTP）或比例（独立模型）；MTP KV / eh_proj 读流 / 树形草稿未建模 |
| LM head | **已齐（工作台）** | 每步读 V·H + GEMM；注意力 TP 时按词表并行；PP 按 head 均衡（/pp）「假设」；原始引擎默认关闭 |


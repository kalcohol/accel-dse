# RELEASE 0.31.1 — accel_dse 冻结说明

> 0.31.1：仓库化（https://github.com/kalcohol/accel-dse），包名 `npu_dse` → `accel_dse`；引擎与数值同 0.31.0。

**版本**：0.31.0（引擎：TP×EP 专家切分 / EP 全卡；PP decode 微批填满流水；prefill 摊销 goodput（分块混合 | 独占，上界可选）；投机解码 / MTP；decode 计入 LM head；帕累托 batch 上限 65536）  
**日期**：2026-10-08（Asia/Shanghai）

## 这是什么

`accel_dse` 是给 **首硅 / 单卡 DSE** 用的 **手算可核对** 解析模型：Inference-only、NPU-like（固定 dataflow / MAC 阵列）+ 片上 SRAM 三分区 + 纯 HBM 或纯 LPDDR；覆盖 **LLM / Video DiT / Protein** 多域。  
透出 TTFT/TPOT（或 TTFC、time/seq）、compute vs mem 墙、scale_efficiency；默认路径 **不要求** 用户填 cache hit-rate。

**不是**：cycle-accurate RTL、GPGPU 模拟器、伪 PDK 功耗/面积、硅后标定声称。

## 怎么跑

```bash
git clone https://github.com/kalcohol/accel-dse.git && cd accel-dse
python3 -m accel_dse serve                  # Web：http://127.0.0.1:8765/
python3 -m accel_dse workbench --help       # CLI MetricsCard
python3 -m accel_dse list-series --product
python3 -m accel_dse list-packages
python3 -m accel_dse list-compute
python3 -m accel_dse list-presets
python3 -m accel_dse report --out out/report.html
python3 -m accel_dse pareto --model qwen3-32b --chips 8   # v0.30 吞吐–交互帕累托 + SLO goodput
python3 tests/run_tests.py
```

可选：`pip install 'accel-dse[web]'` 启用 FastAPI；默认也可用 stdlib HTTP。

## Honesty banner（摘要）

> All absolute bandwidth, efficiency, and frequency numbers are **assumed / uncalibrated** — not measured silicon.  
> MAC/TOPS 由 cores × tops_per_core @ assumed freq **derived**。  
> Energy/cost（`est_*`）是用户 knob 的 **ASSUMED stub**（`tdp_w` / `watts_per_tops` / `cost_per_card_usd`），**不是** 硅功耗。  
> Video/protein 多卡：`tp*pp*ep==chips`；EP 无 MoE 时为 no-op。

完整英文文案见 `accel_dse/serve.py` → `HONESTY_BANNER`；Web 顶栏与 API 响应均带 banner。

## Catalogs（设计空间档，非硅后）

| 目录 | CLI | 内容 |
|------|-----|------|
| **series** | `list-series --product` | 公开 HF dims packs（~58）+ illustrative/toy；gated 跳过 |
| **memory** | `list-packages [--kind LPDDR6 …]` · `GET /api/memory` | LPDDR5 / 5X x64 封装（4×16）· SOCAMM2 / LPCAMM2 模组 · LPDDR6 x96 封装（4×24，payload 8/9）· HBM3 / 3E / 4 / 4E 堆 × 层数 × die 密度；每项来源标签（最弱项） |
| **compute** | `list-compute` | core 4–16T · cluster 64–256T（assumed @1 GHz） |
| **presets** | `list-presets` | `edge-lpddr-4x64` · `edge-lpddr6-4x96` · `card-hbm-4stack` · `card-hbm3e-8x12h` · `server-socamm2-8` · `scaleup-8chip`（只捆 存储 + compute + chips，**不含**功耗/价格） |

## 刻意不做（out of scope）

- 伪 PDK / 工艺节点伪造面积功耗；DRAM pJ/bit、NoC、PUE、BOM/良率  
- Cycle-accurate Softmax/RoPE/LN / 时序 RTL（v0.26 仅 **opt-in assumed** 粗开销，默认 off）  
- 原生低精度 MAC 硅后曲线（默认 bytes-only；可选 `dtype_mac_factors` 为用户/assumed 表）  
- Host/NIC 非理想、完整 BI 仪表盘、MoE 不均路由、DiT temporal KV cache、Evoformer triangle 全量

## 仍需用户数字才能「敢报」绝对能效 / $

| 用途 | 需要用户提供 |
|------|----------------|
| 绝对功耗 / J/token·frame·seq | 实测或目标 **TDP（W/card）** 或 **W/TOPS**（+ 可选 `power_util`） |
| 系统 $ / LLM $/MTok | **cost_per_card_usd**、可选 mem add-on、`usd_per_kwh`、摊销年数 |

引擎 **默认关闭**（不设 knob → `est_*=0`）；`examples/energy_cost.example.json` 仅为假数占位。

## 本版要点（0.31.0）

- **TP×EP 专家切分**：
  - `moe_shard = tp_ep`（默认，专家按 ep 组、组内 TP 切分）或 `ep_all`（专家分布到全部 tp·ep 个 rank）。
  - MoE ep>1 时注意力在 ep 组间按 DP 运行，KV 按 batch 切分；all-to-all 按每 rank token 份额计；decode 只读期望命中的专家（「假设」均匀路由）。
  - **deepseek-v3 ×8 HBM3E 8×12H：可放下的布局 10/16 → 28/28**（含 12 个 EP 全卡布局）。
- **PP decode 微批**：
  - `decode_mb`（0 = min(B, pp)），TPOT_step = max(mb, pp) · t_stage(微批)；B = 1 时仍为 pp 级遍历时延。
  - 例：qwen3-32b TP1·PP8 B=64 时 75.6 → 37.5 ms。
- **prefill 摊销 goodput**（帕累托默认）：
  - 每请求输出 N（默认 256）。分块混合：每个 tick 折入 b·E/N 个 prompt，权重共享；独占：TPOT += B·t_prefill/N。
  - 上界（仅 decode，0.30 口径）可选。
  - qwen3-32b ×8 最优 goodput：上界 1471 → 摊销 **467** tok/s/芯片；deepseek-v3 ×8：上界 1153 → 摊销 **352**（独占 268）。
- **投机解码 / MTP**：
  - `spec_k` / `spec_accept`（默认 0.7「假设」）/ `spec_draft = mtp | model`；验证 M = B·(k+1)，E = (1−a^(k+1))/(1−a)，TPOT = 步时延 / E；开启时容量计入 MTP 层。
  - k=1 a=0.7 B=1：deepseek-v3 TP8 6.18 → **3.77 ms**，qwen3-32b TP8 5.07 → **3.09 ms**；算力受限的大批反而变慢。
- **LM head** 计入每个 decode 步（注意力 TP 时按词表并行；原始引擎默认关闭）；帕累托 batch 上限 4096 → 65536。deepseek 全部 28 个布局 ≈ 0.5 s。
- **UI**：
  - ⚙ 新增「投机解码 / MoE 切分 / PP 微批」；分解新增「草稿」条与「PP 遍历 / 气泡」；KPI TPOT 副标题显示步时延 / E。
  - 「吞吐 / 交互」新增 吞吐口径 / prefill 方式 / 输出长度。
- 测试 173 passed；`out/` 重新生成。

## 0.30.0 要点

- **吞吐–交互性帕累托**（`accel_dse/pareto.py`；CLI `pareto`；API `POST /api/pareto` / `/api/pareto.csv`；Web 标签页「吞吐 / 交互」）：
  batch 1 → KV 容量上限 × 全部有效并行布局（TP×PP×EP，注意力 TP | DP），tokens/s/芯片 vs tokens/s/用户，前沿高亮、被支配点变暗、悬停看配置；导出 CSV / JSON
- **SLO goodput**：TTFT ≤ X、TPOT ≤ Y（默认 2000 / 50 ms，可改）→ 最优吞吐、配置、最大并发、起作用约束（TTFT / TPOT / 容量）；
  KPI「SLO 吞吐 / 芯片」（当前布局）。稳态假设「假设」：TTFT_eff = 单条 prefill + 一个 decode 步，prefill 不与 decode 交织（DECISIONS D-0.30-3）
- **视频 / 蛋白**：等效曲线（帧/s/芯片 vs TTFC；序列/s/芯片 vs 单批时间）
- **MLA 注意力权重**按投影建模（DeepSeek-V3 每层 704.6M → 187.1M；总 734.3B → 670.9B ≈ 公开 671B；激活 37.4B）；含前置稠密层、非共享 LM head、MLA KV 含 RoPE；V4 低秩回退「假设」
- **容量单位**：`capacity_GB` / `capacity_needed_GB` = 厂商标称 GB（2³⁰ B），十进制见 `*_decimal`；LLM 卡片导出需求容量
- 扫描行短标签（原始 id 在 tooltip / CSV `raw_id`）；旧 `lpddr6_{n}x24` 按总线位宽映射（x96，否则推测 x48）；移动端存储配置为底部抽屉
- **参考场景**（HBM3E 8×12H 288 GB，默认算力，×8 卡 tp = 8，batch 1，上下文 512）：deepseek-v3 tp8 每卡存储权重 183.6 → **167.7 GB**，
  decode 每 token 每卡权重流 16.79 → **8.90 GB**，TPOT 10.536 → **6.049 ms**，每卡 KV 32.04 → 36.05 MB；qwen3-32b TPOT 不变 4.957 ms；
  qwen3-32b ×8 帕累托（默认 SLO）最优 **1494 tokens/s/芯片 @ TP8·PP1·EP1 B = 560**（TPOT 约束，每用户 21.3 tokens/s）
- 测试 161 passed；`out/` 重新生成（新增 `out/pareto_qwen3-32b_8.csv`）

## 0.29.0 要点

- **存储目录重建**（`accel_dse/mem_catalog.py`）：类型 / 形态 / 位宽 / 速率（WCK·CK）/ 数量 / 容量（HBM：层数 × die 密度 × 堆数）结构化选择；
  派生 总线 / 原始 / 可用（LPDDR6 ×8/9）/ 有效 / 容量（GB = 2³⁰ B）；来源标签取最弱项（JEDEC … 推测），UI 徽标显示；
  `research/memory_specs_2026-10.md` §5 列出的问题全部修正；旧 id（`?c=` / API / 预设）自动映射到最近新配置并附说明
- **修正 1 —— 暴露的通信同步**：`t_sync = 次数 × α × (1 − overlap)` 叠加在 max(…) 之外；α 默认 3 µs（假设），overlap 默认 0；LLM 与视频 / 蛋白质多卡
- **修正 2 —— KV 切分**：GQA ceil(n_kv/tp)/n_kv；MLA latent 每个 TP rank 全量；`attn_parallel = dp` 按 batch 切分
- **参考场景影响**（每卡 HBM3E 8×12H×24Gb @9200 = 288 GB、默认算力，×8 卡，tp = 8，batch 1，提示 / 上下文 512）：
  qwen3-32b TPOT 4.573 → **4.957 ms**（+0.384 ms = 128 次 × 3 µs；scale_efficiency 1.000 → 0.923）；
  deepseek-v3（MLA）每卡 KV 4.01 MB → **32.04 MB**（×8 复制），TPOT 10.170 → 10.536 ms
- UI：场景栏存储类型 + 摘要 + 弹层；「通信同步（暴露）」分解行；⚙「通信同步 / KV 切分」；新预设 3 个；扫描「存储」轴按 数量 / 速率 / 类型
- 测试 149 passed；`out/` 重新生成；zip 增加 `research/` 资料

## 0.28.1 要点

- 场景栏下拉框短标签 + 按内容自适应宽度（完整信息在 tooltip / 分组），1280 起所选值不截断；TP·PP·EP 摘要移到第 2 行
- 时间（µs / ms / s）与字节（KB … GB）显示自动缩放，悬停可见原始值；CSV / JSON / API 数值不变
- ≥1280 抽屉停靠在吸顶栏下方，场景栏保持两行；残留英文 / 不一致清理；视频分解标注为单步前向
- 仅 UI 层；无建模 / 引擎 / serve.py 改动

## 0.28.0 要点

- Web 工作台布局重构：吸顶两行场景栏（核心参数）+ 自动评估；7 项领域相关 KPI 条（含基线 Δ）；结果标签页（概览 / 对比·扫描 / A|B / 假设与说明）
- ⚙ 高级参数右侧抽屉（≥1280 非模态、KPI 保持可见）；≥1680 分屏；移动端场景摘要 + 编辑场景
- 🔗 复制链接（`?c=` 深链，含当前标签；旧链接兼容）；默认场景不 OOM；`balanced` → 均衡
- 测试版本断言改为读取 `accel_dse.__version__`；无新建模功能；API / JSON 字段 / CLI 输出不变

## 0.27.0 要点

- Web 工作台界面简体中文化（`<html lang="zh-CN">`、CJK 字体栈；专有名词 / 技术术语保留英文；假设值统一标记「假设」）
- 布局拥挤 / 文字重叠修复（表单标签列、复选框行、按钮行换行、指标卡、A|B 卡、对比 / 扫描表与柱状图标签）
- 无新建模功能；API / JSON 字段 / CLI 输出不变

## 0.26.0 要点

- Opt-in：`non_gemm_overhead` + `dtype_mac_factors`（默认 off / 1.0 → 与 0.25 默认路径 bit-identical）  
- 既有：scale_efficiency、energy/cost stub、presets、calib、dual A\|B、Compare/Sweep、`serve` UI（MVP+，非完整 BI）  
- 文档：`README` / `DECISIONS` / `REQUIREMENTS_GAP` / `CHANGELOG`；离线 `out/report.html`

详见 `CHANGELOG.md` · `DECISIONS.md` · `REQUIREMENTS_GAP.md`。

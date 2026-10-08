# npu_dse — NPU + SRAM + HBM/LPDDR 多域推理解析模型（LLM / Video DiT / Protein）

## 章程（Charter）

这是给 **首硅 / 单卡 DSE** 用的 **手算可核对** 解析模型，用来替代「网页式芯片模拟器」那种不可审计、不可复现中间量的做法。

**它是什么**

- Inference-only、NPU-like（固定 dataflow / 专用 MAC 阵列），**不是** GPGPU 模拟器。
- 单卡优先：片上 SRAM（**三分区**）+ **纯 HBM 或纯 LPDDR**（可选）。
- **Scale-up（v0.7–0.9）**：`tp/pp/ep∈{1,2,4,8}`；C2C；**IB/RoCE KV fabric**；**MoE EP** A2A；**MLA** 压缩 KV；`export-csv`。
- 透出 TTFT（prefill）、TPOT（decode）、带宽墙 / 利用率分解。
- 默认路径 **不要求用户填 cache hit rate**：外部流量由 **tiling + 分区容量** 推导。
- 显式给出 **M=1（decode）矩阵利用率曲线/公式**。
- Weight 读与 KV 读 **争用同一外部带宽**，并报告 serialize / fair-share / lower-bound。
- **SKU 峰值模板**（`sku_100t` / `sku_1p`）：用 PE×engines×freq×2flop/MAC 推出 TOPS，使 decode 在 LPDDR 上可观测到 mem-wall。
- **Dtype 灵敏度**（fp16/fp8/int8/int4）：只缩 **存储/流量字节**；MAC 峰值与 FLOPs **不变**（对低精度保守）。
- **上下文扫参**：decode ctx∈{512…128k}，观察 KV 流量何时逼近权重流。
- **SRAM resident knees**：扫 SRAM MiB，观察 R≥1 / R≥L/2 / R≥L 时 decode **W DRAM 字节下降**；staging≠resident。
- **独立 W/KV quant**：`w8k16`/`w4k16` 等，展示 weight-only 量化下长 ctx 时 KV 何时反超。
- **MoE 示意形状**（`illustrative_moe`）：总专家 E、top_k；流量按激活专家计（均衡路由 assumed）。
- **Batch 扫参**：decode `batch∈{1…32}`，OS util 随 M 上升；墙可能翻转。
- **FINDINGS.md**：硅侧中文备忘（权重流 / staging≠resident / SKU 墙 / KV 反超 / IP 含义）。
- **多域模板（v0.10–0.11）**：`illustrative_dit_video`（N_denoise×T² compute）、`illustrative_large_dit`（更大 T / 分钟级 TTFC 示意，**非** MiniMax-H3）、`illustrative_protein_pair`（L² pair 内存）；CLI `sweep-video` / `sweep-protein` / `compare-domains` / `list-shapes`。 **未标定**（BW/freq 仍未硅后标定；当前版本 **0.31.0**）。
- **Workbench 产品层（v0.12–0.14）**：`WorkbenchConfig` → `MetricsCard`；`chip_count`→tp（或显式 tp/pp/ep）；DRAM geometry knobs；`n_cores×tops_per_core`；CLI `workbench` / `workbench-scan` / `workbench-parallel`。
- **Package / compute 设计空间档位（v0.16）**：`list-packages` / `list-compute`；`--sweep-package` / `--sweep-compute`。
- **存储目录（v0.29 重建，`npu_dse/mem_catalog.py`）**：结构化选择 —— 类型 LPDDR5 / LPDDR5X / LPDDR6 / HBM3 / HBM3E / HBM4 / HBM4E；LPDDR 形态 板载封装 / SOCAMM2 / LPCAMM2；
  封装位宽（LPDDR5/5X **x64 = 4×16-bit 通道**；LPDDR6 **x96 = 4×24-bit 通道 = 8×12-bit 子通道**）；速率档（依附代际，UI 显示 WCK / CK MHz）；颗数 / 模组数；单颗容量；
  HBM 堆叠高度 8/12/16 × die 密度 × 堆数 1–12。派生：总线 = 数量 × 位宽；原始 = 总线 × MT/s / 8000；**LPDDR6 可用 = 原始 × 8/9**；有效 = 可用 × 效率（假设 0.70）；
  容量为厂商标称 GB（2³⁰ B）。每个组件带来源标签（JEDEC > 疑似 JEDEC > 厂商量产 > 送样 > 已发布 > 推测），组合取**最弱项**；允许推测组合但必标注。
  资料：`research/memory_specs_2026-10.md`。旧封装 id 自动映射（`GET /api/memory?resolve=<id>`）。
  CLI：`workbench --mem-type LPDDR6 --mem-count 4 --mem-rate 10667`、`--hbm-height 12 --hbm-die-gb 24`、`workbench-scan --package-axis type|count|rate`；API 结构化字段 `mem_type` / `mem_form` / `mem_width_bits` / `mem_rate_MTps` / `mem_count` / `mem_cap_GB` / `hbm_height` / `hbm_die_Gb`。
- **乐观假设修正（v0.29）**：① **暴露的通信同步** `t_sync = 次数 × α × (1 − overlap)` 叠加在 max(计算, 访存, C2C …) 之外（TP 每层 2 次 all-reduce、EP 每层 2 次 all-to-all、PP 每级 1 次；α 默认 3 µs「假设」，overlap 默认 0；`--c2c-latency-us` / `--sync-overlap`）；
  ② **KV 切分**：GQA 每卡 KV = KV × ceil(n_kv/tp)/n_kv；MLA latent 每个 TP rank 全量；`--attn-parallel dp` 改为按 batch 切 KV（注意力权重每卡复制）。α = 0 与 0.28 逐位一致。
- **吞吐–交互性帕累托 + SLO goodput（v0.30，`npu_dse/pareto.py`）**：对当前场景把 batch 从 1 扫到 **KV 容量上限**（每卡 权重 + KV ≤ 容量，KV 切分同评估器），覆盖芯片数允许的全部并行布局（TP×PP×EP，LLM 另含注意力 TP | DP），  得到 tokens/s/芯片（decode 吞吐）vs tokens/s/用户（1/TPOT）与帕累托前沿；SLO（TTFT ≤ 2000 ms、TPOT ≤ 50 ms 默认，可改）下报告最优吞吐、配置、最大并发与**起作用的约束**（TTFT / TPOT / 容量）。
  稳态假设「假设」：TTFT_eff = 单条 prefill + 一个 decode 步，prefill 不与 decode 交织，吞吐只计 decode（见 DECISIONS D-0.30-3）。视频 / 蛋白给出等效曲线（帧/s/芯片 vs TTFC；序列/s/芯片 vs 单批时间）。
  CLI `python3 -m npu_dse pareto --model qwen3-32b --chips 8 --csv out/pareto_qwen3-32b_8.csv`；API `POST /api/pareto`（`/api/pareto.csv`）；Web 标签页「吞吐 / 交互」（手写 SVG）+ KPI「SLO 吞吐 / 芯片」。
- **v0.31 引擎（`scaleup.py` / `pareto.py`，DECISIONS §20）**：
  - **TP×EP 专家切分**：`moe_shard = tp_ep`（专家按 ep 组划分、组内按 tp 切分；默认）或 `ep_all`（专家分布到全部 tp·ep 个 rank）。MoE ep>1 时注意力在 ep 组间按 DP 运行，KV 按 batch 切分；all-to-all 按每 rank 的 token 份额计；decode 只读期望命中的专家（「假设」均匀路由）。deepseek-v3 ×8 HBM3E 的 28 个布局全部放得下。
  - **PP decode 微批**：`decode_mb`（0 = min(B, pp)）让流水填满。TPOT_step = max(mb, pp) · t_stage(微批)，B = 1 时仍为 pp 级遍历时延。
  - **prefill 摊销 goodput**（默认）：每请求输出 N 个 token（默认 256），prefill 用「分块混合 | 独占」方式摊销。旧的「上界（仅 decode）」口径可选。CLI `--goodput-mode` / `--prefill-mode` / `--out-len`。
  - **投机解码 / MTP**：`spec_k` / `spec_accept`（默认 0.7「假设」）/ `spec_draft = mtp | model`。验证步 M = B·(k+1)，E = (1−a^(k+1))/(1−a)，TPOT = 步时延 / E。开启 MTP 时容量计入 DeepSeek MTP 层。
  - **LM head** 计入每个 decode 步（工作台；原始引擎默认关闭）；帕累托 batch 上限 65536。
- **MLA 权重 + 容量单位（v0.30）**：MLA 注意力按 q_lora / kv_lora / qk_nope / qk_rope / v_head 投影逐个建模（DeepSeek-V3 每层 187.1M，总 670.9B ≈ 公开 671B），含前置稠密层、非共享 LM head、MLA KV 含 RoPE；V4 无 kv_lora 走低秩回退「假设」。
  `capacity_GB` / `capacity_needed_GB` 统一为厂商标称 GB（2³⁰ B），十进制见 `*_decimal`。
- **Scaling efficiency（v0.25）**：MetricsCard `scale_efficiency` / `speedup` / `t_single_primary_ms`；primary=LLM TPOT / video TTFC / protein time_per_seq；`eff=(t_single/t_multi)/chips`（理想 1.0）；chips=1 → 1.0 + note `single-card`；诚实：不计 host/NIC；C2C 已在 t_multi。Web 卡片 / A|B Δ / sweep CSV / report `#scale`。
- **能效 / 成本 stub + 场景预设（v0.24，assumed）**：MetricsCard `est_power_W` / `est_energy_per_token_J`（llm）/ `est_energy_per_frame_J`（video）/ `est_energy_per_seq_J`（protein）/ `est_system_cost_usd` / LLM `est_usd_per_Mtok`；knobs `tdp_w` 或 `watts_per_tops`、`power_util`、`cost_per_card_usd`、`mem_addon_usd`、`usd_per_kwh`、`amortize_years`；**引擎默认关闭**，示例假数仅在 `examples/energy_cost.example.json`。预设 `edge-lpddr-4x64` / `edge-lpddr6-4x96` / `card-hbm-4stack` / `card-hbm3e-8x12h` / `server-socamm2-8` / `scaleup-8chip`（CLI `--preset` / `list-presets` / `eval-presets`；API `preset:`；Web 一键按钮）。**不是** PDK / JEDEC 功耗。
- **交互 Web 工作台（v0.17–0.28）**：`python3 -m npu_dse serve` — 本地 knobs 面板（series / chips / package / compute）→ live MetricsCard；**Compare/Sweep** + **dual card A|B**；**Assumed/override**（mem eff / mac-overlap / freq；override badge）；**calib**（CLI `--calib` / API `calib`；`examples/calibration.example.json`）；CSV/JSON export；baseline Δ%；`?c=` deep-link；Parallel×4/×8；零 CDN；**llm + video + protein**；v0.27 起界面为**简体中文**（专有名词 / 技术术语保留英文）；v0.28 布局：吸顶场景栏 + KPI 条 + 结果标签页 + ⚙ 高级参数抽屉 + 🔗 复制链接；v0.28.1：下拉框短标签不截断、时间 / 字节单位自动缩放；v0.29：存储类型 + 摘要按钮 + 存储配置弹层（来源徽标、原始 / 可用 / 有效带宽）、「通信同步（暴露）」分解行、⚙「通信同步 / KV 切分」；v0.30：「吞吐 / 交互」标签页（帕累托 SVG + SLO goodput）、KPI「SLO 吞吐 / 芯片」、扫描行短标签、移动端存储底部抽屉；v0.31：⚙「投机解码 / MoE 切分 / PP 微批」、分解「草稿」条与「PP 遍历 / 气泡」、帕累托「吞吐口径 / prefill 方式 / 输出长度」。
- **Video/protein multi-card（v0.19–0.20）**：`tp/pp/ep`（`tp*pp*ep==chips`）；TP ring/tree act collectives；PP bubble（denoise pipelines poorly）；EP no-op without MoE；assumptions banner。
- **真实公开 HF 系列维数（v0.15）**：`npu_dse/data/series_catalog.json` + `catalog.py`；`list-series --product` 列出数十个 HF-backed packs（GLM-5.3 / Kimi-K3 / Qwen3.8 / MiniMax-H3 / AF2…）；illustrative/toy 仍保留手算。Gated（Llama-4/Gemma/ESM-3）跳过。FLOPs/BW 仍 uncalibrated。

**它不是什么**

- 不是 cycle-accurate RTL / 时序仿真。
- **不**从 3–5nm 工艺节点伪造面积/功耗（`process_nm` 只是标签，留给后续）。
- 不做伪 PDK；C2C / IB / RoCE BW 与 latency 均为 assumed 预设（非 NIC 标定）。
- 不声称 FLOPs/BW 已硅后标定。公开 HF dims 的 packs 会在 metadata 中引用 `hf:<id>`；`illustrative_*` 仍是 placeholder。

---

## 假设与标签

| 量 | 状态 |
|----|------|
| PE 几何 R×C、n_engines、频率、mac_efficiency | assumed / uncalibrated |
| SKU 目标 TOPS（100T / 1P） | assumed 单卡峰值模板；1P 暂作更大单 die 灵敏度，非多芯片 |
| SRAM 总容量与三分区 | assumed；分区可由容量+形状 **derived** |
| HBM/LPDDR 通道数、位宽、速率 | assumed；**带宽由几何公式 derived** |
| efficiency（sustained/peak） | assumed |
| 模型维数 | 公开 HF packs：**derived from config.json/card**；`illustrative_*` / toy：placeholders |
| weight/KV/act bits（均匀或 wNkM） | assumed；**只改字节**，MAC rate 不变 |
| FLOPs、OS cycles、util、DRAM bytes、时间 | **derived** from above |

Dataflow 选择：**output-stationary (OS)** 脉动式阵列（见 `npu.py`）。

---

## 关键公式

### OS GEMM（含 multi-engine）

```
cycles = ceil(M/R) * ceil(N/(C * n_engines)) * K
util   = (M*N) / (R*C*n_engines * ceil(M/R)*ceil(N/(C*n_engines)))
peak_MAC/cycle = R * C * n_engines
peak_TOPS      = peak_MAC/cycle * freq_Hz * 2 / 1e12     # 2 flop/MAC
```

当 `N % (C*n_engines) == 0`：`util(M=1) = 1/R`；大 M 填满后 → 1。Decode 利用率低于大 M。
**绝对 decode 算力**仍 ∝ `C * n_engines * freq`（即便 util≈1/R）——这是 SKU 放大后能撞上带宽墙的原因。

### 外部带宽

```
peak_Bps = n_channels * (width_bits/8) * data_rate_GT/s * 1e9
eff_Bps  = peak_Bps * efficiency
```

### SRAM 三分区策略（assumed policy，无用户 hit-rate）

```
partitions: weight_partition + kv_scratch + act  ≤  capacity
R = floor(weight_partition / W_layer)   # resident layers across tokens

policy (CLI --sram-policy):
  weight_resident (default): maximize R = floor(cap / W_layer); rem → KV then act
  kv_first:                  satisfy KV working set first; rem → maximize R / staging
  balanced:                  ~50/50 weight vs KV after small act floor

When R==0 (cap < W_layer): staging-only fallback
  1. 若 2*W_layer ≤ 0.80 * capacity → staging=2*W_layer (dbl-buf eligible)
  2. 若 W_layer ≤ 0.80 * capacity → staging=W_layer
  3. 否则 staging=capacity/4 (tile)
```

**Honesty — staging vs resident (v0.5)**：
- **Staging**（R=0，哪怕装得下 1×/2× W_layer）**不减少** decode HBM 权重字节：每 token 仍读遍 L 层 → `W DRAM ≈ L * W_layer`。2× staging 仅使 `double_buffer_eligible`；可选 `weight_hide_factor`（assumed）只折 **mem 时间**，不折字节。
- **Resident**（R≥1）：R 层跨 token 常驻片上 → 稳态 decode `W DRAM ≈ (L−R)*W_layer`（R≥L 则 0）。Prefill / cold-start 仍装载整模一次。

**Resident knees（illustrative_27B @ fp16）**：W_layer≈1320 MiB，L=40 →
R≥1 ≈1320 MiB；R≥L/2 ≈26400 MiB；R≥L（full on-die）≈52800 MiB。**这些**才会让 W DRAM 下降。

### 权重路径（显式）

| 标签 | 含义 |
|------|------|
| `hbm_stream_all_layers` | R=0：每 token 流式读满 L 层（staging 或单次/层） |
| `hbm_stream_tiled` | R=0 且 tiles>1：按输出 tile **重载** 权重 |
| `hbm_stream_miss_layers` | 0<R<L：稳态只流式读 miss 层（L−R） |
| `on_die_resident` | R≥L：稳态 decode W DRAM=0（prefill 仍 cold-start 装载） |
| `sram_pinned_body` | `pin_weights` 且 body ≤ weight_partition |

`double_buffer_eligible` 为 **独立标志**（仅 staging R=0 且 staging≥2×W）。

### KV 路径

- Prefill：KV **写穿** 到外部（填充持久 cache）。
- Decode 读：若 `kv_scratch ≥ 本步 KV 工作集` → 片上命中（读 DRAM=0）；否则全量外读。新 token KV **始终写穿**。

### Prefill / Decode 流量要点

- Prefill：大 M=`B*S` GEMM；写 KV。
- Decode：M=`B`（常为 1）；读 KV（或片上命中）+ 写 1 token KV；与 weight 争用带宽。
- 争用：`serialize = (Bw+Bkv+Bact)/BW`；`share_fair = max(2 Bw, 2 Bkv)/BW + Bact/BW`；下界 `max(Bw,Bkv)/BW + Bact/BW`。

### 时间墙

```
t_compute = cycles / f_Hz          # f assumed
t_memory  = contention(mode)
t_phase   = max(t_compute, t_memory)   # roofline 重叠上界
TTFT = t_prefill;  TPOT = t_decode_step
```

**如何解读 compute vs mem wall**：看 `comp=` / `mem=` 分项。小 PE（`sku_baseline`≈8 TOPS）+ OS 的 util≈1/R 时，27B decode 常显示 compute-bound；换 `sku_100t` / `sku_1p` 后同一 LPDDR 流量下 TPOT 翻成 memory-bound（权重流 ≈55GB / ~381GB/s）。

---

## SKU 预设（单卡峰值模板）

| 名字 | 目标 | 几何（assumed） | @1GHz 达成 TOPS（derived） | 说明 |
|------|------|-----------------|---------------------------|------|
| `sku_baseline` | ~8 T | 64×64 × 1 eng | 8.192 | 历史默认小阵列 |
| `sku_100t` | 100 T | 56×56 × 16 eng | 100.35 | 单卡 ~100T 类 |
| `sku_1p` | 1000 T | 88×88 × 64 eng | 991.2 | **暂作更大单 die** 灵敏度；日后或映射多芯片（未实现） |

```
peak_TOPS = R * C * n_engines * freq_Hz * 2 / 1e12
```

`process_nm` 仅为标签，**不**推导功耗/面积。

---

## Dtype / Quant（存储/流量位宽，bytes-only）

| 均匀预设 | bits | 行为 |
|----------|------|------|
| `fp16` | 16 | 默认基线（W+KV） |
| `fp8` / `e4m3` | 8 | 权重+KV 字节减半 |
| `int8` | 8 | 同上（标签不同） |
| `int4` | 4 | 权重+KV 字节 ×0.25 |

| 独立 quant | W bits | KV bits | 用途 |
|------------|--------|---------|------|
| `w16k16` | 16 | 16 | 全精度基线 |
| `w8k16` / `w4k16` | 8 / 4 | 16 | **weight-only**；长 ctx 下 KV 更早反超 |
| `w8k8` / `w4k8` | 8 / 4 | 8 | 权重+KV 同降 |

**诚实约定**：只缩放 DRAM/SRAM **字节**；GEMM FLOPs 与阵列 peak MAC/s **保持不变**。真实低精度阵列往往更密/更快——本模型暂不声称，因此对低精度是 **保守** 上界（算力侧偏慢）。

---

## 目录

```
npu-inference-dse/
├── README.md
├── FINDINGS.md                # 硅侧中文备忘（assumed/uncalibrated）
├── DECISIONS.md               # 假设 vs 需拍板（均非阻塞）
├── REQUIREMENTS_GAP.md        # 原始需求对照 已齐/半齐/缺
├── CHANGELOG.md
├── pyproject.toml
├── examples/handcheck.md      # 全数字手算
├── npu_dse/
│   ├── model_shape.py         # dense + MoE shapes
│   ├── npu.py
│   ├── sku.py                 # 100T / 1P 峰值模板
│   ├── dtype.py               # fp16/fp8/int8/int4 存储位宽
│   ├── memory.py              # HBM/LPDDR + SRAM 分区
│   ├── traffic.py
│   ├── evaluate.py
│   ├── scan.py                # sweep-dtype/ctx/sram/quant/batch/moe
│   ├── series.py              # model series registry / product packs
│   ├── workbench.py           # WorkbenchConfig → MetricsCard
│   ├── package_ranges.py      # LPDDR/HBM package + compute catalogs
│   ├── report.py              # self-contained HTML report
│   ├── serve.py               # local web workbench API (stdlib / FastAPI)
│   ├── web/                   # static UI (index.html / app.js / style.css)
│   ├── scaleup.py / workloads.py / traffic.py
│   ├── cli.py
│   └── __main__.py
└── tests/test_conservation.py
```

---

## 如何运行

需要 Python ≥ 3.10，**零第三方依赖**（测试可选 pytest；Web UI 可选 `pip install 'npu-dse[web]'` 启用 FastAPI）。

```bash
cd /workspace/npu-inference-dse

# 默认扫描：toy + illustrative_27B × HBM/LPDDR，含 SRAM / PE / SKU
python3 -m npu_dse
# 或
python3 -m npu_dse.cli scan

# 100T / 1P SKU 扫描 + LPDDR 带宽墙 crossover 表
python3 -m npu_dse scan-sku

# Dtype / ctx / SRAM knee / independent quant 扫参
python3 -m npu_dse sweep-dtype
python3 -m npu_dse sweep-ctx
python3 -m npu_dse sweep-sram
python3 -m npu_dse sweep-sram --sram-policy kv_first
python3 -m npu_dse sweep-quant
python3 -m npu_dse sweep-batch
python3 -m npu_dse sweep-moe
# 或: python3 -m npu_dse sweep --what dtype|ctx|sram|quant|batch|moe
# SRAM + assumed hide (staging dbl-buf only): python3 -m npu_dse sweep-sram --weight-hide 0.5

# 列出 SKU / dtype / quant / series 预设
python3 -m npu_dse list-sku
python3 -m npu_dse list-dtype
python3 -m npu_dse list-quant
python3 -m npu_dse list-series --product

# 离线 HTML 报告（inline CSS，含 Metrics/wall）
python3 -m npu_dse report --out out/report.html

# 交互 Web 工作台 MVP（浏览器 knobs → live MetricsCard）
python3 -m npu_dse serve                  # http://127.0.0.1:8765/
python3 -m npu_dse serve --host 0.0.0.0 --port 8765
python3 -m npu_dse serve --stdlib         # force stdlib http.server
# UI: knobs → MetricsCard; Compare/Sweep → POST /api/sweep (chips|package|compute|parallel|series)
# v0.30 吞吐 / 交互 → POST /api/pareto (+ /api/pareto.csv)
# Parallel ×4 / ×8 presets = workbench-parallel matrix in-browser (≤32 rows)
# Export CSV/JSON from last sweep; Pin baseline → Δ% column; shareable ?c= deep-link restores knobs
# optional: pip install 'npu-dse[web]'  → FastAPI + uvicorn

# Workbench（统一配置 → MetricsCard）
python3 -m npu_dse workbench --model illustrative_27B --chips 4 --mem hbm --cores 16 --tops-per-core 6.25
python3 -m npu_dse workbench --model illustrative_27B --chips 1 --json out/card.json --md out/card.md
python3 -m npu_dse workbench --help
python3 -m npu_dse workbench-scan --model series/dense-27b --chips-list 1 2 4 8
python3 -m npu_dse workbench-scan --model illustrative_27B --csv out/workbench_scan.csv
python3 -m npu_dse workbench-parallel --model illustrative_27B --chips 8
python3 -m npu_dse workbench-parallel --model series/moe-active13b --chips 8 --csv out/parallel_moe_8.csv
# v0.30 吞吐–交互帕累托 + SLO goodput（全部布局 × batch 至 KV 容量上限）
python3 -m npu_dse pareto --model qwen3-32b --chips 8 --mem-type HBM3E --mem-count 8 --slo-ttft-ms 2000 --slo-tpot-ms 50 --csv out/pareto_qwen3-32b_8.csv
# v0.31 prefill 摊销口径（默认）/ 上界；投机解码 MTP k=1；MoE 切分
python3 -m npu_dse pareto --model deepseek-v3 --chips 8 --mem-type HBM3E --mem-count 8 --goodput-mode amortized --prefill-mode chunked --out-len 256
python3 -m npu_dse pareto --model deepseek-v3 --chips 8 --mem-type HBM3E --mem-count 8 --goodput-mode upper --spec-k 1 --spec-accept 0.7
python3 -m npu_dse workbench --model deepseek-v3 --chips 8 --mem-type HBM3E --mem-count 8 --spec-k 1 --moe-shard ep_all --pp-decode-mb 0
python3 -m npu_dse workbench-scan --mode parallel --chips 8 --model series/dense-27b
python3 -m npu_dse workbench-scan --sweep-mem --mem hbm --chips 1 --mem-vary channels
python3 -m npu_dse list-packages
python3 -m npu_dse list-compute
python3 -m npu_dse workbench-scan --sweep-package --model glm-5.3-flash --chips 1
python3 -m npu_dse workbench-scan --sweep-compute --compute-level core --mem lpddr --chips 1
python3 -m npu_dse list-series --product

# 单点评估（可用 --sku / --dtype / --weight-bits）
python3 -m npu_dse eval --shape 27b --mem lpddr --sku sku_100t --prompt 512 --ctx 512
python3 -m npu_dse eval --shape 27b --mem lpddr --sku sku_100t --dtype int4 --ctx 32768
python3 -m npu_dse eval --shape 27b --mem lpddr --sku sku_100t --quant w4k16 --ctx 128000 --sram-mib 2048
python3 -m npu_dse eval --shape moe --mem lpddr --sku sku_100t --ctx 512
python3 -m npu_dse eval --shape 27b --mem lpddr --pe 64 64 --engines 16 --sram-mib 64

# M=1 利用率曲线
python3 -m npu_dse util-curve --pe 64 64 --engines 1 --k 512 --n 512

# 与 examples/handcheck.md 对齐的玩具配置
python3 -m npu_dse handcheck

# 测试（零依赖 runner；也可 pip install pytest）
python3 tests/run_tests.py
# 或: python3 -m pytest tests/ -q
```

手算示例见 [`examples/handcheck.md`](examples/handcheck.md)；与 `test_toy_handcheck_numbers` 对齐。

---

### Tensor-parallel + C2C（v0.7，assumed）

```
sharding (Megatron-style):
  Q/K/V/gate/up  column-parallel (shard N)
  O / down       row-parallel   (shard K → all-reduce)

per layer: 2 all-reduces on volume V = B * S * H * act_bytes
  (decode: S=1; prefill: S=prompt_len)

ring all-reduce bytes/rank = 2 * (tp-1) / tp * V
phase C2C bytes = (L/pp) * 2 * ring_bytes   # PP stages own L/pp layers

per-card FLOPs ≈ total / (tp*pp*ep)   # MoE EP splits expert work
per-card W DRAM ≈ EP-aware stream (attn/tp + top_k/ep FFN) / pp
per-card KV ≈ total_KV / (tp*pp)      # EP does not shard KV
embed capacity: replicate (default) | shard

t_c2c = C2C_bytes / effective_BW * (1 - c2c_hide)   # hide default 0
t_phase ≈ max(t_compute_card, t_dram_card, t_c2c, t_pp_act, t_fabric)

OOM if (W_body/(tp*pp) + embed_(full|/tp) + KV/(tp*pp)) > mem.capacity
```

CLI: `python -m npu_dse sweep-tp` / `scan-scaleup`。

### KV fabric IB/RoCE（v0.8 MVP，assumed）

```
modes: none (default) | roce_v2 | ib
presets: {roce,ib}_{100,200,400}g
  effective_BW_GBps = (Gbps/8) * 0.80     # assumed
  latency_us        = 5 (RoCE) | 2 (IB)   # assumed per message

remote KV decode:
  remote_bytes = remote_kv_frac * KV_read
  t_fabric = latency + remote_bytes / BW
  local DRAM KV = (1-frac)*KV_read + KV_write
  wall = max(compute, local_dram, c2c, pp_act, fabric)

PD disagg one-shot:
  t_kv_xfer = latency + full_KV_bytes / BW   # separate metric (optional +TTFT)
```

CLI: `python -m npu_dse sweep-kv-fabric`（sku_100t dense × ctx × fabric presets）。

### Pipeline parallel（v0.8 light，assumed）

```
pp ∈ {1,2,4,8};  cards = tp * pp * ep  (DP=1)
activation send ≈ (pp-1) * B * S * H * act_bytes   # via C2C or pp_link

bubble fraction = (pp-1) / (mb + pp - 1)
  decode mb=1 → bubble = (pp-1)/pp   # often bubble-heavy — documented limit
  prefill may use mb ≥ pp to reduce bubbles crudely

wall = stage_time / (1 - bubble)
  stage_time = max(compute, dram, c2c, pp_act, fabric)
tp=1,pp=1 matches single-card
```

CLI: `python -m npu_dse sweep-parallel --tp-list … --pp-list …` / `sweep-pp`。

---

### MoE expert-parallel（v0.9，assumed）

```
ep ∈ {1,2,4,8};  cards = tp * pp * ep  (DP=1);  E % ep == 0
stored experts / rank = E/ep
active W stream / rank = attn/tp + shared + (top_k/ep)*FFN

A2A dispatch+combine (Megatron-MoE style, documented):
  V = B * S * H * act_bytes
  bytes = 2 * (ep-1)/ep * V * (top_k / E_local_adjust) * layers
  default E_local_adjust = 1
t_a2a = a2a_bytes / C2C_BW * (1 - c2c_hide)
wall = max(compute, dram, c2c, pp_act, a2a, fabric)
```

CLI: `python -m npu_dse sweep-moe --ep 1 2 4 8`。

### MLA 压缩 KV（v0.9，assumed）

```
illustrative_mla: same body as 27B; kv_lora_rank=512 (joint latent)
kv_bytes/token = L * kv_lora_rank * kv_bits/8     # vs GQA 2*L*n_kv*d*bytes
→ ~0.25× KV vs GQA at same L/bits (40 KiB vs 160 KiB / token)
```

CLI: `python -m npu_dse sweep-mla`（含 @128k 远程 fabric 对比）。

### CSV 导出

```
python3 -m npu_dse export-csv --out /workspace/npu-inference-dse/out/
# → sweep_{sku,dtype,ctx,sram,tp,kv_fabric,moe,mla}.csv
```

---

## 内置形状

- **toy**：L=2,H=64,… 专供计算器核对。
- **illustrative_27B**：约 **28.7B** 参数（由维数 derived）的 **示意形状**，不是某厂商权重声明。
- **illustrative_moe**：E=8、top_k=2、稠密 GQA 注意力；总参 ≈45.5B / 激活 ≈13.0B（均衡路由 assumed）。
  每 token 流式读 **attn + top_k 专家 FFN**（非全 E）；`--ep` 启用专家并行。
- **illustrative_mla**：与 27B 同 body，`kv_lora_rank=512` 联合压缩 KV（~0.25× GQA KV/tok）。

硅侧结论摘要见 [`FINDINGS.md`](FINDINGS.md)。

---


---

## Workbench（v0.12–0.14 产品层）

统一入口：`WorkbenchConfig` → `evaluate_workbench(cfg)` → `MetricsCard`。

| 旋钮 | 含义 | 默认 / 映射 |
|------|------|-------------|
| `model_id` | series / shape id | `illustrative_27B`；也可用 `series/dense-27b` 等 |
| `chip_count` | 卡数 ≥1 | 默认 `tp=chip_count, pp=ep=1`；显式 `{tp,pp,ep}` 须 `tp*pp*ep==chips`（llm **与** video/protein） |
| `mem_kind` + geometry | HBM|LPDDR + `n_packages`/`n_ranks`/`n_channels`/`data_rate_GTs`/`width_bits`/`efficiency`/`capacity_GB` | 缺省用 preset；带宽 **derived** |
| `n_cores` × `tops_per_core` | 产品算力语言 | **assumed**：`n_engines=n_cores`，每核近方形 PE 使 peak≈T/core；也可直接传 PE rows/cols/engines 或 `--sku` |
| workload | prompt/ctx/batch/dtype bits | 512/512/1/fp16 |
| optional | `sram_mib`, `kv_fabric`, C2C | 64 MiB / none / 400 GB/s |

`MetricsCard` 稳定字段：`domain`, `TTFT_ms`/`TPOT_ms`（llm）, `TTFC_ms`/`frames_per_s`（video）, `time_per_seq_ms`/`pair_bytes`（protein）, `wall`, breakdown, `bytes W/KV/coll`, `util`, `oom`, `scale_efficiency`/`speedup`, `non_gemm_overhead`/`dtype_mac_factor`, `chips`, `peak_tops`, `mem_eff_GBps`, `capacity_*`, `notes`/`assumptions`。

物理仍走 `evaluate_scaleup` / traffic — **不 fork**。chips=1 与既有单卡卡对齐（容差内）。

**CalibrationOverrides（可选、非阻塞）**：`examples/calibration.example.json` 可注入 `mem_efficiency` / `mac_efficiency`（或 `weight_hide`）/ `frequency_hz`（或 `freq_ghz`）。
CLI：`python3 -m npu_dse workbench --calib examples/calibration.example.json`；
API：`POST /api/eval` body 含 `calib: {…}`（亦接受顶层 `efficiency` / `freq_ghz` / `mac_efficiency`）。
Web：Assumed / override 滑条；≠默认显示 **override active**；写入 `?c=`。

**Energy / cost stub（v0.24，ASSUMED — 不是硅 / PDK / JEDEC 功耗）**

| knob | 含义 | 默认 |
|------|------|------|
| `tdp_w` | 每卡功耗基数 (W)；与下者同时给时 **优先** | None（off） |
| `watts_per_tops` | 每卡 W / peak TOPS | None（off） |
| `power_util` | 平均/峰值功耗比 | 1.0（= TDP 上界） |
| `cost_per_card_usd` / `mem_addon_usd` | 每卡价格 / 每卡内存封装加价 | None（off） |
| `usd_per_kwh` / `amortize_years` / `duty_cycle` | LLM `$ / MTok`（电费 + capex 摊销） | None / None / 1.0 |

公式：`est_power_W = chips × P_card × power_util`（仅卡，不含 host/冷却/PUE）；
llm `est_energy_per_token_J = P × TPOT / batch`；video `est_energy_per_frame_J = P × TTFC / (n_frames × batch)`；
protein `est_energy_per_seq_J = P × t_seq / batch`；`est_system_cost_usd = chips × (cost_per_card + mem_addon)`；
LLM `est_usd_per_Mtok = E_tok·1e6/3.6e6·$/kWh + cost / (tok/s · years · 365·86400 · duty) · 1e6`（decode-only）。
不设 knob → 所有 `est_*` 为 0、`econ_configured=false`。

```bash
python3 -m npu_dse workbench --preset card-hbm-4stack --dtype int8 --tdp-w 400 --power-util 0.6 --cost-per-card 10000
python3 -m npu_dse workbench --model illustrative_27B --econ examples/energy_cost.example.json   # EXAMPLE 假数
python3 -m npu_dse list-presets
python3 -m npu_dse eval-presets --model illustrative_27B --econ examples/energy_cost.example.json
curl -s localhost:8765/api/presets
curl -s -XPOST localhost:8765/api/eval -d '{"model_id":"illustrative_27B","preset":"scaleup-8chip","econ":{"tdp_w":400,"cost_per_card_usd":10000}}'
```

**Scenario presets**（v0.29 更新为真实可量产 / 已发布组合）：`edge-lpddr-4x64`（LPDDR5X 4×x64 @8533，64 GB + cluster_64t ×1）、
`edge-lpddr6-4x96`（LPDDR6 4×x96 @10667，64 GB + cluster_128t ×1）、`card-hbm-4stack`（HBM3E 4×12H×24Gb @9200，144 GB + cluster_256t ×1）、
`card-hbm3e-8x12h`（HBM3E 8×12H，288 GB + cluster_256t ×1）、`server-socamm2-8`（SOCAMM2 8×192 GB @9600 + cluster_256t ×1）、
`scaleup-8chip`（HBM3E 4×12H 卡 ×8，tp=8）。只捆 存储 + compute + chips，不含功耗/价格；显式 CLI flag / API 键优先。

Series packs：

- **HF-backed（v0.15）**：`python3 -m npu_dse list-series --product` — 如 `glm-5.3`、`kimi-k3`、`qwen3.8-2.4t`、`minimax-h3`、`alphafold2`…；metadata 引用 `hf:<org/model>` + source URL。
- **Illustrative（手算/回归）**：`series/dense-27b` → `illustrative_27B`；`series/moe-active13b`；`series/mla-27b`；`series/dit-video` / `series/dit-large` / `series/protein-pair`。

## 开放假设 / 后续（需用户拍板的点）

完整清单与「均非阻塞」说明见 [`DECISIONS.md`](DECISIONS.md)。摘要：

- 频率、效率、SRAM MiB、PE / engine 数未校准到硅后数据（**非阻塞**，可扫）。
- Dtype 为存储/流量位宽；默认 **不**声称低精度 MAC 原生吞吐（factor=1.0 = 对低精度保守）；可选 `dtype_mac_factors` 为用户/assumed 表（非硅）。
- Prefill 注意力用「矩形 × 1/2」近似因果；Softmax / RoPE / LN 默认忽略，v0.26 起可选 `non_gemm_overhead`（assumed，非 cycle-accurate）。
- KV 片上命中全有或全无；激活 spill 保守。
- 通信同步 α（默认 3 µs / 次）未标定；不区分拓扑 / 消息大小。注意力 DP 的计算与重分片为近似；MLA 模型注意力权重按 dense MHA 形状建模。
- 存储来源标签基于 2026-10 资料（`research/`），新产品发布后需人工更新。
- `illustrative_*` 仍为 placeholder；公开 HF dims 已入库（见 `list-series --product`）。Hybrid attn / Engram / DSA 未完整建模（metadata only）。
- `sku_1p` 暂作更大单 die；TP/C2C 为独立 scale-up 旋钮（assumed）。
- 功耗/面积：刻意不做伪 PDK；v0.24 只有用户 knob 的 energy/cost **assumed stub**（REQUIREMENTS_GAP #19/#20 = 半齐）。

**版本**：0.31.0（TP×EP 专家切分 + EP 全卡；PP decode 微批；prefill 摊销 goodput；投机解码 / MTP；decode 计入 LM head；0.30：吞吐–交互性帕累托 + SLO goodput；MLA 注意力按投影建模（DeepSeek-V3 ≈ 671B）；容量统一 2³⁰ B；存储目录按 JEDEC / 厂商资料重建：结构化选择器 + 来源标签 + LPDDR6 payload 8/9；暴露的通信同步 α；KV 按 TP 切分 / MLA 复制 / 注意力 DP；0.28.1 Web 工作台打磨：下拉框短标签 + 单位自动缩放；Web 工作台布局重构：场景栏 / KPI 条 / 标签页 / 高级参数抽屉；Web UI 简体中文化 + 布局拥挤修复；opt-in `non_gemm_overhead` + `dtype_mac_factors` assumed；scale_efficiency/speedup；energy/cost assumed stub + scenario presets；calib + dual A|B；Compare/Sweep；multi-domain MetricsCard + HF packs + Workbench / `report`；MoE EP / MLA / scale-up；无 PDK；BW/freq 未硅后标定）。

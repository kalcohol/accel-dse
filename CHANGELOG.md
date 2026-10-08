## 0.31.1 — 2026-10-08

**仓库化，包名改为 `accel_dse`**（引擎与数值同 0.31.0，无行为变化）。

- 项目改名 `npu-inference-dse` → `accel-dse`，发布到 GitHub：https://github.com/kalcohol/accel-dse
- Python 包 `npu_dse` → `accel_dse`（`python3 -m accel_dse …`）；发行名 / 命令行入口 `npu-dse` → `accel-dse`；不保留旧包名兼容层。
- README 顶部新增中英简介、快速开始、「数值均为解析 / 假设、非实测硅片」免责声明与 Issues 反馈入口；文档中的绝对路径改为相对路径。
- 新增 `.gitignore`：生成物 `out/*` 默认忽略，仅保留文档引用的示例输出与 `out/hf_series_lookup.json`（`scripts/bake_series_catalog.py` 的输入）。
- 以下历史条目保留原文（其中的 `npu_dse` / `npu-inference-dse` 指旧名）。

## 0.31.0 — 2026-10-08

**引擎版本（UI 仅在必要处）**：修正 0.30 报告的剩余问题 —— TP×EP 专家切分、PP decode 流水填满、goodput 计入 prefill、投机解码 / MTP、decode 计入 LM head、批量搜索上限。

### A. TP×EP 专家切分（`scaleup.py`；DECISIONS D-0.31-1）
- MoE 新增 `moe_shard`：
  - `tp_ep`（默认）：专家按 ep 组划分，每个专家在组内再按 tp 切分（tp·pp·ep = 芯片数）。
  - `ep_all`：专家分布到全部 tp·ep 个 rank、不切分（「EP 全卡、注意力 TP/DP」常见部署）。
- 每卡权重 = 注意力 / t_a + 共享专家 / tp + 本地路由专家 · P / 专家 TP + 前置稠密 FFN / tp（按层平均）；路由专家在 8 卡上只存一份。
- MoE ep>1 时注意力在 ep 组间按数据并行（DeepSeek「注意力 DP + 专家 EP」）→ KV 按 batch 在 ep 上切分（MLA latent 仍在 TP 组内复制）。
- all-to-all 每 rank 字节 = 2·(D−1)/D·(tokens/D)·top_k·H·act，D = 专家并行度（tp_ep → ep，ep_all → tp·ep）；0.30 按整批计，偏高 ×D。
- decode 权重流：每卡读到的路由专家数取期望值 n_local·(1−(1−k/E)^tokens)（「假设」均匀路由）；单 token = k/E 比例，大批 → 全部本地专家。
- **deepseek-v3 ×8（HBM3E 8×12H 288 GB）**：可放下的布局 **10/16 → 28/28**（16 个 tp_ep + 12 个 ep_all）。0.30 中 6 个 tp>1·ep>1 布局（TP2·EP4、TP2·PP2·EP2、TP4·EP2，含注意力 DP）因专家未按 TP 切分而超容量。
- 帕累托布局：MoE 且 tp>1、tp·ep | E 时额外枚举「EP 全卡」（标签 `· EP全卡`）；不放下的布局给出 `why_not`。

### B. PP decode 流水填满（D-0.31-2）
- batch>1 时 decode 用 mb = min(B, pp) 个微批（新 knob `decode_mb`，0 = 自动；1 = 0.30 行为）。
- 每个微批 ceil(B/mb)：TPOT_step = max(mb, pp) · (t_stage(微批) + t_draft)；气泡 = 1 − mb/max(mb, pp)。
- 单请求时延语义不变（B=1：pp 级遍历）。稳态吞吐不再是 1/pp。例：qwen3-32b TP1·PP8 B=64，TPOT 75.59 → 37.55 ms（= B=1 遍历时延）。
- 分解中「气泡」改名「PP 遍历 / 气泡」。

### C. goodput 计入 prefill（`pareto.py`；D-0.31-3）
- 帕累托标签页新增「吞吐口径」（`#goodput-mode`）：
  - **摊销 prefill**（默认）：每个请求输出 N 个 token（`#out-len`，默认 256），其 prefill 摊到 N 个 token 上。
  - **上界（仅 decode）**：即 0.30 口径，仍可选。
- prefill 方式（`#prefill-mode`）：
  - **分块混合**（默认）：每个 decode tick 折入 r = 微批·E/N 个 prompt，权重只读一次，按分量相加后取 max。TTFT ≈ pp · tick(含整条 prompt) + tick'。
  - **独占**：TPOT = step/E + B · t_prefill(单条)/N，TTFT = 单条 prefill + 一个 step。
- 各点新增 `TPOT_decode_ms` / `TPOT_step_ms` / `prefill_share` / `decode_mb` / `spec_tokens_per_step` / `moe_shard`；结果新增 `serving`；均标「假设」。
- CLI `pareto --goodput-mode amortized|upper --prefill-mode chunked|exclusive --out-len N`；API 同名字段。

### D. 投机解码 / MTP（D-0.31-4）
- 新 knob：`spec_k`（草稿 token 数，0 = 关）、`spec_accept`（接受率，默认 0.7「假设」）、`spec_draft = mtp | model`、`spec_draft_frac`。
- 验证步一次处理 k+1 个 token（M = B·(k+1)），KV 上下文每序列读一次、写 k+1 个。
- 每步期望 token E = (1−a^(k+1))/(1−a)，TPOT = 步时延 / E。
- 草稿开销：
  - mtp：k × (1 层 + LM head) 前向；
  - model：k · frac · t_stage。
- 开启 MTP 时，容量计入 DeepSeek MTP 模块（1 层 + eh_proj 2H²，层数取 HF `num_nextn_predict_layers`，目录已重新烘焙）。
- Web：⚙ 新增「投机解码 / MoE 切分 / PP 微批」；KPI TPOT 副标题显示步时延 / E；分解新增「草稿」条；帕累托提示框显示每步 token。
- 例（×8 HBM3E，k=1，a=0.7，B=1）：

  | 模型 / 布局 | TPOT（ms） | 变化 |
  |---|---|---|
  | deepseek-v3 TP8 | 6.179 → **3.772** | −39% |
  | deepseek-v3 TP1·EP8 | 6.179 → 4.203 | |
  | qwen3-32b TP8 | 5.065 → **3.090** | −39%；草稿 model 10% 时为 3.278 |

  算力受限的大批反而变慢：deepseek-v3 EP8 B=256 时 30.39 → 36.21 ms。

### E. LM head 与批量上限（D-0.31-5）
- 工作台（产品路径）每个 decode 步计入 LM head：读 V·H 字节 + GEMM (M, H, V)。
  - 注意力 TP：按词表并行（V/tp），与存储策略无关。
  - 注意力 DP：复制存储的 head 每 rank 全读。
  - PP：假设按 head 均衡切分（/pp）。
- 原始引擎 `EvalConfig.count_lm_head` 默认关闭（手算测试不变）。
- 例：deepseek-v3 TP8 B=1 TPOT 6.049 → 6.179 ms；qwen3-32b 4.957 → 5.065 ms。
- 帕累托 batch 上限 4096 → **65536**（`LLM_BATCH_LIMIT`；容量二分 + 稀疏网格）。deepseek-v3 全布局（28 个，约 950 次评估）≈ **0.5 s**。

### 参考场景（HBM3E 8×12H，×8，默认算力，SLO TTFT 2 s / TPOT 50 ms，P = 512，上下文 512）

| 场景 | 0.30 | 0.31 上界 | 0.31 摊销（分块混合） | 0.31 摊销（独占） |
|---|---|---|---|---|
| qwen3-32b 最优 goodput（tok/s/芯片） | 1494 @ TP8 B=560 | 1471 @ TP1·PP8 B=448 | **467** @ TP8 B=168 | 465 |
| deepseek-v3 最优 goodput | 1179 @ EP8 B=448 | 1153 @ EP8 B=448 | **352** @ EP8 B=96 | 268 @ TP8 B=107 |

- 上界口径下降约 2%，来自 LM head；PP8 因流水填满而与 TP8 相当（0.30 中 PP8 仅为 1/8）。
- 摊销口径约为上界的 1/3：P = 512 / N = 256 相当于每产出 1 个 token 还要处理 2 个 prompt token，而该区间受算力限制。

### 测试
- 173 passed（新增 12 项 v0.31 手算：TP×EP 容量、KV 在 ep 上切分、a2a 公式、期望专家数、PP 微批、投机公式 / 验证 M = k+1 / MTP 容量、LM head、摊销口径手算、批量上限 + deepseek 全布局 < 3 s、API / CLI、UI 标记）。
- 0.30 关系测试固定在上界口径；deepseek「有布局放不下」断言改为「全部放下」。

## 0.30.0 — 2026-10-08

**引擎 + UI 版本**：修正 0.29 遗留问题（MLA 权重、容量单位、扫描标签、旧 LPDDR6 链接、移动端弹层），新增「吞吐 / 交互」帕累托与 SLO goodput 分析。
依据：`research/related_tools_2026-10.md` 想法 #1（吞吐–交互性帕累托）与 #4（SLO goodput）。

### A. 0.29 遗留修正
- **A1 MLA 注意力权重**（`model_shape.py` / `catalog.py` / `scaleup.py` / `traffic.py` / `scripts/bake_series_catalog.py`）：
  按 HF config 的 `q_lora_rank` / `kv_lora_rank` / `qk_nope_head_dim` / `qk_rope_head_dim` / `v_head_dim` 逐个投影建模
  （q_a: H→q_lora，q_b: q_lora→nh·(nope+rope)，kv_a: H→kv_lora+rope，kv_b: kv_lora→nh·(nope+v)，o: nh·v→H）。
  DeepSeek-V3 每层注意力参数 **704.6M → 187.1M**；缺 `kv_lora_rank` 的 DeepSeek-V4 配置走「低秩回退」（低秩 Q、K/V = H→n_kv·head_dim、分组低秩 O，「假设」）。
  同时补齐：**前置稠密层**（HF `first_k_dense_replace`，DeepSeek 前 3 层为 18432 稠密 FFN）、**非共享 LM head**（`tie_word_embeddings=false` → 嵌入 ×2）、
  **MLA KV 含 RoPE 部分**（每层 512 → 576 元素）。手算核对：DeepSeek-V3 总参数 **670.9B**（公开 ≈ 671B，误差 < 1%）、激活 **37.4B**（公开 ≈ 37B）。
  影响所有权重字节（TP 与注意力 DP 均适用）；示意（illustrative）形状不变。
- **A2 容量单位**：API / CSV / 卡片的 `capacity_GB`、`capacity_needed_GB` 改为**厂商标称 GB（= 2³⁰ B）**，与 UI、存储目录一致；
  十进制值另给 `capacity_GB_decimal` / `capacity_needed_GB_decimal`。手动几何的「容量 GB」也按 2³⁰ B 解释。
  LLM 卡片新增导出每卡需求容量（权重 + KV；此前仅视频 / 蛋白有），容量 KPI 显示「需 X / 可用 Y GB」。
- **A3 扫描行短标签**：存储扫描（目录精选 / 结构化维度）行标签改为短名（如 `HBM3E 8×12H×24Gb @9200 · 288 GB`），原始 id 放在 tooltip 与 CSV 新列 `raw_id`（列追加在末尾，原列顺序不变）。
- **A4 旧 LPDDR6 链接** `lpddr6_{n}x24_*`：按总线位宽保留解析 —— n×24 可被 96 整除 → n×24/96 个 x96 封装；否则用 x48（推测）封装
  （等效于若干 x96 + 1 个 x48，带宽与容量/位均相同）；奇数 die 数取最近位宽（并列取窄，偏保守）。例：`lpddr6_20x24_14400` 4×x96 → **5×x96（480-bit，完全保留）**；
  `lpddr6_6x24_*` 1×x96 → **3×x48（144-bit）**。说明写入 legacy_note。
- **A5 移动端（≤768 px）存储配置弹层 → 底部抽屉**：固定在屏幕底部、全宽、最高 72vh、顶部拖动条、标题栏（含「完成 ✓」）吸顶、半透明遮罩点按关闭；不再遮挡存储芯片按钮。

### B. 新分析：吞吐 / 交互（`npu_dse/pareto.py` 新增）
- **B1 吞吐–交互性帕累托**：对当前场景，在芯片数允许的所有并行布局（TP×PP×EP = 芯片数；MoE 才枚举 EP；LLM 另枚举注意力 TP | DP）上，
  把 batch 从 1 扫到 **KV 容量上限**（每卡容量 − 权重，KV 切分规则与评估器一致：GQA 头切分 / MLA latent 复制 / 注意力 DP 按 batch 切分 / PP 按层），
  计算 **tokens/s/芯片**（decode 吞吐 = B × 1000 / TPOT / 芯片数）与 **tokens/s/用户**（= 1000 / TPOT），标出帕累托前沿。
  - Web：新标签页「吞吐 / 交互」—— 手写 SVG（零依赖）对数散点图，前沿连线、被支配点变暗、违反 SLO 为空心、SLO 最优点金色圈、当前场景虚线框；
    悬停 / 点按显示配置（布局、B、TPOT、TTFT、容量、瓶颈）；颜色 = 并行布局；各布局表（容量上限 B / 单条 prefill / SLO 最大并发 / SLO 吞吐 / 约束）；导出 CSV / JSON
  - CLI：`python3 -m npu_dse pareto --model qwen3-32b --chips 8 [--mem-type HBM3E --mem-count 8] [--slo-ttft-ms 2000 --slo-tpot-ms 50] [--layouts all|current] --csv … --json …`
  - API：`POST /api/pareto`（body = `/api/eval` body + `slo_ttft_ms` / `slo_tpot_ms` / `slo_latency_ms` / `layouts` / `batch_limit`）；`POST /api/pareto.csv` 返回 CSV
- **B2 SLO goodput**：用户设定 TTFT ≤ X ms、TPOT ≤ Y ms（默认 2000 / 50，可在标签页内修改），报告满足两者的最优 tokens/s/芯片、达成配置、最大并发用户数，
  以及**哪个约束起作用**（TTFT / TPOT / 容量；无解时给出共同违反项）。新增 KPI 卡「SLO 吞吐 / 芯片」（当前布局，点击跳转标签页）。
  - 稳态假设（「假设」，DECISIONS D-0.30-3）：decode 以批量 B 连续运行；新请求最多等待一个在途 decode 步后**单独 prefill**（不与 decode 交织）：
    TTFT_eff(B) = TTFT_prefill(单条 prompt) + TPOT(B)；吞吐只计 decode（未扣 prefill 占用，偏乐观上界）；PP 沿用评估器 decode 气泡（偏保守）
- **B3 视频 / 蛋白**：给出等效曲线 —— 视频 帧/s/芯片 vs TTFC，蛋白 序列/s/芯片 vs 单批时间；SLO 为单一时延上限（默认 30 s / 10 s）。两者多为算力受限，曲线主要区分并行布局（标签页内注明）。

### 参考场景影响（每卡 HBM3E 8×12H 288 GB，×8 卡，tp = 8，batch 1，上下文 512，默认算力 16 核 × 6.25 T）
- **deepseek-v3 权重字节**：总参数 734.3B → **670.9B**；全量权重（bf16）1468.7 → **1341.8 GB**；tp8 每卡存储 183.6 → **167.7 GB**；
  decode 每 token 每卡权重流 16.79 → **8.90 GB**（单卡 134.3 → 71.2 GB；激活参数 68.1B → 37.4B）；TPOT 10.536 → **6.049 ms**；
  每卡 KV 32.04 → **36.05 MB**（MLA KV 计入 RoPE 64 维，+12.5%）
- **qwen3-32b**：TPOT 不变（4.957 ms）；非共享 LM head 使总参数 31.98B → 32.76B（每卡存储 8.00 → 8.19 GB）
- **qwen3-32b ×8 帕累托**（默认 SLO TTFT ≤ 2000 ms / TPOT ≤ 50 ms）：最优 **1494 tokens/s/芯片 @ TP8·PP1·EP1（注意力 TP）B = 560**，
  每用户 21.3 tokens/s，TPOT 46.8 ms，TTFT_eff 93 ms，最大并发 560，**TPOT 约束起作用**（容量上限 > 4096，不受限）；
  UI 默认算力（cluster_256t）下为 4184 tokens/s/芯片 @ B = 1632（同为 TPOT 约束）
- 容量单位：同一 HBM3E 8×12H 配置 API `capacity_GB` 309.24 → **288.0**（`capacity_GB_decimal` = 309.24）

### 测试
- 新增 12 项（共 161 项）：DeepSeek-V3 参数手算（671B ±1% / 37B ±3% / 187.1M 每层）、V4 低秩回退、容量单位、扫描短标签 + `raw_id`、
  旧 LPDDR6 位宽保留、帕累托前沿正确性（随机点 + 真实场景单调性）、容量上限 batch 与评估器 OOM 边界逐一一致（TP / DP）、
  goodput 约束（TPOT / 容量 / TTFT / 不可行）、视频 / 蛋白曲线、API + CSV + CLI + HTTP 路由、UI 标记（标签页 / KPI / 底部抽屉）

## 0.29.0 — 2026-10-08

**引擎 + UI 版本**：存储目录按 JEDEC / 厂商资料重建（结构化选择器 + 来源标签），并修正两处乐观假设（暴露的通信同步、KV 按 TP 切分）。
依据：`research/memory_specs_2026-10.md`（+ `.json`）§5 列出的 14 处目录问题、§6 选择器方案；`research/related_tools_2026-10.md` 两处乐观偏差。

### A. 存储目录重建（`npu_dse/mem_catalog.py` 新增；`memory.py` / `package_ranges.py` / `workbench.py` / `serve.py` / `cli.py`）
- **结构化选择**：存储类型 LPDDR5 / LPDDR5X / LPDDR6 / HBM3 / HBM3E / HBM4 / HBM4E；LPDDR 形态 板载封装 / SOCAMM2 / LPCAMM2；
  封装位宽；速率档（LPDDR 显示 WCK / CK MHz，HBM 显示 Gb/s/pin）；颗数 / 模组数 / 堆数；单颗容量（LPDDR）或 堆叠高度 8/12/16H × die 密度（HBM）
- **派生量**：总线位宽 = 数量 × 单元位宽；原始带宽 = 总线 × MT/s / 8000；**LPDDR6 可用带宽 = 原始 × 8/9**（BL24 元数据，单独显示，与效率分开）；
  有效带宽 = 可用 × 效率（假设 0.70）；容量 = 厂商标称 GB（= 2³⁰ B）
- **几何修正**：LPDDR5/5X 封装 = **x64（4×16-bit 通道）**；LPDDR6 封装 = **x96（4×24-bit 通道，每通道 2×12-bit 子通道）**，不再按 x24 die 计；
  SOCAMM2 / LPCAMM2 = 128-bit 模组；速率档按代际绑定（不再出现跨代速率）；HBM4/4E 为 2048-bit 堆；HBM 容量 = 层数 × die 密度
- **来源标签**：每个组件（速率 / 位宽 / 容量 / 数量）各有标签 JEDEC > 疑似 JEDEC > 厂商量产 > 送样 > 已发布 > 推测，
  组合标签取**最弱项**并在 UI 徽标显示；允许推测组合但必定标注；板载 LPDDR 总线 > 576-bit、HBM > 12 堆给出提示
- **向后兼容**：所有 ≤0.28 封装 id（`?c=` 深链、API `package_id`、CLI `--package`、预设）解析到最近的新配置并附说明（`GET /api/memory?resolve=<id>`）；
  API 也接受结构化字段（`mem_type` / `mem_form` / `mem_width_bits` / `mem_rate_MTps` / `mem_count` / `mem_cap_GB` / `hbm_height` / `hbm_die_Gb`）
- **预设**：更贴近真实产品 —— `edge-lpddr-4x64`（LPDDR5X 4×x64 @8533，64 GB）、新增 `edge-lpddr6-4x96`、`card-hbm-4stack`（HBM3E 4×12H，144 GB）、
  新增 `card-hbm3e-8x12h`（288 GB，B300 / MI355X 级）、新增 `server-socamm2-8`（SOCAMM2 8×192 GB）、`scaleup-8chip`；默认场景（HBM3E 8×12H）不 OOM
- **封装扫描**按结构化维度迭代：`package_axis = type | count | rate`（CLI `workbench-scan --package-axis`；API `package_axis` / `mem_axis`），每行带来源标签
- §5 的 14 处目录问题全部修正（逐条对照见 `DECISIONS.md` D-0.29-1）

### B. 两处乐观假设修正（`npu_dse/scaleup.py` / `workbench.py`）
- **修正 1 —— 暴露的通信同步**：每次集合通信固定时延 α（默认 **3 µs**，范围 2–5 µs，「假设」）× 每 token 集合通信次数
  （TP 每层 2 次 all-reduce、EP 每层 2 次 all-to-all、PP 每级 1 次收发）×（1 − 重叠），**叠加在** max(计算, 访存, C2C …) 之外；
  重叠默认 0（完全暴露）。适用于 LLM 与视频 / 蛋白质多卡。α = 0 与 0.28 结果逐位一致（有测试）
- **修正 2 —— KV 切分**：GQA 每卡 KV = KV × ceil(n_kv / tp) / n_kv（tp > n_kv 时复制）；**MLA latent 不能按 TP 切分 → 每个 TP rank 全量**，
  除非使用注意力 DP；新选项 `attn_parallel = tp | dp`（DP：KV 按 batch 切分，注意力权重每卡复制）。容量 / OOM 与 decode KV 流量同步更新
- **参考场景影响**（每卡 HBM3E 8×12H 288 GB，×8 卡，tp = 8，batch 1，上下文 512）：qwen3-32b TPOT 4.573 → **4.957 ms**（+128 × 3 µs；scale_efficiency 1.000 → 0.923）；
  deepseek-v3（MLA）每卡 KV 4.01 MB → **32.04 MB**（×8），TPOT 10.170 → 10.536 ms；默认 illustrative_27B ×8 scale_efficiency 1.000 → 0.945
- 新输出字段：`t_sync_ms`、`n_sync_per_token`、`c2c_latency_us`、`sync_overlap`、`attn_parallel`、`kv_replication`、`mem_*`（summary / tag / raw / payload …）

### UI（`npu_dse/web/*`）
- 场景栏「存储」= 类型下拉 + 摘要按钮（如 `4×96b @10667 · 455 GB/s · 64 GB` + 来源徽标）；点击弹出**存储配置弹层**：
  形态 / 位宽 / 速率（WCK·CK）/ 数量 / 容量（HBM：层数 / die 密度 / 堆数），派生块（总线 / 原始 / 可用 ×8/9 / 有效 / 容量）、
  组件标签 + 综合（最弱项）、提示；1280 下场景栏仍为两行；移动端弹层全宽
- 时间分解新增「通信同步（暴露）」行；硬件表新增「KV 切分」（含复制 ×N 徽标）与「通信同步」行
- ⚙ 高级参数新增「通信同步 / KV 切分」（α µs、同步重叠、注意力并行 TP / DP，标「假设」）；手动几何增加容量输入（标「推测」）
- 扫描「存储」轴：数量 / 速率档 / 类型 / 目录精选；新预设 3 个；旧 `?c=` 链接自动解析（异步 `restoreConfigFromUrl`）
- 容量 KPI 与存储摘要统一按厂商标称 GB（2³⁰ B）显示

### 测试 / 文档 / 产物
- 测试 **149 passed**（0.28.1：136）。修改的旧测试（仅因旧目录 / 旧假设本身有误）：
  `test_package_catalog_membership_and_bw_monotonicity`（x24 die、旧 id、速率未绑代际）、`test_package_sweep_and_cli_list`（新 id；128-bit 仅限模组）、
  `test_serve_module_import_and_api_smoke`（`hbm3e_` 前缀）、`test_scale_efficiency_perfect_tp_near_one_v025`（>0.99 只在 α = 0 时成立）、
  `test_report_has_econ_section_v024`（预设行数随预设数）
- 新测试：手算核对 LPDDR6 4×x96 @10667 → 512 / 455.1 GB/s；HBM3E 8×12H×24Gb → 288 GB、9.42 TB/s；LPDDR5X 8×x64 @8533 → 546.1 GB/s；
  最弱标签、旧 id 解析、同步 α 叠加与 α = 0 恒等、视频 / 蛋白质多卡同步、GQA / MLA / DP KV 切分、结构化扫描、默认不 OOM、UI 标记
- 文档：README / RELEASE / REQUIREMENTS_GAP / DECISIONS；重新生成 `out/report.html`、`out/card.*`、`out/workbench_scan.csv`、`out/parallel_*_8.csv`、`out/sweep_*.csv`
- 版本 0.29.0；zip `npu-inference-dse-0.29.0.zip`（新增 `npu_dse/mem_catalog.py` 与 `research/` 资料）

## 0.28.1 — 2026-10-08

- **Web 工作台小幅打磨（仅 UI 层；无建模 / 引擎 / serve.py 改动；API / JSON / CSV 导出不变）** — `npu_dse/web/{index.html,app.js,style.css}`
  - **场景栏下拉框不再截断**：系列 / 封装 / 算力 / 场景预设下拉框收起时显示短标签（如 `qwen3-32b · dense 32B`、`HBM3e ×4 · 3.3 TB/s · 48 GB`、`256T · 16 核`），
    宽度按所选项文字自适应（canvas 测量）；完整信息放在 tooltip（title）中，选项按世代 / 架构分组（`<optgroup>`）；TP·PP·EP 摘要移到第 2 行
    （「并行」）；1280 / 1440 / 1920 下（含最长组合）所选值均无截断
  - **抽屉打开时场景栏保持两行**：≥1280px 时抽屉停靠在吸顶栏下方（`--head-h`），吸顶栏保持全宽、不再挤压；1280–1439px 时收紧间距、
    隐藏「仅 HF」复选框文字（tooltip 保留）与分隔线
  - **时间单位自动缩放**：KPI / 时间分解 / A|B / 扫描表与柱状图 / 基线提示 —— ≥1000 ms 显示 s，<1 ms 显示 µs；悬停显示原始 ms；
    **字节**自动缩放（如 31205621760 → 31.21 GB，硬件表 / A|B / pair 字节 / 集合通信字节），悬停显示原始字节数；
    CSV / JSON 导出与 API 数值不变
  - A|B 非时间 / 字节字段改为有效数字显示（不再出现 `3297.2800`）；容量显示去掉多余位数（`19.31 / 48 GB`），原值在 tooltip
  - 视频时间分解标题改为「单步前向时间分解」，提示 `TTFC … = N 步 × 单步前向`（分量为单次去噪前向，此前易误读为 TTFC 分量）
  - OOM 时 KPI 副标题精简（徽标「容量不足」+ 容量），1280 下不再省略号截断
  - 残留英文 / 不一致清理：`vs 单卡 · time_per_seq` → `相对单卡 · 每序列时间`；「说明（honesty）」→「诚信说明（假设 / 未标定）」；
    「计算 compute」→「计算」；页脚「本地 MVP」→「本地工具」；场景预设选项改为中文短名（预设 id 保留在 tooltip 与 `data-preset`）；
    A|B 字段名去掉「(ms)」后缀（单位已随数值显示）
- 测试 136 passed；版本 0.28.1；重新生成 `out/report.html`；zip `npu-inference-dse-0.28.1.zip`

## 0.28.0 — 2026-10-08

- **Web 工作台布局重构（仅 UI 层；无建模 / 引擎 / serve.py 改动）** — `npu_dse/web/{index.html,app.js,style.css}`
  - **吸顶场景栏（两行）**：领域 / 系列 / 存储（HBM·LPDDR + 封装）/ 算力 / 芯片数步进器 + TP·PP·EP 摘要；
    批大小 / 各领域负载 / dtype / 量化 / 场景预设（下拉，`<option data-preset>`）/ ⚙ 高级参数（「N 项已修改」徽标）
  - **顶栏**：状态胶囊（已自动更新 · 时间）、⟳ 重新评估、🔗 复制链接（当前 `?c=` 深链）；参数变更仍为防抖自动评估
  - **KPI 条（7 项，按领域切换）**：LLM = TTFT / TPOT / 瓶颈 / 容量·OOM / 扩展效率·加速比 / J/token / $/MTok；
    视频 = TTFC / 帧/秒 / … / J/帧 / 系统 $；蛋白质 = 每序列时间 / pair 字节 / … / J/序列 / 系统 $；
    基线 Δ% 显示在主指标（LLM：TPOT；视频：TTFC；蛋白质：每序列时间）
  - **结果标签页**：概览 | 对比/扫描 | A|B 对比 | 假设与说明；固定基线 / 固定为 A / 固定为 B 位于标签栏右侧；当前标签写入 `?c=`（`tab`，旧链接照常恢复）
  - **概览**：可视化时间分解（堆叠条 + 分量条，瓶颈高亮）、硬件与资源表、前 4 条假设（链接到完整列表）
  - **⚙ 高级参数抽屉**（420px，折叠分组：并行 / 存储几何 / 算力手动 / 校准覆盖 / 算力附加项 / 能耗成本，各组显示状态）；
    ≥1280px 非模态（主区让位，KPI 保持可见），窄屏为遮罩浮层；Esc 关闭
  - **≥1680px 分屏**：左侧固定概览，右侧为所选次级标签（默认 对比/扫描）
  - **移动端（≤768px）**：场景栏折叠为一行摘要 +「编辑场景」+ ⚙
  - honesty 说明合并为**一行**（链接到「假设与说明」完整原文）；错误条移到 KPI 上方
  - 对比/扫描面板首次可见时按当前维度自动运行一次（避免空白面板）；参数变更后标记「已变更，点击运行扫描刷新」；切换领域时清空旧领域结果
  - A|B：精简摘要 + 默认仅显示有差异的字段（可勾选「显示全部字段」）；整数字段不再显示 `.0000`
- **修复**
  - wall `balanced` → 「均衡」
  - 首次加载默认场景改为 qwen3-32b / HBM3e 4-stack / cluster_256t / 2 芯片（三个领域默认均不 OOM）；深链仍覆盖默认值
  - 芯片数步进器按 1/2/4/8（锁定 TP/PP/EP 后可到 64）步进，避免 TP=3 之类引擎拒绝的取值；该类错误附中文修正提示
  - 非 GEMM / dtype MAC 控件此前仅在 `?c=` 恢复时绑定事件（首次打开不生效）→ 启动时绑定一次
- 测试：10 处硬编码版本断言改为读取 `npu_dse.__version__`（并校验与 pyproject.toml 一致），今后升版无需改测试；136 passed
- 版本 0.28.0；重新生成 `out/report.html`；zip `npu-inference-dse-0.28.0.zip`

## 0.27.0 — 2026-10-08

- **Web UI 中文化（简体中文）** — `npu_dse/web/{index.html,app.js,style.css}`
  - 所有面向用户的文案（分区标题、标签、按钮、提示 / tooltip、占位符、表头、状态 / 错误消息、toast、空状态、
    能耗 / 成本、基线、对比 / 扫描、A|B 双卡、预设说明）改为简体中文；TTFT / TPOT / TTFC / TOPS / HBM / LPDDR /
    NPU / PE / GEMM / MoE / MLA / KV / TP/PP/EP / C2C / dtype 名 / 模型与系列名 / 单位等专有名词保留英文
  - `<html lang="zh-CN">`；本地 CJK 字体栈（PingFang SC / Microsoft YaHei / Noto Sans CJK SC …，无 CDN）
  - 假设值统一标记为「假设」；serve.py 返回的 honesty banner 在 UI 层显示中文译文（可展开英文原文）；
    引擎 assumptions / notes / catalog 说明按已知短语做显示层翻译（未知短语保留英文原文，悬停可见原文）
  - **不改** API / JSON key / 字段名 / id / CLI 输出 / CSV 表头
- **布局修复（文字重叠 / 挤压）**
  - 表单：标签列 `110px` 固定宽 → `minmax(84px,128px)`；复选框标签不再被塞进标签列（整行显示）；滑块行同理
  - 按钮行统一 `flex-wrap`（左侧「评估 / 固定基线 / 固定为 A/B / 恢复默认」不再挤成两行小按钮）
  - 扫描按钮行与下方提示文字重叠（`.hint` 负 margin）→ 去除负 margin
  - 指标卡 / A|B 卡 / 长 id、URL、metadata：`min-width:0` + `overflow-wrap:anywhere`；A|B 网格改 `auto-fit,minmax(320px,1fr)`
  - 对比表 / 双卡 Δ 表：放入横向滚动容器；表头 sticky 限定在表格容器内（不再被页面 sticky 顶栏遮挡）
  - 柱状图标签由单行省略改为最多两行换行；顶栏状态文本省略号截断；toast 宽度随视口收缩
- 版本 0.27.0（测试中的版本守卫断言同步改为 `0.27`；测试数不变）；重新生成 `out/report.html`；zip `npu-inference-dse-0.27.0.zip`

## 0.26.0 — 2026-10-08

- **Portability fix** — test + CLI `export-csv` / `report` default output now resolve to `<repo>/out/` instead of hardcoded `/workspace/...` (runs from any directory)
- **Coarse non-GEMM operator overhead (ASSUMED, opt-in)** — Softmax / RoPE / LN / misc
  - Knob `non_gemm_overhead` ∈ [0,1], default **0 (off)**: `t_compute' = t_compute * (1 + overhead)`
  - Optional finer `softmax_frac` / `rope_frac` / `norm_frac` (if any set, sum replaces coarse)
  - **Not cycle-accurate**; off by default for conservation / handcheck
- **Optional dtype MAC peak factor (ASSUMED, opt-in)** — user table, not silicon
  - Map `dtype_mac_factors` (fp16/fp8/int8/int4 or bit keys); **default all 1.0** = bytes-only conservative
  - When set: `peak_TOPS *= factor`, `t_compute /= factor` for the weight dtype
  - CLI `--dtype-mac-factor '{"int4":4,…}'`; workbench field / API / UI JSON
- Wired: `EvalConfig` + scaleup/domain paths via `effective_compute_time_s`; WorkbenchConfig; MetricsCard echo;
  CLI / API `config_echo`; Web **Assumed compute extras** section (sliders + finer details + EXAMPLE factors)
- DECISIONS §17 (Softmax/RoPE/LN + low-prec MAC → assumed opt-in); REQUIREMENTS_GAP; report `#assumed-compute` note
- Tests: default path bit-identical; overhead>0 ↑ latency; int4 factor>1 may flip wall; handcheck / conservation guards
- Regenerated `out/report.html`; zip `npu-inference-dse-0.26.0.zip`

# CHANGELOG

## 0.25.0 — 2026-10-08

- **Scaling efficiency (explicit MetricsCard fields)** — user-facing scale-up quality metric
  - For `chip_count>1` (or `tp*pp*ep>1`): compare domain primary latency to single-card baseline
    (same WorkbenchConfig except `chips=tp=pp=ep=1`)
  - `speedup = t_single / t_multi`; `scale_efficiency = speedup / chip_count` (ideal **1.0**)
  - Domain primary: LLM→**TPOT**, video→**TTFC**, protein→**time_per_seq**
  - On chips=1: `scale_efficiency=1.0`, `speedup=1.0`, note `"single-card"`
  - Also exposes `t_single_primary_ms`, `scale_metric`
  - **Honesty**: ignores host/NIC non-ideal; C2C / PP bubble / EP already inside `t_multi`
- Surfaced in MetricsCard JSON/MD, Web card tiles, dual A|B Δ, sweep CSV columns (server + client)
- UX: OOM badge more visible (border/glow + `CAPACITY OOM` pill); scale_eff / speedup on card
- Report: new `#scale` section — scale_efficiency vs chips tables (dense + MoE)
- DECISIONS §16; REQUIREMENTS_GAP scaling note; README MetricsCard fields
- Tests: chips=1 → 1.0; chips=8 compute-bound ≈1; chips=8 PP/c2c-bound efficiency < 1; CSV/UI/report markers
- Regenerated `out/report.html`; zip `npu-inference-dse-0.25.0.zip`


## 0.24.0 — 2026-10-08

- **Energy / power stub (ASSUMED — not silicon / PDK / JEDEC)** — new `npu_dse/econ.py`
  - Knobs: `tdp_w` (W/card, wins) **or** `watts_per_tops` × per-card peak TOPS; `power_util` (default 1.0 = TDP upper bound)
  - MetricsCard: `est_power_W`, `est_power_per_card_W`, `est_energy_per_token_J` (+ `est_energy_prefill_J`) for LLM,
    `est_energy_per_frame_J` (video), `est_energy_per_seq_J` (protein), `econ_configured`, `econ_basis`
  - **Engine defaults OFF**: no knobs → all `est_*` = 0; placeholder numbers live only in `examples/energy_cost.example.json`
    (mirrored to `npu_dse/data/` for the UI "Load EXAMPLE placeholders" button) and are labeled EXAMPLE / fake
- **Cost stub (ASSUMED)**: `cost_per_card_usd`, `mem_addon_usd` → `est_system_cost_usd = chips × (card + add-on)`;
  LLM `est_usd_per_Mtok` = energy (`usd_per_kwh`) + capex (`amortize_years`, `duty_cycle`) over decode-only tokens; `est_tokens_per_s`
- **Scenario presets** (`npu_dse/scenarios.py`): `edge-lpddr-4x64`, `card-hbm-4stack`, `scaleup-8chip` (package + compute + chips only; no power/price)
  - CLI: `workbench --preset …` (explicit flags win), `list-presets`, `eval-presets [--econ …]`; econ flags `--tdp-w` `--watts-per-tops`
    `--power-util` `--cost-per-card` `--mem-addon-usd` `--usd-per-kwh` `--amortize-years` `--duty-cycle` `--econ path.json`
  - API: `GET /api/presets` (+ `econ_example`); `/api/eval` & `/api/sweep` accept `preset`, flat econ keys, or `econ:{…}`; `config_echo` lists econ knobs
  - Web: quick-apply preset buttons; **Energy / cost (ASSUMED stub)** section (inputs + util/duty sliders + EXAMPLE / Clear);
    est_* tiles; dual A|B + sweep table/CSV carry est_* columns; econ persisted in `?c=`
- Report: new `#econ` section — presets × {llm int8, video, protein} with EXAMPLE placeholder knobs + honesty text
- REQUIREMENTS_GAP: #19 energy efficiency / #20 cost → **半齐（assumed stub）**; #21 scenario presets; #10 note clarifies stub ≠ PDK
- DECISIONS §15; README knob table + formulas
- Tests: econ off-by-default, LLM/video/protein math, tdp vs W/TOPS precedence, example JSON placeholder labeling,
  presets catalog/eval, API/CLI/sweep/CSV, UI markers, report section, gap marker; handcheck guard
- Regenerated `out/report.html`; zip `npu-inference-dse-0.24.0.zip`

## 0.23.0 — 2026-10-08

- **CalibrationOverrides** (optional, non-blocking): `mem_efficiency` / `mac_efficiency` (or `weight_hide`) / `frequency_hz` via `examples/calibration.example.json`
  - CLI: `workbench --calib path.json` (+ `--mem-efficiency` / `--weight-hide`)
  - API: body `calib: {…}` (also top-level `efficiency` / `mem_efficiency` / `freq_ghz` / `mac_efficiency`)
- **Web Assumed / override** section: mem eff + MAC/overlap hide + freq GHz sliders; **override active** badge when ≠ defaults; persisted in `?c=`
- **Dual card A|B**: Pin A / Pin B (or last vs baseline) with field-by-field Δ / Δ%; no CDN
- DECISIONS.md: UI + workbench video/protein no longer `out_of_scope` — synced to serve reality
- REQUIREMENTS_GAP #18 → **已齐（MVP+）**（注：非完整 BI）
- Tests: calib load/apply + API/CLI; dual/override UI markers; version 0.23
- Regenerated `out/report.html`; zip `npu-inference-dse-0.23.0.zip`


## 0.22.0 — 2026-10-08

- **Sweep export**: Compare/Sweep panel buttons download last `/api/sweep` result as **CSV** and **JSON** (client-side; server helpers `sweep_result_to_csv` / `sweep_result_to_json` for the same schema)
- **Baseline pin**: Pin current MetricsCard as baseline; sweep table shows **Δ%** vs baseline for primary metric (TPOT / TTFC / time_per_seq); lower is better
- **Config deep-link**: Current knobs encoded in URL `?c=` (base64url JSON); refresh restores state (documented in README)
- Honesty banner unchanged (assumed/uncalibrated BW/eff/freq)
- Tests: export helpers + delta_pct; UI markers for export/pin/`?c=`; version 0.22
- REQUIREMENTS_GAP #18 note: exports / baseline / deeplink

## 0.21.0 — 2026-10-08

- **Compare / Sweep UI + API** (Gap #18 — scheme comparison / multi-card matrix)
  - Web panel: pick axis (chips / package / compute / tp×pp parallel / series) → table of
    MetricsCards + lightweight CSS bar chart for TPOT / TTFC / time_per_seq (no CDN)
  - Parallel product presets: **Parallel ×4** / **Parallel ×8** (same matrix as `workbench-parallel`)
  - `POST /api/sweep`: returns `{axis, rows[{label, card, primary_ms}], metric_key, capped}`;
    grid hard-capped at **≤32** rows for safety
  - Reuses workbench eval paths; honesty banner unchanged (assumed/uncalibrated BW/eff/freq)
- UX: toast when `tp×pp×ep ≠ chip_count`; package labels emphasize **n×64** / **n×24** granules
- Tests: `/api/sweep` axes + cap + stdlib HTTP + UI asset presence; handcheck unchanged
- REQUIREMENTS_GAP #18 → **半齐** (lightweight charts; not a full BI dashboard)

## 0.20.0 — 2026-10-08

- **Video / protein multi-card PP / EP + finer communication model** (beyond chip_count→TP)
  - Parallel knobs: `tp`, `pp`, `ep` with `tp*pp*ep == chip_count` (same constraint as LLM)
  - **TP**: weight shard; activation all-reduce/all-gather — `ring` (default Megatron
    `2*(tp-1)/tp·V`) or `tree` (`2·ceil(log2(tp))·V` assumed); V ~ B·T·H·act (DiT) / B·L·H·act
  - **PP**: microbatch pipeline; bubble `(pp-1)/(mb+pp-1)`; video denoise is sequential
    (decode-like) → pipelines poorly (documented); protein forward may raise `mb` (`decode_mb`)
  - **EP**: only if MoE experts on shape; DiT/protein → ep>1 accepted as **no-op** with note
  - Wired into `evaluate_workbench` video/protein + `/api/eval`; UI enables pp/ep for all domains
  - Assumptions banner lists modeled vs ignored (NoC topology, denoise cross-step PP, pair partition, …)
- Tests: chip_count=8 `tp=4,pp=2,ep=1` vs `tp=8`; bubble/c2c fields; ring vs tree formulas
- Reuses LLM `scaleup.py` patterns; no invented silicon numbers

## 0.19.0 — 2026-10-08

- **LPDDR granule honesty (USER CORRECTION)**: packages are **64-bit devices**, not 128-bit chips
  - LPDDR5/5X catalog: `n_packages ∈ {2,4,6,8}` × `package_width_bits=64`; SoC bus = n×64
  - Min = **2×64** (128-bit SoC bus); max practical ~**8×64**; removed fictional 128-bit *granule* rows
  - ExternalMemory map: `n_channels = n_packages`, `width_bits = 64` (HBM stack≡channel unchanged)
- **LPDDR6 (JESD209-6, Jul 2025)**: die is **x24** = two **12-bit** sub-channels — **not 48-bit**
  - Rates assumed 10667 / 14400 MT/s (JEDEC press ~10.667–14.4 Gbps); dies {6,8,12,16,20} → ~144–480-bit buses
  - Documented in code comments + REQUIREMENTS_GAP; optional x6 sub-channel roadmap noted only
- **Video / protein multi-card MVP**: `chip_count` → tp weight-shard + C2C ring all-reduce on activations
  - Compute/weight bytes /N; collectives 2×/layer (Megatron-style); wall = max(compute, mem, c2c)
  - Protein pair capacity /N (crude honesty note); pp/ep ignored for these domains
  - Wired through `/api/eval` + workbench; UI domain notes updated
- Tests: catalog membership (no 128-bit granule; LPDDR6×24); BW monotonicity; video/protein chip_count>1
- CLI/report/UI package labels show `n_packages × granule` + bus width

## 0.18.1 — 2026-10-08

- Fix: `/api/eval` accepts `series` / `series_id` aliases for `model_id` (UI was sending `series`, which previously fell back to illustrative_27B).

## 0.18.0 — 2026-10-08

- **Multi-domain MetricsCard via `/api/eval`**: video + protein series produce eval cards (not LLM-only)
  - Video: `TTFC_ms`, `frames_per_s`, `n_denoise` / `n_frames`, BW wall, capacity needed vs package
  - Protein: `time_per_seq_ms`, `pair_bytes`, `seq_len`, capacity / OOM flags
  - Shared fields: `wall`, `mem_eff_GBps`, `peak_tops`, breakdown, `assumptions`, honesty banner
  - LLM path unchanged (`TTFT_ms` / `TPOT_ms` + scale-up)
- Workbench: `evaluate_workbench` dispatches by series domain; knobs `n_denoise` / `n_frames` / `seq_len`
- Web UI: domain filter shows video/protein knobs + domain-appropriate metric labels; eval enabled for all domains
- Video/protein remain **single-card** roofline (no TP/PP/EP scale-up model)
- Tests: API eval for `minimax-h3` / `illustrative_dit_video` and `esmfold` / `illustrative_protein_pair`
- REQUIREMENTS_GAP: multi-domain MetricsCard + UI note updated

## 0.17.0 — 2026-10-08

- **Interactive Web workbench (MVP)**: `python3 -m npu_dse serve [--host --port]`
  - Local single-page UI under `npu_dse/web/` (no CDN): domain/series, chips + tp/pp/ep,
    HBM/LPDDR package presets or manual geometry, core/cluster compute presets or
    `n_cores × tops_per_core`, batch/ctx/dtype/quant + assumed efficiency / MAC-hide sliders
  - Live MetricsCard (TTFT/TPOT, wall, OOM, peak TOPS, eff BW, breakdown, assumptions banner)
  - Debounced `POST /api/eval`; catalogs via `GET /api/series|packages|compute`
  - Backend: FastAPI+uvicorn when installed (`pip install 'npu-dse[web]'`); else stdlib `http.server` (`--stdlib` to force)
  - Honesty banner: BW / efficiency / freq assumed/uncalibrated; MetricsCard path **LLM-only**
    (video/protein remain CLI `sweep-video` / `sweep-protein`)
- Static `report.html` unchanged; prefer `serve` for knob exploration
- Tests: API smoke (glm-5.3 / illustrative_27B), serve module import
- REQUIREMENTS_GAP: UI → **半齐 / MVP**


## 0.16.0 — 2026-10-08

- **Design-space package ranges** (NOT silicon calibration): `npu_dse/package_ranges.py`
  - LPDDR: 4×64 / 4×128 / 8×64 / 8×128 × rates 6400 / 8533 / 9600 MT/s (JEDEC LPDDR5X marketing band; assumed)
  - HBM: stacks 2–8 × gens HBM3 / HBM3e / HBM4 / HBM4e (HBM4e = design-target label; rates/widths assumed from public JEDEC/marketing cites)
  - Maps to `ExternalMemory` via existing `n_channels × width_bits × rate` formula; HBM stack ≡ channel (consistent with `HBM_PRESET`)
- **Compute hierarchy catalog**: core 4/8/12/16 TOPS; cluster 64/128/192/256 TOPS with (n_cores × tops_per_core) decompositions @ assumed 1 GHz
- CLI: `list-packages` / `list-compute`; `workbench-scan --sweep-package` / `--sweep-compute`; expanded `--sweep-mem` default channel/rate grids
- HTML report: package-range primer + LPDDR/HBM sweep tables + core/cluster compute sweeps (wall FLIP notes)
- REQUIREMENTS_GAP: 「标定」redefined as **range DSE (假设档已齐)**; true remaining = optional measured efficiency if ever provided + UI
- Tests: catalog membership, BW monotonicity (more channels / higher rate → higher peak_Bps), handcheck unchanged

## 0.15.0 — 2026-10-08

- **Real public HF / model-card series packs**: bake `npu_dse/data/series_catalog.json` from `out/hf_series_lookup.json` + raw configs; loader in `npu_dse/catalog.py`; register alongside illustrative/toy packs in `series.py`.
- User-named LLMs: `glm-5.3`, `glm-5.3-flash`, `deepseek-v4.1-flash`, `deepseek-v4-pro`, `kimi-k3`, `kimi-k2.7-code`, `qwen3.8-2.4t`, `qwen3.8-27b`, `qwen3.8-flash-next`.
- Recommended extras (public only): DeepSeek-V3/V3.1/V3.2/R1/V4-Flash, GLM-5/5.2/4.5/4.5-Air/4.6, Kimi-K2/K2.5, Qwen3-*, MiniMax-Text-01/M1, Mixtral-8x22B, Mistral-Large, Magistral-Small, Phi-4, Yi-1.5-34B, InternLM2.5/3, Seed-OSS-36B, gpt-oss-20b/120b, …
- Video: MiniMax-H3, CogVideoX-2B/5B, HunyuanVideo, Wan2.1 1.3B/14B, Wan2.2-A14B, LTX-Video, Mochi-1, Open-Sora STDiT2/3.
- Protein: AF2/AF3 (schema), ESMFold, ESM-2 650M/3B, Boltz-1, Protenix, OpenFold→AF2 dims.
- **Skipped (gated / missing dims)**: Llama-4 Scout/Maverick, Gemma-3, ESM-3, Chai-1, RFdiffusion, ambiguous Qwen 3.8 umbrella.
- `list-series --product` → HF-backed packs; illustrative remain for handcheck (`product_only=False`).
- Metadata cites `hf:<id>` + source URL; FLOPs/BW still uncalibrated. Hybrid linear-attn / Engram / DSA → metadata only.
- Tests: key dim checks (GLM-5.3 L=78 H=6144; Kimi-K3 E=896); no gated Llama-4 in registry.
- REQUIREMENTS_GAP: 「真实模型维数」→ 已齐（公开 HF）; remaining locks: BW/freq calib + UI.

## 0.14.0 — 2026-10-07

- **HTML report**: `python3 -m npu_dse report --out out/report.html` — self-contained (inline CSS), sections for walls primer, series list, chips sweep, parallel matrix (chips=8 dense+MoE), mem geometry FLIP note, cross-domain compare, assumptions banner.
- **REQUIREMENTS_GAP.md** (中文): original bullets → 已齐/半齐/缺 + CLI; extra knobs; true remaining locks (real dims, calibrated BW/freq, UI).
- Tests: report file exists and contains Metrics/wall; handcheck green.

## 0.13.0

- Workbench polish: `workbench-parallel` / `workbench-scan --mode parallel` (tp×pp×ep matrix); `--sweep-mem` geometry FLIP; MetricsCard `--json`/`--md`; series aliases `series/video` / `series/protein`.

## 0.12.0

- Product layer: `WorkbenchConfig` → `MetricsCard`; `chip_count`→tp; DRAM geometry; `n_cores×tops_per_core`; series packs (`series/dense-27b` …); CLI `workbench` / `workbench-scan` / `list-series`.

## 0.11.0

- Cross-domain one-pager (`compare-domains`); `illustrative_large_dit` minute-scale TTFC narrative (not MiniMax-H3); CSV compare export. Still uncalibrated → not 1.0.

## 0.10.0

- Multi-domain templates: `illustrative_dit_video` / `illustrative_protein_pair`; CLI `sweep-video` / `sweep-protein` / `list-shapes`.

## 0.9.0

- MoE expert-parallel (`--ep`) + A2A; MLA compressed KV (`illustrative_mla` / `sweep-mla`); `export-csv` bundle.

## 0.8.0

- IB/RoCE KV fabric (`sweep-kv-fabric`); light pipeline-parallel + decode bubble (`sweep-parallel` / `sweep-pp`).

## 0.7.0

- Tensor-parallel + C2C collectives (`sweep-tp` / `scan-scaleup`); OOM capacity flags.

## 0.6.0

- Batch sweep (util vs M); MoE illustrative shape + active-W stream; FINDINGS.md silicon memo.

## 0.5.0

- SRAM honesty: staging ≠ resident; knees R≥1 / L/2 / L; `--sram-policy`; independent W/KV quant (`sweep-quant`).

## 0.4.0

- Context sweep (KV vs weight); SRAM MiB sweep; dtype bytes-only (fp16/fp8/int8/int4).

## 0.3.0

- SKU peak templates `sku_100t` / `sku_1p`; LPDDR mem-wall crossover narrative; `scan-sku`.

## 0.2.0

- Illustrative 27B shape; HBM/LPDDR presets; SRAM three-way partition; contention modes; default scan table.

## 0.1.0

- Charter MVP: OS PE + M=1 util formula; tiling-derived traffic (no default hit-rate); TTFT/TPOT; toy handcheck.

---

All absolute hardware numbers remain **assumed/uncalibrated** unless derived from shape + tiling + capacities. No fake PDK power/area.

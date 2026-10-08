# Changelog

本项目的重要变更记录于此。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)（1.0 之前次版本号可能包含不兼容变更）。
0.31.0 及更早版本以 `npu-inference-dse`（包名 `npu_dse`）发布。

## [0.42.0] - 2026-10-09

其余 6 个视频生成（DiT）发布接入 core v2，可选择、可评估。

### Added
- 可评估的视频模型：Wan2.2-T2V-A14B、HunyuanVideo、LTX-Video（2B v0.9）、Mochi 1 preview、Open-Sora STDiT3（1.2）、MiniMax-H3。均从官方 config + safetensors 头（含最后一层，捕捉异构末层）建模，参数与发布偏差 ≤ 0.001%；dtype 按发布（Wan2.2 / LTX / Mochi / Open-Sora 为 fp32；HunyuanVideo bf16；H3 为 bf16、io 层 fp32）。覆盖均为「部分」：文本编码器、VAE（H3 另有音频 VAE）未建模，标签列出各自大小。
- 结构：Wan2.2 两个 14B 专家按噪声级切换，每步只算一个，另一个计入常驻存储（不计读流量）；HunyuanVideo 20 双流 + 40 单流块（单流块一个融合输入 GEMM、TP 下一次 all-reduce），全 3D 联合注意力；Mochi 非对称双流（末层文本只作 K/V）；Open-Sora 按发布的分解时空注意力（空间块在潜帧内、时间块沿时间轴）；H3 视频 + 立体声音频 + 文本打包为一条序列，AdaLN 分支（13.0B）按 README 预计算缓存、不计存储与读流量。Wan2.2 / HunyuanVideo 为全 3D 注意力（不做时空分解）。
- 工作负载：各发布的潜空间帧规则（因果 VAE 4× / 6× / 8×、Open-Sora 17 帧分块、H3 的 17n+5 补齐）、音频 token、guidance 蒸馏（CFG = 1，HunyuanVideo / H3 每步一次前向，可 what-if 为 2）。Web 工作负载说明与指标卡、命令行 `eval` 显示音频 token 与分解注意力。
- 校验：`validate` 增加 Open-Sora 行（README：H100 上 720p 4s 用时 130 s，DiT FLOPs 折算 H100 利用率 13%，若按全 3D 注意力会超过 100%）；新增测试覆盖参数 / dtype、token 规则、FLOPs 与闭式计数、TP / SP 守恒（含联合文本与音频行）、standby / 缓存存储。
- 发布头抓取支持多子目录仓库（`repo:sub1+sub2`）。

### Changed
- 仍为「暂未接入 v2」：蛋白质结构预测（ESMFold、AlphaFold3 / 2、Protenix、Boltz-1、OpenFold）——需要 pair 表示与三角更新等算子。LLM 与 0.41 模型的结果不变（1176 项指纹逐字节一致）。

## [0.41.0] - 2026-10-08

视频生成（DiT）与蛋白质模型接入 core v2，可选择、可评估。

### Added
- 可评估的视频生成模型：Wan2.1-T2V-14B / 1.3B、CogVideoX-5b / 2b（DiT 去噪主干，全 3D 注意力；CogVideoX 为文本 + 视频联合注意力）；蛋白质：ESM-2 3B / 650M（编码器一次前向）。均从官方 config + safetensors 头建模，参数与发布逐项一致；dtype 按发布（Wan2.1 / ESM-2 为 fp32，逐 GEMM 上转换为 bf16 并计入向量开销）。覆盖：视频「部分」（文本编码器与 VAE 未建模，标签列出其大小），ESM-2「完整」。
- 场景新增 `workload`（帧数、分辨率、去噪步数、CFG、蛋白质序列长度、单段 / 批延迟 SLO；0 = 发布默认），布局新增 SP（Ulysses 序列并行）；视频 / 蛋白质的 DP 切分前向 batch（含 CFG 并行）。
- 指标：视频单段延迟、每帧延迟、每去噪步时间、帧/s/卡、段/小时/卡、实时倍率；蛋白质批延迟、序列/s/卡、残基/s/卡。精确 batch / 布局搜索、映射对比、扫描、Pareto 与容量提示都支持这两个领域（goodput 只用于 LLM）。定义见 docs/MODEL.md §11。
- 全序列前向的激活流式模型（激活超出 SRAM 时分块进出 DRAM，注意力按 flash 式重读 K/V）；全序列注意力的映射可选 GEMM 方向（`Oᵀ = Vᵀ·Pᵀ`）。
- Web：模型选择器中这 6 个模型可选；工作负载输入、SP 输入，领域专用的指标卡与表头；LLM 专用控件（phase、ctx、TPOT/TTFT SLO、投机解码、KV what-if、goodput 目标）对视频 / 蛋白质隐藏。命令行 `eval` 支持 `--frames --height --width --steps --cfg --seq-len --sp`。
- 校验：视频 / 蛋白质的 V0 不变式、FLOPs 与独立计数对照、TP / SP 守恒、精确搜索对照暴力枚举；`validate` 增加一行视频合理性核对（Wan2.1 README 4090 用时折算）。

### Changed
- `/api/health` 的 domains 为 llm / vlm / gen / protein。其余视频 / 蛋白质条目（Wan2.2-A14B、HunyuanVideo、MiniMax-H3、LTX-Video、Mochi 1、Open-Sora、ESMFold、AlphaFold3 / 2、Protenix、Boltz-1、OpenFold）仍为「暂未接入 v2」。LLM 结果不变（266 项指纹逐字节一致）。

## [0.40.2] - 2026-10-08

### Changed
- 模型目录与选择器改为按厂商 → 系列分组，不再把 LLM 与 VLM 分成两组：同一厂商（以及同一系列）的文本版与多模态版放在一起，VLM 行带「VLM · 视觉编码器未建模」标记。同一厂商的不同品牌合并（阿里：Qwen · Wan；智谱：GLM · CogVideoX；Meta：Llama · ESM；字节跳动：Seed · Protenix）。
- 视频生成（DiT）条目列在各自厂商之下、可评估条目之后，标「视频生成 · 暂未接入 v2」；模型选择器中也以不可选项列出。

### Added
- 蛋白质模型回到目录（ESM-2 3B / 650M、ESMFold、AlphaFold3 / AlphaFold2、Protenix、Boltz-1、OpenFold），标「蛋白质 · 暂未接入 v2」，不能评估；维数取自原目录。
- `/api/models` 增加 `catalog`（完整目录的显示顺序：厂商 → 可评估系列 → 暂未接入的系列）；`offline` 条目增加 `evaluable`、`source`。模型目录页顶部给出各领域条目数。

## [0.40.1] - 2026-10-08

Web 工作台打磨与搜索提速。

### Changed
- 外存目录按 2026-10-08 审计修正：标签拆成规范状态 × 产品状态两轴；去掉 LPDDR5X x96（Apple 定制件，其他客户买不到）；LPCAMM2 9600 升为量产；SOCAMM2 256 GB / LPDDR5X x32 64 GB 降为送样；HBM4 12H×32Gb 改为「JEDEC 允许 · 无产品」（样品是 16H×24Gb）；LPDDR6 补 11733 档、12800 升为 JEDEC、x96 的 8/12 GB 标签统一；LPDDR5T = LPDDR5X-9600 别名；可选 LPDDR6 meta 模式（默认关，容量预留 1/16 为「假设」）；8/9 payload 保留。
- 卡数成为唯一的场景字段（每副本卡数 = PP·TP·DP）：映射对比与布局搜索都使用场景的卡数，不再各自有卡数输入；改卡数时按 TP 重新填充布局，「搜索当前卡数的最优布局并应用」一键给出最优布局与 batch。
- 默认场景可行：Web 默认 TPOT SLO 100 ms（LPDDR 类存储器）并开启自动 batch，打开即是满足 SLO 的结果（核心库的默认 SLO 仍为 50 ms）。
- 放不下时不再满屏红色：指标条隐藏，改为中文提示（需求 / 容量 / 权重）和一键修正——最少卡数及其最优布局与 batch、同类型更大容量的存储器、或可放下的最大 batch；TPOT 超 SLO、TTFT 不达标用琥珀色提示。
- 指标条精简为 5 项（TPOT、吞吐 / 卡、瓶颈与有效 MAC、goodput、DRAM 需求 / 容量），1100 px 以上固定一行。
- 映射对比表：次优布局并入布局列第二行，1280 / 1440 下不截断；放不下的映射显示「放不下」且不可点击；点击行同时应用映射、布局（含卡数）与 batch。布局搜索在 decode 目标下不显示空的 goodput / TTFT 列。
- 界面文字统一为中文（技术术语保留英文）：模型注释、结构描述、容量与投机解码警告、稳定性扰动名称、表头等。
- 扫描图的整数横轴不再显示小数。

- 模型目录与选择器按领域 → 厂商 → 系列分组（LLM / VLM / 图像与视频生成），去掉「用户指定 / 推荐补充」这类按来源的分组；名称统一为官方仓库名，系列内按尺寸从大到小，FP8 / AWQ 紧跟原版。
- 每个模型都显示覆盖度（完整 / 部分 / 架构代理，「完整」用低调样式），非「完整」模型附逐项的近似之处（悬停与模型目录可见），并说明覆盖度指建模覆盖而非模型好坏。
- VLM（Qwen3.5 / Qwen3.8-27B / Qwen3.8-Flash-Next / DeepSeek-V4.1-Flash / Kimi-K2.5 / Kimi-K3 / GLM-5.3-Flash）单独成组，注明只评估语言主干、视觉编码器未建模。
- 结构与 dtype 相同的发布合并为一条（DeepSeek-V3 / V3.1 / R1、Kimi-K2.5 / K2.7-Code、GLM-5 / 5.2、GLM-4.5 / 4.6、MiniMax-Text-01 / M1-80k）；Yi-1.5-34B、InternLM2.5-20B、InternLM3-8B、Qwen2.5-3B / 1.5B 不再单列。它们仍可按 id 评估并参与参数核对。
- 图像 / 视频生成（DiT）条目回到目录，单独成组并标「暂未接入 v2」（不可评估）。

### Performance
- 批量搜索的上界改为由所有已评估点构成的分段上界（仍只依赖 step(B) 不减），并且只在上界可能超过门槛时才继续二分 b_max。
- 稳定性检查以原 top-1 布局的精确得分为门槛做跨布局分支定界。
- HTTP 服务把映射对比（每种映射一个任务）与稳定性请求放到工作进程池并行（服务退出或被杀时工作进程随之退出）；Web 端并发请求 5 个稳定性结果并逐行填入。
- DeepSeek-V3 × 8 卡（HBM3E 288 GiB，SLO 100 ms）映射对比 + 5 项稳定性：0.40.0 单进程约 170 s → 0.40.1 单进程约 10–15 s，Web 服务（工作进程并行）冷启动约 4–5 s。

### Added
- `/api/fit`：容量检查与修正方案（见 docs/MODEL.md §7）。
- `/api/models` 增加 `domain` / `provider` / `family` / `coverage_reasons` / `vision_params_B` / `same_as`，以及 `offline`（暂未接入的生成模型）与 `unlisted`（未单列的发布及原因）；去掉 `section`。
- 测试：`/api/fit` 修正方案真实可行且卡数最少；稳定性的门槛搜索与完整搜索一致；目录分组、覆盖度与原因一致、合并条目确为同结构同 dtype。

## [0.40.0] - 2026-10-08

核心重写（core v2）。旧引擎、旧 Web 与旧命令行已移除，API 与 CLI 不向后兼容。

### Added
- 模型按发布建模：从 HF config 与 safetensors 头读取逐角色参数与存储 dtype（fp8 block、MXFP4、NVFP4、int4 AWQ 等），55 个发布与总量偏差 ≤0.5%；官方量化版为独立条目；三轴标签（来源 × 覆盖 × dtype），未逐项建模的结构标「架构代理」。
- 逐 rank 算子图：TP / PP / 注意力 DP / EP / ETP，单卡与多卡同一路径。
- 映射作为设计变量：OS、WS 边缘加载、WS 宽面广播、OS + GEMV、可重构；逐算子 MAC 界与 SRAM 供数界；芯片原生格式矩阵与反量化开销。
- 存储规划（SRAM 驻留、staging、逐 stage 容量）与调度（绑定瓶颈、有效 MAC 比例、α-β 集合通信、投机解码 / MTP）。
- 精确 batch 搜索（分支定界）、布局排名、TPOT–吞吐 Pareto、含 prefill 的 goodput 与 DP prefill TTFT 标记、排名稳定性。
- 校验套件：变形关系、参数对照、H100 类趋势区间、GenZ 对照（`accel-dse validate`）。
- Web 工作台重写：单点评估、映射对比（每种映射的最佳布局、瓶颈、有效 MAC、TTFT、稳定性）、布局搜索、扫描与 Pareto、模型目录；请求按序号丢弃过期响应。
- JSON API：严格解析（拒绝未知字段与 NaN / Infinity），扫描为受控替换。
- 命令行：`serve` / `models` / `eval` / `search` / `compare` / `stability` / `validate`。

### Removed
- 旧解析引擎及其命令（workbench、pareto、report 等）、示例输出与能耗 / 成本 stub。
- 视频与蛋白质领域暂时下线（见 docs/MODEL.md §10）。

## [0.31.1] - 2026-10-08

### Changed
- 项目改名 `accel-dse` 并发布到 GitHub；Python 包 `npu_dse` → `accel_dse`，命令行入口 `accel-dse`（无旧包名兼容层）。

### Added
- MIT License；`.gitignore`（生成物 `out/*` 默认忽略，保留文档引用的示例输出）。

## [0.31.0] - 2026-10-08

### Added
- MoE 专家切分 `moe_shard = tp_ep | ep_all`；MoE ep > 1 时注意力在 ep 组间按 DP 运行。
- PP decode 微批 `decode_mb`（默认 min(B, pp)），流水填满。
- goodput 计入 prefill：吞吐口径「摊销（分块混合 / 独占）| 上界」，`--goodput-mode` / `--prefill-mode` / `--out-len`。
- 投机解码 / MTP：`spec_k` / `spec_accept` / `spec_draft`；开启 MTP 时容量计入 MTP 模块。
- Web：⚙「投机解码 / MoE 切分 / PP 微批」，分解新增「草稿」与「PP 遍历 / 气泡」。

### Changed
- 工作台每个 decode 步计入 LM head（注意力 TP 时按词表并行）。
- 帕累托 LLM batch 搜索上限 4096 → 65536。

### Fixed
- all-to-all 字节按每 rank token 份额计（此前按整批，偏高）。
- tp > 1 且 ep > 1 的 MoE 布局不再把专家重复存储 tp 份。

## [0.30.0] - 2026-10-08

### Added
- 吞吐–交互性帕累托（`pareto` CLI、`POST /api/pareto`、Web「吞吐 / 交互」标签页）与 SLO goodput（KPI「SLO 吞吐 / 芯片」）。
- 视频 / 蛋白等效吞吐–时延曲线。

### Changed
- `capacity_GB` / `capacity_needed_GB` 统一为厂商标称 GB（2³⁰ B），十进制值见 `*_decimal`。
- 扫描行使用短标签（原始 id 在 CSV `raw_id`）；移动端存储配置改为底部抽屉。

### Fixed
- MLA 注意力权重按投影建模（DeepSeek-V3 ≈ 670.9B，此前偏大）；补齐前置稠密层、非共享 LM head、MLA KV 的 RoPE 部分。
- 旧 `lpddr6_{n}x24_*` 链接按总线位宽映射到 x96 / x48 封装。

## [0.29.0] - 2026-10-08

### Added
- 结构化存储目录（`mem_catalog.py`）：LPDDR5 / 5X / 6、SOCAMM2 / LPCAMM2、HBM3 / 3E / 4 / 4E；来源标签取最弱项；LPDDR6 payload 8/9。
- 暴露的通信同步时延 `t_sync = n_sync · α · (1 − overlap)`（α 默认 3 µs，假设）。
- `attn_parallel = tp | dp`；新场景预设 `edge-lpddr6-4x96`、`card-hbm3e-8x12h`、`server-socamm2-8`。

### Fixed
- LPDDR 几何：LPDDR5/5X 封装 x64（4×16），LPDDR6 封装 x96（4×24）；HBM 容量 = 层数 × die 密度；速率档绑定代际。
- KV 按 TP 切分（GQA 按 KV 头；MLA latent 每个 TP rank 全量）。
- ≤ 0.28 的封装 id 自动映射到最近的新配置。

## [0.28.1] - 2026-10-08

### Changed
- Web：下拉框短标签不截断；时间 / 字节单位自动缩放；抽屉打开时场景栏保持两行。

## [0.28.0] - 2026-10-08

### Changed
- Web 布局重构：吸顶场景栏、KPI 条、结果标签页、⚙ 高级参数抽屉、🔗 复制链接、宽屏分屏与移动端摘要。
- 测试的版本断言改为读取 `__version__`。

### Fixed
- 默认场景不再 OOM；芯片数步进器只给出引擎接受的取值。

## [0.27.0] - 2026-10-08

### Changed
- Web UI 简体中文化（技术术语保留英文）；修复表单 / 卡片 / 表格的文字重叠。

## [0.26.0] - 2026-10-08

### Added
- 可选（assumed）`non_gemm_overhead`（Softmax / RoPE / LN 粗开销）与 `dtype_mac_factors`（低精度 MAC 峰值因子），默认关闭。

### Fixed
- `export-csv` / `report` 默认输出改为相对仓库的 `out/`。

## [0.25.0] - 2026-10-08

### Added
- MetricsCard `scale_efficiency` / `speedup`（相对单卡）；报告新增 scale 章节。

## [0.24.0] - 2026-10-08

### Added
- 能耗 / 成本 assumed stub（`tdp_w`、`watts_per_tops`、`cost_per_card_usd` 等，默认关闭）。
- 场景预设（`list-presets`、`--preset`、`GET /api/presets`）。

## [0.23.0] - 2026-10-08

### Added
- CalibrationOverrides（`--calib`、API `calib`）；Web 假设值覆盖滑条；A|B 双卡对比。

## [0.22.0] - 2026-10-08

### Added
- 扫描结果导出 CSV / JSON；固定基线 Δ%；`?c=` 深链保存配置。

## [0.21.0] - 2026-10-08

### Added
- Web 对比 / 扫描面板与 `POST /api/sweep`（chips / package / compute / parallel / series，≤ 32 行）。

## [0.20.0] - 2026-10-08

### Added
- 视频 / 蛋白多卡 TP / PP / EP（ring / tree 集合通信、PP 气泡）。

## [0.19.0] - 2026-10-08

### Added
- 视频 / 蛋白多卡（chip_count → TP）。

### Fixed
- LPDDR 封装粒度改为 64-bit 器件。

## [0.18.1] - 2026-10-08

### Fixed
- `/api/eval` 接受 `series` / `series_id` 作为 `model_id` 别名。

## [0.18.0] - 2026-10-08

### Added
- `/api/eval` 支持视频 / 蛋白 MetricsCard。

## [0.17.0] - 2026-10-08

### Added
- 本地交互 Web 工作台 `serve`（stdlib，可选 FastAPI）。

## [0.16.0] - 2026-10-08

### Added
- 存储封装 / 算力层级目录（`list-packages`、`list-compute`）与对应扫描。

## [0.15.0] - 2026-10-08

### Added
- 公开 HF config / model card 模型维数目录（LLM / 视频 / 蛋白），`list-series --product`。

## [0.14.0] - 2026-10-07

### Added
- 自包含离线 HTML 报告 `report`。

## [0.13.0]

### Added
- `workbench-parallel`（tp×pp×ep 矩阵）、存储几何扫描、MetricsCard JSON / Markdown 导出。

## [0.12.0]

### Added
- 产品层 `WorkbenchConfig → MetricsCard`；`n_cores × tops_per_core`；模型系列包。

## [0.11.0]

### Added
- 跨域对照 `compare-domains`；`illustrative_large_dit`。

## [0.10.0]

### Added
- 视频 DiT / 蛋白示意负载与扫描。

## [0.9.0]

### Added
- MoE 专家并行与 all-to-all；MLA 压缩 KV；`export-csv`。

## [0.8.0]

### Added
- IB / RoCE KV fabric；轻量流水并行与 decode 气泡。

## [0.7.0]

### Added
- 张量并行 + C2C 集合通信；容量 OOM 标记。

## [0.6.0]

### Added
- batch 扫描；MoE 示意形状。

## [0.5.0]

### Changed
- SRAM：区分 staging 与 resident，resident 膝点；`--sram-policy`；独立 W/KV 量化。

## [0.4.0]

### Added
- 上下文扫描、SRAM 容量扫描、dtype（只缩字节）。

## [0.3.0]

### Added
- SKU 峰值模板 `sku_100t` / `sku_1p`；`scan-sku`。

## [0.2.0]

### Added
- 示意 27B 形状；HBM / LPDDR 预设；SRAM 三分区；带宽争用模式。

## [0.1.0]

### Added
- 初版：output-stationary PE 与 M=1 利用率公式；由 tiling 推导外部流量；TTFT / TPOT；toy 手算用例。

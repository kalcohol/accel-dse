# Changelog

本项目的重要变更记录于此。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)（1.0 之前次版本号可能包含不兼容变更）。
0.31.0 及更早版本以 `npu-inference-dse`（包名 `npu_dse`）发布。

## [Unreleased]

### Changed
- 文档结构：精简 README；新增 `docs/MODEL.md`（建模方法、假设与局限）；存储规格调研移到 `docs/research/`；CHANGELOG 改为 Keep a Changelog 格式。

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

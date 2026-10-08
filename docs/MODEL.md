# 建模说明（Modelling method & assumptions）

本文说明 accel-dse 的解析模型怎么算、假设了什么、哪些没建模，方便审阅者逐条挑错。
状态标签：**derived** = 由形状 / 几何 / 公式推出；**assumed** = 可调默认值、未标定（UI / CSV 中标「假设」）。
**所有绝对数都依赖 assumed 旋钮，不是实测硅片数据。**

目录：[1 章程](#1-章程) · [2 单卡](#2-单卡引擎) · [3 存储目录](#3-存储目录) · [4 数值格式](#4-数值格式dtype) · [5 多芯片](#5-多芯片scale-up) · [6 服务模型](#6-服务模型pareto--slo-goodput) · [7 多域](#7-视频-dit--蛋白质) · [8 产品层](#8-产品层metricscard校准能耗成本) · [9 局限](#9-未建模--已知局限) · [10 典型结论](#10-典型定性结论)

---

## 1. 章程

- Inference-only、NPU-like（固定 dataflow / 专用 MAC 阵列），**不是** GPGPU 模拟器，也不是 cycle-accurate RTL / 时序仿真。
- 每个中间量可手算：toy 配置的全部数字见 [`examples/handcheck.md`](../examples/handcheck.md)，并由测试 `test_toy_handcheck_numbers` 守护。
- 默认路径**不要求用户填 cache hit-rate**：外部流量由 tiling + SRAM 分区容量推导。
- **不做伪 PDK**：`process_nm` 仅为标签，不推导面积 / 功耗。

## 2. 单卡引擎

### 2.1 PE 阵列（output-stationary）

```
cycles         = ceil(M/R) · ceil(N/(C·n_engines)) · K
util           = (M·N) / (R·C·n_engines · ceil(M/R) · ceil(N/(C·n_engines)))
peak_TOPS      = R · C · n_engines · freq_Hz · 2 / 1e12          # 2 flop/MAC
t_compute      = cycles / freq_Hz / mac_efficiency
```

- 当 `N % (C·n_engines) == 0` 时 `util(M=1) = 1/R`：decode（M = batch，常为 1）利用率远低于 prefill。
- 产品层用 `n_cores × tops_per_core` 描述算力：assumed `n_engines = n_cores`，每核近方形 PE（16 × 6.25 T → 56×56×16，≈100.35 T）。
- assumed：PE 几何、`freq`（默认 1 GHz）、`mac_efficiency`（默认 1.0）。

### 2.2 外部存储带宽

```
raw_Bps = 总线位宽 / 8 · MT/s · 1e6       # 总线 = 数量 × 单元位宽
usable  = raw · payload                   # LPDDR6 payload = 8/9，其余 1
eff_Bps = usable · efficiency             # efficiency 默认 0.70（assumed）
```

### 2.3 SRAM 三分区与权重路径

```
weight_partition + kv_scratch + act ≤ SRAM 容量
R = floor(weight_partition / W_layer)     # 跨 token 常驻片上的层数
策略 --sram-policy: weight_resident（默认，最大化 R）| kv_first | balanced
```

- **staging ≠ resident**：R = 0 时即使装得下 1×/2× W_layer 做 staging，decode 每 token 仍要读满 L 层（`W_DRAM ≈ L·W_layer`）；2× staging 只使 double-buffer 可用，`weight_hide_factor`（assumed）只折时间不折字节。
- R ≥ 1：稳态 decode `W_DRAM ≈ (L−R)·W_layer`；R ≥ L 时为 0。prefill / 冷启动仍装载整模一次。
- KV：prefill 写穿到外存；decode 若 `kv_scratch ≥ 本步 KV 工作集` 则片上命中，否则全量外读（**全有或全无**，无部分命中）。新 token KV 始终写穿。
- 激活：工作集超出 act 分区时按每层 2× spill（保守）。

### 2.4 带宽争用与时间墙

```
serialize   = (B_w + B_kv + B_act) / BW                     # 默认
share_fair  = max(2·B_w, 2·B_kv)/BW + B_act/BW
lower_bound = max(B_w, B_kv)/BW + B_act/BW
t_phase     = max(t_compute, t_memory)                     # roofline 重叠上界
TTFT = t_prefill，TPOT = t_decode_step
```

- 权重读与 KV 读**争用同一外存端口**；报告中给出 compute / memory / C2C / … 各分量与瓶颈（wall）。
- prefill 因果注意力近似为「矩形 × 1/2」。Softmax / RoPE / LayerNorm 默认忽略；可选 `non_gemm_overhead`（`t_compute' = t_compute · (1 + oh)`，assumed，默认 0）。
- 工作台每个 decode 步计入 LM head（读 V·H + GEMM (B·q, H, V)）；原始引擎 `EvalConfig.count_lm_head` 默认关闭以保持手算用例不变。

## 3. 存储目录

- 选择单位为**封装 / 模组 / 堆**（`n_units × unit_width_bits`），不是 die。
  - LPDDR5 / 5X 板载封装 x64 = 4×16-bit 通道（另有 x32；LPDDR5X x96 为已发布）。
  - LPDDR6 封装 x96 = 4×24-bit 通道 = 8×12-bit 子通道；payload 8/9 单独计，与 efficiency 分开。
  - SOCAMM2 / LPCAMM2：128-bit 模组（LPDDR5X）。
  - HBM：堆宽 1024（HBM3/3E）或 2048（HBM4/4E）；容量 = 层数 × die 密度 / 8；堆数 1–12（16 为推测）。
- 速率档依附代际；容量按厂商标称 GB（= 2³⁰ B），`*_decimal` 字段给出 10⁹ B 值。
- **来源标签**：每个组件（速率 / 位宽 / 容量 / 数量）各带标签 JEDEC > 疑似 JEDEC > 厂商量产 > 送样 > 已发布 > 推测，组合取**最弱项**；允许推测组合但必定标注。资料与来源 URL 见 [`docs/research/memory_specs_2026-10.md`](research/memory_specs_2026-10.md)（截至 2026-10，新产品发布后需人工更新）。

## 4. 数值格式（dtype）

- 均匀预设 fp16 / fp8 / int8 / int4，独立 W/KV 量化 `w16k16 / w8k16 / w4k16 / w8k8 / w4k8`。
- 默认**只缩放存储 / 流量字节**，MAC 峰值与 FLOPs 不变 → 对低精度是保守的（算力侧偏慢）。
- 可选 `dtype_mac_factors`（assumed，用户表，非硅）：`peak_TOPS ×= factor`、`t_compute ÷= factor`；默认全 1.0。

## 5. 多芯片（scale-up）

芯片数 = tp · pp · ep（DP = 1）。所有链路带宽 / 时延均为 assumed 预设，非 PHY / NIC 标定。

| 机制 | 模型 |
|------|------|
| TP | Megatron 式：Q/K/V/gate/up 列并行，O/down 行并行；每层 2 次 all-reduce，ring 每 rank 字节 `2·(tp−1)/tp·V`，`V = B·S·H·act_bytes`（视频 / 蛋白可选 tree `2·ceil(log2 tp)·V`） |
| C2C | 有效带宽预设 100 / 200 / 400（默认）/ 800 GB/s；`t_c2c = bytes / BW · (1 − c2c_hide)`，hide 默认 0 |
| 暴露同步 | `t_sync = n_sync · α · (1 − overlap)` **叠加在** max(…) 之外；TP 每层 2 次、EP 每层 2 次、PP 每级 1 次；α 默认 3 µs，overlap 默认 0 |
| 每级时间 | `t_stage = max(compute, dram, c2c, pp_act, a2a, fabric) + t_sync` |
| KV 切分 | GQA 每卡 KV = KV · ceil(n_kv/tp)/n_kv / pp（tp > n_kv 时复制）；MLA latent 每个 TP rank 全量；`attn_parallel = dp` 时按 batch 切 KV、注意力权重每卡复制 |
| PP | 每级 L/pp 层；激活经 C2C 发送；decode 用 mb = min(B, pp) 个微批（`decode_mb`），`TPOT_step = max(mb, pp) · t_stage(微批)`，B = 1 时为 pp 级遍历时延；prefill 气泡 `(pp−1)/(mb+pp−1)` |
| MoE / EP | `moe_shard = tp_ep`（默认，专家按 ep 组划分、组内按 tp 切分）或 `ep_all`（专家分布到全部 tp·ep 个 rank）；ep > 1 时注意力在 ep 组间按 DP 运行；all-to-all 每 rank 字节 `2·(D−1)/D·(tokens/D)·top_k·H·act`；decode 读到的本地专家数取期望 `n_local·(1−(1−k/E)^T)`（均匀独立路由） |
| MLA | 注意力权重按 q_lora / kv_lora / qk_nope / qk_rope / v_head 投影逐个建模（DeepSeek-V3 ≈ 670.9B，公开 ≈ 671B）；KV 每层每 token = kv_lora + qk_rope 元素 |
| KV fabric | `none` / `roce_v2` / `ib`，有效带宽 = Gbps/8 × 0.80，每消息时延 5 µs（RoCE）/ 2 µs（IB）；`remote_kv_frac` 份 KV 读走 fabric；PD 分离一次性传输 `t_kv_xfer = lat + KV/BW` |
| 投机解码 / MTP | 验证步 M = B·(k+1)；每步期望 token `E = (1−a^(k+1))/(1−a)`，`TPOT = 步时延 / E`；接受率 a 默认 0.7（assumed）；草稿 = k × (1 层 + LM head)（MTP）或 k · frac · t_stage（独立模型）；开启 MTP 时容量计入 MTP 模块 |
| 容量 / OOM | 每卡 权重 + KV（+ 嵌入，默认复制）> 容量则标记 OOM |

**scale efficiency**：`speedup = t_single / t_multi`，`scale_efficiency = speedup / chips`（理想 1.0）；主指标 LLM = TPOT、视频 = TTFC、蛋白 = time/seq；不计 host / NIC 非理想。

## 6. 服务模型（Pareto / SLO goodput）

- 对当前场景枚举芯片数允许的全部布局（TP×PP×EP；EP 仅 MoE 且整除专家数；LLM 另含注意力 TP | DP），batch 从 1 扫到 **KV 容量上限**（与 OOM 检查同口径），稀疏网格 + 二分。
- LLM：tokens/s/用户 = 1000 / TPOT，tokens/s/芯片 = B · 1000 / TPOT / 芯片数；前沿为非支配点。
- **吞吐口径**（assumed 稳态）：
  - 摊销 prefill（默认）：每请求输出 N 个 token（默认 256）。「分块混合」把 r = b·E/N 个 prompt 折入每个 decode tick（权重只读一次，其余分量相加后取 max）；「独占」时 `TPOT = step/E + B · t_prefill/N`。
  - 上界（仅 decode）：不扣 prefill 占用。
- **SLO goodput**：TTFT ≤ X、TPOT ≤ Y（默认 2000 / 50 ms）下每布局求最大可行 B（假设可行性随 B 单调），报告最优吞吐、配置、最大并发与起作用的约束（TTFT / TPOT / 容量）。
- 未建模：排队 / 到达过程、PD 分离调度、prompt 跨多个 tick 分块（分块混合的 TTFT 偏乐观）。
- 视频 / 蛋白：B 个请求成批，时延 = 批 wall，吞吐 = B / 时延 / 芯片数，SLO 为单一时延上限。

## 7. 视频 DiT / 蛋白质

- **视频**（DiT-like）：patchify → transformer × N_denoise 次前向，每步全量重算 T×T 注意力，**无**去噪步间 KV cache；`TTFC ≈ N_denoise · t_forward`，frames/s = F / TTFC。
- **蛋白**：ESM 式 L-token encoder + 可选 L×L pair（每层 outer-product 式 `2·L²·C_z` + (L², C_z)×(C_z, C_z) GEMM）；pair 激活 `L²·C_z·act_bytes` 随 L² 增长。**不是** AF2 Evoformer（无 triangle / MSA column attention）。
- 多卡：同样 tp/pp/ep；EP 对非 MoE 形状为 no-op；去噪串行使 PP 流水效果差。

## 8. 产品层（MetricsCard、校准、能耗成本）

- `WorkbenchConfig → evaluate_workbench → MetricsCard`，物理复用上面的引擎；chips = 1 与单卡结果一致。
- **CalibrationOverrides**（可选）：`mem_efficiency` / `weight_hide`（历史别名 `mac_efficiency`）/ `npu_mac_efficiency` / `frequency_hz`，见 [`examples/calibration.example.json`](../examples/calibration.example.json)；CLI `--calib`、API `calib: {…}`、Web「校准覆盖」。用于将来接入实测值，不影响默认路径。
- **能耗 / 成本 = assumed stub**（引擎默认关闭，不设 knob 时 `est_*` 全为 0）：
  `est_power_W = chips × P_card × power_util`（P_card 来自 `tdp_w` 或 `watts_per_tops × peak_TOPS`）；
  J/token = P × TPOT / B，J/frame = P × TTFC / (F × B)，J/seq = P × t_seq / B；
  `est_system_cost_usd = chips × (cost_per_card + mem_addon)`；LLM `$ / MTok` = 电费 + capex 摊销（decode-only tokens）。
  不含 host / 冷却 / PUE / DRAM pJ/bit / SRAM / NoC 能耗 / BOM / 良率——**不伪造**这些数。示例假数只在 `examples/energy_cost.example.json`。
- 场景预设（`list-presets`）只捆绑 存储 + 算力 + 芯片数，不含功耗 / 价格。

## 9. 未建模 / 已知局限

- 频率、带宽效率、SRAM 容量、PE / engine 数均未标定（可扫参）；C2C / fabric 带宽与时延为预设。
- 原生低精度 MAC 吞吐（默认只缩字节）；cycle-accurate Softmax / RoPE / LN / NoC。
- MoE 路由不均、热点专家、容量因子 / token drop；MLA 以外的稀疏注意力（DSA、DeepSeek-V4 compress_ratios）、hybrid linear attention 仅元数据。DeepSeek-V4 无 kv_lora 的低秩注意力按字段名推断（assumed）。
- 通信同步 α 不区分拓扑（ring / tree / switch）与消息大小；注意力 DP↔TP 重分片为近似。
- PP 假设 stage 负载均衡、无微批调度开销；投机解码按逐位置 i.i.d. 接受，未计 MTP 层 KV、树形草稿与回滚开销。
- 视频无去噪步间 KV cache / temporal causal；蛋白无 Evoformer triangle / MSA。
- host / NIC 非理想、排队论、PD 分离调度。
- 内置 `illustrative_*` / toy 形状是手算用占位维数，不代表任何真实 checkpoint；公开 HF 维数来自 config.json / model card，gated 模型未收录。

## 10. 典型定性结论

以下结论对 assumed 旋钮稳健（具体数值请在工作台中复现）：

1. **27B 级 decode 的外存字节由权重流主导**；除非权重大规模常驻片上（R → L）或激进量化，KV 在短上下文下可忽略。
2. **小 SRAM 只做 staging 不省权重字节**；R = 1 几乎无用，收益出现在 W_layer 的整数倍膝点（R ≥ L/2、R ≥ L）。
3. **同一算力在 LPDDR 上常为 memory-bound、在 HBM 上常为 compute-bound**（受 M=1 利用率限制）：LPDDR SKU 优先降权重字节，HBM SKU 优先提高 decode 的 M 维利用率（batch、投机解码、多序列）。
4. **长上下文 + weight-only 量化时 KV 读反超权重流**，SRAM 策略应转向 KV scratch。
5. **decode 集合通信体积小**，C2C 带宽通常不是墙，但每次集合通信的固定时延（α）会使 TP 扩展效率明显低于 1。
6. **全远程 KV（disaggregated decode）时 fabric 带宽是最硬的墙之一**；MLA 可把 KV 字节压到 GQA 的约 1/4，但仍远高于卡内时间。
7. **视频 DiT 受 N_denoise × T² 算力限制，蛋白质受 L² pair 内存限制**，与 LLM decode 的带宽墙性质不同。

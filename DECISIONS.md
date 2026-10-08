# DECISIONS.md — 假设 vs 需用户拍板（v0.26）

本文列出当前解析模型中的 **assumed** 量与 **最终需要用户锁定** 的产品/工艺旋钮。
**结论先行：下列任一项均不阻塞继续建模与扫参**；未锁定时继续用表内 assumed 默认值推进。

状态标签：
- `assumed` — 已有默认，可改，未标定
- `derived` — 由形状 / 几何 / 公式推出
- `needs_lock` — 产品或硅后数据最终要拍板，但 **非阻塞**
- `out_of_scope` — 本里程碑刻意不做

---

## 1. 计算侧

| 项 | 状态 | 当前默认 | 说明 |
|----|------|----------|------|
| PE 几何 R×C、n_engines | assumed / needs_lock | SKU 模板给出 | 可用 `sku_baseline` / `sku_100t` / `sku_1p` 或 CLI `--pe` |
| 频率 `frequency_hz` | assumed / needs_lock | 1.0 GHz | **非阻塞**；灵敏度可扫 |
| `mac_efficiency` | assumed | 1.0 | 理想阵列；真实调度损耗未建模 |
| Dataflow | assumed | output-stationary (OS) | 章程固定；改 dataflow 需新公式 |
| `process_nm` | assumed (label) | 4.0 | **仅标签**；不做伪 PDK 功耗/面积 |
| 低精度 MAC 吞吐曲线 | assumed（opt-in） | **默认与 fp16 相同 peak（factor=1.0）** | 默认仍只缩 **字节**（保守）。v0.26 起可选 `--dtype-mac-factor` / `dtype_mac_factors` 表（fp16/fp8/int8/int4）；peak_TOPS×=factor；**用户/assumed，非硅**。默认全 1.0 → handcheck 不变 |
| Softmax / RoPE / LN 时间 | assumed（opt-in） | **默认 0（忽略）** | v0.26 粗算子表：`non_gemm_overhead`∈[0,1] 或 finer `softmax_frac`/`rope_frac`/`norm_frac`；`t_compute' = t_compute*(1+oh)`。**非 cycle-accurate**；默认 off 保守恒/手算 |

## 2. 存储与带宽

| 项 | 状态 | 当前默认 | 说明 |
|----|------|----------|------|
| SRAM 总容量 | assumed / needs_lock | 64 MiB | 扫参 `sweep-sram`；**非阻塞** |
| SRAM 三分区策略 | assumed / needs_lock | 默认 `weight_resident` 最大化 R；另有 `kv_first`/`balanced` | **staging≠resident**：仅 R≥1 省 W 字节；`--sram-policy` |
| weight_hide_factor | assumed | 0.0（可选 0.5） | 仅当 R=0 且 staging≥2×W_layer（dbl-buf）；只折时间不折字节 |
| HBM/LPDDR 通道几何 | assumed | 见 `memory.py` presets | 带宽 **derived**；几何需用户锁到真实封装 |
| BW efficiency | assumed / needs_lock | 0.70 | **非阻塞**；可扫 |
| 外存容量 | assumed | HBM≈96GB / LPDDR≈64GB | 仅容量墙提示用；流量模型不依赖装满 |

## 3. 数值格式（dtype）

| 项 | 状态 | 当前默认 | 说明 |
|----|------|----------|------|
| `weight_bits` / `kv_bits` | assumed / needs_lock | 16 / 16（fp16） | 可独立；均匀预设 fp16/fp8/int8/int4；独立 `w8k16`/`w4k16`/`w8k8`/`w4k8`/`w16k16`。**只改字节** |
| `act_bits` | assumed | 16 | dtype/quant sweep 默认不改激活位宽 |
| 与 MAC 阵列精度绑定 | assumed（opt-in） | 默认未声称（factor=1.0） | 不虚构硅后曲线；用户可注入 assumed factor 表（见 §17） |

## 4. 模型形状与上下文

| 项 | 状态 | 当前默认 | 说明 |
|----|------|----------|------|
| toy / illustrative_27B / illustrative_moe / mla 维数 | assumed / needs_lock | 见 `model_shape.py` | **真实 dims** 最终需用户锁到目标 checkpoint；27B≈28.7B；MoE≈45.5B total / ≈13B active，**非阻塞** |
| illustrative_dit_video | assumed | 见 `workloads.py` | F/lat/patch/H/L/N_denoise；全 attn 每步重算；**非** Sora/CogVideo 声称 |
| illustrative_large_dit | assumed | 见 `workloads.py` | 更大 T/L/N_denoise；sku_100t 上 TTFC 分钟级；**非** MiniMax-H3 声称 |
| illustrative_protein_pair | assumed | 见 `workloads.py` | L_aa + optional pair_dim；Evoformer **重度近似**；**非** AF2/ESM 声称 |
| MoE 路由 / expert residency | assumed | 均衡 top_k；EP 可选 | 单卡流量 = attn + shared + top_k FFN；EP 下 E/ep 存储 + top_k/ep 流；不均路由 **未建模** |
| MLA `kv_lora_rank` | assumed | 512 on illustrative_mla | 联合 latent；KV 字节 = L×rank×kv_bytes；非真实 DeepSeek |
| EP `ep∈{1,2,4,8}` | assumed | 1 | cards=tp×pp×ep；E%ep==0 |
| EP A2A `E_local_adjust` | assumed | 1 | 公式 `2*(ep-1)/ep*V*(top_k/adj)` |
| 上下文长度 | assumed（扫参） | decode ctx 可扫 | `{512,2048,8192,32768,128000}`；长 ctx 下 KV 可读流量单调增 |
| Batch | assumed（扫参） | `{1,2,4,8,16,32}` | OS util 随 M=batch 上升；W 字节/步不随 batch 降 |
| Prefill 注意力因果近似 | assumed | 矩形 × 1/2 | 略悲观于精确三角 |

## 5. SKU / 规模

| 项 | 状态 | 当前默认 | 说明 |
|----|------|----------|------|
| `sku_100t` 单卡峰值 | assumed | ~100 TOPS @1GHz | PE×engines×freq×2 推出 |
| `sku_1p` 是否多芯片 | assumed / needs_lock | **暂作更大单 die** | **是否 1P=多芯片** 需产品定义；与下方 TP/C2C 独立 |
| TP degree `tp∈{1,2,4,8}` | assumed | 1（单卡） | v0.7 MVP；列/行并行切分已文档化 |
| C2C effective BW | assumed | 400 GB/s preset | 另有 100/200/800；**非** PHY 标定 |
| `c2c_hide` | assumed | 0.0 | 与算力重叠份额；非真实调度 |
| Embed under TP | assumed | **replicate** | 可选 `shard`；只影响容量记账 |
| `kv_fabric` | assumed | `none` | `none`\|`roce_v2`\|`ib`；Gbps 级预设 + 固定 µs latency（assumed） |
| `remote_kv_frac` | assumed | 0（本地）；扫参可 1.0 | disagg decode 远程 KV 读占比 |
| `pp∈{1,2,4,8}` | assumed | 1 | 与 TP 正交；cards=tp×pp |
| decode bubble mb=1 | assumed | `(pp-1)/pp` | **局限**：decode PP 气泡重；prefill 可用 mb≥pp |

## 6. 流量与争用策略

| 项 | 状态 | 当前默认 | 说明 |
|----|------|----------|------|
| 权重路径 | derived | R=floor(W_part/W_layer)；path=`hbm_stream_*`/`on_die_resident` | staging 不省字节；无用户 hit-rate |
| KV 片上命中 | derived | 全有或全无（scratch≥工作集） | 无部分 hit-rate 旋钮 |
| 争用模式 | assumed | serialize | 另报 share_fair / lower_bound |
| 激活 spill | assumed（保守） | 工作集>act 分区则每层 2× spill | 权重侧可选 2× staging double-buffer + hide |
| 权重 double-buffer | assumed | R=0 且 staging≥2×W_layer 时 eligible | hide 只影响 mem 时间；resident 才降 W DRAM |

## 7. Scale-up（v0.7）新增旋钮

| 项 | 状态 | 当前默认 | 说明 |
|----|------|----------|------|
| Ring all-reduce 公式 | derived | `2*(tp-1)/tp*V` | 标准 ring；V=激活体积 |
| Collectives / layer | assumed | 2 | Megatron O + down |
| 每卡 FLOPs | derived | total/tp | 切分后 GEMM 工作量 |
| OOM 检查 | derived | W/tp (+embed) + KV/tp vs cap | flag only |

## 8. Scale-up 扩展（v0.8）

| 项 | 状态 | 当前默认 | 说明 |
|----|------|----------|------|
| Fabric BW 换算 | assumed | Gbps/8 × 0.80 | 非 NIC 标定 |
| Fabric latency | assumed | RoCE 5µs / IB 2µs | 每 message 固定加性 |
| PD `t_kv_xfer` | derived | lat+KV/BW | 默认可选独立指标 |
| PP 激活链路 | assumed | 复用 C2C 或 `pp_link` | |
| PP bubble 公式 | assumed | `(pp-1)/(mb+pp-1)` | 非真实调度器 |

## 9. Scale-up 扩展（v0.10）

| 项 | 状态 | 当前默认 | 说明 |
|----|------|----------|------|
| EP A2A 公式 | derived/assumed | `2*(ep-1)/ep*V*top_k` | Megatron-MoE 粗估；adj=1 |
| EP 权重流 | derived | attn/tp + top_k/ep FFN | 均衡；共享复制 |
| MLA KV | derived | L×kv_lora_rank×bytes | 联合 latent |
| CSV export | derived | `export-csv` | 扫参落盘 |



## 11. Workbench / series / geometry（v0.12）

| 项 | 状态 | 当前默认 | 说明 |
|----|------|----------|------|
| `WorkbenchConfig` 字段集 | assumed | 见 `workbench.py` | 产品层；物理复用 evaluate/scaleup |
| `chip_count`→parallel | assumed | tp=chips, pp=ep=1 | 显式 tp/pp/ep 须乘积==chips |
| Series packs 命名 | assumed | `series/dense-27b` 等 | **必须** metadata 含 illustrative/placeholder；**非**厂商 checkpoint |
| 真实 vendor dims | needs_lock later | illustrative placeholders | 不阻塞；锁到真实 dims 后再替换 registry 指向 |
| DRAM `n_packages` 乘 channel | assumed | ×n_packages | 封装映射 **needs_lock later** |
| `n_ranks`≡`n_channels` 别名 | assumed | 1:1 | 真实 rank/channel 拓扑 **needs_lock later** |
| cores→engines 映射 | assumed | n_engines=n_cores；近方形 PE/core | 真实 micro-arch **needs_lock later** |
| tops_per_core 侧取整 | derived | prefer_multiple_of=8 | 16×6.25→56×56，peak≈100.35T |
| workbench video/protein | **已齐**（v0.17–0.20） | llm + video + protein | `evaluate_workbench` / `/api/eval` MetricsCard；video/protein **tp/pp/ep** + collectives + PP bubble |



## 12. Parallel matrix / mem sweep / export（v0.13）

| 项 | 状态 | 当前默认 | 说明 |
|----|------|----------|------|
| tp×pp×ep 枚举 | assumed | 2 的幂整除 chips；非 2 幂用全体约数 | dense ep=1；MoE ep\|E |
| parallel 排序 | assumed | TPOT 升序（可 TTFT） | 产品扫参矩阵 |
| mem geometry 网格 | assumed | ch∈{1,2,4,8}；HBM rate∈{3.2,5.2,6.4}；LPDDR∈{6.4,8.5,9.6} | 展示 BW 墙 FLIP |
| MetricsCard JSON/MD | derived | `--json` / `--md` | 稳定字段落盘 |
| series video/protein 别名 | assumed | `series/video`→dit-video 等 | metadata 仍含 illustrative/placeholder |

## 10. 明确不做（本里程碑）

- 伪 PDK 功耗 / 面积（v0.24 的 energy/cost 只是用户 knob 乘法 stub，见 §15，**不是** PDK）
- 真实 NIC 驱动 / RDMA 语义细节（fabric 仍为解析 BW+latency）
- （已部分覆盖）视频 / 蛋白示意模板仍为 assumed，非硅后标定
- 将 `process_nm` 映射为 PPA
- MoE 负载不均 / token 级精确 A2A 调度

---

## 拍板优先级（建议，均非阻塞）

1. **真实目标模型维数**（替换 illustrative_27B / series packs 指向）— 影响绝对 GB 与 TTFT/TPOT 尺度；**needs_lock later，不阻塞**
1b. **cores↔PE / packages↔channels 产品映射** — workbench 已有 assumed；硅后 **needs_lock later**  
2. **频率 + BW efficiency + SRAM MiB** — 影响墙的左右移动  
3. **dtype 产品默认**（权重/KV 是否同宽、激活是否跟降）  
4. **KV-vs-weight SRAM 偏好**（已有 `--sram-policy`；产品默认是否改 kv_first）  
5. **C2C 有效带宽 + tp/pp 产品默认** — scale-up 墙位置；**非阻塞**  
6. **KV fabric 产品默认**（none vs roce/ib + Gbps 级）— 长 ctx disagg；已有 assumed 预设可扫  
7. **EP / MLA 产品默认**（是否上 EP；KV 是否 MLA 级压缩）— 已有 illustrative 可扫  

**当前无任何项「没有用户输入就无法继续建模」。**


## 7+. Video / protein 多域（v0.10）

| 项 | 状态 | 说明 |
|----|------|------|
| DiT / protein 形状维数 | assumed | 示意占位；真实 checkpoint **needs_lock** 但不阻塞 |
| Denoise KV cache / temporal causal | out_of_scope | MVP 假定每步全量重算 |
| Evoformer triangle / MSA column attn | out_of_scope | pair 仅为 L² GEMM 族近似 |
| 多域硅后标定 | out_of_scope | v0.11 **不**声称标定；版本取 0.11 而非 1.0 |
| compare-domains / large_dit | assumed | 跨域一页对照 + 慢生成叙事占位 |


## 13. HTML report / gap doc（v0.14）

| 项 | 状态 | 当前默认 | 说明 |
|----|------|----------|------|
| `report` HTML | derived | `out/report.html` | 自包含 inline CSS；跑 workbench/compare 快照 |
| REQUIREMENTS_GAP | derived | 中文对照 | 已齐/半齐/缺；真缺口主要为硅后 BW·freq / 真实 dims 锁定 |
| UI 产品化 | **已齐（MVP+）**（v0.17–0.26） | `serve` 本地 knobs + MetricsCard + Compare/Sweep + dual A\|B + calib + assumed compute extras + `?c=` | 非完整 BI；无 CDN；BW/eff/freq 仍 assumed |
| CalibrationOverrides | assumed / optional | `examples/calibration.example.json` | mem_efficiency / mac_efficiency(or weight_hide) / frequency_hz；CLI `--calib`；API `calib:{}`；**非阻塞** |


## 14. Web workbench / calib / dual card（v0.17–0.23）

| 项 | 状态 | 当前默认 | 说明 |
|----|------|----------|------|
| `python3 -m accel_dse serve` | derived | localhost knobs UI | llm/video/protein MetricsCard；零 CDN |
| Compare/Sweep | derived | `POST /api/sweep` ≤32 | CSS bar + CSV/JSON export + baseline Δ% |
| Dual card A\|B | derived | Pin A / Pin B | 字段级 Δ / Δ%；可 last vs baseline |
| Assumed / override sliders | assumed | mem_eff=0.70 · mac_hide=0 · freq=1 GHz | ≠默认显示 **override active**；写入 `?c=` |
| CalibrationOverrides JSON | assumed / optional | `examples/calibration.example.json` | 硅后实测可替换；不阻塞 DSE |


## 15. Energy / cost stub + scenario presets（v0.24）

| 项 | 状态 | 当前默认 | 说明 |
|----|------|----------|------|
| `tdp_w`（W/card） | assumed / user knob | **None（off）** | 有则优先；示例 400 W 仅在 `examples/energy_cost.example.json` |
| `watts_per_tops` | assumed / user knob | **None（off）** | P_card = W/TOPS × peak_tops（per card） |
| `power_util` | assumed | **1.0**（= TDP 上界，保守） | 平均/峰值功耗比；用户自行下调 |
| `cost_per_card_usd` / `mem_addon_usd` | assumed / user knob | **None（off）** | `est_system_cost_usd = chips × (card + mem add-on)` |
| `usd_per_kwh` / `amortize_years` / `duty_cycle` | assumed / user knob | None / None / 1.0 | 仅 LLM `est_usd_per_Mtok`（decode-only tokens；不含 prefill/host/网络/冷却/毛利） |
| 能量单位 | derived from stub | — | llm `P×TPOT/B` J/token；video `P×TTFC/(F×B)` J/frame；protein `P×t_seq/B` J/seq |
| 范围 | out_of_scope | — | 动态/静态功耗拆分、DRAM pJ/bit、SRAM/NoC 能耗、PUE、BOM/良率 — **不伪造** |
| Scenario presets | assumed 档 | edge-lpddr-4x64 · card-hbm-4stack · scaleup-8chip | 只捆 catalog `package_id + compute_id + chips`；**不含**功耗/价格；显式 flag/body 键优先 |

## 16. Scaling efficiency（v0.25）

| 项 | 状态 | 当前默认 | 说明 |
|----|------|----------|------|
| `scale_efficiency` | derived | chips=1 → **1.0** | `(t_single / t_multi) / chip_count`；理想 1.0 |
| `speedup` | derived | chips=1 → **1.0** | `t_single / t_multi` |
| Primary metric | derived | LLM TPOT / video TTFC / protein time_per_seq | 与 workbench / sweep primary 一致 |
| Baseline | derived | 同 config、`chips=tp=pp=ep=1` | 每次 multi-card eval 再跑一次 single-card |
| Host / NIC 非理想 | out_of_scope | — | **不**建模；C2C/PP-bubble/EP 已在 `t_multi` |
| chips=1 标注 | note | `"single-card"` | 优先 1.0 + note，而非 null |

## 17. Assumed compute extras（v0.26）

| 项 | 状态 | 当前默认 | 说明 |
|----|------|----------|------|
| `non_gemm_overhead` | assumed / opt-in | **0（off）** | 粗 Softmax/RoPE/LN/misc：`t_compute' = t_compute * (1 + oh)`；非 cycle-accurate；默认 off 保 handcheck |
| `softmax_frac` / `rope_frac` / `norm_frac` | assumed / opt-in | None | 若任一提供，三者之和替换 coarse knob |
| `dtype_mac_factors` | assumed / opt-in | **全 1.0 / None** | 映射 fp16/fp8/int8/int4（或 bits）；peak_TOPS×=factor，t_compute÷=factor；**用户/assumed，非硅** |
| CLI | derived | — | `--non-gemm-overhead` / `--softmax-frac` / `--rope-frac` / `--norm-frac` / `--dtype-mac-factor '{…}'` |
| API / UI | derived | — | body `non_gemm_overhead` / finer / `dtype_mac_factors`；Web Assumed compute extras |
| 守恒 / 手算 | derived | defaults | overhead=0 且 factor=1.0 → 与 v0.25 bit-identical |

## 18. 存储目录重建 + 两处乐观假设修正（v0.29）

### D-0.29-1 存储目录：结构化选择 + 来源标签（最弱项规则）

| 项 | 决定 | 说明 |
|----|------|------|
| 选择单位 | **封装 / 模组 / 堆**（`n_units × unit_width_bits`，`unit_kind ∈ {package, module, stack}`） | 不再以 die 或「channel」为选择粒度；`ExternalMemory.n_channels` 仅为向后兼容字段名 |
| LPDDR5 / 5X 板载封装 | **x64 = 4×16-bit 通道**（默认）；x32；LPDDR5X x96（Micron 6 通道）= **已发布** | 速率：LPDDR5 {5500, 6400}；LPDDR5X {7500, 8533, 9600, 10667}（8533 = JEDEC 明文，其余疑似 JEDEC） |
| LPDDR6 板载封装 | **x96 = 4×24-bit 通道 = 8×12-bit 子通道**；x48 = 推测 | 速率 {10667 JEDEC, 12800 疑似 JEDEC, 14400 已发布}；**payload 8/9**（BL24 中 256/288 为数据）单独计，与效率分开 |
| LPDDR6 14400 | 标 **已发布**（而非 JEDEC） | JESD209-6 定义到该档，但目前只有论文硅 / 产品页上限；按「最弱项」规则若标 JEDEC，会让纯速率组合被误判为量产级 |
| 模组 | SOCAMM2（LPDDR5X，128-bit，JESD328，48–192 GB/条，≤8 条）；LPCAMM2（128-bit，≤2 条）；LPDDR6 CAMM2 = 推测 | 128-bit 只允许出现在模组形态 |
| HBM | 堆宽 1024（HBM3/3E）/ 2048（HBM4/4E）；容量 = 层数 × die 密度 / 8；堆数 {1,2,4,5,6,8,12}，16 = 推测 | HBM3E {8000, 9200 默认, 9600, 9800}；HBM4 {8000 JEDEC, 10000/11000/11700 量产, 13000 已发布}；HBM4E 无 JEDEC 标准 {12800 推测, 14000/16000 送样} |
| 容量单位 | 厂商标称 GB = **2³⁰ B**（`capacity_bytes = GB × 2³⁰`） | `MetricsCard.capacity_GB` 仍为十进制（API 兼容）；UI 按比例换算后与存储摘要同单位显示 |
| 来源标签 | 每个组件（速率 / 位宽 / 容量 / 数量）一个标签；组合 = **最弱项** | 顺序 JEDEC > 疑似 JEDEC > 厂商量产 > 送样 > 已发布 > 推测；允许推测组合（DSE 需要），但 UI 徽标 / 卡片 / CSV `mem_tag` 必标 |
| 提示 | 板载 LPDDR 总线 > 576-bit、HBM > 12 堆 | 不阻止计算，只提示（Grace ≈ 480-bit；Vera 用 SOCAMM2 到 1024-bit；MI455X = 12 堆） |
| 效率 | 仍为假设 0.70（可调） | 有效带宽 = 原始 × payload × 效率 |
| 向后兼容 | 所有 ≤0.28 id 解析到最近的新配置并附说明 | `lpddr6_{n}x24_{rate}` → x96 封装数取最近值（偏保守）；`lpddr_{n}x64_{rate}` → LPDDR5（≤6400）或 LPDDR5X；`hbm_{gen}_{n}s` → 该代默认层数 / 密度 |
| 预设 | `edge-lpddr-4x64`、`edge-lpddr6-4x96`、`card-hbm-4stack`（HBM3E 4×12H 144 GB）、`card-hbm3e-8x12h`（288 GB）、`server-socamm2-8`、`scaleup-8chip` | 预设 id 不变；新增 3 个；全部为可量产或已发布组合；默认场景（HBM3E 8×12H）不 OOM |

**`research/memory_specs_2026-10.md` §5 问题 → 处理**：
1 LPDDR6 x24 die 粒度 → x96 封装（x48 推测）· 2 无 payload → 8/9 · 3 LPDDR6 缺 12800 / 14400 标签 → 三档带标签 ·
4 6400 标为 LPDDR5X → 拆 LPDDR5 / LPDDR5X · 5 只有 x64 → {32, 64, 96} · 6 容量按位宽线性 → 每封装容量列表 ·
7 HBM 容量偏低 → 层数 × die 密度 · 8 HBM4E 过时 → 14 / 16 Gbps 送样 · 9 HBM4 只有 8.0 → 五档 · 10 HBM3E 只有 9.2 → 四档 ·
11 堆数缺 1/5/12/16 → 补齐（16 推测）· 12 `HBM_PRESET` 5.2 GT/s / 96 GB → HBM3E 8×36 GB @9.2 · 13 `LPDDR_PRESET` 8.5 → 8533、128 GB ·
14 扫描速率与代际脱钩 → 速率依附代际（`sweep_specs`）· 15 channel / package 概念混用 → `n_units` / `unit_width_bits` / `unit_kind` ·
16 文档 x24 表述 → 已更正（本文件、REQUIREMENTS_GAP、package_ranges docstring、report 入门）· 17 UI 单下拉 → 多选择器 + 旧 id 映射。

### D-0.29-2 修正 1：暴露的通信同步（α）

| 项 | 决定 | 说明 |
|----|------|------|
| 模型 | `t_stage = max(compute, mem, c2c, pp_act, a2a, fabric) + t_sync` | 带宽项仍取 max（假设可重叠）；**固定时延不可被带宽隐藏**，因此叠加 |
| `t_sync` | `n_sync × α × (1 − overlap)` | 每级每 token：TP 每层 2 次 all-reduce（`2·L/pp`）、EP（MoE）每层 2 次 all-to-all、PP 每级 1 次收发 |
| α 默认 | **3 µs / 次**（「假设」，合理范围 2–5 µs） | 启动 + 同步 + 跳数；未标定；UI 抽屉 / CLI `--c2c-latency-us` / API `c2c_latency_us`（别名 `alpha_us`） |
| overlap 默认 | **0 = 完全暴露** | 用户可设 0–1 表示被计算隐藏的比例 |
| 适用范围 | LLM 与视频 / 蛋白质多卡 | 单卡 n_sync = 0 |
| 兼容 | α = 0 → 与 ≤0.28 逐位一致 | 有测试；默认 α > 0，所以 scale_efficiency 不再「完美 TP ≈ 1.0」 |

### D-0.29-3 修正 2：KV 按 TP 切分 / 注意力 DP

| 项 | 决定 | 说明 |
|----|------|------|
| GQA / MHA | 每卡 KV = KV × ceil(n_kv / tp) / n_kv / pp | tp > n_kv 时 KV 头被复制（`kv_replication > 1`） |
| MLA latent | **每个 TP rank 全量**（1 / pp） | latent 无头维，不能按 TP 切；deepseek tp=8 时每卡 KV ×8 |
| `attn_parallel = dp` | KV 按 batch 切分：ceil(B / tp) / (B·pp)；**注意力权重每卡复制**（容量 + decode 权重流） | 注意力计算按 batch 拆分 ≈ 每卡 FLOPs 不变；DP↔TP 重分片集合通信近似为每层 2 次（计入 n_sync） |
| 影响 | 容量 / OOM 与 decode KV 流量同步更新 | prefill / decode 同口径 |
| 已知限制（v0.30 已修正） | 0.29 中 MLA 注意力权重按 dense MHA 形状建模（~704M / 层 vs 真实 ~187M） | v0.30 按 MLA 投影逐个建模，见 D-0.30-1 |

## 19. MLA 权重 / 容量单位 / 吞吐–交互帕累托 + SLO goodput（v0.30）

### D-0.30-1 注意力权重按投影建模（MLA / 低秩回退 / GQA）

| 项 | 决定 | 说明 |
|----|------|------|
| `attn_kind = mla` | 有 `kv_lora_rank` 且有 nope / v 维：q_a H→q_lora（无 q_lora 时 q: H→nh·(nope+rope)）、q_b q_lora→nh·(nope+rope)、kv_a H→kv_lora+rope、kv_b kv_lora→nh·(nope+v)、o nh·v→H | DeepSeek-V3：187.1M / 层（0.29：704.6M） |
| `attn_kind = lowrank` | 无 `kv_lora_rank`（DeepSeek-V4 配置）：低秩 Q（q_lora）、K/V = H→n_kv·head_dim、O = nh·d·o_lora + o_groups·o_lora·H | 字段含义按名字推断「假设」；V4 的 compress_ratios / 稀疏注意力未建模 |
| `attn_kind = gqa` | 0.29 公式不变（Q/O = H·nh·d，K/V = H·n_kv·d） | 示意形状（含 `illustrative_mla`，无 nope / v 维）结果不变 |
| 前置稠密层 | HF `first_k_dense_replace` → `n_dense_layers` 层用 `dense_intermediate` 稠密 SwiGLU | 每层方法返回层平均；总量由 `body_weight_params()` 精确求和；EP 下稠密层按 /tp、MoE 层按 EP 本地 |
| LM head | `tie_word_embeddings=false` → 嵌入 + LM head = 2·V·H（所有 HF LLM 包） | decode 权重流仍不计 LM head（既有简化） |
| MLA KV | 每层每 token = kv_lora + qk_rope（DeepSeek 576 元素） | 0.29 只计 512 → KV +12.5% |
| 不计 | MTP / nextn 层 | DeepSeek-V3 公开 671B 不含 MTP 层；手算 670.9B |

### D-0.30-2 容量单位

- `capacity_GB` / `capacity_needed_GB`（MetricsCard、API、CSV、markdown）= 字节 / 2³⁰ = 厂商标称 GB（HBM3E 12H×24Gb = 36 GB/堆），与 UI、存储目录一致。
- 十进制值保留为 `capacity_GB_decimal` / `capacity_needed_GB_decimal`（字节 / 10⁹）。手动几何 `capacity_GB` 输入同样按 2³⁰ B。
- LLM 卡片导出 `capacity_needed_GB` = 每卡权重 + KV（与 `capacity_check` 同口径；不含激活）。

### D-0.30-3 吞吐–交互帕累托 + SLO goodput 的稳态假设（「假设」）

| 项 | 决定 |
|----|------|
| 扫描范围 | 当前场景；布局 = `enumerate_parallel_combos(chips)`，EP>1 仅 MoE 且 EP 整除专家数，PP ≤ 层数；LLM 注意力 TP / DP（tp>1） |
| batch 网格 | 1, 2, 3, 4, 6, 8, 12, 16, 24 … ≤ B_max，外加 B_max；SLO 边界处二分到整数 |
| B_max（容量上限） | 每卡 权重 + KV(B, ctx = decode_seq_len) ≤ 容量；与 `scaleup.capacity_check` 完全同口径（测试逐一核对 OOM 边界）；搜索上限 LLM 4096、视频 / 蛋白 256 |
| 指标（LLM） | tokens/s/用户 = 1000 / TPOT(B)；tokens/s/芯片 = B × 1000 / TPOT(B) / 芯片数（仅 decode） |
| TTFT | **TTFT_eff(B) = TTFT_prefill(1 条 prompt，同布局) + TPOT(B)**：新请求最多等待一个在途 decode 步，然后单独 prefill（不与 decode 交织 / 不分块） |
| 未建模 | prefill 占用对 decode 吞吐的扣减（→ tokens/s/芯片 为上界）、排队 / 到达过程、chunked prefill、PD 分离、投机解码 |
| PP | 沿用评估器 decode 气泡（`decode_mb`，默认 1 → (pp−1)/pp）→ PP 布局吞吐偏保守 |
| 前沿 | 仅容量可放下的点；x 更好（LLM 越大越好；视频 / 蛋白时延越小越好）且 y 更大者支配；重复点只保留一个 |
| goodput | 每布局求满足 SLO 的最大 B（假设可行性随 B 单调）→ 该布局最优 y（并列取大 B）；全局取最大 y。约束判定：B = B_max → 容量（达到搜索上限则「批量上限」）；否则看 B+1 违反哪项（TTFT / TPOT）；全不可行时报告所有布局在 B=1 共同违反项 |
| 视频 / 蛋白 | B 个请求成批：时延 = 该批 wall（TTFC / 单批时间），吞吐 = B × 单位 / 时延 / 芯片数；SLO 为单一时延上限（默认 30 s / 10 s） |
| KPI 卡 | 「SLO 吞吐 / 芯片」只算**当前布局**（轻量，每次评估后刷新）；标签页算全部布局 |

## 20. TP×EP / PP 微批 / prefill 摊销 goodput / 投机解码 / LM head（v0.31）

### D-0.31-1 MoE 专家切分（TP×EP 与 EP 全卡）
- **tp_ep（默认）**：tp·pp·ep = 芯片数，专家按 ep 组划分、组内每个专家按 tp 切分（行 / 列切 SwiGLU）；共享专家与前置稠密层按 tp 切分。
- **ep_all**：专家分布在 tp·ep 个 rank 上，不切分；注意力仍按 TP（或 DP）。需 tp·ep | E。
- 两者存储相同，差别在于 all-to-all 的度数（ep 对 tp·ep）与每 rank GEMM 形状。
- 注意力：MoE ep>1 时在 ep 组之间按数据并行（DeepSeek / SGLang 的「DP attention + EP」惯例），因此 KV 与注意力 batch 在 ep 上切分；组内 attn=tp 时 MLA latent 复制，attn=dp 时继续按 batch 切。
- decode 专家权重流取期望专家覆盖：n_local·(1−(1−k/E)^T)，T = 本 tick 经过该 stage 的 token 数（「假设」均匀独立路由；真实路由偏斜会读得更少，热点会让单卡更慢）。
- 拒绝：0.30 只按 ep 划分专家、不按 tp 切分，导致 tp>1·ep>1 布局把专家存 tp 份。

### D-0.31-2 PP decode：微批填满流水
- mb = `decode_mb`（0 = 自动 min(B, pp)），微批大小 b = ceil(B/mb)。
- 每用户 token 间隔 TPOT_step = max(mb, pp) · t_tick，t_tick = t_stage(b) + t_draft。
- mb ≥ pp：流水填满，按最慢 stage 节拍；mb < pp：受 pp 级遍历时延限制。B = 1 → pp · t_stage（单请求时延语义不变）。
- 吞吐 = B / TPOT_step，不再是 1/pp。prefill 气泡仍为 (pp−1)/(mb+pp−1)。
- `decode_mb = 1` 复现 0.30（整批走一遍流水）。
- 未建模：stage 间负载不均、微批调度开销、跨微批 KV 交错访问。

### D-0.31-3 prefill 摊销 goodput（「假设」）
- 每个请求输出 N 个 token（默认 256），prompt 长度 P 取场景值；稳态下每产出 B·E 个 token 就有 B·E/N 个新请求进入。
- **分块混合**（默认）：每个 decode tick 折入 r = b·E/N 个 prompt。各分量相加：计算、非权重内存（KV / 激活）、c2c、pp 激活、a2a。权重只读一次（与 decode 共享）。tick' = max(各分量) + sync + draft，TPOT = slots · tick' / E。
  - 内存受限时 prompt 计算可被权重流隐藏（prefill 份额 → 0），这是分块 prefill 的真实好处。
  - TTFT ≈ pp · tick(dec + 整条 prompt) + tick'（不建模把 prompt 拆成多个块、跨多个 tick 的情形，偏乐观）。
- **独占**：prefill 单独占用芯片。TPOT = step/E + B · t_prefill_stage(单条)/N，TTFT = 单条 prefill + 一个 step。
- 无排队 / 到达过程 / PD 分离；**上界**口径（仅 decode，0.30）保留，可选。

### D-0.31-4 投机解码 / MTP（「假设」接受率）
- 验证步把 k+1 个位置一起算（M = B·(k+1)）；KV 上下文每序列读一次，写 k+1 个；MoE 专家覆盖按 B·(k+1) 个 token 计。
- E[每步 token] = Σ_{i=0..k} a^i = (1−a^(k+1))/(1−a)（逐位置独立接受，含 bonus token）；TPOT = 步时延 / E；步时延 = 验证 + 草稿。
- 草稿：
  - mtp：k 次 1 层（该模型层形状）+ LM head 的前向（pp = 1，草稿在末 stage 串行）；
  - model：k · frac · t_stage（frac = 草稿模型相对目标步的代价）。
- 容量：mtp 时末 stage 卡额外存 max(n_mtp_layers, 1) 个 MTP 模块（1 层，按主体方式切分 + eh_proj 2H²）。
- 未建模：MTP 层 KV、eh_proj 的读流量、树形草稿、拒绝后 KV 回滚开销。
- 结论：内存受限（小 B）时 k=1 约 −39% TPOT；算力受限（大 B）时变慢，帕累托不会自动打开投机。

### D-0.31-5 LM head 与批量上限
- 工作台（产品路径）每个 decode 步计入 LM head：读 V·H，并做 GEMM (B·q, H, V)。
- 注意力 TP 下，TP 组共享 token，因此按词表并行，每 rank 只读 / 算 V/tp。即便存储策略为 replicate 也如此，因为任何实现都能只读自己那一片。
- 注意力 DP 下，每 rank 的 token 不同：复制存储时每 rank 全读，shard 时 V/tp。
- PP：「假设」切分按 head 均衡（末 stage 少放层）→ head 代价按 /pp 摊到各 stage。若不均衡，最慢 stage 会把 head 计入 pp 次。
- 原始引擎 `EvalConfig.count_lm_head` 默认关闭，0.1x 以来的手算测试保持不变；`single_card_reference` 打开它，与工作台保持一致。
- 帕累托 LLM batch 上限 4096 → 65536：容量上限二分查找 + 1-2-3-4-6-8 稀疏网格，评估次数随 log(B) 增长。deepseek-v3 ×8 全部 28 个布局 ≈ 0.5 s。

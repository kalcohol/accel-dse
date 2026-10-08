# FINDINGS.md — 硅侧备忘（v0.14，assumed / uncalibrated）

> 标签约定：下列绝对数均来自当前解析模型的 **assumed** 旋钮 + **derived** 公式；
> **不是** 硅后标定、也不是某家 checkpoint / PDK 声明。示意形状 `illustrative_27B` / `illustrative_moe` / `illustrative_mla` 仅为 DSE 占位。

---

## 1. 章程：单卡 NPU + HBM/LPDDR 记账

- **Inference-only**、固定 dataflow（OS 脉动阵列），非 GPGPU。
- **单卡**：片上 SRAM（三分区）+ **纯 HBM 或纯 LPDDR**（二选一）。
- 外部流量由 **tiling + 分区容量** 推导，默认路径不要求用户填 cache hit-rate。
- Weight 读与 KV 读 **争用同一外部带宽**；时间取 roofline `max(compute, mem)`。
- SKU 峰值模板（`sku_100t` / `sku_1p`）由 `PE×engines×freq×2flop/MAC` **derived**。

---

## 2. Decode 权重流主导（27B 级）

在默认 **64 MiB SRAM、R=0（staging-only）**、fp16、`illustrative_27B` 下：

| 量 | 约数 |
|----|------|
| 权重体（层堆） | ~52.8 GiB 量级（52800 MiB = L×W_layer；W_layer≈1320 MiB） |
| Decode 每步 W DRAM | ≈ L×W_layer ≈ **55 GB**（整模流式） |
| KV @ ctx=512 | ≪ W（~0.08 GB 量级） |

**结论**：除非 **大规模 on-die resident（R→L）** 或 **激进权重量化（W 字节大幅下降）**，
27B 级 decode 的外部字节由 **权重流** 主导。

---

## 3. Staging ≠ Resident；R=1 几乎无用；全驻膝点

| 概念 | 对 decode W DRAM 的影响 |
|------|-------------------------|
| **Staging**（R=0，哪怕 1×/2× W_layer） | **不减字节**：仍 L×W_layer/token；2× 仅使 dbl-buf + 可选 hide（折时间不折字节） |
| **R=1** | W → (L−1)×W_layer ≈ 仍 ~97.5% 流量 → **几乎无用** |
| **R≥L/2** | 约减半；膝点 ≈ **26400 MiB** @ fp16 |
| **R≥L（full resident）** | 稳态 W DRAM=0；膝点 ≈ **52800 MiB ≈ 52.8 GiB** @ fp16；int4 ≈ **13.2 GiB** |

Prefill / cold-start 即使 R=L 仍装载整模一次。

---

## 4. sku_100t：LPDDR 常 mem-bound；HBM 常 compute-bound

同一形状、同一 55 GB 权重流、OS util(M=1)≈1/R：

| SKU × 外存 | Decode 墙（典型） | 含义 |
|------------|-------------------|------|
| `sku_100t` @ **LPDDR**（~381 GB/s eff） | **memory-bound** | 权重流时间 > 低 util 下的算力时间 |
| `sku_100t` @ **HBM**（~TB/s 级） | **compute-bound** | 带宽够；卡在 OS 小 M 利用率 |
| 更大 PE / engines / hide | 可把墙左右推 | 直至再次撞带宽或算力 |

**SKU 分裂含义**：LPDDR SKU 卖的是「带宽墙下的能效/成本」；HBM SKU 更吃「把 decode util 做高 / 藏权重流」。

---

## 5. 长上下文：W 重度量化后 KV 反超

| 配置 | Decode KV/W（约） |
|------|-------------------|
| fp16 / w16k16 @ 128k | KV 可观但仍常 < W |
| **w4k16 @ 128k** | **KV/W ≳ 1.5** → KV 反超权重流 |

Weight-only 量化把 W 字节压下去后，长 ctx 的 KV 读成为新的带宽主角 → SRAM 策略可能要从「尽量抬 R」转向 **KV scratch / kv_first**。

---

## 6. 对 NPU IP 的含义

1. **Weight delivery path** 是一等公民：专用流式通道、dbl-buf、与 PE 的 overlap（hide），否则 27B 级 decode 在 LPDDR 上会被权重流钉死。
2. **SRAM 预算哲学**：小 SRAM 只做 staging → **不省 W 字节**；要字节收益必须按 **W_layer 膝点** 规划 resident（或接受激进 W quant）。R=1 性价比极差。
3. **HBM vs LPDDR SKU 分裂**：
   - LPDDR：优先减 W 字节（quant / 部分 resident）+ 带宽效率；
   - HBM：优先抬 decode **M 维利用率**（batch、投机、多序列）或加大 PE 有效宽度。
4. **MoE（illustrative）**：总参 ≫ 激活参；单卡流量按 **top_k 专家 FFN + 共享 + 稠密注意力**。v0.9 **EP**：专家存 E/ep；激活流 top_k/ep；A2A=`2*(ep-1)/ep*V*top_k`（E_local_adjust=1）。

---

## 7. 明确非声称（non-claims）

- **无 PDK**：`process_nm` 仅为标签；无面积/功耗推导。
- **C2C / TP / PP / EP / KV fabric**：v0.7+ TP+C2C；v0.8 IB/RoCE + PP；v0.9 **MoE EP** + **MLA 压缩 KV**。`sku_1p` 仍可作更大单 die 灵敏度。
- **非真实 checkpoint**：`illustrative_27B` / `illustrative_moe` / `illustrative_mla` / `illustrative_dit_video` / `illustrative_protein_pair` 为占位维数（非厂商模型质量声称）。
- **低精度 MAC**：dtype/quant **只缩字节**；峰值 MAC/FLOPs 不变（对低精度偏保守）。
- **MoE**：假设 **均衡路由**；EP 下专家按 E/ep 分片，A2A 为粗解析估计。

---

## 8. Scale-up：TP + C2C（v0.7 MVP，assumed）

多卡 **tensor-parallel**（`tp∈{1,2,4,8}`）+ **chip-to-chip** 集合通信：

| 项 | 假设（须标 assumed） |
|----|----------------------|
| 切分 | Megatron 风格：Q/K/V/gate/up **列并行**；O/down **行并行** |
| 每层 collective | **2× all-reduce**（attn O 后 + MLP down 后） |
| Ring 字节 | 每 rank：`2*(tp-1)/tp * volume`；`volume=B·S·H·act_bytes`（decode S=1） |
| 算力 / 权重流 / KV | 每卡 ≈ 单卡总量 `/tp`；embed 默认 **replicate**（容量） |
| C2C BW | 预设 100/200/400/800 GB/s **effective**（assumed）；`c2c_hide` 默认 0 |
| 墙 | `TPOT/TTFT ≈ max(per-card compute, per-card DRAM, C2C)` |
| KV fabric | v0.8：`none\|roce_v2\|ib` 见 §9 |

**示意扫参表**（`illustrative_27B` @ `sku_100t`，prompt=ctx=512，C2C=400 GB/s，hide=0，R=0 staging；**assumed**）：

| tp | mem | TPOT ms | wall | c / dram / c2c ms | coll MB | 每卡容量 W+KV |
|----|-----|---------|------|-------------------|---------|---------------|
| 1 | HBM | 33.06 | compute | 33.06 / 14.88 / 0 | 0 | ~57.5 GB |
| 1 | LPDDR | 145.61 | memory | 33.06 / 145.61 / 0 | 0 | ~57.5 GB |
| 2 | HBM | 16.53 | compute | 16.53 / 7.44 / 0.003 | 1.31 | ~29.8 GB |
| 4 | HBM | 8.27 | compute | 8.27 / 3.72 / 0.005 | 1.97 | ~16.0 GB |
| 8 | HBM | 4.13 | compute | 4.13 / 1.86 / 0.006 | 2.29 | ~9.0 GB |
| 8 | LPDDR | 18.20 | memory | 4.13 / 18.20 / 0.006 | 2.29 | ~9.0 GB |

Prefill 时 C2C 更重（S=512）：tp=8 @HBM 时 c2c≈2.94 ms，仍常低于 compute。Decode 集合通信体积小（S=1），400 GB/s 下几乎不挡 TPOT——**墙仍在卡内算力或外存**。更慢的 assumed C2C（100 GB/s）或更大 batch 才会让 c2c 抬头。

CLI：`python -m npu_dse sweep-tp` / `scan-scaleup`。OOM：每卡 weight+KV > `mem.capacity` 则 flag。

**非声称**：非真实 C2C PHY。IB/RoCE / EP 为 assumed 解析模型（见 §9 / §11）。

---

## 9. KV fabric：IB / RoCE（v0.8 MVP，assumed）

远程 KV 与 PD 分离传输的 **解析** 模型（非真实 NIC/驱动标定）：

| 项 | 假设 |
|----|------|
| 模式 | `none`（默认）\| `roce_v2` \| `ib` |
| 预设 | 100/200/400 Gbps 级 → eff GB/s = (Gbps/8)×0.80；RoCE lat≈5µs，IB≈2µs / message |
| Decode 远程读 | `remote_kv_frac` 份 KV **读**走 fabric；本地 DRAM 只计剩余读 + 写；`t_fabric=lat+bytes/BW`；墙=`max(compute,local_dram,c2c,fabric)` |
| PD one-shot | 预填后整份 KV 传输 → 独立指标 `t_kv_xfer`（可选并入 TTFT） |

**示意 @ 128k**（`illustrative_27B` @ `sku_100t` HBM，`remote_kv_frac=1`，tp=1；**assumed**）：

| preset | BW GB/s | lat µs | remote KV GB | t_fab ms | t_xfer ms | TPOT ms | wall |
|--------|---------|--------|--------------|----------|-----------|---------|------|
| none | — | — | 0 | 0 | 0 | 34.7 | compute |
| roce_100g | 10 | 5 | 20.97 | 2097 | 2097 | 2097 | fabric |
| roce_200g | 20 | 5 | 20.97 | 1049 | 1049 | 1049 | fabric |
| roce_400g | 40 | 5 | 20.97 | 524 | 524 | 524 | fabric |
| ib_200g | 20 | 2 | 20.97 | 1049 | 1049 | 1049 | fabric |
| ib_400g | 40 | 2 | 20.97 | 524 | 524 | 524 | fabric |

长 ctx + 全远程 KV 时 fabric 远大于卡内算力/DRAM → **这是当前模型最硬的墙之一**：
@128k GQA fp16 全远程读 ≈21 GB → roce_200g 上 **t_fab≈1049 ms** vs 卡内 compute≈35 ms（**~30×**）。
**disagg decode 必须盯网络 BW**；latency（µs）在 GB 级传输上几乎淹没。`none` 与本地路径字节/时间一致。

CLI：`python -m npu_dse sweep-kv-fabric`。

---

## 10. Pipeline parallel（v0.8 light，assumed）

| 项 | 假设 |
|----|------|
| `pp∈{1,2,4,8}` | 总卡数 = tp×pp（DP=1） |
| 层切分 | 每 stage ≈ L/pp 层；算力/权重流/KV ≈ / (tp·pp) |
| 激活发送 | 每边界 `B·S·H·act_bytes`；(pp−1) 次；走 C2C 或 `pp_link` |
| Decode bubble | **mb=1** → bubble=`(pp-1)/pp`；墙 = stage/(1−bubble)。**局限**：decode PP 常依赖串行，mb=1 时墙回到 ≈ 仅 TP 的时间 |
| Prefill | 允许 mb≥pp → bubble=`(pp-1)/(mb+pp-1)` 粗降气泡 |

**示意 TP×PP**（HBM，ctx=512，decode mb=1，C2C=400；**assumed**）：

| tp | pp | cards | bubble | TPOT ms | stage ms | wall |
|----|----|-------|--------|---------|----------|------|
| 1 | 1 | 1 | 0 | 33.06 | 33.06 | compute |
| 1 | 4 | 4 | 0.75 | 33.06 | 8.27 | bubble |
| 2 | 1 | 2 | 0 | 16.53 | 16.53 | compute |
| 2 | 4 | 8 | 0.75 | 16.53 | 4.13 | bubble |
| 4 | 1 | 4 | 0 | 8.27 | 8.27 | compute |
| 4 | 8 | 32 | 0.875 | 8.27 | 1.03 | bubble |

解读：mb=1 时 PP **不缩短** decode TPOT（气泡吃掉 stage 缩短）——**PP bubble 是 decode 路径的第二堵硬墙**；真正收益在容量（每卡 W+KV /pp）与 prefill（mb≥pp）。CLI：`sweep-parallel` / `sweep-pp`。

---

## 11. MoE expert-parallel（v0.9，assumed）

| 项 | 假设 |
|----|------|
| `ep∈{1,2,4,8}` | 总卡数 = **tp×pp×ep**（DP=1）；`E % ep == 0` |
| 专家分片 | 每 rank 存 **E/ep** 专家（+ shared 复制） |
| 激活权重流 | 每 rank：attn/tp + shared + **(top_k/ep)×FFN**（均衡） |
| A2A（Megatron-MoE 粗估） | `dispatch+combine = 2*(ep-1)/ep * B*S*H*act_bytes * (top_k/E_local_adjust)`；默认 **E_local_adjust=1** |
| KV | **不**按 EP 分片（仍 /tp/pp） |

**示意**（`illustrative_moe` @ sku_100t，ctx=512，tp=pp=1，C2C=400；**assumed**）：

| ep | cards | Wcard GB | a2a MB | TPOT HBM ms | wall | TPOT LPDDR ms |
|----|-------|----------|--------|-------------|------|---------------|
| 1 | 1 | 25.00 | 0 | 15.22 | compute | 65.87 |
| 2 | 2 | 14.18 | 0.66 | 7.61 | compute | 37.45 |
| 4 | 4 | 8.77 | 0.98 | 3.81 | compute | 23.24 |
| 8 | 8 | 6.06 | 1.15 | 1.90 | compute | 16.14 |

Decode S=1 时 A2A 体积小，400 GB/s 下几乎不挡；EP 收益主要在 **每卡权重流 / 容量**。CLI：`sweep-moe --ep 1 2 4 8`。

---

## 12. MLA 压缩 KV（v0.9，assumed）

`illustrative_mla`：与 27B 同 body，**kv_lora_rank=512** 联合 latent（非 DeepSeek 声称）。

| 形状 | kv/tok | @128k KV DRAM | vs GQA |
|------|--------|---------------|--------|
| GQA 27B | 160 KiB | ≈20.97 GB | 1.0× |
| MLA r512 | 40 KiB | ≈5.24 GB | **0.25×** |

**远程 fabric @128k roce_200g**（remote_kv_frac=1；**assumed**）：

| 形状 | remote KV | t_fab ms | TPOT ms | wall |
|------|-----------|----------|---------|------|
| GQA | 20.97 GB | 1048.6 | 1048.6 | fabric |
| MLA | 5.24 GB | 262.1 | 262.1 | fabric |

MLA 把 fabric 墙压到 ~1/4，但仍远高于卡内 compute——长 ctx disagg 仍受网络 BW 约束。本地 LPDDR @128k：MLA 因 KV 更小，mem 时间从 ~200 ms → ~159 ms。CLI：`sweep-mla`。

---


## 13. CSV 导出

`python3 -m npu_dse export-csv --out …` 写出：`sweep_{sku,dtype,ctx,sram,tp,kv_fabric,moe,mla,video,protein}.csv` + `compare_domains.csv`。

---

## 14. 下一阶段自然标定需求（非阻塞清单）

下列为 **下一轮若要对绝对 ms/GB 更敢报数** 的自然标定项（当前 assumed 默认仍可扫参，**不阻塞建模**）：

1. **真实目标 checkpoint 维数**（替换 illustrative_27B / moe / mla）
2. **C2C / IB / RoCE 有效 BW + 延迟**（硅后或 NIC 标定；替换 0.80×Gbps/8 与固定 µs）
3. **频率 + 外存 BW efficiency + SRAM MiB**（推墙左右）
4. **MoE 路由不均 + EP A2A 实测因子**（替换 E_local_adjust=1 粗估）
5. **MLA 真实 c_kv / rope 附加字节**（若非联合 latent）
6. **Decode PP 调度 / microbatch 策略**（mb=1 气泡模型的产品默认）
7. **dtype 产品默认与可选低精度 MAC 因子表**（仍可不做伪 PDK）
8. **Video DiT 真实维数 / denoise 调度 / 是否跨步 KV**（替换 illustrative_dit_video）
9. **Protein pair/MSA 真实算子**（替换 L² GEMM 近似；非 AF2 声称）


## 15. 多域模板：Video DiT / Protein（v0.10，assumed）

同一单卡 NPU + HBM/LPDDR 解析引擎上的 **工作负载形状 + phase program**（非标定、非厂商质量声称）。
仍映射到现有 GEMM/attention-like 计时；Softmax/VAE/三角乘法等均近似掉。

### Video（`illustrative_dit_video`）

| 项 | 假设 |
|----|------|
| 形状 | F=16, lat=32×32, patch=2 → **T=4096**；L=28, H=1024, heads=16, MLP=4096；~0.47B |
| Phase | patchify → transformer（self-attn + MLP）× **N_denoise** 次 forward |
| Attn | **每步全量重算** T×T（常见 DiT）；**无** denoise 步间 KV cache |
| 指标 | time-to-first-clip ≈ N_denoise × t_forward；frames/s = F / TTFC |
| 墙 | 大 T 下偏 **compute**（attn O(T²) × N_denoise）；权重流相对小（~0.94 GB） |

**示意**（`sku_100t`，assumed）：

| N_denoise | mem | fwd ms | TTFC ms | frames/s | wall |
|-----------|-----|--------|---------|----------|------|
| 20 | HBM/LPDDR | ~74.3 | ~1487 | ~10.8 | compute |
| 50 | HBM/LPDDR | ~74.3 | ~3717 | ~4.3 | compute |
| 100 | HBM/LPDDR | ~74.3 | ~7434 | ~2.2 | compute |

**压力点**：算力步数（N_denoise × T² attn），不是 LLM decode 的权重流墙。CLI：`sweep-video`。

### Protein（`illustrative_protein_pair`）

| 项 | 假设 |
|----|------|
| MVP | ESM 风格 **L token encoder** + 可选 **L×L pair**（pair_dim=128） |
| Pair 近似 | 每层：outer-product-ish FLOPs≈2·L²·C_z + GEMM (L²,C_z)×(C_z,C_z)；**非** AF2 triangle |
| 内存 | pair_act = L² · C_z · act_bytes → **随 L² 爆炸** |
| 指标 | time/seq；OOM if W+seq_act+pair_act > mem.capacity |

**示意**（`sku_100t`，assumed）：

| L_aa | pair GB | need GB | t_ms (HBM) | wall | OOM@LPDDR64 |
|------|---------|---------|------------|------|-------------|
| 256 | 0.017 | 0.82 | ~6.9 | compute | no |
| 512 | 0.067 | 0.87 | ~21 | compute | no |
| 1024 | 0.268 | 1.08 | ~71 | compute | no |
| 2048 | 1.074 | 1.88 | ~258 | compute | no |
| 32768 | ≫64 | ≫64 | — | — | **YES** |

**压力点**：**L² 内存**（pair 激活）与 encoder/pair GEMM 算力；与 LLM 长 ctx 的 KV 墙同类但更陡。CLI：`sweep-protein`。

### 与 LLM 墙的对照

| 域 | 主导墙（典型 illustrative @ sku_100t） |
|----|----------------------------------------|
| LLM decode 27B | 权重流（LPDDR mem）或 OS util（HBM compute） |
| Video DiT | N_denoise × T² **compute 步数** |
| Protein pair | **L² pair 内存** + L²/L³-ish GEMM |

---

## 16. Cross-domain compare（v0.11，assumed）

同一 `sku_100t` × {HBM, LPDDR}、fp16、默认 64 MiB SRAM 下的一页对照（CLI：`compare-domains`）。

| domain | mem | metric | metric_ms | wall | dominant_cost |
|--------|-----|--------|-----------|------|---------------|
| llm 27B | HBM | TPOT | ~33.1 | compute | compute_gemm |
| llm 27B | LPDDR | TPOT | ~145.6 | memory | weight_stream |
| llm mla | HBM | TPOT | ~33.1 | compute | compute_gemm |
| llm mla | LPDDR | TPOT | ~145.4 | memory | weight_stream |
| video dit | HBM/LPDDR | TTFC@N=50 | ~3717 | compute | denoise_attn |
| protein | HBM/LPDDR | t/seq@L=512 | ~21.0 | compute | pair_L2 |

video notes：fps≈4.3 @ N=50。LLM 短 ctx 下 MLA≈GQA（权重流主导）；差异在长 ctx / fabric。

### Large slow DiT（`illustrative_large_dit`）

更大 T（F=32, lat=64×64, patch=2 → **T=32768**）、L=40、H=1536、默认 N_denoise=80。
**示意占位**，用于「~100T 上重生成模型会很慢」叙事 — **不是** MiniMax-H3 / 任何厂商 checkpoint 声称。

| N_denoise | TTFC (sku_100t) | ≈ minutes | fps |
|-----------|-----------------|-----------|-----|
| 50 | ~185484 ms | ~3.1 min | ~0.17 |
| 80 | ~296775 ms | ~4.9 min | ~0.11 |
| 100 | ~370969 ms | ~6.2 min | ~0.09 |

相对 `illustrative_dit_video`（T=4096, N=50 → TTFC~3.7 s）：大 DiT 因 T² attn 与更多层，TTFC 慢 **~50×**（同 N=50）。CLI：`sweep-video --shape large`。

---



---

## 17. Workbench 产品层（v0.12，assumed）

统一 `WorkbenchConfig` → `MetricsCard`，复用 scaleup 物理。示例（`illustrative_27B`，cores=16×6.25T ≈ sku_100t，HBM，S=ctx=512）：

| chips | tp | TTFT_ms | TPOT_ms | wall | peak_T |
|------:|---:|--------:|-------:|------|-------:|
| 1 | 1 | ~332.3 | ~33.1 | compute | ~100.4 |
| 4 | 4 | ~83.1 | ~8.3 | compute | ~100.4 |

要点：
- `chip_count` 默认映射 `tp=chips`；显式 `{tp,pp,ep}` 乘积必须等于 chips。
- DRAM geometry：`peak_Bps = n_packages * n_channels * (width/8) * GT/s`；preset 作缺省。
- `n_cores × tops_per_core` → `n_engines=n_cores` + 近方形 PE（assumed）；16×6.25 → 56×56×16。
- Series packs：`series/dense-27b` / `moe-active13b` / `mla-27b` 等，metadata 标明 illustrative/placeholder。
- CLI：`workbench` / `workbench-scan`（可 CSV）/ `list-series`。


## 18. Parallel matrix / mem geometry / card export（v0.13，assumed）

`workbench-parallel`（或 `workbench-scan --mode parallel`）：给定 `--chips N`，枚举 tp×pp×ep=N（N 为 2 的幂时度数 ∈{1,2,4,8,…}），对每个组合评 MetricsCard，按 TPOT（或 TTFT）排序；dense 强制 ep=1，MoE（`series/moe-active13b`）允许 EP。chips=8 @ HBM、cores=16×6.25T 示意：dense 最优 tp=8/pp=1（TPOT~4.1ms，compute），pp↑ 则 bubble 墙主导；MoE 10 组，ep=8 与 tp=8 同位最优（TPOT~1.9ms），中间 EP/TP 混合可出现 memory/balanced。`workbench --json/--md` 导出 MetricsCard；`workbench-scan --sweep-mem` 扫 n_channels/data_rate（HBM ch=1→4 在~1.8TB/s 处 memory→compute FLIP）。series 增 `series/video` / `series/large-dit` / `series/protein` 别名，`list-series` 显示 domain。

*版本：0.13.0 — 与 README / DECISIONS 同步；数字随 assumed 旋钮可扫。 Workbench parallel/mem/export 为产品层 polish；多域仍未声称标定。*


## 19. HTML report / requirements gap（v0.14）

`python3 -m npu_dse report --out out/report.html`：离线自包含页（inline CSS），汇总 walls 解读、series、chips sweep、chips=8 parallel dense+MoE、mem geometry FLIP、`compare-domains` 快照与 uncalibrated banner。对照清单见 `REQUIREMENTS_GAP.md`（真缺口：真实系列维数、标定 BW/freq、UI 产品化）。

*版本：0.14.0 — 与 README / DECISIONS / CHANGELOG 同步；数字随 assumed 旋钮可扫。*

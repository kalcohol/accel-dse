# 建模说明（Modelling method）

本文说明 accel-dse 0.41（core v2）怎么算、假设了什么、用什么校验，方便逐条挑错。
「假设」= 可调输入、未经硅片标定；其余量由模型发布、几何与公式推出。**所有绝对数都依赖「假设」，不是实测数据。**

目录：[1 分层](#1-分层) · [2 模型](#2-模型按发布建模) · [3 算子图与并行](#3-逐-rank-算子图与并行) · [4 映射](#4-映射数据通路组织) · [5 存储](#5-存储规划) · [6 调度](#6-调度) · [7 服务与搜索](#7-服务goodput-与搜索) · [8 稳定性](#8-排名稳定性) · [9 校验](#9-校验) · [10 范围](#10-范围与近似) · [11 视频生成与蛋白质](#11-视频生成dit与蛋白质模型)

---

## 1. 分层

| 层 | 模块 | 内容 |
|----|------|------|
| L0 | `core/scenario.py` | 不可变场景：模型、芯片、存储器 id、链路、映射、布局、服务参数、视频 / 蛋白质工作负载、dtype what-if；`replace(path, value)` 受控替换；规范化哈希；严格解析（未知字段 / 非有限数拒绝） |
| L1 | `core/model.py` `core/catalog.py` `core/domain.py` | HF config + safetensors 头 → `ModelSpec`（逐层注意力 / FFN、逐角色 dtype；视频 DiT / 蛋白质编码器见 §11） |
| L2 | `core/ir.py` `core/parallel.py` | 逐 rank 算子图；布局 PP·TP·DP·EP·ETP（视频 / 蛋白质：PP·TP·DP·SP）与 stage 划分 |
| L3 | `core/mapping.py` | 每个 GEMM 在映射组织下的 MAC 与供数周期 |
| L4 | `core/memplan.py` | SRAM 驻留、staging、逐 stage DRAM 容量与每步流量 |
| L5 | `core/schedule.py` `core/evaluate.py` | 每级时间、集合通信、流水、投机解码 |
| L6 | `core/serving.py` | 含 prefill 的 goodput 与 TTFT 标记 |
| L7 | `core/search.py` `core/stability.py` | 精确 batch 搜索、布局排名、Pareto、稳定性 |

单卡就是 `Layout()`（全 1），与多卡走同一条路径。

## 2. 模型：按发布建模

- 来源：`scripts/fetch_hf_release.py` 读取 `config.json` 与 safetensors 头（HTTP Range，不下载权重），`scripts/summarize_release.py` 汇总为 `accel_dse/data/releases/*.json`：逐角色（attn / mlp / expert / shared_expert / router / embed / lm_head / mtp）的参数量、存储格式与实际字节。
- dtype 按发布：fp8 block-128（W8A8）、fp8 per-tensor、MXFP4（4.25 bit）、NVFP4（4.5 bit）、AWQ / compressed-tensors int4（含 scale / zero 开销）。有效位数取自实际字节，包含 scale。官方量化版（如 Qwen3-*-FP8 / AWQ）是独立目录条目。
- 自由改 dtype 只作为 what-if，结果与模型标签都标注。
- 结构：GQA、MLA、线性注意力（Gated DeltaNet / KDA / lightning）、滑窗、稀疏索引注意力、dense / MoE / latent MoE、共享专家、MTP。
- 修正项：MTP 层按层号 ≥ `num_hidden_layers` 归类；tied lm_head 的重复拷贝不重复计数；配置缺失 MTP 时从发布权重推断；n-gram / engram 查表按发布 embed 角色的大小计入存储，不计入激活参数。
- 三轴标签：来源（official 官方 / mirror 镜像，如 meta-llama 的 unsloth 公开镜像）× 覆盖 × dtype。覆盖度指建模覆盖程度，与模型好坏无关，每个模型都标出：「完整」= 全部算子按发布结构逐项建模；「部分」= 主干逐项建模、个别机制近似（线性注意力的递归状态、DSA 稀疏索引、attention sink）；「架构代理」= 含未建模的结构（超连接多流残差的混合计算、n-gram / engram 查表、压缩稀疏注意力），结果只作量级参考。每个非「完整」模型都附逐项的「近似之处」（`coverage_reasons`，由结构自动生成，UI 悬停与模型目录可见）。
- VLM（发布中带视觉编码器的模型）：只评估语言主干，视觉编码器（约 0.4–0.6B 参数）的权重存储与图像 prefill 不计，在标签中注明。
- 目录：按厂商 → 系列排列，同一厂商的 LLM 与 VLM（甚至同一系列里的文本版与多模态版）放在一起，领域只作为行内标记（VLM 行标「VLM · 视觉编码器未建模」）；系列内按尺寸从大到小，官方量化版（FP8 / AWQ）紧跟原版，名称用官方仓库名。结构与 dtype 完全相同的发布合并为一条（如 DeepSeek-V3 / V3.1 / R1、Kimi-K2.5 / K2.7-Code、GLM-5 / 5.2、GLM-4.5 / 4.6、MiniMax-Text-01 / M1-80k），尺寸已被覆盖的通用稠密 GQA 模型不单列；它们仍可按 id 评估，也都参与参数核对。
- 已接入 v2 的视频 / 蛋白质发布（0.41）：Wan2.1-T2V-14B / 1.3B、CogVideoX-5b / 2b（DiT 去噪主干）与 ESM-2 3B / 650M（编码器），同样从 config + safetensors 头建模，参数与发布逐项一致（偏差 0.00%），建模见 §11。
- 暂未接入 v2 的目录条目：视频生成（Wan2.2-A14B、MiniMax-H3、HunyuanVideo、LTX-Video、Mochi 1、Open-Sora STDiT3）与蛋白质（ESMFold、AlphaFold3 / AlphaFold2、Protenix、Boltz-1、OpenFold）列在各自厂商 / 机构之下、该厂商可评估条目之后，标「暂未接入 v2」，不能评估、不能按 id 解析。维数取自历史目录（`data/series_catalog.json`，公开 config / 论文仓库；AF 类的 FFN 宽度与部分头数是代理值），在接入 v2 之前只是占位。分子动力学 / 机器学习力场（MLFF）从未有过目录条目，不列出。
- 参数核对：与发布 safetensors 总量偏差 > 2% 时附注；当前 61 个发布（55 个 LLM / VLM + 6 个视频 / 蛋白质）全部在 ±0.5% 内。

## 3. 逐 rank 算子图与并行

- 布局 `PP·TP·DP·EP·ETP`：dense 模型 DP = EP = ETP = 1（数据并行副本是独立服务实例）；MoE 要求 `EP·ETP = TP·DP`——注意力按 TP 切分、在 DP 组间复制，专家按 EP 分组、组内按 ETP 切分。
- 每个 rank 生成算子：投影 GEMM、注意力核心（GQA / MLA 吸收 / 线性状态更新）、MoE 路由与专家 GEMM、embedding / 词表并行 lm_head、MTP。
- MoE 命中专家数：`hit = max(ceil(局部期望命中), ceil(全局期望命中 / EP))`，限定在 [1, 本地专家数]；每专家 token 数 `m_e = ceil(本地 token·top_k / hit)`。默认 token 均匀路由（「假设」）；0.49 起可选 EP 负载倾斜（§16）。
- stage 划分按层的整数切分（如 61 层 / 8 → 8,8,8,8,8,7,7,7），容量检查取最重的 stage。
- 视频 / 蛋白质（无 KV 缓存的全序列前向）：布局 `PP·TP·DP·SP`，EP = ETP = 1。DP 切分一次前向的序列（视频含 CFG 的 cond / uncond 两路，即 CFG 并行）；SP 为 Ulysses 序列并行：token 按 SP 切分，注意力前后各一次 all-to-all，每 rank 对完整序列计算 `ceil(ceil(H/TP)/SP)` 个头，权重在 SP 组内复制。LLM 布局保持 SP = 1。

## 4. 映射：数据通路组织

阵列 R × C × E（100T = 56 × 56 × 16 @ 1 GHz，`Ce = C·E = 896`）。SRAM 端口默认 `4·(R + Ce)·2` B/cycle（「假设」，100T 为 7616）。格式速率 `r`：bf16 / fp16 ×1，fp8 / int8 ×2（按芯片的原生格式矩阵）。

| 组织 | MAC 周期 | 供数字节（÷ 端口 B/cycle） |
|------|----------|---------------------------|
| `os` 输出驻留 | `ceil(M/R)·ceil(N/Ce)·K / r` | `ceil(M/R)·K·N·wb + ceil(N/Ce)·M·K·ab + M·N·ob` |
| `ws_edge` 权重驻留·边缘加载 | `ceil(K/R)·ceil(N/Ce)·max(ceil(M/r), R·Ce/(Ce·r))` + 填充 `R+Ce`（每算子一次） | `K·N·wb + ceil(N/Ce)·M·K·ab + 部分和溢出 + M·N·ob` |
| `ws_broad` 权重驻留·宽面广播 | 同上，装载 `R·Ce/(R·Ce·r)` | 同上 |
| `os_vec` | 逐算子 min(os, GEMV) | |
| `reconf` 可重构 | 逐算子 min(os, ws_edge, ws_broad, GEMV) | |

- GEMV 单元默认 `R·Ce/8` MAC/cycle（「假设」）：`M·K·N / (gemv·r)`。
- 部分和溢出：只有超过累加器行数（默认 1024 KiB → 292 行）的行把 fp32 部分和写出 / 读回 SRAM。
- 算子时间 `max(MAC, FEED) / (f · mac_eff)`。
- 格式执行：W 与 A 同为原生格式 → 原生速率；仅权重量化 → 反量化到激活格式；都不原生 → 上转换到 bf16。被转换的元素在向量单元按每元素 2 次操作计时（向量 lanes 默认 `4·C·E`，「假设」）。

## 5. 存储规划

- staging = `max(2 MiB, 2 × 最大激活)`；其余 SRAM 依次驻留热权重（每步都读）、专家与冷 embedding 表，剩余容量放 KV / 线性注意力状态。
- 每步 DRAM 流量 = 未驻留的被触及权重 + KV 读（未驻留部分）+ KV 写 + 状态 + 查表行。
- DRAM 需求 = 存储权重 + KV + 状态 + 1 GiB 预留；按最重 stage 与存储器容量比较，超出给出警告。
- 可选系统级缓存（SLC，0.48，默认无）位于 SRAM 与 DRAM 之间，见 §14。
- 存储器：`mem_catalog.py` 按 JEDEC / 厂商资料给出类型 × 形态 × 位宽 × 速率 × 数量 × 容量。每个选项带**双轴标签**：规范状态（JEDEC / 疑似 JEDEC / 超规格定制 / 无规范）× 产品状态（量产 / 送样 / 已发布 / 无产品），组合各轴分别取最弱项；颗数 / 堆数是 SoC 设计选择，不打规范标签。带宽 = 总线位宽 × 速率 × payload（LPDDR6 为 256/288 = 8/9，与 Samsung/JEDEC 讲稿 114 GB/s 交叉验证）× 效率（默认 0.7，「假设」）。LPDDR6 可选「meta 模式」（默认关）：按假设把阵列的 1/16 划作 metadata 持久区，从可用容量扣除；Meta RD/WR 吞吐损失未建模。LPDDR5T 作为 LPDDR5X-9600 的别名。**LPDDR5X x96（6×16）仅为 Apple 定制件，不提供。** 调研与来源见 [docs/research](research/memory_specs_2026-10.md)。

## 6. 调度

- 每个 stage：`t = max(t_compute, t_dram, t_slc, t_link) + t_sync`，其中 `t_compute = Σ max(MAC, FEED) + VECTOR`，`t_slc` 只在有 SLC 时非零（§14）；绑定项标为 MAC / FEED / VECTOR / DRAM / SLC / LINK / SYNC。
- 集合通信 α-β 模型：带宽项按环形算法（all-reduce `2(g−1)/g`、all-to-all `(g−1)/g`、all-gather `(g−1)` 倍负载）/ 链路带宽（默认 400 GB/s，「假设」），同步 α（默认 3 µs，「假设」）每次集合通信计入，作为暴露时间单独累计。0.50 起为三层互连：可选的封装内 die-to-die（D2D，默认关 = 单片大 die）、节点内 scale-up、跨节点网络（默认单节点 = 不用），集合通信按组跨越的层逐级计（§15）；默认即上面的单层模型。
- decode 一步 = `max(microbatch, PP) × 最慢 stage`；prefill = `(microbatch + PP − 1) × 最慢 stage`。
- 投机解码 / MTP：每步期望 token `E = (1 − a^(k+1)) / (1 − a)`，草稿在最后一个 stage 上运行。
- 有效 MAC 比例（array_util）= 理想 MAC 时间 / 阵列时间，直接反映映射与小 M 的浪费。

## 7. 服务、goodput 与搜索

- decode 目标：在 TPOT ≤ SLO 下每卡 tok/s 最大。
- goodput 目标（合并服务、副本分时，「假设」；prefill / decode 分池见 §18）：每个请求 S 个 prompt token + out_len 个输出 token，`goodput = 1 / (1/R_d + (S/out_len)/R_p)`，`R_p` 取满足 TTFT SLO 的最大 prefill batch。若单请求 prefill 也超过 TTFT SLO（典型：DP 布局下一个请求只占一个 DP 组），标记 `ttft_ok = false` 而不是把 goodput 记为 0。
- 精确 batch 搜索：指数 + 二分求可行上限 b_max，再分支定界。上界只依赖 step(B) 随 B 不减（P1）：对已评估的可行点 p₀ < p₁ < …，`thr(B) ≤ (p_{i+1} − 1)·E / step(p_i)`，B ∈ [p_i, p_{i+1})。测试中与暴力枚举逐一核对。
- 布局枚举：卡数的全部 PP·TP·DP·EP·ETP 分解（满足第 3 节约束）；另给出 TPOT–吞吐 Pareto 前沿。
- 跨布局分支定界（只要前 k 名时）：先用指数阶段的分段上界给布局排序，再以第 k 名的精确得分为门槛逐个求解；二分只在上界仍可能超过门槛时继续，否则直接剪枝。goodput 目标把门槛换算为所需 decode 速率 `1 / (1/门槛 − (S/out_len)/R_p)`。测试核对 top-k 与完整排名一致。
- 记忆化：按模型对象缓存层分组、每层算子和（按形状 / 映射 / 芯片 / 链路）、存储规划与 prefill 结果；GEMM 代价与存储器规格用 LRU。HTTP 服务把映射对比（每种映射一个任务）与稳定性放到 spawn 的工作进程池中并行（默认 min(5, CPU 数) 个进程）。
- 容量提示（`/api/fit`）：放不下时给出 (a) 只是 batch 太大时可放下的最大 batch（容量随 batch 单调，二分）；(b) 2 / 4 / 8 / 16 / 32 / 64 中第一个能在 batch 1 放下的卡数，并给出该卡数下 decode 吞吐最优的布局与 batch；(c) 同类型、同速率下第一个放得下的更大存储器配置。三者都是对场景的真实重算，不是估计。

## 8. 排名稳定性

对基准场景做单因素扰动：映射（可选）、DRAM 效率 0.6 / 0.7 / 0.85、α 1 / 3 / 5 µs、MAC 效率 0.7，以及两个角点。每个扰动下先精确求原 top-1 布局的得分，以它为门槛做跨布局分支定界，只求解可能超过它的布局（结果与不设门槛的完整搜索一致，有测试核对）；若 ≥ 90% 的扰动下 top-1 不变，或原 top-1 与新 top-1 相差 ≤ 5%，判为稳定。UI 的映射对比中每种映射单独给出该标记。

## 9. 校验

`python3 -m accel_dse validate` 复现以下结果；测试守护 V0–V3。

- **V0 变形关系**：受控替换同值不变；更多 DRAM 带宽 / SRAM / 更宽端口 / 更快链路不会更慢；更长上下文不会更快；`reconf` 不慢于任一单一组织；去掉原生 fp8 不会让 fp8 发布更快。另有 TP / EP 分片守恒测试（FLOPs、权重、KV、专家存储、stage 存储）。视频 / 蛋白质：同样的 SRAM / 带宽 / `reconf` 关系、batch 增大延迟不减、延迟 ∝ 去噪步数、TP / SP 分片 FLOPs 守恒（SP 下文本跨注意力 K/V 每 rank 重算，偏差 < 0.2%）、SP 复制权重 / TP 切分权重，以及精确 batch 搜索与暴力枚举一致。
- **V1 参数**：55 个 LLM / VLM 发布与 safetensors 总量偏差全部 ≤ 0.5%，6 个视频 / 蛋白质发布逐项一致；8 个模型卡的激活参数 ≤ 5%；FLOPs 与独立计数对照（视频 / 蛋白质：按 config 的闭式计数，偏差 < 1%）。
- **V2 趋势区间**（H100 类配置：128 × 128 × 16 @ 1.83 GHz ≈ 959 TFLOPS bf16、SRAM 50 MiB、HBM3 5 堆 3.33 TB/s、DRAM 效率 0.8、可重构映射，均为「假设」）：Llama-3.1-8B bf16 B1 TPOT 5.7 ms（区间 4.5–9）；B16/B1 = 1.13；Qwen3-8B FP8/BF16 = 0.54；4K prompt TTFT 83 ms；Qwen3-32B TP2 加速 1.94；B256 吞吐 1.4 万 tok/s。视频只做工作负载 / FLOP 的合理性核对（不是硬件标定）：Wan2.1 README 称 T2V-1.3B 在单张 RTX 4090 上约 4 分钟生成 5 s 480P（含 T5、VAE 与 offload）；本工具每段 28.6 PFLOP（0.44 起含 umT5 与 Wan-VAE 解码 0.29 PFLOP），折合 4090 约 165 TFLOPS（bf16、fp32 累加）的 0.72（区间 0.4–1.0）。
- **V3 GenZ 对照**（Llama-3.1-8B decode，参考值由 `scripts/genz_reference.py` 生成）：LPDDR 191 GB/s 各点比值 1.07（DRAM 效率口径差异）；HBM 6.6 TB/s 下 `reconf` 映射比值 1.07–1.22，长上下文 / 大 batch 点偏高来自注意力小 M 分块，而 GenZ 按理想 FLOPS 计；`os` 映射在 HBM 下比值 3.6–6.8，因为小 M decode 被 SRAM 供数端口限制——这正是映射作为设计变量要暴露的差别，不视为误差。容差：访存受限点 15%，其余 60%。

## 10. 范围与近似

- LLM 推理（VLM 只算语言主干）；视频生成覆盖 DiT 去噪主干（§11），蛋白质覆盖 ESM-2 编码器（§11）与结构预测的神经网络推理（ESMFold、AlphaFold 2、OpenFold、Boltz-1、Protenix，§12）。AlphaFold 3 权重需申请、无可核对的公开发布文件，标「暂未接入 v2」（§2）。不覆盖分子动力学 / 力场（目录中也无此类条目）。
- 「架构代理」模型：超连接多流残差只计参数不计混合计算；查表只计存储与每 token 行读取；压缩稀疏注意力按有效上下文 `ctx/ratio`（+ 窗口，索引层 ≤ top-k）近似；哈希路由层按 top-k MoE 处理。
- 解析模型不模拟周期级行为：无 bank 冲突、无 DRAM 刷新 / 页冲突细节（统一由效率「假设」吸收），集合通信用 α-β 近似，MoE 默认 token 均匀路由（可选倾斜系数，§16）。
- 不内置功耗、面积、成本估计。0.47.1 起给出每输出单位的动作计数，能耗只在用户提供每动作能耗时计算（§13）；0.49 起可填资源 / 面积预算，面积按用户给的密度估算，只报告余量（§17）。

## 11. 视频生成（DiT）与蛋白质模型

0.41 起（0.42 增加 6 个视频发布），以下发布按与 LLM 相同的流水线评估（算子图 → 并行 → 映射 → 存储规划 → 调度 → 指标）。它们都是**非自回归的全序列前向**：没有 KV 缓存，注意力为双向，一次前向处理整条序列。

| id | 发布 | 参数 | 发布 dtype | 覆盖 |
|----|------|------|-----------|------|
| `wan2.1-14b` / `wan2.1-1.3b` | Wan-AI/Wan2.1-T2V-14B / -1.3B | 14.29B / 1.42B | fp32 | 完整（0.44：umT5-XXL 11.4 GB + Wan-VAE 0.5 GB 计入，§11.4） |
| `cogvideox-5b` / `cogvideox-2b` | zai-org/CogVideoX-5b / -2b（`transformer/`） | 5.57B / 1.69B | bf16 / fp16 | 完整（T5-XXL 9.5 GB + VAE 0.9 GB 计入） |
| `esm2-3b` / `esm2-650m` | facebook/esm2_t36_3B_UR50D / esm2_t33_650M_UR50D | 2.84B / 0.65B | fp32 | 完整 |
| `wan2.2-a14b`（0.42） | Wan-AI/Wan2.2-T2V-A14B（`high_noise_model/` + `low_noise_model/`） | 28.58B（每步激活 14.29B） | fp32 | 完整（umT5 11.4 GB + VAE 0.5 GB 计入） |
| `hunyuanvideo`（0.42） | hunyuanvideo-community/HunyuanVideo（`transformer/`） | 12.82B | bf16 | 完整（LLaVA-Llama-3-8B + CLIP-L 15.3 GB + VAE 1.0 GB 计入） |
| `ltx-video`（0.42） | Lightricks/LTX-Video（`transformer/`，即 2B v0.9） | 1.92B | fp32 | 完整（T5-XXL 19.0 GB + VAE 1.7 GB 计入） |
| `mochi-1`（0.42） | genmo/mochi-1-preview（`transformer/`） | 10.03B | fp32 | 完整（T5-XXL 19.0 GB + VAE 1.8 GB 计入） |
| `opensora-stdit3`（0.42） | hpcai-tech/OpenSora-STDiT-v3 | 1.21B | fp32 | 完整（DeepFloyd T5-XXL 19.0 GB + Open-Sora VAE 1.6 GB 计入，均在其他仓库） |
| `minimax-h3`（0.42） | MiniMaxAI/MiniMax-H3（`transformer/`） | 33.12B（推理加载 20.11B） | bf16（io 层 fp32） | 完整（Qwen3-VL 66.7 GB + ViT 视频解码器 10.4 GB + 音频 VAE 0.6 GB 计入） |

ESM-2 3B 的 main 分支只有 `pytorch_model.bin`；参数取自同仓库 `refs/pr/2` 的 safetensors 转换（HF 官方 bot），其字节数与 main 的 `pytorch_model.bin.index.json` 的 total_size 一致（已核对）。其中 tied 的 decoder 拷贝不重复计数。

**结构（逐层，按发布张量形状）**

- Wan2.1：全 3D 自注意力（不是时空分解），带 qk-RMSNorm 与 3D RoPE → 文本跨注意力（Q 来自视频 token；K/V 来自 512 个 umT5 token，参考实现每步重算）→ GELU FFN；每层 AdaLN 调制来自共享的时间嵌入 MLP（每序列一次）。patch 嵌入（1×2×2 卷积，按 GEMM 计）、文本 / 时间嵌入 MLP 与输出头在首 / 末 stage。
- CogVideoX：226 个文本 token 与视频 token 拼成一条序列做**联合**全 3D 注意力（qkv / FFN 对文本 token 也计算）；每层两组 AdaLN（按序列算一次的线性层）；2b 用 sincos 位置，5b 用 3D RoPE。
- Wan2.2-A14B（0.42）：两个结构同 Wan2.1-14B 的专家（高噪声 / 低噪声），按 boundary 0.875 在去噪轨迹上切换——每步只算一个专家；两个专家都常驻 DRAM。建模：算子图按一个专家，另一个专家作为 standby 存储（按 PP stage 切分、TP 均分，计容量不计读流量）。全 3D 注意力。
- HunyuanVideo（0.42）：20 个双流块（视频 / 文本各自 qkv、输出投影、FFN，联合全 3D 注意力）+ 40 个单流块（一个融合 GEMM 同时出 qkv 与 MLP 输入，一个输出 GEMM，TP 下只需一次 all-reduce）；文本 256 token 计入联合序列；token refiner 的 GEMM 计入。guidance 蒸馏：每步 1 次前向。
- LTX-Video（0.42）：28 层，全 3D 自注意力 + 文本跨注意力（128 T5 token）；VAE 32×32×8 压缩、patch 1 → token 数很少（默认 7,392）。
- Mochi 1（0.42）：48 层非对称双流（视频 3072 宽、文本 1536 宽，联合注意力，文本 256 token）；末层 context_pre_only（文本只作 K/V）；VAE 时间压缩 6×。
- Open-Sora STDiT3（0.42）：28 个空间块与 28 个时间块交替——**分解时空注意力是发布结构**（空间块在每个潜帧的 H·W 内注意，时间块在每个空间位置沿 T 注意），不是近似；FLOPs 远低于同 token 数的全 3D 注意力。每块有 T5 跨注意力（300 token）。
- MiniMax-H3（0.42）：视频 + 立体声音频 + 文本打包成一条序列做全注意力（开源版只有全注意力推理）；音频 40 latent/s × 2 声道，文本按 512 token 计「假设」。每层的 AdaLN 分支（共 13.0B 参数）按 README 预计算缓存、推理不加载：只计参数，不计存储与读流量。FL2VA / Ref2VA 等附加 transformer 不在默认 T2AV 路径中，不计。guidance 蒸馏：每步 1 次前向，num_inference_steps 50 = 49 次前向。
- ESM-2：双向编码器（RoPE、pre-LN、GELU FFN）+ MLM 头（dense + LN + 33 词表 decoder）。绝对位置表（随发布存储、推理不读）只计存储；contact head 只计参数（默认不输出接触图）。

**工作负载与 token 数**（场景的 `workload` 块，0 = 发布默认）

- 视频：潜空间帧 `F' = (F − 1)/4 + 1`，空间 `H/8 × W/8`，patch 1×2×2 → 视频 token `F'·(H/16)·(W/16)`；CogVideoX 另加 226 个文本 token。默认值取自官方 README / 配置：Wan2.1-1.3B 832×480、Wan2.1-14B 1280×720，均为 81 帧 16 fps、50 步、CFG（每步 2 次前向）→ 32,760 / 75,600 token；CogVideoX 720×480、49 帧 8 fps、50 步、CFG → 17,550 + 226 token。偏离默认（非 4k+1 帧、分辨率不是 16 的倍数）给出警告。
- 0.42 新增的潜空间帧规则：Wan2.2 / HunyuanVideo 同上（因果 VAE，`(F−1)/4+1`）；LTX `(F−1)/8+1`、空间 /32、patch 1；Mochi `(F−1)/6+1`；Open-Sora 1.2 的 VAE 按 17 帧一块编码，每块 5 个潜帧（不足 17 帧的尾部按比例），默认 720p 4s = 102 帧 → 30 潜帧 × 80 × 45 = 108,000 token；MiniMax-H3 空间 /16，帧数先补齐到 17n+5（参考实现）→ 5n+2 潜帧，默认 1344×768、124 帧 → 37 × 42 × 24 = 37,296 视频 token + 414 音频 + 512 文本 = 38,222。默认值：Wan2.2 1280×720 81 帧 40 步 CFG；HunyuanVideo 1280×720 129 帧 50 步（无 CFG）；LTX 704×512 161 帧 50 步 CFG；Mochi 848×480 163 帧 64 步 CFG；Open-Sora 720×1280 102 帧 30 步 CFG；H3 见上（49 次前向，无 CFG）。
- CFG = 1 表示发布是 guidance 蒸馏（HunyuanVideo、MiniMax-H3），每步一次前向；what-if 设为 2 时前向数翻倍。
- 蛋白质：token = 残基 + `<cls>` / `<eos>`；默认 512 残基（「假设」），超过 1022 残基（训练长度）时警告。

**dtype（按发布）**：Wan2.1 与 ESM-2 发布为 fp32——存储与读流量按 4 B / 参数；激活按 bf16（Wan 参考实现 autocast；ESM 为「假设」）。芯片没有 fp32 MAC，每个 GEMM 把权重上转换为 bf16 执行，转换在向量单元计时（与 LLM 的反量化规则相同）。CogVideoX-2b 为 fp16 / fp16、5b 为 bf16 / bf16。dtype what-if（如 bf16 发布）照常可用并标注。

**激活流式（只用于全序列前向）**：视频的激活远超 SRAM（Wan2.1-14B 720P 一层的 FFN 输入约 1.5 GB）。每个算子的激活按 SRAM/2 的预算分块流进 / 流出 DRAM：GEMM 取「激活分块、权重重复读」与「权重分块、激活重复读」两者中流量较小者；注意力按 flash 式计算，N×N 分数不落 DRAM，Q / O 读写一次，K / V 每个 Q 块重读一次（块行数 `Br = 预算 / (d·(ab + 4))`）。DRAM 容量另计在途序列的残差流与单个算子的最大激活。LLM 的 prefill / decode 不使用此模型（保持 0.40 行为）。

**映射**：OS / WS（K-split）/ GEMV / 可重构照常是设计变量。全序列注意力的两个 GEMM 两侧都是激活，映射可以取任一方向（`O = P·V` 或 `Oᵀ = Vᵀ·Pᵀ`）：`P·V` 的 N = head_dim = 128，会让 896 列宽的阵列大部分空闲，换向后有效 MAC 从约 30% 升到 85–98%。LLM 注意力仍按 0.40 的固定方向（不改变 LLM 结果）。

**调度与指标**

- 一次前向的序列数 `S = batch × CFG`（视频）或 `batch`（蛋白质），微批 `mb = min(PP, S)`（可手动设置）。
- 视频单段延迟 `T_clip = T_text + 步数 × max(mb, PP) × t_stage + T_decode`（每个去噪步像 decode 步：独立微批保持流水线满；文本编码与 VAE 解码见 §11.4，`workload.pipeline = false` 时两项为 0）；每帧延迟 `T_clip / 帧数`；每去噪步时间 `去噪时间 / 步数`；吞吐 `batch × 帧数 / T_clip`（帧/s，每卡再除以卡数）；另给段/小时/卡、实时倍率 `视频秒数 × batch / T_clip`、每段 TFLOP。
- 蛋白质批延迟 `T = (mb + PP − 1) × t_stage`（一次前向，像 prefill）；吞吐 `batch / T` 序列/s，残基/s = 序列/s × 残基数。
- SLO（「假设」，可改）：视频单段 1800 s，蛋白质批延迟 1000 ms。batch 搜索 / 布局搜索 / 映射对比在 SLO 与容量约束下最大化 帧/s/卡 或 序列/s/卡，与 LLM 的 decode 目标同一套精确搜索（P1 / P2 在测试中逐一核对）。goodput 目标只适用于 LLM（API 拒绝）。

**未建模 / 近似**

- 0.44 起文本编码器与 VAE 解码计入（§11.4）；VAE 编码器不运行（文生视频），只计其随 pipeline 加载的存储。
- 调度器（UniPC / DDIM 等）的逐元素更新、CFG 的组合计算不计；跨注意力 K/V 的跨步缓存不作为优化建模。
- 采样步数、CFG、分辨率取默认值。文本长度在联合注意力模型中按全长计入（HunyuanVideo 256、Mochi 256、H3 512「假设」）；参考实现按 mask 跳过的 padding 未扣除。
- HunyuanVideo / H3 的 token refiner 只计 GEMM，其短文本自注意力核忽略；H3 的 AdaLN 缓存表（< 0.5 GB）不计。
- Wan2.2 双专家的切换没有额外开销（两个专家都常驻）；若 DRAM 放不下两个专家（如 64 GiB LPDDR 放不下 fp32 的 115 GiB），按容量不足报告，不建模专家换入换出。
- LTX-Video 13B（0.9.7 / 0.9.8）为单文件原始格式，未单列；`ltx-video` 指 diffusers 版 2B v0.9。
- AF 类结构预测见 §12（0.43）。

### 11.4 pipeline 组件：文本编码器与 VAE 解码（0.44；放置与分块 0.45；DiT FSDP / 主机 CPU 编码器 0.46；多卡分块解码 / 跨请求重叠 0.47）

一次文生视频请求 = 文本编码器 → 步数 × DiT 前向 → VAE 解码（H3 另有音频 VAE 解码）。0.43 及以前只评估 DiT；0.44 起其余组件同样是算子图，取自各自发布检查点的张量头（`accel_dse/data/pipeline/*.json`，由 `scripts/build_pipeline_data.py` 从 HTTP range 读取的 safetensors / `.pth` 头生成，不下载权重），走与 DiT 相同的映射 → 流式 → 调度路径，**默认计入**时间与存储；场景 `workload.pipeline = false`（Web 取消勾选「计入文本编码器与 VAE 解码」、CLI `--dit-only`）只看 DiT，结果与 0.43 逐字节一致（1356 项指纹核对）。

| 模型 | 文本编码器（存储，dtype 按发布） | 文本编码 | VAE 解码器（存储） | 解码 | 100T + HBM3E、batch 1：DiT 去噪 → 整段 |
|------|------|------|------|------|------|
| `wan2.1-14b` | umT5-XXL 11.36 GB bf16（`.pth` 头） | 9.7 TFLOP / 0.11 s | Wan-VAE 0.51 GB fp32 | 639 TFLOP / 41.3 s | 7362 s → 7403 s（+0.6%） |
| `wan2.1-1.3b` | 同上 | 9.7 TFLOP / 0.11 s | 同上（480P） | 275 TFLOP / 17.9 s | 325 s → 343 s（+5.5%） |
| `wan2.2-a14b` | 同上 | 9.7 TFLOP / 0.11 s | 同上（720P） | 639 TFLOP / 41.3 s | 5889 s → 5931 s（+0.7%） |
| `cogvideox-5b` / `-2b` | T5 v1.1 XXL 9.52 GB（5b bf16 / 2b fp16） | 4.2 TFLOP / 0.05 s | CogVideoX VAE 0.86 GB fp32 | 315 TFLOP / 16.5 s | 414 → 431 s（+4.0%）/ 161 → 178 s（+10.3%） |
| `hunyuanvideo` | LLaVA-Llama-3-8B 文本塔 15.01 GB + CLIP-L 文本塔 0.25 GB fp16 | 4.9 TFLOP / 0.06 s | 3D VAE 0.99 GB fp32（tiling） | 4766 TFLOP / 221 s | 6928 s → 7149 s（+3.2%） |
| `ltx-video` | T5 v1.1 XXL 19.05 GB fp32 | 2.4 TFLOP / 0.03 s | LTX VAE 1.68 GB fp32 | 15 TFLOP / 0.7 s | 49 s → 50 s（+1.5%） |
| `mochi-1` | T5 v1.1 XXL 19.05 GB fp32 | 4.8 TFLOP / 0.06 s | AsymmVAE 1.84 GB fp32 | 1053 TFLOP / 42.5 s | 2433 s → 2475 s（+1.7%） |
| `opensora-stdit3` | DeepFloyd T5 v1.1 XXL 19.05 GB fp32 | 2.8 TFLOP / 0.03 s | Open-Sora VAE v1.2 1.57 GB fp32 | 1158 TFLOP / 39.9 s | 244 s → 284 s（+16.4%） |
| `minimax-h3` | Qwen3-VL 文本塔 66.71 GB bf16 | 32.2 TFLOP / 0.37 s | ViT 视频解码器 10.42 GB + 音频 VAE 0.61 GB fp32（按发布分块） | 1893 TFLOP / 24.4 s | 1913 s → 1937 s（+1.3%） |

**文本编码器**：文本 transformer 的每个头部 GEMM 按 `M = 提示数 × 补齐后的 token 数`；注意力核 T5 / umT5 为双向，Llama / Qwen / CLIP 文本塔为 causal。token 数按参考实现的补齐长度：Wan 512、CogVideoX 226、HunyuanVideo Llama 351（模板 95 + 256）与 CLIP 77、LTX 128、Mochi 256、Open-Sora 300、H3 512「假设」。提示数 = batch × CFG（参考实现为 uncond 分支编码负向 / 空提示），Open-Sora（学习到的 null 嵌入）与 guidance 蒸馏模型（HunyuanVideo、H3）为 batch。HunyuanVideo 取倒数第 3 层、H3 取第 50 层隐状态，但参考前向跑满全部层：按全部层计。H3 的 text_encoder 带视觉塔（0.60B）与 lm_head（0.78B），随 pipeline 加载计入存储、不计算。DeepFloyd T5 只有 `.bin`：形状用同构的 T5 v1.1 XXL（CogVideoX 头），dtype 由文件总大小判定为 fp32。

**VAE 解码器**：每个卷积作为隐式 GEMM——`M` = 该层分辨率下的输出体素数，`K = C_in × 卷积核体积`（逐帧 2D 卷积为 kh·kw），`N = C_out`；DRAM 流式按真实输入张量（`M × C_in`）计，im2col 矩阵只在片上（SRAM 端口仍按 `M × K` 读）。分辨率按各家 up-block 的上采样表：Wan / CogVideoX / HunyuanVideo / LTX 时间 ×2（causal：`t → 2t − 1`），Mochi 时间 ×3、×2、×1（`t → e·t − (e − 1)`），空间每级 ×2；Wan 的 `time_conv` 与 LTX / Mochi 的 depth-to-space 卷积在上采样前的分辨率执行，其余上采样卷积在上采样后。中间块注意力：Wan 与 SD-VAE 逐帧单头注意力，HunyuanVideo 为 tile 内整段 causal 3D 注意力。默认输出与发布一致（测试核对）：Wan 720P (81, 720, 1280)、CogVideoX (49, 480, 720)、LTX (161, 128, 176) 再 4×4 unpatchify、Mochi (163, 480, 848)。
- HunyuanVideo：中间块是整段 3D 注意力，参考实现必须分块解码——按 diffusers 默认 tiling（空间 tile 256 px / stride 192，时间 tile 16 + 1 帧 / stride 12）：720P 129 帧 = 308 个 tile，重叠区重复计算 ×2.64 计入（权重每 tile 重读）。
- Open-Sora VAE v1.2：先时间 VAE（MAGVIT-v2 型，潜空间分辨率，时间 ×2 ×2），再 SD-VAE 2D 解码器逐帧（102 帧，含逐帧中间块注意力）。
- MiniMax-H3（0.47 按发布更正）：ViT 解码器（36 层、2048 宽、32 头）。时间上 `5n + 2` 个潜帧解成 n 段、每段 `5 + 2` 个潜帧（`clip_length` 17、`token_drop` 3 → `tokens_chunk_size` 5 + `token_overlap` 2，diffusers `AutoencoderKLMiniMaxH3._decode`），默认 37 潜帧 = 7 段；空间上**总是分块**——发布配置 `FL2VA/video_vae/config.json` 为 `vae_decoder_tiling = 1`、tile 256 px、最小重叠 64 px（diffusers 文档：「spatial tiling is on by default … disabling tiling changes the output」），tile 数取能覆盖且重叠 ≥ 64 px 的最少整 tile（768 px → 4、1344 px → 7）；每 tile 一次 ViT 前向，token = 7 × 16 × 16 + 4 个 register + 1 个 cls = 1797，tile 内全注意力。默认 7 段 × 28 tile = 196 次前向（重叠 ×2.35），解码 1741 → 1893 TFLOP、22.6 → 24.4 s；0.44–0.46 按「8 段 × 5 潜帧、不分块」计，偏差来自没读到发布配置的分块。音频 VAE（BigVGAN 型）按 40 Hz 潜变量 → 32 kHz 波形的 1D 卷积 / 转置卷积，按声道数（2）逐路解码「假设」，反走样激活每输出元素 60 次向量操作「假设」。
- 范数 / 激活 / 残差：每个卷积输出元素 8 次向量操作「假设」。
- 激活 dtype：Wan 参考实现以 fp32 运行 VAE（WanVAE 默认 float32，diffusers 示例亦然）→ 按 fp32；其余按 DiT 的激活 dtype。

**调度「假设」**：组件与去噪循环串行执行（`T_clip = T_text + T_denoise + T_decode (+ T_reload)`；多卡分块解码与跨请求重叠是 0.47 的可选项，见下，默认关——吞吐按单请求串行计）；DP 时每个副本处理自己的 batch 份额。权重按加载的整个检查点计（含 VAE 编码器）。激活峰值 = 一个解码块内最大算子的输入 + 输出（causal 缓存逐潜帧解码：Wan / Mochi 逐潜帧「假设」、CogVideoX 每 2 潜帧、HunyuanVideo 每 tile、H3 每块；LTX 整段），解码在去噪之后运行，所以只在超过 DiT 激活时增加容量需求。

**组件放置（0.45，`workload.placement`，Web「组件放置」，CLI `--placement`）**——都是参考实现提供的运行方式：

| 放置 | 每卡存储 | 额外时间 | 依据 |
|------|------|------|------|
| `resident` 常驻 | 文本编码器在首流水级卡、VAE（+ 音频 VAE）在末流水级卡，与 DiT 同时常驻：需求 = DiT + 组件 + max(0, 组件激活 − DiT 激活) | — | 0.44 行为 |
| `shard` 文本编码器分片 | 文本编码器权重切到本副本全部 c = PP·TP·SP 张卡：te_w / c + 2 层（预取）；VAE 仍在末级卡 | 每卡逐层 all-gather：`max(编码计算, te_w·(c−1)/c / 链路) + α × 层数` | Wan `--t5_fsdp`（FSDP 包装 T5，每 rank 编码同一提示） |
| `offload` 顺序卸载 | 组件与 DiT 分时占用：需求 = max(DiT, 预留 + 文本编码器 + 其激活, 预留 + VAE + 其激活)；Wan2.2 的空闲专家也停在主机 | 每请求主机 → 卡重载 文本编码器 + DiT 级权重 + VAE：`Σ / host_GBps`（默认 50 GB/s「假设」= PCIe 5.0 x16 有效；只计 H2D，主机保留副本「假设」） | diffusers `enable_model_cpu_offload`、Wan `--offload_model`（Wan2.2 在噪声边界换专家） |
| `shard+offload` | 两者 | 两者 | — |
| `auto`（默认） | 依次取 resident → shard（c > 1）→ offload → shard+offload 中第一个放得下的；都放不下取需求最小者 | — | 结果与警告中写明所选放置 |

64 GiB LPDDR5X（100T）上的效果：

| 模型 · 布局 | 常驻需求 | auto 选择 | 需求 | 延迟增加 |
|------|------|------|------|------|
| `wan2.1-14b` 单卡 | 72.5 GiB ✗ | offload | 61.4 GiB ✓ | +1.38 s（7403 s 中） |
| `wan2.2-a14b` 单卡 | 125.7 GiB ✗ | offload（空闲专家停在主机） | 61.4 GiB ✓ | +2.52 s |
| `wan2.2-a14b` PP2 | 69.6 GiB ✗ | offload | 32.4 GiB ✓ | +1.39 s |
| `minimax-h3` 单卡 | 113.7 GiB ✗ | offload | 63.2 GiB ✓（Qwen3-VL 62.1 GiB 单独占满，余量很小） | +2.36 s |
| `minimax-h3` PP2·TP2 | 74.7 GiB ✗ | shard（每卡 te_w / 4 + 2 层） | 39.4 GiB ✓ | +0（all-gather 被编码计算覆盖） |
| `hunyuanvideo` 单卡 | 44.8 GiB ✓ | resident | 44.8 GiB ✓ | 0 |

组件激活按单卡计。

**DiT 权重 FSDP（0.46，`workload.dit_fsdp`，Web「DiT 权重 FSDP 分片」，CLI `--dit-fsdp`，默认关）**——Wan 多卡推理的 `--dit_fsdp`（与 `--ulysses_size` 一起用）：每个流水级的 DiT 权重在该级的 g = SP·DP 张卡之间分片（TP 已切分的部分再切 g 份），每卡存 w / g + 2 层（预取缓冲）；每次流水级前向逐层 all-gather 本级的活动权重（payload w / g，`allgather` 代价 (g−1)·w/g / 链路），与计算按 `max(计算, DRAM, 链路)` 重叠，每层一次 α；gather 结果写入 DRAM（额外 DRAM 写 (g−1)/g · w）后照常流式读取。Wan2.2 的空闲专家同样分片存储，但不 gather。需要 SP·DP > 1，否则忽略并警告；只用于视频 DiT。CFG 的 cond / uncond 在本工具中作为一个 batch 前向，每 tick gather 一次（Wan 参考实现分两次前向，链路量 ×2——400 GB/s 下仍远小于计算）。例（100T + LPDDR5X 64 GiB）：Wan2.1-14B SP2 每卡 53.2 → 26.6 GiB 权重，需求 57.8 GiB（auto→offload）→ 45.1 GiB（常驻）；SP8 60.6 → 25.2 GiB；延迟变化 < 0.1%。

**文本编码器放主机 CPU（0.46，`workload.te_cpu`，Web「文本编码器放主机 CPU」，CLI `--te-cpu`，默认关）**——Wan `--t5_cpu`：卡上不放编码器权重与激活；编码时间 = 编码器 FLOPs / `workload.host_TFLOPS`（默认 2 TFLOPS，「假设」——主机 CPU 的有效算力差别很大，请按实测填写；CLI `--host-TFLOPS`）；嵌入拷到卡上可忽略。此时 shard 无意义，auto 只在 resident → offload 之间选。例：Wan2.1-14B 单卡 64 GiB 常驻放得下（61.9 GiB，Wan README 的单卡方案 `--offload_model True --t5_cpu`），umT5 9.7 TFLOP → 4.8 s（2 TFLOPS），不需要每请求重载。

**VAE 分块解码（0.45，`workload.vae_tiling`，默认关）**：按 diffusers `enable_tiling()` 的默认参数做空间分块，重叠区重复计算、每 tile 重读权重、激活峰值按 tile——CogVideoX：tile 240 × 360 px（= 采样尺寸 / 2），重叠因子 1/6、1/5 → 潜空间 30 × 45、stride 25 / 36（480 × 720：9 个 tile，重叠 ×1.40，解码 315 → 441 TFLOP，激活峰值 2.31 → 0.58 GiB）；Mochi：tile 256 px、stride 192 px → 潜空间 32 / 24（480 × 848：15 个 tile，×1.65，1053 → 1737 TFLOP）；Wan（0.46，diffusers `AutoencoderKLWan`：tile 256 px、stride 192 px → 32 / 24；官方 Wan 仓库不分块）：720P 28 个 tile，×1.65，639 → 1042 TFLOP，解码 41.3 → 67.8 s，激活峰值 3.81 → 0.27 GiB。LTX-Video（0.47，diffusers `AutoencoderKLLTXVideo`：tile 512 px、stride 448 px，空间压缩 32 → 潜空间 16 / 14；`enable_tiling()` 不开逐帧解码）：704 × 512 → 4 个 tile（16 + 2 行 × 16 + 8 列），×1.23，15.3 → 18.8 TFLOP，0.72 → 0.89 s，激活峰值 1.19 → 0.86 GiB。0.47 起分块循环与 diffusers 逐字一致：任一轴超过 tile 即两轴都按 stride 从 0 走，短轴会多出一条窄 tile（上例的 2 行）；0.45–0.46 在短轴不切，默认分辨率下两者相同。HunyuanVideo 与 MiniMax-H3 按发布总是分块（见上）。Open-Sora VAE v1.2 的发布实现没有空间分块（2D VAE 每次 4 帧、时间 VAE 17 帧一块，已按此计），diffusers 也未收录：开启时给出警告并按不分块计。

**对结论的影响**：长视频 / 高分辨率下解码占整段时间 0.6–16%（Open-Sora 720p 的逐帧 SD-VAE 最重），文本编码 < 0.5 s；存储影响大——常驻时 64 GiB LPDDR 上 Wan2.1-14B（fp32 DiT 57 GB + umT5 11.4 GB）、Wan2.2 PP2 / TP2 与 MiniMax-H3 放不下；0.45 的 auto 放置按参考实现的卸载 / FSDP 方式把它们放下，代价是每请求 1–2.5 s 的主机重载（相对 30 min 级的整段可忽略）。更正 0.44 文档：Qwen3-VL 文本塔 66.7 GB = 62.1 GiB，单独并未超过 64 GiB，只是与 DiT 同时常驻放不下。

**多卡分块并行解码（0.47，`workload.vae_parallel`，Web「VAE 多卡分块并行解码」，CLI `--vae-parallel`，默认关）**——依据 MiniMax-H3 发布的 `FL2VA/video_vae`（`vae_parallel_tiling = 1`，`klvae.tiled_decode`）：每个独立的分块调用（一「轮」：H3 的一个时间段、HunyuanVideo 的一个时间 tile、diffusers `enable_tiling()` 的整次空间分块）内，rank r 解第 r、r + N、… 个 tile，再把解码后的像素 tile all-gather 到每张卡做融合。tile 互相独立，算术与单卡相同（请求 FLOPs 不变）；时间 = 最慢卡的 tile（按轮询分配，整除不了时有尾部不均）+ 每轮一次 all-gather（每卡载荷 = 最大份额的解码像素 × 解码 dtype，`allgather` 代价 + α）；N = 本副本的 PP·TP·SP 张卡（解码在去噪之后，全部卡空闲）；VAE 权重每卡一份（放置时 VAE 计入每个流水级）。需要分块解码（H3 / HunyuanVideo 总是分块，其余要同时开 `vae_tiling`），否则忽略并警告；音频 VAE 不拆。H3 以外是设计选项：xDiT 的 DistVAE（patch 并行 + halo 交换）只用于图像 VAE，其 CogVideoX 示例明确不支持并行 VAE。例（100T + HBM3E）：H3 SP 2 / 4 / 8 解码 24.4 → 12.2 / 6.11 / 3.49 s（SP8 每段 28 tile 分到 8 卡、最慢卡 4 个，all-gather 共 4.5 ms）；SP8 时 HunyuanVideo 221 → 34.0 s，Wan 分块 67.8 → 10.4 s，Mochi 分块 70.0 → 13.7 s，CogVideoX 分块 23.2 → 4.7 s（9 个 tile、最慢卡 2 个）。

**跨请求重叠（0.47，`workload.overlap`，Web「跨请求重叠」，CLI `--overlap`，默认关）**：只在文本编码器放主机 CPU（`te_cpu`）时生效——主机编码下一请求与卡上去噪 / 解码当前请求并行，两者是独立资源；队列饱和时稳态周期 = `max(T_clip − T_text, T_text)`，单段延迟不变，吞吐（requests/s、帧/s/卡、段/小时/卡）按周期计。组件与 DiT 同卡时没有独立资源可重叠（卡上的编码 / 解码与去噪抢同一阵列与 DRAM），按串行计并警告；卸载的主机重载仍串行（参考实现是同步拷贝）。分量很小：H3 + 主机编码 2 TFLOPS，单卡 1953 s 一段，周期 1937 s（−0.8%）；Wan2.1-14B SP8 −0.5%。

**仍未建模**：调度器逐元素更新与 CFG 组合；提示词改写（如 H3-Context-IR）；不分块的 VAE 的 patch 并行解码（xDiT DistVAE 式 halo 交换，视频 VAE 无参考实现）；组件同卡时的跨请求重叠（需要抢占式调度，本工具不做）；主机 CPU 编码器只有一个算力旋钮（内存带宽、NUMA、线程数未建模）。

## 12. 蛋白质结构预测（0.43）

以下发布从官方检查点的张量头建模（PyTorch zip 检查点只读中央目录与 `data.pkl`，AlphaFold 2 的 JAX `.npz` 在发布 `.tar` 内按 tar 头 → zip 中央目录 → `.npy` 头读取；均为 HTTP range 请求，不下载权重；`scripts/fetch_torch_ckpt.py`、`scripts/fetch_npz_header.py`、`scripts/summarize_ckpt.py`）。参数与发布逐项一致（按构造：每个 ≥ 2 维权重是一个 GEMM，其余计为杂项参数）。

| id | 发布（检查点） | 参数 | 发布 dtype | 默认工作负载 | 覆盖 |
|----|------|------|-----------|------|------|
| `esmfold` | facebook/esmfold_v1（`pytorch_model.bin`） | 3.528B（ESM-2 3B 2.84B + 折叠部分 0.69B） | ESM-2 fp16、折叠部分 fp32 | 512 残基，单序列，主干 5 遍（max_recycles 4 + 首遍） | 完整 |
| `alphafold2` | google-deepmind/alphafold（`alphafold_params_2022-12-06.tar` → `params_model_1_ptm.npz`，CC BY 4.0） | 93.24M | fp32 | 512 残基；MSA 聚类 508 行（512 − 4 模板）、extra MSA 5120 行、模板 4；主干 4 遍 | 部分 |
| `openfold` | aqlaboratory/openfold（`finetuning_ptm_2.pt`） | 93.24M | fp32 | 512 残基；MSA 512 行、extra MSA 1024 行、模板 4；主干 4 遍 | 部分 |
| `boltz-1` | boltz-community/boltz-1（`boltz1_conf.ckpt`） | 606.38M | fp32 | 512 残基、每残基 8 原子「假设」；MSA 4096 行；主干 4 遍；扩散 200 步 × 1 样本 | 部分 |
| `protenix` | bytedance/Protenix（`model_v0.5.0.pt`） | 368.09M | fp32 | 512 残基、8 原子 / 残基「假设」；MSA 2048 行；主干 4 遍；扩散 200 步 × 5 样本 | 部分 |

覆盖「部分」的原因：只评估神经网络推理——MSA / 模板检索（jackhmmer / HHblits / MMseqs2，CPU 或检索服务）与特征化、AMBER 松弛不在范围内。ESMFold 是单序列模型，没有这些步骤（「完整」）。AlphaFold 2 monomer 预设跑 5 个模型：这里评估一个模型的一次预测。

**表示与 GEMM 行数**：结构模型同时维护几种网格——单一表示（N 个残基 / token）、pair 表示（N² 个位置）、MSA 表示（行数 S × N）、模板（T × N²）、原子（A ≈ 8N）与原子对（局部窗口 A/32 × 32 × 128）。每个权重 GEMM 按名字归到一种网格，行数即该网格大小（AF2 的 `msa_att_row.linear_z` 是 pair 行，`outer_product_mean.linear_out` 是 pair 行，模板 pair 栈是 T·N² 行……）；每个线性层另计 LayerNorm / 门控 / 偏置的逐元素工作（「假设」每输入、输出元素各 4 次）。

**激活 × 激活的核**（按模块名检测，头数 / 维度取自张量形状）：
- 三角乘法（outgoing / incoming）：c 个 N×N×N 批矩阵乘，`2·N³·c` FLOPs；
- 三角注意力（起点 / 终点）：N 组、每组 N 个 query 对 N 个 key，带 pair 偏置：`2·2·N³·h·d`；
- MSA 行注意力（S 组 × N 对 N，pair 偏置）、列注意力（N 组 × S 对 S；AF2 extra MSA 为全局列注意力，每列 1 个 query）；
- 外积均值：`(N·c) × S × (N·c)` 批矩阵乘（`2·S·N²·c²`，c = 32 时是 Evoformer 中最大的单项之一）；
- AF3 类 MSA 模块的 pair 加权平均（每头 N×N 权重乘 S·d 的值）；
- 带 pair 偏置的单一表示注意力（Pairformer、扩散 transformer、ESMFold 主干）；
- IPA（不变点注意力）：qk 维 = 标量 + 点坐标，v 维 = 标量 + 点 + pair 值（AF2：12 头，qk 16 + 12，v 16 + 24 + 128）；刚体更新、扭转角、坐标重建的几何运算未计；
- 模板点注意力（每个 pair 位置对 T 个模板）、原子局部窗口注意力（32 query × 128 key）。

**每请求的执行次数**（评估器把一层的算子和乘以执行次数；算子本身是一次执行的）：主干（嵌入、模板栈、extra MSA 栈、Evoformer / Pairformer、AF2 / ESMFold 的结构模块）每遍都重跑，遍数 = recycle + 1；结构模块 8 次迭代共享权重；扩散模块（原子编码器 → token transformer → 原子解码器）每步一次，样本在 token / 原子网格上成批（行数 × 样本数），pair 条件对样本共享（参考实现按一次计）；置信度头每样本一次（全部网格 × 样本数）；输出头一次。ESM-2 语言模型（ESMFold）每请求一次。

**工作负载**（`workload` 块，0 = 发布默认）：`seq_len` 残基、`msa` MSA 行（AF2 / OpenFold 为聚类行；extra MSA 保持默认）、`recycles` 主干遍数（含首遍）、`steps` 扩散步数、`samples` 扩散样本数；MSA 行数默认按上限计「假设」（浅 MSA 更快）。批延迟 SLO 单列为 `fold_slo_s`（默认 120 s「假设」；ESM-2 编码器仍用 `seq_slo_ms`）。

**并行（0.45：DAP）**：布局为 PP × DP × DAP，DAP 占用布局的 SP 维（`layout.sp`，Web 显示为「DAP」）；pair 的 TP 未建模（c_z = 128 通道太窄，FastFold 也不切通道），TP 被拒绝并说明原因（0.47 复核：在 Boltz-1 / Protenix / OpenFold / FastFold 的发布实现里没有找到 pair 通道切分，DAP + 样本分卡已覆盖多卡扩展，维持不做）。DAP 按 FastFold 的动态轴并行（Cheng et al., *FastFold*, 2022）：pair / 模板网格沿第一个残基轴、MSA 网格沿行（行注意力）或列（列注意力）切到 D 张卡，权重每卡完整复制；单一 / token / 原子轨道（结构模块 IPA、扩散 transformer、原子窗口注意力、置信度单一轨道）每卡重复计算（`Op.replicated = D`，请求 FLOPs 不重复计）。每个核的通信（每卡载荷，与其他集合通信一样按 `max(计算, DRAM, 链路) + α` 与计算重叠）：

| 核 | 通信 |
|------|------|
| 三角乘法（出 / 入） | all-gather 一个投影操作数 N × N × c |
| 三角注意力（起 / 止） | all-gather pair 偏置 N × N × H + 一次 pair 表示 all-to-all 转置（N² × c_z，起点 ↔ 终点） |
| MSA 行注意力 | all-gather pair 偏置（MSA 按行切，注意力本地） |
| MSA 列注意力 | 两次 MSA all-to-all 转置（S × N × c_m，行切 ↔ 列切） |
| 外积均值 | all-gather 右投影 S × N × c2；AF3 类 MSA 模块（无列注意力，MSA 保持行切）先加一次 MSA all-to-all |
| pair 加权平均 | all-gather pair 导出的权重 N × N × H |
| 带 pair 偏置的单一轨道注意力 | all-gather 偏置 N × N × H（单一轨道本身重复） |

pair 残差 / 在途激活与跨 stage 传输按 ⌈N / D⌉ × N × c_z。100T + HBM3E、默认工作负载、batch 1，DAP 1 / 2 / 4 / 8：

| 模型 | 批延迟 | 8 卡加速 | 激活常驻（DAP 1 → 8） | 每卡链路字节（DAP 8） |
|------|------|------|------|------|
| `esmfold` | 5.72 / 2.90 / 1.50 / 0.80 s | 7.1× | 0.38 → 0.09 GiB | 8.8 GB |
| `alphafold2` | 14.17 / 7.10 / 3.57 / 1.81 s | 7.8× | 1.63 → 0.25 GiB | 16.3 GB |
| `openfold` | 12.62 / 6.32 / 3.18 / 1.61 s | 7.8× | 0.69 → 0.16 GiB | 15.0 GB |
| `boltz-1` | 13.05 / 7.64 / 4.94 / 3.59 s | 3.6× | 2.07 → 1.13 GiB | 15.2 GB |
| `protenix` | 19.33 / 10.65 / 6.32 / 3.22 s（0.46 样本分卡；0.45 不分卡 14.37 / 11.89 / 10.65 s） | 6.0×（不分卡 1.8×） | 1.63 → 0.40 GiB | 13.4 GB |

主干主导的 AF2 / OpenFold / ESMFold 接近线性；Boltz-1 / Protenix 的 200 步扩散 transformer 在单一 / 原子轨道上，DAP 不切分它。

**扩散样本分卡（0.46，`workload.sample_split`，默认开；Web「DAP 时扩散样本分卡」，CLI `--no-sample-split` 关闭）**：DAP > 1 且样本数 S > 1 时，扩散模块（`samples = "tok"` 的块：原子编码器、扩散 transformer、原子解码器）的 S 条轨迹分到 D 张 DAP 卡，每卡 ⌈S / D⌉ 条——样本之间相互独立，条件（单一表示与 pair 条件）在 DAP 下已在每张卡上，所以这是精确的并行、不增加通信；块内 pair 网格的工作（pair 偏置投影）仍按 DAP 切分，偏置 all-gather 不变；D > S 时多出的卡空闲（`Op.replicated = D·⌈S/D⌉ / S`，请求 FLOPs 不变）。置信度头（`samples = "all"`，每样本一份 pair 副本）仍按 DAP 切 pair、不分样本。参考实现（Boltz、Protenix）没有现成的多卡样本并行，这里作为布局设计选项给出。100T + HBM3E，DAP 1 / 2 / 4 / 8：Protenix（5 样本）19.33 / 10.65 / 6.32 / 3.22 s（6.0×；不分卡为 1.8×）；Boltz-1 默认 1 个样本（`--diffusion_samples` 默认 1）不变（3.6×），取 5 个样本时 DAP 8 为 26.8 → 4.3 s（不分卡 12.6 s）。ESMFold 的 ESM-2 语言模型层在 SP 维上按 Ulysses 切分（与 ESM-2 相同）。批延迟 `T = (mb + PP − 1) × t_stage`，与 ESM-2 相同。「假设」：400 GB/s 链路下通信基本被计算覆盖（链路变慢时会成为瓶颈，可在链路参数里试）；没有可逐项核对的公开 DAP 推理时延，故不加校验行。

**dtype**：发布权重 fp32（ESMFold 的 ESM-2 为 fp16）；激活按 bf16「假设」（参考实现：ESMFold / OpenFold / AF2 单体 / Boltz-1 为 fp32，Protenix 默认 bf16）。0.44：fp32 激活用激活 dtype what-if 评估（Web「激活 dtype」/ CLI `--act fp32` / 场景 `formats_override: [["act", "fp32"]]`，标注 what-if）——激活存储、DRAM 流式、SRAM 端口读写（含输出）按 4 B；计算仍在 bf16 阵列上，激活逐 GEMM 转换的开销计入向量单元，所以它是「fp32 数据搬运 + bf16 计算」的代价，数值上不等价于 fp32 矩阵乘。例：ESMFold 512 残基、100T + LPDDR5X（DRAM 受限）6.3 s → 12.5 s；HBM3E 上 MAC 受限，延迟不变、容量 +0.4 GiB。芯片无 fp32 MAC：逐 GEMM 转换为 bf16，开销计入向量单元（与 LLM 反量化同一规则）。

**校验**：`validate` 增加 ESMFold 行——论文（Lin et al., Science 2023）：单 V100 上 384 残基 14.2 s；按本模型的 FLOPs（62.2 TFLOP，5 遍主干）折算为 V100 fp32 峰值（15.7 TFLOPS，主干为 fp32）的 28%，落在 [0.05, 0.8] 带内。测试核对：参数逐项一致；AF2 官方 JAX 参数映射后的逐层 GEMM 与 OpenFold 检查点完全相同；每块检测到的核（Evoformer：三角乘法 ×2、三角注意力 ×2、行 / 列注意力、外积均值；AF3 类 MSA 模块：pair 加权平均代替行注意力）；核 FLOPs 与闭式一致；recycle / 扩散步数 / 样本数的线性缩放；ESMFold 中 ESM-2 每请求只算一次。

**100T 芯片（默认 1 GHz、64 MiB SRAM）单卡 batch 1 的默认工作负载**：ESMFold 121 TFLOP / 5.1–6.3 s；AlphaFold 2 396 TFLOP / 11.7–14.2 s；OpenFold 354 TFLOP / 11.0–12.6 s；Boltz-1 296 TFLOP / 8.8–13.1 s；Protenix 419 TFLOP / 11.5–19.3 s（范围 = os / 可重构映射 × LPDDR5X 273 GB/s / HBM3E）。有效 MAC 21–36%：pair 网格上的 GEMM 是 K = N = 128 的窄矩阵，三角注意力 head_dim 只有 32。

**未建模 / 近似**：MSA / 模板检索与特征化、松弛；IPA 与扩散的几何 / 噪声调度向量运算；分块（chunk / subbatch）只降低峰值显存、不改计算量，峰值激活按单个算子计；AF2 的模板扭转角嵌入按每残基一行（实际 T × N 行，量很小）；原子数按每残基 8 个重原子「假设」；Boltz-1 / Protenix 的多链 / 配体 token 化按纯蛋白质计。

## 13. 能耗：动作计数 × 用户能耗表（0.47.1）

引擎本来就在数每个部件做了什么；`core/energy.py` 只做乘法（Accelergy 式能耗动作表）。**工具不内置任何能耗数值**：能耗表每项默认空，由用户按实测、厂商数据或自己认可的文献填写；空项不计并列为「未提供」。不填任何项时仍给出动作计数——这是模型输出，不需要假设。

| 动作 | 计数（全系统：所有流水级、TP·SP·DP 的每个 rank） | 能耗表项 |
|------|------|------|
| MAC | bf16 等效 MAC 槽 = 有效 FLOPs / 2 / 该格式的速率倍数（fp8 在 2× 速率下记 ½）；分块填不满的空闲槽不计 | `pJ_mac` |
| 向量 | 向量元素操作（范数、softmax、激活、格式转换 / 反量化） | `pJ_vec` |
| SRAM | 经 SRAM ↔ 数据通路端口的字节（各 GEMM / 注意力分块的供数流量，即 FEED 界的分子） | `pJ_bit_sram` |
| SLC | 系统级缓存命中的字节（0.48，§14；无 SLC 时为 0） | `pJ_bit_slc` |
| DRAM | 外部存储读写字节（权重、KV、激活流式、FSDP gather 写入）；有 SLC 时只计未命中 | `pJ_bit_dram` |
| D2D | 封装内 die-to-die 层发送的字节（0.48，§15；D2D 关或每封装 1 die 时为 0） | `pJ_bit_d2d` |
| 节点内链路 | 每 rank 在节点内 scale-up 层发送的字节（集合通信载荷、PP 交接、VAE tile all-gather；扣除 D2D 与跨节点份额） | `pJ_bit_link` |
| 跨节点网络 | 跨节点发送的字节（0.50，§15；单节点时为 0） | `pJ_bit_net` |
| 静态 | 卡数 × 时间窗 | `idle_W`（W / 卡） |

**时间窗与单位**：LLM decode 一个步（全部微批与副本）→ token；prefill 一次（TTFT）→ prompt token；视频一次请求批：步数 × DiT 各级 + 卡上的文本编码器 + VAE 解码，时间窗 = 单段延迟（开跨请求重叠时为稳态周期）→ 帧（主机 CPU 编码器的能耗不计）；蛋白质一次前向 → 序列。每级每遍执行「微批数」个 tick，tick 的计数是单 rank 的，乘 TP·SP·DP。PP 下每个微批都重新流式读取本级权重（与级时间一致），所以 PP2 的 DRAM / token 约为 PP1 的 2 倍——这是映射的真实代价，不是重复计数。

**入口**：API `POST /api/eval` 的 body 顶层 `"energy": {"pJ_mac": …, "idle_W": …}`（不放进 scenario，不影响场景哈希与结果缓存），响应 `energy` 字段（`counts_per_unit`、`J_per_unit`、`J_by_action`、`avg_W_per_card`、`provided` / `missing`）；CLI `--pJ-mac --pJ-vec --pJ-bit-sram --pJ-bit-dram --pJ-bit-link --idle-W`（0.48 加 `--pJ-bit-slc --pJ-bit-d2d`，0.50 加 `--pJ-bit-net`）；Web 左侧「能耗」组，单点评估页的「动作计数与能耗」表。

**不做**：动态 / 静态功耗的工艺推导、DVFS、SRAM 容量相关的单次访问能耗（一项一个数）、DRAM 行激活 / 刷新、主机与 PUE、面积与成本。SLC 与两级互连已在 0.48 接入（§14、§15）；新的流量守恒：SLC + DRAM = 无 SLC 时的 DRAM 字节，D2D + 节点内 + 跨节点 = 单层时的链路字节（载荷口径不变，只是按层拆分）。

## 14. 系统级缓存 SLC（0.48，默认关）

芯片字段 `slc_mib`（每卡容量，0 = 无，默认）、`slc_GBps`（命中带宽，默认 2000 GB/s）、`slc_policy`（`pin` | `lru`，默认 `pin`）。**全部数值与规则都是「假设」**，用于看「在 SRAM 与 DRAM 之间加一层缓存」这个设计选择的量级，不是某款芯片的标定。

- **位置与容量**：每卡一份，位于片上 SRAM 与 DRAM 之间；包含式（所有数据仍有 DRAM 副本），**不增加容量**，DRAM 需求与容量检查不变。SLC 不小于 DRAM 容量时给出警告。
- **时间**：命中字节 / `slc_GBps` 作为与 DRAM 端口并行的一项 `t_slc` 进入级时间 `max(t_compute, t_dram, t_slc, t_link) + t_sync`；DRAM 只计未命中字节。SLC 带宽太低时瓶颈标为 SLC。
- **pin（软件钉住，默认）**：与 SRAM 规划同一顺序——SRAM 放不下的热权重 → 路由专家 → KV / 线性注意力状态依次钉在 SLC；每步读到钉住部分即命中（命中比例 = 钉住字节 / 该类存储字节，路由专家按存储比例）。流式激活、FSDP gather 写入、embedding 查表行、KV 写入都绕过 SLC（write-through）。全序列前向里的权重分块重读按权重的 SLC 份额命中。
- **lru（硬件替换）**：一步 decode / 去噪的访问是循环重复的，LRU 要么装下整个工作集、要么反复抖动。规则：SRAM 之外的全部数据（权重 + KV / 状态 + DRAM 中的激活工作集）放得下 → 所有读命中（KV 写仍进 DRAM）；放不下 → 0 命中（循环访问的抖动下界）。真实替换策略介于 lru 与 pin 之间。
- **不覆盖**：视频 pipeline 组件（文本编码器、VAE）不用 SLC；多卡共享 SLC、一致性流量、写回策略差异、SLC 分 bank / 冲突、标签开销都不建模。
- **能耗**：动作 `slc`（命中字节，`pJ_bit_slc`），DRAM 只计未命中；SLC + DRAM = 无 SLC 时的 DRAM 字节（测试守恒）。
- **结果字段**：`StageTime.t_slc / slc_bytes`，`MemPlan.slc / slc_policy / slc_hot / slc_expert / slc_kv / slc_all / slc_residency`，`dram` 字典增加 `slc` 与 `slc_parts`；API 每级 `t_ms.slc`、`slc_GB`、`mem.slc_*`。CLI `--slc-mib --slc-GBps --slc-policy`；Web 芯片组「SLC MiB / GB/s / 策略」，级表与存储表在有 SLC 时多出 SLC 列；扫描可选 `chip.slc_mib`、`chip.slc_GBps`。
- **量级**（100T + LPDDR5X 4×64 8533，Qwen3-8B decode batch 8，存储 ≈ 16 GB 权重 + KV）：TPOT 104.2 ms → pin 1 / 4 / 8 / 16 GiB 98.5 / 81.7 / 59.2 / 20.8 ms；lru 16 GiB 放不下 → 104.2 ms（不变），32 GiB 放得下 → 20.0 ms。

## 15. 三层互连：封装内 D2D（可选）、节点内 scale-up、跨节点网络（0.48 两级；0.50 三级）

| 层 | 场景字段 | 默认 | 何时有流量 |
|----|----|----|----|
| 封装内 die-to-die（D2D） | `d2d_enabled`、`package_cards`（每封装 die 数）、`d2d_std`（档位，见下表）、`d2d_units`（每 die 的 D2D 单元数）、`d2d.alpha_us`；`d2d_std = "custom"` 时用 `d2d.GBps` | **关**（单片大 die：每卡一个封装）；打开后默认 UCIe-A x64 @ 48 GT/s × 4 模块 = 1536 GB/s、α 0.5 µs「假设」 | `d2d_enabled` 且 `package_cards > 1` |
| 节点内 scale-up（封装之间） | `link` | 400 GB/s、α 3 µs「假设」（与 0.47 相同） | 组跨封装、不跨节点 |
| 跨节点 scale-out（IB / RoCEv2 一类） | `node_cards`（每节点卡数）、`net` | `node_cards = 0` = 整个系统一个节点（不用）；`net` 50 GB/s（每卡一张 400 Gb/s NIC）、α 5 µs「假设」 | `node_cards > 0` 且组跨节点 |

默认（D2D 关、单节点）就是 0.47 起的单层模型，LLM / 视频 / 蛋白质结果逐字节不变（1356 项指纹）。D2D 关时 `package_cards` 不生效（警告）；Web 关掉 D2D 时把每封装 die 数复位为 1。`node_cards` 须是 `package_cards` 的整数倍（D2D 开时）。0.48 / 0.49 的场景 JSON：`package_cards > 1` 且没有 `d2d_enabled` → 视为 D2D 开；给了 `d2d.GBps` 而没有 `d2d_std` → 视为自定义档位（照旧求值）。

- **卡的编号**：TP 最内层，其次 SP、DP，PP 最外层；连续 `package_cards` 张卡为一个封装，连续 `node_cards` 张卡为一个节点。于是 TP / ETP 组步长 1，SP（Ulysses all-to-all）与 DAP 组步长 TP，EP 组步长 ETP，DiT FSDP 组（SP·DP）步长 TP，PP 交接跨越 TP·SP·DP 张卡。IR 的通信算子带 `comm_stride`。
- **组如何落到三层**：g 个 rank、步长 s 的组在一个封装内有 k₁ = gcd(clamp(P // s, 1, g), g) 个成员，在一个节点内有 k₂（同样规则用 N，取 k₁ 的倍数）个成员 → 每层规模 n₀ = k₁（D2D）、n₁ = k₂ / k₁（节点内）、n₂ = g / k₂（跨节点）；规模 1 的层略去，只剩一层时就是该层上的单层公式。多层时按 NCCL 式分层，记 B₍ᵢ₎ = B / Π_{j<i} n_j：
  - all-reduce = 逐层 reduce-scatter、最外层 all-reduce、逐层 all-gather：`Σᵢ 2(nᵢ−1)/nᵢ · B₍ᵢ₎ / βᵢ`，α = 内层各 2αᵢ + 最外层 α；
  - all-gather = 先最外层再向内：`Σᵢ (nᵢ−1) · B · Π_{j>i} n_j / βᵢ`，α = Σαᵢ；
  - all-to-all：各层同时发送各自负责的目的地（第 i 层占 (nᵢ−1)·Π_{j<i} n_j / g 的数据），取各层时间的 max，α = max αᵢ；
  - PP 交接：同封装走 D2D，同节点走节点内，否则走跨节点；
  - 两层（D2D + 节点内）时与 0.48 的公式逐位相同（测试核对）。
  - 不在同一步长下的组（DP > 1 且 PP > 1 时一个副本的卡不连续）按副本里连续的 TP·SP 块计封装 / 节点内成员数：VAE 多卡解码的 tile all-gather、文本编码器 FSDP 分片的 all-gather 都按此分层。
- **代价也如实计**：分层的 α 逐层累加（D2D + 节点内 all-reduce 4 µs > 单层 3 µs），小载荷的 decode 在部分跨层时可能略慢；大载荷时外层字节减少到 1/k。

### 15.1 D2D 档位（选择器，用户确认的列表）

带宽为**原始速率、每方向、每单元**（未扣 flit / 协议开销），D2D 层带宽 = 每单元 × `d2d_units`（每个 die 面向封装一侧放几个单元是布图选择，「假设」）。pJ/bit 只作参考显示——能耗表仍由用户填（§13）。

| id | 标准 | 单元 | GB/s / 单元 | 参考 pJ/bit | 封装 / 距离 | 来源与「假设」 |
|----|----|----|----|----|----|----|
| `ucie-a-48`（常用，D2D 开时默认） | UCIe 3.0 Advanced（2.5D） | x64 模块 | 384 | 0.25 | 25–55 µm 凸点，≤ 2 mm | 48 GT/s × 64 lane ÷ 8；每 transfer 1 bit「假设」；pJ/bit 为 UCIe 2.0 目标（3.0 未公开） |
| `ucie-a-64` | UCIe 3.0 Advanced（2.5D） | x64 模块 | 512 | 0.25 | 同上 | 同上；64 GT/s 的 BER 目标 1e-12（48 GT/s 为 1e-15） |
| `ucie-s-32` | UCIe Standard（2D 有机基板） | x16 模块 | 64 | 0.5 | 100–130 µm，≤ 25 mm | UCIe 联盟教程原值（32 GT/s 每模块每方向 64 GB/s） |
| `bow-256` | OCP Bunch of Wires 2.0 | 16 线 slice | 32 | 0.5 | 有机基板 | 16 Gb/s/线；pJ/bit 双端端接 < 0.5–1、非端接 < 0.25–0.5（列 0.5） |
| `bow-512` | OCP Bunch of Wires 2.0 | 16 线 slice | 64 | 0.5 | 有机基板 | 32 Gb/s/线 |
| `nvlink-c2c`（厂商专有，参照） | NVIDIA NVLink-C2C | 整条链路 | 450 | 1.3 | — | Grace Hopper 白皮书：900 GB/s 双向合计 = 450 GB/s 每方向 |
| `custom` | 自定义 | — | `d2d.GBps` | — | — | 用户填写 |

调研过但不在选择器里：NVIDIA NV-HBI（Blackwell 双 die，「10 TB/s」未注明方向）、AMD Infinity Fabric AP（MI300X IOD 之间每方向 2.4 / 3.0 TB/s）、TSMC LIPINCON（2019 VLSI 测试芯片，8 Gb/s/pin、320 GB/s、0.56 pJ/bit）；CCIX 等旧接口不列。不虚构厂商 SKU。

- **量级**（1P + HBM3E，Qwen3-8B prefill TP8 4096 tokens × 8，节点内链路 50 GB/s、α 5 µs）：单片 TTFT 676.8 ms（LINK 瓶颈）→ 每封装 8 die：UCIe-A 48G × 4 / 64G × 4 / UCIe-S 32G × 4 / BoW-512 × 4 / NVLink-C2C × 1 均为 247.8 ms（MAC 瓶颈；每级链路时间 22.0 / 16.5 / 132.1 / 132.1 / 75.2 ms），BoW-256 × 4（128 GB/s）264.3 ms（LINK）。三层（1P + HBM3E、默认链路，同一 prefill TP16）：单节点 199.6 ms（MAC）→ 每节点 8 卡、跨节点 50 GB/s 200.2 ms（链路时间 90.6 → 132.9 ms，仍被计算掩盖）→ 每节点 4 卡 218.2 ms（LINK）。Qwen3-30B-A3B decode TP2·DP8·EP16 batch 64：TPOT 5.39 → 每节点 8 卡 5.58 ms（+ D2D 每封装 2 die 5.46 ms）。
- **能耗**：链路字节仍按载荷口径，按每次集合通信各层实际发送字节之比拆成 `d2d`（`pJ_bit_d2d`）、节点内 `link`（`pJ_bit_link`）、跨节点 `net`（`pJ_bit_net`），三者之和 = 单层时的链路字节（测试守恒）。
- **入口**：API scenario `d2d_enabled`、`package_cards`、`d2d_std`、`d2d_units`、`d2d`、`link`、`node_cards`、`net`；每级 `link_GB`（total / d2d / scaleup / net）；`/api/catalog` 给出 `d2d_standards`。CLI `--d2d --package-cards（> 1 隐含 --d2d） --d2d-std --d2d-units --d2d-GBps（隐含 custom） --d2d-alpha-us --link-GBps --link-alpha-us --node-cards --net-GBps --net-alpha-us --pJ-bit-net`，`accel-dse d2d` 列出档位。Web 并行组「三层互连」：D2D 芯粒堆叠开关（关 = 单片大 die）→ 档位 / 每封装 die 数 / 单元数 / GB/s / α；节点内两项；每节点卡数与跨节点两项。扫描可选 `package_cards`（> 1 时自动开 D2D）、`d2d_units`、`d2d.GBps`（自动切到自定义）、`d2d.alpha_us`、`node_cards`、`net.GBps`、`net.alpha_us`。
- **不做 / 待定**：拓扑（胖树 / torus / 轨道优化）、拥塞与超额订阅、in-network reduction（SHARP）、多路径 / 链路聚合、NIC 与 scale-up 之间的 PCIe 瓶颈都不建模；`Link.topology` 仍只是标签；D2D 的协议效率（flit 开销）不扣——需要时用自定义档位填有效带宽。

## 16. MoE 负载倾斜（0.49，默认关）

默认按均匀路由计（§3）。真实服务里热门专家会让某些 EP rank 收到更多 token；所有 rank 在 combine all-to-all 处同步，**最忙的 rank 决定级时间**。两种输入（二选一）：

- `serving.moe_skew`（默认 1，范围 [1, 64]，「假设」）：最忙 EP rank 的 token-专家对 = skew × 平均值 `T·k / EP`，上限为 `T·k`（全部落在一个 rank）与 `本地专家数 × T`（每个专家每步最多 T 个 token）。
- `serving.moe_expert_load`：实测的每个路由专家 token 数（长度 = 专家数，相对值即可）。按连续放置（专家 e 在 rank `e // ceil(E/EP)`，vLLM / SGLang 无 EPLB 时的默认线性放置）算每个 EP 宽度的倾斜 = 最忙 rank 的负载 / 平均负载——于是布局搜索里不同 EP 看到不同倾斜。例：16 个热门专家各 4 倍负载（Qwen3-30B-A3B 128 专家）→ EP 2 / 4 / 8 / 16 倾斜 1.27 / 1.82 / 2.91 / 2.91。

建模（「假设」）：最忙 rank 的专家 GEMM 行数 `m_e = ceil(pairs_hot / hit)`，命中专家数保持均匀路由的期望（多出的负载落在热门专家上），只有 T 行装不下时才增加；dispatch / combine all-to-all 与 ETP all-reduce 的载荷按最忙 rank 计。EP = 1 或非 MoE 模型时忽略并警告。存储不变；EPLB / 冗余专家不建模（填 EPLB 之后的倾斜即可）。

**能耗**：倾斜只在 rank 之间搬工作，不产生新工作——每输出单位的 MAC / 向量 / SRAM / DRAM / SLC / 链路 / D2D 计数取同一场景均匀路由下的值，时间窗（卡·秒、静态能耗）取倾斜后的值。

**量级**（100T + HBM3E，Qwen3-30B-A3B，TP2·DP4·EP8）：prefill batch 64 × 4096 TTFT 4608 → 倾斜 1.25 / 1.5 / 2 / 4：4967 / 5326 / 6048 / 8924 ms。decode batch 64 不变（13.69 ms：最忙 rank 的专家 GEMM 仍在同一个行分块里，时间由权重读取决定）；batch 512：66.7 → 2 / 4：71.7 / 76.6 ms；batch 2048：258.6 → 4：293.0 ms。

**入口**：API / scenario `serving.moe_skew`、`serving.moe_expert_load`（结果 summary 有 `moe_skew` = 实际使用的倾斜）；CLI `--moe-skew`、`--moe-expert-load 文件.json|逗号列表`；Web 服务组「MoE 倾斜」（每专家分布只在 API / CLI）；扫描 `serving.moe_skew`。

## 17. 资源 / 面积预算（0.49，可选设计约束）

请求体顶层 `budget`（与 `energy` 一样不进 scenario，不改哈希），**工具不内置任何工艺库或密度**，每项默认空。`core/budget.py` 只报告余量 `1 − 用量 / 限额` 并标出超限：

| 项 | 用量来自 | 键 |
|------|------|------|
| SRAM / SLC / MAC 单元 / 峰值 bf16 TFLOPS / DRAM 容量（每卡） | 芯片与存储器配置（精确） | `sram_mib`、`slc_mib`、`macs`、`tflops`、`dram_GiB` |
| 卡数（每副本） | 布局 | `cards` |
| 平均功耗（每卡） | 用户能耗表（§13）：能耗 / 时间窗 / 卡数 | `power_W_card` |
| 面积代理（每卡 / 每副本） | `SRAM MiB · mm²/MiB + SLC MiB · mm²/MiB + MAC/1024 · mm²/kMAC + 固定 mm²`，密度全由用户填「假设」 | `die_mm2`、`system_mm2`；密度 `mm2_per_mib_sram`、`mm2_per_mib_slc`、`mm2_per_kmac`、`mm2_fixed` |

**判定**：未超 / 超出 / 无法判定。无法判定 = 缺面积密度、或只有下界且下界未超：能耗表不全时功耗是下界（计数为 0 的项不算缺），未填 `mm2_fixed` 时面积是下界（密度已含固定部分时填 0）。下界已超出则判定为超出。

**与搜索的关系**：batch / 布局搜索仍按 SLO 与容量精确求解，不把预算放进搜索；API 在布局 / 映射对比 / 扫描的每行加 `budget_ok`、`budget_violations`，超预算的布局排在其余之后（「应用最佳布局」因而取预算内的最优），映射对比在前 8 个布局里取预算内最优；容量检查给出的「最少卡数」不超过卡数预算（超出时返回 `budget_cards_limit`）。功耗随 batch 变化，搜索给出的 batch 超功耗时不会自动降 batch——只标出。

**入口**：API `budget`（eval 响应 `budget`：`ok`、`items`、`area`、`violations`）；CLI `--budget-sram-mib --budget-slc-mib --budget-macs --budget-tflops --budget-dram-GiB --budget-cards --budget-power-W --budget-die-mm2 --budget-system-mm2 --mm2-per-mib-sram --mm2-per-mib-slc --mm2-per-kmac --mm2-fixed`（search / compare 表多一列 budget）；Web 左侧「资源 / 面积预算」组、单点页「资源 / 面积预算」表，布局表与映射对比表的「超预算」标记。

**不做**：工艺库 / 标准单元面积模型、布线与良率、成本（$ / mm²、封装、HBM 价格）、面积与频率 / 功耗的耦合——需要可信数据源，留给用户的密度「假设」。

## 18. prefill / decode 分离（PD，0.50，默认关）

默认是合并服务（§7：每个副本分时做 prefill 与 decode）。`scenario.pd.enabled = true` 时另算一份 **PD 分离**报告（DistServe / Splitwise / Mooncake 式），与同卡数的合并服务对照；单点评估本身（级表、TPOT 等）不变。**稳态流体模型「假设」，第一版**；排队 / 尾延迟 / 连续批处理 / 分块 prefill 见 §18.1（0.51）。

- **两个池**：decode 池用场景的 `layout` 与 `serving.batch`，卡数 `pd.decode_cards`（布局卡数的整数倍；0 = 一个副本）；prefill 池用 `pd.prefill_layout`，卡数 `pd.prefill_cards`，每个副本取满足 TTFT SLO 的最大 prefill batch（2 的幂 ≤ 64，与合并 goodput 同一选择）。两池同芯片、同存储器、同互连设定（不同芯片 / 存储器的异构池未做）。
- **容量**：prefill 池 λ_p = r_p · R_p / S（请求 / s，R_p 每副本 prompt tok/s，S prompt 长度）；decode 池 λ_d = r_d · R_d / out_len；KV 传输 λ_kv = min(N_p, N_d) · β / KV。PD 的请求率 λ = min(λ_p, λ_d, λ_kv)，最小者为瓶颈，其余池的利用率 = λ / 各自容量；goodput = λ · out_len / (N_p + N_d)（tok/s/卡）。
- **KV 交接**：每请求 KV = 整个模型 S 个 token 的逻辑 KV + 索引键 + 循环状态（按发布 dtype；decode 池内的 TP 复制扇出不计，「假设」）。走哪一层（「假设」）：设了 `node_cards`（> 0）时两池在不同节点 → 跨节点 `net`；否则同一 scale-up 域 → 节点内 `link`；`pd.kv_GBps` 可直接给每卡带宽。每请求带宽 β_req = min(c_p, c_d) · β（按卡成对并行），时间 t_kv = α + KV / β_req。`pd.kv_layerwise`：prefill 时逐层流式发送，α 每层一次，只暴露 prefill 掩盖不了的部分 max(α + KV/β_req/L, t_kv − TTFT_p·(L−1)/L)。
- **指标**：TTFT = prefill 池 TTFT + 暴露的 KV 传输；TPOT = decode 池的纯 decode 步（没有 prefill 打断）。合并对照：同卡数按场景布局 ⌊N / c_d⌋ 个副本 × 合并 goodput；其 TTFT 为合并 prefill 延迟，**有效 TPOT = TPOT / decode 时间占比**（prefill 期间 decode 暂停，流体平均的 token 间隔），并标出是否满足 TPOT SLO。
- **切分搜索**：同总卡数下枚举所有 N_p（c_p 的倍数）与 N_d = N − N_p（c_d 的倍数），给出 goodput / 卡最优的切分（`best_split`）与全部切分（`splits`）。池的布局本身不搜索（都是输入）。
- **量级**（Qwen3-8B）：100T + LPDDR5X，batch 16、prompt 4096：KV 576 MiB / 请求，节点内 400 GB/s 传输 1.5 ms（跨节点 50 GB/s 12.1 ms）；PD prefill 2 卡 + decode 2 卡：TTFT 1637 ms、TPOT 129.5 ms、61.8 tok/s/卡（decode 瓶颈），最佳切分 1 + 3 → 92.7；合并 4 卡 112.5 tok/s/卡，但有效 TPOT 142.3 ms（decode 占比 91%）。1P + HBM3E，batch 32、prompt 8192、TPOT SLO 50 ms：PD 2 + 6 TPOT 42.2 ms（满足）、568.9 tok/s/卡；合并 573.9 tok/s/卡但有效 TPOT 55.8 ms（超 SLO）。即：流体模型下 PD 的收益主要是 TPOT 不被 prefill 打断，而不是总吞吐——这些例子里 goodput / 卡不高于合并（按卡整数切分时非瓶颈池有闲置；合并服务的分时没有这个损失）。
- **入口**：API scenario `pd: {enabled, prefill_layout, prefill_cards, decode_cards, kv_GBps, kv_layerwise}`，eval 响应 `pd`（prefill / decode / kv / ttft_ms / tpot_ms / req_s / goodput_per_card / bottleneck / util / splits / best_split / coloc / warnings；只对 LLM / VLM）；CLI `--pd --pd-prefill-pp/tp/dp/ep/etp --pd-prefill-cards --pd-decode-cards --pd-kv-GBps --pd-layerwise`；Web 服务组「PD 分离」与单点页「PD 分离 vs 合并」表。
- **不做（0.50 第一版；0.51 补了排队、连续批处理、分块 prefill、KV 争用与能耗，见 §18.1）**：前缀缓存、异构池（不同芯片 / 存储器）、池布局搜索。

### 18.1 排队、尾延迟、连续批处理与分块 prefill（0.51，随 PD 报告一起给出）

上面的流体模型只回答「容量多大」。0.51 在 `pd.queue` 里加一层**解析排队近似「假设」**（不是仿真器）：在同一个泊松到达率 λ 下比较三种服务方式——PD 分离、合并 · prefill 优先（vLLM 不开分块时的默认调度）、合并 · 分块 prefill（Sarathi / vLLM chunked prefill）。λ = `pd.load` × PD 流体容量（默认 0.8），或直接给 `pd.rate_rps`。共同假设：泊松到达、每请求 prompt / 输出长度固定（= `serving.prompt` / `serving.out_len`）、副本间均分到达、分位数**逐项相加**（偏保守）。每个模式的单步时间都来自现有评估器（同一个 `evaluate`，只是换 batch / phase），没有新的硬件参数。

- **M/D/1**（`core/queueing.py`）：平均等待 W̄ = ρτ / (2(1 − ρ))（P-K，精确）；P(W > 0) = ρ；尾 P(W > t) ≈ C·e^(−θt)，θτ = x 为 ρ(e^x − 1) = x 的正根，C = (1 − ρ) / (ρe^x − 1)（Cramér–Lundberg 渐近）；分位数 w_q = 0（ρ ≤ 1 − q）否则 ln(C / (1 − q)) / θ。测试用 Lindley 递推仿真核对 ρ = 0.5 / 0.8 / 0.9 的均值、p90、p99（误差 < 8%，实测约 2–5%）。decode 槽位等待用 Erlang C × ½（Allen–Cunneen，确定服务）；运行 batch 的波动按 Poisson（M/G/∞）取分位。
- **PD 分离**：prefill 池每副本是一个带静态 batch 上限 b ∈ {1, 2, …, 64} 的 M/D/1（DistServe 式）：服务时间 τ_b = TTFT(b) / b，延迟下限 TTFT(b)，取使平均 TTFT 最小的 b。TTFT 分位 = 排队等待分位 + TTFT(b) + KV 队列等待分位 + 暴露的 KV 传输。decode 池是**连续批处理**：运行 batch n̄ 由 Little 定律的不动点 n̄ = λ_d · out / e · step(⌈n̄⌉) 决定（e = 每步 token 数，含投机解码），TPOT 均值 = step(⌈n̄⌉) / e；p90 / p99 TPOT 用运行 batch 的 Poisson 分位处的步时。部分负载下运行 batch 小于配置 batch，所以排队模型里的 TPOT 低于流体模型（流体按满 batch）。
- **KV 与池内集合通信争用**：没设 `pd.kv_GBps` 时 KV 与两池的 TP / EP 集合通信共用同一层（节点内 link 或跨节点 net）。KV 能用的带宽 = β · (1 − u_coll)，u_coll = 两池最忙级在该层的字节 / β / tick；反过来 KV 占该层 u_kv = λ · KV / (N · β)，各池在该层的时间乘 1 / (1 − u_kv) 后重算级时间（只在该层是瓶颈时才变慢）。KV 传输本身也是 M/D/1（每 prefill 副本一个出口）。给了 `pd.kv_GBps` 视为专用 KV 通道，不争用。
- **合并 · prefill 优先**：prefill 同样是带 b 上限的 M/D/1，占用 ρ_p；decode 只在剩余 1 − ρ_p 的时间里跑，不动点里步时除以 (1 − ρ_p)。新 prefill 到来时整批 decode 停顿 TTFT(b)：最长 token 间隔 = TPOT + TTFT(b)；一个请求生命周期内碰到的停顿数 ~ Poisson((λ/b) · 生命周期)，请求平均 TPOT 的分位 = TPOT(运行 batch 分位) + 停顿数分位 · TTFT(b) / out。TTFT 再加半个（p90 / p99 用一个）decode 步的残余。
- **合并 · 分块 prefill**：每次迭代带 C = `pd.chunk_tokens`（默认 512）个 prompt token。融合迭代在**级层面**合并：max(算力_d + f·算力_p, DRAM_d + f·(prefill 非权重字节) + 前缀 KV 重读, SLC_d, 链路_d + f·链路_p) + 同步，f = C / S（prefill 的权重读由 decode 迭代顺带完成；第 i 块要重读前面 i − 1 块的 KV，平均 KV · (S − C) / (2S) 每块）。prefill 是服务时间 ⌈S/C⌉ · T₁ 的 M/D/1，占用 ρ = λ · S · T₁ / C；decode 平均步时 = ρ · T₁ + (1 − ρ) · T₀，不动点同上。最长 token 间隔 = T₁ / e（停顿被块大小封顶），请求平均 TPOT 分位按「带块迭代数 ~ Poisson(ρ · 迭代数)」。
- **SLO goodput**（DistServe 口径）：满足 p90 TTFT ≤ `ttft_slo_ms` 且 p90 TPOT ≤ `tpot_slo_ms` 的最大 λ（倍增 + 二分；没满足的设置则是稳定上限），× out / 卡数。PD 还在同总卡数的所有切分上搜 SLO goodput 最优的切分（`pd_slo_best_split`、`pd_slo_splits`）。
- **能耗**：用现有的动作计数（§13），每输出 token = (prefill 每 token 计数 × S + decode 每 token 计数 × out) / out，取各模式在该负载下的实际运行点（PD：prefill batch 上限、decode 运行 batch；合并：同上；分块：不重复读权重、加前缀 KV 重读），PD 再加 KV 传输字节（link 或 net）；卡·秒 / token = 卡数 / λ / out（含闲置）。给了能耗表才出 J / token。
- **量级**（Qwen3-8B，1P + HBM3E，TP2，batch 64，prompt 4096，out 512，PD prefill 2 + decode 6，load 0.8 → 7.21 req/s，SLO TTFT 400 / TPOT 10 ms）：PD TTFT p50 / p90 / p99 = 102 / 216 / 380 ms，TPOT 均值 / p90 = 6.5 / 9.0 ms，最长间隔 11 ms，SLO goodput 474 tok/s/卡（切分 2 + 6 最优；4 + 4 为 316）；合并 prefill 优先 79 / 114 / 169 ms、TPOT 5.4 / 7.7 ms 但最长间隔 83 ms，506 tok/s/卡；合并分块（512）134 / 220 / 349 ms、TPOT 9.4 / 12.2 ms、最长间隔 19 ms，433 tok/s/卡。每输出 token DRAM：PD 2.46 GB、prefill 优先 3.56 GB（运行 batch 更小）、分块 2.23 GB。即这个例子里 p90 口径的 SLO goodput 合并 prefill 优先最高，但它的单次停顿（83 ms）是 PD（11 ms）的 7 倍多；PD 的价值体现在 token 间隔的最坏情况，分块 prefill 介于两者之间。prompt 更长 / TPOT SLO 更紧时结论会变（如 prompt 8192 out 256，PD 4 + 4 时 prefill 池先饱和，分块 512 在同到达率下不稳定）。
- **入口**：scenario `pd.load`（0–1，默认 0.8）、`pd.rate_rps`（覆盖 load）、`pd.chunk_tokens`（≥ 16，默认 512）；eval 响应 `pd.queue`（`lambda_rps`、`modes.{pd, coloc_prefill_first, coloc_chunked}` 各含 `ttft_ms.{mean,p50,p90,p99}`、`tpot_mean_ms / tpot_p90_ms / tpot_p99_ms`、`itl_max_ms`、`slo_rate_rps`、`slo_goodput_per_card` 与各自明细，`pd_slo_best_split`、`energy`）；CLI `--pd-load --pd-rate --pd-chunk`（另加 `--out-len --ttft-slo`）；Web PD 输入组三项与单点页「排队与尾延迟」表。
- **不做**：请求长度分布（全部定长）、抢占 / 换出 / KV 容量引起的排队、前缀缓存、调度器开销、多级 SLO 调度、分块 prefill 与 PD 的组合（PD 的 prefill 池内部分块）、异构池、池布局搜索；分位数相加而不是卷积（偏保守）；M/D/1 尾是渐近式，ρ 很小时略偏保守。

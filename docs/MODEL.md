# 建模说明（Modelling method）

本文说明 accel-dse（core v2，0.41 起；本文对应 0.63.x）怎么算、假设了什么、用什么校验，方便逐条挑错。各节标题里的版本号是该机制引入的版本；标明「数值变化（a → b）」的段落记的是当时的数值，之后的变化见后续各节与 CHANGELOG。
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
- 三轴标签：来源（official 官方 / mirror 镜像，如 meta-llama 的 unsloth 公开镜像）× 覆盖 × dtype。覆盖度指建模覆盖程度，与模型好坏无关，每个模型都标出：「完整」= 全部算子按发布结构逐项建模；「部分」= 主干逐项建模、个别机制近似或不在神经网络推理范围内（现仅结构预测的 MSA / 模板检索与松弛，见 §2.1）；「架构代理」= 含未建模的结构（超连接多流残差的混合计算、n-gram / engram 查表、压缩稀疏注意力），结果只作量级参考。每个非「完整」模型都附逐项的「近似之处」（`coverage_reasons`，由结构自动生成，UI 悬停与模型目录可见）。
- VLM（发布中带视觉编码器的模型）：0.62 起视觉编码器（ViT + 合并 / 投影，约 0.45–0.56B 参数，bf16 按发布）按发布 config 与图像预处理器逐项建模：每请求图像数与分辨率是场景输入（默认 0 张 = 只算文本，分辨率默认 1024×1024「假设」），编码器在 prefill 时运行，图像 token 加进 prompt 与 KV（§20）。
- 目录：按厂商 → 系列排列，同一厂商的 LLM 与 VLM（甚至同一系列里的文本版与多模态版）放在一起，领域只作为行内标记（VLM 行标「VLM · 含视觉编码器」）；系列内按尺寸从大到小，官方量化版（FP8 / AWQ）紧跟原版，名称用官方仓库名。结构与 dtype 完全相同的发布合并为一条（如 DeepSeek-V3 / V3.1 / R1、Kimi-K2.5 / K2.7-Code、GLM-4.5 / 4.6、MiniMax-Text-01 / M1-80k），尺寸已被覆盖的通用稠密 GQA 模型不单列；它们仍可按 id 评估，也都参与参数核对。
- 已接入 v2 的视频 / 蛋白质发布（0.41）：Wan2.1-T2V-14B / 1.3B、CogVideoX-5b / 2b（DiT 去噪主干）与 ESM-2 3B / 650M（编码器），同样从 config + safetensors 头建模，参数与发布逐项一致（偏差 0.00%），建模见 §11。
- 暂未接入 v2 的目录条目（0.62 校正：此前这里仍列着已接入的视频 / 结构模型）：AlphaFold3 等没有可核对公开发布文件的条目，列在各自厂商 / 机构之下、该厂商可评估条目之后，标「暂未接入 v2」，不能评估、不能按 id 解析。视频（Wan2.2-A14B、MiniMax-H3、HunyuanVideo、LTX-Video、Mochi 1、Open-Sora STDiT3）与结构预测（ESMFold、AlphaFold2、OpenFold、Boltz-1、Protenix）已按发布接入（§11、§12）。分子动力学 / 机器学习力场（MLFF）从未有过目录条目，不列出。
- 参数核对：与发布 safetensors 总量偏差 > 2% 时附注；当前 72 个发布（55 个 LLM / VLM + 10 个视频 + 7 个蛋白质）全部在 ±0.5% 内，其中视频 / 蛋白质逐项一致（偏差 < 0.001%）。（0.62.1 更正：原写 61 = 55 + 6，是 0.41 时的计数。）

### 2.1 覆盖度清单（Unreleased：按参考实现收口）

收口前（0.65.1）目录里 72 个可评估发布中 22 个不是「完整」（现剩 4 个：结构预测「部分」，LLM 全部「完整」），另有 1 个目录条目「暂未接入 v2」。逐项判断：有公开 config + safetensors 头 + 参考代码的，按参考实现逐项建模并核对；没有的保留标签与精确原因。

| 模型 | 0.65.1 | 现在 | 依据 / 剩余原因 |
|---|---|---|---|
| gpt-oss-120b / -20b | 部分（attention sink 只计参数） | 完整 | openai/gpt-oss `gpt_oss/torch/model.py` `sdpa`：每个 query 行把每头 sink logit 拼进分数、softmax 后丢弃 → softmax 每行多 1 个元素（5 次向量运算 / 元素，与其他 softmax 同约定） |
| deepseek-v3.2 | 部分（indexer 按 GEMM + O(ctx) 近似） | 完整 | DeepSeek-V3.2 `inference/model.py` `Indexer`：wq_b / wk / weights_proj、k LayerNorm、RoPE（qk_rope_head_dim）、q / k Hadamard 旋转（d·log₂d）、block fp8 量化；`fp8_index` 打分 GEMM 在 fp8 执行；每 (query, key) 对 ReLU + 按头加权求和（3·heads）+ key scale 1 + top-k 选择 1 次比较；索引 key 缓存 fp8 + 每 128 元素 1 个 fp32 scale = 132 B / token（原按 KV dtype 256 B） |
| glm-5 / glm-5.2 / glm-5.3 | 部分（同上） | 完整 | transformers `GlmMoeDsaIndexer`（bf16 key 缓存，打分同上，无 Hadamard / fp8）；`indexer_types` 中 57 / 78 层为 `shared`（复用前一 full 层的 top-k，无索引器权重、缓存与打分）→ GLM-5.2 / 5.3 参数与发布从 +0.07 % 变为逐项一致；因此 GLM-5 与 GLM-5.2 结构不再相同，取消合并，GLM-5 单列 |
| qwen3-next-80b-a3b、qwen3.5-397b-a17b、qwen3.8-2.4t、qwen3.8-27b（Gated DeltaNet） | 部分（递归按向量计、状态 fp32 / 分块 64「假设」） | 完整 | transformers `torch_chunk_gated_delta_rule`（chunk 64，fp32 状态）：每块每头 kβ·Kᵀ、Q·Kᵀ [C,dk,C]，UT 三角求解 ½·C²(dv+dk)，k_cumdecay·S、Q·S、Kᵀ·v_new [C·dk·dv]，A·v_new [C,C,dv] 上阵列（bf16 操作数，FLA 内核）；衰减掩码 / β / L2 norm 按向量；decode 按 fused_recurrent：每 token 每头 7·dk·dv |
| kimi-k3（KDA） | 部分（同上） | 完整 | transformers `chunk_kimi_delta_attention`（chunk 64，fp32 状态），GEMM 同 GDN；逐通道门控另计 3·C·dk / 块 |
| minimax-m1-80k / minimax-text-01（Lightning） | 部分（同上） | 完整 | 发布仓库 `modeling_minimax_text_01.py`：BLOCK = 256，kv 状态 fp32；每块 Q·Kᵀ、A·V、Q·S、Kᵀ·V；decode 每 token 每头 5·dk·dv |
| glm-5.3-flash | 架构代理 | 完整 | transformers `glm5_next`：KDA 低秩输出门 g_a / g_b（原按满秩 g 计，参数多 1.05B）；mHC ×4（每层 attn_hc / ffn_hc：N·h 的 RMSNorm、fn GEMM [N·h → (2+N)·N]、sigmoid / softmax + Sinkhorn 20 次、collapse、post ⊗ out + combᵀ·residual；残差 N 流 TP 复制，流水级间传 N·h），末端 HyperHead 均值；DSA 索引器 key 池化（门控 GEMM h → 128 + APE，每次前向对全部缓存 key 重建池：1 + 5 + 2 次 / 元素；按 ⌈ctx/4⌉ 个池打分，top-512 池展开 + 尾池 ≤ 3；缓存 [k, gate, valid] bf16 514 B / token）。逐层参数与 safetensors 头一致；与发布汇总差 270 = 45 × 6 个 hc_*_scale（汇总脚本归为量化 scale） |
| deepseek-v4-flash / -pro | 架构代理 | 完整 | 官方 `inference/model.py`（两个仓库相同）：MQA head_dim 512（K = V 共用一份 512 维条目，attention sink，q 逐头 RMSNorm，o 逆 RoPE），滑窗 128 + 压缩器（比 4 重叠 coff 2 / 比 128；wkv / wgate、softmax 池化 1 + 5 + 2 次 / 元素，每条压缩条目 RMSNorm + RoPE + fp8 量化），比 4 层索引器（自带 Hadamard 压缩器，64 头按 TP 列切分 + fp32 分数 all-reduce，top-512 / Pro 1024），比 128 层读全部压缩条目；哈希路由层 gate 分数照算（与 top-k 同代价）；mHC ×4 + 学习的末端 hc_head（fn GEMM [N·h → N]）；MTP 为比 0 层。参数与发布汇总差 259 / 367（hc_*_scale）。未计：tid2eid 查表（3 层 × 129280 × 6 int64 ≈ 18.6 MB 存储，每 token 读 6 个整数）；压缩器在参考中以 fp32 计算，这里按 bf16 操作数；压缩条目数按 ⌈p/r⌉（参考 ⌊p/r⌋，每查询最多差 1 个） |
| deepseek-v4.1-flash | 架构代理 | 完整 | 官方 V4.1 `inference/model.py` + `engram.py`：每层 MQA-512 滑窗 128 + sink；压缩层（比 2 / 比 1）另读 top-512 压缩条目，keys = min(p, 128) + min(512, ⌊p/r⌋)（逐查询精确）；压缩器（比 1 为纯投影、比 > 1 softmax 池化 7 次 / 元素、无 APE）与压缩 KV 只在 kv 源层 2 / 8 / 14 / 20 计算和存储，其余层只读；索引器只在 2 / 8 / 14 / 20 / 24 / 28 / 32 / 36 运行（index key = wk(latent) 只在 kv 源层算 / 存，头按 TP 切分 + 分数 all-reduce），其余层复用 top-k；第 20 层 select_candidate_blocks（每位置 amax 1 + 块 top-2048），24–36 层按候选块掩码；Engram（第 1 / 14 层）：每 token 哈希（4 乘 3 异或 24 取模 24 加），按行分片的表读 24 行 × (256 B fp8 + 8 B e8m0 scale)、反量化、TP all-reduce、wkv GEMM [6144 → 5·h]、门控（9·N·h）；表存储 196.6B × 8.25 bit；mHC ×4，末端用最后的 pre_mix 加权 collapse（无 head 投影）；DSpark 草稿（spec_k > 0）：prefill 用 main_proj [3h → h] 与各草稿层 wkv 播种窗口，decode 每步一次 5 位置块经 3 层（窗口 + 块内非因果 5 键，块 KV 不写缓存）、共享 head、Markov 头逐位置（rank-256 embed + head）、置信头；spec_k > 5 按 5 计并警告；视觉沿用 0.62 的 DeepSeek ViT + aligner（485.3M，与发布一致）。逐层参数与 safetensors 头一致（fp4 专家按逻辑数），与汇总差 240 = 40 × 6 个 hc_*_scale；MTP 一致。缓存按参考的默认 dtype（bf16）缓冲计：窗口 / 压缩条目 1024 B、index key 256 B（参考的 fp8 / fp4 act_quant 是原位模拟量化；按 fp8 / fp4 存储将是 528 / 288 / 68 B） |
| qwen3.8-flash-next | 架构代理 | 完整 | transformers `qwen4_exp`（主干）+ SGLang `qwen4_exp_mtp.py`（MTP）：GDN（同 Qwen3.5）/ 门控 q 的 GQA（q / k 逐头 RMSNorm、部分 RoPE 64、σ 输出门）；QSA 索引器：index_qk_proj [h → (4+1)·128]，每次前向把缓存的原始 key 按 4 个一块均值池化 + LayerNorm + RoPE（参考循环里每个查询重算一遍，值相同，按每次前向一次计），按 ⌊p/4⌋ 块打分（ReLU + 头求和 + top-512 块），注意力 keys = p（⌊p/4⌋ ≤ 512）否则 2048 + p mod 4（尾块）；原始 key 缓存 256 B / token；门控残差 ×2 / 层（分组 RMSNorm N·h、低秩 down / up GEMM [N·h ↔ 320]、inject GEMM [N·h → 4]、σ 混合与注入；无逐层 RMSNorm），末端 mixer（无 inject、无最终 norm）；PLE（第 1 层）：n-gram 哈希、16 行 × 160 bf16 查表（表 51.2B）、key / value 投影 GEMM、3 个分组 RMSNorm、门控、膨胀 3 的 depthwise conv 4（状态 9 × N·h bf16 / 序列）；MoE 512 选 10 + σ 门控共享专家；MTP 输入融合 fc_embedding + 对 4 条流各做 fc_hidden（5·h² MAC / token）。参数与发布汇总逐项一致（差 0）；视觉 448.9M 一致。门控残差 / 索引器权重在 TP 内复制（残差流复制，与 mHC 同约定） |
| openfold / alphafold2 / boltz-1 / protenix | 部分 | 部分 | 神经网络推理已逐项建模；MSA / 模板检索（jackhmmer / HHblits / MMseqs2，CPU / 检索服务）、特征化与 AMBER 松弛不属于加速器推理，按范围保留 |
| alphafold3 | 暂未接入 v2 | 暂未接入 v2 | 权重需向 Google DeepMind 申请、禁止再分发，无公开 safetensors 头可核对；同架构用 Protenix / Boltz-1 |

数值影响（HBM3e、TP8 / EP8，单卡 100T 默认芯片）：GLM-5.3-Flash / DeepSeek-V4-Flash / V4-Pro 收口后 decode（batch 16、32k）TPOT +5.8 / +8.0 / +7.4 %，prefill 8k TTFT +31 / +29 / +23 %，64k +2.8 / −1.8 / +8.6 %（mHC 混合与压缩器、sink 等原先未计的向量 / 小 GEMM 开销；V4 64k 略降来自索引器 TP 切分）。DSA 模型 decode（batch 16、ctx 32k）TPOT −9 ~ −13 %（DeepSeek-V3.2 fp8 打分；GLM-5.2 / 5.3 少 57 层索引），prefill 64k TTFT −36 ~ −49 %（GLM-5 不变，打分向量开销被其他瓶颈掩盖）；线性注意力模型 prefill 8k / 64k TTFT +4 ~ +17 % / +3 ~ +9 %（小块 GEMM 在大阵列上利用率低，原向量计数偏乐观），decode 不变（DRAM 主导）；gpt-oss 不变（sink 开销被掩盖）。DeepSeek-V4.1-Flash / Qwen3.8-Flash-Next 收口后 decode（batch 16、32k）TPOT −23 / −31 %（共享压缩 KV 与索引、按真实条目读；QSA 2048 键替代原稠密全注意力），prefill 8k TTFT +40 / +127 %、64k −23 / +34 %（TP 复制的 mHC / 门控残差 GEMM 与向量、Engram / PLE；Qwen 在 TP8 下门控残差 4 个 [N·h ↔ 320] GEMM 每 rank 全量计算，占每 rank prefill FLOP 的 42 %）。已是「完整」的 50 个模型默认结果不变（`tests/test_core_coverage.py` 钉住 0.65.1 的 KPI 哈希），V4 / V4-Pro / GLM-5.3-Flash 本轮不变。

## 3. 逐 rank 算子图与并行

- 布局 `PP·TP·DP·EP·ETP`：dense 模型 DP = EP = ETP = 1（数据并行副本是独立服务实例）；MoE 要求 `EP·ETP = TP·DP`——注意力按 TP 切分、在 DP 组间复制，专家按 EP 分组、组内按 ETP 切分。
- 每个 rank 生成算子：投影 GEMM、注意力核心（GQA / MLA 吸收 / 线性状态更新）、MoE 路由与专家 GEMM、embedding / 词表并行 lm_head、MTP。
- MoE 命中专家数：`hit = max(ceil(局部期望命中), ceil(全局期望命中 / EP))`，限定在 [1, 本地专家数]；每专家 token 数 `m_e = ceil(本地 token·top_k / hit)`。默认 token 均匀路由（「假设」）；0.49 起可选 EP 负载倾斜（§16）。
- stage 划分：连续的整层切分。0.62 起默认按代价平衡（`pp_split = "cost"`，§21）：使最慢 stage 最快，计入嵌入 / 输出头 / MTP / 级间传递与异构层栈；`pp_split = "layers"` 保留 0.61 的按层数均分（如 61 层 / 8 → 8,8,8,8,8,7,7,7）。容量检查取最重的 stage。
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

- staging = `max(2 MiB, 2 × 最大激活)`；其余 SRAM 依次驻留热权重（每步都读）、专家，剩余容量放 KV / 线性注意力状态；冷存储（按行查表的 embedding 表、Wan2.2 待机专家、不用的 io 表）不驻留（0.61.3 前与专家一起驻留，白占 SRAM / SLC）。
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

- **V0 变形关系**（PP = 1 或 `pp_split = layers`；cost 切分见 §21 单调性一条）：受控替换同值不变；更多 DRAM 带宽 / SRAM / 更宽端口 / 更快链路不会更慢；更长上下文不会更快；`reconf` 不慢于任一单一组织；去掉原生 fp8 不会让 fp8 发布更快。另有 TP / EP 分片守恒测试（FLOPs、权重、KV、专家存储、stage 存储）。视频 / 蛋白质：同样的 SRAM / 带宽 / `reconf` 关系、batch 增大延迟不减、延迟 ∝ 去噪步数、TP / SP 分片 FLOPs 守恒（SP 下文本跨注意力 K/V 每 rank 重算，偏差 < 0.2%）、SP 复制权重 / TP 切分权重，以及精确 batch 搜索与暴力枚举一致。
- **V1 参数**：55 个 LLM / VLM 发布与 safetensors 总量偏差全部 ≤ 0.5%，17 个视频 / 蛋白质发布逐项一致（< 0.001%）；8 个模型卡的激活参数 ≤ 5%；FLOPs 与独立计数对照（视频 / 蛋白质：按 config 的闭式计数，偏差 < 1%）。
- **V2 趋势区间**（H100 类配置：128 × 128 × 16 @ 1.83 GHz ≈ 959 TFLOPS bf16、SRAM 50 MiB、HBM3 5 堆 3.33 TB/s、DRAM 效率 0.8、可重构映射，均为「假设」）：Llama-3.1-8B bf16 B1 TPOT 5.7 ms（区间 4.5–9）；B16/B1 = 1.13；Qwen3-8B FP8/BF16 = 0.54；4K prompt TTFT 83 ms；Qwen3-32B TP2 加速 1.94；B256 吞吐 1.4 万 tok/s。视频只做工作负载 / FLOP 的合理性核对（不是硬件标定）：Wan2.1 README 称 T2V-1.3B 在单张 RTX 4090 上约 4 分钟生成 5 s 480P（含 T5、VAE 与 offload）；本工具每段 28.6 PFLOP（0.44 起含 umT5 与 Wan-VAE 解码 0.29 PFLOP），折合 4090 约 165 TFLOPS（bf16、fp32 累加）的 0.72（区间 0.4–1.0）。
- **V3 GenZ 对照**（Llama-3.1-8B decode，参考值由 `scripts/genz_reference.py` 生成）：LPDDR 191 GB/s 各点比值 1.07（DRAM 效率口径差异）；HBM 6.6 TB/s 下 `reconf` 映射比值 1.07–1.22，长上下文 / 大 batch 点偏高来自注意力小 M 分块，而 GenZ 按理想 FLOPS 计；`os` 映射在 HBM 下比值 3.6–6.8，因为小 M decode 被 SRAM 供数端口限制——这正是映射作为设计变量要暴露的差别，不视为误差。容差：访存受限点 15%，其余 60%。
- **V4 服务验证**（0.54）：闭式排队模型对照请求级 DES（同一套逐步代价），54 点网格与误差表见 §18.4。
- **V5 硅片实测**：无（没有可测的 NPU，结果只作设计指导）。

## 10. 范围与近似

- LLM / VLM 推理（0.62 起 VLM 含视觉编码器，§20）；视频生成覆盖 DiT 去噪主干与文本编码器 / VAE pipeline（§11），蛋白质覆盖 ESM-2 编码器（§11）与结构预测的神经网络推理（ESMFold、AlphaFold 2、OpenFold、Boltz-1、Protenix，§12）。AlphaFold 3 权重需申请、无可核对的公开发布文件，标「暂未接入 v2」（§2）。不覆盖分子动力学 / 力场（目录中也无此类条目）。
- 「架构代理」：目录中现无此类 LLM（Unreleased 全部按参考实现收口，§2.1）；标签与 `coverage_reasons` 机制保留，新加入的含未建模结构的发布仍会自动标出。
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

**激活流式（只用于全序列前向）**：视频的激活远超 SRAM（Wan2.1-14B 720P 一层的 FFN 输入约 1.5 GB）。每个算子的激活按 SRAM/2 的预算分块流进 / 流出 DRAM：GEMM 取「激活分块、权重重复读」与「权重分块、激活重复读」两者中流量较小者；注意力按 flash 式计算，N×N 分数不落 DRAM，Q / O 读写一次，K / V 每个 Q 块重读一次（块行数 `Br = 预算 / (d·(ab + 4))`）。DRAM 容量另计在途序列的残差流与单个算子的最大激活。LLM 的 prefill / decode 自 0.63 起同样使用此模型（§22.10；decode 的激活通常装得下 SRAM/2，流量≈0）。0.63 起 GEMM 的分块按实例（专家 / 头 / 组各自的权重与激活切片）决定；0.64 起改为二维分块（权重驻留 / 激活驻留 / 输出驻留三种循环嵌套取流量最小者，§23.2）。

**映射**：OS / WS（K-split）/ GEMV / 可重构照常是设计变量。全序列注意力的两个 GEMM 两侧都是激活，映射可以取任一方向（`O = P·V` 或 `Oᵀ = Vᵀ·Pᵀ`）：`P·V` 的 N = head_dim = 128，会让 896 列宽的阵列大部分空闲，换向后有效 MAC 从约 30% 升到 85–98%。LLM 注意力仍按 0.40 的固定方向（不改变 LLM 结果）。

**调度与指标**

- 一次前向的序列数 `S = batch × CFG`（视频）或 `batch`（蛋白质），微批 `mb = min(PP, S)`（可手动设置）。
- 视频单段延迟 `T_clip = T_text + 步数 × ticks × t_stage + T_decode`，每个去噪步 `ticks = max(mb, PP)`（各微批互相独立）；0.63 起若同一请求的 cond / uncond 分在 k > 1 个微批（CFG = 2、mb > batch 时），这一步要等两路都过完全部流水级才能更新潜变量，`ticks = max(mb, k + PP − 1)`（§22.2；文本编码与 VAE 解码见 §11.4，`workload.pipeline = false` 时两项为 0）；每帧延迟 `T_clip / 帧数`；每去噪步时间 `去噪时间 / 步数`；吞吐 `batch × 帧数 / T_clip`（帧/s，每卡再除以卡数）；另给段/小时/卡、实时倍率 `视频秒数 × batch / T_clip`、每段 TFLOP。
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
| `ltx-video` | T5 v1.1 XXL 19.05 GB fp32 | 2.4 TFLOP / 0.03 s | LTX VAE 1.68 GB fp32 | 50.5 TFLOP / 2.7 s（0.61.2 更正，原 15.4 / 0.7 s） | 49.5 s → 52.2 s（+5.4%） |
| `mochi-1` | T5 v1.1 XXL 19.05 GB fp32 | 4.8 TFLOP / 0.06 s | AsymmVAE 1.84 GB fp32 | 1053 TFLOP / 42.5 s | 2433 s → 2475 s（+1.7%） |
| `opensora-stdit3` | DeepFloyd T5 v1.1 XXL 19.05 GB fp32 | 2.8 TFLOP / 0.03 s | Open-Sora VAE v1.2 1.57 GB fp32 | 1140 TFLOP / 39.6 s（0.61.3 更正，原 1158 / 39.9 s） | 244 s → 284 s（+16.2%） |
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

**VAE 分块解码（0.45，`workload.vae_tiling`，默认关）**：按 diffusers `enable_tiling()` 的默认参数做空间分块，重叠区重复计算、每 tile 重读权重、激活峰值按 tile——CogVideoX：tile 240 × 360 px（= 采样尺寸 / 2），重叠因子 1/6、1/5 → 潜空间 30 × 45、stride 25 / 36（480 × 720：9 个 tile，重叠 ×1.40，解码 315 → 441 TFLOP，激活峰值 2.31 → 0.58 GiB）；Mochi：tile 256 px、stride 192 px → 潜空间 32 / 24（480 × 848：15 个 tile，×1.65，1053 → 1737 TFLOP）；Wan（0.46，diffusers `AutoencoderKLWan`：tile 256 px、stride 192 px → 32 / 24；官方 Wan 仓库不分块）：720P 28 个 tile，×1.65，639 → 1042 TFLOP，解码 41.3 → 67.8 s，激活峰值 3.81 → 0.27 GiB。LTX-Video（0.47，diffusers `AutoencoderKLLTXVideo`：tile 512 px、stride 448 px，空间压缩 32 → 潜空间 16 / 14；`enable_tiling()` 不开逐帧解码）：704 × 512 → 4 个 tile（16 + 2 行 × 16 + 8 列），×1.23，50.5 → 62.0 TFLOP，2.66 → 3.26 s，激活峰值 1.73 → 1.26 GiB（0.61.2 起上采样块的 resnet 按块输出分辨率计；0.61.1 及以前为 15.3 → 18.8 TFLOP）。0.47 起分块循环与 diffusers 逐字一致：任一轴超过 tile 即两轴都按 stride 从 0 走，短轴会多出一条窄 tile（上例的 2 行）；0.45–0.46 在短轴不切，默认分辨率下两者相同。HunyuanVideo 与 MiniMax-H3 按发布总是分块（见上）。Open-Sora VAE v1.2 的发布实现没有空间分块（2D VAE 每次 4 帧、时间 VAE 17 帧一块，已按此计），diffusers 也未收录：开启时给出警告并按不分块计。

**对结论的影响**：长视频 / 高分辨率下解码占整段时间 0.6–16%（Open-Sora 720p 的逐帧 SD-VAE 最重），文本编码 < 0.5 s；存储影响大——常驻时 64 GiB LPDDR 上 Wan2.1-14B（fp32 DiT 57 GB + umT5 11.4 GB）、Wan2.2 PP2 / TP2 与 MiniMax-H3 放不下；0.45 的 auto 放置按参考实现的卸载 / FSDP 方式把它们放下，代价是每请求 1–2.5 s 的主机重载（相对 30 min 级的整段可忽略）。更正 0.44 文档：Qwen3-VL 文本塔 66.7 GB = 62.1 GiB，单独并未超过 64 GiB，只是与 DiT 同时常驻放不下。

**多卡分块并行解码（0.47，`workload.vae_parallel`，Web「VAE 多卡分块并行解码」，CLI `--vae-parallel`，默认关）**——依据 MiniMax-H3 发布的 `FL2VA/video_vae`（`vae_parallel_tiling = 1`，`klvae.tiled_decode`）：每个独立的分块调用（一「轮」：H3 的一个时间段、HunyuanVideo 的一个时间 tile、diffusers `enable_tiling()` 的整次空间分块）内，rank r 解第 r、r + N、… 个 tile，再把解码后的像素 tile all-gather 到每张卡做融合。tile 互相独立，算术与单卡相同（请求 FLOPs 不变）；时间 = 最慢卡的 tile（按轮询分配，整除不了时有尾部不均）+ 每轮一次 all-gather（每卡载荷 = 最大份额的解码像素 × 解码 dtype，`allgather` 代价 + α）；N = 本副本的 PP·TP·SP 张卡（解码在去噪之后，全部卡空闲）；VAE 权重每卡一份（放置时 VAE 计入每个流水级）。需要分块解码（H3 / HunyuanVideo 总是分块，其余要同时开 `vae_tiling`），否则忽略并警告；音频 VAE 不拆。H3 以外是设计选项：xDiT 的 DistVAE（patch 并行 + halo 交换）只用于图像 VAE，其 CogVideoX 示例明确不支持并行 VAE。例（100T + HBM3E）：H3 SP 2 / 4 / 8 解码 24.4 → 12.2 / 6.11 / 3.49 s（SP8 每段 28 tile 分到 8 卡、最慢卡 4 个，all-gather 共 4.5 ms）；SP8 时 HunyuanVideo 221 → 34.0 s，Wan 分块 67.8 → 10.4 s，Mochi 分块 70.0 → 13.7 s，CogVideoX 分块 23.2 → 4.7 s（9 个 tile、最慢卡 2 个）。

**跨请求重叠（0.47，`workload.overlap`，Web「跨请求重叠」，CLI `--overlap`，默认关）**：只在文本编码器放主机 CPU（`te_cpu`）时生效——主机编码下一请求与卡上去噪 / 解码当前请求并行，两者是独立资源；队列饱和时稳态周期 = `max(T_clip − T_text, T_text)`，单段延迟不变，吞吐（requests/s、帧/s/卡、段/小时/卡）按周期计。组件与 DiT 同卡时没有独立资源可重叠（卡上的编码 / 解码与去噪抢同一阵列与 DRAM），按串行计并警告；卸载的主机重载仍串行（参考实现是同步拷贝）。分量很小：H3 + 主机编码 2 TFLOPS，单卡 1953 s 一段，周期 1937 s（−0.8%）；Wan2.1-14B SP8 −0.5%。

**仍未建模**：调度器逐元素更新与 CFG 组合；提示词改写（如 H3-Context-IR）；不分块的 VAE 的 patch 并行解码（xDiT DistVAE 式 halo 交换，视频 VAE 无参考实现）；组件同卡时的跨请求重叠（需要抢占式调度，本工具不做）；主机 CPU 编码器只有一个算力旋钮（内存带宽、NUMA、线程数未建模）。

## 12. 蛋白质结构预测（0.43）

以下发布从官方检查点的张量头建模（PyTorch zip 检查点只读中央目录与 `data.pkl`，AlphaFold 2 的 JAX `.npz` 在发布 `.tar` 内按 tar 头 → zip 中央目录 → `.npy` 头读取；均为 HTTP range 请求，不下载权重；`scripts/fetch_torch_ckpt.py`、`scripts/fetch_npz_header.py`、`scripts/summarize_ckpt.py`）。参数与发布逐项一致（按构造：每个 ≥ 2 维权重是一个 GEMM，其余计为杂项参数）。

| id | 发布（检查点） | 参数 | 发布 dtype | 默认工作负载 | 覆盖 |
|----|------|------|-----------|------|------|
| `esmfold` | facebook/esmfold_v1（`pytorch_model.bin`） | 3.528B（ESM-2 3B 2.84B + 折叠部分 0.69B） | ESM-2 fp16、折叠部分 fp32 | 512 残基，单序列，主干 4 遍（num_recycles=None → 循环 max_recycles = 4 次，含首遍；0.61.2 更正，原 5 遍） | 完整 |
| `alphafold2` | google-deepmind/alphafold（`alphafold_params_2022-12-06.tar` → `params_model_1_ptm.npz`，CC BY 4.0） | 93.24M | fp32 | 512 残基；MSA 聚类 508 行（512 − 4 模板）、extra MSA 5120 行、模板 4；主干 4 遍 | 部分 |
| `openfold` | aqlaboratory/openfold（`finetuning_ptm_2.pt`） | 93.24M | fp32 | 512 残基；MSA 512 行、extra MSA 1024 行、模板 4；主干 4 遍 | 部分 |
| `boltz-1` | boltz-community/boltz-1（`boltz1_conf.ckpt`） | 592.01M（0.61.3 去掉别名张量与回调标量；原 606.38M） | fp32 | 512 残基、每残基 8 原子「假设」；MSA 4096 行；主干 4 遍；扩散 200 步 × 1 样本 | 部分 |
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
| `esmfold` | 4.58 / 2.32 / 1.20 / 0.64 s | 7.1× | 0.38 → 0.09 GiB | 7.0 GB |
| `alphafold2` | 14.17 / 7.09 / 3.57 / 1.80 s | 7.9× | 1.63 → 0.25 GiB | 16.3 GB |
| `openfold` | 12.65 / 6.33 / 3.19 / 1.62 s | 7.8× | 0.69 → 0.16 GiB | 15.1 GB |
| `boltz-1` | 7.94 / 4.55 / 2.87 / 2.03 s（0.61.3 起扩散步缓存；原 13.05 / 7.64 / 4.94 / 3.59 s） | 3.9× | 2.07 → 1.13 GiB | 15.2 GB |
| `protenix` | 27.86 / 14.55 / 7.91 / 4.02 s（0.61.3 起 pair 偏置按样本算；原 19.33 / 10.65 / 6.32 / 3.22 s；不分样本 16.84 / 11.34 / 8.59 s） | 6.9×（不分样本 3.2×） | 1.63 → 0.40 GiB | 13.4 GB |

主干主导的 AF2 / OpenFold / ESMFold 接近线性；Boltz-1 / Protenix 的 200 步扩散 transformer 在单一 / 原子轨道上，DAP 不切分它。

**扩散样本分卡（0.46，`workload.sample_split`，默认开；Web「DAP 时扩散样本分卡」，CLI `--no-sample-split` 关闭）**：DAP > 1 且样本数 S > 1 时，扩散模块（`samples = "tok"` 的块：原子编码器、扩散 transformer、原子解码器）的 S 条轨迹分到 D 张 DAP 卡，每卡 ⌈S / D⌉ 条——样本之间相互独立，条件（单一表示与 pair 条件）在 DAP 下已在每张卡上，所以这是精确的并行、不增加通信；块内 pair 网格的工作（pair 偏置投影）仍按 DAP 切分，偏置 all-gather 不变；D > S 时多出的卡空闲（`Op.replicated = D·⌈S/D⌉ / S`，请求 FLOPs 不变）。置信度头（`samples = "all"`，每样本一份 pair 副本）仍按 DAP 切 pair、不分样本。参考实现（Boltz、Protenix）没有现成的多卡样本并行，这里作为布局设计选项给出。100T + HBM3E，DAP 1 / 2 / 4 / 8：Protenix（5 样本）27.86 / 14.55 / 7.91 / 4.02 s（6.9×；不分卡为 3.2×）；Boltz-1 默认 1 个样本（`--diffusion_samples` 默认 1）不变（3.9×），取 5 个样本时 DAP 8 为 17.5 → 2.8 s（不分卡 6.9 s）（0.61.3 数值）。ESMFold 的 ESM-2 语言模型层在 SP 维上按 Ulysses 切分（与 ESM-2 相同）。批延迟 `T = (mb + PP − 1) × t_stage`，与 ESM-2 相同。「假设」：400 GB/s 链路下通信基本被计算覆盖（链路变慢时会成为瓶颈，可在链路参数里试）；没有可逐项核对的公开 DAP 推理时延，故不加校验行。

**dtype**：发布权重 fp32（ESMFold 的 ESM-2 为 fp16）；激活按 bf16「假设」（参考实现：ESMFold / OpenFold / AF2 单体 / Boltz-1 为 fp32，Protenix 默认 bf16）。0.44：fp32 激活用激活 dtype what-if 评估（Web「激活 dtype」/ CLI `--act fp32` / 场景 `formats_override: [["act", "fp32"]]`，标注 what-if）——激活存储、DRAM 流式、SRAM 端口读写（含输出）按 4 B；计算仍在 bf16 阵列上，激活逐 GEMM 转换的开销计入向量单元，所以它是「fp32 数据搬运 + bf16 计算」的代价，数值上不等价于 fp32 矩阵乘。例：ESMFold 512 残基、100T + LPDDR5X（DRAM 受限）5.0 s → 10.0 s（0.61.2 起主干 4 遍；原文 6.3 → 12.5 s 是 5 遍时的数值）；HBM3E 上 MAC 受限，延迟不变、容量 +0.4 GiB。芯片无 fp32 MAC：逐 GEMM 转换为 bf16，开销计入向量单元（与 LLM 反量化同一规则）。

**校验**：`validate` 增加 ESMFold 行——论文（Lin et al., Science 2023）：单 V100 上 384 残基 14.2 s；按本模型的 FLOPs（50.2 TFLOP，4 遍主干；0.61.2 前按 5 遍计为 62.2 TFLOP）折算为 V100 fp32 峰值（15.7 TFLOPS，主干为 fp32）的 23%，落在 [0.05, 0.8] 带内。测试核对：参数逐项一致；AF2 官方 JAX 参数映射后的逐层 GEMM 与 OpenFold 检查点完全相同；每块检测到的核（Evoformer：三角乘法 ×2、三角注意力 ×2、行 / 列注意力、外积均值；AF3 类 MSA 模块：pair 加权平均代替行注意力）；核 FLOPs 与闭式一致；recycle / 扩散步数 / 样本数的线性缩放；ESMFold 中 ESM-2 每请求只算一次。

**100T 芯片（默认 1 GHz、64 MiB SRAM）单卡 batch 1 的默认工作负载**：ESMFold 97.3 TFLOP / 4.1–5.0 s（0.61.2 起主干 4 遍；原 121 TFLOP）；AlphaFold 2 398 TFLOP / 11.7–14.2 s；OpenFold 355 TFLOP / 11.0–12.7 s；Boltz-1 262 TFLOP / 7.0–9.1 s；Protenix 434 TFLOP / 12.7–27.9 s（0.61.3：模板行并入 MSA、Boltz 扩散步缓存与别名去重、Protenix 按样本的 pair 偏置；原 396 / 354 / 296 / 419 TFLOP）（范围 = os / 可重构映射 × LPDDR5X 273 GB/s / HBM3E）。有效 MAC 21–36%：pair 网格上的 GEMM 是 K = N = 128 的窄矩阵，三角注意力 head_dim 只有 32。

**未建模 / 近似**：MSA / 模板检索与特征化、松弛；IPA 与扩散的几何 / 噪声调度向量运算；分块（chunk / subbatch）只降低峰值显存、不改计算量，峰值激活按单个算子计；AF2 / OpenFold 的模板扭转角嵌入按 T × N 行（0.61.3；原每残基一行），扭转角行并入 MSA 网格；原子数按每残基 8 个重原子「假设」；Boltz-1 / Protenix 的多链 / 配体 token 化按纯蛋白质计。

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

**时间窗与单位**：LLM decode 一个步（全部微批与副本）→ token；prefill 一次（TTFT）→ prompt token；视频一次请求批：步数 × DiT 各级 + 卡上的文本编码器 + VAE 解码，时间窗 = 单段延迟（开跨请求重叠时为稳态周期）→ 帧（主机 CPU 编码器的能耗不计）；蛋白质一次前向 → 序列。每级每遍执行「微批数」个 tick，tick 的计数是单 rank 的，乘 TP·SP·DP。PP 下每个微批都重新流式读取本级权重（与级时间一致），所以 PP2 的 DRAM / token 约为 PP1 的 2 倍——这是映射的真实代价，不是重复计数。0.63（外部评审）：MAC / 向量计数按有效工作计——不整除切分（头、列、token、batch）与专家补齐行只占最忙 rank 的时间、不是工作，最后一个微批的补齐也不计；权重反量化每遍每个忙碌 rank 执行一次（真实的重复工作，PP / DP 下随遍数增加）；EP > 1 时没有序列的 DP rank 仍运行其专家（§22.7）。

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
- **量级**（100T + LPDDR5X 4×64 8533，Qwen3-8B decode batch 8，存储 ≈ 16 GB 权重 + KV）：TPOT 104.2 ms → pin 1 / 4 / 8 / 16 GiB 98.5 / 81.7 / 59.2 / 20.0 ms（0.61.3：embedding 表不再占 SLC；原 16 GiB 20.8 ms）；lru 16 GiB 放不下 → 104.2 ms（不变），32 GiB 放得下 → 20.0 ms。

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

默认是合并服务（§7：每个副本分时做 prefill 与 decode）。`scenario.pd.enabled = true` 时另算一份 **PD 分离**报告（DistServe / Splitwise / Mooncake 式），与同卡数的合并服务对照；单点评估本身（级表、TPOT 等）不变。**稳态流体模型「假设」，第一版**；排队 / 尾延迟 / 连续批处理 / 分块 prefill 见 §18.1（0.51）；长度分布、前缀缓存、池布局搜索见 §18.2（0.52）。

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
- **PD 分离**：prefill 池每副本是一个带静态 batch 上限 b ∈ {1, 2, …, 64} 的 M/D/1（DistServe 式）：服务时间 τ_b = TTFT(b) / b，延迟下限 TTFT(b)，取使平均 TTFT 最小的 b；0.64 起若该 b 不满足 SLO 而另有 b 满足，则在满足 SLO 的 b 中取平均 TTFT 最小者（§23.1）。（b 按平均值选，所以平均 TTFT 随负载单调不降，但 p90 可能在 b 换挡时下降：例 Qwen3-Next-80B-A3B，decode TP2·EP2、prefill TP1，b16，prompt 1024，命中率 0.4，负载 0.8 → 0.9 时 b 1 → 2，p90 97.2 → 73.8 ms。0.62.1 审计 114 组 × 5 档负载平均值无反例。）TTFT 分位 = 排队等待分位 + TTFT(b) + KV 队列等待分位 + 暴露的 KV 传输。decode 池是**连续批处理**：运行 batch n̄ 由 Little 定律的不动点 n̄ = λ_d · out / e · step(⌈n̄⌉) 决定（e = 每步 token 数，含投机解码），TPOT 均值 = step(⌈n̄⌉) / e（0.54 起改为生灭过程；0.64 起所见运行 batch 取 ⌊n̄⌋ / ⌊n̄⌋+1 两点混合而非取整，§23.3）；p90 / p99 TPOT 用运行 batch 的 Poisson 分位处的步时。部分负载下运行 batch 小于配置 batch，所以排队模型里的 TPOT 低于流体模型（流体按满 batch）。
- **KV 与池内集合通信争用**：没设 `pd.kv_GBps` 时 KV 与两池的 TP / EP 集合通信共用同一层（节点内 link 或跨节点 net）。KV 能用的带宽 = β · (1 − u_coll)，u_coll = 两池最忙级在该层的字节 / β / tick；反过来 KV 占该层 u_kv = λ · KV / (N · β)，各池在该层的时间乘 1 / (1 − u_kv) 后重算级时间（只在该层是瓶颈时才变慢）。KV 传输本身也是 M/D/1（每 prefill 副本一个出口）。给了 `pd.kv_GBps` 视为专用 KV 通道，不争用。
- **合并 · prefill 优先**：prefill 同样是带 b 上限的 M/D/1，占用 ρ_p；decode 只在剩余 1 − ρ_p 的时间里跑，不动点里步时除以 (1 − ρ_p)。新 prefill 到来时整批 decode 停顿 TTFT(b)：最长 token 间隔 = TPOT + TTFT(b)；一个请求生命周期内碰到的停顿数 ~ Poisson((λ/b) · 生命周期)，请求平均 TPOT 的分位 = TPOT(运行 batch 分位) + 停顿数分位 · TTFT(b) / out。TTFT 再加半个（p90 / p99 用一个）decode 步的残余。
- **合并 · 分块 prefill**：每次迭代带 C = `pd.chunk_tokens`（默认 512）个 prompt token。融合迭代在**级层面**合并：max(算力_d + f·算力_p, DRAM_d + f·(prefill 非权重字节) + 前缀 KV 重读, SLC_d, 链路_d + f·链路_p) + 同步，f = C / S（prefill 的权重读由 decode 迭代顺带完成；第 i 块要重读前面 i − 1 块的 KV，平均 KV · (S − C) / (2S) 每块）。prefill 是服务时间 ⌈S/C⌉ · T₁ 的 M/D/1，占用 ρ = λ · S · T₁ / C；decode 平均步时 = ρ · T₁ + (1 − ρ) · T₀，不动点同上。最长 token 间隔 = T₁ / e（停顿被块大小封顶），请求平均 TPOT 分位按「带块迭代数 ~ Poisson(ρ · 迭代数)」。
- **SLO goodput**（DistServe 口径）：满足 p90 TTFT ≤ `ttft_slo_ms` 且 p90 TPOT ≤ `tpot_slo_ms` 的最大 λ（稳定上限仍用倍增 + 二分；SLO 可行性随 λ 不单调——prefill batch 上限会换挡——0.63 起在 (0, 稳定上限] 上自上而下扫 24 个点、在第一个可行点与其上方的不可行点之间二分，并保证结果不低于任何已知可行点（含当前工作点），§22.3；0.64 起 batch 上限按 SLO 选择，可行集为各上限可行集之并，§23.1；没满足的设置则是稳定上限），× out / 卡数。PD 还在同总卡数的所有切分上搜 SLO goodput 最优的切分（`pd_slo_best_split`、`pd_slo_splits`）。
- **能耗**：用现有的动作计数（§13），每输出 token = (prefill 每 token 计数 × S + decode 每 token 计数 × out) / out，取各模式在该负载下的实际运行点（PD：prefill batch 上限、decode 运行 batch；合并：同上；分块：不重复读权重、加前缀 KV 重读），PD 再加 KV 传输字节（link 或 net）；卡·秒 / token = 卡数 / λ / out（含闲置）。给了能耗表才出 J / token。
- **量级**（Qwen3-8B，1P + HBM3E，TP2，batch 64，prompt 4096，out 512，PD prefill 2 + decode 6，load 0.8 → 7.21 req/s，SLO TTFT 400 / TPOT 10 ms）：PD TTFT p50 / p90 / p99 = 102 / 216 / 380 ms，TPOT 均值 / p90 = 6.5 / 9.0 ms，最长间隔 11 ms，SLO goodput 474 tok/s/卡（切分 2 + 6 最优；4 + 4 为 316）；合并 prefill 优先 79 / 114 / 169 ms、TPOT 5.4 / 7.7 ms 但最长间隔 83 ms，506 tok/s/卡；合并分块（512）134 / 220 / 349 ms、TPOT 9.4 / 12.2 ms、最长间隔 19 ms，433 tok/s/卡（0.51 数值；0.52 修正分块平均迭代后为 114 / 177 / 277 ms、TPOT 5.4 / 7.4 ms、16 ms、511 tok/s/卡，见 §18.2）。每输出 token DRAM：PD 2.46 GB、prefill 优先 3.56 GB（运行 batch 更小）、分块 2.23 GB。即这个例子里 p90 口径的 SLO goodput 合并 prefill 优先最高，但它的单次停顿（83 ms）是 PD（11 ms）的 7 倍多；PD 的价值体现在 token 间隔的最坏情况，分块 prefill 介于两者之间。prompt 更长 / TPOT SLO 更紧时结论会变（如 prompt 8192 out 256，PD 4 + 4 时 prefill 池先饱和，分块 512 在同到达率下不稳定）。
- **入口**：scenario `pd.load`（0–1，默认 0.8）、`pd.rate_rps`（覆盖 load）、`pd.chunk_tokens`（≥ 16，默认 512）；eval 响应 `pd.queue`（`lambda_rps`、`modes.{pd, coloc_prefill_first, coloc_chunked}` 各含 `ttft_ms.{mean,p50,p90,p99}`、`tpot_mean_ms / tpot_p90_ms / tpot_p99_ms`、`itl_max_ms`、`slo_rate_rps`、`slo_goodput_per_card` 与各自明细，`pd_slo_best_split`、`energy`）；CLI `--pd-load --pd-rate --pd-chunk`（另加 `--out-len --ttft-slo`）；Web PD 输入组三项与单点页「排队与尾延迟」表。
- **不做**（0.52 补了长度分布、前缀缓存、池布局搜索，见 §18.2）：抢占 / 换出 / KV 容量引起的排队、调度器开销、多级 SLO 调度、分块 prefill 与 PD 的组合（PD 的 prefill 池内部分块）、异构池；分位数相加而不是卷积（偏保守）；M/D/1 尾是渐近式，ρ 很小时略偏保守。

### 18.2 请求长度分布、前缀缓存、池布局搜索（0.52，默认关）

默认（长度固定、无前缀缓存、不搜布局）走 0.50 / 0.51 的原代码路径，数值不变；唯一的例外是下面的分块 prefill 修正。

- **长度分布**（`core/lengths.py`，「假设」）：`pd.prompt_cv` / `pd.out_cv` > 0 → 独立的对数正态 prompt / 输出长度，均值 = `serving.prompt` / `serving.out_len`，各离散成 N 档等概率（N = 15，0.56；0.52–0.55 为 8，见 §18.6），每档取条件均值 E[X | 第 k 档] = N · 均值 · (Φ(z_{k+1} − σ) − Φ(z_k − σ))，σ² = ln(1 + CV²)（对对数正态精确；均值保持，档内方差丢掉，所以实际 CV 略低：名义 1.0 → 0.90（8 档时 0.85），报告里给 `prompt_cv_eff` / `out_cv_eff`）。或者 `pd.length_mix` 直接给离散联合分布 [权重, prompt, 输出] 至多 16 行（可表达 prompt 与输出的相关），此时替代上面两项与 `serving.prompt / out_len`。
  - **prefill 每个 prompt 值单独求值**（注意力不是 S 的线性函数），不假设线性；排队从 M/D/1 换成 **M/G/1**（离散服务分布；P-K 均值精确，尾用同一 Cramér–Lundberg 渐近：θ 为 λ(E[e^{θτ}] − 1) = θ 的正根，C = (1 − ρ) / (λE[τe^{θτ}] − 1)；只有一个值时就是 M/D/1；测试用三点分布的 Lindley 仿真核对 p90 / p99）。batch 上限 b 时，长度 S_i 的请求在一批混合请求里耗时 τ_i + (b − 1)·τ̄。TTFT 分位 = prefill 等待分位 + 「自身 prefill + 暴露的 KV」的离散分位（两者都随 S_i 单调，按同一分位相加是精确的）+ KV 队列等待分位。KV 交接按每档的 KV 大小也是 M/G/1。
  - **decode**：Little 定律用 E[out]；运行 batch 的 Poisson 分布对输出长度分布不敏感（M/G/∞），所以 TPOT 分位的做法不变；槽位等待的 Allen–Cunneen 因子换成 (1 + c_s²)/2（c_s 为输出长度的 CV）。**长度偏置**：正在 decode 的请求按其输出长度被「看到」，时间平均上下文 = E[out·(S + out/2)] / E[out]；`serving.ctx` 被解读为一个代表性定长请求的上下文，再乘以它与 (E[S] + E[out]/2) 之比（定长时 = 1）。输出越分散、长 prompt 与长输出越相关，decode 上下文越长、步越慢。
  - **流体容量也随之变**：prefill 池 λ_p = r_p · b / Σ w_i·TTFT(b, S_i)（b 为按代表长度 S = round(E[S]) 选出的 prefill batch），decode λ_d = r_d · R_d(ctx_eff) / E[out]，KV λ_kv = min(N_p, N_d)·β / E[KV]；合并对照同样用逐档的每请求 prefill 时间。表里显示的 TTFT 是代表长度请求的。负载 `pd.load` 乘的就是这个容量。每个模式另给 `stable_rate_rps`（排队模型的稳定上限）。
- **前缀缓存**（`pd.prefix_hit` = h，0–0.99，「假设」）：每个请求前 ⌊h·S⌋ 个 token 已在 KV 缓存中。评估器新增 `serving.prefix_cached`（单点也可用，API / CLI `--prefix-cached`）：prefill 走 Phase(prefill, q = S − p, ctx = p)，即 GEMM 只算新 token、注意力对缓存前缀做（因果系数 ½(1 + p/S)）、读前缀 KV；吞吐仍按整个 prompt 计（请求率 = 吞吐 / S）。PD 的 KV 交接：`pd.prefix_on_decode`（默认 true，decode 池也持有同一前缀，如共享系统提示词 / 会话粘性路由）只传未缓存部分 KV ×(1 − p/S)，否则全量。分块 prefill 的块数 ⌈(S − p)/C⌉，每块重读缓存前缀 + 已完成的块。decode 不变（每个请求仍对完整上下文做注意力；跨请求共享前缀的 KV 读合并未建模）。命中率是输入：缓存容量、淘汰、路由命中都不建模。
- **池布局搜索**（`pd.search_layouts`，opt-in）：prefill 与 decode 池各自枚举每副本 1 / 2 / 4 / … / 64 卡（≤ 总卡数）与当前布局的所有布局（PP 小的优先，每池至多 64 个，超出标 `truncated`），× 同总卡数的所有切分。decode 每副本 batch 仍为 `serving.batch`，prefill batch 按 TTFT SLO 选。流体 goodput / 卡对全部布局对排序（同分时 prefill 余量大、TTFT 低、TPOT 低者在前）；排队模型下的 SLO goodput 只对：流体前 4 对、每种 decode 布局与每种 prefill 布局各自的最佳一对、当前布局，至多 24 对计算（流体排名会掩盖 TPOT / TTFT 差异）。输出 `layout_search.{rows, best_fluid, best_slo, current_layouts}`。
- **分块 prefill 修正（0.52 Fixed）**：0.51 的 decode 平均步时用了时间平均 ρ·T₁ + (1 − ρ)·T₀，带块迭代占比也按 ρ，二者都偏向长迭代（长度偏置）。每 token 的 TPOT 应是每次迭代的平均：带块迭代每秒 ν = λ·Σ w_i n_i，占时 ρ = λ·Σ w_i n_i T₁,i，平均迭代 = T₀ / (1 − ρ + ν T₀)，带块迭代占比 x = ν × 该值。块数改为 ⌈(S − p)/C⌉（0.51 的 ρ 用 S/C）。
- **量级**（同 §18.1 例子：Qwen3-8B 1P + HBM3E TP2 batch 64，prompt 4096 / out 512，PD 2 + 6，SLO TTFT 400 / TPOT 10 ms，load 0.8）：
  - 分块修正：合并 · 分块 TPOT 均值 / p90 由 9.4 / 12.2 ms 变为 5.4 / 7.4 ms，TTFT p90 220 → 177 ms，SLO goodput 433 → 511 tok/s/卡（略高于 prefill 优先的 506；PD 474）。PD 与 prefill 优先不变。
  - prompt / 输出 CV 1.0（离散后 0.85）：流体 577 → 556 tok/s/卡（decode 上下文 4096 → 4271），λ = 6.95 req/s 时 PD TTFT p90 216 → 1626 ms，prefill 优先 651、分块 928 ms，三者 SLO goodput 均为 0（长 prompt 档拉高 p90）。
  - 离散分布 70 % (1024, 256) + 30 % (11264, 1109)（均值仍为 4096 / 512，但长 prompt 也长输出）：decode 上下文按偏置 7609，流体 577 → 325 tok/s/卡。
  - 前缀命中 0.5：prefill 池容量 13.0 → 20.8 req/s（同一 TTFT SLO 下 batch 4 → 8），KV 每请求 576 → 288 MiB；PD 仍是 decode 瓶颈，流体不变；合并 625 → 672 tok/s/卡；TTFT p90 PD 216 → 94、prefill 优先 114 → 53、分块 177 → 92 ms；SLO goodput prefill 优先 541、分块 552、PD 474。
  - 布局搜索（定长）：流体最佳是 decode 改 TP1（583 tok/s/卡），但 TPOT 变长，SLO goodput 只有 412；SLO 最佳仍是 decode TP2（474）。长度分布 + 前缀 0.3 时，SLO 最佳是 prefill TP4 × 4 + decode TP1 × 4（132 tok/s/卡），只搜切分时为 117。
- **入口**：scenario `pd.prompt_cv / out_cv / length_mix / prefix_hit / prefix_on_decode / search_layouts`、`serving.prefix_cached`；响应 `pd.lengths`（来源、离散点、均值、有效 CV、`ctx_ratio`、`decode_ctx`、前缀）、`pd.layout_search`、每个模式的 `stable_rate_rps`、分块的 `chunk_share`；CLI `--pd-prompt-cv --pd-out-cv --pd-mix w:S:out,… --pd-prefix-hit --pd-prefix-not-on-decode --pd-search-layouts --prefix-cached`；Web PD 输入组「请求长度与前缀缓存」与「池布局搜索」表。
- **不做**（0.53 补了缓存容量 / LRU 淘汰、异构池、布局搜索里的 decode batch，见 §18.3）：长度感知调度（全部 FCFS）、prompt 与输出长度之外的请求异质性、decode 端共享前缀的 KV 读合并、PD prefill 池内分块、部分前缀匹配（radix tree）。

### 18.3 前缀缓存容量 / LRU 淘汰、异构池（0.53，默认关）

默认（`prefix_len` = 0、两池同一芯片 / 存储器）走 0.52 代码路径，数值不变；显式 `pd.prefix_hit` > 0 时覆盖容量模型（与 0.52 相同）。

- **前缀缓存容量**（`core/prefixcache.py`，「假设」）：工作集是 N = `pd.prefix_count`（默认 1000）个共享前缀（系统提示词 / few-shot 模板 / 多轮历史），每个长 `pd.prefix_len` tokens，按 Zipf(α = `pd.prefix_zipf`，默认 1；0 = 均匀) 流行度被独立请求（IRM）。缓存是整前缀 LRU，容量 K = 能放几个前缀；命中率用 **Che 近似**（Che / Tung / Wang 2002；Fricker / Robert / Roberts 2012）：特征时间 T 满足 Σ_i (1 − e^{−p_i T}) = K，条目 i 驻留概率 h_i = 1 − e^{−p_i T}，请求命中率 H = Σ_i p_i h_i。K ≥ N → H = 1（稳态，忽略冷启动）；K = 0 → H = 0。N ≤ 4096 精确求和；更大 N 前 1024 名精确、尾部按几何 rank 分箱（比 1.01，中点积分质量）——相对精确解误差 ≪ 10⁻⁵，与离散 LRU 仿真（IRM，热身后）误差 < 1 %。
  - **容量**：每前缀足迹 = 该长度下的 KV + indexer + 循环状态（与 memplan 一致，计 TP 分片与 DP 组）；每副本 K = floor(容量 / 足迹)。容量默认 = 权重 + 活跃 batch KV + 运行时预留之后剩余的 DRAM（各流水级取 min 再 × DP），或 `pd.prefix_cache_GB`。prefill 池与 decode / 合并副本各自算一遍（batch 不同 → 剩余不同）。
  - **路由**：随机（默认）→ 每个副本各自缓存全体前缀（H 用每副本 K）；`pd.prefix_affinity` → 前缀感知路由，副本分摊前缀（H 用总容量 K·replicas）。decode 池在 prefill 命中时也持有该前缀的概率 = Σ_i p_i h_i^P h_i^D / H_P（IRM 下独立 LRU）；KV 交接只传未缓存部分的期望比例。
  - **队列 / 流体**：命中 / 未命中两类请求按权重混合进 pts（PD prefill 用 H_P，合并用 H_C），其余与 0.52 的 `prefix_hit` 相同。显式 `pd.prefix_hit` > 0 时关掉容量模型并告警。
- **异构池**：`pd.prefill_chip`（芯片预设名或对象）与 `pd.prefill_mem_id`；None = 与 decode 池（scenario）相同。合并对照始终用 decode 池的芯片 / 存储器。闲置功率按卡计，两池同 `idle_W`（未按芯片区分）。
- **布局搜索 · decode batch**（`pd.search_decode_batch`，与 `search_layouts` 联用）：每种 decode 布局在 {B/2, B, 2B, 4B} ∩ [1, 4096] 中取满足 TPOT SLO（满批流体 TPOT）的最高吞吐；SLO goodput 用该 batch。异构内存时更有用。
- **量级**（同 §18.1：Qwen3-8B 1P + HBM3E TP2 batch 64，prompt 4096 / out 512，PD 2 + 6，SLO 400 / 10，load 0.8；前缀长 2048）：
  - N = 1000（剩余 DRAM ≈ 600 GB / 副本，足迹 302 MB → K ≈ 1900）：H = 100 %，与显式 `prefix_hit` ≈ 0.5（p = 2048）同量级——合并 672 tok/s/卡、TTFT p90 PD 94 / 优先 53 / 分块 92 ms、SLO 474 / 541 / 552。
  - N = 10⁵、α = 1：H_P = 57 %、H_D = 57 %（随机路由）；合并 651、p90 157 / 104 / 157、SLO 474 / 528 / 528。亲和路由 → H_D 68 %、H_C 71 %、合并 658。α = 0.6（更平）→ H ≈ 9 %，收益几乎消失。
  - 给定容量：10 / 50 / 200 GB → H = 19 / 34 / 47 %，合并 goodput 634 / 640 / 646；命中率随容量单调升。
  - 异构：prefill 改 100T → 流体 577 → 154 tok/s/卡（prefill 瓶颈），PD SLO goodput 0；prefill 只换 LPDDR5X（同 1P）→ 本例仍够 TTFT，与 baseline 相同。布局搜索 + decode batch：LPDDR5X decode、TPOT SLO 50 ms 时，SLO 最佳由 batch 64 改为 32（同 91 tok/s/卡）。
- **入口**：`pd.prefix_len / prefix_count / prefix_zipf / prefix_cache_GB / prefix_affinity / prefill_chip / prefill_mem_id / search_decode_batch`；响应 `pd.prefix_cache`（策略、容量、足迹、K、命中率、来源）、`pd.lengths.prefix_hit_source`（`capacity` / `explicit` / `off`）、`pd.prefill.{chip,mem_id,hetero}`；CLI `--pd-prefix-len --pd-prefix-count --pd-prefix-zipf --pd-prefix-cache-GB --pd-prefix-affinity --pd-prefill-chip --pd-prefill-mem --pd-search-decode-batch`；Web「前缀缓存容量」「异构池」输入与「前缀缓存容量」表。
- **不做**：部分前缀匹配（radix / 块粒度）、缓存预热与淘汰代价、队列模型更大 batch 对剩余 DRAM 的反馈、按芯片区分的闲置功率、PD prefill 池内分块、decode 端共享前缀的 KV 读合并、长度感知调度。

### 18.4 V4 服务验证：请求级 DES 对照闭式排队模型（0.54）

目的是验证，不加新功能。`core/pdsim.py` 是一个轻量的请求级离散事件仿真（DES，「假设」）。它复用闭式模型的**同一套逐步代价**：prefill TTFT(b, S, p)、decode step(k)、分块融合迭代、KV 传输，含 KV 与集合通信争用后的缩放；按事件推进泊松到达、FCFS、PD 两池（prefill 静态 batch 上限、每个 prefill 副本一条 KV 传输流、decode 连续批处理 B 槽）、合并 prefill 优先（每个迭代边界先跑等待中的 prefill 批）与分块 prefill（运行中的 decode + 队首 prompt 的 ≤ C token）、离散长度分布，以及前缀缓存（整前缀精确 LRU，Zipf id，缓存预热到稳态后只在测量窗内计命中率）。每个副本同一时刻只有一个迭代在跑，token 在迭代结束时计入，token 间隔从上一个 token 起算。因此 V4 衡量的是**排队 / 批处理近似**的误差，不是逐步代价模型的误差（后者见 V2 / V3）。

- **网格**（`scripts/v4_serving.py`，结果存 `accel_dse/data/v4_serving.json`）：Qwen3-8B 1P + HBM3E TP2 batch 64，prompt 4096 / out 512，SLO 400 / 10 ms，PD prefill 2 + decode 6；load 0.3 / 0.6 / 0.85 × 长度 CV 0 / 0.5 / 1（prompt 与输出同 CV）× 前缀缓存 关 / 开（容量模式：20000 个前缀 × 2048 token，Zipf 1）× 模式 PD / 合并 prefill 优先 / 合并分块 512，共 54 点。每点 DES 1500 个请求（前后各 375 个热身 / 排空）；SLO goodput 用同一组随机数对 DES 二分（分辨率 3 %）。误差 = (闭式 − DES) / DES：**> 0 表示闭式偏保守**（延迟更大）。SLO goodput 的符号反过来，> 0 仍表示保守（容量更小）。前缀命中率给的是绝对差。

| 模式 | 指标 | 最小 | 中位 | 最大 | 平均 \|误差\| | 落在容差内 |
|---|---|---|---|---|---|---|
| PD | TTFT p50 / p90 / p99 | −12 / −10 / −6 % | +3 / 0 / +4 % | +27 / +15 / +50 % | 8 / 3 / 8 % | 78 / 100 / 94 % |
| PD | TPOT p90 | −26 % | −5 % | +2 % | 6 % | 94 % |
| PD | 最长 token 间隔（每请求最长间隔的 p99） | −18 % | −10 % | +27 % | 12 % | 100 % |
| PD | SLO goodput | −8 % | −5 % | −1 % | 5 % | 100 % |
| 合并 prefill 优先 | TTFT p50 / p90 / p99 | −4 / −1 / −6 % | +2 / +1 / +5 % | +48 / +28 / +21 % | 8 / 3 / 8 % | 83 / 94 / 100 % |
| 合并 prefill 优先 | TPOT p90 | −43 % | −18 % | −8 % | 20 % | 61 % |
| 合并 prefill 优先 | 最长 token 间隔 | −39 % | −17 % | +1 % | 16 % | 89 % |
| 合并 prefill 优先 | SLO goodput | −6 % | −3 % | 0 % | 3 % | 100 % |
| 合并分块 | TTFT p50 / p90 / p99 | −22 / −24 / −48 % | +9 / +2 / −3 % | +57 / +7 / +23 % | 14 / 9 / 15 % | 72 / 78 / 89 % |
| 合并分块 | TPOT p90 | −37 % | −14 % | −8 % | 19 % | 67 % |
| 合并分块 | 最长 token 间隔 | −33 % | −12 % | +4 % | 13 % | 89 % |
| 合并分块 | SLO goodput | −5 % | −3 % | 0 % | 3 % | 100 % |
| 三种模式 | 前缀命中率（绝对差） | −0.006 | −0.004 | −0.001 | 0.004 | 100 % |

  容差（`validation.V4_BANDS`）：TTFT p50 ±15 %、p90 ±20 %、p99 ±30 %，TPOT p90 ±20 %，最长间隔 ±30 %，SLO goodput ±20 %，命中率 ±0.03。CV = 1 时 p90 prompt 单独就超过 400 ms TTFT SLO，两边 SLO goodput 都是 0，记为 n/a（每种模式 12 个有效点）。

- **闭式在哪里偏乐观 / 偏保守**
  - **SLO goodput（设计决策用的主指标）**：三种模式全部在 −8 % … 0 % 之间，即闭式略偏乐观，最多高估 8 %。12 个有效场景里，「PD 低于两种合并模式」的结论与 DES 全部一致；完整三模式排名 10 / 12 一致，另外 2 个是两种合并模式相差 < 1 tok/s/卡（DES 二分分辨率内）的并列。
  - **TTFT**：PD 与 prefill 优先的 p90 中位误差 ≈ 0。偏保守的地方有两处：一是 p90 恰好落在 M/D/1 等待原子边缘（ρ ≈ 1 − q，例如 load 0.6 时 prefill 优先 ρ_p ≈ 0.1），此时分位数是病态的，最多 +30 %；二是 CV = 1 时 PD 的 p99（+50 %），Cramér–Lundberg 指数尾在混合服务时偏重。分块 prefill 在 load 0.85 时 TTFT 偏乐观（p90 −24 %、p99 −48 %）：闭式按「平均运行 batch」的融合迭代计 prefill 服务时间，而高负载时运行 batch 涨落大、块迭代变慢。
  - **TPOT（合并模式系统性偏乐观）**：prefill 优先与分块的 TPOT p90 在所有 36 点上都偏乐观，中位 −14 … −18 %，最差 −43 %（load 0.85、CV 1）。原因：prefill 停顿让进入 decode 的到达变成成批的，而泊松输入的 birth–death 看不到这种突发；运行 batch 涨落更大，step(k) 又随 k 线性变陡。PD 的 decode 输入是 prefill 池的输出（比泊松更平滑），中位只有 −5 %。
  - **最长 token 间隔**：PD 用运行 batch 0.99 点的单步时间，中位 −10 %。prefill 优先 = 一个 decode 步 + 一生中遇到的最长 prefill 忙期，固定长度时在 ±2 % 内；CV 越大越偏乐观（CV 1、load 0.6 为 −38 %），因为忙期内批数与各批长度假设独立，而长 prompt 本身会拉长忙期、引来更多到达。
  - **前缀命中率**：Che 与精确 LRU（热身后）相差 < 0.01。

- **DES 发现并修正的问题**（只影响 PD 报告与其中的合并对照；默认合并评估的 1356 项指纹不变）
  1. **decode 连续批处理**（真实建模问题）：0.51–0.53 用 Little 不动点 n̄ = λ·out·TPOT(⌈n̄⌉)，忽略了占用涨落。step(k) 随 k 明显增大时（本例 step(k) ≈ 1.48 + 0.63·k ms）它偏乐观：load 0.8 时平均 TPOT 6.5 ms，DES 9.05 ms。改为 **birth–death 处理器共享模型**：μ(n) = min(n,B)·e/(out·step(min(n,B)))，π 由细致平衡给出，E[TPOT] = E[k]/(λ·out)，得 9.21 ms。运行中的请求看到的 batch 按 π(k)·k/step(k) 加权；请求平均 TPOT 分位数 = (看到的均值 + z·0.858·标准差)/e（OU 窗口因子「假设」）；最长间隔取看到的 batch 分布的 0.99 点（离散）。
  2. **prefill 优先的 TPOT 分位数重复计入停顿**：旧式在已经 ÷ share 拉长的 TPOT 上再加停顿分位数。改为均值 + √(decode 占用离差² + (z·√m·lat/out)²)。
  3. **prefill 优先的最长间隔偏乐观**：旧式「TPOT + 一次 prefill」≈ 87 ms，DES p99 为 322 ms。改为一个 decode 步 + 一生中最长 prefill 忙期的 0.99 点：忙期以 λ(1 − ρ) 起始；每忙期批数 cap 1 时为 Borel(λ·lat)，cap > 1 时为几何分布；时长按长度混合卷积「假设」。得 314 ms。
  4. **服务时间混合时 TTFT 分位数逐项相加**（偏保守，CV = 1 时 p90 +50 % 量级）：改为 W ⊕ L_i 的正确分位数：P(T > t) = Σ w_i·P(W > t − L_i)，W 取 M/G/1 Cramér–Lundberg 尾（`queueing.mg1_sum_quantile`）。服务时间只有一个取值时与旧式完全相同。KV 队列分位数仍相加（量小）。
  5. 分块 prefill 的 decode 也走 birth–death（平均迭代 tbar(k) = T₀(k)/(1 − ν(T₁(k) − T₀(k)))），prefill 服务按「看到的 batch」的融合迭代计。

- **数值变化**（§18.1 例子，load 0.8；0.53 → 0.54）：
  - PD：TTFT 102 / 216 / 380 ms 不变；TPOT 均值 / p90 6.49 / 9.00 → 9.21 / 14.06 ms；最长间隔 10.9 → 22.8 ms；SLO goodput 474 → 410 tok/s/卡（最佳切分仍为 2 + 6）。
  - 合并 prefill 优先：TTFT 79 / 114 / 169 → 80 / 115 / 171 ms；TPOT 5.35 / 7.69 → 7.45 / 11.14 ms；最长间隔 83.5 → 314 ms；SLO goodput 506 → 442。
  - 合并分块：TTFT 114 / 177 / 277 → 129 / 209 / 330 ms；TPOT 5.35 / 7.39 → 7.43 / 11.09 ms；最长间隔 16.1 → 23.0 ms；SLO goodput 511 → 443。
  - CV 0.5：PD TTFT p90 539 → 409 ms，SLO goodput 316 → 404（卷积修正去掉了逐项相加的保守量）。
  - 前缀容量（N = 20000）：PD TTFT p50 49 → 77 ms（命中 / 未命中两类服务时间卷积），p90 147 → 129 ms。

  定性结论不变：SLO goodput 合并模式略高，prefill 优先的单次停顿远大于 PD。量级上停顿差距更大（314 vs 23 ms，0.53 为 83 vs 11 ms）。§18.1–18.3 里写的排队数值是 0.51–0.53 当时的值。

- **入口**：`accel-dse validate` 打印 V4 实时子集（load 0.6 × {CV 0、CV 1、前缀开} × 3 种模式，≈ 15 s；`--no-v4` 跳过）；完整网格 `PYTHONPATH=. python3 scripts/v4_serving.py [--slo]`（SLO 二分约 12 min）。**可选** `pd.simulate` / CLI `--pd-sim` / Web「DES 仿真尾部」：在 PD 报告里给每个稳定模式附 `pd.queue.sim.modes.{mode}`，包括 DES 的 TTFT / TPOT / 最长间隔分位数、命中率，以及相对闭式的误差（n = 1500，单点慢约 3–5 s；默认关，闭式数值不受影响）。
- **剩余缺口**（0.55 的处理与仍剩的缺口见 §18.5）：合并模式 TPOT 尾偏乐观（见上，需要一个调制到达的 QBD 才能修，没做）；CV 大时 prefill 优先的最长停顿偏乐观；分块 prefill 在高负载时 TTFT 尾偏乐观；DES 自身的「假设」：泊松、FCFS、静态 prefill 批、随机路由、无抢占 / 换出 / KV 容量排队、调度开销为 0，单一场景族（8B 稠密模型，单一芯片）。

### 18.5 合并模式尾部、KV 容量排队、更多场景族（0.55）

默认（合并、非 PD）评估不变（1356 项指纹 0 差异）。PD 报告里的闭式排队数值会变：主要是两种合并对照的 TPOT 尾、分块 TTFT 尾，以及三种模式的最长间隔。KV 容量策略默认关。

- **先看 DES 噪声**：load 0.85 时 decode 贴近 KV 带宽饱和，运行 batch 的积分自相关时间 τ_int 为 18–42 s，而一个请求的生命周期只有 4–6 s。单 seed、1500 个请求的 DES 在不同 seed 之间，平均 TPOT 的标准差约 11 %，TPOT p90 约 17 %，最长间隔约 20 %。所以 0.54 表里的「最差 −43 %」大半是 DES 自身的噪声。0.55 的网格每点跑 3 个独立 seed、每个 3000 个请求（热身 750），指标取平均，并报告 `noise` = seed 间标准差 / 均值。SLO goodput 也取 3 个 seed 的平均，二分分辨率 1 %：一次 1500 请求的二分在 SLO 工作点的散布约 ±7 %，而 0.54 只用一个 seed，二分返回下界还会额外偏低。
- **合并 · prefill 优先：成批进入 decode**（0.54 剩余缺口，TPOT 尾偏乐观的主因）。一个 prefill 忙期结束时，忙期里完成的 X 个请求同时进入 decode，所以 decode 的输入不是泊松，而是成批到达。0.55 把 birth–death 换成**批到达的跳降链**（向上可跳多格、向下一次一格）：批率 λ_b = λ(1 − ρ_p)/E[X]（每个 prefill 忙期一批），X 的分布取忙期内完成数（静态 batch 上限下的精确批链），decode 时钟只在 1 − ρ_p 的时间份额里走。只有成批到达还不够，同一批里的请求生命周期也是相关的。对无限服务台，成批到达带来的占用方差多出一项 λ_b·E[X(X − 1)]·∫P(O > t)² dt，相对于独立到达的比值为**配对系数 κ = 2∫P(O > t)² dt / E[O]**：指数输出长度为 1，定长为 2，CV 1 / 0.5 的离散对数正态约为 1.14 / 1.49。实现上不改链，而是在同一平均批率下调整批大小分布（`_shape_batch`）：a² < 1 时做二项抽稀，抽出的部分作为单个到达；a² > 1 时以概率 p = (a² − 1)F / (2E[X]² − (a² − 1)F) 两两合并批次。这样占用方差的 a²·κ 倍就落到链上了。本例运行 batch 方差：链 6.30 → 6.99，DES 7.08。
- **合并 · 分块：忙期内 decode 照常推进**。带块的迭代只多花 n_c·(T₁ − T₀)。忙期里 decode 仍按 T₀ 那部分前进（每周期「补偿量」R = 服务时长 − 多出时长），所以成批程度比 prefill 优先低得多。用更新—报酬过程求有效批强度：周期 C = Exp(λ) 空闲 + 忙期 W，(X, W) 的联合矩用分支恒等式 (1 − ρ)E[Y_a Y_b] = E[y₀a y₀b] + λE[y₀a S]E[Y_b] + λE[y₀b S]E[Y_a] + λ²E[S²]E[Y_a]E[Y_b] 精确算出；σ² = Var(X − rC)/E[C]，a² = (σ² − r)/(λ_b·E[X(X − 1)])，再乘 κ 后用同一个 `_shape_batch`。另外修了一个细节：最后一块往往不满，每个 prompt 的服务时间改为 (n − 1)·T₁ + T₁·(最后一块占比)，0.54 按整块计。
- **TPOT p90 的窗口因子**：请求平均 TPOT = 生命周期内看到的 batch 的时间平均，它的方差随生命周期 L 与占用相关时间之比而变。0.54 用常数 √(2/e) ≈ 0.858「假设」，0.55 改为 OU 过程的精确平均方差因子 win = √(2(x − 1 + e^{−x})/x²)，x = L/τ_int。τ_int 对跳降链直接解出：π_n μ_n d(n) = G(n) − λ_b Σ_{m>n} d(m) Σ_{i<n} π_i P(X ≥ m − i)，自上而下递推，O(N·L)；τ_int = Σ d·G / Var(k)。load 0.85 时 win ≈ 0.96–0.98，比 0.858 大，因为相关时间远长于生命周期，平均不掉。测试：M/M/∞ 时 τ_int = 1/μ（精确）。
- **分块 TTFT 尾：准静态混合 + 休假**。运行 batch 漂移的时间尺度（十几到几十秒）远长于一个 prefill 忙期，所以按看到的 k 分成 8 个等质量环境，每个环境是一个 M/G/1：服务时间是该 k 下的块迭代时长，只有一个服务值且在 6τ 以内时用精确 M/D/1（Erlang 公式），否则用 Cramér–Lundberg 尾。prompt 队列空时副本跑纯 decode 迭代，相当于长度为 T₀(k) 的多次休假，按 Fuhrmann–Cooper 分解等待 = M/G/1 等待 ⊕ U(0, T₀(k))。P(T > t) = Σ_b w_b Σ_i w_i·P(W_b > t − l_bi)（`queueing.mg1_mix_sum_quantile`）。prefill 明细多一个 `iter_ms_env`（各环境的块迭代时长范围）。
- **最长间隔**：PD 与分块原来取「看到的 batch 分布的 0.99 点」，没有考虑一生中 batch 会涨落多次。改为一生中最大 batch 的分位数：P(max_life k ≤ m) ≈ F_seen(m)·exp(−L·π_{m+1}μ_{m+1}/F(m))，指数里是 m → m + 1 的上穿率（水平穿越 + Poisson 成团近似）。prefill 优先仍是一个 decode 步 + 一生中最长的 prefill 忙期，但忙期内批数用上面整形后的批分布。
- **M/D/1 等待分位数**：靠近等待原子（ρ ≈ 1 − q）时，Cramér–Lundberg 渐近 c·e^{−θx} 偏差大（V4 里 TTFT p90 +20 … +30 %）。等待在 6 个服务时间以内时，改用 Erlang 的精确 CDF P(W ≤ x) = (1 − ρ)·Σ_{k ≤ x/τ} [λ(kτ − x)]^k/k!·e^{−λ(kτ − x)}（`queueing.md1_cdf`），再往外仍用渐近式。测试与 Lindley 仿真核对。

- **KV 容量排队 / 抢占**（`pd.kv_policy` = `off`（默认）/ `wait` / `recompute`，`pd.kv_capacity_GB`，「假设」）。每个 decode（或合并）副本能放 K 个 KV token：`kv_capacity_GB` ÷ 每 token KV 字节（KV + indexer，按该副本的 TP / PP / DP 分片；循环状态按序列计，忽略）；不给则取权重与运行时预留之后剩余的 DRAM（评估自身的活跃 batch KV 退回）。
  - `wait`（保守准入，vLLM 预留式）：准入时为 prompt + 全部输出预留 KV。每请求足迹 = E[S] + E[o²]/E[o]（在跑的请求按输出长度被看到，长度偏置）。`recompute`（乐观准入 + 抢占）：只占当前上下文，时间平均足迹 E[S] + E[o²]/(2E[o])。槽位 B_kv = ⌊K / 足迹⌋，有效槽位 min(B, B_kv)。**只有 B_kv < B（容量真的卡住）时才生效**，否则结果与 `off` 完全相同（有测试）。
  - 准入等待：在 birth–death / 批链上把槽位数换成 B_kv，等待均值 = Little 定律 (E[n] − E[k]) / λ × Lee–Longton 因子 (1 + c_s²)/2（链里的生命周期是无记忆的，相当于 M/M/c；c_s 为输出长度的 CV）。口径与 0.51 起的 B 槽位等待相同：在合并模式里先 prefill 后准入，在 PD 里 KV 传完再准入。等待**单独报告**（`kv_cap.slot_wait_mean_ms`），**不计入 TTFT / TPOT / SLO goodput**。
  - `recompute` 的抢占：足迹过程 F（每步长 k·e token，有请求完成时下降）上穿 K 时，抢占最年轻的请求，它回队首，之后重新 prefill S + 已生成长度。抢占率用 Rice 公式：ν = Σ_k π_k·φ_k(K)·k·e / step(k)，F | k ≈ N(k·m_f, k·v_f)。重算 prefill 像 prefill 优先那样占用副本，份额 f_r = ν·T_re，TPOT × 1/(1 − f_r)；被抢占请求自己的 TPOT 加上 T_re / out，最长间隔 ≥ T_re。f_r ≥ 1 标记不稳定。
  - DES 同步实现：每副本记 KV 占用，`wait` 预留、完成时释放；`recompute` 在迭代边界检查并抢占最年轻的请求，重算 prefill 的代价按 256 token 粒度缓存。策略打开时 DES 结果多 `slot_wait`（准入等待分位）与 `preempt_per_req`。
  - **对拍**（dense8b，容量 12 GB → B_kv = 17，多 seed DES）：`wait` 的 TPOT 均值 / p90 误差在 ±2 % 以内，准入等待 1306 ms，DES 1092 ms（+20 %）。`recompute` 的 TPOT 误差 −2 … −7 %；抢占次数明显低估（闭式每请求 0–0.5 %，DES 1–5 %），被抢占请求的最长间隔 p99（DES 约 1 s）没有体现在闭式里；CV 1 时准入等待偏乐观（1159 ms，DES 2460 ms）。
  - **量级**（§18.1 例子，`kv_capacity_GB` = 12）：`wait` 下 PD TPOT 均值 / p90 9.21 / 14.70 → 8.56 / 12.28 ms（运行 batch 被 KV 卡小），SLO goodput 403 → 413 tok/s/卡。这里的上升不代表更好：被卡住的请求在排队，这部分等待没有计入 SLO。合并两种模式 411 → 417。`recompute` 下 PD 8.67 / 12.59 ms，410 tok/s/卡。不给容量时剩余 DRAM 约 600 GB / 副本，不生效，数值与 `off` 相同。
  - **入口**：scenario `pd.kv_policy / pd.kv_capacity_GB`；每个模式的 `kv_cap`（策略、`capacity_tokens`、来源、`footprint_tokens`、`slots_kv`、`binds`、`p_wait`、`slot_wait_mean_ms`、`preempt_per_req`、`recompute_ms`、`recompute_share`）；CLI `--pd-kv-policy --pd-kv-capacity-GB`；Web PD 输入「KV 策略」「KV 容量 GB」，排队表的注释里显示。
- **更多场景族**（`validation.V4_FAMILIES`，均为 1P + HBM3E 6600、batch 64、prompt 4096 / out 512）：
  - `dense8b`：Qwen3-8B TP2，PD 2 + 6 卡，SLO 400 / 10 ms，完整网格 18 点。
  - `moe30b`：Qwen3-30B-A3B（MoE），每副本 2 卡（TP2 · EP2），PD 2 + 6 卡，SLO 400 / 15 ms。
  - `tp4_32b`：Qwen3-32B，每副本 TP4 跨 4 卡，PD 4 + 8 卡，SLO 600 / 15 ms。
  - 后两族各跑 load 0.3 / 0.6 / 0.85 × CV 0 / 1、无前缀，6 点。合计 30 点 × 3 种模式 × 3 个 seed。`scripts/v4_serving.py [--n 3000] [--seeds 3] [--jobs N] [--families …] [--slo]`。带 `--slo`、8 个进程时约 20 分钟（box 计时，受负载影响）。

- **V4 误差（0.55 网格，30 点 × 3 seed × 3000 请求；误差 = (闭式 − DES)/DES，> 0 = 闭式偏保守；SLO goodput 符号反过来，> 0 仍为保守）**。全部场景族合并：

| 模式 | 指标 | 最小 | 中位 | 最大 | 平均 \|误差\| % | 落在容差内 | DES 噪声中位 |
|---|---|---|---|---|---|---|---|
| PD | TTFT p50 / p90 / p99 | −7 % / −6 % / −6 % | +1 % / 0 % / 0 % | +18 % / +8 % / +30 % | 5 / 2 / 6 | 93 % / 100 % / 97 % | 5 % / 5 % / 14 % |
| PD | TPOT 均值 / p90 | −2 % / −5 % | 0 % / 0 % | +4 % / +4 % | 1 / 2 | 100 % / 100 % | 4 % / 4 % |
| PD | 最长间隔 | −23 % | −2 % | +24 % | 10 | 100 % | 10 % |
| PD | SLO goodput | −1 % | +1 % | +3 % | 1 | 100 % | 6 % |
| 合并 prefill 优先 | TTFT p50 / p90 / p99 | −4 % / −1 % / 0 % | +1 % / +1 % / +4 % | +47 % / +8 % / +19 % | 7 / 2 / 6 | 87 % / 100 % / 100 % | 0 % / 0 % / 4 % |
| 合并 prefill 优先 | TPOT 均值 / p90 | −5 % / −11 % | −1 % / −4 % | +4 % / +2 % | 2 / 5 | 100 % / 100 % | 3 % / 4 % |
| 合并 prefill 优先 | 最长间隔 | −26 % | −1 % | +18 % | 5 | 100 % | 8 % |
| 合并 prefill 优先 | SLO goodput | −5 % | −1 % | +1 % | 3 | 100 % | 6 % |
| 合并分块 | TTFT p50 / p90 / p99 | 0 % / −8 % / −10 % | +2 % / −1 % / −1 % | +22 % / +2 % / +14 % | 5 / 2 / 4 | 87 % / 100 % / 100 % | 3 % / 2 % / 5 % |
| 合并分块 | TPOT 均值 / p90 | −4 % / −9 % | −1 % / −4 % | +6 % / +5 % | 2 / 4 | 100 % / 100 % | 3 % / 4 % |
| 合并分块 | 最长间隔 | −7 % | +5 % | +21 % | 6 | 100 % | 3 % |
| 合并分块 | SLO goodput | −5 % | 0 % | +2 % | 2 | 100 % | 6 % |
| 三种模式 | 前缀命中率（绝对差） | +0.005 | +0.005 | +0.005 | 0.005 | 100 % | — |

  分场景族（中位，括号内为绝对值最大的一点）：

| 场景族 | 模式 | TPOT p90 中位（最差） | TTFT p99 中位（最差） | 最长间隔 中位（最差） | SLO goodput 中位 |
|---|---|---|---|---|---|
| dense8b（18 点） | PD | 0 %（−5 %） | −1 %（+16 %） | −5 %（+24 %） | +1 % |
| dense8b（18 点） | 合并 prefill 优先 | −4 %（−10 %） | +5 %（+16 %） | −1 %（+18 %） | −2 % |
| dense8b（18 点） | 合并分块 | −3 %（−8 %） | −1 %（+14 %） | +6 %（+21 %） | −2 % |
| moe30b（6 点） | PD | 0 %（−3 %） | +4 %（+30 %） | 0 %（−12 %） | +1 % |
| moe30b（6 点） | 合并 prefill 优先 | −3 %（−9 %） | +5 %（+19 %） | −1 %（−26 %） | −1 % |
| moe30b（6 点） | 合并分块 | −2 %（−9 %） | −2 %（+11 %） | +6 %（+10 %） | 0 % |
| tp4_32b（6 点） | PD | 0 %（+4 %） | +3 %（+29 %） | −1 %（+18 %） | +1 % |
| tp4_32b（6 点） | 合并 prefill 优先 | −6 %（−11 %） | +2 %（+14 %） | 0 %（−18 %） | +1 % |
| tp4_32b（6 点） | 合并分块 | −5 %（−9 %） | −1 %（−10 %） | +4 %（+5 %） | +2 % |

  容差（`V4_BANDS`）与 0.54 相同，新增 TPOT 均值 ±10 %。「DES 噪声中位」= 每点 3 个 seed 之间的标准差 / 均值，再取各点的中位数。误差比它小的部分，用这个网格分辨不出来。
  - 对照 0.54（dense8b、单 seed）：合并 prefill 优先 / 分块的 TPOT p90 中位 −18 / −14 % → −4 / −3 %（全部族），最差 −43 / −37 % → −11 / −9 %；最长间隔中位 −17 / −12 % → −1 / +5 %；分块 TTFT p99 最差 −48 % → −10 %，p90 最差 −24 % → −8 %；PD TPOT p90 最差 −26 % → −5 %。SLO goodput −8 … 0 % → −5 … +3 %。
  - TPOT 目标（中位 ±10 %、最差 ±25 %）在三族三模式都达到。TTFT p50 与 p99、最长间隔各有几个点超出 ±25 %，见下面「剩余缺口」。

- **数值变化**（§18.1 例子，load 0.8；0.54 → 0.55）：

| 场景 | 模式 | TTFT p50 / p90 / p99 ms | TPOT 均值 / p90 ms | 最长间隔 ms | SLO goodput tok/s/卡 |
|---|---|---|---|---|---|
| 定长 | PD | 102 → 94 / 216 / 380 | 9.21 / 14.06 → 9.21 / 14.70 | 22.8 → 26.5 | 410 → 403 |
| 定长 | 合并 prefill 优先 | 80 / 115 / 171 → 81 / 109 / 164 | 7.45 / 11.14 → 8.22 / 13.35 | 314 → 315 | 442 → 411 |
| 定长 | 合并分块 | 129 / 209 / 330 → 128 / 214 / 377 | 7.43 / 11.09 → 8.20 / 13.38 | 23.0 → 30.5 | 443 → 411 |
| CV 0.5 | PD | 144 / 409 / 778 不变 | 9.26 / 14.16 → 9.26 / 14.80 | 23.0 → 26.9 | 404 → 397 |
| CV 0.5 | 合并 prefill 优先 | 77 / 237 / 335 → 77 / 237 / 336 | 7.91 / 11.90 → 8.65 / 14.11 | 528 → 666 | 427 → 399 |
| CV 0.5 | 合并分块 | 163 / 336 / 568 → 145 / 358 / 624 | 8.08 / 12.13 → 8.63 / 14.14 | 28.6 → 36.2 | 425 → 399 |
| 前缀（N = 20000） | PD | 77 / 129 / 220 不变 | 9.21 / 14.06 → 9.21 / 14.70 | 22.8 → 26.5 | 410 → 403 |
| 前缀 | 合并 prefill 优先 | 51 / 83 / 131 → 52 / 83 / 132 | 6.61 / 9.82 → 7.10 / 11.29 | 230 → 237 | 465 → 438 |
| 前缀 | 合并分块 | 72 / 126 / 220 → 85 / 150 / 241 | 6.61 / 9.79 → 7.09 / 11.29 | 24.2 → 30.4 | 466 → 438 |

  PD 的最佳切分仍为 2 + 6。合并模式的 SLO goodput 降了 6–7 %，是成批到达让 TPOT p90 变大的结果。三种模式的 SLO goodput 差距从 0.54 的 +8 %（合并更高）缩小到 +2 %；PD 单次停顿小一个数量级（26 vs 315 ms）的结论不变。

- **没做**
  - **PD prefill 池内分块**（brief 第 4 项）：评估器没有单独的「C token prefill 块」代价，现有融合迭代 `_fused_step` 需要一个 decode 迭代来搭载。而在 FCFS 的纯 prefill 池里，分块只增加前缀 KV 的重读，对 TTFT 没有收益；要有收益，需要长短 prompt 混合调度（SRPT / 抢占式），超出本轮范围。
  - **swap 抢占**：需要新增一个 host 链路带宽参数，只做了 recompute。

- **剩余缺口**
  - **CV 下的 TTFT p50**（+15 … +47 %，DES 噪声 < 4 %）：8 档等概率长度分布使 P(L ≥ l₅) 恰好为 0.5，p50 落在 TTFT 分布的跳变 / 平台上。例如 tp4_32b load 0.6 CV 1 prefill 优先：DES 的经验 CDF 在 150–190 ms 之间只从 0.525 升到 0.537，在 203 ms 处升到 0.595。概率误差 < 0.04 就能让分位数移动 47 %。这是分位数本身病态，不是排队模型的问题；p90 / p99 不受影响。（0.56 已处理：15 档 + 精确 M/G/1 等待，见 §18.6。）
  - **CV 1、load 0.85 的 PD TTFT p99**：+29 / +30 %（tp4_32b / moe30b；DES 噪声 11–19 %）。Cramér–Lundberg 指数尾在混合服务时偏重。
  - **最长间隔**：moe30b load 0.3 prefill 优先 −26 %（DES 噪声 20 %）；PD 在低负载时 −18 … −23 %（PD 的间隔只有一个 decode 步，≈ 6–17 ms，绝对差 < 4 ms）。
  - **KV `recompute`**：抢占次数低估（闭式每请求 0–0.5 %，DES 1–5 %）。Rice 公式只计上穿，没计重算请求再次进入后的连锁抢占。被抢占请求的最长间隔（DES 约 1 s）没有体现在闭式里；CV 1 时准入等待偏乐观约 2×。
  - **KV 准入等待不进 SLO**：等待单独报告，不计入 TTFT / TPOT / SLO goodput，所以容量卡住时 SLO goodput 会偏高（上例 403 → 413）。合并模式在 DES 里也是先 prefill 后准入，与 vLLM「先分配 KV 再 prefill」的顺序不同「假设」。
  - DES 自身的「假设」不变：泊松、FCFS、静态 prefill 批、随机路由、调度开销为 0；三个场景族都是 1P + HBM3E、prompt 4096 / out 512。

### 18.6 KV 准入顺序、抢占尾部、swap、长度档、按池静态功耗（0.56）

默认（合并、非 PD）评估不变（1356 项指纹 0 差异）。KV 策略默认仍关。PD 报告里只有长度有 CV 时数值变化（长度档 8 → 15）。

- **输入校验**：scenario 的每个字段按 dataclass 的类型注解校验（`typing.get_type_hints`）。类型不符返回 400，例如 `Serving.batch: expected integer, got number 2.5`。整数字段接受整值浮点数，|x| ≤ 2⁵³。`chip` / `pd.prefill_chip` 可以写预置名。模糊测试：每个字段依次换成 null / 布尔 / 字符串 / 列表 / 对象 / 非整数 / 1e300 / 负数，打 7 个接口（eval / layouts / compare / stability / fit / pareto / sweep），结果全部 400，没有 500。
- **长度档 8 → 15**：等概率 N 档时，第 k 个档边界的累计概率是 k/N。N = 8 时 0.5 正好是边界，P(S ≥ s₅) = 0.5，TTFT 中位数落在分布的跳变上：0.55 网格里 tp4 L0.6 CV1 的 DES 经验 CDF 在 150–190 ms 有一个 0.525–0.537 的平台，闭式只要偏 0.03 的概率就落到另一个档，p50 误差 +47 %。N = 15 时 0.5·N = 7.5、0.9·N = 13.5、0.99·N = 14.85，三个报告分位数都在档内部。闭式和 DES 用同一套档。档内展宽（连续长度）没做：DES 要对每个连续长度重算引擎代价，太贵；只要分位数不落在档边界上，档内展宽对分位数只是二阶影响。改档后暴露出的 PD p50 偏差（+15 … +29 %）来自等待分布，见下一条。
- **KV 准入顺序**（`pd.kv_admit`）：vLLM 的调度器先为请求分配 KV block，再跑它的 prefill；PD 下 decode 侧先预留空间，再拉取 KV。所以 KV 不够时的准入等待就是 TTFT 的一部分。`before_prefill`（默认）：容量受限时 TTFT = 原 TTFT ⊕ 准入等待。原 TTFT 分布由 p50 / p90 / p99 分段对数线性重建；等待取「原子 1 − p + Exp(W/p)」，p = P(等待) 来自批链。卷积后的分位数也进入 SLO 到达率的搜索。`after_prefill` = 0.55（等待单列）。KV 策略关时两者完全相同，DES 也逐位一致（与 0.55 对拍：off 与 after_prefill 每个请求的时间都相同）。
  - 注意：KV 受限可以让 SLO goodput 比不限时还高（§18.1 例子 12 GB：403 → 413 tok/s/卡）。这不是计数错误：在 SLO 到达率处 p ≈ 6 %、等待均值 49 ms，TTFT p90 仍满足；而 KV 把运行 batch 压到 17，TPOT p90 14.70 → 12.28 ms，解除了 TPOT 的约束。KV 容量这时起的是准入控制的作用。DES 同样如此。
- **抢占率**（recompute / swap）：0.55 只算足迹过程上穿 K 的 Rice 率，DES 的抢占次数比它多约 10×。原因是乐观准入：副本满时只要当前足迹放得下就准入新请求，留给后续增长的余量只有 ~U(0, S)，几乎每次这样的准入后很快就要抢占。新增饱和项 ν_sat = p·λ_r·E_S[(E[o]/S)(1 − e^{−S/E[o]})]：准入时余量 U(0, S)，余量在请求结束前耗尽的概率；输出长度按指数分布「假设」。
- **抢占尾部**：被抢占的请求停顿 T_fix（recompute：重新 prefill S̄ + ḡ 个 token；swap：换出 + 换入），再回到队首等一个槽位（≈ Exp(1/λ_r)）。每请求抢占 p_v 次。TPOT 均值加 p_v·T_fix / E[o]；TPOT 与最长间隔的 q 分位数取混合：p_v > 1 − q 时，被抢占者的间隔 T_fix + ln(p_v / (1 − q)) / λ_r 决定分位数。恢复占用副本的时间比例 f = ν·T_stall，f ≥ 1 判不稳定（「KV 容量不足：… 抢占的恢复占满副本」）。
- **swap**：`pd.kv_policy = "swap"`，`pd.swap_GBps` = 每卡主机链路 GB/s（不给就用 `workload.host_GBps` = 50 GB/s「假设」：PCIe 5.0 x16 有效带宽）。T_io = (S̄ + ḡ)·KV 字节/token ÷（GB/s × 每副本卡数），停顿 = 换出 + 换入 = 2·T_io。DES：换出在抢占时计入迭代，换入在重新准入时计入。§18.1 例子 12 GB：swap 一次 2 × 6.4 ms，重算一次 86 ms；被抢占者的平均间隔分别为 429 ms 和 501 ms，大部分是回队等待，所以两种策略的最长间隔差得不多（389 vs 462 ms）。
- **M/G/1 等待：精确的 Pollaczek–Khinchine 律**（`queueing._pk_surv`）。0.54 / 0.55 用 P(W > x) ≈ min(ρ, c·e^{−θx})（Cramér–Lundberg），只有尾部准确。CV 1 时 15 档服务时间从 7 ms 跨到 700 ms，正的等待大多很短（排在短请求后面），CL 律给这部分的概率太少。例如 dense8b L0.3 CV1：DES 的 TTFT p50 正好是第 10 档的原子（71.5 ms），闭式落到下一档（92 ms，+27 %）。精确律：W = 几何（ρ）个均衡剩余服务时间之和；离散服务律的剩余密度 P(S > x)/E[S] 是阶梯函数，所以 g(x) = λ·Σ_i w_i[G(x) − G(x − τ_i)]（G 含 0 处原子 1 − ρ）可以在 0 … 4·τ_max 的 800 点网格上推进。原子造成的台阶精确积分，其余用梯形格式（二阶）；4·τ_max 以外接 CL 指数尾，从网格末端的值续上。均值与 P-K 公式相对差 < 1e-6，M/D/1 与 Erlang 精确式一致，生存函数与 300 万样本 Lindley 仿真差 ≤ 0.002。用于：PD / prefill 优先的 TTFT 卷积（`mg1_sum_quantile`）、分块准静态混合里服务时间混合的环境（`mg1_mix_sum_quantile`）、报告的 prefill 等待分位数（`mg1`）。单一服务时间仍走精确 M/D/1，定长结果不变。每个（λ, 服务律）缓存一次，约 2 ms。CV 1 的 PD 报告 3.1 → 6.1 s：15 档让 `_mg1_busy` 变贵，已改为平铺数组。
- **按池静态功耗**：`energy.idle_W_prefill` = PD prefill 池芯片每卡 W（异构 prefill 芯片时填；不给 = `idle_W`；没有默认值）。静态能耗 = W × 每输出 token 的卡·秒（按挂钟时间，忙闲都算），PD 拆成 prefill 池 n_p/λ/out 与 decode 池 n_d/λ/out。动态能耗两池仍用同一张能耗表「假设」。各模式输出 `tok_per_J`、`static_share`。只给 `idle_W_prefill` 时写明 decode 池（及合并部署）的静态能耗未计入。单芯片报告不用这个字段。
- **主机链路能耗**（0.61.4）：`energy.pJ_bit_host` = PD `kv_policy = swap` 换出 / 换入经主机链路（PCIe 等）每 bit 能耗，没有默认值；不给时 swap 的主机链路字节不计能耗并给出 `host_note`。

- **V4 误差（0.56 网格，30 点 × 3 seed × 3000 请求；误差 = (闭式 − DES)/DES，> 0 = 闭式偏保守；SLO goodput 符号相反，> 0 仍为保守）**。全部场景族合并，括号内为 0.55：

| 模式 | 指标 | 最小 | 中位 | 最大 | 平均 \|误差\| % | 落在容差内 | DES 噪声中位 |
|---|---|---|---|---|---|---|---|
| PD | TTFT p50 / p90 / p99 | −18 % / −12 % / −6 % | 0 % / −1 % / +1 % | +2 %（+18）/ +5 % / +30 % | 4 / 3 / 5 | 93 % / 100 % / 97 % | 5 % / 8 % / 15 % |
| PD | TPOT 均值 / p90 | −2 % / −6 % | 0 % / 0 % | +5 % / +7 % | 1 / 2 | 100 % / 100 % | 4 % / 4 % |
| PD | 最长间隔 | −23 % | −3 % | +24 % | 10 | 100 % | 9 % |
| PD | SLO goodput（30 点；0.55：18 点） | −1 % | +3 % | +16 % | 6 | 100 % | 6 % |
| 合并 prefill 优先 | TTFT p50 / p90 / p99 | −12 % / −11 % / −4 % | 0 % / 0 % / +1 % | +5 %（+47）/ +8 % / +5 %（+19） | 2 / 2 / 2 | 100 %（87）/ 100 % / 100 % | 1 % / 3 % / 2 % |
| 合并 prefill 优先 | TPOT 均值 / p90 | −5 % / −11 % | −1 % / −5 % | +4 % / +1 % | 2 / 6 | 100 % / 100 % | 3 % / 5 % |
| 合并 prefill 优先 | 最长间隔 | −26 % | −3 % | +8 % | 6 | 100 % | 6 % |
| 合并 prefill 优先 | SLO goodput | −11 % | −1 % | +5 % | 4 | 100 % | 6 % |
| 合并分块 | TTFT p50 / p90 / p99 | −4 % / −9 % / −15 % | 0 % / −3 % / −3 % | +3 %（+22）/ +1 % / +4 %（+14） | 1 / 4 / 5 | 100 %（87）/ 100 % / 100 % | 3 % / 6 % / 4 % |
| 合并分块 | TPOT 均值 / p90 | −4 % / −11 % | −1 % / −4 % | +6 % / +4 % | 2 / 5 | 100 % / 100 % | 3 % / 5 % |
| 合并分块 | 最长间隔 | −7 % | +5 % | +22 % | 6 | 100 % | 4 % |
| 合并分块 | SLO goodput（27 点） | −5 % | −2 % | +2 % | 2 | 100 % | 6 % |

  分场景族（中位，括号内为绝对值最大的一点）：

| 场景族 | 模式 | TPOT p90 | TTFT p50 | TTFT p99 | 最长间隔 | SLO goodput |
|---|---|---|---|---|---|---|
| dense8b（18 点） | PD | 0 %（+7 %） | 0 %（−13 %） | +1 %（+17 %） | −3 %（+24 %） | +3 %（+15 %） |
| dense8b | 合并 prefill 优先 | −4 %（−11 %） | +1 %（−12 %） | +2 %（+5 %） | −3 %（−11 %） | −5 %（−11 %） |
| dense8b | 合并分块 | −3 %（−10 %） | 0 %（−3 %） | −3 %（−15 %） | +6 %（+22 %） | −2 %（−5 %） |
| moe30b（6 点） | PD | 0 %（−3 %） | 0 %（−15 %） | +4 %（+30 %） | +2 %（−12 %） | +15 %（+16 %） |
| moe30b | 合并 prefill 优先 | −5 %（−11 %） | +1 %（+5 %） | +1 %（+5 %） | −10 %（−26 %） | +5 %（+5 %） |
| moe30b | 合并分块 | −3 %（−10 %） | 0 %（−4 %） | −5 %（−11 %） | +5 %（+11 %） | 0 %（−2 %） |
| tp4_32b（6 点） | PD | 0 %（+4 %） | 0 %（−18 %） | +3 %（+27 %） | −2 %（+18 %） | +14 %（+14 %） |
| tp4_32b | 合并 prefill 优先 | −5 %（−11 %） | 0 %（+2 %） | +2 %（+2 %） | −1 %（−6 %） | +5 %（+5 %） |
| tp4_32b | 合并分块 | −5 %（−11 %） | 0 %（+3 %） | −1 %（−10 %） | +4 %（+9 %） | +2 %（+2 %） |

  - SLO goodput 的统计点数从 18 增加到 30：0.55 时 CV 1 的点闭式和 DES 都是 0（8 档时 TTFT p90 落在最长档上，超过 SLO）。15 档后这些点的零负载 TTFT p90 恰好略低于 SLO，SLO 到达率由一条很平的 TTFT p90 曲线与 SLO 的交点决定，1 % 的 TTFT 差会变成几十 % 的速率差。PD 的 +14 … +16 % 就在这些点上，绝对值很小（moe30b：闭式 52 vs DES 62 tok/s/卡，tp4_32b 19 vs 22；CV 0 时分别约 390 和 150）。中间那次网格（15 档、仍用 CL 律）这里是 +56 %。
  - **PD TTFT p99 在 CV1 L0.85 时 +27 … +30 %（item 5）**：长跑 DES 表明这是 DES 运行太短，不是闭式的偏差。tp4_32b n = 30000（3 seed）：DES 9161 ± 485 vs 闭式 9197 ms（+0.4 %）；moe30b 5939 ± 304 vs 6178 ms（+4 %）；同一点 n = 3000 时 DES 只有 6898 ± 584。这些点 prefill 排队主导，KV 等待为 0。p99 由长 prompt 档 × 长等待决定，3000 个请求里落在 τ_int 窗口内的样本太少。网格仍按 3000 跑，两点注明。

- **KV 策略对拍**（不在网格里；dense8b，12 GB → B_kv = 17，before_prefill，6 seed × 3000 请求的 DES；闭式 / DES）：
  - load 0.6：三种策略都接近（等待 ≈ 0）。
  - load 0.85、CV 0、`wait`：PD TTFT p90 4415 / 3426 ms（+29 %），p99 10733 / 6552；TPOT 10.02 / 9.95 ms；准入等待 1306 / 978 ms。合并 TTFT p90 +17 … +22 %，p99 +40 %。
  - load 0.85、CV 1、`wait`：PD TTFT p90 9385 / 7184，p99 21441 / 11152；等待 2694 / 1870。
  - 抢占次数 / 请求（PD）：recompute CV0 0.053 / 0.030，CV1 0.105 / 0.100；swap CV0 0.053 / 0.019，CV1 0.105 / 0.056（0.55 只有 Rice 项，约低 10×）。被抢占者间隔 recompute 735 / 889 ms（CV0）、1049 / 1282 ms（CV1）。
  - `recompute` CV1 L0.85 的 PD TTFT p90 7913 / 17592 ms，闭式明显偏乐观：DES 中重算抢走 prefill / decode 时间，又引发更多抢占（连锁），闭式只按一阶的恢复占用 f = ν·T_stall 计入。合并分块的抢占次数偏高（0.068 / 0.027）。

- **数值变化**（§18.1 例子，load 0.8；0.55 → 0.56）：定长不变。前缀：PD TTFT p90 129 → 128 ms，分块 TTFT 85 / 150 / 241 → 83 / 146 / 242 ms，prefill 优先 p99 132 → 131 ms。CV 0.5：

| 模式 | TTFT p50 / p90 / p99 ms | TPOT 均值 / p90 ms | 最长间隔 ms | SLO goodput |
|---|---|---|---|---|
| PD | 144 / 409 / 778 → 141 / 439 / 848 | 9.26 / 14.80 不变 | 26.9 不变 | 397 不变 |
| 合并 prefill 优先 | 77 / 237 / 336 → 77 / 214 / 359 | 8.65 / 14.11 → 8.70 / 14.22 | 666 → 722 | 399 → 398 |
| 合并分块 | 145 / 358 / 624 → 138 / 370 / 674 | 8.63 / 14.14 → 8.68 / 14.24 | 36.2 → 37.7 | 399 → 398 |

  KV 12 GB（PD）：`wait` TTFT 94 / 216 / 380 → 121 / 1410 / 4872 ms（准入等待均值 354 ms，P(等待) 24 %，现计入 TTFT），SLO goodput 413 不变；`recompute` TTFT → 114 / 1095 / 4374 ms，最长间隔 12.8 → 462 ms，TPOT 8.67 / 12.59 → 8.73 / 12.68 ms；`swap` 最长间隔 389 ms，TPOT 8.69 / 12.63 ms。

- **入口**：scenario `pd.kv_admit`（before_prefill | after_prefill）、`pd.kv_policy = "swap"`、`pd.swap_GBps`、`energy.idle_W_prefill`。`kv_cap` 新增 `admit`、`in_ttft`、`p_wait`、`preempt_rice`、`preempt_sat`、`victim_gap_mean_ms`、`restore_share`、`swap_ms`、`swap_GBps_card`、`swap_source`；能耗新增 `tok_per_J`、`static_share`、`static_note`。CLI `--pd-kv-policy swap --pd-kv-admit --pd-swap-GBps --idle-W-prefill`。Web「KV 准入」「换出 GB/s / 卡」「静态 W / 卡 · PD prefill」。
- **剩余缺口**：KV `wait` 折叠后 TTFT 偏保守（p90 +20 … +40 %）；`recompute` 高负载 CV1 的连锁抢占；合并模式抢占次数偏高；PD TTFT p99 CV1 L0.85 受 DES 短跑影响；没有档内展宽。

### 18.7 KV 槽位队列（M/G/c）、抢占回填偏移与恢复不动点、DES 恢复记账、长跑 V4（0.57）

只在 `pd.kv_policy` ≠ off 且容量受限（B_kv < B）时生效；默认结果与 0.56 逐字节一致。

- **准入等待均值（M/G/c 两矩近似）**「假设」：槽位 = 服务台，占用时间 = decode 寿命（∝ 输出长度，c_s² 取输出长度的）。生灭链给出 M/M/c 式的 W_chain = (E[n] − E[k]) / λ_r。c_s² ≤ 1：W = c_s²·W_chain + (1 − c_s²)·W_M/D/c，W_M/D/c = ½·W_chain / (1 + (1 − ρ)(c − 1)(√(4 + 5c) − 2)/(16ρc))（Cosmetatos），ρ = E[k]/B；c_s² > 1 仍为 Lee–Longton (1 + c_s²)/2。条件等待仍取指数分布（DES 中 PD 的 c² ≈ 0.6、合并 ≈ 1.6–2；按这两个值取 Gamma 的对拍更差：PD 绝对误差之和 61 → 81 个百分点，合并 CV1 p99 +24 … +37 %，所以不用）。
- **合并模式 prefill 期间占槽**：before_prefill 时请求从准入一直占槽到 decode 结束，槽位负载 a′ = a + λ_r·T_pre（T_pre = 未计等待的 TTFT 均值）。链只描述 decode，所以 P(等待) × C(B, a′)/C(B, a)，W × (C(B, a′)/C(B, a))·(B − a)/(B − a′)·a′/a（M/M/c Erlang C 之比）；a′ ≥ B 判不稳定。PD 的槽在 KV 拉取前几 ms 才占，忽略。
- **饱和抢占的回填偏移**：0.56 认为每次离开后贪心回填，余量 h ~ U(0, S_next)，下一次离开前增长 G ~ Exp(E[o]) 超过 h 就抢占。但离开的请求释放 S_d + o_d，回填只用掉约 S，余量其实按 h → h + o_d − G 游走，所以抢占概率 = P(G − o_d > h) = E_S[(E[o]/S)(1 − e^{−S/E[o]})] × E_o[e^{−o/E[o]}]（无记忆）。定长输出因子 e^{−1}，CV 1 约 0.47。
- **恢复不动点**：恢复占副本时间的比例 f = ν·T_stall，所有槽的占用 × 1/(1 − f)。在服务率 ÷ 1/(1 − f) 的链（`pi_at(slow)`；批到达链按 λ_b 缩放）上重算 P(等待) 与 W，ν_sat ∝ P(等待)，迭代到收敛；f ≥ 0.995 或链不稳定则判不稳定。等待对 f 很敏感（dense8b 12 GB CV1 L0.85 PD：f = 0 / 0.014 / 0.03 / 0.06 时链上等待 2.5 / 3.6 / 6.0 / 34 s）。重算停顿仍取 prefill(S̄ + ḡ)：按长度档平均（凸性）会高估 40 %，因为 DES 的被抢占者都很年轻，平均足迹只有 S̄ + ḡ 的 0.8 倍（DES 每次恢复 92 ms，闭式 92 ms）。
- **DES 修正**：恢复中的序列计入 KV 占用和槽位（`_Replica.restoring`）。之前 PD decode 副本在恢复期间（准入随 prefill 完成异步发生）把被抢占者的空间又分给新请求，恢复后批次溢出，刚恢复的请求（最年轻）被再次抢占：PD recompute CV1 有 27 % 的抢占是重复的，CV0 有 26 %。修正后分别为 11 % 和 9 %（单 seed × 8000）。0.56 记录的「连锁抢占（闭式 7.9 s vs DES 17.6 s）」主要是这个错误。
- **对拍**（dense8b 12 GB → B_kv = 17，before_prefill，load 0.85，4 seed × 8000 请求合并统计；闭式 / DES，0.56 闭式在同一 DES 上的误差写在括号里）：

| 策略 | 模式 | CV | TTFT p90 | TTFT p99 | 准入等待 | 抢占 / 请求 |
|---|---|---|---|---|---|---|
| wait | PD | 0 | 3922 / 4155（−6 %；0.56 +6 %） | 9514 / 8788（+8 %；+22 %） | 1155 / 1171（−1 %；+12 %） | — |
| wait | PD | 1 | 10012 / 9957（+1 %；+1 %） | 22540 / 18320（+23 %；+24 %） | 2785 / 2773（0 %；+1 %） | — |
| wait | prefill 优先 | 0 | 987 / 1259（−22 %；−21 %） | 3998 / 4486（−11 %；+12 %） | 261 / 309 | — |
| wait | 分块 | 0 | 1290 / 1367（−6 %；−21 %） | 4431 / 4595（−4 %；+12 %） | 312 / 323 | — |
| wait | prefill 优先 | 1 | 8533 / 6941（+23 %；0 %） | 21840 / 22860（−4 %；−15 %） | 2443 / 2160 | — |
| wait | 分块 | 1 | 10480 / 7472（+40 %；−5 %） | 24980 / 23930（+4 %；−18 %） | 3136 / 2341 | — |
| recompute | PD | 0 | 3575 / 4312（−17 %；−14 %） | 8896 / 10210（−13 %；−8 %） | 1024 / 1239 | 0.020 / 0.024（−16 %；+117 %） |
| recompute | PD | 1 | 11290 / 14300（−21 %；−40 %） | 25330 / 26780（−5 %；−26 %） | 3235 / 4365 | 0.062 / 0.076（−18 %；+39 %） |
| recompute | prefill 优先 | 1 | 8243 / 8659（−5 %；−32 %） | 21550 / 28520（−24 %；−38 %） | 2323 / 2721 | 0.048 / 0.050（−3 %；+47 %） |
| recompute | 分块 | 1 | 10360 / 8246（+26 %；−26 %） | 24990 / 26760（−7 %；−33 %） | 3061 / 2556 | 0.057 / 0.038（+48 %；+91 %） |
| swap | PD | 0 | 3269 / 3475（−6 %；+7 %） | 8270 / 8031（+3 %；+18 %） | 920 / 940 | 0.019 / 0.020（−4 %；+160 %） |
| swap | PD | 1 | 8780 / 8030（+9 %；+7 %） | 20290 / 15740（+29 %；+27 %） | 2307 / 2041 | 0.055 / 0.060（−8 %；+77 %） |
| swap | prefill 优先 | 1 | 7488 / 5860（+28 %；+1 %） | 20050 / 20780（−3 %；−15 %） | 2074 / 1758 | 0.046 / 0.044（+6 %；+69 %） |
| swap | 分块 | 1 | 9366 / 6320（+48 %；−3 %） | 23030 / 21560（+7 %；−17 %） | 2703 / 1906 | 0.054 / 0.034（+60 %；+117 %） |

  CV0 合并的 recompute / swap：TTFT p90 −29 … 0 %，p99 −19 … −4 %，抢占 −23 … +49 %。load 0.6 时都接近（TTFT −23 … +7 %，等待 ≈ 0）。合并 CV1 的 TTFT p50 偏高 +84 … +164 %：P(等待) ≈ 0.4–0.5，p50 正好落在等待原子的边上，属于病态点。0.56 文档里的「wait 折叠后 p90 +29 / +31 %、p99 最多 2×」主要来自 6 × 3000 的短跑 DES（槽位队列松弛很慢）；在 4 × 8000 上，0.56 的 PD p90 已经在 ±6 % 以内。
- **数值变化**（§18.1 例子，load 0.8，KV 12 GB；0.56 → 0.57，KV 关时不变）：

| 策略 | 模式 | TTFT p50 / p90 / p99 ms | TPOT 均值 / p90 ms | 最长间隔 ms |
|---|---|---|---|---|
| wait | PD | 121 / 1410 / 4872 → 121 / 1138 / 3869 | 8.56 / 12.28 不变 | 12.1 不变 |
| wait | prefill 优先 | 82 / 143 / 2845 → 83 / 167 / 2091 | 不变 | 不变 |
| wait | 分块 | 133 / 317 / 2956 → 137 / 386 / 2377 | 不变 | 不变 |
| recompute | PD | 114 / 1095 / 4374 → 114 / 906 / 3453 | 8.73 / 12.68 → 8.69 / 12.63 | 461.9 → 12.8 |
| recompute | prefill 优先 | 82 / 131 / 2493 → 82 / 143 / 1794 | 8.03 / 12.40 → 8.01 / 12.39 | 314.8 不变 |
| swap | PD | 114 / 1095 / 4374 → 114 / 871 / 3376 | 8.69 / 12.63 → 8.68 / 12.61 | 389.2 → 12.8 |

  抢占次数降到每请求 1 % 以下后，被抢占者不再进入最长间隔的 p99（p_v < 0.01），所以 PD 最长间隔回到 12.8 ms。SLO goodput 不变（413 / 410）。

- **V4 网格长跑**（item 3）：load 0.85 的 10 个点每个 seed 20000 请求（0.56：3000），其余 3000；3 seed；SLO 二分仍用 1500（所以 SLO goodput 一列与 0.56 完全相同）。总耗时 1041 s（8 进程；0.56：891 s）。存储的网格 `accel_dse/data/v4_serving.json` 已更新（行里记 `n_req`）。全部场景族合并，括号内为 0.56：

| 模式 | 指标 | 最小 | 中位 | 最大 | 平均 \|误差\| % | 落在容差内 | DES 噪声中位 |
|---|---|---|---|---|---|---|---|
| PD | TTFT p50 / p90 / p99 | −13 %（−18） / −6 %（−12） / −6 % | 0 % / 0 % / 0 % | +4 %（+2） / +6 %（+5） / +7 %（+30） | 3 / 2 / 3 | 100 %（93） / 100 % / 100 %（97） | 4 %（5） / 4 %（8） / 9 %（15） |
| PD | TPOT 均值 / p90 | −6 %（−2） / −7 %（−6） | 0 % / −1 % | +2 %（+5） / +2 %（+7） | 1 / 2 | 100 % / 100 % | 3 %（4） / 3 %（4） |
| PD | 最长间隔 | −23 % | −4 % | +11 %（+24） | 7 | 100 % | 6 %（9） |
| PD | SLO goodput | −1 % | +3 % | +16 % | 6 | 100 % | 6 % |
| 合并 prefill 优先 | TTFT p50 / p90 / p99 | −12 % / −11 % / −3 %（−4） | 0 % / 0 % / +1 % | +7 %（+5） / +8 % / +7 %（+5） | 2 / 2 / 2 | 100 % / 100 % / 100 % | 0 %（1） / 2 %（3） / 2 % |
| 合并 prefill 优先 | TPOT 均值 / p90 | −6 %（−5） / −15 %（−11） | −2 % / −6 % | +3 %（+4） / −1 %（+1） | 2 / 6 | 100 % / 100 % | 3 % / 4 %（5） |
| 合并 prefill 优先 | 最长间隔 | −26 % | −3 % | +8 % | 5 | 100 % | 4 %（6） |
| 合并 prefill 优先 | SLO goodput | −11 % | −1 % | +5 % | 4 | 100 % | 6 % |
| 合并分块 | TTFT p50 / p90 / p99 | −4 % / −9 % / −23 %（−15） | 0 % / −4 % / −4 % | +2 %（+3） / +1 % / 0 %（+4） | 1 / 4 / 6 | 100 % / 100 % / 100 % | 2 %（3） / 4 %（6） / 5 %（4） |
| 合并分块 | TPOT 均值 / p90 | −5 %（−4） / −12 %（−11） | −1 % / −4 % | +3 %（+6） / +1 %（+4） | 2 / 5 | 100 % / 100 % | 3 % / 4 %（5） |
| 合并分块 | 最长间隔 | −2 %（−7） | +4 % | +11 %（+22） | 4 | 100 % | 4 % |
| 合并分块 | SLO goodput | −5 % | −2 % | +2 % | 2 | 100 % | 6 % |

  load 0.85 各点的 DES 噪声（seed 间 sd / 均值，中位）：PD TTFT p50 / p90 / p99 10.7 / 14.1 / 22.1 % → 3.5 / 3.9 / 8.7 %，TPOT 均值 7.4 → 2.7 %，最长间隔 10.6 → 5.1 %；合并 prefill 优先 TTFT p90 4.5 → 2.0 %，TPOT p90 9.1 → 6.1 %；合并分块 TTFT p50 6.9 → 3.2 %，TTFT p99 9.5 → 9.3 %（CV0 时分块的 p99 在 20000 请求下仍有 12–26 % 噪声）。PD TTFT p99 最差 +30 % → +7 %，PD 最长间隔最差 +24 % → +11 %，分块最长间隔 +22 % → +11 %：这些都是 DES 短跑造成的。长跑后显出来的真实偏差：合并 prefill 优先 TPOT p90 −15 %（tp4_32b L0.85 CV1，噪声 4 %），分块 TPOT p90 −12 %（同点），分块 TTFT p99 −23 %（moe30b L0.85 CV0，噪声 26 %）/ −16 %（tp4，噪声 12 %）。
- **入口**：`kv_cap` 新增 `slot_hold_prefill_ms`、`refill_offset_factor`、`slot_slowdown`；`v4_grid(n_high=, high_load=)`、`scripts/v4_serving.py --n-high --high-load`。
- **剩余缺口**：合并 CV1 的 TTFT p90 偏保守（+23 … +48 %，prefill 期间占槽的 Erlang C 修正在 CV1 过强；不加此修正时 CV0 偏乐观 −21 %）；合并分块的抢占次数仍偏高约 +50 %；PD recompute CV1 TTFT p90 −21 %；条件等待形状（PD c² 0.6、合并 1.8）没有进模型；没有 radix / 部分前缀匹配（item 4 未做）。

### 18.8 合并模式占槽 = prefill 服务、条件等待形状、分块回填受限、radix 前缀树（0.58）

- **占槽时间**：合并 before_prefill 的槽位负载 a′ = run + λ_r·T_pre。0.57 取 T_pre = 未计 KV 等待的 TTFT 均值，0.58 改为减掉 prefill 排队均值 W_q,pre，即 T_pre = E[TTFT] − W_q,pre。DES（与 vLLM）在 prefill 开始的迭代边界分配 KV block，排队在准入之前，所以排队时间不占槽。CV1 时 W_q,pre 是 TTFT 的大头，0.57 把 a′ 放大太多，导致 CV1 TTFT p90 +23 … +48 %。
- **条件等待形状**：等待的请求的等待时间 W | W > 0 取 Gamma(k = 1/c², θ = m·c²)。合并模式 c² = 1 + 0.7·min(2, c_s²)，PD 仍为 c² = 1。DES 实测的条件 c²：prefill 优先 1.1 / 1.56（CV0 / CV1），分块 1.6 / 1.74，PD 0.6。PD 取 0.6 会让 p99 更乐观（PD wait p99 已经 +8 / +23 %，但 recompute 偏乐观），所以保留指数分布。TTFT 分位数在 (prefill 尾部, 等待) 的混合上计算，Gamma 尾部 Q(k, t/θ) 用正则化上不完全 gamma（级数 / 连分式，512 点表）。
- **分块回填受限**：饱和抢占率 p_ev 再乘 (1 − ρ_pre)。分块时，新请求的 prefill 与 decode 在同一批里按 chunk 推进，被释放的空间在 prefill 忙时是逐 chunk 回填的，不会一次填满。只有 prefill 空闲的那部分时间才会出现 0.57 模型里的「一次填满」。这是经验结构，不是推导：
  - 分块抢占次数误差从 +30 … +60 % 降到 −10 … +4 %；
  - prefill 优先模式不乘这个因子。
  `kv_cap.refill_tight_share`。
- **对拍**（dense8b 12 GB，L0.85，4 seed × 8000 / 30000 请求）：
  | 指标 | 0.57（8000） | 0.58（8000） | 0.58（30000 长跑） |
  |---|---|---|---|
  | wait 合并 CV1 TTFT p90（prefill 优先 / 分块） | +23 / +40 % | +14 / +21 % | +11 / +17 % |
  | recompute 合并 CV1 TTFT p90 | −5 / +26 % | −13 / +2 % | −18 / −1 % |
  | swap 合并 CV1 TTFT p90 | +28 / +48 % | +15 / +24 % | — |
  | 分块抢占次数 | +30 … +60 % | −10 … +4 % | −20 %（recompute CV1） |
  | PD recompute CV1 TTFT p90 / 等待 / 抢占 | −21 / −26 / −18 % | 不变 | −45 / −52 / −30 % |

  8000 请求的 DES 从空系统起步，L0.85 CV1 时还没到稳态：PD wait 的 DES 等待 2773 → 3392 ms（30000）。PD recompute 的差距随长度变大，这是剩余最大的缺口。试过用碎片化后的有效槽位数（B = 16）解释：W 4380 vs DES 6777，不够。
- **radix 前缀树**（`pd.prefix_tree = ((L_1, n_1, α_1), …)`，「假设」）：
  - **树与请求**：第 k 层每个节点有 n_k 个子节点，每个子节点 L_k token。请求在每层独立按 Zipf(α_k) 选子节点，第 k 层节点 v 的访问概率 p_v = Π p_{i_j}。
  - **缓存**：按 token 计容量 C 的 LRU。每次访问刷新整条路径（叶 → 根），所以根的最近访问时间总是不早于子节点，子节点驻留蕴含父节点驻留（树形包含），各层共享容量。
  - **Che 近似**：每个节点 h_v = 1 − e^{−p_v·T}，T 由 Σ_v L_v·h_v = C 解出。由于 p_parent ≥ p_child，包含关系在近似里也成立。
  - **匹配深度**：P(深度 ≥ k) = H_k = Σ_{v ∈ 层 k} p_v·h_v。深度律 P(d = k) = H_k − H_{k+1}，匹配 token = cum_k = Σ_{j ≤ k} L_j（上限 S − 1）。
  - **排队**：prefill 的每个深度单独作为一类请求进入排队混合，服务时间按 S − cum_k 计。
  - **PD decode 池**：decode 池独立缓存，两池都持有深度 ≥ k 的概率是 Σ p·h_p·h_d，KV 传输量按两池的最小深度扣减。
  - **容量**：C = 浮点 K × ΣL（亲和路由时 × 副本数）。
  - **组数控制**：组数超过 4096 时按 p 几何分箱（比值 1.01）。
  - **对拍**：只看缓存时，Che 与 RadixLRU 的各层命中最大差 0.004（3 棵树 × 4 个容量）。端到端用 dense8b，树 512×4 → 1536×200 → 1024×100，L0.6 / 0.85 × CV0 / 1 × 5 / 20 / 80 GB，3 seed × 3000：
    - token 命中率闭式 0.235 / 0.360 / 0.502，DES 高 0.003 … 0.004；
    - 各层命中差 ≤ 0.016（5 GB 根层 0.945 vs 0.96：Che 在容量很小时略低估）；
    - TTFT p90 −5 … +9 %，p99 −8 … +20 %；
    - PD L0.85 80 GB 的 TPOT 均值 / p90 +12 / +18 %，原因未查明。
- **剩余缺口**：
  - PD recompute CV1 长跑 −45 %；
  - 合并 CV1 TTFT p50 +60 … +92 %（P(等待) 低约 15 %、条件均值高约 15 %，两者在 W 上抵消）；
  - 合并 TPOT p90 在 tp4 L0.85 CV1 −15 %（未做）；
  - 前缀树不跨层相关（各层独立选子节点），也不建模 decode 生成的 token 回写进树（多轮对话把上一轮输出作为下一轮前缀）。

## 19. 拓扑感知集合通信（0.59，默认关）

`scenario.fabric.enabled = true` 时，每次集合通信按拓扑与算法计算（`core/fabric.py`），所有参数都是「假设」。关闭时走 §15 的 0.50 三层 α-β 模型，结果逐位不变：1356 项默认指纹，另有 985 项多节点 / D2D / PD 指纹，都与 0.58 相同。

**分层**（同 §15）：组内 g 个 rank 分成 D2D n₀ · scale-up n₁ · 网络 n₂ 三层。K_i = Π_{j<i} n_j 是一个第 i 层单元（封装 / 节点）内的组成员数，每个成员在第 i 层有一个端口。

**算法**（allreduce，每 rank 数据 B；与 NCCL tuner 的 ring / tree 模型一致：ring 算法带宽 = busBw·n/(2(n−1))，tree = busBw/2）：

| 算法 | 带宽时间 | 步数（× 每步时延） |
|---|---|---|
| ring（扁平，多通道） | 2(g−1)/g · B · maxᵢ 1/(K_i·β_i) | 2(g−1)；其中跨第 i 层的步数 2(U_i − 1) − Σ_{j>i}，U_i = Π_{j≥i} n_j |
| tree（NCCL：节点内 chain，跨节点双二叉树） | 2B · max(max_{i<top} 1/β_i, 1/(K_top·β_top)) | 节点内 2(K_top − 1) 步，跨节点 2⌈log₂ n₂⌉ 步（tree 每步时延） |
| tree（单节点 = chain） | 2B · maxᵢ 1/β_i | 2(g−1) |
| hier（逐层 reduce-scatter → 顶层 allreduce → 逐层 all-gather） | Σ_{i<top} 2(n_i−1)/n_i · B/K_i/β_i + 顶层 | Σ_{i<top} 2(n_i − 1) + 顶层 |

- **hier 的顶层**（数据 b = B/K_top）有三种选法：ring 2(n−1)/n·b/β；tree 2b/β（仅网络层）；网内归约 b/β（每 rank 只收发一次，时延按 2 × 交换机层数步）。
- **allgather**：ring (g−1)·B·maxᵢ 1/(K_i β_i)，g−1 步；hier 同 §15，步数 Σ(n_i − 1)；没有 tree。
- **all-to-all**：直接交换，各层同时按目的地分摊（§15 的份额），时延一步。
- **PP 交接**：所走层的 β，加一步时延。
- **α 与算法选择**：α = α_launch（所涉各层每次集合通信 α 的最大值）+ Σ 步数 × 每步时延。`algo = auto` 按「带宽时间 + α」取最小，即 NCCL 式的单次通信最优。它不考虑与计算的重叠：在 MAC 绑定的阶段，α 更小的 hier 可能实际更快（例子 B）。
- **每步时延默认值**：节点内 0.6 µs、网络 ring 2.7 µs、网络 tree 5.0 µs，取 NCCL tuner 默认 hwLatencies（LL 协议：NVLink ring / tree 0.6，NET ring 2.7、tree 5.0）作数量级参考；D2D 0.1 µs 为「假设」。它们不是任何系统的实测值。

**节点内 scale-up 拓扑**（`link.topology`）：β 是每卡注入带宽，按端口平分。domain = 每节点封装数（单节点时为整个副本）。

| 拓扑 | ring / chain / tree 类 | all-to-all |
|---|---|---|
| switch | 1 | 1 |
| full_mesh（每卡 n−1 条直连） | (n−1)/(k−1)：k 个成员之间只有 k−1 条可用 | (n−1)/(k−1) |
| ring（双向，2 端口） | 组绕满环 1，否则 2（链），× stride（并发组共享链路） | 最短路径下并发组的最大有向链路负载；k = n = 8 时为 16/7 |
| torus2d（X×Y，4 端口） | 4 / 可用端口数（成员绕满的维 2 个，只跨部分的维 1 个）× 沿该维的 stride | 维序路由的最大链路负载；4×4 为 32/15 |

**跨节点拓扑**（`fabric.net_topology`，上行收敛比 r = 下行 : 上行，spine 非阻塞）：
- **fat_tree**：一个 leaf 下 m 个整节点。
- **rail**：每节点第 j 张 NIC 接第 j 个 rail 交换机，每个 rail 交换机下 m 个节点。跨 rail 的流量先走节点内 scale-up（PXN）：all-to-all 网络字节中的 (k₂ − 1)/k₂ 也计入 scale-up 层。
- **默认 m**：由交换机端口数推出，下行端口 = radix·r/(1+r)；fat_tree 为 下行 // node_cards，rail 为 下行。radix 64 时 fat_tree 每 leaf 4 / 5 / 6 节点（r = 1 / 2 / 4，每节点 8 卡），rail 每交换机 32 节点。
- **收敛惩罚**：SPMD 下 leaf 内所有卡同时通信，所以每卡离开 leaf 的流量份额 f 以 β/r 走上行，网络层 β 除以 max(1, r·f)：
  - ring：f = 1/m（每个 leaf 只有一条环边出 leaf）；
  - tree：f = NCCL 二叉树（`ncclGetBtree`）里跨 leaf 的边 / (n₂ − 1)；
  - all-to-all：f = (n₂ − m)/(n₂ − 1)；
  - 网内归约：f = 0（在 leaf 内先聚合）；
  - PP 交接跨 leaf：r；
  - PD KV（随机配对，共 n 节点）：r·(1 − m/n)。

  这一项就是「EP all-to-all 走收敛网络」的争用因子。ring / 分层算法几乎不受收敛影响（r ≤ m 时无影响）；all-to-all 受全额影响。

**争用**：
- 同一流水级内的 TP / EP / PP 流量在 t_link 中逐项相加（0.50 起就是如此，相当于完全争用、不重叠）。
- 并发的组（DP 副本、EP 组）共享上行，按上面的 f 计。
- PD KV 传输与两池自身的集合通信共用 NIC：KV 可用 β × (1 − u_c)，u_c 为两池最忙流水级上该层的占用率（字节 / β / 级时间，上限 0.95）。另外报告集合通信被 KV 流量拖慢的倍数 1/(1 − u_kv)，但不回灌进 TPOT。

**网内归约**（`fabric.innet_reduce`，厂商选项「假设」）：
- net：跨节点交换机聚合（SHARP / CollNet 类）；
- net+link：另含交换式 scale-up 上的聚合（NVLS 类，只在 topology = switch 时生效）。
- 只用于 allreduce 的 hier 顶层。

**手算对照**（`tests/test_core_059.py`）：单节点 8 卡 ring = 2·7/8·B/β、14 步；2 节点 × 8 的 ring、tree、hier 三种公式；NCCL `ncclGetBtree` 的父节点表（n = 8：−1, 2, 4, 2, 0, 6, 4, 6）；EP all-to-all 4 节点、每 leaf 1 节点时网络时间 × r；rail PXN；full mesh 上 TP2 为 7×；SHARP 的 b/β。

**例子**（1P + HBM3e，8 卡 / 节点，网络 50 GB/s / 卡）：
- **A. DeepSeek-V3 DP64·EP64（8 节点）**（0.62.1 按当前代码重算：0.61.1 起 fp8 激活发布的 MoE combine / all-reduce 按 bf16 线上字节计，§19.10，all-to-all 字节变多；0.59 原文数值附在括号里）：
  - prefill b64：拓扑模型关时步 715.9 ms（LINK 绑定，link 715.3 ms；0.59：707.7 ms，MAC 绑定，link 476.9 ms）。
  - 默认推导的 leaf（radix 64，r = 1 / 2 / 4 → 每 leaf 4 / 5 / 6 节点）：716.2 / 716.2 / 818.3 ms（0.59：r = 4 时 link 545 ms、步不变）。
  - 每 leaf 2 节点：r = 2 → 1227 ms，r = 4 → 2453 ms（3.4×；0.59：818 / 1636 ms）。每 leaf 1 节点：1431 / 2862 ms（0.59：955 / 1909 ms）。
  - rail 用默认推导的规模（每交换机 32+ 节点）时 8 节点全在一个 rail 域内，不受收敛影响（716.2 ms）。
  - decode b512（7.41 ms，MAC 绑定）：link 1.40 → 5.59 ms（每 leaf 1 节点、r = 4；0.59：0.93 → 3.73 ms），仍被计算掩盖（步 7.73 ms，增量来自每步时延）。
- **B. qwen3-32b TP16（2 节点）decode b64**：拓扑模型关 20.84 ms；ring 22.92、tree 22.43、hier 21.85 ms，auto 取 hier。增加的部分来自每步时延（ring 30 步）。
- **C. 单节点 8 卡 TP2·PP4 prefill 的 link 时间**：switch 6.7 ms；ring 13.4（链，2×）；torus 4×2 26.8（行内部分跨度，4×）；full mesh 47.0（7×）。都被计算掩盖。TP8：torus 4×2 为 4/3×，其余为 1×。
- **D. TP16 跨 2 节点 prefill，每 leaf 1 节点、r = 4**：link 617 ms；开网内归约（hier + SHARP 顶层）295 ms。r = 1 时 ring（201 ms）比 hier + SHARP 好。
- **E. PD KV（qwen3-30b-a3b，32 + 32 卡，4 节点 × 16 卡）**：每 leaf 1 节点时，r = 2 / 4 → KV 4.0 → 6.0 / 12.1 ms（β 50 → 33 / 17 GB/s）。KV 不是瓶颈，goodput 不变。

**剩余缺口**（0.59 时；标 ✓ 的在 0.60 补上，见 §19.2–19.4）：
- ✓ auto 不考虑与计算重叠（0.60 `auto_overlap`）；
- ✓ 不同层的流量在 t_link 中相加（0.60 `overlap = ports`）；
- ✓ ring / torus 上的 PP 交接不计多跳（0.60）；
- 树的切边只用第一棵树；
- 不建模协议（LL / LL128 / Simple）、通道数上限和小消息带宽折损（NCCL 的 treeCorrectionFactor / 平台期）；
- ✓ KV 对集合通信的拖慢只报告、不回灌（0.60 `kv_feedback`，只回灌 decode 池）；
- ✓ spine 视为非阻塞（0.60 三层 fat-tree）；
- 拥塞控制、ECMP 哈希冲突、incast 未建模；
- ✓ 网内归约只用于 allreduce（0.61：节点内交换式 scale-up 上的 allgather / reduce-scatter 走 NVLS 类，§19.7）。

### 19.1 大规模卡数（0.60）

- **上限**：每副本卡数上限 64 → 8192（API / UI / CLI；`core.parallel.MAX_REPLICA_CARDS`），PD 池卡数上限 4096 → 65536。`node_cards` ≤ 4096、`fabric.pod_nodes` ≤ 65536（须为 `leaf_nodes` 的整数倍），非法输入返回 400。
- **batch 上限随卡数放大**：布局搜索 / 最佳 batch 的每副本 batch 上限 = max(4096, 64 × 卡数)。原来固定 4096 = 64 × 64；DP1024 时只够每个 DP rank 4 条序列，会截断最优点。≤ 64 卡的结果不变。
- **布局搜索**：分支定界照旧（`search_layouts(top=k)`：先算每个布局的上界排序，再以第 k 名精确分数剪枝）。MoE 布局数 64 / 256 / 1024 / 4096 卡为 139 / 271 / 451 / ~700，dense ≤ 7（PP·TP = 卡数，PP ≤ 层数）。单次评估约 1–2 ms（层级结果有缓存，与卡数基本无关）。
- **EP > 专家数**（如 DeepSeek-V3 256 专家、EP1024）：每 rank 1 个专家，token-专家对按 EP 均分，相当于冗余专家副本均匀分担「假设」（权重存储按 ⌈E/EP⌉ 计）。
- **PD 布局搜索**（`pd.search_layouts`）：总卡数 > 64 时副本尺寸加入 128、256、… ≤ 总卡数，每种尺寸配额相同（同尺寸内 PP / TP 小者优先），上限仍为 64 个候选。≤ 64 卡时完全同 0.52。
- **PD 排队模型**：
  - decode 生灭链状态上限 4096 → max(4096, B + 4096)，原来 B ≥ 4096 时会截断；
  - B > 1024 时 step(k) 在 k ≤ 64 精确，之上按几何网格（每倍频 8 点）线性插值「假设」，相对精确链的平均 TPOT 误差 < 0.5%（测试）；
  - SLO 切分搜索在候选 > 32 个时先取 16 个均匀切分，再在最优点两侧逐个计算（单峰「假设」）。测试中与穷举结果一致；1024 + 1024 卡、8 卡副本时 80 s → 15 s。
  - 修正：λ 极大时 c = 1 − e^{−λ·lat} 舍入为 1 导致除零（大卡数的稳定速率二分会触发），现判为不稳定。
- **运行时间**（本机单进程，1P + HBM3e，8 卡 / 节点，fabric 开）：见 CHANGELOG [0.60.0] 表。单点评估 / 扫描在 4096 卡下仍为毫秒级；布局排名 1024 卡约 8 s，4096 卡约 15 s；稳定性分析（约 10 个扰动）1024 卡约 40 s。

### 19.2 三层 fat-tree（leaf / spine / core，0.60）

`fabric.net_tiers = 3`（只用于 fat_tree；rail 保持两层）：
- 每个 pod `pod_nodes` 个节点。0 = 每 leaf 节点数 × spine 下行端口数，下行端口 = radix·r₂/(1 + r₂)（radix 64、r₂ = 1 → 32 个 leaf / pod）。
- leaf 上行收敛 r₁（`oversub`），spine → core 上行再收敛 r₂（`oversub_spine`）：一张卡在 core 层只分到 β/(r₁·r₂)。
- 一个模式的每卡流量中，离开 leaf 的份额 f_leaf、离开 pod 的份额 f_pod（同 §19 的定义，m 换成组内同 pod 的节点数 p）：

  **网络层 β 除数 = max(1, r₁·f_leaf, r₁·r₂·f_pod)**

  - ring：f = 1/m、1/p；
  - tree：按 NCCL btree 在 leaf / pod 边界的切边；
  - all-to-all：(n − m)/(n − 1)、(n − p)/(n − 1)；
  - 网内归约 0（时延按 3 层交换机计 2·3 步）。
- r₂ = 1 时与两层逐位相同（f_pod ≤ f_leaf）。PP 交接跨 pod 为 r₁·r₂、跨 leaf 为 r₁。PD KV 为 max(r₁(1 − m/n), r₁r₂(1 − P/n))。
- **流量分层**：报告 `fabric.net_traffic` 给出每步跨节点 MB，以及离开 leaf / 离开 pod 的字节份额；每行集合通信给出自己的 f_leaf / f_pod。
- 每步时延不随交换机层数增加（`hop_net_us` 一个值）——0.61 起可加 `hop_spine_us` / `hop_core_us`（§19.6）。

### 19.3 端口并发与暴露时间最小的算法选择（0.60）

**`fabric.overlap = ports`**：
- 一张卡的 D2D PHY、scale-up 端口、NIC 是不同的物理端口，可同时工作；同一端口上的流量串行。
- 每次集合通信给出各层的忙时：
  - ring / tree / all-to-all 是单阶段，各层同时工作，带宽时间 = 最慢层；
  - hier 是逐层阶段，带宽时间 = 各阶段之和，各层忙时 = 本阶段时间；
  - PP / FSDP 计入其所在层。
- 级链路时间 = max(maxₜ Σ 忙时ₜ, 最长的单次集合通信带宽时间)，不超过 sum 模式（0.59 的 Σ 带宽时间），也不低于最忙端口。
- 这是稳态资源模型，与级时间 max(计算, DRAM, SLC, 链路) + α 的重叠假设一致：跨层 / 跨微批流水，依赖关系不显式建模「假设」。

**`fabric.algo = auto_overlap`**：
- `auto` 对每次集合通信单独取 min(带宽 + α)。auto_overlap 则在每个流水级上，对该级各类集合通信（kind, 组, stride, 数据量）的算法组合直接最小化级时间：

  max(窗口, 链路(组合)) + Σ α(组合)，窗口 = max(计算, DRAM, SLC)

  带宽时间被计算掩盖时，偏向 α 小的算法。
- 组合数 ≤ 729 时穷举，否则从 auto 的选择出发做 3 轮坐标下降。按构造不劣于 auto 或任何强制算法（测试覆盖两种 overlap 模式）。
- 报告中「所选算法」为级选择的结果，`fabric.stages` 给出每级的 sum / ports 链路时间、各层忙时、窗口、α。

### 19.4 PP 多跳与 KV 回馈（0.60）

- **PP 交接在 ring / torus / full-mesh scale-up 上**：fabric 开且交接走节点内层时，一个流水级的每个封装 a 同时发给 a + S（S = 每级封装数；不绕回，偏保守「假设」）：
  - ring：最短路径路由，β/2 端口；
  - torus2d：维序路由，β/4 端口；
  - β 除数 = 最大链路负载 × 端口数（ring 8 封装、S = 2 → 4；torus 4×2、S = 4 → 4）；
  - full mesh 每对一条 β/(n − 1) 的直连 → n − 1（不做多路径转发「假设」）；
  - switch 不变。
  - 这是 0.59 → 0.60 在 fabric 开时唯一的默认数值变化（只影响 PP > 1 且 `link.topology` ≠ switch 的场景）。
- **KV 回馈**（`fabric.kv_feedback`，PD，默认关）：
  - u_kv = λ·kv / (min(n_p, n_d)·β)，与 0.59 的 `coll_slowdown` 同一定义，是 KV 流占 NIC 的份额。decode 池的网络带宽 × (1 − u_kv)，重算 decode，λ 随之下降；不动点迭代 ≤ 6 次（|Δu| < 1e-4 停止）。
  - prefill 池的发送侧不回灌（缺口）。
  - 报告 `kv.fabric.feedback`：u_kv、回馈前后 TPOT、decode 池有效网络 GB/s。
  - 例：DeepSeek-V3 PD 960P + 64D（两池 DP64·EP64、batch 512、out 128），网络 12.5 GB/s，每 leaf 1 节点、r = 4：u_kv 3.0%，TPOT 23.25 → 23.94 ms（0.62.1 重算；0.60 原文 u_kv 4.5%、15.80 → 16.50 ms——0.61.1 起 combine 按 bf16，decode 在 12.5 GB/s 网络上变为链路绑定、λ 更低）；decode 被计算绑定时 TPOT 不变。

### 19.5 例子与剩余缺口（0.60）

**例子**（1P + HBM3e，8 卡 / 节点，网络 50 GB/s / 卡，leaf 由 radix 64 推导；0.62.1 按当前代码重算 Kimi-K2 / DeepSeek-V3 两项——0.61.1 起 fp8 激活发布的 combine / all-reduce 按 bf16 线上字节，§19.10；0.60 原文数值见 CHANGELOG [0.60.0]）：
- **Kimi-K2 DP1024·EP1024（1024 卡，128 节点）**：
  - decode b8192：关 5.74 ms；两层 r = 1 / 2 → 6.07 ms（link 1.64 / 3.17 ms 被计算 5.14 ms 掩盖），r = 4 → 7.22 ms（link 6.30 ms，LINK 绑定）；三层 pod 32 节点：r₁/r₂ = 2/2 仍为 6.07（link 4.95），2/4 → 10.83 ms（LINK 绑定，−47% tok/s/卡）。（0.60：两层 r = 1 / 2 / 4 均为 6.07 ms，三层 2/4 → 7.53 ms、−19%。）（0.63：MLA decode 改为逐头吸收后计算变慢，关 fabric 6.93 ms，两层 r = 1 / 2 / 4 均为 7.26 ms，MAC 受限，link 6.30 ms 被掩盖；§22.1。）
  - prefill b1024：关 840 ms；两层 r = 2 / 4 → 1626 / 3225 ms；三层 2/2 → 2538 ms，2/4 → 5075 ms（0.60：560 / 1085 / 2151 / 1693 / 3384 ms）。76% 的跨节点字节离开 pod。
- **DeepSeek-V3 DP256·EP256（256 卡，32 节点）**：
  - 三层 pod 32 = 一个 pod，与两层相同。pod 16 节点、r₁/r₂ = 2/4（对照两层 r = 2）：prefill b256 1380 → 3271 ms，decode b2048 link 2.69 → 6.39 ms，超过计算 5.85 ms，变为 LINK 绑定（7.28 ms）。（0.60：921 → 2181 ms，link 1.80 → 4.26 ms，被计算掩盖。）
  - PP4·DP256（1024 卡）：每级 256 卡，pod 32 时 a2a 不出 pod。
- **端口并发**：DeepSeek-V3 TP8·DP32·EP256 decode（两层 r = 2）链路 3.18 → 2.69 ms（ports；0.60：2.04 → 1.80 ms）。TP16 跨节点 prefill（pod 16，2/4）：ports + auto_overlap 616.5 → 584.1 ms，allreduce 由 ring 改选 hier。
- **auto_overlap**：TP32·DP8 decode 时 allreduce 由 tree（bw + α 最小）改为 hier（α 4.99 → 4.52 ms），TPOT 89.64 → 89.17 ms。

**剩余缺口**：
- 每步时延不随交换机层数变化；
- ECMP 冲突 / 拥塞控制 / incast 未建模；
- 端口模型是稳态资源模型（不模拟集合通信间的依赖与时序）；
- auto_overlap 的窗口按级整体计（不按层内算子时序）；
- KV 回馈不回灌 prefill 池；
- 4096 卡以上的布局搜索与稳定性分析为数十秒级（未并行化）；
- PD 排队模型在 B > 1024 时用插值（误差 < 0.5%）；
- 大规模下 PD 排队报告仍可达 15–25 s。

### 19.6 性能（0.61）

全部为纯 Python、零依赖；除特别注明外与 0.60 逐位一致（1356 + 985 + 3150 项指纹 0 diff，PD 报告全 JSON 对照 0 diff）。
- **PD 排队链**（`core/pdqueue.py`）：
  - `_pi`（泊松到达）的截断判据原为每步 `max(logp)`，O(N²)；改为滚动最大值（逐位相同）。
  - `_occ_tau`：泊松到达（L = 1）时耦合和为空，直接 d = G/(π·μ)（逐位相同）；批到达且 B ≥ 256、N·L > 2·10⁴ 时，两个 O(L) 内积改为对切片的 `math.sumprod`（C 级、手工向量化；舍入差 ~1e-16），`_pi` 的批到达卷积同。
  - B ≥ 256 时，链越过众数且低于峰值 1e-20（泊松：e^−46）就停止，不再必走到 n > B (+ L)：丢弃的质量 < 1e-17（与原 e^−40 截断同一量级）。B ≤ 1024 时每个状态 k 都要一次评估，此项把轻载链从 ~B 个评估降到众数附近。
  - 返回前的 4 次逐 k 扫描改为一次步长表。
  - B < 256 的链保持 0.60 的循环（指纹的 PD 场景 B = 32 逐位相同）。
- **评估**（`core/evaluate.py`）：每个算子的代价（键 = 算子、芯片、映射、模型格式）与每次 fabric 集合通信（键 = System、kind、字节、组、stride）做记忆化。搜索中同样的每 rank 形状反复出现（例如倍增点上 B/dp 相同）：1024 卡布局排名中 `_op_seconds` 命中 29×、集合通信 9×。fabric_report 记录集合通信时不用集合通信缓存。非 head 级的空尾部算子不再求和。
- **布局搜索**（`core/search.py`）：给出 `pool` 时（HTTP 服务的 spawn 进程池，默认 min(8, CPU) 个 worker），各候选布局的指数 b_max 区间（排序上界，约 2/3 的评估）在 worker 中计算，只回传 (fits, slo_ok, tokens/step, step, throughput)；行的 Result 在主进程重算。排名、剪枝计数、评估数与串行完全相同（测试）。
- **稳定性**：扰动用例分发到进程池（原来整个请求占一个 worker）。
- 剪枝支配候选未做（布局间没有可证明的支配关系；分支定界已按精确上界剪掉 ~96% 的候选）。

### 19.7 交换机层数与每步时延（0.61，「假设」，默认 0）

- `fabric.hop_spine_us`：一步网络通信若离开 leaf（leaf → spine → leaf，多两级交换），在 `hop_net_us` / `hop_net_tree_us` 上额外增加的时延；`hop_core_us`：再离开 pod（三层）时的额外时延。
- 设组内 n 个节点、每 leaf m 个、每 pod p 个（与 §19.2 同，按 stride 折算）：
  - ring / 平铺步（ring 的网络步、hier 顶层 ring、allgather、all-to-all 的一跳、p2p 类）：每步 + hop_spine·[n > m] + hop_core·[n > p]（锁步，任一对跨 leaf 即整步跨 leaf）；
  - 二叉树：第 j 层连接相距 2^j 的 rank，2^j ≥ m 时离开 leaf → max(0, ⌈log₂n⌉ − ⌈log₂m⌉) 层付 hop_spine，pod 同理；一次 allreduce 上下各一遍；
  - 网内归约：层级 2 / 3 时往返各加一次 hop_spine / hop_core；
  - PP 交接：两张卡的节点不在同一 leaf / pod 时加；
  - PD KV：随机配对，期望 (1 − m/n)·hop_spine + (1 − P/n)·hop_core。
- 手算：32 节点、每 leaf 2、每 pod 8、hop_spine 1 µs、hop_core 2 µs：ring allreduce 62 个网络步 → +186 µs；tree 5 层中 4 层出 leaf、2 层出 pod → 2 × 8 = +16 µs；all-to-all +3 µs。默认 0 = 0.60。

### 19.8 NVLS 类 allgather / reduce-scatter 与 NCCL 第二棵树（0.61）

- `innet_reduce = net+link` 且 scale-up 为 switch 时，allgather / reduce-scatter 多一个候选 `innet`（厂商选项「假设」）：
  - 该层一次交换机往返（2 步）代替 n − 1 个 ring 步；
  - 字节不变：allgather 多播后每 rank 仍收 (n − 1) 份；reduce-scatter 交换机内归约时每 rank 发 n 份（ring 为 n − 1 份）；
  - 其他层（D2D、网络）保持分层 ring。SHARP 类的网络层归约仍只用于 allreduce。
  - 当前推理图里只有 TP 的 logits allgather 会用到（reduce-scatter 只在 `fabric.collective("reducescatter", …)` 中提供）。例：DeepSeek-V3 TP8·DP128·EP1024 decode，logits allgather 7.2 → 4.2 µs/步，约 −0.05%（NVLS allreduce 是 0.59 起就有的，那部分 −13%）。
- **第二棵树**：NCCL 双二叉树的带宽 2B/β（每棵树一半数据）0.59 起就是这样计的；0.61 修正的是跨 leaf 切边份额：原来只数第一棵树，现在两棵（第二棵 = n 偶数时镜像、n 奇数时平移一位，即 ncclGetDtree）各一半。m | n 且 n 为偶数时镜像把 leaf 映到 leaf，值不变（3150 项 fabric 指纹因此 0 diff）；例 n = 5、m = 2：3/4 → (3 + 2)/2/4 = 0.625。

### 19.9 传输协议（0.61，「假设」，默认 off）

- `fabric.protocol`：off（0.60：LL 的每步时延 + 满带宽，偏乐观）| auto（每次集合通信取最快）| LL | LL128 | Simple。
- 带宽效率：LL ½（每 8 B 含 4 B 标志），LL128 120/128，Simple 1；bw 与各层忙时 ÷ 效率。
- 每步时延 × NCCL tuner hwLat 表的协议比例（hop_* 视为 LL 值）：scale-up / D2D 1 : 1.9/0.6 : 3.4/0.6，网络 ring 1 : 4.0/2.7 : 14/2.7，网络 tree 1 : 8.5/5.0 : 14/5.0。这些是开源 NCCL 的默认表，不是任何系统的实测。
- 候选名带协议后缀（`ring/LL128`）；强制算法时取该算法下最快的协议；PP p2p 同样取最快协议。
- auto 只会让结果变慢或不变（off 是两者最好的组合）。例：DeepSeek-V3 1024 卡 decode +2–3%（all-to-all 选 LL128），prefill +0.02–3%（大消息 Simple）。

**0.61 剩余缺口**：ECMP / 拥塞 / incast；协议通道数（nChannels）与每通道带宽上限、LL 的 llMaxBw 未建模；reduce-scatter 未出现在推理算子图中；KV 回馈只回灌 decode 池；PD + 布局搜索在 B ≤ 1024 时每个状态仍是一次精确评估（256 卡约 16 s）；映射对比在服务中按映射并行，单个映射内部串行。

### 19.10 第一轮审计修正（0.61.1）

逐项手算复核后修正（默认指纹里只有下列模型变化；fits / 瓶颈无翻转）：
- **滑窗 / top-k / 压缩注意力的 prefill**：
  - 原先每个 query 都按上限 c = min(窗口 | top-k | ⌈ctx/r⌉ + 窗口, ctx) 个 key 计，但前面的位置看到的 key 更少。
  - 现在按精确平均计：Σ_p min(p, 上限(p)) / q，p = ctx+1 … ctx+q（`AttnCore.keys_sum`，闭式，与逐位置求和一致）。
  - 例：DSA top-k 2048、prompt 4096 → 1536.25 个 key / query（原 2048）；prompt ≤ 2048 时原来是 2×。
- **DSA / CSA 索引器的 prefill**：位置 p 的 query 只给 ⌈p/r⌉ 个（压缩）key 打分，按精确平均计（原 top-k 与压缩层按整方阵计，2×）。decode 仍是全部 key。
- **压缩层 decode 的 KV 读**：所读条目数（⌈ctx/r⌉ + 窗口，≤ top-k + 窗口）每条都是完整的 latent 条目，原来又除了一次 r（DeepSeek-V4：r = 4 层少 4×，r = 128 层少 128×）。现在与 memplan 的存储一致。
- **纯滑窗 latent 层**（压缩比 1）只存 min(ctx, 窗口) 条 KV（原存全部 ctx；计算与读本来就按窗口）。有前缀缓存的 prefill 中，纯滑窗层只读前缀的最后一个窗口。
- **线性注意力的 conv 状态**：memplan 已计入存储，但每步读写没有计；现在 state 读写 = 2 ×（递归状态 + conv 状态），Qwen3-Next 每层 +4.7%。
- **归约 / 残差流的线上字节 ≥ bf16**「假设」（DeepSeek-V3 报告 §3.3：dispatch FP8、combine BF16；vLLM / SGLang 的行并行 all-reduce 为 bf16）：
  - fp8 激活的发布（DeepSeek-V3/V3.2/V4、Kimi-K2、GLM-5.3、Qwen3-FP8 …）的 TP / ETP / 共享专家 all-reduce、MoE combine、PP 交接，原来按 1 B / 元素计，现在按 2 B；dispatch 仍按 fp8。
  - 链路被计算 / DRAM 隐藏时步时不变，只有链路受限的点变慢（fabric 指纹 1430 项中 20 项：多节点 prefill +2.3 … +26.9 %）。
- **输入校验**：serving.batch / prompt / ctx / out_len / microbatches 上限 2²⁴，并拒绝布尔值（原来 2⁶² 也能通过，true 被当作 1）。

### 19.11 第二轮审计修正（0.61.2）

独立计数对照（torch `FlopCounterMode`，meta 张量，按发布 config 构造的 diffusers / transformers 模块，不下载权重）与小规模精确算例：
- **LTX-Video VAE 解码**：diffusers `LTXVideoUpBlock3d` 的顺序是 conv_in → 上采样 → resnet，resnet 在块的输出分辨率上跑；原来按输入分辨率计，解码少 2.5–3.3×。
  - 默认 161 帧 × 512 × 704：15.4 → 50.5 TFLOP（diffusers 计数 50.52），0.72 → 2.66 s，激活峰值 1.19 → 1.73 GiB；分块 18.8 → 62.0 TFLOP（diffusers 62.01）。
  - 整段（100T + HBM3E，batch 1）50.3 → 52.2 s。
- **ESMFold 主干遍数**：esm / transformers 参考实现在 `num_recycles=None` 时循环 `max_recycles` = 4 次（含首遍；只有显式传 k 时才是 k + 1 次），原来按 5 遍计。
  - 384 残基 62.2 → 50.2 TFLOP（transformers 计数 50.24），512 残基 120.9 → 97.3 TFLOP（97.32）；批延迟 −20 %；`validate` 的 ESMFold 行 0.28 → 0.23（仍在带内）。
- **decode 占用相关时间 τ_int**（TPOT 分位数的窗口因子）：Poisson 方程的 G(n) 从上往下求和，在均值以下抵消成舍入噪声，再除以左尾 π（平均占用 256 时 ≈ e⁻²⁵⁶），τ 可到 1e80。
  - 现在均值以下用 Σπ(k − Ek) = 0 改为从下往上累加（同号求和）。M/M/∞ 的 τ = 1/μ 在任意负载下都精确；B = 300、256 Erlang 与有理数精确值 25.712311352 一致。
  - 平均占用 ≳ 40 时才有影响：PD 默认例 TPOT p90 / p99 变化 ≤ 0.7 %。
- **radix 前缀缓存**：比整个缓存还长的节点不可能驻留，它的子树也不可能（radix 子节点需要父节点）。
  - Che 原来给这些节点 h > 0 并占用容量：例如 [100 tok × 10, 5000 tok × 10] 放进 3000 tok 时 H₁ 算成 0.10，实际为 1。
  - DES 的 radix LRU 原来也会存下够不到的子节点，白占容量。现在两者都在第一个放不下的层截断，Che 与 DES 一致。
- **面积预算**：缺某项密度时，已知项之和已经超过上限就判为超限（与功耗下界、缺固定面积时的规则一致），原来一律记为「无法判断」。
- **界面**：搜索 / 扫描 / Pareto / 对比在途时改了场景，结果到达后标「场景已改变，需重新运行」。表头和单位按发出请求时的模型渲染；原来按当前模型渲染，切到视频模型后，LLM 扫描显示成 帧/s、延迟 0 s。

复核无误（不改数值）：
- **排队公式**：M/D/1（Lindley 模拟）、两点 M/G/1、Erlang C 小例，birth–death 的 E[N] 与 M/M/B 精确值一致。
- **DES 不变量**：顺序、token 数、Little 律，prefill 利用率与 λ·E[S] 相符。
- **Che**：整前缀 LRU 与 radix 树对 LRU 模拟误差 ≤ 0.5 %。节点只有容量 1–2 倍大时，Che 本身的近似误差可达 30–80 %（大对象），属于模型局限。
- **VAE 解码 FLOPs 对 diffusers 计数**：Wan 639.22（分块 1042.27）、CogVideoX 315.03（分块 441.04）、HunyuanVideo 4770 vs 4766（因果掩码按一半计）、MiniMax-H3 视频 1892.96，均吻合。
  - MiniMax-H3 音频 −2.9 %：BigVGAN 的反走样滤波卷积按向量计。
  - Mochi：按 genmo 参考实现逐块裁帧（1053）；diffusers 末尾一次裁帧为 1081（+2.6 %）。
- **其他**：
  - 能耗动作计数：MAC × 2 = 每请求 FLOPs，跨 TP / PP 不变。
  - Wan2.2 待机专家：+14.29 B fp32，PP 均分，SP 复制。
  - MoE 倾斜边界（tokens = 1、EP > 专家数）。
  - HBM 几何（堆数 × 层数 × die Gb / 8；1024 / 2048 bit × 速率）对 JESD238 / JESD270-4。
  - 非默认映射（OS / WS 边 / 宽边加载、GEMV、可重构）的周期公式手推一致。
  - 视频 / 蛋白 PP × SP × batch × 微批 625 点：每请求 FLOPs 不变（SP 补齐 ≤ 0.02 %）。
  - HF 现网可达的发布 config（61 个根 config 及 diffusers transformer / VAE、Wan2.2 高低噪声子 config）逐项一致。

### 19.12 第三轮审计修正（0.61.3）

结构模型端到端 FLOPs 对独立计数（torch `FlopCounterMode`，meta 张量；OpenFold main、AF2 = OpenFold model_1 预设、boltz 0.4.1、protenix 0.5.0 的 PyPI 包、Open-Sora v1.2.0 源码 + SDXL VAE config），工具只计 GEMM / 注意力 MAC：
- **AF2 / OpenFold 模板扭转角行**：参考实现把 T 条模板扭转角行拼到 MSA 上，所有 Evoformer 块的 MSA 网格是 聚类行 + T（`workload.tmpl_msa`，AF2 / OpenFold 默认开）；扭转角嵌入按 T × N 行；OpenFold extra-MSA 全局列注意力的 q 来自行均值（N 行）。384 残基：AF2 245.69（参考 245.74）、OpenFold 221.41（参考 221.44）TFLOP；默认 512 残基 396 → 398 / 354 → 355 TFLOP。
- **Boltz-1 推理缓存**：`boltz predict` 用 `use_inference_model_cache=True`，pair 条件、每层 pair 偏置投影与原子编码器的 c / p 特征只在第一步算，现在放进一个 `diff_cache` 层（每请求一次）。扩散每步约少 28 %；384 残基每请求 173.0 → 153.1 TFLOP；默认 296 → 262 TFLOP，批延迟（100T + HBM3E）13.05 → 7.94 s。剩余差异：每步约 −3 %（k / v 按键窗口、稠密 gather），第一步的 z_to_p einsum 约 19 GFLOP 未计。
- **Boltz-1 检查点别名**：`output_projection_linear` 与 `output_projection.0` 是同一张量（按存储键 + 偏移识别），原来参数与 GEMM 都算两次（14.37 M 参数、每层每步 10.87 GFLOP）；另有 9 个 Lightning 回调标量。参数 606.38 M → 592.01 M。
- **Protenix 按样本的 pair 偏置**：v0.5.0 把 z_pair 沿 N_sample 展开，扩散 transformer 的 pair 偏置投影每个样本算一次（`spair` 行）；原子编码器的 cl / cm / z 投影按原子 / pair 行（原来按原子 pair 窗口，多计）。扩散 transformer 与参考逐项一致（5 样本 384 残基每步 887.85 GFLOP）。默认 419 → 434 TFLOP；这些投影输出只有 16 列（头数），在 os 映射上利用率很低，批延迟 19.33 → 27.86 s（把投影缓存起来的实现约 15.9 s——参考实现没有缓存，按参考计）。
- **Open-Sora 时间 VAE**：`conv_blocks[i-1]` 在第 i 块上采样前的分辨率上跑（原来按上采样后），时间解码每块 37.5 → 34.47 TFLOP；VAE 1158 → 1140 TFLOP，39.9 → 39.6 s。SD-VAE 空间部分（9.144 / 帧）原本一致。

其他修正：
- **冷存储不驻留**：按行查表的 embedding 表、Wan2.2 待机专家、不用的 io 表原来与路由专家一起排进 SRAM / SLC 驻留，白占容量（查表行本来就走 DRAM）。现在单独记为冷存储，只计容量。默认芯片（SRAM 小于热权重）结果不变；SLC 例（§14）pin 16 GiB 20.8 → 20.0 ms。驻留率（SRAM 驻留、SLC 驻留）的分母改为每步读取的权重（热 + 专家），不含冷存储，所以显示值略升（例 Qwen3-8B 100T：分母去掉 1.24 GB embedding 表）。
- **PD 能耗计入抢占恢复**：`kv_policy` recompute 每次抢占重算 S̄ + ḡ 的 prefill，swap 换出 / 换入各一次 KV 的 DRAM 读写；原来能耗里没有。主机链路（PCIe）没有对应的能耗项，未计「假设」。
- **预算范围**：PD 开启时预算只核对合并部署这一配置，响应 `budget.scope_note` 写明 PD prefill 池未核对。
- **最佳 batch 无解**：没有既放得下又满足 SLO 的 batch 时给出警告（原来静默按输入 batch 评估）；CLI decode 行显示 TPOT SLO 是否满足。
- **场景哈希**：芯片格式速率 `["fp8", 2]` 与 `["fp8", 2.0]` 现在哈希相同。

复核无误（不改数值）：
- **DES 不变量**（新场景）：MoE TP2 / TP4·EP4 异构池、整前缀 LRU（亲和开 / 关）、radix 树、kv_policy wait / recompute / swap × 准入 before / after（紧 KV 容量）、gpt-oss 长度混合、投机解码、load 0.97，三种模式各 1500 请求：顺序、token 数、重复、槽位、KV 预留与占用均无违反。
- **回归**：0.61.1 / 0.61.2 的修正在 sweep / search / PD 路径与其他共用代码的模型上一致；指纹 0.61.2 → 0.61.3 默认 68 / 1356 变化（只有 alphafold2、openfold、boltz-1、protenix、opensora-stdit3），多节点 48 / 985，fabric 0 / 3150。
- **其他**：芯片预设峰值（100T 100.35、1P 1048.6、H100-like 959.4 TFLOPS）与文档一致；SLC pin / lru 公式；dtype 执行格式与反量化规则；场景哈希的往返与 int / float 稳定性；CLI 参数与文档；README 示例全部可运行。

### 19.13 第四轮审计修正（0.61.4）

修正：
- **DP 空闲 rank 的能耗 / TFLOP**：batch 不是 dp 的倍数（或 batch < dp）时，部分 DP rank 没有样本，原来能耗计数按 tp·sp·dp 个 rank 全满计 MAC / 向量操作，视频管线的 TFLOP 也乘满 dp。现在按实际有样本的 rank 数与填充率计（每单位 MAC 对 batch × dp 不变，测试覆盖 esmfold / boltz-1 / wan2.1-1.3b）。例（能耗表 0.5 pJ/MAC 等，见测试）：ESMFold batch 1 · DP2 每单位 MAC 194.65 → 97.32 TFLOP、1148 → 1078 J；Boltz-1 batch 3 · DP2 349.14 → 261.85 TFLOP、1329 → 1305 J；Wan2.1-1.3B batch 1 · DP4 每请求 29440.6 → 28585.3 TFLOP、每帧 MAC 712.85 → 352.90 TFLOP、1105 → 997.6 J；Wan2.1-14B batch 2 · DP4 653641.7 → 652992.7 TFLOP。时间与容量不变（dp = 1 的指纹全部不变）。
- **PD prefill 布局卡数不整除 decode_cards**：prefill TP4、decode 池 2 × TP1 时，换入 prefill 布局后 PD 块按新布局校验，报 「pd.decode_cards must be a multiple of the (decode) layout's 4 cards」，整个 PD 结果变成 error。池内评估改用去掉 PD 块的场景。
- **PD prefill 池预算**（0.61.3 只加了范围说明）：给预算且开 PD 时，按 prefill 池（prefill 布局、芯片、内存、prefill batch）再核对一次，结果在 `pd.budget`（含 `pool`）；`budget.ok_with_pd` = 两个池都满足；`budget.scope_note` 指向 `pd.budget`。
- **swap 主机链路能耗**：新增 `energy.pJ_bit_host`（PD kv_policy = swap 换出 / 换入经主机链路的每 bit 能耗，没有默认值）。swap 每次抢占的 KV 字节 × 2（出 + 入）计入 `host`；不填时 `host_note` 写明未计。CLI `--pJ-bit-host`，Web「pJ / bit 主机链路」。
- **混合格式**：gemm_exec 原来按同位宽当作原生（fp8 × int8、fp16 × bf16 按 int8 / fp16 原生速率），现在只有基础格式相同（fp8 各变体视为同一种）才走原生，否则两侧升到 bf16。发布模型无此组合，指纹不变。
- **Pareto**：默认 batch 列表原来止于 512，大 TP 的前沿被截断；现在 512 之后按 1.5k / 2k 步长扩到 `b_cap_for(cards)`，在第一个放不下处停止（显式 batches 不变）。例 Qwen3-8B TP8（100T + HBM3E）：前沿 11 → 15 点，最大 batch 384 → 3072，峰值 630.4 → 633.6 tok/s/卡。pareto 过滤 NaN / inf 点。
- **上下文超出发布长度**：ModelSpec 新增 `max_ctx`（发布 config 的 max_position_embeddings）；decode ctx + 1 + spec_k（或 prefill prompt）超过时警告（需 RoPE 外推 / YaRN，KV 与注意力仍按给定长度计）。
- **PP 流水级不平衡**：流水级按层数均分（不按代价），异构层栈（ESMFold LM + trunk、Boltz trunk + 扩散、AF2）最慢级 / 最快级可达 3×，节拍取最慢级，PP 结果偏悲观。现在比值 > 1.5 时警告（例 ESMFold PP2：2.14 / 7.02 s，3.3×）；不改切分。
- **CLI VLM 标注**：models 表与 eval 表头写明 VLM 视觉编码器未建模（Web 原有标注）。

复核无误（不改数值）：
- 第三轮修正回归：Boltz 缓存 / 别名、Protenix 按样本投影、Open-Sora 时间 VAE、AF2 模板行、冷存储不驻留（所有模型 SRAM 64 GiB / 512 MiB × PP1 / PP2 驻留率在 [0, 1]，Wan2.2 待机专家不占驻留）、PD recompute / swap 能耗、最佳 batch 警告、格式速率哈希。
- 投机解码 / MTP：期望 token 数 (1 − a^{k+1}) / (1 − a)，k 超过 MTP 层数时复用 MTP 层，验证宽度 1 + k，能耗单位按 token。
- 长上下文：全部 LLM 在 TP8 下 ctx 4k → 1M 的 TPOT、KV 单调有限（DSA / 压缩注意力 / 线性注意力族增长平缓）。
- batch > 1 视频 / 蛋白：每单位 TFLOP 与 MAC 对 batch 1 / 3 / PP2·微批 2 不变，延迟随 batch 线性。
- 数值极端：4×4×1 阵列 + 0.1 MiB SRAM、0.01 MiB SRAM、256×256×80 阵列、64 GiB SRAM、4 GiB SLC × 9 个模型 × prefill / decode：无异常、无 NaN / inf、无负值；延迟随芯片规模单调。
- VLM：视觉编码器（图像 token、分辨率）当时未建模，只评估语言主干（0.62 起已建模，§20）。

### 19.14 第五轮审计（0.62.0）

计划内的两项缺口（§20 视觉编码器、§21 PP 按代价切分）之外：

修正：
- **PD KV 交接字节（带缓存前缀）**：decode 侧已持有前缀时，要传的 KV 原来按整请求字节 × (S − p) / S 估算。滑窗层只需传窗口内未持有的 min(W, S − p) 个条目；递归状态（线性注意力）必须整份传（S 个 token 之后的状态不是前缀的状态）；压缩条目 / 索引键按 ⌈S/c⌉ − ⌈p/c⌉。新函数 `memplan.cache_new_bytes` 按层类型计。例 S = 8192、p = 4096（每请求）：Qwen3.8-27B 346.9 → 425.3 MB，Kimi-K3 345.6 → 577.9 MB，GLM-5.3-Flash 134.0 → 210.3 MB，Qwen3.5-397B 223.5 → 321.2 MB，Qwen3.8-2.4T 683.8 → 981.6 MB，MiniMax-Text-01 / M1 314.6 → 461.4 MB，Qwen3-Next-80B 140.2 → 179.7 MB，Qwen3.8-Flash-Next 159.5 → 218.3 MB，DeepSeek-V4.1-Flash 40.4 → 43.0 MB，DeepSeek-V4-Pro 44.3 → 48.3 MB，DeepSeek-V4-Flash 31.0 → 33.8 MB，gpt-oss-120b / 20b 153.4 → 155.7 / 102.2 → 103.8 MB；纯 GQA（Qwen3-8B 等）不变。只影响开 PD、有前缀命中（prefix_len / prefix_hit / prefix_tree）且 prefix_on_decode = true（默认）时的 KV 传输字节与时间；指纹场景没有前缀，不变。
- **DeepSeek-V4.1-Flash 发布角色**：`aligner.w1/w2.*`（视觉对齐投影，73,410,560 参数，bf16）被 `scripts/summarize_release.py` 归为 other（视觉正则缺 aligner），算进了语言主干的发布参数。现归 vision：params_llm 748,568,095,104 → 748,494,684,544，params_vision 411,857,920 → 485,268,480；参数偏差 −0.051 % → −0.042 %。不影响时间 / 容量（发布参数只用于核对）。
- **视频卸载放置的主机流量能耗**：每次重载全副本经主机链路读回的字节 = Σ 每卡（DiT 本级权重 + 文本编码器份额 + VAE 副本），按用户的 `energy.pJ_bit_host` 计入 `host`；未填时给 `host_note`。例 Wan2.1-14B 单卡卸载：69.0 GB / 段、852 MB / 帧。load_s（时间）不变。
- **Kimi-K2.7-Code 视觉编码器**：发布与 K2.5 的 vision_config 与图像处理器相同（只差视频参数），补进预处理表（否则这个 VLM 不计图像）。
- **关于页范围说明过时**：仍写着视频文本编码器 / VAE 未建模、ESMFold / Protenix 等未接入、「不做功耗 / 面积 / 成本估计」；本文 §2 也仍把已接入的视频 / 结构模型列为「暂未接入」。已按现状改写。

复核无误（不改数值）：
- 第四轮修正回归：DP 空闲 rank 能耗、PD prefill 预算 / 整除、swap 主机链路、混合格式、Pareto > 512、上下文 / PP 警告（tests/test_core_061_4.py 全过；PP 警告在 cost 切分下改为「已按代价平衡，仍不均」，按层数的提示只在 `pp_split = "layers"` 时出现）。
- 部分共享前缀（radix 树）的容量：前缀足迹按整条路径 Lp 计，按 token 线性换算成容量 token 数；全注意力模型精确，滑窗 / 递归状态模型是近似（状态按路径摊销），保持「假设」。前缀容量与 KV 容量现在按评估实际的各级层范围计（cost 切分后各级层数不同）。
- 图像 token 经 PD（prefill 池含编码器）、length_mix、prefix_hit、goodput、最佳 batch、扫描（serving.images / image_w / image_h）均一致。

## 20. VLM 视觉编码器（0.62）

目录里每个 VLM（Qwen3.5-397B-A17B、Qwen3.8-27B、Qwen3.8-Flash-Next、GLM-5.3-Flash、Kimi-K2.5 / K2.7-Code、Kimi-K3、DeepSeek-V4.1-Flash）的视觉塔 + 合并 / 投影按发布 `vision_config` 与发布的图像预处理器建模（`core/vision.py`），dtype 按发布（全部 bf16）。

- **图像 → patch 网格**（按各自预处理器）：Qwen `smart_resize`（因子 = patch 16 × 合并 2 = 32，像素 65536–16777216）；GLM `smart_resize`（因子 28，16–8000 token，时间 patch 2 对静态图重复）；Kimi MoonViT `navit_resize`（patch 14，in_patch_limit K2.5 / K2.7 16384、K3 65536，单边 ≤ 512 patch，补齐到 28）；DeepSeek `plan_image_grid`（min_pixels 295936，max_image_tokens 1024，宽高比 ≤ 3）。
- **进入语言模型的 token**：合并后的位置数（Qwen / GLM / Kimi 2×2 合并；DeepSeek 3×3 对齐器后每行加换行，另加首尾 2 个）。1024×1024：Qwen 1024、GLM / Kimi 1369、DeepSeek 652；448×448：196 / 256 / 256 / 184。聊天模板的分隔 token 算作文本 token。
- **算子**：patch 嵌入 GEMM、每层 QKV / O / MLP（GLM、DeepSeek 为门控 MLP）、整图双向注意力（每张图 N×N，N = patch 数；QK、PV 计 MAC，softmax 每个分数 5 次向量操作）、合并 / 投影 GEMM（按合并后的行数）。走与语言主干相同的映射 / 存储 / 调度路径（流式，bf16）。
- **对照参考实现**：torch FlopCounterMode 计 transformers 5.19（qwen3_5 / qwen3_5_moe / qwen4_exp / glm5_next）与 Kimi、DeepSeek 发布 remote code 的视觉模块（在 meta 上构建后 materialise、bf16、eager 注意力）。7 个模型 × 448×448 / 640×480 / 1024×768 的 token 数与 FLOPs 全部精确一致（tests/test_core_062.py），参数与发布 params_vision 逐个相等（DeepSeek 含对齐器，见 §19.14）。K2.7-Code 的视觉代码与 K2.5 相同，沿用其对照。
- **场景输入**：`serving.images`（每请求图像数，默认 0 = 只算文本，所以既有结果全部不变）、`serving.image_w / image_h`（默认 1024×1024「假设」，代表常见的截图 / 照片分辨率）。图像 token 一次性并入 prompt 与上下文（prefill 长度、KV、PD 长度、goodput 随之变化）；结果里的 scenario 原样回显所填的 prompt，`summary.vision` 给出 text_prompt + image_tokens。prefill 的 tok/s 按含图像 token 的 prompt 计。
- **放置与时间「假设」**：编码器在 prefill 时、语言 prefill 之前串行运行；首个流水级的卡各编码一部分图像（数据并行，即 vLLM `mm_encoder_tp_mode = data` 的做法），时间 = 每卡 ⌈本 DP rank 图像数 / (TP·SP)⌉ 张的编码时间；权重（bf16）常驻首级，两个阶段都占容量。decode 不运行编码器。能耗计数计入编码器的动作。
- **未建模**：视频输入（时间合并 / 抽帧）、编码器输出缓存（同一图像重复出现时跳过编码）、前缀缓存命中图像 token 时跳过编码（按全部重新编码计，偏保守）、编码器与语言 prefill 的流水重叠。
- 入口：API `serving.images / image_w / image_h`（扫描路径同名）；CLI `--images N --image-size WxH`；Web「服务」组（只在 VLM 显示）与单点页「视觉编码器」卡片。

## 21. PP 按代价切分（0.62，默认）

0.61 及以前流水级按层数均分。一个 stage 的时间是 `max(Σ阵列, Σ向量, ΣDRAM, Σ链路) + Σ同步`，首级另有嵌入，末级另有输出头 / MTP 草稿；异构层栈（结构模型的 trunk / 结构模块 / 扩散、首层稠密 MoE、滑窗与全注意力交替）各层代价差别很大。按层数均分是建模错误，节拍偏悲观。

- **代理**：每层（按层组记忆）用与评估相同的算子求和得到（阵列, 向量, DRAM 触达字节 / 带宽, 链路, 同步），嵌入 / io-pre 加到首级，输出头 / io-post / MTP 加到末级，非末级加级间传递；DRAM 按全部流式计（忽略 SRAM / SLC 驻留）。
- **切分**：对节拍二分 + 贪心装箱，求连续切分的精确极小极大（代理意义下）；再在可行窗口内把边界调向剩余工作的均分，避免末级空闲。每级存储（权重 + 本级 KV / 状态 × batch）不超过 max(按层数均分的最重级, 每卡 DRAM)。
- **由完整模型裁决**：代理给出的候选（均分调整版、前置贪心版）与按层数均分一起用完整评估（含驻留、激活、fabric）各跑一次，取放得下且节拍最小者，平局取按层数均分。所以 cost 切分的节拍不会劣于 layers，也不会因此从放得下变成放不下（指纹核对：0 个变慢、0 个 fits 变化）。fabric_report 只记录被选中的那次运行的集合通信。
- **单调性（0.62.1 审计）**：候选集由代理给出，而代理随硬件参数变化（DRAM 带宽进入代理的 DRAM 项），所以 cost 切分下 §9 V0 的「更多带宽 / SRAM / 更快链路不会更慢」只是近似成立：在 400 个随机 PP > 1 场景 × DRAM 效率 / 链路带宽 / SRAM 三组扫描中没有出现反例；组合 fuzz 1200 个场景中出现 1 例——Qwen3-8B-AWQ PP4·TP8、H100 类芯片 + 256 MiB SLC、权重全部驻留 SRAM / SLC（代理却按 DRAM 流式计），DRAM 效率 0.5 → 0.7 时换到另一候选，TPOT 0.523 → 0.573 ms（+9.6 %），仍优于按层数均分（0.674 ms）。layers 切分与 PP = 1 的单调关系保持严格（测试只对这两种情形断言）。
- **选项**：场景 `pp_split = "cost" | "layers"`（默认 cost；默认值不进场景哈希）；CLI `--pp-split`；Web 布局组「PP 切分」。CLI eval 打印各级层范围。
- **变化范围**（0.61.4 → 0.62，只在 PP > 1；PP = 1 全部不变）：LLM decode 时末级的输出头使均分偏慢，PP2 节拍 −0.03 % ~ −13.7 %（小模型、量化模型与 lm_head 占比大的 MoE 降得多），PP4 −0.9 % ~ −36 %；prefill 只有 DeepSeek-V4.1-Flash 变化（−5.7 % ~ −6.0 %）；结构模型 PP2 −2.4 % ~ −37.9 %、PP4 −0.6 % ~ −50.4 %（ESMFold、Boltz-1、Protenix 最大）；视频只在 PP4 有 ≤ 0.1 % 的变化（指纹的 PP2 无变化）。逐项见 CHANGELOG 0.62.0。

## 22. 外部评审修正（0.63）

0.62.1 交给外部评审（external review）复查，提出 9 条问题和一个覆盖缺口。每条都先复现，再修正并加上永久的性质测试（`tests/test_core_063.py`）。下面的数值都在 100T 默认芯片上测得，除非另注。

### 22.1 MLA decode 吸收路径（P1）
吸收式 decode 原来把 kv_b 当成一个宽 GEMM（DeepSeek-V3：(1, 512, 32768)）。官方 `model.py`（L447–461）的做法是逐头两次乘法：q_nope · W_UK（每头 128 → 512），以及注意力输出 · W_UV（每头 512 → 128）。现在按这个结构建两组算子 `attn.kv_b.uk`（m = t，k = nope，n = kv_lora，count = 头数）和 `attn.kv_b.uv`（m = t，k = kv_lora，n = v_dim，count = 头数）。参数、FLOPs、权重格式都不变，变化的是映射：逐头小 GEMM 用不满宽阵列。100T、DS-V3 单 token 的 kv_b 周期：OS 9472 → 40960，ws_broad 2215 → 3568。受影响的是全部 MLA 模型的 decode（DeepSeek-V3 / V3.2、Kimi-K2 / K2.5 / K3、GLM-5.x），步时 +0.2 % ~ +11 %。

### 22.2 视频 CFG 依赖（P1）
一个去噪步里，同一请求的 cond / uncond 两次前向都要完成，才能更新潜变量。原来的流水调度把它们当成两个独立请求，按 `max(mb, PP)` 个 tick 计。现在先算 k，即被同一请求连在一起的最大微批组。k > 1 时每步按 `max(mb, k + PP − 1)` 个 tick 计（这一组的最后一个微批要过完全部流水级），k = 1 时仍是 `max(mb, PP)`。例：Wan2.1-1.3B，batch 1，PP2，CFG 2，1 步，3.255 s → 4.883 s（3 tick），所以 CFG2 不再等于 CFG1 的时间。PP4 下 batch 1 / 2 由 4 tick 变为 5 tick。PP1 不变；所有 PP > 1 的视频延迟 +37 % ~ +49 %（指纹 PP2）。性质测试：任意 (batch, CFG, mb, PP) 下，每步 tick 数 ≥ 依赖链长度。

### 22.3 SLO 最大吞吐搜索（P1）
`pdqueue` 的 SLO 速率原来用倍增 + 二分，前提是可行性随 λ 单调。prefill batch 上限会随负载换挡，造成不可行的凹区。复现场景：Qwen3-Next-80B-A3B，1P + HBM3E，TP2·EP2，B16，prompt 1024 / out 512，前缀命中 0.4，SLO 80 / 100 ms。工作点 2.31 req/s 落在凹区里，2.834 req/s 可行，但报告的最大值是 1.724。现在的做法分三步：
- 稳定上限 λ_s 仍用二分求（稳定性单调），并按场景缓存；
- 在 (0, λ_s] 上自上而下扫 24 个点，在第一个可行点与其上方相邻的不可行点之间二分；
- 结果不低于任何已知可行点（含当前工作点）。

该例结果 1.724 → 2.892 req/s（= 稳定上限，可行）。
未做（0.63）：评审建议让 prefill batch 上限的选择考虑 SLO；DES 侧的 `pdsim.slo_rate` 仍是二分。0.64 两项都已完成，见 §23.1。

### 22.4 radix 前缀树的容量依赖（P1）
`prefixcache` 的 Che 近似原来只检查节点自身大小，不检查与祖先路径的累计大小。结果是路径放不下时也给出了整条路径命中。例：两层各 60 token，容量 100：命中 [0.833, 0.833] → [1, 0]，与 RadixLRU（DES）一致。现在只有累计路径放得下的层才可缓存，子节点命中 ≤ 父节点（尾部单调截断）；DES 的 RadixLRU 用同样的累计截断。性质测试：随机树与容量下，子 ≤ 父，且被缓存的路径总长 ≤ 容量。

### 22.5 不整除的 TP（P2）
原来 GQA 在 TP 不整除时会丢头：Qwen3-8B TP3 只覆盖 32 个 query 头中的 27 个。现在考虑两种头完整的切分，取更便宜的一种（先比 KV 头数，即 KV 缓存与读量，再比每 rank 的 query 行数）：
- **组完整**：TP ≤ KV 时每 rank ⌈KV/TP⌉ 个 KV 头及其 G 个 query 头；TP > KV 时 1 个 KV 头，其 G 个 query 头分到 ⌊TP/KV⌋ 个 rank。
- **按头分**：query 头按连续块分配，块大小为 ⌈Q/TP⌉ 或 ⌊Q/TP⌋；一个块触及的每个组的 KV 头都要放在该 rank，跨块的组在两个 rank 上复制。

注意力按 KV 头分块，每块 ⌈Q_loc/KV_loc⌉ 行 query，补齐的行只占时间。q / o_gate / o / k / v 投影按头对齐。示例：
- Qwen3-8B TP3：取组完整（3 KV × 4），decode b8 35.89 → 36.42 ms，prefill b8 2236 → 2404 ms。
- Phi-4（40 Q / 10 KV）TP8：取按头分（5 个 query 头，2 个 KV 头），prefill 1588 → 1699 ms。

每 token 的 MAC 等于 TP1（守恒，见 22.7）。vLLM 不支持这类 TP，这里给出的是可行切分的代价。

### 22.6 PP 级间路由（P2）
级间传递原来只评估一对代表端点：stage i 的最后一个 rank → stage i+1 的第一个 rank。现在逐对评估每一对对应 rank（j → j + 每级卡数），每对按自己跨越的层级（节点内 / leaf / spine / core）计，取最慢的一对，fd / fn 为各层级的对数比例。例：PP2·TP64，每节点 8 卡，4:1 收敛，单次传递 1 ms → 4 ms。Llama-3.3-70B PP2·TP64 prefill 第 0 级链路 111.07 → 127.17 ms（步时不变，1465.3 ms，因为链路不是瓶颈）。其他路径（KV 传输）原本就用平均跨越因子，不受影响。

### 22.7 有效工作守恒（P2）
每个算子新增 `share`，等于平均 rank 的有效工作 / 本 rank 的工作，按算子名对照未切分的算子图得到：`N · Σ work·share = 全局工作`，N = TP·DP·SP。最忙 rank 的算子保留时间；能耗计数（MAC / 向量）和每请求 TFLOP 都按有效工作计。四种情形：
- 不整除的切分（头、列、token、batch）。
- 专家补齐行：hit · m_e ≥ token-专家对，share = 有效对数 / 补齐行数。
- 最后一个微批的补齐：× S / (mb·⌈S/mb⌉)。
- 视觉编码器按卡补齐的图像。

两类真实重复的工作照常计入：
- 复制的计算（MLA 潜变量投影在每个 TP rank 上执行）。
- 权重反量化：每遍、每个忙碌 rank 一次。

另外，EP > 1 时没有序列的 DP rank 仍运行其专家。字节计数仍按忙碌 rank（0.64：空闲 rank 只计其专家权重字节，§23.5；权重反量化按平均 rank 的权重量计，§23.4）。

例：
- ESM-2 650M，batch 3，PP2 的 MAC / 序列 4.756e11 → 3.567e11（= PP1）。
- 视觉编码器 3 张图 TP2 原来按 4 张计，现在与 TP1 相同。
- Mixtral-8x22B batch 3 decode 的 MAC / token，PP1 / PP2 / DP2·EP2 原为 62.9 / 53.8 / 31.9 G，现在都是 40.4 G。

性质测试：LLM / MoE / MLA / 混合 / 视频 / 蛋白质在随机 TP / DP / EP / SP 与不整除 batch / 头 / token 下 Σ 有效工作 = 全局；能耗 MAC 计数不随切分或补齐变化。

### 22.8 DES 投机解码（P2）
`pdsim` 原来把每步期望接受 token 数四舍五入（1.7 → 2），DES 偏乐观约 15 %。现在每个序列每步抽一个截断几何分布的接受数，均值为 (1 − a^{k+1}) / (1 − a)，与闭式一致；使用独立随机流，不扰动到达 / 长度。例：DeepSeek-V3 spec_k 1、a 0.7，DES / 闭式 TPOT 由 3.957 / 5.727 ms 变为 5.809 / 5.943 ms（λ 13.868；闭式变化来自 22.1）。
未改（0.63）：闭式排队里运行 batch 的 k̂ 取整。0.64 改为两点混合，见 §23.3。

### 22.9 线程安全（P2）
`fabric` 的模块级 `_LOG` / `_MULT` 在多线程 HTTP 服务下会串号。现在改为 `contextvars`，接口为 `fabric.log() / set_log() / set_mult()`；`pdsim.capture_ctx` 原来猴子补丁 `disagg`，现在改用 `disagg.CTX_SINK`（ContextVar）。测试：6 个线程并发 fabric_report + evaluate，各自的集合通信记录不混。其余模块级状态都是纯函数的记忆缓存。

### 22.10 LLM prefill 激活流式 / 溢出（覆盖缺口）
0.62 的 LLM prefill 不计激活的 DRAM 流量：Qwen3-8B B4 4K、64 MiB SRAM 时 staging 1792 MiB，激活 DRAM 流量为 0。现在 LLM 的 GEMM、lm_head、MTP 与 prefill 注意力也用全序列前向的流式模型（§11 激活流式），具体如下：
- staging ≤ SRAM/2。
- GEMM 取「激活分块、权重重读」与「权重分块、激活重读」中流量小的一种，按实例（专家 / 头 / 组）分别决定。
- 注意力按 flash 方式计算，已缓存的前缀 KV 只读一次（`kv_pre`）。
- DRAM 容量另加 max(0, 最大单算子激活 − 1 GiB 运行时预留)。

Qwen3-8B B4 4K prefill，SRAM 16 / 64 / 256 / 1024 MiB 时激活 DRAM 分别为 398.6 / 149.8 / 101.5 / 33.8 GB（原为 0）。步时不变（MAC 受限，3262.6 ms），带宽受限的芯片上会显现。decode 的激活通常装得下（16 MiB 时 0.098 GB）。

**近似**（0.63）：两端分块不是最优的二维分块，小 SRAM 下偏保守。0.64 已改为二维分块，见 §23.2。视频 / 蛋白质沿用同一规则，便于横向比较。

### 22.11 数值变化（0.62.1 → 0.63.0 指纹）
所有变化都能归到上面某一条，没有无法解释的变化。
- **单节点指纹（1356 项）**：470 项变化。
  - MLA decode 步时 +0.23 % ~ +3.29 %（34 项，22.1）。
  - 视频 PP2 延迟 +37.4 % ~ +49.5 %（32 项，22.2）。
  - prefill 的 DRAM 需求 +0 % ~ +9.0 %（404 项，22.10：激活工作集超过 1 GiB 预留的部分计入容量）。其中 2 项 fits 由是变否：gpt-oss-120b、LPDDR5X 64 GiB、PP1、B8 prefill（os / reconf），62.98 → 64.09 GiB。
  - prefill 步时全部不变：默认 100T 上 MAC 受限，溢出流量被掩盖。
- **多节点指纹（985 项）**：34 项变化。
  - 视频 PP2·TP2 +30.3 % ~ +48.9 %（24 项）。
  - Phi-4 prefill（KV 10 头不整除）：TP8 +6.97 %，PP2·TP4 +5.55 %（6 项，decode 不变）。
  - PD 4 项：能耗计数（22.7）；Qwen3-8B 合并分块的 decode 迭代 67.9 → 74.3 ms，原因是分块 prefill 的激活 / 前缀 KV 流量（22.10）。
- **fabric 指纹（3150 项）**：2 项变化，均为 DeepSeek-V3 PD 报告中放不下的单卡 prefill 池的 TTFT（3520 → 3871 ms，激活流量）。集合通信行与步时全部不变。
- **0.63 新指纹（840 项）**：653 项变化，大部分是能耗计数（22.7）。
  - 步时：MLA decode +0.47 % ~ +11.1 %（64 项）；TP3 decode +0.29 % ~ +3.46 %，TP3 prefill +0.79 % ~ +8.91 %（各 36 项）；视频 PP2 +37.7 % ~ +49.4 %（16 项）。
  - radix 尾部 2 项：[0.833, 0.833] → [1, 0]；[0.318, 0.067] → [0.657, 0]。
  - SLO 速率：Qwen3-Next 1.724 → 2.892，另有合并分块的 SLO 速率 +0.003 %（扫描细化）。
  - 能耗计数：
    - MAC 在 PP 补齐（B3 PP2 −25 %）与专家补齐处下降；DP·EP 处上升（原来没有序列的 DP rank 不计专家工作），MoE 达 +75 %。
    - 向量操作在不整除 TP 与 PP 补齐处下降；FP8 / MXFP4 权重反量化按遍数计，EP / PP 处最多 +33 %。
    - DRAM 计数在 prefill 处上升（激活溢出，小模型 B3 prompt 2048 最多约 ×10）。

### 22.12 同类模式排查
- 代表端点：只有 PP 路由（已修）。
- 全局状态：fabric 已修，capture_ctx 已修；其余为纯缓存。
- 舍入：DES 投机已修；k̂ 取整保留（0.64 已改为两点混合）。
- 单调二分：
  - api 最大可放 batch 的二分，容量随 batch 单调，仍成立；
  - 精确 batch 搜索依赖 P1（step 随 B 不减），已有测试；
  - DES slo_rate 仍是二分，只用于校验（0.64 已改为扫描 + 细化）。

## 23. 0.63 未做项收尾（0.64）

0.63 留下的「未做」清单在本版全部完成：SLO 感知的 prefill batch 上限、DES SLO 搜索、二维 GEMM 分块、闭式运行 batch 取整、不整除 TP 的反量化、EP 下空闲 DP rank 的 DRAM 字节。每项都加了性质测试（`tests/test_core_064.py`，12 项）。下面的数值在 1P + HBM3E（PD 例）或 100T 默认芯片上测得。

### 23.1 SLO 感知的 prefill batch 上限
默认上限仍取平均 TTFT 最小者（DistServe 规则）。场景给了 SLO（`ctx["slo"]` = (p90 TTFT, p90 TPOT)）而默认上限不满足时，逐一试 B_CAPS 中其余上限，在满足两项 SLO 的上限中取平均 TTFT 最小者（结果 `prefill.cap_rule = "slo"`）；都不满足则保留默认。于是「λ 处 SLO 可行」= 「某个上限可行」，为各上限可行集（对固定上限随 λ 单调）之并。
- PD：decode / KV 的稳定性与 prefill 上限无关，默认上限不稳定时不再试。
- 合并 · prefill 优先：上限同时决定 decode 份额，默认上限不稳定时也试其余上限；没有 SLO（或都不满足）时取稳定上限中平均 TTFT 最小者（`cap_rule = "stable"`）。**新发现**：0.63 只按 prefill 队列选上限，上限 1 的占用让 decode 份额不稳定，而上限 4 稳定且满足 SLO——Qwen3-Next-80B-A3B（下例）负载 0.95 时该模式原报「不稳定」，现在 cap 4、p90 TTFT 67.0 ms、p90 TPOT 12.5 ms。稳定性 = 「某个上限稳定」，仍随 λ 单调，二分有效。
- 分块 prefill 没有上限，不变。
- SLO 速率搜索仍是 §22.3 的稳定上限二分 + 24 点扫描 + 细化；稳定性判断用无 SLO 的选择。
- DES `pdsim.slo_rate`：完成上限 λ_c（全部请求完成）用 ×1.5 增长 + 二分，再在 (0, λ_c] 上自上而下扫 `DES_SLO_SCAN` = 8 点并细化；共同随机数、按 λ 缓存。DES 的 prefill 上限取闭式结果在该 λ 选出的上限。

例：Qwen3-Next-80B-A3B，TP2·EP2 decode、TP1 prefill，B16，prompt 1024 / out 512，SLO 80 / 100 ms：
- 命中 0、负载 0.6：PD 上限 1 → 2，TTFT 均值 / p50 / p90 / p99 = 76.3 / 71.3 / 86.4 / 141.4 → 80.8 / 79.4 / 79.4 / 114.6 ms（均值略升、p90 降到 SLO 内）；PD SLO 速率 1.580 → 2.564 req/s（+62 %），最优切分同。
- 命中 0.4、负载 0.6 / 0.85：上限 1 → 2，p90 80.4 → 73.8、100.4 → 73.8 ms。
- 合并 prefill 优先 SLO 速率 2.605 → 2.779（命中 0），2.619 → 2.828（命中 0.4）；稳定速率同幅上升（稳定上限回退）。
- DES 对照（n_req 1200）：PD 1.500 → 1.489（命中 0），1.594 → 1.615（0.4）。闭式 2.564 / 2.892 明显高于 DES：闭式上限 2 的 p90 TTFT（λ 2 时 79.4 ms）比 DES（约 86 ms）乐观——p90 落在 P(W = 0) 的原子上。DES 的 prefill 优先「速率」23–44 req/s 是 `complete` 作稳定性代理的已知弱点（0.63 即如此），不是本版引入。

### 23.2 二维 GEMM 分块（激活溢出）
`memplan.gemm_blocking(m, k, n, A, W, budget)` 返回 (A 读次数, W 读次数)，DRAM 字节 = A·r_A + W·r_W + C（C 写一次），取三种循环嵌套中最便宜的：
- 权重驻留：W 分 ⌈W / budget⌉ 块驻留，A 每块流过一次 → (⌈W/budget⌉, 1)；
- 激活驻留：A 分 ⌈A / budget⌉ 块，W 每块流过一次 → (1, ⌈A/budget⌉)；
- 输出驻留：bm × bn 的 fp32 部分和驻留（bm·bn·4 ≤ budget「假设」），A / W 按 K 片流入 → (⌈n/bn⌉, ⌈m/bm⌉)，bm 取 2 的幂与 m（经典 I/O 下界式分块，流量约 2·m·n·k·e / √(budget/4)）。

budget 只装驻留块，流入的行 / 列 / K 片双缓冲在预算外（与 0.63 相同「假设」）；前两种的块数按字节而不是整列 / 整行算（块边界把某一列沿 k 切开时，该列的部分和留在片上），所以恰好装下（W = budget）只读一遍。0.63 的两种单边分块被前两种（弱）支配，故 0.64 流量 ≤ 0.63，且随 SRAM 不增。手算：16384×4096 · 4096×12288 bf16、budget 32 MiB：权重驻留 3 块 → A ×3，3·128 + 96 = 480 MiB（输出驻留 1152，激活驻留 512）；8192³ bf16、4 MiB：输出驻留 bm = bn = 1024 → (8, 8) = 2048 MiB，0.63 的最优单边为 4224 MiB。

开发中的一个中间版本把流入片也算进预算、且按整列计块，在恰好装下处（Mochi-1 img_gate_up TP2：fp32 权重 96 MiB = 3 × 32 MiB，整列分块需 4 块）流量反而比 0.63 高 5 %；最终版按上面的规则，指纹中 DRAM 计数全部不升。

Qwen3-8B B4 4K prefill，SRAM 16 / 64 / 256 / 1024 MiB 时溢出（激活 + 权重重读）≈ 305.6 / 143.8 / 96.6 / 33.8 GB（0.63：398.6 / 149.8 / 101.5 / 33.8）；DeepSeek-V3 B8 2K：≈ 431.9 / 281.8 / 212.8 / 133.7 GB（0.63：518.1 / 283.4 / 213.1 / 133.7）。步时不变（MAC 受限）。激活部分单独看只是不增（64 / 256 MiB 都是 96.6 GB），因为一种嵌套可以拿激活重读换权重重读。

### 23.3 闭式运行 batch：两点混合
生灭过程的所见运行 batch 均值 n̄ 是小数。0.54–0.63 取 k̂ = round(n̄) 求步时与 Result（± 半个 batch 的偏差）。现在 k_lo = ⌊n̄⌋、k_hi = k_lo + 1、f = n̄ − k_lo：步时 = (1 − f)·step(k_lo) + f·step(k_hi)；prefill 优先的残余与 decode 份额、KV 层利用率 u_coll、`kv_slowdown`、decode 能耗计数都按两点混合；分块模式的 k_s 也按两点插值（`mean_step` 的 T₀、各块迭代、t₁）。`decode.running_batch` 报小数。指纹：prefill 优先 TTFT −3.1 % ~ +2.5 %；PD TTFT 只有 < 1e-4 的变化（u_coll）；分块 p90 TPOT ±0.03 %。
**新发现**：Qwen3-30B-A3B TP2·EP2（多节点指纹 PD 例，B32，prompt 4096 / out 256，SLO 2000 / 50 ms）的合并分块 SLO 速率 0.715 → 0.177 req/s。0.63 在 λ = 稳定上限处取整的不动点落到一个错误解（k = 2，p90 TTFT 555 ms、TPOT 18.7 ms），把稳定上限本身判为可行；而 0.18 req/s 以上 TPOT 都超过 50 ms。两点混合后边界处为 k ≈ 32、TTFT 发散，结果正确。

### 23.4 不整除 TP / EP 的权重反量化
权重反量化每遍、每个忙碌 rank 一次，0.63 用最忙 rank 的权重切片 × rank 数。新增 `Op.wshare` = 平均 rank 的权重元素 / 本 rank：Σ_rank w = copies · W_global，copies = ⌊N · w_rank / W_global⌋（≥ 1，DP / SP / TP 复制的整数份数；ceil 切分让最忙 rank 多出不到一份），wshare = copies · W_global / (N · w_rank)。「假设」：部分复制的切分（如 TP3 下 2 个 KV 头）按不均匀切分处理。例：qwen3-32b-awq B4 decode 向量计数 / token TP3 16.08 → 15.70 G、TP6 17.21 → 15.70 G（= TP1）；gpt-oss-20b TP2·EP2 / DP2·EP2 decode −8.3 %（各 rank 命中的专家数不均）。

### 23.5 EP 下空闲 DP rank 的 DRAM 字节
EP > 1 时没有序列的 DP rank 仍运行其专家（0.63），但它只读自己的路由专家权重，不读注意力 / dense / KV / 激活。`step_dram_bytes` 新增 `exp_dram` / `exp_slc`（专家权重的 DRAM / SLC 字节），能耗里空闲 rank 只计这部分。例（decode，每 token DRAM）：Qwen3-30B-A3B DP4·EP4 B1 14.81 → 6.42 GB，B1–B4 现在都是 6.42 GB（低 batch 下专家字节 ∝ token）；DeepSeek-V3 DP8·EP8 B1 160.0 → 37.9 GB。空闲 rank 的 SRAM / 链路 / 反量化计数仍按忙碌 rank（略高估，未做）。

### 23.6 数值变化（0.63.0 → 0.64.0 指纹）
- 单节点 0 / 1356；fabric 2 / 3150（DeepSeek-V3 PD 单卡 prefill 池 TTFT 3870.8 → 3860.0 ms，−0.28 %，二维分块）。
- 多节点 170 / 5859 个值，全部在两个 PD 报告里：Qwen3-8B 合并分块 TTFT −5.9 % ~ −9.0 %、prefill 迭代 88.2 → 85.8 ms（二维分块减少块迭代的激活流量）、槽位等待 p99 442.8 → 120.4 ms；Qwen3-30B-A3B 合并分块 SLO 速率 0.715 → 0.177（23.3）；prefill 优先稳定速率 +0.4 % ~ +1.6 %（23.1 回退）；运行 batch 改报小数，能耗 DRAM −2.3 % ~ +0.5 %。
- fp63（840 项）381 项变化：DRAM 计数全部不升——prefill −0 % ~ −50.8 %（二维分块；最大 qwen2.5-72b PP3 B1），视频 / 蛋白质 −0 % ~ −23.1 %，DP·EP decode −12.0 % ~ −46.7 %（23.5）；向量计数 TP3 / TP6 −0.5 % ~ −2.6 %，gpt-oss-20b EP2 −8.3 %（23.4）；步时、MAC、TFLOP 不变。PD 速率见 23.1。
- fp64（PD TTFT，5 场景 × 4 负载 × 命中 2 × CV 2 = 80 行）全部有变化，418 个 TTFT 值：prefill 优先 276 个 −3.1 % ~ +2.5 %（23.3）；PD 110 个 < 1e-4（u_coll）；8 行换上限或稳定性（PD 上限 1 → 2 四行，prefill 优先由不稳定变为 cap 4 四行，见 23.1）；PD SLO 速率 1.580 → 2.564（命中 0 四行）；prefill 优先 SLO 速率 −0.04 % ~ +8.0 %、稳定速率 +0.04 % ~ +8.1 %；分块 p90 TPOT ±0.03 %。

### 23.7 未做 / 剩余
- 闭式上限 2 的 p90 TTFT 比 DES 乐观（M/D/1 批服务器近似 + P(W = 0) 原子）；DES 的 `complete` 是弱稳定性代理（prefill 优先的 DES 速率偏大）。
- 空闲 EP rank 的 SRAM / 链路 / 反量化计数。
- 输出驻留的块大小只在 2 的幂上搜，K 片大小未参与。
- SLO 感知上限只在 B_CAPS 上选，不做连续 / 动态批。

## 24. 批服务排队与 DES 稳定性（0.65）

0.64 留下的「未做」：闭式上限 ≥ 2 的 TTFT 分位比 DES 乐观、DES 的 `complete` 是弱稳定性代理、空闲 EP rank 的 SRAM / 链路 / 反量化计数、输出驻留分块只搜 2 的幂。性质测试 `tests/test_core_065.py`（10 项）。数值在 1P + HBM3E 上测得。

### 24.1 贪心批服务排队（`core/bulkq.py`）
DES 的 prefill 副本（`pdsim.PrefillReplica`）是贪心的：空闲且有人排队时立即取 k = min(队列, b) 个成批，同时离开，墙钟 T(k) = 批内 TTFT(k, S_i) 的均值；空闲时到达的请求单独开批。0.51–0.64 把它当作 M/G/1（每请求服务 τ_b = T(b)/b、自身延迟 T(b)），只有批总是满时才精确。部分负载下批大多很小，服务器每请求忙 ≈ T(1) ≈ T(b)，真实等待更长、自身延迟更短：Qwen3-Next 上限 2 的 p90 79.4 ms，DES ≈ 86 ms。

现在（上限 ≥ 2；上限 1 仍是精确 M/G/1）：
- **嵌入链**（服务开始时刻）：状态 n = 开始时排队数，k = min(n, b)，r = n − k 留下，服务期间到达 A ~ 混合泊松(λ·T)，n′ = r + A（为 0 时下一个到达单独开批）。状态 1..N 带状消元精确求解，n > N 为齐次几何尾 z = 1/x₀（A_b(x) = x^b 的实根）。
- **均值**（选上限用）：Little，L_q = 等待（不在运行批中）人数的时间平均，W_q = L_q / λ；自身墙钟 = 按请求加权的 E[T(k)]。稳定性不变：λ·E[T(b)] < b。
- **TTFT 律**（只对选中上限）：标记请求（PASTA）以空闲时间份额遇到空闲 → 自身律 O₁；否则落在开始状态为 n、时长 t 的服务的年龄 u 处，前面 p = r + j 人（j ~ Pois(λu)），F = ⌊p/b⌋ 个满批（Σ T(b) 用 3 点矩匹配「近似」）之后是自己的批，人数 = p mod b + 1 + L（L ~ Pois(λ·(t − u + ΣF))，封顶 b）。u 在 N_U = 8 个子区间上积分，等待部分落在宽 h = max(E[T(1)]/32, (W_q + E[T(b)])/160) 的网格上（「近似」：分位在格内线性插值）。
- **自身部分按类型精确**：给定自身批大小 k，自身部分 = (T(k, S_i) + (k − 1)·其他人均值)/k + KV 暴露(i, 自身墙钟)，按自己的 prompt 类型 i 混合（i 与它遇到的队列独立，所以这不是独立性近似）。一开始用了无类型的批墙钟 ⊕ 独立暴露，CV = 1 时 PD p90 偏乐观 −13 %，按类型后 ±4 %。
- 批墙钟律 D_k：k ≤ 8 精确卷积（合并到 6 个分位片），k > 8 用精确均值 / 方差的 5 点 Gauss–Hermite（忽略 ∝ 1/√k 的偏度「近似」）；8 以上非 2 的幂的 k 在相邻 2 的幂间线性插值 TTFT（「假设」）。
- 校核：b = 1 与 M/D/1 精确一致；确定性 b = 2 / 4 / 8 / 16 与蒙特卡洛均值 ±1 %、p50 / p90 ±3 %；混合服务 ±2 %（p90 偏保守 ≤ 4 %）。
- prefill 优先：decode 份额用批服务器的时间忙份额（1 − 空闲份额，部分批比满批更占时间），停顿份额用每秒批数 λ / E[批大小]。

### 24.2 与 DES 对照
Qwen3-Next-80B-A3B（TP2·EP2 decode，TP1 prefill，B16，prompt 1024）。DES 每点 12000 请求（预热 1500）× 2 seed。

**(a) prefill 受限**（out 16，SLO 300 / 100 ms），强制上限、负载 = 该上限稳定速率的比例，p90 TTFT 误差（闭式 − DES）/ DES：

| 模式 | CV | 上限 | 0.3 | 0.5 | 0.7 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|
| PD | 0 | 1 | 0 | −0.5 | +1.9 | +9.3 | +23.2 |
| PD | 0 | 2 | −0.1 | −0.4 | +1.6 | +7.6 | +21.2 |
| PD | 0 | 4 | +0.1 | +0.3 | +1.6 | +5.9 | +19.2 |
| PD | 0 | 8 | +0.1 | +0.3 | +1.4 | +9.8 | +18.5 |
| PD | 1 | 1 | −0.2 | +0.4 | +3.3 | +11.2 | +25.1 |
| PD | 1 | 2 | +0.3 | −0.4 | +3.3 | +8.6 | +28.8 |
| PD | 1 | 4 | +0.2 | −1.0 | +2.4 | +6.6 | +31.8 |
| PD | 1 | 8 | −0.9 | −0.4 | +1.0 | +6.6 | +24.0 |
| prefill 优先 | 0 | 1–8 | +3.9 ~ +4.9 | +3.3 ~ +6.3 | +4.8 ~ +8.4 | +10.1 ~ +10.5 | +11.1 ~ +12.0 |
| prefill 优先 | 1 | 1–8 | +3.0 ~ +4.9 | +4.0 ~ +5.6 | +6.0 ~ +6.9 | +7.7 ~ +8.2 | +8.9 ~ +9.4 |

p50 在 PD 中位 +1.1 %（≤ 0.7 负载 ≤ 1.8 %）。0.85 / 0.95 负载的正误差（闭式偏保守）与上限 1 相同——上限 1 是精确 M/D/1，所以这是 12000 请求的 DES 在近饱和处低估尾部（弛豫时间长），不是批模型的误差；p99 同理（+13 ~ +55 %）。prefill 优先的 +4 ~ +12 % 来自残余项（p90 加一整步 decode）「保守」，0.64 即如此。

**(b) decode 受限**（out 512，SLO 80 / 100 ms，负载 0.6 = 1.735 req/s，3 seed × 6000 请求）：PD p50 / p90 / p99 = 71.3 / 85.4 / 137.6 vs DES 71.3 / 84.6 / 137.9 ms；prefill 优先 40.8 / 43.4 / 76.2 vs 40.1 / 44.3 / 71.6。0.64：PD p90 79.4（上限 2，−6 %）。

**(c) SLO 容量**（req/s）：

QN 输出 16、SLO 300 / 100 ms（prefill 受限），强制上限；闭式用基函数（不重选上限），DES 为 `pdsim.slo_rate`（4000 请求、2 seed 平均，0.65 稳定判据）：

| CV | 上限 | 0.65 闭式 | DES | 0.65 误差 |
|---|---|---|---|---|
| 0 | 1 | 10.000 | 10.036 | −0.4 % |
| 0 | 2 | 20.242 | 20.451 | −1.0 % |
| 0 | 4 | 36.090 | 36.360 | −0.7 % |
| 0 | 自动 | 36.666 | 37.599 | −2.5 % |
| 1 | 1 | 9.765 | 9.868 | −1.0 % |
| 1 | 2 | 17.871 | 17.954 | −0.5 % |
| 1 | 4 | 28.001 | 27.928 | +0.3 % |
| 1 | 8 | 31.500 | 31.146 | +1.1 % |
| 1 | 自动 | 31.500 | 31.181 | +1.0 % |

QN decode 受限（负载 0.6，3 seed × 6000）：PD SLO 速率闭式 1.591（0.64：2.564）对 DES ≈ 1.60；SLO 吞吐闭式 271.33 对 DES 273.17（−0.7 %）。合并副本 prefill 优先（自动上限，DES 1 seed，绝对漂移判据）：CV 0 闭式 24.832 对 DES 25.781（−3.7 %，保守）；CV 1 闭式 24.529 对 DES 22.500（+9.0 %，乐观；单 seed，旧判据 2 seed 为 22.42 / 24.34，seed 间差 ±8 %）——prefill 优先 SLO 容量按「±10 %」看。上限 8 · CV 0 闭式 36.663 对 DES 35.752（+2.5 %）。近饱和（负载 ≥ 0.85）有限长 DES 的 p90 / p99 尾部偏低，闭式在那里偏保守（精确的上限 1 M/D/1 也一样）。

### 24.3 DES 稳定性
0.51–0.64 的 DES 把「测量窗口内请求全部完成」当作稳定——过载的有限运行也会完成，DES SLO 速率在饱和以上没有意义（0.64 的 prefill 优先 DES「速率」23–44 req/s）。现在 `pdsim.stability`：预热（前 `warmup` 个请求）丢弃后，稳定 ⇔ 完成，且 PD prefill 忙时 < `UTIL_MAX` = 99.5 %（合并副本只要有 decode 就忙，不判利用率），且测量请求的 TTFT 与 TTFT 之后时间（decode + 槽位 / KV 等待）按到达顺序分 5 个窗口，最小二乘线首尾差 / 均值的绝对值 ≤ `DRIFT_TOL` = 0.5。积压增长使两者线性上升；有限到达流结束后积压排空，最后几个窗口反而变快（prefill 优先：decode 先被饿着、到达停止后放行，漂移为负）——开发中先只判向上，prefill 优先在 2–3 倍过载时被判稳定，改为绝对值。`slo_rate` 的稳定上限、`compare` 的行（`stable_des`、`drift_ttft`、`drift_post`）、`--pd-sim` 的结果（`stable`）都用它。
例：Qwen3-Next PD（decode 受限，稳定 2.89 req/s）在 1.3 倍时全部完成，但 TTFT 之后漂移 +1.13 → 不稳定；DES 无 SLO 的 `slo_rate` 落在闭式稳定速率的 0.8–1.15 倍内（测试）。

### 24.4 EP 空闲 rank 的 SRAM / 链路 / 反量化
`evaluate` 汇总每级的专家部分 `StageResult.exp_acts` = {专家 GEMM 的 SRAM 端口字节、专家权重反量化向量秒、EP dispatch / combine / expert all-reduce 占本级集合通信字节的份额}。能耗里空闲 rank（EP 下没有序列的 DP rank）只计这些：SRAM 与反量化为专家部分，链路 / D2D / 网络按份额（同层级划分「近似」）。0.64 按忙碌 rank 的整步计。例：DP2·EP2 B1 decode 向量计数 MiniMax-M1 −8.4 %、Qwen3-30B-A3B-FP8 −49.3 %、Qwen3-235B-A22B-FP8 −48.7 %；Qwen3-30B-A3B DP4·EP4 B1 的 SRAM 计数按 1 个忙 rank + 3 个专家部分（测试）。batch ≥ dp 时不变。

### 24.5 输出驻留分块：所有整数行块
对每个不同的行遍数 r_W = ⌈m/bm⌉，取达到它的最小 bm = ⌈m/r_W⌉（列块 bn = ⌊budget/(4·bm)⌋ 最宽），沿 O(√m) 个不同的 ⌈m/r⌉ 走一遍，就是输出驻留嵌套在整数块下的精确最优（与逐个 bm 穷举一致，性质测试）。0.64 只搜 2 的幂与 m。可行集仍随预算增长，流量随 SRAM 不增。

### 24.6 数值变化（0.64.0 → 0.65.0 指纹）
- 单节点 0 / 8136；fabric 0 / 126758。
- 多节点 12 / 5859：prefill 优先稳定速率 −0.34 % ~ −0.58 %（忙时份额），SLO 速率 +0.00 % ~ +0.02 %；PD（上限 1）不变。
- fp63 75 / 5854：DRAM 计数（24.5）full −1.6 % ~ −5.3 %（CogVideoX / HunyuanVideo 等）、prefill −0 % ~ −7.4 %；向量计数（24.4）decode −8.4 % ~ −49.3 %、prefill −0 % ~ −2.8 %；PD SLO 速率 Qwen3-Next 2.564 → 1.591（命中 0）、2.892 → 1.754（0.4）；prefill 优先速率 −0.1 % ~ −7.5 %。步时 / MAC 不变。
- fp64（PD TTFT 80 行，1920 值）：PD 上限变化 56 行（1 → 2 四十行、1 → 8 七行、2 → 8 五行、1 → 4 四行；贪心批在多数负载下上限越大均值越小），PD 均值 −7.2 % ~ +0.2 %、p90 −20.1 % ~ +36.0 %（Qwen3-Next 负载 0.85：79.4 → 102.6 ms，上限 2 → 8，原值乐观）、p99 −18.1 % ~ +24.2 %；Qwen3-8B TP2 B64 CV1 p90 993.5 → 793.9 ms（上限 1 → 2）。prefill 优先 52 行换上限，p90 −19.6 % ~ +0.9 %；Qwen3-Next 负载 0.95 的 4 行由「cap 4 稳定」变为不稳定（DES：cap 4 / 8 / 64 在 λ 2.75 时 TTFT 之后漂移 +1.4、prefill 忙 100 %，与闭式一致）；SLO 速率 −7.3 % ~ +17.9 %，稳定速率 −7.5 % ~ 0。

### 24.7 未做 / 剩余
- 能耗仍按选中上限的满批 Result 计（部分批的能耗未分摊）。
- 批墙钟律 k > 8 忽略偏度；非 2 的幂的 k > 8 线性插值。
- 近饱和（≥ 0.85）的 DES 尾部需要更长的运行才能判定闭式偏保守的程度。

### 24.8 0.65.1：V4 网格重跑与批服务律原子数
- **重跑**：`scripts/v4_serving.py --slo --n-high 20000 --partial DIR`，0.65.1 闭式 + 0.65 DES 稳定判据（绝对漂移），30 点 × 3 seed，load 0.85 每 seed 20000 请求、其余 3000，SLO 二分 1500；box 只有 15 GB 内存，单进程顺序跑，每点写一个 JSON（中断后续跑），各点耗时合计 20395 s（CV 1 的点 15–30 分钟）。30 点的三种模式 DES 全部稳定。
- **重跑发现的错误（已修）**：0.65.0 把每请求墙钟律压到 12 个原子、批墙钟律 D_k 压到 6 个、他人均值律 4 个。长度 CV 1 时服务律尾部被削平，批服务嵌入链的等待（依赖服务律的高阶矩）偏小，上限 2 显得比上限 1 好（0.65.0 的 18 个 dense8b 点：PD TTFT p90 最差 −27 %、SLO goodput −27 %、prefill 优先 SLO goodput −27 %，都是乐观）。原子数改为 24 / 48 / 16、自身律 48 后收敛（再加不变）：Qwen3-8B load 0.3 CV 1 强制上限 2 的 TTFT 均值 / p90 / p99 176.5 / 567.3 / 976.7 → 188.6 / 681.5 / 1150.4 ms（DES 703 / 1236）；均值高于上限 1（185 ms），闭式改选上限 1，与 DES 的较优上限一致。
- **误差**（(闭式 − DES) / DES，SLO goodput 符号相反，> 0 都表示保守；括号内为 0.57 网格）：

| 模式 | 指标 | 最小 | 中位 | 最大 | 平均 \|误差\| | 落在容差内 | DES 噪声中位 |
|---|---|---|---|---|---|---|---|
| PD | ttft_p50 | -13 %（-13） | -0 %（-0） | +2 %（+4） | 3（3） | 100 %（100） | 4 %（4 %） |
| PD | ttft_p90 | -6 %（-6） | -0 %（-0） | +5 %（+6） | 2（2） | 100 %（100） | 4 %（4 %） |
| PD | ttft_p99 | -6 %（-6） | +0 %（+0） | +6 %（+7） | 2（3） | 100 %（100） | 9 %（9 %） |
| PD | tpot_mean | -6 %（-6） | -0 %（-0） | +2 %（+2） | 1（1） | 100 %（100） | 3 %（3 %） |
| PD | tpot_p90 | -7 %（-7） | -1 %（-1） | +2 %（+2） | 2（2） | 100 %（100） | 3 %（3 %） |
| PD | itl_max | -23 %（-23） | -3 %（-4） | +11 %（+11） | 7（7） | 100 %（100） | 6 %（6 %） |
| PD | slo_goodput | -2 %（-1） | +4 %（+3） | +16 %（+16） | 7（6） | 100 %（100） | 6 %（6 %） |
| 合并 prefill 优先 | ttft_p50 | -12 %（-12） | +0 %（+0） | +7 %（+7） | 2（2） | 100 %（100） | 0 %（0 %） |
| 合并 prefill 优先 | ttft_p90 | -11 %（-11） | -0 %（+0） | +8 %（+8） | 2（2） | 100 %（100） | 2 %（2 %） |
| 合并 prefill 优先 | ttft_p99 | -3 %（-3） | +1 %（+1） | +7 %（+7） | 2（2） | 100 %（100） | 2 %（2 %） |
| 合并 prefill 优先 | tpot_mean | -6 %（-6） | -2 %（-2） | +3 %（+3） | 2（2） | 100 %（100） | 3 %（3 %） |
| 合并 prefill 优先 | tpot_p90 | -15 %（-15） | -6 %（-6） | -1 %（-1） | 6（6） | 100 %（100） | 4 %（4 %） |
| 合并 prefill 优先 | itl_max | -26 %（-26） | -3 %（-3） | +8 %（+8） | 6（5） | 100 %（100） | 4 %（4 %） |
| 合并 prefill 优先 | slo_goodput | -12 %（-11） | -1 %（-1） | +6 %（+5） | 4（4） | 100 %（100） | 6 %（6 %） |
| 合并分块 | ttft_p50 | -4 %（-4） | -0 %（-0） | +2 %（+2） | 1（1） | 100 %（100） | 2 %（2 %） |
| 合并分块 | ttft_p90 | -9 %（-9） | -4 %（-4） | +1 %（+1） | 4（4） | 100 %（100） | 4 %（4 %） |
| 合并分块 | ttft_p99 | -23 %（-23） | -4 %（-4） | +0 %（+0） | 6（6） | 100 %（100） | 5 %（5 %） |
| 合并分块 | tpot_mean | -5 %（-5） | -1 %（-1） | +3 %（+3） | 2（2） | 100 %（100） | 3 %（3 %） |
| 合并分块 | tpot_p90 | -12 %（-12） | -4 %（-4） | +1 %（+1） | 5（5） | 100 %（100） | 4 %（4 %） |
| 合并分块 | itl_max | -2 %（-2） | +4 %（+4） | +11 %（+11） | 4（4） | 100 %（100） | 4 %（4 %） |
| 合并分块 | slo_goodput | -5 %（-5） | -2 %（-2） | +2 %（+2） | 2（2） | 100 %（100） | 6 %（6 %） |


  与 0.57 基本相同（V4 的点上闭式差别小）；PD SLO goodput 中位 +3 → +4 %、最小 −1 → −2 %；PD TTFT p90 / p99 最大 +6 / +7 → +5 / +6 %。

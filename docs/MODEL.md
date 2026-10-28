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
| L2 | `core/ir.py` `core/parallel.py` | 逐 rank 算子图；布局 PP·TP·DP·EP·ETP（LLM prefill 可加 CP，§34；视频 / 蛋白质：PP·TP·DP·SP）与 stage 划分 |
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
- 参数核对：与发布 safetensors 总量偏差 > 2% 时附注；当前 75 个发布（58 个 LLM / VLM，含 0.66 起仅供外部校核的 Llama-2-7B / 70B、Mistral-7B + 10 个视频 + 7 个蛋白质）全部在 ±0.5% 内，其中视频 / 蛋白质逐项一致（偏差 < 0.001%）。（0.62.1 更正：原写 61 = 55 + 6，是 0.41 时的计数。）

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
- 视频 / 蛋白质（无 KV 缓存的全序列前向）：布局 `PP·TP·DP·SP`，EP = ETP = 1。DP 切分一次前向的序列（视频含 CFG 的 cond / uncond 两路，即 CFG 并行）；SP 为 Ulysses 序列并行：token 按 SP 切分，注意力前后各一次 all-to-all，每 rank 对完整序列计算 `ceil(ceil(H/TP)/SP)` 个头，权重在 SP 组内复制。LLM 布局保持 SP = 1；长上下文 prefill 可用上下文并行 CP（环形注意力，§34）。

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
- ESM-2：双向编码器（RoPE、pre-LN、GELU FFN）+ MLM 头（dense + LN + 33 词表 decoder）。绝对位置表（随发布存储、推理不读）只计存储；contact head 只计参数（默认不输出接触图）。0.437 按 transformers `modeling_esm.py` 把已执行的 LayerNorm 和 erf-GELU 分开计：`nn.LayerNorm` 在 fp32 归约，`vec = token × hidden × 4`，每行一次 `rsqrt`；`gelu` 是 `x·0.5·(1+erf(x/√2))`，`vec = 元素 × 5`，`sfu = 元素`。查询在 RoPE 前乘 `head_dim**-0.5`（一次乘法）。没有 QK-norm，所以不再把 QK-norm 的 3 次元素运算算进去。GEMM 形状和 FLOPs 不变。token_dropout 的缩放取决于样本里的 mask 数，不单列。0.439：ESMFold 的语言层按发布的 ESM-2 宽度（2560）计 LayerNorm 和 erf-GELU，`ModelSpec.hidden` 仍是折叠主干的 1024。`emb_layer_norm_after` 在最后一层语言输出上计一次。`esm_s_mlp` 在去掉 `<cls>` / `<eos>` 之后的残基行上计 LayerNorm（宽 2560）和 ReLU（宽 1024）。没有 ESM-2 的 MLM 头。0.440：每个主干块按发布前向另计 11 次 LayerNorm（残基 3 次、pair 8 次）和 2 次 ReLU。recycle 的 `recycle_s_norm` / `recycle_z_norm` 每次 recycle 计一次。结构模块的 `layer_norm_s`、`layer_norm_z` 和 `linear_in` 每次 recycle 只跑一遍，循环里的 LayerNorm 和 ReLU 按 `num_blocks` 次。lddt 头是 LayerNorm 加三层线性、没有 ReLU，行数是 `num_blocks` 份残基状态，只在最后一次 recycle 之后跑一遍。逐线性 `.ew` 尾项不减。sigmoid 门、IPA 的 softplus、扭转角 clamp 和 `esm_s_combine` 的 softmax 不单列。

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

**文本编码器**：文本 transformer 的每个头部 GEMM 按 `M = 提示数 × 补齐后的 token 数`；注意力核 T5 / umT5 为双向，Llama / Qwen / CLIP 文本塔为 causal。0.436 把已经发布、并且和现成配方一致的 RMSNorm、LayerNorm、SiLU，以及 SiLU 门和 up 投影的一次相乘，按向量另计。范数在 fp32 归约，`vec = 行 * 宽 * 4`，每行一次 `rsqrt`。SiLU 是 `x * sigmoid(x)`，`vec = 元素 * 4`，`sfu = 元素 * 2`，dtype 跟检查点，不跟请求里的 GEMM 激活格式。这些向量的 FLOPs 是 0，GEMM 尾项仍是每个输出元素 3 次，所以上表的 TFLOP 不变。0.437 按 Wan-Video/Wan2.1 `wan/modules/t5.py` 的 `T5LayerNorm` 把 umT5 的 RMSNorm 分开计（检查点没有 config，norm 权重是 4096 的向量，24 层加最终 norm）。门控是 tanh-GELU，不单列。Qwen3.5 文本的 RMSNorm 乘的是 `1 + weight`，T5 的 `gelu_new` 和 CLIP 的 `quick_gelu` 仍留在尾项里。token 数按参考实现的补齐长度：Wan 512、CogVideoX 226、HunyuanVideo Llama 351（模板 95 + 256）与 CLIP 77、LTX 128、Mochi 256、Open-Sora 300、H3 512「假设」。提示数 = batch × CFG（参考实现为 uncond 分支编码负向 / 空提示），Open-Sora（学习到的 null 嵌入）与 guidance 蒸馏模型（HunyuanVideo、H3）为 batch。HunyuanVideo 取倒数第 3 层、H3 取第 50 层隐状态，但参考前向跑满全部层：按全部层计。H3 的 text_encoder 带视觉塔（0.60B）与 lm_head（0.78B），随 pipeline 加载计入存储、不计算。DeepFloyd T5 只有 `.bin`：形状用同构的 T5 v1.1 XXL（CogVideoX 头），dtype 由文件总大小判定为 fp32。

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

**表示与 GEMM 行数**：结构模型同时维护几种网格——单一表示（N 个残基 / token）、pair 表示（N² 个位置）、MSA 表示（行数 S × N）、模板（T × N²）、原子（A ≈ 8N）与原子对（局部窗口 A/32 × 32 × 128）。每个权重 GEMM 按名字归到一种网格，行数即该网格大小（AF2 的 `msa_att_row.linear_z` 是 pair 行，`outer_product_mean.linear_out` 是 pair 行，模板 pair 栈是 T·N² 行……）；每个线性层另计 LayerNorm / 门控 / 偏置的逐元素工作（「假设」每输入、输出元素各 4 次）。0.441：OpenFold 与 AlphaFold 2 把发布前向里已经执行的 LayerNorm 和 ReLU 按现成配方另计。LayerNorm 在 fp32 归约，`vec = 行 × 宽 × 4`，每行一次 `rsqrt`。ReLU 是 `max(x, 0)`，`vec = 元素`，`a_fmt` 跟激活 dtype。逐线性 `.ew` 尾项不减。0.442：Boltz-1 把发布前向里的 LayerNorm（fp32 归约）和 Transition 的 SiLU 与一次相乘按现成配方另计。0.443：Protenix 把发布前向里已经执行的 plain LayerNorm、Transition 的 SiLU 与一次相乘、以及原子编码器的 ReLU 按同一套现成配方另计。AdaLN、扩散 SwiGLU、sigmoid 门、mask 乘、OPM 的除法仍只用这份 `.ew` 假设。mask 乘、sigmoid 门、IPA 的 softplus 和扭转角 clamp 不单列。

**激活 × 激活的核**（按模块名检测，头数 / 维度取自张量形状）：
- 三角乘法（outgoing / incoming）：c 个 N×N×N 批矩阵乘，`2·N³·c` FLOPs；
- 三角注意力（起点 / 终点）：N 组、每组 N 个 query 对 N 个 key，带 pair 偏置：`2·2·N³·h·d`；
- MSA 行注意力（S 组 × N 对 N，pair 偏置）、列注意力（N 组 × S 对 S；AF2 extra MSA 为全局列注意力，每列 1 个 query）；
- 外积均值：`(N·c) × S × (N·c)` 批矩阵乘（`2·S·N²·c²`，c = 32 时是 Evoformer 中最大的单项之一）；
- AF3 类 MSA 模块的 pair 加权平均（每头 N×N 权重乘 S·d 的值）；
- 带 pair 偏置的单一表示注意力（Pairformer、扩散 transformer、ESMFold 主干）；
- IPA（不变点注意力）：qk 维 = 标量 + 点坐标，v 维 = 标量 + 点 + pair 值（AF2：12 头，qk 16 + 12，v 16 + 24 + 128）；刚体更新、扭转角、坐标重建的几何运算未计；
- 模板点注意力（每个 pair 位置对 T 个模板）、原子局部窗口注意力（32 query × 128 key）。

**每请求的执行次数**（评估器把一层的算子和乘以执行次数；算子本身是一次执行的）：主干（嵌入、模板栈、extra MSA 栈、Evoformer / Pairformer、AF2 / ESMFold 的结构模块）每遍都重跑，遍数 = recycle + 1。ESMFold（0.440）以及 OpenFold / AlphaFold 2（0.441）的结构模块输入投影和进入 IPA 循环前的两条 LayerNorm 每次 recycle 只跑一遍；IPA 循环仍是 8 次共享权重。扩散模块（原子编码器 → token transformer → 原子解码器）每步一次，样本在 token / 原子网格上成批（行数 × 样本数），pair 条件对样本共享（参考实现按一次计）；置信度头每样本一次（全部网格 × 样本数）；输出头一次。ESM-2 语言模型（ESMFold）每请求一次。OpenFold / AlphaFold 2 的 pLDDT 读最终 single，不乘 IPA 次数。

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
| 节点内链路 | 每 rank 在节点内 scale-up 层**实际上线**的字节（0.78：按所选算法的物理流量——环形 all-reduce 2(n−1)/n × 载荷、all-gather / reduce-scatter (n−1) × 分片、all-to-all (n−1)/n × 载荷、tree 2(n−1)/n、分层按各层体积、in-network 顶层每 rank 一个分片、协议开销 ÷ 效率；PP 交接 = 载荷；VAE tile all-gather） | `pJ_bit_link` |
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
- **dip（硬件替换 + 防抖插入，0.258，§204.3）**：放得下 → 同 lru 全部命中；放不下 → 每类读按 (K − ε(T − K)) / T 命中（DIP / BIP，Qureshi et al. ISCA 2007，ε = 1/32，「来源」）。
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
- **能耗**（0.78 起）：链路能耗按**物理上线字节**计——每次集合通信按所选算法（ring / tree / hier / in-network / direct）与各层（`d2d` → `pJ_bit_d2d`、节点内 `link` → `pJ_bit_link`、跨节点 `net` → `pJ_bit_net`）的实际发送字节累计（`StageTime.wire_d2d / wire_link / wire_net`）。0.77 及以前为载荷口径（按份额拆分，三者之和 = 载荷），低估环形 all-reduce 至多 2×、高估 all-to-all。`link_bytes / d2d_bytes / net_bytes` 仍为载荷口径，仅用于计时与 PD KV 传输份额，数值不变。
- **入口**：API scenario `d2d_enabled`、`package_cards`、`d2d_std`、`d2d_units`、`d2d`、`link`、`node_cards`、`net`；每级 `link_GB`（total / d2d / scaleup / net）；`/api/catalog` 给出 `d2d_standards`。CLI `--d2d --package-cards（> 1 隐含 --d2d） --d2d-std --d2d-units --d2d-GBps（隐含 custom） --d2d-alpha-us --link-GBps --link-alpha-us --node-cards --net-GBps --net-alpha-us --pJ-bit-net`，`accel-dse d2d` 列出档位。Web 并行组「三层互连」：D2D 芯粒堆叠开关（关 = 单片大 die）→ 档位 / 每封装 die 数 / 单元数 / GB/s / α；节点内两项；每节点卡数与跨节点两项。扫描可选 `package_cards`（> 1 时自动开 D2D）、`d2d_units`、`d2d.GBps`（自动切到自定义）、`d2d.alpha_us`、`node_cards`、`net.GBps`、`net.alpha_us`。
- **不做 / 待定**（0.48 时；0.59–0.61 已补拓扑 `fabric.net_topology`、超额订阅 `oversub`、网内归约 `fabric.innet_reduce`，见 §19；拥塞控制 / ECMP / incast 仍不建模）：拓扑（胖树 / torus / 轨道优化）、拥塞与超额订阅、in-network reduction（SHARP）、多路径 / 链路聚合、NIC 与 scale-up 之间的 PCIe 瓶颈都不建模；`Link.topology` 仍只是标签；D2D 的协议效率（flit 开销）不扣——需要时用自定义档位填有效带宽。

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

- **两个池**：decode 池用场景的 `layout` 与 `serving.batch`，卡数 `pd.decode_cards`（布局卡数的整数倍；0 = 一个副本）；prefill 池用 `pd.prefill_layout`，卡数 `pd.prefill_cards`，每个副本取满足 TTFT SLO 的最大 prefill batch（2 的幂 ≤ 64，与合并 goodput 同一选择）。两池同芯片、同存储器、同互连设定（0.50 时；0.53 起可设异构 prefill 池 `pd.prefill_chip` / `pd.prefill_mem_id`，§18.3）。
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
- **不做**（0.52 补了长度分布、前缀缓存、池布局搜索，见 §18.2；其后 0.55–0.56 补了 KV 容量排队 / 抢占 / 换出 §18.5–18.6，0.53 补了异构池）：抢占 / 换出 / KV 容量引起的排队、调度器开销、多级 SLO 调度、分块 prefill 与 PD 的组合（PD 的 prefill 池内部分块）、异构池；分位数相加而不是卷积（偏保守）；M/D/1 尾是渐近式，ρ 很小时略偏保守。

### 18.2 请求长度分布、前缀缓存、池布局搜索（0.52，默认关）

默认（长度固定、无前缀缓存、不搜布局）走 0.50 / 0.51 的原代码路径，数值不变；唯一的例外是下面的分块 prefill 修正。

- **长度分布**（`core/lengths.py`，「假设」）：`pd.prompt_cv` / `pd.out_cv` > 0 → 独立的对数正态 prompt / 输出长度，均值 = `serving.prompt` / `serving.out_len`，各离散成 N 档等概率（N = 15，0.56；0.52–0.55 为 8，见 §18.6），每档取条件均值 E[X | 第 k 档] = N · 均值 · (Φ(z_{k+1} − σ) − Φ(z_k − σ))，σ² = ln(1 + CV²)（对对数正态精确；均值保持，档内方差丢掉，所以实际 CV 略低：名义 1.0 → 0.90（8 档时 0.85），报告里给 `prompt_cv_eff` / `out_cv_eff`）。或者 `pd.length_mix` 直接给离散联合分布 [权重, prompt, 输出] 至多 16 行（可表达 prompt 与输出的相关），此时替代上面两项与 `serving.prompt / out_len`。
  - **prefill 每个 prompt 值单独求值**（注意力不是 S 的线性函数），不假设线性；排队从 M/D/1 换成 **M/G/1**（离散服务分布；P-K 均值精确，尾用同一 Cramér–Lundberg 渐近：θ 为 λ(E[e^{θτ}] − 1) = θ 的正根，C = (1 − ρ) / (λE[τe^{θτ}] − 1)；只有一个值时就是 M/D/1；测试用三点分布的 Lindley 仿真核对 p90 / p99）。batch 上限 b 时，长度 S_i 的请求在一批混合请求里耗时 τ_i + (b − 1)·τ̄。TTFT 分位 = prefill 等待分位 + 「自身 prefill + 暴露的 KV」的离散分位（两者都随 S_i 单调，按同一分位相加是精确的）+ KV 队列等待分位。KV 交接按每档的 KV 大小也是 M/G/1。
  - **decode**：Little 定律用 E[out]；运行 batch 的 Poisson 分布对输出长度分布不敏感（M/G/∞），所以 TPOT 分位的做法不变；槽位等待的 Allen–Cunneen 因子换成 (1 + c_s²)/2（c_s 为输出长度的 CV）。**长度偏置**：正在 decode 的请求按其输出长度被「看到」，时间平均上下文 = E[out·(S + out/2)] / E[out]；`serving.ctx` 被解读为一个代表性定长请求的上下文，再乘以它与 (E[S] + E[out]/2) 之比（定长时 = 1）。输出越分散、长 prompt 与长输出越相关，decode 上下文越长、步越慢。
  - **流体容量也随之变**：prefill 池 λ_p = r_p · b / Σ w_i·TTFT(b, S_i)（b 为按代表长度 S = round(E[S]) 选出的 prefill batch），decode λ_d = r_d · R_d(ctx_eff) / E[out]，KV λ_kv = min(N_p, N_d)·β / E[KV]；合并对照同样用逐档的每请求 prefill 时间。表里显示的 TTFT 是代表长度请求的。负载 `pd.load` 乘的就是这个容量。每个模式另给 `stable_rate_rps`（排队模型的稳定上限）。
- **前缀缓存**（`pd.prefix_hit` = h，0–0.99，「假设」）：每个请求前 ⌊h·S⌋ 个 token 已在 KV 缓存中。评估器新增 `serving.prefix_cached`（单点也可用，API / CLI `--prefix-cached`）：prefill 走 Phase(prefill, q = S − p, ctx = p)，即 GEMM 只算新 token、注意力对缓存前缀做（因果系数 ½(1 + p/S)）、读前缀 KV；吞吐仍按整个 prompt 计（请求率 = 吞吐 / S）。PD 的 KV 交接：`pd.prefix_on_decode`（默认 true，decode 池也持有同一前缀，如共享系统提示词 / 会话粘性路由）只传未缓存部分 KV ×(1 − p/S)，否则全量。分块 prefill 的块数 ⌈(S − p)/C⌉，每块重读缓存前缀 + 已完成的块。decode 不变（每个请求仍对完整上下文做注意力；跨请求共享前缀的 KV 读合并未建模）。命中率是输入：缓存容量、淘汰、路由命中都不建模。
- **池布局搜索**（`pd.search_layouts`，opt-in）：prefill 与 decode 池各自枚举每副本 1 / 2 / 4 / … / 64 卡（≤ 总卡数；0.60 起总卡数 > 64 时再加 128、256 … ≤ 总卡数，截断时每种副本规模等额）与当前布局的所有布局（PP 小的优先，每池至多 64 个，超出标 `truncated`），× 同总卡数的所有切分。decode 每副本 batch 仍为 `serving.batch`，prefill batch 按 TTFT SLO 选。流体 goodput / 卡对全部布局对排序（同分时 prefill 余量大、TTFT 低、TPOT 低者在前）；排队模型下的 SLO goodput 只对：流体前 4 对、每种 decode 布局与每种 prefill 布局各自的最佳一对、当前布局，至多 24 对计算（流体排名会掩盖 TPOT / TTFT 差异）。输出 `layout_search.{rows, best_fluid, best_slo, current_layouts}`。
- **分块 prefill 修正（0.52 Fixed）**：0.51 的 decode 平均步时用了时间平均 ρ·T₁ + (1 − ρ)·T₀，带块迭代占比也按 ρ，二者都偏向长迭代（长度偏置）。每 token 的 TPOT 应是每次迭代的平均：带块迭代每秒 ν = λ·Σ w_i n_i，占时 ρ = λ·Σ w_i n_i T₁,i，平均迭代 = T₀ / (1 − ρ + ν T₀)，带块迭代占比 x = ν × 该值。块数改为 ⌈(S − p)/C⌉（0.51 的 ρ 用 S/C）。
- **量级**（同 §18.1 例子：Qwen3-8B 1P + HBM3E TP2 batch 64，prompt 4096 / out 512，PD 2 + 6，SLO TTFT 400 / TPOT 10 ms，load 0.8）：
  - 分块修正：合并 · 分块 TPOT 均值 / p90 由 9.4 / 12.2 ms 变为 5.4 / 7.4 ms，TTFT p90 220 → 177 ms，SLO goodput 433 → 511 tok/s/卡（略高于 prefill 优先的 506；PD 474）。PD 与 prefill 优先不变。
  - prompt / 输出 CV 1.0（离散后 0.85）：流体 577 → 556 tok/s/卡（decode 上下文 4096 → 4271），λ = 6.95 req/s 时 PD TTFT p90 216 → 1626 ms，prefill 优先 651、分块 928 ms，三者 SLO goodput 均为 0（长 prompt 档拉高 p90）。
  - 离散分布 70 % (1024, 256) + 30 % (11264, 1109)（均值仍为 4096 / 512，但长 prompt 也长输出）：decode 上下文按偏置 7609，流体 577 → 325 tok/s/卡。
  - 前缀命中 0.5：prefill 池容量 13.0 → 20.8 req/s（同一 TTFT SLO 下 batch 4 → 8），KV 每请求 576 → 288 MiB；PD 仍是 decode 瓶颈，流体不变；合并 625 → 672 tok/s/卡；TTFT p90 PD 216 → 94、prefill 优先 114 → 53、分块 177 → 92 ms；SLO goodput prefill 优先 541、分块 552、PD 474。
  - 布局搜索（定长）：流体最佳是 decode 改 TP1（583 tok/s/卡），但 TPOT 变长，SLO goodput 只有 412；SLO 最佳仍是 decode TP2（474）。长度分布 + 前缀 0.3 时，SLO 最佳是 prefill TP4 × 4 + decode TP1 × 4（132 tok/s/卡），只搜切分时为 117。
- **入口**：scenario `pd.prompt_cv / out_cv / length_mix / prefix_hit / prefix_on_decode / search_layouts`、`serving.prefix_cached`；响应 `pd.lengths`（来源、离散点、均值、有效 CV、`ctx_ratio`、`decode_ctx`、前缀）、`pd.layout_search`、每个模式的 `stable_rate_rps`、分块的 `chunk_share`；CLI `--pd-prompt-cv --pd-out-cv --pd-mix w:S:out,… --pd-prefix-hit --pd-prefix-not-on-decode --pd-search-layouts --prefix-cached`；Web PD 输入组「请求长度与前缀缓存」与「池布局搜索」表。
- **不做**（0.53 补了缓存容量 / LRU 淘汰、异构池、布局搜索里的 decode batch，见 §18.3；radix 部分前缀匹配 0.58 已补，§18.8）：长度感知调度（全部 FCFS）、prompt 与输出长度之外的请求异质性、decode 端共享前缀的 KV 读合并、PD prefill 池内分块、部分前缀匹配（radix tree）。

### 18.3 前缀缓存容量 / LRU 淘汰、异构池（0.53，默认关）

默认（`prefix_len` = 0、两池同一芯片 / 存储器）走 0.52 代码路径，数值不变；显式 `pd.prefix_hit` > 0 时覆盖容量模型（与 0.52 相同）。

- **前缀缓存容量**（`core/prefixcache.py`，「假设」）：工作集是 N = `pd.prefix_count`（默认 1000）个共享前缀（系统提示词 / few-shot 模板 / 多轮历史），每个长 `pd.prefix_len` tokens，按 Zipf(α = `pd.prefix_zipf`，默认 1；0 = 均匀) 流行度被独立请求（IRM）。缓存是整前缀 LRU，容量 K = 能放几个前缀；命中率用 **Che 近似**（Che / Tung / Wang 2002；Fricker / Robert / Roberts 2012）：特征时间 T 满足 Σ_i (1 − e^{−p_i T}) = K，条目 i 驻留概率 h_i = 1 − e^{−p_i T}，请求命中率 H = Σ_i p_i h_i。K ≥ N → H = 1（稳态，忽略冷启动）；K = 0 → H = 0。N ≤ 4096 精确求和；更大 N 前 1024 名精确、尾部按几何 rank 分箱（比 1.01，中点积分质量）——相对精确解误差 ≪ 10⁻⁵，与离散 LRU 仿真（IRM，热身后）误差 < 1 %。
  - **容量**：每前缀足迹 = 该长度下的 KV + indexer + 循环状态（与 memplan 一致，计 TP 分片与 DP 组）；每副本 K = floor(容量 / 足迹)。容量默认 = 权重 + 活跃 batch KV + 运行时预留之后剩余的 DRAM（各流水级取 min 再 × DP），或 `pd.prefix_cache_GB`。prefill 池与 decode / 合并副本各自算一遍（batch 不同 → 剩余不同）。
  - **路由**：随机（默认）→ 每个副本各自缓存全体前缀（H 用每副本 K）；`pd.prefix_affinity` → 前缀感知路由，副本分摊前缀（H 用总容量 K·replicas）。decode 池在 prefill 命中时也持有该前缀的概率 = Σ_i p_i h_i^P h_i^D / H_P（IRM 下独立 LRU）；KV 交接只传未缓存部分的期望比例。
  - **队列 / 流体**：命中 / 未命中两类请求按权重混合进 pts（PD prefill 用 H_P，合并用 H_C），其余与 0.52 的 `prefix_hit` 相同。显式 `pd.prefix_hit` > 0 时关掉容量模型并告警。
- **异构池**：`pd.prefill_chip`（芯片预设名或对象）与 `pd.prefill_mem_id`；None = 与 decode 池（scenario）相同。合并对照始终用 decode 池的芯片 / 存储器。闲置功率按卡计；0.56 起 PD prefill 池可另给 `energy.idle_W_prefill`（§18.6），动态能耗两池仍用同一张能耗表「假设」。
- **布局搜索 · decode batch**（`pd.search_decode_batch`，与 `search_layouts` 联用）：每种 decode 布局在 {B/2, B, 2B, 4B} ∩ [1, max(4096, 64 × 每副本卡数)]（0.60 起上限随副本卡数放大） 中取满足 TPOT SLO（满批流体 TPOT）的最高吞吐；SLO goodput 用该 batch。异构内存时更有用。
- **量级**（同 §18.1：Qwen3-8B 1P + HBM3E TP2 batch 64，prompt 4096 / out 512，PD 2 + 6，SLO 400 / 10，load 0.8；前缀长 2048）：
  - N = 1000（剩余 DRAM ≈ 600 GB / 副本，足迹 302 MB → K ≈ 1900）：H = 100 %，与显式 `prefix_hit` ≈ 0.5（p = 2048）同量级——合并 672 tok/s/卡、TTFT p90 PD 94 / 优先 53 / 分块 92 ms、SLO 474 / 541 / 552。
  - N = 10⁵、α = 1：H_P = 57 %、H_D = 57 %（随机路由）；合并 651、p90 157 / 104 / 157、SLO 474 / 528 / 528。亲和路由 → H_D 68 %、H_C 71 %、合并 658。α = 0.6（更平）→ H ≈ 9 %，收益几乎消失。
  - 给定容量：10 / 50 / 200 GB → H = 19 / 34 / 47 %，合并 goodput 634 / 640 / 646；命中率随容量单调升。
  - 异构：prefill 改 100T → 流体 577 → 154 tok/s/卡（prefill 瓶颈），PD SLO goodput 0；prefill 只换 LPDDR5X（同 1P）→ 本例仍够 TTFT，与 baseline 相同。布局搜索 + decode batch：LPDDR5X decode、TPOT SLO 50 ms 时，SLO 最佳由 batch 64 改为 32（同 91 tok/s/卡）。
- **入口**：`pd.prefix_len / prefix_count / prefix_zipf / prefix_cache_GB / prefix_affinity / prefill_chip / prefill_mem_id / search_decode_batch`；响应 `pd.prefix_cache`（策略、容量、足迹、K、命中率、来源）、`pd.lengths.prefix_hit_source`（`capacity` / `explicit` / `off`）、`pd.prefill.{chip,mem_id,hetero}`；CLI `--pd-prefix-len --pd-prefix-count --pd-prefix-zipf --pd-prefix-cache-GB --pd-prefix-affinity --pd-prefill-chip --pd-prefill-mem --pd-search-decode-batch`；Web「前缀缓存容量」「异构池」输入与「前缀缓存容量」表。
- **不做**（0.53 时；部分前缀匹配 radix 0.58 已补 §18.8，按池闲置功率 0.56 已补 §18.6；块粒度仍不建模，radix 的节点粒度 0.81 计入）：部分前缀匹配（radix / 块粒度）、缓存预热与淘汰代价、队列模型更大 batch 对剩余 DRAM 的反馈、按芯片区分的闲置功率、PD prefill 池内分块、decode 端共享前缀的 KV 读合并、长度感知调度。

### 18.4 V4 服务验证：请求级 DES 对照闭式排队模型（0.54）

> **0.80 长运行复核**：数据文件 `accel_dse/data/v4_serving.json` 改为 4 倍运行长度（load < 0.85 每点 12000 请求、load 0.85 每点 80000 请求，各 3 个 seed；SLO goodput 二分每次 6000 请求，原为 1500）。闭式值与 0.65.1 起的数据逐位一致。DES 各指标的中位数移动 ≤ 1.3 %，最大移动在原 seed 噪声的 2–3 倍以内，说明原网格没有被短仿真系统性拉偏。seed 噪声中位数约降为一半到四分之一（如 PD TTFT p99 9.3 % → 3.2 %、TPOT p90 3.2 % → 1.1 %）。30 点全部仍在误差带内；全部合并的最大 |误差|：PD TTFT p99 0.11、chunked TTFT p99 0.23 → 0.12、prefill 优先 TPOT p90 0.15 → 0.14。SLO goodput 中位数：PD +3.5 % → −2.2 %、prefill 优先 −1.4 % → −3.5 %、chunked −1.7 % → −2.8 %，最大 |误差| chunked 0.05 → 0.17（仍在 0.20 带内，之前被噪声掩盖）。0.79 中 load 0.9、16k prompt 的 +36 % 不在本网格，是短 DES 的偏差（该点为纯 M/D/1，闭式 = Erlang 精确律）。重跑：`val8/v4long.py` 的做法等价于 `scripts/v4_serving.py --n 12000 --n-high 80000 --seeds 3 --slo`。

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
- **AF2 / OpenFold 模板扭转角行**：参考实现把 T 条模板扭转角行拼到 MSA 上，所有 Evoformer 块的 MSA 网格是 聚类行 + T（发布默认项 `tmpl_msa`，AF2 / OpenFold 开；不是可设的 workload 字段）；扭转角嵌入按 T × N 行；OpenFold extra-MSA 全局列注意力的 q 来自行均值（N 行）。384 残基：AF2 245.69（参考 245.74）、OpenFold 221.41（参考 221.44）TFLOP；默认 512 残基 396 → 398 / 354 → 355 TFLOP。
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

## 25. 外部校核：公开实测对照（0.66，V6）

V4 只比较模型与模型（闭式 vs DES，共用逐步代价）。V6 第一次拿**代价模型本身**去对公开发表的实测。

### 25.1 数据（`data/ext_measurements.json`，100 条，数值照抄，逐条带 URL）

| 来源 | 条数 | 内容 | 完整性 |
|---|---|---|---|
| TensorRT-LLM 0.8 performance.md（github.com/NVIDIA/TensorRT-LLM/blob/v0.8.0/docs/source/performance.md） | 36 + 18 | 静态批吞吐（out tok/s/GPU）：H100 / H200 FP8（权重 + KV）、A100 FP16；LLaMA-2 7B / 70B、Mistral 7B；batch 16 ~ 1024、TP 1 ~ 4、ISL / OSL ∈ {128, 2048}；batch 1 首 token 延迟（TP 1 / 8） | 硬件、模型配置（页面给出 ckpt_config）、dtype、batch、TP、长度全给定；A100 页面只写 "A100"，由容量推断 80 GB（70B TP4 B128 128/2048 FP16 每卡 > 40 GB） |
| TensorRT-LLM 0.21 perf-overview（nvidia.github.io/TensorRT-LLM/performance/perf-overview.html） | 44 | 满载 in-flight batching 总输出吞吐：H100 / H200，Llama-3.1-8B TP1、Llama-3.3-70B TP2，FP8，11 组 ISL / OSL | batch 由运行时决定（未给出）——模型取能装下的最大吞吐（2 的幂 batch，静态批代理） |
| Databricks "LLM Inference Performance Engineering"（2023） | 2 | A100-40GB Llama-2-70B FP16 TTFT，512 token，batch 1，TP4 / TP8 | 软件栈只说 "TensorRT-LLM-based" |

MLPerf Inference v5.0 Llama-2-70B（H200 ×8 offline 34,988 tok/s、B200 ×8 98,858 tok/s 等，docs.mlcommons.org/inference_results_v5.0/）只作背景：OpenOrca 输入长度分布未公开为数值（只有输出均值 294.45），不进误差表。B200 / MI300X / DeepSeek-V3 / 分离式（DistServe、Mooncake）尚未纳入（见 §25.6）。

### 25.2 参考硬件（`core/refhw.py`，参考硬件，不进 `CHIPS`）

| | 稠密 BF16 / FP8 TFLOPS | HBM | NVLink（单向） | 出处 |
|---|---|---|---|---|
| H100-SXM | 989.5 / 1979 | 80 GB，3.35 TB/s | 450 GB/s | nvidia.com/en-us/data-center/h100/（表中为稀疏值，取半） |
| H200-SXM | 989.5 / 1979 | 141 GB，4.8 TB/s | 450 GB/s | nvidia.com/en-us/data-center/h200/ |
| A100-SXM-40GB / 80GB | 312 / — | 40 GB 1555 GB/s / 80 GB 2039 GB/s | 300 GB/s | A100 数据手册 PDF |

等效几何：每 SM 一个 32 × (MAC/SM ÷ 32) 阵列、engines = SM 数，频率反推使 R·C·E·2·f = 数据手册峰值；L2 作 `sram_mib`；向量 lanes = FP32 核数。存储器目录没有 HBM2，A100 用总线宽度 / 速率 / 容量相同的 HBM3 编号代替（只影响能耗表）。DRAM 效率两种：目录默认 0.7（「假设」，`catalog`）与 1.0（`peak`）。**没有任何参数按实测调。**

### 25.3 怎么预测

TTFT = `evaluate(prefill, batch, prompt=ISL).ttft`；静态批吞吐 = B·OSL / (TTFT(B, ISL) + (OSL − 1)·TPOT̄) / TP，TPOT̄ 取 4 个中点上下文的平均；满载吞吐同式，对 2 的幂 batch 取装得下的最大值。FP8 行：attn / mlp 权重、激活、KV 为 FP8，embed / lm_head BF16（与 TensorRT-LLM FP8 checkpoint 一致）。

### 25.4 误差表（几何均值 预测/实测 / 平均 |log 误差|，`data/ext_validation.json`）

| 组 | n | catalog（默认） | peak | catalog + kernel | catalog + serial | peak + serial |
|---|---|---|---|---|---|---|
| TRT-LLM 0.21 满载吞吐 H100 | 22 | 1.14 / 24 % | 1.44 / 46 % | 1.01 / 22 % | 0.82 / 25 % | 0.97 / 15 % |
| TRT-LLM 0.21 满载吞吐 H200 | 22 | 1.33 / 42 % | 1.56 / 58 % | 1.04 / 23 % | 0.87 / 20 % | 0.99 / 15 % |
| TRT-LLM 0.8 静态吞吐 A100-80GB | 12 | 1.13 / 20 % | 1.38 / 38 % | 1.04 / 15 % | 0.86 / 16 % | 1.02 / 11 % |
| TRT-LLM 0.8 静态吞吐 H100 | 12 | 1.37 / 44 % | 1.67 / 67 % | 1.18 / 31 % | 1.01 / 19 % | 1.20 / 22 % |
| TRT-LLM 0.8 静态吞吐 H200 | 12 | 1.49 / 50 % | 1.77 / 77 % | 1.29 / 37 % | 1.08 / 22 % | 1.26 / 27 % |
| TRT-LLM 0.8 TTFT A100-80GB | 6 | 0.72 / 53 % | 0.67 / 63 % | 0.80 / 39 % | 1.12 / 21 % | 1.01 / 19 % |
| TRT-LLM 0.8 TTFT H100 | 6 | 0.52 / 91 % | 0.44 / 127 % | 0.60 / 68 % | 0.77 / 42 % | 0.69 / 55 % |
| TRT-LLM 0.8 TTFT H200 | 6 | 0.47 / 115 % | 0.41 / 145 % | 0.55 / 86 % | 0.73 / 48 % | 0.66 / 60 % |
| Databricks TTFT A100-40GB | 2 | 0.47 / 112 % | 0.47 / 112 % | 0.55 / 81 % | 0.73 / 38 % | 0.67 / 48 % |

（TPOT 没有单列：所有吞吐行都是解码主导或混合，TPOT 的误差体现在吞吐里；公开数据里给定 batch 的逐步 TPOT 几乎没有。）

### 25.5 发现与决定

1. **级内全重叠（默认 `stage`）是主要系统误差。** 一个流水级的计算、DRAM、链路取一个 max，相当于假设 attention 的 KV 读、下一个 GEMM 的权重读、TP all-reduce 全都藏在别的工作后面。GPU 逐 kernel 执行做不到：吞吐偏乐观，70B TP2 最明显（满载 1.6 ~ 2.3×，all-reduce 被藏）。新增 `exec_overlap`：`class`（GEMM‖权重、attention‖KV，两类与通信串行）、`kernel`（再加向量 kernel）、`serial`（完全不重叠，上界）。实测落在 `kernel` 与 `serial` 之间。**默认不改**：对设计中的 NPU，双缓冲 / DMA 预取能做到多少级内重叠本身是设计变量，默认 `stage` 应读作乐观界；对照 GPU 类逐 kernel 执行时用 `kernel` / `serial`。
2. **不引入拟合效率系数。** 留出法（`scripts/ext_calib.py`，DRAM 效率 0.5 ~ 1.0 × MAC 效率 0.4 ~ 1.0 × 四种重叠模式；按硬件 H100 → H200 / A100、按模型 7–8B → 70B、按数据源 0.21 → 0.8 / Databricks 三种切分）：

| 切分 / 模式 | 拟合 (mem, mac) | 拟合误差 | 留出误差 | 未调（目录 0.7 / 峰值 1.0）留出误差 |
|---|---|---|---|---|
| 硬件 / stage | 0.60, 0.80 | 32.1 % | 36.1 % | 47.5 % / 66.8 % |
| 硬件 / kernel | 0.60, 0.90 | 29.0 % | 30.7 % | 32.6 % / 40.7 % |
| 硬件 / serial | 1.00, 0.90 | 21.5 % | 21.2 % | 23.0 % / 21.7 % |
| 模型 / stage | 0.60, 1.00 | 28.8 % | 52.3 % | 62.8 % / 83.8 % |
| 模型 / kernel | 0.80, 1.00 | 28.2 % | 41.6 % | 35.9 % / 50.0 % |
| 模型 / serial | 1.00, 1.00 | 22.4 % | 21.5 % | 17.1 % / 21.5 % |
| 数据源 / stage | 0.60, 0.70 | 23.6 % | 43.1 % | 53.4 % / 75.9 % |
| 数据源 / kernel | 1.00, 0.80 | 19.6 % | 45.5 % | 39.5 % / 52.8 % |
| 数据源 / serial | 1.00, 1.00 | 14.9 % | 27.9 % | 25.1 % / 27.9 % |

   拟合系数只在 `stage` 下明显有用——它在补偿重叠假设，而不是测出了效率；换成结构上更对的模式后，拟合相对未调只差 ±4 个百分点，按模型 / 数据源切分甚至更差。结论：保持纯数据手册峰值 + 目录默认效率，把结构项（重叠模式）作为显式「假设」暴露。
3. **短 prompt 的固定开销。** 128 token TTFT 低估 2 ~ 5×，残差约 0.11 ms / 层（H100 / H200 TP1）、0.15 ms / 层（TP8）、0.2 ms / 层（A100）——每层 kernel 启动 / 同步一类固定开销，模型没有（NPU 上取决于调度器设计，未加参数）。2048 token 7B TTFT 在 H100 / H200 误差 −7 % ~ +4 %；A100 上模型偏慢 15 %（等效 32 × 32 × 108 几何的分块利用率偏悲观——等效几何的副作用）。70B TP8 TTFT 低估约 2×（TP 通信暴露 + 每层开销）。
4. **GQA 复用的软件依赖。** TRT-LLM 0.8 上 Mistral-7B（8 KV 头）吞吐 ≈ LLaMA-2-7B（32 KV 头），模型按 KV 少 4× 预测 Mistral 快约 2×；`peak_noreuse` 诊断变体（解码 attention 按 query 头各读一次 KV）能解释 0.8，却把 0.21 的满载吞吐低估一半（几何均值 0.46）——现代 kernel 复用 GQA，模型的复用假设保留。

### 25.6 仍未校核

B200 / GB200（FP4）、MI300X / MI325X、TPU、Gaudi；MoE（Mixtral、DeepSeek-V3/R1：EP 分发 / 合并、专家负载不均）；MLA；PD 分离与大规模 EP（DeepSeek 推理系统报告、SGLang 96×H100、Mooncake、DistServe）；跨节点网络；推测解码；KV / 前缀缓存；能耗；排队尾延迟（V4 只对 DES）。给定 batch 的逐步 TPOT 实测几乎没有公开，TPOT 只通过吞吐间接校核。参考硬件的等效阵列几何只匹配峰值，不代表 tensor core 的真实分块行为。


### 25.7 第二轮：MoE / MLA / 大规模 EP（V6.2，`data/ext_measurements_moe.json`，34 条）

数据（逐条带 URL，照抄）：
- SGLang「PD Disaggregation and Large-Scale EP on 96 H100」（lmsys.org/blog/2025-05-05-large-scale-ep/）：DeepSeek-V3，prefill EP32 / DP32（4 × 8 H100，16,384 token / 卡，ISL 1K / 2K / 4K，默认分布与模拟完美 EPLB），decode EP72（9 节点，256 请求 / 卡，KV 2,000，无 MTP）。1K / 2K 的每卡 token 数原文未给，按 4K 档的 16,384「假设」。
- DeepSeek profile-data（prefill EP32 TP1，4K，16K token / 卡，绝对均衡路由；每节点吞吐取 SGLang 博客的换算值）。decode 档用了 MTP，不入表。Day-6 线上平均值（73.7k / 14.8k tok/s/节点）长度与缓存命中混合，只作背景。
- TensorRT-LLM「Best perf practice on DeepSeek-R1」v0.21：8 × H200，`--tp 8 --ep 8` + attention DP，并发 1024，ISL 1K / OSL 2K 固定，无 MTP：11,489 tok/s。
- TensorRT-LLM 0.17 perf-overview：Mixtral-8x22B TP8，H200 / H100 FP8、A100 FP16，9 组 ISL / OSL，共 27 条。

参考硬件新增 H800-SXM：H100-SXM 同一 die（NVIDIA 没有 SXM H800 数据手册：SM / 频率 / HBM 按 H100「假设」），NVLink 400 GB/s 双向（Lenovo H800 资料 + DeepSeek-V3 报告）。跨节点：每卡 50 GB/s IB（400 Gb/s，DeepSeek-V3 报告；SGLang 集群为 IB，速率「假设」相同）。DeepSeek 行用发布格式（FP8 块缩放 GEMM，BF16 KV 与核心 MLA）。

新增 `exec_overlap="tbo"`：计算按 `kernel` 串行，集合通信藏在计算后面（双微批重叠：一个微批的 all-to-all 和另一个的计算同时跑）。stage ≤ tbo ≤ kernel ≤ serial，有测试。

误差表（几何均值 / 平均 |log 误差|）：

| group | n | catalog | catalog_tbo | catalog_kernel | catalog_serial | peak_serial_m |
| SGLang/DeepSeek large-EP prefill | 5 | 0.94 / 9% | 0.94 / 9% | 0.56 / 78% | 0.53 / 89% | 0.54 / 85% |
| SGLang EP72 decode | 1 | 1.28 / 28% | 1.27 / 27% | 0.78 / 28% | 0.63 / 58% | 0.67 / 49% |
| TRT-LLM DS-R1 H200 EP8 | 1 | 2.10 / 110% | 1.43 / 43% | 1.35 / 35% | 1.10 / 10% | 1.26 / 26% |
| TRT 0.17 Mixtral-8x22B TP8 H200-SXM | 9 | 1.21 / 28% | 1.05 / 14% | 0.80 / 25% | 0.71 / 41% | 0.75 / 33% |
| TRT 0.17 Mixtral-8x22B TP8 H100-SXM | 9 | 1.18 / 30% | 1.08 / 22% | 0.85 / 21% | 0.69 / 45% | 0.77 / 31% |
| TRT 0.17 Mixtral-8x22B TP8 A100-SXM-80GB | 9 | 1.40 / 44% | 1.35 / 40% | 1.18 / 24% | 0.93 / 9% | 1.06 / 11% |

结论：
1. **重叠方式由软件决定，不能统一拟合。** SGLang / DeepSeek 的大规模 EP 显式做双微批重叠，prefill 是 IB all-to-all 瓶颈，`stage` / `tbo` 误差 9 %；第一轮在稠密 TRT-LLM 上最好的 `serial` 这里低估约 2 ×（误差 89 %）。第一轮各切分上拟合出来的组合（serial 1.0 / 0.9、kernel 0.6 / 0.9）拿到这批数据上做留出：大规模 EP 误差 87 % / 77 %。预先定好的规则（软件做通信重叠用 `tbo`，否则用 `kernel`）在 7 条 DeepSeek 行上是 1.03 / 15 %。所以继续不引入拟合效率系数，重叠方式作为显式「假设」给用户选。
2. EP72 decode 仍高估 1.27 ×。原文 EPLB 用的是与数据匹配的分布，剩余的专家不均衡没有数值；`moe_skew` ≈ 1.3 就能解释，但没有依据，不调。
3. TRT-LLM DS-R1（单节点 attention DP，无 TBO）：`serial` 1.10、`kernel` 1.35。
4. Mixtral-8x22B TP8：Hopper 上 `tbo` 1.05 ~ 1.08 / `kernel` 0.80 ~ 0.85，A100 上 `serial` 0.93。长 prompt 档（5000 / 500、20000 / 2000）各模式都偏低：静态批代理把 prefill 和 decode 串起来算，而 in-flight batching 会把 prefill 块和 decode 交错，所以这一档代理偏悲观（§25.3）。

仍未校核：MI300X（公开数据没有完整给出 batch / 长度 / 并行配置），B200 FP4，DistServe / Splitwise / Mooncake（只有 SLO 达成率曲线，没有能照抄的数值表），跨节点 PD 的 KV 传输，MTP / 推测解码，专家不均衡的数值。


### 25.8 第三轮（V6.3，分支 r3，本地）

**(a) 连续批代理。** `extval.predict_cb`：in-flight 行改用工具自己的合并式分块模型——`pdqueue._chunk_plan`（每块 C token，默认 8192 = TensorRT-LLM max_num_tokens「假设」；vLLM / ROCm 行 131072 按原文）+ `_fused_step`（stage 模式原样调用；kernel / serial / tbo 把同样的逐级分量相加，再取该模式的 total）。满载下运行批 b：每次迭代有 x = n·b/OSL 的比例带一个 prefill 块；x > 1 时 prefill 成为瓶颈，运行批被限制在 OSL/n。b 在 2 的幂里取装得下且最快的。
| group | n | catalog static | catalog CB | catalog_tbo static | catalog_tbo CB | catalog_kernel static | catalog_kernel CB | catalog_serial static | catalog_serial CB | peak_serial_m static | peak_serial_m CB |
| all | 71 | 1.24 / 33% | 1.26 / 36% | 1.09 / 25% | 1.09 / 26% | 0.99 / 23% | 0.99 / 23% | 0.81 / 26% | 0.77 / 32% | 0.93 / 18% | 0.88 / 22% |
| prefill-heavy (ISL ≥ 4·OSL) | 21 | 0.95 / 18% | 1.07 / 17% | 0.91 / 22% | 1.01 / 18% | 0.82 / 28% | 0.90 / 22% | 0.73 / 38% | 0.76 / 34% | 0.81 / 26% | 0.83 / 25% |
| decode-heavy | 50 | 1.39 / 40% | 1.35 / 45% | 1.18 / 27% | 1.12 / 29% | 1.07 / 21% | 1.03 / 24% | 0.85 / 21% | 0.77 / 31% | 0.98 / 15% | 0.90 / 21% |

prefill 重的行变好了（kernel 0.82 → 0.90；stage 0.95 → 1.07），decode 重的行稍降，总体持平。代理换成连续批后，原来「静态批代理偏悲观」的那部分误差基本消掉；剩下的误差来自重叠方式和固定开销，不在批处理代理。

**(b) MI300X / B200。** 参考硬件 MI300X（AMD 数据手册：304 CU，BF16 1307.4 / FP8 2614.9 TFLOPS 稠密，192 GB HBM3 5.3 TB/s，Infinity Fabric 7 × 128 GB/s；256 MB Infinity Cache 不建模）与 B200（DGX B200 总量 ÷ 8，稀疏值取半：FP8 4.5 / FP4 9 PF，BF16 取 FP8 的一半「假设」，8 TB/s，NVLink5 1.8 TB/s）。数据（`data/ext_measurements_r3.json`，31 条）：AMD ROCm vLLM 0.11.2 文档（固定 commit 2026250）的 Llama-3.1-70B FP8 TP8 满载吞吐 4 条 + 静态批端到端延迟 16 条（batch 1 ~ 128，ISL 128 / 2048，OSL 2048）；TensorRT-LLM 0.21 的 B200 Llama-3.3-70B NVFP4 TP1 满载 11 条。MI325X 没找到同时给出 batch、长度、并行配置的表；GB200 是 Grace 耦合的 NVL72，跳过；MLPerf v5.x 的长度分布没有公开数值。
| group | n | catalog | catalog_tbo | catalog_kernel | catalog_serial | peak_serial_m |
| MI300X 70B FP8 TP8 max-load (CB proxy) | 4 | 4.49 / 349% | 3.11 / 211% | 2.15 / 115% | 1.87 / 87% | 2.03 / 103% |
| MI300X 70B static-batch latency | 16 | 3.67 / 267% | 3.61 / 261% | 3.27 / 227% | 2.89 / 189% | 3.60 / 260% |
| B200 70B NVFP4 TP1 max-load (CB proxy) | 11 | 1.38 / 39% | 1.16 / 22% | 1.16 / 22% | 0.85 / 18% | 0.99 / 9% |

MI300X 上不加固定开销时，模型比实测快约 3 ×：batch 1 实测 7.8 ms / token，roofline 约 2.4 ms；AMD 自己也注明这些数「不是峰值」，同一篇 AMD 博客里 MI300X 的吞吐和 H100 持平，而 HBM 带宽是 H100 的 1.6 ×。B200 NVFP4：peak + serial 0.99 / 9 %，kernel 1.16 / 22 %。

**(c) 专家不均衡（核对 `moe_skew`，不调）。** 公开统计：TensorRT-LLM wide_ep EPLB 示例（github.com/NVIDIA/TensorRT-LLM/tree/6b1b4cd/examples/wide_ep/ep_load_balancer），DeepSeek-R1：EP32 不做 EPLB 时，最忙 rank / 均值 = 1 + 1.564 = 2.56（各层各迭代平均）；EP36 + 32 个冗余专家的离线 EPLB 后为 1.115。代入 SGLang 行：用 1.115 时，模型给出的默认分布 / 完美 EPLB prefill 减速是 1.115 ×，实测 59,337 / 50,302 = 1.18 ×，方向一致，量级接近；用 2.56 时 prefill 预测只剩实测的 0.41，说明 SGLang 的「默认」已经过 EPLB。**更正 §25.7 第 2 条**：EP72 decode 高估 1.27 ×，用不均衡解释不了——在模型里 skew 1.115 只把它变成 1.25（decode 受 MAC 限制）。

**(d) 每层固定开销。** 新增 `Scenario.layer_overhead_us`（每层每步的固定时间，单位 µs，所有重叠模式下都暴露；默认 0 = 关，与默认指纹逐位相同；取值依赖 NPU / 运行时「假设」，不拟合进默认值）。留出检验：只在 H100 的 128-token TTFT 上选值（kernel 模式约 100 µs / 层最好），拿去测别的组：

| group (speed ratio pred/meas) | n | catalog_kernel | catalog_kernel+L100 | catalog_serial | catalog_serial+L100 | catalog_tbo | catalog_tbo+L100 |
| v1 TRT 0.8 static | 36 | 1.16/27% | 1.01/20% | 0.98/19% | 0.87/21% | 1.19/29% | 1.03/22% |
| v1 TTFT | 20 | 1.59/65% | 1.11/20% | 1.19/37% | 0.91/21% | 1.75/82% | 1.19/29% |
| v1 0.21 max-load (CB) | 44 | 1.04/20% | 0.89/22% | 0.81/26% | 0.72/40% | 1.08/24% | 0.91/23% |
| MoE large-EP | 6 | 0.59/69% | 0.59/70% | 0.55/83% | 0.54/84% | 0.99/12% | 0.97/10% |
| Mixtral TP8 (CB) | 27 | 0.90/29% | 0.81/36% | 0.70/43% | 0.65/54% | 1.11/30% | 0.98/27% |
| MI300X latency | 16 | 3.27/227% | 1.03/23% | 2.89/189% | 0.98/21% | 3.61/261% | 1.07/27% |
| MI300X max-load (CB) | 4 | 2.15/115% | 1.96/96% | 1.87/87% | 1.72/72% | 3.11/211% | 2.74/174% |
| B200 NVFP4 (CB) | 11 | 1.16/22% | 0.90/14% | 0.85/18% | 0.71/41% | 1.16/22% | 0.90/14% |

在 H100 短 prompt 上选出的 100 µs / 层，换到非 H100 的短 prompt TTFT 误差从 130 % 降到 22 %，换到完全不同来源 / 硬件 / 指标的 MI300X 静态批延迟（全部 16 条）从 227 % 降到 23 %。所以 GPU 运行时确实有一项结构性的每层固定开销（约 0.1 ms / 层），模型现在可以显式表达；NPU 设计的默认值仍然是 0。它对大批量吞吐（MI300X 满载仍是 2 ×，Mixtral 稍差）帮助不大。


### 25.9 第四轮（V6.4）

**(a) PD 分离实测。** 找到的公开 PD 结果大多只有图，没有可用的数值表：TensorRT-LLM blog05（GB200 DS-R1，Pareto 曲线与相对倍数）、NVIDIA Dynamo、vLLM disagg、Mooncake、DistServe / Splitwise（图与相对 goodput）。DeepSeek 第 6 天公布的是生产均值（长度分布、缓存命中都不定），不算完整配置。所以这一轮 PD 端到端（含 KV 跨网传输）没有新的实测可对照；能用的只有 SGLang PD 的 prefill / decode 单节点数（§25.7 已用）。新增一条：DeepSeek profile decode（16 × 8 H800，EP128，每卡 128 个请求，KV 4096，18,598 tok/s / 节点）。MTP 按投机解码处理（spec_k = 1，接受率取 DeepSeek-V3 报告 §5.4.3 给的 0.85 ~ 0.90 的中点 0.875，不拟合）。

**(b) EP72 decode 高估：逐算子对照。** SGLang 只公布了 prefill 的 kernel 分解图；decode 用 DeepSeek 公开的 trace（profile-data/decode.json，一次前向，双微批，MTP 校验 q = 2），按 kernel 名归类后和模型（tbo，同一配置，spec_k = 1）逐项比，单位 ms / 步：

| 项 | DeepSeek trace | 模型 0.69 | 说明 |
|---|---|---|---|
| MLA 注意力核 | 31.4 | 88.2（阵列） | 模型把 GPU 当成一整块宽阵列：N = 512 的 PV 只用上 6 % 的列，128 个请求的 GEMM 串行执行，利用率 12 % |
| 非注意力 GEMM（投影 + 专家 + 共享专家） | 33.3 | 阵列 18.6 / 权重 DRAM 21 | 实测里每个微批各读一遍权重 |
| 逐元素 / norm / cast / rope / gate（1734 个小 kernel） | 7.6 | 1.4 | ≈ 每层 125 µs，和 §25.8 独立选出的 100 µs / 层吻合 |
| dispatch / combine kernel | 17.1 | link 48（全部藏起） | 低延迟 a2a 的 kernel 本身占着流 |
| 合计（跨度） | 96 | 111 | |

修正（有原则，只影响非默认模式）：**`exec_overlap="tbo"` 的每个微批各读一遍权重**（t_dram 加一份 weights / BW）。效果：EP72 decode 1.27 → 1.19，加 100 µs / 层后 1.10；DeepSeek profile decode 0.93（加 100 µs 后 0.88）；prefill 行不变。这两条 decode 行把 1 夹在中间，剩下的差异更像是两套软件之间的差别，找不到单独一项能解释。

试过更贴近 GPU 的映射：新增 `Chip.split_instances`（批量 GEMM 实例分到各 SM 上，每个 SM 是一个 R × C 阵列；默认关，目录芯片和参考硬件都不开。0.71 起由 `Chip.instance_sched` 取代：目录 NPU 默认 auto，参考硬件 wide，§26；`split_instances` 仅保留兼容）。打开后注意力项从 88 降到 19 ms（实测 31），但总体变成 1.76 ~ 1.88 ×高估：原来注意力项偏慢，正好抵掉了 GEMM / 小 kernel / 通信暴露这几项偏快的部分。真实 GPU kernel 达不到 roofline，要补就得拟合 kernel 效率，我们不做，所以参考硬件保持 0.69 的宽阵列抽象，并在这里写明它是靠误差互相抵消才准的。

**(c) 重算的总表**（tbo 已含每个微批各读一遍权重）：
| group | n | catalog_kernel+L100 | catalog_tbo | catalog_tbo+L100 | catalog_tbo+S+L100 | catalog_serial+L100 |
| v1 static | 36 | 1.01/20% | 1.01/29% | 0.89/28% | 0.91/27% | 0.87/21% |
| v1 TTFT | 20 | 1.11/20% | 1.31/42% | 1.01/31% | 1.02/29% | 0.91/21% |
| v1 max-load | 44 | 0.89/22% | 0.92/24% | 0.81/33% | 0.81/33% | 0.72/40% |
| MoE large-EP | 7 | 0.59/69% | 0.97/10% | 0.95/10% | 1.13/26% | 0.54/84% |
| Mixtral TP8 | 27 | 0.81/36% | 0.91/32% | 0.83/34% | 0.83/33% | 0.65/54% |
| MI300X lat | 16 | 1.03/23% | 2.12/112% | 0.89/35% | 0.89/35% | 0.98/21% |
| MI300X max-load | 4 | 1.96/96% | 3.11/211% | 2.74/174% | 2.74/174% | 1.72/72% |
| B200 NVFP4 | 11 | 0.90/14% | 0.94/14% | 0.77/30% | 0.77/30% | 0.71/41% |

大规模 EP（tbo）：
```
sgl-pf-4096-default [1.06, 1.06, 1.06]
sgl-pf-4096-perfect [0.9, 0.9, 0.9]
sgl-pf-1024-default [0.93, 0.92, 0.92]
sgl-pf-2048-default [0.98, 0.98, 0.98]
sgl-dec-2000-default [1.19, 1.1, 1.76]
dsp-pf-4096 [0.85, 0.85, 0.85]
dsp-dec-4096-mtp [0.93, 0.88, 1.88]
```
尚未解决、有系统性的：MI300X 满载吞吐仍高估约 2 ×（软件栈；小批延迟加 100 µs / 层后已对上）；kernel 级的注意力效率（见上，不拟合）；PD 端到端没有数值实测。

## 26. 独立 GEMM 实例在多核上的调度（0.71，`Chip.instance_sched`）

**问题。** ≤ 0.70 把 E 个 engine（核）并成一块 R × (C·E) 的宽阵列，一个 op 的 `count` 个独立实例（每请求 / 每 KV 头的注意力 QK·PV、MLA 吸收后的每头乘积、每专家 GEMM）在这块宽阵列上一个接一个跑。N 小（head_dim 128、latent 512）时，绝大多数列空着：1P（128² × 32）上，prefill 注意力的阵列利用率只有 6 ~ 7 %。

**设计变量**（芯片字段，默认 `"wide"` = 0.70 的行为）：
- `"wide"`：同上。
- `"split"`：每个 engine 跑一个实例，共 ⌈count / E⌉ 波（= 0.70 的 `split_instances=True`）。
- `"auto"`：逐 op 取最便宜的方案，候选是 wide，以及每种整除 E 的核组大小 g（共 E/g 组，每组是一块 R × (C·g) 阵列）；实例数少于组数时，再把每个实例的 M 行切到 s 个组上（行并行）。

约束：
- SRAM 端口是全片预算（片上 NoC / 端口带宽共享），按正在工作的组均分：auto 也试只用 par、par/2、… 个组，空闲组让出份额（0.71 审计）；行切分时每组都要把完整的 K × N 操作数流过自己那份端口。
- 每组的实例背靠背流过，流水线填充每组每 op 一次（与 wide 相同）。
- 每组分到 g/E 的累加器。
- SRAM / DRAM 总字节数不变（操作数都是流式的）。
- 每组需要自己的指令流 / sequencer「假设」。
- 每波的调度开销取 0「假设」。

**注意力 / 专家利用率**（固定总 MAC 数 = 1P 的 524,288，1 GHz，单卡，`scripts/instance_sched_study.py`）：
| case | R×C×E | attn util wide / split / auto | attn array ms wide → auto | expert util wide → auto |
|---|---|---|---|---|
| DS-V3 MLA decode b64 ctx4k | 512x512x2 | 16% / 23% / 23% | 27.2 → 18.1 | 1% → 1% |
| DS-V3 MLA decode b64 ctx4k | 256x256x8 | 19% / 48% / 48% | 22.7 → 8.8 | 1% → 1% |
| DS-V3 MLA decode b64 ctx4k | 128x128x32 | 21% / 98% / 98% | 20.5 → 4.3 | 2% → 2% |
| DS-V3 MLA decode b64 ctx4k | 64x64x128 | 12% / 50% / 99% | 36.5 → 4.3 | 3% → 5% |
| DS-V3 MLA decode b64 ctx4k | 32x32x512 | 6% / 17% / 99% | 73.0 → 4.3 | 3% → 9% |
| DS-V3 MLA decode b8 ctx32k | 512x512x2 | 17% / 25% / 25% | 25.3 → 17.1 | 0% → 0% |
| DS-V3 MLA decode b8 ctx32k | 256x256x8 | 20% / 50% / 50% | 20.8 → 8.5 | 1% → 1% |
| DS-V3 MLA decode b8 ctx32k | 128x128x32 | 23% / 40% / 99% | 18.5 → 4.3 | 1% → 2% |
| DS-V3 MLA decode b8 ctx32k | 64x64x128 | 12% / 12% / 99% | 34.8 → 4.3 | 2% → 3% |
| DS-V3 MLA decode b8 ctx32k | 32x32x512 | 6% / 6% / 99% | 67.3 → 4.3 | 2% → 6% |
| Qwen3-32B GQA decode b64 ctx4k | 512x512x2 | 0% / 1% / 1% | 155.2 → 86.0 | — → — |
| Qwen3-32B GQA decode b64 ctx4k | 256x256x8 | 0% / 2% / 2% | 146.8 → 25.7 | — → — |
| Qwen3-32B GQA decode b64 ctx4k | 128x128x32 | 0% / 6% / 6% | 142.6 → 8.5 | — → — |
| Qwen3-32B GQA decode b64 ctx4k | 64x64x128 | 0% / 12% / 12% | 138.4 → 4.2 | — → — |
| Qwen3-32B GQA decode b64 ctx4k | 32x32x512 | 0% / 25% / 25% | 138.4 → 2.1 | — → — |
| Qwen3-32B GQA decode b8 ctx128k | 512x512x2 | 0% / 1% / 1% | 604.5 → 335.8 | — → — |
| Qwen3-32B GQA decode b8 ctx128k | 256x256x8 | 0% / 2% / 2% | 571.0 → 100.7 | — → — |
| Qwen3-32B GQA decode b8 ctx128k | 128x128x32 | 0% / 6% / 6% | 554.2 → 33.6 | — → — |
| Qwen3-32B GQA decode b8 ctx128k | 64x64x128 | 0% / 8% / 12% | 545.8 → 16.8 | — → — |
| Qwen3-32B GQA decode b8 ctx128k | 32x32x512 | 0% / 5% / 17% | 541.6 → 12.6 | — → — |
| Qwen3-32B prefill b1 4k | 512x512x2 | 22% / 40% / 40% | 75.5 → 41.9 | — → — |
| Qwen3-32B prefill b1 4k | 256x256x8 | 12% / 67% / 67% | 142.6 → 25.2 | — → — |
| Qwen3-32B prefill b1 4k | 128x128x32 | 6% / 40% / 100% | 276.8 → 16.8 | — → — |
| Qwen3-32B prefill b1 4k | 64x64x128 | 3% / 11% / 100% | 553.6 → 16.8 | — → — |
| Qwen3-32B prefill b1 4k | 32x32x512 | 2% / 3% / 100% | 1107.3 → 16.8 | — → — |
| DS-V3 prefill b1 4k | 512x512x2 | 26% / 45% / 45% | 151.9 → 87.9 | 25% → 25% |
| DS-V3 prefill b1 4k | 256x256x8 | 14% / 71% / 71% | 279.8 → 56.0 | 48% → 50% |
| DS-V3 prefill b1 4k | 128x128x32 | 7% / 100% / 100% | 535.7 → 40.0 | 95% → 100% |
| DS-V3 prefill b1 4k | 64x64x128 | 4% / 100% / 100% | 1071.4 → 40.0 | 58% → 100% |
| DS-V3 prefill b1 4k | 32x32x512 | 2% / 25% / 100% | 2142.8 → 40.0 | 29% → 100% |
| Qwen3-30B-A3B MoE decode b64 ctx4k | 512x512x2 | 0% / 1% / 1% | 58.2 → 32.2 | 1% → 1% |
| Qwen3-30B-A3B MoE decode b64 ctx4k | 256x256x8 | 0% / 2% / 2% | 55.1 → 9.6 | 2% → 2% |
| Qwen3-30B-A3B MoE decode b64 ctx4k | 128x128x32 | 0% / 6% / 6% | 53.5 → 3.2 | 2% → 4% |
| Qwen3-30B-A3B MoE decode b64 ctx4k | 64x64x128 | 0% / 12% / 12% | 51.9 → 1.6 | 2% → 8% |
| Qwen3-30B-A3B MoE decode b64 ctx4k | 32x32x512 | 0% / 12% / 25% | 51.9 → 0.8 | 2% → 15% |


**审计（0.71，`tests/test_core_071_audit.py`）。** 600 组随机几何 / 形状 / 端口 / 累加器上检查：auto ≤ min(wide, split)；auto ≥ 理想 MAC 下界 cnt·m·k·n / (MAC 数 × 速率)（功守恒）；auto ≥ 全片端口下界 cnt·(K·N + M·K + M·N) 字节 / 端口；每核阵列与每核累加器不变、核数翻倍时不变慢（端口按几何推导或总量固定）。审计前的版本（每组固定 g/E 端口、每实例各付一次填充）在核数翻倍检查上有 44 / 4000 个反例，修正后为 0；上表与目录头条数值逐项不变。count = 1 仍只走 wide；wide 本身在核数翻倍时因填充 R + C·E 变长可略慢，属于 wide 阵列的性质。（0.72 起 count = 1 也走 auto，见 §26.1。）

### 26.1 单个 GEMM、NoC 与波次屏障（0.72）

- **单个 GEMM。** auto 对 count = 1 也比较 wide 与核组方案：g | E 的核组、活跃组数 par, par/2, …，以及把 M 行（每组流过完整 K × N）或 N 列（每组流过完整 M × K）切到 s 组。wide 的流水线填充是 R + C·E（一条跨所有核的 systolic 链「假设」），N 远小于 C·E 时它既浪费列又拉长填充；N 切分把填充降到 R + C·g。`"split"` 只处理 count > 1（0.70 语义）。
- **NoC（可选，默认关）。** `noc_topology` ∈ {crossbar, ring, mesh}、`noc_hop_cycles`：跨 s 个核的调度每个 op 每个 sequencer 付 hop × 直径(s)（1、⌊s/2⌋、2·(⌈√s⌉ − 1)）。wide 跨 E 个核，核组跨 g 个核。只计延迟；带宽仍由共享 SRAM 端口约束。
- **波次屏障（可选，默认 0）。** `wave_overhead_cycles`：每个 op 一次启动；≥ 2 个活跃核组按锁步波次跑实例时每多一波一次屏障。单一 sequencer（wide 或单个活跃组）背靠背流过实例，不付屏障。
- **不变量**（扩展后的审计测试，含 count = 1 与随机 NoC / 屏障）：auto ≤ min(wide, split)；≥ 理想 MAC 下界；≥ 全片端口下界；每核资源不变、核数翻倍不变慢。
- **默认数值变化**（0.71 → 0.72）：

逐数变化（`/workspace/val5/headline.py`，0.71.0 → 0.72.0，3 芯片 × 目录模型 × 3 工作点 × 3 布局，1305 个场景）：451 个数值改变，瓶颈翻转 MAC → DRAM 21，MAC → LINK 9。

| chip | point | metric | n | min | median | max |
|---|---|---|---|---|---|---|
| 100T | decode b1 | throughput | 22 | +0.1% | +16.7% | +82.3% |
| 100T | decode b16 ctx32k | throughput | 25 | +0.1% | +16.7% | +114.3% |
| 100T | prefill b2 8k | throughput | 141 | +0.1% | +5.6% | +86.2% |
| 100T | prefill b2 8k | ttft | 116 | -36.3% | -4.8% | -1.0% |
| 1P | decode b1 | throughput | 21 | +0.0% | +16.5% | +154.3% |
| 1P | decode b16 ctx32k | throughput | 23 | +0.0% | +17.7% | +115.6% |
| 1P | prefill b2 8k | throughput | 32 | +0.0% | +22.2% | +131.7% |
| 1P | prefill b2 8k | ttft | 11 | -56.8% | -37.4% | -9.9% |
| H100-like | decode b1 | throughput | 20 | +0.0% | +3.9% | +40.0% |
| H100-like | decode b16 ctx32k | throughput | 20 | +0.0% | +3.9% | +40.2% |
| H100-like | prefill b2 8k | throughput | 20 | +0.0% | +3.9% | +40.1% |

  小模型 / 窄 N 的 GEMM 在大芯片（1P、TP8）上变化最大，部分场景瓶颈由 MAC 变为 DRAM / LINK；参考硬件 GPU 仍为 wide，外部校核表不变。


**设计结论。**
1. 只能 wide 时，核越多，注意力越差（N = head_dim 只占宽阵列的一小段）：32² × 512 上 prefill 注意力仅 2 %。
2. 核组能各自跑实例（auto）时，MLA（M = 128 头）和 prefill 注意力在 R ≤ 128 的所有几何上都接近 100 %。大阵列（512²、256²）因为 N = head_dim / latent 小于 C，上限只有 23 ~ 71 %。在固定 MAC 下，从注意力角度看，128² × 32 或 64² × 128 加独立调度最好；512² × 2 在注意力算力上至少损失 55 %。
3. GQA decode：上限是 q·g / R，和调度无关（Qwen3-32B：g = 8，R = 32 时 25 %，R = 128 时 6 %）。要更高，得用更小的 R、GEMV 单元或投机解码（q > 1）。MoE decode 的专家 GEMM 每专家 token 极少，也是同样的 M 上限。
4. decode 大多受 DRAM 限制，所以默认 stage 重叠下 TPOT 基本不变。prefill、kernel / serial 重叠模式、以及算力相对带宽紧张的设计，受影响最大。

**默认值（0.71 起）。** 目录里的 NPU 芯片（以及 `Chip()` 的默认值）改为 `"auto"`：各核有独立 sequencer，「所有核锁步跑同一个实例」是 ≤ 0.70 的建模错误。`"wide"` 仍可选（单 sequencer 锁步阵列用它）。参考硬件 GPU 显式设为 `"wide"`：§25.9 的校核表就是在 wide 下得到的，GPU 上更忠实的映射会让总体高估加大，所以外部校核表逐行不变。`instance_sched` 是 omit_default 字段：默认场景的 `hash()` 不变，但含义从 wide 变成了 auto；只有显式写 `"wide"` 的场景才会序列化这个字段。

**数值变化**（0.70 wide → 0.71 auto，3 种目录芯片 × 75 个模型 × 3 个服务点 × 1 卡 / TP8 / DP8·EP8；逐条见 `scripts/instance_sched_diff.py` 的输出 diff_all.csv）：
changed numbers: 555 (models 74); bottleneck flips: 88 — MAC→DRAM 88

| chip | point | metric | n | min | median | max |
|---|---|---|---|---|---|---|
| 100T | decode b1 | throughput | 21 | +0.0% | +0.5% | +6.8% |
| 100T | decode b16 ctx32k | throughput | 27 | +0.0% | +0.6% | +6.8% |
| 100T | prefill b2 8k | throughput | 143 | +0.0% | +30.3% | +169.3% |
| 100T | prefill b2 8k | ttft | 116 | -62.9% | -25.7% | -7.1% |
| 1P | decode b1 | throughput | 11 | +0.0% | +1.7% | +28.6% |
| 1P | decode b16 ctx32k | throughput | 15 | +0.0% | +0.9% | +25.6% |
| 1P | prefill b2 8k | throughput | 82 | +0.0% | +54.2% | +773.5% |
| 1P | prefill b2 8k | ttft | 69 | -88.6% | -40.0% | -0.5% |
| H100-like | decode b1 | throughput | 9 | +0.0% | +0.0% | +1.0% |
| H100-like | decode b16 ctx32k | throughput | 10 | +0.0% | +0.0% | +1.1% |
| H100-like | prefill b2 8k | throughput | 31 | +0.0% | +6.5% | +209.8% |
| H100-like | prefill b2 8k | ttft | 21 | -67.7% | -8.3% | -2.8% |

decode 受 DRAM 限制，几乎不变；prefill TTFT 中位数：100T −25.7 %、1P −40.0 %、H100-like −8.3 %。瓶颈变化全部是 MAC → DRAM（167 处）。

## 27. 第 15 轮对抗性审计（0.81）：前缀缓存、KV 策略、分块、投机解码、异构 prefill、PD 能耗

**前缀缓存 vs DES 缓存（随机工作负载）**
- 整前缀 LRU：Che 近似对 `pdsim.LRUCache` 60 个随机点（N 10–20000，α 0–1.3，任意 K）全部在 0.02 + 3 % 内。
- radix：`tree_resident` 对 `pdsim.RadixLRU`。发现**节点粒度**问题：LRU 只存整节点（淘汰到 used ≤ C），Che 按连续容量 C 求 T，缓存只能放下少数节点时高估命中（10 个 128-token 节点、C = 308：闭式 0.24，LRU 0.20）。修正：所有可达层节点长度相同时 C_eff = ⌊C/L⌋·L（精确）；长度不同时 C_eff = C − ½·被淘汰节点的平均长度（淘汰按节点缺失率 p·(1 − h) 加权，先用 C 做一次 Che）。286 个随机树：平均 |误差| 0.029 → 0.013，p90 0.088 → 0.030；最大 ≈ 0.24 剩在「根节点很长、子树只剩几个节点的空间」这类装箱情形。容量放得下成千上万个节点时修正可忽略；默认场景不用前缀树，指纹不变。
- 节点数很大（4 万–200 万）而请求只有 6 万时 DES 偏低，是冷启动缺失（闭式按稳态）：60 万请求时 0.9930 对 0.9931。

**KV 策略 / 分块 / 投机解码 / 异构 prefill vs DES**（28 个随机点：dense8b / moe30b，负载 0.3 / 0.6 / 0.8，CV 0 / 1；wait / recompute / swap × 2–8 GB × 两种准入；spec_k 1–4 × 接受率 0.5–0.9；prefill 芯片 100T / H100-like × LPDDR5X；chunk 128–2048；6000 请求 × 2 seed）
- 分块、投机解码、异构 prefill：超出 V4 误差带的只有 seed 噪声 ≥ 0.37 的 TTFT p99。
- **DES 修正**：PD before_prefill 且开 KV 策略时，KV 到达的请求在**迭代中途**就记为开始 decode（decode_start / last_tok 提前），而它要等下一次迭代才真正参与；首个间隔多算了在途迭代的剩余时间。同一调度（迭代数、batch 分布逐一相同）开 / 关一个从不绑定的 KV 上限，最长间隔 p99 14.3 对 10.0 ms。现在这些请求在下一个迭代边界开始（仍在 `pending` 中占位），与关闭上限时逐位一致。V4 数据不开 KV 策略，不受影响。
- 两个「闭式稳定、DES 不稳定」点（wait，负载 0.6 / 0.8）在 4 倍长度下 DES 也判稳定（drift_ttft 是短运行伪影），TTFT 尾部一致（p99 7.45 对 6.19 s、14.1 对 15.5 s）。
- **已知缺口（未改）**：合并模式 recompute、容量很紧（2 GB，每副本约 3 个在跑）时，闭式把被抢占者的等待取为 Exp(1/λ_r)（等一个离开，均值 740 ms），DES 测得抢占 → 恢复平均 40 ms（p99 0.24 s）：被抢占者腾出的正是自己的足迹，下一个边界通常就能装回（vLLM 式，31 % 的停顿会连续被抢两次）。闭式最长间隔因此偏保守 5–8 倍（2.06 s 对 DES p99 0.25–0.37 s）。没有找到有依据的替代式（需要「上穿容量时刻到下次可恢复」的剩余时间分布），不拟合常数。

**投机解码能耗**：单位 = 接受的 token（E = Σ aⁱ）；接受率 0 时 k = 2 每 token MAC ≈ 3.1 倍（3 个验证位置 + 草稿），每 token 权重字节按 E 摊薄；无 MTP 的模型给 spec_k 时警告并忽略。

**PD 能耗恒等式**：静态 = W × 卡数 / (λ·out)（PD 按池拆：idle_W_prefill × n_p、idle_W × n_d）；PD 比合并多出的 link 字节 = 每请求 KV 交接字节 / out，只计一次；对能耗表线性。均成立。

**构造函数接受芯片预设名**：`Scenario(chip="1P")` 原来被接受、在 evaluate 中报 AttributeError；`PDConfig(prefill_chip="100T")` 原来被拒绝（报错信息却说可以写预设名）。现在两处都解析为预设（哈希与 Chip 对象相同），未知名称 / 非芯片报 ValueError。

**MODEL.md 过时表述**：引用检查（路径、`pd.` / `serving.` / `workload.` / `energy.` / `chip.` 字段、`模块.属性`）只发现 `workload.tmpl_msa`（是发布默认项，不是 workload 字段）；另外标注 / 改正了后续版本已补上的「不做」：网络拓扑 / 超额订阅 / 网内归约（§15 → §19）、异构池（§18）、按池闲置功率（§18.3）、radix 与 KV 容量排队（§18.1–18.3）、`split_instances` → `instance_sched`（§25）、decode batch 搜索上限（max(4096, 64 × 每副本卡数)）、池布局搜索超过 64 卡的副本规模。

常驻测试 `tests/test_core_081.py`。

## 28. 小 radix 树的精确驻留；被抢占者恢复等待（0.82）

**radix 前缀树：小树精确解（采用）。** Che 是大群体近似，节点很少时偏差大（0.81 剩余最坏情形：根很长、子节点只剩一两个位置）。IRM 下 LRU 的稳态访问次序就是按概率不放回抽取路径（move-to-front）；访问一条路径时从叶到根刷新，所以按节点新近度排序时，每条路径先列根、再列尚未出现的子孙。按 token 计容量的 LRU 保留「能放下的最长新近前缀」（从旧端淘汰到 used ≤ C）。枚举这些次序直到容量放满，就得到每个节点的精确驻留概率；树性质成立，所以 H_k = Σ p_node·P(驻留) 是精确值。只在存储路径数 ≤ 4096 且枚举步数 ≤ 50 000 时启用（按 (levels, C) 记忆），否则仍用 Che + 0.81 的节点粒度修正。例：64-token 根 + 两个 4096-token 子节点、只放得下一个：H₂ = Σp² = 0.536（Che 偏 0.42）。562 棵随机树对 RadixLRU：平均 |误差| 0.030–0.042 → 0.008–0.015，没有一棵变差（>0.005）；剩余最坏（≈ 0.44）是超出枚举预算的大树（中层很长、底层很多小叶）。
- 试过的另一修正（未采用）：把边界节点按长度加权（检查悖论，½·Σ w L²/Σ w L）：比 0.81 的 ½·平均被淘汰长度更差（平均 0.032 → 0.035，最大 0.29 → 0.33）。

**合并 / PD 的 recompute / swap 被抢占者恢复等待（研究，未采用；默认不变）。** DES 里被抢占者（最年轻）在其余某个请求离开后就装回（腾出的正是自己的足迹，vLLM 式），而抢占发生在足迹达到 K 时，即在跑的请求都比较「老」。
- 固定长度：给定 Σ 足迹 = K，k 个请求的剩余 token 之和 Ψ = k(S + o) − K，其余 k − 1 个中的最小剩余寿命均值 Ψ/k²，比 Exp(1/λ_r) 短得多（dense8b CV 0：DES 40 ms，旧闭式 740 ms）。
- 输出长度可变（CV 1）时剩余寿命几乎无记忆，等待接近一个离开间隔（MoE CV 1：DES 0.47 s，1/λ_r = 0.78 s），只用固定长度式会低估约 10 倍。
- 推广式：在跑请求按长度加权、年龄 U·o；剩余 R 取长度分布的平衡剩余律，用 Σ 足迹 = K 的高斯条件期望缩放其均值（固定长度 → Ψ/k，指数输出 → 不变），等待 = k − 1 个 R 的最小值 × TPOT，按每个 k 的 Rice 抢占率混合。
- 42 个随机紧 KV 点（recompute / swap，1.5–6 GB，三种模式，6000 请求 × 2 seed）：最长间隔 |误差| 中位 0.50 → 0.28，但有 9 个模式点变差（MoE CV 1 的 4 个点；另有 2 个固定长度点两种律都偏大 65–140 倍，原因是饱和抢占次数高估，而不是等待律）。不满足「处处更好」，未采用：`pdqueue.VICTIM_WAIT = "exp"`（默认，与 0.81 逐位一致），`"resid"` 留作内部开关。剩余缺口：饱和抢占次数（`preempt_sat`）在固定长度下高估；混合长度下条件剩余寿命的形状。

## 29. 第 17 轮（0.83.0）：饱和抢占增长律、恢复等待混合律、radix 子集 DP

**饱和抢占。** 运行中 footprint 在容量 K 下方随解码增长、随离开下降。≤ 0.82 把两次离开之间的增长量当指数分布，即无记忆寿命；固定输出长度时离开近乎周期，越线概率被高估几个数量级（点 17/18 最坏间隔高 65–140×，DES 实测 0 次抢占）。0.83 用 Gamma 分布（均值 E[o]、形状 1/c²），c² 为寿命 CV²：输出长度 CV² 与逐请求 TPOT 散布合并，封顶 1。越线概率 = 对均匀余量 h 取平均的 Q(1/c², h/(c²·E[o]))。c² = 1 时就是旧的指数式，c² → 0 时为 0。

**恢复等待。** 被抢占请求的等待：c² = 1（无记忆）时 Exp(λ_r) 精确，固定长度时 §28 的条件剩余寿命律更准。取生存函数的线性混合 (1 − c²)·resid + c²·exp。

**验证与结论。** 42 点紧 KV 网格上没有一个点变差，但 test_core_056 的高负载固定长度点上抢占被低估 4–9×：到达噪声也会造成越线，模型只计入了寿命 CV²。因此**未采用**：SAT_MODEL / VICTIM_WAIT 默认仍为 "exp"，gamma / blend 只作内部开关。下一步要在增长律里加入到达方差项（例如按负载的 c²_eff），并把高负载固定长度点加入网格。

**radix 精确解。** 改为对「能放下的路径子集」做 DP，f(A) = P(A 恰为最近 |A| 条路径)，状态数代替排列数，预算 EXACT_BUDGET = 100000。

## 30. 第 18 轮（0.84.0）：饱和增长律的到达方差项（已采用）

§29 的 Gamma 增长律只计入寿命方差。但运行集合里请求的启动时刻是 Poisson 的，离开过程会趋向 Poisson（Burke / M/G/∞）；只有在 KV 较空、门控准入时，集合才保持固定节拍。因此取 c²_eff = c² + (1 − c²)·ρ_K^m，ρ_K = E[运行数]·E[footprint]/K：ρ_K → 1 时还原为指数律，ρ_K → 0 时只剩寿命 CV²。m 是唯一的形状参数，只在偶数网格点上选定（m = 1），奇数点留出检验。

验证（val12，95 个 DES 稳定的点-模式对，含负载 0.85/0.9 的固定长度点）：相对 0.82 / 0.83 默认没有一个点变差，均值改善（见 CHANGELOG 0.84.0）。SAT_MODEL / VICTIM_WAIT / SAT_ARR_M 设为 "exp" / "exp" / None 即还原 ≤ 0.83。仍未解决：低负载 PD 的点 18（最坏间隔 243 ms，DES 3.8 ms），以及 DES 本身不稳定的高负载小容量点（不可评分）。

## 31. 第 19 轮（0.85.0）：抢占的整数装箱界；UI / API / 目录审计

**装箱界（精确）。** recompute / swap 下，新请求的准入条件是已用 KV + 其 prompt ≤ K。因此同时运行的序列数不超过 n = min(B, ⌊K / S_min⌋)。若 n 条序列都长到最大 footprint（再加一步增长 e）仍不超过 K，即 n·(max(S + o) + e) ≤ K，就不会发生任何抢占，与到达过程无关。固定长度、prompt 较长的紧 KV 点（例如 K = 15258、S = 4096、o = 512 → 3·4609 ≤ K）正属于这种情况：连续 footprint 模型（§29–30）会给出很小但非零的越线率，而 DES 中为 0。条件成立时，闭式把 Rice 与饱和两项都置 0。这是充分条件，不改变其他情况。

**审计。** API 现在拒绝未知的 body 顶层键；UI 的「放不下」结论与错误信息也会被标为过期；目录覆盖标签与实现一致（见 CHANGELOG 0.85.0）。

## 32. 第 20 轮（0.86.0）：特性组合与数值极端

**组合。** 随机组合 PD、D2D 小芯片、SLC（pin / lru）、MoE 倾斜、投机解码、前缀缓存（单前缀 / radix 树）、KV 策略和异构 prefill 芯片（960 例）。检查四条验收准则以及内部一致性：
- 容量：fits ⇔ 各段 DRAM 需求 ≤ 容量；
- 部分之和 = 总量：能量按动作求和、PD 流体容量（req_s = min 容量，decode 容量 = 副本 × tok/s / 输出长度）；
- 能量对每动作 pJ 线性，关闭 D2D / SLC 时对应能量为 0；
- 工作量在 TP 上守恒；
- 单调性：倾斜、接受率、已缓存前缀、pin SLC 容量；
- 搜索：best_split 不劣于给定划分。

结果：0 违例。

**共置与 PD 的 1:1 极限。** 两边用相同的 prefill / decode 速率时，共置分时是流体最优解：每卡 goodput = 1 / (1/r_d + (S/o)/r_p)，而 PD 的整数划分只能达到或低于它。因此在 prefill 池与 decode 同布局时，PD ≤ 共置成立（常驻测试 `test_core_086`）。PD prefill 池用更便宜的布局（例如 TP1，没有集合通信）时，可以合理地超过共置，在随机审计里最多高 45%。

**数值极端。** 上下文到 1M、1 → 1024 卡、batch 1 → 65536、主频 / SRAM 极端值：结果全部有限，单调性成立，超过 max_position_embeddings 时有警告。deepseek-v3.2 的 1M prefill 主要花在 indexer 上（O(L²)，按参考实现在每个 TP rank 上重复计算）；这是设计上值得注意的点（长上下文 prefill 应该用 DP 注意力或把 indexer 按 TP 切分），不是模型错误。

## 33. 第 21 轮（0.87.0）：稀疏注意力 indexer 的部署选项；CLI 审计

**选项（`Scenario.indexer`，「假设」）。** DSA 的 indexer 分数是 s(q, k) = Σ_h w_h·ReLU(q_h·k)：只有一个共享的键头，查询有 H_I 个头。在 TP 上有三种精确的部署：

| 选项 | 每 rank 打分 | 通信 | 每 rank indexer 缓存 |
|---|---|---|---|
| replicated（V3.2 / GLM-5 发布） | 全部 H_I 头 × 全部键 | 无 | 全部键 |
| tp_split（V4 发布） | ⌈H_I/TP⌉ 头 × 全部键 | all-reduce fp32 [token × 键] | 全部键 |
| key_split | 全部 H_I 头 × ⌈键/TP⌉（轮转分片，因果均值不变） | all-gather 每 token k_loc ×（fp32 + int32） | ⌈键/TP⌉ |

key_split 每个 rank 取局部 top-k；全局 top-k 一定包含在各 rank 局部 top-k 的并集里，因此合并（TP·k_loc 次比较）是精确的。查询侧的准备（LayerNorm / RoPE / Hadamard / fp8 量化）在每个 rank 上重复。「按头切分后只汇集 top-k」不精确，所以不提供。默认 reference 表示保持发布时的方式，指纹不变。

**设计含义。** 在 1M prefill、TP8 下：replicated 由 O(L²) 的 indexer 主导（2820 s）；tp_split 把计算除以 TP，但打分矩阵的 all-reduce（每层 2.2 TB）占主导（598 s）；key_split 计算同样除以 TP，通信只剩 top-k 候选（425 s），还省下 7/8 的 indexer 缓存。decode（128K 上下文，batch 32）：77 → 59 → 46 ms。在长上下文的 NPU 部署里，indexer 的分片方式是一阶设计变量。DP 注意力只有在 batch ≥ DP 时有用：单请求时 7/8 的副本空闲（新增警告）；batch 8 时 DP8 最好，而且是唯一放得下的布局（TP 下 MLA latent 缓存会按 TP 重复）。

**CLI 审计。** 此前有 11 个场景字段没有命令行参数，现已补齐；常驻测试检查每个字段都能从命令行设置。命令行与 API 结果一致；错误输入都干净退出、没有 traceback。

## 34. 第 22 轮（0.88.0）：上下文并行（环形注意力）；hash 兼容、服务并发、文档一致性

**模型（`layout.cp`，LLM prefill，「假设」：zigzag 环形注意力）。** 把序列切成 2·CP 块，卡 i 持有第 i 块与第 2CP−1−i 块，于是每张卡都有 q = ⌈S/CP⌉ 个查询，并且正好承担 1/CP 的因果工作量 S²/2。前缀为 c 的分块 prefill 的注意力工作量是 q·(c + q/2)；取 c = S²/(2·CP·q) − q/2 就与之相等，而所有随 token 线性增长的算子看到的正好是 q 个 token。因此：一张 CP 卡就是一个注意力 DP 副本，对每条序列做一次分块 prefill（虚拟 batch × CP）。实现直接复用 DP 路径：MoE 的 EP all-to-all、稠密 FFN 在副本间复制。

在此之上：
- **常驻 KV**：每张卡只放自己的分片（一层的两个在途环形块不计入）。
- **环形流量**：每个注意力层 all-gather 一次本卡的 KV 块（GQA ⌈kv/TP⌉·(qk+v)，MLA 的 latent，加上 DSA 的 index 键），组内跨度为 TP（卡编号顺序为 TP → CP → DP → PP）。

不成立、因此拒绝的情况：
- 线性注意力（需要串行传递状态）；
- 滑窗与压缩 KV（块等价不成立）；
- decode（需要 KV 分片的解码，未建模）。

**MLA 与 DSA。** 交换 latent 而不是上投影后的 K/V（每 token 576 B 对 82 KB），代价是每张卡都要对收到的块做 kv_b 上投影：各卡键数之和 = CP·(c + q) ≈ S·(CP+1)/2。这是计入工作量的真实成本。DSA 的前 2048 个查询看到的键少于 top-k，块等价让每个查询都按 top-k 计：工作量 KPI 偏高 0.8 %（131K）/ 0.1 %（1M）；步时取最忙的 zigzag 卡，不受影响。

**设计含义（1M prefill，8 卡）。**
- qwen3-8b：CP8 与 TP8 的时间相同（受 MAC 限制），链路时间 0.34 s 对 2.7 s，但每卡要放完整的权重。
- DeepSeek-V3.2：CP8 410 s，对 TP8 2820 s（indexer 在 TP 上重复计算）、TP8 + key_split 425 s；每卡内存 132 GB，对 181 GB。

**审计。**
- 场景 hash：0.70.1、0.80、0.86、0.88 之间完全相同（omit_default 起了作用）。
- 服务并发：修复了监听队列（突发并发下 39–55 % 的连接被重置），并加了 socket 超时。
- MODEL.md 机械一致性：编号、路径、代码引用、常量值都没有问题。

## 35. 第 23 轮（0.89.0）：CP 工作量精确化；结果导出；CP / indexer × PD / fabric / D2D 对抗

**CP 工作量 KPI 精确化（步时不变）。** 分块等价让每张卡都按最忙卡计费。有两处的平均卡工作量其实更少，现在通过 `Op.share`（平均卡有效工作量 / 最忙卡）补正：
- **DSA top-k：** 平均卡的位置里含有前 top-k 个早期位置（看到的键 < top-k）。修正系数 = keys_sum(S) / (CP·q·keys_mean(c, q))，作用于 qk / pv / softmax。
- **LM 头：** prefill 只投影每条序列的最后一个 token，只有持有最后一块的卡（zigzag 卡 0）才做。final_norm / lm_head / sample 的 share ÷ CP（0.88 在 S = 4096 时工作量偏高 3e-5 – 4e-4）。

修正后，242 个随机 CP 场景（含 PD / fabric / D2D / indexer / PP / 微批）与 CP1 对照的工作量完全相同；MLA 的差别只剩 §34 的 kv_b 项。

**fabric `algo = auto` 对链路带宽不单调（已有设计，这里写明幅度）。** auto 对每次集合通信取 min(带宽 + α)，相当于运行时调优器的选法。链路更快时，它可能改选 α 更大、带宽更省的算法，而带宽时间本来就藏在计算之下。结果：D2D + fabric 场景中 19–24 % 出现「链路 ÷4 反而快 ≤ 0.38 %」。`auto_overlap`（按级最小化级时）在同一批场景上 0 / 160 违例，常驻测试锁定这一点。做带宽敏感性分析时请用 auto_overlap。

**结果导出（Web）。** 映射对比、参数扫描、Pareto 三个结果可以导出 CSV / JSON。
- **CSV：** UTF-8 带 BOM；中文表头带单位；数值是原始值（不取整）；防公式注入（以 = + - @ 开头的文本前加 '）。
- **JSON：** 工具 / 版本 / 时间、发起时的场景快照、列定义（key + 中文标签）、逐行数据，以及原始 API 响应（映射对比含各映射的稳定性结果）。
- 结果运行中隐藏导出按钮，避免导出旧结果。
**CP / indexer × PD / fabric / D2D 对抗（242 + PD 随机场景）。** 检查项：不崩溃；输出有限；结果确定、hash 往返一致；与 CP1 工作量守恒（修正后精确）；能耗分项之和等于总量；CP 加速不超线性（0 例）；PD 下 prefill 池用 CP 布局时标签与容量都正确。

报告中 `coloc.ttft_ms / tpot_eff_ms = inf` 出现在合并部署放不下（decode 份额为 0）时，CP1 也一样，不是 CP 引入的。API 序列化成 null（HTTP 200），UI 显示「—」。

## 36. 第 24 轮（0.90.0）：视频 / 蛋白 × 新功能对抗；auto_overlap 提示

**视频 / 蛋白对抗**（17 个模型，196 个随机场景：PP / DP / SP·DAP / TP × D2D × fabric × SLC × node_cards × exec_overlap × layer_overhead）：不崩溃，输出有限，结果确定。修正与结论如下。
- **PD 不适用但被静默接受**（17 / 17 个模型）：现在给出警告，并按单一部署评估。API 不再附带 `pd` 块（以前是 `{error}`），步时与关闭 PD 时完全相同。
- **CP 的拒绝信息**说明原因：上下文并行只用于 LLM prefill，视频请用 sp（Ulysses），结构模型请用 sp（DAP）。indexer 本来就有警告。
- **SLC pin 反而更慢**（esm2，serial 下慢 1.4 %）是真实现象：默认 SLC 带宽 2000 GB/s 低于 HBM3e 8 堆栈的 6595 GB/s，pin 把热数据固定在更慢的 SLC 上。新增警告「SLC 带宽低于 DRAM」。SLC 带宽高于 DRAM 时，SLC 越大越不慢（常驻测试）。
- **结构模型的 MAC 计数随 DAP 变多**：正好是 1 + f·(sp − 1)（protenix f = 0.1145，boltz-1 f = 0.2543），即每张 DAP 卡都重复计算 single / atom 轨道（§ DAP 的「假设」）。这是真实执行的工作，能耗应当计入。其余模型的 MAC 计数与布局无关（≤ 1e-3）。
- **固定算法（ring / tree / hier）同样不单调**：基本算法固定，但协议 / 变体仍按每次集合通信取 min(带宽 + α)（0.61）。例：esmfold DAP、D2D、tree，链路或 D2D ÷4 反而快 58 µs。auto_overlap 单调。

**UI 提示**：「扫描 · Pareto」页在 fabric 开启 + D2D 芯粒（package_cards > 1）且算法不是 auto_overlap 时显示提示。扫描带宽 / α 类参数时措辞更强，并提供一键「切换为 auto_overlap」。

**PD 扫描**（0.89 未完成的部分，可续跑：val5/pd_sweep.sh → pd_sweep.jsonl）：结论见 CHANGELOG 0.90.0。


## 37. 第 25 轮（0.91.0）：SLC pin_bw；外部验证集 r4（TensorRT-LLM 1.2.0）

**SLC `pin_bw`（可选，默认不变）。** 模型里 SLC 和 DRAM 本来就是两条并行读路径（访存时间 = max(DRAM 字节 / B_dram, SLC 字节 / B_slc)）。`pin` 把驻留集的读全部交给 SLC：这是普通缓存的忠实模型（命中就由缓存服务，不会绕回 DRAM），所以保留为默认。SLC 比 DRAM 慢时，pin 会更慢（0.90 的警告）。`pin_bw` 利用包含式 SLC（驻留行在 DRAM 里也有）：驻留的 H 字节中只让 x = min(H, T·r/(1 + r)) 走 SLC（r = B_slc / B_dram，T = 这一步全部 DRAM 字节，包括只能走 DRAM 的 KV 写入 / lookup），两条路径同时读完，访存时间 = T / (B_dram + B_slc)（H 足够时）。相当于取了最优值的 CUDA accessPolicyWindow.hitRatio。性质（tests/test_core_091）：不比 pin、也不比没有 SLC 更慢；SLC 越大越不慢；SLC 快而小（H < x*）时与 pin 完全相同。手算核对：esm2-650m TP2 serial，4096 MiB @ 2000 GB/s：pin 6.201 ms（比没有 SLC 的 5.769 ms 还慢），pin_bw 5.725 ms，t_dram = t_slc = 0.1448 ms。SLC 快时也可能有收益：4096 MiB @ 20000 GB/s，pin 5.642，pin_bw 5.627（pin 让 SLC 过载）。0.437 的 LayerNorm / erf-GELU 只改向量时间，DRAM 时间仍是 0.1448 ms。能量按实际走 SLC / DRAM 的字节计。

**外部验证集复查。**
- **新数据：** TensorRT-LLM 1.2.0 perf-overview（URL 固定版本）。B200 FP4 / H200 FP8 两列，目录里有的 6 个模型（DeepSeek-R1-0528、gpt-oss-120b / 20b、Llama 3.3 70B、Qwen3-235B-A22B、Qwen3-30B-A3B），共 38 行，存为 `ext_measurements_r4.json`。指标是新的 `max_tok_s_gpu`：满载每卡输出 tok/s，即 max_tok_s_total 的同一搜索 ÷ 副本卡数 tp·dp。DEP N = 注意力 DP N + EP N。
- **没有纳入：** GB200 / B300 / RTX 6000 列（没有参考硬件）；gpt-oss 的 H200「FP8」列（发布的权重是 MXFP4，精度含义不明）；ROCm 表里的 Llama 3.1 405B（目录里没有）；MLPerf Inference v5.1 / v6.0（2026-04）：只有数据集的长度分布，没有每次运行的 ISL / OSL，仍只作背景（v6.0 GB300 NVL72 DeepSeek-R1 离线 9,821 tok/s/GPU）。ROCm 的 MI300X 行与固定提交里的数字逐一一致；它被标为「Llama 3.1 70B」，按同构的 llama-3.3-70b 建模（行备注里已注明）。
- **发现的缺陷（已修）：** extval 的 FP8 / NVFP4 格式覆盖没有包含 `expert` 角色，MoE 的路由专家仍按 bf16 计算。修正后，moe 集 max_tok_s_total 的 |log 误差| 从 21.1 % 降到 18.1 %（gmean 0.93 → 1.13）；r4 从 59 % 降到 33 %（gmean 0.53 → 0.83）。r3 的 B200 行都是稠密模型，不受影响。gpt-oss 行用发布的权重格式 + FP8 KV（与 1.2.0 的配置一致）。
- **发现（诊断，默认不改）：** 参考芯片的 `instance_sched = "wide"` 把一个 GEMM 看作横跨全部 SM 列的一个阵列，大的 prefill GEMM 利用率只有 12–24 %（H100 / B200，例如 131072×4096×2048）。`auto`（M / N 按 SM 组切分，GPU 的 CTA 铺法）约 99 %。但用 `auto` 后，吞吐在 v1 / r3 / moe 上偏乐观 1.2–1.75×，误差变大（v1 max 20 → 25 %，r3 36 → 56 %）；只有 r4 和 moe 的 prefill / decode 变好。也就是说，`wide` 的低利用率在替模型缺少的开销（每层 kernel / 同步、实际 GEMM 效率）买单。所以保留默认，新增诊断变体 `extval.sched_variant(base)`（`+A`）。真正的修法是先给 GPU 的 kernel 开销建模，再改 `auto`，这会改变外部验证的数值，需要单独一轮。
- **剩余差距：** r4 的长上下文 / MoE 行仍偏悲观（0.3–0.6×）。原因有两个：一是静态批的 proxy 按最终上下文计 KV 容量，实际 in-flight 批里的请求处在不同阶段，平均上下文约一半，可容纳约 2 倍请求（Qwen3-235B 1024/8192：容量限制 B = 256；不限容量时 4600 vs 实测 3881）；二是上面的 GEMM 利用率。

**对比表**（catalog_kernel，n / gmean 预测/实测 / 平均 |log 误差|；脚本 val5/extval091.py）：v1 max 44 / 1.03 / 20 %，static 36 / 1.17 / 24 %，ttft 20 / 0.63 / 50 %；moe max 27 / 1.13 / 18 %；r3 max 15 / 1.38 / 36 %，latency 16 / 0.31 / 118 %（per-layer overhead 见 §25.8）；r4 38 / 0.83 / 33 %。

Mixtral 8×7B（TRT-LLM 0.17，TP8，FP8 行）在专家格式修正后的数值：Hopper catalog_kernel 的 gmean 0.82 → 1.10（|log 误差| 21 → 16 %），catalog_serial 0.70 → 0.97（36 → 11 %），stage 1.20 → 1.90（stage 仍是宽松上界）。test_core_067 的区间相应更新；以前的区间把 bf16 专家的错误定价当成了基准。A100 行是 fp16，不受影响。

## 38. 第 26 轮（0.92.0）：参考硬件的 GPU kernel 模型；in-flight 批的内存检查

**GPU kernel 模型（实现了，可选；不作为参考硬件默认）。** 常数全部来自公开资料，没有对任何一行拟合：
- `Chip.instance_sched = "cta"`：输出按线程块 tile 切分。tile 集合取自 NVIDIA《Matrix Multiplication Background》用户指南（256×128 / 128×256 / 128×128 / 256×64 / 64×256 / 128×64 / 64×128 / 64×64），由库挑最快的一个，可以 split-K。每个 SM 一次执行一个 tile，所以有 tile 量化（不完整的 tile 也做完整的计算）和 wave 量化（ceil(tiles / SMs) 个 wave，尾 wave 和满 wave 一样长）。一个 tile 的代价 = 它那段 K 上 max(MAC, 操作数供给)，用一个 SM 的 R × C 阵列和 1 / E 的 SRAM 端口；再加上不重叠的 epilogue，写出 tm × tn 个输出（split-K 时写 fp32 部分和，再做一次归约）。分组 / 批量实例共享 wave。核对：用指南里的例子（A100，固定 256×128，M = 2304，K = 4096），N 从 1536 到 1544 时耗时 ×2.0，和指南一致；允许换 tile 后只 ×1.12。大 GEMM 的利用率约 0.99（wide 只有 0.12–0.24）。
- `Chip.kernel_us`：每个 kernel 的固定代价，串行计入，每个 GEMM 和每次融合注意力（QK·softmax·PV）各计一次；逐元素算子按 TRT-LLM 的做法视为融合进 GEMM。取值 0.53 µs：NVIDIA 博客《Constant Time Launch for Straight-Line CUDA Graphs》里，100 个空 kernel 节点的直线图在设备上重复启动用时 53 µs（CUDA 12.6）。

**采纳检验**（val5/adopt092.py → adopt092.txt；r4 = TRT-LLM 1.2 作为留出测试集，什么都不在上面拟合；catalog_kernel，gmean / |log 误差|）：

| 集 | wide（0.91） | wide + in-flight | cta | cta + kernel_us | cta + kernel_us + in-flight |
|---|---|---|---|---|---|
| v1 max（44） | 1.03 / 20.4 % | 1.03 / 20.4 % | 1.23 / 25.3 % | 1.23 / 25.0 % | 1.23 / 25.7 % |
| v1 static（36） | 1.16 / 24.0 % | = | 1.37 / 36.2 % | 1.36 / 35.8 % | 1.36 / 35.8 % |
| v1 TTFT（20） | 0.63 / 50.0 % | = | 0.48 / 72.5 % | 0.49 / 70.5 % | 0.49 / 70.5 % |
| moe max（27） | 1.13 / 18.1 % | 1.15 / 17.1 % | 1.46 / 37.6 % | 1.45 / 37.2 % | 1.51 / 41.1 % |
| moe prefill / decode | 0.56 / 58 %，0.72 / 33 % | = | 0.72 / 33 %，1.27 / 24 % | ≈ | ≈ |
| r3 max（15） | 1.38 / 36.2 % | = | 1.76 / 56.5 % | 1.75 / 55.9 % | 1.75 / 55.9 % |
| r3 latency（16） | 0.31 / 118 % | = | 0.30 / 120 % | 0.33 / 112 % | 0.33 / 112 % |
| **r4（留出，38）** | 0.83 / 32.6 % | **0.85 / 31.2 %** | 1.31 / 29.3 % | 1.30 / 28.9 % | 1.37 / 32.0 % |

结论：
- **cta 不采纳。** GEMM 层面它更贴近真实 GPU，但吞吐在 v1 / r3 / moe 上偏乐观（1.2–1.8×），误差变大；只有 moe 的 prefill / decode 和留出集 r4 变好。kernel_us 只影响 1 % 量级：0.53 µs × 每层约 8–12 个 kernel ≈ 每层 5 µs，远小于 §25.8 的约 100 µs / 层残差。所以公开的 launch 下限解释不了那部分残差，它来自软件栈（调度、主机侧开销、未融合的小 kernel；DeepSeek trace 里每层约 28 个小 kernel，共 125 µs，§25.9）。
- 诊断：即使把 mac_eff 降到 0.7（大约 cuBLAS 实测 / 峰值的量级，但这里没有逐芯片的公开出处，只作诊断），cta 在 moe max 上仍是 31 %，而 wide 是 17 %。可见 wide 的低利用率替模型缺少的、kernel 以外的开销买单，不能简单用 GEMM 效率替代。
- 下一步要的是逐层的软件开销模型，需要有公开出处的 kernel 级 trace。在那之前，参考硬件保持 `wide`，cta / kernel_us 用 `extval.gpu_variant` 选用。

**In-flight 批的内存检查（采纳，作为满载行的默认）。** 以前满载吞吐的静态批 proxy 一次 prefill 全部 B 个 prompt，所以「放不下」常常卡在 prefill 的激活（例：gpt-oss-120b 8192/1024，B = 512 时 KV 放得下，但 512 × 8192 个 token 的激活放不下）。In-flight 批每轮最多 prefill CB_CHUNK = 8192 个 token（TRT-LLM 默认 max_num_tokens），所以激活检查改为按一个 chunk。B 个常驻请求的 KV 仍按 decode 点检查，最长到 ISL + 7/8·OSL，接近 TRT-LLM 默认的 GUARANTEED_NO_EVICT（为每个请求预留完整长度）。吞吐时间不变。影响：headline（catalog_kernel）和 serial 下，没有一个集变差（moe max 18.1 → 17.1 %，serial 10.1 → 8.2 %；v1 / r3 不变），留出集 r4 32.6 → 31.2 %。stage 变体下 moe max 54.9 → 58.1 % 变差，因为 stage 本来就偏乐观，放宽容量后更乐观；stage 是宽松上界，不作为判断依据。静态批行（static_tok_s_*、latency）不变。`extval.INFLIGHT_FIT[variant] = False` 恢复 0.91 的检查。

**更正 §37：** 0.91 说 Qwen3-235B 1024/8192 被容量卡在 B = 256，这个诊断是在专家格式修正之前做的；修正之后 B = 512 就放得下（预测 5316 vs 实测 3881）。r4 剩下偏悲观的行主要是 gpt-oss（0.4–0.7×）、DeepSeek-R1 B200 和 Qwen3 的长序列。

## 39. 0.93 — r4 under-prediction study

Row-by-row (TRT-LLM 1.2.0, hold-out set, nothing fitted on it):

* **Pattern.** Under-prediction tracks prefill share, for every model: 1024/1024 and decode-heavy rows sit at 0.8–1.5×,
  8K/32K-ISL rows at 0.4–0.7×. Cause is structural: the refhw default `instance_sched = "wide"` gives 12–24 % GEMM /
  attention utilisation on large prefill work (§37). The `cta` model (§38) fixes prefill but over-predicts decode-heavy
  rows, so it is still not the default.
* **gpt-oss (0.42–0.69×).** Sliding window (128, alternating) and attention sinks were already modelled. Missing was the
  Blackwell MoE path: MXFP4 weights × **MXFP8 activations** (TRT-LLM `examples/models/core/gpt_oss/README.md`, the default
  for CUTLASS / TRTLLM backends; Hopper stays bf16). Fixed: expert GEMMs now run at FP8 rate. gpt-oss rows move to
  0.52–1.43×. The rows that remain low are the long-ISL ones (the prefill pattern above).
* **DeepSeek-R1 B200 (0.67–0.86×).** The NVFP4-v2 checkpoint quantises only MoE linears + attention `wo` (model card).
  Attention is now FP8, which is slower, so this fix moves the rows the wrong way for the ratio but is correct. The
  dominant term is MLA decode attention under `wide` (57 of 85 ms per step). The public DeepSeek trace has it at about 1/3
  of the `wide` value (§25.9).
* **MTP.** Ruled out: the published 1.2.0 configs carry no `speculative_config`.
* **FP4 attention.** TRT-LLM never runs attention math in FP4. FP8 context FMHA exists but is not documented as the
  PyTorch-backend default, so it is not modelled. Generation dequantises the FP8 KV in-kernel, which is what the model
  already does.
* **Concurrency.** trtllm-bench uses concurrency −1 with CUDA-graph batch sizes up to 8192, and the batch search covers
  that range. DeepSeek B200 decode is KV-capacity-bound at about 256 requests per GPU in both the model and the configs.

**Per-layer overhead grounding.** Public per-layer traces: TRT-LLM ships the tooling (`layer_wise_benchmarks`, perf-analysis
guide) but publishes no numbers. The only public kernel-level trace found is deepseek-ai/profile-data (§25.9): 1,734
small kernels taking 7.6 ms per decode step, across 61 layers × 2 micro-batches. That gives **62.3 µs per layer per
forward pass** (≈ 14 kernels × 4.4 µs), against the ~100 µs from §25.8. `extval.TRACE_LAYER_US`.

**Candidate.** `candidate_variant()` = attention-only `cta` (`Chip.attn_sched`: FMHA / FlashMLA tile (request, head)
over SMs) + 62.3 µs/layer. GEMMs stay `wide`.

| group | default | candidate |
|---|---|---|
| v1 max / static / ttft | 20.4 / 24.0 / 50.0 % | 17.4 / 20.0 / 28.5 % |
| moe decode / max / prefill | 33.4 / 17.1 / 57.8 % | 10.5 / 14.1 / 45.5 % |
| moe static (1 row, trt-dsr1-h200) | 30.0 % | 63.2 % |
| r3 max / latency | 36.2 / 118.4 % | 30.0 / 32.8 % |
| r4 (hold-out) | 26.7 % | 19.2 % |

It fails the "no set worse" rule on the one-row group, so it is not the default. That row and the r4 H200 DeepSeek rows are
already over-predicted under the default (1.35×, 1.53×), which points to a separate H200 DeepSeek issue.

## 40. 0.94 — H200 DeepSeek over-prediction

Rows: trt-dsr1-h200-maxtput (0.21 blog, 1.35× default / 1.88× candidate) and r4 H200 DEP8 FP8 (1.53× / 2.64× at
1024/1024). Causes checked, each against public sources only:

1. **Hopper FP8 block-scaled GEMM efficiency.** DeepGEMM's public H800 table (README @55ab91f7) replayed through the
   GEMM model: `wide` gmean 1.27×, `cta` 1.61× (`cta` reaches 98 % of peak, while DeepGEMM's best is 1550 TFLOPS = 78 %).
   Memory-bound M = 64 / 128 shapes measure 1.5–2.7 TB/s, consistent with the catalog DRAM efficiency of 0.7.
   → `Chip.fmt_ceiling` ("fp8", 1550 / 1979), Hopper only.
2. **MLA kernel.** FlashMLA README: dense MLA decoding reaches up to 660 TFLOPS on H800 (bf16 peak 989.5).
   → ("attn:bf16", 0.667). TRT-LLM documents FP8 MLA generation with FP8 KV (FP8 FlashMLA on Hopper) but publishes no
   rate for it, so it is not modelled.
3. **Published configs.** 0.21 blog: `--max_batch_size 128 --max_num_tokens 1151 --kv_cache_free_gpu_mem_fraction 0.8`,
   concurrency 1024 (batch fixed, so the fit is irrelevant for this row). 1.2.0: no MTP, CUTLASS MoE, FP8 KV.
   Capacity: the catalog stacks give H200 144 GiB and B200 192 GiB, while NVIDIA's published usable figures are
   141 GB and 180 GB. → `mem_cap_GB` and `kv_mem_frac` (TRT-LLM 0.9; vLLM 0.9 of total).
4. **EP/TP.** DEP8 = attention DP + EP8, as modelled. The MoE all-gather / reduce-scatter cost is about 2 ms per step
   (t_link), so it is not the gap.

Effect (error per group; `default_old` = 0.93 default):

| variant | moe dec / max / pf / static | r3 max / lat | r4 | v1 max / static / ttft |
|---|---|---|---|---|
| default (0.93) | 33.4 / 17.1 / 57.8 / 30.0 | 36.2 / 118.4 | 26.7 | 20.4 / 24.0 / 50.0 |
| default + fixes | 33.8 / 17.3 / 58.8 / 29.8 | 33.9 / 118.4 | 28.9 | 17.7 / 22.1 / 49.6 |
| candidate 0.93 | 10.5 / 14.1 / 45.5 / 63.2 | 30.0 / 32.8 | 19.2 | 17.4 / 20.0 / 28.5 |
| candidate 0.94 | 5.4 / 12.9 / 48.2 / 60.1 | 28.5 / 32.8 | 18.1 | 18.3 / 18.2 / 27.1 |

Candidate 0.94 is not the default, because the trt-dsr1-h200 group is still worse (1.82× vs 1.35×). The fixes are not
the default either: alone, under `wide`, they worsen r4 and moe decode / prefill. All of them stay opt-in.

**Residual (unexplained).** H200 DeepSeek decode at 128 req/GPU: the measured step is about 88 ms (11.35 tok/s/user);
the candidate gives 42 ms, DRAM-bound (expert weights about 82 GB/step). The 1.2.0 1024/1024 row is still 2.4×.
Candidates without public numbers: masked grouped-GEMM efficiency at M ≈ 32 per expert, attention-DP imbalance, and
host / detokenisation overhead at concurrency 1024 (the GPT-OSS blog notes it "cannot be hidden" at high concurrency).

## 41. 0.95 — NPU design-space adversarial review

Base: `1P` (32 × 128 × 128 engines, 1 GHz, 256 MiB SRAM), qwen3-8b / qwen3-30b-a3b / qwen3-32b, HBM3e 8 stacks and
LPDDR6 8×96. Scripts: `val5/npu095/{sweeps,sweeps2,sens,mono}.py`.

**Monotonicity.** None of the SRAM, engine, clock, DRAM-efficiency or memory-bandwidth sweeps lowers throughput.
The randomised check (≈ 890 pairs, three overlap modes, five models, SLC policies) has no violations other than
capacity-losing memory steps.

**Bottleneck flips, and why they happen:**
* **Engines 16 → 32, and clock 0.5 → 1 GHz (decode, any batch ≤ rows): MAC → DRAM.** Weight-stationary B = 1 decode
  loads each 128 × 128 tile in `rows` cycles, so the array takes in weights at engines · cols · 2 B · f = 4.1 TB/s
  (16 engines), which is below the 4.6 TB/s effective HBM bandwidth. 32 engines (8.2 TB/s) are above it. The flip is the
  same at B = 64, because M < rows still leaves the tile load-bound.
* **Partition 8 × 256² vs 32 × 128²** (same MACs): fewer, larger arrays take in half the weights per cycle, so decode
  becomes MAC-bound. Prefill is nearly unchanged.
* **Memory HBM3e → HBM4 (decode): DRAM → MAC.** 11.5 TB/s exceeds the 8.2 TB/s array intake.
* **SRAM on LPDDR prefill 256 → 512 MiB: DRAM → MAC.** With a smaller SRAM the weight tiles are re-streamed per
  activation M-tile (budget SRAM / 2). On HBM the re-streaming is hidden.
* **DRAM efficiency 0.7 → 0.8 / 0.9: DRAM → MAC.** Same crossing, on the efficiency axis.

**Conclusion sensitivity** (base → value under the perturbed constant):

| conclusion | base | exec_overlap kernel / serial | mem_eff 0.5 / 0.9 | mac_eff 0.8 | vector_lanes ÷ 4 |
|---|---|---|---|---|---|
| C1 HBM / LPDDR6 decode ratio | 10.4 | 10.2 / **6.0** | 10.4 / 9.5 | 9.8 | 10.4 |
| C3 LPDDR6 prefill SRAM knee (95 %) | 128 MiB | 128 / **512** | 256 / 64 | 64 | 128 |
| C4 engines for 95 % B = 1 decode | 32 | 32 / **64** | 32 / **64** | **64** | 32 |
| C5 FP8 act + KV gain (HBM, B = 64) | 1.20 | 1.05 / 1.19 | **1.68** / 1.16 | 1.16 | **0.72** |
| C2 D2D 16 / 1 units, prefill | 1.00 | **1.09** / 1.09 | 1.00 | 1.00 | 1.00 |
| C6 2-die / monolithic, decode | 0.995 | 0.99 | 1.00 | 1.00 | 1.00 |

No effect on any conclusion: acc_kib 256 / 4096, gemv_macs, noc_hop_cycles 20, sram_port ×4, d2d α 5 µs (except C6
0.96), wave_overhead 500 (C5 1.15). `sram_port ×0.25` changed C6 (2-die 1.10×), but that is a harness artifact: an
explicit `sram_port_Bpc` is absolute, so it does not shrink with engines when a die is halved. Pass it scaled when you
sweep engines.

**Conclusions driven by an arbitrary constant (treat them as assumption-conditional):**
1. **exec_overlap = "stage"** (one roofline per stage). It hides TP / D2D collectives completely, which is why D2D
   bandwidth "does not matter" (C2). Under serial overlap the HBM-vs-LPDDR ratio drops from 10.4 to 6.0, and the SRAM
   knee and the engine count needed both double. **Highest-impact constant.**
2. **mem_eff = 0.7** (one value for HBM and LPDDR). It sets every decode absolute number, the position of the
   compute / memory flips (C4) and the size of the FP8 gain (C5). Per-type efficiency is not modelled; if LPDDR is
   less efficient than HBM, C1 grows.
3. **vector_lanes = 4 · cols · engines.** With FP8 KV the attention core upcasts the cache on the vector path every
   decode step. At ÷4 lanes, FP8 KV makes decode **slower** (0.72×). Native FP8 attention (no upcast) is not a modelled
   option, so "FP8 KV helps" holds only when the vector budget is large enough.
4. **mac_eff = 1.0 / weight-tile load = rows cycles.** These set the engine count at which decode turns memory-bound (C4).

The fix found along the way: `dequant_elems_per_cycle` is now honoured (it was a dead knob).

## 42. 0.96 — follow-ups to the §41 review

* **Native KV-format attention** (`Chip.attn_fmt = "kv"`, opt-in). §41 found that "FP8 KV helps" depends on the
  vector-lane constant, because the bf16 attention core up-casts the whole cache every decode step. The native path
  computes QK / PV in the KV format: Q is quantised (one vector op per Q element) and P is cast inside the softmax
  epilogue (FlashAttention-3 FP8 / FP8 FMHA). 1P, qwen3-8b, FP8 W/A/KV, B = 64 decode: 10.8k → 15.5k tok/s; at ¼ of
  the lanes 6.5k → 15.5k. The result no longer depends on vector lanes. A KV format the chip does not support natively
  falls back to the default path.
* **Per-engine SRAM port** (`Chip.sram_port_Bpc_per_engine`) removes the §41 die-split trap.
* **Sensitivity panel.** Every `/api/eval` result carries `sensitivity`: re-evaluations with no overlap, DRAM efficiency
  0.5 / 0.9 and ¼ of the vector lanes. A row hinges when |Δ| > 15 % (a display threshold 「假设」) or the bottleneck flips.
  The UI shows the table (敏感度) and a one-line verdict. For DRAM-bound points, DRAM efficiency always hinges (the result
  scales with it): treat absolute numbers as conditional on that constant, and compare design points under the same value.
* **HBM vs LPDDR efficiency.** Kept at one value of 0.7. Public STREAM / BabelStream data shows no systematic gap (HBM3
  H100 Triad ≈ 87–89 %; Grace LPDDR5X 80–95 %, NVIDIA up to 98 %, independent 90.6 %). Both sit well above 0.7, which
  stands for inference traffic (mixed read / write, irregular KV), not for streaming. Type-specific defaults would need
  inference-traffic measurements per type, and none are public.

## 43. 0.97 — adversarial review of 0.96, and sensitivity for sweeps / Pareto

* **FP8 attention vs FA3.** FA3 on H100 SXM5 measures FP16 ≤ 740 TFLOPS and FP8 ≈ 1.2 PFLOPS (≈ 1.6×, hd 256). The
  exponential runs on the MUFU at the same rate in both precisions, so FP8 gains less than 2×. The model keeps softmax
  on the vector path, which reproduces this structure. H100-like preset: 1.76× serial / 2.0× overlapped at hd 256, and
  1.61× / 2.0× at hd 128. The measurement lies in that bracket. Absolute numbers depend on the preset's `mac_eff`.
  DiT full-sequence attention (`orient` ops) runs in the activation format and ignores `attn_fmt`; use an FP8
  activation override there.
* **Per-engine SRAM port × D2D.** Bug fixed: the engine-group split crashed when the field was set. At fixed silicon
  (32 engines split over 1 / 2 / 4 dies, port-bound) the per-engine port gives the same throughput within D2D traffic
  (6.47k → 6.30k → 5.98k tok/s prefill). The absolute port reproduces the trap (6.47k → 12.4k → 14.7k).
* **Sensitivity panel.** The base value equals the summary, including under best_batch (the UI re-sends the chosen
  batch). PD pools are outside its scope (`scope`). Cold server latency: +4–25 ms for LLMs, +0.4 s for DiT, hence
  `/api/sensitivity` and asynchronous filling.
* **Sweep / Pareto.** Monotone resource sweeps never swap order. Their design conclusion is the knee, and the knee
  moves under the alternatives. Examples (1P):
  - LPDDR prefill, SRAM sweep: knee 64 MiB; no overlap → 256 MiB; DRAM efficiency 0.5 → 128 MiB.
  - qwen3-8b HBM decode, clock sweep: knee 0.8 GHz; no overlap → 1.5 GHz; DRAM efficiency 0.9 → 1.2 GHz.
  - DeepSeek-V3 decode, link sweep: 50 GB/s is enough; no overlap → 200 GB/s.
  - FP8 batch sweep: knee 128; ¼ lanes → 16, with the bottleneck flipping to VECTOR.

  These are the conclusions to report with their assumption. The batch Pareto front was unchanged in every case
  tried: it is capacity-limited and monotone in batch, and capacity does not depend on the three constants. The flags
  are still there for safety.

## 44. 0.98 — hardware-derived overlap (`exec_overlap = "pipe"`, opt-in)

§41 and §43 showed that the overlap rule is the constant design conclusions hinge on most. `stage` (everything
overlaps) and `serial` (nothing does) bracket reality. `class` / `kernel` / `tbo` are fixed mixtures. `pipe` derives
the overlap from the mechanism. Rules and constants are in core/overlap.py; summary:

* **Double buffering.** The stage's ops are replayed in program order on two engines. The DMA engine streams each
  op's DRAM bytes: touched bytes × the plan's miss fraction per class, scaled so the stage's DRAM / SLC time is
  unchanged. It may run ahead of compute by at most the SRAM not used for pinned weights and cached KV, which is at
  least the staging area. Compute processes a tile once it has landed. More SRAM means a larger lookahead and fewer
  bytes, so the result is monotone; checked over 1 MiB … 1 GiB.
* **Tiles, prologue and epilogue.** An op is cut into max(32, ⌈bytes / 1 MiB⌉) tiles 「假设」. The tiling is fixed
  by the op, so pinning more of an op shrinks each tile without re-tiling it. The first load and the last compute of
  an op are exposed unless a neighbour covers them.
* **Dependencies.** Inside an op the array and the vector unit pipeline (max). QK · softmax · PV is one fused step
  that carries the K/V reads. Separate vector ops follow the op they depend on. A collective blocks the compute chain,
  but the DMA keeps prefetching. MoE dispatch / combine overlap the shared expert of the same layer. A PP hand-off
  is exposed; FSDP gathers are prefetches.
* **Kernel boundaries** (`Chip.pipe_cross_op = False`). No prefetch across ops, as on a GPU without programmatic
  dependent launch: every op pays its own prologue and epilogue.
* **DMA issue rate** (`Chip.dma_engines × dma_outstanding_KiB / dram_latency_ns`). This caps the DRAM bandwidth in
  every mode. Opt-in only: there are no public NPU figures.

**Where it sits.** On 1P, pipe ≈ stage within 0–3 % for single-card decode and prefill: the vector ops and tile
edges are the only exposure. DeepSeek-V3 TP8·EP8 decode: 1.16× stage (class 1.13×, serial 2.0×), because the
attention and shared all-reduces are exposed. Prefill: 1.74×. Random set: median 6 % of the way from stage to serial.

**Refhw.** With kernel boundaries, pipe improves moe pf / static, v1 ttft / max, and worsens r3 max (36 → 43 %),
r4 (26.7 → 28.3 %) and v1 static. On top of the candidate it improves 7 of 10 groups, but r3 max and moe dec worsen.
Not adopted: rule "every group improves". The measured GPU rows sit between kernel-serial and pipe. Prologue and
epilogue alone do not explain the gap, which is consistent with §39's per-kernel fixed costs.

**NPU default.** Kept at `stage`. Pipe is better grounded in mechanism, but it is not better validated: there are no
NPU measurements, and on the catalog it moves 8 of 89 points by at most 3 %. Switching would change the fingerprint
c034bfe3 → f7963e41. Use pipe, and the sensitivity panel's pipe row, to see whether a conclusion survives a
mechanistic overlap model.

## 45. 0.99 — how robust are the knees under pipe; pipe's own constants

**Knees** (0.97 table, knee = first value within 5 % of the best; stage → pipe → serial):

| sweep | stage | pipe | serial | verdict |
|---|---|---|---|---|
| HBM decode clock | 0.8 | 0.8 | 1.5 | robust (holds under realistic overlap) |
| HBM FP8 decode clock | 1.5 | 1.5 | 1.5 | robust |
| decode batch | 128 | 128 | 128 | robust |
| mac_eff | 0.9 | 0.9 | 0.9 | robust |
| LPDDR prefill SRAM | 64 | 128 | 256 | shifts: exposed tile edges and vector ops make 64 MiB "not enough" |
| D2D package cards | 1 | 2 | 1 | shifts (not monotone across modes: pipe exposes collectives that D2D shortens) |
| DeepSeek-V3 link GB/s | 50 (flat) | 400 | 200 | shifts: under stage the link hides behind compute. With dependent collectives on the chain, link bandwidth matters up to 400 GB/s |
| Wan frames | 33 | 17 | 17 | shifts |

Under stage, a conclusion like "link bandwidth doesn't matter" for MoE EP decode is an artefact of full overlap.
Quote such conclusions only with the pipe column.

**Pipe constants.** Tile 256 KiB … 4 MiB and minimum 8 … 128 tiles move pipe/stage by ≤ 0.7 % on all 7 cases,
with cross-op prefetch on and off. The exception was a tile larger than half the buffer, where double buffering is
impossible. That is now prevented: tile = min(1 MiB, buffer / 2). The conclusions do not hinge on these constants.

**Exactness.** Pipe is now exact for every stage, DiT and protein included. A repeated layer group is fed one
repeat at a time. Once two consecutive repeats leave the same state relative to the compute clock (DMA lead,
buffer window), the rest of the group is added in one step. This is exact, because the flow shop is deterministic.
Worst catalog evaluation: 1.9 s.

## 46. 0.100 — pipe is the default overlap

`Scenario.exec_overlap` now defaults to `"pipe"` (§44, §45). The reason is §45: under `stage`, the LPDDR prefill
SRAM knee (64 → 128 MiB), the D2D package knee and "MoE EP decode is insensitive to link bandwidth" were all
artefacts of full overlap. Pipe derives the overlap from double buffering and data dependencies, and its own
constants move results by ≤ 0.7 %.

* `stage` and `serial` remain selectable. In the sensitivity panel they bracket pipe as the upper and lower bounds
  on throughput.
* Refhw (extval) sets its overlap explicitly, so all 10 validation groups are unchanged. GPU measurements are
  still judged under the 0.92–0.94 variants (§44: pipe was not adopted for refhw).
* Fingerprint c034bfe3 → e51c7ec1: 0/209 points bit-identical, median ratio 1.000, worst −3.0 % (wan2.1-1.3b
  TP2), 5 points beyond 1 %, no bottleneck flips. Explicit stage reproduces c034bfe3.
* Two analytic statements that used to hold in the default become regime-dependent; their tests are now pinned to
  stage:
  * chunked-prefill DRAM per token ≤ prefill-first: under pipe +5 % at the test point;
  * the chunk-share identity (§23): 1.8e-6 off under pipe.
* Saved partial scenarios keep their hash but evaluate under pipe. Full UI dicts carry `"stage"` and reproduce.

## 47. 0.101 — pipe as default: adversarial review

**Bug found.** The co-located chunked-prefill iteration (`pdqueue._fused_step`) always used the stage rule, so
under the pipe default it was the only part of the PD report that ignored overlap. It now follows the mode
(CHANGELOG 0.101). Stage and refhw are bit-identical.

**Invariants under pipe (206 random combos: PD, CP, PP, EP, D2D, SLC, fabric, video, protein).**
* No stage falls outside [stage, serial].
* Action counts (MAC, SRAM, DRAM, link and wire bytes) do not depend on the overlap mode. Only idle card-seconds
  move, i.e. static energy, and they grow with the step time.
* No monotonicity violations for link, D2D, SLC, SRAM or mem_eff.

**PD.** Ordering stage ≥ pipe ≥ serial holds for fluid capacity, PD, prefill-first and chunked co-location
(qwen3-8b, qwen3-30b-a3b). One regime effect: qwen3-30b-a3b PD is unstable under stage ("KV link unstable after
the pool's collectives") but stable under pipe at 40 req/s. The KV transfer shares the link with the pool's
collectives. Stage packs those collectives into a shorter step, so they leave less link time for KV. This is a
property of the stage assumption, not a pipe artefact.

**Knees under the default** (pipe base; stage and serial from the sensitivity cases; same sweeps as §45):

| sweep | pipe (default) | stage | serial |
|---|---|---|---|
| HBM decode clock | 0.8 | 0.8 | 1.5 |
| FP8 decode clock | 1.5 | 1.5 | 1.5 |
| decode batch | 128 | 128 | 128 |
| mac_eff | 0.9 | 0.9 | 0.9 |
| LPDDR prefill SRAM (MiB) | 128 | 64 | 256 |
| D2D package cards | 2 | 1 | 1 |
| DeepSeek-V3 link (GB/s) | 400 | 50 (flat) | 200 |
| Wan frames (best) | 17 | 33 | 17 |

These match §45 exactly, now read from the default side.

**0.95 constant table under the default** (qwen3-8b on 1P unless noted; MODEL §41 conclusions C1–C6):

| conclusion | 0.95 (stage) | 0.101 default (pipe) | moved by (under pipe) |
|---|---|---|---|
| C1 HBM / LPDDR6 decode ratio | 10.35 | 10.35 | serial 6.03; mem_eff 0.9 9.51; mac_eff 0.8 9.78 |
| C3 LPDDR6 prefill SRAM knee | 128 MiB | 128 MiB | serial 512; mem_eff 0.5 256; mem_eff 0.9 / mac_eff 0.8 64 |
| C4 engines for 95 % decode b1 | 32 | 32 | serial 64 |
| C5 FP8-activation gain, HBM decode b64 | 1.20× | 1.30× | dequant 64/cyc 0.37×; lanes ÷4 0.85×; mem_eff 0.5 1.58×; kernel 1.05× |
| C5b int4 vs FP8, LPDDR6 decode | 1.14× | 1.14× | dequant 64/cyc 0.75× |
| C2 D2D 16 vs 1 unit, prefill | 1.00× | 1.10× | d2d α 5 µs 1.00× |
| C6 2-die / mono, decode | 0.995 | 0.995 | d2d α 5 µs 0.957 |
| C6b 2-die / mono, prefill | 0.999 | 0.968 | lanes ÷4 1.08; port ×0.25 1.07 |

Pipe's own constants (tile 256 KiB / 4 MiB, minimum tiles 8 / 128) and `pipe_cross_op=False` move every entry by
≤ 3 %.

What changed under the default:
* **C2.** D2D units now matter: +10 % from 16 units, where stage hid the D2D transfer behind compute.
* **C6b.** A 2-die prefill is 3 % slower than a single die.
* **C5.** The FP8-activation gain rises from 1.20× to 1.30×.

What did not change:
* the HBM / LPDDR ratio, the SRAM knee at qwen3-8b 4K, and the engine count;
* the conclusions' dependence on vector lanes and dequant throughput, which is still the largest
  (C5: 0.37–1.58× across knobs). That dependence is not an overlap effect.


## 48. 0.102 — op-level chunk iteration; vector / dequant sanity

**Chunk iteration under pipe.** Each co-located iteration (decode batch k ⊕ an f share of a prompt) is now
merged op by op and replayed on the pipe flow-shop under the decode's memory plan (`overlap.merge_chunk`). The
0.101 interpolation is no longer used.

- GEMMs fuse at their real merged shape.
- Attention for the chunk is its own fused step, plus rr × the prompt's KV writes.
- Vector ops and collectives scale × f, which is exact because they are linear in tokens.

Audit results (72 points across 6 model×layout combinations, k∈{1,16,128}, f∈{0,1/8,1/2,1}):

- Every point is aligned (no fallback).
- Stage ≤ pipe at every point.
- Pipe ≤ the linear-scaled serial bound, except qwen3-32b PP2 by +0.1%. There the chunk's own GEMMs are less
  efficient than f × the prompt GEMM, which is a real effect the old serial bound missed.

DES validation (`val5/v0102_des.py`, V4 dense8b / moe30b at load 0.3 / 0.6 / 0.85, coloc_chunked, 2 seeds):

- The analytic-vs-DES errors are within ±0.02 of 0.101 at every point.
- Both sides share the step model, so the tails at load 0.85 (+0.2–0.45) are queueing effects, as before.

Remaining approximations in the merged iteration:

- Chunk attention is taken as f × the whole-prompt attention. This averages exactly over the chunks of a prompt.
- MoE expert weights touched by the chunk's tokens are not added. This is unchanged from the stage formula.

**Vector lanes / dequant** (`core/vecsanity.py`). Two ratios are compared against public chips:

| ratio | public chips | presets |
|---|---|---|
| MAC/cycle ÷ lanes | MI300X 8, H100 14.8, A100 16, TPU v5p 16, TPU v4 32, Ascend 910 32 (FP16 vec) / 64 (FP32) | 100T 14, 1P 32, H100-like 16 |
| int4 dequant headroom (conv elems/s ÷ 2·DRAM B/s) | 1.7–7.7 | 100T+LPDDR6 2.8, 1P+HBM3e 1.24 (below band), 100T+HBM3e 0.27 (below band; an unbalanced pairing) |

- The default lanes = 4·cols·engines puts every preset's lane ratio inside the public band, so no preset is
  changed.
- The headroom is low only for the HBM pairings. There, weight-only int4 decode becomes dequant-bound, and the
  new warning says so.
- Caveat: GPUs run exp on SFUs at 1/8 of the FP32 lane rate (FA3: 3.9 TFLOPS exp on H100). The model charges
  exp at the lane rate, so softmax-heavy conclusions on GPU-like vector budgets are optimistic.


## 49. 0.103 — SFU rate; MoE experts touched by a chunk

**SFU** (`Chip.sfu_per_cycle`, opt-in). On GPUs, exp, rsqrt and reciprocal run on the MUFU/SFU at 16 results per
clock per SM, against 64–128 FP32 lanes (CUDA C++ Programming Guide, throughput of native arithmetic
instructions). An op's vector time becomes max(vec / lanes, sfu / sfu_rate). The NPU presets keep None, so they
are bit-identical.

Refhw groups (mean |log err| %, 10 groups):

- catalog: identical with and without SFU.
- catalog_kernel + SFU: 6 groups better or equal; moe pf +0.6, moe dec +0.5, r4 +1.3.
- cand093 + SFU: moe dec 5.4→4.7, r3 max 28.5→27.8; moe pf +0.7, r4 +1.4, v1 max +0.2.

So the SFU is not adopted. A likely reason: FA3-style kernels overlap exp with the MMAs (ping-pong warpgroups),
while the kernel-serial variants charge it serially. That makes the already under-predicted long-sequence r4
rows worse. Not every group improved, so the variant stays opt-in.

**MoE chunk experts.** A co-located iteration's routed experts are the union over decode plus chunk tokens:
hit = n_local·(1−(1−k/E)^T), the same uniform expectation as `ir.moe_routing`, with skew handled the same way.

- Extra DRAM bytes = the decode's expert DRAM bytes × (hit_union / hit_dec − 1). These are added to the fused
  DRAM in every mode.
- The pipe replay merges each expert GEMM into hit_union GEMMs of m_e rows.

Effect (1P + HBM3e, chunk = 1/32 of a 4K prompt; ratios are fused / decode step):

| scenario | before | after |
|---|---|---|
| qwen3-30b-a3b, k=1 | 1.3× | 3.5× (stage) / 4.1× (pipe) |
| deepseek-v3 TP8 EP8 | 1.3–1.6× | 5–6× |
| k=128 (all experts already hit) | unchanged | unchanged |

DES (V4 moe30b, coloc_chunked, loads 0.3 / 0.6): the analytic-vs-DES errors stay within the 0.102 band, since
both sides use the same step model.


## 50. 0.104 — FA3 softmax overlap; H200 DeepSeek gap; chunk-MoE invariants

**FA3 overlap** (`Chip.attn_vec_overlap`, opt-in). In the kernel and tbo modes, the attention core costs
max(t_arr_attn, t_vec_attn, t_dram_kv) and the remaining vector kernels add serially. Under this overlap, the SFU
rate is charged where FA3 actually has it, so the 0.103 test is now fair.

Refhw groups (mean |log err| %; `val5/v0104_groups.log`):

| variant | moe pf | moe dec | moe static | moe max | r3 max | r4 | v1 static | v1 ttft | v1 max |
|---|---|---|---|---|---|---|---|---|---|
| cand093 | 48.2 | 5.4 | 60.1 | 12.9 | 28.5 | 18.1 | 18.2 | 27.1 | 18.3 |
| +fa3 | 47.3 | 6.5 | 60.8 | 14.4 | 33.2 | 16.7 | 18.5 | 27.4 | 14.6 |
| +fa3+sfu | 47.5 | 6.4 | 60.7 | 14.2 | 32.4 | 17.1 | 18.2 | 27.1 | 14.6 |

The overlap helps the rows where attention is long (r4, v1 max). It hurts the rows that were already
over-predicted (r3 max, moe max). Neither change improves every group, so both stay opt-in.

**H200 DeepSeek decode gap** (trt120 DEP8 1024/1024, measured 1,724 tok/s/GPU):

- The max-load proxy `predict()` takes the best static batch that fits. It picks b = 4096 (TTFT 46 s, prefill
  share 37 %) and predicts 4,201 (2.44×).
- The in-flight-batching proxy `predict_cb` predicts 2,051 (1.19×) at b = 1024.
- At 1024/8192 both proxies are within 1.05×.

So the gap is a serving-proxy effect (batch choice under `--concurrency -1`), not a kernel effect. The published
config (perf-overview 1.2.0) uses the CUTLASS MoE backend on Hopper, not DeepGEMM; the candidate's
DeepGEMM-derived FP8 ceiling may therefore be optimistic for MoE. No public CUTLASS blockwise-FP8 grouped-GEMM
efficiency is available, so nothing is changed. Open: choosing the max proxy per source (static vs CB) would need
a rule that is not fitted to r4.

**Chunk MoE invariants.** A randomized audit (4 MoE models; EP / ETP / PP; skew 1.0 or 1.5; stage, pipe, kernel
and serial; 2 memories) checked five properties. Extra expert bytes are ≥ 0 and monotone in f. The fused step is
monotone in f and never below the decode step. Hit experts satisfy hit_dec ≤ hit_union ≤ n_local. Result: 149
points, 0 violations.


## 51. 0.105 — per-source batching estimator (`est_variant`); randomized audits

**Per-source estimator** (`extval.SOURCE_ESTIMATOR`, `estimator()`, `est_variant()`, opt-in). Every max-load row
comes from an in-flight / continuous-batching engine that is fed the whole dataset at once (TRT-LLM
`trtllm-bench throughput` 0.17 / 0.21 / 0.21-b200 / 1.2.0; ROCm vLLM 0.11.2 offline). So each source gets the
continuous-batching proxy `predict_cb` with that engine's published defaults, which are not fitted:

- TRT-LLM: chunk budget `max_num_tokens` = 8192.
- vLLM `LLM` on MI300X: `max_num_batched_tokens` = 16384, `max_num_seqs` = 1024 (`predict_cb(max_seqs=…)`).

Refhw groups (mean |log err| %, `val5/v0105_groups.py`):

| base | moe max | r3 max | r4 | v1 max |
|---|---|---|---|---|
| cand093 | 12.9 | 28.5 | 18.1 | 18.3 |
| cand093 + est_variant | 18.4 | 27.9 | 24.5 | 18.1 |

catalog r4 goes from 32.1 to 42.4. The CB proxy lowers max-load predictions. That fixes the H200 DeepSeek
outlier (2.44× → 1.19×, §50) but makes the r4 rows worse, since they were already under-predicted. Not adopted
as the default.

**Audits** (`val5/v0105_audit.py`, `v0105_pd_q.py`):

- CP: 40 points (dense and MoE with ep = tp·cp; cp 1 / 2 / 4; S 8K–131K; pipe, stage and kernel). Work is
  conserved to 1e-6, and a cp rank is never more than cp× slower. 0 violations.
- Sweep vs eval: 20 sweeps, every row exactly equal to `api_eval`. Pareto: 20 fronts, all non-dominated.
- PD, 60 points: fluid capacity (`cap_req_s`, min over pools) is monotone in KV bandwidth ×4, cards ×2,
  prefix hit 0.5 and link ×2. 0 violations.
- PD TTFT. The top-level `ttft_ms` is the TTFT of the prefill batch that `best_prefill` picks: the
  largest-throughput batch that still meets the TTFT SLO. When prefill gets faster (a prefix hit, or a faster
  link on a LINK-bound prefill), a larger batch fits the SLO, so the reported TTFT can rise. Example: DeepSeek-V3
  TP8·EP8 with link ×2 goes from b = 32 to b = 64, and TTFT from 1361 to 1695 ms. At b = 64 the base would be
  2722 ms. This is a batch choice, not a latency regression. It happens both at a relative load and at a fixed
  absolute rate (38 of 180 checks).
  - At a fixed absolute rate (`pd.rate_rps`), the queueing TTFT (`queue.modes.pd.ttft_ms`, mean / p90 / p99)
    never rises: 0 violations.
  - The fluid TTFT never rises at an unchanged batch.
  - Standing test: `test_core_105::test_pd_ttft_never_rises_with_prefix_or_link_at_fixed_rate`.


## 52. 0.106 — genomics (Carbon); attention-mask mode

**Carbon-A** (`HuggingFaceBio/Carbon-A-1.2B`, `model_type` generanno). The blocks are Llama-style (RMSNorm, GQA
32 / 4 with head_dim 64 and RoPE θ 5e5, SwiGLU 5632), but attention is bidirectional. Each layer is built with
`AttnCore.causal = False` and runs on the full-sequence (encoder) path. That path charges the whole S × S score
matrix (`_full_attn`, causal factor 1), has no KV cache, and writes no KV.

- Output: the two strand heads `score.0` / `score.1` (hidden → 12 = k 6 × 2 classes per token, read from the
  safetensors header) run as replicated io GEMMs on the last stage. There is no MLM decoder over the vocabulary
  (`vocab_head = False`).
- Precision: the card states that every reported result uses BF16 + FlashAttention-2, so weights and activations
  are charged at bf16. The fp32 safetensors set only the storage of the download, not the runtime footprint.
- Workload: the annotation pipeline reads a record in windows of 16,384 tokens (98,304 bp) that overlap by 1,024
  tokens. One sequence of the batch is one window. Throughput is reported in windows/s/card and in new bp/s/card,
  where a window adds (16,384 − 1,024) × 6 = 92,160 bp.
- 「假设」 No special tokens are counted per window, and the overlap-averaging post-processing (host side) is not
  modelled.

**0.438 vector billing.** `modeling_generanno.py` (`GenerannoRMSNorm`, documented as equivalent to T5LayerNorm)
is fp32 `pow(2).mean`, one `rsqrt`, then times `weight`. The class is `input_layernorm`, `post_attention_layernorm`,
and `model.norm`. The MLP is `down_proj(silu(gate_proj(x)) * up_proj(x))` with published `hidden_act="silu"`.
There is no QK-norm. RoPE is applied to q and k before `repeat_kv`, so the k count is the KV heads. RMSNorm is
`vec = tokens * hidden * 4`, `sfu = tokens`, `a_fmt = fp32`. SiLU is `vec = elements * 4`, `sfu = elements * 2`,
and the gate times up product is one multiply. Both follow the billed activation dtype (bf16 on the card, fp8 on
a what-if). Vector flops stay 0. GEMM shapes and FLOPs stay on the shared full-sequence path. The fp32 cosine
and sine table, the padding mask, and the two residual adds stay on that path. A checkpoint whose `hidden_act`
is not `silu`, or that carries a QK-norm tensor, is refused.

**Carbon-500M / 3B / 8B** are stock `LlamaForCausalLM` checkpoints with the 155,776-entry hybrid vocabulary
(6-mer DNA + Qwen3 BPE), released in BF16. The workload is still that causal Llama forward. Other Llama
catalog rows are not retagged.

**0.445 vector billing.** `LlamaRMSNorm` is the T5 recipe in fp32 (`pow(2).mean`, one `rsqrt`, times `weight`).
Block `attn_norm` / `ffn_norm` already counted `vec = tokens * hidden * 4` and one `rsqrt` per token; those
ops now use `a_fmt = fp32`, and `final_norm` adds the same `rsqrt` (`sfu = token rows`). The MLP is
`down_proj(silu(gate_proj(x)) * up_proj(x))`. SiLU stays the existing `act` op (`vec = elements * 4`,
`sfu = elements * 2`). The gate times up product is `mlp_mul`, one multiply, between `mlp.gate_up` and
`mlp.down`. There is no q/k norm. Both the product and SiLU follow the billed activation dtype (bf16 as
released, fp8 on a what-if). Vector flops stay 0. GEMM names, shapes, and FLOPs stay on the causal path.
A checkpoint whose `hidden_act` is not `silu`, whose `model_type` is not `llama`, or whose MLP is not
`gate_up` then `down`, is refused.

**Attention-mask mode** (`Scenario.attn_mode = "bidir"`, a what-if, prefill only). `model.with_attn_mode` turns
every causal GQA / MLA self-attention core non-causal. Windowed, top-k and compressed cores keep their key-count
rule. For a plain core, the LLM prefill path then uses causal factor 1 instead of ½·(1 + ctx/(ctx + q)). So at
ctx 0 the qk / pv FLOPs and the softmax elements are exactly 2× the causal values, and the projections and FFN are
unchanged. KV write is 0 (an encoder-style forward of a causal model). "reference" returns the model unchanged,
so every causal fingerprint is bit-identical; `test_core_106` checks both fingerprints.


## 53. 0.107 — genomics audit; bidirectional context parallelism

**CP under `attn_mode = bidir`.** The zigzag mapping (§ 0.88) gives every CP rank q = ⌈S/cp⌉ queries behind an
equivalent prefix c = S²/(2·cp·q) − q/2, so that the rank does exactly 1/cp of the causal work S²/2. With the mask
removed, every query attends all S keys, so the equivalent prefix is c = S − q (causal factor 1), and the per-rank
work is q·S, i.e. S²/cp. Before 0.107 the causal prefix was kept for bidir, and CP silently dropped up to half of
the bidirectional attention. The KV ring (one all-gather of the rank's block per attention layer) is unchanged.

**Audit** (`val5/v0107_audit.py`, 173 checks, 0 violations after the fix):

- Monotonicity: windows/s/card is non-decreasing in frequency, MAC efficiency, memory efficiency, link bandwidth
  and SRAM.
- Window accounting:
  - overlap ∈ [0, L);
  - new bp = (L − overlap)·k;
  - bp/s = windows/s × new bp;
  - tokens/s = windows/s × L.
- Useful work (Σ flops_u × ranks ÷ sequences per microbatch):
  - conserved for Carbon-A over TP / PP / SP / DP;
  - conserved for bidir LLM prefill over TP / PP / CP.
- Sweep / Pareto / sensitivity over genomics: sweep rows equal `api_eval`, and the fronts are non-dominated in
  (latency, windows/s/card).

Standing tests: `test_core_107`.

## 54. 0.108 — KV stream efficiency (opt-in); VLM / DiT / D2D / SLC audit

**KV stream efficiency.** Long-OSL decode in r4 is DRAM-bound, and KV traffic is 30–60 % of its time. Decode
attention kernels stream the cache at a higher fraction of peak than the global `mem_eff` (0.7): FlashMLA reports
3000 GB/s on the 3350 GB/s H800. `Scenario.kv_mem_eff` charges the KV read and write bytes of a stage at that
efficiency:

t_dram = (bytes + kv·(s − 1)) / BW, where s = BW(mem_eff) / BW(kv_mem_eff).

The same scale applies to the stage's KV time component. Bytes, capacity and energy are unchanged. On the extval
groups (calibrated base + 0.896), r4 goes 18.1 → 16.5 % and v1 max 18.3 → 13.9 %, but moe dec, moe max, r3 max,
r3 lat and v1 ttft get worse. The field therefore stays opt-in, and the default is unchanged.

Rejected mechanisms for r4:

- Exact capacity batch instead of the power-of-two grid: DeepSeek long-OSL overshoots.
- CTA-tiled prefill GEMMs: dense long-ISL rows reach ≈ 1.0, but MoE / gpt-oss overshoot to 1.2–1.8×.

**Audit** (`val5/v0108_audit.py`):

- VLM encoder: useful work is invariant over TP / PP / DP / SP, the encode time is non-increasing in TP and DP, CP
  with images is rejected, and decode carries weights only.
- Video DiT: work is invariant over SP / TP / PP / DP; CP is rejected with an explanation.
- D2D: latency is non-increasing in grade and units, and package_cards without d2d_enabled is inert.
- SLC: `pin_bw` is never slower than `pin` and is monotone in size. `pin` and `lru` get slower with size only when
  slc_GBps < DRAM; this is now warned for `lru` too (it was silent).

Standing tests: `test_core_108`.

## 55. 0.109 — prefill two-batch overlap (extval); API audit

**moe pf.** The prefill rows (SGLang large-scale EP blog and DeepSeek profile-data, DeepSeek-V3 EP32 over 4 nodes)
are LINK-bound. The candidate's `kernel` overlap adds the EP all-to-all (≈ 2.45 s per batch) to the compute
(≈ 1.31 s), so the rows predicted 0.55–0.68 of measured. Both sources document two-batch overlap in prefill: one
micro-batch's dispatch / combine runs during the other's compute. extval therefore evaluates exactly those
sources' prefill with `exec_overlap = "tbo"` (`TBO_PREFILL_SRC`, not fitted). The pf rows move to 0.85–1.06, and
the group error goes 48.2 → 8.5 %. No other group changes.

Turning on TBO for every row was rejected: decode rows go to 1.85–1.91 and Mixtral max rows to 1.1–1.9, because
those sources do not document it.

The single moe-static row (TRT-LLM 0.21 DeepSeek-R1 on H200, 1.82×) is decode-DRAM-bound. It is left as is, because
no documented mechanism explains the gap without fitting.

**API audit** (`val5/v0109_api_audit*.py`):

- Strict parsing holds under fuzz: only 200 / 400, never 5xx, and the JSON is always finite.
- Sweep rows, Pareto points and the `/api/sensitivity` endpoint equal `/api/eval`.
- Finding: the eval summary reported inapplicable latencies as 0.0 ms. They are now null at the API layer
  (`api._api_summary`), with the same rule as sweep rows. `Result.summary()` and the fingerprints are unchanged.
- The UI CSV export round-trips adversarial cells, including the formula-injection guard.

Standing tests: `test_core_109`.

## 56. 0.110 — consolidated validation summary; energy-accounting audit

**Validation table (0.110, all groups re-run).** The sources are public only: TRT-LLM perf-overview pages and blogs,
the SGLang large-scale EP blog, DeepSeek profile-data, and the vLLM / ROCm docs (`core/data/ext_measurements*.json`).
The hardware is the reference catalogue chips (`core/refhw.py`), and every efficiency and ceiling is a published
figure (§39–§40, §54–§55). Nothing is fitted.

Each cell gives:

- the mean |log(pred / meas)|;
- the median pred / meas, with the [min–max] range;
- the share of rows within ±25 %.

| group | rows | `catalog` (stage overlap, catalog efficiencies) | `catalog_kernel` (kernel-serial) | candidate (`candidate_variant`) | candidate + batching estimator |
|---|---|---|---|---|---|
| moe pf | 5 | 8.4 % · 0.93 [0.85–1.06] · 100 % | 57.8 % · 0.59 [0.49–0.61] · 0 % | 8.5 % · 0.93 [0.85–1.06] · 100 % | 8.5 % · 0.93 [0.85–1.06] · 100 % |
| moe dec | 2 | 14.5 % · 1.28 [0.96–1.28] · 50 % | 33.4 % · 0.78 [0.66–0.78] · 0 % | 5.4 % · 1.09 [1.03–1.09] · 100 % | 5.4 % · 1.09 [1.03–1.09] · 100 % |
| moe static | 1 | 74.1 % · 2.10 [2.10–2.10] · 0 % | 30.0 % · 1.35 [1.35–1.35] · 0 % | 60.1 % · 1.82 [1.82–1.82] · 0 % | 60.1 % · 1.82 [1.82–1.82] · 0 % |
| moe max | 27 | 58.1 % · 1.91 [1.08–2.84] · 11 % | 17.1 % · 1.13 [0.85–1.56] · 70 % | 12.9 % · 1.09 [0.84–1.46] · 85 % | 18.4 % · 1.08 [0.43–1.46] · 78 % |
| r3 max | 15 | 66.6 % · 1.70 [0.92–4.66] · 20 % | 36.2 % · 1.37 [0.79–2.26] · 33 % | 28.5 % · 1.12 [0.87–2.18] · 67 % | 27.9 % · 1.12 [0.79–2.00] · 67 % |
| r3 lat | 16 | 130.1 % · 0.29 [0.17–0.36] · 0 % | 118.4 % · 0.31 [0.24–0.37] · 0 % | 32.8 % · 0.79 [0.46–1.01] · 38 % | 32.8 % · 0.79 [0.46–1.01] · 38 % |
| r4 | 38 | 32.1 % · 1.14 [0.59–2.39] · 39 % | 26.7 % · 0.95 [0.50–1.53] · 45 % | 18.1 % · 0.91 [0.68–2.44] · 74 % | 24.5 % · 0.89 [0.39–1.22] · 61 % |
| v1 static | 36 | 31.7 % · 1.23 [0.86–2.71] · 53 % | 24.0 % · 1.16 [0.77–2.40] · 64 % | 18.2 % · 1.09 [0.72–1.77] · 75 % | 18.2 % · 1.09 [0.72–1.77] · 75 % |
| v1 ttft | 20 | 62.7 % · 0.53 [0.19–1.15] · 35 % | 50.0 % · 0.67 [0.27–1.17] · 35 % | 27.1 % · 0.77 [0.55–1.08] · 35 % | 27.1 % · 0.77 [0.55–1.08] · 35 % |
| v1 max | 44 | 28.2 % · 1.24 [0.70–2.13] · 48 % | 20.4 % · 1.02 [0.63–1.52] · 66 % | 18.3 % · 0.88 [0.63–1.29] · 61 % | 18.1 % · 0.88 [0.63–1.25] · 64 % |

Summary:

- The candidate (with or without the estimator) is the most accurate variant, or within 0.1 point of it, on 9 of
  10 groups. Its remaining large error is moe static: the single TRT-LLM DeepSeek-R1 H200 row, at 1.82×, where
  `catalog_kernel` gives 1.35×.
- That row's mechanism candidates without public numbers remain open: grouped-GEMM efficiency at M ≈ 32 per
  expert, attention-DP imbalance, and host overhead at concurrency 1024 (§40).
- r3 lat and v1 ttft under-predict (median 0.77–0.79) and are the next largest.
- `catalog` is the shipped default of the reference chips. Its large r3 lat / v1 ttft errors come from stage
  overlap hiding serial kernels on GPUs.
- No new mechanism with public support was found in this round. No default changed, and the NPU fingerprints are
  unchanged.

**Energy-accounting audit** (`val5/v0110_audit.py`, genomics / protein / Carbon LLM with the newest defaults —
`pipe` overlap, `auto` schedule):

- The API sweep rows equal `/api/eval`, including the energy counts.
- Throughput is monotone in frequency, memory efficiency and link bandwidth.
- A single card has no link / net / D2D bytes.
- Time-only knobs (overlap, `kv_mem_eff`, `layer_overhead_us`) leave every count unchanged.

Findings:

1. **Fixed.** `Chip.sfu_per_cycle` (§ 0.103) is a time-only knob, but it raised the `vec` action count of
   SFU-bound ops, because the energy work took the op's vector *time* (max of lanes and SFU). Energy and KPI vector
   work is now the lane work; time is unchanged.
2. **By design.** `instance_sched` changes `vec` through weight dequant (`conv_w`): each core group converts its
   own weights.
3. **By design.** Under DAP (`sp`) the folding models' MAC count per unit grows with sp, because ops that are not
   split run on every rank. Useful work (`flops_u` × ranks) is conserved exactly.

Standing tests: `test_core_110`.

## 57. 0.111 — r3 lat / v1 ttft investigation; prefill-first decode residual (opt-in)

**r3 lat** (vLLM 0.11.2 `bench latency`, MI300X, Llama-70B FP8 weights and KV, TP8, one static batch):

- The candidate matches at batch 1 (pred / meas 1.0) but falls to 0.46 at batch 128, for both ISL 128 and ISL 2048.
  The rows are decode-dominated (OSL 2048).
- Implied measured step: ≈ 22 ms (ISL 128) and ≈ 28 ms (ISL 2048) at batch 128, against a modelled 10–12 ms. The
  modelled step is DRAM 4.5 ms (KV 2.2), MAC 2.6 ms and per-layer overhead 5.5 ms.
- The gap grows about linearly with batch, at ≈ 0.1–0.16 ms per sequence per step. The plausible mechanisms are
  per-sequence host work (sampling over a 128 K vocabulary, input preparation) and ROCm paged-attention
  efficiency. Neither has a public per-sequence figure, so neither is modelled.

**v1 ttft** (TRT-LLM 0.8, batch 1):

- Long prompts are 0.74–1.08, but ISL 128 gives 0.59–0.79. The gap grows with model size and on slower parts, which
  suggests small-M GEMM wave quantisation.
- The `cta` GEMM schedule captures that effect but overshoots long prompts:

  | probe | v1 ttft | r3 max | r4 |
  |---|---|---|---|
  | `cta` for all GEMMs | 27.1 → 40.9 % | 28.5 → 42.2 % | 18.1 → 24.1 % |
  | `cta` for dense prefill GEMMs only | 27.1 → 40.9 % | 28.5 → 36.0 % | — |

  Both probes were rejected.

**Prefill-first decode residual** (`pd.ttft_resid`). The closed form charged every colocated prefill-first arrival
the in-flight decode iteration: half a step at the median and a full step at p90 / p99. The DES only waits when an
iteration is running. For small models the step (≈ 7 ms) is comparable to the TTFT, so the DES audit flagged p50 at
+55 % (Carbon-500M) and +63 % (Qwen3-0.6B).

`busy` uses p = (1 − ρ)(1 − π₀): the prefill server is idle and decode is non-empty (PASTA, with π from the decode
birth–death law). R = Bern(p)·U(0, step) is convolved with the TTFT law, rebuilt from its quantiles
(`_wait_mix_sum_q(uniform=True)`).

Against the DES (5 models × 2 shapes × load 0.3 / 0.6 / 0.85 × CV 0 / 1, 6,000 requests each), the mean
|log error| goes from `full` to `busy` as follows:

| metric | full | busy |
|---|---|---|
| mean | 6.1 % | 1.3 % |
| p50 | 8.5 % | 4.7 % |
| p90 | 10.9 % | 2.0 % |
| p99 | 6.3 % | 5.2 % |

Two slices get worse: big-model p50 (3.7 → 4.7 %) and CV 1 p99 (4.3 → 5.5 %). An additive quantile shift was tried
first and was worse; the convolution replaced it. Under the no-slice-worse rule `busy` stays opt-in, and `full`
remains the default (conservative).

**DES vs analytical, new domains.** Over Carbon 500M / 3B / 8B (30 scenarios, 90 mode pairs), the errors match the
qwen3-8b reference: median 1–4 %, p90 of |err| ≤ 25 %. Genomics, protein and video encoders ignore `pd` with a
warning.

Standing tests: `test_core_111`.

## 58. 0.112 — randomized audit; radix single-node pin; bit-identical performance work

**Audit.** Random scenarios (6 seeds × 20 per area) were checked against invariants:
- spec_k: tok/s and TPOT are monotone in acceptance, the step time does not depend on acceptance, accept = 0 is
  never faster than no speculation, and acceptance has no effect when k = 0.
- PP: auto microbatches = min(pp, batch), and the microbatch count never exceeds the batch.
- MoE: the step is monotone in skew, the cap warning fires, a uniform measured load equals uniform routing, and a
  single hot expert stays ≤ the cap.
- Prefix: TTFT is monotone in `prefix_cached`, and the LRU/Che hit is monotone in K and lies in [0, 1].
- dtype: the exec format is native, and the conversion flags agree with the formats.
- Radix trees were compared against `RadixLRU` (30k requests each).

One mechanism bug was found. A single-node level reached by every request (a shared system prompt) is always the most
recent node, so it is resident whenever its cumulative path fits. The Che branch charged it 1 − e^{−T}. It is now
pinned (h = 1, its tokens taken out of C), and Che runs on the remaining levels (`_tree_che`).

LRU is not monotone in C across a fitting-depth boundary. When a deeper level starts to fit, its large leaves evict
siblings: in the example below, H₁ falls from 1.0 to 0.73, and `RadixLRU` shows the same drop. So monotonicity is
tested only within a fitting depth.

- Example tree: 3 roots × 64 tokens, each with one 64-token child, each with one 500-token leaf.
- The drop happens when C passes 628 tokens.

Che vs LRU on an adversarial tree (two 500-token roots, 16-token leaves, exact enumeration over budget) is off by
0.14 on H₁. This is a known approximation; it was not fitted.

**Performance (bit-identical).** The suite's wall time is set by single PD kv_cap tests (061_4 603 s, 057 569 s,
061_3 509 s). Profiling showed the time going to:
- the victim-wait residual sums (≈ 10⁹ generator steps);
- `_gamma_q` in the saturation-eviction quadrature;
- `bulkq._tagged` recomputed on cache eviction;
- `overlap.feed` in the layout search.

Fixes:
- The zero-term skip is exact: adding +0.0 to a non-negative partial sum leaves both the plain and the compensated
  sum unchanged.
- The memos are pure-function caches keyed on full-precision inputs.
- The caches are larger.

Checks:
- PD report: 242 → 152 / 190 s, with an identical SHA-256.
- 152 randomized API outputs are byte-identical.
- NPU fingerprints c034bfe3 / e51c7ec1 are unchanged.

`/api/sweep` (8 representative bodies, including sensitivity) already takes 0.02–0.24 s per sweep, so no change was
needed there.

## 59. 0.113 — catalog / CLI / round-trip audits; downcast label

**dtype.** `gemm_exec` falls back to the widest native format when no common native path exists. If an input is wider
than that format (fp32 on bf16 / fp16-only hardware), it is now labelled `downcast`, not `upcast_both`. The converted
element count, and therefore the cost, is unchanged.

**Hardware catalog consistency** (`val5/v0113/hw_audit.py`), checked against the cited specs:
- every memory combination follows the derived formulas (bus × rate, stacks × height × die / 8);
- mem ids round-trip;
- the JEDEC limits hold: HBM3 ≤ 6.4 Gb/s, HBM4 2048-bit at 8 Gb/s, LPDDR5X JEDEC ≤ 8533;
- the reference GPUs reproduce their datasheet peak, HBM and link figures;
- the D2D entries equal lanes × rate / 8.

There are 0 violations. Power and area are not catalog data; they are user budgets with user densities (core/budget.py).

**CLI ↔ API.** All 154 scenario flags change the scenario hash. The scenario's 22 chip micro-architecture leaves
(freq, array geometry, DMA, NoC and others) were reachable only through presets. `eval --scenario FILE` now takes any
scenario JSON, and the API's echoed scenario re-evaluates to the same hash and summary.

**Performance.** `rank_shares` reused the unsharded op-graph callback twice per op list; it now builds the graph once.
Outputs are byte-identical, and the layout search is about 7 % faster. The remaining PD kv_cap tails are spread over
victim-wait sums, the incomplete gamma and `bulkq._stationary`. None of these has a further bit-identical shortcut
without changing the summation order, so they were left as they are.

## 60. 0.114 — coverage refresh (LFM2.5, Ling-3.1-flash); config / weight drift audit

**Selection.** The HF model lists of the in-scope official orgs were checked for releases since 2026-10-01 with a
public, non-gated config.json. Two LLMs qualify: LiquidAI/LFM2.5-24B-A2B (+ official FP8) and
inclusionAI/Ling-3.1-flash. Qwen-Image-2.1-Turbo (image DiT) is out of the modelled domains.

**LFM2 / LFM2-MoE.** The structure follows transformers `Lfm2Moe`.

- The short conv (`Lfm2ShortConv`) is:
  - in_proj h → 3h, split into B, C and x;
  - B ⊙ x;
  - a depthwise causal conv with kernel `conv_L_cache` = 3;
  - C ⊙ (conv output);
  - out_proj h → h.
- It runs as a linear core (`lin = "conv"`, no recurrent heads):
  - the state is (k − 1) columns per channel in `state_fmt`, as for the GDN / KDA conv states;
  - the vector work is 2k + 2 operations per channel per token.
- Attention is GQA with 32 / 8 heads and head_dim 64, plus per-head q / k RMSNorm.
- The MoE has 64 experts, top-4, a sigmoid router and an expert bias.
- The parameter count is 23,971,587,712, equal to the headers.

In the FP8 release the conv projections stay BF16. The per-role format therefore takes the larger group (BF16) for
attn, while the experts and dense MLP are FP8.

**Ling-3.1-flash** (`bailing_hybrid`). The layers come in groups of 8: layers 1–7 are KDA and layer 8 is MLA.

- KDA: q / k / v / g / o are 64 × 128. f_proj is full rank (`no_kda_lora`), b_proj is h → 64, there are 3 depthwise
  convs with kernel 4, plus A_log, dt_bias and o_norm.
- MLA: q_lora 1536, kv_lora 512, rope 64, v 128, and a head-wise sigmoid gate `g_proj` (h → 64 heads).
- The MoE has 512 experts, top-8, 1 shared expert of 1280 and an expert bias; the first 4 layers are dense (10240).
- There is 1 MTP layer (MLA + MoE + eh_proj and enorm / hnorm / final norm).
- Parameter counts are 563,946,483,520 + 8,174,197,248 (MTP), exact.

**Drift.** Every stored config was compared with the current HF `config.json` (with subfolders for diffusers repos).
There is no value change on any shared key. Stored tensor bytes were compared with the current safetensors sizes, and
there is no change apart from known artifacts (non-safetensors checkpoints, and duplicated consolidated copies in the
Mistral repos). Fingerprints c034bfe3 / e51c7ec1 are unchanged, because the new entries are appended at the end of
`EXTRA_TAIL`.

## 61. 0.115 — image DiT (single latent frame); LFM2 / KDA builder audit

**Image as one latent frame.** An image DiT is the video DiT path with F = 1.

- latent (1, ⌈H/s⌉, ⌈W/s⌉), grid (1, ⌈lh/p⌉, ⌈lw/p⌉);
- full 2-D attention over [text ‖ image] (no temporal axis or temporal attention);
- one forward per step and CFG branch, as for video.

The mode is set by the workload `frames = 1, fps = 0`. Frame overrides are clamped to 1 with a warning, and
`info.image = true` appears only for image models. The unit stays "frame" (1 per request), so the image-per-second
and latency metrics are the frame metrics.

**Builders** (from the transformer/ safetensors headers; parameter counts exact):

- `qwenimage` (Qwen-Image): 60 × dual-stream MMDiT.
  - Each stream has its own QKV / out, GELU-tanh MLP 4d and `*_mod.1` AdaLN (6d).
  - RMS QK-norms; joint attention.
- `qwenimage21` (Qwen-Image-2.1-Turbo): 32 × single-stream blocks.
  - QKV / out and a SwiGLU MLP 3d, shared by text and image tokens.
  - One global `modulation.1` (4d) computed from the timestep; `txt_in` is a 2-layer projection of the Qwen3-VL states.
  - `causal_condition`: text attends only to text, and the reference reuses the text prefix K/V across steps. This is
    not modelled as an optimisation: text tokens are charged every step, which is about 1.5 % extra tokens at
    2048².
- `flux2` (FLUX.2-klein-4B): 5 dual-stream blocks (SwiGLU 3d) + 20 single-stream parallel blocks.
  - The single-stream blocks have a fused [QKV ‖ gate-up] input and a fused [attn ‖ mlp] output (`fused_out`).
  - Three shared modulation projections; no biases.

**Audit** (`val5/v0115/audit.py`):

- **Work conservation.** FLOPs are conserved under every TP / ETP / EP / DP / PP layout tried, including uneven
  ones, for LFM2.5 and Ling-3.1 (decode and prefill) and for the image DiTs (TP / SP / PP).
- **KV accounting.**
  - KV = Σ_attn-layers 2 · n_kv · hd · 2 B · (ctx + 1) for LFM.
  - KV = 9 · (512 + 64) · 2 B · (ctx + 1) for Ling. The MLA latent is replicated on each TP rank by design.
- **State accounting.**
  - State is independent of ctx (1K–128K).
  - LFM: 30 · h · (k − 1) · 4 B.
  - Ling: 63 · (H · dk · dv + 3 · H · d · (k − 1)) · 4 B, which is 1.054 GiB per 4 sequences.
  - Σ over TP ranks equals the global state (ceil-split bound at TP 3).
  - Per-step state traffic = read + write of the state.
- **No defects found.**
- **Measurement artefacts (not model defects).**
  - For gen models the micro-batch unit is the CFG sequence.
  - PP with micro-batches that don't divide the sequences pads the busiest micro-batch.

## 62. 0.116 — text-prefix K/V reuse; image pipelines; image-mode audit

**Prefix K/V reuse** (Qwen-Image-2.1, `causal_condition`). The reference attention mask is
`(q_idx ≥ kv_idx) or same_image_block`, with text and condition tokens modulated from t = 0. With T text tokens,
N image tokens and q = T + N:

- Without reuse, every step is one full forward. Attention work is (N·q + T(T+1)/2)·d per head-dim slice, and
  GEMMs run over all q rows.
- With reuse, step 1 is the forward above plus a write of the prefix K/V: T·heads·(dk + dv) per layer.
- Each later step runs the GEMMs over the N image rows only, with no text-stream or text-projection rows.
- Attention in a later step is N queries over T + N keys, plus a read of the cached prefix K/V.
- denoise = per_step · (t₁ + (steps − 1)·t_c).
- The cache is resident per in-flight sequence: layers(stage) × T × ⌈⌈n_kv / TP⌉ / SP⌉ × (dk + dv) × bytes.

Activation is model-declared (`NativeWorkload.prefix_kv`, the reference `use_kv_cache = True` default).
`workload.prefix_kv = off` disables it, and `on` requires `cond_causal`.

**Image pipelines.** Text encoders and VAEs are built from the released `text_encoder/` and `vae/` headers
(`scripts/build_pipeline_data.py`), using the same op-graph costing as the video pipelines (§11.4).

- Qwen-Image uses the Wan-VAE decoder.
- Qwen-Image-2.1 uses a 2-D decoder with 5 up blocks (×16, 1152 → 144 channels, RGBA out). Its mid attention is per
  frame.
- FLUX.2 uses an SD-VAE-style decoder.
- On a single latent frame the causal decoders run no `time_conv` (feat_cache "Rep" on the first chunk).
- Text-encoder token counts: the template prefix + 256 text tokens (an assumption) for the Qwen models, and the
  padded 512 for FLUX.2.

**Image-mode geometry.** The latent is 2·⌊px / (2 s)⌋ and grid = latent / p, using the image pipelines' rounding.

| Model | VAE ×s | DiT patch p |
|---|---|---|
| Qwen-Image | 8 | 2 |
| Qwen-Image-2.1 | 16 | 1 |
| FLUX.2 | 8 | 2 (pipeline packing) |

**Suite time.** The runner parallelises by module, so wall time is bounded below by the slowest module. The
heaviest searches are now split across modules and started first, which bounds the wall time at roughly the
single-module maximum (about 160 s alone) rather than the length of one worker's queue.

## 63. 0.117 — video VAE / latent audit; reference resolution and frame rounding

**Coverage.** The ten catalog video DiTs already carry text-encoder + VAE pipelines built from their released headers
(§ 0.44 onward), so all gen models are `full`; no new components were needed.

**VAE 3-D audit.** For every conv-decoder family the up-block walk (causal temporal upsampling t → e·t − (e − 1), spatial
×2 per upsampling block; LTX additionally unpatchifies 4×4) reproduces exactly the requested frames × H × W from the
resolved latent; VAE configs (spatial / temporal compression, `dim_mult` / `temperal_downsample`, latent channels)
agree with the catalog `vae_t` / `vae_s` and with each DiT's `in_channels` (Wan / Cog / Hunyuan 16, LTX 128, Mochi 12,
Open-Sora 4, H3 24).  No other gen model has the FLUX.2-type downscale error.

**Rounding (mechanism from the references).**
- Spatial: H, W floor to a multiple of `vae_s · patch` (Wan: `height // (vae_s·p) · (vae_s·p)` with a warning; Hunyuan /
  LTX / Mochi / Cog validate divisibility and otherwise floor at `latent = px // vae_s` and the stride-p patch conv).
  0.44–0.116 used a ceil patch grid for video.  Image models keep the 0.116 rule (2 · vae_s).
- Temporal: latent frames `(F − 1) // t + 1`; Wan snaps `F → F // t · t + 1` first (`vae_frames = "wan"`), one more
  latent frame when F ≡ 0 (mod t).  The reported frame count is the decoder output `(lt − 1)·t + 1`.
- Release defaults are exact multiples, so default results and fingerprints are unchanged.

## 64. 0.118 — Open-Sora / H3 decode geometry; protein and genomics audit

**Open-Sora 1.2 latent frames.** `VideoAutoencoderPipeline.get_latent_size`: each full 17-frame micro-chunk gives 5
latent frames; the remainder r is front-padded by `VAE_Temporal` to a multiple of 4 → ⌈r / 4⌉ latent frames.
0.44–0.117 padded the remainder to a full chunk.  The temporal decoder walk (×2 · ×2 per latent, `double` rule)
then equals Σ chunks · 4·latents frames, matching the per-chunk decode; the SD-VAE spatial decoder runs on the F
output frames.

**MiniMax-H3.** `h3_geometry` reproduces diffusers `AutoencoderKLMiniMaxH3._decode` (`tokens_chunk_size` 5,
`token_overlap` 2, `num_chunks = (T + drop + pad)/5 − 1`, trailing latent repeat) and `_split_tiles` (fewest 256-px
tiles with ≥ 64 px overlap); unchanged.

**Protein path.** Closed forms of every pair core, recycle / diffusion-step / sample linearity, DAP work conservation
(SP 2 / 4 on all five structure models), N³ asymptotics, MSA-depth convexity (column attention ∝ S²) and the TP refusal
hold.  The activation plan assumes fused triangle attention (no H·N³ logits); a warning now quantifies the unchunked
reference tensor per card when it alone exceeds DRAM (`chunk_size` lowers the peak, not the work).

**Genomics.** A window no longer than the overlap is read as one window with no overlap (was: overlap clamped to
L − 1, one new token per window).

## 65. 0.119 — triangle-attention kernel choice (structure models)

`workload.tri_attn` selects how triangle attention (starting / ending node; templates included) runs:

| mode | logits | activation peak | DRAM |
|---|---|---|---|
| `fused` (default) | never materialised (flash-style) | q, k, v tiles | flash streaming as before |
| `materialized` | all rows of the module at once | rows · H · N² (+ q, k, v) per rank | 4 passes over the logits when they exceed SRAM / 2 |
| `chunked` (`tri_chunk` c) | c rows at a time (OpenFold `chunk_layer`) | c · H · N² | as materialized when one chunk exceeds SRAM / 2, else 0 |
| `reference` | the release code at this N | — | — |

Reference chunking: ESMFold `chunk_size` None (materialised); OpenFold inference `globals.chunk_size` 4; AlphaFold 2
`subbatch_size` 4; Boltz-1 `chunk_size_tri_attn` 128 when N > 384 (`const.chunk_size_threshold`) else 512; Protenix
`infer_setting.chunk_size_thresholds` (≤ 1024 none, ≤ 1536 512, ≤ 2048 256, ≤ 2560 128; longer 128 「假设」).
FLOPs are identical in every mode (no recompute at inference); the 4 logit passes are write, softmax read + write,
read by P·V.  Rows are the pair rows on the rank (DAP slices them).  Chunk launch overhead is not modelled.

## 66. 0.120 — kernel launch overhead on structure-model kernels; unfused-attention streaming

`chip.kernel_us` (opt-in since 0.92, default 0) adds a fixed cost per kernel to the stage's sync time.  0.92 counted
GEMMs and fused LLM attentions; 0.120 adds explicit launch counts (`Op.launch`) for the structure-model kernels: each
fused pair attention 1, each activation × activation bmm (triangle multiplication, outer-product mean, pair-weighted
averaging) 1, and an unfused triangle attention 3 per chunk (QKᵀ, softmax, ·V) — so a chunked kernel pays
⌈rows / c⌉ × 3 launches per module.

Streaming fix: when one chunk's logits fit the SRAM budget the logits stay on chip, but q, k, v and the output still
cross DRAM once whenever the module's tensors exceed the budget (the same single pass the fused kernel pays).  0.119
charged 0 there, which made small chunks faster than the fused kernel; now chunked = fused at `kernel_us` 0 and slower
with any launch overhead.

## 67. 0.121 — KV contention feedback on the prefill side

With `fabric.kv_feedback` (0.60) the KV hand-off's share of a NIC, u_kv = λ·kv / (min(n_p, n_d)·β), slows the decode
pool's network collectives.  Each prefill → decode pair carries the same bytes on both ends, so the sending NICs are
equally busy; `fabric.kv_feedback_prefill` (opt-in) applies (1 − u_kv)·net.GBps to the prefill pool as well, re-evaluates
the prefill pool at its chosen batch (TTFT, per-request rate over the prompt mix) and iterates λ → u_kv to the same
fixed point.  A slower prefill lowers λ and hence u_kv, so the decode TPOT is no worse than with decode-only feedback.
Report: `kv.fabric.feedback.{ttft_ms_before, ttft_ms_after, net_GBps_prefill}`.

## 68. 0.122 — multi-turn conversations on the radix prefix tree

Opt-in (`pd.prefix_turns` > 1 or `pd.prefix_writeback`; needs `pd.prefix_tree`).  Each leaf path of the tree starts a
conversation of T turns.  Turn t's prompt is turn t − 1's prompt + its output (`serving.out_len`) + a new user message
(`pd.prefix_turn_user`), so the history extends the tree by branching-1 levels o₁, s₂, o₂, …, o_{T−1}, s_T.  A node is
touched by every request whose prompt contains it: s_t by turns ≥ t, o_t by turns ≥ t + 1 — and by turn t itself with
write-back (SGLang `cache_finished_req` inserts prompt + output).  Turn t looks up before its own insert, so it matches at
most turn t − 1's prompt, plus o_{t−1} with write-back.

Two closed forms:

- **Independent turns** (`prefix_sessions` = 0): IRM over (conversation, turn), t uniform; the leaf path is the
  conversation.  Che per node with rate = path probability × the level's touch share; depth tail mixed over turns,
  each stopping at its query depth.  Against a RadixLRU simulation of the same process: −6 … +0.1 % matched tokens.
- **Sequential sessions** (`prefix_sessions` = A): A conversations open, a request comes from a uniformly random one, so
  the gap between its turns is geometric with mean A requests.  Che in request time τ: a history node is resident
  across a gap iff G ≤ τ and lingers τ after its last touch (resident time (n − 1)·E[min(G, τ)] + τ, conversations
  start at 1/T per request); a base node sees Poisson bursts of T touches (P = 1 − exp(−p/T·(τ + (T − 1)·E[min(G,
  τ)]))).  Turn t ≥ 2 finds its whole chain iff its gap ≤ τ, otherwise only the base tree.  Against the simulation:
  −18 … +3 % (conservative; the largest gaps at many sessions with write-back and mid capacity, where part of a
  chain survives).

The serving prompt is not changed: the report gives the implied mean turn prompt (base + (T − 1)/2·(output + user))
and warns when `serving.prompt` differs by more than 10 %.  The PD DES keeps the tree without history (warned).

## 69. 0.123 — multi-turn: simulator fix, DES history, derived prompt

The −18 % "low-side error" of the concurrent-session model in 0.122 came from the reference simulator.
`RadixLRU.access(path)` looped over the tree depth.  For a path shorter than the tree (any multi-turn prompt
before the last turn), `path[:k]` with k > len(path) is the path itself, so the leaf node was stored at the size of the
deepest level.  The simulated cache held too little history and gave too many hits.  With the loop bounded by the path
depth, the 0.122 closed forms match the simulator: sessions −3.9 … +0.6 %, independent turns −1.0 … +0.1 %.  The
partial-chain survival proposed as the missing mechanism happens in ≤ 0.4 % of returns (a chain is refreshed at one
instant, root last, so it is evicted within a few requests), so it is not modelled.

PD DES: requests carry (extended path, insert depth, query depth) from the same conversation process; the warm-up stream
precedes the measured one and shares the open conversations.  Conversations are sticky to a replica (session affinity),
so the closed form uses A / replicas conversations per replica cache (A with prefix-aware routing, which pools capacity).
DES vs closed-form token hit ratio: |Δ| ≤ 0.01.

`pd.prefix_turn_prompt` (opt-in) sets serving.prompt to the turns' mean prompt (tree + (T − 1)/2·(output + user)) and
the decode context to prompt + output/2.  The DES keeps the prompt-length law of `serving` (it does not vary S per turn),
like the closed form.



## 70. Validator audit and collective channel cap (0.124)

**Audit.** Every validator was checked against an independent brute-force implementation on small random cases:
RadixLRU vs a list-LRU over path prefixes (3000 cases), LRUCache (2000), the overlap pipe makespan vs an O(n²)
flow-shop recurrence (4000) and its repeat shortcuts vs flat replay, DecodeReplica / ColocPrefillFirst vs a direct
step loop (600), KVServer vs the Lindley recursion (300), `_pct` vs exact rationals, vecsanity PUBLIC values vs
FLOPS-derived values and the extval summary recomputed. All matched; the only defects were in scoring: a zero ratio
crashed `extval.summary` (now `n_invalid`) and the stored `data/ext_validation.json` dated from 0.66
(regenerated; ≤ 0.4 pt on any summary error, no conclusion changes; the §56 group table is reproduced exactly).

**Channels (opt-in, 「假设」).** A collective is split over `channels` rings / trees, each driven by one copy engine
that moves at most `channel_GBps`. Per scale-up / network tier the collective sees
b = min(link GBps, channels × channel_GBps) before the topology factor. Both 0 (default) = off, bit-identical.
Point-to-point (PP / KV) transfers are not capped. No public per-channel figure is used; values are user assumptions.


## 71. Hostile core review (0.125)

Invariants checked on every catalog model: each weight GEMM's K·N·count equals its weight parameters; the busiest
rank never does less than the mean (share, wshare ≤ 1) under TP / EP / SP; a batch-1 decode reads exactly the active
parameters minus the embedding table (a lookup); the stored weights equal the release bytes up to documented
differences; the batch search returns the brute-force optimum.

**Grouped linears.** A block-diagonal linear with g groups and total K × N has per group K/g × N/g. Under TP each rank
holds ⌈g/TP⌉ whole groups; with TP > g one group whose N is split over ⌈TP/g⌉ ranks. Before 0.125 the per-group N was
the total N (DeepSeek-V4 `o_a`: 8× work, weight reads and storage).

**Search bound.** For B in [p, q) with step(B) ≥ step(p): thr(B) = B·u/step(B) ≤ (q − 1)·thr(p)/p, where u is the
work per sequence (decode: E tokens; prefill: S prompt tokens; video: units per request with the overlap period). The
bound is taken from the evaluated thr(p)/p, which holds for every phase; 0.124 used E/step and missed the prompt
factor.


## 72. Hostile review round 2 (0.126)

Checked against independent references: attention keys per query (window, top-k, compressed, dual, QSA) vs a
position-by-position sum; decode keys at context n = keys of prefill position n, never more than n tokens for
non-dual layers (0.126 fix: compressed + window layers were charged ⌈n/c⌉ + w at short context); GQA KV storage and
reads under uneven TP; KV / state bytes vs config formulas; hierarchical collective volumes (with equal tiers they
reduce to 2(g−1)/g·B, (g−1)·B and ≤ (g−1)/g·B); PP partitions and per-stage work conservation; DiT cross-attention
GEMM rows (conditioning tokens for K/V, replicated over SP ranks); DES vs closed form on PP / odd-TP layouts.


## 73. Hostile review round 3 — hardware side (0.127)

Checked: GEMM mapping lower bounds (peak MACs, every weight byte through the SRAM port) and monotonicity in the
instance count for every organisation and scheduling mode; `reconf` = per-op minimum; energy MAC counts invariant
under PP / TP / DP where no work is replicated; better hardware never slower; the capacity identity per stage; the
Pareto filter against brute-force dominance. The CTA (GPU reference) tile search is not monotone in K below 128,
because split-K needs slices of at least 64 (a library heuristic, kept).

**Fewest cards.** `/api/fit` first finds the smallest power of two that holds the model, then scans the card counts
between the last failing power of two and it (at most 32), because layouts such as TP3 can hold a model that TP2
cannot. Both counts are returned.

## 74. Colocated worst-gap lattice and review round 4 (0.128)

**Worst token gap.** In prefill-first colocation, a request's longest gap is one decode step plus the longest prefill
busy period it meets. With a single prompt length, busy durations take only the values n · lat, with Borel(λ·lat)
masses (cap 1) or geometric masses (cap > 1). The longest of m busy periods therefore has an atomic law, and its p99 is
an atom. Decomposing the DES confirms that the worst gaps are prefill time, not decode-step time. The closed-form tail
probabilities P(max ≥ n) agree with the DES at every atom. At load 0.4 on the edge layouts, the 0.99 level falls at
an atom boundary, so the DES p99 (an empirical quantile over requests) straddles two atoms while the closed form
returns the lower one. The 8–26 % shortfall is this discreteness, not a missing mechanism, so the closed form is
unchanged. Prompt-length spread fills the lattice and removes the effect.

**Review.** Catalog labels (coverage reasons ⇔ partial, well-formed dtype labels, https sources), the sensitivity
block (scenario ratios, not derivatives; ratio = alt / base), and export schemas (sweep rows, sensitivity block,
JSON export `schema: 1`) are locked by tests/test_core_128.py.

## 75. UI / API robustness and DES seeding (0.129)

**HTTP layer.** A malformed `Content-Length` header and over-deep JSON are answered with 400 (input errors), never a
dropped connection or a 500. A process-pool worker that dies abruptly breaks the executor permanently, so the server
replaces a broken pool and retries the request once.

**Validation.** `chip.mac_eff` is a sustained / peak ratio, so it lies in (0, 1]. A sweep over a knob the model does
not read (workload knobs on an LLM, image knobs without a vision tower, MoE skew on a dense model, `spec_k` without
MTP / draft) is rejected rather than plotted as a flat, "insensitive" line.

**DES seeds.** Seeds are integers (a `None` seed would draw arrivals from the clock); multi-seed comparisons need
distinct seeds, otherwise the reported seed-to-seed noise is 0 by construction. The DES uses only seeded
`random.Random` streams and hash-independent routing, so runs are identical across processes and hash seeds.

**Colocated ITL p99.** No interpolated variant: the atomic law of §74 already matches the DES probabilities, so an
interpolated quantile would only imitate the estimator.

## 76. Frontend review and newest-builder parameter pins (0.130)

**Applicability of swept knobs.** A sweep is only meaningful if the result reads the knob. The server rejects a knob
by model and operating point:
- context / prompt on fixed workloads;
- prompt in a decode-only evaluation;
- MoE skew at EP = 1, since routing imbalance only matters across expert-parallel ranks;
- workload fields a workload kind ignores (frames for image DiTs, MSA for sequence encoders, diffusion samples
  without a diffusion head).

A full catalog × path scan finds no flat 200 sweeps.

**UI.** Rejected evaluations dim the last valid result. Numbers are rounded before choosing their display tier.
Image DiTs use per-image terms (单张延迟, 张/s/卡).

**Parameter counts.** The newest builders (LFM2.5, Ling-3.1 KDA+MLA, Carbon, Qwen-Image, FLUX.2-klein) equal the
official checkpoint totals. LFM2.5 differs from HF `total_parameters` only by its fp32 router `expert_bias` buffers
(64 per MoE layer), which the safetensors files store and the index's parameter count omits.

## 77. Browser check and energy-model invariants (0.131)

**UI.** The UI was exercised in headless Chrome (Playwright) across every domain at two widths: eval, sweep, compare,
fit, layouts, Pareto, and the stale-result state. No console errors and no text overflow were found.

**Energy.** The report multiplies action counts by the user's table. Its internal consistency rests on these
invariants, all now tested:
- static power alone gives exactly the table's W per card;
- useful MAC and vector work per output unit does not depend on how the model is split across cards;
- a system-level cache moves bytes between DRAM and SLC without creating or losing any;
- MoE load skew changes the window (card·s), not the work;
- byte entries are converted at 8 bits per byte.

A table with none of the entries a report uses yields no total rather than zero energy.

## 78. EP-skew routing: distinct experts on the busiest rank (0.132)

**Assumption under review.** With `serving.moe_skew = s > 1`, the busiest EP rank receives s × the mean token-expert
pairs. Since 0.49 its distinct-expert count stays at the uniform expectation n·(1−(1−k/E)^T): the extra load is assumed
to land on experts the rank already reads ("popular"), so weight bytes do not grow, only rows per expert.

**Mechanism.** For a token, expert e is selected with inclusion probability p_e. Over T tokens a rank reads expert e
at least once with probability 1−(1−p_e)^T, so its expected distinct experts are Σ_e 1−(1−p_e)^T over its experts
(occupancy). With a measured load, p_e = k·load_e/Σload. The stage waits for the slowest rank, and the rank reading
the most experts need not be the rank with the most pairs: a hot rank whose load sits on a few experts reads few
distinct experts while a flat, cooler rank reads many. The model therefore takes the maximum occupancy over ranks
(contiguous placement), floored at ceil(pairs/T) and capped at n_local.

**Validation (Monte-Carlo routing, Gumbel top-k without replacement).** Configurations (E, k, EP) = (256,6,8),
(128,8,8), (128,4,4), (384,8,16); popularity shapes Zipf 0 / 0.5 / 1.0, one hot rank (1.5×, 3×), one hot expert;
T = 1…256, 300 trials. Expected distinct experts on the busiest rank, mean relative error: popular rule 22 %,
uniform-in-rank spread 11 % (worse than popular on Zipf ≥ 0.5), exact occupancy 1.6 % (max 13 %). Against the
per-step maximum over ranks (what sets time), exact occupancy is 12 % vs 27 % for the popular rule.

**Adoption.** Exact occupancy applies only when `serving.moe_expert_load` is given; no validation group uses loads and
fingerprints are unchanged, so it is the default for loads. For a scalar skew the shape is unknown; the popular rule
stays the default and `serving.moe_skew_spread = "uniform"` (p = k·s/E on every local expert of the hot rank) is
opt-in because it worsens the Zipf group.

**Open gap.** Even with uniform routing the per-step maximum over ranks of distinct experts exceeds the expectation
(random imbalance, e.g. 13.9 vs 10.3 at E=256, k=6, EP=8, T=16). Modelling it would change the default for every
MoE scenario and the fingerprints; it is not done.

## 79. Slowest-rank random imbalance (0.133, opt-in)

**Mechanism.** All EP ranks wait for the slowest one at the combine all-to-all. Even with uniform expert popularity,
the per-step routing is random, so the most-loaded rank reads more distinct experts and computes more token-expert
pairs than the mean rank. The default model routes with the mean rank (plus `moe_skew` / measured loads).

**Model (`serving.moe_rank_max = "expected"`).** Per rank r:
- distinct experts D_r = Σ_e 1[e hit], with q_e = 1−(1−p_e)^T: mean Σ q_e, variance Σ q_e(1−q_e) (independent
  indicators);
- pairs: per token the count on rank r is hypergeometric (k draws from E, share π_r = Σ p_e / k): mean T·k·π_r,
  variance T·k·π_r(1−π_r)(E−k)/(E−1).

E[max over ranks] uses a normal approximation: Blom's constant μ + σ·Φ⁻¹((n−0.375)/(n+0.25)) for identical ranks,
numeric integration of 1−ΠΦ otherwise (measured loads). The result raises the routed hit count (rounded up to whole
experts) and pairs; it never lowers them.

**Validation (Monte-Carlo, Gumbel top-k).** 7 configurations (E, k, EP) from (64, 2, 8) to (256, 8, 32), shapes Zipf 0
/ 1, one hot rank, one hot expert, T = 1…256, 200 trials. Mean relative error of E[max] distinct experts: 3.6 %
(max 23 %; 9.7 % after rounding up to whole experts) vs 23 % for the mean rank; pairs: 4.5 % (max 24 %) vs 22 %.

**Validation rows (EP > 1).** Mixed. Catalog: decode 15.6 → 14.5 %, R1 static 110 → 107 %, B200/H200 max-load 40.6 →
40.4 %, but DeepSeek-V3 prefill 8.8 → 9.2 %. Catalog_kernel: decode 39.6 → 47.0 %, prefill 78.3 → 79.0 %. Since a
group worsens, and turning it on changes every MoE EP prediction, it stays opt-in; adopting it is a user decision.

## 80. DeepSeek-V3 prefill decomposition and the EP all-to-all (0.134)

**Per-op check against deepseek-ai/profile-data `prefill.json`** (H800, EP32 on 4 nodes, 4K prompts, 16K tokens per GPU,
two micro-batches). The kernel span is 2.12 s; the reported rate implies 2.09 s. Per MoE layer, both micro-batches:

| op | trace | model `wide` | model `cta` attention |
|---|---|---|---|
| MLA attention (FlashAttention 192/128) | 5.6 ms | 13.0 ms | 4.9 ms |
| gate/up GEMMs (routed + shared) | 8.0 ms | 9.0 ms | 9.0 ms |
| down GEMMs | 5.4 ms | 2.8 ms | 2.8 ms |
| o_proj / q_b / q_a+kv_a | 3.4 / 1.7 / 0.5 ms | 2.3 / 0.6 / 1.6 ms | 2.5 / 0.6 / 1.6 ms |
| all-to-all kernels (all layers, own stream) | 1.78 s | link 2.45 s | link 2.45 s |

Model compute is 1.77 s. Under the default `stage` overlap the prediction is link-bound, at 0.85×.

**catalog_kernel's 78 %.** The `kernel` overlap serialises the all-to-all with compute. The sources run two-batch
overlap, which `tbopf_variant` (0.109) and `candidate_variant` already apply; there the error is 8.8–8.9 %. So the
78 % comes from the variant, not from a missing mechanism.

**Mechanism 1: node-deduplicating all-to-all** (opt-in `serving.moe_a2a = node | auto`). DeepEP's normal
(prefill) kernels send a token over the scale-out network once per destination node, then forward it over NVLink
(DeepSeek-V3 report §3.2). With k experts drawn uniformly from E and spread over N nodes, network bytes scale by
N·(1 − C(E − E/N, k)/C(E, k))/k, which is 0.45 for DeepSeek-V3 on 4 nodes (Monte-Carlo checked). `auto` applies it in
prefill only, because decode's low-latency kernels send every pair over RDMA. Link time drops from 2.45 s to about
1.1 s, which matches DeepEP's published internode bandwidth (about 44–47 GB/s).

**Mechanism 2: SMs held by communication** (opt-in `serving.moe_a2a_sm`). The trace README states that prefill
all-to-all occupies SMs and decode's does not. DeepEP's default `num_sms` is 20 of 132, so prefill compute is
scaled by 1/(1 − 0.152).

**Result with both on (catalog).** The balanced rows land on the measurement: perfect-EPLB 1.06×, DeepSeek profile
1.00×. The default-EPLB SGLang rows rise to 1.25–1.42×, because routing imbalance is not modelled for them (no
measured expert loads are published). The prefill group goes from 8.8 % to 20.9 %. Today's 8.8 % rests on the
per-pair link over-count cancelling the missing imbalance and the SM share, so the new mechanisms stay opt-in. No
other group changes (decode and single-node rows are untouched).

**r4 max-load (31–38 %).** The ratio falls with ISL in every model, dense and MoE alike: about 1.3–2.2× at 1K/1K and
0.5–0.7× at 32K ISL. This is the prefill-utilisation pattern of §39 (`wide` scheduling on large prefill GEMMs and
attention), not an MoE mechanism. `candidate_variant` brings r4 to 19.8 %, but it still fails the one-row moe-static
group (§39, §40).

## 81. GEMMs with a small inner dimension: per-tile I/O (0.135, opt-in)

**Trace shapes.** All prefill GEMMs in deepseek-ai/profile-data are DeepGEMM FP8 kernels: 128×128 tiles, a 5-stage
pipeline, and a grid of 108 SMs (132 minus the all-to-all's SMs) or 132 for the dense layers. Per call, one
micro-batch of 8192 tokens.

**Mechanism.** Per output tile, the time is (K / 128)·t_k + t_fix. Inside each tile:
- the main loop runs at the format ceiling (DeepGEMM's best is 78 % of FP8 peak, the existing `fmt_ceiling`);
- t_fix is not hidden by the main loop. It is the prologue (one K-stage of A and B, 128 B of K per row, the TMA swizzle
  width) plus the C-tile writeback, both at the SM's 1/E share of DRAM bandwidth.

Tiles quantize along M and N, and waves quantize over the SMs. All of this is in the existing `cta` model except t_fix:
before, its epilogue went to the SRAM port and there was no prologue. With the catalog DRAM efficiency, t_fix is about
3.4 µs, equal to 12 K-iterations. This is why K = 2048 (down) and K = 1536 (q_b) were too fast: their main loop is
only 16 or 12 iterations long.

**Per-op check (no fitting).** Model / trace ratios over 9 trace shapes:

| model | gmean | mean abs error |
|---|---|---|
| default `wide` | 0.97 | 33 % (shape-dependent, up to 2× either way) |
| `cta` + ceiling | 0.74 | 34 % (small-K shapes 0.57–0.63) |
| `cta` + ceiling + tile I/O | 0.93 | 7 % (every shape 0.89–0.98) |

**Opt-in.** `Chip.tile_io = "dram"` applies with `instance_sched = "cta"`; evaluate fills `tile_io_Bpc` from the system
DRAM bandwidth. Test variant: `extval.tileio_variant`.

**Validation groups.** Adding tile I/O on top of `cta` GEMMs improves nearly every group:
- catalog + cta: all groups better (for example v1 max 48 → 26 %, r4 91 → 42 %);
- candidate + cta: better everywhere except v1 max (19.6 → 31.0 %) and r4 (27.3 → 31.5 %), which overshoot to
  0.78× / 0.86×.

Against the shipped default (`wide` GEMMs), it worsens v1 TTFT, v1 static, moe decode, moe static and r4. So it is
opt-in. The default still schedules GEMMs and attention as `wide`. Per-SM attention scheduling (`Chip.attn_sched =
"cta"`) is also opt-in, used only in `candidate_variant`.

## 82. Per-SM path consistency: attention tile I/O, small-kernel floor, v1 decomposition (0.136)

**Double count fixed.** Under `cta`, attention ops also went through the tile model, so 0.135's tile I/O charged them a
per-tile prologue and a C writeback to DRAM. That is wrong twice over: flash attention keeps S / P on chip, and its
K / V stream is already counted in DRAM time. For Llama-3.1-8B decode at batch 256, attention array time rose 4.5×
(2.6 → 11.6 ms) and the step turned MAC-bound. Tile I/O now applies to GEMM kernels only.

Corrected §81 numbers for candidate + tile I/O (GEMMs `cta`): v1 max 18.9 % (was 31.0 %), r4 23.7 % (was 31.5 %).
Against the shipped `wide` default, every group improves (v1 static 37 → 23 %, TTFT 87 → 40 %, v1 max 33 → 19 %,
moe decode 16 → 5 %, moe static 110 → 84 %, Mixtral 78 → 28 %, r3 lat 267 → 40 %, r3 max 90 → 40 %, r4 38 → 24 %)
except moe prefill (8.8 → 8.9 %). The rule therefore still fails, by 0.1 point.

**Per-layer lump vs per-kernel floor (opt-in `Chip.kernel_vec`, `extval.kfloor_variant`).** `TRACE_LAYER_US`
(62.3 µs per layer) is the mean small-kernel *duration* in the DeepSeek decode trace, which includes those kernels'
work: 1,738 kernels, 7.6 ms, median 3.0 µs. The model already charges that work as vector ops, so at large batch it
is counted twice. The alternative is a 2.0 µs floor (the trace's 10th-percentile duration) on every GEMM, attention
and stand-alone vector kernel: v1 max 19.8 → 16.5 %, r4 19.8 → 19.2 %. But TTFT gets worse (31 → 50 %) and so does r3
latency (39 → 114 %). The lump was also standing in for fixed per-request costs at small batch, which the floor does
not cover. Opt-in.

**v1 decomposition (candidate).**
- TTFT at 128 tokens, batch 1, is short by an absolute gap of about 1.5–1.8 ms for 7B on H100 / H200 (3.4–4.4 ms on
  A100). The gap is the same across the two Hopper chips, which suggests host / scheduling time per request. 70B TP8
  is short by 6.5 ms, about 40 µs per all-reduce over 160 per pass, which looks like small-message all-reduce latency.
  Neither is derivable from public sources without fitting, so neither is modelled.
- Max-load 8B decode is DRAM-bound (KV 7.9 of 11.3 ms per step at batch 256). The remaining under-prediction comes
  from the kernel-overlap serialisation plus the 2 ms lump per step.

## 83. TBO hides per-kernel overhead; per-SM path vs the rule (0.137)

**Cause of the 8.8 → 8.9 %.** The per-SM candidate predicted each DeepSeek-V3 prefill row 0.1–0.2 % slower. The
difference was exactly 61 layers × 62.3 µs = 3.8 ms on a 2.45 s TTFT. The candidate runs prefill as two-batch overlap,
and these rows are link-bound. `layer_overhead_us` (and `kernel_us`) were added to `t_sync` after the max, so the
link-bound stage paid them in full. Mechanistically they are GPU-stream time: small kernels on the compute stream,
which the other micro-batch's all-to-all hides exactly as it hides the GEMMs.

**Fix.** `StageTime.t_kern` is the part of `t_sync` that comes from kernel or layer overhead. For `tbo`:
total = max(core + t_kern, t_slc, t_link) + t_sync − t_kern. The `kernel`, `class`, `serial`, `stage` and `pipe`
modes are unchanged; for the first three nothing overlaps communication.

**Mechanisms checked and ruled out.** All five rows stay link-bound in every variant, so per-expert tile
quantisation of the grouped GEMMs, the all-to-all's SM share, and `cta` vs `wide` cannot move them. They only add
compute that the overlap hides.

**Full group table** (speed ratio, see §124 — gmean / mean abs error):

| group | n | default (`catalog`, `wide`) | candidate | candidate + tile I/O (per-SM) | + kfloor | + kfloor + tile I/O |
|---|---|---|---|---|---|---|
| v1 static | 36 | 1.32 / 37.3 % | 1.07 / 20.0 % | 1.12 / 23.1 % | 1.13 / 22.9 % | 1.18 / 27.0 % |
| v1 TTFT | 20 | 1.81 / 87.2 % | 1.29 / 31.2 % | 1.40 / 40.1 % | 1.48 / 50.3 % | 1.61 / 61.3 % |
| v1 max | 44 | 1.28 / 33.0 % | 0.89 / 19.8 % | 0.91 / 18.9 % | 0.96 / 16.5 % | 0.97 / 15.6 % |
| moe prefill | 5 | 0.94 / 8.8 % | 0.94 / 8.8 % | 0.94 / 8.8 % | 0.94 / 8.8 % | 0.94 / 8.8 % |
| moe decode | 2 | 1.11 / 15.6 % | 1.06 / 5.6 % | 1.05 / 5.2 % | 1.08 / 7.6 % | 1.07 / 7.2 % |
| moe static | 1 | 2.10 / 109.9 % | 1.82 / 82.4 % | 1.84 / 84.2 % | 1.89 / 89.1 % | 1.91 / 91.1 % |
| Mixtral max | 27 | 1.71 / 77.8 % | 1.01 / 20.2 % | 1.10 / 27.8 % | 1.07 / 21.2 % | 1.15 / 30.1 % |
| r3 latency | 16 | 3.67 / 267.4 % | 1.39 / 38.8 % | 1.40 / 39.6 % | 2.14 / 113.7 % | 2.16 / 115.6 % |
| r3 max | 15 | 1.89 / 90.2 % | 1.19 / 33.4 % | 1.27 / 39.7 % | 1.31 / 34.0 % | 1.40 / 42.9 % |
| r4 | 38 | 1.12 / 37.8 % | 0.96 / 19.8 % | 1.11 / 23.7 % | 1.00 / 19.2 % | 1.17 / 25.1 % |

Against the default, the candidate and candidate + tile I/O now worsen no group. MoE prefill is identical: those rows
are insensitive, not improved. Candidate alone beats candidate + tile I/O in 6 of 10 groups, even though tile I/O is
more accurate per op (§81). The fingerprints use the default NPU chip with `stage` / `pipe` overlap, so they are not
affected by this fix, nor would they be by a GPU-reference-hardware default switch. The published ext_validation
numbers would change. The default is not switched in this version (user decision).

## 84. Why tile I/O loses to candidate in 6 of 10 groups (0.138)

**Decomposition.** candidate + tile I/O differs from candidate in two ways: GEMMs move from `wide` to `cta`, and tile
I/O is added. With GEMMs on `cta` but no tile I/O, every group is worse than candidate (v1 TTFT 31 → 51 %, r4 20 →
27 %, Mixtral 20 → 37 %). Adding tile I/O recovers part of that in every group. So the loss comes from the `cta`
GEMMs, not from tile I/O.

Split by chip and format (candidate → candidate + tile I/O):

| rows | change |
|---|---|
| A100 fp16 | 1.10 → 1.22× |
| B200 mxfp4 | 0.84 → 1.20× |
| B200 nvfp4 | 0.97 → 1.06× |
| Hopper fp8 | about unchanged (it has the DeepGEMM ceiling) |

**Mechanism 1: main-loop ceilings existed only for Hopper FP8.** Every other chip / format ran the `cta` main loop at
100 % of peak. The opt-in `extval.GEMM_CEILING` (`ceil_variant`) adds published library-best dense-GEMM fractions
for large compute-bound shapes (sources in the code), not fitted to the validation rows:

| chip / format | ceiling |
|---|---|
| A100 fp16 / bf16 | 0.844 |
| Hopper bf16 / fp16 | 0.752 |
| B200 bf16 | 0.689 |
| B200 nvfp4 / mxfp4 | 0.691 |

No public B200 fp8 or MI300X figure was found. gpt-oss's B200 experts execute at fp8 (MXFP4 × MXFP8), so they stay
at peak and keep over-predicting.

**Mechanism 2: fixed costs that `wide` masked.** At batch 1, TTFT rows are short by a fixed per-request time (§82).
`wide` under-utilisation was hiding that, and faster, more accurate GEMMs expose it again: TTFT 1.29× → 1.35× even
with ceilings.

**Persistent kernels (opt-in `tile_io = "persistent"`).** In a persistent ping-pong kernel, tile i's epilogue and tile
i+1's prologue run under the neighbouring tile's main loop. This makes every group worse than `dram`, and the
DeepGEMM trace matches `dram` (§81). Kept as an option only.

**Group table** (gmean / mean abs error):

| group | n | default | candidate | cand + tile I/O | cand + tile I/O + ceilings | cand + ceilings |
|---|---|---|---|---|---|---|
| v1 static | 36 | 1.32 / 37.3 % | 1.07 / 20.0 % | 1.12 / 23.1 % | 1.10 / 21.9 % | 1.06 / 20.1 % |
| v1 TTFT | 20 | 1.81 / 87.2 % | 1.29 / 31.2 % | 1.40 / 40.1 % | 1.35 / 34.7 % | 1.29 / 31.6 % |
| v1 max | 44 | 1.28 / 33.0 % | 0.89 / 19.8 % | 0.91 / 18.9 % | 0.91 / 18.9 % | 0.89 / 19.8 % |
| moe prefill | 5 | 0.94 / 8.8 % | 0.94 / 8.8 % | 0.94 / 8.8 % | 0.94 / 8.8 % | 0.94 / 8.8 % |
| moe decode | 2 | 1.11 / 15.6 % | 1.06 / 5.6 % | 1.05 / 5.2 % | 1.05 / 5.2 % | 1.06 / 5.6 % |
| moe static | 1 | 2.10 / 109.9 % | 1.82 / 82.4 % | 1.84 / 84.2 % | 1.84 / 84.2 % | 1.82 / 82.4 % |
| Mixtral max | 27 | 1.71 / 77.8 % | 1.01 / 20.2 % | 1.10 / 27.8 % | 1.08 / 26.4 % | 1.01 / 20.0 % |
| r3 latency | 16 | 3.67 / 267.4 % | 1.39 / 38.8 % | 1.40 / 39.6 % | 1.40 / 39.6 % | 1.39 / 38.8 % |
| r3 max | 15 | 1.89 / 90.2 % | 1.19 / 33.4 % | 1.27 / 39.7 % | 1.21 / 36.7 % | 1.18 / 32.9 % |
| r4 | 38 | 1.12 / 37.8 % | 0.96 / 19.8 % | 1.11 / 23.7 % | 1.09 / 22.5 % | 0.95 / 19.9 % |

Against candidate, candidate + tile I/O + ceilings is better in v1 max and moe decode, tied in moe prefill and r3
latency, and still worse in v1 static, v1 TTFT, Mixtral, r3 max, r4 and moe static. Against the shipped default, it
is better everywhere except moe prefill (a tie).

**MoE static row (trt-dsr1-h200-maxtput), source re-read** (TRT-LLM v0.21.0 blog). Value and configuration are
confirmed: Total Output Throughput 11,489 tok/s; per-GPU 1,436 tok/s at 128 sequences per rank, which implies about
89 ms per decode step. The model gives 37 ms (default) and 43 ms (candidate). The H200 configuration differs from the
blog's B200 configurations in ways the model does not represent:
- the extra config enables only CUDA graphs (batch 128 only) and attention DP, with no overlap scheduler, so the
  PyTorch backend's per-iteration host scheduling is exposed;
- the KV cache is BF16 (no `kv_cache_dtype`);
- chunked context is not supported (WIP), so with `max_num_tokens` 1151 a 1024-token context fits in an iteration
  only next to at most 127 decode tokens, and those iterations run outside the batch-128 CUDA graph.

None of these can be quantified from public sources, so the row stays a known 1.8–2.1× outlier. It is listed in §73
for a decision on whether a single host-bound row should block the rule.


## 85. NPU hostile review: randomized invariants (0.139)

Scope: the NPU-side mechanisms that GPU validation cannot reach, namely native-format and dequant paths, SRAM residency
(KV vs weights), D2D / UCIe tiers, LPDDR / HBM geometry, OS / WS datapath variables, and multi-core small-GEMM
scheduling. Method: a seeded fuzzer (`val5/v0139/fuzz.py`, about 5 000 random scenarios × 14 knobs). Each scenario
draws a model (dense, MoE, MoE with shared expert, fp8, AWQ), phase, batch, context, mapping, array geometry, engine
count, SRAM size, schedule, format matrix (bf16-only with or without a dequant engine), memory, TP / EP, D2D package,
NoC / wave overhead, explicit port and SLC. Each knob adds one resource. The fuzzer asserts that latency never rises,
that DRAM need is conserved, that SRAM residency never drops, and that fit never flips to non-fit.

Found and fixed:
1. **The split picker ignored replicated conversion.** Under `auto` / `split`, a row split makes each group dequantize
   the full K×N operand (and a column split the full M×K), which `GemmCost.convert_elems` already charged after the
   choice. The choice itself compared array cycles only, so a faster port favoured splits and the vector path then
   paid E/g× the dequant. Example: Qwen3-8B-AWQ decode, reconf, 56×16×8, tpot 31.4 → 221 ms at 4× port. Plans are
   now scored as max(array, 2·conv / rate) (array and vector overlap in every overlap mode), so a split wins only
   when it is faster with its conversion.
2. **The pipe hide credit used compute, not wall time.** `items_of` credited dispatch / combine against the shared
   expert's compute seconds. In the flow shop a DMA-bound shared expert occupies its weight-streaming time, which is
   longer, so less compute (a higher clock) meant more exposed all-to-all. The credit is now taken inside the
   simulator from the replayed wall time of the shared-expert steps (open / close / hide / reset markers). Credit
   does not carry past the op list, as before. When the window is compute-bound the result is bit-identical.

Checked and clean: memory geometry (266 catalog LPDDR / HBM specs: bandwidth and capacity scale exactly with package
or stack count), D2D units 1→8 are monotone, dequant engine rate, native fp8 / int8 added to a bf16-only chip, SLC
size and bandwidth, link bandwidth, the WS accumulator, and the residency / KV-vs-weight plan under 4× SRAM.

Not bugs, by definition:
- More engines can be slower under a *forced* schedule. Under `wide`, the cores are one R × (C·E) array whose fill
  and accumulator rows scale with the width. `acc_kib` and an explicit `sram_port_Bpc` are chip totals, so doubling
  E halves each core's share. Under `cta`, a tile's port share is 1/E (the GPU per-SM model), and tiles below E leave
  that share idle. The `auto` default never showed the effect in the samples.


## 86. System-level invariants; PP split refinement (0.140)

`val5/v0140/sysfuzz.py` extends the 0.139 fuzzer to the system level: multi-card layouts (TP / PP / EP / DP), D2D
packages, node / net tiers, the fabric, overlap modes, every domain, PD pools (KV bandwidth, layer-wise KV, prefix
cache / hit, pool sizes) and the batch search.

**Found: the PP split could miss a neighbouring better split.** The cost split (§ 0.62) lets the proxy (stage rule,
full DRAM streaming, no residency or overlap) propose splits, and the full model picks among those plus equal counts.
The proxy optimum can sit one layer away from the full-model optimum. Worse, when the proxy saw no gain the full model
never ran, so a resource increase could switch from a better proxy split to equal counts: Qwen3-8B-FP8 PP2 prefill
went 2.7 % slower with six LPDDR6 packages instead of four. Fix: a hill-climb under the full model that moves one
boundary ±1 layer while strictly faster (fits first), for up to 8 rounds. This is a local optimum, not an exhaustive
search, so monotonicity is restored in every sampled case but not proven. Catalog impact: 48 of 340 PP points are
faster (up to 11.8 %) and none is slower.

**Not bugs:**
- With `fabric.algo = auto`, the per-collective bw + α pick ignores that the stage hides bw under compute. `auto_overlap`
  fixes this, which is why it exists.
- PD prefill batch = the largest batch inside the TTFT SLO, so TTFT may rise while goodput rises.
- DRAM need at PP > 1 follows the time-balanced partition, which moves with resources.

**Clean:** link / D2D / net bandwidth and α, D2D units, host bandwidth, fabric oversubscription, `best_batch`
versus random feasible batches, and `best_batch` monotone in DRAM / clock / link / SRAM.

## 87. Default changes on overall validation merit (0.141)

Fingerprint stability is no longer a constraint. Defaults change where the mechanism is justified and validation
improves overall. Every pin was re-baselined, with old → new recorded in the CHANGELOG.

**GPU / validation default.** `extval.DEFAULT = "default"` is the per-SM candidate path with ceil
(`ceil_variant(candidate_variant())`). Group errors, catalog → default (gmean / mean |err|):

| group | n | catalog (≤ 0.140) | default (0.141) |
|---|---|---|---|
| v1 static | 36 | 1.32 / 37.3 % | 1.06 / 20.1 % |
| v1 TTFT | 20 | 1.81 / 87.2 % | 1.29 / 31.6 % |
| v1 max | 44 | 1.28 / 33.0 % | 0.89 / 19.8 % |
| MoE prefill | 5 | 0.94 / 8.8 % | 0.94 / 9.2 % |
| MoE decode | 2 | 1.11 / 15.6 % | 0.98 / 1.7 % |
| MoE static | 1 | 2.10 / 109.9 % | 1.82 / 82.0 % |
| MoE max (Mixtral) | 27 | 1.71 / 77.8 % | 1.01 / 20.0 % |
| r3 lat | 16 | 3.67 / 267.4 % | 1.39 / 38.8 % |
| r3 max | 15 | 1.89 / 90.2 % | 1.18 / 32.9 % |
| r4 | 38 | 1.12 / 37.8 % | 0.95 / 19.6 % |
| all rows | 204 | 1.50 / 60.2 % | 1.04 / 23.1 % |
| mean of group errors | | 76.5 % | 27.6 % |

**MoE.** `moe_rank_max = "expected"` is now the default (§ 0.133): the stage waits for the slowest EP rank. `moe_a2a`
node + sm stays opt-in because MoE prefill rises from 8.8 % to 52.7 % (mean of group errors 32.4 % vs 27.6 %).

**Per-core accumulator and port.** Each core owns 64 KiB of accumulator and a 4·(R+C)·2 B/cycle port, so both scale
with the number of engines. Before 0.141 the accumulator was a chip total and the port was 4·(R+C·E)·2, which starved
split schedules as E grew. Under `instance_sched = auto`, more engines are never slower. Forced wide / ws schedules
still fill a ganged R + C·E array, so they can be up to +11 % slower with more engines. That is definitional.
`upgrade_legacy` maps saved ≤ 0.140 chips to per-engine fields that reproduce the old hardware.

**Fabric / PD.** `fabric.algo = auto_overlap` (stage re-pick) is the default. Residual: under `exec_overlap = pipe`, the
re-pick minimises stage time rather than flow-shop time, so a 2× D2D-unit point measured +0.009 % (fuzz seed 41). PD
uses the smallest power-of-two prefill batch that keeps the SLO and the request rate. The default TTFT drops from 1298 ms
to 658 ms at the same goodput.

## 88. Pipe replay with the auto_overlap picks; EPLB imbalance study (0.142)

**Pipe + auto_overlap.** Before 0.142, the stage re-pick (§ 0.60) set the stage's link / sync terms while the flow-shop
replay (§ 0.98) ran each collective at its per-collective default pick. The pipe total therefore mixed one algorithm's
α with another's bandwidth, and could undercut both consistent choices. The sysfuzz 0.141 seed-41 point is an example:
Qwen3-Next TP8/EP8 on 2-card packages, 4 D2D units = hier α + ring bandwidth, so 8 units looked 0.009 % slower. Now
`fabric` takes a per-collective pick override during the replay, and `_pipe_repick` chooses the combination on the pipe
total: the stage-optimal and default picks, then ≤ 2 rounds of coordinate descent. `auto_overlap` ≤ `auto` holds in
pipe mode by construction. The chunked fused replay uses the decode stage's picks; merged-chunk payloads that do not
match fall back to the default pick.

**EPLB.** A published linear-placement statistic exists: TensorRT-LLM wide-EP EPLB README, DeepSeek-R1 EP32,
layer-averaged (max − mean)/mean = 1.564 without EPLB and 0.115 with it. Applying it as the hottest rank's load
(skew 2.564) to the SGLang `eplb = default` prefill rows under `moe_a2a` node + sm moves them from 1.62–1.69× to
0.80–0.92×. The perfectly balanced rows (`deepseek-profile`, SGLang simulated-perfect) stay 1.30–1.38× over, so the
node + sm all-to-all mechanism over-predicts the balanced case. Imbalance cannot explain that gap. MoE prefill error
is 23.7 % vs 8.9 %, so `moe_a2a` stays opt-in. Adopted instead: rows that simulate absolute balance drop the random
slowest-rank excess (`moe_rank_max = off`), which moves MoE prefill from 9.2 % to 8.9 %. Next lead: the balanced-case
over-prediction of the node-dedup all-to-all (achieved RDMA / NVLink-forward bandwidth and the SM share of DeepEP
normal kernels).

## 89. DeepEP node-mode bandwidths; per-shape GEMM ceilings; why moe_a2a stays opt-in (0.143)

**Forwarding.** DeepEP normal kernels send a token once per destination node over RDMA, to the GPU with the same local
rank, and then forward it over NVLink to each expert GPU in that node. The scale-up volume per GPU is therefore
(k−1)/G · P for node-local pairs plus (G−k)/G · (k−1)/k · P for forwarded remote pairs. For EP32 on 8-GPU nodes that is
0.875 P, where ≤ 0.142 counted 0.219 P.

**Achieved rates** (DeepEP README, H800, 2025-06): NVLink dispatch / combine run at 153 / 158 GB/s against the model's
200 GB/s nominal, giving 0.78. RDMA runs at 43 / 58 / 51 GB/s at EP16 / 32 / 64. That is at or above the 50 GB/s CX7
line rate under DeepEP's byte count, so no derating is applied 「假设」. The SM share stays at DeepEP's default of 20 of
132 SMs.

**Where the balanced gap is.** On the deepseek-profile row (EP32, 16K tokens per GPU, TBO) the per-step terms are
attention + shared 746 ms, experts 611 ms, dispatch 369 ms and combine 739 ms. Comm stays below compute even at the
achieved rates, so the prediction is compute-bound and the DeepEP corrections leave it unchanged. The measured step is
2090 ms. With the per-shape DeepGEMM ceilings (2025-03 table: K-small MLA projections 590–990 TFLOPS, grouped down
projection 1099) the balanced rows over-predict by 1.23 / 1.16× instead of 1.38 / 1.30×. The rest is attention and SM
share, which is the next lead.

**Decision.** Mean of group errors: default 27.5 %; with DeepGEMM shapes 27.8 %; with DeepGEMM and node + sm all-to-all
30.4 %. MoE prefill error is 35.2 %, or 19.9 % with the published 2.56× linear-placement imbalance, against 8.9 % for
the default. The default (per-pair all-to-all) stays. Its LINK-bound prefill rows match only because the per-pair
cross-node bytes over-count, which offsets compute that is too fast. Both are known and documented (§ 80, § 88).

## 90. Catalog refresh: Step-5, MiMo-V2.6, Intern-S2, MiniCPM-V-4.7, LingBot-Video MoE (0.144)

**Scope.** These are the notable official open-weight releases from Aug–Oct 2026 that were not yet in the catalog. Each
entry comes from the public `config.json` plus the safetensors headers, keeps its as-released dtypes, and has an exact
parameter count (test_core_144).

| id | release | builder | params (LLM / DiT) | dtypes | coverage |
|---|---|---|---|---|---|
| step-5-preview | https://huggingface.co/stepfun-ai/Step-5-Preview | `step5` (new) | 598,476,053,952 + MTP 2,515,858,560 | experts FP8 (layers 88–90 bf16), rest bf16, A fp8 | partial (vision: overview only) |
| mimo-v2.6-flash | https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL | `mimo_v2` (new) | 308,778,780,864 + MTP 987,820,224 | experts MXFP4, attn / dense FP8 (o_proj bf16), A fp8 | partial (ViT windows as full) |
| intern-s2-397b | https://huggingface.co/internlm/Intern-S2-397B | hybrid (Qwen3.5-MoE) | 396,371,516,160 + MTP 6,595,568,128 | bf16 | full |
| minicpm-v-4.7-35b-a3b | https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B | hybrid (Qwen3.5-MoE text) | 34,659,889,792 | bf16 | partial (vision: one 448² view) |
| lingbot-video-moe-dmd | https://huggingface.co/robbyant/lingbot-video-moe-dmd-30b-a3b | `lingbot` (new, MoE DiT) | 30,084,512,320 | bf16, router / modulation fp32 | full (pipeline: Qwen3-VL-4B text encoder + Wan VAE) |

**Step-5** (official `inference/model.py`, `kernel.py`):
- Attention: 92 layers in the pattern s s s f. All layers are GQA with 64 heads, 4 KV groups and head 192, q / k RMSNorm, and
  a head-wise sigmoid gate (`g_proj` h → 64). The SWA layers use window 512.
- DSA full layers: the indexer projects q (16 heads × 256), k, a gate z and fp32 head weights w. Each completed 8-token
  region is compressed once into a cached summary (`csa_compress_regions`, fp32 + fp8), and the scores are a weighted
  ReLU over the fp8 summaries. Each KV provider group keeps its top-512 regions plus the current one, and attention reads
  those tokens from the full K/V.
- Model: `qsa = 8`, `topk = 4096`, so keys = all ⌈p⌉ until 512 regions, then 4096 + p mod 8. New `AttnCore.idx_sum`:
  only the regions completed in a pass are compressed (no per-forward re-pooling as in QSA), the cache holds idx_dim·5 / 8
  bytes per token, and decode reads idx_dim bytes per region.
- Layers 3–90 are MoE: 352 experts, top-8, one ungated shared expert of 1536. The other layers are dense SwiGLU 13824.
- MTP ×3: SWA + dense layers, each carrying its own `shared_head` vocab projection, counted in the MTP extra.

**MiMo-V2.6** (`modeling_mimo_v2.py`):
- Layers: `hybrid_layer_pattern` has 39 SWA-128 layers (KV 8, learned sink per head) and 9 full layers (KV 4). qk 192 /
  v 128, fused qkv.
- FFN: `moe_layer_freq` gives 1 dense layer and 47 MoE layers (256 experts, top-8, sigmoid noaux_tc, no shared expert).
- Experts are stored MXFP4 (U8 packed, e8m0 per 32). quant_method fp8 with dynamic activations gives A = fp8 under the
  as-released rule.
- The audio `speech_embeddings` (26 M) now count as vision / multimodal, not LLM.

**MiniCPM-V-4.7.** The 30.7 M `model.merger` projector was classified as LLM MLP. It now counts as vision, and the count is
exact. The text config keeps Qwen3.5's `mtp_num_hidden_layers: 1`, but the release ships no MTP tensors, so the MTP is
dropped. It is the only catalog model in that state.

**LingBot-Video MoE** (https://github.com/Robbyant/lingbot-video `transformer_lingbot_video.py`, `pipeline_lingbot_video.py`):
- Blocks: 48 single-stream blocks over [video ‖ text], with no cross-attention. `text_embedder` maps the Qwen3-VL states
  (2560) to d = 2048. MHA has 16 heads × 128 with QK-norm and 3-axis RoPE.
- Modulation: AdaLN = per-token `time_modulation` (6d, evaluated for every joint token in the reference, charged so) plus
  a per-block `scale_shift_table`. Sandwich RMSNorms sit on the attention and FFN outputs (`Layer.post_norm`).
- FFN: 128 experts top-8 (sigmoid, 4 groups, top-2 groups) + 1 shared SwiGLU 768.
- `ir._full_layer_ops` now emits router and shared GEMMs on all local tokens and grouped expert GEMMs with the
  uniform-routing hit count and m_e = tokens·k / E. Ulysses SP splits tokens, TP splits d_expert, and there is no EP, as in
  the reference.
- Workload (README, DMD student): 832×480, 121 frames, 24 fps, 8 steps, guidance 1. The text uses a structured JSON
  caption of ≈ 1k tokens 「假设」.

**Vision encoders** (`core/vision.py`). Every encoder's parameter count equals the release header.
- Intern-S2 uses the Qwen ViT with the Qwen2-VL resize: 456,010,480.
- Step-5 uses the Perception Encoder: conv patch 14 without bias, 52² absolute positions, ln_pre, 47 LayerScale blocks
  with GELU 8960, no ln_post. Two stride-2 3×3 convolutions follow (1536 → 3072 → 6144), then the projector to 4096,
  giving 1,981,014,784 parameters and 169 tokens per 728² view. The crops come from an external processor, so only the
  overview is counted; this is a coverage reason.
- MiniCPM-V-4.7 uses SigLIP (27 × 1152, 70² positions). `vit_merger` sits after block 6 (`insert_layer_id`): attention at
  full resolution, a 2×2 concat, then an MLP 4608 → 17216 → 1152, so blocks 6–26 run on N/4 rows. The final merger
  (2×2, 4608 → 4608 → text) gives 552,986,032 parameters and 64 tokens per 448² view. Slicing (up to 9 slices) is not
  modelled 「假设」.
- MiMo-V2.6 uses a ViT with 28 RMSNorm blocks: GQA qkv (32 + 2·8) × 64 with bias, SiLU-gated MLP 4608, and sinks on the
  24 windowed layers. The merger (5120 → 5120 → 4096) brings it to 728,594,432. The windowed layers are charged as full
  attention (conservative). The audio encoder and speech embeddings (261,127,168) are stored but not run.

**LingBot pipeline.** The text encoder is the Qwen3-VL-4B from the release's own `text_encoder/` (4.44 B stored). The VAE is
the release `vae/`, whose header is identical to the Wan2.1 VAE, so the existing `wan-vae` component (fp32) is reused.
The DiT entry is therefore `full`, like the other video models.

**Considered, not added.**
- LTX-2.5: gated repo.
- ESMC-6B: original release May 2026, outside the window.
- Ling-3.0-flash-VL: same text architecture as Ling-3.1-flash, already covered.
- MiMo-V2.6-Pro (1.02 T): same builder as Flash; it can be added with a fetch.
- No new genomics release beyond Carbon.

**Fingerprints** are unchanged (`0a954915c5fb003f` / `53ee9e08165795c6`). The new entries are appended, and no existing
model's spec changed.

## 91. MiMo-V2.6-Pro, Ling-3.0-flash-VL; MiniCPM slicing and MiMo ViT windows; hostile review of the 0.144 builders (0.145)

**New entries** (public `config.json` + safetensors headers, as-released dtypes, exact; test_core_145):

| id | release | builder | params | vision | coverage |
|---|---|---|---|---|---|
| mimo-v2.6-pro | https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL | `mimo_v2` | 1,021,248,126,336 + MTP 1,934,715,264 | 1,033,761,792 (+ audio 294,681,600 stored) | full |
| ling-3.0-flash-vl | https://huggingface.co/inclusionAI/Ling-3.0-flash-VL | hybrid (KDA + MLA) | 124,414,211,552 | 434,248,944 | full |

- MiMo-V2.6-Pro: 70 layers (60 SWA-128 with a sink, 10 full), 384 experts top-8, 1 dense layer, MXFP4 experts. Its config
  omits `num_nextn_predict_layers` although the release ships `model.mtp.layers.{0,1,2}`. The builder therefore offers up
  to 8 MTP templates and `from_release` trims them to the released MTP parameters, which gives 3, exact.
- Ling-3.0-flash-VL (`bailing_moe_v3_vl`, the text config nested) is a different architecture from Ling-3.1-flash, so it
  is a separate entry:

  | | Ling-3.0-flash-VL | Ling-3.1-flash |
  |---|---|---|
  | hidden / layers | 2560 / 42 | 4096 / 72 |
  | layer groups | 6 (7 MLA + 35 KDA) | 8 |
  | MLA q-LoRA | none (q 32 × 192) | 1536 |
  | experts | 512 × 768, 1 shared | 512 × 1280 |
  | dense layers | 2 | 4 |
  | MTP | none | yes |

  - The head-wise MLA gate and full-rank KDA f/g are shared with 3.1. The default `num_shared_experts` = 1 comes from
    `BailingMoeV3VLConfig`.
  - Vision: a Qwen3-VL ViT (27 × 1152, no deepstack, merger = norm only) plus the BailingMM `linear_proj`
    (4608 → 2560 → 2560, GELU). `linear_proj.*` is now classified as vision in summarize_release; re-summarising every
    other release leaves its parameters unchanged.
  - Image resize: Qwen rule, factor 32, min/max pixels taken from `preprocessor_config.json`.

**MiniCPM-V slicing** (transformers `MiniCPMV4_6ImageProcessor`, which `minicpmv4_7` maps to in `image_processing_auto`):
- `get_sliced_grid`: multiple = min(⌈W·H / 448²⌉, 9), and slicing applies only when multiple > 1. The grid comes from the
  candidates {m−1, m, m+1} with the least |log aspect error|, ties going to more rows.
- `find_best_resize` sizes the overview, with upscaling only when the image is not sliced. `get_refine_size` sizes the
  slices.
- Both sizes snap to multiples of 56: patch 14 × the two 2×2 merges.
- Tokens per view = patches / 16. The LLM placeholder also has:
  - 2 tokens for the image start / end;
  - 3 tokens for `<image_id>0</image_id>`; the index is assumed to be one token for fewer than 10 images (「假设」);
  - 2 tokens for each slice's start / end;
  - rows − 1 newlines.
- `ImageGrid.views` holds (ph, pw, count) per view group, and `vision_ops` runs one tower pass per group.
- Checked against the transcribed processor functions: 0 mismatches over 20k random sizes.
- Example: 1024×768 → a 2×2 grid. Overview and slices are each 28×36 patches (63 tokens), 329 LLM tokens in total. 0.144
  used one 448² view of 64 tokens.

**MiMo ViT windows** (`MiMoVisionAttention._build_window_mask`):
- Blocks outside `fullatt_block_indexes` (0 / 9 / 18 / 27) mask |i − j| > `visual_token_window_size` (64) over each
  image's patch sequence. `vit_window_attn_types` 1 does this in merge-unit column order and 0 in row order; the key
  count is the same.
- Keys per image = `band_keys(N, 64)` = N + 2(wN − w(w+1)/2), charged as the score / PV fraction.
- The sink is a per-head bias on key column 0 of the windowed blocks: no extra key, 1 extra op per score.
- Row ↔ column re-index gathers: 6 per pass.
- Coverage moves from partial to full for mimo-v2.6-flash, mimo-v2.6-pro and minicpm-v-4.7. Step-5 stays partial: its
  crop count is decided by an external processor.

**Hostile review** (val5/v0145/hostile.py, test_core_145) used randomized invariants:
- Step-5 `keys_sum` = Σ min(p, ctx_eff(p)), checked by brute force.
- Indexer summary writes are chunk-additive and equal to the stored growth.
- MiMo: KV writes ≥ stored bytes, equal when S ≤ window. Decode keys are capped at the window on SWA layers only, for any
  TP.
- Video MoE: expert (token, expert) pairs are conserved (m_e · hit · share = t · k) under TP × SP, with hit ≤ E.
- **Break found:** the Step-5 indexer wrote idx_dim bytes per new token, but the cache stores one fp32 + fp8 summary
  (5 · idx_dim) per completed 8-token region, so writes were 8/5× the stored bytes.
  - Writes are now 5 · idx_dim per region completed in the pass.
  - `stage_storage` / `cache_new_bytes` count completed regions only; before, they used ⌈ctx⌉ · 5 · idx_dim / 8.
- System fuzzer: `val5/v0145/sysfuzz.py` (the 0.140 harness with the new models added).
- Fingerprints are unchanged.

## 92. NPU design insight: which hardware knob moves TTFT / TPOT / throughput, per area / power proxy (0.146)

**Purpose.** On the product side, the question is which knob to spend silicon or package budget on, given a chip
configuration and the models it must serve. `core/insight.py` answers it with one-variable probes of the existing engine,
the same `evaluate` that the eval sensitivity and the sweeps use. The new module adds no hardware model. It is served at
`POST /api/insight` (body `scenario`, `chip_preset`, optional `models` ≤ 8, `budget`, `energy`) and shown in the UI tab
「设计洞察」.

**Probes** (each relative to the base point; the sizes are probes, not cost claims):

| knob | change | cost class |
|---|---|---|
| sram_x2 | SRAM × 2 | on-die |
| slc_add | add an SLC of the SRAM size | on-die |
| macs_x2 | MAC rows × 2 | on-die |
| engines_x2 | cores × 2 | on-die |
| shape_tall | rows × 2 / cols ÷ 2 (equal MACs) | none |
| dram_rate | next catalog speed grade | off-die / PHY |
| dram_count | next catalog package or stack count (bandwidth + capacity) | off-die / PHY |
| native_fp8 | value of native FP8 / INT8 (removed vs present, or added at the FormatSupport default × 2 「假设」) | none |
| native_mxfp4 | add native MXFP4 MAC at 2 × FP8 「假设」 (Blackwell dense FP4 / FP8) | none |
| dequant_x4 | conversion throughput × 4 | none |
| d2d_x2 | D2D bandwidth × 2 (D2D packages only) | off-die / PHY |
| c2c_x2 | card link bandwidth × 2 (> 1 card only) | off-die / PHY |

**Metrics.**
- Per model: TTFT is the prefill of the scenario's prompt; TPOT and decode tok/s/card are at the scenario batch.
- Each model gets a valid layout with the same card count. MoE models put experts over the TP·DP ranks.
- A knob's score is the geometric-mean speed-up over the models. A score within 1 % (the sensitivity tie) counts as no
  move, and scores below that are listed under "worse".
- Area: Δ die-area proxy from the user's budget densities (`budget.area_estimate`). Off-die knobs have no mm²: PHY and
  package are not in the proxy.
- Power proxy: Δ J/token from the user's energy table. Without a table it is the Δ DRAM bytes and MAC slots per token
  (action counts, no assumption).

**Default model set:**

| model | why it is in the set |
|---|---|
| qwen3-8b | dense reference |
| gpt-oss-120b | MXFP4 experts, sink, SWA |
| mimo-v2.6-flash | MXFP4 experts, SWA + sink |
| step-5-preview | region-summary DSA indexer, FP8 |
| deepseek-v3.2 | DSA indexer, MLA, FP8 |
| ling-3.0-flash-vl | KDA hybrid |

**What it shows on the defaults** (100T, LPDDR5X 4×64; these are examples, not product conclusions):
- At batch 1–8, decode is DRAM-bound. DRAM package count, then speed grade, lead TPOT. MAC rows × 2 does nothing for
  decode: small M leaves the extra rows idle.
- Cores × 2 and MAC rows × 2 lead TTFT.
- A native MXFP4 MAC alone gives 0 % for the as-released MXFP4-expert models. They run W4A8 / W4A16, and `gemm_exec`
  only uses a native format when weights *and* activations share it. The mechanism-consistent levers for them are DRAM
  bytes (already 4.25 bits per weight) and the dequant path.
- Tests: test_core_146. The probe numbers equal a direct `evaluate` of the one-variable change. Area and energy are
  absent without user inputs. Playwright smoke: `val5/v0146/ui.py`.

## 93. Mixed-precision native MAC (W4A8 / W4A16 / W8A16) as a design variable (0.147)
- Gap left by §92: a native MXFP4 MAC gives 0 % for models released as W4A8 (MiMo experts: mxfp4 × fp8) or W4A16 (gpt-oss
  experts: mxfp4 × bf16), because a native format only applies when W and A share it.
- `FormatSupport.mixed`: (w, a, rate) pairs. If (w, a) is listed (fp8_tensor folds to fp8), `gemm_exec` returns
  `("w*a", rate, "none", 0)`, so there are no `convert_elems` and no vector-path dequant. Otherwise the earlier rule applies
  unchanged. The dequant cost vanishes only when the mixed MAC is native.
- Rates: W4A8 at the FP8 rate is sourced (PTX ISA 8.7, tcgen05.mma kind::f8f6f4 runs FP4/FP6 × FP8 at the f8 rate).
  W4A16 and W8A16 at the bf16 rate are 「假设」.
- Default chip unchanged: no mixed pairs. The gain depends on whether dequant is the bottleneck. On the default 100T chip
  it is hidden (×1.00). With `dequant_elems_per_cycle` = 256, gpt-oss-120b at b64 decode drops from 846 ms to 347 ms TPOT
  with W4A16.
- Tests: test_core_147. Fuzzer knob `mixed_all` (monotone). Playwright: `val5/v0147/ui.py`.

## 94. Insight upgrade path: greedy multi-step design exploration (0.148)
- `upgrade_path(base, models, metric, budget, steps, max_cost)` builds on the §92 probes. At each step it evaluates every
  additive knob of `knobs(current)` on every model (prefill and decode) and keeps a candidate only if:
  - the model-set geomean gain is > TIE;
  - every model's metric is ≥ its start value (exact, cumulative);
  - a model that fit still fits;
  - the budget holds.
  It then picks max ln(gain)/cost. Inverted knobs (removing a feature) are never steps.
- Cost:
  - Area mode (all densities given): die knobs cost their Δmm²; other knobs use REL_COST × 1 % of the start area
    「假设」. The cap is `budget.die_mm2`.
  - Proxy mode: REL_COST units 「假设」 (SRAM ×2 = 4, MAC rows ×2 = 4, cores ×2 = 6, DRAM count = 3, DRAM rate = 1,
    mixed W4A8/W4A16 = 1, W8A16 = 0.5, dequant ×4 = 0.5, …). SRAM and MAC costs scale with the current size. The cap
    is `max_cost`.
- Guarantee: never worse than the start, per model, on the chosen metric. It is asserted in code, tested, and fuzzed.
  Greedy is not globally optimal. The top 3 alternatives are reported per step.

## 95. Cited density presets, interface PHY area, beam-search upgrade path (0.149)
- Presets (`budget.DENSITY_PRESETS`) fill only unset fields; user entries win. Values:
  - **n28_tpuv1**: 4.58 mm²/MiB (24 MiB UB ≈ 1/3 of a < 330 mm² die) and 1.29 mm²/kMAC (int8 MXU = 1/4 of the die).
    Source: Jouppi et al., ISCA 2017, with the die size from ISCA 2021 Table 1. Both are upper bounds.
  - **n7_tpuv4i**: 0.875 mm²/MiB (128 MiB CMEM = 28 % of < 400 mm²) and 0.6875 mm²/kMAC (4 × 128² bf16 MXUs = 11 %).
    Source: Jouppi et al., ISCA 2021. Both are upper bounds.
  - **n5_derived** (「推导」): 0.68 mm²/MiB (N7 → N5 HD bitcell 0.027 → 0.021 µm²) and 0.38 mm²/kMAC (TSMC N5 logic density
    1.8× N7). The array-only lower bound is 31.8 Mib/mm² = 0.25 mm²/MiB.
  - **PHY**, all presets:
    - UCIe-A x64 module 388.8 × 1043 µm = 0.406 mm² (「来源」, HC2023 UCIe tutorial).
    - C2C 0.027 mm² per GB/s per direction: quad 112G 5 nm SerDes 1806 × 825 µm (Varzaghani et al., VLSI 2022; 「二手」).
    - HBM 14 mm² / 1024 DQ (「二手/假设」, die-shot estimates; ISSCC 2023 RX-only 0.0056 mm²/DQ is the lower bound).
    - LPDDR 0.03 mm²/DQ (「假设」, no public figure found).
- Mechanism: PHY area scales with what the hardware actually adds:
  - DRAM: pins (bus bits), so a rate upgrade adds no PHY area and a package-count upgrade does.
  - D2D: units (UCIe modules).
  - C2C: GB/s.
  `full_area` = chip densities + PHY. It is used by the budget area check (only when a PHY density is set), insight
  Δmm² and upgrade-path costs. A knob with no area term keeps the 「假设」 relative cost (1 unit = 1 % of the start area).
- Beam search (width w): each state expands its w−1 best-ratio steps plus the largest-gain step. Kept states are the w−1
  most efficient partial paths plus the highest-gain one. Greedy always runs too, and the beam result is returned only if
  its final gain is ≥ greedy's. The per-model "never worse than the start" rule is unchanged.
- Fix: the `d2d_x2` knob now doubles `d2d_units` for catalog grades. Previously it changed `d2d.GBps`, which only applies to
  `custom`.

## 96. Hostile review of insight / upgrade path / density (0.150)
- Units: SRAM density is mm² per MiB (8 Mib), and the bitcell array density 31.8 Mib/mm² = 0.25 mm²/MiB. kMAC = 1024 MAC
  units of the chip's array. The n28 preset is an int8 MAC, which is optimistic for bf16, and this is noted. PHY: per
  DRAM data pin, per D2D unit, and per GB/s of uni-directional link (`Link.GBps` is per-rank uni-directional).
- No double counting: chip terms (sram / slc / mac / fixed) and PHY terms (dram_phy / d2d_phy / c2c_phy) are disjoint.
  C2C is counted only for cards beyond one package. `mm2_fixed` must exclude any PHY that is set separately, and a note
  says so.
- Precedence: a user value (including 0) wins over a preset in both the API and the UI. Only None or empty is filled.
- The cost cap applies in both modes. Beam is deterministic (no set or dict-hash ordering; scenario hashes are sha
  based) and is checked across PYTHONHASHSEED.
- PD: insight and the path evaluate colocated prefill/decode and say so. Non-LLM domains are not insight models (the API
  rejects them), but their area / budget pass the PHY-aware path (sysfuzz `area`).

## 97. Insight multi-domain metrics and export (0.151)
- One report = one catalog domain. Mixed sets are rejected.
  - llm: TTFT, TPOT, decode tok/s/card (decode + prefill evaluates).
  - gen: latency (clip / image), frames/s/card (image DiT = 1 frame → images/s).
  - protein: batch latency, sequences/s/card.
  - genomics: batch latency, windows/s/card.
- Gain: latency-like metrics (ttft / tpot / lat) are lower-better; thr is higher-better. The upgrade path's
  never-worse rule uses the same direction.
- Export: kinds `insight` and `insight_path` under the shared EXP_SCHEMA = 1 envelope (same as compare / sweep).

## 98. Cited energy presets and insight rank_per_J (0.152)
- `ENERGY_PRESETS` (horowitz_45nm, tpuv4i_7nm): Accelergy-style action costs with the same status labels as
  `DENSITY_PRESETS` (来源 / 推导 / 假设). `apply_energy_preset` fills only unset fields; idle_W never preset.
- Horowitz ISSCC 2014 Fig 1.1.9 (45 nm, 0.9 V): 16b FP mult 1.1 pJ, 16b FP add 0.4 pJ, 32b SRAM 5 pJ → 5/32 pJ/bit,
  32b DRAM 640 pJ → 20 pJ/bit. UCIe Advanced 0.25 pJ/bit; NVLink-C2C 1.3 pJ/bit. net 5 / host 10 pJ/bit are 「假设」.
- TPUv4i ISCA 2021 Table 2: 45→7 nm average energy 2.6×, SRAM 1.3–2.4× (geom. mean), DRAM 6.3× (HBM) → 「推导」 from Horowitz.
- Insight: accumulate absolute ΔJ = hi.J_tok − lo.J_tok (not relative); `rank_per_J` = (score−1)/d_J when d_J>0;
  win-win (score>TIE and d_J≤0) → 1e30; near-zero |d_J|<1e-12 → 0. Multi-domain path hostile: never-worse + beam≥greedy.

## 99. Absolute electricity cost ranking and Step-5 vision review (0.153)
- `Budget.usd_per_kWh` + `Budget.tdp_W_card` (optional). `ELECTRICITY_PRESETS`: EIA 2024 US industrial 8.13 ¢/kWh and
  commercial 12.75 ¢/kWh (「来源」); China 36-city industrial 0.62 CNY/kWh (2024-09, chinaprice.cn) ÷ 7.1 FX 「二手/假设」.
  Presets fill only unset `usd_per_kWh`; never a silent default.
- Insight: J per unit = energy-table `J_tok` if present, else `tdp_W × tpot` (LLM) or `tdp_W / thr` (other domains) as
  constant-power 「假设」. `d_usd = d_J × usd_per_kWh / 3.6e6`; `rank_per_usd` mirrors `rank_per_J` (win-win → 1e30).
- Step-5 vision remains `partial`: open weights expose overview 728² → 169 tokens and an external crop interface
  (504² → 81 tokens/crop via `patch_pixel_values` / `num_patches`); the crop-count algorithm is not published.
  Platform docs only give detail low/high longest-side resize tips.

## 100. Optional TCO ranking (0.154)
- $/unit = CapEx + OpEx. CapEx = card_usd / (amort_years · 365.25 d · thr), thr = units/s/card (LLM decode tok/s/card).
  OpEx = J/unit (energy table, else TDP × time 「假设」) · PUE (missing → 1, IT-only, noted) · $/kWh / 3.6e6.
- Knob score `tco_gain` = geometric mean over models of TCO(lo)/TCO(hi); ranked when > 1 + TIE. The card price is held
  fixed per knob 「假设」; hardware increments are costed separately (Δmm², upgrade path).
- PUE presets: Uptime Institute 2024 survey 1.56 (per-site), 2023 capacity-weighted 1.47 (「来源」).

## 101. PP / fabric standing invariants (0.155)
- PP: stages partition [0, L) contiguously. Σ stored weights = PP1, except that a tied embedding whose LM head sits on
  another stage keeps its own copy (≤ one vocab × hidden table). Σ KV = PP1. max dram_need ≤ PP1. Decode weight reads
  per tick may drop by at most the extra SRAM (pp − 1 cards; with microbatches, staging shrinks so up to pp · SRAM).
- Fabric: per collective, `pick` minimises bw + α, and α is charged in `t_sync`, so the guard compares stage totals,
  not `t_link`. Monotone in link GB/s, α, oversub and channel cap; in-network reduce only adds candidates.

## 102. Datapath / residency / non-LLM multi-card invariants (0.156)
- Dequant throughput and vector lanes are monotone. At default settings the dequant path is hidden for AWQ / MXFP4
  decode (×4 or ÷4 changes TPOT by < 1 ulp) and dominates only near 1 element/cycle.
- Residency: pinned bytes rise with SRAM and fall with batch (activation staging); per-step weight reads may rise with
  batch (MoE expert coverage, less pinning), but per sequence they fall.
- Non-LLM: TP / SP conserve FLOPs; DP for diffusion is the CFG cond / uncond split (latency ≤), for protein /
  genomics it replicates whole requests (latency =).

## 103. Datapath sizing headroom (0.157)
`sizing_report(base, models, budget, resources)` answers the inverse of the knob probes: how far can dequant throughput, vector width or SRAM shrink before any model's native metric (TTFT/TPOT/tok·s; latency/frames·s; …) loses more than TIE = 1 % or stops fitting. Search: halve until failure (floor current/1024), then 3 geometric bisections; min_ok is always an evaluated passing point (no extrapolation), relying on the 0.156 monotonicity invariants (fewer lanes / dequant / SRAM never speeds up). Per-model layouts follow `model_scenario`. Advisory only; no defaults changed, fingerprints unchanged.

## 104. Sizing: MAC cores and DRAM bandwidth (0.158)
`cores` changes `chip.engines` with vector_lanes pinned to the current width, so only the MAC array (plus the per-engine accumulator / SRAM port, which scale with engines) shrinks; area saved = (E − E_min)·R·C/1000·mm2_per_kmac, and only when that density is set. `dram_bw` scales effective bandwidth through `mem_eff` (≤ current, so always valid). Capacity and stack stay fixed; KV streams follow it unless `kv_eff` is set. Same search and acceptance rules as §103.

## 105. queue_report performance (0.159, bit-identical)
Memoising pure sub-laws (victim wait law, saturation-eviction quadrature) and enlarging the bulk-chain LRU cut the recompute KV report by about 40 % while every reported float stays the same. Changing summation order or quadrature resolution to go further is not bit-identical, so it was deliberately not done.

## 106. Joint sizing (0.160)
Single-resource minima are each checked against the reference chip with everything else unchanged. Applied together they compound, because the shrunk resources share critical-path time. The joint pass starts from all minima, measures the worst gain over models × metrics (−∞ on a fit loss), and while it is below 1 − TIE backs off the one resource whose step toward its current value, √(v·cur) (snapping to cur within 3 %; integers move by ≥ 1), maximises the worst gain. It always terminates, because all resources back at current is the reference chip. Order of application: cores (which pins vector width) before lanes. Area totals use only the set densities.

## 107. Area-aware joint sizing and binding summary (0.161)
Step rank with densities: Δ(worst gain) / Δmm² (a resource without a density costs 0 mm², so it is given back first; flagged as unpriced). A beam of width 3 runs next to the greedy path. The objective is priced mm² saved, else Σ log(cur/v). The reported point is always a probed pass, and it is never worse than greedy because greedy's point is a candidate. `binding`: min_ok = current → binding resource; the others ordered by headroom; joint back-offs not in the binding set → bind only in combination.

## 108. Vector-lane / dequant densities and unpriced resources (0.162)
Sources:
- Horowitz, "Computing's energy problem (and what we can do about it)", ISSCC 2014 plenary. 45 nm operator areas, area column as reproduced in Sze et al. (MIT EEMS) and NYCU AAML lecture slides (二手): 8b add 36 µm², 16b FP add 1360, 32b FP add 4184, 16b FP mult 1640, 32b FP mult 7700.
- Jouppi et al., ISCA 2021 (TPUv4i) Fig. 6 / §7.A: MXU 11 %, CMEM 28 %; no VPU share published.

Derivation (推导/二手): node density = node `mm2_per_kmac` × area ratio to a bf16 MAC (FP16 mult + FP32 add = 5824 µm²). Lane (FP32 FMA datapath, 11884 µm²) ×2.0405; dequant element/cycle (FP16 scale mult + 8b zero-point add, 1676 µm²) ×0.2878. Assumes operators scale like the MAC array across nodes. Datapath only (no RF / control / shuffle), so these are lower bounds. TPUv1's int8 MAC density makes its derived values optimistic.

Sizing-only fields: the die-area proxy is untouched. Unpriced resources (DRAM bandwidth via mem_eff) default to `unpriced="last"`: ranked after improving priced steps. "free" ranks them first (0.161). Units: cores R·C/1024 × mm2_per_kmac (fixed from /1000); lanes / dequant per 1024.

## 109. Sizing setter isolation (0.163)
Each sizing resource must move alone. The lanes setter pins the dequant rate, which otherwise defaults to the vector width; cores pins the vector width (0.158). Guarded on all chip presets. Caveat: with the opt-in DMA limit, dram_bw sizes the memory, not min(memory, DMA).

## 110. Host / runtime step overhead (0.164, opt-in)
Per engine step, host h = (host_step_us + host_seq_us · ⌈batch/dp⌉) µs. Decode divides by host_steps (vLLM / SGLang multi-step scheduling; prefill steps run single-step, as in vLLM PR #7000). host_overlap "serial": step = device + h. "async" (output processing / scheduling overlapped with the previous device step): step = max(device, h), with host_bound = h > device. Sources (mechanism only): vLLM v0.6.0 performance blog; WukLab "Can Scheduling Overhead Dominate LLM Inference Performance?" (2024). No numeric default: the published absolute times are runtime- and CPU-specific. Non-LLM domains are unchanged.

dram_bw sizing (§104) now uses delivered bandwidth min(memory, DMA issue limit).

## 111. VLM video input (0.165, opt-in)
Models: those with a released Qwen3VLVideoProcessor (Qwen3.5-397B-A17B, Qwen3.8-27B, Qwen3.8-Flash-Next, Intern-S2-397B). Bounds 「来源」 from video_preprocessor_config.json: min 4096 / max 25165824 px over the clip.
- Resize (transformers smart_resize, verbatim): h̄, w̄ are rounded to 32 and t̄ = round(F/2)·2. If t̄·h̄·w̄ > max: β = √(F·h·w/max), h̄ = ⌊h/β/32⌋·32 (likewise w̄). The min bound is handled symmetrically. Optional per-frame cap (qwen-vl-utils): max = max(min(768·32², ⌊max/F⌋), ⌊1.05·min⌋)·F.
- Grid: grid_t = ⌈F/2⌉, ph = h̄/16, pw = w̄/16. LLM tokens = grid_t·ph·pw/4, added to prompt/ctx.
- Encoder: grid_t passes of the per-image tower op graph on a ph×pw grid. The reason is that cu_seqlens = repeat_interleave(h·w, t) (vision_utils.get_vision_attention_seqlens, merge_temporal=False), so attention is quadratic in ph·pw per slice and linear in grid_t. The Conv3d patch embed has K = 3·2·16².
- 「假设」 defaults: 32 frames, 1280×720. Timestamp text tokens are not counted.

## 112. Kimi video chunks (0.166, opt-in)
Models: Kimi-K2.5 and Kimi-K2.7-Code. Settings are 「来源」 from preprocessor_config.json media_proc_cfg and config.json vision_config (listed in CHANGELOG 0.166).
- Chunks: n = ⌈F/4⌉. The last chunk has F mod 4 frames when that is nonzero.
- Per-frame resize (navit_resize_image): limit = 4096 patches, or min(round(655360/frames in chunk), 4096) when in_patch_limit_video is set. Side limit 512 patches. Pad to 28.
- Tower: one pass per chunk on tframes·ph·pw patches. Attention is joint over the chunk (cu_seqlens = t·h·w). Then a temporal mean (vis.tpool, vector op), then the merger on ph·pw/4 rows.
- LLM tokens = n·ph·pw/4.
- Contrast with Qwen (§111): Qwen merges 2 frames in the Conv3d and attends per slice. Kimi embeds every frame, attends across 4 frames, then pools.
- Not counted: timestamp "hh:mm:ss.fff" and media marker tokens per chunk.

## 113. Step caching for DiT generation (0.167, opt-in)
TeaCache (TeaCache4Wan2.1 teacache_generate.py, 「来源」 for the forced-step rules and which ops run on a skipped step).
- Skipped forward = full_io_ops pre (patch, time, text embedders) + post (head) + one residual add (tokens × hidden vector op). Timed with `_component_time` (weights read from DRAM).
- Clip denoise = steps·per_step·t − k·per_step·(t − t_skip). CFG branches skip in lockstep (the skip test uses only the timestep embedding).
- Memory: one cached residual per forward sequence (cond and uncond each) on the DP rank: ⌈seqs/dp⌉·⌈tokens/sp⌉·hidden·act bytes, added to stage 0.
- Forced steps: teacache = first and last; teacache_ret = first 5.
- k is a user input; there is no default. PP > 1 and use with prefix_kv are not modelled (warning, ignored).

## 114. Host-memory L2 prefix cache (0.168, opt-in)
SGLang HiCache (「来源」 docs.sglang.io hicache_design: L1 GPU / L2 host; write_through / write_back; layer-wise compute-transfer overlap).
- K2 = ⌊prefix_host_GB / footprint⌋, with the same footprint as L1. Reachable prefixes: write_through max(K1, K2), write_back K1 + K2. Hit rate H(·) by Che over the union 「假设」.
- Host share h2 = H(reach) − H(K1). The prefill computes S − H(reach)·Lp tokens per request (through the existing cached-token path).
- Load of an L2 hit: t = max-per-card shard bytes / host GBps. Exposed = max(t/L, t − TTFT_p·(L−1)/L). TTFT += h2·exposed (mean). Host link capacity per prefill replica = GBps / (h2 · bytes per card).
- Whole-prefix model only. Not modelled: tree / multi-turn, D2H write traffic, L3.

## 115. Host L2 on the radix tree / multi-turn; backup writes (0.169)
- Tree / multi-turn: token capacity C = K_float·Lp. Reach = max(C1, C2) (write_through) or C1 + C2 (write_back). H(reach) and H(C1) by the existing per-node Che models. h2 = H(reach) − H(C1).
- Writes: wf = 1 − H (write_through) or 1 − H1 (write_back), as a fraction of the prefix. Bytes per card per request = wf · prefix shard bytes. Capacity per prefill replica = host GBps / that, on the card→host direction (full duplex, separate from loads).

## 116. StripedHyena 2 / Evo 2 (0.170)
Sources: the evo2 inference yml files (layer pattern hcs / hcm / hcl / attn, filter lengths and groups, state_size, inner_mlp_size, max_seqlen 1M) and checkpoint headers read over HTTP range (tensor shapes match exactly). Operator semantics come from vortex `engine.py`.

| op | prefill (L = prompt, N = 2L, per channel) | decode / cached state |
|---|---|---|
| featurizer | 3-tap FIR over 3h, x1·v, x2·z | 2 bf16 columns × 3h |
| SE (7 taps) | 14 flop / token | 6 bf16 columns |
| MR (128 taps) | rfft + irfft (2.5 N log2 N each) + 6(N/2+1); DRAM 44·L B | 127 bf16 columns, 256 flop |
| LR (16 modes) | build h (2·16·L, exp 16·L), rfft(h), fft(x1v), ·H, irfft; DRAM 12L + 60L·b B | 16 fp32, 4·16 + 2 flop, 16 exp |
| LR modal-FFT state | 16 × [fft(p^t) 5N log2 N + b·(6N + ifft)]; DRAM 16·(24 + 80·b)·L B; live 2·b·16·N·8 B | (state above) |

- The FFT operands round-trip DRAM because the reference runs one kernel per torch.fft / elementwise call. 「推导」 Byte counts are input + output per call.
- Chunked prefill (ctx > 0) has no counterpart in the release. 「假设」 It is modelled as a convolution over the chunk plus 2·16 flop per channel per token for the carried state.
- Not modelled:
  - The opt-in HCS / HCM / HCL fused kernels (`use_h*_kernel`, off in the configs).
  - FlashFFTConv (`use_flashfft: False`).
  - The `recurrence` prefill style: the release uses fft.
  - The scoring-only forward without a cache, which skips the modal state.
- **Gated MLP (0.444).** `ParallelGatedMLP` is `l3(act(l1(z)) * l2(z))`. With `mlp_activation: gelu` and `evo2_style_activations: true`, layer 0 uses `F.gelu` (`approximate='none'`, the erf form) and every later layer uses `Identity`. The bill is erf-GELU plus one multiply on layer 0, and the multiply only after that. Neither upcasts. The GEMMs stay `mlp.gate_up` (N = 2 * inner) then `mlp.down`.
- **RMSNorm stays on the generic tail.** The eager vortex norm is `x / (x.norm(2) * hidden**(-0.5) + eps) * scale` (`use_flash_rmsnorm: false`). eps sits outside the square root, so this is not `rsqrt(mean(x^2)+eps)*weight` and is not newly billed.

## 117. Evo 2 long-context prefill (0.171)
- **CP (savanna use_cp_hyena):** per Hyena layer, 4 all-to-alls of the rank's block of x1, x2, v and z (b·S/cp·h/TP·2 B each) plus a p2p halo of (taps − 1)·3h/TP columns. The FFT conv runs at 2S on ch/(TP·cp) channels, which is the same work as TP·cp channel sharding. Attention uses the 0.88 ring.
- **hyena_prefill:**
  - reference (as released).
  - recurrence (vortex direct recurrence): 12·16 flop per token per channel, about 120·b·ch·16 B per token of state traffic, 14·L launches, and a peak of max(h build, 4·b·ch·L·16 B).
  - fused (what-if): single-pass FFT conv. Spill per transform row = (p − 1)·16·N B, with p = ⌈log N / log n_fit⌉ and n_fit = SRAM/2/engines/8.
- **Design insight:**
  - A fused Hyena-LR conv needs about 8·2L B of on-chip memory per in-flight channel to stay single-pass: 128 KiB at 8k, 16 MiB at 1M.
  - Beyond that, every extra out-of-core pass costs 16·N B per channel. At L = 1M (N = 2M) that is 32 MiB of DRAM traffic per channel per extra pass. With H100-class SRAM per engine, p = 2 at 1M.
  - The as-released code moves about 20× the weight bytes at 8k. Fusion, not extra bandwidth, is the lever.



## 118. Long-context attention vs roofline (0.172)

- **Symptom.** llama-3.1-8b prefill on H100-SXM at 131k had an attention array time of 152.5 s against a 6.4 s ideal. The same ~25–34× ratio showed up at every length and in the Evo 2 MHA layers.
- **Mechanism.** On the reference GPUs, `instance_sched = "wide"` gangs 132 SMs into a 32 × 8448 output-stationary array (§26). PV has n = head_dim = 128, so it filled 1.5 % of the columns. QK (n = ctx) was already near 1×.
  - This is not DRAM score materialisation: attention is costed as fused, and t_dram is 0.70 s at 131k.
  - It is not KV re-reads per query tile.
  - Abstract chips (`auto`) and the opt-in `attn_sched = "cta"` were already at 1.0–1.17×.
- **Reference.** 「来源」 FA2 (arXiv 2307.08691): 73 % of peak on A100 and ~35 % on H100, with CTAs over (batch, head, q-block). FA3 (arXiv 2407.08608): ≤ 740 TFLOPS BF16 forward on H100, 75 %.
- **Fix.** Attention ops choose the cheaper of C = A·B and Cᵀ = Bᵀ·Aᵀ (`evaluate.ATTN_ORIENT`, default on). Kernels are free to pick either orientation, as the full-sequence path has done since 0.40.
  - Now 1.00–1.19× on all reference chips and presets, 4k–131k. This is an upper bound on speed: 1.3× faster than FA3 and 2.8× faster than FA2 on H100. The sourced opt-in ceiling (`fmt_ceiling` "attn:bf16") narrows the gap.
- **Invariant.** `tests/test_core_172.py`: attention array time / FLOP time stays in [0.9, 3] for every REF_HW chip plus 100T / 1P at 4k / 32k / 131k.
- **Impact.** See CHANGELOG 0.172.0. llama-3.1-8b 131k TTFT goes from 156 s to 8.0 s. evo2-7b 1M on tp8 × cp8 goes from 25.9 s to 2.78 s.
  - The R4 hold-out for `catalog_kernel` moves from 0.892 / 26.6 % to 1.016 / 16.8 %. The `default` variant is unchanged.

## 119. Mapping-artifact sweep and roofline-ratio invariant (0.173)

- **Sweep.** Every GEMM / attention op of all listed models, phases {prefill 8k, full 4k, decode b1, b64}, chips {H100, B200, MI300X, A100, 100T, 1P}. For each op: array time ÷ FLOP time, against the swapped orientation and the geometric tile underfill.
- **Artifacts found and fixed** (all were a single forced plan, not physics):

  | class | ops | before | after |
  |---|---|---|---|
  | small N on wide | k / v proj (N 1024) | 8–18× | ≈ 1× |
  | | routers (N 256–384) | 37–74× | ≈ 1× |
  | | hc mixers (N 4–24) | 350–4700× | ≈ 1× |
  | instances on wide | MoE experts, few tokens | 4–9× | ≤ geometric |
  | | linear-attention chunks | 66–132× | ≤ geometric |
  | | B200 long attention | 4.6× | ≤ geometric |
  | serial K, tiny output (non-cta) | hc mixer decode on 100T | 29 µs | 1.8 µs |

  Mechanisms: Cᵀ = Bᵀ·Aᵀ (cuBLAS row-major); grouped GEMM / FA grid over (batch, head, q-block); split-K / Flash-Decoding. `evaluate._wide_alt`, `_splitk_alt`.
- **Invariant** (`tests/test_core_173.py`): array ≤ max(3 · g · FLOP, operand bytes / BW) + 2 µs, for 13 model kinds × 4 phases × 9 chips.
- **Kept, by design:**
  - The wide gang width (B200 18944 columns) gives 2–6× on full-sequence DiT shapes. `WIDE_GEMM_AUTO` removes it but worsens extval (default Mixtral TP8 1.14 → 1.17, MI300X max-load 2.09 → 2.39). The underfill stands in for real kernel efficiency, which has no source here.
  - Small-M decode on os arrays is a real design effect.
- **Extval** (speed ratio, see §124, default | catalog_kernel), 0.172 → 0.173:

  | group | default | catalog_kernel |
  |---|---|---|
  | v1 static | 1.06 → 1.13 | 1.19 → 1.31 |
  | v1 TTFT | 1.29 → 1.33 | 1.66 → 1.75 |
  | v1 max-load | 0.89 → 0.92 | 1.08 → 1.12 |
  | MoE large-EP | 0.95 → 0.97 | 0.62 → 0.82 |
  | Mixtral TP8 | 1.01 → 1.14 | 1.13 → 1.34 |
  | MI300X lat | 1.39 → 1.40 | 3.28 → 3.33 |
  | MI300X max-load | 1.98 → 2.09 | 2.15 → 2.32 |
  | B200 NVFP4 | 0.98 → 0.99 | 1.24 → 1.26 |

  Removing artifacts exposes missing real-kernel efficiency and fixed costs; no number was fitted.
- **Over-predicting catalog_kernel groups.** The catalog_kernel variant has no per-layer fixed cost. Adding the public-trace 62.3 µs/layer (§39) gives:
  - v1 static 1.31 → 1.19;
  - v1 TTFT 1.75 → 1.36;
  - B200 NVFP4 1.26 → 1.05.

  Library GEMM ceilings add only about 0.01. The residual is the 128-token TTFT rows (1.3–1.7×, fixed-cost dominated). There is no sourced breakdown of trtllm-0.8 TTFT framework time, so it is left open.

## §120 Muse Spark Open（0.174）

- 结构（来源：PR #61879 / AMD config.json）：62 层全 MoE，hidden 8192；GQA 128/16 头，head_dim 64，带每头 o-gate（q 投影宽 2·H·hd）；256 专家 top-8，专家 FFN 4096，无共享专家；latent pooler/unpooler 8192↔4096 包围专家；sandwich norm；SWA 2048 的 4 层周期锚定在末层（i%4==1 为全注意力，16 层，NoPE）；vocab 202048。
- 推导：1M 上下文 KV ≈ 全注意力的 0.26（博客"略多于 ¼"）。
- TPU v7 sanity（不拟合）：MXFP4 release 每 token 激活字节 53.6 GB（假设 megakernel 用类似 4-bit 专家 + bf16 attn/latent）；HBM 7380 GB/s/芯片（来源：Google Cloud TPU7x 文档）→ 屋顶线 550 / 1101 tok/s（4 / 8 芯片）。博客无投机 270 / 350 tok/s = 0.49 / 0.32 屋顶线；DSpark（接受长度 6）650 / 960 tok/s。仅作量级检查。
- 视觉：PR NaViT（50 层，13 全注意力）+ adapter = 1.89B（博客 1.9B）；37 窗口层按全注意力计（上界）；缩放用 4096 token 像素预算 smart-resize（假设）。
- 未做：audio 编码器（24 kHz，1920 samples/token，来源 processor_config）仅计存储；NVFP4；catalog_kernel 62.3 µs/层固定开销（推迟）。

## §121 Muse Spark Open 跟进（0.175）

- NVFP4（推导）：来源 = 博客（Meta 发布 BF16 + NVFP4）、PR #61879（ModelOptNvFp4Config；MegaMoE 支持 BF16 / NVFP4；pooler/unpooler quant_config=None）。字节 = 0.5 + 1/16 B/参数 + 每张量 2×fp32。假设：仅路由专家量化；若注意力也量化则 −19.4 GB。
- NaViT 窗口：每个窗口层 key 数 = Σ_块 (块内 patch 数)²，块 = pos_emb 32×32；1920×1080（78×138 patch）时窗口层 attention 降到全注意力的约 0.12。
- 音频：AutoEncoder 4 级（256/512/1024/2048，各 2 块：depthwise causal conv k7 + RMSNorm + GELU MLP×4），o_proj → 8192；60 s = 750 token、0.59 TFLOP（手算一致）。放置同视觉塔（stage 0 数据并行）。
- DSpark：5 层稠密草稿（GQA 128/16×64 + o-gate，SWA 2048，SwiGLU 16384），fc 5·8192→8192，自有 lm_head，Markov 头 rank 512，置信头 8704→1，块 7；参数与 safetensors 头精确一致。目标层 hidden 是拼接（模板按 nt·h→h GEMM，同形状）。
- 未建模：Muse 视频处理器（MuseGlimmerVideoProcessor 未公开）；图像缩放规则仍为假设。

## §122 Mechanism-scoped per-layer fixed cost (0.176)

**What the 62.3 µs is.** deepseek-ai/profile-data decode.json, compute stream 7: 1,734 small (non-GEMM / attention /
communication) kernels take 7.55 ms over 61 layers × 2 micro-batches. Their durations are p10 2 / p50 3 / p90 7 µs. At
64 requests per micro-batch the bytes they move need well under 1 µs, so at decode they are latency-bound. The lump
does not include the idle gaps between kernels.

**Second operating point.** prefill.json (16K tokens per GPU): the same kernel families take p50 116 µs. That time is
work, which the model already charges as vector ops, so applying the lump at prefill counts it twice.

**Gaps (launch runtime).** Idle time between consecutive kernels on the compute stream, counting gaps under 50 µs
(longer gaps are all-to-all waits):

| trace | runtime | mean gap per kernel | per layer-forward |
|---|---|---|---|
| decode | CUDA graph (182 cudaLaunchKernel for 3,711 kernels) | 1.025 µs | 31.1 µs |
| prefill | eager | 2.404 µs | 67.5 µs |

The runtime split (graph decode, eager context) is also the TRT-LLM PyTorch backend's documented behaviour
(perf-host-analysis skill).

**Model.** Per kernel the cost is floor (2.0 µs) + gap(runtime of the phase). Kernels are the model's own ops (GEMM,
fused attention, stand-alone vector), so the cost follows the model, not DeepSeek's 30 kernels per layer. It applies to
CUDA rows only.

**Bands, catalog_kernel → catalog_kernel_fixed** (speed ratio, see §124 — gmean / mean abs error):

| group | catalog_kernel | catalog_kernel_fixed |
|---|---|---|
| v1 static | 1.31 / 38 % | 1.23 / 32 % |
| v1 TTFT | 1.75 / 76 % | 1.40 / 41 % |
| v1 max-load | 1.12 / 21 % | 1.03 / 19 % |
| MoE large-EP | 0.82 / 33 % | 0.81 / 32 % |
| Mixtral TP8 | 1.34 / 47 % | 1.25 / 44 % |
| MI300X lat | 3.33 / 233 % | unchanged (no ROCm trace) |
| MI300X max-load | 2.32 / 132 % | unchanged |
| B200 NVFP4 | 1.26 / 26 % | 1.11 / 13 % |

For reference, the lump (+62.3 µs per layer, every phase and every vendor) scores 1.19 / 1.36 / 1.00 / 0.81 / 1.23 /
1.40 / 2.18 / 1.05. It scores better because it stands in for kernels the op list omits (17.2 per layer modelled vs
30.4 in the trace) and for unmodelled ROCm overhead, but it is not mechanism-scoped. The default keeps the lump: default
+ mech would be worse on 6 of 8 groups for the same op-count reason. The next fix is listing the missing ops (fp8 casts,
split-K reduce, sampling), not tuning a constant.

**128-token TTFT residual** (measured − catalog_kernel_fixed): 1.8–2.1 ms (H100 / H200 7B), 3.7–4.7 ms (A100 7B),
7.3 ms (H 70B TP8), 15.0 ms (A100 70B TP8).

Sources checked: TRT-LLM v0.8.0 docs/source/performance.md. TTFT is `gptSessionBenchmark --warm_up 1 --num_runs 5` with
OSL 1 on one max-throughput engine build (max_batch 64, max_input 2048), without `--enable_cuda_graph` (eager TRT
enqueue). The doc's command line uses batch 64 / 32 while its table says batch 1. Its footnote says the low-latency
numbers "will soon be updated to reflect real time latency with inflight-batching". NVIDIA publishes no per-component
host / framework time for GptSession, so the residual stays open; nothing is fitted.

## §123 GPU glue kernels in the op list (0.177)

**What was added.** The opt-in `Chip.gpu_glue` flag adds the GPU-runtime kernels in `ir.glue_ops`. Each one has its own
work (vector elements and the bytes it reads and writes). With `kernel_us` / `kernel_vec` it also pays the 0.176
per-kernel cost: latency floor plus launch gap. The trace column gives kernels per layer-forward in the public DeepSeek
trace (来源).

| kernel | rule | DeepSeek trace per layer-forward |
|---|---|---|
| activation quant cast | one per distinct quantised-activation GEMM input (attention in, q_b, o, FFN in). Down projections take the fused SwiGLU + cast output. The FFN input cast is skipped under a DeepEP dispatch, which casts in-kernel (来源: trace). | per_token_cast_to_fp8 2.95 |
| MLA q_a_layernorm | one per MLA layer with q_lora | layer_norm 2.98 |
| split-KV combine | decode attention; partial O / LSE fp32, 2 splits (假设, lower bound) | mla_combine 0.98 |
| router top-k | MoE layers | top2_sum_gate 0.96 |
| split-K reduce of the bf16 router GEMM | MoE decode | splitKreduce 0.96 |
| shared-expert SwiGLU + cast | MoE with shared experts | swiglu/cast 0.98 |
| collectives counted as kernels | EP dispatch / combine = 2 each (send + recv), others 1 | dispatch_ll / combine_ll 1.92 each |
| sampling passes | 4 per step, fp32 logits | penalty / topk / top_p / gatherTopK |

**Kernel count.** DeepSeek-V3 decode (EP128, batch 128) now has 29.05 kernels per layer (推导), up from 17.2; the trace
has 30.4. The remaining ≈ 1.4 per layer are generic torch elementwise / memcpy kernels with no attributable role and are
not modelled.

**Re-score** (speed ratio predicted / measured, geomean / mean abs error; §124):

| group | ck + mech (0.176 ck_fixed) | ck_fixed 0.177 | ck + lump 62.3 | default |
|---|---|---|---|---|
| v1 static | 1.23 / 32 % | 1.21 / 30 % | 1.19 / 29 % | 1.13 / 24 % |
| v1 TTFT | 1.40 / 41 % | 1.33 / 34 % | 1.36 / 37 % | 1.33 / 35 % |
| v1 max-load | 1.03 / 19 % | 1.00 / 18 % | 1.00 / 18 % | 0.92 / 20 % |
| MoE large-EP | 0.81 / 32 % | 0.80 / 31 % | 0.81 / 32 % | 0.97 / 8 % |
| Mixtral TP8 | 1.25 / 44 % | 1.21 / 42 % | 1.23 / 43 % | 1.14 / 33 % |
| MI300X lat | 3.33 / 233 % | 3.33 / 233 % | 1.40 / 40 % | 1.40 / 40 % |
| MI300X max-load | 2.32 / 132 % | 2.32 / 132 % | 2.18 / 118 % | 2.09 / 109 % |
| B200 NVFP4 | 1.11 / 13 % | 1.07 / 13 % | 1.05 / 12 % | 0.99 / 15 % |

**128-token TTFT residual.** On Hopper 7B the unexplained residual falls from 1.8–2.1 ms to 1.4–1.7 ms.

**Decision rule** (match or beat the lump on every group): **not met**.
- It beats or ties the lump on TTFT, max-load, MoE and Mixtral.
- It loses by 1 point on v1 static and B200.
- On MI300X it is not applied, because there is no ROCm trace.

So `catalog_kernel` and `default` are unchanged, and `catalog_kernel_fixed` (old: mech only → new: mech + glue) stays a
reported variant.

## §124 Ratio direction convention for the validation tables

Two conventions are in use:

| where | ratio | > 1 means |
|---|---|---|
| `scripts/ext_score_all.py` group tables (§83 onward, incl. §119, §122, §123) | **speed ratio**: throughput rows predicted / measured; latency rows (TTFT, static latency) measured / predicted. Geomean over the group, then mean abs error. | model optimistic (too fast) |
| `extval.summary` / `data/ext_validation.json` | **raw predicted / measured value** per metric | throughput: too fast; latency: too slow |

The two are reciprocal on latency rows. For example, 0.176 v1 TTFT for `catalog_kernel_fixed` is 1.40 in the §122
table and ≈ 0.72 in `ext_validation.json`.
- The 0.173 report (§119) labelled speed ratios "pred / meas".
- The 0.176 report called them "measured ÷ predicted".

Both labels were wrong: the values are speed ratios as defined above. The §119 and §122 headings are corrected.

## §125 Default = the per-layer mechanism; ROCm decode sourced; lump only as fallback (0.178)

**Decision** (user rule: a mechanism-justified default change takes priority over a 1-point score loss). The validation
default's flat per-layer lump (`TRACE_LAYER_US` 62.3 µs, 来源 the DeepSeek CUDA trace) is replaced by its mechanisms:
- the per-kernel latency floor plus the launch gap of the phase's runtime (§122);
- the GPU glue kernels (§123).

`default` = `rocm_variant(glue_variant(mech_variant("default_lump")))`. The ≤ 0.177 default stays as the named variant
`default_lump`; nothing else changes (attention schedule, memory runtime, ceilings, TBO). `catalog_kernel_fixed` gets
the same AMD treatment. `catalog_kernel` (no fixed cost) is still the pinned study baseline.

**Scope by vendor and phase:**

| rows | per-layer fixed cost | label |
|---|---|---|
| NVIDIA decode / prefill | per kernel 2.0 µs floor + graph 1.025 / eager 2.404 µs gap, glue kernels | 来源 CUDA trace (§122/§123) |
| AMD decode | per kernel `ROCM_KERNEL_US` = 5.8 ms / 1200 + 0.04 ms / 1970 = 4.85 µs, glue kernels | 来源 / 推导, below |
| AMD prefill | flat lump 62.3 µs per layer (`fallback_us`) | **假设** fallback: no public ROCm eager trace |

**ROCm source.** <https://github.com/ryanzhou/deepseek-v4-flash-mi300x/blob/main/DECODE-OPTIMIZATION-20260815.md> (raw: <https://raw.githubusercontent.com/ryanzhou/deepseek-v4-flash-mi300x/main/DECODE-OPTIMIZATION-20260815.md>; re-verified live 2026-10-16 in 0.179; the "1,970 kernels with only 0.04 ms intra-graph idle" sentence is on line 43–44 and "~1,200 launch-floor kernels (quant 430, …) — ~5.8 ms" on lines 191–193, under "Remaining headroom"): MI300X, vLLM ROCm + AITER,
DeepSeek V4 Flash, HIP-graph decode at M = 1.
- "1,970 kernels with only 0.04 ms intra-graph idle" → gap 0.020 µs per kernel.
- "~1,200 launch-floor kernels … ~5.8 ms" → floor 4.83 µs per kernel.

Caveats: it is a different model and a newer stack than the vLLM 0.11.2 Llama-3.3-70B MI300X rows, and the glue
inventory is still CUDA-trace derived. The lump fallback for AMD prefill (and for any vendor without a trace) should be
replaced when a public ROCm eager / prefill per-kernel trace becomes available. Apply it through `amd_kernel_us`, as for
decode.

**Re-score, old → new** (speed ratio predicted / measured, §124; geomean / mean abs error):

| group | n | default_lump (old default) | **default (0.178)** | catalog_kernel | catalog_kernel_fixed 0.177 → 0.178 |
|---|---|---|---|---|---|
| v1 static | 36 | 1.13 / 24 % | 1.15 / 25 % | 1.31 / 38 % | 1.21 / 30 % (=) |
| v1 TTFT | 20 | 1.33 / 35 % | 1.31 / 33 % | 1.75 / 76 % | 1.33 / 34 % (=) |
| v1 max-load | 44 | 0.92 / 20 % | 0.92 / 19 % | 1.12 / 21 % | 1.00 / 18 % (=) |
| MoE large-EP | 7 | 0.97 / 8 % | 0.96 / 7 % | 0.82 / 33 % | 0.80 / 31 % (=) |
| Mixtral TP8 | 27 | 1.14 / 33 % | 1.12 / 32 % | 1.34 / 47 % | 1.21 / 42 % (=) |
| MI300X lat | 16 | 1.40 / 40 % | 1.13 / 23 % | 3.33 / 233 % | 3.33 / 233 % → 1.13 / 23 % |
| MI300X max-load | 4 | 2.09 / 109 % | 2.02 / 102 % | 2.32 / 132 % | 2.32 / 132 % → 2.12 / 112 % |
| B200 NVFP4 | 11 | 0.99 / 15 % | 1.01 / 14 % | 1.26 / 26 % | 1.07 / 13 % (=) |

The new default is better or equal on 7 of 8 groups and loses 1 point on v1 static (noise, per the decision). MI300X
latency improves most: 40 % → 23 %, from the sourced ROCm per-kernel cost. `data/ext_validation.json` is
regenerated.

## §126 MI300X max-load: sourced mechanisms and the remaining residual (0.179)

**Source.** The four MI300X max-load rows come from the ROCm vLLM dev-docker README (pinned commit
`2026250877c4…`, vLLM 0.11.2 / ROCm 7.0.0, amd/Llama-3.1-70B-Instruct-FP8-KV). Re-read in 0.179.

**Fixes applied** (both are sourced and change the default):

1. **One wave at max_num_seqs.** The README table gives Num Prompts = Max Num Seqs per row: 3200 / 1500 / 2000 / 1500.
   - With `--max-num-batched-tokens 131072`, the run admits every prompt at once and decodes them together.
   - Rows now carry `max_seqs` / `num_prompts`. `predict` evaluates exactly that batch instead of the best power of two
     (old: 4096 / 2048, above the source's cap).
   - The `est` estimator's chunk changed 16384 → 131072 (the README's flag), and its cap now comes from the row.
2. **MI300X GEMM ceiling** (`GEMM_CEILING["MI300X"]`). Old: peak, because no figure had been found. New: the AMD MAF
   numbers from <https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html>
   (MI300X 750 W, ROCm 6.3, hipBLASLt):

   | format | measured | / peak | ceiling |
   |---|---|---|---|
   | FP8 | 1223.6 TFLOPS | 2614.9 | 0.468 |
   | BF16 | 708 | 1307.4 | 0.542 |
   | FP16 | 654 | 1307.4 | 0.500 |

   The cause is clock throttling to about 1.2 GHz under GEMM load, not the library. The blog's own table lists FP8 at
   1273 TFLOPS; the footnote measurement (1223.6) is used.

**Checked, not the cause** (all 来源):

| mechanism | source | measured | model | verdict |
|---|---|---|---|---|
| HBM bandwidth | BabelStream, arXiv 2510.27583 | 4.3 TB/s (81 % of 5.3) | ≈ 3.7 TB/s effective at the decode point | model already lower; can't explain an optimistic model |
| RCCL all-reduce | AMD rocm.blogs `mi300x-rccl-xgmi` (busbw 295 GB/s at 128 MB, 318 GB/s at 16 GB); AMD acceptance ≥ 304 GB/s | 295–318 GB/s busbw | ≈ 256 GB/s effective | model already lower |
| KV capacity | — | 3200 × 2176 tokens × 20 KiB/GPU (fp8 KV, 1 KV head per GPU at TP8) = 143 GB per GPU | — | fits 0.9 × 192 GB minus weights in the model |

The KV check may be marginal in reality: vLLM reserves activation memory for the 131072-token profiling pass, so
preemption is possible. That reserve has no public number, so it is not modelled (假设-only).

**Tokens/s definition.** `vllm bench throughput` prints both total and output tokens/s, and the README does not say
which one the table uses. The model predicts output tokens/s; if the table is total tokens/s, the residual grows by
(isl + osl) / osl, i.e. 1.06–2.0×.

**Score** (speed ratio predicted / measured, §124; gmean / mean abs error):

| group | 0.178 | 0.179 |
|---|---|---|
| MI300X max-load | 2.02 / 102 % | 1.76 / 76 % |
| MI300X lat | 1.13 / 23 % | 1.12 / 22 % |

All other groups are unchanged.

**Residual.** The model is still 1.67–2.00× optimistic per row: at 128/2048 it implies about 236 ms per decode step at
batch 3200, against about 113 ms modelled. No sourced mechanism was found for the rest. Candidates for when a source
appears:
- a per-step vLLM V1 scheduler/CPU cost at thousands of sequences;
- ROCm paged-attention efficiency with fp8 KV at large batch;
- KV preemption.

This remains documented as a residual, not fitted.

## §127 MI300X max-load: tokens/s definition, host overhead, ROCm paged attention (0.180)

**(1) Total or output tokens/s.**

The README table gives a single column, "Throughput (tokens/s)", for `vllm bench throughput`. The vLLM 0.11.2 script
(<https://raw.githubusercontent.com/vllm-project/vllm/v0.11.2/vllm/benchmarks/throughput.py>) does two things:
- it prints `Throughput: {req/s} requests/s, {total_num_tokens/elapsed} total tokens/s, {total_output_tokens/elapsed} output tokens/s`;
- `--output-json` writes `tokens_per_second` = **total** tokens/s.

AMD's MAD harness (`ROCm/MAD scripts/vllm/run_vllm.py`) records both `throughput_tot` and `throughput_gen`. The README
command has no `--output-json`, and none of its 31 versions (commit history at the pinned SHA) says which number the
column holds. **The source does not settle it.**

**Cross-length ratio test** (推导, using the 0.179 model):

| reading | predicted / measured per row | spread, sd(log) |
|---|---|---|
| output | 2.00, 1.79, 1.93, 1.67 | 0.068 |
| total | 2.12, 1.85, 2.41, 3.35 | 0.22 |

Under the total reading the 2048/2048 row diverges by about 2×. Read model-free: the measured 2048/2048 output rate would
be 3876 tok/s, against 12764 at 128/2048 with the same output length and batch size of the same order. Prefill is only
about a third of the work in the model.

**Output tokens/s is the interpretation kept** (the 0.179 behaviour). Rows carry `tok_def: "output"`. The score is
unchanged.

**(2) Host / scheduler per-step time.** The existing mechanism (`Scenario.host_step_us` / `host_seq_us`, §0.164) is
opt-in and is **not** applied to any validation row. Sourced magnitudes:
- vLLM PR #40459: scheduler microbenchmark ≈ 194 µs per loop at 256 requests and ≈ 396 µs at 512, i.e. ≈ 0.77 µs per
  sequence (H100 host).
- vLLM PR #46112: `update_from_output` 0.289 ms at 256 decode sequences, ≈ 1.1 µs per sequence (RTX 5090 host).

Together that is ≈ 1.9 µs per sequence, about 6 ms per step at 3200 sequences: ≈ 5 % of the modelled 113 ms step, against
a residual of ≈ 120 ms. Both sources use different CPUs and vLLM versions, so the mechanism stays without a default
value. This does not explain the residual.

**(3) ROCm paged attention with FP8 KV.** Sources found:
- vLLM ROCm attention-backend blog (2026-02-27): AITER FA gives 2.8–4.6× faster TPOT than legacy `ROCM_ATTN`.
- vLLM PR #12348 (custom paged attention v2): +10 % to +54 % end-to-end on MI300X.

These are end-to-end ratios between kernels, with no absolute bandwidth or efficiency. The 0.11.2 README names AITER
optimisations for Llama but not which attention backend was used.

The residual can be bounded: a pure bandwidth bound for attention is 75 GB of KV at 4.3 TB/s ≈ 17 ms per step. Closing
the ≈ 120 ms gap through attention alone would need it to run about 7× slower than that bound, which no source supports.
**Documented, not applied.**

**Residual unchanged:** 1.76 / 76 %. The remaining sourced-mechanism candidates are KV preemption (vLLM's activation
reserve for the 131072-token profiling pass is unpublished) and absolute ROCm attention-kernel throughput.

**Re-score:** identical to 0.179 for every group. No default change.

## §128 Per-row residuals for v1 TTFT and Mixtral TP8; packing prompts into prefill iterations (0.181)

**Mixtral TP8, per row** (TRT-LLM 0.17, continuous-batching proxy, predicted / measured).

The ≤ 0.180 proxy put one prompt per chunk iteration, so a short prompt (isl ≪ chunk budget C) capped the running batch
at osl. The 128/128 rows therefore ran at batch 128 and came out 0.39–0.62, the only too-slow rows, which made the group
spread look larger than it was.

TensorRT-LLM defines the budget across requests:
- "`max_num_tokens` defines the maximum number of batched input tokens after padding is removed in each batch … the
  tokens from different sequences are packed together" (docs/source/features/paged-attention-ifb-scheduler.md, 来源).

**Mechanism.** `_cb_at` now packs q = ⌊C / isl⌋ prompts per prefill iteration (prefill evaluated at batch q), so each
request needs 1/q prefill iterations. Prompts longer than C are unchanged. With packing, the running-sequence cap binds:
`predict_cb` now takes the source's engine cap by default (`SOURCE_ESTIMATOR`). For TRT-LLM that is max_batch_size 2048
(the build default, previously recorded as "not binding" 假设; it now binds at 128/128). `PACK_PROMPTS[variant] = False`
restores the ≤ 0.180 rule.

| row | old | new |
|---|---|---|
| H200 Mixtral 128/128 | 0.45 | 1.32 |
| H100 Mixtral 128/128 | 0.39 | 1.37 |
| A100 Mixtral 128/128 | 0.62 | 1.48 |

After the fix the Mixtral residual is uniformly optimistic (gmean per GPU: H200 1.24, H100 1.28, A100 1.36). It does not
scale with input length (128 → 20000) or output length, which points to a per-token decode-step factor rather than
prefill.

| mechanism | source | why not applied |
|---|---|---|
| expert imbalance | — | TP8 shards every expert over all ranks: no rank-level imbalance |
| grouped-GEMM efficiency | no sourced figure for TRT-LLM 0.17 Mixtral MoE kernels by M | — |

Documented, not applied.

**v1 TTFT, per row** (speed ratio measured / predicted).

| slice | ratio |
|---|---|
| TP1 7B, isl 2048 | 0.91–1.13 (≈ right) |
| TP1 7B, isl 128 | 1.26–1.40 |
| TP8 70B, isl 128 | 1.47–1.67 |
| TP8 70B, isl 2048 | 1.17–1.51 |
| A100-40GB 70B, isl 512 (Databricks 2023) | 1.64–1.93 |

The error scales with **TP** and is largest at short prompts. Example, H100 70B TP8 isl 2048: measured 84.4 ms against
56.0 modelled. The model's 160 all-reduces of 33.5 MB take 20.9 ms; NCCL busbw ≈ 235–240 GB/s at 32 MiB would give
≈ 43 ms, about +22 ms, most of the gap.

The only 32/64 MiB H100 tables found are multi-node runs:
- nccl-tests issue #309 (2 × HGX H100, NCCL 2.26);
- crusoecloud/crusoe-ml-rke2 (2 nodes).

No single-node table was found. TRT-LLM 0.8 can also use its own custom all-reduce plugin rather than NCCL, depending on
build flags the rows do not state. So the all-reduce-at-prefill-message-size mechanism is **documented, not applied**,
until a single-node NCCL (or TRT-LLM custom all-reduce) table at the TRT-LLM 0.8 software level is found. The short-prompt
residual on TP1 (≈ 1.3 at isl 128) is the §122 framework residual.

**Re-score** (speed ratio, §124; gmean / mean abs error):

| group | 0.180 | 0.181 |
|---|---|---|
| v1 static | 1.15 / 25 % | 1.15 / 25 % |
| v1 TTFT | 1.31 / 33 % | 1.31 / 33 % |
| v1 max-load | 0.92 / 19 % | 0.99 / 21 % |
| MoE large-EP | 0.96 / 7 % | 0.96 / 7 % |
| Mixtral TP8 | 1.12 / 32 % | 1.29 / 30 % |
| MI300X lat | 1.12 / 22 % | 1.12 / 22 % |
| MI300X max-load | 1.76 / 76 % | 1.73 / 73 % |
| B200 NVFP4 | 1.01 / 14 % | 1.06 / 17 % |

The Mixtral gmean moves away from 1 because the too-slow 128/128 rows no longer offset the too-fast ones; its error falls
32 → 30 %. v1 max-load (19 → 21 %) and B200 (14 → 17 %) get slightly worse for the same reason: rows where the one-prompt
cap had hidden optimism. This is a mechanism-justified default change.

## §129 Microsoft MAI models (not added) and TRT-LLM 0.8 TP all-reduce (0.182)

### MAI-Base-2 (2026-10-15)

**Public sources:**
- the PDF <https://microsoft.ai/pdf/Frontier-Pre-training-at-Microsoft-AI.pdf>, page 2 table;
- the blog <https://microsoft.ai/news/frontier-pre-training-at-microsoft-ai/>.

| | MAI-Base-1 | MAI-Base-2 |
|---|---|---|
| active / total | 35B / 1T | 80B / 2.8T |
| layers | 78 | 102 |
| hidden | 6656 | 8704 |
| MoE layout | alternating dense / MoE | MoE in every layer |
| dense FFN | 13312 | — |
| MoE down-projection (latent) dimension | 3072 | 4096 |
| expert FFN | 10240 | 8704 |
| routing (shared + top-k / routed) | 0 + 8 / 512 | 1 + 7 / 512; the shared expert is not latent-projected |
| attention KV / Q heads: global (every 6th layer) | 8 / 80 | 2 / 96 |
| attention KV / Q heads: local | 8 / 80 | 16 / 96 |
| embeddings | tied | untied |
| training FP8 | — | MXFP8 |
| vision | — | native encoder-free VLM |

Also stated: trained on 50T tokens on 16k GB200 GPUs (MFU 21 %).

**Not stated:** head dimension, vocabulary, local window, expert gating (SwiGLU or not), norms, RoPE.

**Totals do not reconcile.** 512 × 102 SwiGLU experts of 4096 × 8704 is 5.6T; non-gated it is 3.7T; neither is 2.8T. The
active count does not close either.

The catalog only builds specs from released configs (`from_release`). Its proxy label marks a *released* config with
unmodelled parts, not a closed model with guessed dimensions. Post-training is ongoing and the weights are licensed only
to partners via Foundry. **Not added.**

### MAI-Code-1.1-Flash (2026-10-07)

**What is public:**
- **Model card**, Foundry catalog `MAI-Code-1.1-Flash-On-Device` (<https://ai.azure.com/catalog/models/MAI-Code-1.1-Flash-On-Device>, version 2):
  - MoE with 137.9B total and 6.8B active parameters;
  - 256K context, 128K output;
  - 53 GB, mixed-precision ≈ 3.3 bits per weight;
  - DFlash2 sliding-window speculative decoding with a 700M drafter;
  - 75.5 GB peak memory at 256K.
- **Distribution:** downloaded as a GGUF from the anonymous Azure ML registry `azureml`. The VS Code agentHost PR
  microsoft/vscode#341325 (merged 2026-10-14) does this through the registry's model-registry resolve API and runs it with
  a user-supplied `llama-server`.
- **llama.cpp architecture:** `maion-coder`, llama.cpp PR #29778 (merged; in v0.7.0). It has:
  - periodic MoE layers (`interleave_moe_layer_step`);
  - a latent MoE (`moe_latent_size`, latent down/up projections around the routed experts);
  - sliding-window attention with a per-layer pattern;
  - per-layer expert FFN sizes;
  - NEOX RoPE.

  This matches MAI-Base-1's layout, but the PR does not name MAI.

**Not public as text:** the hyperparameter values. They exist only in the GGUF header behind the registry's signed
(SAS) download, reached through a client API that requires a specific User-Agent. There is no HF repo, config.json, ONNX
release or Foundry Local manifest with values. microsoft/MAI-Code holds only a README. The GGUF header was not fetched
(a client-impersonating route, not a public document).

**Not added.** Re-check when a config or GGUF metadata is published as a document.

### TP all-reduce for v1 TTFT (§128 follow-up)

**TRT-LLM v0.8.0 source (来源):**
- `PluginConfig.use_custom_all_reduce = True` by default (`tensorrt_llm/plugin/plugin.py`).
- `AllreducePlugin::selectImplementation`: a message above `getMaxRequiredWorkspaceSize` falls back to NCCL ring. That
  limit is 8 MB at world size > 2 and 16 MB at ≤ 2 (`customAllReduceUtils.h`).
- At TP8: ONESHOT below 500 kB, TWOSHOT otherwise (`allreducePlugin.cpp`).

So for Llama-2-70B at TP8, the per-layer all-reduce (isl × 8192 × 2 B) runs as:

| isl | message | path |
|---|---|---|
| 128 | 2.1 MB | TWOSHOT custom kernel |
| 512 | 8.4 MB | NCCL (just above 8 MB) |
| 2048 | 33.5 MB | NCCL |

**NCCL at 32 MiB, single node.** Still not found as a table:
- nccl-tests #309 and crusoe-ml-rke2 are 2-node runs (234.8 / 240.0 GB/s busbw at 32 MiB);
- nccl-tests #212 and nccl #1374 give only the large-message single-node figure (≈ 480 GB/s busbw with NVLS, ≈ 370 without);
- DGX BasePOD's table is a non-allreduce test.

No published TWOSHOT throughput for TRT-LLM 0.8 was found either. **Not applied.** The paths are recorded so that a
sourced curve can be dropped into `schedule.collective_seconds`.

**Re-score:** identical to 0.181 for every group. No default change.

## §130 Discrete DRAM interface sizing and the integer-floor fix (0.183)

`sizing_report` gains `dram_units`: the package (LPDDR) / stack (HBM) count of the current memory, everything else in the part fixed (rate, width, height, die density, `mem_eff`). Mechanism: bus_bits = n × unit_width, so bandwidth and capacity both scale with n; a lost fit is a failure, exactly as in §103. Area per unit = unit_width_bits × the DRAM PHY density of the current memory type (budget.phy_area, §96 presets: HBM 14 mm²/1024 pins 二手/假设, LPDDR 0.03 mm²/pin 假设); no density → no mm². Package / board / DRAM cost are not priced.

`dram_bw` (continuous, via mem_eff) is kept: it answers "how much delivered bandwidth is spare", `dram_units` answers "can a package / stack be dropped". Joint shrink skips `dram_units` (`JOINT_SKIP`) because both act on the same wire and dram_bw is an absolute GB/s defined against the current bus.

Default change: default sizing resources old {dequant, lanes, sram, cores, dram_bw} → new + dram_units. Engine outputs and fingerprints unchanged.

Bug fixed (adversarial, found while adding an integer resource at n = 1): the halving loop for integer resources rounded 0.5 back to 1 and never terminated when the current value was 1 (engines = 1 or a single package). It now stops at the integer floor. Tests: tests/test_core_183.py.

## §131 Sizing UI check, sizing fuzzer, sourced LPDDR PHY density (0.184)

UI: the sizing panel renders rows generically from the API (label, unit, `mm2_saved` null → "—"). Two defects were found in a real-browser run and fixed: integer resources printed as "4.00 个", and a resource already at 1 showed "≤ 100% · 到搜索下限" (looks like unlimited headroom). Rows now carry `integer` and `unit_floor`.

Fuzzer (scripts/fuzz_sizing.py): every sizing/headroom entry point (sizing per-resource, joint with unpriced last/free, upgrade_path, insight_report) over random small chips (engines 1–16, rows/cols down to 1, lanes/dequant down to 1, SRAM 0.25 MiB, 1–4 LPDDR packages or 1–2 HBM stacks) with a SIGALRM wall-clock limit per call and report invariants. 40 cases: no hang, no exception, no invariant break. The 0.183 integer-floor fix holds.

LPDDR PHY density, old→new: 0.03 mm²/DQ 「假设」 → 0.695/24 = 0.02896 「推导/二手」. Source: Samsung SF2 LPDDR6 PHY, ISSCC 2026 paper 37.3, as reported by SemiAnalysis (2026-04-15): 0.695 mm², 2.32 mm shoreline, 55.3 Gb/s/mm² (the article's unit; ×0.695 = 38.4, equal to its own x24 @ 14.4 Gb/s payload GB/s figure). The x24 width is inferred from that identity. The area includes the channel's CA/clock lanes. Limits: 2 nm and LPDDR6; an LPDDR5X PHY on an older node is likely larger (no public number). The HBM density is unchanged: the Maia 200 floorplan shows 3 HBM3E PHYs per long edge, but no area figure is public.

## §132 HBM PHY density from a vendor disclosure; headless UI smoke (0.185)

HBM PHY, old→new: 14/1024 = 0.0137 mm²/DQ 「二手/假设」 → 32/2048 = 0.0156 「推导/二手」. Source: Samsung, Hot Chips 2026 (S. Han), as reported by Tom's Hardware (text checked): "On an 11 × 12.8mm HBM4 base die, the conventional PHY occupies more than 8 × 4mm … Because the matching interface on the XPU shrinks too …". That covers 2048 DQ (JESD270-4) on SF4 and is the memory-side PHY. The host-side PHY is taken as equal (推导); "more than" makes it a lower bound. The old die-shot estimate (0.0137) sits within 14 % of it.

LPDDR5X on N5/N4/N3: no public area. Synopsys and Cadence product pages give only rates (8533, and 9600 overclocked on 3 nm). The 0.184 LPDDR6 SF2 figure is kept.

UI: scripts/ui_headless_smoke.py (opt-in). Every control change must issue a fresh API POST and bring the KPI cards back. It fails on page errors, console errors and HTTP ≥ 400/500. First run: 31 steps, 0 errors.

## §133 Directional UI smoke; DRAM energy by memory kind (0.186)

UI smoke: directional assertions (batch↑ → TPOT↑, TP2 → TPOT↓, fewer packages → bandwidth↓ and capacity↓, restore → bandwidth back). Each step waits for the /api/eval answer. The one alarm (capacity unchanged after 8→6 packages) was a harness timing artefact: the memory change answers /api/memory first, and the KPIs re-render only after /api/eval. No UI defect.

Why this item: energy per token had a single pJ/bit for every DRAM type, so the LPDDR ↔ HBM choice (one of the largest chip-design decisions in the tool) moved energy only through bytes, never through the per-bit cost. Mechanism: E_dram = bytes × 8 × pJ_bit(kind). kind = HBM stack vs any other package (mem_catalog); a field left unset falls back to the generic value, so every old table is unchanged.

Numbers (preset tpuv4i_7nm only), old→new: generic 3.17 (推导, TPUv4i 6.3× on Horowitz) for every kind → HBM 7.0 and LPDDR 10.0 「二手」. Both come from the same table (PIM-AI, arXiv 2411.17309, Table 1: DGX-H100 HBM 7, Snapdragon 8 Gen 3 LPDDR5X 10, A17 Pro LPDDR5 20, each including the SoC- and memory-side interfaces). They are the authors' derived profile parameters, not measurements, hence 二手. Cross-check: O'Connor et al. MICRO 2017 measure HBM2 at ≈ 3.97 pJ/bit memory side only, so ×2 for both sides lands at ≈ 7.9, consistent with 7. The generic pJ_bit_dram of the preset is left at 3.17 for unknown ids.

## §134 Primary-sourced DRAM energy; energy exposure and constraint in sizing (0.187)

DRAM pJ/bit cross-check (tpuv4i_7nm preset). The primary sources contradict the 0.186 PIM-AI values (HBM 7 / LPDDR 10, both-interface ×2 convention, the authors' derived profile), old→new:
- HBM: 7.0 → 3.44 「推导」. SK hynix, IEDM 2023 paper 15.6 handout: "power efficiency of HBM2E is about 4.3 pJ/bit, that of HBM3E is reduced by 20%". Cross-check: NVIDIA O'Connor et al., MICRO 2017, HBM2 ≈ 3.97 pJ/bit.
- LPDDR: 10.0 → 4.0 「推导」. NVIDIA Grace product page and Performance Tuning Guide: "LPDDR5X memory subsystem provides up to 500 GB/s of bandwidth in only about 16W" → 16 / (500e9 × 8) = 4.0 pJ/bit at peak.

Both values are memory side; the SoC-side PHY (HBM3 ≈ 0.29 pJ/bit, Chae JSSC 2023) is excluded for both. The ratio is 1.16, not 1.43. Not found: an LPDDR5X device pJ/bit from IDD data in a public datasheet (JEDEC 2024 slides give only normalised curves), and any pJ/bit for the 2 nm LPDDR6 PHY. horowitz_45nm gets no per-kind values (no era-matched source).

Sizing energy (why this item: SRAM / dequant shrinks can hold speed while moving bytes to DRAM, so the headroom answer ignored energy). Each sizing row now reports, per model at min_ok, DRAM / SRAM bytes per unit and (with an energy table) J per unit as ratios to the current chip. `energy_tie=true` adds J/unit ≤ current × (1 + TIE) to the feasibility contract; it requires an energy table. Example (tpuv4i_7nm): wan2.1-1.3b SRAM goes 25 % → 50 % with the energy constraint (DRAM bytes ×1.64 → ×1.19), limited by J_tok. Invariant (tests/test_core_187): the constraint never loosens min_ok. LLM energy is decode-only (as in insight_report).

## §135 Closing the partial energy: DRAM PHY line, static power, prefill energy (0.188)

- Chip-side DRAM PHY: new `pJ_bit_phy_hbm` / `pJ_bit_phy_lpddr`, priced on the same DRAM bytes as a separate line `dram_phy`, so DRAM = memory side + SoC side. tpuv4i_7nm: HBM 0.29 「来源」 (Chae et al., Samsung, ISSCC 2023 28.4 / JSSC 2023, 4 nm HBM3 interface; cross-check Intel VLSID 2025 host PHY 0.5). LPDDR 1.3 「二手」 (SOCC 2025 LPDDR4X PHY paper: controller PHY 0.65 pJ/b, and "LPDDR5X … halves the energy efficiency" → ×2). No public LPDDR5X/6 controller-PHY pJ/bit was found, including the 2 nm LPDDR6 PHY.
- Static power: `idle_W := idle_frac_tdp × tdp_W` when idle_W is unset. The existing mechanism is unchanged: idle W × card·s per unit, which equals static W × step time / tokens. tpuv4i_7nm idle_frac_tdp = 71.8/700 = 0.103 「二手」 (measured H100 SXM idle without a CUDA context; 121.7 W with one). The fraction is a measured whole-card floor, not a leakage split. `tdp_W` is a product choice. The API takes it from `budget.tdp_W_card` when the energy table has none. Without it, `missing = [tdp_W]` and the result stays partial.
- Prefill energy: insight/sizing `_point` reports `J_prefill` (J per prompt token). The insight detail shows Δ prefill energy. Sizing traffic reports the prefill J ratio, and `energy_tie` on LLM constrains both decode J_tok and J_prefill. J_partial now also considers prefill.
- Example: default chip, LPDDR, tpuv4i_7nm, TDP 175 W. Decode J/tok goes from 0.517 (partial) to 2.15 (complete; static dominates on this memory-bound decode). Prefill J/prompt-tok goes from 0.0038 to 0.0068.

## §136 Provenance tags, static-energy visibility, TDP power cap (0.189)

- Tags: `core/provenance.valid_tag` accepts a base label (来源/推导/二手/假设) or a compound of distinct base labels joined by "/". The preset label tests (149/152/153) use it. tpuv4i_7nm `pJ_bit_phy_lpddr` 1.3 is relabelled 二手 → 推导/二手.
- Static energy: the report gives `static_share`, `J_dynamic_per_unit`, and `idle_source_note`. When the fraction is the preset value, the note states that the H100 whole-card idle (二手) is applied to this NPU and is not a leakage measurement of this chip. If static share > 50 %, the report adds `static_warning`, shown in red in the UI. Example default LPDDR chip at TDP 175 W: static share is 68 %. idle_frac_tdp is not made a sizing/sweep axis. It moves no speed metric, and J is affine in it (J = J_dynamic + frac·TDP·card·s), so the two reported terms already give the sweep. No default TDP (AGENTS §73).
- Power cap (opt-in, `/api/eval` `power_cap: true`, UI checkbox): needs tdp_W. P_dyn = dynamic J per card per window / window, and P_static = idle W. If P_dyn + P_static > TDP, the window stretches by P_dyn / (TDP − P_static), so the capped power equals TDP exactly. Throughput is ×1/slowdown, and static J grows by the slowdown. Energy per action is held constant (no DVFS V² saving), 「假设」, conservative. If P_static ≥ TDP, the result is infeasible. This is report-level: evaluate/latency are not changed.

## §137 UI smoke for energy controls; CLI power cap (0.190)

- `scripts/ui_headless_smoke.py` (opt-in) now drives the energy panel with the tpuv4i_7nm preset. It checks the PHY / idle_frac_tdp / tdp_W inputs and that the preset fills PHY values. At TDP 175 W it checks the DRAM PHY row, that the static source names H100, that the share line is shown, and that the warning appears iff the share is > 50 %. With idle_frac_tdp 0.001 the warning is gone, the note says user-entered, and total J falls. With power_cap on, the cap does not bind at 175 W; at 2 W it binds and the throughput factor is < 1. With power_cap off, the cap line is gone. Result: 0 errors.
- Found and fixed: the UI sends the preset-filled idle_frac_tdp rounded (0.1026), so the report called it "用户填写" and dropped the H100 citation. Preset match is now within 0.5 %. The static share was displayed as "50%（> 50%）" at 50.05 %; it is now shown to 0.1 %.
- CLI: `eval --power-cap` (needs --tdp-W, else the error line). It prints the static source note, the warning, and the power-cap line.

## §138 Decision models and RWKV-7 (0.191)

- Decision readout (`serving.labels` = N, 1..1024, phase prefill). Source: startlux_decision/model.py. The output-head rows for the N labels are index_selected (fp32), and logits = h_last · W_labelᵀ. use_cache=False. Modelled as prefill of the prompt + final norm + a GEMM with n=N + softmax over N; no lm_head over the vocabulary, no decode, no KV write when ctx = 0. Metrics: latency_ms = step, decisions_per_s = batch / step. Multiple questions sharing a prefix (the chunked KV path) are not modelled; each decision is its own forward.
- StartLux-Decision v1 / v1.1: configs read live from HF (coverage 完整). The 27B has full attention every 4th layer; the 35B-A3B is MoE. v1.1 Instant sets `jev_prompt_attention: bidirectional`: full-attention layers are bidirectional inside the prompt (core causal=False; linear layers stay causal). v1.1 AutoThink can emit up to 4096 thought tokens before the readout (596 on average at gate 0.9, 来源 model card). That is a decode run, so model it as a separate decode phase; the readout alone is shown here.
- Cross-check, H200 (vLLM BF16, 1 request, 来源 model card): 9B 256/1024/4096/8192 prompt → 30/39/109/214 ms. Simulation 5.2/21.6/84/126 ms. The gap at short prompts is fixed serving overhead, which is not modelled. This is a comparison only; nothing was fitted.
- Microsoft-Decision-1: closed. The announcement names a Qwen3.5-9B base, so the 9B config is used as a proxy (coverage proxy, 推导).
- Jev (TypeSafe AI): closed, with no published architecture, so it is not added.
- ReaDy-Qwen3.8-27B draft: not added. It does not fit the spec-decoding draft catalog naturally.
- RWKV-7 (Peng et al. 2025, "Goose", eq. for the generalized delta rule; release rwkv7 configs):
  - Per layer: r/k/v/o C×C, plus the LoRAs w, a, g and v (v not on layer 0), and a non-gated FFN with d_ff = 4C.
  - State: H×64×64 fp32 plus 2 token-shift vectors of C fp32. Decode per token: 38·C + 9·H·N² vector ops, 4C transcendentals, and read+write of the state, independent of ctx (an adversarial invariant, tested at 4k/128k/1M).
  - Prefill uses chunk-64 WY form GEMMs, linear in the prompt.
  - State residency uses the existing SRAM order: weights first, then KV and state. State-first could be better, because state is read and written every token (open).

## §139 Prefill scaling on the wide gang; state-first SRAM; AutoThink thought (0.192)

- **Prefill scaling (StartLux / Qwen3.5-9B, H200).** 4096 → 8192 tokens cost 1.50× in the simulation; vLLM measured ≈ 1.96×. The compute-efficiency path was checked op by op, and every GEMM's `gemm_cost` is linear in M. The cause is tile padding at 4096:
  - The "wide" schedule gangs 132 SMs into a 32 × 8448 array, so an output dim < 8448 is padded to 8448. mlp.down (N = 4096), lin_z / lin_out / o (N = 4096) and q / lin_qkv (N = 8192 at M = 4096) run at 48–97 % fill. mlp.down at M = 4096 takes the same cycles as gate_up with twice the FLOPs.
  - At M = 8192 the 0.173 orientation swap (Cᵀ = Wᵀ·Xᵀ, when N < M) puts M = 8192 on the columns, giving 97 % fill. The per-token time drops, and it oscillates (12288 → 2 tiles, 73 %). A100 (gang 3456) pads N = 4096 to 6912.
  - Neither the efficiency curve nor double counting is involved.
- **Mechanism fix: `mapping.WIDE_WAVES`, opt-in.** Each engine takes R × cols output tiles in ⌈tiles / E⌉ waves, the cta wave quantisation without tuned tile shapes. Results:
  - 9B 4096 → 8192: 2.03×. H200 9B read-out 256 / 1024 / 4096 / 8192: 5.2 / 21.6 / 84.0 / 125.6 → 4.1 / 15.0 / 60.2 / 122.4 ms (measured 30 / 39 / 109 / 214).
  - The marginal-cost invariant (`tests/test_core_192.py`, 5 chips × 5 models, prompts 4k–32k, 3 % tolerance) holds with it. Without it, 4 / 25 cases fail at that tolerance; the 4096 → 8192 step is the largest violation.
- **Why it is not the default.** §119 kept the padding as a stand-in for unmodelled kernel efficiency. Removing it moves every GPU group further from measurement (default variant): v1 static 1.15 → 1.18, TTFT 1.31 → 1.44, max-load 0.99 → 1.01, Mixtral TP8 1.29 → 1.31, MI300X max-load 1.73 → 1.81, B200 1.06 → 1.13. It also breaks about 20 calibrated bands and schedule invariants. The shape is right but the level is wrong. Making it the default needs a sourced kernel-efficiency term to replace the padding: an open user decision.
- **State-first SRAM pinning (`memplan.STATE_FIRST`, default on).** Order is by DRAM bytes saved per step per byte of SRAM: recurrent state is read and written every step (2 B / B), a weight byte is read once per pass (1 B / B), a KV byte once. So the order is state → hot weights → expert weights → KV. With ≥ 2 microbatches the weights tie with the state. KV and state still share one miss fraction.
  - Default chip, ctx 4096, batch 1 / 32 / 256: RWKV-7 7.2B 72.517 / 83.779 / 165.223 → 72.335 / 83.475 / 165.047 ms; 1.5B 14.364 / 18.610 / 49.354 → 14.297 / 18.304 / 49.179 ms.
  - Hybrids gain 0.03–0.4 % tok/s (qwen3-next-80b-a3b 26.12 → 26.22). Fingerprints re-baselined; the old values are reproduced with `STATE_FIRST=False`.
- **AutoThink thought.** `serving.think` adds a thought decode of `think_tokens` (default 596, the card mean at gate 0.9 「二手」; ≤ 4096 from the card) after the prompt prefill. It is integrated at 4 context midpoints from prompt to prompt + T.
  - Decision latency = read-out + thought (the closing read-out ≈ one decode step).
  - 9B, prompt 1024, 3 labels: default chip 153.8 + 49 660 ms = 49.8 s (0.020 decisions/s); H200 21.6 + 2 828 ms = 2.85 s.
  - CLI `--think [N]`.

## §140 GPU GEMMs: CTA-tile waves × achieved-peak fraction replace the wide padding (0.193)

- **Decision (user, 0.192 review):** the mechanism wins. The wide-gang padding (§119, §139) was a stand-in, and it is replaced by sourced terms.
- **Mechanism.**
  - Reference GPUs run GEMMs on `instance_sched` "cta" (mapping._cta, 0.92). This uses the NVIDIA "Matrix Multiplication Background" thread-block tile set (256×128 … 64×64) over SM count, with tile and wave quantisation (⌈tiles / SMs⌉ waves, tail wave full length), plus split-K.
  - Each tile also pays its prologue and C writeback at the SM's DRAM share (tile_io "dram", §81).
  - Split-K extends past 16 up to 512 as Stream-K (Osama et al., PPoPP 2023), bounded by tiles·s ≤ 2E and K/s ≥ 64. This keeps tiny-output long-K GEMMs (hc mixers 1 × 28672 × 24) inside the roofline invariant.
- **Achieved-peak fraction.** `extval.GEMM_CEILING` 「来源」 is applied multiplicatively on the main loop (`Chip.gemm_ceil_mul`) instead of as a floor:
  - A100 cuBLAS fp16 263.4 TF at 16384³.
  - Hopper cuBLAS bf16 744 at 8192³.
  - B200 cuBLAS bf16 1551 at 8192³ and fp4 6220.
  - MI300X hipBLASLt fp8 1223.6 / bf16 708 / fp16 654 at the AMD blog shapes.
  - Hopper fp8 from DeepGEMM.
  - Each figure is divided by the cta wave efficiency of its own measured shape (`CEIL_SHAPE`: net 0.845 / 0.758 / 0.692 / 0.468–0.542). Hopper fp8 and the DeepGEMM per-shape rows stay gross 「假设」, because their M is unstated.
- **GEMM-level check** (bf16 cuBLAS, sources fast.cu README and pyptx.dev; not fitted):

  | GEMM | Measured (TF) | Old wide padding (TF) | cta × fraction (TF) |
  |---|---|---|---|
  | H100 4096³ | 716 | 480 | 726 |
  | H100 8192³ | 795 | 744 | 744 |
  | B200 4096³ | 1532 | 486 | 1536 |
  | B200 8192³ | 1610 | 973 | 1551 |

  - B200 2048³ comes out at 1534 TF against 1006 measured: the small-shape launch and prologue cost is under-modelled (open).
- **Re-score** (geo mean / MAE, default_0192 → default):

  | group | old | new |
  |---|---|---|
  | v1 static | 1.150 / 25.0 % | 1.118 / 22.7 % |
  | v1 TTFT | 1.307 / 32.8 % | 1.322 / 32.2 % |
  | v1 max-load | 0.985 / 21.5 % | 0.955 / 19.1 % |
  | MoE large-EP | 0.962 / 7.0 % | 0.943 / 7.7 % |
  | Mixtral TP8 | 1.291 / 29.6 % | 1.213 / 22.1 % |
  | MI300X lat | 1.123 / 22.4 % | 1.110 / 21.0 % |
  | MI300X max-load | 1.726 / 72.6 % | 1.668 / 66.8 % |
  | B200 NVFP4 | 1.061 / 17.5 % | 1.065 / 17.9 % |

  - MAE is better in 6 of 8 groups. MoE (+0.7 pp, 7 rows) and B200 (+0.4 pp) are slightly worse. Without tile I/O, cta × fraction alone gives TTFT 1.42, and WIDE_WAVES alone gives 1.44. Tile I/O is what makes up the difference.
  - Flipped as "equal or better" overall; the two marginal groups are recorded here.
- **Defaults and legacy.**
  - `mapping.WIDE_WAVES` is on, so abstract wide chips get wave-quantised tiles.
  - `Chip.wide_pad` and the extval variants without `ceil_mul` keep the ≤ 0.192 chip, so historical variant bands are unchanged.
  - `evaluate.LEGACY_0192` (WIDE_WAVES off, reference GPUs wide, guide split-K) pins the queue / worked-example tests (pin_legacy) and reproduces the 0.192 fingerprints exactly.
- **Effect on the 9B example.** H200 raw chip (no ceilings), 9B read-out 256 / 1024 / 4096 / 8192: 5.2 / 21.6 / 84.0 / 125.6 → 4.3 / 16.8 / 67.8 / 136.9 ms. 4096 → 8192 goes from 1.50× to 2.02×; the vLLM measurement is 1.96×.
- The prefill marginal-cost invariant (`tests/test_core_192.py`) now runs on the default.

## §141 Small-GEMM fixed cost, net ceilings, regression diagnosis (0.194)

- **Small-GEMM fixed cost.** No new term is needed.
  - Every GPU GEMM in the default already pays the per-kernel cost (§136/§176: 2 µs floor + 1.03 µs graph gap) and the tile prologue / writeback (tile_io "dram", §81). The 0.193 GEMM-level check ran `_op_seconds` bare and omitted both.
  - Pipeline fill (one DRAM latency per tile wave; GH100 658.7 cycles ≈ 0.4 µs, Dissecting Blackwell arXiv 2507.10789) is < 3 % at 2048³. B200 has no published figure. It is not added.
- **Net ceiling (`extval.NET_FULL`).** The main-loop fraction is solved (bisection) so that the model, with tile I/O and kernel_us, reproduces each GEMM_CEILING figure at its own measured shape. ≤ 0.193 only divided out the waves, so tile I/O and kernel cost were charged twice.
- **Hopper ceiling merge (`FMT_CEIL_REPLACE` = False).** The variant `fmt_ceiling` (fp8 / attention) used to replace the GEMM_CEILING entries, so Hopper bf16 GEMMs ran at datasheet peak. They are now merged.
- **GEMM check** (cuBLAS bf16, TF; none of these points was used for matching):

  | GEMM | Measured | 0.193 (full path) | 0.194 |
  |---|---|---|---|
  | H100 4096³ | 716 (fast.cu) | 834 (no bf16 ceiling) | 695 |
  | H100 8192³ | 795 (fast.cu) / 744 (ceiling source) | 903 | 744 |
  | B200 2048³ | 1006 (pyptx) | 997 | 1034 |
  | B200 4096³ | 1532 | 1375 | 1446 |
  | B200 8192³ | 1610 | 1470 | 1551 |

- **MoE large-EP regression (0.192 → 0.193), per row.**
  - Only the decode rows moved: sglang 1.015 → 0.943, profile-decode 1.029 → 0.968. Prefill is unchanged (DeepGEMM shape ceilings).
  - The ops that moved are many-tile small GEMMs under tile_io "dram": the MLA absorbed bmm kv_b.uk 256×128×512 ×128 heads 5.5 → 57 µs, plus expert gate_up / down, q_b and o at +20–25 µs.
  - "dram" assumes one CTA per SM with serial I/O. The sm90 persistent ping-pong kernels (CUTLASS efficient_gemm.md) overlap it. With persistent I/O on Hopper / Blackwell (`default_tioauto`), MoE improves to 6.66 %, but v1 TTFT goes to 1.38 / 38 %, Mixtral to 28 %, and B200 2048³ to 1178 TF against 1006. Occupancy > 1 CTA per SM is the likely middle ground, with no source yet. Kept "dram"; this is not a bug.
- **B200 NVFP4 regression.** The prefill-heavy short-osl rows got faster once the padding went (128/128 1.342 → 1.429, 2048/128 1.251 → 1.301, then 1.651 / 1.496 with NET_FULL). The padding had been compensating for costs that are still missing on B200 (§140). This is not a bug, and B200 goes 17.85 → 22.82 %.
- Tests: `tests/test_core_194.py`.

## §142 CTA occupancy (opt-in); TRT-LLM NVFP4 kernel rates on B200 (0.195)

- **Occupancy (`tile_io` "occupancy", variant `default_occ`).** k = resident CTAs per SM is the minimum of three limits:
  - shared memory (3 stages × (BM + BN) × 128 B of K);
  - accumulator BM·BN·4 B in registers (Hopper) or TMEM (Blackwell);
  - threads (2048 / 256).

  Limits 「来源」: CUDA Programming Guide compute-capability table (smem 164 / 228 / 228 KB per SM for 8.0 / 9.0 / 10.0; 64 K registers; 2048 threads), PTX TMEM 256 KB, CDNA 3 white paper (64 KB LDS, 512 KB VGPR). Tile config 「二手」: CUTLASS sm80 DefaultGemmConfiguration.

  A tile costs max(main, (main + I/O) / k), with k capped at the number of tiles per SM.
  - k is 1 for 128×256 (H100), so large GEMMs behave like "dram"; 64×64 tiles get k = 4. MI300X gets k = 1 (LDS-bound).
  - The picker then favours small tiles with hidden I/O, so it hides more than "persistent" rather than sitting between "dram" and "persistent".
  - GEMM points (dram / occupancy / persistent vs measured): H100 4096³ 695 / 717 / 714 (716); B200 2048³ 1034 / 1214 / 1178 (1006); B200 4096³ 1446 / 1493 / 1487 (1532); 8192³ 1551 for all three. H100 MLA bmm 21.3 / 8.1 / 20.7 µs.
  - Validation (dram → occupancy): static 22.8 → 27.0 %, TTFT 32.9 → 42.1 %, max-load 19.1 → 21.5 %, MoE 7.7 → 6.7 %, Mixtral 22.2 → 30.5 %, B200 22.8 → 23.5 %.
  - Not default: B200 2048³ gets worse. The missing piece is likely the lower main-loop efficiency of small tiles.
- **B200 prefill.** The 8192-token chunk is MAC-bound at about 6.4 PF effective, which matches the library-best source (cuDNN → cuBLASLt 6220 TF at 8192³, flashinfer #1732). That is not the kernel TRT-LLM runs.
  - TensorRT-LLM PR #7943 (B200, autotuned, its default CUTLASS FP4 backend), M = N = 8192: K 1024 / 2048 / 4096 / 8192 / 16384 → 57.34 / 92.83 / 151.17 / 267.49 / 484.86 µs = 2397 / 2962 / 3637 / 4110 / 4535 TF. flashinfer's trtllm_autotune gives 3871 TF at 8192³.
  - Applied as per-shape kernel throughput (nearest (N, K)) on TRT-LLM B200 rows (`TRTLLM_FP4_SHAPES`, variant key `trtllm_fp4`, default). Gross, measured kernel time.
  - Rows: 128/128 1.651 → 1.216, 2048/128 1.496 → 1.108, 5000/500 1.281 → 1.086, 1000/1000 1.225 → 1.087. Group 1.125 / 22.82 % → 1.008 / 13.52 %; other groups unchanged.
  - Attention (FA4 / cuDNN) is not the issue at these contexts: the chunk is GEMM-dominated. Activation FP4 quantisation is already charged as conversion work on the vector path.
- `default_0194` keeps the previous default.

## §143 GEMM DRAM roofline under tile I/O; occupancy fill; SMEM-bound tiles (0.196)

- **DRAM floor (`mapping.CTA_DRAM_FLOOR`).** With tile I/O the array term includes the GEMM kernel's DRAM traffic, so cycles ≥ (M·K·a + K·N·w + M·N·o)·count / tile_io_Bpc 「推导」.
  - Before: the prologue and epilogue went at each SM's DRAM share, and the main loop was fed from the SRAM port as if operands were already on chip. Occupancy then divided the I/O by k, as if each SM had its own DRAM.
  - Effect on the H100 MLA absorbed bmm under occupancy: 2.93 µs → 7.60 µs, its byte floor.
  - Without tile I/O ("legacy"/wide), DRAM is a stage-level term: max(Σ array, Σ vector, DRAM).
  - Invariant: tests/test_core_196 fuzzes GEMM shapes × formats × 4 GPUs × {default, default_occ, default_tioauto, default_0194, default_0192}.
- **Occupancy fill (`CTA_OCC_FILL`).** The k resident CTAs start together. Their k first prologues and the final epilogue have no main loop to hide under, so the model adds k·prologue + epilogue once per SM.
- **SMEM-bound tiles (`CTA_SMEM_BW`, `smem_Bpc_sm`).** In SS-mode MMA both operands stream through SMEM at 128 B/clk/SM. Sources: CUDA guide, 32 banks × 4 B 「来源」; B200 measured, 64×64 tiles ≈ 50 % of FP16 peak (SemiAnalysis 「二手」); CDNA 3 LDS 128 B/clk.
  - So feed ≥ (BM·a + BN·b)·K / 128.
  - It binds 64×64 tiles on B200 (bf16) and on Hopper (fp8). It does not change the GEMM check points: B200 2048³ picks 128×128 at k = 2.
- **Result.**
  - default_occ GEMM points: H100 4096³ 709 (measured 716); B200 2048³ 1129 (1006); 4096³ 1478 (1532).
  - Default for comparison: 695 / 1034 / 1446.
  - Validation, default vs occupancy (MAE %): static 22.2 vs 25.7, TTFT 29.9 vs 36.2, Mixtral 18.8 vs 24.9.
  - Occupancy stays opt-in.

## §144 L2-resident activations in the GEMM floor; w_bits 0 = unset (0.197)

- **L2 discount (`CTA_L2_ACT`).** The §143 floor charges (max(0, activation bytes − L2) + weight bytes) / DRAM BW 「推导」.
  - The input was just written by the previous kernel and the output is read by the next. While both fit in the write-back L2 (refhw `sram_mib` 「来源」) they stay on chip, which matches memplan (stage `dram["act"]` = 0).
  - Weights stay in the floor: each is read once per step and the model is much larger than L2.
  - Tile reuse was never double-counted: the floor counts unique bytes.
- **Remaining regression vs 0.195.** The floor is applied per kernel: Σ max(compute_i, bytes_i) ≥ max(Σ compute, Σ bytes).
  - Small kernels such as MI300X b1 attn.k / attn.v (2.9 µs vs 0.56 µs of weights) add about 0.37 ms per 70B step.
  - That is a real effect of kernels running one after another. The missing piece is QKV fusion (vLLM QKVParallelLinear), which the IR does not model.
- **`W_BITS_ZERO_FIX`.** An `Op.w_bits` of 0.0 means "use the format's bits". Before, the extval ceiling bisection (`_model_tflops`), the GEMM check points and attention ops treated the B operand as 0 bytes.
- **GEMM points** (default / occupancy / persistent vs measured):

  | Point | default | occupancy | persistent | measured |
  |---|---|---|---|---|
  | H100 4096³ | 686 | 703 | 712 | 716 |
  | B200 2048³ | 959 | 945 | 991 | 1006 |
  | B200 4096³ | 1401 | 1469 | 1481 | 1532 |
  | B200 8192³ | 1551 | 1551 | 1551 | — |

  The B200 2048³ overshoot came from the 0-byte operand. The wave tail (256 tiles on 148 SMs) is already modelled, and no throttle term is needed.
- **Occupancy stays opt-in.**

## §145 Fused QKV; short-prompt TTFT diagnosis (0.198)

- **`ir.FUSE_QKV`.** When a layer's q, k and v are ungrouped linears over the same input with the same replication, they become one GEMM [t, h] × [h, (Hq + 2·Hkv)·d / rank] 「来源」. Sources: vLLM / SGLang QKVParallelLinear; TensorRT-LLM Attention.qkv (layers/attention.py), _torch/modules/attention.py qkv_proj.
  - FLOPs and weight bytes are unchanged; one kernel replaces three.
  - MI300X 70B TP8 b1: 10.3 µs → its 5.6 µs weight floor.
  - MLA (q_a / kv_a) and grouped projections are not fused.
- **Validation.** Every row gets faster.
  - Unfused → fused (MAE %): static 22.5 → 23.3, TTFT 31.6 → 36.9, Mixtral 20.2 → 21.3, MI300X latency 21.9 → 23.1, max-load 19.3 → 18.7, B200 13.6 → 13.1.
  - The groups that got worse were already too fast.
- **TTFT diagnosis.** The §144 L2 excess is per op, over its own token count (op.m · count).
  - 2048-token rows are unchanged by it, so long-sequence activations that exceed L2 are still charged.
  - The prefill-heavy regressions all come from 128–512-token prompts, which were already 1.17–1.96× too fast before §143–§145. TP8 rows are the worst (A100 70B 512 TP8: 55.9 vs 114 ms).
  - Missing term: fixed short-request latency (per request / per TP collective), not yet sourced. Not fitted.

## §146 NCCL small-message all-reduce latency; short-request TTFT audit (0.199)

- **All-reduce α.** In a single NVSwitch node, α = max(Link.alpha_us, 6.6 + 2(n − 1) · 0.6 µs).
  - Source: NCCL src/graph/tuning.cc (v2.19.3), baseLat[Ring][LL] and hwLat[NVLINK][Ring][LL], with 2(n − 1) ring steps 「来源」. Tree LL gives the same.
  - Measured cross-check: about 21 µs on 8 × A100 with LL (nccl-tests #123) 「二手」.
  - Engines with a custom all-reduce may be faster; not modelled 「假设」.
- **Short-prompt TTFT audit** (predicted vs measured, ms):

  | Row | 0.198 | 0.199 | measured |
  |---|---|---|---|
  | H200 mistral-7b 128 TP1 | 4.05 | 4.05 | 6.0 |
  | H200 70B 128 TP8 | 9.86 | 11.78 | 16 |
  | A100 70B 128 TP8 | 21.4 | 23.3 | 35.6 |
  | A100 70B 512 TP8 (Databricks server) | 55.9 | 57.9 | 114 |

  - (c) Eager prefill launch gaps were already charged in §123.
  - (b) Tokenizer / HTTP / scheduler cost doesn't apply to gptSessionBenchmark rows. The server rows have no sourced figure.
  - The remaining gap, including about 20 % on 2048-token rows, is not explained by a sourced mechanism. Candidates are TRT-LLM 0.8's context-phase kernel efficiency and session host overhead; both are unsourced and not fitted.

## §147 Custom all-reduce latency; KV-cap binding check; TRT-LLM 0.8 prefill sources (0.200)

- **All-reduce α in an NVLink node (≤ 8 GPUs).** α = 6.8 µs, the custom all-reduce floor (vLLM PR #33839: 8 × H100, 2 KB, CUDA graphs 「二手」).
  - Engines that use it: TRT-LLM v0.8 `use_custom_all_reduce` defaults to True; vLLM / SGLang default to custom AR 「来源」.
  - Groups over 8 GPUs, or engines without a custom AR (CUSTOM_AR False), use the §146 NCCL Ring-LL model.
- **DES KV-cap binding.** `simulate()` reports `kv_blocked`, the number of admissions refused by the cap. A cap counts as non-binding only when kv_blocked = 0, and then the run is bit-identical to cap off (same seed).
- **TRT-LLM 0.8 TTFT rows.** From the 0.8 performance.md and plugin.py v0.8 sources:
  - context FMHA on, TensorRT-native GEMMs (no gemm plugin), custom AR, gptSessionBenchmark, which has no tokenizer or HTTP.
  - Attention is ≤ 7 % of prefill at ≤ 2048 tokens.
  - The remaining ≈ 20 % too-fast bias has no sourced mechanism; TensorRT-native GEMM efficiency is unpublished. Not fitted.

## §148 GPU mechanisms vs the NPU path; gptSessionBenchmark TTFT scope (0.201)
- All-reduce latency now depends on `Link.ar_lat`. "gpu" (refhw NVLink/NVSwitch nodes): CUDA custom AR 6.8 µs for group ≤ 8 「二手」, NCCL Ring-LL beyond (§146–147). "" (NPU links, D2D, cross-node net): t_α = `alpha_us`, the design's own collective model. `GPU_AR_ONLY=False` restores ≤ 0.200.
- CTA tile mechanisms (DRAM floor, SMEM BW, occupancy fill, L2 activation reuse) live only in `_cta` and need `tile_io_Bpc > 0`; NPU chips are unaffected.
- Fused QKV is an IR transform valid for any engine/compiler; it conserves FLOPs and bytes.
- gptSessionBenchmark TTFT (output_len 1) = generate() + sync: host setup (buffers, paged-KV add/remove, decoder newBatch) + context pass + last-token sampling + one host-only loop iteration (no extra engine pass). Modelled: context pass, last-token lm_head and sampling. Not modelled: the host setup, which has no sourced cost.
- Prefill lm_head m = batch (last token), matching TRT-LLM `gather_context_logits=False`.

## §149 Standing GPU→NPU leak audit; MI300X all-reduce latency (0.202)
- Rule: a GPU-kernel mechanism must be gated on a GPU marker: `Chip.gpu` (refhw), `instance_sched="cta"` / `tile_io_Bpc`, `Link.ar_lat`, or extval-only refhw fields. It must never be gated on generic fields such as `instance_sched=="wide"` or `topology`. `tests/test_core_202.py` flips every GPU flag on 100T/1P (decode/prefill, TP1/TP4) and requires bit-identical results. It also lists the mechanisms that apply to NPUs on purpose, with the reason.
- NPU dispatch: no per-op launch cost (`kernel_us = 0`): a statically compiled schedule 「假设」. `layer_overhead_us` / `kernel_us` are the user's knobs. CUDA graph/eager numbers stay in extval.
- `WIDE_WAVES` refill is GPU-only (`GPU_WAVES_ONLY`). On an NPU, "wide" is a lock-step ganged array with N padded to E·cols, consistent with its broadcast feed term.
- All-reduce latency floor by `Link.ar_lat`: "" = alpha_us; "gpu" = 6.8 µs CUDA custom AR (≤ 8) or NCCL Ring-LL; "rocm" = vLLM ROCm custom AR measured on MI300X, {2: 7.26, 4: 7.18, 8: 12.29} µs 「来源」 vLLM PR #19744.

## §150 NPU dispatch model; per-step walls (0.203)
- `Chip.dispatch` modes:
  - "" = ideal (default, 「假设」).
  - static: `dispatch_us` (10 µs 「二手」) once per step per stage. Decode is serial because the sampled token feeds the next step.
  - trace: 1.5 µs 「二手」 per dispatched op.
  - host: 20 µs 「二手」 per dispatched op.
  - Dispatch is serial and is added to t_sync (t_disp).
- `Result.walls()` per step:
  - The window (total − t_sync) goes to argmax(compute, max(DRAM, SLC), link).
  - t_sync is split into dispatch (t_disp + GPU kernel_us), fixed overhead (per-layer fixed time, layer_overhead_us; key `fixed_overhead`, renamed from "static" in 0.204, not static power) and collective latency (the rest).
  - The parts sum to the stage time and the split is exact under "stage" overlap.
- Gaps: no public dispatch figure for Gaudi, Inferentia/Trainium, Groq or Cerebras. The TPU figure covers the host path only. The sensitivity panel scans the modes.

## §151 Walls in the UI; NPU preset TPOT check (0.204)
- UI "每步时间分解 · Per-step time split":
  - A bar of exposed shares (compute / DRAM / collective BW / collective latency / dispatch / fixed overhead) with the wall segment outlined and marked ▲.
  - Bilingual zh/en labels.
  - When `exact = false` (overlap mode other than "stage") a note says the overlapped window is attributed to the dominant resource.
  - It is rendered from the same `/api/eval` response as the KPIs (the request is sequenced; stale answers are dropped), so it cannot show a different step.
  - Dispatch control: `c-dispatch` / `c-dispatch_us`.
- "Fixed overhead" is the per-layer fixed time (`layer_overhead_us`), not static power (energy's idle/static power is separate).
- NPU preset sanity: qwen3-8b decode, TP1, batch 1, ctx 4096, LPDDR5X 4 × 64 bit @ 8533 MT/s = 273.1 GB/s × mem_eff 0.7 = 191.2 GB/s.
  - Bytes per step: streamed bf16 weights 15.136 GB (7.568 B params = 8.19 B − 0.62 B embedding rows, which are looked up), minus the SRAM-resident weights (100T 64 MiB, 1P 256 MiB), plus KV read 0.604 GB (4097 positions × 36 layers × 8 KV heads × 128 × 2 × 2 B).
  - 100T: 15.675 GB / 191.2 GB/s = 82.0 ms; 1P: 15.474 GB = 81.0 ms.
  - Compute is hidden: an M=1 GEMV on a 56-row output-stationary array uses 1/56 of the rows, so 8.9 ms raw vs 82 ms DRAM.
  - 1P is only 1.3 % faster because both presets use the same memory default; the only difference is 192 MiB more SRAM-resident weight. That is the correct DRAM-bound result.


## §152 NPU preset memory pairing; machine balance (0.205)
Machine balance is peak bf16 FLOPS ÷ DRAM bytes/s. Public chips sit at about 200–560 FLOP/B: H100 295, B200 281, MI300X 247, TPU v5e 229, v6e 560, Gaudi 3 496/453 (来源), Ascend 910B/C 200–250 (二手). With 4×64-bit LPDDR5X (273 GB/s), 1P (≈1.05 PF) sat at 3840 FLOP/B and was DRAM-bound at batch 1, so about 94 % of its compute was idle.
- 1P defaults to 4 HBM3E stacks at 9.2 Gb/s (4.71 TB/s, 223 FLOP/B; 推导: the B200/MI300X-class ratio for this compute). H100-like defaults to the H100 memory. 100T keeps LPDDR (368 FLOP/B, in class).
- The defaults are in `PRESET_MEM`, keyed by chip name; the legacy `*-lp4x64` presets keep the old pairing.
- `Result.balance()` compares the chip's FLOP/B at effective bandwidth with the workload intensity (the heaviest stage's flops ÷ DRAM bytes). Beyond a 10× gap (假设), the side named idle is the wasted one. Decode at small batch (≈1 FLOP/B) always warns that compute is idle; this is a property of the workload, not a model defect.


## §153 Critical batch; MI300X max-load residual by row (0.206)
**Critical batch.** Per-step decode intensity AI(B) = flops / DRAM bytes of the heaviest stage, evaluated by the model at batch B:
weights are read once per step, while KV reads, activations and collectives grow with B. AI(B) rises and saturates at the
KV-bound asymptote ≈ FLOP per token ÷ KV bytes per token (qwen3-8b at ctx 4096: ≈ 28.7 FLOP/B; at ctx 64: ≈ 1000).
- B* = min B with AI(B) ≥ chip FLOP/B (effective bandwidth).
- B_lim = max B that fits the KV capacity and the TPOT SLO.

The verdict:

| case | output |
|---|---|
| B* ≤ B_lim, B < B* | hint: raise batch to B* |
| B* ≤ B_lim, B ≥ B* | balanced |
| B* > B_lim | warning: names the binding limit (KV capacity or TPOT SLO) |
| no B* (KV-bound), chip ≤ class top 560 FLOP/B (TPU v6e, 来源) | hint: the lever is KV bytes (fp8 KV / MLA / shorter ctx), not compute; every public chip idles the same way here |
| no B*, chip > 560 | warning: compute oversized for this workload at any batch |

`BALANCE_BMAX = 65536` and `BALANCE_WARN = 10` (bandwidth side) are 假设.

**MI300X max-load, row by row** (ROCm vLLM 0.11.2 README, 来源; predicted / measured, output tokens/s):

| ISL/OSL | B | pred | meas | ratio | prefill share | model step | implied step |
|---|---|---|---|---|---|---|---|
| 128/2048 | 3200 | 25106 | 13562 | 1.85 | 0.04 | 122 ms | 231 ms |
| 128/4096 | 1500 | 20133 | 11801 | 1.71 | 0.02 | 73 ms | 126 ms |
| 500/2000 | 2000 | 20415 | 11250 | 1.81 | 0.13 | 85 ms | 165 ms |
| 2048/2048 | 1500 | 12339 | 7753 | 1.59 | 0.32 | 82 ms | 154 ms |

The error is in the decode step, not in prefill or scheduling.
- Re-read README flags: `--gpu-memory-utilization 0.9`, `--max-model-len 8192`, `--max-num-batched-tokens 131072`, Num Prompts = Max Num Seqs (one wave), "192GiB" GPUs.
- KV at the last step: 143 GB per GPU of ≈ 165 GB available, so no preemption (推导).
- Chunked prefill takes 4 iterations of 131072 tokens at 128/2048.

The same README's latency table (vllm bench latency, 70B FP8 TP8) separates the step into parts (推导):

| quantity | measured | model |
|---|---|---|
| step at b1 | 7.8 ms | 10.9 ms (dispatch 6.25) |
| step slope b64→128, isl 128 | 0.10 ms / seq | 0.026 ms / seq |
| KV slope (isl 2048 − 128, b128) | 5.26 ms → ≈ 1.0 TB/s effective | 1.62 ms → 3.1 TB/s |

So the model is too slow at b1 and too fast per sequence; the MI300X latency group's 21.6 % MAE averages opposite-sign errors. At max-load batches the per-sequence term dominates, which gives 1.6–1.85×. Candidates are hipBLASLt FP8 GEMM efficiency at decode M (the MAF ceiling is a large-square figure) and ROCm paged-attention throughput with fp8 KV. §127 has only relative kernel speedups, and no public absolute figure was found, so **nothing is applied**. Fitting the per-sequence slope to this README would be tuning on the validation source. `test_core_206` pins the localisation.


## §154 MI300X decode suspects: independent sources (0.207)
**ROCm paged attention, fp8 KV** (来源: ROCm/FlyDSL PR #825, production `pa_decode_ps_kernel`, gfx942/MI308X, block 16):

| bs·ctx (q/kv heads) | KV bytes per call | µs | TB/s | ÷ 5.3 |
|---|---|---|---|---|
| 32·4096 (16/1) | 33.6 MB | 34.9 | 0.96 | 0.181 |
| 128·4096 (16/1) | 134 MB | 83.6 | 1.61 | 0.303 |
| 128·16384 (16/1) | 537 MB | 216.3 | 2.48 | 0.468 |
| 256·8192 (8/1) | 537 MB | 235.5 | 2.28 | 0.430 |

The KV stream efficiency (`Scenario.kv_mem_eff`) of an MI300X decode step is read from this curve at its KV bytes per
layer call: log-linear in bytes, with the two 537 MB points averaged. The 0.206 README-derived figure (≈ 1.0 TB/s at
b128) agrees in magnitude, but it was not used.
- The MI308X → MI300X transfer is 推导 (same HBM3 5.3 TB/s, 二手; fewer CUs, so a conservative bound).
- Only `PA_ROCM_HW = {"MI300X"}` decode is affected.
- Effect: lat 1.02 / 23 % → 0.97 / 19 %; max-load 1.65 / 65 % → 1.49 / 49 %.

**Small-M FP8 GEMM** (来源: aiter tuned table, gfx942 / 304 CU, a8w8 int8 = fp8 MFMA rate on CDNA3):
- Times are flat in M up to about 1024: a per-kernel floor of 7.7–17 µs on 6.6–7.9 MB weights, i.e. 0.5–0.9 TB/s.
- Large M sustains 0.38–0.49 of peak.
- The model is faster than the source at M ≤ 256 and within 0.8–1.3× above.

A small-M floor raises the per-step constant, not the per-sequence slope, so it cannot explain the residual. It would
worsen batch 1, where the model is already 40 % slow. **Not applied.**

**NVIDIA.** Max-load gmean is H100 0.88 / H200 1.04, and the TRT-LLM static rows show no batch-slope error of the
MI300X kind. FlashMLA (0.108) remains the sourced NVIDIA attention-stream figure, as a variant. No change.

**Remaining residual** (1.43–1.70×, max-load batch 1500–3200): still a per-sequence cost beyond PA at MI308X
efficiency. Candidates (no source yet): PA efficiency at the 0.9 GB-per-call max-load sizes (beyond the 537 MB top of the
curve, clamped), and the vLLM V1 per-sequence host and sampler costs at thousands of sequences (§127: ≈ 1.9 µs/seq,
not applied).


## §155 Balance levers; fixed-state and decision critical batch (0.208)
**What-if.** One variable per lever, evaluated on the sensitivity case loop (`_eval_cases`):

| lever | qwen3-8b · 1P · ctx 4096 | RWKV-7 7.2B · 1P | StartLux 35B-A3B · 1P |
|---|---|---|---|
| base | B* ∞ (KV), limit 141 (capacity), 28.8 FLOP/B, 4.70 ms | ∞ (state), 1460 (SLO), 181, 4.13 ms | ∞ (kv+state), 218, 29.8, 1.76 ms |
| KV fp8 | 57.1, limit 283 — works | n/a | 36.9, 303 — works |
| state bf16 | n/a | **B* 1541 ≤ 2092 — reachable** | 42.6, 279 — works |
| ctx ÷ 2 | 53.1, 283 — works | unchanged (state is ctx-free) | 35.0, 303 — works |
| SRAM × 2 | no effect | limit 1525, no effect | no effect |
| DRAM × 2, same capacity | balance 159; TPOT 2.35 ms — works (TPOT) | **B* 758 ≤ 2048 — reachable** | TPOT 0.88 ms — works (TPOT) |

**Fixed state.** RWKV's recurrent state does not grow with context, but it is read and written for every sequence on
every step (fp32: capacity 33.0 MiB = 34.6 MB per sequence; traffic read + write ≈ 69.2 MB per sequence per step at 7.2B). AI(B) therefore saturates at FLOP per token ÷ state bytes
≈ 181 FLOP/B, a ctx-independent constant (KV models: ∝ 1 / ctx). Balance is reachable exactly when the chip's FLOP/B is
below that constant. The critical-batch logic needed no special case; only the label and hint (`asymptote_by`)
changed.

**Decision readout.** One forward over prompt + options per decision. The batch of decisions has its own critical batch,
limited by capacity and the TTFT SLO, which is used as the decision latency limit.


## §156 Recurrent-state residency; adversarial review (0.209)
- **Mechanism.**
  - A recurrent state byte is read and written every step, so it saves 2 B of DRAM per step per byte of SRAM.
  - `Chip.state_sram_mib` is a dedicated buffer. Shared SRAM follows STATE_FIRST (§ 0.192).
  - Resident bytes per stream: state_sram = min(st, buf) + shared share. KV and state misses are computed separately (STATE_SPLIT).
  - DRAM capacity still counts the full state, as a backing copy. This is conservative 「假设」.
  - Area and power of the buffer are not modelled.
- **Critical batch under partial residency.**
  - AI(B) rises while extra sequences fit on chip, then drops at the knee where state spills.
  - The monotone search can miss a window only one batch wide, so the knees are probed explicitly.
- **RWKV-7 7.2B on 1P.**
  - Per-sequence state capacity is 33.0 MiB = 34.6 MB (fp32); its DRAM traffic when not resident is 2× that, ≈ 69.2 MB per step (read + write).
  - Minimum dedicated buffer for balance: 10352 MiB (critical batch 315 ≤ 1664) 「推导」.
  - bf16 state remains a separate what-if lever.
- **Review findings.**
  - The UI's ∞ label ignored `asymptote_by`.
  - The open-window overlap replay cost O(tokens).
  - Neither changed any result value. The fingerprint change comes from STATE_SPLIT only.


## §157 Pipe-mode KV / state split; state capacity vs traffic; review closure (0.210)
- **Pipe mode.** Each op's DRAM bytes are weighted by its own stream's miss share: KV ops use 1 − (kv_sram − state_sram)/kv_total, and state ops use 1 − state_sram/state_total. This matches the step totals in `step_dram_bytes`. It only changes which ops the DRAM time lands on, and therefore how much of it overlaps compute.
- **State size.** Capacity per sequence (what SRAM must hold) is S. The DRAM traffic per step when not resident is 2S (read + write).
  - RWKV-7 7.2B fp32: S = 33.0 MiB = 34.6 MB, so traffic is ≈ 69.2 MB per step.
  - The balance asymptote uses 2S. Residency and the state SRAM lever use S.
- **Review closure.** Multi-card PP, walls, the preset memory pairing, cross-route fuzz and the huge-batch timing guard are covered in `tests/test_core_210.py`. No new defects were found beyond those fixed in 0.209.


## §158 Lever cost: state SRAM area / energy vs 2× DRAM bandwidth (0.211)
- **Accounting.**
  - Area = Σ SRAM MiB (shared + dedicated state buffer) × `mm2_per_mib_sram` + DRAM bus bits × `mm2_per_dq_*` (+ D2D / C2C PHY).
  - Energy = action counts × the preset's pJ per action. SRAM-port bytes are unchanged by residency; DRAM bytes drop by the state read+write.
  - Leakage is not modelled 「假设：未建模」.
- **Densities.**
  - `n5_derived` SRAM: 0.68 mm²/MiB 「推导」 = TPUv4i CMEM 28 % of < 400 mm² ÷ 128 MiB × N5/N7 bitcell 0.021/0.027.
  - Array-only N3E bound: 0.25 mm²/MiB (31.8 Mib/mm², IEDM 2022 via WikiChip) 「二手」.
  - HBM PHY: 32 mm² per 2048 DQ 「推导/二手」 (Samsung HC2026).
- **Where the cost is read.** At the lever's critical batch: what the lever buys is the batch it unlocks.
- **RWKV-7 7.2B on 1P.** The state buffer needs +7052 mm² (above the 858 mm² reticle even at the array-only bound) and gives −32 % J/token at B = 314. 2× bandwidth needs +64 mm² of PHY, keeps J/token unchanged and reaches balance at B = 758. Bandwidth is the lever that is cheaper in die area. 「推导」 from the presets; no tuning.


## §159 HBM shoreline; SRAM leakage (0.212)
- **Shoreline.** need = stacks × 11 mm 「二手」. Available edge = k × √(die mm²): k = 2 is typical 「二手」, k = 4 is the limit 「假设」. The die is taken as square, and the area proxy is a lower bound, so a real die is larger and has more edge.
- **Leakage.** P_leak = MiB × 230.7 µW/MiB (3 nm HD at 25 °C, a lower bound 「来源」).
  - The scope avoids double counting: the state buffer is always charged; the shared SRAM is charged only without a measured idle figure.
  - It enters J/token through the time window, like idle.
- **RWKV-7 7.2B on 1P.** The 2× DRAM lever reaches balance at +64 mm² of PHY, but 8 stacks exceed two die sides even at the reticle, so it needs a multi-die package (B200-class). The state SRAM is +7052 mm², with leakage ≥ 2.4 W. Bandwidth is still the cheaper lever, but its real cost is packaging (CoWoS-L / 2 dies), not PHY area.


## §160 Uncore area (0.213)
- **Die proxy.** die = (SRAM incl. state buffer + SLC + MAC + DRAM / D2D / C2C PHY + host IO + MC) / (1 − uncore_frac).
  - uncore_frac = 0.23: TPUv1 control 2 % + unlabelled 21 %, ISCA 2017 「来源/推导」, used for every node 「假设」.
  - host IO = 6.0 mm²: PCIe, 1.8 % of TPUv1 per NeuroMeter 「二手」.
  - MC: no preset, because no public per-stack figure was found.
- **Why a share.** NoC, routing and white space grow with the blocks they connect. Control is near-fixed, but it is 2 %, so a share is adequate.
- **Effect.** All die and shoreline results grow by ÷ 0.77 plus 6 mm². 1P base: shoreline tight → ok. DRAM ×2: still tight, because 8 stacks need edges beyond the two long sides even at 654 mm². State SRAM: 9729 mm². Area conclusions are unchanged, and the bandwidth lever is still the cheaper one.


## §161 Two-die package lever (0.214)
- **Construction.** Two dies, each with half the compute and SRAM and half the stacks of the 2× option. They are joined by the existing D2D tier: `package_cards` = 2, `d2d_enabled`, TP × 2. TP all-reduces cross the D2D link, and their bytes, energy (0.25 pJ/bit, UCIe-A 「来源」) and latency come from the multi-card model.
- **Shoreline per die.** Typical: min(2, 4 − d2d edges) long sides. Maximum: 4 − d2d edges. Splitting area A into two dies raises the total edge by √2.
- **Finding for RWKV-7 7.2B and qwen3-8b on 1P.** Two dies match the 8-stack single die within 3 % on TPOT and 6 % on critical batch, at +1.6 mm² of D2D PHY per die. The packaging trade is real but cheap in model terms. The remaining blocker is per-die shoreline (4 stacks on a 333 mm² die take three edges), not D2D bandwidth or energy. 「推导」

## §162 HBM stack height as a capacity lever (0.215)
Taller stacks (8H→12H→16H, 24/32 Gb dies) multiply the capacity per stack, but bandwidth stays the same: it is set by the 1024/2048-bit interface × rate. The package height is fixed: SK hynix's 12-Hi HBM3E keeps "the same thickness as the previous 8-layer product" (来源, SK hynix 2024-09-26). The 720 µm (HBM3/3E) → 775 µm (HBM4 12/16-Hi) relaxation is reported by TrendForce 2025-04-21 / ZDNet Korea (二手). So the footprint and shoreline are unchanged. Vendor status (re-verified in 0.216, see §163): HBM3E 12H 36 GB is in volume production at SK hynix (来源, 2024-09-26) and was developed by Samsung (来源, 2024-02-27). Micron is shipping production-capable units for qualification (来源, Micron 2024-09-05; product page: designed into AMD MI350), not a volume claim. HBM3E 16H 48 GB was announced by SK hynix (来源, 2024-11-04). The HBM4 tags are in §163. Energy per bit is unchanged (假设); standby/refresh is added in 0.216 (§163).
In KV-bound decode the arithmetic intensity asymptote = FLOP/token ÷ KV bytes/token, which does not depend on capacity (推导). Capacity only raises the batch limit until the TPOT SLO binds, and the throughput ceiling ≈ BW ÷ KV bytes/token. So qwen3-8b at ctx 4096 can never reach balance (asymptote 28.8 FLOP/B with bf16 KV, 57.0 with fp8, chip 318), and with fp8 KV, bandwidth (more stacks) beats height.

## §163 HBM availability re-check and standby/refresh power (0.216)
Sources (fetched 2026-10-17):
- JEDEC HBM3 press release (2022-01-27): JESD238 supports 4/8/12-high, with provision for a future 16-high; 8–32 Gb/layer, up to 64 GB.
- JEDEC JESD270-4 press release (2025-04-16): HBM4 4/8/12/16-high with 24/32 Gb dies, 64 GB max (16H × 32 Gb).
- HBM4 12H 36 GB is in volume: Samsung (2026-02-12) and Micron (2026-03-16, "volume shipment … Q1 2026"); SK hynix mass supply from Q2 2026 (Korea Herald, 二手).
- HBM4 16H 48 GB is sampling: Micron "shipping samples"; SK hynix unveiled it at CES 2026, and its volume shipments are reported but not confirmed by the company (二手); Samsung is "ready" but says demand is limited.

Catalog tags after the check: HBM3E (12,24) shipping; (16,24) announced. HBM4 (12,24) shipping; (16,24) sampling; (16,32) none. No tag changed; every tag now carries its source in `HBM_CAP_SOURCES`.

Standby + refresh (static) per die: 48 mW (IDD2N 40 mA × 1.2 V) + 1.95 mW/Gb (refresh = (IDD5AB − IDD3N) · VDD · tRFC / tREFI per 8 Gb). Source: DRAMsim3 HBM2_8Gb_x128 config (二手 / 推导). These are HBM2-era 1.2 V values (HBM3 uses 1.1 V), so the estimate is high-side. Scaling refresh linearly to 24/32 Gb dies is 假设.
P = stacks × height × (48 + die_Gb × 1.95) mW, so a 4 × 8H24g base is 3.03 W/card.
To avoid double counting, it is charged only when no card idle figure (idle_W / idle_frac_tdp) is set, the same rule as the 0.212 SRAM leakage; the what-if strips idle_W, so the levers show it.
qwen3-8b on 1P with fp8 KV: 12H +1.52 W, 16H +3.03 W, 8 stacks (dram_bw_x2) +3.03 W. This is about 1–3 % of the ~105 W card average.

## §164 What-if cost batch for unreachable levers (0.217)
Cost is compared at the lever's critical batch when it reaches balance; otherwise at its batch limit (the operating point it unlocks). The base is evaluated at min(that batch, base batch limit). Before 0.217 the fallback was the configured batch (1), which priced the lever at batch 1: for qwen3-8b that was 0.48 J/token (105 W × 4.6 ms). Invariant: J/token × (batch / TPOT / cards) = W/card (推导, tested).
qwen3-8b 1P, fp8 KV, at the batch limits: base 15.1 mJ/tok @283; 12H 14.65 @453; 16H 14.74 @479; DRAM×2 15.35 @283; ctx÷2 9.22 @566.

## §165 Catalog provenance audit, Samsung HBM4 24 GB, priced batch (0.218)
Samsung (newsroom 2026-02-12, 来源) offers 12-layer HBM4 at 24–36 GB, so 24 GB means 16 Gb dies (推导). JESD270-4's press release lists 24/32 Gb only, so this entry is tagged custom·shipping.
Every HBM product/limit tag and pin rate now carries a provenance-tagged source string. LPDDR, D2D, chip and link presets carry tags too; chip geometries are 假设 design points. A test enforces this.
What-if rows state the batch they are priced at: critical batch, else batch limit, else configured. The priced batch is ≤ the row's batch limit (invariant).

## §166 Source upgrades, LPDDR per-entry sourcing, residual review (0.219)
**HBM.**
- Samsung newsroom 2023-10-20 (来源): HBM3E Shinebolt at 9.8 Gb/s with samples shipping, so the tag changes from custom·announced to custom·sampling. Samsung's 8H and 12H HBM3 are in mass production.
- SK hynix newsroom 2023-04-20 (来源): 12-layer HBM3 at 24 GB, sampling, with supply from 2H 2023.

**LPDDR.** Sources are per entry (`lpddr_source`).
- JEDEC: JESD209-5 PR 2019-02-19 (6400 MT/s), JESD209-5B PR 2021-07-28 (8533), JESD209-6 PR 2025-07-09 plus the JEDEC Mobile Forum slides (10.667 intro, 14.4 max, x24 channel).
- Vendors:
  - Samsung 10.7 Gb/s LPDDR5X, 32 GB package (2024-04-17).
  - Micron 1γ 10.7 (2025-06-03).
  - SK hynix LPDDR5T 9.6 (2023).
  - Micron SOCAMM2 192 GB sampling (2025-10-22) and 256 GB samples (2026-03-03).
  - Micron LPCAMM2 16/32/64 GB (2024-01-09), 7500 (2024-05-07).
  - CXMT LPDDR6 12.8 Gb/s, 16 GB mass production (2026; TrendForce summary).
- Correction: SOCAMM2 192 GB, shipping → sampling. 24 entries still carry the research-doc citation (二手); deferred.

**MI300X max-load residual (1.43–1.70×).**
- New quantitative source: vLLM PR #40459 (RFC; vLLM 0.19, H100, Qwen3-0.6B, 256 sequences, eager).
  - The scheduler takes `schedule()` 907 ms + `update_from_output` 572 ms of a 12.68 s wall.
  - That is about 940 steps, so about 1.57 ms per step, or about 6 µs per sequence (推导). The update loop alone is 0.76 µs/seq (microbench).
- At max-load batches of 1500–3200 this is 9–20 ms per step, against an implied per-step gap of 53–109 ms (§153). It explains at most about 20 %.
- Not applied: different vLLM version (0.19 vs the README's 0.11.2), different host CPU, and enforce_eager; §127's 1.9 µs/seq stays unapplied too.
- Ruled out: a scheduler-only explanation of the whole residual.

**TRT-LLM 0.8 prefill, about 20 % fast.** No new public source for TensorRT-native GEMM efficiency or GptSession host setup was found. The §146–148 conclusions stand. Nothing fitted.

## §167 LPDDR sourcing complete; model check; what-if monotonicity fuzzer (0.220)
**LPDDR sources.** All 66 LPDDR entries now carry a per-entry source: JEDEC press releases, Samsung product pages, Micron datasheets and pages, SK hynix / CXMT newsrooms, or a 二手 distributor list where no vendor page was found. Entries with no part found are 假设 and tagged JEDEC-allowed · no product.

Corrections:
- SOCAMM2 192 GB: back to shipping (SK hynix mass production, 2026-04-20; the 0.219 downgrade used Micron's sampling release only).
- SOCAMM2 48–128 GB: shipping → none.
- LPDDR5 x32 3/6 GB and x64 3 GB: shipping → none.

No bandwidth or capacity number changed, so no model result moved.

**Models (checked 2026-10-17).**
- RWKV: the newest release is RWKV7-G1k-20260930, already in the catalog.
- StartLux: Decision v1 / v1.1 are in the catalog. ReaDy-Qwen3.8-27B (2026-10-14) is a 5-layer DSpark speculative draft that reads two target recurrent states (rank 8). It needs its target model and speculative decoding, which this tool does not model, so it is not added.
- Microsoft-Decision-1 (still the 0.191 proxy) and Jev: no official public config.

**Fuzzer.** Seeded random models × chips × batch × ctx × KV format. Levers that only add a resource must not worsen their own axis:
- KV fp8 / ctx ÷ 2: batch limit and KV asymptote ≥ base.
- Taller HBM: batch limit ≥ base, chip FLOP/B unchanged.
- SRAM × 2: batch limit ≥ base.
- DRAM × 2: TPOT ≤ base.

The fuzzer checks physical direction only, so it is not a calibration.

## §168 Default memory realism; separate DSpark draft + state reader (ReaDy); draft-path fuzzer (0.221)
**Default audit.** Defaults checked (2026-10-17): PRESET_MEM, MEM_LEGACY_0204, `System.mem_id`, the scenario default,
the UI fallback / placeholders, and every catalog type/form/width default.
- Only two defaults pointed at a no-product configuration:
  - LPDDR5 x32 default capacity 6 GB → **8 GB**. This is the nearest shipping x32 part (Samsung K3LKCKC0BM-MGCP 64 Gb x32
    6400, 来源). Same rate and width, so the bandwidth is the same and only the capacity grows: 4 × x32 24 → 32 GB.
  - The H100-like rate 5234. This is the H100 SXM operating point: 3.35 TB/s ÷ 5120 bit (来源/推导). The Hopper whitepaper
    gives 2619 MHz DDR = 5238, 0.08 % apart. It is now a catalog grade (疑似 JEDEC: below the JESD238 6.4 maximum, bin
    table not public · shipping). The number is unchanged.
- LPDDR6 x48 and LPDDR6 LPCAMM2 have no shipping alternative of the same intent. They are flagged what-ifs
  (`WHATIF_DEFAULTS`, 假设).
- Reference / validation hardware rows (refhw, e.g. HBM3 2430 / 3186) model measured products, not defaults. Tagging
  their rates is deferred.

**Speculative decoding coverage.**
- Covered: MTP self-speculation (k sequential module passes). DSpark block drafts, built in (V4.1) or as a separate draft
  repo attached to a released target (`<target>+<draft>`).
- Not covered: an independent autoregressive draft LM (its own KV, k sequential q = 1 passes, then target verify q = 1+k).
  Design (deferred):
  - `Serving.draft_model` holds a second ModelSpec placed on the same cards.
  - Step = k × draft step(b, ctx) + target step(q = 1 + k).
  - Memory = both weights + both KV caches.
  - The acceptance lever stays 「假设」.

**ReaDy-Qwen3.8-27B.** A DSpark drafter for a separate Qwen3.8-27B target.
- Structure (config + safetensors headers): 5 dense GQA full-attention layers, fc over 5 target layers, Markov rank 256,
  confidence head; no own lm_head (shared).
- The recurrent-state reader (reader_config + MODEL_SPEC) runs per proposal and per sequence:
  - It reads the committed fp32 state [2, 48, 128, 128] of target layers 30 / 62.
  - It is copied into CUDA-graph buffers, so 3 × 6.29 MB of state traffic is charged (推导).
  - Then: RMS norm, value map 128 → 8, rank-8 query lifted to 128, read, + log magnitude → 864 → 128 → 5120 residual
    at draft layer 4.
- Published τ 4.5285 (block 7) ⇒ a = 0.828 under the i.i.d. model (推导, `spec_accept_from_tau`). DSpark τ 4.1739 ⇒ 0.801.
- Mechanism check (no fitting):

  | Quantity | Modelled | Paper |
  |---|---|---|
  | ReaDy / DSpark throughput | 1.085× (reader 0.02 % of the step) | 1.0797× |
  | Speed-up over AR (1P, batch 1) | 4.03× | 3.16× (B200) |

  The second gap is a residual: the modelled verify step is only +12.6 % over an AR step, while the paper implies
  τ / speed-up ≈ 1.43. Engine scheduling and draft-launch overheads are not modelled. Not tuned.

**Fuzzer (seed 221, separate-draft path).**
- Tokens per step = E(min(k, block), a).
- Acceptance changes neither the step time nor raises TPOT.
- Turning speculation on never shortens a step.
- The reader only adds work.

Conclusions: no preset KPI changed.


## §169 Spec-decode verify-step audit; independent draft model (0.222)

Question: the modelled verify step was +12.6% over plain decode (ReaDy, B200, b1, ctx 2048), while the ReaDy paper
implies ~43%. The model gave 4.03× against the paper's 3.16×. Each mechanism was checked against code, not against
the paper.

| Mechanism | Status | Source |
|---|---|---|
| (a) k+1-token attention vs full KV, causal block | Already charged (reads ctx+q keys; causal 1.0 slightly over-charges) | — |
| (b) Draft dispatch | GPU kernel_us already applies per draft op. Static dispatch now ×2 on spec head stage | SGLang separate draft CUDA-graph runner (来源); NPU compiler 假设 |
| (c) Sampling / bookkeeping | DSpark Markov head is now B sequential head GEMM + sample steps | SGLang dsc/dspark_draft.py `draft_markov_block` (来源) |
| (d) Recurrent-state snapshot | State traffic (q+3)× per sequence (read 1, write q intermediates, scatter read+write); scratch q × state in memory | SGLang memory_pool `intermediate_ssm_state_cache` [layers, slots, num_draft_tokens, HV, V, K] + `fused_mamba_state_scatter` (来源) |
| (e) GEMM tile at M=k+1 | No change: the cta M-tile is the same for M=1 and M=8 | — |

Notes:
- ReplaySSM (opt-in, which skips the scratch) is not modelled.
- That the pinned SGLang rev 2948168 takes the same path is 假设.

Gap movement (B200, b1, ctx 2048):

| Metric | Before | After | Paper |
|---|---|---|---|
| Step ratio | 1.126 | 1.165 | 1.433 |
| Speedup | 4.02× | 3.89× | 3.16× |
| Verify extra | 0.067 ms | 0.334 ms | — |
| Draft | 1.353 ms | 1.528 ms | — |

The remaining ≈ 0.27 step-ratio gap is a residual. Engine host and scheduling time between draft and verify has no
sourced figure, so it is not modelled. Nothing was fitted.

Independent draft model (`serving.draft_model`; designed in §168):
- The draft's own scenario is evaluated with the target's TP/DP, pp = 1, as-released formats, and spec off.
- Its stored weights + KV + state are added to the target's last-stage DRAM need.
- Step = target verify (q = 1+k) + k × draft pass.
- Acceptance a is a 假设 lever; tokens per step = (1−a^(k+1))/(1−a).
- Placement on the same cards is 假设. An MTP target keeps its MTP weights but does not run them.
- Example (1P, b8, ctx 4096), qwen3-32b alone vs + qwen3-0.6b (k=4, a=0.7):

  | Metric | qwen3-32b alone | + qwen3-0.6b |
  |---|---|---|
  | TPOT | 21.93 ms | 9.96 ms |
  | DRAM need | 70.0 GiB | 74.6 GiB |

Deferred: draft energy in J/token; UI field for draft_model; tree drafting.

Conclusions that shifted: speculation on hybrid (linear-attention) targets at large batch is much less attractive
than before. ReaDy at b32 drops from 6032 to 3815 tok/s, and needs +37 GiB scratch. Conclusions at b1 are
essentially unchanged.

## §170 Draft energy; spec rollback strategy (snapshot vs ReplaySSM) (0.223)

**Draft energy.** The independent draft model's action counts are added to the step's counts:
- They are taken over all of the draft's cards, using `energy.action_counts` of one pass, × k passes.
- The step window already contained the draft's time, so W/card = J/step ÷ window.
- The invariant J/token × tok/s/card = W/card holds by construction and is tested.

**Rollback strategy.** For hybrid / linear-attention layers, two SGLang paths are modelled:

| Mode | Source | Verify per sequence per layer | Memory |
|---|---|---|---|
| snapshot (default) | `intermediate_ssm_state_cache` + `fused_mamba_state_scatter` (来源) | (q+3) state traffic | q states scratch |
| replay | ReplaySSM `_replayssm_target_verify`, `--enable-linear-replayssm-spec` (来源) | 1 checkpoint read; replay h̄ = (L−q)/2 ring entries (假设, uniform fill); q ring writes; fold every (L−q)/E steps (state read + write + ring read); compute + (h̄+E) recurrent updates | ring L·HV·((K+V)·4 + 4) B |

- L = max(16, 2q): SGLang's default cache_len is 16, and the KDA backstop requires L ≥ 2 × draft tokens. Applying
  that backstop to GDN is 假设.
- Ring entry size is 近似: hi + lo bf16 parts, an fp32 gate, and K counted per HV head.
- Under replay, rejected tokens cost nothing: their ring entries are never committed.

Comparison at 1P, ctx 4096 (also returned as the what-if row's `spec_rollback_compare`):

| Model | k | Batch | Snapshot TPOT / need | Replay TPOT / need |
|---|---|---|---|---|
| ReaDy (a = 0.828) | 7 | 1 | 4.021 ms / 56.2 GiB | 3.960 ms / 55.1 GiB |
| ReaDy | 7 | 32 | 8.388 ms / 107.2 GiB (**does not fit**) | 6.273 ms / 70.9 GiB |
| qwen3.8-27b (a = 0.7) | 1 | 1 | 9.859 / 52.6 | 9.876 / 52.3 |
| qwen3.8-27b | 1 | 32 | 15.795 / 74.4 | 14.112 / 66.2 |
| qwen3.8-27b | 3 | 32 | 12.762 / 83.8 | 10.425 / 66.2 |

For reference, plain decode (AR) at batch 32 is 21.052 ms / 63.8 GiB.

Conclusion: at batch 1 the modes are within about 1.5%. At batch 32, replay recovers most of the speculation loss
that 0.222 introduced: ReaDy goes from 3815 to 5101 tok/s and fits on 1P again. So the 0.222 conclusion "hybrid
spec at large batch is unattractive" holds only for the default snapshot path. The rollback strategy is a
first-order design lever for hybrid targets.

**Static dispatch.** The ×2 applies only to the target's own draft modules. An independent draft model's passes
are separate programs, charged by its own evaluation, so a step is 1 + k programs.

Deferred:
- ReplaySSM KDA fold variant (`replayssm_spec_fold`) details.
- Radix-track forced flushes.
- Tree drafting.

## §171 ReplaySSM from SGLang code (GDN cycle, KDA fold), radix tracking, tree drafting (0.224)

**ReplaySSM.** Re-sourced from SGLang main, `kernels/ops/attention/fla/gdn_replayssm_spec_decode.py` and
`kda_replayssm_spec_decode.py`.

| Item | 0.223 | 0.224 | Basis |
|---|---|---|---|
| GDN history per verify | (L−q)/2 假设 | mean-field run of `_advance_gdn_spec_cursors_kernel` at E (`ir.replay_cycle`) | 来源 rule, 推导 mean |
| GDN fold | E/(L−q) per step | flush steps × write_pos records; exact sequential fold | 来源 |
| History cost in verify | full state update per record | output-only reconstruction ∝ hist·q·(dk+dv) | 推导 from the kernel structure |
| KDA | same as GDN | fold-every-commit: state read + read/write, raw ring, E replays | 来源 (`replayssm_spec_fold`, `commit_kda_replayssm_after_verify`) |
| Ring length | max(16, 2q) | pow2 ≥ max(16, 2q) | 来源 (assert + config) |
| Other recurrences | replay | snapshot + warning | 来源 ("fall back to the recurrent verify kernel") |

**Radix tracking** (`mamba_track_interval`, SGLang default 256) adds a state copy every interval tokens. Its effect
is small, under 0.2% at 256. It is off by default because whether tracking runs in a given deployment is 假设.

**Tree drafting.** Verify covers N tokens. The tree mask computes N × (ctx+N), the same as a causal kernel over N.
- Draft steps: MTP and independent drafts extend topk branches after step 1, following SGLang EAGLE draft_forward.
- Hybrid layers use the snapshot path with N intermediate states, because SGLang ReplaySSM is linear-chain only.
- Acceptance is a 假设 lever: a_t = 1 − (1−a)^min(topk, (N−1)/k).

Results at 1P, ctx 4096, a = 0.7, TPOT in ms, chain k3 → tree k3/t4/N16 → tree k5/t8/N32:

| Target | Batch 1 | Batch 32 |
|---|---|---|
| qwen3-32b + 0.6b draft | 8.26 → 5.51 → 4.30 | 17.50 → 20.62 → 37.36 |
| qwen3.8-27b (GDN, MTP) | 7.48 → 4.95 → 3.74 | 12.76 → 14.62 → 18.36 (memory 83.8 → 139.9 → 214.8 GiB) |
| glm-5.3-flash (MoE, KDA, MTP) | 6.32 → 10.29 → 10.89 | 43.64 → 34.28 → 29.07 |

Conclusions:
- Trees win at batch 1 on dense and hybrid targets, but only under the 假设 acceptance model.
- Trees lose at batch 32 on dense and hybrid targets. Verify of N tokens is no longer weight-bound, and hybrid
  layers pay N state copies.
- MoE reverses the picture: tree verify of N tokens touches many more experts, so trees lose at batch 1 and win at
  batch 32.

Deferred to 0.225:
- UI fields for tree and radix tracking.
- Sourced tree acceptance data.
- Dynamic tree / EAGLE-3 draft structures.

## §172 Tree-drafting acceptance: published references, bounds, chain-vs-tree what-if (0.225)

**References (来源, `data/spec_tree_refs.json`).**
- Matched pairs (same draft model, chain vs tree): EAGLE-1 arXiv 2401.15077v3 Table 5, MT-bench, T=0.
  - Vicuna 7B / 13B / 33B: τ 3.20→3.94, 3.23→3.98, 2.97→3.68.
  - LLaMA2-Chat 7B / 13B / 70B: τ 3.00→3.62, 3.18→3.90, 3.12→3.81.
  - Tree: `mc_sim_7b_63` (EAGLE `choices.py`), 5 deep, nodes per depth 4/8/8/3/2, 26 verified tokens. It contains the
    top-1 chain.
  - The published tree gain is +20.7…+23.9 % τ.
- Unmatched points (reference only):
  - EAGLE-2 dynamic tree (depth 6, top-10, 60 tokens): τ 4.98 (Vicuna 7B), 4.53 (LLaMA3 8B).
  - SGLang PR #4247 EAGLE-3 on Llama-3.1-8B: steps 5 / topk 8 / N 64 τ 4.68; steps 8 τ 5.39.

**Check (no fit).**
- a is the i.i.d. per-draft acceptance that reproduces the chain τ at k = 5.
- The model tree τ is evaluated at the published tree shape (k 5, topk 4, N 26) and compared with the published tree τ.
- `indep`, 1 − (1 − a)^min(topk, (N−1)/k) (0.224 「假设」): **+49 … +63 %**.
  - Why it is optimistic: greedy verification accepts at most one child, and lower-ranked candidates match less often.
    Siblings are therefore neither independent nor equally likely.
- `chain`, τ_tree = τ_chain (structural lower bound, 「来源」 because the tree keeps the top-1 child): **−17 … −19 %**.
- Every published pair lies inside the [chain, indep] band.
- The default stays `indep`: changing it would move every tree KPI. Each tree evaluation now carries the band and a
  warning, and `serving.spec_tree_accept_model = chain` selects the lower bound.
- A rank-dependent sibling model (p_j for the j-th candidate, with Σ p_j ≤ 1) would be the mechanism fix. No per-rank
  source was found, so it is deferred.

**Depth cap (fix).** A tree with N verified tokens is at most N − 1 deep, so τ uses min(k, N − 1) drafts.

**Chain-vs-tree, 1P, ctx 4096, a = 0.7, chain k3 vs tree k3 / topk 4 / N 16. TPOT in ms as chain | tree indep / tree chain-bound:**

| Model | Batch 1 | Batch 8 | Batch 32 |
|---|---|---|---|
| qwen3-32b + 0.6b draft | 8.257 \| 5.511 / 8.597 | 10.345 \| 8.374 / 13.064 | 17.502 \| 20.616 / 32.163 |
| qwen3.8-27b | 7.479 \| 4.946 / 7.716 | 8.672 \| 6.729 / 10.498 | 12.762 \| 14.621 / 22.809 |
| glm-5.3-flash | 6.315 \| 10.293 / 16.058 | 25.569 \| 26.858 / 41.9 | 43.64 \| 34.282 / 53.482 |

- Under the lower bound the tree never wins: it has the same τ and more verify and draft work.
- The 0.224 conclusions held only under the upper bound:
  - "Tree wins at batch 1–8 for the dense / hybrid targets."
  - "Tree wins at batch 32 for glm-5.3-flash."
- With the published EAGLE-1 gain (≈ +22 %, 推导, illustration only, not in the model):
  - qwen3.8-27b: the tree still wins at batch 1 (≈ 6.3 vs 7.48 ms), ties at batch 8, and loses at batch 32.
  - qwen3-32b + draft: wins at batch 1 and loses from batch 8.
  - glm-5.3-flash: ties at batch 32 (≈ 43.8 vs 43.64).
- So the crossover moves from between batches 8 and 32 to about batch 8 or lower.

**Design only (deferred).**
- EAGLE-2 dynamic trees: the per-request tree is reranked by draft confidence. Cost model:
  - draft: k passes of topk candidates;
  - verify: N tokens;
  - acceptance: needs a confidence → acceptance calibration, which EAGLE-2 Fig. 6 shows only as a plot.
- EAGLE-3: multi-layer feature fusion plus a training-time test. For this cost model it changes acceptance only; the
  draft head cost is about one decoder layer, as in EAGLE-1/2.

## §173 Tree acceptance: chain headline, band, Sequoia rank model (0.226)

**Default.** `serving.spec_tree_accept_model` changes from `indep` to `chain`.
- `chain` is a guaranteed lower bound. The tree contains the same-depth top-1 chain: EAGLE `choices.py` and the SGLang selection always keep the top-1 child.
- Every tree evaluation reports the band [chain, indep] and the rank value as τ and TPOT at the same step time. The headline is the selected model.
- Re-baseline: the default fingerprints do not move, because no default preset uses `spec_topk` > 1. Tree scenarios move from the indep values to the chain values (CHANGELOG 0.226).

**Rank model.**
- Positional acceptance (Sequoia, arXiv 2402.12374, Def. 3.1, 「来源」 as an assumption of that paper): the acceptance of a child depends only on its rank among the siblings.
- Vector (App. E, Llama3-70B-Instruct target, Llama3-8B-Instruct draft, CNN DailyMail, temperature not stated): P = 0.7732, 0.1039, 0.0402, 0.0206, 0.0128, …, 31 entries.
- Scaling to the scenario's top-1 acceptance a 「推导」: p₁ = a and p_j = P_j · min(a/P₁, (1−a)/(1−P₁)).
  - The first term scales every rank with a. The second keeps the shape of the miss mass.
  - The minimum keeps Σp ≤ 1 for every a, and at a = P₁ it equals the published vector.
  - Taken alone, the first form exceeds the indep bound at a ≥ 0.8, and the second exceeds it at low a.
- Tree, following SGLang `select_top_k_tokens` 「来源」:
  1. The first step expands topk children of the root.
  2. Each later step expands the topk best leaves by topk children each and keeps the topk best.
  3. The verified tree is the N−1 best-scoring candidates.
  - Under positional acceptance the draft's confidence order is the same as the acceptance order 「假设」.
- τ = 1 + Σ_v f(v), where f(v) is the product of rank acceptances along the path (Sequoia Prop. 3.4).
- The result is clamped to [chain, indep]. It is ≥ chain by construction: the selection maximises Σf, and the chain is a candidate subtree.

**Check against the EAGLE-1 matched pairs** (no fit, tree k5 / topk 4 / N26):

| Model | Error vs published |
|---|---|
| chain | −19.3 … −17.1 % |
| **rank** | **+0.8 … +3.6 %** |
| indep | +49.1 … +62.8 % |

- On EAGLE's actual static tree (`mc_sim_7b_63`, scratch `sd221/rank226.py`), Sequoia's F(T) with the same vector gives −2.6 … +0.1 %.
- Applying the rank vector through the old depth-width approximation (w = min(topk, (N−1)/k) children under every accepted node) gives +14 … +18 %. Tree structure, not only the per-rank vector, drives the 0.224 overestimate.
- Why `rank` is not the default:
  - It rests on one vector, measured with a standalone 8B draft rather than an EAGLE head, on CNN DailyMail rather than MT-bench, at an unstated temperature.
  - The scaling to a is 推导.
  - It is offered as an option and always shown inside the band.

**Conclusions (1P, ctx 4096, a = 0.7; TPOT in ms as chain-k3 | tree k3 / topk 4 / N16 at chain / rank / indep):**

| Model | Batch 1 | Batch 8 | Batch 32 |
|---|---|---|---|
| qwen3-32b + 0.6b draft | 8.257 \| 8.597 / 7.166 / 5.511 | 10.345 \| 13.064 / 10.89 / 8.374 | 17.502 \| 32.163 / 26.811 / 20.616 |
| qwen3.8-27b | 7.479 \| 7.716 / 6.432 / 4.946 | 8.672 \| 10.498 / 8.751 / 6.729 | 12.762 \| 22.809 / 19.013 / 14.621 |
| glm-5.3-flash | 6.315 \| 16.058 / 13.386 / 10.293 | 25.569 \| 41.9 / 34.927 / 26.858 | 43.64 \| 53.482 / 44.581 / 34.282 |

- Headline (chain): a tree never beats the same-depth chain. It has the same τ and more verify and draft work.
- Rank:
  - The tree wins only at batch 1 for qwen3-32b + draft (−13 %) and qwen3.8-27b (−14 %; −20 % for k5 / topk 8 / N32).
  - It ties or loses from batch 8 (qwen3.8-27b +0.9 %), so the crossover lies between batch 1 and batch 8.
  - glm-5.3-flash never wins (+2 % at batch 32). The 0.224 "tree wins at batch 32 for glm" disappears.
- Indep: the 0.224 conclusions, kept as the upper bound only.

**Deferred.**
- A per-depth acceptance matrix (Sequoia Algorithm 4 for EAGLE-like drafts, whose acceptance decays with depth). No published matrix values were found.
- EAGLE-2 dynamic-tree confidence calibration.

## §174 Power-capped operation: clock / voltage scaling and per-lever cap binding (0.227)

**Why.** A TDP decides whether a bandwidth or capacity lever pays off. A lever that lifts throughput by moving more DRAM bytes also lifts W/card, and under a cap that gain can vanish.
- 0.189 throttled by duty cycling and said nothing about clock or voltage.
- The what-if never said whether a cap binds.

**Mechanism.** P(f) = E_core(V)/t(f) + E_other/t(f) + P_static, with W ∝ C·V(f)²·f (arXiv 2607.00819 eq. 1, 「来源」).
- **Time.** The step is re-evaluated at chip.freq_ghz·s.
  - Compute, vector, SFU and SRAM-port time scale as 1/s.
  - DRAM bandwidth (from mem_id), links and host do not.
  - So a DRAM-bound decode must drop its clock a long way before time stretches.
- **Energy.**
  - Core-rail actions (mac, vec, sram, slc) scale with v².
  - DRAM array, PHY, D2D, link, network and host bits are held.
  - Static power (idle, SRAM leakage, HBM standby/refresh) is held. This is conservative, because leakage also falls with V. (SRAM / SLC leakage now ∝ V, §205.4.)
- **Search.** Bisection on s ∈ [0.15, 1].
  - 0.15 is the settable minimum clock of A100 (0.21/1.41 GHz) and H100 (0.345/1.98 GHz), 2607.00819 Table 2, 「来源」.
  - If P(0.15) still exceeds the cap, the 0.189 duty cycle removes the rest.
  - If static power ≥ cap, the point is infeasible.

**Curves.**
- `dfs` (V fixed, headline).
  - Energy per action is unchanged, so the time that fits the cap is E_dyn/(cap − P_static). That is exactly the 0.189 duty-cycle throughput.
  - DFS on its own therefore buys no throughput. All of the DVFS gain comes from voltage.
- `rx5700xt_n7` 「二手」.
  - Source: AMD RX 5700 XT (TSMC N7) default OD_VDDC_CURVE from an amdgpu `pp_od_clk_voltage` dump (sibradzic/amdgpu-clocks issue #39): 800 MHz 704 mV, 1409 MHz 819 mV, 2019 MHz 1185 mV.
  - As V/V_max against f/f_max: (0.396, 0.594), (0.698, 0.691), (1, 1); flat below the first point.
  - Cross-check, not used: H100 board power turns superlinear at f_t ≈ 1005 MHz ≈ 0.51 f_max (2607.00819, FIRESTARTER). The curve is nearly flat over 0.40–0.70 and steep above, which is consistent with a voltage floor around mid-clock.
  - No second modern V–f table was found; H100 and Jetson expose none. It stays a secondary estimate next to the dfs headline.

**Results** (1P, tpuv4i_7nm preset + idle 20 W; cap as a fraction of nominal P):

| Case | Nominal | Cap 0.9 P: dfs / V–f | Cap 0.7 P: dfs / V–f | Cap 0.5 P: dfs / V–f |
|---|---|---|---|---|
| qwen3-8b decode b64 | DRAM-bound, 135.5 W | ×0.883 (s 0.36) / ×0.976 (s 0.40) | ×0.648 / ×0.717 | ×0.413 / ×0.457 |
| qwen3-8b prefill b8 | MAC-bound, 258 W | ×0.892 / ×0.962 | ×0.675 / ×0.874 | ×0.458 / ×0.765 |
| qwen3-32b decode b128 | DRAM-bound, 153 W | ×0.885 / **×1.000** (s 0.75, still DRAM-bound) | ×0.655 / ×0.788 | ×0.425 / ×0.511 |

- Compute-bound prefill gains most from voltage (×0.765 vs ×0.458 at half power).
- Memory-bound decode keeps full throughput under a mild cap if the core voltage can drop.
- Under dfs, the clock must fall to about 0.3–0.4 before power drops 10 %, because DRAM energy per byte does not scale.

**What-if under a cap** (`power_cap_W`; lever energy basis is the tpuv4i_7nm preset at each lever's cost batch; base under the same cap at the same batch):
- qwen3-32b b64, cap 150 W:
  - Only `dram_bw_x2` binds (241 W): gain ×1.22 under dfs and ×1.35 under V–f. It still works.
- qwen3-8b b64, cap 120 W:
  - `dram_bw_x2` binds at 228 W: gain ×0.974 under dfs and ×1.07 under V–f. **It stops working under the cap.**
  - `hbm_12h24g` and `hbm_16h24g` drop to ×1.04 and ×1.03, and `two_die` to ×0.94.
  - `works_capped` = kv_fp8 and ctx_half only.
- Conclusion shift: without a cap, six of seven levers "work". Under a 120 W cap, the bandwidth and capacity levers lose their value, and only the levers that cut bytes (fp8 KV, shorter context) still pay.

**Fix.** The 0.189 cap now treats SRAM leakage and HBM standby/refresh as static (static was idle only). This is a small effect at 1P (3.09 W): throughput ×0.700 → ×0.692 at 0.7 P.

**Deferred.**
- A second independent V–f source.
- Leakage scaling with V.
- HBM / PHY DVFS: the memory clock is held.
- A per-chip V–f curve field.
- (b) prefill/decode interference.
- (c) residuals: no new sources found.

## §175 Per-cycle dynamic energy: clock tree / always-clocked registers (0.228)

**Gap.** Up to 0.227 every dynamic energy term was per action (MAC, vector op, SRAM/DRAM/link bit). Static terms were per card-second: idle, SRAM leakage and HBM standby. At fixed V a clock-only frequency cut leaves energy per action unchanged. When the op rate is set by memory (decode), it also leaves the action rate unchanged. So no modelled power fell with f, and 0.227's dfs needed s ≈ 0.36 to shed 10 %. Real chips pay frequency-proportional power every cycle, even when stalled: the clock tree, ungated pipeline registers and control.

**Model.** `pJ_clk_mac_cycle` is energy per MAC slot per core cycle. P_clk = macs × f × pJ per card, charged over the whole window, busy or stalled: J_clk/unit = P_clk × cards × window / units. It is dynamic: under DVFS it scales with V² (core rail) and with f through the re-evaluated chip. It is not static, so it is left out of `static_share` and of the duty-cycle cap's static. Duty cycling gates the clock in its off-period, so clock energy per unit is unchanged there. Clock-only DFS cuts the number of cycles per unit when the step time is memory-set. Hence dfs ≥ duty, with equality when compute-bound.

**Value (`tpuv4i_7nm` preset), 0.1009 pJ 「推导/二手」.**
- H100 SXM measured idle power: 121.7 W with a CUDA context (P0 clocks, no work) and 71.8 W without one. The second figure is already the source of `idle_frac_tdp`, which is why the two terms do not overlap: idle_frac_tdp is the idle-clock floor.
- The difference is 49.9 W. Divided by 989.4e12 / 2 MAC/s (132 SM × 2048 MAC/clk × 1.83 GHz), that gives 0.1009 pJ per MAC-slot-cycle.
- This is an upper-side proxy. The difference also includes the HBM P-state step and the GPU's non-tensor logic per slot (CUDA cores, register file, L1), which an NPU of the same MAC count has less of.
- At full MAC load the term is ≈ 19 % of MAC + clock power.
- Cross-checks (not fitted):
  - 65 nm DNN accelerator, with clocking (incl. DLL) at 5.5 % of active power and phase distribution at 1.2 %. This is clock-gated and does not count local trees separately (NSF PAR 10300339).
  - CPU clock distribution at 28 % and latches at 46 % with no gating (Bose/Martonosi/Brooks, HC17 tutorial), and clock-related power ≈ ½ of an Intel CPU (Brooks et al., IEEE Micro 2000).
  - AccelWattch (MICRO'21) GV100 NANOSLEEP power rises with f (Fig. 2). Its P_const of 32.5 W at f → 0 is static and is not this term.

**Effect (1P).** P_clk is 52.9 W/card at nominal f. qwen3-8b decode b64 goes from 135.5 to 188.3 W, with clock at 28 % of J/token.

Under a cap, memory-bound decode keeps full throughput down to 0.9 P on clock alone (s ≈ 0.64), against 0.883 at s 0.36 before. At 0.7 P it gives ×0.816 dfs and ×0.996 V–f. Compute-bound prefill is unchanged.

What-if: faster-step levers now save energy per token. RWKV `dram_bw_x2` moves from +0.7 % to −6.3 %. Under a cap the base binds earlier, so cap gains of bandwidth levers shrink: qwen3-32b `dram_bw_x2` goes from ×1.22 to ×1.03 under dfs.

**Limits.**
- The value is a GPU proxy.
- Idle MAC arrays are treated as clocked. Fine-grained clock gating would lower the stalled share; that is left out.
- The PD report's energy (`pdqueue._energy`) does not yet carry the term.
- The memory clock is held.

## §176 Clock gating of idle MAC arrays (0.229)

**Model.** P_clk = macs × f × pJ_clk_mac_cycle × (g + (1 − g)·u). Here g = `Chip.clk_ungated_share`, the share of clock power that is never gated: the global tree, gating cells and control. u is MAC utilisation over the window, computed as useful MAC count ÷ (macs × f × window × cards) and clamped to [0, 1].

The term stays dynamic. Under DVFS it scales with V² and f, and the re-evaluated chip keeps g. At u = 1 it equals the 0.228 term. At u = 0 it is g times that.

**Sources for g.**
- **Eyeriss v1** (JSSC 2017, Fig. 16 and Table V; 65 nm, post-P&R on the measured workloads) 「推导」. The chip clock-gates zero-ifmap cycles and unused PEs.
  - CONV1: clock network 109.2 mW, multipliers and adders 29.5 mW.
  - CONV5: clock network 106.2 mW, multipliers and adders 7.1 mW.
  - Arithmetic activity falls ×0.24 while clock power falls only 2.7 %, which gives g ≈ 0.96.
- **Eyeriss v2** (JETCAS 2019, Fig. 22) 「二手」: the clock share is about 21 % at high utilisation and about 55 % on a bandwidth-limited layer.
- **Bose/Martonosi/Brooks** (HC17) 「推导」: a 60 % saving at 20 % utilisation implies g ≈ 0.25 under aggressive fine-grained gating (CPU, pre-silicon).

**Default g = 1.**
- The `tpuv4i_7nm` value is H100 clocked power at u ≈ 0, so it already is the stalled floor of a gated chip.
- The accelerator evidence (Eyeriss) shows that clock power does not follow activity.

Lowering g is a design claim (fine-grained gating of the array clock tree). It is exposed as a lever, not a default. Effects at g = 0.25 are listed in CHANGELOG 0.229. In short:
- memory-bound decode power drops about 20 % (1P qwen3-8b b64: 188 → 151 W);
- clock-only DFS loses part of its 0.228 advantage (0.9 P: ×1.000 → ×0.941);
- the RWKV bandwidth lever's energy saving shrinks (−6.2 % → −1.3 %).

**Limits.**
- u counts MAC work only; vector lanes and SRAM ports are not gated separately.
- The model is linear in u. Measured data show only the two end points.
- The PD report energy path does not carry the clock term yet (deferred).

## §177 PD report energy at parity with the main report (0.230)

`pdqueue._energy` prices a mode's action counts per output token, together with the card-seconds per output token at the load: cards / λ / out_len, wall-clock, busy or idle. It now applies the same terms as `energy_report`, each pool with its own chip and memory.

**Dynamic terms.**
- Per-kind DRAM energy plus the chip-side PHY line.
- The per-cycle clock term, charged over every provisioned card-second. A serving card is clocked whether or not a request is in flight, which is the H100 "context, no work" state the preset value was measured in.
- Gating: g + (1 − g)·u, with u = MAC count ÷ Σ_pools(macs·f·card-seconds).

**Static terms.**
- `idle_W`, otherwise `idle_frac_tdp × tdp_W`. `idle_W_prefill` still applies to the PD prefill pool.
- SRAM leakage: the state SRAM always, and the shared SRAM only without an idle figure.
- HBM standby/refresh, only without an idle figure.

**Approximations.**
- Heterogeneous PD memory: all DRAM bytes are priced with the decode pool's memory kind.
- One pooled utilisation is used for gating across pools.
- Both closed in 0.257 (§203.3): per-pool counts, memory kind and gating utilisation, plus the `J_by_pool` ledger.

**Invariant.** A single operating point priced with card-seconds per unit = cards·window/units reproduces `energy_report` term by term, and J/token × tok/s/card = W/card holds.

**Effect.** PD-report J/token rises ×1.5–2.1 for qwen3-8b 1P (see CHANGELOG 0.230): mostly the clock term, plus the idle fraction × TDP that was previously dropped. The ranking between modes is unchanged.

## §178 Chunked-prefill engine profiles for the colocated chunked mode (0.231)

The colocated chunked mode is a fused, decode-first iteration scheduler (Sarathi-Serve / vLLM). The prompt tokens one iteration carries now follow `pd.chunk_profile`.

**Profiles.**
- **`vllm` (default, 「来源」).** A token budget b (`max_num_batched_tokens`) covers the running decodes first, and the remainder goes to the head-of-line prompt. So C(k) = max(0, b − k) at running batch k.
  - b comes from the card memory, as the vLLM API-server defaults in `engine/arg_utils.py`: < 70 GiB → 2048, ≥ 70 GiB → 8192, ≥ 160 GiB → 16384.
  - `pd.chunk_budget` overrides b.
- **`vllm_v0`.** b = 2048, the V0 docs default.
- **`legacy_512`.** C = 512, the ≤ 0.230 「假设」.
- **`fixed`.** C = `chunk_tokens`.

**Analytic model.** In `_coloc_chunked_base`, every quantity that depends on C is evaluated at the batch k of the birth-death chain:
- the chunk plan (n chunks, the last chunk's share, the prefix re-read) and the chunk-iteration rate ν(k) = λ·Σw·n(k);
- the fused iteration time and the mean step T̄(k) = T₀/(1 − ν(k)(T₁(k) − T₀)).

The chunk count and the prompt's excess service follow the operating batch k_s by interpolation. If C(B) ≤ 0, the decodes fill the budget at the batch limit, prompts make no progress, and the mode is reported unstable.

**DES.** `pdsim.ColocChunked` reads C(len(running)) at each iteration. If C = 0 with decodes running, it runs a decode-only iteration.

**Why vllm is the default.** The mechanism fixes C as an engine setting with a documented default. A larger C means fewer fused iterations per prompt and less re-reading of the prompt's own KV (each chunk re-reads the earlier chunks). That lowers TTFT. TPOT moves little, because a decode iteration still pays the chunk's compute once per prompt.

**Approximation (「近似」).** One prompt per chunk iteration; packing several short prompts into one budget is not modelled.

**Deferred.** SGLang's default (no mixed chunk) runs prefill-only chunk iterations that pause decodes. That needs a separate non-fused mode.

## §179 SGLang chunked prefill: the non-fused profile (0.232)

SGLang without mixed chunk (`enable_mixed_chunk = False`, the default) runs a prefill batch whenever one can be formed, and re-admits a partly prefilled request first. A prompt of n new tokens therefore runs as ⌈n/C⌉ consecutive prefill-only iterations, and no decode iteration runs between them. The queue is the prefill-first queue (§ prefill-first: an M/G/1 or bulk prefill server, decode in the reduced clock with batch arrivals); only the prefill service changes.

**Prefill service.**
- T(1, S, p) = Σ_j T_prefill(S_j = min(S, p + (j+1)C), cached prefix p + jC).
- Each chunk is an engine prefill with the earlier chunks as cached prefix. So it re-reads all weights and the earlier KV, and its activations are those of a C-token pass. Activations that spill at the full prompt length can stay on chip.
- A batch packs b prompts only if b·(S − p) ≤ C.

**Decode pause.** Prefill-first's stall (the prefill busy period) now contains the chunk iterations. The decode share is 1 − ρ_prefill with ρ computed on the chunked service, and itl_max ≥ the stall.

**Energy.** The summed action counts of the chunk evaluations.

**Chunk size C.** chunked_prefill_size by memory per card (`memory_hook.py`): < 20 GiB 2048, < 60 GiB 4096, < 160 GiB 8192, ≥ 160 GiB 16384. `pd.chunk_budget` overrides it.

**`sglang_mixed`.** `--enable-mixed-chunk` subtracts the running batch from the chunk budget and fuses the decodes in. This is the §178 vLLM rule with budget = chunked_prefill_size.

**Approximation (「近似」).** A prompt's tail chunk is not packed with the next prompt.

**Known gap (→ 0.233).** The fused path's chunk iteration scales the full prompt's non-weight DRAM by f. For long prompts that spill activations this overstates chunk cost relative to the per-chunk evaluation used here.

## §180 Fused chunk iterations evaluated per chunk (0.233)

A fused chunked-prefill iteration at running batch k carries the k decodes plus the prompt tokens [q, e).

**Before (≤ 0.232).** T = fuse(dec_k, f·prefill(S, p)) with f = C/new, plus a re-read of the earlier chunks' KV averaged over the prompt. The whole prompt's non-weight DRAM was scaled by f, and that included its activation spill. A long prompt that spills activations at full length therefore charged a share of that spill to every chunk, even though a C-token chunk fits on chip.

**Now.** T = fuse(dec_k, prefill(e, cached q)) (`pdqueue._chunk_fused`). The chunk is a prefill in its own right:
- its GEMMs and attention cover C tokens against q + C keys, and its activation footprint is that of C tokens;
- its kv_read is the prefix KV;
- its weights are read in the same iteration as the decodes (stage-level fusion, as before).

**Where it is used.**
- Prompt service: Σ_j T(dec_k, chunk_j).
- Worst gap: the longest chunk iteration (the last, with the most prefix).
- Energy: Σ the chunk evaluations' counts, each with its weight DRAM removed (shared with the decode iteration).
- DES and analytic use the same function, memoised per (k, S, p, q, e).

**Invariants.**
- C ≥ new gives one chunk equal to the plain prefill, the same as the ≤ 0.232 formula (f = 1, re-read 0).
- Without spill, the summed chunk cost is ≥ the one-shot cost (weight and prefix re-reads).
- With spill, chunking can lower DRAM traffic and energy.

**Considered and dropped.** Interpolating chunk boundaries on a 256-token grid was 4–6 % off at GEMM tile boundaries.

## §181 Multi-prompt packing within one iteration budget (0.234)

**Source (「来源」, `sd221/v234`, vLLM main `v1/core/sched/scheduler.py` and `config/scheduler.py`; SGLang `schedule_policy.py`).**
- **vLLM V1 `schedule()`.** `token_budget = max_num_scheduled_tokens`, which defaults to `max_num_batched_tokens` (the §178 tiers).
  - The running requests go first: a decode takes 1 token, a running partial prefill takes min(left, budget).
  - Then the waiting queue is served FCFS `while token_budget > 0`. It breaks at `len(running) ≥ max_num_active_reqs`, which defaults to `max_num_seqs` (modelled as the slot count B, including the requests that are prefilling).
  - Every admitted request takes min(new, budget), so the last one is chunked.
  - `long_prefill_token_threshold` defaults to 0, which disables the cap; it is not modelled.
  - `max_num_partial_prefills` / `max_long_partial_prefills` were V0-only and do not exist in V1 main (n/a).
- **SGLang `PrefillAdder`.** The chunked request first; then whole requests while they fit `rem_chunk_tokens`; the first one that does not fit is truncated (`_select_prefill_admission`) and becomes the next `chunked_req`. Bounded by max_running_requests; page alignment is ignored (「近似」).
- **One rule for both engines** (`pack_fill`): an FCFS fill of the iteration's prompt budget R, where R = budget − k for vLLM / sglang_mixed and R = C for sglang with the decodes paused. Under FCFS at most one prompt is partial (「推导」).

**Iteration cost.**
- `_fused_multi`: the k decodes (or, with none, the largest piece) plus several prompt pieces. Compute, non-weight DRAM and link add; the weights are read once; sync = max.
  - Exact in the stage and serial modes.
  - pipe: the one-piece op replay (§ 0.102) × the stage/serial bracket ratio of all pieces to that piece (「近似」, exact for one piece).
- `_packed_fused`: whole same-shape prompts run as one batched prefill (as prefill-first batches them). A partial piece is its own chunk with its prefix cached (§180).
- legacy_512 and fixed do not pack.

**Analytic chunk server at running batch k** (`pack_at`). C(k) = budget − k; k = 0 has the full budget, as in the DES. b = min(⌊C/new⌋, B − k).
- **b ≥ 2.** The greedy bulk queue M/G^[b]/1 (§ 0.65 `bulkq`), with D_j = the packed iteration of the k decodes + j prompts.
  - Busy share per request = bulk_busy/λ. Iterations per request = 1/E[batch].
  - itl_max = the largest packed batch met by ≥ 1 % of requests over the ν·life chunk iterations.
  - The truncated head of a (b+1)-th prompt in the leftover is not modelled (「近似」).
- **b = 1 (tail sharing).** A prompt's last iteration (τ tail tokens, τ < C) is topped up with the next prompt's head if one waits when the iteration starts.
  - π_i = 1 − E[e^{−λW}]·e^{−λS′_i} with E[e^{−λW}] = (1 − ρ)/E[e^{−λS}] (Pollaczek–Khinchine LST at s = λ), where S′_i = the iterations before the tail. Solved as a fixed point with ρ (「近似」: idle decode-only iterations are left out).
  - This gives ≈ ρ for multi-chunk prompts and ≈ ρ² for one-iteration prompts. π = ρ was tried and overstated sharing 2.4× (C: 0.151 vs DES 0.064, before the k = 0 fix).
  - Server share when shared = Σ full chunks + τ/C·T_sh. Own latency = Σ full chunks + T_sh: the prompt finishes at the end of the shared iteration, because the head is processed in the same iteration. Using the token share as latency was 27 % optimistic at p90 (D sglang_mixed).
- **TTFT.** The 8 quasi-static batch bins.
  - Bins with b ≥ 2: mixture of bulk_law ⊕ U(0, decode step) (`_law_mix`).
  - Otherwise: the M/G/1 mixture with a two-point (share, latency) service per prompt type.
  - The decode birth–death uses the effective share and iteration count per request.

**DES.** `ColocChunked._packed` applies the same fill (slot admission = max_num_seqs). sglang now runs its own packed prefill-only replica.
- Invariants (test_core_234): fill ≤ budget (including the k decodes), and every running decode is credited in every vLLM iteration (decode-first, never starved). At most one partial prompt. Packing occurs at load 0.85.
- Parity is inside the V4 bands for A, D (vllm, sglang_mixed), A at 0.85 and a fuzzed misaligned case. C, E: see CHANGELOG 0.234.

**Effect.** Prefill is compute-bound, so a packed iteration of j prompts takes ≈ j × as long, and the first prompt finishes later.
- The TTFT tail grows: A p99 82.9 → 99.5 ms; D sglang_mixed 84.3 → 103.7 ms.
- SLO goodput moves ≤ ±1.1 %.
- Packing cannot close prefill-first's lead in C/E: a 16384-token prompt fills the budget, so only the k-token tail is shared.

**Known gap (→ 0.235).** The SGLang non-mixed analytic still uses the prefill-first bulk caps without tail sharing. The DES has tail sharing, so the two differ when new mod C ≠ 0 and the prompt has more than one chunk. Cases A–E are aligned and unaffected.

## §182 Tail sharing for SGLang non-mixed, and the bulk regime's extra head (0.235)

**SGLang non-mixed** (prefill-first machinery on the `_ChunkPool` view, §179). PrefillAdder fills a prefill-only iteration with C tokens FCFS, so a chunked prompt's last chunk (τ = new − (n−1)C tokens, τ < C) shares its iteration with the next waiting prompt's head (§181).
- At batch cap 1, each prompt type with 2·new > C becomes a two-point service:
  - **alone** (1 − π_i): share = latency = Σ chunks;
  - **shared** (π_i): share = Σ_{j<n} t_j + τ/C·T_sh, latency = Σ_{j<n} t_j + T_sh.
- T_sh is the prefill-only iteration of the tail plus a head of C − τ tokens, k = 0, from `_packed_fused`.
- π_i = 1 − (1−ρ)/E[e^{−λS}]·e^{−λS′_i}, where S′_i = Σ_{j<n} t_j (§181, now shared as `_tail_rows` / `_tail_pi`).
- Cap ≥ 2 (whole prompts packed, b·new ≤ C) keeps the bulk caps.
- The expanded share law drives the M/G/1 wait, the busy-period batches handed to decode, the TPOT excess variance and the itl busy duration. TTFT = W ⊕ the latency atoms (`mg1_sum_quantile`).
- 「近似」: the follower's head start is not credited to its latency. Parity is in band but +2–3 % at p90 off multiples.

**vLLM bulk regime, the (b+1)-th head.** At b = ⌊C/new⌋ ≥ 2 the leftover is L = C − b·new.
- A full batch that starts with more than b waiting also carries L tokens of the next prompt. Its probability is h = P(n > b | n ≥ b) from the bulk chain at service starts.
- The finishing prompts' own wall O_b becomes two-point (wall_b, wall_{b+head}).
- D is kept: the work is token-conserving, since the head leaves the later batch (「近似」).
- itl_max uses the head wall when P(full)·h·(chunk iterations per life) reaches 1 %.

**Slots.** The DES caps running + prefilling at max_num_seqs, matching b ≤ B − k in the analytic. V1 has no max_num_partial_prefills, so FCFS gives at most one partial prompt.

## §183 The follower's head start (phase chain) and the bulk head moving work (0.236)

**Phase chain.** Under FCFS packing (vLLM V1 waiting loop, SGLang PrefillAdder), prompt tokens are cut into C-token iterations as one continuous stream.
- A prompt that starts inside its predecessor's topped-up last iteration starts at phase φ = the predecessor's prompt tokens in that iteration.
- Its pieces are then C − φ tokens, followed by C-sized chunks, with the last one carrying `used(φ)` tokens. The next prompt's phase is `used` when somebody is waiting at its last-iteration start (probability π), and 0 otherwise.
- **Shares:** a piece in a full or topped-up iteration takes tokens/C of the wall. The prompt that ends an iteration without a head takes the remainder (the unused budget is its own).
- **Own latency** runs from its share start, which is the predecessor's part of the first iteration, to the end of its last iteration.
- **π:** π(φ, i) = 1 − (1 − ρ)/E[e^{−λS}]·e^{−λ·pre(φ, i)}, the P-K LST rule of §181/§182. It is solved jointly with the stationary phase law.
- The predecessor's and next prompt's pieces take the prompt's own shape (「近似」; exact for one prompt type).
- **Cost limits (「近似」):**
  - fused running-batch bins k ≥ 1 keep followers at φ = 0, because C(k) and the walls change with k; the chain runs at k = 0 and on the SGLang prefill-only path;
  - phases met by < 0.2 % of prompts fold into φ = 0;
  - mixtures use a C/16 lattice;
  - with more than 2 prompt types (length bins), followers stay at φ = 0 (the §182 rows);
  - at most max(4, 32/types) latency-quantile atoms per type.
- **Multiples of C:** used = C, so there is no sharing and everything matches §181.
- **The credit.** At 6000 tokens and C 4096 the phases run 0 → 1904 → 3808 → 1616.
  - The φ = 1904 follower leaves a 288-token head, against the leader's 2192, so its topped-up overshoot is far smaller.
  - The DES follower share matches within 0.01 (0.149 vs 0.155 at load 0.85).
- **TTFT impact.** The impact on the TTFT quantiles is ≤ 0.012. The 0.235 "+2–3 % p90 pessimism" was n800×2 sampling noise; at n1500×3 the error is +0.01 … +0.03.

**Bulk head moves work** (b ≥ 2, §182).
- **Head batch.** The (b+1)-th prompt's head L = C − b·new goes into the head batch's wall, so D_b is two-point at probability h. h = P(n > b | n ≥ b at a service start) is taken from the bulk chain by fixed point.
- **Next batch.** The next full batch starts with the head-started remainder new − L plus (b − 1) whole prompts, at probability g = h (「近似」: independent of h).
- Its leftover 2L is capped below one more whole prompt, so a batch still has b completions.
- O = D: each prompt's own wall is its batch's.
- **Slot-bound.** No head when k + b = max_num_seqs; this matches pdsim admission.
- **itl_max:** the head wall counts when P(full)·h reaches the ≥ 1 % criterion; the remainder + head wall uses P(full)·h².

**Deferred.**
1. **Per-class b.** M/G^[b]/1 has a single batch cap. Under FCFS token packing, per-class caps depend on the sequence of queued types, which is variable-capacity bin packing with no embedded chain on the queue length. The DES covers it.
2. **Slot-bound b = 1 sharing** (k ≥ B − 1) is still modelled, because the itl_max k = B bin relies on that iteration.
3. **Phase chain for fused k ≥ 1 bins and length-bin mixtures** (cost).

## §184 Wait-correlated tail sharing in the TTFT law (0.237)

§182–§183 combined the prefill wait W and the prompt's own latency L as independent variables: P(T > t) = Σ w_i P(W > t − L_i). With tail sharing, two parts of L depend on W itself.

**1. Topping.** The prompt's last iteration is topped up with the next head iff an arrival came behind it before that iteration started, i.e. during W + S′ (FCFS: later arrivals wait behind it).
- Given W = w: π(w) = 1 − e^{−λ(w+S′)}.
- E_W[π(W)] = 1 − E[e^{−λW}]·e^{−λS′} = the §182 π, by P-K: E[e^{−λW}] = (1−ρ)/E[e^{−λS}].
- The marginal (and so the mean) is unchanged.

**2. Follower status.** A prompt starts inside its predecessor's last iteration (phase φ > 0, §183) iff that iteration had not started at its arrival.
- That means a prompt was waiting ahead of it (n_q ≥ 1), or the one in service was still before its last iteration.
- In share accounting this is W > θ, the predecessor's last-piece share. It is exact for deterministic shares; for M/D/1, P(W > D) = P(N ≥ 2).
- θ is the topped classes' last-piece share, weighted by m·π.

**The law.**
- P(T > t) = Σ_i w_i Σ_{r∈i} ∫ dF_W(w) g_r(w) [e^{−λ(w+S′_r)} 1{w + a_r > t} + (1 − e^{−λ(w+S′_r)}) 1{w + c_r > t}].
  - g_r(w) = 1{w ≤ θ}·m_r/L_i for leader rows, and 1{w > θ}·m_r/F_i for follower rows.
  - A type without follower rows keeps g = m_r/L_i.
  - a_r and c_r are the latency alone and topped; L_i and F_i are type i's leader and follower masses.
- W is the exact P-K law (`_pk_surv`) on a 1500-point grid. Suffix sums of P(W ≥ x) and E[e^{−λW}; W ≥ x] make each survival evaluation O(rows).

**Evidence.**
- DES top-1 % TTFT is 88 % followers (sglang 2500, load 0.85).
- A Lindley Monte Carlo of the topped model matches the closed form (p99 62.5 vs 62.7); the independent convolution gives 57.0.
- sglang 2500 / 3000 p99: −0.149 / −0.118 → +0.007 / −0.001 (n4000 × 10).

**Scope.** The SGLang prefill-only path. Fused b = 1 environments (vLLM, sglang_mixed: mg1_mix_sum_quantile over batch bins) keep the independent form (deferred; vllm 3000 p99 −0.051, sglang_mixed 6000 p99 −0.056).

## §185 Wait-correlated TTFT in the fused chunked paths (0.238)

Extends §184 (0.237) from the SGLang prefill-only path to the fused paths (vLLM, sglang_mixed: `_coloc_chunked_base`, b = 1 tail-sharing environments).

- **Why the fused paths need it most.** The running decodes take k tokens: C(k) = budget − k 「来源: vLLM / SGLang mixed-chunk scheduler」. At k ≥ 1 a prompt of 4096 tokens (a multiple of the nominal budget) is off-multiple, so its tail always shares an iteration with the next head and a follower can start inside it. 0.236 kept the follower phase chain only at k = 0 for cost (§183), so at the "multiples" the analytic tail missed both correlations.
- **Per batch bin b (the 8 quasi-static bins of 0.55):**
  - The follower phase law at the bin's running batch k_b, rounded to a multiple of 8 (≥ 1) 「近似: C(k) changes < 1 % per 8 tokens of 2048」. It gives the server-share atoms (the wait law), the rows (mass, latency alone, latency topped, S′, type, follower) and θ_b (the topped classes' last-piece share).
  - The wait is W_b ⊕ U(0, v_b) (decode-only iteration in flight when the prompt queue is empty, Fuhrmann–Cooper as in 0.55).
  - Topping given W = w and U = u: π = 1 − e^{−λ(w + u + S′)} 「推导: arrivals during the vacation residual also queue behind the prompt」.
  - Follower iff W > θ_b, on the M/G/1 part only 「近似: an arrival that waits out only a vacation found an empty prompt queue, so it is a leader」.
- **Mixture:** P(T > t) = Σ_b ⅛ · E_u[surv_b(t; u)], with u at 8 midpoints. `surv_b` is §184's survival (`queueing._corr_surv`) with S′ + u and t − u. Bins without sharing reduce exactly to the 0.55 independent form.
- **Mean:** taken from the same bin atoms, so followers at k ≥ 1 are included.
- **Cost:** ≤ B/8 phase laws per arrival rate, with walls memoized across rates. The birth–death mean step keeps the 0.236 φ = 0 approximation at k ≥ 1. Fused off-multiple KPI-type cases take 12–14 s.
- **Evidence (DES n4000 × 10, p99):** vllm 4096 −0.052 → +0.002; sglang_mixed 4096 −0.082 → +0.006; vllm 6000 −0.032 → −0.021. vllm 3000 and sglang_mixed 6000 stay at −0.044 / −0.052. There the topping correlation (+) and the follower correlation (−, followers' topped latency is shorter than leaders') cancel, and the residual matches the sglang 6000 one that §184 showed is not a sharing effect. Open.

## §186 Numerics of the correlated M/D/1 wait; chain depth vs wait (0.239)

- **M/D/1 in `_corr_surv`.** Erlang's alternating series is used only for waits ≤ `MD1_EXACT_SPAN` = 6 τ. Beyond that the Cramér–Lundberg tail c·e^{−θx} is used (capped at ρ) 「推导」, as in `mg1_mix_sum_quantile` (0.55). At ρ = 0.95 the series has no correct digit left at 40 τ (true P(W > 40τ) = 0.0165). In 0.237–0.238 this corrupted the fused TTFT quantiles near saturation: case E's SLO rate came out 17.37 in 0.238 instead of 19.76.
- **Near saturation (E, DES n4000 × 10).** The DES holds TPOT p90 at ≈ 23 ms, and TTFT absorbs the load. The analytic decode birth–death lets TPOT p90 climb to 50 ms instead, so the analytic SLO rate is 4–7 % optimistic (DES ≈ 18.43 sglang_mixed, ≈ 18.70 vllm). Open.
- **Chain depth vs wait (open, evidence).** A follower's phase is the residue of the chain of followers before it. Long waits sit deep in a busy period, where the phases are deep-chain ones: small first pieces and three iterations. In the DES (sglang 6000), the first-follower phase is 89 % of followers below the median wait but 6 % in the top 5 %. §184 uses the stationary phase mix for every W > θ.
  - This explains the ≈ −5 % p99 residual at sglang / sglang_mixed 6000 and vllm 3000.
  - The derived remedy is the joint (phase, W) Lindley chain over successive prompts. Not implemented yet.

## §187 max_num_seqs caps a request's fused time (0.240)

- **Bound.** Prompts are admitted only while running + prefilling < B 「来源: vLLM V1 max_num_seqs counts RUNNING incl. prefilling; SGLang max_running_requests」. In one decode life a request sees at most B admissions: the ≤ B − k free slots at its birth, plus one per death of the ≤ k older requests (equal-length lives finish in order).
  - Fused time in its life: t_on ≤ B·S_B, carrying n_on ≤ B·n_B tokens. S_B and n_B are the chunk server's per-prompt busy time and iterations at batch B.
  - TPOT_req ≤ (B·S_B + (out − B·n_B)·T_d(B))/out, or S_B/n_B when B·n_B ≥ out 「推导; mixed prompt types: weighted S_B, n_B and mean out 近似」.
- **Use.** Caps the request TPOT p90/p99 (`tpot_cap`). The mean is untouched: it is never above the cap.
- **Evidence.** Case E DES n4000 × 10: TPOT p90 is flat at 23.3–23.4 ms (sglang_mixed) and 22.9–23.0 (vllm) from λ = 18 to 19.76. The cap gives 24.0 / 22.9–23.1.
- **Open.**
  - Near-saturation TTFT is optimistic ×1.2–1.4. Slot-blocking time is only 2–3 %. The DES chunk iterations see a batch of 18 vs the analytic 13.2, i.e. the batch grows while prompts queue.
  - Mean TPOT misses the birth-in-busy-period residual R: E[R] = L·E[BP] (DES 0.647 s vs 0.68 at λ = 15). It needs the slot cap and the queue–service correlation at high load.

## §188 max_num_seqs blocks the chunk server in a backlog (0.241)

- **Evidence.** Per-prompt chunk-server share is ≈ deterministic and rises only +2.7 % with the queue (no queue–service correlation), yet the DES wait exceeds P-K on its own service law by +37 % at λ18.6 (E sglang_mixed). Lifting the slot cap in the DES removes most of it (wait 743 → 524 ms); in vllm E lifting it changes nothing (642 vs 636).
- **Fill 「推导; mean-field」.** In a busy period each admission rides n_B fused iterations, so its decodes are n_B tokens apart and none of them dies before out tokens; the k0 decodes running at its start (PASTA: the time-average batch) have uniform residual lives. After j admissions k_j = j + k0·(1 − j·n_B/out)^+; the batch is full after J* admissions.
- **Block.** The next prompt waits for the oldest decode: V = (out − B·n_B)·T_d(B). Then one prompt is admitted per death, all B lives spaced n_B apart, so the next block is B admissions later ⇒ one V per B admissions (same mean as spreading (out/B − n_B)·T_d over each admission, but lumped — the lump is what drives the wait). No block when B·n_B ≥ out.
- **Who meets it 「近似: iid」.** A prompt beyond index J* of its busy period, w.p. 1/B. Index law: Borel(ρ′) (M/D/1, deterministic share), customer-weighted φ = E[(N − J*)^+]/E[N]; ρ′ = ρ + λφV/B by fixed point. Applied as an extra atom τ + V with weight φ/B on every chunk-server share atom (envs, cbins, serv_mix); own-latency rows unchanged. Reported as `slot_block`.
- **Result.** E sglang_mixed TTFT p90 at λ18.6: 1298 → 1461 ms (DES 2163); 19.32: 1641 → 2110 (3353); λ15 unchanged; SLO rate 19.79 → 19.23 (DES 18.43). vllm E unchanged.
- **Open.** The no-cap DES is still ~25 % above the no-block analytic in both profiles (sglang_mixed lift p90 1597 vs 1298; vllm 1849 vs 1488): a second, non-cap mechanism. The iid lump also ignores that blocks cluster in long busy periods. Mean TPOT birth-in-busy residual still open.

## §189 The fused share grows along a busy period: queue-weighted batch law (0.242)

- **Evidence.** Without the slot cap the DES prompt queue is work-conserving: a Lindley replay of its own (arrival, share) sequence matches its wait to ±2 %. The order of the shares still matters: a shuffled replay is ×1.10–1.30 lower at p90.
  - In a busy period every finished prompt adds a decode, so the share grows (vllm E λ18.6: k 7 → 27.5, share 180.5 → 189.0 ms).
  - The long waits queue behind the large-k shares.
  - The analytic share mean is right (ρ 0.857 vs 0.859); the time-average batch law used for the TTFT bins is not.
- **Weight 「推导」.** A share is waited out by every prompt queued during it. In the embedded M/D/1 departure chain (a = λS):
  - g₁ = (1 − a)·Poisson(a), and g_{j+1}(x′) = Σ_{x≥1} g_j(x)·Poisson(x′ − x + 1).
  - Customer j's weight is (x_prev − 1)⁺ + a/2. Index 1 has weight a/2, and the total is L_q + ρ/2.
- **Batch 「推导; mean-field」.** The batch follows §188's deterministic-aging fill after j − 1 admissions: k = min(j − 1, out/n_B) + k0·(1 − (j − 1)n_B/out)⁺ ≤ B. Here k0 follows the time-average law 「近似」.
  - Beyond j − 1 ≥ out/n_B, k no longer depends on j, so the remaining weight is assigned there in closed form.
  - The TTFT quasi-static bins are 8 equal-mass bins of this law (`_queue_k_law`).
- **Slot block (§188) re-weighted the same way.** The block falls only past J*, where queues are long, so φ_q is the queue weight past J* (`_slot_block_q`). ρ′ is a fixed point.
- **Result (E, TTFT p90 vs DES n4000×10).**
  - vllm 17.37–19.32: 1180 / 1440 / 1833 / 2657 vs 1158 / 1436 / 1883 / 2748.
  - sglang_mixed: 1125 / 1400 / 1856 / 2992 vs 1219 / 1603 / 2163 / 3353.
  - C operating-point p99: −12 % → −5 % vs DES.
- **Open.** The slot block is still an iid lump (sglang_mixed −8 … −14 %). The idle-time k0 law is not yet modelled.

## §190 Decode fixed point near the capacity limit; clustered slot block (0.243)

- **Fixed point.** The fused path refines the decode-seen batch with two passes, k ← seen(k). Near λ_s the map overshoots: A vllm λ56.8 went 57.2 → 63.2 → 52.3.
  - The TTFT law was then non-monotone in λ, and the SLO-rate scan found a false feasible window.
  - When the two passes oscillate (sign change, |Δ| > 0.5), the fixed point is bisected between the first two iterates. A decode-unstable pass inside the bracket makes the load unstable.
- **Clustered block 「推导」.** The §188 block V = (out − B n_B)·T_d(B) is met only at a full batch, i.e. only in the quasi-static bins at k = B of the §189 queue-weighted law.
  - In the full regime a cycle is B admissions per out decode tokens. The fill leaves J* busy-period decodes n_B apart and m = B − J* survivors with uniform residual lives, all of which die before the cohort.
  - So m of the B admissions wait V/m each. The atom is τ + V/m w.p. m/B in the full bins only.
  - m = 1 recovers §188's lump. Large m (A vllm) gives small, frequent blocks.
- **Evidence.**
  - DES SLO rates (TTFT p90 2 s crossing, 8 seeds): fused analytic within ±1.4 % in A/D/E/F/G (A vllm 0.242 +1.9 % → +0.4 %; G vllm +2.6 % → +1.2 %).
  - E sglang_mixed TTFT p90 at 18.6: 1856 → 1935 (DES 2163).
- **Open.**
  - sglang non-fused: prefilling counts in max_running_requests, so admission blocks at a full batch with β = (out/e)·T_d(B)/B per admission. That is absent from the prefill-first machinery, which leaves the TTFT ≈ 0.1 s up to λ_s; the analytic is +1.8–3.5 % too optimistic.
  - Idle-time starting batch: derived k0 ≈ DES, but the mean-field fill then under-predicts.

## §191 SGLang non-mixed admission: the slot account in a backlog (0.244)

**Rule (来源).** SGLang `scheduler._get_new_batch_prefill_raw` admits a waiting prompt only while len(running_batch) + len(can_run_list) < max_running_requests, so the prefilling requests hold slots. The prefill-only iteration pauses the decodes. The DES `ColocChunked._packed` applies the same rule.

**Cost per admission (推导).** In a backlog with a full batch, each admission waits for a death. In the saturated cycle one death comes per admission, after β = out·T_d(B)/(e·B) of decode time. So the admission server is S = τ + β, ρ′ = λ_c(τ̄ + β̄), and λ_s = r_c/(τ̄ + β̄), which is the decode limit. The time-average P(n ≥ B) is the wrong weight (0.243 iid try): the blocking happens exactly when there is a backlog.

**Slot account (推导; diffusion 近似).** Define U = (τ + β)·q − β·f.
- **Moves.** A prefill lowers U at rate 1. Decoding at batch k lowers it at rate c(k) = k·T_d(B)·e(k)/(B·T_d(k)·e(B)). An arrival adds S.
- **Drift.** −δ = −(1 − ρ′) for U > 0. For U ≤ 0 it is λβ̄ − (1 − ρ_p)·c(B − f), which turns positive a few slots below full: the soft floor.
- **Variance rate.** λE[S²] above 0: arrivals only, since the deaths replay the births (M/D/c heavy traffic). λE[S²] + λβ̄² below 0, adding the deaths' count noise (M/G/∞).
- **Stationary law.** The density ∝ exp(∫2μ/σ²)/σ² on [−Bβ̄, ∞) gives p_sat = P(U > 0) and U | U > 0 ~ Exp(θ = 2δ/σ²).
- **TTFT.** An arrival at U > 0 waits U + β (its q·τ prefills and q − f + 1 deaths), then its own prefill. So TTFT = (1 − p_sat)·T₀ + p_sat·(lat + β + Exp(θ)), applied after the 0.237 wait-correlated tail rows; this is why F now moves.

**Checks.**
- DES A λ53.5: P(U > 0) at arrivals is 0.282 (model 0.276); the batch at arrival has mean 48 and sd 14.3.
- Long-run DES SLO rates: A 55.05 (analytic 55.04), D 54.95 (55.01), F 42.66 (42.80).
- Below ≈ 1 s the TTFT p90 is under-predicted (A 52: 196 vs 434 ms).

**DES run length.** Near saturation the DES relaxation σ²/δ² (≈ 83 s at A 55.06) is comparable to an n4000 run, so those rates are transient (A vllm 55.95 there, 55.28 long-run). DES rate checks now use 6 seeds × n64000 (`sd244/lbr.py`).

**Open (0.245).**
- Fused vllm/mixed are over the long-run DES in A (56.20 vs 55.28, +1.7 %) and G (49.99 vs 49.02, +2.0 %). The A order of v0 vs vllm/mixed (DES v0 > vllm = mixed > sglang; analytic vllm = mixed > v0 > sglang) is a near-tie and is flagged, not fixed.
- The sglang mixture body at small p_sat.

## §192 The slot account for the fused engines (0.245)

**Rule (来源).** vLLM V1 `Scheduler.schedule()` stops admitting at len(running) ≥ max_num_running_reqs, and the partial prefills are in `running`. So, as in SGLang (§191), prefilling requests hold slots.

**Fused account (推导).** U = (τ + β)·q − d·f, where d = out·T_d(B)/(e·B) is the decode time per death at a full batch.
- An admission's n_B chunk iterations at a full batch carry the decodes, giving n_B·B·e/out deaths. Its decode-only wait is β = d − n_B·T_d(B); β ≤ 0 means no block.
- A chunk iteration lowers U at rate 1. Below 0, the iterations run at batch B − f and give fewer deaths, so each arrival nets β + n_B·T_d(B)·f/B. This makes the floor shallower when n_B is large.
- The diffusion, the noise terms and the TTFT mixture are those of §191.

**Admission share where the backlog lives.** Admissions follow death clumps, so a packed iteration needs b free slots. τ and n_B are therefore the bulk queue's realised share at k = B − b_max (pack_at). A gives 39.5 ms and G 48.1, against DES saturated iterations of 39.5 and 48.0. ρ_c/λ is unusable: its k = B bins carry a b = 1 tail-sharing share with no slot for the follower, so it jumps with the decode fixed point.

**DES (n64000 × 6).**
- Every fused case at its crossing has P(U > 0) ≈ 0.5 and U | U > 0 ≈ 1 s, with TTFT ≈ U + β + lat.
- A/G SLO rates: 55.42 / 49.18 against DES 55.28 / 49.02 (0.244: 56.20 / 49.99).

**Scope and why.** The account applies where C(B) fits ≥ 2 whole prompts. The budget-chunked cases (D/F/E, vllm_v0) keep the §190 clustered block because the account needs τ to ≈ 0.3 %: ρ′ = λ(τ + β) near the crossing makes the SLO rate ≈ 30× as sensitive as τ. The full-batch b = 1 tail-shared iteration is priced ≈ 1 % low (D 45.3 vs 45.8 ms, F 66.8 vs 67.5), which would put D/F at +1.4 / +1.6 % against the block's +0.9 / +0.3 %. With the DES τ the account reproduces D/F (p_sat 0.475 / 0.393 vs 0.517 / 0.474; 1/θ 1.02 / 1.04 vs 1.03 / 1.06 s).

**Open.** Pricing of the tail-shared iteration; E vllm long-run DES.

## §193 The full-slot iteration for budget-chunked fused engines; k̄ (0.246)

**Problem.** §192 kept the budget-chunked cases (C(B) < 2 prompts) on the §190 block. pack_at(B) priced their full-batch iteration ≈ 1 % low (D 45.3 vs DES 45.8 ms, F 66.8 vs 67.5), and the account's ρ′ = λ(τ + β) multiplies that error ≈ 30× in the SLO rate.

**What the DES iteration contains (probe, slots full, queue waiting).**
- Resumed middle chunks, tails topped up with the follower's head, tails alone, and fresh heads.
- The running decodes are 63 for single-prompt iterations and 62 for two-prompt ones (B = 64). See the table in CHANGELOG 0.246.

**Mechanism (推导).**
1. The partial prompt holds a slot (max_num_seqs counts it, §192). Its chunks therefore ride B − 1 decodes at budget C(B − 1). A follower head can only join in a slot that a death freed, so the two-prompt iteration runs B − 2 decodes (`_phase_iters(k_top=B − 2)`).
2. The follower starts at phase φ = the predecessor's tail. Its next chunk is a resumed one (the head's KV is re-read), and its own tail is longer. The account evaluates `_phase_law(chain=True)` at that single bin. pack_at keeps followers at φ = 0 at k ≥ 1 because the chain there costs fresh walls per bin.
3. The tail-share probability is the M/G/1 π, as in pack_at 「近似」. D 0.61 / F 0.73 vs DES 0.62 / 0.74.

τ: D 45.78, F 67.55, E mixed 182.9 ms (DES 45.7–45.8 / 67.5 / 182.5). n_B: D 2.39, F 3.27 (DES 2.35–2.38 / 3.26).

**k̄ (推导).** An admission's chunk iterations at a full slot cap carry k̄ = B − n̄ decodes, where n̄ is the number of prompts in flight. Little's law over chunk-iteration time gives n̄ = λ·lat/(λ·τ) = lat/τ. Hence β = d − n_B·T_d(B)·k̄/B, and the floor term below 0 is unchanged in form. k̄: D 62.75, A 62.73 (DES running k in saturated iterations 62.7 / 62.8).

**Result (long-run DES, 6 × n64000).**
- SLO rate errors are now +0.1 to +0.4 % in A/D/F/G for vllm, sglang_mixed and vllm_v0, and +0.77 % for E mixed. Mean |err| 0.27 % (0.245: 0.50 %, mixed signs). All rankings are kept.
- TTFT p90 at the DES crossing is 1.71–1.89 s (DES ≈ 2.0).
- The §190 block stays only for the non-packing fixed-C path.

**KPI-adjacent pins.** vllm prompt 3000, cap 48, load 0.8 (test_core_238): TTFT p99 72.39 → 71.61 ms, mean 34.02 → 34.03, because the backlog branch replaces the block at small p_sat.

**Open.**
- The free-slot density below 0 (A λ52: DES flat near U = 0, model rising ×2.2 over 16 slots → p_sat 0.114 vs 0.153, sub-1 s p90 low). Planned for 0.247.
- The E mixed residual.
- E vllm DES.

## §194 The floor below U = 0: age structure and the near-full prefill tail (0.247, open)

**Observation.** Below ≈ 1 s the slot account's TTFT p90 is 1.1–1.9× low (A λ52: 181 vs DES 341 ms). The gap peaks at p90 0.2–0.6 s and closes at the crossing. The model's free-slot law has the right mean and spread but the wrong shape near full (p_sat 0.114 vs 0.153).

**Mechanism (来源: DES probes; 推导 pending).**
1. *Age structure.* The uniform-age death credit c(k) overstates deaths in a full batch, which is young, and understates them in a thin one, which is old. DES deaths per second at k 63 / 34.5 / 14.7 are 14.5 / 12.8 / 9.4 against 15.3 / 12.6 / 8.0. The restoring drift near U = 0 is therefore weaker than modelled (Kramers–Moyal: −0.02 to −0.05 vs −0.09 to −0.08 U-s/s), and the variance is state-dependent (≈ 0.12 s near full, 2λd² deep).
2. *Near-full prefill tail.* Many 0.2–0.4 s requests wait for the token budget with free slots (q > 0, f ≈ 5). This is a T₀-branch tail in slow-iteration states that the plain-queue law misses.

**Hook.** `_FLOOR_W` scales μ below 0 (death credit w·uniform + (1 − w)·λ). w = 0.65 fixes p_sat (0.140) but not p90 (219 ms), so it is not shipped.

**Next.** Derive the conditional death rate from the k-process correlation time vs the life, and σ²(f). Then derive the near-full T₀ tail. Validate against the λ 50–55 tables (CHANGELOG 0.247).

## §195 The floor below U = 0, derived: age-structured deaths and the queue-ful entry (0.248)

**1. Age-structured death credit 「推导」** (`_AGE_DEATHS`, default on; below 0 only).
- In token time (iterations), births are Poisson(λ·T_step) and a request dies `out` steps after its birth.
- If the step time depended only on k (a mean clock), deaths per step would be exactly k/n (uniform age, w = 1). The long-range k-feedback keeps the ages uniform (micro-model `sd248/br.py`, mean clock: w = 1.00).
- The real clock is arrival-driven. An admission adds Δ of slow-iteration wall time: fused engines Δ = τ − n_B·T_d(B); SGLang non-mixed Δ = τ, the prefill pause. More births land in that time, so births form short clusters with branching ratio η = λΔ (a Hawkes process, which is not time-reversible). The dying (oldest) member of the running window has cluster neighbours on one side only, so it covaries less with the count. Per-step deaths given k therefore regress toward the mean:
  - w = 1 − (1 − g)(I_s − I₀)/(2 I_s)
  - I_s = 1/(1 − η)² is the cluster Fano factor.
  - I₀ = 1 + λ²p(1 − p)δ²/m_b is the per-step Fano factor, with δ = Δ/n_s on p = n_s·m_b of the steps and m_b = λT_d(m)/(1 − η) births per step.
  - g is the elasticity of T_d(k) at the mean batch m. The long-range feedback amplifies the window variance by 1/(1 − g) at w = 1; the cluster gain 1/(1 − η) cancels against λnT_d/m = 1 − η.
- Checks against the micro-model:
  - Constant T_d/T_c: I_s 2.97 vs 2.91, I₀ 1.185 vs 1.19, w 0.694 vs 0.699.
  - Full micro-model: w 0.87 vs formula 0.84–0.87; I/I_s 1.91 vs 1/(1 − g) 1.92.
  - DES A λ52: per-step w ≈ 0.88; the model gives 0.867.
- Below 0 the death credit becomes c(k)·a(k) with a(k) = w + (1 − w)·m/k, and the chunk-iteration credit n_B·T_d·(k̄ − f)/B is scaled the same way. Above 0 nothing changes, because in a backlog births are paced by deaths and the ages stay uniform. Model w is 0.84–0.88 for A/D/G.
- The effect is small: A λ52 p_sat 0.114 → 0.124 (DES 0.155). At high λ, where m → B, the floor gets slightly deeper. The 0.247 value w ≈ 0.65 was a misreading of the per-second blend.

**2. Queue-ful entry law 「推导; 来源: DES sd248/probe14」** (`_SAT_ENTRY`, default on). This replaces the "T₀ near-full tail" reading of 0.247.
- The 0.247 probe over-counted free slots by the prompts in the in-flight iteration. The DES removes them from `curs` before the iteration ends, a mean of 0.67 prompts. With true f:
  - U ≤ 0 arrivals have TTFT p90 115 ms and p99 218 ms, so the T₀ branch has no tail.
  - The 0.2–0.4 s band is U ∈ (0, 0.3] s.
  - At U > 0 the slots are full (f ≈ 0), so U = S̄·q is a lattice.
- (i) The system enters the queue-ful state with a whole jump S̄ = τ + β (q goes 0 → 1 at f = 0), not by creeping through 0+. So the diffusion's Exp(θ) is the law of U − S̄. DES P(U > 0.3 / 0.5 / 0.8 | U > 0) is 0.520 / 0.297 / 0.122; S̄ + Exp(2.79) gives 0.526 / 0.301 / 0.130, while Exp alone gives 0.433 / 0.248 / 0.107.
- (ii) U counts only queued work. The admission in service is not in U, so an arrival first waits its residual E[S²]/(2S̄) (the M/G/1 S_e). DES wait − U is 43–73 ms at U < 0.2 s, against β + S_e = 30.5 + 35 = 65.5 ms.
- The U > 0 base is therefore lat + β + S̄ + E[S²]/(2S̄) when the entry is slot-bound (A λ52: 79.5 → 184.5 ms). TTFT | U > 0 for A λ52:

  | | p50 (ms) | p90 (ms) |
  |---|---|---|
  | DES | 434 | 968 |
  | model, new base | 433 | 1009 |
  | model, old base | 328 | 905 |

- Not modelled: at U > 0.8 s the DES per-prompt cost is 0.89·S̄.
- *Slot-bound share 「近似」.* The lattice entry needs f = 0: with a free slot, a packed prompt is admitted at the next boundary. When the admission's own chunk iterations free its slot (β → 0), the queue-ful state is the budget FCFS queue at any f, and the M/G/1 diffusion (θ from λE[S²], P–K) and T₀ already price it.
  - In that case S̄ + S_e double-counts. E vllm (16k prompts, n_B 8.2, β 0.1 ms of d 22 ms) gave p90 1346 ms at λ17.37 vs DES 1158.
  - The entry term is weighted by β/d, the share of a slot's credit that must come from decode-only time: A 0.89, D ≈ 0.8, E 0.004. The shipped base is lat + β + (β/d)(S̄ + S_e); A λ52 gives 173 ms.
  - E p90 at λ17.37 / 18 / 18.6 / 19.32: 1204 / 1481 / 1879 / 2736 ms vs DES 1158 / 1436 / 1883 / 2748.

**3. σ²(f) below 0 (open).**
- The true-f Kramers–Moyal estimate (dt 77 ms) is σ² 0.13 near full falling to 0.03 deep, with drift −0.005 at f 0 and −0.04 at f 4–16, crossing 0 at f ≈ 21.
- Arrival jumps (d, or S while the budget is busy) plus deaths at the actual rate with cluster Fano I_s give 0.088 near full and 0.052 deep, which is the wrong shape.
- Plugging the DES KM μ(f), σ²(f) directly into the floor integral gives p_sat 0.208 vs DES 0.155. The coefficients at dt = 77 ms are inflated by sub-dt budget-queue transients, so they are not the diffusion's coefficients either.
- A coarse-grained derivation is needed. Residual at A λ52: p_sat 0.124 vs 0.155.

**Validation (p90 ms; DES `sd247/lr_*`):**

| case | λ | DES | 0.247 | 0.248 |
|---|---|---|---|---|
| A | 50 / 52 / 53.5 / 54.3 | 143 / 341 / 734 / 1035 | 124 / 181 / 545 / 937 | 133 / 258 / 633 / 1003 |
| D | 50 / 52 / 53.5 / 54.3 | 146 / 283 / 660 / 952 | 136 / 165 / 385 / 753 | 143 / 201 / 489 / 835 |
| G | 44.5 / 46.5 / 47.5 / 48.3 | 182 / 457 / 797 / 1179 | 165 / 260 / 560 / 1034 | 177 / 349 / 662 / 1106 |

The crossings (0.246 long-run table) moved from mean |err| 0.27 % to 0.15 % (`sd248/xr_sb.txt`). E mixed went from +0.77 % to +0.14 %.

## §196 State-dependent σ_b²(f) below U = 0 (0.249)

**Derivation 「推导」** (`_SIG2_F`, default on). Coarse-grain U over times much longer than one admission.
- With A(t) arrivals, the decode-only time is t − τ_p·A(t), where τ_p = ρ_p/λ is the prefill share one admission takes. So U(t) = d·A − (deaths carried by chunk iterations) − c·a·(t − τ_p·A) − noise.
- Each arrival therefore moves U by J(k) = d − n_B·T_d·k̄_f·a/B + τ_p·c(k)·a(k): its slot, minus the deaths its own chunk iterations carry, plus the decode-only deaths its pause removes.
  - Near full, J → β + τ = S̄. For non-fused engines this is the 1/(1 − ρ_p) amplification of the batch count by the prefill pauses (0.244 note).
  - Deep, c → 0 and J → d.
- The deaths around that conditional mean replay births that were clustered in token time, with Fano factor I_s = 1/(1 − η)² (§195).
- Result: σ_b²(f) = λ·E[J²] + d·R(k)·I_s, where R(k) = λn_B·T_d·k̄_f·a/B + (1 − ρ_p)·c(k)·a(k) is the death rate in U. E[J²] carries the relative variance of τ.
- Mean drift is unchanged: λJ − c·a reproduces the 0.248 μ(f).
- The density exp(Φ)/σ_b²(u) is integrated with σ² inside the Φ trapezoid. The diagnostic `sigma2_f` is returned.

**Check, A λ52 (per replica λ 13).**

| f | model σ_b² | DES true-f KM (dt 77 ms) |
|---|---|---|
| 0 | 0.122 | 0.131 |
| 12 | 0.116 | 0.109 |
| 24 | 0.109 | 0.086 |
| 36 | 0.098 | 0.071 |
| 48 | 0.082 | 0.049 |
| 60 / 56 | 0.056 | 0.026 |

The 0.248 constant was 0.079. Near full matches; deep the model is high, because a(k) > 1 and c(k) has the shape it has.

**Effect.** TTFT p90 (ms) against DES (`sd247/lr_*`):

| case | λ | DES | 0.248 | 0.249 |
|---|---|---|---|---|
| A | 50 / 52 / 53.5 / 54.3 | 143 / 341 / 734 / 1035 | 133 / 258 / 633 / 1003 | 176 / 399 / 781 / 1152 |
| D | 50 / 52 / 53.5 / 54.3 | 146 / 283 / 660 / 952 | 143 / 201 / 489 / 835 | 165 / 289 / 660 / 1006 |
| G | 44.5 / 46.5 / 47.5 / 48.3 | 182 / 457 / 797 / 1179 | 177 / 349 / 662 / 1106 | 217 / 509 / 834 / 1281 |

- Mean |err| over the 12 points goes from 13.9 % (all low) to 10.2 % (now high, so conservative).
- P(U > 0) at A λ52: 0.124 → 0.188, against DES 0.155.
- Crossings: mean |err| 0.148 → 0.152 % (`sd249/xr_s2.txt`). The fused vllm crossings are now within −0.01 to −0.09 %.
- E (16k): vllm p90 is within −2.4 to +1.3 % of DES. sglang_mixed is optimistic again at −4 to −13 %.

**Residual: the near-full lattice (open).** DES `sd248/raw2_A52.npy` shows:
- P(q = 0, f = 0) = 0.0143. Per slot given U ≤ 0 that is 0.0169, against ≈ 0.013 at f 1–4, so there is a pile at f = 0.
- 97 % of arrivals in that state queue.
- With p = λ·π₀·S̄/(1 − ρ′), which is one M/G/1 busy period per entry, the DES π₀ gives 0.147 against the DES P(U > 0) of 0.155. So the jump entry is right.
- The model's continuous law has no pile (0.0118 at f = 0); with it the jump-entry form gives 0.107.
- The diffusion interface (z_pos = 1/(θσ²), density continuous at 0) gives 0.188.
- Next step: a lattice treatment of the last few slots (π₀ from the f = 0 ↔ 1 balance including excursion returns) with the jump entry.

## §197 The near-full lattice at U = 0: f = 0 boundary state and jump entry (0.250, flag off)

**Boundary state 「推导」.** The free-slot process stops at f = 0 (q = 0, slots full) in a sticky way.
- It is entered by an admission, whose prefill is in flight for one phase τ.
- It is left by a death; deaths release slots one at a time.
- An arrival at f = 0 starts a U > 0 excursion. The excursion uses up the rest of the phase and returns with a fresh admission in flight.

The hazards are:
- outside the phase, decode-only deaths: r_do = c(B)a(B)/d;
- during the phase, chunk-carried deaths: r_ch = n_B·T_d·k̄·a/(B·d)/τ.

The reduced in-flight time per phase is m = (1 − e^{−λτ})/λ. The sojourn per entry is

x = (1 + (r_do − r_ch)m) / (r_do − (r_do − r_ch)λm),

and the f = 0 ↔ 1 balance gives π₀ = λx·π₁.

DES check at A λ52 (`sd248/raw2_A52.npy`, f corrected for in-flight prompts):

| quantity | model | DES |
|---|---|---|
| λx = π₀/π₁ | 1.32 | 1.30 |
| in-flight share at f = 0 | 0.71 | 0.71 |
| in-flight share at f = 1 / 2 / 3 | — | 0.54 / 0.48 / 0.46 |

**Jump entry.** Every arrival at f = 0 opens one M/G/1 busy period, so P(U > 0) = λπ₀S̄/(1 − ρ′). With the DES π₀ this gives 0.147, against DES 0.155.

**Implementation** (`_LATTICE0`):
- atom z₀ = λx·d/σ_b²(0) relative to the continuum density;
- z_pos = z₀·λS̄/δ in place of the diffusion interface 1/(θσ²);
- both are weighted by the slot-bound share β/d, as in §195 (the budget-bound E case has no slot lattice);
- the remainder keeps the diffusion interface.

**Result: not the default.**
- A/D/G p90 mean |err| is 10.7 % low (0.249: 10.2 % high).
- A λ52 p_sat is 0.138, against DES 0.155.
- Crossings mean |err| is 0.260 %, above the 0.2 % goal. E mixed is +1.38 %; without the β/d weight it was +2.47 % (0.411 % overall).

**Residual: the continuum below 0.** The boundary is right; the continuum is not.
- The model's law given U ≤ 0 has mean/sd 27.5/14.8 slots, against DES 24.8/13.1.
- The deep log-slope is 0.11 per slot, against DES 0.29.
- The deep σ_b² is ≈ 0.06, against KM 0.026, and the deep drift is ~25 % weak.
- So the near-full level is 0.0121 per slot against DES 0.013. The correct jump entry from a level that is too low under-shoots.
- A death Fano of 1 would fix the deep σ² (0.033) but would break near-full σ² (0.079 vs KM 0.11–0.13). The Fano I_s should therefore be state-dependent; this is the next step, after which `_LATTICE0` can be re-enabled.

## §198 Catalog freshness sweep 2026-10-19 (0.251)

**Method.**
- `api/models?author=<org>&sort=lastModified` over every catalog vendor org, plus tencent, BlinkDL, Inferact, the unsloth mirrors and TypeSafe variants.
- The window is releases created on or after 2026-10-14 that are not in `releases/`.
- Each candidate's config.json is compared with the nearest entry:
  - **identical architecture** → merged entry (resolvable and validated, not listed);
  - **new fields** → the IR, if they can be expressed mechanism-first; otherwise coverage partial, with the unmodelled fields listed.
- Params must match the safetensors headers exactly.
- A gated repo is read from a public mirror only if every git LFS pointer oid matches (byte-identical).

**Tencent Hy3 (`hy_v3`, Hy-Memory-30B-A3B).** The architecture is a Llama-type GQA decoder, 48 layers:

| part | detail |
|---|---|
| attention | 32 q / 4 kv heads × 128, q/k RMSNorm |
| layer 0 | dense SwiGLU 6912 |
| layers 1–47 | MoE: 128 experts top-8, sigmoid router with fp32 selection bias (`expert_bias`), route_norm, scale 2.826, plus 1 shared expert |

- The expert width is `expert_hidden_dim` = 768 (header). The config's `moe_intermediate_size` = 1536 equals gate + up, so reading it as the width double-counts the experts: 58.2B vs 30.1B.
- The FP8 release is static W8A8 (per-tensor input scales), so activations are fp8 (`_act_fmt`); the KV cache is fp8 per tensor.

**FLUX 3 Image Klein (`Flux3Transformer2DModel`).** Read from the header; no public modeling code yet.
- **Early blocks.** `num_layers` = 5 per-modality blocks. Each stream (text, image, reference image) has its own q/k/v/out projections, QK-norm and SwiGLU (3072 → 9216), and attention is joint over the streams. This generalises FLUX.2's dual-stream block to three streams.
- **Single blocks.** `num_single_layers` = 28 shared blocks, with separate q/k/v/out projections and SwiGLU.
- **Modulation.** Per-modality tables for early and single blocks, plus norm_out.
- **Flow-map distillation.** The distilled checkpoint adds `target_timestep_embedder` (the jump end time). Base lacks it.
- **Text conditioning.** Qwen3-VL-4B hidden states of layers 4, 8, …, 32 are concatenated (8 × 2560 = 20480 → context_embedder).
- **Text-to-image** runs the text and image streams. The reference stream (0.62B params) holds no tokens, so it is storage only.
- **Workload** (README / config):
  - 1024², VAE ×8 with 2×2 packing → 4096 image tokens;
  - distilled: 16 steps, no guidance;
  - Base: 50 steps; guidance 3.5 with no guidance embedder ⇒ true CFG 「推导」;
  - text 320 tokens 「假设」.
- **Coverage partial.** The edit / multi-reference stream (≤ 10 reference images) is not modelled. Single blocks are counted as serial attention → MLP; this does not affect FLOPs or bytes.

**Muse Spark Open NVFP4 (§175 update).** The public mirror's `hf_quant_config.json` excludes layers 0 and 61 (and attention and lm_head) from NVFP4.
- 2 × 256 experts stay bf16 (25.8B params).
- Bytes: 533.0 GB, against the 0.175 推导 of 496.0 GB.
- Expert effective bits: 4.87, against 4.50.

**Same architecture, merged.**
- UserRepGen-4B is the Qwen3.5-4B architecture (= StartLux-Decision-v1-4B). It publishes no MTP head and no preprocessor_config.json, so images use the Qwen3.5 base processor 「假设」.
- flux-3-image-klein-pup is the Qwen3.5-9B architecture, with no MTP head.

**Not added.**
- Timestral (time series).
- scFM-dataselection (single-cell ablations).
- Qwen3-8B-DSpine-b16 (DFlash draft: block-diffusion drafting plus a selected-prefix code bridge, not expressible on the DSpark draft path; deferred).
- GGUF-only repos and third-party repacks.
- Jev and Microsoft-Decision-1 have no official weights.

## §199 Closing the 0.251 coverage gaps: FLUX 3 reference images, DSpine draft (0.252)

### §199.1 FLUX 3 Image Klein reference-image edit / multi-reference

**Sources 「来源」.**
- BFL reference code, github.com/black-forest-labs/flux-3-image-klein, `src/flux3/{dit.py, pipeline.py, sampling.py, generate.py, autoencoder.py}`.
- diffusers main, `src/diffusers/models/transformers/transformer_flux3.py`.
- The 0.251 header reading is corrected in two places:
  1. The early ("mode") blocks are **per-mode self-attention**, not joint attention. Text attends to text, the image to the image, and the reference images (all references of one sample together) to themselves.
  2. Every block is a **parallel block**: `x + gate·(attn_out(attn(x_mod)) + mlp_out(SwiGLU(mlp_in(x_mod))))`, with one LayerNorm + modulation.

**Reference stream.**
- **Encoding.** Each reference is VAE-encoded once per request by the fp32 AutoEncoder (`encode_references`, one image per call), then 2×2 packed (16 px per token) and embedded by `x_embedders.image_cond` (128 → 3072).
- **Timestep and RoPE.** The reference timestep is 0. `image_rope_scale` 0.5 halves the h/w position ids of the image and of the references, and the 4th RoPE axis is −(1 + reference index). Both are RoPE only and change no FLOPs.
- **Mode blocks.** `mode_blocks.image_cond` runs per-mode attention over the reference tokens of a sample.
- **Single blocks.** Joint attention over [image, image_cond, text], with `single_stream_modulations.image_cond` applied to the reference rows.
- **Modulation per token** (as in the BFL reference; diffusers uses a per-sample time MLP):
  - the `time_in` MLP runs on the per-token timesteps;
  - the mode modulation is applied per token;
  - `ref_time_in` runs per sample;
  - the final adaLN covers image tokens only.
- **Flow map.** On the distilled model, the target-timestep embedding is added to every mode.
- **Reference sizes 「来源」 (pipeline.py).**
  - `MAX_REFERENCES` = 10.
  - One reference: fitted to a ≤ 4 MiB-pixel canvas on a 16-px grid, sides 64–4096 (padded canvas encoded).
  - Two or more references: each is ≤ 1 MiB pixels, with sides floored to 16.
  - Examples: 3 × 3000×2000 → 1248×832 each; 1 × 4000×3000 → 2352×1760.

**Caching.**
- **Not cached 「来源」.** `denoise` / `denoise_flow_map` call the DiT on the full pack every step, and diffusers Flux3 has no KV-cache argument (the `kv_cache_mode` matches belong to FLUX.2 / Qwen-Image). The reference rows are therefore recomputed at every step.
- **CFG on Base.** Base true CFG packs [uncond, cond], each with the references, so the references count twice per step. They are still encoded only once.
- **No cache knob.** Because "no caching" is sourced, there is no cache knob.
- **Deferred what-if design 「推导」.**
  - On Base, the image_cond mode-block outputs are step-invariant (t = 0, attention within the references only), so they could be cached exactly. That removes 5 of the 33 blocks for the reference rows.
  - On the distilled model the target-time embedding modulates them, so a cache would not be exact.
  - The reference rows in the single blocks attend to the noisy image and are never exactly cacheable.

**Implementation.**
- **IR.**
  - `Phase.ref` with row tag `"ref"`.
  - AttnCore `span="mode"`: separate attention per group, with a single QK-norm/RoPE and SP all-to-all over all tokens.
  - `Layer.fused_out`: the parallel block uses one TP all-reduce 「推导」.
- **`build_flux3`.**
  - Three streams in the mode layer and a fused single layer.
  - IO terms: `ref_embed` / `ref_mod` / `single_ref_mod`; `time_1` / `time_2` per token; `norm_out_mod` per image token.
  - `io_misc` raises on any unmodelled GEMM.
  - Every header tensor is accounted for (param_check is exact: 5,548,681,984, and 5,538,458,368 for Base), so **coverage is full**.
- **Workload.**
  - `workload.ref_images`: 0 = T2I; at most 10, otherwise an error.
  - `ref_width` / `ref_height`: 0 = the output size.
  - CLI: `--ref-images` and `--ref-size WxH`.
  - UI: a reference row shown only for FLUX 3.
- **Pipeline.** `vae_encode` adds encoder ops from the encoder tensors (`flux3-vae.json` `enc_tensors`: 111 tensors, 34.4M params). It costs 4.89 TFLOP and 56 ms per 1-MP image, serial, and is added to the request latency (`enc_s`, emitted only when > 0).

**Numbers** (default chip, output 1024², references 1024² 「推导」):

| refs | 0 (T2I) | 1 | 2 | 4 | 10 |
|---|---|---|---|---|---|
| distilled, 16 steps | 7.79 s | 16.82 s | 28.70 s | 59.52 s | 212.24 s |
| Base, 50 steps, CFG | 47.23 s | 103.27 s | 177.38 s | 368.67 s | 1320.79 s |

- T2I against 0.251 (7.61 s / 46.45 s): FLOPs are +1.3 %. Per-token modulation adds FLOPs, and per-mode early attention removes some.
- Attention grows with seq², and seq is 4416 → 8512 with one reference.

### §199.2 tencent/Qwen3-8B-DSpine-b16 (`qwen3-8b-dspine`)

**Sources 「来源」.**
- HF tencent/Qwen3-8B-DSpine-b16 (sha 9fc2ab3, 2026-10-16): config.json plus the safetensors header. It has 71 tensors, 1,237,763,844 params (bf16 + 4 I64 scalars) and is 2,475,527,712 B. Embeddings and the lm_head are shared with the target.
- github tencent/dspine, `sglang_dspine/{models/dspine.py, worker.py, kernels.py}`.
- arXiv 2609.36173.
- The 0.251 verdict ("not expressible") is withdrawn: DSpine fits the DSpark block-draft path.

**Mechanism.**
- **Draft input.** `fc` over the concatenated hiddens of target layers [1, 9, 17, 25, 33] (20480 → 4096), then `hidden_norm`.
- **Draft layers.** 5 Qwen3 layers (GQA 32/8 × 128, q/k norm, SwiGLU 12288). Attention is **causal within the block** (`causal = (B+1)/(2B)` of full).
- **Block.** B = 16 = one anchor row + 15 drafts, so k = 15 is forced. The reference implementation rejects other block sizes, and the model warns if a different k is set.
- **Code bridge,** after each active layer:
  1. RMSNorm, then the read GEMM h → C = 1024, then L2-normalise. The anchor row takes `codebook[anchor]·√C`.
  2. The packed GEMM C → 3R = 1536.
  3. The gated message `sigmoid(q_i + s_{i−1} + b)·v_{i−1}`.
  4. The write GEMM R = 512 → h, added to the residual.
  5. `out.weight` is all zeros (the terminal is disabled), so it is stored and never run.
- **Tail.**
  - The target lm_head runs on the 15 draft rows, followed by top-K (K = 16).
  - `serial_table` gathers K·(h + 3R) bf16 per row, with K·h + K²·R MACs.
  - The greedy `serial_chain` follows.
  - There is no Markov or confidence head.
- **Resident memory.**
  - Cold codebook V·C = 311 MB.
  - Load-time tables TG + TSV = V·3R·2 B = 466.8 MB 「推导」 (`build_serial_tables`).
  - Both sit in the memory plan as draft storage.
- **TP.** TP = 1 only (reference); a warning is raised for TP > 1.
- **Acceptance.**
  - The model card reports τ 7.83 (T = 0) and 6.98 (T = 1) with 15 drafts, averaged over 7 benchmarks. The τ convention is committed tokens per verify, as for ReaDy.
  - The i.i.d. per-token values are a = 0.893 and 0.873 「推导」.
  - `spec_accept` stays a 「假设」 lever (default unchanged). Setting it to 0.893 reproduces τ 7.82.

**Numbers** (default chip, decode, batch 1, ctx 4096):
- Qwen3-8B TPOT: 82.0 ms.
- DSpine, k = 15:

| a | TPOT | τ |
|---|---|---|
| 0.8 | 20.75 ms | 4.86 |
| 0.873 | 14.45 ms | 6.98 |
| 0.893 | 12.90 ms | 7.82 |

- The draft pass is ≈ 56 GF per sequence; the bridge + tail is 1.26 GF (2 %).
- `serving.spec_k` is now capped at 16 (was 8).

**Deferred.**
- The sampled-round `serial_rs_table` (fp32, K·(2RC + HR + KH) MACs per row) is not modelled; only the greedy path is.
- DFlash context rows written per step are taken as ≈ 1, as for ReaDy.
- The phantom token of an empty FLUX 3 mode is ignored.
- `ffn_norm` is still double-counted for parallel blocks (negligible).
- The FLUX 3 reference mode-output cache (what-if above).

## §200 Freshness sweep from 2026-10-15; looped LM; DSpine sampled rounds (0.253)

### §200.1 Sweep

**Method.** Same as §198, with a wider org list (≈ 80 orgs, adding nvidia, Tongyi-MAI and reflection). Window: created on or after 2026-10-15.

| release | created | verdict |
|---|---|---|
| nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 / -MXFP8 / -NVFP4 | 10-16 | **added** (`dory1.0-4b`, `-mxfp8`, `-nvfp4`), new mechanism §200.2 |
| deepseek-ai/DeepSeek-V4.1-Flash-1017 | 10-17 | already an alias of deepseek-v4.1-flash since 0.248 (config identical; index total_size = stored bytes); unchanged |
| Tongyi-MAI/MAI-Gym-9B | 10-16 | merged → qwen3.5-9b (GUI-agent fine-tune; config identical, no MTP head, same image-processor values) |
| nvidia/Muse-Spark-Open-816B-A42B-DFlash2 | 10-16 | deferred (§200.4) |
| nvidia/Nemotron-3.5-Rerank-8B, microsoft/scFM-dataselection | 10-15/16 | out of domain (ranking, single-cell) |
| unsloth/* (FP8 / NVFP4 / GGUF repacks) | 10-15…17 | third-party, not added |

- **Pre-window.** reflection/Beam-1.0-501B-A23B (10-07, + FP8 / NVFP4 10-10, DFlash2 10-13) is a candidate for a later sweep.
- **Decision models.** Jev and Microsoft-Decision-1 still have no official weights. startlux, RWKV and BlinkDL have nothing new.

### §200.2 NVIDIA Nemotron-Labs Dory (looped normalized-GPT)

**Sources 「来源」.** HF config and safetensors headers (BF16 / MXFP8 / NVFP4), the model card, and vLLM PR #60149 (`models/dory.py`, `configs/dory.py`).

**Structure.**
- 72 physical layers, each holding **one** sublayer:
  - attention (`layer_types` sliding 1024 / full every 4th attention layer), or
  - SwiGLU MLP (7168).
- The layers are grouped 16 / 40 / 16 sublayers = 8 / 20 / 8 layers (input / recurrent / output).
- Attention: GQA 16 / 4 × 256. A `g_proj` sigmoid output gate is fused into q (out_gate). nGPT qk-norm is applied after RoPE, and justnorm before `o_proj`.
- MLP: `suv` scales, with justnorm before `down_proj`.
- Residual: nGPT slerp with per-channel alpha.
- No final norm. `lm_head.logit_scale` (131072) is stored but not counted in the release params.

**Looping.**
- The recurrent block runs `n_recurrent_loops` times per token (config: 2), and the input is reinjected each loop: `justnorm(state + input)`.
- The weights are stored once but **re-read on every loop**.
- Implementation: `with_loops` expands input + recurrent × loops + output. The copies set `Layer.loop > 0`, so their params are 0 and they have no storage, but their ops still run and stream weights.
- **KV.**
  - `per_loop` (vLLM default): one cache per loop.
  - `last_loop`: `kv_shared` copies own no KV storage, while KV reads and writes are unchanged.
  - With `last_loop`, the KV equals the one-loop size at any loop count, as the card states.
- **Knobs.** `workload.loops` (0 = config) and `workload.loop_kv`. The UI shows them only for Dory.
- **Parallelism.** vLLM raises on TP ≠ 1 and Dory has no PP support, so the model warns.
- **Formats.** MXFP8 is W8A16 (e8m0 per 32 → 8.25 b). NVFP4 is W4A16 (4.5 b). The embedding and lm_head stay bf16. Activations are bf16.
- **Exactness.** param_check is exact: 3,975,098,368.

**Numbers 「推导」** (default scenario, bf16, batch 1, ctx 4096):

| loops | 1 | 2 | 4 |
|---|---|---|---|
| decode TPOT | 39.1 ms | 58.9 ms | 98.5 ms |
| KV, per_loop | 0.264 GB | 0.411 GB | 0.705 GB |
| KV, last_loop | 0.264 GB | 0.264 GB | 0.264 GB |

- The weights are 7.95 GB at every loop count.

**Approximation 「近似」.** nGPT justnorm and slerp count as the generic per-layer vector work. They are elementwise, so their FLOPs and bytes are negligible next to the GEMMs.

### §200.3 DSpine sampled rounds and published acceptance (closes the §199.2 deferral)

**Sampled rounds 「来源」.** sglang_dspine `worker.py` `DSpineDraftSampler.sample` plus `kernels.serial_rs_table` / `serial_sample_chain`. A round where any request samples (temperature > 0, top_k > 1) runs, eagerly after the graph:
- per draft row, for each of the K senders: e = √C·codebook[sender] → `pred_gate`, `prev` (R × C) → gated message → write (H × R) → refined state → RMS-norm → dot with the K candidates' lm_head rows.
- Cost: K·(2RC + HR + KH) MACs per row in true fp32 (TF32 off); 51.4 M per row for Qwen3-8B-DSpine.
- The K codebook rows and K lm_head rows are gathered per row.
- Then the K × K sample chain runs (temperature, top-k / top-p / min-p).
- The greedy chain still runs inside the graph.

**Knobs and memory.**
- `serving.spec_sampling` = greedy (default) | sampled.
- **Default change (mechanism, 「来源」 `DSpineDraftSampler.__init__`).** The fp32 copies made at init (`pred_gate`, `prev`, `gate` bias, last write, norm) are resident: (2RC + R + HR + H)·4 = 12.6 MB. DSpine resident tables go from **466.8 → 479.3 MB**.

**Published acceptance.**
- `serving.spec_accept_src` = manual (default; `spec_accept` stays the 「假设」 lever) | published.
- With published, the draft's card τ is used at the round temperature: T = 0 7.83 → a 0.8933; T = 1 6.98 → a 0.8731 「推导」, giving τ 7.83 / 6.98 back.
- Drafts without a published τ at that temperature keep `spec_accept`, with a warning. ReaDy has only a T = 0 τ.

**Numbers.** Default chip, b1, k = 15:

| setting | TPOT |
|---|---|
| greedy, a 0.7 | 30.354 ms |
| sampled, a 0.7 | 30.378 ms |
| published T = 0 | 12.877 ms |
| published T = 1 | 14.456 ms |

### §200.4 Deferred: Muse-Spark-Open DFlash2 draft (nvidia)

**Header (「来源」).** 4,384,162,432 params, bf16:
- `fc` 40960 → 8192 over target layers [3, 19, 31, 47, 59];
- 5 non-causal SWA-2048 Qwen3-style layers (128 / 16 × 64, q/k norm, SwiGLU 24576);
- per layer, `attention_conv` and `mlp_conv`: `base_kernel` [2, 2, 8192] plus `kernel_projection` 8192 → 2048, i.e. an input-dependent kernel-2 conv over block positions, `conv_group_size` 16;
- `candidate_selector`: `hidden_projection` 8192 → 256, and predecessor / successor codebooks [V, 256] — the "pairwise path selector over each position's top-k" (selector_top_k 16);
- block 8 (7 drafts), final logit softcap 20, `output_multiplier` 0.177.

**Published acceptance (card; k = 7, vLLM, NVFP4 target, GB300 TP4 + EP4, concurrency 32, T = 0 / 1).**

| benchmark | T = 0 | T = 1 |
|---|---|---|
| MT-Bench | 3.87 | 3.49 |
| SPEED-Bench qualitative | 3.84 | 3.47 |
| SPEED-Bench 32k | 3.35 | 3.03 |

**Why deferred.** There is no public modeling code for the conv projector or the selector: vLLM main `qwen3_dflash.py` has neither, and the card points to ModelOpt. Their ops cannot be derived mechanism-first.

> **Correction (0.254, §201.3).** This was wrong. vLLM main `vllm/model_executor/models/qwen3_dflash2.py` has had both the conv projector (`DFlashGroupedConv`) and the selector (`CandidateSelector`) since PR #52816 (2026-08-21). 0.253 had searched only `qwen3_dflash.py`. The draft is now modelled to the design below (§201.3).

**Design when the code lands.**
- Base it on the DSpark block path, the same as DSpine:
  - non-causal block, B = 8;
  - fc over the 5 target layers;
  - SWA draft layers.
- Add to that path:
  - per-layer conv ops: kernel_projection GEMM h → 2048 per block row, then a depthwise kernel-2 conv;
  - the selector, with an analogue of serial_table: hidden_projection, then a pairwise K × K score over the codebooks;
  - the codebooks, as cold storage.

## §201 Reflection Beam-1.0; DFlash2 drafts; as-released activation formats (0.254)

### §201.1 Beam-1.0-501B-A23B (BF16 / FP8 / NVFP4)

**Sources 「来源」.**
- The HF config and safetensors headers of all three releases.
- The model card.
- transformers main `models/beam/modular_beam.py` (`BeamForCausalLM`).
- vLLM main `vllm/models/beam/nvidia/` (PR #62074, 2026-10-19).

**Structure.**
- 52 layers.
- **Attention.**
  - GQA 48 q / 8 kv × 128 with q / k RMSNorm.
  - A per-head sigmoid output gate: `gate_proj` h → H·hd, plus `gate_bias` [H]. It is fused into q as `out_gate`.
  - `layer_types`: SWA 512 with RoPE, except every 6th layer and the last, which are full-attention NoPE (9 layers). The NoPE layers scale the query by (1 + pos/2048)^0.1, a vector op.
- **MLP.**
  - `mlp_layer_types`: 3 dense SwiGLU layers (20480 wide), then 49 MoE layers.
  - Each MoE layer has 256 SwiGLU experts × 2560, top-8, and no shared expert.
  - The router is an fp32 sigmoid with `e_score_correction_bias` and group-limited selection (n_group 8, topk_group 4). `routed_scaling_factor` 2.5 is applied after the combine.
- **Other.** Sandwich norms (4 per layer), depth-scaled residual adds (scalars), fp32 SwiGLU products, untied embeddings, and a final tanh soft-cap of 30. These are vector costs.
- **Builder.** `_build_beam`.

**Formats.**
- **FP8** (compressed-tensors).
  - Per-channel static fp8 weights and dynamic per-token fp8 activations (W8A8).
  - Not quantised: embed, lm_head and router; the 48 per-layer gate_bias vectors stay bf16 (2,496 params).
- **NVFP4** (ModelOpt).
  - Only the experts are NVFP4 (4.5 b), as W4A4.
  - Layers 0–2, every `self_attn` and every router are excluded and stay bf16 / fp32.

**Exactness.** param_check is exact for all three: 501,267,612,352.

**Numbers 「推导」** (default 100T scenario, TP 8 / EP 8, batch 1, ctx 4096):

| | decode TPOT | prefill TTFT | weights / card |
|---|---|---|---|
| BF16 | 71.49 ms (does not fit, 118 GiB needed) | 746.7 ms | 116.9 GiB |
| FP8 | 37.08 ms | 382.3 ms | 58.8 GiB |
| NVFP4 | 28.02 ms | 311.3 ms | 34.4 GiB |

**Not modelled.** Group-limited routing changes which experts are hit, but the EP load model stays uniform / skew-lever based 「假设」.

### §201.2 DFlash2 drafts on the DSpark block path

**Sources 「来源」.**
- vLLM main `model_executor/models/qwen3_dflash2.py` (PR #52816 conv + selector, PR #55960 fused grouped conv) on `qwen3_dflash.py` (SWA / causality resolution).
- `models/beam/nvidia/dflash2.py`, `dflash2_ops.py` and `dflash2_conv_ops.py`.
- `v1/worker/gpu/spec_decode/dflash2/speculator.py` (`_selector_walk_kernel`).
- The draft config and safetensors headers.

**Mechanism.**
- **Input.** `fc` over the concatenated hidden states of the 5 target layers (nt·h → h), then `hidden_norm`.
- **Context K/V.** One stacked GEMM h → nl·2·KV·hd with k_norm and RoPE. The rows go into a **sliding-window** paged draft cache: `dspark_main_kv` now writes them for windowed GQA drafts, and storage keeps min(ctx, window).
- **Block.** Anchor (bonus) row + `spec_k` mask rows; attention is **non-causal** (`is_causal` false).
- **Per draft layer.** GQA + q/k norm, SwiGLU, and two grouped dynamic convolutions (`attention_conv`, `mlp_conv`). Each convolution has:
  - a `kernel_projection` GEMM h → 2·taps·(h/16) (Beam 1280, Muse 2048);
  - `base_kernel` [2, taps, h] plus a per-group delta;
  - a taps-tap conv inside the block, side 0 before the sub-block and side 1 after it, fused into the residual norm.
  - Vector cost: 2 convs × 2 sides × taps × 3 per element.
- **Tail.**
  - The target lm_head on the B − 1 draft rows, then top-K (16).
  - `hidden_projection` h → R (256).
  - Edge scores: unary + ⟨pred[q] ⊙ h_p, succ[c]⟩, i.e. K·K·R MACs per row, with 2K·R codebook-row gathers.
  - The walk: one gumbel-noised argmax over K per row. It is the same kernel for greedy and sampled rounds, so a sampled round adds no op.
- **Storage.**
  - The codebooks (2·V·R) are cold: only gathered rows are read.
  - The other draft weights are hot and streamed.
  - The conv, selector and fc GEMMs are ReplicatedLinear (repeated on every TP rank). Attention and MLP shard normally (the reference allows TP).
- **Block size.**
  - The served block is 1 + `num_speculative_tokens`, so `spec_k` sets it, up to the trained block − 1 (Beam 15, Muse 7); a larger `spec_k` warns and is capped.
  - Default `draft_block` = the card's serving block 8.
- **Fields.** `ModelSpec.draft_dflash2` = (taps, group, rank, top-k, trained block). The attach function is `_attach_dflash2`, and draft json kind is `dflash2`.

**Published acceptance 「来源」 → i.i.d. 「推导」.**

| draft | card setting | τ greedy | τ sampled | a (k = 7) |
|---|---|---|---|---|
| Beam DFlash2 | NVFP4 target, TP 4, B300; mean of GSM8K / MATH-500 / HumanEval / MBPP / MT-Bench | 4.876 | 4.67 (T = 0.7, top-p 0.99 — not T = 1, 「二手」) | 0.852 |
| Muse DFlash2 | NVFP4 target, GB300 TP4+EP4; mean of MT-Bench, SPEED-Bench qualitative and 32k | 3.687 | 3.33 (T = 1) | 0.758 |

**Numbers 「推导」** (TP 8 / EP 8, k = 7, `spec_accept_src` published):

| model | decode TPOT, no draft | greedy | sampled |
|---|---|---|---|
| Beam NVFP4 + DFlash2 | 28.02 ms | 16.24 ms | 16.95 ms |
| Muse NVFP4 + DFlash2 | 96.58 ms | 51.37 ms | 56.88 ms |

### §201.3 Muse-Spark-Open DFlash2 — closes §200.4

Built with the same `_attach_dflash2`:
- 5 non-causal SWA-2048 layers (`dflash_config.swa_window_size`), 128 / 16 × 64 heads, SwiGLU 24576;
- trained block 8 (so `spec_k` ≤ 7);
- `output_multiplier` 0.177 and soft-cap 20, both vector.

The draft is exact at 4,384,162,432 params. The §200.4 deferral rested on a wrong code search (see the correction there).

### §201.4 As-released activation formats (default change, mechanism-justified)

- **compressed-tensors.** A `config_groups` entry with `input_activations` {num_bits 8, type float} (dynamic per-token, or per-group 128) is W8A8. vLLM `CompressedTensorsW8A8Fp8` quantises every GEMM input to fp8 「来源」.
  - `_act_fmt` bf16 → **fp8**.
  - LFM2.5-24B-A2B-FP8 prefill TP 8: 33.19 → 24.38 ms. Beam-FP8: 424.39 → 382.34 ms. Decode is unchanged (weight-bound).
- **ModelOpt.**
  - `quant_algo` "NVFP4" (or 4-bit `input_activations`) is W4A4. vLLM `ModelOptNvFp4` and SGLang `modelopt_fp4` quantise the input to NVFP4 (global input_scale, e4m3 per 16) 「来源」.
  - `_w4a4_roles` adds `act_<role>` = nvfp4 for the roles stored in NVFP4. In practice that is the experts of Muse, Step-5 and Beam; their excluded modules keep bf16 inputs.
  - On chips without a native NVFP4 MAC (the default 100T), (nvfp4, nvfp4) runs as bf16 like the old (nvfp4, bf16) dequant, so the numbers are unchanged. On Blackwell-class rates the expert GEMMs run at the NVFP4 rate.
  - "W4A16_NVFP4" (Dory) stays weight-only.


## §202 Group-limited MoE routing; Muse / Step-5 image processors; FLUX 3 mode-stream reuse; "no public code" audit (0.255)

### §202.1 Group-limited routing (n_group / topk_group)

**Sources 「来源」.**
- transformers `DeepseekV3TopkRouter` and `BeamTopKRouter` (models/beam/modular_beam.py), and vLLM `grouped_topk`. `bailing_moe.py` sets `use_grouped_topk` when both fields are set.
- How the router works:
  - Each group is scored by the sum of its top-2 (bias-corrected) sigmoid scores.
  - It keeps `topk_group` of the `n_group` groups.
  - It masks the other groups, then takes top-k over the remaining experts.
- Groups are contiguous expert ranges. EP placement is contiguous as well (vLLM / SGLang default).

**Applies to.**
- `model._moe_groups(c)` returns (G, g) when 1 < g < G and E % G = 0. That gives 8 / 4 for:
  - DeepSeek-V3 / R1 / V3.1 / V3.2;
  - Ling-3.1-flash and Ling-3.0-flash-VL;
  - Beam-1.0.
- Kimi-K2.5 and the GLM-5 family use 1 / 1, which is unrestricted.
- The value is stored on `Ffn.groups` for both the layers and the MTP layers.

**Model 「推导」** (symmetric-routing 「假设」, as in `moe_routing`):
- A token picks g of G groups uniformly, then k distinct experts uniformly among the g·S experts (S = E/G).
- **Marginals.** Each expert is hit with probability (g/G)·k/(gS) = k/E, so mean hits and mean pairs are unchanged.
- **Pairs on a rank** (per token, i.i.d. over tokens):
  - When ep ≥ G (G | ep), a rank holds n_r = E/ep experts of one group, and X = 1[group chosen]·Hyp(gS, n_r, k).
  - When ep < G (ep | G), a rank holds a = G/ep groups. J ~ Hyp(G, a, g) of them are chosen, and X | J ~ Hyp(gS, J·S, k).
  - Var(pairs) = T·Var X.
- **Distinct experts D.** Var D = N·e1(1 − e1) + Σ_pairs (e2 − e1²).
  - Single expert: e1 = (1 − p + p·x)^T, with p = g/G and x = 1 − k/(gS).
  - Same-group pair: e2 = (1 − p + p·y)^T, with y = (gS − k)(gS − k − 1)/(gS(gS − 1)).
  - Cross-group pair: e2 = (P0 + P1·x + P2·y)^T, where P0 / P1 / P2 are the probabilities that neither / one / both groups are chosen.
- If the groups do not tile the ranks, the ungrouped rule is used.
- `_rank_max` feeds these moments into its normal-max rule. That rule is unchanged; only the moments change.

**Check (Monte-Carlo, 4000 runs, 256 / 8 / 4, T = 16).**

| ep | E D, sd D (model / MC) | E pairs, sd pairs (model / MC) | expected max D, pairs (model / MC / ungrouped model) |
|---|---|---|---|
| 8 | 12.75, 3.50 / 12.71, 3.47 | 16, 5.23 / 15.9, 5.21 | 17.77, 23.50 / 17.99, 24.4 / 16.72, 21.29 |
| 4 | 25.49, 4.64 / 25.52, 4.63 | 32, 6.85 / 32.1, 6.88 | 30.36, 39.18 / 30.86, 40.22 / 29.60, 37.07 |
| 16 | 6.37, 2.22 / 6.39, 2.19 | 8, 3.26 / 8.0, 3.23 | 10.31, 13.76 / 10.41, 14.46 / 9.84, 12.78 |

- The moments are exact.
- The normal-max rule under-reads the true maximum by 1–5 %. That is the existing 「假设」, not tuned.

**Node-deduplicated all-to-all.**
- `node_dedup_factor(E, k, N, groups)` gives the expected destination nodes per token divided by k.
  - When N | G, a node holds G/N groups: the nodes touched are those holding a chosen group that receives ≥ 1 expert.
  - When G | N, a group spans N/G nodes, and the per-node hit probability applies only within the chosen groups.
- 256 / 8 / 4: at N = 8, 0.662 → 0.454 (MC 0.455); at N = 4, 0.452 → 0.366 (MC 0.367).
- This is the DeepSeek-V3 report §2.1.2 mechanism ("node-limited routing" bounds network fan-out), now expressed through the group limit.

**Effects** (default chip, decode batch 64, TP 8, EP 8, `moe_rank_max` expected = default):
- Ling-3.1-flash TPOT: 313.59 → 319.19 ms.
- DeepSeek-V3 (527.66 ms) and Beam (645.97 ms) are unchanged: their expert phase is not the binding term at this point.
- Prefill changes by ≤ 0.03 %.

### §202.2 Image / video processors that are now public

**Muse Spark Open.**
- **Sources.** transformers main `models/muse_glimmer/image_processing_muse_glimmer.py` and `video_processing_muse_glimmer.py`.
  - `auto/image_processing_auto.py` maps muse_spark → MuseGlimmerImageProcessor; `video_processing_auto` maps it → MuseGlimmerVideoProcessor.
  - `processing_muse_spark.py` gives the token layout.
  - The repo `processor_config.json` was read from the public amd/…-Quark-MXFP4 mirror, because the meta-models repo is gated. Values:
    - image: `max_image_tokens` 4096, patch 14, merge 2;
    - video: `max_video_frame_tokens` 256, `num_frames` 600, fps 2, temporal patch 2.
- **Resize (`smart_resize`, verbatim → `vision._muse_resize`).**
  - Ideal grid = (h/28, w/28). If it exceeds the cap, it is scaled to the cap at the same aspect.
  - Candidates are the floor / ceil combinations ≥ 1 that fit under the cap. The one closest to h/w in aspect is chosen.
  - There is no lower bound and no rounding to the nearest multiple.
  - Upstream `min` over a set breaks ties in set order; here ties break ascending. Ties are rare.
- **Image tokens.** `<|image_start|>` + merged `<|patch|>` + `<|image_end|>`.
- **Video (`vision._muse_video`).**
  - `sample_frames`: n = max(2, ⌊min(F, 600)/2⌋·2).
  - Per-frame grid with a cap of 256.
  - `grid_t` = n/2. The tower attends per temporal slice (`get_vision_cu_seqlens`, `merge_temporal` False).
  - LLM tokens = t·merged + `<|vid_start|>` + (t − 1) separators + `<|vid_end|>`. The "Time: x.xxs" text is not counted (same as Qwen video).
- **Coverage.** The builder now returns `full`. The six Muse entries are no longer vision-partial.

**Step-5-Preview crops.**
- **Search.** Directory-wide over vLLM, SGLang and transformers main. There is no step5 processor or model. vLLM has `step3_vl` / `step_vl` / `step3p5` / `step3p7`.
- **Closest source.** transformers `models/step3p7` (Step-3.7-Flash). Its config `vision_config` is identical to Step-5's: perception_encoder, 47 × 1536, 16 heads, 728 / 14, quick_gelu, ls 0.1. It also has the same `image_token_len` 169, `patch_token_len` 81 and `image_token_id` 128001.
- **Link to Step-5.** Step-5's `inference/validate_vision.py` loads its reference encoder through a `configuration_step3p7` module. Its README orders tokens crops-first, then the overview, exactly as `Step3p7Processor.replace_image_token` does.
- **Plan (`vision.step_crops`, verbatim from `_plan_patches`).**
  - Normalise:
    - Extreme aspect (min side < 32 and ratio > 4) is padded to a square.
    - A long side over 3024 is scaled down.
  - Choose the window:
    - long side ≤ 728: the short side if the ratio is > 1.5, otherwise no crops;
    - ratio > 4: min(short side, 504);
    - otherwise: 504.
  - Snap each side to a window multiple, rounding up when the remainder is > 20 %.
- **Tokens.**
  - Per crop: `<patch_start>` + 81 + `<patch_end>`.
  - `<patch_newline>` after each crop row except the last.
  - Then `<im_start>` + 169 + `<im_end>`.
- **Tower.** `ImageGrid.views` = ((52, 52, 1), (36, 36, n)).
- **Examples.**
  - 1024×768 → 2×2 crops, 504 tokens (was 169).
  - 1920×1080 → 4×2 crops, 836 tokens.
  - 4000×3000 → 6×5 crops, 2665 tokens.
- **Coverage.** Step-5 stays `partial`. The algorithm is sourced, but applying it to Step-5 is a 「假设」.

### §202.3 FLUX 3 Base static mode-stream reuse

**Sources 「来源」.** SGLang `multimodal_gen/runtime/models/dits/flux3_image.py`:
- `Packed.static: StaticStreams` holds an "eager-only reuse of invariant text/reference computation".
- `_forward_packed` computes `vec`, early modulations, single modulations and embeddings only for the modes not in `static.hidden`.
- `_packed_blocks` skips the cached modes' mode blocks. After step 1 it stores `x[TEXT]`, `x["image_cond"]` and their single-stream modulation tuples.
- The single (joint) blocks run every step on all rows.

**When SGLang enables it.** `stages/model_specific_stages/flux3_image.py` sets `packed.static` only when all of these hold:
- not the flow map;
- no breakable CUDA graph, no torch.compile, no attention-backend autotune;
- no layer-wise offload;
- no CFG-parallel and no sequential CFG;
- no skip-softmax;
- `can_reuse_mode_blocks()`, which requires Ulysses SP world 1 and an FA / SDPA backend.

**Why it is exact on Base 「推导」.** Text and references run at timestep 0 and attend only within their own mode. On the flow map, the target-time embedding is added to every mode, so they change each step.

**IR.**
- `Phase.mreuse` applies on reused steps:
  - `txt` / `ref` rows → 0, for io (txt_in, ref embed, their early / single modulations) and for the mode-block GEMMs;
  - the per-token time MLP covers image rows only;
  - mode attention covers the image group only, with QK-norm / RoPE on image rows;
  - mode-layer vector ops cover image rows.
- `evaluate`:
  - With `workload.mode_cache = on`, step 1 is a full forward and steps 2…n are `mreuse` forwards, the same two-tick structure as `prefix_kv`.
  - Resident cache = 4 · (text + ref rows) · hidden · act bytes per sequence on the DP rank. SGLang holds tensor references, so there is no copy op.
  - Result field `mode_cache` = {rows, cache_GiB_card, first_ms / reused_ms, extract / cached TFLOP per sequence, fits}.
- Distilled, SP > 1, PP > 1 or steps < 2 → ignored with a warning.
- The default is `off`, matching the BFL / diffusers reference (no reuse). The knob is an engine choice, not a default change.

**Numbers** (default chip, 1024², Base 50 steps, CFG 2):

| references | TFLOP / step-seq off → on | latency off → on |
|---|---|---|
| 0 | 44.30 → 43.81 | 47.23 → 47.10 s |
| 1 | 97.26 → 90.17 | 103.27 → 96.50 s |
| 3 | 244.01 → 217.53 | 265.44 → 238.12 s |

Per reused step, 5 of 33 blocks drop the text and reference rows. The single blocks keep everything.

### §202.4 Audit of "no public code" deferrals (vLLM / SGLang / transformers / diffusers main, full git trees)

**Method.**
- Fetched the full recursive git trees.
- Matched paths against muse|glimmer|step|flux3|dory|loop|recurren, across every directory, including `vllm/models/<arch>/`.
- Read the matching files.

| deferred item | where | result |
|---|---|---|
| Muse image resize (「假设」 since 0.174) | transformers muse_glimmer image processor | implemented (§202.2) |
| Muse video processor | transformers MuseGlimmerVideoProcessor (+ vLLM `transformers_utils/processors/muse_glimmer.py`) | implemented (§202.2) |
| Step-5 crop algorithm (0.153) | transformers step3p7 (sibling model) | implemented, applicability 「假设」 (§202.2) |
| FLUX 3 reference mode-output cache (0.252) | SGLang flux3_image DiT StaticStreams | implemented as an opt-in engine knob (§202.3) |
| Dory TP > 1 (0.253) | vLLM PR #60149 (open, head 5507695, updated 2026-10-19) `dory.py:294` still raises for TP ≠ 1; no dory in vLLM / SGLang / transformers main | still deferred |
| FLUX 3 "no public modeling code" (§198, 0.251) | superseded since 0.252 (BFL reference + diffusers `transformer_flux3.py`); SGLang also has `flux3_image.py` now | historical note only |
| MAI-Base GGUF hyperparameters, hardware figures | not code | n/a |

**Not changed.** SGLang skips an empty content mode (image_cond in text-to-image) entirely. The model already ignored that mode's one phantom row, which matches.

### §202.5 Freshness

- 75 vendor orgs plus the HF trending list, created on or after 2026-10-17.
- Only DeepSeek-V4.1-Flash-1017 (already an alias since 0.248) and third-party repacks / fine-tunes were found: unsloth, dealignai, LibertAIDAI, lodestones Kromoss, orcarouter OrcaCyber-Nano.
- Noted for a later sweep: OpenMOSS-Team/MOSS-Image-Base / -SFT (diffusers, 2026-10-11, new DiT). It predates this window, and its org was missing from earlier sweeps.

## §203 Freshness sweep script; MOSS-Image; PD per-pool energy ledger (0.257)

### §203.1 Sweep
`scripts/freshness_sweep.py <since> [out.json]` queries the Hugging Face API (`api/models?author=<org>&sort=createdAt`) for every HF org prefix in `catalog.PROVIDERS` plus `EXTRA_ORGS` (OpenMOSS-Team first): 74 orgs. Window 2026-10-17 onward: DeepSeek-V4.1-Flash-1017 (alias since 0.248) and repacks only. Timestral (mistralai, 10-14) is a time-series model, out of domain.

### §203.2 MOSS-Image Base / SFT (OpenMOSS-Team, 2026-10-11)
**Source 「来源」.** Public diffusers repos: transformer `config.json` + safetensors headers (`MOSSImageTransformer`), the text-encoder and VAE headers.

**DiT** (`domain.build_moss`, builder "moss"):
- 42 single-stream blocks: q / k / v / o, fused gate_up 3072 → 2·8192, down; 24 heads × 128; RoPE over the full head.
- io_pre: `video_embedder` (image rows); `text_embedder.proj` + 2 refiner blocks (text rows; refiner attention core ignored 「近似」, < 0.1 % of FLOPs); `time_embedding` (sequence); `shared_block_modulation` (sequence).
- io_post: `video_head_modulation`, `final_linear_video`.
- param_check exact (5,082,308,624); coverage full. Sequence 1024² / 8 / 2² = 4096 image + 256 text tokens 「假设」 = 4352.

**Workload.** 50 steps (shift 5 rescales, still 50 timesteps 「推导」); CFG 3 → 2 forwards per step. The reference skips block 22 in the unconditional pass and uses an empty negative prompt (~10 tokens); modelled at the conditional shape 「近似」 (+3–4 %). Since 0.258 the negative branch runs at its own shape (§204.1); the refiner attention core is counted (§204.2). Per-sample loop 「近似」. The `_situ` activation is costed as SwiGLU 「近似」.

**Pipeline.**
- Text tower Qwen3.5-9B (`language_model.` tensors, 7.94 B used of 8.39 B stored, bf16). `TextEnc.full_layers` = (3, 7, …, 31) get softmax attention; the other 24 are Gated DeltaNet (`TextEnc.gdn` = (heads 32, d_k 128, d_v 128, conv channels 8192, kernel 4, chunk 64)), costed by `pipeline._gdn_ops` with the LLM IR's chunked form (ir._attn_core_ops): akk / ut_solve / vprime / aqk / intra / inter / state + per-token and per-chunk vector work + the causal conv. No state is kept (encoder pass).
- VAE: AutoencoderKLFlux2 with the FLUX.2 small decoder, family `os_s384` (mid / up attention width 384), fp32 as released (62.4 M parameters).

**Numbers** (default chip, 1024²): 54.61 s per image, 5143.5 TFLOP per request; text tower 0.083 s, VAE decode 0.078 s. FLUX 3 Base: 47.23 s.

### §203.3 PD energy per pool (closes §177's two approximations)
`pdqueue._energy(..., pools, pool_counts, pool_names)`. The PD mode passes:
- decode = the decode iteration mix × out_len + the KV receive (link / net) and preemption-restore extras, booked to decode 「约定」;
- prefill = all counts − decode (exact subtraction, so Σ pools = total).

Per pool p, with its scenario (chip, memory) and card-seconds per output token cs_p:
- DRAM J = bytes_p · 8 · pJ(kind_p) + PHY_p; every other dynamic action = counts_p × the table.
- Clock = macs·f·cs_p · pJ_clk · (g_p + (1 − g_p)·u_p), u_p = MAC_p / (macs·f·cs_p).
- Idle: `idle_W_prefill` on the prefill pool when given, else `idle_W` (or idle_frac_tdp × TDP); leakage and HBM standby per pool as before.

Output: `J_by_pool` {decode, prefill} → J_per_token, J_by_action, card_s_per_token, dram_field, mac_util; `clock_gating.pool_mac_util`.

**Invariant.** With identical chips, memory and g, the result equals the pooled 0.230 form (all terms linear in counts; Σ_p cyc_p·u_p = MAC total). Σ_pools J = J_per_token, per action too.

**Why it matters for chip design.** A heterogeneous PD design (compute-dense prefill chip with cheaper memory, bandwidth-dense decode chip with aggressive gating) was mispriced: the prefill pool's DRAM took the decode memory's pJ/bit, and the decode pool's low utilisation (≈ 0.02) was averaged with the prefill pool's (≈ 0.43). The ledger also shows which pool to optimise. In the qwen3-8b example ~80 % of PD J / token is the decode pool, mostly DRAM. Numbers: CHANGELOG 0.257.

## §204 MOSS-Image CFG branches and refiner attention; SLC thrash-resistant insertion (0.258)

### §204.1 CFG negative branch at its own shape (closes the §203.2 「近似」)
**Source 「来源」.** `pipeline_moss_image.py` `MOSSImagePipeline.__call__` defaults: `guidance_scale=3.0`, `negative_prompt=''`, `uncond_skip_layers=(22,)`, `slg_start=0.0`, `slg_stop=1.0`.
- Per step it makes one transformer call on the prompt, then one on the negative prompt with `skip_layers=(22,)`, because `skip_active` holds at every step.
- `encode_prompt` runs one unpadded text-encoder forward per prompt, after the chat template (`add_generation_prompt`, `enable_thinking=True`).

**Negative prompt length 「推导」.**
- The template renders `<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<think>\n`.
- That is 4 special tokens (`<|im_start|>` × 2, `<|im_end|>`, `<think>`) + `user` + `assistant` + 4 newlines. The Qwen pre-tokenizer splits letters from newlines, so the total is **10 tokens**.

**Mechanism.**
- `NativeWorkload.uncond = (text tokens, skipped blocks)` is model-declared; `build_moss` sets (10, (22,)).
- `Phase.txt` overrides the joint text rows of a forward (`ir._prefix(model, ph)`). `Phase.skip` lists blocks the forward skips; `_evaluate_full._run` drops them from the stage's layer counts.
- `evaluate._cfg_asym` turns it on when the workload is gen, CFG = 2 and `workload.cfg_branches = ref` (the default).
  - The cond phase has B sequences and the release's shape.
  - The uncond phase has B sequences, q = video rows + 10, txt = 10, skip = (22,).
  - Both use the same stage plan.
  - Per step = cond tick + uncond tick: two serial calls, each streaming the weights. `Result.microbatches` = 2.
- Not modelled → falls back to the packed, symmetric forward with a warning: PP > 1, `serving.microbatches` ∉ {0, 2}, or `step_skip`.
- **Accounting:**
  - `_tflop_per_request` = (cond + uncond TFLOP per sequence) × steps.
  - `energy._counts` runs both stage sets × steps.
  - `domain_summary.forward_ms` = the step time.
  - The text encoder runs the negative prompt at 10 tokens in its own call (`_pipeline_cost(neg_tokens)`).
- `cfg_branches = same` gives the pre-0.258 result (plus §204.2).

**Numbers** (default chip, 1024², 50 steps), `same` → `ref`:

| | same | ref |
|---|---|---|
| per step | 1089.1 ms (one packed forward of 2 sequences) | 545.9 + 512.2 ms |
| DiT TFLOP / request | 5130.5 | 4953.8 (−3.4 %) |
| text encoder | 0.083 s | 0.145 s |
| latency | 54.62 s | 53.13 s (−2.7 %) |

The packed forward reads the weights once for both branches. The serial calls read them twice, but the work is compute-bound at 4352 rows, so the smaller uncond shape wins.

### §204.2 Text-refiner attention core
`ModelSpec.io_attn = ((name, rows, heads, head_dim, blocks), …)`; `ir._io_attn_ops` adds, on the first stage:
- the qk / pv attention GEMMs and the softmax;
- the QK-norm + RoPE vector work over the text rows of the forward.

Heads are split over TP and then SP, like the block attention; the SP all-to-all of these few rows is not added 「近似」. A cached step (kvpre) or a reused step (mreuse) runs none.

MOSS: 2 refiner blocks × 24 heads × 128 (`_TextBlock`: SDPA, bidirectional, RMS QK-norm, RoPE). The cost is negligible at 256 rows (< 0.01 s per image); the term is there so the count is complete. HunyuanVideo / MiniMax-H3 token refiners could use the same field (not switched on).

### §204.3 SLC policy `dip`: thrash-resistant insertion
**Source 「来源」.** Qureshi, Jaleel, Patt, Steely, Emer, "Adaptive Insertion Policies for High Performance Caching", ISCA 2007, §4.1, Table 3.
- Cyclic reference (a1…aT)^N through a fully associative K-line cache, K < T:
  - LRU hit rate 0;
  - OPT (K − 1)/T;
  - BIP (K − 1 − ε·(T − K))/T, where ε is the bimodal throttle. The paper uses ε = 1/32 for all experiments.
- DIP set dueling picks between LRU and BIP, so a working set that fits gets LRU's all-hit behaviour.

**Model** (`memplan.dip_hit`, `DIP_EPS`):
- T = the stage's off-SRAM working set (hot + expert weights not in SRAM, KV / state not in SRAM, the activation working set), the same T the `lru` rule uses; K = `slc_mib`.
- T ≤ K → `slc_all`, every read hits.
- Otherwise `slc_frac` = max(0, (K − ε(T − K)))/T. The one-line term is dropped (K ≫ a line).
- The retained lines are an arbitrary subset of the sweep, so every read class (weights + streamed re-reads, KV, state, activations, idle-rank expert reads) hits by the same share 「推导」. KV writes go to DRAM as for `lru`.
- Fully associative, or set-associative with addresses spread evenly over the sets 「近似」.
- Conservation is unchanged: SLC + DRAM bytes = the DRAM bytes without an SLC.

**Limits.**
- One cyclic sweep per step: MoE routed experts that a step does not read are still counted in T 「近似」, as for `lru`. (0.260: replaced by the per-class split, §205.2.)
- No per-class reuse distance (0.260: §205.2); no RRIP variant (BRRIP behaves like BIP on cyclic sweeps).

**Example: why it matters for SLC sizing** (100T + LPDDR5X 4×64 8533, qwen3-8b decode B8, TPOT 104.2 ms without SLC):

| SLC | pin | lru | dip (hit share) |
|---|---|---|---|
| 1 GiB | 98.5 | 104.2 | 101.6 (2 %) |
| 4 GiB | 81.7 | 104.2 | 84.2 (19 %) |
| 8 GiB | 59.2 | 104.2 | 61.1 (41 %) |
| 12 GiB | 36.7 | 104.2 | 37.9 (64 %) |
| 16 GiB | 14.3 | 104.2 | 14.7 (86 %) |
| 32 GiB | 11.7 | 11.7 | 11.7 (100 %) |

`lru` says an SLC is worthless until it holds the whole working set. A hardware SLC with thrash-resistant replacement gains in proportion to its size, within ~3–4 % of software pinning. The sizing decision should not rest on the LRU bound.

### §204.4 Freshness
`freshness_sweep.py 2026-10-19` (74 orgs) found nothing new. HF trending (≥ 10-18) shows only a DeepSeek-V4.1-Flash-1017 repack.

## §205 Token-refiner attention (HunyuanVideo, MiniMax-H3); `dip` per-class reuse; cfg_branches UI; SLC leakage and leakage ∝ V (0.260)

### §205.1 Token-refiner attention cores
- `ModelSpec.io_attn` entries take an optional 6th field for the per-token vector work: `"qknorm_rope"` (default, MOSS), `"qknorm"`, `""`.
- **HunyuanVideo** (`domain.build_hunyuan`): the individual token refiner blocks run bidirectional SDPA over the text rows, 24 heads × 128 (heads = `to_q` width ÷ 128), with no QK-norm and no RoPE 「来源」 (diffusers `HunyuanVideoIndividualTokenRefinerBlock`). The attention mask is taken as full length 「假设」.
- **MiniMax-H3** (`build_h3`): 56 heads × 128 (`to_q` 7168 ÷ `norm_q` 128), QK-norm 「推导」 from the weights. Whether the refiner applies RoPE is not visible in the weights, so it is not counted 「假设」 (a vector-only term). (0.261: no RoPE is now 「来源」, §206.2.)
- Effect is negligible: hunyuanvideo 6908.7231 → 6908.7246 s, minimax-h3 1937.7807 → 1937.7929 s; MOSS unchanged. No weights added (param_check unchanged).

### §205.2 `dip` per-class reuse (closes the §204.3 MoE 「近似」)
- **Why.** One cyclic sweep per step put every routed expert in T, but a decode step reads only the routed share e of the off-SRAM expert bytes E, while the hot weights, KV / state and activations (C bytes) are read every step.
- **Model** (`memplan.dip_split(K, C, E, e)`):
  - C + E ≤ K → all hit (as before).
  - Otherwise the every-step classes are retained before the experts: the A0 order of Aho, Denning, Ullman (J. ACM 1971), optimal under the independent reference model 「近似」 (a DIP cache only approximates frequency order).
  - BIP churn shrinks the retained capacity by the bytes that stream through per step: K′ = K − ε·max(0, C + e − K), ε = 1/32 (§204.3).
  - Every-step share f_c = min(1, K′/C); expert share f_e = clip((K′ − C)/E, 0, 1). Expert reads hit by f_e (uniform routing 「近似」).
  - E = 0 reduces to the 0.258 `dip_hit` formula, so dense models are unchanged.
- `plan(..., exp_read=agg["exp"])` passes the per-step expert reads; `MemPlan.slc_frac_exp` and API `mem.slc_frac_exp` report f_e. Conservation holds (SLC + DRAM = DRAM without an SLC).
- The Che approximation (`prefixcache`) was considered and not used: a cyclic LRU-like sweep is not IRM.

**Example** (qwen3-30b-a3b, 100T + LPDDR5X 4×64 8533, decode B8, TPOT ms; 152.64 without SLC):

| SLC GiB | pin | lru | dip 0.258 | dip 0.260 | f_c | f_e |
|---|---|---|---|---|---|---|
| 2 | 141.4 | 152.6 | 152.1 | 145.8 | 0.23 | 0 |
| 4 | 136.1 | 152.6 | 146.8 | 134.2 | 0.63 | 0 |
| 8 | 126.9 | 152.6 | 136.2 | 118.3 | 1 | 0.04 |
| 16 | 108.7 | 152.6 | 114.9 | 99.5 | 1 | 0.19 |
| 32 | 72.2 | 152.6 | 72.4 | 62.2 | 1 | 0.50 |
| 64 | 17.2 | 17.2 | 17.2 | 17.2 | 1 | 1 |

From 4 GiB `dip` beats `pin`: pin's static order places expert weights before KV, while a frequency-ordered cache keeps the every-step KV.

### §205.3 UI: `cfg_branches`
The gen workload group shows a `CFG 负分支` select (`w-cfg_branches`: ref 参考实现 / same 同形状打包) only when the model declares a negative branch (`workload.uncond` in the model info; MOSS-Image today). The key is dropped from the scenario otherwise. Three small hunks in `app.js`, one row in `index.html`; `serve.py` untouched.

### §205.4 SLC leakage; core-rail leakage ∝ V under DVFS
- **SLC leakage.** The SLC is on-die SRAM (its area already uses `mm2_per_mib_slc` = SRAM density, and `pJ_bit_slc` = SRAM). Its leakage is now charged with the SRAM term: P = (state_sram + sram + slc) MiB × `uW_per_mib_sram_leak` (230.7 µW/MiB, 3 nm HD at 25 °C, a lower bound 「来源」), when no measured idle figure is given. A measured idle figure already contains it (scope note `共享 SRAM / SLC 泄漏已含于 idle 数值`). Same in the PD ledger (`pdqueue._energy`).
- **Leakage vs V.** Under the power cap's V–f curve, core-rail leakage scales linearly with V: static power n·V, AccelWattch eq. 2 (Kandiah et al., MICRO 2021, §4.2: chip and memory static terms b·V + b′·V) 「来源」. Linear is conservative 「近似」: sub-threshold / DIBL leakage falls faster than V. The idle floor (measured at the idle P-state, already at low V) and HBM standby / refresh (memory rail; memory clock held) stay held 「假设」. Closes the 0.227 "static held" limit and the 0.228 "leakage-vs-V" deferral.

**Effect — SLC sizing on energy** (qwen3-30b-a3b, 100T + LPDDR5X, B8, `dip`, tpuv4i_7nm without idle_W; J/token):

| SLC | before | after | leakage W/card |
|---|---|---|---|
| none | 0.2561 | 0.2561 | 0.015 |
| 8 GiB | 0.2001 | 0.2281 | 1.905 |
| 32 GiB | 0.1086 | 0.1673 | 7.574 |

The SLC's energy saving at 32 GiB shrinks from −58 % to −35 %, and that uses the 25 °C lower bound (at operating temperature leakage is several × higher). A large SRAM SLC sized for bandwidth pays a static cost that grows with capacity, not with use.

Leakage ∝ V: RWKV-7 G1k 7.2B on 1P with a 7200 MiB state buffer (1.72 W leakage), cap 0.5 P, rx5700xt_n7: throughput ×0.7888 → ×0.7946; dfs unchanged (V = 1).

**Limits.** No temperature model (no advanced-node leakage-vs-T source; the 40 nm data in §159 is not transferred). The leakage of SRAM and SLC uses the same HD macro density. (0.261: temperature knob, §206.1.)

## §206 SRAM / SLC leakage vs die temperature; H3 refiner RoPE sourced; static-site guards; dip expert popularity (0.261)

### §206.1 SRAM / SLC leakage vs die temperature
- **Source 「来源」.** Aoyagi et al. (TSMC), "A 3-nm FinFET 563-kbit 35.5-Mbit/mm² Dual-Rail SRAM…", VLSI 2025 / JSSC 2025 — the same macro the 25 °C figure (27.5 µW/Mbit) comes from. Its measured mean standby leakage (Fig. 7 text, 0.563 Mbit, low-leak mode) is 15.5 µW at 25 °C and 73.5 µW at 85 °C: ×4.742 over 60 °C.
- **Model** (`energy.SRAM_LEAK_T`, `leak_temp_factor`, `sram_leak_uW_per_mib`): `uW_per_mib_sram_leak` stays the 25 °C figure; the charged value is × 4.742^((T − 25)/60), exponential in T (the sub-threshold form; doubling every 26.7 °C) 「推导」 between 25 and 85 °C, 「近似」 outside (0–125 °C allowed).
- **Knob** `energy.sram_temp_C` (UI 能耗组「SRAM 结温 °C」, API `energy.sram_temp_C`). Unset = 25 °C, the measured lower bound, so default results are unchanged. The report adds `sram_leak_temp_C` and the hot-die band `sram_leak_W_card_85C`. The PD ledger uses the same factor; under the power cap the V scaling (§205.4) multiplies on top.
- The macro is a mobile HD macro in its low-leak mode, so a datacenter macro at the same T is not lower 「假设」.

**Effect — SLC sizing on energy** (qwen3-30b-a3b, 100T + LPDDR5X, B8, `dip`, tpuv4i_7nm without idle; J/token for no SLC / 8 GiB / 32 GiB):

| Die T | none | 8 GiB | 32 GiB | leak @32 GiB |
|---|---|---|---|---|
| 25 °C | 0.2561 | 0.2281 | 0.1673 | 7.6 W |
| 55 °C | 0.2565 | 0.2612 | 0.2366 | 16.5 W |
| 85 °C | 0.2572 | 0.3335 | 0.3875 | 35.9 W |
| 105 °C | 0.2581 | 0.4243 | 0.5773 | 60.3 W |

On a hot die an SRAM SLC sized in GiB costs more energy per token than it saves in DRAM traffic (8 GiB already loses at 55 °C). It still buys TPOT (62.2 vs 152.6 ms at 32 GiB). The SLC decision is a time-vs-energy trade, and the temperature decides which side wins.

### §206.2 MiniMax-H3 token refiner: no RoPE 「来源」
diffusers main `transformer_minimax_h3.py` (last change 2026-10-01): `MiniMaxH3TokenRefinerBlock` — "Plain pre-norm transformer block … No AdaLN and no rotary embedding"; its attention is called without `rotary_emb` and without a mask; `MiniMaxH3Attention` has RMS `norm_q` / `norm_k`. The model is unchanged (RoPE was already not counted); the label moves from 「假设」 to 「来源」.

### §206.3 Static site (GitHub Pages) guards
- `tests/test_static_routes.py`: every `req(chan, '/api/…'[, body])` in app.js (GET without a body, POST with one) is a `serve.GET_ROUTES` / `POST_ROUTES` entry answered by `serve.respond` (≠ 404); no other `/api` literal or direct `fetch('/api…')`; the shim's slow-lane list names real routes; the package zip contains every git-tracked `accel_dse/` file outside `web/`; and the API answers from the unpacked zip alone (`python -I`, cwd = the temp dir: health, catalog, models, memory, eval default / MOSS / tree + draft, sensitivity all 200). Dropping `spec_tree_refs.json` or `series_catalog.json` from the zip makes it fail.
- `build_static.package_zip` packs every package file outside `web/` (it was `*.py` / `*.json` only, so a future non-JSON data file would have been silently dropped).

### §206.4 `dip` expert order under a measured expert load
- **Why.** §205.2 retained a byte share f_e of the off-SRAM experts and hit expert reads by the same share — exact for uniform routing. With a measured, skewed load (`serving.moe_expert_load`), a frequency-ordered cache keeps the most-read experts, so the read share served exceeds f_e.
- **Model** (`memplan.dip_exp_hit(p, ρ, f)`): p_e = k·load_e/Σload (busiest-load rank, `ir.hot_rank_probs`); per step expert e is read with q_e = 1 − (1 − p_e)^T, with T solved so that mean q = ρ = e/E (the op graph's read share, so read bytes are unchanged). The top f·n experts by q are retained (A0 order under IRM, 「近似」); hit share h = Σ_top q / Σ q ≥ f. Off-SRAM experts are a load-neutral subset 「近似」. `MemPlan.slc_frac_exp` is now the read-hit share; the new `slc_frac_exp_bytes` (API `mem.slc_frac_exp_bytes`) is the retained byte share. Uniform or flat loads: unchanged.
- **Example** (input loads Zipf 「假设」, not a measurement; qwen3-30b-a3b, 100T + LPDDR5X, B8, TPOT ms at 8 / 16 / 32 GiB): uniform 118.3 / 99.5 / 62.2; Zipf 0.5: 112.4 / 84.0 / 46.0; Zipf 1: 111.1 / 73.9 / 34.4 (read share 0.099 / 0.401 / 0.721 against byte share 0.04 / 0.193 / 0.496).
- **Limits.** `pin` stays load-unaware (it pins bytes, not chosen experts); per-layer loads are not distinguished (one load vector for every MoE layer).

### §206.5 Freshness
`freshness_sweep.py 2026-10-19` (74 orgs): only `tencent/Hy-Memory-30B-A3B-GGUF` (a repack of the catalogued Hy-Memory). ibm-granite errored in the sweep and was rechecked by the API: its newest repos (granite-switch-4.1-8b / 30b, 10-14, `GraniteSwitchForCausalLM`: control-token-switched LoRA adapters + ASR) predate the window and are deferred.

## §207 J/token-optimal SLC size under die temperature; granite-switch deferred (0.262)

### §207.1 SLC energy curve (`insight.slc_energy_report`, `/api/insight_slc`)
Closes the 0.261 hot-die finding as a chip-design decision tool.
- **Probe.** For each capacity in a GiB grid (default 0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64), set `chip.slc_mib` / `slc_policy` (default `dip`) and re-evaluate via the existing engine. Leakage is charged under the same scope as §205.4 / §206.1 (table has `uW_per_mib_sram_leak`, no measured idle).
- **Outputs per model.** The J/token curve; `opt_J_gib` = argmin J; `opt_time_gib` = argmin TPOT (LLM) or latency (gen); `crossover_gib` = smallest positive capacity whose J exceeds the no-SLC point (leakage dominates the DRAM saving). Report `sram_temp_C`, `leak_charged`, and notes when idle or a missing leak field suppresses the hot-die signal.
- **UI.** Insight tab →「SLC 能耗最优」; uses the energy panel's preset / `sram_temp_C`. Default model = the scenario's model (a GiB × N sweep is expensive).

**Example** (qwen3-30b-a3b, 100T + LPDDR5X, B8, `dip`, tpuv4i_7nm, no idle):

| Die T | opt J GiB | crossover | opt time GiB | reading |
|---|---|---|---|---|
| 25 °C | 64 (0.0507 J) | none | 64 | larger SLC always wins on both |
| 85 °C | 0 (0.257 J) | 4 GiB | 32 (62.2 ms) | any SLC costs energy; still buys TPOT |

The SLC decision is therefore an energy-vs-time trade that the temperature decides. Filling `idle_W` used to hide the leak (≤ 0.262); 0.263 charges SLC leakage as an increment on top of idle (§208.1).

### §207.2 granite-switch (ibm-granite, 10-14) — deferred
`ibm-granite/granite-switch-4.1-8b` / `-30b` (`GraniteSwitchForCausalLM`): public `config.json` exposes a granite-like dense base (8b: 42 L, 4096 d, GQA 32/8, intermediate 12800, vocab 100355, tie_word_embeddings) plus `adapter_names` / `adapter_ranks` (guardian-core r16, decision r64) and ASR fields. Safetensors are published. Without a public description of how the control-token-switched LoRA adapters fold into the serve path (loaded always? swapped per token? counted in the checkpoint total?), an honest `param_check` against the release weight total is not possible from config alone. Left out of the catalog; revisit when the serve path is documented.

### §207.3 Freshness
`freshness_sweep.py 2026-10-19` again: only the already-noted Hy-Memory GGUF repack. Inferact / stabilityai errored in the sweep (network); no new architecture releases in window.

## §208 Idle-aware SLC leak; 25/85 °C one-click compare; load-aware pin (0.263)

### §208.1 Incremental SLC leakage under a measured idle
A measured `idle_W` / `idle_frac_tdp` is of a card that does not include a *speculative* SLC the designer is sizing. Shared SRAM leakage stays suppressed by idle (already in the measurement); **SLC MiB is always charged** when `uW_per_mib_sram_leak` is set 「假设」. Scope string: `state_sram + slc（共享 SRAM 泄漏已含于 idle；SLC 按增量计「假设」)`. Same rule in the PD ledger. The SLC energy insight's `leak_charged` is now true whenever the leak field is set, so a one-click hot-die compare works with a typical idle-filled energy panel.

### §208.2 One-click 25 °C vs 85 °C compare
`slc_energy_report(..., temps=(25, 85))` / API `temps: [25, 85]` / Insight button「25/85 °C 对比」: runs the GiB sweep at each die temperature and returns `summary` (opt_J / opt_time / crossover per T) plus `by_temp`. Example (qwen3-30b-a3b, B8, dip, no idle, grid to 32 GiB): 25 °C → opt J 32 GiB, no crossover; 85 °C → opt J 0 GiB, crossover at 4 GiB.

### §208.3 Load-aware `pin`
With `serving.moe_expert_load`, the pin / pin_bw SLC path now treats the pinned expert bytes as the most-read ones (A0 order, same `dip_exp_hit` helper as §206.4). `slc_frac_exp` = read-hit share; `slc_frac_exp_bytes` = retained byte share. Uniform load unchanged. Example: qwen3-30b-a3b, 16 GiB pin, Zipf-1 load 「假设」: TPOT 108.7 → 80.1 ms (read share 0.487 vs bytes 0.255).

### §208.4 Freshness
`freshness_sweep.py 2026-10-20` (74 orgs): nothing new.

## §209 25/85 °C SLC in the default insight path (0.264)

### §209.1 Packaged into `insight_report`
The 0.262/0.263 SLC energy sweep stayed behind a dedicated button. Chip-design users who already click「运行设计洞察」now get the same 25/85 °C J-optimal / crossover summary when the energy table has `uW_per_mib_sram_leak` (preset or hand-filled). Field: `slc_temp` `{policy, temps, gibs, summary, recommend, notes}`. Grid: `INSIGHT_SLC_GIB = (0,1,2,4,8,16,32)` (coarser than the full curve). Policy = `chip.slc_policy` (else dip). Without a leak field, `slc_temp` is null (no silent assumption).

### §209.2 UI apply
`#ins-slc-temp` paints the compare and offers「应用 25 °C / 85 °C J 最优」→ `#c-slc_mib`. Status line notes when the hot die cuts J-opt. Full per-GiB curve remains under「SLC 能耗最优」.

### §209.3 Freshness
`freshness_sweep.py 2026-10-20` (74 orgs): nothing new.

## §210 Datapath clock gating; PD SLC leak parity (0.265)

### §210.1 Vector / SRAM-port util in P_clk
§176 left vector lanes and SRAM ports out of the gating util. `Chip.clk_datapath_gate` (default False, omit) switches the gated share to u = mean(u_mac, u_vec, u_sram_port), with:
- u_mac = MAC slot-cycles / (macs·f·card_s)
- u_vec = vector element-ops / (lanes·f·card_s)
- u_sram = SRAM-port bytes / (port_Bpc·f·card_s)

Equal weight across the three domains is 「假设」 (no measured clock-tree split for this NPU class). g = `clk_ungated_share` is unchanged. Default False keeps 0.229 fingerprints. Example (qwen3-8b decode B64, g=0.25): mac util ≈ 0.066, mean ≈ 0.042 → clock W/card 15.8 → 14.9.

### §210.2 PD SLC leak under idle
`pdqueue._energy` had kept the 0.260 rule (SLC suppressed by idle). Aligned with §208.1: SLC MiB always charged; shared SRAM idle-gated.

### §210.3 Freshness
`freshness_sweep.py 2026-10-20`: nothing new.

## §211 SLC access energy vs capacity (0.266)

### §211.1 Wire / H-tree term ∝ √C
Until 0.265, `pJ_bit_slc` was a flat 「假设」 equal to SRAM array energy. Real large SRAMs pay interconnect energy that grows with capacity: Horowitz ISSCC 2014 Fig. 1.1.9 (45 nm) lists 64-bit SRAM accesses at 8 KB / 32 KB / 1 MB as 10 / 20 / 100 pJ — E ∝ √C 「推导」. CACTI-based LLC studies (ISQED 2023 / TED 2022) find H-tree energy dominates at cache scale.

**Model.** `energy.slc_access_ref_mib = C_ref` (MiB). Effective pJ/bit = `pJ_bit_slc · max(1, √(slc_mib / C_ref))`. `C_ref = 1` anchors at Horowitz's 1 MB point. Below `C_ref` the factor stays 1 (wire does not get cheaper than the array term). Unset = flat (default; catalog fingerprints unchanged). Report field `slc_access`. PD `_energy` uses the same helper per pool chip.

**Effect** (qwen3-8b, 100T, 8 GiB SLC, tpuv4i_7nm, `C_ref = 1`): scale ≈ 90.5; SLC action J/token rises ~90× vs flat — a GiB-scale SLC can lose on *dynamic* access energy as well as leakage (§206–§208).

### §211.2 Freshness
`freshness_sweep.py 2026-10-20`: nothing new.

## §212 Distance-based SLC wire energy (0.267)

### §212.1 Model
§211's √C rule extrapolates Horowitz's 8 KB–1 MB points to GiB; for a banked SLC the access goes to one bank, so the extra cost is the wire to it, set by distance, not capacity per se. `energy.pJ_bit_mm_onchip` turns on the distance model:

pJ/bit(SLC) = `pJ_bit_slc` (array, = SRAM 「假设」) + `pJ_bit_mm_onchip` · d, d = ½·√(slc_mib · `slc_mm2_per_mib`)

d is the mean Manhattan distance from a die-centre port to a uniform point of a square array of side L (E|x| + E|y| = L/4 + L/4).
- **Wire energy** 「来源/推导」: Keckler, Dally, Khailany, Garland, Glasco, "GPUs and the Future of Parallel Computing", IEEE Micro 2011, table of energy: moving 256 bits across 10 mm = 310 pJ at 40 nm (0.121 pJ/bit/mm, 50 % toggle), projected 200 pJ at 10 nm / 0.75 V (0.078 pJ/bit/mm). Wires scale little with node; the 10 nm figure is the suggested value (UI placeholder). Not preset (would move every SLC energy number).
- **Density** default `SLC_MM2_PER_MIB_DEFAULT` = 8.388608 / 35.5 = 0.236 mm²/MiB, the Aoyagi 3 nm HD macro (VLSI 2025, same source as the leakage). Array only → distance is a lower bound 「近似」.
- `slc_access` reports `wire_mm`, `array_mm2`, `over_reticle` (> 858 mm²: a GiB SLC on one die is physically implausible; read as a multi-die / 3D-stacked proxy).
- Mutually exclusive with `slc_access_ref_mib` (ValueError).

### §212.2 Effect (qwen3-30b-a3b, 100T + LPDDR5X, B8, dip, tpuv4i_7nm, 25 °C, no idle)

| GiB | flat J/tok | distance (0.078) | √C (ref 1 MiB) |
|---|---|---|---|
| 0 | 0.2561 | 0.2561 | 0.2561 |
| 8 | 0.2281 | 0.2393 (22 mm, 1.80 pJ/b) | 0.2801 |
| 32 | 0.1673 | 0.2267 (44 mm, 3.52 pJ/b) | 0.4427 |

Distance keeps 32 GiB J-optimal at 25 °C with the saving cut from −35 % to −11.5 %; √C reverses the conclusion. SLC access stays below LPDDR5X memory + PHY (4.0 + 1.3 pJ/bit) up to ≈ 60 mm, which is why the distance model still favours a large SLC on a cool die. At 85 °C leakage dominates either way (opt 0 GiB).

### §212.3 Freshness
`freshness_sweep.py 2026-10-20`: nothing new; errored orgs (inclusionAI, allenai, hpcai-tech) rechecked by API — newest releases pre-window (inclusionAI Ling-3.1-flash 10-13, Ming-3.0-asr-tiny 10-14; allenai HiRO-ACEv2 10-12).

## §213 √C SLC deprecation; Ming-3.0-asr-tiny (0.268)

### §213.1 √C `slc_access_ref_mib` deprecated
§212.2 showed the √C rule (Horowitz 8 KB / 32 KB / 1 MB, §211) reverses the SLC-size conclusion when extrapolated three decades up to GiB, while the distance model (§212, Keckler 2011 wire energy × mean Manhattan distance) keeps it. Retired softly: behaviour unchanged (configs and fingerprints keep working), but `slc_access` carries `deprecated: true` and `note` = `energy.SLC_SQRTC_DEPRECATION`, the SLC insight report lists the note, and the UI field / KPI are tagged 「已弃用」. Re-anchoring was rejected: any anchor ≥ 1 MiB still applies a capacity law outside its data range; the distance model is the mechanism the √C rule was standing in for.

### §213.2 Ming-3.0-asr-tiny
- **Decoder** 「来源」 config + header: `text_config` of the BailingMoeV3 family (as Ling-3.x): h 1536, 24 layers, `layer_group_size` 4 (3 KDA, full-rank `f_proj` (`no_kda_lora`), short conv 4; 1 MLA q_lora 256 / kv_lora 512 / rope 64 / nope 128 / v 128, head-wise output gate), 128 experts top-8 (d 512) + 1 shared (512), first layer dense (4608), vocab 157,184, MTP 1. Routed via `_LING`; params 7,893,392,800 (+ MTP 316,604,800) = header exactly.
- **Audio encoder** 「来源」 `modeling_whisper_encoder.py`, `modeling_ming_v3_asr_encoder.py`, `audio_spec.py`: log-mel 128 bins, 16 kHz, hop 160 (torch.stft center, last frame dropped → ⌊samples/160⌋ frames); clips are never padded to 30 s; frames split into 3000-frame chunks, each encoded separately (attention within a chunk). conv1 (k3) + conv2 (k3 s2) + GELU, sinusoid positions (stored buffer 3000 × 1280), 32 pre-LN blocks (d 1280, 20 heads, bidirectional, GELU MLP 5120, key without bias), ln_post; projector Conv1d 1280→1536 k3 s2 p1 + GELU + Linear 1536→1536. Tokens per chunk n: t = ⌈n/2⌉, o = ⌊(t − 1)/2⌋ + 1 → 25 tokens/s (30 s → 750; 30.5 s → 763). Run params 647,149,568; `linear_proj` (4096 → 1536 → 1536, listed in `_keys_to_ignore_on_load_unexpected`) 8,653,824 stored only; total 655,803,392 = `params_vision` exactly.
- **Cost** per 30 s chunk: 1.918 TFLOP GEMM + 0.369 TFLOP attention = 2.286 TFLOP (hand count = model). Placement as the Muse Spark audio encoder (§175: data-parallel on the first stage, before prefill). B200 bf16, 1 card: 30 s clip ≈ 1.8 ms encoder; TTFT (64-token prompt) 2.8 → 5.4 ms with the 750 audio tokens.
- **Limits.** Listing domain `asr` (audio_config without vision_config). The encoder rides in a `VisionSpec` of family `audio` (no image path; `serving.images` ignored with a warning). Encoder activation peak is not added to stage-0 memory (same as Muse audio). The decoder's ASR output format/prompt template is not modelled (prompt = user-set text tokens + audio tokens).

### §213.3 Freshness
`freshness_sweep.py 2026-10-20`: nothing new, no errors. inclusionAI rechecked: Ming-3.0-asr-tiny (10-14) added; Ling-3.1-flash (10-13) was already in the catalog (§217 listing it as a gap was wrong). allenai newest HiRO-ACEv2 (10-12, climate emulator — out of scope); hpcai-tech newest 2025-03.

## §214 Speech synthesis decode head (Ming-3.0-tts-tiny); re-read hot weights (0.269)

### §214.1 TTS head
「来源」 inclusionAI/Ming-3.0-tts-tiny: modeling_lingv3_tts.py `generate_speech`, CFM.py, dit.py, aggregator.py, modules.py, vae_modules.py, istft.py, config.json; safetensors header.
- **Decoder** = `llm_config` (BailingMoeV3, identical to Ming-3.0-asr-tiny's decoder, §213.2), no MTP: 7,893,392,800 params exact. The release summary files flow / stop / spk heads under `params_llm` (no "audio" in their names); `release_params` subtracts them.
- **One step** (default `generate`: cfg_scale 2.0, flow_steps 10, EPSS timesteps, Euler + noise): LLM token → last hidden h →
  1. stop_head Linear(1536 → 2);
  2. flow: 10 DiT evaluations, batch × 2 (CFG cond + zeroed cond), sequence 3 = [t-emb + c-emb, x-emb(history 1), x-emb(patch 1)]; 16 pre-RMSNorm blocks d1024 (biased q/k/v/out, GELU-tanh MLP 4096), final RMSNorm + Linear(1024 → 64) on all tokens; t_embedder (256 → 1024 → 1024) and c_embedder (1536 → 1024) re-run per evaluation (not cached in the reference);
  3. aggregator: [cls, x-emb(latent)] → 8 blocks d1024 → Linear(1024 → 1536) = next input embedding;
  4. streaming AudioVAE decode: fc1 (64 → 896), ×4 linear upsample, Qwen2 decoder 24 layers (GQA 14/2, head 64, SwiGLU 4864, sliding window 64, KV cache), ISTFT head Linear(896 → 3530), irfft n_fft = 4 · hop = 3528 + overlap-add. 4 frames × 882 samples / 44.1 kHz = 80 ms per step (max 250 steps = 20 s).
- **Params (header, exact)**: flow 204,526,688; aggregator 102,396,448; AudioVAE decoder 361,127,186; stop 3,074 — read every step (1.34 GB bf16). Stored only: AudioVAE encoder + its aggregator 419,261,824 (voice-prompt latents), spk_head 296,448 (once per request), lm_head 241,434,624 (unused by the TTS loop). Total 8,981,004,468 = HF.
- **Per step, one request**: flow 24.30 GFLOP (10 × 6 rows × 16 × 2 · 12 · 1024²), VAE 2.91, aggregator 0.41, LLM ≈ 2 × active params ≈ 2.8 (incl. the embedding / lm_head tables active_params counts). Weights read 5.02 GB (flow 10×).
- **Placement** 「假设」: the head runs on the head stage, replicated on its TP ranks (vLLM-style small heads), after the final norm; prefill ends with the first step.
- **Limits**: plain TTS only (no voice-prompt encode, no podcast multi-speaker); stop decided per step (no early-exit batching model); EPSS / sway timesteps change no cost; the AudioVAE decoder K/V ring (786 KB / request) is per-sequence state.

### §214.2 Re-read hot weights
The flow DiT is read 10 × per step but stored once, so the ≤ 0.268 uniform miss rule (every hot byte misses at 1 − pinned/hot) undervalues pinning it. Following the 0.192 rule (pin by DRAM bytes saved per step per SRAM byte): a byte read R times saves R B/B, so the re-read class (`StageStorage.rr_w`, `rr_reads` = R) goes before state (2 B/B, when R > 2) and ordinary hot weights in SRAM, and first in the `pin` SLC; its reads miss at their own share (`memplan.hot_miss`, `MemPlan.pinned_rr / slc_rr`). `dip` / `lru` keep the uniform rule. Nothing else declares a re-read class → all earlier results unchanged.

| 100T + LPDDR5X, B1, pin | TPOT ms | RTF |
|---|---|---|
| asr decoder alone, no SLC | 11.7 | — |
| TTS, no SLC | 32.5 | 0.41 |
| TTS, SLC 0.5 GiB | 13.5 | 0.17 |
| TTS, SLC 2 GiB | 5.1 | 0.06 |

(Uniform rule, for comparison: 1 GiB → 20.3 ms.) Finding: for flow-matching TTS heads, a sub-GiB SLC sized to the flow DiT is the lever; B200 (HBM3e) is already 1.6 ms.

`/api/health` `domains` now returns `catalog.DOMAINS` (was hard-coded without asr / tts).

### §214.3 Freshness
`freshness_sweep.py 2026-10-20`: nothing new, no errors (74 orgs).

## §215 Speech round: encoder-decoder ASR, audio-seconds convention (0.270)

### §215.1 Whisper-large-v3 / -turbo
「来源」 transformers `modeling_whisper.py` (WhisperEncoder, WhisperDecoderLayer with `encoder_attn`, EncoderDecoderCache), `feature_extraction_whisper.py` (padding="max_length", n_samples 480000); openai/whisper-large-v3(-turbo) config.json + safetensors header.
- **Encoder** (both): 128 mel, conv1 k3 + conv2 k3 s2 (GELU), sinusoid positions [1500, 1280] saved in the checkpoint, 32 pre-LN blocks d1280 / 20 heads (q / v / o biased, k not), GELU 5120, final LayerNorm. 636,968,960 params, exact. Every pass is padded to 30 s: 3000 frames → 1500 positions, whatever the clip length; a clip > 30 s is sequential windows (long-form), each a full pass. One window ≈ 2.28 TFLOP (hand count = ops, test).
- **Decoder**: per layer self-attention (MHA 20 × 64) + cross-attention over the encoder output + GELU MLP; learned positions [448, 1280] (lookup); final LayerNorm; proj_out tied to embed_tokens (51866). v3 32 layers = 906,521,600, turbo 4 layers = 171,909,120, exact (16d² + 17d per layer). `max_target_positions` 448 = `max_ctx`.
- **Cross-attention decode path** 「推导」 from EncoderDecoderCache: the first prefill pass (ctx = 0) runs `cross.k / cross.v` on 1500 rows per sequence and writes the cache; every later forward runs only `cross.q`, QKᵀ / PV against 1500 keys (non-causal), `cross.o` (+ TP all-reduce), reading the cached K/V. Cross K/V per sequence = L · 1500 · 2 · d · 2 B (v3 245.76 MB, turbo 30.72 MB) is fixed state per request (memplan `state`), also counted in the PD hand-off (`cache_new_bytes`).
- **Limits** 「近似」: decoder re-prompting with previous-window text (long-form condition_on_prev_tokens) not modelled — each window's decode is the request's `out_len` share; the RoPE vector op is still charged on self-attention (Whisper has none; a small per-token vector op); GELU costed as SiLU per element.

### §215.2 Qwen2-Audio-7B-Instruct, Qwen3-ASR-1.7B
- **Qwen2-Audio** (transformers `modeling_qwen2_audio.py`, `processing_qwen2_audio.py` 「来源」): Whisper-v3 encoder + AvgPool1d(2, 2) before the final LayerNorm + Linear(1280 → 4096) = 642,215,936, exact. Processor pads to 30 s and truncates longer audio (`truncation=True`): the encoder always runs 1500 positions, the projector 750 rows; prompt audio tokens = (((L − 1)//2 + 1) − 2)//2 + 1 over the unpadded frames L (750 per 30 s). LLM = Qwen-7B (text_config lists only overrides; the rest from Qwen2Config defaults: 4096, 32 layers, MHA 32, vocab 156032, untied), 7,754,878,976, exact.
- **Qwen3-ASR** (transformers `modeling_qwen3_asr.py` 「来源」): AuT encoder — frames in 100-frame chunks (2 · n_window), three Conv2d k3 s2 p1 (1 → 480 → 480 → 480) over (mel, time) → 16 × 13 per chunk, conv_out Linear 7680 → 1024 (no bias), sinusoid positions, only valid positions (13 per full chunk, the tail chunk by three ⌊(n − 1)/2⌋ + 1) go through 24 blocks d1024 / 16 heads (all of q / k / v / o biased) with attention inside windows of 13 · n_window_infer / 100 = 104 positions, ln_post, projector Linear 1024 → 1024 → GELU → 2048. 317,477,504, exact. Audio tokens = encoder positions (13 / s). LLM = Qwen3-1.7B (thinker_config.text_config; the release stores a tied lm_head copy, deduped), 1,720,574,976, exact.

### §215.3 Audio-seconds serving convention
Speech workloads are reported per audio second (`core/speech.py`, `Result.asr`):
- **ASR**: request = `audio_s` s of audio + `prompt` text tokens → `out_len` text tokens. First-token latency T_ft = encoder windows + prefill (prefill phase); TPOT at the mid-generation context; T = T_ft + out_len · TPOT (static batch, phases serial 「近似」); RTF = T / audio_s; audio s / s / card = batch · audio_s / T / cards. Per phase (`Result.asr`): prefill audio s / s = batch · audio_s / step; decode RTF_decode = out_len · TPOT / audio_s.
- **TTS**: `audio_out_s` s out = ⌈audio_out_s / s_per_step⌉ steps (Ming: 80 ms / step); first audio = T_ft + one step (streaming decoder emits each step's chunk); RTF = steps · TPOT / audio_out_s.
- `out_len` and `audio_out_s` are the caller's workload assumption 「假设」; nothing is fitted.

| 30 s in, 100 tokens out | 100T + LPDDR5X B1 RTF | B32 RTF | B32 audio-s/s/card | H100-like B32 RTF |
|---|---|---|---|---|
| whisper-large-v3 | 0.034 | 0.218 | 147 | 0.022 |
| whisper-large-v3-turbo | 0.0064 | 0.052 | 615 | 0.0054 |
| qwen2-audio-7b-instruct | 0.259 | 0.624 | 51 | 0.054 |
| qwen3-asr-1.7b | 0.061 | 0.106 | 303 | 0.009 |
| ming-3.0-asr-tiny | 0.043 | 0.305 | 105 | — |

Findings: (1) Whisper-v3 decode at batch is bound by the cross K/V, not the weights — B32 reads 7.9 GB of cross K/V per step vs 1.8 GB of weights (TPOT 9.7 → 56 ms). (2) Turbo removes 7/8 of both, after which the unchanged 32-layer encoder dominates (B32: 779 of 798 ms to first token): for turbo-class ASR, encoder FLOPs per audio second are the lever. (3) The AuT encoder's 13 positions / s (vs 50 for the Whisper encoder) makes Qwen3-ASR's encoder ~6× cheaper per audio second.

### §215.4 Freshness
`freshness_sweep.py 2026-10-20`: nothing new (74 orgs). 14 speech orgs added to `EXTRA_ORGS`; nothing since 10-20 there. Seen earlier (candidates for 0.271): fishaudio/drama-3 (10-01), IndexTeam Index-Echo S2ST / S2TT (10-03), FunAudioLLM Fun-ASR-Nano-2512, FireRedTTS3.


## §216 Speech round 2: Qwen2.5-Omni speech out, CosyVoice2 / Fun-CosyVoice3 (0.271)

### §216.1 Qwen2.5-Omni-7B: Thinker and audio tower
「来源」 transformers `modeling_qwen2_5_omni.py` / `configuration_qwen2_5_omni.py` / `processing_qwen2_5_omni.py`; Qwen/Qwen2.5-Omni-7B config.json + safetensors header.
- **Thinker**: Qwen2.5-7B (h3584, 28 layers, GQA 28/4, SwiGLU 18944, vocab 152064, untied, q/k/v bias), from `thinker_config.text_config`. 7,615,616,512, exact. This is the catalog LLM.
- **Audio tower**:
  - Structure: Qwen2_5OmniAudioEncoder = Whisper-v3 (conv1 + conv2 s2, 32 × d1280 / 20 heads, k without bias). The encoder runs only the valid frames, in chunks of 2 · n_window = 200 frames with attention inside a chunk (cu_seqlens). Then ln_post, AvgPool1d(2, 2) and proj 1280 → 3584.
  - Params: the sinusoid positions are a buffer, not in the checkpoint; `audio_bos_eos_token` (2 × 3584) is stored. 639,647,232, exact.
  - Tokens: the processor pads to 30 s with truncation, and feature_lens = ⌈samples / 160⌉ ≤ 3000. Tokens = (((L − 1)//2 + 1) − 2)//2 + 1, which is 25 / s and 750 at 30 s.
  - Code: `AudioSpec.maxf / stored`, `_omni_audio_spec`; the whisper ops are run with `pad = False`.
- **ViT**: the Qwen2.5-VL ViT (`thinker.visual`, 676,550,144) is stored only (`VisionSpec.extra`). Image and video prompts are not costed, and that is the coverage reason (partial).
- **Release roles**: `talker.*` and `token2wav.*` are their own roles (`params_talker` 1,351,360,256 bf16; `params_token2wav` 449,051,296 fp32). LLM + vision + talker + token2wav = the header total.

### §216.2 Qwen2.5-Omni speech output: Talker → DiT → BigVGAN (`core/omni.py`)
「来源」 the same transformers code, plus vLLM-Omni `qwen2_5_omni_token2wav.py` for streaming. All parameter formulas match the header exactly.
- **Talker** (bf16): 24 layers, h896, 12 heads × 128 (q 1536), kv 4 × 128, SwiGLU 18944, q/k/v bias, codec vocab 8448.
  - Input per position = codec embedding (8448 × 3584, looked up) + Thinker embedding / hidden, passed through thinker_to_talker_proj 3584 → 896.
  - It prefills over the Thinker prompt (text + audio positions), then runs one step per codec token.
  - Per-layer params 54,595,840; total 1,351,360,256.
- **Rates** 「推导」: BigVGAN hop = Π(5, 3, 2, 2, 2, 2) = 240 samples at 24 kHz (README sf.write samplerate 24000), so 100 mel/s. DiT `repeats` 2 gives 50 codec tokens/s.
- **DiT** (fp32; the reference forces fp32):
  - Blocks: 22 blocks d1024, 16 × 64 heads. AdaLayerNormZero Linear d → 6d on the time embedding (per row-batch, not per frame). GELU-tanh MLP 2048. Block mask (24 frames; layer 10 looks one block ahead, layers 0 and 20 one block back) is applied as a masked SDPA, so scores are dense T × T.
  - Input: Linear 912 → 1024 over [noisy mel 80, ECAPA vector 128, codec embedding 512 ×2 repeat, speaker embedding 192].
  - ECAPA-TDNN: res2net scale 2, SE 64, ASP 64; 1,729,280 params. It re-runs on the reference mel in every evaluation; the reference length of 300 frames (3 s) is an assumption 「假设」.
  - Sampler: RK4 over linspace(0, 1, 10) with sway −1 gives 9 intervals × 4 = 36 evaluations, each one batch with CFG (cond + uncond, guidance 0.5).
  - Params: 333,607,792.
- **BigVGAN** (fp32): conv_pre k7 80 → 1536, then 6 × [ConvTranspose (k 11, 7, 4, 4, 4, 4; channels halve) + 3 AMPBlocks (k 3/7/11 × dilations 1, 3, 5, two convs each)], then SnakeBeta and conv_post k7 without bias. 115,443,504 params.
  - Anti-aliased SnakeBeta is costed at 56 vector ops per element 「近似」.
- **Streaming** (vLLM-Omni Qwen2_5OmniToken2WavModel, "aligned with Qwen2Code2wav"):
  - Geometry: bs_mel 24, chunk 72 mel (0.72 s), past 48, future 24.
  - DiT window of chunk i = [72i − 48, 72(i + 1) + 24): 96 frames for the first chunk, 144 after that.
  - Vocoder input: 72 frames for the first chunk, then 72 + 2 · future_size 20 = 112.
  - The first chunk emits (72 − 20) · 240 samples = 0.52 s. token2wav supports batch 1 only.
- **KPIs** (`speech._omni_kpis`):
  - Placement assumption 「假设」: Talker + token2wav on one extra card of the same chip per replica (vLLM-Omni runs them as separate pipeline stages).
  - First audio = T_ft(Thinker, incl. audio tower) + one Thinker TPOT 「假设」 + Talker prefill + 48 Talker steps (the codec of DiT window 0) + DiT chunk 0 + vocoder chunk 0.
  - Speech-card s per audio second for b streams = 50 · t_step(b) + b · (100 / 72) · (t_DiT,144 + t_voc,112). This is each stream's RTF; streams_realtime is the largest b with RTF ≤ 1.
  - DiT weights: the 0.269 re-read rule applies (R = 36 > 2 B/B, so pinned first). The first evaluation reads all weights; the other 35 read the miss share 1 − min(W, SRAM − staging) / W.
  - The Talker K/V reads are added to the component path, which otherwise ignores K/V.

| 100T + LPDDR5X, 64-token prompt + 10 s audio in, 10 s out | B1 | B4 |
|---|---|---|
| first audio (s) | 1.08 (Thinker 0.08 + 0.07, Talker 0.01 + 48 × 14 ms = 0.67, DiT 0.24, voc 0.002) | 1.24 (last stream 1.96) |
| speech-card RTF | 1.03 | 2.05 |
| DiT re-read miss | 0.95 (64 MiB SRAM vs 1.32 GB) | |

Finding: the Talker (2.64 GB bf16 = 1.32 B streamed params read per codec token, 50 per second; 0.272 fix: was misprinted as 1.33 GB) costs as much as the whole DiT path. On a bandwidth-poor card, Omni speech output is not real time for even one stream; the lever is Talker weight bandwidth, or batching streams (t_step barely grows with b). The DiT is weight-re-read bound at 36 evaluations per chunk, so a ≥ 1.3 GB SRAM / SLC pin would cut its 0.24 s per chunk (test: 4 GiB SRAM → miss 0, chunk faster).

### §216.3 CosyVoice2-0.5B / Fun-CosyVoice3-0.5B-2512 (`core/cosyvoice.py`)
「来源」 github FunAudioLLM/CosyVoice (cli/model.py, flow/flow.py, flow_matching.py, decoder.py, DiT/dit.py, hifigan/generator.py), cosyvoice2.yaml / cosyvoice3.yaml.
- **Release metadata**: the releases ship torch `.pt` checkpoints only. `scripts/fetch_pt_release.py` reads each state dict's tensor names, shapes and dtypes from the pickled data.pkl, located through the zip central directory with HTTP Range requests (no download, no torch).
  - llm.pt: 641,938,468 / 642,283,136.
  - flow.pt: 112,549,360 / 332,257,120.
  - hift.pt: 20,821,295 / 20,779,887.
  - All fp32. `summarize_release._cosy_role` assigns roles; flow and hift shapes are kept as `speech_tensors`.
- **LM**: the Qwen2.5-0.5B backbone (CosyVoice-BlankEN config, q/k/v bias) decodes speech tokens at 25 per audio second.
  - Vocab = the speech table / `llm_decoder` rows: 6564 for v2 (+ bias), 6761 for v3.
  - The text embedding and the stored text `lm_head` (unused in the TTS loop) are fp32 lookup tables (`lookup_bits` 32).
  - Exact, except the 6564-param llm_decoder bias of v2. Activations stay bf16 under the tool's release rule; CosyVoice's default is fp32 「近似」.
- **Flow** (batch 1): every streamed chunk calls flow.inference from scratch on [prompt tokens + all tokens so far (+3 look-ahead)].
  - CosyVoice2: UpsampleConformerEncoder (6 blocks at 25 Hz, up conv, 4 blocks at 50 Hz, rel-pos attention with linear_pos over 2T − 1 positions and an extra q · posᵀ) + U-Net estimator (256 channels, ResNet1D blocks + 1 + 12 + 1 groups of transformer blocks, attention 8 × 64).
  - CosyVoice3: pre-look-ahead conv + DiT (22 × d1024, the Omni block, plus grouped conv positional embedding).
  - CFM: 10 Euler steps (cosine t), CFG as batch 2 (inference_cfg_rate 0.7).
  - Ops are header-driven: one GEMM per weight at its module's rate.
    - Encoder: token rate or mel rate.
    - Estimator: B · M rows per evaluation; time MLP / AdaLN / ResNet time projection on B rows.
    - Biases and weight-norm gains are folded in, and leftover 1-D tensors go to a read-only op, so weight reads equal the release exactly.
  - The estimator re-read pin is the same 0.269 rule (R = 10).
- **HiFT**: f0 predictor and conv_pre at the mel rate; upsample stages ×8 / ×40 / ×120. v2 uses ConvTranspose per input position; v3 uses causal upsample + Conv1d per output position. ResBlocks and source branches run at their stage rate, conv_post at ×120, and the iSTFT (n_fft 16, hop 4) gives 24 kHz.
  - CosyVoice2 vocodes the new mel + an 8-frame cache.
  - CosyVoice3 re-vocodes the whole generated mel every chunk (hift_cache 'mel' grows) 「来源」.
- **Streaming hops** (cli/model.py):
  - token_hop_len 25, plus the prompt padded to a multiple of 25 on the first chunk; the hop doubles per chunk up to 100.
  - The first chunk waits for hop + 3 look-ahead tokens.
  - `self.token_hop_len` is an instance attribute that is never reset, so on a long-lived model object every later request starts at 100 tokens. This is reported as the warm first audio.
- **KPIs** (`speech._cosy_kpis`), same placement assumption as Omni:
  - First audio = T_ft + (hop₀ + 3) · TPOT + flow(chunk 0) + HiFT(chunk 0).
  - Speech-card RTF = b · Σ_chunks (flow + HiFT) / audio_out_s.
  - LM RTF = 25 · TPOT.
  - The zero-shot prompt is 75 speech tokens (3 s) 「假设」.

| 100T + LPDDR5X B1, 10 s out | first audio | warm first audio | LM RTF | speech-card RTF | real-time streams / card |
|---|---|---|---|---|---|
| CosyVoice2 | 0.22 s | 0.78 s | 0.18 | 0.012 | 86 |
| Fun-CosyVoice3 | 0.28 s | 0.83 s | 0.18 | 0.033 | 30 |

Findings:
1. First audio is LM-paced: 28 tokens × 7.3 ms. The fp32 LM reads 2.0 GB per token.
2. The hop that is never reset makes the warm first audio ≈ 3.5× the cold one.
3. Fun-CosyVoice3's fp32 DiT is weight-bound, at 66 ms per chunk independent of length; CosyVoice2's U-Net grows with the recomputed length (13 → 36 ms).
4. CosyVoice3 re-vocoding grows linearly with the utterance (0.6 → 33 ms).

### §216.4 Freshness and deferred pre-window speech releases
`freshness_sweep.py 2026-10-17` (88 orgs): only DeepSeek-V4.1-Flash-1017 (an alias since 0.248) and repacks / GGUFs (unsloth, tencent Hy-Memory GGUF). Nothing since 10-20.

Deferred, with public configs noted:
- **FunAudioLLM/Fun-ASR-Nano-2512-hf**: transformers `fun_asr_nano`. SAN-M encoder: 70 layers d512, FSMN k11, LFR 7, 80 mel. Adaptor: 2 layers. LLM: Qwen3-0.6B.
- **fishaudio/drama-3**: FishOmniModel, about 40 GB bf16, with a `codebook_mtp_shared` audio decoder (10 codebooks) and codec.pth.
- **IndexTeam Index-Echo-S2ST 2B / 9B**: a bridge mapper plus a vendored CosyVoice pipeline.
- **FireRedTTS3**: custom (base / instruct 8.5 GB + ReDAE 3.8 GB).

## §217 Speech round 3: Fun-ASR-Nano, Step-Audio-2-mini, Omni speech-card SLC pin (0.272)

### §217.1 Fun-ASR-Nano-2512 (`AudioSpec.kind = "sanm"`)
「来源」 transformers `models/fun_asr_nano` (modeling / configuration / feature_extraction / processing, main); FunAudioLLM/Fun-ASR-Nano-2512-hf config.json, processor_config.json, safetensors header (sha 7ef6caf, Apache-2.0).
- **Front end**:
  - Kaldi fbank: 80 mel, 25 ms window, 10 ms hop, snip_edges, so frames = 1 + (N − 400) // 160.
  - LFR: stack 7 frames, stride 6. That gives ⌈frames / 6⌉ positions of 560 features, about 16.7 / s.
  - Audio tokens = the mask sum, one LLM token per LFR frame: 500 per 30 s.
- **Encoder** (SenseVoice SAN-M):
  - Inputs are scaled by √d, plus sinusoid positions counted from 1. The table has 2049 rows, so at most 2048 positions (≈ 122.9 s); longer audio is reported as `truncated_s` 「来源/推导」.
  - 70 pre-LN blocks d512 / 4 heads:
    - q / k / v / o with bias; layer 0 maps 560 → 512 and has no residual.
    - FSMN: a depthwise Conv1d k11 without bias on the unsplit values, added to the attention output.
    - ReLU MLP 2048.
  - Final LayerNorm only on layers 49 and 69. All 70 layers run in forward, including the 20 timestamp-prediction layers.
  - Attention is bidirectional over the whole utterance (no chunking).
- **Projector**: Linear 512 → 2048, ReLU, Linear 2048 → 1024, then 2 adaptor blocks (d1024 / 8 heads, ReLU MLP 256, no FSMN).
- **LLM**: Qwen3-0.6B (tied).
- **Params** (exact): LLM 596,049,920; tower 233,741,920.
- **Ops**: every layer runs on all T positions. FSMN costs 2 · 11 vector ops per element. Weight reads = params minus LayerNorms. 30 s ≈ 0.27 TFLOP, versus 2.28 for a Whisper-v3 window.

### §217.2 Step-Audio-2-mini (audio in)
「来源」 stepfun-ai/Step-Audio-2-mini `modeling_step_audio_2.py`, config.json, safetensors header (sha e36fdd5, Apache-2.0).
- **LLM**: transformers Qwen2Model(text_config): 28 × h3584, GQA 28 / 4, SwiGLU 18944, q / k / v bias, vocab 158720, untied lm_head. 7,663,326,720 params, exact.
- **Encoder**:
  - Front end: log-mel 128 on audio padded by 479 samples; L = (N + 479) // 160 mel frames.
  - conv1 k3 + conv2 k3 s2 (GELU), then learned positions Embedding(1500, 1280), which is stored.
  - 32 Whisper blocks (key without bias), then AvgPool1d(2) and LayerNorm.
  - Adaptor: Conv1d k3 s2 p1 + GELU, Linear 1280 → 2048, ReLU, Linear 2048 → 3584.
  - 651,852,544 params, exact (`summarize_release._step_audio_role`: encoder.* / adapter.* → vision).
  - The whole clip goes through in one pass with full attention. Conv2 positions are capped by the 1500-row table, so L ≤ 2999 (≈ 29.99 s); the rest is reported as `truncated_s` 「近似」. The official app's long-audio segmentation is not modelled.
- **Tokens**: from the forward pass, ((L + 1)//2//2 − 1)//2 + 1, which is 12.5 / s (375 per 30 s).
- **Not modelled**: speech output. token2wav ships as `.pt`: a CosyVoice2 UpsampleConformerEncoderV2 + a DiT estimator (16 × d512) + HiFT. That combination differs from both 0.271 CosyVoice flows, so it is deferred.

| 30 s in, 100 tokens out, 100T + LPDDR5X | B1 RTF | B32 RTF | B32 audio-s / s / card |
|---|---|---|---|
| fun-asr-nano-2512 | 0.021 | 0.068 | 473 |
| step-audio-2-mini | 0.250 | 0.346 | 92 |
| qwen2-audio-7b-instruct (0.270) | 0.259 | 0.634 | 51 |
| qwen3-asr-1.7b (0.270) | 0.061 | 0.108 | 296 |

Finding: Step-Audio-2-mini and Qwen2-Audio share a decoder of the same size, but Step-Audio-2 puts half as many audio positions in the prompt (375 vs 750 per 30 s, from the adaptor's stride 2). That cuts its B32 first token from 4.26 s to 2.54 s and its TPOT from 147 to 78 ms (K/V reads), so B32 throughput is 1.8×. Fun-ASR-Nano's 70-layer encoder is cheap per second (16.7 positions / s, d512): its B32 first token is 0.31 s, of which the encoder is 0.12 s.

### §217.3 Omni speech card: SLC pin (DiT re-read first, then the Talker)
- **Rule** (`omni.omni_slc_plan`, the 0.192 / 0.269 rule): fill the SLC by DRAM bytes saved per audio second per byte, counting only the bytes SRAM does not already hold.
  - DiT weights: read 36 × 100/72 × b = 50 · b times / s (token2wav batch 1).
  - Talker weights (2.64 GB bf16): read once per batched step, 50 times / s whatever b.
  - The DiT therefore goes first, including on the b = 1 tie (re-read class). DiT bytes already in SRAM and BigVGAN are left out 「假设」.
- **Timing**: hits come from `slc_GBps` on a port parallel to DRAM, so t = max(component time on the DRAM share, SLC bytes / slc_GBps) 「近似」. Only `slc_policy` "pin" applies. With no SLC or another policy, the numbers are identical to 0.271 (test).

| 100T + LPDDR5X, slc_GBps 2000 「假设」, 10 s out | B1 RTF | B4 RTF | B8 RTF | Talker step B1 | real-time streams |
|---|---|---|---|---|---|
| no SLC | 1.03 | 2.05 | 3.40 | 14.0 ms | 0 |
| 1 GiB (DiT 86 %) | 0.79 | 1.07 | 1.45 | 14.0 ms | 3 |
| 2 GiB (DiT 100 %, Talker 34 %) | 0.55 | 0.84 | 1.21 | 9.3 ms | 5 |
| 4 GiB (both) | 0.17 | 0.46 | 0.85 | 1.7 ms | 9 |

Findings:
1. Batching alone does not help. The Talker step barely grows (14.0 → 15.0 ms from B1 to B8), but token2wav runs at batch 1, so DiT time per audio second grows with b (0.33 → 2.63 s). The speech card is not real time at any b without an SLC.
2. A 1 GiB pin of the DiT re-read weights already gives 3 streams.
3. Holding both the DiT (1.27 GB miss share) and the Talker (2.64 GB) needs about 3.9 GB of SLC. That cuts the Talker step 8× and gives 9 real-time streams.
4. The design lever for Omni speech output is SLC capacity of at least the DiT + Talker weight set, not more MACs. Bandwidth-poor cards stay Talker-bound without it.

### §217.4 Freshness and deferred
- `freshness_sweep.py 2026-10-20` (88 orgs): nothing. HF trending, created on or after 10-18: nothing.
- **Deferred**:
  - fishaudio/drama-3: FishOmni. Qwen3.5-style hybrid text model: 40 layers, 30 GDN + 10 full, 256 experts top-8, fp8 experts, 35.9 B stored. Plus a 4-layer, 10-codebook MTP audio decoder and codec.pth. The modeling code is not in the repo.
  - IndexTeam Index-Echo-S2ST: STLM omni (15 shards) + distcp checkpoint + bridge mapper + vendored CosyVoice3.
  - FireRedTTS3: custom FireRedTTS3BaseCore (patch encoder 8 × 1024, DiT 11 × 1024, ReDAE); the HF repo has configs + weights only, no modeling code.
  - Kimi-Audio-7B; Step-Audio-2 token2wav.

## §218 Speech round 4: speech blocks in the UI, Step-Audio-2 token2wav, Kimi-Audio-7B-Instruct (0.273)

### §218.1 UI
The web client now renders every speech result block the API returns:
- Omni: `split_first`, `slc` (0.272) and `audio_in`.
- CosyVoice / Step-Audio-2: `split_first`, `flow_s` / `hift_s`, `flow_pin_miss`, `label` / `stream`.
- Kimi-Audio: kind `kimi`.
- ASR: `max_audio_s`, `encoder_s`, `prompt_audio_tokens`.

`evaluate._asr_info` adds `max_audio_s`, the encoder cap in seconds (None = no cap), and the UI shows that instead of a hard-coded 30 s.

### §218.2 Step-Audio-2-mini token2wav (`CosySpec.stream = "cache"`)
Sources 「来源」 (github stepfun-ai/Step-Audio2): token2wav.py, examples-vllm-stream.py, cosyvoice2/flow/flow.py (`inference_chunk`, `setup_cache`), flow/decoder_dit.py, transformer/upsample_encoder_v2.py.

Params 「来源」, the released .pt metadata read via HTTP Range:
- flow 155,802,352 (fp32).
- HiFT 20,821,295: flashcosyvoice `HiFTGenerator()` defaults (ups 8/5/3, iSTFT 16/4, 24 kHz), identical to CosyVoice2's.

Per chunk 「来源」:
- N = 25 new tokens (CHUNK_SIZE, fixed) plus 3 look-ahead into the pre-look-ahead conv.
- Conformer:
  - 6 blocks on N rows over enc_off + N keys. The att cache holds every token since `setup_cache` and is never trimmed.
  - espnet rel-pos: `linear_pos` and q·posᵀ over 2·keys − 1 positions.
  - Then the ×2 up conv, and 4 blocks on M = 2N rows over 2·enc_off + M keys.
- Estimator: DiT with 16 blocks, d512, 8 × 64 heads, causal conv block k3 ×2 and adaLN 9d.
  - Per Euler step: B·M rows (CFG B = 2). The t-embedding / adaLN run on B rows.
  - Attention covers prompt mel + min(100, frames so far) cached frames + M.
- 10 Euler steps. HiFT on M + 8 cache frames.
- The voice prompt (`setup_cache`) is computed once per voice, not per request 「假设」.

First audio = T_ft + 28 · TPOT + flow(chunk 0) + HiFT(chunk 0). LM RTF = 25 · TPOT.

Text tokens interleaved with the speech tokens are not counted 「假设」, so the LM time is a lower bound.

The listing domain becomes `omni` once the release carries `token2wav`.

### §218.3 Kimi-Audio-7B-Instruct
Sources 「来源」:
- HF moonshotai/Kimi-Audio-7B-Instruct: config.json, modeling_moonshot_kimia.py, whisper-large-v3/, audio_detokenizer/config.yaml + model.pt, vocoder/config.json + model.pt.
- github MoonshotAI/Kimi-Audio: kimia_infer/api/kimia.py, models/detokenizer/*, models/tokenizer/whisper_Lv3/whisper.py.

`KimiAudioConfig` subclasses Qwen2Config without a model_type of its own, so it is recognised by `architectures` = MoonshotKimiaForCausalLM.

**LLM.**
- 28 Qwen2 layers (q/k/v bias, GQA 28/4, d3584, ffn 18944) plus `kimia_mimo_layers` = 6 layers of the same shape. The mimo layers take the hidden state after layer 21 and feed `mimo_norm` → `mimo_output` [168448, 3584].
- Every step runs both branches and both heads, and each layer has its own KV cache. It is costed as one 34-layer stack 「推导」: identical FLOPs, weight bytes and KV bytes; the possible overlap of the two branches is not modelled.
- `mimo_output` is an `io_post` Linear with rows "seq". In the autoregressive path, `ir._head_ops` now runs every `io_post` Linear of a KV-cache model next to the lm_head: vocab-parallel, with its own sample op and a TP logits gather. Prefill uses the last position only, like the lm_head 「推导」. Kimi is the only LLM with io_post.
- 9,735,124,992 params, exact. The 34 × 64 `rotary_emb.inv_freq` bf16 buffers in the safetensors are classed as buffers.

**Audio in.**
- whisper-large-v3 encoder: 636,968,960 params.
- Run per 30 s segment padded to 3000 mel frames. Output sliced to 4 · token_len positions, where token_len = (L − 1) // 1280 + 1 for a segment of L samples.
- Every 4 × 1280 positions are concatenated to 5120 → VQAdaptor (Linear 5120 → 3584, SiLU, Linear, LayerNorm): 31,209,472 params. 12.5 positions / s.
- The discrete GLM-4-Voice semantic tokens at the same positions come from THUDM/glm-4-voice-tokenizer: Whisper-v3 first 16 layers + 4× pooling + VQ 16384, in another repo. Not costed; this is the coverage reason (partial).

**Speech out** (`core/kimi_audio.py`):
- Each step emits a text token and an audio token. The first 6 audio slots are blank (`kimia_mimo_audiodelaytokens`).
- Detokenizer (bf16 at run time: `.to(bfloat16)`):
  - Tokens are repeat_interleaved ×4 to 50 Hz.
  - `detokenize_audio` chunks are 30 tokens (the first one too). `look_ahead_tokens` = 12 carried in front, so T = 168 rows (chunk 0: 120). The last 48 frames are dropped and recomputed by the next chunk.
  - t_span = linspace(0, 1, 30) through torchdyn Euler gives 29 evaluations. Then one evaluation at t = 0.999 over the kept frames writes their K/V (`infer_chunk`).
  - DiT: 16 blocks, d2304, 18 × 128 heads, MLP 4d. adaLN-Zero 6d **per frame**: c = t-embedding + semantic-token embedding, (N, T, D), so the modulation GEMM runs on T rows. RoPE; LayerNorm without affine.
  - Attention covers the cached kept frames (`max_kv_cache_tokens` 900, newest kept; the reserve stays 0) + T. The cached K/V is re-read every evaluation (16 × 2 × frames × 2304 × 2 bytes) 「推导」.
  - The final chunk keeps everything and runs no commit evaluation.
- BigVGAN (ups 5·2·2·2·2·3·2 = hop 480, 2048 initial channels, AMP resblocks k 3/5/7/11, anti-aliased SnakeBeta 「近似」 as §216): runs on held-back + new mel. `detokenize_streaming` returns half of chunk 0, then 2 × the held length (Hamming overlap); `KimiSpec.plan` follows it frame for frame (Σ returned = 4 · tokens).
- 1,584,023,120 DiT + 241,118,777 vocoder params, exact.

Formulas:
- First audio (feeding the chunk API as tokens arrive 「假设」) = T_ft + (6 + 30 − 1) · TPOT + DiT chunk 0 + BigVGAN(72). That yields 0.72 s of audio.
- The released kimia.py generates every token first and then detokenizes: `first_audio_released_s` = T_ft + (6 + n − 1) · TPOT + chunk 0.
- LM RTF = 12.5 · TPOT. Speech card: batch 1 per stream, as released 「来源」, on one extra card per replica 「假设」.
- DiT pin: the 0.269 re-read rule on the 3.09 GB bf16 weights.

**Finding** (100T + LPDDR5X, 10 s out, prompt 120 + 10 s audio in):
- TPOT 95 ms, so LM RTF 1.19.
- Streaming first audio 3.94 s, of which 3.34 s is the 35 LM steps. Released flow: 13.0 s.
- DiT ≈ 0.48 s per 2.4 s chunk, identical at 120 and 168 rows. It is weight-bound: 30 reads × 3.09 GB per chunk.
- BigVGAN ≈ 4–9 ms per chunk. Speech-card RTF 0.24, 4 real-time streams.
- Design lever: batching streams through the DiT would be nearly free (rows are tiny next to the weights), but the released detokenizer asserts batch 1. On-chip SRAM / SLC capacity for the 3.1 GB DiT is the other lever.

**Deferred.** Moshi (Temporal + Depth transformer, Mimi codec), FireRedTTS3 (no modeling code), drama-3, Index-Echo; the GLM-4-Voice tokenizer cost.

## §219 Speech round 5: Kyutai Moshi, cross-model speech insight (0.274)

### §219.1 Release and config
Sources 「来源」:
- github kyutai-labs/moshi@e6a55d2: `moshi/models/loaders.py` (`_lm_kwargs`, `_mimi_config`, `get_mimi`), `models/lm.py`, `modules/transformer.py`, `models/compression.py`, `modules/seanet.py`, `quantization/vq.py`, `server.py`.
- HF `kyutai/moshiko-pytorch-bf16`: model.safetensors and tokenizer-e351c8d8-checkpoint125.safetensors (Mimi) headers.

Config:
- The repo has no config.json. `fetch_pt_release.py --moshi` transcribes `lm` / `mimi` from loaders.py into `local/hf_cache/.../config.json` (`_source` cites the file and commit).
- test_core_274 checks every number used against the header shapes.

Temporal transformer (bf16), the catalog LLM `moshiko-7b`:
- d4096, 32 layers, 32 MHA heads; gated SiLU with hidden_scale 4.125, so 2·4.125·4096/3 = 11264.
- Text vocab 32000, untied head.
- Context 3000 is a KV ring buffer, modelled as a sliding window of 3000 「近似」.
- The input is the sum of the text embedding and 16 audio embeddings (8 user + 8 Moshi), modelled as lookups of h + 16 · 2049 · h.

Depformer (bf16):
- d1024, 6 layers, 16 heads, dep_q 8.
- `weights_per_step`: in_proj [24576, 1024] = 8 slices of 3·1024; out_proj [8192, 1024]; gating.0..7 (linear_in [5632, 1024] / linear_out [1024, 2816]).
- depformer_in × 8 [1024, 4096]; linears × 8 [2048, 1024]; embedding tables depformer_emb × 7 [2049, 1024] and text_emb [32001, 1024].

Mimi (fp32: `get_mimi` passes no dtype):
- 24 kHz, 12.5 Hz frames (1920 samples).
- SEANet ratios [8, 6, 5, 4]; the encoder applies them reversed.
- 8-layer transformer: d512, 8 heads, ffn 2048, context 250, at 25 Hz (2 positions per frame).
- Stride-2 conv; split RVQ (rvq_first + rvq_rest, 32 stored, 8 used), 2048 × 256.
- The decoder mirrors the encoder with depthwise ConvTranspose upsampling.

Params exact: 6,973,366,272 / 714,362,880 / 96,151,393. Depformer GEMM weights = 666,906,624; the rest are embedding tables (lookups).

### §219.2 Per-frame cost (`core/moshi.py`, `speech._moshi_kpis`)
`server.py` runs one conversation (`streaming_forever(1)`, a lock per connection) with Mimi and the LM on one device 「来源」. Each 80 ms frame runs serially 「假设」:

    t_frame = t_mimi_enc(b) + TPOT(b, ctx = min(serving.ctx or 3000, 3000)) + t_depformer(b) + t_mimi_dec(b)
    RTF = t_frame / 80 ms ;  real time ⇔ t_frame ≤ 80 ms ;  response latency = (1 + max delay) · 80 ms + t_frame

Kyutai's "160 ms theoretical" is the latency floor. The temporal step uses `evaluate` decode. The other parts use `evaluate._component_time` (an op roofline at the module rate: bf16 depformer, fp32 Mimi).

Depformer re-read pattern:
- Step k (one per Moshi codebook) runs its own weight slices over one row, with attention over the k + 1 positions of this frame.
- Every frame therefore reads all 667 M depformer GEMM weights once (1.33 GB), i.e. 12.5 times per second. Mimi encode + decode reads 0.33 GB fp32 per frame.
- Per-op launch overhead of the 48 serial one-row layer passes is not modelled 「近似」.

`streams_realtime` is the largest batch with t_frame ≤ 80 ms. That is a design question; the released server runs batch 1.

**Finding** (100T + LPDDR5X, full ring buffer):
- t_frame = 0.91 + 78.1 + 7.0 + 0.84 = 86.8 ms, so RTF 1.085 and latency 247 ms.
- Re-read bytes per frame: temporal 13.4 + KV 1.57 + depformer 1.33 + Mimi 0.33 = 16.7 GB, which needs 208 GB/s effective (191 available).
- Real time holds up to a 518-frame KV (the first ~41 s of a conversation). At 100 frames, 5 streams fit, because the weights are shared across the batch.
- Each weight is read exactly once per frame, so every GB pinned in an SLC saves 12.5 GB/s whatever it holds. The depformer + Mimi (1.67 GB) is the natural pin set: it is ≥ the 1.30 GB that §219.3 says unlocks real time.

### §219.3 Cross-model speech insight (`core/speech_insight.py`, `POST /api/insight_speech`)
Method:
- Each catalog speech model is evaluated at batch 1: prompt 64, reply 64 tokens, 10 s audio (in for ASR, out for TTS).
- Every audio-clock stage is reported as x = busy seconds per audio second: ASR encoder / LM; TTS LM; CosyVoice and Kimi LM / speech card; Omni Thinker / speech card; Moshi frame.
- The binding stage is argmax x. If x > 1 and the stage is bound by weight re-reads 「推导」:

      BW_need = BW_eff · x ;   or pin S = (x − 1) · BW_eff / r in an SLC (r = re-reads / audio s = token / frame rate),
      SLC read bandwidth ≥ S · r

- `verify` re-runs the KPIs with the chip's SLC = S rounded up to 64 MiB (policy pin), through the engine's own SLC path.
- Nothing is tuned.

100T + LPDDR5X (BW_eff 191 GB/s):

| Model | Binding stage | x | First audio / token (dominant part) | Unlock: BW_eff | or SLC S (verified x) |
|---|---|---|---|---|---|
| moshiko-7b | frame (temporal 90 %) | 1.085 | 247 ms (temporal step; floor 160 ms) | 207 GB/s | 1.30 GB (0.997) |
| kimi-audio-7b-instruct | LM, 12.5 tok/s | 1.19 | 3.94 s (35 LM steps) | 228 GB/s | 2.92 GB (0.984) |
| step-audio-2-mini | LM, 25 tok/s | 1.85 | 2.18 s (28 LM steps) | 354 GB/s | 6.5 GB (0.998) |
| qwen2.5-omni-7b | speech card (Talker 50/s + DiT) | 1.03 | 1.08 s (48 Talker steps) | 197 GB/s | 0.11 GB (0.995) |
| cosyvoice2 / fun-cosyvoice3 | LM | 0.18 | 0.22 / 0.28 s (LM tokens) | — | — |
| ming-3.0-tts-tiny | LM | 0.41 | 0.13 s | — | — |
| ASR (qwen2-audio, qwen3-asr, fun-asr-nano, whisper ×2, ming-asr) | LM decode | 0.01–0.49 | 9–101 ms | — | — |

Reading the table for chip design:
- On a ~190 GB/s LPDDR5X card, every speech model that misses real time misses it on weight re-reads in a token- or frame-clocked stage. No stage is compute-bound.
- The fix is bytes per audio second. A few GB of SLC (0.1–6.5 GB) or +8 % to +85 % bandwidth.
- Moshi and Omni are within 10 % of real time; Step-Audio-2 needs the most (it emits 25 speech tokens / s through a 7B LM).
- First audio is LM-step bound for the token2wav-style models (chunk size × TPOT). Pinning shortens it in proportion.

### §219.4 FireRedTTS3 (deferred)
HF `FireRedTeam/FireRedTTS3` (2026-08):
- `fireredtts3_base/config.json` lists only the DiT (11 × d1024, mlp 3), the patch encoder (8 × d1024), patch_size 4 and redae_dim 64. `redae/config.json` describes the RedAE codec (18 + 18 layers d896, 480-sample patches → 50 Hz).
- The base header (fp32, 2,120,632,897 params) has a `backbone_llm` of 28 × d2048 with vocab 151,936. That is Qwen3-1.7B-shaped 「推导」 from shapes, but it has no config.
- DiT sampling steps, CFG and the AR loop exist only in the code (github FireRedTeam/FireRedTTS3).
- The configs do not suffice. This is next-round work that requires reading the code.

## §220 Speech round 6: speech bottleneck panel, FireRedTTS3-Base, Moshi per-layer overhead (0.275)

### §220.1 Speech bottleneck panel
UI:
- The Insight tab's "语音瓶颈" panel posts `/api/insight_speech` with the UI scenario (chip, memory, system knobs), `audio_s` and `verify`.
- It renders the §219.3 table, with the per-stage x of every row.

Static build:
- app.js `req()` → `serve.respond` (test_static_routes). Slow lane (pyodide-shim.js) += `/api/insight_slc`, `/api/insight_speech`.
- The unpacked-zip probe calls `insight_speech`.

Row isolation:
- `speech_insight._scn` keeps the scenario's hardware / system fields: chip, memory and efficiencies, link / D2D / net, mapping, overheads.
- It resets the model-specific fields to their defaults: serving → batch 1, prompt 64, reply 64, audio_s; layout; format overrides; workload; PD; pp_split; indexer; attn_mode; hyena_prefill.
- Before this, a draft model on the UI's qwen3-8b leaked into the Moshi row (x 0.61 instead of 1.085).
- Verify uses SLC = the chip's existing SLC + S.

### §220.2 FireRedTTS3-Base (`core/fireredtts3.py`)
Sources 「来源」:
- github FireRedTeam/FireRedTTS3@7a1f3a7: `llm/fireredtts3_base.py` (Qwen3_1_7B_ConfigDict, `generate`, `_flow_one_step`, `_backbone_one_step`), `llm/dit.py`, `llm/patch_encoder.py`, `llm/modules.py`, `redae/redae.py`, `core.py`.
- HF headers: `fireredtts3_base/model.safetensors`, `redae/model.safetensors`, all fp32.

Structure:
- **Backbone:** transformers Qwen3Model, with the config dict from the code: 28 × d2048, GQA 16/8, head 128, SwiGLU 6144, vocab 151,936, tied embeddings (no lm_head is used).
  - It runs under bf16 autocast while the weights are stored fp32. Each decorated call re-casts them 「推导」; modelled as the fp32 read only 「近似」.
- **Step = 4 latents** (RedAE 25 Hz = 960 samples) = 160 ms of audio. In order:
  - stop_head;
  - dit_head over the last history_patches + 1 = 3 backbone outputs;
  - 10 Euler evaluations (t = 1 − cos(π/2 · linspace(0, 1, 11))) × CFG 2 (`inference_cfg > 0` doubles the batch).
    - DiT rows = 8 history + 4 noisy latents, input 64 + 1024 + 512.
    - Per block: adaLN 9d from the timestep; RMSNorm; full attention over 12 positions (16 × 64); conv1d k3 ×2 (Mish); GELU-tanh MLP 3d.
    - Then an adaLN FinalLayer. fp32: no autocast there 「来源」.
  - Patch encoder: [cls + 4] through 8 plain blocks (MLP 4d), then RMSNorm + Linear(1024 → 2048) on all 5 rows. The cls row becomes the next input.
- **RedAE decoder:** after generation, over all latents in one pass. in_proj 64 → 2 × 896 (50 Hz); Qwen3 18 × d896 (q 14 × 128, kv 2 × 128, SwiGLU 3584, sliding 64); ISTFT head 896 → 1922, irfft n_fft 1920, hop 480.
- **Per request:** RedAE encoder on the prompt audio. Input patches of 480 samples; in_proj ×2; Qwen3 18 layers; 4-layer cls downsample over pairs; out 64. Then the patch encoder on the prompt latents and spk_proj ×2.
  - The prompt patches join the backbone prefill: 1 + text + prompt patches.
  - The campplus speaker encoder is not costed (pickled .bin).

Params exact: 1,720,574,976 / 400,057,921 / 943,776,322. The stored RedAE embed_tokens (151,936 × 896 ×2) are unused (`inputs_embeds`).

Formulas:

    first audio (one segment, non-streaming 「来源」) = t_prompt + TTFT + steps · TPOT + t_RedAE(4 · steps latents)
    RTF = (steps · TPOT + t_RedAE) / audio_out_s ;  steps = ⌈audio_out_s / 0.16⌉

**Finding** (100T + LPDDR5X, 10 s out, 5 s prompt):
- TPOT 89.8 ms. Standalone, the head is 63.7 ms (12.2 GB fp32 per step: the DiT's 1.17 GB × 10) and the backbone ≈ 36 ms (6.9 GB fp32).
- RTF 0.57, so real time.
- First audio 5.76 s = 63 steps; prompt encode 8 ms, RedAE decode 5 ms.
- Chip levers:
  - pinning the 1.17 GB DiT saves 9 of its 10 reads per step;
  - running the backbone in bf16 storage halves 6.9 GB;
  - first audio needs a streaming decode, which the release does not do.

### §220.3 Moshi per-layer overhead
- `Scenario.layer_overhead_us` (0.69; a fixed time per transformer layer pass, used by `evaluate` per stage; default 0) is reused for the serial passes per frame (`moshi.layer_passes`): depformer dep_q × layers = 48 one-row passes, and Mimi encoder / decoder transformer 8 + 8. SEANet convs are not counted 「近似」.
- At 10 µs: +0.64 ms on these and +0.32 ms on the temporal step's 32 layers, so the frame grows from 86.8 to 87.8 ms. The bandwidth excess (6.8 ms) dominates unless the per-layer overhead approaches ~100 µs.

## §221 Speech round 7: streams per card vs SLC, FireRedTTS3 DiT pin, Fish Audio Drama 3 (0.276)

### §221.1 Real-time streams per card vs SLC size
`speech_insight.speech_streams` / `POST /api/insight_speech_streams`. For each speech model, at each SLC size S (pin policy; S is added on top of the chip as `slc_mib` = S · 1024), the result comes from rerunning `speech_kpis`. It is not extrapolated:

    streams_lm(S) = max b ≤ 1024 such that fits(b) and max_{stage ≠ speech_card} x_stage(b) ≤ 1     (engine reruns)
    lm_limit      = realtime | memory | cap   (why b + 1 fails)
    N_sc          = streams_realtime of the batch-1 speech card (its own search, core/speech.py)
    speech_cards  = ⌈streams_lm / N_sc⌉ ;  cards = LM cards (layout) + speech_cards
    streams/card  = streams_lm / cards        (0 when N_sc = 0: the speech card alone is not real time)
    pred          = ⌊(1 − (x₁ − d)) / d⌋, d = x₂ − x₁    「推导」 linear read-out, weight-read-bound regime only

Search: gallop (doubling) from max(1, pred), then bisection. The calls are memoized, and the predicate is assumed monotone in b 「假设」 (more rows never make a step faster).

**Finding** (100T + LPDDR5X, 10 s audio, prompt 64 / reply 64; streams per card at SLC 0 / 0.5 / 1 / 2 / 4 / 8 GiB):

| model | streams / card | note |
|---|---|---|
| moshiko-7b | 0 / 0 / 0 / 1 / 2 / 5 | the ~1.3 GB per-frame working set must be pinned; pred exact |
| kimi-audio-7b-instruct | 0 / 0 / 0 / 0 / 3.83 / 3.92 | LM 92 → 333; the detokenizer card carries only 4 streams, so speech cards dominate |
| step-audio-2-mini | 0 ×5 / 28.0 | LM real time only at 8 GiB (112); token2wav 44 per card |
| qwen2.5-omni-7b | 0 / 1.00 / 2.98 / 4.96 / 8.84 / 8.83 | Thinker 623–724; the Talker + DiT speech card is the limit (0 → 9 streams) |
| cosyvoice2-0.5b / fun-cosyvoice3 | 78.8+ / 28.4+ flat | LM ≥ 1024 (cap); flow + HiFT 86 / 30 per card, not SLC-sensitive |
| fireredtts3-base | 102 / 112 / 121 / 121 / 126 / 130 | pred 436–1024 is far off: as b grows the DiT head (10 × CFG 2 × 12 rows × b) turns compute-bound |
| drama-3 | 3 / 4 / 4 / 5 / 8 / 9 | MoE: each extra stream touches new experts, so bytes per step grow with b; pred ≈ exact |
| ming-3.0-tts-tiny | 16 / 32 / 36 / 41 / 56 / 89 | MoE expert union saturates, so pred (10–55) is low |
| ASR (qwen3-asr, fun-asr-nano, whisper ×2, ming-asr, qwen2-audio) | +6 % … +59 % from 0 → 8 GiB | whisper-large-v3-turbo flat at 412 (encoder compute) |

**Reading:**
- For the 7B dialogue models (Moshi, Kimi, Step-Audio) the SLC decides whether one card serves anyone at all.
- For Omni / CosyVoice-type models the speech card, not the LM, sets streams per card.
- The batch-1/2 linear read-out is reliable only for dense weight-read-bound LMs. Use the engine column.

### §221.2 FireRedTTS3 DiT SLC pin (0.269 re-read rule) and bf16 re-cast
- The fp32 DiT (`flow_params` · 4 B = 1.175 GB) is already the memplan re-read class (`RR_PREFIX` "tts.flow", `rr_reads` = 10). It is pinned SRAM-first, then SLC (pin policy), ahead of the once-read backbone (0.269). `_firered_kpis` now reports the plan as `dit_pin`:
  - `sram_pinned_GB`, `slc_GB`;
  - miss = 1 − (pinned + slc) / rr;
  - DRAM GB per step = miss · rr · 10;
  - `slc_full_GiB` = (rr − SRAM-pinned) / 2³⁰.
- 100T + LPDDR5X:
  - SRAM holds 0.065 GB, so the miss is 0.945 and 11.1 GB of DRAM traffic per step comes from the DiT;
  - SLC 0.5 → 5.73 GB; 1 GiB → 0.36 GB; 2 GiB → 0;
  - TPOT 89.8 → 61.7 → 33.7 → 27.3 ms.
  - ΔTPOT / ΔGB is constant (28.1 ms per 5.37 GB ≈ effective DRAM bandwidth), so the step is DRAM-bound until the DiT is resident 「推导」.
  - 1.03 GiB of SLC takes FireRedTTS3 from RTF 0.57 to 0.21.
- **bf16 re-cast** (option `bf16_recast`, `counted: false`):
  - `_backbone_one_step` is decorated `@torch.autocast(cuda, bf16)` 「来源」. The weights are stored fp32, and the autocast cast cache is cleared when the context exits, so each step re-casts every Linear weight.
  - Per non-embedding backbone param: read fp32 4 B (already counted) + write bf16 2 B + read bf16 2 B 「推导」, so +4 B per param.
  - 1.409 B params give +5.64 GB per step, added serially at effective DRAM bandwidth (upper bound 「近似」): +29.5 ms per step, RTF 0.57 → 0.75.
  - The DiT runs outside the autocast region and embeddings are not autocast ops, so they get no re-cast 「来源」.
  - An accelerator that casts on load pays 0, which is why this is not in `tpot_s`.

### §221.3 Fish Audio Drama 3 (`drama-3`)
**Source** 「来源」: HF fishaudio/drama-3 (config.json, headers; README "35B parameters, 3B active, FP8 MoE experts, rest bf16"), and vllm-omni @ da572cd `model_executor/models/fish_omni/{fish_omni_model,codebook_tower,pipeline}.py` + `stage_input_processors/fish_omni.py`.

**Backbone:** text_config = the Qwen3.5-MoE shape:
- 40 layers: 30 Gated-DeltaNet + 10 gated full attention (GQA 16/2, head 256);
- 256 experts, top-8, d 512, plus a gated shared expert of 512;
- FP8 block-128 experts.
- Built by the existing `qwen3_5_moe_text` builder, with vocab = 4096 candidate rows: `main_candidate_ids` = EOS + semantic 1..4095. vllm-omni keeps only these rows.

**Tower:** 13 layers = 4 shared + 9 per codebook (`shared_enrichment_layers` + `num_codebooks` − 1). Each layer:
- eh_proj (layer 0: [e; h] 4096 → 2048; layers ≥ 1: the h half 2048 → 2048, after folding the embedding half into a per-code table at load);
- wqkv 2048 → 9216 (q ‖ gate 2 · 16 · 256 + k, v 2 · 2 · 256), o 4096 → 2048, SwiGLU 6144;
- causal attention over the whole sequence with a paged KV, one row per backbone position.
- Modelled as decoder layers 41–53 on every row 「推导」: same FLOPs, weights and KV, with 26.6 KB per position of extra KV (more than the backbone's 20.5 KB).
- 9 codebook heads 2048 → 4096, one row per sequence per step. The serial sample between tower layers is not a separate latency 「近似」.

**Params:**
- Release 35,899,643,008 − (252,288 − 4,096) · 2048 non-candidate lm_head rows − 12 · 2048² folded eh_proj halves = 35,341,014,144 resident = spec (exact).
- Lookup (embed role): 0.70 B = text embedding 252,288 rows + frame-input codebook embeddings 40,960 rows + tower code tables.

**KPIs** (`_fish_kpis`, streaming):
- frame = 2048 / 44,100 = 46.44 ms;
- prompt = text + ⌈ref_s · 44,100 / 2048⌉ rows (ref 10 s 「假设」 → 216);
- first audio = TTFT + 6 · TPOT (async-chunk ramp 6, 8, 12, 16, 24, 32, then 64; 32 frames left context);
- RTF = TPOT / frame.
- Stage 1, the DAC (codec.pth, a 1.87 GB torch pickle), is **not costed**. The README shows it can cap throughput at ≥ 128 streams on one GPU.

**Finding:**
- 100T + LPDDR5X, b = 1: TPOT 30.7 ms (≈ 5.9 GB per step: bf16 attention / tower / shared 4.8 GB + 8 × 40 fp8 experts 1.0 GB, at effective bandwidth; test ±15 %), RTF 0.66, first audio 378 ms.
- Streams per card: 3 at SLC 0 → 9 at 8 GiB.
- External reference (not fitted, not in ext_validation): the README measures 166.7 frames/s at concurrency 1 on one H200 (6.0 ms per frame) and 1687 frames/s at 64. The engine on H100-like gives 2.5 ms (398 frames/s) at b = 1 and 3537 frames/s at 64, so it is ≈ 2.1–2.4× optimistic. Likely causes are per-step fixed costs (53 layers, 10 serial code samples, scheduler) and the DAC sharing the GPU. Left as a known gap; `layer_overhead_us` exists but is not set from this data.

## §222 Speech round 8: faster streams vs SLC, Drama 3 DAC decoder (0.277)

### §222.1 streams vs SLC: same results, fewer and cheaper reruns
The §221.1 definition is unchanged:

    streams_lm(S) = max b ≤ 1024 such that fits(b) and max_{stage ≠ speech_card} x_stage(b) ≤ 1

Only the way it is evaluated changes:
- **What a rerun computes.** The verdict reads only the LM-card stage loads and `fits`. `speech_kpis(lm_only=True)` returns exactly those for omni / cosyvoice / kimi / moshi / Drama 3, using the same expressions (「推导」: identical numbers, see `test_lm_only_matches_full_kpis`). The speech-card component times (flow / HiFT / DiT / BigVGAN / talker / Mimi) and Moshi's inner streams search are skipped.
- **Which b are rerun.** One memo {b: verdict} per SLC point is shared by the search and the b + 1 limit probe. From S_{i−1}'s answer n, the search tries n ± 1, ± 2, ± 4 … until the verdict flips, then bisects. Because x(b) is monotone (the §221 assumption), the result does not depend on the start point; only the rerun count does (`test_near_search_exact`).
- **Memoised pure functions** of frozen specs:
  - `mapping._sub_chip(chip, engines, act)` (the core-group sub-chip of a split schedule);
  - `evaluate._tts_step_flops(ts)`;
  - `speech._firered_fixed(t, prompt_s, steps, sys, org)`;
  - `speech._dac_window(fs, new, ctx, hist, batch, sys, org)`.

Native timing, 15 models × 6 SLC sizes on the box: 128 s (0.276) → 46 s; 936 → 679 engine reruns. What remains is the generic evaluate path (overlap feed, GEMM cost), which this round leaves alone.

The UI issues one request per model and appends rows as they arrive. The run token plus the per-channel stale discard keep it non-blocking and order-safe. In Pyodide one model takes about 3–12 s in the slow-lane worker.

### §222.2 Drama 3 DAC decoder
**Shapes and dtype.** These come from codec.pth via `scripts/fetch_fish_codec.py` 「来源」. The architecture comes from fish-speech `fish_speech/models/dac/rvq.py` (`DownsampleResidualVectorQuantize.decode`) and `modded_dac.py` (`Decoder`, `DecoderBlock`, `ResidualUnit`, `WindowLimitedTransformer`) 「来源」. codec.pth stores no hyper-parameters. The window of 128 and head_dim of 64 come from `configs/modded_dac_vq.yaml` 「假设」 (head_dim is consistent with freqs_cis [4096, 32, 2]).

The decode path of one frame (2048 samples at 44.1 kHz):

    lookups (1 semantic 4096×8 + 9 residual 1024×8) → out_proj 8→1024 (×10, summed)
    → post_module: 8 × [GQA-free MHA 16×64 over ≤ 128 keys + SwiGLU 3072] at the frame rate
    → upsample: [ConvT k=s=2 (1024→1024) + ConvNeXt(1024, 4×)] × 2   (×4 positions)
    → decoder: conv7 1024→1536; DecoderBlock(s) for s = 8, 8, 4, 2: Snake, ConvT k=2s (C→C/2), 3 × ResidualUnit
      (Snake, conv7 dil 1/3/9, Snake, conv1); Snake, conv7 96→1          (×2048 positions)

- **Rates** 「推导」 from the shapes: ConvT weights are [in, out, k]. The quantizer has k = s and the decoder has k = 2s, giving 2·2 · 8·8·4·2 = 2048 = the codec hop.
- **Cost model.** One fp32 GEMM per weight tensor, with rows = the positions it runs at; a ConvT is counted per input position with n = out·k. Snake costs 5 element ops + 1 sin per input element. Attention uses the last row's min(128, position) keys 「近似」 (an upper bound).
- **Total:** 6.77 GFLOP / frame, of which the decoder is 6.43 and the transformer 0.22. `test_flops_per_frame_by_hand` reproduces this from a separate closed form.
- **Weights:** 184.5 M fp32 = 0.74 GB, read once per window forward. `_component_time` streams them from DRAM; the SLC does not hold them 「近似」.

**Windows** (vllm-omni `stage_input_processors/fish_omni.py` 「来源」): 6, 8, 12, 16, 24, 32, then 64 frames. Each window has min(32, emitted) left-context frames.
- **Stateful** (the CUDA `fish_speech_dac_stateful` path, which keeps convolution history and window K/V) decodes only the new frames. This is the default 「假设」, since an accelerator port would keep that state.
- **Module decoder** re-decodes the left context; that cost is reported as `redecode_s_per_request`. At batch 1 on 100T it is 0.174 s against 0.106 s stateful per 10 s of audio.

**Placement** 「来源」: the recipe's measured and default deploy runs both stages on one GPU (memory 0.78 / 0.1). On a card the two stages are time-shared 「假设」 (their loads add):

    RTF(b) = TPOT(b) / frame + Σ_windows DAC(window, batch b) / audio_s
    first audio = TTFT + 6 · TPOT + DAC(window 0, batch 1)
    fits = engine fits and dram_need(last stage) + 0.74 GB ≤ dram_cap

The b streams' windows are decoded in one forward (`per_step_batch` 「来源」), with the windows taken as aligned 「近似」: rows × b, weights read once.

**100T + LPDDR5X results:**
- DAC load at b = 1: 0.0106 (1.6 % of the LM's 0.660).
- First audio +5.0 ms (383 ms in total).
- Streams per card vs SLC 0 / 0.5 / 1 / 2 / 4 / 8 GiB: 3 / 3 / 4 / 5 / 7 / 8 (0.276 without the DAC: 3 / 4 / 4 / 5 / 8 / 9).
- At large SLC the LM step is cheap, so the fixed DAC share costs one stream.
- The README H200 gap (§221.3) is not explained by the DAC.

## §223 NVIDIA Nemotron-3.5-Super-VL (Nemotron-H hybrid + RADIO), Drama 3 DAC DRAM split (0.278)

### §223.1 Release and formats
- 「来源」 HF `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` (2026-10-12): config.json (`nemotron_h_omni`, `llm_config` = `nemotron_h`), hf_quant_config / `quantization_config` (`modelopt_mixed`), safetensors headers.
- Header totals: LLM 120,668,707,840; MTP 2,942,325,248; vision 842,362,886; 74,986,817,096 bytes.
- Formats: FP8 for Mamba in/out, q/k/v/o and lm_head; NVFP4 (g16) for routed and shared experts; BF16 for embeddings, latent projections, router, conv and norms. KV cache FP8 (`kv_cache_scheme` 8-bit float).
- `_compact_qc` keeps the scheme keys and replaces the per-module lists with counts per algorithm. Otherwise the release JSON would be 16 MB.

### §223.2 Nemotron-H language model
「来源」 transformers `modeling_nemotron_h.py` and the release `modeling_nemotron_h_omni.py`.
- Each of the 88 layers has ONE mixer, chosen by `layers_block_type`: 40 mamba, 8 attention, 40 moe. There is no FFN after Mamba or attention.
- **Mamba-2:**
  - `in_proj` h → 2·d_in + 2·G·N + H = 18560 (d_in = 128·64 = 8192, G = 8, N = 128);
  - depthwise conv kernel 4 over d_in + 2GN channels;
  - SSD scan, chunk 128; gated RMSNorm; `out_proj` d_in → h;
  - decode state = H·P·N fp32 + conv history.
- **SSD prefill** (`mamba_chunk_scan_combined` 「来源」): C·Bᵀ (`ssd_bmm`) is per (chunk, group). With n_groups = 8 the B / C projections are shared by 16 heads each, so `lin_aqk` has count b·⌈G/tp⌉·chunks. The intra-chunk (·X), inter-chunk (C·state) and state-update GEMMs stay per head 「推导」. Decode uses the plain recurrent update (no delta rule).
- **Attention:** GQA 32 q / 2 kv, head 128, no RoPE parameters.
- **MoE (latent):**
  - router sigmoid with correction bias, 512 experts top-22;
  - `fc1_latent_proj` h → 1024, experts relu² non-gated 1024 → I → 1024, `fc2_latent_proj` 1024 → h;
  - shared expert relu² 5376 on h.
- **MTP:** 1 module = [attention, moe] plus its projection and norms, modelled as one combined layer. Params are exact.

### §223.3 C-RADIO vision
「来源」 `modeling_radio.py`, `modeling_nemotron_h_omni.py`, `image_processing_nemotron_h_omni.py`.
- **Tower:** patch embedder Linear(3·16², 1280) with no bias; CPE position grid 128² × 1280 (interpolated, vector work); 10 cls / register tokens; 32 pre-LN ViT blocks (biased qkv / proj, GELU MLP 5120).
- **Projector:** final LayerNorm (present because the LLM has MTP), pixel shuffle 0.5 (2×2 → 5120), mlp1 = RMSNorm(5120) → Linear 5120 → 20480 → ReLU² → Linear 20480 → 4096 (no bias).
- **Parameters:** the tower + projector formula gives 840,396,800. Release 842,362,886 = that + video_embedder (3·2·16²·1280 = 1,966,080) + 6 normalizer values (stored only). Exact.
- **Image size** (`radio_patches`, the processor rule verbatim): the budget is max(min((16384 − 4)·4, 13312), 1024) patches. Patches per side are round(side / 16 + ½), scaled by min(√(budget / patches), 1) and floored; if the result is below 1024 it is scaled up; each side is then rounded to even, up if that stays within the budget, otherwise down.
- **Tokens** = wp·hp / 4 + 2. A 1024² image gives 64 × 64 patches → 1026 tokens. 4000 × 3000 gives 132 × 100 → 3302 tokens.
- 「近似」: the 10 cls / register tokens are left out of the attention length (≤ 1 % at ≥ 1024 patches). Video (tubelet 2, `video_embedder`) is not modelled, so coverage is partial.

### §223.4 Drama 3 DAC: DRAM split and the SLC-pin question
- `dac.dram_weights_GB` and `dac.dram_act_GB` (per request set, all windows): weights = hot + exp + w_extra, activations = act. These are the same `_sum_ops` terms as `_component_time` (test: they sum to `dram_GB`).
- 100T, 10 s of audio: weights 5.9 GB (8 windows × 0.74 GB, independent of b); activations 14.3 GB at b = 1 and 126.3 GB at b = 8.
- **Pinning the DAC weights in SLC is not mechanism-justified** 「推导」:
  - at most 30 % of DAC traffic at b = 1, and 4.5 % at b = 8;
  - the DAC is read once per window (≈ 1 read per 6–64 frames), against the LM weights read every frame, so a GB of LM weights in SLC saves 64× more traffic in steady-state 64-frame windows, and ≈ 27× averaged over a 10 s request (216 frames / 8 windows);
  - the 0.277 loss of one stream at large SLC is the DAC's compute and activation share of a cheap LM step.
- No model change.


## §224 test_core_237 work guard, Index-Echo S2ST, Nemotron VL video (0.279)

### §224.1 test_core_237 phase-law guard
- 「来源」 measurement on the box: the same work takes vllm_v0 14.3 s of CPU when run alone and 136 s inside the -j 8 suite (× 9.5: SMT / frequency / cache contention inflate `process_time`). Neither CPU time nor wall time is a stable guard under load.
- The primary guard is now deterministic: call counts and summed result sizes (`_size`, nested lengths) of the eight phase-law kernels per case, ≤ 1.5 × the table + 10. The table was measured in a fresh process at 0.279.
- The CPU bounds are unchanged alone (45 / 90 s); under the parallel runner they are × 6 (backstop only).

### §224.2 Index-Echo S2ST (IndexTeam/Index-Echo-S2ST-2B / -9B)
- 「来源」 HF release: modeling_dubbing.py, code/tts/pipeline.py, code/stlm/audio_model.py + stlm_api.py, code/bridge/hidden2cv.py, README default inference settings.
- **AuT + connector:** the Qwen3-Omni audio tower (d_model 1280, 32 layers, 20 heads, FFN 5120, output 2048; 647,927,168, the same as Qwen3-ASR `_aut_spec`).
  - 2B connector: w1 / w2 2048² with no bias, plus scalars beta and log_alpha = 8,388,610.
  - 9B connector: proj [4096, 2048] = 8,388,608.
- **ST LM:** Qwen3.5 text. 2B 1,881,825,088 (tied); 9B 8,953,803,264. The config declares MTP but no MTP weights are released. The 2B checkpoint is torch_dist; its shapes come from the `.metadata` (stub unpickler).
- **Per request** 「来源」: greedy decode of transcript + newline + translation (max_new_tokens 1024), then a second full forward over prompt + output for hidden states (no KV reuse; the audio embeddings are reused).
- **Hidden2CVMapper** (fp32): d_h → 896, 3 pre-LN bidirectional layers, 8 heads, d_ff 2048, max_len 512, 5 languages, zero-init output, residual on the CV3 text embedding. 24,586,624 (2B) / 26,425,728 (9B), exact; the GEMM weight sum = params − pos − lang_emb.
- **CosyVoice3** (fp32 LM): prompt = sos + text(INSTRUCT + source transcript) + mapped m + task + prompt speech tokens. Speech tokens are clip(25·out_s, 2m, min(20m, 1500)). token2wav runs non-streaming: flow once (CFM 10 × CFG) and HiFT once.
- 「假设」 TEXT_TOK_PER_S 3.0, ST prompt 36 tokens, CV3 instruct 8 tokens. 「近似」 the ST and CV3 BPE counts of the translation are taken as equal.
- Not costed (coverage partial): speech_tokenizer_v3 and CampPlus (ONNX frontend).
- On 100T + LPDDR5X, 10 s → 10 s:
  - 2B: 3.23 s (ST decode 1.21, CV3 decode 1.83, flow 0.11);
  - 9B: 7.30 s (ST decode 5.15).

### §224.3 Nemotron-3.5-Super-VL video
- 「来源」 image_processing_nemotron_h_omni.py `_compute_target_patches_video`:
  - ph = round(√(1024 / aspect)) and pw = round(√(1024 · aspect));
  - each side is rounded to a multiple of 2: up if ph·pw stays ≤ 1024, else both down.
  - Example: 1280×720 → 24 × 42.
- processing_nemotron_h_omni.py: frames are grouped in tubelets of 2, with the last one padded. Each tubelet is one `<img>` chunk with a frame label.
- modeling `forward_video`: video_embedder Linear(3·2·16², 1280) → RADIO → pixel shuffle → mlp1.
- **EVS** (`compute_retention_mask`, q = video_pruning_rate 0.7, applied after the projector, so the encoder still runs on everything): keep max(h·w, ⌊T·h·w·(1 − q)⌋) tokens.
  - 「近似」 we use this count formula. The HF reference reshapes with a square side int(√tokens), which matches only square tiles.
- 「近似」: the tubelet embed is costed with the single-frame patch embed K (3·16²; < 0.1 % of tower FLOPs), and frame-label text tokens are not counted.
- 32 frames at 1280×720 → 16 tubelets, 4032 → 1209 video tokens + 32 tags = 1241 LLM tokens. TTFT at TP2·EP2 goes from 0.615 s to 0.901 s.


## §225 Qwen3.8-Flash-Next-FP4-Local, Index-Echo speech views, CosyVoice3 ONNX frontend (0.280)

### §225.1 nvidia/Qwen3.8-Flash-Next-FP4-Local
- 「来源」 HF config.json: qwen4_exp, identical to Qwen/Qwen3.8-Flash-Next except `ple_embedding_dtype` nvfp4 and the quantization config. The pre/video processor configs are identical. The quantization config is ModelOpt `quant_method` modelopt, `quant_algo` MIXED_PRECISION, with per-module algorithms:
  - NVFP4 g16 csf (48 routed-expert groups);
  - MXFP8 g32 (438: linear attention, attention, shared experts);
  - NVFP4 g16 (the layer-1 PLE n-gram table);
  - W4A16_NVFP4 (MTP experts, vision MLPs / merger).
- Headers: 375,487 tensors, since experts are stored per expert. NVFP4 weights are u8 packed e2m1. Expert scales are `weight_scale.nvfp4_csf_fixed` (u8) plus `nvfp4_csf_exceptions` (u32); MXFP8 scales are `weight_scale` (u8 e8m0 per 32).
  - Treating all scales as non-parameters gives LLM 176,943,899,520, MTP 2,607,150,848 and vision 448,931,056 — the same as the BF16 release.
  - Bits per weight 「推导」 bytes / params: experts 4.263; attention 8.254 (MXFP8 + BF16 conv / A_log); shared experts 8.25.
- Activations 「来源」 ModelOpt algorithm names: NVFP4 = W4A4 (the release stores an `input_scale` per expert linear), W4A16_NVFP4 = weight-only, FP8 / MXFP8 = W8A8. `algo_by_role` (per-module algorithm sets mapped by `classify`) decides per role. 「近似」 MXFP8 activation bytes are taken as fp8 (the e8m0 / 32 scales are left out).
- Nemotron-3.5-Super-VL (modelopt_mixed, FP8 + NVFP4) gets the same rule: act_attn fp8, act_expert / act_shared_expert nvfp4, act_lm_head fp8. TTFT at 4096 tokens on 100T TP2·EP2:
  - text prompt: 0.615 → 0.516 s;
  - + one 1024² image: 0.792 → 0.669 s;
  - + one 32-frame 720p video: 0.901 → 0.775 s.
- **Roles:** `attn_` / `mlp_hyper_connection` → `other` (they used to fall under attn through the `.attn_` pattern). The Qwen4Exp gated-residual and PLE key / value Linears use role `other`, which resolves to the release's `other` dtype (BF16 in both releases). `ir` honours `Linear.role` for attention-block Linears. The PLE lookup reads rows at `fmt("embed")` bits 「近似」: that figure mixes the BF16 token embedding with the NVFP4 table, 4.64 against the table's 4.5.
- 「近似」 the vision tower is still costed with BF16 weights (stored as MXFP8 / NVFP4 in this release).

### §225.2 Index-Echo in the speech views
- Stage `pipeline`: x = Σ split / output audio seconds (one card, serial, non-streaming). r (re-reads per output second) is that of the dominant decode stage: ST text tokens n_out / out_s, or CV3 speech tokens / out_s 「近似」.
- Concurrent clips 「假设」 (`dub` itself is batch 1): b clips per replica, ST / CV3 LM steps batched over b, mapper / flow / HiFT / speech tokenizer per clip.
- `repin_s` 「推导」: each ST / CV3 evaluation pins its own hot set in SRAM / SLC. On one card the two phases alternate, so per round both pinned sets are re-read once: (pinned_hot + slc_hot)(ST decode) + (same)(CV3 decode), divided by DRAM bandwidth. Staggered clips across phases are not modelled 「假设」.
- On 100T + LPDDR5X, streams for 10 s → 10 s at SLC 0 / 0.5 / 1 / 2 / 4 / 8 GiB:
  - 2B: 38 / 43 / 48 / 53 / 57 / 58;
  - 9B: 12 / 15 / 19 / 22 / 24 / 30.
  - All are real-time limited.

### §225.3 CosyVoice3 frontend (ONNX)
- 「来源」 CosyVoice `cosyvoice/cli/frontend.py`:
  - speech tokenizer session on CUDAExecutionProvider (if CUDA), input whisper log-mel 128 at 100 frames / s, asserted ≤ 30 s;
  - CampPlus session on CPUExecutionProvider, input kaldi fbank 80 bins.
- `fetch_onnx_graph.py` walks the protobuf (onnx.proto field numbers) over Range requests: 79 requests for the 969 MB tokenizer, 25 for CampPlus.
- `frontend_ops` divisor propagation 「推导」:
  - Conv multiplies dt by the time stride (and df by the frequency stride for 2-D);
  - Shape-derived tensors are not activations;
  - ReduceMean with keepdims 0 over time → one position;
  - CAM++ context / segment pooling counts as passthrough 「近似」.
- speech_tokenizer_v3: conv1 128→1280 k3 s2, conv2 1280→1280 k3 s2 (25 Hz), 12 × [LN, q / k / v 1280², rotary, FSMN depthwise k31 on v, full attention 20 heads × 64 (head count from the rotary table width), o 1280², LN, MLP 1280→5120→1280 GELU], FSQ project_in 1280 → 8.
  - 10 s: 2·384·1280·500 + 2·3840·1280·250 + 12·(8·1280²·250 + 4·1280·5120·250 + 4·250²·1280) + 2·1280·8·250 = 124.76 GFLOP, exact against the graph (test).
- CampPlus: 2-D conv head (80 bins; frequency strides 2 ×3 → 10), TDNN k5 s2 → 50 Hz, 3 dense CAM blocks (12 / 24 / 16 layers), statistics pooling → 192-d. 11.26 GFLOP per 10 s, on the host CPU and not on the card.
- 「近似」: biases / norms are read once with no extra vector work beyond 2 per GEMM output.


## §226 Activation-format audit of quantized releases; fused-expert scale suffix (0.281)

### §226.1 Per-role GEMM input formats
- Audit of the 38 quantized releases (`quantization_config` present), by quant method:
  - `fp8` + `activation_scheme` dynamic / static (DeepSeek, Kimi K2, Qwen3 FP8, GLM-5.3, MiMo, Ling, Step-5, Hy-Memory): global act fp8 since 0.251 — unchanged.
  - compressed-tensors W8A8 float (LFM2.5 FP8, Beam FP8): act fp8 since 0.254 — unchanged. compressed-tensors weight-only (Kimi K2.5 / K2.7 int4, Kimi K3 MXFP4, Dory MXFP8: `input_activations` null) — bf16 inputs, correct.
  - ModelOpt NVFP4 (Muse NVFP4, Beam NVFP4, Step-5 NVFP4) W4A4 since 0.254; W4A16_NVFP4 (Dory NVFP4) weight-only; MIXED_PRECISION (Nemotron VL, Qwen3.8 FP4-Local) per role since 0.280 — unchanged.
  - AWQ (Qwen3 AWQ), `mxfp4` (gpt-oss) — weight-only, bf16 inputs (Blackwell MXFP8 override from 0.93 stays a what-if).
  - **Changed:** Quark (two Muse releases) and `bsgemm-moe` (Drama 3), below.
- Quark 「来源」 vLLM `quantization/quark/quark.py` scheme selection and `schemes/quark_w8a8_fp8.py`: input_tensors `is_dynamic` + `qscheme` per_channel → `GroupShape.PER_TOKEN`, `Fp8LinearOp(act_quant_static=False)`. vLLM Quark docs: OCP MX schemes take dynamic MXFP4 inputs; MI355 runs them natively, MI300 / MI325 emulate by dequantising to half precision.
  - PTPC-FP8 release: input fp8_e4m3 dynamic per_channel, weight fp8_e4m3 per_channel → act fp8 on the fp8 roles (attn, expert).
  - MXFP4 release: input fp4 g32 e8m0 dynamic, weight fp4 g32 → act mxfp4 on the mxfp4 role (expert).
  - Only `global_quant_config` is read. Releases with `layer_quant_config` / `layer_type_quant_config`, or with different weight / input families (W4A8 MXFP4 × FP8: vLLM `QuarkW4A8_MXFP4_FP8`), get no activation formats 「假设」 (none in the catalog).
- Drama 3 「来源」 config `{"quant_method": "bsgemm-moe", "weight_block_size": [128, 128], "activation_scheme": "dynamic"}`; vllm-omni `recipes/fishaudio/Drama-3.md`: "block-FP8 routed experts" on vLLM's block-FP8 MoE path, which quantises inputs per token-group 128 to fp8 → act_expert fp8. Shared experts, attention and the codebook tower are bf16 and keep bf16 inputs.
- Excluded modules keep bf16 inputs (per-role entries, not a global act_fmt), as in §225.1.
- 「近似」 MXFP4 activation bytes 4.25 b (e2m1 + e8m0 / 32); fp8 activation scales are not counted.
- Effect 「推导」 (4096-token prompt, TP8·EP8, TPOT unchanged):

  | model | chip | TTFT ≤ 0.280 | TTFT 0.281 |
  |---|---|---|---|
  | muse-spark-open-fp8 | B200 | 76.2 ms | 70.8 ms |
  | muse-spark-open-mxfp4 | B200 | 76.1 ms | 66.4 ms |
  | muse-spark-open-fp8 | 100T TP8 / TP16 | 1.102 / 0.749 s | 1.026 / 0.716 s |
  | muse-spark-open-mxfp4 | 100T | 0.926 / 0.666 s | unchanged (no fp4 MACs → bf16 both ways) |

  Drama 3 on 100T: batch 1 unchanged (decode-bound); 32 clips, first audio 1.647 → 1.630 s.

### §226.2 Fused-expert scale suffix
- Two audits for the 0.280 csf bug class:
  - every release JSON: effective bits per (role, format) against the format's range. Only Muse NVFP4 was off (nvfp4 4.0 b and an "fp32 expert" of 46,080 parameters holding 48.3 GB);
  - the 93 cached header sets: tensors counted as weights whose last name component looks like a scale / zero / index / codebook. Only the DiT `scale_shift_table`s matched, and they are real parameters.
- 「来源」 Muse NVFP4 headers (byte-identical public mirror unsloth/Muse-Spark-Open-816B-A42B-NVFP4, re-read 2026-10-20): fused experts `layers.N.mlp.experts.{gate_up_proj, down_proj}` (u8 e2m1) with `…_weight_scale` (e4m3 per 16), `…_weight_scale_2` (fp32 per expert) and `…_input_scale` (fp32 per expert). The old `_scale$` alternative left `_weight_scale_2` as a weight, and the base `…_weight` of `_weight_scale` matched it by prefix.
- Fix: `SCALE_RE` first alternative `_(weight|input)_scale(_2)?$` (whole suffix stripped → base = the fused expert tensor). Expert nvfp4 bytes = params · 9/16 + 368,640 (60 layers × 2 projections × 384 experts × 2 fp32 scales). The per-role effective bits (4.871 b, incl. the bf16 experts of layers 0 / 61) were already right because role bytes include every format; no cost changes.
- Reverse direction (parameters treated as scales), left as is: `hc_attn_scale` / `hc_ffn_scale` (3 values per layer, DeepSeek-V4 / GLM-5.3) and Dory `lm_head.logit_scale` (131,072 values; 3e-5 of the model).

## §227 Drama 3 on H200 as an external speech diagnostic; hyper-connection scales (0.282)

### §227.1 Drama 3 H200 (SGLang-Omni cookbook)
- 「来源」 https://sgl-project.github.io/sglang-omni/cookbook/fishaudio_drama3.html (last updated 2026-10-20). One H200; streaming; Seed-TTS English prompts, each cloning one pre-encoded reference; temperature 1.0, top_p 0.9, top_k 30, max_new_tokens 500. Tables 1–2 with `gpu_memory_fraction 0.85` and `max_running_requests 256`. Table 1 (frames/s): c 1 → 166.7, 2 → 287.0, 4 → 467.6, 8 → 678.1, 16 → 927.6, 32 → 1244.7, 64 → 1687.3. Table 2: 128 / 256 streams with the vocoder on the same GPU 2177 / 2422, on a second GPU 3692 / 4160. Table 3 (non-streaming, c = 8): latency mean 1.235 s, RTF mean 0.281.
- 「来源」 sglang-omni @ 0029194 `models/fishaudio_drama3`: `config.py` stream_stride 10, followup ramp (10, 12, 32), followup stride 64, stream_overlap_tokens 32; `vocoder.py` window CUDA graphs captured at batch 1 (one stream per decode); `stream_windows.py` window length = stride + min(overlap, emitted).
- Loop 「推导」: frames/s = c·N / (c·ttft₁ + N·tpot(c) + c·V), where V = Σ windows DAC(new + overlap, batch 1) from `speech._dac_window`. TTFA = ttft₁ + 10·tpot(c) + DAC(first window).
  - 「近似」 prefills are serial at batch 1 (SGLang batches up to 16 first frames).
  - 「假设」 vocoder and engine loads add (two processes on one GPU, no MPS by default).
- Workload: N = ⌈4.395 s / 46.44 ms⌉ = 95 「推导」 (Table 3: 1.235 / 0.281); reference 3.9 s (the cookbook's clip) 「假设」; 64 text rows 「假设」. Engine: H200 reference chip, drama-3 as released (bf16 + fp8 experts, act_expert fp8).
- Results (frames/s, predicted / measured):

  | variant | c=1 | 2 | 4 | 8 | 16 | 32 | 64 | gmean |
  |---|---|---|---|---|---|---|---|---|
  | catalog (no kernel cost) | 3.13 | 2.87 | 2.51 | 2.24 | 2.01 | 1.80 | 1.63 | 2.26 |
  | default (trace kernel floor) | 1.24 | 1.29 | 1.33 | 1.40 | 1.43 | 1.40 | 1.33 | 1.34 |

  Table 2 under default: same GPU 1.31 (128) / 1.37 (256); second GPU 0.91 / 0.97.
- Reading:
  1. At low concurrency the frame step is dispatch-bound on a GPU: about 900 kernels (40 backbone + 13 tower layers + 10 code samples) at the trace floor (3.0 µs) = 2.7 ms against 1.75 ms of weight streaming. For a chip, per-op launch / sync latency matters as much as bandwidth for streaming TTS with deep frame steps.
  2. At high concurrency, short utterances make request prefill a third of engine time (c = 64).
  3. The remaining same-GPU gap is the vocoder: measured 1.70× from moving it to a second GPU, modelled 1.17×. Candidates are DAC cost on GPU (fp16 batch-1 convs and Snake vector work at 44.1 kHz) and process time-slicing; not resolved, not fitted.
  4. TTFA: predicted 61 ms vs 89 ms at c = 1; the gap grows with queueing.
- Diagnostic only: nothing here changes a default or any catalog number.

### §227.2 Hyper-connection scales
- 「来源」 DeepSeek-V4 inference `model.py` / GLM-5.3 modeling: `hc_attn_scale`, `hc_ffn_scale` (shape [3]) and `hc_head_scale` ([1]) are `nn.Parameter`s of the mHC mixing (the templates' `hc_p` / `head_p` already include them). `summarize_release` dropped them as scales through `_scale$`; `NOT_SCALE_RE` keeps them. Release LLM params now equal the templates for V4-Flash (+259), V4-Pro (+367), V4.1-Flash (+240) and GLM-5.3-Flash (+270); MTP +7 / +7 / +18 / 0, still exact.
- Dory `lm_head.logit_scale` [131,072] stays excluded: the template has no logit-scale term, so counting it would open a 3e-5 gap.
  - 0.321 (§266.2): counted. It is an nn.Parameter applied to every logit (vLLM PR #60149), and the template now has the term (`ModelSpec.logit_scale`).

## §228 Per-op dispatch on components; speech op-dispatch sensitivity; colocated vocoder (0.283)

### §228.1 Components pay dispatch / kernel cost
- The existing design variable (0.203, `hardware.Chip`): `dispatch` ∈ {"", static, trace, host}, `dispatch_us` (None = mode default). `DISPATCH_US`:
  - static 10 µs per step 「二手」 JAX jit dispatch 9.1–10.1 µs;
  - trace 1.5 µs per op 「二手」 tt-metal #20681;
  - host 20 µs per op 「二手」 vllm-ascend #13021.
  - Default "" (ideal) — unchanged.
- `evaluate._component_time` now builds its `StageTime` with t_sync = Σ per-op dispatch (trace / host) + static per-step cost × executions + `kernel_us` per kernel. t_disp and t_kern are split out the same way `_sum_ops` already counted them for stages 「推导」. Returns `sync_s` and `dispatch_s` only when non-zero (so default summaries and fingerprints are unchanged).
- Applies to speech cards, codecs, mapper / tokenizers and VAE / text encoders. Bit-identical when both knobs are off (all catalog chips). Moved numbers: the ext speech default variant only (kernel_us set) — see CHANGELOG 0.283.

### §228.2 Speech op-dispatch sensitivity (`speech_insight._dispatch_row`)
- t_stage = max(compute, DRAM, SLC, link) + t_sync, so each stage's x is exactly linear in the per-op latency u:
  - x_s(u) = x_s(0) + n_s·u 「推导」;
  - n_s = (x_s(1 µs) − x_s(0)) / 1 µs (engine-measured dispatched ops per audio second; the chip's own dispatch is stripped for x(0)).
- u_rt = min over stages with x_s(0) < 1 of (1 − x_s(0)) / n_s; it is 0 if the binding stage misses real time even when ideal.
- u_eq = x_b(0) / n_b. Above u_eq, dispatch rather than bandwidth / compute dominates the binding stage.
- x at trace / host is evaluated at the sourced per-op defaults. Exactness is tested: the engine run at host matches to 1e-9, and x(u_rt) = 1.
- 100T + LPDDR5X 4×64-8533, batch 1, 10 s audio:

  | model | x(0) | n (ops / audio s) | u_rt µs | u_eq µs | x@trace | x@host |
  |---|---|---|---|---|---|---|
  | drama-3 | 0.671 | 15.3 k | 21.5 | 43.8 | 0.694 | 0.977 |
  | ming-3.0-tts-tiny | 0.406 | 17.8 k | 33.5 | 22.8 | 0.432 | 0.761 |
  | index-echo-s2st-9b | 0.730 | 8.4 k | 32.3 | 87.3 | 0.743 | 0.898 |
  | fireredtts3-base | 0.566 | 8.0 k | 54.3 | 71.0 | 0.578 | 0.726 |
  | cosyvoice2-0.5b | 0.182 | 6.1 k (LM) + 1.7 k (card) | 134.6 | 30.0 | 0.191 | 0.304 |
  | whisper-large-v3 | 0.063 | 2.7 k | 343 | 23.3 | 0.068 | 0.118 |
  | moshiko-7b | 1.085 | 9.3 k | 0 | 116 | 1.099 | 1.272 |

- Reading for chip architects:
  - Streaming TTS decoders dispatch 8–18 k ops per audio second (frame / token step × layers × ops), against ≈ 2–3 k for ASR decoders.
  - With graph-trace dispatch (≈ 1.5 µs) bandwidth still binds. Host-launched per-op dispatch (≈ 20 µs) is enough to put Drama 3 at the edge of real time and to make Ming-TTS-tiny dispatch-dominated.
  - The per-op latency budget for real-time streaming TTS on this memory system is ≈ 20–35 µs/op.

### §228.3 Colocated vocoder (Drama 3 H200), unresolved
- After §228.1 the ext default variant's DAC pays 93 kernels per window × 3.03 µs (+0.27 ms per window). The modelled same → second-GPU speed-up rises from 1.17× / 1.21× to 1.22× / 1.27×; measured 1.70× / 1.72×.
- 「来源」 sglang-omni @ 0029194:
  - `stream_windows.stream_window_lengths` = {10, 20, 32, 64, 96};
  - `vocoder.WindowGraphDecoder` pads each batch-1 window to the next captured length (only the final window pads);
  - `stages.py` decodes `ChannelsLastDecoder(codec, dtype=float16)`.
  - Neither padding nor fp16 (cheaper than our fp32) closes the gap.
- The measured split implies vocoder ≈ 0.70 × (prefill + decode) ≈ 20 ms per request against the modelled 6.4 ms. The remaining candidate is two-process time-slicing on one GPU without MPS, which needs a kernel trace. Not fitted; documented.
- 0.321 (§266.1): the two-process / MPS-off layout is sourced. The switch cost (145 µs × 2 per window) is now in the default; c = 256 lies inside the published band, while c = 128 stays above the full-slice upper bound.
- 0.322 (§267.2): a sourced candidate for the c = 128 residual is 9 copy-only GPU submissions per request (single-frame stream items). Whether such copies cost a compute time-slice switch is undocumented, so it is documented, not adopted.

## §229 Dispatch default decision; per-op dispatch sensitivity beyond speech (0.284)

### §229.1 Default dispatch for NPU presets (kept ideal)
- Design variable (0.203): `Chip.dispatch` ∈ {"", static, trace, host}, `DISPATCH_US`:
  - static 10 µs per step 「二手」 JAX jit dispatch;
  - trace 1.5 µs per op 「二手」 tt-metal #20681;
  - host 20 µs per op 「二手」 vllm-ascend #13021.
- Survey: all catalog LLMs that fit on 100T + LPDDR5X 4×64-8533 (41), decode batch 1, ctx 1024.
  - trace: median TPOT +1.7 %, max +8 % (carbon-500m).
  - host: median +23 %, 29 / 41 above +10 %.
  - u_10 median 8.9 µs (1.9–60).
- Decision 「假设」 kept: catalog presets are runtime-agnostic design points. The lowest-cost sourced runtime moves the median by less than the model's other approximations, and host dispatch is a software choice, so a non-ideal default would be an unsourced assumption about the chip. Old = new = ideal. Sensitivity is reported instead (§229.2).

### §229.2 Step sensitivity (`core/dispatch_insight.py`)
- Dispatch is a serial term of every stage, so a single-stage, one-microbatch step is exactly linear in the per-op latency u:
  - step(u) = step(0) + n·u 「推导」;
  - n = (step(1 µs) − step(0)) / 1 µs (engine-measured dispatched ops per step; the chip's own dispatch is stripped for step(0)).
- u_10 = 0.1·step(0)/n; u_eq = step(0)/n.
- A pipelined step is a max over stages, so the line is a 「近似」 there (`exact` false). The LLM insight table therefore evaluates TPOT at trace / static / host directly.
- Tested: the host run matches the line to 1e-9, and step(u_10) = 1.1·step(0).
- 100T, decode, ctx 1024:

  | model | b | TPOT ms | ops / step | u_10 µs | ×trace | ×host |
  |---|---|---|---|---|---|---|
  | qwen3-0.6b | 1 | 6.5 | 283 | 2.3 | 1.065 | 1.87 |
  | qwen3-0.6b | 32 | 25.7 | 283 | 9.1 | 1.017 | 1.22 |
  | llama-3.2-1b | 1 | 12.8 | 163 | 7.8 | 1.019 | 1.26 |
  | qwen3-8b | 1 | 79.6 | 363 | 21.9 | 1.007 | 1.09 |
  | qwen3-30b-a3b | 1 | 32.0 | 531 | 6.0 | 1.025 | 1.33 |
  | qwen3-30b-a3b | 32 | 294.9 | 531 | 55.5 | 1.003 | 1.04 |
  | gpt-oss-20b | 1 | 19.2 | 267 | 7.2 | 1.021 | 1.28 |

- Reading for chip architects: small dense models and small-active MoE at batch 1 tolerate only 2–8 µs per op before TPOT degrades by 10 %. A host-launched runtime (≈ 20 µs per op) costs them 30–90 %. Graph-trace dispatch (≈ 1.5 µs) costs ≤ 7 %.

## §230 Static-build route classification; quantized vision tower (0.285)

### §230.1 Static (Pyodide) build
- Every `/api` call goes through `app.js req(chan, path, body)` → the shim → a Pyodide worker → `accel_dse.serve.respond`. Routing is therefore by the server's own table; the shim only picks the lane:
  - fast: eval / sensitivity / memory / sweep / pareto / catalog;
  - slow: searches and insights. Requests here are abortable: a running request is terminated together with its worker.
- 0.285: both lane lists are explicit (`fastRoutes` / `slowRoutes`). Tests require UI calls ⊆ fast ∪ slow, the two lists disjoint, every insight route in slow, and both lists ⊆ the server routes. An unknown route runs on the slow lane with a console warning.
- Stale results:
  - `req` aborts the previous request of the same channel and drops replies with an old sequence number;
  - a scenario edit (`invalidate`) tags the compare / layouts / sweep statuses, and (0.285) the speech, streams and dispatch insight statuses;
  - long panels compare the scenario version at start (`staleTag(v0)`);
  - the streams panel sends one scenario snapshot for all rows.
- `scripts/ui_static_smoke.py` checks all of this live on a built dist-static. At 0.285: cold load 4 s, 19 endpoints answered, speech insight 17 rows, streams 11 rows, dispatch panel 4 rows with the superseded request killed.

### §230.2 Quantized vision tower
- 「来源」 nvidia/Qwen3.8-Flash-Next-FP4-Local safetensors headers (local/hf_cache), 27 blocks:
  - `attn.qkv.weight` F8_E4M3 (3456, 1152) + `weight_scale` U8 (3456, 36);
  - `attn.proj` F8_E4M3 (1152, 1152) + U8 (1152, 36);
  - → MXFP8, e8m0 per 32: 8 + 8/32 = 8.25 bits.
  - `mlp.linear_fc1` U8 (4304, 576) + F8_E4M3 scale (4304, 72) + F32 `weight_scale_2`; `linear_fc2` U8 (1152, 2152); `merger.linear_fc{1,2}` alike;
  - → NVFP4, e4m3 per 16: 4.5 bits. `patch_embed` is bf16.
  - hf_quant_config `algo_by_role.vision` = [MXFP8, W4A16_NVFP4].
- Activation rule, as for LLM roles 「来源」 ModelOpt algorithm names:
  - FP8 / MXFP8 = W8A8 → fp8 GEMM inputs (e8m0 scales ≈ fp8 bytes 「近似」);
  - W4A16_NVFP4 = weight-only → bf16 inputs.
- Stored weights = release vision bytes / params × tower params (0.7276 B/param; was 2.0).
- 100T (fp8 at 2× bf16), 1 image at 1024², prompt 2048: encode 64.1 → 57.9 ms; tower weights 0.898 → 0.327 GB; TTFT 648 → 642 ms. The attention GEMMs (≈ 1/3 of tower FLOPs) run fp8. MLP / merger dequantise to bf16 MACs, with 3.6× fewer weight bytes.


## §231 Structural UI tests; expert residency across SRAM / SLC (0.286)

### §231.1 UI wiring checked through parsed structure
- `tests/webstruct.py` (stdlib) tokenizes app.js / pyodide-shim.js (comments and whitespace dropped, string literals decoded, template literals split into static parts and `${…}` expressions, regex literals recognised by the previous token) and parses index.html with `html.parser`.
- A source assert becomes one of:
  - a token-sequence match (`has`), where quote style, spacing, comments, line breaks and trailing commas don't matter but any identifier, operator or literal change does;
  - a parsed set: ids, `<select>` options, string literals, identifiers, function names;
  - a function-scoped check (`functions(js)[name]`), instead of a character window;
  - call-site structure (`req_calls`: chan, path, enclosing function, body tokens).
- Generalized guards (test_core_286):
  - every literal `$('id')` resolves;
  - every status `invalidate()` tags stale sets `dataset.done` on success and in each catch block that writes it;
  - speech `kind`s from `core/speech.py` (AST) ⊆ the renderer's `p.kind === …` branches;
  - every UI endpoint has a STAGE progress label;
  - tests consume UI source only through webstruct (AST lint).
- `snapshot_request(js, path)`: the request body is `{ ...SNAP, … }`, SNAP is bound once, from `body(…)`, before the first loop that issues the request. The headless smoke checks the same thing live: edit after the first streams row → one scenario across all requests and a stale tag.
- `scripts/ui_pin_mutations.py`: 8 format-only edits must pass and 17 behaviour edits must fail, on a scratch copy of the tree. At 0.286: 25/25.

### §231.2 Expert residency in popularity order across SRAM and SLC
- E = expert bytes per rank, p = SRAM-pinned expert bytes, s = SLC-pinned expert bytes (0.192 pin order: hot → experts → KV; unchanged).
- H(f) = share of expert reads served by the most-read byte fraction f (memplan.dip_exp_hit). It uses the per-expert inclusion probabilities from `serving.moe_expert_load`, with T set so that the mean per-step read probability equals the step's read share ρ = exp_read / E (independent reference model, A0 order 「近似」). Uniform or no load: H(f) = f.
- Read shares 「推导」 (the pin rule ranks a byte by the DRAM reads it saves, i.e. by read probability; the SLC takes the next-ranked bytes after the SRAM):

  | tier | share of expert reads |
  |---|---|
  | SRAM | h₁ = H(p/E) (`sram_exp_hit`) |
  | SLC pin / pin_bw | H((p+s)/E) − h₁ |
  | DRAM | 1 − H((p+s)/E) |

  - dip: retained share fe of the off-SRAM expert bytes → SLC hit share of the off-SRAM reads = (H(f₁ + fe(1−f₁)) − h₁)/(1 − h₁), with f₁ = p/E.
- ≤ 0.285, pin / pin_bw: slc_frac_exp = s/(E−p) (0.263 uniform) or H over a load-neutral off-SRAM subset, applied to every expert read in `step_dram_bytes`. That gave SLC hits > SRAM misses once p > 0, e.g. E = 10, p = 5.1, s = 4.9 GiB: DRAM expert bytes −5.1 GiB per step. Before 0.263 the uniform rule s/E was correct, and 0.286 restores it under uniform load.
- 100T, sram_mib 4 / 16 GiB × SLC 1 / 4 / 16 GiB (8 TB/s), 29 MoE LLMs: 87 / 261 points changed, TPOT +0 … +83 %, and 8 crashes (division by zero) now evaluate. No change at 1 GiB SRAM or at any catalog preset (5 presets × 29 × SLC 0.25–16 GiB).
  - Examples, old → new TPOT: lfm2.5-24b-a2b, SRAM 16 / SLC 16 GiB: 3.24 → 5.92 ms; hy-memory-30b-a3b-fp8, SRAM 16 / SLC 4 GiB: 2.82 → 4.34 ms (old values were optimistic).
- Load-aware SRAM: qwen3-30b-a3b on 100T, sram_mib 4096, Zipf-1 load over 128 experts 「假设」: h₁ = 0.336 at byte share ≈ 0.032; TPOT 21.4 → 15.6 ms.

## §232 Memory-hierarchy ledger; draft / chunk / empty-class fixes; Ornith-1.5-122B (0.287)

### §232.1 The ledger invariant
- Per stage step and demand class c ∈ {weights (hot + re-read + experts + extra), KV read, KV write, state, lookup, activations}: SRAM_c + SLC_c + DRAM_c = touched_c, each ≥ 0.
  - SRAM_c is derived from the plan alone: resident share of the class (hot: pinned / hot_w, split ordinary / re-read as 0.269; experts: `sram_exp_hit` or pinned / expert_w; KV: (kv_sram − state_sram) / kv_total; state: state_sram / state_total). SLC_c = `slc_parts[c]`, DRAM_c = `step_dram_bytes`[c].
  - Capacity: pins ≤ SRAM − staging − sram_resv (+ state buffer); SLC pins and `slc_held` ≤ SLC; SRAM + SLC of a class ≤ its stored bytes.
- `StageResult.ledger` = (StageStorage, touched sums incl. `hot_rr`, SLC/DRAM bandwidth ratio, SLC B/s) on every LLM stage; diagnostic, not serialised.
- A class with no planned storage has nothing on chip: miss share 1 (`hot_miss`, `exp_miss`, `kv_state_miss`). ≤ 0.286: 0, so its touched bytes were charged to no tier. Case: image DiT cached-prefix K/V (`prefix_kv`, kept in the activation reserve): qwen-image-2.1-turbo b1, +134 MB DRAM per cached step (b8: 1.07 GB); t_dram 823.5 → 824.2 ms, step compute-bound (3975 ms) and unchanged.
- After the SLC hit subtraction, DRAM bytes are clamped at 0 (full-hit round-off −4e-15 B).

### §232.2 Independent draft model on the head-stage cards
- The draft (0.222, 「假设」 placement on the target's head-stage cards) and the target share SRAM, the state buffer and the SLC. Rank by the 0.192 rule: a draft byte is read k times per step (k passes), a target byte once → the draft takes the on-chip tiers first 「推导」. `memplan.held(draft head plan)` = (shared-SRAM pins, state-buffer bytes, `slc_held`) becomes the target head stage's `plan(sram_resv, buf_resv, slc_resv)`: free SRAM = SRAM − staging − sram_resv; state buffer and SLC reduced likewise.
  - `slc_held`: pin / pin_bw = pinned bytes; lru = the off-SRAM working set if it fits, else 0; dip = fc·(hot + KV + act off SRAM) + fe·experts off SRAM, ≤ SLC.
  - Staging is transient and shared (the programs alternate), sized by the larger. If the target's staging exceeds the draft's and the draft's SRAM pins do not fit beside it, the draft is re-planned with SRAM reduced by the difference and its SLC capped at the share the target was planned around (fuzz: without the cap the re-run over-committed the SLC by exactly the staging difference, 7.5 MB … 181 MB; TPOT effect ≤ 0.001 ms in the hits found).
  - The draft pass therefore depends on the target's staging (1 + k verify tokens, tree rows): 1P qwen3-32b + qwen3-0.6b b4 k 3, topk 4 / 16 tokens: pass 0.854 → 0.863 ms vs the chain; k-independent when the SRAM is not contended.
  - lru: if the target's off-SRAM set does not fit beside the draft, the step's sweep (target + k draft passes) is cyclic and larger than the SLC → the draft's SLC hits go too (re-run with no SLC).
- ≤ 0.286 both were planned on the whole chip (SRAM 4 GiB: draft 4091 MiB + target 4076 MiB resident). Old → new TPOT: 1P qwen3-8b + qwen3-0.6b b1 2.354 → 2.386 ms; 100T default b8 k 3 71.468 → 71.659 ms; 100T SRAM 4 GiB b1 24.81 → 28.24 ms; 100T SLC 4 GiB pin b1 24.67 → 28.10 ms.

### §232.3 Co-located chunk bytes under the decode plan
- A co-located (chunked-prefill) iteration runs the prompt chunk inside the decode batch's step, so the chunk's KV / state / activation bytes see the decode batch's plan. `evaluate.chunk_mem_bytes(sd, sp, f, reread)`: the prefill-alone touched sums (scaled by the chunk fraction f, plus the re-read of the chunk's own new KV) go through `step_dram_bytes(decode plan)`; returns DRAM bytes, SLC bytes, KV + state DRAM bytes, and the ≤ 0.286 KV bytes. `_fused_step` / `_multi_bracket`: t_dram += (chunk DRAM + extra experts) / BW, t_slc += chunk SLC / SLC B/s; the KV seconds of the bracket scale by new / old KV bytes.
- ≤ 0.286 the chunk's bytes came from the prefill-alone plan, which holds the chunk's KV on chip — bytes the decode batch already occupies — and its SLC hits were not timed. Identical when neither plan holds KV / state on chip (every default chip). qwen3-0.6b b64 ctx 4096: SRAM 4 GiB 171.49 → 173.58 ms per iteration (470 MB of chunk KV had been charged 0); SLC 4 GiB pin 169.55 → 170.57 ms.

### §232.4 Two-batch overlap SLC time
- tbo reads the weights twice per step; the second pass was charged to DRAM time only. Now `t_slc` also adds the second pass's SLC bytes / SLC B/s. Example: t_slc 4.29 → 8.59 ms; TPOT unchanged (DRAM-bound).

### §232.5 Audited without a hit
- KV SRAM / SLC shares under pin / pin_bw / lru / dip; dense re-read (rr) pins; omni / speech components (separate cards, components get no SRAM credit; omni_slc_plan); TP / EP (per-rank plan, per-card SLC, `exp_p` from the busiest rank); PD disaggregation (`cache_new_bytes`, free-KV clip, host-L2 hit_device + hit_host + miss = 1). FSDP pins against the full weights and adds a gather write: conservative, unchanged.
- Default results: 142 models × decode / prefill unchanged; fingerprints unchanged.

### §232.6 Ornith-1.5-122B
- ornith-ai/Ornith-1.5-122B (MIT, public 2026-10-20): qwen3_5_moe as released bf16 — 48 layers in a 3 : 1 Gated-DeltaNet : gated-GQA pattern, 256 routed experts top-8 + shared expert, hidden 3072, partial rotary 0.25, untied head, 1 MTP layer; Qwen3.5 ViT (27 × 1152, patch 16, merge 2, out 3072). Params from safetensors headers: LLM 122 111 526 912, MTP 2 523 679 232, vision 451 290 864 (VisionSpec exact). Processor: min / max px 65 536 / 16 777 216 (image), video 4 096 / 25 165 824 — its own preprocessor configs.
- `vision_spec` accepts `qwen3_5_moe_vision` (the transformers-5 model_type of the MoE ViT config; the dense one is `qwen3_5_vision`, 0.191).

## §233 Ornith family; ledger beyond LLM stages; component bytes; insight stale tags (0.288)

### §233.1 Ornith (ornith-ai)
- Qwen3.5 post-trains (MIT). As released, from config.json and the safetensors headers:
  - 1.5: 9B dense, 35B-A3B, 122B-A10B (§232.6) and 397B-A17B in bf16, each with one MTP layer. Official FP8 for 35B-A3B, 122B and 397B: compressed-tensors `float-quantized` with per-channel weights and dynamic per-token activations, targets Linear. The ignore list (lm_head, embed_tokens, mlp.gate, shared_expert_gate, linear_attn, visual, mtp) leaves those bf16, so the FP8 tensors are the routed / shared experts and the full-attention q/k/v/o (`act_fmt` fp8 on those roles, via the 0.281 compressed-tensors rule).
  - 1.0 (9B / 35B / 397B, bf16 and FP8) is the same configs without the MTP head.
- Catalog rule (MERGED = same architecture as a listed entry; the merged id still evaluates from its own release, so dtypes stay as released):
  - 1.5 bf16 9B / 35B-A3B / 397B and 1.0 bf16 merge into qwen3.5-9b / startlux-decision-v1-35b-a3b / qwen3.5-397b-a17b (identical header shapes and text / vision config).
  - 1.0 FP8 merges into the 1.5 FP8 entries.
  - The 1.5 FP8 entries are listed: no Qwen3.5 FP8 release is catalogued.
- Processor: all repos ship Qwen3.5's preprocessor and video-preprocessor configs (65 536–16 777 216 px; video 4 096–25 165 824; patch 16, temporal 2, merge 2).

### §233.2 Ledger on the non-autoregressive trunks
- `_evaluate_full` stages (video / image DiT, protein, genomics, and the prefix-extract, mode-cache and uncond variants) record `StageResult.ledger` like the LLM stages (§232.1). With `workload.dit_fsdp` the gathered copy written to DRAM, `dram["fsdp_gather"]` = act_w·(g−1)/g, is demand of its own on top of the touched bytes.
- Audit: 146 random scenarios (25 models, 7 SRAM / SLC / policy sets, PP / SP / DP, FSDP, placement, tbo / pipe / kernel): every class closes. No hit — the trunk uses the same plan / step_dram_bytes path that 0.287 fixed.

### §233.3 Components: every touched byte is DRAM traffic
- A component (text encoder, VAE encode / decode, vision / audio encoder, speech-card module) runs off the stage plan with no on-chip credit (its weights are taken as larger than SRAM). Its DRAM bytes are now the sum of all touched classes:

  DRAM = hot + experts + w_extra + act + kv_read + kv_write + state + lookup (+ extra_dram)

  `t_dram` = DRAM / BW and `t_dram_kv` = (K/V + state + extra) / BW, timed with the stage roofline (max of compute and DRAM, plus sync).
- ≤ 0.287: DRAM = hot + experts + w_extra + act only. Lookup bytes (codec / token embedding gathers) went to no tier. The speech callers added K/V seconds after the roofline (Omni `_omni_time`: + (kv_read + kv_write)/BW; CosyVoice / Index-Echo / Step-Audio flow: + chunk kv_read / BW; Kimi-Audio DiT: + kv_bytes(cached frames) / BW, now passed as `extra_dram`).
  - Serial addition double-counts time whenever the component is compute-bound or weight-bound, since K/V streams through the same DRAM port as the weights. The stage path never did that.
- Old → new (10 s audio):
  - qwen2.5-omni-7b b4, 100T + SLC 4 GiB pin: Talker step 2.562 → 1.783 ms, RTF 0.4775 → 0.4382, first audio 0.5832 → 0.5451 s; b1: RTF 0.1749 → 0.1651.
  - Default chip: Omni b4 first audio 1.38140 → 1.38081 s; Kimi-Audio b4 RTF 0.96747 → 0.96766 (its DiT lookups now charged); Drama 3 b4 RTF 1.07719 → 1.07724.
  - Drama 3 DAC: conv-history K/V (`kv_read`) and codebook lookups now DRAM traffic; `dac.dram_act_GB` = all non-weight touched bytes (was `act` only): 100T b1 14.3114 → 14.3364 GB, b8 126.2905 → 126.4903 GB (weights 5.8976 GB unchanged). ext speech (H200 stream) frames/s: default c16 1300.1 → 1300.0, c64 2177.1 → 2177.0, T2 c256 3175.7 → 3175.4; catalog T2 c256 3990.5 → 3990.0.
  - LLM / DiT / protein / genomics evaluations: unchanged (no K/V, state or lookup bytes in their pipeline components). Checked on 143 models × decode / prefill × {default, SRAM 4 GiB + SLC 4 GiB pin}: 570 of 572 points identical, boltz-1 pin identical separately; protenix pin not compared (> 700 s per tree).
- Speech-card SRAM / SLC pins (§214, §216): per DiT re-read evaluation SRAM 1 − miss, SLC miss·h, DRAM miss·(1 − h); the first evaluation reads 1 − miss·h from DRAM and miss·h from the SLC; the Talker reads (1 − h_t) from DRAM and h_t from the SLC; h_d·miss·W_dit + h_t·W_talker ≤ SLC. Property-tested; no hit.

### §233.4 Insight panels stale-tagged
- `invalidate()` now also tags `ins-status`, `ins-slc-status`, `ins-size-status` and `ins-path-status`. Each run snapshots `S.ver` at click time and appends `staleTag` to its result, and the success and catch paths both set `dataset.done` (the §231.1 structural guard).

## §234 Nemotron 3.5 family; heterogeneous MoE routing; pipe flow-shop state view (0.289)

### §234.1 Nemotron 3.5 (nvidia)
- Bolt 74B-A9B = `nemotron_h_omni` with `llm_config.model_type` `nemotron_h_puzzle` 「来源」 configuration_nemotron_h_puzzle.py: `block_configs[i]` updates the global config for layer i (`get_nemotron_h_config_for_layer`); blockwise fields (`n_routed_experts`, `moe_intermediate_size`, `num_experts_per_tok`) are set on every MoE block. `_build_nemotron_h` builds each layer from its merged config; per-block MTP configs are refused (no release has them).
  - 40 MoE layers: E ∈ {240, 304, 368, 432, 496}, width ∈ {1536, 2048, 2304, 2560}, top-9, latent 1024, shared expert 5376. LLM params 74 042 720 768 (exact).
- 24GB checkpoint (ModelOpt MIXED_PRECISION, `quantized_layers_by_algo` IQ2_XXS/g256 27 728, Q8_0/g32 113, W4A16_NVFP4/g16 80): ggml blocks stored as U8 `[out, in / QK, block bytes]` 「来源」 ggml-common.h — IQ2_XXS 66 B / 256 (fp16 d + 32 uint16 grid / sign words) = 2.0625 b, Q8_0 34 B / 32 = 8.5 b. Formats `iq2_xxs` (compute bits 2) and `q8_0` (8), weight-only: `gemm_exec` dequantises to the bf16 path (「近似」: the served b12x kernels are not modelled per format; coverage note).
  - Roles as released: experts / shared IQ2_XXS 2.0625 b; Mamba in / out NVFP4 (attn role, bytes-weighted 4.749 b with the Q8_0 GQA projections); latent fc1 / fc2 and lm_head Q8_0 8.5 b; embeddings, router, norms, conv, ViT bf16. No `act_` role: inputs bf16. KV bf16 (no kv_cache_scheme), state fp32 (`mamba_ssm_cache_dtype`).
- Lightning 30B-A3B: plain `nemotron_h` (52 layers, 128 experts top-6, no latent, 1 MTP). NVFP4 = MIXED_PRECISION (W4A16 NVFP4 experts / shared / lm_head, FP8 attention → fp8 inputs via the 0.280 rule, fp8 KV).
- Catalog: Bolt-24GB and Lightning-NVFP4 listed (the as-served releases); Bolt-BF16, Lightning-BF16 and Super-VL-BF16 are listed too: same architecture as those entries, but MERGED requires identical params and dtypes (test_core_model). Defaults (100T, b1, 4096): Bolt-24GB TPOT 25.91 ms / TTFT 0.817 s; Bolt-BF16 96.03 ms / 1.330 s (does not fit); Lightning-NVFP4 12.72 ms / 0.239 s; Lightning-BF16 33.82 ms / 0.455 s; Super-VL-BF16 129.28 ms / 1.827 s (does not fit).

### §234.2 MoE layers of different expert counts
- `evaluate._moe_groups(m, first, last)`: distinct (E, k, groups, width, mats, input width) of the stage's MoE layers → summed per-expert params. `_exp_route` returns the 0.103 tuple when all layers share (E, k, groups), else `("het", ((E, k, groups, w), …), tokens, EP, skew)`.
- `union_routing` then returns {(E, k, groups): (decode routing, decode + chunk routing)}; `extra_expert_bytes` = exp_dram · Σ_g w_g (hit′_g − hit_g) / Σ_g w_g hit_g (exp_dram = Σ_g w_g hit_g × bytes); the fused pipe merge routes each expert op with its own layer (`m.layers[op.layer].ffn`).
- `_moe_skew`: `moe_expert_load` refused when E differs across layers; the skew cap is min over layers of EP · min(k, ⌈E / EP⌉) / k.
- Uniform models: identical (one group = the old tuple).

### §234.3 Pipe flow shop: state as a view
- `overlap._Sim` window = `log[h:]` (append-only list + head index). `state()` = (d_end − c, last − c, (log, h, len(log), c), 0, hw); `_same` compares lengths and scalars first, then entries as (size, compute end − clock) with the 0.288 expressions. `shift` writes the shifted window into a new list, compaction (h > 4096 and > half the log) copies — a stored view never observes a later change, so every comparison sees exactly the 0.288 tuple's values.
- Bit identity: 600 random cases vs the 0.288 simulator (test), 872 evaluation points. Cost: O(1) per state; entry-wise work only when the window lengths match (protenix pin: 0 entries; boltz-1: 242 699).
- ≤ 0.288 a large buffer (SRAM 4 GiB) never evicts, so the window held every tile so far and each repeat's state cost O(tiles): protenix pin > 700 s → 2.4 s.

### §234.4 Kimi-Audio DiT cached-frame K/V
- `kimi_audio.dit_eval_ops`: each block's attention op reads its K / V of the cached frames (`kv_read` = 2 · frames · d · 2 B; Σ blocks = `kv_bytes(frames)`). speech.py no longer passes `extra_dram`. Speech KPIs bit-identical.

### §234.5 Default behaviour, old → new
- Existing models: unchanged (872 / 876 points identical across the default and two pin chips; the other 4 never finished on 0.288); speech KPIs 68 / 68 identical; ext_validate numbers unchanged. New: 5 listed Nemotron 3.5 entries (§234.1).

## §235 IQ2_XXS codebook dequant cost; dequant-bound tag; queue hot paths (0.290)

### §235.1 Dequant element-ops per format
- `dtypes.DEQUANT_OPS` / `dequant_ops(fmt)`: vector element-ops per converted element; default 2 (the 0.40 convention: unpack / convert + scale multiply 「假设」).
- IQ2_XXS 「推导」 from ggml `dequantize_row_iq2_xxs` (ggml-quants.c) and ggml-cuda `dequantize_iq2_xxs` (llama.cpp 839a572a4c): per weight 2 + 1 (sign select `ksigns_iq2xs[…] & kmask_iq2xs[j]`); per 8 weights 4 (grid lookup in `iq2xxs_grid` 256 × u64, sign lookup in `ksigns_iq2xs` 128 B, 2 index ops); per 32 weights 5 (db = d · (0.5 + (aux32[1] >> 28)) · 0.25). Σ = 2 + 1 + 4/8 + 5/32 = 3.65625. The fp16 d per 256 is not charged (as nvfp4 / int4 scales). A lookup = one lane op (tables in local memory 「假设」).
- Q8_0 (`dequantize_row_q8_0`: qs · d) = 2. Block bytes incl. scales (ggml-common.h): IQ2_XXS 66 B / 256, Q8_0 34 B / 32 (§234.1).
- `mapping.gemm_cost`: `convert_x` = Σ converted elements · (dequant_ops − 2), `convert_wx` its weight part; `_op_seconds` vector = convert_elems · 2 + convert_x, weight dequant = convert_w · 2 + convert_wx. `_wide_alt` / `_splitk_alt` / the swapped-orientation path carry both. A (fmt, act) pair in `FormatSupport.mixed` → `conversion` none, 0 ops.
- 「近似」: the served kernels (vLLM b12x, llama.cpp MMQ) take dot products on packed values; the model charges the reference dequant to bf16 then a bf16 GEMM.

### §235.2 Headroom and tag
- `vecsanity.format_headroom(chip, bw, f)` = (dequant lanes · f_clk · 2 / dequant_ops(f)) / (bw / bytes(f)); int4 → `ratios()["dequant_headroom"]`. IQ2_XXS / int4 = (66/256)/0.5 · 2/3.65625 = 0.282.
- `_dequant_warn(stages, sys, warnings, m)`: appends 「f 余量 x×」 for each weight format of m with headroom below int4.
- `api.dequant_bound(s)`: bound VECTOR and conv_w ≥ 0.5 · t_vector; stage dict `dequant_ms`, `dequant_bound`; `api_catalog.dequant_ops`. app.js `boundTag(bound, dq)` → 「VECTOR · 反量化」; assumptions list the ops per format.
- Bolt-24GB decode b8 on 1P: default lanes MAC-bound, TPOT 5.616 ms (unchanged; vector 2.52 → 3.95 ms); 4096 lanes VECTOR + dequant-bound, 10.38 → 16.11 ms.

### §235.3 Queue hot paths (bit-identical)
- `pdqueue._queue_weights` → `_queue_weights_c` (lru_cache 4096, tuple out; wrapper returns a list). Inner loops iterate x ≥ 1 directly; same products and summation order.
- `bulkq.bulk_law` histogram update: first / middle / last split (the dropped terms are +0.0 on non-negative masses — exact). `r_surv`: o > r ⇔ o − r > 0.0 for finite floats.
- Verified: 300 `_queue_weights`, 3 `bulk_law`, 2 `disagg_report` reprs identical to 0.289; 604 model points identical.

### §235.4 Default behaviour, old → new
- All 604 default points identical (Bolt TPOT included). Bolt-24GB vector / dequant ms up 1.83× on the IQ2_XXS share; the warning text and coverage note change.

## §236 Decision readout checkpoints (pplx-decider); Jais 3 / Mellum2.1; per-run test report cache (0.291)

### §236.1 Readout head instead of an lm_head
- `readout.safetensors` 「来源」 (perplexity-ai/pplx-decider-*): one bf16 tensor `weight` [255, hidden]; `fetch_hf_release` stores its header as `readout.weight`, `summarize_release` counts it in the lm_head role and records `extra.decision` = {readout, attention_mode, pooling, temperature, codes, max_pixels, joint} from `decision_config.json` / `joint_config.json`.
- `ModelSpec.readout_rows` (0 = none): `head_params` = readout_rows · hidden; the tied-embedding dedup is skipped for decision releases (the readout is not the embedding).
- `ir._head_ops`: in a decision readout the head GEMM is m = batch · questions, n = labels; outside it (decode / plain prefill) the full readout n = readout_rows is projected (no vocab head exists). `labels > readout_rows` → ValueError.
- `attention_mode: noncausal_full_attention` 「来源」 → full-attention (gqa / mla) cores `causal=False` (the same rule as StartLux v1.1 Instant); linear-attention (Gated-DeltaNet) layers stay causal. Missing attention_mode (v1) = causal.

### §236.2 Joint questions
- `joint_config.json` template `joint-question-inline-readout-v1` 「来源」: several questions inline in one sequence, one readout per question marker. `Serving.questions` (1 … 4096, needs labels): one prefill of the whole prompt, readout m = batch · questions. `decision()`: latency = the prefill + readout; `decisions_per_s` = batch · questions / latency 「推导」. The prompt length is the user's (the questions are part of it).
- v1.2, b1, prompt 1050, 4 labels on 100T: 548.15 ms either way; 1.824 → 14.594 decisions/s at 8 questions.

### §236.3 Jais 3 (k2_horizon) and Mellum2.1
- `k2_horizon` 「来源」 modeling_k2_horizon.py: sigmoid router + per-expert bias (`moe_gate_bias`, E params per MoE layer) × `router_scaling_factor`; `num_shared_experts` × `moe_intermediate_size` shared experts; dense layers = `mlp_only_layers`; q/k norm when `query_key_norm` (2 · head_dim misc params); partial RoPE and group RMSNorm change no params. MoVA (routed V projections + V router + softplus gate) is not modelled → not listed.
- IFM FP8: compressed-tensors float W8A8, 128 × 128 block weights, dynamic per-group-128 activations → act fp8; 375B keeps attention / shared / router / dense bf16.
- `mellum` 「来源」 modular_mellum.py: Qwen3-MoE attention with q_norm / k_norm [head_dim] (misc 2 · hd), `layer_types` 3 sliding (window 1024) : 1 full, 64 experts top-8.

### §236.4 Per-run test report cache
- `tests/runcache.py` `cached(fn)`: key = SHA-256(code version = all bytes under accel_dse/; model state = upper-case data globals of every accel_dse module (exact perf memos excluded), code locations of module functions and class methods, effective ContextVar values; function; canonical argument text). Hits return unpickled copies; storage in-process + a per-run directory set by the runners (removed after the run).
- Measured 0.291: 58 reports computed, 0 hits. The apparent cross-file duplicates of the 0.290 profile were differently pinned (model_flags) or internal `capture_ctx` contexts that cannot cross processes. Overhead ≈ 10 ms per call.

### §236.5 Default behaviour, old → new
- 756 / 756 points identical (126 LLMs × 3 chips × decode / prefill). `decision()` adds `questions`. New: 14 listed entries.

## §237 Mixture-of-value-experts attention (Jais-3-MoVA); runner start order; burst-test client (0.292)

### §237.1 MoVA mechanism 「来源」
- modeling_k2_horizon.py: `K2HorizonDecoderLayer` uses `K2HorizonMoVAAttention` when the layer is sparse (not in `mlp_only_layers`, `num_experts` > 0, (i+1) % `decoder_sparse_step` = 0) and `mova_num_experts` > 0. q / k / o and the optional `gate_proj` are the same as `K2HorizonAttention`. v_proj is replaced by `v_router` (Linear h → E_v, bias = `moe_gate_bias`) and `v_experts` (E_v × Linear h → n_kv·hd, no bias).
- Routing (`calc_router_weights`): sigmoid scores; bias added to the selection scores only; top-k; the selected unbiased scores are renormalised (k > 1) and × `router_scaling_factor`. Combine (`combine_routed_experts`): Σ_e w_e · silu(W_e x), so the value is no longer linear in x.
- Output gate (`attention_gate_func`): attn_out · softplus(gate_proj x, β = ln 2) (or silu), per element before o_proj. Jais-3-MoVA-36B-A4B: E_v = 64, k_v = 4, softplus, 45 MoVA layers of 48; MoE FFN 100 experts top-8 + 1 shared.

### §237.2 Model / IR
- `AttnCore.v_experts`, `v_topk`, `out_gate_fn`. The q Linear carries the gate width (`_gqa(out_gate=True)`), as for Qwen3-Next. `active_params` subtracts (E_v − k_v) · h · n_kv · hd per MoVA layer.
- `_mova_v_ops` 「推导」: r = `moe_routing(E_v, k_v, t, ep=1)` over the rank's t tokens (attention TP: all tokens on every rank; each rank holds every expert's slice of its gqa_local KV heads). One GEMM per touched expert: m_e × h × (kv_loc·hd), count = hit, share = pairs / (hit · m_e). Only the touched experts' weights stream. Vector: pairs · n · 6 + t · E_v · 6, SFU pairs · n · 2 + t · E_v.
- `attn_out_gate` (softplus): t · n_q,loc · v_dim · (5 vector + 2 SFU) 「假设」. Not charged for the Qwen3-Next sigmoid gate (unchanged convention).
- Hit counts (uniform 「假设」): b1 4, b8 26, b64 64 of 64; prefill 4096 tokens: all 64, 256 rows each.

### §237.3 Dtypes and defaults
- inception42 safetensors fp32 (as released; config says bf16). IFM FP8: block-128 fp8 on routed FFN experts only.
- 100T default b1, 4096: fp32 TPOT 115.16 ms / TTFT 0.973 s (no fit); FP8 48.53 ms / 0.471 s.

### §237.4 Test infrastructure
- `tests/run_tests.py order_longest_first`: LPT start order from scripts/test_durations.json; SLOW_FIRST is the fallback.
- test_core_088 burst client: progress-based wait (fails after 60 s with no server progress, or on a reset). The 0.291 runcache was removed (0 hits).

### §237.5 Default behaviour, old → new
- 840 / 840 points identical (140 LLMs × 3 chips × decode / prefill). New: the two Jais-3-MoVA entries.

## §238 Value-expert routing and residency; per-layer expert loads (0.293)

### §238.1 What the release says about value routing 「来源」
- config.json: `mova_num_experts` 64, `mova_num_experts_per_tok` 4, `moe_gate_bias` true, `router_scaling_factor` 2.5, `router_aux_loss_coef` 0.001, `output_router_logits` false. modeling_k2_horizon.py: the aux loss (`load_balancing_loss_func`) takes `outputs.router_logits`, recorded from `K2HorizonSparseMoeBlock` only (the FFN router). The `v_router` bias is added to the selection scores only (aux-loss-free balancing); no loss term reaches it.
- Model card and blog: benchmarks, training stages, serving recipes; no expert-load figure. The W&B training-log page is a JavaScript application with no readable data.
- So uniform value routing stays the default 「假设」. The selection bias is a mechanism that pushes loads toward balance, which is consistent with that default but is not a measurement.

### §238.2 Measured value-expert load
- `serving.mova_expert_load` (E_v values) → p_e = min(1, k_v·load_e/Σload). Touched experts per step = ⌈Σ_e 1 − (1 − p_e)^t⌉, clipped to [k_v, E_v] (`ir.mova_hits`); m_e = ⌈t·k_v/hit⌉; uniform load = `moe_routing` exactly. Attention TP: t = every token of the rank (no EP for the value experts).

### §238.3 Routed hot class (memplan)
- The 0.192 pin rule ranks a byte by the DRAM bytes it saves per step per byte of SRAM, which is its read share per step. Value-expert bytes are read when their expert is routed: per step ρ_rt = touched / stored value-expert bytes (b1: 4/64, b8: 26/64, prefill: 1). ≤ 0.292 they were in `hot` (ρ = 1), so on a large SRAM they took the place of every-step bytes, and the uniform hot miss share spread the saving over them.
- 0.293 order 「推导」: re-read hot (0.269, if > 2 reads), state, the rest of hot (every step), then {value experts, routed FFN experts} by ρ (equal ρ: value experts first), then KV. Value-expert read share served by the SRAM: H_rt(p_rt / rt_w), with H_rt = `dip_exp_hit(v_p, ρ_rt, f)` for a measured load and f for a uniform one. Miss shares: `hot_miss` (the rest of hot, excluding rr and rt) and `rt_miss`.
- SLC `pin`: the same order over the off-SRAM bytes; read share served H_rt((p + s)/rt_w) − H_rt(p/rt_w). `dip`: the value experts join the expert pool of `dip_split` (bytes r_rt, read e_rt per step); the retained share fe applies to both populations alike 「近似」, and within each population the A0 curve applies. `lru`: unchanged.
- Ledger property (random plans, every policy, measured loads): SRAM + SLC + DRAM = demand per class, no negative tier (test_core_293). A routed class that is stored but untouched in a step still splits the SLC shares.

### §238.4 Per-layer expert loads
- `serving.moe_expert_load_layers` / `mova_expert_load_layers`: one vector per MoE / MoVA layer (layer order; each with its layer's E).
- Per layer: the one-vector rule (busiest-rank skew `skew_from_load`, `SkewSpec(skew, hot_rank_probs)`, rank_max) → `Phase.skew_l[li]`; value experts → `Phase.v_pl[li]`. `_route_groups` splits identical layers that route differently.
- Residency (memplan A0): the items are every expert of the stage's layers with the busiest-load rank's probabilities. Byte-weighted when sizes differ (het MoE, `_dip_exp_hit_w`: the means and the resident fraction are over bytes; most-read bytes first).
- Phase-wide skew (MTP layers, chunk union routing, summary) = the layer mean 「近似」.

### §238.5 Default behaviour, old → new
- The 840 non-MoVA points are identical. MoVA decode TPOT changes by −0.03 to −0.5 % at the default SRAM sizes (fewer misses on the every-step bytes), and prefill TTFT moves by ≤ 0.04 % (pipe replay). With large SRAM / SLC the effect is much larger (CHANGELOG 0.293 table: 4 GiB SRAM b1 −37 %). ext_validate unchanged.

## §239 Drama 3 closed-loop TTFA; LFM2.5-24B-A2B DSpark draft (0.294)

### §239.1 Why TTFA
- 0.282 TTFA = ttft₁ + stride·tpot(c) + DAC(window 0), with no queueing. Predicted / measured (Table 1 p50) fell from 0.69 at c = 1 to 0.38 at c = 64, a load-dependent gap. A per-op cost error would give a roughly constant ratio, so the growing gap pointed at the serving loop. Time to first audio at concurrency is the speech SLO metric, so leaving the gap would misstate SLO-bound capacity.

### §239.2 Serving loop 「来源」
- SGLang-Omni Drama 3 @ 0029194: `prefill_decode_interval` = DECODE_STEPS_BETWEEN_PREFILLS = 4, no chunked or mixed prefill, overlap schedule off.
- sglang scheduler: a prefill batch arms a 4-round counter; while it is armed, `get_next_batch_to_run` skips prefill and decodes. Once it expires, waiting requests are prefilled first, all in one batch (first-frame graphs up to 16 requests).
- The vocoder stage decodes stream windows one at a time (WindowGraphDecoder captured at batch 1; `max_batch_size` 8 applies to non-streaming payloads only).

### §239.3 Model 「推导」
- Closed loop: every stream restarts when it ends, so other streams arrive at a = (c − 1)/N per decode round.
- Mean field: a decode round lasts T_round = tpot(c) + a·(ttft₁ + v), where v is the vocoder GPU seconds per request when it shares the GPU (time-sliced, loads add 「假设」) and 0 on a second GPU. The `frames/s` cycle uses the same shares.
- Prefill gate: arrivals come at round boundaries. A gate-open period ends with exactly one arrival, which is prefilled at once. So a fraction p_win = 4a/(4a + 1) of arrivals land in a deferred window, uniformly over its 4 rounds, and wait (4 − 1)/2 = 1.5 rounds on average: W_gate = p_win·1.5·T_round.
- Co-batching: the batch holds k = 1 + p_win·4a requests (≤ 16 for c ≤ 256: 4a ≤ 10.6). Its cost is k·ttft₁ (serial, as in the cycle 「近似」), and the first frames come at the end of the batch.
- Vocoder queue: windows reach the vocoder at λ = c·n_windows / cycle; s are the DAC window seconds (batch 1). M/G/1: W_voc = λE[s²]/(2(1 − ρ)). This is < 1 ms here, because a window is ≈ 0.65 ms.
- TTFA = W_gate + k·ttft₁ + stride·T_round + W_voc + DAC(window 0). At c = 1 it is the 0.282 form. With the vocoder on a second GPU, the window still decodes there and stays in TTFA.

### §239.4 Diagnostic (ext default variant, never fitted)
- Predicted / measured TTFA, c1 … c64: 0.69 0.70 0.73 0.67 0.70 0.75 0.81 (before: 0.69 → 0.38). The level is consistent with decode being slower than modelled (frames/s 1.23–1.40× high) and a c = 1 fixed ≈ 28 ms (host / IPC hops).
- Table 2: c128 same GPU 0.89, second GPU 1.28. c256 0.32 / 0.41: the benchmark's start burst dominates there (all c requests at t = 0; 4.25 requests per slot over the 1,088), and it is not modelled.

### §239.5 LFM2.5-24B-A2B-DSpark
- `lfm2.5-24b-a2b-fp8-dspark` = LiquidAI/LFM2.5-24B-A2B-FP8 + LiquidAI/LFM2.5-24B-A2B-DSpark, from the README serve command (block 9, 10 draft tokens).
- The draft is the SGLang native DSpark layout (`_attach_ready_dspark`, no reader, no own lm_head): 5 GQA layers, fc over target layers 2 / 10 / 18 / 26 / 34, Markov rank 256, confidence head. 327 707 521 bf16 params.
- Published τ 5.16 (k = 9, c = 1, temperature 0, B300) → i.i.d. acceptance 0.840.
- Modelled b1 speed-up at that acceptance: 1.11–1.17× (1P / H100-like / 100T), against 2.38× published. On a bandwidth roofline the 10 verified tokens touch ≈ 7× the experts of one token. The measured baseline is far from bandwidth-bound.

### §239.6 Default behaviour, old → new
- See CHANGELOG 0.294.0.

## §240 Speculative decoding on a small-active MoE (LFM2.5 DSpark, B300); Drama 3 start burst (0.295)

### §240.1 Data 「来源」
- LiquidAI/LFM2.5-24B-A2B-DSpark `benchmarks/b300-fp8.csv` (72 rows: 8 datasets × c ∈ {1, 2, …, 256}) and `protocol.json`.
  - Each row gives target-only and DSpark output tok/s and the acceptance length τ (output tokens per verification, including the bonus token).
  - Setup: FP8 target, bf16 KV, temperature 0, no radix cache, no torch.compile; complete dataset passes with ≥ 3.5 request waves.

### §240.2 Prediction 「推导」
- Closed loop: tok/s = c·OUT / (c·ttft₁(256) + OUT/τ·step(c)). step = evaluate(decode, batch c, ctx = 256 + OUT/2), with spec_k 9 and spec_accept = spec_accept_from_tau(τ_row, 9) for DSpark.
- Chip: the extval B200 reference stands in for the B300 「近似」 (same dense FP8, NVLink 5 and 8 TB/s HBM3e per the HGX table; B300 has 160 vs 148 SMs and more memory, neither binding here).
- Variants: `catalog` = bandwidth roofline; `default` = per-SM GPU path with the trace-sourced per-kernel floor.

### §240.3 Mechanisms
- (a) Expert union: uniform routing gives E·(1 − (1 − k/E)^(c·n)) touched experts (ir.moe_routing). At c = 1, n = 10: 31 vs 4.
- (b) Locality bound: the block routes like one token per sequence. Step − (union expert bytes − target-alone expert bytes) / achieved bandwidth 「近似」.
- (c) Per-kernel floor: a fixed cost per kernel, independent of the verified tokens. It is 74 % of the target step at c = 1 and 24 % at c = 256.
- Reading: the roofline reproduces the catalog-chip 1.1× at c = 1, and the per-kernel floor gives 2.59× (measured 2.38×). The locality bound gives 4.15×, so strong locality is inconsistent with the measurement. The gap is the hardware baseline; the union is real.

### §240.4 Drama 3 synchronized start
- Batches of ⌊max_prefill_tokens / prompt rows⌋ = 110 requests (SGLang max_prefill_tokens 16384, chunked prefill off), with 4 gate rounds between batches. Batched prefill P(n); first audio after `stride` rounds plus the first window, and the batch's first windows queue one at a time (same GPU: shared timeline; second GPU: its own).
- c = 256 wave-1 mean 777 / 727 ms is below the steady state. Measured p50 − 10·c / (measured frames/s) = 2.35 s / 1.53 s, a pre-run wait that the burst cannot produce. Deferred: its source (vocoder load per the cookbook, for the same GPU; an admission limit is suspected for the second GPU) and batched prefill in the steady-state cycle.
  - 0.322 (§267.1): the admission-limit suspicion is excluded from sources: decode-graph max_bs 256, a state pool equal to max_running_requests, and prefill graphs that fall back eagerly rather than capping. The remaining sourced mechanism is serial host work (overlap disabled); its cost is unpublished.

### §240.5 Default behaviour, old → new
- Cost model unchanged. See CHANGELOG 0.295.0.

## §241 Drama 3 batched prefill in the steady state; the LFM2.5 DSpark mid-c gap by token count (0.296)

### §241.1 Batched prefill 「来源」/「推导」
- SGLang prefills all waiting requests in one forward (PrefillAdder). No request cap by default (`--prefill-max-requests` unset); `max_prefill_tokens` 16384 → B = ⌊16384 / 148⌋ = 110. The 16-request first-frame CUDA-graph cap picks graph vs eager launch only.
- The batch a request is in: n = 1 + Poisson(k − 1) 「近似」, clipped at B. Here k = 1 + p_win·I·a (§239), I = 4, a = (c − 1)/N.
- p̄ = E[P(n)/n] is the GPU prefill seconds per request; P̂ = E[P(n)] is what a request waits for. P(n) = evaluate(prefill, batch n).
- Cycle = c·p̄ + N·tpot(c) + c·vocoder. T_round = tpot + a·(p̄ + vocoder). TTFA = W_gate + P̂ + stride·T_round + W_voc + DAC(first window). With n ≡ 1 (c = 1) or `prefill="serial"` (P̂ = k·ttft₁, p̄ = ttft₁) this is §239.

### §241.2 Result and reading
- Default variant, predicted / measured, serial → batched: frames/s c64 1.29 → 1.55, c128 1.26 → 1.77 (second GPU 0.91 → 1.40), c256 1.31 → 2.13 (second GPU 0.97 → 1.88). TTFA c8 … c64 0.67 – 0.81 → 0.64 – 0.67.
- The serial pricing hid an engine step that, at high batch, is slower than modelled. With the prefill priced as run, the TTFA ratio is flat over c = 1 … 64 (0.64 – 0.73).

### §241.3 LFM2.5 DSpark: steps as a function of tokens 「推导」
- Measured-implied step = (c·OUT / tok_s − c·ttft₁) / OUT · τ (τ = 1 for the target alone).
- At equal token count (c·10 for a verify, c for the target alone), the DSpark step costs 0.91 – 1.15× the target-alone step, measured, and 1.01 – 1.13× in the model. So the speculative structure is priced right.
- The model's target-alone marginal cost falls to 4.7 µs/token once the expert union saturates. The measured cost stays at 78 → 40 → 24 µs/token over 32 → 256 sequences. The c = 8 … 64 speed-up over-prediction follows from this flatter curve between c and 10c tokens. It is the same signature as §241.2.
- Ruled out, with sources:
  - The overlap scheduler is on for DSPARK on CUDA.
  - The radix cache is off, so the mamba `no_buffer` branch that disables overlap is not taken.
  - The README launch command sets no graph or chunk flags.
- Open: the per-token kernel cost at 64 – 2560 tokens. The candidate is FP8 fused-MoE efficiency at small per-expert row counts. It needs a trace and is not fitted.

### §241.4 Default behaviour, old → new
- The cost model is unchanged; only the diagnostics changed. See CHANGELOG 0.296.0.

## §242 Grouped-GEMM efficiency at small rows per expert vs public kernel timings; NPU M granularity (0.297)

### §242.1 Data 「来源」
- DeepGEMM masked grouped FP8 GEMM on H800 (DeepSeek_Simulator @ 7c69101, `results/H800_group_gemm.csv`). Kineto kernel time; 122 distinct (groups, m, N, K) for m = 4 … 2048 and the DeepSeek-V3 expert shapes.
- TensorRT-LLM tech blog 24: B200 NVFP4 DeepSeek-V3-style MoE module at TP8, TRTLLM-Gen grouped GEMM, 1 … 384 tokens.

### §242.2 Model 「推导」
- Grouped time = max(array time of the `groups` GEMMs (gemm_cost on the extval default chip), operand bytes / achieved DRAM bandwidth), optionally + kernel_us per kernel.
- Module: moe_routing (uniform) → hit × m_e; FC1 + FC2 in NVFP4 (4.5 bits); + bf16 hidden in / out; 6 kernels 「假设」.

### §242.3 Result
- H800, 4 – 128 rows: model / measured 1.03 (1.13 with the floor). 256 – 2048 rows: 0.73 (the masked kernel runs below the FP8 ceiling).
  - 0.320 (§265.2): the 256 – 2048-row gap was mostly a pricing scope error: the rows ran at the generic Hopper FP8 cap, not DeepGEMM's per-shape ceilings. 0.73 → 0.86.
- B200 module: 0.84 – 1.00 with the floor. The post-saturation slope is 0.115 µs/token measured vs 0.005 modelled. As an LFM2.5 upper bound 「近似」 that is ≤ 3.95 µs/token per step of the ≈ 19 missing; for Drama 3 it is 2.6.
- Conclusion: grouped-GEMM small-M efficiency is priced. The shared signature (§241) lies mostly elsewhere: per-token non-GEMM kernels or host work, with no public data.

### §242.4 NPU designs
- Small-M inefficiency on an NPU is the array's M granularity: rows padded to the array dimension (gemm_cost). On 1P / 100T / H100-like that gives 6 – 14 % utilisation at 1 row and > 95 % from 16 – 32 rows.
- It binds only when array time exceeds DRAM time. MoE decode is weight-streaming and stays DRAM-bound to ≥ 128 rows per expert.
- The design variable already exists (array rows × cols); no new parameter.

### §242.5 Default behaviour, old → new
- The cost model is unchanged. See CHANGELOG 0.297.0.

## §243 Energy: MAC = multiply + FP32 accumulate; local SRAM access energy vs slice size (0.298)

### §243.1 MAC 「来源/推导」
- Matrix MACs take bf16 inputs and accumulate in FP32 (Google Cloud TPU docs "TPU architecture"; Cloud blog "BFloat16: the secret to high performance on Cloud TPUs").
- E_MAC = E(16b FP mult) + E(32b FP add) = 1.1 + 0.9 = 2.0 pJ at 45 nm (Horowitz ISSCC 2014). 7 nm: / 2.6 (TPUv4i logic scaling) = 0.769 pJ.
- ≤ 0.297: multiply only (1.1, 0.423); kept in `energy.PJ_MAC_MULT_ONLY`.
- Low-precision MACs still count as bf16-equivalent slots (fp8 at 2× rate = ½), so the accumulate share scales the same way 「近似」.

### §243.2 Local SRAM wire 「推导/假设」
- e_SRAM = pJ_bit_sram + w · ½√(A_slice), A_slice = (sram_mib / engines) · ρ, w = pJ_bit_mm_onchip (Keckler 2011), ρ = 3 nm HD macro density (Aoyagi 2025; array only → lower-bound distance).
- Assumption: every engine streams its operand partition from its own slice; cross-engine (NoC) traffic is not in this term.
- Opt-in through w (no preset sets it), as for the SLC (§212).

### §243.3 Result (tpuv4i_7nm, qwen3-8b, prompt 4096)
- MAC: prefill J/prompt-token +56–61 % (1P / 100T / H100-like); decode b8 +2–3 %, b128 +6–8 %. Prefill MAC share 68–74 % → 80–84 %.
- Under a power cap at the old prefill power: 0.85× clock / throughput (rx5700xt_n7 V–f).
- SRAM wire at w = 0.078: slice 8 / 4 / 3.1 MiB → 0.69 / 0.49 / 0.43 mm. On 1P, decode b8 J/token gain from 64 → 512 MiB SRAM shrinks from 0.00267 to 0.00168 J.

### §243.4 Not modelled
- NoC / inter-PE operand movement (Eyeriss normalized: NoC 2× an ALU op, global buffer 6×). On a systolic array this is ≈ a constant per-MAC factor, so it does not separate designs; not added.

### §243.5 Default behaviour, old → new
- See CHANGELOG 0.298.0.

## §244 NoC / on-chip data-movement energy from the dataflow (0.299)

### §244.1 Traffic 「推导」
- Ganged GEMM over E engines:
  - Weight columns stay in each engine's slice (0.298 local wire).
  - The streamed A operand (the feed term's A bytes: ⌈N/(E·cols)⌉·M·K·ab, or M·K·ab for GEMV) is multicast to the a = min(E, ⌈N/cols⌉) engines holding output columns.
  - Outputs stay local and become the next op's A.
- Core-group plans (auto / split):
  - The in-group multicast comes from the sub-chip cost.
  - A row split adds count·K·N·wb delivered to s groups; a column split adds count·M·K·ab.
  - Spread instances keep their operands local 「假设」.
- Split-K over s groups: s × the group's traffic + the m·n·count·4 B partial tiles reduced over a tree of s blocks of E/s engines.
- Delivery of 1 byte to d blocks of u engines:

  | Topology | Router traversals | Link (pitches) |
  |---|---|---|
  | mesh | (d−1)·√u | = traversals |
  | ring | (d−1)·u | = traversals |
  | crossbar | d−1 | d·½√(d·u) |

  Mesh tree: Park et al. DAC 2012 Table 1 (broadcast = k² crossbar + (k²−1) link traversals).

### §244.2 Energy 「来源/推导」
- E = 8·B_hop·pJ_bit_hop_noc + 8·B_pitch·pJ_bit_mm_onchip·pitch_mm.
- pJ_bit_hop_noc reference:
  - (427.3 − 76.7) mW / 653 Gb/s = 0.537 pJ/bit per router + 1 mm link (Park DAC 2012, 45 nm).
  - Router = 0.537 − 0.121 (Keckler 1 mm full-swing wire) = 0.416, a lower bound since Park's link is low-swing.
  - 7 nm: ÷ 2.6 = 0.160.
- pitch = √(sram_mib/E · ρ_SRAM + MACs/E/1024 · ρ_MAC):
  - ρ_SRAM = 3 nm HD macro (array only).
  - ρ_MAC = TPUv4i MXU 0.6875 mm²/kMAC (upper bound).
  - Uncore is omitted, so the pitch is a lower bound.
- Opt-in; the link term needs the wire coefficient too.

### §244.3 Result (insight, tpuv4i_7nm + reference NoC, qwen3-8b)
- 1P prefill NoC + link share by engine count (mesh), with J/token:

  | E | 2 | 8 | 32 | 128 | 512 | 2048 |
  |---|---|---|---|---|---|---|
  | NoC + link share | 2.0 % | 3.9 % | 5.3 % | 9.5 % | 23 % | 40 % |
  | J/token | 0.00723 | 0.00726 | 0.00736 | 0.00784 | 0.00950 | 0.01269 |

  - Throughput plateaus from E = 32.
- Decode NoC ≤ 2 %.
- Crossbar costs +6 % prefill J at E32 (long wires). Ring router energy is 1.7–7× mesh at E ≥ 512.
- Central SRAM: +24 % decode / prefill J on 1P E32, growing with E.

### §244.4 Not modelled
- DRAM → slice placement traffic (DMA lands in the owning slice 「假设」).
- GPU (cta) references: SM ↔ L2 crossbar traffic. The insight warns on them.
- NoC contention / latency: the time model is unchanged.

### §244.5 Default behaviour, old → new
- See CHANGELOG 0.299.0 (no default change; opt-in).

## §245 Energy presets include on-chip movement; energy-aware design search (0.300)

### §245.1 Preset default 「来源/推导」
- Both energy presets now fill the on-chip wire and the NoC router:

  | Preset | pJ_bit_mm_onchip | pJ_bit_hop_noc |
  |---|---|---|
  | horowitz_45nm | 0.121 (Keckler 2011, 40 nm: 310 pJ / 256 bit / 10 mm) 「来源」 | 0.416 (§244.2, 45 nm) 「推导」 |
  | tpuv4i_7nm | 0.078 (Keckler 2011, 10 nm projection as a 7 nm proxy, low side) 「来源/假设」 | 0.160 (0.416 ÷ 2.6) 「推导」 |

- Why (mechanism): ≤ 0.299 a preset J/token left out every on-chip-movement cost (SRAM / SLC distance §243, NoC traffic and its links §244). Those are exactly the costs that change with the design variables in question (engine count, SRAM size, topology), so omitting them biased comparisons toward many small engines and a big flat SRAM.
- Precedence is unchanged: a user value (including 0) wins. A user √C SLC model (`slc_access_ref_mib`) stops the preset from filling `pJ_bit_mm_onchip` (the two are exclusive), in the API and in the UI preset fill.
- Old → new J/token (qwen3-8b, batch 8, ctx 4096, prompt 2048):

  | Chip / phase | tpuv4i_7nm | horowitz_45nm |
  |---|---|---|
  | 1P decode | 0.1239 → 0.1254 (+1.2 %) | 0.4149 → 0.4174 (+0.6 %) |
  | 1P prefill | 0.006654 → 0.007048 (+5.9 %) | 0.0157 → 0.01644 (+4.7 %) |
  | 100T decode | 0.1802 → 0.1816 (+0.7 %) | 0.419 → 0.4214 (+0.6 %) |
  | 100T prefill | 0.006988 → 0.00752 (+7.6 %) | 0.01647 → 0.01754 (+6.5 %) |

- Conclusions that moved:
  - qwen3-8b on 100T + LPDDR5X (b8): the 25 °C J-optimal SLC drops 32 → 16 GiB (per-access wire grows with √C), equal to the 85 °C optimum, so "hot die cuts the J-optimal SLC" no longer holds there.
  - wan2.1-1.3b SRAM sizing with `energy_tie`: J/token no longer limits the shrink (smaller SRAM is also shorter wire); latency does.
- Time and the summary fingerprints are unchanged (energy only).

### §245.2 Energy-aware design search 「推导」 (`core/energy_dse.py`, `/api/insight_dse`)
- Grid on the scenario's chip:
  - MAC count × `mac_scales` (default 0.5, 1, 2).
  - Engine side r ∈ {r/2, r, 2r}, kept when MACs / r² is a whole engine count.
  - SRAM × `sram_scales` (0.5, 1, 2).
  - NoC `topologies` (default: the chip's).
  - At most 96 points.
- Each point is evaluated by the cost model (time, fit) and priced with the energy table, so every term is in: MAC, SRAM, wire, NoC, DRAM and static. If the table has no NoC or wire numbers, the §244 reference values are filled and the report says so.
- Constraints. All come from the budget and none is defaulted silently:
  - Area: `full_area` (user densities) ≤ `die_mm2`. Over-area points are dropped, since area cannot be throttled.
  - Power: average card W ≤ `power_W_card` (else `energy.tdp_W`).
    - `power_mode = dvfs` (default): the point runs at the highest clock that fits, using the 0.227 DVFS model on `curve` (default dfs). Its throughput and J are taken at that clock.
    - `power_mode = reject`: the point is dropped.
  - Fit: the model must fit in memory.
- Objectives:
  - Throughput per card ↑ and J/unit ↓. For one workload, perf/W = units/s ÷ W = 1 / (J/unit), so this front is also the throughput–perf/W front.
  - `front3` adds die area ↓.
  - Dominance treats relative differences ≤ `tie` (default 1 %, as insight.TIE) as equal. The relation is then not transitive; an empty result falls back to exact dominance.

### §245.3 Result (tpuv4i_7nm, n7_tpuv4i densities, die ≤ 800 mm², ≤ 250 W/card, qwen3-8b)
- 1P decode, b64 (DRAM-bound):
  - The front is only MAC × 0.5 designs (64×64 or 128×128 engines, SRAM × 0.5–1): ≈ 3.93 k tok/s, 0.0407 J/token.
  - The 1P base (839 mm²) is over the area budget. At full MACs it would cost 0.0474 J/token (+16 %) for the same throughput. The extra J is the clocked MAC array and its static power.
- 1P prefill:
  - At 250 W the full-MAC chips run clock-capped (f × 0.46–0.53) to ≈ 33–34 k tok/s, the same as the uncapped half-MAC chips.
  - The front: 16 × 128² with 256 MiB (33.7 k tok/s, 7.42 mJ/token) and 8 × 256² capped with 128 MiB (34.0 k, 7.35 mJ).
  - Under a power budget, extra MACs buy area, not throughput.
- 100T decode, b1: J/token falls with MACs (0.90 at × 0.5 vs 1.10 at × 1) at equal throughput (DRAM-bound).

### §245.4 Not modelled
- Vector lanes / uncore area per engine (the area proxy is SRAM + MAC + PHY + uncore share).
- Frequency as a free grid variable: the clock only moves under the power cap.
- Other memory kinds or cards as design variables.

### §245.5 Default behaviour, old → new
- Energy presets: `pJ_bit_mm_onchip` unset → 0.121 (45 nm) / 0.078 (7 nm); `pJ_bit_hop_noc` unset → 0.416 / 0.160.
- See §245.1 for the numbers.

## §246 Energy-aware design search over compute + memory (0.301)

### §246.1 Memory-side axes 「推导」 (`core/energy_dse.py`)
- DRAM:
  - Either explicit catalog ids (`mems`), or `mem_types` × `mem_count_scales` × `mem_caps`.
  - Count scales apply to the scenario's own unit count when the type matches, else to the catalog default. The result snaps to the catalog's allowed counts (HBM 1–12 stacks).
  - Per-unit capacities: LPDDR uses the catalog package capacities; HBM uses height × die density / 8. Combinations the catalog does not list are skipped.
  - Rate, width and form are taken from the scenario's memory, or else from the type default.
- SLC: `slc_mibs`, absolute MiB, where 0 = no SLC.
- Default for both axes: the scenario's own memory and SLC, so 0.300 results are unchanged.
- Grid cap 96 → 240 points (compute × memory × SLC).

### §246.2 Terms per point
- Time, bandwidth and fit: the cost model at that memory, i.e. the same path as a normal eval with `mem_id`.
- Energy, from the preset's per-kind fields:
  - pJ_bit_dram_hbm 3.44 (SK hynix IEDM 2023, HBM3E) or pJ_bit_dram_lpddr 4.0 (Grace LPDDR5X).
  - Plus the chip-side PHY: 0.29 (HBM3, Chae ISSCC 2023) or 1.3 (LPDDR5X).
  - Plus HBM standby / refresh (§216).
  - Approximations, labelled in the notes 「近似（假设）」: HBM4 / HBM4E and HBM3 use the HBM3E figure; LPDDR6 / LPDDR5 use the LPDDR5X figure.
  - LPDDR background / refresh power is not modelled, so LPDDR static power is low. This is stated in the notes.
- Area: the die proxy adds the DRAM PHY = bus bits × mm2_per_dq_* (density preset). HBM is per bit from the HBM4 base-die PHY; LPDDR is per bit from the SF2 LPDDR6 x24 PHY. Both are applied across generations 「近似」.
  - The SLC adds mm2_per_mib_slc.
  - The uncore share scales with both.
- HBM shoreline (budget.shoreline_check, 11 mm per stack): "over" (more than 4 die edges) drops the point; "tight" is kept and flagged.
- Memory $ per card: capacity × the user's `mem_usd_per_GB[type]`. No price is built in. `max_mem_usd` is a limit, and `front_usd` adds $ ↓ as an objective.

### §246.3 Workload mix and SLO 「推导」
- `mix_out_tokens` = D: a request is prompt P (serving.prompt) prefill + D decode on one card, time-shared.
  - Card-s per request = P / thr_prefill + D / thr_decode.
  - J per request = P·J_prefill + D·J_decode. Shares are energy-weighted.
  - Each phase runs at its own batch and at its own capped clock under the power budget.
  - Queueing and prefill/decode interference are not modelled. Use pdqueue / pdsim for those.
- `slo_tpot_ms` / `slo_ttft_ms` limit the point. Latency is taken at the capped clock (nominal ÷ throughput factor; exact at a fixed batch).

### §246.4 Result (tpuv4i_7nm, n7_tpuv4i densities, qwen3-8b, ctx = prompt = 4096)
- 1P decode b64, ≤ 800 mm² / 250 W, MAC × {0.5, 1}, DRAM {HBM3E, HBM4, LPDDR5X, LPDDR6} × {½, 1, 2}:
  - The front is half-MAC with 2048-bit-per-stack-equivalent bandwidth: HBM4 4 stacks or HBM3E 8 stacks. Both give ≈ 4.81 k tok/s at 0.040 J/token (693 mm²).
  - The 1P base (HBM3E 4 stacks, full MACs) is over the area budget, at 3.94 k tok/s and 0.047 J/token.
- Mix (D = 512, TPOT ≤ 50 ms): the same two designs win (≈ 4.36 req/s, ≈ 51 J/request).
  - Every LPDDR point fails TPOT (70–340 ms at b64).
  - A 256 MiB SLC never makes the front: it adds 291 mm² for < 0.3 % J.
- 100T decode b8, ≤ 300 mm² / 60 W:
  - 4 HBM3E stacks (12-high) instead of LPDDR5X 4×x64 give 630 vs 77 tok/s and 0.095 vs 0.18 J/token, because static, clock and idle power are amortised over 8× the tokens.
  - It needs the 3rd / 4th die edge (shoreline tight), and its cost is only comparable once $/GB is given.

### §246.5 Not modelled
- LPDDR background power.
- Memory controller area (no public figure; `mm2_per_unit_mc` stays a user input).
- Per-generation PHY area and pJ/bit beyond the labelled proxies.
- Package / interposer cost.

### §246.6 Default behaviour, old → new
- None. The new axes default to the scenario's memory and SLC, and the 0.300 search output is unchanged apart from added row fields: mem_*, slc_mib, tpot_ms / ttft_ms, shoreline.

## §247 Sourced HBM4 / LPDDR6 energy, LPDDR background power, DVFS operating point as a design axis (0.302)

### §247.1 Per-generation DRAM pJ/bit 「推导」 (`core/energy.py`)
- New fields `pJ_bit_dram_hbm4` (HBM4 / HBM4E) and `pJ_bit_dram_lpddr6` (LPDDR6). `dram_pJ_field` picks them when the scenario's memory type matches; unset → the per-kind field (`pJ_bit_dram_hbm` / `_lpddr`) → `pJ_bit_dram` (≤ 0.301 order).
- tpuv4i_7nm preset:
  - HBM4 2.46 pJ/bit = HBM3E 3.44 ÷ 1.4. Samsung (HBM4 product page; newsroom 2026-02, 1c DRAM + 4 nm logic base die) and SK hynix (newsroom 2025-09) both state ≈ 40 % better power efficiency than HBM3E. Vendor claims, not a measured pJ/bit; the efficiency ratio is applied to the preset's HBM3E figure. HBM4E reuses it 「假设」. Context only, not used: Samsung ISSCC 2026 HBM4 lowers VDDQ (1.1 → 0.75 V class) and doubles the interface to 2048 bit.
  - LPDDR6 3.31 pJ/bit = LPDDR5X 4.0 ÷ 1.21. Kim et al. (Samsung), "A 16Gb 12.8Gb/s LPDDR6 SDRAM …", ISSCC 2026 pp. 268–270 (doi 10.1109/ISSCC49663.2026.11409018): "21.0 % better energy efficiency than LPDDR5x". Samsung's LPDDR6 page states "up to 21 %".
- The chip-side PHY pJ/bit and PHY area stay the HBM3 / LPDDR5X figures for the new generations 「近似（假设）」, as labelled in the design-search notes.

### §247.2 LPDDR background (standby + refresh) power 「来源/推导」
- New fields `mW_lpddr_ch_standby` (per x16 channel of 8 Gb) and `mW_lpddr_Gb_refresh` (per Gb). `hbm_static_W` covers LPDDR when either is set: slices = capacity_Gb / 8, W = slices · standby + capacity_Gb · refresh. Same gating as HBM (§216): charged only when no idle figure is set; the report key stays `hbm_static` / `hbm_static_W_card` (= DRAM background).
- Source: Micron Y52Q LPDDR5X data sheet (MT62F512M32D1 / MT62F512M64D2, Rev. G 05/2025), Table 16, per x16 channel of a 2-channel 16 Gb die (8 Gb per channel), maximum values:
  - IDD2N (precharge standby): VDD1 1.3 mA · 1.8 V + VDD2H 28 mA · 1.05 V + VDD2L 0.2 mA · 0.9 V + VDDQ 0.6 mA · 0.5 V = 32.2 mW 「来源」.
  - Refresh: (IDD5 − IDD3N) = (14 − 2.7) mA · 1.8 V + (124 − 35) mA · 1.05 V = 113.8 mW during tRFCab = 210 ns (Table 5), once per tREFI = 3.9 µs (JEDEC LPDDR5 at ≤ 85 °C 「二手」) → 6.12 mW per 8 Gb = 0.765 mW/Gb 「推导」.
- Upper side 「假设」: every slice is charged IDD2N. Idle ranks in power-down draw IDD2P (2.3 mA VDD2H, ≈ 12× lower on that rail), and self-refresh is lower still (Y52Q IDD61 / IDD62H 0.17 / 0.42 mA per channel at 25 °C). The tool does not model power-down residency. LPDDR6 / LPDDR5 reuse these figures 「假设」.
- Scale: 64 GB LPDDR5X = 64 slices → 2.06 W standby + 0.39 W refresh = 2.45 W; 128 GB → 4.9 W.

### §247.3 DVFS operating point as a design-search axis 「推导」 (`core/energy_dse.py`)
- `f_scales` (f / f_max in [0.15, 1], default [1]). For each grid point and each f, the chip is evaluated at freq_ghz·f. Core-rail action energies (mac, vec, sram, slc, clock, wire, noc) are scaled × V(f)², and SRAM leakage × V(f), with V from the `curve` (powercap._point, the same code path as the 0.227 cap). DRAM, links and static terms are held.
- Power budget: a point over the cap is capped from f_max with the cap search. Power rises with f, so the capped clock is below f_op and the row equals the f_op = 1 twin's capped result. `f_scale` = running clock and `f_op` = design point.
- Under "dfs" (V flat) a lower f_op never lowers J/unit: the per-action energies are unchanged and static power is amortised over fewer units. The voltage benefit needs a V–f curve (rx5700xt_n7 「二手」).
- Example (tpuv4i_7nm, n7_tpuv4i densities, die ≤ 3000 mm², 300 W, 1P qwen3-8b prefill b1 prompt 2048, rx5700xt_n7):

| MAC × | f_op | running f | k tok/s | mJ/token | W nominal at f_op |
|---|---|---|---|---|---|
| 1 | 1.0 | 0.838 (capped) | 59.1 | 5.07 | 504 |
| 1 | 0.6 | 0.6 | 42.4 | 3.29 | 140 |
| 2 | 1.0 | 0.623 (capped) | 87.4 | 3.43 | 1028 |
| 2 | 0.6 | 0.6 | 84.3 | 3.36 | 283 |
| 0.5 | 0.6 | 0.6 | 21.3 | 3.36 | 71 |

  - Under the cap, more MACs at a lower V–f point beats fewer MACs at f_max, both in throughput (+43 %) and in J/token (−34 %).
  - The front holds 1× @ 0.6 (lowest J), 2× @ 0.6 and 2× capped (highest throughput).
- Decode at small batch is memory-bound. The axis then changes only J (clock-tree term × f·V², SRAM leakage × V); throughput is flat.

### §247.4 Suite time (test-only)
- PD-queue reports spend ≥ 90 % of their time in the SLO-goodput / stability rate bisection (`_slo_rate`; cProfile, test_core_057 `_rep`: 190 of 199 s). Tests that read only operating-point outputs now replace it with a −1 sentinel (`tests/fast_queue.py`). Its outputs are exactly the keys in `SLO_KEYS`.
- Equivalence: `ACCEL_FASTQ_VERIFY=1` runs each queue_report both ways and compares every other field. All 43 wrapped modules showed no difference. The two runtime-guard tests (test_core_237 / 238) are not wrapped, since they time the full report.
- The SLO search stays covered by the unwrapped tests that read it (test_core_051 / 052 / 063 / 064 / 065 / 080 and the per-test exclusions in 054 / 056 / 060 / 061 / 073_audit / 101).

### §247.5 Not modelled
- LPDDR power-down / self-refresh residency (idle ranks). The background figure is the non-power-down upper side (0.303: modelled, §248.3).
- Per-generation PHY energy and area.
- Overclocking (f > f_max).
- Per-rail voltage of the memory (memory energy is held under DVFS).

### §247.6 Default behaviour, old → new
- tpuv4i_7nm preset (no idle figure):
  - `pJ_bit_dram_hbm4` unset → 2.457.
  - `pJ_bit_dram_lpddr6` unset → 3.306.
  - `mW_lpddr_ch_standby` unset → 32.22.
  - `mW_lpddr_Gb_refresh` unset → 0.765.
- Preset J/token, qwen3-8b, ctx 4096, prompt 2048 (old → new):

| Chip / memory | decode b8 | prefill b1 |
|---|---|---|
| 100T / LPDDR5X 4×16 GB (default) | 0.1816 → 0.2135 (+17.6 %) | 0.00777 → 0.00816 (+5.0 %) |
| 100T / LPDDR5X 4×32 GB | 0.1816 → 0.2455 (+35.2 %) | +10.0 % |
| 100T / LPDDR6 4×x96 16 GB | 0.1551 → 0.1605 (+3.4 %) | +4.1 % |
| 1P / HBM3E (default) | unchanged | unchanged |
| 1P / HBM4 4×12-high | 0.1083 → 0.0889 (−17.9 %) | −0.8 % |

- §246.4, 100T decode b8: LPDDR5X J/token 0.18 → 0.21, which widens the HBM advantage. The design-search note "LPDDR background not modelled" now appears only when the energy table lacks the LPDDR fields.
- §246.4 (1P decode b64, ≤ 800 mm² / 250 W): the tie between HBM4 4 stacks and HBM3E 8 stacks (≈ 4.81 k tok/s, 0.040 J/token) is broken. HBM4 4 stacks is now the sole front point, at 6.85 k tok/s and 0.031 J/token (693 mm²). The cheaper DRAM bits take it off the power cap, so it also runs faster.

## §248 NoC congestion (bisection bandwidth) and LPDDR power-down / self-refresh residency (0.303)

### §248.1 NoC congestion 「推导」 (`core/hardware.py`, `core/mapping.py`, `core/evaluate.py`)
- New chip field `noc_link_Bpc` (bytes / cycle / link / direction). Default 0 = unlimited, which is the ≤ 0.302 model, so there is no default change (old → new: none). It is opt-in on the chip, in `/api/insight_noc` and `/api/insight_dse` (`noc_link_Bpc`), and in the UI (NoC 链路 B/周期 on both insight panels).
- Reference value `NOC_LINK_BPC_REF` = 64 B/cycle. Source: Tenstorrent Blackhole, one 512-bit flit per cycle per link per NoC (tt-isa-documentation, BlackholeA0/NoC/README.md 「来源」). Wormhole is 32 B/cycle. Measured multicast throughput is ≈ 24 B/cycle (cross-check only, not used).
- Bisection links of the s engines an op spans (`Chip.noc_bisection_links`): mesh ⌊√s⌋, ring 2, crossbar s/2. s ≤ 1 → no NoC.
- Traffic is the 0.299 dataflow payload (§244):
  - the wide-op A-operand multicast to min(E, ⌈N/cols⌉) engines;
  - cross-group copies of core-group split plans;
  - the split-K partial-sum reduction.
- Each payload is assumed to cross the bisection once (tree multicast / reduction across the two halves 「假设」). The op then takes ≥ payload / (links × noc_link_Bpc) cycles. This is overlapped with the array and the feed (max), so a time penalty appears only when the bisection is exceeded. GemmCost gains `noc_cycles`, and `bound` = "noc" when it dominates.
- Plan choice sees the NoC time, so a congested topology can switch to a sub-group or split-K plan with less traffic. Split-K is kept only if it still wins after its reduction time.
- `noc_design_report` rows gain `tok_s_free` (the same point with an unlimited NoC) and `noc_penalty` = 1 − tok_s / tok_s_free.

### §248.2 Effect on the many-small-cores conclusion (§244)
- Setup: 1P qwen3-8b, tpuv4i_7nm preset, prefill b1 prompt 2048. Throughput penalty at 64 B/cycle:

| Prefill E × array | crossbar | ring | mesh |
|---|---|---|---|
| 2 × 512² | −28.6 % | −24.7 % | −28.6 % |
| 8 × 256² | 0 | −26.1 % | −26.1 % |
| 32 × 128² | 0 | −26.7 % | −3.9 % |
| 128 × 64² | 0 | −26.9 % | 0 |
| 512 × 32² | 0 | −57.5 % | 0 |
| 2048 × 16² | 0 | −60.3 % | 0 |

- At 32 B/cycle (Wormhole class), mesh: E2 −35 %, E8 −53 %, E32 −19 %, E128 −16 %, E ≥ 512 0. Ring: −29 to −75 %.
- Decode throughput is memory-bound and unchanged. J/token still moves, because plan choice now avoids congested plans: ring E512 0.1326 → 0.0483, ring E2048 0.478 → 0.0547, mesh E2048 0.0701 → 0.0492.
- Conclusion:
  - Without a link limit (≤ 0.302), few big engines (2 × 512²) were the lowest-J prefill design, and throughput plateaued from E = 32.
  - With a realistic link, the A-operand multicast of a few big engines saturates the narrow bisection (one or two links), and they lose about 25–30 % of prefill throughput.
  - Mesh designs with E ≥ 128 and crossbars with E ≥ 8 have enough bisection and lose no time. They still pay NoC energy (§244).
  - Rings get worse with E, because the bisection stays at 2 links.
  - Best by J: decode 128 × 64² ring (0.0473 J/token); prefill 8 × 256² crossbar (6.92 mJ/token).
  - The efficient region moves to many-small-cores on mesh (E ≈ 32–128) or a modest crossbar (E = 8), not to the fewest engines.

### §248.3 LPDDR power-down / self-refresh residency 「来源/推导/假设」 (`core/energy.py`)
- New fields, each per x16 channel slice of 8 Gb:
  - `mW_lpddr_ch_pd` (power-down);
  - `mW_lpddr_ch_sr` (self-refresh; includes refresh);
  - `ns_lpddr_pd_gap` / `ns_lpddr_sr_gap` (the shortest idle gap that pays for each state: exit latency + minimum residency).
- tpuv4i_7nm preset values:
  - Power-down: Micron Y52Q LPDDR5X (Rev. G 05/2025) Table 16, IDD2P on all rails, max: 1.4 mA·1.8 V + 2.3·1.05 + 0.2·0.9 + 0.6·0.5 = 5.42 mW 「来源」.
  - Self-refresh: Y52Q Table 18, power-down self refresh at 85 °C max: 1.2·1.8 + 7.0·1.05 + 0.2·0.9 + 0.6·0.5 = 9.99 mW 「来源」. Typical values: 0.77 mW at 25 °C, 1.03 mW at 45 °C. The max is the conservative side, matching IDD2N in §247.2.
  - Gaps (JEDEC LPDDR5 general spec, which the data sheet references 「二手」): power-down tXP 7.5 ns + entry ≈ 7.5 ns = 15 ns. Self-refresh tXSR = tRFCab 210 + 7.5 ns, plus minimum residency tSR 15 ns = 232.5 ns.
- Mechanism (`lpddr_residency`), assuming a statically scheduled accelerator knows its gaps and enters the deepest state that fits 「假设」:
  - u = DRAM busy fraction = modelled DRAM bytes per card-second ÷ effective bandwidth (≤ 1).
  - Ranks per channel n_r = slices ÷ (bus bits / 16). While a channel streams, one rank is accessed: that slice is in standby u / n_r of the time, and the others are in power-down (n_r − 1)·u / n_r (rank bursts ≫ tXP).
  - The channel-idle fraction 1 − u arrives in gaps of g = (1 − u)·window ÷ layers on the card (one weight / KV stream per layer 「假设」). If g ≥ the SR gap, the slice is in self-refresh for g − gap and in standby for the rest. Otherwise, if g ≥ the PD gap, it is in power-down likewise. Otherwise it stays in standby.
  - Refresh is charged on the non-self-refresh fraction only.
  - The PD queue's pool ledger has no window, so it credits power-down only and no self-refresh (the high side).
- Result keys: `hbm_static` gains `powerdown_W`, `selfrefresh_W`, `residency` and `W_no_lowpower` (the 0.302 bound). `hbm_static_W(mem_id, table)` without a utilisation still returns the 0.302 bound. Unset fields → the 0.302 behaviour. HBM is unchanged.
- LPDDR6 uses the LPDDR5X figures and 16-bit channel slices 「假设」.

### §248.4 Default behaviour, old → new
- tpuv4i_7nm preset (no idle figure): `mW_lpddr_ch_pd` unset → 5.415, `mW_lpddr_ch_sr` unset → 9.99, `ns_lpddr_pd_gap` unset → 15, `ns_lpddr_sr_gap` unset → 232.5.
- 100T, qwen3-8b, ctx 4096, prompt 2048. J/token (background W/card), old → new:

| Memory | decode b8 | decode b64 | prefill b1 |
|---|---|---|---|
| LPDDR5X 4×16 GB (default) | 0.2135 → 0.1968 (−7.8 %; 2.45 → 1.17 W) | 0.0780 → 0.0724 (−7.2 %) | 0.00816 → 0.00791 (−3.1 %; u 0.41, 59 % self-refresh, 0.85 W) |
| LPDDR5X 4×32 GB | 0.2455 → 0.2064 (−15.9 %; 4.91 → 1.91 W) | −14.9 % | −6.2 % |
| LPDDR6 4×x96 16 GB | 0.1605 → 0.1521 (−5.2 %) | −4.7 % | −3.2 % |
| HBM3E (1P default) | unchanged | unchanged | unchanged |

- Memory-bound decode keeps u = 1, so the saving comes from rank-level power-down. Capacity added to a channel as ranks now costs mostly power-down power. Of the +2.45 W LPDDR background in 0.302, about half remains.

### §248.5 Not modelled
- Per-link NoC routing and hot spots: one bisection cut only, no adaptive routing, no contention between concurrent ops.
- Temperature dependence of self-refresh (85 °C max used).
- LPDDR6 channel geometry; HBM power-down / self-refresh.
- Exit latency on the access path: the scheduler is assumed to wake the rank ahead of time.

## §249 PD-queue LPDDR window; per-link (XY) NoC contention; suite-time check (0.304)

### §249.1 PD-queue LPDDR residency with an evaluation window 「推导/假设」 (`core/pdqueue.py`, `core/energy.py`)
- 0.303 gave the queue ledger (`pdqueue._energy`) a pool's average DRAM busy fraction but no window. So it credited power-down for every idle moment (any gap ≥ tXP assumed) and never credited self-refresh. The single-evaluation path (`energy_report`) did credit self-refresh.
- 0.304 gives every pool a representative iteration (`_pool_window`): its memoised evaluation of the pool's phase at the largest batch it ran (decode pool: decode; prefill pool: prefill; colocated: decode 「近似」). `lpddr_window(r)` returns three values:
  - the iteration's DRAM busy fraction u_iter = DRAM bytes ÷ (cards × window) ÷ effective bandwidth;
  - the window;
  - the card's layers.
- `hbm_static_W_window(mem, table, u, win)`:
  - The cards iterate a fraction ρ = u ÷ u_iter (≤ 1) of the time. While iterating, the residency is the single-evaluation one: `lpddr_residency` at u_iter, with gaps (1 − u_iter)·window ÷ layers (§248.3).
  - An empty-queue period lasts at least one iteration window, because the next request starts an iteration 「假设」. It is spent at u = 0 with that gap, so in self-refresh.
  - W = ρ·W_iter + (1 − ρ)·W_idle. Residency fractions are linear in time 「推导」.
- Consistency: a single-phase operating point priced through `_energy` with its own window equals `energy_report` exactly (test_core_304).
- Without a window, the 0.303 figure is returned. It is not strictly an upper bound: it ignored the tXP exit window. On decode b8 it is 1e-5 below the windowed figure.
- Example: 100T LPDDR5X 16 GB, qwen3-8b, PD 2 prefill + 6 decode cards, prompt 2048, out 512.

| Load | Mode | J/token 0.303 → 0.304 | DRAM background J/token | Prefill pool ρ / self-refresh | Decode pool ρ / self-refresh |
|---|---|---|---|---|---|
| 0.3 | PD | 0.42891 → 0.42747 (−0.34 %) | 0.03572 → 0.03428 | 0.088 / 0.97 | 0.78 / 0.22 |
| 0.3 | colocated prefill-first | 0.46777 → 0.46677 | 0.03763 → 0.03663 | — | ρ 0.72 / 0.28 |
| 0.7 | PD | 0.21739 → 0.21694 | 0.01603 → 0.01557 | 0.21 / 0.93 | 0.92 / 0.085 |

- The change is small: at the 85 °C maximum, self-refresh (9.99 mW per slice) is only 13 % below power-down + refresh (11.5 mW). At typical temperatures, self-refresh is about 10× lower (0.77 mW at 25 °C), and the window would matter more. The temperature dependence is still not modelled.
- Old → new: J/token of PD-queue reports on LPDDR memories with the tpuv4i_7nm preset drops by ≤ 0.4 % in the cases above. HBM and single-evaluation reports are unchanged.

### §249.2 Per-link NoC contention, dimension-ordered routing 「推导」 (opt-in)
- New chip field `noc_routing`:
  - "bisection" is the default and the 0.303 model: payload ÷ (bisection links × L).
  - "xy" prices the busiest link of the dataflow's trees.
  - It has no effect while `noc_link_Bpc` = 0. There is no default change.
- Parameters: `/api/insight_noc` and `/api/insight_dse` take `noc_routing`. UI: 拥塞模型 select on both panels.
- Busiest-link load for a payload among s blocks of g engines (`Chip.noc_xy_load`):
  - **all-gather.** The A operand is sharded over the engines, because outputs stay local and become the next op's A (§244.1), and every engine needs all of it.
    - Mesh R × C (R = ⌊√s⌋), X then Y routing: after the row phase every node forwards down its column. So the column link below row r carries every shard of rows ≤ r, and the busiest carries (R − 1)/R of the payload. A 1 × C row: (C − 1)/C.
    - Ring (shortest way round): ⌊s/2⌋/s, which equals the 0.303 bisection figure.
    - Crossbar: an output port ingests (s − 1)/s.
  - **reduce.** Split-K partial tiles go over a tree that mirrors the multicast; every tree edge carries one tile.
  - Load is divided by the block-boundary links a tree edge spreads over: mesh ⌊√g⌋, ring 1, crossbar g.
- **Concurrent traffic.**
  - A core-group split's cross-group copy (all-gather over the sp groups) uses the same links as the groups' own in-group multicasts, so the two loads add. Under "bisection" they overlap (max).
  - Instances spread over groups stay inside their own rectangle / arc 「假设」, so groups do not contend with each other.
- Why it matters: the bisection average assumes perfectly spread traffic. Under XY multicast, each engine ingests the whole A through one column link, so mesh time ≈ (1 − 1/R)·A / L whatever the mesh size. The bisection model gave A / (⌊√s⌋·L), optimistic by ≈ √s / 2.
- Effect on the many-small-cores conclusion: 1P qwen3-8b prefill b1, prompt 2048, tpuv4i_7nm preset, 64 B/cycle. Cells are k tok/s / mJ/token.

| E × array | crossbar bisection | crossbar XY | ring (both) | mesh bisection | mesh XY |
|---|---|---|---|---|---|
| 2 × 512² | 45.1 / 7.27 | 47.6 / 7.19 | 47.6 / 7.19 | 45.1 / 7.27 | 47.6 / 7.19 |
| 8 × 256² | 68.0 / 6.92 | 39.0 / 7.47 | 50.3 / 7.17 | 50.3 / 7.17 | 50.3 / 7.17 |
| 32 × 128² | 70.4 / 7.55 | 33.6 / 7.94 | 51.6 / 7.31 | 67.7 / 7.16 | 39.8 / 7.58 |
| 128 × 64² | 70.9 / 8.71 | 47.3 / 8.40 | 51.9 / 7.62 | 70.9 / 7.78 | 45.3 / 7.77 |
| 512 × 32² | 71.2 / 11.17 | 69.1 / 9.39 | 30.3 / 8.98 | 71.2 / 9.95 | 58.0 / 8.31 |
| 2048 × 16² | 71.3 / 15.56 | 71.3 / 15.53 | 27–28 / 21.1 | 71.3 / 14.03 | 71.3 / 10.49 |

  - Unlimited NoC: 63.1 / 68.0 / 70.4 / 70.9 / 71.2 / 71.3 k tok/s for E = 2 … 2048.
  - Under XY, the mid-size meshes and crossbars (E = 8–128) lose 30–52 % of prefill throughput. Their ganged plans all-gather A through single column links / ports.
  - At E = 2048 the planner spreads instances (heads / rows) over small groups that keep their operands local. That avoids the all-gather: full throughput, and on mesh 25 % less J than under the bisection model (less NoC traffic chosen).
  - Lowest prefill J moves from 8 × 256² crossbar (6.92 mJ, bisection) to 8 × 256² ring / mesh (7.17 mJ). The 0.303 "mesh E ≈ 32–128" sweet spot disappears under XY. The fastest designs are either the fewest engines with the narrowest tree (E ≤ 8, ring / mesh) or the most engines with local plans (E ≥ 512 mesh).
  - Decode is memory-bound and unchanged.
- Not modelled: adaptive / YX-alternating routing (which would halve the column load), virtual-channel or head-of-line effects, contention between consecutive ops (they are serialised).

### §249.3 Suite wall 471 s (0.302) → 648 s (0.303): not a test-cost regression
- Per-file walls: the +1407 s summed difference sits in the 8 longest files, which run concurrently at the start of the longest-first schedule. Each was ≈ 1.8× slower: 051 / 052 / 061 / domain / scenario_search / topk2 rose by 150–170 s each, 210 by 225 s, 086 by 218 s (10 → 228 s). All other files changed by ≤ 17 s.
- Re-run of those 8 files with -j 8, back to back on the 0.302 and 0.303(+0.304 WIP) trees:
  - wall 197 s vs 190 s;
  - machine CPU 925 vs 872 CPU-s;
  - per file within ±15 s;
  - 086 at 9.7 vs 9.1 s, and 9 s alone.
- The code costs the same. The 0.303 run met external load on the shared box (other agents' work), so there is nothing to cut.
- test_core_210 guard: the request's own CPU time (`time.process_time`; the API's process pool is server-only, so library calls are in-process) must stay under the same thresholds (20 s; 80 s for sensitivity / fit / sweep). Wall time is checked against the threshold × the contention factor of a calibration loop measured before and after the request, with 2× slack. A compute regression (the 0.209 127 s replay) moves CPU time and is caught as before. Suite contention no longer trips it.
- Wall-clock steps: the first 0.304 full suite failed test_core_238's 60 s runtime guard at a measured 318.9 s, while the runner's monotonic clock timed the whole file at 78.5 s. Rerun alone, it passed in 44 s. The box's wall clock (time.time) had stepped mid-run. The runtime guards in test_core_209 / 210 / 238 and the harness's per-test times now use time.perf_counter (monotonic). Same thresholds, so this is not a weakening.

## §250 NoC model consolidation (XY default, DRAM ingress through the NoC); LPDDR self-refresh / refresh vs temperature (0.305)

### §250.1 Default NoC contention model and DRAM ingress 「来源/推导」 (`core/hardware.py`)
- **Routing default.** `noc_routing` default "bisection" → "xy". This only matters when `noc_link_Bpc` > 0; the preset chips leave it at 0, so their results are unchanged.
- **Why XY is the default (mechanism).**
  - The bisection model (§248) assumes the payload spreads perfectly over every cut link. For a multicast that is not physically achievable: every destination engine must ingest the whole operand through its own input links.
  - A mesh routes dimension-ordered. Tenstorrent Wormhole / Blackhole route X then Y on each NoC (tt-isa-documentation 「来源」), and the busiest-link load of XY trees is what the hardware sees.
  - Ring: XY equals the bisection figure (½ the payload per link), so nothing changes there.
  - Crossbar: the output port's ingress (s − 1)/s is the physical limit. The bisection's s/2 links overstated it by up to s/2.
  - "bisection" stays selectable as the optimistic lower bound.
- **DRAM ingress, new mechanism.** DRAM bytes reach the engines through the same NoC.
  - Blackhole NoC 0 places its GDDR tiles in two columns (x = 0 and x = 9), one tile per Tensix row in each; each NIU → router is one 512-bit flit per cycle (tt-isa-documentation BlackholeA0/NoC 「来源」).
  - New chip field `noc_mem_attach`:
    - "edge" (default): two memory columns, one attach link per engine row in each. Mesh and crossbar ports: 2·⌊√E⌋ links. Ring: two memory stops injecting both ways, 4 links 「假设」.
    - "none": ≤ 0.304, DRAM not limited by the NoC.
  - `System.dram_GBps` = min(memory effective, DMA issue limit, links × noc_link_Bpc × clock). The insight's `_dram_delivered` mirrors it.
  - Cross-check: Blackhole's 10 rows × 2 × 64 B × 1.35 GHz = 1.7 TB/s per NoC, against 512 GB/s of GDDR6. A balanced design is not NoC-limited, as expected.
- **Inter-op contention.**
  - Under XY, the DRAM stream rides the row (X) links out of the memory columns. An activation all-gather's busiest links are column (Y) links, so weight streaming and activation multicast mostly use different links and do not add at the bottleneck.
  - The residual is the all-gather row phase: (C − 1)/(R·C) of the payload per row link. It is not added 「近似」, and it is small next to the column load.
  - Consecutive ops are serialised in the cost model, so they do not overlap on links.
  - Concurrent sharing within an op (cross-group copy + in-group multicast) adds, as in 0.304.
- **Old → new** (only when `noc_link_Bpc` > 0 and the fields are not given):
  - routing bisection → xy (§249.2 tables);
  - DRAM ingress unlimited → 2·⌊√E⌋·L·f.
  - 1P (HBM3E 4 stacks, 3.30 TB/s effective, 1 GHz) at L = 64: decode b64 at E = 32 drops 3.94 → 0.76 k tok/s.

### §250.2 The many-small-cores conclusion, re-derived
- Setup: 1P, qwen3-8b, tpuv4i_7nm preset; decode b64 ctx 4096; prefill b1 prompt 2048. Throughput in k tok/s, energy in mJ/token. Each cell: crossbar | ring | mesh.

| L (B/cycle) | E × array | decode k tok/s | decode mJ/token | prefill k tok/s | prefill mJ/token |
|---|---|---|---|---|---|
| 0 (unlimited) | 2 … 2048 | 1.15 / 3.45 / 3.94 (E ≥ 32) | 82.8 → 47.2 (min E32) | 63.1 → 71.3 | 6.86 (E2) → 14.0 |
| 64 | 2 × 512² | 0.15 / 0.31 / 0.15 | 400 / 217 / 400 | 17.1 / 34.2 / 17.1 | 9.28 / 7.65 / 9.28 |
| 64 | 32 × 128² | 0.76 / 0.31 / 0.76 | 106 / 216 / 106 | 32.9 / 34.2 / 38.7 | 7.98 / 7.86 / 7.62 |
| 64 | 128 × 64² | 1.68 / 0.31 / 1.68 | 66.6 / 216 / 66.4 | 46.8 / 34.2 / 44.9 | 8.41 / 8.18 / 7.78 |
| 64 | 512 × 32² | 3.36 / 0.31 / 3.36 | 52.9 / 217 / 50.5 | 69.0 / 28.5 / 57.9 | 9.39 / 9.10 / 8.32 |
| 64 | 2048 × 16² | 3.94 / 0.31 / 3.94 | 57.1 / 219 / 49.1 | 71.3 / 25.7 / 71.3 | 15.5 / 21.3 / 10.5 |
| 256 | 2 × 512² | 0.61 / 1.04 / 0.61 | 126 / 88 / 126 | 60.2 / 62.2 / 60.2 | 6.98 / 6.93 / 6.96 |
| 256 | 32 × 128² | 3.06 / 1.22 / 3.06 | 51.9 / 78.9 / 51.5 | 70.2 / 66.0 / 70.2 | 7.55 / 7.18 / 7.15 |
| 256 | 128 × 64² | 3.94 / 1.22 / 3.94 | 48.4 / 79.0 / 47.4 | 70.9 / 62.3 / 70.9 | 8.71 / 8.37 / 7.65 |

- Design rule 「推导」: the NoC delivers the memory's bandwidth only if L ≥ BW_eff ÷ (2·⌊√E⌋·f). For 1P / 3.30 TB/s that is 1649 / 824 / 330 / 150 / 75 / 37 B/cycle at E = 2 / 8 / 32 / 128 / 512 / 2048.
- 100T (LPDDR5X, 191 GB/s) needs ≤ 48 B/cycle from E = 8. Its results at L = 64 are unchanged, and the fewest engines (E16) stay best.
- Conclusion:
  1. With HBM-class memory and a Blackhole-class link (64 B/cycle), many small cores on a mesh are required. Ingress grows with √E, so decode reaches full memory bandwidth only at E ≥ 2048, and E ≥ 512 gets 85 %. Few big engines lose 81–91 % of decode.
     - Lowest decode J: 2048 × 16² mesh (49.1 mJ).
     - Lowest prefill J: 32 × 128² mesh (7.62 mJ).
     - Highest prefill throughput: E = 2048 mesh / crossbar (71.3 k).
     - Rings are unusable: 4 ingress links.
  2. With 4× wider links (256 B/cycle), E ≥ 128 mesh / crossbar is fully fed for decode. Prefill is within 5 % of unlimited from E = 8. The best J is E = 32–128 mesh for decode (47.4 mJ) and 2 × 512² ring / mesh for prefill (6.93 mJ), close to the unlimited-NoC result.
  3. So the real choice is between link width and core count. The 0.303 bisection-model claim, that mid-size meshes are free and few big engines are only mildly penalised, does not survive. Under XY + DRAM ingress, a narrow NoC forces many small cores, and a wide NoC restores the few-big-engines energy optimum.
- Not modelled:
  - NoC energy of the DRAM bytes' hops;
  - the second NoC (Blackhole has two, which would double every capacity: read L as the total per direction);
  - adaptive routing.

### §250.3 LPDDR self-refresh / refresh power vs temperature 「来源/推导/二手」 (`core/energy.py`)
- New field `dram_temp_C` (0–95 °C). Unset → `sram_temp_C` (DRAM case ≈ die junction 「假设」; conservative for discrete packages) → unset → the 85 °C maximum (≤ 0.304).
- Self-refresh source: Micron Y52Q Tables 18 / 21, full-array power-down self refresh per x16 channel of 8 Gb, all rails:

| Temperature | Value type | mW per channel |
|---|---|---|
| 25 °C | typical | 0.771 |
| 45 °C | typical | 1.23 |
| 65 °C | typical | 2.84 |
| 85 °C | maximum (guaranteed) | 9.99 |

  - `lpddr_sr_temp_factor` interpolates log-linearly between the points (self-refresh current is dominated by the temperature-compensated refresh rate and leakage, both ≈ exponential 「推导」), clamps below 25 °C, and scales the preset's `mW_lpddr_ch_sr` (the 85 °C value). A user value is read as the 85 °C value.
  - Below 85 °C the curve mixes typical and maximum points (labelled).
- Refresh issued by the controller (standby / power-down time):
  - 1× tREFI = 3.9 µs at ≤ 85 °C;
  - 2× above 85 °C (MR4 refresh multiplier, JEDEC LPDDR5 「二手」), also added to the self-refresh figure as one more slice refresh share.
  - The slower low-temperature MR4 rates are vendor thresholds, not in the data sheet, and are not credited.
- Not temperature-scaled: IDD2N / IDD2P (the data sheet gives maxima only) and HBM.
- Effect, 100T LPDDR5X 16 GB, qwen3-8b. DRAM background W/card, prefill / decode b8:

| Temperature | prefill W | decode b8 W |
|---|---|---|
| 25 °C | 0.504 | 1.167 |
| 45 °C | 0.521 | 1.167 |
| 65 °C | 0.582 | 1.167 |
| 85 °C / unset | 0.854 | 1.167 |
| 95 °C | 1.246 | 1.559 |

  - Prefill J/token at 25 °C: 0.0079085 → 0.0078532 (−0.7 %).
  - Memory-bound decode has no self-refresh, so it is unchanged at ≤ 85 °C. At 95 °C it is +2.6 %.
- Old → new: none unless `dram_temp_C` or `sram_temp_C` is set. With `sram_temp_C` set, LPDDR background now follows it (default change for those inputs).

## §251 NoC: DRAM-hop energy, chip-panel NoC controls, NoC link width as an energy-DSE axis (0.306)

### §251.1 NoC hop energy of DRAM traffic 「推导/假设」 (`core/hardware.py`, `core/energy.py`)
- Since 0.305, DRAM bytes enter the array through the NoC (`noc_mem_attach` "edge": a memory column on each side of the engine array, one attach per engine row, Blackhole NoC 0 「来源」). Their router traversals and link pitches were not yet priced. They are now.
- `Chip.noc_mem_hops()` gives (router traversals, link pitches) per DRAM byte:
  - **Mesh** R × C (R = ⌊√E⌋, C = ⌈E/R⌉): an engine in column c is d(c) = min(c, C + 1 − c) hops from the nearer memory column (the first hop is the attach → router link). Traversals = pitches = the mean of d over the columns 「推导」. Values: E2 1.0, E16 1.5, E32 2.29, E512 6.5, E2048 12.0.
  - **Ring**: two memory stops split the ring into two arcs of E/2. Same nearest-stop mean, ≈ E/8 (E2048: 256.5).
  - **Crossbar**: 1 switch traversal; the wire runs the same floor-plan distance as the mesh 「假设」.
  - "none" or E = 1 → 0 (≤ 0.305).
- `action_counts`: `noc` += DRAM bytes × traversals and `noc_wire` += DRAM bytes × pitches. This covers every DRAM byte booked (stages, draft passes, VLM encoder, gen parts), priced by the existing `pJ_bit_hop_noc` / `pJ_bit_mm_onchip × pitch`. The report adds `noc.dram` {byte_hops, byte_pitches, hops_per_byte, pitches_per_byte, J}.
- Timing is untouched. DRAM delivery is already capped by the ingress links (§250.1).
- SLC hits are not routed: the SLC access already carries its own distance term (§246).
- **Old → new** (tpuv4i_7nm preset, qwen3-8b, default chips — the preset sets `pJ_bit_hop_noc`, so this is a default change):

| chip / phase | old J/token | new J/token | Δ |
|---|---|---|---|
| 1P decode b8 | 0.12538 | 0.14522 | +15.8 % |
| 1P decode b64 | 0.047366 | 0.054109 | +14.2 % |
| 1P prefill b1 p2048 | 0.0071614 | 0.0072217 | +0.8 % |
| 100T decode b64 | 0.072369 | 0.075358 | +4.1 % |
| 100T decode b8 | 0.19675 | 0.20561 | +4.5 % |

  - On 1P (32 engines) one pitch is 3.6 mm. A DRAM byte crosses 2.29 hops × (0.16 router + 0.078 × 3.6 wire) ≈ 1.0 pJ/bit, against 3.44 + 0.29 pJ/bit for HBM + PHY. The wire is ⅔ of it.
  - Sweep (1P, decode b64): mesh E2 / 32 / 512 / 2048 +10 / +14 / +19 / +22 %; crossbar +10 / +11 / +8 / +7 %; ring at E512 / 2048 +75 / +70 % (rings are already unusable, §250.2). Prefill +0.2–10 %.
  - Design conclusion moved: the wan2.1-1.3b energy-tied SRAM sizing (insight_sizing, energy_tie) is now limited by J/token. Minimum SRAM 16 → 19.0 MiB of 64, because a smaller SRAM's extra DRAM bytes now also pay NoC hops. test_core_300 pins "none" for its 0.300 record.

### §251.2 NoC link width as an energy-DSE axis 「来源/推导/假设」 (`core/energy_dse.py`)
- `energy_design_search(noc_links=[…])` (API `noc_links`, 1–6 widths in B/cycle; UI 「NoC 链路轴」). Every grid point is evaluated at each width. It is exclusive with the single-value `noc_link_Bpc`; without the axis nothing changes.
- Each point is charged the routers' area and width-dependent static power (`noc_router_cost`). Reference: Park et al., DAC 2012, the MIT 16-node 4 × 4 mesh, 45 nm SOI, 1.1 V, 1 GHz, 5-port, 64-bit flits 「来源」:
  - Table 4: router with a full-swing 5 × 5 crossbar 227,230 µm²; the crossbar alone 26,840 µm².
  - §4.1: clocking plus full-swing datapath at near-zero load 5.6 mW/router (the non-data floor); measured leakage 76.7 mW for the chip → 4.8 mW/router (includes NICs, so an upper bound 「推导」).
- Width scaling 「推导/假设」:
  - Crossbar ∝ (ports · width)²: a wire-limited P × P switch of w-bit buses (Dally & Towles ch. 17).
  - Buffers, pipeline registers and datapath ∝ ports · width. Allocators and route logic do not scale with width; they are lumped into the linear part, which overstates wide links and understates narrow ones 「假设」.
  - Clock floor ∝ width (clocked flops); leakage ∝ area 「假设」. Ring routers have 3 ports. The crossbar topology (one switch for all engines) is not costed.
- Node scaling: area × (the user's `mm2_per_kmac` / 1024 ÷ Horowitz's 45 nm bf16 MAC 5824 µm² 「二手」), i.e. logic scales like the MAC datapath 「推导/假设」. Power ÷ 2.6, the preset's 45 → 7 nm factor 「推导」, at the running clock. No area density → power only.
- Resulting 7 nm router (n7_tpuv4i density, 1 GHz, mesh):

| link B/cycle | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| mm² / router | 0.026 | 0.059 | 0.142 | 0.383 | 1.16 | 3.91 |
| mW / router | 4.0 | 8.4 | 18.6 | 44.2 | 116 | 344 |

  - Cross-check: a Blackhole-class 64 B NoC at ~140 tiles would be ≈ 54 mm² per NoC — plausible for a 600+ mm² die, probably high (no public router area) 「近似」.
- Accounting: NoC area is added to the point's die-area proxy. NoC static W is added to W/card and J/unit after the power cap; a point then over budget is dropped, not re-capped 「近似」. `uncore_frac` (routing / white space) may already contain part of it, so it may be counted twice 「假设」.
- **The trade-off on the front.** 1P, qwen3-8b decode b64, HBM3E, n7_tpuv4i densities, die ≤ 1300 mm², ≤ 300 W; sides 16–256 at a fixed MAC count; links 8 / 32 / 128 B/cycle (each cell: k tok/s · mJ/token · NoC mm² / W):

| E × array | 8 B | 32 B | 128 B |
|---|---|---|---|
| 8 × 256² | 0.04 · 1506 · 0.2 / 0.0 | 0.15 · 408 · 1.1 / 0.1 | 0.61 · 134 · 9 / 0.9 |
| 32 × 128² | 0.10 · 627 · 0.8 / 0.1 | 0.38 · 188 · 4.5 / 0.6 | 1.53 · 78.9 · 37 / 3.7 |
| 128 × 64² | 0.21 · 309 · 3.4 / 0.5 | 0.84 · 110 · 18 / 2.4 | **3.36 · 61.3 · 149 / 15** ★ |
| 512 × 32² | 0.42 · 181 · 13 / 2.0 | 1.68 · 82.8 · 73 / 9.5 | 3.94 · 73.2 · 595 / 60 (area ✗) |
| 2048 × 16² | 0.86 · 126 · 54 / 8.2 | **3.44 · 77.9 · 291 / 38** ★ | 3.94 · 126 · 2379 / 238 (✗) |

  - Front (★): narrow NoC × many small cores (2048 @ 32 B) reaches the throughput, but pays 2.6× the NoC power and 2× the NoC area. Wide NoC × fewer bigger cores (128 @ 128 B) is 21 % lower J at −2 % throughput.
  - Wide NoC × many cores is unaffordable: router area grows as width², times E.
  - So with a priced NoC, the 0.305 "many small cores are required" conclusion becomes a choice: buy ingress with routers (E) or with link width (L). The energy optimum sits at the smallest E that L can still feed (L ≥ BW ÷ (2·⌊√E⌋·f), §250.2).

### §251.3 Row-link sharing between weight streaming and activation all-gather 「推导」 (documented; mesh exact)
- 0.305 (§250.1) left the all-gather's row-phase load beside the DRAM stream unadded 「近似」. Derivation for one row link, direction away from the left memory column, mesh R × C, g = 1, h = ⌈C/2⌉ columns fed from each side, op DRAM bytes D and all-gather payload P over s = E engines:
  - DRAM on link j (between columns j and j + 1, j < h): (h − j)·D/E. The attach link carries h·D/E = D/(2R), the modelled ingress per link.
  - All-gather X phase on link j: j shards = j·P/s. The modelled busiest link is the column link, (R − 1)/R·P ≥ (C − 1)·P/s for R ≥ 2. For R = 1, the X maximum (C − 1)/C·P is itself the modelled figure.
  - Sum (h − j)·D/E + j·P/s is linear in j, so its maximum is at j = 1 or at the far end. At j = 1: if P/s ≤ D/E it is ≤ h·D/E (the ingress); else it is < h·P/s ≤ (C − 1)·P/s (the X maximum).
  - **So on a mesh, the shared row links never exceed max(ingress link, busiest all-gather link), which is what the model takes. The 0.305 approximation is exact; nothing to add.**
- Not exact (documented, not modelled, needs per-op DRAM bytes in the mapping):
  - **Ring**: the all-gather loads every link uniformly, ⌊s/2⌋/s·P. The link next to a stop carries (q − 1)·D/E + ½P (q = E/4). The excess over the modelled max is ≤ min(½P − D/E, (q − 1)·D/E), up to ~2× when the two are equal. Rings are already NoC-bound.
  - **Crossbar output port**: an engine ingests D/E + (s − 1)/s·P through one port. The excess over the model is ≤ D/E, relative ≤ 1/h (up to 2× at E = 2).
  - **Mesh ejection port** (a separate gap found here): each engine ingests (s − 1)/s·P + D/E through its local port. The model bounds only network links, (R − 1)/R·P. At equal width the ejection port is busier, by (1 − 1/s)/(1 − 1/R): +9 % at E = 128, 1.5× at E = 4, plus D/E. Wider NIU ports avoid it in real designs.
  - Split-K reduce trees run after the K-sliced GEMM (sequential in the model), so they do not share with that op's stream.

### §251.4 Chip panel
- 高级芯片参数 gains NoC 拓扑 (mesh / ring / crossbar), NoC 链路 B/周期 (empty = unlimited), NoC 拥塞模型 (XY default / 二分) and DRAM 接入 (edge default / none). They are bound through `S.chipOver`; empty means the preset or default. The fold summary shows the link width.

## §252 NoC: router power before the power cap, router area counted once, local ports shared with the DRAM stream (0.307)

### §252.1 The two 0.306 approximations, removed 「来源/推导」 (`core/energy.py`, `core/powercap.py`, `core/budget.py`, `core/energy_dse.py`)
- **Router static power before the cap.**
  - 0.306 added the NoC routers' W to a point after `_run` had capped it, and dropped the point if that exceeded the budget 「近似」.
  - The routers are now part of the energy table, so the power cap and DVFS see them:
    - `pJ_noc_router_cycle`: clock and always-clocked datapath floor per cycle of a reference router (64-bit flits, 5 ports). Core rail, so DVFS scales it × V² (powercap `CORE_ACTIONS["noc_clk"]`) and × f through the re-evaluation.
    - `mW_noc_router_leak`: leakage of the reference router. Under DVFS it scales ∝ V, the same rule as SRAM leakage (AccelWattch eq. 2, §221). Under the duty-cycle cap it is static (`static_W`, `_power_cap` static keys).
  - `noc_router_W(table, chip)`: E × lin × pJ × f + E × area_x × mW, with `noc_router_scale` (§251.2: lin = ports/5 · width/64 bit; area_x = crossbar ∝ lin² plus the rest ∝ lin, over the reference area).
    - Applies only with `noc_link_Bpc` > 0 on a mesh / ring with E > 1. The report gains `noc_router` and `noc_leak_W_card`; J rows `noc_clk` (breakdown "noc") and `noc_leak` (breakdown "static").
    - No preset fills the fields, so default results are unchanged. The DSE `noc_links` axis fills them with NOC_ROUTER_CLK_PJ_45 / 2.6 = 2.15 pJ and NOC_ROUTER_LEAK_MW_45 / 2.6 = 1.84 mW (Park DAC 2012 「来源」, 45 → 7 nm 「推导」). User values win.
  - Old → new (1P, qwen3-8b decode b64, HBM3E, n7_tpuv4i, die ≤ 1300 mm², sides 16–256, links 8 / 32 / 128):

| point | 0.306 | 0.307 |
|---|---|---|
| 250 W: 2048 × 16² @ 32 B | dropped: 268 W incl. 38.1 W NoC | capped f × 0.93: 3.19 k tok/s, 78.4 mJ, 250 W |
| 250 W: 512 × 32² @ 128 B | dropped (power and area) | capped f × 0.56, 3.77 k, 66.3 mJ (area ✗) |
| 300 W: 2048 × 16² @ 128 B | 495 W (power and area ✗) | capped f × 0.16, 2.15 k, 300 W (area ✗) |
| uncapped points | — | identical (the router W was already in W/J) |
| front | {128 × 64² @ 128 B} at 250 W | unchanged |

- **Router area counted once.** What `uncore_frac` covers:
  - The 0.23 preset is TPUv1's control 2 % plus the unlabelled 21 % of the floor plan (Jouppi ISCA 2017 Fig. 2: buffers 37 %, compute 30 %, I/O 10 %, control 2 % 「来源」).
  - TPUv1 has one 256 × 256 MXU and no on-chip network. "The internal blocks are typically connected together by 256-byte-wide paths" (§2 「来源」). So the unlabelled share is global wiring (those dedicated paths) and white space, not packet routers 「推导」.
  - Split 「推导/假设」:
    - The NoC's **link wires** play the role of TPUv1's block-to-block paths and stay inside the uncore share (not added).
    - The **routers** are logic that TPUv1 does not have. They are added once as `terms["noc_routers"]`, after the uncore share, not scaled by it: Park's router area is a post-layout figure that already includes its own placement.
    - `mm2_fixed` means "everything else, including the NoC" (Budget field note), so routers are not added when it is given (a note says so).
  - Code: `budget.noc_router_mm2(chip, b)` and `full_area(scn, b, noc_routers=True)`; the DSE takes the term from there (`row["noc_mm2"]`).
  - Old → new: with the preset densities (uncore_frac, no mm2_fixed) the area is unchanged. With mm2_fixed set, a point's mm² drops by its router area (0.306 counted it twice).

### §252.2 Local (eject) ports and the DRAM stream sharing them 「推导/假设」 (`core/hardware.py`, `core/mapping.py`, `core/evaluate.py`)
- §251.3 left three bounded excesses unmodelled: ring stop links, crossbar ports and mesh local ports. They are now modelled under the opt-in `chip.noc_eject` = "link" (needs `noc_link_Bpc` > 0 and XY): every engine's local (eject) port is one link wide 「假设」 (Blackhole NIU ↔ router: one 512-bit flit per cycle, the same as a link 「来源」).
  - `noc_eject_load`: all-gather (s − 1)/s · P / g per port; reduce P / g (a lower bound: XY tree nodes with 2–3 children ingest more). `noc_traffic_cycles` takes max(busiest link, local port), so the port can bind an op by itself. Mesh 4 × 4: port 15/16 P vs column 3/4 P. Ring: port (s − 1)/s P vs link ½ P, about 2×. Crossbar: the port is already its figure, so the multicast bound does not change. Only the DRAM share below applies (the E ≤ 32 rows).
  - The mapping carries op-total `noc_link` / `noc_eject` cycles (GemmCost; wide, split and split-K paths). `_op_seconds` returns them as seconds; the stage sums `noc_lk_s` / `noc_ej_s`.
  - **DRAM sharing.** `Chip.noc_share_seconds`: every engine's port carries Σ op port load + D / E. On a ring, the link next to a memory stop (stops are ring nodes, so the all-gather passes them) carries Σ op link load + D / 4. The stage DRAM time `_dram_t(sys, dram, agg)` (and the component path) is ≥ that. Mesh row links need no term (§251.3 proof).
- **Why stage-level, not per-op.** The cost model has no per-op DRAM timing. DRAM is a stage roofline term, max(Σ op time, D / BW), which assumes the stream overlaps the whole stage, and memplan decides residency per stage. Attributing D to ops would invent a schedule the rest of the model does not have. The shared-resource bound is therefore taken at the same level and under the same overlap assumption: per-op multicast loads summed from the mapping, plus the stage's DRAM bytes.
- Effect (1P, qwen3-8b, HBM3E 3.30 TB/s; throughput eject "link" vs "off"; decode b64 / prefill b1 p2048):

| L | E | crossbar | ring | mesh |
|---|---|---|---|---|
| 64 | 2 | −1.7 / −7.3 % | −50.9 / −53.7 % | −1.7 / −7.3 % |
| 64 | 32 | 0 / −0.5 % | −0.4 / −33.0 % | 0 / −15.5 % |
| 64 | 128 | 0 / 0 | −0.4 / −33.0 % | 0 / −8.7 % |
| 64 | 512 | 0 / 0 | −0.6 / −23.2 % | 0 / −5.4 % |
| 64 | 2048 | 0 / 0 | −0.6 / −33.7 % | 0 / −0.5 % |
| 256 | 2 | −1.7 / −40.4 % | −42.1 / −42.3 % | −1.7 / −40.4 % |
| 256 | 32 | 0 / 0 | −0.9 / 0 | 0 / −2.2 % |
| 256 | 128–2048 | 0 / 0 | −0.8…−1.2 / −4…−29 % | 0 / 0 |

  - Decode stays ingress-bound (§250.1), so sharing matters little there. E = 2 is the exception: each port carries half the DRAM stream plus the whole all-gather.
  - Prefill on a mesh loses up to 15 % (E = 32, L = 64): the local port, not the column link, is the busiest resource of an all-gather.
  - Rings lose ~⅓ of prefill: one local port ingests twice what a ring link carries.
  - The §250.2 design rule is unchanged for decode. For prefill with a 1-link local port, the mesh optimum moves toward larger E (fewer bytes per port).
- Why opt-in: the port width is a design choice (wider NIU ports remove the port term) 「假设」, and the ≤ 0.306 figures stay reproducible. Default "off" is bit-identical.
- Not modelled: per-op DRAM attribution (see above); reduce-tree nodes with more than one child; the pipe-overlap path (`_pipe_time`).

## §253 Chiplet partitioning in the energy-aware DSE; D2D consistent with the NoC (0.308)

### §253.1 Axes and mechanism 「推导」 (`core/energy_dse.py`)
- `dies` (N per package, 1–16) × `d2d_stds` (catalog grades) × `d2d_units` (per die) × `die_split` ("tp" / "pp" / "ep"). Each design point of the existing grid is a **package**: its MACs, SRAM, SLC and memory stacks (the mem axis id) are split evenly over the N dies.
  - Each die has engines/N (same r × r array), SRAM/N, SLC/N, state SRAM/N and stacks/N (`_mem_split`; mem_cap_GB/N).
  - A point whose engine count or stack count does not divide by N is marked infeasible, not dropped.
- **Cross-die traffic is not a new model.** The replica layout is multiplied along one axis (`_split_layout`), and the existing collective model routes a package's inter-card bytes over the D2D tier (schedule: TP-fastest numbering; `package_cards = N`, `d2d_enabled`):
  - tp: TP × N (MoE: expert TP × N, keeping ep·etp = tp·dp). Per-layer all-reduce / all-gather over D2D.
  - pp: PP × N. Stage-boundary activations over D2D, plus the micro-batch bubble.
  - ep (MoE only): attention DP × N, experts EP × N. Dispatch / combine all-to-all over D2D.
  - D2D time uses the grade's bandwidth (units × catalog figure) and α (`d2d.alpha_us`, 「假设」).
  - D2D energy: with the `d2d_stds` axis, each grade's catalog pJ/bit (UCIe-A 0.25, UCIe-S 0.5, BoW 0.5 「来源」; 48 / 64 GT/s keep the 2.0 target 「假设」, see §253.4) replaces the table's `pJ_bit_d2d`. Without the axis, the table value is used.
- **Package figures.** Throughput and W are per die × N; J/unit and latency are unchanged.
  - The power budget is per package: each die runs under cap / N, so DVFS caps per die.
  - The area budget `die_mm2` limits Σ die area (`si_mm2`, which equals `mm2` for one die). `front3` ranks on Σ die area.
- **Limits** (each labelled; all overridable):
  - Reticle: each die ≤ `reticle_mm2`, default 858 = 26 × 33 mm when the axis is on 「来源」 (0 = off). Stitched dies are not modelled 「近似」.
  - Interposer `pkg_mm2` (number, "cowos-s" 2700 = 3.3× reticle, or "cowos-l" 4719 = 5.5×, TSMC 「来源/推导」): Σ dies + HBM footprints ≤ limit. Footprints: HBM3 / 3E 11 × 11 mm 「近似」, HBM4 11 × 12.8 mm 「二手」. Spacing and passives are not counted 「近似」.
  - Yield `d0_per_cm2`: die yield Y = exp(−A·D0) (Poisson 「假设」; Murphy / negative binomial give higher yield for large dies). Dies are known-good-tested before assembly 「假设」; assembly / bonding yield is not modelled.
    - `kgd_mm2` = N · A / Y(A) is the wafer area per good package; `front_kgd` ranks on it.
    - D0 reference: TSMC N5 ≈ 0.10 /cm² at HVM ramp, N7 0.09 (TSMC symposium / VLSI 2019 via AnandTech 「二手」).
  - D2D shoreline: units × the unit's edge length ≤ D2D edges × die side (`budget.d2d_shoreline`).
- **Verdict** (`chiplet.compare`, `_split_compare`): per design (same package totals), the monolithic row against the best feasible split by throughput and by J/unit. The 1 % tie of `insight.TIE` applies.
  - only_split: the monolithic die is infeasible (reticle / area / power / shoreline) and a split is feasible.
  - split_thr / split_J / split_both: the split is better on that metric.
  - mono: no split beats the monolithic die.

### §253.2 Worked example (1P, qwen3-8b, 8 × HBM3E 8H @ 9.2, n5_derived densities, tpuv4i_7nm energy, 700 W per package, D0 0.1, CoWoS-L; MAC ×1 / ×2 at 128², SRAM ×1; dies 1 / 2 / 4 × UCIe-A 48 / UCIe-S 32 / BoW-256, 4 units)

| case | mono | best split | verdict |
|---|---|---|---|
| decode b64, MAC ×1 | 7875 tok/s, 47.4 mJ, 654 mm² | 2 × UCIe-A: 7839, 45.1 mJ, 2 × 333 | split_J (−4.8 % J, −0.5 % thr) |
| decode b64, MAC ×2 | 908 mm² > reticle | 2 × UCIe-A 7840, 53.9 mJ; 4 dies 51.7 mJ | only_split |
| prefill p2048, MAC ×1 | 70.8 k tok/s, 7.26 mJ | UCIe-A 68.4 k (−3.4 %); UCIe-S 60.5 k (−15 %); BoW 53.1 k (−25 %) | mono |
| prefill, MAC ×2 | > reticle | 2 × UCIe-A 96.2 k, 7.27 mJ | only_split |
| decode, NoC links 32 B/cycle | 382 tok/s, 194 mJ | 2 dies 611 (+60 %), 4 dies 134 mJ (−31 %) | split_both |
| prefill, NoC links 64 B/cycle | 38.7 k, 7.76 mJ | 4 × UCIe-A 60.2 k (+55 %), 7.40 mJ (−4.6 %) | split_both |
| MoE qwen3-30b-a3b decode, ep | 4959, 67.6 mJ | 2 dies 4754 (−4.1 %), 66.3 mJ (−1.9 %) | split_J |
| MoE, tp | 4959, 67.6 mJ | 2 dies 4939 (−0.4 %), 64.1 mJ (−5.3 %) | split_J |
| yield (all cases) | 1258 mm² wafer / good package | 2 dies 929; 4 dies 814 | (front_kgd) |

Reading the example:
- **J/token.** Splitting lowers decode J (−5 %) because each die's NoC is smaller: DRAM and multicast bytes travel fewer hops. This is the existing §249 / §250 NoC energy model, not a D2D effect; D2D energy is ≤ 0.2 % of J at 0.25–0.5 pJ/bit.
  - Caveat 「近似」: the monolithic die is mapped as one NoC domain. A monolithic die with an internal TP-like partition would recover part of this saving. 0.309 (§254): most of the saving was the memory-attach geometry, not the NoC domain; with the shoreline edge rule the verdict becomes mono.
- **Throughput.** Throughput costs come from D2D bandwidth on the TP collectives, which matter in prefill. A grade with 128–256 GB/s per die (BoW-256 / UCIe-S × 4) loses 15–25 %; UCIe-A × 4 (1.5 TB/s) loses 3 %.
  - PP is the wrong split for b1 prefill (the bubble halves throughput). EP vs TP for the MoE: TP keeps throughput, EP loses 4 %.
- **Edge-limited NoC.** When the NoC limits DRAM ingress (§250.1, L = 32–64), splitting adds die edge (2⌊√E⌋ memory-attach links per die). Throughput rises by up to 60 %; this is the 0.305 "narrow links force many small cores" rule applied across dies.
- **Area.** Split dies cost +1–6 % Σ silicon (D2D PHY and the uncore share). Yield more than repays it at D0 0.1 (−26 % to −47 % wafer area per good package).
- Script: `python3 scripts/chiplet_example.py --flat` (0.309; self-contained) reproduces this table; without `--flat` it gives the 0.309 view (§254.2).

### §253.3 D2D with the 0.305–0.307 NoC (die-edge attach) 「推导」 (`core/hardware.py`, `core/scenario.py`, `core/energy.py`, `core/budget.py`, `core/d2d_catalog.py`)
- **Edges** (`d2d_edges`): a die in a two-die package gives one inner edge to D2D (B200 「二手」). With ≥ 3 dies it gives two (a 2 × 2 like MI300X's IODs, or the inner dies of a chain 「二手/假设」).
  - The HBM shoreline check now uses 4 − edges sides; it was always 4 − 1.
  - The NoC's memory columns are the west / east edges (0.305), so the D2D PHYs sit on north / south. That matches 2 HBM sides + up to 2 D2D sides.
  - Not modelled 「近似」: a "tight" HBM case that needs a third edge would conflict with a 2-edge D2D layout and would attach outside the two memory columns.
- **Bandwidth** (`Chip.noc_d2d_links` / `noc_d2d_GBps`; `Scenario.d2d_link`):
  - The D2D PHY attaches to the routers along its edge, one per engine column → ⌈E/R⌉ links per edge (mesh; crossbar like the mesh, as `noc_mem_links`). Ring: one stop per edge, both directions → 2.
  - The usable D2D bandwidth per direction is min(PHY, links × L × f). This is the same rule as DRAM ingress, under the same switch (`noc_mem_attach` "edge"), and only with a link width set.
  - Example: 16 engines / die, L = 32 → 4 × 32 B × 1 GHz = 128 GB/s, so UCIe-A × 4 (1536) is throttled to BoW-256 × 4 (128). The DSE row flags this (`d2d_noc_capped`).
  - DRAM and D2D share interior links; that is not modelled (each is bounded at its own edge) 「近似」.
- **Energy** (`action_counts` "noc_d2d"): every D2D wire byte crosses the sender's NoC from its engine to the edge and the receiver's NoC from the edge to its engine. Each end costs (R + 1)/2 hops and pitches (mesh: the engine row below the D2D edge), ring mean min(i, E − i), crossbar one traversal plus the mesh distance.
  - It is booked on noc / noc_wire like the 0.306 DRAM hops and reported under `energy.noc.d2d`.
- **PHY footprint per grade** (`D2D_PHY_GEOM`; `budget.phy_area`):
  - UCIe-A x64 0.3888 × 1.043 mm, UCIe-S x16 0.5715 × 1.540 mm (UCIe tutorial module table 「来源」).
  - BoW: one TX + one RX slice = 5.2 mm / 8 × 1.35 mm, from the BoW 2.0 §8.11 example link of 8 + 8 slices on organic laminate at 130 µm pitch 「推导」.
  - Shoreline density: UCIe-A 988 GB/s/mm at 48 GT/s (658 at 32 GT/s, the consortium's figure), UCIe-S 112, BoW-256 49.
  - UCIe-A keeps `mm2_per_d2d_unit` (the preset is the same module); UCIe-S and BoW use their own footprint (0.88 / 0.88 mm² per unit, was the UCIe-A 0.41). NVLink-C2C has no published footprint, so none is counted.
- Old → new (1P halves, TP over 2 / 4 dies, HBM3E):
  - D2D PHY area, 4 units, UCIe-S / BoW: 1.62 → 3.52 / 3.51 mm².
  - Shoreline, 4 dies × 4 stacks on a 300 mm² die: tight → over.
  - J/token (prefill, edge attach default): +0.29 % / +0.85 %.
  - D2D bandwidth with L = 32: TP prefill throughput −15.5 % / −14.8 %; L = 128: −4.8 % / −3.4 %.
  - No link width, monolithic, or D2D off: identical.

### §253.4 Freshness and labels
- UCIe 3.0 (Aug 2025) is the latest release; 4.0 is targeted for the end of 2026 and not released.
- SemiEngineering (2026-02-12, citing the consortium): 48 / 64 GT/s keep the bump locations, so the footprint label moves from 「假设」 to 「二手」. Energy target: below 0.5 pJ/bit at the lower rates, 0.75 pJ/bit with the enhanced equalization of the higher rates (package type not stated in the text).
  - The catalog keeps 0.25 for UCIe-A 48 / 64 and notes 0.75 as the upper reference 「二手」.
  - Sensitivity: with 0.75 for UCIe-A 48, prefill J changes by ≤ 0.3 % and no verdict changes.
- Not modelled: stitched / multi-exposure dies; assembly yield; interposer cost; D2D link power states; heterogeneous dies (I/O die + compute dies); 3D stacking (UCIe-3D).

## §254 Fair monolithic-vs-split comparison: on-die NoC clusters, memory attach on the HBM edges (0.309)

### §254.1 Mechanism 「推导」 (`core/hardware.py`, `core/scenario.py`, `core/energy.py`, `core/budget.py`, `core/energy_dse.py`)
- **The 0.308 bias.** In the chiplet DSE, the monolithic die was one flat NoC domain with two memory columns. Each split die got its own smaller NoC and its own memory edges. Two things were confounded: the NoC partition (which clustering can give a monolithic die too) and the memory-attach geometry.
- **On-die NoC clusters** (`Chip.noc_clusters = K`, 1–64; hierarchical NoC). A monolithic die is K clusters of engines/K, each with its own mesh / ring, joined by an on-die global tier.
  - Precedent: A100 / H100, "The L2 cache is divided into two partitions … Each L2 partition localizes and caches data for memory accesses from SMs in the GPCs directly connected to the partition" (NVIDIA A100 / H100 whitepapers 「来源」).
  - Representation: the chip object is one cluster, and the scenario is K cards in one package (`package_cards = K`, `d2d_enabled`; validated). The cluster tier replaces the D2D tier, so cross-cluster TP / PP / EP bytes use the existing collective model. No new traffic model.
  - Bandwidth (`Scenario.d2d_link`): the cluster's boundary attach links × `noc_link_Bpc` × clock (`noc_d2d_GBps(d2d_edges(K))`, the 0.308 edge rule). Without a link width it is unlimited (`CLUSTER_UNLIMITED_GBps` = 1e9). α is the D2D α 「假设」: synchronisation-dominated, not PHY-dominated.
  - Energy (`action_counts` "noc_d2d", `on_die`): each byte crosses half of each cluster's NoC, as 0.308 D2D bytes do. It also pays +½ hop and +½ pitch per end, so the boundary crossing costs 1 router + 1 link pitch (`pJ_bit_hop_noc`, `pJ_bit_mm_onchip`, Keckler 2011 「来源」). No D2D PHY energy.
  - Area (`budget.whole_die`, `full_area`, `phy_area`, `shoreline_check`): the whole die (K × the cluster's engines / SRAM / SLC / stacks, one card), so area equals the flat die (654 mm² for 1P). No D2D PHY area. `d2d_shoreline` is None. Yield and reticle come from the whole die.
  - DRAM: `noc_mem_hops` and `noc_mem_links` are the whole die's (the memory columns are the die's edges); a cluster gets links / K.
- **Memory attach on the HBM edges** (`Chip.noc_mem_edges` 2 / 3 / 4; `budget.hbm_mem_edges`).
  - The 0.305 model attaches memory on the west / east columns (2·⌊√E⌋ links). A die whose stacks need a 3rd / 4th edge (shoreline "tight") has its HBM PHYs on those edges too, so the NoC attaches there as well.
  - Each extra edge adds one memory row (north, then south) with C = ⌈E/R⌉ attach routers. Links = 2R + (m − 2)·C (ring: 2m stops).
  - DRAM hops: mean over the R × C grid of the distance to the nearest attached edge. For m = 2 the formula is unchanged. Ring: m stops; crossbar: one traversal + the same wire.
  - In the chiplet DSE (`mem_edges_shoreline=True`), m = ⌈stacks × 11 mm / side⌉ clamped to [2, 4 − D2D edges]. Monolithic and split dies use the same rule. Example: 1P mono, 8 stacks on 654 mm² (side 25.6, need 88 mm) → 4; 2-die halves, 4 stacks on 333 mm² (one D2D edge) → 3; 4-die quarters → 2.
  - Elsewhere the default is 2, so every non-chiplet path is bit-identical. This closes the §253.3 "not modelled" note (a tight die used only two memory columns).
- **Verdicts** (`_split_compare`): the best split by throughput / J is compared with the best monolithic row (flat or clustered) by the same metric (`mono_thr`, `mono_J`). `verdict_flat`, `d_thr_flat` and `d_J_flat` compare against the flat monolithic row only (the 0.308 semantics, under the same memory-edge rule). `mono_clusters=False, mem_edges_shoreline=False` reproduces 0.308 row for row.
- **DSE rows**: with the dies axis on, each K > 1 in the axis adds a monolithic row with K clusters (`clusters`; no `d2d_std`; split axis = `die_split`; no catalog pJ override). Every chiplet row carries `mem_edges`.

### §254.2 Re-derived verdicts, old → new (the §253.2 set; MAC ×1; `scripts/chiplet_example.py` vs `--flat`)

| case | 0.308 (flat mono, 2 memory columns) | edge rule only (`verdict_flat`) | 0.309 (best of flat / clustered) |
|---|---|---|---|
| decode b64 | split_J: mono 7875 / 47.39 mJ; 2 × UCIe-A −0.5 % thr, −4.8 % J | mono: mono 45.11 mJ, best split 44.75 (−0.8 %) | mono |
| prefill p2048 | mono (−3.4 % thr) | mono | mono (2 clusters 70.3 k, −0.8 %; 4 clusters −2.0 %) |
| decode, L = 32 | split_both: mono 382 tok/s / 194 mJ; split +60 %, −31 % | mono: mono 917 / 101.8 mJ; 2 dies 917 / 101.4 (−0.3 %) | mono |
| prefill, L = 64 | split_both: mono 38.7 k; 4 × UCIe-A 60.2 k (+55 %), −4.6 % J | split_both: mono 39.5 k; split +52.5 %, −4.0 % J | mono: 4 clusters 60.9 k / 7.40 mJ; split −1.1 % thr, +0.1 % J |
| MoE ep | split_J (−4.1 % / −1.9 %) | mono (−4.1 % / +2.7 %) | mono |
| MoE tp | split_J (−0.4 % / −5.3 %) | mono (−0.4 % / −0.8 %) | mono |
| prefill, UCIe-A 0.75 pJ/bit | mono | mono | mono |
| MAC ×2 (908 mm²) | only_split | only_split | only_split (clustered mono is also > reticle) |

Reading:
- **Where the 0.308 split advantage came from.** Decomposition, decode L = 0 (0.308 geometry): flat mono, mono with 2 clusters and 2 split dies have noc + wire 7.16 / 7.17 / 4.84 mJ. The DRAM-hop part is 6.74 / 6.74 / 4.43 mJ (hops / B 2.29 / 2.29 / 1.50).
  - So clustering alone recovers nothing. The saving was the memory-attach distance: two dies had 2.8 S of memory edge, the flat die only 2 S, while its 8 stacks actually occupy all four edges.
  - The L = 32 throughput gap (+60 %) has the same cause: DRAM ingress links 10 vs 2 × 8. With the stacks' edges attached (24 links), the mono die reaches 917 tok/s.
- **What clustering does recover.** The NoC-domain effect, which shows when a narrow link width makes the single-domain on-die traffic congest (prefill L = 64). Clusters give the mono die the same partitioned dataflow without a D2D bandwidth cap. The best clustered mono (K = 4) is 1.1 % faster than the best split.
  - At L = 0 clustering only adds cluster-tier syncs (−0.8 % / −2 % thr), so the flat die stays best. This is why the verdict takes the best of both.
- **What splitting still buys** (unchanged by 0.309): yield (wafer area per good package 1258 → 929 / 814 mm² at D0 0.1) and the reticle (MAC ×2 only as 2 or 4 dies). Splitting costs +1–6 % Σ silicon and the D2D bandwidth in prefill (BoW / UCIe-S −15 to −25 %).
- **Labels.** The memory-edge rule is 「推导」 from the shoreline check (itself 「近似」: square die, 11 mm per stack). The cluster tier's α 「假设」. Cluster boundary cost 1 hop + 1 pitch 「推导」 from the existing per-hop / per-mm energies 「来源」.

### §254.3 Example script
- `scripts/chiplet_example.py` is self-contained: inline energy table (`tpuv4i_7nm` preset fields without idle / TDP), `n5_derived` densities, 700 W, and the 7 cases of §253.2.
  - `--flat` gives the 0.308 view; it reproduces the 0.308 example output row for row.
  - The default is the 0.309 view, which prints `verdict` and `verdict_flat`. `--case N` runs one case.
  - The UCIe-A 0.75 pJ/bit sensitivity case restores the catalog afterwards.
- Runtime ≈ 1.3 s. It replaces the box-only /workspace/ex_0308.py + /tmp/a308/t1.py.

### §254.4 Freshness
- Model-release sweep (`scripts/freshness_sweep.py 2026-10-21`): identical to the 0.308 sweep.
- UCIe: 3.0 (2025-08-05) is still the latest released specification. 4.0 is targeted for end-2026 (power reduction / AVS, GUC CTO post 2026-05 「二手」), not released. The catalog is unchanged.
- Not modelled: a clustered die's global tier as a separate topology (it is the K-card collective over the boundary links); memory-attach on non-square dies; interior links shared between DRAM and cluster / D2D traffic.

## §255 Memory-attach edges as the mechanism default; NoC-cluster axis; chiplet economics (0.310)

### §255.1 Memory edges everywhere 「推导」 (`core/hardware.py`, `core/scenario.py`, `core/budget.py`, `core/energy_dse.py`)
- **The inconsistency.** In 0.309 the shoreline rule (§254.1) ran only inside the chiplet DSE. Single eval, the plain energy DSE, the insight panels and the PD pools kept two memory columns. The same chip + memory therefore had different DRAM hops and ingress depending on the panel.
- **Mechanism.** `Chip.noc_mem_edges` gains 0 = auto as the default (2 / 3 / 4 stay fixed). `Scenario.__post_init__` resolves auto into the internal `noc_mem_edges_auto` (never serialised or accepted) with `budget.mem_edges_auto`. `Chip.mem_edges` is the value in effect.
  - Every Scenario construction or replace re-resolves the value, so every path that builds a scenario sees it. This covers PD prefill pools with their own memory or chip, insight what-ifs, the DSE and single eval.
  - A bare Chip outside a Scenario uses 2.
- **Rule.** m = ⌈edge needed / die side⌉, clamped to [2, 4 − D2D edges]. When the interface does not fit on all four sides ("over") the result is 2, because that point is flagged infeasible elsewhere.
  - HBM: stacks × 11 mm (`shoreline_check`, 「近似」).
  - LPDDR (new): bus bits ÷ 18.75 pins / mm. Credo's Weaver brief gives 0.18 Tb/s per mm for conventional LPDDR5X 「二手」; at the brief's 9.6 GT/s PHY that is 18.75 data pins / mm 「推导」. Pins rather than rate set the edge, so the figure is used for all LPDDR generations 「近似」. Cross-check: Grace's ≈ 480-bit LPDDR5X needs 25.6 mm, about one edge of a large die.
  - DDR / GDDR: no figure, so 2 edges.
- **Die side** (`mem_edges_auto`):
  - A preset's published die area is used while the chip keeps the preset's area-relevant geometry (`hardware.PRESET_DIE_MM2`, `preset_die_mm2`): H100-like 814 mm² (NVIDIA H100 whitepaper 「来源」). The density proxy says 283 mm² there, which would wrongly give 4 edges; with 814 the rule gives the real H100's 2 HBM edges.
  - Otherwise: the die-area proxy `full_area` with the n5_derived densities (`AUTO_EDGE_DENSITY`; 5 nm-class 「近似」 for other nodes). Cached per (chip, memory, package, layout).
  - Energy-DSE rows: the search's own densities, so the edges match the reported `mm2`.
- **Energy DSE.**
  - Auto chip: every row resolves from its own die. HBM + LPDDR rule.
  - Fixed chip: the value is kept, except that the chiplet comparison applies its 0.309 HBM-only rule (`mem_edges_shoreline=True`). `False` forces 2 (the 0.308 view).
  - Every row reports `mem_edges`.
- **Switch.** `chip.noc_mem_edges = 2` (chip panel 外存接入边 = 2) reproduces 0.309 on every path, including the chiplet DSE. The 0.309 example output is identical in both views.
- **Old → new (auto vs fixed 2; tpuv4i_7nm energy; qwen3-8b, decode b64 / prefill p2048):**

| chip + memory | edges | decode L = 0 | decode, L = 32 B/cycle | prefill |
|---|---|---|---|---|
| 1P + 4 × HBM3E (default), H100-like + 5 × HBM3 (default), 100T + LPDDR5X 4 × 64 (default) | 2 → 2 | unchanged | unchanged | unchanged |
| 1P + 8 × HBM3E | 2 → 4 | J 47.39 → 45.11 mJ (−4.8 %) | 382 → 917 tok/s (+140 %), J −47.6 % | J −0.3 % (L = 32: thr +2.9 %) |
| 100T + LPDDR5X 8 × 64 | 2 → 3 | J −0.4 % | 305 → 455 (+49 %), J −10.2 % | ≤ 0.5 % |
| 100T + 4 × HBM3E | 2 → 4 | J −1.2 % | 305 → 609 (+100 %), J −21.9 % | ≤ 0.8 % |

  Script: /tmp/a310/oldnew.py (box), table in NOTES_0310.

### §255.2 NoC clusters in the energy DSE; chip panel
- `energy_design_search(clusters=[K…])` (1–5 values, 1–64; API `clusters`; UI NoC 簇): a monolithic die searched as K on-die NoC clusters (the 0.309 mechanism, split along `die_split`), with or without the dies axis. With the dies axis it replaces the K set taken from it. Rows carry `clusters`; area / yield / cost are the whole die.
- Example: 1P, 8 × HBM3E, prefill, links 64 B/cycle: K = 1 / 2 / 4 → 39.5 k / 48.0 k / 60.9 k tok/s, 7.71 / 7.49 / 7.40 mJ. With unlimited links clustering costs −0.8 % / −2 % (§254.2), so it is an axis, not a default.
- **Chip panel: not offered.** A clustered chip only exists as K cards of one package (`package_cards = K`, d2d tier = cluster tier, layout × K along the split axis, memory / K). The chip panel's chip is the whole die, so offering clusters there would mean rewriting layout / memory / package behind the user's entries. The DSE does exactly that transform and reports the whole-die figures. The scenario API keeps it reachable for experts (`chip.noc_clusters` with the matching package).

### §255.3 Chiplet economics (`energy_dse.COST_PRESETS`, `package_cost`, `cost=`)
- Cost per good package = (N · good-die cost + packaging + test / assembly) / assembly yield. HBM is excluded because it is the same in every point of one memory axis value; with `mem_usd_per_GB` the memory $ is already a separate column. Terms:
  - Die: wafer $ / DPW, with DPW = π R²/A − π · 2R / √(2A), R = 150 mm (standard gross-die approximation 「来源」; scribe lanes / edge exclusion not modelled 「近似」). Wafer: TSMC N5 $16,988 (CSET 2020 "AI Chips" foundry sale-price model 「二手」). A 2025 N4/N5 price of ≈ $18–20 k (Morgan Stanley via heise 「二手」) is a user override (`cost.wafer_usd`).
  - Good die = die / Y. The Poisson yield is the 0.308 term (D0; Y = 1 without D0). Dies are known-good-tested, so bad dies are not packaged.
  - Packaging per mm² of the package area (Σ dies + HBM footprints, the 0.308 `pkg_mm2`) 「推导」:
    - CoWoS-S: $750 per H100 (Silicon Analysts 2026 「二手」) ÷ (814 + 6 × 121) mm² → $0.49 / mm².
    - CoWoS-L: $1,100 per B200 ÷ (2 × 800 + 8 × 121) mm² → $0.43 / mm².
    - The interposer yield is inside these figures.
  - Test / assembly: $920 per package (H100, same source 「二手」). Multi-die packages pay +22.5 %: KGD testing adds 15–30 % to test cost (same source; midpoint 「二手」). Clusters are one die, so they pay no extra.
  - Assembly yield 1.0: bonding loss of good dies is not modelled 「假设」 (`cost.asm_yield` override). 0.319: replaced by the per-component model (§264); `asm_model: "flat"` keeps this rule.
- Rows: `dpw`, `good_die_usd`, `si_usd`, `pkg_usd`, `test_usd`, `usd_good_pkg`. `front_cost` = non-dominated on (throughput ↑, J ↓, cost ↓). The UI shows a $/好封装 column (★ on the cost front).
- **Result** (1P, qwen3-8b decode b64, 8 × HBM3E, CoWoS-L, MAC × 1; `scripts/chiplet_example.py --cost n5_cowos_l --d0 …`):

| D0 /cm² | monolithic 654 mm² (flat = clustered) | 2 × 333 | 4 × 171 |
|---|---|---|---|
| 0.05 | $1902 | $2055 | $2040 |
| 0.10 | $2013 (si 399 + pkg 695 + test 920) | $2097 (+4.2 %) | $2059 (+2.3 %; si 223 + pkg 708 + test 1127) |
| 0.20 | $2382 | $2203 | $2100 (−11.8 %) |
| 0.30 | $3090 | $2352 | $2150 |

  - Break-even D0 vs the monolithic die: ≈ 0.12 /cm² for 4 dies and 0.14 for 2 dies. Over the KGD +15–30 % range it is 0.09–0.14 (4 dies) and 0.11–0.17 (2 dies); CoWoS-S gives the same within 0.001.
  - At the N5 HVM reference D0 ≈ 0.10 (§253.1) the yield saving of splitting ($176) is smaller than the extra KGD test ($207) and package area. Splitting pays only on an immature process (D0 ≳ 0.12) or beyond the reticle: MAC × 2 is split-only, 4 dies $2278 vs 2 dies $2375.
  - With §254 (no J / throughput advantage once the comparison is fair), cost and reticle are the remaining split criteria, and cost only above that D0.
- Not modelled: NRE / mask sets (amortisation dominates at low volume and favours reusing one small die), interposer cost as a function of the die count (bridges per die edge), wafer-edge exclusion, the cost of D2D PHY yield.

### §255.4 Freshness
- Model-release sweep 2026-10-21, new since 0.309: inclusionAI/Ling-3.0-tiny-UBTree (created 2026-10-21 13:49 UTC = 22:49 JST). It is a 183 M-parameter tree-speculation draft head (DFlash anchor + UBTree factorizer, 5 layers) for Ling-3.0-tiny. (Corrected in §256.6: the targets build with the existing bailing_hybrid builder.) The target's `bailing_hybrid` architecture is not in the catalog, so it was not added (deferred with the target).
- UCIe: 4.0 still unreleased (§254.4). CoWoS / wafer cost sources are 2020–2026 estimates, labelled 「二手」.

## §256 Audit of the 0.310 memory-edge default: host-I/O edge, per-edge interleave; robust headless smoke; Ling-3.0 + UBTree (0.311)

### §256.1 Which metric was +49 %, and why
- Case: 100T + LPDDR5X 8 × 64 (8533 MT/s), qwen3-8b decode batch 64, prompt 2048 / ctx 4096, `noc_link_Bpc = 32`, energy table tpuv4i_7nm.
- **The +49 % is decode throughput per card**: 304.65 → 454.92 tok/s (+49.3 %). J/token went the other way, 70.052 → 62.913 mJ (−10.2 %).
- Mechanism: 100T has 16 engines on a 4 × 4 mesh (R = C = 4) with 1 GHz × 32 B links. Memory attached on the W / E columns enters through 2R = 8 links (256 GB/s). A third (N) edge adds C = 4 → 12 links (384 GB/s). LPDDR5X 8 × 64 delivers ≈ 546 GB/s peak, so with 8 links decode is NoC-ingress-bound. With 12 links it reaches the DRAM roofline, and 454.92 equals the L = 0 value (no NoC limit). Of the J gain only −0.4 % comes from the edge count itself (the L = 0 J moves 64.20 → 63.95 mJ, shorter attach paths); the rest is static / idle energy spread over 49 % more tokens per second 「推导」.

### §256.2 Is a third LPDDR edge physical?
- Die: 100T keeps no published area, so the side comes from the n5_derived proxy, 107.9 mm² → 10.39 mm 「近似」.
- Need: 512 bus bits ÷ 18.75 pins / mm (§255.1 「二手/推导」) = 27.3 mm = 2.63 edges. **Two edges (0.309) cannot hold this bus, so the third edge is real.** Balanced across 3 edges it is 9.1 mm per edge (< 10.39).
- What was wrong in 0.310: (a) the clamp was 4 − D2D edges, so a single die could give all four edges to memory and keep none for host I/O (PCIe / NVLink-class SerDes). Shipping parts keep at least one: GH100, B200 and MI300X put HBM on two (or the long) edges and D2D / PCIe / NVLink on the rest (0.212 note 「二手」). (b) The attach links of all memory edges were pooled, as if every unit could feed every edge's routers. With address interleave over all units, each edge carries traffic in proportion to the units it holds, so the most-loaded edge per link sets the rate.
- Fix A1, host-I/O edge: `Chip.noc_mem_io_edges` (default 1 「假设」, 0 allowed). `budget.mem_sides_max(scn) = max(2, 4 − D2D edges − io edges)`. Shoreline checks and warnings use it ("> 可用 N 边 X mm").
- Fix A2, published geometry: `hardware.PRESET_DIE_EDGE_MM` (H100-like 33 mm long edge: 6 HBM sites, 3 per long side × 11 mm 「推导」; short edge 814 / 33 = 24.7 mm). `die_edges(chip, mm2)` returns the long edges first, as a rectangle only while the chip keeps the preset area. Every other die is square.
- Fix A3, interleave: `budget.mem_edge_units(spec)` = HBM stacks × 11 mm, or LPDDR channels (x16; LPDDR6 x24) × bits / 18.75 mm. `mem_edge_load` water-fills the units onto the memory edges in proportion to each edge's links (W, E = R; N, S = C; ring 2 per stop), capped by each edge's length. Then `Chip.noc_mem_links() = min_e links_e · N / u_e / k` for `noc_mem_balance = "interleave"` (default); "pooled" = Σ links_e / k (0.310). Interleave ≤ pooled, and they are equal when the load is proportional.
- Result for the audit case: edges 3, load (11, 11, 10) channels, links 12 → 11.636. Throughput 443.12 tok/s, J 63.285 mJ. **vs 0.309: throughput +45.4 %, J −9.7 %.** The I/O edge does not bind here (3 ≤ 4 − 1).

### §256.3 Broad old → new (840 cases, /tmp/a311/broad3.py on the box)
- Grid: 7 presets × 14 memories (HBM3E 1 / 2 / 4 / 5 / 6 / 8 / 12 stacks, HBM3 5, HBM4 8, LPDDR5X 2 / 4 / 6 / 8 / 9 × 64) × L ∈ {0, 16, 32, 64} × {qwen3-8b decode b64, qwen3-8b prefill b1, gpt-oss-120b decode b16}.
- 0.310 → 0.311: 240 cases lower throughput, 0 higher. 0.309 → 0.311: 99 lower, 108 higher. Default presets with default memory at L = 0: unchanged.
- Outliers, each checked against the mechanism:
  - 1P / 1P-lp4x64 + 8 × HBM3E and 100T + 4 / 5 × HBM3E: 0.310 gave 4 edges; now 2 (3 edges cannot hold the stacks either, so this is shoreline "over" and kept at 2, the 0.309 value). L16 / L32 throughput −58 % vs 0.310. **The 0.310 +140 % for 1P + 8 × HBM3E was a die with no host I/O.**
  - One HBM stack: the stack sits on one edge, so links = R instead of 2R. −50 % at L32 vs pooled (1P: 382 → 191 tok/s). This is physical: one stack's PHY is on one edge.
  - H100-like + 8 × HBM3E: 3 edges, load (3, 3, 2), L32 836.2 → 743.3 tok/s (−11 % vs 0.310, +33 % vs 0.309). H100-like + 6 stacks: 2 edges, as on GH100.
- Switches: `noc_mem_io_edges = 0` + `noc_mem_balance = "pooled"` reproduces 0.310, except H100-like with ≥ 6 stacks (now the 33 mm edge). Adding `noc_mem_edges = 2` reproduces 0.309.

### §256.4 Chiplet consequence (§255.3 revisited)
- `scripts/chiplet_example.py --cost n5_cowos_l --d0 0.1` (new `--io-edges`, default 1): 1P + 8 × HBM3E monolithic 654 mm² needs 88 mm of HBM edge but has 3 usable edges = 77 mm ("over"). 2 × 333 mm² (1 D2D + 1 I/O edge kept) have 37 mm for 44 mm ("over"). Only the 4 × 171 mm² split is buildable (memE 2, $2059).
- So the 0.310 comparison of monolithic $2013 vs 4-die $2059 at D0 0.1 priced a die that cannot carry 8 stacks plus host I/O. With 8 stacks, splitting is required by shoreline, not chosen on cost; the cost comparison stands only while the stacks fit the usable edges (1P monolithic: 3 edges = 77 mm, ≤ 7 stacks). `--io-edges 0` gives the 0.310 view.

### §256.5 ui_headless_smoke robustness
- Hang cause (most likely): the server ran with `stderr=PIPE` that nobody read. Under concurrent load a chatty server fills the pipe and blocks, and the page waits then never resolve. Now stderr goes to a temp log, kept on failure.
- Every page wait goes through `wait_js(expr, step, timeout)` / `wait_sel` (11 + 2 sites). A timeout raises `StepTimeout` naming the step, the expression and the panel status text it watched. `_Guard` turns any browser-part exception into one "ABORT after step '…'" error with the server state and the log tail. `_Steps` prints `STEP [elapsed]` as each step completes, so a hang shows where it stopped. A global watchdog (`--budget`, 1800 s) kills a stuck run with the same context (exit 3).
- Every assertion is unchanged: same steps and same checked strings and values.

### §256.6 Freshness: Ling-3.0 targets + UBTree
- inclusionAI/Ling-3.0-tiny and Ling-3.0-flash have public configs (fetched 2026-10-22). Both are `bailing_hybrid` (KDA linear attention + MLA full attention, MoE), the builder already used for Ling-3.1-flash. **The §255.4 note that the architecture was not in the catalog was wrong.** Built at full coverage, param_err 0: tiny 7.89 B total / 1.38 B active (24 layers, h 1536), flash 124.4 B / 5.51 B (42 layers, h 2560).
- inclusionAI/Ling-3.0-tiny-UBTree (rev 05abc072…), `ling-3.0-tiny-ubtree` (target + draft):
  - Backbone (DFlash, config + safetensors header 「来源」): fc [1536, 7680] over the hidden states of target layers [4, 8, 12, 16, 22] → hidden_norm → 5 Qwen3 layers (16 / 16 heads × 128, SwiGLU 4608) → norm. Block 16 including the anchor row. It uses the target's embedding and lm_head.
  - Selector: target lm_head on the draft rows → top-K = 64 candidates per row; hidden_projection 256 × 1536 with per-depth affine scales (2 × 15); predecessor / successor MLPs 1536 → 512 → 256 on the candidates' target-embedding rows; bigram edges K × K × 256 between adjacent rows; pruned tree, budget 64 non-root nodes (the card's benchmarks use 49).
  - Params 183,127,326 exactly (layers 169,099,520 + fc / norms / selector 14,027,806).
  - Model: `ModelSpec.draft_ubtree = (K, rank, width, depths, budget, nodes)`. IR ops: hidden_proj GEMM, candidate embed gather, 4 factorizer GEMMs over n · K rows + SiLU, K × K × R edge scores per row, tree-build vector pass. The target verifies q = 1 + nodes (default 50; `serving.spec_draft_tokens` in [1 + k, 64]). The block is fixed at 16 「假设」; another spec_k is warned and set to 15.
  - Acceptance: card AL with the 49-node tree, greedy — GSM8K 9.53, MATH-500 9.86, AIME 2025 8.44, HumanEval 10.04, MBPP 9.35, LiveCodeBench 7.46; mean τ = 9.113 「来源/推导」 over 15 draft depths → `spec_published`.
  - The SGLang serving code is not released, so the selector / tree kernels and the non-causal block attention are 「假设」 (warned).
  - Example: 1P + 4 × HBM3E, batch 1, ctx 4096: TPOT 0.626 → 0.556 ms with the published τ (default `spec_accept` 0.7 → 1.53 ms: the 50-token tree verify only pays off at the card's acceptance).


## §257 Shoreline feasibility as a first-class result; sized I/O edge; per-part scale-up link (0.312)

### §257.1 The report
`budget.shoreline_report(scn, die_mm2=None)` places memory, D2D and I/O on the four edges of one die and returns the per-edge ledger (length, whole memory units, mem / d2d / io mm, used / free, role), `feasible`, `status` (ok; tight = memory beyond two edges; over; unknown when the die area is unknown) and readable reasons.
- Every evaluation carries it (`Result.shoreline`, `/api/eval`). Every energy-DSE row carries it (`buildable`, `shoreline`, `shore`).
- An infeasible row is excluded from all fronts and verdicts unless `include_unbuildable=True`.
- The legacy modes keep their own rules: `noc_mem_io_edges = 1` (0.311, whole I/O edge, I/O "reserved" and not checked) and `0` (0.310, no I/O).

### §257.2 Sizing the I/O edge from lanes
need = (host + scale-up lanes) × lane edge, counted in x4 macros that are never split across edges 「假设」.
- Host: PCIe x16 per package 「来源」 (H100 datasheet), split over the dies.
- Scale-up: GB/s per direction × 8 ÷ 100 Gb/s per lane 「假设」, only when the replica leaves the package (the c2c-PHY rule).
- Lane edge 0.285 mm 「来源」 (Intel 32 Gb/s PCIe5 transceiver, 10 nm, CICC 2020: "die edge usage per lane of 285 µm"). The same pitch is used for 112G long reach 「假设」 because no public figure exists; 112G XSR 0.8–2 Tb/s/mm is the floor 「二手」.
- Check: H100-like at tp8 with 450 GB/s gives 16 + 36 lanes (NVLink4 18 links × 2 lanes 「二手」) = 14.82 mm. That fits beside 5–6 HBM sites on 2 × 33 + 2 × 24.7 mm.

### §257.3 Discrete fit and die shape (sized mode)
- Each edge holds ⌊len / unit⌋ whole stacks (11 mm) or channels (16 / 18.75 mm).
- A die without published geometry may be any rectangle of its area in the 26 × 33 mm field with aspect ≤ 2 (`DIE_ASPECT_MAX` 「假设」; GH100 1.34 and B200-class dies 1.2–1.4 「二手」), in 0.25 mm steps. The squarest shape that fits is chosen.
- D2D goes on the short or the long edges. Memory goes on the fewest non-D2D edges (water-filled by attach links). I/O is placed first-fit in what is left.
- Examples:
  - 1P + 8 × HBM3E (654 mm², 25.6 mm square) holds 2 stacks per edge on all 4 edges, with 14.3 mm left for I/O (4.56 mm needed at tp1).
  - At tp8 / 400 GB/s the I/O needs 13.7 mm: buildable.
  - At tp8 / 900 GB/s the I/O needs 25.1 mm against 16.7 mm left: over.
  - 2 × 333 mm² dies reshape to 22.2 × 15.0 mm and hold 4 stacks each.
  - 100T cannot hold 8 stacks.

### §257.4 Per-part scale-up link
A split or clustered package keeps its scale-up bandwidth: each part gets Link.GBps / parts. Through 0.311 every part had the whole link, i.e. K × the package's off-package bandwidth. The sized I/O check exposed this: K = 4 would have needed 288 lanes. 1P + 8 × HBM3E prefill L64 at tp8 / 900: K = 1 / 2 / 4 = 16.5k / 17.6k / 23.3k tok/s (was 16.5k / 19.2k / 27.7k). tp1 is unchanged (39.4k / 47.8k / 60.6k).

### §257.5 Old → new and buildable-only verdicts
- 840 cases, 0.311 switch → 0.312 default: 45 change, all up.
  - 1P / 1P-lp4x64 + 8 × HBM3E / HBM4: memory edges 2 → 4, L16 / L32 throughput +100 %, J −42…−47 %.
  - 100T + 4 × HBM3E: 2 → 4 edges.
  - Buildable at tp1 / 400 (70 chip × memory): ok / tight / over = 42 / 13 / 15 → 41 / 18 / 11. (Corrected in §258.5: the shipped 0.312 code gives 42 / 18 / 10.)
- Chiplet economics (qwen3-8b decode, n5_cowos_l, buildable designs only). D0 from which the split is cheaper:

| memory | tp1/400 MAC×1 / ×2 | tp8/400 | tp8/900 |
|---|---|---|---|
| 4 × HBM3E | 0.16 / 0.08 | 0.16 / 0.08 | 0.16 / 0.08 |
| 6 × HBM3E | 0.16 / mono > reticle | 0.16 / mono > reticle | 0.16 / mono > reticle |
| 8 × HBM3E | 0.12 / mono > reticle | 0.13 / mono > reticle | mono unbuildable (I/O) / mono > reticle |

- Below the break-even D0, the monolithic die is buildable and cheaper (e.g. 8 × tp1 at D0 0.1: $2013 vs 4-die $2059).
- 0.311's "only_split" for the 8-stack example was an artefact of the whole reserved I/O edge. All 7 example cases are "mono" at MAC×1 again; the I/O is now checked.
- The one physically unbuildable monolithic case is a fast scale-up domain (900 GB/s at tp8) on an 8-stack die. There the split is forced by I/O shoreline, not by yield.

### §257.6 test_core_238 and freshness
- test_core_238 measures its bound in CPU time (`time.process_time`) with the same 60 s limit. Alone it takes 17.5 / 10.9 / 17.0 s per case, and box load no longer counts.
- Freshness, 2026-10-22: HF since 2026-10-21 has nothing new (only Ling-3.0-tiny-UBTree, already in). UCIe 4.0 is unpublished. JEDEC has no HBM4E.

## §258 Adversarial review of 0.310–0.312; thermal / power buildability (0.313)

### §258.1 Review method
These versions flipped the monolithic-vs-split verdict twice: 0.310 mono, 0.311 only_split, 0.312 mono. The review asked, for each shoreline input, whether it is sourced and which way an error would move the verdict:
1. Scale-up lane rate.
2. Die edge per lane.
3. Memory on more than two edges.
4. Interposer area.
5. HBM stack and PHY widths.
6. The H100-like rectangle.
7. The memory-edge and scale-up split accounting (Link / parts, cap ÷ dies: both checked consistent).

### §258.2 Scale-up lanes: rate and edge per lane
- **Lane rate.** The shipping per-direction pair rates are NVLink3 50 G (A100: 12 links × 4 pairs = 300 GB/s), NVLink4 100 G (H100: 18 × 2 = 450 GB/s) and NVLink5 200 G (B200: 18 × 2 × 224G PAM4 = 900 GB/s), all 「来源」 (NVIDIA whitepapers). 0.312 used 100 G for every link, which doubles the lane count of a 900 GB/s link against its only shipping precedent.
- **Edge per lane.** The edge grows with rate. PCIe5 32 G is 0.285 mm (Intel CICC 2020 「来源」). 224G is about 0.5 mm per duplex lane: "~4 mm per Rx/Tx 8-lane port" and 128 mm for 32 ports × 8 lanes (OIF 448G Signaling for AI workshop, MaxLinear, April 2025, slides 14–15 「来源」, approximate). 50 / 112 G: no public figure, so 0.285 「假设」.
- **New defaults.** `io_su_lane_Gbps` = 200 and `io_su_lane_mm` = 0, meaning by rate (`SU_LANE_MM`). The H100-like presets are pinned to 100.
- **Sensitivity** (1P + 8 × HBM3E, tp8, plain monolithic die, from chiplet_example.py --lane). I/O need in mm, against the free edge (ok / over):

| link GB/s/dir | 50 G (0.285) | 100 G (0.285) | 200 G (0.5) | free edge |
|---|---|---|---|---|
| 400 | 22.8 over | 13.7 ok | 12.6 ok | 15.4 |
| 450 | 25.1 over | 14.8 ok | 14.6 ok | 15.5 |
| 900 | 45.6 over | 25.1 over | 22.6 over | 16.7 |

- Per Tb/s, 200 G at 0.5 mm (2.5 mm) and 100 G at 0.285 mm (2.85 mm) are close, so the 0.312 verdicts hold, with the 900 GB/s case now on sourced numbers.

### §258.3 Memory on more than two edges
- Since 0.310 a die may carry memory on all non-D2D edges (status tight, 「假设」 geometric limit). No shipping die does: GH100, B200 and MI300X IODs use the long edges only 「二手」.
- The 6- and 8-stack monolithic verdicts depend on this. With `shore_strict`:
  - 6- and 8-stack monolithic dies are unbuildable in every deployment (split only).
  - The 7 example cases at MAC×1 are only_split.
  - The 4-stack break-evens are unchanged: 0.16 (MAC×1) and 0.08 (MAC×2).
- Default off; the verdict tables state both readings.

### §258.4 Interposer box; stack and PHY widths
- **Bounding box.** The interposer must hold the bounding box of the die and its placed stacks: (short + depths W/E) × (long + depths N/S), with depth 11 mm for HBM3/3E (Micron 11 × 11 mm 「来源」) and 12.8 mm for HBM4 「二手」. Σ areas understates this for a 3–4-sided row: 1P + 8 × HBM3E is 2264 vs 1622 mm², H100-like + 8 is 2053 vs 1782. Used for single-die packages in the energy DSE (max with Σ). No verdict changes (all < CoWoS-S 2700 mm²). Multi-die packages keep Σ 「近似」.
- **PHY width.** The host-side HBM PHY is about 8 mm wide (HBM3 8 × 3, HBM4 8 × 4 mm; Samsung via inlevel9 「二手」), narrower than the 11 mm stack. The stack sets the occupied edge, because I/O cannot face a stack.
- **H100-like rectangle.** GH100's outline is not published, so its 33 mm long edge (3 × 11) is derived from that same rule and is not independent evidence.

### §258.5 Corrections to 0.312 documentation
- §257.5's tp1 count "41 / 18 / 11" was taken before the die-shape freedom landed. The shipped code gives 42 / 18 / 10: 100T + 1 × HBM3E reshapes to 11.2 × 9.8 mm and holds its stack.
- The 0.312 DSE switch "含不可建点" was sent only with dies set. The 0.313 switches are sent with every DSE.

### §258.6 Thermal / power buildability (opt-in judgement)
`thermal.thermal_report(pkg_W, die_mm2, dies, dram_dev_W, cooling)`:
- Die heat = (package W − DRAM-device W, i.e. dram + hbm_static J share) ÷ dies. The card idle / static figure is counted on the die 「近似」.
- Flux = die heat ÷ area. Current density = die heat ÷ 0.75 V 「假设」 ÷ area (reported only).

| preset | package W | flux W/mm² | sources |
|---|---|---|---|
| air_oam | 600 | 0.85 | OCP OAM r2.0 「来源」; HIR 2021 ch. 20 air 0.55–0.85 「来源」 |
| air_hgx | 1000 | 0.85 | HGX B200 air 「二手」 |
| liquid | 1400 | 2.5 | HGX B300 「二手」; single-phase cold plate 1.7–2.5 「来源」 |

- **Known part:** H100 SXM 700 W / 814 mm² = 0.86 W/mm² card-level, at the air limit 「二手」.
- **Where it binds** (tpuv4i_7nm energy, qwen3-8b prefill 4k, b1):

| configuration | power and flux | air | liquid |
|---|---|---|---|
| 1P + 4 × HBM3E | 504 W, 0.86 W/mm² | over | ok |
| 1P + LPDDR5X 8 × 64 | 0.93 W/mm² | over | ok |
| 1P + 8 × HBM3E | 0.75 W/mm² (the PHY area spreads the heat) | ok | ok |
| H100-like | 0.55 W/mm² | ok | ok |

  Decode stays at 0.1–0.25 W/mm².
- **Default:** report only (`cooling` None). The package limits are 二手 product configurations, so making them default constraints would change verdicts on weaker sources than the shoreline ones.
- **Power delivery** is not judged. The only per-bump figure found is the legacy ≈ 0.2 A per 50–100 µm C4 bump 「二手」, which does not apply to current micro-bump pitches.

### §258.7 Freshness (2026-10-22)
- HF sweep since 2026-10-21: only Ling-3.0-tiny-UBTree (in).
- KRAFTON Raon-1-mini-preview (30B-A4B): config gated and custom code, so not added. KRAFTON added to the watched orgs.
- UCIe 4.0 unpublished. JEDEC has no HBM4E. 448G is in OIF definition.

## §259 Thermal model audited against shipped parts; shore_strict evidence (0.314)

### §259.1 Flux definition
- A shipped part's figure can only be formed as TDP / TBP ÷ logic-die area, because its HBM power and VR loss are not published. Since 0.314 the model judges the same quantity: `flux_W_mm2` = package W ÷ (dies × die mm²).
- 0.313 judged (card W − DRAM-device W) ÷ area against 0.85 W/mm², which equals H100's card-level figure. That mixed two definitions.
- The die-level figure stays reported as `die_flux_W_mm2`. Both are averages.
- Hot spots are not modelled. A shipped part's average already carries its own hot-spot ratio, so the comparison holds for a design of similar uniformity 「近似」.
- For 3D stacks (MI300X / MI350X) the area is the heat-exiting footprint (the IODs).

### §259.2 What the modelled power contains
- **Included:**
  - Dynamic MAC, vector, SRAM, SLC, NoC + wire, clock, DRAM device, DRAM PHY (host side, on die), D2D and link energy.
  - SRAM and NoC-router leakage, and HBM standby.
  - Whole-card static (logic leakage, uncore) only when the energy table has `idle_W` or `idle_frac_tdp` × `tdp_W`. `chiplet_example.table()` drops these; the eval API includes them when the user sets TDP.
- **HBM:** device power is in the package figure, like a TBP, and out of the die figure. The host PHY is on the die in both.
- **Not modelled:** VR loss on the module.
- Both omissions bias the modelled power low against a shipped TDP. An "over" verdict is therefore robust; an "ok" within that margin is not. `static_included` and `notes` say so per report.

### §259.3 Limits
Rule: the larger of (a) the highest average flux or package power of a shipped part with that cooling and (b) the HIR 2021 Table 1 thermal-budget figure (50 °C junction-to-inlet, 4 cm² chip, demonstrated hardware). The HIR uniform-heat-flux idealisations (air 0.85, water 2.50 W/mm²) are not used.

| preset | package W | flux W/mm² | basis |
|---|---|---|---|
| air_oam | 600 → 1000 | 0.85 → 0.86 | MI350X OAM air 1000 W TBP (AMD 「来源」); Gaudi 3 HL-325L OAM air 900 W (Hot Chips 2024 「来源」); flux: H100 / H200 SXM 700 W ÷ 814 mm² 「推导」 |
| air_hgx | 1000 | 0.85 → 0.86 | HGX B200 configurable to 1000 W, air (NVIDIA Blackwell datasheet 「来源」) |
| liquid | 1400 | 2.5 → 1.70 | MI355X DLC 1400 W TBP (AMD 「来源」); HIR water cold plate at a 50 °C budget 「来源」 |

Shipped parts, as TDP ÷ logic-die area. Air:

| part | W/mm² | inputs |
|---|---|---|
| H100 / H200 SXM | 0.86 | 700 W ÷ 814 mm²; Dell XE9680 air, 35 °C inlet 「来源」 |
| MI350X | 0.68 | 1000 W ÷ 2 × 732 mm² (IOD 「二手」, ASCII.jp) |
| B200 | ≥ 0.58 | 1000 W ÷ 2 × ≤ 858 mm² 「二手」 |
| MI300X | 0.51 | 750 W ÷ 4 × 370 mm² (IOD 「二手」) |

Liquid:

| part | W/mm² | inputs |
|---|---|---|
| MI355X | 0.96 | 1400 W ÷ 2 × 732 mm² 「二手」 |
| B200 | ≥ 0.70 | 1200 W 「二手」 |

Gaudi 3 publishes package power only (air 900 W / liquid 1200 W). TPU v5p / v6e have no official TDP or area, and third-party estimates are not used.

HIR's air figure at a 50 °C budget (0.55; text: 40–53 W/cm² for a 45–50 °C budget) is exceeded by every high-end shipped air part, which use larger heat sinks, a 35 °C inlet and vapour chambers. Shipped parts are the better buildability evidence.

### §259.4 Effect
Scan over 80 chip × memory × phase cases (qwen3-8b, tpuv4i_7nm dynamic table, 3 presets each; /tmp/a314/therm2.py): **0 verdict changes** from 0.313.
- Over air: 1P and 1P-lp4x64 prefill with 1 / 2 / 4 × HBM3E and LPDDR5X 8 × 64. Card-level 0.88–0.97 W/mm², 1.03–1.13× the H100 envelope.
- Over liquid: none.
- With the H100 static floor charged (idle_frac_tdp 0.103 × 700 W 「二手」, meaningful only for 700 W-class chips), 1P prefill over air goes from 4 to 6 memory configurations per chip: + 8 × HBM3E 0.88, HBM3 5-stack 0.97.
- The 1P + 4 × HBM3E prefill flag that prompted the audit holds under the consistent definition. It is marginal (1.03×) without static and clear (1.17×) with it.

### §259.5 Memory on more than two edges (shore_strict)
- **Shipped HBM parts, ≤ 2 edges per logic die** 「二手」:
  - GV100 / GA100 / GH100 / B200 (long edges).
  - MI250X GCDs.
  - MI300X / MI350X IODs.
  - TPU v5p (3 + 3 stacks on two sides, I/O die at the front).
  - Trainium2 (stacks on both sides).
- **Announced:** Rubin Ultra 4-die / 16-stack (stacks on outer edges; reportedly dropped for a 2-die package 「二手」).
- **Patent:** Google US 12,315,860 B2 claim 14 claims 12 stacks with some "along a third side of the second logic die" 「来源」. This is the only third-edge layout found. Claims 15–18 also place stacks two deep (2 × 2 arrays fed through organic substrate traces), a mechanism not modelled here.
- No source excludes a third edge and none ships one, so `shore_strict` stays opt-in. Every "tight" shoreline report now carries `precedent` (`budget.TIGHT_PRECEDENT`).
- The same patent gives a 650 mm² 5 nm die → about 2250 mm² interposer with surrounding HBM, against the 0.313 bbox for 1P + 8 × HBM3E: 654 → 2264 mm² (independent, not fitted).

### §259.6 Freshness (2026-10-22 JST)
- + OvisOCR3 (ATH-MaaS, Apache-2.0, 2026-10-21 11:36 JST). Qwen3.5-0.8B post-train, qwen3_5 VLM, bf16, 0.752 B.
  - Same config and header shapes as StartLux-Decision-v1-0.8B, minus the MTP head → merged (0.251 / 0.288 rule). Resolvable and validated; the ViT is built (identical preprocessor_config).
  - ATH-MaaS added to the sweep's EXTRA_ORGS.
- Raon-1-mini-preview: config still gated (HTTP 401).

## §260 Complete package power for thermal / power buildability (0.315)

### §260.1 Problem
In 0.314 the judged package power was the modelled card W. It held the whole-card static only when the energy table had an idle figure, and it held no voltage-regulator loss. A shipped TDP / TBP holds both: OCP OAM r2.0 §8.4 defines TDP at the 44–59.5 V module input 「来源」. So the two sides of the comparison had different scopes, and the report needed the caveat "over is trustworthy, a marginal ok is not".

### §260.2 Mechanism (`thermal.package_power`)
pkg_W = (card_W + S_add) / η − idle_board × (1/η − 1), where:
- **Static floor.** S_add = max(0, ρ_s · A_logic · η − S_charged).
  - ρ_s = **0.088 W/mm²** of logic-die footprint: the idle (no workload, idle clocks) card power per mm².
    - H100 SXM: 71.8 W (lo0 tech blog 「二手」) ÷ GH100 814 mm² 「二手」 = 0.0882.
    - MI300X: 131–133 W socket at SCLK 132 MHz and 0 % utilisation (AMD technical article, rocm-smi output 「来源」) ÷ 4 × 370 mm² IOD 「二手」 = 0.089.
    - Two vendors agree within 1 % 「推导」.
  - The measured idle contains logic leakage, uncore, PLLs and HBM self-refresh. So the static the model already charges (S_charged = SRAM leakage + NoC-router leakage + HBM standby, `powercap.static_W`) is subtracted. This is the same no-double-count rule energy.py applies to a measured idle.
  - The figure is board-level, so it enters on the delivered side as ρ_s · A · η.
- **VR / PDN loss.** η = **0.85** 「近似」, range 0.77–0.90.
  - ECCE 2024 single-stage 48 V → 1 V VPD: 84.4 % system efficiency at full load (1.0 V / 400 A), 77.2 % at 0.65 V / 600 A 「来源」.
  - APEC 2024 two-stage, first stage 48 → 1.8 V: ≤ 95 % 「来源」, times a second-stage VR.
  - Conventional lateral delivery from the platform input is estimated below 70 % (arXiv 2606.28837), but that is rack scope and not used.
- **Table with an idle figure** (idle_W, or idle_frac_tdp × tdp_W): S_add = 0, and the loss applies only to card_W − idle (the idle figure is board-level).
- **Placement.**
  - HBM / DRAM devices (dram + hbm_static): in the package, off the die.
  - Host DRAM PHY: on the die.
  - VR: on the module.
  - die_W = (card_W + S_add − DRAM devices) ÷ dies.
  - The judged card-level flux = pkg_W ÷ (dies × die mm²), the shipped definition (TDP ÷ die area).

### §260.3 Limits per definition
The rule "the larger of the shipped part and HIR" compared HIR's chip-level heat flux against card-level power. Each precedent is now judged in its own definition, and a point passes on flux when either holds:
- Card-level `flux_W_mm2` against the highest shipped part: air 0.86 (H100 / H200 SXM 700 W ÷ 814 mm²), liquid **1.70 → 0.96** (MI355X 1400 W ÷ 1464 mm²) 「推导」.
- Die heat `die_flux_W_mm2` against the HIR 2021 Table 1 thermal-budget row (50 °C, 4 cm²): air 0.55, liquid 1.70 「来源」.
- Package power is unchanged: air 1000 W, liquid 1400 W, now against a module-input pkg_W.

### §260.4 What remains of the caveat
- The floor is measured at idle temperature: MI300X idle junctions read 34–47 °C 「二手」, taken as 40 °C 「近似」.
- `pkg_W_hot` = pkg_W + S_total · (k − 1) / η, with k = leak_temp_factor(85) / leak_temp_factor(40) = 3.21. This is the TSMC 3 nm SRAM ratio applied to all static, an upper bound 「假设」.
- An "ok" point that is over under this bound carries `hot_ok` False and a note. Only those points are marginal.
- The bound is loose. It flags H100-like prefill (627 W package; the real H100 ships at 700 W on air), so it is reported, not judged.
- Hot spots are still not modelled (the same caveat as §259).

### §260.5 Effect (old → new)
80 chip × memory × phase cases × 3 presets (qwen3-8b; tpuv4i_7nm table without idle; /tmp/a315/scan.py run against 959ec95 and 0.315):
- **Air (both presets): 8 → 12 over.** Flips ok → over: 1P and 1P-lp4x64 prefill with 8 × HBM3E (card-level 0.776 → 0.990 W/mm²; package 507 → 647 W = + 43 static + 97 VR) and with HBM3 5-stack (0.849 → 1.081).
  - The 8 already over stay over: 0.88–0.97 → 1.12–1.23 card-level; die heat 0.93–1.02 > 0.55.
- **Liquid: 0 → 0.** Card-level is up to 1.33 > 0.96, but die heat ≤ 0.72 × 1.70.
- **Package / card ratio** 1.26–2.35 (median 1.38). The high end is the 100T chip, whose ~100 W dynamic power is small next to the area floor.
- **Table with the H100 idle fraction:** 20 → 20 over, 0 flips.
- **Calibration check (not tuned):** H100-like on HBM3 5-stack gives a complete package power of 627 W prefill and 243 W decode, under the 700 W TDP (a rated maximum). The floor path and the measured-idle path agree to < 1 %. That agreement is the consistency test, and it holds by construction of ρ_s from the same measurement, independent of chip.
- **Hot-margin flags:** 17 of the 68 air "ok" cases are prefill points. They are informational, not judged.

### §260.6 Not changed
- The DVFS power cap (`power_W_card` / `tdp_W`) still compares the energy-table card power. Making the cap module-input scope would change every capped DSE throughput and needs its own audit; it is deferred.
- Power delivery stays reported, not judged.

### §260.7 Suite wall time
0.314's 553 s (from 452 s in 0.313) was not reproducible:
- The unchanged 0.314 code gives 446 s, 1670 passed.
- The excess sat in 8 small files, each ~4.5 → ~113 s in one concurrent window. In the rerun they take 3.9–4.9 s.
- The median per-file ratio against 0.313 is 1.00, and steal was 0.01 %.
- Cause found (0.315 runs): **host-level VM freezes**, not code and not a co-tenant.
  - A 2 s `date` / `sleep` sampler stopped for 123 s (05:20:20 → 05:22:23 JST) and 47 s (05:22:46 → 05:23:33). Across the first gap /proc/stat advanced only 289 ticks (≈ 3.6 s of CPU on 8 CPUs).
  - virtio_balloon moved ≈ 2.0 M pages each way (≈ 7.6 GB inflated / 8 GB deflated) in the same window. steal does not count a paused VM, and cgroup throttling was 0.
  - Every file running during a freeze gains its length: 8 files × ~100 s in 0.314; test_core_184's fuzz guard read a 122.4 s "hang" (passes alone in 12.7 s).
- Real, separate fix: the committed `scripts/test_durations.json` was last refreshed in 0.238. test_core_237 (113–193 s actual) was ranked at 28 s, so longest-first started it near the end and it set every run's tail.
  - Refreshed from the quiet base rerun; 237 now ranks 6th. The 0.315 commit carries this file instead of the base's copy, a deliberate exception to the reset step.
- 0.315 runs:

  | Run | Result | Wall |
  |---|---|---|
  | Base rerun | 1670 passed | 446 s |
  | 0.315 code, stale durations | 1678 passed | 538 s (237 tail 193 s) |
  | Refreshed durations | 1678 passed | 551 s (one freeze) |
  | Third run | 1677 passed + 184 freeze-flake | 583 s (two freezes) |

  Wall time on this box is bounded by freezes, not by code.

### §260.8 Freshness (2026-10-22 JST)
- + Flint (RecursiveSI/flint, Apache-2.0, 2026-10-21 23:20 JST), model_type `flint` on the generic GQA-MoE path.
  - 49 layers: 2 dense (`mlp_only_layers`) + 47 MoE (256 experts top-6, sigmoid router without bias, 1 shared expert of 1024 with no `shared_expert_gate` tensor).
  - 35 sliding-window (513, including the current token) + 14 full-attention NoPE layers.
  - q / k RMSNorm [128], learned per-head sinks (`self_attn.softmax_offset` [32]). bf16.
  - Header total 39.785 B = config build (param_err 0). Active 2.82 B; the card's ≈ 2.5 B excludes the input embedding (0.31 B).
  - Quantised siblings (int4, nvfp4-gguf) are not added.
- Raon-1-mini-preview: config.json HTTP 401 (gated "auto").

## §261 Power cap in package scope; thermal front exclusion; freeze-aware guards (0.316)

### §261.1 One function for both checks
- Thermal (0.315, §260) judges the module-input package power `P_pkg = package_power(P_card, A_die, n, P_static, P_idle)`: card W + whole-card static floor + VR / PDN loss.
- 0.315's DVFS power cap still compared `P_card` with the budget. A capped point therefore sat at `P_pkg ≈ 1.28 × budget` (896–900 W on the 700 W chiplet examples), and the same point could pass the power budget while failing thermal at the same watts. That is a scope mismatch, not a model error in either part.
- 0.316: `thermal.cap_power_fn(A_die, n, scope)` returns `pk(P_card, P_static, P_idle) = package_power(...)["pkg_W"]`. It is passed into `powercap.dvfs_cap`, `energy._power_cap` and the DSE row.
  - Every cap comparison is `pk(·) ≤ cap`.
  - The thermal check calls the same `package_power`, so on a capped row `thermal.pkg_W == cap_pkg_W` (verified, max relative difference 0 over 42 rows).
- Static at the capped point: DVFS changes V, so the static W passed to both checks is the capped point's `static_W(f)` (package scope only; card scope keeps 0.315's row static, so it is reproduced bit-for-bit).
- The duty-cycle branch (below f_min) needs the card W whose package power equals the cap: `card_W_for_cap` bisects `pk` on [0, cap] (60 steps). `pk` is monotone and `pk(c) ≥ c` 「推导」. It returns None, so the point is infeasible, when `pk(0) > cap` (the floor + loss alone exceed the budget).
- Scope is 「来源」 OCP OAM r2.0 §8.4: TDP is defined at the module input (§260.1). `power_scope="card"` keeps 0.315.
- The eval API uses the shoreline die area. The what-if lever stays a card-W probe by design.

### §261.2 Effect (old → new)
See CHANGELOG 0.316.0.
- 126 example rows: capped 14 → 21; air_oam feasible 83 → 94, because capping now also brings package W under the air limit; front points 26 → 24. No mono / split verdict changes.
- Capped throughput factor × 0.758–0.936 on the prefill cases. Decode cases are unchanged.

### §261.3 Thermal in the DSE front
- The rule is unchanged since 0.313: with a cooling preset, a row over its limit is unbuildable and leaves every front, as shoreline-unbuildable rows do (`include_unbuildable` keeps both). Without a preset, thermal is reported only.
- 0.316 records the effect. Front without the cooling filter (rows feasible except for the thermal reason) minus the buildable front → `front_thermal_out` rows, `n_front_thermal_out`. These are shown in the UI summary and table.

### §261.4 Freeze-aware timing guards
- A host / VM freeze (§260.7) stops the process while wall time runs, so wall guards (fuzz 120 s watchdog, 20 s / 2 s bounds) report hangs that are not hangs.
- `scripts/freeze_clock.py`: a 50 ms sampler. A gap ≥ 0.5 s with process CPU < 20 % of the gap 「假设」 counts as frozen; `now()` = monotonic − frozen.
- Detection is not weakened:
  - a blocking hang has no gaps (the sampler keeps waking);
  - a CPU-bound hang, including a GIL-holding C call, shows CPU ≈ gap;
  - only non-execution is excused.
- The test_core_210 CPU bound is unchanged.

### §261.5 Freshness (2026-10-22 JST)
- No new tracked-org release since 0.315. Raon-1-mini-preview is still gated (401).

## §262 Cost per unit of work — TCO (0.317, opt-in)

### §262.1 Definition
Per package at its operating point:

    $/M unit = [C_yr + E_yr] ÷ [thr · util · 3600 · 8760] × 1e6
    C   = usd_good_pkg + mem_GB · $/GB + system_usd_per_card
    C_yr = C / L                      (straight line)  or  C · r / (1 − (1 + r)^−L)
    E_yr = [util · (P_pkg + P_sys) + (1 − util) · (P_pkg,idle + P_sys)] · PUE · 8760 / 1000 · $/kWh

- `P_pkg` = `thermal.package_power` at the row's operating point: card W including DRAM devices + static floor + VR / PDN loss (§260). On capped rows it is the capped point (= `cap_pkg_W`, §261). `P_pkg,idle` = the same function with only the static.
- `thr` is at the scenario's batch. SLO-violating points are already infeasible, so a ranked point meets its SLO.
- `unit/s/W` = thr ÷ P_pkg (module input; facility W = (P_pkg + P_sys) · PUE).

### §262.2 What each term contains (no double count)
- `usd_good_pkg` (0.310) = Σ good dies + packaging $/mm² × (Σ dies + HBM footprints) + test / assembly. The HBM footprint is interposer area (packaging), **not** HBM devices; memory devices are priced only as `mem_GB × $/GB`.
- `P_pkg` contains the DRAM device power (energy table) once, and the static once: the floor is net of SRAM / NoC / HBM-standby leakage already in card W (§260.1).
- PUE carries all facility overhead (cooling, distribution) once. The thermal cooling preset is a buildability limit, not an energy term.
- System terms (host / network) are zero in `accel_2026`. In `server_2026` they are a lower-bound capex share and an upper-bound power share, both 「推导」.

### §262.3 Buildability
- The $/M front and the cheapest point use rows that are feasible **and** buildable. `include_unbuildable` keeps unbuildable rows on the throughput / J fronts as before, but never on the $/M front: an unbuildable design has no cost.
- Thermal-out rows (cooling preset, §261.3) are infeasible and are excluded.

### §262.4 Sources
- PUE 1.54 (Uptime 2025 「来源」).
- $0.0891 / kWh (EIA Table 5.3, industrial, rolling 12 months to Jul 2026 「来源」).
- Life 5 years (10-K disclosures, 5–6 years 「来源」; low end 「假设」).
- HBM3E $17.5 / GB and HBM4 $31.5 / GB (TrendForce citing Fubon 「二手」); LPDDR5X $19.5 / GB 「推导」.
- Types without a recent public price are unpriced.

### §262.5 Verdicts under $/Mtok
- Mono vs split: unchanged on all 7 cases × 2 MAC scales.
- Per-package cheaper option: 16 / 72 sweep cells flip split → mono, all at tp 8. Split saves yield $ but loses 40 % throughput; package $ is ~40 % of TCO.
- Memory choice: $/Mtok differs from the throughput choice at tp 8 MAC × 2, D0 ≤ 0.2 (4 vs 8 stacks): 8 stacks force a split and double memory $.

### §262.6 Freshness (2026-10-22 JST)
- No new tracked-org release. Raon still 401.

## §263 TCO sensitivity (0.318)

### §263.1 Mechanism
For one design, throughput, package power and the idle floor are outputs of the cost / energy model at the operating point. None of the TCO inputs changes them, so only these terms move:

    package $ = package_cost(wafer $, Y = exp(−A·D0/100), CoWoS $/mm² × s, test)      (§255, same function)
    memory $  = GB × $/GB                                                            (§262.2)
    $/M       = tco_point(life, PUE, $/kWh, r, host $ / W)                           (§262.1)

- `row_usd` recomputes these from the row (mm², dies, Σ die mm² + HBM footprint, GB, thr, P_pkg, P_pkg,idle). It equals the row's $/M at the base.
- **Tornado:** one input at a time to each end of its range; bars are Δ$/M ÷ base, sorted by the larger magnitude.
- **Flip:** `verdict(overrides)` on a 33-point grid over the range, then each label change is bisected (40 steps).
  - Mono / split uses the 0.317 rule: split if its best $/M is more than tie (1 %) below the mono best, over buildable, priced rows.
  - The sweep "cheaper" rule and the memory choice (argmin of the best $/M per memory option) are as in the 0.317 study.

### §263.2 Ranges
Each end carries its own tag; the table is in CHANGELOG [0.318.0].
- Every priced default lies inside its range (tested).
- The D0 low end (0.05) is 「假设」, because no public mature-N5 D0 exists. The high end, 0.2, is TSMC's pre-volume N5 figure 「二手」.
- The CoWoS-S relative band is borrowed from CoWoS-L 「假设」.

### §263.3 Results (chiplet example, n5_cowos_l, accel_2026)
- **Size of the effects:**
  - Host share, capital rate, $/kWh, memory $/GB, life and PUE each move $/Mtok by 10–130 %.
  - Wafer $, D0 and CoWoS move it by ≤ 5 %, because good-die + packaging $ is a minority of a 5-year TCO.
- **Mono / split:**
  - 3 of 7 priced mono verdicts flip to split at D0 ≈ 0.16–0.17 /cm². Above that, the yield loss of the large mono die outweighs the throughput gain.
  - No other input flips them.
  - With a server_2026 base, none flip.
- **Sweep:** 1 / 44 cells flips: 4 × HBM3E, tp 1, D0 0.05, MAC × 2, mono → split at wafer ≥ $19.8k or $/kWh ≥ 0.1245.
- **4- vs 8-stack:**
  - The tp 8 MAC × 2 D0 ≤ 0.2 choice of 4 stacks holds over every range.
  - Flips appear only in D0 0.3 cells, which lie outside the sourced D0 range and are near ties.

### §263.4 Memory prices (default change)
- HBM3: none → $13.0 「二手」.
- HBM4E: none → $28 「二手」 (a 2027 forecast).
- LPDDR5: none → $10 「二手」.
- LPDDR6: unpriced.
- Sources and dates are in CHANGELOG [0.318.0] and `tco.MEM_USD_PER_GB`.

### §263.5 Limits
- One input at a time; joint corners are not searched (0.319: both joint corners, §264.3).
- Bonding / assembly yield loss is not modelled (asm_yield 1) (0.319: §264.1).
- The memory range is the row's type range even when the user set a price.

### §263.6 Freshness (2026-10-22 JST)
- No new tracked-org release. Raon still 401.

## §264 Package / bonding yield by component; joint TCO corners (0.319)

### §264.1 Mechanism (`energy_dse.asm_yield`, `package_cost`)
§255.3 divided the whole package by an assembly yield of 1.0 「假设」. Assembly is a sequence of placements on a carrier. Each placement can fail, and with chip-first CoWoS a failure scraps what is already on the carrier:

    Y_asm   = Y_carrier × y_attach^S × y_bond^N          (S HBM stacks, N logic dies; y_bond = microbump attach or hybrid bond)
    Y_exp   = Y_asm (chip_first) | y_attach^S · y_bond^N (chip_last: carrier tested before the KGDs go on)
    k       = (1 − ρ (1 − Y_exp)) / Y_exp                (ρ = reworked fraction of failures)
    $/good package = N · good die · k + packaging $/mm² · area + test
    TCO memory $   = GB · $/GB · k   (HBM on the carrier; LPDDR is off it, so ×1)

- Packaging and test are not divided by Y. The CoWoS $ per package (Silicon Analysts, Epoch) is a top-down price per shipped package with the interposer yield inside it, so dividing would count it twice 「推导」.
- The HBM factor lives in the TCO memory term, never in `usd_good_pkg` (§267 rule: memory counted once).
- Calibration and sources are in CHANGELOG [0.319.0]:
  - CoWoS-S carrier 0.985 「二手」; attach 0.998 「推导」 from Epoch 95–99 % per CoWoS-S package (H100 1 + 6 sites).
  - CoWoS-L carrier 0.81 「推导」 from Epoch's B200 yield-loss $ (Y ≈ 0.79).
  - Hybrid bond 0.97 「二手」 (ECTC D2W).
  - Scrap of KGD + HBM on failure 「二手」, ρ = 0 「假设」.
  - Check: B200 0.794, H100 0.971.
- `asm_model: "flat"` is the §255.3 / 0.318 rule (÷ asm_yield, memory unscaled), reproduced bit-for-bit.

### §264.2 Effects
- **Case 0 (CoWoS-L, 8 × HBM3E, D0 0.1):**
  - Cheapest design $2,013 → $2,116 per good package.
  - $/Mtok +11.8 %. Most of it is HBM scrap: 1.257 × $3,360 of HBM, against $103 of KGD scrap.
- **Break-even D0 (§255.3 table, package only):**
  - 4 dies vs mono: 0.12 → 0.100. 2 dies: 0.14 → 0.122.
  - Including the HBM scrap: 0.109 and 0.125.
  - Why it moves: scrap ∝ the silicon $ at risk, which is largest for the 654 mm² mono die (k 1.257 vs 1.265 for 4 dies, but $399 vs $223 of good silicon).
  - The 4- / 8-stack trade gains a term: each extra stack lowers Y by 0.2 % and is itself scrapped with a failed package. The verdict change is limited to the D0 0.3 cell below.
- **Verdict changes, flat → component** (`tco_sens.py --asm-compare`):
  - n5_cowos_l / accel_2026: 1 of 230 verdicts — memory choice tp8/400 D0 0.30 MAC × 1, by $/M 8 → 6 stacks, outside the sourced D0 range.
  - Everything else is unchanged, including n5_cowos_s and server_2026.

### §264.3 Joint corners (`tco_sens.corner`)
- All applicable inputs go to their adverse ends together (low is adverse for life and the yields; memory goes to the high end of the row's type range), and likewise to their favourable ends.
- D0 is skipped where it is a grid axis.
- The joint corner bounds every single-input end (tested).
- **Results (accel_2026):**
  - Mono → split at the adverse corner in the same 3/7 cases that flip on D0 alone (now at D0 0.150–0.155), and none at the favourable corner.
  - Sweep: 1/44 at adverse.
  - Memory choice: 2/24 at both corners, the D0 0.3 cells going to 8 stacks.
  - The tp8 MAC × 2 4-stack choice and the tp8/400 MAC × 1 8-stack choice hold at both corners. The latter is now a near tie with 6 stacks under single inputs (5/24 vs 3/24 single-input memory flips under flat).
- **server_2026:** cases 3/7 (D0 0.186–0.194), sweep 1/44 at adverse; memory 0/24.
- **Case 0 band:** −52 % … +283 % $/M.

### §264.4 Limits
- One carrier yield per technology; no learning curve over time and no per-interposer-size dependence beyond the per-carrier calibration.
- The hybrid-bond option uses an ECTC D2W yield, not a production SoIC yield.
- Rework is a user input (default 0).
- Corners are the two ends only; there is no search over mixed corners.

### §264.5 Freshness (2026-10-22 08:29 JST)
- No new tracked-org release; HF newest are derivatives. Raon-1-mini still 401.

## §265 Design verdict report; DeepGEMM masked-kernel ceiling scope; Raon-1-mini (0.320)

### §265.1 Design verdict report (`core/report.py`)
- The report is a view, not a model: every value is read from one `insight_dse` row and its result (perf, energy, shoreline, thermal, cost, tco, tco.sens). It recomputes nothing except the near-tie checks below, which reuse `tco_sens`.
- Item = `{key, label, value, unit, tag, src}`.
  - The tag is the source tag of the underlying quantity: preset numbers keep their preset tag, model outputs are 推导, and inputs the user set are 用户输入.
  - A value the run lacks is `None` with a `missing` reason. There is no zero default (§267 rule).
- Near-tie warnings:
  1. The DSE compare `tco_flips` / `tco_corners` entries for this row's `compare_key`.
  2. `_pair_check(base, this, other)`: `scan_flip` over every TCO input range (§263), plus `corner` adverse / favourable (§264). `other` is (a) the cheapest other priced design and (b) the best alternative memory of the same design.
  3. Margins (SLO, shoreline edge, flux, package W) ≤ tie (1 %).
- `_base` rebuilds the tornado base from the request body, and `tornado` raises if `row_usd` does not reproduce the row's $/M (§268 invariant). The report therefore cannot silently price a different design.
- Markdown is rendered from the JSON (`to_markdown`), with every item's tag and source in the table.

### §265.2 DeepGEMM masked grouped FP8 (extkern), 256+ rows
- §242.3 left 0.73 at 256 – 2048 rows. Re-examined, it turned out to be a scope error, not a missing mechanism. `grouped_h800` is DeepGEMM, and `DEEPGEMM_0325` per-shape ceilings 「来源」 exist for exactly these (N, K). The extkern scenario had no `src`, so the generic cap applied.
- 0.320 applies the DeepGEMM stack scope (`deepgemm_variant` + `DEEPGEMM_SRC`), as for the DeepGEMM validation rows. By m:

  | m | 256 | 512 | 1024 | 2048 |
  |---|---|---|---|---|
  | 0.319 | 0.844 | 0.643 | 0.672 | 0.726 |
  | 0.320 | 0.85 | 0.839 | 0.882 | 0.973 |

  - All 256+ rows: 0.729 → 0.86 (0.93 with the floor). Rows of ≤ 128 are unchanged (1.03).
- README cross-check 「来源」: the DeepGEMM README @ 55ab91f masked table matches the CSV (0.985–0.997).
- What remains (0.84 – 0.88 at m 256 – 1024) is the masked layout running below the dense / contiguous per-shape ceiling. No public masked per-shape ceiling exists, so it is documented, not fitted.
- Diagnostic only: no evaluation / DSE / TCO number moves. `summary(deepgemm=False)` = 0.319.

### §265.3 Raon-1-mini (`raon1_mini` in `_build_hybrid`)
- RDA → `_kda` (`rda_config`), + `rda_w` Linear h → H.
- MLA (NoPE: rope dims still in q / k, no rotation) with an output gate.
- Raon_mHC:
  - misc per layer 2 · (17 · N·h + 21); `Layer(hc=4, hc_iters=1, hc_rows=17)`.
  - IR selector GEMM [N·h → 17] instead of (2+N)·N. The vector term uses one normalisation pass for the closed-form chart 「近似」.
- Param check: 28,783,218,602 = release header (rel_err 0).

## §266 Colocated vocoder time-slicing; Dory logit scale; design comparison (0.321)

### §266.1 Colocated vocoder (extspeech, diagnostic)
- 「来源」 sglang-omni @ 0029194: engine and vocoder are separate processes on gpu 0, and `mps` defaults to "off". Without MPS the contexts time-slice and never overlap (NVIDIA; Gilman & Walls 2021 §4.2).
- Per vocoder window, while the engine is busy:

      lost GPU time = s + 2·t_sw            (switch in + out; t_sw = 145 µs, Gilman & Walls §6, RTX 3090 「二手」 → H200 「近似」)
      upper bound   = max(s, T_slice) + 2·t_sw   (T_slice ≈ 2 ms fixed, round-robin; holding a whole slice is not documented)

  - s is the DAC window time at batch 1 (`speech._dac_window`), ≈ 0.65 – 1.9 ms; 5 windows per request.
- Same → second-GPU speed-up, measured vs model:
  - c128: 1.696 vs 1.343 → 1.421 (bound 1.615).
  - c256: 1.718 vs 1.520 → 1.639 (bound 1.934).
- Reading: time-slicing explains c = 256 within published numbers. at c = 128 the measured speed-up is 5 % above even the full-slice bound (1.696 vs 1.615), so another per-window cost exists (DAC kernel efficiency, transfer interference). It is not fitted.
- For an accelerator: if the runtime cannot co-schedule two streams (one context, priority queues), each vocoder window costs at least a switch (≈ 0.3 ms per window here), and up to a whole slice. Streaming TTS with ≥ 5 windows per request should keep vocoder and engine in one context or give the vocoder its own engine.

### §266.2 Dory logit scale
- `logits.float() × (logit_scale / initializer_range)`: +vocab params (bf16, in lm_head bytes) and +1 op per logit in the sample vector op. Params exact for BF16 / MXFP8 / NVFP4 (3,975,229,440).

### §266.3 Design comparison (`report.compare_report`)
- Columns are `design_report` items, unmodified. Alignment is by item key. Δ = v / v_A − 1 (None if v_A is 0 or missing).
- Winner only for directed metrics (`DIRECTION`). Near tie: |best − runner-up| / max ≤ tie (the DSE tie).
- $/M: `_pair_check(A, X)` for each X (one-at-a-time scans over the sourced ranges plus both joint corners, §263 / §264). No pair check when any design is unpriced (reported as such).
- Flags: `differs`, `missing_in`, `tag_differs` (same quantity, different evidence label).

## §267 Drama 3 c256 wait and vocoder c128 residual re-examined; report / compare on the static build (0.322)

### §267.1 Second-GPU c = 256 wait: admission limits excluded
- Measured (cookbook Table 2, second GPU, c = 256): frames/s 4,160 (predicted 1.88×), TTFA p50 / p95 2,146 / 2,469 ms (predicted 403 ms).
  - p95 / p50 = 1.15, so this is a steady-state wait, not the t = 0 burst 「推导」.
- Checked against sglang-omni @ 0029194 and its pinned sglang 0.5.21 「来源」:

  | candidate | source | verdict |
  |---|---|---|
  | decode CUDA-graph capture size | `sglang/srt/arg_groups/memory_hook.py`: H20 / H200, tp < 4 → `max_bs` 256 | c256 replays a graph; excluded |
  | GDN / mamba state pool | Drama 3 `disable_radix_cache` True + `max_running_requests` → `kv_cache_configurator` sizes the pool to `max_running_requests` (256 in the cookbook) | cannot cap below 256; excluded |
  | first-frame graphs (16 requests), prefill graph (2048 tokens) | `sglang_model.py` falls back to an eager step above the limit (no cap) | steady prefill batch 10.8 × 148 ≈ 1,600 rows is inside both; burst only |
  | reference encode in preprocessing | cookbook: pre-encoded `vq_codes` | does not run; excluded |

- What remains is a sourced mechanism whose cost is unpublished. `disable_overlap_schedule` is True for Drama 3, so per-request scheduler and streamer CPU work is serial with each GPU round. This includes `Drama3FrameStreamer` building and sending one tensor message per frame for the first `stride` frames.
  - Diagnostic `host_implied_us` = (1/meas − 1/pred) × 10⁶ µs per frame 「推导」, measured-only, written by ext_validate:
    - Table 1: 393 (c8) → 194 µs (c64);
    - Table 2 second GPU: 77 (c128) → 113 µs (c256);
    - Table 2 same GPU: 184 → 204 µs.
  - A constant per-request host cost would make these equal across c. They fall in Table 1 and rise in Table 2, so no single host term explains the gap. Nothing is fitted, and predictions are unchanged.

### §267.2 Colocated vocoder c = 128 residual
- 「来源」 Drama 3's vocoder scheduler does not set `can_batch_stream_chunks` (each chunk is handled alone).
  - `build_stream_vocoder_chunk` copies every incoming item to the GPU (`codes.to(device, non_blocking=True)`).
  - The engine's `Drama3FrameStreamer` sends frames 1 … `stride` (10) as single-frame items, then complete windows.
- Per request that is 14 GPU submissions on the vocoder context: 9 copy-only plus 5 windows, against the 5 the switch term counts.
- If a copy cost a switch pair like a window, the vocoder-side loss would grow by 9 × 2 × 145 µs ≈ 2.6 ms per request (≈ +33 % over the ≈ 7.8 ms of the windows) 「推导」.
- Whether a small host-to-device DMA from another process's context forces a compute time-slice switch is not documented. Gilman & Walls 2021 (O4) report transfer interference but no per-copy cost. The candidate is documented, not added as a reading or a default (§271 gotcha: do not pick the reading that matches the measurement).

### §267.3 Report / compare on the static (Pyodide) build
- `/api/report` was already on the slow lane (0.320). Gaps found and fixed (app.js):
  1. Report and compare shared one channel, so starting one cancelled the other. Now `report` / `report_cmp`. The pill was keyed by endpoint, so compare showed as the report; `CHAN_STAGE` (opt-in) keys `report_cmp` by channel.
  2. A superseded request cleared the newer request's status line. Now it returns silently.
  3. A reply for a replaced design search would still download. Now `reportStale` (by `S.dseRes` identity) discards it.
  4. Scenario edits and new searches left in-flight reports running, and finished ones untagged. Now `dropReports` cancels them (the static worker is terminated), and both statuses are in `invalidate`'s stale list.
- Static smoke step 2e (`scripts/ui_static_smoke.py`) checks this live; tests/test_core_322.py pins it structurally.


## §268 Cross-module consistency; clock-free mapping memo (0.323)

### §268.1 One design, one set of numbers
The same design reaches the user through six paths: `/api/eval`, an energy-DSE row (`/api/insight_dse`), the design report and the comparison (`/api/report`, which read the row), the static Pages build (Pyodide) and the CLI. Server and static build both answer through `serve.respond`, which calls `api.clean` on the same `api_*` function; the CLI prints `api.clean` of that call too. So the transport adds nothing. The comparison has to be between eval and the DSE row, which compute separately.

Conventions, now shared (the DSE / report convention is kept; it is the more mechanistic one):

| quantity | convention | eval ≤ 0.322 |
|---|---|---|
| die yield without D0 | missing input: yield 1, flagged 「缺项」 | D0 0.1 /cm² default |
| auto memory edges | `hbm_mem_edges` on the budget-density die the row reports | on the preset's published area (H100-like 814 mm²) |
| power cap | `budget.power_W_card`, else `energy.tdp_W` (`cap_src`) | `energy.tdp_W` only |
| capped operating point for thermal / TCO | dfs clock (§222 bisection) | duty-cycle throttle |
| bisection depth | `CAP_ITERS` = 16 for every caller | eval 22, DSE 12 |

- The duty-cycle throttle stays as eval's headline throughput factor, labelled conservative (it can never beat the dfs clock without the per-cycle term, §223). Thermal and TCO need one operating point, and the DSE's is the dfs clock.
- Depth 16 「推导」: bisection on [F_MIN, 1] = [0.15, 1] leaves (1 − F_MIN)/2¹⁶ ≈ 1.3e-5 of the clock. The dfs ≡ duty identity (no per-cycle term) is checked to 2e-4, which 12 steps (2.1e-4) cannot meet. A deeper bisection only continues the shallower one, so DSE capped clocks rise by ≤ 2.1e-4.

Property tests (tests/test_core_323.py) draw seeded random designs and check eval ≡ row (J/unit, card / package W, thermal status, buildable, shoreline status, memory edges, good-package $, $/M, capped), report ≡ row, compare ≡ report (value, tag, unit), wire ≡ API and CLI ≡ wire.

### §268.2 Clock-free mapping memo
`_gemm_cost` returns cycles. It reads the chip's array, port, NoC and occupancy fields but never `freq_ghz`; seconds are formed by `gemm_seconds` / `vector_seconds` with the caller's own chip. The memo used the whole chip as its key, so each of the 16 bisection clocks of a capped design, and each `f_scales` value, re-ran the same search. `gemm_cost` now keys it on the chip at 1 GHz (`_cycle_chip`, memoised; `mapping.FREQ_FREE_KEY` False restores the old key).
- The result is bit-identical by construction. Tests check this two ways: the whole DSE JSON with the flag on and off, and `_gemm_cost` unmemoised at three clocks for 40 seeded shapes.
- The layer IR (`build_rank_ops`) is likewise free of chip fields and was rebuilt at every clock. It is memoised for the duration of one `energy_design_search` (`ir.rank_memo()`), not globally, so a module flag toggled between runs is never served stale.
- Measured, static build: a 216-point DSE with a binding 150 W budget went from 15.3 s cold / 13.2 s warm to 6.9 / 5.0 s (prefill: 15.4 / 11.4 → 9.9 / 5.2 s). What remains is the overlap feed simulation, whose op times depend on the clock. The 0.323 target was about 10 s, so no pruning of dominated or unbuildable points was added. Such pruning would also drop rows the UI can show (`include_unbuildable`).

## §269 Custom HBM4 / HBM4E base die; exact reuse in the energy DSE (0.324)

### §269.1 What a custom base die is
From HBM4 on, the stack's base die is made on a logic process. SK hynix uses TSMC N12, Samsung SF4, Micron its own CMOS 「二手」. TSMC quotes N12 at 0.8 V against 1.1 V for the HBM3E base die. Custom HBM4E (C-HBM4E) and NVIDIA's NVHBM go to N3P / N3 at 0.75 V 「二手」.

A *standard* base die still speaks the JEDEC interface: 2048 DQ to a PHY and a memory controller on the host. A *custom* base die takes the controller onto the memory and talks to the host over a narrow die-to-die link:
- Samsung (Hot Chips 2026): an 8.5 × 1.5 mm block against > 8 × 4 mm; channel depth 5.5 → 2 mm.
- Marvell: 512-bit D2D at 32 GT/s, interface power −45 … −70%.
- JEDEC SPHBM4 (JESD330-4, 2026-06) is the standard cousin: 512 DQ, 4:1 serialization.

`Chip.hbm_base` = "custom" selects it; "std" (default) is the ≤ 0.323 model. It applies only to HBM4 / HBM4E. On other memory it is ignored, with a warning in eval, a note in the area report and the row flag `hbm_custom` False.

### §269.2 What changes, and what does not

| quantity | std | custom | tag / source |
|---|---|---|---|
| host PHY area / stack | 2048 DQ × `mm2_per_dq_hbm` (n5_derived: 32 mm²) | 12.75 mm² | Samsung HC 2026 「二手」; host side = memory side 「推导」 |
| memory controller | `mm2_per_unit_mc` × units (user field) | on the base die (term dropped) | Samsung HC 2026: controller 5–10% of XPU area 「二手」 |
| host PHY pJ/bit | `pJ_bit_dram_phy` | × 0.55 (range 0.30 … 0.55) | Marvell / EE Times 「二手」, low end |
| memory $/GB | sourced preset price | × 1.10 (range 1.05 … 1.10) | EE Times part 1 (speculation) 「二手/假设」, high end; base die 3–4× a core die (TrendForce) as context |
| shoreline / stack | 11 mm | 11 mm | 「推导」: the stack footprint, ≥ the D2D width |
| stack bandwidth | unchanged | unchanged | not a base-die property (NVHBM's +30% = faster spec) |

The die-area change carries the existing uncore share of the final die (`budget.full_area`: uncore = u / (1 − u) of the rest). So 8 stacks remove 8 × 19.25 / (1 − 0.23) = 200 mm² on n5_derived.

The shoreline consequence is indirect. The per-stack need is the same, but the host die is smaller, so its perimeter is shorter. On the 1P example (820 → 620 mm²) the status stays "tight". On the H100-like example (428 → 278 mm²) six stacks no longer fit: "tight" → "over". The custom die's area saving is bounded below by the shoreline the stacks need, and a real design would keep the edge and spend the area on compute.

The thermal consequence is direct. The same die power on less area raises W/mm²: qwen3-8b prefill MAC×1 mono on air goes 0.81 → 1.04 W/mm², which is unbuildable. The DSE's TCO verdict then becomes only_split.

Listed, not modelled: near-memory compute / offload (no published throughput or energy); base-die process energy (its share of the device pJ/bit is unpublished); Marvell's "+33% stacks" (no mechanism, the footprint is unchanged); the D2D hotspot (Samsung's Heat Path Block, −35% peak temperature).

A user-entered memory price is taken as given: the premium applies only to the sourced default.

### §269.3 The examples (std → custom)
The eval figures and the energy-DSE study (`scripts/chiplet_example.py --hbm-base-study`) are tabulated in CHANGELOG [0.324.0]. The pattern:
- Uncapped: throughput is unchanged, J/unit falls by the PHY share (−2.3% decode, −0.1% prefill), the package $ falls by 14%, the memory $ rises by 10%, and $/Mtok rises by about 3%. HBM dominates the bill of materials.
- Under a binding power cap the PHY saving buys clock: HBM4E at 300 W gives +10.4% throughput and −6.0% $/Mtok. This is where a custom base die pays in this model.

### §269.4 Exact reuse in the energy DSE
A point at f_op < 1 whose power is over the cap is capped from f_max (the 0.302 operating-point rule). Its bisection is then exactly the bisection its f_op = 1 twin already ran: `dvfs_cap(evaluate(s), table, cap, curve, pk)` with every argument equal.
- `energy_dse._memo` keys that call, and `evaluate(s)`, on (the frozen Scenario, the frozen EnergyTable, cap, curve, (pk.scope, pk.die_mm2, pk.dies)). `pk` is rebuilt per row but is a pure function of those three.
- The store lives for one `energy_design_search` call (`_scoped_rank_memo`). A hit returns a deep copy of the cap dict; evaluate results are only read.
- Results are byte-identical (whole DSE JSON, `CAP_MEMO` on vs off; test_core_324).
- The prefill body at 150 W made 2844 evaluate calls, 1368 of them repeats. Pyodide prefill cap 150 W: 9.9 → 8.2 s cold, 5.2 → 4.0 s warm.

Unbuildable points are not skipped. Every row is kept with its metrics, and `include_unbuildable` only decides front membership, so skipping them would change the output. The pruning is restricted to work whose result is already known exactly.

## §270 Custom HBM base die: adversarial review (0.325)

### §270.1 Shoreline
On a silicon interposer a short-reach D2D link only works with the stack adjacent to the host, so the stack footprint (11 mm) still sets the edge per stack 「推导」. Eliyan reports 2 stacks per 25 mm on silicon 「二手」. The UCIe link itself needs only ≈ 2.72 mm of edge (7 × `ucie-a-48` modules).

A closer stack pitch needs a longer-reach interface on organic substrate:
- JEDEC SPHBM4 (JESD330-4): 512 DQ at 4:1, ≥ 90 µm bumps, 20 mm reach 「二手」, PR 2026-07-13;
- Eliyan: 3 stacks per 25 mm.

That is a different package and is not modelled (`ORGANIC_STACKS_PER_25MM`, reference only).

### §270.2 Interface energy
DRAM energy is device `pJ_bit_dram_hbm4` (memory side, which includes the base-die PHY) plus host `pJ_bit_phy_hbm` 0.29 (Chae ISSCC 2023).

A custom base die replaces both PHY ends with one D2D link. The link energy is that of `D2D_STD = ucie-a-48`, 0.25 pJ/bit; the memory end is taken symmetric to the host at 0.29 「推导」. It is split half and half:
- host: `phy_pJ_per_bit` = 0.125;
- memory: `dram_dev_pJ_delta` = 0.125 − 0.29 = −0.165, added to the DRAM term in `energy_report` and in `pdqueue`.

The ratio link ÷ pair = 0.43 lies inside Marvell's −45 … −70% (`VENDOR_IFACE_RATIO`, check only). With no `pJ_bit_phy_hbm` in the table, `iface_pJ` is None and nothing changes.

### §270.3 Area, and the shoreline floor
Host D2D per stack = ⌈`stack_GBps` ÷ 384⌉ × 0.3888 × 1.043 mm (HBM4 10 Gb/s: 7 × 0.4055 = 2.84 mm²). The Samsung 12.75 mm² is the base-die block (`base_die_d2d_mm2`, reference).

A custom die smaller than the shoreline the stacks need is padded to the floor:
- `shore_floor_mm2` bisects on `shoreline_report`;
- `full_area` adds `terms.shore_pad`.

All consumers (shoreline, thermal, eval TCO, DSE cost / yield, package) read the same die. NoC pitch and hops come from the engine tile and grid and do not depend on the PHY.

### §270.4 Thermal
The card-level precedent (H100 700 W / 814 mm²) is a die that includes the memory-interface ring. For custom, the card flux is computed on die + `std_equiv_delta` (`thermal_report(card_area_mm2=…)`), and the die-heat flux on the real die.

This removes 0.324's "smaller die → split" flips, which were an artifact of the definition. Controller power has no separate std term, so it is not moved (Samsung: controller on the base die, "thermal impact manageable"); this is listed in `NOT_MODELLED`.

### §270.5 Near-memory compute
Not modelled: no vendor base-die compute or energy figures exist, and per-cube bandwidth does not rise (AMMA).

Upper bound 「推导」: attention / KV offload saves at most KV-byte share × whole-link energy. For 1P qwen3-8b decode b64 custom that is ≈ 3.4% J at 4K ctx and ≈ 5.3% at 32K, gross of base-die compute.

## §271 SPHBM4 on an organic substrate; exact speed-ups of the custom-HBM DSE (0.326)

### §271.1 What SPHBM4 is
JEDEC JESD330-4 「来源」 keeps the HBM4 DRAM core dies and swaps in a buffer die whose external interface is 512 DQ, as 32 × 16-bit channels in 8 Quad Channels. Each channel is "four times faster than the corresponding HBM4 channel". Bumps are > 90 µm and channels up to 20 mm, which allows routing on a standard organic substrate instead of a silicon interposer.

Consequences:
- Stack bandwidth = 512 × pin rate ÷ 8 = the HBM4 stack at pin rate ÷ 4 「推导」, and capacity per stack is unchanged.
- The standard's 22.4 – 46.0 GT/s (2.944 TB/s at 46 「二手」) bounds the HBM4-equivalent rate to 5600 – 11500 MT/s. The option applies only to HBM4 in that range (`hbm_pkg.sphbm4_applies`); otherwise it is ignored with a warning.

### §271.2 Shoreline per stack
- On silicon a stack abuts the host: 11 mm per stack (the stack footprint, §259).
- On organic, longer reach and angled routing let stacks fan out. Eliyan's published layout is "three memory devices … along the same edge" of 25 mm, versus two on silicon 「二手」. The model takes 25 / 3 = 8.33 mm per stack as published (`SPHBM4_EDGE_MM`), not Eliyan's silicon-to-organic ratio applied to 11 mm.
- Every shoreline consumer reads `budget._stack_edge(chip, spec)`: `shoreline_check`, `mem_edge_need_mm`, `mem_edge_units`, `sized_layout`, `_sized_report`, `hbm_mem_edges` and `mem_edge_load`. With the discrete fit, a 25 mm edge holds exactly 3.
- The memory-edge count then follows the existing fewest-edges rule (0.309 / 0.312). Narrower stacks can need fewer edges, hence fewer NoC attach edges and more DRAM hop energy. That is the placement rule's effect, not the package's: pinning `noc_mem_edges` makes energy byte-identical. eval's `hbm_pkg.notes` says so when the count drops.
- 3rd / 4th-edge memory is still "tight" (no shipped precedent; Eliyan's 12-device package is a concept figure).

### §271.3 Interposer
The stacks are off the interposer, so the DSE package-area limit and the 0.313 bounding box count the logic dies only.

The package $ (`pkg_usd_per_mm2` × area) is NOT reduced. The organic-substrate $/mm² and the flip-chip attach yield have no public figure (SemiAnalysis says only "20 – 28+ layer" ABF). The CoWoS-rate stack footprint is kept as an upper bound, and eval reports the most it could be (`pkg_usd_bound`).

### §271.4 Not modelled (no public number; HBM4 values kept)
- host-PHY area and interface pJ/bit;
- SPHBM4 $/GB;
- substrate size / layer limits and 20 mm routing feasibility;
- SerDes latency.

All are listed in `hbm_pkg.NOT_MODELLED`.

### §271.5 Exact speed-ups (custom-HBM DSE)
The shoreline-floor bisection (§270.3) made custom DSEs about 1 s slower cold. Four exact changes:
1. `shore_floor_mm2` is memoised on (scenario, die).
2. `sized_layout` caches its die-independent inputs per key-without-die.
3. After the first die shape, a candidate whose memory or D2D is over by `_place`'s own expressions is skipped (`_surely_bad`). For shapes after the first, only nothing-over candidates are eligible, and shape 0 still supplies the fallback.
4. The water-fill is an explicit scan with identical arithmetic and tie order.

`_SIZED_CACHE` (keyed on `round(die, 6)`) is untouched, so the bisection sees the same layouts. Whole-output JSON is byte-identical to 0.325 on custom / std DSEs, chiplet grids, evals and the example script outputs.

## §272 HBM memory-controller area: sourced opt-in preset (0.327)

### §272.1 Source and number
- S. Han (Samsung), Hot Chips 2026 tutorial (2026-08-23), as quoted by MK (2026-08-24) 「二手」: "Memory controllers account for about 5-10% of the SoC area". Moving them to a custom base die "can be filled with operation cores".
- The same talk's slide 9 draws the reference xPU: 26 × 33 mm (858 mm², reticle class) with 6 HBM stacks per xPU 「二手」 (ServeTheHome slide images).
- Per stack: 0.05 × 858 ÷ 6 = **7.15 mm²** and 0.10 × 858 ÷ 6 = **14.3 mm²** 「推导」. This reads the quote as referring to the slide's xPU 「假设」. The quote names no product, so the range is kept rather than a point.
- Mechanism: a controller per HBM channel set, so the area scales with stacks, which is what `mm2_per_unit_mc` already does. The new field is HBM-only because the source concerns HBM controllers; an LPDDR package controller is not 7–14 mm².

### §272.2 Model
- `Budget.mm2_per_hbm_mc` (HBM memory only); `mm2_per_unit_mc` (generic) wins when both are set.
- `mc_preset` fills `mm2_per_hbm_mc` from `MC_PRESETS`.
- `full_area`: std host + stacks × value, before the uncore share (so the final die grows by value ÷ 0.77). A custom base die drops the term (§269), so a custom host reclaims it.
- Consequences inside the 0.325 rules:
  - `std_equiv_delta` (card-level thermal precedent) now adds the controller back, consistent with "the precedent area counts the PHY / controller ring".
  - The DSE interposer limit and the reticle check see the larger std die.

### §272.3 Why the default stays off
The bottom-up area is SRAM + MAC + PHY + host IO + 23% uncore. The uncore share is TPUv1's control 2% plus an unlabelled 21%. TPUv1's DDR3 controllers are not broken out in its floor plan, so part of a controller may already sit inside that 21%. Adding the controller to every std die could double count, and the source is a 2× range. The preset is therefore opt-in (precedent: §269 custom $/GB, upper end inside the option only). The eval's custom block and the design report flag the omission: "控制器面积未计", with the stacks × 7.15–14.3 mm² range.

### §272.4 Effect (8 × HBM4 10 Gb/s, 1P qwen3-8b decode b64)
- **Eval** (n5_cowos_s): std die 820 → 895 / 969 mm²; $/Mtok std 9.7658 → 9.8969 / 10.0395 m$ against custom 9.8678 (unchanged). The cheaper option flips std → custom already at 5%.
- **DSE** (n5_cowos_l, air_oam): the std monolithic MAC×1 die passes the 858 mm² reticle without the preset and fails with it. The verdict goes mono → only_split for qwen3-8b decode / prefill and qwen3-30b-a3b decode, while custom stays mono.

### §272.5 Consistency fix
`hbm_base.NOT_MODELLED` no longer lists organic SPHBM4 (modelled in §271), and the custom shoreline note points to `hbm_pkg=sphbm4`.

## §273 HBM4 / HBM4E memory-controller area on by default (0.328)

### §273.1 The double-count question (§272.3), answered from floor plans
The uncore share (§160) copies TPUv1's control 2% + unlabelled 21%. The question was whether that 21% already holds memory-controller logic. Public, annotated Google floor plans and the NeuroMeter validation paper say it does not:

| Design | What the floor plan labels | Memory port / controller | Unknown / wiring |
|---|---|---|---|
| TPUv1, 28 nm, < 331 mm² (Jouppi ISCA 2017 Fig. 2) 「来源」 | DRAM port 3% × 2, PCIe 3%, Host Interf. 2%, Misc I/O 1%, Control 2%, UB 29%, MMU 24%, Acc 6%, Act 6% | DRAM port 6% (own block) | 21% unlabelled |
| TPUv1 per NeuroMeter (Tang et al. HPCA 2021 §IV) 「来源」 | "DRAM port (6.0% v.s. 2.8%)", "do not model host interface, controller, and misc I/O, with 5% in total" | 6.0% | "unknown components … ∼21%" |
| TPUv2, 16 nm, < 611 mm² (NeuroMeter §IV; Norrie et al. IEEE Micro 2021 Fig. 4) 「来源」 | 4 × "HBM port"; PCIe drawn as "PCIe Link" + "PCIe ctrl"; no HBM controller block | "HBM ports (9% v.s 5%)" | "unknown components (which probably includes the inter-component interconnection) … ∼21%"; white areas "filled with wiring" |
| TPUv4i, 7 nm, < 400 mm² (Jouppi et al. ISCA 2021 Fig. 6, Fig. 5 lists "HBM Controller (HBMC)") | 2 × "HBMC & SerDes", OCI, CMEM 28%, ICR, LST, PCIe, UHI, MGR | 6.6% by pixel count 「推导」 (CMEM measures 27.4% vs caption 28%) | OCI 20.5% (pixels; the caption says it is stretched to fill) |

- TPUv2 draws its PCIe controller as a separate block but no HBM controller block, so the HBM controller is inside the "HBM port" 「推导」. TPUv4i names it explicitly ("HBMC & SerDes").
- In every case the memory port, with its controller, is a labelled block **outside** the ~21% unknown / interconnect share. The model's 21% is that share, so a controller term is not a double count 「推导」.
- The model's PHY term is the memory-side PHY: Samsung HC 2026 slides 10 / 13 draw "HBM Ctrl" and "HBM PHY" as separate xPU blocks for standard HBM, and slide 11 gives the PHY alone 「二手」.

### §273.2 Value
- 7.15 mm² per stack = 0.05 × 858 ÷ 6 (Samsung HC 2026 "about 5-10% of the SoC area" × its reference xPU, §272.1) 「二手/推导」, the low end.
- Cross-check from Google's numbers 「推导/假设」:
  - TPUv2: 9% × < 611 mm² ÷ 4 ≈ ≤ 13.7 mm² per HBM2 port (PHY + controller). Samsung's HBM2 memory-side PHY 6 × 1.2 = 7.2 mm² (slide 11, 「二手」), taking host PHY ≈ memory-side PHY 「假设」, leaves ≤ 6.5 mm² of controller per HBM2 stack at 16 nm.
  - TPUv4i: 6.6% × < 400 mm² ÷ 2 stacks (614 GB/s 「推导」) ≈ ≤ 13.3 mm² per HBM2(E) stack, PHY + controller, at 7 nm.
  - Scaling to HBM4: the controller grows with channels, 8 per HBM2 / 2E stack → 32 per HBM4 stack (4×, JEDEC 「来源」), and shrinks N7 → N5 by the n5_derived ÷ 1.8 rule 「假设」. From TPUv4i: controller ≤ 13.3 − ~7.2 (PHY, as above) ≈ 6.1 mm² → × 4 ÷ 1.8 ≈ ≤ 13.6 mm² per HBM4 stack 「推导」. That bound sits inside Samsung's 7.15–14.3 and does not single out a point, so the low end is used as the default and 14.3 stays a credible ceiling (`mc_preset=samsung_hc2026_hi`). (No 16 nm → 5 nm factor is sourced in the model, so TPUv2's ≤ 6.5 mm² is not scaled.)
- Not tuned: no validation row or DES result was consulted; the value is Samsung's low end unchanged.

### §273.3 Scope: HBM4 / HBM4E only (old → new: any HBM)
- For HBM4 / 4E the PHY term `mm2_per_dq_hbm` × 2048 = 32 mm² is Samsung's own HBM4 PHY (8 × 4 mm), so PHY + controller is the complete host interface.
- For HBM3 / 3E the same per-DQ value gives 16 mm² per 1024-DQ stack, while Samsung's HBM3 PHY is 8 × 3 = 24 mm². The PHY error (−8 mm²) exceeds the controller (7.15). Adding the controller alone would correct the smaller half of the error with a value sourced for HBM4-era xPUs. Deferred: per-generation PHY + controller together (pending). HBM3 / 3E carry the note "只用于 HBM4 / HBM4E 堆栈（0.328）…PHY 项按代修正前暂缓".
- HBM2 / 2E are not in the memory catalog. For them the PHY term (16 mm²) already exceeds Google's measured PHY + controller (≤ 13.3–13.7 mm²): another reason not to stack a controller on that term.
- Only `n5_derived` carries the default (Samsung's xPU is a 4–5 nm reticle-class die). `n7_tpuv4i` / `n28_tpuv1` stay without it: no node scaling invented. The auto memory-edge proxy (`auto_area_budget`, n5_derived) follows.

### §273.4 Precedence
`mm2_per_unit_mc` (generic) > explicit `mm2_per_hbm_mc` > `mc_preset` > density preset default. `_budget` now applies `mc_preset` before `density_preset` (both fill unset fields only). `mm2_per_hbm_mc: 0` reproduces 0.327.

### §273.5 Effect (defaults, old → new)
- **Eval** 1P qwen3-8b decode b64, 8 × HBM4 10 Gb/s, n5_derived, n5_cowos_s: std die 820.4 → 894.7 mm²; $/Mtok 9.7658 → 9.8969 m$; custom 535.5 mm², 9.8678 m$. The cheaper option flips std → custom by default.
- **DSE / hbm_base study** (n5_cowos_l, air_oam): std MAC×1 mono 820 → 895 mm², over the 858 mm² reticle → verdict mono → only_split for qwen3-8b decode / prefill and qwen3-30b-a3b decode. Custom stays mono. Std MAC×2 dies (1074 → 1149 mm²) get 3 instead of 4 auto memory edges (the fewest-edges rule on a longer perimeter), so their throughput / J move (qwen3-8b decode 13758 → 13290).
- **Unchanged:** HBM3 / 3E / LPDDR / DDR numbers; n7 / n28 presets.

### §273.6 Citation fix
`_UNCORE.mm2_host_io` cited "NeuroMeter (Shao et al., MICRO 2021)". The paper is Tang, Li, Nai, Jouppi, Xie, HPCA 2021 (industry track). Its 1.8% PCIe is NeuroMeter's modelled value; the published TPUv1 share is 3.0% (Fig. 2). The value (6.0 mm²) is unchanged; a modern PCIe 5/6 × 16 PHY + controller source is pending.

### §273.7 Not modelled / pending
- Per-generation HBM PHY area (HBM3 24 mm², HBM3E, HBM4 32 mm² per Samsung slide 11) with a matching per-generation controller.
- Controller area at N7 / N28 (no node scaling invented).
- Host-IO (PCIe) area from a modern source.
- Controller power (the controller's pJ/bit is inside the device / PHY energy terms, not split).

## §274 Per-generation HBM host interface; host-IO reconciliation (0.329)

### §274.1 PHY by generation
Samsung Hot Chips 2026 (S. Han) slide 11 「来源」 (ServeTheHome slide image) gives HBM PHY footprints: HBM2 6 × 1.2 mm, HBM3 8 × 3 mm, sHBM4 8 × 4 mm, cHBM4 D2D 8.5 × 1.5 mm, sHBM5 9.5 × 1.7 mm. The footnote reads "PHY area is based on the bump map". Slide 10 draws the xPU-side PHY mirroring the base-die PHY. The footprint is set by the microbump map (pitch × DQ / CA count), not by the logic node, so the host side equals the memory side 「推导」. That holds for HBM3, whose base die is a 1x-nm DRAM process, and for HBM4 (4 nm logic).
- `mm2_per_dq_hbm3` = 8 × 3 / 1024 = 0.0234375 mm²/DQ 「来源/推导」 is used for HBM3 / HBM3E (24 mm² per stack; old 16, from the HBM4 per-DQ value). HBM3E is taken as the HBM3 bump map 「假设」 (same 1024 DQ / 16 channels).
- `mm2_per_dq_hbm` = 8 × 4 / 2048 is used for HBM4 / HBM4E (32 mm² per stack, unchanged).
- Both fields are in all three density presets (node-independent). `hbm_dq_field(b, type)` picks the field; an unset `mm2_per_dq_hbm3` falls back to `mm2_per_dq_hbm`. Users: `budget.full_area`, `hbm_base.report` (std PHY), `insight._density`, `report.hbm_phy`.

### §274.2 Controller by channel count
`mm2_per_hbm_mc` (n5_derived 7.15, §273) is per HBM4 stack (32 channels, JESD270-4). Controllers are instantiated per channel group: NVIDIA GH100 has "12 512-bit memory controllers" for 6 HBM3 stacks 「来源」. So a stack gets `mm2_per_hbm_mc × channels / 32`, with channels HBM3 16 (JESD238), HBM3E 16, HBM4 32 「来源」, HBM4E 32 「假设」. HBM3 / 3E: 3.575 mm² 「推导」.
- `MC_TYPES` = HBM3 / 3E / 4 / 4E (0.328: HBM4 / 4E). The generic `mm2_per_unit_mc` stays unscaled and wins. n7 / n28 carry no controller (no node scaling invented).

### §274.3 Floor-plan check 「推导」
- TPUv4i (ISCA 2021 Fig. 6): "HBMC & SerDes" ≤ 13.3 mm² per HBM2 port at N7. Samsung HBM2 PHY 7.2 + controller 8 channels × 7.15 / 32 = 1.79 at N5 (× 1.8 to N7 「假设」, 3.2) = 10.4 ≤ 13.3: consistent. Bus-width scaling (1024 / 2048 × 7.15 = 3.6 at N5) fits less well.
- No public HBM3 TPU floor plan exists, so HBM3 is not checked directly. H100 die-shot estimates (≈ 24 vs ≈ 14 mm² per stack) conflict and are unsourced: not used.

### §274.4 Host IO
`mm2_host_io` = 1.8% × 331 = 5.958 mm² (二手) is kept. A VLSI 2022 5 nm DSP SerDes for Ethernet / PCIe (doi 10.1109/VLSITechnologyandCir46769.2022.9830304) measures 1806 × 825 µm per quad = 0.3725 mm² per lane 「来源」 → x16 = 5.96 mm² 「推导」 (within 0.04%). That is an upper bound for a PCIe 6 / CXL 3 lane (64 GT/s vs 112 Gb/s). CXL rides on the PCIe 6 PHY (Intel 3 VLSI 2024 transceiver, no area). The PCIe / CXL controller is not counted: TPUv2 "PCIe Link + PCIe Ctrl" 2% of < 611 mm² ≈ ≤ 12.2 mm² at 16 nm 「来源/推导」, with no N5 figure.

### §274.5 Effect (defaults, old → new)
- 8 × HBM3E 1P die 654.2 → 774.5 mm² (+(64 + 28.6) / 0.77); 4 × HBM3E RWKV 571 → 631 mm²; J unchanged.
- Shoreline I/O: tp8 / 900 GB/s scale-up (100 G / 200 G lanes) goes from over to buildable; 400 GB/s at 50 G goes to feasible.
- Air thermal: 8 × HBM3E prefill flux 0.99 → 0.85 W/mm², now feasible.
- Chiplet: mono / 4-die cost crossover D0 (0.1, 0.2) → ≈ 0.07; at D0 0.1, 4-die $2232.9 < mono $2354.5.
- tco_sens: D0 threshold 0.150 → 0.110; wafer $ and carrier yield also flip; tp8 / 400 / MAC×2 TCO memory 4-stack → 8-stack.
- HBM4 / 4E, LPDDR / DDR unchanged.

### §274.6 Not modelled / pending
- PCIe / CXL controller area.
- HBM controller area at N7 / N28.
- HBM3E's own bump map (assumed = HBM3).
- Controller power split.
- HBM2 / 2E (not in the catalog; Samsung 7.2 mm² PHY would apply).
- PixText (new image DiT, needs a domain builder).
- OrcaCyber Nano 1.0.

## §275 Adversarial review of the 0.329 flips; PixText-MeanFlow (0.330)

### §275.1 HBM3E PHY = HBM3 bump map 「推导」 (was 「假设」; value unchanged)
- JEDEC publishes no separate HBM3E standard: HBM3 = JESD238.
- Synopsys HBM3 PHY IP: "JEDEC standard HBM3 DRAMs with data rates up to 9600 Mbps", "16 independent 64-bit memory channels", micro-bump array 「来源」. So HBM3E speeds run on the HBM3 interface and bump map.
- NVIDIA: H200 = Hopper with HBM3e; the GH100 whitepaper lists HBM3 controllers 「来源」; H200 uses the same die 「二手」.
- AMD CDNA 3: 16 channels per stack; MI300X HBM3 → MI325X HBM3E 「来源」.
- No public measured HBM3E PHY area (B200 / MI325X die shots unannotated; Gaudi 3 = HBM2e). An unsourced "GH100 ≈ 120 mm² for 5 PHYs" blog figure is not used.

### §275.2 Channels and controller
- JEDEC JESD270-4 release (2025-04-16): "doubles the number of independent channels per stack, from 16 channels (HBM3) to 32" 「来源」. The HBM3 / 3E scale is therefore 16 / 32.
- The SK hynix IEDM 2023 "16 Ch × 128 I/O" HBM4 figure was pre-standard.
- Residual: Samsung slide 9's 5–10% SoC quote uses a generic 6-stack xPU (26 × 33 mm, no generation printed) in an HBM4 custom-base-die context 「推导」. Were it an HBM3E xPU, HBM3E would carry 7.15 mm² per stack. Range 3.575–7.15; value kept at 3.575.

### §275.3 No double count 「推导」
- `full_area` grows by exactly stacks/die × (8 + 3.575) / 0.77 per die: 15.03 mm² per stack. Verified for mono, 2-die and 4-die at MAC×1 and ×2 (test_core_330).
- The shoreline uses `HBM_STACK_EDGE_MM` only, not PHY area. D2D is multi-die and custom HBM4 only.
- The DRAM port is outside the 21% unknown (§273). The ÷ (1 − 0.23) markup applies the uncore share proportionally to the whole die, ports included, as in the TPUv1 fraction 「假设」. This is a convention, not a second PHY.

### §275.4 Each flip from the increment alone
- **Cost (D0 0.1, MAC×1):** mono +$238.7 ($2115.8 → $2354.5), 4-die +$115.2 ($2117.7 → $2232.9). Yield is super-linear in die area, so the mono / 4-die crossover moves to D0 ≈ 0.07.
- **I/O (tp8):** lane needs are unchanged. 900 GB/s fits once the die grows (paired old / new budget test); 1200 GB/s needs 31.9 mm vs 26.3 mm left and stays over. Square-die perimeter gain ≈ the deeper-PHY geometry 「推导」.
- **Air thermal:** limit 0.86 W/mm² = H100 / H200 700 W ÷ 814 mm² of the *whole* die (PHYs included), the same definition as `full_area`.
  - 8 × HBM3E prefill flux 0.99 → 0.85.
  - air_oam card front [4, 17] → [1, 0, 15] with thr and J unchanged: only feasibility changed.
  - Rows 0 / 15 sit at 0.858 vs 0.86 (0.2% margin): fragile.
- **TCO memory (tp8 / 400 / D0 0.1, MAC×2): 4s → 8s is a reticle crossing.** The 4-stack mono die goes 838.9 → 899.0 mm² > 858, so only split rows remain: 4s $/M 0.01164 → 0.01966; 8s 0.01641 → 0.01681. The 0.329 test comment ("thr edge over cost") was wrong and is corrected.
  - MAC×1 also moved 8s → 6s, within the 1% tie (0.00949 vs 0.00952). Not reported in 0.329; recorded now.
- **tco_sens:** D0 threshold 0.150 → 0.110, and the wafer / carrier-yield flips follow from the narrower mono / split margin.
- **eval = DSE = report = static:** same die mm² and $/M; report hbm_phy × stacks = dram_phy and hbm_mc × stacks = mem_ctrl. Static is the same Python under pyodide.

### §275.5 PixText-MeanFlow (ATH-MaaS/PixText, Apache-2.0) 「来源」 unless tagged
**Release data**
- Config `models/PixText-MeanFlow/config.json`: hidden 1280, depth 32, 16 heads, mlp_ratio 4, patch 16, in_channels 3, axes_dims [12, 34, 34], interval_conditioning true.
- Header: 465 BF16 tensors, 953,948,480 params (EMA). `summarize_release.domain_of` recognises the format by `model.image_attention_routing`.

**Per block** (`build_pixtext`)

| Tensor | Shape | Rows |
|---|---|---|
| qkv | 1280 → 3840 (+bias) | tok |
| proj | 1280 → 1280 | tok |
| adaLN_modulation | 1280 → 7680 | tok (conditioning_ids = arange(tokens); role cond) |
| SwiGLU w12 | 1280 → 6826 | tok (hidden int(5120 · 2/3) = 3413) |
| w3 | 3413 → 1280 | tok |

- RMS QK-norm(80); RoPE over the whole head (12 + 34 + 34 = 80); bidirectional.

**io**

| Tensor | Shape | Rows |
|---|---|---|
| patch_projection | 768 → 1280 | img |
| glyph_projection | 256 → 1280 | txt |
| timestep_embedder | 256 → 1280 → 1280 | tok |
| interval_embedder | 256 → 1280 → 1280 | tok, twice: f(r) − f(0); the second pair is compute only, subtracted from io_misc |
| final adaLN | 1280 → 2560 | seq |
| final linear | 1280 → 768 | img |

**Workload** (release reference case examples/pipeline_case, pass 1)
- main 1024² → 4096 tokens (vae_s 1, patch 16);
- 12 items: crops 11 × 64² + 64 × 80 → 196 RGB + 196 glyph = 392 prefix tokens;
- sequence 4488;
- 7 forwards (4 MeanFlow steps + guidance-branch forwards at c = 0.25 / 0.5 / 0.75 inside (0.1, 1.0)) = steps 7, cfg 1.
- More than 12 items run as further sequential passes (13 items = 2 passes).

**Approximations 「近似」**
- Dense attention vs the routing mask (main → main + items; items → main + items + own glyph; glyph → own glyph + own item): 18,430,912 vs 20,142,144 pairs, +9.3% ≈ +2.6% FLOP.
- Guidance branch counted at the conditional shape (4292 → 4488) ≈ +1.9%.
- Item RGB rows counted on the glyph projection ≈ 0.005%.
- Glyph rasterisation (CPU) not timed.

**Checks:** 11.84 TFLOP per forward; 82.9 TFLOP per request; parameters exact.

**Pending:** PixText-Teacher (25 Euler steps → 47 forwards 「推导」, 951,979,840 params, no interval embedder) needs a per-subfolder release id.

### §275.6 Not modelled / pending
- PCIe / CXL controller area; HBM controller at N7 / N28; controller power split.
- HBM3E-specific PHY measurement (none public).
- Partial guidance as a workload field: PixText counts forwards as steps.
- PixText-Teacher.
- OrcaCyber Nano 1.0 (drafter-only GGUF metadata, config 401).

## §276 Release keys; partial guidance; PixText-Teacher; front thermal sensitivity (0.331)

### §276.1 One repo, several release files
- A catalogued model reads `releases/<key>.json` with key = `catalog.RELEASE_KEYS.get(id, hf_id)`.
  - Key form `"<repo>:<subfolder>"` → `<repo /→__>_<subfolder /→__>.json` (`model.release_path`, unchanged).
  - hf_id stays the public repo, used for links, `provider_of` and the freshness org list. The keyed model's alias is its key, so the bare repo id still resolves to the first (default) checkpoint.
- Fetch spec `repo:sub!` → own file (`scripts/fetch_hf_release.py`). The cache dir and `source.json` carry `release_key`; summarize reads it back.

### §276.2 Partial guidance (`NativeWorkload.cfg_interval`) 「来源」 mechanism
- With guidance scale > 1 on an interval (lo, hi), the negative branch runs only on steps k with lo < k/steps < hi.
  - Strict inequalities: PixText `sampling/flow.py` `_guided_prediction` and `sampling/meanflow.py` `_guided_interval_prediction`. The pipeline passes `guidance_interval` with no progress grid, so progress = k/steps.
  - The limited-interval method is Kynkäänniemi et al., NeurIPS 2024.
- `guided_steps(steps, (lo, hi))`; an empty interval means every step, the old behaviour for every other model.
- Two-forward CFG (§ 0.258):
  - TFLOP per request = cond × steps + uncond × guided;
  - denoise time = steps × t_cond + guided × t_uncond;
  - energy counts the uncond stages guided times.
- The packed symmetric path (`cfg_branches = same`, or the PP > 1 / micro-batch / step-skip fallback) counts every step, with a warning.

### §276.3 PixText fidelity, old → new
| | ≤ 0.330 | 0.331 |
|---|---|---|
| MeanFlow workload | steps 7, cfg 1 | steps 4, cfg 2, interval (0.1, 1.0) → 4 + 3 = 7 forwards |
| negative branch shape | 4488 tokens (cond) 「近似」 | 4292 tokens (main 4096 + items 196, no glyphs) 「来源」 |
| per forward | 11.84 TF | cond 11.84 / uncond 11.19 TF |
| TFLOP / request | 82.879 | 80.916 (−2.37%) |
| default-chip latency | 0.935 s | 0.907 s |
- **Routing mask: still dense 「近似」 on the conditional branch** (18,430,912 allowed vs 20,142,144 pairs, +9.3%).
  - Rule: main → main + items; items → main + items + own glyph; glyph → own glyph + own item.
  - The glyph-row and own-glyph terms scale with Σ nᵢ² over items. The workload has no per-item sizes, so modelling it needs a PixText-only rule, which is out of scope.
  - The no-glyph branch has no glyph rows: main + items are mutually dense, so it is exact.

### §276.4 PixText-Teacher
- `ATH-MaaS/PixText:models/PixText-Teacher`: 461 BF16 tensors, 951,979,840 params = builder, exactly.
  - Same GroupJiT as MeanFlow without the interval embedder (−1,968,640 params).
- 25 Euler steps; guidance 3.3 on (0.1, 1.0) → k = 3…24 guided = 22 → 47 forwards 「推导」.
- Result: 540.45 TFLOP per request; 6.056 s on the default chip.

### §276.5 Design report: front thermal sensitivity
- Thermal items (「推导」):
  - flux margin = limit − flux (W/mm²);
  - headroom = limit / flux − 1 (power increase at fixed area that reaches the limit).
- `front_thermal` lists each perf/W front point; points with margin ≤ the DSE tie (1%) are flagged and warned. The limit is not retuned.
- air_oam (0.86 W/mm², 700 W / 814 mm² GH100 「推导」 0.315), qwen3-8b prefill tp, card scope:
  - row 0: 0.8576 (margin 0.0024 W/mm², 0.28%);
  - row 15: 0.8581 (0.0019, 0.23%);
  - row 1: 0.8509 (1.06%).
  - A +0.3% power or −0.3% area change removes rows 0 / 15 from the front.

### §276.7 Suite freeze clock (test infrastructure)
- `scripts/freeze_clock.py`: excused per stall gap = gap − max(period, CPU in the gap). It was gap − period.
  - So active ≥ CPU executed. A starved-but-running process no longer has its executed CPU excused (8× oversubscription: GIL call active 0.24 → 1.61 s for 1.41 s CPU).
  - Thresholds unchanged 「假设」.
- The logic is a pure `observe(w, c)` on injectable clocks; test_core_316 checks it on a fake clock.

### §276.6 Not modelled / pending
- PixText routing mask (above); glyph rasterisation time.
- Guidance interval on the packed / PP > 1 fallback path.
- PCIe / CXL controller area; HBM controller at N7 / N28; controller power.
- OrcaCyber Nano 1.0 (0.331 retry: sha 1af1635, drafter-only GGUF metadata, config 401).


## §277 Load-robust runtime guards; decision timers; NPU pending audit (0.332)

### §277.1 Work meter (test infrastructure)
- The quantity a runtime guard bounds is work per request: exploding loops, the 22-min regressions. Wall and CPU seconds are proxies that inflate under load. rt278b measured 9.5× the CPU seconds for identical work at -j 8 「来源」 (AGENTS §229).
- `tests/workmeter.py` counts PY_START + CALL events in `accel_dse` code (sys.monitoring, Python 3.13.5). The count is exactly deterministic for a code path from a given cache state 「来源」 (measured: repeat runs equal; 8 hogs on 8 cores, 15/15 labels equal).
- Budget per guard = bound_s × r_label, where r_label = calls / unmetered CPU s on the quiet box (2026-10-22, single process), rounded down to 2 significant figures 「近似」. So calls ≤ budget ⇔ quiet CPU ≤ bound: the pre-0.332 threshold, never looser.

| guard | quiet CPU s / bound | rate M/s |
|---|---|---|
| 237 report vllm | 57.70 / 90 | 7.9 |
| 237 phase-law vllm | 29.39 / 45 | 8.9 |
| 238 vl6000 / vl4096 / mx4096 | 16.15 / 16.06 / 10.61 of 60 | 8.5 / 10 / 8.7 |
| 210 fuzz 13 sensitivity / 6 fit | 7.13 / 5.86 of 80 | 4.8 / 6.8 |
| 289 protenix / boltz-1 | 2.25 / 0.91 of 20 | 2.3 / 3.6 |
| 099 wan2.1-14b | 0.094 / 20 | 11 |

- Rates span 2.3–18.8 M/s (call-heavy vs NumPy-free arithmetic paths), so no single rate is equivalent. Blocks under 0.05 s CPU use the floor of 2.3 M/s 「假设」: their rate is timer noise, and the floor is the strict side.
- Meter overhead is 1.6–1.8× CPU inside guarded blocks. Budgets are in calls, so the overhead does not consume headroom.

### §277.2 Timers
- Decision timeouts in scripts and `validation.elapsed_s` use `time.monotonic()`; `time.time()` can step.
- Audit: no other wall/CPU mixing in the package (freeze_clock fixed in §276.7). Pyodide worker `ms` and shim `wallMs` are both `performance.now()` and labelled separately.

### §277.3 NPU item (2): skipped — no public numbers
- PCIe / CXL controller area: vendor IP pages give no mm².
- HBM controller at N7 / N28 and controller power: nothing public found in 2026.
- SPHBM4 PHY: JESD330-4 describes the interface only; already a package option (0.326).
- Near-memory compute: HC 2026 talks give no TOPS/W.
- Candidate for decision, not applied: arXiv 2610.14244 (Olgun et al., MICRO 2026) measures H200 HBM3E memory power at 248 W for 4.52 TB/s → ≈ 6.86 pJ/bit all-in 「推导」, vs `pJ_bit_dram_hbm` 3.44.
  - Its HBM4 projection is 6.24–6.73 pJ/bit vs the model's 2.46.
  - The paper's all-in figure includes idle / refresh (Fig. 22 split is figure-only). Adopting it changes a core energy default ~2×, so it needs the user's decision and a data-vs-I/O split.

### §277.4 Not modelled / pending (carried)
- PixText routing mask.
- PCIe / CXL controller area; HBM controller at N7 / N28; controller power; SPHBM4 PHY area.
- Near-memory compute.
- HBM all-in energy (§277.3).
- OrcaCyber Nano 1.0 (0.332 retry: sha 1af1635 unchanged, drafter GGUF only, config 401).

## §278 HBM energy reconciliation against measured H200 HBM3E (0.333)

Source: Olgun et al. (ETH), "Understanding the Power Consumption of Modern High Bandwidth Memory", arXiv 2610.14244 (MICRO 2026), §6.2, plus its released data (github.com/CMU-SAFARI/HBM-Power, `case_studies/hbm3e_validation/data`). Every number below is read from the released data tables 「来源」 or derived from them 「推导」. No plot was digitized.

### §278.1 What each side covers
- **Model ≤ 0.332, per bit** (`pJ_bit_dram_hbm` 3.44): the memory device (array, TSV, base-die I/O, memory-side PHY) at a vendor-quoted operating point. Host PHY is a separate term (`pJ_bit_phy_hbm` 0.29, Chae ISSCC 2023). The controller is area only (§273). There is no activation / row-locality term.
- **Model ≤ 0.332, static** (charged only when the table has no idle figure; the tpuv4i_7nm preset charges it): 48 mW per die standby (DRAMsim3 HBM2 IDD2N × 1.2 V) + 1.95 mW/Gb refresh.
- **Paper**: nvidia-smi GPU memory power (treated as device-side 「假设」; supported by its device-only model Ayna matching the read workload within 2.4 %). It reports:
  - idle = NOP + refresh with clocks running;
  - a streaming read microbenchmark, total = idle + activation + read data movement + I/O.
  - The quoted ≈ 6.86 pJ/bit is total ÷ bandwidth, i.e. it includes idle.
  - Power delivery and the host PHY are outside nvidia-smi's memory rail 「假设」.

### §278.2 Measured numbers (H200, 6 × HBM3E 8-high 24 Gb, 96 64-bit channels; uniform-random data) 「来源」

| unit | idle W | read total W | read GB/s | (total − idle) / BW pJ/bit 「推导」 |
|---|---|---|---|---|
| 1 | 35.55 | 247.75 | 4520.6 | 5.87 |
| 2 | 34.75 | 241.71 | 4517.3 | 5.73 |
| 3 | 44.95 | 255.22 | 4523.3 | 5.81 |
| mean | 38.42 | | | 5.80 |

- Const-0 data read on the same H200: 226.70 W total, 42.24 W idle → 5.09 pJ/bit. Random data adds +14 %.
- Write, const-0: 189.30 W total, 44.86 W idle, 4327.5 GB/s → ≈ 4.17 pJ/bit.

### §278.3 Internal composition (Ayna HBM3E config, 192 pseudo-channels, VDD 1.1 V, trace = 4 reads per ACT + REFab every tREFI) 「推导」

| component | per-PC / per-bit | H200 total |
|---|---|---|
| background (IDD2N/3N) | ≈ 156 mW per PC | ≈ 30.1 W |
| refresh (IDD5B, tRFC 350 / tREFI 3900 ns) | 9.02 mW per PC | 1.73 W |
| activation (1772 pJ per ACT ÷ 1024 bits) | 1.73 pJ/bit | ≈ 62 W |
| read data movement (IDD4R, data-pattern scale 1.588) | 4.17 pJ/bit | ≈ 150 W |
| sum | | ≈ 244 W (measured 248) |

- Ayna's idle prediction is 12–20 % below the measured idle. The model takes the measured idle, not Ayna's.
- The activation share is set by the benchmark's locality (4 reads per ACT). A gather-heavy access stream would pay more per bit; this is not modelled (§278.6).

### §278.4 Adopted (mechanism → term)
- **Dynamic**: `pJ_bit_dram_hbm` 3.44 → 5.8017 「推导」 = measured (total − idle) ÷ BW.
  - Why 3.44 was low: vendor figure at standard-loop data, no real access-stream activations, and the paper's measured HBM2 read current is 1.78× FGDRAM's implied value.
  - FGDRAM (O'Connor MICRO 2017) cross-check: HBM2 ≈ 3.97 pJ/bit, 909 pJ per activation.
- **HBM4**: `pJ_bit_dram_hbm4` = HBM3E ÷ 1.4, 2.457 → 4.144 (vendor "≈ 40 % better" ratio kept).
  - Sensitivity: the paper's voltage-only projection is 0.94–1.00× HBM3E per bit, i.e. 5.45–5.80 pJ/bit. It assumes no process gain, so it is an upper bound, not adopted.
- **Static**: new `mW_hbm_ch_standby` 376.8 mW per 64-bit channel 「推导」 = (38.42 − 2.2464 refresh) ÷ 96.
  - Per channel, not per die: standby scales with data rate (Finding 6), i.e. it is set by the clocked interface and peripherals, not by the number of core dies. Per-die × height would give 77 W for HBM4 8 × 12-high.
  - HBM4 cross-check: 32 channels per stack → 12.06 W per stack vs the paper's extrapolation IDD2N 176.2 mA/PC × 1.05 V × 64 PC = 11.84 W (not fitted).
  - No data-rate scaling is applied to other parts 「假设」.
  - The preset drops `mW_hbm_die_standby`; the field stays as a user input and sums with the channel form.
- **Refresh**: unchanged at 1.95 mW/Gb (Ayna's config gives 2.26).
- **By construction**: the H200-like hbm3e_6s card's HBM static equals the measured idle mean (38.417 W), and its per-bit term equals the measured active energy. test_core_333 asserts this reproduction, not a validation.

### §278.5 Package floor (mechanism fix)
- `thermal.package_power` adds `max(0, STATIC_FLOOR_W_MM2 × area × VR_EFF − static already charged)`.
- The floor comes from H100 / MI300X idle with no workload at idle clocks, which contains HBM self-refresh. Refresh and SRAM / NoC leakage overlap it; **clocked HBM standby does not**. Clocked standby is also not die leakage, so it does not scale with die temperature (`pkg_W_hot`).
- The new `standby_W` argument (from `powercap.hbm_standby_W`) is excluded from both. All callers pass it: cap function, cap inverse, thermal report, energy power cap, TCO, energy DSE rows, api.
- Without this, 8 × HBM3E standby (48 W) filled most of the floor and die area stopped affecting package power. test_core_327's custom-base-die decision flipped for that reason only; with the fix it holds as in 0.332.

### §278.6 Not modelled / pending
- Activation / row-locality term: the per-bit value is at a streaming read's locality, so gathers would pay more.
- Standby data-rate scaling across HBM speed grades.
- Write energy (≈ 4.2 vs read 5.8 pJ/bit) is not split out.
- LPDDR `pJ_bit_dram_lpddr` 4.0 is a Grace all-in subsystem figure (16 W at 500 GB/s), a different scope from the HBM device-dynamic 5.80. An LPDDR measured split is pending; test_core_186 now pins the per-bit ratio and the total-J ordering.
- nvidia-smi memory-rail scope (device only vs including PHY / VR) 「假设」.
- HBM4 measured data.
- Temperature dependence (paper: +6.7 mA/°C).
- SC25 NERSC GH200 "HBM datapath 2.99 + control 8.69 pJ/bit" is GPU-level and not decomposable into this model's scope; it is a cross-check only.
- Carried: PixText routing mask; PCIe / CXL controller area; HBM controller at N7 / N28 and controller power; SPHBM4 PHY area; near-memory compute; OrcaCyber Nano 1.0 (0.333 retry: sha 1af1635 unchanged, config 401).

## §279 Adversarial review of the 0.333 HBM energy change (0.334)

### §279.1 Double-count check
- Idle term = measured H200 idle (NOP + refresh) → `mW_hbm_ch_standby` + refresh.
- Dynamic term = measured (total − idle) / BW → `pJ_bit_dram_hbm` 5.80.
- At the measured streaming bandwidth, idle_W + 5.80 pJ/bit × bits ≈ measured total (within ~3 % across the three units). So standby is not inside the per-bit term, and the per-bit term is not inside standby.
- Host PHY (`pJ_bit_phy_hbm` 0.29) stays separate; the paper's nvidia-smi rail is treated as device-side 「假设」.
- No separate activation charge: Ayna's ACT share is already in the 5.80.

### §279.2 Package floor vs hot leakage (0.333 bug → 0.334 fix)
- Floor measurement (H100 / MI300X idle) includes HBM self-refresh. Refresh and on-die leakage overlap it; **clocked** standby does not → `standby_W` out of the floor offset (0.333, kept).
- Hot bound scales static with die temperature (`_hot_k`). DRAM-device power sits beside the die. 0.333 removed only clocked standby from hot, leaving refresh in — wrong.
- 0.334: `device_static_W` = standby + refresh (`hbm_device_static_W`) out of hot; floor still uses clocked `standby_W` only. Two knobs, two mechanisms.
- `pkg_W` (cap / TCO) unchanged; only `pkg_W_hot` / `hot_ok` move (≈ −8 W on 8 × HBM3E).

### §279.3 HBM4 ÷ 1.4
- Still Samsung / SK hynix vendor "≈ 40 % better than HBM3E" only. No public absolute pJ/bit in 2026 shipping / product pages. Ratio kept; Olgun voltage-only 0.94–1.00× remains a sensitivity in the source text.
- Stale UI / MEM_PROXY / comment strings that still said 2.46 were updated to 4.14.

### §279.4 TCO / J mechanics
- J/token moves with DRAM bytes × ΔpJ and with the static energy over the window. $/M follows package power and throughput. No coefficient was fitted to the paper's all-in 6.86.

### §279.5 Not modelled / pending (carried)
- Activation / row-locality term; standby data-rate scaling; write per-bit; LPDDR all-in vs HBM device scope; nvidia-smi rail scope; HBM4 measured absolute; temperature (+6.7 mA/°C).
- PixText routing mask; PCIe / CXL area; N7 / N28 controller; controller power; SPHBM4 PHY; near-memory compute.
- OrcaCyber Nano 1.0 (0.334 retry: sha 1af1635, gated auto, config 401, public GGUF = drafter/mix only).

## §280 LPDDR device energy aligned with the HBM 0.333 split (0.335)

### §280.1 Gap vs HBM
- HBM 0.333 used a measured H200 idle / active split (nvidia-smi memory rail).
- LPDDR has no public counterpart: NVIDIA Grace publishes only the subsystem peak ≈ 16 W at 500 GB/s → 4.0 pJ/bit all-in (no idle figure). Micron’s GH200 brief gives multichase DRAM watts, still not an idle/active pair.

### §280.2 Adopted (datasheet mechanism, same source as standby)
- Device-side active: Micron Y52Q LPDDR5X Rev. G Table 16 WT, x16 @ 9600 Mb/s 「来源」:
  - IDD2N = (1.3, 28, 0.2, 0.6) mA on (1.8, 1.05, 0.9, 0.5) V = 32.22 mW (already `mW_lpddr_ch_standby`).
  - IDD4R = (17, 416, 0.2, 121.5) mA → 528.33 mW (IDD4RQ typical 「近似」).
  - Active = (528.33 − 32.22) mW ÷ (9600 Mb/s × 16) = **3.23 pJ/bit** 「推导」.
- LPDDR6: 3.23 ÷ 1.21 = **2.67** (Samsung ISSCC 2026 “21.0 % better than LPDDR5x”).
- Host PHY (`pJ_bit_phy_lpddr` 1.3) stays separate — Micron IDD is device-side.
- Standby / refresh / PD / SR unchanged (Micron Y52Q).

### §280.3 Cross-checks (not fitted)
- Grace all-in 4.0 ≈ 3.23 + 1.3 PHY.
- NVIDIA slide “5 pJ/bit vs DDR 36” 「二手」 — same order.
- Old default 4.0 remains in the source text as the all-in sensitivity.

### §280.4 Thermal consistency with 0.334
- LPDDR device power is off-die. `hbm_standby_W` / `hbm_device_static_W` now cover `kind == "LPDDR"` (clocked IDD2N out of the floor offset; full W out of hot). Eval / DSE / api already pass both knobs.

### §280.5 Not modelled / pending
- System-level LPDDR idle/active measurement (Grace / GH200).
- OmniASR (bezzam/omniasr-*-v2): public config, needs a new `omniasr_audio` builder.
- Carried: activation locality; HBM4 absolute; PixText mask; PCIe/CXL; controllers; NMC; OrcaCyber Nano.

## §281 OmniASR LLM v2 Wav2Vec2-style audio encoder (0.336)
- **Source:** bezzam/omniasr-llm-{300m,1b,3b,7b}-v2 `config.json` + safetensors headers 「来源」(2026-10-22); facebook/omniASR-* official checkpoints are `.pt` without HF config — not used.
- **Audio tower** (`audio_config.model_type = omniasr_audio`, kind `w2v`):
  - Feature extractor: 7 Conv1d + LayerNorm (conv_dim / conv_kernel / conv_stride) → LN → Linear → `hidden_size`.
  - Grouped positional Conv1d (`num_conv_pos_embeddings` × `num_conv_pos_embedding_groups`).
  - `num_hidden_layers` pre-LN blocks (q/k/v/o + GELU FF, all with bias) + final LN.
  - LLM: `multi_modal_projector` Linear `hidden_size` → `text_config.hidden_size` (bias).
  - Output length: Wav2Vec2 `(L − k) // s + 1` per layer; Π stride = 320 → 50 Hz at 16 kHz.
- **Text:** `text_config` Llama (shared across sizes: h=4096, L=12, nh=8, head_dim=512, ff=2816, vocab=11984) 「来源」.
- **Param check:** builder `params` = `params_vision` (incl. projector) for every released size; `params_llm` matches the Llama trunk.
- **CTC (`omniasr_ctc`, 0.395):** same audio tower plus `ctc_head` Linear(hidden, vocab, bias) on every encoder frame. No Llama decoder. Catalogued as encoder-only ASR (`omniasr-ctc-*-v2`).
- **0.335 LPDDR:** no change; adversarial pass reconfirmed device 3.23 + PHY 1.3, HBM untouched.

## §282 OmniASR w2v as-released dtype; Carbon-X; speech pending (0.337)

### §282.1 OmniASR adversarial (shapes / dtype / workloads)
- Shapes, catalog dims, eval speech, DSE, and static paths for the four bezzam OmniASR LLM v2 ports check out (same builder as 0.336).
- **As-released weight dtype gap (fixed):** bezzam OmniASR safetensors / release `roles.vision` are **fp32**; `_w2v_ops` had hard-coded `w_bits=16` / `w_fmt="bf16"` (same default as Whisper / AuT / SAN-M audio paths) → understated encoder weight traffic by 2×.
  - `AudioSpec.w_bits` / `w_fmt` (default 16 / bf16) 「来源」 for as-released accounting.
  - `_omniasr_audio_spec` sets 32 / fp32 when `dtype` or `torch_dtype` is float32 「来源」.
  - `_w2v_ops` reads `a.w_bits` / `a.w_fmt`. Other audio kinds unchanged.
- Activation format still follows the scenario / eval path (not forced to fp32).

### §282.2 Carbon-X
- `HuggingFaceBio/Carbon-X` — public `Qwen3ForCausalLM` + `dna_config` 6-mer tags, bf16 as released, 14.822B params, param_err 0 「来源」.
- Catalog id `carbon-x` (same SERIES prefix `carbon-` as carbon-8b / 3b / 500m). DNA tagging is metadata only; workload is stock Qwen3 causal LM.
- **0.445.** `Qwen3RMSNorm` is the same T5 recipe as Llama (`eps` inside `rsqrt`, not `1 + weight`). Block norms and `final_norm` are fp32 with one `rsqrt` per token row. `q_norm` / `k_norm` are that RMSNorm on `head_dim` after the projections (one `rsqrt` per head). SiLU stays `act`. The gate times up product is `mlp_mul`. Other Qwen3 catalog rows are not retagged, so the first-60 fingerprints stay put.

### §282.3 Speech pick not added
- **Qwen3-TTS** (`qwen3_tts`: talker + codebooks + tokenizer-12Hz) has newly public usable config but needs a new TTS builder — pending (same posture as OmniASR CTC).
- No other TTS / omni with newly public usable dims this pass.

### §282.4 Hotfix: portable UI-patch assert (0.336 suite)
- `test_core_336::test_docs_and_no_ui_patch` no longer requires `/workspace/ui-audit/ui.patch` (authoring-box only). It asserts the 0.336 CHANGELOG section contains **UI patch not applied**.

### §282.5 Freshness / pending
- OrcaCyber Nano: sha `1af1635` unchanged, gated auto, config 401 → still pending.
- Carried: OmniASR CTC; Qwen3-TTS builder; PixText routing mask; PCIe / CXL area; N7 / N28 controller; controller power; SPHBM4 PHY; NMC; HBM4 absolute pJ; LPDDR system idle/active.


## §283 Qwen3-TTS 12 Hz multi-codebook LM (0.338)

### §283.1 Architecture 「来源」
- HF `Qwen/Qwen3-TTS-12Hz-*` + github QwenLM/Qwen3-TTS `modeling_qwen3_tts.py`.
- **Talker** (`talker_config`, bf16): Qwen3 decoder (0.6B: h=1024 L=28; 1.7B: h=2048 L=28), `codec_head` vocab 3072. Text via `text_embedding` (text_h) → `text_projection` into talker h (prompt / trailing); per-frame input = sum of 16 codebook embeds.
- **Code predictor** (`code_predictor_config`, bf16): 5-layer Qwen3 (h=1024), AR `num_code_groups−1` steps per frame; embeds width = talker_h; optional `small_to_mtp_projection` when talker_h ≠ cp_h (1.7B).
- **Speech tokenizer** (`speech_tokenizer/*`, fp32): encode_downsample / decode_upsample 1920 @ 24 kHz → **12.5 Hz**. Decoder path costed; encoder used for Base voice-clone prompt only.
- **Speaker encoder** (Base only): CampPlus/Res2Net, once per clone request.

### §283.2 Catalog / roles
- Talker → LLM (`params_llm`); code_predictor → `depth`; tokenizer → `codec`; speaker_encoder → `spk`; text_projection → prompt-only (subtracted from `release_params`).

### §283.3 Adversarial (Carbon-X / OmniASR fp32)
- Carbon-X: public Qwen3ForCausalLM, param_err 0, eval prefill/decode OK; dna_config does not change GEMMs.
- OmniASR: `AudioSpec.w_bits`/`w_fmt` fp32 on bezzam ports held; whisper etc. still bf16 default.

### §283.4 Pending
- OrcaCyber Nano (sha 1af1635, gated). Tokenizer encode path detail beyond header GEMMs 「近似」. Carried: OmniASR CTC; PixText mask; PCIe/CXL; controllers; NMC; HBM4 absolute; LPDDR system idle/active.

## §284 Qwen3-TTS predictor prefill; speech freshness (0.339)

### §284.1 Qwen3-TTS adversarial fix 「来源」
- HF `code_predictor.generate(inputs_embeds=cat(past_hidden, last_id_hidden), max_new_tokens=num_code_groups−1)`: **one prefill** over the 2-row seed, then `cp_steps` decode steps with KV cache.
- 0.338 approximated 15× decode(1) with growing KV → undercounted the first step (2-row attention / MLP vs 1). Fixed in `code_predictor_ops` (seed emb + `cp.prefill.*` + `cp.1..cp_steps-1`).
- Talker / speech_tokenizer / as-released dtypes / eval / DSE / report / static paths otherwise held.

### §284.2 Freshness adds (existing builders)
- `qwen2.5-omni-3b` — same `qwen2_5_omni` Thinker+Talker+token2wav as 7B; Thinker text h=2048 L=36; Talker h=896 L=24; token2wav fp32 「来源」.
- `step-audio-2-mini-base` / `step-audio-2-mini-think` — same `step_audio_2` dims + token2wav sidecar as mini (`fetch_pt_release.py --token2wav`).
- `moshika-7b` — female-voice twin of moshiko; identical LM/depformer/Mimi dims (`fetch_pt_release.py --moshi`); SPEECH 26 → 30.

### §284.3 Kept pending
- OrcaCyber Nano 1.0: sha `1af1635`, gated auto, config 401 (drafter/mix GGUF only).
- New-arch speech: Qwen3-Omni, MOSS-TTS, Ming-3.0-tts-0.5B, pocket-tts.
- NPU: no public numbers this pass (PCIe/CXL controller area; HBM controller N7/N28; controller power; SPHBM4 PHY; NMC).
- UI patch not applied.

## §285 Adversarial 0.339 speech; Ming-3.0-tts-0.5B DiTAR (0.340)

### §285.1 Adversarial (Omni-3B / Step Base·Think / moshika / Qwen3-TTS prefill)
- param_err 0, domain omni/tts, speech KPIs, eval fits, SPS↔SPEECH parity held.
- Qwen3-TTS `cp.prefill.0.q` M=2; 15 codebook heads (prefill + 14 decode) 「来源」 HF generate.

### §285.2 Ming-3.0-tts-0.5B 「来源」
- HF `inclusionAI/Ming-3.0-tts-0.5B` + github `inclusionAI/Ming-3.0-tts` `modeling_bailingmm.py` / `fm/{flowloss,CFM,dit}.py`.
- Same TTS step as tiny: LLM token → FlowLoss (DiT depth 8, Euler 10, CFG 2) → Aggregator → streaming AudioVAE (44.1 kHz, 80 ms/step).
- Backbone: Qwen2.5-0.5B (`llm_config`); `ditar_config` maps to flow fields; `audio_tokenizer_config` = AudioVAE.
- Release roles: `flowloss`/`stop_head`/`spk_head` → `tts` (outside `params_llm`); audio + `linear_proj_audio` → vision.
- pocket-tts gated — not added.

### §285.3 Pending
- OrcaCyber Nano (sha 1af1635, gated). MOSS-TTS / Qwen3-Omni (new arch). NPU (no public numbers). UI patch not applied.

## §286 Adversarial Ming-0.5B DiTAR; Ming-omni-tts; Qwen3-Omni/MOSS pending (0.341)

### §286.1 Adversarial Ming-3.0-tts-0.5B
- `tts_spec` dense+DiTAR: flow_depth 8, Euler 10, CFG 2, rate 44100, s_per_step 80 ms; ops `flow0`…`flow9`; param_err 0; eval/speech/SPS OK.

### §286.2 Ming-omni-tts-0.5B 「来源」
- Same builder as §285.2; `ditar_config.patch_size=4`, `history_patch_size=32` → 320 ms audio/step. Release role `tts` for flowloss/stop/spk.

### §286.3 Why Qwen3-Omni / MOSS stay pending
- **Qwen3-Omni-30B-A3B** (`qwen3_omni_moe`): thinker MoE 128-exp + talker MoE + 16-codebook predictor + **code2wav** (not Qwen2.5-Omni token2wav DiT/BigVGAN). Needs a new omni mode, not a thin `omni_spec` fork.
- **MOSS-TTS-Nano / v1.5 / Local**: `moss_tts_nano` / `_delay` / `_local` — GPT2 global±local over N VQ + external audio tokenizer. New family vs Moshi/Qwen3-TTS.
- OrcaCyber / pocket-tts: still gated (config 401). UI patch not applied.

## §287 Adversarial ming-omni; energy/thermal/TCO consistency; NPU still pending (0.342)

### §287.1 Adversarial ming-omni-tts-0.5b
- Same DiTAR as §286.1 / §285.2: flow_depth 8, Euler 10, CFG 2, AudioVAE 44.1 kHz; `patch=4` / `history=32` → `s_per_step` 320 ms (4× Ming-3.0-tts-0.5B's 80 ms); ops `flow0`…`flow9`; param_err 0; eval/speech/SPS OK.

### §287.2 NPU design — no public numbers this pass
- Searched again: PCIe/CXL controller area (vendor IP pages, no mm²); HBM4 absolute pJ (Samsung shipping note has BW only; blog ranges not used); JEDEC SPHBM4 PHY (JESD330-4 interface only; Eliyan NuLink/UMI marketing ≠ SPHBM4); HBM NMC TOPS/W (LPDDR5X-PIM / CXL MX1 are other products). Carried pending unchanged.

### §287.3 Consistency audit (energy / thermal / TCO, 0.333–0.341) 「推导」
- Presets: HBM3E 5.80 / HBM4 4.14 (=÷1.4) / LPDDR 3.23; `EnergyTable` ↔ `ENERGY_PRESETS`; `MEM_PROXY` DSE labels track 4.14 / 5.80 / 2.67.
- `dram_pJ_field`: HBM3E→`pJ_bit_dram_hbm`, HBM4→`pJ_bit_dram_hbm4`, LPDDR→`pJ_bit_dram_lpddr`; `energy_report` J_dram ∝ pJ (synthetic ratio 1.400).
- `package_power`: standby adds to `static_W` / `pkg_W`; `static_floor_W` unchanged (0.333 invariant).
- LPDDR accepted by `hbm_standby_W` (0.335). No stale 3.44 / 2.46 live assignments in code. No fix required.

### §287.4 Freshness
- OrcaCyber Nano sha `1af1635`, gated auto, config 401. pocket-tts sha `db04f0e`, gated auto, config 401.

## §288 Functional UI bugfixes from ui-audit; visual restyle still pending (0.343)

### §288.1 Ported (functional only) 「来源」 ui-audit/ui.patch on f53ac7a
- **`.row` flex:** `.row` had no rule; 设计洞察 labels/inputs/checkboxes stacked. Share `.toolbar` bottom-aligned flex wrap (concrete px, no new type/color tokens).
- **`#wl-kind` clear:** `paintDomain` else-branch clears the caption when `m.workload` is absent (LLM / speech).
- **Empty `.scroll` tables:** `:has(> table:empty)` drops `padding-bottom` so unrun sections do not stack scrollbar gutters.

### §288.2 Not applied — waiting on user
- The rest of ui-audit visual harmony (type scale `--fs-*`, control height, segment restyle, KPI fonts, tag tones, chart max-width, etc.). AGENTS §293: do not apply until the user asks.

### §288.3 Freshness / NPU
- OrcaCyber Nano sha `1af1635` / pocket-tts sha `db04f0e`: still gated (config 401).
- NPU: no new public numbers; skipped.

## §289 Adversarial 0.343 UI; Qwen3-Omni gate; NPU skip (0.344)

### §289.1 Adversarial 0.343 functional UI 「推导」
- Headless + Playwright smoke: insight `.row` → `display:flex` / `flex-wrap:wrap`; empty `#ins-size-tbl` scroll `padding-bottom: 0`; gen (`wan2.1-1.3b`) fills `#wl-kind`, LLM (`qwen3-8b`) clears it. Unit: CSS / `paintDomain` else still present; no visual tokens (`--fs-kpi`, `--ctl-h`).

### §289.2 Qwen3-Omni — still new arch 「来源」 HF config 2026-10-23
- `Qwen/Qwen3-Omni-30B-A3B-Instruct` `model_type=qwen3_omni_moe`: thinker MoE (128 exp, top-8, h=2048, L48); talker MoE (128 exp, top-6, h=1024, L20, vocab 3072); nested `code_predictor` (5L, 16 code groups); **code2wav** ConvNet (8L, h=1024, upsample [8,5,4,3], sliding_window 72) ≠ Qwen2.5-Omni `token2wav` DiT+BigVGAN.
- Gate: `omni_spec` / `qwen3_tts_spec` return None on a minimal public-dim stub; no catalog id. AGENTS §291 / §294: do not thin-wrap.

### §289.3 NPU / freshness
- NPU: no new absolute public numbers this pass (SPHBM4 JEDEC interface only; SemiAnalysis ~16% Rubin estimate and HBM-side Samsung PHY maps already covered by existing HBM4 `mm2_per_dq_hbm`).
- OrcaCyber Nano sha `1af1635` / pocket-tts sha `db04f0e`: gated auto, config 401.

## §290 Qwen3-Omni dual-MoE + code2wav (0.345)

### §290.1 Architecture 「来源」
HF `Qwen/Qwen3-Omni-30B-A3B-Instruct` config + safetensors headers; transformers `modeling_qwen3_omni_moe.py` / `configuration_qwen3_omni_moe.py`; vllm-omni code2wav.

- **Thinker** (`qwen3_omni_moe_text`): MoE h=2048, L=48, 128 experts top-8, moe_inter 768, shared 0; AuT audio + ViT (storage only).
- **Talker** MoE: h=1024, L=20, 128 experts top-6, moe_inter 384, shared 768; codec vocab 3072; bf16.
- **Code predictor**: 5L h=1024, 15 AR steps (16 groups − 1); bf16 — same shape as Qwen3-TTS.
- **Code2wav** (not token2wav DiT): embed+mean RVQ → 8L sliding-window pre_transformer (window 72) → upsampling_ratios [2,2] → decoder upsample_rates [8,5,4,3] + SnakeBeta → 24 kHz. `total_upsample = 1920` → 12.5 Hz. **bf16 as released** (headers; not fp32).

### §290.2 Catalog / roles
`qwen3-omni-30b-a3b` → `core/qwen3_omni.py`. Release roles: thinker → params_llm; `talker.*` → params_talker; `talker.code_predictor` → params_depth; `code2wav.*` → params_code2wav. Gated on `model_type == qwen3_omni_moe` only — `omni_spec` / `qwen3_tts_spec` stay None.

### §290.3 NPU / UI / freshness
NPU dig: no new absolute adopted. No visual UI restyle. Freshness light.

### §290.4 Unmodeled
Image/video ViT forward (storage only); Thinker→Talker projection traffic detail; code2wav streaming window schedule (costed per-frame); SPHBM4 / PCIe5/6×16+CXL N5 area / HBM4 all-in pJ still pending public numbers.

## §291 Adversarial Qwen3-Omni; MOSS-TTS Delay (0.346)

### §291.1 Qwen3-Omni adversarial 「来源」 HF modeling + config 2026-10-23
- code2wav: streaming one new frame with `past_key_values` attends up to `sliding_window` (72), not `min(1, window)`.
- Talker: `text_projection` ResizeMLP once at setup (thinker_hidden → intermediate → talker_hidden); counted in Talker prefill, not per frame.
- Talker MoE top-6 + shared 768 via `build_rank_ops` — unchanged. As-released bf16.
- Joint Thinker∥Talker: no new mechanism beyond existing first+max(audio·rtf, text·tpot).

### §291.2 MOSS-TTS Delay 「来源」 OpenMOSS-Team/MOSS-TTS
- `moss_tts_delay`: Qwen3 dense (h=4096, L=36) + n_vq=32 delay channels; audio_vocab=1024; 1s≈12.5 tokens; sampling_rate 24000; bf16.
- Roles: `language_model.*` → catalog; `emb_ext` → lookup; `lm_heads.0` text; `lm_heads.1..32` → params_depth / audio heads on same card.
- External tokenizer/vocoder absent from checkpoint — waveform unmodeled (coverage partial).
- Nano (GPT2+local float32) / Local (gated) deferred.

### §291.3 Pending
- NPU absolute still none public this pass.
- MOSS-TTS-Nano / Local; MOSS waveform tokenizer if released into same repo.
- Qwen3-Omni Thinker/Talker true pipelined overlap (needs a sourced schedule).

## §292 Adversarial MOSS-TTS; sidecar_params for same-card speech fits (0.347)

### §292.1 Adversarial 「来源」
- MOSS Delay: every generate step runs text + n_vq audio lm_heads; delay fill still pays full frame. first_audio = ttft + delay_frames·frame + frame.
- Qwen3-Omni 0.346 fixes held: code2wav keys=`sliding_window` when frames=1; Talker `text_projection` once at setup.

### §292.2 `sidecar_params` 「推导」
- Problem: params_depth (MOSS audio heads / Moshi depformer) lived on the replica card but outside `params_llm`, so `evaluate.fits` ignored them (speech KPI only reported GB).
- Mechanism: `sidecar_params` storage-only (not in `Model.params()`); memplan head-stage `store_extra` like `tts.stored_params`. bf16 via `sidecar_bits=16`.
- Not used for speech-card stacks (Qwen3-Omni Talker/code2wav) — those keep `stored_bytes ≤ dram` on the extra card.

### §292.3 Deferred
- MOSS-TTS-Nano (GPT2+local, fp32, pickle bin); MOSS-Local (gated); Mimi fp32 into sidecar; NPU absolute; visual UI restyle.

## §293 Adversarial sidecar; `sidecar_bytes` folds Mimi (0.348)

### §293.1 Adversarial 「来源」/「推导」
- `params()` / param_err exclude sidecar — no double-count with memplan `store_extra`.
- PP: sidecar only on `has_head` stage (same as `tts.stored_params`). TP: full sidecar per rank (speech TP usually 1).
- MOSS: `sidecar_bytes = audio_head_params × 2` (bf16) 「来源」 release headers.
- Moshi depth: bf16 via tensors.json; single `sidecar_params*bits` could not mix Mimi fp32.

### §293.2 `sidecar_bytes` 「推导」
- Prefer exact resident bytes when dtypes mix. Moshi: `sidecar_bytes = MoshiSpec.stored_bytes` = depth×2 + mimi×4 「来源」 `speech_tensors` roles.
- memplan: if `sidecar_bytes` set, use it; else fall back to `sidecar_params * (sidecar_bits/8)`.
- Speech-card stacks (Qwen3-Omni Talker/code2wav) stay off sidecar.

### §293.3 Deferred
- MOSS-TTS-Nano / Local; Thinker∥Talker sourced pipeline overlap; NPU absolute; visual UI restyle (§293 AGENTS wait).

## §294 Adversarial sidecar_bytes; package-power margin in design report (0.349)

### §294.1 Adversarial 「来源」/「推导」
- Moshi `sidecar_bytes = stored_bytes` (depth bf16 + Mimi fp32); MOSS heads × 2. memplan / evaluate.fits / PP-head / TP-full per rank match; `params()` excludes sidecar.
- Speech-card stacks (Qwen3-Omni etc.) stay off sidecar. DSE filters on `r.fits` — sidecar already in the gate.

### §294.2 NPU absolute (empty this pass)
- HBM4: Samsung / SK hynix still ≈40% vs HBM3E only; no public absolute pJ/bit. HBM4E “<1.95 pJ/bit” second-hand / other product — not adopted.
- SPHBM4: JESD330-4 interface (512 DQ, 4:1); Eliyan UMI/NuLink marketing mm² ≠ JEDEC SPHBM4 host PHY.
- PCIe/CXL N5 controller area; HBM controller power: vendor IP pages, no public mm² / pJ.

### §294.3 Report mechanism 「推导」 from sourced COOLING
- Parallel to flux (0.331): `pkg_margin_W`, `pkg_headroom`, `thermal_binding` (tighter of flux vs package).
- Front points gain `pkg_*` fields; `front_pkg` near-tie when package margin ≤ DSE tie. Limits never retuned.

## §295 Adversarial thermal margins; card-axis consistency (0.350)

### §295.1 Adversarial 「推导」
- `COOLING` pkg_W / flux_W_mm2 / die_flux_W_mm2 unchanged (air 1000 W / 0.86; liquid 1400 / 0.96).
- Package margin / front_pkg warn sets match row `thermal.pkg_W`. No-cooling → no limit items / no front_thermal.

### §295.2 Feasibility vs design margin axes
- `thermal.flux_ratio` = min(card_ratio, die_ratio): flux fails only if **both** exceed (MODEL §259 / thermal docstring).
- 0.349 used that min for `flux_headroom`, `thermal_binding`, front flux ties, and margins「热流」, while `flux_margin_W_mm2` stayed card-level → liquid headroom +140% vs ~1.5% card left.

### §295.3 Fix 「推导」
- Design margins / headroom / binding / front flux ties use **card** `flux_W_mm2 ÷ COOLING.flux_W_mm2`.
- Report exposes `card_flux_ratio`; when different, `die_flux_ratio` and `feas_flux_ratio`. Binding = max(card, package).
- Visual UI restyle: no written user approval → not applied. Thinker∥Talker: no sourced schedule → not invented.

## §296 Adversarial card-axis across DSE/UI; persist ratios (0.351)

### §296.1 Adversarial 「推导」
- Liquid card headroom ≈1.5%; feas-min headroom would be ~140%. COOLING air 1000/0.86/0.55, liquid 1400/0.96/1.70 unchanged.
- Report/design path fixed in 0.350; eval/DSE/static still surfaced feas `flux_ratio` as the only 「× 上限」.

### §296.2 Mechanism
- `thermal_report` top-level `card_flux_ratio` / `die_flux_ratio`; DSE rows persist them; report `flux_ratio` label = feasibility min; UI chip + DSE tooltip show 卡级.
- DIRECTION: `pkg_margin_W`/`pkg_headroom` (+1), `card_flux_ratio` (−1).

### §296.3 Deferred
- NPU: no new absolute public mm² / pJ → skip. No speech catalog churn. Visual UI restyle: still no written user yes.

## §297 Adversarial ratio persist; power-cap three-way binding (0.352)

### §297.1 Adversarial 「推导」
- Card/die ratios persist on thermal_report and DSE rows; report/UI show 卡级 vs 可行性 min. Liquid design headroom ≈1.5%.
- Static: `build_static` copies `accel_dse/web/app.js` verbatim. Pages (0.259) lags until rebuilt from main.

### §297.2 Power-budget binding 「推导」
- DVFS compares `power_cap_W` to card W (`power_scope=card`) or package W (`package`). Report now exposes that axis as margin/headroom/ratio.
- `thermal_binding` = argmax of card flux/limit, cool pkg/limit, cap occupancy. liquid+package → 功耗预算 (was 热流).

### §297.3 Deferred
- NPU absolutes still absent. Visual UI restyle: no written user yes. Pages rebuild: separate track.

## §298 Front power-budget near-ties (0.353)

设计结论报告的前沿热敏感性（`front_thermal`）在 0.349/0.350 已覆盖卡级热流与冷却封装功耗持平点。0.352 把设计点的 `thermal_binding` 扩成三方（含功耗预算），但前沿点仍不报告预算持平——液冷 + `power_scope=package` 时前沿可坐在 DVFS 预算刀口上却只有设计级 `margin` 警告。

0.353：前沿点按与 DVFS 相同的 `power_scope` 轴计算 `cap_ratio` / 余量；余量 ≤ 持平阈时发 `front_cap` 警告；markdown 表增加「预算占用」。冷却上限与用户预算均不调，只报告。

## §361 RWKV-8 Heron pile（0.361）

`BlinkDL/rwkv-8-pile` 有两个 Apache-2.0 检查点，没有 `config.json`。维度来自 `.pth` 头，计算来自 `RWKV-LM` 的 `rwkv_v8_rc00_demo.py` 与 `rwkv_v8_rc00_hybrid_demo.py`。

时间混合就是 RWKV-7（`lin=rwkv7`，fp32 状态 H×64×64，prefill 用已有的 chunk 64）。通道混合在 FFN 之后按当前 token 乘 `ffn.enn` 的一行（DeepEmbed，查表，不是 KV）。hybrid 在时间混合旁边加 DeepEmbedAttention：每层缓存 fp16 的 32 维 kk 和 32 维 vv，整栈另存一份 int32 token id 并每步读一次，再用 `kkup` / `vvup` 重建整段 K/V，乘 `ennk` / `emm`，soft-cap 后做单头因果注意力。纯模型没有这条缓存，解码代价不随上下文变。参考代码是单卡，DEA 投影按复制计。

## §362 DEA 重建 K/V 的重读（0.362）

0.361 把低秩缓存记在 `kv_read`，重建出的 K（256）和 V（768）是激活。放得进 SRAM 激活半区时不出 DRAM。放不下时 GEMM 写出，`dea_qk` 按真实宽度读回（解码 q=1 读一遍；prefill 按 flash 的 Q 分块重读）。等宽公式会把 V 当成另一个 K，长上下文解码因此少算约一半激活流量。`dea_qk` / `dea_pv` 与投影一样按 TP 复制，请求 FLOPs 只计一次。

ROSA 不在这两个检查点里。TypeSafe Jev、Microsoft-Decision-1 仍没有公开权重；StartLux 的 bf16 家族已经在目录中。


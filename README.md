# accel-dse

> 推理加速器设计空间探索（LLM · 视频生成 DiT · 蛋白质）· Analytical design-space exploration for inference accelerators

[English](README.en.md) · [建模说明](docs/MODEL.md) · [更新日志](CHANGELOG.md) · [MIT](LICENSE)

**accel-dse** 用可核对的解析模型回答流片前的问题：给定一个模型（LLM / VLM，或视频生成 DiT、蛋白质语言模型；按官方发布的权重与 dtype）、一种数据通路映射、片上 SRAM、存储器（HBM / LPDDR）和卡数，最好的并行布局与 batch 是什么，瓶颈在哪里，结论对假设有多敏感。零第三方依赖，Python ≥ 3.10。

## 它做什么

- **模型按发布建模**：从 HF `config.json` 与 safetensors 头（不下载权重）得到逐角色的参数量与存储 dtype（bf16 / fp8 block / MXFP4 / int4 AWQ 等），与发布总量偏差 ≤0.5%；官方量化版是独立条目。每个模型带三轴标签：来源（官方 / 镜像）× 覆盖（完整 / 部分 / 架构代理，附逐项的近似之处）× dtype；目录按厂商 → 系列排列（同一厂商的 LLM、VLM、视频与蛋白质模型在一起）。
- **视频生成与蛋白质（0.41 – 0.43）**：Wan2.1-T2V（1.3B / 14B）、CogVideoX（2b / 5b）的 DiT 去噪主干（全 3D 注意力，每步按 CFG 前向 1–2 次）与 ESM-2（650M / 3B）编码器可评估：单段延迟、每帧延迟、帧/s/卡，或批延迟、序列/s/卡、残基/s/卡；Ulysses 序列并行（SP）与 CFG 并行（DP）是布局维度；文本编码器与 VAE 在 0.44 前未建模。0.42 增加 Wan2.2-T2V-A14B（双专家，每步激活一个）、HunyuanVideo（双流 / 单流，全 3D 注意力）、LTX-Video 2B、Mochi 1、Open-Sora STDiT3（发布即为分解时空注意力）与 MiniMax-H3（视频 + 音频 + 文本联合序列）。0.43 接入蛋白质结构预测：ESMFold、AlphaFold 2（官方 JAX 参数 model_1_ptm）、OpenFold、Boltz-1、Protenix——pair 表示、三角乘法 / 三角注意力、MSA 行 / 列注意力、外积均值、IPA、扩散模块按检查点张量形状建模，recycle / 扩散步数 / 样本数可调（MSA / 模板检索未建模，覆盖「部分」；ESMFold「完整」）。AlphaFold 3 权重需申请，仍标「暂未接入 v2」。
- **视频整条 pipeline（0.44）**：文本编码器（umT5 / T5-XXL / LLaVA-Llama-3 + CLIP / Qwen3-VL 文本塔）与 VAE 解码器（含 HunyuanVideo tiling、Open-Sora 逐帧 SD-VAE、H3 ViT 解码器与音频 VAE）按发布检查点的张量头建成算子图，时间与存储默认计入（与去噪串行「假设」，见 docs/MODEL.md §11.4），视频覆盖全部为「完整」；`workload.pipeline = false` / CLI `--dit-only` 只看 DiT。另增激活 dtype what-if（Web「激活 dtype」/ CLI `--act fp32`），评估结构模型参考实现 fp32 激活的流量代价。
- **多卡结构模型与组件放置（0.45）**：结构模型支持 DAP（FastFold 动态轴并行，占用布局的 SP 维）：pair / MSA / 模板网格按残基轴切分，三角乘法 / 偏置 / 外积均值的 all-gather 与行 ↔ 列 all-to-all 逐核计入（AF2 8 卡 7.8×，扩散为主的 Protenix 1.8×）。视频组件放置 `workload.placement`：常驻 / 文本编码器 FSDP 分片（Wan `--t5_fsdp`）/ 顺序卸载（diffusers `enable_model_cpu_offload`、Wan `--offload_model`），默认 auto 取第一个放得下的——64 GiB LPDDR 上 Wan2.1-14B、Wan2.2-A14B、MiniMax-H3 可放下（每请求 1–2.5 s 主机重载）。可选 VAE 分块解码（`--vae-tiling`，CogVideoX / Mochi 按 diffusers 默认 tile）。请求 TFLOP 改为整个副本的有用 FLOPs（此前多卡布局按单 rank 计）。
- **0.46**：DAP 时扩散样本分到各卡（Protenix 5 样本，8 卡 1.8× → 6.0×；精确、无额外通信）；DiT 权重 FSDP（Wan `--dit_fsdp`，`workload.dit_fsdp` / CLI `--dit-fsdp`）；文本编码器放主机 CPU（Wan `--t5_cpu`，`--te-cpu --host-TFLOPS`，主机算力为「假设」）；Wan VAE 可选分块（diffusers 参数）。
- **0.47**：VAE 多卡分块并行解码（按 MiniMax-H3 发布的 `parallel_tiling`，`--vae-parallel`；H3 SP8 解码 24.4 → 3.5 s）；LTX VAE 可选分块；MiniMax-H3 视频解码按发布更正为总是分块（256 px tile）；跨请求重叠（主机编码器 ∥ 去噪，`--overlap`）。Open-Sora 分块与结构模型 pair TP 经核实无参考实现，不做。
- **0.47.1**：能耗动作计数（MAC / 向量 / SRAM / DRAM / 链路 / 卡·秒，每 token / 帧 / 序列）× 用户提供的每动作能耗（`--pJ-mac … --idle-W`，API `energy`）；工具不内置任何能耗数值。
- **0.48**：系统级缓存 SLC（`--slc-mib`，pin / lru 两种策略，默认关）与两级互连（封装内 D2D `--package-cards --d2d-GBps` 与跨封装网络层，集合通信按组跨越的层分级计）；能耗新增 SLC / D2D 两项。默认结果与 0.47.1 一致。
- **0.49**：MoE 专家负载倾斜（`--moe-skew` 或实测每专家分布 `--moe-expert-load`，最忙 EP rank 决定时间，能耗计数守恒）；资源 / 面积预算（SRAM / MAC / 卡数 / 功耗 / 面积代理限额，密度由用户填写，报告余量并标出超预算布局）。默认结果不变。
- **0.50**：prefill / decode 分离（`--pd`，prefill 池与 decode 池可用不同卡数与布局，KV 交接按互连层计时，给出 TTFT / TPOT / goodput 并与同卡数合并服务对照；稳态流体模型，第一版）；三层互连——可选的封装内 D2D（默认关 = 单片大 die；打开默认 UCIe-A 48 GT/s，可选 UCIe-A 64G / UCIe-S 32G / BoW / NVLink-C2C，`--d2d --d2d-std`）、节点内 scale-up、跨节点网络（`--node-cards --net-GBps`）。默认结果不变。
- **0.51**：PD 排队与尾延迟（解析近似「假设」）：同一泊松到达率下比较 PD 分离、合并 prefill 优先、合并分块 prefill 的 TTFT p50 / p90 / p99、TPOT 分位、最长 token 间隔与 SLO goodput；decode 连续批处理按 Little 不动点、KV 与池内集合通信争用、PD 能耗（`--pd-load --pd-rate --pd-chunk`）。默认结果不变。
- **0.52**：PD 请求长度分布（`--pd-prompt-cv / --pd-out-cv` 或离散 `--pd-mix`，M/G/1 排队、长度偏置的 decode 上下文、按分布的容量）、前缀缓存命中率（`--pd-prefix-hit`，只 prefill 未缓存部分、KV 交接相应缩小）、池布局搜索（`--pd-search-layouts`）；修正合并分块 prefill 的平均迭代时间。全部「假设」，默认结果不变。
- **0.53**：PD 前缀缓存容量 + LRU 淘汰（`--pd-prefix-len / --pd-prefix-count / --pd-prefix-zipf`：Zipf 工作集、Che 近似，命中率由剩余 DRAM 或 `--pd-prefix-cache-GB` 推出，`--pd-prefix-affinity` 前缀感知路由；显式 `--pd-prefix-hit` 仍可覆盖）、异构池（`--pd-prefill-chip / --pd-prefill-mem`，prefill 池可用不同芯片 / 存储器）、布局搜索里的 decode batch（`--pd-search-decode-batch`）。全部「假设」、默认关，默认结果不变。
- **0.54**：验证版本。新增请求级离散事件仿真（DES），用同一套逐步代价对照 PD 闭式排队模型（V4 服务验证：54 点网格，`accel-dse validate` / `scripts/v4_serving.py`；SLO goodput 误差 −8 % … 0 %，合并模式 TPOT 尾偏乐观，见 MODEL.md §18.4）。修正 decode 连续批处理（birth–death 替代 Little 不动点）、prefill 优先的 TPOT 分位数与最长停顿、混合服务时间的 TTFT 分位数（卷积），PD 报告数值随之变化。可选 `--pd-sim` 附上模拟尾部。默认结果不变。
- **0.55**：合并模式尾部与覆盖面。V4 改为多 seed（30 点 × 3 seed，含 MoE Qwen3-30B-A3B 与 TP4 Qwen3-32B 两个新场景族）；prefill 停顿造成的 decode 成批到达（批到达链 + 配对系数）、TPOT 窗口因子按相关时间、分块 TTFT 尾（准静态混合 + 休假）、最长间隔按一生最大 batch，合并模式 TPOT p90 中位误差由 −14 … −18 % 收敛到 −4 % 左右（MODEL.md §18.5）。可选 decode KV 容量策略 `--pd-kv-policy wait|recompute`（默认关）。默认结果不变。
- **0.67.0**：0.66.0 + 目录覆盖收口——全部 LLM 覆盖「完整」（建模说明 §2.1）；已完整模型数值不变。
- **0.66.0**：外部校核——参考硬件（H100 / H200 / A100，数据手册峰值）对照 100 条公开实测（TensorRT-LLM、Databricks），留出法校准研究；可选 `exec_overlap` 级内重叠模式（默认不变）。结论：级内全重叠是主要系统误差，不引入拟合效率系数（建模说明 §25）。
- **0.65.1**：V4 网格按 0.65 闭式与 DES 稳定判据重跑（可续跑 `--partial`），修正批服务律原子截断（CV 1 时上限 2 的 TTFT 尾部乐观 −19 %）（建模说明 §24.8）。
- **0.65.0**：prefill 批服务器改为精确的贪心批服务排队（PD 闭式 SLO 容量与 DES 相差 ≤ 3 %，prefill 优先 ≤ 10 %，原上限 2 乐观 +72 %）、DES 稳定性按漂移 / 利用率判定、EP 空闲 rank 的 SRAM / 链路 / 反量化计数、输出驻留分块搜所有整数块（建模说明 §24）。
- **0.64.0**：SLO 感知的 prefill batch 上限（PD / prefill 优先，含稳定上限回退）、DES SLO 搜索改扫描 + 细化、二维 GEMM 分块（激活溢出）、闭式运行 batch 两点混合、不整除 TP / EP 的反量化与 EP 空闲 rank 的 DRAM 字节（建模说明 §23）。
- **0.63.0**：外部评审的 9 条修正：MLA decode 逐头吸收、视频 CFG 依赖、SLO 搜索不再假设单调、radix 容量依赖、不整除 TP 不丢头、PP 逐对路由、有效工作守恒（能耗 / TFLOP）、DES 投机抽样、线程安全；新增 LLM prefill 激活流式 / 溢出（建模说明 §22）。
- **0.62.1**：第六轮收敛审计，代码与结果不变：组合模糊测试（2400 个场景）与 PD 恒等测试无新缺陷；更正建模说明里过时的 fabric 例（DeepSeek-V3 / Kimi-K2，0.61.1 起 combine 按 bf16）、发布数（72）与 ESMFold fp32 例。
- **0.62**：VLM 视觉编码器按发布建模（ViT + 合并 / 投影，按各自图像预处理器得到 patch 网格与图像 token；FLOPs 与 transformers / 发布 remote code 的参考计数逐一精确一致）：`serving.images`（默认 0 = 只算文本）、`image_w / image_h`（默认 1024×1024「假设」），编码器在 prefill 时运行，图像 token 进入 prompt / KV；CLI `--images --image-size`。PP 流水级默认按代价平衡切分（`pp_split = "cost"`，含输出头 / 嵌入 / MTP 与异构层栈，由完整模型在候选间裁决，节拍不劣于按层数均分；`layers` 保留旧行为）：PP > 1 的节拍下降 0.03 % ~ 50 %（结构模型最大）。另修 PD 带前缀时的 KV 交接字节（滑窗 / 递归状态按层类型计）、DeepSeek-V4.1 对齐器归入视觉参数、视频卸载重载的主机链路能耗。见 MODEL.md §19.14、§20、§21。
- **0.61.4**：第四轮审计修正。DP 空闲 rank 不再计能耗 / TFLOP（例 ESMFold batch 1 · DP2 每单位 MAC 194.65 → 97.32 TFLOP）；PD prefill 布局不整除 decode 池时不再报错，预算核对 prefill 池；新增 `energy.pJ_bit_host`（swap 主机链路，无默认值）；混合 fp8 × int8 升 bf16；Pareto 扩到 batch 512 以上；上下文超长与 PP 不平衡警告。见 MODEL.md §19.13。
- **0.61.3**：第三轮审计修正。结构模型对发布实现的独立 FLOP 计数：AF2 / OpenFold 模板行并入 MSA（+0.4 %）；Boltz-1 推理缓存与别名去重（−11 % FLOPs，批延迟 13.05 → 7.94 s，参数 592.01 M）；Protenix pair 偏置按样本计（+3.6 % FLOPs，批延迟 19.33 → 27.86 s）；Open-Sora 时间 VAE −1.6 %。embedding 表等冷存储不再占 SRAM / SLC；PD 能耗计入抢占恢复；预算范围说明、最佳 batch 无解警告。见 MODEL.md §19.12。
- **0.61.2**：第二轮审计修正。LTX-Video VAE 解码原来少 2.5–3.3×（上采样块的 resnet 按输出分辨率计，与 diffusers 计数一致）；ESMFold 主干按参考实现跑 4 遍（原 5 遍，FLOPs −20 %）；大占用时 TPOT 分位数的相关时间不再数值溢出；radix 前缀缓存中放不下的节点不占容量；面积预算的下界判定；界面中过期的搜索 / 扫描结果会标出。见 MODEL.md §19.11。
- **0.61.1**：第一轮审计修正。DeepSeek-V4 压缩层 decode 的 KV 读原来少了压缩比倍；滑窗 / top-k / 压缩注意力的 prefill 改按实际平均 key 数计（DSA prefill −14 … −20 %），索引器改为因果；纯滑窗 latent 层只存窗口内的 KV；线性注意力计入 conv 状态读写；fp8 激活的发布，all-reduce / combine 按 bf16 计「假设」；serving 尺寸设上限。见 MODEL.md §19.10。
- **0.61**：性能与网络细节（默认结果不变）。大规模 PD 排队报告 20–50 s → 3–4 s，布局对比 / 稳定性在服务中走进程池（1024 卡约 7 s）；可选跨 leaf / pod 的每步附加时延、NVLS 类 allgather / reduce-scatter、NCCL 第二棵树的切边、LL / LL128 / Simple 协议效率（「假设」）。见 MODEL.md §19.6–19.9。
- **0.60**：大规模卡数与网络细化（「假设」，默认结果不变）。
  - 每副本卡数上限 64 → 8192（API / UI / CLI），可评估数百到数千卡、多节点的 1P–10P 多芯片 SKU；batch 上限随卡数放大，PD 搜索 / 排队模型做了剪枝与插值，1024 卡布局排名约 8 s。
  - 三层 fat-tree（leaf / spine / core，逐层收敛比，每 pod 节点数，流量离开 leaf / pod 的份额）。
  - 不同层的物理端口并发（`overlap = ports`），以及按暴露时间（计入与计算重叠）选算法的 `auto_overlap`。
  - PP 交接在 ring / torus / full mesh 上计多跳；PD 的 KV 争用可回灌 decode TPOT。
  - 见 MODEL.md §19.1–19.5。
- **0.59**：硬件侧的跨节点网络拓扑与集合通信算法（`fabric`，「假设」，默认关）。跨节点支持 fat-tree / leaf-spine（上行收敛比）与 rail-optimized（PXN），节点内 scale-up 支持 switch / full mesh / ring / 2D torus。每次集合通信按 α-β + 每步时延在 ring / tree（NCCL 双二叉树）/ 分层中取最快，可选网内归约（SHARP / NVLS 类厂商选项）；收敛网络上的 EP all-to-all、跨 leaf 的 PP 交接、PD KV 传输与集合通信争用都按份额计。用 NCCL ring / tree 公式手算对照（MODEL.md §19）。默认结果不变。
- **0.58**：修正 0.57 合并模式 KV 准入在 CV1 下过保守：占槽时间只计 prefill 服务，合并模式条件等待取 Gamma（c² 随服务 CV），分块模式抢占乘 (1 − ρ_pre)。合并 CV1 TTFT p90 +23 / +40 % → +14 / +21 %，分块抢占次数 +30 … +60 % → −10 … +4 %。新增 radix / 部分前缀匹配（`pd.prefix_tree`，「假设」，默认关）：多层共享前缀树，token 容量的 LRU，按节点做 Che 近似，部分命中省掉已匹配的 token；DES 用 RadixLRU 对拍，token 命中率差 ≤ 0.004（MODEL.md §18.8）。默认结果不变。
- **0.57**：KV 容量策略的模型修正。准入等待按 M/G/c 槽位队列（Cosmetatos M/D/c 混合；合并模式 prefill 期间也占槽）；饱和抢占加「回填偏移」，恢复占用在减速后的槽位链上迭代不动点。修正 DES 中恢复期间不记 KV 占用的错误：0.56 记录的 recompute「连锁抢占」主要来自这个错误。抢占次数误差 +39 … +190 % → −23 … +60 %，PD recompute CV1 TTFT p90 −40 % → −21 %（MODEL.md §18.7）。V4 网格高负载点改用 20000 请求的 DES（`--n-high`）。默认结果不变。
- **0.56**：服务模型收尾。KV 容量按 vLLM 的顺序先准入、后 prefill，准入等待计入 TTFT 与 SLO goodput（`--pd-kv-admit`）；抢占进入 TPOT / 最长间隔尾部；新增 swap 抢占（`--pd-kv-policy swap --pd-swap-GBps`，主机链路带宽默认 50 GB/s「假设」）。长度档 8 → 15，M/G/1 等待改用精确 Pollaczek–Khinchine 律，CV 下 TTFT p50 最差误差 +47 % → ±18 %（MODEL.md §18.6）。API 输入严格校验（400 而不是 500）。PD 能耗可按池给静态功耗（`--idle-W-prefill`）。默认结果不变。
- **映射是设计变量**：输出驻留（OS）、权重驻留（边缘加载 / 宽面广播）、OS + GEMV 单元、可重构，逐算子计算 MAC 界与 SRAM 供数界；芯片不原生支持的格式计入反量化开销。
- **逐 rank 算子图**：TP / PP / 注意力 DP / EP / ETP，单卡就是全 1 布局，没有第二条路径。
- **存储规划与调度**：权重 / KV 的 SRAM 驻留、staging、逐 stage 容量；每级 `max(MAC/FEED, VECTOR, DRAM, LINK) + SYNC`，绑定瓶颈与有效 MAC 比例直接给出。
- **精确搜索**：每个布局在 TPOT SLO 下的最大 batch（分支定界，已用暴力枚举核对），decode 或含 prefill 的 goodput 目标，DP prefill 的 TTFT 标记，排名在「假设」扰动下是否稳定。
- **校验**：参数对照发布、H100 类配置的趋势区间、与 GenZ 的对照，见 [建模说明](docs/MODEL.md)。

## 快速开始

```bash
git clone https://github.com/kalcohol/accel-dse.git
cd accel-dse
python3 tests/run_tests.py          # 自检（零依赖；也可 python3 -m pytest tests/ -q）
python3 -m accel_dse serve          # Web 工作台 → http://127.0.0.1:8765
```

命令行：

```bash
python3 -m accel_dse models                                   # 模型目录与三轴标签
python3 -m accel_dse eval --model qwen3-8b --best-batch       # 单点评估（默认 100T 芯片 + LPDDR5X）
python3 -m accel_dse compare --model deepseek-v3 --cards 8 \
    --mem hbm3e_8s_12h24g_9200 --ctx 4096                     # 每种映射的最佳布局
python3 -m accel_dse search --model qwen3-32b --cards 8 --objective goodput --mem hbm3e_8s_12h24g_9200
python3 -m accel_dse stability --model qwen3-32b --cards 8 --mem hbm3e_8s_12h24g_9200
python3 -m accel_dse eval --model wan2.1-1.3b --steps 20 --sp 2   # 视频：单段 / 每帧延迟（480P、81 帧）
python3 -m accel_dse eval --model qwen3.8-27b --phase prefill --prompt 1024 \
    --images 2 --image-size 1280x720 --tp 2                   # VLM：每请求 2 张图，视觉编码器 + 图像 token
python3 -m accel_dse eval --model esmfold --pp 4 --pp-split layers   # PP 按层数均分（默认按代价平衡）
python3 -m accel_dse eval --model esm2-650m --seq-len 1022    # 蛋白质：批延迟、序列/s
python3 -m accel_dse eval --model boltz-1 --msa 1024 --samples 5   # 结构预测：主干 / 扩散 / 置信度，TFLOP/序列
python3 -m accel_dse validate                                 # 趋势区间 + GenZ 对照
```

所有子命令都支持 `--json`。HTTP API（`/api/eval`、`/api/compare`、`/api/layouts`、`/api/stability`、`/api/sweep`、`/api/pareto`、`/api/memory`）接受与命令行相同的场景描述，严格解析（拒绝未知字段与 NaN / Infinity）。

## 目录结构

```
accel-dse/
├── accel_dse/
│   ├── core/          # 场景、模型规格、算子图、映射、存储规划、调度、搜索、稳定性、校验
│   ├── api.py         # JSON API（Web 与 CLI 共用）
│   ├── serve.py web/  # 本地 Web 工作台
│   ├── cli.py
│   ├── mem_catalog.py # HBM / LPDDR 结构化目录（带来源标签）
│   └── data/          # 发布摘要（releases/）、模型目录、GenZ 参考值
├── docs/MODEL.md      # 建模方法、公式、校验与范围
├── docs/research/     # 存储规格调研与来源
├── scripts/           # 发布摘要抓取、GenZ 参考值生成
└── tests/
```

## 免责声明

硬件参数（频率、阵列几何、SRAM 端口、DRAM 效率、链路带宽与同步时延、MAC 效率）都是标注为「假设」的可调输入，未经硅片标定；结果用于方案之间的相对比较与趋势判断，不是性能承诺。模型参数来自公开发布，不代表任何厂商的性能声明。

## 反馈

欢迎通过 [GitHub Issues](https://github.com/kalcohol/accel-dse/issues) 报告问题或指出建模错误，附上命令行或 API 请求体便于复现。

## License

[MIT](LICENSE) © 2026 accel-dse Project

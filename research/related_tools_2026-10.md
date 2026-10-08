# 同类开源工具调研与可吸收实践（2026-10）

> 范围：只做调研和报告，**未改动** `npu_dse` 代码。
> 数据快照：GitHub ★ 和最近 push 日期取自 GitHub REST API，时间 2026-10-08 ~15:50 JST；push 日期按 API 返回的 UTC 日期记。论文数字引自原文或摘要。
> 判断口径：站在「芯片架构师评估 NPU SKU」的角度（单卡 → scale-up，HBM/LPDDR 二选一，推理专用，要能手算核对），评估各工具哪些东西**值得搬进解析模型和 Web 工作台**。

---

## 0. TL;DR

1. **我们的定位基本是独一份**：能把「NPU 微架构旋钮（PE 阵列 / SRAM 分区 / DRAM 几何）」和「多芯片推理部署（TP/PP/EP、C2C、KV fabric）」在**同一张手算可核对的卡片**里接起来，同时覆盖 LLM + DiT + 蛋白，而且零依赖。最接近的是 **LLMCompass**（微架构 + 面积/成本，但只有 LLM，没有服务级指标）和 **GenZ**（NPU 平台抽象 + 服务用例，但没有 SRAM 分区和 DRAM 几何）。
2. 2025–2026 业界工具的「主语言」已经从单点 TTFT/TPOT 换成了 **tokens/s/chip vs tokens/s/user 的 Pareto 前沿 + SLO 约束**（AIConfigurator/AISimulate、Frontier、InferenceX、Vidur-Search、DistServe）。**这是我们差得最多、性价比最高的一块。**
3. 对照中发现我们模型里有两个**会让结果偏乐观的点**，建议先修（工作量都是 S）：
   - **C2C 集合通信没有延迟项（α），而且被 `max()` 和算力/访存完全重叠**（`scaleup.py`：`t_c2c = coll/bw`，`t_stage = max(...)`）。LIMINAL（NVIDIA，2025）的核心结论是：decode 时每层 all-reduce 的**暴露同步延迟**必须做到约 1 µs，否则高带宽白给。TP=8、L=60 的模型，每 token 要做 2L≈120 次 all-reduce，单次 5 µs 就是约 0.6 ms/token，可能和 TPOT 同一量级。
   - **TP 下 KV 一律按 `/tp` 切**（`per_card_kv_bytes`）。实际上当 `tp > n_kv_heads`（GQA）时 KV 头会被**复制**；MLA 的 latent KV 被所有头共享，**TP 根本切不动**（所以 DeepSeek/SGLang/AIConfigurator/InferSim 都用 Attention-DP + MoE-EP）。对 MLA/少 KV 头的模型，TP 卡片会低估每卡 KV 容量和 decode KV 流量。
4. 最值得吸收的前 8 项见 §4，概括为：Pareto（S–M）、C2C α + 暴露同步（S）、KV 复制 / Attention-DP（S）、SLO goodput（S–M）、投机解码 / MTP（S–M）、PD 分离 xPyD 配比 + 异构 SKU（M）、按活动计的能耗表（Accelergy 式，M）、chunked prefill / 连续批处理稳态（M）。

---

## 1. 候选核实结果

| 用户列出的候选 | 核实结果 |
|---|---|
| LLMCompass (Princeton, ISCA'24) | ✅ `PrincetonUniversity/LLMCompass`（不是 princeton-nlp） |
| GenZ / GenZ-LLM-Analyzer | ✅ `abhibambhaniya/GenZ-LLM-Analyzer`（Georgia Tech + Intel，pip `genz-llm`），现在把 aiconfigurator 作为 submodule 用作 ASTRA-sim 后端 |
| LLM-Viewer | ✅ `hahnyuan/LLM-Viewer`（2024-09 后基本停更） |
| llm-analysis (cli99) | ✅ `cli99/llm-analysis`（2025-04 后停更） |
| Vidur | ✅ `microsoft/vidur` |
| LLMServingSim | ✅ `casys-kaist/LLMServingSim`，现为 2.0（ISPASS'26） |
| Calculon (NVIDIA) | ✅ 仓库是 `calculon-ai/calculon`，**以训练为主**，2024-02 后停更 |
| ASTRA-sim + Chakra | ✅ `astra-sim/astra-sim`、`mlcommons/chakra` |
| Timeloop / Accelergy | ✅ `NVlabs/timeloop`、`Accelergy-Project/accelergy` |
| ScaleSim v2/v3 | ✅ 主仓库合并为 `scalesim-project/SCALE-Sim`（v2 链接重定向到这里）；`scale-sim-v3` 是 legacy |
| MAESTRO | ✅ `maestro-project/maestro`（2024-04 后停更） |
| NeuroSim | ✅ `neurosim/DNN_NeuroSim_V2.1`（CIM 专用，对我们参考价值低） |
| Optimus / AMPeD | ⚠️ Optimus 是 **imec 内部平台**（现名 imec.kelis，有 Web UI，不开源），论文 arXiv 2407.14645；AMPeD 是 `CSA-infra/AMPeD`（训练，16★，停更） |
| LIMINAL | ⚠️ 作者是 **NVIDIA Research**，不是 Google（arXiv 2507.14397）；**只有论文，没找到开源代码** |
| Splitwise sim | ✅ `mutinifni/splitwise-sim`（不在 microsoft 组织下；2024-04 后停更） |
| DistServe sim | ✅ `LLMServe/DistServe` 仓库内的 `simdistserve/` 目录 |
| TokenSim | ✅ `pku-lemonade/TokenSim`（2026 年仍活跃，大改版） |
| APEX | ✅ `microsoft/apex_plus`（APEX+，LLM serving 并行计划模拟器）。`microsoft/APEX` 是无关的已归档仓库 |
| InferenceMAX | ✅ 已改名 **InferenceX**：`SemiAnalysisAI/InferenceX`（实测基准平台 + 开源功耗模型） |
| NVIDIA GenAI-Perf | ✅ 已被 **AIPerf** `ai-dynamo/aiperf` 取代（纯实测，与我们相关性低） |
| 「transformer inference arithmetic」 | ✅ kipply 博客（方法论）；同类系统化教材是 `jax-ml/scaling-book` 的 inference 章 |

**新发现、值得关注的（2025–2026）**：NVIDIA **AIConfigurator → AISimulate**（`ai-dynamo/aiconfigurator` / `ai-dynamo/aisimulate`）、**Frontier**（`NetX-lab/Frontier`）、**InferSim**（`alibaba/InferSim`）、**SimAI**（`aliyun/SimAI`，NSDI'25）、**ZigZag / Stream**（KU Leuven）、**PyTorchSim / ONNXim**（周期级 NPU）、**LLM-Para**（多级存储 roofline + 能耗/TCO，体量小）。

---

## 2. 对照表

### 2.1 基本信息（按与我们的相关度排序）

| # | 工具 | URL | ★（2026-10-08） | 最近 push | 粒度 | 一句话 |
|---|---|---|---|---|---|---|
| 1 | **LLMCompass** | https://github.com/PrincetonUniversity/LLMCompass | 282 | 2025-10-24 | 分块/映射器 + 解析式（systolic + vector + 缓冲层级），内含 SCALE-Sim | 面向芯片设计的 LLM 推理评估，带 **面积 / 成本模型** |
| 2 | **GenZ** | https://github.com/abhibambhaniya/GenZ-LLM-Analyzer | 127 | 2026-07-30 | 按算子的 roofline 解析式；可插 SCALE-Sim / ASTRA-sim 后端 | NPU 平台 ↔ 推理用例的需求分析，带 Streamlit 网页 |
| 3 | **AIConfigurator → AISimulate** (NVIDIA) | https://github.com/ai-dynamo/aiconfigurator · https://github.com/ai-dynamo/aisimulate | 455 / 54 | 2026-09-18 / 2026-10-08 | 实测算子库插值 + 合成（非事件级） | 在 SLA 下搜索 agg / disagg 部署，输出 **Pareto** |
| 4 | **Vidur** (MSR, MLSys'24) | https://github.com/microsoft/vidur | 690 | 2026-08-24 | 离散事件 + 算子 profiling / 回归 | 服务级仿真 + Vidur-Search（QPS/$ @SLO） |
| 5 | **LLMServingSim 2.0** (KAIST) | https://github.com/casys-kaist/LLMServingSim | 422 | 2026-10-06 | 调度器事件 + ASTRA-sim 网络 + profiling | 异构 / 分离式服务：PIM、CXL、前缀缓存、功耗 |
| 6 | **LIMINAL** (NVIDIA, 论文) | https://arxiv.org/abs/2507.14397 | — | 2025-07 | 纯解析式 | decode 极限：带宽 / 容量 / **同步延迟** / 算力 |
| 7 | **TokenSim** (PKU) | https://github.com/pku-lemonade/TokenSim | 40 | 2026-09-30 | 离散事件；实测表优先，roofline 兜底 | 软硬件协同探索；**每个数带来源等级** |
| 8 | **Frontier** (NetX) | https://github.com/NetX-lab/Frontier | 159 | 2026-09-27 | 离散事件 + 校准算子 | PDD / AFD 分离、投机 / MTP、SLA Pareto |
| 9 | **InferSim** (Alibaba) | https://github.com/alibaba/InferSim | 102 | 2026-09-04 | 解析式（FLOPs / bytes ÷ MFU 表），纯 Python 零依赖 | 模型-系统协同设计；发布了**实测对比表** |
| 10 | **ASTRA-sim 2.0 + Chakra** | https://github.com/astra-sim/astra-sim · https://github.com/mlcommons/chakra | 703 / 200 | 2026-09 / 2026-08 | 网络事件级（解析 / ns-3 后端） | 分布式系统集合通信 / 拓扑 |
| 11 | **Timeloop + Accelergy** | https://github.com/NVlabs/timeloop · https://github.com/Accelergy-Project/accelergy | 527 / 176 | 2026-07 / 2025-05 | 映射搜索 + 解析式；**能量 / 面积查表** | 加速器单层 dataflow / 能耗 / 面积 |
| 12 | **ZigZag + Stream** (KU Leuven) | https://github.com/KULeuven-MICAS/zigzag · https://github.com/KULeuven-MICAS/stream | 205 / 73 | 2026-10 | 解析式 + 映射搜索；Stream 做多核 layer-fusion（MILP） | 硬件架构-映射 DSE，含能量 / 时延 |
| 13 | **SCALE-Sim v3** | https://github.com/scalesim-project/SCALE-Sim | 526 | 2026-06-28 | 周期级 systolic（OS/WS/IS）+ Ramulator + Accelergy | 脉动阵列 GEMM，**正好可以交叉验证我们的 OS 公式** |
| 14 | **SimAI** (Alibaba, NSDI'25) | https://github.com/aliyun/SimAI | 1203 | 2026-09-20 | 训练 / 推理全栈，ns-3 网络 + Vidur 调度 | 大规模集群；1.5+ 支持 PD 分离推理 |
| 15 | **LLM-Viewer** | https://github.com/hahnyuan/LLM-Viewer | 682 | 2024-09-11 | 逐层 roofline | 网页可点层；**支持 DiT** |
| 16 | **llm-analysis** | https://github.com/cli99/llm-analysis | 494 | 2025-04-19 | 解析式（FLOPS / 内存效率旋钮） | 训练 + 推理时延 / 内存计算器 |
| 17 | **APEX+** (MSR) | https://github.com/microsoft/apex_plus | 51 | 2025-06-16 | 迭代级动态仿真 + op profiling | 自动并行计划，**同时报能耗最优和时延最优** |
| 18 | **Splitwise-sim** | https://github.com/mutinifni/splitwise-sim | 167 | 2024-04-25 | 离散事件 | 分阶段异构集群（iso-power / iso-cost） |
| 19 | **DistServe / simdistserve** | https://github.com/LLMServe/DistServe | 839 | 2025-04-06 | 系统 + 简易仿真器 | **goodput（SLO 达成率）**提出者 |
| 20 | **Calculon** | https://github.com/calculon-ai/calculon | 179 | 2024-02-22 | 解析式 | 训练协同设计；验证方法可借鉴 |
| 21 | **InferenceX** (SemiAnalysis) | https://github.com/SemiAnalysisAI/InferenceX | 1812 | 2026-10-08 | 实测基准 | Pareto 看板 + 开源系统功耗模型（BoM） |
| 22 | **scaling-book**（JAX） | https://github.com/jax-ml/scaling-book | 1469 | 2026-09-22 | 教材 / 方法论 | inference 章：roofline、时延-吞吐、分离 |
| 23 | PyTorchSim / ONNXim | https://github.com/PSAL-POSTECH/PyTorchSim · https://github.com/PSAL-POSTECH/ONNXim | 142 / 216 | 2026 | 周期级多核 NPU（BookSim + Ramulator） | NPU 微架构细节仿真 |
| 24 | MAESTRO | https://github.com/maestro-project/maestro | 261 | 2024-04-15 | 解析式（data-centric reuse） | 单层 dataflow 代价 |
| 25 | LLM-Para | https://github.com/dengls24/LLM-para | 22 | 2026-04-16 | 一阶 roofline | 多级存储（SRAM/DRAM）+ 能耗 roofline + TCO / 碳排 |

不列入主表：NeuroSim（存内计算）、AMPeD / Calculon（训练）、llm-d-inference-sim（只是 vLLM 接口 mock）、AIPerf（只做实测）、imec.kelis / Optimus（闭源）。

### 2.2 能力矩阵

图例：✅ 有 · △ 部分 / 粗略 · ✗ 无 · ? 未核实

| 工具 | 粒度 | 片上 SRAM / 层级 | DRAM 几何 | 多芯片 / 网络 | MoE | MLA | KV 容量 → 批 | 连续批 / chunked | 投机 / MTP | PD 分离 | DiT / 视频 | TP / PP / EP / SP·CP | SLO / goodput | Pareto | 能耗 | 面积 / 成本 | 实测验证 | UI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **npu_dse（我们）** | 解析式，可手算 | ✅ 三分区 + resident / staging | ✅ 通道×位宽×速率 | ✅ C2C ring/tree，IB/RoCE | ✅ 均衡 | ✅ | △ 只有 OOM 标志 | ✗ | ✗ | △ 只有 `t_kv_xfer` | ✅（示意） | ✅ / ✅ / ✅ / ✗ | ✗ | ✗ | △ TDP stub | △ 只有 $ stub | ✗（只有 handcheck） | Web 工作台 + A\|B |
| LLMCompass | 映射器 + 解析式 | ✅ 寄存器 / 本地 / 全局缓冲 | ✅ 通道×pin×速率 | ✅ FC / ring | ✗ | ✗ | △ | ✗ | ✗ | ✗ | ✗ | ✅ TP（PP 粗） | ✗ | ✗ | ✗ | ✅ 面积 + 晶圆 / 内存成本 | ✅ A100 / MI210 / TPUv3 | CLI + 图 |
| GenZ | 算子 roofline | △ 片上容量 / 带宽 | △ 只填带宽 | ✅ | ✅ | ? | ✅ 平台规模估算 | ✅ chunked | ✅ | △ | ✗ | ✅ / ✅ / ✅ / △ | △ | △ | △ 按利用率分摊功耗 | ✗ | △ | Streamlit |
| AIConfigurator / AISimulate | 实测算子插值 | ✗ | ✗ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ MTP | ✅ xPyD | ✗ | ✅ / ✅ / ✅ + Attn-DP | ✅ TTFT / TPOT | ✅ tok/s/gpu vs tok/s/user | ✗ | ✗ | ✅ | CLI + ASCII 图 + CSV |
| Vidur | 事件 + profiling | ✗ | ✗ | ✅ | △ | ? | ✅ | ✅ | △ | △ | ✗ | ✅ / ✅ / ✗ / ✗ | ✅ P90 / P99 | ✅ QPS/$ vs SLO | ✗ | △ 租金 | ✅ <9% | wandb + chrome trace |
| LLMServingSim 2.0 | 事件 + ASTRA-sim | △ 多级（CPU / CXL / PIM） | △ | ✅ | ✅ 路由 / 卸载 | ✅ | ✅ | ✅ | ✅ | ✅ | ✗ | ✅ / ✅ / ✅ / ✗ | △ 只报 p99 | ✗ | ✅ 7 部件三态 | ✗ | ✅ vLLM | CLI + 文档站 |
| LIMINAL | 解析式 | ✅（SRAM 方案） | △ 带宽 / 容量 | ✅ **同步延迟** | ✅ | ✅ | ✅ | ✗ | ✗ | ✗ | ✗ | ✅ TP / PP | △ UTPS | ✅ UTPS vs 系统 TPS | ✅ perf/W | ✅ perf/$ | ✅ MAE 7.6% | 论文 |
| TokenSim | 事件 + 表 / roofline | △ 设备 YAML 含 SRAM | ✗ | ✅ 分层 α-β | ✅ 不均 / 放置 | ? | ✅ paged | ✅ | ? | ✅ Mooncake | ✗ | ✅ / ✅ / ✅ / ✗ | ✅ | △ | ✗ | ✗ | ✅ 多 GPU | CLI |
| Frontier | 事件 + 校准 | ✗ | ✗ | ✅ | ✅ | ✅ | ✅ 分级缓存 | ✅ | ✅ | ✅ PDD + AFD | ✗ | ✅ | ✅ | ✅ | ✗ | △ 异构 $ | ✅ | CLI |
| InferSim | 解析式 ÷ MFU | ✗ | ✗ | △ | ✅ DeepEP | ✅ | △ bs = tgs×tpot | ✗ | ✗ | △ P / D 分开 | ✗ | Attn-DP + EP | ✅ 目标 TPOT | ✗ | ✗ | ✗ | ✅ 公开表 | CLI |
| Timeloop + Accelergy | 映射 + 查表 | ✅ 任意层级 | △ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✅ ERT | ✅ ART | ✅ | YAML |
| ZigZag / Stream | 映射 + 解析式 | ✅ | △ | △ 片上多核 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✅ | △ | ✅ | YAML / MCP |
| SCALE-Sim v3 | 周期级 | ✅ SRAM 缓冲 | ✅ Ramulator | △ 多核 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✅ Accelergy | ✗ | △ | CSV |
| LLM-Viewer | roofline | ✗ | ✗ | ✗ | ✗ | ✗ | △ 峰值内存 | ✗ | ✗ | ✗ | ✅ DiT | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 网页 |
| llm-analysis | 解析式 | ✗ | ✗ | ✅ | △ | ✗ | ✅ | ✗ | ✗ | ✗ | ✗ | ✅ / ✅ / ✅ / ✗ | △ | ✗ | ✗ | △ GPU 时 | △ | CLI / Python |
| APEX+ | 迭代级仿真 | ✗ | ✗ | ✅ | ✅ | ? | ✅ | ✅ | ✗ | ✗ | ✗（Whisper ✅） | ✅ | ✅ | ✗ | ✅ 能耗最优计划 | ✗ | ✅ vLLM / SGLang | CLI |

---

## 3. 逐个工具笔记（重点放在「对我们有什么用」）

### 3.1 LLMCompass —— 最接近的「芯片设计」同行
- URL：https://github.com/PrincetonUniversity/LLMCompass；论文：https://parallel.princeton.edu/papers/isca24_llmcompass.pdf （arXiv 2312.03134）
- **建模**：硬件按「system → device → compute chiplet → core → sublane（systolic array + vector unit + 寄存器堆）→ 本地 SRAM → 全局缓冲 → 内存通道」层层描述。内置**映射器**会搜索 tiling / 调度（GPT-3 175B 在 4×A100 上跑 26,400 轮，约 16 分钟）。互连用 link 参数（带宽 / 延迟 / flit / header）+ 拓扑（FC / RING）描述。
- **工作负载**：只有 LLM 的 prefill / decode（GPT 类），带 TP。没有 MoE、MLA、服务调度。
- **验证**：A100 / MI210 / TPUv3。算子级平均误差约 10.4–10.9%，端到端 4.1%（prefill 0.69%，decode 7.5%）。
- **面积 / 成本**：从晶体管数（开源设计 / 流片数据）+ CACTI 推 SRAM 面积 + die photo 标注的 PHY / 控制器面积 + 晶圆成本供应链模型（ttm-cas）；内存成本用 DDR 现货价和 HBM2e 估计。GA100 / Aldebaran 面积误差 5.1% / 8.1%。
- **结论**：用 DDR 替换 HBM、降低算力，perf/$ 最多比 A100 高 3.41 倍——这正是我们 LPDDR SKU 的论点，**可以在 FINDINGS 里引用作外部佐证**。
- **可借鉴**：①把 **vector unit 当成独立资源**来算 softmax / LN / 激活（替代我们的 `non_gemm_overhead` 百分比）；②做**面积 / 成本模型**（PHY 不随工艺缩放这一点很关键）；③验证方法：先做算子级、再做端到端，分 prefill / decode 报误差。

### 3.2 GenZ（Georgia Tech / Intel）
- URL：https://github.com/abhibambhaniya/GenZ-LLM-Analyzer；论文 arXiv 2406.01698；在线版 https://genz-llm-analyzer.streamlit.app/
- **建模**：按算子的 roofline。`System` 的参数有 `flops`、`mxu_shape`、片上内存容量 / 带宽、片外容量 / 带宽、`external_mem_bw`、算力 / 内存 / 通信三个效率、片间链路带宽 + **延迟**、拓扑、`parallelism_hierarchy="TP{}_EP{}_PP{}"`。算力引擎可以换成 SCALE-Sim / 真实硬件 / profiling，集合通信可以换成 ASTRA-sim。
- **工作负载**：prefill、decode、**chunked**（`llm_chunked.py`）、**投机解码**（`llm_spec_decode.py`）、Mamba、多家模型族；`platform_size.py` 用来反推平台规模。
- **能耗**（`power.py`）：总功耗按 Static / Compute / Memory / Network 拆比例，按每个算子的利用率加权积分得到能量。和我们的 stub 一样简单，但多了一步「按利用率分摊」。
- **UI**：Streamlit，4 个页面：用例对比 / 模型对比 / 平台对比 / **技术对比**（比如 HBM 带宽 +10% 的影响）。
- **可借鉴**：①**用例预设**（聊天 / 摘要 / RAG / 代码 → ISL/OSL/beam），见论文；②投机解码和 chunked prefill 的解析写法；③利用率加权的功耗；④「技术敏感度」页（某参数 ±10% → 指标弹性），可以做成我们的 tornado 图。

### 3.3 AIConfigurator → AISimulate（NVIDIA Dynamo）
- URL：https://github.com/ai-dynamo/aiconfigurator（已进入只维护状态）→ https://github.com/ai-dynamo/aisimulate；论文 arXiv 2601.06288
- **建模**：在目标 GPU + 框架上采集算子耗时（GEMM / attention / MLA / MoE / NCCL / P2P…），插值 / 外推后合成端到端时间。在这之上建模 in-flight batching（agg）和 **disagg（xPyD）**，几秒内搜完几千个配置。
- **并行**：TP、PP、EP / ETP、**Attention-DP**；调度方式：static / agg / disagg / **MTP**。
- **输出（最值得抄）**：给定 `--isl/--osl/--prefix/--ttft/--tpot`，输出
  - **Pareto：tokens/s/gpu_cluster vs tokens/s/user**（agg 和 disagg 两条曲线叠画）；
  - Top-N 配置表：tokens/s/gpu、tokens/s/user、TTFT、request_latency、concurrency、replicas、gpus/worker、parallel、bs；
  - `recommend` 模式：给定目标 QPS + SLA，**反推最少卡数**（采购口径的 sizing）。
- **可借鉴**：Pareto 的坐标轴定义、agg vs disagg 叠图、Top-N 表的列设计、「满足 SLA 的最少卡数」这个问题的表述。芯片架构师可以直接改问「满足 SLA 需要多少颗 SKU-A / SKU-B」。

### 3.4 Vidur（Microsoft，MLSys'24）
- URL：https://github.com/microsoft/vidur；论文 arXiv 2405.05465
- **建模**：离散事件仿真；算子先 profiling，再用回归模型预测；调度器可插拔（vLLM、Sarathi chunked prefill 等）；输入是真实 trace（Azure）。
- **指标**：TTFT、TPOT / TBT、E2E、batch size 随时间变化；Vidur-Search 在 **TTFT-P90 < 2 s、TBT-P99 < 200 ms** 的约束下最大化 **QPS/$**（容量定义为「P99 调度延迟 < 5 s 时的最大 QPS」）。
- **验证**：时延误差 < 9%，在 85% 容量以内通常 < 5%（越接近满载误差越大）。
- **可借鉴**：①SLO 用分位数表达（P90 / P99），而且 TTFT 和 TBT 用不同分位数；②容量用「排队不发散」来定义；③chrome trace 输出（我们可以对单个 token 步导出算力 / 访存 / C2C 的甘特图）。

### 3.5 LLMServingSim 2.0（KAIST，ISPASS'26）
- URL：https://github.com/casys-kaist/LLMServingSim；论文 arXiv 2602.23036；文档 https://llmservingsim.ai
- **建模**：Python 前端照搬 vLLM 的连续批调度，后端是 ASTRA-sim + Chakra；硬件时延来自 vLLM 逐层 profiler。覆盖异构加速器、**CPU / CXL / PIM 内存分层**、MoE 路由 + 专家卸载、投机解码（按模型公布的接受率）、前缀缓存（RadixAttention；NPU 本地 / CPU 共享 / CXL 池）、TP/PP/EP/DP、P/D 分离。
- **功耗**：每个节点按 7 部件拆（加速器 / CPU / DRAM / 互连 + 交换机 / NIC / 存储 / 其他）。加速器用 **idle / active / standby 三态**；**DRAM 和链路的能量按传输字节计**；其余按常数功率。
- **验证**：RTX 4090 profile 与真实 vLLM 的平均 TTFT / TPOT 误差在 1% 以内；58 个 TP/PP/EP/DP 组合场景做回归。
- **可借鉴**：①**能量 = Σ 字节 × pJ/bit + 状态功率 × 时间**，这正好能直接用上我们已经算出来的 W / KV / act / C2C 字节；②多级 KV（HBM ↔ LPDDR / CXL）和前缀缓存，对「LPDDR 当容量层」的 SKU 论证有用。

### 3.6 LIMINAL（NVIDIA Research，论文）
- 论文：https://arxiv.org/abs/2507.14397 （没有开源代码）
- **模型**：`T_batch = max(T_compute, T_mem) + T_exposed`。应用抽象成「数据量 / 计算量 / 同步需求」，硬件抽象成「算力 / 带宽 / 容量 / 集合通信延迟」。覆盖 HBM3 / HBM4 / 3D-DRAM / 纯 SRAM / 晶圆级方案。端到端 MAE 7.6%。
- **结论**：①一个模型实例要几百 GB 到 1 TB 以上的容量；②单用户吞吐（UTPS）取决于带宽；③**all-reduce 暴露延迟要做到约 1 µs**，否则带宽再高也用不上；④DRAM 方案在 perf/W、perf/$ 上占优（相对纯 SRAM）；⑤2000+ UTPS 容易，10,000+ 需要算法层面的变化。
- **可借鉴（重要）**：**把同步延迟单列为加性项**，不能用 `max()` 吃掉。我们现在 `t_stage = max(t_comp, t_mem, t_c2c, …)`，C2C 只有 bytes/BW。建议改成 `t_stage = max(t_comp, t_mem) + t_sync_exposed`，其中 `t_sync = n_collectives × (α_hop × hops + bytes/BW) × (1 − hide)`。另外它的 **UTPS vs 系统吞吐**图就是一张 Pareto。

### 3.7 TokenSim（PKU，2025–2026 大改版）
- URL：https://github.com/pku-lemonade/TokenSim；论文 arXiv 2503.08415
- **建模**：离散事件，覆盖 static / dynamic / paged-attention 批处理（含 vLLM V1 chunked prefill 的 token budget）、前缀复用、hybrid / 分离式 P/D、TP/PP/DP/EP、**MoE 放置和路由分布**、Mooncake KV store + SSD 卸载，分层拓扑（chip / node / rack / cluster）用 α-β 集合通信模型。设备目录里甚至有 **Groq TSP**（纯 SRAM）。
- **算子时延的解析顺序**：精确命中 → 插值 → 有界外推（≤16×）→ 解析 roofline 兜底；每一步记 `match_type`，每个数都带 `source_id` 和**证据等级**；缺失形状导出成下一轮采集清单。
- **可借鉴**：这套**来源 / 证据等级体系**和我们的 assumed / derived 标签同源，但更细。建议把 MetricsCard 的每个关键量都挂上 `provenance ∈ {derived, assumed, calibrated(src), public-spec(src)}`，在 UI 上用徽章显示。

### 3.8 Frontier（NetX-lab，2026）
- URL：https://github.com/NetX-lab/Frontier
- 离散事件；co-location、**PDD**、**AFD（Attention-FFN 分离：prefill / decode-attention / decode-FFN 三种角色）**；投机 / MTP、前缀缓存、chunked prefill、分级缓存、CUDA Graph；用例包括 **SLA 约束下的 Pareto 搜索**、**异构 GPU 分配**（便宜卡在分离角色里是否划算）。
- **可借鉴**：AFD 对 MoE 大模型 + NPU 很有意义：attention 部分吃 KV 带宽 / 容量（适合 LPDDR 大容量），FFN 部分吃权重带宽 / 算力（适合 HBM）。**这是一个可以让 HBM 与 LPDDR 两种 SKU 互补的部署形态**，值得作为 PD 分离之后的扩展（L）。

### 3.9 InferSim（Alibaba）
- URL：https://github.com/alibaba/InferSim
- **纯 Python、零依赖、解析式**（理念和我们最像）：TTFT / TPOT / TGS = FLOPs ÷ (FLOPS × MFU)，MFU 查 kernel benchmark 表；支持 MHA / GQA / MLA、GroupedGEMM MoE、**DP-Attn + EP-MoE**、DeepEP normal / low-latency、TBO（two-batch overlap）。
- **验证方式（值得照搬）**：README 里直接放 **Actual vs Sim 对照表**，例如 DeepSeek-V3 / H800：prefill TGS 实测 7839 / 仿真 9034，decode 2324 / 2675；数据来源写明（deepseek-ai/profile-data、SGLang 实测）。
- **可借鉴**：①`bs = target_tgs × target_tpot` 这种由 SLO 反推 batch 的写法；②公开数据对照表作为回归测试；③Attention-DP。

### 3.10 ASTRA-sim 2.0 + Chakra
- URL：https://github.com/astra-sim/astra-sim、https://github.com/mlcommons/chakra
- 分层网络、集合通信算法（ring / tree / halving-doubling / 多维）、分离式系统；输入是 Chakra 执行图。GenZ 和 LLMServingSim 都把它当网络后端。
- **可借鉴**：多维拓扑（例如 2D torus / 板内全互连 + 板间 ring）下集合通信的**分层 α-β 公式**。我们现在只有单层 ring / tree，可以加 `topology ∈ {ring, fc, 2d-torus, switch}` 和每跳延迟 α。不建议直接集成（重、而且不能手算）。

### 3.11 Timeloop + Accelergy（以及 SCALE-Sim v3 / ZigZag / Stream）
- URL：https://github.com/NVlabs/timeloop、https://github.com/Accelergy-Project/accelergy、https://github.com/scalesim-project/SCALE-Sim、https://github.com/KULeuven-MICAS/zigzag、https://github.com/KULeuven-MICAS/stream
- **Accelergy 的做法**：能量 = Σ(动作次数 × 每动作能量)，动作包括 MAC、SRAM 读 / 写、DRAM 读 / 写、NoC 传输。每动作能量和面积来自 **ERT / ART 表**（插件：CACTI、Aladdin、查表、NeuroSim）。**这和我们「不伪造 PDK」的章程不冲突**：表由用户提供或标成 assumed，引擎只做乘加。
- **SCALE-Sim v3**：周期级脉动阵列（OS / WS / IS 三种 dataflow），带 SRAM 缓冲、Ramulator、Accelergy、多核、稀疏。**纯 Python**，可以用来**交叉验证我们 `cycles = ceil(M/R)·ceil(N/(C·eng))·K` 的 OS 公式**，以及 M=1 时 util = 1/R 的结论（工作量 S，对审计价值很高）。
- **ZigZag / Stream**：解析式映射搜索 + 能量 / 时延，Stream 做多核 layer-fusion（MILP 决定张量放置）。和我们的「SRAM resident R 层」策略可以互相对照。不建议集成。

### 3.12 SimAI（Alibaba，NSDI'25）
- URL：https://github.com/aliyun/SimAI
- 训练 / 推理全栈；1.5 起支持多请求推理 + P/D 分离（调度模块改自 Vidur）；1.6 加了推理显存建模（参数 + KV）、P/D 独立显存预算。网络用 ns-3，集合通信 SimCCL 已对齐 NCCL v2.30。
- **可借鉴**：P/D 分离时**各自的容量预算**（prefill 池不用存长期 KV，decode 池要存所有在途请求的 KV），这和我们的 OOM 检查直接相关。

### 3.13 LLM-Viewer / llm-analysis / scaling-book（roofline 计算器类）
- LLM-Viewer（https://github.com/hahnyuan/LLM-Viewer，survey arXiv 2402.16363）：逐层 roofline + 峰值内存（按数据依赖追踪）；可视化网络图，点层看详情；**支持 DiT**；明确声明「只看相对关系」。
- llm-analysis（https://github.com/cli99/llm-analysis）：FLOPS 效率 / 内存效率两个旋钮 + TP/PP/DP/EP；回答「满足时延约束下的最优 batch / 并行」；README 写清楚了推理假设。
- scaling-book（https://github.com/jax-ml/scaling-book）：inference 章讲 roofline、**时延-吞吐随 batch 的取舍曲线**、分离式服务和投机。方法论最清楚。
- **可借鉴**：**per-op roofline 散点图**（横轴算术强度，纵轴 TOPS，叠画 HBM / LPDDR / SRAM 三条屋顶），对芯片架构师很直观；我们已经有逐项分解数据，只差图（S）。

### 3.14 APEX+ / Splitwise-sim / DistServe（服务策略类）
- APEX+（https://github.com/microsoft/apex_plus）：给定集群和请求分布，枚举并行计划；同时给出**时延最优和能耗最优**两个计划（能耗最多低 45%）；用 vLLM / SGLang 验证；支持 Whisper（编码器-解码器）。→ 借鉴「**能耗最优 vs 时延最优**」双目标。
- Splitwise-sim（https://github.com/mutinifni/splitwise-sim）：按阶段拆分到不同机型（prefill 用高算力，decode 用功率封顶的卡），在 **iso-power / iso-cost** 下比较吞吐。→ 借鉴「**等功耗 / 等成本对比**」口径。prefill SKU / decode SKU 的设定和我们 HBM vs LPDDR 的讨论直接相关。
- DistServe（https://github.com/LLMServe/DistServe，`simdistserve/`）：**goodput = 在 SLO 达成率 ≥ X%（比如 90%）时能承受的最大请求率**；按 P / D 分别选并行度。→ 借鉴 goodput 定义。

### 3.15 InferenceX（SemiAnalysis，原 InferenceMAX）
- URL：https://github.com/SemiAnalysisAI/InferenceX；看板 https://inferencex.semianalysis.com
- 实测基准平台：多框架 × 多硬件，**以 Pareto 前沿（吞吐 / 卡 vs 交互性）**为主图；2026 年加了 AgentX（1M+ 长上下文、多轮）。附带 `power_model/`：从 GPU 级功耗 + BoM 估算机箱 / 机柜总功耗（NIC、交换机、CPU 卸载……）。
- **可借鉴**：①Pareto 的标准坐标；②把「卡功耗」扩成「系统功耗」的 BoM 拆法（对 $/MTok 口径有用）；③公开实测曲线可以当我们 GPU 类预设的 sanity check（拿 H100 / H200 / MI 系列的公开带宽 / 算力填进我们的模型，看 decode tok/s 落在实测 Pareto 的哪一段）。

### 3.16 周期级 NPU 仿真（PyTorchSim / ONNXim / MAESTRO）
- PyTorchSim（MICRO'25）：RISC-V NPU + PyTorch2 编译后端 + tile 级仿真（TOG），用 BookSim / Ramulator2 建模共享 DRAM / NoC。ONNXim：多核 NPU 周期级，支持自回归生成和迭代级批处理。
- **对我们**：不吸收进主模型（违反「可手算」章程）。但可以作为**外部参照**：挑 1–2 个小模型 / 小形状，用 ONNXim 跑我们的 SKU 参数，量化「解析模型 vs 周期级」的偏差范围，写进 DECISIONS（L，可选）。

---

## 4. 可吸收的维度 / 实践（按优先级排序）

图例：**价值**指对「芯片架构师比较 NPU SKU」的价值（高 / 中 / 低）；**工作量**：S（≤2 天，纯公式 + 卡片字段）、M（约 1 周，新模块 + UI 视图）、L（多周）；**现状**：✅ 已有 / △ 部分 / ✗ 无。

### P0（建议下一版就做）

**① 吞吐-交互性 Pareto：tokens/s/chip vs tokens/s/user**　价值：高｜工作量：S–M｜现状：△（有 `sweep-batch`、并行矩阵，`est_tokens_per_s` 只在开启 econ 后才算）
- 来源：AIConfigurator、InferenceX、LIMINAL（UTPS）、scaling-book、Frontier。
- 怎么接进解析模型：对每个 (SKU, 并行, batch B, ctx) 点已经有 TPOT(B)。定义 `tok/s/user = 1/TPOT`、`tok/s/chip = B/(TPOT·chips)`。B 的上限由**容量决定**：`B_max = floor((cap − W_card − act_reserve) / KV_per_seq_card)`（见 ③）。扫 B ∈ {1…B_max} × 并行矩阵 → 取上包络就是 Pareto。可选第二张图：`tok/s/chip` vs TTFT。
- UI：Compare 标签新增「Pareto」子页，零 CDN 的 SVG 散点图；多个 SKU 用不同颜色叠画（A|B 天然就是两条曲线），悬停显示 (B, tp/pp/ep, wall)。点「拐点」可以一键 Pin 成 A / B 卡。
- 为什么对 NPU 特别重要：OS 阵列的 util(M) 随 B 上升，B 小时 LPDDR SKU 撞带宽墙，B 大时撞容量墙。**Pareto 能把「带宽墙 → 算力墙 → 容量墙」三段一图看完**，比单点 TPOT 更能说明 SKU 差异。

**② C2C 同步延迟 α + 暴露同步项（LIMINAL 式）**　价值：高｜工作量：S｜现状：✗（C2C 只有 bytes/BW，被 `max()` 和算力 / 访存重叠）
- 改法：`t_stage = max(t_comp, t_mem, t_fabric) + t_sync_exposed`，其中 `t_sync = Σ_collectives (α_link·steps(algo, tp) + bytes/BW) × (1 − c2c_hide)`；ring 的 steps = 2(tp−1)，tree 的 steps ≈ 2·log2(tp)。新增 knob `c2c_latency_us`（assumed，默认可以给 1–5 µs 的扫描范围）。保留旧口径作为 `overlap_mode = "max"`，保证 handcheck 不变。
- 价值：decode 时 TP 每层 2 次 all-reduce，数据量很小，**几乎纯延迟**。不加 α 会系统性高估多芯片 decode 的扩展效率（`scale_efficiency`）。这也是 C2C PHY / SerDes 选型（低延迟 vs 高带宽）的直接依据。
- UI：MetricsCard 的 breakdown 里新增 `sync_exposed`；`sweep` 支持 `c2c_latency_us`。

**③ TP 下 KV 复制修正 + Attention-DP 选项**　价值：高｜工作量：S｜现状：✗（`per_card_kv_bytes = KV/(tp·pp)`）
- 规则：GQA 的每卡 KV 头数 = `max(1, n_kv_heads/tp)`（tp > n_kv 时复制，复制因子 = tp/n_kv）；**MLA latent 不随 TP 切**（每卡都存完整的 L/pp 层 latent）。新增 `attn_parallel ∈ {tp, dp}`：dp 时每卡存 B/dp 个请求的完整 KV，attention 权重复制，FFN / MoE 走 TP / EP（AIConfigurator、InferSim、SGLang DeepSeek 的做法）。
- 价值：对 GLM / Kimi / DeepSeek 类 MLA 模型和 n_kv=4/8 的模型，多芯片 KV 容量和 decode KV 流量现在被低估了 tp/n_kv 倍。这直接影响 HBM vs LPDDR 容量结论。
- UI：卡片新增 `kv_replication_factor`，>1 时显示警示徽章。

**④ SLO 约束 goodput（解析版）**　价值：高｜工作量：S–M｜现状：✗
- 来源：DistServe（goodput）、Vidur-Search（P90 TTFT / P99 TBT）、AIConfigurator（`--ttft --tpot`）、InferSim（`bs = tgs × tpot`）。
- 解析写法（不做事件仿真）：给定 `TTFT_SLO`、`TPOT_SLO`、ISL/OSL，在 Pareto 点集里筛出 `TPOT(B) ≤ TPOT_SLO` 且 `TTFT_eff ≤ TTFT_SLO` 的点，取 tok/s/chip 最大者 = **goodput/chip**。`TTFT_eff = TTFT_prefill + 排队项`：排队项先用 M/D/1 近似（ρ = λ·t_prefill），或者先只用「prefill 占用率 ≤ ρ_max」这个保守约束。再用 Little's law 换算 `QPS = B / (TTFT + OSL·TPOT)`。
- 输出：`goodput_tok_s_per_chip@SLO`、`max_qps@SLO`、`binding_constraint ∈ {TPOT, TTFT, capacity}`。配合 econ 字段可以得到 `$ / MTok @SLO`、`J / tok @SLO`。
- UI：顶部场景栏加两个 SLO 输入框，Pareto 图上画 SLO 竖线，可行区高亮。**对 SKU 评审是最有力的单一数字**（「在 50 ms TPOT 下每颗芯片能出多少 token/s」）。

### P1（下一阶段）

**⑤ 投机解码 / MTP**　价值：高（对 NPU 尤其高）｜工作量：S–M｜现状：✗
- 来源：GenZ `llm_spec_decode`、AIConfigurator MTP、Frontier、LLMServingSim（按模型公布的接受率）。
- 解析：草稿长度 k、期望接受长度 τ（用户 assumed，或者按公布的 MTP 接受率），草稿开销 `t_draft`（MTP 头 / 小模型，单独形状）。验证步是 M = B·(k+1) 的「小 prefill」：权重读一次、KV 读一次，**OS 阵列 util 从 1/R 上升到约 (k+1)/R**。`TPOT_eff = (t_verify(B,k) + t_draft) / τ`。
- 价值：NPU 在 M=1 时阵列浪费最严重，投机解码是**把闲置算力换成 TPOT 的头号手段**，能改变「算力该配多少」的结论（算力 / 带宽配比）。很适合在 FINDINGS 里写一节。
- UI：⚙ 高级抽屉加 `spec_k` / `accept_len` / `draft_model`；卡片显示 `tpot_eff` 和 util 提升。

**⑥ PD 分离 xPyD 配比 + 异构 SKU**　价值：高｜工作量：M｜现状：△（有 `t_kv_xfer` 和 IB/RoCE fabric）
- 来源：DistServe、Splitwise、AIConfigurator（xPyD）、SimAI（P / D 独立显存预算）、Frontier（异构分配）。
- 解析：prefill 池吞吐 `R_p = chips_p · ISL / TTFT_card`（token/s），decode 池 `R_d = B·chips_d/(TPOT·chips_per_inst)`。按请求率匹配：`x : y = (λ·ISL/R_p) : (λ·OSL/R_d)`。KV 传输占 fabric 带宽 `λ·KV_bytes(ISL)` 的比例要单独校验。**允许 P 池和 D 池用不同 SKU**（例如 prefill 用高 TOPS + 小 LPDDR，decode 用 HBM 或大 LPDDR）。
- 输出：`tok/s/chip_total`、`P:D 比`、`fabric_util`；和 agg 对比（AIConfigurator 风格的「disagg 是 agg 的 1.67 倍」）。
- 价值：直接回答「要不要做专门的 prefill 芯片 / decode 芯片」。业界已经出现 prefill 专用（GDDR 类）产品形态，这个问题对 SKU 规划很现实。

**⑦ 按活动计的能耗（Accelergy 式 ERT 表）**　价值：高｜工作量：M｜现状：△（只有 TDP × 时间的 stub）
- 来源：Accelergy / Timeloop、SCALE-Sim v3 + Accelergy、LLMServingSim 2.0（DRAM / 链路按字节计 + 加速器三态）、GenZ（按利用率分摊）、LIMINAL（perf/W）。
- 解析：我们已经有完整的「动作计数」：MAC 数（按 dtype）、SRAM 读写字节（三个分区）、DRAM 字节（W / KV / act，分 HBM / LPDDR）、C2C 字节、fabric 字节。新增一张 **用户提供 / assumed 的 ERT 表**：`pJ/MAC[dtype]`、`pJ/bit_SRAM`、`pJ/bit_HBM`、`pJ/bit_LPDDR`、`pJ/bit_C2C`、`pJ/bit_fabric`、`P_static_W`、`P_idle_W`。`E_token = Σ count × e + P_static × TPOT / B`。默认关闭，示例表标「literature-range, assumed」。
- 价值：HBM vs LPDDR 的能效差异（每 bit 能量 + 静态功耗）是 SKU 决策的关键维度之一，而现在的 TDP stub 反映不出来。分解后能回答「J/token 里 DRAM 占多少」。
- UI：Energy 区新增**能耗堆叠条**（compute / SRAM / DRAM / C2C / static）；A|B 对比能耗分解。
- 面积 / 成本（LLMCompass 式，L）：PE 阵列 mm²/MAC、SRAM mm²/MiB、PHY mm²/通道（不随工艺缩放）→ die 面积 → 晶圆成本 / 良率 → $/die。**按章程只接受用户表**，可以作为 v0.3x 的可选模块。

**⑧ 连续批处理 / chunked prefill 稳态模型（+ 可选 KV 占用时间线）**　价值：中–高｜工作量：M（稳态）/ L（时间线）｜现状：✗
- 来源：Vidur（Sarathi）、TokenSim（token budget）、LLMServingSim、GenZ `llm_chunked`。
- 解析（稳态，可手算）：每步 token 预算 T_b = B_decode + C_chunk。一步的时间 = 权重读一次（共享）+ decode 的 KV 读 + chunk 的算力（M = C_chunk + B_decode 的 GEMM），作为 max / 加和。TBT = t_step，TTFT = ceil(ISL/C_chunk)·t_step。扫 C_chunk 可以看到 **TTFT ↔ TBT 的干扰曲线**。
- 可选时间线：非常轻的离散时间循环（泊松到达 + 固定 ISL/OSL 分布），输出 KV 占用 / batch 随时间变化、P50/P99 TTFT。作为独立的 `serve-sim` 子命令，**不进 MetricsCard 主路径**，以免破坏可手算性。
- 价值：NPU 的 OS 阵列在混合步里能用 decode 的空闲算力做 prefill，这是 NPU 侧的「免费午餐」量化。

### P2（有价值，排在后面）

| # | 维度 | 来源 | 价值 | 工作量 | 现状 | 接入方式 |
|---|---|---|---|---|---|---|
| ⑨ | **Attention 细化**：decode attention 实际是 GEMV（M = 每个 KV 头的 q 头数 g），阵列 util ≈ g/R；FlashAttention 融合（S² 不落 DRAM）vs 非融合（S²·bytes 落盘）；精确因果三角（现在是「矩形×½」）；滑窗 / hybrid 线性注意力的 KV 上限；MLA absorbed / non-absorbed 两种算法的 FLOPs | LLM-Viewer（FlashAttention 开关）、GenZ（operator fusion）、AIConfigurator（MLA BMM） | 中–高 | M | △（有 `_attn_score_flops`，因果按 ½） | `attn_impl ∈ {fused, unfused}`、`attn_util_mode`；hybrid / DSA 从 metadata 升级为 KV 公式 |
| ⑩ | **非 GEMM 算子走 vector unit**：softmax / RoPE / norm / 激活按元素数 ÷ vector 吞吐（ops/cycle），并可设置与阵列重叠的比例 | LLMCompass（systolic + vector 双资源） | 中 | M | △（`non_gemm_overhead` 百分比） | 新增 `vector_ops_per_cycle` knob；对长 ctx softmax、DiT 大 T 特别有用 |
| ⑪ | **长上下文 CP / SP**：prefill ring-attention（按序列切，KV 环传）；decode 时 KV 跨卡分片 + partial softmax 合并（每层一次小 all-gather / reduce） | Optimus（SP）、GenZ（TODO SP）、scaling-book | 中–高（128k+ ctx） | M | ✗ | 新增 `cp` 维度：`tp·pp·ep·cp == chips`；KV/cp，加一次合并通信（同样要算 α） |
| ⑫ | **MoE 现实化**：负载不均因子（hot-expert 系数 ≥1，乘在最慢 rank 上）；A2A 走低延迟模式（α 主导）；TBO 双批重叠 | TokenSim（路由分布）、InferSim（DeepEP、TBO）、LLMServingSim | 中 | S | △（均衡路由） | `moe_imbalance`、`a2a_latency_us`、`tbo` knob |
| ⑬ | **验证方法学** | LLMCompass（算子 → E2E，分 P / D 报误差）、InferSim（Actual vs Sim 公开表）、LIMINAL（MAE）、Vidur（容量区间分段）、TokenSim（证据等级） | 高（可信度） | S–M | ✗（只有 handcheck / 守恒） | 见 §5 |
| ⑭ | **多级 KV / 前缀缓存**：HBM ↔ LPDDR / CXL / SSD 分层，前缀命中率 h 降低有效 ISL | LLMServingSim、TokenSim（Mooncake）、Frontier | 中（LPDDR 当容量层时重要） | M | △（只有 IB/RoCE 远程 KV） | `prefix_hit`、`kv_tier2` 参数；TTFT 按 (1−h)·ISL 计 |
| ⑮ | **per-op roofline 图** | LLM-Viewer、GenZ、imec.kelis | 中（直观） | S | ✗ | 报告和 Web 新增 SVG：横轴算术强度，纵轴有效 TOPS，叠画 SRAM / DRAM 屋顶 |
| ⑯ | **用例预设**（chat / RAG / code / agentic 长上下文多轮 → ISL/OSL/prefix） | GenZ、InferenceX AgentX、AIConfigurator `--prefix` | 中 | S | △（有硬件场景预设，没有负载预设） | 和 `--preset` 并列加 `--usecase` |
| ⑰ | **系统级功耗 / 成本口径**：卡 → 机箱 / 机柜 BoM（NIC、交换机、CPU、冷却 / PUE），iso-power / iso-cost 对比 | InferenceX power_model、Splitwise、LLMServingSim 7 部件 | 中 | S–M | △（只有卡级 stub） | econ 加 `system_overhead_W`、`pue`；sweep 加「等功耗 / 等成本归一化」列 |
| ⑱ | **灵敏度 tornado 图**（每个参数 ±10% → TPOT / goodput 弹性） | GenZ「技术对比」页 | 中 | S | △（有单维 sweep） | 复用 `/api/sweep`，自动跑 ±10%，输出排序条形图 |
| ⑲ | **能耗最优 vs 时延最优双目标** | APEX+ | 中 | S（依赖 ⑦） | ✗ | 并行矩阵排序键加 `J/tok`、`goodput/W` |
| ⑳ | **AFD（Attention-FFN 分离）** | Frontier、SimAI | 中（MoE + 异构 SKU） | L | ✗ | 在 ⑥ 的基础上把 decode 再拆成 attention 池（KV 容量型）和 FFN 池（权重带宽型） |

### 明确不建议吸收

- **基于 profiling 的算子表**（Vidur / AIConfigurator / TokenSim / InferSim 的核心）：我们评估的是**还不存在的硅**，没有东西可 profile。只借鉴「表优先、解析兜底、标注来源」的**流程**，等硅后标定时用 `--calib` 注入。
- **周期级 DRAM / NoC**（Ramulator / BookSim / ONNXim / PyTorchSim）、**ns-3 网络**（SimAI）：违反可手算章程。只作为一次性外部交叉验证（可选，L）。
- **完整事件仿真进主路径**：会破坏 MetricsCard 的确定性和可审计性。若要做（⑧ 时间线），放在独立子命令里。
- **由工艺节点伪造面积 / 功耗**（imec.kelis 的节点缩放、LLMCompass 的 7nm 晶体管模型）：只接受用户表，坚持章程。

---

## 5. 验证方法学建议（对应 ⑬）

1. **公式级交叉验证（S）**：用 SCALE-Sim v3（纯 Python，OS dataflow）跑 10–20 个 GEMM 形状（含 M=1、M=B、K/N 不整除），和 `npu.py` 的 `cycles` / `util` 对比，结果写进 `tests/`（标注外部工具版本）。证明我们的 OS 公式和周期级一致，或者给出偏差范围。
2. **公开数据回放（S–M）**：建一个 `validation/` 目录，用 GPU 类「伪 SKU」（公开峰值 TOPS / 带宽 / 容量）回放：
   - InferSim README 里的 DeepSeek-V3 / H800、Qwen3 / H20 的 prefill / decode TGS；
   - deepseek-ai/profile-data；
   - InferenceX 看板上若干 (B, tok/s/user, tok/s/gpu) 点；
   - LIMINAL / LLMCompass 论文里的 decode 时延点。
   像 LLMCompass 一样**按 prefill / decode 分开报 MAE**，像 Vidur 一样**按负载区间分段报**。目的不是标定 NPU，而是证明「roofline + 效率」骨架在已知硬件上的误差区间（典型 5–20%）。
3. **来源 / 证据等级（S）**：仿照 TokenSim 的 `source_id + evidence grade`，在 MetricsCard 的 `assumptions` 里给每个输入量挂上 `{derived | assumed | public-spec:<url> | calibrated:<file>}`，在 UI 上用徽章显示，A|B 对比时提示「两边证据等级不同」。
4. **回归场景库（S）**：仿照 LLMServingSim 的 58 个场景 `validate.sh`，把 parallel 矩阵、MLA / MoE、video / protein 的关键卡片落盘为 golden JSON，防止后续加 α / KV 复制等修改时静默漂移（**这些修改应当显式改 golden，并在 CHANGELOG 里说明**）。

---

## 6. 建议的落地顺序（供拍板，不阻塞）

| 版本 | 内容 | 工作量 |
|---|---|---|
| v0.29 | ② C2C α + 暴露同步（保留旧 max 口径作 legacy）、③ KV 复制 / Attention-DP、⑫ MoE 不均因子 | S×3 |
| v0.30 | ① Pareto + KV 容量 → B_max、④ SLO goodput、⑮ roofline 图 | S–M |
| v0.31 | ⑤ 投机 / MTP、⑯ 用例预设、⑬ 验证目录（SCALE-Sim 交叉 + 公开数据回放） | S–M |
| v0.32 | ⑥ PD 分离 xPyD + 异构 SKU、⑦ ERT 能耗表 | M×2 |
| v0.33+ | ⑧ chunked 稳态 / 时间线、⑨ attention 细化、⑩ vector unit、⑪ CP、⑭ 多级 KV；可选面积 / 成本、AFD | M–L |

---

## 附：参考链接汇总

- LLMCompass：https://github.com/PrincetonUniversity/LLMCompass · https://parallel.princeton.edu/papers/isca24_llmcompass.pdf
- GenZ：https://github.com/abhibambhaniya/GenZ-LLM-Analyzer · https://arxiv.org/abs/2406.01698
- AIConfigurator / AISimulate：https://github.com/ai-dynamo/aiconfigurator · https://github.com/ai-dynamo/aisimulate · https://arxiv.org/abs/2601.06288
- Vidur：https://github.com/microsoft/vidur · https://arxiv.org/abs/2405.05465
- LLMServingSim 2.0：https://github.com/casys-kaist/LLMServingSim · https://arxiv.org/abs/2602.23036
- LIMINAL：https://arxiv.org/abs/2507.14397
- TokenSim：https://github.com/pku-lemonade/TokenSim · https://arxiv.org/abs/2503.08415
- Frontier：https://github.com/NetX-lab/Frontier
- InferSim：https://github.com/alibaba/InferSim
- SimAI：https://github.com/aliyun/SimAI
- ASTRA-sim / Chakra：https://github.com/astra-sim/astra-sim · https://github.com/mlcommons/chakra
- Timeloop / Accelergy：https://github.com/NVlabs/timeloop · https://github.com/Accelergy-Project/accelergy
- SCALE-Sim：https://github.com/scalesim-project/SCALE-Sim
- ZigZag / Stream：https://github.com/KULeuven-MICAS/zigzag · https://github.com/KULeuven-MICAS/stream
- LLM-Viewer：https://github.com/hahnyuan/LLM-Viewer · https://arxiv.org/abs/2402.16363
- llm-analysis：https://github.com/cli99/llm-analysis
- APEX+：https://github.com/microsoft/apex_plus
- Splitwise-sim：https://github.com/mutinifni/splitwise-sim
- DistServe：https://github.com/LLMServe/DistServe · https://arxiv.org/abs/2401.09670
- Calculon：https://github.com/calculon-ai/calculon
- InferenceX：https://github.com/SemiAnalysisAI/InferenceX
- scaling-book：https://github.com/jax-ml/scaling-book
- PyTorchSim / ONNXim：https://github.com/PSAL-POSTECH/PyTorchSim · https://github.com/PSAL-POSTECH/ONNXim
- MAESTRO：https://github.com/maestro-project/maestro
- LLM-Para：https://github.com/dengls24/LLM-para
- Optimus / imec.kelis（闭源）：https://arxiv.org/abs/2407.14645
- AMPeD（训练）：https://github.com/CSA-infra/AMPeD

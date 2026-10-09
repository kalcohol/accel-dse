# Changelog

本项目的重要变更记录于此。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)（1.0 之前次版本号可能包含不兼容变更）。
0.31.0 及更早版本以 `npu-inference-dse`（包名 `npu_dse`）发布。

## [0.61.2] - 2026-10-09

第二轮对抗式审计：PD 排队 / DES 数学（小规模精确算例、Lindley 模拟、不变量），VAE 解码与结构模型端到端 FLOPs 对独立计数（torch FlopCounterMode 跑 diffusers / transformers 模块），能耗与预算，radix 前缀缓存，HBM 几何，非默认映射，MoE 倾斜与 Wan2.2 待机专家，视频 / 蛋白非默认布局，界面竞态（脚本化浏览器），HF 发布 config 现网复核。

### 修正（见建模说明 §19.11）
- **LTX-Video VAE 解码**：上采样块的 resnet 按块输出分辨率计（diffusers 先上采样再跑 resnet），原来少 2.5–3.3×。默认 15.4 → 50.5 TFLOP，与 diffusers 计数一致。
- **ESMFold 主干遍数** 5 → 4（参考实现 `num_recycles=None` 时循环 `max_recycles` 次）。512 残基 120.9 → 97.3 TFLOP，与 transformers 计数一致。
- **decode 占用相关时间 τ_int**：平均占用较大时 Poisson 方程求和抵消，τ 可到 1e80；改为均值以下从下往上累加。
- **radix 前缀缓存**：比缓存还长的节点及其子树不驻留、不占容量（Che 与 DES 两侧）。
- **面积预算**：缺密度时，已知项已超限就判为超限。
- **界面**：在途的搜索 / 扫描 / 对比 / Pareto 结果在场景改变后标为过期，并按发出请求时的模型渲染。

### 数值变化（默认指纹 1356 项中 116 项、多节点 985 项中 63 项、fabric 3150 项中 0 项；fits / 瓶颈无翻转）
- LTX-Video 整段延迟：单节点指纹 +3.9 … +8.5 %（单卡 +3.9 … +4.3 %；多卡时 VAE 不并行，占比更大），多节点 +7.6 … +14.6 %；DRAM 需求 +0.54 GiB（解码激活峰值 1.19 → 1.73 GiB）。
- ESMFold 批延迟 −19.9 … −20.0 %，每请求 FLOPs −19.2 %（384 残基 62.2 → 50.2 TFLOP）/ −19.5 %（512 残基 120.9 → 97.3）。
- PD 报告：TPOT p90 / p99 −0.0 … −0.7 %（平均占用大的 decode 池），其余不变。

## [0.61.1] - 2026-10-09

第一轮对抗式审计：手算复核稠密 / MoE / MLA / DSA / CSA / 线性注意力 / 视频 DiT / 结构模型 / PD 的 FLOPs、字节、KV、权重驻留与通信，跨模块（eval / sweep / layouts / CLI / API）一致性，默认关闭项等价，API 模糊测试（约 1.3 万个用例），以及 1440 / 1280 / 1024 / 820 px 宽度下的界面。

### 修正（影响默认数值，见建模说明 §19.10）
- **DeepSeek-V4 压缩层 decode 的 KV 读**被多除了一次压缩比（r = 4 层少 4×，r = 128 层少 128×），现在按完整条目计。
- **滑窗 / top-k / 压缩注意力的 prefill** 改按每个 query 实际 key 数的精确平均计（原按上限计，DSA prompt 4096 时多 33%，prompt ≤ 2048 时 2×）。
- **DSA / CSA 索引器的 prefill** 改为因果（原 top-k 与压缩层按整方阵计）。
- **纯滑窗 latent 层**只存窗口内的 KV；有前缀缓存的 prefill 中，纯滑窗层只读最后一个窗口。
- **线性注意力**每步读写 conv 状态（原只计存储）。
- **fp8 激活的发布**：all-reduce、MoE combine、PP 交接按 bf16 计（dispatch 仍按 fp8）「假设」。
- **serving 尺寸**上限 2²⁴，并拒绝布尔值。
- 界面：窄卡片的数值自动缩小字号，不再被截成「…」；SLC 策略选项文字缩短。

### 数值变化（默认指纹 1356 项中 197 项、多节点 985 项中 168 项、fabric 3150 项中 1432 项；fits / 瓶颈无翻转）
- prefill 步时（100T 默认芯片，batch 8，prompt 4096）：
  - DeepSeek-V3.2 −13.8 … −15.2 %（多节点 −15.2 … −19.5 %）
  - GLM-5.3 −10.4 … −11.5 %，GLM-5.2 −6.4 … −7.3 %
  - DeepSeek-V4.1-Flash −9.5 … −16.8 %，V4-Pro −5.6 … −5.7 %，V4-Flash −3.2 … −4.1 %
  - GLM-5.3-Flash −3.1 … −3.9 %，gpt-oss −0.03 … −0.04 %
- decode 步时：
  - DeepSeek-V4 系 +0.1 … +0.6 %（多节点 +0.2 … +2.8 %；KV 读 / 步在 1P、ctx 32k–128k 时 +19 … +98 %，计算受限点步时不变）
  - 线性注意力混合模型（Qwen3-Next、Qwen3.5/3.8、Kimi-K3、GLM-5.3-Flash）+0.03 … +0.3 %
- DRAM 需求：DeepSeek-V4.1-Flash / V4-Flash −0.02 … −0.42 %
- fabric 开、多节点、链路受限的 DeepSeek-V3 prefill：+2.3 … +26.9 %（20 项）；其余 1412 项只有链路字节变化。

## [0.61.0] - 2026-10-09

性能（纯 Python、零依赖），以及交换机层数时延、NVLS 类 allgather / reduce-scatter、NCCL 第二棵树、传输协议（「假设」，默认关）。默认结果（1356 项指纹）与 0.60.0 逐字节一致（与旧 fp471 基线的 26 项 MiniMax-H3 / LPDDR 差异在 0.60.0 中已存在），985 项多节点 / D2D / PD 指纹与 3150 项 fabric 开的指纹均 0 diff；大规模 PD 报告全 JSON 与 0.60 对照 0 diff（B ≤ 1024 的个别值有 ≤ 3e-16 的舍入差）。

### 性能
- PD 排队链：泊松链截断判据的 O(N²) `max(logp)` 改为滚动最大值；`_occ_tau` 泊松闭式、批到达内积用 `math.sumprod`；B ≥ 256 时链越过众数并低于峰值 1e-20 即停（丢弃质量 < 1e-17）；步长表只建一次。
- 评估：算子代价与 fabric 集合通信记忆化（1024 卡布局排名中命中 29× / 9×），空尾部算子跳过。
- 布局搜索：HTTP 服务下各候选的 b_max 区间在进程池中计算（排名、剪枝、评估数与串行相同）；稳定性的扰动用例分发到进程池；默认 worker 数 5 → min(8, CPU)。

| 操作（box 8 核；1P + HBM3e，8 卡 / 节点） | 0.60 | 0.61 单进程 | 0.61 服务（8 worker） |
|---|---|---|---|
| PD 排队报告 DeepSeek-V3 256 副本 B4096 | 21.2 s | 3.5 s | — |
| PD 排队报告 Kimi-K2 B8192 | 48.4 s | 3.5 s | — |
| PD 排队报告 8 副本 B256 | 16.3 s | 2.4 s | — |
| PD 排队报告 B2048 / B1024 | 18.1 / 6.4 s | 3.8 / 2.3 s | — |
| 布局排名 top-16，DeepSeek-V3 1024 卡 | 7.5 s | 6.2 s | 2.7 s |
| 映射对比 1024 卡（DeepSeek-V3 / Kimi-K2） | 27.6 / 28.4 s | 20.3 / 21.8 s | 7.2 / 5.3 s |
| 映射对比 DeepSeek-V3 4096 卡 | 43.1 s | 35.6 s | 8.3 s |
| 排名稳定性 1024 卡 | 30.9 s | 25.0 s | 5.5 s |
| PD + 排队 + 布局搜索 256 / 1024 卡 | 22.5 / 14.6 s | 16.5 / 8.6 s | — |

### 新增（「假设」，默认关）
- `fabric.hop_spine_us` / `hop_core_us`：网络步离开 leaf / pod 时的额外每步时延（ring 每步、tree 按出 leaf / pod 的层数、网内归约、all-to-all、PP、PD KV 的期望值）。
- `innet_reduce = net+link`：交换式 scale-up 上 allgather / reduce-scatter 的 NVLS 类候选 `innet`（字节不变，一次交换机往返代替 n − 1 步）；`fabric.collective` 支持 `reducescatter`。
- NCCL 第二棵树：跨 leaf 切边份额按两棵树各一半（m | n 且 n 偶数时不变）。
- `fabric.protocol` = off | auto | LL | LL128 | Simple：带宽效率 ½ / 120/128 / 1，每步时延按 NCCL tuner 协议比例；候选名带后缀（`ring/LL128`）。
- CLI：`--hop-spine-us --hop-core-us --fabric-protocol`；Web：跨 leaf / 跨 pod 每步附加、传输协议输入，算法列显示协议；扫描参数 `fabric.hop_spine_us`、`fabric.hop_core_us`。
- 建模说明 §19.6–19.9；测试 `tests/test_core_061.py`（14 项手算 / 等价性）。

### 数值例子（1P + HBM3e，三层：每 leaf 4、每 pod 32 节点，r₁ = r₂ = 2）
- DeepSeek-V3 PP4·DP256·EP256（1024 卡）decode b1024：5.075 ms；hop_spine/core 1/2 µs → +2.4%；protocol auto → +3.1%（all-to-all 选 LL128）；全开 +5.4%。prefill +0.02–0.04%。Kimi-K2 同布局 decode +2.4 / +3.2 / +5.6%。
- DeepSeek-V3 TP8·DP128·EP1024 decode：hop 1/2 µs → +5.4%；NVLS 类 allgather 把 logits allgather 7.2 → 4.2 µs（−0.05%）。

### 剩余缺口
- ECMP / 拥塞 / incast；协议的通道数与每通道带宽上限未建模；
- reduce-scatter 不在推理算子图中；KV 回馈只回灌 decode 池；
- PD + 布局搜索在 B ≤ 1024 时每个状态仍需一次精确评估（256 卡约 16 s）；
- 映射对比在服务中按映射并行，单个映射内部串行；未做支配候选剪枝（没有可证明的支配关系）。

## [0.60.0] - 2026-10-09

大规模卡数，以及三层 fat-tree、端口并发、按暴露时间选算法（「假设」，默认关）。默认结果（1356 项指纹）与 0.59.0 逐字节一致，985 项多节点 / D2D / PD 指纹也一致。另有 3150 项 fabric 开、默认参数的指纹，只在 PP > 1 且 scale-up 拓扑 ≠ switch 的 144 项上不同（新增的 PP 多跳，见下）。

### 新增
- **大规模卡数**：
  - 每副本卡数上限 64 → 8192（API / UI / CLI；`catalog.max_cards`），PD 池卡数上限 4096 → 65536。
  - 布局搜索 / 最佳 batch 的每副本 batch 上限 = max(4096, 64 × 卡数)（原来固定 4096，DP1024 时截断最优点）。
  - 容量修复建议（`/api/fit`）的候选卡数延伸到 8192。
  - PD 布局搜索：总卡数 > 64 时加入 128、256、… 的副本尺寸，各尺寸配额相同。
  - PD 排队：decode 生灭链在 B > 1024 时对 step(k) 做几何网格插值（误差 < 0.5%）；链长不再截断于 4096；SLO 切分在候选 > 32 时先粗后细（与穷举一致，1024 + 1024 卡 80 s → 15 s）。
  - Web：卡数输入上限 8192，大于 64 卡时默认布局为 TP ≤ 每节点卡数，其余 DP（MoE）/ PP（dense）。
- **三层 fat-tree**（`fabric.net_tiers = 3`，`pod_nodes`，`oversub_spine`）：
  - leaf / spine / core，网络层 β 除数 = max(1, r₁·f_leaf, r₁·r₂·f_pod)；r₂ = 1 时与两层逐位相同。
  - PP 交接跨 pod 为 r₁·r₂；PD KV 为 max(r₁(1 − m/n), r₁r₂(1 − P/n))。
  - 每 pod 节点数默认 = 每 leaf 节点数 × spine 下行端口数（radix·r₂/(1+r₂)）。
  - 报告每步跨节点流量，以及离开 leaf / pod 的份额（`fabric.net_traffic`，每行 `net_leaf_share` / `net_pod_share`）。
- **端口并发**（`fabric.overlap = ports`）：D2D / scale-up / NIC 是不同端口，可同时工作；同端口串行。级链路时间 = max(最忙端口的忙时之和, 最长的单次集合通信)，在 sum（0.59）与最忙端口之间。报告 `fabric.stages`（每级 sum / ports、各层忙时、窗口、α）。
- **按暴露时间选算法**（`fabric.algo = auto_overlap`）：每个流水级对各类集合通信的算法组合最小化 max(计算 / DRAM 窗口, 链路) + Σα（≤ 729 种穷举，否则坐标下降），不劣于 auto 或任何强制算法。
- **PP 多跳**：fabric 开时，节点内 ring / torus2d / full-mesh 上的 PP 交接按链路负载计（ring 8 封装、S = 2 → 4×；torus 4×2、S = 4 → 4×；full mesh n − 1）。
- **KV 回馈**（`fabric.kv_feedback`）：KV 流占 decode 池 NIC 的份额 u_kv 回灌 decode 池的网络集合通信（不动点），报告 `kv.fabric.feedback`（u_kv、回馈前后 TPOT）。
- CLI：`--net-tiers --pod-nodes --oversub-spine --fabric-overlap --kv-feedback`，`--fabric-algo auto_overlap`；输出流量分层、端口并发和 KV 回馈。Web：网络层数、每 pod 节点数、spine 收敛比、层间重叠、KV 回馈的输入，集合通信表新增「离开 leaf / pod」列。扫描参数 `fabric.oversub_spine`、`fabric.pod_nodes`。
- 建模说明 §19.1–19.5；测试 `tests/test_core_060.py`（13 项手算 / 性质）。

### 修正
- PD 排队：λ 极大时 1 − e^{−λ·lat} 舍入为 1 导致除零（大卡数的稳定速率二分会触发），现判为不稳定。

### 运行时间（box 单进程，测试套件同时在跑；1P + HBM3e，8 卡 / 节点，fabric 开）

| 操作 | DeepSeek-V3 256 卡 | DeepSeek-V3 1024 卡 | Kimi-K2 1024 卡 | DeepSeek-V3 4096 卡 |
|---|---|---|---|---|
| 单点评估（含 fabric 报告） | 0.02 s | 0.01 s | 0.01 s | 0.01 s |
| 评估 + 最佳 batch | 0.02 s | 0.03 s | 0.03 s | 0.05 s |
| 布局排名 top-16（271 / 451 / 451 / ~700 个布局） | 3.8 s | 8.6 s | 7.0 s | 13.6 s |
| 映射 × 布局对比 | 14.9 s | 29.2 s | 27.3 s | 46.8 s |
| 扫描（4 点） | < 0.01 s | < 0.01 s | < 0.01 s | < 0.01 s |
| 排名稳定性 | 15.0 s | 33.5 s | 31.5 s | — |
| PD + 排队 + 布局搜索 | 23.8 s | 21.0 s | 22.1 s | — |

### 数值例子（1P + HBM3e，8 卡 / 节点，网络 50 GB/s / 卡，leaf 由 radix 64 推导）
- **Kimi-K2 DP1024·EP1024（128 节点）**：
  - decode b8192：关 5.74 ms，两层 r = 1 / 2 / 4 都是 6.07 ms（link 被计算掩盖）；三层 pod 32、r₁/r₂ = 2/4 → 7.53 ms（LINK 绑定，−19% tok/s/卡）。
  - prefill b1024：关 560 ms，两层 r = 2 → 1085 ms，三层 2/2 → 1693 ms、2/4 → 3384 ms。76% 的跨节点字节离开 pod。
- **DeepSeek-V3 DP256·EP256（32 节点）**：pod 32 时与两层相同；pod 16、2/4 时 prefill b256 921 → 2181 ms，decode link 1.80 → 4.26 ms（仍被计算 5.85 ms 掩盖）。
- **端口并发 + auto_overlap**：
  - TP16 跨节点 prefill（pod 16，2/4）：616.5 → 584.1 ms（allreduce ring → hier）；
  - TP32 decode：tree → hier，89.64 → 89.17 ms；
  - TP8·DP32 decode：link 2.04 → 1.80 ms。
- **KV 回馈**：DeepSeek-V3 PD 960P + 64D，网络 12.5 GB/s，每 leaf 1 节点、r = 4：u_kv 4.5%，decode TPOT 15.80 → 16.50 ms。

### 剩余缺口
- 每步时延不随交换机层数增加；
- ECMP / 拥塞控制 / incast 未建模；
- 端口模型是稳态资源模型；
- KV 回馈只回灌 decode 池；
- 4096 卡以上的布局对比 / 稳定性为数十秒（未并行化）；
- PD 排队报告在大规模下仍需约 20 s。

## [0.59.0] - 2026-10-09

硬件侧：三层互连的跨节点网络拓扑与集合通信算法（`scenario.fabric`，「假设」，默认关）。默认结果（1356 项指纹）与 0.58.0 逐字节一致，另有 985 项多节点 / D2D / PD 指纹也一致。

### 新增
- **拓扑感知集合通信**（`core/fabric.py`，`fabric.enabled`）：每次 allreduce 在 ring（扁平多通道环）、tree（NCCL 式：节点内 chain，跨节点双二叉树）、hier（逐层 reduce-scatter → 顶层 allreduce → 逐层 all-gather）之间取最快（`fabric.algo = auto`，按 带宽时间 + α），也可指定。allgather 用 ring / hier，all-to-all 用直接交换。α = 每次集合通信的启动 α + 步数 × 每步时延，每步时延默认取 NCCL tuner 的 LL 协议默认值作数量级参考：节点内 0.6 µs、网络 ring 2.7 / tree 5.0 µs，D2D 0.1 µs「假设」。
- **跨节点拓扑**（`fabric.net_topology`）：fat-tree / leaf-spine（每 leaf m 个整节点，上行收敛比 `oversub`）或 rail-optimized（每 rail 交换机 m 个节点，跨 rail 流量走 PXN，计入节点内 scale-up 字节）。m 可指定（`leaf_nodes`），或由交换机端口数推出（`switch_radix` 64，下行端口 = radix·r/(1+r)）。离开 leaf 的流量份额 f 以 max(1, r·f) 变慢：
  - ring f = 1/m；
  - tree 按 NCCL `ncclGetBtree` 的跨 leaf 边计；
  - all-to-all f = (n − m)/(n − 1)；
  - PP 交接跨 leaf 为 r；
  - PD KV 为 r·(1 − m/n)。
- **节点内 scale-up 拓扑**：`link.topology` 原有字段（switch / ring）此前不参与计算，现在扩展为 switch / full_mesh / ring / torus2d（`fabric.torus_x`），只在 fabric 开时生效：
  - full_mesh：(n−1)/(k−1)；
  - ring：未绕满时为链 2×，× stride；
  - torus2d：按可用端口数；
  - all-to-all：按最短路径 / 维序路由的最大链路负载计（ring 8 卡 16/7，torus 4×4 为 32/15）。
- **网内归约**（`fabric.innet_reduce`，厂商选项「假设」）：net（SHARP / CollNet 类）、net+link（另含 NVLS 类，需交换式 scale-up），用于 allreduce 的 hier 顶层，每 rank 只收发一次。
- **争用**（「假设」）：并发组共享收敛上行（上述 f）。PD KV 传输与两池自身的集合通信共用 NIC，KV 可用带宽 × (1 − u_c)（`fabric.contention`），并报告集合通信被 KV 拖慢的倍数（不回灌）。PD 报告的 `kv.fabric` 给出 leaf 因子、u_coll、原始 / 有效 GB/s。
- 适用于 TP / EP / SP（DAP）/ DP / FSDP 集合通信、PP 交接、视频 VAE 并行与文本编码器分片的 all-gather，以及 PD KV 传输。
- **报告**：API `fabric`（每种集合通信的分层、所选算法、带宽 / 时延、其他候选，以及拓扑模型关时的步时间）；Web「拓扑感知集合通信」开关、各项输入和「集合通信」表；CLI `--fabric --oversub --leaf-nodes --net-topology --link-topology --fabric-algo --innet-reduce --switch-radix --torus-x`；扫描参数 `fabric.oversub`。
- `link.topology` 现在校验取值（非法值返回 400）。建模说明 §19；测试 `tests/test_core_059.py`（9 项）。

### 数值例子（fabric 开；1P + HBM3e，8 卡 / 节点，网络 50 GB/s / 卡）
- **DeepSeek-V3 DP64·EP64，8 节点，prefill b64**（关：707.7 ms，MAC 绑定）：
  - 每 leaf 2 节点：1:1 / 2:1 / 4:1 → 708.0 / 818.5 / 1636.1 ms（2:1 起 LINK 绑定）；
  - 每 leaf 1 节点：708.0 / 954.7 / 1908.6 ms；
  - 默认推导的 leaf（4 / 5 / 6 节点）：708.0 / 708.0 / 708.0 ms（link 476.9 → 545.1 ms）。
  - decode b512 一直是 7.73 ms（link 0.93 → 最多 3.73 ms，被计算掩盖）。
- **qwen3-32b TP16，2 节点，decode b64**：关 20.84 ms；ring / tree / hier = 22.92 / 22.43 / 21.85 ms，auto 选 hier。
- **单节点 TP2·PP4 prefill 的 link 时间**：switch 6.7 / ring 13.4 / torus 4×2 26.8 / full mesh 47.0 ms。
- **TP16 跨 2 节点 prefill，每 leaf 1 节点、4:1**：link 617 → 295 ms（开网内归约）。
- **PD KV**（qwen3-30b-a3b，4 节点 × 16 卡，每 leaf 1 节点）：2:1 / 4:1 → KV 传输 4.0 → 6.0 / 12.1 ms。

## [0.58.0] - 2026-10-09

修正 0.57 合并模式 KV 准入的 CV1 过保守，新增 radix / 部分前缀匹配（前缀树，「假设」，默认关）。默认（合并、非 PD）结果（1356 项指纹）与 0.57.0 逐字节一致；KV 策略和前缀树默认都关。

### 修正（只在 KV 策略开且容量受限时生效）
- **合并模式先准入后 prefill 的占槽时间 = prefill 服务时间**（`kv_cap.slot_hold_prefill_ms`）。0.57 取 T_pre = 未计等待的 TTFT 均值，里面包含 prefill 排队。DES 里准入发生在 prefill 开始的那个迭代边界，排队在准入之前，所以排队时间不占槽。现在 T_pre = TTFT 均值 − prefill 排队均值。CV1 下排队占 TTFT 的大头，0.57 的 Erlang C 放大因此过强。
- **合并模式的条件等待形状**（「假设」）：Gamma，c² = 1 + 0.7·min(2, c_s²)（`kv_cap.wait_c2`）。PD 仍为指数分布（c² = 1）。DES 实测合并条件等待 c²：prefill 优先 1.1 / 1.56（CV0 / CV1），分块 1.6 / 1.74；PD 0.6。0.57 试过固定 c² 1.8 更差，是因为当时的均值偏高，叠加后过保守。用正则化不完全 gamma `_gammq` 打表（512 点）。
- **分块模式的饱和抢占 × (1 − ρ_pre)**（`kv_cap.refill_tight_share`）：分块时 prefill 与 decode 同批，回填受 prefill 进度限制，只有 prefill 空闲的那部分时间才会被一次填满。分块抢占次数 +30 … +60 % → −10 … +4 %（4 seed × 8000；30000 请求长跑 CV1 −20 %）。

### 新增
- **radix / 部分前缀匹配**（`pd.prefix_tree`，「假设」，默认关）：前缀树按层给出 (tokens, 分支数, Zipf α)，例如 系统提示 512 tok × 4 → 文档 1536 tok × 200 → 对话 1024 tok × 100。每个请求在每层独立按 Zipf 选子节点。缓存是按 token 计容量的 LRU，命中时刷新从叶到根的整条路径，所以子节点驻留蕴含父节点驻留，各层共享同一份容量。闭式按节点用 Che 近似：h = 1 − e^{−pT}，T 由 Σ L_k · count · h = C_tokens 解出。匹配深度分布 P(深度 ≥ k) = Σ p·h，部分命中省掉已匹配的 token（cum_k，上限 S − 1）。prefill 的每个匹配深度单独作为一类请求进入排队混合；PD decode 池按深度持有率 q_dec 传输剩余 KV。与 `prefix_len` / `prefix_hit` 互斥；最多 6 层，叶路径 ≤ 1e9，超过 4096 组时按几何 p 分箱（比值 1.01）。
- DES：`pdsim.RadixLRU`（token 容量，刷新叶 → 根，按 LRU 淘汰），路径抽样只在开启前缀树时进行，所以其他场景的随机数流不变。`compare()` 输出 `prefix_level_hit`。
- 报告 `prefix_cache.tree`，每池 `token_hit` / `level_hit` / `depth_law` / `capacity_tokens`。CLI `--pd-prefix-tree 512:4:1,1536:200:1,1024:100:0`。Web 「前缀树」输入框和缓存表（各层命中 P(深度 ≥ k)、token 命中率、容量 tok）。
- 建模说明 §18.8；测试 `tests/test_core_058.py`。

### 误差
- **KV 策略**（dense8b 12 GB，before_prefill，L0.85，(闭式 − DES)/DES；4 seed × 8000 请求，0.57 → 0.58）：
  - `wait`：合并 CV1 TTFT p90 +23 / +40 % → +14 / +21 %（prefill 优先 / 分块），准入等待 W +4 / +9 %。CV0 p90 −22 / −6 % → −23 / −11 %。CV1 p99 −4 / +4 % → +12 / +16 %。
  - recompute / swap 合并 CV1 TTFT p90：−5 / +26 / +28 / +48 % → −13 / +2 / +15 / +24 %（prefill 优先 rec / 分块 rec / prefill 优先 swap / 分块 swap）。
  - 分块抢占次数 +30 … +60 % → −10 … +4 %。合并 CV1 TTFT p50 仍偏高 +60 … +92 %（中位数落在不等待的那部分，P(等待) 闭式低约 15 %，条件均值高约 15 %）。
- **长跑 DES**（4 seed × 30000 请求，CV1，L0.85）：8000 请求的 DES 从空系统起步，CV1 下低估稳态等待。以长跑为准，0.58 的合并 CV1 TTFT p90 误差为 wait +11 / +17 %，recompute −18 / −1 %，准入等待 +4 / +8 % 和 −23 / −9 %。p99 偏高 +5 … +47 %。分块抢占次数 −20 %。
- **PD recompute CV1**（长跑）：TTFT p90 −45 %，准入等待 −52 %，抢占次数 −30 %（8000 请求时 −21 / −26 / −18 %）。没找到有原则的修正。试过用碎片化后的有效 B 建模，B = 16 时 W 4380 vs DES 6777，不够，未采用。PD wait 长跑 p90 −16 %、p99 +2 %；swap p90 −9 %、p99 +11 %。
- **前缀树**（dense8b，树 512×4 → 1536×200 → 1024×100，3 seed × 3000，L0.6 / 0.85 × CV0 / 1 × 5 / 20 / 80 GB）：token 命中率闭式 vs DES 偏低 0.003 … 0.004（0.235 / 0.360 / 0.502），各层命中差 ≤ 0.016（5 GB 根层 0.945 vs 0.96），只看缓存时最大差 0.004。延迟误差：TTFT p90 −5 … +9 %，p99 −8 … +20 %（PD CV1），TPOT p90 −9 … +9 %；例外是 PD L0.85 80 GB：TPOT 均值 / p90 +12 / +18 %，最长间隔 +35 %（未查明）。

### 数值变化（§18.1 例子，load 0.8，KV 12 GB；KV 关时不变）
- `wait`：prefill 优先 TTFT 83 / 167 / 2091 → 83 / 167 / 2073 ms；分块 137 / 386 / 2377 → 135 / 362 / 2318。PD 不变（121 / 1138 / 3869）。
- `recompute`：prefill 优先 p99 1794 → 1776，分块 p99 2074 → 2013。`swap`：1785 → 1767，2063 → 2006。PD、TPOT、SLO goodput 不变（413 / 410 / 415 / 417）。

## [0.57.0] - 2026-10-09

KV 容量策略的准入等待与抢占模型重做，并修正 DES 里一个导致「连锁抢占」的记账错误；V4 网格高负载点用更长的 DES 重跑。默认（合并、非 PD）结果（1356 项指纹）与 0.56.0 逐字节一致；KV 策略默认仍关，所以 PD 报告只有开了 `pd.kv_policy` 且容量受限时数值才变。

### 修正
- **DES：恢复中的序列不占 KV / 槽位**（`pdsim._Replica`）。recompute / swap 的被抢占者开始重算 / 换入后，它的 KV 和槽位没有计入占用，所以 PD decode 副本在恢复期间仍会接收新的 KV 准入（PD 的准入在 prefill 完成时异步发生，不只在迭代边界）。恢复结束时批次溢出，刚恢复的请求（排在最后，最年轻）又被抢占：PD recompute CV1 L0.85 有 27 % 的抢占落在刚恢复的请求上，DES 准入等待 16.5 s、TTFT p90 49 s。vLLM 在调度重算时就分配 block，所以这是 DES 的错误，不是真实的连锁效应。现在恢复中的序列计入 `_kv_used()` 和槽位数（`restoring`），重复抢占降到 11 %，同一点的 DES 准入等待 4.4 s、TTFT p90 14.3 s、抢占 0.119 → 0.076 次/请求。合并模式只在迭代边界准入，基本不受影响（重复抢占 8 %）。0.56 报告里「recompute 连锁抢占，闭式 7.9 s vs DES 17.6 s」主要来自这个错误。
- CLI：PD 模式不稳定时，KV 行的表头误写「admission after_prefill」（取了第一个模式的 kv_cap，不稳定的模式没有 admit 字段）。

### 改进（只在 KV 策略开且容量受限时生效）
- **准入等待 = M/G/c 槽位队列**（「假设」）。均值：c_s² ≤ 1 时 W = c_s²·W_M/M/c + (1 − c_s²)·W_M/D/c，W_M/D/c 用 Cosmetatos 修正 ½·W_M/M/c / (1 + (1 − ρ)(c − 1)(√(4 + 5c) − 2)/(16ρc))；c_s² > 1 仍用 Lee–Longton。0.56 的 Lee–Longton 因子 (1 + c_s²)/2 不随槽位数 c 变化；c = 17、ρ ≈ 0.75 时 Cosmetatos 修正使定长输出的等待再低约 13 %（槽越多，近似等长的寿命越快排空队列）。条件等待仍取指数分布：DES 的条件等待 c² 在 PD 约 0.6、合并 1.6–2，试过按 c² 取 Gamma，两边都更差，未采用。
- **合并模式先准入后 prefill 时，prefill 期间也占槽**：槽位负载 a′ = a + λ_r·T_pre（T_pre = 该模式未计等待的 TTFT 均值），P(等待) 与 W 按 Erlang C 比例放大；a′ ≥ B 判不稳定（「KV 槽位不足：先准入后 prefill 时，prefill 期间也占着 KV 槽」）。`kv_cap.slot_hold_prefill_ms`。
- **饱和抢占率加「回填偏移」**：离开的请求释放的是 S + o_d，回填只用掉约 S，所以余量按 h → h + o_d − G 游走，而不是每次重新从 U(0, S) 抽。P(G − o_d > h) = 0.56 的项 × E_o[e^{−o/E[o]}]（G ~ Exp(E[o])，无记忆）：定长输出 e^{−1} ≈ 0.37，CV 1 约 0.47。`kv_cap.refill_offset_factor`。合并模式抢占次数偏高 +78 … +304 % → −23 … +60 %。
- **恢复占用的不动点**：恢复（重算 / 换入换出）占副本时间的比例 f，使每个槽的占用时间 × 1/(1 − f)，排队更多，饱和抢占更多，f 更大。在把服务率除以 1/(1 − f) 的槽位链上迭代 f ← ν(f)·T_stall（取最小不动点，发散 → 不稳定）。`_decode_birth_death` 返回 `pi_at(slow)`，可按减速因子重算稳态分布。`kv_cap.slot_slowdown`。
- Web 排队表注释、CLI 的 KV 行显示恢复占比 / 槽位占用倍数，以及合并模式的 prefill 期间占槽时间。

### 新增
- `v4_grid(n_high=, high_load=0.85)` / `scripts/v4_serving.py --n-high N --high-load L`：高负载点单独设 DES 请求数（SLO 二分仍用 n // 2），长任务先调度。
- 建模说明 §18.7；测试 `tests/test_core_057.py`。

### 误差
- **V4 网格**（30 点 × 3 seed；load 0.85 的点 20000 请求，其余 3000；(闭式 − DES)/DES）：PD TTFT p99 最差 +30 % → +7 %，落在容差内 97 → 100 %；PD TTFT p50 最差 −18 % → −13 %，落在容差内 93 → 100 %；PD / 分块最长间隔最差 +24 / +22 % → +11 / +11 %。load 0.85 点的 DES 噪声中位：PD TTFT p99 22 → 9 %，p90 14 → 4 %，TPOT 均值 7 → 3 %。长跑后显出来的真实偏差：合并 TPOT p90 最差 −11 → −15 %（prefill 优先）/ −12 %（分块），都在 tp4_32b L0.85 CV1；分块 TTFT p99 最差 −15 → −23 %（moe30b L0.85 CV0，该点噪声仍有 26 %）。SLO goodput 不变（二分的 DES 长度没变）。闭式没变（KV 关），所以这一节的变化全部来自 DES。
- **KV 策略**（dense8b 12 GB → B_kv 17，before_prefill，load 0.85，4 seed × 8000 请求，修正后的 DES；0.56 → 0.57）：
  - 抢占次数：+39 … +190 % → −23 … +60 %（PD swap −4 / −8 %，PD recompute −16 / −18 %，prefill 优先 −23 … +6 %，分块 +40 … +60 %）。
  - PD recompute CV1：TTFT p90 −40 % → −21 %，p99 −26 % → −5 %。合并 recompute CV1 TTFT p90 −32 / −26 % → −5 / +26 %。
  - `wait`：PD TTFT p90 +6 / +1 % → −6 / +1 %（CV0 / CV1），p99 +22 / +24 % → +8 / +23 %，准入等待 +12 / +1 % → −1 / 0 %；合并 CV0 p99 +12 % → −11 / −4 %，分块 CV0 p90 −21 % → −6 %。变差的：合并 CV1 TTFT p90 0 / −5 % → +23 / +40 %（prefill 期间占槽的修正在 CV1 过强）。
  - 0.56 写的「wait 折叠后 p90 +29 / +31 %、p99 最多 2×」主要来自 6 × 3000 的短跑 DES：同样的 0.56 闭式在 4 × 8000 上 PD p90 只差 +6 / +1 %。

### 数值变化（§18.1 例子，load 0.8，KV 12 GB；KV 关时不变）
- `wait`：PD TTFT 121 / 1410 / 4872 → 121 / 1138 / 3869 ms；prefill 优先 82 / 143 / 2845 → 83 / 167 / 2091；分块 133 / 317 / 2956 → 137 / 386 / 2377。TPOT、SLO goodput 不变。
- `recompute`：PD TTFT 114 / 1095 / 4374 → 114 / 906 / 3453 ms，TPOT 8.73 / 12.68 → 8.69 / 12.63 ms，最长间隔 461.9 → 12.8 ms（每请求抢占降到 1 % 以下，不再进入 p99）；prefill 优先 TTFT p99 2493 → 1794 ms。
- `swap`：PD TTFT → 114 / 871 / 3376 ms，最长间隔 389.2 → 12.8 ms。
- SLO goodput 不变（413 / 410 / 415 / 417）。

## [0.56.0] - 2026-10-09

API 输入校验收紧（400 而不是 500），KV 容量策略按 vLLM 的准入顺序计入 TTFT / SLO goodput，抢占进入 TPOT / 最长间隔尾部，新增 swap 抢占，长度档 8 → 15，PD 能耗可按池给静态功耗。默认（合并、非 PD）结果（1356 项指纹）与 0.55.0 逐字节一致；PD 报告里只有长度有 CV 时数值变化（见下），KV 策略默认仍关。

### 修正
- **`/api/eval` 等接口的 500**（AGENTS §25）：scenario 按 dataclass 类型注解严格校验，类型不符返回 400，并写明字段，例如 `Serving.batch: expected integer, got number 2.5`。整数字段接受整值浮点数（|x| ≤ 2⁵³，1e300 不再 OverflowError）。`scenario.chip` / `pd.prefill_chip` 可以直接写预置名（"1P"），未知名返回 400 并列出预置。`chip_preset`、sweep `path` 必须是字符串。模糊测试（每个字段换成错误类型 / 极值，7 个接口）全部 400。
- **PD 场景进布局搜索 / 稳定性 / fit 时的 500**：`search_layouts`、`ranking_stability`、`api_fit` 改布局时先回到合并部署（`Scenario.colocated()`），不再因 `pd.decode_cards` 与新布局不整除抛 ValueError。
- **长度档 8 → 15**（`lengths.BINS`）：8 个等概率档时 P(S ≥ s₅) 正好 = 0.5，TTFT p50 落在分布的跳变上，0.03 的概率差就让 p50 变几十 %。15 档时 0.5·N、0.9·N、0.99·N 都不是整数，三个分位数都不在档边界上。闭式和 DES 用同一套档，所以对拍仍然自洽。

- **M/G/1 等待分布改为精确的 Pollaczek–Khinchine 律**（`queueing._pk_surv`，用于 PD / prefill 优先的 TTFT 卷积、分块的准静态混合，以及报告里的 prefill 等待分位数）。等待 = 几何（ρ）个均衡剩余服务时间之和；离散服务律下剩余密度是阶梯函数，所以更新方程 g(x) = λ·Σ wᵢ[G(x) − G(x − τᵢ)] 可以在 800 点网格上推进：等待原子（1 − ρ）造成的台阶精确积分，其余部分用梯形格式；4·τ_max 以外接 Cramér–Lundberg 指数尾。均值与 P-K 公式差 < 1e-6（相对），M/D/1 与 Erlang 精确式一致，生存函数与 300 万样本的 Lindley 仿真差 ≤ 0.002。0.54 / 0.55 的 min(ρ, c·e^{−θx}) 只有尾部准确：服务时间跨 100×（CV 1）时，它给短等待（排在短请求后面）的概率太少，15 档后 PD TTFT p50 偏高 15–30 %。单一服务时间（定长）仍走精确 M/D/1，数值不变。
- `_mg1_busy` 改用平铺数组（结果相同，只是求和顺序不同），抵消 15 档带来的开销：CV 1 的 PD 报告 10.3 → 6.1 s（0.55：3.1 s）。

### 新增（默认关，或只在 KV 策略开时生效）
- **KV 准入顺序** `pd.kv_admit`：`before_prefill`（vLLM，默认）先占 KV 槽与显存再 prefill（PD：再拉取 KV），容量受限时准入等待计入 TTFT 与 SLO goodput。TTFT 分布由其 p50 / p90 / p99 重建，再卷积等待（原子 1 − p，Exp(W/p)）。`after_prefill` 保留 0.55 的顺序：等待单列，不计入。`kv_cap` 新增 `admit`、`in_ttft`、`p_wait`。DES 两种顺序都实现了：合并模式在 prefill 前准入，PD 在 KV 传输前准入，decode 侧保留到传输完成。
- **抢占进入尾部**（recompute / swap）：抢占率 = Rice 上穿 + 满副本准入时的饱和项 ν_sat = p·λ_r·E[(E[o]/S)(1 − e^{−S/E[o]})]「假设」。0.55 只有 Rice，比 DES 低约 10×，现在在 2× 以内。被抢占者的间隔 = 恢复时间 + Exp(1/λ_r)，按每请求抢占次数 p_v 混合进 TPOT 均值 / p90 / p99 与最长间隔。恢复占副本时间的比例 ≥ 1 时判不稳定。`kv_cap` 新增 `preempt_rice`、`preempt_sat`、`victim_gap_mean_ms`、`restore_share`。
- **swap 抢占** `pd.kv_policy = "swap"`，`pd.swap_GBps`（主机链路 GB/s / 卡）。不给时取 `workload.host_GBps` = 50 GB/s「假设」（PCIe 5.0 x16 有效带宽，0.46 起的已有假设）。换出 + 换入 = 2 ×（S̄ + ḡ）× KV 字节/token ÷（GB/s × 每副本卡数）。闭式和 DES 都实现了。`kv_cap` 新增 `swap_ms`、`swap_GBps_card`、`swap_source`。
- **PD 能耗按池给静态功耗**：`energy.idle_W_prefill`（PD prefill 池芯片每卡 W，不给 = `idle_W`，无默认值），用于异构 prefill 芯片。PD 模式的能耗新增 `tok_per_J`、`static_share`。只给 `idle_W_prefill` 时会写明另一池的静态能耗未计入。单芯片能耗报告不使用这个字段（也不把它列为缺项）。
- CLI：`--pd-kv-policy swap`、`--pd-kv-admit`、`--pd-swap-GBps`、`--idle-W-prefill`；PD 排队输出加能耗行（J/token、tok/J、静态占比）。Web：「KV 策略」加 swap，新增「KV 准入」「换出 GB/s / 卡」「静态 W / 卡 · PD prefill」；排队表的能耗列显示 tok/J。
- 建模说明 §18.6；测试 `tests/test_core_056.py`。

### 误差（V4 网格 30 点 × 3 seed × 3000 请求，(闭式 − DES)/DES；SLO goodput 符号相反，> 0 仍为保守）
- **TTFT p50**：prefill 优先最差 +47 % → −12 %，分块 +22 % → ±4 %，落在容差内的比例 87 % → 100 %（两种合并模式）。PD 最差 +18 % → −18 %（tp4 L0.85 CV1，该点 DES seed 噪声 28 %），中位 +1 % → 0 %。
- **TTFT p99**：prefill 优先最差 +19 % → +5 %，分块 +14 % → −15 %（dense8b L0.85 CV0.5 前缀）；PD 最差 +30 % 不变（moe30b L0.85 CV1，见下面「剩余缺口」）。
- **SLO goodput**：CV 1 的 12 个点 0.55 时闭式和 DES 都是 0（TTFT p90 落在最长档上，超过 SLO），现在为正值，所以统计点数 18 → 30。新加入的点上，零负载 TTFT p90 恰好略低于 SLO，速率对 TTFT 很敏感：PD 最差 +16 %（moe30b / tp4 CV1：闭式 52 / 19，DES 62 / 22 tok/s/卡，绝对值都很小），合并 −11 … +5 %。
- TPOT 均值 / p90、最长间隔与 0.55 基本相同（TPOT p90 中位 PD 0 %，合并 −4 / −5 %，最差 −11 %）。
- **KV 策略**（不在网格里，dense8b 12 GB → B_kv 17，6 seed × 3000）：load 0.6 时都接近。load 0.85：
  - `wait`：PD 折叠后 TTFT p90 偏保守（CV0 +29 %，CV1 +31 %，p99 最多 2×），准入等待 +34 … +44 %，TPOT ±1 %。
  - 抢占次数（recompute / swap）：0.55 低估约 10×，现在在 2× 以内（PD CV1 recompute 0.105 vs 0.100 次/请求）。
  - `recompute` CV1 L0.85：DES 出现连锁抢占（TTFT p90 17.6 s），闭式 7.9 s，明显偏乐观。

### 数值变化（§18.1 例子，load 0.8；0.55 → 0.56）
- 定长：不变。前缀（N = 20000，命中 / 未命中两类服务时间）：PD TTFT p90 129 → 128 ms；分块 TTFT 85 / 150 / 241 → 83 / 146 / 242 ms；其余不变。
- CV 0.5（15 档 + 精确 P-K）：
  - PD：TTFT 144 / 409 / 778 → 141 / 439 / 848 ms。
  - prefill 优先：TTFT 77 / 237 / 336 → 77 / 214 / 359 ms，TPOT 8.65 / 14.11 → 8.70 / 14.22 ms，最长间隔 666 → 722 ms，SLO goodput 399 → 398。
  - 分块：TTFT 145 / 358 / 624 → 138 / 370 / 674 ms，TPOT 8.63 / 14.14 → 8.68 / 14.24 ms，最长间隔 36.2 → 37.7 ms，SLO goodput 399 → 398。
- KV 12 GB（`kv_policy` 非默认）：
  - `wait`（默认 before_prefill）：PD TTFT 94 / 216 / 380 → 121 / 1410 / 4872 ms（准入等待 354 ms，P(等待) 24 %，现在计入 TTFT）；SLO goodput 413 不变（在 SLO 到达率处 TTFT 仍满足，TPOT 才是约束）。`after_prefill` 与 0.55 逐位相同。
  - `recompute`：PD TTFT → 114 / 1095 / 4374 ms，最长间隔 12.8 → 462 ms（被抢占者：重算 86 ms + 回队等待，每请求 0.025 次），TPOT 8.67 / 12.59 → 8.73 / 12.68 ms，SLO goodput 410 不变。
  - `swap`（新，50 GB/s/卡「假设」）：PD 最长间隔 389 ms（换出 + 换入 2 × 6.4 ms），TPOT 8.69 / 12.63 ms。

### 剩余缺口
- PD TTFT p99 在 CV1 L0.85 时 +27 … +30 %（tp4 / moe30b）。这不是闭式的偏差，是 DES 运行太短：n = 30000 时 tp4 DES 9161 ± 485 vs 闭式 9197（+0.4 %），moe30b 5939 ± 304 vs 6178（+4 %）；n = 3000 时，长尾在 τ_int 窗口里取样不足。网格仍按 3000 跑（30000 要多约 10× 时间）。
- KV：`wait` 折叠后 TTFT 偏保守；`recompute` 在高负载、CV1 下的连锁抢占没有建模；合并模式的 swap / recompute 抢占次数偏高（2–3×）。
- 档内展宽（连续长度）没做，用 15 档 + 精确 M/G/1 代替；CV 1 的 PD 报告 3.1 → 6.1 s。

## [0.55.0] - 2026-10-09

收敛 0.54 V4 找出的合并模式尾部误差，加 decode KV 容量策略（可选），把 V4 扩到 MoE 和多卡 TP。默认（合并、非 PD）结果（1356 项指纹）与 0.54.0 逐字节一致；PD 报告里的排队数值有变化，见下。

### 新增
- **KV 容量排队 / 抢占**（默认关）：`pd.kv_policy` = `wait`（为 prompt + 全部输出预留 KV，准入等待）/ `recompute`（只按当前上下文准入，超出时抢占最年轻的请求并重新 prefill），`pd.kv_capacity_GB`（不给则取剩余 DRAM）。只有 KV 槽位 < batch 时才生效。闭式（批链换槽位数、Little + Lee–Longton 准入等待、Rice 抢占率 + 重算停顿）与 DES 都实现了。每个模式输出 `kv_cap`；CLI `--pd-kv-policy --pd-kv-capacity-GB`；Web「KV 策略」「KV 容量 GB」。准入等待单独报告，不计入 TTFT / TPOT / SLO goodput。
- **V4 场景族**：`moe30b`（Qwen3-30B-A3B，TP2 · EP2）、`tp4_32b`（Qwen3-32B，每副本 TP4 跨 4 卡），各 6 点，与 dense8b（18 点）合计 30 点。`v4_scenario(..., family=)`、`V4_FAMILIES`。
- **多 seed 验证**：`pdsim.compare(seeds=…, slo_tol=…)` 对 DES 指标和 SLO 速率取平均，并报告 `noise`（seed 间标准差 / 均值）。网格默认 3 seed × 3000 请求，`scripts/v4_serving.py --seeds --jobs --families`（多进程）。摘要按场景族和全部给出，附 `noise_median`；`V4_BANDS` 加 TPOT 均值 ±10 %。
- `queueing.md1_cdf`（Erlang 精确 M/D/1 等待 CDF）、`queueing.mg1_mix_sum_quantile`（准静态混合 M/G/1 + 休假）；建模说明 §18.5；测试 `tests/test_core_055.py`。

### 修正（只影响 PD 报告，含其中的合并对照）
- **合并 prefill 优先**：prefill 忙期结束时一批请求同时进入 decode。birth–death 换成批到达链，并按配对系数 κ = 2∫P(O > t)²dt / E[O] 整形批分布，修正同批请求生命周期的相关（运行 batch 方差：链 6.99，DES 7.08；0.54 为 6.30）。
- **合并分块**：用更新—报酬过程算忙期内的有效成批程度（decode 在忙期内照常推进）；最后一个不满的块按实际比例计（0.54 按整块）。
- **TPOT p90 窗口因子**：常数 √(2/e) 改为按占用积分自相关时间 τ_int 的 OU 平均因子。
- **分块 TTFT 尾**：按运行 batch 环境做准静态混合 M/G/1，加纯 decode 迭代的休假。
- **最长间隔**（PD、分块）：从「看到的 batch 的 0.99 点」改为一生中最大 batch 的分位数（上穿率）。
- **M/D/1 等待分位数**：6 个服务时间以内用精确 CDF（原渐近式在 ρ ≈ 1 − q 附近偏保守 20–30 %）。
- 数值变化（§18.1 例子，load 0.8）：
  - PD：TTFT p50 102 → 94 ms，TPOT p90 14.06 → 14.70 ms，最长间隔 22.8 → 26.5 ms，SLO goodput 410 → 403 tok/s/卡（最佳切分仍为 2 + 6）。
  - prefill 优先：TPOT 7.45 / 11.14 → 8.22 / 13.35 ms，SLO goodput 442 → 411。
  - 分块：TTFT p99 330 → 377 ms，TPOT 7.43 / 11.09 → 8.20 / 13.38 ms，最长间隔 23.0 → 30.5 ms，SLO goodput 443 → 411。
  - CV 0.5 时 prefill 优先最长间隔 528 → 666 ms。

### 误差（30 点 × 3 seed，(闭式 − DES)/DES）
- TPOT p90 中位：PD 0 %，prefill 优先 −4 %，分块 −4 %（0.54：−5 / −18 / −14 %）；最差 −11 %（0.54：−43 %）。TPOT 均值全部在 ±6 % 内。
- SLO goodput：−5 % … +3 %。
- 最长间隔中位 −2 … +5 %，TTFT p90 全部在 ±10 % 内，前缀命中率绝对差 < 0.01。
- 没做：PD prefill 池内分块（FCFS 下没有 TTFT 收益，评估器也缺独立的块代价）、swap 抢占。剩余缺口（CV 下 TTFT p50 落在长度分档边界、CV 1 高负载 PD TTFT p99 +30 %、recompute 抢占次数低估、准入等待不进 SLO）见 MODEL.md §18.5。

## [0.54.0] - 2026-10-09

验证版本，不加新功能：新增请求级离散事件仿真（DES），用它量化 PD 闭式排队模型的误差（V4 服务验证），并修正 DES 揭示的几处闭式问题。默认（合并、非 PD）结果（1356 项指纹）与 0.53.0 逐字节一致。PD 报告里的排队数值有变化，见下。

### 新增
- `core/pdsim.py`：请求级 DES（「假设」）。复用闭式的逐步代价（prefill TTFT(b, S, p)、decode step(k)、分块融合迭代、KV 传输与争用缩放）；模拟泊松到达、FCFS、PD 两池（prefill 静态 batch 上限、每个 prefill 副本一条 KV 流、decode 连续批处理）、合并 prefill 优先与分块 prefill、离散长度分布、整前缀精确 LRU + Zipf（缓存预热，只在测量窗内计命中率）。提供 `simulate`、`slo_rate`（同一组随机数二分）、`compare`。
- V4 服务验证：`validation.v4_serving()`（实时子集，`accel-dse validate` 打印，约 15 s，`--no-v4` 跳过）、`validation.v4_grid()` 与 `scripts/v4_serving.py`（54 点网格：load 0.3 / 0.6 / 0.85 × CV 0 / 0.5 / 1 × 前缀 关 / 开 × PD / prefill 优先 / 分块）。结果存 `accel_dse/data/v4_serving.json`。原 V4（硅片，n/a）改名为 V5。
- 可选 `pd.simulate` / CLI `--pd-sim` / Web「DES 仿真尾部」（默认关）：PD 报告附 `pd.queue.sim.modes.{mode}`，包括 DES 的 TTFT / TPOT / 最长间隔分位数、命中率和相对闭式的误差；闭式数值不变。
- `queueing.mg1_sum_quantile`；建模说明 §18.4（误差表、偏乐观 / 偏保守的地方、数值变化）；测试 `tests/test_core_054.py`。

### 修正（只影响 PD 报告，含其中的合并对照）
- **decode 连续批处理**：Little 不动点 n̄ = λ·out·TPOT(⌈n̄⌉) 忽略了占用涨落，step(k) 随 k 变陡时偏乐观（§18.1 例子 load 0.8：平均 TPOT 6.49 ms，DES 9.05 ms）。改为 birth–death 处理器共享模型（9.21 ms）；TPOT 分位数与最长间隔按运行中的请求看到的 batch 分布计算。
- **prefill 优先 TPOT 分位数**原先重复计入停顿，改为平方和开根。
- **prefill 优先最长间隔**原先「TPOT + 一次 prefill」偏乐观（87 ms，DES 322 ms），改为一生中最长 prefill 忙期的 0.99 点（314 ms）；长度混合时按忙期时长卷积。
- **TTFT 分位数**：服务时间混合（长度 CV > 0、前缀命中 / 未命中）时，由等待分位数 + 时延分位数逐项相加（偏保守）改为 W ⊕ L_i 的正确分位数；只有一个服务时间取值时不变。
- 分块 prefill 的 decode 也走 birth–death，prefill 服务按看到的 batch 计。
- 数值变化（§18.1 例子，load 0.8）：
  - PD：TPOT 均值 / p90 6.49 / 9.00 → 9.21 / 14.06 ms，最长间隔 10.9 → 22.8 ms，SLO goodput 474 → 410 tok/s/卡（TTFT 不变）。
  - prefill 优先：TPOT 5.35 / 7.69 → 7.45 / 11.14 ms，最长间隔 83.5 → 314 ms，SLO goodput 506 → 442。
  - 分块：TTFT p90 177 → 209 ms，TPOT 5.35 / 7.39 → 7.43 / 11.09 ms，最长间隔 16.1 → 23.0 ms，SLO goodput 511 → 443。
  - CV 0.5 时 PD TTFT p90 539 → 409 ms，SLO goodput 316 → 404。

### 误差（54 点，(闭式 − DES)/DES）
- SLO goodput：三种模式均在 −8 % … 0 % 之间。
- TTFT p90：PD 全部在 ±20 % 内（中位 0）。
- TPOT p90：PD 中位 −5 %；合并模式系统性偏乐观（中位 −14 … −18 %，最差 −43 %）。
- 最长间隔：中位 −10 … −17 %。
- 前缀命中率：绝对差 < 0.01。
- 剩余缺口见 MODEL.md §18.4。

## [0.53.0] - 2026-10-09

前缀缓存容量 + LRU（Che）淘汰模型、异构 PD 池、布局搜索里的 decode batch。全部「假设」、默认关；默认结果（1356 项指纹）与 0.52.0 逐字节一致。

### Added
- `core/prefixcache.py`：Zipf 工作集 + 整前缀 LRU 的 Che 近似命中率（N ≤ 4096 精确，更大 N 对数分箱）；容量 = 剩余 DRAM（或 `pd.prefix_cache_GB`）/ 前缀足迹；随机路由 vs `pd.prefix_affinity`；PD prefill / decode / 合并各自独立命中率，decode 在 prefill 命中时也持有该前缀的条件概率进入 KV 交接。显式 `pd.prefix_hit` > 0 覆盖容量模型。
- 异构池：`pd.prefill_chip` / `pd.prefill_mem_id`（None = 与 decode 相同）；合并对照用 decode 池芯片 / 存储器。
- 布局搜索 `pd.search_decode_batch`：每种 decode 布局在 {B/2, B, 2B, 4B} 中取满足 TPOT SLO 的最高吞吐。
- CLI / Web 对应输入；响应 `pd.prefix_cache`、`pd.lengths.prefix_hit_source`、`pd.prefill.{chip,mem_id,hetero}`；建模说明 §18.3；测试 `tests/test_core_053.py`。

### Fixed
- `mdc_wait`：到达率贴着流体容量时 `c/service − λ` 舍入为 0 不再 ZeroDivisionError（标不稳定）。

### Changed
- `PDConfig` 新增 8 个字段（场景哈希随之变化）；默认值下 PD 报告与 0.52 相同。

### 不做 / 待定
- 部分前缀匹配、缓存预热 / 淘汰代价、队列 batch 对剩余 DRAM 的反馈、按芯片的闲置功率、PD prefill 池内分块、decode 端共享前缀 KV 读合并、长度感知调度；离散事件仿真对拍。

## [0.52.0] - 2026-10-09

PD 报告的三项补充：请求长度分布、前缀缓存命中率、池布局搜索。全部「假设」、默认关；默认结果（1356 项指纹）与 0.51.0 逐字节一致。

### Added
- 请求长度分布 `core/lengths.py`：`pd.prompt_cv` / `pd.out_cv`（对数正态，8 档等概率条件均值）或 `pd.length_mix`（离散联合分布 [权重, prompt, 输出]，≤ 16 行）。排队改为 M/G/1（`queueing.mg1`，离散服务分布的 P-K 均值 + Cramér–Lundberg 尾，单点即 M/D/1），每个 prompt 值单独求值；TTFT 分位用「自身 prefill + KV」的离散分位；decode 用 E[out]、长度偏置的上下文（`serving.ctx × ctx_ratio`）与 Allen–Cunneen (1 + c_s²)/2；流体容量（prefill / decode / KV 与合并对照）也按分布计算。`queueing.dquantile`、`mdc_wait(cs2=…)`。
- 前缀缓存：`serving.prefix_cached`（prefill 的 Phase(q = S − p, ctx = p)：只算新 token、读前缀 KV；吞吐仍按整个 prompt）与 `pd.prefix_hit`（每个 prompt 的命中比例），`pd.prefix_on_decode`（默认 true：KV 交接只传未缓存部分）。分块 prefill 块数与前缀重读随之变化。
- 池布局搜索 `pd.search_layouts`（opt-in）：两池每副本 1 / 2 / 4 / … / 64 卡的全部布局 × 同总卡数切分，流体 goodput 排序，部分布局对另算 SLO goodput（`layout_search`）。
- 每个模式的 `stable_rate_rps`；分块的 `chunk_share`；`pd.lengths` 摘要。
- CLI `--pd-prompt-cv --pd-out-cv --pd-mix --pd-prefix-hit --pd-prefix-not-on-decode --pd-search-layouts --prefix-cached`；Web PD 输入组「请求长度与前缀缓存」、单点页「池布局搜索」表；建模说明 §18.2；测试 `tests/test_core_052.py`。

### Fixed
- 合并 · 分块 prefill 的 decode 平均步时：0.51 用时间平均 ρ·T₁ + (1 − ρ)·T₀（偏向长迭代），改为每次迭代平均 T₀ / (1 − ρ + νT₀)，带块迭代占比 x = ν × 该值；块数用 ⌈S/C⌉。§18.1 例子里分块 TPOT 均值 9.4 → 5.4 ms、SLO goodput 433 → 511 tok/s/卡。PD 与 prefill 优先不变。

### Changed
- `Serving` 新增 `prefix_cached`，`PDConfig` 新增 6 个字段（场景哈希随之变化）；默认值下 PD 报告与 0.51 相同（分块修正除外）。

### 不做 / 待定
- 长度感知调度、缓存容量 / 淘汰 / 路由、decode 端共享前缀 KV 读合并、两池 batch 搜索、异构池、PD prefill 池内分块；离散事件仿真对拍。（0.53 已补容量 / LRU、异构池、布局搜索 decode batch）

## [0.51.0] - 2026-10-09

PD 报告从稳态流体模型往下深入一层：排队与尾延迟、连续批处理、分块 prefill、KV 与池内集合通信争用、PD 能耗。全部是解析近似「假设」，只在 `pd.enabled` 时计算，默认结果与 0.50.0 逐字节一致。

### Added
- `core/queueing.py`：M/D/1（P-K 均值、P(W>0) = ρ、Cramér–Lundberg 尾分位）、Erlang C（Allen–Cunneen ½）槽位等待、Poisson 分位；测试用 Lindley 仿真核对 p90 / p99。
- `core/pdqueue.py` → eval 响应 `pd.queue`：同一泊松到达率（`pd.load` × PD 流体容量，默认 0.8，或 `pd.rate_rps`）下比较 PD 分离、合并 · prefill 优先、合并 · 分块 prefill（`pd.chunk_tokens`，默认 512）的 TTFT p50 / p90 / p99、TPOT 均值 / p90 / p99（请求平均）、最长 token 间隔、端到端均值、SLO goodput（p90 TTFT 与 p90 TPOT 都满足的最大到达率 × out / 卡）。PD 的 prefill 池为带 batch 上限的 M/D/1（取平均 TTFT 最小的上限），decode 池连续批处理按 Little 不动点求运行 batch；prefill 优先模式计 decode 停顿；分块模式按级融合迭代（prefill 权重读与 decode 共用、前缀 KV 重读计入）。
- KV 与池内集合通信争用（没设 `pd.kv_GBps` 时）：KV 可用带宽扣除两池在该层的集合通信占用，KV 流量反过来拉长池的链路时间；`pd.kv_GBps` 视为专用通道。
- PD 的 SLO goodput 切分搜索（`pd_slo_splits`、`pd_slo_best_split`）。
- PD / 合并两种模式每输出 token 的动作计数与卡·秒（`pd.queue.energy`），给了能耗表时出 J / token。
- CLI `--pd-load --pd-rate --pd-chunk`、`--out-len`、`--ttft-slo`，`eval --pd` 输出排队表；Web PD 输入组加「负载 / 到达率 / 分块 token」，单点页加「排队与尾延迟」表；建模说明 §18.1；测试 `tests/test_core_051.py`。

### Changed
- 默认结果不变：1356 项指纹与 0.50.0 逐字节一致。`PDConfig` 新增 `load`、`rate_rps`、`chunk_tokens`（场景哈希随之变化）。`disagg_report` 新增可选 `energy`、`queue` 参数。

### 不做 / 待定
- 请求长度分布、抢占 / 换出 / KV 容量排队、前缀缓存、调度器开销、PD prefill 池内分块、异构池、池布局搜索；分位数逐项相加（偏保守），不做卷积或仿真。

## [0.50.0] - 2026-10-09

prefill / decode 分离（PD）作为可选服务模式；互连拆成三层（可选封装内 D2D、节点内 scale-up、跨节点网络）。全部默认关，默认结果与 0.49.0 逐字节一致。

### Added
- PD 分离 `core/disagg.py`（`scenario.pd`，默认关 = 合并服务）：decode 池用场景布局，prefill 池用 `pd.prefill_layout`，两池卡数 `pd.prefill_cards` / `pd.decode_cards`；prefill 池取满足 TTFT SLO 的最大 batch。请求率 = min(prefill 池、decode 池、KV 传输)，给出瓶颈与各池利用率、TTFT（prefill + 暴露的 KV 传输）、TPOT（纯 decode 步）、goodput / 卡，同总卡数的全部切分与最优切分，以及同卡数合并服务的对照（含有效 TPOT = TPOT / decode 占比）。KV 每请求 = 整模型 S token 的 KV + 索引键 + 状态；走跨节点网络（设了 `node_cards`）或节点内链路，`pd.kv_GBps` 可覆盖，`pd.kv_layerwise` 逐层流式只暴露掩盖不了的部分。Qwen3-8B 1P + HBM3E batch 32 prompt 8192：PD 2 + 6 卡 TPOT 42.2 ms（满足 50 ms SLO）569 tok/s/卡；合并 574 tok/s/卡但有效 TPOT 55.8 ms。只对 LLM / VLM。API `pd`，CLI `--pd --pd-prefill-* --pd-prefill-cards --pd-decode-cards --pd-kv-GBps --pd-layerwise`，Web 服务组「PD 分离」与单点页对照表。
- 三层互连：`d2d_enabled`（默认关 = 单片大 die）+ `package_cards` + D2D 档位 `d2d_std` / `d2d_units`；`link` 改称节点内 scale-up；新增跨节点 `net`（默认 50 GB/s、5 µs「假设」）与 `node_cards`（默认 0 = 单节点）。集合通信按组在封装 / 节点内的成员数逐级分层（all-reduce / all-gather 逐级、all-to-all 各层同时），PP 交接按边界选层；两层时与 0.48 公式逐位相同。能耗新增 `net`（`pJ_bit_net`），D2D + 节点内 + 跨节点 = 单层链路字节。每级 `link_GB` 增加 `scaleup`，`net` 改为跨节点字节。
- D2D 档位目录 `core/d2d_catalog.py`（公开数据，原始速率每方向每单元）：UCIe-A x64 @ 48 GT/s（384 GB/s，常用，D2D 开时默认 × 4 模块 = 1536 GB/s）、UCIe-A @ 64 GT/s（512）、UCIe-S x16 @ 32 GT/s（64）、BoW-256 / BoW-512（32 / 64 每 16 线 slice）、NVIDIA NVLink-C2C（450，厂商专有，参照）与自定义。48 / 64 GT/s 每 transfer 1 bit 与每 die 单元数标「假设」。API `/api/catalog` `d2d_standards`，CLI `accel-dse d2d`、`--d2d --d2d-std --d2d-units --node-cards --net-GBps --net-alpha-us --pJ-bit-net`，Web「三层互连」组（D2D 开关、档位选择、单元数、跨节点三项）；扫描新增 `d2d_units`、`node_cards`、`net.GBps`、`net.alpha_us`。
- 建模说明 §15 重写为三层互连并加 D2D 档位表（§15.1），新增 §18（PD）；测试 `tests/test_core_050.py`。

### Changed
- 默认结果不变：1356 项指纹与 0.49.0 逐字节一致。场景新增字段（`d2d_enabled`、`d2d_std`、`d2d_units`、`net`、`node_cards`、`pd`），场景哈希随之变化。
- D2D 默认带宽 2000 → 1536 GB/s（UCIe-A 48G × 4，只在 D2D 开时使用）。D2D 关时 `package_cards` 不生效并警告；0.48 / 0.49 的场景 JSON 中 `package_cards > 1` 自动视为 D2D 开，给了 `d2d.GBps` 视为自定义档位，照旧求值。
- Web「网络层」改称「节点内」；能耗表的链路项按三层拆分显示。

### 不做 / 待定
- PD：排队与到达波动、连续批处理动态与 chunked prefill、KV 与池内集合通信争用、前缀缓存、异构池、PD 能耗、池布局搜索。
- 互连：拓扑、拥塞 / 超额订阅、SHARP、NIC 的 PCIe 瓶颈、D2D flit / 协议效率（用自定义档位填有效带宽）。

## [0.49.0] - 2026-10-09

两个可选项：MoE 专家负载倾斜（工作负载旋钮）与资源 / 面积预算（设计约束，只报告余量）。默认关，默认结果与 0.48.0 逐字节一致。

### Added
- MoE 负载倾斜：`serving.moe_skew`（默认 1 = 均匀路由，[1, 64]，「假设」）或实测的每专家 token 分布 `serving.moe_expert_load`（按连续放置求每个 EP 宽度的倾斜，搜索中不同 EP 看到不同倾斜）。最忙 EP rank 决定级时间：其专家 GEMM 行数与 dispatch / combine / ETP 载荷按 skew × 平均计（上限 T·k 与每专家 T 行），命中专家数保持均匀期望。EP = 1 / 非 MoE 时忽略并警告。能耗计数取均匀路由下的每单位值（倾斜只搬工作），卡·秒按倾斜后的时间窗。100T + HBM3E，Qwen3-30B-A3B TP2·DP4·EP8：prefill 64 × 4096 TTFT 4608 → 倾斜 2 / 4：6048 / 8924 ms；decode batch 64 不变（行分块吸收），batch 2048：258.6 → 293.0 ms（倾斜 4）。summary 新增 `moe_skew`；CLI `--moe-skew --moe-expert-load`；Web「MoE 倾斜」；扫描 `serving.moe_skew`。
- 资源 / 面积预算 `core/budget.py`：请求体顶层 `budget`（不进 scenario）。限额：每卡 SRAM / SLC / MAC 单元 / 峰值 TFLOPS / DRAM 容量、每副本卡数、每卡平均功耗（用用户能耗表）、面积代理（每卡 / 每副本；SRAM、SLC、MAC 与固定面积的密度全部由用户填写，工具不内置工艺库）。报告余量与「未超 / 超出 / 无法判定」（缺密度，或功耗 / 面积只是下界且未超）。布局、映射对比、扫描每行加 `budget_ok`；超预算的布局排在后面，映射对比取预算内最优，容量检查的最少卡数受卡数预算限制。CLI `--budget-* --mm2-*`（search / compare 多一列）；Web「资源 / 面积预算」组与单点页预算表、布局表「超预算」标记。
- 建模说明 §16（MoE 倾斜）、§17（预算）；测试 `tests/test_core_049.py`。

### Changed
- 默认结果不变：1356 项指纹与 0.48.0（= 0.47.1）逐字节一致。Serving 新增字段，场景哈希随之变化。

### 不做 / 待定
- EPLB / 冗余专家、专家放置优化；工艺库、成本、面积与频率 / 功耗耦合。
- 三级互连与 SLC / D2D 默认数值仍待用户答复（0.48 的两个问题），本版未改动。

## [0.48.0] - 2026-10-09

硬件侧：系统级缓存（SLC）作为 SRAM 与 DRAM 之间的设计变量；两级互连（封装内 die-to-die 与跨封装网络）。两者默认关，默认结果与 0.47.1 逐字节一致。

### Added
- SLC（`chip.slc_mib`，默认 0 = 无；`slc_GBps` 默认 2000、`slc_policy` = `pin` | `lru`，全部「假设」）：命中字节 / `slc_GBps` 作为与 DRAM 并行的 `t_slc` 进入级时间，瓶颈可标 SLC；DRAM 只计未命中；不增加容量。`pin` 按 SRAM 同序钉住剩余热权重 → 专家 → KV / 状态，流式激活、FSDP gather、查表行、KV 写绕过；`lru` 按循环访问取全有或全无（SRAM 之外的工作集放得下才全部读命中）。视频 pipeline 组件不用 SLC。100T + LPDDR5X，Qwen3-8B decode batch 8：TPOT 104.2 → pin 4 / 8 / 16 GiB 81.7 / 59.2 / 20.8 ms；lru 16 GiB 放不下 → 不变，32 GiB → 20.0 ms。
- 两级互连：`link` 为跨封装 / 节点的网络层（默认不变 400 GB/s、3 µs），新增 `d2d`（同封装 die-to-die，默认 2000 GB/s、0.5 µs「假设」）与 `package_cards`（每封装卡数，默认 1 = 无 D2D 层）。卡按 TP → SP → DP → PP 编号，通信算子带步长 `comm_stride`（TP / ETP 1，SP / DAP 为 TP，EP 为 ETP，FSDP 为 TP），组在封装内的成员数 k 决定分层：all-reduce / all-gather 两级（NCCL 式）、all-to-all 按比例两路并行、PP 交接按流水级边界是否跨封装；k = 1 时与单层公式相同。VAE tile all-gather 与文本编码器分片 gather 同样分层。1P + HBM3E、网络 50 GB/s：Qwen3-8B prefill TP8 TTFT 676.8 ms（LINK）→ 每封装 4 卡 248.2 ms（MAC）。分层 all-reduce 的 α = 2α_d + α_n，小载荷 decode 部分跨封装时略慢（如实计）。
- 能耗：动作 `slc`（`pJ_bit_slc`）与 `d2d`（`pJ_bit_d2d`）；DRAM 只计 SLC 未命中，`link` 只计网络层份额（按每次集合通信两层发送字节之比拆分）。守恒：SLC + DRAM = 无 SLC 的 DRAM 字节，D2D + 网络 = 单层链路字节。
- 入口：API scenario `package_cards`、`d2d`、`chip.slc_*`，每级 `t_ms.slc`、`slc_GB`、`link_GB`（total / d2d / net）、`mem.slc_*`；扫描路径 `chip.slc_mib`、`chip.slc_GBps`、`package_cards`、`d2d.GBps`、`d2d.alpha_us`。CLI `--slc-mib --slc-GBps --slc-policy --package-cards --d2d-GBps --d2d-alpha-us --link-GBps --link-alpha-us --pJ-bit-slc --pJ-bit-d2d`。Web：芯片组 SLC 三项、并行组「每封装卡数 / D2D」、能耗组两项，级表 / 存储表 / 能耗表在启用时多出 SLC / D2D 列。
- 建模说明 §14（SLC）、§15（两级互连）；测试 `tests/test_core_048.py`（默认不变、闭式、映射到层、pin / lru、能耗拆分、校验 / API / CLI）。

### Changed
- 默认结果不变：1356 项指纹与 0.47.1 逐字节一致。场景新增字段（`d2d`、`package_cards`、芯片 `slc_*`），场景哈希随之变化。
- Web「链路 GB/s / 同步 α」改称「网络层 GB/s / 网络 α」。

### 不做 / 待定
- 第三层互连（封装内 D2D、节点内 scale-up、跨节点 scale-out）未分开——网络层一项代表封装以外的全部；拓扑、拥塞、SHARP 不建模。
- SLC 多卡共享、一致性、写回差异、bank 冲突不建模；默认 D2D / SLC 数值为占位「假设」。

## [0.47.1] - 2026-10-09

硬件侧第一小步：能耗动作计数（每动作能耗由用户提供，工具不内置数值）。

### Added
- `core/energy.py`：每输出单位（token / prompt token / 帧 / 序列）的动作计数——bf16 等效 MAC、向量操作、SRAM 端口字节、DRAM 字节、链路字节、卡·秒——全系统累计；视频含卡上的文本编码器与 VAE 解码（主机 CPU 编码器不计），VAE 多卡解码含 tile all-gather。能耗 = 计数 × 用户能耗表（`pJ_mac`、`pJ_vec`、`pJ_bit_sram`、`pJ_bit_dram`、`pJ_bit_link`、`idle_W`），空项不计并列出。API body 顶层 `energy`（不进 scenario、不改哈希），响应 `energy`；CLI `--pJ-mac … --idle-W`；Web「能耗」输入组与「动作计数与能耗」表。测试 `tests/test_core_energy.py`（计数守恒与闭式、线性、校验、API / CLI）。
- 计数器：`StageResult.sram_bytes`（SRAM 端口字节 = 供数周期 × 端口宽度），组件的 `acts`。

### Changed
- 无结果变化：1356 项指纹与 0.47.0 逐字节一致。

### 仍未建模
- 工艺推导的功耗、DVFS、SRAM 容量相关的访问能耗、DRAM 行激活 / 刷新、主机 / PUE、面积 / 成本。SLC、D2D 与网络两级互连尚未开始。

## [0.47.0] - 2026-10-09

视频 pipeline 收尾：VAE 多卡分块并行解码、LTX 分块、MiniMax-H3 解码按发布（总是分块）、跨请求重叠（主机编码器）。

### Added
- VAE 多卡分块并行解码 `workload.vae_parallel`（默认关；Web「VAE 多卡分块并行解码」，CLI `--vae-parallel`）：依据 MiniMax-H3 发布的 `vae_parallel_tiling`——每个分块调用内 rank r 解第 r、r + N、… 个 tile，再 all-gather 解码后的像素 tile；N = 副本的 PP·TP·SP 卡，算术不变，时间 = 最慢卡 + 每轮 all-gather，VAE 权重每卡一份。需分块解码（H3 / HunyuanVideo 总是分块，其余开 `vae_tiling`），否则忽略并警告。100T + HBM3E：H3 SP 2 / 4 / 8 解码 24.4 → 12.2 / 6.11 / 3.49 s；SP8 HunyuanVideo 221 → 34.0 s、Wan 分块 67.8 → 10.4 s。H3 以外为设计选项（xDiT DistVAE 只用于图像 VAE）。
- LTX-Video VAE 可选分块（diffusers `AutoencoderKLLTXVideo`：512 px tile / stride 448 → 潜空间 16 / 14）：默认 704 × 512 → 4 tile，×1.23，解码 0.72 → 0.89 s。
- 跨请求重叠 `workload.overlap`（默认关；CLI `--overlap`）：只在 `te_cpu` 时生效，主机编码下一请求与卡上去噪并行，稳态周期 = max(T_clip − T_text, T_text)，延迟不变、吞吐按周期计（结果新增 `period_s`）；组件同卡时按串行计并警告。
- 测试 `tests/test_core_047.py`。

### Changed
- MiniMax-H3 视频解码按发布更正：发布配置 `vae_decoder_tiling = 1`（tile 256 px、最小重叠 64 px；diffusers 文档写明默认分块、关掉会改变输出），时间上 5n + 2 潜帧 → n 段 × (5 + 2) 潜帧（原按 ⌈(T + 3)/5⌉ 段 × 5 潜帧、不分块）。默认 1344 × 768 124 帧：7 段 × 28 tile，解码 1741 → 1893 TFLOP、22.6 → 24.4 s，整段 1936 → 1937 s。1356 项指纹只有 minimax-h3 的 20 项变化（其余逐字节一致，LLM 不变）。
- 分块循环与 diffusers 一致：任一轴超过 tile 时两轴都按 stride 从 0 走（短轴多一条窄 tile）；默认分辨率下与 0.46 相同。
- `vae_tiling` 对没有参考分块的 VAE（Open-Sora）的警告改为「参考实现没有空间分块」。

### 不做 / 仍未建模
- Open-Sora VAE 分块：发布实现没有空间分块（只有逐帧 / 17 帧微批，已计），diffusers 未收录——不做。
- 结构模型 pair 的 TP：参考实现里没有，DAP + 样本分卡已覆盖多卡——维持不做。
- 不分块 VAE 的 patch 并行（halo 交换）；组件同卡时的跨请求重叠；AlphaFold 3 仍未接入。

## [0.46.0] - 2026-10-09

扩散样本分卡、DiT 权重 FSDP、文本编码器放主机 CPU、Wan VAE 分块。

### Added
- 结构模型扩散样本分卡 `workload.sample_split`（默认开；Web「DAP 时扩散样本分卡」，CLI `--no-sample-split` 关闭）：DAP > 1 且样本数 > 1 时，扩散模块（原子编码器 / 扩散 transformer / 原子解码器）的样本分到 DAP 各卡，每卡 ⌈S / D⌉ 条轨迹——样本独立、条件已在每卡，精确且无额外通信；块内 pair 网格工作仍按 DAP 切分。100T + HBM3E：Protenix（5 样本）DAP 8 10.65 → 3.22 s（相对单卡 1.8× → 6.0×）；Boltz-1 默认 1 个样本不变，5 个样本时 DAP 8 12.6 → 4.3 s。置信度头不分样本。参考实现无现成的多卡样本并行：作为布局设计选项给出。
- DiT 权重 FSDP `workload.dit_fsdp`（默认关；Web 复选框，CLI `--dit-fsdp`）：按 Wan `--dit_fsdp`，每级权重在 SP·DP 卡之间分片（w / g + 2 层预取），每次前向逐层 all-gather 活动权重（与计算重叠，每层 α）并把 gather 结果写入 DRAM；Wan2.2 空闲专家分片存储、不 gather；SP·DP = 1 或非视频模型时忽略并警告。例：Wan2.1-14B SP2 64 GiB LPDDR 每卡需求 57.8 GiB（auto 卸载）→ 45.1 GiB（常驻），延迟变化 < 0.1%。
- 文本编码器放主机 CPU `workload.te_cpu`（默认关；CLI `--te-cpu`）：按 Wan `--t5_cpu`，卡上不放编码器；编码时间 = 编码器 FLOPs / `workload.host_TFLOPS`（默认 2 TFLOPS「假设」，CLI `--host-TFLOPS`，请按实测填写）。例：Wan2.1-14B 单卡 64 GiB 常驻放得下（61.9 GiB，不再需要每请求重载），编码 4.8 s。
- Wan VAE 可选分块（`vae_tiling`，diffusers `AutoencoderKLWan` 的 256 px tile / stride 192；官方仓库不分块）：720P 28 tile，重叠 ×1.65，解码 41.3 → 67.8 s，激活峰值 3.81 → 0.27 GiB。
- Web：「放不下」面板增加「DiT 权重 FSDP」与「文本编码器放主机 CPU」修正；假设列表写明 FSDP / 主机 CPU / 样本分卡。测试 `tests/test_core_046.py`。

### Changed
- `Op.replicated` 可为分数（D > S 时空闲的样本槽）。默认场景（单卡、PP、DP、无 DAP）的 1356 项结果与 0.45 逐字节一致；变化只在 DAP > 1 且样本数 > 1 的结构模型，以及新的可选项。

### 仍未建模
- 置信度头的样本分卡；单个样本的扩散 transformer 的 token 维切分（Boltz-1 默认 1 个样本时 DAP 帮不上扩散部分）；pair 的 TP。
- 主机 CPU 编码器只有一个算力旋钮；VAE 多卡并行解码；跨请求的组件 / 去噪重叠；LTX / Open-Sora / H3 的可选 VAE 分块。
- AlphaFold 3 仍为「暂未接入 v2」（无公开权重）。

## [0.45.0] - 2026-10-09

结构模型多卡（DAP）、视频组件放置（分片 / 卸载）、可选 VAE 分块解码。

### Added
- 结构模型 DAP（FastFold 动态轴并行，`layout.sp` 即 DAP 度，Web 显示为「DAP」）：pair / 模板网格沿残基轴、MSA 网格沿行或列切到 D 张卡，权重复制；单一 / token / 原子轨道每卡重复（不重复计入请求 FLOPs）。逐核通信：三角乘法 all-gather 投影操作数，三角注意力 all-gather 偏置 + pair all-to-all 转置，MSA 行注意力 all-gather 偏置，列注意力两次 MSA all-to-all，外积均值 all-gather 右投影（AF3 类 MSA 模块先加一次 all-to-all），pair 加权平均 all-gather 权重，单一轨道注意力 all-gather 偏置。转置载荷用检查点里各模块输入宽度（`PairCore.width`：c_z / c_m / c_t）。pair 在途激活与跨 stage 传输按 ⌈N / D⌉ × N × c_z。100T + HBM3E、DAP 8：AF2 14.2 → 1.81 s（7.8×）、OpenFold 7.8×、ESMFold 7.1×、Boltz-1 3.6×、Protenix 1.8×（扩散 transformer 在重复的单一轨道上）。布局枚举 / 搜索 / 「放不下」修正都包含 DAP。
- 视频组件放置 `workload.placement`（Web「组件放置」、CLI `--placement`）：`resident`（0.44）、`shard`（文本编码器权重 FSDP 切到本副本全部卡，每卡 te_w / c + 2 层，逐层 all-gather 与编码计算重叠——Wan `--t5_fsdp`）、`offload`（组件与 DiT 分时占用显存，需求取最大值，每请求按 `workload.host_GBps`（默认 50 GB/s「假设」）重载权重——diffusers `enable_model_cpu_offload` / Wan `--offload_model`；Wan2.2 的空闲专家也停在主机）、`shard+offload`、`auto`（默认：第一个放得下的）。结果、警告、KPI 与假设列表写明所选放置与重载时间。
- 可选 VAE 分块解码 `workload.vae_tiling`（Web 复选框、CLI `--vae-tiling`，默认关）：CogVideoX（tile 240 × 360 px，重叠 1/6、1/5 → 480p 9 tile ×1.40）与 Mochi（256 px / stride 192 → 15 tile ×1.65）按 diffusers 默认参数；重叠重复计算与每 tile 权重重读计入，激活峰值按 tile。其余 VAE 开启时给出警告。
- 测试 `tests/test_core_dap_place.py`：DAP 有用 FLOPs 守恒与通信闭式、布局 / 延迟 / 显存、请求 FLOPs 与布局无关、放置（auto / 分片 / 卸载 / 重载闭式 / Wan2.2 空闲专家）、分块 tile 数与重叠。

### Changed
- 64 GiB LPDDR5X 上（默认 auto 放置）：Wan2.1-14B 单卡 / DP2 / SP2、Wan2.2-A14B PP2 / TP2、MiniMax-H3 全部布局从「放不下」变为「放得下」（卸载 +1.4–2.5 s / 请求，或 H3 PP2·TP2 分片 +0 s）。显式 `placement: resident` 与 0.44 完全一致；其余 1330 项结果（全部 LLM、蛋白质、能常驻的视频场景）逐字节不变。
- 请求 TFLOP（`tflop_per_request` / `dit_tflop_per_request`）改为整个副本的有用 FLOPs（每 rank 有用 FLOPs × TP·SP ÷ 每 rank 序列数），此前多卡布局只计一个 rank（例：Wan2.1-1.3B SP2 显示 14.2 PFLOP，实为 28.3）；按 SP 不切分的条件行（跨注意力 K/V、每序列行）记为重复。
- 结构模型的说明与 Web 工作负载说明改为「PP × DP × DAP」。

### Fixed
- 0.44 文档误称 Qwen3-VL 文本塔（66.7 GB = 62.1 GiB）单独超过 64 GiB；实际是与 DiT 同时常驻放不下。
- 两个组件在同一张卡时，激活超出量按两者最大值计（此前分别相加；现有场景中数值未变）。

### 仍未建模
- pair 的 TP；扩散样本的 DP 切分；DAP 无公开推理时延可逐项校验。
- DiT 权重 FSDP（Wan `--dit_fsdp`）、文本编码器在主机 CPU 上运行（`--t5_cpu`）、VAE 多卡并行解码、跨请求的组件 / 去噪重叠；Wan / LTX / Open-Sora / H3 的可选 VAE 分块。
- AlphaFold 3 仍为「暂未接入 v2」（无公开权重）。

## [0.44.0] - 2026-10-09

视频整条 pipeline：文本编码器与 VAE 解码的时间与存储计入评估；激活 dtype what-if。

### Added
- 视频 pipeline 组件（`accel_dse/core/pipeline.py`，数据 `accel_dse/data/pipeline/*.json` 由 `scripts/build_pipeline_data.py` 从发布检查点张量头生成，不下载权重）：文本编码器 umT5-XXL（Wan）、T5 v1.1 XXL（CogVideoX / LTX / Mochi / Open-Sora 的 DeepFloyd 版）、LLaVA-Llama-3-8B + CLIP-L 文本塔（HunyuanVideo）、Qwen3-VL 文本塔（H3）；VAE 解码器 Wan-VAE、CogVideoX、HunyuanVideo（diffusers 默认 tiling，720P 308 tile、重叠 ×2.64）、LTX、Mochi、Open-Sora v1.2（时间 VAE + 逐帧 SD-VAE）、H3 ViT 视频解码器与音频 VAE。卷积按隐式 GEMM（输入流量按真实张量），文本塔按补齐 token 数，提示数随 batch × CFG。定义与假设见 docs/MODEL.md §11.4。
- 默认计入：`T_clip = 文本编码 + 去噪 + 解码`，组件 FLOPs 计入每请求 TFLOP，组件权重（加载的整个检查点，dtype 按发布）计入首 / 末流水级卡的容量。`workload.pipeline = false`（Web 复选框、CLI `eval --dit-only`）只看 DiT，结果与 0.43 逐字节一致。
- 激活 dtype what-if：Web「激活 dtype」、CLI `eval --act fp32|bf16`（即 `formats_override [["act", …]]`），用于评估结构模型参考实现的 fp32 激活流量（例：ESMFold 100T + LPDDR5X 6.3 s → 12.5 s）。
- Web：单段延迟卡拆分「文本编码 + 去噪 + 解码」，吞吐卡显示组件 TFLOP，假设列表随模型 / pipeline / 激活 dtype 动态生成，容量面板显示组件权重并提供「只评估 DiT」修正；API 的级存储与 fit 返回 `pipe_w_GiB`。
- 测试：`tests/test_core_pipeline.py`（检查点大小、umT5 闭式 FLOPs、Qwen GQA / causal、各家 VAE 分辨率表与发布输出形状、Wan-VAE 手算、pipeline 开关等价与延迟加和、batch / DP / 容量、场景 / API / CLI、fp32 激活 what-if）。

### Changed
- 视频模型覆盖全部为「完整」（说明中列出组件与假设）。`validate` 的 Wan2.1 / Open-Sora 行改为整条 pipeline FLOPs（0.72 / 0.14，仍在带内）。
- 容量结论变化（64 GB LPDDR5X）：Wan2.1-14B（+umT5 11.4 GB，所有单卡 / DP / SP 布局）、Wan2.2-A14B 的 PP2 / TP2、MiniMax-H3（Qwen3-VL 66.7 GB 单独超过容量，组件不跨卡切分）从「放得下」变为「放不下」。HBM 上全部仍放得下。
- 延迟（100T + HBM3E、batch 1）：整段增加 0.6%（Wan2.1-14B）至 16.4%（Open-Sora 720p，逐帧 SD-VAE 1158 TFLOP）；HunyuanVideo 解码 221 s。LLM 与蛋白质结果不变。
- 隐式 GEMM 的 DRAM 流量按算子真实输入张量（`Op.in_elems`），映射的输出字节按 max(2, 激活字节)。

### Fixed
- Web：切换模型时 badge 停留在「只评估 DiT 主干」；pipeline 偏好跨模型保留（现随模型重置为默认，只保留 SLO）；结构模型的 badge 与假设文本（原显示编码器前向）；pair 模型 SLO 单位。

### 仍未建模
- 组件的多卡切分与并行 VAE 解码、文本编码器卸载、跨请求的组件 / 去噪重叠；CogVideoX / Mochi / H3 可选 VAE tiling 的重叠开销。
- 结构模型 pair 表示的 DAP 切分；fp32 激活 what-if 只计数据搬运，计算仍在 bf16 阵列。
- AlphaFold 3 仍为「暂未接入 v2」（无公开权重）。

## [0.43.0] - 2026-10-09

蛋白质结构预测接入 core v2：ESMFold、AlphaFold 2、OpenFold、Boltz-1、Protenix 可选择、可评估。

### Added
- 可评估的结构模型：ESMFold（facebook/esmfold_v1）、AlphaFold 2（官方 JAX 参数 `params_model_1_ptm`，CC BY 4.0）、OpenFold（`finetuning_ptm_2`）、Boltz-1（`boltz1_conf.ckpt`）、Protenix v0.5.0。参数取自检查点张量头（PyTorch zip 检查点只读 `data.pkl`；AF2 的 `.npz` 在发布 `.tar` 内按 tar / zip / `.npy` 头读取；HTTP range 请求，不下载权重），与发布逐项一致。dtype 按发布（fp32；ESMFold 的 ESM-2 为 fp16）。覆盖：ESMFold「完整」，其余「部分」（MSA / 模板检索与特征化、松弛未建模）。
- 结构模型算子：按模块名检测的三角乘法、三角注意力、MSA 行 / 列（含全局）注意力、外积均值、pair 加权平均、带 pair 偏置的注意力、IPA、模板点注意力、原子局部窗口注意力；每个权重 GEMM 按名字归到单一 / pair / MSA / 模板 / 原子网格，行数随残基数、MSA 行数、模板数、原子数变化。主干每遍重跑（recycle + 1 遍），扩散模块每步一次、样本成批，置信度头每样本一次。定义见 docs/MODEL.md §12。
- 工作负载新增 `msa`、`recycles`、`samples`（扩散步数沿用 `steps`）与结构模型的批延迟 SLO `fold_slo_s`（默认 120 s「假设」）；命令行 `eval --msa --recycles --samples`；扫描路径 `workload.msa / recycles / samples`。
- Web：结构模型的工作负载输入（MSA 行数、主干遍数、扩散步数 / 样本数、SLO s）、说明与指标卡（每序列 TFLOP、MSA / 遍数 / 扩散）；布局只显示 PP × DP。目录中离线条目显示原因。
- 校验：`validate` 增加 ESMFold 行（论文：V100 上 384 残基 14.2 s → 折算 V100 fp32 峰值的 28%）；新增 `tests/test_core_structure.py`（参数、AF2 JAX 参数与 OpenFold 逐层 GEMM 一致、核检测、闭式 FLOPs、recycle / 步数 / 样本线性缩放、PP × DP 布局、API）。

### Changed
- 结构模型只允许 PP × DP 布局（pair 表示的 DAP 切分未建模），TP / SP 报错并说明原因。
- 仍为「暂未接入 v2」：AlphaFold 3（权重需向 Google DeepMind 申请、条款禁止再分发，没有可公开核对的发布文件），目录中写明原因并指向 Protenix / Boltz-1。
- LLM、0.41 与 0.42 模型的结果不变（1296 项指纹逐字节一致）。

### Fixed
- Web：选择蛋白质模型（ESM-2）时工作负载说明的脚本错误（读取了视频专用字段）。

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

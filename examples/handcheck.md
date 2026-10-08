# 手算校验示例（toy）

本页每一步都给出中间数，可用计算器复现。对应代码路径：
`TOY_SHAPE` + `NPUConfig(4,4)` + `SRAMConfig(128 KiB)` + `HANDCHECK_MEM`。

**标签约定**：未标注 “derived” 的硬件旋钮均为 **assumed / uncalibrated**。

---

## 0. 形状（assumed placeholders）

| 符号 | 值 |
|------|----|
| L | 2 |
| H | 64 |
| n_heads | 4 |
| n_kv | 2 |
| d | 16 |
| F | 128 |
| V | 100 |
| weight/act/kv bits | 16 |

派生：

- `q_dim = n_heads * d = 64`
- `kv_dim = n_kv * d = 32`

### 每层权重参数（derived）

- Q: `H * q_dim = 64*64 = 4096`
- K: `H * kv_dim = 64*32 = 2048`
- V: `H * kv_dim = 64*32 = 2048`
- O: `q_dim * H = 64*64 = 4096`
- **Attn 合计** = `12288`
- gate: `H*F = 64*128 = 8192`
- up: `H*F = 8192`
- down: `F*H = 128*64 = 8192`
- **FFN 合计** = `24576`
- **每层参数** = `36864`
- **每层字节** (`*2`) = `73728`

### KV 每 token（derived）

`2 * L * n_kv * d * 2 bytes = 2*2*2*16*2 = 256 bytes/token`

---

## 1. NPU：output-stationary 利用率（derived）

Assumed PE：`R=4, C=4, n_engines=1`，`mac_efficiency=1`。

公式：

```
cycles = ceil(M/R) * ceil(N/(C*n_engines)) * K
util   = (M*N) / (R*C*n_engines * ceil(M/R) * ceil(N/(C*n_engines)))
```

取 `K=64, N=64`（与 Q 投影同形状）：

| M | cycles | util |
|---|--------|------|
| 1 | `ceil(1/4)*ceil(64/4)*64 = 1*16*64 = 1024` | `64/(16*1*16) = 0.25 = 1/R` |
| 8 | `2*16*64 = 2048` | `8*64/(16*2*16) = 1.0` |

**显式 M=1 曲线**：当 `N % (C*n_engines) == 0` 时，`util(M=1) = 1/R`；大 M 填满行后 → 1.0。
因此 decode（M=1）利用率 **严格小于** 大 M prefill。绝对吞吐仍随 `C * n_engines * freq` 增长（SKU 放大靠 engines）。

---

## 2. 外部带宽与 SRAM 分区（derived）

`HANDCHECK_MEM`：`1 ch × 64 bit × 2.0 GT/s × eff=1.0`

```
peak = 1 * (64/8) * 2.0 * 1e9 = 16e9 bytes/s = 16 GB/s
```

SRAM = `128 * 1024 = 131072` bytes。分区策略（**weight_resident**，见 README）：

- `cap // W_layer = 131072 // 73728 = 1` → **R=1** 层常驻；`weight_partition = 73728`
- `kv_needed (decode ctx=8) = 256*8 = 2048` → `kv_scratch = min(rem, 2048) = 2048`
- `act = rem - kv = 57344 - 2048 = 55296`
- 校验：`73728+2048+55296 = 131072`

权重路径标签：`hbm_stream_miss_layers`（R=1：稳态 decode 只流式读 L−R=1 层；staging-only 不会省字节）。

---

## 3. Prefill（S=8, B=1 → M=8）

### 3.1 每层投影 GEMM FLOPs（derived）

`FLOPs = 2*M*K*N`：

| GEMM | (M,K,N) | FLOPs |
|------|---------|-------|
| Q | (8,64,64) | 65536 |
| K | (8,64,32) | 32768 |
| V | (8,64,32) | 32768 |
| O | (8,64,64) | 65536 |
| gate | (8,64,128) | 131072 |
| up | (8,64,128) | 131072 |
| down | (8,128,64) | 131072 |
| **层合计** | | **589824** |

Attention（因果近似：矩形的 1/2）：

```
4 * n_heads * M * S * d / 2 = 4*4*8*8*16 / 2 = 8192
```

两层：`2 * (589824 + 8192) = 1196032` FLOPs。

### 3.2 每层投影 cycles（OS, derived）

对每个 GEMM：`cycles = ceil(M/4)*ceil(N/4)*K`，M=8 → `ceil(M/4)=2`：

| GEMM | cycles |
|------|--------|
| Q (8,64,64) | 2*16*64 = 2048 |
| K (8,64,32) | 2*8*64 = 1024 |
| V | 1024 |
| O | 2048 |
| gate (8,64,128) | 2*32*64 = 4096 |
| up | 4096 |
| down (8,128,64) | 2*16*128 = 4096 |
| **层合计** | **18432** |

Attn cycles（按峰值 FLOPs/cycle = 2*R*C*engines = 32）：`8192/32 = 256`。

两层 compute cycles：`2*(18432+256) = 37376`。

Assumed `f = 1 GHz` → `t_compute = 37376 / 1e9 = 3.7376e-5 s = 0.037376 ms`。

### 3.3 DRAM 字节（derived from capacities）

- Weights：`2 * 73728 = 147456`（cold-start：整模装入；R=1 常驻层也计入一次装载）
- KV write-through：`256 * 8 = 2048`（prefill 始终写外部持久 cache）
- Act spill：0（激活工作集 `8*64*2=1024` << act 分区）

`t_mem (serialize) = (147456+2048)/16e9 = 149504/16e9 = 9.344e-6 s = 0.009344 ms`

### 3.4 TTFT

`t = max(t_compute, t_mem) = 0.037376 ms` → **compute-bound**。

---

## 4. Decode 一步（ctx=8, B=1 → M=1）

### 4.1 FLOPs

层投影合计：`2*1*K*N` 对各 GEMM → **73728**/层  
（即为 prefill 层 FLOPs 的 1/8）。

Attn：`4*4*1*8*16 = 2048`。

两层：`2*(73728+2048) = 151552` FLOPs。

### 4.2 Cycles

M=1 → `ceil(1/4)=1`，各投影 cycles 恰为 M=8 时的一半行块：层投影 **9216** cycles；attn `2048/32=64`。  
两层：`2*(9216+64)=18560` → `t_compute = 0.018560 ms`。

均值 GEMM util = **0.25**（=1/R）。

### 4.3 DRAM + 争用（分区派生）

- Weights：`73728`（R=1 常驻 → 稳态只流式读 L−R=1 层；`hbm_stream_miss_layers`）
  - 若仅为 staging、R=0，则仍为 `147456`（staging **不**省字节）
- KV：`kv_scratch=2048 >= kv_full=2048` → **读命中**；新 token write-through `256`
  - `kv_bytes_dram = 0 + 256 = 256`
- `t_W = 73728/16e9 = 4.608e-6 s`
- `t_KV = 256/16e9 = 1.6e-8 s`
- serialize：`4.624e-6 s = 0.004624 ms`
- lower_bound：`max(t_W,t_KV)=4.608e-6 s`
- share_fair：`max(2*t_W, 2*t_KV)=9.216e-6 s`

**守恒检查**：`serialize >= max(t_W, t_KV)` ✓

### 4.4 TPOT

`t = max(0.018560, 0.004624) ms = 0.018560 ms/tok` → **compute-bound**（toy + 理想 BW；真实 27B + 100T SKU @ LPDDR decode 通常 memory-bound）。

---

## 5. 一键复现

```bash
cd /workspace/npu-inference-dse
python3 -m npu_dse handcheck
# 或跑 pytest 中的 test_toy_handcheck_numbers
python3 tests/run_tests.py
```

> 注意：CLI `eval --mem hbm` 默认用 `HBM_PRESET`（~TB/s），与本页 `HANDCHECK_MEM`（16 GB/s）不同。
> 手算对齐请用 `handcheck` 子命令 / pytest / 直接调 API。

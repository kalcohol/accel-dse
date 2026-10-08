# 外部存储规格调研（用于重建 accel_dse 外存目录）— 2026-10-08

> 仅调研 + 报告，未改任何代码。机器可读版：`research/memory_specs_2026-10.json`（含全部来源 URL key → URL 映射、选择器定义）。
>
> 标签约定：**JEDEC** = 已发布 JEDEC 标准/新闻稿明文；**JEDEC?** = 很可能在 JEDEC 规范内（厂商声明“JEDEC compliant”或 JEDEC 讲稿引用）但未从规范原文核实；**厂商量产** / **厂商送样** / **厂商公布**（论文/产品页上限，未确认量产档）；**推测** = 路线图、在研标准或本文推导。
>
> 带宽公式：`raw_GBps = 位宽 × MT/s / 8 / 1000`；`payload = raw × payload_factor`（LPDDR6 = 256/288 = 0.889，其余 = 1.0）；`effective = payload × efficiency`（假设旋钮，默认 0.70，需与 payload_factor 分开）。

---

## 0. 结论速览（与当前目录最相关的几条）

1. **LPDDR6 的建模单位应为“封装”，手机标准封装 = x96（4×24-bit 通道，每通道 2×12-bit 子通道）**——用户纠正正确。JESD209-6 定义的是“x24 单通道 SDRAM *器件*（die）”，但出货封装含 4 个通道：CXMT 16GB 1295-ball PoP（小米 18 Fold）、小米 XRING O3 = **4×24-bit @10.667 → 113.8 GB/s**（=96×10.667/8×8/9），三星 “16GB LPDDR6 … 最高 114 GB/s” 同样等于一颗 x96 封装。当前代码以 24-bit die 为粒度、die 数 {6,8,12,16,20} → **错误**。
2. **LPDDR6 必须计入 8/9 payload 系数**：每子通道 BL24 = 12×24 = 288 bit = 256 数据 + 16 metadata（存入阵列）+ 16 DBI/Link-ECC。JEDEC 讲稿给出单 x24 通道有效带宽 28.5 GB/s@10.667、38.4 GB/s@14.4。当前代码直接用 raw → 高估 12.5%。
3. **LPDDR6 速率档**：10667（JEDEC 入门档，已量产）、12800（三星 ISSCC'26 / CXMT 产品上限，JEDEC?）、14400（JEDEC 最高定义档，SK hynix ISSCC 硅片达到）。当前缺 12800。**未发现 LPDDR6X**（截至 2026-10-08）。
4. **LPDDR5X**：x64 封装（4×16）是主流，用户纠正正确；但另有 x32（2 通道，SOCAMM2 使用）和 **x96（6 通道，Micron FQ4'26 称已为旗舰手机出货，厂商扩展）**。速率档应为 7500 / 8533（JEDEC）/ 9600 / 10667（JEDEC?，三星/美光均量产 “10.7 Gbps”）。当前 6400 属于 LPDDR5，不应标 LPDDR5X；缺 7500、10667。
5. **HBM 容量模型全错**：当前 HBM3/3E = 12 GB/stack、HBM4 = 24 GB/stack。实际：HBM3 16/24 GB；HBM3E 24/36(/48) GB；HBM4 36 GB 量产、48 GB 送样、JEDEC 最大 64 GB；HBM4E 48 GB 送样、规划 32/64 GB。
6. **HBM4E 10 Gbps “设计目标”已过时**：三星 12H 48GB 样品稳定 14 Gbps、可到 16 Gbps（3.6 TB/s）；SK hynix 12H 样品最高 16 Gbps（~4 TB/s）。JEDEC 尚无 HBM4E 标准。HBM4 JEDEC 8 Gbps，而厂商量产 10–11.7 Gbps（三星产品页上限 13 Gbps）。
7. **SOCAMM2（JESD328，2026-06 发布）**：128-bit/模组，LPDDR5X 9600 → 153.6 GB/s/模组；48–256 GB/模组。NVIDIA Vera = 8 模组 × 128 = 1024-bit @9600 = 1.2 TB/s、256 GB–1.5 TB（Micron 称 256GB 模组可到 2 TB）。对 “LPDDR 推理卡” 极具参考价值。

---

## 1. LPDDR5 / LPDDR5X

### 1.1 标准与器件

| 项 | 值 | 标签 | 来源 |
|---|---|---|---|
| 标准 | JESD209-5 (2019, LPDDR5 ≤6400)；JESD209-5B (2021, 加入 LPDDR5X ≤8533)；**JESD209-5C (2023-07，当前版，LPDDR5/5X 合一)** | JEDEC | jedec.org/standards-documents/docs/jesd209-5c ；en.wikipedia.org/wiki/LPDDR |
| 器件 | “x16 单通道 SDRAM 器件和 x8 单通道器件”（x8 = byte-mode，高容量堆叠用） | JEDEC | 同上 |
| die 密度（JEDEC） | 2 Gb – 32 Gb（含 6/12/24 非二进制） | JEDEC | 同上 |
| 时钟关系 | 数据在 WCK 双沿：WCK = 速率/2；高速时 WCK:CK = 4:1 → CK = 速率/8（例：8533 MT/s ↔ WCK 4267 MHz ↔ CK 1067 MHz） | JEDEC（常识性，未逐条核规范原文） | Wikipedia LPDDR 表 |
| metadata | DMI 走独立引脚，不占数据带宽 → payload_factor = 1.0 | JEDEC | jedec.org Moon 讲稿 (LP5: (DQ16+DM2)×BL16) |

### 1.2 速率档（MT/s，单封装带宽以 x64 计）

| 代 | MT/s | WCK MHz | CK MHz | x64 封装 GB/s | 标签 | 说明 / 来源 |
|---|---|---|---|---|---|---|
| LPDDR5 | 5500 | 2750 | 688 | 44.0 | JEDEC | Wikipedia |
| LPDDR5 | 6400 | 3200 | 800 | 51.2 | JEDEC | LPDDR5 上限；三星 LPDDR5 “up to 6.4 Gbps” semiconductor.samsung.com/dram/lpddr/ |
| LPDDR5X | 7500 | 3750 | 938 | 60.0 | JEDEC? | Micron 1β 7.5 Gbps（micron.com LPDDR5X 页） |
| LPDDR5X | 8533 | 4267 | 1067 | 68.3 | **JEDEC** | JESD209-5B LPDDR5X 上限；Synopsys 博客 |
| LPDDR5X | 9600 | 4800 | 1200 | 76.8 | JEDEC? | 原 SK hynix “LPDDR5T”，后并入 LPDDR5X-9600（Wikipedia 引 JEDEC 车规讲稿）；SOCAMM2/Vera/LPDDR5X-PIM 均跑 9600 |
| LPDDR5X | 10667 | 5333 | 1333 | 85.3 | JEDEC? / 厂商量产 | 三星、美光 “10.7 Gbps” 量产；Micron 脚注称“the highest LPDDR5X speed grade … compliant with JEDEC standards”；Wikipedia 表 5X 上限 10667。**未从 JESD209-5C 原文核实** |

> x32 封装带宽减半；x96 封装 ×1.5（8533→102.4，9600→115.2，10667→128.0 GB/s）。

### 1.3 封装宽度

| 宽度 | 通道结构 | 用途 / 证据 | 标签 |
|---|---|---|---|
| x32 | 2×16 | 三星 LPDDR5X “Organization x64, x32”；SOCAMM2 Raw Card A 附录 “x32 LP5/5X DRAM up to 4 Rank” | 厂商量产 / JEDEC 模组附录 |
| **x64（默认）** | 4×16 | 手机 PoP 496-ball、分立 441/315/245/563-ball；三星 LPDDR5X-PIM 561-ball 16GB @9600 = 76.8 GB/s 正好是 x64 | 厂商量产 |
| x96 | 6×16 | Micron FY26 Q4：“now shipping 6-channel 1-gamma LPDDR5X for flagship mobile devices”（二手报道转述财报；球位未公开） | 厂商扩展 |

### 1.4 die / 封装容量

| 项 | 值 | 标签 | 来源 |
|---|---|---|---|
| Micron 1γ 16Gb | 高量产（某头部手机 OEM） | 厂商量产 | Micron FQ3'26 新闻稿 |
| Micron 1γ 24Gb | 送样手机客户 | 厂商送样 | 同上 |
| Micron 1γ 32Gb 单体 die | 用于 256GB SOCAMM2（64 颗） | 厂商量产 | micron.com SOCAMM2 页；TechRadar |
| SK hynix 1c 24Gb LPDDR5X | 10.7 Gbps（二手报道） | 厂商公布 | smbom.com（弱来源） |
| 三星 LPDDR5X 1b/24Gb/32Gb die | **未找到三星官方 die 级规格**；三星只公布封装密度 16Gb~192Gb（2–24 GB），以及 2024 年 “up to 32GB package” | 未核实 | samsung LPDDR 页；news.samsung.com 10.7Gbps 新闻 |
| 手机/PC 封装容量 | 4、6、8、12、16、24、32 GB（6/12/24 为 Micron 非二进制） | 厂商量产（32 GB 为厂商公布） | Micron 非二进制博客；三星 |
| 服务器 x32 封装 | 最高 64 GB（16×32Gb，x8 byte-mode，4 rank × 4 die）— 由 256GB SOCAMM2 = 4 封装 × 64 die 推导 | 推导 | Micron/TechRadar + JEDEC RCA 附录 |

---

## 2. LPDDR6

### 2.1 标准要点（JESD209-6，2025-07-09 发布）

| 项 | 值 | 标签 | 来源 |
|---|---|---|---|
| 器件定义 | “JEDEC compliant **x24 one channel** SDRAM device” | JEDEC | jedec.org/standards-documents/docs/jesd209-6 |
| 通道结构 | 每 die 2 个独立子通道，每子通道 12 DQ + 4 CA（+CS、独立 CK） | JEDEC | JEDEC 新闻稿；Moon 讲稿 |
| 突发 | BL24（32B 访问）/ BL48（64B），on-the-fly BL 控制 | JEDEC | JEDEC 新闻稿 |
| **数据包** | 12 DQ × BL24 = **288 bit = 256 data + 16 metadata(存阵列) + 16 DBI 或 Link ECC/EDC**，无 DMI 引脚 | JEDEC | Moon 讲稿；Synopsys(Brett Murdock) 讲稿；Wikipedia |
| payload 系数 | 256/288 = **0.8889**（BL48 同为 512/576） | JEDEC 推导 | 同上；讲稿：28.5 GB/s@10.667、38.4 GB/s@14.4（每 x24 通道） |
| die 密度 | 4–64 Gb（MBIST 表列出 4/8/12/16/24/32/48/64 Gb） | JEDEC | jesd209-6 页；Takahashi 讲稿 |
| 速率 | 入门 10.667 Gbps，最高定义 14.4 Gbps | JEDEC | Murdock 讲稿；Synopsys |
| 时钟 | CK:WCK 比恒定；SK hynix 论文 CA 频率 1.6–3.6 GHz → CK ≈ 速率/4，WCK = 速率/2（细节未核规范） | JEDEC?/推导 | Takahashi 讲稿；MoreThanMoore ISSCC 综述 |
| 效率模式 | 关闭一个子通道 I/O，由一个子通道访问两边 bank（省电/高容量 static efficiency mode） | JEDEC | 新闻稿；ISSCC |
| RAS | on-die ECC 强制、PRAC（逐行激活计数）、MBIST、ALERT 引脚 | JEDEC | 同上 |
| 下一版（2026-04 预告） | 新增 **x6 子通道模式**（每封装更多 die、更高容量）、**可调 metadata 区**（会影响 payload 系数）、512 GB 密度、LPDDR6 SOCAMM2、LPDDR6-PIM 标准“接近完成” | 推测（在研） | JEDEC 2026-04-22 新闻稿 |

### 2.2 封装宽度（核心纠正）

| 宽度 | 结构 | 证据 | 标签 |
|---|---|---|---|
| **x96（默认）** | 4 × x24 通道 = 8 × x12 子通道 | CXMT 16GB、1295-ball FBGA PoP、16Gb die，用于小米 18 Fold；XRING O3 “4 个 24-bit 通道 @10.667 GT/s，113.8 GB/s”；三星 16GB LPDDR6 “up to 114 GB/s”；三星料号 PDBEREAGY0B1 128Gbit 1295-ball | 厂商量产 |
| x48（？） | 2 × x24 | 三星产品页列 357 / 518 / 1295 FBGA 三种封装，但未公布 357/518 的通道数 → x48 只是推测 | 推测 |

> 注意：JEDEC 把“通道”定义在 die 上（x24）；“96-bit”是**封装/PoP 惯例**（类比 LPDDR5X 的 x64 = 4×16），不是 JEDEC 对 die 的定义。两种说法并不矛盾，但 DSE 的选择粒度应是封装。

### 2.3 速率档与单封装带宽（x96）

| MT/s | WCK MHz | CK MHz(≈) | raw GB/s | payload GB/s (×8/9) | 标签 | 来源 |
|---|---|---|---|---|---|---|
| 10667 | 5333 | 2667 | 128.0 | **113.8** | **JEDEC** + 厂商量产 | 三星/Qualcomm 验证、SK hynix 1c、CXMT 量产、XRING O3 |
| 12800 | 6400 | 3200 | 153.6 | 136.5 | JEDEC? | 三星 ISSCC'26 16Gb（论文称为 JEDEC 最低电压点）、CXMT 产品最高 12800 |
| 14400 | 7200 | 3600 | 172.8 | 153.6 | **JEDEC**（最高定义） / 厂商公布 | SK hynix ISSCC'26 1c 16Gb @1.025V；三星产品页 “up to 14.4 Gbps” |

单 x24 通道：28.4 / 34.1 / 38.4 GB/s（payload）。

### 2.4 容量

| 项 | 值 | 标签 | 来源 |
|---|---|---|---|
| 量产 die | 16 Gb（三星、SK hynix 1c、Micron 1γ 送样、CXMT 量产） | 厂商量产 | SK hynix 2026-03 新闻；CXMT；ISSCC 综述 |
| 封装容量 | 16 GB 量产（CXMT、三星）；三星产品页密度 16Gb–128Gb（2–16 GB/封装） | 厂商量产 | CXMT；三星 LPDDR 页 |
| 8/12 GB | 合理 SKU，未见公告 | 推测 | — |
| 24/32 GB | 需 24/32Gb die 或更多叠层；JEDEC 允许，未公告 | 推测 | — |

### 2.5 LPDDR6X / PIM / 模组

- **LPDDR6X**：未发现 JEDEC 标准或厂商产品（截至 2026-10-08）。
- **LPDDR6-PIM**：JEDEC 2026-04 称“接近完成”；三星希望 2026 年底前拿到初版规范。三星 **LPDDR5X-PIM** 已有硅片（Hot Chips 2026）：561-ball、16GB、外部 76.8 GB/s、内部 PIM 带宽 614 GB/s、SINT4 2.4 TOPS/封装（techtimes 转述 STH/Tom's）。SK hynix LPDDR6 PIM 约 2028（二手）。
- **LPDDR6 CAMM2**：JEDEC 2024 讲稿 → 192-bit/模组、48-bit 通道、至 14.4 GT/s（推测/在研，TechPowerUp）。
- **LPDDR6 SOCAMM2**：JEDEC 在研（2026-04），宽度/容量未公开。

---

## 3. LPDDR 服务器模组（LPDDR 推理卡强相关）

| 模组 | 标准 | 位宽 | 速率 | 单模组 BW | 容量 | 标签 / 来源 |
|---|---|---|---|---|---|---|
| SOCAMM（gen1, Micron） | 非 JEDEC | 128 | 8533 | 136.5 GB/s | 128 GB（4 × 16-die 堆叠） | 厂商量产；Micron 2025-03 新闻稿 |
| **SOCAMM2** | **JESD328 v1.00 (2026-06)** + RCA 附录（x32 DRAM，≤4 rank） | 128（4 × x32 封装，推导） | 9600 | **153.6 GB/s**（三星官方） | 48/64/96/128/192/256 GB（Micron 48–256；SK hynix 192 量产；三星 ≤256） | JEDEC + 厂商量产 |
| LPCAMM2 | JESD318（CAMM2，LP5/5X） | 128（2×64 通道，8×16 子通道） | 7500/8533（三星 ≤8533） | 120/136.5 GB/s | 16/32/64 GB（96 GB 需 24Gb die） | JEDEC + 厂商量产 |
| LPDDR6 CAMM2 | 在研 | 192（48-bit 通道） | ≤14400 | — | — | 推测 |

参考系统：

| 系统 | 配置 | 带宽 | 容量 | 来源 |
|---|---|---|---|---|
| NVIDIA **Vera** CPU | 8 × SOCAMM2，1024-bit，9600 MT/s | 1.2 TB/s（=1228.8 GB/s） | 256 GB – 1.5 TB（Micron：256GB 模组可达 2 TB） | Tom's Hardware Vera 深析；Micron GTC'26 |
| NVIDIA **Grace** | 板载 LPDDR5X（带 ECC） | 512 GB/s（120/240GB）；384 GB/s（480GB） | 120/240/480 GB | Grace 数据手册；Chips&Cheese：480GB SKU 为 480-bit LPDDR5X-6400 |
| Intel **Crescent Island**（推理 PCIe 卡，350W） | LPDDR5X | 官方未公布（估算 0.68–1.5 TB/s） | 参考 160 GB，伙伴板最高 480 GB | Tom's Hardware Hot Chips 2026 |
| Qualcomm AI200 | LPDDR（报道为 LPDDR5X） | 未公布 | 768 GB/卡 | Qualcomm 新闻稿 |

---

## 4. HBM

### 4.1 代际总表（每 stack）

| 代 | JEDEC 文档 | 接口位宽 | 通道 | JEDEC 最高速率 | 厂商量产速率 | 每 stack BW | 堆叠层数 | die 密度 | 每 stack 容量 |
|---|---|---|---|---|---|---|---|---|---|
| HBM3 | JESD238 (2022-01)，现行 JESD238B.01 (2025-04) | 1024 | 16 ch × 64b（32 伪通道） | **6.4 Gbps → 819 GB/s** | 6.4（三星） | 819 GB/s | 4/8/12（16 为“未来扩展”） | 8–32 Gb（首代 16Gb） | 16 GB(8H×16Gb)、24 GB(12H×16Gb) 量产；JEDEC 最大 64 GB |
| HBM3E | **无独立 JEDEC 文档**（HBM3 厂商扩展；JESD238A/B 是否加入 >6.4 档无法公开核实） | 1024 | 同 HBM3 | — | 8.0（B200/MI355X 级有效 ~7.8）、9.2（Micron >9.2、三星产品页 9.2/1180 GB/s）、9.6（SK hynix 12H）、9.8（三星 Shinebolt 公布） | 1.02–1.25 TB/s | 8/12/16 | 24 Gb（8H 24GB 用 24Gb） | 24 GB(8H)、36 GB(12H) 量产；48 GB(16H, SK hynix) 公布 |
| HBM4 | **JESD270-4 (2025-04-16)**；JESD270-4A v1.1 (2025-12，改动不公开) | **2048** | 32 ch × 64b（64 伪通道） | **8 Gbps → 2.0 TB/s** | SK hynix >10、Micron >11（>2.8 TB/s）、三星 11.7（产品页上限 13.0 / 3.3 TB/s） | 2.0–3.3 TB/s | 4/8/12/16 | **24 / 32 Gb** | 36 GB(12H×24Gb) 量产；48 GB(16H, Micron 送样)；JEDEC 最大 64 GB(16H×32Gb) |
| HBM4E | **无 JEDEC 标准**（厂商定义） | 2048 | 同 HBM4（假设） | — | 三星样品稳定 14、可到 16（3.6 TB/s）；SK hynix 样品最高 16（~4 TB/s）；Micron 1γ HBM4E 2027 量产 | 3.6–4.1 TB/s | 8/12/16 | 32 Gb（1c / 1γ） | 48 GB(12H) 送样；三星计划 32 GB(8H)、64 GB(16H) |
| SPHBM4 | JESD330-4 (2026-07) | **512**（4:1 串行化） | — | 与 HBM4 同总带宽 | — | ≈HBM4 | 同 HBM4 die | 同 HBM4 | 同 HBM4；可贴有机基板 |

来源：JEDEC HBM3 新闻稿；JEDEC JESD238B.01 页；JEDEC HBM4 新闻稿与 JESD270-4A 页；三星 HBM 产品页；三星 36GB HBM3E 12H 新闻；Micron HBM3E 页；SK hynix 12H HBM3E 量产新闻、16H 报道；SK hynix HBM4 量产准备新闻（“远超 JEDEC 8Gbps，>10Gbps”）；Micron GTC'26 新闻稿；三星 HBM4E 样品新闻；TheElec SK hynix HBM4E；JEDEC SPHBM4 新闻稿。

**base die**：HBM4 起 base die 改为逻辑工艺 — 三星自家 4nm；SK hynix HBM4 用 TSMC 12nm 级，HBM4E 据报 TSMC 3nm 级；Micron HBM4E 用 TSMC base die；三家都在做客户定制 base die（custom HBM / C-HBM4E，TSMC/GUC 目标 2027 年 12.8 GT/s，Tom's Hardware）。

### 4.2 加速器上的 stack 配置（反推每 pin 速率）

| 产品 | HBM | stack 数 | 容量 | 带宽 | 反推 Gbps/pin | 来源 |
|---|---|---|---|---|---|---|
| H100 SXM | HBM3 | 5（6 位置启用 5，二手） | 80 GB | 3.35 TB/s | 5.23 | NVIDIA H100/H200 页 |
| H200 | HBM3E | 6 | 141 GB | 4.8 TB/s | 6.25 | nvidia.com/h200 |
| B200（HGX） | HBM3E 8H | 8 | 180 GB（另有 192 GB 口径） | 7.7–8 TB/s | ~7.5–7.8 | NVIDIA HGX 页（8 GPU 共 1.4 TB）；AMD 对比页 |
| B300（HGX） | HBM3E 12H | 8 | ~270–288 GB（8 GPU 共 2.1 TB） | 8 TB/s | ~7.8 | NVIDIA HGX 页 |
| AMD MI355X | HBM3E 12H | 8 | 288 GB | 8 TB/s | 7.8 | amd.com MI355X |
| **NVIDIA Rubin** | HBM4 12H | 8 | 288 GB | **NVL72 页 19.2 TB/s；HGX 页 22 TB/s；HGX Rubin NVL8 总 130 TB/s（=16.25/GPU）** | 9.4–10.7 | NVIDIA 官方页面互相矛盾；TechPowerUp：22 TB/s 目标因供应商未达标而下调 |
| AMD MI455X | HBM4 | **12** | 432 GB | 23.3 TB/s | 7.6 | amd.com MI455X（2026-07 发布） |
| Google TPU7x Ironwood | HBM3E | 8 | 192 GB | 7.37 TB/s | 7.2 | Google Cloud TPU7x 文档 |
| Google TPU 8t / 8i | HBM3E | 6 / 8 | 216 / 288 GB | 6.53 / 8.60 TB/s | — | Tom's Hardware（二手，Hot Chips 2026） |

要点：**加速器实际跑的 pin 速率普遍低于厂商 HBM 颗粒标称上限**（B200/MI355X HBM3E ~7.8 vs 颗粒 9.2–9.6；MI455X HBM4 ~7.6 < JEDEC 8；Rubin ~9.4–10.7 vs 颗粒 11–11.7）。DSE 里应把“颗粒速率档”和“系统可达速率”分开，或至少默认取保守值。

### 4.3 功耗（仅作“报道值”）

| 项 | 值 | 标签 | 来源 |
|---|---|---|---|
| HBM4 vs HBM3E 能效 | SK hynix >40%；Micron >20%（对比其 HBM3E 12H @9.2）；三星 HBM4E 比 HBM4 +16% | 厂商公布（相对值） | SK hynix / Micron / 三星新闻稿 |
| 绝对 HBM pJ/bit | 未找到厂商一手绝对值；二手资料称 HBM3E ~4 pJ/bit | 未核实 | — |
| Grace LPDDR5X 子系统 | “约 500 GB/s、约 16 W” → ~4 pJ/bit（系统级，含背景功耗，推导） | 推导自厂商数据 | Grace 数据手册 |
| Vera SOCAMM2 子系统 | 1.2 TB/s，满配 30–40 W → ~3–4 pJ/bit（推导） | 推导自厂商数据 | Tom's Hardware Vera |
| LPDDR6 vs LPDDR5X | 三星 ~21% 能效提升，读功耗 73%/写 78%（ISSCC）；SK hynix >20% | 厂商公布 | 三星、SK hynix、ISSCC 综述 |

---

## 5. 当前目录（`accel_dse/package_ranges.py` / `memory.py`）的问题清单

| # | 位置 | 现状 | 问题 | 应改为 |
|---|---|---|---|---|
| 1 | `LPDDR6_DIE_WIDTH_BITS=24`、`LPDDR6_N_DIES=(6,8,12,16,20)`、`build_lpddr6_packages` | 以 24-bit die 为选择粒度，总线 144–480 bit | **建模单位错误**；20×24=480 之类组合非物理封装；docstring 写“用户 48-bit 猜测错误、粒度 = 24-bit die” | 粒度 = **x96 封装**（4×24），`bus = n_packages × 96`；x48 仅作“推测”选项 |
| 2 | LPDDR6 带宽 | raw 直接 × efficiency | 未计 256/288 payload → 峰值高估 12.5% | 新增 `payload_factor`（LPDDR6 = 8/9），与 `efficiency` 分离 |
| 3 | `LPDDR6_RATE_BAND_GTS=(10.667,14.4)` | 两档 | 缺 12800；14400 目前只是论文/上限，不是量产档 | 10667（JEDEC/量产）、12800（JEDEC?/厂商）、14400（JEDEC 上限/厂商公布） |
| 4 | `LPDDR5X_RATE_BAND_GTS=(6.4,8.533,9.6)` 且 generation 一律标 “LPDDR5X” | 6400 被标为 LPDDR5X | 6400 是 LPDDR5 上限；缺 7500、10667（三星/美光已量产 10.7） | 拆成 LPDDR5{5500,6400} 与 LPDDR5X{7500,8533,9600,10667}，各带标签 |
| 5 | `LPDDR5X_PACKAGE_WIDTH_BITS=64` 唯一 | 只有 x64 | x64 作默认正确；但缺 x32（SOCAMM2 用）与 x96（Micron 6 通道，厂商扩展） | 封装宽度选择器 {32, 64(默认), 96(厂商扩展)} |
| 6 | `_lpddr_capacity_gb` | 容量 = max(16, 64×总位宽/512) | **非物理**：容量与位宽线性绑定，LPDDR6 24-bit 粒度下更离谱 | 容量 = n_packages × 每封装容量（可选列表，见 §6） |
| 7 | `_hbm_capacity_gb` | HBM3/3E 12 GB/stack，HBM4 24 GB/stack | **全部偏低/错误** | 由 堆叠层数 × die 密度 / 8 计算：HBM3 16/24；HBM3E 24/36/48；HBM4 24/32/36/48/64；HBM4E 32/48/64 |
| 8 | `HBM_GEN_LABELS["HBM4e"] = 2048b @ 10.0` | “设计目标” | 过时：厂商样品 14–16 Gbps | HBM4E {12800 推测, 14000 送样, 16000 送样}；标注“无 JEDEC 标准” |
| 9 | `HBM_GEN_LABELS["HBM4"] = 8.0` 唯一 | 仅 JEDEC 档 | 无法表达量产 10–11.7 / 上限 13 | HBM4 {8000 JEDEC, 10000/11000/11700 量产, 13000 厂商公布} |
| 10 | `HBM_GEN_LABELS["HBM3e"] = 9.2` 唯一 | — | 9.2 可作默认；缺 8.0（系统实跑）、9.6、9.8 | HBM3E {8000, 9200(默认), 9600, 9800} |
| 11 | `HBM_STACK_COUNTS=(2,4,6,8)` | — | 缺 5（H100）、12（MI455X）、1（小卡）、16（未来/路线图） | {1,2,4,5,6,8,12,16}；>12 标推测 |
| 12 | `HBM_PRESET` | 8 × 1024b × 5.2 GT/s，96 GB | 5.2 GT/s 无对应代际档；96 GB/8 = 12 GB/stack 不存在 | 例如 HBM3E 8×36GB @9.2（MI355X/B300 级）或标明为“H100 级 HBM3 5 stack” |
| 13 | `LPDDR_PRESET` | 8 × 64b × 8.5 GT/s，64 GB | 8.5 不是档位（应 8.533）；64 GB = 8×8 GB 可行但偏小 | 8 × x64 @8533，8 × 16 GB = 128 GB |
| 14 | `DEFAULT_MEM_RATE_SWEEP_HBM_V16=(6.4,9.2,8.0,10.0)` | 速率与代际脱钩 | 扫参会产生如 “HBM3 @10.0” 这类无效组合 | 速率必须依附代际选择 |
| 15 | `MemGeometry.n_packages` 乘通道、`PackageOption` 强制 n_packages=1 | 概念混用 | “channel”在代码里实为 封装/stack，易误导 | 显式字段：`n_units`、`unit_width_bits`、`unit_kind ∈ {package, module, stack}` |
| 16 | 文档 | `REQUIREMENTS_GAP.md:52,77`、`package_ranges.py` 顶部 docstring | 写有“LPDDR6 = x24 die（非用户猜的 48-bit）” | 同步更正为 x96 封装 + payload 8/9 |
| 17 | `web/app.js` | 预设 id `lpddr_8x64_8533`、`hbm_hbm3e_4s` 等；单一下拉 | id 将随重构变化；单下拉无法表达维度 | 改为多选择器（见 §6），保留旧 id 映射 |

---

## 6. 选择器设计建议

### 6.1 选择器与派生量

```
mem_type      ─┬─ LPDDR5 | LPDDR5X | LPDDR6            （LPDDR 分支）
               └─ HBM3 | HBM3E | HBM4 | HBM4E          （HBM 分支）

LPDDR 分支：
  form_factor        discrete | SOCAMM2 | LPCAMM2 | (LPDDR6 CAMM2 / LPDDR6 SOCAMM2 = 推测)
  package_width_bits 依代际/形态（下表）
  speed_MTps         依代际（下表），UI 同时显示 WCK/CK MHz
  n_packages / n_modules
  package_capacity_GB（每封装/每模组）
HBM 分支：
  speed_MTps, stack_height, die_density_Gb, n_stacks
公共：efficiency（默认 0.70，可调）

派生：
  bus_width_bits = n_packages × package_width_bits   (HBM: n_stacks × 1024|2048；模组: n_modules × 128|192)
  raw_GBps       = bus_width_bits × MTps / 8000
  payload_GBps   = raw_GBps × payload_factor          (LPDDR6 = 256/288，其它 = 1)
  effective_GBps = payload_GBps × efficiency
  capacity_GB    = n_packages × package_capacity_GB   (HBM: n_stacks × height × die_Gb / 8)
  tag            = 各分量标签中最弱者（JEDEC > JEDEC? > 厂商量产 > 厂商送样 > 厂商公布 > 推测）
```

### 6.2 各代允许值（默认值加粗）

| 选择器 | LPDDR5 | LPDDR5X | LPDDR6 |
|---|---|---|---|
| 封装宽度 | 32, **64** | 32（厂商）, **64**, 96（厂商扩展） | **96**, 48（推测） |
| 速率 MT/s | 5500, **6400** (JEDEC) | 7500 (JEDEC?), **8533** (JEDEC), 9600 (JEDEC?), 10667 (JEDEC?/量产) | **10667** (JEDEC/量产), 12800 (JEDEC?), 14400 (JEDEC 上限/厂商公布) |
| 封装数 | 1,2,4,6,8 | 1,2,3,4,6,**8**,10,12,16 | 1,2,3,**4**,6,8 |
| 每封装容量 GB | 3,6,8,**12** | x64: 4,6,8,12,**16**,24,32(厂商公布)；x32: 8,16,32,64；x96: 12,16,24 | 8(推测),12,**16**,24(推测),32(推测) |
| payload_factor | 1.0 | 1.0 | 0.8889 |

| 模组 | 位宽 | 速率 | 模组数 | 每模组容量 GB | 标签 |
|---|---|---|---|---|---|
| SOCAMM2 (LPDDR5X) | 128 | 8533, **9600** | 1,2,4,6,**8** | 48,64,96,128,**192**,256 | JEDEC + 量产 |
| LPCAMM2 (LPDDR5X) | 128 | 7500, **8533**, 9600(推测) | 1,2 | 16,**32**,64,96(厂商公布) | JEDEC + 量产 |
| LPDDR6 CAMM2 / SOCAMM2 | 192 / 未定 | 10667–14400 | — | — | 推测 |

| 选择器 | HBM3 | HBM3E | HBM4 | HBM4E |
|---|---|---|---|---|
| stack 位宽 | 1024 | 1024 | 2048 | 2048 |
| 速率 MT/s | **6400** (JEDEC) | 8000 (系统实跑), **9200**, 9600 (量产), 9800 (公布) | **8000** (JEDEC), 10000/11000/11700 (量产), 13000 (公布) | 12800 (推测), **14000** (送样), 16000 (送样) |
| 层数 | 8, **12** | 8, **12**, 16(公布) | 8, **12**, 16(送样) | 8, **12**, 16 |
| die 密度 Gb | **16**, 24 | **24** | **24**, 32 | **32** |
| 每 stack 容量 GB | 16, **24**, (64 JEDEC 上限) | 24, **36**, 48 | 24, 32, **36**, 48, 64 | 32, **48**, 64 |
| stack 数 | 1,2,4,5,6,**8**,12,16(推测) | 同左 | 同左 | 同左 |

### 6.3 推荐默认组合与带宽（raw / payload）

| 组合 | 总线 | raw | payload | 容量 | 标签 |
|---|---|---|---|---|---|
| LPDDR5X 8 × x64 @8533，16 GB/封装 | 512-bit | 546.1 GB/s | 546.1 | 128 GB | JEDEC |
| LPDDR5X 8 × x64 @10667，16 GB | 512-bit | 682.7 | 682.7 | 128 GB | JEDEC? |
| LPDDR6 4 × x96 @10667，16 GB | 384-bit | 512.0 | **455.1** | 64 GB | JEDEC + 量产 |
| LPDDR6 8 × x96 @14400，16 GB | 768-bit | 1382.4 | 1228.8 | 128 GB | JEDEC 上限（厂商未量产此档） |
| SOCAMM2 8 × 128 @9600，192 GB | 1024-bit | 1228.8 | 1228.8 | 1536 GB | JEDEC + 量产（=Vera） |
| HBM3E 8 × 12H×24Gb @9200 | 8192-bit | 9420.8 | 9420.8 | 288 GB | 厂商量产 |
| HBM4 8 × 12H×24Gb @8000 | 16384-bit | 16384 | 16384 | 288 GB | JEDEC |
| HBM4 8 × 12H @10700（≈Rubin 22 TB/s 口径） | 16384-bit | 21913.6 | 21913.6 | 288 GB | 厂商量产/系统 |

### 6.4 实现提示（供后续改代码时参考，本次未改）

- 用数据表驱动（直接读 `research/memory_specs_2026-10.json` 或其精简版放进 `accel_dse/data/`），不要在 Python 里再硬编码元组。
- `ExternalMemory` 增加 `payload_factor`、`unit_kind`、`n_units`、`unit_width_bits`、`spec_tag` 字段；`peak_bandwidth_Bps` 返回 raw，另给 `payload_bandwidth_Bps`。
- Web UI：一个下拉拆成 5–6 个级联下拉，非法组合（如 HBM3 @10000、LPDDR6 x64）直接不出现；每个组合显示标签徽章。
- 旧 `package_id`（如 `lpddr_8x64_8533`、`hbm_hbm3e_8s`）做兼容映射，`lpddr6_*x24_*` 旧 id 映射到最近的 x96 组合并给出弃用提示。

---

## 7. 不确定项 / 待核实

1. **LPDDR5X 9600/10667 是否写入 JESD209-5C 原文**：只有 Micron“compliant with JEDEC standards”脚注、Wikipedia 表与 JEDEC 车规讲稿间接证据（规范需付费，未读原文）。
2. **LPDDR6 x96 是否为 JEDEC 定义的封装**：JESD209-6 含封装/球位定义，但公开摘要只说“x24 单通道器件”。x96 的证据来自量产产品（CXMT 1295-ball PoP、XRING O3、三星 114 GB/s）。三星 357/518-ball 封装的宽度未知。
3. **LPDDR6 12800 是否为正式速率 bin**：三星论文措辞暗示是 JEDEC 定义点；未见 JEDEC 速率 bin 表。
4. **LPDDR6 下一版“可调 metadata 区”** 可能改变 8/9 payload 系数（数据中心模式）——建议 payload_factor 做成可配置。
5. **Micron “6-channel LPDDR5X”** 来自财报二手转述，宽度 96-bit 为推断（6×16），封装细节未公开。
6. **HBM3E 是否有 JEDEC 速率档**：没有名为 HBM3E 的 JEDEC 文档；JESD238A/B 的改动内容不公开。
7. **Rubin 带宽**：NVIDIA 自家页面 19.2 / 22 / 130÷8=16.25 TB/s 互相矛盾，建议 DSE 默认按保守值 ~19–20 TB/s 或直接按 per-pin 档位算。
8. **H100 5 stack、H200 6 stack、TPU 8t/8i** 的 stack 数来自二手资料。
9. **Grace 总线宽度**（480-bit 数据 vs 512-bit 含 ECC）仅 Chips&Cheese 说明；NVIDIA 未公布封装数。
10. **功耗 pJ/bit**：无厂商一手绝对值；表中数值是由厂商系统功耗/带宽推导的系统级量，仅供量级参考。
11. **SOCAMM2 = 4 × x32 封装**、x32 封装 64 GB = 16 × 32Gb x8 die：由 128-bit 模组、JEDEC x32 附录和 Micron“64 颗 32Gb”推导，非官方逐项声明。
12. 三星 LPDDR5X 1b nm 16/24/32Gb **die** 规格：未找到三星官方 die 级数据（只有封装密度）。

---

## 8. 主要来源（URL）

- JEDEC LPDDR6 JESD209-6：https://www.jedec.org/standards-documents/docs/jesd209-6
- JEDEC LPDDR6 新闻稿（2025-07-09）：https://www.jedec.org/news/pressreleases/jedec%C2%AE-releases-new-lpddr6-standard-enhance-mobile-and-ai-memory-performance
- JEDEC LPDDR6 讲稿（Moon，metadata 288=256+16+16）：https://www.jedec.org/sites/default/files/Seunghyun%20Moon_03_29_25_Final.pdf
- JEDEC/Synopsys 讲稿（10.667/14.4、28.5/38.4 GB/s）：https://www.jedec.org/sites/default/files/Brett%20Murdock_FINAL_Mobile_2024.pdf
- JEDEC LPDDR6 讲稿（Takahashi，密度表、CK:WCK）：https://www.jedec.org/sites/default/files/JEDEC-PPT-16-9_LPDDR6_2025_hitakahashi.pdf
- JEDEC LPDDR6 路线图（2026-04-22）：https://www.jedec.org/news/pressreleases/jedec%C2%AE-previews-lpddr6-roadmap-expanding-lpddr-data-centers-and-processing-memory
- JEDEC LPDDR5/5X JESD209-5C：https://www.jedec.org/standards-documents/docs/jesd209-5c
- Wikipedia LPDDR：https://en.wikipedia.org/wiki/LPDDR
- Synopsys LPDDR5X 博客：https://www.synopsys.com/blogs/chip-design/lpddr5x-specification-memory-design.html
- 三星 LPDDR 产品总页（LPDDR6 x24/16–128Gb/357·518·1295 FBGA；LPDDR5X x64/x32；SOCAMM2/LPCAMM2）：https://semiconductor.samsung.com/dram/lpddr/
- 三星 LPDDR6 × Snapdragon（16GB、10.7Gbps、114 GB/s）：https://semiconductor.samsung.com/news-events/tech-blog/samsung-lpddr6-advancing-low-power-memory-for-ai-with-industry-first-validation-on-qualcomm-technologies-new-snapdragon-platform/
- 三星 CES 2026 LPDDR6：https://news.samsungsemiconductor.com/global/ces-innovation-awards-2026-honoree-lpddr6-worlds-first-next-gen-lpddr-optimized-for-high-performing-on-device-ai/
- SK hynix 1c LPDDR6：https://news.skhynix.com/en/1c-lpddr6-development-2026/
- ISSCC 2026 LPDDR6 综述（MoreThanMoore）：https://morethanmoore.substack.com/p/lpddr6-samsungsk-hynix-at-isscc-2026
- CXMT LPDDR6 量产（16GB、1295-ball PoP、12800）：https://www.cxmt.com/en/news/info_21.html
- 小米 XRING O3（4×24-bit、113.8 GB/s）：https://xenospectrum.com/en/xiaomi-xring-o3-lpddr6-performance/
- 三星 LPDDR6 料号（128Gbit 1295-ball）：https://www.puris.net/archives/12646
- Micron LPDDR5X 页：https://www.micron.com/products/memory/lpddr-components/lpddr5x
- Micron 非二进制 LPDDR5X：https://www.micron.com/about/blog/memory/dram/micron-broadens-lpddr5x-options-for-next-generation-edge-designs
- Micron FQ3'26 新闻稿：https://investors.micron.com/news/press-release/2026/Micron-Technology-Inc--Reports-Record-Results-for-the-Third-Quarter-of-Fiscal-2026/default.aspx
- Micron FY26 结果（6-channel LPDDR5X，二手）：https://www.unite.ai/micron-posts-record-revenue-and-earnings-for-fiscal-2026/
- 三星 LPDDR5X-PIM（Hot Chips 2026，二手）：https://www.techtimes.com/articles/325678/20260826/samsung-moves-ai-compute-dram-drop-memory-chip-triples-inference-speed.htm
- JEDEC SOCAMM2 JESD328：https://www.jedec.org/standards-documents/docs/jesd328 ；RCA 附录：https://www.jedec.org/standards-documents/docs/jesd328-j0-rca
- 三星 SOCAMM2（153.6 GB/s/模组）：https://semiconductor.samsung.com/dram/module/socamm2/
- SK hynix 192GB SOCAMM2：https://news.skhynix.com/en/mass-production-socamm2-192gb/
- Micron SOCAMM2：https://www.micron.com/products/memory/lpddr-modules/socamm
- Micron GTC'26（HBM4 36GB 12H >11Gb/s >2.8TB/s；SOCAMM2 48–256GB）：https://investors.micron.com/news/press-release/2026/Micron-in-High-Volume-Production-of-HBM4-Designed-for-NVIDIA-Vera-Rubin-PCIe-Gen6-SSD-and-SOCAMM2-03-16-2026/default.aspx
- Tom's Hardware Vera：https://www.tomshardware.com/pc-components/cpus/nvidia-spills-the-beans-on-vera-cpu-spec-benchmarks-revealed-olympus-architecture-detailed-and-more/2
- NVIDIA Grace 数据手册：https://dam-cdn.nvd.orangelogic.com/AssetLink/nf681l40ormkgaopeqq4e0l4nj25qv5d.pdf ；Chips&Cheese：https://chipsandcheese.com/p/grace-hopper-nvidias-halfway-apu
- Micron LPCAMM2 简介：https://assets.micron.com/adobe/assets/urn:aaid:aem:4e076108-df95-4c2d-8785-06c30049afb5/original/as/lpddr5x-camm2-technical-brief.pdf
- LPDDR6 CAMM2（TechPowerUp 引 JEDEC 讲稿）：https://www.techpowerup.com/322754/lpddr6-lpcamm2-pictured-and-detailed-courtesy-of-jedec
- Intel Crescent Island（Tom's Hardware）：https://www.tomshardware.com/pc-components/gpus/hot-chips-2026-intel-dives-deep-on-crescent-island-ai-accelerator-larger-caches-and-deeper-xmx-engines-target-maximum-ai-flops-per-watt
- JEDEC HBM3 新闻稿：https://www.jedec.org/news/pressreleases/jedec-publishes-hbm3-update-high-bandwidth-memory-hbm-standard ；JESD238B.01：https://www.jedec.org/standards-documents/docs/jesd238b01
- JEDEC HBM4 新闻稿：https://www.jedec.org/news/pressreleases/jedec%C2%AE-and-industry-leaders-collaborate-release-jesd270-4-hbm4-standard-advancing ；JESD270-4A：https://www.jedec.org/standards-documents/docs/jesd270-4a
- JEDEC SPHBM4：https://www.jedec.org/news/pressreleases/new-jedec%C2%AE-sphbm4-standard-enables-hbm4-class-bandwidth-organic-substrates
- 三星 HBM 产品页（HBM3/3E/4/4E）：https://semiconductor.samsung.com/dram/hbm/
- 三星 36GB HBM3E 12H：https://news.samsung.com/global/samsung-develops-industry-first-36gb-hbm3e-12h-dram
- Micron HBM3E：https://www.micron.com/products/memory/hbm/hbm3e
- SK hynix 12H HBM3E 量产：https://news.skhynix.com/en/sk-hynix-begins-volume-production-of-the-world-first-12-layer-hbm3e/
- SK hynix 16H HBM3E：https://www.storagenewsletter.com/2024/11/13/sk-ai-summit-sk-hynix-introduces-1st-16-high-hbm3e/
- SK hynix HBM4：https://news.skhynix.com/en/sk-hynix-completes-worlds-first-hbm4-development-and-readies-mass-production/
- 三星 HBM4E 样品：https://news.samsungsemiconductor.com/global/samsung-electronics-begins-shipment-of-industry-first-hbm4e-samples/
- SK hynix HBM4E 样品（TheElec）：https://www.thelec.net/news/articleView.html?idxno=11457
- TSMC/GUC C-HBM4E（Tom's）：https://www.tomshardware.com/pc-components/dram/hbm-undergoes-major-architectural-shakeup-as-tsmc-and-guc-detail-hbm4-hbm4e-and-c-hbm4e-3nm-base-dies-to-enable-2-5x-performance-boost-with-speeds-of-up-to-12-8gt-s-by-2027
- NVIDIA Vera Rubin NVL72：https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/ ；HGX：https://www.nvidia.com/en-us/data-center/hgx/ ；H200：https://www.nvidia.com/en-us/data-center/h200/
- Rubin 带宽下调（TechPowerUp）：https://www.techpowerup.com/346983/nvidia-lowers-hbm4-specs-for-vera-rubin-vr200-as-memory-suppliers-miss-22-tb-s-target
- AMD MI355X：https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html ；MI455X：https://www.amd.com/en/products/accelerators/instinct/mi400/mi455x.html
- Google TPU7x：https://docs.cloud.google.com/tpu/docs/tpu7x ；TPU 8（Tom's）：https://www.tomshardware.com/tech-industry/semiconductors/google-splits-its-tpu-into-two-chips-for-the-first-time-with-training-and-inference-variants

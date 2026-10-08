"""Validation tiers for core v2.

V0  invariants / metamorphic relations            (tests/test_core_metamorphic.py)
V1  params / active params / FLOPs vs releases     (tests/test_core_model.py)
V2  GPU-like trend sanity (caveated)               v2_trends()
V3  cross-tool: GenZ-LLM on identical abstract systems   v3_genz()
V4  silicon measurements                           n/a (no NPU exists — design guidance only)

V2 caveat: "H100-like" is OUR abstraction (128×128×16 MACs @1.83 GHz ≈ 989 dense
bf16 TFLOPS, HBM3 5 stacks @5.2 Gbps = 3.33 TB/s raw, mem_eff 0.8 「假设」).
Bands are roofline-plausibility bands informed by public serving reports
(e.g. vLLM / ClusterBid H100 Llama-3.1-8B: TTFT ≈ 92 ms for a 4 K prompt at
batch 1), not calibration targets.
"""

from __future__ import annotations

import json
from pathlib import Path

from .evaluate import evaluate
from .hardware import CHIP_100T, CHIP_H100_LIKE, Link
from .parallel import Layout
from .scenario import Scenario, Serving

H100_MEM = "hbm3_5s_8h16g_5200"
LPDDR_191 = "lpddr5x_4x64_8533_16g"
HBM_6600 = "hbm3e_8s_12h24g_9200"
REF = Path(__file__).resolve().parents[2] / "tests" / "data" / "genz_reference.json"


def _h100(model: str, **sv) -> Scenario:
    return Scenario(model=model, chip=CHIP_H100_LIKE, mem_id=H100_MEM, mem_eff=0.8, mapping="reconf",
                    link=Link(GBps=450.0, alpha_us=3.0), serving=Serving(**sv))


def v2_trends() -> list[dict]:
    rows = []

    def add(name, value, lo, hi, unit, note=""):
        rows.append({"check": name, "value": value, "lo": lo, "hi": hi, "unit": unit, "ok": lo <= value <= hi,
                     "note": note})

    b1 = evaluate(_h100("llama-3.1-8b", batch=1, ctx=1024))
    add("Llama-3.1-8B bf16 H100-like TPOT @B1", b1.tpot * 1e3, 4.5, 9.0, "ms",
        "bandwidth roofline 16 GB / 2.7 TB/s ≈ 6 ms")
    b16 = evaluate(_h100("llama-3.1-8b", batch=16, ctx=1024))
    add("TPOT(B16)/TPOT(B1) (bandwidth-bound regime)", b16.tpot / b1.tpot, 1.0, 1.5, "×")
    fp8 = evaluate(_h100("qwen3-8b-fp8", batch=1, ctx=1024))
    bf = evaluate(_h100("qwen3-8b", batch=1, ctx=1024))
    add("Qwen3-8B FP8/BF16 TPOT @B1", fp8.tpot / bf.tpot, 0.45, 0.65, "×", "weights halve; KV + head stay bf16")
    p = evaluate(_h100("llama-3.1-8b", phase="prefill", batch=1, prompt=4096))
    add("Llama-3.1-8B TTFT 4K prompt @B1", p.ttft * 1e3, 50.0, 200.0, "ms", "public H100 reports ≈ 92 ms")
    tp1 = evaluate(_h100("qwen3-32b", batch=1, ctx=1024))
    tp2 = evaluate(_h100("qwen3-32b", batch=1, ctx=1024).replace("layout", Layout(tp=2)))
    add("Qwen3-32B TP2 speed-up @B1", tp1.tpot / tp2.tpot, 1.5, 2.0, "×")
    big = evaluate(_h100("llama-3.1-8b", batch=256, ctx=1024))
    add("Llama-3.1-8B tok/s @B256 (compute regime)", big.throughput, 4000.0, 25000.0, "tok/s",
        "public H100 vLLM ≈ 9 k tok/s @B128")
    return rows


def v3_genz(path: Path = REF) -> list[dict]:
    ref = json.loads(path.read_text())
    out = []
    for r in ref["rows"]:
        mem = LPDDR_191 if r["mem_GBps"] < 1000 else HBM_6600
        for org in (("os", "reconf") if mem == HBM_6600 else ("os",)):
            ours = evaluate(Scenario(model="llama-3.1-8b", chip=CHIP_100T, mem_id=mem, mapping=org,
                                     serving=Serving(batch=r["batch"], ctx=r["ctx"]))).tpot * 1e3
            comparable = mem == LPDDR_191 or org == "reconf"
            out.append({**r, "mapping": org, "ours_ms": ours, "ratio": ours / r["tpot_ms"], "comparable": comparable})
    return out

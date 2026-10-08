#!/usr/bin/env python3
"""V3 cross-tool reference: run GenZ-LLM (pip genz-llm) on the same abstract
systems and store its decode latencies in tests/data/genz_reference.json.

Run inside a venv with GenZ installed, e.g.  /tmp/genzvenv/bin/python scripts/genz_reference.py
GenZ has no mapping / SRAM-port model (ideal compute at the stated FLOPS), so
the comparable point on our side is the 'reconf' organisation on HBM and any
mapping on LPDDR (memory-bound).
"""
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
from GenZ import decode_moddeling  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tests" / "data" / "genz_reference.json"
rows = []
for bw in (191, 6600):
    for ctx in (1024, 4096):
        for B in (1, 8, 64):
            sysd = {"real_values": True, "Flops": 100, "Memory_BW": bw, "Memory_size": 288, "ICN": 400, "ICN_LL": 3}
            out = decode_moddeling(model="meta-llama/Llama-3.1-8B", batch_size=B, input_tokens=ctx, output_tokens=1,
                                   Bb=1, system_name=sysd, bits="bf16")
            rows.append({"model": "llama-3.1-8b", "tflops": 100, "mem_GBps": bw, "ctx": ctx, "batch": B,
                         "tpot_ms": float(out["Latency"])})
            print(rows[-1])
OUT.write_text(json.dumps({"tool": "GenZ-LLM", "generated": time.strftime("%Y-%m-%d"),
                           "settings": "decode_moddeling(Bb=1, bits=bf16, output_tokens=1), real_values=True",
                           "rows": rows}, indent=1))

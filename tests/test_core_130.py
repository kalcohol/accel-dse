"""0.130 hostile review of the frontend and the newest catalog builders: sweeps of knobs the model does not read
(per workload kind / phase / EP), UI number rounding tiers, image-DiT labels, stale-result marking, non-JSON error
bodies; param counts pinned to the official checkpoints (HF index metadata / safetensors headers, 2026-10-15)."""
import json
import re
import shutil
import subprocess
from pathlib import Path

from accel_dse import serve
from accel_dse.core import catalog as C

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "accel_dse/web/app.js").read_text()


def _sweep(path, values, **sc):
    return serve.handle("POST", "/api/sweep", json.dumps({"path": path, "values": values, "scenario": sc}).encode())


def test_inapplicable_sweeps_rejected_applicable_vary():
    rejected = [("workload.frames", "qwen-image", {}), ("workload.frames", "esm2-650m", {}),
                ("workload.msa", "esm2-650m", {}), ("workload.samples", "openfold", {}),
                ("serving.ctx", "wan2.1-14b", {}), ("serving.prompt", "carbon-a-1.2b", {}),
                ("serving.prompt", "llama-3.1-8b", {}), ("serving.moe_skew", "qwen3-30b-a3b", {})]
    for path, m, extra in rejected:
        st, out = _sweep(path, [1, 2], model=m, **extra)
        assert st == 400 and out["error"].startswith(path), (path, m, out)
    vary = [("workload.frames", [17, 49], {"model": "wan2.1-1.3b"}), ("workload.steps", [4, 8], {"model": "flux.2-klein-4b"}),
            ("workload.msa", [64, 256], {"model": "openfold"}), ("workload.seq_len", [128, 256], {"model": "carbon-a-1.2b"}),
            ("serving.prompt", [512, 4096], {"model": "llama-3.1-8b", "serving": {"phase": "prefill"}}),
            ("serving.moe_skew", [1, 3], {"model": "deepseek-v4-flash", "layout": {"tp": 8, "ep": 8}})]
    for path, vals, sc in vary:
        st, out = _sweep(path, vals, **sc)
        assert st == 200, (path, out)
        r = out["rows"]
        assert (r[0]["tok_s"], r[0]["latency_ms"], r[0]["ttft_ms"]) != (r[1]["tok_s"], r[1]["latency_ms"], r[1]["ttft_ms"]), path


def test_ui_rounding_tiers():
    i, j = JS.index("function num("), JS.index("function put(e")
    if not shutil.which("node"):
        assert "round first" in JS[i:j]
        return
    code = JS[i:j] + ";for (const x of [99.96, 9.999, 0.9996, 9999.6, 12345.4, 0.012345, -99.96, 555.5, 1]) console.log(num(x));"
    out = subprocess.run(["node", "-e", code], capture_output=True, text=True, timeout=30).stdout.split()
    assert out == ["100", "10.0", "1.00", "10,000", "12,345", "0.012", "-100", "556", "1.00"], out


def test_ui_static_fixes():
    assert "function markEvalStale(on)" in JS and "markEvalStale(!!S.last)" in JS
    assert ".stale-res" in (ROOT / "accel_dse/web/style.css").read_text()
    assert "j = await r.json().catch(() => null);" in JS and "响应不是 JSON" in JS
    for ln in JS.splitlines():          # latency column labels go through clipZh (image DiT → 单张延迟)
        if "latL" in ln and "gen ?" in ln:
            assert "clipZh(" in ln, ln
    assert "unitZh(m) + '/s/卡'" in JS
    assert "!(p === 'workload.frames' && m.workload && m.workload.image)" in JS
    assert all("draft" in C.labels(C.get_model(m)) for m in ("deepseek-v4-flash", "llama-3.1-8b"))


def test_newest_builders_param_counts():
    # official totals: HF index metadata total_parameters / total_size (bf16 → /2) or summed safetensors headers
    pins = {"carbon-8b": 8255705088, "carbon-a-1.2b": 1153656832, "flux.2-klein-4b": 3875544576,
            "qwen-image": 40860802176 // 2, "qwen-image-2.1-turbo": 14230249472 // 2,
            "lfm2.5-24b-a2b": 23971585280 + 64 * 38}     # + the fp32 router expert_bias buffers (not nn.Parameters)
    for m, n in pins.items():
        s = C.get_model(m)
        assert s.params() == n == s.release_params, (m, s.params(), n)
    ling = C.get_model("ling-3.1-flash")
    assert ling.params() + ling.mtp_params() == 572120680768     # bytes of the snapshot = index total_size 573753281152

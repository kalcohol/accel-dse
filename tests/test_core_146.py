"""0.146: NPU design-insight report (/api/insight) — one-knob probes of the existing engine over a model set incl.
sparse / indexer attention (Step-5, DeepSeek-V3.2) and MXFP4-expert (gpt-oss, MiMo) models."""
from accel_dse import api as A
from accel_dse.core.budget import Budget
from accel_dse.core.energy import EnergyTable
from accel_dse.core.evaluate import evaluate
from accel_dse.core.insight import DEFAULT_MODELS, insight_report, knobs, model_scenario
from accel_dse.core.scenario import Scenario
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)

_B = Scenario().replace("layout.tp", 4).replace("serving.batch", 8)
_R = insight_report(_B, ["qwen3-8b", "gpt-oss-120b", "mimo-v2.6-flash", "step-5-preview"],
                    Budget(mm2_per_mib_sram=1.0, mm2_per_kmac=0.5, mm2_fixed=100.0), EnergyTable(pJ_mac=0.5, pJ_bit_dram=3.0))
_K = {k["key"]: k for k in _R["knobs"]}


def test_default_set_covers_sparse_and_mxfp4():
    feats = {r["model"]: r["features"] for r in insight_report(_B, DEFAULT_MODELS)["models"]}
    assert any("稀疏/索引注意力" in f for f in feats.values()) and any("MXFP4 专家" in f for f in feats.values())
    assert all("error" not in r for r in _R["models"])


def test_probes_match_direct_evaluation():
    """A knob's per-model number is exactly the engine's ratio for that one-variable change."""
    s = model_scenario(_B, "gpt-oss-120b")
    f = next(t for k, _, t, _ in knobs(_B) if k == "dram_count")
    a, b = evaluate(s.replace("serving.phase", "decode")), evaluate(f(s).replace("serving.phase", "decode"))
    row = next(x for x in _K["dram_count"]["models"] if x["model"] == "gpt-oss-120b")
    assert abs(row["tpot"] - a.tpot / b.tpot) < 1e-12 and abs(row["thr"] - b.per_card / a.per_card) < 1e-12


def test_mechanism_directions():
    # decode at batch 8 is DRAM-bound: more DRAM bandwidth speeds TPOT, a taller MAC array does not
    assert _K["dram_count"]["score"]["tpot"] > 1.1 and abs(_K["macs_x2"]["score"]["tpot"] - 1) < 0.01
    # as-released MXFP4 experts run W4A8 / W4A16 (dtypes.gemm_exec): a native W4A4 MAC buys nothing
    assert all(abs(x["tpot"] - 1) < 1e-9 and abs(x["ttft"] - 1) < 1e-9 for x in _K["native_mxfp4"]["models"])
    # native fp8 is present on the default chip → reported as the value of having it (≥ 1)
    assert _K["native_fp8"]["inverted"] and _K["native_fp8"]["score"]["ttft"] >= 1
    assert "dram_count" in _R["rank"]["tpot"] and "macs_x2" not in _R["rank"]["tpot"]


def test_area_and_power_only_from_user_inputs():
    assert _K["sram_x2"]["d_mm2"] == 64.0 * 1.0                     # ΔMiB × the user's mm²/MiB
    assert _K["macs_x2"]["d_mm2"] == 56 * 56 * 16 / 1024 * 0.5
    assert _K["dram_rate"]["d_mm2"] is None and _K["dram_rate"]["cost"] == "offdie"
    assert _R["energy_provided"] and any(x.get("d_J_tok") is not None for x in _K["dram_count"]["models"])
    r0 = insight_report(_B, ["qwen3-8b"])
    assert r0["area"] is None and all(k["d_mm2"] is None for k in r0["knobs"]) and r0["rank_per_mm2"]["tpot"] == []
    assert not r0["energy_provided"] and all(x.get("d_J_tok") is None for k in r0["knobs"] for x in k["models"])


def test_api_insight_validation():
    r = A.api_insight({"scenario": {"layout": {"tp": 2}}, "models": ["qwen3-8b", "deepseek-v3.2"]})
    assert {m["model"] for m in r["models"]} == {"qwen3-8b", "deepseek-v3.2"} and r["rank"].keys() == {"ttft", "tpot", "thr"}
    # 0.151: non-LLM domains are accepted when homogeneous; mixed domains and junk still fail
    assert A.api_insight({"models": ["wan2.1-1.3b"]})["domain"] == "gen"
    for bad in ({"models": ["wan2.1-1.3b", "qwen3-8b"]}, {"models": ["nope"]}, {"models": "qwen3-8b"},
                {"modls": ["qwen3-8b"]}, {"models": ["qwen3-8b"] * 9}):
        try:
            A.api_insight(bad)
        except A.ApiError:
            continue
        raise AssertionError(bad)


def test_fingerprints_unchanged():
    from tests.test_core_0100 import _fp
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"   # 0.437 re-baseline (umT5 RMSNorm; was cd25d689b7492f42) ← 0.209 re-baseline (STATE_SPLIT; was b47ec84b00fc0892) ← 0.202 re-baseline (GPU_WAVES_ONLY; was 6dca732e67c83a75) ← 0.201 re-baseline (GPU_AR_ONLY, = 0.198 value; was be3e00dcc13fcdda) ← 0.200 re-baseline (CUSTOM_AR; was 841b5c4e00c68d8c) ← 0.199 re-baseline (NCCL_AR_LAT; was 6dca732e67c83a75) ← 0.198 re-baseline (FUSE_QKV; was 5e72913c351849ac) ← 0.193 re-baseline (WIDE_WAVES, cta refhw; 0.192 8410808fd4f1e578) ← 0.192 re-baseline (state pinned first; 0.173 84df3be517a53ef4) (was 0a954915c5fb003f)
    assert _fp({}) == "19cac027be730bc4"   # 0.468 re-baseline (Qwen3.5 MoE expert mul; was 7d44cdadbf38cc9e) ← 0.467 re-baseline (Qwen3 head width; was 555d25ec2cbe1bfe) ← 0.466 re-baseline (Qwen2.5-3B and 1.5B RMSNorm; was 25e9cd50ab57455e) ← 0.464 re-baseline (Llama-2 and Yi-1.5 RMSNorm; was a9cc980add47b869) ← 0.462 re-baseline (Mistral-Large RMSNorm; was 1e15ae85430a04f3) ← 0.437 re-baseline (umT5 RMSNorm; was 99a72aef99a2e674) ← 0.210 re-baseline (pipe KV/state split; was 62fd08857bc4c055) ← 0.209 re-baseline (STATE_SPLIT; was b4db8aea9e5680b6) ← 0.202 re-baseline (GPU_WAVES_ONLY; was bc06528a82e19e2a) ← 0.201 re-baseline (GPU_AR_ONLY; was 7d5cdb465b922571) ← 0.200 re-baseline (CUSTOM_AR; was 727e05f1eee3e647) ← 0.199 re-baseline (NCCL_AR_LAT; was bc06528a82e19e2a) ← 0.198 re-baseline (FUSE_QKV; was eb634f2e8b513ffb) ← 0.193 re-baseline (WIDE_WAVES, cta refhw; 0.192 b8ca44d92e7589cd) ← 0.192 re-baseline (state pinned first; 0.173 1b755ac357ec6d2e) (was 53ee9e08165795c6)


def test_ui_tab_and_route_wired():
    from pathlib import Path
    root = Path(A.__file__).parent
    html, js = W.read(W.INDEX_HTML), W.read()          # 0.286: parsed
    assert "insight" in W.html_attr_values(html, "data-tab") and {"tab-insight", "ins-run"} <= W.html_ids(html)
    assert any(c["path"] == "/api/insight" and c["fn"] == "runInsight" for c in W.req_calls(js)) and "paintInsight" in W.functions(js)
    assert '"/api/insight"' in (root / "serve.py").read_text()

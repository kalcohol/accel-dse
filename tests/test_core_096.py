"""0.96: native KV-format attention (opt-in), per-engine SRAM port, sensitivity block (API / UI / CLI)."""
from dataclasses import replace
from accel_dse import api
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario, Serving

C = CHIPS["1P"]
HBM = "hbm3e_8s_12h24g_9200"
DEC = Serving(phase="decode", batch=64, ctx=4096)
FP8 = (("attn", "fp8"), ("mlp", "fp8"), ("act", "fp8"), ("kv", "fp8"))


def _ev(chip=C, ov=FP8, sv=DEC, mem=HBM):
    return evaluate(Scenario(model="qwen3-8b", chip=chip, mem_id=mem, serving=sv, formats_override=ov))


def test_native_kv_attention():
    base, nat = _ev(), _ev(replace(C, attn_fmt="kv"))
    assert nat.throughput > base.throughput * 1.2
    assert nat.stages[0].time.t_vector < 0.2 * base.stages[0].time.t_vector          # no per-step KV up-cast
    assert nat.stages[0].time.t_arr_attn < base.stages[0].time.t_arr_attn           # fp8 MAC rate
    few = replace(C, vector_lanes=C.cols * C.engines)                                # the 0.95 trap is gone
    assert _ev(replace(few, attn_fmt="kv")).throughput > _ev(few).throughput * 2
    bf = (("kv", "bf16"),)                                                          # bf16 KV: nothing to do
    assert _ev(replace(C, attn_fmt="kv"), bf).throughput == _ev(C, bf).throughput
    i4 = (("kv", "int4"),)                                                          # non-native KV format: falls back
    assert _ev(replace(C, attn_fmt="kv"), i4).throughput == _ev(C, i4).throughput
    try:
        replace(C, attn_fmt="fp8"); assert False
    except ValueError:
        pass
    assert "attn_fmt" not in str(Scenario(chip=C).to_dict())


def test_sram_port_per_engine():
    pe = C.port_Bpc / C.engines
    c1 = replace(C, sram_port_Bpc_per_engine=pe)
    assert abs(c1.port_Bpc - C.port_Bpc) < 1e-9
    assert abs(replace(c1, engines=16).port_Bpc - C.port_Bpc / 2) < 1e-9          # halves with the die
    try:
        replace(C, sram_port_Bpc=4096.0, sram_port_Bpc_per_engine=128.0); assert False
    except ValueError:
        pass
    for bad in (0, -1.0, float("inf")):
        try:
            replace(C, sram_port_Bpc_per_engine=bad); assert False
        except ValueError:
            pass


def test_sensitivity_block():
    body = {"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "mem_id": "lpddr6_8x96_10667_16g",
                                              "serving": {"phase": "decode", "batch": 64, "ctx": 4096}}}
    o = api.api_eval(body)
    s = o["sensitivity"]
    keys = [x["key"] for x in s["rows"]]
    assert keys == ["pipe", "exec_overlap", "mem_eff", "mem_eff", "vector_lanes"]     # 0.99: pipe first (现实), serial (下界)
    assert abs(s["base_tok_s"] - o["summary"]["tok_s"]) < 1e-9
    eff = {x["label"]: x["ratio"] for x in s["rows"] if x["key"] == "mem_eff"}
    assert any(abs(v - 0.5 / 0.7) < 1e-6 for v in eff.values())                   # DRAM-bound: ∝ efficiency
    assert "mem_eff" in s["hinges"]
    assert "sensitivity" not in api.api_eval({**body, "sensitivity": False})
    b2 = {**body, "scenario": {**body["scenario"], "mem_eff": 0.5, "exec_overlap": "serial"}}
    s2 = api.api_eval(b2)["sensitivity"]
    assert [x["key"] for x in s2["rows"]] == ["exec_overlap", "mem_eff", "vector_lanes"]   # 0.99: + stage (上界)          # no self-comparisons


def test_ui_and_cli_wiring():
    import pathlib
    root = pathlib.Path(api.__file__).parent
    js, html = (root / "web/app.js").read_text(), (root / "web/index.html").read_text()
    for t in ("sens-box", "sens-tbl", "c-attn_fmt", "c-sram_port_Bpc_per_engine"):
        assert f'id="{t}"' in html
    assert "renderSensitivity(r)" in js and "本结果取决于" in js
    from accel_dse import cli
    p = cli.build_parser() if hasattr(cli, "build_parser") else None
    src = (root / "cli.py").read_text()
    for f in ("--attn-fmt", "--sram-port-per-engine", "--no-sensitivity"):
        assert f in src

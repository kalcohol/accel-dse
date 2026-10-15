"""0.115: image DiT in the gen domain (one latent frame, no temporal attention) — Qwen-Image-2.1-Turbo, Qwen-Image,
FLUX.2-klein-4B at released dtype with exact parameter counts; adversarial audit of the LFM2 short-conv and Ling KDA
builders (work conservation, KV / state accounting, decode state vs sequence length)."""
import math
from accel_dse import api as A
from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, Shard, build_rank_ops
from accel_dse.core.scenario import Scenario


def _ev(sc):
    return evaluate(Scenario.from_dict(sc))


def _work(r):
    lay = r.scenario.layout
    n = lay.tp * lay.sp * lay.dp
    if r.workload is not None and r.workload.kind == "gen":        # CFG sequences are the micro-batch unit
        return sum(st.flops_u for st in r.stages) * n * r.microbatches
    return sum(st.flops_u for st in r.stages) * n / math.ceil(r.scenario.serving.batch / r.microbatches)


def test_image_dits_exact_params_and_structure():
    exp = {"qwen-image-2.1-turbo": (7_115_124_736, 32, 0), "qwen-image": (20_430_401_088, 60, 0),
           "flux.2-klein-4b": (3_875_544_576, 5, 20)}
    for mid, (p, n1, n2) in exp.items():
        m = get_model(mid)
        assert m.params() == m.release_params == p and m.domain == "gen", mid
        assert len(m.layers) == n1 + n2 and sum(1 for l in m.layers if l.fused_out) == n2
        assert dict(m.formats)["attn"].fmt == "bf16" and all(l.core.span == "full" for l in m.layers)
        assert m.workload.frames == 1 and m.workload.fps == 0 and m.workload.patch[0] == 1
    # Qwen-Image-2.1 is single-stream with shared weights over [text ‖ image]; Qwen-Image / FLUX.2 dual-stream
    assert {l.rows for l in get_model("qwen-image-2.1-turbo").layers[0].attn_linears} == {"tok"}
    assert {l.rows for l in get_model("qwen-image").layers[0].attn_linears} >= {"img", "txt"}


def test_image_workload_one_latent_frame():
    for mid, grid, steps, cfg in (("qwen-image-2.1-turbo", [1, 128, 128], 8, 1), ("qwen-image", [1, 83, 83], 50, 2),
                                  ("flux.2-klein-4b", [1, 64, 64], 4, 1)):
        g = _ev({"model": mid, "serving": {"batch": 1}}).summary()["gen"]
        w = g["workload"]
        assert w["grid"] == grid and w["image"] is True and w["video_s"] is None and w["attention"] == "full"
        assert w["steps"] == steps and w["cfg"] == cfg and w["seq_tokens"] == grid[1] * grid[2] + w["joint_text"]
        assert not any("帧数" in x for x in g.get("warnings", []))
        r5 = _ev({"model": mid, "workload": {"frames": 5}, "serving": {"batch": 1}}).summary()["gen"]
        assert r5["workload"]["frames"] == 1 and r5["tflop_per_request"] == g["tflop_per_request"]
    # video models carry no image key (summaries / fingerprints unchanged)
    assert "image" not in _ev({"model": "wan2.1-1.3b"}).summary()["gen"]["workload"]


def test_image_flops_closed_form():
    """Qwen-Image per forward: image tokens through the image stream, text through the text stream, joint attention."""
    m = get_model("qwen-image")
    g = _ev({"model": "qwen-image", "serving": {"batch": 1}}).summary()["gen"]
    N, T, d, L = 83 * 83, 256, 3072, 60
    per_tok = sum(l.k * l.n for l in m.layers[0].attn_linears + m.layers[0].ffn_linears if l.rows == "img")
    gemm = 2 * L * (N + T) * per_tok
    attn = 4 * L * (N + T) ** 2 * d
    assert abs(g["tflop_per_request"] * 1e12 / (100 * (gemm + attn)) - 1) < 0.01


def test_listing_and_api():
    ids = {d["id"]: d for d in list_models()}
    assert ids["qwen-image-2.1-turbo"]["family"] == "Qwen-Image" and ids["flux.2-klein-4b"]["family"] == "FLUX.2"
    assert ids["flux.2-klein-4b"]["provider_label"].startswith("Black Forest Labs")
    assert ids["qwen-image"]["workload"]["image"] is True and ids["wan2.1-14b"]["workload"]["image"] is False
    for mid in ("qwen-image-2.1-turbo", "flux.2-klein-4b"):
        s = A.api_eval({"scenario": {"model": mid, "layout": {"tp": 2}}, "sensitivity": False})["summary"]
        assert s["step_ms"] > 0 and s["domain"] == "gen"


def test_image_work_conserved():
    for mid, lays in (("qwen-image-2.1-turbo", ({"tp": 2}, {"sp": 2}, {"pp": 2}, {"tp": 3})),
                      ("flux.2-klein-4b", ({"tp": 2}, {"sp": 4}, {"pp": 5}, {"tp": 2, "sp": 2}))):
        w0 = _work(_ev({"model": mid, "serving": {"batch": 2}}))
        for lay in lays:
            assert abs(_work(_ev({"model": mid, "layout": lay, "serving": {"batch": 2}})) / w0 - 1) < 1e-6, (mid, lay)


def test_lfm_ling_work_conserved():
    for mid, base, lays in (("lfm2.5-24b-a2b", {"tp": 1}, ({"tp": 2, "etp": 2}, {"tp": 4, "ep": 4}, {"tp": 3, "ep": 3},
                                                          {"pp": 3}, {"dp": 2, "ep": 2}, {"tp": 8, "ep": 8, "pp": 5})),
                            ("ling-3.1-flash", {"tp": 1, "pp": 8}, ({"tp": 8, "ep": 8}, {"tp": 4, "ep": 4, "pp": 3},
                                                                    {"dp": 8, "ep": 8}))):
        for srv in ({"phase": "decode", "batch": 16, "ctx": 8192}, {"phase": "prefill", "batch": 2, "prompt": 2048}):
            w0 = _work(_ev({"model": mid, "layout": base, "serving": srv}))
            for lay in lays:
                w = _work(_ev({"model": mid, "layout": lay, "serving": srv}))
                assert abs(w / w0 - 1) < 1e-6, (mid, lay, srv["phase"], w / w0)


def _mem(mid, lay, b, ctx):
    r = _ev({"model": mid, "layout": lay, "serving": {"phase": "decode", "batch": b, "ctx": ctx}})
    return sum(s.mem.kv_total for s in r.stages), sum(s.mem.state_total for s in r.stages)


def test_kv_and_state_accounting_vs_seq_len():
    h = get_model("lfm2.5-24b-a2b").hidden
    ling_state = 63 * (64 * 128 * 128 + 3 * 64 * 128 * 3) * 4          # fp32 recurrent S + q/k/v conv (k − 1 = 3)
    for ctx in (1024, 32768, 131072):
        kv, st = _mem("lfm2.5-24b-a2b", {"tp": 1}, 8, ctx)
        assert kv == 10 * 2 * 8 * 64 * 2 * (ctx + 1) * 8 and st == 30 * h * 2 * 4 * 8
        kv, st = _mem("ling-3.1-flash", {"tp": 1, "pp": 8}, 4, ctx)
        assert kv == 9 * (512 + 64) * 2 * (ctx + 1) * 4 and st == ling_state * 4
    # Σ_rank state = global (ceil channel split bound); MLA latent replicated on every TP rank
    _, s1 = _mem("lfm2.5-24b-a2b", {"tp": 1}, 8, 4096)
    for n in (2, 4, 8):
        _, sn = _mem("lfm2.5-24b-a2b", {"tp": n, "ep": n}, 8, 4096)
        assert sn * n == s1
    kv1, s1 = _mem("ling-3.1-flash", {"tp": 1, "pp": 8}, 8, 4096)
    kv8, s8 = _mem("ling-3.1-flash", {"tp": 8, "ep": 8}, 8, 4096)
    assert s8 * 8 == s1 and kv8 == kv1


def test_state_traffic_once_per_step():
    for mid, lay in (("lfm2.5-24b-a2b", {"tp": 1}), ("ling-3.1-flash", {"tp": 1, "pp": 8})):
        m = get_model(mid)
        for ph in (Phase("decode", 4, 1, 1024), Phase("decode", 4, 1, 65536), Phase("prefill", 4, 2048, 0)):
            rw = sum(o.state_rw for li in range(len(m.layers)) for o in build_rank_ops(m, li, ph, Shard()))
            _, st = _mem(mid, lay, 4, 1024)
            assert rw == 2 * st, (mid, ph.kind, rw / st)


def test_fingerprints_unchanged():
    from tests.test_core_0100 import _fp
    assert _fp({"exec_overlap": "stage"}) == "e2176af2610c144c"
    assert _fp({}) == "392692e6937f69ef"

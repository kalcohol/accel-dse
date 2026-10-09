"""0.62 — VLM vision encoders (per released config + preprocessor, checked against the transformers / remote reference
modules), cost-balanced PP split (default; equal layer counts kept as pp_split = "layers"), host-link energy of
video offload reloads, per-layer-kind KV bytes of a PD hand-off with a cached prefix, DeepSeek-V4.1 aligner role."""

from __future__ import annotations

from accel_dse import api
from accel_dse.core.catalog import get_model
from accel_dse.core.disagg import kv_bytes_per_request
from accel_dse.core.energy import EnergyTable, energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.fabric import fabric_report
from accel_dse.core.hardware import Fabric
from accel_dse.core.memplan import cache_new_bytes
from accel_dse.core.parallel import Layout, plan_stages, plan_stages_balanced
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import expand_images, image_grid, vision_flops

HBM = "hbm3e_8s_12h24g_9200"

# (LLM tokens, GFLOP) per image at 448×448 / 640×480 / 1024×768: torch FlopCounterMode over the reference vision
# modules (transformers 5.19 qwen3_5 / qwen3_5_moe / qwen4_exp / glm5_next, Kimi + DeepSeek remote code), bf16,
# eager attention, built on meta then materialised (see docs/MODEL.md §20)
REF = {
    "qwen3.8-27b": ((196, 741.378097152), (300, 1196.8708608), (768, 3779.478945792)),
    "qwen3.8-flash-next": ((196, 736.753876992), (300, 1189.7929728), (768, 3761.359552512)),
    "qwen3.5-397b-a17b": ((196, 739.528409088), (300, 1194.0397056), (768, 3772.23118848)),
    "glm-5.3-flash": ((256, 1011.783565312), (414, 1739.127914496), (1036, 5365.561556992)),
    "kimi-k2.5": ((256, 1001.502277632), (414, 1749.82975488), (1036, 5661.563830272)),
    "kimi-k3": ((256, 1010.550439936), (414, 1807.86659328), (1036, 6234.383712256)),
    "deepseek-v4.1-flash": ((184, 1580.25740288), (206, 1693.43082496), (496, 5591.70625536)),
}
PARAMS = {"qwen3.8-27b": 460730096, "qwen3.8-flash-next": 448931056, "qwen3.5-397b-a17b": 456010480,
          "glm-5.3-flash": 563627008, "kimi-k2.5": 471143920, "kimi-k3": 447358976,
          "deepseek-v4.1-flash": 485268480}       # = release params_vision (DeepSeek: tower + aligner)


def _eval(scn, **kw):
    return api.api_eval({"scenario": scn, **kw})


def test_vision_tokens_flops_params_match_reference():
    for mid, rows in REF.items():
        m = get_model(mid)
        assert m.vision is not None and m.vision.params == PARAMS[mid] == m.vision_params, mid
        for (w, h), (tok, gf) in zip(((448, 448), (640, 480), (1024, 768)), rows):
            g = image_grid(m.vision, w, h)
            assert g.llm_tokens == tok, (mid, w, h, g.llm_tokens)
            assert abs(vision_flops(m.vision, g, 1) / 1e9 - gf) <= 1e-9 * gf, (mid, w, h, vision_flops(m.vision, g, 1))


def test_every_catalog_vlm_has_an_encoder_and_k27_matches_k25():
    from accel_dse.core.catalog import entries
    vlm = [e["id"] for e in entries() if e.get("domain") == "vlm"]
    assert len(vlm) >= 8 and all(get_model(i).vision is not None for i in vlm), vlm
    a, b = get_model("kimi-k2.5").vision, get_model("kimi-k2.7-code").vision
    assert a == b                               # same vision_config and image processor in the two releases


def test_deepseek_aligner_counted_as_vision():
    m = get_model("deepseek-v4.1-flash")
    pc = m.param_check()
    assert pc["release"] == 748494684544 and abs(pc["rel_err"]) < 5e-4


def test_images_default_off_and_hash_unchanged():
    assert Scenario(model="qwen3.8-27b").hash() == "a6463624c5748f5f"          # same as 0.61.4
    assert "images" not in Scenario(model="qwen3.8-27b").to_dict()["serving"]
    full = Scenario().to_dict(full=True)
    assert full["serving"]["images"] == 0 and full["pp_split"] == "cost" and "image_tokens" not in full["serving"]


def test_expand_images_idempotent_and_non_vlm_warns():
    m = get_model("glm-5.3-flash")
    s = Scenario(model="glm-5.3-flash", serving=Serving(phase="prefill", batch=2, prompt=1000, images=2))
    e1 = expand_images(s, m)
    assert e1.serving.prompt == 1000 + 2 * 1369 and e1.serving.image_tokens == 2 * 1369
    assert expand_images(e1, m) == e1
    r = evaluate(Scenario(model="qwen3-8b", serving=Serving(images=1)))
    assert any("serving.images 已忽略" in w for w in r.warnings)


def test_encoder_time_in_ttft_and_kv_grows():
    base = {"model": "qwen3.8-27b", "mem_id": HBM, "layout": {"tp": 2},
            "serving": {"phase": "prefill", "batch": 4, "prompt": 1024}}
    r0 = _eval(base)
    r1 = _eval({**base, "serving": {**base["serving"], "images": 1}})
    v = r1["summary"]["vision"]
    assert v["image_tokens"] == 1024 and v["text_prompt"] == 1024 and v["s"] > 0 and v["cards"] == 2
    assert r1["summary"]["ttft_ms"] > r0["summary"]["ttft_ms"] + v["s"] * 1e3 * 0.999
    assert r1["scenario"]["serving"]["prompt"] == 1024                    # echoed as entered
    d0 = _eval({**base, "serving": {"batch": 8, "ctx": 4096}})
    d1 = _eval({**base, "serving": {"batch": 8, "ctx": 4096, "images": 2}})
    assert d1["summary"]["dram_need_GiB"] > d0["summary"]["dram_need_GiB"]  # image KV + encoder weights
    assert d1["summary"]["vision"]["s"] == 0.0                              # no encoder at decode


def test_images_through_pd_and_sweep():
    sc = {"model": "qwen3.8-27b", "layout": {"tp": 2}, "mem_id": HBM,
          "serving": {"batch": 16, "prompt": 2048, "out_len": 256},
          "pd": {"enabled": True, "prefill_cards": 2, "decode_cards": 2}}
    a = _eval(sc)["pd"]
    b = _eval({**sc, "serving": {**sc["serving"], "images": 1}})["pd"]
    assert b["ttft_ms"] > a["ttft_ms"] and b["goodput_per_card"] < a["goodput_per_card"]
    sw = api.api_sweep({"scenario": {**sc, "pd": {"enabled": False}, "serving": {"phase": "prefill", "batch": 2,
                                                                                "prompt": 512, "images": 1}},
                        "path": "serving.images", "values": [0, 1, 4]})
    t = [r["ttft_ms"] for r in sw["rows"]]
    assert t[0] < t[1] < t[2]


def test_balanced_split_unit():
    one = (0.0, 0.0, 1.0, 0.0, 0.0)
    head = (0.0, 0.0, 4.0, 0.0, 0.0)
    z = (0.0,) * 5
    st = plan_stages_balanced([one] * 8, 2, z, head, z)
    assert [(s.first, s.last) for s in st] == [(0, 6), (6, 8)]             # 6 | 2 + head 4
    assert plan_stages_balanced([one] * 8, 2, z, z, z) == plan_stages(8, 2)  # homogeneous: equal counts kept
    # a capacity cap stops the move
    st = plan_stages_balanced([one] * 8, 2, z, head, z, mem=[1.0] * 8, mem_cap=5.0)
    assert max(s.last - s.first for s in st) <= 5


def test_cost_split_never_slower_and_layers_option():
    cases = [("qwen3.8-27b", Layout(pp=4), Serving(batch=8, ctx=4096)),
             ("glm-5.3-flash", Layout(pp=3, tp=2, ep=2), Serving(batch=16, ctx=4096)),
             ("esmfold", Layout(pp=4), Serving(batch=1)),
             ("protenix", Layout(pp=2), Serving(batch=1)),
             ("wan2.1-1.3b", Layout(pp=4), Serving(batch=1)),
             ("qwen3-8b", Layout(pp=2), Serving(phase="prefill", batch=4, prompt=2048))]
    for mid, lay, sv in cases:
        lc = evaluate(Scenario(model=mid, mem_id=HBM, layout=lay, serving=sv, pp_split="layers"))
        co = evaluate(Scenario(model=mid, mem_id=HBM, layout=lay, serving=sv))
        assert [s.layers for s in lc.stages] == [(p.first, p.last) for p in plan_stages(get_model(mid).n_layers, lay.pp)]
        t_l, t_c = max(s.time.total for s in lc.stages), max(s.time.total for s in co.stages)
        assert t_c <= t_l and (co.fits or not lc.fits), (mid, t_c, t_l)
    co = evaluate(Scenario(model="esmfold", mem_id=HBM, layout=Layout(pp=4), serving=Serving(batch=1)))
    lc = evaluate(Scenario(model="esmfold", mem_id=HBM, layout=Layout(pp=4), serving=Serving(batch=1), pp_split="layers"))
    assert co.latency < 0.7 * lc.latency                  # trunk vs folding head: the equal split was far off
    try:
        Scenario(pp_split="even")
        raise AssertionError
    except ValueError:
        pass


def test_fabric_report_counts_one_run_under_cost_split():
    s = Scenario(model="qwen3-32b", layout=Layout(pp=8, tp=2), node_cards=4, serving=Serving(batch=64),
                 fabric=Fabric(enabled=True, leaf_nodes=2))
    rows = {}
    for sp in ("layers", "cost"):
        sc = s.replace("pp_split", sp)
        rep = fabric_report(sc, evaluate(sc))
        rows[sp] = sum(x["count"] for x in rep["rows"] if x["kind"] != "p2p")
    assert rows["cost"] == rows["layers"]                 # collectives move between stages, never doubled


def test_offload_reload_host_energy():
    sc = {"model": "wan2.1-14b", "workload": {"placement": "offload"}}
    r = _eval(sc, energy={"pJ_mac": 0.5})
    pl = r["summary"]["gen"]["pipeline"]
    w = r["stages"][0]["mem"]["stored_w_GiB"] * 2 ** 30
    assert abs(pl["load_bytes"] - (w + pl["te_w"] + pl["vae_w"])) <= 1e-6 * pl["load_bytes"]
    e = r["energy"]
    assert abs(e["counts_per_unit"]["host"] - pl["load_bytes"] / e["units_per_window"]) <= 1e-6 * e["counts_per_unit"]["host"]
    assert "host_note" in e and "host" not in e["J_by_action"]
    e2 = _eval(sc, energy={"pJ_mac": 0.5, "pJ_bit_host": 10})["energy"]
    assert abs(e2["J_by_action"]["host"] - e["counts_per_unit"]["host"] * 10 * 8e-12) < 1e-12
    res = _eval({"model": "wan2.1-14b", "mem_id": HBM, "workload": {"placement": "resident"}}, energy={"pJ_mac": 0.5})
    assert "host" not in res["energy"]["counts_per_unit"]


def test_pd_handoff_bytes_per_layer_kind():
    for mid in ("qwen3-8b", "qwen3.8-27b", "gpt-oss-20b", "deepseek-v4.1-flash", "glm-5.3-flash"):
        m = get_model(mid)
        assert abs(cache_new_bytes(m, 8192, 0) / kv_bytes_per_request(m, 8192) - 1) < 1e-12
    m = get_model("qwen3-8b")                             # plain GQA: proportional, as before
    assert abs(cache_new_bytes(m, 8192, 4096) / (kv_bytes_per_request(m, 8192) / 2) - 1) < 1e-12
    m = get_model("qwen3.8-27b")                          # hybrid: the recurrent state is always sent whole
    full, half = kv_bytes_per_request(m, 8192), cache_new_bytes(m, 8192, 4096)
    st = kv_bytes_per_request(m, 2) - 2 * (kv_bytes_per_request(m, 4) - kv_bytes_per_request(m, 2)) / 2
    assert half > full / 2 and half >= st

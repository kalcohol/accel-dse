"""0.114: coverage refresh — LFM2.5-24B-A2B (+FP8, Liquid short-conv hybrid MoE) and Ling-3.1-flash (KDA + MLA hybrid,
FP8, 512 experts) at released dtype, exact parameter counts vs the safetensors headers."""
from accel_dse import api as A
from accel_dse.core.catalog import get_model, list_models


def test_lfm25_exact_params_and_structure():
    for mid in ("lfm2.5-24b-a2b", "lfm2.5-24b-a2b-fp8"):
        m = get_model(mid)
        assert m.params() == m.release_params == 23_971_587_712 and m.coverage == "full"
        conv = [l for l in m.layers if l.core.lin == "conv"]
        assert len(conv) == 30 and len(m.layers) == 40 and conv[0].core.conv_kernel == 3
        assert sum(1 for l in m.layers if l.ffn.kind == "moe") == 38 and m.tie_embeddings
    assert dict(get_model("lfm2.5-24b-a2b").formats)["expert"].fmt == "bf16"
    assert dict(get_model("lfm2.5-24b-a2b-fp8").formats)["expert"].fmt == "fp8"


def test_ling31_flash_exact_params_and_structure():
    m = get_model("ling-3.1-flash")
    assert m.params() == m.release_params == 563_946_483_520
    mtp = sum(l.params(m.hidden) for l in m.mtp_layers) + m.mtp_extra_params
    assert mtp == m.release_mtp_params == 8_174_197_248
    kinds = [l.core.kind for l in m.layers]
    assert kinds.count("linear") == 63 and kinds.count("mla") == 9 and kinds[7] == "mla"
    assert all(l.core.lin == "kda" for l in m.layers if l.core.kind == "linear")
    f = dict(m.formats)
    assert f["expert"].fmt == "fp8" and f["attn"].fmt == "fp8" and m.act_fmt == "fp8"


def test_new_models_listed_and_evaluate():
    ids = {d["id"]: d for d in list_models()}
    assert ids["lfm2.5-24b-a2b"]["provider_label"].startswith("Liquid") and ids["ling-3.1-flash"]["family"] == "Ling-3.1"
    for mid, lay in (("lfm2.5-24b-a2b", {"tp": 1}), ("ling-3.1-flash", {"tp": 8, "ep": 8})):
        for ph in ("decode", "prefill"):
            s = A.api_eval({"scenario": {"model": mid, "layout": lay, "serving": {"phase": ph, "batch": 8}},
                            "sensitivity": False})["summary"]
            assert s["step_ms"] > 0 and s["tok_s"] > 0


def test_short_conv_state_and_vector_work():
    """Conv layer: state = (kernel − 1) columns per channel; vector = conv MACs ×2 + two gate products per element."""
    from accel_dse.core.ir import Phase, Shard, build_rank_ops
    m = get_model("lfm2.5-24b-a2b")
    ops = build_rank_ops(m, 0, Phase("decode", 4, 1, 1024), Shard())
    st = next(o for o in ops if o.name == "state_update")
    h = m.hidden
    assert st.vec == 4 * h * 3 * 2 + 4 * h * 2
    assert st.state_rw == 2 * (h * 2 * 4) * 4      # state_fmt fp32 (as GDN / KDA conv state), read + write, 4 seqs


def test_fingerprints_unchanged():
    from tests.test_core_0100 import _fp
    assert _fp({"exec_overlap": "stage"}) == "c034bfe3e3a3afa2"
    assert _fp({}) == "e51c7ec17bbdb998"

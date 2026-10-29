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
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"   # 0.437 re-baseline (umT5 RMSNorm; was cd25d689b7492f42) ← 0.209 re-baseline (STATE_SPLIT; was b47ec84b00fc0892) ← 0.202 re-baseline (GPU_WAVES_ONLY; was 6dca732e67c83a75) ← 0.201 re-baseline (GPU_AR_ONLY, = 0.198 value; was be3e00dcc13fcdda) ← 0.200 re-baseline (CUSTOM_AR; was 841b5c4e00c68d8c) ← 0.199 re-baseline (NCCL_AR_LAT; was 6dca732e67c83a75) ← 0.198 re-baseline (FUSE_QKV; was 5e72913c351849ac) ← 0.193 re-baseline (WIDE_WAVES, cta refhw; 0.192 8410808fd4f1e578) ← 0.192 re-baseline (state pinned first; 0.173 84df3be517a53ef4) (was 0a954915c5fb003f)
    assert _fp({}) == "25e9cd50ab57455e"   # 0.464 re-baseline (Llama-2 and Yi-1.5 RMSNorm; was a9cc980add47b869) ← 0.462 re-baseline (Mistral-Large RMSNorm; was 1e15ae85430a04f3) ← 0.437 re-baseline (umT5 RMSNorm; was 99a72aef99a2e674) ← 0.210 re-baseline (pipe KV/state split; was 62fd08857bc4c055) ← 0.209 re-baseline (STATE_SPLIT; was b4db8aea9e5680b6) ← 0.202 re-baseline (GPU_WAVES_ONLY; was bc06528a82e19e2a) ← 0.201 re-baseline (GPU_AR_ONLY; was 7d5cdb465b922571) ← 0.200 re-baseline (CUSTOM_AR; was 727e05f1eee3e647) ← 0.199 re-baseline (NCCL_AR_LAT; was bc06528a82e19e2a) ← 0.198 re-baseline (FUSE_QKV; was eb634f2e8b513ffb) ← 0.193 re-baseline (WIDE_WAVES, cta refhw; 0.192 b8ca44d92e7589cd) ← 0.192 re-baseline (state pinned first; 0.173 1b755ac357ec6d2e) (was 53ee9e08165795c6)

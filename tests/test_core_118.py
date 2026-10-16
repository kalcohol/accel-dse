"""0.118: Open-Sora latent rule (partial 17-frame chunk), MiniMax-H3 decode geometry, protein audit (DAP conservation,
unfused triangle-attention warning), genomics windows shorter than the overlap, fingerprints."""
import dataclasses

from accel_dse.core.catalog import get_model
from accel_dse.core.domain import resolve_workload
from accel_dse.core.evaluate import evaluate
from accel_dse.core.parallel import Layout
from accel_dse.core.pipeline import component_data, h3_geometry
from accel_dse.core.scenario import Scenario, Workload

HBM = "hbm3e_8s_12h24g_9200"


def _lat(mid, **kw):
    return resolve_workload(get_model(mid), dataclasses.replace(Workload(), **kw)).info["latent"]


def test_opensora_partial_chunk_latents():
    # VideoAutoencoderPipeline.get_latent_size: 5 per full 17-frame chunk + ⌈r / 4⌉ for the remainder (front-padded)
    for F, lt in ((102, 30), (17, 5), (20, 6), (100, 29), (34, 10), (16, 4), (1, 1)):
        assert _lat("opensora-stdit3", frames=F)[0] == lt, F


def test_h3_decode_geometry_matches_reference():
    c = component_data("h3-vae")["config"]
    g = h3_geometry((37, 48, 84), c)                   # default 124 frames, 768 × 1344
    assert (g["clips"], g["clip_t"], len(g["hs"]), len(g["ws"])) == (7, 7, 4, 7)
    # diffusers _decode: num_chunks = (T + drop + pad) / 5 − 1, every clip 5 + 2 latent frames
    for T in (2, 7, 12, 13, 40):
        pad = (-(T + 3)) % 5
        assert h3_geometry((T, 16, 16), c)["clips"] == max(1, (T + 3 + pad) // 5 - 1)


def test_protein_dap_conserves_work_and_unfused_warning():
    for mid in ("esmfold", "alphafold2", "openfold", "boltz-1", "protenix"):
        t1 = evaluate(Scenario(model=mid, mem_id=HBM)).domain_summary()["tflop_per_request"]
        for sp in (2, 4):
            r = evaluate(Scenario(model=mid, mem_id=HBM, layout=Layout(sp=sp)))
            assert abs(r.domain_summary()["tflop_per_request"] / t1 - 1) < 1e-9, (mid, sp)
    w = lambda n, sp=1: [x for x in evaluate(Scenario(model="esmfold", workload=Workload(seq_len=n),
                                                      layout=Layout(sp=sp))).warnings if "三角注意力" in x]
    assert not w(512) and w(4096) and not w(4096, sp=8)       # DAP slices the logits too


def test_genomics_window_shorter_than_overlap():
    r = evaluate(Scenario(model="carbon-a-1.2b", workload=Workload(seq_len=100)))
    wl = r.domain_summary()["workload"]
    assert wl["overlap_tokens"] == 0 and wl["new_bp_per_window"] == 600 and any("重叠" in x for x in r.warnings)
    d = evaluate(Scenario(model="carbon-a-1.2b")).domain_summary()["workload"]
    assert (d["overlap_tokens"], d["new_bp_per_window"]) == (1024, 92160)


def test_fingerprints_unchanged():
    from tests.test_core_0100 import _fp
    assert _fp({"exec_overlap": "stage"}) == "8410808fd4f1e578"   # 0.192 re-baseline (state pinned first; 0.173 84df3be517a53ef4) (was 0a954915c5fb003f)
    assert _fp({}) == "b8ca44d92e7589cd"   # 0.192 re-baseline (state pinned first; 0.173 1b755ac357ec6d2e) (was 53ee9e08165795c6)

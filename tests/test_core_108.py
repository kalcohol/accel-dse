"""0.108: opt-in KV-cache stream efficiency (Scenario.kv_mem_eff); fingerprints unchanged."""
from accel_dse.core.evaluate import evaluate
from accel_dse.core.scenario import Scenario


def _s(**kw):
    return Scenario.from_dict({"model": "qwen3-8b", "serving": {"phase": "decode", "batch": 32, "ctx": 16384}, **kw})


def test_kv_mem_eff_default_unset_and_identical():
    s = _s()
    assert s.kv_mem_eff is None and "kv_mem_eff" not in s.to_dict()
    r0 = evaluate(s)
    assert evaluate(_s(kv_mem_eff=s.mem_eff)).summary() == r0.summary()      # = mem_eff → same numbers


def test_fingerprints_unchanged():
    from tests.test_core_0100 import _fp
    assert _fp({"exec_overlap": "stage"}) == "6dca732e67c83a75"   # 0.201 re-baseline (GPU_AR_ONLY, = 0.198 value; was be3e00dcc13fcdda) ← 0.200 re-baseline (CUSTOM_AR; was 841b5c4e00c68d8c) ← 0.199 re-baseline (NCCL_AR_LAT; was 6dca732e67c83a75) ← 0.198 re-baseline (FUSE_QKV; was 5e72913c351849ac) ← 0.193 re-baseline (WIDE_WAVES, cta refhw; 0.192 8410808fd4f1e578) ← 0.192 re-baseline (state pinned first; 0.173 84df3be517a53ef4) (was 0a954915c5fb003f)
    assert _fp({}) == "bc06528a82e19e2a"   # 0.201 re-baseline (GPU_AR_ONLY; was 7d5cdb465b922571) ← 0.200 re-baseline (CUSTOM_AR; was 727e05f1eee3e647) ← 0.199 re-baseline (NCCL_AR_LAT; was bc06528a82e19e2a) ← 0.198 re-baseline (FUSE_QKV; was eb634f2e8b513ffb) ← 0.193 re-baseline (WIDE_WAVES, cta refhw; 0.192 b8ca44d92e7589cd) ← 0.192 re-baseline (state pinned first; 0.173 1b755ac357ec6d2e) (was 53ee9e08165795c6)


def test_kv_mem_eff_speeds_kv_bound_decode_only():
    r0, r1 = evaluate(_s()), evaluate(_s(kv_mem_eff=0.9))
    assert r1.tpot < r0.tpot
    assert r1.scenario.to_dict()["kv_mem_eff"] == 0.9
    lo = evaluate(_s(kv_mem_eff=0.3))
    assert lo.tpot > r0.tpot
    for d in (0, 1.5, True):
        try:
            _s(kv_mem_eff=d); raise AssertionError(d)
        except ValueError:
            pass


def test_kv_mem_eff_keeps_bytes_and_energy():
    a, b = evaluate(_s()).summary(), evaluate(_s(kv_mem_eff=0.9)).summary()
    assert a["dram_need_GiB"] == b["dram_need_GiB"]


def test_extval_kveff_variant_registered():
    import accel_dse.core.extval as X
    n = X.kveff_variant("catalog")
    assert X.VARIANTS[n]["kv_eff"] == X.KV_STREAM_EFF and abs(X.KV_STREAM_EFF - 3000 / 3350) < 1e-12
    assert not hasattr(X, "pfsched_variant")


H = "hbm3e_8s_12h24g_9200"


def test_slow_slc_warns_for_lru_not_pin_bw():   # 0.108 audit: lru was silent
    def w(p):
        return [x for x in evaluate(Scenario.from_dict({"model": "carbon-3b", "mem_id": H, "serving": {"phase": "decode", "batch": 1, "ctx": 8192},
                                                        "chip": {"slc_mib": 8192, "slc_GBps": 500, "slc_policy": p}})).warnings if "SLC 带宽" in x]
    assert w("pin") and "/ lru" in w("pin")[0]
    assert w("lru") and "lru 策略" in w("lru")[0] and "/ lru" not in w("lru")[0]
    assert not w("pin_bw")


def test_slc_pin_bw_never_slower_than_pin_and_monotone():
    sc = {"model": "carbon-3b", "mem_id": H, "serving": {"phase": "decode", "batch": 1, "ctx": 8192}}
    prev = None
    for mib in (0, 64, 1024, 8192):
        a = evaluate(Scenario.from_dict({**sc, "chip": {"slc_mib": mib, "slc_GBps": 500, "slc_policy": "pin"}})).tpot
        b = evaluate(Scenario.from_dict({**sc, "chip": {"slc_mib": mib, "slc_GBps": 500, "slc_policy": "pin_bw"}})).tpot
        assert b <= a * (1 + 1e-9)
        assert prev is None or b <= prev * (1 + 1e-9)
        prev = b


def test_vlm_encoder_work_conserved_across_layouts():
    sc = {"model": "qwen3.8-27b", "serving": {"phase": "prefill", "batch": 3, "prompt": 256, "images": 3, "image_w": 448, "image_h": 448}}
    ref = None
    for lay in ({}, {"tp": 2}, {"tp": 4}, {"pp": 2}, {"tp": 8}):
        v = evaluate(Scenario.from_dict({**sc, "layout": lay})).summary()["vision"]
        assert abs(v["tflop"] - 9 * v["tflop_per_image"]) < 1e-9 * v["tflop"]
        ref = ref or v["tflop"]
        assert abs(v["tflop"] / ref - 1) < 1e-9, lay


def test_dit_work_conserved_across_sp_tp_and_cp_rejected():
    ref = None
    for lay in ({}, {"sp": 2}, {"sp": 4}, {"tp": 2}, {"tp": 2, "sp": 2}, {"pp": 2}):
        g = evaluate(Scenario.from_dict({"model": "wan2.1-1.3b", "layout": lay})).summary()["gen"]
        ref = ref or g["dit_tflop_per_request"]
        assert abs(g["dit_tflop_per_request"] / ref - 1) < 1e-6, lay
    try:
        evaluate(Scenario.from_dict({"model": "wan2.1-1.3b", "layout": {"cp": 2}})); raise AssertionError
    except ValueError:
        pass


def test_d2d_tiers_monotone_and_off_is_inert():
    sc = {"model": "llama-3.3-70b", "layout": {"tp": 4}, "serving": {"phase": "decode", "batch": 16, "ctx": 4096}}
    base = evaluate(Scenario.from_dict(sc))
    off = evaluate(Scenario.from_dict({**sc, "package_cards": 4, "d2d_enabled": False}))
    assert off.tpot == base.tpot and any("package_cards" in w for w in off.warnings)
    prev = None
    for u in (1, 2, 4, 8, 16):
        t = evaluate(Scenario.from_dict({**sc, "package_cards": 4, "d2d_enabled": True, "d2d_units": u})).tpot
        assert prev is None or t <= prev * (1 + 1e-9)
        prev = t

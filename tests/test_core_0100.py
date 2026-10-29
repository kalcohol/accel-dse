"""0.100: exec_overlap = "pipe" is the default (catalog NPU chips); stage stays selectable; refhw unchanged."""
import hashlib, json
from accel_dse.core.catalog import entries
from accel_dse.core.evaluate import evaluate
from accel_dse.core.scenario import Scenario


def _fp(extra_top: dict) -> str:
    h = hashlib.sha256()
    for e in entries()[:60]:
        for extra in ({}, {"chip": {"slc_mib": 256}}, {"chip": {"slc_mib": 256, "slc_policy": "lru"}}, {"layout": {"tp": 2}}):
            try:
                r = evaluate(Scenario.from_dict({"model": e["id"], **extra_top, **extra}))
                h.update(json.dumps(r.summary(), sort_keys=True, default=str).encode())
            except Exception as ex:  # noqa: BLE001
                h.update(type(ex).__name__.encode())
    return h.hexdigest()[:16]


def test_default_is_pipe_and_stage_selectable():
    s = Scenario(model="qwen3-8b")
    assert s.exec_overlap == "pipe" and "exec_overlap" not in s.to_dict()
    st = Scenario.from_dict({"model": "qwen3-8b", "exec_overlap": "stage"})
    assert st.to_dict()["exec_overlap"] == "stage" and st.hash() != s.hash()
    assert s.hash() == "723894c17df5d7bb"           # partial-dict hash unchanged (the scenario dict did not change)


def test_fingerprints():
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"   # 0.437 re-baseline (umT5 RMSNorm; was cd25d689b7492f42) ← 0.209 re-baseline (STATE_SPLIT; was b47ec84b00fc0892) ← 0.202 re-baseline (GPU_WAVES_ONLY; was 6dca732e67c83a75) ← 0.201 re-baseline (GPU_AR_ONLY, = 0.198 value; was be3e00dcc13fcdda) ← 0.200 re-baseline (CUSTOM_AR; was 841b5c4e00c68d8c) ← 0.199 re-baseline (NCCL_AR_LAT; was 6dca732e67c83a75) ← 0.198 re-baseline (FUSE_QKV; was 5e72913c351849ac) ← 0.193 re-baseline (WIDE_WAVES, cta refhw; 0.192 8410808fd4f1e578) ← 0.192 re-baseline (state pinned first; 0.173 84df3be517a53ef4) (was 0a954915c5fb003f)     # the 0.99 default numbers, explicitly stage
    assert _fp({}) == "555d25ec2cbe1bfe"   # 0.466 re-baseline (Qwen2.5-3B and 1.5B RMSNorm; was 25e9cd50ab57455e) ← 0.464 re-baseline (Llama-2 and Yi-1.5 RMSNorm; was a9cc980add47b869) ← 0.462 re-baseline (Mistral-Large RMSNorm; was 1e15ae85430a04f3) ← 0.437 re-baseline (umT5 RMSNorm; was 99a72aef99a2e674) ← 0.210 re-baseline (pipe KV/state split; was 62fd08857bc4c055) ← 0.209 re-baseline (STATE_SPLIT; was b4db8aea9e5680b6) ← 0.202 re-baseline (GPU_WAVES_ONLY; was bc06528a82e19e2a) ← 0.201 re-baseline (GPU_AR_ONLY; was 7d5cdb465b922571) ← 0.200 re-baseline (CUSTOM_AR; was 727e05f1eee3e647) ← 0.199 re-baseline (NCCL_AR_LAT; was bc06528a82e19e2a) ← 0.198 re-baseline (FUSE_QKV; was eb634f2e8b513ffb) ← 0.193 re-baseline (WIDE_WAVES, cta refhw; 0.192 b8ca44d92e7589cd) ← 0.192 re-baseline (state pinned first; 0.173 1b755ac357ec6d2e) (was 53ee9e08165795c6)                            # 0.100 default (pipe)


def test_refhw_unchanged():
    import accel_dse.core.extval as X
    r = X.rows(X.DATA)[0]
    for v in ("catalog", "catalog_kernel"):
        assert X.scenario(r, v, phase="decode", batch=8, ctx=1024, out_len=8).exec_overlap != "pipe"

"""0.119: triangle-attention kernel choice for structure models (fused default unchanged / materialized / chunked /
reference), reference chunk table, workbench labels for image / genomics models."""
from pathlib import Path

from accel_dse.core.domain import tri_reference_chunk
from accel_dse.core.evaluate import evaluate
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import Scenario, Workload

HBM = "hbm3e_8s_12h24g_9200"
STRUCT = ("esmfold", "alphafold2", "openfold", "boltz-1", "protenix")


def _r(mid, lay=None, **wl):
    return evaluate(Scenario(model=mid, mem_id=HBM, workload=Workload(**wl), **({"layout": lay} if lay else {})))


def test_modes_same_work_more_memory_and_time():
    for mid in STRUCT:
        f = _r(mid, seq_len=1024)
        m = _r(mid, seq_len=1024, tri_attn="materialized")
        c = _r(mid, seq_len=1024, tri_attn="chunked", tri_chunk=16)
        tf = f.domain_summary()["tflop_per_request"]
        assert m.domain_summary()["tflop_per_request"] == tf == c.domain_summary()["tflop_per_request"], mid
        fm, mm, cm = (x.summary()["dram_need_GiB"] for x in (f, m, c))
        assert mm > fm and fm <= cm < mm, (mid, fm, cm, mm)
        lf, lm = f.domain_summary()["latency_s"], m.domain_summary()["latency_s"]
        assert lm >= lf * (1 - 1e-9), mid                  # extra DRAM passes (hidden when compute-bound on HBM)
        lp = lambda t: evaluate(Scenario(model=mid, workload=Workload(seq_len=1024, tri_attn=t))).domain_summary()["latency_s"]
        assert lp("materialized") > 1.2 * lp("fused"), mid  # LPDDR default: the logits passes bind
        assert f.workload.info.get("tri_attn") is None and m.workload.info["tri_attn"]["chunk"] == -1


def test_materialized_peak_is_the_logits():
    # ESMFold trunk: 4 heads; one tri_att's logits = N · N² · 4 heads · act bytes, sliced by DAP
    a = _r("esmfold", seq_len=2048).summary()["dram_need_GiB"]
    b = _r("esmfold", seq_len=2048, tri_attn="materialized").summary()["dram_need_GiB"]
    from accel_dse.core.catalog import get_model
    from accel_dse.core.dtypes import fmt
    lg = 2048 ** 3 * 4 * fmt(get_model("esmfold").act_fmt).bytes / 2**30
    assert lg <= b < lg + a + 2, (a, b, lg)              # the logits tensor becomes the activation peak
    d2 = _r("esmfold", Layout(sp=2), seq_len=2048, tri_attn="materialized").summary()["dram_need_GiB"]
    assert d2 < b * 0.6


def test_reference_chunk_table():
    assert tri_reference_chunk("esmfold", 512) == -1
    assert tri_reference_chunk("openfold", 4096) == tri_reference_chunk("alphafold2", 64) == 4
    assert (tri_reference_chunk("boltz-1", 384), tri_reference_chunk("boltz-1", 385)) == (512, 128)
    assert [tri_reference_chunk("protenix", n) for n in (1024, 1536, 2048, 2560, 4096)] == [-1, 512, 256, 128, 128]
    r = _r("boltz-1", seq_len=1024, tri_attn="reference")
    assert r.workload.info["tri_attn"] == {"mode": "reference", "chunk": 128, "reference_chunk": 128}
    assert _r("esmfold", seq_len=4096, tri_attn="reference").summary()["fits"] is False
    assert _r("esmfold", seq_len=4096, tri_attn="chunked").summary()["fits"] is True


def test_ignored_outside_structure_models_and_validation():
    r = evaluate(Scenario(model="qwen3-32b", workload=Workload(tri_attn="chunked")))
    assert any("tri_attn" in w for w in r.warnings)
    for bad in ({"tri_attn": "flash"}, {"tri_chunk": -1}):
        try:
            Workload(**bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    assert "tri_attn" not in Scenario(model="esmfold").to_dict().get("workload", {})


def test_workbench_labels():
    js = (Path(__file__).resolve().parent.parent / "accel_dse" / "web" / "app.js").read_text()
    for s in ("图像生成", "单张延迟", "w-tri_attn", "unitZh", "window: '窗口'", "DNA 自回归语言模型"):
        assert s in js, s
    assert "UNIT_ZH[m" not in js


def test_fingerprints_unchanged():
    from tests.test_core_0100 import _fp
    assert _fp({"exec_overlap": "stage"}) == "b47ec84b00fc0892"   # 0.202 re-baseline (GPU_WAVES_ONLY; was 6dca732e67c83a75) ← 0.201 re-baseline (GPU_AR_ONLY, = 0.198 value; was be3e00dcc13fcdda) ← 0.200 re-baseline (CUSTOM_AR; was 841b5c4e00c68d8c) ← 0.199 re-baseline (NCCL_AR_LAT; was 6dca732e67c83a75) ← 0.198 re-baseline (FUSE_QKV; was 5e72913c351849ac) ← 0.193 re-baseline (WIDE_WAVES, cta refhw; 0.192 8410808fd4f1e578) ← 0.192 re-baseline (state pinned first; 0.173 84df3be517a53ef4) (was 0a954915c5fb003f)
    assert _fp({}) == "b4db8aea9e5680b6"   # 0.202 re-baseline (GPU_WAVES_ONLY; was bc06528a82e19e2a) ← 0.201 re-baseline (GPU_AR_ONLY; was 7d5cdb465b922571) ← 0.200 re-baseline (CUSTOM_AR; was 727e05f1eee3e647) ← 0.199 re-baseline (NCCL_AR_LAT; was bc06528a82e19e2a) ← 0.198 re-baseline (FUSE_QKV; was eb634f2e8b513ffb) ← 0.193 re-baseline (WIDE_WAVES, cta refhw; 0.192 b8ca44d92e7589cd) ← 0.192 re-baseline (state pinned first; 0.173 1b755ac357ec6d2e) (was 53ee9e08165795c6)

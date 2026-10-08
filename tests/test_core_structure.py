"""0.43 — protein structure models on core v2: ESMFold, AlphaFold 2 (official JAX parameters), OpenFold, Boltz-1 and
Protenix.  Exact parameters from the released checkpoints (headers read over HTTP range requests), module cores
detected from tensor names, pair-grid FLOPs vs closed forms, recycle / diffusion-step / sample scaling, PP × DP only
layouts, API round trip and the AlphaFold 3 offline reason."""
from __future__ import annotations

from collections import Counter

from accel_dse import api
from accel_dse.core.catalog import get_model, labels, offline_entries
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import PairDims, _pair_core_ops, layer_repeat
from accel_dse.core.model import PairCore
from accel_dse.core.parallel import Layout, enumerate_layouts
from accel_dse.core.scenario import Scenario, Workload

HBM = "hbm3e_8s_12h24g_9200"
LP = "lpddr5x_4x64_8533_16g"
STRUCT = ("esmfold", "alphafold2", "openfold", "boltz-1", "protenix")


def _tf(mid: str, **wl) -> float:
    return evaluate(Scenario(model=mid, mem_id=HBM, workload=Workload(**wl))).domain_summary()["tflop_per_request"]


def _cores(mid: str) -> dict[str, Counter]:
    out: dict[str, Counter] = {}
    for L in get_model(mid).layers:
        out.setdefault(L.stack, Counter(c.kind + ("·glob" if c.glob else "") for c in L.pair_cores))
    return {k: v for k, v in out.items() if v}


def test_params_exact_coverage_and_dtype():
    for mid in STRUCT:
        m = get_model(mid)
        lb = labels(m)
        assert m.is_pair and m.domain == "protein" and not m.kv_cache, mid
        assert m.params() == m.release_params, (mid, m.params(), m.release_params)   # every tensor accounted
        assert lb["is_pair"] and lb["workload"]["structure"] and lb["workload"]["recycles"] >= 1, mid
    assert labels(get_model("esmfold"))["coverage"] == "full"
    for mid in ("alphafold2", "openfold", "boltz-1", "protenix"):
        lb = labels(get_model(mid))
        assert lb["coverage"] == "partial" and any("MSA" in r for r in lb["coverage_reasons"]), mid
        assert lb["dtype"] == "W fp32 · A bf16", mid
    assert get_model("alphafold2").release_params == 93_237_338 == get_model("openfold").release_params
    assert round(get_model("boltz-1").release_params / 1e6, 3) == 606.382
    assert round(get_model("protenix").release_params / 1e6, 3) == 368.088


def test_alphafold2_jax_names_map_onto_the_openfold_structure():
    """The official haiku .npz (stacked layers, [in, out] weights) maps to exactly OpenFold's per-block GEMMs."""
    af, of = get_model("alphafold2"), get_model("openfold")
    sig = lambda L: (L.stack, L.repeat, L.iters, sorted((l.k, l.n, l.rows) for l in L.pair_linears),
                     sorted(map(repr, L.pair_cores)), L.misc_params)
    assert [sig(L) for L in af.layers] == [sig(L) for L in of.layers]
    w = af.workload
    assert (w.msa, w.xmsa, w.templates, w.recycles) == (508, 5120, 4, 4)    # model_1_ptm: 512 − 4 templates, 5120
    assert (of.workload.msa, of.workload.xmsa) == (512, 1024)


def test_module_cores_detected_per_block():
    af = _cores("alphafold2")
    assert af["evoformer"] == Counter({"trimul": 2, "tri_att": 2, "row_att": 1, "col_att": 1, "opm": 1})
    assert af["extra_msa"] == Counter({"trimul": 2, "tri_att": 2, "row_att": 1, "col_att·glob": 1, "opm": 1})
    assert af["template"] == Counter({"trimul": 2, "tri_att": 2}) and af["embed"] == Counter({"pt_att": 1})
    assert af["structure"] == Counter({"seq_att": 1})
    for mid in ("boltz-1", "protenix"):                       # AF3 family: MSA module uses pair-weighted averaging
        c = _cores(mid)
        assert c["msa"] == Counter({"trimul": 2, "tri_att": 2, "opm": 1, "pwa": 1}), mid
        assert c["pairformer"] == Counter({"trimul": 2, "tri_att": 2, "seq_att": 1}), mid
        assert c["diffusion"] == Counter({"seq_att": 1}) and c["diff_atom_enc"] == Counter({"local_att": 1}), mid
    assert _cores("esmfold")["trunk"] == Counter({"trimul": 2, "tri_att": 2, "seq_att": 1})
    ipa = next(c for L in get_model("alphafold2").layers if L.stack == "structure" for c in L.pair_cores)
    assert (ipa.heads, ipa.dim, ipa.v_dim) == (12, 16 + 12, 16 + 24 + 128)  # scalar + point qk; scalar + point + pair v
    pf = next(c for L in get_model("boltz-1").layers if L.stack == "pairformer" for c in L.pair_cores
              if c.kind == "seq_att")
    assert (pf.heads, pf.dim) == (16, 24)


def test_pair_core_flops_closed_forms():
    m = get_model("openfold")
    pd = PairDims(n=256, msa=128, xmsa=1024, tmpl=4)
    f = lambda c: sum(o.flops for o in _pair_core_ops(m, 0, c, pd, 1, 1, 1, "t") if o.kind != "vector")
    N, S = 256, 128
    assert f(PairCore("trimul", dim=128)) == 2 * N ** 3 * 128
    assert f(PairCore("tri_att", heads=4, dim=32)) == 2 * 2 * N ** 3 * 4 * 32
    assert f(PairCore("row_att", heads=8, dim=32, over="msa")) == 2 * 2 * S * N * N * 8 * 32
    assert f(PairCore("col_att", heads=8, dim=32, over="msa")) == 2 * 2 * N * S * S * 8 * 32
    assert f(PairCore("opm", dim=32, dim2=32, over="msa")) == 2 * (N * 32) * S * (N * 32)
    assert f(PairCore("col_att", heads=8, dim=8, over="xmsa", glob=True)) == 2 * 2 * N * 1024 * 8 * 8
    assert f(PairCore("tri_att", heads=4, dim=16, over="tmpl")) == 4 * (2 * 2 * N ** 3 * 4 * 16)   # × templates


def test_recycle_step_and_sample_scaling_linear():
    a1, a2, a4 = (_tf("openfold", recycles=r) for r in (1, 2, 4))
    assert abs((a4 - a2) - 2 * (a2 - a1)) < 1e-9 * a4 and a2 - a1 > 0.9 * a1     # trunk dominates, once-only heads
    e1, e2 = _tf("esmfold", recycles=1), _tf("esmfold", recycles=2)
    assert abs((2 * e1 - e2) / _tf("esm2-3b") - 1) < 0.01         # ESM-2 3B runs once per request, not per recycle
    b1, b2, b3 = (_tf("boltz-1", samples=s) for s in (1, 2, 3))
    assert abs((b3 - b1) - 2 * (b2 - b1)) < 1e-9 * b3 and b2 > b1
    s100, s200 = _tf("boltz-1", steps=100), _tf("boltz-1", steps=200)
    s100x2, s200x2 = _tf("boltz-1", steps=100, samples=2), _tf("boltz-1", steps=200, samples=2)
    assert s200x2 - s200 > s100x2 - s100 > 0                    # samples multiply the diffusion part (∝ steps) …
    assert abs((s200 - s100) - (_tf("boltz-1", steps=300) - s200)) < 1e-9 * s200   # … which is linear in steps
    m = get_model("boltz-1")
    r = evaluate(Scenario(model="boltz-1", mem_id=HBM, workload=Workload(steps=50, recycles=2)))
    assert r.workload.info["diff_steps"] == 50 and r.workload.info["recycles"] == 2
    ph = r.workload.pair
    diff = next(L for L in m.layers if L.repeat == "diff")
    rec = next(L for L in m.layers if L.repeat == "recycle")
    from accel_dse.core.ir import Phase
    p = Phase("full", 1, 1, 0, pair=ph)
    assert layer_repeat(diff, p) == 50 and layer_repeat(rec, p) == 2
    m1, m2 = _tf("alphafold2", msa=64), _tf("alphafold2", msa=128)
    assert m2 > m1 and _tf("esmfold", msa=128) == _tf("esmfold")                # single-sequence: MSA ignored


def test_layouts_pp_dp_only_and_errors():
    for mid in STRUCT:
        m = get_model(mid)
        lays = enumerate_layouts(4, len(m.layers), m.is_moe, full=True, pair=True)
        assert lays and all(l.tp == 1 and l.sp == 1 for l in lays), mid
    for bad in (Layout(tp=2), Layout(sp=2)):
        try:
            evaluate(Scenario(model="openfold", mem_id=HBM, layout=bad))
            raise AssertionError(bad)
        except ValueError as e:
            assert "DAP" in str(e)
    r1 = evaluate(Scenario(model="protenix", mem_id=LP))
    r2 = evaluate(Scenario(model="protenix", mem_id=LP, layout=Layout(pp=2)))
    assert r1.summary()["fits"] and r2.summary()["fits"]
    assert abs(r1.domain_summary()["tflop_per_request"] - r2.domain_summary()["tflop_per_request"]) < 1e-6


def test_slo_api_and_offline_alphafold3():
    r = evaluate(Scenario(model="alphafold2", mem_id=HBM, workload=Workload(fold_slo_s=1.0)))
    assert not r.slo_ok and r.domain_summary()["slo_ms"] == 1000.0
    assert evaluate(Scenario(model="alphafold2", mem_id=HBM)).slo_ok                  # 120 s default
    assert evaluate(Scenario(model="esm2-3b", mem_id=HBM)).domain_summary()["slo_ms"] == 1000.0
    out = api.api_eval({"scenario": {"model": "boltz-1", "mem_id": HBM, "workload": {"samples": 2, "msa": 256}}})
    w = out["summary"]["gen"]["workload"]
    assert w["samples"] == 2 and w["msa"] == 256 and w["diff_steps"] == 200 and w["atoms"] == 4096
    off = {o["id"]: o for o in offline_entries()}
    assert set(off) == {"alphafold3"} and "申请" in off["alphafold3"]["reason"]
    ms = {m["id"]: m for m in api.api_models()["models"]}
    assert all(ms[i]["is_pair"] for i in STRUCT) and not ms["esm2-3b"]["is_pair"]

"""0.66 — 参考硬件 catalog, external-measurement dataset / harness, exec_overlap modes."""
from dataclasses import replace

import pytest

from accel_dse.core import extval as X
from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.parallel import Layout
from accel_dse.core.refhw import REF_HW
from accel_dse.core.scenario import Scenario, Serving


def test_refhw_matches_datasheet_and_stays_out_of_design_presets():
    for h in REF_HW.values():
        assert abs(h.chip.peak_tflops_bf16 - h.peak_bf16_tflops) / h.peak_bf16_tflops < 1e-3
        fp8 = h.chip.formats.rate("fp8")
        assert (fp8 or 0) * h.peak_bf16_tflops == pytest.approx(h.peak_fp8_tflops, rel=2e-3)
        s = Scenario(chip=h.chip, mem_id=h.mem_id, mem_eff=1.0)
        from accel_dse.core.hardware import System
        sysm = System(h.chip, h.mem_id, 1.0)
        assert abs(sysm.dram_GBps - h.hbm_GBps) / h.hbm_GBps < 0.01
        assert sysm.dram_bytes >= 0.99 * h.hbm_GB * 1e9 * (0.9 if h.name.startswith("H200") else 1.0)
        assert h.link.GBps == pytest.approx(h.link_GBps_bidir / 2)
        assert h.url.startswith("https://") and "参考硬件" in h.chip.name
        assert h.chip.name not in CHIPS and s.chip.mac_eff == 1.0      # utilisation not tuned


def test_measurement_rows_are_cited_and_resolvable():
    rows = X.rows()
    assert len(rows) >= 100
    for r in rows:
        assert r["url"].startswith("https://") and r["value"] > 0 and r["hw"] in REF_HW
        assert r["metric"] in ("ttft_ms", "static_tok_s_gpu", "max_tok_s_total")
        get_model(r["model"])


def test_exec_overlap_default_unchanged_and_modes_ordered():
    h = REF_HW["H100-SXM"]
    base = Scenario(model="llama-3.3-70b", chip=h.chip, mem_id=h.mem_id, link=h.link, layout=Layout(tp=2),
                    mapping="reconf")
    assert "exec_overlap" not in base.to_dict()          # scenario hashes of earlier versions unchanged
    with pytest.raises(ValueError):
        replace(base, exec_overlap="bogus")
    for sv in (dict(phase="decode", batch=1, ctx=512), dict(phase="decode", batch=256, ctx=4096),
               dict(phase="prefill", batch=1, prompt=2048)):
        s0 = replace(base, serving=Serving(**sv))
        t = {m: evaluate(replace(s0, exec_overlap=m)) for m in ("stage", "class", "kernel", "serial")}
        key = "tpot" if sv["phase"] == "decode" else "ttft"
        v = {m: getattr(r, key) for m, r in t.items()}
        assert v["stage"] <= v["class"] * (1 + 1e-12) <= v["kernel"] * (1 + 1e-12) <= v["serial"] * (1 + 1e-12)
        st = t["class"].stages[0].time
        assert 0 <= st.t_arr_attn <= st.t_array and 0 <= st.t_dram_kv <= st.t_dram
        if st.t_link > 0:     # collectives are exposed in the kernel-serial modes
            assert v["class"] >= max(st.t_compute, st.t_dram) + st.t_link - 1e-12


def test_extval_harness_headline_bands():
    """Regression bands for the published-measurement comparison (docs/MODEL.md §25)."""
    t = X.table("catalog")
    s = X.summary(t)
    assert 1.0 < s["max_tok_s_total"]["gmean_ratio"] < 1.5        # stage overlap: optimistic throughput
    assert s["ttft_ms"]["gmean_ratio"] < 0.8                      # short prompts: per-forward overheads unmodelled
    k = X.summary(X.table("catalog_kernel"))
    assert k["max_tok_s_total"]["mean_abs_err_pct"] < s["max_tok_s_total"]["mean_abs_err_pct"]
    assert all(x["ratio"] > 0 for x in t)


def test_no_gqa_reuse_variant_only_scales_kv():
    m, n = X._spec("llama-3.1-8b", True), X._spec("llama-3.1-8b", False)
    assert n.id != m.id and len(n.layers) == len(m.layers)
    assert all(L.core.n_kv == L.core.n_q for L in n.layers)


def test_exec_overlap_ordering_with_pp_ep_fabric():
    from accel_dse.core.hardware import CHIPS, Fabric
    for m, lay in (("qwen3-30b-a3b", Layout(tp=4, ep=4, etp=1)), ("llama-3.3-70b", Layout(tp=4, pp=2))):
        for fab in (Fabric(), Fabric(enabled=True, overlap="ports", algo="auto_overlap")):
            for sv in (Serving(batch=32, ctx=4096), Serving(phase="prefill", batch=2, prompt=4096)):
                base = Scenario(model=m, chip=CHIPS["H100-like"], layout=lay, fabric=fab, serving=sv)
                v = [evaluate(replace(base, exec_overlap=o)) for o in ("stage", "class", "kernel", "serial")]
                k = "tpot" if sv.phase == "decode" else "ttft"
                x = [getattr(r, k) for r in v]
                assert all(x[i] <= x[i + 1] * (1 + 1e-12) for i in range(3))
                assert len({r.fits for r in v}) == 1

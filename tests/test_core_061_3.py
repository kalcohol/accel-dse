"""0.61.3 — round-3 audit fixes.

Reference counts: torch.utils.flop_counter on meta tensors of the published modules (OpenFold main, AF2 via the
OpenFold model_1 preset, boltz 0.4.1, protenix 0.5.0, Open-Sora v1.2.0 VAE + the SDXL VAE config) — see MODEL.md
§19.12.  The tool counts GEMM / attention MACs only, so the matches are within the per-model residuals listed there."""

from __future__ import annotations

import collections
import copy
import dataclasses

from accel_dse import api
from accel_dse.core.catalog import get_model
from accel_dse.core.domain import resolve_workload
from accel_dse.core.dtypes import FormatSupport
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, Shard, build_rank_ops, layer_repeat
from accel_dse.core.memplan import StageStorage, plan
from accel_dse.core.scenario import Scenario, Serving, Workload


def _tflop(mid, **kw):
    m = get_model(mid)
    wl = resolve_workload(m, Workload(**kw))
    ph = Phase("full", 1, wl.tokens, wl.ctx, frames=wl.frames, aux=wl.aux, pair=wl.pair)
    agg = collections.Counter()
    for li, L in enumerate(m.layers):
        rep = layer_repeat(L, ph)
        for o in build_rank_ops(m, li, ph, Shard()):
            if o.kind not in ("comm", "vector"):
                agg[L.stack] += o.flops * rep
    return sum(agg.values()) / 1e12, {k: v / 1e12 for k, v in agg.items()}, wl


def test_af2_openfold_template_rows_join_msa():
    """AF2 / OpenFold concatenate the T template torsion-angle rows to the MSA for every Evoformer block."""
    t, _, wl = _tflop("openfold", seq_len=384)
    assert wl.pair.msa == 512 + 4
    assert abs(t / 221.443 - 1) < 1e-3, t           # OpenFold main, FlopCounterMode, model_1_ptm defaults
    t, _, _ = _tflop("alphafold2", seq_len=384)
    assert abs(t / 245.739 - 1) < 1e-3, t
    t, _, _ = _tflop("alphafold2", seq_len=256)
    assert abs(t / 131.749 - 1) < 1e-3, t


def test_boltz_inference_cache_and_aliases():
    t, st, _ = _tflop("boltz-1", seq_len=384)
    assert "diff_cache" in st and st["diff_cache"] > 0
    assert abs(st["pairformer"] / 4 - 12.207) < 0.01 and abs(st["msa"] / 4 - 8.642) < 0.01
    assert abs(t - 153.115) < 0.05, t                # was 173.04 before the step cache (0.61.2)
    m = get_model("boltz-1")
    assert abs(m.release_params / 1e6 - 592.009) < 0.01     # output_projection_linear alias no longer counted twice


def test_protenix_per_sample_pair_bias():
    t, st, wl = _tflop("protenix", seq_len=384)
    assert wl.pair.samples == 5
    assert abs(st["pairformer"] / 4 - 12.207) < 0.01
    # GFLOP per step, 5 samples: the 24-block diffusion transformer matches the reference exactly (887.85); the rest of
    # the stack (conditioning, token projections) is the documented ≈ +0.6 % residual
    assert abs(st["diffusion"] / 200 * 1e3 - 985.73) < 0.5, st["diffusion"]
    assert abs(t - 280.425) < 0.05, t


def test_opensora_temporal_vae_block_resolution():
    r = api.api_eval({"scenario": {"model": "opensora-stdit3", "mem_id": "hbm3e_8s_12h24g_9200"}})
    vae = [p for p in r["summary"]["gen"]["pipeline"]["parts"] if p["role"] == "vae"][0]
    assert abs(vae["tflop"] - (932.686 + 206.83)) < 0.05, vae["tflop"]       # was 1157.86


def test_alias_dedup():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("ftc", pathlib.Path(__file__).parents[1] / "scripts" / "fetch_torch_ckpt.py")
    ftc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ftc)
    T = ftc.Tensor
    ts = {"b.w": T("float32", (4, 4), "0", 0), "a.w": T("float32", (4, 4), "0", 0), "c.w": T("float32", (4, 4), "0", 64)}
    kept, dropped = ftc.dedup_aliases(ts)
    assert sorted(kept) == ["a.w", "c.w"] and dropped == ["b.w"], (kept, dropped)


def test_cold_storage_not_pinned():
    """Looked-up embedding table / standby expert are stored but never pinned in SRAM or the SLC."""
    st = StageStorage(hot_w=1e9, expert_w=0.0, kv_per_seq=1e8, idx_per_seq=0.0, state_per_seq=0.0, cold_w=5e8)
    assert st.weights == 1.5e9
    mp = plan(st, 4, 2e9, 1e11, 1e6, slc_bytes=2e9, slc_policy="pin")
    assert mp.pinned_expert == 0 and mp.slc_expert == 0
    assert abs(mp.kv_sram + mp.slc_kv - 4e8) < 1        # all KV on chip once the hot weights are pinned
    s = Scenario(model="qwen3-8b", serving=Serving(batch=8))
    ch = dataclasses.replace(s.chip, slc_mib=16 * 1024.0, slc_policy="pin")
    r = evaluate(s.replace("chip", ch))
    assert r.stages[0].mem.slc_expert == 0 and r.tpot * 1e3 < 20.5      # 20.8 ms while the table took 1.2 GB


def test_format_rates_hash_int_float():
    x = Scenario(model="qwen3-8b").to_dict()["chip"]
    y = copy.deepcopy(x)
    y["formats"]["rates"] = [[n, int(r)] for n, r in y["formats"]["rates"]]
    assert api.scenario_from_body({"scenario": {"model": "qwen3-8b", "chip": y}}).hash() == Scenario(model="qwen3-8b").hash()
    assert FormatSupport(rates=(("bf16", 1),)).rates == (("bf16", 1.0),)


def test_best_batch_none_warns():
    r = api.api_eval({"scenario": {"model": "qwen3-8b"}, "best_batch": True})
    assert any(w.startswith("最佳 batch") for w in r["summary"]["warnings"])
    r = api.api_eval({"scenario": {"model": "qwen3-8b", "mem_id": "hbm3e_8s_12h24g_9200"}, "best_batch": True})
    assert not any(w.startswith("最佳 batch") for w in r["summary"]["warnings"])


def test_pd_energy_counts_restores():
    """kv_policy recompute: the re-prefills add MAC energy per output token (was not counted)."""
    from accel_dse.core.disagg import disagg_report
    from accel_dse.core.energy import EnergyTable
    from accel_dse.core.scenario import PDConfig
    T = EnergyTable(pJ_mac=0.5, pJ_bit_dram=5.0, idle_W=100.0)
    j = {}
    for pol in ("wait", "recompute"):
        scn = Scenario(model="qwen3-8b", mem_id="hbm3e_8s_12h24g_9200", serving=Serving(batch=64, prompt=4096, out_len=512),
                       node_cards=8, pd=PDConfig(enabled=True, prefill_cards=2, decode_cards=2, kv_policy=pol,
                                                 kv_capacity_GB=8.0, load=0.5, prompt_cv=0.7, out_cv=0.9))
        q = disagg_report(scn, energy=T)["queue"]
        j[pol] = (q["energy"]["pd"]["J_by_action"]["mac"], q["modes"]["pd"]["kv_cap"]["preempt_per_req"])
    (m0, _), (m1, pv) = j["wait"], j["recompute"]
    assert pv > 0 and m1 > m0 * (1 + 0.5 * pv), j


def test_pd_budget_scope_note():
    r = api.api_eval({"scenario": {"model": "qwen3-8b", "mem_id": "hbm3e_8s_12h24g_9200", "node_cards": 8,
                                   "pd": {"enabled": True, "prefill_cards": 2, "decode_cards": 2}},
                      "budget": {"cards": 4}})
    assert "PD prefill 池" in r["budget"]["scope_note"]

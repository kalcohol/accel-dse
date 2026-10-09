"""0.46 — diffusion samples split over DAP ranks (structure models); DiT weight FSDP (Wan --dit_fsdp); text encoder
on the host CPU (Wan --t5_cpu); Wan VAE tiling option."""
from __future__ import annotations

import argparse

from accel_dse import api, cli
from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, PairDims, Shard, build_rank_ops
from accel_dse.core.parallel import Layout
from accel_dse.core.pipeline import TILING, _spatial_tiles, stored_bytes
from accel_dse.core.scenario import Scenario, Workload

HBM = "hbm3e_8s_12h24g_9200"
LP = "lpddr5x_4x64_8533_16g"


def _diff_ops(mid: str, dap: int, samples: int, split: bool):
    m = get_model(mid)
    li = next(i for i, L in enumerate(m.layers) if L.stack == "diffusion")
    pd = PairDims(n=512, msa=0, atoms=4096, recycles=1, diff_steps=1, samples=samples, split=split)
    return m, m.layers[li], build_rank_ops(m, li, Phase("full", 1, 512, 0, pair=pd), Shard(1, 1, 1, 1, dap))


def test_sample_split_rows_flops_and_latency():
    for mid in ("protenix", "boltz-1"):
        m, L, one = _diff_ops(mid, 1, 5, False)
        u1 = sum(o.flops for o in one)
        for d in (2, 4, 8):
            _, _, rep = _diff_ops(mid, d, 5, False)
            _, _, spl = _diff_ops(mid, d, 5, True)
            # useful FLOPs conserved either way; the split rank does ⌈5/D⌉ samples of token work instead of 5
            for ops in (rep, spl):
                assert abs(sum(o.flops / o.replicated for o in ops) * d / u1 - 1) < 1e-9, (mid, d)
            assert sum(o.flops for o in spl) < sum(o.flops for o in rep)
            tok = [o for o in spl if o.kind == "gemm" and o.replicated != 1]
            s_loc = -(-5 // d)
            assert tok and all(abs(o.replicated - d * s_loc / 5) < 1e-12 for o in tok)
            assert any(o.m == s_loc * 512 for o in tok)          # token rows = ⌈S/D⌉ · N
            # no new communication: the comm ops are the DAP pair-bias gathers of the unsplit block
            assert [o.name for o in spl if o.kind == "comm"] == [o.name for o in rep if o.kind == "comm"]
    # end-to-end: Protenix (5 samples) DAP 8 3.3× faster than replicated; Boltz-1 default 1 sample unchanged
    a = evaluate(Scenario(model="protenix", mem_id=HBM, layout=Layout(sp=8), workload=Workload(sample_split=False)))
    b = evaluate(Scenario(model="protenix", mem_id=HBM, layout=Layout(sp=8)))
    # 0.61.3: Protenix's per-sample pair-bias projections are pair-grid work (DAP-split in both modes), so the
    # split / replicated gap narrowed from < 0.4 to 0.47
    assert b.latency < 0.5 * a.latency
    assert abs(a.domain_summary()["tflop_per_request"] - b.domain_summary()["tflop_per_request"]) < 1e-6
    c = evaluate(Scenario(model="boltz-1", mem_id=HBM, layout=Layout(sp=8), workload=Workload(sample_split=False)))
    d = evaluate(Scenario(model="boltz-1", mem_id=HBM, layout=Layout(sp=8)))
    assert c.latency == d.latency
    one = evaluate(Scenario(model="protenix", mem_id=HBM))
    assert one.latency == evaluate(Scenario(model="protenix", mem_id=HBM,
                                            workload=Workload(sample_split=False))).latency


def test_dit_fsdp_storage_link_and_noop():
    for mid, lay in (("wan2.1-14b", Layout(sp=4)), ("wan2.1-1.3b", Layout(sp=2, dp=2)), ("minimax-h3", Layout(sp=2))):
        g = lay.sp * lay.dp
        wl = dict(placement="resident")
        a = evaluate(Scenario(model=mid, mem_id=HBM, layout=lay, workload=Workload(**wl)))
        b = evaluate(Scenario(model=mid, mem_id=HBM, layout=lay, workload=Workload(dit_fsdp=True, **wl)))
        sa, sb = a.stages[0], b.stages[0]
        assert abs(sb.mem.stored_w * g / sa.mem.stored_w - 1) < 1e-12
        assert sb.mem.dram_need < sa.mem.dram_need
        assert abs(sb.time.link_bytes - sa.time.link_bytes - sa.mem.stored_w / g) < 1.0     # all-gather shard
        assert sb.dram["fsdp_gather"] > 0 and sb.time.t_link > sa.time.t_link
        assert b.latency >= a.latency and b.latency < 1.01 * a.latency                      # overlapped with compute
    # needs SP·DP > 1 and a video model; ignored with a warning otherwise
    for mid, lay in (("wan2.1-14b", Layout()), ("wan2.1-14b", Layout(tp=2)), ("alphafold2", Layout(sp=2))):
        a = evaluate(Scenario(model=mid, mem_id=HBM, layout=lay))
        b = evaluate(Scenario(model=mid, mem_id=HBM, layout=lay, workload=Workload(dit_fsdp=True)))
        assert a.latency == b.latency and a.stages[0].mem.dram_need == b.stages[0].mem.dram_need
        assert any("dit_fsdp" in w for w in b.warnings)
    # Wan2.2: the idle expert is sharded (stored) but not gathered
    a = evaluate(Scenario(model="wan2.2-a14b", mem_id=HBM, layout=Layout(sp=2), workload=Workload(placement="resident")))
    b = evaluate(Scenario(model="wan2.2-a14b", mem_id=HBM, layout=Layout(sp=2),
                          workload=Workload(placement="resident", dit_fsdp=True)))
    gathered = b.stages[0].time.link_bytes - a.stages[0].time.link_bytes
    assert 0.2 * a.stages[0].mem.stored_w < gathered < 0.3 * a.stages[0].mem.stored_w       # ≈ half of one expert


def test_te_cpu_placement():
    a = evaluate(Scenario(model="wan2.1-1.3b", mem_id=LP))
    b = evaluate(Scenario(model="wan2.1-1.3b", mem_id=LP, workload=Workload(te_cpu=True, host_TFLOPS=4.0)))
    pa, pb = a.pipeline, b.pipeline
    te_tf = sum(p["tflop"] for p in pb["parts"] if p["role"] == "text_encoder")
    assert abs(pb["te_s"] - te_tf / 4.0) < 1e-9 and pb["te_cpu"] and pb["te_card_w"] == 0.0
    assert abs(b.stages[0].mem.pipe_w - stored_bytes("wan-vae")) < 1.0
    assert abs((a.stages[0].mem.dram_need - b.stages[0].mem.dram_need) - stored_bytes("umt5-xxl")) < 2 ** 30
    assert abs(b.latency - (a.latency - pa["te_s"] + pb["te_s"])) < 1e-6
    # Wan2.1-14B on one 64 GiB card: --t5_cpu fits resident (Wan README's single-GPU recipe) — no host reload
    c = evaluate(Scenario(model="wan2.1-14b", mem_id=LP, workload=Workload(te_cpu=True)))
    assert c.fits and c.pipeline["place"] == "resident" and c.pipeline["load_s"] == 0.0
    # shard is moot with the encoder on the host
    d = evaluate(Scenario(model="minimax-h3", mem_id=LP, layout=Layout(pp=2, tp=2), workload=Workload(te_cpu=True)))
    assert d.pipeline["te_cards"] == 1 and d.pipeline["place"] in ("resident", "offload")


def test_wan_tiling_and_validation_cli():
    # Wan 720P: 90 × 160 latent, tiles 32 every 24 → 4 × 7 = 28 tiles
    t = _spatial_tiles((21, 90, 160), TILING["wan"])
    assert sum(len(rd) for rd in t) == 28
    r = evaluate(Scenario(model="wan2.1-14b", mem_id=HBM, workload=Workload(vae_tiling=True)))
    v = [p for p in r.pipeline["parts"] if p["role"] == "vae"][0]
    assert v["tiles"] == 28 and v["tiling"] and not any("vae_tiling" in w for w in r.warnings)
    for bad in (dict(dit_fsdp=1), dict(te_cpu="yes"), dict(sample_split=None), dict(host_TFLOPS=0),
                dict(host_TFLOPS=True)):
        try:
            Workload(**bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    p = argparse.ArgumentParser()
    cli._scenario_args(p)
    a = p.parse_args(["--model", "wan2.1-14b", "--sp", "2", "--dit-fsdp", "--te-cpu", "--host-TFLOPS", "3",
                      "--no-sample-split", "--vae-tiling"])
    wl = cli._body(a)["scenario"]["workload"]
    assert wl == {"dit_fsdp": True, "te_cpu": True, "host_TFLOPS": 3.0, "sample_split": False, "vae_tiling": True}
    out = api.api_eval({"scenario": {"model": "wan2.1-14b", "mem_id": LP, "layout": {"sp": 2},
                                     "workload": {"dit_fsdp": True, "te_cpu": True, "host_TFLOPS": 3}}})
    pl = out["summary"]["gen"]["pipeline"]
    assert pl["te_cpu"] and pl["host_TFLOPS"] == 3 and out["summary"]["fits"]

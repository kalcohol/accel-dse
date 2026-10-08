"""0.45 — structure models: DAP (dynamic axial parallelism, FastFold) on the SP layout axis; video pipeline component
placement (resident / FSDP-sharded text encoder / sequential CPU offload, auto); optional VAE tiling (CogVideoX / Mochi);
request-FLOP KPI over the whole replica."""
from __future__ import annotations

from accel_dse import api
from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, PairDims, Shard, build_rank_ops
from accel_dse.core.parallel import Layout, enumerate_layouts
from accel_dse.core.pipeline import _spatial_tiles, TILING, stored_bytes, te_layers, pipeline_for
from accel_dse.core.scenario import Scenario, Workload

HBM = "hbm3e_8s_12h24g_9200"
LP = "lpddr5x_4x64_8533_16g"
STRUCT = ("esmfold", "alphafold2", "openfold", "boltz-1", "protenix")


def _layer_ops(mid: str, kind: str, dap: int):
    m = get_model(mid)
    li = next(i for i, L in enumerate(m.layers) if any(c.kind == kind for c in L.pair_cores)
              and L.stack in ("evoformer", "pairformer"))
    pd = PairDims(n=512, msa=512, xmsa=1024, tmpl=4, atoms=4096, recycles=1, diff_steps=1, samples=1)
    ph = Phase("full", 1, 512, 0, pair=pd)
    return m, build_rank_ops(m, li, ph, Shard(1, 1, 1, 1, dap))


def test_dap_rows_flops_and_comm_closed_form():
    for mid in ("alphafold2", "boltz-1"):
        m, one = _layer_ops(mid, "trimul", 1)
        for d in (2, 4, 8):
            _, ops = _layer_ops(mid, "trimul", d)
            # useful FLOPs conserved: Σ rank FLOPs / replication × D = single-card FLOPs (N = 512 divisible by D)
            u = sum(o.flops / o.replicated for o in ops) * d
            assert abs(u / sum(o.flops for o in one) - 1) < 1e-9, (mid, d)
            # trimul: all-gather of one N × N × c operand → per-rank shard N²·c / D (bf16)
            for c in (c for L in m.layers if L.stack in ("evoformer", "pairformer") for c in L.pair_cores
                      if c.kind == "trimul"):
                g = [o for o in ops if o.kind == "comm" and o.name.endswith("dap_gather") and "trimul" in o.name]
                assert g and all(o.comm_kind == "allgather" and o.comm_group == d for o in g)
                assert abs(g[0].comm_bytes - 512 * 512 * c.dim / d * 2) < 1e-6
                break
            # tri_att: pair bias all-gather + one all-to-all transpose of the c_z = 128 pair rep
            t = [o for o in ops if o.kind == "comm" and "tri_att" in o.name and o.comm_kind == "alltoall"]
            assert t and abs(t[0].comm_bytes - 512 * 512 / d * 128 * 2) < 1e-6
        assert not [o for o in one if o.kind == "comm"]
    # MSA column attention: two all-to-all transposes of the MSA (c_m = 256)
    _, ops = _layer_ops("alphafold2", "col_att", 4)
    col = [o for o in ops if o.kind == "comm" and "col_att" in o.name]
    assert col
    assert abs(col[0].comm_bytes - 2 * 512 / 4 * 512 * 256 * 2) < 1e-6


def test_dap_layouts_latency_and_memory():
    for mid in STRUCT:
        m = get_model(mid)
        lays = enumerate_layouts(8, len(m.layers), False, full=True, pair=True)
        assert all(l.tp == 1 for l in lays) and Layout(1, 1, 1, 1, 1, 8) in lays
        r1 = evaluate(Scenario(model=mid, mem_id=HBM))
        r4 = evaluate(Scenario(model=mid, mem_id=HBM, layout=Layout(sp=4)))
        assert r4.latency < r1.latency, mid
        assert r4.stages[0].mem.dram_need <= r1.stages[0].mem.dram_need, mid
        assert r4.stages[0].time.link_bytes > 0 and r1.stages[0].time.link_bytes == 0
        # request FLOPs are a property of the request, not of the layout
        t1, t4 = (r.domain_summary()["tflop_per_request"] for r in (r1, r4))
        assert abs(t4 / t1 - 1) < 0.01, (mid, t1, t4)
    # trunk-dominated AF2 scales near-linearly; Protenix with its 200-step diffusion replicated much less
    # (0.46: the default splits the 5 samples over the DAP ranks — see test_core_046)
    af = [evaluate(Scenario(model="alphafold2", mem_id=HBM, layout=Layout(sp=d))).latency for d in (1, 8)]
    px = [evaluate(Scenario(model="protenix", mem_id=HBM, layout=Layout(sp=d),
                            workload=Workload(sample_split=False))).latency for d in (1, 8)]
    assert af[0] / af[1] > 6 and px[0] / px[1] < 3
    try:
        evaluate(Scenario(model="alphafold2", mem_id=HBM, layout=Layout(tp=2)))
        raise AssertionError("TP accepted")
    except ValueError as e:
        assert "DAP" in str(e)


def test_video_request_flops_layout_invariant():
    base = evaluate(Scenario(model="wan2.1-1.3b", mem_id=HBM)).domain_summary()["dit_tflop_per_request"]
    for lay in (Layout(sp=2), Layout(tp=2), Layout(dp=2), Layout(pp=2)):
        t = evaluate(Scenario(model="wan2.1-1.3b", mem_id=HBM, layout=lay)).domain_summary()["dit_tflop_per_request"]
        assert abs(t / base - 1) < 0.01, (lay.label, t, base)


def test_placement_auto_resident_shard_offload():
    # fits resident → auto = resident, identical to the explicit choice (0.44 numbers)
    a = evaluate(Scenario(model="wan2.1-1.3b", mem_id=LP))
    r = evaluate(Scenario(model="wan2.1-1.3b", mem_id=LP, workload=Workload(placement="resident")))
    assert a.pipeline["place"] == "resident" and a.latency == r.latency and a.pipeline["load_s"] == 0
    # Wan2.1-14B on 64 GiB LPDDR: resident does not fit, auto → offload (fits), reload = (TE + DiT + VAE) / host
    w = evaluate(Scenario(model="wan2.1-14b", mem_id=LP))
    p = w.pipeline
    assert p["place"] == "offload" and w.fits and any("auto" in x for x in w.warnings)
    dit_w = w.stages[0].mem.stored_w
    assert abs(p["load_s"] - (p["te_w"] + dit_w + p["vae_w"]) / 50e9) < 1e-9
    assert abs(w.latency - (p["te_s"] + p["denoise_s"] + p["decode_s"] + p["load_s"])) < 1e-6
    w2 = evaluate(Scenario(model="wan2.1-14b", mem_id=LP, workload=Workload(host_GBps=25.0)))
    assert abs(w2.pipeline["load_s"] / p["load_s"] - 2) < 1e-9
    # MiniMax-H3 on PP2·TP2: the FSDP-sharded Qwen3-VL text tower fits (te_w / 4 + 2 layers per card)
    h = evaluate(Scenario(model="minimax-h3", mem_id=LP, layout=Layout(pp=2, tp=2)))
    hp = h.pipeline
    nl, lw = te_layers(pipeline_for("minimax-h3").text[0])
    assert hp["place"] == "shard" and h.fits and hp["te_cards"] == 4 and nl == 64
    assert abs(hp["te_card_w"] - (stored_bytes("qwen3-vl-32b") / 4 + 2 * lw)) < 1
    assert hp["gather_s"] > 0 and hp["te_s"] >= hp["te_compute_s"]
    assert not evaluate(Scenario(model="minimax-h3", mem_id=LP, layout=Layout(pp=2, tp=2),
                                 workload=Workload(placement="resident"))).fits
    # shard on one card = resident
    s1 = evaluate(Scenario(model="wan2.1-1.3b", mem_id=LP, workload=Workload(placement="shard")))
    assert s1.pipeline["place"] == "resident" and s1.latency == a.latency
    # Wan2.2 A14B under offload parks the idle expert on the host: one 14B fp32 expert resident
    q = evaluate(Scenario(model="wan2.2-a14b", mem_id=LP, workload=Workload(placement="offload")))
    q0 = evaluate(Scenario(model="wan2.2-a14b", mem_id=LP, workload=Workload(placement="resident")))
    assert q.fits and not q0.fits and q.stages[0].mem.dram_need < q0.stages[0].mem.dram_need - 50 * 2**30
    for bad in ("gpu", "", "Auto"):
        try:
            Workload(placement=bad)
            raise AssertionError(bad)
        except ValueError:
            pass


def test_vae_tiling_option():
    # CogVideoX 480×720 → 60×90 latent: tiles 30×45 every 25 / 36 → 3 × 3 tiles, overlap (70·108)/(60·90) = 1.4
    t = _spatial_tiles((13, 60, 90), TILING["cog"])
    assert sum(n for _, n in t) == 9 and abs(sum(a * b * c * n for (a, b, c), n in t) / (13 * 60 * 90) - 1.4) < 1e-9
    for mid, tiles in (("cogvideox-5b", 9), ("mochi-1", 15)):
        a = evaluate(Scenario(model=mid, mem_id=HBM)).pipeline
        b = evaluate(Scenario(model=mid, mem_id=HBM, workload=Workload(vae_tiling=True))).pipeline
        va, vb = ([p for p in x["parts"] if p["role"] == "vae"][0] for x in (a, b))
        assert vb["tiles"] == tiles and vb["tiling"] and 1.2 < vb["tflop"] / va["tflop"] < vb["overlap"] + 0.05
        assert b["vae_act"] < a["vae_act"] and b["decode_s"] > a["decode_s"]
    w = evaluate(Scenario(model="ltx-video", mem_id=HBM, workload=Workload(vae_tiling=True)))   # 0.46: Wan tiles
    assert any("vae_tiling" in x for x in w.warnings)
    out = api.api_eval({"scenario": {"model": "mochi-1", "mem_id": HBM,
                                     "workload": {"vae_tiling": True, "placement": "offload"}}})
    pl = out["summary"]["gen"]["pipeline"]
    assert pl["place"] == "offload" and pl["load_s"] > 0

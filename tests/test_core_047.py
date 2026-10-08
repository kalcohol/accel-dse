"""0.47 — LTX-Video VAE tiling option, faithful diffusers tile loop, MiniMax-H3 decode as released (always tiled,
5+2-frame clips), multi-card tile-parallel VAE decode, cross-request overlap with a host-CPU text encoder."""
from __future__ import annotations

import argparse
import math

from accel_dse import api, cli
from accel_dse.core.evaluate import evaluate
from accel_dse.core.parallel import Layout
from accel_dse.core.pipeline import TILING, _h3_split, _spatial_tiles, component_data, h3_geometry, pipeline_for, vae_ops
from accel_dse.core.scenario import Scenario, Workload
from accel_dse.core.schedule import collective_seconds

HBM = "hbm3e_8s_12h24g_9200"
LP = "lpddr5x_4x64_8533_16g"


def _vae(r):
    return [p for p in r.pipeline["parts"] if p["role"] == "vae"][0]


def test_ltx_tiling_and_faithful_tile_loop():
    # LTX 704×512 → 22×16 latent; tile 512 px / stride 448 px at ×32 → 16 / 14 latent.  W > 16 triggers tiling;
    # then both axes step by 14 from 0 as diffusers does: H → 16 + a 2-row sliver, W → 16 + 8
    assert _spatial_tiles((21, 16, 22), TILING["ltx"]) == [[(21, 16, 16), (21, 16, 8), (21, 2, 16), (21, 2, 8)]]
    assert _spatial_tiles((21, 16, 16), TILING["ltx"]) == [[(21, 16, 16)]]          # fits one tile: untiled
    assert _spatial_tiles((13, 30, 90), TILING["cog"]) == [[(13, a, b) for a in (30, 5) for b in (45, 45, 18)]]
    a = evaluate(Scenario(model="ltx-video", mem_id=HBM))
    b = evaluate(Scenario(model="ltx-video", mem_id=HBM, workload=Workload(vae_tiling=True)))
    va, vb = _vae(a), _vae(b)
    ov = (16 * 16 + 16 * 8 + 2 * 16 + 2 * 8) / (16 * 22)
    assert vb["tiles"] == 4 and vb["tiling"] and abs(vb["overlap"] - ov) < 1e-12
    assert 1.0 < vb["tflop"] / va["tflop"] < ov + 0.05 and b.pipeline["vae_act"] < a.pipeline["vae_act"]
    assert not any("vae_tiling" in w for w in b.warnings)
    o = evaluate(Scenario(model="opensora-stdit3", mem_id=HBM, workload=Workload(vae_tiling=True)))
    assert any("没有空间分块" in w for w in o.warnings) and "tiles" not in _vae(o)


def test_h3_decode_as_released():
    cfg = component_data("h3-vae")["config"]
    # diffusers AutoencoderKLMiniMaxH3._split_tiles: 768 px → 4 tiles (256·4 − 64·3 = 832 ≥ 768), 1344 → 7
    assert _h3_split(768, 16) == [16] * 4 and _h3_split(1344, 16) == [16] * 7 and _h3_split(240, 16) == [15]
    assert _h3_split(256, 16) == [16] and _h3_split(448, 16) == [16] * 2          # 512 − 64 = 448 exactly
    g = h3_geometry((37, 48, 84), cfg)                    # 5·7 + 2 latent frames → 7 clips of 5 + 2
    assert g == {"clips": 7, "clip_t": 7, "hs": [16] * 4, "ws": [16] * 7}
    assert h3_geometry((12, 48, 84), cfg)["clips"] == 2   # 5·2 + 2
    _, _, vi = vae_ops(pipeline_for("minimax-h3").vae, (37, 48, 84), 124, 1, "bf16")
    L = 7 * 256 + 5
    (ops, n), = vi["groups"]
    assert n == 7 * 28 and vi["tiles"] == 196 and all(o.m == L for o in ops if o.kind == "gemm")
    wsum = sum(o.k * o.n for o in ops if o.kind == "gemm")
    tens = component_data("h3-vae")["tensors"]
    assert wsum == sum(math.prod(sh) for nm, sh in tens.items() if len(sh) >= 2 and nm.endswith("weight"))
    att = [o for o in ops if o.kind == "attn"]
    assert len(att) == 2 * cfg["decoder_num_layers"] and all(o.count == 32 for o in att)


def test_vae_parallel_tiles_gather_and_storage():
    base = evaluate(Scenario(model="minimax-h3", mem_id=HBM, layout=Layout(sp=4)))
    par = evaluate(Scenario(model="minimax-h3", mem_id=HBM, layout=Layout(sp=4), workload=Workload(vae_parallel=True)))
    v0, v1 = _vae(base), _vae(par)
    assert v1["par"] == 4 and v1["rank_tiles"] == 7 * 7 and v1["tflop"] == v0["tflop"]   # 28 tiles / clip → 7 each
    info = par.workload.info
    px = info["frames"] * info["height"] * info["width"] * 3 / math.prod(info["latent"])
    top = 7 * (7 * 16 * 16) * px * 2                                                     # bf16 decoded pixels
    bw, a = collective_seconds("allgather", top, 4, par.scenario.link)
    assert abs(v1["gather_s"] - 7 * (bw + a)) < 1e-12
    assert v1["single_s"] == v0["s"] and abs((v1["s"] - v1["gather_s"]) / (v0["s"] / 4) - 1) < 1e-9   # 49 of 196 tiles
    assert par.latency < base.latency and abs(par.pipeline["decode_s"] - base.pipeline["decode_s"] - (v1["s"] - v0["s"])) < 1e-9
    # VAE weights replicated on every card: PP2 puts them on stage 0 too
    p2 = evaluate(Scenario(model="minimax-h3", mem_id=HBM, layout=Layout(pp=2, sp=2), workload=Workload(vae_parallel=True)))
    q2 = evaluate(Scenario(model="minimax-h3", mem_id=HBM, layout=Layout(pp=2, sp=2)))
    vw = p2.pipeline["vae_w"]
    assert abs(p2.stages[0].mem.pipe_w - q2.stages[0].mem.pipe_w - vw) < 1 and _vae(p2)["par"] == 4
    # untiled decode / single card: ignored with a warning
    w = evaluate(Scenario(model="wan2.1-14b", mem_id=HBM, layout=Layout(sp=2), workload=Workload(vae_parallel=True)))
    assert any("vae_parallel" in x for x in w.warnings) and "par" not in _vae(w) and w.pipeline["vae_par"] == 1
    s = evaluate(Scenario(model="minimax-h3", mem_id=HBM, workload=Workload(vae_parallel=True)))
    assert any("vae_parallel" in x for x in s.warnings) and s.latency == evaluate(Scenario(model="minimax-h3", mem_id=HBM)).latency
    # Wan tiled at 720P: 28 tiles over 8 cards → slowest rank 4 tiles; Hunyuan rounds per temporal tile
    wt = evaluate(Scenario(model="wan2.1-14b", mem_id=HBM, layout=Layout(sp=8),
                           workload=Workload(vae_tiling=True, vae_parallel=True)))
    assert _vae(wt)["rank_tiles"] == 4 and _vae(wt)["s"] < _vae(wt)["single_s"] / 3


def test_overlap_host_encoder():
    kw = dict(model="wan2.1-14b", mem_id=HBM, layout=Layout(sp=8))
    a = evaluate(Scenario(**kw, workload=Workload(te_cpu=True)))
    b = evaluate(Scenario(**kw, workload=Workload(te_cpu=True, overlap=True)))
    te = b.pipeline["te_s"]
    assert b.latency == a.latency and abs(b.pipeline["period_s"] - (b.latency - te)) < 1e-9
    d = b.domain_summary()
    assert abs(d["requests_per_s"] - 1 / (b.latency - te)) < 1e-12 and d["latency_s"] == b.latency
    assert b.throughput > a.throughput
    slow = evaluate(Scenario(**kw, workload=Workload(te_cpu=True, overlap=True, host_TFLOPS=0.0005)))
    assert slow.pipeline["period_s"] == slow.pipeline["te_s"]                       # host-bound
    c = evaluate(Scenario(**kw, workload=Workload(overlap=True)))
    assert any("workload.overlap" in x for x in c.warnings) and "period_s" not in c.pipeline
    assert c.domain_summary()["period_s"] == c.latency


def test_validation_cli_api():
    for bad in (dict(vae_parallel=1), dict(overlap="yes")):
        try:
            Workload(**bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    p = argparse.ArgumentParser()
    cli._scenario_args(p)
    a = p.parse_args(["--model", "minimax-h3", "--sp", "2", "--vae-parallel", "--te-cpu", "--overlap"])
    assert cli._body(a)["scenario"]["workload"] == {"te_cpu": True, "vae_parallel": True, "overlap": True}
    out = api.api_eval({"scenario": {"model": "minimax-h3", "mem_id": HBM, "layout": {"sp": 2},
                                     "workload": {"vae_parallel": True, "te_cpu": True, "overlap": True}}})
    g = out["summary"]["gen"]
    assert g["pipeline"]["vae_par"] == 2 and g["pipeline"]["overlap"] and g["period_s"] < g["latency_s"]

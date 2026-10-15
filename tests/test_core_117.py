"""0.117: video VAE / latent audit — decoder geometry reproduces the requested clip for every conv family, VAE configs
agree with the catalog compression, reference resolution / frame rounding (floor to vae_s · patch; Wan frame snap)."""
import dataclasses

from accel_dse.core import pipeline as P
from accel_dse.core.catalog import get_model
from accel_dse.core.domain import resolve_workload
from accel_dse.core.scenario import Workload


def _rw(mid, **kw):
    return resolve_workload(get_model(mid), dataclasses.replace(Workload(), **kw))


def test_decoder_geometry_reproduces_clip():
    for mid, pl in P.PIPELINES.items():
        v = pl.vae
        if v.family not in P.FAMILIES:
            continue
        d = P.component_data(v.key)
        for kw in ({}, {"frames": 33, "height": 256, "width": 384}):
            if get_model(mid).workload.frames == 1:
                kw = {k: x for k, x in kw.items() if k != "frames"}
            info = _rw(mid, **kw).info
            _, _, fin = P._conv_walk(d["tensors"], "decoder.", P.FAMILIES[v.family], tuple(info["latent"]),
                                     d["dtype"], "bf16", 1, 1, "x")
            unpatch = d["config"].get("patch_size", 1) if v.family == "ltx" else 1       # LTX: 4×4 unpatchify
            assert (fin[0], fin[1] * unpatch, fin[2] * unpatch) == (info["frames"], info["height"], info["width"]), mid


def test_vae_configs_match_catalog_compression():
    for mid, pl in P.PIPELINES.items():
        c, w = P.component_data(pl.vae.key)["config"], get_model(mid).workload
        if "spatial_compression_ratio" in c:
            assert c["spatial_compression_ratio"] == w.vae_s, mid
        if "temporal_compression_ratio" in c and w.frames > 1:
            assert c["temporal_compression_ratio"] == w.vae_t, mid
        if "dim_mult" in c:                                   # Wan-VAE family: one ×2 per level after the first
            assert 2 ** (len(c["dim_mult"]) - 1) == w.vae_s, mid
            if w.frames > 1:
                assert 2 ** sum(c["temperal_downsample"]) == w.vae_t, mid


def test_reference_rounding():
    i = _rw("wan2.1-1.3b", frames=84, height=490, width=840).info
    assert (i["frames"], i["height"], i["width"], i["latent"]) == (85, 480, 832, [22, 60, 104])
    assert _rw("wan2.1-1.3b", frames=82).info["latent"][0] == 21
    assert _rw("hunyuanvideo", height=728, width=1288).info["grid"] == [33, 45, 80]
    assert _rw("ltx-video", height=500, width=700).info["latent"] == [21, 15, 21]
    assert _rw("mochi-1", height=488, width=856).info["grid"] == [28, 30, 53]
    assert _rw("cogvideox-5b", frames=50).info["latent"][0] == 13
    assert _rw("qwen-image-2.1-turbo", height=2040, width=2056).info["latent"] == [1, 126, 128]
    for mid in P.PIPELINES:                                   # release defaults need no rounding
        assert not [x for x in _rw(mid).warnings if "取整" in x or "4k+1" in x], mid


def test_fingerprints_unchanged():
    from tests.test_core_0100 import _fp
    assert _fp({"exec_overlap": "stage"}) == "e2176af2610c144c"
    assert _fp({}) == "392692e6937f69ef"

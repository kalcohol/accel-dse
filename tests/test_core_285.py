"""0.285: quantized vision tower (Qwen3.8-Flash-Next-FP4-Local: attention MXFP8 W8A8, MLP + merger W4A16 NVFP4) runs
at the release's formats and is stored at the release's bytes; every other vision tower unchanged."""
import dataclasses
import json
from pathlib import Path

from accel_dse.core.catalog import entries, get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario
from accel_dse.core.vision import image_grid, vision_ops

ROOT = Path(__file__).resolve().parents[1]
M = "qwen3.8-flash-next-fp4-local"


def test_release_vision_fmt_only_for_quantized_tower():
    has = {}
    for p in sorted((ROOT / "accel_dse/data/releases").glob("*.json")):
        d = json.loads(p.read_text())
        if "vision_fmt" in d:
            has[p.name] = d["vision_fmt"]
    assert list(has) == ["nvidia__Qwen3.8-Flash-Next-FP4-Local.json"]
    vf = has["nvidia__Qwen3.8-Flash-Next-FP4-Local.json"]
    assert vf["attn"]["fp8"]["params"] == 27 * (3456 * 1152 + 1152 * 1152) and vf["attn"]["fp8"]["bits"] == 8.25
    assert vf["mlp"]["nvfp4"]["params"] == 27 * 2 * 1152 * 4304 and vf["mlp"]["nvfp4"]["bits"] == 4.5
    assert vf["merger"]["nvfp4"]["params"] == 4608 * 4608 + 4608 * 2560
    assert list(vf["patch"]) == ["bf16"]


def test_tower_formats_and_bytes():
    v = get_model(M).vision
    fm = {k: (f, round(b, 3), a) for k, f, b, a in v.wfmt}
    assert fm == {"attn": ("fp8", 8.25, "fp8"), "mlp": ("nvfp4", 4.5, "bf16"), "merger": ("nvfp4", 4.5, "bf16"),
                  "patch": ("bf16", 16.0, "bf16")}
    rel = json.loads((ROOT / "accel_dse/data/releases/nvidia__Qwen3.8-Flash-Next-FP4-Local.json").read_text())
    vb = sum(x["bytes"] for x in rel["roles"]["vision"].values())
    assert abs(v.params * v.w_Bpp - vb) < 1
    ops = {o.name: o for o in vision_ops(v, image_grid(v, 1024, 1024), 1)}
    assert (ops["vis.0.qkv"].w_fmt, ops["vis.0.qkv"].a_fmt) == ("fp8", "fp8")
    assert (ops["vis.0.mlp_out"].w_fmt, ops["vis.0.mlp_out"].a_fmt) == ("nvfp4", "bf16")
    assert ops["vis.merger.0"].w_fmt == "nvfp4" and ops["vis.patch_embed"].w_fmt == "bf16"
    # every other vision tower keeps bf16, except a Qwen3-VL tower whose safetensors are fp32
    fp32_qwen = set()
    for e in entries():
        m = get_model(e["id"])
        if getattr(m, "vision", None) is None or e["id"] == M:
            continue
        if e["id"] == "d1-omni-600m":
            # Released weights and activations are fp32. vision_fmt stays unset.
            fm = {k: (f, b, a) for k, f, b, a in m.vision.wfmt}
            assert fm == {k: ("fp32", 32.0, "fp32") for k in ("attn", "merger", "mlp", "patch")}, e["id"]
            assert m.vision.w_Bpp == 4.0 and m.vision.audio.w_fmt == "fp32" and m.vision.audio.w_bits == 32.0
            continue
        if m.ctc_only and m.arch == "OmniASRForCTC":
            # 0.395: the whole checkpoint is fp32. vision_fmt stays unset; storage is 4 bytes per param.
            assert m.vision.wfmt == () and m.vision.w_Bpp == 4.0, e["id"]
            assert m.vision.audio.w_fmt == "fp32" and m.vision.audio.w_bits == 32.0, e["id"]
            rel = json.loads((ROOT / "accel_dse/data/releases" / (e["hf_id"].replace("/", "__") + ".json")).read_text())
            vb = sum(x["bytes"] for x in rel["roles"]["vision"].values())
            assert abs(m.vision.params * m.vision.w_Bpp - vb) < 1, e["id"]
            assert set(rel["roles"]["vision"]) == {"fp32"}, e["id"]
            continue
        if e["id"] == "granite-speech-5.0-470m-turboctc":
            # BatchNorm affine is fp32; every GEMM weight is bf16. Storage uses the mixed byte count.
            assert m.vision.wfmt == () and m.vision.audio.w_fmt == "bf16"
            rel = json.loads((ROOT / "accel_dse/data/releases/ibm-granite__granite-speech-5.0-470m-turboctc.json").read_text())
            vb = sum(x["bytes"] for x in rel["roles"]["vision"].values())
            assert abs(m.vision.params * m.vision.w_Bpp - vb) < 1
            assert rel["roles"]["vision"]["fp32"]["params"] > 0
            continue
        if m.vision.family == "minimax":
            # Conv3d patch_embedding is fp32. The rest of the tower and the projector are bf16.
            # patch_merge_mlp stays in the language role. wfmt stays empty; the patch GEMM
            # overrides w_fmt. Storage uses the mixed byte count.
            assert m.vision.wfmt == (), e["id"]
            rel = json.loads((ROOT / "accel_dse/data/releases" / (e["hf_id"].replace("/", "__") + ".json")).read_text())
            vb = sum(x["bytes"] for x in rel["roles"]["vision"].values())
            assert abs(m.vision.params * m.vision.w_Bpp - vb) < 1, e["id"]
            assert set(rel["roles"]["vision"]) == {"bf16", "fp32"}, e["id"]
            continue
        if m.vision.family == "qwen" and m.vision.w_Bpp == 4.0:
            fm = {k: (f, b, a) for k, f, b, a in m.vision.wfmt}
            assert fm == {k: ("fp32", 32.0, "bf16") for k in ("attn", "merger", "mlp", "patch")}, e["id"]
            fp32_qwen.add(e["id"])
            continue
        assert m.vision.wfmt == () and m.vision.w_Bpp == 2.0, e["id"]
    assert fp32_qwen == {"wevisone-vl-2b", "wevisone-vl-4b"}


def test_encoder_faster_and_smaller_than_bf16_tower():
    s = (Scenario(model=M, chip=CHIPS["100T"]).replace("serving.phase", "prefill").replace("serving.images", 1)
         .replace("serving.prompt", 2048))
    r = evaluate(s)
    m = get_model(M)
    bf = dataclasses.replace(m, vision=dataclasses.replace(m.vision, wfmt=(), w_Bpp=2.0))
    r0 = evaluate(s, bf)
    assert r.vision["weights_GB"] < 0.4 * r0.vision["weights_GB"]
    assert 0.8 < r.vision["s"] / r0.vision["s"] < 0.97        # fp8 attention GEMMs on an fp8-capable chip (100T 2×)

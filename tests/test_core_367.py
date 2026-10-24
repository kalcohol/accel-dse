"""0.367: Qwen3-VL deepstack prefill RMW is encoder DRAM, including when it fits SRAM.

The indexed add is read hidden + read feature + write hidden. Later LLM layers consume the feature, so the
bytes are not an encoder epilogue that the activation half may absorb. CHIP 1P's activation budget is 128 MiB;
the 8B 1024^2 case is 75497472 bytes and would vanish under a GEMM-style SRAM filter.
"""
from __future__ import annotations

from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import _component_time
from accel_dse.core.hardware import CHIPS, System
from accel_dse.core.model import load_release
from accel_dse.core.vision import image_grid, video_grid_of, vision_act_peak, vision_ops, vision_spec
from accel_dse.core.scenario import Scenario


def test_deepstack_rmw_is_dram_on_a_chip_that_could_hold_it():
    m = get_model("qwen3-vl-8b")
    g = image_grid(m.vision, 1024, 1024)
    ops = vision_ops(m.vision, g, 1)
    ds = next(o for o in ops if o.name == "vis.deepstack")
    rmw = 3 * 3 * 1024 * 4096 * 2
    assert ds.stream is True and ds.kind == "vector"
    assert ds.vec == 3 * 1024 * 4096 and ds.act_bytes == rmw
    assert ds.kv_read == ds.kv_write == ds.state_rw == 0.0
    sys_ = System(CHIPS["1P"])
    budget = sys_.chip.sram_bytes / 2.0
    assert ds.act_bytes < budget
    full = _component_time([(ops, 1)], sys_, "os", "bf16")
    bare = _component_time([([o for o in ops if o.name != "vis.deepstack"], 1)], sys_, "os", "bf16")
    assert abs(full["dram_GB"] * 1e9 - bare["dram_GB"] * 1e9 - rmw) <= 1.0
    peak = vision_act_peak(m.vision, g, 1)
    assert peak == max(o.act_bytes for o in ops if o.kind == "gemm")
    assert peak == vision_act_peak(m.vision, g, 1, "bf16")
    fp8 = get_model("qwen3-vl-8b-fp8")
    assert fp8.act_fmt == "fp8"
    ds8 = next(o for o in vision_ops(fp8.vision, image_grid(fp8.vision, 1024, 1024), 1)
               if o.name == "vis.deepstack")
    assert ds8.act_bytes == rmw and ds8.stream is True
    clip = video_grid_of(m, Scenario(model="qwen3-vl-8b").replace(
        "serving.videos", 1).replace("serving.video_frames", 4).serving)
    dsv = next(o for o in vision_ops(m.vision, clip.slice, 1) if o.name == "vis.deepstack")
    assert dsv.stream is True and dsv.act_bytes == 3 * 3 * clip.slice.merged * 4096 * 2


def test_qwen35_and_qwen38_reject_nonempty_deepstack():
    for mid, indexes in (("qwen3.5-397b-a17b", [8, 16, 24]), ("qwen3.8-27b-fp8", [8, 16, 24])):
        m = get_model(mid)
        cfg = load_release(m.hf_id)["config"]
        base = vision_spec(m.hf_id, cfg)
        assert base is not None and base.deepstack == 0 and base.label == "Qwen ViT"
        deep = {**cfg, "vision_config": {**cfg["vision_config"], "deepstack_visual_indexes": indexes}}
        assert vision_spec(m.hf_id, deep) is None

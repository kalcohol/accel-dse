"""0.204: walls in the UI (wiring), fixed-overhead naming, and the NPU preset TPOT back-of-envelope."""
import sys
import pathlib
sys.path.insert(0, ".")
from accel_dse import api
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)


def test_ui_wiring():
    root = pathlib.Path(api.__file__).parent / "web"
    js, html = W.read(), W.read(W.INDEX_HTML)          # 0.286: parsed
    assert {"walls-bar", "walls-note", "c-dispatch", "c-dispatch_us"} <= W.html_ids(html)
    assert W.has(js, "renderWalls(r)") and W.any_string_contains(js, "fixed overhead") and "dispatch_us" in W.strings(js)
    lbl = W.literal(js, "WALL_LBL")                  # no 'static' label in the walls legend (keys or labels)
    assert "fixed_overhead" in W.object_keys(lbl) and not any("static" in str(v) for _, v in lbl)
    smoke = (pathlib.Path(api.__file__).parents[1] / "scripts/ui_headless_smoke.py").read_text()
    assert "#walls-bar" in smoke and "burst ends on ideal" in smoke


def test_walls_keys_renamed():
    w = api.api_eval({"scenario": {"model": "qwen3-8b"}})["walls"]
    assert "fixed_overhead" in w["share"] and "static" not in w["share"]


def test_npu_preset_tpot_back_of_envelope():
    # LPDDR5X 4 × 64 bit @ 8533 MT/s = 273.1 GB/s × mem_eff 0.7; bytes = streamed bf16 weights (8.19 B − 0.62 B
    # embedding rows, looked up not streamed) − SRAM-resident share + KV read (4096 ctx × 36 layers × 8 kv × 128 × 2 × 2 B)
    bw = 4 * 64 / 8 * 8533e6 * 0.7
    for n in ("100T", "1P"):     # 0.205: 1P defaults to HBM3E; pin the ≤ 0.204 LPDDR pairing this check is about
        r = evaluate(Scenario(chip=CHIPS[n], mem_id="lpddr5x_4x64_8533_16g"))
        st = r.stages[0]
        kv = 4097 * 36 * 8 * 128 * 2 * 2      # ctx 4096 + the new token
        assert abs(st.dram["kv_read"] - kv) / kv < 1e-9
        resident = CHIPS[n].sram_mib * 2**20
        assert abs((st.dram["w_touched"] - st.dram["weights"]) - resident) / resident < 0.1   # SRAM holds ≈ its size of weights
        assert abs(r.tpot - st.dram["total"] / bw) / r.tpot < 1e-3      # DRAM-limited: TPOT = bytes / bandwidth
        assert r.walls()["wall"] == "dram"

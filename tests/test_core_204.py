"""0.204: walls in the UI (wiring), fixed-overhead naming, and the NPU preset TPOT back-of-envelope."""
import sys
import pathlib
sys.path.insert(0, ".")
from accel_dse import api
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario


def test_ui_wiring():
    root = pathlib.Path(api.__file__).parent / "web"
    js, html = (root / "app.js").read_text(), (root / "index.html").read_text()
    for t in ('id="walls-bar"', 'id="walls-note"', 'id="c-dispatch"', 'id="c-dispatch_us"'):
        assert t in html, t
    assert "renderWalls(r);" in js and "fixed overhead" in js and "'dispatch_us'" in js
    assert "static" not in js.split("const WALL_LBL")[1].split("};")[0]   # no 'static' label in the walls legend
    smoke = (pathlib.Path(api.__file__).parents[1] / "scripts/ui_headless_smoke.py").read_text()
    assert "#walls-bar" in smoke and "burst ends on ideal" in smoke


def test_walls_keys_renamed():
    w = api.api_eval({"scenario": {"model": "qwen3-8b"}})["walls"]
    assert "fixed_overhead" in w["share"] and "static" not in w["share"]


def test_npu_preset_tpot_back_of_envelope():
    # LPDDR5X 4 × 64 bit @ 8533 MT/s = 273.1 GB/s × mem_eff 0.7; bytes = streamed bf16 weights (8.19 B − 0.62 B
    # embedding rows, looked up not streamed) − SRAM-resident share + KV read (4096 ctx × 36 layers × 8 kv × 128 × 2 × 2 B)
    bw = 4 * 64 / 8 * 8533e6 * 0.7
    for n in ("100T", "1P"):
        r = evaluate(Scenario(chip=CHIPS[n]))
        st = r.stages[0]
        kv = 4097 * 36 * 8 * 128 * 2 * 2      # ctx 4096 + the new token
        assert abs(st.dram["kv_read"] - kv) / kv < 1e-9
        resident = CHIPS[n].sram_mib * 2**20
        assert abs((st.dram["w_touched"] - st.dram["weights"]) - resident) / resident < 0.1   # SRAM holds ≈ its size of weights
        assert abs(r.tpot - st.dram["total"] / bw) / r.tpot < 1e-3      # DRAM-limited: TPOT = bytes / bandwidth
        assert r.walls()["wall"] == "dram"

"""0.184: sizing API payload for dram_units as the UI panel consumes it; unit-floor flag; sizing fuzz smoke;
LPDDR PHY density sourced (Samsung SF2 LPDDR6 PHY)."""
import json
import subprocess
import sys
from pathlib import Path

from accel_dse import api as A
from accel_dse.core.budget import DENSITY_PRESETS
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)

ROOT = Path(A.__file__).resolve().parents[1]
WEB = Path(A.__file__).parent / "web"
ROW_KEYS = {"key", "label", "unit", "current", "min_ok", "frac", "floor_hit", "fail_at", "limited_by", "integer",
            "unit_floor"}


def _sizing(body):
    return json.loads(json.dumps(A.api_insight_sizing(body)))      # what the panel receives over HTTP


def test_payload_dram_units_row_no_density():
    r = _sizing({"models": ["qwen3-8b"]})
    row = next(x for x in r["resources"] if x["key"] == "dram_units")
    assert ROW_KEYS <= set(row) and row["label"] == "DRAM 封装 / 堆栈数" and row["unit"] == "个"
    assert row["integer"] is True and row["current"] == 4 and "mm2_saved" not in row     # UI renders '—'
    assert any(x["key"] == "dram_units" for x in r["binding"]["binding"] + r["binding"]["most_slack"])


def test_payload_priced_and_joint_skip():
    r = _sizing({"models": ["wan2.1-1.3b"], "budget": {"density_preset": "n7_tpuv4i"}, "joint": True})
    row = next(x for x in r["resources"] if x["key"] == "dram_units")
    assert row["min_ok"] == 3 and abs(row["mm2_saved"] - 64 * 0.695 / 24) < 1e-9
    assert all(e["key"] != "dram_units" for e in r["joint"]["resources"])


def test_payload_unit_floor_single_package():
    r = _sizing({"models": ["qwen3-8b"], "scenario": {"mem_id": "lpddr5x_1x64_8533_16g"}})
    row = next(x for x in r["resources"] if x["key"] == "dram_units")
    assert row["unit_floor"] is True and row["floor_hit"] is True and row["current"] == row["min_ok"] == 1
    assert next(x for x in r["resources"] if x["key"] == "dram_bw")["unit_floor"] is False


def test_frontend_handles_new_fields():
    js = W.read()                                      # 0.286: token match
    assert W.has(js, "x.unit_floor") and W.has(js, "x.integer") and W.has(js, "x.mm2_saved != null")


def test_lpddr_phy_density_sourced():
    for name, p in DENSITY_PRESETS.items():
        v, tag, src = p["fields"]["mm2_per_dq_lpddr"] if "fields" in p else p["mm2_per_dq_lpddr"]
        assert abs(v - 0.695 / 24) < 1e-12 and tag == "推导/二手" and "LPDDR6" in src


def test_fuzz_smoke_no_hang():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "fuzz_sizing.py"), "6", "7", "120"],
                         capture_output=True, text=True, timeout=900)
    assert out.returncode == 0 and "0 finding(s)" in out.stdout, out.stdout[-2000:] + out.stderr[-2000:]


def test_hbm_phy_density_sourced_0185():
    for p in DENSITY_PRESETS.values():
        v, tag, src = p["fields"]["mm2_per_dq_hbm"]
        assert v == 8 * 4 / 2048 and tag == "推导/二手" and "Hot Chips 2026" in src


def test_ui_headless_smoke_script_compiles():
    import py_compile
    py_compile.compile(str(ROOT / "scripts" / "ui_headless_smoke.py"), doraise=True)

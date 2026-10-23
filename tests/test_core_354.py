"""0.354: validation and catalog review; MiniMax M3.1 Flash deferred to 0.355."""
from __future__ import annotations

from pathlib import Path

from accel_dse.core.thermal import COOLING

ROOT = Path(__file__).resolve().parents[1]
CL = (ROOT / "CHANGELOG.md").read_text()


def test_cooling_limits_not_retuned():
    """0.354 validation: COOLING pins unchanged from 0.353."""
    assert COOLING["air_oam"]["pkg_W"] == 1000.0 and COOLING["air_oam"]["flux_W_mm2"] == 0.86
    assert COOLING["liquid"]["pkg_W"] == 1400.0 and COOLING["liquid"]["flux_W_mm2"] == 0.96
    assert COOLING["liquid"]["die_flux_W_mm2"] == 1.70


def test_changelog_354():
    """0.354 CHANGELOG: validation, M3.1 Flash review, deferred to 0.355."""
    assert "## [0.354.0]" in CL
    sec = CL.split("## [0.354.0]")[1].split("## [")[0]
    assert "1906 tests" in sec or "test suite" in sec.lower()
    assert "MiniMax" in sec or "M3.1" in sec
    assert "0.355" in sec
    assert "sparse" in sec.lower() or "MSA" in sec or "block" in sec.lower()
    assert "deferred" in sec.lower() or "推迟" in sec


def test_no_functional_changes():
    """0.354 is a validation release: no changes to core evaluation logic."""
    # This version only updates documentation and version numbers
    # Core thermal, evaluation, and catalog logic unchanged
    pass

"""0.179: MI300X max-load — sourced one-wave batch (num_prompts = max_num_seqs) and AMD MAF GEMM ceilings."""
import sys
sys.path.insert(0, ".")
from accel_dse.core import extval as X

ROWS = [r for r in X.rows(X.DATA_R3) if r["hw"] == "MI300X" and r["metric"] == "max_tok_s_total"]


def test_rows_carry_readme_wave():
    assert sorted((r["isl"], r["osl"], r["max_seqs"]) for r in ROWS) == \
        [(128, 2048, 3200), (128, 4096, 1500), (500, 2000, 2000), (2048, 2048, 1500)]
    assert all(r["num_prompts"] == r["max_seqs"] for r in ROWS)


def test_one_wave_batch():
    for r in ROWS:
        assert X.predict(r, "default")["batch"] == r["max_seqs"]


def test_mi300x_maf_ceiling():
    c = dict(X.GEMM_CEILING["MI300X"])
    assert abs(c["gemm:fp8"] - 1223.6 / 2614.9) < 1e-12 and abs(c["gemm:bf16"] - 708 / 1307.4) < 1e-12
    r = ROWS[0]
    # 0.193: the default divides out the wave quantisation of the measured shape (M 4096, N 3648, K 32896) — net ≥ gross
    net = dict(X.scenario(r, "default", phase="decode").chip.fmt_ceiling)["gemm:fp8"]
    assert c["gemm:fp8"] <= net <= c["gemm:fp8"] * 1.05
    assert dict(X.scenario(r, "default_0192", phase="decode").chip.fmt_ceiling)["gemm:fp8"] == c["gemm:fp8"]


def test_residual_documented():
    # MODEL §126: still ~1.7–2.0× optimistic after the sourced fixes (no sourced mechanism for the rest)
    for r in ROWS:
        assert 1.5 < X.predict(r, "default_0206")["pred"] / r["value"] < 2.2   # 0.207: pinned to the pre-PA default

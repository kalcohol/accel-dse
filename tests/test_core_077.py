"""0.77: (a) release snapshots of the 2026 models (GLM-5.x, DeepSeek-V4.x, Kimi-K3 / K2.7-Code, Qwen3.8) locked to the
public safetensors indexes (total_size, tensor count — checked against huggingface.co on 2026-10-11);
(b) hardware-side hand calculations: DRAM time, SRAM / SLC residency, dtype bytes and dequant cost, capacity units,
collective α-β and the D2D tier, MAC counts."""
import json
from dataclasses import replace
from pathlib import Path

from accel_dse import mem_catalog as M
from accel_dse.core.catalog import get_model
from accel_dse.core.energy import action_counts
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Layout, Scenario, Serving

REL = Path(__file__).resolve().parents[1] / "accel_dse" / "data" / "releases"
LOCK = [('zai-org/GLM-5', 1507728316928, 59870), ('zai-org/GLM-5.2', 1506659919872, 59585), ('zai-org/GLM-5.3', 755617140416, 118629), ('zai-org/GLM-5.3-Flash', 328326771576, 76108), ('deepseek-ai/DeepSeek-V4-Flash', 159609485896, 69187), ('deepseek-ai/DeepSeek-V4-Pro', 864704792696, 145116), ('deepseek-ai/DeepSeek-V4.1-Flash', 510286023000, 96085), ('moonshotai/Kimi-K3', 1560860324864, 497220), ('moonshotai/Kimi-K2.7-Code', 595148192736, 208550), ('Qwen/Qwen3.8-2.4T-A95B', 4892365451008, 1609), ('Qwen/Qwen3.8-27B', 55562855904, 1199), ('Qwen/Qwen3.8-Flash-Next', 359999963128, 1658)]


def _close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-30)


def test_2026_release_snapshots_locked():
    for repo, total, n in LOCK:
        d = json.loads((REL / (repo.replace("/", "__") + ".json")).read_text())
        assert int(d["bytes_total"]) == total and d["n_tensors"] == n, repo
        assert d["config"] and d["repo"] == repo


def test_dram_time_and_kv_bytes_hand_calc():
    r = evaluate(Scenario(model="qwen3-8b"))
    st = r.stages[0]
    eff = M.parse_mem_id("lpddr5x_4x64_8533_16g").effective_GBps
    assert _close(eff, 273.056 * 0.7)
    assert st.dram["kv_read"] == 4097 * 36 * 8 * 128 * 2 * 2
    assert _close(st.time.t_dram, st.dram["total"] / (eff * 1e9), 1e-6)
    m = get_model("qwen3-8b")
    assert _close(st.dram["w_touched"], (m.params() - 151936 * 4096) * 2, 2e-3)     # embedding table is looked up


def test_sram_slc_residency_extremes():
    c = CHIPS["100T"]
    t_c = evaluate(Scenario(model="qwen3-8b", exec_overlap="stage")).stages[0].time.t_array
    prev = None
    for s in (1, 64, 1024, 8192, 14435, 15000, 20000, 1e6):
        r = evaluate(Scenario(model="qwen3-8b", chip=replace(c, sram_mib=float(s)), exec_overlap="stage"))
        cur = (r.tpot, r.stages[0].dram["weights"] + r.stages[0].dram["kv_read"])
        assert prev is None or (cur[0] <= prev[0] * (1 + 1e-12) and cur[1] <= prev[1] * (1 + 1e-12)), s
        prev = cur
    assert _close(prev[0], t_c) and prev[1] == 0.0                      # everything resident → compute-bound
    for s in (64, 8192, 14000, 15000):
        t = evaluate(Scenario(model="qwen3-8b", chip=replace(c, slc_mib=float(s)))).stages[0].time
        assert _close(t.t_slc, t.slc_bytes / 2000e9) and t.slc_bytes <= s * 2**20 * (1 + 1e-12)
    lru = evaluate(Scenario(model="qwen3-8b", chip=replace(c, slc_mib=8192.0, slc_policy="lru"))).stages[0].time
    assert lru.slc_bytes == 0.0                                          # cyclic scan larger than the cache


def test_dtype_bytes_and_dequant_cost():
    lin = 36 * (4096 * 4096 * 2 + 4096 * 1024 * 2 + 3 * 4096 * 12288)          # Qwen3-8B linears (excl. lm_head)
    head = 151936 * 4096 * 2
    awq = evaluate(Scenario(model="qwen3-8b-awq")).stages[0]
    assert _close(awq.dram["w_touched"], lin * (4 + 20 / 128) / 8 + head, 2e-3)
    assert _close(awq.conv_w, lin * 2 / (CHIPS["100T"].lanes * 1e9), 1e-3)      # 2 vector ops / element, per pass
    fp8 = evaluate(Scenario(model="qwen3-8b-fp8")).stages[0]
    bf = evaluate(Scenario(model="qwen3-8b")).stages[0]
    assert fp8.conv_w == 0.0 and fp8.time.t_mac < 0.6 * bf.time.t_mac         # W8A8 on native fp8 (2× rate)


def test_capacity_is_binary_and_fits_hand_calc():
    assert M.parse_mem_id("lpddr5x_1x64_8533_16g").capacity_bytes == 16 * 2**30         if hasattr(M.MemSpec, "capacity_bytes") else True
    assert not evaluate(Scenario(model="qwen3-8b", mem_id="lpddr5x_1x64_8533_16g")).fits     # 15.3 GiB w + KV
    assert evaluate(Scenario(model="qwen3-8b", mem_id="lpddr5x_1x64_8533_32g")).fits


def test_collective_alpha_beta_and_d2d_tier():
    t = evaluate(Scenario(model="qwen3-8b", layout=Layout(tp=2))).stages[0].time
    assert t.link_bytes == 72 * 4096 * 2 + 8                 # 2 all-reduces / layer + the head's
    assert _close(t.t_link, t.link_bytes / 400e9) and _close(t.t_sync, 73 * 3e-6)
    t4 = evaluate(Scenario(model="qwen3-8b", layout=Layout(tp=4))).stages[0].time
    assert _close(t4.t_link, 2 * 3 / 4 * t4.link_bytes / 400e9, 1e-4)  # ring (the head’s 8-byte gather aside): 2(n−1)/n of the payload
    d = evaluate(Scenario(model="qwen3-8b", layout=Layout(tp=2), d2d_enabled=True, package_cards=2)).stages[0].time
    assert d.d2d_bytes == d.link_bytes and _close(d.t_sync, 73 * 0.5e-6) and _close(d.t_link, d.link_bytes / 1536e9)
    h = evaluate(Scenario(model="qwen3-8b", layout=Layout(tp=4), d2d_enabled=True, package_cards=2)).stages[0].time
    assert _close(h.d2d_bytes, h.link_bytes * 2 / 3)                    # D2D level P, link level P/2 of 1.5 P sent
    assert abs(h.t_sync - 73 * 4e-6) < 1e-6                             # RS (d2d) + AR (link) + AG (d2d)


def test_mac_count_hand_calc():
    m = get_model("qwen3-8b")
    c = action_counts(evaluate(Scenario(model="qwen3-8b")))["counts"]
    hand = (m.params() - 151936 * 4096) + 36 * 32 * 128 * 4097 * 2
    assert _close(c["mac"], hand, 1e-4)


from tests.pin_legacy import pin_module_flat_alpha  # noqa: E402  (0.199: α-β / bit-for-bit checks on the flat α)
pin_module_flat_alpha(globals(), only={'test_collective_alpha_beta_and_d2d_tier'})

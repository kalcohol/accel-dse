"""L4 — memory planning: residency on stored bytes, staging, partial KV, heaviest stage."""
from __future__ import annotations

from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIP_100T, Chip
from accel_dse.core.ir import Shard
from accel_dse.core.memplan import StageStorage, plan, stage_storage, step_dram_bytes
from accel_dse.core.parallel import Layout, plan_stages
from accel_dse.core.scenario import Scenario, Serving

MiB = 2**20
GiB = 2**30


def test_toy_residency_hot_first_then_experts():
    st = StageStorage(hot_w=10 * MiB, expert_w=100 * MiB, kv_per_seq=1 * MiB, idx_per_seq=0, state_per_seq=0)
    mp = plan(st, batch_local=4, sram_bytes=64 * MiB, dram_cap=64 * GiB, max_act=1 * MiB)
    assert mp.staging == 2 * MiB
    assert mp.pinned_hot == 10 * MiB and mp.pinned_expert == 52 * MiB and mp.kv_sram == 0
    t = {"hot": 10 * MiB, "exp": 20 * MiB, "kv_read": 4 * MiB, "kv_write": 0.0, "state": 0.0, "lookup": 0.0}
    d = step_dram_bytes(mp, st, t)
    assert d["weights"] == 20 * MiB * (1 - 52 / 100)
    assert d["kv_read"] == 4 * MiB


def test_toy_partial_kv_hit():
    st = StageStorage(hot_w=1 * MiB, expert_w=0, kv_per_seq=10 * MiB, idx_per_seq=0, state_per_seq=0)
    mp = plan(st, 4, sram_bytes=23 * MiB, dram_cap=GiB * 64, max_act=0)
    assert mp.kv_sram == 20 * MiB                          # 23 − 2 staging − 1 weights
    d = step_dram_bytes(mp, st, {"hot": 1 * MiB, "exp": 0, "kv_read": 40 * MiB, "kv_write": 0, "state": 0, "lookup": 0})
    assert abs(d["kv_read"] - 40 * MiB * 0.5) < 1


def test_toy_not_resident_when_staging_eats_sram():
    # 284 KiB of weights do NOT fit when the double-buffered staging takes the whole 2 MiB SRAM
    st = StageStorage(hot_w=284 * 1024, expert_w=0, kv_per_seq=0, idx_per_seq=0, state_per_seq=0)
    mp = plan(st, 1, sram_bytes=2 * MiB, dram_cap=GiB, max_act=1 * MiB)
    assert mp.pinned_hot == 0 and mp.residency == 0
    mp2 = plan(st, 1, sram_bytes=3 * MiB, dram_cap=GiB, max_act=1 * MiB)
    assert mp2.residency == 1.0


def test_full_residency_removes_weight_traffic():
    big = CHIP_100T.__class__(**{**CHIP_100T.__dict__, "sram_mib": 4096.0})
    r = evaluate(Scenario(model="qwen3-0.6b", chip=big, serving=Serving(batch=1, ctx=128)))
    assert r.stages[0].mem.residency == 1.0 and r.stages[0].dram["weights"] == 0.0
    r0 = evaluate(Scenario(model="qwen3-0.6b", serving=Serving(batch=1, ctx=128)))
    assert r0.stages[0].dram["weights"] > 0 and r.tpot < r0.tpot


def test_deepseek_pp8_heaviest_stage_capacity_hand_count():
    s = get_model("deepseek-v3")
    bits = s.fmt("expert").bits
    # independent count for stage 0 (layers 0-7: 3 dense + 5 MoE) + embedding, KV at 4096 ctx × 16 seq
    dense_ffn = 3 * 7168 * 18432
    moe = 256 * 3 * 7168 * 2048 + 3 * 7168 * 2048 + 7168 * 256        # routed + shared + router
    attn = 7168 * 1536 + 1536 * 128 * 192 + 7168 * 576 + 512 * 128 * 256 + 128 * 128 * 7168
    w = (3 * (attn + dense_ffn) + 5 * (attn + moe)) * bits / 8 + 129280 * 7168 * s.fmt("embed").bits / 8
    kv = 8 * 4096 * 576 * 2 * 16
    st = stage_storage(s, 0, 8, True, False, Shard(), 4096)
    assert abs(st.weights - w) / w < 0.002, (st.weights / GiB, w / GiB)
    assert abs(st.kv_per_seq * 16 - kv) / kv < 1e-9
    r = evaluate(Scenario(model="deepseek-v3", mem_id="hbm3e_8s_12h24g_9200", layout=Layout(pp=8),
                          serving=Serving(batch=16, ctx=4096 - 1)))
    needs = [x.mem.dram_need for x in r.stages]
    # stages 1-4 hold 8 MoE layers each → heavier than stage 0 (3 dense + 5 MoE + embedding)
    assert needs.index(max(needs)) == 1 and abs(needs[1] - needs[4]) < 1
    w8 = 8 * (attn + moe) * bits / 8
    assert abs(r.stages[1].mem.stored_w - w8) / w8 < 0.002
    assert r.fits


def test_capacity_failure_reported():
    r = evaluate(Scenario(model="deepseek-v3", serving=Serving(batch=1, ctx=1024)))   # 64 GiB LPDDR, 1 card
    assert not r.fits and any("容量不足" in w for w in r.warnings)


def test_linear_attention_state_counted():
    s = get_model("qwen3-next-80b-a3b")
    st = stage_storage(s, 0, s.n_layers, True, True, Shard(), 4096)
    lin = [l for l in s.layers if l.core.kind == "linear"]
    exp = sum(l.core.n_state_heads * l.core.state_dk * l.core.state_dv * 4 + l.core.conv_channels * 3 * 4 for l in lin)
    assert abs(st.state_per_seq - exp) < 1


def test_reviewer_oracle_deepseek_pp8_bf16_what_if_171_44_gib():
    """Reviewer oracle (0.32 review): DeepSeek-V3 PP8 heaviest stage = 171.44 GiB of weights when
    every weight is bf16.  As released (fp8) it is ~half; the bf16 number is reproduced as a what-if."""
    from accel_dse.core.model import with_formats
    s = with_formats(get_model("deepseek-v3"), {r: "bf16" for r in ("attn", "mlp", "expert", "shared_expert",
                                                                    "router", "embed", "lm_head", "mtp")})
    st = stage_storage(s, 8, 16, False, False, Shard(), 4096)
    assert abs(st.weights / GiB - 171.44) < 0.05, st.weights / GiB
    rel = stage_storage(get_model("deepseek-v3"), 8, 16, False, False, Shard(), 4096)
    assert abs(rel.weights / st.weights - 8.0 / 16.0) < 0.01


def test_spec_k_sweep_identity():
    base = Scenario(model="deepseek-v3", mem_id="hbm3e_8s_12h24g_9200", layout=Layout(tp=8, ep=8),
                    serving=Serving(batch=32, ctx=2048, spec_k=2))
    a = evaluate(base)
    b = evaluate(base.replace("serving.spec_k", 2))
    assert a.tpot == b.tpot and a.throughput == b.throughput

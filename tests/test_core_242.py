"""0.242: the fused prompt queue is work-conserving without the slot cap (a Lindley replay of the DES's own (arrival,
share) sequence reproduces its waits to ±2 %), but the share grows along a busy period (each finished prompt adds a
decode) and the long waits queue behind the large-k shares.  The TTFT bins follow the batch law weighted by the queue
that waits out each share (pdqueue._queue_k_law); the 0.241 slot block is weighted the same way (_slot_block_q)."""
import math

from accel_dse.core import pdqueue, pdsim
from accel_dse.core.hardware import CHIPS
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import PDConfig, Scenario, Serving

# DES TTFT p90 (ms), n_req 4000, warmup 800, seeds 1..10 (/workspace/sd239/des_*.txt; DES unchanged since 0.235)
DES_TTFT90 = {"vllm": {17.37: 1158, 18.0: 1436, 18.6: 1883, 19.32: 2748},
              "sglang_mixed": {17.37: 1219, 18.0: 1603, 18.6: 2163, 19.32: 3353}}
OLD_TTFT90 = {"sglang_mixed": {17.37: 1008, 18.0: 1199, 18.6: 1461, 19.32: 2110}}   # 0.241 analytic


def _ctx(prof):
    s = Scenario(chip=CHIPS["1P"], layout=Layout(tp=2), model="qwen3-8b", mem_id="hbm3e_8s_12h24g_9200", mem_cap_GB=48.0,
                 serving=Serving(batch=32, prompt=16384, out_len=256),
                 pd=PDConfig(enabled=True, prefill_layout=Layout(tp=2), prefill_cards=2, decode_cards=6, load=0.6,
                             chunk_profile=prof))
    return pdsim.capture_ctx(s)


def test_queue_weights_closed_form():
    for a in (0.3, 0.7, 0.86):
        wj, rest, tot = pdqueue._queue_weights(a, 40)
        assert math.isclose(tot, a * a / (2 * (1 - a)) + a / 2)
        assert math.isclose(sum(wj) + rest, tot) and rest >= 0
        assert math.isclose(wj[0], (1 - a) * a / 2)
    # light load: nearly all weight on the busy period's first customers
    wj, rest, tot = pdqueue._queue_weights(0.1, 3)
    assert (wj[0] + wj[1]) / tot > 0.9


def test_queue_k_law():
    k0 = [(5, 0.3), (10, 0.4), (18, 0.3)]
    m0 = sum(k * p for k, p in k0)
    L = pdqueue._queue_k_law(4.65, 0.1843, k0, 8.23, 256, 32)
    assert math.isclose(sum(w for _, w in L), 1.0) and all(0 <= k <= 32 for k, _ in L)
    m = sum(k * w for k, w in L)
    assert m > m0 + 8                                   # queued work is served deep in busy periods
    m_lo = sum(k * w for k, w in pdqueue._queue_k_law(1.0, 0.1843, k0, 8.23, 256, 32))
    assert m0 < m_lo < m0 + 1.5                         # light load ≈ the time-average law
    assert pdqueue._queue_k_law(5.5, 0.1843, k0, 8.23, 256, 32) is None


def test_slot_block_q_heavier_than_customer_weighted():
    b = pdqueue._slot_block(4.65, 0.837, 13.2, 32, 4.19, 256, 0.0019)
    q = pdqueue._slot_block_q(4.65, 0.837, 13.2, 32, 4.19, 256, 0.0019)
    assert q[0] == b[0] and q[2] == b[2] and 0 < b[3] < q[3] < 1 and math.isclose(q[1], q[3] / 32)
    assert pdqueue._slot_block_q(4.0, 0.8, 10.0, 32, 8.2, 256, 0.003) is None


def test_fused_ttft_p90_vs_des():
    for prof, pts in DES_TTFT90.items():
        rep, ctx = _ctx(prof)
        for lam, des in pts.items():
            new = pdqueue._coloc_chunked({**ctx, "slo": (2.0, 0.05)}, lam)["ttft"]["p90"] * 1e3
            if prof == "vllm":
                assert abs(new - des) / des < 0.06, (prof, lam, new, des)
            else:   # sglang_mixed: the iid slot-block lump was optimistic (0.247: −4 / −9 / −12 / −11 %); 0.248's
                # queue-ful entry base (slot-bound share β/d ≈ 0.5 here) gives 1264 / 1542 / 1982 / 3150 vs DES
                # 1219 / 1603 / 2163 / 3353 (+3.7 / −3.8 / −8.4 / −6.1 %; mean |err| 9.2 → 5.5 %) — band ±10 %
                assert OLD_TTFT90[prof][lam] < new and abs(new - des) / des < 0.10, (prof, lam, new, des)

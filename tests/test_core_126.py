"""0.126 hostile review round 2: attention key counts (window / compressed / top-k) decode = prefill, GQA under
uneven TP, KV / state bytes per family vs hand formulas, hierarchical collective volumes vs textbook, PP partition
legality + work conservation."""
import dataclasses
import random

from accel_dse.core import catalog as C
from accel_dse.core import ir
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import Link
from accel_dse.core.ir import Phase, Shard
from accel_dse.core.memplan import stage_storage
from accel_dse.core.model import AttnCore
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import Scenario
from accel_dse.core.schedule import collective_seconds, collective_wire, fabric_collective, wire_single


def _cores():
    seen = {}
    for e in C.list_models():
        if not e["evaluable"]:
            continue
        for L in C.get_model(e["id"]).layers:
            c = getattr(L, "core", None)
            if c is not None and c.kind in ("gqa", "mla"):
                seen.setdefault((c.window, c.topk, c.compress, c.dual, c.qsa), (e["id"], c))
    return seen.values()


def test_decode_keys_equal_prefill_last_position():
    """Keys attended by a decode query at context n = keys of the last prefill position (never more than n);
    0.125 charged DeepSeek-V4 compressed + window layers ⌈n/c⌉ + w > n keys at short context."""
    for mid, c in _cores():
        for n in (1, 2, 7, 64, 100, 129, 130, 300, 1000, 5000):
            if not c.dual:
                assert c.ctx_eff(n) <= n, (mid, n)
            assert c.ctx_eff(n) == c.keys_sum(n) - c.keys_sum(n - 1), (mid, n, c.ctx_eff(n))


def test_keys_sum_bruteforce():
    rng = random.Random(126)
    for _ in range(1500):
        w, cc, k = rng.randint(1, 40), rng.randint(2, 6), rng.randint(1, 30)
        kw = rng.choice([dict(window=w), dict(topk=k), dict(compress=cc), dict(compress=cc, window=w),
                         dict(compress=cc, window=w, topk=k), dict(compress=cc, window=w, topk=k, dual=True)])
        a = AttnCore(kind="gqa", n_q=4, n_kv=2, qk_dim=8, v_dim=8, **kw)
        n = rng.randint(1, 150)
        ref = sum(a.ctx_eff(p) if a.dual else min(p, a.ctx_eff(p)) for p in range(1, n + 1))
        assert a.keys_sum(n) == ref, (kw, n)


def test_gqa_uneven_tp_kv_consistent():
    for mid in ("qwen3-8b", "gpt-oss-20b", "llama-3.3-70b"):
        m = C.get_model(mid)
        c = m.layers[0].core
        kv1 = stage_storage(m, 0, m.n_layers, True, True, Shard(), 1000, 0).kv_per_seq
        r1 = sum(o.kv_read for o in ir.build_rank_ops(m, 0, Phase("decode", 1, 1, 999), Shard()))
        for tp in (2, 3, 5, 6, 16, 64):
            sh = Shard(tp=tp, ep=tp) if m.is_moe else Shard(tp=tp)
            kv, hq = ir.gqa_local(c.n_q, c.n_kv, tp)
            assert kv * tp >= c.n_kv and hq * tp >= c.n_q and kv >= 1
            st = stage_storage(m, 0, m.n_layers, True, True, sh, 1000, 0).kv_per_seq
            rd = sum(o.kv_read for o in ir.build_rank_ops(m, 0, Phase("decode", 1, 1, 999), sh))
            assert abs(st / kv1 - kv / c.n_kv) < 1e-9 and abs(rd / r1 - kv / c.n_kv) < 1e-9, (mid, tp)


def test_kv_state_bytes_hand_formulas():
    P = 8192
    ref = {"llama-3.1-8b": 32 * 8 * 128 * 2 * 2 * P, "qwen3-8b": 36 * 8 * 128 * 2 * 2 * P,
           "deepseek-v3": 61 * (512 + 64) * 2 * P,
           "gpt-oss-120b": 18 * 8 * 64 * 2 * 2 * P + 18 * 8 * 64 * 2 * 2 * 128,      # sliding layers keep 128
           "qwen3-next-80b-a3b": 12 * 2 * 256 * 2 * 2 * P}
    for mid, v in ref.items():
        m = C.get_model(mid)
        st = stage_storage(m, 0, m.n_layers, True, True, Shard(), P, 0)
        assert st.kv_per_seq == v, (mid, st.kv_per_seq, v)
    m = C.get_model("qwen3-next-80b-a3b")      # 36 GDN layers: fp32 state 32 heads × 128 × 128 + conv (kernel − 1)
    st = stage_storage(m, 0, m.n_layers, True, True, Shard(), P, 0)
    assert st.state_per_seq == 36 * (32 * 128 * 128 + (2 * 16 * 128 + 32 * 128) * 3) * 4


def test_hierarchical_collectives_textbook():
    """Equal tiers: the hierarchical volumes telescope to the flat ring / direct formulas; per-tier wire bytes sum to
    the flat bytes for all-reduce and all-to-all."""
    ln = Link(100.0, 0.0)
    B = 1e6
    for g, pk, nd in ((8, 2, 4), (16, 2, 8), (12, 3, 6), (32, 4, 8)):
        for kind in ("allreduce", "allgather", "alltoall"):
            bw, _, _, _ = fabric_collective(kind, B, g, ln, d2d=ln, package=pk, net=ln, node=nd)
            flat, _ = collective_seconds(kind, B, g, ln)
            if kind == "allgather":
                assert abs(bw - flat) / flat < 1e-12, (g, kind)
            elif kind == "allreduce":
                assert abs(bw - flat) / flat < 1e-12, (g, kind)
            else:
                assert bw <= flat * (1 + 1e-12)        # tiers carry disjoint destination sets in parallel
            w = collective_wire(kind, B, g, d2d=True, package=pk, node=nd)
            assert abs(sum(w.values()) - wire_single(kind, B, g)) / wire_single(kind, B, g) < 1e-12, (g, kind)


def test_pp_partition_legal_and_conserving():
    for mid in ("qwen3-8b", "deepseek-v4-flash", "qwen3-next-80b-a3b", "lfm2.5-24b-a2b", "wan2.1-1.3b"):
        m = C.get_model(mid)
        base = Scenario(model=mid).replace("serving.microbatches", 1).replace("serving.batch", 4)
        f1 = sum(s.flops_u for s in evaluate(base).stages)
        for pp in (2, 3, 7, m.n_layers):
            for split in ("cost", "layers"):
                r = evaluate(base.replace("layout", Layout(pp, 1, 1, 1, 1)).replace("pp_split", split))
                L = [s.layers for s in r.stages]
                assert len(L) == pp and L[0][0] == 0 and L[-1][1] == m.n_layers
                assert all(a < b for a, b in L) and all(L[i][1] == L[i + 1][0] for i in range(pp - 1))
                assert abs(sum(s.flops_u for s in r.stages) / f1 - 1) < 1e-9, (mid, pp, split)
        try:
            evaluate(base.replace("layout", Layout(m.n_layers + 1, 1, 1, 1, 1)))
            raise AssertionError("pp > layers accepted")
        except ValueError:
            pass

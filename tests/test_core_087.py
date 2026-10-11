"""0.87: sparse-attention indexer deployment option (Scenario.indexer) — default = as released (fingerprints
unchanged), work conservation across options, hand-checked op shapes / traffic / memory, attention-DP idle warning."""
from __future__ import annotations

import math
from pathlib import Path

from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, Shard, build_rank_ops
from accel_dse.core.model import has_indexer, with_indexer
from accel_dse.core.parallel import INDEXER_MODES, Layout
from accel_dse.core.scenario import Scenario, Serving

H = "hbm3e_8s_12h24g_9200"


def _ev(mid, mode, tp=8, ph="prefill", **kw):
    sv = Serving(phase=ph, batch=kw.pop("batch", 2), prompt=kw.pop("prompt", 65536), ctx=kw.pop("ctx", 65536))
    return evaluate(Scenario(model=mid, layout=Layout(tp=tp, ep=tp), mem_id=H, node_cards=8, indexer=mode, serving=sv))


def test_default_is_release_and_hash_unchanged():
    assert Scenario(model="deepseek-v3.2").hash() == Scenario(model="deepseek-v3.2", indexer="reference").hash()
    assert "indexer" not in Scenario(model="deepseek-v3.2").to_dict()
    for ph in ("prefill", "decode"):
        assert _ev("deepseek-v3.2", "reference", ph=ph).step == _ev("deepseek-v3.2", "replicated", ph=ph).step
        assert _ev("deepseek-v4-flash", "reference", ph=ph).step == _ev("deepseek-v4-flash", "tp_split", ph=ph).step
    m = get_model("qwen3-8b")
    assert not has_indexer(m) and with_indexer(m, "key_split") is m
    r = evaluate(Scenario(model="qwen3-8b", indexer="key_split"))
    assert any("indexer = key_split 已忽略" in w for w in r.warnings)


def test_work_conservation_across_options():
    for mid in ("deepseek-v3.2", "glm-5"):
        for tp in (1, 2, 8):
            for ph in ("prefill", "decode"):
                R = {m: _ev(mid, m, tp=tp, ph=ph) for m in INDEXER_MODES}
                u = [sum(s.flops_u for s in r.stages) for r in R.values()]
                assert max(u) / min(u) - 1 < 1e-9, (mid, tp, ph, u)
                if tp == 1:
                    assert len({r.step for r in R.values()}) == 1, (mid, ph)


def test_key_split_hand_check():
    m = get_model("deepseek-v3.2")
    c = m.layers[0].core
    P, tp = 1048576, 8
    ops = {mode: {o.name: o for o in build_rank_ops(with_indexer(m, mode), 0, Phase("prefill", 1, q=P, ctx=0), Shard(tp=tp, ep=tp))}
           for mode in ("replicated", "tp_split", "key_split")}
    rp, ts, ks = ops["replicated"]["indexer_score"], ops["tp_split"]["indexer_score"], ops["key_split"]["indexer_score"]
    assert rp.n == P and rp.m == P * c.idx_heads and rp.replicated == tp
    assert ts.m == P * math.ceil(c.idx_heads / tp) and ts.replicated == 1
    assert ks.n == math.ceil(P / tp) and ks.m == P * c.idx_heads and ks.replicated == 1
    assert abs(ks.causal - rp.causal) < 1e-12 and abs(ks.flops * tp / rp.flops - 1) < 1e-12
    # traffic: score all-reduce of fp32 [P × P × causal]; top-k gather of (fp32, int32) × k per token per rank
    assert abs(ops["tp_split"]["indexer_allreduce"].comm_bytes - P * P * rp.causal * 4) < 1e-6 * P * P
    assert ops["key_split"]["indexer_topk_gather"].comm_bytes == P * c.topk * 8
    assert "indexer_allreduce" not in ops["key_split"] and "indexer_topk_gather" not in ops["tp_split"]
    # per-rank indexer cache: key_split holds 1/TP of the index keys (V3.2: 132 B / key / layer)
    a, b = _ev("deepseek-v3.2", "replicated", prompt=P, batch=1), _ev("deepseek-v3.2", "key_split", prompt=P, batch=1)
    saved = a.stages[0].mem.dram_need - b.stages[0].mem.dram_need
    per = c.idx_bytes_per_token(1.0) * len(m.layers)
    assert abs(saved - (P - math.ceil(P / tp)) * per) <= 1e-6 * saved, (saved, per)
    # long prefill: indexer time / TP; head split pays the score all-reduce, key split only the small gather
    assert b.step < a.step / 4 and _ev("deepseek-v3.2", "tp_split", prompt=P, batch=1).step > b.step


def test_key_split_rejected_for_compressed_or_block_indexers():
    for mid in ("deepseek-v4-flash", "qwen3.8-flash-next", "glm-5.3-flash"):
        try:
            _ev(mid, "key_split", ph="decode")
        except ValueError as e:
            assert "key_split" in str(e)
        else:
            raise AssertionError(mid)


def test_attention_dp_idle_warning_and_long_prefill_dp():
    r1 = evaluate(Scenario(model="deepseek-v3.2", layout=Layout(tp=1, dp=8, ep=8), mem_id=H, node_cards=8,
                           serving=Serving(phase="prefill", batch=1, prompt=262144)))
    assert any("注意力 DP 副本空闲" in w for w in r1.warnings)
    r8 = evaluate(Scenario(model="deepseek-v3.2", layout=Layout(tp=1, dp=8, ep=8), mem_id=H, node_cards=8,
                           serving=Serving(phase="prefill", batch=8, prompt=262144)))
    assert not any("注意力 DP 副本空闲" in w for w in r8.warnings) and r8.fits


def test_ui_selector():
    html = (Path(__file__).resolve().parents[1] / "accel_dse/web/index.html").read_text(encoding="utf-8")
    js = (Path(__file__).resolve().parents[1] / "accel_dse/web/app.js").read_text(encoding="utf-8")
    assert 'id="l-indexer"' in html and "indexer 部署" in html
    for v in INDEXER_MODES:
        assert f'<option value="{v}"' in html
    assert "S.sc.indexer = e.target.value" in js

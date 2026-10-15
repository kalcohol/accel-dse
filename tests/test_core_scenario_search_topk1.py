"""0.116 suite-time split of test_core_scenario_search::test_topk_search_equals_full_ranking (cases 1–2; body
unchanged): each module runs in its own process, so the four searches no longer serialise on one worker."""
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.search import search_layouts
from tests.test_core_scenario_search import HBM


def _case(model, obj, org):
    """Cross-layout branch and bound (top=k) returns exactly the head of the full ranking."""
    base = Scenario(model=model, mem_id=HBM, mapping=org, serving=Serving(ctx=4096))
    full = search_layouts(base, 8, objective=obj)
    stats: dict = {}
    top = search_layouts(base, 8, objective=obj, top=3, stats=stats)
    assert [round(r.score(obj), 9) for r in top] == [round(r.score(obj), 9) for r in full[:3]], (model, obj)
    assert top[0].layout == full[0].layout and top[0].batch == full[0].batch
    assert stats["layouts"] == len(full)
    if len(full) >= 10:                     # a 4-layout dense case may legitimately prune nothing
        assert stats["pruned"] > 0


def test_topk_qwen3_32b_decode_os():
    _case('qwen3-32b', 'decode', 'os')


def test_topk_qwen3_32b_goodput_reconf():
    _case('qwen3-32b', 'goodput', 'reconf')

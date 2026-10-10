import sys, collections
sys.path.insert(0, '.')
from accel_dse.core import evaluate as E, extval as X
from accel_dse.core.refhw import REF_HW
log = []
o_s, o_c = E._op_seconds, E._comm
def ops(o, sys_, org, model):
    r = o_s(o, sys_, org, model); log.append(("op", o, r, sys_)); return r
def comm(sys_, kind, b, g, st):
    r = o_c(sys_, kind, b, g, st); log.append(("comm", (kind, b, g), r, sys_)); return r
o_sum = E._sum_ops
def sum_ops(ops_, sys_, org, model, memos=None):
    for o in ops_:
        if o.kind == "comm":
            log.append(("comm", (o.name, o.comm_kind), o_c(sys_, o.comm_kind, o.comm_bytes, o.comm_group, o.comm_stride), sys_))
        else:
            log.append(("op", o, o_s(o, sys_, org, model), sys_))
    return o_sum(ops_, sys_, org, model, memos)
E._sum_ops = sum_ops
row = dict(id="x", hw="H800-SXM", model="deepseek-v3", dtype="native", tp=1, dp=128, ep=128, node_cards=8, net_GBps=50.0)
v = sys.argv[1] if len(sys.argv) > 1 else "catalog_tbo"
if v.endswith("+S"): v = X.split_variant(v[:-2])
scn = X.scenario(row, v, phase="decode", batch=128 * 128, ctx=4096, out_len=2, spec_k=1, spec_accept=0.875)
import accel_dse.core.evaluate as EE
EE._MEMO_OFF = True
r = E.evaluate(scn)
bw = r.stages[0].time.dram_bytes / r.stages[0].time.t_dram
cat = collections.defaultdict(float)
def c(name):
    n = name.lower()
    for k in ("core", "attn_score", "attn_pv", "score", "pv"):
        pass
    return name
seen = set()
for kind, o, res, s in log:
    if kind == "op":
        a, ma, fe, vv = res[:4]
        wb = o.w_params * o.w_bits / 8 + o.kv_read + o.kv_write + o.state_rw
        cat[(o.name, o.kind, o.role)] += max(a, vv, wb / bw) * 1  # per occurrence
    else:
        cat[("comm:" + o[0] + ":" + o[1], "comm", "")] += res[0] + res[1]
print("step tpot ms", r.tpot * 1e3, "stage step ms", r.step * 1e3, "bound", r.bound, "dram GB/s eff", bw / 1e9)
t = r.stages[0].time
print({k: round(getattr(t, k) * 1e3, 2) for k in ("t_array", "t_arr_attn", "t_vector", "t_dram", "t_dram_kv", "t_link", "t_sync")})
for k, val in sorted(cat.items(), key=lambda x: -x[1])[:30]:
    print(f"{val*1e3:9.3f} ms  {k}")
cnt = collections.Counter(o.name for k, o, _, _ in log if k == "op")
print(dict(cnt.most_common(12)))
ex = [o for k, o, _, _ in log if k == "op" and o.name in ("expert.gate_up", "pv_latent", "attn.o")]
for o in ex[:3]: print(o.name, o.m, o.k, o.n, o.count, o.w_params, o.w_bits, o.kv_read, o.replicated, o.share)
for k_, o, res, _ in log:
    if k_ == "op" and o.name in ("qk_latent", "pv_latent", "expert.gate_up", "attn.o"):
        a, ma, fe, vv, ce, idl = res[:6]
        print(o.name, "arr %.4f mac %.4f feed %.4f ideal %.4f ms" % (a*1e3, ma*1e3, fe*1e3, idl*1e3), "m,k,n,count", o.m, o.k, o.n, o.count, o.a_fmt, o.w_fmt)

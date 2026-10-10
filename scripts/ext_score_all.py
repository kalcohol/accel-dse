import sys, math
sys.path.insert(0, '.')
from accel_dse.core import extval as X
def st(rs, v):
    l = []
    for r in rs:
        p = (X.predict_cb(r, v) if r['metric'] == 'max_tok_s_total' else X.predict(r, v))['pred']
        x = p / r['value']; l.append(-math.log(x) if r['metric'] in ('static_latency_s', 'ttft_ms') else math.log(x))
    return '%.2f/%.0f%%' % (math.exp(sum(l) / len(l)), 100 * (math.exp(sum(abs(a) for a in l) / len(l)) - 1))
A = {'v1 static': [r for r in X.rows() if r['metric'] == 'static_tok_s_gpu'], 'v1 TTFT': [r for r in X.rows() if r['metric'] == 'ttft_ms'],
     'v1 max-load': [r for r in X.rows() if r['metric'] == 'max_tok_s_total'],
     'MoE large-EP': [r for r in X.rows(X.DATA_MOE) if r['metric'] in ('prefill_tok_s_node', 'decode_tok_s_node')],
     'Mixtral TP8': [r for r in X.rows(X.DATA_MOE) if r['src'] == 'trtllm-0.17'],
     'MI300X lat': [r for r in X.rows(X.DATA_R3) if r['metric'] == 'static_latency_s'],
     'MI300X max-load': [r for r in X.rows(X.DATA_R3) if r['hw'] == 'MI300X' and r['metric'] == 'max_tok_s_total'],
     'B200 NVFP4': [r for r in X.rows(X.DATA_R3) if r['hw'] == 'B200']}
vs = sys.argv[1].split(',')
vv = []
for v in vs:
    b = v.split('+')[0]; n = b
    if '+S' in v: n = X.split_variant(n)
    for part in v.split('+')[1:]:
        if part.startswith('L'): n = X.overhead_variant(n, float(part[1:]))
    vv.append(n)
print('| group | n | ' + ' | '.join(vv) + ' |')
for g, rs in A.items(): print(f'| {g} | {len(rs)} | ' + ' | '.join(st(rs, v) for v in vv) + ' |')

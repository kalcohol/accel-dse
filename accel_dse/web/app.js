/* accel-dse 0.40 workbench (core v2). Zero dependencies. */
'use strict';

const $ = (id) => document.getElementById(id);
function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'style') e.style.cssText = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat()) if (c !== null && c !== undefined && c !== false) e.append(c.nodeType ? c : String(c));
  return e;
}
function num(x, d = 3) {
  if (x === null || x === undefined || !isFinite(x)) return '—';
  const a = Math.abs(x);
  if (a >= 1e4) return Math.round(x).toLocaleString('en-US');
  if (a >= 100) return x.toFixed(0);
  if (a >= 10) return x.toFixed(1);
  if (a >= 1) return x.toFixed(2);
  if (a === 0) return '0';
  return x.toPrecision(Math.max(1, d - 1));
}
function put(e, ...kids) {
  e.replaceChildren(...kids.flat().filter((k) => k !== null && k !== undefined && k !== false));
}
const pct = (x) => (x === null || x === undefined || !isFinite(x)) ? '—' : (x * 100).toFixed(x < 0.1 ? 1 : 0) + '%';

const SECTION_ZH = { user_requested: '用户指定', recommended_extra: '推荐补充', official_quant: '官方量化版', reference: '参考模型' };
const COVER_ZH = { full: '完整 full', partial: '部分 partial', proxy: '「架构代理」' };
const PROV_ZH = { official: '官方 official', mirror: '镜像 mirror' };
const BOUND_ZH = { MAC: 'MAC 算力', FEED: 'FEED 供数', VECTOR: 'VECTOR 向量', DRAM: 'DRAM 带宽', LINK: 'LINK 互连', SYNC: 'SYNC 同步' };
const COMPONENTS = ['mac', 'feed', 'vector', 'dram', 'link', 'sync'];
const MAP_DESC = {
  os: '输出驻留：权重与激活都从 SRAM 流入，供数端口常是小 M 时的瓶颈。',
  ws_edge: '权重驻留，边缘逐列加载权重（K-split）；装载慢，适合大 M。',
  ws_broad: '权重驻留，宽面广播加载；装载快，小 M 也能复用阵列。',
  os_vec: 'OS 阵列 + 独立 GEMV 单元，逐算子取较快者（小 M decode 友好）。',
  reconf: '可重构：逐算子在全部组织中取最快（上界参考）。',
};

/* ------------------------------------------------------------------ state */
const S = {
  cat: null, models: [], byId: {}, preset: '100T', chipOver: {}, sc: null,
  mem: {}, memInfo: null, wiW: '', wiKV: '', best: false, last: null,
  cmpObj: 'decode', layObj: 'decode', tab: 'eval',
};

/* ------------------------------------------------------------------ requests (sequenced: stale responses dropped) */
const seq = {}, ctl = {};
async function req(chan, path, body) {
  const my = (seq[chan] = (seq[chan] || 0) + 1);
  if (ctl[chan]) ctl[chan].abort();
  const c = (ctl[chan] = new AbortController());
  const opt = body === undefined ? { signal: c.signal }
    : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: c.signal };
  let r, j;
  try {
    r = await fetch(path, opt);
    j = await r.json();
  } catch (e) {
    if (e.name === 'AbortError' || my !== seq[chan]) return null;
    throw e;
  }
  if (my !== seq[chan]) return null;
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}
let busy = 0;
function status(msg, cls = '') {
  const s = $('status');
  s.textContent = msg;
  s.className = 'status ' + cls;
}
async function track(p) {
  busy++;
  status('计算中…', 'busy');
  try { return await p; } finally { busy--; if (!busy && $('status').classList.contains('busy')) status('就绪'); }
}

/* ------------------------------------------------------------------ scenario */
const model = () => S.byId[S.sc.model];
function overrides() {
  const fo = [];
  if (S.wiW) for (const r of Object.keys(model().roles)) if (!['embed', 'lm_head', 'router'].includes(r)) fo.push([r, S.wiW]);
  if (S.wiKV) fo.push(['kv', S.wiKV]);
  return fo;
}
function body(extra = {}) {
  const sc = JSON.parse(JSON.stringify(S.sc));
  sc.chip = { ...S.chipOver };
  sc.formats_override = overrides();
  if (S.memInfo) sc.mem_id = S.memInfo.id;
  return { chip_preset: S.preset, scenario: sc, ...extra };
}

/* ------------------------------------------------------------------ inputs */
function readNum(inp, int) {
  const v = inp.value.trim();
  if (v === '') return null;
  const x = Number(v);
  const lo = inp.min !== '' ? Number(inp.min) : -Infinity, hi = inp.max !== '' ? Number(inp.max) : Infinity;
  if (!isFinite(x) || x < lo || x > hi || (int && !Number.isInteger(x))) return undefined;
  return x;
}
function bindNumber(id, get, set, { int = false, nullable = false } = {}) {
  const inp = $(id);
  inp.addEventListener('input', () => {
    const x = readNum(inp, int);
    const ok = x !== undefined && (x !== null || nullable);
    inp.classList.toggle('bad', !ok);
    if (!ok) return;
    set(x);
    if (inp.dataset.after) AFTER[inp.dataset.after]();
    schedule();
  });
  inp._sync = () => { const v = get(); inp.value = v === null || v === undefined ? '' : v; inp.classList.remove('bad'); };
}
const AFTER = {};
function syncInputs() { document.querySelectorAll('input[type=number]').forEach((i) => i._sync && i._sync()); }

function seg(id, items, get, set) {
  const box = $(id);
  if (items) put(box, ...items.map(([v, label]) => h('button', { 'data-v': v }, label)));
  box.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    set(b.dataset.v);
    paintSeg(id, get());
  });
  paintSeg(id, get());
}
function paintSeg(id, v) { $(id).querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.v === v)); }

/* ------------------------------------------------------------------ model */
function fillModels() {
  const sel = $('model');
  const groups = {};
  for (const m of S.models) (groups[m.section] = groups[m.section] || []).push(m);
  put(sel, ...Object.keys(SECTION_ZH).filter((k) => groups[k]).map((k) =>
    h('optgroup', { label: SECTION_ZH[k] }, groups[k].map((m) =>
      h('option', { value: m.id }, `${m.label}${m.proxy_badge ? ' 「架构代理」' : m.coverage === 'partial' ? ' (部分)' : ''}`)))));
  sel.value = S.sc.model;
  sel.addEventListener('change', () => { S.sc.model = sel.value; onModel(); schedule(); });
}
function badges(m, whatIf) {
  return [
    h('span', { class: 'badge ' + m.provenance }, h('span', { class: 'ax' }, '来源'), PROV_ZH[m.provenance] || m.provenance),
    h('span', { class: 'badge ' + m.coverage }, h('span', { class: 'ax' }, '覆盖'), COVER_ZH[m.coverage] || m.coverage),
    h('span', { class: 'badge' }, h('span', { class: 'ax' }, 'dtype'), m.dtype),
    whatIf ? h('span', { class: 'badge wi' }, 'what-if dtype') : null,
  ].filter(Boolean);
}
function onModel() {
  const m = model();
  put($('model-badges'), ...badges(m, S.wiW || S.wiKV));
  put($('model-facts'), 
    h('span', {}, '参数 ', h('b', {}, num(m.params_B) + 'B')),
    h('span', {}, '激活 ', h('b', {}, num(m.active_B) + 'B')),
    h('span', {}, '层 ', h('b', {}, m.n_layers)),
    h('span', {}, m.arch),
    m.mtp_layers ? h('span', {}, `MTP ×${m.mtp_layers}`) : null,
    h('span', { class: 'muted', title: '与发布 safetensors 头统计值的偏差' }, `vs 发布 ${(m.param_err * 100).toFixed(2)}%`));
  put($('model-notes'), ...m.notes.map((n) => h('li', {}, n)));
  $('model-notes-wrap').hidden = !m.notes.length;
  document.querySelectorAll('.moe-only').forEach((e) => (e.hidden = !m.is_moe));
  const L = S.sc.layout;
  if (!m.is_moe) { L.dp = 1; L.ep = 1; L.etp = 1; } else fixMoe('tp');
  if (L.pp > m.n_layers) L.pp = 1;
  syncInputs();
  cardsNote();
}
function fixMoe(changed) {
  const L = S.sc.layout;
  if (!model().is_moe) return;
  const want = L.tp * L.dp;
  if (changed === 'etp') { if (want % L.etp === 0) { L.ep = want / L.etp; return; } }
  if (want % L.ep === 0) L.etp = want / L.ep; else { L.ep = want; L.etp = 1; }
}
function cardsNote() {
  const L = S.sc.layout;
  $('cards-note').textContent = `${L.pp * L.tp * L.dp} 卡 / 副本`;
}

/* ------------------------------------------------------------------ chip */
function chipVal(k) { return k in S.chipOver ? S.chipOver[k] : S.cat.chips[S.preset][k]; }
function onPreset() {
  const c = S.cat.chips[S.preset];
  put($('chip-facts'), 
    h('span', {}, '阵列 ', h('b', {}, `${c.rows}×${c.cols}×${c.engines}`)),
    h('span', {}, 'bf16 峰值 ', h('b', {}, num(c.peak_tflops_bf16 * (chipVal('freq_ghz') / c.freq_ghz)) + ' TFLOPS')),
    h('span', {}, 'SRAM 端口 ', h('b', {}, num(c.port_GBps / c.freq_ghz) + ' B/cycle'), ' (默认)'));
  $('c-sram_port_Bpc').placeholder = `默认 ${num(c.port_GBps / c.freq_ghz)}`;
  $('c-gemv_macs').placeholder = `默认 ${c.gemv}`;
  $('chip-formats').textContent = '原生格式 × 速率：' + c.formats.map(([f, r]) => `${f}×${r}`).join(' · ') +
    '；其余格式反量化 / 上转换，在向量单元计入开销。';
}

/* ------------------------------------------------------------------ memory */
const memType = () => S.cat.memory.types.find((t) => t.id === S.mem.type);
const memForm = () => memType().forms.find((f) => f.id === S.mem.form) || memType().forms[0];
function tagSuffix(tag) { return tag && tag !== 'jedec' && tag !== 'vendor_shipping' ? ` (${S.cat.memory.tag_zh[tag] || tag})` : ''; }
function opts(sel, list, value) {
  put(sel, ...list.map(([v, l]) => h('option', { value: v }, l)));
  sel.value = String(value);
  if (sel.value !== String(value) && list.length) sel.value = String(list[0][0]);
}
function fillMem() {
  const t = memType(), f = memForm();
  opts($('m-type'), S.cat.memory.types.map((x) => [x.id, x.id]), S.mem.type);
  opts($('m-form'), t.forms.map((x) => [x.id, x.label]), f.id);
  opts($('m-width'), f.widths.map((w) => [w.bits, w.bits + ' bit' + tagSuffix(w.tag)]), S.mem.width);
  opts($('m-rate'), f.rates.map((r) => [r.MTps, r.MTps + tagSuffix(r.tag)]), S.mem.rate);
  opts($('m-count'), f.counts.map((c) => [c.n, `${c.n} × ${f.unit === 'stack' ? '堆栈' : '颗'}` + tagSuffix(c.tag)]), S.mem.count);
  const hbm = t.kind === 'HBM';
  $('m-cap-wrap').hidden = hbm;
  $('m-height-wrap').hidden = !hbm;
  if (hbm) opts($('m-height'), f.cap_tags.map((c) => [`${c.height}:${c.die_Gb}`, `${c.height}-Hi ${c.die_Gb}Gb · ${c.GB} GB` + tagSuffix(c.tag)]), `${S.mem.height}:${S.mem.die}`);
  else opts($('m-cap'), (f.caps[String(S.mem.width)] || []).map((c) => [c.GB, c.GB + ' GB' + tagSuffix(c.tag)]), S.mem.cap);
  // read back the (possibly clamped) selection
  S.mem.form = $('m-form').value; S.mem.width = +$('m-width').value; S.mem.rate = +$('m-rate').value; S.mem.count = +$('m-count').value;
  if (hbm) [S.mem.height, S.mem.die] = $('m-height').value.split(':').map(Number); else S.mem.cap = +$('m-cap').value;
}
function memDefaults(level) {
  const t = memType();
  if (level <= 0) S.mem.form = t.default_form;
  const f = memForm();
  if (level <= 1) {
    S.mem.width = f.default_width; S.mem.rate = f.default_rate; S.mem.count = f.default_count;
    if (t.kind === 'HBM') { S.mem.height = f.default_height; S.mem.die = f.default_density; }
  }
  if (t.kind !== 'HBM' && level <= 2) S.mem.cap = (f.default_cap || {})[String(S.mem.width)];
}
async function resolveMem() {
  const t = memType();
  const b = { type: S.mem.type, form: S.mem.form, width: S.mem.width, rate: S.mem.rate, count: S.mem.count };
  if (t.kind === 'HBM') { b.height = S.mem.height; b.die = S.mem.die; } else b.cap = S.mem.cap;
  try {
    const r = await req('mem', '/api/memory', b);
    if (!r) return;
    S.memInfo = r;
    paintMem();
    schedule();
  } catch (e) { showErr('存储器：' + e.message); }
}
function paintMem() {
  const r = S.memInfo;
  put($('mem-facts'), 
    h('span', {}, '带宽 ', h('b', {}, num(r.raw_GBps) + ' GB/s'), ' 原始'),
    h('span', {}, '容量 ', h('b', {}, num(r.capacity_GiB) + ' GiB')),
    h('span', { class: 'mono muted' }, r.id),
    h('span', {}, r.tag_zh),
    ...r.warnings.map((w) => h('span', { style: 'color:var(--warn)' }, w)));
  $('mem_eff').placeholder = String(r.efficiency);
}
function bindMem() {
  const on = (id, fn) => $(id).addEventListener('change', () => { fn($(id).value); fillMem(); resolveMem(); });
  on('m-type', (v) => { S.mem.type = v; memDefaults(0); });
  on('m-form', (v) => { S.mem.form = v; memDefaults(1); });
  on('m-width', (v) => { S.mem.width = +v; memDefaults(2); });
  on('m-rate', (v) => { S.mem.rate = +v; });
  on('m-count', (v) => { S.mem.count = +v; });
  on('m-cap', (v) => { S.mem.cap = +v; });
  on('m-height', (v) => { [S.mem.height, S.mem.die] = v.split(':').map(Number); });
}

/* ------------------------------------------------------------------ evaluation */
let timer = null;
function schedule() { clearTimeout(timer); timer = setTimeout(runEval, 120); invalidate(); }
function invalidate() {
  for (const id of ['cmp-status', 'lay-status', 'sw-status'])
    if ($(id).dataset.done) $(id).textContent = '场景已改变，需重新运行';
}
function showErr(msg) {
  put($('warns'), h('div', { class: 'err' }, msg));
  status('错误', 'err');
}
async function runEval() {
  const t0 = performance.now();
  try {
    const r = await track(req('eval', '/api/eval', body({ best_batch: S.best })));
    if (!r) return;
    S.last = r;
    renderEval(r);
    status(`就绪 · ${Math.round(performance.now() - t0)} ms`);
  } catch (e) { showErr(e.message); }
}
function kpi(label, value, sub, cls = '') {
  return h('div', { class: 'kpi ' + cls }, h('div', { class: 'l' }, label), h('div', { class: 'v' }, value), h('div', { class: 's' }, sub));
}
function boundTag(b) { return b ? h('span', { class: 'bound ' + b, title: BOUND_ZH[b] }, b) : '—'; }
function ttftFlag(ok, ms) {
  if (ok === null || ok === undefined) return h('span', { class: 'flag pend' }, 'n/a');
  return h('span', { class: 'flag ' + (ok ? 'ok' : 'no'), title: 'DP 组内 prefill 的 TTFT 与 SLO 比较' },
    (ok ? '✓ ' : '✗ ') + (ms === undefined ? '' : ms === null ? '不可行' : num(ms) + ' ms'));
}
function renderEval(r) {
  const s = r.summary, sv = r.scenario.serving, g = r.goodput;
  const m = model();
  put($('model-badges'), ...badges(m, r.model.what_if));
  const k = [];
  if (s.phase === 'decode') {
    k.push(kpi('TPOT', num(s.tpot_ms) + ' ms', `SLO ${sv.tpot_slo_ms} ms`, s.tpot_ms > sv.tpot_slo_ms ? 'bad' : ''));
    k.push(kpi('吞吐 / 卡', num(s.tok_s_card) + ' tok/s', `共 ${num(s.tok_s)} tok/s · ${s.cards} 卡`));
  } else {
    k.push(kpi('TTFT', num(s.ttft_ms) + ' ms', `SLO ${sv.ttft_slo_ms} ms · prompt ${sv.prompt}`, s.ttft_ms > sv.ttft_slo_ms ? 'bad' : ''));
    k.push(kpi('Prefill 吞吐 / 卡', num(s.tok_s_card) + ' tok/s', `共 ${num(s.tok_s)} tok/s`));
  }
  k.push(kpi('batch' + (S.best ? '（自动）' : ''), String(s.batch), `${s.layout} · ${s.mapping}`));
  k.push(h('div', { class: 'kpi' }, h('div', { class: 'l' }, '绑定瓶颈'), h('div', { class: 'v' }, boundTag(s.bound)),
    h('div', { class: 's' }, `${BOUND_ZH[s.bound] || ''} · 最重 stage ${s.heaviest_stage}`)));
  k.push(kpi('有效 MAC 比例', pct(s.array_util), '理想 MAC 时间 / 阵列时间（array_util）'));
  if (g) k.push(h('div', { class: 'kpi ' + (g.ttft_ok ? '' : 'bad') }, h('div', { class: 'l' }, 'Goodput / 卡（含 prefill）'),
    h('div', { class: 'v' }, num(g.tok_s_card) + ' tok/s'),
    h('div', { class: 's' }, 'DP prefill TTFT ', ttftFlag(g.ttft_ok, g.ttft_ms), ` · prefill batch ${g.prefill_batch}`)));
  k.push(kpi('DRAM 需求 / 卡', num(s.dram_need_GiB) + ' GiB', s.fits ? `容量 ${num(S.memInfo ? S.memInfo.capacity_GiB : NaN)} GiB · 可放下` : '超出容量', s.fits ? '' : 'bad'));
  put($('kpis'), ...k);
  put($('warns'), ...s.warnings.map((w) => h('div', {}, w)),
    ...(r.model.what_if ? [h('div', {}, 'what-if：dtype 已偏离官方发布，结果仅供假设分析。')] : []));
  renderStages(r);
  renderAssumptions(r);
}
function renderStages(r) {
  const head = h('tr', {}, h('th', { class: 'l' }, 'stage'), h('th', { class: 'l' }, '层'), h('th', {}, '瓶颈'),
    ...COMPONENTS.map((c) => h('th', {}, c.toUpperCase() + ' ms')), h('th', {}, '合计 ms'), h('th', { class: 'l' }, '有效 MAC'),
    h('th', {}, 'TFLOP'));
  const rows = r.stages.map((st) => {
    const comp = { mac: st.t_ms.mac, feed: st.t_ms.feed, vector: st.t_ms.vector, dram: st.t_ms.dram, link: st.t_ms.link, sync: st.t_ms.sync };
    return h('tr', { class: st.index === r.summary.heaviest_stage ? 'best' : '' },
      h('td', { class: 'l' }, st.index), h('td', { class: 'l' }, `${st.layers[0]}–${st.layers[1] - 1}`), h('td', {}, boundTag(st.bound)),
      ...COMPONENTS.map((c) => h('td', { style: c.toUpperCase() === st.bound ? 'color:var(--b-' + c + ');font-weight:700' : '' }, num(comp[c]))),
      h('td', {}, num(st.t_ms.total)),
      h('td', { class: 'l' }, h('span', { class: 'ubar' }, h('i', { style: `width:${Math.min(100, st.array_util * 100)}%` })), pct(st.array_util)),
      h('td', {}, num(st.tflops)));
  });
  put($('stage-tbl'), h('thead', {}, head), h('tbody', {}, rows));
  put($('legend'), ...COMPONENTS.map((c) => h('span', { style: `--c:var(--b-${c})` }, BOUND_ZH[c.toUpperCase()])),
    h('span', { style: '--c:transparent' }, 'MAC / FEED 为逐算子 max(MAC, FEED) 前的分项和'));
  const mh = h('tr', {}, h('th', { class: 'l' }, 'stage'), h('th', {}, '权重 GiB'), h('th', {}, 'KV GiB'), h('th', {}, '状态 GiB'),
    h('th', {}, '需求 / 容量 GiB'), h('th', {}, 'SRAM 驻留'), h('th', {}, 'KV in SRAM MiB'), h('th', {}, 'staging MiB'),
    h('th', {}, 'DRAM 流量 / step GB'), h('th', {}, '反量化 M elem'));
  const mr = r.stages.map((st) => h('tr', {}, h('td', { class: 'l' }, st.index), h('td', {}, num(st.mem.stored_w_GiB)),
    h('td', {}, num(st.mem.kv_GiB)), h('td', {}, num(st.mem.state_GiB)),
    h('td', { style: st.mem.fits ? '' : 'color:var(--danger)' }, `${num(st.mem.need_GiB)} / ${num(st.mem.cap_GiB)}`),
    h('td', {}, pct(st.mem.residency)), h('td', {}, num(st.mem.kv_sram_MiB)), h('td', {}, num(st.mem.staging_MiB)),
    h('td', {}, num(st.dram_GB.total)), h('td', {}, num(st.convert_Melems))));
  put($('mem-tbl'), h('thead', {}, mh), h('tbody', {}, mr));
}
function renderAssumptions(r) {
  const c = r.scenario.chip, cc = S.cat.chips[S.preset], sc = r.scenario;
  const port = c.sram_port_Bpc || cc.port_GBps / cc.freq_ghz;
  const items = [
    `芯片 ${c.name}：${c.rows}×${c.cols}×${c.engines} @ ${c.freq_ghz} GHz，MAC 效率 ${c.mac_eff}`,
    `SRAM ${c.sram_mib} MiB，端口 ${num(port)} B/cycle${c.sram_port_Bpc ? '' : '（默认 4·(R+C·E)·2）'}`,
    `GEMV 单元 ${c.gemv_macs || cc.gemv} MAC/cycle${c.gemv_macs ? '' : '（默认 = 阵列 MAC / 8）'}；向量 lanes ${c.vector_lanes || cc.lanes}`,
    `累加器 ${c.acc_kib} KiB（超出的部分和行溢出到 SRAM）`,
    `DRAM 效率 ${sc.mem_eff ?? (S.memInfo ? S.memInfo.efficiency : 0.7)}；预留 1 GiB；staging = max(2 MiB, 2·最大激活)`,
    `链路 ${sc.link.GBps} GB/s，每次集合通信同步 α = ${sc.link.alpha_us} µs（${sc.link.topology}）`,
    `投机解码：k = ${sc.serving.spec_k}，接受率 ${sc.serving.spec_accept}（期望 token = (1−a^(k+1))/(1−a)）`,
    'MoE：每 rank 命中专家数取 max(局部期望, 全局期望/EP)；token 均匀路由',
    '芯片不支持的权重格式：反量化每元素 2 次向量操作',
  ];
  put($('assume-list'), ...items.map((t) => h('li', {}, t)));
}

/* ------------------------------------------------------------------ mapping comparison */
function caseText(c) {
  const v = c.same ? 'top-1 不变' : c.within5 ? `top-1 → ${c.top}，原布局差距 ≤5%` : `top-1 → ${c.top}（原布局 ${num(c.base_top_tok_s_card)} vs ${num(c.top_tok_s_card)}）`;
  return `${c.case}：${v}`;
}
function stabFlag(st) {
  if (!st) return h('span', { class: 'flag pend' }, '…');
  if (st.error) return h('span', { class: 'flag no', title: st.error }, '错误');
  const n = st.cases.length;
  return h('span', { class: 'flag ' + (st.stable ? 'ok' : 'no'), title: st.rule + '\n' + st.cases.map(caseText).join('\n') },
    (st.stable ? '✓ 稳定 ' : '⚠ 不稳定 ') + `${Math.round(st.agree * n)}/${n}`);
}
async function runCompare() {
  const cards = readNum($('cmp-cards'), true);
  if (!cards) { $('cmp-cards').classList.add('bad'); return; }
  const btn = $('cmp-run');
  btn.disabled = true;
  $('cmp-status').textContent = '搜索中（5 种映射 × 全部布局）…';
  delete $('cmp-status').dataset.done;
  const t0 = performance.now();
  try {
    const b = body({ cards, objective: S.cmpObj });
    const r = await track(req('compare', '/api/compare', b));
    if (!r) return;
    const mySeq = seq.compare;
    const stab = {};
    paintCompare(r, stab);
    $('cmp-status').textContent = `完成 · ${((performance.now() - t0) / 1000).toFixed(1)} s · 稳定性计算中…`;
    $('cmp-status').dataset.done = '1';
    for (const row of r.rows) {
      const bb = JSON.parse(JSON.stringify(b));
      bb.scenario.mapping = row.mapping;
      try {
        const st = await track(req('stab-' + row.mapping, '/api/stability', bb));
        if (st === null || seq.compare !== mySeq) return;
        stab[row.mapping] = st;
      } catch (e) { stab[row.mapping] = { error: e.message }; }
      if (seq.compare !== mySeq) return;
      paintCompare(r, stab);
    }
    $('cmp-status').textContent = `完成 · ${((performance.now() - t0) / 1000).toFixed(1)} s`;
  } catch (e) { $('cmp-status').textContent = '错误：' + e.message; }
  finally { btn.disabled = false; }
}
function paintCompare(r, stab) {
  const gp = r.objective === 'goodput';
  const head = h('tr', {}, h('th', { class: 'l' }, '映射组织'), h('th', { class: 'l' }, '最佳布局'), h('th', {}, 'batch'),
    h('th', {}, 'tok/s/卡'), h('th', {}, 'goodput/卡'), h('th', {}, 'TPOT ms'), h('th', {}, '瓶颈'), h('th', { class: 'l' }, '有效 MAC'),
    h('th', {}, 'DP prefill TTFT'), h('th', {}, '排名稳定性'), h('th', { class: 'l' }, '次优布局'));
  const rows = r.rows.map((x) => h('tr', { class: 'click ' + (x.mapping === r.best_mapping ? 'best' : ''), title: '点击应用到场景',
    onclick: () => applyRow(x) },
    h('td', { class: 'l' }, x.label), h('td', { class: 'l mono' }, x.layout), h('td', {}, x.batch),
    h('td', { style: gp ? '' : 'font-weight:700' }, num(x.tok_s_card)),
    h('td', { style: gp ? 'font-weight:700' : '' }, num(x.goodput_card)), h('td', {}, num(x.tpot_ms)), h('td', {}, boundTag(x.bound)),
    h('td', { class: 'l' }, h('span', { class: 'ubar' }, h('i', { style: `width:${Math.min(100, (x.array_util || 0) * 100)}%` })), pct(x.array_util)),
    h('td', {}, ttftFlag(x.ttft_ok, x.ttft_ms)), h('td', {}, stabFlag(stab[x.mapping])),
    h('td', { class: 'l mono muted' }, x.runner_up ? `${x.runner_up.layout} (${num(x.runner_up.score)})` : '—')));
  put($('cmp-tbl'), h('thead', {}, head), h('tbody', {}, rows));
}
function applyRow(x) {
  S.sc.mapping = x.mapping;
  paintSeg('mapping', x.mapping);
  $('mapping-desc').textContent = MAP_DESC[x.mapping];
  Object.assign(S.sc.layout, x.layout_obj);
  S.sc.serving.batch = x.batch;
  S.best = false; $('best-batch').checked = false;
  syncInputs(); cardsNote();
  activate('eval');
  schedule();
}

/* ------------------------------------------------------------------ layouts + stability */
async function runLayouts() {
  const cards = readNum($('lay-cards'), true);
  if (!cards) { $('lay-cards').classList.add('bad'); return; }
  $('lay-status').textContent = '搜索中…';
  delete $('lay-status').dataset.done;
  const t0 = performance.now();
  try {
    const r = await track(req('layouts', '/api/layouts', body({ cards, objective: S.layObj })));
    if (!r) return;
    const head = h('tr', {}, h('th', {}, '#'), h('th', { class: 'l' }, '布局'), h('th', {}, 'batch'), h('th', {}, 'tok/s/卡'),
      h('th', {}, 'goodput/卡'), h('th', {}, 'TPOT ms'), h('th', {}, '瓶颈'), h('th', { class: 'l' }, '有效 MAC'), h('th', {}, 'TTFT'));
    const rows = r.rows.map((x, i) => h('tr', { class: 'click ' + (i === 0 ? 'best' : ''), onclick: () => applyRow(x) },
      h('td', {}, i + 1), h('td', { class: 'l mono' }, x.layout), h('td', {}, x.batch), h('td', {}, num(x.tok_s_card)),
      h('td', {}, num(x.goodput_card)), h('td', {}, num(x.tpot_ms)), h('td', {}, boundTag(x.bound)),
      h('td', { class: 'l' }, h('span', { class: 'ubar' }, h('i', { style: `width:${Math.min(100, (x.array_util || 0) * 100)}%` })), pct(x.array_util)),
      h('td', {}, x.ttft_ok === undefined ? '—' : ttftFlag(x.ttft_ok, x.ttft_ms))));
    put($('lay-tbl'), h('thead', {}, head), h('tbody', {}, rows));
    $('lay-status').textContent = `${r.n_layouts} 个布局 · 显示前 ${r.rows.length} · ${((performance.now() - t0) / 1000).toFixed(1)} s`;
    $('lay-status').dataset.done = '1';
  } catch (e) { $('lay-status').textContent = '错误：' + e.message; }
}
async function runStability() {
  const cards = readNum($('lay-cards'), true);
  if (!cards) return;
  put($('stab-box'), h('div', { class: 'stab muted' }, '稳定性计算中（每个扰动重新搜索全部布局）…'));
  try {
    const r = await track(req('stab', '/api/stability', body({ cards, objective: S.layObj, include_mapping: false })));
    if (!r) return;
    put($('stab-box'), h('div', { class: 'stab' },
      h('div', {}, '基准 top-1：', h('b', { class: 'mono' }, r.base_top), '  ', stabFlag(r), '  ', h('span', { class: 'muted' }, r.rule)),
      h('ul', {}, r.cases.map((c) => h('li', { style: c.same || c.within5 ? '' : 'color:var(--warn)' }, caseText(c))))));
  } catch (e) { put($('stab-box'), h('div', { class: 'stab' }, '错误：' + e.message)); }
}

/* ------------------------------------------------------------------ sweep + pareto */
const SWEEP_ZH = {
  'serving.batch': 'batch', 'serving.ctx': '上下文 ctx', 'serving.prompt': 'prompt 长度', 'serving.spec_k': '投机 k',
  'chip.sram_mib': 'SRAM MiB', 'chip.sram_port_Bpc': 'SRAM 端口 B/cycle', 'chip.freq_ghz': '频率 GHz', 'chip.mac_eff': 'MAC 效率',
  'chip.gemv_macs': 'GEMV MAC/cycle', mem_eff: 'DRAM 效率', 'link.GBps': '链路 GB/s', 'link.alpha_us': '同步 α µs',
};
const SWEEP_DEFAULT = {
  'serving.batch': '1,2,4,8,16,32,64', 'serving.ctx': '1024,4096,16384,32768,131072', 'serving.prompt': '512,2048,8192,32768',
  'serving.spec_k': '0,1,2,3,4,6,8', 'chip.sram_mib': '16,32,64,128,256,512', 'chip.sram_port_Bpc': '2048,4096,7616,16384,32768',
  'chip.freq_ghz': '0.6,0.8,1,1.2,1.5', 'chip.mac_eff': '0.5,0.6,0.7,0.8,0.9,1', 'chip.gemv_macs': '1024,4096,12544,50176',
  mem_eff: '0.5,0.6,0.7,0.8,0.9', 'link.GBps': '50,100,200,400,900', 'link.alpha_us': '0,1,3,5,10',
};
function line(points, { xl, yl, y2l, log }) {
  if (!points.length) return h('div', { class: 'empty' }, '无数据');
  const W = 760, H = 270, P = { l: 56, r: 56, t: 30, b: 36 };
  const xs = points.map((p) => p.x), fx = log ? Math.log2 : (v) => v;
  const x0 = Math.min(...xs.map(fx)), x1 = Math.max(...xs.map(fx));
  const y1 = Math.max(...points.map((p) => p.y)) * 1.08 || 1;
  const y21 = y2l ? Math.max(...points.map((p) => p.y2)) * 1.08 || 1 : 1;
  const X = (v) => P.l + (x1 === x0 ? 0.5 : (fx(v) - x0) / (x1 - x0)) * (W - P.l - P.r);
  const Y = (v, m) => H - P.b - (v / m) * (H - P.t - P.b);
  const ns = 'http://www.w3.org/2000/svg';
  const s = (tag, a, txt) => { const e = document.createElementNS(ns, tag); for (const k in a) e.setAttribute(k, a[k]); if (txt !== undefined) e.textContent = txt; return e; };
  const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });
  svg.append(s('line', { x1: P.l, y1: H - P.b, x2: W - P.r, y2: H - P.b, stroke: '#2e3f55' }));
  for (let i = 0; i <= 4; i++) {
    const yv = (y1 * i) / 4;
    svg.append(s('line', { x1: P.l, x2: W - P.r, y1: Y(yv, y1), y2: Y(yv, y1), stroke: '#1f2b3b' }));
    svg.append(s('text', { x: P.l - 6, y: Y(yv, y1) + 4, 'text-anchor': 'end', fill: '#34d399', 'font-size': 11 }, num(yv)));
    if (y2l) svg.append(s('text', { x: W - P.r + 6, y: Y((y21 * i) / 4, y21) + 4, fill: '#f59e0b', 'font-size': 11 }, num((y21 * i) / 4)));
  }
  for (const p of points) svg.append(s('text', { x: X(p.x), y: H - P.b + 16, 'text-anchor': 'middle', fill: '#8fa3b8', 'font-size': 11 }, p.xlabel ?? num(p.x)));
  svg.append(s('text', { x: (W) / 2, y: H - 4, 'text-anchor': 'middle', fill: '#8fa3b8', 'font-size': 11 }, xl));
  svg.append(s('text', { x: 4, y: 11, fill: '#34d399', 'font-size': 11 }, yl));
  if (y2l) svg.append(s('text', { x: W - 4, y: 11, 'text-anchor': 'end', fill: '#f59e0b', 'font-size': 11 }, y2l));
  const path = (key, m, col) => {
    svg.append(s('polyline', { points: points.map((p) => `${X(p.x)},${Y(p[key], m)}`).join(' '), fill: 'none', stroke: col, 'stroke-width': 2 }));
    for (const p of points) svg.append(s('circle', { cx: X(p.x), cy: Y(p[key], m), r: 3, fill: col }));
  };
  path('y', y1, '#34d399');
  if (y2l) path('y2', y21, '#f59e0b');
  return svg;
}
async function runSweep() {
  const path = $('sw-path').value;
  const vals = $('sw-values').value.split(/[,\s]+/).filter(Boolean).map(Number);
  if (!vals.length || vals.some((v) => !isFinite(v)) || vals.length > 32) { $('sw-values').classList.add('bad'); return; }
  $('sw-values').classList.remove('bad');
  $('sw-status').textContent = '计算中…';
  try {
    const r = await track(req('sweep', '/api/sweep', body({ path, values: vals })));
    if (!r) return;
    const decode = S.sc.serving.phase === 'decode';
    put($('sw-chart'), line(r.rows.map((x) => ({ x: x.value, y: x.tok_s_card, y2: decode ? x.tpot_ms : x.ttft_ms })),
      { xl: SWEEP_ZH[path] || path, yl: 'tok/s/卡', y2l: decode ? 'TPOT ms' : 'TTFT ms', log: vals.every((v) => v > 0) && Math.max(...vals) / Math.min(...vals) >= 16 }));
    const head = h('tr', {}, h('th', { class: 'l' }, SWEEP_ZH[path] || path), h('th', {}, 'TPOT ms'), h('th', {}, decode ? 'TTFT ms (n/a)' : 'TTFT ms'),
      h('th', {}, 'tok/s'), h('th', {}, 'tok/s/卡'), h('th', {}, '瓶颈'), h('th', {}, '有效 MAC'), h('th', {}, '放得下'));
    put($('sw-tbl'), h('thead', {}, head), h('tbody', {}, r.rows.map((x) => h('tr', {},
      h('td', { class: 'l' }, x.value), h('td', {}, num(x.tpot_ms)), h('td', {}, num(x.ttft_ms)), h('td', {}, num(x.tok_s)),
      h('td', {}, num(x.tok_s_card)), h('td', {}, boundTag(x.bound)), h('td', {}, pct(x.array_util)),
      h('td', {}, x.fits ? '✓' : h('span', { style: 'color:var(--danger)' }, '✗'))))));
    $('sw-status').textContent = `${r.rows.length} 点`;
    $('sw-status').dataset.done = '1';
  } catch (e) { $('sw-status').textContent = '错误：' + e.message; }
}
async function runPareto() {
  $('sw-status').textContent = '计算 Pareto…';
  try {
    const r = await track(req('sweep', '/api/pareto', body()));
    if (!r) return;
    const pts = r.front.map((p) => ({ x: p.batch, xlabel: 'B' + p.batch, y: p.tok_s_card, y2: p.tpot_ms }));
    put($('sw-chart'), line(pts, { xl: `batch（布局 ${r.layout}，Pareto 前沿：TPOT ↑ 换吞吐 ↑）`, yl: 'tok/s/卡', y2l: 'TPOT ms', log: true }));
    const head = h('tr', {}, h('th', { class: 'l' }, 'batch'), h('th', {}, 'TPOT ms'), h('th', {}, 'tok/s/卡'));
    put($('sw-tbl'), h('thead', {}, head), h('tbody', {}, r.front.map((p) => h('tr', {},
      h('td', { class: 'l' }, p.batch), h('td', {}, num(p.tpot_ms)), h('td', {}, num(p.tok_s_card))))));
    $('sw-status').textContent = `Pareto ${r.front.length} 点`;
    $('sw-status').dataset.done = '1';
  } catch (e) { $('sw-status').textContent = '错误：' + e.message; }
}

/* ------------------------------------------------------------------ models table */
function renderModels() {
  const head = h('tr', {}, h('th', { class: 'l' }, '模型'), h('th', { class: 'l' }, '分组'), h('th', { class: 'l' }, '来源'),
    h('th', { class: 'l' }, '覆盖'), h('th', { class: 'l' }, 'dtype（按发布）'), h('th', {}, '参数 B'), h('th', {}, '激活 B'),
    h('th', {}, 'vs 发布'), h('th', { class: 'l' }, '结构'));
  const rows = S.models.map((m) => h('tr', { class: 'click', onclick: () => { S.sc.model = m.id; $('model').value = m.id; onModel(); activate('eval'); schedule(); } },
    h('td', { class: 'l' }, m.label, h('div', { class: 'small muted mono' }, m.hf_id)), h('td', { class: 'l' }, SECTION_ZH[m.section] || m.section),
    h('td', { class: 'l' }, h('span', { class: 'badge ' + m.provenance }, PROV_ZH[m.provenance])),
    h('td', { class: 'l' }, h('span', { class: 'badge ' + m.coverage }, COVER_ZH[m.coverage])),
    h('td', { class: 'l' }, m.dtype), h('td', {}, num(m.params_B)), h('td', {}, num(m.active_B)),
    h('td', {}, (m.param_err * 100).toFixed(2) + '%'), h('td', { class: 'l' }, m.arch)));
  put($('models-tbl'), h('thead', {}, head), h('tbody', {}, rows));
}

/* ------------------------------------------------------------------ tabs */
function activate(tab) {
  S.tab = tab;
  document.querySelectorAll('#tabs button').forEach((b) => b.classList.toggle('on', b.dataset.tab === tab));
  document.querySelectorAll('.tab').forEach((t) => (t.hidden = t.id !== 'tab-' + tab));
}

/* ------------------------------------------------------------------ init */
async function init() {
  try {
    const [health, cat, models] = await Promise.all([req('h', '/api/health'), req('c', '/api/catalog'), req('m', '/api/models')]);
    S.cat = cat;
    S.models = models.models;
    for (const m of S.models) S.byId[m.id] = m;
    $('ver').textContent = 'v' + health.version;
    $('honesty-chip').title = cat.honesty;
    $('honesty-text').textContent = cat.honesty;
    S.sc = JSON.parse(JSON.stringify(cat.defaults));
    delete S.sc.chip;
    const mi = await req('mi', '/api/memory', { id: S.sc.mem_id });
    S.memInfo = mi;
    S.mem = { ...mi.fields };
  } catch (e) { showErr('初始化失败：' + e.message); return; }

  fillModels();
  seg('chip-preset', Object.keys(S.cat.chips).map((k) => [k, k]), () => S.preset, (v) => { S.preset = v; S.chipOver = {}; onPreset(); syncInputs(); schedule(); });
  for (const k of ['freq_ghz', 'sram_mib', 'sram_port_Bpc', 'gemv_macs', 'mac_eff', 'acc_kib']) {
    const nullable = k === 'sram_port_Bpc' || k === 'gemv_macs';
    bindNumber('c-' + k, () => chipVal(k), (x) => { if (x === null) delete S.chipOver[k]; else S.chipOver[k] = x; if (k === 'freq_ghz') onPreset(); },
      { int: k === 'gemv_macs', nullable });
  }
  seg('mapping', S.cat.mappings.map((m) => [m.id, m.label]), () => S.sc.mapping, (v) => { S.sc.mapping = v; $('mapping-desc').textContent = MAP_DESC[v]; schedule(); });
  $('mapping-desc').textContent = MAP_DESC[S.sc.mapping];
  for (const k of ['pp', 'tp', 'dp', 'ep', 'etp']) {
    $('l-' + k).dataset.after = 'layout';
    bindNumber('l-' + k, () => S.sc.layout[k], (x) => { S.sc.layout[k] = x; AFTER.lastLayout = k; }, { int: true });
  }
  AFTER.layout = () => {
    const k = AFTER.lastLayout;
    if (k !== 'pp') fixMoe(k);
    if (S.sc.layout.pp > model().n_layers) S.sc.layout.pp = model().n_layers;
    for (const x of ['pp', 'ep', 'etp']) if (x !== k) $('l-' + x)._sync();
    cardsNote();
  };
  for (const k of ['GBps', 'alpha_us']) bindNumber('k-' + k, () => S.sc.link[k], (x) => (S.sc.link[k] = x));
  for (const k of ['batch', 'ctx', 'prompt', 'out_len', 'spec_k', 'microbatches'])
    bindNumber('s-' + k, () => S.sc.serving[k], (x) => (S.sc.serving[k] = x), { int: true });
  for (const k of ['spec_accept', 'tpot_slo_ms', 'ttft_slo_ms']) bindNumber('s-' + k, () => S.sc.serving[k], (x) => (S.sc.serving[k] = x));
  bindNumber('mem_eff', () => S.sc.mem_eff, (x) => (S.sc.mem_eff = x), { nullable: true });
  seg('phase', null, () => S.sc.serving.phase, (v) => { S.sc.serving.phase = v; schedule(); });
  $('best-batch').addEventListener('change', (e) => { S.best = e.target.checked; schedule(); });
  $('wi-w').addEventListener('change', (e) => { S.wiW = e.target.value; schedule(); });
  $('wi-kv').addEventListener('change', (e) => { S.wiKV = e.target.value; schedule(); });
  bindMem();
  fillMem();
  paintMem();
  onPreset();
  onModel();
  syncInputs();

  $('tabs').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) activate(b.dataset.tab); });
  $('honesty-chip').addEventListener('click', () => activate('about'));
  seg('cmp-obj', null, () => S.cmpObj, (v) => (S.cmpObj = v));
  seg('lay-obj', null, () => S.layObj, (v) => (S.layObj = v));
  $('cmp-run').addEventListener('click', runCompare);
  $('lay-run').addEventListener('click', runLayouts);
  $('lay-stab').addEventListener('click', runStability);
  opts($('sw-path'), S.cat.sweep_paths.map((p) => [p, SWEEP_ZH[p] || p]), 'serving.batch');
  $('sw-path').addEventListener('change', () => { $('sw-values').value = SWEEP_DEFAULT[$('sw-path').value] || ''; });
  $('sw-run').addEventListener('click', runSweep);
  $('pf-run').addEventListener('click', runPareto);
  renderModels();
  window.__accel = { S, req, seq };
  runEval();
}
init();

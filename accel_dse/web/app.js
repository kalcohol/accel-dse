/* accel-dse 0.41 workbench (core v2: LLM / VLM / video DiT / protein). Zero dependencies. */
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

const COVER_ZH = { full: '完整', partial: '部分', proxy: '架构代理' };
const COVER_LEGEND = '覆盖度表示本工具对该模型结构的建模覆盖程度，与模型好坏无关：完整 = 全部算子按发布结构逐项建模；部分 = 主干逐项建模，个别机制近似；架构代理 = 有未建模的结构，结果只作量级参考。';
function coverTip(m) {
  const r = m.coverage_reasons || [];
  return (r.length ? '近似之处：\n· ' + r.join('\n· ') : '全部算子按发布结构逐项建模') +
    (m.vision_params_B ? `\n· 视觉编码器（${num(m.vision_params_B)}B 参数）未建模：只评估语言主干` : '');
}
function coverBadge(m) {
  return h('span', { class: 'badge cov ' + m.coverage, title: coverTip(m) }, h('span', { class: 'ax' }, '覆盖'), COVER_ZH[m.coverage]);
}
const PROV_ZH = { official: '官方', mirror: '镜像' };
const DOMAIN_ZH = { gen: '视频生成', protein: '蛋白质' };
const isFull = (m) => !!m && (m.domain === 'gen' || m.domain === 'protein');   // non-autoregressive full-sequence forward
const UNIT_ZH = { frame: '帧', seq: '序列', token: 'tok' };
function fmtDur(sec) {
  if (sec === null || sec === undefined || !isFinite(sec)) return '—';
  if (sec >= 7200) return num(sec / 3600) + ' h';
  if (sec >= 120) return num(sec / 60) + ' min';
  if (sec >= 1) return num(sec) + ' s';
  return num(sec * 1e3) + ' ms';
}
function domainBadge(m) {
  if (m.domain === 'gen') return h('span', { class: 'badge vlm', title: coverTip(m) },
    S.sc.workload.pipeline === false ? '视频生成 · 只评估 DiT 主干' : '视频生成 · DiT + 文本编码器 + VAE 解码');
  if (m.domain === 'protein') return h('span', { class: 'badge vlm', title: coverTip(m) }, m.is_pair ? '蛋白质 · 结构预测' : '蛋白质 · 编码器前向');
  return null;
}
const BOUND_ZH = { MAC: 'MAC 算力', FEED: 'FEED 供数', VECTOR: 'VECTOR 向量', DRAM: 'DRAM 带宽', SLC: 'SLC 系统级缓存', LINK: 'LINK 互连', SYNC: 'SYNC 同步' };
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
  mem: {}, memInfo: null, wiW: '', wiKV: '', wiAct: '', best: false, last: null,
  cmpObj: 'decode', layObj: 'decode', tab: 'eval',
  kindMem: {},   // per model kind (llm / full): batch + auto-batch, restored when switching between kinds
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
  if (S.wiKV && !isFull(model())) fo.push(['kv', S.wiKV]);
  if (S.wiAct && isFull(model())) fo.push(['act', S.wiAct]);
  return fo;
}
function body(extra = {}) {
  const sc = JSON.parse(JSON.stringify(S.sc));
  sc.chip = { ...S.chipOver };
  sc.formats_override = overrides();
  if (S.memInfo) sc.mem_id = S.memInfo.id;
  const en = Object.fromEntries(Object.entries(S.energy || {}).filter(([, v]) => v !== null && v !== undefined));
  return { chip_preset: S.preset, scenario: sc, ...(Object.keys(en).length ? { energy: en } : {}), ...extra };
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
function syncInputs() {
  document.querySelectorAll('input[type=number]').forEach((i) => i._sync && i._sync());
  if ($('w-pipeline')) $('w-pipeline').checked = S.sc.workload.pipeline !== false;
  if ($('w-placement')) $('w-placement').value = S.sc.workload.placement || 'auto';
  if ($('w-vae_tiling')) $('w-vae_tiling').checked = !!S.sc.workload.vae_tiling;
  for (const k of ['dit_fsdp', 'te_cpu', 'vae_parallel', 'overlap']) if ($('w-' + k)) $('w-' + k).checked = !!S.sc.workload[k];
  if ($('w-sample_split')) $('w-sample_split').checked = S.sc.workload.sample_split !== false;
  if ($('c-slc_policy') && S.cat) $('c-slc_policy').value = chipVal('slc_policy') || 'pin';
}

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
function catalogRows() {
  // full catalog in API order: vendor (厂商) → evaluable families → catalog-only families (video generation / protein)
  const off = Object.fromEntries(S.offline.map((o) => [o.id, o]));
  return S.catalog.map((id) => S.byId[id] || off[id]).filter(Boolean);
}
function vendorGroups(rows) {
  const groups = [];
  for (const m of rows) {
    let g = groups[groups.length - 1];
    if (!g || g.key !== m.provider) groups.push((g = { key: m.provider, label: m.provider_label, items: [] }));
    g.items.push(m);
  }
  return groups;
}
const OFF_DOMAIN = { gen: '视频生成', protein: '蛋白质' };
const offBadge = (o) => h('span', { class: 'badge off', title: (OFF_DOMAIN[o.domain] || o.domain_label) + '模型在 core v2 中暂未接入，只列在目录中，不能评估' + (o.reason ? '：' + o.reason : '') }, o.status);
function fillModels() {
  const sel = $('model');
  put(sel, ...vendorGroups(catalogRows()).map((g) => h('optgroup', { label: g.label }, g.items.map((m) => m.evaluable
    ? h('option', { value: m.id, title: coverTip(m) }, `${m.label}${m.domain === 'vlm' ? '（VLM）' : DOMAIN_ZH[m.domain] ? `（${DOMAIN_ZH[m.domain]}）` : ''}　· ${COVER_ZH[m.coverage]}`)
    : h('option', { value: m.id, disabled: true, title: '暂未接入 v2，不能评估' + (m.reason ? '：' + m.reason : '') }, `${m.label}　· ${OFF_DOMAIN[m.domain]} · ${m.status}`)))));
  sel.value = S.sc.model;
  sel.addEventListener('change', () => { setModel(sel.value); schedule(); });
}
function badges(m, whatIf) {
  return [
    h('span', { class: 'badge ' + m.provenance, title: m.provenance === 'mirror' ? '官方仓库需授权，使用字节相同的公开镜像' : '官方发布' },
      h('span', { class: 'ax' }, '来源'), PROV_ZH[m.provenance] || m.provenance),
    coverBadge(m),
    m.domain === 'vlm' ? h('span', { class: 'badge vlm', title: coverTip(m) }, 'VLM · 视觉编码器未建模') : domainBadge(m),
    h('span', { class: 'badge' }, h('span', { class: 'ax' }, 'dtype'), m.dtype),
    whatIf ? h('span', { class: 'badge wi' }, 'what-if dtype') : null,
  ].filter(Boolean);
}
function setModel(id) {
  // switching between LLM and video / protein models keeps a separate batch + auto-batch per kind
  const was = model(), next = S.byId[id];
  const kw = isFull(was) ? 'full' : 'llm', kn = isFull(next) ? 'full' : 'llm';
  if (kw !== kn) {
    S.kindMem[kw] = { batch: S.sc.serving.batch, best: S.best };
    const k = S.kindMem[kn] || (kn === 'full' ? { batch: 1, best: false } : { batch: 1, best: true });
    S.sc.serving.batch = k.batch; S.best = k.best; $('best-batch').checked = S.best;
  }
  if (!was || !next || was.id !== next.id) S.sc.workload = { ...S.cat.defaults.workload, clip_slo_s: S.sc.workload.clip_slo_s, seq_slo_ms: S.sc.workload.seq_slo_ms, fold_slo_s: S.sc.workload.fold_slo_s };
  S.sc.model = id;
  $('model').value = id;
  onModel();
}
function paintDomain(m) {
  const full = isFull(m);
  document.querySelectorAll('.llm-only').forEach((e) => (e.hidden = full));
  document.querySelectorAll('.full-only').forEach((e) => (e.hidden = !full));
  document.querySelectorAll('.gen-only').forEach((e) => (e.hidden = m.domain !== 'gen'));
  document.querySelectorAll('.protein-only').forEach((e) => (e.hidden = m.domain !== 'protein'));
  document.querySelectorAll('.dp-in').forEach((e) => (e.hidden = !(m.is_moe || full)));
  document.querySelectorAll('.pair-only').forEach((e) => (e.hidden = !m.is_pair));
  document.querySelectorAll('.seq-slo').forEach((e) => (e.hidden = !!m.is_pair));
  document.querySelectorAll('.diff-only').forEach((e) => (e.hidden = !(m.is_pair && m.workload && m.workload.diff_steps)));
  document.querySelectorAll('.tp-in').forEach((e) => (e.hidden = !!m.is_pair));   // structure models: PP × DP × DAP, no TP
  document.querySelectorAll('.sp-in').forEach((e) => (e.hidden = !full));
  $('l-sp-name').textContent = m.is_pair ? 'DAP' : 'SP';
  $('l-sp-lab').title = m.is_pair
    ? 'DAP（动态轴并行，FastFold）：pair / MSA / 模板网格沿一个残基轴切到 DAP 张卡，单一 / 原子轨道每卡重复；三角乘法、偏置、外积均值 all-gather，轴切换 all-to-all'
    : 'Ulysses 序列并行：token 按 SP 切分，注意力前后各一次 all-to-all（每 rank 算 heads/SP 个头的完整序列）';
  if (full) {
    if (S.cmpObj === 'goodput') { S.cmpObj = 'decode'; paintSeg('cmp-obj', 'decode'); }
    if (S.layObj === 'goodput') { S.layObj = 'decode'; paintSeg('lay-obj', 'decode'); }
  }
  const unit = m.domain === 'gen' ? '帧/s/卡' : m.domain === 'protein' ? '序列/s/卡' : null;
  for (const id of ['cmp-obj', 'lay-obj']) $(id).querySelector('[data-v=decode]').textContent = unit ? `目标：吞吐（${unit}）` : '目标：decode 吞吐';
  $('batch-label').textContent = m.domain === 'gen' ? 'batch（视频段数）' : m.domain === 'protein' ? 'batch（序列数）' : 'batch';
  $('best-batch-label').textContent = m.domain === 'gen' ? '自动取满足单段延迟 SLO 的最大 batch（最大化帧/s/卡）'
    : m.domain === 'protein' ? '自动取满足批延迟 SLO 的最大 batch（最大化序列/s/卡）' : '自动取满足 TPOT SLO 的最大 batch';
  const w = m.workload;
  if (w) {
    $('wl-kind').textContent = m.domain === 'gen' ? `视频 · 默认 ${w.width}×${w.height} · ${w.frames} 帧 · ${w.fps} fps · ${w.steps} 步 · CFG ${w.cfg}`
      : w.structure ? `蛋白质结构 · 默认 ${w.seq_len} 残基 · ${w.msa ? `MSA ${w.msa} 行` : w.xmsa ? `MSA ${w.xmsa} 行` : '单序列'} · 主干 ${w.recycles} 遍${w.diff_steps ? ` · 扩散 ${w.diff_steps} 步 × ${w.samples} 样本` : ''}`
      : `蛋白质 · 默认 ${w.seq_len} 残基（训练上限 ${w.max_seq}）`;
    for (const [k, v] of Object.entries({ frames: w.frames, height: w.height, width: w.width, steps: w.steps, cfg: w.cfg, seq_len: w.seq_len,
      msa: w.msa || w.xmsa, recycles: w.recycles, psteps: w.diff_steps, samples: w.samples }))
      if ($('w-' + k)) $('w-' + k).placeholder = v ? `默认 ${v}` : '';
    if ($('w-msa')) $('w-msa').disabled = !!w.structure && !w.msa && !w.xmsa;
    const fr = m.domain !== 'gen' ? '' : w.vae_frames === 'chunk17' ? `帧每 17 帧一块 → ${(16 / w.vae[0] | 0) + 1} 潜帧`
      : w.vae_frames === 'h3' ? '帧补齐到 17n+5 → 5n+2 潜帧' : `帧 (F−1)/${w.vae[0]}+1`;
    $('wl-note').textContent = '留空 = 按发布默认（' + w.source + '）。' + (m.domain === 'gen'
      ? `token 数 = 潜空间网格（${fr}，像素 /${w.vae[1]}，patch ${w.patch.join('×')}）${w.joint_text ? ` + ${w.text_tokens} 个文本 token（联合注意力）` : `；文本 ${w.text_tokens} token 走跨注意力`}`
        + (w.audio_per_s ? ` + 音频 ${w.audio_per_s}/s × ${w.audio_channels} 声道 token（同一序列）` : '')
        + (w.attention === 'factorized' ? '；注意力按发布结构分解：空间块在潜帧内、时间块沿时间轴' : '；全 3D 注意力')
        + (w.pipeline ? `；pipeline 组件：${w.pipeline.map((p) => `${p.label}（${num(p.GB)} GB ${p.dtype}${p.tokens ? `，${p.tokens} token` : ''}）`).join('、')}` : '') + '。'
      : w.structure ? `N 残基 → pair 表示 N² × ${w.pair_dim} 维（三角乘法 / 三角注意力 ∝ N³）`
        + (w.msa || w.xmsa ? '；MSA 表示 行数 × N（行 / 列注意力、外积均值 ∝ 行数 · N²）' : '')
        + (w.templates ? `；模板 ${w.templates} 个 × N² 网格` : '')
        + (w.atoms_per_res ? `；原子 ≈ ${w.atoms_per_res} / 残基「假设」，局部窗口注意力` : '')
        + '。只评估网络推理（MSA / 模板检索与特征化不在范围内）；布局取 PP × DP × DAP（DAP = pair / MSA 网格按残基轴切分，FastFold 动态轴并行；pair 的 TP 未建模）。'
      : 'token 数 = 残基 + <cls>/<eos>。');
  }
}
function onModel() {
  const m = model();
  paintDomain(m);
  if ($('sw-path').options.length) fillSweepPaths();
  put($('model-badges'), ...badges(m, S.wiW || S.wiKV || (S.wiAct && isFull(m))));
  put($('model-facts'), 
    h('span', {}, '参数 ', h('b', {}, num(m.params_B) + 'B')),
    m.is_pair ? h('span', {}, 'pair ', h('b', {}, m.workload.pair_dim + ' 维')) : isFull(m) ? h('span', {}, '头 ', h('b', {}, m.heads)) : h('span', {}, '激活 ', h('b', {}, num(m.active_B) + 'B')),
    h('span', {}, '层 ', h('b', {}, m.n_layers)),
    h('span', {}, m.arch),
    m.mtp_layers ? h('span', {}, `MTP ×${m.mtp_layers}`) : null,
    h('span', { class: 'muted', title: '与发布 safetensors 头统计值的偏差' }, `vs 发布 ${(m.param_err * 100).toFixed(2)}%`));
  put($('model-approx'), ...(m.coverage_reasons || []).map((r) => h('li', {}, r)),
    ...(m.vision_params_B ? [h('li', {}, `视觉编码器（${num(m.vision_params_B)}B 参数）未建模：不计其权重存储与图像 prefill，只评估语言主干`)] : []));
  $('model-approx-wrap').hidden = !(m.coverage_reasons || []).length && !m.vision_params_B;
  put($('model-notes'), ...m.notes.map((n) => h('li', {}, n)));
  $('model-notes-wrap').hidden = !m.notes.length;
  document.querySelectorAll('.moe-only').forEach((e) => (e.hidden = !m.is_moe));
  const L = S.sc.layout;
  if (isFull(m)) { L.ep = 1; L.etp = 1; L.sp = L.sp || 1; if (m.is_pair) L.tp = 1; }
  else { L.sp = 1; if (!m.is_moe) { L.dp = 1; L.ep = 1; L.etp = 1; } else fixMoe('tp'); }
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
const cards = () => S.sc.layout.pp * S.sc.layout.tp * S.sc.layout.dp * (S.sc.layout.sp || 1);
function setCards(n) {
  // default fill for a new card count: all TP (MoE: experts spread with EP = TP); 「布局搜索」finds the best
  const L = S.sc.layout;
  L.pp = 1; L.tp = n; L.dp = 1; L.sp = 1;
  if (model().is_pair) { L.tp = 1; L.sp = n; }   // structure models: no TP — fill with DAP (pair grid split)
  if (model().is_moe) { L.ep = n; L.etp = 1; } else { L.ep = 1; L.etp = 1; }
}
function cardsNote() {
  const L = S.sc.layout;
  const m = model();
  $('cards-note').textContent = m.is_pair ? `PP${L.pp} × DP${L.dp} × DAP${L.sp || 1} = ${cards()} 卡；结构模型无 TP（改卡数按 DAP 填充）`
    : `PP${L.pp} × TP${L.tp} × DP${L.dp}${isFull(m) ? ` × SP${L.sp || 1}` : ''} = ${cards()} 卡；改卡数会按 TP 重新填充布局`;
  const c = $('l-cards');
  if (c._sync && document.activeElement !== c) c._sync();
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
function tagSuffix(o) {
  // Prefer dual-axis 「规范 · 产品」; fall back to legacy single tag.
  if (o && typeof o === 'object') {
    const zh = o.status_zh || '';
    if (!zh || zh === 'SoC 设计选择' || (o.product === 'shipping' && (o.spec === 'jedec' || o.spec === 'jedec_likely'))) return '';
    return ` (${zh})`;
  }
  return o && o !== 'jedec' && o !== 'vendor_shipping' ? ` (${S.cat.memory.tag_zh[o] || o})` : '';
}
function opts(sel, list, value) {
  put(sel, ...list.map(([v, l]) => h('option', { value: v }, l)));
  sel.value = String(value);
  if (sel.value !== String(value) && list.length) sel.value = String(list[0][0]);
}
function fillMem() {
  const t = memType(), f = memForm();
  opts($('m-type'), S.cat.memory.types.map((x) => [x.id, x.id]), S.mem.type);
  opts($('m-form'), t.forms.map((x) => [x.id, x.label]), f.id);
  opts($('m-width'), f.widths.map((w) => [w.bits, w.bits + ' bit' + tagSuffix(w)]), S.mem.width);
  opts($('m-rate'), f.rates.map((r) => [r.MTps, r.MTps + tagSuffix(r)]), S.mem.rate);
  opts($('m-count'), f.counts.map((c) => [c.n, `${c.n} × ${f.unit === 'stack' ? '堆栈' : '颗'}` + tagSuffix(c)]), S.mem.count);
  const hbm = t.kind === 'HBM';
  $('m-cap-wrap').hidden = hbm;
  $('m-height-wrap').hidden = !hbm;
  if (hbm) opts($('m-height'), f.cap_tags.map((c) => [`${c.height}:${c.die_Gb}`, `${c.height}-Hi ${c.die_Gb}Gb · ${c.GB} GB` + tagSuffix(c)]), `${S.mem.height}:${S.mem.die}`);
  else opts($('m-cap'), (f.caps[String(S.mem.width)] || []).map((c) => [c.GB, c.GB + ' GB' + tagSuffix(c)]), S.mem.cap);
  $('m-meta-wrap').hidden = t.id !== 'LPDDR6';
  if (t.id !== 'LPDDR6') S.mem.meta_mode = false;
  $('m-meta').checked = !!S.mem.meta_mode;
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
  if (t.kind === 'HBM') { b.height = S.mem.height; b.die = S.mem.die; } else { b.cap = S.mem.cap; if (t.id === 'LPDDR6') b.meta_mode = !!S.mem.meta_mode; }
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
    h('span', { title: (S.cat.memory.spec_hint || {})[r.spec_status] || '' }, '规范 ', h('b', {}, r.spec_zh || r.tag_zh), ' · 产品 ', h('b', {}, r.product_zh || '')),
    ...(r.meta_mode ? [h('span', { class: 'tag assume' }, 'meta')] : []),
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
  $('m-meta').addEventListener('change', () => { S.mem.meta_mode = $('m-meta').checked; resolveMem(); });
}

/* ------------------------------------------------------------------ evaluation */
let timer = null;
function schedule() { clearTimeout(timer); timer = setTimeout(runEval, 120); invalidate(); }
function invalidate() {
  if (S.cat) paintScope();
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
  if (ok === null || ok === undefined) return h('span', { class: 'flag pend' }, '—');
  return h('span', { class: 'flag ' + (ok ? 'ok' : 'no'), title: 'DP 组内 prefill 的 TTFT 与 SLO 比较' },
    (ok ? '✓ ' : '✗ ') + (ms === undefined ? '' : ms === null ? '不可行' : num(ms) + ' ms'));
}
function renderEval(r) {
  const s = r.summary, sv = r.scenario.serving, g = r.goodput;
  const m = model();
  put($('model-badges'), ...badges(m, r.model.what_if));
  put($('warns'), ...s.warnings.filter((w) => !w.startsWith('容量不足')).map((w) => h('div', {}, w)),
    ...(r.model.what_if ? [h('div', {}, 'what-if：dtype 已偏离官方发布，结果仅供推演。')] : []));
  renderStages(r);
  renderEnergy(r);
  { const P = r.scenario.package_cards || 1, tot = r.stages.reduce((x, st) => x + (st.link_GB ? st.link_GB.total : 0), 0),
      d = r.stages.reduce((x, st) => x + (st.link_GB ? st.link_GB.d2d : 0), 0);
    $('tier-note').textContent = P > 1 ? `每封装 ${P} 卡：DiT / 流水级通信字节中 ${tot > 0 ? pct(d / tot) : '—'} 走 D2D，其余走网络层` : ''; }
  renderAssumptions(r);
  if (!s.fits) { $('kpis').hidden = true; runFit(); return; }
  $('fit').hidden = true;
  $('kpis').hidden = false;
  const cap = S.memInfo ? S.memInfo.capacity_GiB : NaN;
  const k = [];
  const auto = S.best ? '（自动）' : '';
  if (s.gen) { put($('kpis'), ...domainKpis(s, cap, auto)); return; }
  if (s.phase === 'decode') {
    const over = s.tpot_ms > sv.tpot_slo_ms;
    k.push(kpi('TPOT', num(s.tpot_ms) + ' ms', `batch ${s.batch}${auto} · SLO ${sv.tpot_slo_ms} ms${over ? ' · 超出' : ''}`, over ? 'warn' : ''));
    k.push(kpi('吞吐 / 卡', num(s.tok_s_card) + ' tok/s', `共 ${num(s.tok_s)} tok/s · ${s.cards} 卡 · ${s.layout}`));
  } else {
    const over = s.ttft_ms > sv.ttft_slo_ms;
    k.push(kpi('TTFT', num(s.ttft_ms) + ' ms', `batch ${s.batch} · prompt ${sv.prompt} · SLO ${sv.ttft_slo_ms} ms`, over ? 'warn' : ''));
    k.push(kpi('prefill 吞吐 / 卡', num(s.tok_s_card) + ' tok/s', `共 ${num(s.tok_s)} tok/s · ${s.cards} 卡 · ${s.layout}`));
  }
  k.push(h('div', { class: 'kpi' }, h('div', { class: 'l' }, '绑定瓶颈 · 有效 MAC'),
    h('div', { class: 'v' }, boundTag(s.bound), ' ', pct(s.array_util)),
    h('div', { class: 's' }, `${BOUND_ZH[s.bound] || ''} · 最重流水级 ${s.heaviest_stage}`)));
  if (g) k.push(h('div', { class: 'kpi ' + (g.ttft_ok ? '' : 'warn') }, h('div', { class: 'l' }, 'goodput / 卡（含 prefill）'),
    h('div', { class: 'v' }, num(g.tok_s_card) + ' tok/s'),
    h('div', { class: 's' }, 'DP prefill TTFT ', ttftFlag(g.ttft_ok, g.ttft_ms), g.prefill_batch ? ` · prefill batch ${g.prefill_batch}` : '')));
  else k.push(kpi('映射组织', (S.cat.mappings.find((x) => x.id === s.mapping) || {}).label || s.mapping, MAP_DESC[s.mapping] || ''));
  const frac = isFinite(cap) && cap > 0 ? Math.min(1, s.dram_need_GiB / cap) : 0;
  k.push(h('div', { class: 'kpi' }, h('div', { class: 'l' }, 'DRAM 需求 / 容量（每卡）'),
    h('div', { class: 'v' }, `${num(s.dram_need_GiB)} / ${num(cap)} GiB`),
    h('div', { class: 's' }, h('span', { class: 'ubar wide' }, h('i', { style: `width:${frac * 100}%` })), `SRAM 驻留 ${pct(s.residency)}`)));
  put($('kpis'), ...k);
}
function memKpi(s, cap) {
  const frac = isFinite(cap) && cap > 0 ? Math.min(1, s.dram_need_GiB / cap) : 0;
  return h('div', { class: 'kpi' }, h('div', { class: 'l' }, 'DRAM 需求 / 容量（每卡）'),
    h('div', { class: 'v' }, `${num(s.dram_need_GiB)} / ${num(cap)} GiB`),
    h('div', { class: 's' }, h('span', { class: 'ubar wide' }, h('i', { style: `width:${frac * 100}%` })),
      s.gen && s.gen.act_GiB !== undefined ? `激活常驻 ${num(s.gen.act_GiB)} GiB · SRAM 驻留 ${pct(s.residency)}` : `SRAM 驻留 ${pct(s.residency)}`));
}
function boundKpi(s) {
  return h('div', { class: 'kpi' }, h('div', { class: 'l' }, '绑定瓶颈 · 有效 MAC'),
    h('div', { class: 'v' }, boundTag(s.bound), ' ', pct(s.array_util)),
    h('div', { class: 's' }, `${BOUND_ZH[s.bound] || ''} · 最重流水级 ${s.heaviest_stage}`));
}
function domainKpis(s, cap, auto) {
  // video: clip latency, per-frame latency, frames/s/card; protein: batch latency, sequences/s/card, residues/s/card
  const g = s.gen, w = g.workload;
  const k = [];
  if (g.unit === 'frame') {
    const over = !s.slo_ok;
    const pl = g.pipeline;
    k.push(kpi('单段延迟（clip）', fmtDur(g.clip_s),
      `${w.width}×${w.height} · ${w.frames} 帧 · ${w.steps} 步 × CFG ${w.cfg} · batch ${s.batch}${auto} · SLO ${fmtDur(g.slo_s)}${over ? ' · 超出' : ''}`
      + (pl ? ` · 文本编码 ${fmtDur(pl.te_s)} + 去噪 ${fmtDur(g.denoise_s)} + 解码 ${fmtDur(pl.decode_s)}${pl.load_s ? ` + 主机重载 ${fmtDur(pl.load_s)}` : ''} · 组件${pl.placement === 'auto' ? '（自动）' : ''}${pl.place_label}` : ' · 只计 DiT 去噪（文本编码器 / VAE 未计）'), over ? 'warn' : ''));
    k.push(kpi('每帧延迟', fmtDur(g.s_per_frame),
      `每去噪步 ${fmtDur(g.step_ms / 1e3)} · ${num(w.seq_tokens)} token / 前向${w.audio_tokens ? `（含音频 ${num(w.audio_tokens)}）` : ''}${w.attention === 'factorized' ? ' · 分解注意力' : ''} · 视频 ${num(w.video_s)} s${g.realtime_x ? ` · 实时倍率 ${num(g.realtime_x)}×` : ''}`));
    k.push(kpi('吞吐 / 卡', num(g.frames_per_s_card) + ' 帧/s',
      `${num(g.clips_per_hour_card)} 段/小时/卡 · ${s.cards} 卡 · ${s.layout} · ${num(g.tflop_per_request)} TFLOP/段${pl ? `（其中文本编码 + 解码 ${num(pl.tflop / s.batch)}）` : ''}`));
  } else {
    const over = !s.slo_ok;
    k.push(kpi('批延迟', fmtDur(g.batch_ms / 1e3),
      `batch ${s.batch}${auto} × ${w.seq_len} 残基 · SLO ${w.recycles ? num(g.slo_ms / 1e3) + ' s' : num(g.slo_ms) + ' ms'}${over ? ' · 超出' : ''}`, over ? 'warn' : ''));
    k.push(kpi('吞吐 / 卡', num(g.seq_per_s_card) + ' 序列/s',
      `${num(g.residues_per_s_card)} 残基/s/卡 · ${s.cards} 卡 · ${model().is_pair ? s.layout.replace('SP', 'DAP') : s.layout}`));
    k.push(w.recycles
      ? kpi('每序列计算', num(g.tflop_per_request) + ' TFLOP', `${w.seq_len} 残基${w.msa ? ` · MSA ${num(w.msa)} 行` : ''}${w.xmsa ? ` · extra MSA ${num(w.xmsa)} 行` : ''} · 主干 ${w.recycles} 遍${w.diff_steps ? ` · 扩散 ${w.diff_steps} 步 × ${w.samples} 样本` : ''}`)
      : kpi('每序列计算', num(g.tflop_per_request * 1e3) + ' GFLOP', `${w.tokens} token（含 <cls>/<eos>）· 单次编码器前向`));
  }
  k.push(boundKpi(s));
  k.push(memKpi(s, cap));
  return k;
}
function placeText(pl, sc) {
  const gb = (x) => num(x / 1e9) + ' GB';
  const cpu = pl.te_cpu ? `文本编码器在主机 CPU 上运行（Wan --t5_cpu；${fmtDur(pl.te_s)} = ${num(pl.parts.filter((p) => p.role === 'text_encoder').reduce((a, p) => a + p.tflop, 0))} TFLOP ÷ ${num(pl.host_TFLOPS)} TFLOPS「假设」，卡上不放编码器）；` : '';
  const fsdp = sc && sc.workload.dit_fsdp && (sc.layout.sp || 1) * sc.layout.dp > 1 ? `DiT 权重 FSDP 切到每级 ${(sc.layout.sp || 1) * sc.layout.dp} 张卡（Wan --dit_fsdp，逐层 all-gather 与计算重叠）；` : '';
  const vp = pl.vae_par > 1;
  const vae = vp ? `VAE 每卡一份（分块解码拆到 ${pl.vae_par} 张卡）` : 'VAE 在末级卡';
  const all = (pl.te_cpu ? 0 : pl.te_w) + pl.vae_w;
  if (pl.place === 'resident') return `${cpu}${fsdp}常驻——${pl.te_cpu ? '' : '文本编码器在首级卡、'}${vae}，与 DiT 同时占用显存（共 ${gb(all)}）`;
  const sh = pl.te_cards > 1 ? `文本编码器权重按 FSDP 切到 ${pl.te_cards} 张卡（每卡 ${gb(pl.te_card_w)}，含 2 层预取；逐层 all-gather ${fmtDur(pl.gather_s)}，与编码计算重叠）` : '';
  const off = pl.load_s ? `组件与 DiT 分时占用显存（需求取三者最大值），每请求从主机重载文本编码器 + DiT + VAE 权重 ${fmtDur(pl.load_s)}（${num(pl.host_GBps)} GB/s「假设」，主机保留副本）` : '';
  return cpu + fsdp + [sh, off].filter(Boolean).join('；') + '；' + vae;
}
async function runFit() {
  const box = $('fit');
  box.hidden = false;
  put(box, h('div', { class: 'muted' }, '容量不足，正在计算可行的修正方案…'));
  try {
    const f = await track(req('fit', '/api/fit', body()));
    if (!f) return;
    const m = model();
    const lines = [h('div', { class: 'fit-title' }, h('b', {}, '放不下：'),
      `${m.label} 在当前 ${f.cards} 卡上，最重流水级每卡需要 ${num(f.need_GiB)} GiB（其中权重 ${num(f.weights_GiB)} GiB${f.pipe_w_GiB ? `，含文本编码器 / VAE ${num(f.pipe_w_GiB)} GiB` : ''}），每卡容量 ${num(f.cap_GiB)} GiB。`)];
    const acts = [];
    if (f.max_batch) {
      acts.push(h('button', { class: 'btn', onclick: () => { S.sc.serving.batch = f.max_batch; syncInputs(); schedule(); } },
        `batch 改为 ${f.max_batch}（能放下的最大值）`));
      if (!S.best) acts.push(h('button', { class: 'btn ghost', onclick: () => { S.best = true; $('best-batch').checked = true; schedule(); } }, '开启自动 batch'));
    }
    if (f.min_cards) {
      const c = f.min_cards;
      acts.push(h('button', { class: 'btn', onclick: () => applyLayout(c.layout_obj, c.batch) },
        `改为 ${c.cards} 卡 · ${c.layout} · batch ${c.batch}（每卡 ${num(c.need_GiB)} GiB${c.meets_slo ? '' : isFull(m) ? '，延迟 SLO 未满足' : '，TPOT SLO 未满足'}）`));
    }
    if (m.domain === 'gen' && S.sc.workload.pipeline !== false && (S.sc.workload.placement || 'auto') !== 'auto') acts.push(h('button', { class: 'btn', onclick: () => { S.sc.workload.placement = 'auto'; syncInputs(); schedule(); } },
      '组件放置改为「自动」（常驻 → 文本编码器分片 → 顺序卸载，取第一个放得下的）'));
    if (m.domain === 'gen' && !S.sc.workload.dit_fsdp && (S.sc.layout.sp || 1) * S.sc.layout.dp > 1) acts.push(h('button', { class: 'btn', onclick: () => { S.sc.workload.dit_fsdp = true; syncInputs(); schedule(); } },
      `DiT 权重 FSDP 分片到 ${(S.sc.layout.sp || 1) * S.sc.layout.dp} 张卡（Wan --dit_fsdp）`));
    if (m.domain === 'gen' && !S.sc.workload.te_cpu && S.sc.workload.pipeline !== false) acts.push(h('button', { class: 'btn ghost', onclick: () => { S.sc.workload.te_cpu = true; syncInputs(); schedule(); } },
      '文本编码器放主机 CPU（Wan --t5_cpu；编码时间按主机 TFLOPS「假设」）'));
    if (m.domain === 'gen' && S.sc.workload.pipeline !== false) acts.push(h('button', { class: 'btn ghost', onclick: () => { S.sc.workload.pipeline = false; syncInputs(); schedule(); } },
      '只评估 DiT（文本编码器 / VAE 权重不计，相当于卸载到主机）'));
    if (f.min_mem) {
      const mm = f.min_mem;
      acts.push(h('button', { class: 'btn ghost', onclick: () => applyMem(mm.id) },
        `存储器改为 ${mm.count} ${S.memInfo && S.memInfo.kind === 'HBM' ? '堆栈' : '颗'} × ${+(mm.capacity_GiB / mm.count).toFixed(1)} GiB = ${num(mm.capacity_GiB)} GiB / 卡（${mm.tag_zh}）`));
    }
    if (!acts.length) lines.push(h('div', {}, '在 64 卡以内、当前存储器类型的任何容量下都放不下；请换容量更大的存储器类型或更小的模型。'));
    else lines.push(h('div', { class: 'small muted' }, `一键修正（最少卡数的布局按${m.domain === 'gen' ? '帧/s/卡' : m.domain === 'protein' ? '序列/s/卡' : ' decode 吞吐'}取最优；存储器只在当前类型、速率内加大容量）：`));
    put(box, ...lines, h('div', { class: 'fit-acts' }, acts));
  } catch (e) { put(box, h('div', { class: 'err' }, '容量检查失败：' + e.message)); }
}
function applyLayout(lay, batch) {
  Object.assign(S.sc.layout, lay);
  if (batch) S.sc.serving.batch = batch;
  syncInputs(); cardsNote(); schedule();
}
async function applyMem(id) {
  try {
    const r = await req('mem', '/api/memory', { id });
    if (!r) return;
    S.memInfo = r; S.mem = { ...r.fields };
    fillMem(); paintMem(); schedule();
  } catch (e) { showErr('存储器：' + e.message); }
}
const E_UNIT = { token: 'token', 'prompt token': 'prompt token', frame: '帧', seq: '序列' };
function renderEnergy(r) {
  const e = r.energy;
  if (!e) { put($('energy-tbl')); return; }
  const c = e.counts_per_unit, j = e.J_by_action || {};
  const u = E_UNIT[e.unit] || e.unit;
  const sci = (x) => (x === 0 ? '0' : x.toExponential(3));
  const rows = [['MAC（bf16 等效）', c.mac, '', 'mac', 'pJ_mac'], ['向量操作', c.vec, '', 'vec', 'pJ_vec'],
    ['SRAM 端口', c.sram, ' B', 'sram', 'pJ_bit_sram'],
    ...(c.slc ? [['SLC 命中', c.slc, ' B', 'slc', 'pJ_bit_slc']] : []),
    [c.slc ? 'DRAM 读写（SLC 未命中）' : 'DRAM 读写', c.dram, ' B', 'dram', 'pJ_bit_dram'],
    ...(c.d2d ? [['D2D 发送（封装内）', c.d2d, ' B', 'd2d', 'pJ_bit_d2d']] : []),
    [c.d2d ? '网络层发送（跨封装）' : '链路发送', c.link, ' B', 'link', 'pJ_bit_link'], ['卡·秒（静态）', c.idle_card_s, ' 卡·s', 'idle', 'idle_W']];
  const head = h('tr', {}, h('th', {}, '动作'), h('th', {}, `次数 / ${u}`), h('th', {}, `能耗 J / ${u}`));
  const body = rows.map(([lab, n, sfx, k, key]) => h('tr', {}, h('td', {}, lab), h('td', { class: 'num' }, sci(n) + sfx),
    h('td', { class: 'num' }, e.provided.includes(key) ? sci(j[k] || 0) : h('span', { class: 'muted' }, '未提供'))));
  if (e.J_per_unit !== undefined) body.push(h('tr', {}, h('td', {}, h('b', {}, '合计')), h('td', { class: 'num' }, ''),
    h('td', { class: 'num' }, h('b', {}, sci(e.J_per_unit)), ` · 平均 ${num(e.avg_W_per_card)} W/卡`)));
  put($('energy-tbl'), h('thead', {}, head), h('tbody', {}, ...body));
}
function renderStages(r) {
  const hasSlc = r.stages.some((st) => st.mem.slc_MiB > 0);
  const COMPS = hasSlc ? ['mac', 'feed', 'vector', 'dram', 'slc', 'link', 'sync'] : COMPONENTS;
  const head = h('tr', {}, h('th', { class: 'l' }, '流水级'), h('th', { class: 'l' }, '层'), h('th', {}, '瓶颈'),
    ...COMPS.map((c) => h('th', {}, c.toUpperCase() + ' ms')), h('th', {}, '合计 ms'), h('th', { class: 'l' }, '有效 MAC'),
    h('th', {}, 'TFLOP / 步'));
  const rows = r.stages.map((st) => {
    const comp = { mac: st.t_ms.mac, feed: st.t_ms.feed, vector: st.t_ms.vector, dram: st.t_ms.dram, slc: st.t_ms.slc || 0, link: st.t_ms.link, sync: st.t_ms.sync };
    return h('tr', { class: st.index === r.summary.heaviest_stage ? 'best' : '' },
      h('td', { class: 'l' }, st.index), h('td', { class: 'l' }, `${st.layers[0]}–${st.layers[1] - 1}`), h('td', {}, boundTag(st.bound)),
      ...COMPS.map((c) => h('td', { style: c.toUpperCase() === st.bound ? 'color:var(--b-' + c + ');font-weight:700' : '' }, num(comp[c]))),
      h('td', {}, num(st.t_ms.total)),
      h('td', { class: 'l' }, h('span', { class: 'ubar' }, h('i', { style: `width:${Math.min(100, st.array_util * 100)}%` })), pct(st.array_util)),
      h('td', {}, num(st.tflops)));
  });
  put($('stage-tbl'), h('thead', {}, head), h('tbody', {}, rows));
  put($('legend'), ...COMPS.map((c) => h('span', { style: `--c:var(--b-${c})` }, BOUND_ZH[c.toUpperCase()])),
    h('span', { style: '--c:transparent' }, 'MAC / FEED 为逐算子取 max 之前的分项和'));
  const mh = h('tr', {}, h('th', { class: 'l' }, '流水级'), h('th', {}, '权重 GiB'), h('th', {}, r.summary.gen ? 'KV GiB（无 KV 缓存）' : 'KV GiB'), h('th', {}, '状态 GiB'),
    h('th', {}, '需求 / 容量 GiB'), h('th', {}, 'SRAM 驻留'), h('th', {}, 'SRAM 中 KV MiB'), h('th', {}, '暂存区 MiB'),
    h('th', {}, r.summary.gen ? 'DRAM 流量 / 前向 GB（含激活流式）' : 'DRAM 流量 / 步 GB'),
    ...(hasSlc ? [h('th', {}, '含 SLC 驻留'), h('th', {}, 'SLC 命中 GB')] : []), h('th', {}, '反量化 百万元素'));
  const mr = r.stages.map((st) => h('tr', {}, h('td', { class: 'l' }, st.index), h('td', {}, num(st.mem.stored_w_GiB)),
    h('td', {}, num(st.mem.kv_GiB)), h('td', {}, num(st.mem.state_GiB)),
    h('td', { style: st.mem.fits ? '' : 'color:var(--danger)' }, `${num(st.mem.need_GiB)} / ${num(st.mem.cap_GiB)}`),
    h('td', {}, pct(st.mem.residency)), h('td', {}, num(st.mem.kv_sram_MiB)), h('td', {}, num(st.mem.staging_MiB)),
    h('td', {}, num(st.dram_GB.total)),
    ...(hasSlc ? [h('td', { title: st.mem.slc_policy === 'lru' ? (st.mem.slc_all ? 'lru：片外工作集放得下 → 全部读命中' : 'lru：放不下 → 循环访问全部未命中') : `pin：SLC 中权重 ${num(st.mem.slc_w_MiB)} MiB，KV / 状态 ${num(st.mem.slc_kv_MiB)} MiB` }, pct(st.mem.slc_residency)),
      h('td', {}, num(st.dram_GB.slc || 0))] : []), h('td', {}, num(st.convert_Melems))));
  put($('mem-tbl'), h('thead', {}, mh), h('tbody', {}, mr));
}
function renderAssumptions(r) {
  const c = r.scenario.chip, cc = S.cat.chips[S.preset], sc = r.scenario;
  const port = c.sram_port_Bpc || cc.port_GBps / cc.freq_ghz;
  const items = [
    `芯片 ${c.name}：${c.rows}×${c.cols}×${c.engines} @ ${c.freq_ghz} GHz，MAC 效率 ${c.mac_eff}`,
    `SRAM ${c.sram_mib} MiB，端口 ${num(port)} B/cycle${c.sram_port_Bpc ? '' : '（默认 4·(R+C·E)·2）'}`,
    `GEMV 单元 ${c.gemv_macs || cc.gemv} MAC/cycle${c.gemv_macs ? '' : '（默认 = 阵列 MAC / 8）'}；向量通道 ${c.vector_lanes || cc.lanes}`,
    `累加器 ${c.acc_kib} KiB（超出的部分和行溢出到 SRAM）`,
    `DRAM 效率 ${sc.mem_eff ?? (S.memInfo ? S.memInfo.efficiency : 0.7)}；预留 1 GiB；暂存区 = max(2 MiB, 2·最大激活)`,
    (sc.package_cards || 1) > 1
      ? `两层互连：每封装 ${sc.package_cards} 卡走 D2D ${sc.d2d.GBps} GB/s、α ${sc.d2d.alpha_us} µs；封装之间走网络层 ${sc.link.GBps} GB/s、α ${sc.link.alpha_us} µs。卡按 TP → SP → DP → PP 编号，通信组按落在同一封装内的成员数 k 分层（allreduce / allgather 两级、all-to-all 按比例分摊；PP 交接是否跨封装按流水级边界）`
      : `链路 ${sc.link.GBps} GB/s，每次集合通信同步 α = ${sc.link.alpha_us} µs（${sc.link.topology === 'ring' ? '环形' : '交换'}拓扑；每封装 1 卡 = 无 D2D 层）`,
    c.slc_mib > 0
      ? `系统级缓存 SLC ${c.slc_mib} MiB @ ${c.slc_GBps} GB/s，策略 ${c.slc_policy === 'lru' ? 'lru（循环访问：片外工作集放得下全部读命中，否则 0 命中——上界 / 下界之间）' : 'pin（SRAM 之后按热权重 → 专家 → KV / 状态钉住，流式激活与 KV 写入绕过）'}；不增加容量；文本编码器 / VAE 不用 SLC`
      : '无系统级缓存（SLC MiB = 0）',
    '芯片不支持的权重格式：反量化每元素 2 次向量操作',
  ];
  if (r.summary.gen) {
    const g = r.summary.gen;
    items.splice(4, 1, `DRAM 效率 ${sc.mem_eff ?? (S.memInfo ? S.memInfo.efficiency : 0.7)}；预留 1 GiB；激活超出 SRAM/2 时分块流式进出 DRAM（GEMM 取激活分块 / 权重分块中较省者；注意力按 flash 式 K/V 重读）`);
    if (g.unit === 'frame') items.push(
      `每个去噪步 ${g.workload.cfg} 次前向（CFG 的 cond / uncond 作为 batch，DP 可切分 = CFG 并行），${g.workload.steps} 步；单段延迟 = ${g.pipeline ? '文本编码 + ' : ''}步数 × max(微批, PP) × 最重流水级时间${g.pipeline ? ' + VAE 解码' : ''}${g.pipeline && g.pipeline.load_s ? ' + 主机重载' : ''}`,
      g.pipeline
        ? `文本编码器与 VAE 解码：按发布检查点头的算子图与去噪串行执行「假设」（${g.pipeline.parts.map((p) => `${p.label} ${fmtDur(p.s)} · ${p.bound}${p.tiles ? ` · ${p.tiles} 个 tile（重叠 ×${num(p.overlap)}${p.tiling ? '，分块解码' : ''}）` : ''}${p.par ? ` · 拆到 ${p.par} 卡（最慢卡 ${p.rank_tiles} 个 tile，all-gather ${fmtDur(p.gather_s)}；单卡 ${fmtDur(p.single_s)}）` : ''}`).join('；')}）；${g.pipeline.overlap ? `跨请求重叠：主机编码下一请求与卡上去噪并行，稳态周期 ${fmtDur(g.period_s)}（单段延迟不变）` : '吞吐按单请求串行计（跨请求重叠仅在文本编码器放主机 CPU 时可选）'}`
          + `；组件放置${g.pipeline.placement === 'auto' ? '（自动）' : ''}：${placeText(g.pipeline, sc)}`
        : '文本编码器与 VAE 解码未计时、未计存储（工作负载里已取消勾选「计入文本编码器与 VAE 解码」）',
      '时间步嵌入与 AdaLN 调制按每序列一次计入',
      `单段延迟 SLO ${fmtDur(g.slo_s)}（「假设」，可在左侧修改）`);
    else items.push(r.model.is_pair
        ? `结构预测：主干（pair / MSA 表示）${g.workload.recycles} 遍${g.workload.diff_steps ? ` + 扩散 ${g.workload.diff_steps} 步 × ${g.workload.samples} 样本` : ' + 结构模块'}，无 KV 缓存；批延迟 = (微批 + PP − 1) × 最重流水级时间`
          + ((sc.layout.sp || 1) > 1 ? `；DAP ${sc.layout.sp}：pair / MSA / 模板网格按残基轴切到 ${sc.layout.sp} 卡，单一 / 原子轨道每卡重复，通信按 FastFold DAP（all-gather / all-to-all，与计算重叠取 max，每次集合通信 α）${g.workload.samples > 1 ? (sc.workload.sample_split !== false ? `；扩散 ${g.workload.samples} 个样本分到各卡（每卡 ${Math.ceil(g.workload.samples / sc.layout.sp)} 条）` : '；扩散样本每卡重复') : ''}` : '')
        : '单次编码器前向（双向注意力，无 KV 缓存）；批延迟 = (微批 + PP − 1) × 最重流水级时间',
      `批延迟 SLO ${r.model.is_pair ? num(g.slo_ms / 1e3) + ' s' : num(g.slo_ms) + ' ms'}（「假设」）；激活 dtype ${r.model.act_fmt || 'bf16'}${r.model.what_if ? '（what-if）' : `（「假设」${/^W fp32/.test(r.model.dtype) ? '，发布权重 fp32' : ''}；可用激活 dtype what-if 看 fp32 激活）`}`);
  } else items.splice(6, 0, `投机解码：k = ${sc.serving.spec_k}，接受率 ${sc.serving.spec_accept}（期望 token = (1−a^(k+1))/(1−a)）`,
    'MoE：每 rank 命中专家数取 max(局部期望, 全局期望/EP)；token 均匀路由');
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
function scopeText() {
  const sv = S.sc.serving, m = model();
  const mi = S.memInfo;
  const w = S.sc.workload, mw = m.workload;
  const tail = m.domain === 'gen' ? `${w.width || mw.width}×${w.height || mw.height} · ${w.frames || mw.frames} 帧 · ${w.steps || mw.steps} 步 · 单段 SLO ${fmtDur(w.clip_slo_s)}`
    : m.domain === 'protein' ? `${w.seq_len || mw.seq_len} 残基 · 批延迟 SLO ${m.is_pair ? w.fold_slo_s + ' s' : w.seq_slo_ms + ' ms'}` : `ctx ${sv.ctx} · TPOT SLO ${sv.tpot_slo_ms} ms`;
  return `场景：${m.label} · ${cards()} 卡 · 芯片 ${S.preset} · 存储器 ${mi ? `${mi.kind} ${num(mi.raw_GBps)} GB/s ${num(mi.capacity_GiB)} GiB` : S.sc.mem_id} · ${tail}（在左侧面板修改）`;
}
function paintScope() { $('cmp-scope').textContent = scopeText(); $('lay-scope').textContent = scopeText(); }
async function runCompare() {
  paintScope();
  const btn = $('cmp-run');
  btn.disabled = true;
  $('cmp-status').textContent = '搜索中（5 种映射 × 全部布局）…';
  delete $('cmp-status').dataset.done;
  const t0 = performance.now();
  try {
    const b = body({ cards: cards(), objective: S.cmpObj });
    const r = await track(req('compare', '/api/compare', b));
    if (!r) return;
    const mySeq = seq.compare;
    const stab = {};
    paintCompare(r, stab);
    const none = r.rows.every((x) => !x.batch);
    $('cmp-status').textContent = none ? `${cards()} 卡下任何布局都放不下或不满足 SLO —— 先在「单点评估」按提示修正卡数` :
      `完成 · ${((performance.now() - t0) / 1000).toFixed(1)} s · 稳定性计算中（0/${r.rows.length}）…`;
    if (none) return;
    let done = 0;
    $('cmp-status').dataset.done = '1';
    // the server runs these in parallel worker processes; each row fills in as its result arrives
    const one = async (row) => {
      const bb = JSON.parse(JSON.stringify(b));
      bb.scenario.mapping = row.mapping;
      try {
        const st = await track(req('stab-' + row.mapping, '/api/stability', bb));
        if (st === null || seq.compare !== mySeq) return false;
        stab[row.mapping] = st;
      } catch (e) { stab[row.mapping] = { error: e.message }; }
      if (seq.compare !== mySeq) return false;
      done++;
      $('cmp-status').textContent = `完成 · ${((performance.now() - t0) / 1000).toFixed(1)} s · 稳定性计算中（${done}/${r.rows.filter((x) => x.batch).length}）…`;
      paintCompare(r, stab);
      return true;
    };
    const res = await Promise.all(r.rows.filter((x) => x.batch).map(one));
    if (res.some((x) => !x)) return;
    $('cmp-status').textContent = `完成 · ${((performance.now() - t0) / 1000).toFixed(1)} s`;
  } catch (e) { $('cmp-status').textContent = '错误：' + e.message; }
  finally { btn.disabled = false; }
}
function latCell(x) { return x.unit === 'token' || !x.unit ? num(x.tpot_ms) : fmtDur(x.latency_ms / 1e3); }
function paintCompare(r, stab) {
  const gp = r.objective === 'goodput';
  const full = isFull(model());
  const ux = full ? `${UNIT_ZH[model().domain === 'gen' ? 'frame' : 'seq']}/s/卡` : 'tok/s/卡';
  const head = h('tr', {}, h('th', { class: 'l' }, '映射组织'), h('th', { class: 'l' }, '最佳布局 / 次优'), h('th', {}, 'batch'),
    h('th', {}, ux), full ? null : h('th', {}, 'goodput/卡'), h('th', {}, full ? (model().domain === 'gen' ? '单段延迟' : '批延迟') : 'TPOT ms'),
    h('th', { class: 'l' }, '瓶颈 · 有效 MAC'), full ? null : h('th', {}, 'prefill TTFT'), h('th', {}, '排名稳定性'));
  const rows = r.rows.map((x) => {
    const ok = !!x.batch;
    return h('tr', { class: (ok ? 'click ' : '') + (x.mapping === r.best_mapping && ok ? 'best' : ''), title: ok ? '点击把映射、布局（含卡数）和 batch 应用到场景' : '',
      onclick: ok ? () => applyRow(x) : null },
      h('td', { class: 'l' }, x.label),
      h('td', { class: 'l mono' }, ok ? x.layout : '放不下',
        x.runner_up ? h('div', { class: 'small muted' }, `次优 ${x.runner_up.layout}（${num(x.runner_up.score)}）`) : null),
      h('td', {}, ok ? x.batch : '—'),
      h('td', { style: gp ? '' : 'font-weight:700' }, ok ? num(x.tok_s_card) : '—'),
      full ? null : h('td', { style: gp ? 'font-weight:700' : '' }, ok ? num(x.goodput_card) : '—'), h('td', {}, ok ? latCell(x) : '—'),
      h('td', { class: 'l nowrap' }, ok ? boundTag(x.bound) : '—', ' ',
        ok ? h('span', { class: 'ubar sm' }, h('i', { style: `width:${Math.min(100, (x.array_util || 0) * 100)}%` })) : null, ok ? pct(x.array_util) : ''),
      full ? null : h('td', {}, ok ? ttftFlag(x.ttft_ok, x.ttft_ms) : '—'), h('td', {}, ok ? stabFlag(stab[x.mapping]) : '—'));
  });
  put($('cmp-tbl'), h('thead', {}, head), h('tbody', {}, rows));
}
function applyRow(x) {
  // mapping + full layout (card count follows from PP·TP·DP) + the batch the search found
  S.sc.mapping = x.mapping;
  paintSeg('mapping', x.mapping);
  $('mapping-desc').textContent = MAP_DESC[x.mapping];
  Object.assign(S.sc.layout, x.layout_obj);
  S.sc.serving.batch = x.batch;
  syncInputs(); cardsNote();
  activate('eval');
  schedule();
}
async function bestLayout() {
  const btn = $('best-layout');
  btn.disabled = true;
  const old = btn.textContent;
  btn.textContent = '搜索中…';
  try {
    const r = await track(req('best-layout', '/api/layouts', body({ cards: cards(), objective: 'decode' })));
    if (!r) return;
    const top = r.rows[0];
    if (!top || !top.batch) { showErr(`${cards()} 卡下没有满足容量和 SLO 的布局`); return; }
    applyLayout(top.layout_obj, top.batch);
  } catch (e) { showErr('布局搜索：' + e.message); }
  finally { btn.disabled = false; btn.textContent = old; }
}

/* ------------------------------------------------------------------ layouts + stability */
async function runLayouts() {
  paintScope();
  $('lay-status').textContent = '搜索中…';
  delete $('lay-status').dataset.done;
  const t0 = performance.now();
  try {
    const r = await track(req('layouts', '/api/layouts', body({ cards: cards(), objective: S.layObj })));
    if (!r) return;
    const hasG = r.rows.some((x) => x.goodput_card !== null && x.goodput_card !== undefined);
    const full = isFull(model());
    const head = h('tr', {}, h('th', {}, '#'), h('th', { class: 'l' }, '布局'), h('th', {}, 'batch'),
      h('th', {}, full ? `${UNIT_ZH[model().domain === 'gen' ? 'frame' : 'seq']}/s/卡` : 'tok/s/卡'),
      hasG ? h('th', {}, 'goodput/卡') : null, h('th', {}, full ? (model().domain === 'gen' ? '单段延迟' : '批延迟') : 'TPOT ms'), h('th', {}, '瓶颈'), h('th', { class: 'l' }, '有效 MAC'),
      hasG ? h('th', {}, 'prefill TTFT') : null);
    const rows = r.rows.map((x, i) => h('tr', { class: (x.batch ? 'click ' : '') + (i === 0 && x.batch ? 'best' : ''), onclick: x.batch ? () => applyRow(x) : null },
      h('td', {}, i + 1), h('td', { class: 'l mono' }, x.layout), h('td', {}, x.batch || '放不下'), h('td', {}, num(x.tok_s_card)),
      hasG ? h('td', {}, num(x.goodput_card)) : null, h('td', {}, x.batch ? latCell(x) : '—'), h('td', {}, boundTag(x.bound)),
      h('td', { class: 'l' }, h('span', { class: 'ubar' }, h('i', { style: `width:${Math.min(100, (x.array_util || 0) * 100)}%` })), pct(x.array_util)),
      hasG ? h('td', {}, x.ttft_ok === undefined ? '—' : ttftFlag(x.ttft_ok, x.ttft_ms)) : null));
    put($('lay-tbl'), h('thead', {}, head), h('tbody', {}, rows));
    $('lay-status').textContent = `${r.n_layouts} 个布局 · 精确前 ${r.rows.length} 名 · ${((performance.now() - t0) / 1000).toFixed(1)} s`;
    $('lay-status').dataset.done = '1';
  } catch (e) { $('lay-status').textContent = '错误：' + e.message; }
}
async function runStability() {
  paintScope();
  put($('stab-box'), h('div', { class: 'stab muted' }, '稳定性计算中（每个扰动重新精确搜索 top-1 布局）…'));
  try {
    const r = await track(req('stab', '/api/stability', body({ cards: cards(), objective: S.layObj, include_mapping: false })));
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
  'chip.gemv_macs': 'GEMV MAC/cycle', mem_eff: 'DRAM 效率', 'link.GBps': '网络层 GB/s', 'link.alpha_us': '网络 α µs',
  'chip.slc_mib': 'SLC MiB', 'chip.slc_GBps': 'SLC GB/s', 'd2d.GBps': 'D2D GB/s', 'd2d.alpha_us': 'D2D α µs', package_cards: '每封装卡数',
  'workload.frames': '帧数', 'workload.steps': '去噪步数', 'workload.height': '高 px', 'workload.width': '宽 px',
  'workload.seq_len': '序列长度（残基）', 'workload.msa': 'MSA 行数', 'workload.recycles': '主干遍数',
  'workload.samples': '扩散样本数',
};
const SWEEP_DOMAIN = { 'serving.ctx': 'llm', 'serving.prompt': 'llm', 'serving.spec_k': 'llm', 'workload.frames': 'gen',
  'workload.steps': 'gen', 'workload.height': 'gen', 'workload.width': 'gen', 'workload.seq_len': 'protein',
  'workload.msa': 'pair', 'workload.recycles': 'pair', 'workload.samples': 'pair' };
function fillSweepPaths() {
  const m = model(), d = isFull(m) ? m.domain : 'llm';
  const cur = $('sw-path').value;
  const list = S.cat.sweep_paths.filter((p) => !SWEEP_DOMAIN[p] || SWEEP_DOMAIN[p] === d
    || (SWEEP_DOMAIN[p] === 'pair' && m.is_pair && (p !== 'workload.msa' || m.workload.msa || m.workload.xmsa)
        && (p !== 'workload.samples' || m.workload.diff_steps))
    || (p === 'workload.steps' && m.is_pair && m.workload.diff_steps));
  opts($('sw-path'), list.map((p) => [p, SWEEP_ZH[p] || p]), list.includes(cur) ? cur : 'serving.batch');
  if (!list.includes(cur)) $('sw-values').value = SWEEP_DEFAULT[$('sw-path').value] || '';
}
const SWEEP_DEFAULT = {
  'serving.batch': '1,2,4,8,16,32,64', 'serving.ctx': '1024,4096,16384,32768,131072', 'serving.prompt': '512,2048,8192,32768',
  'serving.spec_k': '0,1,2,3,4,6,8', 'chip.sram_mib': '16,32,64,128,256,512', 'chip.sram_port_Bpc': '2048,4096,7616,16384,32768',
  'chip.freq_ghz': '0.6,0.8,1,1.2,1.5', 'chip.mac_eff': '0.5,0.6,0.7,0.8,0.9,1', 'chip.gemv_macs': '1024,4096,12544,50176',
  mem_eff: '0.5,0.6,0.7,0.8,0.9', 'link.GBps': '50,100,200,400,900', 'link.alpha_us': '0,1,3,5,10',
  'chip.slc_mib': '0,64,256,1024,4096', 'chip.slc_GBps': '500,1000,2000,4000', 'd2d.GBps': '500,1000,2000,4000',
  'd2d.alpha_us': '0,0.5,1,2', package_cards: '1,2,4,8',
  'workload.frames': '17,33,49,81,121', 'workload.steps': '10,20,30,50', 'workload.height': '240,480,720',
  'workload.width': '416,832,1280', 'workload.seq_len': '128,256,512,1022,2048',
  'workload.msa': '64,256,512,1024,4096', 'workload.recycles': '1,2,3,4,6', 'workload.samples': '1,5,10,25',
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
  for (const p of points) svg.append(s('text', { x: X(p.x), y: H - P.b + 16, 'text-anchor': 'middle', fill: '#8fa3b8', 'font-size': 11 }, p.xlabel ?? (Number.isInteger(p.x) ? p.x.toLocaleString('en-US') : num(p.x))));
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
    const m = model(), full = isFull(m), gen = m.domain === 'gen';
    const decode = S.sc.serving.phase === 'decode';
    const u = full ? UNIT_ZH[gen ? 'frame' : 'seq'] : 'tok';
    const latL = full ? (gen ? '单段延迟 s' : '批延迟 ms') : decode ? 'TPOT ms' : 'TTFT ms';
    const lat = (x) => full ? (gen ? x.latency_ms / 1e3 : x.latency_ms) : decode ? x.tpot_ms : x.ttft_ms;
    put($('sw-chart'), line(r.rows.map((x) => ({ x: x.value, y: x.tok_s_card, y2: lat(x) })),
      { xl: SWEEP_ZH[path] || path, yl: `${u}/s/卡`, y2l: latL, log: vals.every((v) => v > 0) && Math.max(...vals) / Math.min(...vals) >= 16 }));
    const head = h('tr', {}, h('th', { class: 'l' }, SWEEP_ZH[path] || path), h('th', {}, latL),
      h('th', {}, `${u}/s`), h('th', {}, `${u}/s/卡`), h('th', {}, '瓶颈'), h('th', {}, '有效 MAC'), h('th', {}, '放得下'));
    put($('sw-tbl'), h('thead', {}, head), h('tbody', {}, r.rows.map((x) => h('tr', {},
      h('td', { class: 'l' }, x.value), h('td', {}, num(lat(x))), h('td', {}, num(x.tok_s)),
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
    const m = model(), full = isFull(m), gen = m.domain === 'gen';
    const u = full ? UNIT_ZH[gen ? 'frame' : 'seq'] : 'tok';
    const latL = full ? (gen ? '单段延迟 s' : '批延迟 ms') : 'TPOT ms';
    const lat = (p) => full ? (gen ? p.latency_ms / 1e3 : p.latency_ms) : p.tpot_ms;
    const thr = (p) => full ? p.units_s_card : p.tok_s_card;
    const pts = r.front.map((p) => ({ x: p.batch, xlabel: 'B' + p.batch, y: thr(p), y2: lat(p) }));
    put($('sw-chart'), line(pts, { xl: `batch（布局 ${r.layout}，Pareto 前沿：延迟 ↑ 换吞吐 ↑）`, yl: `${u}/s/卡`, y2l: latL, log: true }));
    const head = h('tr', {}, h('th', { class: 'l' }, 'batch'), h('th', {}, latL), h('th', {}, `${u}/s/卡`));
    put($('sw-tbl'), h('thead', {}, head), h('tbody', {}, r.front.map((p) => h('tr', {},
      h('td', { class: 'l' }, p.batch), h('td', {}, num(lat(p))), h('td', {}, num(thr(p)))))));
    $('sw-status').textContent = `Pareto ${r.front.length} 点`;
    $('sw-status').dataset.done = '1';
  } catch (e) { $('sw-status').textContent = '错误：' + e.message; }
}

/* ------------------------------------------------------------------ models table */
function renderModels() {
  const head = h('tr', {}, h('th', { class: 'l' }, '模型'), h('th', { class: 'l' }, '系列'), h('th', { class: 'l' }, '来源'),
    h('th', { class: 'l' }, '覆盖'), h('th', { class: 'l' }, '近似之处'), h('th', { class: 'l' }, 'dtype（按发布）'),
    h('th', {}, '参数 B'), h('th', {}, '激活 B'), h('th', {}, 'vs 发布'), h('th', { class: 'l' }, '结构'));
  const NC = 10;
  const rows = [];
  const all = catalogRows();
  for (const g of vendorGroups(all)) {
    const ne = g.items.filter((m) => m.evaluable).length, no = g.items.length - ne;
    rows.push(h('tr', { class: 'grp dom', 'data-vendor': g.key }, h('td', { class: 'l', colspan: NC }, g.label,
      h('span', { class: 'cnt' }, [ne ? `${ne} 个可评估` : null, no ? `${no} 个暂未接入 v2` : null].filter(Boolean).join(' · ')))));
    let fam = null;
    for (const m of g.items) {
      const famCell = h('td', { class: 'l' + (m.family === fam ? ' muted' : '') }, m.family);
      fam = m.family;
      if (!m.evaluable) {
        rows.push(h('tr', { class: 'off', 'data-id': m.id, 'data-domain': m.domain },
          h('td', { class: 'l' }, m.label.replace(/_/g, '_\u200b'), h('div', {}, h('span', { class: 'badge dom' }, OFF_DOMAIN[m.domain])),
            h('div', { class: 'small muted mono' }, m.hf_id || 'github: ' + m.source.replace(/^https:\/\/github\.com\//, '').split('/').slice(0, 2).join('/'))),
          famCell, h('td', { class: 'l' }, '—'), h('td', { class: 'l' }, offBadge(m)),
          h('td', { class: 'wrap small muted' }, `${m.arch_detail}；${OFF_DOMAIN[m.domain]}模型在 core v2 中暂未接入，不能评估${m.reason ? '：' + m.reason : ''}`),
          h('td', { class: 'l' }, '—'), h('td', {}, '—'), h('td', {}, '—'), h('td', {}, '—'), h('td', { class: 'l' }, m.arch)));
        continue;
      }
      const approx = [...(m.coverage_reasons || []), ...(m.vision_params_B ? [`视觉编码器（${num(m.vision_params_B)}B）未建模`] : [])];
      rows.push(h('tr', { class: 'click', 'data-id': m.id, 'data-domain': m.domain, onclick: () => { setModel(m.id); activate('eval'); schedule(); } },
        h('td', { class: 'l' }, m.label, m.domain === 'vlm' ? h('div', {}, h('span', { class: 'badge vlm', title: coverTip(m) }, 'VLM · 视觉编码器未建模'))
          : DOMAIN_ZH[m.domain] ? h('div', {}, h('span', { class: 'badge dom' }, DOMAIN_ZH[m.domain] + ' · 可评估')) : null,
          h('div', { class: 'small muted mono' }, m.hf_id),
          m.same_as.length ? h('div', { class: 'small muted' }, '同结构：' + m.same_as.map((x) => x.label).join('、')) : null),
        famCell,
        h('td', { class: 'l' }, h('span', { class: 'badge ' + m.provenance }, PROV_ZH[m.provenance])),
        h('td', { class: 'l' }, coverBadge(m)),
        h('td', { class: 'wrap small' }, approx.length ? approx.join('；') : h('span', { class: 'muted' }, '—')),
        h('td', { class: 'l' }, m.dtype), h('td', {}, num(m.params_B)), h('td', {}, num(m.active_B)),
        h('td', {}, (m.param_err * 100).toFixed(2) + '%'), h('td', { class: 'l' }, m.arch)));
    }
  }
  const cnt = (f) => all.filter(f).length;
  put($('models-summary'), `共 ${all.length} 个：可评估 ${cnt((m) => m.evaluable)} 个（LLM ${cnt((m) => m.domain === 'llm')} · VLM ${cnt((m) => m.domain === 'vlm')}`
    + ` · 视频生成 ${cnt((m) => m.evaluable && m.domain === 'gen')} · 蛋白质 ${cnt((m) => m.evaluable && m.domain === 'protein')}）；`,
    `暂未接入 v2（灰色行，不能评估）${cnt((m) => !m.evaluable)} 个（视频生成 ${cnt((m) => !m.evaluable && m.domain === 'gen')} · 蛋白质 ${cnt((m) => !m.evaluable && m.domain === 'protein')}）。`,
    '分子动力学 / 机器学习力场（MLFF）目录中暂无条目。');
  put($('models-tbl'), h('thead', {}, head), h('tbody', {}, rows));
  put($('models-unlisted'), ...S.unlisted.map((u) => h('li', {}, h('b', {}, u.label), '：', u.reason)));
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
    S.offline = models.offline;
    S.catalog = models.catalog;
    S.unlisted = models.unlisted;
    for (const m of S.models) S.byId[m.id] = m;
    $('ver').textContent = 'v' + health.version;
    $('honesty-chip').title = cat.honesty;
    $('honesty-text').textContent = cat.honesty;
    S.sc = JSON.parse(JSON.stringify(cat.defaults));
    delete S.sc.chip;
    // UI default: LPDDR-class memory → 100 ms TPOT SLO and auto batch, so the opening scenario is feasible and meets its SLO
    S.sc.serving.tpot_slo_ms = 100;
    S.best = true;
    const mi = await req('mi', '/api/memory', { id: S.sc.mem_id });
    S.memInfo = mi;
    S.mem = { ...mi.fields };
  } catch (e) { showErr('初始化失败：' + e.message); return; }

  fillModels();
  seg('chip-preset', Object.keys(S.cat.chips).map((k) => [k, k]), () => S.preset, (v) => { S.preset = v; S.chipOver = {}; onPreset(); syncInputs(); schedule(); });
  for (const k of ['freq_ghz', 'sram_mib', 'sram_port_Bpc', 'gemv_macs', 'mac_eff', 'acc_kib', 'slc_mib', 'slc_GBps']) {
    const nullable = k === 'sram_port_Bpc' || k === 'gemv_macs';
    bindNumber('c-' + k, () => chipVal(k), (x) => { if (x === null) delete S.chipOver[k]; else S.chipOver[k] = x; if (k === 'freq_ghz') onPreset(); },
      { int: k === 'gemv_macs', nullable });
  }
  seg('mapping', S.cat.mappings.map((m) => [m.id, m.label]), () => S.sc.mapping, (v) => { S.sc.mapping = v; $('mapping-desc').textContent = MAP_DESC[v]; schedule(); });
  $('mapping-desc').textContent = MAP_DESC[S.sc.mapping];
  for (const k of ['pp', 'tp', 'dp', 'ep', 'etp', 'sp']) {
    $('l-' + k).dataset.after = 'layout';
    bindNumber('l-' + k, () => S.sc.layout[k], (x) => { S.sc.layout[k] = x; AFTER.lastLayout = k; }, { int: true });
  }
  bindNumber('l-cards', cards, (x) => { setCards(x); for (const k of ['pp', 'tp', 'dp', 'ep', 'etp', 'sp']) $('l-' + k)._sync(); cardsNote(); }, { int: true });
  for (const k of ['frames', 'height', 'width', 'steps', 'cfg', 'seq_len', 'msa', 'recycles', 'samples'])   // empty = release default (0)
    bindNumber('w-' + k, () => S.sc.workload[k] || null, (x) => (S.sc.workload[k] = x === null ? 0 : x), { int: true, nullable: true });
  bindNumber('w-psteps', () => S.sc.workload.steps || null, (x) => (S.sc.workload.steps = x === null ? 0 : x), { int: true, nullable: true });
  for (const k of ['clip_slo_s', 'seq_slo_ms', 'fold_slo_s']) bindNumber('w-' + k, () => S.sc.workload[k], (x) => (S.sc.workload[k] = x));
  bindNumber('w-host_GBps', () => S.sc.workload.host_GBps, (x) => (S.sc.workload.host_GBps = x));
  bindNumber('w-host_TFLOPS', () => S.sc.workload.host_TFLOPS, (x) => (S.sc.workload.host_TFLOPS = x));
  for (const k of ['pJ_mac', 'pJ_vec', 'pJ_bit_sram', 'pJ_bit_dram', 'pJ_bit_link', 'idle_W', 'pJ_bit_slc', 'pJ_bit_d2d'])
    bindNumber('e-' + k, () => (S.energy || {})[k] ?? null, (x) => { S.energy = { ...(S.energy || {}), [k]: x }; }, { nullable: true });
  $('best-layout').addEventListener('click', bestLayout);
  $('best-batch').checked = S.best;
  AFTER.layout = () => {
    const k = AFTER.lastLayout;
    if (k !== 'pp') fixMoe(k);
    if (S.sc.layout.pp > model().n_layers) S.sc.layout.pp = model().n_layers;
    for (const x of ['pp', 'ep', 'etp']) if (x !== k) $('l-' + x)._sync();
    cardsNote();
  };
  for (const k of ['GBps', 'alpha_us']) bindNumber('k-' + k, () => S.sc.link[k], (x) => (S.sc.link[k] = x));
  for (const k of ['GBps', 'alpha_us']) bindNumber('d-' + k, () => S.sc.d2d[k], (x) => (S.sc.d2d[k] = x));
  bindNumber('p-package_cards', () => S.sc.package_cards || 1, (x) => (S.sc.package_cards = x), { int: true });
  $('c-slc_policy').addEventListener('change', (e) => { S.chipOver.slc_policy = e.target.value; schedule(); });
  for (const k of ['batch', 'ctx', 'prompt', 'out_len', 'spec_k', 'microbatches'])
    bindNumber('s-' + k, () => S.sc.serving[k], (x) => (S.sc.serving[k] = x), { int: true });
  for (const k of ['spec_accept', 'tpot_slo_ms', 'ttft_slo_ms']) bindNumber('s-' + k, () => S.sc.serving[k], (x) => (S.sc.serving[k] = x));
  bindNumber('mem_eff', () => S.sc.mem_eff, (x) => (S.sc.mem_eff = x), { nullable: true });
  seg('phase', null, () => S.sc.serving.phase, (v) => { S.sc.serving.phase = v; schedule(); });
  $('best-batch').addEventListener('change', (e) => { S.best = e.target.checked; schedule(); });
  $('wi-w').addEventListener('change', (e) => { S.wiW = e.target.value; schedule(); });
  $('wi-kv').addEventListener('change', (e) => { S.wiKV = e.target.value; schedule(); });
  $('wi-act').addEventListener('change', (e) => { S.wiAct = e.target.value; schedule(); });
  $('w-placement').addEventListener('change', (e) => { S.sc.workload.placement = e.target.value; schedule(); });
  $('w-vae_tiling').addEventListener('change', (e) => { S.sc.workload.vae_tiling = e.target.checked; schedule(); });
  for (const k of ['dit_fsdp', 'te_cpu', 'sample_split', 'vae_parallel', 'overlap']) $('w-' + k).addEventListener('change', (e) => { S.sc.workload[k] = e.target.checked; schedule(); });
  $('w-pipeline').addEventListener('change', (e) => { S.sc.workload.pipeline = e.target.checked; put($('model-badges'), ...badges(model(), S.wiW || S.wiKV || (S.wiAct && isFull(model())))); schedule(); });
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
  fillSweepPaths();
  renderModels();
  paintScope();
  window.__accel = { S, req, seq };
  runEval();
}
init();

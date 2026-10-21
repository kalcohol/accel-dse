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
  if (a === 0) return '0';
  // 0.130: round first, then pick the tier of the rounded value (99.96 was "100.0", 9999.6 "10000" vs "10,000")
  const dec = a >= 100 ? 0 : a >= 10 ? 1 : a >= 1 ? 2 : null;
  if (dec === null) {
    const t = x.toPrecision(Math.max(1, d - 1));
    return Math.abs(Number(t)) >= 1 ? num(Number(t), d) : t;
  }
  const r = Number(x.toFixed(dec));
  const dec2 = Math.abs(r) >= 1e4 ? -1 : Math.abs(r) >= 100 ? 0 : Math.abs(r) >= 10 ? 1 : 2;
  return dec2 === dec ? x.toFixed(dec) : num(r, d);
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
    (m.vision ? `\n· 视觉编码器 ${m.vision.label}（${num(m.vision_params_B)}B 参数，bf16）已按发布建模：图像数 / 分辨率在「服务」里设，默认 0 张 = 只算文本`
      : m.vision_params_B ? `\n· 视觉编码器（${num(m.vision_params_B)}B 参数）未建模：只评估语言主干` : '');
}
function coverBadge(m) {
  return h('span', { class: 'badge cov ' + m.coverage, title: coverTip(m) }, h('span', { class: 'ax' }, '覆盖'), COVER_ZH[m.coverage]);
}
const PROV_ZH = { official: '官方', mirror: '镜像' };
const DOMAIN_ZH = { gen: '视频生成', protein: '蛋白质', genomics: '基因组', asr: '语音识别', tts: '语音合成', omni: '全模态' };
const isFull = (m) => !!m && (m.domain === 'gen' || m.domain === 'protein' || m.domain === 'genomics');   // non-autoregressive full-sequence forward
const UNIT_ZH = { frame: '帧', seq: '序列', token: 'tok', window: '窗口' };
// 0.119: throughput unit of a full-forward model — image DiT 张, video 帧, genomics 窗口, protein 序列
const unitZh = (m) => (m.domain === 'gen' ? (m.workload && m.workload.image ? '张' : '帧') : m.domain === 'genomics' ? '窗口' : '序列');
const clipZh = (m) => (m && m.workload && m.workload.image ? '单张延迟' : '单段延迟');   // 0.130: image DiTs
const isSeqEnc = (m) => !!m && (m.domain === 'protein' || m.domain === 'genomics');   // per-sequence encoders (0.106)
function fmtDur(sec) {
  if (sec === null || sec === undefined || !isFinite(sec)) return '—';
  if (sec >= 7200) return num(sec / 3600) + ' h';
  if (sec >= 120) return num(sec / 60) + ' min';
  if (sec >= 1) return num(sec) + ' s';
  return num(sec * 1e3) + ' ms';
}
function domainBadge(m) {
  if (m.domain === 'gen') {        // 0.119: image DiTs (one latent frame) are labelled as image generation
    const g = m.workload && m.workload.image ? '图像生成' : '视频生成';
    return h('span', { class: 'badge vlm', title: coverTip(m) },
      S.sc.workload.pipeline === false ? `${g} · 只评估 DiT 主干` : `${g} · DiT + 文本编码器 + VAE 解码`);
  }
  if (m.domain === 'llm' && /^HuggingFaceBio\/Carbon-/.test(m.hf_id || ''))     // 0.119: Carbon base DNA LMs (decoder path)
    return h('span', { class: 'badge vlm', title: coverTip(m) }, '基因组 · DNA 自回归语言模型（按 LLM 解码评估）');
  if (m.domain === 'protein') return h('span', { class: 'badge vlm', title: coverTip(m) }, m.is_pair ? '蛋白质 · 结构预测' : '蛋白质 · 编码器前向');
  if (m.domain === 'genomics') return h('span', { class: 'badge vlm', title: coverTip(m) }, '基因组 · 双向编码器前向（逐 token 分类）');
  // 0.268: audio-only release (Ming-3.0-asr-tiny) — the encoder is costed via serving.audio_s (API / CLI --audio-s)
  if (m.domain === 'tts' && /CosyVoice/.test(m.hf_id || '')) return h('span', { class: 'badge vlm', title: coverTip(m) }, '语音合成 · LM 每秒 25 个语音 token → 分块流匹配 → HiFT 声码器（按输出音频秒计）');   // 0.271
  if (m.domain === 'tts' && m.family === 'Drama 3') return h('span', { class: 'badge vlm', title: coverTip(m) },   // 0.276
    'TTS · Qwen3.5-MoE 主干 + 13 层码本塔 · 21.5 帧/s + DAC 解码（同卡）');
  if (m.domain === 'tts' && m.family === 'FireRedTTS3') return h('span', { class: 'badge vlm', title: coverTip(m) },   // 0.275
    '语音合成 · Qwen3-1.7B 每步 → DiT 流匹配头（10 步 × CFG 2）出 4 帧潜变量（160 ms）→ 整段生成完后 RedAE 一次解码（非流式）');
  if (m.domain === 'tts') return h('span', { class: 'badge vlm', title: coverTip(m) }, '语音合成 · 每 token = 1 帧潜变量（流匹配头 + AudioVAE 解码）');
  if (m.domain === 'omni' && m.family === 'Moshi') return h('span', { class: 'badge vlm', title: coverTip(m) },   // 0.274
    '全双工语音 · 每 80 ms 帧：Mimi 编码 → 时间 Transformer 7B 一步 → 深度 Transformer 8 步（每步独立权重切片，每帧复读）→ Mimi 解码');
  if (m.domain === 'omni' && m.family === 'Kimi-Audio') return h('span', { class: 'badge vlm', title: coverTip(m) },   // 0.273
    '全模态 · Whisper 编码器 → LLM（每步文本 token + 12.5 Hz 音频 token 并行两头）→ 流匹配 DiT 去分词器 → BigVGAN（按输出音频秒计）');
  if (m.domain === 'omni') return h('span', { class: 'badge vlm', title: coverTip(m) }, m.family === 'Step-Audio-2'   // 0.273
    ? '全模态 · 音频编码器 → LLM（文本 + 25 Hz 语音 token）→ token2wav（缓存式分块 flow + HiFT）语音输出（按输出音频秒计）'
    : '全模态 · Thinker（+ 音频编码器）→ Talker → token2wav 语音输出（按输出音频秒计）');   // 0.271
  if (m.domain === 'asr') return h('span', { class: 'badge vlm', title: coverTip(m) }, '语音识别 · 含音频编码器（serving.audio_s，按音频秒计）');
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
  cat: null, models: [], byId: {}, preset: '100T', chipOver: {}, mixed: [], sc: null,
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
  const pid = PROG.start(chan, path);
  let r, j;
  try {
    r = await fetch(path, opt);
    // 0.130: a non-JSON error body (proxy 502 page, truncated reply) showed "Unexpected token <" instead of the status
    j = await r.json().catch(() => null);
  } catch (e) {
    if (e.name === 'AbortError' || my !== seq[chan]) return null;
    throw e;
  } finally { PROG.end(pid); }
  if (my !== seq[chan]) return null;
  if (j === null) throw new Error(`HTTP ${r.status}：响应不是 JSON`);
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}
// a scenario edit makes an in-flight request of this channel stale: drop its reply (and, in the static build, free
// its worker — a running slow-lane search is terminated)
function cancel(chan) {
  seq[chan] = (seq[chan] || 0) + 1;
  if (ctl[chan]) { ctl[chan].abort(); ctl[chan] = null; }
}

/* ------------------------------------------------------------------ progress (non-blocking: never disables or dims
   inputs; overlay parts are pointer-events: none).  A thin animated bar at the top of the results area + a floating
   pill near the metric cards ("计算中… 3.2 s · 阶段"), shown only after 150 ms so fast evals don't flicker.  The static
   (Pyodide) build also reports its first-load phases here (download / init / package). */
const STAGE = { eval: 'eval 评估', sensitivity: 'sensitivity 敏感度', fit: 'fit 容量修正搜索', memory: 'memory 内存',
  compare: 'compare 映射对比', layouts: 'layouts 布局搜索', stability: 'stability 排名稳定性', sweep: 'sweep 扫描',
  pareto: 'pareto', insight: 'insight 设计洞察', insight_path: 'path 升级路径', insight_sizing: 'sizing 缩减余量', insight_slc: 'SLC 能耗最优', insight_speech: '语音瓶颈', insight_speech_streams: '语音实时路数 vs SLC', insight_dispatch: 'LLM 派发敏感度', insight_noc: 'NoC 能耗', insight_dse: '能耗设计搜索',
  health: 'catalog 载入目录', catalog: 'catalog 载入目录', models: 'catalog 载入目录' };
const LOAD_ZH = { download: '下载 Python 运行时', init: '初始化 Pyodide', package: '载入 accel_dse 包', ready: '就绪', error: '静态版加载失败' };
const PROG = {
  act: new Map(), n: 0, timer: null, load: null, DELAY: 150,
  start(chan, path) {
    const id = ++this.n;
    this.act.set(id, { chan, stage: path.replace(/^\/api\//, '').split('?')[0], t0: performance.now() });
    this.tick();
    return id;
  },
  end(id) { this.act.delete(id); this.tick(); },
  setLoad(st) { this.load = st && st.phase !== 'ready' ? { ...st, t0: (this.load && this.load.t0) || performance.now() } : null; this.tick(); },
  tick() {
    const el = $('busy');
    if (!el) return;
    const now = performance.now();
    const shown = [...this.act.values()].filter((a) => now - a.t0 >= this.DELAY).sort((a, b) => a.t0 - b.t0);
    const on = shown.length > 0 || !!this.load;
    el.classList.toggle('on', on);
    const ev = shown.some((a) => a.stage === 'eval' || a.stage === 'sensitivity' || a.stage === 'fit');
    $('kpis').classList.toggle('updating', ev);
    let txt = '';
    if (this.load) {
      const l = this.load;
      txt = l.phase === 'error' ? `${LOAD_ZH.error}：${l.msg || ''}`
        : `${LOAD_ZH[l.phase] || l.phase}… ${((now - l.t0) / 1000).toFixed(1)} s` + (l.mb ? ` · ${l.mb.toFixed(1)} MB` : '') + ' · 静态版首次加载';
    } else if (shown.length) {
      const a = shown[0];
      txt = `计算中… ${((now - a.t0) / 1000).toFixed(1)} s · ${STAGE[a.stage] || a.stage}` + (shown.length > 1 ? ` (+${shown.length - 1})` : '');
    }
    $('busy-pill').textContent = txt;
    el.dataset.stage = this.load ? 'load' : (shown[0] ? shown[0].stage : '');
    const want = this.act.size > 0 || !!this.load;
    if (want && !this.timer) this.timer = setInterval(() => this.tick(), 100);
    if (!want && this.timer) { clearInterval(this.timer); this.timer = null; }
  },
};
window.addEventListener('accel-static-progress', (e) => PROG.setLoad(e.detail));
if (window.__accelStatic && window.__accelStatic.progress) PROG.setLoad(window.__accelStatic.progress);   // phases before app.js loaded

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
  if (S.mixed.length) sc.chip.formats = { mixed: [...S.cat.chips[S.preset].mixed, ...S.mixed.flatMap((k) => S.cat.mixed_presets[k].pairs)] };
  sc.formats_override = overrides();
  if (sc.pd && !model().is_moe) Object.assign(sc.pd.prefill_layout, { dp: 1, ep: 1, etp: 1 });
  if (sc.pd && isFull(model())) sc.pd.enabled = false;
  if (!sc.d2d_enabled) sc.package_cards = 1;            // D2D off = monolithic die: one die per package
  if (S.memInfo) sc.mem_id = S.memInfo.id;
  const en = Object.fromEntries(Object.entries(S.energy || {}).filter(([, v]) => v !== null && v !== undefined));
  const bu = Object.fromEntries(Object.entries(S.budget || {}).filter(([, v]) => v !== null && v !== undefined));
  return { chip_preset: S.preset, scenario: sc, ...(Object.keys(en).length ? { energy: en } : {}),
    ...(Object.keys(bu).length ? { budget: bu } : {}), ...extra };
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
function bindNumber(id, get, set, { int = false, nullable = false, check = null } = {}) {
  const inp = $(id);
  inp.addEventListener('input', () => {
    const x = readNum(inp, int);
    const ok = x !== undefined && (x !== null || nullable) && (x === null || !check || check(x));
    inp.classList.toggle('bad', !ok);
    if (!ok) return;
    set(x);
    if (inp.dataset.after) AFTER[inp.dataset.after]();
    schedule();
  });
  inp._sync = () => { const v = get(); inp.value = v === null || v === undefined ? '' : v; inp.classList.remove('bad'); };
}
const AFTER = {};
/* three-tier interconnect (0.50): D2D off = monolithic die; on = catalog grade × units, or a custom figure */
function d2dStd() { return ((S.cat && S.cat.d2d_standards) || []).find((d) => d.id === S.sc.d2d_std); }
function d2dGBps() { const d = d2dStd(); return d ? d.per_unit_GBps * S.sc.d2d_units : S.sc.d2d.GBps; }
function d2dNote() {
  const d = d2dStd();
  $('d2d-note').textContent = d ? `${num(d.per_unit_GBps)} GB/s / ${d.unit} × ${S.sc.d2d_units} = ${num(d2dGBps())} GB/s；参考 ${d.pJ_bit} pJ/bit${d.open ? '' : '（厂商专有）'}。${d.note}` : '自定义 D2D 带宽「假设」';
  $('d2d-units').disabled = !d;
}
function d2dPaint() { $('d2d-inputs').hidden = !S.sc.d2d_enabled; d2dNote(); }
/* 0.59 topology-aware collectives */
function fab() { if (!S.sc.fabric) S.sc.fabric = JSON.parse(JSON.stringify(S.cat.defaults.fabric)); return S.sc.fabric; }
function fabPaint() {
  $('fab-enabled').checked = !!fab().enabled; $('fab-inputs').hidden = !fab().enabled;
  for (const k of ['algo', 'net_topology', 'innet_reduce', 'overlap']) $('fab-' + k).value = fab()[k];
  $('fab-protocol').value = fab().protocol || 'off';
  $('fab-net_tiers').value = String(fab().net_tiers || 2);
  $('fab-link_topology').value = S.sc.link.topology || 'switch';
  $('fab-contention').checked = !!fab().contention;
  $('fab-kv_feedback').checked = !!fab().kv_feedback;
  $('fab-kv_feedback_prefill').checked = !!fab().kv_feedback_prefill;
  const three = (fab().net_tiers || 2) === 3;
  $('fab-pod_nodes').disabled = !three; $('fab-oversub_spine').disabled = !three;
}
const FAB_ALGO_ZH = { ring: 'ring（扁平环）', tree: 'tree（双二叉树）', hier: 'hier（分层）', direct: 'direct（直接）', p2p: 'p2p', innet: 'innet（交换机内，NVLS 类）' };
function fabAlgoZh(a) { const [b, p] = String(a).split('/'); return (FAB_ALGO_ZH[b] || b) + (p ? ` · ${p}` : ''); }
const FAB_TOP_ZH = { ring: '顶层 ring', tree: '顶层 tree', innet: '顶层网内归约' };
function renderFabric(r) {
  const f = r.fabric;
  $('fab-box').hidden = !f;
  if (!f) return;
  const off = f.step_ms_off;
  const nt = f.net_traffic, stg = (f.stages || []).reduce((a, b) => (!a || b.link_us > a.link_us ? b : a), null);
  $('fab-basis').textContent = `${f.basis} 步时间 ${num(f.step_ms)} ms` + (off ? `（拓扑模型关时 ${num(off)} ms，${f.step_ms >= off ? '+' : ''}${pct(f.step_ms / off - 1)}）` : '')
    + (nt ? `。跨节点流量 ${num(nt.net_MB)} MB/步，其中 ${pct(nt.leaf_up_share)} 离开 leaf${f.pod_nodes ? `、${pct(nt.pod_up_share)} 离开 pod（每 pod ${f.pod_nodes} 节点）` : ''}` : '')
    + (f.overlap === 'ports' && stg ? `。端口并发：最重级链路 ${num(stg.link_us / 1e3)} ms（相加为 ${num(stg.link_sum_us / 1e3)} ms；忙时 ${Object.entries(stg.busy_us).filter(([, v]) => v > 0).map(([t, v]) => `${{ d2d: 'D2D', link: '节点内', net: '网络' }[t]} ${num(v / 1e3)}`).join(' · ')} ms）` : '');
  const tier = { d2d: 'D2D', link: '节点内', net: '网络' };
  const head = h('tr', {}, h('th', { class: 'l' }, '集合通信'), h('th', {}, '组大小'), h('th', { class: 'l' }, '分层'), h('th', {}, '每 rank 数据'),
    h('th', {}, '次数 / 步'), h('th', { class: 'l' }, '所选算法'), h('th', {}, '带宽时间 µs'), h('th', {}, '时延 µs'), h('th', {}, '离开 leaf / pod'), h('th', { class: 'l' }, '其他候选（带宽 + 时延 µs）'));
  const rows = f.rows.slice(0, 24).map((x) => h('tr', {}, h('td', { class: 'l' }, x.kind === 'alltoall' ? 'all-to-all' : x.kind),
    h('td', {}, String(x.group)), h('td', { class: 'l' }, x.levels.map(([t, n]) => `${tier[t] || t} × ${n}`).join(' · ')),
    h('td', {}, num(x.bytes / 1024) + ' KiB'), h('td', {}, String(x.count)),
    h('td', { class: 'l' }, fabAlgoZh(x.algo) + (x.top ? `，${FAB_TOP_ZH[x.top] || x.top}` : '')),
    h('td', {}, num(x.bw_us)), h('td', {}, num(x.alpha_us)),
    h('td', {}, x.net_leaf_share === undefined ? '—' : `${pct(x.net_leaf_share)}${f.pod_nodes ? ' / ' + pct(x.net_pod_share) : ''}`),
    h('td', { class: 'l small' }, Object.entries(x.cands).filter(([k]) => k !== x.algo).map(([k, v]) => `${k} ${num(v.bw_us + v.alpha_us)}`).join(' · ') || '—')));
  put($('fab-tbl'), h('thead', {}, head), h('tbody', {}, ...rows));
}
// 0.223: independent draft model picker — catalog LLMs; a different vocabulary (tokenizer) is flagged here and by the server
function fillDrafts() {
  const sel = $('s-draft_model');
  if (!sel || sel._filled === S.sc.model) return;
  sel._filled = S.sc.model;
  const opts = [h('option', { value: '' }, '（目标自带 MTP / DSpark）')];
  for (const m of S.models || []) if (m.domain === 'llm' && m.id !== S.sc.model) opts.push(h('option', { value: m.id }, m.label || m.id));
  put(sel, ...opts);
}
function draftWarn() {
  const w = $('s-draft-warn');
  if (!w) return;
  const d = S.byId[S.sc.serving.draft_model || ''], t = S.byId[S.sc.model];
  w.textContent = d && t && d.vocab && t.vocab && d.vocab !== t.vocab ? ` ⚠ 词表 ${d.vocab} ≠ 目标 ${t.vocab}（需同一分词器）` : '';
}
function treeWarn() {     // 0.225: client-side checks of the tree fields (the server's warnings show in #warns)
  const w = $('s-tree-warn');
  if (!w) return;
  const sv = S.sc.serving, k = sv.spec_k || 0, t = sv.spec_topk || 1, n = sv.spec_draft_tokens || 0, out = [];
  if (t > 1 && k) {
    const cap = 1 + t + (k - 1) * t * t;
    if (n && n - 1 > cap) out.push(`⚠ 树 token ${n} 超过 k = ${k}、topk = ${t} 的候选节点数 + 1（${cap + 1}）`);
    if (sv.spec_rollback === 'replay') out.push('⚠ 树形验证不支持 ReplaySSM，按快照计');
    const m = S.byId[S.sc.model];
    if (m && m.draft === 'dspark' && !sv.draft_model) out.push('⚠ DSpark 是块草稿，topk 被忽略');
    const am = sv.spec_tree_accept_model || 'chain';
    out.push(`树接受率主值 ${am}：对照 EAGLE-1 ${{ chain: '−17…−19 %（下界）', rank: '+1…+4 %（Sequoia 位次向量）', indep: '+49…+63 %（上界）' }[am]}`);
  } else if (n && k) out.push('线性链（topk 1）：树 token 被忽略');
  if (!k && (t > 1 || n)) out.push('spec_k = 0：未启用投机解码');
  w.textContent = out.join('；');
}
function syncInputs() {
  document.querySelectorAll('input[type=number]').forEach((i) => i._sync && i._sync());
  if ($('l-pp_split')) $('l-pp_split').value = S.sc.pp_split || 'cost';
  if ($('l-indexer')) $('l-indexer').value = S.sc.indexer || 'reference';
  if ($('l-attn_mode')) $('l-attn_mode').value = S.sc.attn_mode || 'reference';
  if ($('h-host_overlap')) $('h-host_overlap').value = S.sc.host_overlap || 'serial';
  if ($('s-spec_rollback')) $('s-spec_rollback').value = S.sc.serving.spec_rollback || 'snapshot';
  if ($('s-draft_model')) { fillDrafts(); $('s-draft_model').value = S.sc.serving.draft_model || ''; draftWarn(); }
  if ($('s-spec_tree_accept_model')) $('s-spec_tree_accept_model').value = S.sc.serving.spec_tree_accept_model || 'chain';
  if ($('s-spec_sampling')) $('s-spec_sampling').value = S.sc.serving.spec_sampling || 'greedy';      // 0.253
  if ($('s-spec_accept_src')) $('s-spec_accept_src').value = S.sc.serving.spec_accept_src || 'manual';
  if ($('w-loop_kv')) $('w-loop_kv').value = (S.sc.workload && S.sc.workload.loop_kv) || 'per_loop';
  treeWarn();
  if ($('pd-length_mix') && $('pd-length_mix')._sync) $('pd-length_mix')._sync();
  if ($('pd-prefix_tree') && $('pd-prefix_tree')._sync) $('pd-prefix_tree')._sync();
  if ($('w-pipeline')) $('w-pipeline').checked = S.sc.workload.pipeline !== false;
  if ($('w-placement')) $('w-placement').value = S.sc.workload.placement || 'auto';
  if ($('w-vae_tiling')) $('w-vae_tiling').checked = !!S.sc.workload.vae_tiling;
  for (const k of ['dit_fsdp', 'te_cpu', 'vae_parallel', 'overlap']) if ($('w-' + k)) $('w-' + k).checked = !!S.sc.workload[k];
  if ($('w-sample_split')) $('w-sample_split').checked = S.sc.workload.sample_split !== false;
  if ($('w-tri_attn')) $('w-tri_attn').value = S.sc.workload.tri_attn || 'fused';
  if ($('pd-chunk_profile')) {   // 0.231
    const cp = S.sc.pd.chunk_profile || 'vllm';
    $('pd-chunk_profile').value = cp;
    $('pd-chunk_budget').disabled = !(cp.startsWith('vllm') || cp.startsWith('sglang'));
    $('pd-chunk_tokens').disabled = cp !== 'fixed';
  }
  if ($('pd-prefix_host_policy')) $('pd-prefix_host_policy').value = S.sc.pd.prefix_host_policy || 'write_through';
  if ($('w-step_cache')) $('w-step_cache').value = S.sc.workload.step_cache || 'teacache';
  if ($('w-mode_cache')) $('w-mode_cache').value = S.sc.workload.mode_cache || 'off';   // 0.255 FLUX 3 mode-stream reuse
  if ($('w-cfg_branches')) $('w-cfg_branches').value = S.sc.workload.cfg_branches || 'ref';   // 0.260 CFG negative branch
  if ($('w-tri_chunk')) { $('w-tri_chunk').value = S.sc.workload.tri_chunk || ''; $('w-tri_chunk').disabled = (S.sc.workload.tri_attn || 'fused') !== 'chunked'; }
  if ($('c-slc_policy') && S.cat) $('c-slc_policy').value = chipVal('slc_policy') || 'pin';
  if ($('c-attn_fmt') && S.cat) $('c-attn_fmt').value = chipVal('attn_fmt') || '';
  if ($('c-dispatch') && S.cat) $('c-dispatch').value = chipVal('dispatch') || '';
  for (const k of ['noc_topology', 'noc_routing', 'noc_mem_attach', 'noc_eject']) {   // 0.306: NoC selects ('' = preset / default)
    if ($('c-' + k) && S.cat) $('c-' + k).value = k in S.chipOver ? S.chipOver[k] : '';
  }
  if ($('c-noc_mem_edges') && S.cat) $('c-noc_mem_edges').value = 'noc_mem_edges' in S.chipOver ? String(S.chipOver.noc_mem_edges) : '';   // 0.310
  if ($('c-clk_datapath_gate')) $('c-clk_datapath_gate').checked = !!chipVal('clk_datapath_gate');
  if ($('fab-enabled') && S.sc && S.cat) fabPaint();
  if ($('d2d-enabled') && S.sc) { $('d2d-enabled').checked = !!S.sc.d2d_enabled; if (S.cat) $('d2d-std').value = S.sc.d2d_std; d2dPaint(); }
  if ($('pd-enabled') && S.sc) { $('pd-enabled').checked = !!S.sc.pd.enabled; $('pd-inputs').hidden = !S.sc.pd.enabled; $('pd-kv_layerwise').checked = !!S.sc.pd.kv_layerwise;
    $('pd-prefix_on_decode').checked = S.sc.pd.prefix_on_decode !== false; $('pd-search_layouts').checked = !!S.sc.pd.search_layouts; $('pd-search_decode_batch').checked = !!S.sc.pd.search_decode_batch; $('pd-simulate').checked = !!S.sc.pd.simulate;
    $('pd-prefix_affinity').checked = !!S.sc.pd.prefix_affinity;
    $('pd-kv_policy').value = S.sc.pd.kv_policy || 'off';
    $('pd-kv_admit').value = S.sc.pd.kv_admit || 'before_prefill';
    $('pd-ttft_resid').value = S.sc.pd.ttft_resid || 'full';
    $('pd-prefix_writeback').checked = !!S.sc.pd.prefix_writeback;
    $('pd-prefix_turn_prompt').checked = !!S.sc.pd.prefix_turn_prompt;
    if (S.cat) {
      const sel = $('pd-prefill_chip'), pc = S.sc.pd.prefill_chip;
      if (!sel.options.length) put(sel, h('option', { value: '' }, '同 decode'), ...Object.keys(S.cat.chips).map((k) => h('option', { value: k }, k)));
      sel.value = typeof pc === 'string' ? pc : (pc && pc.name && S.cat.chips[pc.name] ? pc.name : '');
    }
    $('pd-prefill_mem_id').value = S.sc.pd.prefill_mem_id || ''; }
  paintPanel();
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
    m.domain === 'vlm' ? h('span', { class: 'badge vlm', title: coverTip(m) }, m.vision ? 'VLM · 含视觉编码器' : 'VLM · 视觉编码器未建模') : domainBadge(m),
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
  document.querySelectorAll('.protein-only').forEach((e) => (e.hidden = !isSeqEnc(m)));
  if ($('w-seq_len-name')) $('w-seq_len-name').textContent = m.domain === 'genomics' ? '窗口长度（token）' : '序列长度（残基）';
  document.querySelectorAll('.dp-in').forEach((e) => (e.hidden = !(m.is_moe || full)));
  document.querySelectorAll('.pair-only').forEach((e) => (e.hidden = !m.is_pair));
  const img = m.domain === 'gen' && !!(m.workload && m.workload.image);      // 0.119: image DiT — no frame axis
  document.querySelectorAll('.vgen-only').forEach((e) => (e.hidden = img));     // 0.256: own class (was 'video-only', which the VLM video toggle then hid)
  if ($('w-clip_slo_s-name')) $('w-clip_slo_s-name').textContent = img ? '单张 SLO s' : '单段 SLO s';
  const refs = m.domain === 'gen' && !!(m.workload && m.workload.max_refs);     // 0.252: FLUX 3 reference images
  document.querySelectorAll('.ref-only').forEach((e) => (e.hidden = !refs));
  if (!refs && S.sc && S.sc.workload) for (const k of ['ref_images', 'ref_width', 'ref_height', 'mode_cache']) delete S.sc.workload[k];
  const unc = m.domain === 'gen' && !!(m.workload && m.workload.uncond);       // 0.260: model-declared CFG negative branch
  document.querySelectorAll('.uncond-only').forEach((e) => (e.hidden = !unc));
  if (!unc && S.sc && S.sc.workload) delete S.sc.workload.cfg_branches;
  const loops = !!m.loop_default;                                              // 0.253: Dory looped LM
  document.querySelectorAll('.loop-only').forEach((e) => (e.hidden = !loops));
  if (!loops && S.sc && S.sc.workload) for (const k of ['loops', 'loop_kv']) delete S.sc.workload[k];
  document.querySelectorAll('.seq-slo').forEach((e) => (e.hidden = !!m.is_pair));
  document.querySelectorAll('.diff-only').forEach((e) => (e.hidden = !(m.is_pair && m.workload && m.workload.diff_steps)));
  document.querySelectorAll('.tp-in').forEach((e) => (e.hidden = !!m.is_pair));   // structure models: PP × DP × DAP, no TP
  document.querySelectorAll('.sp-in').forEach((e) => (e.hidden = !full));
  document.querySelectorAll('.cp-in').forEach((e) => (e.hidden = full));
  $('l-sp-name').textContent = m.is_pair ? 'DAP' : 'SP';
  $('l-sp-lab').title = m.is_pair
    ? 'DAP（动态轴并行，FastFold）：pair / MSA / 模板网格沿一个残基轴切到 DAP 张卡，单一 / 原子轨道每卡重复；三角乘法、偏置、外积均值 all-gather，轴切换 all-to-all'
    : 'Ulysses 序列并行：token 按 SP 切分，注意力前后各一次 all-to-all（每 rank 算 heads/SP 个头的完整序列）';
  if (full) {
    if (S.cmpObj === 'goodput') { S.cmpObj = 'decode'; paintSeg('cmp-obj', 'decode'); }
    if (S.layObj === 'goodput') { S.layObj = 'decode'; paintSeg('lay-obj', 'decode'); }
  }
  const unit = m.domain === 'gen' ? (m.workload && m.workload.image ? '张/s/卡' : '帧/s/卡') : m.domain === 'genomics' ? '窗口/s/卡' : m.domain === 'protein' ? '序列/s/卡' : null;
  for (const id of ['cmp-obj', 'lay-obj']) $(id).querySelector('[data-v=decode]').textContent = unit ? `目标：吞吐（${unit}）` : '目标：decode 吞吐';
  $('batch-label').textContent = m.domain === 'gen' ? (m.workload && m.workload.image ? 'batch（图像数）' : 'batch（视频段数）') : m.domain === 'genomics' ? 'batch（窗口数）' : m.domain === 'protein' ? 'batch（序列数）' : 'batch';
  $('best-batch-label').textContent = m.domain === 'gen' ? (m.workload && m.workload.image ? '自动取满足单张延迟 SLO 的最大 batch（最大化张/s/卡）' : '自动取满足单段延迟 SLO 的最大 batch（最大化帧/s/卡）')
    : isSeqEnc(m) ? `自动取满足批延迟 SLO 的最大 batch（最大化${m.domain === 'genomics' ? '窗口' : '序列'}/s/卡）` : '自动取满足 TPOT SLO 的最大 batch';
  const w = m.workload;
  if (w) {
    $('wl-kind').textContent = m.domain === 'gen' && w.image ? `图像 · 默认 ${w.width}×${w.height} · 单帧 · ${w.steps} 步 · CFG ${w.cfg}`
      : m.domain === 'gen' ? `视频 · 默认 ${w.width}×${w.height} · ${w.frames} 帧 · ${w.fps} fps · ${w.steps} 步 · CFG ${w.cfg}`
      : w.structure ? `蛋白质结构 · 默认 ${w.seq_len} 残基 · ${w.msa ? `MSA ${w.msa} 行` : w.xmsa ? `MSA ${w.xmsa} 行` : '单序列'} · 主干 ${w.recycles} 遍${w.diff_steps ? ` · 扩散 ${w.diff_steps} 步 × ${w.samples} 样本` : ''}`
      : w.kind === 'genomics' ? `基因组 · 默认窗口 ${w.seq_len} token（${w.kmer}-mer，${num(w.seq_len * w.kmer)} bp）· 重叠 ${w.overlap} token`
      : `蛋白质 · 默认 ${w.seq_len} 残基（训练上限 ${w.max_seq}）`;
    for (const [k, v] of Object.entries({ frames: w.frames, height: w.height, width: w.width, steps: w.steps, cfg: w.cfg, seq_len: w.seq_len,
      msa: w.msa || w.xmsa, recycles: w.recycles, psteps: w.diff_steps, samples: w.samples }))
      if ($('w-' + k)) $('w-' + k).placeholder = v ? `默认 ${v}` : '';
    if ($('w-msa')) $('w-msa').disabled = !!w.structure && !w.msa && !w.xmsa;
    const fr = m.domain !== 'gen' ? '' : w.vae_frames === 'chunk17' ? `帧每 17 帧一块 → ${(16 / w.vae[0] | 0) + 1} 潜帧`
      : w.vae_frames === 'h3' ? '帧补齐到 17n+5 → 5n+2 潜帧'
      : w.vae_frames === 'wan' ? `帧取 ⌊F/${w.vae[0]}⌋·${w.vae[0]}+1 → F/${w.vae[0]}+1 潜帧` : `帧 (F−1)/${w.vae[0]}+1`;
    $('wl-note').textContent = '留空 = 按发布默认（' + w.source + '）。' + (m.domain === 'gen'
      ? `token 数 = 潜空间网格（${fr}，像素 /${w.vae[1]}，patch ${w.patch.join('×')}）${w.joint_text ? ` + ${w.text_tokens} 个文本 token（联合注意力）` : `；文本 ${w.text_tokens} token 走跨注意力`}`
        + (w.audio_per_s ? ` + 音频 ${w.audio_per_s}/s × ${w.audio_channels} 声道 token（同一序列）` : '')
        + (w.attention === 'factorized' ? '；注意力按发布结构分解：空间块在潜帧内、时间块沿时间轴'
          : w.attention === 'per_mode' ? '；分模态块在各模态（文本 / 图像 / 参考图）内自注意力，单流块联合注意力' : '；全 3D 注意力')
        + (w.max_refs ? `；参考图 ≤ ${w.max_refs} 张（每张 VAE 编码 → (宽/16)(高/16) token，每步重算）` : '')
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
    ...(m.vision_params_B && !m.vision ? [h('li', {}, `视觉编码器（${num(m.vision_params_B)}B 参数）未建模：不计其权重存储与图像 prefill，只评估语言主干`)] : []));
  $('model-approx-wrap').hidden = !(m.coverage_reasons || []).length && !(m.vision_params_B && !m.vision);
  put($('model-notes'), ...m.notes.map((n) => h('li', {}, n)));
  $('model-notes-wrap').hidden = !m.notes.length;
  document.querySelectorAll('.moe-only').forEach((e) => (e.hidden = !m.is_moe));
  document.querySelectorAll('.vlm-only').forEach((e) => (e.hidden = !m.vision));
  if (!m.vision && S.sc.serving.images) { S.sc.serving.images = 0; $('s-images')._sync(); }
  const vid = !!(m.vision && m.vision.video);   // 0.166: only models with a modelled video processor
  document.querySelectorAll('.video-only').forEach((e) => (e.hidden = !vid));
  if (!vid && S.sc.serving.videos) { delete S.sc.serving.videos; $('s-videos')._sync(); }   // images only mean something on a VLM
  const L = S.sc.layout;
  if (isFull(m)) { L.ep = 1; L.etp = 1; L.sp = L.sp || 1; delete L.cp; if (m.is_pair) L.tp = 1; }
  else { L.sp = 1; if (!m.is_moe) { L.dp = 1; L.ep = 1; L.etp = 1; } else fixMoe('tp'); }
  if (L.pp > m.n_layers) L.pp = 1;
  syncInputs();
  cardsNote();
}
function fixMoe(changed) {
  const L = S.sc.layout;
  if (!model().is_moe) return;
  const want = L.tp * L.dp * (L.cp || 1);
  if (changed === 'etp') { if (want % L.etp === 0) { L.ep = want / L.etp; return; } }
  if (want % L.ep === 0) L.etp = want / L.ep; else { L.ep = want; L.etp = 1; }
}
const cards = () => S.sc.layout.pp * S.sc.layout.tp * S.sc.layout.dp * (S.sc.layout.sp || 1) * (S.sc.layout.cp || 1);
function setCards(n) {
  // default fill for a new card count: all TP (MoE: experts spread with EP = TP); 「布局搜索」finds the best
  const L = S.sc.layout;
  L.pp = 1; L.tp = n; L.dp = 1; L.sp = 1; delete L.cp;
  if (model().is_pair) { L.tp = 1; L.sp = n; }   // structure models: no TP — fill with DAP (pair grid split)
  if (model().is_moe) { L.ep = n; L.etp = 1; } else { L.ep = 1; L.etp = 1; }
  if (n > 64 && !model().is_pair) {   // 0.60: large replicas — TP inside a node, the rest DP (MoE) / PP (dense)
    const divs = [...Array(n).keys()].map((i) => i + 1).filter((d) => n % d === 0);
    const tpMax = Math.max(1, S.sc.node_cards || 8);
    let tp = Math.max(...divs.filter((d) => d <= tpMax));
    if (model().is_moe) { L.tp = tp; L.dp = n / tp; }
    else if (!isFull(model())) {
      const nl = model().n_layers || 64;
      while (n / tp > nl) tp = divs.find((d) => d > tp) || n;
      L.tp = tp; L.pp = n / tp;
    }
  }
}
function cardsNote() {
  const L = S.sc.layout;
  const m = model();
  $('cards-note').textContent = m.is_pair ? `PP${L.pp} × DP${L.dp} × DAP${L.sp || 1} = ${cards()} 卡；结构模型无 TP（改卡数按 DAP 填充）`
    : `PP${L.pp} × TP${L.tp} × DP${L.dp}${isFull(m) ? ` × SP${L.sp || 1}` : ''}${(L.cp || 1) > 1 ? ` × CP${L.cp}` : ''} = ${cards()} 卡；改卡数会重新填充布局（≤ 64 卡全 TP；更大时 TP ≤ 每节点卡数，其余 MoE 用 DP、dense 用 PP）`;
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
    h('span', {}, 'SRAM 端口 ', h('b', {}, num(c.port_GBps / c.freq_ghz) + ' B/cycle'), ' (默认)'),
    h('span', {}, '累加器 ', h('b', {}, num(c.acc_total_kib) + ' KiB'), ` (默认 ${c.acc_kib_per_engine_default} KiB/核)`));
  $('c-sram_port_Bpc').placeholder = `默认 ${num(c.port_GBps / c.freq_ghz)}`;
  $('c-gemv_macs').placeholder = `默认 ${c.gemv}`;
  $('c-acc_kib').placeholder = `默认 ${num(c.acc_total_kib)}`;
  $('c-acc_kib').title = `留空 = 默认 ${c.acc_kib_per_engine_default} KiB/核 × ${c.engines} 核 = ${num(c.acc_total_kib)} KiB（填入值为整片总量）`;
  const mx = [...c.mixed, ...S.mixed.flatMap((k) => S.cat.mixed_presets[k].pairs)];
  $('chip-formats').textContent = '原生格式 × 速率：' + c.formats.map(([f, r]) => `${f}×${r}`).join(' · ') +
    (mx.length ? '；混合精度 MAC（W×A）：' + mx.map(([w, a, r]) => `${w}×${a}@${r}`).join(' · ') : '') +
    '；其余 W/A 组合：权重反量化到激活格式（或双方上转 bf16），在向量单元计入开销。';
  paintMixed();
}
/* 0.147: mixed-precision MAC toggles (design variable; default chip has none) */
function paintMixed() {
  const box = $('chip-mixed'); if (!box || !S.cat.mixed_presets) return;
  put(box, '混合精度 MAC：', ...Object.entries(S.cat.mixed_presets).map(([k, p]) => {
    const cb = h('input', { type: 'checkbox', id: 'mx-' + k });
    cb.checked = S.mixed.includes(k);
    cb.addEventListener('change', () => { S.mixed = cb.checked ? [...S.mixed, k] : S.mixed.filter((x) => x !== k); onPreset(); schedule(); });
    return h('label', { title: p.pairs.map(([w, a, r]) => `${w}×${a} @${r}`).join('，') + ' — ' + p.source }, cb, ' ' + k.toUpperCase() + ' ');
  }));
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
  else opts($('m-cap'), (f.caps[String(S.mem.width)] || []).map((c) => [c.GB, c.GB + ' GiB' + tagSuffix(c)]), S.mem.cap);
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
function schedule() { clearTimeout(timer); timer = setTimeout(runEval, 120); invalidate(); cancel('fit'); cancel('sens'); swHint(); paintPanel(); }
// 0.90: fabric algo = auto picks each collective's algorithm alone (bw + α, like a runtime tuner) — with D2D chiplets a
// faster link can switch to a higher-α algorithm whose bandwidth hides under compute (≤ 0.4 % non-monotone, 0.89
// audit, MODEL §35); auto_overlap minimises the stage time and is monotone
const BW_PATH = /^(link\.|d2d\.|net\.|fabric\.|hop_)|GBps|alpha_us/;
function swHint() {
  const el = $('sw-hint');
  if (!el || !S.sc || !S.cat) return;
  const f = S.sc.fabric;
  const on = !!(f && f.enabled && f.algo !== 'auto_overlap' && S.sc.d2d_enabled && (S.sc.package_cards || 1) > 1);
  el.hidden = !on;
  if (!on) return;
  const bw = BW_PATH.test($('sw-path').value || '');
  put(el, h('b', {}, bw ? '带宽敏感性扫描：建议把集合通信算法改为 auto_overlap。' : '提示：fabric + D2D 下做带宽敏感性分析时建议用 auto_overlap。'),
    (f.algo === 'auto' ? ' 当前 fabric 算法为 auto：每次集合通信单独取「带宽 + α」最小的算法（类似运行时调优器）。'
      : ` 当前 fabric 算法为 ${f.algo}：算法固定，但协议 / 变体仍按每次集合通信单独取「带宽 + α」最小。`) + '开启 D2D 芯粒时，链路更快可能改选 α 更大、带宽更省的算法，而这部分带宽本来就被计算掩盖，于是结果对带宽不单调（实测 ≤ 0.4 %，约 1/5 的场景）。auto_overlap 按流水级的暴露时间选算法，对带宽单调。 ',
    h('button', { class: 'btn ghost', id: 'sw-hint-fix', onclick: () => { fab().algo = 'auto_overlap'; $('fab-algo').value = 'auto_overlap'; schedule(); } }, '切换为 auto_overlap'));
}
// 0.61.2: every scenario change bumps S.ver; a search / sweep that lands after an edit is labelled stale and is
// rendered with the model / phase it was computed for (it used to read the current ones: an LLM sweep that finished
// after switching to a video model showed 帧/s and a 0 s latency column)
const STALE = '场景已改变，需重新运行';
function staleTag(v0) { return S.ver !== v0 ? ` · ${STALE}` : ''; }
function invalidate() {
  S.ver = (S.ver || 0) + 1;
  if (S.cat) paintScope();
  for (const id of ['cmp-status', 'lay-status', 'sw-status'])
    if ($(id).dataset.done) $(id).textContent = STALE;
  // 0.285: speech / streams / dispatch insight panels keep their text and gain the stale tag once
  // 0.288: + the design-insight panels (insight / SLC size / sizing / upgrade path) — they kept old results untagged
  for (const id of ['ins-sp-status', 'ins-sps-status', 'ins-dp-status', 'ins-status', 'ins-slc-status', 'ins-size-status',
    'ins-path-status', 'ins-noc-status', 'ins-dse-status']) {
    const el = $(id);
    if (el && el.dataset.done && !el.textContent.includes(STALE)) el.textContent += ` · ${STALE}`;
  }
  // 0.301: a scenario edit drops an in-flight design search (the long slow-lane job; static build frees its worker)
  const ds = $('ins-dse-status');
  if (ds && !ds.dataset.done && ds.textContent.startsWith('计算中')) {
    cancel('insight_dse');
    ds.textContent = `已丢弃进行中的搜索 · ${STALE}`;
    ds.dataset.done = '1';
  }
  // 0.78: the stability verdict was only tagged at render time — later edits left it looking current
  const sb = $('stab-box');
  if (sb && sb.dataset.done && !sb.querySelector('.stale')) {
    delete sb.dataset.done;
    sb.prepend(h('div', { class: 'stale warn' }, STALE));
  }
}
function showErr(msg) {
  put($('warns'), h('div', { class: 'err' }, msg));
  status('错误', 'err');
}
async function runEval() {
  const t0 = performance.now();
  try {
    const r = await track(req('eval', '/api/eval', body({ best_batch: S.best, sensitivity: false, ...(S.powerCap ? { power_cap: true } : {}) })));
    if (!r) return;
    S.last = r;
    markEvalStale(false);
    renderEval(r);
    status(`就绪 · ${Math.round(performance.now() - t0)} ms`);
    runSensitivity(r);
  } catch (e) { showErr(e.message); markEvalStale(!!S.last); }
}
// 0.130: a rejected eval (400) left the previous result on screen looking current — dim it and say so
function markEvalStale(on) {
  for (const id of ['kpis', 'tab-eval']) $(id).classList.toggle('stale-res', on);
  if (on) $('warns').append(h('div', { class: 'warn' }, '下方结果来自上一次有效输入，与当前参数不符'));
}
// 0.97: the 敏感度 panel is filled after the main result (≈ 4 extra evaluations; a DiT point costs ≈ 5× the eval)
async function runSensitivity(r) {
  const b = body();
  if (S.best) b.scenario.serving = { ...b.scenario.serving, batch: r.summary.batch };
  // 0.227: the what-if checks the cap only when the user turned the cap on and gave a TDP (no default value)
  const capW = S.powerCap ? ((S.energy || {}).tdp_W ?? (S.budget || {}).tdp_W_card) : null;
  if (capW) b.power_cap_W = capW;
  $('sens-note').textContent = '敏感度计算中…';
  try {
    const x = await req('sens', '/api/sensitivity', b);
    if (!x || S.last !== r) return;
    r.sensitivity = x.sensitivity;
    renderSensitivity(r);
  } catch (e) { $('sens-note').textContent = '敏感度：' + e.message; }
}
function kpi(label, value, sub, cls = '') {
  return h('div', { class: 'kpi ' + cls }, h('div', { class: 'l' }, label), h('div', { class: 'v' }, value), h('div', { class: 's' }, sub));
}
function boundTag(b, dq) {
  // 0.290: dq = the VECTOR-bound stage's vector time is mostly weight dequant (api.dequant_bound)
  if (!b) return '—';
  const t = h('span', { class: 'bound ' + b, title: BOUND_ZH[b] + (dq ? ' · 主要是权重反量化' : '') }, b);
  return dq ? h('span', {}, t, h('span', { class: 'small', title: '向量时间过半是权重反量化（码本 / 打包格式在向量单元解码）' }, ' 反量化')) : t;
}
const DQ_ZH = ' · 向量时间过半为权重反量化';
function ttftFlag(ok, ms) {
  if (ok === null || ok === undefined) return h('span', { class: 'flag pend' }, '—');
  return h('span', { class: 'flag ' + (ok ? 'ok' : 'no'), title: 'DP 组内 prefill 的 TTFT 与 SLO 比较' },
    (ok ? '✓ ' : '✗ ') + (ms === undefined ? '' : ms === null ? '不可行' : num(ms) + ' ms'));
}
function renderEval(r) {
  const s = r.summary, sv = r.scenario.serving, g = r.goodput;
  const hv = (r.stages || [])[s.heaviest_stage];
  S.dq = !!(hv && hv.dequant_bound);
  const m = model();
  put($('model-badges'), ...badges(m, r.model.what_if));
  put($('warns'), ...s.warnings.filter((w) => !w.startsWith('容量不足')).map((w) => h('div', {}, w)),
    ...(r.model.what_if ? [h('div', {}, 'what-if：dtype 已偏离官方发布，结果仅供推演。')] : []));
  renderStages(r);
  renderWalls(r);
  renderEnergy(r);
  renderFabric(r);
  renderBudget(r);
  renderSensitivity(r);
  renderPD(r);
  { const sc = r.scenario, P = sc.d2d_enabled ? (sc.package_cards || 1) : 1, N = sc.node_cards || 0,
      // 0.78: shares of the bytes actually sent (per algorithm and tier: what the energy table charges)
      sum = (k) => r.stages.reduce((x, st) => x + (st.link_GB && st.link_GB.wire ? st.link_GB.wire[k] || 0 : 0), 0),
      tot = sum('d2d') + sum('link') + sum('net');
    const parts = [];
    if (P > 1) parts.push(`D2D ${tot > 0 ? pct(sum('d2d') / tot) : '—'}`);
    if (N > 0) parts.push(`跨节点 ${tot > 0 ? pct(sum('net') / tot) : '—'}`);
    $('tier-note').textContent = parts.length ? `上线字节分层（按所选集合通信算法的实际发送量；每封装 ${P} die${N ? `，每节点 ${N} 卡` : ''}）：${parts.join('，')}，其余走节点内 scale-up` : ''; }
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
    // 0.226: tree drafting — headline TPOT uses the chain lower bound by default; the band and the upper bound are secondary
    const sp = r.spec;
    if (sp && sp.tau_indep !== undefined)
      k.push(h('div', { class: 'kpi', id: 'kpi-tree-band' }, h('div', { class: 'l' }, `树 TPOT 区间（主值 ${sp.accept_model}）`),
        h('div', { class: 'v' }, `${num(sp.tpot_ms_chain)} … ${num(sp.tpot_ms_indep)} ms`),
        h('div', { class: 's' }, `τ chain ${num(sp.tau_chain)} · rank ${num(sp.tau_rank)} · indep ${num(sp.tau_indep)}；上界 TPOT ${num(sp.tpot_ms_indep)} ms 仅供参考（EAGLE-1 对照：indep ${pct(sp.check.indep[0])}…${pct(sp.check.indep[1])}，rank ${pct(sp.check.rank[0])}…${pct(sp.check.rank[1])}，chain ${pct(sp.check.chain[0])}…${pct(sp.check.chain[1])}）`)));
  } else {
    const over = s.ttft_ms > sv.ttft_slo_ms;
    const vz = s.vision;
    k.push(kpi('TTFT', num(s.ttft_ms) + ' ms', `batch ${s.batch} · prompt ${vz ? `${vz.text_prompt} 文本 + ${vz.image_tokens} ${vz.video ? '视觉（图像 + 视频）' : '图像'} token` : sv.prompt} · SLO ${sv.ttft_slo_ms} ms`
      + (vz && vz.cards ? ` · 含视觉编码 ${num(vz.s * 1e3)} ms` : ''), over ? 'warn' : ''));
    k.push(kpi('prefill 吞吐 / 卡', num(s.tok_s_card) + ' tok/s', `共 ${num(s.tok_s)} tok/s · ${s.cards} 卡 · ${s.layout}`));
  }
  k.push(h('div', { class: 'kpi' }, h('div', { class: 'l' }, '绑定瓶颈 · 有效 MAC'),
    h('div', { class: 'v' }, boundTag(s.bound, S.dq), ' ', pct(s.array_util)),
    h('div', { class: 's' }, `${BOUND_ZH[s.bound] || ''}${S.dq ? DQ_ZH : ''} · 最重流水级 ${s.heaviest_stage}`)));
  if (g) k.push(h('div', { class: 'kpi ' + (g.ttft_ok ? '' : 'warn') }, h('div', { class: 'l' }, 'goodput / 卡（含 prefill）'),
    h('div', { class: 'v' }, num(g.tok_s_card) + ' tok/s'),
    h('div', { class: 's' }, 'DP prefill TTFT ', ttftFlag(g.ttft_ok, g.ttft_ms), g.prefill_batch ? ` · prefill batch ${g.prefill_batch}` : '')));
  else k.push(kpi('映射组织', (S.cat.mappings.find((x) => x.id === s.mapping) || {}).label || s.mapping, MAP_DESC[s.mapping] || ''));
  const frac = isFinite(cap) && cap > 0 ? Math.min(1, s.dram_need_GiB / cap) : 0;
  k.push(h('div', { class: 'kpi' }, h('div', { class: 'l' }, 'DRAM 需求 / 容量（每卡）'),
    h('div', { class: 'v' }, `${num(s.dram_need_GiB)} / ${num(cap)} GiB`),
    h('div', { class: 's' }, h('span', { class: 'ubar wide' }, h('i', { style: `width:${frac * 100}%` })), `SRAM 驻留 ${pct(s.residency)}`)));
  if (s.vision) {
    const v = s.vision;
    if (v.images || !v.video) k.push(kpi('视觉编码器', v.cards ? num(v.s * 1e3) + ' ms' : '—',
      `${v.label} · ${v.images} 张/请求 · ${v.px[0]}×${v.px[1]} → ${v.resized_px[0]}×${v.resized_px[1]} · ${v.tokens_per_image} token/张`
      + (v.cards ? ` · ${num(v.tflop)} TFLOP · ${v.cards} 卡各 ${v.per_card} 张（数据并行「假设」）· ${BOUND_ZH[v.bound] || v.bound}` : ' · decode 不跑编码器，图像 token 在 KV 中')
      + ` · 权重 ${num(v.weights_GB)} GB 在首级`));
    const vv = v.video;
    if (vv) k.push(kpi('视频编码', vv.s ? num(vv.s * 1e3) + ' ms' : '—',
      `${vv.videos} 段/请求 · ${vv.frames} 帧 → ${vv.slices} 时间片 · ${vv.px[0]}×${vv.px[1]} → ${vv.resized_px[0]}×${vv.resized_px[1]} · ${vv.tokens_per_video} token/段`
      + (vv.cap ? ' · 逐帧上限' : '') + (vv.s ? ` · ${num(vv.tflop)} TFLOP · 每卡 ${vv.slices_per_card} 片` : ' · decode 不跑编码器，视频 token 在 KV 中')
      + (v.images ? '' : ` · ${v.label} 权重 ${num(v.weights_GB)} GB 在首级`)));
  }
  if (s.tts) {         // 0.269: TTS — one decode step = one latent = s_per_step of audio
    const t = s.tts, g = t.gflop_per_step || {};
    k.push(kpi('语音合成 TTS', t.rtf != null ? `RTF ${num(t.rtf)}` : '—',
      `每步 ${num(t.s_per_step * 1e3)} ms 音频 · 流匹配 ${t.flow_steps} 步 × CFG ${t.cfg}`
      + (t.audio_s_per_s != null ? ` · ${num(t.audio_s_per_s)} 音频秒/秒 · 实时流 ${t.streams_realtime}` : '')
      + ` · 每步 GFLOP：头 ${num(g.total)}（流 ${num(g.flow)} · VAE ${num(g.vae)}）vs LLM ≈ ${num(g.llm_approx)}`
      + ` · 头权重每步读 ${num(t.head_w_GB_per_step)} GB（存 ${num(t.head_w_GB_stored)} GB）`, t.rtf != null && t.rtf > 1 ? 'warn' : ''));
  }
  if (s.asr) {         // 0.270: speech input — cost per audio second (this phase)
    const a = s.asr;
    k.push(kpi('语音识别 ASR（本阶段）', a.audio_s_per_s_card != null ? `${num(a.audio_s_per_s_card)} 音频秒/秒/卡` : '—',
      `${num(a.audio_s)} s 音频 → 编码器 ${a.enc_positions} 位置` + (a.windows ? `（${a.windows} 个 30 s 填充窗）` : '')
      + (a.cross ? ` · 交叉注意力（解码器 prompt 不含音频 token）· 交叉 K/V ${num(a.cross_kv_MB_per_seq)} MB/序列` : ` · prompt 音频 token ${a.prompt_audio_tokens}`)
      + (a.truncated_s ? ` · 超过 ${num(a.max_audio_s)} s 的 ${num(a.truncated_s)} s 被截断（处理器 / 位置表上限）` : '')   // 0.273: per-model cap
      + (a.first_token_s != null ? ` · 首 token ${num(a.first_token_s * 1e3)} ms（编码器 ${num(a.encoder_s * 1e3)} ms）` : '')
      + (a.rtf_decode != null ? ` · 解码 RTF ${num(a.rtf_decode)}` : '')));
  }
  if (r.speech && !r.speech.error) {   // 0.270: end to end over both phases (core/speech.py)
    const p = r.speech;
    if (p.kind === 'asr') k.push(kpi('语音端到端 · ASR', `RTF ${num(p.rtf)}`,
      `${num(p.audio_s)} s 音频 → ${p.out_len} token · 首 token ${num(p.first_token_s * 1e3)} ms（编码器 ${num(p.encoder_s * 1e3)} ms · prompt 音频 token ${p.prompt_audio_tokens}）· 共 ${num(p.total_s)} s · ${num(p.audio_s_per_s_card)} 音频秒/秒/卡（静态 batch，两阶段串行「近似」）`,
      p.rtf > 1 ? 'warn' : ''));
    else if (p.kind === 'cosyvoice') k.push(kpi(`语音端到端 · ${p.label || 'CosyVoice' + p.version}`, `首音频 ${num(p.first_audio_s * 1e3)} ms`,   // 0.271; 0.273 label
      `${num(p.audio_out_s)} s 输出音频 = ${p.speech_tokens} 语音 token（LM RTF ${num(p.lm_rtf)}）· 分块 ${p.hops.length > 6 ? `${p.hops[0]} token × ${p.hops.length}` : p.hops.join('/') + ' token'}，`
      + (p.stream === 'cache' ? '每块只算新 token（编码器 / 估计器 K/V 缓存）' : '每块 flow 从头重算') + `（Euler ${p.flow_steps} 步 × CFG ${p.cfg}）`
      + ` · 语音卡 RTF ${num(p.rtf)}（${p.streams_per_card} 路）· 实时路数 ${p.streams_realtime} · 常驻实例首音频 ${num(p.first_audio_warm_s * 1e3)} ms（${p.stream === 'cache' ? '编码器缓存不裁剪' : 'hop 不复位'}）· ${num(p.audio_out_s_per_s_card)} 输出音频秒/秒/卡`
      + (p.audio_in ? ` · 输入 ${num(p.audio_in.audio_s)} s 音频 = ${p.audio_in.prompt_audio_tokens} token（编码器 ${num(p.audio_in.encoder_s * 1e3)} ms）` : ''),
      p.rtf > 1 || p.lm_rtf > 1 ? 'warn' : ''));
    if (p.kind === 'cosyvoice' && p.split_first) {   // 0.273: first-audio split + per-chunk flow / HiFT
      const f = p.split_first, ms = (a) => a.map((x) => num(x * 1e3)).join(' / ');
      k.push(kpi(`${p.label || 'CosyVoice'} 首音频拆分`, `${num(p.first_audio_s * 1e3)} ms`,
        `LM 首 token ${num(f.lm_ttft_s * 1e3)} ms + ${p.hops[0]}+3 token ${num(f.lm_tokens_s * 1e3)} ms + flow ${num(f.flow_chunk0_s * 1e3)} ms + HiFT ${num(f.hift_chunk0_s * 1e3)} ms`
        + ` · 每块 flow ${ms(p.flow_s)} ms · HiFT ${ms(p.hift_s)} ms · 估计器权重 SRAM 未命中 ${num(p.flow_pin_miss * 100)}%`));
    }
    if (p.kind === 's2st') {     // 0.279 Index-Echo: speech → translated speech, serial, non-streaming
      const f = p.split || {}, ms = (x) => num((x || 0) * 1e3);
      k.push(kpi(`语音翻译端到端 · ${p.label}`, `RTF ${num(p.rtf)}`,
        `${num(p.audio_s)} s 源音频 → 原文 ${p.src_tokens} + 译文 ${p.tgt_tokens} token → ${p.speech_tokens} 语音 token → ${num(p.audio_out_s)} s 输出（非流式：首音频 = 全程 ${ms(p.total_s)} ms）`
        + ` · 编码器 ${ms(f.encoder_s)} · ST 预填充 ${ms(f.st_prefill_s)} + 解码 ${ms(f.st_decode_s)} + 隐状态前向 ${ms(f.st_hidden_pass_s)} · 映射器 ${ms(f.mapper_s)}`
        + ` · CV3 预填充 ${ms(f.cv3_prefill_s)} + 解码 ${ms(f.cv3_decode_s)} · flow ${ms(f.flow_s)} · HiFT ${ms(f.hift_s)} ms · 单卡 ${num(p.card_GB)} GB`
        + (f.speech_tokenizer_s !== undefined ? ` · 语音分词器 ${ms(f.speech_tokenizer_s)} ms · 重载固定权重 ${ms(f.repin_s)} ms` : '')   // 0.280
        + (p.frontend && p.frontend.campplus_gflop_host ? ` · CampPlus 主机 CPU ${num(p.frontend.campplus_gflop_host)} GFLOP（不计卡时间）` : '')
        + (p.clips > 1 ? ` · 并发 ${p.clips} 段` : '')
        + (p.fits ? '' : ' · 放不下'), p.rtf > 1 || !p.fits ? 'warn' : ''));
    }
    if (p.kind === 'omni') k.push(kpi('语音端到端 · Omni 语音输出', `首音频 ${num(p.first_audio_s * 1e3)} ms`,   // 0.271
      `${num(p.audio_out_s)} s 输出音频 = ${p.codec_tokens} codec token（Talker 每步 ${num(p.talker_step_s * 1e3)} ms）· DiT 每 ${num(p.chunk_s)} s 分块 ${num(p.dit_chunk_s * 1e3)} ms（RK4 ${p.dit_evals} 次 × CFG ${p.cfg}）`
      + ` · 语音卡 RTF ${num(p.rtf)}（${p.streams_per_card} 路）· 实时路数 ${p.streams_realtime} · 共 ${num(p.total_s)} s · ${num(p.audio_out_s_per_s_card)} 输出音频秒/秒/卡（含 ${p.speech_cards} 张语音卡「假设」）`
      + (p.speech_card_fits ? '' : ' · 语音卡放不下'), p.rtf > 1 || !p.speech_card_fits ? 'warn' : ''));
    if (p.kind === 'omni') {     // 0.273: first-audio split, speech-card SLC block (0.272), audio input
      const f = p.split_first || {};
      k.push(kpi('Omni 首音频拆分', `${num(p.first_audio_s * 1e3)} ms`,
        `Thinker 首 token ${num(f.thinker_ttft_s * 1e3)} + 1 步 ${num(f.thinker_tpot_s * 1e3)} · Talker 预填充 ${num(f.talker_prefill_s * 1e3)} + 48 步 ${num(f.talker_steps_s * 1e3)}`
        + ` · DiT 块 0 ${num(f.dit_chunk0_s * 1e3)} · 声码器 ${num(f.voc_chunk0_s * 1e3)} ms · 首块 ${num(p.first_chunk_audio_s)} s 音频`
        + (p.streams_per_card > 1 ? ` · 最后一路 ${num(p.first_audio_last_s * 1e3)} ms（token2wav batch 1）` : '')));
      const c = p.slc;
      if (c) k.push(kpi('Omni 语音卡 · SLC 钉住', c.mib ? `SLC ${num(c.mib)} MiB` : 'SLC 无',
        `每音频秒：Talker ${num(c.talker_s_per_audio_s)} s（权重 ${num(c.talker_weight_GB)} GB/步，命中 ${num(c.talker_hit * 100)}%）`
        + ` · DiT ${num(c.dit_s_per_audio_s)} s（SRAM 未命中 ${num(p.dit_pin_miss * 100)}%，SLC 命中 ${num(c.dit_hit * 100)}%）`
        + ' · 钉住顺序：DiT 复读权重 → Talker（芯片 slc_mib / slc_policy = pin）', p.rtf > 1 ? 'warn' : ''));
      if (p.audio_in) k.push(kpi('Omni 音频输入', `${p.audio_in.prompt_audio_tokens} token`,
        `${num(p.audio_in.audio_s)} s 音频 · 编码器 ${num(p.audio_in.encoder_s * 1e3)} ms`));
    }
    if (p.kind === 'kimi') {     // 0.273 Kimi-Audio: parallel text / audio heads → chunked DiT detokenizer → BigVGAN
      const f = p.split_first || {}, ms = (a) => a.map((x) => num(x * 1e3)).join(' / ');
      k.push(kpi(`语音端到端 · ${p.label}`, `首音频 ${num(p.first_audio_s * 1e3)} ms`,
        `${num(p.audio_out_s)} s 输出音频 = ${p.audio_tokens} 音频 token（${p.token_rate} Hz，前 ${p.delay_steps} 步空，LM RTF ${num(p.lm_rtf)}）`
        + ` · 分块 ${p.chunks.join('/')} token（+${p.look_ahead_tokens} 前瞻），DiT Euler ${p.dit_evals} 次 + 1 次 K/V 提交`
        + ` · 语音卡 RTF ${num(p.rtf)}（${p.streams_per_card} 路）· 实时路数 ${p.streams_realtime}`
        + ` · 发布版 kimia.py（先生成完再去分词）首音频 ${num(p.first_audio_released_s * 1e3)} ms · ${num(p.audio_out_s_per_s_card)} 输出音频秒/秒/卡`
        + (p.speech_card_fits ? '' : ' · 语音卡放不下'), p.rtf > 1 || p.lm_rtf > 1 || !p.speech_card_fits ? 'warn' : ''));
      k.push(kpi(`${p.label} 首音频拆分`, `${num(p.first_audio_s * 1e3)} ms`,
        `LM 首 token ${num(f.lm_ttft_s * 1e3)} ms + ${p.delay_steps}+${p.chunks[0]}−1 步 ${num(f.lm_tokens_s * 1e3)} ms + DiT ${num(f.dit_chunk0_s * 1e3)} ms + BigVGAN ${num(f.voc_chunk0_s * 1e3)} ms`
        + ` · 首块 ${num(p.first_chunk_audio_s)} s 音频 · 每块 DiT ${ms(p.dit_s)} ms（K/V 帧 ${p.dit_kv_frames.join('/')}）· BigVGAN ${ms(p.voc_s)} ms`
        + ` · DiT 权重 SRAM 未命中 ${num(p.dit_pin_miss * 100)}%`));
      if (p.audio_in) k.push(kpi(`${p.label.split(' ')[0]} 音频输入`, `${p.audio_in.prompt_audio_tokens} token`,
        `${num(p.audio_in.audio_s)} s 音频 · Whisper 编码器 + 适配器 ${num(p.audio_in.encoder_s * 1e3)} ms`));
    }
    if (p.kind === 'moshi') {     // 0.274 Moshi: per 80 ms frame, serial on one card
      const f = p.split_frame || {}, b = p.bytes_per_frame || {};
      k.push(kpi(`语音端到端 · ${p.label}`, `帧 ${num(p.frame_s * 1e3)} ms / ${num(p.frame_ms)} ms`,
        `RTF ${num(p.rtf)}（${p.streams_per_card} 路）· Mimi 编码 ${num(f.mimi_encode_s * 1e3)} + 时间步 ${num(f.temporal_s * 1e3)}（KV ${p.ctx_frames} 帧）`
        + ` + 深度 ${p.depformer_steps} 步 × ${p.depformer_layer_passes / p.depformer_steps} 层 ${num(f.depformer_s * 1e3)} + Mimi 解码 ${num(f.mimi_decode_s * 1e3)} ms`
        + ` · 响应延迟 ${num(p.first_audio_s * 1e3)} ms（下限 ${num(p.latency_floor_s * 1e3)} ms）· 实时路数 ${p.streams_realtime}`
        + (p.layer_overhead_s ? ` · 层固定开销 ${num(p.layer_overhead_us)} µs × ${Object.values(p.layer_passes).reduce((a, x) => a + x, 0)} 次串行层（深度 + Mimi）= ${num(p.layer_overhead_s * 1e3)} ms` : ''),   // 0.275
        p.rtf > 1 || !p.fits ? 'warn' : ''));
      k.push(kpi('Moshi 每帧复读字节', `${num(p.bytes_per_frame_GB)} GB`,
        `时间 Transformer ${num(b.temporal_GB)} + KV ${num(b.kv_GB)} + 深度 Transformer ${num(b.depformer_GB)} + Mimi ${num(b.mimi_GB)} GB`
        + ` · 实时需有效 DRAM ${num(p.dram_GBps_for_realtime)} GB/s（当前 ${num(p.dram_GBps)}）· 每帧全部权重读一遍 → SLC 每钉住 1 GB 省 12.5 GB/s`));
    }
    if (p.kind === 'tts') k.push(kpi(`语音端到端 · ${p.label || 'TTS'}`, `RTF ${num(p.rtf)}`,
      `${num(p.audio_out_s)} s 音频输出 = ${p.steps} 步 · 首音频 ${num(p.first_audio_s * 1e3)} ms · 共 ${num(p.total_s)} s · ${num(p.audio_out_s_per_s_card)} 输出音频秒/秒/卡`
      + (p.split_first && p.split_first.prompt_s != null ? ` · 非流式：提示音频 ${num(p.prompt_audio_s)} s 编码 ${num(p.split_first.prompt_s * 1e3)} + 预填充 ${num(p.split_first.prefill_s * 1e3)}`   // 0.275 FireRedTTS3
        + ` + ${p.steps} 步 ${num(p.split_first.steps_s * 1e3)}（每步 ${num(p.tpot_s * 1e3)} ms：主干 + DiT ${p.flow_steps} 步 × CFG ${p.cfg} + patch 编码器）+ RedAE 解码 ${num(p.split_first.redae_decode_s * 1e3)} ms` : '')
      + (p.first_chunk_frames ? ` · 流式：每步 1 帧 ${num(p.frame_ms)} ms（主干 + 13 层码本塔，TPOT ${num(p.tpot_s * 1e3)} ms）· 首音频 = 预填充 ${num(p.split_first.prefill_s * 1e3)}`   // 0.276 Drama 3
        + `（文本 + 参考音频 ${p.ref_rows} 行）+ 首块 ${p.first_chunk_frames} 帧 ${num(p.split_first.first_chunk_steps_s * 1e3)} ms`
        + (p.dac ? ` + DAC 首窗 ${num(p.split_first.dac_window0_s * 1e3)} ms · DAC 与 LM 同卡分时（fp32 ${num(p.dac.GB)} GB，${num(p.dac.gflop_per_frame)} GFLOP/帧，有状态，${p.dac.batch} 路同批）`   // 0.277
          + `：RTF ${num(p.lm_rtf)}（LM）+ ${num(p.dac.load)}（DAC）· 无状态重解左上下文 ${num(p.dac.redecode_s_per_request * 1e3)} ms/请求（有状态 ${num(p.dac.s_per_request_batch1 * 1e3)}）` : ' · DAC 解码未计') : '')
      + (p.dit_pin ? ` · fp32 DiT ${num(p.dit_pin.dit_GB)} GB × ${p.dit_pin.reads_per_step} 读/步：SRAM ${num(p.dit_pin.sram_pinned_GB)} + SLC ${num(p.dit_pin.slc_GB)} GB，`   // 0.276
        + `DRAM ${num(p.dit_pin.dram_GB_per_step)} GB/步（全驻留需 SLC ≈ ${num(p.dit_pin.slc_full_GiB)} GiB）` : '')
      + (p.bf16_recast ? ` · 若按 GPU autocast 逐步重铸 bf16：+${num(p.bf16_recast.extra_GB_per_step)} GB/步 → 每步 ${num(p.bf16_recast.tpot_s * 1e3)} ms、RTF ${num(p.bf16_recast.rtf)}（未计入）` : ''),
      p.rtf > 1 ? 'warn' : ''));
  }
  if (r.balance) {     // 0.205: machine balance vs workload arithmetic intensity
    const b = r.balance;
    k.push(kpi('机器平衡 · machine balance', `${num(b.chip_flop_per_byte)} FLOP/B`,
      `负载 workload ${num(b.workload_flop_per_byte)} FLOP/B · 比值 ratio ${num(b.ratio)}× · 原始带宽 raw ${num(b.chip_flop_per_byte_raw)} FLOP/B`
      + (b.reachable !== null && b.reachable !== undefined
        ? ` · 临界 batch critical batch ${b.critical_batch ?? `∞（${ASYM_ZH[b.asymptote_by] || b.asymptote_by || 'KV'} 受限 · saturation-bound）`} · 上限 limit ${b.batch_limit}`
          + (b.limited_by ? `（${LIMIT_ZH[b.limited_by] || b.limited_by}）` : '')
          + (b.asymptote_by ? ` · 饱和来源 ${ASYM_ZH[b.asymptote_by] || b.asymptote_by}` : '')
          + ` · 饱和强度 asymptote ${num(b.ai_asymptote)} FLOP/B` : '')
      + (b.idle ? ` · ${b.idle === 'compute' ? '算力闲置 compute idle' : '带宽闲置 bandwidth idle'}` : '')
      + (b.hint ? ` · ${b.hint}` : ''), b.warning ? 'warn' : ''));
    if (b.warning) $('warns').append(h('div', { class: 'warn balance-warn' }, b.warning));   // 0.206: only when balance is unreachable within the limits
  }
  if (s.host) {
    const hh = s.host;
    k.push(kpi('主机开销', num(hh.exposed_s * 1e3) + ' ms 暴露', `${hh.overlap} · 每 ${hh.steps} 步 · ${num(hh.s * 1e3)} ms 主机 / ${num(hh.device_s * 1e3)} ms 设备`
      + (hh.host_bound ? ' · 主机受限' : ''), hh.host_bound ? 'warn' : ''));
  }
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
    h('div', { class: 'v' }, boundTag(s.bound, S.dq), ' ', pct(s.array_util)),
    h('div', { class: 's' }, `${BOUND_ZH[s.bound] || ''}${S.dq ? DQ_ZH : ''} · 最重流水级 ${s.heaviest_stage}`));
}
function domainKpis(s, cap, auto) {
  // video: clip latency, per-frame latency, frames/s/card; protein: batch latency, sequences/s/card, residues/s/card
  const g = s.gen, w = g.workload;
  const k = [];
  if (g.unit === 'frame') {
    const over = !s.slo_ok;
    const pl = g.pipeline;
    const im = !!w.image;           // 0.119: image DiT — per-image wording, no frame / fps axis
    k.push(kpi(im ? '单张延迟' : '单段延迟（clip）', fmtDur(g.clip_s),
      `${w.width}×${w.height}${im ? '' : ` · ${w.frames} 帧`} · ${w.steps} 步 × CFG ${w.cfg} · batch ${s.batch}${auto} · SLO ${fmtDur(g.slo_s)}${over ? ' · 超出' : ''}`
      + (pl ? ` · 文本编码 ${fmtDur(pl.te_s)}${pl.enc_s ? ` + 参考图编码 ${fmtDur(pl.enc_s)}` : ''} + 去噪 ${fmtDur(g.denoise_s)} + 解码 ${fmtDur(pl.decode_s)}${pl.load_s ? ` + 主机重载 ${fmtDur(pl.load_s)}` : ''} · 组件${pl.placement === 'auto' ? '（自动）' : ''}${pl.place_label}` : ' · 只计 DiT 去噪（文本编码器 / VAE 未计）'), over ? 'warn' : ''));
    const sc = g.step_cache;
    if (sc) k.push(kpi('跳步缓存', `${sc.skip} / ${sc.steps} 步跳过`,
      `${sc.mode} · 跳过的步 ${fmtDur(sc.t_skip_ms / 1e3)}（只跑嵌入 + 输出头 + 残差加）· 省去噪 ${fmtDur(sc.denoise_saved_s)} · 残差缓存 ${num(sc.buffer_GiB_card)} GiB/卡`));
    const mc = g.mode_cache;      // 0.255 FLUX 3 Base mode-stream reuse (SGLang StaticStreams)
    if (mc) k.push(kpi('模态流复用', `${fmtDur(mc.first_ms / 1e3)} → ${fmtDur(mc.reused_ms / 1e3)}`,
      `第 1 步全算，之后每步只跑图像分模态块 · 复用 ${num(mc.rows)} 行（文本 + 参考图）· ${num(mc.extract_tflop_seq)} → ${num(mc.cached_tflop_seq)} TFLOP/步/序列 · 缓存 ${num(mc.cache_GiB_card)} GiB/卡`));
    if (im) k.push(kpi('每去噪步', fmtDur(g.step_ms / 1e3), `${num(w.seq_tokens)} token / 前向 · ${w.steps} 步 × CFG ${w.cfg}`));
    else k.push(kpi('每帧延迟', fmtDur(g.s_per_frame),
      `每去噪步 ${fmtDur(g.step_ms / 1e3)} · ${num(w.seq_tokens)} token / 前向${w.audio_tokens ? `（含音频 ${num(w.audio_tokens)}）` : ''}${w.attention === 'factorized' ? ' · 分解注意力' : ''} · 视频 ${num(w.video_s)} s${g.realtime_x ? ` · 实时倍率 ${num(g.realtime_x)}×` : ''}`));
    k.push(kpi('吞吐 / 卡', im ? num(g.frames_per_s_card * 3600) + ' 张/小时' : num(g.frames_per_s_card) + ' 帧/s',
      `${im ? `${num(g.frames_per_s_card)} 张/s/卡` : `${num(g.clips_per_hour_card)} 段/小时/卡`} · ${s.cards} 卡 · ${s.layout} · ${num(g.tflop_per_request)} TFLOP/${im ? '张' : '段'}${pl ? `（其中文本编码 + 解码 ${num(pl.tflop / s.batch)}）` : ''}`));
  } else if (g.unit === 'window') {     // 0.106 genomics encoder: windows of a long record
    const over = !s.slo_ok;
    k.push(kpi('批延迟', fmtDur(g.batch_ms / 1e3),
      `batch ${s.batch}${auto} × ${w.seq_len} token（${num(w.bp_per_window)} bp）· SLO ${num(g.slo_ms)} ms${over ? ' · 超出' : ''}`, over ? 'warn' : ''));
    k.push(kpi('吞吐 / 卡', num(g.bp_per_s_card / 1e6) + ' Mbp/s',
      `${num(g.windows_per_s_card)} 窗口/s/卡（每窗口新增 ${num(w.new_bp_per_window)} bp，重叠 ${w.overlap_tokens} token）· ${s.cards} 卡 · ${s.layout}`));
    k.push(kpi('每窗口计算', num(g.tflop_per_request) + ' TFLOP', `${w.seq_len} token · 双向全注意力（S²）· 单次编码器前向`));
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
  const v0 = S.ver;
  box.hidden = false;
  put(box, h('div', { class: 'muted' }, '容量不足，正在计算可行的修正方案…'));
  try {
    const f = await track(req('fit', '/api/fit', body()));
    if (!f || S.ver !== v0) return;      // an edit during the search: its own eval decides (and refits if needed)
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
        `改为 ${c.cards} 卡 · ${c.layout} · batch ${c.batch}（每卡 ${num(c.need_GiB)} GiB${c.meets_slo ? '' : isFull(m) ? '，延迟 SLO 未满足' : '，TPOT SLO 未满足'}${c.pow2_cards && c.pow2_cards !== c.cards ? `；2 的幂卡数为 ${c.pow2_cards}` : ''}）`));
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
    if (!acts.length) lines.push(h('div', {}, `在 ${S.cat.max_cards || 8192} 卡以内、当前存储器类型的任何容量下都放不下；请换容量更大的存储器类型或更小的模型。`));
    else lines.push(h('div', { class: 'small muted' }, `一键修正（最少卡数的布局按${isFull(m) ? unitZh(m) + '/s/卡' : ' decode 吞吐'}取最优；存储器只在当前类型、速率内加大容量）：`));
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
const E_UNIT = { token: 'token', 'prompt token': 'prompt token', frame: '帧', seq: '序列', window: '窗口' };
function renderEnergy(r) {
  const e = r.energy;
  if (!e) { put($('energy-tbl')); return; }
  const c = e.counts_per_unit, j = e.J_by_action || {};
  const u = e.unit === 'frame' && model() && model().workload && model().workload.image ? '张' : (E_UNIT[e.unit] || e.unit);   // 0.130
  const sci = (x) => (x === 0 ? '0' : x.toExponential(3));
  const rows = [['MAC（bf16 等效）', c.mac, '', 'mac', 'pJ_mac'], ['向量操作', c.vec, '', 'vec', 'pJ_vec'],
    ['SRAM 端口' + (e.sram_access && e.sram_access.scale > 1 ? `（本地分片 ${num(e.sram_access.slice_mib)} MiB · 线缆 ${num(e.sram_access.wire_mm)} mm · ×${num(e.sram_access.scale)}）` : ''), c.sram, ' B', 'sram', 'pJ_bit_sram'],   // 0.298
    ...(c.slc ? [['SLC 命中' + (e.slc_access && e.slc_access.scale > 1 ? `（${e.slc_access.deprecated ? '√C 已弃用 · ' : ''}线缆 ×${num(e.slc_access.scale)} @ ${num(e.slc_access.slc_mib)} MiB` + (e.slc_access.wire_mm != null ? ` · ${num(e.slc_access.wire_mm)} mm` + (e.slc_access.over_reticle ? ' · 超光罩' : '') : '') + '）' : ''), c.slc, ' B', 'slc', 'pJ_bit_slc']] : []),
    [c.slc ? 'DRAM 读写（SLC 未命中）' : 'DRAM 读写', c.dram, ' B', 'dram', e.dram_pJ_field || 'pJ_bit_dram'],
    ...(e.phy_pJ_field ? [['DRAM PHY（芯片侧）', c.dram, ' B', 'dram_phy', e.phy_pJ_field]] : []),   // 0.188
    ...(c.noc ? [['NoC 路由（字节·跳）' + (e.noc ? `（${e.noc.topology} · ${num(e.noc.engines)} 引擎 · 间距 ${num(e.noc.pitch_mm)} mm）` : ''), c.noc, ' B·跳', 'noc', 'pJ_bit_hop_noc'],
      ['NoC 链路导线（字节·间距；需同时填 NoC 与导线）', c.noc_wire, ' B·间距', 'noc_wire', 'pJ_bit_hop_noc']] : []),   // 0.299
    ...(c.d2d ? [['D2D 发送（封装内）', c.d2d, ' B', 'd2d', 'pJ_bit_d2d']] : []),
    [c.d2d || c.net ? '节点内 scale-up 发送' : '链路发送', c.link, ' B', 'link', 'pJ_bit_link'],
    ...(c.net ? [['跨节点网络发送', c.net, ' B', 'net', 'pJ_bit_net']] : []),
    ...(c.host ? [['主机链路重载（卸载放置）', c.host, ' B', 'host', 'pJ_bit_host']] : []),
    ...(c.clock != null ? [['时钟（MAC 槽·周期，忙或停顿）' + (e.clock_W_card != null ? `（${num(e.clock_W_card)} W/卡` + (e.clock_gating && e.clock_gating.ungated_share < 1 ? `，门控：未门控 ${num(e.clock_gating.ungated_share)} · 利用率 ${num(e.clock_gating.util != null ? e.clock_gating.util : e.clock_gating.mac_util)}` + (e.clock_gating.datapath_gate ? `（数据通路 mean MAC/向量/SRAM口）` : '') : '') + '）' : ''), c.clock, '', 'clock', 'pJ_clk_mac_cycle']] : []),   // 0.228/0.265
    ...(e.noc_router ? [[`NoC 路由器时钟底（卡·秒；${num(e.noc_router.routers)} 个 · ${num(e.noc_router.link_Bpc)} B/周期 · ${num(e.noc_router.clk_W)} W/卡）`, c.idle_card_s, ' 卡·s', 'noc_clk', 'pJ_noc_router_cycle'],
      [`NoC 路由器泄漏（卡·秒；${num(e.noc_router.leak_W)} W/卡，静态）`, c.idle_card_s, ' 卡·s', 'noc_leak', 'mW_noc_router_leak']] : []),   // 0.307
    ['卡·秒（静态）' + (e.idle_source && e.idle_source !== 'idle_W' ? `（${num(e.idle_W_used)} W = idle/TDP × TDP）` : ''), c.idle_card_s, ' 卡·s', 'idle', 'idle_W']];
  const head = h('tr', {}, h('th', {}, '动作'), h('th', {}, `次数 / ${u}`), h('th', {}, `能耗 J / ${u}`));
  const body = rows.map(([lab, n, sfx, k, key]) => h('tr', {}, h('td', {}, lab), h('td', { class: 'num' }, sci(n) + sfx),
    h('td', { class: 'num' }, (e.provided.includes(key) || k in j) ? sci(j[k] || 0) : h('span', { class: 'muted' }, '未提供'))));
  if (e.J_per_unit !== undefined) body.push(h('tr', {}, h('td', {}, h('b', {}, '合计')), h('td', { class: 'num' }, ''),
    h('td', { class: 'num' }, h('b', {}, sci(e.J_per_unit)), ` · 平均 ${num(e.avg_W_per_card)} W/卡`)));
  if (e.idle_source_note) body.push(h('tr', {}, h('td', { colspan: 3, class: 'small muted' }, '静态功耗来源：' + e.idle_source_note)));   // 0.189
  if (e.static_share != null) body.push(h('tr', {}, h('td', { colspan: 3, class: 'small', style: e.static_warning ? 'color:var(--danger)' : '' },
    e.static_warning || `静态能耗占 ${(e.static_share * 100).toFixed(1)}%`)));
  const pc = e.power_cap;
  if (pc) body.push(h('tr', {}, h('td', { colspan: 3, class: 'small' }, pc.error ? '功耗封顶：' + pc.error
    : `功耗封顶：需 ${num(pc.P_needed_W)} W / TDP ${num(pc.tdp_W)} W → ` + (pc.capped ? `吞吐 ×${num(pc.throughput_factor)}，J/单位 ${sci(pc.J_per_unit_capped)}` : '未超 TDP')
      // 0.227: the clock / voltage the cap implies
      + (pc.dvfs ? '；' + Object.values(pc.dvfs).map((d) => d.error ? d.error
        : `${d.curve === 'dfs' ? '仅降频' : 'V–f（' + d.tag + '）'} f ${num(d.f_ghz)} GHz（×${num(d.s)}）V ×${num(d.v)} → 吞吐 ×${num(d.throughput_factor)}，${d.bound}`).join('；') : ''))));
  if (e.host_note) body.push(h('tr', {}, h('td', { colspan: 3, class: 'small muted' }, e.host_note)));
  if (e.note) body.push(h('tr', {}, h('td', { colspan: 3, class: 'small muted' }, e.note)));   // 0.131
  put($('energy-tbl'), h('thead', {}, head), h('tbody', {}, ...body));
  renderEnergyBreakdown(e);
}
/* 0.299: component breakdown (energy.energy_breakdown): MAC / SRAM / wire / NoC / DRAM / static …, Σ = J per unit */
const EB_COLOR = { mac: '#4e79a7', vec: '#76b7b2', sram: '#59a14f', slc: '#8cd17d', wire: '#edc948', noc: '#f28e2b',
  dram: '#e15759', io: '#b07aa1', clock: '#9c755f', static: '#bab0ac' };
function renderEnergyBreakdown(e) {
  const box = $('energy-breakdown');
  const sci = (x) => (x === 0 ? '0' : x.toExponential(3));
  const b = e && e.breakdown;
  if (!b || !(b.total > 0)) { put(box); return; }
  const keys = Object.keys(b.J).filter((k) => b.J[k] > 0);
  put(box, h('div', { class: 'wbar' }, ...keys.map((k) => h('i', { style: `width:${(b.share[k] * 100).toFixed(2)}%;background:${EB_COLOR[k] || '#999'}`,
    title: `${b.labels[k] || k}：${sci(b.J[k])} J（${(b.share[k] * 100).toFixed(1)}%）` }))),
    h('div', { class: 'legend' }, ...keys.map((k) => h('span', { style: `--c:${EB_COLOR[k] || '#999'}` },
      `${b.labels[k] || k} ${(b.share[k] * 100).toFixed(1)}%`))));
}
function renderPD(r) {
  const pd = r.pd;
  $('pd-box').hidden = !pd;
  if (!pd) return;
  if (pd.error) { put($('pd-tbl')); $('pd-note').textContent = pd.error; $('pdpc-box').hidden = true; return; }
  const c = pd.coloc, P = pd.prefill, D = pd.decode, K = pd.kv;
  const ok = (b) => b ? '' : 'color:var(--danger)';
  const head = h('tr', {}, h('th', { class: 'l' }, '模式'), h('th', { class: 'l' }, '卡'), h('th', {}, 'TTFT ms'), h('th', {}, 'TPOT ms'),
    h('th', {}, '有效 TPOT ms'), h('th', {}, 'goodput tok/s/卡'), h('th', { class: 'l' }, '瓶颈 / 说明'));
  const rows = [
    h('tr', { class: pd.goodput_per_card >= c.goodput_per_card ? 'best' : '' }, h('td', { class: 'l' }, h('b', {}, 'PD 分离')),
      h('td', { class: 'l small', style: 'white-space:normal;min-width:150px' }, `prefill ${P.cards}（${P.replicas} × ${P.layout}，batch ${P.batch}）+ decode ${D.cards}（${D.replicas} × ${D.layout}，batch ${D.batch}）`),
      h('td', { style: ok(pd.ttft_ok) }, num(pd.ttft_ms)), h('td', { style: ok(pd.tpot_ok) }, num(pd.tpot_ms)), h('td', { style: ok(pd.tpot_ok) }, num(pd.tpot_ms)),
      h('td', {}, h('b', {}, num(pd.goodput_per_card))),
      h('td', { class: 'l small', style: 'white-space:normal;min-width:150px' }, `${{ prefill: 'prefill 池', decode: 'decode 池', kv: 'KV 传输', host: '主机→卡加载', host_write: '卡→主机备份' }[pd.bottleneck]}；利用率 P ${pct(pd.util.prefill)} · D ${pct(pd.util.decode)} · KV ${pct(pd.util.kv)}${pd.util.host != null ? ` · 主机读 ${pct(pd.util.host)} · 主机写 ${pct(pd.util.host_write)}` : ''}`)),
    h('tr', { class: c.goodput_per_card > pd.goodput_per_card ? 'best' : '' }, h('td', { class: 'l' }, '合并（同卡数）'),
      h('td', { class: 'l small', style: 'white-space:normal;min-width:150px' }, `${c.cards}（${c.replicas} × ${c.layout}${c.idle_cards ? `，${c.idle_cards} 卡闲置` : ''}）`),
      h('td', { style: ok(c.ttft_ok) }, num(c.ttft_ms)), h('td', {}, num(c.tpot_ms)), h('td', { style: ok(c.tpot_eff_ok) }, num(c.tpot_eff_ms)),
      h('td', {}, h('b', {}, num(c.goodput_per_card))), h('td', { class: 'l small', style: 'white-space:normal;min-width:150px' }, `decode 时间占比 ${pct(c.decode_share)}（prefill 期间 decode 暂停）`)),
  ];
  put($('pd-tbl'), h('thead', {}, head), h('tbody', {}, ...rows));
  const bs = pd.best_split;
  $('pd-note').textContent = `KV 每请求 ${num(K.bytes_per_req / 2 ** 20)} MiB，${num(K.GBps_req)} GB/s（${K.source}，min(c_p, c_d) 卡并行），传输 ${num(K.t_ms)} ms，暴露 ${num(K.exposed_ms)} ms${K.layerwise ? '（逐层流式）' : ''}。`
    + (K.fabric ? ` 拓扑「假设」：上行系数 ${num(K.fabric.leaf_factor)}，NIC 被集合通信占用 ${pct(K.fabric.u_coll)}，${num(K.fabric.GBps_raw)} → ${num(K.fabric.GBps_eff)} GB/s/卡`
      + (K.fabric.feedback ? `；回馈：KV 占 decode 池 NIC ${pct(K.fabric.feedback.u_kv)} → decode TPOT ${num(K.fabric.feedback.tpot_ms_before)} → ${num(K.fabric.feedback.tpot_ms_after)} ms`
        + (K.fabric.feedback.ttft_ms_after != null ? `，prefill TTFT ${num(K.fabric.feedback.ttft_ms_before)} → ${num(K.fabric.feedback.ttft_ms_after)} ms` : '') : '') + '。' : '')
    + (bs ? ` 同 ${pd.cards} 卡最佳切分：prefill ${bs.prefill_cards} + decode ${bs.decode_cards} → ${num(bs.goodput_per_card)} tok/s/卡（瓶颈 ${bs.bottleneck}）。` : '')
    + (pd.warnings.length ? ' ⚠ ' + pd.warnings.join('；') : '');
  renderPDQueue(pd.queue, pd.cards, pd.lengths);
  renderPDLayouts(pd.layout_search);
  renderPDCache(pd.prefix_cache);
  if (P.hetero) $('pd-note').textContent = `异构池：prefill ${P.chip} / ${P.mem_id}，decode ${D.chip} / ${D.mem_id}（合并对照用 decode 池的芯片与存储器）。` + $('pd-note').textContent;
}
function renderPDCache(pc) {
  $('pdpc-box').hidden = !pc;
  if (!pc) return;
  const hn = pc.host ? (() => { const x = pc.host, l = x.load || {};   // 0.168/0.169 host L2 tier
    return `；主机 L2 ${num(x.GB)} GB/副本（${x.policy === 'write_through' ? 'write_through 包含式' : 'write_back 互斥式'}，`
      + `${x.K_host} 前缀）：显存命中 ${pct(x.hit_device)} + 主机命中 ${pct(x.hit_host)}；每次主机命中每卡加载 ${num(l.bytes_per_card_MB)} MB `
      + `${num(l.t_load_ms)} ms（逐层重叠后露出 ${num(l.exposed_ms)} ms，TTFT 均值 +${num(l.mean_added_ms)} ms）；`
      + `备份写 ${num(l.write_MB_card_req)} MB/卡/请求（主机链路 ${num(l.host_GBps_card)} GB/s「假设」）`; })() : '';
  if (pc.tree) {   // 0.58 radix tree
    $('pdpc-basis').textContent = `前缀树 ${pc.tree.map((r) => `${r[0]} tok × ${r[1]}（Zipf ${r[2]}）`).join(' → ')}，${pc.policy}；${pc.basis}`
      + hn + (pc.turns ? `；${pc.turns.turns} 轮 / 会话（每轮输出 ${pc.turns.output_tokens} + 用户 ${pc.turns.user_tokens} tok，平均 prompt ${num(pc.turns.mean_turn_prompt)} tok${pc.turns.auto_prompt ? '，已用作 prompt' : ''}）${pc.turns.session_survival != null ? `，会话历史存活 ${pct(pc.turns.session_survival)}` : ''}；${pc.turns.basis}` : '');
    const headT = h('tr', {}, h('th', { class: 'l' }, '缓存'), h('th', {}, '副本'), h('th', {}, '容量 GB', h('br'), '/ 副本'),
      h('th', {}, '容量', h('br'), 'tok / 副本'), h('th', {}, 'token', h('br'), '命中率'), h('th', {}, '各层命中', h('br'), 'P(深度 ≥ k)'), h('th', { class: 'l' }, '说明'));
    const rowsT = [['prefill', 'PD · prefill 池'], ['decode', 'PD · decode 池'], ['coloc', '合并副本']].map(([k, nm]) => {
      const x = pc[k];
      const note = (x.source === 'pd.prefix_cache_GB' ? '给定容量' : '剩余 DRAM（权重 + 活跃 KV 之后）')
        + (k === 'decode' ? `；decode 池也持有的匹配 token 比例 ${pct(x.holds_given_prefill_hit)}（只传其余 KV）` : '');
      return h('tr', {}, h('td', { class: 'l' }, nm), h('td', {}, String(x.replicas)), h('td', {}, num(x.capacity_GB)),
        h('td', {}, num(x.capacity_tokens)), h('td', {}, h('b', {}, pct(x.token_hit))), h('td', {}, x.level_hit.map((v) => pct(v)).join(' / ')),
        h('td', { class: 'l small', style: 'white-space:normal;min-width:180px' }, note));
    });
    put($('pdpc-tbl'), h('thead', {}, headT), h('tbody', {}, ...rowsT));
    return;
  }
  $('pdpc-basis').textContent = `${pc.prefix_count} 个前缀 × ${pc.prefix_len} tok，Zipf α ${pc.zipf}，${pc.policy}；${pc.basis}` + hn;
  const head = h('tr', {}, h('th', { class: 'l' }, '缓存'), h('th', {}, '副本'), h('th', {}, '容量 GB', h('br'), '/ 副本'),
    h('th', {}, '每前缀', h('br'), 'MB'), h('th', {}, '可存前缀', h('br'), '/ 副本'), h('th', {}, '命中率'), h('th', { class: 'l' }, '说明'));
  const src = (x) => x.source === 'pd.prefix_cache_GB' ? '给定容量' : '剩余 DRAM（权重 + 活跃 KV 之后）';
  const rows = [['prefill', 'PD · prefill 池'], ['decode', 'PD · decode 池'], ['coloc', '合并副本']].map(([k, nm]) => {
    const x = pc[k];
    const note = src(x) + (k === 'decode' ? `；prefill 命中时 decode 池也持有该前缀的概率 ${pct(x.holds_given_prefill_hit)}（只传未缓存部分 KV）` : '');
    return h('tr', {}, h('td', { class: 'l' }, nm), h('td', {}, String(x.replicas)), h('td', {}, num(x.capacity_GB)), h('td', {}, num(x.footprint_MB)),
      h('td', {}, String(x.K)), h('td', {}, h('b', {}, pct(x.hit))), h('td', { class: 'l small', style: 'white-space:normal;min-width:180px' }, note));
  });
  put($('pdpc-tbl'), h('thead', {}, head), h('tbody', {}, ...rows));
}
function renderPDLayouts(ls) {
  $('pdls-box').hidden = !ls;
  if (!ls) return;
  $('pdls-basis').textContent = `${ls.cards} 卡，${ls.candidates} 种布局${ls.truncated ? '（已截断）' : ''}，${ls.pairs} 对；${ls.basis}`;
  const showB = ls.decode_batch_searched;
  const head = h('tr', {}, h('th', { class: 'l' }, 'prefill 布局 × 卡'), h('th', { class: 'l' }, 'decode 布局 × 卡'),
    h('th', {}, 'prefill', h('br'), 'TTFT ms'), h('th', {}, 'decode', h('br'), 'TPOT ms'),
    ...(showB ? [h('th', {}, 'decode', h('br'), 'batch')] : []), h('th', {}, '流体 goodput', h('br'), 'tok/s/卡'),
    h('th', { class: 'l' }, '瓶颈'), h('th', {}, 'SLO goodput', h('br'), 'tok/s/卡'));
  const bs = ls.best_slo, cur = ls.current_layouts;
  const same = (a, b) => a && b && a.prefill_layout === b.prefill_layout && a.decode_layout === b.decode_layout && a.prefill_cards === b.prefill_cards;
  const rows = [...ls.rows.slice(0, 8)];
  for (const x of [bs, cur]) if (x && !rows.some((r) => same(r, x))) rows.push(x);
  const body = rows.map((r) => h('tr', { class: same(r, bs) ? 'best' : '' },
    h('td', { class: 'l' }, `${r.prefill_layout} × ${r.prefill_cards}`, same(r, cur) ? h('span', { class: 'tag', style: 'margin-left:6px' }, '当前布局') : null),
    h('td', { class: 'l' }, `${r.decode_layout} × ${r.decode_cards}`), h('td', {}, num(r.prefill_ttft_ms)), h('td', {}, num(r.decode_tpot_ms)),
    ...(showB ? [h('td', {}, String(r.decode_batch))] : []), h('td', {}, num(r.goodput_per_card)), h('td', { class: 'l' }, { prefill: 'prefill 池', decode: 'decode 池', kv: 'KV 传输', host: '主机→卡加载', host_write: '卡→主机备份' }[r.bottleneck]),
    h('td', {}, r.slo_goodput_per_card === undefined ? h('span', { class: 'muted' }, '未算') : h('b', {}, num(r.slo_goodput_per_card)))));
  put($('pdls-tbl'), h('thead', {}, head), h('tbody', {}, ...body));
  $('pdls-note').textContent = bs ? `SLO 下最佳：prefill ${bs.prefill_layout} × ${bs.prefill_cards} + decode ${bs.decode_layout} × ${bs.decode_cards} → ${num(bs.slo_goodput_per_card)} tok/s/卡（流体 ${num(bs.goodput_per_card)}）。流体排名与 SLO 排名可以不同：流体只看容量，SLO 还看 TTFT / TPOT 尾部。` : '';
}
function renderPDQueue(q, cards, L) {
  if (!q || q.error || !q.modes) { put($('pdq-tbl')); $('pdq-basis').textContent = ''; $('pdq-note').textContent = q && q.error ? q.error : ''; return; }
  $('pdq-basis').textContent = `到达率 ${num(q.lambda_rps)} req/s${q.load ? `（PD 容量的 ${pct(q.load)}）` : ''}；${q.basis}`;
  const sci = (x) => (x === 0 ? '0' : x.toExponential(3));
  const names = { pd: 'PD 分离', coloc_prefill_first: '合并 · prefill 优先', coloc_chunked: `合并 · 分块 prefill（${(q.chunk_profile || {}).profile || 'fixed'} · ${q.chunk_tokens} tok${(q.chunk_profile || {}).budget ? ' · 预算 ' + q.chunk_profile.budget : ''}）` };
  const ok = (b) => b === false ? 'color:var(--danger)' : '';
  const E = q.energy || {};
  const hasJ = Object.values(E).some((e) => e.J_per_token !== undefined);
  const head = h('tr', {}, h('th', { class: 'l' }, '模式'), h('th', {}, 'TTFT ms', h('br'), 'p50/p90/p99'), h('th', {}, 'TPOT ms', h('br'), '均值/p90/p99'),
    h('th', {}, '最长停顿', h('br'), 'ms'), h('th', {}, 'SLO 到达率', h('br'), 'req/s'), h('th', {}, 'SLO goodput', h('br'), 'tok/s/卡'),
    h('th', {}, hasJ ? 'J /' : 'DRAM B /', h('br'), '输出 token'), h('th', { class: 'l' }, '说明'));
  const best = Math.max(...Object.values(q.modes).map((x) => x.slo_goodput_per_card || 0));
  const rows = Object.entries(q.modes).map(([k, x]) => {
    const e = E[k] || {};
    const en = hasJ ? (e.J_per_token !== undefined ? `${sci(e.J_per_token)}${e.tok_per_J ? ` (${num(e.tok_per_J)} tok/J${e.static_share ? `，静态 ${pct(e.static_share)}` : ''})` : ''}` : '—') : (e.counts_per_token ? sci(e.counts_per_token.dram) : '—');
    let note = '';
    if (!x.stable) note = '不稳定：' + (x.why || '');
    else if (k === 'pd') note = `prefill batch 上限 ${x.prefill.batch_cap}，排队均值 ${num(x.prefill.wait_ms.mean)} ms；decode 运行 batch ≈ ${num(x.decode.occupancy)}（p90 ${x.decode.running_p90}）/ ${x.decode.slots} 槽；KV 链路被集合通信占用 ${pct(x.kv.u_coll)}`;
    else if (k === 'coloc_prefill_first') note = `decode 时间占比 ${pct(x.decode.share)}，运行 batch ≈ ${num(x.decode.occupancy)}；新请求 prefill 时整批 decode 停顿`;
    else if (x.prefill && x.prefill.nonfused) note = `SGLang 不混合：每请求 ${num(x.prefill.chunks)} 个 prefill-only 块连续执行，期间整批 decode 暂停；decode 时间占比 ${pct(x.decode.share)}，运行 batch ≈ ${num(x.decode.occupancy)}`;
    else note = `每请求 ${Number.isInteger(x.prefill.chunks) ? x.prefill.chunks : num(x.prefill.chunks)} 块，融合迭代 ${num(x.prefill.iter_ms)} ms（无分块 ${num(x.decode.iter_ms_no_chunk)} ms），prefill 占用 ${pct(x.prefill.rho)}`;
    if (x.stable && x.kv_cap) {
      const c = x.kv_cap;
      const pre = c.policy === 'recompute' ? `，抢占 ${num(c.preempt_per_req)} 次/请求（重算 ${num(c.recompute_ms)} ms）`
        : c.policy === 'swap' ? `，抢占 ${num(c.preempt_per_req)} 次/请求（换出 + 换入 ${num(2 * c.swap_ms)} ms，${num(c.swap_GBps_card)} GB/s/卡${c.swap_source === 'pd.swap_GBps' ? '' : '「假设」'}）` : '';
      const rs = (c.restore_share || 0) > 0 ? `，恢复停顿占副本 ${pct(c.restore_share)} → 槽位占用 ×${num(c.slot_slowdown)}「假设」` : '';
      const hold = c.slot_hold_prefill_ms ? `，prefill 期间占槽 ${num(c.slot_hold_prefill_ms)} ms` : '';
      note += `；KV ${c.policy}：${c.capacity_tokens} tok / 副本 → ${c.slots_kv} 槽` + (c.binds ? `（生效，准入等待均值 ${num(c.slot_wait_mean_ms)} ms（P ${pct(c.p_wait || 0)}，M/G/c 槽位队列${hold}），`
        + (c.in_ttft ? '先准入后 prefill（vLLM）：已计入 TTFT 与 SLO goodput' : 'prefill 后准入：单列、不计入 TTFT / TPOT / SLO goodput') + pre + rs + '）' : '（不生效）');
    }
    if (e.static_note) note += '；' + e.static_note;
    const t = x.ttft_ms;
    return h('tr', { class: x.slo_goodput_per_card > 0 && x.slo_goodput_per_card >= best ? 'best' : '' }, h('td', { class: 'l' }, k === 'pd' ? h('b', {}, names[k]) : names[k]),
      h('td', { style: ok(x.ttft_p90_ok) }, x.stable ? `${num(t.p50)}/${num(t.p90)}/${num(t.p99)}` : '—'),
      h('td', { style: ok(x.tpot_p90_ok) }, x.stable ? `${num(x.tpot_mean_ms)}/${num(x.tpot_p90_ms)}/${num(x.tpot_p99_ms)}` : '—'),
      h('td', {}, x.stable ? num(x.itl_max_ms) : '—'), h('td', {}, num(x.slo_rate_rps)), h('td', {}, h('b', {}, num(x.slo_goodput_per_card))),
      h('td', {}, en), h('td', { class: 'l small', style: 'white-space:normal;min-width:180px' }, note));
  });
  const sm = (q.sim && q.sim.modes) || {};
  const ep = (e) => { if (e === undefined || e === null || !isFinite(e)) return ''; const r = Math.round(e * 100) || 0; return ` (${r > 0 ? '+' : ''}${r}%)`; };
  for (const [k, x] of Object.entries(sm)) {
    const t = x.ttft_ms, tp = x.tpot_ms, e = x.err || {};
    rows.push(h('tr', { class: 'muted' }, h('td', { class: 'l' }, `DES · ${names[k]}`),
      h('td', {}, `${num(t.p50)}/${num(t.p90)}/${num(t.p99)}`), h('td', {}, `${num(tp.mean)}/${num(tp.p90)}/${num(tp.p99)}`),
      h('td', {}, num(x.itl_max_ms.p99)), h('td', {}, '—'), h('td', {}, '—'), h('td', {}, '—'),
      h('td', { class: 'l small', style: 'white-space:normal;min-width:180px' },
        `闭式相对 DES：TTFT p90${ep(e.ttft_p90)}，TPOT p90${ep(e.tpot_p90)}，最长停顿${ep(e.itl_max)}` + (x.prefix_hit != null ? `；DES 前缀命中 ${pct(x.prefix_hit)}` : ''))));
  }
  put($('pdq-tbl'), h('thead', {}, head), h('tbody', {}, ...rows));
  const b = q.pd_slo_best_split;
  $('pdq-note').textContent = '「SLO 到达率」= p90 TTFT 与 p90 TPOT 都满足 SLO（未设 SLO 时即稳定上限）的最大泊松到达率；SLO goodput = 该到达率 × 输出长度 / 卡数（DistServe 口径）。'
    + (b ? ` 同 ${cards} 卡 PD 在 SLO 下的最佳切分：prefill ${b.prefill_cards} + decode ${b.decode_cards} → ${num(b.slo_goodput_per_card)} tok/s/卡。` : '')
    + (L && !L.plain ? ` 长度「假设」（${{ fixed: '固定', cv: '对数正态 8 档', mix: '离散分布' }[L.source]}）：prompt 均值 ${num(L.mean_prompt)}（CV ${num(L.prompt_cv_eff)}），输出均值 ${num(L.mean_out)}（CV ${num(L.out_cv_eff)}），decode 上下文按长度偏置取 ${L.decode_ctx}${L.prefix_hit ? `；前缀命中 ${pct(L.prefix_hit)}${L.prefix_hit_source === 'capacity' ? '（容量模型，PD prefill 池）' : ''}${L.prefix_on_decode ? '（KV 只传未缓存部分）' : '（KV 全量交接）'}` : ''}。` : '')
    + ' 稳定上限：' + Object.entries(q.modes).map(([k, x]) => `${names[k]} ${num(x.stable_rate_rps)}`).join('，') + ' req/s。'
    + (q.sim ? ' DES 行「假设」：请求级离散事件仿真（FCFS、泊松、同一套逐步代价，n = ' + q.sim.n_req + '），括号内为（闭式 − DES）/ DES，> 0 = 闭式偏保守。' : '')
    + ' TTFT 分位数：服务时间混合时等待与自身时延按独立和计算（0.54），单一服务时间时逐项相加（精确）；KV 容量排队 / recompute / swap 抢占只在「KV 策略」非 off 且容量受限时计入（0.55 / 0.56；默认先准入后 prefill = vLLM，等待计入 TTFT 与 SLO goodput；抢占者的停顿按混合计入 TPOT 尾部与最长停顿）；未建模：调度器开销（前缀缓存容量与 LRU 淘汰只在设了「前缀长度」时按 Che 近似计入，见上表）。';
}
function budgetMark(x) {
  if (x.budget_ok === false) return h('span', { class: 'tag danger', title: '超出预算：' + (x.budget_violations || []).join(', ') }, '超预算');
  if (x.budget_ok === null) return h('span', { class: 'tag', title: '预算无法判定（缺输入或只有下界）' }, '预算?');
  return null;
}
// 0.96: 敏感度 panel — the result under the 「假设」 constants that drive design conclusions (MODEL §41)
const SENS_ZH = { exec_overlap: '重叠上下界（stage / serial）', pipe: '硬件推导重叠（pipe，现实估计）', mem_eff: 'DRAM 效率', vector_lanes: '向量 lane 数' };
// 0.208: balance limit / saturation labels, shared by the KPI and the what-if table
const LIMIT_ZH = { kv_capacity: '内存容量 capacity', tpot_slo: 'TPOT SLO', ttft_slo: 'TTFT SLO' };
const ASYM_ZH = { kv: 'KV（随 ctx 增长）', state: '循环状态（每序列固定）', 'kv+state': 'KV + 状态（混合）', weights: '权重（MoE 专家）', act: '激活' };
// 0.208: machine-balance what-if (one lever each), from the same /api/sensitivity response — no extra request
// 0.227: does the user's power cap bind at this lever's operating point — dfs (clock only, headline) / V–f curve (二手)
function capCell(pc, base) {
  if (!pc) return '';
  if (pc.error) return h('div', { class: 'small' }, '封顶：' + pc.error);
  const g = (v) => v == null ? '—' : num(v) + '×';
  return h('div', { class: 'small cap-cell' + (pc.binds ? ' warn' : ''), style: pc.binds ? 'color:var(--danger)' : '' },
    !pc.feasible ? `封顶 ${num(pc.cap_W)} W：静态功耗已超`
      : pc.binds ? `封顶 ${num(pc.cap_W)} W 生效：吞吐 dfs ×${num(pc.tf_dfs)} · V–f ×${num(pc.tf_vf)}` + (base ? '' : `；对基线 dfs ${g(pc.gain_dfs)} · V–f ${g(pc.gain_vf)}`)
      : `封顶 ${num(pc.cap_W)} W 未生效` + (base ? '' : `；对基线 dfs ${g(pc.gain_dfs)}`));
}
function renderBalanceWhatif(w) {
  let box = $('bal-whatif');
  if (!box) { box = h('div', { id: 'bal-whatif' }); $('sens-tbl').after(box); }
  if (!w) { put(box); return; }
  const cell = (x) => x.reachable ? `${x.critical_batch} ✓` : (x.critical_batch ? `${x.critical_batch} ✗` : '∞');
  const tr = (lab, x, base) => h('tr', { class: x.works ? 'ok' : '' }, h('td', { class: 'l' }, lab),
    h('td', {}, num(x.chip_flop_per_byte)), h('td', {}, cell(x)), h('td', {}, `${x.batch_limit}${x.limited_by ? '（' + (LIMIT_ZH[x.limited_by] || x.limited_by) + '）' : ''}`),
    h('td', {}, num(x.ai_asymptote)), h('td', {}, num(x.tpot_ms) + ' ms'),
    // 0.211: lever cost — Δ die area (SRAM incl. state buffer + DRAM PHY) and dynamic J/token at the lever's critical batch
    h('td', { class: x.over_reticle ? 'warn' : '' }, base ? (x.area_mm2 != null ? num(x.area_mm2) + ' mm²' : '—')
      : (x.d_area_mm2 != null ? (x.d_area_mm2 >= 0 ? '+' : '') + num(x.d_area_mm2) + ' mm²' + (x.over_reticle ? ' ⚠ 超光罩' : '') : '—'),
      // 0.212: HBM shoreline (stacks × 11 mm vs the square die's two long sides / four sides)
      x.dies > 1 ? h('div', { class: 'small' }, `每晶粒 per die × ${x.dies}；D2D PHY ${num(x.d2d_phy_mm2 || 0)} mm²/晶粒`
        + (x.d2d_J_per_tok != null ? ` · D2D ${num(x.d2d_J_per_tok * 1e3)} mJ/tok` : '') + (x.coll_lat_ms ? ` · 集合延迟 ${num(x.coll_lat_ms)} ms/步` : '')) : '',   // 0.214
      x.shoreline ? h('div', { class: 'small' + (x.shoreline.status !== 'ok' ? ' warn' : ''), title: x.shoreline.warning || '' },
        `岸线 ${x.shoreline.stacks} 堆栈 ${num(x.shoreline.need_mm)} / ${num(x.shoreline.edge_typical_mm)} mm（两边）`
        + (x.shoreline.status === 'tight' ? ' ⚠ 需三、四边' : x.shoreline.status === 'over' ? ' ⚠ 超出四边 · 需多晶粒' : '')) : ''),
    h('td', {}, x.J_per_tok == null ? '—' : `${num(x.J_per_tok * 1e3)} mJ @B${x.cost_batch}${x.cost_basis ? '（' + ({critical_batch: '临界', batch_limit: '上限', configured: '配置'}[x.cost_basis] || x.cost_basis) + '）' : ''}`
      + (x.base_J_per_tok_at_cost_batch != null ? `（基线 ${num(x.base_J_per_tok_at_cost_batch * 1e3)}${x.base_cost_batch != null && x.base_cost_batch !== x.cost_batch ? ' @B' + x.base_cost_batch : ''}）` : '')),
    h('td', {}, (x.W_card != null ? num(x.W_card) + ' W' + (x.cost_batch != null ? ` @B${x.cost_batch}` : '') : '—')
      + (x.d_leak_W ? `（泄漏 ${x.d_leak_W > 0 ? '+' : ''}${num(x.d_leak_W)} W）` : (base && x.leak_W ? `（泄漏 ${num(x.leak_W)} W）` : ''))
      + (x.d_hbm_W ? `（HBM ${x.d_hbm_W > 0 ? '+' : ''}${num(x.d_hbm_W)} W）` : (base && x.hbm_W ? `（HBM ${num(x.hbm_W)} W）` : '')),
      capCell(x.power_cap, base)),
    h('td', { class: 'l small' }, base ? '当前' : (x.error ? x.error : (x.works ? '有效 · works' : '无效 · no effect')),
      // 0.223: spec rollback lever — both modes at batch 1 / 32
      x.spec_rollback_compare ? h('div', { class: 'small', id: 'spec-roll-cmp' }, x.spec_rollback_compare.map((c) => c.error
        ? `B${c.batch} ${c.mode}: ${c.error}`
        : `B${c.batch} ${c.mode}: ${num(c.tpot_ms)} ms · ${num(c.tok_s_card)} tok/s/卡 · ${num(c.dram_need_GiB)} GiB${c.fits ? '' : ' ⚠ 放不下'}`).join('；')) : '',
      // 0.225: drafting shape lever — chain vs tree (both acceptance bounds) per batch + crossover
      x.spec_tree_compare ? treeCmp(x.spec_tree_compare) : ''));
  put(box, h('h4', {}, '机器平衡 what-if · balance levers（每次只改一个变量 · one variable each）'),
    h('table', { class: 'tbl', id: 'bal-whatif-tbl' },
      h('thead', {}, h('tr', {}, ...['杠杆 lever', '芯片 FLOP/B', '临界 batch critical batch', '上限 batch limit', '饱和强度 asymptote', 'TPOT', 'Δ面积 area', 'J/token @计价 batch（临界 / 上限 / 配置）', '功耗 W/卡（含 SRAM 泄漏 + DRAM 后台/刷新）', '判定'].map((t, i) => h('th', i ? {} : { class: 'l' }, t)))),
      h('tbody', {}, tr('当前 base', w.base, true), ...w.rows.map((x) => x.error ? h('tr', {}, h('td', { class: 'l' }, x.label), h('td', { colspan: 9, class: 'l small' }, x.error)) : tr(x.label, x, false)))),
    w.cost_basis ? h('div', { class: 'small' }, w.cost_basis.note) : '',
    h('div', { class: 'small' }, w.works.length ? `有效杠杆 · levers that work：${w.works.join('、')}（阈值 ${pct(w.gain)}）` : '没有单一杠杆有效 · no single lever works'),
    w.power_cap_W ? h('div', { class: 'small', id: 'whatif-cap' }, `功耗封顶 ${num(w.power_cap_W)} W/卡下仍有效 · works under the cap（dfs 主值）：${w.works_capped.length ? w.works_capped.join('、') : '无'}；封顶生效：${w.binds.length ? w.binds.join('、') : '无'}。${w.cap_basis}`) : '');
}
function treeCmp(t) {
  // 0.226: headline = the scenario's acceptance model (default chain); the band [chain, indep] and rank alongside
  const one = (v) => !v ? '—' : v.error ? v.error : `${num(v.tpot_ms)} ms（τ ${num(v.tau)}）${v.fits ? '' : ' ⚠ 放不下'}`;
  const fl = (a) => !a ? '—' : a.length ? a.map(([p, q]) => `B${p}→B${q}`).join('、') : '无翻转';
  const hd = t.headline || 'chain';
  return h('div', { class: 'small', id: 'spec-tree-cmp' },
    `链 vs 树（k ${t.k}，topk ${t.topk}，N ${t.draft_tokens}；主值 ${hd}）：`,
    ...t.rows.map((x) => h('div', {}, `B${x.batch} 链 ${one(x.chain)} · 树（${hd}）${one(x['tree_' + hd])} · 区间 [chain ${one(x.tree_chain)}, indep ${one(x.tree_indep)}] · rank ${one(x.tree_rank)}`)),
    h('div', {}, `胜负翻转 · crossover（主值 ${hd}）：${fl(t.crossover)}；chain ${fl(t.crossover_chain)}；rank ${fl(t.crossover_rank)}；indep ${fl(t.crossover_indep)}。对照 EAGLE-1 已发表链/树配对：chain ${pct(t.check.chain[0])}…${pct(t.check.chain[1])}，rank ${pct(t.check.rank[0])}…${pct(t.check.rank[1])}，indep ${pct(t.check.indep[0])}…${pct(t.check.indep[1])}；已发表树增益 ${pct(t.check.gain_published[0])}…${pct(t.check.gain_published[1])}（不拟合）`));
}
function renderSensitivity(r) {
  const s = r.sensitivity;
  $('sens-box').hidden = false;
  if (!s) { put($('sens-tbl')); renderBalanceWhatif(null); return; }
  const m = model(), u = isFull(m) ? unitZh(m) : 'tok';
  const head = h('tr', {}, h('th', { class: 'l' }, '假设变化'), h('th', {}, `${u}/s`), h('th', {}, '相对当前'), h('th', {}, '瓶颈'), h('th', { class: 'l' }, '判定'));
  const rows = s.rows.map((x) => x.error ? h('tr', {}, h('td', { class: 'l' }, x.label), h('td', { colspan: 4, class: 'l small' }, x.error))
    : h('tr', { class: x.hinge ? 'warn' : '' }, h('td', { class: 'l' }, x.label), h('td', {}, num(x.tok_s)),
      h('td', { style: x.hinge ? 'color:var(--danger);font-weight:700' : '' }, num(x.ratio) + '×'),
      h('td', {}, boundTag(x.bound), x.bound_flip ? h('span', { class: 'small' }, ' ⇄') : ''),
      h('td', { class: 'l small' }, x.hinge ? (x.bound_flip ? '瓶颈翻转' : `变化 > ${pct(s.threshold)}`) : '稳健')));
  put($('sens-tbl'), h('thead', {}, head), h('tbody', {}, ...rows));
  renderBalanceWhatif(s.balance_whatif);
  $('sens-note').textContent = s.hinges.length
    ? `⚠ 本结果取决于：${s.hinges.map((k) => SENS_ZH[k] || k).join('、')}——比较设计点时请在这些假设下复核（当前 ${num(s.base_tok_s)} ${u}/s，瓶颈 ${s.base_bound}）`
    : `本结果对重叠方式、DRAM 效率、向量 lane 的变化不敏感（阈值 ${pct(s.threshold)}）`;
  if (s.scope && s.scope !== '本场景') $('sens-note').textContent += `；范围：${s.scope}`;
  if (s.base_label) $('sens-note').textContent += `；当前重叠：${s.base_label}`;   // 0.99: stage = 上界, pipe = 现实, serial = 下界
}
// 0.97: sweep / Pareto sensitivity summaries
function renderSweepSens(s, xl, u) {
  $('sw-sens-box').hidden = !s;
  if (!s) return;
  const head = h('tr', {}, h('th', { class: 'l' }, '假设变化'), h('th', { class: 'l' }, '最佳取值（1% 内并列）'),
    h('th', { class: 'l' }, `拐点（达最佳 ${pct(1 - s.knee_tol)}）`), h('th', {}, '排序翻转对数'), h('th', { class: 'l' }, '瓶颈翻转点'), h('th', { class: 'l' }, '判定'));
  const base = h('tr', {}, h('td', { class: 'l' }, '当前假设'), h('td', { class: 'l' }, s.best_values.join(', ') || '—'),
    h('td', { class: 'l' }, s.knee_value ?? '—'), h('td', {}, '—'), h('td', { class: 'l' }, '—'), h('td', { class: 'l small' }, '基准'));
  const rows = s.cases.map((c) => {
    const bad = c.best_changed || c.flips || c.knee_changed;
    const why = [c.best_changed && '最佳取值改变', c.knee_changed && '拐点移动', c.flips && `${c.flips} 对排序翻转`].filter(Boolean).join('；');
    return h('tr', { class: bad ? 'warn' : '' }, h('td', { class: 'l' }, c.label),
      h('td', { class: 'l', style: c.best_changed ? 'color:var(--danger);font-weight:700' : '' }, c.best_values.join(', ') || '—'),
      h('td', { class: 'l', style: c.knee_changed ? 'color:var(--danger);font-weight:700' : '' }, c.knee_value ?? '—'),
      h('td', {}, c.flips), h('td', { class: 'l small' }, c.rows.filter((x) => x.bound_flip).map((x) => `${x.value}→${x.bound}`).join(', ') || '—'),
      h('td', { class: 'l small' }, bad ? '⚠ ' + why : '稳健'));
  });
  put($('sw-sens-tbl'), h('thead', {}, head), h('tbody', {}, base, ...rows));
  $('sw-sens-note').textContent = s.hinges.length
    ? `⚠ 「${xl}」扫描的结论取决于：${s.hinges.map((k) => SENS_ZH[k] || k).join('、')}（最佳取值 / 拐点 / 排序在该假设下改变）`
    : `「${xl}」扫描的最佳取值、拐点与排序在各假设下不变`;
}
function renderParetoSens(s) {
  $('sw-sens-box').hidden = !s;
  if (!s) return;
  const head = h('tr', {}, h('th', { class: 'l' }, '假设变化'), h('th', { class: 'l' }, '前沿 batch'), h('th', { class: 'l' }, '离开前沿'), h('th', { class: 'l' }, '新进前沿'));
  put($('sw-sens-tbl'), h('thead', {}, head), h('tbody', {}, ...s.cases.map((c) => c.error
    ? h('tr', {}, h('td', { class: 'l' }, c.label), h('td', { colspan: 3, class: 'l small' }, c.error))
    : h('tr', { class: c.off_front.length || c.new_on_front.length ? 'warn' : '' }, h('td', { class: 'l' }, c.label),
      h('td', { class: 'l small' }, c.front_batches.join(', ')), h('td', { class: 'l', style: c.off_front.length ? 'color:var(--danger)' : '' }, c.off_front.join(', ') || '—'),
      h('td', { class: 'l' }, c.new_on_front.join(', ') || '—')))));
  $('sw-sens-note').textContent = s.hinges.length
    ? `⚠ Pareto 前沿取决于：${s.hinges.map((k) => SENS_ZH[k] || k).join('、')}`
    : 'Pareto 前沿上的点在各假设下不变';
}
function renderBudget(r) {
  const b = r.budget;
  $('budget-box').hidden = !b;
  if (!b) return;
  const mark = (ok) => ok === true ? h('span', { style: 'color:var(--ok)' }, '✓ 未超') : ok === false ? h('span', { style: 'color:var(--danger);font-weight:700' }, '✗ 超出') : h('span', { class: 'muted' }, '? 无法判定');
  const head = h('tr', {}, h('th', { class: 'l' }, '项目'), h('th', {}, '用量'), h('th', {}, '限额'), h('th', {}, '余量'), h('th', { class: 'l' }, '判定'), h('th', { class: 'l' }, '说明'));
  const rows = b.items.map((i) => h('tr', {}, h('td', { class: 'l' }, i.label), h('td', {}, i.value === null ? '—' : num(i.value) + ' ' + i.unit),
    h('td', {}, num(i.limit) + ' ' + i.unit), h('td', {}, i.headroom === null ? '—' : pct(i.headroom)), h('td', { class: 'l' }, mark(i.ok)),
    h('td', { class: 'l small' }, i.note || '')));
  if (b.shoreline) rows.push(h('tr', { class: b.shoreline.status !== 'ok' ? 'warn' : '' }, h('td', { class: 'l' }, 'HBM 岸线 shoreline'),
    h('td', {}, num(b.shoreline.need_mm) + ' mm'), h('td', {}, num(b.shoreline.edge_typical_mm) + ' mm'), h('td', {}, '—'),
    h('td', { class: 'l' }, b.shoreline.status === 'ok' ? '✓' : '⚠'), h('td', { class: 'l small' }, b.shoreline.warning || `${b.shoreline.stacks} 堆栈 × 11 mm；两长边可放 ${b.shoreline.fit_typical}`)));   // 0.212
  if (b.area && b.area.mm2 !== null) rows.push(h('tr', {}, h('td', { class: 'l' }, '面积代理分项'), h('td', { colspan: 5, class: 'l small' },
    Object.entries(b.area.terms).map(([k, v]) => `${{ sram: 'SRAM', slc: 'SLC', mac: 'MAC 阵列', fixed: '固定' }[k] || k} ${num(v)} mm²`).join(' + ') + ` = ${num(b.area.mm2)} mm² / 卡「假设」`)));
  if (b.scope_note) rows.push(h('tr', {}, h('td', { class: 'l' }, '范围'), h('td', { colspan: 5, class: 'l small' }, b.scope_note)));
  put($('budget-tbl'), h('thead', {}, head), h('tbody', {}, ...rows));
}
// 0.204: per-step walls (r.walls from the same /api/eval response — rendered with it, so it can never show a stale
// or mismatched step; no extra request channel to race)
const WALL_LBL = { compute: ['计算', 'compute'], dram: ['DRAM', 'DRAM'], collective_bw: ['集合通信带宽', 'collective BW'],
  collective_lat: ['集合通信延迟', 'collective latency'], dispatch: ['派发', 'dispatch'], fixed_overhead: ['固定开销', 'fixed overhead'] };
function renderWalls(r) {
  const w = r.walls, box = $('walls-box');
  if (!box) return;
  if (!w || !(w.stage_ms > 0)) { box.hidden = true; return; }
  box.hidden = false;
  const keys = Object.keys(WALL_LBL);
  put($('walls-bar'), ...keys.filter((k) => w.share[k] > 0).map((k) => h('i', {
    class: `wk-${k}` + (k === w.wall ? ' wall' : ''), 'data-k': k, style: `width:${w.share[k] * 100}%`,
    title: `${WALL_LBL[k][0]} · ${WALL_LBL[k][1]}：${num(w.exposed_ms[k])} ms（${pct(w.share[k])}）· 原始 raw ${num(w.raw_ms[k])} ms` })));
  put($('walls-legend'), ...keys.map((k) => h('span', { class: 'wleg' + (k === w.wall ? ' wall' : '') },
    h('i', { class: `wk-${k}` }), h('b', {}, `${WALL_LBL[k][0]} ${WALL_LBL[k][1]}`), ` ${num(w.exposed_ms[k])} ms · ${pct(w.share[k])}`)));
  const lb = WALL_LBL[w.wall] || [w.wall, w.wall];
  $('walls-note').textContent = `墙 wall：${lb[0]} · ${lb[1]} · 流水级 stage ${w.stage} ${num(w.stage_ms)} ms / 步 step ${num(w.step_ms)} ms`
    + ` · 派发 dispatch：${w.dispatch_mode}` + dispSensNote(r.dispatch_sens)
    + (w.exact ? '' : ` · 近似：exec_overlap = ${w.overlap} 下重叠窗口按主导资源归因 · approximate: overlapped window attributed to the dominant resource under exec_overlap = ${w.overlap}`)
    + ' · 固定开销 = 每层固定开销时间（layer_overhead_us），非静态功耗 · fixed overhead = per-layer fixed time, not static power';
}
/* 0.284: per-op dispatch sensitivity of the step (Result.dispatch_sens) — where the dispatch knob lives and what it costs */
function dispSensNote(d) {
  if (!d || !(d.ops_per_step > 0)) return '';
  return ` · 派发敏感度 dispatch sensitivity：${num(d.ops_per_step)} op/步，每 op ${d.us_10 != null ? num(d.us_10) : '—'} µs 时步长 +10 %`
    + `（trace ${num(d.us_at.trace)} µs → ×${num(d.slowdown_at.trace)}，host ${num(d.us_at.host)} µs → ×${num(d.slowdown_at.host)}${d.exact ? '' : '，流水线近似'}）`
    + ' — 在芯片面板 › 高级芯片参数 › 派发 dispatch 中设置';
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
      h('td', { class: 'l' }, st.index), h('td', { class: 'l' }, `${st.layers[0]}–${st.layers[1] - 1}`), h('td', {}, boundTag(st.bound, st.dequant_bound)),
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
    ...(hasSlc ? [h('td', { title: st.mem.slc_policy === 'lru' ? (st.mem.slc_all ? 'lru：片外工作集放得下 → 全部读命中' : 'lru：放不下 → 循环访问全部未命中') : st.mem.slc_policy === 'dip' ? `dip：每类读命中 ${pct(st.mem.slc_frac)}（${st.mem.slc_all ? '放得下 → 全部命中' : '循环访问 (K − ε(T − K)) / T，ε = 1/32'}）` : `${st.mem.slc_policy === 'pin_bw' ? 'pin_bw（带宽均衡分流）' : 'pin'}：SLC 中权重 ${num(st.mem.slc_w_MiB)} MiB，KV / 状态 ${num(st.mem.slc_kv_MiB)} MiB` }, pct(st.mem.slc_residency)),
      h('td', {}, num(st.dram_GB.slc || 0))] : []), h('td', {}, num(st.convert_Melems))));
  put($('mem-tbl'), h('thead', {}, mh), h('tbody', {}, mr));
}
function renderAssumptions(r) {
  const c = r.scenario.chip, cc = S.cat.chips[S.preset], sc = r.scenario;
  const port = c.sram_port_Bpc || cc.port_GBps / cc.freq_ghz;
  const items = [
    `芯片 ${c.name}：${c.rows}×${c.cols}×${c.engines} @ ${c.freq_ghz} GHz，MAC 效率 ${c.mac_eff}`,
    `SRAM ${c.sram_mib} MiB，端口 ${num(port)} B/cycle${c.sram_port_Bpc ? '' : '（默认每核 4·(R+C)·2 × E，0.141）'}`,
    `GEMV 单元 ${c.gemv_macs || cc.gemv} MAC/cycle${c.gemv_macs ? '' : '（默认 = 阵列 MAC / 8）'}；向量通道 ${c.vector_lanes || cc.lanes}`,
    `累加器 ${num(c.acc_kib ?? (c.acc_kib_per_engine ?? cc.acc_kib_per_engine_default) * c.engines)} KiB${c.acc_kib == null && c.acc_kib_per_engine == null ? `（默认 ${cc.acc_kib_per_engine_default} KiB/核 × ${c.engines}）` : ''}（超出的部分和行溢出到 SRAM）`,
    `DRAM 效率 ${sc.mem_eff ?? (S.memInfo ? S.memInfo.efficiency : 0.7)}；预留 1 GiB；暂存区 = max(2 MiB, 2·最大激活)`,
    `三层互连：${sc.d2d_enabled && (sc.package_cards || 1) > 1 ? `封装内 ${sc.package_cards} die 走 D2D ${num(sc.d2d_std === 'custom' ? sc.d2d.GBps : ((S.cat.d2d_standards || []).find((d) => d.id === sc.d2d_std) || {}).per_unit_GBps * sc.d2d_units)} GB/s（${sc.d2d_std === 'custom' ? '自定义' : sc.d2d_std + ' × ' + sc.d2d_units}）、α ${sc.d2d.alpha_us} µs` : 'D2D 关（单片大 die，每卡一个封装）'}；节点内 scale-up ${sc.link.GBps} GB/s、α ${sc.link.alpha_us} µs（${sc.fabric && sc.fabric.enabled ? ({ switch: '交换', ring: '环形', full_mesh: '全互连', torus2d: '2D torus' }[sc.link.topology] || sc.link.topology) : '交换；拓扑只在拓扑感知集合通信开启时生效'}）；${sc.node_cards ? `每节点 ${sc.node_cards} 卡，跨节点网络 ${sc.net.GBps} GB/s、α ${sc.net.alpha_us} µs` : '单节点（跨节点层未用）'}。卡按 TP → SP → DP → PP 编号，通信组按落在同一封装 / 节点内的成员数分层（allreduce / allgather 逐级、all-to-all 各层同时按目的地分摊；PP 交接按流水级边界判断所走层级）`,
    c.slc_mib > 0
      ? `系统级缓存 SLC ${c.slc_mib} MiB @ ${c.slc_GBps} GB/s，策略 ${c.slc_policy === 'lru' ? 'lru（循环访问：片外工作集放得下全部读命中，否则 0 命中——上界 / 下界之间）' : c.slc_policy === 'dip' ? 'dip（防抖插入 DIP / BIP：放得下全部命中，否则循环访问命中 (K − ε(T − K)) / T，ε = 1/32，各类读同比例）' : c.slc_policy === 'pin_bw' ? 'pin_bw（同 pin 钉住，读命中按 SLC / DRAM 带宽均衡分流：两条路径同时读完，SLC 慢于 DRAM 也不变慢）' : 'pin（SRAM 之后按热权重 → 专家 → KV / 状态钉住，流式激活与 KV 写入绕过）'}；不增加容量；文本编码器 / VAE 不用 SLC`
      : '无系统级缓存（SLC MiB = 0）',
    `芯片不支持的权重格式：反量化每元素 2 次向量操作「假设」${Object.entries(S.cat.dequant_ops || {}).map(([f, n]) => `；${f} ${num(n)} 次（码本查表 + 符号 + 子块缩放，按 ggml 参考反量化「推导」）`).join('')}`,
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
  const tail = m.domain === 'gen' ? (mw.image ? `${w.width || mw.width}×${w.height || mw.height} · ${w.steps || mw.steps} 步 · 单张 SLO ${fmtDur(w.clip_slo_s)}`
      : `${w.width || mw.width}×${w.height || mw.height} · ${w.frames || mw.frames} 帧 · ${w.steps || mw.steps} 步 · 单段 SLO ${fmtDur(w.clip_slo_s)}`)
    : m.domain === 'genomics' ? `窗口 ${w.seq_len || mw.seq_len} token · 批延迟 SLO ${w.seq_slo_ms} ms`
    : m.domain === 'protein' ? `${w.seq_len || mw.seq_len} 残基 · 批延迟 SLO ${m.is_pair ? w.fold_slo_s + ' s' : w.seq_slo_ms + ' ms'}` : `ctx ${sv.ctx} · TPOT SLO ${sv.tpot_slo_ms} ms`;
  return `场景：${m.label} · ${cards()} 卡 · 芯片 ${S.preset} · 存储器 ${mi ? `${mi.kind} ${num(mi.raw_GBps)} GB/s ${num(mi.capacity_GiB)} GiB` : S.sc.mem_id} · ${tail}（在左侧面板修改）`;
}
function paintScope() { $('cmp-scope').textContent = scopeText(); $('lay-scope').textContent = scopeText(); }

/* ------------------------------------------------------------------ 0.89: CSV / JSON export of compare / sweep / Pareto */
const EXP = {};
const EXP_SCHEMA = 1;   // 0.128: JSON export layout version (bump on any key / meaning change); booleans are exported as 是/否 labels
function csvCell(v) {
  if (v === null || v === undefined) return '';
  if (typeof v === 'number') return isFinite(v) ? String(v) : '';
  if (typeof v === 'boolean') return v ? '是' : '否';
  let t = String(v);
  if (/^[=+\-@\t\r]/.test(t)) t = "'" + t;            // spreadsheet formula injection guard
  return /[",\r\n]/.test(t) || t !== String(v) ? '"' + t.replace(/"/g, '""') + '"' : t;
}
function expCSV(cols, rows) {
  return '\ufeff' + [cols.map((c) => csvCell(c.label)).join(','), ...rows.map((r) => cols.map((c) => csvCell(c.v(r))).join(','))].join('\r\n') + '\r\n';
}
function expJSON(kind, e) {
  const rows = e.rows();
  return JSON.stringify({ tool: 'accel-dse', schema: EXP_SCHEMA, version: S.version || null, exported_at: new Date().toISOString(), kind, title: e.title,
    scenario: e.scenario, columns: e.cols.map((c) => ({ key: c.key, label: c.label })),
    rows: rows.map((r) => Object.fromEntries(e.cols.map((c) => { const v = c.v(r); return [c.key, v === undefined || (typeof v === 'number' && !isFinite(v)) ? null : v]; }))),
    raw: e.raw }, null, 2);
}
function expName(e, ext) {
  const d = new Date(), z = (n) => String(n).padStart(2, '0');
  const stamp = `${d.getFullYear()}${z(d.getMonth() + 1)}${z(d.getDate())}-${z(d.getHours())}${z(d.getMinutes())}${z(d.getSeconds())}`;
  return `accel-dse_${e.kind}_${String(e.scenario.model || 'model').replace(/[^\w.\-]+/g, '_')}_${stamp}.${ext}`;
}
function download(name, text, type) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
}
function setExport(box, spec) { EXP[box] = spec; $(box + '-exp').hidden = false; }
function clearExport(box) { delete EXP[box]; $(box + '-exp').hidden = true; }
function doExport(box, fmt) {
  const e = EXP[box];
  if (!e) return;
  if (fmt === 'csv') download(expName(e, 'csv'), expCSV(e.cols, e.rows()), 'text/csv;charset=utf-8');
  else download(expName(e, 'json'), expJSON(e.kind, e), 'application/json');
}
const yesNo = (x) => (x === null || x === undefined ? null : x ? '是' : '否');
const snap = (b) => JSON.parse(JSON.stringify(b.scenario));
async function runCompare() {
  paintScope();
  const btn = $('cmp-run');
  $('cmp-status').textContent = '搜索中（5 种映射 × 全部布局）…';
  delete $('cmp-status').dataset.done;
  const t0 = performance.now();
  const v0 = S.ver, m0 = model();
  try {
    const b = body({ cards: cards(), objective: S.cmpObj });
    clearExport('cmp');
    const r = await track(req('compare', '/api/compare', b));
    if (!r) return;
    const mySeq = seq.compare;
    const stab = {};
    paintCompare(r, stab, m0);
    {
      const full = isFull(m0), ux = full ? `${unitZh(m0)}/s/卡` : 'tok/s/卡';
      const tokLat = (x) => x.unit === 'token' || !x.unit;
      const cols = [
        { key: 'mapping', label: '映射组织', v: (x) => x.label }, { key: 'mapping_id', label: '映射 id', v: (x) => x.mapping },
        { key: 'layout', label: '最佳布局', v: (x) => (x.batch ? x.layout : '放不下') }, { key: 'cards', label: '卡数', v: () => b.cards },
        { key: 'batch', label: 'batch', v: (x) => x.batch || null },
        { key: 'thr_card', label: `吞吐（${ux}）`, v: (x) => (x.batch ? x.tok_s_card : null) },
        ...(full ? [] : [{ key: 'goodput_card', label: '有效吞吐 goodput（tok/s/卡）', v: (x) => (x.batch ? x.goodput_card : null) }]),
        full ? { key: 'latency_s', label: m0.domain === 'gen' ? `${clipZh(m0)}（s）` : '批延迟（s）', v: (x) => (x.batch ? x.latency_ms / 1e3 : null) }
          : { key: 'tpot_ms', label: 'TPOT（ms）', v: (x) => (x.batch ? (tokLat(x) ? x.tpot_ms : x.latency_ms) : null) },
        { key: 'bound', label: '瓶颈', v: (x) => (x.batch ? x.bound : null) }, { key: 'bound_zh', label: '瓶颈说明', v: (x) => (x.batch ? BOUND_ZH[x.bound] : null) },
        { key: 'mac_util_pct', label: '有效 MAC 利用率（%）', v: (x) => (x.batch && x.array_util != null ? x.array_util * 100 : null) },
        ...(full ? [] : [{ key: 'ttft_ms', label: 'prefill TTFT（ms）', v: (x) => (x.batch ? x.ttft_ms : null) },
          { key: 'ttft_ok', label: 'TTFT 满足 SLO', v: (x) => (x.batch ? yesNo(x.ttft_ok) : null) }]),
        { key: 'runner_up', label: '次优布局', v: (x) => (x.runner_up ? x.runner_up.layout : null) },
        { key: 'runner_up_score', label: '次优得分（同目标单位）', v: (x) => (x.runner_up ? x.runner_up.score : null) },
        { key: 'stable', label: '排名稳定性', v: (x) => { const st = stab[x.mapping]; return !x.batch || !st ? null : st.error ? '错误' : st.stable ? '稳定' : '不稳定'; } },
        { key: 'stable_agree_pct', label: '稳定性一致率（%）', v: (x) => { const st = stab[x.mapping]; return st && !st.error ? st.agree * 100 : null; } },
      ];
      setExport('cmp', { kind: 'compare', title: `映射对比（目标 ${r.objective}）`, cols, rows: () => r.rows, raw: { compare: r, stability: stab }, scenario: snap(b) });
    }
    const none = r.rows.every((x) => !x.batch);
    $('cmp-status').textContent = none ? `${cards()} 卡下任何布局都放不下或不满足 SLO —— 先在「单点评估」按提示修正卡数${staleTag(v0)}` :
      `完成 · ${((performance.now() - t0) / 1000).toFixed(1)} s · 稳定性计算中（0/${r.rows.length}）…${staleTag(v0)}`;
    // 0.85: the "nothing fits" verdict is a result too — mark it done so a later edit (e.g. more cards) flags it stale
    if (none) { $('cmp-status').dataset.done = '1'; return; }
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
      $('cmp-status').textContent = `完成 · ${((performance.now() - t0) / 1000).toFixed(1)} s · 稳定性计算中（${done}/${r.rows.filter((x) => x.batch).length}）…${staleTag(v0)}`;
      paintCompare(r, stab, m0);
      return true;
    };
    const res = await Promise.all(r.rows.filter((x) => x.batch).map(one));
    if (res.some((x) => !x)) return;
    $('cmp-status').textContent = `完成 · ${((performance.now() - t0) / 1000).toFixed(1)} s${staleTag(v0)}`;
  } catch (e) { $('cmp-status').textContent = '错误：' + e.message; $('cmp-status').dataset.done = '1'; }   // 0.85: stale-flag errors too

}
function latCell(x) { return x.unit === 'token' || !x.unit ? num(x.tpot_ms) : fmtDur(x.latency_ms / 1e3); }
function paintCompare(r, stab, m = model()) {
  const gp = r.objective === 'goodput';
  const full = isFull(m);
  const ux = full ? `${unitZh(m)}/s/卡` : 'tok/s/卡';
  const head = h('tr', {}, h('th', { class: 'l' }, '映射组织'), h('th', { class: 'l' }, '最佳布局 / 次优'), h('th', {}, 'batch'),
    h('th', {}, ux), full ? null : h('th', {}, 'goodput/卡'), h('th', {}, full ? (m.domain === 'gen' ? clipZh(m) : '批延迟') : 'TPOT ms'),
    h('th', { class: 'l' }, '瓶颈 · 有效 MAC'), full ? null : h('th', {}, 'prefill TTFT'), h('th', {}, '排名稳定性'));
  const rows = r.rows.map((x) => {
    const ok = !!x.batch;
    return h('tr', { class: (ok ? 'click ' : '') + (x.mapping === r.best_mapping && ok ? 'best' : ''), title: ok ? '点击把映射、布局（含卡数）和 batch 应用到场景' : '',
      onclick: ok ? () => applyRow(x) : null },
      h('td', { class: 'l' }, x.label),
      h('td', { class: 'l mono' }, ok ? x.layout : '放不下', ' ', budgetMark(x),
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
  const old = btn.dataset.label || (btn.dataset.label = btn.textContent);   // re-clicking restarts the search (no disabled button)
  btn.textContent = '搜索中…';
  const v0 = S.ver;
  try {
    const r = await track(req('best-layout', '/api/layouts', body({ cards: cards(), objective: 'decode' })));
    if (!r) return;
    // 0.78: an edit during the search (model, cards, phase…) — don't overwrite it with a layout for the old scenario
    if (S.ver !== v0) { showErr(`最佳布局：${STALE}（搜索期间场景已修改，结果未应用）`); return; }
    const top = r.rows[0];
    if (!top || !top.batch) { showErr(`${cards()} 卡下没有满足容量和 SLO 的布局`); return; }
    applyLayout(top.layout_obj, top.batch);
  } catch (e) { showErr('布局搜索：' + e.message); }
  finally { btn.textContent = old; }
}

/* ------------------------------------------------------------------ layouts + stability */
async function runLayouts() {
  paintScope();
  $('lay-status').textContent = '搜索中…';
  delete $('lay-status').dataset.done;
  const t0 = performance.now();
  const v0 = S.ver, m0 = model();
  try {
    const r = await track(req('layouts', '/api/layouts', body({ cards: cards(), objective: S.layObj })));
    if (!r) return;
    const hasG = r.rows.some((x) => x.goodput_card !== null && x.goodput_card !== undefined);
    const full = isFull(m0);
    const head = h('tr', {}, h('th', {}, '#'), h('th', { class: 'l' }, '布局'), h('th', {}, 'batch'),
      h('th', {}, full ? `${unitZh(m0)}/s/卡` : 'tok/s/卡'),
      hasG ? h('th', {}, 'goodput/卡') : null, h('th', {}, full ? (m0.domain === 'gen' ? clipZh(m0) : '批延迟') : 'TPOT ms'), h('th', {}, '瓶颈'), h('th', { class: 'l' }, '有效 MAC'),
      hasG ? h('th', {}, 'prefill TTFT') : null);
    const rows = r.rows.map((x, i) => h('tr', { class: (x.batch ? 'click ' : '') + (i === 0 && x.batch ? 'best' : ''), onclick: x.batch ? () => applyRow(x) : null },
      h('td', {}, i + 1), h('td', { class: 'l mono' }, x.layout, ' ', budgetMark(x)), h('td', {}, x.batch || '放不下'), h('td', {}, num(x.tok_s_card)),
      hasG ? h('td', {}, num(x.goodput_card)) : null, h('td', {}, x.batch ? latCell(x) : '—'), h('td', {}, boundTag(x.bound)),
      h('td', { class: 'l' }, h('span', { class: 'ubar' }, h('i', { style: `width:${Math.min(100, (x.array_util || 0) * 100)}%` })), pct(x.array_util)),
      hasG ? h('td', {}, x.ttft_ok === undefined ? '—' : ttftFlag(x.ttft_ok, x.ttft_ms)) : null));
    put($('lay-tbl'), h('thead', {}, head), h('tbody', {}, rows));
    $('lay-status').textContent = `${r.n_layouts} 个布局 · 精确前 ${r.rows.length} 名 · ${((performance.now() - t0) / 1000).toFixed(1)} s${staleTag(v0)}`;
    $('lay-status').dataset.done = '1';
  } catch (e) { $('lay-status').textContent = '错误：' + e.message; $('lay-status').dataset.done = '1'; }   // 0.85: stale-flag errors too
}
async function runStability() {
  paintScope();
  delete $('stab-box').dataset.done;
  put($('stab-box'), h('div', { class: 'stab muted' }, '稳定性计算中（每个扰动重新精确搜索 top-1 布局）…'));
  const v0 = S.ver;
  try {
    const r = await track(req('stab', '/api/stability', body({ cards: cards(), objective: S.layObj, include_mapping: false })));
    if (!r) return;
    put($('stab-box'), h('div', { class: 'stab' },
      h('div', {}, '基准 top-1：', h('b', { class: 'mono' }, r.base_top), '  ', stabFlag(r), '  ', h('span', { class: 'muted' }, r.rule + staleTag(v0))),
      h('ul', {}, r.cases.map((c) => h('li', { style: c.same || c.within5 ? '' : 'color:var(--warn)' }, caseText(c))))));
    $('stab-box').dataset.done = '1';
  } catch (e) { put($('stab-box'), h('div', { class: 'stab' }, '错误：' + e.message)); }
}

/* ------------------------------------------------------------------ sweep + pareto */
const SWEEP_ZH = {
  'serving.batch': 'batch', 'serving.ctx': '上下文 ctx', 'serving.prompt': 'prompt 长度', 'serving.spec_k': '投机 k',
  'chip.sram_mib': 'SRAM MiB', 'chip.sram_port_Bpc': 'SRAM 端口 B/cycle', 'chip.freq_ghz': '频率 GHz', 'chip.mac_eff': 'MAC 效率',
  'chip.gemv_macs': 'GEMV MAC/cycle', mem_eff: 'DRAM 效率', 'link.GBps': '节点内 GB/s', 'link.alpha_us': '节点内 α µs',
  'serving.moe_skew': 'MoE 倾斜', 'chip.slc_mib': 'SLC MiB', 'chip.slc_GBps': 'SLC GB/s', 'd2d.GBps': 'D2D GB/s（自定义）', 'd2d.alpha_us': 'D2D α µs', package_cards: '每封装 die 数（开 D2D）',
  d2d_units: 'D2D 单元数', 'net.GBps': '跨节点 GB/s', 'net.alpha_us': '跨节点 α µs', node_cards: '每节点卡数', 'fabric.oversub': '收敛比 r（需开拓扑感知）', 'fabric.oversub_spine': 'spine 收敛比 r₂（需三层）', 'fabric.pod_nodes': '每 pod 节点数（需三层）', 'fabric.hop_spine_us': '跨 leaf 每步附加 µs（需拓扑感知）', 'fabric.hop_core_us': '跨 pod 每步附加 µs（需三层）',
  'workload.frames': '帧数', 'workload.steps': '去噪步数', 'workload.height': '高 px', 'workload.width': '宽 px',
  'workload.seq_len': '序列长度（残基）', 'workload.msa': 'MSA 行数', 'workload.recycles': '主干遍数',
  'workload.samples': '扩散样本数', 'serving.images': '图像 / 请求', 'serving.image_w': '图像宽 px', 'serving.image_h': '图像高 px',
};
const SWEEP_DOMAIN = { 'serving.moe_skew': 'llm', 'serving.ctx': 'llm', 'serving.prompt': 'llm', 'serving.spec_k': 'llm', 'workload.frames': 'gen',
  'workload.steps': 'gen', 'workload.height': 'gen', 'workload.width': 'gen', 'workload.seq_len': 'seqenc',
  'workload.msa': 'pair', 'workload.recycles': 'pair', 'workload.samples': 'pair',
  'serving.images': 'vlm', 'serving.image_w': 'vlm', 'serving.image_h': 'vlm' };
function fillSweepPaths() {
  const m = model(), d = isFull(m) ? m.domain : 'llm';
  const cur = $('sw-path').value;
  const list = S.cat.sweep_paths.filter((p) => !SWEEP_DOMAIN[p] || SWEEP_DOMAIN[p] === d
    || (SWEEP_DOMAIN[p] === 'pair' && m.is_pair && (p !== 'workload.msa' || m.workload.msa || m.workload.xmsa)
        && (p !== 'workload.samples' || m.workload.diff_steps))
    || (p === 'workload.steps' && m.is_pair && m.workload.diff_steps)
    || (SWEEP_DOMAIN[p] === 'seqenc' && isSeqEnc(m))
    || (SWEEP_DOMAIN[p] === 'vlm' && !!m.vision))
    // 0.130: knobs the model does not read (server answers 400 since 0.129): MoE skew on dense, spec_k without
    // MTP / draft, frames on an image DiT
    .filter((p) => !(p === 'serving.moe_skew' && !m.is_moe) && !(p === 'serving.spec_k' && !(m.mtp_layers || m.draft || S.sc.serving.draft_model))
      && !(p === 'workload.frames' && m.workload && m.workload.image));
  opts($('sw-path'), list.map((p) => [p, SWEEP_ZH[p] || p]), list.includes(cur) ? cur : 'serving.batch');
  if (!list.includes(cur)) $('sw-values').value = SWEEP_DEFAULT[$('sw-path').value] || '';
}
const SWEEP_DEFAULT = {
  'serving.batch': '1,2,4,8,16,32,64', 'serving.ctx': '1024,4096,16384,32768,131072', 'serving.prompt': '512,2048,8192,32768',
  'serving.spec_k': '0,1,2,3,4,6,8', 'chip.sram_mib': '16,32,64,128,256,512', 'chip.sram_port_Bpc': '2048,4096,7616,16384,32768',
  'chip.freq_ghz': '0.6,0.8,1,1.2,1.5', 'chip.mac_eff': '0.5,0.6,0.7,0.8,0.9,1', 'chip.gemv_macs': '1024,4096,12544,50176',
  mem_eff: '0.5,0.6,0.7,0.8,0.9', 'link.GBps': '50,100,200,400,900', 'link.alpha_us': '0,1,3,5,10',
  'serving.moe_skew': '1,1.25,1.5,2,3,4', 'chip.slc_mib': '0,64,256,1024,4096', 'chip.slc_GBps': '500,1000,2000,4000', 'd2d.GBps': '500,1000,2000,4000',
  'd2d.alpha_us': '0,0.5,1,2', package_cards: '1,2,4,8',
  d2d_units: '1,2,4,8', 'net.GBps': '12.5,25,50,100,200', 'net.alpha_us': '2,5,10,20', node_cards: '0,2,4,8', 'fabric.oversub': '1,2,4,8', 'fabric.oversub_spine': '1,2,4,8', 'fabric.pod_nodes': '8,16,32,64', 'fabric.hop_spine_us': '0,0.5,1,2', 'fabric.hop_core_us': '0,0.5,1,2',
  'workload.frames': '17,33,49,81,121', 'workload.steps': '10,20,30,50', 'workload.height': '240,480,720',
  'workload.width': '416,832,1280', 'workload.seq_len': '128,256,512,1022,2048',
  'workload.msa': '64,256,512,1024,4096', 'workload.recycles': '1,2,3,4,6', 'workload.samples': '1,5,10,25',
  'serving.images': '0,1,2,4,8', 'serving.image_w': '448,640,1024,1536,2048', 'serving.image_h': '448,640,1024,1536,2048',
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
  delete $('sw-status').dataset.done;
  const v0 = S.ver, m = model(), decode = S.sc.serving.phase === 'decode';
  try {
    clearExport('sw'); $('sw-sens-box').hidden = true;
    const sens = $('sw-sens').checked;
    const bs = body({ path, values: vals, ...(sens ? { sensitivity: true } : {}) });
    const r = await track(req('sweep', '/api/sweep', bs));
    if (!r) return;
    const full = isFull(m), gen = m.domain === 'gen';
    const u = full ? unitZh(m) : 'tok';
    const latL = full ? (gen ? `${clipZh(m)} s` : '批延迟 ms') : decode ? 'TPOT ms' : 'TTFT ms';
    const lat = (x) => full ? (gen ? x.latency_ms / 1e3 : x.latency_ms) : decode ? x.tpot_ms : x.ttft_ms;
    put($('sw-chart'), line(r.rows.map((x) => ({ x: x.value, y: x.tok_s_card, y2: lat(x) })),
      { xl: SWEEP_ZH[path] || path, yl: `${u}/s/卡`, y2l: latL, log: vals.every((v) => v > 0) && Math.max(...vals) / Math.min(...vals) >= 16 }));
    const head = h('tr', {}, h('th', { class: 'l' }, SWEEP_ZH[path] || path), h('th', {}, latL),
      h('th', {}, `${u}/s`), h('th', {}, `${u}/s/卡`), h('th', {}, '瓶颈'), h('th', {}, '有效 MAC'), h('th', {}, '放得下'),
      ...(r.sensitivity ? [h('th', { class: 'l' }, '敏感度（排序 / 瓶颈翻转）')] : []));
    const fl = r.sensitivity ? r.sensitivity.row_flags : null;
    put($('sw-tbl'), h('thead', {}, head), h('tbody', {}, r.rows.map((x, i) => h('tr', { class: fl && fl[i].length ? 'warn' : '' },
      h('td', { class: 'l' }, x.value), h('td', {}, num(lat(x))), h('td', {}, num(x.tok_s)),
      h('td', {}, num(x.tok_s_card)), h('td', {}, boundTag(x.bound)), h('td', {}, pct(x.array_util)),
      h('td', {}, x.fits ? '✓' : h('span', { style: 'color:var(--danger)' }, '✗')),
      ...(fl ? [h('td', { class: 'l small' }, fl[i].length ? '⚠ ' + fl[i].join('、') : '稳健')] : [])))));
    renderSweepSens(r.sensitivity, SWEEP_ZH[path] || path, u);
    const latK = full ? (gen ? `${clipZh(m)}（s）` : '批延迟（ms）') : decode ? 'TPOT（ms）' : 'TTFT（ms）';
    setExport('sw', { kind: 'sweep', title: `参数扫描：${SWEEP_ZH[path] || path}`, raw: r, scenario: snap(bs), rows: () => r.rows, cols: [
      { key: 'path', label: '扫描参数', v: () => path }, { key: 'param', label: SWEEP_ZH[path] || path, v: (x) => x.value },
      { key: 'latency', label: latK, v: lat }, { key: 'thr', label: `吞吐（${u}/s）`, v: (x) => x.tok_s }, { key: 'thr_card', label: `吞吐（${u}/s/卡）`, v: (x) => x.tok_s_card },
      { key: 'bound', label: '瓶颈', v: (x) => x.bound }, { key: 'mac_util_pct', label: '有效 MAC 利用率（%）', v: (x) => (x.array_util != null ? x.array_util * 100 : null) },
      { key: 'fits', label: '放得下', v: (x) => yesNo(x.fits) },
      ...(fl ? [{ key: 'sens_flags', label: '敏感度翻转（假设）', v: (x) => fl[r.rows.indexOf(x)].join('；') }] : [])] });
    $('sw-status').textContent = `${r.rows.length} 点${staleTag(v0)}`;
    $('sw-status').dataset.done = '1';
  } catch (e) { $('sw-status').textContent = '错误：' + e.message; $('sw-status').dataset.done = '1'; }   // 0.85: stale-flag errors too
}
async function runPareto() {
  $('sw-status').textContent = '计算 Pareto…';
  delete $('sw-status').dataset.done;
  const v0 = S.ver, m = model();
  try {
    clearExport('sw'); $('sw-sens-box').hidden = true;
    const bp = body($('sw-sens').checked ? { sensitivity: true } : {});
    const r = await track(req('sweep', '/api/pareto', bp));
    if (!r) return;
    const full = isFull(m), gen = m.domain === 'gen';
    const u = full ? unitZh(m) : 'tok';
    const latL = full ? (gen ? `${clipZh(m)} s` : '批延迟 ms') : 'TPOT ms';
    const lat = (p) => full ? (gen ? p.latency_ms / 1e3 : p.latency_ms) : p.tpot_ms;
    const thr = (p) => full ? p.units_s_card : p.tok_s_card;
    const pts = r.front.map((p) => ({ x: p.batch, xlabel: 'B' + p.batch, y: thr(p), y2: lat(p) }));
    put($('sw-chart'), line(pts, { xl: `batch（布局 ${r.layout}，Pareto 前沿：延迟 ↑ 换吞吐 ↑）`, yl: `${u}/s/卡`, y2l: latL, log: true }));
    const pf = r.sensitivity ? r.sensitivity.point_flags : null;
    const head = h('tr', {}, h('th', { class: 'l' }, 'batch'), h('th', {}, latL), h('th', {}, `${u}/s/卡`),
      ...(pf ? [h('th', { class: 'l' }, '敏感度（在该假设下被支配、离开前沿）')] : []));
    put($('sw-tbl'), h('thead', {}, head), h('tbody', {}, r.front.map((p, i) => h('tr', { class: pf && pf[i].length ? 'warn' : '' },
      h('td', { class: 'l' }, p.batch), h('td', {}, num(lat(p))), h('td', {}, num(thr(p))),
      ...(pf ? [h('td', { class: 'l small' }, pf[i].length ? '⚠ ' + pf[i].join('、') : '稳健')] : [])))));
    renderParetoSens(r.sensitivity);
    setExport('sw', { kind: 'pareto', title: `Pareto 前沿（布局 ${r.layout}）`, raw: r, scenario: snap(bp), rows: () => r.front, cols: [
      { key: 'layout', label: '布局', v: () => r.layout }, { key: 'batch', label: 'batch', v: (p) => p.batch },
      { key: 'latency', label: full ? (gen ? `${clipZh(m)}（s）` : '批延迟（ms）') : 'TPOT（ms）', v: lat },
      { key: 'thr_card', label: `吞吐（${u}/s/卡）`, v: thr },
      ...(pf ? [{ key: 'sens_flags', label: '敏感度：离开前沿（假设）', v: (p) => pf[r.front.indexOf(p)].join('；') }] : [])] });
    $('sw-status').textContent = `Pareto ${r.front.length} 点${staleTag(v0)}`;
    $('sw-status').dataset.done = '1';
  } catch (e) { $('sw-status').textContent = '错误：' + e.message; $('sw-status').dataset.done = '1'; }   // 0.85: stale-flag errors too
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
      const approx = [...(m.coverage_reasons || []), ...(m.vision_params_B && !m.vision ? [`视觉编码器（${num(m.vision_params_B)}B）未建模`] : [])];
      rows.push(h('tr', { class: 'click', 'data-id': m.id, 'data-domain': m.domain, onclick: () => { setModel(m.id); activate('eval'); schedule(); } },
        h('td', { class: 'l' }, m.label, m.domain === 'vlm' ? h('div', {}, h('span', { class: 'badge vlm', title: coverTip(m) }, m.vision ? 'VLM · 含视觉编码器' : 'VLM · 视觉编码器未建模'))
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
    + ` · 视频生成 ${cnt((m) => m.evaluable && m.domain === 'gen')} · 蛋白质 ${cnt((m) => m.evaluable && m.domain === 'protein')} · 基因组编码器 ${cnt((m) => m.evaluable && m.domain === 'genomics')}）；`,
    `暂未接入 v2（灰色行，不能评估）${cnt((m) => !m.evaluable)} 个（视频生成 ${cnt((m) => !m.evaluable && m.domain === 'gen')} · 蛋白质 ${cnt((m) => !m.evaluable && m.domain === 'protein')}）。`,
    '分子动力学 / 机器学习力场（MLFF）目录中暂无条目。');
  put($('models-tbl'), h('thead', {}, head), h('tbody', {}, rows));
  put($('models-unlisted'), ...S.unlisted.map((u) => h('li', {}, h('b', {}, u.label), '：', u.reason)));
}


/* ------------------------------------------------------------------ 0.146 design insight */
const INS_COST_ZH = { die: '片上（面积代理）', offdie: '片外 / PHY（不在面积代理内）', none: '无面积项' };
const insX = (x) => (x === null || x === undefined ? '—' : '×' + x.toFixed(3));
async function runInsight() {
  const btn = $('ins-run');
  const v0 = S.ver;
  $('ins-status').textContent = '计算中（每个模型 × 每个旋钮 × prefill + decode）…';
  delete $('ins-status').dataset.done;
  try {
    const ms = $('ins-models').value.split(',').map((x) => x.trim()).filter(Boolean);
    const r = await track(req('insight', '/api/insight', body(ms.length ? { models: ms } : {})));
    if (!r) return;
    S.insight = r;
    S.insMetric = r.default_metric || S.insMetric;
    paintInsMetric(r);
    paintInsight();
    setInsightExport();
    const st = r.slc_temp, tip = (st && st.recommend && st.recommend.some((x) => x.hot_cuts_J_opt))
      ? ' · SLC 85 °C 削减 J 最优' : (st ? ' · 含 SLC 25/85 °C' : '');
    $('ins-status').textContent = `完成 · ${r.domain} · ${r.chip} · ${r.mem_id} · ${r.layout} · batch ${r.serving.batch}${tip}${staleTag(v0)}`;
    $('ins-status').dataset.done = '1';
  } catch (e) {
    $('ins-status').textContent = '错误：' + e.message;
    $('ins-status').dataset.done = '1';
  }
}
/* 0.149: cited density presets per process node — fill the budget densities (still editable), sources on hover */
function initDensityPresets() {
  const sel = $('b-preset'), P = S.cat.density_presets || {};
  for (const [k, v] of Object.entries(P)) sel.appendChild(h('option', { value: k }, v.label));
  sel.addEventListener('change', () => {
    const p = P[sel.value];
    const b = { ...(S.budget || {}) };
    S.presetFilled = S.presetFilled || {};
    if (!p) {                                   // 0.150: back to 「不用预设」 → drop the values a preset filled (unchanged)
      for (const [f, v] of Object.entries(S.presetFilled)) if (b[f] === v) { delete b[f]; const inp = $('b-' + f); if (inp) inp.title = ''; }
      S.presetFilled = {};
      S.budget = b; syncInputs(); $('b-preset-note').textContent = ''; schedule(); return;
    }
    const kept = [], st = {};
    for (const [f, x] of Object.entries(p.fields)) {
      // 0.150: a value the user typed wins — fill only empty fields or ones a preset filled earlier (unchanged since)
      const cur = b[f], mine = cur !== null && cur !== undefined && S.presetFilled[f] !== cur;
      const inp = $('b-' + f);
      if (mine) { kept.push(f); if (inp) inp.title = '你的输入（预设未覆盖）'; continue; }
      b[f] = +x.value.toPrecision(4); S.presetFilled[f] = b[f];
      st[x.status] = (st[x.status] || 0) + 1;
      if (inp) inp.title = `「${x.status}」${x.source}`;
    }
    S.budget = b;
    syncInputs();
    $('b-preset-note').textContent = '已填入：' + Object.entries(st).map(([k, n]) => `${n} 项「${k}」`).join('，') +
      (kept.length ? `；保留你的输入：${kept.join('、')}` : '') + '（悬停各输入框看出处）';
    schedule();
  });
}
/* 0.151: metric buttons follow the report domain; export uses the shared EXP_SCHEMA envelope */
function paintInsMetric(r) {
  const box = $('ins-metric'); if (!box || !r || !r.metrics) return;
  const keys = Object.keys(r.metrics);
  const cur = keys.includes(S.insMetric) ? S.insMetric : (r.default_metric || keys[0]);
  S.insMetric = cur;
  put(box, ...keys.map((k) => {
    const b = h('button', { 'data-v': k }, '按 ' + r.metrics[k].replace(/加速$/, ''));
    if (k === cur) b.classList.add('on');
    b.addEventListener('click', () => { S.insMetric = k; paintInsMetric(r); paintInsight(); });
    return b;
  }));
}
function setInsightExport() {
  const r = S.insight; if (!r) { clearExport('ins'); return; }
  const m = S.insMetric || r.default_metric || 'tpot';
  const lab = (k) => (r.knobs.find((x) => x.key === k) || {}).label || k;
  setExport('ins', {
    kind: 'insight', title: `设计洞察 · ${r.domain} · ${r.metrics[m] || m}`,
    scenario: { chip: r.chip, mem_id: r.mem_id, layout: r.layout, serving: r.serving, models: r.models.map((x) => x.model), domain: r.domain, metric: m },
    cols: [
      { key: 'rank', label: '排名', v: (row) => row._rank },
      { key: 'key', label: '旋钮', v: (row) => row.key },
      { key: 'label', label: '说明', v: (row) => row.label },
      { key: 'score', label: r.metrics[m] || m, v: (row) => row.score[m] },
      { key: 'd_mm2', label: 'Δmm²', v: (row) => row.d_mm2 },
      { key: 'gain_per_mm2', label: '增益/mm²', v: (row) => row.gain_per_mm2[m] },
      { key: 'cost', label: '成本类', v: (row) => row.cost },
      { key: 'inverted', label: '取反', v: (row) => yesNo(row.inverted) },
    ],
    rows: () => r.knobs.slice().sort((a, b) => (b.score[m] || 0) - (a.score[m] || 0)).map((k, i) => ({ ...k, _rank: i + 1 })),
    raw: r,
  });
}
function setInsightPathExport() {
  const r = S.insPath; if (!r) { clearExport('ins-path'); return; }
  setExport('ins-path', {
    kind: 'insight_path', title: `升级路径 · ${r.domain} · ${r.metrics[r.metric] || r.metric}`,
    scenario: { models: Object.keys(r.start), domain: r.domain, metric: r.metric, cost_mode: r.cost_mode, search: r.search, beam: r.beam },
    cols: [
      { key: 'step', label: '步', v: (p) => p.step },
      { key: 'key', label: '旋钮', v: (p) => p.key },
      { key: 'label', label: '说明', v: (p) => p.label },
      { key: 'gain', label: '本步增益', v: (p) => p.gain },
      { key: 'cum_gain', label: '累计增益', v: (p) => p.cum_gain },
      { key: 'cost', label: '本步成本', v: (p) => p.cost },
      { key: 'cost_basis', label: '成本依据', v: (p) => p.cost_basis },
      { key: 'cum_cost', label: '累计成本', v: (p) => p.cum_cost },
      { key: 'gain_per_cost', label: '增益/成本', v: (p) => p.gain_per_cost },
    ],
    rows: () => r.path,
    raw: r,
  });
}
/* 0.152: cited energy presets — fill unset energy fields (idle_W never preset); same precedence as density */
function initEnergyPresets() {
  const sel = $('e-preset'), P = S.cat.energy_presets || {};
  if (!sel) return;
  for (const [k, v] of Object.entries(P)) sel.appendChild(h('option', { value: k }, v.label));
  sel.addEventListener('change', () => {
    const p = P[sel.value];
    const e = { ...(S.energy || {}) };
    S.energyPresetFilled = S.energyPresetFilled || {};
    if (!p) {
      for (const [f, v] of Object.entries(S.energyPresetFilled)) if (e[f] === v) { delete e[f]; const inp = $('e-' + f); if (inp) inp.title = ''; }
      S.energyPresetFilled = {};
      S.energy = e; syncInputs(); $('e-preset-note').textContent = ''; schedule(); return;
    }
    const kept = [], st = {};
    for (const [f, x] of Object.entries(p.fields)) {
      const cur = e[f], mine = cur !== null && cur !== undefined && S.energyPresetFilled[f] !== cur;
      const inp = $('e-' + f);
      if (f === 'pJ_bit_mm_onchip' && e.slc_access_ref_mib != null) { kept.push(f + '（已用 √C 模型）'); continue; }
      if (mine) { kept.push(f); if (inp) inp.title = '你的输入（预设未覆盖）'; continue; }
      e[f] = +x.value.toPrecision(4); S.energyPresetFilled[f] = e[f];
      st[x.status] = (st[x.status] || 0) + 1;
      if (inp) inp.title = `「${x.status}」${x.source}`;
    }
    S.energy = e; syncInputs();
    $('e-preset-note').textContent = '已填入：' + Object.entries(st).map(([k, n]) => `${n} 项「${k}」`).join('，') +
      (kept.length ? `；保留你的输入：${kept.join('、')}` : '') + '（idle_W 不进预设；悬停看出处）';
    schedule();
  });
}
/* 0.153: labelled electricity-rate presets — fill usd_per_kWh only if unset; never a silent default */
function initElectricityPresets(selId = 'b-elec-preset', catKey = 'electricity_presets', fillKey = 'elecPresetFilled') {
  const sel = $(selId), P = S.cat[catKey] || {}, noteId = selId + '-note';
  if (!sel) return;
  for (const [k, v] of Object.entries(P)) sel.appendChild(h('option', { value: k }, v.label));
  sel.addEventListener('change', () => {
    const p = P[sel.value];
    const b = { ...(S.budget || {}) };
    S[fillKey] = S[fillKey] || {};
    if (!p) {
      for (const [f, v] of Object.entries(S[fillKey])) if (b[f] === v) { delete b[f]; const inp = $('b-' + f); if (inp) inp.title = ''; }
      S[fillKey] = {};
      S.budget = b; syncInputs(); $(noteId).textContent = ''; schedule(); return;
    }
    const kept = [], st = {};
    for (const [f, x] of Object.entries(p.fields)) {
      const cur = b[f], mine = cur !== null && cur !== undefined && S[fillKey][f] !== cur;
      const inp = $('b-' + f);
      if (mine) { kept.push(f); if (inp) inp.title = '你的输入（预设未覆盖）'; continue; }
      b[f] = +x.value.toPrecision(4); S[fillKey][f] = b[f];
      st[x.status] = (st[x.status] || 0) + 1;
      if (inp) inp.title = `「${x.status}」${x.source}`;
    }
    S.budget = b; syncInputs();
    $(noteId).textContent = '已填入：' + Object.entries(st).map(([k, n]) => `${n} 项「${k}」`).join('，') +
      (kept.length ? `；保留你的输入：${kept.join('、')}` : '') + (catKey === 'tco_presets' ? '（卡价 / 摊销需自填；悬停看出处）' : '（不静默填电价；悬停看出处）');
    schedule();
  });
}

/* 0.187: worst model's DRAM-byte and energy ratio at min_ok (energy only with an energy table) */
function trafficCell(t) {
  const v = Object.values(t || {});
  if (!v.length) return '—';
  const mx = (k) => { const xs = v.map((e) => e[k]).filter((x) => x != null); return xs.length ? Math.max(...xs) : null; };
  const d = mx('dram_B'), e = mx('J'), ep = mx('J_prefill');
  return `DRAM ×${d != null ? num(d) : '—'}` + (e != null ? ` · 能耗 ×${num(e)}${v.some((x) => x.J_partial) ? '（部分）' : ''}` : '')
    + (ep != null ? ` · prefill ×${num(ep)}` : '');   // 0.188
}

/* 0.275: speech bottleneck panel (/api/insight_speech, core/speech_insight.py) */
const SP_STAGE = { lm: 'LM', speech_card: '语音卡', thinker: 'Thinker', frame: '帧（串行）', encoder: '编码器', pipeline: 'S2ST 全流程（串行）' };   // 0.280 pipeline
function speechInsightRows(r) {
  return (r.rows || []).map((x) => {
    const st = Object.entries(x.stages || {}).map(([k, v]) => `${SP_STAGE[k] || k} ${num(v)}`).join(' · ');
    const unlock = x.realtime ? '—' : (x.slc_GB_needed != null
      ? `${num(x.slc_GB_needed)} GB` + (x.verify ? `（复算 x ${num(x.verify.x)}${x.verify.realtime ? ' ✓' : ' ✗'}）` : '')
      : (x.unlock_note || '—'));
    return h('tr', { style: x.realtime ? '' : 'color:var(--warn)' }, h('td', {}, x.model), h('td', {}, x.kind),
      h('td', {}, SP_STAGE[x.binding] || x.binding), h('td', {}, num(x.x)), h('td', {}, x.realtime ? '是' : '否'),
      h('td', {}, `${num(x.first_audio_s * 1e3)} ms（${x.first_audio_dominant}）`),
      h('td', {}, x.dram_GBps_eff_needed != null ? num(x.dram_GBps_eff_needed) : '—'), h('td', {}, unlock),
      h('td', { class: 'small muted' }, st), ...(r.rows.some((y) => y.dispatch) ? [h('td', { class: 'small' }, speechDispCell(x.dispatch))] : []));
  });
}
/* 0.283: per-op dispatch sensitivity — x(u) = x(0) + n·u (core/speech_insight._dispatch_row) */
function speechDispCell(d) {
  if (!d) return '—';
  const ops = Object.values(d.ops_per_audio_s || {}).reduce((a, b) => Math.max(a, b), 0);
  const rt = d.us_rt == null ? '∞' : (d.us_rt === 0 ? '理想下已不实时' : `${num(d.us_rt)} µs/op`);
  return `实时临界 ${rt} · 调度主导 > ${d.us_eq == null ? '—' : num(d.us_eq) + ' µs/op'} · ${num(ops)} op/音频秒`
    + ` · x@trace ${num(d.x_at.trace)} / host ${num(d.x_at.host)} · 首音频 +${num(d.first_audio_ms_per_us)} ms/(µs/op)`;
}
async function runInsightSpeech() {
  const v0 = S.ver;      // 0.285: an edit during the run tags the result stale (invalidate tags it afterwards)
  $('ins-sp-status').textContent = '计算中（每个语音模型 batch 1）…';
  delete $('ins-sp-status').dataset.done;
  try {
    const a = Number($('ins-sp-audio').value) || 10;
    const r = await track(req('insight_speech', '/api/insight_speech', body({ audio_s: a, verify: $('ins-sp-verify').checked, dispatch: $('ins-sp-disp').checked })));
    if (!r) return;
    const miss = (r.rows || []).filter((x) => !x.realtime);
    $('ins-sp-summary').textContent = `${r.chip} · ${r.mem_id} · 有效 DRAM ${num(r.dram_GBps_eff)} GB/s · batch ${r.batch} · ${num(r.audio_s)} s 音频`
      + ` · ${miss.length}/${(r.rows || []).length} 个模型达不到实时` + (r.method ? ' · ' + r.method : '');
    put($('ins-sp-tbl'), h('thead', {}, h('tr', {},
      ['模型', '类型', '瓶颈阶段', 'x（忙碌秒/音频秒）', '实时', '首音频 / 首 token（主导）', '所需有效带宽 GB/s', '或 SLC 钉住', '各阶段 x',
        ...((r.rows || []).some((y) => y.dispatch) ? ['op 调度敏感度（每 op 延迟 u）'] : [])]
        .map((x) => h('th', {}, x)))), h('tbody', {}, speechInsightRows(r)));
    $('ins-sp-status').textContent = `完成 · ${(r.rows || []).length} 模型${staleTag(v0)}`;
    $('ins-sp-status').dataset.done = '1';
  } catch (e) {
    $('ins-sp-status').textContent = '错误：' + e.message;
    $('ins-sp-status').dataset.done = '1';
  }
}

/* 0.284: LLM decode per-op dispatch sensitivity (/api/insight_dispatch, core/dispatch_insight.py) */
async function runInsightDispatch() {
  const v0 = S.ver;
  $('ins-dp-status').textContent = '计算中…';
  delete $('ins-dp-status').dataset.done;
  try {
    const models = $('ins-dp-models').value.split(/[\s,]+/).filter(Boolean);
    const batches = $('ins-dp-batches').value.split(/[\s,]+/).filter(Boolean).map(Number);
    const r = await track(req('insight_dispatch', '/api/insight_dispatch', body({ models, batches })));
    if (!r) return;
    $('ins-dp-summary').textContent = `${r.chip} · ${r.mem_id} · ctx ${r.ctx} · 芯片当前派发 ${r.chip_dispatch} · ${r.method}`;
    const rows = (r.rows || []).map((x) => h('tr', { style: x.us_10 != null && x.us_10 < r.us_at.host ? 'color:var(--warn)' : '' },
      h('td', {}, x.model), h('td', {}, String(x.batch)), h('td', {}, num(x.tpot_ms)), h('td', {}, x.bound + (x.fits ? '' : ' ✗放不下')),
      h('td', {}, num(x.ops_per_step)), h('td', {}, num(x.ops_per_token)), h('td', {}, x.us_10 != null ? num(x.us_10) : '—'),
      h('td', {}, x.us_eq != null ? num(x.us_eq) : '—'),
      ...['trace', 'static', 'host'].map((m) => h('td', {}, `${num(x.tpot_ms_at[m])}（×${num(x.slowdown_at[m])}）`))));
    put($('ins-dp-tbl'), h('thead', {}, h('tr', {}, ['模型', 'batch', 'TPOT ms（理想）', '瓶颈', 'op / 步', 'op / token',
      'TPOT +10 % 的 µs/op', '派发 = 其余时间的 µs/op', `trace ${r.us_at.trace} µs`, `static ${r.us_at.static} µs/步`, `host ${r.us_at.host} µs`]
      .map((x) => h('th', {}, x)))), h('tbody', {}, ...rows));
    $('ins-dp-status').textContent = `完成 · ${(r.rows || []).length} 行${staleTag(v0)}`;
    $('ins-dp-status').dataset.done = '1';
  } catch (e) {
    $('ins-dp-status').textContent = '错误：' + e.message;
    $('ins-dp-status').dataset.done = '1';
  }
}

/* 0.276: real-time streams per card vs SLC size (/api/insight_speech_streams) */
const SPS_OUT = ['moshiko-7b', 'kimi-audio-7b-instruct', 'step-audio-2-mini', 'qwen2.5-omni-7b', 'cosyvoice2-0.5b',
  'fun-cosyvoice3-0.5b-2512', 'fireredtts3-base', 'drama-3', 'ming-3.0-tts-tiny', 'index-echo-s2st-2b', 'index-echo-s2st-9b'];
function speechStreamsCell(c) {
  const lim = { realtime: '实时', memory: '容量', cap: '上限' }[c.lm_limit] || c.lm_limit;
  const tip = `LM 卡 ${c.streams_lm} 路（线性预测 ${c.streams_lm_pred}，止于${lim}）`
    + (c.streams_speech_card != null ? ` · 语音卡 ${c.streams_speech_card} 路/卡 × ${c.speech_cards ?? '—'} 张` : '') + ` · 共 ${c.cards} 张卡`;
  return h('td', { title: tip, style: c.streams ? '' : 'color:var(--warn)' },
    c.streams ? `${num(c.streams_per_card)}${c.capped ? '+' : ''}` : '0');
}
// 0.277: one request per model (rows render as they land; the slow lane stays interleavable between models); a new
// click supersedes the run (req drops the old reply, the static build frees its worker); a scenario edit marks it stale
const SPS_ASR = ['qwen2-audio-7b-instruct', 'qwen3-asr-1.7b', 'fun-asr-nano-2512', 'whisper-large-v3',
  'whisper-large-v3-turbo', 'ming-3.0-asr-tiny'];
let spsRun = 0;
async function runInsightSpeechStreams() {
  const run = ++spsRun, v0 = S.ver, t0 = performance.now();
  const gib = $('ins-sps-gib').value.split(',').map((x) => Number(x.trim())).filter((x) => Number.isFinite(x) && x >= 0);
  const slc = gib.length ? gib : [0, 1, 2, 4, 8];
  const models = $('ins-sps-asr').checked ? SPS_OUT.concat(SPS_ASR) : SPS_OUT;
  const st = $('ins-sps-status');
  st.textContent = `计算中（0/${models.length} 模型 × ${slc.length} 档，逐点重算）…`;
  delete st.dataset.done;
  const tb = h('tbody', {});
  const b0 = body({ slc_gib: slc });     // 0.285: one scenario snapshot for every row (an edit mid-run must not mix chips)
  put($('ins-sps-tbl'), h('thead', {}, h('tr', {}, h('th', {}, '模型'), h('th', {}, '类型'),
    ...slc.map((g) => h('th', {}, `SLC ${num(g)} GiB`)), h('th', {}, '首次实时'))), tb);
  try {
    for (let i = 0; i < models.length; i++) {
      const r = await track(req('insight_speech_streams', '/api/insight_speech_streams', { ...b0, models: [models[i]] }));
      if (!r || run !== spsRun) return;          // superseded (new click / cancelled): the newer run owns the panel
      if (i === 0) $('ins-sps-summary').textContent = `${r.chip} · ${r.mem_id} · 单元格 = 实时路数 / 卡（LM 卡 + 所需语音卡；悬停看 LM / 语音卡路数）`
        + ` · 上限 ${r.max_streams} 路（+）· ${r.method}`;
      for (const x of r.rows || []) tb.appendChild(h('tr', {}, h('td', {}, x.model), h('td', {}, x.kind),
        ...x.curve.map(speechStreamsCell),
        h('td', {}, x.first_realtime_slc_gib != null ? `${num(x.first_realtime_slc_gib)} GiB` : '—')));
      st.textContent = `${i + 1 < models.length ? '计算中' : '完成'} · ${i + 1}/${models.length} 模型 × ${slc.length} 档 · `
        + `${((performance.now() - t0) / 1000).toFixed(1)} s${staleTag(v0)}`;
    }
    st.dataset.done = '1';
  } catch (e) {
    if (run === spsRun) { st.textContent = '错误：' + e.message; st.dataset.done = '1'; }
  }
}

/* 0.262/0.263: J/token-optimal SLC size under energy.sram_temp_C; optional 25/85 °C compare */
async function runInsightSlc(cmp) {
  const v0 = S.ver;
  $('ins-slc-status').textContent = cmp ? '计算中（25 °C + 85 °C × 逐 GiB）…' : '计算中（逐 GiB × 每个模型）…';
  delete $('ins-slc-status').dataset.done;
  try {
    const ms = $('ins-models').value.split(',').map((x) => x.trim()).filter(Boolean);
    const payload = { ...(ms.length ? { models: ms } : { models: [S.sc.model] }), policy: $('ins-slc-pol').value };
    if (cmp) payload.temps = [25, 85];
    const r = await track(req('insight_slc', '/api/insight_slc', body(payload)));
    if (!r) return;
    if (r.compare) {
      const notes = (r.notes || []).join('；');
      $('ins-slc-summary').textContent = `策略 ${r.policy} · 漏电计入 ${r.leak_charged ? '是' : '否'} · 结温对比 ${r.temps.join('/')} °C`
        + (notes ? ' · ' + notes : '');
      put($('ins-slc-tbl'), h('thead', {}, h('tr', {},
        ['结温 °C', '模型', 'J 最优 GiB', 'J/单位', '时间最优 GiB', 'J 交叉 GiB'].map((x) => h('th', {}, x)))),
        h('tbody', {}, (r.summary || []).map((row) => row.error
          ? h('tr', {}, h('td', {}, num(row.temp_C)), h('td', {}, row.model), h('td', { colspan: 4 }, row.error))
          : h('tr', {}, h('td', {}, num(row.temp_C)), h('td', {}, row.model),
              h('td', {}, String(row.opt_J_gib)), h('td', {}, sci(row.opt_J)),
              h('td', {}, String(row.opt_time_gib)),
              h('td', {}, row.crossover_gib != null ? String(row.crossover_gib) : '无')))));
      $('ins-slc-status').textContent = `完成 · 对比 · ${r.domain} · ${r.chip}${staleTag(v0)}`;
      $('ins-slc-status').dataset.done = '1';
      return;
    }
    const m0 = (r.models || []).find((x) => !x.error) || r.models[0];
    const notes = (r.notes || []).join('；');
    $('ins-slc-summary').textContent = (!m0 || m0.error) ? ('错误：' + ((m0 && m0.error) || '无结果'))
      : (`结温 ${num(r.sram_temp_C)} °C · 策略 ${r.policy} · 漏电计入 ${r.leak_charged ? '是' : '否'} · `
        + `J 最优 ${m0.opt_J_gib} GiB（${sci(m0.opt_J)} J）· 时间最优 ${m0.opt_time_gib} GiB`
        + (m0.crossover_gib != null ? ` · J 交叉（>无 SLC）自 ${m0.crossover_gib} GiB` : ' · 无 J 交叉')
        + (notes ? ' · ' + notes : ''));
    const rows = (m0 && m0.curve) ? m0.curve : [];
    const timeH = r.domain === 'llm' ? 'TPOT ms' : '延迟 s';
    put($('ins-slc-tbl'), h('thead', {}, h('tr', {},
      ['GiB', 'J/单位', timeH, '漏电 W', 'SLC 命中', '放得下'].map((x) => h('th', {}, x)))),
      h('tbody', {}, rows.map((c) => {
        if (c.error) return h('tr', {}, h('td', {}, String(c.gib)), h('td', { colspan: 5 }, c.error));
        const mark = (c.gib === m0.opt_J_gib ? ' ←J' : '') + (c.gib === m0.opt_time_gib ? ' ←时' : '')
          + (m0.crossover_gib != null && c.gib === m0.crossover_gib ? ' ←交叉' : '');
        return h('tr', {}, h('td', {}, String(c.gib) + mark), h('td', {}, sci(c.J_tok)),
          h('td', {}, r.domain === 'llm' ? num(c.tpot_ms) : num(c.lat_s)),
          h('td', {}, c.leak_W != null ? num(c.leak_W) : '—'),
          h('td', {}, c.slc_frac != null ? num(c.slc_frac * 100) + '%' : '—'),
          h('td', {}, c.fits ? '是' : '否'));
      })));
    $('ins-slc-status').textContent = `完成 · ${r.domain} · ${r.chip} · ${(r.models || []).length} 模型${staleTag(v0)}`;
    $('ins-slc-status').dataset.done = '1';
  } catch (e) {
    $('ins-slc-status').textContent = '错误：' + e.message;
    $('ins-slc-status').dataset.done = '1';
  }
}

/* 0.299: NoC / on-chip movement energy vs engine count × topology (/api/insight_noc, core/noc_insight.py) */
async function runInsightNoc() {
  const v0 = S.ver;
  const sci = (x) => (x == null ? '—' : x === 0 ? '0' : x.toExponential(3));
  $('ins-noc-status').textContent = '计算中（引擎切分 × 拓扑 × decode/prefill）…';
  delete $('ins-noc-status').dataset.done;
  try {
    const pr = parseInt($('ins-noc-prompt').value, 10);
    const lk = $('ins-noc-link').value === '' ? null : +$('ins-noc-link').value;   // 0.303: NoC link B/cycle
    const r = await track(req('insight_noc', '/api/insight_noc', body({ ...(pr > 0 ? { prompt: pr } : {}), ...(lk != null && lk >= 0 ? { noc_link_Bpc: lk } : {}), ...($('ins-noc-routing').value ? { noc_routing: $('ins-noc-routing').value } : {}) })));
    if (!r) return;
    const best = Object.entries(r.best || {}).map(([ph, b]) => `${ph} J 最优 ${b.engines}×${b.array} ${b.topology}（${sci(b.J_per_unit)} J）`).join(' · ');
    $('ins-noc-summary').textContent = `${r.chip} · ${r.model} · ${num(r.macs)} MAC · SRAM ${num(r.sram_mib)} MiB · 路由 ${num(r.pJ_bit_hop_noc)} pJ/bit/跳 · 导线 ${num(r.pJ_bit_mm_onchip)} pJ/bit/mm`
      + (best ? ' · ' + best : '') + ((r.notes || []).length ? ' · ' + r.notes.join('；') : '');
    const pct = (x) => (x * 100).toFixed(1) + '%';
    put($('ins-noc-tbl'), h('thead', {}, h('tr', {},
      ['阶段', '引擎 × 阵列', '拓扑', 'J/单位', 'tok/s', 'NoC 拥塞损失', 'MAC', 'SRAM', '导线', 'NoC 路由', 'DRAM', '静态', 'NoC+链路占比', '中心 SRAM J'].map((x) => h('th', {}, x)))),
      h('tbody', {}, (r.rows || []).map((x) => x.error
        ? h('tr', {}, h('td', {}, x.phase), h('td', {}, `${x.engines}×${x.array}`), h('td', {}, x.topology), h('td', { colspan: 11 }, x.error))
        : h('tr', {}, h('td', {}, x.phase), h('td', {}, `${x.engines}×${x.array}` + (x.base ? ' ←当前' : '')), h('td', {}, x.topology),
            h('td', {}, sci(x.J_per_unit)), h('td', {}, num(x.tok_s)), h('td', {}, x.noc_penalty != null ? pct(x.noc_penalty) : '—'),
            ...['mac', 'sram', 'wire', 'noc', 'dram', 'static'].map((k) => h('td', {}, pct(x.share[k] || 0))),
            h('td', {}, pct(x.noc_share)), h('td', {}, x.central ? sci(x.central.J_per_unit) : '—')))));
    $('ins-noc-status').textContent = `完成 · ${(r.rows || []).length} 点${staleTag(v0)}`;
    $('ins-noc-status').dataset.done = '1';
  } catch (e) {
    $('ins-noc-status').textContent = '错误：' + e.message;
    $('ins-noc-status').dataset.done = '1';
  }
}

/* 0.300: energy-aware design search — throughput vs J/unit Pareto front (/api/insight_dse, core/energy_dse.py) */
function paretoChart(rows) {
  const pts = rows.filter((x) => x.thr != null && x.J_per_unit != null);
  if (!pts.length) return h('div', { class: 'empty' }, '无数据');
  const W = 760, H = 300, P = { l: 70, r: 20, t: 24, b: 40 };
  const xs = pts.map((p) => p.J_per_unit), ys = pts.map((p) => p.thr);
  const x0 = Math.min(...xs) * 0.97, x1 = Math.max(...xs) * 1.03, y1 = Math.max(...ys) * 1.08 || 1;
  const X = (v) => P.l + (x1 === x0 ? 0.5 : (v - x0) / (x1 - x0)) * (W - P.l - P.r);
  const Y = (v) => H - P.b - (v / y1) * (H - P.t - P.b);
  const ns = 'http://www.w3.org/2000/svg';
  const s = (tag, a, txt) => { const e = document.createElementNS(ns, tag); for (const k in a) e.setAttribute(k, a[k]); if (txt !== undefined) e.textContent = txt; return e; };
  const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', id: 'ins-dse-svg' });
  svg.append(s('line', { x1: P.l, y1: H - P.b, x2: W - P.r, y2: H - P.b, stroke: '#2e3f55' }));
  for (let i = 0; i <= 4; i++) {
    const yv = (y1 * i) / 4, xv = x0 + ((x1 - x0) * i) / 4;
    svg.append(s('line', { x1: P.l, x2: W - P.r, y1: Y(yv), y2: Y(yv), stroke: '#1f2b3b' }));
    svg.append(s('text', { x: P.l - 6, y: Y(yv) + 4, 'text-anchor': 'end', fill: '#8fa3b8', 'font-size': 11 }, num(yv)));
    svg.append(s('text', { x: X(xv), y: H - P.b + 16, 'text-anchor': 'middle', fill: '#8fa3b8', 'font-size': 11 }, xv.toExponential(2)));
  }
  svg.append(s('text', { x: W / 2, y: H - 4, 'text-anchor': 'middle', fill: '#8fa3b8', 'font-size': 11 }, 'J/单位（越左越省电 = perf/W 越高）'));
  svg.append(s('text', { x: 4, y: 12, fill: '#8fa3b8', 'font-size': 11 }, '吞吐 /s/卡'));
  const front = pts.filter((p) => p.front).sort((a, b) => a.J_per_unit - b.J_per_unit);
  if (front.length > 1) svg.append(s('polyline', { points: front.map((p) => `${X(p.J_per_unit)},${Y(p.thr)}`).join(' '), fill: 'none', stroke: '#34d399', 'stroke-width': 2 }));
  for (const p of pts) {
    const c = s('circle', { cx: X(p.J_per_unit), cy: Y(p.thr), r: p.base ? 6 : 4,
      fill: !p.feasible ? 'none' : p.front ? '#34d399' : '#5b6b80', stroke: p.base ? '#f59e0b' : !p.feasible ? '#ef4444' : 'none', 'stroke-width': 2 });
    c.append(s('title', {}, `${p.engines}×${p.array} · MAC×${p.mac_x} · SRAM ${num(p.sram_mib)} MiB · ${p.topology} · ${p.mem_id || ''} · SLC ${num(p.slc_mib || 0)} MiB` + (p.why ? ` · ${p.why}` : '')));
    svg.append(c);
  }
  return svg;
}
async function runInsightDse() {
  const v0 = S.ver;
  const sci = (x) => (x == null ? '—' : x === 0 ? '0' : x.toExponential(3));
  const list = (id) => $(id).value.split(/[,\s]+/).filter(Boolean).map(Number);
  const macs = list('ins-dse-macs'), sram = list('ins-dse-sram'), memn = list('ins-dse-memn'), freq = list('ins-dse-freq');
  const links = $('ins-dse-links') ? list('ins-dse-links') : [];      // 0.306: NoC link-width axis (optional)
  {
    const bad = links.length > 6 || links.some((x) => !isFinite(x) || x <= 0);
    $('ins-dse-links').classList.toggle('bad', bad);
    if (bad) return;
  }
  for (const [id, v] of [['ins-dse-macs', macs], ['ins-dse-sram', sram], ['ins-dse-memn', memn], ['ins-dse-freq', freq]]) {
    const bad = !v.length || v.length > 6 || v.some((x) => !isFinite(x) || x <= 0) || (id === 'ins-dse-freq' && v.some((x) => x < 0.15 || x > 1));
    $(id).classList.toggle('bad', bad);
    if (bad) return;
  }
  // 0.301: memory-side axes, workload mix, SLO, memory $/GB (all optional; empty = the scenario's own)
  const memT = $('ins-dse-mem').value.split(/[,\s]+/).filter(Boolean);
  const memcap = list('ins-dse-memcap'), slc = list('ins-dse-slc');
  const usd = {};
  let badUsd = false;
  for (const kv of $('ins-dse-usd').value.split(/[,\s]+/).filter(Boolean)) {
    const [k, v] = kv.split(':');
    if (!k || !isFinite(+v) || +v < 0) badUsd = true; else usd[k] = +v;
  }
  $('ins-dse-usd').classList.toggle('bad', badUsd);
  for (const [id, v] of [['ins-dse-memcap', memcap], ['ins-dse-slc', slc]]) {
    const bad = v.length > 8 || v.some((x) => !isFinite(x) || x < 0);
    $(id).classList.toggle('bad', bad);
    if (bad) return;
  }
  if (badUsd || memT.length > 12) return;
  $('ins-dse-status').textContent = '计算中（MAC × 引擎边长 × SRAM 网格）…';
  delete $('ins-dse-status').dataset.done;
  try {
    const ex = { mac_scales: macs, sram_scales: sram, power_mode: $('ins-dse-power').value, curve: $('ins-dse-curve').value };
    if (freq.length > 1 || freq[0] !== 1) ex.f_scales = freq;     // 0.302: DVFS operating point axis
    if (links.length) ex.noc_links = links;                        // 0.306: link width as an axis (wins over the single value)
    else if ($('ins-dse-link').value !== '' && +$('ins-dse-link').value >= 0) ex.noc_link_Bpc = +$('ins-dse-link').value;   // 0.303
    if ($('ins-dse-routing').value) ex.noc_routing = $('ins-dse-routing').value;   // 0.304
    if ($('ins-dse-phase').value) ex.phase = $('ins-dse-phase').value;
    {   // 0.308: chiplet partitioning (dies per package × D2D grade, split axis, reticle / package / yield)
      const dies = list('ins-dse-dies'), d2d = $('ins-dse-d2d').value.split(/[,\s]+/).filter(Boolean);
      const bad = dies.length > 5 || dies.some((x) => !Number.isInteger(x) || x < 1 || x > 16);
      $('ins-dse-dies').classList.toggle('bad', bad);
      if (bad) { $('ins-dse-status').textContent = ''; return; }
      if (dies.length) {
        ex.dies = dies;
        if (d2d.length) ex.d2d_stds = d2d;
        if (+$('ins-dse-d2dunits').value >= 1) ex.d2d_units = parseInt($('ins-dse-d2dunits').value, 10);
        ex.die_split = $('ins-dse-split').value;
        if ($('ins-dse-reticle').value !== '' && +$('ins-dse-reticle').value >= 0) ex.reticle_mm2 = +$('ins-dse-reticle').value;
        if ($('ins-dse-pkg').value) ex.pkg_mm2 = $('ins-dse-pkg').value;
        if (+$('ins-dse-d0').value > 0) ex.d0_per_cm2 = +$('ins-dse-d0').value;
        if (!$('ins-dse-clusters').checked) { ex.mono_clusters = false; ex.mem_edges_shoreline = false; }   // 0.309
      }
      const kclu = list('ins-dse-kclu');                 // 0.310: NoC-cluster axis, cost per good package
      const badk = kclu.length > 5 || kclu.some((x) => !Number.isInteger(x) || x < 1 || x > 64);
      $('ins-dse-kclu').classList.toggle('bad', badk);
      if (badk) { $('ins-dse-status').textContent = ''; return; }
      if (kclu.length) ex.clusters = kclu;
      if ($('ins-dse-cost').value) ex.cost = $('ins-dse-cost').value;
    }
    if ($('ins-dse-topo').checked) ex.topologies = ['mesh', 'ring', 'crossbar'];
    if (memT.length) { ex.mem_types = memT; ex.mem_count_scales = memn; }
    if (memT.length && memcap.length) ex.mem_caps = memcap;
    if (slc.length) ex.slc_mibs = slc;
    const out = parseInt($('ins-dse-out').value, 10), tp = +$('ins-dse-tpot').value, tt = +$('ins-dse-ttft').value;
    if (out > 0) ex.mix_out_tokens = out;
    if (tp > 0) ex.slo_tpot_ms = tp;
    if (tt > 0) ex.slo_ttft_ms = tt;
    if (Object.keys(usd).length) ex.mem_usd_per_GB = usd;
    if (+$('ins-dse-maxusd').value > 0) ex.max_mem_usd = +$('ins-dse-maxusd').value;
    const r = await track(req('insight_dse', '/api/insight_dse', body(ex)));
    if (!r) return;
    const rows = r.rows || [];
    const u = rows.find((x) => x.unit)?.unit || '单位';
    const bp = r.best && r.best.perf_W, bt = r.best && r.best.thr;
    const desc = (b) => `${b.engines}×${b.array} · SRAM ${num(b.sram_mib)} MiB · ${b.topology} · ${b.mem_id} · SLC ${num(b.slc_mib)} MiB${b.f_op != null && b.f_op !== 1 ? ` · f×${b.f_op}` : ''}${(r.noc_links || []).length ? ` · NoC ${num(b.noc_link_Bpc)} B/周期` : ''}${b.dies > 1 ? ` · ${b.dies} 晶粒 ${b.d2d_std}` : ''}（${num(b.thr)} /s · ${sci(b.J_per_unit)} J）`;
    $('ins-dse-summary').textContent = `${r.chip} · ${r.model} · ${r.phase === 'mix' ? `混合（prompt ${r.mix.prompt} + 输出 ${r.mix.out_tokens}）` : r.phase} · 外存 ${(r.mems || []).length} 档 · SLC ${(r.slc_mibs || []).length} 档 · 频率 ${(r.f_scales || [1]).length} 档 · 可行 ${r.n_feasible}/${rows.length} 点 · 前沿 ${(r.front || []).length} 点`
      + (r.power_cap_W ? ` · 功耗预算 ${num(r.power_cap_W)} W（${r.power_mode === 'dvfs' ? 'DVFS 封顶' : '剔除'}）` : '')
      + (r.die_mm2 ? ` · 面积预算 ${num(r.die_mm2)} mm²` : '') + ((r.noc_links || []).length ? ` · NoC 链路轴 ${r.noc_links.map(num).join(' / ')} B/周期` : r.noc_link_Bpc ? ` · NoC 链路 ${num(r.noc_link_Bpc)} B/周期${r.noc_routing === 'xy' ? '（逐链路 XY）' : ''}` : '')
      + (bp ? ` · perf/W 最优 ${desc(bp)}` : '') + (bt ? ` · 吞吐最优 ${desc(bt)}` : '')
      + (r.chiplet ? ' · 单片 vs 拆分：' + (r.chiplet.compare || []).map((e) => {
        const v = { only_split: '仅拆分可行', split_both: '拆分吞吐与 J 均更优', split_thr: '拆分吞吐更高', split_J: '拆分 J 更低', mono: '单片更优', none: '均不可行' }[e.verdict] || e.verdict;
        const d = e.d_thr != null ? `（吞吐 ${(e.d_thr * 100).toFixed(1)}%，J ${(e.d_J * 100).toFixed(1)}%）` : e.mono_why ? `（单片：${e.mono_why}）` : '';
        const vf = e.verdict_flat && e.verdict_flat !== e.verdict ? `（对平面单片：${e.verdict_flat}）` : '';   // 0.309
        return `MAC×${e.mac_x} SRAM×${e.sram_x} ${e.mem_id} → ${v}${d}${vf}`;
      }).join('；') : '')
      + (r.costed ? ` · 成本前沿（吞吐↑ J↓ $/好封装↓）${rows.filter((x) => x.front_cost).length} 点` : '')
      + ((r.notes || []).length ? ' · ' + r.notes.join('；') : '');
    put($('ins-dse-chart'), paretoChart(rows));
    const pct = (x) => (x * 100).toFixed(1) + '%';
    put($('ins-dse-tbl'), h('thead', {}, h('tr', {},
      ['前沿', 'MAC ×', '引擎 × 阵列', 'SRAM MiB', '拓扑', 'NoC B/周期', 'NoC mm² / W', '晶粒 × D2D', 'D2D GB/s', 'Σ mm² / 良率', '$/好封装', '外存', 'GB/s', 'GB', 'SLC MiB', `${u}/s/卡`, `J/${u}`, `perf/W（${u}/J）`, 'W/卡', '工作点 f ×', '运行 f ×', 'mm²', 'TPOT ms', 'TTFT ms', '内存 $', 'MAC', 'SRAM', '导线', 'NoC', 'DRAM', '静态', '说明'].map((x) => h('th', {}, x)))),
      h('tbody', {}, rows.map((x) => h('tr', { class: x.front ? 'best' : '' },
        h('td', {}, x.front ? '★' : x.front3 ? '☆' : x.front_usd ? '$' : ''), h('td', {}, String(x.mac_x)),
        h('td', {}, `${x.engines}×${x.array}` + (x.base ? ' ←当前' : '')), h('td', {}, num(x.sram_mib)), h('td', {}, x.topology),
        h('td', {}, x.noc_link_Bpc ? num(x.noc_link_Bpc) : '不限'),
        h('td', {}, x.noc_W != null ? `${x.noc_mm2 != null ? num(x.noc_mm2) : '—'} / ${num(x.noc_W)}` : '—'),
        h('td', {}, x.dies > 1 ? `${x.dies} × ${x.d2d_std}×${x.d2d_units}（${(x.die_split || '').toUpperCase()}）` : x.dies === 1 || x.clusters ? (x.clusters > 1 ? `单片×${x.clusters}簇` : '单片') : '—'),
        h('td', {}, x.d2d_GBps != null ? num(x.d2d_GBps) + (x.d2d_noc_capped ? '（NoC 限）' : '') : '—'),
        h('td', {}, x.si_mm2 != null ? num(x.si_mm2) + (x.die_yield != null ? ` / ${(x.die_yield * 100).toFixed(0)}%` : '') : '—'),
        h('td', { title: x.usd_good_pkg != null ? `硅 $${num(x.si_usd)} + 封装 $${num(x.pkg_usd)} + 测试 $${num(x.test_usd)}` : '' }, x.usd_good_pkg != null ? '$' + num(x.usd_good_pkg) + (x.front_cost ? ' ★' : '') : '—'),   // 0.310
        h('td', {}, x.mem_id || '—'), h('td', {}, x.mem_GBps != null ? num(x.mem_GBps) : '—'), h('td', {}, x.mem_GB != null ? num(x.mem_GB) : '—'),
        h('td', {}, x.slc_mib != null ? num(x.slc_mib) : '—'),
        h('td', {}, x.thr != null ? num(x.thr) : '—'), h('td', {}, sci(x.J_per_unit)), h('td', {}, x.perf_W != null ? num(x.perf_W) : '—'),
        h('td', {}, x.W_card != null ? num(x.W_card) : '—'), h('td', {}, x.f_op != null ? x.f_op.toFixed(2) : '—'), h('td', {}, x.f_scale != null ? x.f_scale.toFixed(2) : '—'),
        h('td', {}, x.mm2 != null ? num(x.mm2) : '—'),
        h('td', {}, x.tpot_ms != null ? num(x.tpot_ms) : '—'), h('td', {}, x.ttft_ms != null ? num(x.ttft_ms) : '—'),
        h('td', {}, x.mem_usd != null ? num(x.mem_usd) : '—'),
        ...['mac', 'sram', 'wire', 'noc', 'dram', 'static'].map((k) => h('td', {}, x.share ? pct(x.share[k] || 0) : '—')),
        h('td', { class: 'l' }, x.why || [x.capped ? 'DVFS 封顶' : '', x.shoreline === 'tight' ? 'HBM 岸线紧（占第三 / 四边）' : '', x.front_kgd ? '良率前沿' : ''].filter(Boolean).join('；'))))));
    $('ins-dse-status').textContent = `完成 · ${rows.length} 点${staleTag(v0)}`;
    $('ins-dse-status').dataset.done = '1';
  } catch (e) {
    $('ins-dse-status').textContent = '错误：' + e.message;
    $('ins-dse-status').dataset.done = '1';
  }
}

/* 0.157: datapath sizing headroom */
async function runInsightSizing() {
  const btn = $('ins-size-run');
  const v0 = S.ver;
  $('ins-size-status').textContent = '计算中（每个资源逐次减半再二分 × 每个模型）…';
  delete $('ins-size-status').dataset.done;
  try {
    const ms = $('ins-models').value.split(',').map((x) => x.trim()).filter(Boolean);
    const r = await track(req('insight_sizing', '/api/insight_sizing', body({ ...(ms.length ? { models: ms } : {}), joint: $('ins-size-joint').checked, unpriced: $('ins-size-unpriced').value, energy_tie: $('ins-size-energy').checked })));
    if (!r) return;
    const mz = (k) => (k === 'J_tok' ? '能耗' : (r.metrics[k] || k).replace(/ 加速$/, ''));
    const nv = (x, v) => (x.integer ? String(Math.round(v)) : num(v));   // 0.184: whole cores / packages
    put($('ins-size-tbl'), h('thead', {}, h('tr', {}, ['资源', '当前', '最小可行', '可缩到', '受限于', '省 mm²', 'DRAM B / 能耗（最差模型，× 当前）'].map((x) => h('th', {}, x)))),
      h('tbody', {}, r.resources.map((x) => h('tr', {}, h('td', {}, x.label), h('td', {}, `${nv(x, x.current)} ${x.unit}`),
        h('td', {}, `${nv(x, x.min_ok)} ${x.unit}`), h('td', {}, (x.floor_hit && !x.unit_floor ? '≤ ' : '') + num(x.frac * 100) + '%'),
        h('td', {}, x.unit_floor ? `已是最小值 1 ${x.unit}` : x.limited_by ? `${x.limited_by.model} · ${x.limited_by.metric === 'fits' ? '放不下' : mz(x.limited_by.metric)}${x.limited_by.gain ? ' ×' + num(x.limited_by.gain) : ''}` : '—（到搜索下限）'),
        h('td', {}, x.mm2_saved != null ? num(x.mm2_saved) : '—'), h('td', {}, trafficCell(x.traffic))))));
    const j = r.joint;
    $('ins-size-joint-sum').textContent = j ? `各项最小值同时生效：${j.all_minima_hold ? '仍在 ±1% 内' : `不成立（最差 ×${num(j.all_minima_worst_gain)}，${(j.all_minima_limited_by || []).join(' · ')}）`}；回退 ${j.backoffs.length} 步后 ${j.holds ? '成立' : '仍不成立'}（最差 ×${num(j.worst_gain)}）${j.mm2_saved_total != null ? `，合计省 ${num(j.mm2_saved_total)} mm²（贪心 ${num(j.greedy_mm2_saved)}）` : ''} · 搜索 ${j.search} · 排序 ${j.rank}` : '';
    const bd = r.binding;
    $('ins-size-binding').textContent = bd ? `受限资源（无单项余量）：${bd.binding.map((x) => x.label).join('、') || '无'}；余量最大：${bd.most_slack.slice(0, 2).map((x) => `${x.label} ${num(x.frac * 100)}%`).join('、') || '—'}${bd.joint_only && bd.joint_only.length ? `；仅联合时受限：${bd.joint_only.join('、')}` : ''}` : '';
    put($('ins-size-joint-tbl'), j ? h('thead', {}, h('tr', {}, ['资源', '单项最小', '联合可行', '可缩到', '省 mm²'].map((x) => h('th', {}, x)))) : '',
      j ? h('tbody', {}, j.resources.map((e) => h('tr', {}, h('td', {}, e.key), h('td', {}, num(e.solo_min)), h('td', {}, num(e.joint)), h('td', {}, num(e.frac * 100) + '%'), h('td', {}, e.mm2_saved != null ? num(e.mm2_saved) : '—')))) : '');
    $('ins-size-status').textContent = `完成 · ${r.domain} · ${r.chip} · ${r.mem_id} · ${r.layout}${staleTag(v0)}`;
    $('ins-size-status').dataset.done = '1';
  } catch (e) {
    $('ins-size-status').textContent = '错误：' + e.message;
    $('ins-size-status').dataset.done = '1';
  }
}
/* 0.148: greedy multi-step upgrade path */
async function runInsightPath() {
  const btn = $('ins-path-run');
  const v0 = S.ver;
  $('ins-path-status').textContent = '计算中（每步 × 每个旋钮 × 每个模型）…';
  delete $('ins-path-status').dataset.done;
  try {
    const ms = $('ins-models').value.split(',').map((x) => x.trim()).filter(Boolean);
    const extra = { metric: S.insMetric || (S.insight && S.insight.default_metric) || undefined, steps: Math.round(+$('ins-steps').value || 6), beam: $('ins-beam').checked ? 3 : 1 };
    if (ms.length) extra.models = ms;
    if ($('ins-maxcost').value !== '') extra.max_cost = +$('ins-maxcost').value;
    const r = await track(req('insight_path', '/api/insight_path', body(extra)));
    if (!r) return;
    S.insPath = r;
    paintInsightPath();
    setInsightPathExport();
    $('ins-path-status').textContent = `完成 · ${r.path.length} 步 · ${r.search} · 成本 = ${r.cost_unit}${staleTag(v0)}`;
    $('ins-path-status').dataset.done = '1';
  } catch (e) {
    $('ins-path-status').textContent = '错误：' + e.message;
    $('ins-path-status').dataset.done = '1';
  }
}
function paintInsightPath() {
  const r = S.insPath;
  if (!r) return;
  const last = r.path.length ? r.path[r.path.length - 1] : null;
  const ml = (r.metrics && r.metrics[r.metric]) || r.metric;
  put($('ins-path-summary'),
    h('div', {}, h('b', {}, `${ml}：`), last ? `${r.path.length} 步后 ×${num(last.cum_gain)}（几何平均，相对起点），累计成本 ${num(last.cum_cost)} ${r.cost_unit}`
      : '无可取的升级（没有旋钮在不使任何模型变差、且不超预算的前提下提升 > 1%）'),
    r.beam > 1 ? h('div', {}, `搜索：${r.search}；贪心为 ×${num(r.greedy_cum_gain)} @ ${num(r.greedy_cum_cost)}`) : null,
    (r.notes || []).length ? h('div', {}, '注：', r.notes.join('；')) : null,
    r.skipped.length ? h('div', {}, '跳过：', r.skipped.map((x) => `${x.model}（${x.error}）`).join('；')) : null,
    h('div', { class: 'muted' }, r.basis));
  put($('ins-path-tbl'), h('thead', {}, h('tr', {}, ['步', '旋钮', '本步增益', '累计增益', '本步成本（依据）', '累计成本', '增益/成本', '逐模型累计', '次优备选'].map((x) => h('th', {}, x)))),
    h('tbody', {}, r.path.map((p) => h('tr', {}, h('td', {}, String(p.step)), h('td', {}, p.label), h('td', {}, '×' + num(p.gain)),
      h('td', {}, '×' + num(p.cum_gain)), h('td', {}, `${num(p.cost)}（${p.cost_basis}）`), h('td', {}, num(p.cum_cost)), h('td', {}, num(p.gain_per_cost)),
      h('td', {}, p.models.map((x) => `${x.model} ×${num(x.cum)}`).join('，')),
      h('td', {}, p.alternatives.map((a) => `${a.key} ×${num(a.gain)}@${num(a.cost)}`).join('，') || '—')))));
}
function paintInsight() {
  const r = S.insight;
  if (!r) return;
  const m = S.insMetric || r.default_metric || Object.keys(r.metrics)[0];
  if (!$('ins-models').placeholder) $('ins-models').placeholder = r.default_models.join(', ');
  const rk = r.rank[m] || [], ra = r.rank_per_mm2[m] || [], hu = r.hurts[m] || [];
  const lab = (k) => (r.knobs.find((x) => x.key === k) || {}).label || k;
  put($('ins-summary'),
    h('div', {}, h('b', {}, `${r.domain} · ${r.metrics[m]}：最能改善的旋钮 `), rk.length ? rk.map((k, i) => (i ? ' > ' : '') + lab(k)).join('') : '无（全部在 ±1% 内）'),
    h('div', {}, '每 mm²（按你的面积密度）：', ra.length ? ra.map(lab).join(' > ') : (r.area ? '—' : '未填面积密度（预算 → mm²/MiB、mm²/kMAC、固定面积），不排序')),
    h('div', {}, '每焦耳（能耗表 / 预设，或 TDP×时间）：', (r.rank_per_J && r.rank_per_J[m] && r.rank_per_J[m].length) ? r.rank_per_J[m].map(lab).join(' > ') : (r.j_source ? '—' : '未填能耗（或选能耗预设）且未填 TDP')),
    h('div', {}, '每美元（电价 × ΔJ）：', (r.rank_per_usd && r.rank_per_usd[m] && r.rank_per_usd[m].length) ? r.rank_per_usd[m].map(lab).join(' > ') : (r.usd_per_kWh != null && r.j_source ? '—' : '未填电价（或选电价预设）或缺少 J 来源')),
    h('div', {}, '每单位 TCO（CapEx + OpEx）：', (r.rank_tco && r.rank_tco.length) ? r.rank_tco.map(lab).join(' > ') : (r.tco && Object.keys(r.tco).length ? '—' : '未填卡价 + 摊销年数')),
    hu.length ? h('div', {}, '变差：', hu.map(lab).join('；')) : null,
    (r.notes || []).length ? h('div', {}, '注：', r.notes.join('；')) : null,
    h('div', { class: 'muted' }, r.basis));
  const llm = r.domain === 'llm';
  const heads = llm
    ? ['模型', '布局', '特征', 'TTFT (ms)', 'TPOT (ms)', 'decode tok/s/卡', '瓶颈 d/p', '放得下']
    : ['模型', '布局', '特征', (r.metrics.lat || '延迟').replace(' 加速','') + ' (s)', (r.metrics.thr || '吞吐').replace(' 加速',''), '瓶颈', '放得下'];
  put($('ins-models-tbl'), h('thead', {}, h('tr', {}, heads.map((x) => h('th', {}, x)))),
    h('tbody', {}, r.models.map((x) => x.error ? h('tr', {}, h('td', {}, x.model), h('td', { colspan: heads.length - 1 }, '错误：' + x.error))
      : llm
        ? h('tr', {}, h('td', {}, x.model), h('td', {}, x.layout), h('td', {}, x.features.join('、')), h('td', {}, num(x.ttft * 1e3)),
            h('td', {}, num(x.tpot * 1e3)), h('td', {}, num(x.thr)), h('td', {}, `${x.bound_d} / ${x.bound_p}`), h('td', {}, x.fits ? '是' : '否'))
        : h('tr', {}, h('td', {}, x.model), h('td', {}, x.layout), h('td', {}, x.features.join('、')), h('td', {}, num(x.lat)),
            h('td', {}, num(x.thr)), h('td', {}, x.bound_d), h('td', {}, x.fits ? '是' : '否')))));
  const ks = [...r.knobs].sort((a, b) => (b.score[m] || 0) - (a.score[m] || 0));
  const mkeys = Object.keys(r.metrics);
  put($('ins-tbl'), h('thead', {}, h('tr', {}, ['旋钮', '成本类别', ...mkeys.map((q) => r.metrics[q].replace(/加速$/, '')), 'Δ面积 mm²', `${r.metrics[m]} 收益 / mm²`].map((x) => h('th', {}, x)))),
    h('tbody', {}, ks.map((k) => h('tr', { class: 'ins-row', 'data-key': k.key, onclick: () => paintInsightDetail(k.key) },
      h('td', {}, k.label), h('td', {}, INS_COST_ZH[k.cost]), ...mkeys.map((q) => h('td', { class: q === m ? 'b' : '' }, insX(k.score[q]))),
      h('td', {}, k.d_mm2 === null ? '—' : num(k.d_mm2)), h('td', {}, k.gain_per_mm2[m] === null ? '—' : pct(k.gain_per_mm2[m]))))));
  paintInsightDetail(ks[0] && ks[0].key);
  paintInsightSlcTemp(r);
  setInsightExport();
}
function paintInsightSlcTemp(r) {
  const box = $('ins-slc-temp'); if (!box) return;
  const st = r && r.slc_temp;
  if (!st || !st.recommend || !st.recommend.length) { box.hidden = true; put(box); return; }
  box.hidden = false;
  const rows = st.recommend;
  const apply = (gib, label) => {
    const b = h('button', { class: 'btn ghost', type: 'button', title: '写入左侧 SLC MiB（' + label + '）' },
                `应用 ${label} ${gib == null ? '—' : gib + ' GiB'}`);
    b.disabled = gib == null;
    b.addEventListener('click', () => {
      if (gib == null) return;
      const inp = $('c-slc_mib');
      if (inp) { inp.value = String(gib * 1024); inp.dispatchEvent(new Event('input', { bubbles: true })); }
      if (typeof schedule === 'function') schedule();
    });
    return b;
  };
  put(box,
    h('div', {}, h('b', {}, 'SLC 25/85 °C（随设计洞察）'), ` · 策略 ${st.policy} · 漏电计入`),
    h('div', { class: 'scroll' }, h('table', { class: 'tbl tight' },
      h('thead', {}, h('tr', {}, ['模型', 'J 最优 25 °C', '交叉 25 °C', 'J 最优 85 °C', '交叉 85 °C', '时间最优 25/85'].map((x) => h('th', {}, x)))),
      h('tbody', {}, rows.map((x) => h('tr', {},
        h('td', {}, x.model),
        h('td', {}, x.opt_J_gib_25C == null ? '—' : x.opt_J_gib_25C + ' GiB'),
        h('td', {}, x.crossover_gib_25C == null ? '—' : x.crossover_gib_25C + ' GiB'),
        h('td', {}, x.opt_J_gib_85C == null ? '—' : x.opt_J_gib_85C + ' GiB'),
        h('td', {}, x.crossover_gib_85C == null ? '—' : x.crossover_gib_85C + ' GiB'),
        h('td', {}, `${x.opt_time_gib_25C ?? '—'} / ${x.opt_time_gib_85C ?? '—'} GiB`)))))),
    h('div', { class: 'row' },
      apply(rows[0].opt_J_gib_25C, '25 °C J 最优'),
      apply(rows[0].opt_J_gib_85C, '85 °C J 最优'),
      h('span', { class: 'small muted' }, '写入第一个模型的推荐容量；完整曲线见下方「SLC 能耗最优」。')));
}
function paintInsightDetail(key) {
  const r = S.insight, k = r && r.knobs.find((x) => x.key === key);
  if (!k) return put($('ins-detail'));
  const mkeys = Object.keys(r.metrics);
  const pw = r.energy_provided ? 'Δ能耗/单位' : 'ΔDRAM 字节/单位';
  put($('ins-detail'), h('caption', {}, '逐模型：' + k.label + (k.inverted ? '（数值为「有」相对「无」）' : '')),
    h('thead', {}, h('tr', {}, ['模型', ...mkeys.map((q) => r.metrics[q].replace(/加速$/, '')), pw, 'ΔMAC/单位', '瓶颈', '瓶颈变化', '放得下'].map((x) => h('th', {}, x)))),
    h('tbody', {}, k.models.map((x) => x.error ? h('tr', {}, h('td', {}, x.model), h('td', { colspan: mkeys.length + 5 }, '错误：' + x.error))
      : h('tr', {}, h('td', {}, x.model), ...mkeys.map((q) => h('td', {}, insX(x[q]))),
        h('td', {}, pct(r.energy_provided ? x.d_J_tok : x.d_dram_B_tok)
          + (r.energy_provided && x.d_J_prefill != null ? ` · prefill ${pct(x.d_J_prefill)}` : '')), h('td', {}, pct(x.d_mac_tok)),
        h('td', {}, r.domain === 'llm' ? `${x.bound_d} / ${x.bound_p}` : x.bound_d), h('td', {}, x.bound_flip ? '是' : ''), h('td', {}, x.fits ? '是' : '否')))));
}

/* ------------------------------------------------------------------ tabs */
function activate(tab) {
  S.tab = tab;
  document.querySelectorAll('#tabs button').forEach((b) => b.classList.toggle('on', b.dataset.tab === tab));
  document.querySelectorAll('.tab').forEach((t) => (t.hidden = t.id !== 'tab-' + tab));
}

/* ------------------------------------------------------------------ init */
/* ------------------------------------------------------------------ 0.256 panel layout */
// Each label's caption (text + tags, everything before its first control) goes into one block so tags stay inline and
// controls bottom-align across a grid row; a label whose only control is a checkbox lays out as a row.
function wrapCaps() {
  for (const l of document.querySelectorAll('#panel label')) {
    if (l.classList.contains('inline') || l.querySelector(':scope > .cap')) continue;
    const ctrls = l.querySelectorAll('input, select');
    if (ctrls.length && [...ctrls].every((c) => c.type === 'checkbox')) { l.classList.add('chk'); continue; }
    const cap = document.createElement('span'); cap.className = 'cap';
    for (const n of [...l.childNodes]) {
      if (n.nodeType === 1 && (n.matches('input, select') || n.querySelector('input, select'))) break;
      cap.append(n);
    }
    if (cap.textContent.trim()) l.prepend(cap);
  }
}
const SEC_KEY = 'accel-dse.ui.sections';
function secState() { try { return JSON.parse(localStorage.getItem(SEC_KEY) || '{}'); } catch (e) { return {}; } }
function initSections() {
  const st = secState();
  for (const d of document.querySelectorAll('#panel details.sec')) {
    if (d.id in st) d.open = !!st[d.id];
    d.addEventListener('toggle', () => { const x = secState(); x[d.id] = d.open; try { localStorage.setItem(SEC_KEY, JSON.stringify(x)); } catch (e) { /* private mode */ } });
  }
  const all = (open) => document.querySelectorAll('#panel details.sec').forEach((d) => { d.open = open; });
  $('sec-open-all').addEventListener('click', () => all(true));
  $('sec-close-all').addEventListener('click', () => all(false));
}
function filledIn(id) {      // fields the user filled in a section (budget / energy have no defaults: empty = not counted)
  let n = 0;
  for (const c of $(id).querySelectorAll('input, select')) {
    if (c.type === 'checkbox') n += c.checked ? 1 : 0;
    else if (c.value !== '' && !(c.tagName === 'SELECT' && c.closest('.seg'))) n += 1;
  }
  return n;
}
function paintPanel() {
  if (!S.sc || !S.cat) return;
  const sv = S.sc.serving, k = sv.spec_k || 0, L = S.sc.layout, m = model();
  // speculative decoding: the draft / acceptance / tree knobs only matter once k ≥ 1; the tree acceptance model only for a tree
  document.querySelectorAll('#sec-spec .spec-adv').forEach((e) => (e.hidden = !k));
  document.querySelectorAll('#sec-spec .spec-tree').forEach((e) => (e.hidden = !k || !((sv.spec_topk || 1) > 1 || sv.spec_draft_tokens)));
  $('spec-off-note').hidden = !!k;
  const set = (id, t) => { const e = $(id); if (e) e.textContent = t; };
  if (m) {
    const full = isFull(m), w = S.sc.workload || {};
    set('sec-model-sum', m.id + (S.wiW || S.wiKV ? ' · what-if dtype' : ''));
    const bt = S.best ? 'batch 自动' : `batch ${sv.batch}`;
    set('sec-workload-sum', full ? bt
      : `${sv.phase} · ${bt} · ${sv.phase === 'prefill' ? 'prompt ' + sv.prompt : 'ctx ' + sv.ctx}${m.loop_default ? ` · 循环 ${w.loops || m.loop_default}` : ''}`);
    set('sec-spec-sum', k ? `k ${k}${sv.draft_model ? ' · 草稿 ' + sv.draft_model : ''}${(sv.spec_topk || 1) > 1 ? ' · 树 topk ' + sv.spec_topk : ''}${sv.spec_sampling === 'sampled' ? ' · sampled' : ''}` : '关');
  }
  set('sec-serve-sum', [S.sc.pd && S.sc.pd.enabled ? 'PD 分离' : '合并', S.sc.host_step_us || S.sc.host_seq_us ? '主机开销' : ''].filter(Boolean).join(' · '));
  set('sec-par-sum', `${cards()} 卡 · PP${L.pp} TP${L.tp} DP${L.dp}${L.ep > 1 ? ' EP' + L.ep : ''}${(L.sp || 1) > 1 ? ' SP' + L.sp : ''}${(L.cp || 1) > 1 ? ' CP' + L.cp : ''}${S.sc.d2d_enabled ? ' · D2D' : ''}${S.sc.node_cards ? ` · ${S.sc.node_cards} 卡/节点` : ''}`);
  set('sec-chip-sum', `${S.preset} · ${S.sc.mem_id || ''} · ${S.sc.mapping || ''}`);
  {   // 0.256 nested 高级芯片参数 fold: dispatch / kernel launch / SLC / attention format at a glance
    const v = (id) => ($(id) ? $(id).value : ''), opt = (id) => { const e = $(id); return e && e.selectedIndex >= 0 ? e.options[e.selectedIndex].text : ''; };
    const kus = +v('c-kernel_us') || 0, slc = +v('c-slc_mib') || 0;
    set('sec-chip-adv-sum', [`派发 ${v('c-dispatch') ? opt('c-dispatch').split(' ')[0] : '理想'}`, kus ? `kernel ${kus} µs` : '',
      slc ? `SLC ${slc} MiB` : 'SLC 无', v('c-attn_fmt') ? '原生 KV 注意力' : '',
      +v('c-noc_link_Bpc') ? `NoC ${v('c-noc_link_Bpc')} B/周期` : ''].filter(Boolean).join(' · '));   // 0.306
  }
  for (const id of ['sec-budget', 'sec-energy']) { const n = filledIn(id); set(id + '-sum', n ? `已填 ${n} 项` : '未填（不计）'); }
}
// metric cards: notes clamp to 3 lines; a card whose note overflows gets an expand toggle (full text also on hover)
function markKpis() {
  for (const c of document.querySelectorAll('#kpis .kpi')) {
    const s = c.querySelector('.s');
    if (!s) continue;
    const more = c.classList.contains('open') || s.scrollHeight > s.clientHeight + 2;
    c.classList.toggle('more', more);
    if (more) s.title = s.textContent; else s.removeAttribute('title');
  }
}
function initKpis() {
  const k = $('kpis');
  new MutationObserver(() => requestAnimationFrame(markKpis)).observe(k, { childList: true });
  window.addEventListener('resize', () => requestAnimationFrame(markKpis));
  k.addEventListener('click', (e) => {
    const c = e.target.closest('.kpi.more');
    if (c && !getSelection().toString()) { c.classList.toggle('open'); S.kpiOpen = S.kpiOpen || {}; S.kpiOpen[c.querySelector('.l').textContent] = c.classList.contains('open'); }
  });
  // keep a card the user expanded open across re-evaluations
  new MutationObserver(() => {
    for (const c of k.querySelectorAll('.kpi')) { const l = c.querySelector('.l'); if (l && S.kpiOpen && S.kpiOpen[l.textContent]) c.classList.add('open'); }
  }).observe(k, { childList: true });
}
async function init() {
  wrapCaps(); initSections(); initKpis();
  try {
    const [health, cat, models] = await Promise.all([req('h', '/api/health'), req('c', '/api/catalog'), req('m', '/api/models')]);
    S.cat = cat;
    S.models = models.models;
    S.offline = models.offline;
    S.catalog = models.catalog;
    S.unlisted = models.unlisted;
    for (const m of S.models) S.byId[m.id] = m;
    $('ver').textContent = 'v' + health.version;
    S.version = health.version;
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
  // 0.205: a preset carries its default memory pairing (chip.default_mem) — switching presets applies it (one eval,
  // scheduled by applyMem after the memory resolves, so no eval runs on the old pairing)
  seg('chip-preset', Object.keys(S.cat.chips).map((k) => [k, k]), () => S.preset, (v) => {
    S.preset = v; S.chipOver = {}; S.mixed = []; onPreset(); syncInputs();
    const dm = S.cat.chips[v].default_mem || S.cat.mem_legacy_default || 'lpddr5x_4x64_8533_16g';
    if (!S.memInfo || S.memInfo.id !== dm) applyMem(dm); else schedule();
  });
  for (const k of ['freq_ghz', 'sram_mib', 'state_sram_mib', 'sram_port_Bpc', 'sram_port_Bpc_per_engine', 'gemv_macs', 'mac_eff', 'acc_kib', 'slc_mib', 'slc_GBps', 'kernel_us', 'dispatch_us', 'clk_ungated_share', 'noc_link_Bpc']) {
    const nullable = k === 'sram_port_Bpc' || k === 'acc_kib' || k === 'gemv_macs' || k === 'sram_port_Bpc_per_engine' || k === 'kernel_us' || k === 'dispatch_us' || k === 'clk_ungated_share' || k === 'noc_link_Bpc';
    bindNumber('c-' + k, () => chipVal(k), (x) => { if (x === null) delete S.chipOver[k]; else S.chipOver[k] = x; if (k === 'freq_ghz') onPreset(); },
      { int: k === 'gemv_macs', nullable });
  }
  seg('mapping', S.cat.mappings.map((m) => [m.id, m.label]), () => S.sc.mapping, (v) => { S.sc.mapping = v; $('mapping-desc').textContent = MAP_DESC[v]; schedule(); });
  $('mapping-desc').textContent = MAP_DESC[S.sc.mapping];
  for (const k of ['pp', 'tp', 'dp', 'ep', 'etp', 'sp']) {
    $('l-' + k).dataset.after = 'layout';
    bindNumber('l-' + k, () => S.sc.layout[k], (x) => { S.sc.layout[k] = x; AFTER.lastLayout = k; }, { int: true });
  }
  $('l-cp').dataset.after = 'layout';
  bindNumber('l-cp', () => S.sc.layout.cp || 1, (x) => { if (x > 1) S.sc.layout.cp = x; else delete S.sc.layout.cp; AFTER.lastLayout = 'cp'; }, { int: true });
  bindNumber('l-cards', cards, (x) => { setCards(x); for (const k of ['pp', 'tp', 'dp', 'ep', 'etp', 'sp', 'cp']) $('l-' + k)._sync(); cardsNote(); }, { int: true });
  for (const k of ['frames', 'height', 'width', 'steps', 'cfg', 'seq_len', 'msa', 'recycles', 'samples'])   // empty = release default (0)
    bindNumber('w-' + k, () => S.sc.workload[k] || null, (x) => (S.sc.workload[k] = x === null ? 0 : x), { int: true, nullable: true });
  for (const k of ['ref_images', 'ref_width', 'ref_height'])   // 0.252: 0 / empty = text-to-image / the output size
    bindNumber('w-' + k, () => S.sc.workload[k] || null, (x) => { if (x) S.sc.workload[k] = x; else delete S.sc.workload[k]; }, { int: true, nullable: true });
  bindNumber('w-psteps', () => S.sc.workload.steps || null, (x) => (S.sc.workload.steps = x === null ? 0 : x), { int: true, nullable: true });
  for (const k of ['clip_slo_s', 'seq_slo_ms', 'fold_slo_s']) bindNumber('w-' + k, () => S.sc.workload[k], (x) => (S.sc.workload[k] = x));
  bindNumber('w-host_GBps', () => S.sc.workload.host_GBps, (x) => (S.sc.workload.host_GBps = x));
  // 0.167: at most steps − forced (teacache: first + last; teacache_ret: first 5) — the server refuses more
  const skipCap = () => { const m = model(), st = S.sc.workload.steps || (m && m.workload && m.workload.steps) || 0;
    return Math.max(0, st - (S.sc.workload.step_cache === 'teacache_ret' ? 5 : 2)); };
  bindNumber('w-step_skip', () => S.sc.workload.step_skip || 0, (x) => { if (x) S.sc.workload.step_skip = x; else delete S.sc.workload.step_skip; },
    { int: true, check: (x) => x >= 0 && x <= skipCap() });
  $('w-mode_cache').addEventListener('change', (e) => {     // 0.255: off = reference (omitted from the scenario)
    if (e.target.value === 'on') S.sc.workload.mode_cache = 'on'; else delete S.sc.workload.mode_cache; schedule(); });
  $('w-cfg_branches').addEventListener('change', (e) => {   // 0.260: ref = default (omitted from the scenario)
    if (e.target.value === 'same') S.sc.workload.cfg_branches = 'same'; else delete S.sc.workload.cfg_branches; schedule(); });
  $('w-step_cache').addEventListener('change', (e) => { if (e.target.value === 'teacache') delete S.sc.workload.step_cache; else S.sc.workload.step_cache = e.target.value;
    if ((S.sc.workload.step_skip || 0) > skipCap()) { S.sc.workload.step_skip = skipCap(); $('w-step_skip')._sync(); } schedule(); });
  bindNumber('w-host_TFLOPS', () => S.sc.workload.host_TFLOPS, (x) => (S.sc.workload.host_TFLOPS = x));
  for (const k of ['sram_mib', 'macs', 'tflops', 'cards', 'power_W_card', 'usd_per_kWh', 'tdp_W_card', 'card_usd', 'amort_years', 'pue', 'dram_GiB', 'die_mm2', 'mm2_per_mib_sram', 'mm2_per_kmac',
    'mm2_per_mib_slc', 'mm2_fixed', 'system_mm2', 'mm2_per_dq_hbm', 'mm2_per_dq_lpddr', 'mm2_per_d2d_unit', 'mm2_per_GBps_c2c', 'uncore_frac', 'mm2_host_io', 'mm2_per_unit_mc'])
    bindNumber('b-' + k, () => (S.budget || {})[k] ?? null, (x) => { S.budget = { ...(S.budget || {}), [k]: x }; },
      { nullable: true, int: k === 'cards' });
  for (const k of ['pJ_mac', 'pJ_vec', 'pJ_bit_sram', 'pJ_bit_dram', 'pJ_bit_link', 'idle_W', 'pJ_bit_slc', 'pJ_bit_d2d', 'pJ_bit_net', 'idle_W_prefill', 'pJ_bit_host', 'pJ_bit_dram_hbm', 'pJ_bit_dram_lpddr', 'pJ_bit_dram_hbm4', 'pJ_bit_dram_lpddr6', 'pJ_bit_phy_hbm', 'pJ_bit_phy_lpddr', 'idle_frac_tdp', 'tdp_W', 'pJ_clk_mac_cycle', 'sram_temp_C', 'dram_temp_C', 'slc_access_ref_mib', 'pJ_bit_mm_onchip', 'slc_mm2_per_mib', 'pJ_bit_hop_noc', 'noc_mm2_per_kmac', 'pJ_noc_router_cycle', 'mW_noc_router_leak'])
    bindNumber('e-' + k, () => (S.energy || {})[k] ?? null, (x) => { S.energy = { ...(S.energy || {}), [k]: x }; }, { nullable: true });
  $('e-power_cap').addEventListener('change', (e) => { S.powerCap = e.target.checked; schedule(); });   // 0.189
  initDensityPresets();
  initEnergyPresets();
  initElectricityPresets();
  initElectricityPresets('b-tco-preset', 'tco_presets', 'tcoPresetFilled');
  $('best-layout').addEventListener('click', bestLayout);
  for (const bx of ['cmp', 'sw']) for (const f of ['csv', 'json']) $(`${bx}-exp-${f}`).addEventListener('click', () => doExport(bx, f));
  $('best-batch').checked = S.best;
  AFTER.layout = () => {
    const k = AFTER.lastLayout;
    if (k !== 'pp') fixMoe(k);
    if (S.sc.layout.pp > model().n_layers) S.sc.layout.pp = model().n_layers;
    for (const x of ['pp', 'ep', 'etp']) if (x !== k) $('l-' + x)._sync();
    cardsNote();
  };
  for (const k of ['GBps', 'alpha_us']) bindNumber('k-' + k, () => S.sc.link[k], (x) => (S.sc.link[k] = x));
  bindNumber('d-GBps', () => d2dGBps(), (x) => { S.sc.d2d.GBps = x; S.sc.d2d_std = 'custom'; $('d2d-std').value = 'custom'; d2dNote(); });
  bindNumber('d-alpha_us', () => S.sc.d2d.alpha_us, (x) => (S.sc.d2d.alpha_us = x));
  bindNumber('p-package_cards', () => S.sc.package_cards || 1, (x) => (S.sc.package_cards = x), { int: true });
  bindNumber('d2d-units', () => S.sc.d2d_units, (x) => { S.sc.d2d_units = x; $('d-GBps')._sync(); d2dNote(); }, { int: true });
  for (const k of ['GBps', 'alpha_us']) bindNumber('n-' + k, () => S.sc.net[k], (x) => (S.sc.net[k] = x));
  bindNumber('n-node_cards', () => S.sc.node_cards || 0, (x) => (S.sc.node_cards = x), { int: true });
  for (const [k, o] of [['oversub', {}], ['oversub_spine', {}], ['pod_nodes', { int: true }], ['leaf_nodes', { int: true }], ['switch_radix', { int: true }], ['torus_x', { int: true }],
    ['hop_d2d_us', {}], ['hop_link_us', {}], ['hop_net_us', {}], ['hop_net_tree_us', {}], ['hop_spine_us', {}], ['hop_core_us', {}]])
    bindNumber('fab-' + k, () => fab()[k], (x) => (fab()[k] = x), o);
  for (const k of ['channels', 'channel_GBps'])     // 0.124: both > 0 or both off (read the pair from the inputs)
    bindNumber('fab-' + k, () => fab()[k] || null, () => {
      const c = readNum($('fab-channels'), true), g = readNum($('fab-channel_GBps'), false);
      if (c > 0 && g > 0) { fab().channels = c; fab().channel_GBps = g; } else { delete fab().channels; delete fab().channel_GBps; }
    }, { int: k === 'channels', nullable: true, check: (x) => x >= 0 });
  opts($('fab-link_topology'), [['switch', 'switch（非阻塞交换）'], ['full_mesh', 'full mesh（全互连）'], ['ring', 'ring（双向环）'], ['torus2d', '2D torus']], S.sc.link.topology || 'switch');
  opts($('fab-net_topology'), [['fat_tree', 'fat-tree / leaf-spine'], ['rail', 'rail-optimized']], fab().net_topology);
  opts($('fab-algo'), [['auto', 'auto（取最快）'], ['auto_overlap', 'auto_overlap（按暴露时间，计入与计算重叠）'], ['ring', 'ring'], ['tree', 'tree（双二叉树）'], ['hier', 'hier（分层）']], fab().algo);
  opts($('fab-innet_reduce'), [['off', '关'], ['net', '跨节点（SHARP 类）'], ['net+link', '跨节点 + 节点内（NVLS 类）']], fab().innet_reduce);
  $('fab-enabled').addEventListener('change', (e) => { fab().enabled = e.target.checked; fabPaint(); schedule(); });
  $('fab-contention').addEventListener('change', (e) => { fab().contention = e.target.checked; schedule(); });
  opts($('fab-net_tiers'), [['2', '2 层（leaf-spine）'], ['3', '3 层（leaf / spine / core）']], String(fab().net_tiers || 2));
  opts($('fab-overlap'), [['sum', 'sum（相加，0.59）'], ['ports', 'ports（不同端口并发）']], fab().overlap || 'sum');
  $('fab-net_tiers').addEventListener('change', (e) => { fab().net_tiers = Number(e.target.value); fabPaint(); schedule(); });
  $('fab-overlap').addEventListener('change', (e) => { fab().overlap = e.target.value; schedule(); });
  opts($('fab-protocol'), [['off', 'off（0.60：LL 时延 + 满带宽）'], ['auto', 'auto（每次取最快）'], ['LL', 'LL'], ['LL128', 'LL128'], ['Simple', 'Simple']], fab().protocol || 'off');
  $('fab-protocol').addEventListener('change', (e) => { fab().protocol = e.target.value; schedule(); });
  $('fab-kv_feedback').addEventListener('change', (e) => { fab().kv_feedback = e.target.checked; schedule(); });
  $('fab-kv_feedback_prefill').addEventListener('change', (e) => { fab().kv_feedback_prefill = e.target.checked; schedule(); });
  $('fab-link_topology').addEventListener('change', (e) => { S.sc.link.topology = e.target.value; schedule(); });
  for (const k of ['algo', 'net_topology', 'innet_reduce']) $('fab-' + k).addEventListener('change', (e) => { fab()[k] = e.target.value; schedule(); });
  fabPaint();
  opts($('d2d-std'), [...S.cat.d2d_standards.map((d) => [d.id, d.label]), ['custom', '自定义（直接填 GB/s）']], S.sc.d2d_std);
  $('d2d-std').addEventListener('change', (e) => {
    if (e.target.value === 'custom') S.sc.d2d.GBps = d2dGBps();      // keep the current figure as the custom start
    S.sc.d2d_std = e.target.value; $('d-GBps')._sync(); d2dNote(); schedule();
  });
  $('d2d-enabled').addEventListener('change', (e) => {
    S.sc.d2d_enabled = e.target.checked;
    if (e.target.checked && (S.sc.package_cards || 1) < 2) S.sc.package_cards = 2;
    $('p-package_cards')._sync(); d2dPaint(); schedule();
  });
  $('c-slc_policy').addEventListener('change', (e) => { S.chipOver.slc_policy = e.target.value; schedule(); });
  $('c-attn_fmt').addEventListener('change', (e) => { if (e.target.value) S.chipOver.attn_fmt = e.target.value; else delete S.chipOver.attn_fmt; schedule(); });
  $('c-dispatch').addEventListener('change', (e) => { if (e.target.value) S.chipOver.dispatch = e.target.value; else delete S.chipOver.dispatch; schedule(); });
  for (const k of ['noc_topology', 'noc_routing', 'noc_mem_attach', 'noc_eject']) {   // 0.306: chip-panel NoC selects
    $('c-' + k).addEventListener('change', (e) => { if (e.target.value) S.chipOver[k] = e.target.value; else delete S.chipOver[k]; schedule(); });
  }
  $('c-noc_mem_edges').addEventListener('change', (e) => { if (e.target.value) S.chipOver.noc_mem_edges = parseInt(e.target.value, 10); else delete S.chipOver.noc_mem_edges; schedule(); });   // 0.310
  $('c-clk_datapath_gate').addEventListener('change', (e) => { if (e.target.checked) S.chipOver.clk_datapath_gate = true; else delete S.chipOver.clk_datapath_gate; schedule(); });
  $('l-pp_split').addEventListener('change', (e) => { S.sc.pp_split = e.target.value; schedule(); });
  $('l-indexer').addEventListener('change', (e) => { S.sc.indexer = e.target.value; schedule(); });
  $('l-attn_mode').addEventListener('change', (e) => { if (e.target.value === 'reference') delete S.sc.attn_mode; else S.sc.attn_mode = e.target.value; schedule(); });
  for (const k of ['images', 'image_w', 'image_h'])
    bindNumber('s-' + k, () => S.sc.serving[k] ?? (k === 'images' ? 0 : 1024), (x) => (S.sc.serving[k] = x), { int: true });
  for (const k of ['batch', 'ctx', 'prompt', 'out_len', 'spec_k', 'microbatches'])
    bindNumber('s-' + k, () => S.sc.serving[k], (x) => (S.sc.serving[k] = x), { int: true });
  for (const k of ['spec_accept', 'tpot_slo_ms', 'ttft_slo_ms']) bindNumber('s-' + k, () => S.sc.serving[k], (x) => (S.sc.serving[k] = x));
  bindNumber('s-moe_skew', () => S.sc.serving.moe_skew ?? 1, (x) => (S.sc.serving.moe_skew = x));
  const VDEF = { videos: 0, video_frames: 32, video_w: 1280, video_h: 720 };   // 0.165 video input
  for (const k of Object.keys(VDEF))
    bindNumber('s-' + k, () => S.sc.serving[k] ?? VDEF[k], (x) => { if (x === VDEF[k]) delete S.sc.serving[k]; else S.sc.serving[k] = x; }, { int: true });
  const HDEF = { host_step_us: 0, host_seq_us: 0, host_steps: 1 };              // 0.164 host / runtime overhead
  for (const k of Object.keys(HDEF))
    bindNumber('h-' + k, () => S.sc[k] ?? HDEF[k], (x) => { if (x === HDEF[k]) delete S.sc[k]; else S.sc[k] = x; }, { int: k === 'host_steps' });
  for (const [id, dflt] of [['s-spec_sampling', 'greedy'], ['s-spec_accept_src', 'manual']])   // 0.253
    $(id).addEventListener('change', (e) => { const k = id.slice(2); if (e.target.value === dflt) delete S.sc.serving[k]; else S.sc.serving[k] = e.target.value; schedule(); });
  bindNumber('w-loops', () => S.sc.workload.loops || null, (x) => { if (x) S.sc.workload.loops = x; else delete S.sc.workload.loops; }, { int: true, nullable: true });
  $('w-loop_kv').addEventListener('change', (e) => { if (e.target.value === 'per_loop') delete S.sc.workload.loop_kv; else S.sc.workload.loop_kv = e.target.value; schedule(); });
  $('s-spec_rollback').addEventListener('change', (e) => { if (e.target.value === 'snapshot') delete S.sc.serving.spec_rollback; else S.sc.serving.spec_rollback = e.target.value; treeWarn(); schedule(); });
  // 0.225: tree drafting + radix tracking fields (defaults omitted from the scenario)
  const TDEF = { spec_topk: [1, 1, 10], spec_draft_tokens: [0, 0, 64], mamba_track_interval: [0, 0, 1048576] };
  for (const k of Object.keys(TDEF)) {
    const [d, lo, hi] = TDEF[k];
    bindNumber('s-' + k, () => S.sc.serving[k] ?? d, (x) => { if (x === d) delete S.sc.serving[k]; else S.sc.serving[k] = x; treeWarn(); },
      { int: true, check: (x) => x >= lo && x <= hi && !(k === 'spec_draft_tokens' && x === 1) });
  }
  $('s-spec_tree_accept_model').addEventListener('change', (e) => { if (e.target.value === 'chain') delete S.sc.serving.spec_tree_accept_model; else S.sc.serving.spec_tree_accept_model = e.target.value; treeWarn(); schedule(); });
  $('s-draft_model').addEventListener('change', (e) => { if (e.target.value) S.sc.serving.draft_model = e.target.value; else delete S.sc.serving.draft_model; draftWarn(); schedule(); });
  $('h-host_overlap').addEventListener('change', (e) => { if (e.target.value === 'serial') delete S.sc.host_overlap; else S.sc.host_overlap = e.target.value; schedule(); });
  for (const k of ['pp', 'tp', 'dp', 'ep', 'etp'])
    bindNumber('pd-' + k, () => S.sc.pd.prefill_layout[k], (x) => (S.sc.pd.prefill_layout[k] = x), { int: true });
  for (const k of ['prefill_cards', 'decode_cards'])
    bindNumber('pd-' + k, () => S.sc.pd[k] || null, (x) => (S.sc.pd[k] = x === null ? 0 : x), { int: true, nullable: true });
  bindNumber('pd-prefix_turns', () => S.sc.pd.prefix_turns ?? 1, (x) => { if (x === 1) delete S.sc.pd.prefix_turns; else S.sc.pd.prefix_turns = x; }, { int: true, check: (x) => x >= 1 && x <= 64 });
  bindNumber('pd-prefix_turn_user', () => S.sc.pd.prefix_turn_user ?? 0, (x) => { if (!x) delete S.sc.pd.prefix_turn_user; else S.sc.pd.prefix_turn_user = x; }, { int: true, check: (x) => x >= 0 });
  bindNumber('pd-prefix_sessions', () => S.sc.pd.prefix_sessions ?? 0, (x) => { if (!x) delete S.sc.pd.prefix_sessions; else S.sc.pd.prefix_sessions = x; }, { int: true, check: (x) => x >= 0 });
  $('pd-prefix_turn_prompt').addEventListener('change', (e) => { if (e.target.checked) S.sc.pd.prefix_turn_prompt = true; else delete S.sc.pd.prefix_turn_prompt; schedule(); });
  $('pd-prefix_writeback').addEventListener('change', (e) => { if (e.target.checked) S.sc.pd.prefix_writeback = true; else delete S.sc.pd.prefix_writeback; schedule(); });
  bindNumber('pd-kv_GBps', () => S.sc.pd.kv_GBps, (x) => (S.sc.pd.kv_GBps = x), { nullable: true });
  bindNumber('pd-load', () => S.sc.pd.load ?? 0.8, (x) => (S.sc.pd.load = x), { check: (x) => x > 0 && x < 1 });
  bindNumber('pd-rate_rps', () => S.sc.pd.rate_rps ?? null, (x) => (S.sc.pd.rate_rps = x), { nullable: true, check: (x) => x > 0 });
  bindNumber('pd-chunk_budget', () => S.sc.pd.chunk_budget ?? null, (x) => { if (x === null) delete S.sc.pd.chunk_budget; else S.sc.pd.chunk_budget = x; }, { int: true, nullable: true, check: (x) => x >= 16 && x <= 1048576 });
  $('pd-chunk_profile').addEventListener('change', (e) => {
    if (e.target.value === 'vllm') delete S.sc.pd.chunk_profile; else S.sc.pd.chunk_profile = e.target.value;
    $('pd-chunk_budget').disabled = !(e.target.value.startsWith('vllm') || e.target.value.startsWith('sglang'));
    $('pd-chunk_tokens').disabled = e.target.value !== 'fixed';
    schedule();
  });
  bindNumber('pd-chunk_tokens', () => S.sc.pd.chunk_tokens ?? 512, (x) => (S.sc.pd.chunk_tokens = x), { int: true, check: (x) => x >= 16 });
  for (const k of ['prompt_cv', 'out_cv'])
    bindNumber('pd-' + k, () => S.sc.pd[k] ?? 0, (x) => (S.sc.pd[k] = x), { check: (x) => x >= 0 && x <= 4 });
  bindNumber('pd-prefix_hit', () => S.sc.pd.prefix_hit ?? 0, (x) => (S.sc.pd.prefix_hit = x), { check: (x) => x >= 0 && x <= 0.99 });
  const mixInp = $('pd-length_mix');
  mixInp.addEventListener('input', () => {
    const t = mixInp.value.trim();
    let rows = [];
    let ok = true;
    if (t) {
      rows = t.split(/[,，;]/).map((p) => p.trim()).filter((p) => p).map((p) => p.split(':').map(Number));
      ok = rows.length <= 16 && rows.every((r) => r.length === 3 && r[0] > 0 && Number.isInteger(r[1]) && r[1] >= 1 && Number.isInteger(r[2]) && r[2] >= 1);
    }
    mixInp.classList.toggle('bad', !ok);
    if (!ok) return;
    S.sc.pd.length_mix = rows;
    schedule();
  });
  mixInp._sync = () => { mixInp.value = (S.sc.pd.length_mix || []).map((r) => r.join(':')).join(', '); mixInp.classList.remove('bad'); };
  const treeInp = $('pd-prefix_tree');
  treeInp.addEventListener('input', () => {
    const t = treeInp.value.trim();
    let rows = [];
    let ok = true;
    if (t) {
      rows = t.split(/[,，;]/).map((p) => p.trim()).filter((p) => p).map((p) => p.split(':').map(Number));
      ok = rows.length <= 6 && rows.every((r) => r.length === 3 && Number.isInteger(r[0]) && r[0] >= 1 && Number.isInteger(r[1]) && r[1] >= 1 && r[2] >= 0 && r[2] <= 3);
    }
    treeInp.classList.toggle('bad', !ok);
    if (!ok) return;
    S.sc.pd.prefix_tree = rows;
    if (rows.length) S.sc.pd.prefix_len = 0;
    schedule();
  });
  treeInp._sync = () => { treeInp.value = (S.sc.pd.prefix_tree || []).map((r) => r.join(':')).join(', '); treeInp.classList.remove('bad'); };
  for (const k of ['prefix_on_decode', 'search_layouts', 'prefix_affinity', 'search_decode_batch', 'simulate'])
    $('pd-' + k).addEventListener('change', (e) => { S.sc.pd[k] = e.target.checked; schedule(); });
  bindNumber('pd-prefix_len', () => S.sc.pd.prefix_len || null, (x) => (S.sc.pd.prefix_len = x === null ? 0 : x), { int: true, nullable: true, check: (x) => x >= 0 });
  bindNumber('pd-prefix_count', () => S.sc.pd.prefix_count ?? 1000, (x) => (S.sc.pd.prefix_count = x), { int: true, check: (x) => x >= 1 && x <= 1e9 });
  bindNumber('pd-prefix_zipf', () => S.sc.pd.prefix_zipf ?? 1, (x) => (S.sc.pd.prefix_zipf = x), { check: (x) => x >= 0 && x <= 3 });
  bindNumber('pd-prefix_host_GB', () => S.sc.pd.prefix_host_GB || 0, (x) => { if (x) S.sc.pd.prefix_host_GB = x; else delete S.sc.pd.prefix_host_GB; }, { check: (x) => x >= 0 });
  $('pd-prefix_host_policy').addEventListener('change', (e) => { if (e.target.value === 'write_through') delete S.sc.pd.prefix_host_policy; else S.sc.pd.prefix_host_policy = e.target.value; schedule(); });
  bindNumber('pd-prefix_cache_GB', () => S.sc.pd.prefix_cache_GB ?? null, (x) => (S.sc.pd.prefix_cache_GB = x), { nullable: true, check: (x) => x >= 0 });
  $('pd-kv_policy').addEventListener('change', (e) => { S.sc.pd.kv_policy = e.target.value || 'off'; schedule(); });
  bindNumber('pd-kv_capacity_GB', () => S.sc.pd.kv_capacity_GB ?? null, (x) => (S.sc.pd.kv_capacity_GB = x), { nullable: true, check: (x) => x > 0 });
  $('pd-kv_admit').addEventListener('change', (e) => { S.sc.pd.kv_admit = e.target.value || 'before_prefill'; schedule(); });
  $('pd-ttft_resid').addEventListener('change', (e) => { S.sc.pd.ttft_resid = e.target.value || 'full'; schedule(); });
  bindNumber('pd-swap_GBps', () => S.sc.pd.swap_GBps ?? null, (x) => (S.sc.pd.swap_GBps = x), { nullable: true, check: (x) => x > 0 && x < 1e5 });
  $('pd-prefill_chip').addEventListener('change', (e) => { S.sc.pd.prefill_chip = e.target.value || null; schedule(); });
  $('pd-prefill_mem_id').addEventListener('change', (e) => { S.sc.pd.prefill_mem_id = e.target.value.trim() || null; schedule(); });
  $('pd-enabled').addEventListener('change', (e) => { S.sc.pd.enabled = e.target.checked; $('pd-inputs').hidden = !e.target.checked; schedule(); });
  $('pd-kv_layerwise').addEventListener('change', (e) => { S.sc.pd.kv_layerwise = e.target.checked; schedule(); });
  bindNumber('mem_eff', () => S.sc.mem_eff, (x) => (S.sc.mem_eff = x), { nullable: true });
  seg('phase', null, () => S.sc.serving.phase, (v) => { S.sc.serving.phase = v; schedule(); });
  $('best-batch').addEventListener('change', (e) => { S.best = e.target.checked; schedule(); });
  $('wi-w').addEventListener('change', (e) => { S.wiW = e.target.value; schedule(); });
  $('wi-kv').addEventListener('change', (e) => { S.wiKV = e.target.value; schedule(); });
  $('wi-act').addEventListener('change', (e) => { S.wiAct = e.target.value; schedule(); });
  $('w-placement').addEventListener('change', (e) => { S.sc.workload.placement = e.target.value; schedule(); });
  $('w-vae_tiling').addEventListener('change', (e) => { S.sc.workload.vae_tiling = e.target.checked; schedule(); });
  $('w-tri_attn').addEventListener('change', (e) => {     // 0.119 structure models: triangle-attention kernel
    if (e.target.value === 'fused') delete S.sc.workload.tri_attn; else S.sc.workload.tri_attn = e.target.value;
    if (e.target.value !== 'chunked') delete S.sc.workload.tri_chunk;
    $('w-tri_chunk').disabled = e.target.value !== 'chunked'; if (e.target.value !== 'chunked') $('w-tri_chunk').value = '';
    schedule();
  });
  $('w-tri_chunk').addEventListener('change', (e) => {
    const v = parseInt(e.target.value, 10);
    if (v > 0) S.sc.workload.tri_chunk = v; else delete S.sc.workload.tri_chunk;
    schedule();
  });
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
  $('sw-path').addEventListener('change', swHint);
  $('pf-run').addEventListener('click', runPareto);
  $('ins-run').addEventListener('click', runInsight);
  $('ins-path-run').addEventListener('click', runInsightPath);
  $('ins-size-run').addEventListener('click', runInsightSizing);
  $('ins-slc-run').addEventListener('click', () => runInsightSlc(false));
  $('ins-noc-run').addEventListener('click', runInsightNoc);   // 0.299
  $('ins-dse-run').addEventListener('click', runInsightDse);   // 0.300
  $('ins-sp-run').addEventListener('click', runInsightSpeech);
  $('ins-sps-run').addEventListener('click', runInsightSpeechStreams);
  $('ins-dp-run').addEventListener('click', runInsightDispatch);
  $('ins-slc-cmp').addEventListener('click', () => runInsightSlc(true));
  for (const f of ['csv', 'json']) { $('ins-exp-' + f).addEventListener('click', () => doExport('ins', f)); $('ins-path-exp-' + f).addEventListener('click', () => doExport('ins-path', f)); }
  fillSweepPaths();
  renderModels();
  paintScope();
  window.__accel = { S, req, seq, EXP, expCSV, expJSON, csvCell };
  runEval();
}
init();

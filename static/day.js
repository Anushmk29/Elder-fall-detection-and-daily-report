const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const form = $('#day');
const rows = [];          // {file, tr, room, start, zones, bar, camera}
let zoneFiles = [];
const MAX_STREAMS = 4;    // browsers allow ~6 connections per host; leave room for the status polls

const pad = (n) => String(n).padStart(2, '0');
const today = new Date();
form.date.value = `${today.getFullYear()}-${pad(today.getMonth() + 1)}-${pad(today.getDate())}`;
fetch('/api/day/zones').then((r) => r.json()).then((z) => { zoneFiles = z; });

const ROOM_WORDS = { kitchen: 'Kitchen', bath: 'Bathroom', toilet: 'Bathroom', bed: 'Bedroom', living: 'Living room',
  sofa: 'Living room', hall: 'Hallway' };
const guessRoom = (name) => ROOM_WORDS[Object.keys(ROOM_WORDS).find((w) => name.toLowerCase().includes(w))] || '';

function guessTime(name, k) {
  const m = name.match(/(?:^|\D)([01]\d|2[0-3])[-_:.h]?([0-5]\d)(?:\D|$)/);  // "kitchen_0715.mp4", "07-15"
  return m ? `${m[1]}:${m[2]}` : `${pad(Math.min(8 + k, 23))}:00`;
}

function addFiles(files) {
  for (const file of files) {
    if (!file.type.startsWith('video/') && !/\.(mp4|mov|webm|mkv|avi)$/i.test(file.name)) continue;
    const tr = el('tr');
    const room = Object.assign(el('input'), { value: guessRoom(file.name), placeholder: 'Kitchen', maxLength: 40, required: true });
    room.setAttribute('list', 'rooms');
    const start = Object.assign(el('input'), { type: 'time', value: guessTime(file.name, rows.length), required: true });
    const zones = el('select');
    zones.append(new Option('none', ''), ...zoneFiles.map((z) => new Option(z, z)));
    const yaml = file.name.replace(/\.\w+$/, '.yaml');
    if (zoneFiles.includes(yaml)) zones.value = yaml;
    const bar = el('div', 'bar'); bar.append(el('i'));
    const x = el('button', 'x', '×'); x.type = 'button'; x.title = 'Remove';
    const row = { file, tr, room, start, zones, bar };
    x.onclick = () => { rows.splice(rows.indexOf(row), 1); tr.remove(); refreshForm(); };
    const name = el('td', null, file.name);
    name.append(el('div', 'muted small', `${(file.size / 2 ** 20).toFixed(1)} MB`));
    tr.append(name, ...[room, start, zones, bar, x].map((c) => { const td = el('td'); td.append(c); return td; }));
    $('#clips tbody').append(tr);
    rows.push(row);
  }
  refreshForm();
}

function refreshForm() {
  $('#clips').hidden = !rows.length;
  $('#go').disabled = !rows.length;
  $('#msg').textContent = rows.length ? `${rows.length} video(s). Times are when each recording started.` : '';
}

$('#files').addEventListener('change', (e) => { addFiles(e.target.files); e.target.value = ''; });
const drop = $('#drop');
drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', (e) => { e.preventDefault(); drop.classList.remove('over'); addFiles(e.dataTransfer.files); });

form.addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const seen = new Set();
  for (const r of rows) {
    const key = `${r.room.value.trim().toLowerCase()} ${r.start.value}`;
    if (seen.has(key)) { $('#msg').textContent = `Two videos are "${r.room.value}" at ${r.start.value}: give one a different time.`; return; }
    seen.add(key);
  }
  if (!form.reportValidity()) return;
  $('#go').disabled = true;
  const date = form.date.value;
  if (form.fresh.checked) {
    await fetch('/api/day/clear', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ date }) });
  }
  for (const [k, r] of rows.entries()) {
    $('#msg').textContent = `Uploading ${k + 1} of ${rows.length}: ${r.file.name}…`;
    const fd = new FormData();
    fd.append('file', r.file);
    fd.append('date', date);
    fd.append('start', r.start.value);
    fd.append('room', r.room.value.trim());
    fd.append('zones', r.zones.value);
    fd.append('notify', form.notify.checked ? 'yes' : '');
    try {
      const res = await fetch('/api/day/add', { method: 'POST', body: fd });
      const j = await res.json();
      if (!res.ok) { r.tr.title = j.detail; r.bar.replaceWith(el('span', 'error', j.detail || 'Failed')); continue; }
      r.camera = j.camera;
    } catch { r.bar.replaceWith(el('span', 'error', 'Upload failed')); }
  }
  $('#msg').textContent = 'Processing… you can watch below.';
  $('#status').textContent = 'Processing';
  $('#watching').hidden = false;
  poll(date);
});

const streams = {};
async function poll(date) {
  let st;
  try { st = await (await fetch('/api/live/state')).json(); } catch { setTimeout(() => poll(date), 2000); return; }
  const cams = Object.fromEntries((st.cameras || []).map((c) => [c.camera, c]));
  const mine = rows.filter((r) => r.camera);
  let running = 0;
  for (const r of mine) {
    const c = cams[r.camera];
    if (!c) continue;
    const pct = Math.round(100 * (c.finished ? 1 : c.progress || 0));
    r.bar.firstChild.style.width = `${pct}%`;
    r.bar.title = `${pct}%`;
    if (!c.finished) running += 1;
    stream(r, c);
  }
  const done = mine.length && mine.every((r) => cams[r.camera]?.finished);
  if (done || !mine.length) {
    $('#status').textContent = 'Done';
    $('#msg').textContent = mine.length ? 'All videos processed.' : 'Nothing was processed.';
    $('#go').disabled = false;
    $('#watching').hidden = true;
    if (mine.length) loadResult(date);
    return;
  }
  $('#status').textContent = `Processing ${running}`;
  setTimeout(() => poll(date), 1500);
}

function stream(r, c) {
  let s = streams[r.camera];
  const live = Object.values(streams).filter((x) => x.img.src.includes('.mjpg')).length;
  if (!s) {
    const card = el('figure', 'cam');
    const img = el('img');
    img.alt = r.camera;
    const cap = el('figcaption');
    card.append(img, cap);
    $('#cams').append(card);
    s = streams[r.camera] = { card, img, cap };
  }
  if (c.finished) { s.card.remove(); delete streams[r.camera]; return; }  // closes its MJPEG connection
  if (!s.img.src.includes('.mjpg') && live < MAX_STREAMS) s.img.src = `/live/${encodeURIComponent(r.camera)}.mjpg`;
  s.cap.textContent = `${r.room.value} · started ${r.start.value} · ${Math.round(100 * (c.progress || 0))}%` +
    (c.people ? ` · ${c.people} person(s)` : '') + (c.posture ? ` · ${c.posture}` : '');
}

const LABELS = {
  wake_min: 'Woke up', bed_min: 'Went to bed', kitchen_visits: 'Kitchen visits (meals)', bathroom_count: 'Bathroom visits',
  bathroom_longest_min: 'Longest bathroom visit (min)', active_min: 'Active time (min)',
  rest_longest_min: 'Longest rest on sofa (min)', night_episodes: 'Up at night',
};
const clock = (m) => (m == null ? '–' : `${Math.floor(m / 60)}:${pad(Math.round(m % 60))}`);
const show = (k, v) => (v == null ? '–' : ['wake_min', 'bed_min'].includes(k) ? clock(v) : Math.round(v));
const dur = (m) => (m >= 60 ? `${Math.floor(m / 60)} h ${pad(m % 60)} min` : `${m} min`);
const color = (room) => { const r = (room || '').toLowerCase(); return r.includes('kitchen') ? '#d08c3a' : /bath|toilet|wc/.test(r) ? '#4a8fb8' : r.includes('bed') ? '#7a6bb0' : /porch|garden|path|stairs|outside|yard|gate/.test(r) ? '#6b8f3a' : '#2f6f6a'; };
const FALLISH = ['fall_suspected', 'fall_recovered', 'fall_alert', 'fall_dismissed', 'floor_ai_alert', 'lying_outside_rest'];
const NOBODY = /^(no (person|one|people)|nobody|person not)/i;  // AI reviews of clips where the detector was wrong

let clips = {};      // camera -> ["HHMMSS", ...] (scripts/dvr.py names each clip after its start time)
let moments = [];    // one per clip with events: {camera, room, i, ts, said, falls, empty}
let shown = [];      // moments left after the filters
let current = -1;    // index into shown
let filter = 'all';  // 'all' | 'falls' | a camera name
let tlX = null;      // ts -> left % of the timeline, for the "now playing" mark
const video = $('#video');
const secs = (hhmmss) => +hhmmss.slice(0, 2) * 3600 + +hhmmss.slice(2, 4) * 60 + +hhmmss.slice(4, 6);
const hms = (c) => `${c.slice(0, 2)}:${c.slice(2, 4)}:${c.slice(4, 6)}`;
const day0 = () => new Date(`${form.date.value}T00:00`).getTime() / 1000;

async function loadResult(date) {
  form.date.value = $('#pick-day').value = date;
  history.replaceState(null, '', `/day?date=${date}`);
  $('#result').hidden = $('#day-nav').hidden = false;
  $('#add').open = false;
  $('#day-title').textContent = new Date(`${date}T12:00`).toLocaleDateString(undefined, { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' });
  $('#report').textContent = 'Building the report…';
  let d;
  try { d = await (await fetch(`/api/day/${date}`)).json(); } catch { $('#report').textContent = 'Could not load the report.'; return; }
  clips = d.clips || {};
  $('#report').textContent = d.text;
  tiles(d.metrics);
  compare(d);
  buildMoments(d.events);
  chips();
  render();
  timeline(d.events);
  if (current < 0 && shown.length) openMoment(0, true);
}

function tiles(m) {
  const box = $('#tiles');
  box.textContent = '';
  const ai = m.ai_reviewed?.segments ? ' · AI-checked' : '';
  const falls = (m.falls || []).length;
  const checked = (m.ai_dismissed || []).length;
  const add = (label, value, sub, tone, onclick) => {
    const t = el(onclick ? 'button' : 'div', `tile ${tone || ''}`);
    if (onclick) { t.type = 'button'; t.onclick = onclick; }
    t.append(el('span', 'tile-label', label), el('b', null, value), el('span', 'tile-sub', sub || ''));
    box.append(t);
  };
  add(m.bedroom ? 'Woke up' : 'First up', clock(m.wake_min), m.first_seen ? `seen ${m.first_seen}–${m.last_seen}` : 'nobody seen');
  add('Went outside', `${(m.outings || []).length}×`, (m.outings || []).length ? `${(m.outings || []).reduce((s, o) => s + o.min, 0)} min in all` : '');
  add('Active', dur(m.active_min || 0), `sitting ${dur(m.sitting_min || 0)} · lying ${dur(m.lying_min || 0)}${ai}`);
  add('Falls', String(falls), checked ? `${checked} more looked at by the AI: not falls` : 'none seen',
    falls ? 'alert' : 'ok', () => setFilter('falls'));
  add('AI reviewed', m.ai_reviewed?.segments ? `${m.ai_reviewed.segments} clips` : 'off', m.ai_reviewed?.min ? `${dur(m.ai_reviewed.min)} of footage` : 'set VLM_SECOND_OPINION=1');
}

function compare(d) {
  $('#compare').hidden = !d.baseline.days;  // nothing to compare with yet: don't show a table of dashes
  const t = $('#metrics');
  t.textContent = '';
  t.append(Object.assign(el('tr'), { innerHTML: '<th></th><th>This day</th><th>7-day average</th><th></th>' }));
  for (const k of d.numeric) {
    const tr = el('tr');
    const b = d.baseline.metrics[k];
    const f = d.flags[k];
    tr.append(el('td', null, LABELS[k] || k), el('td', null, show(k, d.metrics[k])), el('td', 'muted', b ? show(k, b.mean) : '–'));
    const td = el('td');
    if (f) td.append(el('span', `badge ${f === 'ok' ? '' : 'warn'}`, f === 'ok' ? 'usual' : f));
    tr.append(td);
    t.append(tr);
  }
}

function clipAt(camera, ts) {
  const list = clips[camera] || [];
  const t = ts - day0();
  let i = -1;
  for (let k = 0; k < list.length && secs(list[k]) <= t + 1; k++) i = k;  // the last clip started by then
  return i < 0 ? null : { i, offset: Math.max(t - secs(list[i]) - 3, 0) };  // 3 s early, to see it coming
}

function buildMoments(events) {
  const by = {};
  for (const e of events) {
    const at = clipAt(e.camera, e.ts);
    if (!at) continue;  // uploaded on this page, or the clip was deleted
    const key = `${e.camera}:${at.i}`;
    const m = by[key] = by[key] || { camera: e.camera, room: e.room, i: at.i, ts: day0() + secs(clips[e.camera][at.i]), said: [], falls: [] };
    if (e.type === 'ai_review' && !m.said.includes(e.title)) m.said.push(e.title);
    if (FALLISH.includes(e.type) || e.severity !== 'info') m.falls.push(e);
  }
  moments = Object.values(by).sort((a, b) => a.ts - b.ts);
  for (const m of moments) m.empty = m.said.length > 0 && m.said.every((s) => NOBODY.test(s)) && !m.falls.length;
}

// what the AI saw; "no person" remarks only when that's all it said (the person was out of view part of the time)
const said = (m) => (m.said.filter((x) => !NOBODY.test(x)).length ? m.said.filter((x) => !NOBODY.test(x)) : m.said).join(' · ');

function chips() {
  const box = $('#chips');
  box.textContent = '';
  const rooms = Object.fromEntries(moments.map((m) => [m.camera, m.room || m.camera]));
  const opts = [['all', 'All', moments.filter((m) => !m.empty).length],
    ['falls', 'Falls & alerts', moments.filter((m) => m.falls.length).length],
    ...Object.keys(rooms).map((c) => [c, rooms[c], moments.filter((m) => m.camera === c && !m.empty).length])];
  for (const [key, label, n] of opts) {
    const b = el('button', `chip ${filter === key ? 'on' : ''} ${key === 'falls' && n ? 'alert' : ''}`, `${label} ${n}`);
    b.type = 'button';
    if (!['all', 'falls'].includes(key)) b.style.setProperty('--c', color(label));
    b.onclick = () => setFilter(key);
    box.append(b);
  }
}

function setFilter(key) {
  filter = key;
  current = -1;
  chips();
  render();
  $('#moments').scrollTop = 0;
  if (window.innerWidth < 900) $('.day-right').scrollIntoView({ behavior: 'smooth' });
}

function render() {
  const all = $('#show-empty').checked;
  shown = moments.filter((m) => (all || !m.empty) && (filter === 'all' || (filter === 'falls' ? m.falls.length : m.camera === filter)));
  const list = $('#moments');
  list.textContent = '';
  $('#mo-count').textContent = `${shown.length} of ${moments.length}`;
  if (!shown.length) list.append(el('li', 'empty', filter === 'falls' ? 'No falls or alerts this day.' : 'Nothing here.'));
  shown.forEach((m, k) => {
    const li = el('li', `mo ${m.empty ? 'is-empty' : ''} ${m.falls.some((e) => e.severity === 'alert') ? 'is-alert' : ''}`);
    li.tabIndex = 0;
    const top = el('div', 'mo-top');
    const room = el('span', 'room', m.room || m.camera); room.style.setProperty('--c', color(m.room));
    top.append(el('time', null, hms(clips[m.camera][m.i])), room);
    for (const e of m.falls) {
      top.append(el('span', `badge ${e.type === 'fall_dismissed' ? 'info' : e.severity === 'info' ? 'warn' : e.severity}`, e.title));
    }
    li.append(top, el('div', 'mo-text', said(m) || 'Person seen (no AI review)'));
    const why = m.falls.find((e) => e.type !== 'fall_suspected');
    if (why) li.append(el('div', 'mo-why', why.detail));
    li.onclick = () => openMoment(k);
    li.onkeydown = (ev) => { if (ev.key === 'Enter') openMoment(k); };
    list.append(li);
  });
  markActive();
}

function openMoment(k, quiet, offset) {
  if (k < 0 || k >= shown.length) return;
  current = k;
  const m = shown[k];
  const fall = m.falls.find((e) => e.type === 'fall_suspected') || m.falls[0];
  const start = offset ?? (fall ? clipAt(m.camera, fall.ts).offset : 0);  // a fall: start just before it
  $('#player-title').textContent = `${m.room || m.camera} · ${hms(clips[m.camera][m.i])}`;
  $('#player-sub').textContent = [said(m), ...m.falls.map((e) => `${e.clock} ${e.title}`)].filter(Boolean).join('  —  ');
  $('#prev-clip').disabled = k === 0;
  $('#next-clip').disabled = k === shown.length - 1;
  video.src = `/media/dvr/${form.date.value}/${encodeURIComponent(m.camera)}/${clips[m.camera][m.i]}.mp4#t=${start.toFixed(1)}`;
  markActive();
  if (quiet) return;  // first load: show it without starting playback
  video.play().catch(() => {});
  if (window.innerWidth < 900) $('#player').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function markActive() {
  document.querySelectorAll('.mo.active').forEach((x) => x.classList.remove('active'));
  const li = $('#moments').children[current];
  if (li && current >= 0) { li.classList.add('active'); li.scrollIntoView({ block: 'nearest' }); }
  const now = $('#tl-now');
  const m = shown[current];
  if (now && m && tlX) { now.hidden = false; now.style.left = tlX(m.ts); } else if (now) now.hidden = true;
}

function play(camera, ts) {  // from the timeline: the moment for that clip if it's listed, else the clip itself
  const at = clipAt(camera, ts);
  if (!at) return;
  let k = shown.findIndex((m) => m.camera === camera && m.i === at.i);
  if (k < 0) { filter = 'all'; $('#show-empty').checked = true; chips(); render(); k = shown.findIndex((m) => m.camera === camera && m.i === at.i); }
  if (k >= 0) openMoment(k, false, at.offset);
}

function timeline(events) {
  const box = $('#timeline');
  box.textContent = '';
  tlX = null;
  const visits = []; const open = {};
  for (const e of [...events].sort((a, b) => a.ts - b.ts)) {
    if (e.type === 'zone_enter') open[e.camera] = e;
    else if ((e.type === 'zone_exit' || e.type === 'camera_offline') && open[e.camera]) {
      visits.push({ room: open[e.camera].room, camera: e.camera, a: open[e.camera].ts, b: e.ts }); delete open[e.camera];
    }
  }
  if (!visits.length) { box.append(el('p', 'empty', 'Nobody was seen in these videos.')); return; }
  const d0 = day0();
  const h0 = Math.floor((Math.min(...visits.map((v) => v.a)) - d0) / 3600);
  const h1 = Math.ceil((Math.max(...visits.map((v) => v.b)) - d0) / 3600);
  const span = Math.max(h1 - h0, 1) * 3600;
  const x = tlX = (ts) => `${(100 * (ts - d0 - h0 * 3600)) / span}%`;
  const wrap = el('div', 'tl-wrap');
  for (const room of [...new Set(visits.map((v) => v.room))]) {
    const row = el('div', 'tl-row');
    const track = el('div', 'tl-track');
    for (const v of visits.filter((w) => w.room === room)) {
      const seg = el('div', 'tl-seg');
      seg.style.cssText = `left:${x(v.a)};width:max(4px, calc(${x(v.b)} - ${x(v.a)}));background:${color(room)}`;
      seg.title = `${new Date(v.a * 1000).toLocaleTimeString()} – ${new Date(v.b * 1000).toLocaleTimeString()} (click to watch)`;
      seg.onclick = () => play(v.camera, v.a);
      track.append(seg);
    }
    for (const e of events.filter((e) => e.room === room && e.severity !== 'info')) {
      const m = el('div', `tl-mark ${e.severity}`); m.style.left = x(e.ts); m.title = `${e.clock} ${e.title}`;
      m.onclick = () => play(e.camera, e.ts);
      track.append(m);
    }
    row.append(el('span', null, room), track);
    wrap.append(row);
  }
  const now = el('div', 'tl-now'); now.id = 'tl-now'; now.hidden = true;
  const lane = el('div', 'tl-lane'); lane.append(now);
  wrap.append(lane);
  const axis = el('div', 'tl-axis');
  const every = Math.max(1, Math.ceil((h1 - h0) / 8));
  for (let h = h0; h <= h1; h += every) axis.append(el('span', null, `${pad(h % 24)}:00`));
  box.append(wrap, axis);
  markActive();
}

const full = () => (document.fullscreenElement ? document.exitFullscreen() : video.requestFullscreen());
$('#full').onclick = full;
video.ondblclick = (e) => { e.preventDefault(); full(); };
video.onerror = () => { $('#player-note').textContent = 'This browser cannot play the clip (the DVR records HEVC/H.265). ' +
  'Use Edge, or Chrome with hardware video decoding on.'; };
$('#prev-clip').onclick = () => openMoment(current - 1);
$('#next-clip').onclick = () => openMoment(current + 1);
$('#show-empty').onchange = () => { current = -1; chips(); render(); };
document.addEventListener('keydown', (e) => {
  if (/INPUT|SELECT|TEXTAREA/.test(document.activeElement?.tagName) || e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === 'n' || k === 'j') openMoment(current + 1);
  else if (k === 'p' || k === 'k') openMoment(current - 1);
  else if (k === 'f' && !$('#result').hidden) full();
});

const shiftDay = (n) => {
  const d = new Date(`${form.date.value}T12:00`); d.setDate(d.getDate() + n);
  current = -1; filter = 'all';
  loadResult(`${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`);
};
$('#prev-day').onclick = () => shiftDay(-1);
$('#next-day').onclick = () => shiftDay(1);
$('#pick-day').onchange = (e) => { if (e.target.value) { current = -1; filter = 'all'; loadResult(e.target.value); } };
$('#refresh').addEventListener('click', () => loadResult(form.date.value));
const wanted = new URLSearchParams(location.search).get('date');  // opened from History
if (/^\d{4}-\d{2}-\d{2}$/.test(wanted || '')) loadResult(wanted);

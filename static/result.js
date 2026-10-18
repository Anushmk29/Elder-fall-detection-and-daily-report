const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const ZONE_COLORS = ['#2f6f6a', '#c08a3e', '#7a6fb0', '#4f8fc0', '#b0606f', '#6f9a4a', '#8a8a8a'];
const video = $('#video');
const id = decodeURIComponent(location.pathname.slice(3));  // after /r/; batch ids contain slashes

fetch(`/api/results/${encodeURI(id)}`)
  .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
  .then(render)
  .catch(() => { $('#title').textContent = 'Result not found or expired'; });

function seek(t) { video.currentTime = Math.max(0, t - 0.5); video.play(); }
function pct(t, dur) { return `${Math.min(100, (t / dur) * 100)}%`; }

function render(r) {
  const m = r.media_base, dur = r.duration;
  document.title = `${r.title} · ElderCare Vision`;
  $('#title').textContent = r.title;
  $('#sub').textContent = `${Math.round(dur)} s clip · starts ${r.clock_start}`;
  if (id.startsWith('batch/')) { const back = el('a', null, ' · ← batch review'); back.href = '/batch'; $('#sub').append(back); }
  video.src = m + r.video;
  video.poster = m + r.poster;

  for (const g of r.glance) {
    const t = el('div', `tile ${g.tone === 'alert' && g.value !== '0' ? 'alert' : ''}`);
    t.append(el('small', null, g.label), el('b', null, g.value), el('span', null, g.note || ''));
    $('#tiles').append(t);
  }

  // timeline: one row per person, coloured by room; event markers on top
  const zoneColor = {};
  r.zones.filter((z) => z.kind === 'room').forEach((z, i) => { zoneColor[z.name] = ZONE_COLORS[i % ZONE_COLORS.length]; });
  const heads = [];
  const track = (parent) => {
    parent.addEventListener('click', (e) => { if (e.target === parent) seek(((e.offsetX / parent.clientWidth) * dur) + 0.5); });
    const h = el('div', 'tl-head'); parent.append(h); heads.push(h);
    return parent;
  };
  const evTrack = track($('#tl-events'));
  for (const e of r.events) {
    const mk = el('div', `tl-mark ${e.severity}`);
    mk.style.left = pct(e.t, dur); mk.title = `${e.clock} ${e.title}`;
    mk.onclick = () => seek(e.t);
    evTrack.append(mk);
  }
  for (const who of [...new Set(r.people.map((p) => p.label))]) {
    const row = el('div', 'tl-row'), tr = track(el('div', 'tl-track'));
    row.append(el('span', null, who), tr);
    for (const s of r.timeline.filter((s) => s.who === who)) {
      const seg = el('div', 'tl-seg');
      seg.style.left = pct(s.start, dur);
      seg.style.width = pct(Math.max(s.end - s.start, dur / 200), dur);
      seg.style.background = zoneColor[s.zone] || '#999';
      seg.title = `${s.zone}: ${Math.round(s.end - s.start)} s`;
      seg.onclick = () => seek(s.start + 0.5);
      tr.append(seg);
    }
    $('#tl-rows').append(row);
  }
  $('#tl-start').textContent = r.clock_start;
  $('#tl-end').textContent = `+${Math.round(dur)} s`;
  for (const [name, c] of Object.entries(zoneColor)) {
    const s = el('span'); const i = el('i'); i.style.background = c; s.append(i, name); $('#legend').append(s);
  }

  // alerts
  const cards = [];
  if (!r.events.length) $('#alerts').append(el('p', 'empty', 'No alerts in this clip.'));
  for (const e of r.events) {
    const c = el('button', `alert-card ${e.severity}`);
    const img = el('img'); img.src = m + e.thumb; img.alt = '';
    const body = el('div');
    const label = { alert: 'Alert', warn: 'Check', info: 'Note' }[e.severity];
    body.append(el('span', `badge ${e.severity}`, `${label} · ${e.clock}`), el('h3', null, `${e.title} · ${e.who}`), el('p', null, e.detail));
    c.append(img, body);
    c.onclick = () => seek(e.t);
    cards.push([e, c]);
    $('#alerts').append(c);
  }

  // report: shown as plain text (never innerHTML, it comes from an LLM)
  $('#report').textContent = r.report.text;
  const badge = $('#ai-badge');
  badge.textContent = r.report.ai ? `AI-written · ${r.report.model}` : r.report.note;
  badge.className = `badge ${r.report.ai ? '' : 'warn'}`;

  for (const c of r.captions) {
    const d = el('div', 'moment');
    const img = el('img'); img.src = m + c.img; img.alt = ''; img.loading = 'lazy';
    const p = el('p'); p.append(el('time', null, c.clock), c.text);
    if (!c.ai) p.append(' ', el('span', 'badge warn', 'AI text unavailable'));
    d.append(img, p);
    d.onclick = () => seek(c.t);
    $('#moments').append(d);
  }

  video.addEventListener('timeupdate', () => {
    const t = video.currentTime;
    heads.forEach((h) => { h.style.left = pct(t, dur); });
    cards.forEach(([e, c]) => c.classList.toggle('active', t >= e.t - 0.5 && t <= Math.max(e.t_end, e.t + 3)));
  });
}

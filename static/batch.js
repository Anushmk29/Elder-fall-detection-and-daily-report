const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const when = (t) => new Date(t * 1000).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
let data = { groups: [], todo: [], history: [] };

// your label in the UI wins over expected_fall from labels.csv
const label = (c) => (c.review.fall !== undefined ? c.review.fall : (c.expected_fall ? Number(c.expected_fall) : null));
function verdict(c) {
  if (c.status === 'error') return ['error', 'alert'];
  if (c.status === 'not processed') return ['—', 'info'];
  const l = label(c), det = Number(c.falls) > 0;
  if (l == null) return [det ? 'fall detected' : 'no fall', det ? 'warn' : 'info'];
  return l ? (det ? ['hit', ''] : ['MISSED', 'alert']) : (det ? ['FALSE ALARM', 'alert'] : ['ok', '']);
}
const STATUS_TONE = { reviewed: '', 'to review': 'warn', 'not processed': 'info', error: 'alert' };

function load() {
  fetch('/api/batch').then((r) => (r.ok ? r.json() : Promise.reject(r.status))).then((d) => { data = d; render(); })
    .catch(() => { $('#groups').textContent = 'Could not load batch results (this page only works on the machine running the server).'; });
}
load();
setInterval(() => { if (!document.activeElement || document.activeElement.tagName !== 'INPUT') load(); }, 15000);
$('#filter').onchange = render;

async function save(c, group, patch) {
  const body = { group, file: c.file, fall: label(c), note: c.review.note || '', ...patch };
  const res = await fetch('/api/batch/review', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!res.ok) { alert('Could not save'); return; }
  load();
}

function play(c, group) {
  if (!c.video) return;
  $('#player').hidden = false;
  $('#p-title').textContent = `${group} / ${c.file}`;
  $('#p-open').href = `/r/${c.id}`;
  const v = $('#p-video');
  v.src = c.video; v.poster = c.poster; v.play();
  const marks = $('#p-marks'); marks.textContent = '';
  const times = (c.fall_at_s || '').split(' ').filter(Boolean);
  if (!times.length) marks.append(el('span', 'badge info', 'no fall detected in this clip'));
  for (const t of times) {
    const b = el('button', 'badge alert', `fall at ${t} s ▶`);
    b.onclick = () => { v.currentTime = Math.max(0, Number(t) - 1); v.play(); };
    marks.append(b);
  }
  $('#player').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

const find = (group, file) => data.groups.find((g) => g.group === group)?.clips.find((c) => c.file === file);

function render() {
  const all = data.groups.flatMap((g) => g.clips);
  const n = (s) => all.filter((c) => c.status === s).length, v = all.map((c) => verdict(c)[0]);
  const ov = $('#overall'); ov.textContent = '';
  ov.append(el('span', 'badge', `folders run ${data.groups.length}/${data.groups.length + data.todo.length}`),
    el('span', 'badge', `processed ${all.length - n('not processed')}/${all.length} clips`),
    el('span', `badge ${n('to review') ? 'warn' : ''}`, `reviewed ${n('reviewed')} · to review ${n('to review')}`),
    el('span', `badge ${v.includes('MISSED') ? 'alert' : ''}`, `missed ${v.filter((x) => x === 'MISSED').length}`),
    el('span', `badge ${v.includes('FALSE ALARM') ? 'alert' : ''}`, `false alarms ${v.filter((x) => x === 'FALSE ALARM').length}`));
  if (n('error')) ov.append(el('span', 'badge alert', `errors ${n('error')}`));

  const todo = $('#todo'); todo.textContent = '';
  if (data.todo.length) {
    const d = el('details'), s = el('summary', null, `Not run yet: ${data.todo.length} folder(s) — ${data.todo.map((t) => t.name).join(', ')}`);
    d.append(s);
    for (const t of data.todo) {
      const cmd = `.venv\\Scripts\\python scripts\\batch_eval.py "${t.folder}" --out data\\batch\\${t.name}`;
      const row = el('div', 'todo-row'), copy = el('button', 'badge', 'copy command');
      copy.onclick = () => navigator.clipboard.writeText(cmd).then(() => { copy.textContent = 'copied'; });
      row.append(el('b', null, t.name), el('span', 'muted', ` ${t.videos} videos `), copy, el('code', null, cmd));
      d.append(row);
    }
    todo.append(d);
  }

  const f = $('#filter').value, g = $('#groups');
  g.textContent = '';
  if (!data.groups.length) g.append(el('p', 'muted', 'No batch results yet. Run: python scripts/batch_eval.py <folder> --out data/batch/<name>'));
  for (const grp of data.groups) {
    const cs = grp.clips, vv = cs.map((c) => verdict(c)[0]), k = (x) => vv.filter((y) => y === x).length;
    const sec = el('section', 'card');
    sec.append(el('h2', null, grp.group === '.' ? 'data/batch' : grp.group));
    if (grp.source) sec.append(el('p', 'muted', `source: ${grp.source}`));
    const stats = el('div', 'batch-stats');
    stats.append(el('span', 'badge', `${cs.length} clips`),
      el('span', 'badge', `falls caught ${k('hit')}/${k('hit') + k('MISSED')}`),
      el('span', `badge ${k('MISSED') ? 'alert' : ''}`, `missed ${k('MISSED')}`),
      el('span', `badge ${k('FALSE ALARM') ? 'alert' : ''}`, `false alarms ${k('FALSE ALARM')}/${k('FALSE ALARM') + k('ok')}`),
      el('span', 'badge warn', `to review ${cs.filter((c) => c.status === 'to review').length}`));
    sec.append(stats);

    const shown = cs.filter((c) => f === 'all' || c.status === f
      || (f === 'problems' && ['MISSED', 'FALSE ALARM', 'error'].includes(verdict(c)[0]))
      || (f === 'detected' && Number(c.falls) > 0));
    if (!shown.length) { sec.append(el('p', 'empty', 'Nothing matches this filter.')); g.append(sec, el('br')); continue; }

    const t = el('table', 'batch-table'), head = el('tr');
    ['Annotated video', 'Clip', 'Status', 'Detected', 'Really a fall?', 'Verdict', 'Note'].forEach((h) => head.append(el('th', null, h)));
    t.append(head);
    for (const c of shown) {
      const tr = el('tr'), cell = (x) => { const td = el('td'); td.append(x); tr.append(td); };
      const thumb = el('button', 'thumb');
      if (c.poster) { const img = el('img'); img.alt = ''; img.loading = 'lazy'; img.src = c.poster; thumb.append(img, el('span', null, '▶')); }
      else thumb.append(el('span', 'muted', 'no video yet'));
      thumb.disabled = !c.video; thumb.onclick = () => play(c, grp.group);
      cell(thumb);
      const clip = el('div'), name = el(c.video ? 'a' : 'b', null, c.file);
      if (c.video) name.href = `/r/${c.id}`;
      clip.append(name, el('div', 'muted', c.error || (c.processed_at ? `${c.duration_s || '?'} s · processed ${when(c.processed_at)}` : 'waiting')));
      cell(clip);
      cell(el('span', `badge ${STATUS_TONE[c.status]}`, c.status));
      cell(c.status === 'not processed' || c.status === 'error' ? '' : (Number(c.falls) > 0 ? `${c.falls} at ${c.fall_at_s} s` : 'none'));
      const seg = el('div', 'seg');
      for (const [txt, val] of [['Fall', 1], ['No fall', 0], ['?', null]]) {
        const b = el('button', label(c) === val ? 'on' : '', txt);
        b.disabled = !c.video;
        b.onclick = () => save(c, grp.group, { fall: val });
        seg.append(b);
      }
      cell(seg);
      const [vt, vc] = verdict(c);
      cell(el('span', `badge ${vc}`, vt));
      const note = el('input'); note.value = c.review.note || ''; note.placeholder = 'what went wrong?';
      note.onchange = () => save(c, grp.group, { note: note.value });
      cell(note);
      t.append(tr);
    }
    const wrap = el('div', 'table-wrap'); wrap.append(t);
    sec.append(wrap);
    g.append(sec, el('br'));
  }

  const h = $('#history'); h.textContent = '';
  if (!data.history.length) h.append(el('p', 'empty', 'Nothing yet.'));
  const list = el('ul', 'history');
  for (const e of data.history) {
    const li = el('li'), c = find(e.group, e.file), who = el(c?.video ? 'a' : 'b', null, `${e.group}/${e.file}`);
    if (c?.video) { who.href = '#'; who.onclick = (ev) => { ev.preventDefault(); play(c, e.group); }; }
    li.append(el('time', null, when(e.at)), el('span', `badge ${e.what === 'reviewed' ? '' : 'info'}`, e.what), who, el('span', 'muted', ` ${e.detail}`));
    list.append(li);
  }
  h.append(list);
}

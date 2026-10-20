const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const link = (href, text) => { const a = el('a', 'button-link', text); a.href = href; return a; };
let data = { days: [], uploads: [] };

async function remove(kind, id, ask = true) {
  const what = kind === 'day' ? `everything from ${id}` : 'this clip and its result';
  if (ask && !confirm(`Delete ${what}? This can't be undone.`)) return false;
  const r = await fetch(`/api/history/${kind}/${encodeURIComponent(id)}`, { method: 'DELETE' });
  if (!r.ok) alert((await r.json().catch(() => ({}))).detail || 'Could not delete.');
  return r.ok;
}

function row(media, title, sub, badges, actions) {
  const r = el('div', media ? 'hist-row' : 'hist-row no-thumb');
  if (media) r.append(media);
  const body = el('div', 'hist-body');
  body.append(el('h3', null, title), el('p', 'muted', sub));
  const b = el('div', 'hist-badges');
  for (const [cls, text] of badges) b.append(el('span', `badge ${cls}`, text));
  body.append(b);
  const act = el('div', 'hist-actions');
  act.append(...actions);
  r.append(body, act);
  return r;
}

function delButton(kind, id) {
  const b = el('button', 'danger small', 'Delete');
  b.type = 'button';
  b.onclick = async () => { if (await remove(kind, id)) load(); };
  return b;
}

function render() {
  const days = $('#days'), ups = $('#uploads');
  days.replaceChildren(); ups.replaceChildren();
  if (!data.days.length) days.append(el('p', 'empty', 'No days yet.'));
  for (const d of data.days) {
    const date = new Date(`${d.day}T12:00`).toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' });
    const badges = [];
    if (d.synthetic) badges.push(['info', 'demo data']);
    if (d.falls) badges.push(['alert', `${d.falls} fall alert${d.falls > 1 ? 's' : ''}`]);
    else if (d.alerts) badges.push(['alert', `${d.alerts} alert${d.alerts > 1 ? 's' : ''}`]);
    if (d.report_sent) badges.push(['', 'report sent']);
    const sub = [`${d.events} event${d.events === 1 ? '' : 's'}`, d.videos ? `${d.videos} video${d.videos > 1 ? 's' : ''}` : ''].filter(Boolean).join(' · ');
    days.append(row(null, date, sub, badges,
      [link(`/day?date=${d.day}`, 'Open'), link(`/live/report/${d.day}`, 'Report'), delButton('day', d.day)]));
  }
  if (!data.uploads.length) ups.append(el('p', 'empty', 'No clips yet.'));
  for (const u of data.uploads) {
    let img = null;
    if (u.poster) { img = el('img', 'hist-thumb'); img.src = u.poster; img.alt = ''; img.loading = 'lazy'; } else img = el('div', 'hist-thumb');
    const when = new Date(u.created * 1000).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
    const badges = u.status !== 'done' ? [[u.status === 'error' ? 'alert' : 'info', u.status]]
      : u.falls ? [['alert', `${u.falls} fall${u.falls > 1 ? 's' : ''}`]]
      : u.alerts ? [['warn', `${u.alerts} alert${u.alerts > 1 ? 's' : ''}`]] : [['', 'no alerts']];
    const sub = [when, u.duration ? `${Math.round(u.duration)} s` : ''].filter(Boolean).join(' · ');
    const actions = u.status === 'done' ? [link(`/r/${u.id}`, 'Open')] : [];
    if (u.status !== 'queued' && u.status !== 'processing') actions.push(delButton('upload', u.id));
    ups.append(row(img, u.name, sub, badges, actions));
  }
  $('#keep').textContent = `Clips uploaded on the home page. They are deleted automatically after ${Math.round(data.retention_hours)} hours.`;
}

async function load() {
  try { data = await (await fetch('/api/history')).json(); } catch { $('#days').replaceChildren(el('p', 'empty', 'Could not load the history.')); return; }
  render();
}

for (const b of document.querySelectorAll('[data-all]')) {
  b.onclick = async () => {
    const kind = b.dataset.all, items = kind === 'day' ? data.days.map((d) => d.day)
      : data.uploads.filter((u) => u.status !== 'queued' && u.status !== 'processing').map((u) => u.id);
    if (!items.length) return;
    if (!confirm(`Delete all ${items.length} ${kind === 'day' ? 'day(s)' : 'clip(s)'}? This can't be undone.`)) return;
    b.disabled = true;
    for (const id of items) await remove(kind, id, false);
    b.disabled = false;
    load();
  };
}
load();

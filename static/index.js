const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const fmt = (s) => { s = Math.round(s); return s < 60 ? `${s} s` : `${Math.floor(s / 60)} min ${s % 60} s`; };

fetch('/api/config').then((r) => r.json()).then((c) => {
  $('#passwrap').hidden = !c.passcode;
  $('[data-max-mb]').textContent = c.max_mb;
  $('[data-max-seconds]').textContent = c.max_seconds;
});

fetch('/api/examples').then((r) => r.json()).then((list) => {
  const g = $('#gallery');
  g.textContent = '';
  if (!list.length) g.append(el('p', 'muted', 'No examples yet.'));
  for (const ex of list) {
    const a = el('a', 'ex');
    a.href = `/r/${ex.id}`;
    const img = el('img'); img.src = ex.poster; img.alt = ''; img.loading = 'lazy';
    const body = el('div');
    body.append(el('h3', null, ex.title), el('p', null, ex.description || ''));
    const meta = el('p');
    meta.append(`${fmt(ex.duration)} · `, el('span', `badge ${ex.alerts ? 'alert' : ''}`, ex.alerts ? `${ex.alerts} alert${ex.alerts > 1 ? 's' : ''}` : 'no alerts'));
    body.append(meta);
    a.append(img, body);
    g.append(a);
  }
});

const form = $('#form'), input = form.file, drop = $('#drop');
input.addEventListener('change', () => { $('#fname').textContent = input.files[0]?.name || 'Choose a video, or drop it here'; });
['dragenter', 'dragover'].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((e) => drop.addEventListener(e, () => drop.classList.remove('over')));
drop.addEventListener('drop', (ev) => { ev.preventDefault(); input.files = ev.dataTransfer.files; input.dispatchEvent(new Event('change')); });

function status(pct, text, isError) {
  $('#status').hidden = false;
  $('#status i').style.width = `${pct}%`;
  const p = $('#status p'); p.textContent = text; p.className = isError ? 'error' : '';
}

form.addEventListener('submit', (ev) => {
  ev.preventDefault();
  $('#go').disabled = true;
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/jobs');
  xhr.upload.onprogress = (e) => e.lengthComputable && status(e.loaded / e.total * 20, `Uploading… ${Math.round(e.loaded / e.total * 100)}%`);
  xhr.onload = () => {
    let body = {};
    try { body = JSON.parse(xhr.responseText); } catch { /* non-JSON error page */ }
    if (xhr.status !== 200) { status(0, body.detail || `Upload failed (${xhr.status}).`, true); $('#go').disabled = false; return; }
    poll(body.id);
  };
  xhr.onerror = () => { status(0, 'Network error, please try again.', true); $('#go').disabled = false; };
  xhr.send(new FormData(form));
});

async function poll(id) {
  const r = await fetch(`/api/jobs/${id}`).catch(() => null);
  if (!r || !r.ok) { setTimeout(() => poll(id), 1500); return; }  // transient: retry
  const j = await r.json();
  if (j.status === 'done') { status(100, 'Done! Opening the result…'); location.href = `/r/${id}`; return; }
  if (j.status === 'error') { status(100, j.error, true); $('#go').disabled = false; return; }
  if (j.status === 'queued') status(20, `Waiting in queue (position ${j.queue_position})…`);
  else status(20 + j.progress * 0.8, `${j.stage}… ${j.progress}%`);
  setTimeout(() => poll(id), 1500);
}

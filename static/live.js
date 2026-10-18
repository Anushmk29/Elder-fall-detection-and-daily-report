const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const SHOWN = new Set(['fall_suspected', 'fall_alert', 'fall_recovered', 'lying_outside_rest', 'stillness', 'zone_enter',
  'camera_offline', 'bathroom_long', 'rest_long', 'night_activity', 'not_seen']);
const camEls = {};

function cameraCard(c) {
  const card = el('figure', 'cam');
  const img = el('img');
  img.alt = `Camera ${c.camera}`;
  img.src = `/live/${encodeURIComponent(c.camera)}.mjpg`;
  const cap = el('figcaption');
  card.append(img, cap);
  camEls[c.camera] = { card, cap, img };
  return card;
}

function render(st) {
  const status = st.running ? st.status : 'Not running';
  $('#status').textContent = status;
  $('#status').className = `pill ${status === 'ALERT' ? 'alert' : status === 'MONITOR' ? 'warn' : ''}`;
  $('#where').textContent = st.room ? `In the ${st.room.toLowerCase()}${st.posture ? ` · ${st.posture}` : ''}` :
    st.running ? 'Not seen right now' : 'Live engine is not running';
  const online = st.cameras.filter((c) => c.online).length;
  $('#sub').textContent = st.running ? `${online} of ${st.cameras.length} camera(s) online` : '';

  const cams = $('#cams');
  if (st.cameras.length && !Object.keys(camEls).length) cams.textContent = '';
  for (const c of st.cameras) {
    if (!camEls[c.camera]) cams.append(cameraCard(c));
    const { cap, card } = camEls[c.camera];
    cap.textContent = `${c.camera} · ${c.finished ? 'finished' : c.online ? `${c.fps || '…'} fps · ${c.people} person(s)` : 'offline'}` +
      (c.room ? ` · ${c.room}` : '') + (c.posture ? ` · ${c.posture}` : '');
    card.classList.toggle('off', !c.online);
  }

  const alert = st.events.find((e) => e.type === 'fall_alert' && Date.now() / 1000 - e.ts < 3600);
  $('#banner').hidden = !alert;
  if (alert) $('#banner').textContent = `${alert.clock} · ${alert.title} — ${alert.detail}`;

  const feed = $('#feed');
  const items = st.events.filter((e) => SHOWN.has(e.type)).slice(0, 40);
  feed.textContent = '';
  if (!items.length) feed.append(el('li', 'muted', 'Nothing yet.'));
  for (const e of items) {
    const li = el('li');
    li.append(el('span', 'clock', e.clock), el('span', `badge ${e.severity}`, e.title));
    if (e.detail) li.append(el('span', 'muted', ` ${e.detail}`));
    if (e.snapshot) {
      const a = el('a', null, ' snapshot');
      a.href = `/live/snapshots/${e.snapshot}`; a.target = '_blank';
      li.append(a);
    }
    feed.append(li);
  }
}

async function poll() {
  try { render(await (await fetch('/api/live/state')).json()); } catch { $('#status').textContent = 'Offline'; }
  setTimeout(poll, 1500);
}
poll();

const form = $('#play');
form.file.addEventListener('change', () => { $('#fname').textContent = form.file.files[0]?.name || 'Choose a video'; });
form.addEventListener('submit', async (ev) => {
  ev.preventDefault();
  $('#go').disabled = true;
  $('#playmsg').textContent = 'Uploading…';
  try {
    const r = await fetch('/api/live/play', { method: 'POST', body: new FormData(form) });
    const j = await r.json();
    $('#playmsg').textContent = r.ok ? `Playing as camera "${j.camera}". It may take a few seconds to appear.` : (j.detail || 'Failed.');
  } catch { $('#playmsg').textContent = 'Upload failed.'; }
  $('#go').disabled = false;
});

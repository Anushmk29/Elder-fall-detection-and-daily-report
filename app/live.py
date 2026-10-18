"""Live mode. Every camera in cameras.yaml (webcam index, stream URL or video file) runs the same perception, tracks
and rules as uploads, continuously, on a rolling window. Events go to data/live/events-YYYY-MM-DD.jsonl, the
latest annotated frame feeds an MJPEG stream, and alerts go to notify (Home Assistant + Telegram).

cameras.yaml:
  cameras:
    - {name: living, source: "http://192.168.1.21:8080/video", room: Living room, rules: videos/living.yaml}
    - {name: kitchen, source: 0, room: Kitchen}                      # webcam index
    - {name: demo, source: videos/synthetic_fall.mp4, room: Living room, loop: true}   # file played as a camera
"""
import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from . import events, notify, perception, pipeline, render, tracks

log = logging.getLogger("live")
LIVE_DIR = pipeline.DATA_DIR / "live"
SNAP_DIR = LIVE_DIR / "snapshots"
CAMERAS_PATH = Path(os.environ.get("CAMERAS_FILE", pipeline.BASE / "cameras.yaml"))
PUSH = {"alert", "warn"}  # severities sent to the family; "info" is only logged
_predict_lock = threading.Lock()  # one model shared by all cameras; Ultralytics predict isn't thread-safe


def is_file_source(src):
    return isinstance(src, str) and not src.isdigit() and "://" not in src


def open_source(src):
    if isinstance(src, int) or str(src).isdigit():
        return cv2.VideoCapture(int(src), cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    if is_file_source(src):
        p = Path(src)
        return cv2.VideoCapture(str(p if p.is_absolute() else pipeline.BASE / p))
    return cv2.VideoCapture(src)


def window_seconds(rules):
    """Long enough for the slowest rule that runs on the window, plus slack."""
    live = rules.get("live", {})
    return max(rules["lying_outside_rest_s"], rules["stillness"]["min_s"], live.get("no_recovery_s", 30)) + 30


def live_rules(path=None, overrides=None):
    """rules.yaml (+ per-camera file), with the `live:` block's thresholds replacing the upload ones."""
    rules = pipeline.load_rules(path)
    live = rules.get("live", {})
    for k in ("lying_outside_rest_s",):
        if k in live:
            rules[k] = live[k]
    if "stillness_s" in live:
        rules["stillness"] = {**rules["stillness"], "min_s": live["stillness_s"]}
    rules.update(overrides or {})
    return rules


class Camera:
    def __init__(self, cfg, engine, model):
        self.name, self.source = str(cfg["name"]), cfg["source"]
        self.room = cfg.get("room")
        self.is_file = is_file_source(self.source)
        self.loop, self.realtime = cfg.get("loop", False), cfg.get("realtime", True)
        self.rules = live_rules(cfg.get("rules"), {k: cfg[k] for k in ("source_fps",) if k in cfg})
        self.no_recovery_s = self.rules.get("live", {}).get("no_recovery_s", 30)
        self.window_s = window_seconds(self.rules)
        self.engine, self.model = engine, model
        self.window = deque()
        self.labels = {}      # raw tracker id -> (label, role), from the last evaluation
        self.falls = []       # [{"t", "state"}]: suspected -> alert | recovered
        self.logged = []      # [(type, t)] for lying/stillness de-duplication
        self.banner_until = -1.0
        self.jpeg = None
        self.state = {"camera": self.name, "online": False, "finished": False, "fps": 0.0, "people": 0,
                      "room": None, "posture": None, "updated": None}
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"cam-{self.name}")

    # ---- source loop -------------------------------------------------------------------------------------
    def _run(self):
        backoff = 2
        while not self.stop_event.is_set():
            cap = open_source(self.source)
            if not cap.isOpened():
                self._offline("could not open source")
                self.stop_event.wait(backoff)
                backoff = min(backoff * 2, 60)
                continue
            backoff = 2
            try:
                self._read(cap)
            except Exception:  # a bad frame or rule bug must not kill the camera for good
                log.exception("camera %s crashed; restarting", self.name)
            finally:
                cap.release()
            if self.is_file and not self.loop:
                self.state.update(online=False, finished=True)
                return
            if not self.is_file:
                self._offline("stream ended")
                self.stop_event.wait(backoff)

    def _offline(self, why):
        if self.state["online"] or self.state["updated"] is None:
            self.engine.emit(self, "camera_offline", "info", "Camera offline", f"{self.name}: {why}.")
        self.state.update(online=False, updated=time.time())

    def _read(self, cap):
        src_fps = self.rules.get("source_fps") or cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps != src_fps or src_fps > 240:
            src_fps = 25.0
        step = 1.0 / self.rules["sample_fps"]
        tracker, fragment_ids = perception.new_tracker()
        self.window.clear()
        self.falls, self.logged, self.labels = [], [], {}
        t0_wall, t0 = time.time(), time.monotonic()
        midnight = datetime.fromtimestamp(t0_wall).replace(hour=0, minute=0, second=0, microsecond=0)
        self.clock_start = t0_wall - midnight.timestamp()
        self.t0_wall = t0_wall
        idx, n, next_t, last_eval, fps_mark = 0, 0, 0.0, -1.0, (time.monotonic(), 0)
        while not self.stop_event.is_set():
            if not cap.grab():
                return
            if self.is_file:
                t = idx / src_fps
                idx += 1
                if self.realtime and (lag := t0 + t - time.monotonic()) > 0:
                    time.sleep(lag)
            else:
                t = time.monotonic() - t0
            if t + 1e-6 < next_t:
                continue
            next_t = max(next_t + step, t) if not self.is_file else next_t + step
            ok, frame = cap.retrieve()
            if not ok:
                continue
            with _predict_lock:
                people = perception._detect(self.model, tracker, fragment_ids, frame, self.rules)
            self.window.append({"i": n, "t": round(t, 3), "people": people})
            n += 1
            while self.window and self.window[0]["t"] < t - self.window_s:
                self.window.popleft()
            if t - last_eval >= 1.0:
                last_eval = t
                self._evaluate(t, frame)
            self._draw(frame, people, t)
            now = time.monotonic()
            if now - fps_mark[0] >= 5:
                self.state["fps"] = round((n - fps_mark[1]) / (now - fps_mark[0]), 1)
                fps_mark = (now, n)
            self.state.update(online=True, people=len(people), updated=time.time())

    # ---- rules on the rolling window ---------------------------------------------------------------------
    def meta(self, frame, t):
        return {"width": frame.shape[1], "height": frame.shape[0], "fps": self.rules["sample_fps"], "duration": t,
                "clock_start": self.clock_start}

    def _evaluate(self, t, frame):
        meta = self.meta(frame, t)
        trk = tracks.build(list(self.window), meta, self.rules)
        an = events.analyze(trk, meta, self.rules)
        self.labels = {s["id"]: (tr["label"], tr["role"]) for tr in trk for s in tr["samples"]}
        res = next((tr for tr in trk if tr["role"] == "resident"), None)
        last = res["samples"][-1] if res and t - res["samples"][-1]["t"] <= 3 else None
        rooms = {z["name"] for z in self.rules.get("zones", []) if z.get("kind", "room") == "room"}
        room = (next((z for z in last["zones"] if z in rooms), None) or self.room) if last else None
        posture = last["posture"] if last else None  # smoothed posture, not walking/standing: fewer log lines
        if room != self.state["room"]:
            if self.state["room"]:
                self.engine.emit(self, "zone_exit", "info", f"Left {self.state['room']}", "", t=t,
                                 room=self.state["room"])
            if room:
                self.engine.emit(self, "zone_enter", "info", f"Entered {room}", "", t=t, room=room, posture=posture)
            self.state["room"] = room
        if posture != self.state["posture"]:
            if posture:
                self.engine.emit(self, "posture_change", "info", posture.capitalize(), "", t=t, room=room,
                                 posture=posture)
            self.state["posture"] = posture
        for e in an["events"]:
            if e["type"] == "fall":
                self._fall(e, t, frame)
            elif not any(typ == e["type"] and abs(t0 - e["t"]) < 10 for typ, t0 in self.logged):
                self.logged.append((e["type"], e["t"]))
                sev = "alert" if e["type"] == "lying_outside_rest" else "info"
                self.engine.emit(self, e["type"], sev, e["title"], e["detail"], t=t, room=room, posture=posture,
                                 frame=frame)
        self.logged = [x for x in self.logged if x[1] > t - self.window_s]
        self.falls = [f for f in self.falls if f["t"] > t - self.window_s]

    def _fall(self, e, t, frame):
        """detect_falls hit -> suspected; still down after no_recovery_s -> ALERT; got up first -> recovered."""
        f = next((f for f in self.falls if abs(f["t"] - e["t"]) < 3), None)
        if f is None:
            f = {"t": e["t"], "state": "suspected"}
            self.falls.append(f)
            self.banner_until = t + self.no_recovery_s
            self.engine.emit(self, "fall_suspected", "info", "Possible fall",
                             f"{e['who']} went down at {events.clock({'clock_start': self.clock_start}, e['t'])}; "
                             f"waiting {self.no_recovery_s:.0f} s to see if they get up.", t=e["t"], frame=frame)
        if f["state"] != "suspected":
            return
        if e["got_up_s"] is not None:
            f["state"] = "recovered"
            self.banner_until = -1
            self.engine.emit(self, "fall_recovered", "warn", "Fall, got up",
                             f"{e['who']} fell and got up again after {e['got_up_s']:.0f} s.", t=t, frame=frame)
        elif t - e["t"] >= self.no_recovery_s:
            f["state"] = "alert"
            self.banner_until = t + 3600
            self.engine.emit(self, "fall_alert", "alert", "FALL — not getting up",
                             f"{e['who']} fell and has not got up for {t - e['t']:.0f} s. Please check now.",
                             t=t, frame=frame)

    # ---- annotated frame for the MJPEG stream ------------------------------------------------------------
    def _draw(self, frame, people, t):
        h, w = frame.shape[:2]
        s = min(1.0, 960 / w, 540 / h)
        img = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame.copy()
        zones = self.rules.get("zones", [])
        layer, mask = render._zone_layer(zones, img.shape[1], img.shape[0])
        items = []
        for p in people:
            label, role = self.labels.get(p["id"], ("Person", None))
            items.append(({"id": p["id"] if p["id"] > 0 else "?", "label": label, "role": role},
                          {**p, "activity": p["posture"]}))
        banner = [(t, t, f"FALL · {self.name}")] if t <= self.banner_until else []
        render._annotate(img, items, s, s, layer, mask, zones, banner, t, {"clock_start": self.clock_start})
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            self.jpeg = buf.tobytes()


class Engine:
    def __init__(self, cameras_path=CAMERAS_PATH, sender=notify.send):
        cfg = yaml.safe_load(Path(cameras_path).read_text(encoding="utf-8")) or {} if cameras_path else {}
        self.model = perception.load_model()
        self.sender = sender
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.recent = deque(maxlen=300)
        self.cameras = {}
        for c in cfg.get("cameras", []):
            self.add(c, start=False)
        SNAP_DIR.mkdir(parents=True, exist_ok=True)

    def add(self, cfg, start=True):
        cam = Camera(cfg, self, self.model)
        old = self.cameras.get(cam.name)
        if old:
            old.stop_event.set()
        self.cameras[cam.name] = cam
        if start:
            cam.thread.start()
        return cam

    def start(self):
        for cam in self.cameras.values():
            if not cam.thread.is_alive():
                cam.thread.start()
        threading.Thread(target=self._daily_loop, daemon=True, name="daily-report").start()

    def stop(self):
        self.stopped.set()
        for cam in self.cameras.values():
            cam.stop_event.set()

    def _daily_loop(self):
        from . import daily  # daily imports live
        while not self.stopped.wait(60):
            day = daily.due()
            if day:
                try:
                    daily.send(day)
                except Exception:
                    log.exception("daily report for %s failed", day)

    def emit(self, cam, typ, severity, title, detail, t=None, room=None, posture=None, frame=None):
        ts = cam.t0_wall + t if t is not None and hasattr(cam, "t0_wall") else time.time()
        ev = {"ts": round(ts, 1), "clock": datetime.fromtimestamp(ts).strftime("%H:%M:%S"), "camera": cam.name,
              "room": room or cam.state.get("room") or cam.room, "type": typ, "severity": severity, "title": title,
              "detail": detail, "posture": posture, "snapshot": None}
        if frame is not None and cam.jpeg is not None:
            name = f"{int(ts)}-{cam.name}-{typ}.jpg"
            (SNAP_DIR / name).write_bytes(cam.jpeg)
            ev["snapshot"] = name
        day = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        with self.lock:
            with open(LIVE_DIR / f"events-{day}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(ev) + "\n")
            self.recent.append(ev)
        log.info("%s %s %s %s", cam.name, severity, typ, detail)
        if severity in PUSH:
            threading.Thread(target=self.sender, args=(ev, SNAP_DIR), daemon=True).start()
        if typ in ("zone_enter", "posture_change", "fall_alert", "fall_recovered"):
            threading.Thread(target=self._ha_sensors, daemon=True).start()
        return ev

    def _ha_sensors(self):
        st = self.status()
        notify.ha_state("eldercare_room", st["room"] or "unknown")
        notify.ha_state("eldercare_posture", st["posture"] or "unknown")
        notify.ha_state("eldercare_status", st["status"])

    def status(self):
        cams = [dict(c.state) for c in self.cameras.values()]
        seen = [c for c in cams if c["room"]]
        latest = max(seen, key=lambda c: c["updated"] or 0) if seen else None
        alerts = [e for e in self.recent if e["severity"] == "alert" and time.time() - e["ts"] < 3600]
        return {"cameras": cams, "room": latest and latest["room"], "posture": latest and latest["posture"],
                "status": "ALERT" if alerts else "MONITOR" if any(e["severity"] == "warn" for e in self.recent
                                                                    if time.time() - e["ts"] < 3600) else "NORMAL"}


def read_events(day=None):
    """One day's event log (YYYY-MM-DD, default today)."""
    p = LIVE_DIR / f"events-{day or datetime.now().strftime('%Y-%m-%d')}.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

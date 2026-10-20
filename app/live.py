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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import events, notify, perception, pipeline, render, tracks

log = logging.getLogger("live")
LIVE_DIR = pipeline.DATA_DIR / "live"
SNAP_DIR = LIVE_DIR / "snapshots"
CAMERAS_PATH = Path(os.environ.get("CAMERAS_FILE", pipeline.BASE / "cameras.yaml"))
PUSH = {"alert", "warn"}  # severities sent to the family; "info" is only logged
_scorer = None  # the GRU (app/fall_model.py), loaded on first use when live.fall_model is on
_predict_lock = threading.Lock()  # one model shared by all cameras; Ultralytics predict isn't thread-safe
# AI review (VLM_SECOND_OPINION=1): while someone is in view, a 768 px frame every AI_STEP_S; each AI_SEGMENT_S of it
# (or until they're gone AI_GONE_S) goes to the vision model in one request (~258 tokens a frame)
AI_STEP_S, AI_SEGMENT_S, AI_GONE_S, AI_CONF, AI_WIDTH = 2.0, 30.0, 4.0, 0.5, 768


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


class LatestFrame:
    """A live stream decoded on its own thread, keeping only the newest frame. Without it a model slower than the
    camera reads an ever-older backlog from the decoder's buffer: the 'live' view drifts minutes behind."""

    def __init__(self, cap):
        self.cap, self.frame, self.seq, self.seen, self.ok = cap, None, 0, 0, True
        self.cond = threading.Condition()
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()

    def _pump(self):
        while self.ok:
            ok, frame = self.cap.read()
            with self.cond:
                self.ok = self.ok and ok
                self.frame, self.seq = frame, self.seq + 1
                self.cond.notify_all()

    def grab(self):  # waits for a frame newer than the last one handed out; False when the stream ended or stalled
        with self.cond:
            self.cond.wait_for(lambda: self.seq > self.seen or not self.ok, timeout=10)
            if not self.ok or self.seq == self.seen:
                return False
            self.seen = self.seq
            return True

    def retrieve(self):
        return True, self.frame

    def get(self, prop):
        return self.cap.get(prop)

    def release(self):
        self.ok = False
        self.thread.join(timeout=15)  # releasing while read() runs on the other thread can crash the decoder
        self.cap.release()


def _is_bath(name):
    return any(w in (name or "").lower() for w in ("bath", "toilet", "wc"))


class Watch:
    """The slow rules, across all cameras (they outlast any camera's rolling window):
    - in the bathroom (camera room or zone named bath*/toilet*/wc) for > bathroom_s -> warn, > bathroom_alert_s ->
      alert; counted from when they were first seen there, also while out of sight behind the bathroom door;
    - nobody seen on any camera for > not_seen_s during day_hours, while a camera is online -> warn.
    Each episode fires once."""
    def __init__(self, live):
        self.cfg = live
        self.last_seen, self.bath_since, self.fired = None, None, set()

    def saw(self, places, ts):
        self.last_seen = ts
        if any(_is_bath(p) for p in places):
            self.bath_since = self.bath_since or ts
        else:
            self.bath_since = None

    def check(self, now, online):
        """[(type, severity, title, detail)] newly due at `now` (unix time)."""
        out, c = [], self.cfg
        if self.bath_since is not None:
            mins = (now - self.bath_since) / 60
            for limit, sev in ((c.get("bathroom_alert_s", 2700), "alert"), (c.get("bathroom_s", 1800), "warn")):
                key = ("bath", sev, self.bath_since)
                if now - self.bath_since >= limit:
                    if key not in self.fired:
                        self.fired.add(key)
                        out.append(("bathroom_long", sev, "Long time in the bathroom",
                                    f"In the bathroom for {mins:.0f} min." + (" Please check." if sev == "alert" else "")))
                    break
        elif self.last_seen is not None and online:
            a, b = c.get("day_hours", [8, 21])
            gone = now - self.last_seen
            if (a <= datetime.fromtimestamp(now).hour < b and gone >= c.get("not_seen_s", 14400)
                    and ("gone", self.last_seen) not in self.fired):
                self.fired.add(("gone", self.last_seen))
                out.append(("not_seen", "warn", "Not seen for a while",
                            f"Nobody has been seen on any camera for {gone / 3600:.1f} h during the day."))
        return out


class Camera:
    def __init__(self, cfg, engine, model):
        self.name, self.source = str(cfg["name"]), cfg["source"]
        self.room = cfg.get("room")
        self.is_file = is_file_source(self.source)
        self.loop, self.realtime = cfg.get("loop", False), cfg.get("realtime", True)
        # a recording of an earlier day ("day replay"): its events are stamped from start_at, not from now, and
        # by default aren't pushed to the family
        self.start_at, self.notify = cfg.get("start_at"), cfg.get("notify", True)
        self.last_t = 0.0
        self.rules = live_rules(cfg.get("rules"), {k: cfg[k] for k in ("source_fps", "sample_fps") if k in cfg})
        self.no_recovery_s = self.rules.get("live", {}).get("no_recovery_s", 30)
        self.window_s = window_seconds(self.rules)
        self.engine, self.model = engine, model
        self.window = deque()
        self.labels = {}      # raw tracker id -> (label, role), from the last evaluation
        self.falls = []       # [{"t", "state"}]: suspected -> alert | recovered
        self.model_open = {}  # GRU falls: track key -> fall time
        self.logged = []      # [(type, t)] for lying/stillness de-duplication
        self.banner_until = -1.0
        self.rest_since, self.rest_flagged = None, None  # the resident's current sofa/bed (kind: rest) stretch
        self.jpeg = None
        self.shots = deque()  # (t, raw 768 px jpeg) every 0.5 s, last no_recovery_s + 15 s: the AI fall check's frames
        self.ai_frames, self.ai_seen = [], -1e9  # the current stretch with a person, for the AI review
        self.state = {"camera": self.name, "online": False, "finished": False, "fps": 0.0, "people": 0,
                      "room": None, "posture": None, "updated": None, "progress": None, "start_at": self.start_at}
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
            if not self.is_file:
                cap = LatestFrame(cap)
            try:
                self._read(cap)
            except Exception:  # a bad frame or rule bug must not kill the camera for good
                log.exception("camera %s crashed; restarting", self.name)
            finally:
                cap.release()
                self._send_review()
            if self.is_file and not self.loop:
                if self.state["room"]:  # the clip ended: they left the view (else the visit runs to the end of the day)
                    self.engine.emit(self, "zone_exit", "info", f"Left {self.state['room']}", "", t=self.last_t,
                                     room=self.state["room"])
                self.state.update(online=False, finished=True, progress=1.0, room=None, posture=None)
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
        frames_total = cap.get(cv2.CAP_PROP_FRAME_COUNT) if self.is_file else 0
        t0_wall, t0 = self.start_at or time.time(), time.monotonic()
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
            # same duplicate filter as tracks.build, so the stream and the person count show each person once
            people = tracks.dedupe(people, self.rules.get("tracks", {}).get("dedupe_cover", 1.01))
            if getattr(self.engine, "ai", None):
                self._collect(frame, people, t)
            self.window.append({"i": n, "t": round(t, 3), "people": people})
            n += 1
            self.last_t = t
            if frames_total > 0:
                self.state["progress"] = round(min(idx / frames_total, 1.0), 3)
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
        if last:
            self.engine.saw(self, last["zones"] + [room or self.room], t)
            self._rest(t, last)
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
        for e in an["events"] + self._model_falls(trk, t):
            if e["type"] == "fall":
                self._fall(e, t, frame)
            elif not any(typ == e["type"] and abs(t0 - e["t"]) < 10 for typ, t0 in self.logged):
                self.logged.append((e["type"], e["t"]))
                sev = "info"
                if e["type"] == "lying_outside_rest":
                    # on the floor is not the same as fallen: a fall alert already covers it; not moving -> alert;
                    # moving (stretching, reading on a mat) -> warn, and a vision model may escalate it
                    sev = ("info" if any(f["state"] == "alert" for f in self.falls)
                           else "warn" if e["moving"] else "alert")
                detail = e["detail"]
                if sev == "alert":  # someone down and still: the AI labels it, never cancels it (see _fall)
                    verdict, why, _ = self._ai_check(t - 5, t)
                    detail += _ai_note(verdict, why)
                ev = self.engine.emit(self, e["type"], sev, e["title"], detail, t=t, room=room,
                                      posture=posture, frame=frame)
                if e["type"] == "lying_outside_rest" and sev == "warn":
                    self.engine.second_opinion(self, ev)
        if getattr(self.engine, "ai", None):
            self._lost_check(trk, t)
        self.logged = [x for x in self.logged if x[1] > t - self.window_s]
        self.falls = [f for f in self.falls if f["t"] > t - self.window_s]
        self.model_open = {k: v for k, v in self.model_open.items() if v > t - self.window_s}

    def _small(self, frame):
        small = cv2.resize(frame, (AI_WIDTH, round(frame.shape[0] * AI_WIDTH / frame.shape[1])), interpolation=cv2.INTER_AREA)
        return cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

    def _collect(self, frame, people, t):
        # raw frames, no stick figure: drawn "lying" labels made the AI call a man bending over a scooter fallen
        if not self.shots or t - self.shots[-1][0] >= 0.5:
            self.shots.append((t, self._small(frame)))
            while self.shots[0][0] < t - self.no_recovery_s - 15:
                self.shots.popleft()
        if self.ai_frames and (t - self.ai_frames[0][0] >= AI_SEGMENT_S or t - self.ai_seen >= AI_GONE_S):
            self._send_review()
        if not any(p["conf"] >= AI_CONF for p in people):
            return
        self.ai_seen = t
        if not self.ai_frames or t - self.ai_frames[-1][0] >= AI_STEP_S:
            self.ai_frames.append((t, self.shots[-1][1] if self.shots[-1][0] == t else self._small(frame)))

    def _send_review(self):
        frames, self.ai_frames = self.ai_frames, []
        if frames:
            self.engine.review(self, frames, self.state["room"] or self.room)

    def _ai_check(self, onset, t):
        """The vision model's look at a fall before it is pushed: (verdict, why, strip jpeg); (None, '', None) when
        the AI is off. Frames ~1 s before the fall, 2 s after and now, side by side."""
        ai = getattr(self.engine, "ai", None)
        if not ai or not self.shots:
            return None, "", None
        imgs = [cv2.imdecode(np.frombuffer(min(self.shots, key=lambda s: abs(s[0] - x))[1], np.uint8), cv2.IMREAD_COLOR)
                for x in (onset - 1, onset + 2, t)]
        strip = cv2.hconcat([cv2.resize(i, (640, round(i.shape[0] * 640 / i.shape[1]))) for i in imgs])
        jpeg = cv2.imencode(".jpg", strip, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        path = SNAP_DIR / f"{int(getattr(self, 't0_wall', time.time()) + t)}-{self.name}-check.jpg"
        path.write_bytes(jpeg)
        verdict, why = ai.fall_check(path)
        return verdict, why, jpeg

    def _model_falls(self, trk, t):
        """Second fall signal (rules.yaml live.fall_model): the GRU scores each person's last 3 s. A hit becomes a
        fall at t-1 s (as in training; the stay-down check starts there) once the person is still
        down after the checkpoint's confirm_s, so the model alone never pushes "fall, got up". It then runs through
        the same suspected -> 30 s -> ALERT state machine as the rules; a rules fall within 10 s wins."""
        if not self.rules.get("live", {}).get("fall_model"):
            return []
        global _scorer
        if _scorer is None:
            from .fall_model import Scorer
            _scorer = Scorer()
        from .fall_model import got_up
        out = []
        for tr in trk:
            S, key = tr["samples"], (tr["samples"][0]["t"], tr["samples"][0]["id"])
            ft = self.model_open.get(key)
            if ft is None:
                if any(abs(f["t"] - (t - 1)) < 10 for f in self.falls) or _scorer.p(tr, t) < _scorer.thr:
                    continue
                ft = self.model_open[key] = t - 1.0  # as in training (train_fall_model.events)
            up = got_up(S, ft, t - ft)
            if t - ft >= _scorer.confirm_s and not (up and not any(abs(f["t"] - ft) < 3 for f in self.falls)):
                out.append({"type": "fall", "t": ft, "who": tr["label"], "got_up_s": round(t - ft, 1) if up else None})
        return out

    def _lost_check(self, trk, t):
        """AI net for falls the pose model can't follow (far away, fisheye, frame edge: a person on the ground is often
        not detected at all). Someone tracked for >= 2 s has been out of sight for vanish_s, and nobody is seen
        standing -> the vision model looks at the current frame. Once per person; only with VLM_SECOND_OPINION=1.
        ponytail: also fires when someone just walks out of view (~15-45 asks/h of busy footage on Le2i/MCFD, cents a
        day, capped by MAX_LLM_USD_PER_DAY); skip edge exits if the cap gets hit."""
        gap = self.rules["fall"].get("vanish_s", 3)
        if any(tr["samples"][-1]["t"] >= t - 0.5 and tr["samples"][-1]["posture"] == "standing" for tr in trk):
            return
        for tr in trk:
            s0, s1 = tr["samples"][0], tr["samples"][-1]
            if (gap <= t - s1["t"] < gap + 5 and s1["t"] - s0["t"] >= 2
                    and not any(typ == "lost" and abs(t0 - s1["t"]) < 10 for typ, t0 in self.logged)):
                self.logged.append(("lost", s1["t"]))
                if self.jpeg is None:
                    return
                name = f"{int(getattr(self, 't0_wall', time.time()) + t)}-{self.name}-lost.jpg"  # not logged: walking
                (SNAP_DIR / name).write_bytes(self.jpeg)                                         # out of view is normal
                ev = {"snapshot": name, "detail": f"{tr['label']} dropped out of sight."}
                self.engine.second_opinion(self, ev, self.engine.ai_prompts[1], "Possible fall (AI check)")
                return

    def _rest(self, t, last):
        """Resident on the sofa/bed (kind: rest) for > rest_long_s between rest_long_hours -> one MONITOR warning."""
        rest = {z["name"] for z in self.rules.get("zones", []) if z.get("kind") == "rest"}
        on = set(last["zones"]) & rest
        if not on:
            self.rest_since = None
            return
        live = self.rules.get("live", {})
        self.rest_since = self.rest_since if self.rest_since is not None else t
        a, b = live.get("rest_long_hours", [9, 20])
        if (t - self.rest_since >= live.get("rest_long_s", 7200) and self.rest_flagged != self.rest_since
                and a <= datetime.fromtimestamp(self.t0_wall + t).hour < b):
            self.rest_flagged = self.rest_since
            self.engine.emit(self, "rest_long", "warn", "Long time on the sofa",
                             f"On the {sorted(on)[0].lower()} for {(t - self.rest_since) / 3600:.1f} h "
                             f"without getting up.", t=t)

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
        if f["state"] != "suspected" or (e["got_up_s"] is None and t - e["t"] < self.no_recovery_s):
            return
        # about to push: the AI looks first. A clear NO stops only "fell, got up" (eval_live --vlm: it stopped 7/7 false
        # ones, but would also have cancelled 10/137 real stay-down alerts, below the 95 % rule) -> a FALL ALERT is
        # only labelled, never cancelled. Unsure, error, no key, cap -> push as before.
        verdict, why, strip = self._ai_check(e["t"], t)
        if verdict == "NO" and e["got_up_s"] is not None:
            f["state"] = "dismissed"
            self.banner_until = -1
            self.engine.emit(self, "fall_dismissed", "info", "Not a fall (AI check)",
                             f"{e['who']} looked like a fall at {events.clock({'clock_start': self.clock_start}, e['t'])}."
                             f" AI: {why or 'not a fall'}.", t=t, jpeg=strip)
            return
        note = _ai_note(verdict, why)
        if e["got_up_s"] is not None:
            f["state"] = "recovered"
            self.banner_until = -1
            self.engine.emit(self, "fall_recovered", "warn", "Fall, got up",
                             f"{e['who']} fell and got up again after {e['got_up_s']:.0f} s.{note}", t=t, frame=frame)
        else:
            f["state"] = "alert"
            self.banner_until = t + 3600
            self.engine.emit(self, "fall_alert", "alert", "FALL — not getting up",
                             f"{e['who']} fell and has not got up for {t - e['t']:.0f} s. Please check now.{note}",
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


def _ai_note(verdict, why):
    words = {"FALLEN": " AI check: looks like a fall", "UNSURE": " AI check: unsure", "NO": " AI check: not a fall"}
    return words.get(verdict, "") + (f" ({why})." if verdict and why else "." if verdict else "")


def load_cameras(path=CAMERAS_PATH):
    return (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).get("cameras", []) if path else []


class Engine:
    def __init__(self, cameras_path=CAMERAS_PATH, sender=notify.send):
        self.model = perception.load_model()
        self.sender = sender
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.recent = deque(maxlen=300)
        self.watch = Watch(pipeline.load_rules().get("live", {}))
        self.ai = None
        if os.environ.get("VLM_SECOND_OPINION") == "1":  # sends a home camera frame to the cloud: opt-in only
            from . import llm
            self.ai = llm.LLM(pipeline.DATA_DIR / "llm")
            self.ai_prompts = (llm.FLOOR_PROMPT, llm.FALL_CHECK_PROMPT)
            self.pool = ThreadPoolExecutor(4, thread_name_prefix="ai-review")  # reviews never block a camera
        self.cameras = {}
        for c in load_cameras(cameras_path):
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
        threading.Thread(target=self._watch_loop, daemon=True, name="watch").start()

    def stop(self):
        self.stopped.set()
        for cam in self.cameras.values():
            cam.stop_event.set()

    def saw(self, cam, places, t):
        if cam.start_at:  # ponytail: a replayed day's times are in the past, so the now-based bathroom/not-seen
            return        # checks would misfire; replays skip them (the daily report still counts bathroom visits)
        self.last_cam = cam
        self.watch.saw(places, cam.t0_wall + t)

    def second_opinion(self, cam, ev, prompt=None, title="On the floor, likely fallen"):
        """Escalate-only: the rules aren't sure (on the floor but moving, or someone dropped out of sight). A vision
        model looks at the snapshot; if it says FALLEN the family gets an alert. It can never cancel or downgrade a
        rule's alert."""
        if not self.ai or not ev.get("snapshot"):
            return

        def ask():
            if self.ai.floor_verdict(SNAP_DIR / ev["snapshot"], prompt or self.ai_prompts[0]) == "FALLEN":
                self.emit(cam, "floor_ai_alert", "alert", title,
                          f"{ev['detail']} An AI check of the image says this looks like a fall. Please check.")
        threading.Thread(target=ask, daemon=True).start()

    def review(self, cam, frames, room):
        """AI review of one stretch with a person (frames 2 s apart): logged as ai_review (the daily report takes
        postures and activities from these). FALLEN with no fall from the local model -> alert: the pose model often
        doesn't see someone lying on the ground at all."""
        if not self.ai:
            return

        def run():
            r = self.ai.review([j for _, j in frames], room)
            if not r:
                return
            t0, t1 = frames[0][0], frames[-1][0]
            self.emit(cam, "ai_review", "info", (r["activity"] or "Person seen").capitalize(), r["why"], t=t0,
                      room=room, extra={"ai": {**r, "frames": len(frames), "span_s": round(t1 - t0 + AI_STEP_S, 1)}})
            if r["fall"] == "FALLEN" and not any(t0 - 10 <= f["t"] <= t1 + 5 for f in list(cam.falls)):
                self.emit(cam, "floor_ai_alert", "alert", "Possible fall (AI check)",
                          f"The AI review of this camera says someone may have fallen ({r['why'] or 'on the ground'}). "
                          f"Please check.", t=t1, room=room, jpeg=frames[len(frames) // 2][1])
        self.pool.submit(run)

    def _watch_loop(self):
        from . import daily  # daily imports live
        while not self.stopped.wait(30):
            online = any(c.state["online"] for c in self.cameras.values())
            cam = getattr(self, "last_cam", None) or next(iter(self.cameras.values()), None)
            for typ, sev, title, detail in self.watch.check(time.time(), online):
                if cam:
                    self.emit(cam, typ, sev, title, detail)
            day = daily.due()
            if day:
                try:
                    daily.send(day)
                except Exception:
                    log.exception("daily report for %s failed", day)

    def emit(self, cam, typ, severity, title, detail, t=None, room=None, posture=None, frame=None, jpeg=None,
             extra=None):
        ts = cam.t0_wall + t if t is not None and hasattr(cam, "t0_wall") else time.time()
        ev = {"ts": round(ts, 1), "clock": datetime.fromtimestamp(ts).strftime("%H:%M:%S"), "camera": cam.name,
              "room": room or cam.state.get("room") or cam.room, "type": typ, "severity": severity, "title": title,
              "detail": detail, "posture": posture, "snapshot": None, **(extra or {})}
        jpeg = jpeg if jpeg is not None else cam.jpeg if frame is not None else None
        if jpeg is not None:
            name = f"{int(ts)}-{cam.name}-{typ}.jpg"
            (SNAP_DIR / name).write_bytes(jpeg)
            ev["snapshot"] = name
        day = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        with self.lock:
            with open(LIVE_DIR / f"events-{day}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(ev) + "\n")
            self.recent.append(ev)
        log.info("%s %s %s %s", cam.name, severity, typ, detail)
        if severity in PUSH and cam.notify:
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
    """One day's event log (YYYY-MM-DD, default today), in time order (cameras, and replayed clips, write concurrently)."""
    p = LIVE_DIR / f"events-{day or datetime.now().strftime('%Y-%m-%d')}.jsonl"
    if not p.exists():
        return []
    return sorted((json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()),
                  key=lambda e: e["ts"])

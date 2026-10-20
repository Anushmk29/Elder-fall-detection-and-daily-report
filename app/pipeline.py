"""video -> perception -> tracks -> events -> render -> LLM -> result.json. Used by the job worker and scripts."""
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from . import events, llm, perception, render, tracks

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE / "data"))
RULES_PATH = BASE / "rules.yaml"


def load_rules(path=None):
    """rules.yaml, with top-level keys from a per-camera file (e.g. its zones) overriding it."""
    rules = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
    if path:
        rules.update(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    return rules


def probe(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                       capture_output=True, text=True, timeout=30)
    if p.returncode:
        raise ValueError("Not a readable video file")
    info = json.loads(p.stdout)
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if not v:
        raise ValueError("No video stream found")
    fmt = info.get("format", {})
    return {"format": fmt.get("format_name", ""), "duration": float(fmt.get("duration") or v.get("duration") or 0)}


def _clock_start(rules, clock):
    """Seconds since midnight at the start of the clip: --clock, else clock_start from the rules."""
    h, m, *sec = (int(x) for x in (clock or rules.get("clock_start", "09:00:00")).split(":"))
    return h * 3600 + m * 60 + (sec[0] if sec else 0)


def caption_times(evts, duration):
    n = int(max(1, min(12, max(8, len(evts)), duration // 2)))  # 8-12, but at most one per 2 s of video
    times = [e["t"] for e in evts][:n]
    k = n - len(times)
    for j in range(k):
        t = duration * (j + 0.5) / k
        if all(abs(t - x) > 1.5 for x in times):
            times.append(t)
    return sorted(round(t, 2) for t in times)


def ai_fall_check(ai, out_dir, captions, analysis, meta):
    """Escalate-only safety net: the pose model often loses people once they're on the ground (far away, fisheye,
    frame edge), so the rules see no fall. Ask the vision model about the last two strips (where a fallen person
    would still be lying); FALLEN adds an alert. It never removes anything. No key -> no call."""
    for c in captions[-2:]:
        if ai.floor_verdict(out_dir / c["img"], llm.FALL_CHECK_PROMPT) == "FALLEN":
            evts = analysis["events"]
            evts.append({"id": len(evts) + 1, "type": "ai_fall", "t": c["t"], "clock": c["clock"], "severity": "alert",
                         "who": "Someone", "thumb": c["img"], "title": "Possible fall (AI check)",
                         "detail": "The pose rules saw no fall, but an AI check of this moment says someone looks "
                                   "fallen. Please watch the video."})
            g = next(x for x in analysis["glance"] if x["label"] == "Alerts")
            g.update(value=str(int(g["value"]) + 1), note=g["note"] + ", 1 by AI check", tone="alert")
            return


def run(video, out_dir, progress=lambda pct, stage: None, rules=None, clock=None, title=None):
    rules = rules or load_rules()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe(video)  # fail early on unreadable files
    progress(1, "Finding people and poses")
    frames, meta = perception.analyze(video, rules, lambda f: progress(1 + 64 * f, "Finding people and poses"))
    meta["clock_start"] = _clock_start(rules, clock)
    trk = tracks.build(frames, meta, rules)
    analysis = events.analyze(trk, meta, rules)
    evts = analysis["events"]
    ctimes = caption_times(evts, meta["duration"])
    poster_t = evts[0]["t"] + 0.5 if evts else meta["duration"] / 2
    progress(65, "Drawing the annotated video")
    render.render(video, out_dir, frames, trk, analysis, rules, meta,
                  thumbs=[(e["t"] + 0.3, e["thumb"]) for e in evts] + [(poster_t, "poster.jpg")],
                  strips=[(t, f"strip{n}.jpg") for n, t in enumerate(ctimes, 1)],
                  progress=lambda f: progress(65 + 20 * f, "Drawing the annotated video"))

    progress(85, "Writing the summary")
    ai = llm.LLM(DATA_DIR / "llm")
    contexts = [events.describe_at(trk, t) for t in ctimes]
    with ThreadPoolExecutor(4) as ex:
        caps = list(ex.map(lambda a: ai.caption(out_dir / f"strip{a[0]}.jpg", a[1]), enumerate(contexts, 1)))
    captions = [{"t": t, "clock": events.clock(meta, t), "img": f"strip{n}.jpg", **c}
                for n, (t, c) in enumerate(zip(ctimes, caps), 1)]
    if not any(e["type"] == "fall" for e in evts):
        ai_fall_check(ai, out_dir, captions, analysis, meta)
    facts = {"glance": analysis["glance"], "people": analysis["people"],
             "events": [{k: e[k] for k in ("clock", "severity", "title", "detail")} for e in evts],
             "moments": [{"clock": c["clock"], "text": c["text"]} for c in captions]}
    report = ai.report(facts) if analysis["people"] else {  # never reassure about a clip where nobody was seen
        "text": "Nobody was detected in this clip, so nothing could be checked. Make sure the person is in view of "
                "the camera and the room is lit.", "ai": False}

    result = {"app": "eldercare", "title": title or "Uploaded clip", "created": int(time.time()),
              "duration": meta["duration"], "video": "annotated.mp4", "poster": "poster.jpg",
              "clock_start": events.clock(meta, 0), **analysis, "captions": captions, "report": report}
    (out_dir / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    progress(100, "Done")
    return result

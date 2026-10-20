"""OpenRouter (OpenAI-compatible) captions + report, with disk cache, daily USD cap and template fallback."""
import base64
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path

import httpx

API_URL = "https://openrouter.ai/api/v1/chat/completions"
UNAVAILABLE = "AI text unavailable"
_ledger_lock = threading.Lock()
log = logging.getLogger("llm")

CAPTION_PROMPT = (
    "You see three frames from a home camera, one second apart (left to right). "
    "In one short, neutral sentence describe what the people are doing and where. "
    "Do not guess identity, age, health conditions or emotions. Rule-based detections for this moment: {context}")
REPORT_PROMPT = (
    "You write a short daily note for the family and caregiver of an older person living at home, "
    "based only on automated video-analysis facts below (pose detection + simple rules, may contain errors). "
    "Warm, plain English, no medical diagnosis, no speculation beyond the facts. Format:\n"
    "A 3-5 sentence summary of the day.\n\nThings to check:\n- bullet\n- bullet\n"
    "(1-4 bullets; if no alert was found say so in one bullet, and that the camera can miss people who are far away, "
    "at the edge of the view or lying down. Never say everything is fine.)\n\nFacts (JSON):\n{facts}")
DAILY_NOTE_PROMPT = (
    "Below is today's automated routine summary for an older person living at home (camera pose detection and "
    "simple rules; it may contain errors). Write 1-2 warm, plain sentences for their family that mention only what "
    "is marked ⚠️, or say the day looks normal. No diagnosis, no facts that are not in the summary.\n\n{summary}")


FLOOR_PROMPT = (
    "A home camera shows an older person on the floor, outside the bed and sofa. Does this look like someone who "
    "has fallen or cannot get up, or like someone on the floor on purpose (exercise, sitting or resting on a mat, "
    "playing, looking for something)? Answer with exactly one word: FALLEN, DELIBERATE or UNSURE.")
FALL_CHECK_PROMPT = (
    "Three frames from a camera, left to right in time. Is anyone lying or sitting on the ground in a way that looks "
    "like they have fallen? Answer with exactly one word: FALLEN, NO or UNSURE.")
LOOK_ALIKES = ("Things that look like a fall to the automatic detector but are not: sweeping, bending, picking something "
               "up, crouching, gardening, sitting on a step or chair, walking out of view.")
FALL_VERIFY_PROMPT = (
    "Three frames from a home camera, left to right in time (about 1 s before, 2 s after, and some seconds after a "
    "possible fall). Boxes and stick figures are drawn by an automatic detector and can be wrong. " + LOOK_ALIKES +
    " Someone who really fell may be partly hidden or lying at the edge of the view. Has someone fallen and is on, "
    "or was on, the ground? Answer FALLEN, NO or UNSURE, then ' - ' and at most 8 words why.")
REVIEW_PROMPT = (
    "{n} frames from a home camera ({room}), 2 s apart, in time order. " + LOOK_ALIKES +
    " Someone who really fell may be partly hidden or lying at the edge of the view. Reply with JSON only: "
    '{{"people": <most people visible in one frame>, "postures": {{"standing": <frames>, "walking": <frames>, '
    '"sitting": <frames>, "bending": <frames>, "lying": <frames>}}, "activity": "<what the main person is doing, at '
    'most 5 words>", "fall": "FALLEN" | "NO" | "UNSURE", "why": "<at most 8 words>"}}. Posture values are whole numbers '
    "of frames; each frame is counted under exactly one posture, for the main person; frames with nobody count nowhere.")
POSTURES = ("standing", "walking", "sitting", "bending", "lying")


class LLM:
    def __init__(self, data_dir, client=None):
        self.dir = Path(data_dir)
        (self.dir / "cache").mkdir(parents=True, exist_ok=True)
        self.key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        self.model = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash")
        self.cap = float(os.environ.get("MAX_LLM_USD_PER_DAY", "2"))
        self.client = client or httpx.Client(timeout=60)

    def chat(self, messages, max_tokens=400, extra=None):
        """Returns text, or None when there is no key, the daily cap is reached, or the call fails."""
        if not self.key:
            return None
        ck = hashlib.sha256(json.dumps([self.model, messages, max_tokens, extra]).encode()).hexdigest()
        cached = self.dir / "cache" / f"{ck}.json"
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))["text"]
        if self.spent_today() >= self.cap:
            log.warning("LLM daily cap reached")
            return None
        try:
            r = self.client.post(API_URL, headers={"Authorization": f"Bearer {self.key}", "X-Title": "ElderCare demo"},
                                 json={"model": self.model, "messages": messages, "max_tokens": max_tokens,
                                       "temperature": 0.3, "usage": {"include": True}, **(extra or {})})
            r.raise_for_status()
            data = r.json()
            text = (data["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:  # network, HTTP, malformed body: all fall back to templates
            log.warning("LLM call failed: %s", e)
            return None
        if not text:
            return None
        self._add_spend(_cost(data.get("usage") or {}))
        cached.write_text(json.dumps({"text": text}), encoding="utf-8")
        return text

    def spent_today(self):
        try:
            return json.loads((self.dir / "spend.json").read_text()).get(time.strftime("%Y-%m-%d"), 0.0)
        except (OSError, ValueError):
            return 0.0

    def _add_spend(self, usd):
        with _ledger_lock:
            p, day = self.dir / "spend.json", time.strftime("%Y-%m-%d")
            try:
                ledger = json.loads(p.read_text())
            except (OSError, ValueError):
                ledger = {}
            ledger = {day: ledger.get(day, 0.0) + usd}  # only today matters
            p.write_text(json.dumps(ledger))

    def caption(self, strip_path, context):
        b64 = base64.b64encode(Path(strip_path).read_bytes()).decode()
        text = self.chat([{"role": "user", "content": [
            {"type": "text", "text": CAPTION_PROMPT.format(context=context)},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}], max_tokens=80)
        return {"text": text, "ai": True} if text else {"text": context[:1].upper() + context[1:] + ".", "ai": False}

    def floor_verdict(self, image_path, prompt=FLOOR_PROMPT):
        """'FALLEN' | 'DELIBERATE' | 'NO' | 'UNSURE' | None (no key, cap reached, failure, or an unclear answer)."""
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        text = self.chat([{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}], max_tokens=5)
        word = (text or "").strip().strip(".").upper()
        return word if word in ("FALLEN", "DELIBERATE", "NO", "UNSURE") else None

    def fall_check(self, image_path):
        """Second look at a fall the local model is about to push: ('FALLEN' | 'NO' | 'UNSURE' | None, reason)."""
        text = self.chat([{"role": "user", "content": [
            {"type": "text", "text": FALL_VERIFY_PROMPT}, _image(Path(image_path).read_bytes())]}], 40, NO_THINKING)
        word, _, why = (text or "").partition("-")
        word = word.strip().strip(".").upper()
        return (word if word in ("FALLEN", "NO", "UNSURE") else None), why.strip().rstrip(".")[:80]

    def review(self, jpegs, room):
        """One stretch of footage with a person (frames 2 s apart) -> {"people", "postures", "activity", "fall", "why"}
        or None (no key, cap, failure, unreadable answer)."""
        text = self.chat([{"role": "user", "content": [
            {"type": "text", "text": REVIEW_PROMPT.format(n=len(jpegs), room=room or "a room")},
            *map(_image, jpegs)]}], 150, NO_THINKING)
        try:
            d = json.loads((text or "").strip().removeprefix("```json").strip("`").strip())
            got = d.get("postures") or {}
            postures = {k: len(v) if isinstance(v, list) else max(int(v or 0), 0)  # some answers list frame numbers
                        for k, v in ((k, got.get(k)) for k in POSTURES)}
            fall = str(d.get("fall", "")).upper()
            return {"people": int(d.get("people") or 0), "postures": postures,
                    "activity": str(d.get("activity") or "")[:60], "fall": fall if fall in ("FALLEN", "NO") else "UNSURE",
                    "why": str(d.get("why") or "")[:80]}
        except (ValueError, TypeError, AttributeError):
            return None

    def daily_note(self, summary):
        """1-2 sentences for the family on top of the rule-based daily summary, or None (no key/cap/failure)."""
        return self.chat([{"role": "user", "content": DAILY_NOTE_PROMPT.format(summary=summary)}], 150)

    def report(self, facts):
        text = self.chat([{"role": "user", "content": REPORT_PROMPT.format(facts=json.dumps(facts, indent=1))}], 500)
        if text:
            return {"text": text, "ai": True, "model": self.model}
        return {"text": template_report(facts), "ai": False, "note": UNAVAILABLE}


NO_THINKING = {"reasoning": {"enabled": False}}  # one-word / small-JSON answers: thinking tokens only add cost


def _image(jpeg):
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}


def _cost(usage):
    if "cost" in usage:
        return float(usage["cost"])
    per_mtok = float(os.environ.get("LLM_USD_PER_MTOK_ESTIMATE", "2.5"))  # used only if OpenRouter omits cost
    return usage.get("total_tokens", 0) * per_mtok / 1e6


def template_report(facts):
    g = {t["label"]: t["value"] for t in facts.get("glance", [])}
    lines = [f"Over {g.get('Observed', 'the clip')} the resident was active for {g.get('Active', '0 s')}, "
             f"sitting for {g.get('Sitting', '0 s')} and lying for {g.get('Lying', '0 s')}."]
    if g.get("Visitors", "0") != "0":
        lines[0] += f" {g['Visitors']} other person(s) were seen."
    lines += ["", "Things to check:"]
    alerts = [e for e in facts.get("events", []) if e["severity"] != "info"]
    lines += [f"- {e['clock']} {e['title']}: {e['detail']}" for e in alerts] or ["- No fall detected. The camera can miss people who are far away, at the edge of the view or lying down."]
    return "\n".join(lines)

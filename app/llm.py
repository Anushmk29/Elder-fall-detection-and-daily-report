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
    "(1-4 bullets; if nothing needs attention say so in one bullet.)\n\nFacts (JSON):\n{facts}")
DAILY_NOTE_PROMPT = (
    "Below is today's automated routine summary for an older person living at home (camera pose detection and "
    "simple rules; it may contain errors). Write 1-2 warm, plain sentences for their family that mention only what "
    "is marked ⚠️, or say the day looks normal. No diagnosis, no facts that are not in the summary.\n\n{summary}")


class LLM:
    def __init__(self, data_dir, client=None):
        self.dir = Path(data_dir)
        (self.dir / "cache").mkdir(parents=True, exist_ok=True)
        self.key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        self.model = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash")
        self.cap = float(os.environ.get("MAX_LLM_USD_PER_DAY", "2"))
        self.client = client or httpx.Client(timeout=60)

    def chat(self, messages, max_tokens=400):
        """Returns text, or None when there is no key, the daily cap is reached, or the call fails."""
        if not self.key:
            return None
        ck = hashlib.sha256(json.dumps([self.model, messages, max_tokens]).encode()).hexdigest()
        cached = self.dir / "cache" / f"{ck}.json"
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))["text"]
        if self.spent_today() >= self.cap:
            log.warning("LLM daily cap reached")
            return None
        try:
            r = self.client.post(API_URL, headers={"Authorization": f"Bearer {self.key}", "X-Title": "ElderCare demo"},
                                 json={"model": self.model, "messages": messages, "max_tokens": max_tokens,
                                       "temperature": 0.3, "usage": {"include": True}})
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

    def daily_note(self, summary):
        """1-2 sentences for the family on top of the rule-based daily summary, or None (no key/cap/failure)."""
        return self.chat([{"role": "user", "content": DAILY_NOTE_PROMPT.format(summary=summary)}], 150)

    def report(self, facts):
        text = self.chat([{"role": "user", "content": REPORT_PROMPT.format(facts=json.dumps(facts, indent=1))}], 500)
        if text:
            return {"text": text, "ai": True, "model": self.model}
        return {"text": template_report(facts), "ai": False, "note": UNAVAILABLE}


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
    lines += [f"- {e['clock']} {e['title']}: {e['detail']}" for e in alerts] or ["- Nothing needed attention in this clip."]
    return "\n".join(lines)

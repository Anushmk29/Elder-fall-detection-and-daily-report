"""Build (and optionally send) the daily report from the live event log.

Usage: python scripts/daily_report.py [--date YYYY-MM-DD] [--send]
"""
import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import daily, llm, pipeline  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--date", default=date.today().isoformat())
ap.add_argument("--send", action="store_true", help="send to Telegram / Home Assistant too")
a = ap.parse_args()
if a.send:
    print(daily.send(a.date))
else:
    text, m, base, flags = daily.report(a.date, llm.LLM(pipeline.DATA_DIR / "llm"))
    print(text)
    print("\nflags:", flags)

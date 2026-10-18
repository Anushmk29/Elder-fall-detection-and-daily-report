"""Write 7 synthetic past days (marked "synthetic": true) so the baseline comparison can be shown on day one.
Real days overwrite nothing: existing daily files are kept unless --force.

Usage: python scripts/seed_history.py [--date YYYY-MM-DD] [--force]
"""
import argparse
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import daily  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--date", default=date.today().isoformat(), help="the day to build a baseline for")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
rnd = random.Random(7)
daily.DAILY_DIR.mkdir(parents=True, exist_ok=True)
d0 = date.fromisoformat(a.date)
for k in range(1, 8):
    p = daily.DAILY_DIR / f"{d0 - timedelta(days=k)}.json"
    if p.exists() and not a.force:
        print("kept", p.name)
        continue
    p.write_text(json.dumps({
        "date": str(d0 - timedelta(days=k)), "observed": True, "synthetic": True,
        "wake_min": rnd.randint(390, 435), "bed_min": rnd.randint(1260, 1320),
        "kitchen_visits": rnd.choice([3, 3, 4]), "bathroom_count": rnd.randint(5, 7),
        "bathroom_longest_min": rnd.randint(6, 12), "active_min": rnd.randint(220, 270),
        "rest_longest_min": rnd.randint(50, 90), "night_episodes": rnd.choice([0, 1, 1]),
    }, indent=1), encoding="utf-8")
    print("wrote", p.name)

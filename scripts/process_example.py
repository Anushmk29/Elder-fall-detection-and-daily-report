"""Run the full pipeline on a video and publish it as an example: examples/<slug>/.

Usage: python scripts/process_example.py videos/kitchen.mp4 --slug kitchen --title "Morning in the kitchen"
         [--description "..."] [--rules videos/kitchen.yaml] [--clock 08:30]
"""
import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from app import pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--slug", required=True, help="lowercase letters, digits and dashes")
    ap.add_argument("--title", required=True)
    ap.add_argument("--description", default="")
    ap.add_argument("--rules", help="per-camera yaml; its top-level keys (zones, thresholds) override rules.yaml")
    ap.add_argument("--clock", help="clock time at the start of the clip, HH:MM[:SS]")
    a = ap.parse_args()
    out = BASE / "examples" / a.slug
    result = pipeline.run(Path(a.video), out, lambda p, s: print(f"\r{p:5.1f}% {s:<40}", end="", flush=True),
                          rules=pipeline.load_rules(a.rules), clock=a.clock, title=a.title)
    result["description"] = a.description
    (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\n{out}: {len(result['events'])} events, report ai={result['report']['ai']}")
    for e in result["events"]:
        print(f"  {e['clock']} [{e['severity']}] {e['title']}: {e['detail']}")


if __name__ == "__main__":
    main()

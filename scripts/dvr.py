"""Pull a day's recordings off the Hikvision DVR (cameras.yaml has the address and login).

  python scripts/dvr.py list 2026-10-18                  motion minutes per camera per hour
  python scripts/dvr.py fetch 2026-10-18 [--from 7 --to 21] [--cams living,stairs]
        -> data/dvr/<day>/<cam>/<HHMMSS>.mp4, one file per motion run (back-to-back DVR segments are joined)

The DVR records on motion (VMD), so the empty hours are never downloaded. Files come over ISAPI download (~5 MB/s
for all cameras together); RTSP playback is real time at best and doesn't stop at endtime. Times are the DVR's
clock (it writes local time with a 'Z'; ours runs ~4.5 min fast).
"""
import argparse
import re
import subprocess
import uuid
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import requests
from xml.sax.saxutils import escape
import yaml
from requests.auth import HTTPDigestAuth

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "data" / "dvr"
NS = {"h": "http://www.hikvision.com/ver20/XMLSchema"}
MERGE_S = 2    # the DVR cuts one long motion into back-to-back segments; real gaps stay separate files


def cameras(only=None):
    out = []
    for c in yaml.safe_load((BASE / "cameras.yaml").read_text())["cameras"]:
        u = urllib.parse.urlparse(c["source"])
        ch = re.search(r"Channels/(\d+)", u.path).group(1)
        if not only or c["name"] in only:
            out.append({**c, "host": u.hostname, "user": urllib.parse.unquote(u.username),
                        "pw": urllib.parse.unquote(u.password), "track": ch[:-2] + "01"})  # recordings are main stream
    return out


def search(cam, day, h0=0, h1=24):
    """[(start, end, playbackURI)] of the DVR's recorded segments for this camera on `day` (main stream: the
    sub-stream isn't recorded, searching x02 returns x01)."""
    a, b, out, pos = f"{day}T{h0:02d}:00:00Z", f"{day}T{min(h1, 23):02d}:{'59:59' if h1 >= 24 else '00:00'}Z", [], 0
    while True:
        body = (f'<?xml version="1.0" encoding="UTF-8"?><CMSearchDescription><searchID>{uuid.uuid4()}</searchID>'
                f'<trackList><trackID>{cam["track"]}</trackID></trackList><timeSpanList><timeSpan><startTime>{a}'
                f'</startTime><endTime>{b}</endTime></timeSpan></timeSpanList><maxResults>400</maxResults>'
                f'<searchResultPostion>{pos}</searchResultPostion></CMSearchDescription>')
        r = requests.post(f"http://{cam['host']}/ISAPI/ContentMgmt/search", data=body, timeout=30,
                          auth=HTTPDigestAuth(cam["user"], cam["pw"]))
        r.raise_for_status()
        x = ET.fromstring(r.content)
        items = x.findall(".//h:searchMatchItem", NS)
        for it in items:
            s, e = (datetime.strptime(it.find(f".//h:{k}", NS).text, "%Y-%m-%dT%H:%M:%SZ")
                    for k in ("startTime", "endTime"))
            out.append((s, e, it.find(".//h:playbackURI", NS).text))
        pos += len(items)
        if x.find("h:responseStatusStrg", NS).text != "MORE" or not items:
            return sorted(set(out))


def runs(segs):
    """[[start, end, [uri, ...]]]: segments less than MERGE_S apart joined."""
    out = []
    for s, e, uri in segs:
        if out and (s - out[-1][1]).total_seconds() <= MERGE_S:
            out[-1][1] = max(out[-1][1], e)
            out[-1][2].append(uri)
        else:
            out.append([s, e, [uri]])
    return out


def fetch(cam, uris, path):
    """The run's segments, downloaded back to back (MPEG-PS concatenates byte-wise), remuxed to MP4 without the
    DVR's mu-law audio (MP4 can't hold it)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ps = path.with_suffix(".ps")
    with open(ps, "wb") as f:
        for uri in uris:
            body = f"<downloadRequest><playbackURI>{escape(uri)}</playbackURI></downloadRequest>"
            with requests.post(f"http://{cam['host']}/ISAPI/ContentMgmt/download", data=body, stream=True,
                               timeout=60, auth=HTTPDigestAuth(cam["user"], cam["pw"])) as r:
                r.raise_for_status()
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    tmp = path.with_suffix(".tmp.mp4")  # renamed when complete, so replay_day --follow never reads half a file
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-fflags", "+genpts", "-i", str(ps), "-an", "-c", "copy",
                    str(tmp)], check=True)
    tmp.replace(path)
    ps.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "fetch"])
    ap.add_argument("day")
    ap.add_argument("--from", dest="h0", type=int, default=0)
    ap.add_argument("--to", dest="h1", type=int, default=24)
    ap.add_argument("--cams", default="")
    a = ap.parse_args()
    cams = cameras(set(filter(None, a.cams.split(","))))
    if a.cmd == "list":
        for cam in cams:
            rs = runs(search(cam, a.day, a.h0, a.h1))
            per_h = {}
            for s, e, _ in rs:
                per_h[s.hour] = per_h.get(s.hour, 0) + (e - s).total_seconds() / 60
            tot = sum(per_h.values())
            print(f"{cam['name']:9} {len(rs):4} runs {tot:6.0f} min  " +
                  " ".join(f"{h:02d}:{m:.0f}" for h, m in sorted(per_h.items())))
        return
    from concurrent.futures import ThreadPoolExecutor

    def one(cam):
        for s, e, uris in runs(search(cam, a.day, a.h0, a.h1)):
            p = OUT / a.day / cam["name"] / f"{s:%H%M%S}.mp4"
            if not p.exists():
                try:
                    fetch(cam, uris, p)
                except Exception as ex:  # one bad segment shouldn't stop the day
                    print(f"{cam['name']} {s:%H:%M:%S}: {ex}", file=sys.stderr, flush=True)
                    continue
            print(f"{cam['name']} {s:%H:%M:%S} +{(e - s).total_seconds():.0f}s", flush=True)
    with ThreadPoolExecutor(len(cams)) as ex:
        list(ex.map(one, cams))
    (OUT / a.day / ".done").touch()  # tells replay_day --follow that no more files are coming


if __name__ == "__main__":
    main()

"""Capture dependency evidence and run a short real no-person pipeline probe."""
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / 'docs' / 'audit'
sys.path.insert(0, str(BASE))
packages = []
for dist in metadata.distributions():
    m = dist.metadata
    packages.append(dict(name=m['Name'], version=dist.version,
                         license=m.get('License-Expression') or (m.get('License') or '').split('\n')[0],
                         license_classifiers=[c for c in m.get_all('Classifier', []) if c.startswith('License')]))
(OUT / 'installed-dependencies.json').write_text(json.dumps(sorted(packages, key=lambda p: p['name'].lower()), indent=2), encoding='utf-8')
evidence = dict(python=platform.python_version(), platform=platform.platform())
evidence['model_sha256'] = hashlib.sha256((BASE / 'models' / 'yolo11n-pose.pt').read_bytes()).hexdigest()
evidence['pip_check'] = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True, timeout=60).stdout.strip()
evidence['examples'] = []
for video in sorted((BASE / 'examples').glob('*/annotated.mp4')):
    probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_name,pix_fmt,width,height,r_frame_rate',
                            '-show_entries', 'format=duration,size', '-of', 'json', str(video)], capture_output=True, text=True, check=True, timeout=30)
    evidence['examples'].append(dict(slug=video.parent.name, video=json.loads(probe.stdout)))
with tempfile.TemporaryDirectory(prefix='blank-probe-', dir=OUT) as tmp:
    with patch.dict(os.environ, {'DATA_DIR': tmp, 'OPENROUTER_API_KEY': ''}):
        import cv2
        import torch
        from app import pipeline
        evidence['torch_threads_at_import'] = torch.get_num_threads()
        src = Path(tmp) / 'blank.mp4'
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=gray:s=320x240:r=25:d=3',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(src)], check=True, timeout=30)
        start = time.monotonic()
        result = pipeline.run(src, Path(tmp) / 'result')
        evidence['real_blank_clip'] = dict(seconds=round(time.monotonic()-start, 2), people=result['people'], events=result['events'],
                                           report=result['report'], outcome='pipeline returned success')
        evidence['torch_threads_after_pipeline'] = torch.get_num_threads()
        strip = cv2.imread(str(Path(tmp) / 'result' / 'strip1.jpg'))
        evidence['caption_strip_size'] = dict(width=strip.shape[1], height=strip.shape[0])
(OUT / 'environment-results.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
print(json.dumps(evidence, indent=2))

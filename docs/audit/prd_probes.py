"""Read-only audit probes; disposable files are confined to a temporary directory.
Run from project root: .venv/Scripts/python docs/audit/prd_probes.py
No external API requests. Does not modify application code or existing data.
"""
import copy
import hashlib
import json
import os
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))


def kp(hip_y=500):
    points = [(500, 320)] * 5 + [(500, 380)] * 6 + [(500, hip_y)] * 2 + [(500, 600)] * 2 + [(500, 700)] * 2
    return [[x, y, .9] for x, y in points]


def samples(phases):
    out = []
    for a, b, posture in phases:
        for n in range(a * 6, b * 6):
            lying = posture == 'lying'
            out.append(dict(i=n, t=n / 6, id=1, box=[290, 650, 610, 710] if lying else [450, 300, 550, 700],
                            kp=kp(685 if lying else 500), conf=.9, posture=posture,
                            activity=posture, pt=(500, 680), speed=0))
    return out


def main():
    observations = {}
    with tempfile.TemporaryDirectory(prefix='eldercare-audit-', dir=BASE / 'docs' / 'audit') as tmp:
        with patch.dict(os.environ, {'DATA_DIR': tmp, 'OPENROUTER_API_KEY': ''}):
            import httpx
            from fastapi.testclient import TestClient
            from app import events, llm, main as api, pipeline, posture, tracks
            rules = pipeline.load_rules()
            meta = dict(width=1000, height=1000, fps=6, duration=28, clock_start=0)
            seq = samples([(0, 4, 'standing'), (4, 10, 'lying'), (10, 12, 'standing')])
            rest_rules = copy.deepcopy(rules)
            rest_rules['zones'] = [dict(name='Bed', kind='rest', polygon=[[0, 0], [1, 0], [1, 1], [0, 1]])]
            rest = events.analyze([dict(id=1, samples=copy.deepcopy(seq))], meta, rest_rules)
            observations['fall_inside_rest_zone'] = rest['events']
            observations['bending_to_lying_fall'] = events.detect_falls(samples([(0, 4, 'bending'), (4, 10, 'lying')]), rules['fall'])
            for name, phases in [('recovered', [(0, 4, 'standing'), (4, 10, 'lying'), (10, 12, 'standing')]),
                                 ('unrecovered', [(0, 4, 'standing'), (4, 28, 'lying')])]:
                result = events.analyze([dict(id=1, samples=samples(phases))], meta, rules)
                observations[name + '_fall'] = result['events']
            gap = samples([(0, 4, 'standing')]) + samples([(4, 5, 'lying')])[:1] + samples([(20, 21, 'lying')])[:1]
            observations['fall_confirmed_across_16s_detection_gap'] = events.detect_falls(gap, rules['fall'])
            observations['stillness_despite_posture_change'] = events.stillness(samples([(0, 3, 'standing'), (3, 7, 'sitting')]), 5, .15)
            observations['ambiguous_box_no_pose'] = posture.classify([0, 0, 100, 100], [[0, 0, 0]] * 17)
            observations['default_upload_zones'] = rules['zones']
            observations['empty_tracks_analysis'] = events.analyze([], meta, rules)
            large = dict(id=1, samples=[dict(t=0, box=[0, 0, 100, 400])])
            small = dict(id=2, samples=[dict(t=1, box=[0, 0, 50, 100])])
            observations['rejoin_different_sized_people_track_count'] = len(tracks.rejoin([large, small], 3, 1, 1.5))
            blank_analysis = events.analyze([], meta, rules)
            observations['empty_tracks_template'] = llm.template_report(blank_analysis)
            with TestClient(api.app) as client:
                observations['api_config'] = client.get('/api/config').json()
                observations['health'] = client.get('/healthz').json()
                observations['progress_route_status'] = client.get('/p/example').status_code
                observations['security_headers'] = dict(client.get('/').headers)
                observations['range_request_status'] = client.get('/media/examples/synthetic-fall/annotated.mp4', headers={'Range': 'bytes=0-99'}).status_code
                observations['traversal_status'] = client.get('/media/results/%2e%2e/%2e%2e/.env').status_code
                rid = 'audit12345678901234567890'
                debug = api.jobs.results / rid / 'debug'
                debug.mkdir(parents=True)
                (debug / 'observations.jsonl').write_text('{"audit_marker":true}')
                observations['debug_media_status'] = client.get(f'/media/results/{rid}/debug/observations.jsonl').status_code
                observations['bad_video_status'] = client.post('/api/jobs', files={'file': ('bad.mp4', b'not video')}, data={'consent': 'yes'}).status_code
                api._hits.clear()
                with patch.object(api, 'PASSCODE', 'audit-pass'):
                    observations['bad_passcode_status'] = client.post('/api/jobs', files={'file': ('bad.mp4', b'not video')}, data={'consent': 'yes', 'passcode': 'wrong'}).status_code
                with patch.object(api, 'MAX_BYTES', 2):
                    observations['oversize_status'] = client.post('/api/jobs', files={'file': ('bad.mp4', b'123')}, data={'consent': 'yes'}).status_code
            with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'fake-audit-key'}):
                def response(req):
                    return httpx.Response(200, json={'choices': [{'message': {'content': 'Everything is safe. No fall occurred.'}}], 'usage': {'cost': .01}})
                ai = llm.LLM(Path(tmp) / 'contradiction', httpx.Client(transport=httpx.MockTransport(response)))
                facts = dict(glance=[], events=rest['events'])
                observations['contradictory_llm_report_accepted'] = ai.report(facts)
                messages = [{'role': 'user', 'content': 'audit'}]
                cache_key = hashlib.sha256(json.dumps([ai.model, messages, 400]).encode()).hexdigest()
                (ai.dir / 'cache' / (cache_key + '.json')).write_text('invalid-json')
                try:
                    ai.chat(messages)
                    observations['corrupt_cache_behavior'] = 'no exception'
                except Exception as exc:
                    observations['corrupt_cache_behavior'] = type(exc).__name__
                barrier = threading.Barrier(4)
                def concurrent_response(req):
                    barrier.wait(timeout=10)
                    return response(req)
                with patch.dict(os.environ, {'MAX_LLM_USD_PER_DAY': '.01'}):
                    parallel_ai = llm.LLM(Path(tmp) / 'parallel', httpx.Client(transport=httpx.MockTransport(concurrent_response)))
                    with ThreadPoolExecutor(4) as pool:
                        list(pool.map(lambda n: parallel_ai.chat([{'role': 'user', 'content': str(n)}]), range(4)))
                    observations['parallel_budget'] = dict(cap=.01, spent=parallel_ai.spent_today())
            summaries = []
            for path in sorted((BASE / 'examples').glob('*/result.json')):
                result = json.loads(path.read_text(encoding='utf-8'))
                required = ['id', 'kind', 'version', 'created_at', 'clips', 'segments', 'zone_segments', 'metrics', 'ai', 'processing', 'disclaimer']
                summaries.append(dict(example=path.parent.name, missing_required_keys=[k for k in required if k not in result],
                                      events=[e['type'] for e in result['events']], ai=result['report'].get('ai')))
            observations['example_schema_checks'] = summaries
    destination = BASE / 'docs' / 'audit' / 'probe-results.json'
    destination.write_text(json.dumps(observations, indent=2), encoding='utf-8')
    print(json.dumps(observations, indent=2))


if __name__ == '__main__':
    main()

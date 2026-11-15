"""R3.4 follow-up: do tasks whose skill was lost to a false merge fare worse when the curriculum proposes them again?"""
import os as _os
LOGS = _os.environ.get('AUTOSKILL_LOGS', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'logs'))
import json, re, collections, numpy as np
from scipy.stats import fisher_exact
src = open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'r34_dedup_errors.py')).read()
exec(re.sub(r"^rows = .*$", '', src[:src.index('events = [')], flags=re.M))
rows = json.load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'dedup_events.json')))
by_run = collections.defaultdict(list)
for r in rows:
    if r['kind'] in ('added', 'dedup'): by_run[r['run']].append(r)
later = {'lost (false merge)': [0, 0], 'stored': [0, 0]}  # [successes, attempts] on later proposals of the same reference task
for run, evs in by_run.items():
    status = {}  # ref key -> 'stored' | 'lost'
    for r in evs:
        k = ref_key(r['task'])
        if k in status: continue
        if r['kind'] == 'added': status[k] = 'stored'
        elif ref_key(r['rep_task'] or '') != k: status[k] = 'lost'
    steps = [json.loads(l) for l in open(f'{LOGS}/{run}/step_log.jsonl')]
    first_ok = {}
    for i, s in enumerate(steps):
        k = ref_key(s['task'])
        if s['success'] and k not in first_ok: first_ok[k] = i
    for i, s in enumerate(steps):
        k = ref_key(s['task'])
        if k in first_ok and i > first_ok[k] and k in status:
            c = later['lost (false merge)' if status[k] == 'lost' else 'stored']
            c[0] += s['success']; c[1] += 1
for k, (ok, n) in later.items(): print(f'{k:20s} later attempts {n:4d}  success {ok}/{n} = {100*ok/max(n,1):.1f}%')
(a, n1), (b, n2) = later['lost (false merge)'], later['stored']
print('Fisher exact (two-sided) p =', round(fisher_exact([[a, n1 - a], [b, n2 - b]])[1], 3))

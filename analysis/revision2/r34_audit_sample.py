"""R3.4 follow-up: random sample of gate decisions NOT flagged by the reference key, for manual audit."""
import os as _os
LOGS = _os.environ.get('AUTOSKILL_LOGS', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'logs'))
import json, random, re, collections
src = open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'r34_dedup_errors.py')).read()
exec(re.sub(r"^rows = .*$", '', src[:src.index('events = [')], flags=re.M))
rows = json.load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'dedup_events.json')))
ev = [r for r in rows if r['kind'] in ('added', 'dedup')]
seen = collections.defaultdict(list); ok_d, ok_a = [], []
for r in ev:
    k = ref_key(r['task'])
    if r['kind'] == 'dedup' and ref_key(r['rep_task'] or '') == k: ok_d.append((r['task'], r['rep_task']))
    if r['kind'] == 'added':
        if k not in [ref_key(t) for t in seen[r['run']]]: ok_a.append((r['task'], list(seen[r['run']])))
        seen[r['run']].append(r['task'])
rng = random.Random(0)
print(f'unflagged discards {len(ok_d)}, unflagged stored {len(ok_a)}')
print('== discards: task  <=  representative task')
for t, rt in rng.sample(ok_d, 40): print(f'  {t!r:45} <= {rt!r}')
print('== stored: task  |  earlier stored tasks in run with same verb')
for t, prev in rng.sample(ok_a, 40):
    v = t.split()[0].lower(); print(f'  {t!r:40} | {[p for p in prev if p.split()[0].lower()==v][-8:]}')

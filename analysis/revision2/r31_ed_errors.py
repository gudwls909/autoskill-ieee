"""R3.1: grouping errors of the embedding-dedup baseline (same reference key as r34_dedup_errors.py)."""
import os as _os
LOGS = _os.environ.get('AUTOSKILL_LOGS', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'logs'))
import json, glob, re, collections
src = open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'r34_dedup_errors.py')).read()
exec(re.sub(r"^rows = .*$", '', src[:src.index('events = [')], flags=re.M))  # NUM/ALIAS/ref_key only, same reference as R3.4
fm = dd = mm = ad = 0
for f in sorted(glob.glob(f'{LOGS}/run-r2-mini-ed-seed*/skill_log.jsonl')):
    if 'hung' in f: continue
    rows = [json.loads(l) for l in open(f)]
    task_of = {}; seen = set()
    for r in rows:
        k = ref_key(r['task'])
        if r['kind'] == 'skill_add':
            ad += 1; mm += k in seen; seen.add(k); task_of[r['program_name']] = r['task']
        elif r['kind'] == 'dedup':
            dd += 1; fm += ref_key(task_of.get(r['matched_neighbor'], '')) != k
print(f'embedding dedup: discards {dd}, false merges {fm} ({100*fm/dd:.1f}%); stored {ad}, missed merges {mm} ({100*mm/ad:.1f}%)')

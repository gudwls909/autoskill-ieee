"""R3.1: calibrate the cosine threshold of the embedding-dedup baseline on the five original Voyager libraries.
Label: a stored skill is a duplicate if its task's reference key (r34) matches an earlier stored skill's."""
import os as _os
LOGS = _os.environ.get('AUTOSKILL_LOGS', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'logs'))
import json, sys, numpy as np
_Z = np.load(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'voyager_lib_embeddings.npz'))
sys.path.insert(0, '.')
from r34_dedup_errors import ref_key
RUNS = ['run-2026-04-03_14-13-42', 'run-2026-04-03_19-03-57', 'run-2026-04-04_21-18-15', 'run-2026-04-05_15-29-49', 'run-2026-04-06_02-00-16']
sims, labels = [], []
for r in RUNS:
    got = {'ids': list(_Z[r + '__ids']), 'embeddings': list(_Z[r + '__emb'])}  # exported from each run's Chroma store
    emb = {i: np.asarray(e) / np.linalg.norm(e) for i, e in zip(got['ids'], got['embeddings'])}
    order = []
    for l in open(f'{LOGS}/{r}/skill_log.jsonl'):
        e = json.loads(l)
        if e['kind'] in ('skill_add', 'added') and e['program_name'] in emb and e['program_name'] not in [o[0] for o in order]:
            order.append((e['program_name'], ref_key(e['task'])))
    for k in range(1, len(order)):
        name, key = order[k]
        prev = order[:k]
        sims.append(max(float(emb[name] @ emb[p]) for p, _ in prev))
        labels.append(any(key == pk for _, pk in prev))
sims, labels = np.array(sims), np.array(labels)
print('pairs', len(sims), 'duplicates', labels.sum())
best = None
for t in np.arange(0.80, 0.995, 0.005):
    pred = sims >= t; tp = (pred & labels).sum(); fp = (pred & ~labels).sum(); fn = (~pred & labels).sum()
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    if best is None or f1 > best[1]: best = (t, f1, tp, fp, fn)
    if abs(t * 1000 % 20) < 1e-6: print(f'tau={t:.3f} F1={f1:.3f} tp={tp} fp={fp} fn={fn}')
print('best tau=%.3f F1=%.3f tp=%d fp=%d fn=%d' % best)

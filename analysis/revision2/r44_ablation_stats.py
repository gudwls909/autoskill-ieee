"""R4.4 follow-up: ablation tests with Holm correction, bootstrap CI of the mean TTS difference, and a same-revision (B-only) comparison."""
import os as _os, json, sys, glob, numpy as np
from scipy.stats import mannwhitneyu
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from plot_dedup_efficiency import RUNS as MAIN
from plot_ablation_ci import RUNS as ABL
LOGS = _os.environ.get('AUTOSKILL_LOGS', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'logs'))
tts = lambda r: [json.loads(l) for l in open(f'{LOGS}/{r}/step_log.jsonl') if l.strip()][-1]['tech_tree_score']
full = MAIN['AutoSkill (ours)']; fullB = [r for r in full if 'resub-fix2' in r]  # revision-B AutoSkill runs
cond = {'w/o Dedup': ABL['w/o Dedup'], 'w/o Critique': ABL['w/o Critique'], 'w/o Curriculum': ABL['w/o Curriculum']}
isB = lambda r: r.startswith('run-r2-') or 'resub-fix2' in r
rng = np.random.default_rng(0)
def ci(a, b, B=20000):
    a, b = np.array(a), np.array(b)
    d = rng.choice(a, (B, len(a))).mean(1) - rng.choice(b, (B, len(b))).mean(1)
    return np.percentile(d, [2.5, 97.5])
rows = []
for k, rs in cond.items():
    A = [tts(r) for r in full]; X = [tts(r) for r in rs]
    p = mannwhitneyu(A, X, alternative='greater').pvalue
    AB = [tts(r) for r in fullB]; XB = [tts(r) for r in rs if isB(r)]
    pB = mannwhitneyu(AB, XB, alternative='greater').pvalue
    rows.append([k, np.mean(A) - np.mean(X), ci(A, X), p, len(XB), np.mean(AB) - np.mean(XB), ci(AB, XB), pB])
# Holm over the three ablation tests (all runs)
order = np.argsort([r[3] for r in rows]); m = len(rows); adj = [0] * m; run_max = 0
for i, j in enumerate(order):
    run_max = max(run_max, min(1, (m - i) * rows[j][3])); adj[j] = run_max
for r, a in zip(rows, adj):
    print(f"{r[0]:15s} all: diff {r[1]:+.1f} CI[{r[2][0]:+.1f},{r[2][1]:+.1f}] p={r[3]:.3f} Holm={a:.3f} | B-only (AutoSkill n=5 vs n={r[4]}): diff {r[5]:+.1f} CI[{r[6][0]:+.1f},{r[6][1]:+.1f}] p={r[7]:.3f}")
print('AutoSkill B runs TTS', [tts(r) for r in fullB], 'A runs', [tts(r) for r in full if r not in fullB])

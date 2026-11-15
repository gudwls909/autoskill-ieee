"""R3.2: TTS vs an independent progress measure (Voyager tool milestones) + tier-weight sensitivity."""
import os as _os
LOGS = _os.environ.get('AUTOSKILL_LOGS', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'logs'))
import json, sys, itertools, numpy as np
from scipy.stats import mannwhitneyu, spearmanr
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import tech_tree_score as T
from plot_dedup_efficiency import RUNS
LOG = LOGS
MAT = ['wooden', 'stone', 'iron', 'diamond']
TOOLS = ['pickaxe', 'axe', 'shovel', 'sword', 'hoe']

def load(r): return [json.loads(l) for l in open(f'{LOG}/{r}/step_log.jsonl') if l.strip()]

def milestones(rows, horizon=160):
    # first step at which any tool of each material is held; unreached -> horizon (censored)
    out = []
    for m in MAT:
        names = {f'{m}_{t}' for t in TOOLS}
        out.append(next((i for i, s in enumerate(rows, 1) if names & set(s.get('seen_items') or [])), horizon))
    return out

data = {}
for lab, rs in RUNS.items():
    for r in rs:
        rows = load(r); items = rows[-1]['seen_items']
        tiers = np.array([T.get_tier(x) for x in items])
        data[r] = dict(lab=lab, tiers=tiers, ms=milestones(rows), tts=rows[-1]['tech_tree_score'])
        assert data[r]['tts'] == tiers.sum()

V = [r for r in data if data[r]['lab'].startswith('Voy')]; A = [r for r in data if data[r]['lab'].startswith('Auto')]

print('== Milestones (first step; 160 = not reached) ==')
for g, rs in [('Voyager', V), ('AutoSkill', A)]:
    M = np.array([data[r]['ms'] for r in rs])
    print(g, 'median', np.median(M, 0), 'reached', (M < 160).sum(0))
for k, m in enumerate(MAT):
    a = [data[r]['ms'][k] for r in A]; v = [data[r]['ms'][k] for r in V]
    print(f'  {m:8s} MWU (AutoSkill earlier) p=%.4f' % mannwhitneyu(a, v, alternative='less').pvalue)

# independent progress score: mean milestone step (lower = faster), compare to TTS across 20 runs
prog = {r: np.mean(data[r]['ms']) for r in data}
rs = list(data)
print('Spearman(TTS, mean milestone step) over 20 runs: rho=%.2f p=%.4f' % spearmanr([data[r]['tts'] for r in rs], [prog[r] for r in rs]))
print('Spearman(items, mean milestone step):           rho=%.2f p=%.4f' % spearmanr([len(data[r]['tiers']) for r in rs], [prog[r] for r in rs]))

print('\n== Tier-weight sensitivity ==')
W = {'constant (=items)': lambda k: 1.0, 'linear (paper)': lambda k: k, 'sqrt': np.sqrt,
     'log2(1+k)': lambda k: np.log2(1 + k), 'quadratic': lambda k: k ** 2, 'exponential 2^(k-1)': lambda k: 2.0 ** (k - 1)}
def score(r, w): return sum(w(k) for k in data[r]['tiers'])
for name, w in W.items():
    a = np.array([score(r, w) for r in A]); v = np.array([score(r, w) for r in V])
    U, p = mannwhitneyu(a, v, alternative='greater')
    print(f'  {name:20s} A={a.mean():7.1f} V={v.mean():7.1f} gain={100*(a.mean()/v.mean()-1):+6.1f}%  U={U:5.1f} p={p:.2e}  separation={a.min()>v.max()}')

# robustness to the tier table itself: shift every item's tier by -1/0/+1 at random (clipped 1..6)
rng = np.random.default_rng(0); gains = []; sep = 0; sig = 0
for _ in range(1000):
    shift = {}
    def t2(r):
        out = []
        for it, k in zip(json.loads(json.dumps(load_items[r])), data[r]['tiers']):
            if it not in shift: shift[it] = rng.integers(-1, 2)
            out.append(min(6, max(1, k + shift[it])))
        return np.array(out)
    load_items = {r: load(r)[-1]['seen_items'] for r in data} if not gains else load_items
    a = np.array([t2(r).sum() for r in A]); v = np.array([t2(r).sum() for r in V])
    gains.append(100 * (a.mean() / v.mean() - 1)); sep += a.min() > v.max()
    sig += mannwhitneyu(a, v, alternative='greater').pvalue < 0.01
g = np.array(gains)
print(f'  random ±1 tier perturbation (1000 draws): gain median {np.median(g):+.1f}% [5–95%: {np.percentile(g,5):+.1f}, {np.percentile(g,95):+.1f}], p<0.01 in {sig/10:.1f}%, full separation in {sep/10:.1f}%')

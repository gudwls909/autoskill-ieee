"""Round-2 results: R3.1 (embedding dedup), R3.3/R5.4 (model swap), R4.4 (ablations n=10, w/o Curriculum)."""
import os as _os
LOGS = _os.environ.get('AUTOSKILL_LOGS', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'logs'))
import json, glob, re, sys, numpy as np, contextlib, io
from scipy.stats import mannwhitneyu
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from plot_dedup_efficiency import RUNS as MAIN
from plot_ablation_ci import RUNS as ABL
import analyze_run
V = LOGS + '/'

def r2(tag, m, repo='voyager'):
    out = []
    for d in sorted(glob.glob(f'{LOGS}/run-r2-{tag}-{m}-seed*')):
        if re.search(r'hung|crash', d): continue
        out.append(d)
    return out

def final(d):
    L = [json.loads(l) for l in open(d + '/step_log.jsonl') if l.strip()]
    assert len(L) >= 140, d
    with contextlib.redirect_stdout(io.StringIO()):
        try: m = analyze_run.extract_metrics(d)
        except Exception: m = {}
    # true reuse = share of generations whose program actually calls a known library skill (paper Table "True Reuse")
    rl = [json.loads(l) for l in open(d + '/reuse_log.jsonl')] if glob.glob(d + '/reuse_log.jsonl') else []
    reuse = np.mean([bool(x.get('true_called_known_skills')) for x in rl]) * 100 if rl else np.nan
    return dict(tts=L[-1]['tech_tree_score'], items=L[-1]['distinct_items'], lib=L[-1].get('skill_library_size'),
                reuse=reuse, comp=m.get('composite_pct', np.nan))

C = {
 'Voyager':            [V + r for r in MAIN['Voyager (main)']],
 'AutoSkill':          [V + r for r in MAIN['AutoSkill (ours)']],
 # plot_ablation_ci.RUNS already lists all ten runs per ablation (original five + round-2 seeds 6-10)
 'w/o Dedup':          [V + r for r in ABL['w/o Dedup']],
 'w/o Critique':       [V + r for r in ABL['w/o Critique']],
 'w/o Curriculum':     r2('mini', 'ncur'),
 'Voyager+EmbDedup':   r2('mini', 'ed', 'voyager_ed'),
 'Voyager g5.4':       r2('g54', 'vb', 'voyager_vb'),
 'AutoSkill g5.4':     r2('g54', 'as'),
 'Voyager qwen14b':    r2('qwen14b', 'vb', 'voyager_vb'),
 'AutoSkill qwen14b':  r2('qwen14b', 'as'),
}
R = {k: [final(d) for d in v] for k, v in C.items()}

def ci(x, B=10000, rng=np.random.default_rng(0)):
    b = rng.choice(x, (B, len(x))).mean(1); return np.percentile(b, [2.5, 97.5])

print(f"{'condition':20s} n   TTS mean±sd [95%CI]        items  lib   reuse%  comp%")
for k, rs in R.items():
    t = np.array([r['tts'] for r in rs]); lo, hi = ci(t)
    f = lambda key: np.nanmean([r[key] for r in rs])
    print(f"{k:20s} {len(t):2d} {t.mean():6.1f}±{t.std():5.1f} [{lo:5.1f},{hi:5.1f}]  {f('items'):5.1f} {f('lib'):5.1f} {f('reuse'):6.1f} {f('comp'):6.1f}  {sorted(t.tolist())}")

def test(a, b, key='tts', alt='greater'):
    x = [r[key] for r in R[a]]; y = [r[key] for r in R[b]]
    p = mannwhitneyu(x, y, alternative=alt).pvalue
    d = (np.mean(x) - np.mean(y)) / np.mean(y) * 100
    print(f"  {a} vs {b} [{key}]: {d:+.1f}%  one-sided MWU p={p:.4f}")
print('\n== tests ==')
for a, b in [('AutoSkill','w/o Dedup'),('AutoSkill','w/o Critique'),('AutoSkill','w/o Curriculum'),
             ('w/o Curriculum','Voyager'),('w/o Dedup','Voyager'),('w/o Critique','Voyager'),
             ('AutoSkill','Voyager+EmbDedup'),('Voyager','Voyager+EmbDedup'),
             ('AutoSkill g5.4','Voyager g5.4'),('AutoSkill qwen14b','Voyager qwen14b')]:
    test(a, b)
for key in ['reuse', 'comp']:
    test('AutoSkill', 'w/o Critique', key); test('AutoSkill', 'w/o Dedup', key); test('AutoSkill', 'w/o Curriculum', key)
test('w/o Dedup', 'AutoSkill', 'lib')

# ---- extra columns for the paper tables (sd, tokens, redefinition) ----
def extra(d):
    tok = sum(json.loads(l).get('total', 0) for l in open(d + '/token_log.jsonl')) / 1e3 if glob.glob(d + '/token_log.jsonl') else np.nan
    rl = [json.loads(l) for l in open(d + '/reuse_log.jsonl')] if glob.glob(d + '/reuse_log.jsonl') else []
    redef = np.mean([bool(x.get('redefined_known_skills')) for x in rl]) * 100 if rl else np.nan
    return tok, redef
print('\n== table columns (mean±sd) ==')
for k, ds in C.items():
    rs = R[k]; ex = [extra(d) for d in ds]
    s = lambda a: f"{np.nanmean(a):.1f}±{np.nanstd(a):.1f}"
    print(f"{k:20s} TTS {s([r['tts'] for r in rs])} lib {s([r['lib'] for r in rs])} comp {s([r['comp'] for r in rs])} "
          f"reuse {s([r['reuse'] for r in rs])} redef {s([e[1] for e in ex])} tok {np.nanmean([e[0] for e in ex]):.0f}K")
# sanity: first-5 subsets should reproduce the paper's n=5 values (AutoSkill reuse 36.9 / comp 16.4; w/o Critique reuse 3.8)
for k in ['AutoSkill', 'w/o Critique', 'w/o Dedup']:
    print('n=5 check', k, np.mean([r['reuse'] for r in R[k][:5]]).round(1), np.mean([r['comp'] for r in R[k][:5]]).round(1), np.mean([r['tts'] for r in R[k][:5]]).round(1))

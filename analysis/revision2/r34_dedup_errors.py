"""R3.4: false-merge and missed-merge rates of the dedup gate, against a reference key parsed from the curriculum task text."""
import os as _os
LOGS = _os.environ.get('AUTOSKILL_LOGS', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'logs'))
import json, re, collections, sys
rows = json.load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'dedup_events.json')))
NUM = r'(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|some|more|the)'
ALIAS = {'wood log': 'log', 'oak log': 'log', 'logs': 'log', 'wood': 'log', 'raw iron': 'iron ore', 'raw copper': 'copper ore',
         'raw gold': 'gold ore', 'diamonds': 'diamond',
         'diamond ore': 'diamond', 'cobblestones': 'cobblestone', 'stone': 'cobblestone'}

def ref_key(task):
    t = task.lower().replace('_', ' ').strip().rstrip('.')
    t = re.sub(r'\b(at|in|into|from|near|using|with|to)\b.*$', '', t).strip()  # drop locations/tool clauses
    words = t.split()
    if not words: return None
    verb = words[0]
    rest = ' '.join(w for w in words[1:] if not re.fullmatch(NUM, w) and w != 'nearby')
    rest = re.sub(r'(?<=[a-z])s$', '', rest) if not rest.endswith(('ss', 'glass')) else rest
    rest = ALIAS.get(rest, rest)
    if verb == 'obtain': verb = 'mine'
    return f'{verb}:{rest}'

events = [r for r in rows if r['kind'] in ('added', 'dedup')]
for r in events: r['ref'] = ref_key(r['task'])

# false merges: a discarded skill whose reference key differs from its representative's
dedup = [r for r in events if r['kind'] == 'dedup']
fm = [r for r in dedup if ref_key(r['rep_task'] or '') != r['ref']]
# missed merges: a stored skill whose reference key matches an earlier stored skill in the same run
mm = []; seen = collections.defaultdict(dict)
for r in events:
    if r['kind'] != 'added': continue
    if r['ref'] in seen[r['run']]: mm.append((r, seen[r['run']][r['ref']]))
    else: seen[r['run']][r['ref']] = r
added = [r for r in events if r['kind'] == 'added']
print(f'dedup decisions {len(dedup)}  false merges {len(fm)} ({100*len(fm)/len(dedup):.1f}%)')
print(f'stored skills   {len(added)}  missed merges {len(mm)} ({100*len(mm)/len(added):.1f}%)')
print('false merges by parse mode', collections.Counter(r['pm'] for r in fm), 'of', collections.Counter(r['pm'] for r in dedup))
print('missed merges by parse mode', collections.Counter(r['pm'] for r, _ in mm), 'of', collections.Counter(r['pm'] for r in added))
print('false-merge target keys', collections.Counter(r['fk'] for r in fm).most_common(4))
print('missed-merge verbs', collections.Counter(r['ref'].split(':')[0] for r, _ in mm))
print('deepslate variants stored separately (not counted):', sum('deepslate' in (r['task'] or '') for r in added))
print('per run (fm, mm):', {r[-8:]: (sum(x['run']==r for x in fm), sum(x[0]['run']==r for x in mm)) for r in dict.fromkeys(x['run'] for x in events)})
if '-v' in sys.argv:
    print('\n-- false merges --')
    for r in fm: print(f"  {r['task'][:40]:40s} -> {str(r['rep_task'])[:34]:34s} key={r['fk']} pm={r['pm']}")
    print('\n-- missed merges --')
    for r, e in mm: print(f"  {r['task'][:34]:34s} [{r['fk']}] vs earlier {e['task'][:30]:30s} [{e['fk']}] pm={r['pm']}")

# counterfactual: quantity made optional in the task-text regex -> key comes from task text for these verbs
OPT = re.compile(r'^(Mine|Craft|Smelt|Place|Kill)\s+(?:(\d+|a|an)\s+)?(.+)$', re.I)
def fixed(task): return bool(OPT.match(task.strip()))
fm_fix = [r for r in fm if fixed(r['task']) and fixed(r['rep_task'] or '')]
mm_fix = [(r, e) for r, e in mm if fixed(r['task']) and fixed(e['task'])]
print(f'optional-quantity regex would resolve {len(fm_fix)}/{len(fm)} false merges and {len(mm_fix)}/{len(mm)} missed merges')
print('tasks without a quantity among stored+discarded:', sum(1 for r in events if fixed(r['task']) and not re.match(r'^\w+\s+(\d+|a|an)\s', r['task'])), '/', len(events))

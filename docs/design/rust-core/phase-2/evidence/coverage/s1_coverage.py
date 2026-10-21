# S1 coverage from coverage.py's own generator (rule 7, --gate S1, is not built).
import sys
from pathlib import Path
sys.path.insert(0, 'scripts/rust_core')
import coverage as c
rev = sys.argv[1]
ratchet = c._module('ratchet')
tree = ratchet.Tree(Path('.'), ratchet.git(Path('.'), 'rev-parse', f'{rev}^{{commit}}').strip())
report = c.Report()
validator = c.read_schema(tree, report)
manifest = c.read_manifest(tree, 'python', validator, report)
corpus = c.read_cases(tree, report)
assert not report.problems, report.problems
made = c.generated('python', manifest, corpus)
print(f'# S1 coverage at {tree.sha}: python manifest, acceptance cases only ({len(corpus)} cases read)')
paths = [p for p in manifest['paths'] if p['slice'] in ('S0', 'S1')]
empty_p = [p['id'] for p in paths if not made['paths'][str(p['id'])]]
print(f'paths in S0 and S1: {len(paths)}; with an acceptance case: {len(paths) - len(empty_p)}; empty: {len(empty_p)}')
for p in empty_p: print(f'  empty path {p}')
cells = manifest['cells']
inscope = [x for x in cells if x.get('expect') != 'unknown']
empty_c = [x for x in inscope if not made['cells'][str(x['id'])]]
print(f"cells: {len(cells)}; expect not 'unknown': {len(inscope)}; with an acceptance case: {len(inscope) - len(empty_c)}; empty: {len(empty_c)}")
for x in empty_c: print(f"  empty cell {x['id']} (expect {x.get('expect')})")

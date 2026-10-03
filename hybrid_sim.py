"""
Second Look — offline simulation of the hybrid ranking (rules + vision on the residual).

    .venv/bin/python hybrid_sim.py        writes results/hybrid_metrics.md

Needs data/items.json, data/labels.csv (blind labels) and data/vision_results.json (from vision_check.py).
No API calls. For every gold item the rules called "keep" ("no problem found") and the vision model
judged, the rule score is replaced by a fixed score on the same scale as the rules:
delete 0.75, unsure 0.45, keep 0.15 (set before running). Everything else keeps its rule score.
Orderings are then replayed over the blind gold set exactly as in evaluate.py.

Caveat: simulated on the same labels the vision model was scored on; not a product run.
"""
import csv, json
from pathlib import Path

K = 50
MAP = {'delete': 0.75, 'unsure': 0.45, 'keep': 0.15}
DATA, OUT = Path('data'), Path('results')


def main():
    items = {it['id']: it for it in json.loads((DATA / 'items.json').read_text())}
    vision = {v['id']: v['decision'] for v in json.loads((DATA / 'vision_results.json').read_text())
              if v['decision'] in MAP}
    last = {}
    for r in csv.DictReader((DATA / 'labels.csv').open()):
        if r['mode'] == 'blind':
            last[r['id']] = r['action']
    blind = set(json.loads((DATA / 'blind_sample.json').read_text()))
    lab = [items[i] | {'gold': a} for i, a in last.items() if a != 'undone' and i in blind and i in items]
    n, k = len(lab), min(K, len(lab))
    is_del = lambda it: it['gold'] == 'delete'
    total = sum(it['size'] for it in lab if is_del(it)) or 1

    def p_h(it):
        return MAP[vision[it['id']]] if it['status'] == 'keep' and it['id'] in vision else it['p_delete']

    orders = {
        'Largest files first (no judgement)': lambda it: -it['size'],
        'Second Look: rules p(delete) x size': lambda it: -(it['p_delete'] * it['size']),
        'Hybrid (simulated): rules + vision on residual, x size': lambda it: -(p_h(it) * it['size']),
        'Hybrid (simulated), p(delete) only': lambda it: (-p_h(it), -it['size']),
    }
    L = ['# Hybrid ranking — offline simulation\n',
         f'Gold set: {n} blind labels; first {k} scored. Rule scores replaced on '
         f'{sum(1 for it in lab if it["status"] == "keep" and it["id"] in vision)} residual items by the vision decision '
         f'(delete 0.75 / unsure 0.45 / keep 0.15). Simulated on the same labels; not a product run.\n',
         f'| Ordering | GB freed @{k} | Precision @{k} | Byte recall @{k} |', '|---|---:|---:|---:|']
    for name, key in orders.items():
        top = sorted(lab, key=key)[:k]
        freed = sum(it['size'] for it in top if is_del(it))
        L.append(f'| {name} | {freed / 1e9:.3f} | {sum(map(is_del, top)) / k:.0%} | {freed / total:.0%} |')
    OUT.mkdir(exist_ok=True)
    (OUT / 'hybrid_metrics.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()

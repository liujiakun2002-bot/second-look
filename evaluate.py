"""
Second Look — evaluation.

    python3 evaluate.py            writes results/metrics.md and results/metrics.json

Gold set = my labels from BLIND mode on the fixed random sample (no recommendation shown),
so the ranker cannot hide its own misses. Each ordering is replayed over that sample and
scored on its first K cards:

  GB freed @K        bytes I labelled "delete" among the first K        <- headline (value)
  precision @K       share of the first K I labelled "delete"           <- safety of what is shown
  byte recall @K     share of all delete-bytes in the sample reached in K
Abstention:          how often the system says "unsure", and what I said on those.
Silent failures:     system said "delete", I said "keep" (the costly error).

Only aggregate numbers are written to results/; no file names or paths.
"""
import csv, json
from collections import Counter, defaultdict
from pathlib import Path

K = 50
DATA, OUT = Path('data'), Path('results')
STRATEGIES = {
    'chrono':      'Reverse-chronological (what shipped swipe apps do)',
    'size_only':   'Largest files first (no judgement)',
    'rules_only':  'Rules only: hash duplicates + screenshots',
    'p_only':      'Full pipeline, ranked by p(delete) only',
    'second_look': 'Full pipeline, ranked by p(delete) x size  [Second Look]',
}


def last_labels(mode):
    last = {}
    if (DATA / 'labels.csv').exists():
        for r in csv.DictReader((DATA / 'labels.csv').open()):
            if r['mode'] == mode:
                last[r['id']] = r['action']
    return {k: v for k, v in last.items() if v != 'undone'}


def main():
    items = {it['id']: it for it in json.loads((DATA / 'items.json').read_text())}
    blind = [i for i in json.loads((DATA / 'blind_sample.json').read_text()) if i in items]
    gold = {i: a for i, a in last_labels('blind').items() if i in set(blind)}
    n = len(gold)
    if n == 0:
        raise SystemExit('no blind labels yet — run server.py and use 盲标 mode first')
    k = min(K, n)
    lab = [items[i] | {'gold': gold[i]} for i in gold]
    is_del = lambda it: it['gold'] == 'delete'
    total_del_bytes = sum(it['size'] for it in lab if is_del(it)) or 1
    base_rate = sum(map(is_del, lab)) / n

    rows = {}
    for s in STRATEGIES:
        top = sorted(lab, key=lambda it: it['rank'][s])[:k]
        freed = sum(it['size'] for it in top if is_del(it))
        rows[s] = dict(gb_at_k=freed / 1e9, precision_at_k=sum(map(is_del, top)) / k,
                       byte_recall_at_k=freed / total_del_bytes,
                       wrongly_first=sum(it['gold'] == 'keep' for it in top))
    rows['random_expected'] = dict(gb_at_k=total_del_bytes / 1e9 * k / n, precision_at_k=base_rate,
                                   byte_recall_at_k=k / n, wrongly_first=None)

    # abstention and silent failure (system status vs my blind label)
    cm = defaultdict(Counter)
    for it in lab:
        cm[it['status']][it['gold']] += 1
    sys_del = sum(cm['delete'].values()) or 1
    silent = cm['delete']['keep']
    abst = sum(cm['unsure'].values())

    by_type = {}
    for t in sorted({it['type'] for it in lab}):
        sub = [it for it in lab if it['type'] == t]
        rec = [it for it in sub if it['status'] == 'delete']
        by_type[t] = dict(n=len(sub), i_deleted=sum(map(is_del, sub)),
                          system_said_delete=len(rec),
                          precision_of_delete=(sum(map(is_del, rec)) / len(rec)) if rec else None)

    # real use: GB per 50 photos seen in ranked mode (biased — only what the ranker surfaced)
    ranked = last_labels('ranked')
    ranked_gb = sum(items[i]['size'] for i, a in ranked.items() if a == 'delete' and i in items) / 1e9
    per50 = ranked_gb / len(ranked) * 50 if ranked else None

    res = dict(k=k, n_gold=n, sample_size=len(blind), gold_mix=dict(Counter(gold.values())),
               strategies=rows, confusion={s: dict(c) for s, c in cm.items()},
               precision_of_system_delete=cm['delete']['delete'] / sys_del,
               silent_failures=silent, abstention_rate=abst / n,
               unsure_outcomes=dict(cm['unsure']), by_type=by_type,
               ranked_session=dict(seen=len(ranked), gb_freed=ranked_gb, gb_per_50=per50))
    OUT.mkdir(exist_ok=True)
    (OUT / 'metrics.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))

    L = [f'# Second Look — evaluation\n',
         f'Gold set: **{n}** blind labels from a fixed random sample of {len(blind)} '
         f'(mix: {dict(Counter(gold.values()))}). Each ordering replayed over the gold set; first **{k}** scored.\n',
         f'| Ordering | GB freed @{k} | Precision @{k} | Byte recall @{k} | Keeps shown as top-{k} |',
         '|---|---:|---:|---:|---:|']
    for s, r in rows.items():
        name = STRATEGIES.get(s, 'Random order (expected value)')
        L.append(f'| {name} | {r["gb_at_k"]:.3f} | {r["precision_at_k"]:.0%} | {r["byte_recall_at_k"]:.0%} | '
                 f'{"—" if r["wrongly_first"] is None else r["wrongly_first"]} |')
    L += ['', '## System recommendation vs my label', '',
          '| System said \\ I said | delete | keep | unsure |', '|---|---:|---:|---:|']
    for s in ('delete', 'unsure', 'keep'):
        L.append(f'| {s} | {cm[s]["delete"]} | {cm[s]["keep"]} | {cm[s]["unsure"]} |')
    L += ['', f'- Precision of the system\'s "delete" recommendations: **{res["precision_of_system_delete"]:.0%}**',
          f'- Silent failures (system "delete", I "keep"): **{silent}**',
          f'- Abstention rate (system "unsure"): **{res["abstention_rate"]:.0%}**; my labels on those: {dict(cm["unsure"])}',
          '', '## By content type', '', '| Type | n | I deleted | System said delete | Precision of its delete |', '|---|---:|---:|---:|---:|']
    for t, r in by_type.items():
        p = '—' if r['precision_of_delete'] is None else f'{r["precision_of_delete"]:.0%}'
        L.append(f'| {t} | {r["n"]} | {r["i_deleted"]} | {r["system_said_delete"]} | {p} |')
    if per50 is not None:
        L += ['', f'Ranked (product) mode: {len(ranked)} photos seen, {ranked_gb:.3f} GB marked → '
                  f'**{per50:.3f} GB per 50 swipes** (only what the ranker surfaced; not comparable across orderings).']
    (OUT / 'metrics.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()

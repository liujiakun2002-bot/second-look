"""
Second Look — vision-model escalation experiment (Problem Statement §4c, §5).

Rules and pixel statistics cannot tell an expired receipt from a precious message, or a
throwaway snapshot from a memory. This script tests whether a RENTED vision model can, on
the photos the rules could not judge, using my blind labels as ground truth.

    .venv/bin/python vision_check.py            # asks for the OpenRouter key (hidden, not saved)
    .venv/bin/python vision_check.py --dry-run  # no API calls, checks the plumbing

Scope (deliberately small, chosen before seeing any model output):
  - only items in the blind gold set, so every answer can be scored
  - screenshots and single photos only: duplicates and bursts are relational judgements a
    single-image prompt cannot see, and videos are out of scope
  - photos where the face detector found a face are NOT uploaded (privacy) unless --include-faces
  - images are downscaled to 512 px and sent with detail=low

Output: data/vision_results.json (private, per item) and results/vision_metrics.md (aggregate only).
"""
import argparse, base64, csv, getpass, io, json, os, random, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import request

from PIL import Image, ImageOps
try:
    import pillow_heif; pillow_heif.register_heif_opener()
except Exception:
    pass

DATA, OUT = Path('data'), Path('results')
URL = 'https://openrouter.ai/api/v1/chat/completions'
PRICE_IN_PER_M, PRICE_OUT_PER_M = 0.15, 0.60      # openai/gpt-4o-mini list price on OpenRouter

SYSTEM = (
    "You help one person clean up their own phone photo library when storage is full. "
    "You see one image at a time. Decide whether it is a reasonable DELETE CANDIDATE.\n"
    "delete: clearly disposable now — e.g. an expired or completed-task screenshot (receipts, codes, "
    "tickets, forms, chats or pages saved for a one-off purpose), a meme or saved social-media screen, "
    "an accidental or meaningless shot.\n"
    "keep: likely to matter later — people, places, events, documents still useful, anything personal.\n"
    "unsure: you cannot tell from the image alone.\n"
    "Deleting a meaningful photo is far worse than keeping a useless one, so prefer unsure over a "
    "guess. Nothing is deleted automatically; a person confirms every suggestion.\n"
    'Reply with JSON only: {"decision": "delete|keep|unsure", "confidence": 0-1, "reason": "<= 15 words"}'
)


def last_blind():
    if not (DATA / 'labels.csv').exists():
        raise SystemExit('no labels yet — run server.py and finish blind labelling (盲标) first')
    last = {}
    for r in csv.DictReader((DATA / 'labels.csv').open()):
        if r['mode'] == 'blind':
            last[r['id']] = r['action']
    return {k: v for k, v in last.items() if v != 'undone'}


def encode(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    im.thumbnail((512, 512))
    buf = io.BytesIO(); im.save(buf, 'JPEG', quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def ask(key, model, it, dry):
    t0 = time.time()
    if dry:
        time.sleep(0.01)
        d = random.Random(it['id']).choice(['delete', 'keep', 'unsure'])
        return dict(decision=d, confidence=0.5, reason='dry run', tin=0, tout=0, ms=10, raw='')
    body = dict(model=model, temperature=0, max_tokens=80, messages=[
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': [
            {'type': 'text', 'text': 'Should this image be a delete candidate?'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + encode(it['path']), 'detail': 'low'}}]}])
    req = request.Request(URL, json.dumps(body).encode(), {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
    for attempt in range(3):
        try:
            r = json.load(request.urlopen(req, timeout=60))
            break
        except Exception as e:
            if attempt == 2:
                return dict(decision='error', confidence=0, reason=str(e)[:120], tin=0, tout=0, ms=int((time.time() - t0) * 1000), raw='')
            time.sleep(2 * (attempt + 1))
    raw = r['choices'][0]['message']['content'].strip()
    u = r.get('usage', {})
    try:
        j = json.loads(raw[raw.find('{'): raw.rfind('}') + 1])
        d = str(j.get('decision', '')).lower().strip()
        d = d if d in ('delete', 'keep', 'unsure') else 'unparseable'
        conf, reason = float(j.get('confidence', 0)), str(j.get('reason', ''))[:160]
    except Exception:
        d, conf, reason = 'unparseable', 0.0, raw[:160]
    return dict(decision=d, confidence=conf, reason=reason, tin=u.get('prompt_tokens', 0),
                tout=u.get('completion_tokens', 0), ms=int((time.time() - t0) * 1000), raw=raw[:300])


def pct(a, b):
    return f'{a / b:.0%}' if b else '—'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='openai/gpt-4o-mini')
    ap.add_argument('--include-faces', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()

    items = {it['id']: it for it in json.loads((DATA / 'items.json').read_text())}
    gold = last_blind()
    pool = [items[i] | {'gold': g} for i, g in gold.items() if i in items]
    subset = [it for it in pool if it['type'] in ('screenshot', 'photo') and not it['exact_dup_of']]
    skipped_faces = [it for it in subset if it.get('faces')]
    if not a.include_faces:
        subset = [it for it in subset if not it.get('faces')]
    print(f'gold set {len(pool)} · eligible screenshots+single photos {len(subset) + (0 if a.include_faces else len(skipped_faces))}'
          f' · not uploaded (face detected) {0 if a.include_faces else len(skipped_faces)} · sending {len(subset)}')

    key = '' if a.dry_run else (os.environ.get('OPENROUTER_API_KEY') or getpass.getpass('Paste your OpenRouter API key (hidden, not saved): ').strip())
    t0 = time.time()
    done = [0]

    def run(it):
        ans = ask(key, a.model, it, a.dry_run)
        done[0] += 1
        if done[0] == 1 and ans['decision'] == 'error':
            print(f'  first call failed: {ans["reason"]}', flush=True)
        if done[0] % 10 == 0 or done[0] == len(subset):
            print(f'  {done[0]}/{len(subset)} images checked · {time.time() - t0:.0f}s', flush=True)
        return ans

    print('  sending… (progress every 10 images)', flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        answers = list(ex.map(run, subset))
    wall = time.time() - t0
    rows = [it | {'vision': ans} for it, ans in zip(subset, answers)]
    if not a.dry_run:
        (DATA / 'vision_results.json').write_text(json.dumps(
        [dict(id=r['id'], type=r['type'], status=r['status'], gold=r['gold'], **r['vision']) for r in rows], ensure_ascii=False, indent=1))

    # ---- metrics ------------------------------------------------------------------------
    ok = [r for r in rows if r['vision']['decision'] in ('delete', 'keep', 'unsure')]
    tin = sum(r['vision']['tin'] for r in rows); tout = sum(r['vision']['tout'] for r in rows)
    cost = tin / 1e6 * PRICE_IN_PER_M + tout / 1e6 * PRICE_OUT_PER_M
    per_img = cost / max(1, len(rows))
    lat = sorted(r['vision']['ms'] for r in rows)

    def block(name, rs, pred):
        cm = defaultdict(Counter)
        for r in rs:
            cm[pred(r)][r['gold']] += 1
        said_del = sum(cm['delete'].values()); hit = cm['delete']['delete']
        gold_del = sum(r['gold'] == 'delete' for r in rs)
        return dict(name=name, n=len(rs), said_delete=said_del, precision=pct(hit, said_del), recall=pct(hit, gold_del),
                    silent=cm['delete']['keep'], abstain=pct(sum(cm['unsure'].values()), len(rs)), cm=cm)

    vis = lambda r: r['vision']['decision']
    rules = lambda r: r['status']
    residual = [r for r in ok if r['status'] == 'keep']            # what rules called "no problem"
    shots = [r for r in ok if r['type'] == 'screenshot']
    hybrid = lambda r: 'delete' if r['status'] == 'delete' and r['type'] == 'screenshot' and vis(r) != 'keep' else (
        vis(r) if r['status'] == 'keep' else r['status'])
    blocks = [block('Rules (pipeline status)', ok, rules), block('Vision model alone', ok, vis),
              block('Hybrid: rules, vision vetoes screenshots & judges the rules\' "keep" residual', ok, hybrid),
              block('Screenshots — rules', shots, rules), block('Screenshots — vision', shots, vis),
              block('Rules\' "no problem found" residual — vision', residual, vis)]

    lib = len(items); resid_share = sum(1 for it in items.values() if it['status'] == 'keep' and it['kind'] == 'photo') / lib
    L = ['# Vision-model escalation — results\n',
         f'Model `{a.model}` via OpenRouter, temperature 0, 512 px, detail=low. Ground truth = my blind labels. '
         f'{len(rows)} images sent ({len(skipped_faces) if not a.include_faces else 0} with detected faces withheld for privacy); '
         f'{len(rows) - len(ok)} errors/unparseable.\n',
         '| Policy | n | Said delete | Precision of delete | Recall of my deletes | Silent failures (delete→I kept) | Unsure rate |',
         '|---|---:|---:|---:|---:|---:|---:|']
    for b in blocks:
        L.append(f'| {b["name"]} | {b["n"]} | {b["said_delete"]} | {b["precision"]} | {b["recall"]} | {b["silent"]} | {b["abstain"]} |')
    L += ['', '## Vision decision vs my label (all sent)', '', '| Vision \\ I said | delete | keep | unsure |', '|---|---:|---:|---:|']
    for s in ('delete', 'unsure', 'keep'):
        c = blocks[1]['cm'][s]; L.append(f'| {s} | {c["delete"]} | {c["keep"]} | {c["unsure"]} |')
    L += ['', '## Cost and latency (measured)', '',
          f'- Tokens: {tin:,} in / {tout:,} out → **${cost:.4f}** for {len(rows)} images = **${per_img:.6f} per image**',
          f'- Latency per call: median {lat[len(lat) // 2] if lat else 0} ms, p90 {lat[int(len(lat) * .9)] if lat else 0} ms; wall time {wall:.0f} s with {a.workers} parallel calls',
          f'- Escalating only the rules\' "keep" residual ({resid_share:.0%} of this library) for a 30,000-item library: '
          f'≈ {int(30000 * resid_share):,} calls ≈ **${30000 * resid_share * per_img:.2f}**; sending everything ≈ ${30000 * per_img:.2f}',
          '', 'Privacy: images were my own, uploaded with my consent for this run only, downscaled, faces withheld; '
          'the product design keeps a local-only mode with no upload.']
    if a.dry_run:
        L.insert(1, '**DRY RUN — random answers, not results.**\n')
    else:
        OUT.mkdir(exist_ok=True)
        (OUT / 'vision_metrics.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    unp = [r for r in rows if r['vision']['decision'] not in ('delete', 'keep', 'unsure')]
    if unp:
        print('\nfirst error/unparseable:', unp[0]['vision']['reason'])


if __name__ == '__main__':
    main()

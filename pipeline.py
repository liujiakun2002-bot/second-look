"""
Second Look — pipeline.

Scans one folder of photos/videos, computes cheap features, groups near-duplicates,
scores each item, and writes data/items.json plus a fixed random blind sample.

    python3 pipeline.py ~/Desktop/SecondLookPhotos

Stages (all local, no network, no foundation model in this version):
  1. rules     exact duplicates (SHA-256), screenshots (filename / no camera / phone aspect),
               dark frames (mean brightness), age
  2. quality   sharpness = variance of the Laplacian (classical statistic, not ML)
  3. grouping  near-duplicates via 64-bit difference hash + capture-time proximity
  4. scoring   p_delete (hand-set, explainable) and a status: delete / unsure / keep
  5. ranking   five orderings written side by side so evaluate.py can compare them
"""
import argparse, hashlib, json, os, random, re, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:                                   # iPhone HEIC support, optional
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

try:                                   # face detector, optional (opencv-python)
    import cv2
    _FACE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    cv2, _FACE = None, None

IMG_EXT = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp'}
VID_EXT = {'.mov', '.mp4', '.m4v', '.avi'}
SHOT_NAME = re.compile(r'screenshot|screen shot|截屏|屏幕快照|屏幕截图', re.I)

# ---- thresholds: fixed here, before any labels are seen -----------------------------
BLUR_VAR = 60.0          # Laplacian variance on a 512px grey image; below = blurry
DARK_MEAN = 25.0         # mean brightness 0-255; below = probably an accidental frame
NEAR_DUP_BITS = 10       # dHash Hamming distance for "same moment"
BURST_SECONDS = 30       # near-duplicates must also be this close in time...
LOOSE_DUP_BITS = 4       # ...unless they are almost identical
OLD_DAYS = 730
DELETE_AT, UNSURE_AT = 0.60, 0.35   # p_delete >= 0.60 -> delete, >= 0.35 -> unsure
BLIND_N, BLIND_SEED = 300, 42


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def capture_time(img, path):
    try:
        ex = img.getexif()
        raw = ex.get_ifd(0x8769).get(36867) or ex.get(306)   # DateTimeOriginal, DateTime
        if raw:
            return datetime.strptime(str(raw).strip()[:19], '%Y:%m:%d %H:%M:%S').timestamp(), True
    except Exception:
        pass
    return os.path.getmtime(path), False


def camera_make(img):
    try:
        return str(img.getexif().get(271) or '').strip()
    except Exception:
        return ''


def laplacian_var(grey):
    g = grey.astype(np.float32)
    lap = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1])
    return float(lap.var())


def dhash(img):
    small = np.asarray(img.convert('L').resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(''.join('1' if b else '0' for b in bits), 2)


def faces(grey):
    if _FACE is None:
        return 0
    try:
        return len(_FACE.detectMultiScale(grey, scaleFactor=1.15, minNeighbors=6, minSize=(40, 40)))
    except Exception:
        return 0


def analyse_image(path):
    with Image.open(path) as im:
        make = camera_make(im)
        ts, exif_time = capture_time(im, path)
        im = ImageOps.exif_transpose(im)
        w, h = im.size
        im.thumbnail((512, 512))
        grey = np.asarray(im.convert('L'))
        return dict(width=w, height=h, make=make, ts=ts, exif_time=exif_time,
                    sharpness=round(laplacian_var(grey), 1),
                    brightness=round(float(grey.mean()), 1),
                    dhash=dhash(im), faces=faces(grey))


def scan(folder):
    items = []
    files = sorted(p for p in Path(folder).expanduser().rglob('*')
                   if p.is_file() and not p.name.startswith('.') and p.suffix.lower() in IMG_EXT | VID_EXT)
    for n, p in enumerate(files, 1):
        kind = 'video' if p.suffix.lower() in VID_EXT else 'photo'
        it = dict(id=f'{n:05d}', path=str(p.resolve()), name=p.name, kind=kind,
                  size=p.stat().st_size, sha=sha256(p))
        if kind == 'photo':
            try:
                it.update(analyse_image(p))
            except Exception as e:
                print(f'  skip {p.name}: {e}', file=sys.stderr)
                continue
        else:
            it.update(ts=p.stat().st_mtime, exif_time=False)
        items.append(it)
        if n % 100 == 0:
            print(f'  analysed {n}/{len(files)}')
    return items


def is_screenshot(it):
    if it['kind'] != 'photo':
        return False
    if SHOT_NAME.search(it['name']):
        return True
    aspect = max(it['width'], it['height']) / max(1, min(it['width'], it['height']))
    return not it['make'] and (it['name'].lower().endswith('.png') or aspect >= 1.9)


class UF:
    def __init__(self, n): self.p = list(range(n))
    def f(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def u(self, a, b): self.p[self.f(a)] = self.f(b)


def group(items):
    """Exact duplicates by hash; near-duplicate groups by dHash (+ time for looser matches)."""
    first_by_sha = {}
    for it in sorted(items, key=lambda x: x['ts']):
        it['exact_dup_of'] = first_by_sha.get(it['sha'])
        first_by_sha.setdefault(it['sha'], it['id'])
    photos = [it for it in items if it['kind'] == 'photo' and not it['exact_dup_of']]
    uf = UF(len(photos))
    for i in range(len(photos)):
        for j in range(i + 1, len(photos)):
            a, b = photos[i], photos[j]
            d = bin(a["dhash"] ^ b["dhash"]).count("1")
            if d <= LOOSE_DUP_BITS or (d <= NEAR_DUP_BITS and abs(a['ts'] - b['ts']) <= BURST_SECONDS):
                uf.u(i, j)
    groups = {}
    for i, it in enumerate(photos):
        groups.setdefault(uf.f(i), []).append(it)
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda x: x['ts'])
        best = max(members, key=lambda x: x['sharpness'])
        gid = 'G' + members[0]['id']
        for m in members:
            m.update(group=gid, group_size=len(members), group_best=(m is best))


def mb(b):
    return f'{b / 1e9:.1f} GB' if b >= 1e9 else (f'{b / 1e6:.0f} MB' if b >= 1e7 else f'{b / 1e6:.1f} MB')


def score(it, now):
    """Hand-set, explainable delete score. Returns (p_delete, p_rules, reason)."""
    age_y = (now - it['ts']) / 86400 / 365
    old = age_y * 365 >= OLD_DAYS
    shot = it['screenshot']
    # rules-only view: what hashing + metadata alone would say
    if it['exact_dup_of']:
        p_rules = 0.95
    elif shot:
        p_rules = 0.60
    else:
        p_rules = 0.10

    if it['exact_dup_of']:
        return 0.95, p_rules, '和另一个文件完全相同（哈希一致）'
    if it['kind'] == 'video':
        p = 0.45 + (0.10 if old else 0)
        return p, p_rules, f'视频 {mb(it["size"])}' + (f'，拍摄于 {age_y:.0f} 年前' if old else '') + '，系统看不懂视频内容'
    if it.get('group') and not it['group_best']:
        p = 0.80 if shot else 0.75
        what = '重复截图' if shot else f'{it["group_size"]} 张相似照片之一'
        return p, p_rules, f'{what}，同组里有更清晰的一张'
    if it['brightness'] < DARK_MEAN:
        return 0.70, p_rules, '画面几乎全黑，可能是误拍'
    blurry = it['sharpness'] < BLUR_VAR
    if blurry and it['faces']:
        return 0.45, p_rules, '有人脸但比较模糊，可能是重要的人'
    if shot:
        p = 0.60 + (0.10 if old else 0)
        return p, p_rules, '截图' + (f'，{age_y:.0f} 年前的' if old else '') + '（内容是否还有用需要你判断）'
    if blurry:
        return 0.62, p_rules, '照片模糊（清晰度低）'
    if it.get('group_best'):
        return 0.10, p_rules, f'{it["group_size"]} 张相似照片里最清晰的一张'
    return 0.15 + (0.05 if old else 0), p_rules, '没有发现问题'


def status(p, it):
    if it['kind'] == 'video':
        return 'unsure'                  # never pretend to judge a video it cannot see
    return 'delete' if p >= DELETE_AT else ('unsure' if p >= UNSURE_AT else 'keep')


def rank(items):
    """Five orderings. Lower rank = shown earlier."""
    orders = {
        'chrono':       sorted(items, key=lambda x: -x['ts']),                    # what shipped apps do
        'size_only':    sorted(items, key=lambda x: -x['size']),                  # "just show big files"
        'rules_only':   sorted(items, key=lambda x: (-x['p_rules'], -x['ts'])),   # hash + metadata only
        'p_only':       sorted(items, key=lambda x: (-x['p_delete'], -x['size'])),# full pipeline, no size weight
        'second_look':  sorted(items, key=lambda x: -(x['p_delete'] * x['size'])),# expected bytes freed
    }
    for name, order in orders.items():
        for r, it in enumerate(order):
            it.setdefault('rank', {})[name] = r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--out', default='data')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    t0 = time.time()
    print(f'scanning {a.folder} (face detector: {"on" if _FACE is not None else "off — pip install opencv-python"})')
    items = scan(a.folder)
    if not items:
        sys.exit('no photos or videos found')
    for it in items:
        it['screenshot'] = is_screenshot(it)
        it.setdefault('group', None); it.setdefault('group_size', 1); it.setdefault('group_best', False)
    group(items)
    now = time.time()
    for it in items:
        it['p_delete'], it['p_rules'], it['reason'] = score(it, now)
        it['status'] = status(it['p_delete'], it)
        it['type'] = ('video' if it['kind'] == 'video' else 'screenshot' if it['screenshot']
                      else 'burst' if it['group'] else 'photo')
        it['date'] = datetime.fromtimestamp(it['ts']).strftime('%Y-%m-%d')
        it.pop('dhash', None); it.pop('sha', None)
    rank(items)
    (out / 'items.json').write_text(json.dumps(items, ensure_ascii=False, indent=1))

    blind = out / 'blind_sample.json'
    if blind.exists():
        print(f'kept existing {blind} (the gold sample is fixed once drawn)')
    else:
        ids = [it['id'] for it in items]
        random.Random(BLIND_SEED).shuffle(ids)
        blind.write_text(json.dumps(ids[:BLIND_N]))
        print(f'drew blind sample of {min(BLIND_N, len(ids))} (seed {BLIND_SEED}) -> {blind}')

    from collections import Counter
    c = Counter(it['status'] for it in items); t = Counter(it['type'] for it in items)
    print(f'{len(items)} items in {time.time() - t0:.1f}s · types {dict(t)} · status {dict(c)}')
    print(f'groups: {len({it["group"] for it in items if it["group"]})} · exact duplicates: '
          f'{sum(1 for it in items if it["exact_dup_of"])} · total {mb(sum(it["size"] for it in items))}')


if __name__ == '__main__':
    main()

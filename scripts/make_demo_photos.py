"""
Generate a synthetic demo library so the repository runs without anyone's private photos.

    python3 scripts/make_demo_photos.py demo_photos

Produces bursts with varying blur, exact duplicates, phone-shaped screenshots (no camera EXIF),
dark accidental frames, ordinary photos, and a few dummy "videos" (random bytes; the pipeline
only reads their size and date). Ground truth for the demo is NOT generated here on purpose:
labels come from a person swiping in blind mode.
"""
import os, random, shutil, sys, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

R = random.Random(7)
W, H = 1600, 1200


def exif(ts, make='Apple'):
    e = Image.Exif()
    e[271] = make
    e[306] = time.strftime('%Y:%m:%d %H:%M:%S', time.localtime(ts))
    return e


def scene(seed):
    r = random.Random(seed)
    im = Image.new('RGB', (W, H), (r.randint(80, 200), r.randint(120, 220), r.randint(160, 240)))
    d = ImageDraw.Draw(im)
    for _ in range(40):
        x, y = r.randint(0, W), r.randint(H // 3, H)
        s = r.randint(40, 260)
        d.rectangle([x, y, x + s, y + s // 2], fill=(r.randint(0, 255), r.randint(0, 255), r.randint(0, 255)))
    for _ in range(25):
        x, y = r.randint(0, W), r.randint(0, H)
        d.ellipse([x, y, x + 60, y + 60], outline=(20, 20, 20), width=6)
    return im


def shot(seed):
    r = random.Random(seed)
    im = Image.new('RGB', (1179, 2556), (245, 245, 245))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 1179, 220], fill=(r.randint(0, 120), r.randint(60, 160), r.randint(120, 230)))
    for y in range(300, 2400, 90):
        d.rectangle([60, y, 60 + r.randint(300, 1000), y + 40], fill=(200, 200, 200))
    return im


def main(out):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    now = time.time()
    n = 0

    def save(im, name, ts, make='Apple', fmt='JPEG'):
        nonlocal n
        n += 1
        p = out / name
        if fmt == 'JPEG':
            im.save(p, 'JPEG', quality=88, exif=exif(ts, make) if make else Image.Exif())
        else:
            im.save(p, fmt)
        os.utime(p, (ts, ts))
        return p

    # 12 bursts of 3-8 frames, one sharp frame each
    for b in range(12):
        base = scene(100 + b)
        ts = now - R.randint(10, 1400) * 86400
        k = R.randint(3, 8)
        sharp = R.randrange(k)
        for i in range(k):
            im = base if i == sharp else base.filter(ImageFilter.GaussianBlur(R.uniform(2.5, 7)))
            im = im.transform(im.size, Image.AFFINE, (1, 0, R.randint(-6, 6), 0, 1, R.randint(-6, 6)))
            save(im, f'IMG_{1000 + n}.JPG', ts + i)

    # 40 ordinary distinct photos, a few blurry, a few dark
    for i in range(40):
        im = scene(500 + i)
        if i % 9 == 0:
            im = im.filter(ImageFilter.GaussianBlur(8))
        if i % 13 == 0:
            im = Image.eval(im, lambda v: v // 14)
        save(im, f'IMG_{1000 + n}.JPG', now - R.randint(1, 1500) * 86400)

    # 20 screenshots (no camera make), 3 of them repeated with identical bytes
    shots = []
    for i in range(20):
        shots.append(save(shot(900 + i), f'Screenshot_{2024 + i % 3}-{i:02d}.png', now - R.randint(1, 1200) * 86400, make=None, fmt='PNG'))
    for s in shots[:3]:
        n += 1
        shutil.copy2(s, out / s.name.replace('.png', ' copy.png'))

    # 4 exact duplicate photos
    photos = sorted(out.glob('IMG_*.JPG'))
    for p in R.sample(photos, 4):
        n += 1
        shutil.copy2(p, out / p.name.replace('.JPG', '(1).JPG'))

    # 5 dummy videos, 8-40 MB of random bytes
    for i in range(5):
        p = out / f'MOV_{i:03d}.MOV'
        p.write_bytes(os.urandom(R.randint(8, 40) * 1_000_000))
        ts = now - R.randint(30, 1400) * 86400
        os.utime(p, (ts, ts))
        n += 1

    print(f'wrote {n} files to {out}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'demo_photos')

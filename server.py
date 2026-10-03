"""
Second Look — local swipe server (standard library only + Pillow).

    python3 server.py            then open http://localhost:8501

Two modes, both writing to data/labels.csv:
  ranked  the product: cards in Second Look order, with recommendation and reason
  blind   the gold set: the fixed random sample, shown with NO recommendation or reason,
          so my labels are independent of the ranker (instructor feedback, Section 7)
Nothing is deleted. A "delete" swipe is only a label.
"""
import csv, io, json, sys, time, uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from PIL import Image, ImageOps, ImageDraw
try:
    import pillow_heif; pillow_heif.register_heif_opener()
except Exception:
    pass
try:
    import cv2
except Exception:
    cv2 = None

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
LABELS = DATA / 'labels.csv'
THUMBS = DATA / 'thumbs'
FIELDS = ['ts', 'session', 'mode', 'id', 'action', 'system_status', 'p_delete', 'size', 'type']
SESSION = uuid.uuid4().hex[:8]

ITEMS = {it['id']: it for it in json.loads((DATA / 'items.json').read_text())}
BLIND = json.loads((DATA / 'blind_sample.json').read_text())


def labelled(mode):
    """Latest action per id for a mode (an 'undone' row cancels the previous one)."""
    last = {}
    if LABELS.exists():
        for r in csv.DictReader(LABELS.open()):
            if r['mode'] == mode:
                last[r['id']] = r['action']
    return {k for k, v in last.items() if v != 'undone'}


def append(rows):
    new = not LABELS.exists()
    with LABELS.open('a', newline='') as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def public(it, blind=False):
    base = dict(id=it['id'], size=it['size'], kind=it['kind'], date=it['date'])
    if blind:
        return base
    return base | dict(type=it['type'], status=it['status'], reason=it['reason'],
                       p=round(it['p_delete'], 2), best=it['group_best'])


def queue(mode):
    done = labelled(mode)
    if mode == 'blind':
        return [dict(card='single', items=[public(ITEMS[i], blind=True)]) for i in BLIND if i in ITEMS and i not in done]
    cards, used = [], set()
    for it in sorted(ITEMS.values(), key=lambda x: x['rank']['second_look']):
        if it['id'] in used or it['id'] in done:
            continue
        if it['group']:
            members = [m for m in ITEMS.values() if m['group'] == it['group'] and m['id'] not in done]
            members.sort(key=lambda m: m['ts'])
            used.update(m['id'] for m in members)
            if len(members) > 1:
                cards.append(dict(card='burst', items=[public(m) for m in members]))
                continue
        used.add(it['id'])
        cards.append(dict(card='single', items=[public(it)]))
    return cards


def summary():
    items = list(ITEMS.values())
    by = lambda t: sum(it['size'] for it in items if it['type'] == t and it['status'] != 'keep')
    return dict(count=len(items), bytes=sum(it['size'] for it in items),
                photos=sum(it['kind'] == 'photo' for it in items),
                videos=sum(it['kind'] == 'video' for it in items),
                reclaim=dict(video=by('video'), burst=by('burst'), screenshot=by('screenshot'),
                             photo=by('photo')),
                blind_total=len(BLIND), blind_done=len(labelled('blind') & set(BLIND)))


def placeholder(text):
    im = Image.new('RGB', (600, 750), (28, 32, 30))
    d = ImageDraw.Draw(im)
    d.ellipse([250, 325, 350, 425], fill=(235, 238, 236))
    d.polygon([(285, 345), (285, 405), (335, 375)], fill=(28, 32, 30))
    d.text((24, 700), text, fill=(200, 205, 202))
    return im


def thumb(it, w):
    THUMBS.mkdir(parents=True, exist_ok=True)
    cache = THUMBS / f'{it["id"]}_{w}.jpg'
    if not cache.exists():
        im = None
        if it['kind'] == 'video' and cv2 is not None:
            cap = cv2.VideoCapture(it['path']); ok, frame = cap.read(); cap.release()
            if ok:
                im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if it['kind'] == 'video' and im is None:
            im = placeholder(it['name'])
        if im is None:
            im = ImageOps.exif_transpose(Image.open(it['path']))
        im = im.convert('RGB'); im.thumbnail((w, w * 2))
        im.save(cache, 'JPEG', quality=82)
    return cache.read_bytes()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype='application/json', code=200):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Cache-Control', 'max-age=3600' if ctype == 'image/jpeg' else 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def stream(self, path):
        """Serve a local video with HTTP Range support so the browser can play and seek."""
        size = Path(path).stat().st_size
        ext = Path(path).suffix.lower()
        ctype = {'.mp4': 'video/mp4', '.m4v': 'video/mp4', '.mov': 'video/quicktime'}.get(ext, 'application/octet-stream')
        start, end = 0, size - 1
        rng = self.headers.get('Range')
        if rng and rng.startswith('bytes='):
            a, _, b = rng[6:].partition('-')
            start = int(a) if a else max(0, size - int(b))
            end = int(b) if a and b else end
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        else:
            self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end - start + 1))
        self.end_headers()
        try:
            with open(path, 'rb') as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path in ('/', '/index.html'):
            return self.send((ROOT / 'ui' / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        if u.path == '/api/summary':
            return self.send(summary())
        if u.path == '/api/queue':
            return self.send(queue(q.get('mode', ['ranked'])[0]))
        if u.path.startswith('/video/'):
            it = ITEMS.get(u.path[7:])
            if not it or it['kind'] != 'video':
                return self.send({'error': 'not found'}, code=404)
            return self.stream(it['path'])
        if u.path.startswith('/img/'):
            it = ITEMS.get(u.path[5:])
            if not it:
                return self.send({'error': 'not found'}, code=404)
            return self.send(thumb(it, int(q.get('w', ['600'])[0])), 'image/jpeg')
        self.send({'error': 'not found'}, code=404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
        mode = body.get('mode', 'ranked')
        if self.path in ('/api/label', '/api/undo'):
            rows = []
            for lab in body.get('labels', []):
                it = ITEMS.get(lab['id'])
                if not it:
                    continue
                act = 'undone' if self.path == '/api/undo' else lab['action']
                if act not in ('keep', 'delete', 'unsure', 'undone'):
                    continue
                rows.append(dict(ts=int(time.time()), session=SESSION, mode=mode, id=it['id'], action=act,
                                 system_status=it['status'], p_delete=round(it['p_delete'], 3),
                                 size=it['size'], type=it['type']))
            append(rows)
            return self.send({'ok': True, 'written': len(rows)})
        self.send({'error': 'not found'}, code=404)


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    print(f'Second Look running on http://localhost:{port}  (Ctrl+C to stop) · {len(ITEMS)} items · session {SESSION}')
    ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()

# Second Look — on-device triage for overloaded photo libraries

PE6201 End-of-Course Project · Liu Jiakun

Second Look ranks a photo library so the few hundred items worth a human glance come first,
then lets a person swipe through them. **It never deletes anything.** A swipe is only a label.

This repository is the smallest end-to-end version (Problem Statement §9):
one folder → hashing + screenshot rules + sharpness + near-duplicate grouping → ranked cards →
a swipe page that writes every decision to a labels file. No foundation model is called.

## Run it (macOS, Python 3.9+)

```bash
git clone <this repo> && cd second-look
./run.sh                                  # synthetic demo library, no private photos needed
./run.sh ~/Desktop/SecondLookPhotos       # your own exported folder
```

`run.sh` creates a virtual environment, installs `requirements.txt`, runs the pipeline and opens
`http://localhost:8501`. Everything runs on your machine; nothing is uploaded.
After labelling, run `.venv/bin/python evaluate.py` to write `results/metrics.md`.

Manual equivalent:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/make_demo_photos.py demo_photos   # optional
.venv/bin/python pipeline.py demo_photos
.venv/bin/python server.py
.venv/bin/python evaluate.py
```

## How it works

| Stage | Method | AI? |
|---|---|---|
| Exact duplicates | SHA-256 of file bytes | No — rule |
| Screenshots | file name, no camera make in EXIF, PNG or phone aspect ratio | No — rule |
| Accidental frames | mean brightness < 25 | No — rule |
| Sharpness | variance of the Laplacian on a 512 px grey image | No — classical statistic |
| Near-duplicates / bursts | 64-bit difference hash, Hamming ≤ 10 within 30 s (≤ 4 at any time); sharpest frame kept | No — perceptual hash |
| Faces (optional) | OpenCV Haar cascade; a blurry photo with a face becomes "unsure" | Classical ML detector |
| Score | hand-set `p_delete` per rule, then **rank by `p_delete × file size`** (expected bytes freed) | No |
| Status | `delete` ≥ 0.60, `unsure` ≥ 0.35, else `keep`. Videos are always `unsure`: the system cannot see their content | — |

All thresholds are fixed at the top of `pipeline.py` and were set before any labels were collected.

## Evaluation design

Two swipe modes, both writing to `data/labels.csv`:

- **Ranked** — the product: cards in Second Look order, with recommendation and reason.
- **Blind** — the gold set: a random sample of 300 (seed 42, drawn once by `pipeline.py` and never redrawn),
  shown **without** recommendation or reason. Labelling what the ranker surfaced would never reveal what it buried,
  so the gold set is labelled independently of the ranking.

`evaluate.py` replays five orderings over the blind labels and scores the first 50:

| Ordering | Why it is here |
|---|---|
| Reverse-chronological | what shipped swipe-to-delete apps do (baseline) |
| Largest first | isolates the contribution of size weighting alone |
| Rules only | hash duplicates + screenshots, no quality or grouping |
| `p_delete` only | the full pipeline without size weighting |
| **Second Look** (`p_delete × size`) | the proposed ranking |

Headline metric: **GB freed per 50 swipes**. Reported alongside: precision@50, byte recall@50, the
system-vs-me confusion table, abstention rate (how often it says "unsure" and what I said on those),
silent failures (system said delete, I said keep), and precision by content type.

## Vision-model test (optional, uploads images)

`vision_check.py` sends the blind-labelled screenshots and single photos (512 px; photos with a
detected face are withheld) to `openai/gpt-4o-mini` via OpenRouter and scores the answers against the
blind labels. It writes per-item answers to `data/` (git-ignored) and aggregates to
`results/vision_metrics.md`.

```bash
.venv/bin/python vision_check.py --dry-run          # no API calls, checks the plumbing
export OPENROUTER_API_KEY=sk-or-v1-...              # your own key
.venv/bin/python vision_check.py
```

## Reproducing without private photos

`./run.sh` with no folder builds a synthetic library (`scripts/make_demo_photos.py`), so every step —
pipeline, swipe page, blind labelling, `evaluate.py`, `vision_check.py --dry-run` — runs on any Mac.
My own results are in `results/`; the photos and labels behind them are not published.

## Privacy

Photos, file paths, thumbnails and raw labels live in `data/` and are git-ignored. Only aggregate
numbers in `results/` are committed. The demo library is synthetic so the repository runs without them.

## Limitations of this version

- The gold set is one person's library labelled by that same person, so it measures fit to one user's preferences.
- No content understanding: an expired boarding pass and a precious message screenshot score the same.
  That judgement needs a vision model and is the next layer, escalated only for the ambiguous residual.
- Video duplicates and "last opened" dates are not available from files, so video ranking is by size and age only.
- Sentimental value is not in the pixels. The mitigation is design, not modelling: nothing is deleted without a person.

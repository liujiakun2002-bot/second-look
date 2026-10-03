#!/bin/bash
# Second Look — one-command run on macOS.
#   ./run.sh                                  demo library (synthetic, no private photos)
#   ./run.sh ~/Desktop/SecondLookPhotos       your own folder
#   ./run.sh ~/Desktop/SecondLookPhotos --rescan   re-run the pipeline (keeps labels and blind sample)
set -e
cd "$(dirname "$0")"
FOLDER="${1:-demo_photos}"
if [ ! -d .venv ]; then
  echo "== first run: creating a Python environment (1-2 min) =="
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
if [ "$FOLDER" = "demo_photos" ] && [ ! -d demo_photos ]; then
  .venv/bin/python scripts/make_demo_photos.py demo_photos
fi
if [ ! -f data/items.json ] || [ "$2" = "--rescan" ]; then
  .venv/bin/python pipeline.py "$FOLDER"
fi
(sleep 1.5 && open http://localhost:8501) &
.venv/bin/python server.py 8501

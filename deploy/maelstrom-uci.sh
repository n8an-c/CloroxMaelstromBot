#!/usr/bin/env bash
# Launcher lichess-bot calls as the "engine". Adjust paths if you move things.
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
exec "$HERE/.venv/bin/maelstrom-uci" --model maia3-23m --adapter models/adapter_maia3-23m.pt \
     --think-time models/think_time.pkl --book data/processed/positions.parquet --book-splits train,val,test --threads 2

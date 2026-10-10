#!/usr/bin/env bash
# Phase 0 end to end. Run from the repo root on a machine that can reach lichess.org / api.chess.com.
# Requires configs/local.yaml with your usernames (see configs/data.yaml) and optionally LICHESS_TOKEN.
set -euo pipefail
source .venv/bin/activate
LI=$(python -c "from maelstrom.config import load_config; print(load_config()['players'].get('lichess') or '')")
CC=$(python -c "from maelstrom.config import load_config; print(load_config()['players'].get('chesscom') or '')")
PERF=$(python -c "from maelstrom.config import load_config; print(','.join(load_config()['perf_types']))")
[ -n "$LI" ] && maelstrom-export-lichess --user "$LI" --perf "$PERF"
[ -n "$CC" ] && maelstrom-export-chesscom --user "$CC"
maelstrom-parse
maelstrom-split
[ -x tools/stockfish/stockfish-ubuntu-x86-64-avx2 ] || scripts/fetch_stockfish.sh
maelstrom-annotate            # ~1-3 h for 160k positions at depth 13 on 4 cores; resumable
maelstrom-eda
echo "done -> reports/eda.md"

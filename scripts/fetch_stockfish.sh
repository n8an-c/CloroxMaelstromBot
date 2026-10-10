#!/usr/bin/env bash
# Download a Stockfish 17.1 binary into tools/stockfish/ (Linux x86-64 avx2 by default).
# Usage: scripts/fetch_stockfish.sh [avx2|bmi2|sse41-popcnt|x86-64]
set -euo pipefail
ARCH="${1:-avx2}"
URL="https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1/stockfish-ubuntu-x86-64-${ARCH}.tar"
mkdir -p tools
cd tools
curl -sS -L -o sf.tar "$URL"
tar -xf sf.tar && rm sf.tar
chmod +x stockfish/stockfish-ubuntu-x86-64-${ARCH}
echo "installed: tools/stockfish/stockfish-ubuntu-x86-64-${ARCH}"
printf 'uci\nquit\n' | "stockfish/stockfish-ubuntu-x86-64-${ARCH}" | grep '^id name'

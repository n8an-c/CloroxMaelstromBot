# CloroxMaelstromBot: a chess bot that plays like *me*

Goal: a Lichess bot that reproduces one specific ~2000-rated player's moves, mistakes,
opening choices and time use, learned from ~4,000 of their Lichess + Chess.com games.

Approach (see `docs/plan.md`): a frozen pretrained human-move model (Maia-3 or Maia-2)
plus a small residual "you-adapter" trained on the player's games, a personal opening
book, policy sampling instead of argmax, and a think-time model. Deployed via lichess-bot.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Data: export, parse to Parquet, temporal game-level split, Stockfish annotation, EDA | **code done, tested on synthetic games; waiting for real exports** |
| 1 | Baselines: Maia-2, Maia-3, Stockfish (depth / Skill Level), opening book, kNN | not started |
| 2 | Personalisation: residual adapter on frozen Maia features; ablations, data-size curve | not started |
| 3 | Hybrid UCI engine: book -> sampled policy -> think-time delay; lichess-bot | not started |
| 4 | Human-likeness: distributional tests, stylometry AUC, blinded test | not started |
| 5 | Ship: repo, model card, write-up | not started |

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip setuptools wheel      # Debian's setuptools breaks the `chess` wheel
pip install -e ".[dev]"                          # add ".[model]" for torch (Phase 1+)
scripts/fetch_stockfish.sh                       # Stockfish 17.1 -> tools/stockfish/
cp configs/data.yaml configs/local.yaml          # fill in your usernames (local.yaml is git-ignored)
```

## Phase 0: data

```bash
export LICHESS_TOKEN=lip_...                     # optional; 60 games/s instead of 20
scripts/run_phase0.sh                            # or step by step:
maelstrom-export-lichess --user NAME --perf blitz,rapid     # -> data/raw/lichess/NAME.pgn
maelstrom-export-chesscom --user NAME                       # -> data/raw/chesscom/NAME/YYYY-MM.pgn
maelstrom-parse                                   # -> data/processed/{games,positions}.parquet
maelstrom-split                                   # adds split + seen-in-train flags (by game, temporal test)
maelstrom-annotate --depth 13                     # -> annotations.parquet (your moves; resumable)
maelstrom-eda                                     # -> reports/eda.md + figures
```

`positions.parquet` has one row per ply for *both* sides (`is_user` marks yours) with the
FEN before the move, the move, clock before/after, think time (`before - after + increment`),
both ratings (native and Lichess-equivalent), `%eval` where Lichess provides it, and
phase/material features. `games.parquet` keeps the full UCI move list so any model can rebuild
position history. Splits are assigned by game: the most recent 10% of games are the test set,
a seeded random 10% of the rest is validation.

Daily/correspondence games and chess variants are skipped. Chess.com's `TimeControl`
header is bucketed with the same `base + 40*increment` rule both sites use.

Everything under `data/` is git-ignored: it is your personal game history.

## Tests

```bash
python -m pytest -q      # synthetic Lichess/Chess.com PGNs with %clk and %eval comments
```

## Licensing note

Maia-2 is MIT; Maia-3 (code and weights) is AGPL-3.0, so a bot built on Maia-3 and served
over the network must publish its source. This repository is open source either way.

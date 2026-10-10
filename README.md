# CloroxMaelstromBot: a chess bot that plays like *me*

Goal: a Lichess bot that reproduces one specific ~2000-rated player's moves, mistakes,
opening choices and time use, learned from ~4,000 of their Lichess + Chess.com games.

Approach (see `docs/plan.md`): a frozen pretrained human-move model (Maia-3 or Maia-2)
plus a small residual "you-adapter" trained on the player's games, a personal opening
book, policy sampling instead of argmax, and a think-time model. Deployed via lichess-bot.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Data: export, parse to Parquet, temporal game-level split, Stockfish annotation, EDA | done (4,076 games in scope, 143k own moves) |
| 1 | Baselines: Maia-3 5M/23M, Stockfish (depth / Skill Level), opening book, kNN | done except Maia-2 (weights host blocked) |
| 2 | Personalisation: residual adapter on frozen Maia-3 features; ablations, data-size curve | done: final model (adapter on Maia-3 23M) 61.0% top-1 / NLL 1.145 vs 58.8% / 1.237 for frozen Maia-3 5M on the temporal test split |
| 3 | Hybrid UCI engine: book -> sampled policy -> think-time delay; lichess-bot config | done (live BOT account pending) |
| 4 | Human-likeness: self-play vs Maia-3, distributional tests, stylometry AUC | done: final bot matches ACPL (50 vs 53), blunder rate (1.4 vs 1.5/100), game length and clock use; stylometry AUC 0.76 move / 0.83 game, driven by time-pressure behaviour |
| 5 | Ship: write-up (`docs/results.md`), model card (`docs/model_card.md`), AGPL-3.0 | done; live BOT account and blinded human test are the owner's steps |

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

## Phases 1-4: models and evaluation

```bash
maelstrom-baselines --split test                       # opening book + exact-position lookup
maelstrom-stockfish-baseline --depth 2 5 8 --skill 1 5 10
maelstrom-maia3-features --model maia3-5m              # frozen Maia-3 predictions + features (data/features/)
maelstrom-adapter train && maelstrom-adapter eval --split test
maelstrom-think-time train
maelstrom-selfplay --games 200 && maelstrom-parse --lichess-user CloroxMaelstromBot --no-scope \
    --out-dir data/selfplay data/selfplay/bot_vs_maia.pgn && maelstrom-annotate --data-dir data/selfplay --all-games
maelstrom-humanlike                                    # distributional tests + stylometry AUC
maelstrom-uci                                          # the engine; see deploy/ for lichess-bot
```

Results and discussion: `docs/results.md`. Model card and privacy notes: `docs/model_card.md`.

## Tests

```bash
python -m pytest -q      # synthetic Lichess/Chess.com PGNs with %clk and %eval comments
```

## Licensing note

Maia-2 is MIT; Maia-3 (code and weights) is AGPL-3.0, so a bot built on Maia-3 and served
over the network must publish its source. This repository is open source either way.

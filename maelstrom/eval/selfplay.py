"""Self-play harness: the bot (book -> Maia-3 + adapter, sampled, think-time clock) vs a
Maia-3 opponent at the real opponent's rating, starting from the first N plies of each of
YOUR test games (same colour as you had). Games are written as Lichess-style PGN with %clk
comments so the normal parse -> annotate pipeline can score them like your own games.

    maelstrom-selfplay --games 200 --start-plies 10 --out data/selfplay/bot_vs_maia.pgn
    maelstrom-parse --lichess-user CloroxMaelstromBot --out-dir data/selfplay data/selfplay/bot_vs_maia.pgn
    maelstrom-annotate --data-dir data/selfplay --all-games
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys
import time
import types

import chess
import chess.pgn
import numpy as np
import pandas as pd
import torch

from maelstrom.bot.engine import MaelstromEngine

BOT_NAME = "CloroxMaelstromBot"
OPP_NAME = "Maia3Opponent"


def _fmt_clock(s: float) -> str:
    s = max(0.0, s)
    return f"{int(s // 3600)}:{int(s % 3600 // 60):02d}:{int(s % 60):02d}"


def play_game(eng: MaelstromEngine, start_moves: list[str], bot_color: chess.Color, base: float, inc: float,
              bot_elo: int, opp_elo: int, rng: np.random.Generator, max_plies: int = 300, opp_temperature: float = 1.0) -> chess.pgn.Game:
    eng.board = chess.Board(); eng.moves = []
    for u in start_moves:
        eng.board.push_uci(u); eng.moves.append(u)
    eng.base, eng.inc = base, inc
    clocks = {chess.WHITE: base, chess.BLACK: base}
    game = chess.pgn.Game()
    node = game
    for u in start_moves:
        node = node.add_variation(chess.Move.from_uci(u)); node.set_clock(base)
    rows = []
    while not eng.board.is_game_over(claim_draw=True) and len(eng.moves) < max_plies:
        side = eng.board.turn
        clock_left = clocks[side]
        if side == bot_color:
            eng.opts["SelfElo"], eng.opts["OppoElo"] = bot_elo, opp_elo
            ply = len(eng.moves) + 1
            move = None
            if eng.book is not None and ply <= eng.opts["BookMaxPly"]:
                move = eng.book.sample(eng.board.fen(), eng.rng, eng.opts["BookMinCount"])
            entropy, p1 = 1.0, 0.5
            if move is None:
                moves, probs, entropy = eng.policy(clock_left)
                p1 = float(probs.max()); move = eng.sample(moves, probs)
            think = eng.think_seconds(clock_left, entropy, p1, None)
            eng._last_think = think
        else:
            eng.opts["SelfElo"], eng.opts["OppoElo"] = opp_elo, bot_elo
            saved_adapter, eng.adapter = eng.adapter, None  # opponent = plain Maia-3
            moves, probs, entropy = eng.policy(clock_left)
            eng.adapter = saved_adapter
            T = eng.opts["Temperature"]; eng.opts["Temperature"] = opp_temperature
            move = eng.sample(moves, probs); eng.opts["Temperature"] = T
            think = eng.think_seconds(clock_left, entropy, float(probs.max()), None)  # same clock model as a stand-in human
        clocks[side] = clock_left - think + inc
        if clocks[side] <= 0:
            break  # flag
        eng.board.push_uci(move); eng.moves.append(move)
        node = node.add_variation(chess.Move.from_uci(move)); node.set_clock(clocks[side])
    outcome = eng.board.outcome(claim_draw=True)
    if outcome is not None:
        result = outcome.result()
    elif min(clocks.values()) <= 0:
        result = "0-1" if clocks[chess.WHITE] <= 0 else "1-0"
    else:
        result = "1/2-1/2"
    h = game.headers
    h["Event"] = f"Self-play {'Blitz' if base < 480 else 'Rapid'} game"
    h["Site"] = f"https://lichess.org/selfplay{rng.integers(10**8):08d}"
    h["Variant"] = "Standard"
    now = dt.datetime.now(dt.timezone.utc)
    h["UTCDate"], h["UTCTime"], h["Date"] = now.strftime("%Y.%m.%d"), now.strftime("%H:%M:%S"), now.strftime("%Y.%m.%d")
    h["White"], h["Black"] = (BOT_NAME, OPP_NAME) if bot_color == chess.WHITE else (OPP_NAME, BOT_NAME)
    h["WhiteElo"], h["BlackElo"] = (str(bot_elo), str(opp_elo)) if bot_color == chess.WHITE else (str(opp_elo), str(bot_elo))
    h["TimeControl"] = f"{int(base)}+{int(inc)}"
    h["Result"] = result
    h["Termination"] = "Time forfeit" if min(clocks.values()) <= 0 else "Normal"
    return game


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--start-plies", type=int, default=10)
    p.add_argument("--split", default="test")
    p.add_argument("--model", default="maia3-5m")
    p.add_argument("--adapter", default="models/adapter_maia3-5m.pt")
    p.add_argument("--think-time", default="models/think_time.pkl")
    p.add_argument("--book-splits", default="train,val")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--opp-temperature", type=float, default=1.0)
    p.add_argument("--no-book", action="store_true")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/selfplay/bot_vs_maia.pgn")
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    games = pd.read_parquet(pathlib.Path(a.data_dir) / "games.parquet")
    src = games[games.in_scope & (games.split == a.split)].sample(frac=1.0, random_state=a.seed).head(a.games)
    ea = types.SimpleNamespace(model=a.model, adapter=a.adapter, think_time=a.think_time,
                               book=None if a.no_book else str(pathlib.Path(a.data_dir) / "positions.parquet"),
                               book_splits=a.book_splits, elo=1950, threads=a.threads, seed=a.seed, no_wait=True)
    eng = MaelstromEngine(ea); eng.ensure_loaded()
    eng.opts["Temperature"] = a.temperature
    eng.opts["ThinkTime"] = True
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out, "w") as f:
        for i, g in enumerate(src.itertuples(), 1):
            start = g.moves_uci.split()[: a.start_plies]
            bot_color = chess.WHITE if g.user_color == "white" else chess.BLACK
            game = play_game(eng, start, bot_color, float(g.base_seconds), float(g.increment or 0),
                             int(g.user_elo_lichess or 1950), int(g.opp_elo_lichess or 1950), rng)
            print(game, file=f, end="\n\n"); f.flush()
            if i % 10 == 0:
                print(f"  {i}/{len(src)} games, {(time.time() - t0) / i:.1f}s/game", file=sys.stderr, flush=True)
    print(f"{len(src)} games -> {out}")


if __name__ == "__main__":
    main()

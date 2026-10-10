"""Generate synthetic Lichess- and Chess.com-style PGNs with %clk comments for tests.

Games are random legal play, so they are only useful for exercising the pipeline,
not for modelling. Clock comments follow each platform's real formatting.
"""
from __future__ import annotations

import datetime as dt
import random

import chess
import chess.pgn

USER = "TestUser"


def _fmt_clock(seconds: float, tenths: bool) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:04.1f}" if tenths else f"{h}:{m:02d}:{int(s):02d}"


def random_game(rng: random.Random, platform: str, user_white: bool, base: int, inc: int,
                n_plies: int, start: dt.datetime, idx: int, with_eval: bool = False) -> chess.pgn.Game:
    game = chess.pgn.Game()
    h = game.headers
    white, black = (USER, f"Opp{idx}") if user_white else (f"Opp{idx}", USER)
    h["White"], h["Black"] = white, black
    h["WhiteElo"], h["BlackElo"] = str(1950 + rng.randint(-50, 50)), str(1950 + rng.randint(-100, 100))
    h["TimeControl"] = f"{base}+{inc}" if (inc or platform == "lichess") else str(base)
    h["UTCDate"] = start.strftime("%Y.%m.%d")
    h["UTCTime"] = start.strftime("%H:%M:%S")
    h["Date"] = h["UTCDate"]
    h["ECO"] = "C50"
    if platform == "lichess":
        h["Event"] = "Rated Blitz game" if base < 480 else "Rated Rapid game"
        h["Site"] = f"https://lichess.org/{idx:08d}"
        h["Opening"] = "Italian Game"
        h["Variant"] = "Standard"
    else:
        h["Event"] = "Live Chess"
        h["Site"] = "Chess.com"
        h["Link"] = f"https://www.chess.com/game/live/{100000 + idx}"
        h["ECOUrl"] = "https://www.chess.com/openings/Italian-Game"
    board = game.board()
    node = game
    clocks = {chess.WHITE: float(base), chess.BLACK: float(base)}
    for ply in range(n_plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        mv = rng.choice(moves)
        think = 0.0 if ply < 2 else min(clocks[board.turn] - 1, abs(rng.gauss(3.0, 4.0)))
        clocks[board.turn] = max(0.5, clocks[board.turn] - think + inc)
        node = node.add_variation(mv)
        node.set_clock(clocks[not board.turn] if False else clocks[board.turn])
        if with_eval and platform == "lichess":
            node.set_eval(chess.engine.PovScore(chess.engine.Cp(rng.randint(-300, 300)), chess.WHITE))
        board.push(mv)
    outcome = board.outcome()
    h["Result"] = outcome.result() if outcome else rng.choice(["1-0", "0-1", "1/2-1/2"])
    h["Termination"] = "Normal"
    return game


def write_synthetic_pgns(out_dir, n_lichess: int = 12, n_chesscom: int = 10, seed: int = 0) -> list:
    import pathlib
    out_dir = pathlib.Path(out_dir)
    rng = random.Random(seed)
    t0 = dt.datetime(2024, 1, 1, 12, 0, 0)
    paths = []
    li = out_dir / "lichess" / f"{USER}.pgn"
    li.parent.mkdir(parents=True, exist_ok=True)
    with open(li, "w") as f:
        for i in range(n_lichess):
            base, inc = rng.choice([(180, 0), (180, 2), (300, 0), (600, 0), (60, 0)])
            g = random_game(rng, "lichess", i % 2 == 0, base, inc, rng.randint(20, 90), t0 + dt.timedelta(hours=i), i, with_eval=True)
            print(g, file=f, end="\n\n")
    paths.append(li)
    cc = out_dir / "chesscom" / USER / "2024-02.pgn"
    cc.parent.mkdir(parents=True, exist_ok=True)
    with open(cc, "w") as f:
        for i in range(n_chesscom):
            base, inc = rng.choice([(180, 0), (180, 2), (600, 0), (900, 10), (120, 1)])
            g = random_game(rng, "chesscom", i % 2 == 1, base, inc, rng.randint(20, 90), t0 + dt.timedelta(days=40, hours=i), 1000 + i)
            print(g, file=f, end="\n\n")
    paths.append(cc)
    # one daily game, which must be skipped
    daily = out_dir / "chesscom" / USER / "2024-03.pgn"
    with open(daily, "w") as f:
        g = random_game(rng, "chesscom", True, 600, 0, 10, t0 + dt.timedelta(days=70), 2000)
        g.headers["TimeControl"] = "1/86400"
        print(g, file=f, end="\n\n")
    paths.append(daily)
    return paths

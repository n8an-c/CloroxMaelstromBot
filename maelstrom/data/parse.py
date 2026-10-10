"""Parse Lichess / Chess.com PGN exports into two Parquet tables.

games.parquet      one row per game (metadata, user colour, result, moves)
positions.parquet  one row per ply: FEN before the move, the move, clocks,
                   think time, ratings, phase features; `is_user` marks your moves.

Both sides' moves are kept so generic baselines can be evaluated on identical
positions; filter on `is_user` for personalisation.

    maelstrom-parse --lichess-user NAME --chesscom-user NAME [pgn files...]

Without explicit files, every *.pgn under data/raw/ is parsed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import math
import pathlib
import sys
from typing import Iterable, Iterator

import chess
import chess.pgn
import pandas as pd
from tqdm import tqdm

from maelstrom.config import load_config

PIECE_VALUE = {chess.QUEEN: 9, chess.ROOK: 5, chess.BISHOP: 3, chess.KNIGHT: 3, chess.PAWN: 1}
MATE_CP = 10_000


# --------------------------------------------------------------------------- helpers
def parse_time_control(tc: str | None) -> tuple[int | None, int | None]:
    """'180+2' -> (180, 2); '600' -> (600, 0); '-' / '1/86400' (daily) -> (None, None)."""
    if not tc or tc in ("-", "?"):
        return None, None
    if "/" in tc:  # correspondence / daily
        return None, None
    if "+" in tc:
        base, inc = tc.split("+", 1)
        try:
            return int(base), int(inc)
        except ValueError:
            return None, None
    try:
        return int(tc), 0
    except ValueError:
        return None, None


def speed_class(base: int | None, inc: int | None, platform: str) -> str | None:
    """Lichess and Chess.com both bucket by estimated duration base + 40*inc."""
    if base is None:
        return None
    est = base + 40 * (inc or 0)
    if platform == "lichess":
        if est < 29:
            return "ultrabullet"
        if est < 179:
            return "bullet"
        if est < 479:
            return "blitz"
        if est < 1499:
            return "rapid"
        return "classical"
    # chess.com: <3 min bullet, 3-<10 min blitz, >=10 min rapid (no classical class)
    if est < 180:
        return "bullet"
    if est < 600:
        return "blitz"
    return "rapid"


def rated_from_event(event: str) -> bool | None:
    """Lichess exports use lower-case 'rated blitz game' / 'casual rapid game'; arenas and
    team battles carry only the tournament name, so their rating status is unknown (None)."""
    e = event.lower()
    if e.startswith("rated"):
        return True
    if e.startswith("casual"):
        return False
    return None


def detect_platform(headers: chess.pgn.Headers) -> str:
    site = headers.get("Site", "")
    link = headers.get("Link", "")
    if "lichess.org" in site:
        return "lichess"
    if "chess.com" in site.lower() or "chess.com" in link.lower():
        return "chesscom"
    return "unknown"


def game_identifier(headers: chess.pgn.Headers, platform: str) -> str:
    if platform == "lichess":
        return "lichess:" + headers.get("Site", "").rstrip("/").split("/")[-1]
    if platform == "chesscom":
        link = headers.get("Link", "")
        if link:
            return "chesscom:" + link.rstrip("/").split("/")[-1]
    # fallback: hash of key headers
    key = "|".join(headers.get(k, "") for k in ("UTCDate", "UTCTime", "White", "Black", "Result"))
    return f"{platform}:{abs(hash(key)) % 10**12}"


def game_datetime(headers: chess.pgn.Headers) -> dt.datetime | None:
    date = headers.get("UTCDate") or headers.get("Date")
    time_ = headers.get("UTCTime") or headers.get("StartTime") or "00:00:00"
    if not date or "?" in date:
        return None
    try:
        return dt.datetime.strptime(f"{date} {time_}", "%Y.%m.%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        try:
            return dt.datetime.strptime(date, "%Y.%m.%d").replace(tzinfo=dt.timezone.utc)
        except ValueError:
            return None


def _int_or_none(s: str | None) -> int | None:
    try:
        return int(s) if s not in (None, "", "?") else None
    except ValueError:
        return None


def score_cp(score: chess.engine.PovScore | None, pov: chess.Color) -> float | None:
    """PovScore -> centipawns from `pov` perspective, mates clipped to ±MATE_CP."""
    if score is None:
        return None
    s = score.pov(pov)
    if s.is_mate():
        m = s.mate()
        return float(MATE_CP if m > 0 else -MATE_CP)
    return float(s.score())


def lichess_equiv(elo: int | None, speed: str | None, platform: str, rating_map: dict) -> int | None:
    if elo is None:
        return None
    if platform != "chesscom" or speed not in rating_map:
        return elo
    m = rating_map[speed]
    return int(round(m["a"] * elo + m["b"]))


def nonpawn_material(board: chess.Board) -> tuple[int, int]:
    """(total non-pawn material in pawns, number of minor+major pieces)."""
    mat, n = 0, 0
    for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
        c = len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK))
        mat += c * PIECE_VALUE[pt]
        n += c
    return mat, n


def phase_of(ply: int, n_pieces: int) -> str:
    if ply <= 20:
        return "opening"
    if n_pieces <= 6:  # Lichess divider criterion for endgame
        return "endgame"
    return "middlegame"


# --------------------------------------------------------------------------- core
def iter_games(paths: Iterable[pathlib.Path]) -> Iterator[tuple[pathlib.Path, chess.pgn.Game]]:
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            while True:
                try:
                    game = chess.pgn.read_game(f)
                except Exception as e:  # malformed game; python-chess normally recovers
                    print(f"  skipping unreadable game in {path}: {e}", file=sys.stderr)
                    continue
                if game is None:
                    break
                yield path, game


def parse_game(game: chess.pgn.Game, users: dict[str, str], rating_map: dict) -> tuple[dict, list[dict]] | None:
    h = game.headers
    platform = detect_platform(h)
    variant = h.get("Variant", "Standard")
    if variant not in ("Standard", "From Position") or h.get("SetUp") == "1" or h.get("FEN"):
        return None
    base, inc = parse_time_control(h.get("TimeControl"))
    if base is None:
        return None  # daily / correspondence / unknown
    speed = speed_class(base, inc, platform)

    user = users.get(platform)
    white, black = h.get("White", ""), h.get("Black", "")
    if user is None:
        user_color = None
    elif white.lower() == user.lower():
        user_color = chess.WHITE
    elif black.lower() == user.lower():
        user_color = chess.BLACK
    else:
        user_color = None

    w_elo, b_elo = _int_or_none(h.get("WhiteElo")), _int_or_none(h.get("BlackElo"))
    result = h.get("Result", "*")
    res_white = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(result)
    event = h.get("Event", "")
    rated = rated_from_event(event) if platform == "lichess" else None
    gid = game_identifier(h, platform)
    when = game_datetime(h)

    board = game.board()
    clocks = {chess.WHITE: float(base), chess.BLACK: float(base)}
    rows: list[dict] = []
    moves_uci: list[str] = []
    prev_eval = None  # cp from white POV for position before current move
    node = game
    ply = 0
    has_clocks = False
    has_evals = False
    while node.variations:
        nxt = node.variation(0)
        move = nxt.move
        ply += 1
        mover = board.turn
        fen = board.fen()
        legal_n = board.legal_moves.count()
        is_capture = board.is_capture(move)
        piece = board.piece_type_at(move.from_square)
        castle = board.is_castling(move)
        san = board.san(move)
        mat, n_pieces = nonpawn_material(board)
        clock_after = nxt.clock()
        if clock_after is not None:
            has_clocks = True
        clock_before = clocks[mover]
        think = None
        if clock_after is not None:
            think = max(0.0, clock_before - clock_after + (inc or 0))
            clocks[mover] = clock_after
        ev = nxt.eval()
        eval_after = score_cp(ev, chess.WHITE) if ev is not None else None
        if eval_after is not None:
            has_evals = True
        board.push(move)
        rows.append({
            "game_id": gid,
            "ply": ply,
            "fullmove": (ply + 1) // 2,
            "mover_color": "white" if mover == chess.WHITE else "black",
            "is_user": (user_color is not None and mover == user_color),
            "fen": fen,
            "move_uci": move.uci(),
            "move_san": san,
            "mover_elo": w_elo if mover == chess.WHITE else b_elo,
            "opp_elo": b_elo if mover == chess.WHITE else w_elo,
            "clock_before": clock_before if clock_after is not None else None,
            "clock_after": clock_after,
            "think_time": think,
            "eval_before_cp": prev_eval,
            "eval_after_cp": eval_after,
            "legal_moves": legal_n,
            "is_capture": is_capture,
            "is_check": board.is_check(),
            "is_castle": castle,
            "is_promotion": move.promotion is not None,
            "piece": chess.piece_symbol(piece) if piece else None,
            "nonpawn_material": mat,
            "n_pieces": n_pieces,
            "phase": phase_of(ply, n_pieces),
        })
        moves_uci.append(move.uci())
        prev_eval = eval_after
        node = nxt

    if ply == 0:
        return None

    u_is_w = user_color == chess.WHITE
    user_elo = (w_elo if u_is_w else b_elo) if user_color is not None else None
    opp_elo = (b_elo if u_is_w else w_elo) if user_color is not None else None
    user_result = None
    if res_white is not None and user_color is not None:
        user_result = res_white if u_is_w else 1.0 - res_white
    grow = {
        "game_id": gid,
        "platform": platform,
        "datetime_utc": when,
        "event": event,
        "rated": rated,
        "speed": speed,
        "base_seconds": base,
        "increment": inc,
        "time_control": h.get("TimeControl"),
        "white": white,
        "black": black,
        "white_elo": w_elo,
        "black_elo": b_elo,
        "user_color": ("white" if u_is_w else "black") if user_color is not None else None,
        "user_elo": user_elo,
        "opp_elo": opp_elo,
        "user_elo_lichess": lichess_equiv(user_elo, speed, platform, rating_map),
        "opp_elo_lichess": lichess_equiv(opp_elo, speed, platform, rating_map),
        "result": result,
        "user_score": user_result,
        "termination": h.get("Termination"),
        "eco": h.get("ECO"),
        "opening": h.get("Opening") or (h.get("ECOUrl", "").rstrip("/").split("/")[-1] or None),
        "n_plies": ply,
        "has_clocks": has_clocks,
        "has_evals": has_evals,
        "moves_uci": " ".join(moves_uci),
    }
    for r in rows:
        r["platform"] = platform
        r["speed"] = speed
        r["increment"] = inc
        r["base_seconds"] = base
        r["mover_elo_lichess"] = lichess_equiv(r["mover_elo"], speed, platform, rating_map)
        r["opp_elo_lichess"] = lichess_equiv(r["opp_elo"], speed, platform, rating_map)
    return grow, rows


def in_scope_mask(games: pd.DataFrame, scope_speeds: set[str] | None, min_date: str | None) -> pd.Series:
    """Modelling scope: speed in scope_speeds and datetime >= min_date (both optional)."""
    m = pd.Series(True, index=games.index)
    if scope_speeds:
        m &= games["speed"].isin(scope_speeds)
    if min_date:
        cutoff = pd.Timestamp(min_date, tz="UTC")
        m &= games["datetime_utc"].notna() & (games["datetime_utc"] >= cutoff)
    return m


def parse_files(paths: list[pathlib.Path], users: dict[str, str], rating_map: dict,
                scope_speeds: set[str] | None = None, min_date: str | None = None,
                show_progress: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    games, positions = [], []
    seen: set[str] = set()
    n_skipped = 0
    it = iter_games(paths)
    if show_progress:
        it = tqdm(it, unit="game", desc="parsing")
    for _, game in it:
        out = parse_game(game, users, rating_map)
        if out is None:
            n_skipped += 1
            continue
        g, rows = out
        if g["game_id"] in seen:
            continue
        seen.add(g["game_id"])
        games.append(g)
        positions.extend(rows)
    if n_skipped:
        print(f"skipped {n_skipped} games (variants, from-position, daily/correspondence)", file=sys.stderr)
    gdf = pd.DataFrame(games)
    pdf = pd.DataFrame(positions)
    if len(gdf):
        gdf["datetime_utc"] = pd.to_datetime(gdf["datetime_utc"], utc=True)
        gdf = gdf.sort_values(["datetime_utc", "game_id"]).reset_index(drop=True)
        gdf["in_scope"] = in_scope_mask(gdf, scope_speeds, min_date).values
        pdf = pdf.merge(gdf[["game_id", "in_scope"]], on="game_id", how="left")
    return gdf, pdf


def default_pgn_paths(root: pathlib.Path = pathlib.Path("data/raw")) -> list[pathlib.Path]:
    return sorted(pathlib.Path(p) for p in glob.glob(str(root / "**" / "*.pgn"), recursive=True))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pgns", nargs="*", help="PGN files (default: data/raw/**/*.pgn)")
    p.add_argument("--lichess-user", default=None)
    p.add_argument("--chesscom-user", default=None)
    p.add_argument("--no-scope", action="store_true", help="mark every game in_scope (ignore perf_types/min_date)")
    p.add_argument("--out-dir", default="data/processed")
    a = p.parse_args(argv)
    cfg = load_config()
    users = {
        "lichess": a.lichess_user or cfg["players"].get("lichess"),
        "chesscom": a.chesscom_user or cfg["players"].get("chesscom"),
    }
    users = {k: v for k, v in users.items() if v}
    if not users:
        sys.exit("No usernames: pass --lichess-user/--chesscom-user or set configs/local.yaml")
    paths = [pathlib.Path(x) for x in a.pgns] or default_pgn_paths()
    if not paths:
        sys.exit("No PGN files found under data/raw/")
    scope = None if a.no_scope else set(cfg.get("perf_types", []))
    min_date = None if a.no_scope else cfg.get("min_date")
    gdf, pdf = parse_files(paths, users, cfg.get("rating_map", {}), scope, min_date)
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out / "games.parquet", index=False)
    pdf.to_parquet(out / "positions.parquet", index=False)
    n_user = int(pdf["is_user"].sum()) if len(pdf) else 0
    print(f"{len(gdf)} games, {len(pdf)} positions ({n_user} your moves) -> {out}")
    if len(gdf):
        print(gdf.groupby(["platform", "speed", "in_scope"]).size().unstack("in_scope", fill_value=0).to_string())
        print(f"in scope: {int(gdf.in_scope.sum())} games, {int((pdf.is_user & pdf.in_scope).sum())} your moves "
              f"(speeds={sorted(scope) if scope else 'all'}, min_date={min_date})")
        no_user = int(gdf["user_color"].isna().sum())
        if no_user:
            print(f"WARNING: {no_user} games where neither player matched a configured username", file=sys.stderr)


if __name__ == "__main__":
    main()

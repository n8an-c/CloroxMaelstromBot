"""Annotate positions with Stockfish: best-move eval, played-move eval, win-prob loss.

For every (game_id, ply) selected (default: your moves only) we compute, from the
mover's perspective:

    best_cp        eval of the best move at fixed depth
    played_cp      eval of the move actually played (searchmoves = [move])
    cp_loss        best_cp - played_cp  (>= 0 up to search noise)
    best_wp / played_wp / wp_loss   Lichess win-probability (0-100) and its drop
    quality        'best' | 'good' | 'inaccuracy' | 'mistake' | 'blunder'
    best_move_uci  engine's choice (for "did I find the best move" slices)

Resumable: existing rows in annotations.parquet are skipped. Work is sharded
across `workers` processes, each owning one Stockfish instance.

    maelstrom-annotate [--depth 13] [--workers 4] [--all-moves] [--limit N]
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import chess
import chess.engine
import numpy as np
import pandas as pd

from maelstrom.config import load_config

MATE_CP = 10_000


def win_prob(cp: float) -> float:
    """Lichess accuracy model: win% from centipawns, mover's perspective."""
    cp = max(-1000.0, min(1000.0, cp))
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


def quality_bucket(wp_loss: float) -> str:
    # Lichess thresholds on win-probability drop (percentage points)
    if wp_loss < 2:
        return "best"
    if wp_loss < 10:
        return "good"
    if wp_loss < 20:
        return "inaccuracy"
    if wp_loss < 30:
        return "mistake"
    return "blunder"


def _cp(score: chess.engine.PovScore, pov: chess.Color) -> float:
    s = score.pov(pov)
    if s.is_mate():
        return float(MATE_CP if s.mate() > 0 else -MATE_CP)
    return float(s.score())


def analyse_chunk(rows: list[tuple[str, int, str, str]], sf_path: str, depth: int,
                  threads: int, hash_mb: int) -> list[dict]:
    out = []
    eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    try:
        eng.configure({"Threads": threads, "Hash": hash_mb})
        limit = chess.engine.Limit(depth=depth)
        for gid, ply, fen, uci in rows:
            board = chess.Board(fen)
            move = chess.Move.from_uci(uci)
            pov = board.turn
            best = eng.analyse(board, limit)
            best_cp = _cp(best["score"], pov)
            best_move = best.get("pv", [None])[0]
            if best_move == move:
                played_cp = best_cp
            else:
                played = eng.analyse(board, limit, root_moves=[move])
                played_cp = _cp(played["score"], pov)
            bwp, pwp = win_prob(best_cp), win_prob(played_cp)
            wp_loss = max(0.0, bwp - pwp)
            out.append({
                "game_id": gid, "ply": ply,
                "best_cp": best_cp, "played_cp": played_cp,
                "cp_loss": max(0.0, best_cp - played_cp),
                "best_wp": bwp, "played_wp": pwp, "wp_loss": wp_loss,
                "quality": quality_bucket(wp_loss),
                "best_move_uci": best_move.uci() if best_move else None,
                "is_best": best_move == move,
                "depth": depth,
            })
    finally:
        eng.quit()
    return out


def annotate(pos: pd.DataFrame, sf_path: str, depth: int, workers: int, threads: int, hash_mb: int,
             existing: pd.DataFrame | None, out_path: pathlib.Path, chunk_size: int = 200) -> pd.DataFrame:
    todo = pos[["game_id", "ply", "fen", "move_uci"]]
    if existing is not None and len(existing):
        done = set(zip(existing.game_id, existing.ply))
        todo = todo[[(g, p) not in done for g, p in zip(todo.game_id, todo.ply)]]
    rows = list(todo.itertuples(index=False, name=None))
    print(f"{len(rows)} positions to annotate at depth {depth} with {workers} workers", file=sys.stderr)
    chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
    results: list[dict] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(analyse_chunk, c, sf_path, depth, threads, hash_mb) for c in chunks]
        for i, f in enumerate(as_completed(futs), 1):
            results.extend(f.result())
            if i % 5 == 0 or i == len(futs):
                rate = len(results) / max(1e-9, time.time() - t0)
                print(f"  {len(results)}/{len(rows)}  {rate:.1f} pos/s", file=sys.stderr)
                _save(existing, results, out_path)  # checkpoint
    return _save(existing, results, out_path)


def _save(existing: pd.DataFrame | None, results: list[dict], out_path: pathlib.Path) -> pd.DataFrame:
    new = pd.DataFrame(results)
    df = pd.concat([existing, new], ignore_index=True) if existing is not None and len(existing) else new
    if len(df):
        df = df.drop_duplicates(["game_id", "ply"]).sort_values(["game_id", "ply"]).reset_index(drop=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
    return df


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--stockfish", default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--all-moves", action="store_true", help="annotate opponents' moves too")
    p.add_argument("--all-games", action="store_true", help="include out-of-scope games too")
    p.add_argument("--limit", type=int, default=None, help="only the first N positions (smoke test)")
    a = p.parse_args(argv)
    cfg = load_config()["stockfish"]
    sf = a.stockfish or cfg["path"]
    depth = a.depth or cfg["depth"]
    workers = a.workers or cfg["workers"]
    d = pathlib.Path(a.data_dir)
    pos = pd.read_parquet(d / "positions.parquet")
    if not a.all_moves and cfg.get("user_moves_only", True):
        pos = pos[pos.is_user]
    if "in_scope" in pos and not a.all_games:
        pos = pos[pos.in_scope]
    if a.limit:
        pos = pos.head(a.limit)
    out_path = d / "annotations.parquet"
    existing = pd.read_parquet(out_path) if out_path.exists() else None
    df = annotate(pos, sf, depth, workers, cfg.get("threads", 1), cfg.get("hash_mb", 64), existing, out_path)
    print(f"{len(df)} annotated positions -> {out_path}")
    if len(df):
        print(df.quality.value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()

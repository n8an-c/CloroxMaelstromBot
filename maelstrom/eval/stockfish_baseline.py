"""Stockfish as a human-move predictor (the weak baseline the literature reports at 33-41%).

Two families, both producing the shared predictions contract:

* depth-N : full-strength search at a fixed depth; distribution = softmax over the
            MultiPV centipawn scores with a temperature (fit on val, default 100 cp)
* skill-S : `Skill Level` S (0-20); same MultiPV softmax

    maelstrom-stockfish-baseline --split test --depth 6 --skill 5 [--workers 4]
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import chess
import chess.engine
import numpy as np
import pandas as pd

from maelstrom.config import load_config
from maelstrom.eval.metrics import evaluate, format_table

MATE_CP = 10_000


def _cp(score, pov):
    s = score.pov(pov)
    return float(MATE_CP if s.mate() > 0 else -MATE_CP) if s.is_mate() else float(s.score())


def _chunk(rows, sf_path, depth, skill, multipv, temp_cp):
    eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    opts = {"Threads": 1, "Hash": 32}
    if skill is not None:
        opts["Skill Level"] = skill
    eng.configure(opts)
    limit = chess.engine.Limit(depth=depth)
    out = []
    try:
        for gid, ply, fen, played in rows:
            board = chess.Board(fen)
            n_legal = board.legal_moves.count()
            infos = eng.analyse(board, limit, multipv=min(multipv, n_legal))
            if isinstance(infos, dict):
                infos = [infos]
            cps = np.array([_cp(i["score"], board.turn) for i in infos])
            moves = [i["pv"][0].uci() for i in infos]
            w = np.exp((cps - cps.max()) / temp_cp)
            # mass for unlisted legal moves: share of the smallest listed weight, spread evenly
            rest = n_legal - len(moves)
            tail = w.min() * 0.5 * rest
            probs = w / (w.sum() + tail)
            p_other = (w.min() * 0.5) / (w.sum() + tail) if rest else 0.0
            pm = dict(zip(moves, probs))
            pred = moves[0]
            if skill is not None:
                # Skill Level only randomises the move Stockfish *plays*, not MultiPV scores
                pred = eng.play(board, limit).move.uci()
            out.append({"game_id": gid, "ply": ply, "pred_move": pred,
                        "p_played": pm.get(played, p_other), "top3": [pred] + [m for m in moves if m != pred][:2]})
    finally:
        eng.quit()
    return out


def predict(pos: pd.DataFrame, sf_path: str, depth: int, skill: int | None, multipv: int = 5, temp_cp: float = 100.0,
            workers: int = 4, chunk: int = 150) -> pd.DataFrame:
    rows = list(pos[["game_id", "ply", "fen", "move_uci"]].itertuples(index=False, name=None))
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_chunk, chunks, [sf_path] * len(chunks), [depth] * len(chunks),
                                     [skill] * len(chunks), [multipv] * len(chunks), [temp_cp] * len(chunks)), 1):
            out += r
            if i % 10 == 0:
                print(f"  {len(out)}/{len(rows)}", file=sys.stderr, flush=True)
    return pd.DataFrame(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--preds-dir", default="data/preds")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--depth", type=int, nargs="*", default=[6], help="fixed-depth variants")
    p.add_argument("--skill", type=int, nargs="*", default=[], help="Skill Level variants (searched at --skill-depth)")
    p.add_argument("--skill-depth", type=int, default=8)
    p.add_argument("--temp-cp", type=float, default=100.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    sf = load_config()["stockfish"]["path"]
    d = pathlib.Path(a.data_dir)
    pos = pd.read_parquet(d / "positions.parquet")
    ann = d / "annotations.parquet"
    if ann.exists():
        pos = pos.merge(pd.read_parquet(ann)[["game_id", "ply", "quality"]], on=["game_id", "ply"], how="left")
    u = pos[pos.is_user & pos.in_scope.fillna(False) & (pos.split == a.split)]
    if a.limit:
        u = u.head(a.limit)
    preds_dir = pathlib.Path(a.preds_dir)
    preds_dir.mkdir(parents=True, exist_ok=True)
    results = []
    variants = [(f"stockfish depth {dp}", dp, None) for dp in a.depth] + [(f"stockfish skill {s}", a.skill_depth, s) for s in a.skill]
    for name, depth, skill in variants:
        print(name, file=sys.stderr)
        preds = predict(u, sf, depth, skill, temp_cp=a.temp_cp, workers=a.workers)
        preds.to_parquet(preds_dir / f"{a.split}_{name.replace(' ', '_')}.parquet", index=False)
        results.append(evaluate(u, preds, name))
    res = pd.concat(results, ignore_index=True)
    res.to_parquet(preds_dir / f"{a.split}_results_stockfish.parquet", index=False)
    lines = [f"# Stockfish baselines on the {a.split} split ({len(u)} of your moves)\n", format_table(res, "all")]
    for sl in ("phase", "clock_bucket", "quality"):
        if (res["slice"] == sl).any():
            lines += [f"\n## by {sl}\n", format_table(res, sl)]
    out = pathlib.Path(a.out or f"reports/baselines_stockfish_{a.split}.md")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

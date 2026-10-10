"""Cheap, data-only baselines: personal opening book / exact-position lookup (kNN with k=exact).

Both are the same mechanism, a table of (FEN -> your move counts) built from the
training split, differing only in where they stop:

* book:  used up to `max_ply`, then no prediction (coverage drops to 0)
* knn:   used at every ply where the exact position was seen in training

Probabilities are Laplace-smoothed move frequencies over legal moves, so NLL is
defined wherever the position is covered. The FEN key drops the halfmove and
fullmove counters so transpositions at different move numbers merge.

    maelstrom-baselines   -> reports/baselines_book_knn.md + data/preds/*.parquet
"""
from __future__ import annotations

import argparse
import pathlib
from collections import Counter, defaultdict

import chess
import numpy as np
import pandas as pd

from maelstrom.eval.metrics import evaluate, format_table


def fen_key(fen: str) -> str:
    return " ".join(fen.split(" ")[:4])


class PositionTable:
    def __init__(self, train_pos: pd.DataFrame, alpha: float = 0.5):
        self.counts: dict[str, Counter] = defaultdict(Counter)
        for f, m in zip(train_pos["fen"], train_pos["move_uci"]):
            self.counts[fen_key(f)][m] += 1
        self.alpha = alpha

    def predict(self, fen: str) -> tuple[str, dict[str, float]] | None:
        c = self.counts.get(fen_key(fen))
        if not c:
            return None
        legal = [m.uci() for m in chess.Board(fen).legal_moves]
        total = sum(c.values()) + self.alpha * len(legal)
        probs = {m: (c.get(m, 0) + self.alpha) / total for m in legal}
        best = max(c.items(), key=lambda kv: kv[1])[0]
        return best, probs

    def sample(self, fen: str, rng: np.random.Generator, min_count: int = 1) -> str | None:
        c = self.counts.get(fen_key(fen))
        if not c or sum(c.values()) < min_count:
            return None
        moves, w = zip(*c.items())
        return rng.choice(moves, p=np.array(w) / sum(w))


def predict_table(table: PositionTable, pos: pd.DataFrame, max_ply: int | None) -> pd.DataFrame:
    rows = []
    for gid, ply, fen, played in zip(pos["game_id"], pos["ply"], pos["fen"], pos["move_uci"]):
        if max_ply is not None and ply > max_ply:
            continue
        out = table.predict(fen)
        if out is None:
            continue
        best, probs = out
        top3 = sorted(probs, key=probs.get, reverse=True)[:3]
        rows.append({"game_id": gid, "ply": ply, "pred_move": best, "p_played": probs.get(played, np.nan), "top3": top3})
    return pd.DataFrame(rows, columns=["game_id", "ply", "pred_move", "p_played", "top3"])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--preds-dir", default="data/preds")
    p.add_argument("--out", default="reports/baselines_book_knn.md")
    p.add_argument("--book-max-ply", type=int, default=20)
    p.add_argument("--split", default="test", choices=["val", "test"])
    a = p.parse_args(argv)
    d = pathlib.Path(a.data_dir)
    pos = pd.read_parquet(d / "positions.parquet")
    ann_path = d / "annotations.parquet"
    if ann_path.exists():
        pos = pos.merge(pd.read_parquet(ann_path)[["game_id", "ply", "quality", "cp_loss"]], on=["game_id", "ply"], how="left")
    u = pos[pos.is_user & pos.in_scope.fillna(False)]
    train, evalset = u[u.split == "train"], u[u.split == a.split]
    table = PositionTable(train)
    preds_dir = pathlib.Path(a.preds_dir)
    preds_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, max_ply in (("book(ply<=%d)" % a.book_max_ply, a.book_max_ply), ("knn(exact FEN)", None)):
        preds = predict_table(table, evalset, max_ply)
        preds.to_parquet(preds_dir / f"{a.split}_{name.split('(')[0]}.parquet", index=False)
        results.append(evaluate(evalset, preds, name))
    res = pd.concat(results, ignore_index=True)
    res.to_parquet(preds_dir / f"{a.split}_results_book_knn.parquet", index=False)
    lines = [f"# Book / kNN baselines on the {a.split} split ({len(evalset)} of your moves, {evalset.game_id.nunique()} games)\n",
             "Coverage = share of positions where the model makes any prediction; uncovered positions count as misses.\n",
             format_table(res, "all")]
    for sl in ("phase", "clock_bucket", "seen", "quality"):
        if (res["slice"] == sl).any():
            lines += [f"\n## by {sl} (ply > 10 except phase)\n", format_table(res, sl)]
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

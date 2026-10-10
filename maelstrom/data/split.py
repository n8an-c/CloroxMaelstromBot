"""Assign train / val / test splits BY GAME, never by position.

* test  = the most recent `test_frac` of games (temporal hold-out, "future" set)
* val   = a seeded random `val_frac` of the remaining games
* train = everything else

Writes the `split` column into games.parquet and positions.parquet in place,
and prints how many test positions also occur (same FEN + your move) in train,
so results can be reported on unseen positions too.

    maelstrom-split [--test-frac 0.1] [--val-frac 0.1] [--seed N]
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

from maelstrom.config import load_config


def assign_splits(games: pd.DataFrame, test_frac: float, val_frac: float, seed: int) -> pd.Series:
    g = games.sort_values(["datetime_utc", "game_id"]).reset_index()
    n = len(g)
    n_test = int(round(n * test_frac))
    split = np.array(["train"] * n, dtype=object)
    if n_test:
        split[n - n_test:] = "test"
    rest_idx = np.arange(n - n_test)
    rng = np.random.default_rng(seed)
    n_val = int(round(len(rest_idx) * val_frac))
    val_idx = rng.choice(rest_idx, size=n_val, replace=False) if n_val else np.array([], dtype=int)
    split[val_idx] = "val"
    out = pd.Series(split, index=g["index"]).sort_index()
    out.name = "split"
    return out


def mark_seen_positions(pos: pd.DataFrame) -> pd.Series:
    """True if (fen, move) of a user move in val/test also appears as a user move in train."""
    train_pairs = set(zip(pos.loc[(pos.split == "train") & pos.is_user, "fen"],
                          pos.loc[(pos.split == "train") & pos.is_user, "move_uci"]))
    train_fens = set(pos.loc[(pos.split == "train") & pos.is_user, "fen"])
    seen_pair = [(f, m) in train_pairs for f, m in zip(pos.fen, pos.move_uci)]
    seen_fen = pos.fen.isin(train_fens)
    return pd.DataFrame({"seen_fen_in_train": seen_fen.values, "seen_pair_in_train": seen_pair}, index=pos.index)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--test-frac", type=float, default=None)
    p.add_argument("--val-frac", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args(argv)
    cfg = load_config()["split"]
    test_frac = a.test_frac if a.test_frac is not None else cfg["test_frac"]
    val_frac = a.val_frac if a.val_frac is not None else cfg["val_frac"]
    seed = a.seed if a.seed is not None else cfg["seed"]

    d = pathlib.Path(a.data_dir)
    games = pd.read_parquet(d / "games.parquet")
    pos = pd.read_parquet(d / "positions.parquet")
    games["split"] = assign_splits(games, test_frac, val_frac, seed).values
    pos = pos.drop(columns=[c for c in ("split", "seen_fen_in_train", "seen_pair_in_train") if c in pos])
    pos = pos.merge(games[["game_id", "split"]], on="game_id", how="left")
    seen = mark_seen_positions(pos)
    pos = pd.concat([pos, seen], axis=1)
    games.to_parquet(d / "games.parquet", index=False)
    pos.to_parquet(d / "positions.parquet", index=False)

    print(games.groupby("split").agg(games=("game_id", "size"),
                                     first=("datetime_utc", "min"), last=("datetime_utc", "max")).to_string())
    u = pos[pos.is_user]
    summ = u.groupby("split").agg(user_moves=("ply", "size"),
                                  seen_fen=("seen_fen_in_train", "mean"),
                                  seen_pair=("seen_pair_in_train", "mean"),
                                  after_ply10=("ply", lambda s: (s > 10).mean()))
    print(summ.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()

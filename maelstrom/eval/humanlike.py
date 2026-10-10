"""Human-likeness evaluation: does the bot's play *distribution* match yours?

Inputs: your test-split positions + Stockfish annotations, and the same tables for the
bot's self-play games (data/selfplay/, produced by maelstrom-selfplay -> parse -> annotate).

1. Distributional similarity, by phase and by clock bucket: ACPL, best-move %, inaccuracy /
   mistake / blunder per 100 moves, median think time; KS and Wasserstein distances on
   cp-loss and think-time distributions.
2. Stylometry: a gradient-boosted classifier on move-level features tries to tell your
   moves from the bot's (grouped CV by game). AUC near 0.5 = indistinguishable. Also a
   game-level AUC on per-game aggregates, which is the harder test.

    maelstrom-humanlike [--bot-dir data/selfplay] [--out reports/humanlike.md]
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

CLOCK_BINS = [-1, 10, 30, 120, 1e9]
CLOCK_LABELS = ["<10s", "10-30s", "30s-2m", ">2m"]
MOVE_FEATS = ["cp_loss", "wp_loss", "is_best", "think_time", "ply", "clock_frac", "legal_moves", "nonpawn_material",
              "is_capture", "is_check", "is_castle", "piece_idx", "log_think", "clock_before"]


def load(data_dir: pathlib.Path, user_only_split: str | None) -> pd.DataFrame:
    pos = pd.read_parquet(data_dir / "positions.parquet")
    ann = pd.read_parquet(data_dir / "annotations.parquet")
    df = pos.merge(ann, on=["game_id", "ply"], how="inner")
    df = df[df.is_user]
    if user_only_split and "split" in df:
        df = df[df.split == user_only_split]
    df = df[df.ply > 10]  # skip the shared opening / book plies
    df["clock_bucket"] = pd.cut(df.clock_before.fillna(600), CLOCK_BINS, labels=CLOCK_LABELS)
    df["clock_frac"] = df.clock_before / df.base_seconds.clip(lower=1)
    df["piece_idx"] = df.piece.map({p: i for i, p in enumerate("pnbrqk")}).fillna(0)
    df["log_think"] = np.log1p(df.think_time.clip(lower=0))
    df["cp_loss"] = df.cp_loss.clip(upper=1000)
    return df


def stat_table(you: pd.DataFrame, bot: pd.DataFrame, by: str) -> list[str]:
    lines = [f"| {by} | who | moves | ACPL | best % | inacc/100 | mistake/100 | blunder/100 | median think s | KS cp-loss | KS think | W1 cp-loss |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    cats = [c for c in (you[by].unique() if by != "clock_bucket" else CLOCK_LABELS) if c in set(you[by]) and c in set(bot[by])]
    for c in cats:
        a, b = you[you[by] == c], bot[bot[by] == c]
        if len(a) < 30 or len(b) < 30:
            continue
        ks_cp = stats.ks_2samp(a.cp_loss, b.cp_loss).statistic
        ks_t = stats.ks_2samp(a.think_time.dropna(), b.think_time.dropna()).statistic if a.think_time.notna().any() and b.think_time.notna().any() else np.nan
        w1 = stats.wasserstein_distance(a.cp_loss, b.cp_loss)
        for who, d in (("you", a), ("bot", b)):
            q = d.quality.value_counts(normalize=True)
            lines.append(f"| {c} | {who} | {len(d)} | {d.cp_loss.mean():.0f} | {d.is_best.mean():.1%} | {100 * q.get('inaccuracy', 0):.1f} | "
                         f"{100 * q.get('mistake', 0):.1f} | {100 * q.get('blunder', 0):.1f} | {d.think_time.median():.1f} | "
                         f"{ks_cp:.3f} | {ks_t:.3f} | {w1:.1f} |")
    return lines


def stylometry(you: pd.DataFrame, bot: pd.DataFrame, seed: int = 0) -> dict:
    df = pd.concat([you.assign(label=0), bot.assign(label=1)], ignore_index=True)
    X = df[MOVE_FEATS].to_numpy(dtype=float); y = df.label.to_numpy(); g = df.game_id.to_numpy()
    oof = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed).fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    move_auc = roc_auc_score(y, oof)
    # game level: aggregate features per game, then classify games
    agg = df.groupby("game_id").agg(label=("label", "first"), acpl=("cp_loss", "mean"), best=("is_best", "mean"),
                                    blunder=("quality", lambda s: (s == "blunder").mean()), think_med=("think_time", "median"),
                                    think_p90=("think_time", lambda s: s.quantile(0.9)), think_lt1=("think_time", lambda s: (s < 1).mean()),
                                    captures=("is_capture", "mean"), checks=("is_check", "mean"), n=("ply", "size"), last_ply=("ply", "max"))
    Xg = agg.drop(columns="label").to_numpy(dtype=float); yg = agg.label.to_numpy()
    oofg = np.zeros(len(agg))
    for tr, te in GroupKFold(n_splits=5).split(Xg, yg, agg.index.to_numpy()):
        clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, random_state=seed).fit(Xg[tr], yg[tr])
        oofg[te] = clf.predict_proba(Xg[te])[:, 1]
    game_auc = roc_auc_score(yg, oofg)
    # a per-game vote of the move-level classifier
    vote = pd.DataFrame({"g": g, "p": oof, "y": y}).groupby("g").agg(p=("p", "mean"), y=("y", "first"))
    return {"move_auc": float(move_auc), "game_auc": float(game_auc), "move_vote_game_auc": float(roc_auc_score(vote.y, vote.p)),
            "n_you_moves": int((y == 0).sum()), "n_bot_moves": int((y == 1).sum()), "n_you_games": int((agg.label == 0).sum()),
            "n_bot_games": int((agg.label == 1).sum())}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--bot-dir", default="data/selfplay")
    p.add_argument("--split", default="test")
    p.add_argument("--name", default="bot")
    p.add_argument("--out", default="reports/humanlike.md")
    a = p.parse_args(argv)
    you = load(pathlib.Path(a.data_dir), a.split)
    bot = load(pathlib.Path(a.bot_dir), None)
    lines = [f"# Human-likeness: you ({a.split} split, {you.game_id.nunique()} games, {len(you)} moves after ply 10) vs {a.name} "
             f"({bot.game_id.nunique()} games, {len(bot)} moves)\n", "## By phase\n"] + stat_table(you, bot, "phase")
    lines += ["\n## By clock remaining\n"] + stat_table(you, bot, "clock_bucket")
    s = stylometry(you, bot)
    lines += ["\n## Stylometry: can a classifier tell you apart?\n",
              f"- move-level AUC: **{s['move_auc']:.3f}** (0.5 = indistinguishable)",
              f"- game-level AUC (per-game aggregates): **{s['game_auc']:.3f}**",
              f"- game-level AUC from averaged move-level scores: {s['move_vote_game_auc']:.3f}",
              f"- n: you {s['n_you_games']} games / {s['n_you_moves']} moves; bot {s['n_bot_games']} games / {s['n_bot_moves']} moves"]
    pathlib.Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

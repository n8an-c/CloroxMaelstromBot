"""Exploratory analysis of your games: opening tree, think-time curves, mistakes vs clock.

Writes reports/eda.md plus PNG figures under reports/.

    maelstrom-eda [--data-dir data/processed] [--out reports]
"""
from __future__ import annotations

import argparse
import pathlib
from collections import Counter, defaultdict

import chess
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLOCK_BINS = [0, 10, 30, 60, 120, 300, 10_000]
CLOCK_LABELS = ["<10s", "10-30s", "30-60s", "1-2m", "2-5m", ">5m"]


def opening_tree(games: pd.DataFrame, pos: pd.DataFrame, max_ply: int = 12, min_count: int = 3) -> list[str]:
    """Your most common lines by colour, as SAN sequences with counts and scores."""
    lines = []
    for color in ("white", "black"):
        g = games[games.user_color == color]
        if not len(g):
            continue
        seqs = Counter()
        score = defaultdict(float)
        for mv, sc in zip(g.moves_uci, g.user_score.fillna(0.5)):
            toks = mv.split()[:max_ply]
            for k in range(2, len(toks) + 1, 2):
                key = " ".join(toks[:k])
                seqs[key] += 1
                score[key] += sc
        lines.append(f"\n### As {color} ({len(g)} games)\n")
        lines.append("| line | games | share | score |\n|---|---|---|---|")
        for key, n in seqs.most_common(60):
            if n < min_count:
                break
            b = chess.Board()
            san = []
            for u in key.split():
                m = chess.Move.from_uci(u)
                san.append(b.san(m))
                b.push(m)
            lines.append(f"| {_san_line(san)} | {n} | {n / len(g):.1%} | {score[key] / n:.2f} |")
    return lines


def _san_line(san: list[str]) -> str:
    out = []
    for i, s in enumerate(san):
        if i % 2 == 0:
            out.append(f"{i // 2 + 1}.{s}")
        else:
            out.append(s)
    return " ".join(out)


def think_time_curves(u: pd.DataFrame, out: pathlib.Path) -> list[str]:
    lines = ["\n## Think time\n"]
    u = u[u.think_time.notna()]
    if not len(u):
        return lines + ["No clock data."]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # by ply
    byply = u.groupby(pd.cut(u.ply, bins=range(0, 141, 10))).think_time.median()
    axes[0].plot([iv.mid for iv in byply.index], byply.values, marker="o")
    axes[0].set(title="median think time by ply", xlabel="ply", ylabel="seconds")
    # by clock remaining
    cb = pd.cut(u.clock_before, bins=CLOCK_BINS, labels=CLOCK_LABELS)
    byclock = u.groupby(cb, observed=True).think_time.median()
    axes[1].bar(byclock.index.astype(str), byclock.values)
    axes[1].set(title="median think time by clock remaining", ylabel="seconds")
    # distribution by speed
    for spd, grp in u.groupby("speed"):
        axes[2].hist(np.log10(grp.think_time.clip(lower=0.1)), bins=40, alpha=0.5, label=f"{spd} (n={len(grp)})", density=True)
    axes[2].set(title="log10 think time", xlabel="log10 seconds")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out / "think_time.png", dpi=120)
    plt.close(fig)
    lines.append("![think time](think_time.png)\n")
    lines.append("| speed | moves | median s | mean s | p90 s | share < 1s |\n|---|---|---|---|---|---|")
    for spd, grp in u.groupby("speed"):
        t = grp.think_time
        lines.append(f"| {spd} | {len(t)} | {t.median():.1f} | {t.mean():.1f} | {t.quantile(.9):.1f} | {(t < 1).mean():.1%} |")
    lines.append("\n| clock remaining | moves | median think s |\n|---|---|---|")
    for lab, grp in u.groupby(cb, observed=True):
        lines.append(f"| {lab} | {len(grp)} | {grp.think_time.median():.1f} |")
    return lines


def mistakes_vs_clock(u: pd.DataFrame, ann: pd.DataFrame | None, out: pathlib.Path) -> list[str]:
    lines = ["\n## Move quality vs clock\n"]
    if ann is None or not len(ann):
        return lines + ["Run `maelstrom-annotate` first to get Stockfish move-quality labels."]
    m = u.merge(ann, on=["game_id", "ply"], how="inner")
    if not len(m):
        return lines + ["No annotated positions among your moves."]
    m["clock_bin"] = pd.cut(m.clock_before, bins=CLOCK_BINS, labels=CLOCK_LABELS)
    tab = m.groupby("clock_bin", observed=True).agg(
        moves=("ply", "size"),
        acpl=("cp_loss", lambda s: s.clip(upper=1000).mean()),
        best=("is_best", "mean"),
        inacc=("quality", lambda s: (s == "inaccuracy").mean()),
        mistake=("quality", lambda s: (s == "mistake").mean()),
        blunder=("quality", lambda s: (s == "blunder").mean()),
    )
    lines.append("| clock remaining | moves | ACPL | best-move % | inaccuracy/100 | mistake/100 | blunder/100 |\n|---|---|---|---|---|---|---|")
    for lab, r in tab.iterrows():
        lines.append(f"| {lab} | {int(r.moves)} | {r.acpl:.0f} | {r.best:.1%} | {100 * r.inacc:.1f} | {100 * r.mistake:.1f} | {100 * r.blunder:.1f} |")
    phase = m.groupby("phase").agg(moves=("ply", "size"), acpl=("cp_loss", lambda s: s.clip(upper=1000).mean()),
                                   blunder=("quality", lambda s: (s == "blunder").mean()))
    lines.append("\n| phase | moves | ACPL | blunder/100 |\n|---|---|---|---|")
    for lab, r in phase.iterrows():
        lines.append(f"| {lab} | {int(r.moves)} | {r.acpl:.0f} | {100 * r.blunder:.1f} |")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(tab.index.astype(str), 100 * tab.blunder, label="blunders/100")
    ax.bar(tab.index.astype(str), 100 * tab.mistake, bottom=100 * tab.blunder, label="mistakes/100")
    ax.set(title="mistakes by clock remaining", ylabel="per 100 moves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "mistakes_vs_clock.png", dpi=120)
    plt.close(fig)
    lines.append("\n![mistakes](mistakes_vs_clock.png)")
    return lines


def overview(games: pd.DataFrame, pos: pd.DataFrame) -> list[str]:
    u = pos[pos.is_user]
    lines = ["# EDA: your games\n"]
    lines.append(f"- games: **{len(games)}**, your moves: **{len(u)}**, all plies: {len(pos)}")
    if games.datetime_utc.notna().any():
        lines.append(f"- date range: {games.datetime_utc.min():%Y-%m-%d} to {games.datetime_utc.max():%Y-%m-%d}")
    lines.append("\n| platform | speed | games | your mean rating | lichess-equiv | score | mean plies |\n|---|---|---|---|---|---|---|")
    for (pf, sp), g in games.groupby(["platform", "speed"]):
        lines.append(f"| {pf} | {sp} | {len(g)} | {g.user_elo.mean():.0f} | {g.user_elo_lichess.mean():.0f} | {g.user_score.mean():.3f} | {g.n_plies.mean():.0f} |")
    if "split" in games:
        lines.append("\nsplit: " + ", ".join(f"{k}={v}" for k, v in games.split.value_counts().items()))
    lines.append("\n| phase | your moves | share |\n|---|---|---|")
    for ph, n in u.phase.value_counts().items():
        lines.append(f"| {ph} | {n} | {n / len(u):.1%} |")
    tc = games.groupby("time_control").size().sort_values(ascending=False).head(8)
    lines.append("\nmost common time controls: " + ", ".join(f"{k} ({v})" for k, v in tc.items()))
    return lines


def rating_history(games: pd.DataFrame, out: pathlib.Path) -> list[str]:
    g = games[games.user_elo.notna() & games.datetime_utc.notna()]
    if not len(g):
        return []
    fig, ax = plt.subplots(figsize=(10, 4))
    for (pf, sp), grp in g.groupby(["platform", "speed"]):
        ax.plot(grp.datetime_utc, grp.user_elo.rolling(20, min_periods=1).mean(), label=f"{pf} {sp}")
    ax.set(title="your rating over time (20-game rolling mean, native scale)", ylabel="rating")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "rating_history.png", dpi=120)
    plt.close(fig)
    return ["\n## Rating history\n", "![rating](rating_history.png)"]


def run(data_dir: pathlib.Path, out: pathlib.Path) -> pathlib.Path:
    games = pd.read_parquet(data_dir / "games.parquet")
    pos = pd.read_parquet(data_dir / "positions.parquet")
    ann_path = data_dir / "annotations.parquet"
    ann = pd.read_parquet(ann_path) if ann_path.exists() else None
    out.mkdir(parents=True, exist_ok=True)
    u = pos[pos.is_user]
    lines = overview(games, pos)
    lines += rating_history(games, out)
    lines += ["\n## Opening repertoire (first 12 plies)\n"] + opening_tree(games, pos)
    lines += think_time_curves(u, out)
    lines += mistakes_vs_clock(u, ann, out)
    report = out / "eda.md"
    report.write_text("\n".join(lines) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--out", default="reports")
    a = p.parse_args(argv)
    r = run(pathlib.Path(a.data_dir), pathlib.Path(a.out))
    print(f"report -> {r}")


if __name__ == "__main__":
    main()

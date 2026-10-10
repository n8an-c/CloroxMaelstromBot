"""Move-prediction metrics with game-level bootstrap confidence intervals.

Every model writes a *predictions* table with one row per evaluated position:

    game_id, ply, pred_move (argmax, UCI), p_played (probability assigned to the
    move actually played; NaN if the model has no distribution), top3 (list of
    up to 3 UCI moves, most likely first)

`evaluate()` joins that with positions.parquet and reports top-1, top-3 and
NLL (mean -log p_played, with p floored at 1e-6) overall, after ply 10, and
by slice. Positions the model could not score (no prediction) count as misses
for accuracy and are excluded from NLL, with the coverage reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

P_FLOOR = 1e-6
CLOCK_BINS = [-1, 10, 30, 10_000]
CLOCK_LABELS = ["<10s", "10-30s", ">30s"]


def _top3_hit(top3, played) -> bool:
    return played in (list(top3) if top3 is not None else [])


def join_predictions(pos: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    df = pos.merge(preds, on=["game_id", "ply"], how="left")
    df["covered"] = df["pred_move"].notna()
    df["hit1"] = df["pred_move"] == df["move_uci"]
    df["hit3"] = [_top3_hit(t, m) if c else False for t, m, c in zip(df.get("top3", [None] * len(df)), df["move_uci"], df["covered"])]
    df["nll"] = -np.log(df["p_played"].clip(lower=P_FLOOR))
    return df


def summarize(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"n": 0}
    has_p = df["p_played"].notna()
    return {
        "n": n,
        "coverage": float(df["covered"].mean()),
        "top1": float(df["hit1"].mean()),
        "top3": float(df["hit3"].mean()),
        "nll": float(df.loc[has_p, "nll"].mean()) if has_p.any() else float("nan"),
        "nll_n": int(has_p.sum()),
    }


def bootstrap_ci(df: pd.DataFrame, metric: str, n_boot: int = 500, seed: int = 0) -> tuple[float, float]:
    """95% CI by resampling games (not positions)."""
    rng = np.random.default_rng(seed)
    col = {"top1": "hit1", "top3": "hit3", "nll": "nll"}[metric]
    d = df if metric != "nll" else df[df["p_played"].notna()]
    if d.empty:
        return (float("nan"), float("nan"))
    per_game = d.groupby("game_id")[col].agg(["sum", "size"])
    s, c = per_game["sum"].to_numpy(dtype=float), per_game["size"].to_numpy(dtype=float)
    G = len(s)
    idx = rng.integers(0, G, size=(n_boot, G))
    vals = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def add_slices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clock_bucket"] = pd.cut(df["clock_before"].fillna(10_000), bins=CLOCK_BINS, labels=CLOCK_LABELS)
    df["after_ply10"] = df["ply"] > 10
    if "seen_fen_in_train" in df:
        df["seen"] = np.where(df["seen_fen_in_train"], "seen", "unseen")
    return df


def evaluate(pos: pd.DataFrame, preds: pd.DataFrame, name: str, slices: tuple[str, ...] = ("phase", "speed", "platform", "clock_bucket", "seen", "quality")) -> pd.DataFrame:
    """Return a long table: model, slice, value, n, coverage, top1, top1_lo, top1_hi, top3, nll, nll_lo, nll_hi."""
    df = add_slices(join_predictions(pos, preds))
    rows = []

    def emit(slice_name, value, d):
        s = summarize(d)
        if s["n"] == 0:
            return
        lo1, hi1 = bootstrap_ci(d, "top1")
        lon, hin = bootstrap_ci(d, "nll")
        rows.append({"model": name, "slice": slice_name, "value": str(value), **s,
                     "top1_lo": lo1, "top1_hi": hi1, "nll_lo": lon, "nll_hi": hin})

    emit("all", "all plies", df)
    emit("all", "ply > 10", df[df["after_ply10"]])
    for sl in slices:
        if sl not in df:
            continue
        d10 = df[df["after_ply10"]] if sl != "phase" else df
        for v, d in d10.groupby(sl, observed=True, dropna=True):
            emit(sl, v, d)
    return pd.DataFrame(rows)


def format_table(results: pd.DataFrame, slice_name: str = "all") -> str:
    r = results[results["slice"] == slice_name]
    lines = ["| model | subset | n | coverage | top-1 | 95% CI | top-3 | NLL | 95% CI |", "|---|---|---|---|---|---|---|---|---|"]
    for _, x in r.iterrows():
        lines.append(f"| {x.model} | {x.value} | {x.n} | {x.coverage:.1%} | {x.top1:.1%} | [{x.top1_lo:.1%}, {x.top1_hi:.1%}] | "
                     f"{x.top3:.1%} | {x.nll:.3f} | [{x.nll_lo:.3f}, {x.nll_hi:.3f}] |")
    return "\n".join(lines)

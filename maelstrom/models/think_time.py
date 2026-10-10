"""Think-time model: predict how long *you* take on a move, so the bot can wait that long.

Target: log1p(seconds). Inputs are only things known before the move is played:
clock left, base time, increment, ply, legal-move count, material, whether the
opponent's last move was a capture or check, and (if Maia-3 features exist) the base
model's entropy, top-1 probability and its own ponder head. Gradient-boosted trees
(scikit-learn HistGradientBoostingRegressor) fitted on the train split; the residual
spread per prediction is kept so the bot can *sample* a think time rather than always
play the mean.

Reported: Pearson/Spearman r on log think time, MAE in seconds, and a KS statistic
between sampled and real think-time distributions on the test split, by speed and clock bucket.

    maelstrom-think-time train [--features data/features/maia3-5m_all.parquet]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import pickle
import sys

import chess
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

FEATS = ["clock_before", "log_clock", "clock_frac", "base_seconds", "increment", "ply", "legal_moves", "nonpawn_material",
         "n_pieces", "is_blitz", "is_chesscom", "is_black", "prev_capture", "prev_check", "prev_think",
         "entropy", "p_top1", "ponder"]


def prev_move_flags(pos_all: pd.DataFrame) -> pd.DataFrame:
    """For every ply, flags about the move that led to it (the opponent's last move) and the
    user's own previous think time; computed over the full positions table (both sides)."""
    p = pos_all.sort_values(["game_id", "ply"])
    g = p.groupby("game_id")
    out = pd.DataFrame({
        "game_id": p.game_id, "ply": p.ply,
        "prev_capture": g.is_capture.shift(1).fillna(False).astype(float),
        "prev_check": g.is_check.shift(1).fillna(False).astype(float),
        "prev_think": g.think_time.shift(2).fillna(0.0),  # same player's previous think
    })
    return out


def make_frame(u: pd.DataFrame, flags: pd.DataFrame, feats: pd.DataFrame | None) -> pd.DataFrame:
    df = u.merge(flags, on=["game_id", "ply"], how="left")
    if feats is not None:
        f = feats[["game_id", "ply", "entropy", "topk_logp", "ponder"]].copy()
        f["p_top1"] = f.topk_logp.map(lambda l: math.exp(l[0]))
        df = df.merge(f.drop(columns="topk_logp"), on=["game_id", "ply"], how="left")
    else:
        df["entropy"] = np.nan; df["p_top1"] = np.nan; df["ponder"] = np.nan
    df["log_clock"] = np.log1p(df.clock_before.clip(lower=0))
    df["clock_frac"] = df.clock_before / df.base_seconds.clip(lower=1)
    df["is_blitz"] = (df.speed == "blitz").astype(float)
    df["is_chesscom"] = (df.platform == "chesscom").astype(float)
    df["is_black"] = (df.mover_color == "black").astype(float)
    df["y"] = np.log1p(df.think_time.clip(lower=0))
    return df


class ThinkTimeModel:
    def __init__(self, seed: int = 0, feats: list[str] | None = None):
        self.feats = list(feats or FEATS)
        self.reg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
                                                 l2_regularization=1.0, random_state=seed, early_stopping=True, validation_fraction=0.1)
        self.resid_std_by_bucket: dict[tuple, float] = {}
        self.resid_samples: dict[tuple, np.ndarray] = {}   # empirical residuals per bucket (keeps the premove spike and the heavy tail)
        self.global_resid = np.zeros(1)
        self.global_std = 0.5

    def fit(self, df: pd.DataFrame):
        self.feats = [f for f in self.feats if df[f].notna().any()]  # drop all-NaN columns (ablations)
        X, y = df[self.feats].to_numpy(dtype=float), df["y"].to_numpy()
        self.reg.fit(X, y)
        r = y - self.reg.predict(X)
        self.global_std = float(r.std())
        b = _bucket(df)
        self.resid_std_by_bucket = {k: float(v) for k, v in pd.Series(r).groupby(b.values).std().items()}
        rng = np.random.default_rng(0)
        self.global_resid = rng.choice(r, size=min(len(r), 5000), replace=False)
        for k, idx in pd.Series(np.arange(len(r))).groupby(b.values).groups.items():
            rr = r[np.asarray(list(idx))]
            self.resid_samples[k] = rng.choice(rr, size=min(len(rr), 3000), replace=False)
        return self

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        return self.reg.predict(df[self.feats].to_numpy(dtype=float))

    def sample_seconds(self, df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        mu = self.predict_log(df)
        if self.resid_samples:
            eps = np.array([rng.choice(self.resid_samples.get(k, self.global_resid)) for k in _bucket(df)])
        else:  # legacy pickles
            eps = rng.normal(0, 1, len(mu)) * np.array([self.resid_std_by_bucket.get(k, self.global_std) for k in _bucket(df)])
        s = np.expm1(mu + eps).clip(0.0)
        return np.minimum(s, df.clock_before.to_numpy() * 0.9)  # never flag


def _bucket(df: pd.DataFrame) -> pd.Series:
    return pd.Series(list(zip(df.speed, pd.cut(df.clock_before.fillna(600), [-1, 10, 30, 120, 1e9], labels=False))), index=df.index)


def report(model: ThinkTimeModel, test: pd.DataFrame, seed: int = 0) -> tuple[str, dict]:
    rng = np.random.default_rng(seed)
    mu = model.predict_log(test)
    samp = model.sample_seconds(test, rng)
    real = test.think_time.to_numpy()
    pear = float(stats.pearsonr(mu, test.y)[0]); spear = float(stats.spearmanr(mu, test.y)[0])
    mae = float(np.abs(np.expm1(mu) - real).mean())
    ks_all = float(stats.ks_2samp(samp, real).statistic)
    lines = [f"- Pearson r (log s): **{pear:.3f}**, Spearman: {spear:.3f}, MAE: {mae:.1f}s, KS(sampled vs real): {ks_all:.3f}",
             "\n| speed | clock | n | real median s | predicted median s | sampled median s | KS |", "|---|---|---|---|---|---|---|"]
    lab = {0: "<10s", 1: "10-30s", 2: "30s-2m", 3: ">2m"}
    for (sp, cb), idx in test.groupby([test.speed, pd.cut(test.clock_before, [-1, 10, 30, 120, 1e9], labels=False)]).groups.items():
        i = test.index.get_indexer(idx)
        if len(i) < 30:
            continue
        lines.append(f"| {sp} | {lab[cb]} | {len(i)} | {np.median(real[i]):.1f} | {np.median(np.expm1(mu[i])):.1f} | {np.median(samp[i]):.1f} | {stats.ks_2samp(samp[i], real[i]).statistic:.3f} |")
    return "\n".join(lines), {"pearson": pear, "spearman": spear, "mae": mae, "ks": ks_all}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["train"])
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--features", default="data/features/maia3-5m_all.parquet")
    p.add_argument("--no-maia", action="store_true", help="ablation: without Maia-3 entropy/ponder features")
    p.add_argument("--out", default="models/think_time.pkl")
    p.add_argument("--report", default="reports/think_time.md")
    a = p.parse_args(argv)
    d = pathlib.Path(a.data_dir)
    pos_all = pd.read_parquet(d / "positions.parquet")
    flags = prev_move_flags(pos_all)
    u = pos_all[pos_all.is_user & pos_all.in_scope.fillna(False) & pos_all.think_time.notna()]
    feats = None if a.no_maia or not pathlib.Path(a.features).exists() else pd.read_parquet(a.features, columns=["game_id", "ply", "entropy", "topk_logp", "ponder"])
    df = make_frame(u, flags, feats)
    tr, te = df[df.split == "train"], df[df.split == "test"]
    print(f"train {len(tr)}  test {len(te)}  maia feats: {feats is not None}", file=sys.stderr)
    model = ThinkTimeModel().fit(tr)
    text, m = report(model, te)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(model, f)
    pathlib.Path(a.report).write_text(f"# Think-time model on the test split ({len(te)} of your moves)\n\n{text}\n")
    print(text); print(json.dumps(m), "->", a.out)


if __name__ == "__main__":
    main()

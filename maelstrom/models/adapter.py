"""Residual "you-adapter": re-rank a frozen Maia-3's top-K candidate moves for one player.

For each position we have (from maia3_features): the pooled hidden state h, the top-K
candidate moves with base log-probs b_k, and context c (platform, colour, speed, base
time, increment, clock left, ply, material). The adapter scores

    s_k = b_k / T  +  g(h, c, move_k)            g is a small MLP, zero-initialised

and takes softmax over the K candidates. Mass outside the top-K keeps the base model's
probabilities, so the output is a full distribution and NLL is comparable to the base.
At initialisation (T=1, g=0) the adapter IS the base model; training can only move
away from it where the player's data says so.

    maelstrom-adapter train  [--features data/features/maia3-5m_all.parquet] [--no-clock] [--max-train-games N]
    maelstrom-adapter eval   --ckpt models/adapter.pt --split test
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import chess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from maelstrom.eval.metrics import evaluate, format_table

K = 16
PIECE_IDX = {p: i for i, p in enumerate("pnbrqk")}
SPEED_IDX = {"blitz": 0, "rapid": 1, "bullet": 2, "classical": 3}
PLATFORM_IDX = {"lichess": 0, "chesscom": 1}
N_MOVE_FEATS = 8  # capture, check, promo, castle, piece one-hot is separate


# ----------------------------------------------------------------------------- features
def move_features(fen: str, moves: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-candidate: from-square, to-square, piece index, dense flags."""
    board = chess.Board(fen)
    n = len(moves)
    frm = np.zeros(K, dtype=np.int64)
    to = np.zeros(K, dtype=np.int64)
    piece = np.zeros(K, dtype=np.int64)
    flags = np.zeros((K, N_MOVE_FEATS), dtype=np.float32)
    mirror = board.turn == chess.BLACK
    for i, u in enumerate(moves[:K]):
        mv = chess.Move.from_uci(u)
        pt = board.piece_type_at(mv.from_square)
        f_sq, t_sq = mv.from_square, mv.to_square
        if mirror:  # side-to-move perspective, like Maia
            f_sq, t_sq = chess.square_mirror(f_sq), chess.square_mirror(t_sq)
        frm[i], to[i], piece[i] = f_sq, t_sq, (pt - 1) if pt else 0
        flags[i, 0] = board.is_capture(mv)
        flags[i, 1] = board.gives_check(mv)
        flags[i, 2] = mv.promotion is not None
        flags[i, 3] = board.is_castling(mv)
        flags[i, 4] = board.is_en_passant(mv)
        victim = board.piece_type_at(mv.to_square)
        flags[i, 5] = (victim or 0) / 6.0
        flags[i, 6] = (pt or 0) / 6.0
        flags[i, 7] = i / K  # base rank
    return frm, to, piece, flags


def _num(x, default: float) -> float:
    try:
        v = float(x)
        return default if math.isnan(v) else v
    except (TypeError, ValueError):
        return default


def context_features(row, use_clock: bool) -> np.ndarray:
    base = _num(row.base_seconds, 0.0)
    inc = _num(row.increment, 0.0)
    clk = _num(row.clock_before, -1.0) if use_clock else -1.0
    return np.array([
        math.log1p(base) / 8, inc / 10,
        (math.log1p(max(clk, 0)) / 8) if use_clock else 0.0,
        (clk / max(base, 1)) if use_clock else 0.0,
        float(clk < 30) if use_clock else 0.0,
        float(clk < 10) if use_clock else 0.0,
        row.ply / 150, row.nonpawn_material / 62, row.n_pieces / 14,
        math.log1p(row.legal_moves) / 4,
        row.entropy / 3, _num(row.mover_elo_lichess, 2000) / 3000, _num(row.opp_elo_lichess, 2000) / 3000,
        float(row.mover_color == "black"),
    ], dtype=np.float32)


def build_dataset(pos: pd.DataFrame, feats: pd.DataFrame, use_clock: bool = True) -> dict:
    df = pos.merge(feats, on=["game_id", "ply"], how="inner")
    n = len(df)
    H = np.stack(df["hidden"].to_numpy()).astype(np.float32)
    B = np.full((n, K), -30.0, dtype=np.float32)  # base log-probs, padded
    y = np.full(n, -1, dtype=np.int64)
    in_topk_mass = np.zeros(n, dtype=np.float32)
    frm = np.zeros((n, K), dtype=np.int64); to = np.zeros((n, K), dtype=np.int64); pc = np.zeros((n, K), dtype=np.int64)
    fl = np.zeros((n, K, N_MOVE_FEATS), dtype=np.float32)
    C = np.zeros((n, 14), dtype=np.float32)
    cat = np.zeros((n, 3), dtype=np.int64)  # platform, colour, speed
    for i, row in enumerate(df.itertuples(index=False)):
        moves = list(row.topk_moves)
        lp = np.asarray(row.topk_logp, dtype=np.float32)
        B[i, :len(moves)] = lp
        in_topk_mass[i] = float(np.exp(lp).sum())
        if row.move_uci in moves:
            y[i] = moves.index(row.move_uci)
        frm[i], to[i], pc[i], fl[i] = move_features(row.fen, moves)
        C[i] = context_features(row, use_clock)
        cat[i] = (PLATFORM_IDX.get(row.platform, 0), int(row.mover_color == "black"), SPEED_IDX.get(row.speed, 0))
    return {"H": H, "B": B, "y": y, "mass": in_topk_mass, "frm": frm, "to": to, "pc": pc, "fl": fl, "C": C, "cat": cat,
            "n_cand": df["topk_moves"].map(len).to_numpy(), "meta": df[["game_id", "ply", "move_uci", "logp_played", "topk_moves"]].reset_index(drop=True)}


# ----------------------------------------------------------------------------- model
class Adapter(nn.Module):
    def __init__(self, d_hidden: int, d_ctx: int = 14, d_model: int = 128, dropout: float = 0.2):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))
        self.sq_from = nn.Embedding(64, 16)
        self.sq_to = nn.Embedding(64, 16)
        self.piece = nn.Embedding(6, 8)
        self.platform = nn.Embedding(2, 4)
        self.colour = nn.Embedding(2, 4)
        self.speed = nn.Embedding(4, 4)
        d_in_pos = d_hidden + d_ctx + 12
        self.pos_mlp = nn.Sequential(nn.LayerNorm(d_in_pos), nn.Linear(d_in_pos, d_model), nn.GELU(), nn.Dropout(dropout))
        d_in_mv = d_model + 16 + 16 + 8 + N_MOVE_FEATS + 1
        self.mv_mlp = nn.Sequential(nn.Linear(d_in_mv, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1))
        nn.init.zeros_(self.mv_mlp[-1].weight); nn.init.zeros_(self.mv_mlp[-1].bias)  # residual start

    def forward(self, H, B, frm, to, pc, fl, C, cat, n_cand):
        ctx = torch.cat([self.platform(cat[:, 0]), self.colour(cat[:, 1]), self.speed(cat[:, 2])], -1)
        z = self.pos_mlp(torch.cat([H, C, ctx], -1))                     # (N, d)
        zk = z.unsqueeze(1).expand(-1, K, -1)
        mv = torch.cat([zk, self.sq_from(frm), self.sq_to(to), self.piece(pc), fl, B.unsqueeze(-1) / 10], -1)
        g = self.mv_mlp(mv).squeeze(-1)                                    # (N, K)
        s = B * torch.exp(-self.log_T) + g
        mask = torch.arange(K, device=B.device).unsqueeze(0) < n_cand.unsqueeze(1)
        return s.masked_fill(~mask, float("-inf"))                         # logits over K


def _tensors(d: dict, idx: np.ndarray, device: str):
    f = lambda k, dt: torch.as_tensor(d[k][idx], dtype=dt, device=device)
    return (f("H", torch.float32), f("B", torch.float32), f("frm", torch.long), f("to", torch.long), f("pc", torch.long),
            f("fl", torch.float32), f("C", torch.float32), f("cat", torch.long), f("n_cand", torch.long))


def full_logp_played(logits: torch.Tensor, d: dict, idx: np.ndarray) -> np.ndarray:
    """log p(played) under the adapter: top-K mass * softmax_k, or the base prob if played is outside top-K."""
    lsm = F.log_softmax(logits, -1).cpu().numpy()
    y = d["y"][idx]
    base_lp = d["meta"]["logp_played"].to_numpy()[idx]
    mass = np.log(np.clip(d["mass"][idx], 1e-9, 1.0))
    out = np.where(y >= 0, mass + lsm[np.arange(len(idx)), np.clip(y, 0, K - 1)], base_lp)
    return out


@torch.no_grad()
def predict(model: Adapter, d: dict, device: str = "cpu", batch: int = 4096) -> pd.DataFrame:
    model.eval()
    n = len(d["y"])
    rows = []
    for s in range(0, n, batch):
        idx = np.arange(s, min(n, s + batch))
        logits = model(*_tensors(d, idx, device))
        lp = full_logp_played(logits, d, idx)
        order = torch.argsort(logits, dim=-1, descending=True).cpu().numpy()
        meta = d["meta"].iloc[idx]
        for j, (gid, ply, moves) in enumerate(zip(meta.game_id, meta.ply, meta.topk_moves)):
            top = [moves[k] for k in order[j] if k < len(moves)]
            rows.append({"game_id": gid, "ply": ply, "pred_move": top[0], "p_played": float(np.exp(lp[j])), "top3": top[:3],
                         "logp_played": float(lp[j])})
    return pd.DataFrame(rows)


def train(d_train: dict, d_val: dict, d_hidden: int, epochs: int = 30, lr: float = 1e-3, wd: float = 1e-2, batch: int = 512,
          device: str = "cpu", seed: int = 0, patience: int = 5, d_model: int = 128, dropout: float = 0.2, log=print) -> tuple[Adapter, dict]:
    torch.manual_seed(seed); np.random.seed(seed)
    model = Adapter(d_hidden, d_model=d_model, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = len(d_train["y"])
    keep = np.where(d_train["y"] >= 0)[0]  # train only where the played move is a candidate
    best, best_state, bad, hist = float("inf"), None, 0, []

    def val_nll():
        p = predict(model, d_val, device)
        return float(-p["logp_played"].mean()), float((p["pred_move"] == d_val["meta"]["move_uci"]).mean())

    base_nll = float(-d_val["meta"]["logp_played"].mean())
    v0, a0 = val_nll()
    log(f"  init: val NLL {v0:.4f} (base {base_nll:.4f}) top-1 {a0:.4f}")
    for ep in range(1, epochs + 1):
        model.train(); perm = np.random.permutation(keep); tot, t0 = 0.0, time.time()
        for s in range(0, len(perm), batch):
            idx = perm[s:s + batch]
            logits = model(*_tensors(d_train, idx, device))
            loss = F.cross_entropy(logits, torch.as_tensor(d_train["y"][idx], device=device))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss) * len(idx)
        v, a = val_nll()
        hist.append({"epoch": ep, "train_ce": tot / len(perm), "val_nll": v, "val_top1": a})
        log(f"  epoch {ep:2d} train CE {tot / len(perm):.4f}  val NLL {v:.4f}  top-1 {a:.4f}  T={math.exp(float(model.log_T)):.3f}  {time.time() - t0:.0f}s")
        if v < best - 1e-4:
            best, best_state, bad = v, {k: x.detach().clone() for k, x in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, {"base_val_nll": base_nll, "best_val_nll": best, "history": hist}


# ----------------------------------------------------------------------------- CLI
def load_inputs(data_dir: pathlib.Path, features: pathlib.Path):
    pos = pd.read_parquet(data_dir / "positions.parquet")
    pos = pos[pos.is_user & pos.in_scope.fillna(False)]
    ann = data_dir / "annotations.parquet"
    if ann.exists():
        pos = pos.merge(pd.read_parquet(ann)[["game_id", "ply", "quality", "cp_loss"]], on=["game_id", "ply"], how="left")
    feats = pd.read_parquet(features)
    return pos, feats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["train", "eval"])
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--features", default="data/features/maia3-5m_all.parquet")
    p.add_argument("--ckpt", default="models/adapter_maia3-5m.pt")
    p.add_argument("--name", default="adapter(maia3-5m)")
    p.add_argument("--split", default="test")
    p.add_argument("--no-clock", action="store_true")
    p.add_argument("--max-train-games", type=int, default=None, help="data-size curve: use the most recent N train games")
    p.add_argument("--val-recent", type=float, default=None,
                   help="temporal validation: hold out this fraction of the most recent train games as val instead of the random val split")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--preds-dir", default="data/preds")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    pos, feats = load_inputs(pathlib.Path(a.data_dir), pathlib.Path(a.features))
    use_clock = not a.no_clock
    ck = pathlib.Path(a.ckpt)
    if a.cmd == "train":
        tr = pos[pos.split == "train"]
        va = pos[pos.split == "val"]
        games = pd.read_parquet(pathlib.Path(a.data_dir) / "games.parquet")
        if a.val_recent:
            order = games[games.game_id.isin(tr.game_id.unique())].sort_values("datetime_utc").game_id
            n_val = int(round(len(order) * a.val_recent))
            va_ids = set(order.tail(n_val)); va = tr[tr.game_id.isin(va_ids)]; tr = tr[~tr.game_id.isin(va_ids)]
            print(f"temporal validation: {len(va_ids)} most recent train games held out", file=sys.stderr)
        if a.max_train_games:
            recent = games[games.game_id.isin(tr.game_id.unique())].sort_values("datetime_utc").game_id.tail(a.max_train_games)
            tr = tr[tr.game_id.isin(recent)]
        print(f"building datasets: train {len(tr)} positions / {tr.game_id.nunique()} games, use_clock={use_clock}", file=sys.stderr)
        d_tr = build_dataset(tr, feats, use_clock)
        d_va = build_dataset(va, feats, use_clock)
        print(f"  played move in top-{K}: train {(d_tr['y'] >= 0).mean():.4f}  val {(d_va['y'] >= 0).mean():.4f}", file=sys.stderr)
        model, info = train(d_tr, d_va, d_tr["H"].shape[1], a.epochs, a.lr, a.wd, device="cpu", seed=a.seed,
                            d_model=a.d_model, dropout=a.dropout, log=lambda s: print(s, file=sys.stderr))
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "d_hidden": d_tr["H"].shape[1], "use_clock": use_clock,
                    "d_model": a.d_model, "dropout": a.dropout, "info": info, "features": str(a.features),
                    "n_train_games": int(tr.game_id.nunique()), "n_train_pos": int(len(tr))}, ck)
        print(json.dumps({k: v for k, v in info.items() if k != "history"}), "->", ck)
    else:
        state = torch.load(ck, map_location="cpu")
        model = Adapter(state["d_hidden"], d_model=state.get("d_model", 128), dropout=state.get("dropout", 0.2))
        model.load_state_dict(state["state_dict"])
        ev = pos[pos.split == a.split]
        d = build_dataset(ev, feats, state["use_clock"])
        preds = predict(model, d)
        pathlib.Path(a.preds_dir).mkdir(parents=True, exist_ok=True)
        preds.to_parquet(pathlib.Path(a.preds_dir) / f"{a.split}_{a.name}.parquet", index=False)
        base = feats.merge(ev[["game_id", "ply"]], on=["game_id", "ply"])[["game_id", "ply", "pred_move", "p_played", "top3"]]
        res = pd.concat([evaluate(ev, base, state["features"].split("/")[-1].split("_")[0]), evaluate(ev, preds, a.name)], ignore_index=True)
        lines = [f"# {a.name} vs base on the {a.split} split ({len(ev)} of your moves)\n", format_table(res, "all")]
        for sl in ("phase", "clock_bucket", "seen", "quality", "platform", "speed"):
            if (res["slice"] == sl).any():
                lines += [f"\n## by {sl}\n", format_table(res, sl)]
        out = pathlib.Path(a.out or f"reports/{a.name.replace('(', '_').replace(')', '')}_{a.split}.md")
        out.write_text("\n".join(lines) + "\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main()

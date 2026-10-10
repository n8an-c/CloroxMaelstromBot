"""Batch Maia-3 inference over positions: predictions for evaluation + features for the adapter.

For every selected (game_id, ply) we rebuild the 8-position history from the game's
move list, tokenize exactly as maia3's UCI engine does (side-to-move perspective,
earliest position repeated when the history is short), and run the frozen model in
batches. Output table (one row per position):

    game_id, ply, pred_move, p_played, top3                  -> evaluation contract
    topk_moves (list[str]), topk_logp (list[float])           -> top-K candidates for the re-ranker
    logp_played, entropy, n_legal
    wdl (list[3])  win/draw/loss from the side to move; ponder (Maia-3's think-time head)
    hidden (list[dim_vit])  mean-pooled final-layer state, input to the adapter

Elo inputs are the Lichess-equivalent ratings of the mover and the opponent
(`--elo-mode` lets you fix both to a constant for ablations).

    maelstrom-maia3-features --model maia3-5m [--split test] [--user-only] [--limit N]
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time
from collections import deque

import chess
import numpy as np
import pandas as pd
import torch

from maia3.dataset import get_historical_tokens, get_legal_moves_mask, tokenize_board
from maia3.model_registry import resolve_checkpoint_path
from maia3.models import MAIA3Model
from maia3.uci import load_model, parse_args as maia3_parse_args
from maia3.utils import get_all_possible_moves, mirror_move

ALL_MOVES = get_all_possible_moves()
ALL_MOVES_DICT = {m: i for i, m in enumerate(ALL_MOVES)}


def build_cfg(model: str, checkpoint_path: str | None, device: str):
    argv = ["--model", model, "--device", device, "--no-use-amp"]
    if checkpoint_path:
        argv += ["--checkpoint-path", checkpoint_path]
    return maia3_parse_args(argv)


def load_maia3(model: str = "maia3-5m", checkpoint_path: str | None = None, device: str = "cpu",
               random_init: bool = False):
    """Return (cfg, model). random_init=True builds the architecture without weights (tests)."""
    cfg = build_cfg(model, checkpoint_path, device)
    if random_init:
        net = MAIA3Model(cfg).to(device).eval()
        return cfg, net
    if cfg.checkpoint_path is None:
        cfg.checkpoint_path = resolve_checkpoint_path(cfg.model_spec, checkpoint_filename=cfg.checkpoint_filename,
                                                      cache_dir=cfg.cache_dir, local_files_only=cfg.local_files_only)
    return cfg, load_model(cfg)


class GameHistory:
    """Replays a game's UCI moves, yielding (board, history-deque) before each ply."""

    def __init__(self, moves_uci: str, history_len: int):
        self.moves = moves_uci.split()
        self.history_len = history_len

    def positions(self, wanted_plies: set[int]):
        board = chess.Board()
        hist: deque = deque(maxlen=self.history_len)
        hist.append(tokenize_board(board))
        for ply, u in enumerate(self.moves, start=1):
            if ply in wanted_plies:
                yield ply, board.copy(stack=False), deque(hist, maxlen=self.history_len)
            board.push(chess.Move.from_uci(u))
            hist.append(tokenize_board(board))


def _pooled_hidden(model: MAIA3Model, tokens, self_elos, oppo_elos):
    """Re-run the trunk to grab the pooled hidden state (model.forward only returns heads)."""
    cfg = model.cfg
    t = tokens[:, :, :12 * cfg.history + (3 if cfg.include_time_info else 0)]
    se = model.interpolate_elo(self_elos).unsqueeze(1).expand(-1, 64, -1)
    oe = model.interpolate_elo(oppo_elos).unsqueeze(1).expand(-1, 64, -1)
    x = model.token_projection(torch.cat([t, se, oe], dim=-1))
    if hasattr(model, "abs_pe"):
        x = model.abs_pe(x)
    x = model.transformer(x)
    return model.last_ln(x.mean(dim=1))


@torch.no_grad()
def run_batch(model, cfg, items: list[dict], device: str, topk: int, want_hidden: bool) -> list[dict]:
    tokens = torch.stack([get_historical_tokens(it["hist"], cfg, 0.0, 0.0, 0.0, 0.0) for it in items]).to(device)
    self_elos = torch.tensor([it["self_elo"] for it in items], dtype=torch.long, device=device)
    oppo_elos = torch.tensor([it["oppo_elo"] for it in items], dtype=torch.long, device=device)
    logits_move, logits_value, logits_ponder = model(tokens, self_elos, oppo_elos)
    hidden = _pooled_hidden(model, tokens, self_elos, oppo_elos).cpu().numpy() if want_hidden else None
    wdl = torch.softmax(logits_value, dim=-1).cpu().numpy()
    ponder = logits_ponder.reshape(-1).cpu().numpy()
    out = []
    for i, it in enumerate(items):
        board = it["board"]
        mask = get_legal_moves_mask(board, ALL_MOVES_DICT).to(device)
        lg = logits_move[i].masked_fill(~mask, float("-inf"))
        logp = torch.log_softmax(lg, dim=-1)
        k = min(topk, int(mask.sum()))
        top_lp, top_idx = torch.topk(logp, k)
        black = board.turn == chess.BLACK
        top_moves = [mirror_move(ALL_MOVES[j]) if black else ALL_MOVES[j] for j in top_idx.tolist()]
        played_key = mirror_move(it["played"]) if black else it["played"]
        lp_played = float(logp[ALL_MOVES_DICT[played_key]]) if played_key in ALL_MOVES_DICT else float("nan")
        p = logp.exp()
        ent = float(-(p[mask] * logp[mask]).sum())
        row = {
            "game_id": it["game_id"], "ply": it["ply"],
            "pred_move": top_moves[0], "p_played": math.exp(lp_played) if not math.isnan(lp_played) else np.nan,
            "top3": top_moves[:3],
            "topk_moves": top_moves, "topk_logp": top_lp.cpu().numpy().astype(np.float32).tolist(),
            "logp_played": lp_played, "entropy": ent, "n_legal": int(mask.sum()),
            "wdl": wdl[i].astype(np.float32).tolist(), "ponder": float(ponder[i]),
        }
        if want_hidden:
            row["hidden"] = hidden[i].astype(np.float32).tolist()
        out.append(row)
    return out


def featurize(games: pd.DataFrame, pos: pd.DataFrame, model, cfg, device: str = "cpu", batch_size: int = 256,
              topk: int = 16, want_hidden: bool = True, elo_mode: str = "actual", const_elo: int = 2000,
              log_every: int = 20) -> pd.DataFrame:
    moves_by_game = dict(zip(games["game_id"], games["moves_uci"]))
    wanted = pos.groupby("game_id")["ply"].apply(set).to_dict()
    meta = pos.set_index(["game_id", "ply"])[["move_uci", "mover_elo_lichess", "opp_elo_lichess"]]
    items, rows, t0, nb = [], [], time.time(), 0
    for gid, plies in wanted.items():
        for ply, board, hist in GameHistory(moves_by_game[gid], cfg.history).positions(plies):
            m = meta.loc[(gid, ply)]
            se, oe = (const_elo, const_elo) if elo_mode == "const" else (m["mover_elo_lichess"], m["opp_elo_lichess"])
            items.append({"game_id": gid, "ply": ply, "board": board, "hist": hist, "played": m["move_uci"],
                          "self_elo": int(se if pd.notna(se) else const_elo), "oppo_elo": int(oe if pd.notna(oe) else const_elo)})
            if len(items) >= batch_size:
                rows += run_batch(model, cfg, items, device, topk, want_hidden)
                items, nb = [], nb + 1
                if nb % log_every == 0:
                    print(f"  {len(rows)}/{len(pos)}  {len(rows) / (time.time() - t0):.0f} pos/s", file=sys.stderr, flush=True)
    if items:
        rows += run_batch(model, cfg, items, device, topk, want_hidden)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="maia3-5m")
    p.add_argument("--checkpoint-path", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--out-dir", default="data/features")
    p.add_argument("--split", default=None, help="train|val|test; default all in-scope")
    p.add_argument("--all-moves", action="store_true", help="include opponents' moves")
    p.add_argument("--no-hidden", action="store_true")
    p.add_argument("--elo-mode", default="actual", choices=["actual", "const"])
    p.add_argument("--const-elo", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard", default=None, help="k/n: process the k-th of n shards (by game, in order); output tag gets _shard{k}of{n}")
    a = p.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    d = pathlib.Path(a.data_dir)
    games = pd.read_parquet(d / "games.parquet")
    pos = pd.read_parquet(d / "positions.parquet")
    pos = pos[pos.in_scope.fillna(False)]
    if not a.all_moves:
        pos = pos[pos.is_user]
    if a.split:
        pos = pos[pos.split == a.split]
    if a.shard:
        k, n = (int(x) for x in a.shard.split("/"))
        gids = sorted(pos.game_id.unique())
        pos = pos[pos.game_id.isin(gids[k::n])]
    if a.limit:
        pos = pos.head(a.limit)
    cfg, model = load_maia3(a.model, a.checkpoint_path, a.device)
    print(f"{a.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params; {len(pos)} positions", file=sys.stderr)
    df = featurize(games, pos, model, cfg, a.device, a.batch_size, a.topk, not a.no_hidden, a.elo_mode, a.const_elo)
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model}_{a.split or 'all'}{'_all-moves' if a.all_moves else ''}{'_const' + str(a.const_elo) if a.elo_mode == 'const' else ''}"
    if a.shard:
        tag += f"_shard{a.shard.replace('/', 'of')}"
    df.to_parquet(out / f"{tag}.parquet", index=False)
    acc = (df["pred_move"] == pos.set_index(["game_id", "ply"]).loc[list(zip(df.game_id, df.ply)), "move_uci"].values).mean()
    print(f"{len(df)} rows -> {out / (tag + '.parquet')}   top-1 {acc:.3%}   mean NLL {-df['logp_played'].mean():.3f}")


if __name__ == "__main__":
    main()

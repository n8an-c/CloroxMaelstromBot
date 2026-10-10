"""UCI engine: personal opening book -> Maia-3 (+ you-adapter) sampled policy -> think-time delay.

    maelstrom-uci [--model maia3-5m] [--adapter models/adapter_maia3-5m.pt] [--think-time models/think_time.pkl]
                  [--book data/processed/positions.parquet] [--book-splits train,val] [--no-wait]

UCI options: Temperature (policy sampling; 0 = argmax), TopP, BookMaxPly, BookMinCount,
SelfElo, OppoElo, ThinkTime (true/false), MaxThinkFrac (cap on share of remaining clock).
`go wtime/btime/winc/binc` supplies the clock; the engine sleeps for the predicted think
time before answering, capped by the real clock. Designed for lichess-bot.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import pickle
import sys
import time
from collections import deque

import chess
import numpy as np
import pandas as pd
import torch

from maelstrom.data.parse import nonpawn_material, speed_class
from maelstrom.eval.baselines import PositionTable
from maelstrom.models.maia3_features import load_maia3, run_batch
from maelstrom.models.adapter import Adapter, K, context_features, move_features, PLATFORM_IDX, SPEED_IDX


class _Row:  # duck-typed row for context_features / think-time features
    def __init__(self, **kw):
        self.__dict__.update(kw)


class MaelstromEngine:
    def __init__(self, a):
        self.a = a
        self.opts = {"Temperature": 1.0, "TopP": 1.0, "BookMaxPly": 20, "BookMinCount": 2, "SelfElo": a.elo, "OppoElo": a.elo,
                     "ThinkTime": not a.no_wait, "MaxThinkFrac": 0.25}
        self.rng = np.random.default_rng(a.seed)
        self.board = chess.Board()
        self.moves: list[str] = []
        self.base, self.inc = 180.0, 2.0
        self.model = self.cfg = self.adapter = self.think = self.book = None

    # ---- loading (lazy, on isready)
    def ensure_loaded(self):
        if self.model is not None:
            return
        self.cfg, self.model = load_maia3(self.a.model, device="cpu")
        torch.set_num_threads(self.a.threads)
        if self.a.adapter and pathlib.Path(self.a.adapter).exists():
            st = torch.load(self.a.adapter, map_location="cpu")
            self.adapter = Adapter(st["d_hidden"], d_model=st.get("d_model", 128), dropout=st.get("dropout", 0.2))
            self.adapter.load_state_dict(st["state_dict"]); self.adapter.eval()
            self.adapter_use_clock = st["use_clock"]
        if self.a.think_time and pathlib.Path(self.a.think_time).exists():
            with open(self.a.think_time, "rb") as f:
                self.think = pickle.load(f)
        if self.a.book and pathlib.Path(self.a.book).exists():
            pos = pd.read_parquet(self.a.book, columns=["fen", "move_uci", "is_user", "in_scope", "split", "ply"])
            pos = pos[pos.is_user & pos.in_scope.fillna(False) & pos.split.isin(self.a.book_splits.split(","))]
            self.book = PositionTable(pos)
        print(f"info string loaded model={self.a.model} adapter={self.adapter is not None} think={self.think is not None} book={self.book is not None}", flush=True)

    # ---- UCI plumbing
    def cmd_uci(self):
        print("id name CloroxMaelstromBot")
        print("id author n8an-c")
        print("option name Temperature type string default 1.0")
        print("option name TopP type string default 1.0")
        print("option name BookMaxPly type spin default 20 min 0 max 60")
        print("option name BookMinCount type spin default 2 min 1 max 100")
        print(f"option name SelfElo type spin default {self.a.elo} min 0 max 5000")
        print(f"option name OppoElo type spin default {self.a.elo} min 0 max 5000")
        print("option name ThinkTime type check default true")
        print("option name MaxThinkFrac type string default 0.25")
        print("uciok", flush=True)

    def cmd_setoption(self, line: str):
        toks = line.split()
        if "name" in toks and "value" in toks:
            name = " ".join(toks[toks.index("name") + 1:toks.index("value")])
            value = " ".join(toks[toks.index("value") + 1:])
            if name in ("Temperature", "TopP", "MaxThinkFrac"):
                self.opts[name] = float(value)
            elif name in ("BookMaxPly", "BookMinCount", "SelfElo", "OppoElo"):
                self.opts[name] = int(value)
            elif name == "ThinkTime":
                self.opts[name] = value.lower() == "true"

    def cmd_position(self, line: str):
        toks = line.split()
        if "startpos" in toks:
            self.board = chess.Board()
        elif "fen" in toks:
            i = toks.index("fen")
            self.board = chess.Board(" ".join(toks[i + 1:i + 7]))
        self.moves = []
        if "moves" in toks:
            for u in toks[toks.index("moves") + 1:]:
                self.board.push_uci(u); self.moves.append(u)

    # ---- choosing a move
    def _history(self):
        from maia3.dataset import tokenize_board
        b = chess.Board(); hist = deque(maxlen=self.cfg.history); hist.append(tokenize_board(b))
        for u in self.moves:
            b.push_uci(u); hist.append(tokenize_board(b))
        return hist

    def policy(self, clock_left: float | None) -> tuple[list[str], np.ndarray, float]:
        """Returns (candidate moves, probabilities over them, model entropy); sets self.last_expected_score
        (Maia-3 value head, side-to-move POV, win + 0.5 * draw) for resignation logic."""
        board = self.board
        item = {"game_id": "live", "ply": len(self.moves) + 1, "board": board, "hist": self._history(), "played": "0000",
                "self_elo": self.opts["SelfElo"], "oppo_elo": self.opts["OppoElo"]}
        r = run_batch(self.model, self.cfg, [item], "cpu", K, want_hidden=self.adapter is not None)[0]
        moves, logp = list(r["topk_moves"]), np.asarray(r["topk_logp"], dtype=np.float64)
        l, d, w = r["wdl"]  # Maia-3 value labels are [loss, draw, win] for the side to move
        self.last_expected_score = float(w + 0.5 * d)
        if self.adapter is not None:
            mat, n_pieces = nonpawn_material(board)
            speed = speed_class(int(self.base), int(self.inc), "lichess") or "blitz"
            row = _Row(base_seconds=self.base, increment=self.inc, clock_before=clock_left if clock_left is not None else self.base,
                       ply=len(self.moves) + 1, nonpawn_material=mat, n_pieces=n_pieces, legal_moves=board.legal_moves.count(),
                       entropy=r["entropy"], mover_elo_lichess=self.opts["SelfElo"], opp_elo_lichess=self.opts["OppoElo"],
                       mover_color="black" if board.turn == chess.BLACK else "white")
            frm, to, pc, fl = move_features(board.fen(), moves)
            B = np.full((1, K), -30.0, dtype=np.float32); B[0, :len(moves)] = logp
            t = lambda x, dt: torch.as_tensor(np.asarray(x)[None], dtype=dt)
            with torch.no_grad():
                logits = self.adapter(t(r["hidden"], torch.float32), t(B[0], torch.float32), t(frm, torch.long), t(to, torch.long),
                                      t(pc, torch.long), t(fl, torch.float32), t(context_features(row, self.adapter_use_clock), torch.float32),
                                      t([PLATFORM_IDX["lichess"], int(board.turn == chess.BLACK), SPEED_IDX.get(speed, 0)], torch.long),
                                      torch.tensor([len(moves)]))
            lsm = torch.log_softmax(logits[0, :len(moves)], -1).numpy()
            logp = np.log(np.exp(logp).sum()) + lsm  # keep top-K mass, re-rank inside
        return moves, np.exp(logp), r["entropy"]

    def sample(self, moves: list[str], probs: np.ndarray) -> str:
        T = self.opts["Temperature"]
        if T <= 0:
            return moves[int(np.argmax(probs))]
        p = probs ** (1.0 / T); p /= p.sum()
        if self.opts["TopP"] < 1.0:
            order = np.argsort(-p); cum = np.cumsum(p[order])
            keep = order[: max(1, int(np.searchsorted(cum, self.opts["TopP"]) + 1))]
            mask = np.zeros_like(p); mask[keep] = 1; p = p * mask; p /= p.sum()
        return moves[int(self.rng.choice(len(moves), p=p))]

    def think_seconds(self, clock_left: float | None, entropy: float, p_top1: float, ponder: float | None) -> float:
        if not self.opts["ThinkTime"] or self.think is None or clock_left is None:
            return 0.0
        board = self.board; prev = board.peek() if board.move_stack else None
        prev_capture = prev_check = 0.0
        if prev is not None:
            b2 = board.copy(); b2.pop(); prev_capture = float(b2.is_capture(prev)); prev_check = float(board.is_check())
        mat, n_pieces = nonpawn_material(board)
        speed = speed_class(int(self.base), int(self.inc), "lichess") or "blitz"
        df = pd.DataFrame([{
            "clock_before": clock_left, "log_clock": math.log1p(max(clock_left, 0)), "clock_frac": clock_left / max(self.base, 1),
            "base_seconds": self.base, "increment": self.inc, "ply": len(self.moves) + 1, "legal_moves": board.legal_moves.count(),
            "nonpawn_material": mat, "n_pieces": n_pieces, "is_blitz": float(speed == "blitz"), "is_chesscom": 0.0,
            "is_black": float(board.turn == chess.BLACK), "prev_capture": prev_capture, "prev_check": prev_check,
            "prev_think": self._last_think, "entropy": entropy, "p_top1": p_top1, "ponder": ponder if ponder is not None else np.nan,
            "speed": speed}])
        s = float(self.think.sample_seconds(df, self.rng)[0])
        return float(min(s, self.opts["MaxThinkFrac"] * clock_left))

    _last_think = 0.0
    last_expected_score = 0.5

    @staticmethod
    def score_cp(expected: float) -> int:
        """Lichess win-probability formula inverted: expected score -> centipawns (clipped)."""
        e = min(max(expected, 1e-3), 1 - 1e-3)
        return int(max(-1500, min(1500, math.log(e / (1 - e)) / 0.00368208)))

    def cmd_go(self, line: str):
        self.ensure_loaded()
        toks = line.split()
        def val(k):
            return float(toks[toks.index(k) + 1]) / 1000 if k in toks else None
        wtime, btime, winc, binc = val("wtime"), val("btime"), val("winc"), val("binc")
        me_white = self.board.turn == chess.WHITE
        clock_left = wtime if me_white else btime
        inc = winc if me_white else binc
        if inc is not None:
            self.inc = inc
        if clock_left is not None and not self.moves:
            self.base = clock_left
        t0 = time.time()
        ply = len(self.moves) + 1
        move, src = None, "book"
        if self.book is not None and ply <= self.opts["BookMaxPly"]:
            move = self.book.sample(self.board.fen(), self.rng, self.opts["BookMinCount"])
        entropy, p1, ponder = 1.0, 0.5, None
        if move is None:
            moves, probs, entropy = self.policy(clock_left)
            p1 = float(probs.max())
            move, src = self.sample(moves, probs), "policy"
        wait = self.think_seconds(clock_left, entropy, p1, ponder)
        if src == "policy":  # lets lichess-bot's resign logic see how the model judges the position
            print(f"info depth 1 score cp {self.score_cp(self.last_expected_score)} pv {move}", flush=True)
        print(f"info string src={src} ply={ply} wait={wait:.1f}s clock={clock_left}", flush=True)
        remaining = wait - (time.time() - t0)
        if remaining > 0:
            time.sleep(remaining)
        self._last_think = wait
        print(f"bestmove {move}", flush=True)

    def run(self):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line == "uci":
                self.cmd_uci()
            elif line == "isready":
                self.ensure_loaded(); print("readyok", flush=True)
            elif line.startswith("setoption"):
                self.cmd_setoption(line)
            elif line == "ucinewgame":
                self.board = chess.Board(); self.moves = []; self._last_think = 0.0
            elif line.startswith("position"):
                self.cmd_position(line)
            elif line.startswith("go"):
                self.cmd_go(line)
            elif line == "stop":
                pass
            elif line == "quit":
                break


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="maia3-23m")
    p.add_argument("--adapter", default="models/adapter_maia3-23m.pt")
    p.add_argument("--think-time", default="models/think_time.pkl")
    p.add_argument("--book", default="data/processed/positions.parquet")
    p.add_argument("--book-splits", default="train,val")
    p.add_argument("--elo", type=int, default=1950)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-wait", action="store_true")
    a = p.parse_args(argv)
    MaelstromEngine(a).run()


if __name__ == "__main__":
    main()

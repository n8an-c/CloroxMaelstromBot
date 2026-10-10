import pathlib

import pandas as pd
import pytest

from maelstrom.data import parse, split
from tests.synth import USER, write_synthetic_pgns

RATING_MAP = {"blitz": {"a": 1.0, "b": 250}, "rapid": {"a": 1.0, "b": 200}, "bullet": {"a": 1.0, "b": 300}}


@pytest.fixture(scope="module")
def parsed(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    paths = write_synthetic_pgns(d)
    games, pos = parse.parse_files(paths, {"lichess": USER, "chesscom": USER}, RATING_MAP, None, show_progress=False)
    return games, pos


def test_time_control_parsing():
    assert parse.parse_time_control("180+2") == (180, 2)
    assert parse.parse_time_control("600") == (600, 0)
    assert parse.parse_time_control("1/86400") == (None, None)
    assert parse.parse_time_control("-") == (None, None)


def test_speed_classes():
    assert parse.speed_class(180, 0, "lichess") == "blitz"
    assert parse.speed_class(180, 2, "lichess") == "blitz"
    assert parse.speed_class(600, 0, "lichess") == "rapid"
    assert parse.speed_class(60, 0, "lichess") == "bullet"
    assert parse.speed_class(120, 1, "chesscom") == "bullet"  # 2|1 is bullet on chess.com
    assert parse.speed_class(600, 0, "chesscom") == "rapid"
    assert parse.speed_class(300, 0, "chesscom") == "blitz"


def test_games_table(parsed):
    games, pos = parsed
    assert len(games) == 22, "daily game must be skipped"
    assert set(games.platform) == {"lichess", "chesscom"}
    assert games.user_color.notna().all()
    assert games.has_clocks.all()
    assert games[games.platform == "lichess"].has_evals.all()
    assert not games[games.platform == "chesscom"].has_evals.any()
    assert games.datetime_utc.is_monotonic_increasing
    cc = games[games.platform == "chesscom"].iloc[0]
    assert cc.user_elo_lichess == cc.user_elo + RATING_MAP[cc.speed]["b"]
    li = games[games.platform == "lichess"].iloc[0]
    assert li.user_elo_lichess == li.user_elo
    assert games.n_plies.sum() == len(pos)


def test_positions_table(parsed):
    games, pos = parsed
    assert pos.groupby("game_id").ply.apply(lambda s: (s.values == range(1, len(s) + 1)).all()).all()
    # roughly half the plies are the user's
    frac = pos.is_user.mean()
    assert 0.4 < frac < 0.6
    # user moves have the right colour
    m = pos.merge(games[["game_id", "user_color"]], on="game_id")
    assert (m.loc[m.is_user, "mover_color"] == m.loc[m.is_user, "user_color"]).all()
    assert (m.loc[~m.is_user, "mover_color"] != m.loc[~m.is_user, "user_color"]).all()
    # clocks and think times
    assert pos.clock_after.notna().all()
    assert (pos.think_time >= 0).all()
    assert pos.think_time.median() > 0.5
    # think time reconstruction: before - after + inc
    recon = pos.clock_before - pos.clock_after + pos.increment
    assert ((pos.think_time - recon.clip(lower=0)).abs() < 1e-6).all()
    # first move of each colour starts from base time
    first = pos[pos.ply <= 2]
    assert (first.clock_before == first.base_seconds).all()
    # evals: eval_before at ply k equals eval_after at ply k-1 within a game (lichess only)
    li = pos[pos.platform == "lichess"].sort_values(["game_id", "ply"])
    shifted = li.groupby("game_id").eval_after_cp.shift(1)
    assert (li.eval_before_cp.fillna(-1) == shifted.fillna(-1)).all()
    # legality / FEN consistency
    import chess
    r = pos.iloc[100]
    assert chess.Move.from_uci(r.move_uci) in chess.Board(r.fen).legal_moves
    assert set(pos.phase) <= {"opening", "middlegame", "endgame"}
    assert (pos.loc[pos.ply <= 20, "phase"] == "opening").all()


def test_split(parsed):
    games, pos = parsed
    games = games.copy()
    games["split"] = split.assign_splits(games, 0.2, 0.25, 1).values
    assert games.split.value_counts()["test"] == round(0.2 * len(games))
    # test games are the most recent ones
    t_min = games.loc[games.split == "test", "datetime_utc"].min()
    assert (games.loc[games.split != "test", "datetime_utc"] < t_min).all()
    pos = pos.merge(games[["game_id", "split"]], on="game_id")
    seen = split.mark_seen_positions(pos)
    # the starting position is always "seen" for white user moves in test
    start = pos[(pos.ply == 1) & pos.is_user & (pos.split == "test")]
    if len(start):
        assert seen.loc[start.index, "seen_fen_in_train"].all()
    # deterministic
    again = split.assign_splits(games, 0.2, 0.25, 1).values
    assert (again == games.split.values).all()

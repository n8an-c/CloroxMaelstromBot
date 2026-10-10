Local data directory (git-ignored).

    data/raw/lichess/<user>.pgn          raw Lichess export (clocks, evals, opening)
    data/raw/chesscom/<user>/YYYY-MM.pgn raw Chess.com monthly archives
    data/processed/games.parquet         one row per game
    data/processed/positions.parquet     one row per ply (both sides, is_user flag)
    data/processed/annotations.parquet   Stockfish eval per (game_id, ply)

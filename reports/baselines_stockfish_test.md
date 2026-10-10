# Stockfish baselines on the test split (150 of your moves)

| model | subset | n | coverage | top-1 | 95% CI | top-3 | NLL | 95% CI |
|---|---|---|---|---|---|---|---|---|
| stockfish depth 4 | all plies | 150 | 100.0% | 40.7% | [36.0%, 46.1%] | 64.0% | 2.588 | [2.439, 2.807] |
| stockfish depth 4 | ply > 10 | 125 | 100.0% | 40.8% | [35.1%, 45.3%] | 64.8% | 2.536 | [2.389, 2.779] |

## by phase

| model | subset | n | coverage | top-1 | 95% CI | top-3 | NLL | 95% CI |
|---|---|---|---|---|---|---|---|---|
| stockfish depth 4 | endgame | 42 | 100.0% | 50.0% | [25.0%, 52.6%] | 71.4% | 2.186 | [2.166, 2.376] |
| stockfish depth 4 | middlegame | 58 | 100.0% | 36.2% | [24.2%, 50.0%] | 62.1% | 2.676 | [2.505, 2.811] |
| stockfish depth 4 | opening | 50 | 100.0% | 38.0% | [26.0%, 52.0%] | 60.0% | 2.822 | [2.579, 3.081] |

## by clock_bucket

| model | subset | n | coverage | top-1 | 95% CI | top-3 | NLL | 95% CI |
|---|---|---|---|---|---|---|---|---|
| stockfish depth 4 | 10-30s | 3 | 100.0% | 33.3% | [0.0%, 50.0%] | 66.7% | 2.089 | [1.296, 3.675] |
| stockfish depth 4 | >30s | 122 | 100.0% | 41.0% | [36.2%, 45.4%] | 64.8% | 2.547 | [2.421, 2.779] |

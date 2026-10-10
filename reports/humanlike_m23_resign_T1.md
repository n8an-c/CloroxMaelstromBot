# Human-likeness: you (test split, 407 games, 12161 moves after ply 10) vs m23_resign_T1 (200 games, 6395 moves)

## By phase

| phase | who | moves | ACPL | best % | inacc/100 | mistake/100 | blunder/100 | median think s | KS cp-loss | KS think | W1 cp-loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| opening | you | 2021 | 24 | 45.6% | 4.2 | 0.5 | 0.2 | 3.0 | 0.021 | 0.127 | 3.8 |
| opening | bot | 1000 | 27 | 44.1% | 3.5 | 0.7 | 1.0 | 4.0 | 0.021 | 0.127 | 3.8 |
| middlegame | you | 6162 | 56 | 41.7% | 8.4 | 2.3 | 1.6 | 5.0 | 0.034 | 0.084 | 10.2 |
| middlegame | bot | 3018 | 67 | 39.5% | 9.6 | 3.1 | 2.3 | 6.0 | 0.034 | 0.084 | 10.2 |
| endgame | you | 3978 | 61 | 40.8% | 5.2 | 1.5 | 1.9 | 2.6 | 0.028 | 0.135 | 6.0 |
| endgame | bot | 2377 | 66 | 42.9% | 4.0 | 1.7 | 1.9 | 3.0 | 0.028 | 0.135 | 6.0 |

## By clock remaining

| clock_bucket | who | moves | ACPL | best % | inacc/100 | mistake/100 | blunder/100 | median think s | KS cp-loss | KS think | W1 cp-loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <10s | you | 776 | 67 | 41.1% | 5.8 | 1.5 | 2.8 | 2.0 | 0.065 | 0.153 | 22.4 |
| <10s | bot | 222 | 80 | 39.2% | 3.2 | 0.9 | 0.9 | 2.0 | 0.065 | 0.153 | 22.4 |
| 10-30s | you | 1588 | 76 | 36.7% | 6.1 | 1.9 | 2.8 | 2.0 | 0.071 | 0.197 | 8.2 |
| 10-30s | bot | 850 | 76 | 43.4% | 3.4 | 1.6 | 2.4 | 3.0 | 0.071 | 0.197 | 8.2 |
| 30s-2m | you | 2764 | 62 | 42.1% | 8.5 | 2.5 | 1.6 | 5.0 | 0.032 | 0.065 | 13.9 |
| 30s-2m | bot | 1442 | 76 | 41.0% | 8.1 | 2.6 | 3.2 | 5.0 | 0.032 | 0.065 | 13.9 |
| >2m | you | 7033 | 42 | 43.4% | 6.1 | 1.5 | 1.0 | 4.4 | 0.028 | 0.095 | 7.7 |
| >2m | bot | 3881 | 50 | 41.3% | 6.9 | 2.3 | 1.5 | 5.0 | 0.028 | 0.095 | 7.7 |

## Stylometry: can a classifier tell you apart?

- move-level AUC: **0.767** (0.5 = indistinguishable)
- game-level AUC (per-game aggregates): **0.854**
- game-level AUC from averaged move-level scores: 0.920
- n: you 407 games / 12161 moves; bot 200 games / 6395 moves

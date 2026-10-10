# Human-likeness: you (test split, 407 games, 12161 moves after ply 10) vs resign_T0.65 (199 games, 5480 moves)

## By phase

| phase | who | moves | ACPL | best % | inacc/100 | mistake/100 | blunder/100 | median think s | KS cp-loss | KS think | W1 cp-loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| opening | you | 2021 | 24 | 45.6% | 4.2 | 0.5 | 0.2 | 3.0 | 0.027 | 0.119 | 3.2 |
| opening | bot | 994 | 22 | 44.4% | 3.2 | 0.4 | 0.3 | 4.0 | 0.027 | 0.119 | 3.2 |
| middlegame | you | 6162 | 56 | 41.7% | 8.4 | 2.3 | 1.6 | 5.0 | 0.033 | 0.111 | 4.8 |
| middlegame | bot | 2906 | 55 | 43.3% | 7.7 | 2.5 | 2.0 | 6.0 | 0.033 | 0.111 | 4.8 |
| endgame | you | 3978 | 61 | 40.8% | 5.2 | 1.5 | 1.9 | 2.6 | 0.018 | 0.232 | 4.3 |
| endgame | bot | 1580 | 59 | 44.6% | 6.1 | 2.3 | 2.4 | 4.0 | 0.018 | 0.232 | 4.3 |

## By clock remaining

| clock_bucket | who | moves | ACPL | best % | inacc/100 | mistake/100 | blunder/100 | median think s | KS cp-loss | KS think | W1 cp-loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <10s | you | 776 | 67 | 41.1% | 5.8 | 1.5 | 2.8 | 2.0 | 0.181 | 0.155 | 20.4 |
| <10s | bot | 43 | 68 | 60.5% | 4.7 | 0.0 | 7.0 | 2.0 | 0.181 | 0.155 | 20.4 |
| 10-30s | you | 1588 | 76 | 36.7% | 6.1 | 1.9 | 2.8 | 2.0 | 0.082 | 0.236 | 8.5 |
| 10-30s | bot | 444 | 70 | 45.9% | 7.4 | 2.0 | 3.8 | 3.0 | 0.082 | 0.236 | 8.5 |
| 30s-2m | you | 2764 | 62 | 42.1% | 8.5 | 2.5 | 1.6 | 5.0 | 0.031 | 0.105 | 5.1 |
| 30s-2m | bot | 1324 | 62 | 43.3% | 8.0 | 2.8 | 2.3 | 6.0 | 0.031 | 0.105 | 5.1 |
| >2m | you | 7033 | 42 | 43.4% | 6.1 | 1.5 | 1.0 | 4.4 | 0.017 | 0.111 | 2.9 |
| >2m | bot | 3669 | 43 | 43.6% | 5.8 | 1.8 | 1.4 | 5.0 | 0.017 | 0.111 | 2.9 |

## Stylometry: can a classifier tell you apart?

- move-level AUC: **0.766** (0.5 = indistinguishable)
- game-level AUC (per-game aggregates): **0.845**
- game-level AUC from averaged move-level scores: 0.924
- n: you 407 games / 12161 moves; bot 199 games / 5480 moves

# Human-likeness: you (test split, 407 games, 12161 moves after ply 10) vs noadapter_T1 (199 games, 8589 moves)

## By phase

| phase | who | moves | ACPL | best % | inacc/100 | mistake/100 | blunder/100 | median think s | KS cp-loss | KS think | W1 cp-loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| opening | you | 2021 | 24 | 45.6% | 4.2 | 0.5 | 0.2 | 3.0 | 0.045 | 0.134 | 9.2 |
| opening | bot | 994 | 33 | 43.9% | 4.2 | 1.5 | 1.3 | 4.0 | 0.045 | 0.134 | 9.2 |
| middlegame | you | 6162 | 56 | 41.7% | 8.4 | 2.3 | 1.6 | 5.0 | 0.040 | 0.102 | 13.9 |
| middlegame | bot | 3359 | 70 | 38.0% | 8.8 | 2.4 | 2.2 | 6.0 | 0.040 | 0.102 | 13.9 |
| endgame | you | 3978 | 61 | 40.8% | 5.2 | 1.5 | 1.9 | 2.6 | 0.050 | 0.127 | 26.4 |
| endgame | bot | 4236 | 85 | 41.1% | 3.3 | 1.4 | 1.9 | 3.0 | 0.050 | 0.127 | 26.4 |

## By clock remaining

| clock_bucket | who | moves | ACPL | best % | inacc/100 | mistake/100 | blunder/100 | median think s | KS cp-loss | KS think | W1 cp-loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <10s | you | 776 | 67 | 41.1% | 5.8 | 1.5 | 2.8 | 2.0 | 0.113 | 0.162 | 37.6 |
| <10s | bot | 518 | 94 | 40.7% | 2.9 | 1.5 | 1.9 | 2.0 | 0.113 | 0.162 | 37.6 |
| 10-30s | you | 1588 | 76 | 36.7% | 6.1 | 1.9 | 2.8 | 2.0 | 0.091 | 0.189 | 14.6 |
| 10-30s | bot | 1373 | 76 | 41.4% | 3.4 | 1.2 | 2.3 | 3.0 | 0.091 | 0.189 | 14.6 |
| 30s-2m | you | 2764 | 62 | 42.1% | 8.5 | 2.5 | 1.6 | 5.0 | 0.032 | 0.078 | 20.9 |
| 30s-2m | bot | 2119 | 80 | 39.3% | 5.4 | 2.0 | 2.4 | 5.0 | 0.032 | 0.078 | 20.9 |
| >2m | you | 7033 | 42 | 43.4% | 6.1 | 1.5 | 1.0 | 4.4 | 0.045 | 0.101 | 24.6 |
| >2m | bot | 4579 | 67 | 40.3% | 6.6 | 2.0 | 1.6 | 5.0 | 0.045 | 0.101 | 24.6 |

## Stylometry: can a classifier tell you apart?

- move-level AUC: **0.795** (0.5 = indistinguishable)
- game-level AUC (per-game aggregates): **0.911**
- game-level AUC from averaged move-level scores: 0.949
- n: you 407 games / 12161 moves; bot 199 games / 8589 moves

# Human-likeness: you (test split, 407 games, 12161 moves after ply 10) vs full_resign (200 games, 5217 moves)

## By phase

| phase | who | moves | ACPL | best % | inacc/100 | mistake/100 | blunder/100 | median think s | KS cp-loss | KS think | W1 cp-loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| opening | you | 2021 | 24 | 45.6% | 4.2 | 0.5 | 0.2 | 3.0 | 0.043 | 0.128 | 8.3 |
| opening | bot | 997 | 32 | 44.1% | 4.0 | 1.3 | 1.1 | 4.0 | 0.043 | 0.128 | 8.3 |
| middlegame | you | 6162 | 56 | 41.7% | 8.4 | 2.3 | 1.6 | 5.0 | 0.045 | 0.116 | 11.6 |
| middlegame | bot | 2850 | 68 | 37.8% | 9.2 | 2.9 | 2.8 | 6.0 | 0.045 | 0.116 | 11.6 |
| endgame | you | 3978 | 61 | 40.8% | 5.2 | 1.5 | 1.9 | 2.6 | 0.048 | 0.218 | 11.0 |
| endgame | bot | 1370 | 70 | 39.1% | 6.1 | 2.6 | 3.5 | 4.0 | 0.048 | 0.218 | 11.0 |

## By clock remaining

| clock_bucket | who | moves | ACPL | best % | inacc/100 | mistake/100 | blunder/100 | median think s | KS cp-loss | KS think | W1 cp-loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <10s | you | 776 | 67 | 41.1% | 5.8 | 1.5 | 2.8 | 2.0 | 0.149 | 0.153 | 18.6 |
| <10s | bot | 87 | 51 | 37.9% | 2.3 | 2.3 | 4.6 | 2.0 | 0.149 | 0.153 | 18.6 |
| 10-30s | you | 1588 | 76 | 36.7% | 6.1 | 1.9 | 2.8 | 2.0 | 0.019 | 0.233 | 10.0 |
| 10-30s | bot | 378 | 76 | 40.2% | 6.6 | 2.1 | 5.6 | 3.0 | 0.019 | 0.233 | 10.0 |
| 30s-2m | you | 2764 | 62 | 42.1% | 8.5 | 2.5 | 1.6 | 5.0 | 0.044 | 0.099 | 10.3 |
| 30s-2m | bot | 1271 | 72 | 38.6% | 7.2 | 2.4 | 3.4 | 6.0 | 0.044 | 0.099 | 10.3 |
| >2m | you | 7033 | 42 | 43.4% | 6.1 | 1.5 | 1.0 | 4.4 | 0.054 | 0.117 | 14.4 |
| >2m | bot | 3481 | 56 | 39.6% | 7.7 | 2.6 | 2.0 | 5.0 | 0.054 | 0.117 | 14.4 |

## Stylometry: can a classifier tell you apart?

- move-level AUC: **0.770** (0.5 = indistinguishable)
- game-level AUC (per-game aggregates): **0.839**
- game-level AUC from averaged move-level scores: 0.928
- n: you 407 games / 12161 moves; bot 200 games / 5217 moves

# Human-likeness: you (test split, 407 games, 12161 moves after ply 10) vs resign_T0.8 (200 games, 5222 moves)

## By phase

| phase | who | moves | ACPL | best % | inacc/100 | mistake/100 | blunder/100 | median think s | KS cp-loss | KS think | W1 cp-loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| opening | you | 2021 | 24 | 45.6% | 4.2 | 0.5 | 0.2 | 3.0 | 0.031 | 0.106 | 5.5 |
| opening | bot | 999 | 29 | 45.1% | 4.0 | 1.3 | 0.8 | 4.0 | 0.031 | 0.106 | 5.5 |
| middlegame | you | 6162 | 56 | 41.7% | 8.4 | 2.3 | 1.6 | 5.0 | 0.011 | 0.105 | 4.2 |
| middlegame | bot | 2826 | 61 | 41.0% | 8.8 | 2.6 | 1.9 | 6.0 | 0.011 | 0.105 | 4.2 |
| endgame | you | 3978 | 61 | 40.8% | 5.2 | 1.5 | 1.9 | 2.6 | 0.027 | 0.216 | 8.8 |
| endgame | bot | 1397 | 57 | 42.9% | 5.9 | 2.4 | 3.2 | 4.0 | 0.027 | 0.216 | 8.8 |

## By clock remaining

| clock_bucket | who | moves | ACPL | best % | inacc/100 | mistake/100 | blunder/100 | median think s | KS cp-loss | KS think | W1 cp-loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <10s | you | 776 | 67 | 41.1% | 5.8 | 1.5 | 2.8 | 2.0 | 0.143 | 0.185 | 22.3 |
| <10s | bot | 93 | 85 | 37.6% | 9.7 | 3.2 | 3.2 | 2.0 | 0.143 | 0.185 | 22.3 |
| 10-30s | you | 1588 | 76 | 36.7% | 6.1 | 1.9 | 2.8 | 2.0 | 0.047 | 0.255 | 14.7 |
| 10-30s | bot | 427 | 67 | 40.7% | 7.5 | 3.3 | 3.0 | 3.0 | 0.047 | 0.255 | 14.7 |
| 30s-2m | you | 2764 | 62 | 42.1% | 8.5 | 2.5 | 1.6 | 5.0 | 0.026 | 0.096 | 5.9 |
| 30s-2m | bot | 1212 | 65 | 43.5% | 6.8 | 2.2 | 3.0 | 6.0 | 0.026 | 0.096 | 5.9 |
| >2m | you | 7033 | 42 | 43.4% | 6.1 | 1.5 | 1.0 | 4.4 | 0.023 | 0.103 | 5.5 |
| >2m | bot | 3490 | 48 | 42.2% | 7.2 | 2.2 | 1.6 | 5.0 | 0.023 | 0.103 | 5.5 |

## Stylometry: can a classifier tell you apart?

- move-level AUC: **0.762** (0.5 = indistinguishable)
- game-level AUC (per-game aggregates): **0.847**
- game-level AUC from averaged move-level scores: 0.925
- n: you 407 games / 12161 moves; bot 200 games / 5222 moves

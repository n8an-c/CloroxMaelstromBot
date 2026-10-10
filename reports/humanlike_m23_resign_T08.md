# Human-likeness: you (test split, 407 games, 12161 moves after ply 10) vs m23_resign_T08 (199 games, 6207 moves)

## By phase

| phase | who | moves | ACPL | best % | inacc/100 | mistake/100 | blunder/100 | median think s | KS cp-loss | KS think | W1 cp-loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| opening | you | 2021 | 24 | 45.6% | 4.2 | 0.5 | 0.2 | 3.0 | 0.033 | 0.125 | 3.3 |
| opening | bot | 995 | 22 | 45.3% | 2.3 | 0.8 | 0.3 | 4.0 | 0.033 | 0.125 | 3.3 |
| middlegame | you | 6162 | 56 | 41.7% | 8.4 | 2.3 | 1.6 | 5.0 | 0.019 | 0.086 | 4.0 |
| middlegame | bot | 2961 | 56 | 40.9% | 7.9 | 2.6 | 1.6 | 6.0 | 0.019 | 0.086 | 4.0 |
| endgame | you | 3978 | 61 | 40.8% | 5.2 | 1.5 | 1.9 | 2.6 | 0.041 | 0.159 | 6.7 |
| endgame | bot | 2251 | 55 | 44.3% | 3.0 | 1.5 | 1.7 | 3.0 | 0.041 | 0.159 | 6.7 |

## By clock remaining

| clock_bucket | who | moves | ACPL | best % | inacc/100 | mistake/100 | blunder/100 | median think s | KS cp-loss | KS think | W1 cp-loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <10s | you | 776 | 67 | 41.1% | 5.8 | 1.5 | 2.8 | 2.0 | 0.054 | 0.172 | 10.8 |
| <10s | bot | 193 | 60 | 39.4% | 5.2 | 0.5 | 0.0 | 2.0 | 0.054 | 0.172 | 10.8 |
| 10-30s | you | 1588 | 76 | 36.7% | 6.1 | 1.9 | 2.8 | 2.0 | 0.074 | 0.195 | 12.5 |
| 10-30s | bot | 629 | 71 | 45.0% | 2.4 | 1.3 | 1.6 | 2.0 | 0.074 | 0.195 | 12.5 |
| 30s-2m | you | 2764 | 62 | 42.1% | 8.5 | 2.5 | 1.6 | 5.0 | 0.054 | 0.062 | 10.7 |
| 30s-2m | bot | 1496 | 51 | 44.1% | 6.0 | 2.4 | 1.7 | 5.0 | 0.054 | 0.062 | 10.7 |
| >2m | you | 7033 | 42 | 43.4% | 6.1 | 1.5 | 1.0 | 4.4 | 0.012 | 0.104 | 4.4 |
| >2m | bot | 3889 | 46 | 42.2% | 5.4 | 1.9 | 1.4 | 5.0 | 0.012 | 0.104 | 4.4 |

## Stylometry: can a classifier tell you apart?

- move-level AUC: **0.759** (0.5 = indistinguishable)
- game-level AUC (per-game aggregates): **0.831**
- game-level AUC from averaged move-level scores: 0.904
- n: you 407 games / 12161 moves; bot 199 games / 6207 moves

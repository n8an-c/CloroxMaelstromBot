# Human-likeness: you (test split, 407 games, 12161 moves after ply 10) vs full_T1_nobook (199 games, 5249 moves)

## By phase

| phase | who | moves | ACPL | best % | inacc/100 | mistake/100 | blunder/100 | median think s | KS cp-loss | KS think | W1 cp-loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| opening | you | 2021 | 24 | 45.6% | 4.2 | 0.5 | 0.2 | 3.0 | 0.051 | 0.116 | 7.7 |
| opening | bot | 995 | 31 | 42.3% | 5.1 | 1.1 | 0.8 | 4.0 | 0.051 | 0.116 | 7.7 |
| middlegame | you | 6162 | 56 | 41.7% | 8.4 | 2.3 | 1.6 | 5.0 | 0.040 | 0.105 | 6.4 |
| middlegame | bot | 2830 | 63 | 38.5% | 9.4 | 2.6 | 2.4 | 6.0 | 0.040 | 0.105 | 6.4 |
| endgame | you | 3978 | 61 | 40.8% | 5.2 | 1.5 | 1.9 | 2.6 | 0.046 | 0.236 | 10.9 |
| endgame | bot | 1424 | 71 | 39.7% | 7.4 | 2.5 | 3.4 | 4.0 | 0.046 | 0.236 | 10.9 |

## By clock remaining

| clock_bucket | who | moves | ACPL | best % | inacc/100 | mistake/100 | blunder/100 | median think s | KS cp-loss | KS think | W1 cp-loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <10s | you | 776 | 67 | 41.1% | 5.8 | 1.5 | 2.8 | 2.0 | 0.080 | 0.153 | 39.0 |
| <10s | bot | 115 | 101 | 45.2% | 3.5 | 4.3 | 7.0 | 2.0 | 0.080 | 0.153 | 39.0 |
| 10-30s | you | 1588 | 76 | 36.7% | 6.1 | 1.9 | 2.8 | 2.0 | 0.059 | 0.284 | 11.7 |
| 10-30s | bot | 422 | 86 | 36.3% | 7.6 | 2.8 | 4.0 | 3.0 | 0.059 | 0.284 | 11.7 |
| 30s-2m | you | 2764 | 62 | 42.1% | 8.5 | 2.5 | 1.6 | 5.0 | 0.049 | 0.113 | 6.5 |
| 30s-2m | bot | 1187 | 66 | 39.8% | 9.7 | 2.9 | 2.3 | 6.0 | 0.049 | 0.113 | 6.5 |
| >2m | you | 7033 | 42 | 43.4% | 6.1 | 1.5 | 1.0 | 4.4 | 0.042 | 0.102 | 10.0 |
| >2m | bot | 3525 | 52 | 39.7% | 7.7 | 2.0 | 2.0 | 5.0 | 0.042 | 0.102 | 10.0 |

## Stylometry: can a classifier tell you apart?

- move-level AUC: **0.754** (0.5 = indistinguishable)
- game-level AUC (per-game aggregates): **0.823**
- game-level AUC from averaged move-level scores: 0.913
- n: you 407 games / 12161 moves; bot 199 games / 5249 moves

# Human-likeness: you (test split, 407 games, 12161 moves after ply 10) vs adapter_argmax (199 games, 7910 moves)

## By phase

| phase | who | moves | ACPL | best % | inacc/100 | mistake/100 | blunder/100 | median think s | KS cp-loss | KS think | W1 cp-loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| opening | you | 2021 | 24 | 45.6% | 4.2 | 0.5 | 0.2 | 3.0 | 0.046 | 0.126 | 5.0 |
| opening | bot | 995 | 20 | 48.2% | 2.8 | 0.5 | 0.3 | 4.0 | 0.046 | 0.126 | 5.0 |
| middlegame | you | 6162 | 56 | 41.7% | 8.4 | 2.3 | 1.6 | 5.0 | 0.094 | 0.108 | 15.1 |
| middlegame | bot | 3046 | 42 | 47.4% | 5.5 | 1.3 | 1.0 | 6.0 | 0.094 | 0.108 | 15.1 |
| endgame | you | 3978 | 61 | 40.8% | 5.2 | 1.5 | 1.9 | 2.6 | 0.115 | 0.140 | 15.8 |
| endgame | bot | 3869 | 48 | 44.2% | 2.6 | 0.9 | 0.8 | 3.0 | 0.115 | 0.140 | 15.8 |

## By clock remaining

| clock_bucket | who | moves | ACPL | best % | inacc/100 | mistake/100 | blunder/100 | median think s | KS cp-loss | KS think | W1 cp-loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <10s | you | 776 | 67 | 41.1% | 5.8 | 1.5 | 2.8 | 2.0 | 0.244 | 0.153 | 30.7 |
| <10s | bot | 303 | 40 | 46.5% | 1.3 | 0.7 | 1.0 | 2.0 | 0.244 | 0.153 | 30.7 |
| 10-30s | you | 1588 | 76 | 36.7% | 6.1 | 1.9 | 2.8 | 2.0 | 0.161 | 0.167 | 25.6 |
| 10-30s | bot | 1183 | 53 | 43.1% | 2.6 | 0.6 | 0.8 | 2.0 | 0.161 | 0.167 | 25.6 |
| 30s-2m | you | 2764 | 62 | 42.1% | 8.5 | 2.5 | 1.6 | 5.0 | 0.133 | 0.094 | 19.2 |
| 30s-2m | bot | 2308 | 46 | 48.2% | 3.9 | 1.3 | 1.0 | 5.0 | 0.133 | 0.094 | 19.2 |
| >2m | you | 7033 | 42 | 43.4% | 6.1 | 1.5 | 1.0 | 4.4 | 0.063 | 0.112 | 9.7 |
| >2m | bot | 4116 | 37 | 45.5% | 4.1 | 1.0 | 0.7 | 5.0 | 0.063 | 0.112 | 9.7 |

## Stylometry: can a classifier tell you apart?

- move-level AUC: **0.791** (0.5 = indistinguishable)
- game-level AUC (per-game aggregates): **0.904**
- game-level AUC from averaged move-level scores: 0.946
- n: you 407 games / 12161 moves; bot 199 games / 7910 moves

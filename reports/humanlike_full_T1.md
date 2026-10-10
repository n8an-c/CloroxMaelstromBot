# Human-likeness: you (test split, 407 games, 12161 moves after ply 10) vs full_T1 (200 games, 8618 moves)

## By phase

| phase | who | moves | ACPL | best % | inacc/100 | mistake/100 | blunder/100 | median think s | KS cp-loss | KS think | W1 cp-loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| opening | you | 2021 | 24 | 45.6% | 4.2 | 0.5 | 0.2 | 3.0 | 0.054 | 0.124 | 8.7 |
| opening | bot | 1000 | 32 | 41.3% | 6.0 | 1.3 | 0.8 | 4.0 | 0.054 | 0.124 | 8.7 |
| middlegame | you | 6162 | 56 | 41.7% | 8.4 | 2.3 | 1.6 | 5.0 | 0.028 | 0.102 | 12.6 |
| middlegame | bot | 3151 | 69 | 38.6% | 8.2 | 3.0 | 2.0 | 6.0 | 0.028 | 0.102 | 12.6 |
| endgame | you | 3978 | 61 | 40.8% | 5.2 | 1.5 | 1.9 | 2.6 | 0.042 | 0.130 | 13.4 |
| endgame | bot | 4467 | 68 | 40.1% | 2.9 | 1.1 | 1.6 | 3.0 | 0.042 | 0.130 | 13.4 |

## By clock remaining

| clock_bucket | who | moves | ACPL | best % | inacc/100 | mistake/100 | blunder/100 | median think s | KS cp-loss | KS think | W1 cp-loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <10s | you | 776 | 67 | 41.1% | 5.8 | 1.5 | 2.8 | 2.0 | 0.087 | 0.153 | 26.9 |
| <10s | bot | 323 | 82 | 37.5% | 3.1 | 0.3 | 2.2 | 2.0 | 0.087 | 0.153 | 26.9 |
| 10-30s | you | 1588 | 76 | 36.7% | 6.1 | 1.9 | 2.8 | 2.0 | 0.082 | 0.167 | 15.9 |
| 10-30s | bot | 1609 | 70 | 39.3% | 2.8 | 0.8 | 1.6 | 2.0 | 0.082 | 0.167 | 15.9 |
| 30s-2m | you | 2764 | 62 | 42.1% | 8.5 | 2.5 | 1.6 | 5.0 | 0.040 | 0.078 | 15.7 |
| 30s-2m | bot | 2077 | 72 | 40.3% | 5.2 | 2.6 | 2.2 | 5.0 | 0.040 | 0.078 | 15.7 |
| >2m | you | 7033 | 42 | 43.4% | 6.1 | 1.5 | 1.0 | 4.4 | 0.029 | 0.084 | 15.1 |
| >2m | bot | 4609 | 57 | 39.8% | 6.2 | 1.9 | 1.4 | 5.0 | 0.029 | 0.084 | 15.1 |

## Stylometry: can a classifier tell you apart?

- move-level AUC: **0.801** (0.5 = indistinguishable)
- game-level AUC (per-game aggregates): **0.899**
- game-level AUC from averaged move-level scores: 0.939
- n: you 407 games / 12161 moves; bot 200 games / 8618 moves

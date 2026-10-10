# Human-likeness: you (test split, 407 games, 12161 moves after ply 10) vs bot(adapter, T=1, book, think-time) (193 games, 8030 moves)

## By phase

| phase | who | moves | ACPL | best % | inacc/100 | mistake/100 | blunder/100 | median think s | KS cp-loss | KS think | W1 cp-loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| opening | you | 2021 | 24 | 45.6% | 4.2 | 0.5 | 0.2 | 3.0 | 0.053 | 0.087 | 3.2 |
| opening | bot | 965 | 27 | 42.0% | 3.3 | 0.7 | 0.6 | 3.4 | 0.053 | 0.087 | 3.2 |
| middlegame | you | 6162 | 56 | 41.7% | 8.4 | 2.3 | 1.6 | 5.0 | 0.036 | 0.071 | 14.8 |
| middlegame | bot | 3186 | 71 | 38.6% | 9.1 | 2.7 | 2.4 | 5.9 | 0.036 | 0.071 | 14.8 |
| endgame | you | 3978 | 61 | 40.8% | 5.2 | 1.5 | 1.9 | 2.6 | 0.063 | 0.084 | 17.2 |
| endgame | bot | 3879 | 72 | 41.7% | 3.3 | 0.9 | 1.5 | 2.8 | 0.063 | 0.084 | 17.2 |

## By clock remaining

| clock_bucket | who | moves | ACPL | best % | inacc/100 | mistake/100 | blunder/100 | median think s | KS cp-loss | KS think | W1 cp-loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <10s | you | 776 | 67 | 41.1% | 5.8 | 1.5 | 2.8 | 2.0 | 0.229 | 0.512 | 31.5 |
| <10s | bot | 31 | 47 | 45.2% | 0.0 | 0.0 | 6.5 | 0.6 | 0.229 | 0.512 | 31.5 |
| 10-30s | you | 1588 | 76 | 36.7% | 6.1 | 1.9 | 2.8 | 2.0 | 0.099 | 0.103 | 28.0 |
| 10-30s | bot | 1164 | 86 | 41.5% | 3.2 | 0.9 | 1.8 | 2.0 | 0.099 | 0.103 | 28.0 |
| 30s-2m | you | 2764 | 62 | 42.1% | 8.5 | 2.5 | 1.6 | 5.0 | 0.052 | 0.120 | 17.2 |
| 30s-2m | bot | 2204 | 71 | 39.1% | 5.7 | 1.8 | 1.9 | 4.0 | 0.052 | 0.120 | 17.2 |
| >2m | you | 7033 | 42 | 43.4% | 6.1 | 1.5 | 1.0 | 4.4 | 0.033 | 0.062 | 17.1 |
| >2m | bot | 4631 | 59 | 40.9% | 6.2 | 1.7 | 1.7 | 4.6 | 0.033 | 0.062 | 17.1 |

## Stylometry: can a classifier tell you apart?

- move-level AUC: **0.862** (0.5 = indistinguishable)
- game-level AUC (per-game aggregates): **0.883**
- game-level AUC from averaged move-level scores: 0.977
- n: you 407 games / 12161 moves; bot 193 games / 8030 moves

# Book / kNN baselines on the test split (14201 of your moves, 408 games)

Coverage = share of positions where the model makes any prediction; uncovered positions count as misses.

| model | subset | n | coverage | top-1 | 95% CI | top-3 | NLL | 95% CI |
|---|---|---|---|---|---|---|---|---|
| book(ply<=20) | all plies | 14201 | 14.9% | 10.3% | [9.6%, 11.2%] | 13.6% | 1.504 | [1.453, 1.560] |
| book(ply<=20) | ply > 10 | 12161 | 3.2% | 2.5% | [2.1%, 2.9%] | 2.7% | 2.207 | [2.096, 2.329] |
| knn(exact FEN) | all plies | 14201 | 15.0% | 10.4% | [9.7%, 11.3%] | 13.7% | 1.511 | [1.461, 1.567] |
| knn(exact FEN) | ply > 10 | 12161 | 3.3% | 2.6% | [2.2%, 3.0%] | 2.7% | 2.221 | [2.118, 2.340] |

## by phase (ply > 10 except phase)

| model | subset | n | coverage | top-1 | 95% CI | top-3 | NLL | 95% CI |
|---|---|---|---|---|---|---|---|---|
| book(ply<=20) | endgame | 3978 | 0.0% | 0.0% | [0.0%, 0.0%] | 0.0% | nan | [nan, nan] |
| book(ply<=20) | middlegame | 6162 | 0.0% | 0.0% | [0.0%, 0.0%] | 0.0% | nan | [nan, nan] |
| book(ply<=20) | opening | 4061 | 52.1% | 36.1% | [33.9%, 38.4%] | 47.5% | 1.504 | [1.453, 1.560] |
| knn(exact FEN) | endgame | 3978 | 0.0% | 0.0% | [0.0%, 0.0%] | 0.0% | nan | [nan, nan] |
| knn(exact FEN) | middlegame | 6162 | 0.2% | 0.2% | [0.0%, 0.3%] | 0.2% | 2.661 | [2.228, 3.003] |
| knn(exact FEN) | opening | 4061 | 52.1% | 36.1% | [33.9%, 38.4%] | 47.5% | 1.504 | [1.453, 1.560] |

## by clock_bucket (ply > 10 except phase)

| model | subset | n | coverage | top-1 | 95% CI | top-3 | NLL | 95% CI |
|---|---|---|---|---|---|---|---|---|
| book(ply<=20) | <10s | 776 | 0.0% | 0.0% | [0.0%, 0.0%] | 0.0% | nan | [nan, nan] |
| book(ply<=20) | 10-30s | 1588 | 0.0% | 0.0% | [0.0%, 0.0%] | 0.0% | nan | [nan, nan] |
| book(ply<=20) | >30s | 9797 | 4.0% | 3.1% | [2.6%, 3.6%] | 3.3% | 2.207 | [2.096, 2.329] |
| knn(exact FEN) | <10s | 776 | 0.0% | 0.0% | [0.0%, 0.0%] | 0.0% | nan | [nan, nan] |
| knn(exact FEN) | 10-30s | 1588 | 0.0% | 0.0% | [0.0%, 0.0%] | 0.0% | nan | [nan, nan] |
| knn(exact FEN) | >30s | 9797 | 4.2% | 3.2% | [2.7%, 3.8%] | 3.4% | 2.221 | [2.118, 2.340] |

## by seen (ply > 10 except phase)

| model | subset | n | coverage | top-1 | 95% CI | top-3 | NLL | 95% CI |
|---|---|---|---|---|---|---|---|---|
| book(ply<=20) | seen | 365 | 97.3% | 76.7% | [72.3%, 80.9%] | 81.6% | 2.149 | [2.035, 2.269] |
| book(ply<=20) | unseen | 11796 | 0.3% | 0.2% | [0.1%, 0.3%] | 0.2% | 2.733 | [2.476, 2.967] |
| knn(exact FEN) | seen | 365 | 100.0% | 78.6% | [73.9%, 82.7%] | 83.6% | 2.164 | [2.055, 2.281] |
| knn(exact FEN) | unseen | 11796 | 0.4% | 0.2% | [0.1%, 0.3%] | 0.2% | 2.725 | [2.490, 2.968] |

# Think-time model on the test split (14201 of your moves)

- Pearson r (log s): **0.621**, Spearman: 0.629, MAE: 4.3s, KS(sampled vs real): 0.074

| speed | clock | n | real median s | predicted median s | sampled median s | KS |
|---|---|---|---|---|---|---|
| blitz | <10s | 734 | 2.0 | 1.9 | 1.7 | 0.177 |
| blitz | 10-30s | 1501 | 2.0 | 2.3 | 2.2 | 0.090 |
| blitz | 30s-2m | 2571 | 5.0 | 5.7 | 5.1 | 0.051 |
| blitz | >2m | 5020 | 3.0 | 3.3 | 2.9 | 0.098 |
| rapid | <10s | 42 | 1.5 | 1.1 | 0.7 | 0.333 |
| rapid | 10-30s | 87 | 1.7 | 2.0 | 1.8 | 0.149 |
| rapid | 30s-2m | 193 | 3.0 | 3.9 | 3.8 | 0.119 |
| rapid | >2m | 4053 | 4.2 | 5.1 | 4.8 | 0.062 |

# Think-time model on the test split (14201 of your moves)

- Pearson r (log s): **0.621**, Spearman: 0.629, MAE: 4.3s, KS(sampled vs real): 0.070

| speed | clock | n | real median s | predicted median s | sampled median s | KS |
|---|---|---|---|---|---|---|
| blitz | <10s | 734 | 2.0 | 1.9 | 1.8 | 0.168 |
| blitz | 10-30s | 1501 | 2.0 | 2.3 | 2.3 | 0.110 |
| blitz | 30s-2m | 2571 | 5.0 | 5.7 | 5.3 | 0.059 |
| blitz | >2m | 5020 | 3.0 | 3.3 | 3.2 | 0.091 |
| rapid | <10s | 42 | 1.5 | 1.1 | 1.0 | 0.238 |
| rapid | 10-30s | 87 | 1.7 | 2.0 | 1.9 | 0.115 |
| rapid | 30s-2m | 193 | 3.0 | 3.9 | 4.1 | 0.124 |
| rapid | >2m | 4053 | 4.2 | 5.1 | 5.0 | 0.062 |

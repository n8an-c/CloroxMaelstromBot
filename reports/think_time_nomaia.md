# Think-time model on the test split (14201 of your moves)

- Pearson r (log s): **0.567**, Spearman: 0.579, MAE: 4.4s, KS(sampled vs real): 0.072

| speed | clock | n | real median s | predicted median s | sampled median s | KS |
|---|---|---|---|---|---|---|
| blitz | <10s | 734 | 2.0 | 1.9 | 1.8 | 0.173 |
| blitz | 10-30s | 1501 | 2.0 | 2.3 | 2.3 | 0.113 |
| blitz | 30s-2m | 2571 | 5.0 | 5.7 | 5.1 | 0.055 |
| blitz | >2m | 5020 | 3.0 | 3.3 | 3.4 | 0.100 |
| rapid | <10s | 42 | 1.5 | 1.2 | 1.1 | 0.190 |
| rapid | 10-30s | 87 | 1.7 | 2.0 | 1.9 | 0.115 |
| rapid | 30s-2m | 193 | 3.0 | 3.7 | 4.1 | 0.109 |
| rapid | >2m | 4053 | 4.2 | 5.0 | 4.9 | 0.060 |

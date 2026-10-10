# Adapter ablations on the temporal test split (14201 of your moves)

| variant | train games | val NLL (random val) | test top-1 | test top-3 | test NLL | test NLL ply>10 | test top-1 ply>10 |
|---|---|---|---|---|---|---|---|
| adapter(maia3-5m) | 3301 (all) | 1.1275 | 59.4% | 87.1% | 1.191 | 1.222 | 58.0% |
| adapter_g1000 | 1000 most recent | 1.2039 | 59.8% | 86.8% | 1.196 | 1.228 | 58.0% |
| adapter_g2000 | 2000 most recent | 1.1716 | 59.9% | 86.9% | 1.200 | 1.220 | 58.5% |
| adapter_g250 | 250 most recent | 1.2594 | 59.1% | 85.5% | 1.234 | 1.238 | 57.9% |
| adapter_noclock | 3301 (all) | 1.1284 | 59.6% | 87.3% | 1.195 | 1.224 | 58.2% |
| maia3-5m | - | 1.2594 (base) | 58.8% | 85.4% | 1.237 | 1.240 | 58.0% |

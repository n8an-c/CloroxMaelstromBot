# Results: how much of one player's chess style is learnable from ~4,000 games?

Player: a ~2000 Lichess-rated amateur (blitz 1943, rapid 2012 on Lichess; concurrent Chess.com
games mapped to the same scale). Data in scope: blitz and rapid games from 1 Feb 2022 to
10 Oct 2026 on both sites.

| split | games | the player's moves | period |
|---|---|---|---|
| train | 3,301 | 116,608 | 2022-02-09 to 2026-07-25 |
| val (random 10% of the rest) | 367 | 12,893 | interleaved with train |
| test (most recent 10%, temporal) | 408 | 14,201 | 2026-07-25 to 2026-10-10 |

All numbers below are on the **temporal test split** unless stated. 95% CIs are bootstrap over games.
"ply > 10" excludes the opening, the convention in the Maia papers.

## 1. Move prediction: can a model predict *my* moves?

| model | top-1 | top-1 (ply > 10) | top-3 | NLL | NLL (ply > 10) |
|---|---|---|---|---|---|
| Opening book (own moves, ply <= 20; 15% coverage) | 10.3% | 2.5% | 13.6% | - | - |
| Exact-position lookup (kNN, 15% coverage) | 10.4% | 2.6% | 13.7% | - | - |
| Stockfish 17.1 depth 2 (MultiPV softmax) | 38.3% | 38.3% | 66.6% | 2.532 | 2.501 |
| Stockfish depth 8 | 40.8% | 39.9% | 69.2% | 2.457 | 2.436 |
| Stockfish Skill Level 1 / 5 / 10 (depth 8) | 27.0% / 29.8% / 33.8% | - | ~68.6% | 2.460 | - |
| **Maia-3 5M, frozen** (actual Elo inputs) | 58.8% [58.1, 59.7] | 58.0% | 85.4% | 1.237 [1.218, 1.258] | 1.240 |
| Maia-3 23M, frozen | 60.7% [59.9, 61.4] | 59.9% | 86.7% | 1.184 [1.161, 1.206] | 1.180 |
| **Maia-3 5M + you-adapter** (all 3,301 train games) | **59.4%** [58.6, 60.2] | 58.0% | **87.1%** | **1.191** [1.168, 1.215] | 1.222 |
| Maia-3 5M + adapter, 2,000 most recent games | 59.9% | 58.5% | 86.9% | 1.200 | 1.220 |
| **Maia-3 23M + you-adapter** (final model) | **61.0%** [60.1, 61.8] | **60.2%** | **87.9%** | **1.145** [1.121, 1.171] | **1.168** |

Where the opening book covers a position (positions seen in training, 365 after ply 10),
it predicts the move 77-79% of the time; the adapter reaches 71% top-1 / 0.80 NLL on those
same positions vs 61.6% / 1.00 for frozen Maia-3.

**Reading:** Maia-3 already predicts this player far better than any engine-based baseline
(58.8% vs 41%). Scaling the frozen base from 5M to 23M (+1.9 pp, -4.3% NLL) is worth more than
personalising the 5M model (+0.6 pp, -3.7% NLL); the adapter on 23M combines both: 61.0% / 1.145, i.e. +2.2 pp and -7.4% NLL over the frozen 5M
baseline, of which the adapter contributes +0.3 pp and -3.3% NLL on top of frozen 23M. Personalisation on 4k games adds a small, real gain: -3.7% NLL and +0.6 pp
top-1 overall, -1.5% NLL after the opening. The gain is concentrated in the opening and in
previously seen positions, i.e. the repertoire. Middlegame/endgame top-1 is flat, with a
small NLL improvement. This is in line with the literature (Maia4All: +1-2 pp at 2,500 games).

### Ablations (see `reports/adapter_ablations_test.md`)

- **Data-size curve** (most recent N train games): 250 -> no gain; 1,000 -> NLL 1.196;
  2,000 -> 1.200 (best ply>10 top-1, 58.5%); 3,301 -> 1.191. Most of the gain is there by 1,000 games.
- **Clock features:** no measurable effect (1.195 vs 1.191). The base model does not know the
  clock and the adapter cannot learn time-pressure behaviour from 143k moves.
- **Validation choice:** the random val split overstates gains (val NLL 1.259 -> 1.128, -10%)
  because val games are interleaved in time with train games. Holding out the 330 most recent
  train games as validation loses the *current* opening repertoire and the resulting model is
  worse over all plies on test (1.245) while equal after ply 10. **Recency matters for openings.**
- **Training length:** 60 epochs = 30 epochs (early stopping on val NLL).

### Where the frozen model fails: my mistakes

By Stockfish move quality (ply > 10, Maia-3 5M): top-1 is 70% on my best moves, 39% on
"good" moves, 29% on inaccuracies, 31% on mistakes and **20% on blunders** (top-3 66%).
The adapter does not change this. Predicting *which* mistake a specific person makes is the
hard part; argmax play would make the bot stronger and more consistent than the person,
which is why the bot samples from the policy instead.

## 2. Think time

Gradient-boosted regressor on log think time, inputs known before the move (clock, time
control, ply, material, opponent's last move, own previous think, Maia-3 entropy / top-1
probability / ponder head). Test split, 14,201 moves:

| model | Pearson r (log s) | Spearman | MAE | KS(sampled vs real) |
|---|---|---|---|---|
| clock + position features only | 0.567 | 0.579 | 4.4 s | 0.072 |
| + Maia-3 entropy / ponder | **0.621** | 0.629 | 4.3 s | 0.074 |

For scale, Allie (355M params, full game history) reports r = 0.70; ChessMimic r = 0.41.
Sampled think times are drawn from the model mean plus an empirical residual per
(speed, clock bucket), which keeps the premove spike and the heavy tail.

## 3. Human-likeness of the deployed bot (self-play vs Maia-3 at my opponents' ratings)

200 games from the first 10 plies of my test games, same colour as I had, simulated clocks.
Scored with the same Stockfish annotation as my own games. See `reports/humanlike_*.md`.

| bot variant | score vs Maia-3 | middlegame ACPL (me: 56) | blunders/100 mid (me: 1.6) | stylometry AUC move / game |
|---|---|---|---|---|
| first run (Gaussian think-time residuals, continuous clocks) | 0.48 | 71 | 2.4 | 0.862 / 0.883 |
| full bot (adapter, T=1, book, empirical think-time, whole-second clocks) | 0.54 | 69 | 2.0 | 0.801 / 0.899 |
| no adapter (plain Maia-3 5M, T=1, book, think-time) | 0.485 | 70 | 2.2 | 0.795 / 0.911 |
| adapter, argmax (T=0), book, think-time | 0.802 | 42 | 1.0 | 0.791 / 0.904 |
| full bot + calibrated resignation (T=1) | 0.475 | 68 | 2.8 | **0.770 / 0.839** |
| full bot without opening book (T=1) | 0.478 | 63 | 2.4 | 0.754 / 0.823 |

In the first run the classifier separated us almost entirely on **timing** (timing-only
AUC 0.862, move-quality-only 0.558): 59% of my recorded think times are whole seconds
(Lichess clock resolution) vs 3% of the bot's, and the bot played under 1 s twice as often.
That is a measurement artifact of the simulation, fixed in the later runs. Move quality
alone is close to indistinguishable at the move level.

With whole-second clocks and empirical think-time residuals (full bot row), the feature-group
ablation reads: all 0.801, timing only 0.697, clock only 0.639, position/move only 0.614,
move quality only 0.568. The remaining separation comes from *game length and clock use*:
the simulated games never resign, so the bot plays 43 moves per game after ply 10 vs my 30
(median ply 55 vs 42) and runs its clock lower (median 53% left vs 67%). A resignation rule
calibrated on my own games is the next fix; move quality itself is nearly indistinguishable.

With resignation (both sides resign below a Maia-3 expected score of 0.08 with probability 0.75 per move, matching my
75% resignation rate at ~7% win probability), games average 62 plies vs my 70 and the AUC drops to 0.770 / 0.839. What is
left is mostly *strength*: at temperature 1 the sampled bot is a little weaker and more erratic than me (ACPL 68 vs 56,
endgame blunders 3.5 vs 1.9 per 100), so the temperature is tuned next to match my error rate.

Argmax play (T=0) makes the bot *stronger* than me (middlegame ACPL 42 vs 56, blunders 1.0 vs 1.6 per 100,
best-move rate 47% vs 42%), which is exactly why the bot samples from the policy.

## 4. What I would do next

- Per-square hidden states for the candidate moves (instead of the pooled state) in the
  adapter, Matilda-style; engine features for the candidates.
- Recency-weighted training and book; a blinded human test with clubmates; live games on
  the BOT account and a stylometry test on *real* Lichess games (no simulation artifacts).
- Maia-2 (MIT) as the base for a closed-source deployment, if ever needed.

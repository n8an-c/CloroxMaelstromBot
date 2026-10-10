# Building a "Plays Like Me" Chess Bot: Approaches, Evidence, and a Budget Plan

The best plan is to use a strong, pretrained human-move model as a frozen base (Maia-3, or Maia-2 if you want permissive licensing and existing training code), and then train a small personalization layer on your ~4,000 games. Add a personal opening book, sampling instead of argmax, and a clock model on top. Don't train from scratch, and don't fine-tune an LLM. Training this way costs close to $0 on Kaggle or Colab, and the bot runs on your ThinkPad's CPU.

## TL;DR
- **4,000 games is enough, but only with the right method.** Plain full fine-tuning of Maia-1 *hurt* accuracy at 1,000 games (49.7% vs 52.7% for the nearest Maia) and gave +1.8 pp at 5,000 games. Newer methods on top of Maia-2 (Maia4All) rose from 51.4% to 53.2% with only 20 games, which Tang et al. (arXiv 2507.21488) call 250 times more data-efficient than prior methods. Expect your personalized model to beat the generic baseline by roughly 1–4 pp top-1 (around 53–58%), with a clearly lower perplexity. The bigger gain in how it *feels* comes from sampling, your opening book, and clock behavior.
- **Recommended stack:** frozen Maia-3-5M/23M (or Maia-2), plus a residual "you-adapter" conditioned on a player embedding, time control and clock, plus your opening book. Sample moves from the policy, and add a think-time model plus time-pressure noise. Deploy it through lichess-bot on a separate BOT account. Stockfish-with-weakness and kNN/opening-book bots are useful baselines, not the product.
- **Cost and resume value:** about $0–$30 in total (Kaggle's docs say the free GPU quota resets weekly at 30 hours, sometimes more; a RunPod Community Cloud RTX 4090 is $0.34/hr, billed per second, if you want to rent). What makes it stand out on a resume: a rigorous held-out evaluation against Maia-2, Maia-3 and Stockfish, broken down by phase, move quality and time pressure; a stylometry "can a classifier tell us apart?" test; and a live Lichess bot plus a write-up.

## Key Findings

### 1. The research landscape (what's been shown, and what's available)

| Work | What it is | Reported human move-matching | Per-player data | Code / weights / license |
|---|---|---|---|---|
| **Maia** (McIlroy-Young et al., KDD 2020) | 9 Leela-style ResNets, one per rating band (1100–1900) [arxiv](https://arxiv.org/html/2605.19091v1) | Up to >52% within band; 46% at worst. Depth-limited Stockfish: 33–41%. [microsoft](https://www.microsoft.com/en-us/research/blog/the-human-side-of-ai-for-chess/) Predicts the exact human blunder >25% of the time [arxiv](https://arxiv.org/pdf/2006.01855) | n/a (population) | CSSLab/maia-chess; lc0-format weights [github](https://github.com/CSSLab/maia-chess) |
| **Maia-Individual** (McIlroy-Young et al., KDD 2022) | Transfer learning from Maia-1900 to single players [arxiv](https://arxiv.org/html/2008.10086v3) | +4–5 pp on average over nearest Maia for prolific players. [toronto](https://www.cs.toronto.edu/~ashton/pubs/maia-individual-kdd2022.pdf) By games: 1k → 49.7% (vs 52.7% base), 5k → 55.0% (vs 53.2%), 10k → 55.6%, 40k → 58.0% [arxiv](https://arxiv.org/html/2008.10086v3) | Gains only from about 5,000 games [arxiv](https://arxiv.org/pdf/2507.21488) [arxiv](https://arxiv.org/html/2008.10086v3) | CSSLab/maia-individual, AGPL-3.0, full per-player training pipeline (old TF/lc0 stack). No player models released, for privacy reasons [github](https://github.com/CSSLab/maia-individual) |
| **Behavioral stylometry** (NeurIPS 2021) | Transformer embeddings that identify players | Identifies a player among thousands with 98% accuracy from 100 games [neurips](https://proceedings.neurips.cc/paper/2021/hash/ccf8111910291ba472b385e9c5f59099-Abstract.html) | 100 games for reference | Paper |
| **Maia-2** (Tang et al., NeurIPS 2024) | One 23.3M-param model with skill-aware attention, conditioned on both players' Elo [arxiv +2](https://arxiv.org/pdf/2608.05206) | Almost 2 full pp better than original Maia (Tang et al., NeurIPS 2024); ~52–53% average | n/a | CSSLab/maia2, **MIT**. `pip install maia2`, Rapid and Blitz weights, training presets included [github](https://github.com/CSSLab/maia2) |
| **Allie** (Zhang et al., ICLR 2025) | 355M GPT-2-style decoder over move sequences; predicts move, think time and value; time-adaptive MCTS [arxiv](https://arxiv.org/pdf/2410.03893) [tiptreesystems](https://lacuna.tiptreesystems.com/work/human-aligned-chess-with-a-bit-of-search/wrk_33f9d3c37a5e7693e582901240439d97) | 55.7% (vs Maia* 51.6%, GPT-3.5 53.7%). Calibrated to within 49 Elo on average across 7,483 games with humans. Think-time correlation r≈0.70 [arxiv +3](https://arxiv.org/html/2410.03893) | n/a | Paper. Uses full game history; hard to personalize [arxiv](https://arxiv.org/pdf/2507.21488) |
| **Maia4All** (Tang et al., arXiv 2507.21488, 2025) | Maia-2 enriched with prototype players, then a per-player embedding [arxiv](https://arxiv.org/pdf/2507.21488) | 20 games: 51.4% → 53.2%. ~500 games: 53.65%. ~2,500 games: 53.8% overall, 55.5% for 2000+ blitz players. Direct Maia-2 fine-tuning reached only 52.0% [arxiv](https://arxiv.org/pdf/2507.21488) [arxiv](https://arxiv.org/pdf/2507.21488) | 20–2,500 games | CSSLab/maia4all (GPL-3.0) is marked "work in progress, to be officially released soon" [github](https://github.com/CSSLab/maia4all) |
| **Maia-3 / Chessformer** (Monroe et al., ICLR 2026) | Encoder transformer with squares as tokens and an 8-position history [arxiv](https://arxiv.org/html/2605.19091v1) | 57.1% (79M), 56.6% (23M), 55.4% (5M) vs Maia-2 at 52.0% on the Allie test set [arxiv](https://arxiv.org/html/2606.04473v1) [arxiv](https://arxiv.org/html/2605.19091v1) | n/a | CSSLab/maia3, **AGPL-3.0 (code and weights)**. Inference/UCI only; no training code published. Blitz-trained; no clock input [github](https://github.com/CSSLab/maia3) [arxiv](https://arxiv.org/pdf/2606.25176) |
| **Matilda** (Carlson, arXiv 2606.25176, July 2026; single-author preprint) | 1.7M-param set-transformer that re-ranks a frozen Maia-3's top-16 moves, using an optional 32-d player-style vector and time control [arxiv](https://arxiv.org/html/2606.25176v2) [arxiv](https://arxiv.org/html/2606.25176v2) | Style vectors gave +0.41% relative NLL overall; extended to 4,038 club players. A new player breaks even at about 60 moves. The played move is in Maia-3's top-16 99.9% of the time [arxiv](https://arxiv.org/html/2606.25176v2) | Very little | Preprint; not peer reviewed |
| **ChessMimic / Otter** (2026 preprints) | Small transformers conditioned on clock and history | ChessMimic 56.2% on the Allie test set, but its think-time model is weaker than Allie's (r=0.41). Otter 55.23% vs Maia-2's reported 53.25% (+1.98 pp) | n/a | Preprints |
| **DeepMind "Grandmaster-level chess without search"** (Ruoss et al., 2024) | 270M transformer trained on 15B Stockfish action-values [arxiv](https://arxiv.org/pdf/2402.04494) | Lichess blitz Elo 2895 vs humans (2299 vs bots) [chess-journal](https://www.chess-journal.com/googleSearchless.html) | n/a | google-deepmind/searchless_chess. Trained to imitate Stockfish, so it is **not human-like**. Use only as a reference [arxiv](https://arxiv.org/pdf/2402.04494) |

**What this means for you:**
- **Your data is borderline for the 2022 recipe but enough for the 2025–26 recipes.** You have about 4,000 games, roughly 160k of your own positions. The KDD 2022 recipe only helped from about 5,000 games. [arxiv](https://arxiv.org/html/2008.10086v3) The newer embedding and residual methods work with tens to thousands of games. [arxiv](https://arxiv.org/pdf/2507.21488)
- **Prefer residual or embedding approaches over naive full fine-tuning.** Maia4All showed that directly fine-tuning Maia-2 barely helps: in low-data settings it added only 0.5 percentage points over the base model (Tang et al., arXiv 2507.21488).
- **Your rating suits these models.** At around 2000 you sit in the band where human-move models are strongest and most distinctive. Maia-Individual's own GM attempts didn't yield gains. [arxiv](https://arxiv.org/html/2008.10086v3)
- **Set realistic expectations.** The Maia4All authors note that the accuracy ceiling for human move prediction is "far below 100%", because even the same player won't always make the same decision in the same position. A personalized model at 55–58% top-1 is research-grade. "Feels like me" will depend as much on distributional behavior (opening choices, blunder rate, time use) as on top-1 accuracy.

### 2. Approach comparison

| Approach | Difficulty | Compute | Expected fidelity | Verdict |
|---|---|---|---|---|
| **a1. Full fine-tune of Maia-2 (MIT)** on your games | Medium. `train.run` starts a new model unless you enable checkpoint restore, so you need to adapt it to start from pretrained weights [github](https://github.com/CSSLab/maia2) | Under 1 GPU-hour per run on a T4/P100 | +0–2 pp. Real risk of overfitting at 4k games | Good secondary and ablation |
| **a2. Maia-3 + your own fine-tuning loop** | Medium–high (inference code only; you write the training loop around `maia3/models.py`) | 5M/23M: about 1–3 GPU-hours | Best base accuracy (55–57%); personalization gain is unknown | Strong, but AGPL |
| **a3. Frozen base + residual "you-adapter"** (Matilda/Maia4All-style: player embedding + time control + clock + optional Stockfish features, re-ranking the base's top-k) | Medium | Minutes to about 1 GPU-hour; featurize once | Highest expected gain for your data size. By design it cannot score worse than the base | **Recommended core** |
| **a4. Maia-Individual pipeline (Maia-1/lc0)** | Low–medium, but old CUDA/TF/lc0 tooling | Small | Weakest base; ≤5k games gave little or negative gain | Only as a historical baseline |
| **b. Small policy net from scratch**: pretrain on Lichess 1900–2100 blitz/rapid, then fine-tune | High | Pretraining on tens of millions of positions takes several to tens of GPU-hours ($5–30 on a rented 4090) | Likely 50–54% before personalization; you'd be re-deriving Maia-2 | Learning exercise only. Skip unless you want the ML depth |
| **c1. Personal opening book** (your move frequencies per position, by color) | Low | CPU | Very high in the first 10–20 plies; this is where stylometry finds you easiest | **Include** |
| **c2. Stockfish with Skill Level / UCI_Elo / error injection** | Low | CPU | Poor: 33–41% move-matching, with "un-human" mistakes [arxiv](https://arxiv.org/pdf/2006.01855) | Baseline only |
| **c3. kNN / statistical** (position-hash lookup, plus features like "captures toward the center") | Low | CPU | Only works for repeated positions; falls apart once your moves no longer match a familiar position (fewer than 10% of positions are shared by ply 20) [arxiv](https://arxiv.org/html/2008.10086v3) | Baseline only |
| **d. Hybrid**: book → personal policy → optional shallow search; clock/think-time model; time-control conditioning | Medium | CPU at inference | Best "feels like me" | **Recommended product** |
| **e. LLM fine-tuning** (PGN-as-text) | Medium–high | 10–100× the compute | GPT-3.5-instruct reached 53.7%, behind Allie and Maia-3. Prompting GPT-4o mini lost to chess-specific models. [usf](https://digitalcommons.usf.edu/cgi/viewcontent.cgi?article=11928&context=etd) [arxiv](https://arxiv.org/html/2410.03893) Illegal moves and latency are problems | Not worth it. Mention it in the write-up as a considered alternative |

Some notes on these options:

**Why sample instead of taking the argmax.** Argmax maximizes top-1 accuracy but collapses onto the most common move. That makes the bot more consistent, and usually stronger, than you are. Sampling from the policy, with a temperature tuned on validation NLL, reproduces your *rate* of second-best moves and blunders. Maia-3's UCI exposes Temperature and TopP, and you can do the same in your own wrapper. [github](https://github.com/CSSLab/maia3)

**Time pressure is a feature, not noise.** The research models deliberately drop time-scramble data. Maia4All and Maia-3 discard positions where a player has under 30 seconds, and none of Maia-2 or Maia-3 take clock inputs. [arxiv](https://arxiv.org/html/2605.19091v1) [arxiv](https://arxiv.org/pdf/2507.21488) For "mistakes and all," keep those positions and feed in remaining clock, increment and time control as inputs. ChessMimic, Otter and Matilda all show that clock and time-control context carries signal. [arxiv](https://arxiv.org/html/2606.25176v2) [arxiv](https://arxiv.org/html/2606.04473)

**Think time.** Train a small regressor that predicts log-seconds per move. Inputs can include clock left, increment, ply, the base model's entropy over moves, the eval swing, and whether the move is a recapture. Make the bot wait that long before playing. Allie got r≈0.70 with a big model; [researchgate](https://www.researchgate.net/publication/405922498_ChessMimic_Per-Rating_Transformer_Models_for_Human_Move_Clock_and_Outcome_Prediction_in_Online_Blitz_Chess) a simple model on your own `%clk` data is a credible first version.

### 3. Data pipeline
- **Lichess:** call `GET https://lichess.org/api/games/user/{username}` with `clocks=true`, `evals=true`, `opening=true` and `perfType=blitz,rapid`, and request ndjson. Use a personal token: per the Lichess API spec, export runs at 20 games/s anonymously, 30/s with OAuth2, and 60/s when downloading your own games. Expect about 2 minutes for your full history. The `berserk` or `lichess` Python clients wrap this.
- **Chess.com:** the Published-Data API is unauthenticated and read-only. Get the list of monthly archives from `https://api.chess.com/pub/player/{username}/games/archives`, then fetch each month from `.../games/{YYYY}/{MM}/pgn`. [chess](https://www.chess.com/announcements/view/published-data-api) [chessnerd](https://chessnerd.net/chesscom-api.html) Live-game PGNs normally carry `[%clk]` comments; check this on a sample.
- **Parsing:** use `python-chess` (`chess.pgn.read_game`, `node.clock()`). For each of *your* moves, emit: FEN, the previous 7 positions (for Maia-3), the move in UCI, both players' ratings, time control, clock before the move, think time (the difference between successive clocks plus increment), platform, color, ply and result.
- **Handling color:** mirror Black-to-move positions to White's perspective, as the Maia family does. [arxiv](https://arxiv.org/pdf/2507.21488) Also keep a `color` flag, because your White and Black repertoires and styles differ.
- **Ratings:** map Chess.com ratings onto a Lichess-equivalent scale before conditioning, since the base models were trained on Lichess ratings. Your 1800 Chess.com blitz is not 1800 on Lichess. Also add a platform flag so the adapter can learn differences.
- **Splits:** split **by game**, never by position. Hold out your most recent ~10% of games as a temporal "future" test set, and take a random 10% of the rest as validation. Maia-Individual used 80/10/10 plus a future month. [arxiv](https://arxiv.org/html/2008.10086v3) Dedupe identical (position, move) pairs across splits when reporting results on unseen positions.
- **Augmentation:** keep it limited. Left-right mirroring breaks castling rights and changes the style you're trying to learn. Color-flipping is already handled by the mirroring. The more useful lever is up-weighting middlegame and endgame positions, since the opening is memorized.
- **Pretraining or prototype data (optional):** the Lichess open database has about 8.13 billion standard rated games. A monthly file is about 30 GB as `.zst` and decompresses to about 7× that, [lichess +2](http://database.lichess.org/) so stream it with `zstandard` and filter to 1900–2100 blitz/rapid. You only need this for approach (b), or to build "prototype players" à la Maia4All.

### 4. Evaluation protocol
1. **Move-matching:** top-1 and top-3 accuracy, plus NLL/perplexity, on the held-out test games. Report every result two ways: all plies, and plies after 10. The convention in the literature is to exclude the first 10 plies, because the opening inflates accuracy. [arxiv](https://arxiv.org/html/2008.10086v3)
2. **Baselines on the same positions:**
   - Maia-2 (blitz/rapid) at your rating
   - Maia-3 5M/23M at your rating
   - Stockfish at fixed depth and with Skill Level 0–20 (UCI_Elo floor is 1320) [github](https://official-stockfish.github.io/docs/stockfish-wiki/UCI-Protocol-and-Stockfish-Commands.html)
   - your opening book combined with Maia
   - kNN
3. **Slices:**
   - Game phase: opening ≤ ply 20; middlegame; endgame defined by material
   - Time control and platform
   - Clock: above vs below 30 s, and below 10 s
   - Move quality: bucket by Stockfish win-probability loss, as in KDD 2022 Figure 3 [arxiv](https://arxiv.org/html/2008.10086v3)
   - Positions seen vs unseen in training
4. **Distributional similarity:** let the bot play many games against Maia-at-your-rating opponents from your real openings. Compare its average centipawn loss and blunder/mistake/inaccuracy rates per 100 moves against yours, by phase and by clock bucket. Use KS or Wasserstein distances, and do the same for think-time distributions.
5. **Stylometry test:**
   - Train a classifier, for example the NeurIPS 2021-style embedding or gradient boosting on move-level features, to separate *your* games from the bot's games.
   - Report AUC; an AUC near 0.5 means the bot is hard to tell apart from you.
   - Also run the KDD 2022 approach: which model (you-model vs Maia vs bot) best predicts held-out *you*?
6. **Subjective "Turing" test:** show friends or clubmates blinded game pairs (you vs bot) and record how often they guess right. Also report your own self-play impressions.

### 5. Deployment
- **Lichess:** use **lichess-bot**, the free Python bridge for UCI engines, or BotLi.
  - Create a *new* account. The BOT upgrade only works on an account that has never played a game, and it is irreversible. [github](https://github.com/Torom/BotLi) [lichess](https://lichess.org/forum/general-chess-discussion/can-a-human-account-become-bot-account)
  - Bots can't use the lobby or pools; they play challenges only. [github](https://github.com/dylan-almonte/chess/issues/13)
  - Bot-vs-bot games are capped at 100 per day; games against humans are unlimited. [github](https://github.com/lichess-bot-devs/lichess-bot/wiki/Configure-lichess-bot)
  - Keep games casual while you test. [github](https://github.com/dylan-almonte/chess/issues/13)
  - Implement the think-time delay inside your engine's `go` handling, capped by the real clock.
- **Chess.com:** its public API is read-only ("You cannot send game-moves"), [chessnerd](https://chessnerd.net/chesscom-api.html) so don't plan on a Chess.com bot.
- **UCI:**
  - `maia3-uci --checkpoint-path your.pt --device cpu` already serves custom Maia-3 checkpoints in Nibbler or any UCI GUI. [github](https://github.com/CSSLab/maia3)
  - For Maia-2 or the adapter, write a ~150-line python-chess-style UCI loop.
- **CPU feasibility:** one forward pass per move. CSSLab labels Maia3-5M "First try, CPU, chess GUIs", [github](https://github.com/CSSLab/maia3) and the adapter adds about 1.7M parameters at most. That is milliseconds to tens of milliseconds on a T14s. Only optional Stockfish features or search would add cost.
- **Hosting:** $0 if you run the bot from your laptop during demo windows. A small always-on VPS or free CPU Space for a web demo costs roughly $0–6/month (a rough estimate).
- **License note:** if you build on Maia-3 and serve it over a network, AGPL-3.0 requires publishing your source. That's fine for an open-source repo, but it rules out a closed product. Maia-2 (MIT) has no such constraint. [github](https://github.com/CSSLab/maia2)

### 6. Compute and costs

| Resource | What you get | Use it for |
|---|---|---|
| ThinkPad T14s (CPU) | Free | Data download and parsing, opening book, Stockfish annotation (overnight), inference, the bot |
| Kaggle Notebooks | 30 GPU-hours/week free (sometimes more, per Kaggle's docs); P100 or 2×T4; 9–12 h sessions | All fine-tuning and adapter training |
| Colab Free / Pro | Free T4 (not guaranteed); Pro $9.99 for 100 compute units. A T4 uses ~1.2–2 CU/hr (sources disagree) and an A100 40GB ~5.4 CU/hr [thundercompute +2](https://www.thundercompute.com/blog/colab-alternatives-for-cheap-deep-learning-in-2025) | Backup and quick experiments |
| RunPod Community / Vast.ai | RTX 4090 at $0.34/hr on RunPod Community Cloud, billed per second (Secure Cloud is $0.59–0.74/hr depending on source), or ~$0.14–0.39/hr (Vast, interruptible) | Only for from-scratch pretraining or the 23M/79M sweeps |

Your data is about 160k positions. A full fine-tune of a 5–23M model for 10–20 epochs is about 2–3M samples, which takes minutes to an hour on a T4. For scale, Maia-Individual saw most of its gains within 12,000 steps, or 3M board–move pairs. [arxiv](https://arxiv.org/html/2008.10086v3)

**Budget scenarios:**
- **$0:** Kaggle only.
- **About $10–30:** rented-4090 sweeps and optional pretraining.
- **About $0–6/month:** optional always-on hosting.

## Recommended Path and Phased Roadmap

**Phase 0 – Data (weekend 1, $0)**
- Export both platforms and parse to Parquet.
- Run Stockfish annotation of your positions on the laptop overnight (for example depth 12–14 on your moves only), which feeds both the move-quality slices and the adapter's engine features.
- Do exploratory analysis: your opening tree, think-time curves, blunder rate vs clock.

**Phase 1 – Baselines (week 2, $0)**
- Evaluate Maia-2, Maia-3 (5M/23M), Stockfish (depth and Skill), the opening book and kNN on your test set.
- This produces your first results table and fixes the bar any personalized model has to beat.

**Phase 2 – Personalization (weeks 3–4, Kaggle)**
- Run the base model over all your positions once and save its features (top-16 logits, pooled hidden state, history).
- Train the residual adapter with: a player/platform/color embedding; time control; clock left; think-time features; optional engine features.
- Ablations:
  - full fine-tune of Maia-2 (MIT) vs adapter-on-Maia-3
  - data-size curve at 250/1k/2k/4k games
  - with vs without clock features
  - argmax vs tuned temperature
- Aim: beat the best base model on test NLL and top-1, especially on unseen positions and below-30-second positions.

**Phase 3 – Hybrid bot (week 5, $0)**
- Opening book: up to ply N, sample by your frequencies, then hand off to the model.
- Policy sampling with the tuned temperature.
- Think-time regressor.
- Optional: very shallow, rare search only where your data shows you calculate (long thinks). Allie-style "search when you'd ponder" is a nice stretch goal.

**Phase 4 – Human-likeness evaluation (week 6)**
- Distributional similarity tests and the stylometry-classifier AUC.
- A blinded friend test.
- Live casual games on the BOT account.

**Phase 5 – Ship (week 7)**
- Open-source repo with reproducible configs.
- Model card (including privacy notes; CSSLab declined to release player models because of the stylometry risk). [github](https://github.com/CSSLab/maia-individual)
- Write-up or blog post with plots.
- Short demo video, plus the live bot link and optionally a web board demo.

**Optional stretch:** pretrain a small transformer from scratch on Lichess 1900–2100 data (about $10–30), to show you understand the full pipeline and to compare it against Maia.

## Resume value: what makes it impressive
- **A clear question:** "How much of an individual's chess style is learnable from 4,000 games?" Answer it with data-size curves and ablations.
- **Strong, fair baselines:** Maia-2, Maia-3 and Stockfish evaluated on identical held-out, temporally split games, with confidence intervals (bootstrap over games).
- **Behavioral fidelity beyond top-1:** matching your blunder rate under time pressure and your think-time distribution, plus stylometry AUC. Few hobby projects measure "mistakes and all."
- **A working artifact:** a live Lichess bot, a UCI engine, and a clean repo with a Kaggle notebook anyone can rerun.
- **Responsible-AI framing:** the stylometry and privacy discussion, plus the license choice (AGPL vs MIT).

## Caveats
- **Several 2026 sources are preprints.** Matilda, ChessMimic and Otter are not peer reviewed, and Matilda is single-author. Treat their numbers as suggestive. Matilda's gains are relative NLL improvements, not top-1 accuracy points.
- **Results aren't directly comparable across papers.** Test sets and filters differ: some drop the first 10 plies and sub-30-second positions, and some mix blitz and rapid. Only compare models on *your* test set.
- **Mixed platforms add noise.** Chess.com and Lichess games differ in pairing pools, time controls and rating scales.
- **Data is thinner than it looks.** Your blitz and rapid games split the data further, and only about half the plies in each game are yours.
- **Training code gaps:**
  - Maia-3 has no public training code; the sources disagree, but the public repo has none.
  - Maia4All's code is unreleased. [github](https://github.com/CSSLab/maia4all)
  - Maia-Individual's pipeline depends on an aging lc0/TensorFlow stack. [github](https://github.com/CSSLab/maia-individual)
- **Prices and free-tier quotas change.** Colab's compute-unit burn rates differ between sources (T4 at 1.19 to 1.96 CU/hr). [mccormickml](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/) [dev](https://dev.to/rushichaudhari/is-google-colab-pro-really-worth-it-5531) Re-check before spending.
- **Expected accuracy gains are extrapolations** from published results, not guarantees. Some players are simply more predictable than others. [arxiv](https://arxiv.org/html/2008.10086v3)
- **Lichess rules:** never run the model on your human account. Use a separate BOT account, keep games casual while testing, and follow the bot terms of service.
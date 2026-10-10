# Model card: CloroxMaelstromBot you-adapter

**What it is.** A 0.3M-parameter residual re-ranker on top of frozen Maia-3 5M (CSSLab,
AGPL-3.0) that re-scores Maia-3's top-16 candidate moves for one specific player, plus a
personal opening book, a think-time model and a sampling policy. Served as a UCI engine
through lichess-bot.

**Training data.** ~4,100 blitz and rapid games (143k moves) of one consenting player, the
repository owner, exported from their own Lichess and Chess.com accounts. No other
individual's games are modelled; opponents' moves are used only as context.

**Intended use.** Playing casual games on a clearly labelled Lichess BOT account; research
on personalised human-move models; a portfolio project.

**Not intended for.** Playing on a human account (this violates Lichess and Chess.com
terms and is cheating); impersonating the player elsewhere; rated games until the player
decides so.

**Privacy.** CSSLab declined to release Maia-Individual player models because stylometry
can identify players from their moves. This repository releases code and aggregate
reports only. Trained weights (`models/`) and the game data (`data/`) are git-ignored and
stay with the player. Anyone training this on someone else's games should have their
consent.

**Performance.** Temporal test split (408 games): top-1 59.4% vs 58.8% frozen Maia-3,
NLL 1.191 vs 1.237. See `docs/results.md` for slices, ablations and human-likeness tests.

**Licence.** Code in this repository: AGPL-3.0, because Maia-3's weights and code are
AGPL-3.0 and the bot serves them over the network. Maia-2 (MIT) is a drop-in alternative
base for a closed deployment.

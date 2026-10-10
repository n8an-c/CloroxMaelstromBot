"""Export a Lichess user's games as PGN with clocks, evals and opening tags.

Uses GET https://lichess.org/api/games/user/{username}. With a personal API
token (LICHESS_TOKEN env var or --token) the export of your *own* games runs at
60 games/s, so ~4,000 games take about a minute or two.

    maelstrom-export-lichess --user NAME [--perf blitz,rapid] [--since 2020-01-01]

Writes data/raw/lichess/NAME.pgn (appends when --since is given).
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import sys
import time

import requests

API = "https://lichess.org/api/games/user/{user}"


def export(user: str, out: pathlib.Path, perf: str = "blitz,rapid", token: str | None = None,
           since: dt.datetime | None = None, rated_only: bool = False, retries: int = 4) -> int:
    params = {
        "clocks": "true",
        "evals": "true",
        "opening": "true",
        "perfType": perf,
        "moves": "true",
        "tags": "true",
    }
    if rated_only:
        params["rated"] = "true"
    if since is not None:
        params["since"] = int(since.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    headers = {"Accept": "application/x-chess-pgn", "User-Agent": "maelstrom-chess-bot research"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "ab" if since is not None and out.exists() else "wb"
    n_games = 0
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            with requests.get(API.format(user=user), params=params, headers=headers, stream=True, timeout=(15, 300)) as r:
                if r.status_code == 429:
                    raise requests.HTTPError("429 rate limited", response=r)
                r.raise_for_status()
                with open(out, mode) as f:
                    for chunk in r.iter_lines(decode_unicode=False):
                        if chunk.startswith(b"[Event "):
                            n_games += 1
                            if n_games % 200 == 0:
                                print(f"  {n_games} games...", file=sys.stderr)
                        f.write(chunk + b"\n")
            return n_games
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            if attempt == retries:
                raise
            wait = 60 if "429" in str(e) else delay
            print(f"  retry in {wait}s after: {e}", file=sys.stderr)
            time.sleep(wait)
            delay *= 2
            mode = "wb" if mode == "wb" else "ab"
            n_games = 0
    return n_games


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user", required=True)
    p.add_argument("--perf", default="blitz,rapid", help="comma list: ultraBullet,bullet,blitz,rapid,classical")
    p.add_argument("--token", default=os.environ.get("LICHESS_TOKEN"))
    p.add_argument("--since", default=None, help="YYYY-MM-DD; append games after this date")
    p.add_argument("--rated-only", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    out = pathlib.Path(a.out) if a.out else pathlib.Path("data/raw/lichess") / f"{a.user}.pgn"
    since = dt.datetime.fromisoformat(a.since) if a.since else None
    n = export(a.user, out, a.perf, a.token, since, a.rated_only)
    print(f"wrote {n} games -> {out}")


if __name__ == "__main__":
    main()

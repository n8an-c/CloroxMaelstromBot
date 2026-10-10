"""Export a Chess.com user's games via the Published-Data API (read-only, no auth).

    GET https://api.chess.com/pub/player/{user}/games/archives   -> list of month URLs
    GET https://api.chess.com/pub/player/{user}/games/YYYY/MM/pgn -> PGN for that month

Chess.com returns 403 without a descriptive User-Agent, so we always send one.

    maelstrom-export-chesscom --user NAME [--contact you@example.com]

Writes data/raw/chesscom/NAME/YYYY-MM.pgn, skipping months already on disk
(except the current month, which is always refreshed).
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys
import time

import requests

ARCHIVES = "https://api.chess.com/pub/player/{user}/games/archives"


def _get(session: requests.Session, url: str, retries: int = 4) -> requests.Response:
    delay = 2.0
    for attempt in range(retries + 1):
        r = session.get(url, timeout=60)
        if r.status_code == 429:
            time.sleep(max(delay, 10))
        elif r.status_code >= 500:
            time.sleep(delay)
        else:
            r.raise_for_status()
            return r
        if attempt == retries:
            r.raise_for_status()
        delay *= 2
    raise RuntimeError("unreachable")


def export(user: str, out_dir: pathlib.Path, contact: str = "github.com/n8an-c/CloroxMaelstromBot",
           pause: float = 0.5) -> list[pathlib.Path]:
    s = requests.Session()
    s.headers["User-Agent"] = f"maelstrom-chess-bot research ({contact})"
    s.headers["Accept"] = "application/json, application/x-chess-pgn"
    archives = _get(s, ARCHIVES.format(user=user)).json()["archives"]
    out_dir.mkdir(parents=True, exist_ok=True)
    this_month = dt.date.today().strftime("%Y-%m")
    written = []
    for url in archives:
        yyyy, mm = url.rstrip("/").split("/")[-2:]
        target = out_dir / f"{yyyy}-{mm}.pgn"
        if target.exists() and f"{yyyy}-{mm}" != this_month:
            continue
        pgn = _get(s, url + "/pgn").text
        target.write_text(pgn, encoding="utf-8")
        written.append(target)
        print(f"  {target.name}: {pgn.count('[Event ')} games", file=sys.stderr)
        time.sleep(pause)
    return written


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user", required=True)
    p.add_argument("--contact", default="github.com/n8an-c/CloroxMaelstromBot")
    p.add_argument("--out-dir", default=None)
    a = p.parse_args(argv)
    out_dir = pathlib.Path(a.out_dir) if a.out_dir else pathlib.Path("data/raw/chesscom") / a.user
    files = export(a.user, out_dir, a.contact)
    print(f"wrote/refreshed {len(files)} monthly files -> {out_dir}")


if __name__ == "__main__":
    main()

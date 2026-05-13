"""Fetch get_odds for the dates covered by events.parquet and unpack into Parquet tables.

Input : events.parquet (for date range + event_key filter)
Output: $TENNIS_ROOT/tables/{odds_long, odds_captures, markets_meta}.parquet
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import polars as pl

ROOT = Path(os.environ.get("TENNIS_ROOT", Path(__file__).resolve().parent.parent))
OUT = ROOT / "tables"
APIKEY = os.environ.get("API_TENNIS_KEY")
if not APIKEY:
    raise SystemExit("set API_TENNIS_KEY in the environment (api-tennis.com api key)")
BASE = "https://api.api-tennis.com/tennis/"


def fetch_odds(date_start: str, date_stop: str) -> dict:
    url = f"{BASE}?method=get_odds&APIkey={APIKEY}&date_start={date_start}&date_stop={date_stop}"
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def parse_odds_value(v):
    """Cast '2.40' -> 2.40; bad/empty -> None."""
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def walk(node, market: str, path: tuple = ()):
    """Recursively walk a market body, emitting (market, outcome, line, bookmaker, odds_str) tuples.

    A leaf is a {bookmaker: odds_string} dict. Path elements above the leaf are interpreted as
    (outcome, line) — when there are >2 path elements, all but the last fold into outcome with '|'.
    """
    if not isinstance(node, dict):
        return
    vals = list(node.values())
    if vals and all(isinstance(v, str) for v in vals):
        if len(path) == 0:
            outcome, line = None, None
        elif len(path) == 1:
            outcome, line = path[0], None
        elif len(path) == 2:
            outcome, line = path[0], path[1]
        else:
            outcome = "|".join(path[:-1])
            line = path[-1]
        for book, odds in node.items():
            yield (market, outcome, line, book, odds)
        return
    for k, v in node.items():
        yield from walk(v, market, path + (str(k),))


def main():
    events_path = OUT / "events.parquet"
    if not events_path.exists():
        print(f"missing {events_path}", file=sys.stderr)
        sys.exit(1)

    ev = pl.read_parquet(events_path, columns=["event_key", "event_date"])
    event_dates = sorted({d for d in ev["event_date"].to_list() if d})
    event_keys = set(ev["event_key"].to_list())
    print(f"events.parquet: {ev.height:,} rows, {len(event_keys):,} unique event_keys, dates: {event_dates}")

    odds_cols = ["ts_ns", "event_key", "match_date", "market", "outcome", "line", "bookmaker", "odds", "in_events"]
    cap_cols = ["ts_ns", "date_start", "date_stop", "n_matches", "n_odds_rows"]
    odds_buf = {c: [] for c in odds_cols}
    cap_buf = {c: [] for c in cap_cols}

    for d in event_dates:
        t0 = time.time()
        ts_ns = time.time_ns()
        data = fetch_odds(d, d)
        result = data.get("result") or {}
        if not isinstance(result, dict):
            print(f"  {d}: result not a dict (got {type(result).__name__}), skipping")
            continue
        n_match = len(result)
        n_rows_before = len(odds_buf["ts_ns"])

        for match_key, markets in result.items():
            try:
                ek = int(match_key)
            except (ValueError, TypeError):
                continue
            in_ev = ek in event_keys
            if not isinstance(markets, dict):
                continue
            for market, body in markets.items():
                for market_, outcome, line, book, odds_str in walk(body, market):
                    odds_buf["ts_ns"].append(ts_ns)
                    odds_buf["event_key"].append(ek)
                    odds_buf["match_date"].append(d)
                    odds_buf["market"].append(market_)
                    odds_buf["outcome"].append(outcome)
                    odds_buf["line"].append(line)
                    odds_buf["bookmaker"].append(book)
                    odds_buf["odds"].append(parse_odds_value(odds_str))
                    odds_buf["in_events"].append(in_ev)

        n_rows = len(odds_buf["ts_ns"]) - n_rows_before
        cap_buf["ts_ns"].append(ts_ns)
        cap_buf["date_start"].append(d)
        cap_buf["date_stop"].append(d)
        cap_buf["n_matches"].append(n_match)
        cap_buf["n_odds_rows"].append(n_rows)
        print(f"  {d}: {n_match} matches, {n_rows:,} odds rows ({time.time()-t0:.1f}s)")

    # ---- write tables ----
    odds_schema = {
        "ts_ns": pl.Int64,
        "event_key": pl.Int64,
        "match_date": pl.Utf8,
        "market": pl.Utf8,
        "outcome": pl.Utf8,
        "line": pl.Utf8,
        "bookmaker": pl.Utf8,
        "odds": pl.Float64,
        "in_events": pl.Boolean,
    }
    cap_schema = {
        "ts_ns": pl.Int64,
        "date_start": pl.Utf8,
        "date_stop": pl.Utf8,
        "n_matches": pl.UInt32,
        "n_odds_rows": pl.UInt32,
    }

    odds_df = pl.DataFrame(odds_buf, schema=odds_schema, strict=False)
    cap_df = pl.DataFrame(cap_buf, schema=cap_schema, strict=False)

    # markets_meta — one row per market with coverage summary
    meta_df = (
        odds_df.group_by("market")
        .agg(
            pl.col("event_key").n_unique().alias("n_matches"),
            pl.col("outcome").n_unique().alias("n_outcomes"),
            pl.col("line").n_unique().alias("n_lines"),
            pl.col("bookmaker").n_unique().alias("n_bookmakers"),
            pl.col("line").is_not_null().any().alias("has_line"),
            pl.len().alias("n_rows"),
        )
        .sort("n_rows", descending=True)
    )

    for name, df in [("odds_long", odds_df), ("odds_captures", cap_df), ("markets_meta", meta_df)]:
        t = time.time()
        path = OUT / f"{name}.parquet"
        df.write_parquet(path, compression="zstd", compression_level=6, statistics=True)
        mb = os.path.getsize(path) / (1024 * 1024)
        print(f"wrote {name}: rows={df.height:,} cols={df.width} -> {path.name} ({mb:.2f} MB) in {time.time()-t:.1f}s")


if __name__ == "__main__":
    main()

# Tennis API + Kalshi — unpacked tables

Star-schema parquet layout combining three data sources:

| Source | Producer | Tables |
|---|---|---|
| api-tennis.com REST poll (live events) | `./tools/unpack.py` | `snapshots`, `events`, `scores_long`, `stats_long`, `games_long`, `points_long` |
| api-tennis.com REST poll (pre-match odds) | `./tools/fetch_odds.py` | `odds_long`, `odds_captures`, `markets_meta` |
| Kalshi raw orderbook NDJSON | `./tools/build_kalshi_orderbook.py` | `orderbook_l2`, `orderbook_top` |
| api-tennis.com **WebSocket** (live stream) | `./tools/ws_daemon.py` | `live/<table>/seg_<ts>.parquet` (rolling segments with the same schema as the six REST-derived tables) |

All files are Parquet with ZSTD compression and dictionary encoding.

**Universal join keys:** every tennis long table joins back to `events` (and through it to `snapshots`) on `(ts_ns, event_key)`. Odds tables join to `events` on `event_key`. Kalshi orderbook tables join to events via `event_key` once you parse it out of the Kalshi `ticker` (e.g. `KXATPMATCH-26MAY12DARZVE-ZVE` → match stem `26MAY12DARZVE`, side `ZVE`). `ts_ns` everywhere is nanoseconds since Unix epoch.

---

## 1. `snapshots.parquet` — 9,896 rows × 2 cols

One row per API poll. Use this when you need to reason about polling cadence or align snapshots across matches.

| Column | Type | Meaning |
|---|---|---|
| `ts_ns` | Int64 | Nanoseconds since Unix epoch. **Monotonically increasing.** Primary key. |
| `n_events` | UInt32 | Number of live matches returned in this snapshot (1–9 in this dataset). |

---

## 2. `events.parquet` — 201,221 rows × 34 cols

One row per `(snapshot, match)` — i.e. the scalar state of one match at one moment in time. This is the table you'll join everything else against.

**Primary key:** `(ts_ns, event_key)`.

### Identity / metadata
| Column | Type | Notes |
|---|---|---|
| `ts_ns` | Int64 | Snapshot time. |
| `event_key` | Int64 | Match id. |
| `event_date` | Utf8 | `YYYY-MM-DD`. |
| `event_time` | Utf8 | `HH:MM` scheduled start time. |
| `tournament_key` | Int64 | |
| `tournament_name` | Utf8 | e.g. `Bengaluru 2`. |
| `tournament_round` | Utf8 | e.g. `Bengaluru 2 - 1/16-finals`. |
| `tournament_season` | Utf8 | |
| `event_type_type` | Utf8 | e.g. `Challenger Men Singles`, `Itf Women Doubles`. |
| `event_qualification` | Boolean | Parsed from `"True"`/`"False"`. |
| `event_live` | Boolean | Parsed from `"1"`/`"0"`. |

### Players
| Column | Type | Notes |
|---|---|---|
| `first_player_key`, `second_player_key` | Int64 | API player ids. |
| `event_first_player`, `event_second_player` | Utf8 | Display names. |
| `event_first_player_logo`, `event_second_player_logo` | Utf8 | URL or null (~⅔ are null). |

### Current state (changes every snapshot)
| Column | Type | Notes |
|---|---|---|
| `event_winner` | Utf8 | `"First Player"`, `"Second Player"`, or null until the match ends. |
| `event_status_raw` | Utf8 | Verbatim API string, e.g. `Set 1`, `Finished`, `Interrupted`. |
| `event_status_code` | Utf8 | Normalized: `set1`, `set2`, `set3`, `finished`, `interrupted`. |
| `is_finished` | Boolean | Convenience flag (= `event_status_code == 'finished'`). |
| `is_interrupted` | Boolean | Convenience flag. |
| `event_serve_raw` | Utf8 | Verbatim, e.g. `First Player`. |
| `serve_side` | UInt8 | `1` (first player) / `2` (second player) / null. |
| `event_final_result_raw` | Utf8 | e.g. `"2 - 1"` (sets won). |
| `sets_won_a`, `sets_won_b` | Int32 | Parsed from `event_final_result_raw`. |
| `event_game_result_raw` | Utf8 | e.g. `"40 - A"` (points in current game). |
| `game_point_a`, `game_point_b` | Int16 | `0/15/30/40/50`. **`50` = advantage** (originally `A`). Null for unrecognized tokens (e.g. tiebreak scores past 7). |
| `score_summary` | Utf8 | Compact set-by-set summary, e.g. `"6-4,3-6,2-1"`. Built from `scores_long`. |

### Cardinality hints (saves a `JOIN ... GROUP BY` for common queries)
| Column | Type | Meaning |
|---|---|---|
| `n_sets` | UInt32 | Number of entries this row contributes to `scores_long`. |
| `n_stat_rows` | UInt32 | Number of entries this row contributes to `stats_long`. |
| `n_pbp_games` | UInt32 | Number of entries this row contributes to `games_long`. |

---

## 3. `scores_long.parquet` — 329,849 rows × 6 cols

One row per `(snapshot, match, completed-or-in-progress set)`. Source: the `scores` array on each event.

**Primary key:** `(ts_ns, event_key, set_index)`.

| Column | Type | Notes |
|---|---|---|
| `ts_ns` | Int64 | |
| `event_key` | Int64 | |
| `set_index` | UInt8 | Set number, 1-based. |
| `score_first` | Int32 | Games won by first player in this set. |
| `score_second` | Int32 | Games won by second player in this set. |
| `score_set_raw` | Utf8 | Verbatim `score_set` string from the API. |

> Note: tiebreak scores are reported by the API as decimals on the games column (e.g. `"6.5"` / `"7.7"`). Those rows have `score_first` / `score_second` = null (int parse fails); the raw form is preserved in `score_set_raw` and the surrounding game numbers (e.g. `"6"` vs `"7"`) in the parsed columns. If you need tiebreak-exact handling, parse from `score_set_raw` or use `events.score_summary`, which keeps the decimal form (e.g. `"6.5-7.7,6-4,10-8"`).

---

## 4. `stats_long.parquet` — 14,859,183 rows × 10 cols

One row per `(snapshot, match, player, stat_period, stat_type, stat_name)`. Source: the `statistics` array.

**Primary key:** `(ts_ns, event_key, player_key, stat_period, stat_type, stat_name)`.

| Column | Type | Notes |
|---|---|---|
| `ts_ns` | Int64 | |
| `event_key` | Int64 | |
| `player_key` | Int64 | Joins to `events.first_player_key` or `events.second_player_key`. |
| `stat_period` | Utf8 | `match`, `set1`, `set2`, `set3`. |
| `stat_type` | Utf8 | `Service`, `Return`, `Points`, `Games`. |
| `stat_name` | Utf8 | e.g. `Aces`, `1st serve percentage`, `Break Points Saved`. |
| `stat_value_raw` | Utf8 | Verbatim, e.g. `"67%"`, `"0/4"`, `"4"`. |
| `stat_won` | Int32 | Numerator for ratio stats (e.g. break points converted). Null when the API doesn't report one. |
| `stat_total` | Int32 | Denominator for ratio stats. Null when not applicable. |
| `stat_value_pct` | Float32 | Parsed from `stat_value_raw` when it ends in `%`. Null otherwise. |

**Distinct `(stat_type, stat_name)` combinations observed in this dataset (22):**

- **Service (8):** `Aces`, `Double Faults`, `1st serve percentage`, `1st serve points won`, `2nd serve points won`, `Break Points Saved`, `Average 1st serve speed`, `Average 2nd serve speed`
- **Return (3):** `1st return points won`, `2nd return points won`, `Break Points Converted`
- **Points (8):** `Winners`, `Unforced errors`, `Net points won`, `Service Points Won`, `Return Points Won`, `Total Points Won`, `Match points saved`, `Last 10 balls`
- **Games (3):** `Total games won`, `Service games won`, `Return games won`

> The schema is open: any future stat the API adds will appear as new rows without any schema change.

---

## 5. `games_long.parquet` — 2,406,909 rows × 12 cols

One row per `(snapshot, match, set, game)`. Source: the `pointbypoint` array on each event (each entry is a game, not a set).

**Primary key:** `(ts_ns, event_key, set_index, number_game)`.

| Column | Type | Notes |
|---|---|---|
| `ts_ns` | Int64 | |
| `event_key` | Int64 | |
| `set_number_raw` | Utf8 | Verbatim, e.g. `"Set 1"`. |
| `set_index` | UInt8 | Parsed integer set number. |
| `number_game` | Int32 | 1-based game number within the set. |
| `player_served` | Utf8 | `"First Player"` / `"Second Player"`. |
| `serve_winner` | Utf8 | Side that won the game. |
| `serve_lost` | Utf8 | Often null; the side that lost serve when broken. |
| `score_raw` | Utf8 | Game-end score, e.g. `"0 - 1"`. |
| `score_a`, `score_b` | Int32 | Parsed from `score_raw`. |
| `n_points` | UInt32 | Number of `points_long` rows this game contributes. |

---

## 6. `points_long.parquet` — 10,891,508 rows × 11 cols

One row per `(snapshot, match, set, game, point)`. Source: the `points` array inside each `pointbypoint` game.

**Primary key:** `(ts_ns, event_key, set_index, number_game, number_point)`.

| Column | Type | Notes |
|---|---|---|
| `ts_ns` | Int64 | |
| `event_key` | Int64 | |
| `set_index` | UInt8 | Inherited from the parent game. |
| `number_game` | Int32 | Inherited from the parent game. |
| `number_point` | Int32 | 1-based point number within the game. |
| `score_raw` | Utf8 | Verbatim point score, e.g. `"0 - 15"`, `"40 - A"`. |
| `point_a`, `point_b` | Int16 | Parsed: `0/15/30/40/50` (`50` = advantage). Null for unrecognized tokens. |
| `break_point` | Boolean | Truthy when the API flags this point as a break point. Null = not flagged. |
| `set_point` | Boolean | Same convention. |
| `match_point` | Boolean | Same convention. |

> **Repetition warning:** because point history is replayed in every snapshot for which a match is live, the same logical point can appear under many distinct `ts_ns`. To get the *unique* points of a match, group by `(event_key, set_index, number_game, number_point)`. To get the latest known view, take the row with the max `ts_ns` per match (or per match+game).

---

---

## 7. `odds_long.parquet` — pre-match bookmaker odds (long format)

One row per `(ts_ns, event_key, market, outcome, line, bookmaker)`. Built by `./tools/fetch_odds.py`, which calls `get_odds` once per match date that appears in `events.parquet` and walks every market/outcome/line/bookmaker branch.

**Primary key:** `(ts_ns, event_key, market, outcome, line, bookmaker)`.

| Column | Type | Notes |
|---|---|---|
| `ts_ns` | Int64 | Wall-clock time of the `get_odds` fetch (same for every row of one fetch). |
| `event_key` | Int64 | API-Tennis match id. Joins to `events.event_key`. |
| `match_date` | Utf8 | `YYYY-MM-DD` — the `date_start` passed to `get_odds`. |
| `market` | Utf8 | Market category (see `markets_meta` for the full list). |
| `outcome` | Utf8 | Outcome label within the market (e.g. `"Home"`, `"Away"`, `"Over"`, `"2:0"`). Null for single-outcome markets. |
| `line` | Utf8 | Handicap / total line where applicable (e.g. `"-1.5"`, `"22.5"`). Null when the market has no line. |
| `bookmaker` | Utf8 | Bookmaker name. Observed values include `bet365`, `1xBet`, `Marathon`, `Unibet`, `Betfair`, `WilliamHill`, `10Bet`, `Betano`, `888Sport`, `Pncl`, `BetVictor`, `Betsson`, `Sportingbet`, `Betcris`, `bwin`. |
| `odds` | Float64 | Decimal odds (e.g. `2.04`). Null when the source string didn't parse. |
| `in_events` | Boolean | True if this `event_key` also appears in `events.parquet`. Useful for filtering down to matches we actually have live data for. |

> **Cardinality this dataset:** ~61k rows from one fetch of one date (2026-05-12), covering 198 matches × 28 markets × ~9 bookmakers (sparse). 28 markets observed — see `markets_meta` for the breakdown.

---

## 8. `odds_captures.parquet` — one row per `get_odds` fetch

A tiny ledger of when odds were pulled. Lets you slice `odds_long` by capture and trace pre-match odds evolution if you re-fetch over time.

**Primary key:** `(ts_ns, date_start)`.

| Column | Type | Notes |
|---|---|---|
| `ts_ns` | Int64 | Fetch time. Matches `odds_long.ts_ns` exactly. |
| `date_start` | Utf8 | Date passed to the API. |
| `date_stop` | Utf8 | Same as `date_start` in this run (single-day fetches). |
| `n_matches` | UInt32 | Number of matches returned by the API for that date. |
| `n_odds_rows` | UInt32 | Number of `odds_long` rows the fetch produced. |

> To track odds movement, re-run `fetch_odds.py` periodically; each run appends a new `ts_ns` capture across both tables.

---

## 9. `markets_meta.parquet` — derived market summary

One row per `market` observed in `odds_long`, with coverage counts. Useful for picking which market to model.

| Column | Type | Meaning |
|---|---|---|
| `market` | Utf8 | Market name. |
| `n_matches` | UInt32 | Distinct `event_key` values that had this market. |
| `n_outcomes` | UInt32 | Distinct outcome labels under this market. |
| `n_lines` | UInt32 | Distinct line values (1 if the market has no line). |
| `n_bookmakers` | UInt32 | Distinct bookmakers offering this market. |
| `has_line` | Boolean | Does any row of this market have a non-null `line`. |
| `n_rows` | UInt32 | Total `odds_long` rows under this market. |

---

## 10. `orderbook_l2.parquet` — Kalshi raw L2 long table

One row per `(ts_ns, ticker, side, price_cents)` — full price ladder, lossless. Built by `./tools/build_kalshi_orderbook.py` from the raw poll-recorder NDJSON (`Kalshi_Tennis/Research_2026_05/atp_rome_orderbook_*.ndjson`).

**Primary key:** `(ts_ns, ticker, side, price_cents)`.

| Column | Type | Notes |
|---|---|---|
| `ts_ns` | Int64 | Kalshi poll time. |
| `ticker` | Utf8 | Kalshi match-side ticker, e.g. `KXATPMATCH-26MAY06BASMER-MER`. Last 3 chars = side player code; the rest is the match stem. |
| `side` | Categorical | `"yes"` or `"no"` — which side of the Kalshi binary contract these are **bids** for. (Kalshi reports both sides as resting bids; YES asks are derived from NO bids.) |
| `price_cents` | Int16 | Price in cents, ∈ [1, 99]. Parsed from the dollar-string price in the raw feed. |
| `size` | Float64 | Resting size at that level (Kalshi `count_fp`, can be fractional). |

**Yes-equivalent quoting recipe** (the convention used by the rest of the Kalshi tooling, e.g. `Kalshi_Tennis/Tools/data_transform.py`):

- *Best YES bid* = max price among rows with `side="yes"`.
- *Best YES ask in cents* = `100 - max(price_cents)` among rows with `side="no"` (because a 39¢ resting NO bid is equivalently a 61¢ offer to sell YES).
- *Best NO bid* and *Best NO ask* by symmetry.

> Row count this dataset: **22.4M** L2 rows from 494k snapshots over the 2026-05-08 → 2026-05-12 window. On disk: ~32 MB (~75× smaller than the 524 MB source NDJSON).

---

## 11. `orderbook_top.parquet` — Kalshi top-of-book + depth summary (derived)

One row per `(ts_ns, ticker)`. Materialised by the same builder so that common questions ("what's the mid?", "how thick was the book?") don't need a group-by over the L2 table.

**Primary key:** `(ts_ns, ticker)`.

| Column | Type | Notes |
|---|---|---|
| `ts_ns` | Int64 | |
| `ticker` | Utf8 | |
| `best_yes_bid_cents` | Int16 | Best (= highest) YES-side resting bid price in cents. |
| `best_yes_bid_size` | Float64 | Size at that level. |
| `best_no_bid_cents` | Int16 | Best (= highest) NO-side resting bid price in cents. |
| `best_no_bid_size` | Float64 | Size at that level. |
| `yes_ask_cents` | Int16 | `100 - best_no_bid_cents` — yes-equivalent best ask. |
| `yes_mid_cents` | Float32 | `(best_yes_bid + yes_ask) / 2`. |
| `yes_spread_cents` | Int16 | `yes_ask - best_yes_bid`. |
| `yes_total_size` | Float64 | Sum of sizes across all YES-side levels. |
| `no_total_size` | Float64 | Sum of sizes across all NO-side levels. |
| `yes_n_levels` | Int32 | Count of YES-side levels in this snapshot. |
| `no_n_levels` | Int32 | Count of NO-side levels in this snapshot. |

**Consistency check (built into the builder):** `sum(yes_n_levels + no_n_levels)` over `orderbook_top` equals `orderbook_l2` row count exactly.

> Row count: **494,102** snapshots × tickers. On disk: ~6.4 MB.

---

## 12. `live/` — WebSocket segments (incremental)

Live-streamed data goes into rolling per-window segment files under `tables/live/`. Each table has its own subdirectory and one file per flush window (default 60 s) is written by `./tools/ws_daemon.py`:

```
tables/live/snapshots/seg_<YYYYMMDDTHHMMSSZ>.parquet
tables/live/events/seg_<YYYYMMDDTHHMMSSZ>.parquet
tables/live/scores_long/seg_<YYYYMMDDTHHMMSSZ>.parquet
tables/live/stats_long/seg_<YYYYMMDDTHHMMSSZ>.parquet
tables/live/games_long/seg_<YYYYMMDDTHHMMSSZ>.parquet
tables/live/points_long/seg_<YYYYMMDDTHHMMSSZ>.parquet
```

**Schemas are identical to the six REST-derived tables (1–6 above).** The only operational differences are:

- `ts_ns` is the **arrival time** of the WS frame (the daemon stamps it with `time.time_ns()` on receipt), not a server-side timestamp.
- Reading the full live stream is just `pl.scan_parquet("tables/live/events/*.parquet")` — polars handles the union transparently. Merge with the historic tables by `pl.concat([scan_static, scan_live])`.
- On clean exit (Ctrl-C / SIGTERM) the daemon flushes its pending buffer to `seg_<...>_final.parquet`.

---

## Example queries

```python
import polars as pl

ev = pl.scan_parquet("events.parquet")
st = pl.scan_parquet("stats_long.parquet")
pt = pl.scan_parquet("points_long.parquet")

# Final scores of every completed match
(ev.filter(pl.col("is_finished"))
   .unique(subset=["event_key"], keep="last")
   .select(["event_key","event_first_player","event_second_player",
            "score_summary","event_winner"])
   .collect())

# Match-level ace counts per player, latest snapshot only
(st.filter((pl.col("stat_period") == "match") & (pl.col("stat_name") == "Aces"))
   .group_by(["event_key","player_key"])
   .agg(pl.col("ts_ns").max().alias("ts_ns"),
        pl.col("stat_value_raw").last().alias("aces"))
   .collect())

# Deduplicated point-by-point trajectory of one match
(pt.filter(pl.col("event_key") == 12126512)
   .unique(subset=["set_index","number_game","number_point"])
   .sort(["set_index","number_game","number_point"])
   .collect())
```

## Schema invariants (verified at build time)

- `events.n_stat_rows.sum()` == `stats_long` row count (14,859,183)
- `events.n_pbp_games.sum()` == `games_long` row count (2,406,909)
- `events.n_sets.sum()` == `scores_long` row count (329,849)
- All 201,221 events in the source parsed successfully (`bad_raw=0`).
- `orderbook_top.(yes_n_levels + no_n_levels).sum()` == `orderbook_l2` row count (22,435,862).

---

## Tools (all in `./tools/`)

| Script | Inputs | Outputs |
|---|---|---|
| `unpack.py` | `combined.ndjson` (concatenated REST-poll dumps) | `snapshots`, `events`, `scores_long`, `stats_long`, `games_long`, `points_long`. |
| `fetch_odds.py` | API key + `events.parquet` (for the date list) | `odds_long`, `odds_captures`, `markets_meta`. One-shot pull; re-run to add a new `ts_ns` capture. |
| `build_kalshi_orderbook.py` | `Kalshi_Tennis/Research_2026_05/atp_rome_orderbook_*.ndjson` (or any path glob via positional args) | `orderbook_l2`, `orderbook_top`. Streams batches through `pyarrow.parquet.ParquetWriter` so memory stays bounded. |
| `ws_daemon.py` | `wss://wss.api-tennis.com/live?APIkey=...` (long-running) | `live/<table>/seg_<ts>.parquet`. Reconnects with exponential backoff; flushes on signal exit. |

### Environment

Set these before running any of the tools:

| Variable | Required by | Purpose |
|---|---|---|
| `API_TENNIS_KEY` | `fetch_odds.py`, `ws_daemon.py` | api-tennis.com API key. |
| `TENNIS_ROOT` | all four | Root that holds `combined.ndjson`, `tables/`, and (optionally) `Data/Orderbook/`. Defaults to the repo root, so the tools work in-place after `git clone`. |
| `KALSHI_ORDERBOOK_GLOB` | `build_kalshi_orderbook.py` | Glob of raw NDJSON dumps to ingest. Defaults to `$TENNIS_ROOT/Data/Orderbook/*.ndjson`. |

### Run examples

```bash
export API_TENNIS_KEY=...
# optional: export TENNIS_ROOT=/path/to/data/root

# Rebuild static REST tables from combined.ndjson
python3 tools/unpack.py

# Fetch pre-match odds for every date present in events.parquet
python3 tools/fetch_odds.py

# Convert Kalshi raw NDJSON dumps to L2 + top-of-book parquet
python3 tools/build_kalshi_orderbook.py
# ...or point it at a specific file:
python3 tools/build_kalshi_orderbook.py /path/to/some_orderbook.ndjson

# Start the WebSocket daemon (default flush window 60 s; Ctrl-C to stop)
python3 tools/ws_daemon.py
# Optionally filter by tournament / match / player or change the flush cadence:
python3 tools/ws_daemon.py --tournament-key 1234 --flush-secs 30
```

### Dependencies

Python 3.10+ and:

```
polars   pyarrow   websockets
```

"""Unpack combined.ndjson into a star-schema of Parquet tables.

Input : $TENNIS_ROOT/combined.ndjson  (one {ts_ns, raw} record per line; raw is a stringified JSON list of events)
Output: $TENNIS_ROOT/tables/{snapshots,events,scores_long,stats_long,games_long,points_long}.parquet
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(os.environ.get("TENNIS_ROOT", Path(__file__).resolve().parent.parent))
SRC = ROOT / "combined.ndjson"
OUT = ROOT / "tables"
OUT.mkdir(exist_ok=True)


# ---------- parsers ----------

_GAME_POINT_MAP = {"0": 0, "15": 15, "30": 30, "40": 40, "A": 50, "AD": 50}


def parse_game_point(s):
    """'40' -> 40, 'A' -> 50, None/'' -> None."""
    if s is None:
        return None
    s = s.strip()
    return _GAME_POINT_MAP.get(s)  # returns None for unknown tokens (e.g. tiebreak '7')


def parse_int_safe(s):
    if s is None:
        return None
    if isinstance(s, int):
        return s
    s = str(s).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


_PAIR_RE = re.compile(r"^\s*([^\s-]+)\s*-\s*([^\s-]+)\s*$")


def parse_pair(s):
    """'1 - 0' -> ('1', '0'); '40 - A' -> ('40', 'A'); '0 - 15' -> ('0','15')."""
    if not s:
        return None, None
    m = _PAIR_RE.match(s)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def parse_pair_ints(s):
    a, b = parse_pair(s)
    return parse_int_safe(a), parse_int_safe(b)


def parse_pair_game_points(s):
    a, b = parse_pair(s)
    return parse_game_point(a), parse_game_point(b)


def parse_pct(s):
    """'67%' -> 67.0; '0%' -> 0.0; None/'' -> None; '0/0' -> None (handled by stat_won/total)."""
    if s is None:
        return None
    s = str(s).strip()
    if not s or s.endswith("/0") or s in ("-", "--"):
        return None
    if s.endswith("%"):
        try:
            return float(s[:-1])
        except ValueError:
            return None
    return None


def parse_bool_str(s):
    """'1'/'True'/'true' -> True; '0'/'False' -> False; else None."""
    if s is None:
        return None
    if isinstance(s, bool):
        return s
    s = str(s).strip().lower()
    if s in ("1", "true", "yes"):
        return True
    if s in ("0", "false", "no"):
        return False
    return None


_SET_NUM_RE = re.compile(r"(\d+)")


def parse_set_index(s):
    """'Set 1' -> 1; '1' -> 1; None -> None."""
    if s is None:
        return None
    m = _SET_NUM_RE.search(str(s))
    return int(m.group(1)) if m else None


def encode_serve(s):
    """'First Player' -> 1; 'Second Player' -> 2; else None."""
    if not s:
        return None
    s = s.strip().lower()
    if s.startswith("first"):
        return 1
    if s.startswith("second"):
        return 2
    return None


def status_code(s):
    if not s:
        return None
    s = s.strip().lower().replace(" ", "")
    return s  # e.g. 'set1', 'set2', 'finished', 'interrupted'


def score_summary(scores):
    """[{score_set:'1', score_first:'6', score_second:'4'}, ...] -> '6-4,3-6'."""
    if not scores:
        return None
    try:
        sorted_scores = sorted(scores, key=lambda s: parse_set_index(s.get("score_set")) or 0)
    except Exception:
        sorted_scores = scores
    parts = []
    for s in sorted_scores:
        a = s.get("score_first") or "?"
        b = s.get("score_second") or "?"
        parts.append(f"{a}-{b}")
    return ",".join(parts)


def truthy_to_bool(v):
    """For break_point/set_point/match_point: API uses 'True'/None/sometimes the side name."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip()
    if not s:
        return None
    sl = s.lower()
    if sl in ("true", "1"):
        return True
    if sl in ("false", "0"):
        return False
    # Sometimes the value is non-null (e.g. "First Player") meaning the point is flagged.
    return True


# ---------- column buffers ----------


def new_buf(cols):
    return {c: [] for c in cols}


SNAP_COLS = ["ts_ns", "n_events"]
EVENT_COLS = [
    "ts_ns",
    "event_key",
    "event_date",
    "event_time",
    "tournament_key",
    "tournament_name",
    "tournament_round",
    "tournament_season",
    "event_type_type",
    "event_qualification",
    "event_live",
    "first_player_key",
    "second_player_key",
    "event_first_player",
    "event_second_player",
    "event_first_player_logo",
    "event_second_player_logo",
    "event_winner",
    "event_status_raw",
    "event_status_code",
    "is_finished",
    "is_interrupted",
    "event_serve_raw",
    "serve_side",
    "event_final_result_raw",
    "sets_won_a",
    "sets_won_b",
    "event_game_result_raw",
    "game_point_a",
    "game_point_b",
    "score_summary",
    "n_sets",
    "n_stat_rows",
    "n_pbp_games",
]
SCORE_COLS = ["ts_ns", "event_key", "set_index", "score_first", "score_second", "score_set_raw"]
STAT_COLS = [
    "ts_ns",
    "event_key",
    "player_key",
    "stat_period",
    "stat_type",
    "stat_name",
    "stat_value_raw",
    "stat_won",
    "stat_total",
    "stat_value_pct",
]
GAME_COLS = [
    "ts_ns",
    "event_key",
    "set_number_raw",
    "set_index",
    "number_game",
    "player_served",
    "serve_winner",
    "serve_lost",
    "score_raw",
    "score_a",
    "score_b",
    "n_points",
]
POINT_COLS = [
    "ts_ns",
    "event_key",
    "set_index",
    "number_game",
    "number_point",
    "score_raw",
    "point_a",
    "point_b",
    "break_point",
    "set_point",
    "match_point",
]


# ---------- main pass ----------


def main():
    if not SRC.exists():
        print(f"missing {SRC}", file=sys.stderr)
        sys.exit(1)

    snaps = new_buf(SNAP_COLS)
    events = new_buf(EVENT_COLS)
    scores_b = new_buf(SCORE_COLS)
    stats_b = new_buf(STAT_COLS)
    games_b = new_buf(GAME_COLS)
    points_b = new_buf(POINT_COLS)

    n_lines = 0
    n_events = 0
    n_bad_raw = 0
    t0 = time.time()

    with SRC.open() as fp:
        for line in fp:
            n_lines += 1
            try:
                rec = json.loads(line)
                ts_ns = rec["ts_ns"]
                raw_str = rec["raw"]
                raw = json.loads(raw_str) if isinstance(raw_str, str) else raw_str
            except Exception:
                n_bad_raw += 1
                continue

            if not isinstance(raw, list):
                # Occasionally the API may return error objects; skip but count.
                n_bad_raw += 1
                continue

            snaps["ts_ns"].append(ts_ns)
            snaps["n_events"].append(len(raw))

            for ev in raw:
                n_events += 1

                # ---- scalar event row ----
                fr_raw = ev.get("event_final_result")
                sets_a, sets_b = parse_pair_ints(fr_raw)
                gr_raw = ev.get("event_game_result")
                gp_a, gp_b = parse_pair_game_points(gr_raw)
                stt_raw = ev.get("event_status")
                stt_code = status_code(stt_raw)
                scores = ev.get("scores") or []
                stats = ev.get("statistics") or []
                pbp = ev.get("pointbypoint") or []

                events["ts_ns"].append(ts_ns)
                events["event_key"].append(ev.get("event_key"))
                events["event_date"].append(ev.get("event_date"))
                events["event_time"].append(ev.get("event_time"))
                events["tournament_key"].append(ev.get("tournament_key"))
                events["tournament_name"].append(ev.get("tournament_name"))
                events["tournament_round"].append(ev.get("tournament_round"))
                events["tournament_season"].append(ev.get("tournament_season"))
                events["event_type_type"].append(ev.get("event_type_type"))
                events["event_qualification"].append(parse_bool_str(ev.get("event_qualification")))
                events["event_live"].append(parse_bool_str(ev.get("event_live")))
                events["first_player_key"].append(ev.get("first_player_key"))
                events["second_player_key"].append(ev.get("second_player_key"))
                events["event_first_player"].append(ev.get("event_first_player"))
                events["event_second_player"].append(ev.get("event_second_player"))
                events["event_first_player_logo"].append(ev.get("event_first_player_logo"))
                events["event_second_player_logo"].append(ev.get("event_second_player_logo"))
                events["event_winner"].append(ev.get("event_winner"))
                events["event_status_raw"].append(stt_raw)
                events["event_status_code"].append(stt_code)
                events["is_finished"].append(stt_code == "finished")
                events["is_interrupted"].append(stt_code == "interrupted")
                events["event_serve_raw"].append(ev.get("event_serve"))
                events["serve_side"].append(encode_serve(ev.get("event_serve")))
                events["event_final_result_raw"].append(fr_raw)
                events["sets_won_a"].append(sets_a)
                events["sets_won_b"].append(sets_b)
                events["event_game_result_raw"].append(gr_raw)
                events["game_point_a"].append(gp_a)
                events["game_point_b"].append(gp_b)
                events["score_summary"].append(score_summary(scores))
                events["n_sets"].append(len(scores))
                events["n_stat_rows"].append(len(stats))
                events["n_pbp_games"].append(len(pbp))

                ev_key = ev.get("event_key")

                # ---- scores_long ----
                for s in scores:
                    scores_b["ts_ns"].append(ts_ns)
                    scores_b["event_key"].append(ev_key)
                    scores_b["set_index"].append(parse_set_index(s.get("score_set")))
                    scores_b["score_first"].append(parse_int_safe(s.get("score_first")))
                    scores_b["score_second"].append(parse_int_safe(s.get("score_second")))
                    scores_b["score_set_raw"].append(s.get("score_set"))

                # ---- stats_long ----
                for st in stats:
                    stats_b["ts_ns"].append(ts_ns)
                    stats_b["event_key"].append(ev_key)
                    stats_b["player_key"].append(st.get("player_key"))
                    stats_b["stat_period"].append(st.get("stat_period"))
                    stats_b["stat_type"].append(st.get("stat_type"))
                    stats_b["stat_name"].append(st.get("stat_name"))
                    sv = st.get("stat_value")
                    stats_b["stat_value_raw"].append(None if sv is None else str(sv))
                    stats_b["stat_won"].append(parse_int_safe(st.get("stat_won")))
                    stats_b["stat_total"].append(parse_int_safe(st.get("stat_total")))
                    stats_b["stat_value_pct"].append(parse_pct(sv))

                # ---- games_long & points_long ----
                for g in pbp:
                    set_raw = g.get("set_number")
                    set_idx = parse_set_index(set_raw)
                    num_game = parse_int_safe(g.get("number_game"))
                    g_score_raw = g.get("score")
                    g_a, g_b = parse_pair_ints(g_score_raw)
                    pts = g.get("points") or []

                    games_b["ts_ns"].append(ts_ns)
                    games_b["event_key"].append(ev_key)
                    games_b["set_number_raw"].append(set_raw)
                    games_b["set_index"].append(set_idx)
                    games_b["number_game"].append(num_game)
                    games_b["player_served"].append(g.get("player_served"))
                    games_b["serve_winner"].append(g.get("serve_winner"))
                    games_b["serve_lost"].append(g.get("serve_lost"))
                    games_b["score_raw"].append(g_score_raw)
                    games_b["score_a"].append(g_a)
                    games_b["score_b"].append(g_b)
                    games_b["n_points"].append(len(pts))

                    for p in pts:
                        p_score_raw = p.get("score")
                        p_a, p_b = parse_pair_game_points(p_score_raw)
                        points_b["ts_ns"].append(ts_ns)
                        points_b["event_key"].append(ev_key)
                        points_b["set_index"].append(set_idx)
                        points_b["number_game"].append(num_game)
                        points_b["number_point"].append(parse_int_safe(p.get("number_point")))
                        points_b["score_raw"].append(p_score_raw)
                        points_b["point_a"].append(p_a)
                        points_b["point_b"].append(p_b)
                        points_b["break_point"].append(truthy_to_bool(p.get("break_point")))
                        points_b["set_point"].append(truthy_to_bool(p.get("set_point")))
                        points_b["match_point"].append(truthy_to_bool(p.get("match_point")))

            if n_lines % 500 == 0:
                dt = time.time() - t0
                print(
                    f"  [{dt:6.1f}s] {n_lines} snapshots, {n_events} events, "
                    f"{len(stats_b['ts_ns'])} stat rows, {len(points_b['ts_ns'])} point rows",
                    flush=True,
                )

    print(
        f"done parsing: {n_lines} snapshots, {n_events} events, "
        f"bad_raw={n_bad_raw}, elapsed={time.time()-t0:.1f}s"
    )

    # ---- schemas (be explicit so empties / all-null cols don't get weird types) ----
    schemas = {
        "snapshots": {"ts_ns": pl.Int64, "n_events": pl.UInt32},
        "events": {
            "ts_ns": pl.Int64,
            "event_key": pl.Int64,
            "event_date": pl.Utf8,
            "event_time": pl.Utf8,
            "tournament_key": pl.Int64,
            "tournament_name": pl.Utf8,
            "tournament_round": pl.Utf8,
            "tournament_season": pl.Utf8,
            "event_type_type": pl.Utf8,
            "event_qualification": pl.Boolean,
            "event_live": pl.Boolean,
            "first_player_key": pl.Int64,
            "second_player_key": pl.Int64,
            "event_first_player": pl.Utf8,
            "event_second_player": pl.Utf8,
            "event_first_player_logo": pl.Utf8,
            "event_second_player_logo": pl.Utf8,
            "event_winner": pl.Utf8,
            "event_status_raw": pl.Utf8,
            "event_status_code": pl.Utf8,
            "is_finished": pl.Boolean,
            "is_interrupted": pl.Boolean,
            "event_serve_raw": pl.Utf8,
            "serve_side": pl.UInt8,
            "event_final_result_raw": pl.Utf8,
            "sets_won_a": pl.Int32,
            "sets_won_b": pl.Int32,
            "event_game_result_raw": pl.Utf8,
            "game_point_a": pl.Int16,
            "game_point_b": pl.Int16,
            "score_summary": pl.Utf8,
            "n_sets": pl.UInt32,
            "n_stat_rows": pl.UInt32,
            "n_pbp_games": pl.UInt32,
        },
        "scores_long": {
            "ts_ns": pl.Int64,
            "event_key": pl.Int64,
            "set_index": pl.UInt8,
            "score_first": pl.Int32,
            "score_second": pl.Int32,
            "score_set_raw": pl.Utf8,
        },
        "stats_long": {
            "ts_ns": pl.Int64,
            "event_key": pl.Int64,
            "player_key": pl.Int64,
            "stat_period": pl.Utf8,
            "stat_type": pl.Utf8,
            "stat_name": pl.Utf8,
            "stat_value_raw": pl.Utf8,
            "stat_won": pl.Int32,
            "stat_total": pl.Int32,
            "stat_value_pct": pl.Float32,
        },
        "games_long": {
            "ts_ns": pl.Int64,
            "event_key": pl.Int64,
            "set_number_raw": pl.Utf8,
            "set_index": pl.UInt8,
            "number_game": pl.Int32,
            "player_served": pl.Utf8,
            "serve_winner": pl.Utf8,
            "serve_lost": pl.Utf8,
            "score_raw": pl.Utf8,
            "score_a": pl.Int32,
            "score_b": pl.Int32,
            "n_points": pl.UInt32,
        },
        "points_long": {
            "ts_ns": pl.Int64,
            "event_key": pl.Int64,
            "set_index": pl.UInt8,
            "number_game": pl.Int32,
            "number_point": pl.Int32,
            "score_raw": pl.Utf8,
            "point_a": pl.Int16,
            "point_b": pl.Int16,
            "break_point": pl.Boolean,
            "set_point": pl.Boolean,
            "match_point": pl.Boolean,
        },
    }

    bufs = {
        "snapshots": snaps,
        "events": events,
        "scores_long": scores_b,
        "stats_long": stats_b,
        "games_long": games_b,
        "points_long": points_b,
    }

    for name, buf in bufs.items():
        t = time.time()
        df = pl.DataFrame(buf, schema=schemas[name], strict=False)
        path = OUT / f"{name}.parquet"
        df.write_parquet(path, compression="zstd", compression_level=6, statistics=True)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"wrote {name}: rows={df.height:,} cols={df.width} -> {path.name} ({size_mb:.1f} MB) in {time.time()-t:.1f}s")


if __name__ == "__main__":
    main()

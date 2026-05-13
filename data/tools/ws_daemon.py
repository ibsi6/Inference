"""Long-running api-tennis WebSocket daemon.

Connects to:    wss://wss.api-tennis.com/live?APIkey=<KEY>[&...filters]
Parses each frame (same event-object shape as `get_livescore`) and appends
into the same star-schema parquet layout produced by unpack.py:

    snapshots, events, scores_long, stats_long, games_long, points_long

Layout on disk (rolling segments — Parquet does not support append in place):

    tables/live/<table>/seg_<ts_iso>.parquet

Each "segment" is the rows captured during one flush window (default 60s).
Reading the live data later is just `pl.scan_parquet("tables/live/events/*.parquet")`.

Run:
    python ws_daemon.py [--apikey KEY] [--flush-secs 60] [--out-dir DIR]
                        [--tournament-key K] [--match-key K] [--player-key K]
                        [--timezone TZ]

CTRL-C flushes the in-memory buffer and exits cleanly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import polars as pl
import websockets
from websockets.exceptions import ConnectionClosed

# ---- defaults ----

DEFAULT_OUT_DIR = Path(
    os.environ.get("TENNIS_ROOT", Path(__file__).resolve().parent.parent)
) / "tables" / "live"
DEFAULT_FLUSH_SECS = 60
DEFAULT_WS_URL = "wss://wss.api-tennis.com/live"
DEFAULT_APIKEY = os.environ.get("API_TENNIS_KEY")

TABLES = ("snapshots", "events", "scores_long", "stats_long", "games_long", "points_long")


# ---- parsing helpers (mirror unpack.py exactly) ----

import re

_GAME_POINT_MAP = {"0": 0, "15": 15, "30": 30, "40": 40, "A": 50, "AD": 50}
_PAIR_RE = re.compile(r"^\s*([^\s-]+)\s*-\s*([^\s-]+)\s*$")
_SET_NUM_RE = re.compile(r"(\d+)")


def _gpt(s):
    if s is None:
        return None
    return _GAME_POINT_MAP.get(s.strip())


def _intsafe(s):
    if s is None:
        return None
    if isinstance(s, int):
        return s
    try:
        return int(str(s).strip())
    except (ValueError, TypeError):
        return None


def _pair(s):
    if not s:
        return None, None
    m = _PAIR_RE.match(s)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def _pair_ints(s):
    a, b = _pair(s)
    return _intsafe(a), _intsafe(b)


def _pair_gp(s):
    a, b = _pair(s)
    return _gpt(a) if a else None, _gpt(b) if b else None


def _pct(s):
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


def _boolstr(s):
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


def _set_idx(s):
    if s is None:
        return None
    m = _SET_NUM_RE.search(str(s))
    return int(m.group(1)) if m else None


def _serve(s):
    if not s:
        return None
    s = s.strip().lower()
    if s.startswith("first"):
        return 1
    if s.startswith("second"):
        return 2
    return None


def _status_code(s):
    if not s:
        return None
    return s.strip().lower().replace(" ", "")


def _truthy(v):
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
    return True


def _score_summary(scores):
    if not scores:
        return None
    try:
        scores = sorted(scores, key=lambda s: _set_idx(s.get("score_set")) or 0)
    except Exception:
        pass
    return ",".join(f"{s.get('score_first') or '?'}-{s.get('score_second') or '?'}" for s in scores)


# ---- buffers ----

def _new_bufs():
    return {
        "snapshots": {"ts_ns": [], "n_events": []},
        "events": {k: [] for k in [
            "ts_ns","event_key","event_date","event_time","tournament_key","tournament_name",
            "tournament_round","tournament_season","event_type_type","event_qualification","event_live",
            "first_player_key","second_player_key","event_first_player","event_second_player",
            "event_first_player_logo","event_second_player_logo","event_winner",
            "event_status_raw","event_status_code","is_finished","is_interrupted",
            "event_serve_raw","serve_side","event_final_result_raw","sets_won_a","sets_won_b",
            "event_game_result_raw","game_point_a","game_point_b","score_summary",
            "n_sets","n_stat_rows","n_pbp_games",
        ]},
        "scores_long": {k: [] for k in
            ["ts_ns","event_key","set_index","score_first","score_second","score_set_raw"]},
        "stats_long": {k: [] for k in
            ["ts_ns","event_key","player_key","stat_period","stat_type","stat_name",
             "stat_value_raw","stat_won","stat_total","stat_value_pct"]},
        "games_long": {k: [] for k in
            ["ts_ns","event_key","set_number_raw","set_index","number_game","player_served",
             "serve_winner","serve_lost","score_raw","score_a","score_b","n_points"]},
        "points_long": {k: [] for k in
            ["ts_ns","event_key","set_index","number_game","number_point","score_raw",
             "point_a","point_b","break_point","set_point","match_point"]},
    }


SCHEMAS = {
    "snapshots": {"ts_ns": pl.Int64, "n_events": pl.UInt32},
    "events": {
        "ts_ns": pl.Int64, "event_key": pl.Int64, "event_date": pl.Utf8, "event_time": pl.Utf8,
        "tournament_key": pl.Int64, "tournament_name": pl.Utf8, "tournament_round": pl.Utf8,
        "tournament_season": pl.Utf8, "event_type_type": pl.Utf8,
        "event_qualification": pl.Boolean, "event_live": pl.Boolean,
        "first_player_key": pl.Int64, "second_player_key": pl.Int64,
        "event_first_player": pl.Utf8, "event_second_player": pl.Utf8,
        "event_first_player_logo": pl.Utf8, "event_second_player_logo": pl.Utf8,
        "event_winner": pl.Utf8, "event_status_raw": pl.Utf8, "event_status_code": pl.Utf8,
        "is_finished": pl.Boolean, "is_interrupted": pl.Boolean,
        "event_serve_raw": pl.Utf8, "serve_side": pl.UInt8,
        "event_final_result_raw": pl.Utf8, "sets_won_a": pl.Int32, "sets_won_b": pl.Int32,
        "event_game_result_raw": pl.Utf8, "game_point_a": pl.Int16, "game_point_b": pl.Int16,
        "score_summary": pl.Utf8,
        "n_sets": pl.UInt32, "n_stat_rows": pl.UInt32, "n_pbp_games": pl.UInt32,
    },
    "scores_long": {
        "ts_ns": pl.Int64, "event_key": pl.Int64, "set_index": pl.UInt8,
        "score_first": pl.Int32, "score_second": pl.Int32, "score_set_raw": pl.Utf8,
    },
    "stats_long": {
        "ts_ns": pl.Int64, "event_key": pl.Int64, "player_key": pl.Int64,
        "stat_period": pl.Utf8, "stat_type": pl.Utf8, "stat_name": pl.Utf8,
        "stat_value_raw": pl.Utf8, "stat_won": pl.Int32, "stat_total": pl.Int32,
        "stat_value_pct": pl.Float32,
    },
    "games_long": {
        "ts_ns": pl.Int64, "event_key": pl.Int64, "set_number_raw": pl.Utf8,
        "set_index": pl.UInt8, "number_game": pl.Int32, "player_served": pl.Utf8,
        "serve_winner": pl.Utf8, "serve_lost": pl.Utf8, "score_raw": pl.Utf8,
        "score_a": pl.Int32, "score_b": pl.Int32, "n_points": pl.UInt32,
    },
    "points_long": {
        "ts_ns": pl.Int64, "event_key": pl.Int64, "set_index": pl.UInt8,
        "number_game": pl.Int32, "number_point": pl.Int32, "score_raw": pl.Utf8,
        "point_a": pl.Int16, "point_b": pl.Int16,
        "break_point": pl.Boolean, "set_point": pl.Boolean, "match_point": pl.Boolean,
    },
}


def emit_event(bufs, ts_ns, ev):
    """Same per-event row emission as unpack.py."""
    fr_raw = ev.get("event_final_result")
    sets_a, sets_b = _pair_ints(fr_raw)
    gr_raw = ev.get("event_game_result")
    gp_a, gp_b = _pair_gp(gr_raw)
    stt_raw = ev.get("event_status")
    stt_code = _status_code(stt_raw)
    scores = ev.get("scores") or []
    stats = ev.get("statistics") or []
    pbp = ev.get("pointbypoint") or []

    eb = bufs["events"]
    eb["ts_ns"].append(ts_ns)
    eb["event_key"].append(ev.get("event_key"))
    eb["event_date"].append(ev.get("event_date"))
    eb["event_time"].append(ev.get("event_time"))
    eb["tournament_key"].append(ev.get("tournament_key"))
    eb["tournament_name"].append(ev.get("tournament_name"))
    eb["tournament_round"].append(ev.get("tournament_round"))
    eb["tournament_season"].append(ev.get("tournament_season"))
    eb["event_type_type"].append(ev.get("event_type_type"))
    eb["event_qualification"].append(_boolstr(ev.get("event_qualification")))
    eb["event_live"].append(_boolstr(ev.get("event_live")))
    eb["first_player_key"].append(ev.get("first_player_key"))
    eb["second_player_key"].append(ev.get("second_player_key"))
    eb["event_first_player"].append(ev.get("event_first_player"))
    eb["event_second_player"].append(ev.get("event_second_player"))
    eb["event_first_player_logo"].append(ev.get("event_first_player_logo"))
    eb["event_second_player_logo"].append(ev.get("event_second_player_logo"))
    eb["event_winner"].append(ev.get("event_winner"))
    eb["event_status_raw"].append(stt_raw)
    eb["event_status_code"].append(stt_code)
    eb["is_finished"].append(stt_code == "finished")
    eb["is_interrupted"].append(stt_code == "interrupted")
    eb["event_serve_raw"].append(ev.get("event_serve"))
    eb["serve_side"].append(_serve(ev.get("event_serve")))
    eb["event_final_result_raw"].append(fr_raw)
    eb["sets_won_a"].append(sets_a)
    eb["sets_won_b"].append(sets_b)
    eb["event_game_result_raw"].append(gr_raw)
    eb["game_point_a"].append(gp_a)
    eb["game_point_b"].append(gp_b)
    eb["score_summary"].append(_score_summary(scores))
    eb["n_sets"].append(len(scores))
    eb["n_stat_rows"].append(len(stats))
    eb["n_pbp_games"].append(len(pbp))

    ek = ev.get("event_key")

    sb = bufs["scores_long"]
    for s in scores:
        sb["ts_ns"].append(ts_ns)
        sb["event_key"].append(ek)
        sb["set_index"].append(_set_idx(s.get("score_set")))
        sb["score_first"].append(_intsafe(s.get("score_first")))
        sb["score_second"].append(_intsafe(s.get("score_second")))
        sb["score_set_raw"].append(s.get("score_set"))

    stb = bufs["stats_long"]
    for st in stats:
        stb["ts_ns"].append(ts_ns)
        stb["event_key"].append(ek)
        stb["player_key"].append(st.get("player_key"))
        stb["stat_period"].append(st.get("stat_period"))
        stb["stat_type"].append(st.get("stat_type"))
        stb["stat_name"].append(st.get("stat_name"))
        sv = st.get("stat_value")
        stb["stat_value_raw"].append(None if sv is None else str(sv))
        stb["stat_won"].append(_intsafe(st.get("stat_won")))
        stb["stat_total"].append(_intsafe(st.get("stat_total")))
        stb["stat_value_pct"].append(_pct(sv))

    gb = bufs["games_long"]
    pb = bufs["points_long"]
    for g in pbp:
        set_raw = g.get("set_number")
        set_idx = _set_idx(set_raw)
        num_game = _intsafe(g.get("number_game"))
        g_score_raw = g.get("score")
        g_a, g_b = _pair_ints(g_score_raw)
        pts = g.get("points") or []
        gb["ts_ns"].append(ts_ns)
        gb["event_key"].append(ek)
        gb["set_number_raw"].append(set_raw)
        gb["set_index"].append(set_idx)
        gb["number_game"].append(num_game)
        gb["player_served"].append(g.get("player_served"))
        gb["serve_winner"].append(g.get("serve_winner"))
        gb["serve_lost"].append(g.get("serve_lost"))
        gb["score_raw"].append(g_score_raw)
        gb["score_a"].append(g_a)
        gb["score_b"].append(g_b)
        gb["n_points"].append(len(pts))
        for p in pts:
            p_score_raw = p.get("score")
            p_a, p_b = _pair_gp(p_score_raw)
            pb["ts_ns"].append(ts_ns)
            pb["event_key"].append(ek)
            pb["set_index"].append(set_idx)
            pb["number_game"].append(num_game)
            pb["number_point"].append(_intsafe(p.get("number_point")))
            pb["score_raw"].append(p_score_raw)
            pb["point_a"].append(p_a)
            pb["point_b"].append(p_b)
            pb["break_point"].append(_truthy(p.get("break_point")))
            pb["set_point"].append(_truthy(p.get("set_point")))
            pb["match_point"].append(_truthy(p.get("match_point")))


def flush_segment(bufs, out_dir: Path, seg_iso: str) -> dict[str, int]:
    """Write each non-empty buffer to tables/live/<table>/seg_<iso>.parquet."""
    counts = {}
    for name in TABLES:
        buf = bufs[name]
        n = len(buf["ts_ns"])
        if n == 0:
            counts[name] = 0
            continue
        df = pl.DataFrame(buf, schema=SCHEMAS[name], strict=False)
        sub = out_dir / name
        sub.mkdir(parents=True, exist_ok=True)
        path = sub / f"seg_{seg_iso}.parquet"
        df.write_parquet(path, compression="zstd", compression_level=6, statistics=True)
        counts[name] = n
    return counts


# ---- WS driver ----

def _coerce_events(payload):
    """The WS may push either a single event object or a list of them per frame."""
    if isinstance(payload, list):
        return [e for e in payload if isinstance(e, dict)]
    if isinstance(payload, dict):
        # Some providers wrap as {"event": {...}} or {"data": [...]}. Be permissive.
        for key in ("data", "events", "result"):
            inner = payload.get(key)
            if isinstance(inner, list):
                return [e for e in inner if isinstance(e, dict)]
        if "event_key" in payload:
            return [payload]
    return []


async def run(args):
    qs = {"APIkey": args.apikey, "timezone": args.timezone}
    for opt in ("tournament_key", "match_key", "player_key"):
        v = getattr(args, opt)
        if v is not None:
            qs[opt] = v
    url = f"{args.ws_url}?{urlencode(qs)}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stop = asyncio.Event()

    def _sigterm(*_):
        print("\n[ws] caught signal — flushing and exiting", flush=True)
        stop.set()

    signal.signal(signal.SIGINT, _sigterm)
    signal.signal(signal.SIGTERM, _sigterm)

    backoff = 1.0
    while not stop.is_set():
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                print(f"[ws] connected to {args.ws_url} (timezone={args.timezone})", flush=True)
                backoff = 1.0

                bufs = _new_bufs()
                seg_start = time.time()
                seg_iso_start = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                seg_events = 0
                seg_msgs = 0
                total_msgs = 0

                while not stop.is_set():
                    timeout = max(0.5, args.flush_secs - (time.time() - seg_start))
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    except asyncio.TimeoutError:
                        msg = None
                    except ConnectionClosed:
                        raise

                    if msg is not None:
                        try:
                            payload = json.loads(msg)
                        except json.JSONDecodeError:
                            print(f"[ws] non-JSON frame ignored ({len(msg)} bytes)", flush=True)
                            payload = None

                        if payload is not None:
                            ts = time.time_ns()
                            events = _coerce_events(payload)
                            if events:
                                bufs["snapshots"]["ts_ns"].append(ts)
                                bufs["snapshots"]["n_events"].append(len(events))
                                for ev in events:
                                    emit_event(bufs, ts, ev)
                                seg_events += len(events)
                            seg_msgs += 1
                            total_msgs += 1

                    if time.time() - seg_start >= args.flush_secs:
                        counts = flush_segment(bufs, out_dir, seg_iso_start)
                        print(
                            f"[ws] seg {seg_iso_start}: msgs={seg_msgs} events={seg_events} "
                            f"-> {counts}",
                            flush=True,
                        )
                        bufs = _new_bufs()
                        seg_start = time.time()
                        seg_iso_start = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                        seg_events = 0
                        seg_msgs = 0

                # final flush before exit
                counts = flush_segment(bufs, out_dir, seg_iso_start + "_final")
                print(f"[ws] final seg: {counts}; total_msgs={total_msgs}", flush=True)

        except (ConnectionClosed, OSError) as e:
            if stop.is_set():
                break
            print(f"[ws] disconnected ({e!r}); reconnecting in {backoff:.1f}s", flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ws-url", default=DEFAULT_WS_URL)
    ap.add_argument("--apikey", default=DEFAULT_APIKEY)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--flush-secs", type=int, default=DEFAULT_FLUSH_SECS)
    ap.add_argument("--tournament-key", default=None)
    ap.add_argument("--match-key", default=None)
    ap.add_argument("--player-key", default=None)
    ap.add_argument("--timezone", default="UTC")
    args = ap.parse_args()
    if not args.apikey:
        raise SystemExit("missing API key: set API_TENNIS_KEY in the environment or pass --apikey")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

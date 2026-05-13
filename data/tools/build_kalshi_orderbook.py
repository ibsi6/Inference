"""Convert raw Kalshi orderbook NDJSON to L2 long-format Parquet tables.

Input record shape (one per NDJSON line, as recorded by the Kalshi poller):
    {
      "ts_ns":   <int64 nanoseconds since epoch (poll time)>,
      "ticker":  "KXATPMATCH-26MAY06BASMER-MER",
      "response": {
        "orderbook_fp": {
          "yes_dollars": [["0.4500", "100.00"], ...],   # YES-side bids (price/size as strings)
          "no_dollars":  [["0.5400", "100.00"], ...]    # NO-side bids
        }
      }
    }

Kalshi quirk: prices are dollar strings in 1-cent increments (`"0.4500"`). We store
them as **integer cents** in [1, 99] to make the long table tight and indexable.

Both raw sides are kept as-is (one row per `(ts_ns, ticker, side, price_cents)`).
The "yes-equivalent" view (yes ask = 100 - best NO bid, etc.) is materialised in the
companion top-of-book table.

Outputs (relative to --out-dir, default $TENNIS_ROOT/tables/):
    orderbook_l2.parquet         long L2 rows (millions of them)
    orderbook_top.parquet        one row per snapshot with best/sizes/levels
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# ---- defaults ----

DEFAULT_OUT_DIR = Path(
    os.environ.get("TENNIS_ROOT", Path(__file__).resolve().parent.parent)
) / "tables"
DEFAULT_INPUT_GLOB = os.environ.get(
    "KALSHI_ORDERBOOK_GLOB",
    str(Path(os.environ.get("TENNIS_ROOT", Path(__file__).resolve().parent.parent))
        / "Data" / "Orderbook" / "*.ndjson"),
)

# ---- schemas ----

L2_SCHEMA = pa.schema([
    pa.field("ts_ns", pa.int64(), nullable=False),
    pa.field("ticker", pa.string(), nullable=False),
    pa.field("side", pa.dictionary(pa.int8(), pa.string()), nullable=False),   # "yes" | "no"
    pa.field("price_cents", pa.int16(), nullable=False),                       # 1..99
    pa.field("size", pa.float64(), nullable=False),
])

TOP_SCHEMA = pa.schema([
    pa.field("ts_ns", pa.int64(), nullable=False),
    pa.field("ticker", pa.string(), nullable=False),
    pa.field("best_yes_bid_cents", pa.int16()),     # max yes_dollars price
    pa.field("best_yes_bid_size", pa.float64()),
    pa.field("best_no_bid_cents", pa.int16()),      # max no_dollars price
    pa.field("best_no_bid_size", pa.float64()),
    pa.field("yes_ask_cents", pa.int16()),          # 100 - best_no_bid (yes-equivalent ask)
    pa.field("yes_mid_cents", pa.float32()),        # (best_yes_bid + yes_ask)/2
    pa.field("yes_spread_cents", pa.int16()),       # yes_ask - best_yes_bid
    pa.field("yes_total_size", pa.float64()),
    pa.field("no_total_size", pa.float64()),
    pa.field("yes_n_levels", pa.int32()),
    pa.field("no_n_levels", pa.int32()),
])

# ---- helpers ----

def _to_cents(price_str: str) -> int | None:
    """'0.4500' -> 45. Returns None on parse failure."""
    try:
        return int(round(float(price_str) * 100))
    except (TypeError, ValueError):
        return None


def _to_size(size_str) -> float | None:
    try:
        return float(size_str)
    except (TypeError, ValueError):
        return None


# ---- main ----

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="*", default=[],
                    help="One or more NDJSON files. Defaults to "
                         f"{DEFAULT_INPUT_GLOB}")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help="Output directory for parquet tables.")
    ap.add_argument("--batch-rows", type=int, default=500_000,
                    help="Flush L2 rows to parquet every N rows (default 500k).")
    ap.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "gzip", "none"])
    args = ap.parse_args()

    if not args.inputs:
        args.inputs = sorted(glob.glob(DEFAULT_INPUT_GLOB))
    if not args.inputs:
        raise SystemExit(f"no inputs found; tried {DEFAULT_INPUT_GLOB}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    l2_path = out_dir / "orderbook_l2.parquet"
    top_path = out_dir / "orderbook_top.parquet"

    compression = None if args.compression == "none" else args.compression

    l2_writer = pq.ParquetWriter(l2_path, L2_SCHEMA, compression=compression, use_dictionary=True)

    # Streaming L2 buffer (column-oriented for fast Arrow conversion).
    l2_buf = {"ts_ns": [], "ticker": [], "side": [], "price_cents": [], "size": []}
    # Top buffer (kept in memory — small).
    top_buf = {f.name: [] for f in TOP_SCHEMA}

    n_lines = 0
    n_bad = 0
    n_l2 = 0
    n_top = 0
    t0 = time.time()

    def flush_l2():
        nonlocal n_l2
        if not l2_buf["ts_ns"]:
            return
        batch = pa.Table.from_pydict(l2_buf, schema=L2_SCHEMA)
        l2_writer.write_table(batch)
        n_l2 += batch.num_rows
        for k in l2_buf:
            l2_buf[k] = []

    for path in args.inputs:
        with open(path, "rb") as fp:
            for line in fp:
                n_lines += 1
                try:
                    rec = json.loads(line)
                    ts = int(rec["ts_ns"])
                    ticker = rec["ticker"]
                    book = rec["response"]["orderbook_fp"]
                except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                    n_bad += 1
                    continue

                yes_levels = book.get("yes_dollars") or []
                no_levels = book.get("no_dollars") or []

                # ---- L2 long rows (raw sides) ----
                best_yes_price = None
                best_yes_size = None
                yes_total = 0.0
                for lvl in yes_levels:
                    if len(lvl) < 2:
                        continue
                    p, s = _to_cents(lvl[0]), _to_size(lvl[1])
                    if p is None or s is None:
                        continue
                    l2_buf["ts_ns"].append(ts)
                    l2_buf["ticker"].append(ticker)
                    l2_buf["side"].append("yes")
                    l2_buf["price_cents"].append(p)
                    l2_buf["size"].append(s)
                    yes_total += s
                    if best_yes_price is None or p > best_yes_price:
                        best_yes_price, best_yes_size = p, s

                best_no_price = None
                best_no_size = None
                no_total = 0.0
                for lvl in no_levels:
                    if len(lvl) < 2:
                        continue
                    p, s = _to_cents(lvl[0]), _to_size(lvl[1])
                    if p is None or s is None:
                        continue
                    l2_buf["ts_ns"].append(ts)
                    l2_buf["ticker"].append(ticker)
                    l2_buf["side"].append("no")
                    l2_buf["price_cents"].append(p)
                    l2_buf["size"].append(s)
                    no_total += s
                    if best_no_price is None or p > best_no_price:
                        best_no_price, best_no_size = p, s

                # ---- top-of-book derived ----
                yes_ask_cents = None if best_no_price is None else 100 - best_no_price
                yes_mid = None
                if best_yes_price is not None and yes_ask_cents is not None:
                    yes_mid = (best_yes_price + yes_ask_cents) / 2.0
                yes_spread = (
                    (yes_ask_cents - best_yes_price)
                    if (best_yes_price is not None and yes_ask_cents is not None)
                    else None
                )

                top_buf["ts_ns"].append(ts)
                top_buf["ticker"].append(ticker)
                top_buf["best_yes_bid_cents"].append(best_yes_price)
                top_buf["best_yes_bid_size"].append(best_yes_size)
                top_buf["best_no_bid_cents"].append(best_no_price)
                top_buf["best_no_bid_size"].append(best_no_size)
                top_buf["yes_ask_cents"].append(yes_ask_cents)
                top_buf["yes_mid_cents"].append(yes_mid)
                top_buf["yes_spread_cents"].append(yes_spread)
                top_buf["yes_total_size"].append(yes_total)
                top_buf["no_total_size"].append(no_total)
                top_buf["yes_n_levels"].append(len(yes_levels))
                top_buf["no_n_levels"].append(len(no_levels))
                n_top += 1

                if len(l2_buf["ts_ns"]) >= args.batch_rows:
                    flush_l2()

                if n_lines % 50_000 == 0:
                    dt = time.time() - t0
                    print(
                        f"  [{dt:6.1f}s] {n_lines:,} records, l2_rows~{n_l2 + len(l2_buf['ts_ns']):,}, "
                        f"top_rows={n_top:,}",
                        flush=True,
                    )

    flush_l2()
    l2_writer.close()

    top_table = pa.Table.from_pydict(top_buf, schema=TOP_SCHEMA)
    pq.write_table(top_table, top_path, compression=compression, use_dictionary=True)

    dt = time.time() - t0
    l2_mb = os.path.getsize(l2_path) / (1024 * 1024)
    top_mb = os.path.getsize(top_path) / (1024 * 1024)
    print(
        f"done: {n_lines:,} records ({n_bad:,} bad), "
        f"l2={n_l2:,} rows ({l2_mb:.1f} MB), "
        f"top={n_top:,} rows ({top_mb:.1f} MB), "
        f"elapsed={dt:.1f}s"
    )


if __name__ == "__main__":
    main()

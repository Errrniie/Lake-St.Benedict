#!/usr/bin/env python3
"""
Parse DATE column from DO_lakedata_parsed.csv (or argv path).
Outputs:
  - daily_time_coverage.csv: per calendar day — rows, time span, distinct hours, hours missing
  - consecutive_day_blocks.csv: runs of consecutive calendar days + whole-week count
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Analyze date/time coverage in lake CSV.")
    p.add_argument(
        "csv_path",
        nargs="?",
        default=None,
        help="Path to parsed CSV (default: repo root DO_lakedata_parsed.csv)",
    )
    p.add_argument(
        "-o",
        "--out-dir",
        default=None,
        help="Output directory (default: same folder as this script)",
    )
    return p.parse_args()


def resolve_input_path(csv_path_arg: str | None, default_csv: str) -> str:
    if csv_path_arg:
        return os.path.abspath(csv_path_arg)

    prompt = f"Enter path to parsed CSV [{default_csv}]: "
    try:
        entered = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("No input provided. Exiting.")

    if entered.lower() in {"q", "quit", "exit"}:
        raise SystemExit("Cancelled.")

    return os.path.abspath(entered or default_csv)


def consecutive_day_blocks(sorted_dates: list[date]) -> list[dict]:
    if not sorted_dates:
        return []
    blocks: list[dict] = []
    start = sorted_dates[0]
    prev = sorted_dates[0]
    for d in sorted_dates[1:]:
        if (d - prev).days == 1:
            prev = d
            continue
        blocks.append({"start_date": start, "end_date": prev})
        start = d
        prev = d
    blocks.append({"start_date": start, "end_date": prev})
    return blocks


def longest_hour_streak(hours: set[int]) -> int:
    if not hours:
        return 0
    h = sorted(hours)
    best = cur = 1
    for i in range(1, len(h)):
        if h[i] == h[i - 1] + 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def main() -> int:
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    default_csv = os.path.join(repo_root, "DO_lakedata_parsed.csv")
    default_out_dir = os.path.join(repo_root, "temp", "DataCoverageCSVs")
    csv_path = resolve_input_path(args.csv_path, default_csv)
    out_dir = os.path.abspath(args.out_dir or default_out_dir)

    if not os.path.isfile(csv_path):
        print(f"File not found: {csv_path}", file=sys.stderr)
        return 1

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if "DATE" not in df.columns:
        print("Expected a DATE column.", file=sys.stderr)
        return 1

    dt = pd.to_datetime(df["DATE"], format="mixed", errors="coerce")
    valid = dt.notna()
    dropped = int((~valid).sum())
    df = df.loc[valid].copy()
    df["_dt"] = dt[valid]

    # Per calendar day (local date of timestamp)
    df["_day"] = df["_dt"].dt.normalize()

    daily_rows = []
    for day, g in df.groupby("_day", sort=True):
        d = day.date() if hasattr(day, "date") else pd.Timestamp(day).date()
        times = g["_dt"]
        hours = set(int(t.hour) for t in times)
        all_hours = set(range(24))
        missing = sorted(all_hours - hours)
        daily_rows.append(
            {
                "calendar_date": d.isoformat(),
                "row_count": len(g),
                "first_datetime": times.min().isoformat(sep=" "),
                "last_datetime": times.max().isoformat(sep=" "),
                "distinct_hours_with_data": len(hours),
                "hours_missing_in_day": len(missing),
                "missing_hours_list": ",".join(str(x) for x in missing) if missing else "",
                "longest_consecutive_hour_streak_in_day": longest_hour_streak(hours),
            }
        )

    daily_df = pd.DataFrame(daily_rows)
    daily_path = os.path.join(out_dir, "daily_time_coverage.csv")
    daily_df.to_csv(daily_path, index=False)

    sorted_days = [pd.Timestamp(r["calendar_date"]).date() for _, r in daily_df.iterrows()]
    blocks = consecutive_day_blocks(sorted_days)

    block_rows = []
    for i, b in enumerate(blocks, start=1):
        sd, ed = b["start_date"], b["end_date"]
        days = (ed - sd).days + 1
        block_rows.append(
            {
                "block_id": i,
                "start_date": sd.isoformat(),
                "end_date": ed.isoformat(),
                "consecutive_calendar_days": days,
                "whole_weeks_in_block": days // 7,
                "remainder_days_after_full_weeks": days % 7,
            }
        )

    # Gap between blocks (calendar days not covered between end of one and start of next)
    for i in range(len(block_rows) - 1):
        end_curr = pd.Timestamp(block_rows[i]["end_date"]).date()
        start_next = pd.Timestamp(block_rows[i + 1]["start_date"]).date()
        gap = (start_next - end_curr).days - 1
        block_rows[i]["gap_calendar_days_before_next_block"] = max(0, gap)
    if block_rows:
        block_rows[-1]["gap_calendar_days_before_next_block"] = ""

    blocks_df = pd.DataFrame(block_rows)
    blocks_path = os.path.join(out_dir, "consecutive_day_blocks.csv")
    blocks_df.to_csv(blocks_path, index=False)

    print(f"Read: {csv_path}")
    print(f"Dropped rows with unparseable DATE: {dropped}")
    print(f"Unique calendar days with data: {len(daily_df)}")
    print(f"Consecutive-day blocks: {len(blocks_df)}")
    print(f"Wrote: {daily_path}")
    print(f"Wrote: {blocks_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fetch minute-by-minute ThetaData option chain snapshots (default root=SPXW)."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


def parse_date_yyyymmdd(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%d")


def hhmm_to_ms_of_day(value: str) -> int:
    hh, mm = value.split(":", maxsplit=1)
    h = int(hh)
    m = int(mm)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"Invalid HH:MM time: {value}")
    return (h * 60 * 60 + m * 60) * 1000


def fetch_text(url: str) -> str:
    with urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def list_expirations(base_url: str, root: str) -> set[str]:
    url = f"{base_url.rstrip('/')}/v2/list/expirations?{urlencode({'root': root, 'use_csv': 'true'})}"
    body = fetch_text(url)

    expiries: set[str] = set()
    reader = csv.reader(body.splitlines())
    for row in reader:
        for cell in row:
            m = re.search(r"\b(\d{8})\b", cell)
            if m:
                expiries.add(m.group(1))
    return expiries


def build_bulk_at_time_url(base_url: str, root: str, expiry: str, date_yyyymmdd: str, ms_of_day: int) -> str:
    params = {
        "root": root,
        "exp": expiry,
        "start_date": date_yyyymmdd,
        "end_date": date_yyyymmdd,
        "ivl": str(ms_of_day),
        "use_csv": "true",
    }
    return f"{base_url.rstrip('/')}/v2/bulk_at_time/option/quote?{urlencode(params)}"


def iter_minutes(start_ms: int, end_ms: int, step_min: int):
    cur = start_ms
    step = step_min * 60 * 1000
    while cur <= end_ms:
        yield cur
        cur += step


def ms_to_hhmmss(ms: int) -> str:
    sec = ms // 1000
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}{m:02d}{s:02d}"


def run(args: argparse.Namespace) -> int:
    date_dt = parse_date_yyyymmdd(args.date)
    expiry = args.expiry or args.date

    try:
        expiries = list_expirations(args.base_url, args.root)
    except (HTTPError, URLError) as exc:
        print(f"ERROR: Could not query expirations from ThetaData: {exc}", file=sys.stderr)
        return 2

    if expiry not in expiries:
        print(
            f"WARN: expiry {expiry} not found in list/expirations for root={args.root}. Continuing anyway.",
            file=sys.stderr,
        )

    start_ms = hhmm_to_ms_of_day(args.start)
    end_ms = hhmm_to_ms_of_day(args.end)
    if end_ms < start_ms:
        raise ValueError("end time must be >= start time")

    day_out = args.out_dir / date_dt.strftime("%Y-%m-%d")
    day_out.mkdir(parents=True, exist_ok=True)

    ok = 0
    nodata = 0
    failed = 0

    for ms in iter_minutes(start_ms, end_ms, args.step_min):
        url = build_bulk_at_time_url(args.base_url, args.root, expiry, args.date, ms)
        ts_name = f"{args.date}_{ms_to_hhmmss(ms)}.csv"
        out_path = day_out / ts_name

        try:
            body = fetch_text(url)
        except (HTTPError, URLError) as exc:
            failed += 1
            if args.verbose:
                print(f"FAIL {ms_to_hhmmss(ms)} {exc}")
            continue

        trimmed = body.strip()
        if not trimmed or trimmed.startswith(":No data"):
            nodata += 1
            continue

        out_path.write_text(body, encoding="utf-8")
        ok += 1
        if args.verbose:
            print(f"OK   {ms_to_hhmmss(ms)} -> {out_path}")

    print(
        f"Done root={args.root} date={args.date} expiry={expiry} "
        f"saved={ok} no_data={nodata} failed={failed} dir={day_out}"
    )
    return 0 if ok > 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch ThetaData option quote snapshots for an intraday window.")
    parser.add_argument("--base-url", default="http://127.0.0.1:25510")
    parser.add_argument("--root", default="SPXW", help="ThetaData option root (default: SPXW)")
    parser.add_argument("--date", required=True, help="Trade date YYYYMMDD")
    parser.add_argument("--expiry", default="", help="Expiration YYYYMMDD (default: same as --date)")
    parser.add_argument("--start", default="10:00", help="Start time HH:MM ET")
    parser.add_argument("--end", default="14:00", help="End time HH:MM ET")
    parser.add_argument("--step-min", type=int, default=1, help="Minute step between requests")
    parser.add_argument("--out-dir", type=Path, default=Path("data/thetadata/chains"))
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.expiry == "":
        args.expiry = args.date
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

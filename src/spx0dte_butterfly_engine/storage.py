from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .contracts import ChainSnapshot


@dataclass
class DataStore:
    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def write_bars(self, symbol: str, df: pd.DataFrame) -> None:
        path = self.root / "bars" / f"{symbol}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        out = df.copy()
        if "ts" in out.columns:
            out["ts"] = pd.to_datetime(out["ts"], utc=True).dt.tz_convert("America/New_York")
        out.to_csv(path, index=False)

    def read_bars(self, symbol: str, date_range: tuple[datetime | date, datetime | date] | None = None) -> pd.DataFrame:
        path = self.root / "bars" / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("America/New_York")
        if date_range and "ts" in df.columns:
            start, end = date_range
            start_ts = pd.Timestamp(start, tz="America/New_York")
            end_ts = pd.Timestamp(end, tz="America/New_York")
            df = df[(df["ts"] >= start_ts) & (df["ts"] <= end_ts)]
        return df.reset_index(drop=True)

    def write_chain_snapshot(self, session_date: date, ts: datetime, chain_df: pd.DataFrame) -> None:
        path = self.root / "chains" / session_date.isoformat() / f"{ts.strftime('%H%M%S')}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        chain_df.to_csv(path, index=False)

    def read_chain_snapshot(self, session_date: date, ts: datetime) -> ChainSnapshot:
        path = self.root / "chains" / session_date.isoformat() / f"{ts.strftime('%H%M%S')}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        expiry = pd.to_datetime(df["expiry"]).dt.date.iloc[0] if "expiry" in df.columns and len(df) else session_date
        return ChainSnapshot(ts=ts, expiry=expiry, rows_df=df, quality_flags={"source": "datastore"})

    def write_features(self, session_date: date, features_df: pd.DataFrame) -> None:
        path = self.root / "features" / f"{session_date.isoformat()}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(path, index=False)

    def read_features(self, date_range: tuple[date, date]) -> pd.DataFrame:
        start, end = date_range
        out: list[pd.DataFrame] = []
        features_dir = self.root / "features"
        if not features_dir.exists():
            return pd.DataFrame()
        for path in sorted(features_dir.glob("*.csv")):
            d = date.fromisoformat(path.stem)
            if start <= d <= end:
                out.append(pd.read_csv(path))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    def append_event(self, stream_name: str, event_dict: dict) -> None:
        path = self.root / "events" / f"{stream_name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event_dict, default=str) + "\n")

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .contracts import AlignedBars, SessionInfo


@dataclass
class DataIngestor:
    def ingest_tradingview_csv(self, path: str | Path, symbol: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]

        if "time" in df.columns:
            raw_time = pd.to_numeric(df["time"], errors="coerce")
            unit = "ms" if raw_time.dropna().median() > 1e11 else "s"
            ts = pd.to_datetime(raw_time, unit=unit, utc=True)
            df = df.drop(columns=["time"])
        elif "datetime" in df.columns:
            ts = pd.to_datetime(df["datetime"], utc=True)
            df = df.drop(columns=["datetime"])
        elif "date" in df.columns:
            ts = pd.to_datetime(df["date"], utc=True)
            df = df.drop(columns=["date"])
        else:
            raise ValueError(f"CSV {path} missing time/datetime/date column")

        df["ts"] = ts.dt.tz_convert("America/New_York")
        numeric_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if "volume" not in df.columns:
            df["volume"] = 0.0

        return df[["ts", "open", "high", "low", "close", "volume"]].sort_values("ts").reset_index(drop=True)

    def normalize_timezone(self, df: pd.DataFrame, tz: str = "America/New_York") -> pd.DataFrame:
        out = df.copy()
        out["ts"] = pd.to_datetime(out["ts"], utc=True).dt.tz_convert(tz)
        return out

    def resample(self, df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
        out = df.copy()
        out["ts"] = pd.to_datetime(out["ts"])
        out = out.set_index("ts").sort_index()
        ohlc = out[["open", "high", "low", "close"]].resample(freq).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        )
        vol = out[["volume"]].resample(freq).sum()
        resampled = pd.concat([ohlc, vol], axis=1)
        return resampled.dropna(subset=["open", "high", "low", "close"]).reset_index()

    def align_series(self, series_dict: dict[str, pd.DataFrame], session: SessionInfo) -> AlignedBars:
        idx = pd.date_range(session.open_dt, session.close_dt, freq="1min")

        aligned: dict[str, pd.DataFrame] = {}
        quality: dict[str, float | str | bool] = {}

        for name, df in series_dict.items():
            tmp = df.copy()
            tmp["ts"] = pd.to_datetime(tmp["ts"]).dt.tz_convert("America/New_York")
            tmp = tmp.set_index("ts").sort_index()
            tmp = tmp.reindex(idx)
            missing = float(tmp["close"].isna().mean()) if "close" in tmp.columns else 1.0
            quality[f"{name}_missing_ratio"] = missing
            tmp[["open", "high", "low", "close"]] = tmp[["open", "high", "low", "close"]].ffill()
            if "volume" in tmp.columns:
                tmp["volume"] = tmp["volume"].fillna(0.0)
            aligned[name] = tmp.reset_index().rename(columns={"index": "ts"})

        return AlignedBars(
            spx_df=aligned.get("spx", pd.DataFrame(index=idx).reset_index().rename(columns={"index": "ts"})),
            vix_df=aligned.get("vix", pd.DataFrame(index=idx).reset_index().rename(columns={"index": "ts"})),
            vix1d_df=aligned.get("vix1d", pd.DataFrame(index=idx).reset_index().rename(columns={"index": "ts"})),
            quality_flags=quality,
        )

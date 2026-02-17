from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .calendar import SessionCalendar
from .contracts import AlignedBars, ChainSnapshot
from .ingest import DataIngestor


class MarketDataProvider(ABC):
    @abstractmethod
    def get_aligned_bars(self, session_date: date) -> AlignedBars:
        raise NotImplementedError

    @abstractmethod
    def get_chain_snapshot(self, ts: datetime) -> ChainSnapshot:
        raise NotImplementedError

    @abstractmethod
    def get_spot(self, ts: datetime) -> float:
        raise NotImplementedError


@dataclass
class LocalProvider(MarketDataProvider):
    data_dir: Path
    ingestor: DataIngestor
    calendar: SessionCalendar
    file_map: dict[str, str] = field(
        default_factory=lambda: {
            "spx": "SP_SPX, 5 (1).csv",
            "vix": "TVC_VIX, 5 (1).csv",
            "vix1d": "CBOE_DLY_VIX1D, 5 (1).csv",
        }
    )

    _raw_cache: dict[str, pd.DataFrame] = field(default_factory=dict, init=False)
    _aligned_cache: dict[date, AlignedBars] = field(default_factory=dict, init=False)

    def _load_symbol(self, symbol: str) -> pd.DataFrame:
        if symbol in self._raw_cache:
            return self._raw_cache[symbol]
        path = self.data_dir / self.file_map[symbol]
        df = self.ingestor.ingest_tradingview_csv(path, symbol)
        df = self.ingestor.resample(df, "1min")
        self._raw_cache[symbol] = df
        return df

    def get_aligned_bars(self, session_date: date) -> AlignedBars:
        if session_date in self._aligned_cache:
            return self._aligned_cache[session_date]

        session = self.calendar.get_session(session_date)
        sliced: dict[str, pd.DataFrame] = {}
        for symbol in ["spx", "vix", "vix1d"]:
            df = self._load_symbol(symbol)
            day_df = df[pd.to_datetime(df["ts"]).dt.date == session_date].copy()
            if day_df.empty:
                # Fallback for bootstrap mode: use latest available day.
                latest_date = pd.to_datetime(df["ts"]).dt.date.max()
                day_df = df[pd.to_datetime(df["ts"]).dt.date == latest_date].copy()
            sliced[symbol] = day_df

        aligned = self.ingestor.align_series(sliced, session)
        self._aligned_cache[session_date] = aligned
        return aligned

    def get_spot(self, ts: datetime) -> float:
        bars = self.get_aligned_bars(ts.date())
        spx = bars.spx_df.copy()
        spx["ts"] = pd.to_datetime(spx["ts"])
        row = spx[spx["ts"] <= ts].tail(1)
        if row.empty:
            row = spx.head(1)
        return float(row.iloc[0]["close"])

    def get_chain_snapshot(self, ts: datetime) -> ChainSnapshot:
        spot = self.get_spot(ts)
        center = int(round(spot / 5.0) * 5)
        strikes = np.arange(center - 50, center + 55, 5)

        rows: list[dict[str, float | str]] = []
        for strike in strikes:
            dist = abs(strike - spot)
            time_value = max(0.2, 8.0 - 0.08 * dist)
            iv = 0.12 + 0.002 * min(dist, 40)
            for right in ["C", "P"]:
                intrinsic = max(0.0, spot - strike) if right == "C" else max(0.0, strike - spot)
                mid = intrinsic + time_value
                spread = max(0.05, 0.03 * time_value)
                bid = max(0.01, mid - spread / 2.0)
                ask = bid + spread
                theta = -0.015 * time_value
                delta = np.tanh((spot - strike) / 25.0) if right == "C" else -np.tanh((strike - spot) / 25.0)
                rows.append(
                    {
                        "ts": ts,
                        "expiry": ts.date(),
                        "strike": float(strike),
                        "right": right,
                        "bid": float(bid),
                        "ask": float(ask),
                        "mid": float((bid + ask) / 2.0),
                        "iv": float(iv),
                        "delta": float(delta),
                        "gamma": float(0.01 / (1.0 + dist)),
                        "theta": float(theta),
                        "vega": float(0.05 * time_value),
                        "volume": float(max(5, 150 - dist * 2)),
                    }
                )

        chain_df = pd.DataFrame(rows)
        return ChainSnapshot(ts=ts, expiry=ts.date(), rows_df=chain_df, quality_flags={"synthetic": True})


@dataclass
class IBProvider(MarketDataProvider):
    """Placeholder for live IB streaming implementation."""

    def get_aligned_bars(self, session_date: date) -> AlignedBars:
        raise NotImplementedError("IBProvider.get_aligned_bars is not implemented in MVP scaffold")

    def get_chain_snapshot(self, ts: datetime) -> ChainSnapshot:
        raise NotImplementedError("IBProvider.get_chain_snapshot is not implemented in MVP scaffold")

    def get_spot(self, ts: datetime) -> float:
        raise NotImplementedError("IBProvider.get_spot is not implemented in MVP scaffold")

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

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


def _parse_ts_from_name(name: str) -> datetime | None:
    candidates = [
        (r"(\d{8})[_-](\d{6})", "%Y%m%d%H%M%S"),
        (r"(\d{8})[_-](\d{4})", "%Y%m%d%H%M"),
        (r"(\d{4}-\d{2}-\d{2})[_-](\d{6})", "%Y-%m-%d%H%M%S"),
        (r"(\d{4}-\d{2}-\d{2})[_-](\d{4})", "%Y-%m-%d%H%M"),
    ]
    for pattern, fmt in candidates:
        m = re.search(pattern, name)
        if not m:
            continue
        s = "".join(m.groups())
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


@dataclass
class ChainSnapshotStore:
    """Loads pre-exported ThetaData snapshots from disk for deterministic replay/live marking."""

    root: Path
    max_staleness_min: int = 5

    def load(self, ts: datetime) -> ChainSnapshot | None:
        paths = self._candidate_paths(ts.date())
        if not paths:
            return None

        best_path: Path | None = None
        best_delta = float("inf")
        for path in paths:
            pts = _parse_ts_from_name(path.stem)
            if pts is None:
                pts = datetime.fromtimestamp(path.stat().st_mtime)
            if ts.tzinfo is not None:
                pts = pd.Timestamp(pts, tz="America/New_York").to_pydatetime()
            delta = abs((ts - pts).total_seconds())
            if delta < best_delta:
                best_delta = delta
                best_path = path

        if best_path is None:
            return None

        if best_delta > self.max_staleness_min * 60:
            return None

        raw = _read_table(best_path)
        return self._normalize(raw, ts, best_path, best_delta)

    def _candidate_paths(self, session_date: date) -> list[Path]:
        if not self.root.exists():
            return []

        day_dir = self.root / session_date.isoformat()
        paths: list[Path] = []
        if day_dir.exists():
            paths.extend(sorted(day_dir.glob("*.csv")))
            paths.extend(sorted(day_dir.glob("*.parquet")))

        if not paths:
            ymd = session_date.strftime("%Y%m%d")
            paths.extend(sorted(self.root.glob(f"*{ymd}*.csv")))
            paths.extend(sorted(self.root.glob(f"*{ymd}*.parquet")))

        return paths

    @staticmethod
    def _normalize(raw: pd.DataFrame, ts: datetime, source_path: Path, staleness_sec: float) -> ChainSnapshot:
        df = raw.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        rename = {
            "option_type": "right",
            "type": "right",
            "underlying_price": "underlying",
            "expiration": "expiry",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        required = {"strike", "right", "bid", "ask"}
        if not required.issubset(set(df.columns)):
            missing = sorted(required.difference(set(df.columns)))
            raise ValueError(f"Chain snapshot {source_path} missing required columns: {missing}")

        for col in ["strike", "bid", "ask", "mid", "iv", "delta", "gamma", "theta", "vega", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ThetaData option strikes are commonly scaled by 1000 (e.g. 6000000 -> 6000.0).
        if "strike" in df.columns and not df["strike"].dropna().empty:
            strike_med = float(df["strike"].dropna().median())
            if strike_med > 100000:
                df["strike"] = df["strike"] / 1000.0

        df["right"] = df["right"].astype(str).str.upper().str[0]
        if "mid" not in df.columns:
            df["mid"] = (df["bid"] + df["ask"]) / 2.0

        for col in ["iv", "delta", "gamma", "theta", "vega", "volume"]:
            if col not in df.columns:
                df[col] = 0.0

        df = df.dropna(subset=["strike", "bid", "ask", "mid", "right"])
        df["ts"] = ts

        expiry = ts.date()
        if "expiry" in df.columns and len(df):
            try:
                raw_exp = str(df["expiry"].iloc[0]).strip()
                m = re.search(r"\b(\d{8})\b", raw_exp)
                if m:
                    expiry = datetime.strptime(m.group(1), "%Y%m%d").date()
                else:
                    expiry = pd.to_datetime(df["expiry"]).dt.date.iloc[0]
            except Exception:
                expiry = ts.date()

        return ChainSnapshot(
            ts=ts,
            expiry=expiry,
            rows_df=df,
            quality_flags={
                "source": "theta_cache",
                "path": str(source_path),
                "staleness_sec": float(staleness_sec),
            },
        )


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
    theta_chain_dir: Path | None = None
    chain_max_staleness_min: int = 5

    _raw_cache: dict[str, pd.DataFrame] = field(default_factory=dict, init=False)
    _aligned_cache: dict[date, AlignedBars] = field(default_factory=dict, init=False)
    _chain_store: ChainSnapshotStore | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.theta_chain_dir is not None:
            self._chain_store = ChainSnapshotStore(self.theta_chain_dir, self.chain_max_staleness_min)

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
        cached = self._chain_store.load(ts) if self._chain_store is not None else None
        if cached is not None:
            return cached

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
    """IB spot provider with optional ThetaData chain cache fallback for options snapshots."""

    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 17
    symbol: str = "SPX"
    exchange: str = "CBOE"
    chain_store: ChainSnapshotStore | None = None
    fallback_provider: MarketDataProvider | None = None

    _ib: Any = field(default=None, init=False)
    _ib_mod: Any = field(default=None, init=False)

    def connect(self) -> None:
        if self._ib is not None:
            return
        try:
            import ib_insync as ib_mod
        except Exception as exc:
            raise RuntimeError("ib_insync is required for IBProvider. Install with: pip install ib-insync") from exc

        ib = ib_mod.IB()
        ib.connect(self.host, self.port, clientId=self.client_id, readonly=True)
        self._ib = ib
        self._ib_mod = ib_mod

    def disconnect(self) -> None:
        if self._ib is None:
            return
        self._ib.disconnect()
        self._ib = None
        self._ib_mod = None

    def _ensure_connected(self) -> None:
        if self._ib is None:
            self.connect()

    def get_aligned_bars(self, session_date: date) -> AlignedBars:
        if self.fallback_provider is None:
            raise RuntimeError("IBProvider.get_aligned_bars requires fallback_provider or dedicated bar pipeline")
        return self.fallback_provider.get_aligned_bars(session_date)

    def get_chain_snapshot(self, ts: datetime) -> ChainSnapshot:
        if self.chain_store is not None:
            snap = self.chain_store.load(ts)
            if snap is not None:
                return snap
        if self.fallback_provider is not None:
            return self.fallback_provider.get_chain_snapshot(ts)
        raise RuntimeError("No options chain source available (set chain_store or fallback_provider)")

    def get_spot(self, ts: datetime) -> float:
        try:
            self._ensure_connected()
            index_cls = self._ib_mod.Index
            contract = index_cls(symbol=self.symbol, exchange=self.exchange)
            self._ib.qualifyContracts(contract)

            ticker = self._ib.reqMktData(contract, "", False, False)
            self._ib.sleep(0.5)
            market_price = ticker.marketPrice()
            if market_price is None or (isinstance(market_price, float) and math.isnan(market_price)) or market_price <= 0:
                market_price = ticker.last
            if market_price is None or (isinstance(market_price, float) and math.isnan(market_price)) or market_price <= 0:
                market_price = ticker.close
            self._ib.cancelMktData(contract)

            if market_price is not None and not (isinstance(market_price, float) and math.isnan(market_price)) and market_price > 0:
                return float(market_price)
        except Exception:
            if self.fallback_provider is not None:
                return self.fallback_provider.get_spot(ts)
            raise

        if self.fallback_provider is not None:
            return self.fallback_provider.get_spot(ts)
        raise RuntimeError("Unable to fetch SPX spot from IB and no fallback provider configured")

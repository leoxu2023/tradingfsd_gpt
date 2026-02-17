from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from .contracts import AlignedBars, FeatureRow, SessionInfo


@dataclass
class FeatureEngine:
    def compute_features(self, aligned: AlignedBars, session: SessionInfo) -> pd.DataFrame:
        spx = aligned.spx_df[["ts", "open", "high", "low", "close", "volume"]].rename(
            columns={"close": "spx_close", "high": "spx_high", "low": "spx_low", "volume": "spx_volume"}
        )
        vix = aligned.vix_df[["ts", "close"]].rename(columns={"close": "vix_close"})
        vix1d = aligned.vix1d_df[["ts", "close"]].rename(columns={"close": "vix1d_close"})

        df = spx.merge(vix, on="ts", how="left").merge(vix1d, on="ts", how="left")
        df["ret_1m"] = df["spx_close"].pct_change()
        df["ret_5m"] = df["spx_close"].pct_change(5)
        df["ret_15m"] = df["spx_close"].pct_change(15)
        df["rv_15"] = df["ret_1m"].rolling(15).std() * np.sqrt(390)
        df["range_frac"] = (df["spx_high"] - df["spx_low"]) / df["spx_close"].replace(0, np.nan)
        df["vix_term"] = df["vix1d_close"] - df["vix_close"]
        df["session_progress"] = (
            (pd.to_datetime(df["ts"]) - session.open_dt) / (session.close_dt - session.open_dt)
        ).clip(lower=0, upper=1)

        feature_cols = [
            "ret_1m",
            "ret_5m",
            "ret_15m",
            "rv_15",
            "range_frac",
            "vix_close",
            "vix1d_close",
            "vix_term",
            "spx_volume",
            "session_progress",
        ]

        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df[["ts", *feature_cols]]

    def update_incremental(self, ts: datetime, latest_bars: dict[str, pd.DataFrame]) -> FeatureRow:
        # Use the most recent lookback window to compute the same feature set as batch mode.
        ts_ny = pd.Timestamp(ts)
        ts_ny = ts_ny.tz_convert("America/New_York") if ts_ny.tzinfo else ts_ny.tz_localize("America/New_York")
        day_start = ts_ny.normalize()
        aligned = AlignedBars(
            spx_df=latest_bars["spx"].tail(60).copy(),
            vix_df=latest_bars["vix"].tail(60).copy(),
            vix1d_df=latest_bars["vix1d"].tail(60).copy(),
            quality_flags={"incremental": True},
        )
        pseudo_session = SessionInfo(
            date=ts.date(),
            is_half_day=False,
            open_dt=day_start + pd.Timedelta(hours=9, minutes=30),
            close_dt=day_start + pd.Timedelta(hours=16),
            entry_start=day_start + pd.Timedelta(hours=10),
            entry_end=day_start + pd.Timedelta(hours=14),
        )
        feat_df = self.compute_features(aligned, pseudo_session)
        row = feat_df.tail(1).iloc[0]
        features = {k: float(row[k]) for k in feat_df.columns if k != "ts"}
        return FeatureRow(ts=pd.to_datetime(row["ts"]).to_pydatetime(), features=features)

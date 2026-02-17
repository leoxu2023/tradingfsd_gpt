from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from ..contracts import FeatureRow


@dataclass
class DatasetBuilder:
    calendar: Any
    feature_engine: Any
    regime_engine: Any
    candidate_gen: Any
    pricer: Any

    def build_candidates(self, date_range: tuple[date, date], policy_cfg, data_provider, out_path: Path | None = None) -> Path:
        rows: list[dict] = []
        start, end = date_range
        d = start
        while d <= end:
            session = self.calendar.get_session(d)
            aligned = data_provider.get_aligned_bars(d)
            feat_df = self.feature_engine.compute_features(aligned, session)
            for _, feat in feat_df.iterrows():
                ts = pd.to_datetime(feat["ts"]).to_pydatetime()
                if not self.calendar.is_in_entry_window(ts, session):
                    continue

                spot = data_provider.get_spot(ts)
                chain = data_provider.get_chain_snapshot(ts)
                feature_row = {k: float(feat[k]) for k in feat_df.columns if k != "ts"}
                feature_row["spot"] = spot
                regime = self.regime_engine.infer(ts, FeatureRow(ts=ts, features=feature_row))
                candidates = self.candidate_gen.generate(
                    ts=ts,
                    spot=spot,
                    chain=chain,
                    regime=regime,
                    constraints={"candidate_grid": policy_cfg.candidate_grid},
                )
                candidates = self.candidate_gen.filter_theta_positive(candidates, chain, self.pricer)
                for spec, quote in self.candidate_gen.attach_quotes(candidates, chain, self.pricer):
                    rows.append(
                        {
                            "ts": ts,
                            "date": d.isoformat(),
                            "direction": spec.direction,
                            "k_center": spec.k_center,
                            "wing": spec.wing,
                            "mid": quote.mid,
                            "spread": quote.spread,
                            "theta": quote.theta,
                            "liquidity_score": quote.liquidity_score,
                            "regime": regime.regime,
                            "bias": regime.bias,
                            "confidence": regime.confidence,
                        }
                    )
            d += timedelta(days=1)

        df = pd.DataFrame(rows)
        if out_path is None:
            out_path = Path("artifacts") / "datasets" / f"candidates_{start}_{end}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_table(df, out_path)
        return out_path

    def label_candidates(self, candidate_rows_path: Path, simulator, risk_cfg, out_path: Path | None = None) -> Path:
        df = self._read_table(candidate_rows_path)
        if df.empty:
            raise ValueError("No candidate rows found to label")

        # Placeholder utility label for bootstrap; replace with full replay trade labels.
        df["label_utility"] = (
            0.6 * df["theta"].astype(float)
            - 0.4 * df["spread"].astype(float)
            + 0.02 * df["liquidity_score"].astype(float)
            - 0.03 * df["mid"].astype(float)
        )
        df["group_id"] = self.make_groups(df)

        if out_path is None:
            out_path = candidate_rows_path.with_name(candidate_rows_path.stem + "_labeled.parquet")
        self._write_table(df, out_path)
        return out_path

    def make_groups(self, rows: pd.DataFrame) -> pd.Series:
        return pd.to_datetime(rows["ts"]).dt.strftime("%Y%m%d%H%M")

    @staticmethod
    def _write_table(df: pd.DataFrame, path: Path) -> None:
        if path.suffix == ".parquet":
            try:
                df.to_parquet(path, index=False)
                return
            except Exception:
                fallback = path.with_suffix(".csv")
                df.to_csv(fallback, index=False)
                return
        df.to_csv(path, index=False)

    @staticmethod
    def _read_table(path: Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            try:
                return pd.read_parquet(path)
            except Exception:
                csv_fallback = path.with_suffix(".csv")
                if csv_fallback.exists():
                    return pd.read_csv(csv_fallback)
                raise
        return pd.read_csv(path)

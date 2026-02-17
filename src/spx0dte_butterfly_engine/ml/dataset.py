from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from ..contracts import FeatureRow
from ..strategy import ButterflyFactory


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
                    moneyness = (float(spec.k_center) - float(spot)) / max(float(spot), 1e-6)
                    wing_pct = float(spec.wing) / max(float(spot), 1e-6)
                    rows.append(
                        {
                            "ts": ts,
                            "date": d.isoformat(),
                            "direction": spec.direction,
                            "direction_is_call": 1.0 if spec.direction == "call" else 0.0,
                            "direction_is_put": 1.0 if spec.direction == "put" else 0.0,
                            "spot": float(spot),
                            "k_center": float(spec.k_center),
                            "wing": float(spec.wing),
                            "moneyness": float(moneyness),
                            "abs_moneyness": abs(float(moneyness)),
                            "wing_pct": float(wing_pct),
                            "distance_from_spot": abs(float(spec.k_center) - float(spot)),
                            "mid": float(quote.mid),
                            "spread": float(quote.spread),
                            "theta": float(quote.theta),
                            "liquidity_score": float(quote.liquidity_score),
                            "theta_per_dollar": float(quote.theta) / max(abs(float(quote.mid)), 1e-6),
                            "liquidity_per_spread": float(quote.liquidity_score) / max(float(quote.spread), 1e-6),
                            "regime": regime.regime,
                            "bias": regime.bias,
                            "confidence": float(regime.confidence),
                        }
                    )
            d += timedelta(days=1)

        df = pd.DataFrame(rows)
        if out_path is None:
            out_path = Path("artifacts") / "datasets" / f"candidates_{start}_{end}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_table(df, out_path)
        return out_path

    def label_candidates(
        self,
        candidate_rows_path: Path,
        simulator,
        risk_cfg,
        data_provider=None,
        hold_minutes: int = 30,
        out_path: Path | None = None,
    ) -> Path:
        df = self._read_table(candidate_rows_path)
        if df.empty:
            raise ValueError("No candidate rows found to label")

        if data_provider is None:
            # Fallback utility if no provider is supplied.
            df["label_utility"] = (
                0.8 * df["theta"].astype(float)
                - 0.6 * df["spread"].astype(float)
                + 0.02 * df["liquidity_score"].astype(float)
                - 0.03 * df["mid"].astype(float)
                - 0.15 * df["abs_moneyness"].astype(float)
            )
        else:
            df["label_utility"] = self._label_with_forward_marks(df, data_provider=data_provider, hold_minutes=hold_minutes)

        df["group_id"] = self.make_groups(df)

        # Group-normalized target supports ranking behavior by timestamp decision set.
        grp = df.groupby("group_id")["label_utility"]
        grp_mean = grp.transform("mean")
        grp_std = grp.transform("std").replace(0, 1.0).fillna(1.0)
        df["label_group_z"] = (df["label_utility"] - grp_mean) / grp_std

        if out_path is None:
            out_path = candidate_rows_path.with_name(candidate_rows_path.stem + "_labeled.parquet")
        self._write_table(df, out_path)
        return out_path

    def _label_with_forward_marks(self, df: pd.DataFrame, data_provider, hold_minutes: int) -> pd.Series:
        factory = ButterflyFactory()
        utilities: list[float] = []

        quote_cache: dict[tuple[str, str, float, float], float] = {}

        for row in df.itertuples(index=False):
            ts = pd.to_datetime(getattr(row, "ts")).to_pydatetime()
            direction = str(getattr(row, "direction"))
            k_center = float(getattr(row, "k_center"))
            wing = float(getattr(row, "wing"))

            spec = (
                factory.build_call_fly(ts, ts.date(), k_center, wing)
                if direction == "call"
                else factory.build_put_fly(ts, ts.date(), k_center, wing)
            )

            session = self.calendar.get_session(ts.date())
            exit_ts = min(ts + timedelta(minutes=hold_minutes), session.close_dt)

            entry_mid = self._cached_mid(ts, spec, data_provider, quote_cache)
            exit_mid = self._cached_mid(exit_ts, spec, data_provider, quote_cache)

            spread = float(getattr(row, "spread"))
            theta = float(getattr(row, "theta"))
            liquidity = float(getattr(row, "liquidity_score"))
            abs_moneyness = float(getattr(row, "abs_moneyness"))

            forward_move = (exit_mid - entry_mid) * 100.0
            theta_carry = theta * min(hold_minutes, 60)
            transaction_penalty = spread * 20.0
            moneyness_penalty = abs_moneyness * 20.0
            liquidity_bonus = 0.01 * liquidity
            utility = forward_move + theta_carry + liquidity_bonus - transaction_penalty - moneyness_penalty
            utilities.append(float(utility))

        return pd.Series(utilities)

    def _cached_mid(self, ts, spec, data_provider, cache: dict[tuple[str, str, float, float], float]) -> float:
        key = (pd.Timestamp(ts).isoformat(), spec.direction, float(spec.k_center), float(spec.wing))
        if key in cache:
            return cache[key]
        try:
            chain = data_provider.get_chain_snapshot(ts)
            quote = self.pricer.quote_fly(chain, spec)
            mid = float(quote.mid)
        except Exception:
            mid = float("nan")

        if not pd.notna(mid):
            mid = 0.0
        cache[key] = mid
        return mid

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

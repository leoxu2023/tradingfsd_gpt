from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import ChainSnapshot, FlyQuote, FlySpec, Position


@dataclass
class OptionPricer:
    def quote_leg(self, chain: ChainSnapshot, strike: float, right: str) -> tuple[float, float, float, dict[str, float]]:
        right_norm = right.upper()[0]
        rows = chain.rows_df
        match = rows[(rows["strike"].round(4) == round(float(strike), 4)) & (rows["right"].str.upper() == right_norm)]
        if match.empty:
            return (float("nan"), float("nan"), float("nan"), {"delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan, "iv": np.nan, "volume": 0.0})

        row = match.iloc[0]
        bid = float(row.get("bid", np.nan))
        ask = float(row.get("ask", np.nan))
        mid = float(row.get("mid", (bid + ask) / 2.0 if np.isfinite(bid) and np.isfinite(ask) else np.nan))
        greeks = {
            "delta": float(row.get("delta", np.nan)),
            "gamma": float(row.get("gamma", np.nan)),
            "theta": float(row.get("theta", np.nan)),
            "vega": float(row.get("vega", np.nan)),
            "iv": float(row.get("iv", np.nan)),
            "volume": float(row.get("volume", 0.0)),
        }
        return bid, ask, mid, greeks

    def quote_fly(self, chain: ChainSnapshot, fly: FlySpec) -> FlyQuote:
        combo_bid = 0.0
        combo_ask = 0.0
        combo_mid = 0.0
        agg = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "iv": 0.0}
        liquidity = []

        for strike, right, ratio, side in fly.legs:
            bid, ask, mid, leg_greeks = self.quote_leg(chain, strike, right)
            if not np.isfinite(mid):
                return FlyQuote(mid=np.nan, bid=np.nan, ask=np.nan, greeks=agg, theta=np.nan, spread=np.nan, liquidity_score=0.0)

            signed = 1.0 if side.lower() == "buy" else -1.0
            combo_mid += signed * ratio * mid

            if side.lower() == "buy":
                combo_ask += ratio * ask
                combo_bid += ratio * bid
            else:
                combo_ask -= ratio * bid
                combo_bid -= ratio * ask

            for g in ["delta", "gamma", "theta", "vega"]:
                agg[g] += signed * ratio * float(leg_greeks[g])
            agg["iv"] += ratio * float(leg_greeks["iv"])
            liquidity.append(float(leg_greeks["volume"]))

        agg["iv"] /= max(1, len(fly.legs))
        spread = max(0.0, combo_ask - combo_bid)
        liquidity_score = float(np.mean(liquidity)) if liquidity else 0.0
        return FlyQuote(
            mid=float(combo_mid),
            bid=float(combo_bid),
            ask=float(combo_ask),
            greeks=agg,
            theta=float(agg["theta"]),
            spread=float(spread),
            liquidity_score=liquidity_score,
        )

    def mark_position(self, position: Position, chain: ChainSnapshot, method: str = "mid") -> float:
        quote = self.quote_fly(chain, position.fly_spec)
        if method == "model" and np.isfinite(quote.mid):
            # Lightweight model mark: tilt mid toward bid when spreads are wide.
            model_mark = quote.mid - 0.2 * quote.spread
            return float(model_mark)
        return float(quote.mid)

    def estimate_slippage(self, fly_quote: FlyQuote, urgency: str) -> float:
        urgency_mult = {"low": 0.25, "normal": 0.5, "high": 0.8}.get(urgency, 0.5)
        return float(max(0.01, fly_quote.spread * urgency_mult))

from __future__ import annotations

from dataclasses import dataclass

from .contracts import FeatureRow, RegimeState


@dataclass
class RegimeEngine:
    def infer(self, ts, feature_row: FeatureRow) -> RegimeState:
        f = feature_row.features
        rv_15 = float(f.get("rv_15", 0.0))
        vix = float(f.get("vix_close", 0.0))
        ret_15m = float(f.get("ret_15m", 0.0))

        if rv_15 > 0.02 or vix > 30:
            regime = "high_vol"
        elif rv_15 > 0.01 or vix > 22:
            regime = "normal_vol"
        else:
            regime = "low_vol"

        if ret_15m > 0.001:
            bias = "bullish"
        elif ret_15m < -0.001:
            bias = "bearish"
        else:
            bias = "neutral"

        confidence = min(1.0, abs(ret_15m) * 200 + rv_15 * 10)
        return RegimeState(ts=ts, regime=regime, bias=bias, confidence=confidence)

    def explain(self, regime_state: RegimeState) -> dict:
        return {
            "ts": regime_state.ts,
            "regime": regime_state.regime,
            "bias": regime_state.bias,
            "confidence": regime_state.confidence,
        }

from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import TradeIntent


@dataclass
class ModelPolicy:
    version: str
    weights: dict[str, float] = field(default_factory=dict)

    @classmethod
    def load(cls, model_version: str) -> "ModelPolicy":
        default_weights = {
            "theta": 1.0,
            "liquidity_score": 0.01,
            "spread": -1.5,
            "mid": -0.15,
            "regime_confidence": 0.25,
            "distance_from_spot": -0.02,
        }
        return cls(version=model_version, weights=default_weights)

    def featurize_candidate(self, feature_row, regime_state, fly_spec, fly_quote) -> dict:
        return {
            **feature_row.features,
            "regime_confidence": float(regime_state.confidence),
            "regime_is_high_vol": 1.0 if regime_state.regime == "high_vol" else 0.0,
            "bias_is_bullish": 1.0 if regime_state.bias == "bullish" else 0.0,
            "theta": float(fly_quote.theta),
            "spread": float(fly_quote.spread),
            "liquidity_score": float(fly_quote.liquidity_score),
            "mid": float(fly_quote.mid),
            "k_center": float(fly_spec.k_center),
            "wing": float(fly_spec.wing),
            "distance_from_spot": abs(float(fly_spec.k_center) - float(feature_row.features.get("spot", fly_spec.k_center))),
        }

    def score_candidates(self, rows: list[dict]) -> list[float]:
        scores: list[float] = []
        for row in rows:
            score = 0.0
            for k, w in self.weights.items():
                score += w * float(row.get(k, 0.0))
            scores.append(score)
        return scores

    def select(self, candidates_scored: list[dict], risk_state, thresholds: dict) -> TradeIntent | None:
        if not candidates_scored:
            return None
        scored = sorted(candidates_scored, key=lambda x: x["score"], reverse=True)
        scores = [r["score"] for r in scored]
        if self.should_abstain(scores, thresholds):
            return None

        best = scored[0]
        qty_cap = int(thresholds.get("max_qty", 1))
        qty = max(1, min(qty_cap, 2))

        return TradeIntent(
            ts=best["fly_spec"].ts,
            fly_spec=best["fly_spec"],
            qty=qty,
            entry_mode="model",
            tags={"model_version": self.version, "score": f"{best['score']:.4f}"},
        )

    def should_abstain(self, scores: list[float], context: dict) -> bool:
        if not scores:
            return True
        min_score = float(context.get("min_score", 0.05))
        min_margin = float(context.get("min_margin", 0.02))
        if scores[0] < min_score:
            return True
        if len(scores) > 1 and (scores[0] - scores[1]) < min_margin:
            return True
        return False

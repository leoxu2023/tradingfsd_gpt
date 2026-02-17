from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import TradeIntent


@dataclass
class ModelPolicy:
    version: str
    weights: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    feature_schema: list[str] = field(default_factory=list)
    feature_mean: dict[str, float] = field(default_factory=dict)
    feature_std: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    model_type: str = "heuristic"

    @classmethod
    def load(cls, model_version: str, model_dir: Path | None = None, registry_dir: Path | None = None) -> "ModelPolicy":
        heuristic = cls._default_heuristic(model_version)

        candidate_paths: list[Path] = []
        if model_dir is not None:
            candidate_paths.append(model_dir / model_version / "model.json")
        else:
            candidate_paths.append(Path("runtime/models") / model_version / "model.json")

        if registry_dir is not None:
            candidate_paths.append(registry_dir / "models" / f"{model_version}.json")

        for path in candidate_paths:
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return cls(
                    version=payload.get("version", model_version),
                    weights={k: float(v) for k, v in payload.get("weights", {}).items()},
                    intercept=float(payload.get("intercept", 0.0)),
                    feature_schema=list(payload.get("feature_schema", [])),
                    feature_mean={k: float(v) for k, v in payload.get("feature_mean", {}).items()},
                    feature_std={k: float(v) for k, v in payload.get("feature_std", {}).items()},
                    thresholds={k: float(v) for k, v in payload.get("thresholds", {}).items()},
                    model_type="linear",
                )
            except Exception:
                continue

        return heuristic

    @staticmethod
    def _default_heuristic(version: str) -> "ModelPolicy":
        return ModelPolicy(
            version=version,
            weights={
                "theta": 1.25,
                "liquidity_score": 0.01,
                "spread": -1.4,
                "mid": -0.18,
                "regime_confidence": 0.18,
                "abs_moneyness": -0.5,
                "wing_pct": -0.2,
            },
            thresholds={
                "min_score": -1.0,
                "min_margin": 0.0,
                "high_score": -0.7,
                "aggressive_max_spread": 0.35,
                "max_qty": 1,
            },
            model_type="heuristic",
        )

    def featurize_candidate(self, feature_row: Any, regime_state: Any, fly_spec: Any, fly_quote: Any) -> dict:
        spot = float(feature_row.features.get("spot", fly_spec.k_center))
        moneyness = (float(fly_spec.k_center) - spot) / max(spot, 1e-6)
        wing_pct = float(fly_spec.wing) / max(spot, 1e-6)
        spread = float(fly_quote.spread)
        mid = float(fly_quote.mid)

        return {
            **feature_row.features,
            "regime_confidence": float(regime_state.confidence),
            "regime_is_high_vol": 1.0 if regime_state.regime == "high_vol" else 0.0,
            "bias_is_bullish": 1.0 if regime_state.bias == "bullish" else 0.0,
            "bias_is_bearish": 1.0 if regime_state.bias == "bearish" else 0.0,
            "direction_is_call": 1.0 if fly_spec.direction == "call" else 0.0,
            "direction_is_put": 1.0 if fly_spec.direction == "put" else 0.0,
            "theta": float(fly_quote.theta),
            "spread": spread,
            "liquidity_score": float(fly_quote.liquidity_score),
            "mid": mid,
            "k_center": float(fly_spec.k_center),
            "wing": float(fly_spec.wing),
            "distance_from_spot": abs(float(fly_spec.k_center) - spot),
            "moneyness": float(moneyness),
            "abs_moneyness": abs(float(moneyness)),
            "wing_pct": float(wing_pct),
            "theta_per_dollar": float(fly_quote.theta) / max(abs(mid), 1e-6),
            "liquidity_per_spread": float(fly_quote.liquidity_score) / max(spread, 1e-6),
        }

    def score_candidates(self, rows: list[dict]) -> list[float]:
        return [self._score_row(row) for row in rows]

    def _score_row(self, row: dict) -> float:
        if self.model_type == "linear" and self.feature_schema:
            x = []
            for name in self.feature_schema:
                val = float(row.get(name, 0.0))
                mu = float(self.feature_mean.get(name, 0.0))
                sigma = float(self.feature_std.get(name, 1.0))
                if sigma == 0.0:
                    sigma = 1.0
                x.append((val - mu) / sigma)
            x_arr = np.asarray(x, dtype=float)
            w_arr = np.asarray([float(self.weights.get(name, 0.0)) for name in self.feature_schema], dtype=float)
            return float(self.intercept + np.dot(x_arr, w_arr))

        score = 0.0
        for k, w in self.weights.items():
            score += float(w) * float(row.get(k, 0.0))
        return float(score)

    def select(self, candidates_scored: list[dict], risk_state: Any, thresholds: dict | None) -> TradeIntent | None:
        if not candidates_scored:
            return None

        cfg = {**self.thresholds, **(thresholds or {})}
        scored = sorted(candidates_scored, key=lambda x: x["score"], reverse=True)
        scores = [r["score"] for r in scored]
        if self.should_abstain(scores, cfg):
            return None

        best = scored[0]
        best_score = float(best["score"])
        margin = best_score - float(scores[1]) if len(scores) > 1 else best_score

        qty_cap = int(cfg.get("max_qty", 1))
        high_score = float(cfg.get("high_score", float(cfg.get("min_score", -1.0)) + 0.25))
        qty = 2 if (qty_cap >= 2 and best_score >= high_score and margin > 0) else 1

        spread = float(best["fly_quote"].spread)
        aggressive_spread = float(cfg.get("aggressive_max_spread", 0.35))
        entry_mode = "aggressive" if best_score >= high_score and spread <= aggressive_spread else "passive"

        return TradeIntent(
            ts=best["fly_spec"].ts,
            fly_spec=best["fly_spec"],
            qty=qty,
            entry_mode=entry_mode,
            tags={
                "model_version": self.version,
                "score": f"{best_score:.4f}",
                "margin": f"{margin:.4f}",
                "entry_mode": entry_mode,
            },
        )

    def should_abstain(self, scores: list[float], context: dict) -> bool:
        if not scores:
            return True
        min_score = float(context.get("min_score", -1.0))
        min_margin = float(context.get("min_margin", 0.0))
        if float(scores[0]) < min_score:
            return True
        if len(scores) > 1 and (float(scores[0]) - float(scores[1])) < min_margin:
            return True
        return False

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date
from typing import Any

from ..contracts import RiskConfig


@dataclass
class RiskParamOptimizer:
    backtest_engine: Any
    policy_version: str
    date_range: tuple[date, date]
    seed: int = 7

    def define_search_space(self) -> dict:
        return {
            "trailing_stop_pct": (0.2, 0.6),
            "max_single_entry_price": (2.0, 8.0),
            "max_batch_total_price": (4.0, 14.0),
            "max_total_premium_risk": (500.0, 2500.0),
            "max_spread": (0.2, 1.5),
        }

    def objective(self, risk_cfg: RiskConfig) -> float:
        report = self.backtest_engine.run(self.date_range, self.policy_version, risk_cfg)
        daily_pnls = [d.realized_pnl + d.unrealized_pnl for d in report.day_results]
        max_drawdown_proxy = abs(min(daily_pnls)) if daily_pnls else 0.0
        stop_penalty = report.stop_days * 50.0
        return float(report.total_pnl - 0.3 * max_drawdown_proxy - stop_penalty)

    def optimize(self, n_trials: int) -> tuple[RiskConfig, dict]:
        rng = random.Random(self.seed)
        space = self.define_search_space()

        best_score = float("-inf")
        best_cfg = RiskConfig()
        trials: list[dict] = []

        for i in range(n_trials):
            cfg = RiskConfig(
                trailing_stop_pct=rng.uniform(*space["trailing_stop_pct"]),
                max_single_entry_price=rng.uniform(*space["max_single_entry_price"]),
                max_batch_total_price=rng.uniform(*space["max_batch_total_price"]),
                max_total_premium_risk=rng.uniform(*space["max_total_premium_risk"]),
                max_spread=rng.uniform(*space["max_spread"]),
            )
            score = self.objective(cfg)
            trials.append({"trial": i + 1, "score": score, "cfg": cfg})
            if score > best_score:
                best_score = score
                best_cfg = cfg

        study_report = {
            "trials": n_trials,
            "best_score": best_score,
            "scores": [t["score"] for t in trials],
        }
        return best_cfg, study_report

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .contracts import BacktestReport, RiskConfig
from .policy import ModelPolicy


@dataclass
class BacktestEngine:
    simulator: Any
    data_provider: Any
    exec_sim: Any
    model_dir: Path = Path("runtime/models")
    registry_dir: Path = Path("runtime/registry")

    def run(self, date_range: tuple[date, date], policy_version: str, risk_cfg_version) -> BacktestReport:
        policy = ModelPolicy.load(policy_version, model_dir=self.model_dir, registry_dir=self.registry_dir)
        risk_cfg = risk_cfg_version if isinstance(risk_cfg_version, RiskConfig) else RiskConfig()

        day_results = []
        for d in self._iter_dates(*date_range):
            day_results.append(self.simulator.simulate_day(d, policy, risk_cfg, self.data_provider, self.exec_sim))

        total_pnl = sum(x.realized_pnl + x.unrealized_pnl for x in day_results)
        total_trades = sum(len(x.trades) for x in day_results)
        stop_days = sum(1 for x in day_results if x.stopped)

        return BacktestReport(
            day_results=day_results,
            total_pnl=float(total_pnl),
            total_trades=total_trades,
            stop_days=stop_days,
            metrics={
                "avg_daily_pnl": float(total_pnl / max(1, len(day_results))),
                "trade_days": float(len(day_results)),
            },
        )

    def baseline_run(self, heuristic_policy: ModelPolicy, date_range: tuple[date, date] | None = None) -> BacktestReport:
        if date_range is None:
            today = date.today()
            date_range = (today, today)
        risk_cfg = RiskConfig()
        day_results = [
            self.simulator.simulate_day(d, heuristic_policy, risk_cfg, self.data_provider, self.exec_sim)
            for d in self._iter_dates(*date_range)
        ]
        total_pnl = sum(x.realized_pnl + x.unrealized_pnl for x in day_results)
        total_trades = sum(len(x.trades) for x in day_results)
        return BacktestReport(day_results=day_results, total_pnl=total_pnl, total_trades=total_trades, stop_days=0)

    @staticmethod
    def _iter_dates(start: date, end: date):
        d = start
        while d <= end:
            yield d
            d += timedelta(days=1)

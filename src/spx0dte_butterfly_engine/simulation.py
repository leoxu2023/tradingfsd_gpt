from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from .calendar import SessionCalendar
from .contracts import DayResult, FeatureRow, Position, RiskConfig, TradeIntent, TradeOutcome
from .execution import ExecutionManager
from .pricing import OptionPricer
from .regime import RegimeEngine
from .risk import RiskManager
from .strategy import CandidateGenerator


@dataclass
class Simulator:
    calendar: SessionCalendar
    feature_engine: Any
    regime_engine: RegimeEngine
    candidate_gen: CandidateGenerator
    pricer: OptionPricer
    risk_mgr: RiskManager
    exec_mgr: ExecutionManager

    def simulate_day(self, session_date: date, policy: Any, risk_cfg: RiskConfig, data_provider: Any, exec_sim: Any) -> DayResult:
        self.risk_mgr.cfg = risk_cfg
        session = self.calendar.get_session(session_date)
        aligned = data_provider.get_aligned_bars(session_date)
        features_df = self.feature_engine.compute_features(aligned, session)

        risk_state = self.risk_mgr.start_day(session_date, equity_start=10000.0)
        positions: list[Position] = []
        trade_outcomes: list[TradeOutcome] = []

        for _, row in features_df.iterrows():
            ts = pd.to_datetime(row["ts"]).to_pydatetime()
            if not self.calendar.is_in_session(ts):
                continue

            chain = data_provider.get_chain_snapshot(ts)
            marks = {p.position_id: self.pricer.mark_position(p, chain, method="model") for p in positions if p.state == "OPEN"}
            risk_state = self.risk_mgr.update_marks(ts, positions, marks, risk_state)

            # Exit management on every tick.
            for pos in [p for p in positions if p.state == "OPEN"]:
                should_exit, reason = self.risk_mgr.should_exit(ts, pos, risk_state)
                if not should_exit:
                    continue
                q = self.pricer.quote_fly(chain, pos.fly_spec)
                exit_order = self.exec_mgr.build_exit_order(pos, q)
                fill = self.exec_mgr.execute_exit(exit_order, exec_sim)
                if not fill or fill.status != "FILLED":
                    continue
                risk_state = self.risk_mgr.on_exit(ts, pos, fill, risk_state)
                realized = (fill.avg_price - pos.entry_fill.avg_price) * pos.qty * 100.0
                trade_outcomes.append(
                    TradeOutcome(
                        position_id=pos.position_id,
                        entry_ts=pos.entry_fill.ts,
                        exit_ts=fill.ts,
                        realized_pnl=realized,
                        max_unrealized=pos.peak_unrealized,
                        stop_hit=(reason == "trailing_stop"),
                        hold_to_close=False,
                        metrics={"exit_reason": 1.0 if reason == "trailing_stop" else 0.0},
                    )
                )

            if not self.calendar.is_in_entry_window(ts, session):
                continue

            can_enter, _ = self.risk_mgr.can_enter(ts, risk_state)
            if not can_enter:
                continue

            spot = data_provider.get_spot(ts)
            feature_map = {k: float(row[k]) for k in features_df.columns if k != "ts"}
            feature_map["spot"] = float(spot)
            feature_row = FeatureRow(ts=ts, features=feature_map)
            regime_state = self.regime_engine.infer(ts, feature_row)

            candidates = self.candidate_gen.generate(
                ts=ts,
                spot=spot,
                chain=chain,
                regime=regime_state,
                constraints={
                    "candidate_grid": {
                        "strike_step": 5,
                        "center_offsets": [-20, -10, -5, 0, 5, 10, 20],
                        "wings": [10, 15, 20],
                        "directions": ["call", "put"],
                    }
                },
            )
            candidates = self.candidate_gen.filter_theta_positive(candidates, chain, self.pricer)
            quoted = self.candidate_gen.attach_quotes(candidates, chain, self.pricer)

            scored_rows = []
            for spec, quote in quoted:
                if quote.ask <= 0 or quote.ask > risk_cfg.max_single_entry_price:
                    continue
                is_ok, _ = self.exec_mgr.safety_checks(
                    quote,
                    staleness=0.0,
                    max_spread=risk_cfg.max_spread,
                    max_slip=risk_cfg.max_slippage,
                )
                if not is_ok:
                    continue
                row_features = policy.featurize_candidate(feature_row, regime_state, spec, quote)
                score = policy.score_candidates([row_features])[0]
                scored_rows.append({"fly_spec": spec, "fly_quote": quote, "score": score})

            intent: TradeIntent | None = policy.select(
                scored_rows,
                risk_state,
                thresholds={"min_score": -0.2, "min_margin": 0.0, "max_qty": 1},
            )
            if not intent:
                continue

            selected = next(x for x in scored_rows if x["fly_spec"] == intent.fly_spec)
            entry_order = self.exec_mgr.build_entry_order(intent, selected["fly_quote"])
            fill = self.exec_mgr.execute_entry(entry_order, exec_sim)
            if not fill or fill.status != "FILLED":
                continue

            risk_state = self.risk_mgr.on_entry(ts, intent, fill, risk_state)
            positions.append(
                Position(
                    position_id=f"{session_date.isoformat()}-{len(positions)+1}",
                    fly_spec=intent.fly_spec,
                    qty=intent.qty,
                    entry_fill=fill,
                )
            )

            if risk_state.current_batch_combos >= risk_cfg.max_combos_per_batch:
                self.risk_mgr.on_batch_end(ts, risk_state.unrealized, risk_state)

            if self.risk_mgr.kill_switch(risk_state):
                break

        # Flatten remaining open positions at end of day.
        if positions:
            close_chain = data_provider.get_chain_snapshot(session.close_dt)
            for pos in [p for p in positions if p.state == "OPEN"]:
                q = self.pricer.quote_fly(close_chain, pos.fly_spec)
                exit_order = self.exec_mgr.build_exit_order(pos, q)
                fill = self.exec_mgr.execute_exit(exit_order, exec_sim)
                if not fill or fill.status != "FILLED":
                    continue
                risk_state = self.risk_mgr.on_exit(session.close_dt, pos, fill, risk_state)
                realized = (fill.avg_price - pos.entry_fill.avg_price) * pos.qty * 100.0
                trade_outcomes.append(
                    TradeOutcome(
                        position_id=pos.position_id,
                        entry_ts=pos.entry_fill.ts,
                        exit_ts=fill.ts,
                        realized_pnl=realized,
                        max_unrealized=pos.peak_unrealized,
                        stop_hit=False,
                        hold_to_close=True,
                    )
                )

        return DayResult(
            date=session_date,
            trades=trade_outcomes,
            realized_pnl=risk_state.realized,
            unrealized_pnl=risk_state.unrealized,
            stopped=risk_state.stopped,
            metrics={
                "batch_count": float(risk_state.batch_count),
                "loss_streak": float(risk_state.loss_streak),
                "net_pnl": float(risk_state.net_pnl),
            },
        )

    def simulate_trade(self, trade_intent: TradeIntent, replay: Any, exec_sim: Any, risk_mgr: RiskManager):
        raise NotImplementedError("simulate_trade is deferred; use simulate_day for MVP")

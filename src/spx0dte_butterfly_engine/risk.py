from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from .contracts import FillResult, Position, RiskConfig, RiskState, TradeIntent


@dataclass
class RiskManager:
    cfg: RiskConfig

    def start_day(self, session_date: date, equity_start: float = 10000.0) -> RiskState:
        return RiskState(date=session_date, equity_start=equity_start)

    def can_enter(self, ts: datetime, risk_state: RiskState) -> tuple[bool, str]:
        if self.kill_switch(risk_state):
            return False, "kill_switch"
        if risk_state.cooldown_until and ts < risk_state.cooldown_until:
            return False, f"cooldown_until_{risk_state.cooldown_until.isoformat()}"

        if risk_state.current_batch_combos == 0 and risk_state.batch_count >= self.cfg.max_batches_per_day:
            return False, "max_batches_reached"

        if risk_state.current_batch_combos >= self.cfg.max_combos_per_batch:
            return False, "max_combos_in_batch"

        return True, "ok"

    def on_entry(self, ts: datetime, trade_intent: TradeIntent, fill: FillResult, risk_state: RiskState) -> RiskState:
        if risk_state.current_batch_combos == 0:
            risk_state.batch_count += 1

        risk_state.current_batch_combos += 1
        risk_state.current_batch_cost += fill.avg_price * fill.filled_qty

        if fill.avg_price > self.cfg.max_single_entry_price:
            risk_state.stopped = True

        if risk_state.current_batch_cost > self.cfg.max_batch_total_price:
            risk_state.stopped = True

        if (risk_state.current_batch_cost * 100.0) > self.cfg.max_total_premium_risk:
            risk_state.stopped = True

        return risk_state

    def update_marks(self, ts: datetime, positions: list[Position], marks: dict[str, float], risk_state: RiskState) -> RiskState:
        unrealized = 0.0
        for pos in positions:
            if pos.state != "OPEN":
                continue
            mark = float(marks.get(pos.position_id, pos.entry_fill.avg_price))
            pnl = (mark - pos.entry_fill.avg_price) * pos.qty * 100.0
            pos.current_unrealized = pnl
            pos.peak_unrealized = max(pos.peak_unrealized, pnl)
            if pos.peak_unrealized > 0:
                pos.trail_level = pos.peak_unrealized * (1.0 - self.cfg.trailing_stop_pct)
            unrealized += pnl

        risk_state.unrealized = unrealized
        if risk_state.net_pnl <= self.cfg.daily_stop_loss:
            risk_state.stopped = True
        return risk_state

    def should_exit(self, ts: datetime, position: Position, risk_state: RiskState) -> tuple[bool, str]:
        if position.state != "OPEN":
            return False, "not_open"

        if risk_state.net_pnl <= self.cfg.daily_stop_loss:
            return True, "daily_stop"

        if risk_state.stopped:
            return True, "risk_stopped"

        if position.peak_unrealized > 0 and position.current_unrealized <= position.trail_level:
            return True, "trailing_stop"

        return False, "hold"

    def on_exit(self, ts: datetime, position: Position, fill: FillResult, risk_state: RiskState) -> RiskState:
        realized_trade = (fill.avg_price - position.entry_fill.avg_price) * position.qty * 100.0
        risk_state.realized += realized_trade
        risk_state.unrealized -= position.current_unrealized

        if realized_trade < 0:
            risk_state.loss_streak += 1
            if risk_state.loss_streak == 1:
                risk_state.cooldown_until = ts + timedelta(minutes=self.cfg.cooldown_after_1_loss_min)
            elif risk_state.loss_streak == 2:
                risk_state.cooldown_until = ts + timedelta(minutes=self.cfg.cooldown_after_2_loss_min)
            elif risk_state.loss_streak >= self.cfg.stop_after_losses:
                risk_state.stopped = True
        else:
            risk_state.loss_streak = 0
            risk_state.cooldown_until = None

        position.state = "CLOSED"

        if risk_state.current_batch_combos > 0:
            risk_state.current_batch_combos -= 1
        if risk_state.current_batch_combos == 0:
            risk_state.current_batch_cost = 0.0

        if risk_state.net_pnl <= self.cfg.daily_stop_loss:
            risk_state.stopped = True

        return risk_state

    def on_batch_end(self, ts: datetime, batch_pnl: float, risk_state: RiskState) -> RiskState:
        risk_state.current_batch_combos = 0
        risk_state.current_batch_cost = 0.0

        if batch_pnl < 0 and risk_state.loss_streak >= self.cfg.stop_after_losses:
            risk_state.stopped = True
        return risk_state

    def kill_switch(self, risk_state: RiskState) -> bool:
        if risk_state.stopped:
            return True
        if risk_state.loss_streak >= self.cfg.stop_after_losses:
            return True
        if risk_state.net_pnl <= self.cfg.daily_stop_loss:
            return True
        return False

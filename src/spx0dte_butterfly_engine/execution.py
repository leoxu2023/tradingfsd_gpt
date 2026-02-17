from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from .broker import BrokerAdapter
from .contracts import FillResult, FlyQuote, OrderIntent, Position, TradeIntent


@dataclass
class ExecutionManager:
    entry_timeout_sec: int = 300
    exit_timeout_sec: int = 120
    poll_step_sec: int = 15

    def build_entry_order(self, trade_intent: TradeIntent, fly_quote: FlyQuote) -> OrderIntent:
        mode = str(trade_intent.entry_mode).lower()
        if mode == "aggressive":
            # Start closer to ask for higher-conviction entries.
            start_limit = float(fly_quote.mid + 0.35 * max(0.0, fly_quote.ask - fly_quote.mid))
            urgency = "normal"
        else:
            start_limit = float(fly_quote.mid)
            urgency = "low"
        return OrderIntent(
            ts=trade_intent.ts,
            legs=trade_intent.fly_spec.legs,
            limit_price=start_limit,
            tif="DAY",
            deadline=trade_intent.ts + timedelta(seconds=self.entry_timeout_sec),
            urgency=urgency,
            side="buy",
        )

    def build_exit_order(self, position: Position, fly_quote: FlyQuote) -> OrderIntent:
        return OrderIntent(
            ts=position.fly_spec.ts,
            legs=position.fly_spec.legs,
            limit_price=float(fly_quote.mid),
            tif="DAY",
            deadline=position.fly_spec.ts + timedelta(seconds=self.exit_timeout_sec),
            urgency="high",
            side="sell",
        )

    def execute_entry(self, order_intent: OrderIntent, broker: BrokerAdapter) -> FillResult | None:
        return self._execute(order_intent, broker, mode="entry")

    def execute_exit(self, order_intent: OrderIntent, broker: BrokerAdapter) -> FillResult | None:
        return self._execute(order_intent, broker, mode="exit")

    def reprice_limit(self, mode: str, fly_quote: FlyQuote, elapsed: float) -> float:
        if mode == "entry":
            # Slow walk from mid toward ask over 5 minutes.
            progress = min(1.0, elapsed / 300.0)
            return float(fly_quote.mid + progress * max(0.0, fly_quote.ask - fly_quote.mid))

        # Fast walk from mid toward bid over 2 minutes.
        progress = min(1.0, elapsed / 120.0)
        return float(fly_quote.mid - progress * max(0.0, fly_quote.mid - fly_quote.bid))

    def safety_checks(self, fly_quote: FlyQuote, staleness: float, max_spread: float, max_slip: float) -> tuple[bool, str]:
        if staleness > 15:
            return False, "stale_quote"
        if fly_quote.spread > max_spread:
            return False, "spread_blowout"
        est_slip = fly_quote.spread * (0.7 if fly_quote.spread > 0.2 else 0.3)
        if est_slip > max_slip:
            return False, "slippage_too_high"
        return True, "ok"

    def _execute(self, order_intent: OrderIntent, broker: BrokerAdapter, mode: str) -> FillResult | None:
        order_id = broker.place_combo_limit(order_intent)

        timeout = self.entry_timeout_sec if mode == "entry" else self.exit_timeout_sec
        elapsed = 0.0
        while elapsed <= timeout:
            fill = broker.get_fill(order_id)
            if fill and fill.status == "FILLED" and fill.filled_qty > 0:
                return fill

            ref_quote = FlyQuote(
                mid=order_intent.limit_price,
                bid=max(0.0, order_intent.limit_price - 0.15),
                ask=order_intent.limit_price + 0.15,
                greeks={},
                theta=0.0,
                spread=0.3,
                liquidity_score=50.0,
            )
            new_limit = self.reprice_limit(mode=mode, fly_quote=ref_quote, elapsed=elapsed)
            broker.modify_order(order_id, new_limit)
            elapsed += self.poll_step_sec

        broker.cancel_order(order_id)
        return broker.get_fill(order_id)

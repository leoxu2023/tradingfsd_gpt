import importlib.util
import unittest
from datetime import date, datetime
from zoneinfo import ZoneInfo

PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None

if PANDAS_AVAILABLE:
    from spx0dte_butterfly_engine.contracts import FillResult, FlyQuote, FlySpec, Position, RiskConfig, TradeIntent
    from spx0dte_butterfly_engine.execution import ExecutionManager
    from spx0dte_butterfly_engine.risk import RiskManager


@unittest.skipUnless(PANDAS_AVAILABLE, "requires pandas dependency")
class RiskExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tz = ZoneInfo("America/New_York")
        self.ts = datetime(2026, 2, 2, 10, 0, tzinfo=self.tz)

    def _make_fly(self):
        return FlySpec(
            ts=self.ts,
            expiry=date(2026, 2, 2),
            direction="call",
            k_center=6000.0,
            wing=10.0,
            legs=[(5990.0, "C", 1, "buy"), (6000.0, "C", 2, "sell"), (6010.0, "C", 1, "buy")],
        )

    def test_trailing_stop_and_cooldown(self):
        cfg = RiskConfig(trailing_stop_pct=0.35)
        mgr = RiskManager(cfg=cfg)
        state = mgr.start_day(self.ts.date())

        fly = self._make_fly()
        intent = TradeIntent(ts=self.ts, fly_spec=fly, qty=1, entry_mode="model")
        entry_fill = FillResult(
            order_id="e1",
            ts=self.ts,
            avg_price=2.0,
            filled_qty=1,
            status="FILLED",
            fees=1.0,
            slippage_est=0.01,
        )
        state = mgr.on_entry(self.ts, intent, entry_fill, state)
        pos = Position(position_id="p1", fly_spec=fly, qty=1, entry_fill=entry_fill)

        # Peak unrealized = +100
        state = mgr.update_marks(self.ts, [pos], {"p1": 3.0}, state)
        # Pullback to +40, below trail level (65)
        state = mgr.update_marks(self.ts, [pos], {"p1": 2.4}, state)

        should_exit, reason = mgr.should_exit(self.ts, pos, state)
        self.assertTrue(should_exit)
        self.assertEqual(reason, "trailing_stop")

        loss_fill = FillResult(
            order_id="x1",
            ts=self.ts,
            avg_price=1.5,
            filled_qty=1,
            status="FILLED",
            fees=1.0,
            slippage_est=0.02,
        )
        state = mgr.on_exit(self.ts, pos, loss_fill, state)
        self.assertEqual(state.loss_streak, 1)
        self.assertIsNotNone(state.cooldown_until)

    def test_reprice_walk_rules(self):
        exec_mgr = ExecutionManager()
        quote = FlyQuote(
            mid=2.0,
            bid=1.8,
            ask=2.4,
            greeks={},
            theta=0.1,
            spread=0.6,
            liquidity_score=50.0,
        )

        self.assertAlmostEqual(exec_mgr.reprice_limit("entry", quote, 0), 2.0)
        self.assertAlmostEqual(exec_mgr.reprice_limit("entry", quote, 300), 2.4)
        self.assertAlmostEqual(exec_mgr.reprice_limit("exit", quote, 0), 2.0)
        self.assertAlmostEqual(exec_mgr.reprice_limit("exit", quote, 120), 1.8)


if __name__ == "__main__":
    unittest.main()

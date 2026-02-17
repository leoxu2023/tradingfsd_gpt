from __future__ import annotations

import math
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from .contracts import FillResult, OrderIntent, Position


NY_TZ = ZoneInfo("America/New_York")


class BrokerAdapter(ABC):
    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def place_combo_limit(self, order_intent: OrderIntent) -> str:
        raise NotImplementedError

    @abstractmethod
    def modify_order(self, order_id: str, new_limit: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_fill(self, order_id: str) -> FillResult | None:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> list[Position]:
        raise NotImplementedError

    @abstractmethod
    def get_account_snapshot(self) -> dict:
        raise NotImplementedError


@dataclass
class SimBrokerAdapter(BrokerAdapter):
    """Simple deterministic fill simulator for replay/backtest mode."""

    slippage_per_fill: float = 0.01
    _orders: dict[str, dict] = field(default_factory=dict, init=False)

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def place_combo_limit(self, order_intent: OrderIntent) -> str:
        order_id = f"sim-{uuid.uuid4().hex[:10]}"
        delay_polls = {"low": 2, "normal": 1, "high": 0}.get(order_intent.urgency, 1)
        self._orders[order_id] = {
            "intent": order_intent,
            "polls": 0,
            "delay": delay_polls,
            "status": "WORKING",
        }
        return order_id

    def modify_order(self, order_id: str, new_limit: float) -> None:
        order = self._orders.get(order_id)
        if not order or order["status"] != "WORKING":
            return
        order["intent"].limit_price = float(new_limit)

    def cancel_order(self, order_id: str) -> None:
        order = self._orders.get(order_id)
        if not order:
            return
        order["status"] = "CANCELLED"

    def get_fill(self, order_id: str) -> FillResult | None:
        order = self._orders.get(order_id)
        if not order:
            return None

        if order["status"] == "CANCELLED":
            return FillResult(
                order_id=order_id,
                ts=datetime.now(tz=NY_TZ),
                avg_price=0.0,
                filled_qty=0,
                status="CANCELLED",
                fees=0.0,
                slippage_est=0.0,
            )

        order["polls"] += 1
        if order["polls"] <= order["delay"]:
            return None

        intent = order["intent"]
        signed_slippage = self.slippage_per_fill if intent.side == "buy" else -self.slippage_per_fill
        avg_price = max(0.0, intent.limit_price + signed_slippage)
        fees = 1.0 * len(intent.legs)
        order["status"] = "FILLED"

        return FillResult(
            order_id=order_id,
            ts=datetime.now(tz=NY_TZ),
            avg_price=avg_price,
            filled_qty=1,
            status="FILLED",
            fees=fees,
            slippage_est=abs(signed_slippage),
        )

    def get_positions(self) -> list[Position]:
        return []

    def get_account_snapshot(self) -> dict:
        return {"equity": 10000.0, "timestamp": datetime.now(tz=NY_TZ)}


@dataclass
class IBBrokerAdapter(BrokerAdapter):
    """IB paper/live combo-order adapter using ib_insync."""

    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 7
    account: str = ""
    symbol: str = "SPX"
    exchange: str = "SMART"
    currency: str = "USD"
    trading_class: str = "SPXW"
    enable_order_routing: bool = True

    _ib: Any = field(default=None, init=False)
    _ib_mod: Any = field(default=None, init=False)
    _orders: dict[str, Any] = field(default_factory=dict, init=False)
    _contracts: dict[str, Any] = field(default_factory=dict, init=False)
    _sim_fallback: SimBrokerAdapter = field(default_factory=SimBrokerAdapter, init=False)

    def connect(self) -> None:
        if not self.enable_order_routing:
            self._sim_fallback.connect()
            return

        if self._ib is not None:
            return

        try:
            import ib_insync as ib_mod
        except Exception as exc:
            raise RuntimeError("ib_insync is required for IBBrokerAdapter. Install with: pip install ib-insync") from exc

        ib = ib_mod.IB()
        ib.connect(self.host, self.port, clientId=self.client_id)
        self._ib = ib
        self._ib_mod = ib_mod

    def disconnect(self) -> None:
        if not self.enable_order_routing:
            self._sim_fallback.disconnect()
            return

        if self._ib is None:
            return
        self._ib.disconnect()
        self._ib = None
        self._ib_mod = None

    def _ensure_connected(self) -> None:
        if not self.enable_order_routing:
            return
        if self._ib is None:
            self.connect()

    def place_combo_limit(self, order_intent: OrderIntent) -> str:
        if not self.enable_order_routing:
            return self._sim_fallback.place_combo_limit(order_intent)

        self._ensure_connected()
        bag = self._build_combo_contract(order_intent)

        action = "BUY" if order_intent.side.lower() == "buy" else "SELL"
        order = self._ib_mod.LimitOrder(action, totalQuantity=1, lmtPrice=float(order_intent.limit_price), tif=order_intent.tif)
        if self.account:
            order.account = self.account

        trade = self._ib.placeOrder(bag, order)
        self._ib.sleep(0.1)

        order_id = f"ib-{trade.order.orderId}"
        self._orders[order_id] = trade
        self._contracts[order_id] = bag
        return order_id

    def modify_order(self, order_id: str, new_limit: float) -> None:
        if not self.enable_order_routing:
            self._sim_fallback.modify_order(order_id, new_limit)
            return

        self._ensure_connected()
        trade = self._orders.get(order_id)
        contract = self._contracts.get(order_id)
        if trade is None or contract is None:
            return

        trade.order.lmtPrice = float(new_limit)
        self._ib.placeOrder(contract, trade.order)

    def cancel_order(self, order_id: str) -> None:
        if not self.enable_order_routing:
            self._sim_fallback.cancel_order(order_id)
            return

        self._ensure_connected()
        trade = self._orders.get(order_id)
        if trade is None:
            return
        self._ib.cancelOrder(trade.order)

    def get_fill(self, order_id: str) -> FillResult | None:
        if not self.enable_order_routing:
            return self._sim_fallback.get_fill(order_id)

        self._ensure_connected()
        trade = self._orders.get(order_id)
        if trade is None:
            return None

        status = str(getattr(trade.orderStatus, "status", "")).strip()
        if status in {"Submitted", "PreSubmitted", "PendingSubmit", "PendingCancel"}:
            return None

        if status in {"Cancelled", "ApiCancelled", "Inactive"}:
            return FillResult(
                order_id=order_id,
                ts=datetime.now(tz=NY_TZ),
                avg_price=0.0,
                filled_qty=0,
                status="CANCELLED",
                fees=0.0,
                slippage_est=0.0,
            )

        if status == "Filled":
            filled_qty = int(round(float(getattr(trade.orderStatus, "filled", 0) or 0)))
            avg_price = float(getattr(trade.orderStatus, "avgFillPrice", 0.0) or 0.0)
            if avg_price <= 0:
                avg_price = float(getattr(trade.order, "lmtPrice", 0.0) or 0.0)

            fees = 0.0
            for fill in getattr(trade, "fills", []):
                commission_report = getattr(fill, "commissionReport", None)
                if commission_report is None:
                    continue
                commission = getattr(commission_report, "commission", None)
                if commission is None or (isinstance(commission, float) and math.isnan(commission)):
                    continue
                fees += float(commission)

            side = "buy" if str(getattr(trade.order, "action", "BUY")).upper() == "BUY" else "sell"
            limit_price = float(getattr(trade.order, "lmtPrice", avg_price))
            slippage = max(0.0, avg_price - limit_price) if side == "buy" else max(0.0, limit_price - avg_price)

            return FillResult(
                order_id=order_id,
                ts=datetime.now(tz=NY_TZ),
                avg_price=avg_price,
                filled_qty=max(1, filled_qty),
                status="FILLED",
                fees=fees,
                slippage_est=float(slippage),
            )

        return None

    def get_positions(self) -> list[Position]:
        # Mapping raw IB positions back to FlySpec requires persistent order/position mapping.
        return []

    def get_account_snapshot(self) -> dict:
        if not self.enable_order_routing:
            return self._sim_fallback.get_account_snapshot()

        self._ensure_connected()
        summary_rows = self._ib.accountSummary(account=self.account or None)
        snapshot = {"timestamp": datetime.now(tz=NY_TZ).isoformat()}
        wanted = {"NetLiquidation", "CashBalance", "AvailableFunds", "ExcessLiquidity"}
        for row in summary_rows:
            tag = getattr(row, "tag", "")
            if tag in wanted:
                snapshot[tag] = getattr(row, "value", "")
        return snapshot

    def _build_combo_contract(self, order_intent: OrderIntent) -> Any:
        expiry = order_intent.ts.astimezone(NY_TZ).strftime("%Y%m%d") if order_intent.ts.tzinfo else order_intent.ts.strftime("%Y%m%d")
        combo_side = order_intent.side.lower()

        qualified_legs = []
        for strike, right, ratio, leg_side in order_intent.legs:
            opt = self._ib_mod.Option(
                symbol=self.symbol,
                lastTradeDateOrContractMonth=expiry,
                strike=float(strike),
                right=str(right).upper()[0],
                exchange=self.exchange,
                currency=self.currency,
                tradingClass=self.trading_class,
                multiplier="100",
            )
            contracts = self._ib.qualifyContracts(opt)
            if not contracts:
                raise RuntimeError(f"Could not qualify IB option contract for {self.symbol} {expiry} {strike}{right}")
            qc = contracts[0]

            effective_leg_side = str(leg_side).lower()
            if combo_side == "sell":
                effective_leg_side = "buy" if effective_leg_side == "sell" else "sell"

            action = "BUY" if effective_leg_side == "buy" else "SELL"
            combo_leg = self._ib_mod.ComboLeg(conId=qc.conId, ratio=int(ratio), action=action, exchange=self.exchange)
            qualified_legs.append(combo_leg)

        bag = self._ib_mod.Bag(symbol=self.symbol, secType="BAG", currency=self.currency, exchange=self.exchange)
        bag.comboLegs = qualified_legs
        return bag

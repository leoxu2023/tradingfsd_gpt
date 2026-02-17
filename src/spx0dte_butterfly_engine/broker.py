from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
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
    """Placeholder for IB paper adapter; implement using ib_insync in production."""

    def connect(self) -> None:
        raise NotImplementedError("IBBrokerAdapter.connect is not implemented in MVP scaffold")

    def disconnect(self) -> None:
        raise NotImplementedError("IBBrokerAdapter.disconnect is not implemented in MVP scaffold")

    def place_combo_limit(self, order_intent: OrderIntent) -> str:
        raise NotImplementedError("IBBrokerAdapter.place_combo_limit is not implemented in MVP scaffold")

    def modify_order(self, order_id: str, new_limit: float) -> None:
        raise NotImplementedError("IBBrokerAdapter.modify_order is not implemented in MVP scaffold")

    def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError("IBBrokerAdapter.cancel_order is not implemented in MVP scaffold")

    def get_fill(self, order_id: str) -> FillResult | None:
        raise NotImplementedError("IBBrokerAdapter.get_fill is not implemented in MVP scaffold")

    def get_positions(self) -> list[Position]:
        raise NotImplementedError("IBBrokerAdapter.get_positions is not implemented in MVP scaffold")

    def get_account_snapshot(self) -> dict:
        raise NotImplementedError("IBBrokerAdapter.get_account_snapshot is not implemented in MVP scaffold")

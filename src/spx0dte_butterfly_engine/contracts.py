from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd


@dataclass
class SessionInfo:
    date: date
    is_half_day: bool
    open_dt: datetime
    close_dt: datetime
    entry_start: datetime
    entry_end: datetime


@dataclass
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class AlignedBars:
    spx_df: pd.DataFrame
    vix_df: pd.DataFrame
    vix1d_df: pd.DataFrame
    quality_flags: dict[str, float | str | bool] = field(default_factory=dict)


@dataclass
class FeatureRow:
    ts: datetime
    features: dict[str, float]


@dataclass
class RegimeState:
    ts: datetime
    regime: str
    bias: str
    confidence: float


@dataclass
class ChainSnapshot:
    ts: datetime
    expiry: date
    rows_df: pd.DataFrame
    quality_flags: dict[str, float | str | bool] = field(default_factory=dict)


@dataclass
class FlySpec:
    ts: datetime
    expiry: date
    direction: str
    k_center: float
    wing: float
    legs: list[tuple[float, str, int, str]]


@dataclass
class FlyQuote:
    mid: float
    bid: float
    ask: float
    greeks: dict[str, float]
    theta: float
    spread: float
    liquidity_score: float


@dataclass
class TradeIntent:
    ts: datetime
    fly_spec: FlySpec
    qty: int
    entry_mode: str
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class OrderIntent:
    ts: datetime
    legs: list[tuple[float, str, int, str]]
    limit_price: float
    tif: str
    deadline: datetime
    urgency: str
    side: str = "buy"


@dataclass
class FillResult:
    order_id: str
    ts: datetime
    avg_price: float
    filled_qty: int
    status: str
    fees: float
    slippage_est: float


@dataclass
class Position:
    position_id: str
    fly_spec: FlySpec
    qty: int
    entry_fill: FillResult
    current_unrealized: float = 0.0
    peak_unrealized: float = 0.0
    trail_level: float = 0.0
    state: str = "OPEN"


@dataclass
class RiskConfig:
    max_combos_per_batch: int = 2
    max_batches_per_day: int = 3
    max_total_premium_risk: float = 1500.0
    max_single_entry_price: float = 6.0
    max_batch_total_price: float = 12.0
    trailing_stop_pct: float = 0.35
    daily_stop_loss: float = -1300.0
    cooldown_after_1_loss_min: int = 3
    cooldown_after_2_loss_min: int = 10
    stop_after_losses: int = 3
    entry_order_timeout_sec: int = 300
    exit_order_timeout_sec: int = 120
    max_spread: float = 1.2
    max_slippage: float = 0.6


@dataclass
class RiskState:
    date: date
    equity_start: float
    realized: float = 0.0
    unrealized: float = 0.0
    batch_count: int = 0
    loss_streak: int = 0
    cooldown_until: datetime | None = None
    stopped: bool = False
    current_batch_combos: int = 0
    current_batch_cost: float = 0.0

    @property
    def net_pnl(self) -> float:
        return self.realized + self.unrealized


@dataclass
class TradeOutcome:
    position_id: str
    entry_ts: datetime
    exit_ts: datetime
    realized_pnl: float
    max_unrealized: float
    stop_hit: bool
    hold_to_close: bool
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class PolicyConfig:
    model_version: str
    abstain_thresholds: dict[str, float]
    candidate_grid: dict[str, list[float] | float | int]
    feature_set_version: str


@dataclass
class ModelArtifacts:
    version: str
    model_path: Path
    feature_schema: list[str]
    metrics: dict[str, float]
    created_at: datetime


@dataclass
class DayResult:
    date: date
    trades: list[TradeOutcome]
    realized_pnl: float
    unrealized_pnl: float
    stopped: bool
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class BacktestReport:
    day_results: list[DayResult]
    total_pnl: float
    total_trades: int
    stop_days: int
    metrics: dict[str, float] = field(default_factory=dict)

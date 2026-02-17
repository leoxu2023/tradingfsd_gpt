from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass
class Monitoring:
    def emit_metrics(self, event_stream: list[dict]) -> dict:
        if not event_stream:
            return {"events": 0, "fills": 0, "errors": 0}
        df = pd.DataFrame(event_stream)
        return {
            "events": int(len(df)),
            "fills": int((df.get("stream_name") == "fills").sum()) if "stream_name" in df else 0,
            "errors": int((df.get("level") == "ERROR").sum()) if "level" in df else 0,
        }

    def daily_report(self, session_date: date, pnl: float, trades: int, stopped: bool) -> str:
        return (
            f"# Daily Report {session_date.isoformat()}\n\n"
            f"- Net PnL: {pnl:.2f}\n"
            f"- Trades: {trades}\n"
            f"- Risk Stopped: {stopped}\n"
        )

    def alert_on(self, kill_switch: bool = False, data_stale: bool = False, execution_failures: int = 0) -> list[str]:
        alerts = []
        if kill_switch:
            alerts.append("KILL_SWITCH_TRIGGERED")
        if data_stale:
            alerts.append("DATA_STALE")
        if execution_failures > 0:
            alerts.append("EXECUTION_FAILURES")
        return alerts

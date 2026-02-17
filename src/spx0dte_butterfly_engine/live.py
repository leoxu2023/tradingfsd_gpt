from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from .contracts import FeatureRow, Position, RiskConfig


@dataclass
class LiveOrchestrator:
    calendar: Any
    data_provider: Any
    feature_engine: Any
    regime_engine: Any
    candidate_gen: Any
    pricer: Any
    risk_mgr: Any
    exec_mgr: Any
    broker: Any
    event_sink: Any = None

    _risk_state: Any = field(default=None, init=False)
    _positions: list[Position] = field(default_factory=list, init=False)
    _session: Any = field(default=None, init=False)
    _policy: Any = field(default=None, init=False)
    _risk_cfg: RiskConfig | None = field(default=None, init=False)

    def run_paper(self, session_date: date, policy_version, risk_cfg_version) -> dict:
        self._session = self.calendar.get_session(session_date)
        self._risk_cfg = risk_cfg_version if isinstance(risk_cfg_version, RiskConfig) else RiskConfig()
        self.risk_mgr.cfg = self._risk_cfg
        self._risk_state = self.risk_mgr.start_day(session_date)
        self._policy = policy_version

        aligned = self.data_provider.get_aligned_bars(session_date)
        feat_df = self.feature_engine.compute_features(aligned, self._session)

        self.broker.connect()
        try:
            for _, row in feat_df.iterrows():
                ts = pd.to_datetime(row["ts"]).to_pydatetime()
                self.loop_tick(ts, row)
                if self.risk_mgr.kill_switch(self._risk_state):
                    self.shutdown("kill_switch")
                    break
        finally:
            self.broker.disconnect()

        return {
            "date": session_date.isoformat(),
            "realized": self._risk_state.realized,
            "unrealized": self._risk_state.unrealized,
            "stopped": self._risk_state.stopped,
            "positions": len(self._positions),
        }

    def loop_tick(self, ts, feature_row) -> None:
        if not self.calendar.is_in_session(ts):
            return

        chain = self.data_provider.get_chain_snapshot(ts)
        marks = {p.position_id: self.pricer.mark_position(p, chain, method="model") for p in self._positions if p.state == "OPEN"}
        self._risk_state = self.risk_mgr.update_marks(ts, self._positions, marks, self._risk_state)

        for pos in [p for p in self._positions if p.state == "OPEN"]:
            should_exit, reason = self.risk_mgr.should_exit(ts, pos, self._risk_state)
            if not should_exit:
                continue
            q = self.pricer.quote_fly(chain, pos.fly_spec)
            order = self.exec_mgr.build_exit_order(pos, q)
            fill = self.exec_mgr.execute_exit(order, self.broker)
            if fill and fill.status == "FILLED":
                self._risk_state = self.risk_mgr.on_exit(ts, pos, fill, self._risk_state)
                if self.event_sink:
                    self.event_sink.append_event("fills", {"ts": ts, "position_id": pos.position_id, "reason": reason})

        if not self.calendar.is_in_entry_window(ts, self._session):
            return

        can_enter, reason = self.risk_mgr.can_enter(ts, self._risk_state)
        if not can_enter:
            if self.event_sink:
                self.event_sink.append_event("risk", {"ts": ts, "can_enter": False, "reason": reason})
            return

        spot = self.data_provider.get_spot(ts)
        feat_map = {k: float(feature_row[k]) for k in feature_row.index if k != "ts"}
        feat_map["spot"] = float(spot)
        frow = FeatureRow(ts=ts, features=feat_map)
        regime = self.regime_engine.infer(ts, frow)

        candidates = self.candidate_gen.generate(
            ts=ts,
            spot=spot,
            chain=chain,
            regime=regime,
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
        scored = []
        for spec, quote in self.candidate_gen.attach_quotes(candidates, chain, self.pricer):
            feat = self._policy.featurize_candidate(frow, regime, spec, quote)
            score = self._policy.score_candidates([feat])[0]
            scored.append({"fly_spec": spec, "fly_quote": quote, "score": score})

        intent = self._policy.select(scored, self._risk_state, {"max_qty": 1})
        if not intent:
            return

        selected = next(x for x in scored if x["fly_spec"] == intent.fly_spec)
        ok, _ = self.exec_mgr.safety_checks(
            selected["fly_quote"],
            staleness=0.0,
            max_spread=self._risk_cfg.max_spread,
            max_slip=self._risk_cfg.max_slippage,
        )
        if not ok:
            return

        entry_order = self.exec_mgr.build_entry_order(intent, selected["fly_quote"])
        fill = self.exec_mgr.execute_entry(entry_order, self.broker)
        if fill and fill.status == "FILLED":
            self._risk_state = self.risk_mgr.on_entry(ts, intent, fill, self._risk_state)
            self._positions.append(
                Position(
                    position_id=f"{ts.strftime('%Y%m%d%H%M%S')}-{len(self._positions)+1}",
                    fly_spec=intent.fly_spec,
                    qty=intent.qty,
                    entry_fill=fill,
                )
            )
            if self.event_sink:
                self.event_sink.append_event("orders", {"ts": ts, "event": "entry_filled", "price": fill.avg_price})

    def shutdown(self, reason: str) -> None:
        if self.event_sink:
            self.event_sink.append_event("system", {"event": "shutdown", "reason": reason})

from __future__ import annotations

from dataclasses import dataclass

from .contracts import ChainSnapshot, FlyQuote, FlySpec
from .pricing import OptionPricer


@dataclass
class ButterflyFactory:
    def build_call_fly(self, ts, expiry, k_center: float, wing: float) -> FlySpec:
        legs = [
            (k_center - wing, "C", 1, "buy"),
            (k_center, "C", 2, "sell"),
            (k_center + wing, "C", 1, "buy"),
        ]
        return FlySpec(ts=ts, expiry=expiry, direction="call", k_center=k_center, wing=wing, legs=legs)

    def build_put_fly(self, ts, expiry, k_center: float, wing: float) -> FlySpec:
        legs = [
            (k_center + wing, "P", 1, "buy"),
            (k_center, "P", 2, "sell"),
            (k_center - wing, "P", 1, "buy"),
        ]
        return FlySpec(ts=ts, expiry=expiry, direction="put", k_center=k_center, wing=wing, legs=legs)

    def list_candidate_params(self, spot: float, grid_cfg: dict) -> list[tuple[str, float, float]]:
        step = float(grid_cfg.get("strike_step", 5.0))
        center_base = round(spot / step) * step
        offsets = grid_cfg.get("center_offsets", [-20, -10, 0, 10, 20])
        wings = grid_cfg.get("wings", [10, 15, 20])
        directions = grid_cfg.get("directions", ["call", "put"])

        params: list[tuple[str, float, float]] = []
        for direction in directions:
            for offset in offsets:
                for wing in wings:
                    params.append((direction, float(center_base + offset), float(wing)))
        return params


@dataclass
class CandidateGenerator:
    factory: ButterflyFactory

    def generate(self, ts, spot: float, chain: ChainSnapshot, regime, constraints: dict) -> list[FlySpec]:
        params = self.factory.list_candidate_params(spot, constraints.get("candidate_grid", constraints))
        out: list[FlySpec] = []
        available = set(zip(chain.rows_df["strike"].round(4), chain.rows_df["right"].str.upper()))

        for direction, k_center, wing in params:
            spec = (
                self.factory.build_call_fly(ts, chain.expiry, k_center, wing)
                if direction == "call"
                else self.factory.build_put_fly(ts, chain.expiry, k_center, wing)
            )
            if self._exists_in_chain(spec, available):
                out.append(spec)
        return out

    def filter_theta_positive(self, candidates: list[FlySpec], chain: ChainSnapshot, pricer: OptionPricer) -> list[FlySpec]:
        out: list[FlySpec] = []
        for spec in candidates:
            quote = pricer.quote_fly(chain, spec)
            if quote.theta >= 0 and quote.ask > 0:
                out.append(spec)
        return out

    def attach_quotes(
        self,
        candidates: list[FlySpec],
        chain: ChainSnapshot,
        pricer: OptionPricer,
    ) -> list[tuple[FlySpec, FlyQuote]]:
        return [(spec, pricer.quote_fly(chain, spec)) for spec in candidates]

    def _exists_in_chain(self, spec: FlySpec, available: set[tuple[float, str]]) -> bool:
        for strike, right, _, _ in spec.legs:
            if (round(float(strike), 4), right.upper()) not in available:
                return False
        return True

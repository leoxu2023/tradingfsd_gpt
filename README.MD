# spx0dte-butterfly-engine

MVP codebase for an SPX 0DTE butterfly trading engine with shared logic across backtest and paper-trade modes.

## Current scope

- Strategy: SPX 0DTE butterflies only
- Entry window: 10:00-14:00 ET (auto-shortened on half-days)
- Risk controls:
  - Combo-level caps
  - Account daily stop (`realized + unrealized <= -1300`)
  - Max 3 batches/day
  - Losing-streak cooldown circuit breaker (`1 -> 3 min`, `2 -> 10 min`, `3 -> stop day`)
- Execution behavior:
  - Entry walk: `mid -> ask` over 5 minutes
  - Exit walk: `mid -> bid` over 2 minutes
- Modes:
  - `backtest`
  - `ml-build`
  - `ml-train`
  - `ml-optimize-risk`
  - `paper-trade`
    - defaults to simulated broker
    - supports IB broker routing via `--ib-live` (requires `ib-insync`)
    - supports IB market data provider via `paper-trade --provider ib`
    - supports ThetaData snapshot cache via `--theta-chain-dir`

## Project layout

- `src/spx0dte_butterfly_engine/contracts.py`: shared dataclasses
- `src/spx0dte_butterfly_engine/risk.py`: deterministic risk state machine
- `src/spx0dte_butterfly_engine/execution.py`: entry/exit order logic + limit walk
- `src/spx0dte_butterfly_engine/simulation.py`: replay simulator
- `src/spx0dte_butterfly_engine/cli.py`: mode orchestration

## Quickstart

```bash
cd /Users/leo/Documents/TradingFSD_GPT
python3 -m venv .venv
source .venv/bin/activate
pip install '.[dev]'
spx-engine backtest --start 2025-02-10 --end 2025-02-14
```

## Live hooks

- IB broker routing (paper/live):

```bash
source .venv/bin/activate
pip install ib-insync
spx-engine paper-trade \
  --date 2025-02-10 \
  --ib-live \
  --provider ib \
  --ib-host 127.0.0.1 \
  --ib-port 7497 \
  --ib-client-id 17
```

- ThetaData chain snapshot cache (CSV or parquet):
  - use options root `SPXW` for 0DTE chain pulls
  - Directory layout supported:
    - `/path/to/chains/2025-02-10/*.csv`
    - `/path/to/chains/*20250210*.csv`
  - Required columns: `strike,right,bid,ask`
  - Optional columns: `mid,iv,delta,gamma,theta,vega,volume,expiry`

- Collect ThetaData snapshots (default root = `SPXW`) for this engine:

```bash
source .venv/bin/activate
python3 scripts/fetch_thetadata_option_snapshots.py \
  --date 20260217 \
  --root SPXW \
  --start 10:00 \
  --end 14:00 \
  --out-dir data/thetadata/chains
```

- Run paper mode using collected snapshots:

```bash
source .venv/bin/activate
PYTHONPATH=src python3 -m spx0dte_butterfly_engine.cli paper-trade \
  --date 2026-02-17 \
  --theta-chain-dir data/thetadata/chains \
  --provider local
```

## Data notes

CSV inputs currently available under `data/`:

- `SP_SPX, 5 (1).csv`
- `TVC_VIX, 5 (1).csv`
- `CBOE_DLY_VIX1D, 5 (1).csv`
- `SPY_1min.csv`

These are normalized by `DataIngestor` into New York timezone and 1-minute bars.

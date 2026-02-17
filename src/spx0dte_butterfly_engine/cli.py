from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .backtest import BacktestEngine
from .broker import IBBrokerAdapter, SimBrokerAdapter
from .calendar import SessionCalendar
from .contracts import PolicyConfig, RiskConfig
from .execution import ExecutionManager
from .features import FeatureEngine
from .ingest import DataIngestor
from .live import LiveOrchestrator
from .ml.dataset import DatasetBuilder
from .ml.optimize import RiskParamOptimizer
from .ml.train import ModelTrainer
from .policy import ModelPolicy
from .pricing import OptionPricer
from .provider import ChainSnapshotStore, IBProvider, LocalProvider
from .regime import RegimeEngine
from .registry import ModelRegistry
from .risk import RiskManager
from .simulation import Simulator
from .storage import DataStore
from .strategy import ButterflyFactory, CandidateGenerator


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _build_stack(data_dir: Path, runtime_dir: Path, theta_chain_dir: Path | None = None):
    calendar = SessionCalendar()
    ingestor = DataIngestor()
    feature_engine = FeatureEngine()
    regime_engine = RegimeEngine()
    pricer = OptionPricer()
    candidate_gen = CandidateGenerator(factory=ButterflyFactory())
    risk_mgr = RiskManager(cfg=RiskConfig())
    exec_mgr = ExecutionManager()

    provider = LocalProvider(data_dir=data_dir, ingestor=ingestor, calendar=calendar, theta_chain_dir=theta_chain_dir)
    datastore = DataStore(root=runtime_dir / "store")

    simulator = Simulator(
        calendar=calendar,
        feature_engine=feature_engine,
        regime_engine=regime_engine,
        candidate_gen=candidate_gen,
        pricer=pricer,
        risk_mgr=risk_mgr,
        exec_mgr=exec_mgr,
    )
    return {
        "calendar": calendar,
        "provider": provider,
        "datastore": datastore,
        "feature_engine": feature_engine,
        "regime_engine": regime_engine,
        "candidate_gen": candidate_gen,
        "pricer": pricer,
        "risk_mgr": risk_mgr,
        "exec_mgr": exec_mgr,
        "simulator": simulator,
    }


def cmd_backtest(args) -> int:
    stack = _build_stack(args.data_dir, args.runtime_dir, args.theta_chain_dir)
    broker = SimBrokerAdapter()
    engine = BacktestEngine(simulator=stack["simulator"], data_provider=stack["provider"], exec_sim=broker)

    risk_cfg = RiskConfig()
    report = engine.run((args.start, args.end), policy_version=args.policy_version, risk_cfg_version=risk_cfg)

    print(json.dumps(
        {
            "mode": "backtest",
            "start": args.start.isoformat(),
            "end": args.end.isoformat(),
            "policy": args.policy_version,
            "total_pnl": report.total_pnl,
            "total_trades": report.total_trades,
            "stop_days": report.stop_days,
            "metrics": report.metrics,
        },
        indent=2,
        default=str,
    ))
    return 0


def cmd_ml_build(args) -> int:
    stack = _build_stack(args.data_dir, args.runtime_dir, args.theta_chain_dir)
    policy_cfg = PolicyConfig(
        model_version=args.policy_version,
        abstain_thresholds={"min_score": 0.05, "min_margin": 0.01},
        candidate_grid={
            "strike_step": 5,
            "center_offsets": [-20, -10, -5, 0, 5, 10, 20],
            "wings": [10, 15, 20],
            "directions": ["call", "put"],
        },
        feature_set_version="v1",
    )

    builder = DatasetBuilder(
        calendar=stack["calendar"],
        feature_engine=stack["feature_engine"],
        regime_engine=stack["regime_engine"],
        candidate_gen=stack["candidate_gen"],
        pricer=stack["pricer"],
    )

    candidate_path = builder.build_candidates((args.start, args.end), policy_cfg, stack["provider"])
    labeled_path = builder.label_candidates(candidate_path, simulator=stack["simulator"], risk_cfg=RiskConfig())

    print(json.dumps({"mode": "ml-build", "candidates": str(candidate_path), "labeled": str(labeled_path)}, indent=2))
    return 0


def cmd_ml_train(args) -> int:
    path = Path(args.dataset)
    if path.suffix == ".parquet":
        try:
            df = pd.read_parquet(path)
        except Exception:
            df = pd.read_csv(path.with_suffix(".csv"))
    else:
        df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty")

    split = max(1, int(len(df) * 0.8))
    train_df = df.iloc[:split].copy()
    valid_df = df.iloc[split:].copy()
    if valid_df.empty:
        valid_df = train_df.tail(max(1, len(train_df) // 5)).copy()

    trainer = ModelTrainer(model_dir=args.runtime_dir / "models")
    artifacts = trainer.train_ranker(train_df, valid_df, params={"version": args.version})

    registry = ModelRegistry(root=args.runtime_dir / "registry")
    registry.save_model(artifacts)
    if args.promote:
        registry.promote(artifacts.version)

    print(json.dumps({"mode": "ml-train", "version": artifacts.version, "metrics": artifacts.metrics}, indent=2))
    return 0


def cmd_ml_optimize_risk(args) -> int:
    stack = _build_stack(args.data_dir, args.runtime_dir, args.theta_chain_dir)
    broker = SimBrokerAdapter()
    engine = BacktestEngine(simulator=stack["simulator"], data_provider=stack["provider"], exec_sim=broker)
    optimizer = RiskParamOptimizer(backtest_engine=engine, policy_version=args.policy_version, date_range=(args.start, args.end))
    best_cfg, report = optimizer.optimize(args.trials)

    registry = ModelRegistry(root=args.runtime_dir / "registry")
    version = args.risk_version or f"risk-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    registry.save_risk_config(version, best_cfg)

    print(
        json.dumps(
            {
                "mode": "ml-optimize-risk",
                "risk_version": version,
                "best_cfg": best_cfg.__dict__,
                "study": report,
            },
            indent=2,
        )
    )
    return 0


def cmd_paper_trade(args) -> int:
    stack = _build_stack(args.data_dir, args.runtime_dir, args.theta_chain_dir)
    registry = ModelRegistry(root=args.runtime_dir / "registry")

    policy = ModelPolicy.load(args.policy_version)
    if args.risk_version:
        risk_cfg = registry.load_risk_config(args.risk_version)
    else:
        risk_cfg = RiskConfig()

    data_provider = stack["provider"]
    provider_name = "local"
    if args.provider == "ib":
        chain_store = (
            ChainSnapshotStore(args.theta_chain_dir, max_staleness_min=args.chain_max_staleness_min)
            if args.theta_chain_dir
            else None
        )
        data_provider = IBProvider(
            host=args.ib_host,
            port=args.ib_port,
            client_id=args.ib_client_id,
            symbol=args.ib_symbol,
            exchange=args.ib_spot_exchange,
            chain_store=chain_store,
            fallback_provider=stack["provider"],
        )
        provider_name = "ib"

    broker = (
        IBBrokerAdapter(
            host=args.ib_host,
            port=args.ib_port,
            client_id=args.ib_client_id,
            account=args.ib_account,
            symbol=args.ib_symbol,
            exchange=args.ib_order_exchange,
            trading_class=args.ib_trading_class,
            enable_order_routing=True,
        )
        if args.ib_live
        else SimBrokerAdapter()
    )
    orchestrator = LiveOrchestrator(
        calendar=stack["calendar"],
        data_provider=data_provider,
        feature_engine=stack["feature_engine"],
        regime_engine=stack["regime_engine"],
        candidate_gen=stack["candidate_gen"],
        pricer=stack["pricer"],
        risk_mgr=stack["risk_mgr"],
        exec_mgr=stack["exec_mgr"],
        broker=broker,
        event_sink=stack["datastore"],
    )

    try:
        summary = orchestrator.run_paper(args.session_date, policy_version=policy, risk_cfg_version=risk_cfg)
    finally:
        if isinstance(data_provider, IBProvider):
            data_provider.disconnect()
    print(json.dumps({"mode": "paper-trade", "provider": provider_name, "broker": "ib" if args.ib_live else "sim", **summary}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spx-engine", description="SPX 0DTE butterfly engine")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--runtime-dir", type=Path, default=Path("runtime"))
    parser.add_argument("--theta-chain-dir", type=Path, default=None, help="Optional ThetaData snapshot directory.")

    sub = parser.add_subparsers(dest="mode", required=True)

    p_backtest = sub.add_parser("backtest")
    p_backtest.add_argument("--start", type=_parse_date, required=True)
    p_backtest.add_argument("--end", type=_parse_date, required=True)
    p_backtest.add_argument("--policy-version", default="heuristic-v1")
    p_backtest.set_defaults(func=cmd_backtest)

    p_ml_build = sub.add_parser("ml-build")
    p_ml_build.add_argument("--start", type=_parse_date, required=True)
    p_ml_build.add_argument("--end", type=_parse_date, required=True)
    p_ml_build.add_argument("--policy-version", default="heuristic-v1")
    p_ml_build.set_defaults(func=cmd_ml_build)

    p_ml_train = sub.add_parser("ml-train")
    p_ml_train.add_argument("--dataset", required=True)
    p_ml_train.add_argument("--version", default=f"ranker-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}")
    p_ml_train.add_argument("--promote", action="store_true")
    p_ml_train.set_defaults(func=cmd_ml_train)

    p_ml_opt = sub.add_parser("ml-optimize-risk")
    p_ml_opt.add_argument("--start", type=_parse_date, required=True)
    p_ml_opt.add_argument("--end", type=_parse_date, required=True)
    p_ml_opt.add_argument("--policy-version", default="heuristic-v1")
    p_ml_opt.add_argument("--trials", type=int, default=20)
    p_ml_opt.add_argument("--risk-version", default="")
    p_ml_opt.set_defaults(func=cmd_ml_optimize_risk)

    p_paper = sub.add_parser("paper-trade")
    p_paper.add_argument("--date", dest="session_date", type=_parse_date, default=date.today())
    p_paper.add_argument("--policy-version", default="heuristic-v1")
    p_paper.add_argument("--risk-version", default="")
    p_paper.add_argument("--provider", choices=["local", "ib"], default="local", help="Market data provider for paper loop.")
    p_paper.add_argument("--ib-live", action="store_true", help="Route combo orders through IB using ib_insync.")
    p_paper.add_argument("--ib-host", default="127.0.0.1")
    p_paper.add_argument("--ib-port", type=int, default=7497)
    p_paper.add_argument("--ib-client-id", type=int, default=17)
    p_paper.add_argument("--ib-account", default="")
    p_paper.add_argument("--ib-symbol", default="SPX")
    p_paper.add_argument("--ib-spot-exchange", default="CBOE")
    p_paper.add_argument("--ib-order-exchange", default="SMART")
    p_paper.add_argument("--ib-trading-class", default="SPXW")
    p_paper.add_argument("--chain-max-staleness-min", type=int, default=5)
    p_paper.set_defaults(func=cmd_paper_trade)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from spx0dte_butterfly_engine.provider import ChainSnapshotStore


class ChainSnapshotStoreTests(unittest.TestCase):
    def test_loads_nearest_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            day_dir = root / "2025-02-10"
            day_dir.mkdir(parents=True, exist_ok=True)

            snap_path = day_dir / "20250210_100000.csv"
            pd.DataFrame(
                [
                    {"strike": 6000, "right": "C", "bid": 2.1, "ask": 2.3, "iv": 0.2, "delta": 0.5},
                    {"strike": 6000, "right": "P", "bid": 2.2, "ask": 2.4, "iv": 0.2, "delta": -0.5},
                ]
            ).to_csv(snap_path, index=False)

            ts = datetime(2025, 2, 10, 10, 2, tzinfo=ZoneInfo("America/New_York"))
            store = ChainSnapshotStore(root=root, max_staleness_min=5)
            snap = store.load(ts)

            self.assertIsNotNone(snap)
            assert snap is not None
            self.assertEqual(snap.quality_flags.get("source"), "theta_cache")
            self.assertIn("mid", snap.rows_df.columns)
            self.assertEqual(len(snap.rows_df), 2)
            self.assertAlmostEqual(float(snap.rows_df.iloc[0]["strike"]), 6000.0)

    def test_returns_none_if_too_stale(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            day_dir = root / "2025-02-10"
            day_dir.mkdir(parents=True, exist_ok=True)

            snap_path = day_dir / "20250210_090000.csv"
            pd.DataFrame([{"strike": 6000, "right": "C", "bid": 2.1, "ask": 2.3}]).to_csv(snap_path, index=False)

            ts = datetime(2025, 2, 10, 10, 2, tzinfo=ZoneInfo("America/New_York"))
            store = ChainSnapshotStore(root=root, max_staleness_min=5)
            snap = store.load(ts)

            self.assertIsNone(snap)

    def test_scales_thetadata_strike_and_maps_expiration(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            day_dir = root / "2026-02-17"
            day_dir.mkdir(parents=True, exist_ok=True)

            snap_path = day_dir / "20260217_100000.csv"
            pd.DataFrame(
                [
                    {
                        "root": "SPXW",
                        "expiration": 20260217,
                        "strike": 6000000,
                        "right": "C",
                        "bid": 1.2,
                        "ask": 1.4,
                    }
                ]
            ).to_csv(snap_path, index=False)

            ts = datetime(2026, 2, 17, 10, 0, tzinfo=ZoneInfo("America/New_York"))
            store = ChainSnapshotStore(root=root, max_staleness_min=5)
            snap = store.load(ts)

            self.assertIsNotNone(snap)
            assert snap is not None
            self.assertEqual(float(snap.rows_df.iloc[0]["strike"]), 6000.0)
            self.assertEqual(snap.expiry.isoformat(), "2026-02-17")


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from spx0dte_butterfly_engine.policy import ModelPolicy


class _Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class PolicyModelTests(unittest.TestCase):
    def test_load_linear_model_and_thresholds(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            version = "ranker-test"
            p = model_dir / version
            p.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": version,
                "intercept": 0.1,
                "weights": {"theta": 2.0},
                "feature_schema": ["theta"],
                "feature_mean": {"theta": 0.0},
                "feature_std": {"theta": 1.0},
                "thresholds": {"min_score": 0.2, "min_margin": 0.0, "max_qty": 2},
            }
            (p / "model.json").write_text(json.dumps(payload), encoding="utf-8")

            policy = ModelPolicy.load(version, model_dir=model_dir)
            self.assertEqual(policy.model_type, "linear")
            self.assertEqual(policy.thresholds["min_score"], 0.2)
            self.assertAlmostEqual(policy.score_candidates([{"theta": 1.0}])[0], 2.1)

    def test_select_uses_threshold_and_entry_mode(self):
        policy = ModelPolicy._default_heuristic("heuristic-v1")
        policy.thresholds.update({"min_score": -1.0, "min_margin": 0.0, "high_score": 0.5, "max_qty": 2})

        ts = _Obj()
        fly = _Obj(ts=ts)
        strong = {
            "fly_spec": fly,
            "fly_quote": _Obj(spread=0.2),
            "score": 0.8,
        }
        weak = {
            "fly_spec": fly,
            "fly_quote": _Obj(spread=0.6),
            "score": 0.3,
        }

        intent = policy.select([strong, weak], None, {"max_qty": 2})
        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.entry_mode, "aggressive")
        self.assertEqual(intent.qty, 2)


if __name__ == "__main__":
    unittest.main()

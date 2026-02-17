from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import ModelArtifacts


@dataclass
class ModelTrainer:
    model_dir: Path = Path("artifacts/models")

    def train_ranker(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, params: dict | None = None) -> ModelArtifacts:
        params = params or {}
        label_col = params.get("label_col", "label_group_z")
        ridge_alpha = float(params.get("ridge_alpha", 1e-4))
        target_trade_rate = float(params.get("target_trade_rate", 0.25))

        if label_col not in train_df.columns:
            if "label_group_z" in train_df.columns:
                label_col = "label_group_z"
            elif "label_utility" in train_df.columns:
                label_col = "label_utility"
            else:
                raise ValueError("No supported label column found in training dataset")

        feature_cols = self._feature_columns(train_df, label_col)

        x_train = np.nan_to_num(train_df[feature_cols].astype(float).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(train_df[label_col].astype(float).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        x_valid = np.nan_to_num(valid_df[feature_cols].astype(float).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        y_valid = np.nan_to_num(valid_df[label_col].astype(float).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)

        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0)
        std[(std == 0.0) | (std < 1e-6)] = 1.0
        x_train_n = np.clip((x_train - mean) / std, -100.0, 100.0)
        x_valid_n = np.clip((x_valid - mean) / std, -100.0, 100.0)

        x_train_aug = np.c_[np.ones(len(x_train_n)), x_train_n]
        x_valid_aug = np.c_[np.ones(len(x_valid_n)), x_valid_n]

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            xtx = x_train_aug.T @ x_train_aug
            xty = x_train_aug.T @ y_train
        reg = ridge_alpha * np.eye(xtx.shape[0], dtype=float)
        reg[0, 0] = 0.0
        weights = np.linalg.solve(xtx + reg, xty)

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            preds = x_valid_aug @ weights

        metrics = {
            "rmse": float(np.sqrt(np.mean((preds - y_valid) ** 2))),
            "mae": float(np.mean(np.abs(preds - y_valid))),
            "corr": float(np.corrcoef(preds, y_valid)[0, 1]) if len(y_valid) > 1 else 0.0,
            "top1_hit": self._top1_hit_rate(valid_df, preds, label_col),
        }

        thresholds = self._calibrate_thresholds(valid_df, preds, target_trade_rate=target_trade_rate)

        version = params.get("version") or f"ranker-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        out_dir = self.model_dir / version
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "model.json"

        payload = {
            "version": version,
            "intercept": float(weights[0]),
            "weights": {c: float(w) for c, w in zip(feature_cols, weights[1:])},
            "feature_schema": feature_cols,
            "feature_mean": {c: float(v) for c, v in zip(feature_cols, mean)},
            "feature_std": {c: float(v) for c, v in zip(feature_cols, std)},
            "label_col": label_col,
            "thresholds": thresholds,
            "metrics": metrics,
            "created_at": datetime.utcnow().isoformat(),
        }
        model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return ModelArtifacts(
            version=version,
            model_path=model_path,
            feature_schema=feature_cols,
            metrics=metrics,
            created_at=datetime.utcnow(),
        )

    def evaluate(self, model: dict, test_df: pd.DataFrame) -> dict:
        features = model["feature_schema"]
        weights = np.array([model["weights"][k] for k in features], dtype=float)
        intercept = float(model["intercept"])
        mean = np.array([model.get("feature_mean", {}).get(k, 0.0) for k in features], dtype=float)
        std = np.array([model.get("feature_std", {}).get(k, 1.0) for k in features], dtype=float)
        std[std == 0.0] = 1.0

        label_col = model.get("label_col", "label_group_z")
        if label_col not in test_df.columns:
            label_col = "label_utility"

        x = np.nan_to_num(test_df[features].astype(float).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(test_df[label_col].astype(float).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        x_norm = np.clip((x - mean) / std, -100.0, 100.0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            preds = intercept + x_norm @ weights

        return {
            "rmse": float(np.sqrt(np.mean((preds - y) ** 2))),
            "mae": float(np.mean(np.abs(preds - y))),
            "corr": float(np.corrcoef(preds, y)[0, 1]) if len(y) > 1 else 0.0,
            "top1_hit": self._top1_hit_rate(test_df, preds, label_col),
        }

    def walk_forward_cv(self, date_splits: list[tuple[pd.DataFrame, pd.DataFrame]]) -> dict:
        scores = []
        hits = []
        for i, (train_df, test_df) in enumerate(date_splits, start=1):
            artifacts = self.train_ranker(train_df, test_df, params={"version": f"wf-{i}"})
            model = json.loads(artifacts.model_path.read_text(encoding="utf-8"))
            ev = self.evaluate(model, test_df)
            scores.append(ev["rmse"])
            hits.append(ev["top1_hit"])

        return {
            "splits": len(date_splits),
            "rmse_mean": float(np.mean(scores)) if scores else 0.0,
            "rmse_std": float(np.std(scores)) if scores else 0.0,
            "top1_hit_mean": float(np.mean(hits)) if hits else 0.0,
        }

    @staticmethod
    def _top1_hit_rate(df: pd.DataFrame, preds: np.ndarray, label_col: str) -> float:
        if "group_id" not in df.columns or len(df) == 0:
            return 0.0

        tmp = df.copy()
        tmp["_pred"] = preds
        groups = tmp.groupby("group_id", sort=False)
        total = 0
        hits = 0
        for _, g in groups:
            if g.empty:
                continue
            pred_idx = g["_pred"].idxmax()
            true_idx = g[label_col].idxmax()
            hits += int(pred_idx == true_idx)
            total += 1
        return float(hits / total) if total else 0.0

    @staticmethod
    def _calibrate_thresholds(valid_df: pd.DataFrame, preds: np.ndarray, target_trade_rate: float = 0.25) -> dict[str, float]:
        if len(valid_df) == 0:
            return {
                "min_score": -1.0,
                "min_margin": 0.0,
                "high_score": -0.75,
                "aggressive_max_spread": 0.35,
                "max_qty": 2,
            }

        tmp = valid_df.copy()
        tmp["_pred"] = preds
        if "group_id" not in tmp.columns:
            values = np.asarray(preds, dtype=float)
            q = float(np.quantile(values, max(0.0, min(1.0, 1.0 - target_trade_rate))))
            return {
                "min_score": q,
                "min_margin": 0.0,
                "high_score": float(np.quantile(values, 0.85)),
                "aggressive_max_spread": 0.35,
                "max_qty": 2,
            }

        grp = tmp.groupby("group_id", sort=False)["_pred"]
        top = grp.max()

        def _second_best(s: pd.Series) -> float:
            vals = s.nlargest(2).to_numpy()
            if len(vals) < 2:
                return float(vals[0])
            return float(vals[1])

        second = grp.apply(_second_best)
        margin = top - second

        q = max(0.0, min(1.0, 1.0 - target_trade_rate))
        min_score = float(top.quantile(q)) if len(top) else -1.0
        min_margin = float(margin.quantile(0.25)) if len(margin) else 0.0
        high_score = float(top.quantile(0.85)) if len(top) else min_score

        if not np.isfinite(min_score):
            min_score = -1.0
        if not np.isfinite(min_margin):
            min_margin = 0.0
        if not np.isfinite(high_score):
            high_score = min_score + 0.2

        return {
            "min_score": min_score,
            "min_margin": max(0.0, min_margin),
            "high_score": high_score,
            "aggressive_max_spread": 0.35,
            "max_qty": 2,
        }

    @staticmethod
    def _feature_columns(df: pd.DataFrame, label_col: str) -> list[str]:
        excluded = {label_col, "label_utility", "label_group_z", "group_id", "date", "ts", "regime", "bias", "direction"}
        cols = [c for c in df.columns if c not in excluded]
        numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric:
            raise ValueError("No numeric features available for training")
        return numeric

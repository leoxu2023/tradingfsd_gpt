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
        label_col = params.get("label_col", "label_utility")
        feature_cols = self._feature_columns(train_df, label_col)
        ridge_alpha = float(params.get("ridge_alpha", 1e-4))

        x_train = np.nan_to_num(train_df[feature_cols].astype(float).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(train_df[label_col].astype(float).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        x_valid = np.nan_to_num(valid_df[feature_cols].astype(float).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        y_valid = np.nan_to_num(valid_df[label_col].astype(float).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)

        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0)
        std[(std == 0.0) | (std < 1e-6)] = 1.0
        x_train_n = (x_train - mean) / std
        x_valid_n = (x_valid - mean) / std
        x_train_n = np.clip(x_train_n, -100.0, 100.0)
        x_valid_n = np.clip(x_valid_n, -100.0, 100.0)

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
        }

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

        x = np.nan_to_num(test_df[features].astype(float).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(test_df["label_utility"].astype(float).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        x_norm = (x - mean) / std
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            preds = intercept + x_norm @ weights

        return {
            "rmse": float(np.sqrt(np.mean((preds - y) ** 2))),
            "mae": float(np.mean(np.abs(preds - y))),
            "corr": float(np.corrcoef(preds, y)[0, 1]) if len(y) > 1 else 0.0,
        }

    def walk_forward_cv(self, date_splits: list[tuple[pd.DataFrame, pd.DataFrame]]) -> dict:
        scores = []
        for i, (train_df, test_df) in enumerate(date_splits, start=1):
            artifacts = self.train_ranker(train_df, test_df, params={"version": f"wf-{i}"})
            model = json.loads(artifacts.model_path.read_text(encoding="utf-8"))
            scores.append(self.evaluate(model, test_df)["rmse"])

        return {
            "splits": len(date_splits),
            "rmse_mean": float(np.mean(scores)) if scores else 0.0,
            "rmse_std": float(np.std(scores)) if scores else 0.0,
        }

    @staticmethod
    def _feature_columns(df: pd.DataFrame, label_col: str) -> list[str]:
        excluded = {label_col, "group_id", "date", "ts", "regime", "bias"}
        cols = [c for c in df.columns if c not in excluded]
        numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric:
            raise ValueError("No numeric features available for training")
        return numeric

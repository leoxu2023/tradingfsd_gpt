from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .contracts import ModelArtifacts, RiskConfig


@dataclass
class ModelRegistry:
    root: Path = Path("artifacts/registry")

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "models").mkdir(parents=True, exist_ok=True)
        (self.root / "risk").mkdir(parents=True, exist_ok=True)

    def save_model(self, artifacts: ModelArtifacts) -> None:
        path = self.root / "models" / f"{artifacts.version}.json"
        payload = {
            **asdict(artifacts),
            "model_path": str(artifacts.model_path),
            "created_at": artifacts.created_at.isoformat(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_model(self, version: str) -> dict:
        path = self.root / "models" / f"{version}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def promote(self, version: str) -> None:
        path = self.root / "models" / "current_model.txt"
        path.write_text(version, encoding="utf-8")

    def rollback(self) -> None:
        path = self.root / "models" / "current_model.txt"
        if path.exists():
            path.unlink()

    def save_risk_config(self, version: str, cfg: RiskConfig) -> None:
        path = self.root / "risk" / f"{version}.json"
        path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    def load_risk_config(self, version: str) -> RiskConfig:
        path = self.root / "risk" / f"{version}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return RiskConfig(**payload)

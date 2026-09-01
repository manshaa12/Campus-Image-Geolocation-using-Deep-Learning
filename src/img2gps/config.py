"""Configuration helpers for command-line training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml


@dataclass
class DataConfig:
    dataset: str = "yyss114/CIS-5190-project-6"
    split: str = "train"
    val_fraction: float = 0.10
    split_strategy: str = "location_group"  # location_group or random
    location_round_decimals: int = 5
    reference_csv: Optional[str] = None
    reference_image_dir: Optional[str] = None


@dataclass
class TrainConfig:
    epochs: int = 1
    batch_size: int = 32
    num_workers: int = 0
    seed: int = 42
    lr_backbone: float = 1e-4
    lr_head: float = 3e-4
    weight_decay: float = 1e-3
    offset_loss_weight: float = 5.0
    label_smoothing: float = 0.1
    top_k: int = 5
    pretrained: bool = True
    freeze_backbone: bool = True
    warmup_epochs: int = 0
    use_amp: bool = False
    output_dir: str = "checkpoints"
    run_name: str = "img2gps_resnet50_grid"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Optional[str | Path] = None, overrides: Optional[Mapping[str, Any]] = None) -> Config:
    """Load a YAML config and merge optional command-line overrides."""
    cfg_dict = Config().to_dict()
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        _deep_update(cfg_dict, loaded)
    if overrides:
        _deep_update(cfg_dict, overrides)
    return Config(data=DataConfig(**cfg_dict["data"]), train=TrainConfig(**cfg_dict["train"]))

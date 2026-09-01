"""Model definition for img2GPS.

The final model formulates campus geolocation as grid-based localization rather than direct
latitude/longitude regression. A ResNet-50 backbone predicts a distribution over 10 x 10 spatial
cells, while an auxiliary head learns within-cell offsets during training. The default inference
path uses soft top-k grid-center decoding for compatibility with the submitted checkpoint.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Literal, Tuple

import torch
from torch import nn
from .constants import GRID_N, LAT_MIN, LON_MIN, LAT_STEP, LON_STEP, NUM_CELLS

_cell_ids = torch.arange(NUM_CELLS)
CELL_CENTER_LATS = LAT_MIN + (_cell_ids // GRID_N).float() * LAT_STEP + 0.5 * LAT_STEP
CELL_CENTER_LONS = LON_MIN + (_cell_ids % GRID_N).float() * LON_STEP + 0.5 * LON_STEP

DecodeMode = Literal["soft_topk", "argmax_offset"]


class GPSGridModel(nn.Module):
    """ResNet-50 based grid-localization model."""

    def __init__(self, pretrained: bool = False, freeze_backbone: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        import torchvision.models as models

        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)
        children = list(backbone.children())

        # conv1 through layer3. These layers contain generic visual features and are frozen by default.
        self.frozen = nn.Sequential(*children[:7])
        self.layer4 = children[7]
        self.avgpool = children[8]

        if freeze_backbone:
            for param in self.frozen.parameters():
                param.requires_grad_(False)

        self.cell_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, NUM_CELLS),
        )
        self.offset_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
            nn.Tanh(),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # no_grad saves memory when the frozen block is not trainable, while still allowing full
        # fine-tuning if a caller initializes the model with freeze_backbone=False.
        frozen_trainable = any(param.requires_grad for param in self.frozen.parameters())
        if frozen_trainable:
            x = self.frozen(x)
        else:
            with torch.no_grad():
                x = self.frozen(x)
        x = self.layer4(x)
        return self.avgpool(x).flatten(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(x)
        cell_logits = self.cell_head(features)
        offsets = self.offset_head(features) * 0.5
        return cell_logits, offsets

    @staticmethod
    def decode_soft_topk(cell_logits: torch.Tensor, top_k: int = 5) -> torch.Tensor:
        """Decode grid logits to GPS coordinates by probability-weighted top-k centers."""
        if top_k < 1 or top_k > NUM_CELLS:
            raise ValueError(f"top_k must be in [1, {NUM_CELLS}], got {top_k}")
        probs = torch.softmax(cell_logits, dim=1)
        topk_probs, topk_cells = probs.topk(top_k, dim=1)
        topk_probs = topk_probs / topk_probs.sum(dim=1, keepdim=True)
        center_lats = CELL_CENTER_LATS.to(cell_logits.device)
        center_lons = CELL_CENTER_LONS.to(cell_logits.device)
        pred_lats = (topk_probs * center_lats[topk_cells]).sum(dim=1)
        pred_lons = (topk_probs * center_lons[topk_cells]).sum(dim=1)
        return torch.stack([pred_lats, pred_lons], dim=1)

    @staticmethod
    def decode_argmax_with_offset(cell_logits: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Decode the most likely cell plus predicted normalized offset to GPS coordinates."""
        pred_cells = cell_logits.argmax(dim=1)
        center_lats = CELL_CENTER_LATS.to(cell_logits.device)
        center_lons = CELL_CENTER_LONS.to(cell_logits.device)
        pred_lats = center_lats[pred_cells] + offsets[:, 0] * LAT_STEP
        pred_lons = center_lons[pred_cells] + offsets[:, 1] * LON_STEP
        return torch.stack([pred_lats, pred_lons], dim=1)

    def predict_tensor(self, x: torch.Tensor, top_k: int = 5, mode: DecodeMode = "soft_topk") -> torch.Tensor:
        """Return latitude/longitude predictions for an input tensor batch."""
        self.eval()
        with torch.no_grad():
            cell_logits, offsets = self.forward(x)
            if mode == "soft_topk":
                return self.decode_soft_topk(cell_logits, top_k=top_k)
            if mode == "argmax_offset":
                return self.decode_argmax_with_offset(cell_logits, offsets)
            raise ValueError(f"Unknown decode mode: {mode}")

    def predict(self, batch: Iterable[Any]) -> List[List[float]]:
        """Compatibility helper used by some course submission systems."""
        self.eval()
        with torch.no_grad():
            if isinstance(batch, torch.Tensor):
                x = batch
            elif isinstance(batch, list):
                x = torch.stack([item if isinstance(item, torch.Tensor) else torch.tensor(item) for item in batch])
            else:
                x = torch.tensor(batch)
            device = next(self.parameters()).device
            return self.predict_tensor(x.to(device)).cpu().tolist()


def load_checkpoint(model: nn.Module, checkpoint_path: str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load either a plain state_dict or a metadata checkpoint produced by train.py."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
    model.load_state_dict(checkpoint)
    return {"model_state_dict": checkpoint}


# Course-submission compatibility alias.
Model = GPSGridModel


def get_model() -> GPSGridModel:
    return GPSGridModel(pretrained=False, freeze_backbone=True)

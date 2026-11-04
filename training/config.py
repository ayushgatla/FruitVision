"""
FruitVision AI — ML Training & Pipeline Configuration
Centralized settings for hyperparameters, model architecture, hardware, and paths.
"""

import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainingConfig:
    # Model
    architecture: str = "EfficientNetB0"
    num_classes: int = 4
    classes: List[str] = field(default_factory=lambda: ["Apple_Bad", "Apple_Good", "Banana_Bad", "Banana_Good"])
    model_version: str = "2.0.0-efficientnet-b0-4cls"
    input_size: int = 224

    # Optimization
    batch_size: int = 16
    epochs_stage1: int = 5       # Head alignment (backbone frozen)
    epochs_stage2: int = 10      # Upper layer fine-tuning
    lr_stage1: float = 1e-3
    lr_stage2_backbone: float = 1e-5
    lr_stage2_head: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    early_stopping_patience: int = 4

    # Hardware & DataLoader
    num_workers: int = 0         # Safe default for Windows multiprocessing
    use_amp: bool = True         # Automatic Mixed Precision for RTX 4050
    seed: int = 42

    # Paths
    dataset_dir: str = "dataset/processed"
    weights_dir: str = "weights"
    reports_dir: str = "reports"
    best_weights_name: str = "best.pt"
    backup_weights_name: str = "fruit_efficientnet_b0.pt"

    def get_best_weights_path(self) -> str:
        return os.path.join(self.weights_dir, self.best_weights_name)

    def get_backup_weights_path(self) -> str:
        return os.path.join(self.weights_dir, self.backup_weights_name)


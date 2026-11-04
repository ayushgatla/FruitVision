"""
Unit tests for FruitVision AI EfficientNet-B0 model architecture,
dynamic head sizing, forward pass, loss calculation, and dataset loaders.
"""

import torch
import torch.nn as nn
from training.train_local import build_model
from training.dataset import FruitDataset, create_dataloaders, CLASSES_4CLS
from training.config import TrainingConfig

def test_build_model_4class():
    """Verify 4-class EfficientNetB0 structure and output shape."""
    model = build_model(num_classes=4, pretrained=False)
    assert isinstance(model.classifier, nn.Sequential)
    # Check final linear layer
    final_layer = model.classifier[-1]
    assert isinstance(final_layer, nn.Linear)
    assert final_layer.out_features == 4

    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    assert out.shape == (2, 4)

def test_build_model_3class():
    """Verify backwards-compatible 3-class EfficientNetB0 structure."""
    model = build_model(num_classes=3, pretrained=False)
    final_layer = model.classifier[-1]
    assert final_layer.out_features == 3

    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    assert out.shape == (2, 3)

def test_dataloader_batch_shapes():
    """Verify processed dataset loader yields correct tensor dimensions."""
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        dataset_root="dataset/processed",
        batch_size=8,
        num_workers=0,
        classes=CLASSES_4CLS
    )
    for images, labels in train_loader:
        assert images.shape == (8, 3, 224, 224)
        assert labels.shape == (8,)
        assert images.dtype == torch.float32
        assert labels.dtype == torch.int64
        break

def test_single_forward_backward_pass():
    """Verify loss calculation and backpropagation without crashing."""
    model = build_model(num_classes=4, pretrained=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    dummy_images = torch.randn(4, 3, 224, 224)
    dummy_labels = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    model.train()
    optimizer.zero_grad()
    outputs = model(dummy_images)
    loss = criterion(outputs, dummy_labels)
    assert loss.item() > 0.0
    loss.backward()
    optimizer.step()


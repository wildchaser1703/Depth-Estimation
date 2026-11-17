"""Minimal paired-image training loop, independent of the legacy notebook trainer."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tensornet.model import DSResNet
from tensornet.model.loss import rmse_loss, bce_dice_loss


def load_split(path):
    with np.load(path, allow_pickle=False) as archive:
        required = {"bg", "bg_fg", "depth", "mask", "scene_ids"}
        if not required <= set(archive.files):
            raise ValueError(f"Required arrays: {sorted(required)}")
        values = {k: archive[k] for k in required}
    n = len(values["bg"])
    if n < 1 or any(len(v) != n for v in values.values()):
        raise ValueError("Arrays must be nonempty and aligned")
    shape = values["bg"].shape
    if len(shape) != 4 or shape[1] != 3 or values["bg_fg"].shape != shape:
        raise ValueError("Images must be matching NCHW RGB tensors")
    for name in ["depth", "mask"]:
        if values[name].shape != (n, 1, *shape[-2:]):
            raise ValueError(f"{name} must have shape N x 1 x H x W")
    tensors = []
    for name in ["bg", "bg_fg", "depth", "mask"]:
        data = values[name]
        if not np.isfinite(data).all() or np.any(data < 0) or np.any(data > 1):
            raise ValueError("Inputs/targets must be finite and scaled to [0,1]")
        tensors.append(torch.as_tensor(data, dtype=torch.float32))
    if values["scene_ids"].ndim != 1:
        raise ValueError("scene_ids must be one dimensional")
    return TensorDataset(*tensors), set(values["scene_ids"].tolist())


def train(train_path, validation_path, output, epochs=1):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    training, train_ids = load_split(train_path)
    validation, val_ids = load_split(validation_path)
    if train_ids & val_ids:
        raise ValueError("Training and validation must not share scenes/backgrounds")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = DSResNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(epochs):
        model.train()
        for bg, composite, depth, mask in DataLoader(
            training, batch_size=1, shuffle=True
        ):
            predicted_depth, logits = model({"bg": bg, "bg_fg": composite})
            loss = rmse_loss()(predicted_depth, depth) + bce_dice_loss()(logits, mask)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    square_error = 0.0
    pixels = 0
    intersection = 0
    union = 0
    with torch.no_grad():
        for bg, composite, depth, mask in DataLoader(validation, batch_size=1):
            predicted_depth, logits = model({"bg": bg, "bg_fg": composite})
            square_error += float(((predicted_depth - depth) ** 2).sum())
            pixels += depth.numel()
            predicted = logits >= 0
            target = mask >= 0.5
            intersection += int((predicted & target).sum())
            union += int((predicted | target).sum())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "epochs": epochs,
        "seed": 42,
        "training_samples": len(training),
        "validation_samples": len(validation),
        "normalized_depth_rmse": (square_error / pixels) ** 0.5,
        "mask_iou": intersection / union if union else 1.0,
        "train_sha256": hashlib.sha256(Path(train_path).read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(
            Path(validation_path).read_bytes()
        ).hexdigest(),
        "scope": "Validation metrics on supplied paired-image data, not single-image depth estimation",
    }
    torch.save({"model_state_dict": model.state_dict()}, output / "model.pt")
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/latest"))
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            train(args.train, args.validation, args.output, args.epochs), indent=2
        )
    )


if __name__ == "__main__":
    main()

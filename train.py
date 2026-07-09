import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

LABEL_TO_IDX = {"live": 0, "dead": 1}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


def get_data_root(data_root=None):
    return Path(data_root or os.getenv("DATA_ROOT", ".")).resolve()


def load_dataframe(data_root):
    data_root = get_data_root(data_root)
    metadata = pd.read_csv(data_root / "images" / "metadata.csv")
    splits = pd.read_csv(data_root / "split.csv")
    return metadata.merge(splits, on="sample_id")


def compute_class_weights(train_df):
    counts = train_df["label"].value_counts()
    n_train = len(train_df)
    weights = {label: n_train / (2 * counts[label]) for label in LABEL_TO_IDX}
    weight_tensor = torch.tensor(
        [weights["live"], weights["dead"]], dtype=torch.float32
    )
    return weights, weight_tensor


def compute_normalization_stats(train_df, data_root):
    data_root = get_data_root(data_root)
    values = []
    for filepath in tqdm(train_df["filepath"], desc="Computing normalization stats"):
        img = np.array(Image.open(data_root / "images" / filepath), dtype=np.float32) / 255.0
        values.append(img.mean())
    mean = float(np.mean(values))
    std = float(np.std(values)) or 1.0
    return [mean, mean, mean], [std, std, std]


class CellDataset(Dataset):
    def __init__(self, dataframe, data_root, transform):
        self.df = dataframe.reset_index(drop=True)
        self.data_root = get_data_root(data_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.data_root / "images" / row.filepath)
        x = self.transform(img)
        y = LABEL_TO_IDX[row.label]
        return x, y


def get_transforms(mean, std, train=False):
    base = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
    ]
    if train:
        base.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])
    base.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transforms.Compose(base)


def build_model(num_classes=2):
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    losses = []
    y_true, y_pred = [], []

    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(x)
                loss = criterion(logits, y)

            if train_mode:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        losses.append(loss.item())
        y_true.extend(y.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = {
        "loss": float(np.mean(losses)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train live/dead classifier")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    data_root = get_data_root(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"DATA_ROOT: {data_root}")

    df = load_dataframe(data_root)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}")

    class_weights, weight_tensor = compute_class_weights(train_df)
    print("Class weights:", class_weights)

    mean, std = compute_normalization_stats(train_df, data_root)
    train_ds = CellDataset(train_df, data_root, get_transforms(mean, std, train=True))
    val_ds = CellDataset(val_df, data_root, get_transforms(mean, std, train=False))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    history = []
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler
        )
        val_metrics = run_epoch(model, val_loader, criterion, device)

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_f1": val_metrics["f1"],
            "val_balanced_acc": val_metrics["balanced_acc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            f"train_loss={record['train_loss']:.4f}  "
            f"val_loss={record['val_loss']:.4f}  "
            f"val_f1={record['val_f1']:.4f}  "
            f"val_balanced_acc={record['val_balanced_acc']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "label_to_idx": LABEL_TO_IDX,
                "class_weights": class_weights,
                "normalize_mean": mean,
                "normalize_std": std,
                "best_val_f1": best_f1,
            }
            torch.save(checkpoint, output_dir / "best.pt")
            print(f"Saved best checkpoint (val_f1={best_f1:.4f})")

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Best val F1: {best_f1:.4f}")
    print(f"Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import CellDataset, build_model, get_data_root, get_transforms, load_dataframe


def compute_metrics(y_true, y_pred):
    report = classification_report(
        y_true, y_pred, target_names=["live", "dead"], output_dict=True, zero_division=0
    )
    return {
        "accuracy": float((np.array(y_true) == np.array(y_pred)).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_live": report["live"]["precision"],
        "recall_live": report["live"]["recall"],
        "precision_dead": report["dead"]["precision"],
        "recall_dead": report["dead"]["recall"],
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate_split(model, loader, device):
    model.eval()
    y_true, y_pred, probs = [], [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
            x = x.to(device)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            y_true.extend(y.tolist())
            y_pred.extend(preds.cpu().tolist())
            probs.extend(prob.cpu().tolist())

    return y_true, y_pred, probs


def grouped_metrics(df, y_true, y_pred, group_col):
    results = {}
    for group_value, group_df in df.groupby(group_col):
        idxs = group_df.index.tolist()
        group_true = [y_true[i] for i in idxs]
        group_pred = [y_pred[i] for i in idxs]
        results[str(group_value)] = compute_metrics(group_true, group_pred)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate live/dead classifier")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", default=None, help="Path for eval_results.json")
    args = parser.parse_args()

    data_root = get_data_root(args.data_root)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output) if args.output else checkpoint_path.parent / "eval_results.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mean = checkpoint["normalize_mean"]
    std = checkpoint["normalize_std"]

    df = load_dataframe(data_root)
    eval_df = df[df["split"] == args.split].copy().reset_index(drop=True)
    print(f"Evaluating split={args.split}: {len(eval_df):,} samples")

    dataset = CellDataset(eval_df, data_root, get_transforms(mean, std, train=False))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred, probs = evaluate_split(model, loader, device)
    results = {
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "overall": compute_metrics(y_true, y_pred),
        "by_exp_id": grouped_metrics(eval_df, y_true, y_pred, "exp_id"),
        "by_assay_id": grouped_metrics(eval_df, y_true, y_pred, "assay_id"),
    }

    print("\nOverall metrics:")
    for key, value in results["overall"].items():
        if key != "confusion_matrix":
            print(f"  {key}: {value:.4f}")

    print("\nConfusion matrix [[live_live, live_dead], [dead_live, dead_dead]]:")
    print(np.array(results["overall"]["confusion_matrix"]))

    print("\nPer experiment:")
    for exp_id, metrics in results["by_exp_id"].items():
        print(f"  {exp_id}: f1={metrics['f1_macro']:.4f}, balanced_acc={metrics['balanced_accuracy']:.4f}")

    print("\nPer assay:")
    for assay_id, metrics in results["by_assay_id"].items():
        print(f"  {assay_id}: f1={metrics['f1_macro']:.4f}, balanced_acc={metrics['balanced_accuracy']:.4f}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()

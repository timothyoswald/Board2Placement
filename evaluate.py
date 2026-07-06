"""
Evaluation utilities for placement prediction and recommendation sanity checks.
"""
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from model import Board2Placement
from train import compute_metrics, load_dataset


def evaluate_model(
    data_path: str = "data/cleaned17.6_partial",
    model_path: str = "board2placement.pth",
    use_distribution: bool = True,
):
    fallback = "data/cleaned16.2"
    path = data_path if os.path.exists(data_path) else fallback
    if not os.path.exists(path):
        print(f"No dataset at {path}")
        return

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    boards, contexts, targets = load_dataset(path)
    dataset = TensorDataset(boards, contexts, targets)
    loader = DataLoader(dataset, batch_size=64)

    model = Board2Placement(predict_distribution=use_distribution).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        print(f"No model at {model_path}, evaluating untrained weights")

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for b, c, t in loader:
            b, c = b.to(device), c.to(device)
            preds = model(b, c)
            all_preds.append(preds.cpu())
            all_targets.append(t)

    metrics = compute_metrics(
        torch.cat(all_preds), torch.cat(all_targets), use_distribution
    )
    print("=== Placement Model Evaluation ===")
    print(f"Samples: {len(dataset)}")
    print(f"MAE: {metrics['mae']:.3f}")
    print(f"Top-4 Accuracy: {metrics['top4_acc']:.3f}")
    print(f"Exact Placement Accuracy: {metrics['exact_acc']:.3f}")
    return metrics


if __name__ == "__main__":
    evaluate_model()

"""
Evaluation utilities for placement prediction and recommendation sanity checks.
"""
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from model import Board2Placement
from settings import CLEANED_PARTIAL_DIR, PLACEMENT_MODEL_PATH
from train import compute_metrics, load_dataset


def evaluate_model(
    data_path: str = CLEANED_PARTIAL_DIR,
    model_path: str = PLACEMENT_MODEL_PATH,
    use_distribution: bool = True,
):
    path = data_path
    if not os.path.exists(path):
        print(f"No dataset at {path}")
        return

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    boards, contexts, traits, targets = load_dataset(path)
    dataset = TensorDataset(boards, contexts, traits, targets)
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
        for b, c, tr, t in loader:
            b, c, tr = b.to(device), c.to(device), tr.to(device)
            preds = model(b, c, tr)
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

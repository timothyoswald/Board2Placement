import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import Board2Placement
from settings import CLEANED_DIR, CLEANED_PARTIAL_DIR, PLACEMENT_MODEL_PATH
from state import CONTEXT_FEATURES

batchSize = 32
learningRate = 0.001
epochs = 25
trueData = CLEANED_PARTIAL_DIR
fallbackData = CLEANED_DIR
modelSavePath = PLACEMENT_MODEL_PATH
useDistributionHead = True


def load_dataset(path: str):
    data = torch.load(path, weights_only=False)
    # Support legacy formats: 2/3/4-tuple, or 5-tuple with trait counts
    first = data[0]
    if len(first) == 2:
        boards = torch.stack([x[0] for x in data])
        contexts = torch.zeros(len(data), CONTEXT_FEATURES)
        traits = torch.zeros(len(data), 1)
        targets = torch.stack([x[1] for x in data])
    elif len(first) == 3:
        boards = torch.stack([x[0] for x in data])
        contexts = torch.stack([x[1] for x in data])
        traits = torch.zeros(len(data), 1)
        targets = torch.stack([x[2] for x in data])
    elif len(first) == 5:
        boards = torch.stack([x[0] for x in data])
        contexts = torch.stack([x[1] for x in data])
        traits = torch.stack([x[2] for x in data])
        targets = torch.stack([x[4] for x in data])
    else:
        boards = torch.stack([x[0] for x in data])
        contexts = torch.stack([x[1] for x in data])
        traits = torch.stack([x[2] for x in data])
        targets = torch.stack([x[3] for x in data])
    return boards, contexts, traits, targets


def compute_metrics(preds, targets, is_distribution=False):
    if is_distribution:
        placements = torch.arange(1, 9, device=preds.device, dtype=preds.dtype)
        probs = torch.softmax(preds, dim=1)
        expected = (probs * placements).sum(dim=1)
        pred_class = probs.argmax(dim=1) + 1
    else:
        expected = preds.squeeze()
        pred_class = preds.squeeze().round().clamp(1, 8)

    targets_flat = targets.squeeze()
    mae = (expected - targets_flat).abs().mean().item()
    top4_acc = ((pred_class <= 4) == (targets_flat <= 4)).float().mean().item()
    exact = (pred_class.round() == targets_flat).float().mean().item()
    return {"mae": mae, "top4_acc": top4_acc, "exact_acc": exact}


def train():
    import os

    data_path = trueData if os.path.exists(trueData) else fallbackData
    bestValidationLoss = float("inf")
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    print(f"Loading data from {data_path}")

    boards, contexts, traits, targets = load_dataset(data_path)
    dataset = TensorDataset(boards, contexts, traits, targets)
    print(f"loaded {len(dataset)} samples")

    trainingSize = int(0.8 * len(dataset))
    validationSize = len(dataset) - trainingSize
    trainingData, validationData = random_split(dataset, [trainingSize, validationSize])
    trainingLoader = DataLoader(trainingData, batch_size=batchSize, shuffle=True)
    validationLoader = DataLoader(validationData, batch_size=batchSize)

    model = Board2Placement(predict_distribution=useDistributionHead).to(device)

    if useDistributionHead:
        criterion = nn.CrossEntropyLoss()
        target_fn = lambda t: (t.squeeze().long() - 1).clamp(0, 7)
    else:
        criterion = nn.MSELoss()
        target_fn = lambda t: t

    optimizer = optim.Adam(model.parameters(), lr=learningRate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print("training started!")
    for epoch in range(epochs):
        model.train()
        totalTrainingLoss = 0

        for batchBoard, batchContext, batchTraits, batchTargets in trainingLoader:
            batchBoard = batchBoard.to(device)
            batchContext = batchContext.to(device)
            batchTraits = batchTraits.to(device)
            batchTargets = batchTargets.to(device)

            optimizer.zero_grad()
            guesses = model(batchBoard, batchContext, batchTraits)
            loss = criterion(guesses, target_fn(batchTargets))
            loss.backward()
            optimizer.step()
            totalTrainingLoss += loss.item()

        scheduler.step()
        avgTrainingLoss = totalTrainingLoss / len(trainingLoader)

        model.eval()
        totalValidationLoss = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for valBoard, valContext, valTraits, valTarget in validationLoader:
                valBoard = valBoard.to(device)
                valContext = valContext.to(device)
                valTraits = valTraits.to(device)
                valTarget = valTarget.to(device)
                guesses2 = model(valBoard, valContext, valTraits)
                loss2 = criterion(guesses2, target_fn(valTarget))
                totalValidationLoss += loss2.item()
                all_preds.append(guesses2.cpu())
                all_targets.append(valTarget.cpu())

        avgValidationLoss = totalValidationLoss / len(validationLoader)
        metrics = compute_metrics(
            torch.cat(all_preds), torch.cat(all_targets), useDistributionHead
        )
        print(
            f"Epoch {epoch + 1} | Train Loss: {avgTrainingLoss:.4f} | "
            f"Val Loss: {avgValidationLoss:.4f} | MAE: {metrics['mae']:.3f} | "
            f"Top4 Acc: {metrics['top4_acc']:.3f}"
        )

        if avgValidationLoss < bestValidationLoss:
            bestValidationLoss = avgValidationLoss
            torch.save(model.state_dict(), modelSavePath)

    print("Training Complete + Model Saved")


if __name__ == "__main__":
    train()

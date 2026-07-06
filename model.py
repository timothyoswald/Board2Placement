import torch
from torch import nn
import json

from state import CONTEXT_FEATURES


class Board2Placement(nn.Module):
    """
    Predicts final placement from board state + economy context.
    Supports masked pooling over occupied unit slots only.
    """

    def __init__(self, predict_distribution: bool = False):
        super().__init__()
        with open("data/IDs", "r") as f:
            IDs = json.load(f)

        self.unitsCount = len(IDs["unitIDs"])
        self.itemsCount = len(IDs["itemIDs"])
        self.predict_distribution = predict_distribution

        self.unitEmbedding = nn.Embedding(self.unitsCount, 64)
        self.itemEmbedding = nn.Embedding(self.itemsCount, 32)

        self.processingLayers = nn.Sequential(
            nn.Linear(64 + 32 * 3 + 1 + 1, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.contextLayers = nn.Sequential(
            nn.Linear(CONTEXT_FEATURES, 16),
            nn.ReLU(),
        )

        head_out = 8 if predict_distribution else 1
        self.outputHead = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, head_out),
        )

    def forward(self, board, context):
        """
        board: [batch, 14, 6]
        context: [batch, 3] stage, level, gold
        """
        unitIDs = board[:, :, 0]
        unitCosts = board[:, :, 1].unsqueeze(-1).float()
        starLevels = board[:, :, 2].unsqueeze(-1).float()
        item1IDs = board[:, :, 3]
        item2IDs = board[:, :, 4]
        item3IDs = board[:, :, 5]

        unitEmbs = self.unitEmbedding(unitIDs)
        item1Embs = self.itemEmbedding(item1IDs)
        item2Embs = self.itemEmbedding(item2IDs)
        item3Embs = self.itemEmbedding(item3IDs)

        unitProperties = torch.cat(
            [unitEmbs, unitCosts, starLevels, item1Embs, item2Embs, item3Embs], dim=2
        )
        processedUnits = self.processingLayers(unitProperties)

        # Masked mean: only average occupied slots
        mask = (unitIDs != 0).unsqueeze(-1).float()
        masked_sum = (processedUnits * mask).sum(dim=1)
        unit_count = mask.sum(dim=1).clamp(min=1.0)
        boardVector = masked_sum / unit_count

        contextVector = self.contextLayers(context)
        combined = torch.cat([boardVector, contextVector], dim=1)
        output = self.outputHead(combined)

        if self.predict_distribution:
            return output  # logits for placements 1-8
        return output

    def expected_placement(self, board, context):
        """Return expected placement from distribution head."""
        if not self.predict_distribution:
            return self.forward(board, context)
        logits = self.forward(board, context)
        probs = torch.softmax(logits, dim=1)
        placements = torch.arange(1, 9, device=probs.device, dtype=probs.dtype)
        return (probs * placements).sum(dim=1, keepdim=True)

    def top4_probability(self, board, context):
        if not self.predict_distribution:
            pred = self.forward(board, context)
            return (pred <= 4.0).float()
        logits = self.forward(board, context)
        probs = torch.softmax(logits, dim=1)
        return probs[:, :4].sum(dim=1, keepdim=True)

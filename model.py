import torch
from torch import nn
import json

from settings import IDS_PATH
from state import CONTEXT_FEATURES


class Board2Placement(nn.Module):
    """
    Predicts final placement from board state + economy context + traits.
    Supports masked pooling over occupied unit slots only.
    """

    def __init__(self, predict_distribution: bool = False):
        super().__init__()
        with open(IDS_PATH, "r") as f:
            IDs = json.load(f)

        self.unitsCount = len(IDs["unitIDs"])
        self.itemsCount = len(IDs["itemIDs"])
        self.traitsCount = len(IDs.get("traitIDs", {"reserve": 0}))
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

        self.traitLayers = nn.Sequential(
            nn.Linear(max(self.traitsCount, 1), 16),
            nn.ReLU(),
        )

        head_out = 8 if predict_distribution else 1
        self.outputHead = nn.Sequential(
            nn.Linear(64 + 16 + 16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, head_out),
        )

    def forward(self, board, context, traits=None):
        """
        board: [batch, 14, 6]
        context: [batch, 3] stage, level, gold
        traits: [batch, num_traits] style/4 encodings (optional for back-compat)
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

        if traits is None:
            traits = torch.zeros(
                board.size(0), max(self.traitsCount, 1), device=board.device, dtype=torch.float32
            )
        elif traits.size(-1) != max(self.traitsCount, 1):
            # Pad or truncate to expected size
            fixed = torch.zeros(
                board.size(0), max(self.traitsCount, 1), device=board.device, dtype=torch.float32
            )
            n = min(traits.size(-1), fixed.size(-1))
            fixed[:, :n] = traits[:, :n]
            traits = fixed

        traitVector = self.traitLayers(traits.float())
        combined = torch.cat([boardVector, contextVector, traitVector], dim=1)
        output = self.outputHead(combined)

        if self.predict_distribution:
            return output  # logits for placements 1-8
        return output

    def expected_placement(self, board, context, traits=None):
        logits_or_value = self.forward(board, context, traits)
        if self.predict_distribution:
            probs = torch.softmax(logits_or_value, dim=1)
            placements = torch.arange(1, 9, device=board.device, dtype=probs.dtype)
            return (probs * placements).sum(dim=1)
        return logits_or_value.squeeze(-1)

    def top4_probability(self, board, context, traits=None):
        if not self.predict_distribution:
            raise ValueError("top4_probability requires predict_distribution=True")
        logits = self.forward(board, context, traits)
        probs = torch.softmax(logits, dim=1)
        return probs[:, :4].sum(dim=1)

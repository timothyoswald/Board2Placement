import torch
from torch import nn
import json

class Board2Placement(nn.Module):
    def __init__(self):
        super().__init__()
        with open("data/IDs", "r") as f:
            IDs = json.load(f)

        self.unitsCount = len(IDs["unitIDs"])
        self.itemsCount = len(IDs["itemIDs"])

        # embeddings
        self.unitEmbedding = nn.Embedding(self.unitsCount, 32)
        self.itemEmbedding = nn.Embedding(self.itemsCount, 32)

        # analyzes individual units
        self.processingLayers = nn.Sequential(
            nn.Linear(32 * 4 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # assesses entire board
        self.outputHead = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

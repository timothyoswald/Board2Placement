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
        self.unitEmbedding = nn.Embedding(self.unitsCount, 64)
        self.itemEmbedding = nn.Embedding(self.itemsCount, 32)

        # analyzes individual units
        self.processingLayers = nn.Sequential(
            nn.Linear(64 + 32 * 3 + 1, 128),
            nn.ReLU(),
            nn.Dropout(0.2), # randomly turn off 20% neurons to prevent overfitting
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # assesses entire board
        self.outputHead = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        unitIDs = x[:, :, 0]
        # not embedded like others so need to pad extra dimension
        starLevels = x[:, :, 1].unsqueeze(-1).float()
        item1IDs = x[:, :, 2]
        item2IDs = x[:, :, 3]
        item3IDs = x[:, :, 4]

        # get embeddings
        unitEmbs = self.unitEmbedding(unitIDs)
        item1Embs = self.itemEmbedding(item1IDs)
        item2Embs = self.itemEmbedding(item2IDs)
        item3Embs = self.itemEmbedding(item3IDs)

        # combine units with their items and star level
        unitProperties = torch.cat([unitEmbs, item1Embs, item2Embs, item3Embs, starLevels], dim=2)

        # process units
        processedUnits = self.processingLayers(unitProperties)

        # average board into single value
        boardVector = torch.mean(processedUnits, dim = 1)

        # predict value of board
        return self.outputHead(boardVector)

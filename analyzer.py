import torch
import json
import numpy as np
import pickle
import os
from sklearn.cluster import KMeans

dataDir = "data/cleanedData"
IDsDir = "data/IDs"
modelDir = "boardFinder.pkl"
clusterCount = 15

class BoardFinder:
    def __init__(self):
        with open(IDsDir, "r") as f:
            self.IDsFile = json.load(f)
            # invert dictionary to map IDs to name
            self.IDtoName = {ID: name for name, ID in self.IDsFile["unitIDs"].items()}
            self.IDtoItem = {ID: name for name, ID in self.IDsFile["itemIDs"].items()}
            self.numUnits = len(self.IDtoName)

        if os.path.exists(modelDir):
            self.loadState()
        else:
            self.train()
            self.saveState()
    
    def train(self):
        # now load matches and only keep comps that go top 4
        data = torch.load(dataDir)
        winningBoards = []
        winningBoardsTensors = []

        for boardTensor, placementTensor in data:
            placement = placementTensor.item()

            if placement <= 4.0:
                # extract only units
                unitIDs = boardTensor[:, 0].tolist()
                unitCosts = boardTensor[:, 1].tolist()
                # 1D vector of ~unit cost if unit in board and 0 o.w
                vector = np.zeros(self.numUnits)
                for i in range(len(unitIDs)):
                    ID = unitIDs[i]
                    if ID != 0:
                        vector[ID] = 1 + (unitCosts[i] - 1) * 0.25
                winningBoards.append(vector)
                winningBoardsTensors.append(boardTensor)
        self.vectors = np.array(winningBoards)
        print(f"found {len(self.vectors)} winning boards")

        # KMeans to find clusters
        self.KMeans = KMeans(n_clusters=clusterCount)
        self.KMeans.fit(self.vectors)
        print("found clusters")

        # find item frequencies for each composition
        print("finding items")
        self.clusterItemsForUnits = {i : dict() for i in range(clusterCount)}
        self.clusterLabels = self.KMeans.labels_

        for i, boardTensor in enumerate(winningBoardsTensors):
            clusterIdx = self.clusterLabels[i]

            # check all 14 possible slots to get items
            for j in range(14):
                unitID = int(boardTensor[j, 0].item())
                # skip empty slots
                if unitID != 0:
                    items = [int(boardTensor[j, 3].item()),
                             int(boardTensor[j, 4].item()),
                             int(boardTensor[j, 5].item())]
                    # record items if they have any
                    if sum(items) > 0:
                        unitName = self.IDtoName[unitID]

                        if unitName not in self.clusterItemsForUnits[clusterIdx]:
                            self.clusterItemsForUnits[clusterIdx][unitName] = dict()

                        for itemID in items:
                            # if holding item
                            if itemID != 0:
                                itemName = self.IDtoItem[itemID]

                                if itemName in self.clusterItemsForUnits[clusterIdx][unitName]:
                                    self.clusterItemsForUnits[clusterIdx][unitName][itemName] += 1
                                else:
                                    self.clusterItemsForUnits[clusterIdx][unitName][itemName] = 1
        print("found items")

    def printGoodBoards(self):
        centroids = self.KMeans.cluster_centers_
        print("-----the good boards-----")
        for i, centroid in enumerate(centroids):
            coreUnits = [] # appear in >50% of comps in this cluster
            for ID, score in enumerate(centroid):
                if score > 0.5: 
                    unitName = self.IDtoName[ID]
                    coreUnits.append((unitName, score))
            coreUnits.sort(key = lambda x : x[1], reverse = True)
            boardNames = [unit[0] for unit in coreUnits]
            print(f"Board {i + 1} : {', '.join(boardNames)}")        
    
    # currentBoard is a list of strings of unit names
    def completeBoard(self, currentBoard):
        unitIDs = self.IDsFile["unitIDs"]
        boardVector = np.zeros(self.numUnits)
        boardIDs = []

        for name in currentBoard:
            if name in unitIDs:
                ID = unitIDs[name]
                boardVector[ID] = 1
                boardIDs.append(ID)
        
        bestScore = -1
        bestClusterIdx = -1

        for i, centroid in enumerate(self.KMeans.cluster_centers_):
            # dot product to score, highly orthogonal comps close to 0
            score = np.dot(boardVector, centroid)

            if score > bestScore:
                bestScore = score
                bestClusterIdx = i
        
        bestCentroid = self.KMeans.cluster_centers_[bestClusterIdx]

        tmp = []
        for ID, score in enumerate(bestCentroid):
            if ID not in boardIDs and score > 0.5:
                unitName = self.IDtoName[ID]
                tmp.append((unitName, score))
        tmp.sort(key = lambda x : x[1], reverse = True)

        return bestClusterIdx, tmp

    def loadState(self):
        with open(modelDir, "rb") as f:
            state = pickle.load(f)
        self.KMeans = state["model"]
        self.clusterItemsForUnits = state["compItems"]
        print("model loaded")

    def saveState(self):
        state = {"model": self.KMeans, "compItems" : self.clusterItemsForUnits}
        with open(modelDir, "wb") as f:
            pickle.dump(state, f)
        print(f"model saved to {modelDir}")

analyzer = BoardFinder()
analyzer.printGoodBoards()

testBoard = ["TFT16_Rumble", "TFT16_JarvanIV", "TFT16_Lulu"]
print(f"completing {testBoard}...")
i, suggestedUnits = analyzer.completeBoard(testBoard)
print(f"this is closest to Board #{i}")
print(f"recomended units:")
for name, score in suggestedUnits:
    print(f"{name} with score {score}")
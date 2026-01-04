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
            self.numUnits = len(self.IDtoName)

        if os.path.exists(modelDir):
            self.loadModel()
        else:
            self.train()
            self.saveModel()
    
    def train(self):
        # now load matches and only keep comps that go top 4
        data = torch.load(dataDir)
        winningBoards = []

        for boardTensor, placementTensor in data:
            placement = placementTensor.item()

            if placement <= 4.0:
                # extract only units
                unitIDs = boardTensor[:, 0].tolist()
                unitCosts = boardTensor[:, 1].tolist()
                # 1D vector of 1 if unit in board and 0 o.w
                vector = np.zeros(self.numUnits)
                for i in range(len(unitIDs)):
                    ID = unitIDs[i]
                    if ID != 0:
                        vector[ID] = 1 + (unitCosts[i] - 1) * 0.25
                winningBoards.append(vector)
        self.vectors = np.array(winningBoards)
        print(f"found {len(self.vectors)} winning boards")

        # KMeans to find clusters
        self.KMeans = KMeans(n_clusters=clusterCount)
        self.KMeans.fit(self.vectors)
        print("found clusters")

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

    def loadModel(self):
        with open(modelDir, "rb") as f:
            self.KMeans = pickle.load(f)
        print("model loaded")

    def saveModel(self):
        with open(modelDir, "wb") as f:
            pickle.dump(self.KMeans, f)
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
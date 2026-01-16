import torch
import json
import numpy as np
import pickle
import os
import copy
import componentData
from sklearn.cluster import KMeans

dataDir = "data/cleaned16.2"
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
        print("found items for clusters")

        # figure out what items are good on what units
        # for this type of board
        self.itemWeightsPerCluster = dict()
        for clusterIdx, unitsDict in self.clusterItemsForUnits.items():
            self.itemWeightsPerCluster[clusterIdx] = dict()

            for unitName, itemCounts in unitsDict.items():
                unitCount = sum(itemCounts.values()) # how many times does this unit show up
                if unitCount < 10: continue # ignore uncommon units

                goodItems = dict()

                for item, count in itemCounts.items():
                    slamRate = count / unitCount
                    if slamRate > 0.05: # if this item is made >5% of time add it
                        goodItems[item] = slamRate
                
                self.itemWeightsPerCluster[clusterIdx][unitName] = goodItems

        print("found items for units in each cluster")
    
    def getBestItems(self, clusterIdx, currItems):
        clusterWeights = self.itemWeightsPerCluster[clusterIdx]

        possibleItems = [] # collect what are make-able items

        centroid = self.KMeans.cluster_centers_[clusterIdx]

        coreUnits = []
        for ID, val in enumerate(centroid):
            if val > 0.5: # only consider important units
                coreUnits.append((self.IDtoName[ID], val))
            
        for unitName, unitImportance in coreUnits:
            if unitName not in clusterWeights: continue
            
            for itemName, itemWeight in clusterWeights[unitName].items():
                if itemName not in componentData.itemToComponent: continue
                
                # score how important it is for 
                # this unit to hold this item
                score = unitImportance * itemWeight * 10
                
                possibleItems.append({"id": itemName,
                                      "recipe": componentData.itemToComponent[itemName],
                                      "score": score,
                                      "readText": f"Make {itemName} on {unitName}"
                                     })
        
        # sort by best score
        possibleItems.sort(key=lambda x: x['score'], reverse=True)
        
        # 
        bestScore, itemsToMake = self.findOptimalItems(currItems, possibleItems)
        
        return bestScore, itemsToMake
    
    def findOptimalItems(self, currItems, possibleItems, memo = None):
        # make this into a tuple so it is hashable
        currItemsKey = tuple(sorted(currItems.items()))

        if memo == None:
            memo = dict()
        if currItemsKey in memo:
            return memo[currItemsKey]
        
        bestScore = 0
        itemsToMake = []

        for item in possibleItems:
            comp1, comp2 = item["recipe"][0], item["recipe"][1]

            # check if we can make this item
            # given our components
            canMake = False
            if comp1 == comp2:
                if currItems.get(comp1, 0) >= 2:
                    canMake = True
            else:
                if currItems.get(comp1, 0) >= 1 and currItems.get(comp2, 0) >= 1:
                    canMake = True
            
            if canMake:
                leftoverItems = copy.deepcopy(currItems) # so we can backtrack w/o aliasing
                leftoverItems[comp1] -= 1
                leftoverItems[comp2] -= 1

                # recurse to find other items
                trialScore, trialItems = self.findOptimalItems(leftoverItems, possibleItems, memo)

                currScore = trialScore + item["score"]
                if currScore > bestScore:
                    bestScore = currScore
                    itemsToMake = [item] + trialItems
        
        # store state in memo
        memo[currItemsKey] = (bestScore, itemsToMake)
        return (bestScore, itemsToMake)


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
    def completeBoard(self, currentBoard, currItems):
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
        bestItems = None

        for i, centroid in enumerate(self.KMeans.cluster_centers_):
            # dot product to score units, highly orthogonal comps close to 0
            unitsScore = np.dot(boardVector, centroid)
            itemsScore, itemSlams = self.getBestItems(i, currItems)

            totalScore = unitsScore + itemsScore

            if totalScore > bestScore:
                bestScore = totalScore
                bestClusterIdx = i
                bestItems = itemSlams
        
        bestCentroid = self.KMeans.cluster_centers_[bestClusterIdx]

        suggestedUnits = []
        for ID, score in enumerate(bestCentroid):
            if ID not in boardIDs and score > 0.5:
                unitName = self.IDtoName[ID]
                suggestedUnits.append(unitName)
        suggestedUnits.sort(key = lambda x : x[1], reverse = True)

        return bestClusterIdx, suggestedUnits, bestItems

    def loadState(self):
        with open(modelDir, "rb") as f:
            state = pickle.load(f)
        self.KMeans = state["model"]
        self.clusterItemsForUnits = state["compItems"]
        self.itemWeightsPerCluster = state["itemWeights"]
        print("model loaded")

    def saveState(self):
        state = {"model": self.KMeans, "compItems" : self.clusterItemsForUnits,
                 "itemWeights": self.itemWeightsPerCluster}
        with open(modelDir, "wb") as f:
            pickle.dump(state, f)
        print(f"model saved to {modelDir}")

analyzer = BoardFinder()
analyzer.printGoodBoards()

testBoard = ["TFT16_Jhin", "TFT16_Shen", "TFT16_XinZhao"]
testItems = {"TFT_Item_BFSword": 1, "TFT_Item_RecurveBow": 2, "TFT_Item_NeedlesslyLargeRod": 1,
             "TFT_Item_NegatronCloak": 1, "TFT_Item_SparringGloves": 1}
print(f"completing {testBoard}...")
i, suggestedUnits, bestItems = analyzer.completeBoard(testBoard, testItems)
print(f"this is closest to Board #{i}")
print(f"recomended units: {suggestedUnits}")
print(f"recommended items:")
for d in bestItems:
    print(d["readText"])
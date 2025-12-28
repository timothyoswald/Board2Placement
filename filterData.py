import json
import os
import torch

dataDir = "data/rawMatches"
IDsDir = "data/IDs"
outputDir = "data/trueData"

def filterData():
    print("started")

    matches = [f for f in os.listdir(dataDir) if f.endswith(".json")]

    # vocab
    unitIDs = {"reserve": 0}
    itemIDs = {"reserve": 0}

    doneData = []

    print("processing boards")

    for i in range(len(matches)):
        fileName = matches[i]
        with open(os.path.join(dataDir, fileName), "r") as f:
            match = json.load(f)
        players = match["info"]["participants"]

        # for every player we save their board as
        # units, star levels, items, placements
        for player in players:
            placement = float(player["placement"])
            board = []

            for unit in player["units"]:
                unitName = unit["character_id"]
                if unitName not in unitIDs:
                    unitIDs[unitName] = len(unitIDs)

                starLevel = unit["tier"]
                unitItems = [0, 0, 0]
                itemNames = unit["itemNames"]
                for i in range(len(itemNames)):
                    item = itemNames[i]
                    if item not in itemIDs:
                        itemIDs[item] = len(itemIDs)
                    unitItems[i] = itemIDs[item]
                
                unitVector = [unitIDs[unitName], starLevel] + unitItems
                board.append(unitVector)
        
            # even with augments/items no board should
            # have more than 14 units
            while len(board) < 14:
                board.append([0, 0, 0, 0, 0])
            
            # ints for input tensor for indexing
            inputTensor = torch.tensor(board, dtype = torch.long)
            # floats for target
            targetTensor = torch.tensor([placement], dtype = torch.float32)
            
            doneData.append((inputTensor, targetTensor))

    IDs = {"unitIDs": unitIDs, "itemIDs": itemIDs}
    with open(IDsDir, "w") as f:
        json.dump(IDs, f)
    print("vocab saved")

    torch.save(doneData, outputDir)
    print("tensors saved")
        
    print("boards processed")

filterData()
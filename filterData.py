import json
import os
import torch

dataDir = "data/patch16.2" # where your scraped match data is
IDsDir = "data/IDs" # help to sort the cleaned data
outputDir = "data/cleaned16.2" # where you want the cleaned data to go

def filterData():
    print("started")

    matches = [f for f in os.listdir(dataDir) if f.endswith(".json")]

    # vocab
    unitIDs = {"reserve": 0}
    itemIDs = {"reserve": 0}

    doneData = []

    print("processing boards")

    for i in range(len(matches)):
        if i % 100 == 0:
            print(f"processed {i} matches")
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

                unitCost = unit["rarity"] + 1
                starLevel = unit["tier"]
                unitItems = [0, 0, 0]
                itemNames = unit["itemNames"]
                
                # for some reason the match data sometimes will double save items
                # so this mod is to fix it
                for j in range(len(itemNames)):
                    item = itemNames[j]
                    if item not in itemIDs:
                        itemIDs[item] = len(itemIDs)
                    unitItems[j % 3] = itemIDs[item]
                
                unitVector = [unitIDs[unitName], unitCost, starLevel] + unitItems
                board.append(unitVector)
        
            # even with augments/items no board should
            # have more than 14 units
            while len(board) < 14:
                board.append([0, 0, 0, 0, 0, 0])
            
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
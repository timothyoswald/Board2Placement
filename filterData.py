import json
import os
import torch

from partial_data import generate_partial_samples, merge_datasets
from state import parse_player_board, parse_player_context

dataDir = "data/patch17.6"
IDsDir = "data/IDs"
outputDir = "data/cleaned17.6"
partialOutputDir = "data/cleaned17.6_partial"
generatePartial = True
partialSamplesPerBoard = 3


def filterData():
    print("started")

    if not os.path.isdir(dataDir):
        print(f"data directory not found: {dataDir}")
        return

    matches = [f for f in os.listdir(dataDir) if f.endswith(".json")]

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

        for player in players:
            placement = float(player["placement"])
            inputTensor = parse_player_board(player, unitIDs, itemIDs)
            contextTensor = parse_player_context(player)
            targetTensor = torch.tensor([placement], dtype=torch.float32)
            doneData.append((inputTensor, contextTensor, targetTensor))

    IDs = {"unitIDs": unitIDs, "itemIDs": itemIDs}
    with open(IDsDir, "w") as f:
        json.dump(IDs, f)
    print("vocab saved")

    torch.save(doneData, outputDir)
    print(f"saved {len(doneData)} final-board samples to {outputDir}")

    if generatePartial:
        id_to_item = {v: k for k, v in itemIDs.items()}
        partial_data = generate_partial_samples(
            doneData,
            id_to_item,
            itemIDs,
            samples_per_board=partialSamplesPerBoard,
        )
        merged = merge_datasets(doneData, partial_data)
        torch.save(merged, partialOutputDir)
        print(f"saved {len(merged)} samples (incl. partial) to {partialOutputDir}")

    print("boards processed")


if __name__ == "__main__":
    filterData()

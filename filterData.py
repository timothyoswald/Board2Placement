import json
import os
import torch

from build_champion_map import champion_id_set, save_champion_trait_map
from partial_data import generate_partial_samples, merge_datasets
from settings import (
    CHAMPION_MAP_PATH,
    CLEANED_DIR,
    CLEANED_PARTIAL_DIR,
    DATA_DIR,
    IDS_PATH,
)
from state import (
    parse_player_board,
    parse_player_context,
    parse_player_trait_counts,
    parse_player_traits,
)

dataDir = DATA_DIR
IDsDir = IDS_PATH
outputDir = CLEANED_DIR
partialOutputDir = CLEANED_PARTIAL_DIR
generatePartial = True
partialSamplesPerBoard = 3


def _pad_vec(vec: torch.Tensor, size: int) -> torch.Tensor:
    if vec.numel() == size:
        return vec
    padded = torch.zeros(size, dtype=torch.float32)
    n = min(vec.numel(), size)
    padded[:n] = vec[:n]
    return padded


def filterData():
    print("started")

    if not os.path.isdir(dataDir):
        print(f"data directory not found: {dataDir}")
        return

    if not os.path.exists(CHAMPION_MAP_PATH):
        save_champion_trait_map()
    champ_ids = champion_id_set()
    if champ_ids:
        print(f"filtering to {len(champ_ids)} champions from Community Dragon")
    else:
        print("warning: empty champion map; keeping all units")

    matches = [f for f in os.listdir(dataDir) if f.endswith(".json")]

    unitIDs = {"reserve": 0}
    itemIDs = {"reserve": 0}
    traitIDs = {"reserve": 0}
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
            inputTensor = parse_player_board(
                player, unitIDs, itemIDs, champion_ids=champ_ids or None
            )
            contextTensor = parse_player_context(player)
            traitTensor = parse_player_traits(player, traitIDs)
            countTensor = parse_player_trait_counts(player, traitIDs)
            targetTensor = torch.tensor([placement], dtype=torch.float32)
            doneData.append(
                (inputTensor, contextTensor, traitTensor, countTensor, targetTensor)
            )

    num_traits = len(traitIDs)
    resized = []
    for board, context, traits, counts, placement in doneData:
        traits = _pad_vec(traits, num_traits)
        counts = _pad_vec(counts, num_traits)
        resized.append((board, context, traits, counts, placement))
    doneData = resized

    IDs = {"unitIDs": unitIDs, "itemIDs": itemIDs, "traitIDs": traitIDs}
    os.makedirs(os.path.dirname(IDsDir) or ".", exist_ok=True)
    with open(IDsDir, "w") as f:
        json.dump(IDs, f)
    print(f"vocab saved ({len(unitIDs)} units, {len(itemIDs)} items, {len(traitIDs)} traits)")

    torch.save(doneData, outputDir)
    print(f"saved {len(doneData)} final-board samples to {outputDir}")

    if generatePartial:
        id_to_item = {v: k for k, v in itemIDs.items()}
        partial_data = generate_partial_samples(
            doneData,
            id_to_item,
            itemIDs,
            samples_per_board=partialSamplesPerBoard,
            trait_ids=traitIDs,
            unit_ids=unitIDs,
        )
        merged = merge_datasets(doneData, partial_data)
        torch.save(merged, partialOutputDir)
        print(f"saved {len(merged)} samples (incl. partial) to {partialOutputDir}")

    print("boards processed")


if __name__ == "__main__":
    filterData()

"""
Generate synthetic partial-game states from final boards for training.
Riot match history lacks mid-game snapshots, so we simulate them.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import torch

import componentData
from settings import CHAMPION_MAP_PATH
from state import (
    BOARD_SLOTS,
    UNIT_FEATURES,
    load_champion_trait_map,
    recompute_traits_from_board,
)


def _item_id_to_name(item_id: int, id_to_item: dict) -> str:
    return id_to_item.get(item_id, "")


def _name_to_item_id(name: str, item_ids: dict) -> int:
    return item_ids.get(name, 0)


def mask_board_partial(
    board: torch.Tensor,
    stage: float,
    id_to_item: dict,
    item_ids: dict,
    rng: random.Random,
) -> Tuple[torch.Tensor, dict]:
    """
    Create a partial board by removing high-cost units, down-starring, and
    converting some completed items back to components.
    Returns modified board and component dict.
    """
    partial = board.clone()
    components: dict = {}

    # Stage-based cost threshold: early game hides expensive units
    cost_by_stage = {1: 5, 2: 4, 3: 4, 4: 3, 5: 3, 6: 2, 7: 1}
    stage_int = max(1, min(7, int(stage)))
    max_cost_visible = cost_by_stage.get(stage_int, 3)

    for i in range(BOARD_SLOTS):
        unit_id = int(partial[i, 0].item())
        if unit_id == 0:
            continue

        cost = int(partial[i, 1].item())
        star = int(partial[i, 2].item())

        # Remove units above stage-appropriate cost with some randomness
        if cost > max_cost_visible and rng.random() < 0.7:
            partial[i] = torch.zeros(UNIT_FEATURES, dtype=torch.long)
            continue

        # Down-star higher star units in early stages
        if stage_int <= 4 and star > 1 and rng.random() < 0.5:
            partial[i, 2] = max(1, star - 1)

        # Convert completed items back to components
        for j in range(3, 6):
            item_id = int(partial[i, j].item())
            if item_id == 0:
                continue
            if rng.random() < 0.4:
                item_name = _item_id_to_name(item_id, id_to_item)
                recipe = componentData.itemToComponent.get(item_name)
                if recipe:
                    for comp in recipe:
                        components[comp] = components.get(comp, 0) + 1
                partial[i, j] = 0

    return partial, components


def sample_economy(stage: float, rng: random.Random) -> Tuple[float, float]:
    """Sample plausible level and gold for a given stage."""
    stage_int = max(1, min(7, int(stage)))
    level_ranges = {
        1: (2, 4), 2: (3, 5), 3: (5, 7), 4: (6, 9),
        5: (7, 10), 6: (8, 10), 7: (8, 10),
    }
    lo, hi = level_ranges.get(stage_int, (6, 8))
    level = float(rng.randint(lo, hi))
    gold = float(rng.choice([0, 10, 20, 30, 40, 50]))
    return level, gold


def generate_partial_samples(
    dataset: List[Tuple],
    id_to_item: dict,
    item_ids: dict,
    samples_per_board: int = 3,
    seed: int = 42,
    trait_ids: Optional[dict] = None,
    unit_ids: Optional[dict] = None,
) -> List[Tuple]:
    """
    From final-board dataset entries (board, context, traits[, counts], placement),
    generate synthetic partial states with recomputed traits.
    """
    rng = random.Random(seed)
    partial_data = []
    champ_map = load_champion_trait_map(CHAMPION_MAP_PATH)
    id_to_name = {v: k for k, v in (unit_ids or {}).items()} if unit_ids else {}
    trait_ids = trait_ids or {"reserve": 0}
    num_traits = len(trait_ids)

    for entry in dataset:
        if len(entry) == 5:
            board, _final_context, _final_traits, _final_counts, placement = entry
        elif len(entry) == 4:
            board, _final_context, _final_traits, placement = entry
        elif len(entry) == 3:
            board, _final_context, placement = entry
        else:
            board, placement = entry

        for _ in range(samples_per_board):
            stage = float(rng.randint(2, 5))
            level, gold = sample_economy(stage, rng)
            partial_board, _components = mask_board_partial(
                board, stage, id_to_item, item_ids, rng
            )
            context = torch.tensor([stage, level, gold], dtype=torch.float32)
            if id_to_name and champ_map:
                traits = recompute_traits_from_board(
                    partial_board, id_to_name, trait_ids, champ_map
                )
            else:
                traits = torch.zeros(num_traits, dtype=torch.float32)
            # Counts unused by placement net; zeros keep tuple shape consistent
            counts = torch.zeros(num_traits, dtype=torch.float32)
            partial_data.append((partial_board, context, traits, counts, placement))

    return partial_data


def merge_datasets(final_data: List[Tuple], partial_data: List[Tuple]) -> List[Tuple]:
    return final_data + partial_data

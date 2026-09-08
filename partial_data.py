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
    build_context_tensor,
    load_champion_trait_map,
    max_cost_visible,
    midgame_stage_rounds,
    recompute_traits_from_board,
)


def _item_id_to_name(item_id: int, id_to_item: dict) -> str:
    return id_to_item.get(item_id, "")


def _name_to_item_id(name: str, item_ids: dict) -> int:
    return item_ids.get(name, 0)


def mask_board_partial(
    board: torch.Tensor,
    stage: int,
    round_num: int,
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
    cap = max_cost_visible(stage, round_num)

    for i in range(BOARD_SLOTS):
        unit_id = int(partial[i, 0].item())
        if unit_id == 0:
            continue

        cost = int(partial[i, 1].item())
        star = int(partial[i, 2].item())

        # Remove units above the stage-round cost cap with some randomness
        if cost > cap and rng.random() < 0.7:
            partial[i] = torch.zeros(UNIT_FEATURES, dtype=torch.long)
            continue
        # 5-costs stay rare until stage 4
        if cost >= 5 and cap < 5 and rng.random() < 0.85:
            partial[i] = torch.zeros(UNIT_FEATURES, dtype=torch.long)
            continue

        # Down-star higher star units in early/mid game
        if stage <= 4 and star > 1 and rng.random() < 0.5:
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


def sample_economy(stage: int, round_num: int, rng: random.Random) -> Tuple[float, float]:
    """Sample plausible level and gold for a given stage-round."""
    level_ranges = {
        1: (2, 4),
        2: (3, 5),
        3: (5, 7),
        4: (6, 9),
        5: (7, 10),
        6: (8, 10),
        7: (8, 10),
    }
    lo, hi = level_ranges.get(stage, (6, 8))
    if round_num >= 5:
        lo = min(lo + 1, hi)
    if round_num <= 2:
        hi = max(lo, hi - 1)
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
    generate synthetic partial states with recomputed traits and 5-D context.
    """
    rng = random.Random(seed)
    partial_data = []
    champ_map = load_champion_trait_map(CHAMPION_MAP_PATH)
    id_to_name = {v: k for k, v in (unit_ids or {}).items()} if unit_ids else {}
    trait_ids = trait_ids or {"reserve": 0}
    num_traits = len(trait_ids)
    clocks = midgame_stage_rounds()

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
            stage, round_num = rng.choice(clocks)
            level, gold = sample_economy(stage, round_num, rng)
            partial_board, _components = mask_board_partial(
                board, stage, round_num, id_to_item, item_ids, rng
            )
            context = build_context_tensor(stage, round_num, level, gold)
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

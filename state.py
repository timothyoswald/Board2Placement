"""
Shared game-state representation and feature building for Board2Placement.
Used by analyzer, model training, and recommendation engine.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

BOARD_SLOTS = 14
UNIT_FEATURES = 6  # unitID, cost, star, item1, item2, item3
CONTEXT_FEATURES = 3  # stage, level, gold


@dataclass
class GameState:
    """Unified representation of a TFT board + economy context."""

    board: torch.Tensor  # shape [14, 6], long
    stage: float = 4.0
    level: float = 8.0
    gold: float = 30.0
    components: Dict[str, int] = field(default_factory=dict)

    def context_tensor(self) -> torch.Tensor:
        return torch.tensor([self.stage, self.level, self.gold], dtype=torch.float32)

    def occupied_slots(self) -> List[Tuple[int, int, int, int, int, int]]:
        """Return non-empty unit slots as tuples."""
        rows = []
        for i in range(BOARD_SLOTS):
            unit_id = int(self.board[i, 0].item())
            if unit_id != 0:
                rows.append(tuple(int(self.board[i, j].item()) for j in range(UNIT_FEATURES)))
        return rows

    def unit_names(self, id_to_name: Dict[int, str]) -> List[str]:
        return [
            id_to_name[int(self.board[i, 0].item())]
            for i in range(BOARD_SLOTS)
            if int(self.board[i, 0].item()) != 0
        ]

    def board_cost(self) -> float:
        total = 0.0
        for i in range(BOARD_SLOTS):
            unit_id = int(self.board[i, 0].item())
            if unit_id != 0:
                cost = int(self.board[i, 1].item())
                star = int(self.board[i, 2].item())
                copies = {1: 1, 2: 3, 3: 9}.get(star, 1)
                total += cost * copies
        return total

    def count_units_by_cost(self, min_cost: int = 1) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for i in range(BOARD_SLOTS):
            unit_id = int(self.board[i, 0].item())
            if unit_id != 0:
                cost = int(self.board[i, 1].item())
                if cost >= min_cost:
                    counts[cost] = counts.get(cost, 0) + 1
        return counts

    def completed_item_count(self) -> int:
        total = 0
        for i in range(BOARD_SLOTS):
            if int(self.board[i, 0].item()) != 0:
                for j in range(3, 6):
                    if int(self.board[i, j].item()) != 0:
                        total += 1
        return total

    def item_assignments(self, id_to_item: Dict[int, str], id_to_name: Dict[int, str]) -> List[Tuple[str, str]]:
        """List of (unit_name, item_name) for slotted items."""
        pairs = []
        for i in range(BOARD_SLOTS):
            unit_id = int(self.board[i, 0].item())
            if unit_id == 0:
                continue
            unit_name = id_to_name.get(unit_id, "")
            for j in range(3, 6):
                item_id = int(self.board[i, j].item())
                if item_id != 0:
                    pairs.append((unit_name, id_to_item.get(item_id, "")))
        return pairs

    def unit_vector(self, num_units: int, include_cost: bool = True) -> np.ndarray:
        """Build clustering / similarity vector over units."""
        vector = np.zeros(num_units)
        for i in range(BOARD_SLOTS):
            unit_id = int(self.board[i, 0].item())
            if unit_id != 0 and unit_id < num_units:
                if include_cost:
                    cost = int(self.board[i, 1].item())
                    vector[unit_id] = max(vector[unit_id], 1 + (cost - 1) * 0.25)
                else:
                    vector[unit_id] = 1.0
        return vector

    def mask_tensor(self) -> torch.Tensor:
        """1 for occupied slots, 0 for empty."""
        return (self.board[:, 0] != 0).float()


def load_id_maps(ids_path: str = "data/IDs") -> Tuple[dict, Dict[int, str], Dict[int, str], int, int]:
    with open(ids_path, "r") as f:
        ids_file = json.load(f)
    id_to_name = {v: k for k, v in ids_file["unitIDs"].items()}
    id_to_item = {v: k for k, v in ids_file["itemIDs"].items()}
    return ids_file, id_to_name, id_to_item, len(ids_file["unitIDs"]), len(ids_file["itemIDs"])


def empty_board_tensor() -> torch.Tensor:
    return torch.zeros((BOARD_SLOTS, UNIT_FEATURES), dtype=torch.long)


def board_from_unit_names(
    unit_names: List[str],
    ids_file: dict,
    star_levels: Optional[List[int]] = None,
    items_by_unit: Optional[Dict[str, List[str]]] = None,
) -> torch.Tensor:
    """Build board tensor from unit name strings."""
    board = empty_board_tensor()
    unit_ids = ids_file["unitIDs"]
    item_ids = ids_file["itemIDs"]
    items_by_unit = items_by_unit or {}

    for idx, name in enumerate(unit_names[:BOARD_SLOTS]):
        if name not in unit_ids:
            continue
        unit_id = unit_ids[name]
        board[idx, 0] = unit_id
        # default cost from name rarity is unknown; caller can override via full tensor
        board[idx, 1] = 1
        board[idx, 2] = star_levels[idx] if star_levels and idx < len(star_levels) else 1

        unit_items = items_by_unit.get(name, [])
        for j, item_name in enumerate(unit_items[:3]):
            if item_name in item_ids:
                board[idx, 3 + j] = item_ids[item_name]
    return board


def game_state_from_names(
    unit_names: List[str],
    components: Dict[str, int],
    ids_file: dict,
    stage: float = 3.0,
    level: float = 6.0,
    gold: float = 20.0,
    items_by_unit: Optional[Dict[str, List[str]]] = None,
) -> GameState:
    board = board_from_unit_names(unit_names, ids_file, items_by_unit=items_by_unit)
    return GameState(board=board, stage=stage, level=level, gold=gold, components=components)


def parse_player_board(player: dict, unit_ids: dict, item_ids: dict) -> torch.Tensor:
    """Parse Riot API participant units into board tensor."""
    board_rows = []
    for unit in player["units"]:
        unit_name = unit["character_id"]
        if unit_name not in unit_ids:
            unit_ids[unit_name] = len(unit_ids)

        unit_cost = unit["rarity"] + 1
        star_level = unit["tier"]
        unit_items = [0, 0, 0]
        item_names = unit["itemNames"]

        for j in range(len(item_names)):
            item = item_names[j]
            if item not in item_ids:
                item_ids[item] = len(item_ids)
            unit_items[j % 3] = item_ids[item]

        board_rows.append([unit_ids[unit_name], unit_cost, star_level] + unit_items)

    while len(board_rows) < BOARD_SLOTS:
        board_rows.append([0, 0, 0, 0, 0, 0])

    return torch.tensor(board_rows[:BOARD_SLOTS], dtype=torch.long)


def parse_player_context(player: dict) -> torch.Tensor:
    """Extract stage/level/gold from participant JSON."""
    level = float(player.get("level", 8))
    gold = float(player.get("gold_left", 0))
    last_round = player.get("last_round", 30)
    # Approximate stage from round number (4 rounds per stage in standard TFT)
    stage = max(1.0, min(7.0, last_round / 4.0))
    return torch.tensor([stage, level, gold], dtype=torch.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

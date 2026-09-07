"""
Shared game-state representation and feature building for Board2Placement.
Used by analyzer, model training, and recommendation engine.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from settings import CHAMPION_MAP_PATH, IDS_PATH

BOARD_SLOTS = 14
UNIT_FEATURES = 6  # unitID, cost, star, item1, item2, item3
CONTEXT_FEATURES = 3  # stage, level, gold


@dataclass
class GameState:
    """Unified representation of a TFT board + economy context + traits."""

    board: torch.Tensor  # shape [14, 6], long
    stage: float = 4.0
    level: float = 8.0
    gold: float = 30.0
    components: Dict[str, int] = field(default_factory=dict)
    traits: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.float32))

    def context_tensor(self) -> torch.Tensor:
        return torch.tensor([self.stage, self.level, self.gold], dtype=torch.float32)

    def trait_tensor(self) -> torch.Tensor:
        return self.traits

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


def load_id_maps(ids_path: str = IDS_PATH) -> Tuple[dict, Dict[int, str], Dict[int, str], int, int, int]:
    with open(ids_path, "r") as f:
        ids_file = json.load(f)
    id_to_name = {v: k for k, v in ids_file["unitIDs"].items()}
    id_to_item = {v: k for k, v in ids_file["itemIDs"].items()}
    trait_ids = ids_file.get("traitIDs", {"reserve": 0})
    return (
        ids_file,
        id_to_name,
        id_to_item,
        len(ids_file["unitIDs"]),
        len(ids_file["itemIDs"]),
        len(trait_ids),
    )


def empty_board_tensor() -> torch.Tensor:
    return torch.zeros((BOARD_SLOTS, UNIT_FEATURES), dtype=torch.long)


def empty_trait_tensor(num_traits: int) -> torch.Tensor:
    return torch.zeros(num_traits, dtype=torch.float32)


def normalize_trait_name(name: str) -> str:
    """Normalize Riot / Community Dragon trait API names for matching."""
    if not name:
        return ""
    n = name.strip()
    for prefix in (
        "TFT18_",
        "TFT17_",
        "TFT16_",
        "Set18_",
        "Set17_",
        "DA_18_",
        "DA_",
        "TFT_",
    ):
        if n.startswith(prefix):
            n = n[len(prefix) :]
            break
    if n.startswith("Trait_"):
        n = n[len("Trait_") :]
    return n.lower()


def parse_player_traits(player: dict, trait_ids: dict) -> torch.Tensor:
    """
    Encode Riot participant traits as style/4.0 vector.
    style: 0=none, 1=bronze, 2=silver, 3=gold, 4=chromatic.
    """
    for trait in player.get("traits", []):
        name = trait.get("name", "")
        if not name:
            continue
        if name not in trait_ids:
            trait_ids[name] = len(trait_ids)

    vec = empty_trait_tensor(len(trait_ids))
    for trait in player.get("traits", []):
        name = trait.get("name", "")
        if name not in trait_ids:
            continue
        style = float(trait.get("style", 0) or 0)
        tid = trait_ids[name]
        if tid < len(vec):
            vec[tid] = max(float(vec[tid].item()), min(style, 4.0) / 4.0)
    return vec


def parse_player_trait_counts(player: dict, trait_ids: dict) -> torch.Tensor:
    """Encode Riot participant trait num_units as a float vector (same index as style)."""
    for trait in player.get("traits", []):
        name = trait.get("name", "")
        if not name:
            continue
        if name not in trait_ids:
            trait_ids[name] = len(trait_ids)

    vec = empty_trait_tensor(len(trait_ids))
    for trait in player.get("traits", []):
        name = trait.get("name", "")
        if name not in trait_ids:
            continue
        count = float(trait.get("num_units", 0) or 0)
        tid = trait_ids[name]
        if tid < len(vec):
            vec[tid] = max(float(vec[tid].item()), count)
    return vec


def load_champion_trait_map(path: str = CHAMPION_MAP_PATH) -> dict:
    """Load {character_id: {traits: [...], cost: int, breakpoints: {trait: [n,...]}}}."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def style_from_count(count: int, breakpoints: List[int]) -> int:
    """Map unit count to Riot-like style 0-4 using sorted breakpoints."""
    if not breakpoints or count <= 0:
        return 0
    bps = sorted(breakpoints)
    tier = 0
    for bp in bps:
        if count >= bp:
            tier += 1
        else:
            break
    if tier <= 0:
        return 0
    if tier == 1:
        return 1
    if tier == 2:
        return 2
    if tier == 3:
        return 3
    return 4


def recompute_traits_from_board(
    board: torch.Tensor,
    id_to_name: Dict[int, str],
    trait_ids: dict,
    champ_map: dict,
) -> torch.Tensor:
    """Recompute trait style vector from remaining champion units on a partial board."""
    trait_counts: Dict[str, int] = {}
    trait_breakpoints: Dict[str, List[int]] = {}

    # Build reverse lookup from normalized names for mismatched prefixes
    name_aliases: Dict[str, str] = {}
    for cid, info in champ_map.items():
        name_aliases[normalize_trait_name(cid)] = cid
        name_aliases[cid.lower()] = cid

    for i in range(BOARD_SLOTS):
        unit_id = int(board[i, 0].item())
        if unit_id == 0:
            continue
        unit_name = id_to_name.get(unit_id, "")
        info = champ_map.get(unit_name)
        if info is None:
            alias = name_aliases.get(normalize_trait_name(unit_name)) or name_aliases.get(
                unit_name.lower()
            )
            info = champ_map.get(alias or "", {})
        if not info:
            continue
        for trait_name in info.get("traits", []):
            # Prefer Riot trait ID keys when present in vocab
            resolved = trait_name
            if trait_name not in trait_ids:
                # try match by normalized name
                target = normalize_trait_name(trait_name)
                for known in trait_ids:
                    if normalize_trait_name(known) == target:
                        resolved = known
                        break
            trait_counts[resolved] = trait_counts.get(resolved, 0) + 1
            bps = info.get("breakpoints", {}).get(trait_name) or info.get("breakpoints", {}).get(
                resolved, []
            )
            if bps:
                trait_breakpoints[resolved] = bps

    # Also pull breakpoints from any champ that lists them for these traits
    if not any(trait_breakpoints.values()):
        for info in champ_map.values():
            for t, bps in info.get("breakpoints", {}).items():
                resolved = t
                if t not in trait_ids:
                    target = normalize_trait_name(t)
                    for known in trait_ids:
                        if normalize_trait_name(known) == target:
                            resolved = known
                            break
                if resolved in trait_counts and resolved not in trait_breakpoints:
                    trait_breakpoints[resolved] = bps

    vec = empty_trait_tensor(len(trait_ids))
    for trait_name, count in trait_counts.items():
        if trait_name not in trait_ids:
            continue
        style = style_from_count(count, trait_breakpoints.get(trait_name, []))
        tid = trait_ids[trait_name]
        if tid < len(vec):
            vec[tid] = style / 4.0
    return vec


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
    champ_map: Optional[dict] = None,
) -> GameState:
    board = board_from_unit_names(unit_names, ids_file, items_by_unit=items_by_unit)
    trait_ids = ids_file.get("traitIDs", {"reserve": 0})
    id_to_name = {v: k for k, v in ids_file["unitIDs"].items()}
    if champ_map is None:
        champ_map = load_champion_trait_map()
    traits = recompute_traits_from_board(board, id_to_name, trait_ids, champ_map)
    return GameState(
        board=board,
        stage=stage,
        level=level,
        gold=gold,
        components=components,
        traits=traits,
    )


def parse_player_board(
    player: dict,
    unit_ids: dict,
    item_ids: dict,
    champion_ids: Optional[set] = None,
) -> torch.Tensor:
    """Parse Riot API participant units into board tensor. Optionally filter summons."""
    board_rows = []
    for unit in player["units"]:
        unit_name = unit["character_id"]
        if champion_ids is not None and unit_name not in champion_ids:
            # Allow fuzzy match for prefix variants
            matched = False
            target = normalize_trait_name(unit_name)
            for cid in champion_ids:
                if normalize_trait_name(cid) == target:
                    unit_name = cid
                    matched = True
                    break
            if not matched:
                continue

        if unit_name not in unit_ids:
            unit_ids[unit_name] = len(unit_ids)

        unit_cost = unit["rarity"] + 1
        star_level = unit["tier"]
        unit_items = [0, 0, 0]
        item_names = unit.get("itemNames") or []

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

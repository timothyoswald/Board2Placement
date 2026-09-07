import copy
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import componentData
import numpy as np
import torch
from sklearn.cluster import KMeans

from settings import (
    BOARD_FINDER_PATH,
    CHAMPION_MAP_PATH,
    CLEANED_DIR,
    CLUSTER_COUNT,
    IDS_PATH,
    PLACEMENT_MODEL_PATH,
    TRAIT_WEIGHT_IN_CLUSTER,
)
from state import (
    GameState,
    cosine_similarity,
    game_state_from_names,
    load_champion_trait_map,
    load_id_maps,
)

dataDir = CLEANED_DIR
IDsDir = IDS_PATH
modelDir = BOARD_FINDER_PATH
clusterCount = CLUSTER_COUNT

# Comp matching weights (items weighted highest per plan)
WEIGHT_ITEM_FIT = 0.45
WEIGHT_UNIT_FIT = 0.25
WEIGHT_ECONOMY = 0.20
WEIGHT_TRANSITION = 0.10

# Unit must appear on at least this fraction of cluster boards to be a "carry core"
MIN_CARRY_PRESENCE = 0.40
# Emblem slam rate among cluster boards to flag emblem-gated trait
MIN_EMBLEM_RATE = 0.20


@dataclass
class ClusterProfile:
    """Rich profile for a learned composition archetype."""

    cluster_idx: int
    unit_centroid: np.ndarray
    core_units: List[Tuple[str, float]]
    item_weights: Dict[str, Dict[str, float]]  # unit -> item -> lift-weighted rate
    item_lift: Dict[str, Dict[str, float]]  # unit -> item -> lift over global
    avg_board_cost: float = 0.0
    avg_level: float = 8.0
    count_4cost: float = 0.0
    count_5cost: float = 0.0
    required_level: float = 7.0
    carries: List[str] = field(default_factory=list)
    tanks: List[str] = field(default_factory=list)
    # (api_name, mean_num_units) for labeling
    core_traits: List[Tuple[str, float]] = field(default_factory=list)
    display_name: str = ""
    trait_core_units: List[str] = field(default_factory=list)
    splash_units: List[str] = field(default_factory=list)
    needs_emblem: List[str] = field(default_factory=list)  # trait api names
    emblem_items: List[str] = field(default_factory=list)


@dataclass
class CompRecommendation:
    cluster_idx: int
    total_score: float
    unit_fit: float
    item_fit: float
    craftable_score: float
    economy_fit: float
    transition_cost: float
    feasibility: str
    core_units_owned: List[str]
    missing_units: List[str]
    recommended_items: List[dict]
    core_units: List[str]
    expected_placement: Optional[float] = None
    projected_placement: Optional[float] = None
    display_name: str = ""
    core_traits: List[str] = field(default_factory=list)


def _fallback_short_trait(name: str) -> str:
    for prefix in (
        "TFT18_",
        "TFT17_",
        "TFT16_",
        "Set18_",
        "Set17_",
        "DA_18_",
        "DA_",
        "TFT_",
        "Trait_",
    ):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if name.endswith("18"):
        name = name[:-2]
    if name.endswith("_UniqueTrait") or name.endswith("UniqueTrait"):
        name = name.replace("_UniqueTrait", "").replace("UniqueTrait", "")
    return name or "Unknown"


def _short_unit_name(name: str) -> str:
    for prefix in ("DA_18_", "DA_", "TFT18_", "TFT17_", "TFT16_", "TFT_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    name = name.replace("18_AP", "").replace("18_AD", "").replace("_AP", "").replace("_AD", "")
    if name.endswith("18"):
        name = name[:-2]
    name = name.replace("_Base", "").replace("_Sunbeam", " (Sunbeam)").replace("_Fae", " (Fae)")
    name = name.replace("_Primal", " (Primal)")
    return name or "Unknown"


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


def _nearest_breakpoint(count: float, breakpoints: List[int]) -> int:
    """Map mean unit count to the highest met breakpoint (or round if no bps)."""
    if not breakpoints:
        return max(1, int(round(count)))
    bps = sorted(breakpoints)
    met = [bp for bp in bps if count + 1e-6 >= bp]
    if met:
        return met[-1]
    return max(1, int(round(count)))


class BoardFinder:
    def __init__(self, placement_model_path: str = PLACEMENT_MODEL_PATH):
        (
            self.ids_file,
            self.id_to_name,
            self.id_to_item,
            self.num_units,
            _,
            self.num_traits,
        ) = load_id_maps(IDsDir)
        self.name_to_id = self.ids_file["unitIDs"]
        self.item_name_to_id = self.ids_file["itemIDs"]
        self.trait_ids = self.ids_file.get("traitIDs", {"reserve": 0})
        self.id_to_trait = {v: k for k, v in self.trait_ids.items()}
        self.champ_map = load_champion_trait_map(CHAMPION_MAP_PATH)
        meta = self.champ_map.get("_meta", {})
        self.trait_display_names: Dict[str, str] = meta.get("trait_display_names", {})
        self.unique_traits: set = set(meta.get("unique_traits", []))
        self.trait_breakpoints: Dict[str, List[int]] = meta.get("trait_breakpoints", {})
        self.emblem_to_trait: Dict[str, str] = meta.get("emblem_to_trait", {})
        self.trait_to_emblem: Dict[str, str] = {v: k for k, v in self.emblem_to_trait.items()}
        self.trait_champions: Dict[str, List[str]] = meta.get("trait_champions", {})
        self.placement_model = None
        self.placement_device = "cpu"

        # Global item baselines for lift calculation
        self.global_item_counts: Dict[str, Dict[str, int]] = {}
        self.global_unit_item_totals: Dict[str, int] = {}
        self.cluster_profiles: Dict[int, ClusterProfile] = {}

        if os.path.exists(modelDir):
            self.loadState()
        else:
            self.train()
            self.saveState()

        self._load_placement_model(placement_model_path)

    def trait_display_name(self, api_name: str) -> str:
        return self.trait_display_names.get(api_name) or _fallback_short_trait(api_name)

    def _unit_has_trait(self, unit_name: str, trait_api: str) -> bool:
        info = self.champ_map.get(unit_name) or {}
        return trait_api in (info.get("traits") or [])

    def _cluster_vector(self, unit_vec: np.ndarray, trait_vec: np.ndarray) -> np.ndarray:
        u = _l2_normalize(unit_vec.astype(np.float64))
        t = _l2_normalize(trait_vec.astype(np.float64)) * TRAIT_WEIGHT_IN_CLUSTER
        return np.concatenate([u, t])

    def _load_placement_model(self, model_path: str):
        if not os.path.exists(model_path):
            return
        try:
            from model import Board2Placement

            self.placement_device = (
                torch.accelerator.current_accelerator().type
                if torch.accelerator.is_available()
                else "cpu"
            )
            self.placement_model = Board2Placement(predict_distribution=True).to(
                self.placement_device
            )
            self.placement_model.load_state_dict(
                torch.load(model_path, map_location=self.placement_device, weights_only=True)
            )
            self.placement_model.eval()
            print("placement model loaded")
        except Exception as e:
            print(f"could not load placement model: {e}")
            self.placement_model = None

    def _predict_placement(self, game_state: GameState) -> Optional[float]:
        if self.placement_model is None:
            return None
        board = game_state.board.unsqueeze(0).to(self.placement_device)
        context = game_state.context_tensor().unsqueeze(0).to(self.placement_device)
        traits = game_state.traits
        if traits is None or traits.numel() == 0:
            traits = torch.zeros(self.num_traits, dtype=torch.float32)
        if traits.numel() != self.num_traits:
            fixed = torch.zeros(self.num_traits, dtype=torch.float32)
            n = min(traits.numel(), self.num_traits)
            fixed[:n] = traits[:n]
            traits = fixed
        traits = traits.unsqueeze(0).to(self.placement_device)
        with torch.no_grad():
            expected = self.placement_model.expected_placement(board, context, traits)
        return float(expected.item())

    def _build_projected_state(self, game_state: GameState, cluster_idx: int) -> GameState:
        """Add missing core units at 1-star to simulate capped board."""
        projected = GameState(
            board=game_state.board.clone(),
            stage=game_state.stage,
            level=max(game_state.level, self.cluster_profiles[cluster_idx].required_level),
            gold=game_state.gold,
            components=game_state.components.copy(),
            traits=game_state.traits.clone()
            if isinstance(game_state.traits, torch.Tensor) and game_state.traits.numel()
            else torch.zeros(self.num_traits),
        )
        occupied = sum(1 for i in range(14) if int(projected.board[i, 0].item()) != 0)
        profile = self.cluster_profiles[cluster_idx]
        owned = set(game_state.unit_names(self.id_to_name))

        slot = 0
        for unit_name, _ in profile.core_units:
            if unit_name in owned or occupied >= 14:
                continue
            while slot < 14 and int(projected.board[slot, 0].item()) != 0:
                slot += 1
            if slot >= 14:
                break
            unit_id = self.name_to_id.get(unit_name, 0)
            if unit_id:
                projected.board[slot, 0] = unit_id
                projected.board[slot, 1] = 1
                projected.board[slot, 2] = 1
                occupied += 1
        return projected

    def train(self):
        data = torch.load(dataDir, weights_only=False)
        winning_boards = []
        winning_boards_tensors = []
        winning_contexts = []
        winning_traits = []
        winning_counts = []

        for entry in data:
            counts = torch.zeros(self.num_traits)
            if len(entry) == 2:
                board_tensor, placement_tensor = entry
                context = torch.tensor([4.0, 8.0, 0.0])
                traits = torch.zeros(self.num_traits)
            elif len(entry) == 3:
                board_tensor, context, placement_tensor = entry
                traits = torch.zeros(self.num_traits)
            elif len(entry) == 5:
                board_tensor, context, traits, counts, placement_tensor = entry
            else:
                board_tensor, context, traits, placement_tensor = entry

            if traits.numel() != self.num_traits:
                fixed = torch.zeros(self.num_traits)
                n = min(traits.numel(), self.num_traits)
                fixed[:n] = traits[:n]
                traits = fixed
            if counts.numel() != self.num_traits:
                fixed_c = torch.zeros(self.num_traits)
                n = min(counts.numel(), self.num_traits)
                fixed_c[:n] = counts[:n]
                counts = fixed_c

            placement = placement_tensor.item()

            if placement <= 4.0:
                gs = GameState(board=board_tensor, traits=traits)
                unit_vec = gs.unit_vector(self.num_units)
                trait_vec = traits.numpy() if isinstance(traits, torch.Tensor) else np.asarray(traits)
                winning_boards.append(self._cluster_vector(unit_vec, trait_vec))
                winning_boards_tensors.append(board_tensor)
                winning_contexts.append(context)
                winning_traits.append(traits)
                winning_counts.append(counts)

        self.vectors = np.array(winning_boards)
        print(f"found {len(self.vectors)} winning boards")

        self.KMeans = KMeans(n_clusters=clusterCount, random_state=42, n_init=10)
        self.KMeans.fit(self.vectors)
        self.cluster_labels = self.KMeans.labels_
        print("found clusters")

        self._build_global_item_baselines(winning_boards_tensors)
        self.cluster_items_for_units = {i: dict() for i in range(clusterCount)}

        for i, board_tensor in enumerate(winning_boards_tensors):
            cluster_idx = self.cluster_labels[i]
            for j in range(14):
                unit_id = int(board_tensor[j, 0].item())
                if unit_id == 0:
                    continue
                items = [
                    int(board_tensor[j, 3].item()),
                    int(board_tensor[j, 4].item()),
                    int(board_tensor[j, 5].item()),
                ]
                if sum(items) == 0:
                    continue
                unit_name = self.id_to_name[unit_id]
                if unit_name not in self.cluster_items_for_units[cluster_idx]:
                    self.cluster_items_for_units[cluster_idx][unit_name] = dict()
                for item_id in items:
                    if item_id != 0:
                        item_name = self.id_to_item[item_id]
                        bucket = self.cluster_items_for_units[cluster_idx][unit_name]
                        bucket[item_name] = bucket.get(item_name, 0) + 1

        self.item_weights_per_cluster = {}
        for cluster_idx, units_dict in self.cluster_items_for_units.items():
            self.item_weights_per_cluster[cluster_idx] = {}
            for unit_name, item_counts in units_dict.items():
                unit_count = sum(item_counts.values())
                if unit_count < 10:
                    continue
                good_items = {}
                for item, count in item_counts.items():
                    slam_rate = count / unit_count
                    global_rate = self._global_item_rate(unit_name, item)
                    lift = slam_rate / max(global_rate, 0.01)
                    if slam_rate > 0.05:
                        good_items[item] = slam_rate * min(lift, 3.0)
                if good_items:
                    self.item_weights_per_cluster[cluster_idx][unit_name] = good_items

        self._build_cluster_profiles(
            winning_boards_tensors, winning_contexts, winning_traits, winning_counts
        )
        print("cluster profiles built")

    def _build_global_item_baselines(self, boards: List[torch.Tensor]):
        for board_tensor in boards:
            for j in range(14):
                unit_id = int(board_tensor[j, 0].item())
                if unit_id == 0:
                    continue
                unit_name = self.id_to_name[unit_id]
                if unit_name not in self.global_item_counts:
                    self.global_item_counts[unit_name] = {}
                for k in range(3, 6):
                    item_id = int(board_tensor[j, k].item())
                    if item_id != 0:
                        item_name = self.id_to_item[item_id]
                        self.global_item_counts[unit_name][item_name] = (
                            self.global_item_counts[unit_name].get(item_name, 0) + 1
                        )
                        self.global_unit_item_totals[unit_name] = (
                            self.global_unit_item_totals.get(unit_name, 0) + 1
                        )

    def _global_item_rate(self, unit_name: str, item_name: str) -> float:
        total = self.global_unit_item_totals.get(unit_name, 0)
        if total == 0:
            return 0.01
        count = self.global_item_counts.get(unit_name, {}).get(item_name, 0)
        return count / total

    def _extract_core_traits(
        self,
        style_centroid: np.ndarray,
        count_centroid: np.ndarray,
    ) -> List[Tuple[str, float]]:
        """Return (trait_api, mean_num_units) for traits active on the cluster."""
        core = []
        for tid, style_val in enumerate(style_centroid):
            if tid == 0:
                continue
            if style_val > 0.35 and tid in self.id_to_trait:
                mean_count = float(count_centroid[tid]) if tid < len(count_centroid) else 0.0
                # Prefer count when available; fall back to style hint only if zero
                if mean_count < 0.05:
                    mean_count = float(style_val)  # weak fallback; display uses breakpoints
                core.append((self.id_to_trait[tid], mean_count))
        core.sort(key=lambda x: x[1], reverse=True)
        return core[:6]

    def _display_name_from_traits(
        self,
        core_traits: List[Tuple[str, float]],
        needs_emblem: Optional[List[str]] = None,
    ) -> str:
        if not core_traits:
            return "Flexible / splash"
        needs_emblem = needs_emblem or []
        parts = []
        for name, mean_count in core_traits[:3]:
            display = self.trait_display_name(name)
            bps = self.trait_breakpoints.get(name, [])
            is_unique = name in self.unique_traits or bps == [1]
            emblem_tag = " (emblem)" if name in needs_emblem else ""

            if is_unique and name != "DA_18_Rival":
                # Single-champion unique: no fake multi-count
                parts.append(f"{display}{emblem_tag}")
            elif name == "DA_18_Rival" or (is_unique and len(bps) > 1):
                bp = _nearest_breakpoint(mean_count, bps or [1, 2])
                parts.append(f"{bp} {display}{emblem_tag}")
            else:
                bp = _nearest_breakpoint(mean_count, bps)
                parts.append(f"{bp} {display}{emblem_tag}")
        return " / ".join(parts)

    def _unit_presence(
        self, boards: List[torch.Tensor], indices: List[int]
    ) -> Dict[str, float]:
        if not indices:
            return {}
        counts: Dict[str, int] = {}
        for i in indices:
            board = boards[i]
            seen = set()
            for j in range(14):
                uid = int(board[j, 0].item())
                if uid == 0 or uid in seen:
                    continue
                seen.add(uid)
                uname = self.id_to_name.get(uid)
                if uname:
                    counts[uname] = counts.get(uname, 0) + 1
        n = len(indices)
        return {u: c / n for u, c in counts.items()}

    def _split_trait_cores(
        self,
        core_units: List[Tuple[str, float]],
        core_traits: List[Tuple[str, float]],
    ) -> Tuple[List[str], List[str]]:
        """Units that belong to the top trait(s) vs splash fillers."""
        if not core_traits:
            return [u for u, _ in core_units], []
        main_traits = [t for t, _ in core_traits[:2]]
        trait_cores = []
        splash = []
        for unit_name, _ in core_units:
            if any(self._unit_has_trait(unit_name, t) for t in main_traits):
                trait_cores.append(unit_name)
            else:
                splash.append(unit_name)
        # If nothing matched (map miss), keep all as cores
        if not trait_cores:
            return [u for u, _ in core_units], []
        return trait_cores, splash

    def _detect_emblem_needs(
        self,
        boards: List[torch.Tensor],
        indices: List[int],
        core_traits: List[Tuple[str, float]],
        presence: Dict[str, float],
    ) -> Tuple[List[str], List[str]]:
        """Return (trait_apis needing emblem, emblem item names)."""
        if not indices:
            return [], []
        needs: List[str] = []
        emblems: List[str] = []
        n = len(indices)

        # Emblem slam rates on boards in this cluster
        emblem_hits: Dict[str, int] = {e: 0 for e in self.emblem_to_trait}
        for i in indices:
            board = boards[i]
            found = set()
            for j in range(14):
                for k in range(3, 6):
                    item_id = int(board[j, k].item())
                    if item_id == 0:
                        continue
                    item_name = self.id_to_item.get(item_id, "")
                    if item_name in self.emblem_to_trait:
                        found.add(item_name)
            for e in found:
                emblem_hits[e] = emblem_hits.get(e, 0) + 1

        for trait_api, mean_count in core_traits:
            bps = self.trait_breakpoints.get(trait_api, [])
            if not bps or trait_api in self.unique_traits:
                continue
            # Natural holders among frequent units
            natural = 0.0
            for unit_name, rate in presence.items():
                if rate >= 0.25 and self._unit_has_trait(unit_name, trait_api):
                    natural += 1.0
            # Lux Avatar counts as 2 toward origins — approximate if Lux variant present
            for unit_name, rate in presence.items():
                if rate >= 0.15 and "Lux" in unit_name and self._unit_has_trait(unit_name, trait_api):
                    natural += 1.0  # extra Avatar weight
                    break

            emblem_item = self.trait_to_emblem.get(trait_api)
            emblem_rate = (emblem_hits.get(emblem_item, 0) / n) if emblem_item else 0.0
            activated = _nearest_breakpoint(mean_count, bps)
            # Emblem-gated if activated breakpoint exceeds natural holders, or emblems are common
            if emblem_item and (
                emblem_rate >= MIN_EMBLEM_RATE
                or (activated > natural + 0.5 and activated >= (bps[1] if len(bps) > 1 else bps[0]))
            ):
                if trait_api not in needs:
                    needs.append(trait_api)
                    emblems.append(emblem_item)
        return needs, emblems

    def _build_cluster_profiles(
        self,
        boards: List[torch.Tensor],
        contexts: List[torch.Tensor],
        traits_list: List[torch.Tensor],
        counts_list: Optional[List[torch.Tensor]] = None,
    ):
        counts_list = counts_list or []
        cluster_costs = {i: [] for i in range(clusterCount)}
        cluster_levels = {i: [] for i in range(clusterCount)}
        cluster_4cost = {i: [] for i in range(clusterCount)}
        cluster_5cost = {i: [] for i in range(clusterCount)}
        cluster_style_sums = {i: [] for i in range(clusterCount)}
        cluster_count_sums = {i: [] for i in range(clusterCount)}
        cluster_indices = {i: [] for i in range(clusterCount)}

        for i, board_tensor in enumerate(boards):
            cidx = self.cluster_labels[i]
            cluster_indices[cidx].append(i)
            gs = GameState(board=board_tensor)
            cluster_costs[cidx].append(gs.board_cost())
            cost_counts = gs.count_units_by_cost(4)
            cluster_4cost[cidx].append(cost_counts.get(4, 0))
            cluster_5cost[cidx].append(cost_counts.get(5, 0))
            if i < len(contexts):
                cluster_levels[cidx].append(float(contexts[i][1].item()))
            if i < len(traits_list):
                t = traits_list[i]
                arr = t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
                cluster_style_sums[cidx].append(arr)
            if i < len(counts_list):
                c = counts_list[i]
                carr = c.numpy() if isinstance(c, torch.Tensor) else np.asarray(c)
                cluster_count_sums[cidx].append(carr)

        for cluster_idx in range(clusterCount):
            centroid = self.KMeans.cluster_centers_[cluster_idx]
            unit_part = centroid[: self.num_units]

            core_units = []
            for uid, val in enumerate(unit_part):
                if val > 0.15 and uid in self.id_to_name:
                    core_units.append((self.id_to_name[uid], float(val)))
            core_units.sort(key=lambda x: x[1], reverse=True)
            core_units = [(n, v) for n, v in core_units if v > 0.12][:10]

            if cluster_style_sums[cluster_idx]:
                style_centroid = np.mean(cluster_style_sums[cluster_idx], axis=0)
            else:
                style_centroid = np.zeros(self.num_traits)
            if cluster_count_sums[cluster_idx]:
                count_centroid = np.mean(cluster_count_sums[cluster_idx], axis=0)
            else:
                count_centroid = np.zeros(self.num_traits)

            core_traits = self._extract_core_traits(style_centroid, count_centroid)
            presence = self._unit_presence(boards, cluster_indices[cluster_idx])
            trait_cores, splash = self._split_trait_cores(core_units, core_traits)
            needs_emblem, emblem_items = self._detect_emblem_needs(
                boards, cluster_indices[cluster_idx], core_traits, presence
            )
            display_name = self._display_name_from_traits(core_traits, needs_emblem)

            item_lift = {}
            cluster_weights = self.item_weights_per_cluster.get(cluster_idx, {})
            for unit_name, items in cluster_weights.items():
                item_lift[unit_name] = {}
                for item_name, weighted_rate in items.items():
                    global_rate = self._global_item_rate(unit_name, item_name)
                    item_lift[unit_name][item_name] = weighted_rate / max(global_rate, 0.01)

            # Prefer trait-holding cores for carry/tank labels
            carry_pool = (
                [(u, 1.0) for u in trait_cores]
                if trait_cores
                else core_units
            )
            carries, tanks = self._infer_roles(
                cluster_idx, cluster_weights, carry_pool, presence
            )

            avg_4 = float(np.mean(cluster_4cost[cluster_idx])) if cluster_4cost[cluster_idx] else 0
            avg_5 = float(np.mean(cluster_5cost[cluster_idx])) if cluster_5cost[cluster_idx] else 0
            required_level = 7.0 + min(avg_5, 2) * 0.5 + min(avg_4, 4) * 0.1

            self.cluster_profiles[cluster_idx] = ClusterProfile(
                cluster_idx=cluster_idx,
                unit_centroid=unit_part,
                core_units=core_units,
                item_weights=cluster_weights,
                item_lift=item_lift,
                avg_board_cost=float(np.mean(cluster_costs[cluster_idx]))
                if cluster_costs[cluster_idx]
                else 0,
                avg_level=float(np.mean(cluster_levels[cluster_idx]))
                if cluster_levels[cluster_idx]
                else 8.0,
                count_4cost=avg_4,
                count_5cost=avg_5,
                required_level=required_level,
                carries=carries,
                tanks=tanks,
                core_traits=core_traits,
                display_name=display_name,
                trait_core_units=trait_cores,
                splash_units=splash,
                needs_emblem=needs_emblem,
                emblem_items=emblem_items,
            )

    def _infer_roles(
        self,
        cluster_idx: int,
        cluster_weights: Dict[str, Dict[str, float]],
        core_units: Optional[List[Tuple[str, float]]] = None,
        presence: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[str], List[str]]:
        """Units with offensive items -> carries; defensive -> tanks.

        Only frequent core units are eligible so flex item holders are excluded.
        """
        offensive_items = {
            "TFT_Item_BlueBuff",
            "TFT_Item_JeweledGauntlet",
            "TFT_Item_RabadonsDeathcap",
            "TFT_Item_InfinityEdge",
            "TFT_Item_GuinsoosRageblade",
            "TFT_Item_Deathblade",
            "TFT_Item_ArchangelsStaff",
            "TFT_Item_HextechGunblade",
            "TFT_Item_StatikkShiv",
            "DA_BlueBuff",
            "DA_JeweledGauntlet",
            "DA_RabadonsDeathcap",
            "DA_InfinityEdge",
            "DA_GuinsoosRageblade",
            "DA_Deathblade",
            "DA_ArchangelsStaff",
            "DA_HextechGunblade",
            "DA_LastWhisper",
            "DA_GiantSlayer",
            "DA_StrikersFlail",
            "DA_SpearOfShojin",
            "DA_NashorsTooth",
        }
        defensive_items = {
            "TFT_Item_BrambleVest",
            "TFT_Item_DragonsClaw",
            "TFT_Item_WarmogsArmor",
            "TFT_Item_GargoyleStoneplate",
            "TFT_Item_SunfireCape",
            "TFT_Item_Redemption",
            "DA_BrambleVest",
            "DA_DragonsClaw",
            "DA_WarmogsArmor",
            "DA_GargoyleStoneplate",
            "DA_SunfireCape",
            "DA_Evenshroud",
            "DA_SteadfastHeart",
            "DA_SpiritVisage",
        }

        core_names = {u for u, _ in (core_units or [])}
        presence = presence or {}

        def eligible(unit_name: str) -> bool:
            if core_names and unit_name not in core_names:
                return False
            if presence and presence.get(unit_name, 0.0) < MIN_CARRY_PRESENCE:
                return False
            return True

        carry_scores = {}
        tank_scores = {}
        for unit_name, items in cluster_weights.items():
            if not eligible(unit_name):
                continue
            carry_scores[unit_name] = sum(w for i, w in items.items() if i in offensive_items)
            tank_scores[unit_name] = sum(w for i, w in items.items() if i in defensive_items)

        # Fallback: if filters wiped everyone, use eligible cores by presence / order
        if not carry_scores and core_units:
            for unit_name, _ in core_units[:5]:
                if not eligible(unit_name):
                    continue
                items = cluster_weights.get(unit_name, {})
                carry_scores[unit_name] = sum(
                    w for i, w in items.items() if i in offensive_items
                ) + 0.01 * presence.get(unit_name, 0.5)

        # Last resort: top presence cores even without offensive items logged
        if not carry_scores and core_units:
            ranked = sorted(
                [u for u, _ in core_units if eligible(u)],
                key=lambda u: presence.get(u, 0.0),
                reverse=True,
            )
            carry_scores = {u: presence.get(u, 0.01) for u in ranked[:3]}

        carries = sorted(carry_scores, key=carry_scores.get, reverse=True)[:3]
        carries = [u for u in carries if carry_scores[u] > 0]
        tanks = sorted(tank_scores, key=tank_scores.get, reverse=True)[:3]
        tanks = [u for u in tanks if tank_scores[u] > 0]
        return carries, tanks

    # --- Scoring methods ---

    def score_unit_fit(self, game_state: GameState, cluster_idx: int) -> float:
        profile = self.cluster_profiles[cluster_idx]
        board_vec = game_state.unit_vector(self.num_units)
        trait_vec = (
            game_state.traits.numpy()
            if isinstance(game_state.traits, torch.Tensor) and game_state.traits.numel()
            else np.zeros(self.num_traits)
        )
        if trait_vec.shape[0] != self.num_traits:
            fixed = np.zeros(self.num_traits)
            n = min(len(trait_vec), self.num_traits)
            fixed[:n] = trait_vec[:n]
            trait_vec = fixed
        board_cluster = self._cluster_vector(board_vec, trait_vec)
        # Rebuild profile centroid in same space
        unit_c = profile.unit_centroid
        if len(unit_c) != self.num_units:
            fixed_u = np.zeros(self.num_units)
            n = min(len(unit_c), self.num_units)
            fixed_u[:n] = unit_c[:n]
            unit_c = fixed_u
        trait_c = np.zeros(self.num_traits)
        for name, val in profile.core_traits:
            tid = self.trait_ids.get(name)
            if tid is not None and tid < self.num_traits:
                trait_c[tid] = val
        profile_vec = self._cluster_vector(unit_c, trait_c)
        return cosine_similarity(board_cluster, profile_vec)

    def score_item_fit(self, game_state: GameState, cluster_idx: int) -> float:
        profile = self.cluster_profiles[cluster_idx]
        assignments = game_state.item_assignments(self.id_to_item, self.id_to_name)
        if not assignments:
            return 0.0

        total = 0.0
        for unit_name, item_name in assignments:
            unit_weights = profile.item_weights.get(unit_name, {})
            lift_map = profile.item_lift.get(unit_name, {})
            if item_name in unit_weights:
                total += unit_weights[item_name]
            elif item_name in lift_map:
                total += lift_map[item_name] * 0.1
            for core_unit, lift_items in profile.item_lift.items():
                if item_name in lift_items and core_unit in profile.carries:
                    total += lift_items[item_name] * 0.3

        return min(total / max(len(assignments), 1), 1.0)

    def score_craftable_items(
        self, game_state: GameState, cluster_idx: int
    ) -> Tuple[float, List[dict]]:
        craft_score, items = self.get_best_items(cluster_idx, game_state.components)
        return min(craft_score / 50.0, 1.0), items

    def score_economy_fit(self, game_state: GameState, cluster_idx: int) -> float:
        profile = self.cluster_profiles[cluster_idx]
        level = game_state.level
        gold = game_state.gold
        stage = game_state.stage

        score = 1.0
        if level < profile.required_level - 1:
            score -= 0.3 * (profile.required_level - level)
        if profile.count_5cost >= 2 and level < 8:
            score -= 0.4
        if profile.count_5cost >= 1 and level < 7 and stage <= 4:
            score -= 0.3
        if gold < 20 and profile.avg_board_cost > game_state.board_cost() + 30:
            score -= 0.25
        if gold < 10 and profile.count_4cost >= 3:
            score -= 0.2
        return max(0.0, min(1.0, score))

    def score_transition_cost(self, game_state: GameState, cluster_idx: int) -> float:
        """Lower is better; returns cost penalty 0-1."""
        profile = self.cluster_profiles[cluster_idx]
        owned = set(game_state.unit_names(self.id_to_name))
        if not profile.core_units:
            return 0.0

        missing_cost = 0.0
        for unit_name, importance in profile.core_units:
            if unit_name not in owned:
                missing_cost += importance
        return min(missing_cost / max(len(profile.core_units), 1), 1.0)

    def feasibility_rating(self, economy_fit: float, transition_cost: float) -> str:
        combined = economy_fit * 0.6 + (1 - transition_cost) * 0.4
        if combined >= 0.75:
            return "easy"
        if combined >= 0.55:
            return "medium"
        if combined >= 0.35:
            return "greedy"
        return "unrealistic"

    def score_cluster(self, game_state: GameState, cluster_idx: int) -> CompRecommendation:
        unit_fit = self.score_unit_fit(game_state, cluster_idx)
        item_fit = self.score_item_fit(game_state, cluster_idx)
        craftable, recommended_items = self.score_craftable_items(game_state, cluster_idx)
        economy_fit = self.score_economy_fit(game_state, cluster_idx)
        transition = self.score_transition_cost(game_state, cluster_idx)

        total = (
            WEIGHT_ITEM_FIT * (item_fit + craftable * 0.5)
            + WEIGHT_UNIT_FIT * unit_fit
            + WEIGHT_ECONOMY * economy_fit
            + WEIGHT_TRANSITION * (1 - transition)
        )

        profile = self.cluster_profiles[cluster_idx]
        owned = set(game_state.unit_names(self.id_to_name))
        core_owned = [u for u, _ in profile.core_units if u in owned]
        missing = [u for u, imp in profile.core_units if u not in owned and imp > 0.12]
        missing.sort(key=lambda u: next(v for n, v in profile.core_units if n == u), reverse=True)

        return CompRecommendation(
            cluster_idx=cluster_idx,
            total_score=total,
            unit_fit=unit_fit,
            item_fit=item_fit,
            craftable_score=craftable,
            economy_fit=economy_fit,
            transition_cost=transition,
            feasibility=self.feasibility_rating(economy_fit, transition),
            core_units_owned=core_owned,
            missing_units=missing,
            recommended_items=recommended_items,
            core_units=[u for u, _ in profile.core_units],
            expected_placement=self._predict_placement(game_state),
            projected_placement=self._predict_placement(
                self._build_projected_state(game_state, cluster_idx)
            ),
            display_name=profile.display_name,
            core_traits=[self.trait_display_name(n) for n, _ in profile.core_traits],
        )

    def recommend(self, game_state: GameState, top_k: int = 3) -> List[CompRecommendation]:
        scores = [self.score_cluster(game_state, i) for i in range(clusterCount)]
        scores.sort(key=lambda r: r.total_score, reverse=True)
        return scores[:top_k]

    def complete_board(
        self,
        current_board: List[str],
        curr_items: dict,
        stage: float = 3.0,
        level: float = 6.0,
        gold: float = 20.0,
    ) -> Tuple[int, List[str], List[dict], CompRecommendation]:
        """Backward-compatible API; returns best cluster recommendation."""
        game_state = game_state_from_names(
            current_board,
            curr_items,
            self.ids_file,
            stage=stage,
            level=level,
            gold=gold,
        )
        best = self.recommend(game_state, top_k=1)[0]
        return (
            best.cluster_idx,
            best.missing_units,
            best.recommended_items,
            best,
        )

    def _normalize_components(self, curr_items: dict) -> dict:
        """Map TFT_Item_* component names to DA_Component_* when aliases exist."""
        aliases = getattr(componentData, "componentAliases", {})
        normalized = {}
        for name, count in curr_items.items():
            key = aliases.get(name, name)
            normalized[key] = normalized.get(key, 0) + count
        return normalized

    def get_best_items(self, cluster_idx: int, curr_items: dict):
        cluster_weights = self.item_weights_per_cluster[cluster_idx]
        possible_items = []
        profile = self.cluster_profiles.get(cluster_idx)
        core_units = profile.core_units if profile else []

        for unit_name, unit_importance in core_units:
            if unit_name not in cluster_weights:
                continue
            for item_name, item_weight in cluster_weights[unit_name].items():
                if item_name not in componentData.itemToComponent:
                    continue
                score = unit_importance * item_weight * 10
                possible_items.append(
                    {
                        "id": item_name,
                        "recipe": componentData.itemToComponent[item_name],
                        "score": score,
                        "readText": f"Make {item_name} on {unit_name}",
                    }
                )

        possible_items.sort(key=lambda x: x["score"], reverse=True)
        return self.find_optimal_items(self._normalize_components(curr_items), possible_items)

    def find_optimal_items(self, curr_items, possible_items, memo=None):
        curr_items_key = tuple(sorted(curr_items.items()))
        if memo is None:
            memo = dict()
        if curr_items_key in memo:
            return memo[curr_items_key]

        best_score = 0
        items_to_make = []

        for item in possible_items:
            comp1, comp2 = item["recipe"][0], item["recipe"][1]
            can_make = False
            if comp1 == comp2:
                if curr_items.get(comp1, 0) >= 2:
                    can_make = True
            else:
                if curr_items.get(comp1, 0) >= 1 and curr_items.get(comp2, 0) >= 1:
                    can_make = True

            if can_make:
                leftover_items = copy.deepcopy(curr_items)
                leftover_items[comp1] -= 1
                leftover_items[comp2] -= 1
                trial_score, trial_items = self.find_optimal_items(
                    leftover_items, possible_items, memo
                )
                curr_score = trial_score + item["score"]
                if curr_score > best_score:
                    best_score = curr_score
                    items_to_make = [item] + trial_items

        memo[curr_items_key] = (best_score, items_to_make)
        return best_score, items_to_make

    def print_good_boards(self):
        print("-----the good boards-----")
        for i in range(clusterCount):
            profile = self.cluster_profiles.get(i)
            if not profile:
                continue
            trait_cores = profile.trait_core_units or [u for u, _ in profile.core_units]
            splash = profile.splash_units or []
            core_names = ", ".join(_short_unit_name(u) for u in trait_cores[:8])
            splash_names = ", ".join(_short_unit_name(u) for u in splash[:4])
            carries = ", ".join(_short_unit_name(c) for c in profile.carries[:2]) or "n/a"
            if carries == "n/a" and profile.tanks:
                carries = ", ".join(_short_unit_name(c) for c in profile.tanks[:2]) + " (tank)"
            traits = self._display_name_from_traits(profile.core_traits, profile.needs_emblem)
            profile.display_name = traits
            line = (
                f"Board {i + 1}: [{traits}] | "
                f"cores: {core_names} | "
                f"carries: {carries} | req level: {profile.required_level:.1f}"
            )
            if splash_names:
                line += f" | splash: {splash_names}"
            if profile.emblem_items:
                emblems = ", ".join(_short_unit_name(e) for e in profile.emblem_items)
                line += f" | emblems: {emblems}"
            print(line)

    def loadState(self):
        with open(modelDir, "rb") as f:
            state = pickle.load(f)
        self.KMeans = state["model"]
        self.cluster_items_for_units = state["compItems"]
        self.item_weights_per_cluster = state["itemWeights"]
        self.global_item_counts = state.get("globalItemCounts", {})
        self.global_unit_item_totals = state.get("globalUnitItemTotals", {})
        saved_profiles = state.get("clusterProfiles")
        if saved_profiles:
            self.cluster_profiles = saved_profiles
        else:
            self._rebuild_profiles_from_saved()
        print("model loaded")

    def _rebuild_profiles_from_saved(self):
        """Rebuild profiles from saved KMeans + item weights when loading old saves."""
        centroids = self.KMeans.cluster_centers_
        for cluster_idx in range(len(centroids)):
            centroid = centroids[cluster_idx]
            unit_part = centroid[: self.num_units]
            core_units = []
            for uid, val in enumerate(unit_part):
                if val > 0.12 and uid in self.id_to_name:
                    core_units.append((self.id_to_name[uid], float(val)))
            core_units.sort(key=lambda x: x[1], reverse=True)
            cluster_weights = self.item_weights_per_cluster.get(cluster_idx, {})
            item_lift = {}
            for unit_name, items in cluster_weights.items():
                item_lift[unit_name] = {}
                for item_name, weighted_rate in items.items():
                    global_rate = self._global_item_rate(unit_name, item_name)
                    item_lift[unit_name][item_name] = weighted_rate / max(global_rate, 0.01)
            carries, tanks = self._infer_roles(cluster_idx, cluster_weights, core_units)
            self.cluster_profiles[cluster_idx] = ClusterProfile(
                cluster_idx=cluster_idx,
                unit_centroid=unit_part,
                core_units=core_units,
                item_weights=cluster_weights,
                item_lift=item_lift,
                carries=carries,
                tanks=tanks,
                required_level=7.0 + len([u for u, v in core_units if v > 0.2]) * 0.3,
                display_name=f"Cluster {cluster_idx + 1}",
                trait_core_units=[u for u, _ in core_units],
            )

    def saveState(self):
        state = {
            "model": self.KMeans,
            "compItems": self.cluster_items_for_units,
            "itemWeights": self.item_weights_per_cluster,
            "globalItemCounts": self.global_item_counts,
            "globalUnitItemTotals": self.global_unit_item_totals,
            "clusterProfiles": self.cluster_profiles,
        }
        with open(modelDir, "wb") as f:
            pickle.dump(state, f)
        print(f"model saved to {modelDir}")


if __name__ == "__main__":
    analyzer = BoardFinder()
    analyzer.print_good_boards()

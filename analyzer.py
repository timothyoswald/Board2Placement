import copy
import json
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import componentData
import numpy as np
import torch
from sklearn.cluster import KMeans

from state import GameState, cosine_similarity, game_state_from_names, load_id_maps

dataDir = "data/cleaned17.6"
IDsDir = "data/IDs"
modelDir = "boardFinder.pkl"
clusterCount = 15

# Comp matching weights (items weighted highest per plan)
WEIGHT_ITEM_FIT = 0.45
WEIGHT_UNIT_FIT = 0.25
WEIGHT_ECONOMY = 0.20
WEIGHT_TRANSITION = 0.10


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


class BoardFinder:
    def __init__(self, placement_model_path: str = "board2placement.pth"):
        self.ids_file, self.id_to_name, self.id_to_item, self.num_units, _ = load_id_maps(IDsDir)
        self.name_to_id = self.ids_file["unitIDs"]
        self.item_name_to_id = self.ids_file["itemIDs"]
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
            self.placement_model = Board2Placement(predict_distribution=True).to(self.placement_device)
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
        with torch.no_grad():
            expected = self.placement_model.expected_placement(board, context)
        return float(expected.item())

    def _build_projected_state(self, game_state: GameState, cluster_idx: int) -> GameState:
        """Add missing core units at 1-star to simulate capped board."""
        projected = GameState(
            board=game_state.board.clone(),
            stage=game_state.stage,
            level=max(game_state.level, self.cluster_profiles[cluster_idx].required_level),
            gold=game_state.gold,
            components=game_state.components.copy(),
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

        for entry in data:
            if len(entry) == 2:
                board_tensor, placement_tensor = entry
                context = torch.tensor([4.0, 8.0, 0.0])
            else:
                board_tensor, context, placement_tensor = entry
            placement = placement_tensor.item()

            if placement <= 4.0:
                gs = GameState(board=board_tensor)
                vector = gs.unit_vector(self.num_units)
                winning_boards.append(vector)
                winning_boards_tensors.append(board_tensor)
                winning_contexts.append(context)

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

        self._build_cluster_profiles(winning_boards_tensors, winning_contexts)
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

    def _build_cluster_profiles(
        self, boards: List[torch.Tensor], contexts: List[torch.Tensor]
    ):
        cluster_costs = {i: [] for i in range(clusterCount)}
        cluster_levels = {i: [] for i in range(clusterCount)}
        cluster_4cost = {i: [] for i in range(clusterCount)}
        cluster_5cost = {i: [] for i in range(clusterCount)}

        for i, board_tensor in enumerate(boards):
            cidx = self.cluster_labels[i]
            gs = GameState(board=board_tensor)
            cluster_costs[cidx].append(gs.board_cost())
            cost_counts = gs.count_units_by_cost(4)
            cluster_4cost[cidx].append(cost_counts.get(4, 0))
            cluster_5cost[cidx].append(cost_counts.get(5, 0))
            if i < len(contexts):
                cluster_levels[cidx].append(float(contexts[i][1].item()))

        for cluster_idx in range(clusterCount):
            centroid = self.KMeans.cluster_centers_[cluster_idx]
            core_units = []
            for uid, val in enumerate(centroid):
                if val > 0.5 and uid in self.id_to_name:
                    core_units.append((self.id_to_name[uid], float(val)))
            core_units.sort(key=lambda x: x[1], reverse=True)

            item_lift = {}
            cluster_weights = self.item_weights_per_cluster.get(cluster_idx, {})
            for unit_name, items in cluster_weights.items():
                item_lift[unit_name] = {}
                for item_name, weighted_rate in items.items():
                    global_rate = self._global_item_rate(unit_name, item_name)
                    item_lift[unit_name][item_name] = weighted_rate / max(global_rate, 0.01)

            carries, tanks = self._infer_roles(cluster_idx, cluster_weights)

            avg_4 = float(np.mean(cluster_4cost[cluster_idx])) if cluster_4cost[cluster_idx] else 0
            avg_5 = float(np.mean(cluster_5cost[cluster_idx])) if cluster_5cost[cluster_idx] else 0
            required_level = 7.0 + min(avg_5, 2) * 0.5 + min(avg_4, 4) * 0.1

            self.cluster_profiles[cluster_idx] = ClusterProfile(
                cluster_idx=cluster_idx,
                unit_centroid=centroid,
                core_units=core_units,
                item_weights=cluster_weights,
                item_lift=item_lift,
                avg_board_cost=float(np.mean(cluster_costs[cluster_idx])) if cluster_costs[cluster_idx] else 0,
                avg_level=float(np.mean(cluster_levels[cluster_idx])) if cluster_levels[cluster_idx] else 8.0,
                count_4cost=avg_4,
                count_5cost=avg_5,
                required_level=required_level,
                carries=carries,
                tanks=tanks,
            )

    def _infer_roles(
        self, cluster_idx: int, cluster_weights: Dict[str, Dict[str, float]]
    ) -> Tuple[List[str], List[str]]:
        """Units with offensive items -> carries; defensive -> tanks."""
        offensive_items = {
            "TFT_Item_BlueBuff", "TFT_Item_JeweledGauntlet", "TFT_Item_RabadonsDeathcap",
            "TFT_Item_InfinityEdge", "TFT_Item_GuinsoosRageblade", "TFT_Item_Deathblade",
            "TFT_Item_ArchangelsStaff", "TFT_Item_HextechGunblade", "TFT_Item_StatikkShiv",
        }
        defensive_items = {
            "TFT_Item_BrambleVest", "TFT_Item_DragonsClaw", "TFT_Item_WarmogsArmor",
            "TFT_Item_GargoyleStoneplate", "TFT_Item_SunfireCape", "TFT_Item_Redemption",
        }
        carry_scores = {}
        tank_scores = {}
        for unit_name, items in cluster_weights.items():
            carry_scores[unit_name] = sum(w for i, w in items.items() if i in offensive_items)
            tank_scores[unit_name] = sum(w for i, w in items.items() if i in defensive_items)
        carries = sorted(carry_scores, key=carry_scores.get, reverse=True)[:3]
        carries = [u for u in carries if carry_scores[u] > 0]
        tanks = sorted(tank_scores, key=tank_scores.get, reverse=True)[:3]
        tanks = [u for u in tanks if tank_scores[u] > 0]
        return carries, tanks

    # --- Scoring methods ---

    def score_unit_fit(self, game_state: GameState, cluster_idx: int) -> float:
        profile = self.cluster_profiles[cluster_idx]
        board_vec = game_state.unit_vector(self.num_units)
        return cosine_similarity(board_vec, profile.unit_centroid)

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
            # Also score items on core units even if on wrong unit (direction signal)
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
        core_names = [u for u, _ in profile.core_units if u not in owned]
        if not core_names:
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
        missing = [u for u, imp in profile.core_units if u not in owned and imp > 0.5]
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
            current_board, curr_items, self.ids_file,
            stage=stage, level=level, gold=gold,
        )
        best = self.recommend(game_state, top_k=1)[0]
        return (
            best.cluster_idx,
            best.missing_units,
            best.recommended_items,
            best,
        )

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
                possible_items.append({
                    "id": item_name,
                    "recipe": componentData.itemToComponent[item_name],
                    "score": score,
                    "readText": f"Make {item_name} on {unit_name}",
                })

        possible_items.sort(key=lambda x: x["score"], reverse=True)
        return self.find_optimal_items(curr_items, possible_items)

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
            board_names = [u for u, _ in profile.core_units]
            carries = ", ".join(profile.carries[:2]) if profile.carries else "n/a"
            print(
                f"Board {i + 1}: {', '.join(board_names)} | "
                f"carries: {carries} | req level: {profile.required_level:.1f}"
            )

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
            core_units = []
            for uid, val in enumerate(centroid):
                if val > 0.5 and uid in self.id_to_name:
                    core_units.append((self.id_to_name[uid], float(val)))
            core_units.sort(key=lambda x: x[1], reverse=True)
            cluster_weights = self.item_weights_per_cluster.get(cluster_idx, {})
            item_lift = {}
            for unit_name, items in cluster_weights.items():
                item_lift[unit_name] = {}
                for item_name, weighted_rate in items.items():
                    global_rate = self._global_item_rate(unit_name, item_name)
                    item_lift[unit_name][item_name] = weighted_rate / max(global_rate, 0.01)
            carries, tanks = self._infer_roles(cluster_idx, cluster_weights)
            self.cluster_profiles[cluster_idx] = ClusterProfile(
                cluster_idx=cluster_idx,
                unit_centroid=centroid,
                core_units=core_units,
                item_weights=cluster_weights,
                item_lift=item_lift,
                carries=carries,
                tanks=tanks,
                required_level=7.0 + len([u for u, v in core_units if v > 0.7]) * 0.3,
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

    test_board = [
        "TFT16_Briar", "TFT16_Draven", "TFT16_Sion", "TFT16_Illaoi",
        "TFT16_TwistedFate", "TFT16_Nautilus",
    ]
    test_items = {
        "TFT_Item_TearOfTheGoddess": 2,
        "TFT_Item_RecurveBow": 2,
        "TFT_Item_NeedlesslyLargeRod": 3,
        "TFT_Item_NegatronCloak": 2,
        "TFT_Item_ChainVest": 1,
        "TFT_Item_BFSword": 1,
    }
    gs = game_state_from_names(
        test_board, test_items, analyzer.ids_file,
        stage=3.0, level=6.0, gold=20.0,
    )
    print(f"\ncompleting {test_board} at stage 3, level 6, 20g...")
    recs = analyzer.recommend(gs, top_k=3)
    for rec in recs:
        print(
            f"\nBoard #{rec.cluster_idx + 1} | score={rec.total_score:.3f} | "
            f"feasibility={rec.feasibility}"
        )
        print(f"  unit_fit={rec.unit_fit:.2f} item_fit={rec.item_fit:.2f} "
              f"economy={rec.economy_fit:.2f} transition={rec.transition_cost:.2f}")
        print(f"  missing units: {rec.missing_units[:5]}")
        if rec.expected_placement is not None:
            print(f"  expected placement now: {rec.expected_placement:.2f}")
        if rec.projected_placement is not None:
            print(f"  projected placement at cap: {rec.projected_placement:.2f}")
        print("  recommended items:")
        for d in rec.recommended_items:
            print(f"    {d['readText']}")

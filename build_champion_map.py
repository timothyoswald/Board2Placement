"""Build champion→trait map and trait breakpoints from Community Dragon."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Set

import requests

from settings import CDRAGON_TFT_URL, CHAMPION_MAP_PATH, TFT_SET


def _set_key(data: dict) -> str | None:
    sets = data.get("sets") or data.get("setData")
    if isinstance(sets, dict):
        if str(TFT_SET) in sets:
            return str(TFT_SET)
        for k in sets:
            if str(k) == str(TFT_SET):
                return str(k)
    return None


def _extract_breakpoints(trait_obj: dict) -> List[int]:
    bps: List[int] = []
    effects = trait_obj.get("effects") or []
    for eff in effects:
        mn = eff.get("minUnits")
        if mn is not None:
            bps.append(int(mn))
    return sorted(set(bps))


def _normalize_key(name: str) -> str:
    """Loose key for matching display names to API ids."""
    if not name:
        return ""
    n = name.strip().lower()
    n = re.sub(r"[^a-z0-9]+", "", n)
    return n


def _resolve_trait_api(
    raw: str,
    display_to_api: Dict[str, str],
    api_names: Set[str],
) -> Optional[str]:
    if not raw:
        return None
    if raw in api_names:
        return raw
    key = _normalize_key(raw)
    if key in display_to_api:
        return display_to_api[key]
    # Try stripping common prefixes from raw if it looks like an api name
    for prefix in ("DA_18_", "DA_", "TFT18_", "TFT_"):
        if raw.startswith(prefix):
            stripped = raw[len(prefix) :]
            k2 = _normalize_key(stripped)
            if k2 in display_to_api:
                return display_to_api[k2]
    return None


def _emblem_to_trait(item_api: str, trait_apis: Set[str], display_to_api: Dict[str, str]) -> Optional[str]:
    """Map DA_18_EmblemFae → DA_18_Fae (best-effort)."""
    m = re.match(r"DA_18_Emblem(.+)$", item_api)
    if not m:
        return None
    suffix = m.group(1)
    # Skip augment-only emblems
    if "Augment" in suffix:
        return None
    candidates = [
        f"DA_18_{suffix}",
        f"DA_{suffix}18",
        f"DA_{suffix}",
    ]
    for c in candidates:
        if c in trait_apis:
            return c
    key = _normalize_key(suffix)
    if key in display_to_api:
        return display_to_api[key]
    # Slayer emblem may map to Ravager display, etc. — leave unmapped if unknown
    return None


def build_champion_trait_map(data: dict) -> Dict[str, Any]:
    """
    Returns champion entries keyed by character apiName, plus _meta:
      trait_breakpoints, trait_display_names, unique_traits, emblem_to_trait, champions
    Champion traits are stored as Riot/CDragon API ids (e.g. DA_18_Solar).
    """
    champ_map: Dict[str, Any] = {}
    trait_breakpoints: Dict[str, List[int]] = {}
    trait_display_names: Dict[str, str] = {}
    display_to_api: Dict[str, str] = {}
    champion_ids: Set[str] = set()
    trait_champions: Dict[str, Set[str]] = {}

    set_key = _set_key(data)
    set_blob = None
    if set_key and isinstance(data.get("sets"), dict):
        set_blob = data["sets"][set_key]
    elif isinstance(data.get("setData"), list):
        for entry in data["setData"]:
            if entry.get("number") == TFT_SET or str(entry.get("mutator", "")).endswith(
                str(TFT_SET)
            ):
                set_blob = entry
                break

    champions = []
    traits = []
    if set_blob:
        champions = set_blob.get("champions") or []
        traits = set_blob.get("traits") or []
    else:
        champions = data.get("champions") or []
        traits = data.get("traits") or []

    for trait in traits:
        api = trait.get("apiName") or ""
        if not api:
            continue
        display = (trait.get("name") or api).strip()
        trait_breakpoints[api] = _extract_breakpoints(trait)
        trait_display_names[api] = display
        display_to_api[_normalize_key(display)] = api
        display_to_api[_normalize_key(api)] = api
        # Also index short forms
        for prefix in ("DA_18_", "DA_", "TFT18_", "TFT_"):
            if api.startswith(prefix):
                display_to_api[_normalize_key(api[len(prefix) :])] = api

    api_names = set(trait_breakpoints.keys())

    for champ in champions:
        api = champ.get("apiName") or champ.get("characterName")
        if not api:
            continue
        cost = champ.get("cost")
        if cost is None:
            cost = champ.get("tier")
        if cost is None:
            continue

        raw_traits = []
        for t in champ.get("traits") or []:
            if isinstance(t, str):
                raw_traits.append(t)
            elif isinstance(t, dict):
                raw_traits.append(t.get("apiName") or t.get("name") or "")

        trait_apis: List[str] = []
        for raw in raw_traits:
            resolved = _resolve_trait_api(raw, display_to_api, api_names)
            if resolved and resolved not in trait_apis:
                trait_apis.append(resolved)

        bps = {t: trait_breakpoints.get(t, []) for t in trait_apis}
        champ_map[api] = {
            "traits": trait_apis,
            "cost": int(cost),
            "breakpoints": bps,
        }
        champion_ids.add(api)
        for t in trait_apis:
            trait_champions.setdefault(t, set()).add(api)

    # Unique: breakpoint list is [1] only, or exactly one champion owns the trait
    unique_traits: List[str] = []
    for api, bps in trait_breakpoints.items():
        owners = trait_champions.get(api, set())
        if bps == [1] or len(owners) == 1:
            unique_traits.append(api)
    unique_traits = sorted(set(unique_traits))

    # Emblem item → trait from top-level items list
    emblem_to_trait: Dict[str, str] = {}
    for it in data.get("items") or []:
        item_api = it.get("apiName") or ""
        if "Emblem" not in item_api or not item_api.startswith("DA_18_"):
            continue
        mapped = _emblem_to_trait(item_api, api_names, display_to_api)
        if mapped:
            emblem_to_trait[item_api] = mapped

    # Known Set 18 emblem aliases that regex may miss
    manual_emblems = {
        "DA_18_EmblemSlayer": "DA_18_Slayer",  # display sometimes Ravager
    }
    for item_api, trait_api in manual_emblems.items():
        if trait_api in api_names:
            emblem_to_trait[item_api] = trait_api

    champ_map["_meta"] = {
        "set": TFT_SET,
        "champions": sorted(champion_ids),
        "trait_breakpoints": trait_breakpoints,
        "trait_display_names": trait_display_names,
        "unique_traits": unique_traits,
        "emblem_to_trait": emblem_to_trait,
        "trait_champions": {k: sorted(v) for k, v in trait_champions.items()},
    }
    return champ_map


def save_champion_trait_map(path: str = CHAMPION_MAP_PATH) -> dict:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"fetching Community Dragon data from {CDRAGON_TFT_URL}")
    data = requests.get(CDRAGON_TFT_URL, timeout=120).json()
    champ_map = build_champion_trait_map(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(champ_map, f, indent=2)
    meta = champ_map.get("_meta", {})
    n = len(meta.get("champions", []))
    print(
        f"wrote {n} champions, {len(meta.get('trait_display_names', {}))} traits, "
        f"{len(meta.get('unique_traits', []))} uniques, "
        f"{len(meta.get('emblem_to_trait', {}))} emblems to {path}"
    )
    return champ_map


def champion_id_set(champ_map: dict | None = None) -> Set[str]:
    if champ_map is None:
        if not os.path.exists(CHAMPION_MAP_PATH):
            return set()
        with open(CHAMPION_MAP_PATH, "r", encoding="utf-8") as f:
            champ_map = json.load(f)
    meta = champ_map.get("_meta", {})
    return set(meta.get("champions", []))


if __name__ == "__main__":
    save_champion_trait_map()

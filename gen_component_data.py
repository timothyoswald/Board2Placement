"""Generate componentData.py from Community Dragon TFT data."""
import requests

from settings import CDRAGON_TFT_URL, TFT_SET

# Set 18 (Unreal) uses DA_Component_*; keep TFT_Item_* aliases for older tooling.
DA_BASE_COMPONENTS = [
    "DA_Component_BFSword",
    "DA_Component_RecurveBow",
    "DA_Component_NeedlesslyLargeRod",
    "DA_Component_TearOfTheGoddess",
    "DA_Component_ChainVest",
    "DA_Component_NegatronCloak",
    "DA_Component_GiantsBelt",
    "DA_Component_SparringGloves",
    "DA_Component_Spatula",
    "DA_Component_FryingPan",
]
TFT_BASE_COMPONENTS = [
    "TFT_Item_BFSword",
    "TFT_Item_RecurveBow",
    "TFT_Item_NeedlesslyLargeRod",
    "TFT_Item_TearOfTheGoddess",
    "TFT_Item_ChainVest",
    "TFT_Item_NegatronCloak",
    "TFT_Item_GiantsBelt",
    "TFT_Item_SparringGloves",
    "TFT_Item_Spatula",
    "TFT_Item_FryingPan",
]
BASE_COMPONENTS = DA_BASE_COMPONENTS + TFT_BASE_COMPONENTS
BASE_SET = set(BASE_COMPONENTS)
SKIP_SUBSTR = ("Corrupted", "Grant", "Empty", "Random", "Augment")


def build_item_map(data: dict) -> dict:
    item_to_component = {}
    for it in data["items"]:
        api = it.get("apiName", "")
        comp = it.get("composition", [])
        if len(comp) != 2 or not all(c in BASE_SET for c in comp):
            continue
        if any(s in api for s in SKIP_SUBSTR):
            continue
        # Keep Set 18 DA craftables + classic TFT_Item_ recipes
        if (
            api.startswith("DA_")
            or api.startswith("TFT_Item_")
            or api.startswith(f"TFT{TFT_SET}_Item_")
        ):
            item_to_component[api] = comp
    return item_to_component


def write_component_data(item_to_component: dict, path: str = "componentData.py"):
    lines = [
        f"# Auto-generated from Community Dragon TFT data for Set {TFT_SET}.",
        "# Riot API names may differ from in-game display names.",
        "",
        "components = {",
    ]
    for c in BASE_COMPONENTS:
        lines.append(f'    "{c}",')
    lines.append("}")
    lines.append("")
    # Alias map so callers can pass either naming scheme
    lines.append("componentAliases = {")
    for da, tft in zip(DA_BASE_COMPONENTS, TFT_BASE_COMPONENTS):
        lines.append(f'    "{tft}": "{da}",')
        lines.append(f'    "{da}": "{da}",')
    lines.append("}")
    lines.append("")
    lines.append("itemToComponent = {")

    da_standard = sorted(
        k
        for k in item_to_component
        if k.startswith("DA_") and "Emblem" not in k and not k.startswith("DA_18_")
    )
    da_emblems = sorted(k for k in item_to_component if k.startswith("DA_18_") or "Emblem" in k)
    tft_standard = sorted(k for k in item_to_component if k.startswith("TFT_Item_"))

    sections = [
        ("# --- SET 18 STANDARD CRAFTABLE ITEMS (DA_) ---", da_standard),
        (f"# --- SET {TFT_SET} EMBLEMS ---", da_emblems),
        ("# --- LEGACY TFT_Item_ RECIPES ---", tft_standard),
    ]
    for header, keys in sections:
        if not keys:
            continue
        lines.append(f"    {header}")
        for key in keys:
            c1, c2 = item_to_component[key]
            lines.append(f'    "{key}": ["{c1}", "{c2}"],')

    lines.append("}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {len(item_to_component)} items to {path}")


if __name__ == "__main__":
    data = requests.get(CDRAGON_TFT_URL, timeout=120).json()
    write_component_data(build_item_map(data))

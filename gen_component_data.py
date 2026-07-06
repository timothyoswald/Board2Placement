"""Generate componentData.py from Community Dragon TFT data."""
import requests

BASE_COMPONENTS = [
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
BASE_SET = set(BASE_COMPONENTS)
SKIP_SUBSTR = ("Corrupted", "Grant", "Empty", "Random")


def build_item_map(data: dict) -> dict:
    item_to_component = {}
    for it in data["items"]:
        api = it.get("apiName", "")
        comp = it.get("composition", [])
        if len(comp) != 2 or not all(c in BASE_SET for c in comp):
            continue
        if any(s in api for s in SKIP_SUBSTR):
            continue
        if api.startswith("TFT_Item_") or api.startswith("TFT17_Item_"):
            item_to_component[api] = comp
    return item_to_component


def write_component_data(item_to_component: dict, path: str = "componentData.py"):
    lines = [
        "# Auto-generated from Community Dragon TFT data for Set 17.",
        "# Riot API names may differ from in-game display names.",
        "",
        "components = {",
    ]
    for c in BASE_COMPONENTS:
        lines.append(f'    "{c}",')
    lines.append("}")
    lines.append("")
    lines.append("itemToComponent = {")

    standard = sorted(k for k in item_to_component if k.startswith("TFT_Item_"))
    set17 = sorted(k for k in item_to_component if k.startswith("TFT17_"))

    sections = [
        ("# --- STANDARD CRAFTABLE ITEMS ---", standard),
        ("# --- SET 17 EMBLEMS (API trait names) ---", set17),
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
    data = requests.get(
        "https://raw.communitydragon.org/latest/cdragon/tft/en_us.json", timeout=120
    ).json()
    write_component_data(build_item_map(data))

"""Committed set/patch configuration. Secrets stay in gitignored config.py."""
import os

TFT_SET = 18
PATCH = "18.1"
# After Unreal Engine migration, TFT patch prefixes often match the TFT patch
# number. Accept any Set 18 ranked match when this is True (safe until 18.2).
ACCEPT_ANY_VERSION_FOR_SET = True
FALLBACK_PATCHES: list[str] = []

DATA_DIR = f"data/patch{PATCH}"
CLEANED_DIR = f"data/cleaned{PATCH}"
CLEANED_PARTIAL_DIR = f"data/cleaned{PATCH}_partial"
IDS_PATH = "data/IDs"
SCRAPER_STATE_FILE = f"scraperState_patch{PATCH}.json"
BOARD_FINDER_PATH = "boardFinder.pkl"
PLACEMENT_MODEL_PATH = "board2placement.pth"

# Unreal / Set 18 uses DA_* item and champion IDs; keep TFT_Item_ for legacy recipes.
ITEM_API_PREFIXES = ("TFT_Item_", "TFT18_Item_", "DA_18_", "DA_Item_")
COMPONENT_PREFIXES = ("TFT_Item_", "DA_Component_")
TRAIT_WEIGHT_IN_CLUSTER = 0.5
CLUSTER_COUNT = 15

CDRAGON_TFT_URL = "https://raw.communitydragon.org/latest/cdragon/tft/en_us.json"
CHAMPION_MAP_PATH = "data/championTraits.json"

MATCH_COUNT = int(os.environ.get("B2P_MATCH_COUNT", "2000"))

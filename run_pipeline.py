"""End-to-end pipeline: scrape, preprocess, train archetypes, train placement model, evaluate."""
import os
import subprocess
import sys

from settings import (
    BOARD_FINDER_PATH,
    DATA_DIR,
    MATCH_COUNT,
    PLACEMENT_MODEL_PATH,
)


def run(cmd: list[str], env: dict | None = None):
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.run(cmd, check=True, env=merged)


def count_matches() -> int:
    if not os.path.isdir(DATA_DIR):
        return 0
    return len([f for f in os.listdir(DATA_DIR) if f.endswith(".json")])


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    target_matches = MATCH_COUNT
    existing = count_matches()
    if existing < target_matches:
        run(
            [sys.executable, "dataScraper.py"],
            env={"B2P_MATCH_COUNT": str(target_matches)},
        )
    else:
        print(f"using existing {existing} scraped matches", flush=True)

    run([sys.executable, "build_champion_map.py"])
    run([sys.executable, "gen_component_data.py"])

    for stale in (BOARD_FINDER_PATH, PLACEMENT_MODEL_PATH):
        if os.path.exists(stale):
            os.remove(stale)
            print(f"removed stale {stale}", flush=True)

    run([sys.executable, "filterData.py"])
    run([sys.executable, "-c", "from analyzer import BoardFinder; BoardFinder().saveState()"])
    run([sys.executable, "train.py"])
    run([sys.executable, "evaluate.py"])
    run([sys.executable, "-c", "from analyzer import BoardFinder; BoardFinder().print_good_boards()"])


if __name__ == "__main__":
    main()

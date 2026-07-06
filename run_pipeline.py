"""End-to-end pipeline: scrape, preprocess, train archetypes, train placement model, evaluate."""
import os
import subprocess
import sys
import time


def run(cmd: list[str], env: dict | None = None):
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.run(cmd, check=True, env=merged)


def count_matches() -> int:
    data_dir = "data/patch17.6"
    if not os.path.isdir(data_dir):
        return 0
    return len([f for f in os.listdir(data_dir) if f.endswith(".json")])


def main():
    os.makedirs("data/patch17.6", exist_ok=True)

    target_matches = int(os.environ.get("B2P_MATCH_COUNT", "2000"))
    existing = count_matches()
    if existing < target_matches:
        run(
            [sys.executable, "dataScraper.py"],
            env={"B2P_MATCH_COUNT": str(target_matches)},
        )
    else:
        print(f"using existing {existing} scraped matches", flush=True)

    for stale in ("boardFinder.pkl", "board2placement.pth"):
        if os.path.exists(stale):
            os.remove(stale)
            print(f"removed stale {stale}", flush=True)

    run([sys.executable, "filterData.py"])
    run([sys.executable, "-c", "from analyzer import BoardFinder; BoardFinder().saveState()"])
    run([sys.executable, "train.py"])
    run([sys.executable, "evaluate.py"])


if __name__ == "__main__":
    main()

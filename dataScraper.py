import requests
import json
import time
import os
import sys
from collections import deque
from config import API_KEY, REGION_ROUTING, PLATFORM_ROUTING

HEADER = {"X-Riot-Token": API_KEY}
matchCount = int(os.environ.get("B2P_MATCH_COUNT", 10000))
saveDir = "data/patch17.6"
desiredSet = 17  # note that this only scrapes recent match history
desiredPatch = "17.6"  # LoL client patch prefix (not TFT set number)
# Set 17 currently runs on LoL patch 16.13.x until 17.6 is live on the API.
fallbackPatches = ["16.13", "16.12"]
stateFile = "scraperState_patch17.6.json"


class LargeScraper:
    def __init__(self, apiKey, region, platform, saveDir):
        self.apiKey = apiKey
        self.region = region
        self.platform = platform
        self.saveDir = saveDir

        self.session = requests.Session()
        self.playerQueue = deque()
        self.seenPlayers = set()
        self.seenMatches = set()
        self.checkedMatches = 0
        self.patchCandidates = [os.environ.get("B2P_PATCH", desiredPatch)]
        for patch in fallbackPatches:
            if patch not in self.patchCandidates:
                self.patchCandidates.append(patch)

        self.loadState()

        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

    def makeRequest(self, url):
        while True:
            try:
                response = self.session.get(url, headers=HEADER)

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    waitTime = int(response.headers.get("Retry-After", 5))
                    print(f"rate limit hit: waiting {waitTime} seconds", flush=True)
                    time.sleep(waitTime)
                    continue
                elif response.status_code == 403:
                    print("api key expired", flush=True)
                    sys.exit(1)
                elif response.status_code == 404:
                    print("url not found", flush=True)
                    return None
                else:
                    print(f"response status code: {response.status_code}", flush=True)
                    return None
            except Exception as e:
                print(f"request failed: {e}", flush=True)
                time.sleep(5)

    def getPlayers(self):
        url = f"https://{self.platform}.api.riotgames.com/tft/league/v1/challenger"
        data = self.makeRequest(url)

        if not data:
            print("failed to get challenger players", flush=True)
            return None

        entries = data.get("entries", [])
        for entry in entries[:25]:
            if "puuid" in entry:
                self.queuePlayer(entry["puuid"])

    def queuePlayer(self, puuid):
        if puuid not in self.seenPlayers and puuid not in self.playerQueue:
            self.playerQueue.append(puuid)

    def _patch_matches(self, version_short: str) -> bool:
        return any(version_short.startswith(patch) for patch in self.patchCandidates)

    def _maybe_fallback_patch(self):
        if len(self.seenMatches) > 0 or self.checkedMatches < 150:
            return
        if len(self.patchCandidates) <= 1:
            return
        dropped = self.patchCandidates.pop(0)
        print(
            f"no matches saved for LoL patch prefix '{dropped}'; "
            f"falling back to '{self.patchCandidates[0]}'",
            flush=True,
        )

    def scrape(self):
        print(
            f"starting scrape | set={desiredSet} | patch prefixes={self.patchCandidates} | "
            f"target={matchCount}",
            flush=True,
        )
        while len(self.seenMatches) < matchCount and self.playerQueue:
            currentPlayer = self.playerQueue.popleft()
            self.seenPlayers.add(currentPlayer)

            matchIDs_url = (
                f"https://{self.region}.api.riotgames.com/tft/match/v1/matches/"
                f"by-puuid/{currentPlayer}/ids?count=20"
            )
            matchIDs = self.makeRequest(matchIDs_url)
            if not matchIDs:
                continue

            for matchID in matchIDs:
                if matchID in self.seenMatches:
                    continue

                match_url = (
                    f"https://{self.region}.api.riotgames.com/tft/match/v1/matches/{matchID}"
                )
                matchData = self.makeRequest(match_url)
                if not matchData or "info" not in matchData:
                    continue

                info = matchData["info"]
                if info["tft_set_number"] != desiredSet or info["queue_id"] != 1100:
                    continue

                version = info["game_version"]
                versionShort = version.split(" ")[2]
                self.checkedMatches += 1
                self._maybe_fallback_patch()

                if self._patch_matches(versionShort):
                    self.saveMatch(matchID, matchData)
                    self.snowballPlayers(matchData)

            if len(self.seenPlayers) % 10 == 0:
                self.saveState()

        print(f"done! scraped {len(self.seenMatches)} matches!", flush=True)

    def saveMatch(self, matchID, matchData):
        path = os.path.join(self.saveDir, f"{matchID}.json")
        with open(path, "w") as f:
            json.dump(matchData, f)
        self.seenMatches.add(matchID)
        print(
            f"saved match {matchID}. in total {len(self.seenMatches)} matches saved",
            flush=True,
        )

    def snowballPlayers(self, matchData):
        participants = matchData["metadata"]["participants"]
        for puuid in participants:
            self.queuePlayer(puuid)

    def saveState(self):
        with open(stateFile, "w") as f:
            json.dump(
                {
                    "seenMatches": list(self.seenMatches),
                    "seenPlayers": list(self.seenPlayers),
                    "playerQueue": list(self.playerQueue),
                },
                f,
            )
        print("state saved", flush=True)

    def loadState(self):
        if os.path.exists(stateFile):
            with open(stateFile, "r") as f:
                data = json.load(f)
                self.seenMatches = set(data["seenMatches"])
                self.seenPlayers = set(data["seenPlayers"])
                self.playerQueue = deque(data["playerQueue"])
            print(f"state resumed with {len(self.seenMatches)} matches", flush=True)


def main():
    scraper = LargeScraper(API_KEY, REGION_ROUTING, PLATFORM_ROUTING, saveDir)
    if not scraper.playerQueue:
        scraper.getPlayers()
    scraper.scrape()


if __name__ == "__main__":
    main()

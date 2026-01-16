import requests
import json
import time
import os
from collections import deque
from config import API_KEY, REGION_ROUTING, PLATFORM_ROUTING

HEADER = {"X-Riot-Token": API_KEY}
matchCount = 10000 # adjust how many matches you want here
saveDir = "data/patch16.2" # change to your directory
desiredSet = 16 # note that this only scrapes recent match history
desiredPatch = "16.1" # note that this is based on LoL patches and the season

class LargeScraper():
    def __init__(self, apiKey, region, platform, saveDir):
        self.apiKey = apiKey
        self.region = region
        self.platform = platform
        self.saveDir = saveDir

        self.session = requests.Session()
        self.playerQueue = deque()
        self.seenPlayers = set()
        self.seenMatches = set()

        # load existing state if there is one
        self.loadState()

        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

    def makeRequest(self, url):
        while True:
            try:
                response = self.session.get(url, headers = HEADER)

                if response.status_code == 200: # successful request
                    return response.json()
                elif response.status_code == 429: # too many requests recently, rate limit hit
                    waitTime = int(response.headers.get("Retry-After", 5))
                    print(f"rate limit hit: waiting {waitTime} seconds")
                    time.sleep(waitTime)
                    continue
                elif response.status_code == 403: # API key expired
                    print("api key expired")
                    exit() # stop running entirely
                elif response.status_code == 404: # url not found
                    print("url not found")
                    return None
                else:
                    print(f"response status code: {response.status_code}")
                    return None
            except Exception as e:
                print(f"request failed: {e}")
                time.sleep(5)

    def getPlayers(self):
        url = f"https://{self.platform}.api.riotgames.com/tft/league/v1/challenger"
        data = self.makeRequest(url)

        if not data:
            print("failed to get challenger players")
            return None
        
        entries = data.get("entries", [])

        for entry in entries[:25]:
            if "puuid" in entry:
                self.queuePlayer(entry["puuid"])
    
    def queuePlayer(self, puuid):
        if puuid not in self.seenPlayers and puuid not in self.playerQueue:
            self.playerQueue.append(puuid)
    
    def scrape(self):
        print("starting to scrape")
        while len(self.seenMatches) < matchCount and self.playerQueue:
            currentPlayer = self.playerQueue.popleft()
            self.seenPlayers.add(currentPlayer)

            matchIDs_url = f"https://{self.region}.api.riotgames.com/tft/match/v1/matches/by-puuid/{currentPlayer}/ids?count=20"
            matchIDs = self.makeRequest(matchIDs_url)

            if not matchIDs: continue

            for matchID in matchIDs:
                if matchID in self.seenMatches:
                    continue

                match_url = f"https://{self.region}.api.riotgames.com/tft/match/v1/matches/{matchID}"
                matchData = self.makeRequest(match_url)

                if matchData:
                    # make sure we are only saving ranked games
                    # from current set
                    if (matchData["info"]["tft_set_number"] != desiredSet or
                        matchData["info"]["queue_id"] != 1100):
                        continue
                    version = matchData["info"]["game_version"]
                    versionShort = version.split(' ')[2]
                    # also only save games from current patch
                    if versionShort.startswith(desiredPatch):
                        self.saveMatch(matchID, matchData)
                        # grab other players in the match for more data
                        self.snowballPlayers(matchData)
                
            if len(self.seenPlayers) % 10 == 0:
                self.saveState()

        print(f"done! scraped {len(self.seenMatches)} matches!")
        
    def saveMatch(self, matchID, matchData):
        path = os.path.join(self.saveDir, f"{matchID}.json")
        with open(path, "w") as f:
            json.dump(matchData, f)
        self.seenMatches.add(matchID)
        print(f"saved match {matchID}. in total {len(self.seenMatches)} matches saved")
    
    def snowballPlayers(self, matchData):
        participants = matchData["metadata"]["participants"]
        for puuid in participants:
            self.queuePlayer(puuid)
    
    def saveState(self):
        with open("scraperState.json", "w") as f:
            json.dump({
                "seenMatches": list(self.seenMatches),
                "seenPlayers": list(self.seenPlayers),
                "playerQueue": list(self.playerQueue)
            }, f)
        print("state saved")
    
    def loadState(self):
        if os.path.exists("scraperState.json"):
            with open("scraperState.json", "r") as f:
                data = json.load(f)
                self.seenMatches = set(data["seenMatches"])
                self.seenPlayers = set(data["seenPlayers"])
                self.playerQueue = deque(data["playerQueue"])
            print(f"state resumed with {len(self.seenMatches)} matches")

scraper = LargeScraper(API_KEY, REGION_ROUTING, PLATFORM_ROUTING, saveDir)
if not scraper.playerQueue:
    scraper.getPlayers()
scraper.scrape()
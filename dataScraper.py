import requests
import json
import time
import os
from config import API_KEY, REGION_ROUTING, PLATFORM_ROUTING, SLEEP_TIME

# Standard headers for Riot API
HEADERS = {
    "X-Riot-Token": API_KEY
}

def make_request(url):
    """
    Sends a request to Riot API. 
    Handles 429 (Rate Limit) errors by waiting automatically.
    """
    while True:
        try:
            response = requests.get(url, headers=HEADERS)
            
            if response.status_code == 200:
                return response.json()
            
            elif response.status_code == 429:
                # We hit the limit. Wait a bit longer than usual.
                print("⚠️ Rate limit hit! Sleeping for 10 seconds...")
                time.sleep(10)
            
            elif response.status_code == 403:
                print("❌ API Key Expired or Forbidden! Check config.py")
                return None
                
            else:
                print(f"❌ Error {response.status_code}: {url}")
                return None
                
        except Exception as e:
            print(f"Network Error: {e}")
            return None

def get_challenger_players():
    """Step 1: Get list of top players to use as 'seeds'"""
    print("Fetching Challenger League...")
    url = f"https://{PLATFORM_ROUTING}.api.riotgames.com/tft/league/v1/challenger"
    data = make_request(url)
    if not data: return []
    return data['entries'] # Returns a list of players

def get_match_ids(puuid, count=10):
    """Step 3: Get last 10 match IDs for a player"""
    url = f"https://{REGION_ROUTING}.api.riotgames.com/tft/match/v1/matches/by-puuid/{puuid}/ids?start=0&count={count}"
    data = make_request(url)
    time.sleep(SLEEP_TIME)
    return data if data else []

def save_match_json(match_id):
    """Step 4: Download and save the match details"""
    # Check if we already downloaded it to avoid duplicates
    save_path = f"data/rawMatches/{match_id}.json"
    if os.path.exists(save_path):
        print(f"Skipping {match_id} (Already Exists)")
        return

    url = f"https://{REGION_ROUTING}.api.riotgames.com/tft/match/v1/matches/{match_id}"
    match_data = make_request(url)
    
    if match_data:
        # Save to file
        with open(save_path, 'w') as f:
            json.dump(match_data, f)
        print(f"✅ Saved {match_id}")
    
    time.sleep(SLEEP_TIME)

# --- MAIN EXECUTION LOOP ---
if __name__ == "__main__":
    # Ensure data folder exists
    os.makedirs("data/rawMatches", exist_ok=True)

    print("--- STARTING DATA COLLECTION ---")
    
    # 1. Get Seed Players
    challengers = get_challenger_players()
    # Let's grab the top 25 players
    targets = challengers[:25] 
    
    print(f"Found {len(targets)} Challenger players. Starting harvest...")

    # 2. Loop through players and find matches
    unique_match_ids = set()
    
    for i, player in enumerate(targets):
        print(f"Processing Player {i+1}...")
        
        # Use the PUUID directly provided by the Challenger list
        puuid = player.get('puuid') 
        
        if puuid:
            matches = get_match_ids(puuid)
            unique_match_ids.update(matches)

    print(f"\n--- DOWNLOADING {len(unique_match_ids)} MATCHES ---")
    
    # 3. Download all unique matches
    for i, match_id in enumerate(unique_match_ids):
        print(f"[{i+1}/{len(unique_match_ids)}] ", end="")
        save_match_json(match_id)

    print("\n--- DONE! ---")
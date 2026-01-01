import torch
import json
import difflib # For "Fuzzy Matching" (e.g., typing 'jin' -> 'Jinx')
from model import Board2Placement

# --- CONFIG ---
MODEL_PATH = "board2placement.pth" # Make sure this matches your saved file
VOCAB_FILE = "data/vocab.json"

class TFTAssistant:
    def __init__(self):
        print("Loading Brain...")
        
        # 1. Load Vocab
        with open(VOCAB_FILE, 'r') as f:
            self.vocab = json.load(f)
            
        # Create reverse lookups (Name -> ID)
        self.name_to_id = self.vocab['unit_to_idx']
        self.item_to_id = self.vocab['item_to_idx']
        
        # 2. Load Model
        self.model = Board2Placement()
        try:
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        except:
            # Fallback if you didn't save "best" yet
            self.model.load_state_dict(torch.load("tft_model.pth", map_location='cpu'))
            
        self.model.eval()
        print("✅ System Ready!\n")
        
        # State of the current board being built
        self.current_board = [] # List of tuples: (UnitID, Star, [Items])

    def find_closest(self, user_input, collection):
        # Helper to find "Kai'sa" when user types "kaisa"
        matches = difflib.get_close_matches(user_input, collection.keys(), n=1, cutoff=0.4)
        return matches[0] if matches else None

    def add_unit(self):
        raw_name = input("Unit Name: ").strip().lower()
        name = self.find_closest(raw_name, self.name_to_id)
        
        if not name:
            print("❌ Unit not found.")
            return

        try:
            star = int(input(f"Star Level for {name} (1-3): "))
        except:
            star = 1
            
        # Items (Optional)
        items = []
        while len(items) < 3:
            i_name = input(f"Item {len(items)+1} (or Enter to skip): ").strip().lower()
            if not i_name: break
            
            found_item = self.find_closest(i_name, self.item_to_id)
            if found_item:
                items.append(self.item_to_id[found_item])
                print(f"   + Added {found_item}")
            else:
                print("   ❌ Item not found.")

        # Save to board
        u_id = self.name_to_id[name]
        self.current_board.append({
            "id": u_id,
            "star": star,
            "items": items
        })
        print(f"✅ Added {star}-Star {name} to board.\n")

    def predict(self):
        if not self.current_board:
            print("⚠️ Board is empty!")
            return

        # 1. Prepare Tensor
        # We need to fill a [1, 14, 5] tensor with our board data
        input_tensor = torch.zeros(1, 14, 5, dtype=torch.long)
        
        # Fill in the slots
        for i, unit in enumerate(self.current_board):
            if i >= 14: break # Max 14 units
            
            # Column 0: Unit ID
            input_tensor[0, i, 0] = unit['id']
            # Column 1: Star Level
            input_tensor[0, i, 1] = unit['star']
            # Columns 2-4: Items
            for j, item_id in enumerate(unit['items']):
                input_tensor[0, i, 2+j] = item_id
        
        # 2. Ask the AI
        with torch.no_grad():
            prediction = self.model(input_tensor).item()
            
        print("\n" + "="*30)
        print(f"🔮 PREDICTED PLACEMENT: {prediction:.2f}")
        print("="*30 + "\n")

    def run(self):
        while True:
            print(f"Current Board: {len(self.current_board)} Units")
            cmd = input("[A]dd Unit | [C]lear | [P]redict | [Q]uit: ").lower()
            
            if cmd == 'a': self.add_unit()
            elif cmd == 'c': self.current_board = []
            elif cmd == 'p': self.predict()
            elif cmd == 'q': break

if __name__ == "__main__":
    app = TFTAssistant()
    app.run()
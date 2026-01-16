# Board2Placement

This project began as an exploration of **applied machine learning in a domain I was already deeply familiar with**.

I figured that Teamfight Tactics is an especially interesting environment for experimentation because:

* the state space is complex but **finite and well-structured**,
* the number of meaningful features (units, items, traits, positions) is relatively small,
* and outcomes are clearly measurable (placement 1–8).

My guiding belief was that TFT is constrained enough that a machine should be able to find patterns in the decision space and "solve" the game. This was further reinforced by the fact that Riot Games (creator of Teamfight Tactics) removed Augment Data from their public API.

Rather than building a heuristic-heavy system like I tried previously, the goal was to see how far I could get by:

* learning structure directly from data,
* enumerating valid decisions algorithmically,
* and using machine learning to score and compare outcomes.

---

## 🧠 Core Architecture

### 1. Data Pipeline — *The Foundation*

#### Automated Scraper (`dataScraper.py`)

A robust crawler that interfaces directly with the **Riot Games API**.

**Key features:**

* Fetches **Ranked-only** matches
* Enforces **Current Set** and **Current Patch** consistency to prevent stale data
* **Challenger-first seeding** (Challenger & Grandmaster players)
* Built-in **rate limit handling** with automatic sleep logic

This ensures the model learns exclusively from **top-level competitive play**.

---

#### Data Preprocessing (`filterData.py`)

Transforms raw match JSON into ML-ready datasets.

**Responsibilities:**

* Converts boards into **PyTorch-compatible tensors**
* **Normalizes unit and item names** into fixed integer IDs
* Sanitizes API inconsistencies:

  * Clamps illegal item counts (e.g., >3 items per unit)
  * Handles missing or malformed entries safely

This step guarantees clean, consistent training data.

---

### 2. Meta Discovery — *The Brain*

#### Archetype Learning (`analyzer.py`)

Uses **K-Means Clustering (Scikit-Learn)** to group **Top 4 boards** into distinct archetypes.

Instead of hardcoding comps like *"Ionia"*, the model:

* Observes which units consistently appear together
* Clusters boards based on statistical similarity
* Discovers archetypes organically from winning data

If a composition becomes strong, the cluster shifts automatically.

---

#### Item Weighting Logic

Within each archetype, the system computes:

* **Item pick-rate per unit**
* **Statistical significance thresholds** (e.g. >5% usage)

This allows the model to distinguish between:

* **Playable items** (flexible, situational)
* **Best-in-Slot (BiS)** items (high-confidence optimals)

This avoids rigid, one-size-fits-all item rules.

---

### 3. Recommendation Engine — *The Coach*

#### Board Similarity Matching

The user's current board is compared against learned meta clusters using:

* **Cosine Similarity (Dot Product)**

This identifies which archetype the player is *closest to*—even with an incomplete board.

---

#### Recursive Item Solver ("Knapsack" Logic)

A custom **Depth-First Search (DFS) with Memoization** algorithm that explores **all valid crafting paths** from raw components.

**Optimization goal:** Maximize total **Board Power Score** based on:

* **Unit Importance** (carry vs tank)
* **Item Fit** (statistical success on that unit)

**Output:**

* A concrete **step-by-step crafting plan**, not just a score

Example:

> *"Make Blue Buff on Ahri and Bramble Vest on Wukong"*

---

### 4. Predictive Modeling — *Experimental*

#### Placement Prediction Network (`model.py`, `train.py`)

A **PyTorch neural network** that predicts **final placement (1–8)** from a board state.

**Key techniques:**

* Embedding layers for **Units** and **Items**
* Dense layers for board-level reasoning

This model is experimental but lays the groundwork for:

* Expected placement estimates
* Risk-aware decision making

---

## ✨ Key Capabilities

* **Context-Aware Recommendations**
  Recommends the *best board you can actually build*, not a theoretical ideal.

* **Dynamic Meta Adaptation**
  Automatically responds to buffs, nerfs, and meta shifts via live data.

* **Flexible Item Logic**
  Supports optimal, playable, and greedy crafting paths.

* **Complex Recipe Handling**
  Fully understands Emblems, Spatula/Frying Pan recipes, and component logic.

* **High-ELO Bias**
  Trained exclusively on Challenger/GM-level gameplay.

---

## 🛠 Tech Stack

* **Language:** Python 3.x
* **Machine Learning:**

  * PyTorch (Neural Networks)
  * Scikit-Learn (K-Means Clustering)
  * NumPy (Vector Math)
* **Data Processing:** Pandas, JSON
* **Networking:** Requests (Riot Games API)

---

## 📌 Status & Roadmap

* ✅ Live data scraping and preprocessing
* ✅ Unsupervised meta discovery
* ✅ Recursive item crafting solver
* 🧪 Experimental placement prediction model
* 🔮 Future work:

  * Making items vs waiting for better components
  * Real-time in-game assistant integration
  * Quality User Interface
  * Integrating Predictive Model into Recommendation Engine

---

## ⚠️ Disclaimer

This project is **not affiliated with or endorsed by Riot Games**.
All game data is accessed via the official Riot API and used for educational and analytical purposes.

---

## 📄 License

MIT License

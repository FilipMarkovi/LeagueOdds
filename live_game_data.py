import os
import time
import math
import threading
import requests
import urllib3
import pandas as pd
import joblib
import tkinter as tk
import sys

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

BASE_URL = "https://127.0.0.1:2999/liveclientdata"
MODEL_PATH = resource_path("lol_win_predictor_model.pkl")

FEATURE_COLUMNS = [
    "game_time",
    "gold_diff",
    "xp_diff",
    "team_towers",
    "team_dragons",
    "team_barons",
    "team_void_grubs",
    "lobby_rank_num"
]

TEAM_ORDER = "ORDER"
TEAM_CHAOS = "CHAOS"


def safe_get_json(endpoint, timeout=2):
    response = requests.get(f"{BASE_URL}/{endpoint}", verify=False, timeout=timeout)
    response.raise_for_status()
    return response.json()


def get_live_game_data():
    try:
        all_game_data = safe_get_json("allgamedata")
        game_stats = safe_get_json("gamestats")
        return all_game_data, game_stats
    except requests.exceptions.RequestException:
        return None, None


def _team_from_player(player):
    return str(player.get("team", "")).upper()


def _safe_float(value, default=0.0):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _extract_objectives_from_events(events):
    objective_counts = {
        TEAM_ORDER: {"towers": 0, "dragons": 0, "barons": 0, "void_grubs": 0},
        TEAM_CHAOS: {"towers": 0, "dragons": 0, "barons": 0, "void_grubs": 0},
    }

    for event in events:
        event_name = str(event.get("EventName", "")).lower()
        monster_type = str(event.get("MonsterType", "")).lower()
        killer_team = str(event.get("KillerTeam", "")).upper()

        if killer_team not in objective_counts:
            continue

        if "turret" in event_name or "tower" in event_name:
            objective_counts[killer_team]["towers"] += 1
        elif "dragon" in event_name or "dragon" in monster_type:
            objective_counts[killer_team]["dragons"] += 1
        elif "baron" in event_name or "baron" in monster_type:
            objective_counts[killer_team]["barons"] += 1
        elif "grub" in event_name or "horde" in monster_type:
            objective_counts[killer_team]["void_grubs"] += 1

    return objective_counts


def build_feature_row(all_game_data, game_stats):
    all_players = all_game_data.get("allPlayers", [])
    events = (all_game_data.get("events", {}) or {}).get("Events", [])
    objective_counts = _extract_objectives_from_events(events)

    blue_players = [p for p in all_players if _team_from_player(p) == TEAM_ORDER]
    red_players = [p for p in all_players if _team_from_player(p) == TEAM_CHAOS]

    if len(blue_players) != 5 or len(red_players) != 5:
        raise ValueError("Incomplete player snapshot")

    blue_total_gold = sum(_safe_float(p.get("scores", {}).get("totalGoldEarned", 0)) for p in blue_players)
    red_total_gold = sum(_safe_float(p.get("scores", {}).get("totalGoldEarned", 0)) for p in red_players)

    blue_xp_sum = sum(_safe_float(p.get("level", 1)) * 1000 for p in blue_players)
    red_xp_sum = sum(_safe_float(p.get("level", 1)) * 1000 for p in red_players)

    game_time_seconds = _safe_float(game_stats.get("gameTime", 0), default=0.0)
    game_time_minutes = float(int(game_time_seconds // 60))

    feature_row = {
        "game_time": game_time_minutes,
        "gold_diff": blue_total_gold - red_total_gold,
        "xp_diff": blue_xp_sum - red_xp_sum,
        "team_towers": objective_counts[TEAM_ORDER]["towers"],
        "team_dragons": objective_counts[TEAM_ORDER]["dragons"],
        "team_barons": objective_counts[TEAM_ORDER]["barons"],
        "team_void_grubs": objective_counts[TEAM_ORDER]["void_grubs"],
        "lobby_rank_num": 4
    }

    feature_df = pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)
    return feature_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)


class LoLPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LoL Win Predictor")
        
        # Window configuration for transparent overlay appearance
        self.root.overrideredirect(True) # Removes window borders/titlebar
        self.root.attributes("-topmost", True) # Keeps it floating above the game
        self.root.attributes("-alpha", 0.85) # Semi-transparent background
        self.root.geometry("220x90+50+50") # Compact size, placed near top-left
        self.root.configure(bg="#1e1e1e")

        # Make window draggable so users can place it anywhere on screen
        self.root.bind("<Button-1>", self.start_move)
        self.root.bind("<B1-Motion>", self.do_move)

        # UI Elements
        self.title_label = tk.Label(root, text="MATCH WIN ODDS", fg="#aaaaaa", bg="#1e1e1e", font=("Arial", 9, "bold"))
        self.title_label.pack(pady=(5, 0))

        self.status_label = tk.Label(root, text="Waiting for game...", fg="#ffcc00", bg="#1e1e1e", font=("Arial", 10))
        self.status_label.pack(pady=5)

        self.odds_label = tk.Label(root, text="Blue: --% | Red: --%", fg="#ffffff", bg="#1e1e1e", font=("Arial", 11, "bold"))
        self.odds_label.pack(pady=(0, 5))

        # Load ML Model
        if not os.path.exists(MODEL_PATH):
            self.status_label.config(text="Model missing!", fg="#ff4444")
            return

        self.model = joblib.load(MODEL_PATH)

        # Start background polling thread
        self.running = True
        self.thread = threading.Thread(target=self.poll_game_loop, daemon=True)
        self.thread.start()

    def start_move(self, event):
        self.x = event.x
        self.y = event.y

    def do_move(self, event):
        x = self.root.winfo_x() + (event.x - self.x)
        y = self.root.winfo_y() + (event.y - self.y)
        self.root.geometry(f"+{x}+{y}")

    def poll_game_loop(self):
        last_evaluated_minute = -1
        while self.running:
            all_game_data, game_stats = get_live_game_data()
            if all_game_data is None:
                self.root.after(0, lambda: self.status_label.config(text="Not in game...", fg="#888888"))
                self.root.after(0, lambda: self.odds_label.config(text="Blue: --% | Red: --%"))
                time.sleep(10)
                continue

            game_time_seconds = _safe_float(game_stats.get("gameTime", 0), default=0.0)
            current_minute = int(game_time_seconds // 60)

            if current_minute > 0 and current_minute != last_evaluated_minute:
                try:
                    X_live = build_feature_row(all_game_data, game_stats)
                    blue_win_prob = float(self.model.predict_proba(X_live)[0][1])
                    red_win_prob = 1.0 - blue_win_prob

                    status_text = f"Game Time: {current_minute}m"
                    odds_text = f"Blue: {blue_win_prob*100:.1f}%  Red: {red_win_prob*100:.1f}%"

                    # Safely update GUI elements from background thread
                    self.root.after(0, lambda s=status_text, o=odds_text: [
                        self.status_label.config(text=s, fg="#00ffcc"),
                        self.odds_label.config(text=o)
                    ])

                    last_evaluated_minute = current_minute
                except Exception as e:
                    print(f"Prediction error: {e}")

            time.sleep(10)


if __name__ == "__main__":
    root = tk.Tk()
    app = LoLPredictorApp(root)
    root.mainloop()
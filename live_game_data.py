import os
import time
import requests
import urllib3
import pandas as pd
import joblib

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://127.0.0.1:2999/liveclientdata"
MODEL_PATH = "lol_win_predictor_model.pkl"

FEATURE_COLUMNS = [
    "game_time",
    "gold_diff",
    "level_diff",
    "towers_diff",
    "dragons_diff",
    "barons_diff",
    "void_grubs_diff",
]


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


def _player_total_gold(player):
    current_gold = float(player.get("currentGold", 0) or 0)
    item_worth = sum(float((item or {}).get("price", 0) or 0) for item in player.get("items", []))
    return current_gold + item_worth


def _player_xp_proxy(player):
    return float(player.get("level", 1) or 1)


def _extract_objectives_from_events(events):
    objective_counts = {
        "ORDER": {"towers": 0, "dragons": 0, "barons": 0, "void_grubs": 0},
        "CHAOS": {"towers": 0, "dragons": 0, "barons": 0, "void_grubs": 0},
    }

    for event in events:
        event_name = str(event.get("EventName", "")).lower()
        killer_team = str(event.get("KillerTeam", "")).upper()

        if killer_team not in objective_counts:
            continue

        if "turret" in event_name:
            objective_counts[killer_team]["towers"] += 1
        elif "dragon" in event_name:
            objective_counts[killer_team]["dragons"] += 1
        elif "baron" in event_name:
            objective_counts[killer_team]["barons"] += 1
        elif "grub" in event_name:
            objective_counts[killer_team]["void_grubs"] += 1

    return objective_counts


def build_feature_row(all_game_data, game_stats):
    all_players = all_game_data.get("allPlayers", [])
    events = (all_game_data.get("events", {}) or {}).get("Events", [])
    objective_counts = _extract_objectives_from_events(events)

    blue_players = [p for p in all_players if _team_from_player(p) == "ORDER"]
    red_players = [p for p in all_players if _team_from_player(p) == "CHAOS"]

    blue_total_gold = sum(_player_total_gold(player) for player in blue_players)
    red_total_gold = sum(_player_total_gold(player) for player in red_players)

    blue_level_sum = sum(_player_xp_proxy(player) for player in blue_players)
    red_level_sum = sum(_player_xp_proxy(player) for player in red_players)

    game_time_minutes = float(game_stats.get("gameTime", 0) or 0) / 60.0

    feature_row = {
        "game_time": game_time_minutes,
        "gold_diff": blue_total_gold - red_total_gold,
        "level_diff": blue_level_sum - red_level_sum,
        "towers_diff": objective_counts["ORDER"]["towers"] - objective_counts["CHAOS"]["towers"],
        "dragons_diff": objective_counts["ORDER"]["dragons"] - objective_counts["CHAOS"]["dragons"],
        "barons_diff": objective_counts["ORDER"]["barons"] - objective_counts["CHAOS"]["barons"],
        "void_grubs_diff": objective_counts["ORDER"]["void_grubs"] - objective_counts["CHAOS"]["void_grubs"],
    }

    return pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        print("Run train.py first to generate the model.")
        return

    model = joblib.load(MODEL_PATH)

    print("Live predictor started. Waiting for active LoL match...")
    print("Evaluates every 5 in-game minutes (5, 10, 15, ...).")

    last_evaluated_mark = -1

    while True:
        all_game_data, game_stats = get_live_game_data()
        if all_game_data is None:
            print("No active match detected. Retrying in 10 seconds...")
            time.sleep(10)
            continue

        game_time_seconds = float(game_stats.get("gameTime", 0) or 0)
        current_mark = int(game_time_seconds // 300)

        if current_mark > 0 and current_mark != last_evaluated_mark:
            X_live = build_feature_row(all_game_data, game_stats)
            blue_win_prob = float(model.predict_proba(X_live)[0][1])
            red_win_prob = 1.0 - blue_win_prob

            minute = int(game_time_seconds // 60)
            second = int(game_time_seconds % 60)

            print("-" * 55)
            print(f"Time: {minute:02d}:{second:02d}")
            print(f"Blue (ORDER) win chance: {blue_win_prob * 100:.2f}%")
            print(f"Red  (CHAOS) win chance: {red_win_prob * 100:.2f}%")
            print(f"Feature snapshot: {X_live.to_dict(orient='records')[0]}")

            last_evaluated_mark = current_mark

        time.sleep(15)


if __name__ == "__main__":
    main()
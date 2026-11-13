import os
import pandas as pd

EXTRACT_FOLDER = "lol_dataset"
INTERVALS_PATH = os.path.join(EXTRACT_FOLDER, "intervals.csv")
MATCHES_PATH = os.path.join(EXTRACT_FOLDER, "matches.csv")
OUTPUT_PATH = os.path.join(EXTRACT_FOLDER, "team_diff_training_data.csv")

def build_training_dataset(intervals_path: str, matches_path: str, output_path: str) -> None:
    print("Loading raw datasets...")
    df_intervals = pd.read_csv(intervals_path)
    df_matches = pd.read_csv(matches_path)

    print("Preparing match labels and ranks...")
    if "winning_team" not in df_matches.columns:
        raise ValueError("matches.csv must contain 'winning_team' column")

    # Map winning_team to binary blue_win with robust handling for mixed dtypes
    winning_team_raw = df_matches["winning_team"]
    winning_team_num = pd.to_numeric(winning_team_raw, errors="coerce")
    df_matches["blue_win"] = 0

    numeric_mask = winning_team_num.notna()
    if numeric_mask.any():
        df_matches.loc[numeric_mask, "blue_win"] = (winning_team_num.loc[numeric_mask] == 100).astype(int)

    string_mask = ~numeric_mask
    if string_mask.any():
        winning_team_text = winning_team_raw.loc[string_mask].astype(str).str.strip().str.lower()
        df_matches.loc[string_mask, "blue_win"] = winning_team_text.isin(["blue", "order", "100"]).astype(int)

    # Map average rank to ordinal numeric value
    rank_mapping = {
        'Iron': 1, 'Bronze': 2, 'Silver': 3, 'Gold': 4, 
        'Platinum': 5, 'Emerald': 6, 'Diamond': 7, 'Master': 8, 
        'Grandmaster': 9, 'Challenger': 10, 'UNRANKED': 0
    }
    if "average_rank" in df_matches.columns:
        df_matches["lobby_rank_num"] = df_matches["average_rank"].map(rank_mapping).fillna(0)
    else:
        df_matches["lobby_rank_num"] = 0

    print("Aggregating match-minute team snapshots with strict team-consistency validation...")
    
    snapshot_records = []
    grouped = df_intervals.groupby(["match_id", "minute"])
    
    skipped_count = 0
    for (match_id, minute), group in grouped:
        group_sorted = group.sort_values("player_id").drop_duplicates(subset=["player_id"], keep="last")
        if len(group_sorted) != 10:
            skipped_count += 1
            continue  # Skip incomplete snapshots
            
        blue_team = group_sorted.iloc[:5]
        red_team = group_sorted.iloc[5:]
        
        # --- STRICT TEAM CONSISTENCY CHECK ---
        # Verify that all 5 players in the blue partition share identical team_gold_diff
        # and all 5 players in the red partition share identical team_gold_diff.
        if "team_gold_diff" in blue_team.columns:
            blue_tgd_unique = blue_team["team_gold_diff"].nunique()
            red_tgd_unique = red_team["team_gold_diff"].nunique()
            
            if blue_tgd_unique > 1 or red_tgd_unique > 1:
                skipped_count += 1
                continue # Inconsistent internal team states, skip snapshot
        
        # Calculate team aggregates
        blue_total_gold = blue_team["total_gold"].sum()
        red_total_gold = red_team["total_gold"].sum()
        
        blue_total_xp = blue_team["xp"].sum()
        red_total_xp = red_team["xp"].sum()
        
        blue_level = blue_team["level"].sum()
        red_level = red_team["level"].sum()
        
        # Objectives are shared team metrics (take max across the team players)
        blue_towers = blue_team["team_towers"].max()
        red_towers = red_team["team_towers"].max()
        
        blue_dragons = blue_team["team_dragons"].max()
        red_dragons = red_team["team_dragons"].max()
        
        blue_barons = blue_team["team_barons"].max()
        red_barons = red_team["team_barons"].max()
        
        blue_void_grubs = blue_team["team_void_grubs"].max()
        red_void_grubs = red_team["team_void_grubs"].max()

        snapshot_records.append({
            "match_id": match_id,
            "minute": minute,
            "game_time": minute,
            "gold_diff": blue_total_gold - red_total_gold,
            "team_gold_diff": blue_team["team_gold_diff"].iloc[0], 
            "xp_diff": blue_total_xp - red_total_xp,
            "level_diff": blue_level - red_level,
            "towers_diff": blue_towers - red_towers,
            "dragons_diff": blue_dragons - red_dragons,
            "barons_diff": blue_barons - red_barons,
            "void_grubs_diff": blue_void_grubs - red_void_grubs
        })

    print(f"Skipped {skipped_count} invalid or inconsistent snapshots.")
    feature_table = pd.DataFrame(snapshot_records)

    # Merge with match metadata (blue_win, lobby_rank_num)
    label_table = df_matches[["match_id", "blue_win", "lobby_rank_num"]].drop_duplicates(subset=["match_id"])
    output_df = feature_table.merge(label_table, on="match_id", how="inner")

    output_df.to_csv(output_path, index=False)

    num_matches = output_df["match_id"].nunique()
    num_rows = len(output_df)
    avg_rows_per_match = num_rows / num_matches if num_matches else 0

    print("Preprocessing complete.")
    print(f"Saved: {output_path}")
    print(f"Rows: {num_rows}")
    print(f"Unique matches: {num_matches}")
    print(f"Average snapshots per match: {avg_rows_per_match:.2f}")

if __name__ == "__main__":
    build_training_dataset(INTERVALS_PATH, MATCHES_PATH, OUTPUT_PATH)
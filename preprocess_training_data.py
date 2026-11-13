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

    print("Preparing match labels...")
    if "winning_team" not in df_matches.columns:
        raise ValueError("matches.csv must contain 'winning_team' column")

    sample_val = df_matches["winning_team"].iloc[0]
    if sample_val in [100, 200]:
        df_matches["blue_win"] = (df_matches["winning_team"] == 100).astype(int)
    elif isinstance(sample_val, str):
        df_matches["blue_win"] = (
            df_matches["winning_team"].astype(str).str.lower().isin(["blue", "100", "win"]).astype(int)
        )
    else:
        df_matches["blue_win"] = (df_matches["winning_team"] == 1).astype(int)

    print("Aggregating match-minute team snapshots...")
    df_intervals["is_blue"] = df_intervals["player_id"].isin([1, 2, 3, 4, 5])

    team_agg = df_intervals.groupby(["match_id", "minute"]).apply(
        lambda group: pd.Series(
            {
                "blue_total_gold": group[group["is_blue"]]["total_gold"].sum(),
                "red_total_gold": group[~group["is_blue"]]["total_gold"].sum(),
                "blue_level_sum": group[group["is_blue"]]["level"].sum(),
                "red_level_sum": group[~group["is_blue"]]["level"].sum(),
                "blue_team_towers": group[group["is_blue"]]["team_towers"].max(),
                "red_team_towers": group[~group["is_blue"]]["team_towers"].max(),
                "blue_team_dragons": group[group["is_blue"]]["team_dragons"].max(),
                "red_team_dragons": group[~group["is_blue"]]["team_dragons"].max(),
                "blue_team_barons": group[group["is_blue"]]["team_barons"].max(),
                "red_team_barons": group[~group["is_blue"]]["team_barons"].max(),
                "blue_team_void_grubs": group[group["is_blue"]]["team_void_grubs"].max(),
                "red_team_void_grubs": group[~group["is_blue"]]["team_void_grubs"].max(),
            }
        )
    ).reset_index()

    team_agg["game_time"] = team_agg["minute"]
    team_agg["gold_diff"] = team_agg["blue_total_gold"] - team_agg["red_total_gold"]
    team_agg["level_diff"] = team_agg["blue_level_sum"] - team_agg["red_level_sum"]
    team_agg["towers_diff"] = team_agg["blue_team_towers"] - team_agg["red_team_towers"]
    team_agg["dragons_diff"] = team_agg["blue_team_dragons"] - team_agg["red_team_dragons"]
    team_agg["barons_diff"] = team_agg["blue_team_barons"] - team_agg["red_team_barons"]
    team_agg["void_grubs_diff"] = team_agg["blue_team_void_grubs"] - team_agg["red_team_void_grubs"]

    keep_columns = [
        "match_id",
        "minute",
        "game_time",
        "gold_diff",
        "level_diff",
        "towers_diff",
        "dragons_diff",
        "barons_diff",
        "void_grubs_diff",
    ]
    feature_table = team_agg[keep_columns]

    label_table = df_matches[["match_id", "blue_win"]].drop_duplicates(subset=["match_id"])
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

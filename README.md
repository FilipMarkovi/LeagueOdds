# LoL Winning Odds Predictor

Predicts the win probability for **Blue side (ORDER)** vs **Red side (CHAOS)** in League of Legends from simple team-difference features over time.

## What this project does

- Builds training-ready team-diff snapshots from raw match interval data.
- Trains an XGBoost classifier to predict whether Blue team wins.
- Reports:
  - overall test accuracy
  - ROC AUC
  - accuracy at 5-minute checkpoints (5, 10, 15, ...)
- Runs live prediction from the LoL Live Client API and prints win chance every 5 in-game minutes.

## Repository structure

- `preprocess_training_data.py` — preprocess raw CSVs into training-ready feature CSV
- `train.py` — train and evaluate model
- `live_game_data.py` — live inference loop during active matches
- `lol_dataset/` — data folder (`intervals.csv`, `matches.csv`, output training CSV)
- `lol_win_predictor_model.pkl` — trained model artifact
- `index.html` — project landing page / usage guide

## Feature set used

- `game_time` (minutes)
- `gold_diff` (blue total gold - red total gold)
- `level_diff` (sum(blue champion levels) - sum(red champion levels))
- `towers_diff`
- `dragons_diff`
- `barons_diff`
- `void_grubs_diff`

Target:

- `blue_win` (1 if blue wins, else 0)

## Quick start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

1) Build training CSV

```bash
python preprocess_training_data.py
```

2) Train model

```bash
python train.py
```

3) Run live predictions (while in active LoL match)

```bash
python live_game_data.py
```

## Notes

- The live client API is expected at `https://127.0.0.1:2999/liveclientdata`.
- `team_diff_training_data.csv` has multiple rows per match (one row per tracked minute), so total rows are much larger than number of matches.
- `index.html` includes placeholder ZIP download buttons. You can add ZIP files under `downloads/` later.

#!/usr/bin/env bash
set -euo pipefail

mkdir -p downloads

zip -r downloads/lol_win_predictor_program.zip \
  live_game_data.py \
  preprocess_training_data.py \
  train.py \
  requirements.txt \
  README.md \
  index.html

if [[ -f "lol_win_predictor_model.pkl" ]]; then
  zip -j downloads/lol_win_predictor_model.zip lol_win_predictor_model.pkl
  echo "Created model ZIP: downloads/lol_win_predictor_model.zip"
else
  echo "Model file not found: lol_win_predictor_model.pkl"
  echo "Skipping model ZIP."
fi

echo "Created program ZIP: downloads/lol_win_predictor_program.zip"

import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
from preprocess_training_data import build_training_dataset

extract_folder = "lol_dataset"
intervals_path = os.path.join(extract_folder, "intervals.csv")
matches_path = os.path.join(extract_folder, "matches.csv")
preprocessed_path = os.path.join(extract_folder, "team_diff_training_data.csv")

# ==========================================
# 1. BUILD/LOAD PREPROCESSED TRAIN DATA
# ==========================================
if not os.path.exists(preprocessed_path):
    print("Preprocessed CSV not found. Building it now...")
    build_training_dataset(intervals_path, matches_path, preprocessed_path)

print("Loading preprocessed training dataset...")
df_model = pd.read_csv(preprocessed_path)
print(f"Preprocessed dataset shape: {df_model.shape}")

# ==========================================
# 4. DEFINE FEATURES & TRAIN XGBOOST
# ==========================================
feature_columns = [
    'game_time',
    'gold_diff',
    'level_diff',
    'towers_diff',
    'dragons_diff',
    'barons_diff',
    'void_grubs_diff'
]

target_column = 'blue_win'

match_labels = df_model[['match_id', target_column]].drop_duplicates()
train_match_ids, test_match_ids = train_test_split(
    match_labels['match_id'],
    test_size=0.2,
    random_state=42,
    stratify=match_labels[target_column]
)

train_mask = df_model['match_id'].isin(train_match_ids)
test_mask = df_model['match_id'].isin(test_match_ids)

X_train = df_model.loc[train_mask, feature_columns]
y_train = df_model.loc[train_mask, target_column]
X_test = df_model.loc[test_mask, feature_columns]
y_test = df_model.loc[test_mask, target_column]

print("Training XGBoost Classifier on clean team-level snapshots...")
model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities)

print(f"--- Model Retrained Successfully! ---")
print(f"Test Accuracy: {acc * 100:.2f}%")
print(f"ROC AUC Score: {roc_auc:.4f}")

print("Accuracy by game_time checkpoints (minutes):")
test_eval = df_model.loc[test_mask, ['game_time']].copy()
test_eval['y_true'] = y_test.values
test_eval['y_pred'] = predictions

checkpoints = sorted(test_eval['game_time'].dropna().unique())
for minute in checkpoints:
    if minute % 5 != 0:
        continue
    minute_rows = test_eval[test_eval['game_time'] == minute]
    if len(minute_rows) == 0:
        continue
    minute_acc = accuracy_score(minute_rows['y_true'], minute_rows['y_pred'])
    print(f"  {int(minute):02d} min -> {minute_acc * 100:.2f}% (n={len(minute_rows)})")

joblib.dump(model, "lol_win_predictor_model.pkl")
print("Model saved to disk as 'lol_win_predictor_model.pkl'.")
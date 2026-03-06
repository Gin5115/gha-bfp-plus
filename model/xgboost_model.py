import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load and prepare data ──────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('data/input.csv')

rename = {
    'totalFilesPushed':    'files_pushed',
    'totalLinesAdded':     'lines_added',
    'totalLinesDeleted':   'lines_deleted',
    'commitMsgLen':        'commit_msg_len',
    'timeElapse':          'time_elapse',
    'timeLastFailedBuild': 'time_last_failed_build',
    'lastBuildResult':     'last_build_result',
    'projectHistory':      'failure_rate',
    'projectRecent':       'failure_rate_recent',
    'dayTime':             'day_time',
    'configLines':         'config_lines',
    'configWarn':          'config_warn',
    'configErr':           'config_err',
    'actionLintErr':       'action_lint_err',
    'jobsNum':             'jobs_num',
    'stepsNum':            'steps_num',
    'projectFiles':        'repository_files',
    'intervalProject':     'repository_age',
    'authorHistory':       'author_failure_rate',
    'authorNum':           'author_build_num',
    'projectLines':        'repository_lines',
    'projectComments':     'repository_comments',
    'projectOwnerType':    'repository_owner_type',
    'projectLanguage':     'repository_language',
}
df.rename(columns=rename, inplace=True)

df['createdTime']           = pd.to_datetime(df['createdTime'], dayfirst=True)
df                          = df.sort_values('createdTime').reset_index(drop=True)
df['last_build_result']     = (df['last_build_result'] == 'failure').astype(int)

le_lang  = LabelEncoder()
le_owner = LabelEncoder()
df['repository_language']   = le_lang.fit_transform(
    df['repository_language'].fillna('Unknown'))
df['repository_owner_type'] = le_owner.fit_transform(df['repository_owner_type'])
df['label']                 = (df['result'] == 'failure').astype(int)

FEATURES = [
    'files_pushed', 'lines_added', 'lines_deleted', 'commit_msg_len',
    'day_time', 'weekday', 'monthday', 'repository_age',
    'time_elapse', 'time_last_failed_build', 'last_build_result',
    'author_failure_rate', 'author_build_num', 'failure_rate',
    'failure_rate_recent',
    'config_lines', 'config_warn', 'config_err', 'action_lint_err',
    'jobs_num', 'steps_num',
    'repository_files', 'repository_lines', 'repository_comments',
    'repository_owner_type', 'repository_language',
]

split   = int(len(df) * 0.8)
X_train = df[FEATURES].fillna(0).iloc[:split]
y_train = df['label'].iloc[:split]
X_test  = df[FEATURES].fillna(0).iloc[split:]
y_test  = df['label'].iloc[split:]

print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# ── 2. Baseline RF for comparison ─────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
print("\nBaseline Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print(f"  Acc={accuracy_score(y_test,rf_preds):.4f}  "
      f"Prec={precision_score(y_test,rf_preds):.4f}  "
      f"Rec={recall_score(y_test,rf_preds):.4f}  "
      f"F1={f1_score(y_test,rf_preds):.4f}")

# ── 3. XGBoost baseline ───────────────────────────────────────────────────────
print("\nXGBoost baseline (default params)...")
xgb = XGBClassifier(
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
    verbosity=0
)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
print(f"  Acc={accuracy_score(y_test,xgb_preds):.4f}  "
      f"Prec={precision_score(y_test,xgb_preds):.4f}  "
      f"Rec={recall_score(y_test,xgb_preds):.4f}  "
      f"F1={f1_score(y_test,xgb_preds):.4f}")

# ── 4. XGBoost hyperparameter tuning ─────────────────────────────────────────
print("\nTuning XGBoost (40 iterations)...")
print("This will take 2-3 minutes...")

param_dist = {
    'n_estimators':     [100, 200, 300, 500],
    'max_depth':        [3, 4, 5, 6, 7, 8],
    'learning_rate':    [0.01, 0.05, 0.1, 0.2, 0.3],
    'subsample':        [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7],
    'gamma':            [0, 0.1, 0.2, 0.3, 0.5],
    'reg_alpha':        [0, 0.01, 0.1, 1.0],
    'reg_lambda':       [0.1, 1.0, 5.0, 10.0],
}

search = RandomizedSearchCV(
    XGBClassifier(random_state=42, n_jobs=-1,
                  eval_metric='logloss', verbosity=0),
    param_distributions=param_dist,
    n_iter=40,
    cv=3,
    scoring=make_scorer(f1_score),
    random_state=42,
    n_jobs=-1,
    verbose=1
)
search.fit(X_train, y_train)

print(f"\nBest XGBoost params:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")

best_xgb   = search.best_estimator_
xgbt_preds = best_xgb.predict(X_test)

print(f"\nTuned XGBoost results:")
print(f"  Acc={accuracy_score(y_test,xgbt_preds):.4f}  "
      f"Prec={precision_score(y_test,xgbt_preds):.4f}  "
      f"Rec={recall_score(y_test,xgbt_preds):.4f}  "
      f"F1={f1_score(y_test,xgbt_preds):.4f}")

# ── 5. Final comparison table ─────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
print(f"{'Model':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
print("-"*60)

models = {
    'Paper (RF)':         (0.7804, 0.7532, 0.8376, 0.7927),
    'Our RF (default)':   (accuracy_score(y_test,rf_preds),
                           precision_score(y_test,rf_preds),
                           recall_score(y_test,rf_preds),
                           f1_score(y_test,rf_preds)),
    'XGBoost (default)':  (accuracy_score(y_test,xgb_preds),
                           precision_score(y_test,xgb_preds),
                           recall_score(y_test,xgb_preds),
                           f1_score(y_test,xgb_preds)),
    'XGBoost (tuned)':    (accuracy_score(y_test,xgbt_preds),
                           precision_score(y_test,xgbt_preds),
                           recall_score(y_test,xgbt_preds),
                           f1_score(y_test,xgbt_preds)),
}

for name, (acc, prec, rec, f1) in models.items():
    print(f"  {name:<23} {acc:>7.4f} {prec:>7.4f} {rec:>7.4f} {f1:>7.4f}")

print("="*60)

# ── 6. Save best model ────────────────────────────────────────────────────────
best_xgb_f1 = f1_score(y_test, xgbt_preds)
best_rf_f1  = f1_score(y_test, rf_preds)

if best_xgb_f1 > best_rf_f1:
    print(f"\nXGBoost wins! (+{(best_xgb_f1-best_rf_f1)*100:.2f}% F1)")
    print("Saving XGBoost as production model...")
    with open('model/rf_model.pkl', 'wb') as f:
        pickle.dump(best_xgb, f)
    with open('model/encoders.pkl', 'rb') as f:
        enc = pickle.load(f)
    enc['model_type'] = 'xgboost'
    with open('model/encoders.pkl', 'wb') as f:
        pickle.dump(enc, f)
    print("  Saved: model/rf_model.pkl (XGBoost)")
else:
    print(f"\nRandom Forest wins. XGBoost didn't improve.")
    print("Keeping existing RF model.")

print("\nDone!")

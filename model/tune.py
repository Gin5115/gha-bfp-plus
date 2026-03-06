import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, make_scorer)
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load and prepare data (same as train.py) ───────────────────────────────
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

split         = int(len(df) * 0.8)
X_train       = df[FEATURES].fillna(0).iloc[:split]
y_train       = df['label'].iloc[:split]
X_test        = df[FEATURES].fillna(0).iloc[split:]
y_test        = df['label'].iloc[split:]

print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")

# ── 2. Baseline RF (default params) ──────────────────────────────────────────
print("\nBaseline Random Forest (default params)...")
baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
baseline.fit(X_train, y_train)
preds = baseline.predict(X_test)
print(f"  Acc={accuracy_score(y_test,preds):.4f}  "
      f"Prec={precision_score(y_test,preds):.4f}  "
      f"Rec={recall_score(y_test,preds):.4f}  "
      f"F1={f1_score(y_test,preds):.4f}")

# ── 3. Hyperparameter search space ───────────────────────────────────────────
param_dist = {
    'n_estimators':      [100, 200, 300, 500],
    'max_depth':         [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':      ['sqrt', 'log2', 0.3, 0.5],
    'class_weight':      [None, 'balanced'],
    'bootstrap':         [True, False],
}

print("\nRunning RandomizedSearchCV (50 iterations, 3-fold CV)...")
print("This will take 3-5 minutes...")

scorer = make_scorer(f1_score)
search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=50,
    cv=3,
    scoring=scorer,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
search.fit(X_train, y_train)

print(f"\nBest params found:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")

# ── 4. Evaluate best RF ───────────────────────────────────────────────────────
best_rf = search.best_estimator_
preds   = best_rf.predict(X_test)
print(f"\nTuned Random Forest results:")
print(f"  Acc={accuracy_score(y_test,preds):.4f}  "
      f"Prec={precision_score(y_test,preds):.4f}  "
      f"Rec={recall_score(y_test,preds):.4f}  "
      f"F1={f1_score(y_test,preds):.4f}")

# ── 5. Compare ────────────────────────────────────────────────────────────────
base_f1  = f1_score(y_test, baseline.predict(X_test))
tuned_f1 = f1_score(y_test, preds)
print(f"\nImprovement: F1 {base_f1:.4f} → {tuned_f1:.4f} "
      f"({(tuned_f1-base_f1)*100:+.2f}%)")

# ── 6. Save best params for train.py ─────────────────────────────────────────
import json
with open('model/best_params.json', 'w') as f:
    # Convert None to null for JSON
    params = {k: (v if v is not None else None)
              for k, v in search.best_params_.items()}
    json.dump(params, f, indent=2)
print("\nSaved best params to model/best_params.json")
print("Run model/train_tuned.py to retrain with these params.")

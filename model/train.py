import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('data/input.csv')
print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

# ── 2. Rename columns ─────────────────────────────────────────────────────────
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

# ── 3. Parse and sort by time (critical for time-series validation) ───────────
print("Parsing dates and sorting by time...")
df['createdTime'] = pd.to_datetime(df['createdTime'], dayfirst=True)
df = df.sort_values('createdTime').reset_index(drop=True)
print(f"  Date range: {df['createdTime'].iloc[0].date()} "
      f"→ {df['createdTime'].iloc[-1].date()}")

# ── 4. Encode categorical columns ─────────────────────────────────────────────
# last_build_result — binary: failure=1, everything else=0
# Handles noise: cancelled/skipped/startup_failure all become 0
df['last_build_result'] = (df['last_build_result'] == 'failure').astype(int)
print(f"  last_build_result: failure=1, all others=0")

# repository_language — LabelEncoder (RF handles this well)
le_lang = LabelEncoder()
df['repository_language'] = le_lang.fit_transform(
    df['repository_language'].fillna('Unknown')
)

# repository_owner_type — LabelEncoder
le_owner = LabelEncoder()
df['repository_owner_type'] = le_owner.fit_transform(
    df['repository_owner_type']
)
print(f"  owner_type mapping: "
      f"{dict(zip(le_owner.classes_, le_owner.transform(le_owner.classes_)))}")
print(f"  languages known: {len(le_lang.classes_)}")

# ── 5. Target label ───────────────────────────────────────────────────────────
df['label'] = (df['result'] == 'failure').astype(int)
print(f"  Class balance — "
      f"success: {(df['label']==0).sum()}, "
      f"failure: {(df['label']==1).sum()}")

# ── 6. Feature list (26 features — Table I from paper) ───────────────────────
FEATURES = [
    # current build (8)
    'files_pushed', 'lines_added', 'lines_deleted', 'commit_msg_len',
    'day_time', 'weekday', 'monthday', 'repository_age',
    # historical build (7)
    'time_elapse', 'time_last_failed_build', 'last_build_result',
    'author_failure_rate', 'author_build_num', 'failure_rate',
    'failure_rate_recent',
    # configuration file (6)
    'config_lines', 'config_warn', 'config_err', 'action_lint_err',
    'jobs_num', 'steps_num',
    # repository (5)
    'repository_files', 'repository_lines', 'repository_comments',
    'repository_owner_type', 'repository_language',
]

X = df[FEATURES].fillna(0)
y = df['label']

# ── 7. Time-series validation (matches paper exactly) ────────────────────────
# Initial window: 20,000 | Horizon: 3,000 | 4 folds
print("\nTime-series validation (4 folds)...")
print(f"  Initial window: 20,000 | Horizon: 3,000")

INIT_WINDOW = 20000
HORIZON     = 3000
fold_results = []

for fold in range(4):
    train_end  = INIT_WINDOW + fold * HORIZON
    test_start = train_end
    test_end   = test_start + HORIZON

    if test_end > len(X):
        break

    X_train = X.iloc[:train_end]
    X_test  = X.iloc[test_start:test_end]
    y_train = y.iloc[:train_end]
    y_test  = y.iloc[test_start:test_end]

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)
    fold_results.append((acc, prec, rec, f1))

    print(f"  Fold {fold+1}: "
          f"Acc={acc:.4f}  Prec={prec:.4f}  "
          f"Rec={rec:.4f}  F1={f1:.4f}")

avg = [sum(x)/len(x) for x in zip(*fold_results)]
print(f"\n  Average : "
      f"Acc={avg[0]:.4f}  Prec={avg[1]:.4f}  "
      f"Rec={avg[2]:.4f}  F1={avg[3]:.4f}")
print(f"  Paper   : Acc>=0.78  Prec>=0.75  Rec>=0.83  F1>=0.79")

# ── 8. Final model on 80% of data (time-based, not random) ───────────────────
# We use 80% for training so the holdout test is honest
# (no data leakage — the model never sees holdout rows)
print("\nTraining final model on first 80% of data...")
split         = int(len(X) * 0.8)
X_train_final = X.iloc[:split]
y_train_final = y.iloc[:split]
X_holdout     = X.iloc[split:]
y_holdout     = y.iloc[split:]

final_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
final_rf.fit(X_train_final, y_train_final)

holdout_preds = final_rf.predict(X_holdout)
print(f"  Holdout Acc : {accuracy_score(y_holdout, holdout_preds):.4f}")
print(f"  Holdout Prec: {precision_score(y_holdout, holdout_preds):.4f}")
print(f"  Holdout Rec : {recall_score(y_holdout, holdout_preds):.4f}")
print(f"  Holdout F1  : {f1_score(y_holdout, holdout_preds):.4f}")

# ── 9. Feature importances ────────────────────────────────────────────────────
print("\nTop 10 feature importances:")
importances = pd.Series(final_rf.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=False)
for feat, score in importances.head(10).items():
    bar = '█' * int(score * 300)
    print(f"  {feat:<30} {score:.4f}  {bar}")

# ── 10. Save model and encoders ───────────────────────────────────────────────
print("\nSaving artifacts...")
with open('model/rf_model.pkl', 'wb') as f:
    pickle.dump(final_rf, f)

with open('model/encoders.pkl', 'wb') as f:
    pickle.dump({
        'le_lang':          le_lang,
        'le_owner':         le_owner,
        'features':         FEATURES,
        'result_is_binary': True,
        'train_date_range': (
            str(df['createdTime'].iloc[0].date()),
            str(df['createdTime'].iloc[split].date())
        ),
    }, f)

print("  Saved: model/rf_model.pkl")
print("  Saved: model/encoders.pkl")
print("\nDone!")

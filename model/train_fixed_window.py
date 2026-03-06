import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

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

X = df[FEATURES].fillna(0)
y = df['label']

# ── Fixed window validation (matches paper exactly) ───────────────────────────
# fixedWindow=TRUE means training set is always exactly 20k rows
# not expanding — same size every fold
print("\nFixed window time-series validation (matches paper R code)...")
print("  fixedWindow=TRUE | initialWindow=20000 | horizon=3000")

INIT_WINDOW = 20000
HORIZON     = 3000

# caret default mtry for RF = sqrt(n_features) = sqrt(26) ≈ 5
MTRY = int(np.sqrt(len(FEATURES)))
print(f"  mtry (max_features) = sqrt({len(FEATURES)}) = {MTRY} (matches caret default)")

fold_results = []
for fold in range(4):
    # FIXED window — always exactly 20k rows, sliding forward
    train_start = fold * HORIZON
    train_end   = train_start + INIT_WINDOW
    test_start  = train_end
    test_end    = test_start + HORIZON

    if test_end > len(X):
        break

    # Fixed window: slice exactly 20k rows starting from fold*horizon
    X_train = X.iloc[train_start:train_end]
    X_test  = X.iloc[test_start:test_end]
    y_train = y.iloc[train_start:train_end]
    y_test  = y.iloc[test_start:test_end]

    # seed=123 matches paper's set.seed(123)
    rf = RandomForestClassifier(
        n_estimators=500,        # caret RF default is 500
        max_features=MTRY,       # caret default: sqrt(p)
        random_state=123,        # matches set.seed(123)
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)
    fold_results.append((acc, prec, rec, f1))
    print(f"  Fold {fold+1}: Acc={acc:.4f}  Prec={prec:.4f}  "
          f"Rec={rec:.4f}  F1={f1:.4f}")

avg = [sum(x)/len(x) for x in zip(*fold_results)]
print(f"\n  Average : Acc={avg[0]:.4f}  Prec={avg[1]:.4f}  "
      f"Rec={avg[2]:.4f}  F1={avg[3]:.4f}")
print(f"  Paper   : Acc=0.7804  Prec=0.7532  Rec=0.8376  F1=0.7927")

# ── Train final model on 80% ──────────────────────────────────────────────────
print("\nTraining final model...")
split         = int(len(X) * 0.8)
X_train_final = X.iloc[:split]
y_train_final = y.iloc[:split]
X_holdout     = X.iloc[split:]
y_holdout     = y.iloc[split:]

final_rf = RandomForestClassifier(
    n_estimators=500,
    max_features=MTRY,
    random_state=123,
    n_jobs=-1
)
final_rf.fit(X_train_final, y_train_final)

holdout_preds = final_rf.predict(X_holdout)
print(f"  Holdout Acc : {accuracy_score(y_holdout, holdout_preds):.4f}")
print(f"  Holdout Prec: {precision_score(y_holdout, holdout_preds):.4f}")
print(f"  Holdout Rec : {recall_score(y_holdout, holdout_preds):.4f}")
print(f"  Holdout F1  : {f1_score(y_holdout, holdout_preds):.4f}")

# ── Feature importances ───────────────────────────────────────────────────────
print("\nTop 10 feature importances:")
importances = pd.Series(final_rf.feature_importances_, index=FEATURES)
for feat, score in importances.sort_values(ascending=False).head(10).items():
    bar = '█' * int(score * 300)
    print(f"  {feat:<30} {score:.4f}  {bar}")

# ── Save ──────────────────────────────────────────────────────────────────────
print("\nSaving...")
with open('model/rf_model.pkl', 'wb') as f:
    pickle.dump(final_rf, f)

with open('model/encoders.pkl', 'wb') as f:
    pickle.dump({
        'le_lang':          le_lang,
        'le_owner':         le_owner,
        'features':         FEATURES,
        'result_is_binary': True,
        'model_type':       'random_forest',
        'mtry':             MTRY,
        'n_estimators':     500,
        'seed':             123,
    }, f)

print("  Saved: model/rf_model.pkl")
print("  Saved: model/encoders.pkl")
print("\nDone!")

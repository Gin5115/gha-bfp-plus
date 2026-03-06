import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

# ── NEW ENGINEERED FEATURES ───────────────────────────────────────────────────
print("Engineering new features...")

# 1. lines_changed_ratio — how much of the change is additions vs deletions
#    High ratio = mostly additions (new code = more risk)
df['lines_changed_ratio'] = df['lines_added'] / (
    df['lines_added'] + df['lines_deleted'] + 1)

# 2. config_complexity — more jobs × steps = more things that can fail
df['config_complexity'] = df['jobs_num'] * df['steps_num']

# 3. config_error_density — errors per line of config
#    High density = poorly written workflow
df['config_error_density'] = df['config_err'] / (df['config_lines'] + 1)

# 4. author_experience_score — how prolific is this author relative to repo age
#    Low score = new contributor = higher risk
df['author_experience_score'] = df['author_build_num'] / (df['repository_age'] + 1)

# 5. failure_momentum — weighted combo of overall and recent failure rates
#    Recent failures weighted 2x because they're more predictive
df['failure_momentum'] = (df['failure_rate'] + 2 * df['failure_rate_recent']) / 3

# 6. is_peak_hour — builds during peak hours (9am-6pm) are riskier
#    (more concurrent changes, more integration conflicts)
df['is_peak_hour'] = ((df['day_time'] >= 9) & (df['day_time'] <= 18)).astype(int)

# 7. is_weekend — weekend builds often less tested
df['is_weekend'] = (df['weekday'] >= 5).astype(int)

# 8. code_churn — total lines changed, normalized by repo size
#    Large churn relative to repo size = higher risk
df['code_churn'] = (df['lines_added'] + df['lines_deleted']) / (
    df['repository_lines'] + 1)

# 9. time_since_failure_normalized — normalize by repo age
#    Recent failures relative to repo lifetime
df['time_since_failure_norm'] = df['time_last_failed_build'] / (
    df['repository_age'] * 86400 + 1)

# 10. author_vs_project_failure — how much worse is this author than the project?
#     Positive = author fails more than average = risky contributor
df['author_vs_project'] = df['author_failure_rate'] - df['failure_rate']

print(f"  Added 10 new features")

# ── ORIGINAL + ENGINEERED FEATURES ───────────────────────────────────────────
ORIGINAL_FEATURES = [
    'files_pushed', 'lines_added', 'lines_deleted', 'commit_msg_len',
    'day_time', 'weekday', 'monthday', 'repository_age',
    'time_elapse', 'time_last_failed_build', 'last_build_result',
    'author_failure_rate', 'author_build_num', 'failure_rate',
    'failure_rate_recent', 'config_lines', 'config_warn', 'config_err',
    'action_lint_err', 'jobs_num', 'steps_num', 'repository_files',
    'repository_lines', 'repository_comments',
    'repository_owner_type', 'repository_language',
]

NEW_FEATURES = [
    'lines_changed_ratio', 'config_complexity', 'config_error_density',
    'author_experience_score', 'failure_momentum', 'is_peak_hour',
    'is_weekend', 'code_churn', 'time_since_failure_norm', 'author_vs_project',
]

ALL_FEATURES = ORIGINAL_FEATURES + NEW_FEATURES
print(f"  Total features: {len(ALL_FEATURES)} (was 26, now {len(ALL_FEATURES)})")

X = df[ALL_FEATURES].fillna(0)
y = df['label']

# ── FIXED WINDOW CV (matches paper methodology) ───────────────────────────────
INIT_WINDOW = 20000
HORIZON     = 3000
MTRY        = int(np.sqrt(len(ALL_FEATURES)))

print(f"\nFixed window CV with engineered features...")
print(f"  mtry = sqrt({len(ALL_FEATURES)}) = {MTRY}")

# Baseline RF (original 26 features) for fair comparison
X_orig = df[ORIGINAL_FEATURES].fillna(0)

baseline_results = []
engineered_results = []

for fold in range(4):
    train_start = fold * HORIZON
    train_end   = train_start + INIT_WINDOW
    test_start  = train_end
    test_end    = test_start + HORIZON
    if test_end > len(X):
        break

    # Baseline
    rf_base = RandomForestClassifier(
        n_estimators=500, max_features=int(np.sqrt(26)),
        random_state=123, n_jobs=-1)
    rf_base.fit(X_orig.iloc[train_start:train_end],
                y.iloc[train_start:train_end])
    bp = rf_base.predict(X_orig.iloc[test_start:test_end])
    baseline_results.append((
        accuracy_score(y.iloc[test_start:test_end], bp),
        precision_score(y.iloc[test_start:test_end], bp, zero_division=0),
        recall_score(y.iloc[test_start:test_end], bp, zero_division=0),
        f1_score(y.iloc[test_start:test_end], bp, zero_division=0)
    ))

    # Engineered
    rf_eng = RandomForestClassifier(
        n_estimators=500, max_features=MTRY,
        random_state=123, n_jobs=-1)
    rf_eng.fit(X.iloc[train_start:train_end],
               y.iloc[train_start:train_end])
    ep = rf_eng.predict(X.iloc[test_start:test_end])
    engineered_results.append((
        accuracy_score(y.iloc[test_start:test_end], ep),
        precision_score(y.iloc[test_start:test_end], ep, zero_division=0),
        recall_score(y.iloc[test_start:test_end], ep, zero_division=0),
        f1_score(y.iloc[test_start:test_end], ep, zero_division=0)
    ))

    print(f"  Fold {fold+1} Base : Acc={baseline_results[-1][0]:.4f} "
          f"Prec={baseline_results[-1][1]:.4f} "
          f"Rec={baseline_results[-1][2]:.4f} "
          f"F1={baseline_results[-1][3]:.4f}")
    print(f"  Fold {fold+1} Eng  : Acc={engineered_results[-1][0]:.4f} "
          f"Prec={engineered_results[-1][1]:.4f} "
          f"Rec={engineered_results[-1][2]:.4f} "
          f"F1={engineered_results[-1][3]:.4f}")
    print()

base_avg = [sum(x)/len(x) for x in zip(*baseline_results)]
eng_avg  = [sum(x)/len(x) for x in zip(*engineered_results)]

print("=" * 65)
print("RESULTS SUMMARY")
print("=" * 65)
print(f"{'Model':<30} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
print("-" * 65)
print(f"  {'Paper (RF, 26 features)':<28} "
      f"{'0.7804':>7} {'0.7532':>7} {'0.8376':>7} {'0.7927':>7}")
print(f"  {'Our RF (26 features)':<28} "
      f"{base_avg[0]:>7.4f} {base_avg[1]:>7.4f} "
      f"{base_avg[2]:>7.4f} {base_avg[3]:>7.4f}")
print(f"  {'Our RF (36 features)':<28} "
      f"{eng_avg[0]:>7.4f} {eng_avg[1]:>7.4f} "
      f"{eng_avg[2]:>7.4f} {eng_avg[3]:>7.4f}")
print(f"  {'Improvement':<28} "
      f"{eng_avg[0]-base_avg[0]:>+7.4f} {eng_avg[1]-base_avg[1]:>+7.4f} "
      f"{eng_avg[2]-base_avg[2]:>+7.4f} {eng_avg[3]-base_avg[3]:>+7.4f}")
print("=" * 65)

# ── Train final model on 80% ──────────────────────────────────────────────────
print("\nTraining final model on 80% of data...")
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
print("\nTop 15 feature importances:")
importances = pd.Series(final_rf.feature_importances_, index=ALL_FEATURES)
for feat, score in importances.sort_values(ascending=False).head(15).items():
    tag = " ← NEW" if feat in NEW_FEATURES else ""
    bar = '█' * int(score * 300)
    print(f"  {feat:<35} {score:.4f}  {bar}{tag}")

# ── Save ──────────────────────────────────────────────────────────────────────
print("\nSaving model...")
with open('model/rf_model.pkl', 'wb') as f:
    pickle.dump(final_rf, f)

with open('model/encoders.pkl', 'wb') as f:
    pickle.dump({
        'le_lang':           le_lang,
        'le_owner':          le_owner,
        'features':          ALL_FEATURES,
        'original_features': ORIGINAL_FEATURES,
        'new_features':      NEW_FEATURES,
        'result_is_binary':  True,
        'model_type':        'random_forest_engineered',
    }, f)

print("  Saved: model/rf_model.pkl")
print("  Saved: model/encoders.pkl")
print("\nDone!")

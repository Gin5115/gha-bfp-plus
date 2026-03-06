import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
import shap
import warnings
warnings.filterwarnings('ignore')

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':  'DejaVu Sans',
    'font.size':    11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi':   150,
})
COLORS = {'failure': '#e74c3c', 'success': '#2ecc71', 'neutral': '#3498db'}

# ── Load model and data ───────────────────────────────────────────────────────
print("Loading model and data...")
with open('model/rf_model.pkl', 'rb') as f:
    MODEL = pickle.load(f)
with open('model/encoders.pkl', 'rb') as f:
    ENC = pickle.load(f)

FEATURES          = ENC['features']
NEW_FEATURES      = ENC.get('new_features', [])
ORIGINAL_FEATURES = ENC.get('original_features', FEATURES)

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
df['repository_language']   = ENC['le_lang'].transform(
    df['repository_language'].fillna('Unknown'))
df['repository_owner_type'] = ENC['le_owner'].transform(df['repository_owner_type'])
df['label']                 = (df['result'] == 'failure').astype(int)

# Engineered features
df['lines_changed_ratio']      = df['lines_added'] / (df['lines_added'] + df['lines_deleted'] + 1)
df['config_complexity']        = df['jobs_num'] * df['steps_num']
df['config_error_density']     = df['config_err'] / (df['config_lines'] + 1)
df['author_experience_score']  = df['author_build_num'] / (df['repository_age'] + 1)
df['failure_momentum']         = (df['failure_rate'] + 2 * df['failure_rate_recent']) / 3
df['is_peak_hour']             = ((df['day_time'] >= 9) & (df['day_time'] <= 18)).astype(int)
df['is_weekend']               = (df['weekday'] >= 5).astype(int)
df['code_churn']               = (df['lines_added'] + df['lines_deleted']) / (df['repository_lines'] + 1)
df['time_since_failure_norm']  = df['time_last_failed_build'] / (df['repository_age'] * 86400 + 1)
df['author_vs_project']        = df['author_failure_rate'] - df['failure_rate']

split   = int(len(df) * 0.8)
X_test  = df[FEATURES].fillna(0).iloc[split:]
y_test  = df['label'].iloc[split:]
probs   = MODEL.predict_proba(X_test)[:, 1]
preds   = (probs >= 0.5).astype(int)

print(f"Test set: {len(X_test)} samples")
print(f"Acc={accuracy_score(y_test,preds):.4f}  "
      f"Prec={precision_score(y_test,preds):.4f}  "
      f"Rec={recall_score(y_test,preds):.4f}  "
      f"F1={f1_score(y_test,preds):.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/5] Confusion matrix...")
cm = confusion_matrix(y_test, preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
            xticklabels=['Predicted\nSuccess', 'Predicted\nFailure'],
            yticklabels=['Actual\nSuccess', 'Actual\nFailure'],
            linewidths=2, linecolor='white', ax=ax,
            annot_kws={'size': 14, 'weight': 'bold'})
ax.set_title('Confusion Matrix — GHA-BFP+ (36 Features)', pad=15, fontweight='bold')

tn, fp, fn, tp = cm.ravel()
ax.text(0.5, -0.12,
        f'TN={tn}  FP={fp}  FN={fn}  TP={tp}  |  '
        f'Acc={accuracy_score(y_test,preds):.3f}  '
        f'Prec={precision_score(y_test,preds):.3f}  '
        f'Rec={recall_score(y_test,preds):.3f}  '
        f'F1={f1_score(y_test,preds):.3f}',
        transform=ax.transAxes, ha='center', fontsize=9, color='#555')

plt.tight_layout()
plt.savefig('outputs/plots/1_confusion_matrix.png', bbox_inches='tight')
plt.close()
print("   Saved: outputs/plots/1_confusion_matrix.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — ROC Curve
# ══════════════════════════════════════════════════════════════════════════════
print("[2/5] ROC curve...")
fpr, tpr, thresholds = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color=COLORS['failure'], lw=2.5,
        label=f'GHA-BFP+ RF (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random classifier')
ax.fill_between(fpr, tpr, alpha=0.08, color=COLORS['failure'])

# Mark optimal threshold (closest to top-left corner)
optimal_idx = np.argmax(tpr - fpr)
ax.scatter(fpr[optimal_idx], tpr[optimal_idx],
           color=COLORS['failure'], s=100, zorder=5,
           label=f'Optimal threshold = {thresholds[optimal_idx]:.2f}')

ax.set_xlabel('False Positive Rate (1 - Specificity)')
ax.set_ylabel('True Positive Rate (Sensitivity / Recall)')
ax.set_title('ROC Curve — GHA-BFP+ Build Failure Prediction', fontweight='bold')
ax.legend(loc='lower right', framealpha=0.9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/plots/2_roc_curve.png', bbox_inches='tight')
plt.close()
print(f"   AUC = {roc_auc:.4f}")
print("   Saved: outputs/plots/2_roc_curve.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Feature Importances (Top 20)
# ══════════════════════════════════════════════════════════════════════════════
print("[3/5] Feature importances...")
importances = pd.Series(MODEL.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=True).tail(20)

colors = [COLORS['failure'] if f in NEW_FEATURES else COLORS['neutral']
          for f in importances.index]

fig, ax = plt.subplots(figsize=(9, 8))
bars = ax.barh(range(len(importances)), importances.values,
               color=colors, edgecolor='white', linewidth=0.5)

ax.set_yticks(range(len(importances)))
ax.set_yticklabels([
    f"★ {f}" if f in NEW_FEATURES else f
    for f in importances.index
], fontsize=9)

ax.set_xlabel('Feature Importance (Mean Decrease in Impurity)')
ax.set_title('Top 20 Feature Importances — GHA-BFP+\n'
             '(★ = Engineered features, 🔵 = Original paper features)',
             fontweight='bold')

from matplotlib.patches import Patch
legend = [
    Patch(color=COLORS['failure'], label='Engineered features (new)'),
    Patch(color=COLORS['neutral'], label='Original paper features'),
]
ax.legend(handles=legend, loc='lower right')
ax.grid(True, axis='x', alpha=0.3)
ax.set_xlim(0, importances.values.max() * 1.15)

for i, (val, feat) in enumerate(zip(importances.values, importances.index)):
    ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('outputs/plots/3_feature_importance.png', bbox_inches='tight')
plt.close()
print("   Saved: outputs/plots/3_feature_importance.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — SHAP Summary (Beeswarm)
# ══════════════════════════════════════════════════════════════════════════════
print("[4/5] SHAP beeswarm plot (this takes ~2 min)...")
sample_idx  = X_test.sample(500, random_state=42).index
X_sample    = X_test.loc[sample_idx]
explainer   = shap.TreeExplainer(MODEL)
shap_values = explainer.shap_values(X_sample)

# Extract failure class SHAP values
arr = np.array(shap_values)
if arr.ndim == 3:
    sv = arr[:, :, 1] if arr.shape[0] == len(X_sample) else arr[0, :, 1]
    if arr.shape[0] != len(X_sample):
        sv = arr[:, :, 1].squeeze()
        sv = np.array(shap_values)[:, :, 1] if np.array(shap_values).shape[2] == 2 else arr
else:
    sv = arr

# Recompute cleanly
raw = explainer.shap_values(X_sample)
if isinstance(raw, list):
    sv_clean = raw[1]
else:
    sv_clean = np.array(raw)
    if sv_clean.ndim == 3:
        sv_clean = sv_clean[:, :, 1]

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    sv_clean, X_sample,
    feature_names=FEATURES,
    show=False,
    plot_size=None,
    max_display=15,
    color_bar_label='Feature value'
)
plt.title('SHAP Beeswarm Plot — Feature Impact on Failure Prediction\n'
          '(Red = high feature value, Blue = low feature value)',
          fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('outputs/plots/4_shap_beeswarm.png', bbox_inches='tight')
plt.close()
print("   Saved: outputs/plots/4_shap_beeswarm.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Model Comparison Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
print("[5/5] Model comparison chart...")
models = ['Paper\n(RF 26f)', 'Our RF\n(26f)', 'Our RF\n(36f, tuned)']
metrics = {
    'Accuracy':  [0.7804, 0.7762, accuracy_score(y_test, preds)],
    'Precision': [0.7532, 0.8087, precision_score(y_test, preds)],
    'Recall':    [0.8376, 0.7281, recall_score(y_test, preds)],
    'F1':        [0.7927, 0.7653, f1_score(y_test, preds)],
}

x     = np.arange(len(models))
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))
metric_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

for i, (metric, values) in enumerate(metrics.items()):
    bars = ax.bar(x + i * width, values, width,
                  label=metric, color=metric_colors[i],
                  alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylabel('Score')
ax.set_ylim(0.65, 0.95)
ax.set_title('Model Comparison — GHA-BFP+ vs Paper Baseline',
             fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, axis='y', alpha=0.3)
ax.axhline(y=0.7927, color='gray', linestyle='--',
           alpha=0.5, label='Paper F1 baseline')

plt.tight_layout()
plt.savefig('outputs/plots/5_model_comparison.png', bbox_inches='tight')
plt.close()
print("   Saved: outputs/plots/5_model_comparison.png")

print("\n✅ All plots saved to outputs/plots/")
print("   1_confusion_matrix.png")
print("   2_roc_curve.png")
print("   3_feature_importance.png")
print("   4_shap_beeswarm.png")
print("   5_model_comparison.png")

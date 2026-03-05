import os
import sys
import json
import math
import requests
import yaml
from datetime import datetime
from github import Github

# ── Environment ───────────────────────────────────────────────────────────────
GITHUB_TOKEN      = os.environ.get("GITHUB_TOKEN", "")
PREDICTOR_API_URL = os.environ.get("PREDICTOR_API_URL", "").rstrip("/")
PR_NUMBER         = int(os.environ.get("PR_NUMBER", "0"))
REPO_NAME         = os.environ.get("REPO_NAME", "")

if not all([GITHUB_TOKEN, PREDICTOR_API_URL, PR_NUMBER, REPO_NAME]):
    print("ERROR: Missing required environment variables.")
    sys.exit(1)

# ── GitHub client ─────────────────────────────────────────────────────────────
g    = Github(GITHUB_TOKEN)
repo = g.get_repo(REPO_NAME)
pr   = repo.get_pull(PR_NUMBER)

print(f"Processing PR #{PR_NUMBER}: {pr.title}")

# ── Helper ────────────────────────────────────────────────────────────────────
def safe(val, default=0):
    try:
        return float(val) if val is not None else float(default)
    except:
        return float(default)


# ── 1. CURRENT BUILD FEATURES ────────────────────────────────────────────────
files_pushed   = pr.changed_files
lines_added    = pr.additions
lines_deleted  = pr.deletions
commit_msg_len = len(pr.title) + len(pr.body or "")

now       = datetime.utcnow()
day_time  = now.hour
weekday   = now.weekday()       # 0=Monday, 6=Sunday
monthday  = now.day

# Repository age in days
repo_created = repo.created_at
repo_age     = (now - repo_created.replace(tzinfo=None)).days + \
               (now - repo_created.replace(tzinfo=None)).seconds / 86400

print(f"  Files changed: {files_pushed}, +{lines_added}/-{lines_deleted}")

# ── 2. HISTORICAL BUILD FEATURES ─────────────────────────────────────────────
# Fetch last 50 workflow runs for this repo
runs      = list(repo.get_workflow_runs()[:50])
total     = len(runs)
failures  = [r for r in runs if r.conclusion == 'failure']
successes = [r for r in runs if r.conclusion == 'success']

# Overall failure rate
failure_rate = len(failures) / total if total > 0 else 0.0

# Recent failure rate (last 5 runs)
recent_runs         = runs[:5]
recent_failures     = [r for r in recent_runs if r.conclusion == 'failure']
failure_rate_recent = len(recent_failures) / len(recent_runs) if recent_runs else 0.0

# Last build result
last_build_result = runs[0].conclusion if runs else 'success'
if last_build_result not in ['success', 'failure', 'cancelled',
                              'skipped', 'startup_failure']:
    last_build_result = 'success'

# Time since last run (seconds)
time_elapse = 0
if len(runs) >= 2:
    t1 = runs[0].created_at.replace(tzinfo=None)
    t2 = runs[1].created_at.replace(tzinfo=None)
    time_elapse = abs((t1 - t2).total_seconds())

# Time since last failure (seconds)
time_last_failed_build = 9999999
if failures:
    last_fail_time = failures[0].created_at.replace(tzinfo=None)
    time_last_failed_build = (now - last_fail_time).total_seconds()

# Author failure rate
pr_author = pr.user.login
author_runs     = [r for r in runs if r.head_commit and
                   r.head_commit.author and
                   r.head_commit.author.get('name', '') != '']
author_failures = [r for r in author_runs if r.conclusion == 'failure']
author_build_num     = len(author_runs)
author_failure_rate  = len(author_failures) / author_build_num \
                       if author_build_num > 0 else 0.0

print(f"  failure_rate={failure_rate:.3f}, "
      f"failure_rate_recent={failure_rate_recent:.3f}, "
      f"last_build={last_build_result}")

# ── 3. CONFIGURATION FILE FEATURES ───────────────────────────────────────────
config_lines    = 0
config_warn     = 0
config_err      = 0
action_lint_err = 0
jobs_num        = 1
steps_num       = 1

# Find workflow YAML files in the PR
workflow_files = [
    f for f in pr.get_files()
    if f.filename.startswith('.github/workflows/') and
    f.filename.endswith(('.yml', '.yaml'))
]

# Also check existing workflows in repo
try:
    existing_workflows = repo.get_contents(".github/workflows")
    if not isinstance(existing_workflows, list):
        existing_workflows = [existing_workflows]
except:
    existing_workflows = []

# Parse the first workflow file found
yaml_content = None
if workflow_files:
    try:
        raw = requests.get(workflow_files[0].raw_url).text
        yaml_content = yaml.safe_load(raw)
        config_lines = len(raw.splitlines())
        # Count YAML warnings (common anti-patterns)
        config_warn = raw.count('continue-on-error: true')
        config_err  = raw.count('!!') + raw.count('undefined')
    except:
        pass
elif existing_workflows:
    try:
        raw = existing_workflows[0].decoded_content.decode('utf-8')
        yaml_content = yaml.safe_load(raw)
        config_lines = len(raw.splitlines())
        config_warn  = raw.count('continue-on-error: true')
        config_err   = raw.count('!!') + raw.count('undefined')
    except:
        pass

# Count jobs and steps
if yaml_content and isinstance(yaml_content, dict):
    jobs = yaml_content.get('jobs', {})
    if isinstance(jobs, dict):
        jobs_num  = len(jobs)
        steps_num = sum(
            len(job.get('steps', []))
            for job in jobs.values()
            if isinstance(job, dict)
        )
        steps_num = max(steps_num, 1)

print(f"  config_lines={config_lines}, jobs={jobs_num}, steps={steps_num}")

# ── 4. REPOSITORY FEATURES ───────────────────────────────────────────────────
repository_files    = repo.get_git_tree(
    pr.head.sha, recursive=True
).tree
repository_files    = len(repository_files)
repository_lines    = repo.size * 100   # rough estimate: KB * 100
repository_comments = 0                 # not directly available via API
repository_language = repo.language or 'Python'
repository_owner_type = 'Organization' \
    if repo.organization else 'User'

print(f"  repo_files={repository_files}, "
      f"language={repository_language}, "
      f"owner={repository_owner_type}")

# ── 5. Call prediction API ────────────────────────────────────────────────────
payload = {
    "files_pushed":           safe(files_pushed),
    "lines_added":            safe(lines_added),
    "lines_deleted":          safe(lines_deleted),
    "commit_msg_len":         safe(commit_msg_len),
    "day_time":               safe(day_time),
    "weekday":                safe(weekday),
    "monthday":               safe(monthday),
    "repository_age":         safe(repo_age),
    "time_elapse":            safe(time_elapse),
    "time_last_failed_build": safe(time_last_failed_build),
    "last_build_result":      last_build_result,
    "author_failure_rate":    safe(author_failure_rate),
    "author_build_num":       safe(author_build_num),
    "failure_rate":           safe(failure_rate),
    "failure_rate_recent":    safe(failure_rate_recent),
    "config_lines":           safe(config_lines),
    "config_warn":            safe(config_warn),
    "config_err":             safe(config_err),
    "action_lint_err":        safe(action_lint_err),
    "jobs_num":               safe(jobs_num),
    "steps_num":              safe(steps_num),
    "repository_files":       safe(repository_files),
    "repository_lines":       safe(repository_lines),
    "repository_comments":    safe(repository_comments),
    "repository_owner_type":  repository_owner_type,
    "repository_language":    repository_language,
}

print(f"\nCalling API: {PREDICTOR_API_URL}/predict-explain?llm=gemini")
response = requests.post(
    f"{PREDICTOR_API_URL}/predict-explain?llm=gemini",
    json=payload,
    timeout=60
)
response.raise_for_status()
result = response.json()

print(f"  Result: {result['prediction']} "
      f"({result['failure_probability']*100:.1f}% failure probability)")

# ── 6. Format PR comment ──────────────────────────────────────────────────────
pred        = result['prediction']
prob        = result['failure_probability']
confidence  = result['confidence']
top_feats   = result.get('top_features', {})
explanation = result.get('explanation', '')

# Risk emoji and color
if prob >= 0.7:
    emoji  = "🔴"
    risk   = "HIGH RISK"
elif prob >= 0.4:
    emoji  = "🟡"
    risk   = "MEDIUM RISK"
else:
    emoji  = "🟢"
    risk   = "LOW RISK"

# Visual probability bar (20 blocks)
filled = round(prob * 20)
bar    = "🟥" * filled + "⬜" * (20 - filled)

# Top features table
feat_rows = ""
for feat, val in top_feats.items():
    direction = "⬆️ toward failure" if val > 0 else "⬇️ toward success"
    feat_rows += f"| `{feat}` | `{val:+.4f}` | {direction} |\n"

comment = f"""## {emoji} GHA-BFP+ Build Failure Prediction

| | |
|---|---|
| **Prediction** | `{pred.upper()}` |
| **Failure Probability** | `{prob*100:.1f}%` |
| **Confidence** | `{confidence*100:.1f}%` |
| **Risk Level** | **{risk}** |

**Failure Probability:** `{prob*100:.1f}%`
{bar}

---

### 🔍 Top Contributing Factors (SHAP)

| Feature | SHAP Value | Influence |
|---|---|---|
{feat_rows}
> SHAP values show how much each feature pushed the prediction toward or away from failure.

---

### 🤖 AI Explanation (Gemini)

> {explanation}

---

### 📊 Build Context Collected

| Feature | Value |
|---|---|
| Files changed | `{files_pushed}` |
| Lines added | `{lines_added}` |
| Lines deleted | `{lines_deleted}` |
| Failure rate (overall) | `{failure_rate:.3f}` |
| Failure rate (recent 5) | `{failure_rate_recent:.3f}` |
| Last build result | `{last_build_result}` |
| Author failure rate | `{author_failure_rate:.3f}` |
| Config lines | `{config_lines}` |
| Jobs in workflow | `{jobs_num}` |
| Steps in workflow | `{steps_num}` |

---
<sub>🤖 Powered by GHA-BFP+ | Random Forest + SHAP + Gemini | Based on Li et al. APSEC 2024</sub>
"""

# ── 7. Post comment to PR ─────────────────────────────────────────────────────
pr.create_issue_comment(comment)
print(f"\nComment posted to PR #{PR_NUMBER} ✅")

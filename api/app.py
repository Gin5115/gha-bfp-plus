from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import pickle
import numpy as np
import os

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(ROOT, 'model', 'rf_model.pkl')
ENCODER_PATH = os.path.join(ROOT, 'model', 'encoders.pkl')

# ── Load model and encoders ───────────────────────────────────────────────────
with open(MODEL_PATH, 'rb') as f:
    MODEL = pickle.load(f)

with open(ENCODER_PATH, 'rb') as f:
    ENC = pickle.load(f)

FEATURES = ENC['features']

print(f"Model loaded — {len(FEATURES)} features")
print(f"Languages known: {len(ENC['le_lang'].classes_)}")
print(f"Owner types: {ENC['le_owner'].classes_}")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="GHA-BFP+ API",
    description="GitHub Actions Build Failure Prediction with Explainability",
    version="1.0.0"
)

# ── Request schema ────────────────────────────────────────────────────────────
class BuildFeatures(BaseModel):
    # current build
    files_pushed:           float = 0
    lines_added:            float = 0
    lines_deleted:          float = 0
    commit_msg_len:         float = 0
    day_time:               float = 0
    weekday:                float = 0
    monthday:               float = 1
    repository_age:         float = 0
    # historical build
    time_elapse:            float = 0
    time_last_failed_build: float = 9999999
    last_build_result:      str   = 'success'
    author_failure_rate:    float = 0
    author_build_num:       float = 0
    failure_rate:           float = 0
    failure_rate_recent:    float = 0
    # configuration file
    config_lines:           float = 0
    config_warn:            float = 0
    config_err:             float = 0
    action_lint_err:        float = 0
    jobs_num:               float = 1
    steps_num:              float = 1
    # repository
    repository_files:       float = 0
    repository_lines:       float = 0
    repository_comments:    float = 0
    repository_owner_type:  str   = 'User'
    repository_language:    str   = 'Python'


# ── Encoding ──────────────────────────────────────────────────────────────────
def encode_features(data: dict) -> np.ndarray:
    d = data.copy()

    # last_build_result — binary, matches training exactly
    d['last_build_result'] = 1 if d['last_build_result'] == 'failure' else 0

    # repository_language
    try:
        d['repository_language'] = int(
            ENC['le_lang'].transform([d['repository_language']])[0]
        )
    except ValueError:
        d['repository_language'] = 0

    # repository_owner_type
    try:
        d['repository_owner_type'] = int(
            ENC['le_owner'].transform([d['repository_owner_type']])[0]
        )
    except ValueError:
        d['repository_owner_type'] = 1

    return np.array([[d[f] for f in FEATURES]])


# ── SHAP extractor ────────────────────────────────────────────────────────────
def get_shap_values(explainer, x: np.ndarray) -> np.ndarray:
    """Extract SHAP values — shape is (1, 26, 2) for this shap version."""
    shap_vals = explainer.shap_values(x)
    arr = np.array(shap_vals)

    if arr.ndim == 3:
        # shape: (n_samples, n_features, n_classes) — index 1 = failure class
        return arr[0, :, 1]
    elif arr.ndim == 2:
        # shape: (n_samples, n_features)
        return arr[0, :]
    else:
        return arr.flatten()


# ── LLM explanation ───────────────────────────────────────────────────────────
def get_llm_explanation(
    prob: float,
    top_features: dict,
    provider: str = "gemini"
) -> str:
    """
    Generate human-readable explanation using an LLM.
    Supports: gemini (free), anthropic (paid), none (rule-based)
    """

    top_str = "\n".join(
        f"  - {feat}: SHAP value {val:+.4f} "
        f"({'pushes toward failure' if val > 0 else 'pushes toward success'})"
        for feat, val in top_features.items()
    )

    prompt = (
        f"A GitHub Actions build has a {prob*100:.1f}% predicted failure probability.\n\n"
        f"The top contributing factors (SHAP values) are:\n{top_str}\n\n"
        f"In 2-3 sentences:\n"
        f"1. Explain in plain English why this build is likely to fail.\n"
        f"2. Suggest one concrete action the developer can take to reduce the risk.\n"
        f"Be specific and practical. Do not use jargon."
    )

    if provider == "gemini":
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"[Gemini unavailable: {e}] Top factors: {', '.join(top_features.keys())}"

    elif provider == "anthropic":
        try:
            import anthropic
            client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", "")
            )
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"[Anthropic unavailable: {e}] Top factors: {', '.join(top_features.keys())}"

    else:
        # No LLM — rule-based fallback
        top_feat  = list(top_features.keys())[0]
        top_val   = list(top_features.values())[0]
        direction = "high" if top_val > 0 else "low"
        return (
            f"This build has a {prob*100:.1f}% failure probability. "
            f"The strongest signal is a {direction} '{top_feat}' value. "
            f"Review recent build history and workflow configuration to reduce risk."
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name":      "GHA-BFP+ API",
        "status":    "running",
        "endpoints": ["/predict", "/predict-explain", "/health"]
    }


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": MODEL is not None,
        "n_features":   len(FEATURES)
    }


@app.post("/predict")
def predict(features: BuildFeatures):
    """Fast prediction — returns label and probability only."""
    try:
        x    = encode_features(features.model_dump())
        prob = float(MODEL.predict_proba(x)[0][1])
        return {
            "prediction":          "failure" if prob > 0.5 else "success",
            "failure_probability": round(prob, 4),
            "confidence":          round(abs(prob - 0.5) * 2, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-explain")
def predict_explain(
    features: BuildFeatures,
    llm: str = "none"
):
    """
    Prediction + SHAP top features + optional LLM explanation.
    Pass ?llm=gemini    → uses Gemini 1.5 Flash (free tier)
    Pass ?llm=anthropic → uses Claude Sonnet (paid)
    Pass ?llm=none      → rule-based explanation, no API key needed
    """
    try:
        import shap

        x    = encode_features(features.model_dump())
        prob = float(MODEL.predict_proba(x)[0][1])

        # SHAP
        explainer    = shap.TreeExplainer(MODEL)
        sv           = get_shap_values(explainer, x)

        top5 = sorted(
            zip(FEATURES, sv.tolist()),
            key=lambda t: abs(t[1]),
            reverse=True
        )[:5]

        top_features = {f: round(float(v), 4) for f, v in top5}

        # LLM explanation
        explanation = get_llm_explanation(prob, top_features, provider=llm)

        return {
            "prediction":          "failure" if prob > 0.5 else "success",
            "failure_probability": round(prob, 4),
            "confidence":          round(abs(prob - 0.5) * 2, 4),
            "top_features":        top_features,
            "explanation":         explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# app.py
import os
import traceback
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc

print(sklearn.__version__)

MODEL_PATH = os.environ.get("IDS_MODEL_PATH", "C:\\Users\\HarshaSri\\Desktop\\IDS_PROJECT\\models\\xgboost.joblib")
PREPROCESSOR_PATH = os.environ.get("IDS_PREPROCESSOR_PATH", "C:\\Users\\HarshaSri\\Desktop\\IDS_PROJECT\\data\\processed\\preprocessor.joblib")
TEST_PRED_OUT = os.environ.get("IDS_TEST_PRED_OUT", "C:\\Users\\HarshaSri\\Desktop\\IDS_PROJECT\\models\\test_predictions.csv")
app = Flask(__name__)
CORS(app)

# Lazy load model & preprocessor
_model = None
_preprocessor = None

def load_model_and_preprocessor():
    global _model, _preprocessor
    if _preprocessor is None:
        if not os.path.exists(PREPROCESSOR_PATH):
            raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}")
        _preprocessor = joblib.load(PREPROCESSOR_PATH)
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model, _preprocessor

@app.route("/health", methods=["GET"])
def health():
    """Simple health check."""
    try:
        load_model_and_preprocessor()
        return jsonify({"status": "ok", "model": os.path.basename(MODEL_PATH)}), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict_single():
    """
    Accepts JSON body with feature names as keys (matching training columns),
    returns predicted label and probability (if available).
    Example:
    {
      "duration": 0,
      "protocol_type": "tcp",
      ...
    }
    """
    try:
        model, preproc = load_model_and_preprocessor()
        data = request.json
        if data is None:
            return jsonify({"error": "Invalid JSON body"}), 400
        # single-row DataFrame
        df = pd.DataFrame([data])
        # transform
        X = preproc.transform(df)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[:, 1][0]
            label = int(prob >= 0.5)
        else:
            label = int(model.predict(X)[0])
            prob = None
        return jsonify({"label": int(label), "probability": float(prob) if prob is not None else None}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    Batch prediction via file upload (form-data 'file' field) with CSV.
    Input CSV must have the *raw features* (same columns as train before preprocessing).
    Returns a CSV file with added columns: pred and prob (if available).
    """
    try:
        model, preproc = load_model_and_preprocessor()
        if "file" not in request.files:
            return jsonify({"error": "Missing file part (form field name 'file')"}), 400
        file = request.files["file"]
        df = pd.read_csv(file)
        if df.empty:
            return jsonify({"error": "Empty CSV uploaded"}), 400
        X = preproc.transform(df)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
            preds = (probs >= 0.5).astype(int)
        else:
            preds = model.predict(X)
            probs = [None] * len(preds)
        out = df.copy()
        out["pred"] = preds
        out["prob"] = probs
        # Save server-side copy
        out.to_csv(TEST_PRED_OUT, index=False)
        # Return the CSV as attachment
        return send_file(TEST_PRED_OUT, mimetype="text/csv", as_attachment=True, download_name="predictions.csv")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/metrics", methods=["GET"])
def metrics():
    """
    If you have ground-truth labels for the test set saved on the server (y_test),
    this endpoint will compute classification metrics. It expects:
    - preprocessed data_splits.joblib to exist at PREPROCESSOR_DIR/data_splits.joblib
    or you can pass `y_true` & `y_pred` via POST in a different implementation.
    """
    try:
        # Try to find saved splits
        splits_path = os.environ.get("IDS_DATA_SPLITS", "data/processed/data_splits.joblib")
        if not os.path.exists(splits_path):
            return jsonify({"error": f"data_splits not found at {splits_path}"}), 400
        X_train_bal, y_train_bal, X_test_t, y_test = joblib.load(splits_path)
        if y_test is None:
            return jsonify({"error": "No ground-truth labels available for test set (y_test is None)"}), 400
        # load model
        model, preproc = load_model_and_preprocessor()
        # predict on X_test_t (already preprocessed)
        try:
            probs = model.predict_proba(X_test_t)[:, 1]
            preds = (probs >= 0.5).astype(int)
        except Exception:
            preds = model.predict(X_test_t)
            probs = None
        report = classification_report(y_test, preds, output_dict=True)
        cm = confusion_matrix(y_test, preds).tolist()
        response = {"classification_report": report, "confusion_matrix": cm}
        if probs is not None:
            response["roc_auc"] = float(roc_auc_score(y_test, probs))
            precision, recall, thresholds = precision_recall_curve(y_test, probs)
            response["pr_auc"] = float(auc(recall, precision))
        return jsonify(response), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # For local dev
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)

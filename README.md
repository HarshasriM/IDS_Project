# Intrusion Detection System (IDS)

**Project**: ML-based Intrusion Detection System using NSL-KDD (KDD99) features  
**Author**: Your Name — add your GitHub handle  
**Status**: Prototype — preprocessing, EDA, baseline & XGBoost models, Flask API and Streamlit UI included

---

## Overview

This repository implements an end-to-end IDS prototype using machine learning on the **NSL-KDD** feature schema (41 features). It includes:

- Data preprocessing pipeline (encoding, scaling, balancing)  
- Exploratory Data Analysis (PCA, t-SNE, correlations, feature importance)  
- Model training (Logistic Regression, RandomForest, XGBoost example)  
- Model evaluation and thresholding (ROC/PR)  
- Flask backend (`app.py`) for single & batch predictions  
- Streamlit frontend (`streamlit_app.py`) for manual testing / demo  
- Notebooks and simple scripts to reproduce experiments

This repo is intended for research/education and local testing only — do **not** expose the API to the public without authentication and hardening.

---

## Repository structure

```
ids-project/
├─ data/
│  ├─ raw/                   # place Train_data.csv, Test_data.csv here
│  └─ processed/             # preprocessor.joblib, data_splits.joblib saved here
├─ models/                   # saved models (.joblib)
├─ notebooks/
│  ├─ 00_data_preprocessing.ipynb
│  ├─ 01_eda.ipynb
│  ├─ 02_training_models.ipynb
│  └─ 03_testing_and_deployment.ipynb
├─ src/                      # optional scripts (preprocess.py, train.py)
├─ app.py                    # Flask API
├─ streamlit_app.py          # Streamlit demo UI          
├─ requirements.txt
├─ Dockerfile                # optional
└─ README.md
```

---

## Quick start (local, recommended)

> These commands assume you are in the repository root.

### 1. Create environment (recommended: conda)

```bash
# create & activate
conda create -n ids-env python=3.10 -y
conda activate ids-env

# install core packages
conda install -y numpy pandas scikit-learn matplotlib seaborn joblib jupyterlab

# install other libs (conda-forge recommended)
conda install -y -c conda-forge imbalanced-learn xgboost lightgbm

# install Flask & Streamlit
pip install flask flask-cors streamlit requests
```

Or using `venv` + `pip`:

```bash
python -m venv venv
# activate venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Add your data

Place your CSV files in `data/raw/`:

- `data/raw/Train_data.csv` — training CSV (NSL-KDD schema)  
- `data/raw/Test_data.csv` — test CSV (optional labels)

The repository was developed using the NSL-KDD / KDD99 feature set (41 features + label). If your CSV uses different names, either rename columns or adapt the preprocessing code.

---

### 3. Run notebooks or scripts

Open Jupyter Lab and run notebooks in order:

```bash
jupyter lab
# then run in this order:
# 1) notebooks/00_data_preprocessing.ipynb
# 2) notebooks/01_eda.ipynb
# 3) notebooks/02_training_models.ipynb
# 4) notebooks/03_testing_and_deployment.ipynb
```

Each notebook is self-contained and saves artifacts to `data/processed/` and `models/`.

---

### 4. Run Flask API

Make sure `data/processed/preprocessor.joblib` and a model (e.g., `models/rf.joblib`) exist.

```bash
python app.py
# Flask runs on http://0.0.0.0:5000 by default
```

Endpoints:
- `GET /health` — health check  
- `POST /predict` — single JSON sample prediction  
- `POST /predict_batch` — CSV upload (multipart/form-data) returns CSV with `pred` & `prob`  
- `GET /metrics` — evaluation metrics if `y_test` present in `data_splits.joblib`

Example `curl` (single instance):
```bash
curl -X POST http://localhost:5000/predict   -H "Content-Type: application/json"   -d '{"duration":0,"protocol_type":"tcp","service":"http","flag":"SF", ... }'
```

---

### 5. Run Streamlit UI

In another terminal:

```bash
streamlit run streamlit_app.py --server.port 8501
# open http://localhost:8501
```

Use the “Single instance” mode (paste JSON) or “Batch CSV” to upload a CSV and get predictions.

---

## Notes on model & preprocessing compatibility

- If you see unpickling errors like:
  ```
  Can't get attribute '_RemainderColsList' on <module 'sklearn.compose._column_transformer' ...>
  ```
  this usually indicates a **scikit-learn version mismatch** between the environment that saved `preprocessor.joblib` and the current environment that tries to `joblib.load` it.

  **Fixes:**
  1. Install the same scikit-learn version used for saving. Example (in notebook):
     ```python
     import sys
     !{sys.executable} -m pip install scikit-learn==1.7.2 --upgrade
     ```
     then restart kernel/process.
  2. Or, **rebuild the preprocessor** in the target environment by running `00_data_preprocessing.ipynb` on the original training CSV and saving a fresh `preprocessor.joblib`.

- Always save the environment spec:
  - `pip freeze > requirements.txt` or `conda env export > environment.yml` so collaborators can reproduce.

---

## Dummy test sample (single JSON)

Use this JSON to test a normal record:

```json
{
  "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF",
  "src_bytes": 215, "dst_bytes": 45076, "land": 0, "wrong_fragment": 0,
  "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 1,
  "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
  "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
  "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
  "count": 9, "srv_count": 9, "serror_rate": 0.0, "srv_serror_rate": 0.0,
  "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
  "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 9,
  "dst_host_srv_count": 9, "dst_host_same_srv_rate": 1.0,
  "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 1.0,
  "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
  "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
  "dst_host_srv_rerror_rate": 0.0, "class": "normal"
}
```

---

## Evaluation & interpretation guidance

- **ROC-AUC** is a good global metric. Report **precision/recall/F1** and **per-attack** performance for a more complete picture.
- Top reported AUCs on NSL-KDD often exceed 0.99 for state-of-the-art models; very high AUCs can still hide high false-positive rates in practice — always inspect confusion matrices and PR curves.
- For operational IDS, tune thresholds to control False Positive Rate (FPR) and measure alerts per hour/day.

---

## References

- Tavallaee et al., “A detailed analysis of the KDD Cup 99 data set.” (NSL-KDD) — commonly used corpus.  
- NSL-KDD dataset (public): many research papers leverage this benchmark.  

---

## Next steps (suggested)

- Add experiment logging (W&B or MLflow).  
- Try window/flow-level features (aggregate packets per source IP/time window).  
- Evaluate model on a real traffic capture (pcap) converted to NSL-KDD features.  
- Hardening: add authentication & rate limiting to the Flask API before exposing it.
---

## License

Add your license of choice (MIT / Apache 2.0 / BSD). Example:

```
MIT License
```

---

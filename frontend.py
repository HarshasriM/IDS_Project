# streamlit_app.py
import streamlit as st
import pandas as pd
import requests
from io import StringIO, BytesIO

API_URL = "http://localhost:5000"

st.set_page_config(page_title="IDS Demo", layout="wide")
st.title("Intrusion Detection System — Demo UI")

st.sidebar.header("Options")
mode = st.sidebar.radio("Mode", ["Single instance", "Batch CSV"])

if mode == "Single instance":
    st.subheader("Single instance prediction")
    st.write("Provide feature values (keys must match your training features).")
    # For simplicity, allow user to paste JSON-like dict or CSV-style single-row
    raw_input = st.text_area("Paste single-row JSON (e.g. {\"duration\":0, \"protocol_type\":\"tcp\", ...})", height=180)
    if st.button("Predict single"):
        if not raw_input.strip():
            st.error("Please paste a JSON object representing a single sample.")
        else:
            try:
                import json
                data = json.loads(raw_input)
                resp = requests.post(f"{API_URL}/predict", json=data, timeout=30)
                if resp.status_code == 200:
                    st.success("Prediction received")
                    st.json(resp.json())
                else:
                    st.error(f"Error: {resp.text}")
            except Exception as e:
                st.error(f"Invalid JSON or request error: {e}")

else:
    st.subheader("Batch predictions (CSV upload)")
    st.markdown("Upload a CSV with the same feature columns used in training (raw features, not preprocessed).")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write("Preview of uploaded file:", df.head())
            if st.button("Send to API for batch prediction"):
                files = {"file": ("upload.csv", uploaded.getvalue())}
                resp = requests.post(f"{API_URL}/predict_batch", files=files, timeout=120)
                if resp.status_code == 200:
                    st.success("Predictions ready — downloading results")
                    # Streamlit can't save attachments directly from requests easily; we will show table
                    # Save to local temp buffer then load
                    out_df = pd.read_csv(BytesIO(resp.content))
                    st.write("Predictions preview:", out_df.head())
                    # Provide download
                    csv = out_df.to_csv(index=False).encode()
                    st.download_button("Download predictions CSV", csv, "predictions.csv", "text/csv")
                else:
                    st.error(f"Batch predict failed: {resp.text}")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

st.markdown("---")
st.write("Notes: This demo expects the backend Flask API to be running and accessible.")

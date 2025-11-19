import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.title("🧠 Model Explainability (SHAP)")
st.markdown(
    "<h4 style='color:gray;margin-top:-10px;'>Tree-based SHAP Explanation for Anomaly Detection</h4><br>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------
# Load model artifacts
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    pipeline = joblib.load("model.pkl")
    label_enc = joblib.load("label_encoder.pkl")
    return pipeline, label_enc

pipeline, label_enc = load_model()

# ----------------------------------------------------------
# Ensure history exists
# ----------------------------------------------------------
if "history" not in st.session_state or len(st.session_state.history) == 0:
    st.warning("⚠ No data yet. Visit System Overview first.")
    st.stop()

df = pd.DataFrame(st.session_state.history)

drop_cols = ["prediction", "timestamp", "vm_id"]
input_cols = [c for c in df.columns if c not in drop_cols]

sample = df[input_cols].tail(1)

st.write("### 📌 Sample Row Used")
st.dataframe(sample, use_container_width=True)

# ----------------------------------------------------------
# Extract model + preprocessor
# ----------------------------------------------------------
preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

# Transform sample & full data
X_full = df[input_cols]
X_trans = preprocessor.transform(X_full)

# Convert sparse → dense
if hasattr(X_trans, "toarray"):
    X_trans = X_trans.toarray()

# Extract feature names
try:
    feature_names = preprocessor.get_feature_names_out()
except Exception:
    feature_names = [f"f{i}" for i in range(X_trans.shape[1])]

# ----------------------------------------------------------
# Create TreeExplainer
# ----------------------------------------------------------
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)
except Exception as e:
    st.error("SHAP cannot explain this model.")
    st.info(f"Error: {e}")
    st.stop()

# ----------------------------------------------------------
# GLOBAL FEATURE IMPORTANCE (Bar Plot)
# ----------------------------------------------------------
st.write("### 🌍 Global Feature Importance")

fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    X_trans,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)
st.pyplot(fig)

# ----------------------------------------------------------
# SHAP SUMMARY (Bee Swarm)
# ----------------------------------------------------------
st.write("### 🐝 SHAP Summary (Distribution Across Samples)")

fig2, ax2 = plt.subplots(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    X_trans,
    feature_names=feature_names,
    show=False
)
st.pyplot(fig2)

# ----------------------------------------------------------
# SAMPLE-LEVEL FEATURE CONTRIBUTION (Top 15)
# ----------------------------------------------------------
st.write("### 📊 Feature Contributions for This Prediction")

sample_index = len(X_trans) - 1
sample_shap = shap_values[sample_index]

# Create sorted contribution table
contrib_df = pd.DataFrame({
    "Feature": feature_names,
    "SHAP Value": sample_shap
}).sort_values("SHAP Value", ascending=False).head(15)

st.dataframe(contrib_df, use_container_width=True)

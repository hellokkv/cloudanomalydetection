import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

st.title("📊 Model Insights")

model = joblib.load("model.pkl")

try:
    importances = model.named_steps["model"].feature_importances_
    pre = model.named_steps["preprocessor"]
    feat_names = pre.get_feature_names_out()

    df_imp = pd.DataFrame({
        "Feature": feat_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    st.write("### 🔍 Feature Importance")
    fig = px.bar(df_imp.head(20), x="Importance", y="Feature", orientation='h')
    st.plotly_chart(fig, use_container_width=True)

except Exception:
    st.warning("This model does not provide feature importances.")

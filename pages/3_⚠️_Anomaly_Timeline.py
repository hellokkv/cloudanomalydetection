import streamlit as st
import pandas as pd
import plotly.express as px

st.title("⚠️ Anomaly Timeline")

data = pd.DataFrame(st.session_state.history)

anom = data[data["prediction"] == 1]

if anom.empty:
    st.success("No anomalies detected yet.")
else:
    st.write("### 🔥 Timeline of Anomaly Events")
    fig = px.scatter(
        anom,
        x="timestamp",
        y="cpu_usage",
        color="task_priority",
        size="network_traffic",
        title="Anomaly Timeline",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("### Detailed Anomaly Log")
    st.dataframe(anom.tail(50), use_container_width=True)

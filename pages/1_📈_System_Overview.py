import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import plotly.express as px
# ----------------------------------------------------------
# INITIALIZE RUNTIME CLOCK
# ----------------------------------------------------------
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()   # store start timestamp

# ----------------------------------------------------------
# PAGE TITLE
# ----------------------------------------------------------
st.title("📈 System Overview")
st.markdown(
    "<h4 style='color:gray;margin-top:-10px;'>"
    "Real-Time Cloud VM Monitoring & Anomaly Detection"
    "</h4><br>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------
# REFRESH SETTINGS
# ----------------------------------------------------------
refresh_rate = st.sidebar.slider("Auto Refresh Every (seconds)", 2, 10, 3)

if hasattr(st, "autorefresh"):
    st.sidebar.success("Auto-refresh enabled ✔")
    st.autorefresh(interval=refresh_rate * 1000, key="system_overview_autorefresh")
else:
    st.sidebar.warning("Auto-refresh API unavailable. Use Refresh Now.")
    if st.sidebar.button("🔄 Refresh Now"):
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            st.info("Reload the browser (Ctrl+R / Cmd+R)")

# ----------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    label_enc = joblib.load("label_encoder.pkl")
    try:
        scaler = joblib.load("scaler.pkl")
    except:
        scaler = None
    return model, label_enc, scaler

model, label_enc, scaler = load_artifacts()

# ----------------------------------------------------------
# SESSION HISTORY
# ----------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------------------------------------
# VM SETTINGS
# ----------------------------------------------------------
num_vms = st.sidebar.selectbox("Number of Virtual Machines", [3, 5, 8, 10], index=1)

# ----------------------------------------------------------
# GENERATE SAMPLE
# ----------------------------------------------------------
def generate_sample(vm_id):
    return {
        "vm_id": vm_id,
        "cpu_usage": np.random.uniform(5, 95),
        "memory_usage": np.random.uniform(10, 97),
        "network_traffic": np.random.uniform(100, 6000),
        "power_consumption": np.random.uniform(20, 250),
        "num_executed_instructions": np.random.uniform(1000, 20000),
        "execution_time": np.random.uniform(0.01, 1.5),
        "energy_efficiency": np.random.uniform(0.1, 0.95),
        "task_type": np.random.choice(["compute", "io", "network"]),
        "task_priority": np.random.choice(["low", "medium", "high"]),
        "task_status": np.random.choice(["running", "waiting", "idle", "done"]),
    }

# ----------------------------------------------------------
# PROCESS VM STREAM
# ----------------------------------------------------------
batch = []
for i in range(1, num_vms + 1):
    vm_id = f"VM-{i}"
    sample = generate_sample(vm_id)

    model_input = pd.DataFrame([{k: sample[k] for k in sample if k != "vm_id"}])

    try:
        prediction = model.predict(model_input)[0]
        label = label_enc.inverse_transform([prediction])[0]
    except:
        prediction = 0
        label = "normal"

    sample["prediction"] = prediction
    sample["prediction_label"] = "🔴 ANOMALY" if prediction == 1 else "🟢 NORMAL"
    sample["timestamp"] = time.strftime("%H:%M:%S")
    batch.append(sample)

st.session_state.history.extend(batch)
data = pd.DataFrame(st.session_state.history)

# ----------------------------------------------------------
# ANOMALY SUMMARY BADGE
# ----------------------------------------------------------
st.write("### 🚨 Anomaly Summary")

latest_preds = data.tail(num_vms)[["vm_id", "prediction"]]

anomaly_badges = ""
for _, row in latest_preds.iterrows():
    if row["prediction"] == 1:
        anomaly_badges += f"<div style='padding:6px;margin:4px;border-radius:8px;background:#ffcccc;color:#b30000;display:inline-block;'>🔴 {row['vm_id']} - Anomaly</div>"
    else:
        anomaly_badges += f"<div style='padding:6px;margin:4px;border-radius:8px;background:#ccffcc;color:#006600;display:inline-block;'>🟢 {row['vm_id']} - Normal</div>"

st.markdown(anomaly_badges, unsafe_allow_html=True)
st.write("")

# ----------------------------------------------------------
# KPI CARDS
# ----------------------------------------------------------
st.write("### 📊 Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

total_samples = len(data)
anomaly_count = int((data["prediction"] == 1).sum())
anomaly_rate = anomaly_count / total_samples * 100 if total_samples > 0 else 0
avg_cpu = data["cpu_usage"].tail(num_vms).mean()
avg_memory = data["memory_usage"].tail(num_vms).mean()

col1.metric("Total Samples", total_samples)
col2.metric("Total Anomalies", anomaly_count)
col3.metric("Anomaly Rate (%)", f"{anomaly_rate:.2f}")
col4.metric("Avg CPU (%)", f"{avg_cpu:.2f}")
col5.metric("Avg Memory (%)", f"{avg_memory:.2f}")

# ----------------------------------------------------------
# RUNTIME CLOCK DISPLAY
# ----------------------------------------------------------
st.write("### ⏱️ Runtime Clock")

elapsed_seconds = int(time.time() - st.session_state.start_time)

hours = elapsed_seconds // 3600
minutes = (elapsed_seconds % 3600) // 60
seconds = elapsed_seconds % 60

runtime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

st.markdown(
    f"""
    <div style="
        background-color:#111827;
        padding:12px 20px;
        border-radius:10px;
        border:1px solid #333;
        width:200px;
        text-align:center;
        font-size:22px;
        color:#10B981;
        font-weight:bold;
        margin-bottom:15px;">
        🟢 Uptime: {runtime_str}
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------
# ANOMALY SCATTER PLOT
# ----------------------------------------------------------
st.write("### 🔥 System-Wide Anomaly Events")

if data.empty:
    st.info("No data available yet.")
else:
    fig = px.scatter(
        data,
        x=data.index,
        y="cpu_usage",
        color="prediction",
        color_discrete_map={0: "green", 1: "red"},
        hover_data=["vm_id", "timestamp", "cpu_usage"],
        title="CPU Usage with Normal (Green) & Anomaly (Red) Indicators",
        labels={"prediction": "Anomaly"},
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# LATEST VM TABLE WITH BADGES
# ----------------------------------------------------------
st.write("### 📝 Latest VM Metrics")

if not data.empty:
    view = data.tail(num_vms).copy()
    view["Status"] = view["prediction"].apply(lambda x: "🔴 Anomaly" if x == 1 else "🟢 Normal")
    st.dataframe(view, use_container_width=True)

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------
st.write("---")
st.write(f"Last Update: **{time.strftime('%Y-%m-%d %H:%M:%S')}**")
